"""Agent runner for the active engine graph."""

from __future__ import annotations

import logging
import os
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.budget import BudgetTracker
from agent.cost import CostTracker
from agent.memory.store import MemoryStore
from agent.runtime_clock import reset_runtime_steps
from agent.state import AgentState
from agent.tracer import finalize_tracer, start_tracer
from context_manager import summarize_and_window

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_memory_store: MemoryStore | None = None


def _same_message(a: BaseMessage, b: BaseMessage) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, HumanMessage):
        return str(a.content or "").strip() == str(b.content or "").strip()
    if isinstance(a, AIMessage):
        return (
            str(a.content or "").strip() == str(b.content or "").strip()
            and bool(a.tool_calls) == bool(b.tool_calls)
        )
    if isinstance(a, ToolMessage):
        return (
            str(a.content or "").strip() == str(b.content or "").strip()
            and str(getattr(a, "name", "")) == str(getattr(b, "name", ""))
        )
    return False


def _dedupe_adjacent_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    deduped: list[BaseMessage] = []
    for msg in messages:
        if deduped and _same_message(deduped[-1], msg):
            continue
        deduped.append(msg)
    return deduped


def _benchmark_stateless_mode() -> bool:
    return os.getenv("BENCHMARK_STATELESS", "").strip().lower() in {"1", "true", "yes", "on"}


def _get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def _strip_think_markup(text: str) -> str:
    clean = _THINK_BLOCK_RE.sub("", text)
    clean = clean.replace("<think>", "").replace("</think>", "")
    return clean.strip()


def _extract_final_answer(text: str) -> str:
    clean = _strip_think_markup(text)
    return clean or text.strip()


def _extract_steps(final_state: AgentState, tracker: CostTracker, budget: BudgetTracker) -> list[dict]:
    steps = list(final_state.get("workpad", {}).get("events", []))
    for msg in final_state.get("messages", []):
        if isinstance(msg, ToolMessage):
            steps.append({"node": "executor", "action": f"Tool result: {msg.name}"})
    steps.append({"node": "cost_summary", **tracker.summary()})
    steps.append({"node": "budget_summary", **budget.summary()})
    return steps


def _build_updated_history(final_state: AgentState, final_answer: str | None = None) -> list[BaseMessage]:
    updated_history = [msg for msg in final_state.get("messages", []) if isinstance(msg, HumanMessage)]
    if final_answer:
        updated_history.append(AIMessage(content=_extract_final_answer(final_answer)))
    return _dedupe_adjacent_messages(updated_history)


async def run_agent(
    graph,
    input_text: str,
    history: list[BaseMessage] | None = None,
) -> tuple[str, list[dict], list[BaseMessage]]:
    trace = await run_agent_trace(graph, input_text, history=history)
    return trace["answer"], trace["steps"], trace["updated_history"]


async def run_agent_trace(
    graph,
    input_text: str,
    history: list[BaseMessage] | None = None,
) -> dict:
    if _benchmark_stateless_mode():
        history = []

    messages = list(history) if history else []
    incoming_human = HumanMessage(content=input_text)
    if not (messages and isinstance(messages[-1], HumanMessage) and str(messages[-1].content or "").strip() == input_text.strip()):
        messages.append(incoming_human)
    messages = _dedupe_adjacent_messages(messages)
    messages = summarize_and_window(messages)
    messages = _dedupe_adjacent_messages(messages)

    reset_runtime_steps()
    tracker = CostTracker()
    budget = BudgetTracker()

    # ── Start RunTracer if enabled ──
    tracer = start_tracer()
    if tracer:
        tracer.set_task(input_text)

    initial_state: AgentState = {
        "messages": messages,
        "profile_decision": {},
        "task_profile": "general",
        "capability_flags": [],
        "ambiguity_flags": [],
        "execution_template": {},
        "answer_contract": {},
        "evidence_pack": {},
        "assumption_ledger": [],
        "provenance_map": {},
        "solver_stage": "PLAN",
        "workpad": {"events": [], "stage_outputs": {}, "tool_results": []},
        "pending_tool_call": None,
        "last_tool_result": None,
        "risk_feedback": None,
        "compliance_feedback": None,
        "review_feedback": None,
        "reflection_feedback": None,
        "checkpoint_stack": [],
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "budget_tracker": budget,
        "cost_tracker": tracker,
        "memory_store": None,
        "task_intent": {},
        "tool_plan": {},
        "source_bundle": {},
        "curated_context": {},
        "review_packet": {},
        "execution_journal": {
            "events": [],
            "tool_results": [],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 0,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        "quality_report": {},
        "progress_signature": {},
        "unsupported_capability_report": {},
        "fast_path_used": False,
    }

    last_state = initial_state
    try:
        async for streamed_state in graph.astream(initial_state, config={"recursion_limit": 80}, stream_mode="values"):
            if isinstance(streamed_state, dict):
                last_state = streamed_state
        final_state = last_state
    except GraphRecursionError as exc:
        logger.warning("[Safety] Recursion limit hit: %s", exc)
        final_state = last_state if isinstance(last_state, dict) else initial_state
        final_state = dict(final_state)
        final_state["messages"] = _dedupe_adjacent_messages(list(final_state.get("messages", [])))
        final_state["memory_store"] = None
        workpad = dict(final_state.get("workpad", {}))
        events = list(workpad.get("events", []))
        events.append({"node": "runner", "action": "recursion limit hit"})
        workpad["events"] = events
        final_state["workpad"] = workpad
        budget = final_state.get("budget_tracker", budget)
        budget.log_budget_exit("recursion_limit", str(exc))

        partial_answer = ""
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                partial_answer = _extract_final_answer(str(msg.content))
                break
        if not partial_answer:
            partial_answer = "I was unable to complete this task within the step limit."
        finalize_tracer(partial_answer, tracker.summary(), budget.summary())
        return {
            "answer": partial_answer,
            "steps": _extract_steps(final_state, final_state.get("cost_tracker", tracker), budget),
            "updated_history": _build_updated_history(final_state, partial_answer),
            "final_state": final_state,
        }

    final_state = dict(final_state)
    final_state["messages"] = _dedupe_adjacent_messages(list(final_state.get("messages", [])))
    final_state["memory_store"] = None
    tracker = final_state.get("cost_tracker", initial_state["cost_tracker"])
    budget = final_state.get("budget_tracker", initial_state["budget_tracker"])
    steps = _extract_steps(final_state, tracker, budget)
    final_public_ai: AIMessage | None = None
    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_public_ai = msg
            break
    updated_history = _build_updated_history(
        final_state,
        _extract_final_answer(str(final_public_ai.content)) if final_public_ai is not None else None,
    )

    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            answer = _extract_final_answer(str(msg.content))
            finalize_tracer(answer, tracker.summary(), budget.summary())
            return {
                "answer": answer,
                "steps": steps,
                "updated_history": updated_history,
                "final_state": final_state,
            }

    finalize_tracer("I was unable to generate a response.", tracker.summary(), budget.summary())
    return {
        "answer": "I was unable to generate a response.",
        "steps": steps,
        "updated_history": updated_history,
        "final_state": final_state,
    }
