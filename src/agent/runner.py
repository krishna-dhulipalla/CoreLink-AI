"""
Agent Runner
============
Executes the staged finance-first graph and returns the final answer plus trace.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.budget import BudgetTracker
from agent.cost import CostTracker
from agent.memory.schema import RouterMemory, _infer_task_family, _normalize_memory_text, _task_signature
from agent.memory.store import MemoryStore
from agent.pruning import truncate_memory_fields
from agent.runtime_clock import reset_runtime_steps
from agent.state import AgentState
from context_manager import summarize_and_window

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_memory_store: MemoryStore | None = None


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
            steps.append({"node": "tool_runner", "action": f"Tool result: {msg.name}"})
    steps.append({"node": "cost_summary", **tracker.summary()})
    steps.append({"node": "budget_summary", **budget.summary()})
    return steps


def _persist_memory(input_text: str, final_state: AgentState, tracker: CostTracker) -> None:
    try:
        mem_store = _get_memory_store()
        workpad = final_state.get("workpad", {})
        task_profile = final_state.get("task_profile", "general")
        capability_flags = list(final_state.get("capability_flags", []))
        success = final_state.get("solver_stage") == "COMPLETE"
        route_path = list(
            dict.fromkeys(
                event.get("node", "")
                for event in workpad.get("events", [])
                if event.get("node")
            )
        )
        rec = RouterMemory(
            task_signature=_task_signature(input_text),
            task_summary=input_text[:120],
            semantic_text=_normalize_memory_text(
                f"task: {input_text}\nprofile: {task_profile}\nflags: {' '.join(capability_flags)}\nsuccess: {success}"
            ),
            task_family=_infer_task_family(input_text),
            selected_layers=route_path or ["staged_runtime"],
            success=success,
            cost_usd=tracker.total_cost(),
            latency_ms=tracker.wall_clock_ms,
            tags=[task_profile, *capability_flags[:4]],
            metadata={
                "task_profile": task_profile,
                "capability_flags": capability_flags,
                "requires_adapter": bool(final_state.get("answer_contract", {}).get("requires_adapter")),
            },
        )
        truncate_memory_fields(rec)
        mem_store.store_router(rec)
    except Exception as exc:
        logger.warning("[Memory] Failed to store router memory: %s", exc)


async def run_agent(
    graph,
    input_text: str,
    history: list[BaseMessage] | None = None,
) -> tuple[str, list[dict], list[BaseMessage]]:
    messages = list(history) if history else []
    messages.append(HumanMessage(content=input_text))
    messages = summarize_and_window(messages)

    reset_runtime_steps()
    tracker = CostTracker()
    budget = BudgetTracker()

    initial_state: AgentState = {
        "messages": messages,
        "task_profile": "general",
        "capability_flags": [],
        "answer_contract": {},
        "evidence_pack": {},
        "solver_stage": "PLAN",
        "workpad": {"events": [], "stage_outputs": {}, "tool_results": []},
        "pending_tool_call": None,
        "last_tool_result": None,
        "review_feedback": None,
        "checkpoint_stack": [],
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "budget_tracker": budget,
        "cost_tracker": tracker,
        "memory_store": _get_memory_store(),
    }

    try:
        final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 40})
    except GraphRecursionError as exc:
        logger.warning("[Safety] Recursion limit hit: %s", exc)
        partial_answer = "I was unable to complete this task within the step limit."
        return partial_answer, _extract_steps(initial_state, tracker, budget), messages

    steps = _extract_steps(final_state, tracker, budget)
    _persist_memory(input_text, final_state, tracker)

    updated_history = [
        msg
        for msg in final_state.get("messages", [])
        if isinstance(msg, HumanMessage) or (isinstance(msg, AIMessage) and msg.content and not msg.tool_calls)
    ]

    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return _extract_final_answer(str(msg.content)), steps, updated_history

    return "I was unable to generate a response.", steps, updated_history
