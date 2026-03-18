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
from agent.memory.store import MemoryStore
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
    messages = list(history) if history else []
    messages.append(HumanMessage(content=input_text))
    messages = summarize_and_window(messages)

    reset_runtime_steps()
    tracker = CostTracker()
    budget = BudgetTracker()

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
        return {
            "answer": partial_answer,
            "steps": _extract_steps(initial_state, initial_state["cost_tracker"], initial_state["budget_tracker"]),
            "updated_history": initial_state["messages"],
            "final_state": initial_state,
        }

    tracker = initial_state["cost_tracker"]
    budget = initial_state["budget_tracker"]
    steps = _extract_steps(final_state, tracker, budget)
    updated_history = [
        msg
        for msg in final_state.get("messages", [])
        if isinstance(msg, HumanMessage) or (isinstance(msg, AIMessage) and msg.content and not msg.tool_calls)
    ]

    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return {
                "answer": _extract_final_answer(str(msg.content)),
                "steps": steps,
                "updated_history": updated_history,
                "final_state": final_state,
            }

    return {
        "answer": "I was unable to generate a response.",
        "steps": steps,
        "updated_history": updated_history,
        "final_state": final_state,
    }
