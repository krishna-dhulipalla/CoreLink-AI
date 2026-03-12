"""
Agent Runner: Convenience function to execute the compiled graph
=================================================================
Used by executor.py to bridge A2A requests into LangGraph runs.
"""

import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.state import AgentState
from agent.cost import CostTracker
from agent.model_config import primary_runtime_model
from agent.nodes.reasoner import reset_step_counter
from agent.nodes.reflector import _is_reflection_message
from agent.memory.store import MemoryStore
from agent.memory.schema import (
    RouterMemory,
    _normalize_memory_text,
    _task_signature,
    _task_type_to_family,
)
from agent.budget import BudgetTracker
from agent.pruning import prune_for_persistence, truncate_memory_fields
from context_manager import summarize_and_window

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer post-processing (Qwen3 <think> blocks + orphan tool-call fallback)
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(
    r'^\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', re.DOTALL
)


def _extract_final_answer(text: str) -> str:
    """Extract the actual answer from model output.

    Handles two common issues with Qwen3-style models:
    1. <think>...</think> reasoning blocks that pollute the output.
    2. Orphan tool-call JSON when the model tried to call a tool but
       the graph exited before executing it.

    In case (2), we fall back to the reasoning inside the <think> block.
    """
    # Extract think block content before stripping (used as fallback)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""

    # Strip <think>...</think> blocks
    clean = _THINK_BLOCK_RE.sub("", text).strip()

    # If what remains is an orphan tool-call JSON, use think content instead
    if clean and _TOOL_CALL_JSON_RE.match(clean):
        logger.info("[AnswerExtract] Detected orphan tool-call JSON; using <think> content as answer.")
        return think_content if think_content else clean

    # If stripping left nothing, return original text
    if not clean:
        return think_content if think_content else text

    return clean


# Module-level singleton so the store persists across runs
_memory_store: MemoryStore | None = None


def _get_memory_store() -> MemoryStore:
    """Lazy-init a single MemoryStore for agent lifetime."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


async def run_agent(
    graph,
    input_text: str,
    history: list[BaseMessage] | None = None,
) -> tuple[str, list[dict], list[BaseMessage]]:
    """Run the compiled graph and return (final_answer, steps, updated_history).

    Args:
        graph: The compiled LangGraph.
        input_text: The new user message.
        history: Optional prior conversation messages for multi-turn support.

    Returns:
        final_answer: The text content of the last AIMessage (excluding reflections).
        steps: A list of dicts describing each node that executed, plus a cost_summary.
        updated_history: The cleaned message list after execution, for persistence.
    """
    messages = list(history) if history else []
    messages.append(HumanMessage(content=input_text))

    # Reset step counter for this run
    reset_step_counter()

    # Initialize cost tracker
    tracker = CostTracker(model_name=primary_runtime_model())

    # Sprint 4: Initialize budget tracker
    budget = BudgetTracker()

    # Front-gate pruning: apply context windowing BEFORE graph entry
    messages = summarize_and_window(messages)

    initial_state = {
        "messages": messages,
        "reflection_count": 0,
        "tool_fail_count": 0,
        "last_tool_signature": "",
        # Sprint 1.5: MaAS-lite fields
        "selected_layers": [],
        "format_required": False,
        "policy_confidence": 0.0,
        "estimated_steps": 0,
        "early_exit_allowed": False,
        "architecture_trace": [],
        "checkpoint_stack": [],
        "task_type": "general",
        "cost_tracker": tracker,
        # Sprint 3: Execution Memory
        "memory_store": _get_memory_store(),
        # Sprint 4: Budget Control
        "budget_tracker": budget,
    }

    try:
        final_state = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": 25},
        )
    except GraphRecursionError as e:
        logger.warning(f"[Safety] Recursion limit hit: {e}")
        partial_answer = None
        for msg in reversed(initial_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not _is_reflection_message(msg) and not msg.tool_calls:
                partial_answer = msg.content
                break

        # Log cost even on recursion failure
        cost_summary = tracker.summary()
        cost_summary["architecture_trace"] = tracker.architecture_trace()
        logger.info(f"[CostTracker] (recursion limit) {cost_summary}")

        return (
            partial_answer or "I was unable to complete this task within the step limit.",
            [
                {"node": "safety", "action": "Recursion limit reached"},
                {"node": "cost_summary", **cost_summary},
            ],
            initial_state["messages"],
        )

    # Build step list from the final state
    steps = []
    all_messages = final_state.get("messages", [])
    for msg in all_messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_names = [tc["name"] for tc in msg.tool_calls]
                steps.append({
                    "node": "reasoner",
                    "action": f"Calling tools: {', '.join(tool_names)}",
                })
            elif _is_reflection_message(msg):
                steps.append({
                    "node": "reflector",
                    "action": f"Self-review: {msg.content[:80]}",
                })
            else:
                steps.append({
                    "node": "reasoner",
                    "action": "Generating draft answer",
                })
        elif isinstance(msg, ToolMessage):
            steps.append({
                "node": "tool_executor",
                "action": f"Tool result: {msg.name}",
            })

    # Log cost summary
    cost_summary = tracker.summary()
    cost_summary["selected_layers"] = final_state.get("selected_layers", [])
    cost_summary["format_required"] = final_state.get("format_required", False)
    cost_summary["policy_confidence"] = final_state.get("policy_confidence", 0.0)
    cost_summary["estimated_steps"] = final_state.get("estimated_steps", 0)
    cost_summary["early_exit_allowed"] = final_state.get("early_exit_allowed", False)
    cost_summary["architecture_trace"] = tracker.architecture_trace()
    steps.append({"node": "cost_summary", **cost_summary})

    # Sprint 4: Log budget summary
    budget_summary = budget.summary()
    steps.append({"node": "budget_summary", **budget_summary})
    logger.info(f"[CostTracker] {cost_summary}")
    logger.info(f"[BudgetTracker] {budget_summary}")

    # Sprint 3: Store RouterMemory post-run
    try:
        mem_store = _get_memory_store()
        task_summary = input_text[:120] if input_text else ""
        # Sprint 4 Fix: derive success from whether a verifier PASS was achieved
        # (pending_verifier_feedback is None when PASS or direct_answer)
        run_success = final_state.get("pending_verifier_feedback") is None
        router_rec = RouterMemory(
            task_signature=_task_signature(input_text),
            task_summary=task_summary,
            semantic_text=_normalize_memory_text(
                f"task: {input_text}\n"
                f"layers: {' '.join(final_state.get('selected_layers', []))}\n"
                f"success: {run_success}"
            ),
            task_family=_task_type_to_family(final_state.get("task_type", "general"), input_text),
            selected_layers=final_state.get("selected_layers", []),
            success=run_success,
            cost_usd=tracker.total_cost(),
            latency_ms=tracker.wall_clock_ms,
            tags=list(final_state.get("selected_layers", [])),
            metadata={
                "task_type": final_state.get("task_type", "general"),
                "policy_confidence": final_state.get("policy_confidence", 0.0),
                "estimated_steps": final_state.get("estimated_steps", 0),
                "early_exit_allowed": final_state.get("early_exit_allowed", False),
                "format_required": final_state.get("format_required", False),
            },
        )
        truncate_memory_fields(router_rec)
        mem_store.store_router(router_rec)
    except Exception as mem_err:
        logger.warning(f"[Memory] Failed to store router memory: {mem_err}")

    def _is_internal_node_message(m: BaseMessage) -> bool:
        return _is_reflection_message(m) or getattr(m, "additional_kwargs", {}).get("is_warning", False)

    # Sprint 4: Use prune_for_persistence instead of simple filter
    updated_history = prune_for_persistence([
        msg for msg in all_messages if not _is_internal_node_message(msg)
    ])

    # Extract final answer: last AIMessage that is NOT internal
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            if not _is_internal_node_message(msg):
                return _extract_final_answer(msg.content), steps, updated_history

    return "I was unable to generate a response.", steps, updated_history
