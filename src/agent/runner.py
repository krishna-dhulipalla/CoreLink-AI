"""
Agent Runner: Convenience function to execute the compiled graph
=================================================================
Used by executor.py to bridge A2A requests into LangGraph runs.
"""

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.state import AgentState
from agent.cost import CostTracker
from agent.prompts import MODEL_NAME
from agent.nodes.reasoner import reset_step_counter
from agent.nodes.reflector import _is_reflection_message
from context_manager import summarize_and_window

logger = logging.getLogger(__name__)


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
    tracker = CostTracker(model_name=MODEL_NAME)

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
        "cost_tracker": tracker,
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

    logger.info(f"[CostTracker] {cost_summary}")

    def _is_internal_node_message(m: BaseMessage) -> bool:
        return _is_reflection_message(m) or getattr(m, "additional_kwargs", {}).get("is_warning", False)

    # Strip reflection and warning messages from persisted history
    updated_history = [
        msg for msg in all_messages if not _is_internal_node_message(msg)
    ]

    # Extract final answer: last AIMessage that is NOT internal
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            if not _is_internal_node_message(msg):
                return msg.content, steps, updated_history

    return "I was unable to generate a response.", steps, updated_history
