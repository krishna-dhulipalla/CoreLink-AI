"""
Agent Runner: Convenience function to execute the compiled graph
=================================================================
Used by executor.py to bridge A2A requests into LangGraph runs.
"""

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.state import AgentState
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
        steps: A list of dicts describing each node that executed.
        updated_history: The cleaned message list after execution, for persistence.
    """
    messages = list(history) if history else []
    messages.append(HumanMessage(content=input_text))

    # Reset step counter for this run
    reset_step_counter()

    # Front-gate pruning: apply context windowing BEFORE graph entry
    messages = summarize_and_window(messages)

    initial_state = {
        "messages": messages,
        "reflection_count": 0,
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "route": "",
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
        return (
            partial_answer or "I was unable to complete this task within the step limit.",
            [{"node": "safety", "action": "Recursion limit reached"}],
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

    # Strip reflection messages from persisted history
    updated_history = [
        msg for msg in all_messages
            if not _is_reflection_message(msg)
        ]

    # Extract final answer: last AIMessage that is NOT a reflection
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            if not _is_reflection_message(msg):
                return msg.content, steps, updated_history

    return "I was unable to generate a response.", steps, updated_history
