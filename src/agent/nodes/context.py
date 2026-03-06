"""
Context Window Node: Summarize-and-Forget
==========================================
Applies context windowing to keep message history within token budget.
"""

import logging

from agent.state import AgentState, ReplaceMessages
from agent.nodes.reasoner import _increment_step
from context_manager import summarize_and_window

logger = logging.getLogger(__name__)


def context_window(state: AgentState) -> dict:
    """Graph node: apply Summarize-and-Forget windowing to message history."""
    step = _increment_step()
    messages = state["messages"]
    compressed = summarize_and_window(messages)

    if len(compressed) < len(messages):
        logger.info(f"[Step {step}] context_window → compressed {len(messages)} → {len(compressed)} msgs")
        return {"messages": ReplaceMessages(compressed)}

    logger.info(f"[Step {step}] context_window → no compression needed")
    return {"messages": []}
