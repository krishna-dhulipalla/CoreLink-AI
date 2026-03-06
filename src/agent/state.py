"""
Agent State: TypedDict & Custom Reducer
========================================
Defines the shared state schema used by all LangGraph nodes.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class ReplaceMessages(list):
    """Sentinel wrapper: tells the reducer to *replace* instead of append."""
    pass


def _messages_reducer(
    existing: list[BaseMessage], update: list[BaseMessage] | ReplaceMessages
) -> list[BaseMessage]:
    """Custom reducer that supports both append and full replace.

    - Normal node output (plain list) → appended to existing.
    - ReplaceMessages wrapper → existing is replaced entirely.
    """
    if isinstance(update, ReplaceMessages):
        return list(update)
    return existing + update


class AgentState(TypedDict):
    """Typed state for the LangGraph reasoning engine.

    - messages: The conversation history (LangChain message format).
                Uses a custom reducer that supports both append and replace.
    - reflection_count: Number of reflection-revision cycles completed.
    - tool_fail_count: Consecutive tool failures. Triggers forced fallback at threshold.
    - last_tool_signature: Hash of (tool_name + args) to detect duplicate calls.
    - route: Coordinator's decision: "direct" or "heavy_research".
    """
    messages: Annotated[list[BaseMessage], _messages_reducer]
    reflection_count: int
    tool_fail_count: int
    last_tool_signature: str
    route: str
