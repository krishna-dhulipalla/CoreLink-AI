"""
Agent State: TypedDict & Custom Reducer
========================================
Defines the shared state schema used by all LangGraph nodes.
"""

from typing import Annotated, Any, TypedDict

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

    Core fields:
        messages: Conversation history with custom append/replace reducer.
        reflection_count: Number of self-critique cycles completed.
        tool_fail_count: Consecutive tool failures (triggers forced fallback).
        last_tool_signature: Hash of last tool call for dedup detection.

    MaAS-lite fields (Sprint 1.5):
        selected_layers: Operator names chosen by the coordinator.
        format_required: Whether the format_normalizer should fire.
        architecture_trace: Serialized OperatorTrace entries for cost tracking.
        cost_tracker: Live CostTracker instance (not persisted).
    """
    messages: Annotated[list[BaseMessage], _messages_reducer]
    reflection_count: int
    tool_fail_count: int
    last_tool_signature: str
    # Sprint 1.5: MaAS-lite policy & cost fields
    selected_layers: list[str]
    format_required: bool
    architecture_trace: list[dict]
    cost_tracker: Any  # agent.cost.CostTracker (Any avoids circular import)
