"""
Agent State: TypedDict & Custom Reducer
======================================
Defines the shared runtime state for the staged finance-first graph.
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
    """Typed state for the staged runtime.

    The runtime now moves explicit artifacts between nodes instead of
    passing coordinator telemetry through the whole graph.
    """
    messages: Annotated[list[BaseMessage], _messages_reducer]
    profile_decision: dict[str, Any]
    task_profile: str
    capability_flags: list[str]
    ambiguity_flags: list[str]
    execution_template: dict[str, Any]
    answer_contract: dict[str, Any]
    evidence_pack: dict[str, Any]
    assumption_ledger: list[dict[str, Any]]
    provenance_map: dict[str, dict[str, Any]]
    solver_stage: str
    workpad: dict[str, Any]
    pending_tool_call: dict[str, Any] | None
    last_tool_result: dict[str, Any] | None
    review_feedback: dict[str, Any] | None
    checkpoint_stack: list[dict]
    tool_fail_count: int
    last_tool_signature: str
    budget_tracker: Any
    cost_tracker: Any
    memory_store: Any
