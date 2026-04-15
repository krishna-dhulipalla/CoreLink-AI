"""Shared engine state and message reducer."""

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
    """Typed state for the active engine graph."""
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
    risk_feedback: dict[str, Any] | None
    compliance_feedback: dict[str, Any] | None
    review_feedback: dict[str, Any] | None
    reflection_feedback: dict[str, Any] | None
    checkpoint_stack: list[dict]
    tool_fail_count: int
    last_tool_signature: str
    budget_tracker: Any
    cost_tracker: Any
    memory_store: Any
    task_intent: dict[str, Any]
    benchmark_overrides: dict[str, Any]
    tool_plan: dict[str, Any]
    source_bundle: dict[str, Any]
    retrieval_intent: dict[str, Any]
    curated_context: dict[str, Any]
    review_packet: dict[str, Any]
    evidence_sufficiency: dict[str, Any]
    execution_journal: dict[str, Any]
    quality_report: dict[str, Any]
    progress_signature: dict[str, Any]
    unsupported_capability_report: dict[str, Any]
    trace_identity: dict[str, Any]
    fast_path_used: bool


RuntimeState = AgentState
