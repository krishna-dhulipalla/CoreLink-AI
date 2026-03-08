"""
State Pruning (Sprint 4A)
==========================
Functions to strip low-signal messages before they reach nodes or storage.

Design:
- prune_for_reasoner: strips internal warnings, stale old tool-call bundles
  (AIMessage+ToolMessage pairs), and memory-hint SystemMessages.
- prune_for_persistence: strips warnings and memory hints from persisted history.
- truncate_memory_fields: caps long string fields before memory writes.

Inspired by AgentPrune spatial-temporal message pruning.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Keep at most this many tool-call bundles in reasoner context
MAX_TOOL_RESULTS_IN_CONTEXT = 6

# Max characters for memory record string fields
MAX_MEMORY_FIELD_LEN = 120

# Marker used to identify memory-hint SystemMessages
_HINT_MARKERS = (
    "TOOL-SELECTION MEMORY",
    "PAST ROUTING MEMORY",
    "PAST REPAIR MEMORY",
)


def _is_memory_hint(msg: BaseMessage) -> bool:
    """Check if a message is an internally-injected memory hint."""
    if not isinstance(msg, SystemMessage):
        return False
    content = str(msg.content)
    return any(content.startswith(m) for m in _HINT_MARKERS)


def _is_internal_warning(msg: BaseMessage) -> bool:
    """Check if a message is a verifier warning (REVISE/BACKTRACK)."""
    return bool(getattr(msg, "additional_kwargs", {}).get("is_warning", False))


# ---------------------------------------------------------------------------
# Pruning Functions
# ---------------------------------------------------------------------------

def prune_for_reasoner(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Strip low-signal messages before the Reasoner LLM call.

    Removes:
    1. Internal verifier warnings (is_warning=True).
    2. Old memory-hint SystemMessages (fresh ones are re-injected each call).
    3. Stale tool-call bundles (AIMessage with tool_calls + their paired
       ToolMessages) beyond the most recent MAX_TOOL_RESULTS_IN_CONTEXT,
       keeping the pairs intact to avoid orphaning tool_calls.
    """
    # Phase 1: remove warnings and old hints
    cleaned = [
        m for m in messages
        if not _is_memory_hint(m) and not _is_internal_warning(m)
    ]

    # Phase 2: identify tool-call bundles and prune old ones as complete pairs.
    # A "bundle" is an AIMessage with tool_calls followed by its ToolMessages.
    # We find all bundles, keep the most recent MAX_TOOL_RESULTS_IN_CONTEXT,
    # and remove the rest (both the AIMessage and its ToolMessages).
    bundle_starts: list[int] = []
    for i, m in enumerate(cleaned):
        if isinstance(m, AIMessage) and m.tool_calls:
            bundle_starts.append(i)

    if len(bundle_starts) > MAX_TOOL_RESULTS_IN_CONTEXT:
        stale_starts = bundle_starts[:-MAX_TOOL_RESULTS_IN_CONTEXT]
        # Collect the tool_call_ids from stale AIMessages
        stale_call_ids: set[str] = set()
        stale_indices: set[int] = set()
        for idx in stale_starts:
            stale_indices.add(idx)
            ai_msg = cleaned[idx]
            for tc in ai_msg.tool_calls:
                stale_call_ids.add(tc.get("id", ""))

        # Also mark the paired ToolMessages for removal
        for i, m in enumerate(cleaned):
            if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", "") in stale_call_ids:
                stale_indices.add(i)

        pruned_count = len(stale_indices)
        cleaned = [m for i, m in enumerate(cleaned) if i not in stale_indices]
        logger.info(
            f"[Prune] Stripped {pruned_count} messages "
            f"({len(stale_starts)} stale tool-call bundles) from reasoner context."
        )

    return cleaned


def prune_for_persistence(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Strip internal artifacts before persisting conversation history.

    Removes:
    1. Internal verifier warnings (is_warning=True).
    2. Memory hint SystemMessages.
    """
    result = []
    for m in messages:
        if _is_internal_warning(m):
            continue
        if _is_memory_hint(m):
            continue
        result.append(m)
    return result


def truncate_memory_fields(record: Any, max_len: int = MAX_MEMORY_FIELD_LEN) -> Any:
    """Truncate long string fields on a Pydantic memory record in-place.

    Caps: arguments_pattern, failure_pattern, partial_context_summary,
    repair_action, task_summary.
    """
    for field_name in (
        "arguments_pattern",
        "failure_pattern",
        "partial_context_summary",
        "repair_action",
        "task_summary",
    ):
        value = getattr(record, field_name, None)
        if isinstance(value, str) and len(value) > max_len:
            object.__setattr__(record, field_name, value[:max_len])
    return record
