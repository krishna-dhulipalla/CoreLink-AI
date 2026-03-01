"""
Context Manager: Observation Masking & Message Windowing
=========================================================
Implements the "Summarize-and-Forget" and "Observation Masking" strategies
from DESIGN.md to keep the LangGraph agent within its context window budget.

Two mechanisms:
1. Tool Output Truncation — caps verbose tool responses before they enter history.
2. Message Windowing — compresses older messages into a summary when token
   count exceeds the budget.

Configuration via .env:
    MAX_CONTEXT_TOKENS=80000
    MAX_TOOL_OUTPUT_CHARS=4000
    CONTEXT_KEEP_RECENT=6
"""

import os
import logging
from typing import Sequence

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "80000"))
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000"))
CONTEXT_KEEP_RECENT = int(os.getenv("CONTEXT_KEEP_RECENT", "6"))
MODEL_ENCODING = os.getenv("TIKTOKEN_MODEL", "gpt-4o-mini")


# ── Token Counting ────────────────────────────────────────────────────────

def _get_encoding():
    """Get the tiktoken encoding for the configured model."""
    try:
        return tiktoken.encoding_for_model(MODEL_ENCODING)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: Sequence[BaseMessage]) -> int:
    """Estimate total token count for a list of LangChain messages.

    Uses tiktoken for fast client-side counting. Adds overhead per message
    for role tokens and separators (~4 tokens per message).
    """
    enc = _get_encoding()
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += len(enc.encode(content)) + 4  # +4 for role + separators
    return total


# ── Tool Output Truncation ────────────────────────────────────────────────

def truncate_tool_output(content: str, max_chars: int | None = None) -> str:
    """Truncate long tool output with a marker.

    Args:
        content: The raw tool output string.
        max_chars: Maximum characters to keep. Defaults to MAX_TOOL_OUTPUT_CHARS.

    Returns:
        The original content if within limits, otherwise truncated with a
        [TRUNCATED] marker showing how much was cut.
    """
    limit = max_chars or MAX_TOOL_OUTPUT_CHARS
    if len(content) <= limit:
        return content

    truncated_chars = len(content) - limit
    return (
        content[:limit]
        + f"\n\n[TRUNCATED: {truncated_chars} characters removed. "
        f"Original length: {len(content)} chars]"
    )


# ── Message Windowing (Summarize-and-Forget) ──────────────────────────────

def _format_message_for_summary(msg: BaseMessage) -> str:
    """Format a single message into a concise summary line."""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    # Truncate individual messages in the summary to keep it tight
    if len(content) > 200:
        content = content[:200] + "..."

    if isinstance(msg, HumanMessage):
        return f"User: {content}"
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            tool_names = [tc["name"] for tc in msg.tool_calls]
            return f"Agent called tools: {', '.join(tool_names)}"
        return f"Agent: {content}"
    elif isinstance(msg, ToolMessage):
        return f"Tool ({msg.name}): {content}"
    elif isinstance(msg, SystemMessage):
        return f"System: {content}"
    else:
        return f"{msg.__class__.__name__}: {content}"


def _adjust_boundary_for_tool_bundle(
    messages: Sequence[BaseMessage],
    boundary: int,
    start_idx: int,
) -> int:
    """Move the compression boundary left if it would split a tool-call bundle.

    OpenAI chat-completions requires an assistant message with ``tool_calls`` to
    be followed by the corresponding ``ToolMessage`` entries. If the windowing
    boundary lands on any ``ToolMessage``, move it left to the originating
    ``AIMessage`` so the whole bundle stays in the recent working set.
    """
    if boundary <= start_idx or boundary >= len(messages):
        return boundary

    if not isinstance(messages[boundary], ToolMessage):
        return boundary

    cursor = boundary - 1
    while cursor >= start_idx and isinstance(messages[cursor], ToolMessage):
        cursor -= 1

    if (
        cursor >= start_idx
        and isinstance(messages[cursor], AIMessage)
        and messages[cursor].tool_calls
    ):
        logger.info(
            "Adjusted context window boundary: %d -> %d to preserve tool-call bundle",
            boundary,
            cursor,
        )
        return cursor

    return boundary


def summarize_and_window(
    messages: list[BaseMessage],
    max_tokens: int | None = None,
    keep_recent: int | None = None,
) -> list[BaseMessage]:
    """Apply the Summarize-and-Forget windowing strategy.

    If the total token count of `messages` exceeds `max_tokens`:
    1. Keep the SystemMessage at index 0 (if present).
    2. Keep the last `keep_recent` messages (the active working set).
    3. Summarize everything in between into a single SystemMessage.
    4. Return the compressed list.

    If under budget, returns the original list unchanged.

    Args:
        messages: The full conversation history.
        max_tokens: Token budget. Defaults to MAX_CONTEXT_TOKENS.
        keep_recent: Number of recent messages to always preserve.
                     Defaults to CONTEXT_KEEP_RECENT.

    Returns:
        The (possibly compressed) message list.
    """
    budget = max_tokens or MAX_CONTEXT_TOKENS
    recent_count = keep_recent or CONTEXT_KEEP_RECENT

    current_tokens = count_tokens(messages)
    if current_tokens <= budget:
        return messages

    logger.info(
        f"Context window exceeded: {current_tokens} tokens > {budget} budget. "
        f"Compressing {len(messages)} messages..."
    )

    # Separate system prompt, middle, and recent
    has_system = isinstance(messages[0], SystemMessage) if messages else False
    system_msg = messages[0] if has_system else None
    start_idx = 1 if has_system else 0

    # Ensure we don't try to keep more recent messages than we have
    available = len(messages) - start_idx
    actual_recent = min(recent_count, available)

    if actual_recent >= available:
        # Not enough messages to summarize — return as-is
        logger.warning(
            "Not enough messages to summarize (only %d non-system messages). "
            "Skipping compression.",
            available,
        )
        return messages

    boundary = len(messages) - actual_recent
    boundary = _adjust_boundary_for_tool_bundle(messages, boundary, start_idx)

    if boundary <= start_idx:
        logger.warning(
            "Compression boundary would split the active tool-call bundle. "
            "Skipping compression for now."
        )
        return messages

    middle_msgs = messages[start_idx:boundary]
    recent_msgs = messages[boundary:]

    if not middle_msgs:
        logger.warning("No safe middle segment available to summarize.")
        return messages

    # Build summary of middle messages
    summary_lines = [
        _format_message_for_summary(msg) for msg in middle_msgs
    ]
    summary_text = (
        "[CONTEXT SUMMARY — older messages were compressed to save tokens]\n"
        + "\n".join(f"• {line}" for line in summary_lines)
    )

    # Construct the compressed message list
    compressed: list[BaseMessage] = []
    if system_msg:
        compressed.append(system_msg)
    compressed.append(SystemMessage(content=summary_text))
    compressed.extend(recent_msgs)

    new_tokens = count_tokens(compressed)
    logger.info(
        f"Compression complete: {len(messages)} → {len(compressed)} messages, "
        f"{current_tokens} → {new_tokens} tokens "
        f"(saved {current_tokens - new_tokens} tokens)"
    )

    return compressed
