"""
Context Manager: Observation Masking & Message Windowing
=========================================================
Implements the "Summarize-and-Forget" and "Observation Masking" strategies
from DESIGN.md to keep the LangGraph agent within its context window budget.

Two mechanisms:
1. Tool Output Truncation — caps verbose tool responses before they enter history.
2. Message Windowing — compresses older messages into a summary when token
   count exceeds the budget.

IMPORTANT: Windowing is **tool-call safe**. Messages are grouped into atomic
blocks so that an AIMessage(tool_calls=...) is never separated from its
following ToolMessage(s). This prevents OpenAI 400 errors.

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
MODEL_ENCODING = os.getenv("TIKTOKEN_MODEL", "gpt-oss-20b")


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


# ── Message Grouping (Tool-Call Safety) ───────────────────────────────────

def _group_messages(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """Group messages into atomic blocks that must never be split.

    An AIMessage with tool_calls and its subsequent ToolMessage(s) form
    a single block. All other messages are individual blocks.

    This ensures we never break the OpenAI invariant:
    'An assistant message with tool_calls must be followed by tool messages
    responding to each tool_call_id.'
    """
    groups: list[list[BaseMessage]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Start an atomic block: AIMessage + all following ToolMessages
            block = [msg]
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                block.append(messages[j])
                j += 1
            groups.append(block)
            i = j
        else:
            groups.append([msg])
            i += 1
    return groups


# ── Message Windowing (Summarize-and-Forget) ──────────────────────────────

def _format_message_for_summary(msg: BaseMessage) -> str:
    """Format a single message into a concise summary line."""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
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


def summarize_and_window(
    messages: list[BaseMessage],
    max_tokens: int | None = None,
    keep_recent: int | None = None,
) -> list[BaseMessage]:
    """Apply the Summarize-and-Forget windowing strategy (tool-call safe).

    Messages are first grouped into atomic blocks so that an AIMessage with
    tool_calls is never separated from its ToolMessage responses.

    If the total token count exceeds ``max_tokens``:
    1. Keep the SystemMessage at index 0 (if present).
    2. Keep the last N *groups* (not individual messages) as the working set.
    3. Summarize everything in between into a single SystemMessage.
    4. Return the compressed list.

    If under budget, returns the original list unchanged.
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

    # Separate system prompt
    has_system = isinstance(messages[0], SystemMessage) if messages else False
    system_msg = messages[0] if has_system else None
    body = messages[1:] if has_system else messages[:]

    # Group into atomic blocks (tool-call safe)
    groups = _group_messages(body)

    if len(groups) <= recent_count:
        logger.warning(
            "Not enough message groups to summarize (%d groups, need > %d). "
            "Skipping compression.",
            len(groups),
            recent_count,
        )
        return messages

    # Split: middle groups (to summarize) and recent groups (to keep)
    middle_groups = groups[:-recent_count]
    recent_groups = groups[-recent_count:]

    # Flatten each section
    middle_msgs = [msg for group in middle_groups for msg in group]
    recent_msgs = [msg for group in recent_groups for msg in group]

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
