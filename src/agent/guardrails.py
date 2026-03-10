"""
Guardrails (Sprint 4D)
=======================
Content sanitization, tool-description validation, and external-content tagging.

Inspired by AgentPrune adversarial isolation — severs malicious inputs
from affecting agent reasoning.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt-injection detection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[SYSTEM\s*(PROMPT|MESSAGE)\]", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"override\s+(your|the)\s+(role|instructions|prompt)", re.IGNORECASE),
]

# Max length for MCP tool descriptions before flagging
MAX_TOOL_DESC_LEN = int(os.getenv("MAX_TOOL_DESC_LEN", 2000))

# Suspicious patterns in tool descriptions
_DESC_SUSPICIOUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+must\s+(always|never)", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
]

EXTERNAL_START = "[EXTERNAL CONTENT START]"
EXTERNAL_END = "[EXTERNAL CONTENT END]"


# ---------------------------------------------------------------------------
# Content Sanitization
# ---------------------------------------------------------------------------

def sanitize_tool_output(content: str) -> tuple[str, bool]:
    """Scan tool output for prompt-injection signatures.

    Returns:
        (cleaned_content, was_sanitized)
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(content):
            original_len = len(content)
            logger.warning(
                f"[Guardrail] Prompt injection detected in tool output "
                f"(pattern: {pattern.pattern!r}, len={original_len}). Sanitizing."
            )
            return (
                f"[CONTENT SANITIZED: potential injection pattern detected in "
                f"{original_len} chars of tool output. The raw content has been "
                f"removed for safety.]",
                True,
            )
    return content, False


# ---------------------------------------------------------------------------
# Tool Description Validation
# ---------------------------------------------------------------------------

def validate_tool_descriptions(tools: list, max_desc_len: int = MAX_TOOL_DESC_LEN) -> list[str]:
    """Check dynamically loaded tool descriptions for suspicious content.

    Returns a list of warning strings (empty if all clean).
    """
    warnings = []
    for t in tools:
        name = getattr(t, "name", None) or (t.get("function", {}).get("name") if isinstance(t, dict) else "unknown")
        desc = getattr(t, "description", None) or (t.get("function", {}).get("description", "") if isinstance(t, dict) else "")
        desc = str(desc)

        if len(desc) > max_desc_len:
            msg = f"[Guardrail] Tool '{name}' description is suspiciously long ({len(desc)} chars, cap={max_desc_len})."
            warnings.append(msg)
            logger.warning(msg)

        for pattern in _DESC_SUSPICIOUS_PATTERNS:
            if pattern.search(desc):
                msg = f"[Guardrail] Tool '{name}' description contains suspicious pattern: {pattern.pattern!r}."
                warnings.append(msg)
                logger.warning(msg)

    return warnings


# ---------------------------------------------------------------------------
# External Content Tagging
# ---------------------------------------------------------------------------

def tag_external_content(content: str) -> str:
    """Wrap content from external file fetches with safety markers."""
    return f"{EXTERNAL_START}\n{content}\n{EXTERNAL_END}"
