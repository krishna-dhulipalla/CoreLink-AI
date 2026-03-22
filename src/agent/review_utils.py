"""Shared review heuristics for the active engine."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.context.legal_dimensions import (
    legal_allocation_groups,
    legal_employee_transfer_groups,
    legal_execution_groups,
    legal_regulatory_execution_groups,
    legal_tax_execution_groups,
    normalize_legal_task_text,
)
from agent.profile_packs import get_profile_pack


def looks_truncated(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith((":", ",", "(", "[", "{", "/", "-", "**")):
        return True
    last_line = stripped.splitlines()[-1].strip()
    if last_line.startswith(("-", "*")) and len(last_line) < 8:
        return True
    return False


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value.strip()))
    return False


def matches_exact_json_contract(answer_text: str, answer_contract: dict[str, Any]) -> bool:
    stripped = (answer_text or "").strip()
    wrapper_key = answer_contract.get("wrapper_key")
    if _is_numeric_like(stripped):
        return True
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    if wrapper_key:
        return isinstance(parsed, dict) and wrapper_key in parsed and _is_numeric_like(parsed.get(wrapper_key))
    return isinstance(parsed, (dict, list, int, float, str))


def _keyword_gaps(answer_text: str, dimensions: dict[str, list[str]]) -> list[str]:
    normalized = re.sub(r"\s+", " ", (answer_text or "").lower()).strip()
    gaps = []
    for label, tokens in dimensions.items():
        if not any(token in normalized for token in tokens):
            gaps.append(label)
    return gaps


def _count_token_group_hits(answer_text: str, groups: list[list[str]]) -> int:
    normalized = re.sub(r"\s+", " ", (answer_text or "").lower()).strip()
    hits = 0
    for group in groups:
        if any(token in normalized for token in group):
            hits += 1
    return hits


def legal_depth_gaps(answer_text: str, task_text: str = "") -> list[str]:
    gaps = _keyword_gaps(answer_text, get_profile_pack("legal_transactional").reviewer_dimensions)
    normalized_task = normalize_legal_task_text(task_text)
    if _count_token_group_hits(answer_text, legal_allocation_groups()) < 3:
        gaps.append("liability allocation mechanics")
    if _count_token_group_hits(answer_text, legal_execution_groups()) < 2:
        gaps.append("execution timing and closing mechanics")
    if any(token in normalized_task for token in ("stock consideration", "stock-for-stock", "equity consideration", "rollover", "tax reasons", "tax")):
        if _count_token_group_hits(answer_text, legal_tax_execution_groups()) < 2:
            gaps.append("tax execution mechanics")
    if any(token in normalized_task for token in ("eu", "us", "cross-border", "compliance", "regulatory", "employment", "labor", "workforce", "consultation")):
        if _count_token_group_hits(answer_text, legal_regulatory_execution_groups()) < 2:
            gaps.append("regulatory execution specifics")
        if _count_token_group_hits(answer_text, legal_employee_transfer_groups()) < 2:
            gaps.append("workforce transfer or consultation considerations")
    return sorted(set(gaps))


def options_gaps(answer_text: str) -> list[str]:
    return _keyword_gaps(answer_text, get_profile_pack("finance_options").reviewer_dimensions)
