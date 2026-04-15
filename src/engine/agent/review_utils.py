"""Shared review helpers for the active OfficeQA runtime."""

from __future__ import annotations

import json
import re
from typing import Any


def looks_truncated(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith((":", ",", "(", "[", "{", "/", "-", "**")):
        return True
    last_line = stripped.splitlines()[-1].strip()
    if last_line.startswith(("-", "*")) and len(last_line) < 8:
        return True
    if stripped.count("(") > stripped.count(")"):
        return True
    if stripped.count("[") > stripped.count("]"):
        return True
    if stripped.count("{") > stripped.count("}"):
        return True
    if stripped.count("`") % 2 == 1:
        return True
    if "\\(" in stripped and "\\)" not in stripped:
        return True
    if "\\[" in stripped and "\\]" not in stripped:
        return True
    if re.search(r"(=|\+|-|\*|/)\s*$", stripped):
        return True
    if re.search(r"\b(?:therefore|thus|so|hence)\s*$", last_line.lower()):
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
