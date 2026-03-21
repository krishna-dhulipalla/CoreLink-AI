"""
Inline quantitative helpers.
"""

from __future__ import annotations

import json
import re

from agent.runtime_support import latest_human_text
from agent.solver.common import extract_inline_assignments, format_scalar_number, safe_arithmetic_eval
from agent.state import AgentState

_FORMULA_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_ ]{1,80})\s*=\s*([^\n.;]+)")
_PERCENT_LABEL_RE = re.compile(r"(roe|roa|return on equity|return on assets)", re.IGNORECASE)
_TEXT_BLOCK_RE = re.compile(r"\\text\{([^{}]+)\}")


def _coerce_numeric(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if stripped.endswith("%"):
            try:
                return float(stripped[:-1].strip()) / 100.0
            except ValueError:
                return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _assignments_from_relevant_rows(state: AgentState) -> dict[str, float]:
    evidence_pack = state.get("evidence_pack", {}) or {}
    relevant_rows = list(evidence_pack.get("relevant_rows", []))
    assignments: dict[str, float] = {}

    def _assign_once(name: str, numeric: float) -> bool:
        existing = assignments.get(name)
        if existing is None:
            assignments[name] = numeric
            return True
        return abs(existing - numeric) <= 1e-12

    for table in relevant_rows:
        for row in table.get("rows", []):
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                numeric = _coerce_numeric(value)
                if numeric is None:
                    continue
                key_text = str(key)
                lowered = key_text.lower()
                if _PERCENT_LABEL_RE.search(lowered):
                    if "roe" in lowered or "return on equity" in lowered:
                        if not _assign_once("ROE", numeric):
                            return {}
                    if "roa" in lowered or "return on assets" in lowered:
                        if not _assign_once("ROA", numeric):
                            return {}
                short_key = re.sub(r"[^A-Za-z0-9_]+", "_", key_text).strip("_")
                if short_key:
                    if not _assign_once(short_key, numeric):
                        return {}
                    if not _assign_once(short_key.upper(), numeric):
                        return {}
    return assignments


def _strip_latex_text(expression: str) -> str:
    previous = None
    current = expression
    while previous != current:
        previous = current
        current = _TEXT_BLOCK_RE.sub(lambda match: match.group(1), current)
    return current


def _extract_braced_segment(text: str, start_index: int) -> tuple[str, int] | tuple[None, None]:
    if start_index >= len(text) or text[start_index] != "{":
        return None, None
    depth = 0
    segment_chars: list[str] = []
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(segment_chars), index + 1
        if depth >= 1:
            segment_chars.append(char)
    return None, None


def _latex_fraction_to_expression(expression: str) -> str:
    normalized = expression
    while "\\frac" in normalized:
        frac_index = normalized.find("\\frac")
        num, next_index = _extract_braced_segment(normalized, frac_index + len("\\frac"))
        if num is None or next_index is None:
            break
        den, end_index = _extract_braced_segment(normalized, next_index)
        if den is None or end_index is None:
            break
        replacement = f"(({num})/({den}))"
        normalized = normalized[:frac_index] + replacement + normalized[end_index:]
    return normalized


def _normalize_formula_expression(expression: str) -> str:
    normalized = expression.replace("\\\\", " ")
    normalized = _strip_latex_text(normalized)
    normalized = _latex_fraction_to_expression(normalized)
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = re.sub(
        r"\(\s*[A-Za-z][A-Za-z0-9_ ]{0,80}\s*=\s*",
        "(",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if "=" in normalized:
        normalized = normalized.split("=", 1)[1].strip()
    return normalized


def deterministic_inline_quant_value(state: AgentState) -> float | None:
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    if template_id != "quant_inline_exact":
        return None

    task_text = latest_human_text(state.get("messages", []))
    evidence_pack = state.get("evidence_pack", {}) or {}
    formulas = list(evidence_pack.get("relevant_formulae", [])) or list(evidence_pack.get("formulas", []))
    assignments = extract_inline_assignments(task_text)
    assignments.update(_assignments_from_relevant_rows(state))
    if not assignments:
        return None

    candidates: list[str] = []
    for formula in formulas:
        formula_text = str(formula or "").strip()
        if not formula_text:
            continue
        if "=" in formula_text or "\\frac" in formula_text:
            candidates.append(formula_text)

    for source in [*formulas, task_text]:
        for match in _FORMULA_RE.finditer(source or ""):
            rhs = match.group(2).strip()
            if re.fullmatch(r"-?\d+(?:\.\d+)?%?", rhs):
                continue
            if any(ch.isalpha() for ch in rhs) or any(op in rhs for op in "+-*/"):
                candidates.append(rhs)

    seen_candidates: set[str] = set()
    for expression in candidates:
        normalized_candidate = _normalize_formula_expression(expression)
        if not normalized_candidate or normalized_candidate in seen_candidates:
            continue
        seen_candidates.add(normalized_candidate)
        rewritten = normalized_candidate.replace("^", "**")
        for name in sorted(assignments, key=len, reverse=True):
            rewritten = re.sub(rf"\b{re.escape(name)}\b", str(assignments[name]), rewritten)
        if re.search(r"[A-Za-z]", rewritten):
            continue
        value = safe_arithmetic_eval(rewritten)
        if value is not None:
            return float(value)
    return None


def deterministic_quant_compute_summary(state: AgentState) -> str | None:
    value = deterministic_inline_quant_value(state)
    if value is None:
        return None
    return f"Exact inline computation result: {format_scalar_number(value)}"


def deterministic_quant_final_answer(state: AgentState) -> str | None:
    value = deterministic_inline_quant_value(state)
    if value is None:
        return None
    formatted = format_scalar_number(value)
    answer_contract = state.get("answer_contract", {}) or {}
    if answer_contract.get("requires_adapter") and answer_contract.get("format") == "json":
        wrapper_key = str(answer_contract.get("wrapper_key") or "answer")
        return json.dumps({wrapper_key: float(formatted)}, ensure_ascii=True)
    return formatted
