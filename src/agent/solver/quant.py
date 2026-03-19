"""
Inline quantitative helpers.
"""

from __future__ import annotations

import re

from agent.runtime_support import latest_human_text
from agent.solver.common import extract_inline_assignments, format_scalar_number, safe_arithmetic_eval
from agent.state import AgentState

_FORMULA_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_ ]{1,80})\s*=\s*([^\n.;]+)")


def deterministic_inline_quant_value(state: AgentState) -> float | None:
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    if template_id != "quant_inline_exact":
        return None

    task_text = latest_human_text(state.get("messages", []))
    evidence_pack = state.get("evidence_pack", {}) or {}
    formulas = list(evidence_pack.get("formulas", []))
    assignments = extract_inline_assignments(task_text)
    if not assignments:
        return None

    candidates: list[str] = []
    for source in [*formulas, task_text]:
        for match in _FORMULA_RE.finditer(source or ""):
            rhs = match.group(2).strip()
            if re.fullmatch(r"-?\d+(?:\.\d+)?%?", rhs):
                continue
            if any(ch.isalpha() for ch in rhs) or any(op in rhs for op in "+-*/"):
                candidates.append(rhs)

    for expression in candidates:
        rewritten = expression.replace("^", "**")
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
    return format_scalar_number(value)
