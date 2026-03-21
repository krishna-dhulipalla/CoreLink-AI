"""
Self-Reflection Node
====================
Final-only bounded quality check for benchmark-style qualitative tasks.
Runs after reviewer/compliance pass, and can trigger at most one targeted
final revise cycle.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.context.legal_dimensions import (
    legal_allocation_groups,
    legal_employee_transfer_groups,
    legal_execution_groups,
    legal_regulatory_execution_groups,
    legal_tax_execution_groups,
    normalize_legal_task_text,
)
from agent.cost import CostTracker
from agent.model_config import get_model_name, invoke_structured_output
from agent.runtime_clock import increment_runtime_step
from agent.state import AgentState
from agent.tracer import get_tracer
from context_manager import count_tokens

logger = logging.getLogger(__name__)

_ELIGIBLE_TEMPLATES = {
    "legal_reasoning_only",
    "legal_with_document_evidence",
    "equity_research_report",
    "portfolio_risk_review",
    "event_driven_finance",
    "regulated_actionable_finance",
}

_PROMPT = """You are the final answer self-checker.
Review the candidate final answer only after reviewer, risk, and compliance checks have already passed.
Your job is not to restate the answer. Your job is to decide whether one targeted final improvement pass is still justified.

Be conservative. Do not request another rewrite unless the answer is materially incomplete on actionable depth, execution detail, or risk framing.

Return ONLY one JSON object with:
- score: 0.0-1.0
- complete: true | false
- missing_dimensions: list of short strings
- improve_prompt: one short sentence describing exactly what to add
"""


class ReflectionResult(BaseModel):
    score: float = 0.85
    complete: bool = True
    missing_dimensions: list[str] = Field(default_factory=list)
    improve_prompt: str = ""


def _self_reflection_enabled() -> bool:
    disable_flag = os.getenv("DISABLE_FINAL_SELF_REFLECTION", "").strip().lower()
    if disable_flag in {"1", "true", "yes", "on"}:
        return False
    flag = os.getenv("ENABLE_FINAL_SELF_REFLECTION", "").strip().lower()
    benchmark_mode = os.getenv("BENCHMARK_STATELESS", "").strip().lower() in {"1", "true", "yes", "on"}
    return benchmark_mode or flag in {"1", "true", "yes", "on"}


def should_run_self_reflection(state: AgentState) -> bool:
    workpad = state.get("workpad", {}) or {}
    benchmark_or_explicit = _self_reflection_enabled()
    complexity_tier = str(workpad.get("task_complexity_tier", ""))
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))

    if not benchmark_or_explicit and complexity_tier != "complex_qualitative":
        return False
    if int(workpad.get("self_reflection_attempts", 0)) >= 1:
        return False
    if state.get("answer_contract", {}).get("requires_adapter"):
        return False
    return template_id in _ELIGIBLE_TEMPLATES


def route_from_self_reflection(state: AgentState) -> str:
    if state.get("solver_stage") == "COMPLETE":
        if state.get("answer_contract", {}).get("requires_adapter"):
            return "output_adapter"
        return "reflect"
    return "solver"


def _final_answer_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return str(msg.content)
    return str((state.get("workpad", {}) or {}).get("draft_answer", ""))


def _keyword_hits(text: str, groups: list[list[str]]) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    hits = 0
    for group in groups:
        if any(token in normalized for token in group):
            hits += 1
    return hits


def _heuristic_reflection(answer: str, state: AgentState) -> ReflectionResult:
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    normalized = re.sub(r"\s+", " ", (answer or "").lower()).strip()
    score = 0.78
    missing: list[str] = []

    if len(answer.strip()) > 1200:
        score += 0.08
    elif len(answer.strip()) < 650:
        score -= 0.12

    if template_id in {"legal_reasoning_only", "legal_with_document_evidence"}:
        allocation_groups = legal_allocation_groups()
        execution_groups = legal_execution_groups()
        allocation_hits = _keyword_hits(answer, allocation_groups)
        execution_hits = _keyword_hits(answer, execution_groups)
        if allocation_hits >= 4:
            score += 0.10
        if execution_hits >= 3:
            score += 0.08
        if "next step" in normalized or "next steps" in normalized:
            score += 0.04

        if allocation_hits < 3:
            score -= 0.16
            missing.append("liability allocation detail")
        if execution_hits < 2:
            score -= 0.12
            missing.append("execution timing detail")
        if "next step" not in normalized and "next steps" not in normalized:
            score -= 0.06
            missing.append("actionable next steps")
        task_text = normalize_legal_task_text(
            next(
            (str(msg.content) for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)),
            "",
        ))
        if any(token in task_text for token in ("stock consideration", "stock-for-stock", "tax reasons", "tax")):
            tax_hits = _keyword_hits(answer, legal_tax_execution_groups())
            if tax_hits < 2:
                score -= 0.10
                missing.append("tax execution detail")
        if any(token in task_text for token in ("eu", "us", "cross-border", "compliance")):
            regulatory_hits = _keyword_hits(answer, legal_regulatory_execution_groups())
            if regulatory_hits < 2:
                score -= 0.10
                missing.append("regulatory execution detail")
            employee_hits = _keyword_hits(answer, legal_employee_transfer_groups())
            if employee_hits < 2:
                score -= 0.10
                missing.append("employee-transfer detail")

    elif template_id == "equity_research_report":
        for label, tokens in {
            "thesis clarity": ["thesis", "view", "recommendation"],
            "valuation support": ["valuation", "multiple", "dcf", "target"],
            "risk framing": ["risk", "downside", "watch"],
            "view-change triggers": ["would change", "watch for", "changes the view", "invalidates"],
        }.items():
            if not any(token in normalized for token in tokens):
                score -= 0.06
                missing.append(label)

    elif template_id == "portfolio_risk_review":
        for label, tokens in {
            "recommended actions": ["recommend", "rebalance", "trim", "hedge", "reduce"],
            "limit or breach framing": ["limit", "breach", "cap", "budget"],
            "scenario framing": ["scenario", "stress", "drawdown", "var"],
        }.items():
            if not any(token in normalized for token in tokens):
                score -= 0.06
                missing.append(label)

    elif template_id in {"event_driven_finance", "regulated_actionable_finance"}:
        for label, tokens in {
            "scenario coverage": ["base", "upside", "downside", "scenario"],
            "execution discipline": ["entry", "exit", "sizing", "monitor", "watch"],
            "risk disclosure": ["risk", "uncertainty", "disclosure"],
        }.items():
            if not any(token in normalized for token in tokens):
                score -= 0.06
                missing.append(label)

    score = max(0.0, min(1.0, score))
    complete = score >= 0.88 and not missing
    improve_prompt = ""
    if not complete and missing:
        improve_prompt = "Add one concise layer of actionable depth for: " + ", ".join(missing[:3]) + "."
    return ReflectionResult(score=score, complete=complete, missing_dimensions=missing[:5], improve_prompt=improve_prompt)


def _fallback_result(heuristic: ReflectionResult) -> ReflectionResult:
    if heuristic.complete:
        return heuristic
    return heuristic


def _targeted_legal_improve_prompt(missing_dimensions: list[str]) -> str:
    normalized = [str(item).lower() for item in missing_dimensions]
    additions: list[str] = []
    if any("liability allocation" in item for item in normalized):
        additions.append("explicit indemnities, escrow or holdback, caps or baskets, and survival periods")
    if any("regulatory execution" in item for item in normalized):
        additions.append("regulatory approvals, remediation covenants, and closing-condition mechanics")
    if any("tax execution" in item for item in normalized):
        additions.append("tax execution steps, who gets the tax benefit, required elections or qualification conditions, and what could break the intended treatment")
    if any("employee-transfer" in item for item in normalized):
        additions.append("employee-transfer and cross-border employment transition points")
    if any("execution timing" in item for item in normalized):
        additions.append("signing-to-closing timing, consent sequencing, and cure mechanics")
    if not additions:
        additions.append("one concrete layer of execution detail for each missing legal dimension")
    return "Add concise but concrete legal execution detail covering " + ", ".join(additions[:3]) + "."


def _review_loop_reflection_result(state: AgentState) -> ReflectionResult | None:
    workpad = state.get("workpad", {}) or {}
    events = list(workpad.get("events", []))
    if not any(
        str(event.get("action", "")).startswith("repeat-review-loop -> terminate")
        or str(event.get("action", "")) == "budget exit -> revise cap exhausted"
        for event in events
        if isinstance(event, dict)
    ):
        return None

    review_feedback = state.get("review_feedback") or {}
    missing = [str(item) for item in review_feedback.get("missing_dimensions", []) if str(item).strip()]
    if not missing:
        review_results = list(workpad.get("review_results", []))
        for item in reversed(review_results):
            if isinstance(item, dict) and item.get("verdict") == "revise" and item.get("is_final"):
                missing = [str(entry) for entry in item.get("missing_dimensions", []) if str(entry).strip()]
                break
    if not missing:
        return None

    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    improve_prompt = ""
    if template_id in {"legal_reasoning_only", "legal_with_document_evidence"}:
        improve_prompt = _targeted_legal_improve_prompt(missing)
    else:
        improve_prompt = "Add one targeted final improvement pass for: " + ", ".join(missing[:3]) + "."
    return ReflectionResult(score=0.6, complete=False, missing_dimensions=missing[:5], improve_prompt=improve_prompt)


def self_reflection(state: AgentState) -> dict[str, Any]:
    step = increment_runtime_step()
    workpad = dict(state.get("workpad", {}) or {})
    attempts = int(workpad.get("self_reflection_attempts", 0))
    answer = _final_answer_text(state)

    if not should_run_self_reflection(state):
        workpad["self_reflection_attempts"] = attempts
        events = list(workpad.get("events", []))
        events.append({"node": "self_reflection", "action": "skipped"})
        workpad["events"] = events
        tracer = get_tracer()
        if tracer:
            tracer.record("self_reflection", {
                "action": "skipped",
                "reason": "not eligible or already attempted",
            })
        return {"solver_stage": "COMPLETE", "workpad": workpad, "reflection_feedback": None}

    forced_result = _review_loop_reflection_result(state)
    if forced_result is not None:
        workpad["self_reflection_attempts"] = attempts + 1
        history = list(workpad.get("self_reflection_results", []))
        history.append(forced_result.model_dump())
        workpad["self_reflection_results"] = history
        events = list(workpad.get("events", []))
        review_feedback = {
            "verdict": "revise",
            "reasoning": forced_result.improve_prompt or "Final answer needs one targeted improvement pass.",
            "missing_dimensions": list(forced_result.missing_dimensions),
            "repair_target": "final",
            "repair_class": "missing_section",
        }
        events.append({"node": "self_reflection", "action": f"REVISE: {review_feedback['reasoning']}"})
        workpad["events"] = events
        return {
            "solver_stage": "REVISE",
            "review_feedback": review_feedback,
            "reflection_feedback": forced_result.model_dump(),
            "workpad": workpad,
        }

    heuristic = _heuristic_reflection(answer, state)
    result = heuristic
    tracker: CostTracker = state.get("cost_tracker")
    used_llm = False
    latency = 0.0
    model_name = get_model_name("reflection")

    if not heuristic.complete:
        invocation_messages = [
            SystemMessage(content=_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "task_profile": state.get("task_profile", "general"),
                        "execution_template": state.get("execution_template", {}),
                        "task": next(
                            (str(msg.content) for msg in reversed(state.get("messages", [])) if isinstance(msg, HumanMessage)),
                            "",
                        ),
                        "answer": answer,
                        "heuristic": heuristic.model_dump(),
                        "risk_requirements": (workpad.get("risk_requirements") or {}),
                        "compliance_requirements": (workpad.get("compliance_requirements") or {}),
                    },
                    ensure_ascii=True,
                )
            ),
        ]
        try:
            used_llm = True
            t0 = time.monotonic()
            result, _ = invoke_structured_output(
                "reflection",
                ReflectionResult,
                invocation_messages,
                temperature=0,
                max_tokens=220,
            )
            latency = (time.monotonic() - t0) * 1000
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            logger.warning("Self-reflection parse failed: %s. Using heuristic fallback.", exc)
            result = _fallback_result(heuristic)
        if tracker and used_llm:
            tracker.record(
                operator="self_reflection",
                model_name=model_name,
                tokens_in=count_tokens(invocation_messages),
                tokens_out=count_tokens([AIMessage(content=result.model_dump_json())]),
                latency_ms=latency,
                success=result.complete,
            )

    workpad["self_reflection_attempts"] = attempts + 1
    history = list(workpad.get("self_reflection_results", []))
    history.append(result.model_dump())
    workpad["self_reflection_results"] = history
    events = list(workpad.get("events", []))

    if result.complete or not (result.missing_dimensions or result.improve_prompt):
        events.append({"node": "self_reflection", "action": "PASS"})
        workpad["events"] = events
        return {
            "solver_stage": "COMPLETE",
            "workpad": workpad,
            "reflection_feedback": result.model_dump(),
        }

    review_feedback = {
        "verdict": "revise",
        "reasoning": result.improve_prompt or "Final answer needs one targeted improvement pass.",
        "missing_dimensions": list(result.missing_dimensions),
        "repair_target": "final",
        "repair_class": "missing_section",
    }
    events.append({"node": "self_reflection", "action": f"REVISE: {review_feedback['reasoning']}"})
    workpad["events"] = events
    tracer = get_tracer()
    if tracer:
        tracer.record("self_reflection", {
            "action": "REVISE",
            "used_llm": used_llm,
            "score": result.score,
            "missing_dimensions": list(result.missing_dimensions),
            "improve_prompt": result.improve_prompt,
        })
    return {
        "solver_stage": "REVISE",
        "review_feedback": review_feedback,
        "reflection_feedback": result.model_dump(),
        "workpad": workpad,
    }
