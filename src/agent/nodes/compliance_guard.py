"""
Compliance Guard Node
=====================
Template-scoped finance policy gate for actionable recommendations.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage

from agent.contracts import ComplianceResult
from agent.runtime_clock import increment_runtime_step
from agent.state import AgentState

logger = logging.getLogger(__name__)

_COMPLIANCE_TEMPLATES = {"options_tool_backed"}
_NAKED_OPTION_TOKENS = ("short straddle", "short strangle", "naked call", "naked put", "uncovered call")
_DEFINED_RISK_TOKENS = ("iron condor", "defined-risk", "defined risk", "credit spread", "debit spread", "vertical spread")


def requires_compliance_guard(state: AgentState) -> bool:
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    review_stage = str((state.get("workpad") or {}).get("review_stage", state.get("solver_stage", "SYNTHESIZE")))
    policy_context = dict((state.get("evidence_pack") or {}).get("policy_context", {}))
    if review_stage != "SYNTHESIZE":
        return False
    if template_id in _COMPLIANCE_TEMPLATES:
        return True
    return bool(policy_context.get("action_orientation") and state.get("task_profile") == "finance_quant")


def route_from_compliance_guard(state: AgentState) -> str:
    if state.get("solver_stage") == "REVISE":
        return "solver"
    return "reviewer"


def _record_event(workpad: dict[str, Any], action: str) -> dict[str, Any]:
    updated = dict(workpad)
    events = list(updated.get("events", []))
    events.append({"node": "compliance_guard", "action": action})
    updated["events"] = events
    return updated


def _append_compliance_result(workpad: dict[str, Any], result: ComplianceResult) -> dict[str, Any]:
    updated = dict(workpad)
    items = list(updated.get("compliance_results", []))
    items.append(result.model_dump())
    updated["compliance_results"] = items
    return updated


def _latest_answer_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return str((state.get("workpad") or {}).get("draft_answer", ""))


def _extract_section(answer_text: str, section_name: str) -> str:
    lines = (answer_text or "").splitlines()
    collecting = False
    collected: list[str] = []
    target = section_name.strip().lower()
    for raw_line in lines:
        line = raw_line.strip()
        normalized = re.sub(r"[*:_\s]+", " ", line.lower()).strip()
        if target in normalized:
            collecting = True
            continue
        if collecting and line.startswith("**") and line.endswith("**"):
            break
        if collecting:
            collected.append(line)
    return " ".join(part for part in collected if part).strip()


def _has_structured_timestamp(state: AgentState) -> bool:
    for result in (state.get("workpad") or {}).get("tool_results", []):
        if not isinstance(result, dict) or result.get("errors"):
            continue
        source = result.get("source", {}) if isinstance(result.get("source", {}), dict) else {}
        facts = result.get("facts", {}) if isinstance(result.get("facts", {}), dict) else {}
        assumptions = result.get("assumptions", {}) if isinstance(result.get("assumptions", {}), dict) else {}
        if any(source.get(key) for key in ("timestamp", "as_of_date")):
            return True
        if any(facts.get(key) for key in ("as_of_date", "window_end")):
            return True
        if assumptions.get("as_of_date"):
            return True
    return False


def compliance_guard(state: AgentState) -> dict[str, Any]:
    step = increment_runtime_step()
    workpad = dict(state.get("workpad", {}))

    if not requires_compliance_guard(state):
        workpad = _record_event(workpad, "skipped")
        logger.info("[Step %s] compliance_guard -> skipped", step)
        return {"workpad": workpad, "compliance_feedback": None}

    policy_context = dict((state.get("evidence_pack") or {}).get("policy_context", {}))
    answer_text = _latest_answer_text(state)
    normalized = re.sub(r"\s+", " ", answer_text.lower()).strip()
    primary_strategy_text = _extract_section(answer_text, "Primary Strategy").lower() or normalized

    violation_codes: list[str] = []
    policy_findings: list[str] = []
    required_disclosures: list[str] = []

    if policy_context.get("requires_recommendation_class") and "recommendation class" not in normalized:
        violation_codes.append("MISSING_RECOMMENDATION_CLASS")
        policy_findings.append("Action-oriented finance answer did not declare a recommendation class.")
        required_disclosures.append("State the recommendation class explicitly.")

    if policy_context.get("requires_timestamped_evidence") and not _has_structured_timestamp(state):
        violation_codes.append("MISSING_TIMESTAMPED_EVIDENCE")
        policy_findings.append("Action-oriented answer lacks timestamped structured evidence.")
        required_disclosures.append("Disclose the source timestamp or retrieval date for action-oriented finance claims.")

    if policy_context.get("defined_risk_only") and not any(token in primary_strategy_text for token in _DEFINED_RISK_TOKENS):
        violation_codes.append("DEFINED_RISK_REQUIRED")
        policy_findings.append("Prompt requires a defined-risk structure, but the current primary strategy is not framed as defined-risk.")
        required_disclosures.append("Use a defined-risk structure and say that the mandate requires defined-risk only.")

    if policy_context.get("no_naked_options") and any(token in primary_strategy_text for token in _NAKED_OPTION_TOKENS):
        violation_codes.append("NAKED_OPTIONS_PROHIBITED")
        policy_findings.append("Prompt prohibits naked options, but the current primary strategy is still a naked short structure.")
        required_disclosures.append("State that naked options are not permitted under the mandate.")

    max_position_risk_pct = policy_context.get("max_position_risk_pct")
    if isinstance(max_position_risk_pct, (int, float)):
        if f"{max_position_risk_pct:g}%" not in normalized and "position sizing" not in normalized and "risk cap" not in normalized:
            violation_codes.append("MISSING_POSITION_RISK_CAP")
            policy_findings.append(f"Prompt provided a max position risk cap of about {max_position_risk_pct:g}% but the answer did not carry it through.")
            required_disclosures.append(f"Carry the approximate {max_position_risk_pct:g}% position-risk cap into the recommendation.")

    deduped_disclosures = list(dict.fromkeys(required_disclosures))
    if violation_codes:
        blocked = "NAKED_OPTIONS_PROHIBITED" in violation_codes and bool(policy_context.get("retail_or_retirement_account"))
        result = ComplianceResult(
            verdict="blocked" if blocked else "revise",
            reasoning="Compliance guard requires the final finance recommendation to respect explicit mandate or disclosure constraints.",
            violation_codes=violation_codes,
            policy_findings=policy_findings,
            required_disclosures=deduped_disclosures,
            repair_target="final",
        )
        workpad = _append_compliance_result(workpad, result)
        workpad["compliance_requirements"] = {
            "required_disclosures": deduped_disclosures,
            "policy_findings": policy_findings,
            "policy_context": policy_context,
        }
        workpad["review_ready"] = False
        workpad["review_stage"] = None
        workpad = _record_event(workpad, f"{result.verdict.upper()}: {', '.join(violation_codes)}")
        logger.info("[Step %s] compliance_guard -> %s", step, result.verdict.upper())
        return {
            "solver_stage": "REVISE",
            "compliance_feedback": result.model_dump(),
            "review_feedback": None,
            "workpad": workpad,
            "pending_tool_call": None,
        }

    result = ComplianceResult(
        verdict="pass",
        reasoning="Compliance guard accepted the final finance recommendation.",
        violation_codes=[],
        policy_findings=policy_findings,
        required_disclosures=[],
        repair_target="final",
    )
    workpad = _append_compliance_result(workpad, result)
    workpad["compliance_requirements"] = {
        "required_disclosures": [],
        "policy_findings": policy_findings,
        "policy_context": policy_context,
    }
    workpad = _record_event(workpad, "PASS")
    logger.info("[Step %s] compliance_guard -> PASS", step)
    return {
        "workpad": workpad,
        "compliance_feedback": None,
    }
