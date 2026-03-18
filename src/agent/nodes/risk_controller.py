"""
Risk Controller Node
====================
Template-scoped finance risk gate that validates compute artifacts before
final synthesis on risk-bearing finance templates.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.contracts import RiskResult
from agent.runtime_clock import increment_runtime_step
from agent.state import AgentState

logger = logging.getLogger(__name__)

_RISK_CONTROLLED_TEMPLATES = {"options_tool_backed", "portfolio_risk_review"}
_PRIMARY_OPTIONS_TOOLS = {"analyze_strategy", "black_scholes_price", "option_greeks", "mispricing_analysis"}
_RISK_TOOLS = {
    "scenario_pnl",
    "run_stress_test",
    "calculate_var",
    "portfolio_limit_check",
    "concentration_check",
    "calculate_portfolio_greeks",
}
_PORTFOLIO_RISK_TOOLS = {
    "factor_exposure_summary",
    "drawdown_risk_profile",
    "liquidity_stress",
    "calculate_risk_metrics",
    "calculate_var",
    "portfolio_limit_check",
    "concentration_check",
    "run_stress_test",
}


def requires_risk_control(state: AgentState) -> bool:
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    review_stage = str((state.get("workpad") or {}).get("review_stage", state.get("solver_stage", "SYNTHESIZE")))
    return template_id in _RISK_CONTROLLED_TEMPLATES and review_stage == "COMPUTE"


def route_from_risk_controller(state: AgentState) -> str:
    if state.get("solver_stage") == "REVISE":
        return "solver"
    return "reviewer"


def _record_event(workpad: dict[str, Any], action: str) -> dict[str, Any]:
    updated = dict(workpad)
    events = list(updated.get("events", []))
    events.append({"node": "risk_controller", "action": action})
    updated["events"] = events
    return updated


def _append_risk_result(workpad: dict[str, Any], result: RiskResult) -> dict[str, Any]:
    updated = dict(workpad)
    items = list(updated.get("risk_results", []))
    items.append(result.model_dump())
    updated["risk_results"] = items
    return updated


def _latest_successful_tool(tool_results: list[dict[str, Any]], tool_names: set[str]) -> dict[str, Any] | None:
    for result in reversed(tool_results):
        if not isinstance(result, dict):
            continue
        if str(result.get("type", "")) not in tool_names:
            continue
        if result.get("errors"):
            continue
        if isinstance(result.get("facts"), dict) and result.get("facts"):
            return result
    return None


def _looks_unbounded_short_vol(tool_result: dict[str, Any]) -> bool:
    assumptions = tool_result.get("assumptions", {})
    legs = assumptions.get("legs")
    if not isinstance(legs, list):
        return False
    short_calls = []
    short_puts = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        if str(leg.get("action", "")).lower() != "sell":
            continue
        option_type = str(leg.get("option_type", "")).lower()
        if option_type == "call":
            short_calls.append(float(leg.get("K", 0.0)))
        elif option_type == "put":
            short_puts.append(float(leg.get("K", 0.0)))
    if short_calls and not any(str(leg.get("action", "")).lower() == "buy" and str(leg.get("option_type", "")).lower() == "call" for leg in legs if isinstance(leg, dict)):
        return True
    return bool(set(short_calls) & set(short_puts))


def _has_risk_controls(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    tokens = ("stop-loss", "stop loss", "sizing", "position size", "hedge", "max loss", "risk limit")
    return any(token in normalized for token in tokens)


def _has_required_disclosure(answer_text: str, disclosure: str) -> bool:
    normalized = re.sub(r"\s+", " ", (answer_text or "").lower()).strip()
    lowered = disclosure.lower()
    if "short-volatility" in lowered or "volatility-spike" in lowered:
        return "short vol" in normalized or "volatility spike" in normalized or "vol spike" in normalized
    if "tail loss" in lowered or "gap risk" in lowered or "unbounded" in lowered:
        return "gap risk" in normalized or "unbounded" in normalized or "tail risk" in normalized or "tail loss" in normalized
    if "max loss" in lowered:
        return "max loss" in normalized
    if "position sizing" in lowered:
        return "position size" in normalized or "sizing" in normalized
    return False


def _has_portfolio_actions(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return any(token in normalized for token in ("rebalance", "trim", "reduce", "hedge", "hold", "increase", "decrease", "limit"))


def risk_controller(state: AgentState) -> dict[str, Any]:
    step = increment_runtime_step()
    workpad = dict(state.get("workpad", {}))
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))

    if not requires_risk_control(state):
        workpad = _record_event(workpad, "skipped")
        logger.info("[Step %s] risk_controller -> skipped", step)
        return {"workpad": workpad, "risk_feedback": None}

    compute_artifact = str((workpad.get("stage_outputs") or {}).get("COMPUTE", ""))
    tool_results = list(workpad.get("tool_results", []))

    if template_id == "portfolio_risk_review":
        portfolio_tool = _latest_successful_tool(tool_results, _PORTFOLIO_RISK_TOOLS)
        limit_tool = _latest_successful_tool(tool_results, {"portfolio_limit_check"})
        concentration_tool = _latest_successful_tool(tool_results, {"concentration_check"})
        var_tool = _latest_successful_tool(tool_results, {"calculate_var"})
        drawdown_tool = _latest_successful_tool(tool_results, {"drawdown_risk_profile"})
        liquidity_tool = _latest_successful_tool(tool_results, {"liquidity_stress"})

        violation_codes: list[str] = []
        risk_findings: list[str] = []
        required_disclosures: list[str] = []

        if portfolio_tool is None:
            violation_codes.append("MISSING_PORTFOLIO_RISK_EVIDENCE")
            risk_findings.append("No structured portfolio risk evidence was available for review.")

        if concentration_tool:
            facts = concentration_tool.get("facts", {})
            if facts.get("has_breach"):
                risk_findings.append("Concentration limits are breached for at least one name or sector.")
                required_disclosures.append("Explicitly state the concentration breach and the trim or rebalance response.")

        if limit_tool:
            facts = limit_tool.get("facts", {})
            if facts.get("hard_limit_breached"):
                result = RiskResult(
                    verdict="blocked",
                    reasoning="Portfolio risk review found a hard risk-limit breach.",
                    violation_codes=[
                        str(item.get("code", "LIMIT_BREACH"))
                        for item in facts.get("breaches", [])
                        if isinstance(item, dict)
                    ] or ["LIMIT_BREACH"],
                    risk_findings=risk_findings + ["Portfolio limit breach detected."],
                    required_disclosures=list(dict.fromkeys(required_disclosures)),
                    recommendation_class="insufficient_evidence_no_action",
                    repair_target="compute",
                )
                workpad = _append_risk_result(workpad, result)
                workpad = _record_event(workpad, f"BLOCKED: {', '.join(result.violation_codes)}")
                workpad["review_ready"] = False
                workpad["review_stage"] = None
                logger.info("[Step %s] risk_controller -> BLOCKED", step)
                return {
                    "solver_stage": "REVISE",
                    "risk_feedback": result.model_dump(),
                    "review_feedback": None,
                    "workpad": workpad,
                    "pending_tool_call": None,
                }

        if var_tool:
            var_facts = var_tool.get("facts", {})
            if isinstance(var_facts.get("var_amount"), (int, float)):
                risk_findings.append(f"Estimated VaR is approximately {float(var_facts['var_amount']):.2f}.")
                required_disclosures.append("State the estimated VaR or worst-case loss reference.")

        if drawdown_tool:
            dd_facts = drawdown_tool.get("facts", {})
            if isinstance(dd_facts.get("max_drawdown_decimal"), (int, float)):
                risk_findings.append(
                    f"Historical max drawdown is approximately {float(dd_facts['max_drawdown_decimal']) * 100:.2f}%."
                )

        if liquidity_tool:
            liq_facts = liquidity_tool.get("facts", {})
            if str(liq_facts.get("stress_assessment", "")) == "tight":
                risk_findings.append("Liquidity stress indicates the portfolio could struggle to meet a redemption or rebalance demand cleanly.")
                required_disclosures.append("State the liquidity constraint and the staged reduction plan.")

        if not _has_portfolio_actions(compute_artifact):
            violation_codes.append("MISSING_PORTFOLIO_ACTIONS")
            risk_findings.append("Compute artifact does not yet translate risk findings into concrete portfolio actions.")
            required_disclosures.append("Include specific trim, hedge, rebalance, or hold actions tied to the risk evidence.")

        deduped_disclosures = list(dict.fromkeys(required_disclosures))
        if violation_codes:
            result = RiskResult(
                verdict="revise",
                reasoning="Portfolio risk review needs clearer risk evidence or concrete mitigation actions before synthesis.",
                violation_codes=violation_codes,
                risk_findings=risk_findings,
                required_disclosures=deduped_disclosures,
                recommendation_class="scenario_dependent_recommendation",
                repair_target="compute",
            )
            workpad = _append_risk_result(workpad, result)
            workpad = _record_event(workpad, f"REVISE: {', '.join(violation_codes)}")
            workpad["review_ready"] = False
            workpad["review_stage"] = None
            logger.info("[Step %s] risk_controller -> REVISE", step)
            return {
                "solver_stage": "REVISE",
                "risk_feedback": result.model_dump(),
                "review_feedback": None,
                "workpad": workpad,
                "pending_tool_call": None,
            }

        result = RiskResult(
            verdict="pass",
            reasoning="Portfolio risk review accepted the compute branch.",
            violation_codes=[],
            risk_findings=risk_findings,
            required_disclosures=deduped_disclosures,
            recommendation_class="scenario_dependent_recommendation",
            repair_target="final",
        )
        workpad = _append_risk_result(workpad, result)
        workpad["risk_requirements"] = {
            "required_disclosures": deduped_disclosures,
            "risk_findings": risk_findings,
            "recommendation_class": "scenario_dependent_recommendation",
        }
        workpad = _record_event(workpad, "PASS")
        logger.info("[Step %s] risk_controller -> PASS", step)
        return {
            "workpad": workpad,
            "risk_feedback": None,
        }

    primary_tool = _latest_successful_tool(tool_results, _PRIMARY_OPTIONS_TOOLS)
    risk_tool = _latest_successful_tool(tool_results, _RISK_TOOLS)

    violation_codes: list[str] = []
    risk_findings: list[str] = []
    required_disclosures: list[str] = []
    recommendation_class = "scenario_dependent_recommendation"

    if primary_tool is None:
        violation_codes.append("MISSING_PRIMARY_STRATEGY_FACTS")
        risk_findings.append("No structured primary strategy facts were available for risk review.")
    else:
        facts = primary_tool.get("facts", {})
        total_vega = facts.get("total_vega_per_vol_point", facts.get("vega"))
        total_theta = facts.get("total_theta_per_day", facts.get("theta"))
        if isinstance(total_vega, (int, float)) and total_vega < 0:
            risk_findings.append("Primary strategy is short volatility and can lose value sharply in a vol spike.")
            required_disclosures.append("Explicitly disclose short-volatility / volatility-spike risk.")
        if isinstance(total_theta, (int, float)) and total_theta > 0:
            risk_findings.append("Primary strategy relies on time decay and needs exit/monitoring discipline.")
        if _looks_unbounded_short_vol(primary_tool):
            risk_findings.append("Primary strategy carries potentially unbounded or very large tail loss.")
            required_disclosures.append("Explicitly disclose potentially unbounded tail loss and gap risk.")
        elif "max_loss" in facts:
            required_disclosures.append("State max loss explicitly.")

    if risk_tool is None:
        violation_codes.append("MISSING_SCENARIO_ANALYSIS")
        risk_findings.append("No scenario, stress, or limit-check evidence was attached to the strategy.")
    else:
        facts = risk_tool.get("facts", {})
        breaches = facts.get("breaches", []) if isinstance(facts, dict) else []
        if facts.get("hard_limit_breached") and breaches:
            result = RiskResult(
                verdict="blocked",
                reasoning="Risk limits were breached for the current compute branch.",
                violation_codes=[str(item.get("code", "LIMIT_BREACH")) for item in breaches if isinstance(item, dict)],
                risk_findings=risk_findings + ["Portfolio limit breach detected."],
                required_disclosures=sorted(dict.fromkeys(required_disclosures)),
                recommendation_class="insufficient_evidence_no_action",
                repair_target="compute",
            )
            workpad = _append_risk_result(workpad, result)
            workpad = _record_event(workpad, f"BLOCKED: {', '.join(result.violation_codes) or 'limit breach'}")
            workpad["review_ready"] = False
            workpad["review_stage"] = None
            logger.info("[Step %s] risk_controller -> BLOCKED", step)
            return {
                "solver_stage": "REVISE",
                "risk_feedback": result.model_dump(),
                "review_feedback": None,
                "workpad": workpad,
                "pending_tool_call": None,
            }
        worst_case_pnl = facts.get("worst_case_pnl")
        if isinstance(worst_case_pnl, (int, float)) and worst_case_pnl < 0:
            risk_findings.append(f"Scenario analysis shows downside loss of approximately {worst_case_pnl:.2f}.")
            required_disclosures.append("State downside scenario loss and the exit or sizing response.")

    if not _has_risk_controls(compute_artifact):
        violation_codes.append("MISSING_RISK_CONTROLS")
        risk_findings.append("Compute artifact does not yet include position sizing, stop-loss, or hedge controls.")
        required_disclosures.append("Include explicit position sizing and stop-loss or hedge controls.")

    deduped_disclosures = list(dict.fromkeys(required_disclosures))
    if violation_codes:
        result = RiskResult(
            verdict="revise",
            reasoning="Risk controller requires additional scenario coverage or explicit controls before synthesis.",
            violation_codes=violation_codes,
            risk_findings=risk_findings,
            required_disclosures=deduped_disclosures,
            recommendation_class=recommendation_class,
            repair_target="compute",
        )
        workpad = _append_risk_result(workpad, result)
        workpad = _record_event(workpad, f"REVISE: {', '.join(violation_codes)}")
        workpad["review_ready"] = False
        workpad["review_stage"] = None
        logger.info("[Step %s] risk_controller -> REVISE", step)
        return {
            "solver_stage": "REVISE",
            "risk_feedback": result.model_dump(),
            "review_feedback": None,
            "workpad": workpad,
            "pending_tool_call": None,
        }

    result = RiskResult(
        verdict="pass",
        reasoning="Risk controller accepted the compute branch.",
        violation_codes=[],
        risk_findings=risk_findings,
        required_disclosures=deduped_disclosures,
        recommendation_class=recommendation_class,
        repair_target="final",
    )
    workpad = _append_risk_result(workpad, result)
    workpad["risk_requirements"] = {
        "required_disclosures": deduped_disclosures,
        "risk_findings": risk_findings,
        "recommendation_class": recommendation_class,
    }
    workpad = _record_event(workpad, "PASS")
    logger.info("[Step %s] risk_controller -> PASS", step)
    return {
        "workpad": workpad,
        "risk_feedback": None,
    }
