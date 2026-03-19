"""
Portfolio-risk deterministic helpers.
"""

from __future__ import annotations

from typing import Any

from agent.solver.market import best_available_timestamp, latest_successful_tool_result
from agent.state import AgentState


def table_positions_from_evidence(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    exposures: list[dict[str, Any]] = []
    for table in (evidence_pack.get("tables", []) or []):
        rows = table.get("rows", []) if isinstance(table, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = {str(key).strip().lower(): value for key, value in row.items()}
            weight_value = None
            for key in ("weight", "portfolio weight", "position weight", "% weight", "pct weight"):
                if key in normalized:
                    weight_value = normalized[key]
                    break
            if weight_value is None:
                continue
            try:
                if isinstance(weight_value, str) and weight_value.strip().endswith("%"):
                    weight = float(weight_value.strip().rstrip("%")) / 100.0
                else:
                    weight = float(weight_value)
                    if weight > 1.0:
                        weight /= 100.0
            except Exception:
                continue
            exposures.append(
                {
                    "ticker": str(normalized.get("ticker", normalized.get("name", normalized.get("asset", "unknown")))),
                    "name": str(normalized.get("name", normalized.get("ticker", normalized.get("asset", "unknown")))),
                    "sector": str(normalized.get("sector", normalized.get("industry", "unknown"))),
                    "weight": weight,
                    "avg_daily_volume_weight": float(normalized.get("adv_weight", normalized.get("avg daily volume weight", 0.2)) or 0.2),
                }
            )
    return exposures


def portfolio_positions_from_evidence(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_facts = (evidence_pack or {}).get("prompt_facts", {}) or {}
    positions = prompt_facts.get("portfolio_positions")
    if isinstance(positions, list) and positions and all(isinstance(item, dict) for item in positions):
        return [dict(item) for item in positions]
    return table_positions_from_evidence(evidence_pack or {})


def returns_series_from_evidence(evidence_pack: dict[str, Any]) -> list[float]:
    prompt_facts = (evidence_pack or {}).get("prompt_facts", {}) or {}
    returns = prompt_facts.get("returns_series")
    if isinstance(returns, list):
        parsed: list[float] = []
        for item in returns:
            try:
                parsed.append(float(item))
            except Exception:
                continue
        if parsed:
            return parsed
    values: list[float] = []
    for table in (evidence_pack.get("tables", []) or []):
        rows = table.get("rows", []) if isinstance(table, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = {str(key).strip().lower(): value for key, value in row.items()}
            for key in ("return", "returns", "daily return", "monthly return"):
                if key not in normalized:
                    continue
                raw = normalized[key]
                try:
                    if isinstance(raw, str) and raw.strip().endswith("%"):
                        values.append(float(raw.strip().rstrip("%")) / 100.0)
                    else:
                        values.append(float(raw))
                except Exception:
                    pass
    return values


def limit_constraints_from_evidence(evidence_pack: dict[str, Any], policy_context: dict[str, Any]) -> dict[str, Any]:
    prompt_facts = (evidence_pack or {}).get("prompt_facts", {}) or {}
    limits = prompt_facts.get("limit_constraints")
    if isinstance(limits, dict):
        return dict(limits)
    derived: dict[str, Any] = {}
    if isinstance(policy_context.get("max_position_risk_pct"), (int, float)):
        derived["max_loss_pct"] = float(policy_context["max_position_risk_pct"]) / 100.0
    return derived


def portfolio_limit_metrics(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    var_result = latest_successful_tool_result(tool_results, {"calculate_var"})
    if var_result:
        facts = var_result.get("facts", {})
        if isinstance(facts.get("var_decimal"), (int, float)):
            metrics["var_decimal"] = float(facts["var_decimal"])
        if isinstance(facts.get("var_amount"), (int, float)):
            metrics["var_amount"] = float(facts["var_amount"])
    drawdown_result = latest_successful_tool_result(tool_results, {"drawdown_risk_profile"})
    if drawdown_result:
        facts = drawdown_result.get("facts", {})
        if isinstance(facts.get("max_drawdown_decimal"), (int, float)):
            metrics["worst_case_pnl_pct"] = -abs(float(facts["max_drawdown_decimal"]))
    factor_result = latest_successful_tool_result(tool_results, {"factor_exposure_summary"})
    if factor_result:
        facts = factor_result.get("facts", {})
        if isinstance(facts.get("largest_factor_weight"), (int, float)):
            metrics["largest_factor_weight"] = float(facts["largest_factor_weight"])
    return metrics


def deterministic_portfolio_compute_summary(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    findings: list[str] = []
    actions: list[str] = []

    concentration = latest_successful_tool_result(tool_results, {"concentration_check"})
    factor = latest_successful_tool_result(tool_results, {"factor_exposure_summary"})
    drawdown = latest_successful_tool_result(tool_results, {"drawdown_risk_profile"})
    var_result = latest_successful_tool_result(tool_results, {"calculate_var"})
    liquidity = latest_successful_tool_result(tool_results, {"liquidity_stress"})
    limits = latest_successful_tool_result(tool_results, {"portfolio_limit_check"})

    if concentration:
        facts = concentration.get("facts", {})
        if facts.get("name_breaches"):
            top = facts["name_breaches"][0]
            findings.append(f"Single-name concentration breach in {top.get('name')} at {float(top.get('weight', 0.0)) * 100:.1f}%.")
            actions.append("Trim the oversized name exposure toward the portfolio cap.")
        if facts.get("sector_breaches"):
            top = facts["sector_breaches"][0]
            findings.append(f"Sector concentration breach in {top.get('sector')} at {float(top.get('weight', 0.0)) * 100:.1f}%.")
            actions.append("Rebalance sector weight or offset it with lower-correlated positions.")

    if factor:
        facts = factor.get("facts", {})
        if facts.get("largest_factor"):
            findings.append(
                f"Largest factor exposure is {facts.get('largest_factor')} at about {float(facts.get('largest_factor_weight', 0.0)) * 100:.1f}%."
            )

    if drawdown:
        facts = drawdown.get("facts", {})
        if isinstance(facts.get("max_drawdown_decimal"), (int, float)):
            findings.append(f"Historical max drawdown is approximately {float(facts['max_drawdown_decimal']) * 100:.2f}%.")
            actions.append("Set rebalance thresholds and reduce risk if drawdown tolerance is exceeded.")

    if var_result:
        facts = var_result.get("facts", {})
        if isinstance(facts.get("var_amount"), (int, float)):
            findings.append(f"Estimated VaR is approximately {float(facts['var_amount']):.2f}.")

    if liquidity:
        facts = liquidity.get("facts", {})
        if str(facts.get("stress_assessment", "")) == "tight":
            findings.append("Liquidity stress is tight under the current redemption assumption.")
            actions.append("Stage reductions and avoid forced liquidation in the least-liquid positions.")

    if limits:
        facts = limits.get("facts", {})
        if facts.get("hard_limit_breached"):
            findings.append("At least one hard portfolio limit is breached.")
            actions.append("Do not increase risk until the limit breach is remediated.")

    if not findings:
        return None

    unique_actions = list(dict.fromkeys(actions or ["Maintain current sizing and continue monitoring key limits."]))
    summary = ["Portfolio risk review is now grounded in structured risk evidence."]
    summary.extend(findings[:4])
    summary.append("Recommended actions: " + " ".join(unique_actions[:3]))
    return " ".join(summary)


def deterministic_portfolio_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    risk_results = list(workpad.get("risk_results", []))
    if not risk_results or str(risk_results[-1].get("verdict", "")) != "pass":
        return None

    tool_results = list(workpad.get("tool_results", []))
    concentration = latest_successful_tool_result(tool_results, {"concentration_check"})
    factor = latest_successful_tool_result(tool_results, {"factor_exposure_summary"})
    drawdown = latest_successful_tool_result(tool_results, {"drawdown_risk_profile"})
    var_result = latest_successful_tool_result(tool_results, {"calculate_var"})
    liquidity = latest_successful_tool_result(tool_results, {"liquidity_stress"})
    limits = latest_successful_tool_result(tool_results, {"portfolio_limit_check"})
    risk_requirements = workpad.get("risk_requirements") or {}
    timestamp = best_available_timestamp(state)

    concentration_breach = bool(concentration and concentration.get("facts", {}).get("has_breach"))
    hard_limit_breach = bool(limits and limits.get("facts", {}).get("hard_limit_breached"))
    liquidity_tight = bool(liquidity and str(liquidity.get("facts", {}).get("stress_assessment", "")) == "tight")

    if hard_limit_breach:
        recommendation_line = "Recommendation: reduce risk before adding any new exposure because the current evidence shows a hard-limit breach."
    elif concentration_breach or liquidity_tight:
        recommendation_line = "Recommendation: de-risk or rebalance the dominant exposures first, then reassess whether incremental risk is justified."
    else:
        recommendation_line = "Recommendation: keep the core allocation only if the current risk budget is acceptable, and use hedges or rebalancing triggers instead of discretionary timing."

    lines = ["**Recommendation**", recommendation_line, "", "**Portfolio Risk Summary**"]
    if timestamp:
        lines.append(f"Source timestamp: {timestamp}.")
    if factor:
        facts = factor.get("facts", {})
        lines.append(
            f"Largest factor exposure: {facts.get('largest_factor', 'n/a')} "
            f"at about {float(facts.get('largest_factor_weight', 0.0)) * 100:.1f}%."
        )
    if concentration:
        facts = concentration.get("facts", {})
        lines.append(
            f"Concentration check: {'breach detected' if facts.get('has_breach') else 'within stated limits on the available evidence'}."
        )

    lines.extend(["", "**Stress and Scenario Evidence**"])
    if var_result:
        facts = var_result.get("facts", {})
        if isinstance(facts.get("var_amount"), (int, float)):
            lines.append(f"Estimated VaR: {float(facts['var_amount']):.2f} ({float(facts.get('var_decimal', 0.0)) * 100:.2f}% of portfolio).")
    if drawdown:
        facts = drawdown.get("facts", {})
        if isinstance(facts.get("max_drawdown_decimal"), (int, float)):
            lines.append(f"Historical max drawdown: {float(facts['max_drawdown_decimal']) * 100:.2f}%.")
    if liquidity:
        facts = liquidity.get("facts", {})
        lines.append(f"Liquidity stress: {facts.get('stress_assessment', 'n/a')}.")

    lines.extend(["", "**Limit Status**"])
    if limits:
        facts = limits.get("facts", {})
        lines.append("Hard limit breached." if facts.get("hard_limit_breached") else "No hard limit breach detected from current limit checks.")
    else:
        lines.append("No explicit hard-limit check was available.")
    if risk_requirements.get("risk_findings"):
        for finding in list(dict.fromkeys(risk_requirements.get("risk_findings", [])))[:3]:
            lines.append(f"- {finding}")

    immediate_actions: list[str] = []
    hedge_actions: list[str] = []
    monitoring_actions: list[str] = []

    if concentration and concentration.get("facts", {}).get("has_breach"):
        immediate_actions.append("Trim the breached name or sector concentration toward policy limits.")
        hedge_actions.append("Offset the dominant concentration with lower-correlated exposure or a sector hedge while reductions are staged.")
    if liquidity and str(liquidity.get("facts", {}).get("stress_assessment", "")) == "tight":
        immediate_actions.append("Stage reductions in the least-liquid positions before adding new risk.")
        monitoring_actions.append("Track liquidation-days assumptions against actual trading capacity before any rebalance.")
    if drawdown:
        hedge_actions.append("Tie rebalance and hedge actions to drawdown or VaR thresholds rather than discretionary timing.")
        monitoring_actions.append("Escalate if drawdown or VaR trends move beyond the current tolerance band.")
    if factor and factor.get("facts", {}).get("largest_factor"):
        hedge_actions.append(
            f"Reduce or hedge the dominant factor exposure in {factor.get('facts', {}).get('largest_factor', 'the main factor bucket')} rather than relying only on name-level trims."
        )
    if not any([immediate_actions, hedge_actions, monitoring_actions]):
        monitoring_actions.append("Maintain current positioning but keep the main risk indicators on a tighter monitoring cadence.")

    lines.extend(["", "**Immediate Actions**"])
    for item in list(dict.fromkeys(immediate_actions)) or ["- No immediate hard-risk intervention is required on the current evidence."]:
        lines.append(item if item.startswith("-") else f"- {item}")

    lines.extend(["", "**Hedging / Rebalance Alternatives**"])
    for item in list(dict.fromkeys(hedge_actions)) or ["- No hedge or rebalance alternative is required beyond the current allocation discipline."]:
        lines.append(item if item.startswith("-") else f"- {item}")

    lines.extend(["", "**Monitoring Triggers**"])
    for item in list(dict.fromkeys(monitoring_actions)):
        lines.append(item if item.startswith("-") else f"- {item}")

    lines.extend(["", "**Disclosures**"])
    for item in risk_requirements.get("required_disclosures", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Recommendation class: {risk_requirements.get('recommendation_class', 'scenario_dependent_recommendation')}.")
    return "\n".join(lines)
