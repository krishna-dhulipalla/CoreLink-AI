"""
Options-strategy deterministic helpers.
"""

from __future__ import annotations

from typing import Any

from agent.solver.market import (
    infer_options_market_inputs,
    latest_successful_tool_result,
    reference_price_from_tool,
)
from agent.state import AgentState


def scenario_args_from_primary_tool(tool_result: dict[str, Any]) -> dict[str, Any] | None:
    tool_type = str(tool_result.get("type", ""))
    facts = tool_result.get("facts", {}) if isinstance(tool_result, dict) else {}
    assumptions = tool_result.get("assumptions", {}) if isinstance(tool_result, dict) else {}

    if tool_type == "analyze_strategy":
        net_premium = facts.get("net_premium")
        total_delta = facts.get("total_delta")
        if not isinstance(net_premium, (int, float)) or not isinstance(total_delta, (int, float)):
            return None
        args = {
            "net_premium": float(net_premium),
            "total_delta": float(total_delta),
            "total_gamma": float(facts.get("total_gamma", 0.0) or 0.0),
            "total_theta_per_day": float(facts.get("total_theta_per_day", 0.0) or 0.0),
            "total_vega_per_vol_point": float(facts.get("total_vega_per_vol_point", 0.0) or 0.0),
        }
        reference_price = reference_price_from_tool(tool_result)
        if isinstance(reference_price, (int, float)):
            args["reference_price"] = float(reference_price)
        return args

    if tool_type in {"black_scholes_price", "mispricing_analysis"}:
        option_type = str(assumptions.get("option_type", "call")).lower()
        premium = facts.get("market_price")
        if not isinstance(premium, (int, float)):
            if option_type == "put" and isinstance(facts.get("put_price"), (int, float)):
                premium = facts.get("put_price")
            elif isinstance(facts.get("call_price"), (int, float)):
                premium = facts.get("call_price")
            elif isinstance(facts.get("theoretical_price"), (int, float)):
                premium = facts.get("theoretical_price")
        delta = facts.get("delta")
        if not isinstance(premium, (int, float)) or not isinstance(delta, (int, float)):
            return None
        args = {
            "net_premium": float(premium),
            "total_delta": float(delta),
            "total_gamma": float(facts.get("gamma", 0.0) or 0.0),
            "total_theta_per_day": float(facts.get("theta", 0.0) or 0.0),
            "total_vega_per_vol_point": float(facts.get("vega", 0.0) or 0.0),
        }
        reference_price = reference_price_from_tool(tool_result)
        if isinstance(reference_price, (int, float)):
            args["reference_price"] = float(reference_price)
        return args

    return None


def deterministic_options_compute_summary(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    latest_risk_result = (workpad.get("risk_results") or [])[-1] if workpad.get("risk_results") else {}
    if str(latest_risk_result.get("verdict", "")) == "pass":
        return None

    scenario_result = state.get("last_tool_result") or {}
    if str(scenario_result.get("type", "")) != "scenario_pnl" or scenario_result.get("errors"):
        return None

    primary_tool = latest_successful_tool_result(
        tool_results,
        {"analyze_strategy", "black_scholes_price", "option_greeks", "mispricing_analysis"},
    )
    primary_facts = primary_tool.get("facts", {}) if isinstance(primary_tool, dict) else {}
    scenario_facts = scenario_result.get("facts", {}) if isinstance(scenario_result, dict) else {}
    scenario_assumptions = scenario_result.get("assumptions", {}) if isinstance(scenario_result, dict) else {}

    max_loss = primary_facts.get("max_loss")
    delta = primary_facts.get("total_delta", primary_facts.get("delta"))
    gamma = primary_facts.get("total_gamma", primary_facts.get("gamma"))
    theta = primary_facts.get("total_theta_per_day", primary_facts.get("theta"))
    vega = primary_facts.get("total_vega_per_vol_point", primary_facts.get("vega"))
    worst_case_pnl = scenario_facts.get("worst_case_pnl")
    best_case_pnl = scenario_facts.get("best_case_pnl")

    summary_lines = ["Primary risk summary is now tool-backed and ready for review."]
    greeks_bits: list[str] = []
    if isinstance(delta, (int, float)):
        greeks_bits.append(f"delta {float(delta):.3f}")
    if isinstance(gamma, (int, float)):
        greeks_bits.append(f"gamma {float(gamma):.3f}")
    if isinstance(theta, (int, float)):
        greeks_bits.append(f"theta {float(theta):.3f}/day")
    if isinstance(vega, (int, float)):
        greeks_bits.append(f"vega {float(vega):.3f} per vol point")
    if greeks_bits:
        summary_lines.append("Key Greeks: " + ", ".join(greeks_bits) + ".")
    if isinstance(max_loss, (int, float)):
        summary_lines.append(f"Max loss is approximately {float(max_loss):.2f}.")
    if isinstance(worst_case_pnl, (int, float)):
        summary_lines.append(f"Downside scenario loss is approximately {float(worst_case_pnl):.2f}.")
    if isinstance(best_case_pnl, (int, float)):
        summary_lines.append(f"Base or favorable scenario P&L is approximately {float(best_case_pnl):.2f}.")

    summary_lines.append(
        "Risk controls: use 1-2% position sizing, place a stop-loss near a 1x premium loss or a breakeven breach, and hedge or reduce exposure if delta or gamma expands materially."
    )

    reference_price = scenario_assumptions.get("reference_price")
    if isinstance(reference_price, (int, float)):
        summary_lines.append(f"Reference spot for the scenario grid is {float(reference_price):.2f}.")

    return " ".join(summary_lines)


def infer_options_strategy_label(primary_tool: dict[str, Any]) -> str:
    assumptions = primary_tool.get("assumptions", {}) if isinstance(primary_tool, dict) else {}
    legs = assumptions.get("legs")
    if isinstance(legs, list) and legs:
        short_calls: list[float] = []
        short_puts: list[float] = []
        long_calls = 0
        long_puts = 0
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            option_type = str(leg.get("option_type", "")).lower()
            action = str(leg.get("action", "")).lower()
            strike = leg.get("K")
            if action == "sell" and option_type == "call" and isinstance(strike, (int, float)):
                short_calls.append(float(strike))
            elif action == "sell" and option_type == "put" and isinstance(strike, (int, float)):
                short_puts.append(float(strike))
            elif action == "buy" and option_type == "call":
                long_calls += 1
            elif action == "buy" and option_type == "put":
                long_puts += 1
        if short_calls and short_puts and long_calls and long_puts:
            return "iron condor"
        if short_calls and short_puts:
            if len(short_calls) == 1 and len(short_puts) == 1 and abs(short_calls[0] - short_puts[0]) < 1e-9:
                return "short straddle"
            return "short strangle"
    if str(primary_tool.get("type", "")) == "black_scholes_price":
        return "single-option premium sale"
    return "options premium strategy"


def infer_breakeven_text(primary_tool: dict[str, Any]) -> str:
    facts = primary_tool.get("facts", {}) if isinstance(primary_tool, dict) else {}
    assumptions = primary_tool.get("assumptions", {}) if isinstance(primary_tool, dict) else {}
    if isinstance(facts.get("breakeven"), (int, float)):
        return f"{float(facts['breakeven']):.2f}"
    net_premium = facts.get("net_premium")
    legs = assumptions.get("legs")
    if isinstance(net_premium, (int, float)) and isinstance(legs, list):
        short_calls: list[float] = []
        short_puts: list[float] = []
        for leg in legs:
            if not isinstance(leg, dict) or str(leg.get("action", "")).lower() != "sell":
                continue
            strike = leg.get("K")
            if not isinstance(strike, (int, float)):
                continue
            option_type = str(leg.get("option_type", "")).lower()
            if option_type == "call":
                short_calls.append(float(strike))
            elif option_type == "put":
                short_puts.append(float(strike))
        if short_calls and short_puts:
            lower = min(short_puts) - float(net_premium)
            upper = max(short_calls) + float(net_premium)
            return f"{lower:.2f} / {upper:.2f}"
    return "manage around the short strikes adjusted by collected premium"


def primary_tool_is_policy_compliant(primary_tool: dict[str, Any], policy_context: dict[str, Any]) -> bool:
    if not primary_tool:
        return False
    strategy_label = infer_options_strategy_label(primary_tool)
    assumptions = primary_tool.get("assumptions", {}) if isinstance(primary_tool, dict) else {}
    legs = assumptions.get("legs")
    if policy_context.get("defined_risk_only"):
        if strategy_label == "iron condor":
            pass
        elif isinstance(legs, list):
            has_buy = any(isinstance(leg, dict) and str(leg.get("action", "")).lower() == "buy" for leg in legs)
            has_sell = any(isinstance(leg, dict) and str(leg.get("action", "")).lower() == "sell" for leg in legs)
            if not (has_buy and has_sell):
                return False
        else:
            return False
    if policy_context.get("no_naked_options"):
        normalized_label = strategy_label.lower()
        if normalized_label in {"short straddle", "short strangle", "single-option premium sale"}:
            return False
        if isinstance(legs, list):
            has_long_call = any(isinstance(leg, dict) and str(leg.get("action", "")).lower() == "buy" and str(leg.get("option_type", "")).lower() == "call" for leg in legs)
            has_long_put = any(isinstance(leg, dict) and str(leg.get("action", "")).lower() == "buy" and str(leg.get("option_type", "")).lower() == "put" for leg in legs)
            short_calls = any(isinstance(leg, dict) and str(leg.get("action", "")).lower() == "sell" and str(leg.get("option_type", "")).lower() == "call" for leg in legs)
            short_puts = any(isinstance(leg, dict) and str(leg.get("action", "")).lower() == "sell" and str(leg.get("option_type", "")).lower() == "put" for leg in legs)
            if short_calls and not has_long_call:
                return False
            if short_puts and not has_long_put:
                return False
    return True


def deterministic_policy_options_tool_call(state: AgentState) -> dict[str, Any] | None:
    evidence_pack = state.get("evidence_pack", {}) or {}
    derived_facts = evidence_pack.get("derived_facts", {}) or {}
    policy_context = evidence_pack.get("policy_context", {}) or {}
    if not (policy_context.get("defined_risk_only") or policy_context.get("no_naked_options")):
        return None

    spot, sigma, r, t_days = infer_options_market_inputs(state)
    width = max(5.0, round(spot * 0.03 / 5.0) * 5.0)
    inner = max(5.0, round(spot * 0.015 / 5.0) * 5.0)
    vol_bias = str(derived_facts.get("vol_bias", "short_vol"))

    if vol_bias == "short_vol":
        legs = [
            {"option_type": "put", "action": "buy", "S": spot, "K": spot - inner - width, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "put", "action": "sell", "S": spot, "K": spot - inner, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "call", "action": "sell", "S": spot, "K": spot + inner, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "call", "action": "buy", "S": spot, "K": spot + inner + width, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
        ]
    else:
        legs = [
            {"option_type": "call", "action": "buy", "S": spot, "K": spot, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "call", "action": "sell", "S": spot, "K": spot + width, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
        ]
    return {"name": "analyze_strategy", "arguments": {"legs": legs}}


def deterministic_standard_options_tool_call(state: AgentState) -> dict[str, Any] | None:
    evidence_pack = state.get("evidence_pack", {}) or {}
    derived_facts = evidence_pack.get("derived_facts", {}) or {}

    spot, sigma, r, t_days = infer_options_market_inputs(state)
    vol_bias = str(derived_facts.get("vol_bias", "short_vol"))
    atm = round(spot / 5.0) * 5.0

    if vol_bias == "short_vol":
        legs = [
            {"option_type": "call", "action": "sell", "S": spot, "K": atm, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "put", "action": "sell", "S": spot, "K": atm, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
        ]
    else:
        legs = [
            {"option_type": "call", "action": "buy", "S": spot, "K": atm, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "put", "action": "buy", "S": spot, "K": atm, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
        ]
    return {"name": "analyze_strategy", "arguments": {"legs": legs}}


def deterministic_options_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    risk_results = list(workpad.get("risk_results", []))
    if not risk_results or str(risk_results[-1].get("verdict", "")) != "pass":
        return None

    primary_tool = latest_successful_tool_result(
        list(workpad.get("tool_results", [])),
        {"analyze_strategy", "black_scholes_price", "option_greeks", "mispricing_analysis"},
    )
    scenario_result = latest_successful_tool_result(
        list(workpad.get("tool_results", [])),
        {"scenario_pnl", "run_stress_test", "calculate_var", "portfolio_limit_check", "concentration_check", "calculate_portfolio_greeks"},
    )
    if primary_tool is None or scenario_result is None:
        return None

    primary_facts = primary_tool.get("facts", {}) if isinstance(primary_tool, dict) else {}
    derived_facts = (state.get("evidence_pack", {}) or {}).get("derived_facts", {}) or {}
    risk_requirements = workpad.get("risk_requirements") or {}
    strategy_label = infer_options_strategy_label(primary_tool)
    recommendation = "net seller of options" if derived_facts.get("vol_bias") == "short_vol" else "options seller with scenario-dependent conviction"
    net_premium = primary_facts.get("net_premium")
    premium_direction = str(primary_facts.get("premium_direction", "credit")).upper()
    delta = primary_facts.get("total_delta", primary_facts.get("delta"))
    gamma = primary_facts.get("total_gamma", primary_facts.get("gamma"))
    theta = primary_facts.get("total_theta_per_day", primary_facts.get("theta"))
    vega = primary_facts.get("total_vega_per_vol_point", primary_facts.get("vega"))
    max_loss = primary_facts.get("max_loss")
    breakevens = infer_breakeven_text(primary_tool)
    scenario_facts = scenario_result.get("facts", {}) if isinstance(scenario_result, dict) else {}
    worst_case_pnl = scenario_facts.get("worst_case_pnl")
    best_case_pnl = scenario_facts.get("best_case_pnl")
    reference_price = reference_price_from_tool(primary_tool)

    alternative = "iron condor with defined wings" if strategy_label in {"short straddle", "short strangle"} else "defined-risk spread"
    tradeoff = "lower premium but better tail-risk control" if alternative == "iron condor with defined wings" else "more controlled downside at the cost of smaller carry"

    disclosures: list[str] = []
    for item in risk_requirements.get("required_disclosures", []):
        normalized = str(item).lower()
        if "short-volatility" in normalized or "volatility-spike" in normalized:
            disclosures.append("Short-volatility / volatility-spike risk is material: losses can accelerate if implied volatility expands.")
        elif "tail loss" in normalized or "gap risk" in normalized or "unbounded" in normalized:
            disclosures.append("Tail loss and gap risk are material, especially if the underlying gaps through the short strikes.")
        elif "downside scenario loss" in normalized:
            if isinstance(worst_case_pnl, (int, float)):
                disclosures.append(f"Downside scenario loss is approximately {float(worst_case_pnl):.2f}; the exit / sizing response is to keep exposure at 1-2% of capital and cut risk on a breach.")
            else:
                disclosures.append("Downside scenario loss should be treated as the hard sizing and exit reference.")
    if not disclosures and isinstance(worst_case_pnl, (int, float)):
        disclosures.append(f"Downside scenario loss is approximately {float(worst_case_pnl):.2f}.")

    assumption_lines: list[str] = []
    for record in state.get("assumption_ledger", []):
        if not isinstance(record, dict) or not record.get("requires_user_visible_disclosure"):
            continue
        assumption_lines.append(str(record.get("assumption", "")))

    lines = ["**Recommendation**", f"Be a {recommendation}.", "", "**Primary Strategy**", f"{strategy_label.title()} with {premium_direction.lower()} premium" + (f" of {float(net_premium):.2f}." if isinstance(net_premium, (int, float)) else "."), "", "**Alternative Strategy Comparison**", f"{alternative.title()} is the cleaner alternative when you want {tradeoff}.", "", "**Key Greeks and Breakevens**"]
    greeks_line = []
    if isinstance(delta, (int, float)):
        greeks_line.append(f"delta {float(delta):.3f}")
    if isinstance(gamma, (int, float)):
        greeks_line.append(f"gamma {float(gamma):.3f}")
    if isinstance(theta, (int, float)):
        greeks_line.append(f"theta {float(theta):.3f}/day")
    if isinstance(vega, (int, float)):
        greeks_line.append(f"vega {float(vega):.3f} per vol point")
    if greeks_line:
        lines.append(", ".join(greeks_line) + ".")
    lines.append(f"Breakevens: {breakevens}.")
    if isinstance(max_loss, (int, float)):
        lines.append(f"Max loss reference: {float(max_loss):.2f}.")
    lines.extend(["", "**Risk Management**", "Use 1-2% position sizing, predefine a stop-loss at a breakeven breach or roughly a 1x premium loss, and hedge or reduce exposure if delta/gamma expands."])
    if isinstance(best_case_pnl, (int, float)) or isinstance(worst_case_pnl, (int, float)):
        parts = []
        if isinstance(best_case_pnl, (int, float)):
            parts.append(f"base-case P&L about {float(best_case_pnl):.2f}")
        if isinstance(worst_case_pnl, (int, float)):
            parts.append(f"stress downside about {float(worst_case_pnl):.2f}")
        lines.append("Scenario summary: " + "; ".join(parts) + ".")
    lines.extend(["", "**Disclosures**"])
    if assumption_lines:
        for item in assumption_lines[:3]:
            lines.append(f"- Assumption: {item}")
    elif isinstance(reference_price, (int, float)):
        lines.append(f"- Assumption: spot/reference price treated as {float(reference_price):.2f}.")
    for item in disclosures:
        lines.append(f"- {item}")
    if risk_requirements.get("recommendation_class"):
        lines.append("")
        lines.append(f"Recommendation class: {risk_requirements.get('recommendation_class')}.")

    return "\n".join(lines)


def deterministic_policy_options_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    risk_results = list(workpad.get("risk_results", []))
    if not risk_results or str(risk_results[-1].get("verdict", "")) != "pass":
        return None

    policy_context = (state.get("evidence_pack", {}) or {}).get("policy_context", {}) or {}
    primary_tool = latest_successful_tool_result(
        list(workpad.get("tool_results", [])),
        {"analyze_strategy", "black_scholes_price", "option_greeks", "mispricing_analysis"},
    )
    scenario_result = latest_successful_tool_result(
        list(workpad.get("tool_results", [])),
        {"scenario_pnl", "run_stress_test", "calculate_var", "portfolio_limit_check", "concentration_check", "calculate_portfolio_greeks"},
    )
    if primary_tool is None or scenario_result is None:
        return None
    if not primary_tool_is_policy_compliant(primary_tool, policy_context):
        return None

    primary_facts = primary_tool.get("facts", {}) if isinstance(primary_tool, dict) else {}
    scenario_facts = scenario_result.get("facts", {}) if isinstance(scenario_result, dict) else {}
    risk_requirements = workpad.get("risk_requirements") or {}
    strategy_label = infer_options_strategy_label(primary_tool)
    short_vol = str((state.get("evidence_pack", {}) or {}).get("derived_facts", {}).get("vol_bias", "")) == "short_vol"
    recommendation = (
        "Be a net seller of options through a defined-risk structure"
        if short_vol
        else "Be a net buyer of options through a defined-risk spread"
    )
    alternative = "put credit spread" if strategy_label == "iron condor" else "iron condor"
    tradeoff = "retains short-vol carry with simpler one-sided management" if alternative == "put credit spread" else "diversifies tail exposure across both sides at the cost of more moving parts"
    net_premium = primary_facts.get("net_premium")
    premium_direction = str(primary_facts.get("premium_direction", "credit")).upper()
    delta = primary_facts.get("total_delta", primary_facts.get("delta"))
    gamma = primary_facts.get("total_gamma", primary_facts.get("gamma"))
    theta = primary_facts.get("total_theta_per_day", primary_facts.get("theta"))
    vega = primary_facts.get("total_vega_per_vol_point", primary_facts.get("vega"))
    breakevens = infer_breakeven_text(primary_tool)
    worst_case_pnl = scenario_facts.get("worst_case_pnl")
    best_case_pnl = scenario_facts.get("best_case_pnl")
    risk_cap = policy_context.get("max_position_risk_pct")
    disclosures: list[str] = []
    for item in risk_requirements.get("required_disclosures", []):
        normalized = str(item).lower()
        if "short-volatility" in normalized or "volatility-spike" in normalized:
            disclosures.append("Short-volatility / volatility-spike risk is still material even with defined wings.")
        elif "tail loss" in normalized or "gap risk" in normalized or "unbounded" in normalized:
            disclosures.append("Tail loss is capped by the long wings, but gap risk into the short strikes still matters.")
        elif "downside scenario loss" in normalized:
            if isinstance(worst_case_pnl, (int, float)):
                disclosures.append(f"Stress downside scenario loss is approximately {float(worst_case_pnl):.2f}, which defines the sizing and exit response.")
        elif "max loss" in normalized:
            max_loss = primary_facts.get("max_loss")
            if isinstance(max_loss, (int, float)):
                disclosures.append(f"Max loss reference is approximately {float(max_loss):.2f}.")

    lines = ["**Recommendation**", f"{recommendation}. This mandate requires defined-risk only and prohibits naked options.", "", "**Primary Strategy**", f"Defined-risk {strategy_label} with {premium_direction.lower()} premium" + (f" of {float(net_premium):.2f}." if isinstance(net_premium, (int, float)) else "."), "", "**Alternative Strategy Comparison**", f"{alternative.title()} is the cleaner backup when you want {tradeoff}.", "", "**Key Greeks and Breakevens**"]
    greeks_line = []
    if isinstance(delta, (int, float)):
        greeks_line.append(f"delta {float(delta):.3f}")
    if isinstance(gamma, (int, float)):
        greeks_line.append(f"gamma {float(gamma):.3f}")
    if isinstance(theta, (int, float)):
        greeks_line.append(f"theta {float(theta):.3f}/day")
    if isinstance(vega, (int, float)):
        greeks_line.append(f"vega {float(vega):.3f} per vol point")
    if greeks_line:
        lines.append(", ".join(greeks_line) + ".")
    lines.append(f"Breakevens: {breakevens}.")
    lines.extend(["", "**Risk Management**", (f"Cap position risk at about {float(risk_cap):g}% of capital, use defined exit points near the short strikes or at roughly a 1x premium-loss threshold, and reduce exposure if gamma or vol expands sharply.") if isinstance(risk_cap, (int, float)) else "Use defined exit points near the short strikes or at roughly a 1x premium-loss threshold, and reduce exposure if gamma or vol expands sharply."])
    if isinstance(best_case_pnl, (int, float)) or isinstance(worst_case_pnl, (int, float)):
        parts = []
        if isinstance(best_case_pnl, (int, float)):
            parts.append(f"base-case P&L about {float(best_case_pnl):.2f}")
        if isinstance(worst_case_pnl, (int, float)):
            parts.append(f"stress downside about {float(worst_case_pnl):.2f}")
        lines.append("Scenario summary: " + "; ".join(parts) + ".")
    lines.extend(["", "**Disclosures**"])
    lines.append("- Mandate: defined-risk only; naked options are not permitted in this account.")
    if isinstance(risk_cap, (int, float)):
        lines.append(f"- Position-risk cap: keep exposure around {float(risk_cap):g}% of capital or lower.")
    for item in disclosures:
        lines.append(f"- {item}")
    if isinstance(worst_case_pnl, (int, float)) and not disclosures:
        lines.append(f"- Stress downside scenario loss is approximately {float(worst_case_pnl):.2f}.")
    for record in state.get("assumption_ledger", []):
        if isinstance(record, dict) and record.get("requires_user_visible_disclosure"):
            lines.append(f"- Assumption: {record.get('assumption', '')}")
    lines.append("")
    lines.append(f"Recommendation class: {risk_requirements.get('recommendation_class', 'scenario_dependent_recommendation')}.")
    return "\n".join(lines)
