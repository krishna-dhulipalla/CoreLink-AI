"""
Risk Metrics MCP Server
=======================
Provides portfolio and scenario-oriented risk analytics in structured envelopes.
"""

from __future__ import annotations

import math
import random
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Risk Metrics")


def _envelope(
    tool_name: str,
    *,
    facts: dict[str, Any] | None = None,
    assumptions: dict[str, Any] | None = None,
    provider: str = "local_risk",
    missing_fields: list[str] | None = None,
    is_estimated: bool = False,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": tool_name,
        "facts": facts or {},
        "assumptions": assumptions or {},
        "source": {"tool": tool_name, "provider": provider},
        "quality": {
            "is_synthetic": False,
            "is_estimated": is_estimated,
            "cache_hit": False,
            "missing_fields": list(missing_fields or []),
        },
        "errors": list(errors or []),
    }


def _error(tool_name: str, message: str, *, assumptions: dict[str, Any] | None = None) -> dict[str, Any]:
    return _envelope(tool_name, assumptions=assumptions, errors=[message])


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_inv_approx(p: float) -> float:
    if p < 0.5:
        return -_norm_inv_approx(1 - p)
    q = math.sqrt(-2 * math.log(1 - p))
    a = [2.515517, 0.802853, 0.010328]
    b = [1.432788, 0.189269, 0.001308]
    num = a[0] + a[1] * q + a[2] * q**2
    den = 1 + b[0] * q + b[1] * q**2 + b[2] * q**3
    return q - num / den


def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    e_rT = math.exp(-r * T)
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * _norm_pdf(d1) * math.sqrt(T) / 100.0

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = (-S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * e_rT * _norm_cdf(d2)) / 365.0
        rho = K * T * e_rT * _norm_cdf(d2) / 100.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * e_rT * _norm_cdf(-d2)) / 365.0
        rho = -K * T * e_rT * _norm_cdf(-d2) / 100.0
    return delta, gamma, theta, vega, rho


def _default_scenarios() -> list[dict[str, Any]]:
    return [
        {"name": "base", "spot_pct": 0.0, "vol_point_change": 0.0, "days": 1},
        {"name": "upside", "spot_pct": 0.05, "vol_point_change": -0.03, "days": 5},
        {"name": "downside", "spot_pct": -0.05, "vol_point_change": 0.04, "days": 5},
        {"name": "stress", "spot_pct": -0.12, "vol_point_change": 0.12, "days": 1},
    ]


@mcp.tool()
def calculate_portfolio_greeks(positions: list[dict]) -> dict[str, Any]:
    assumptions = {"position_count": len(positions or [])}
    try:
        total = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        for pos in positions:
            S = float(pos["S"])
            K = float(pos["K"])
            T = float(pos["T_days"]) / 365.0
            r = float(pos["r"])
            sigma = float(pos["sigma"])
            contracts = int(pos.get("contracts", 1)) * 100
            sign = 1 if str(pos.get("action", "buy")).lower() == "buy" else -1
            d, g, th, v, rh = _bs_greeks(S, K, T, r, sigma, str(pos.get("option_type", "call")))
            total["delta"] += d * contracts * sign
            total["gamma"] += g * contracts * sign
            total["theta"] += th * contracts * sign
            total["vega"] += v * contracts * sign
            total["rho"] += rh * contracts * sign

        facts = {
            "position_count": len(positions),
            "total_delta": round(total["delta"], 6),
            "total_gamma": round(total["gamma"], 6),
            "total_theta_per_day": round(total["theta"], 6),
            "total_vega_per_vol_point": round(total["vega"], 6),
            "total_rho_per_rate_point": round(total["rho"], 6),
            "directional_bias": "bullish" if total["delta"] > 0 else "bearish" if total["delta"] < 0 else "neutral",
            "time_decay_profile": "positive" if total["theta"] > 0 else "negative" if total["theta"] < 0 else "flat",
            "volatility_exposure": "long_vol" if total["vega"] > 0 else "short_vol" if total["vega"] < 0 else "flat",
        }
        return _envelope("calculate_portfolio_greeks", facts=facts, assumptions=assumptions)
    except Exception as exc:
        return _error("calculate_portfolio_greeks", f"Error aggregating portfolio Greeks: {exc}", assumptions=assumptions)


@mcp.tool()
def calculate_var(
    portfolio_value: float,
    daily_vol: float,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
    method: str = "parametric",
) -> dict[str, Any]:
    assumptions = {
        "portfolio_value": portfolio_value,
        "daily_vol": daily_vol,
        "confidence_level": confidence_level,
        "holding_period_days": holding_period_days,
        "method": method,
    }
    try:
        if portfolio_value <= 0:
            return _error("calculate_var", "portfolio_value must be positive.", assumptions=assumptions)
        if daily_vol < 0:
            return _error("calculate_var", "daily_vol must be non-negative.", assumptions=assumptions)

        t_sqrt = math.sqrt(holding_period_days)
        z = _norm_inv_approx(confidence_level)
        if method.lower() == "monte_carlo":
            random.seed(42)
            returns = sorted(random.gauss(0, daily_vol * t_sqrt) for _ in range(10_000))
            cutoff_idx = int((1 - confidence_level) * len(returns))
            pct_var = -returns[cutoff_idx]
            method_label = "monte_carlo"
            is_estimated = True
        else:
            pct_var = z * daily_vol * t_sqrt
            method_label = "parametric"
            is_estimated = False

        cvar_pct = pct_var * 1.15
        facts = {
            "portfolio_value": float(portfolio_value),
            "var_decimal": float(pct_var),
            "var_amount": float(pct_var * portfolio_value),
            "cvar_decimal": float(cvar_pct),
            "cvar_amount": float(cvar_pct * portfolio_value),
            "confidence_level": float(confidence_level),
            "holding_period_days": int(holding_period_days),
            "method": method_label,
        }
        return _envelope("calculate_var", facts=facts, assumptions=assumptions, is_estimated=is_estimated)
    except Exception as exc:
        return _error("calculate_var", f"Error computing VaR: {exc}", assumptions=assumptions)


@mcp.tool()
def calculate_risk_metrics(returns: list[float], risk_free_rate: float = 0.05) -> dict[str, Any]:
    assumptions = {"risk_free_rate": risk_free_rate, "period_count": len(returns or [])}
    try:
        if len(returns) < 2:
            return _error("calculate_risk_metrics", "Provide at least 2 return observations.", assumptions=assumptions)

        n = len(returns)
        mean_r = sum(returns) / n
        variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_r = math.sqrt(variance)
        ann_return = (1 + mean_r) ** 252 - 1
        ann_vol = std_r * math.sqrt(252)
        daily_rf = risk_free_rate / 252
        sharpe = (mean_r - daily_rf) / std_r * math.sqrt(252) if std_r > 0 else 0.0
        downside = [r for r in returns if r < daily_rf]
        down_std = math.sqrt(sum((r - daily_rf) ** 2 for r in downside) / max(len(downside), 1))
        sortino = (mean_r - daily_rf) / down_std * math.sqrt(252) if down_std > 0 else 0.0

        peak = cum = 1.0
        max_dd = 0.0
        for ret in returns:
            cum *= (1 + ret)
            peak = max(peak, cum)
            max_dd = max(max_dd, (peak - cum) / peak)

        calmar = ann_return / max_dd if max_dd > 0 else float("inf")
        assessment = "strong" if sharpe > 1.5 else "acceptable" if sharpe > 0.5 else "weak"
        facts = {
            "annualized_return_decimal": float(ann_return),
            "annualized_volatility_decimal": float(ann_vol),
            "max_drawdown_decimal": float(max_dd),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar) if math.isfinite(calmar) else None,
            "assessment": assessment,
        }
        return _envelope("calculate_risk_metrics", facts=facts, assumptions=assumptions, is_estimated=True)
    except Exception as exc:
        return _error("calculate_risk_metrics", f"Error computing risk metrics: {exc}", assumptions=assumptions)


@mcp.tool()
def run_stress_test(
    portfolio_value: float,
    portfolio_delta: float,
    portfolio_vega: float,
    portfolio_theta: float,
) -> dict[str, Any]:
    assumptions = {
        "portfolio_value": portfolio_value,
        "portfolio_delta": portfolio_delta,
        "portfolio_vega": portfolio_vega,
        "portfolio_theta": portfolio_theta,
    }
    try:
        if portfolio_value <= 0:
            return _error("run_stress_test", "portfolio_value must be positive.", assumptions=assumptions)

        scenarios = [
            {"name": "market_up_5", "spot_move": 0.05, "vol_move": -0.02, "days": 1},
            {"name": "market_down_5", "spot_move": -0.05, "vol_move": 0.03, "days": 1},
            {"name": "vol_spike_20", "spot_move": 0.0, "vol_move": 0.20, "days": 1},
            {"name": "one_week_decay", "spot_move": 0.0, "vol_move": 0.0, "days": 7},
            {"name": "crash_20", "spot_move": -0.20, "vol_move": 0.15, "days": 1},
        ]
        rows = []
        for scenario in scenarios:
            spot_pnl = portfolio_delta * scenario["spot_move"] * portfolio_value / 100.0
            vega_pnl = portfolio_vega * scenario["vol_move"] * 100.0
            theta_pnl = portfolio_theta * scenario["days"]
            total_pnl = spot_pnl + vega_pnl + theta_pnl
            rows.append(
                {
                    "name": scenario["name"],
                    "spot_move": scenario["spot_move"],
                    "vol_move": scenario["vol_move"],
                    "days": scenario["days"],
                    "approx_pnl": round(total_pnl, 6),
                    "approx_pnl_pct": round(total_pnl / portfolio_value, 6),
                }
            )

        worst_case = min(rows, key=lambda item: item["approx_pnl"])
        facts = {
            "scenarios": rows,
            "worst_case_pnl": worst_case["approx_pnl"],
            "worst_case_pnl_pct": worst_case["approx_pnl_pct"],
            "stress_scenario": worst_case["name"],
        }
        return _envelope("run_stress_test", facts=facts, assumptions=assumptions, is_estimated=True)
    except Exception as exc:
        return _error("run_stress_test", f"Error running stress test: {exc}", assumptions=assumptions)


@mcp.tool()
def calculate_max_drawdown(returns: list[float]) -> dict[str, Any]:
    assumptions = {"period_count": len(returns or [])}
    try:
        if not returns:
            return _error("calculate_max_drawdown", "Empty returns list.", assumptions=assumptions)

        peak = cum = 1.0
        max_dd = 0.0
        dd_start = dd_end = 0
        current_start = 0
        for idx, ret in enumerate(returns):
            cum *= (1 + ret)
            if cum >= peak:
                peak = cum
                current_start = idx
            else:
                dd = (peak - cum) / peak
                if dd > max_dd:
                    max_dd = dd
                    dd_start = current_start
                    dd_end = idx

        facts = {
            "max_drawdown_decimal": float(max_dd),
            "drawdown_start_index": dd_start,
            "drawdown_end_index": dd_end,
            "drawdown_duration_periods": dd_end - dd_start,
            "recovery_periods": len(returns) - dd_end if max_dd > 0 else 0,
            "total_return_decimal": float(cum - 1),
        }
        return _envelope("calculate_max_drawdown", facts=facts, assumptions=assumptions)
    except Exception as exc:
        return _error("calculate_max_drawdown", f"Error computing max drawdown: {exc}", assumptions=assumptions)


@mcp.tool()
def scenario_pnl(
    net_premium: float,
    total_delta: float,
    total_gamma: float = 0.0,
    total_theta_per_day: float = 0.0,
    total_vega_per_vol_point: float = 0.0,
    reference_price: float = 100.0,
    scenarios: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assumptions = {
        "net_premium": net_premium,
        "total_delta": total_delta,
        "total_gamma": total_gamma,
        "total_theta_per_day": total_theta_per_day,
        "total_vega_per_vol_point": total_vega_per_vol_point,
        "reference_price": reference_price,
    }
    try:
        scenario_rows = scenarios or _default_scenarios()
        rows = []
        for scenario in scenario_rows:
            spot_pct = float(scenario.get("spot_pct", 0.0))
            vol_points = float(scenario.get("vol_point_change", 0.0))
            days = int(scenario.get("days", 1))
            price_move = reference_price * spot_pct
            pnl = (
                float(total_delta) * price_move
                + 0.5 * float(total_gamma) * (price_move ** 2)
                + float(total_vega_per_vol_point) * (vol_points * 100.0)
                + float(total_theta_per_day) * days
            )
            rows.append(
                {
                    "name": str(scenario.get("name", "scenario")),
                    "spot_pct": spot_pct,
                    "vol_point_change": vol_points,
                    "days": days,
                    "approx_pnl": round(pnl, 6),
                }
            )

        worst_case = min(rows, key=lambda item: item["approx_pnl"])
        best_case = max(rows, key=lambda item: item["approx_pnl"])
        facts = {
            "scenarios": rows,
            "worst_case_pnl": worst_case["approx_pnl"],
            "worst_case_scenario": worst_case["name"],
            "best_case_pnl": best_case["approx_pnl"],
            "best_case_scenario": best_case["name"],
            "stress_loss_ratio": round(abs(worst_case["approx_pnl"]) / max(abs(net_premium), 1e-6), 6),
        }
        return _envelope("scenario_pnl", facts=facts, assumptions=assumptions, is_estimated=True)
    except Exception as exc:
        return _error("scenario_pnl", f"Error computing scenario P&L: {exc}", assumptions=assumptions)


@mcp.tool()
def concentration_check(
    exposures: list[dict[str, Any]],
    max_single_name_weight: float = 0.20,
    max_single_sector_weight: float = 0.35,
) -> dict[str, Any]:
    assumptions = {
        "max_single_name_weight": max_single_name_weight,
        "max_single_sector_weight": max_single_sector_weight,
    }
    try:
        total_weight = sum(float(item.get("weight", 0.0)) for item in exposures)
        name_breaches = []
        sector_weights: dict[str, float] = {}
        for item in exposures:
            weight = float(item.get("weight", 0.0))
            name = str(item.get("name", item.get("ticker", "unknown")))
            sector = str(item.get("sector", "unknown"))
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
            if weight > max_single_name_weight:
                name_breaches.append({"name": name, "weight": weight})

        sector_breaches = [
            {"sector": sector, "weight": weight}
            for sector, weight in sector_weights.items()
            if weight > max_single_sector_weight
        ]
        facts = {
            "total_weight": total_weight,
            "name_breaches": name_breaches,
            "sector_breaches": sector_breaches,
            "has_breach": bool(name_breaches or sector_breaches),
        }
        return _envelope("concentration_check", facts=facts, assumptions=assumptions)
    except Exception as exc:
        return _error("concentration_check", f"Error computing concentration check: {exc}", assumptions=assumptions)


@mcp.tool()
def portfolio_limit_check(metrics: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    assumptions = {"limits": limits}
    try:
        breaches = []
        warnings = []
        metric_value = float(metrics.get("worst_case_pnl_pct", metrics.get("var_decimal", 0.0)) or 0.0)
        max_loss_limit = limits.get("max_loss_pct")
        if max_loss_limit is not None and abs(metric_value) > float(max_loss_limit):
            breaches.append(
                {
                    "code": "LIMIT_BREACH_MAX_LOSS_PCT",
                    "metric": "worst_case_pnl_pct",
                    "observed": metric_value,
                    "limit": float(max_loss_limit),
                }
            )

        var_limit = limits.get("max_var_pct")
        var_value = metrics.get("var_decimal")
        if var_limit is not None and var_value is not None and float(var_value) > float(var_limit):
            breaches.append(
                {
                    "code": "LIMIT_BREACH_VAR_PCT",
                    "metric": "var_decimal",
                    "observed": float(var_value),
                    "limit": float(var_limit),
                }
            )

        delta_limit = limits.get("max_abs_delta")
        delta_value = metrics.get("total_delta")
        if delta_limit is not None and delta_value is not None and abs(float(delta_value)) > float(delta_limit):
            warnings.append(
                {
                    "code": "WARNING_DIRECTIONAL_DELTA",
                    "metric": "total_delta",
                    "observed": float(delta_value),
                    "limit": float(delta_limit),
                }
            )

        facts = {
            "hard_limit_breached": bool(breaches),
            "breaches": breaches,
            "warnings": warnings,
        }
        return _envelope("portfolio_limit_check", facts=facts, assumptions=assumptions)
    except Exception as exc:
        return _error("portfolio_limit_check", f"Error running portfolio limit check: {exc}", assumptions=assumptions)


@mcp.tool()
def factor_exposure_summary(
    exposures: list[dict[str, Any]],
    factor_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    assumptions = {"factor_map": factor_map or {}, "position_count": len(exposures or [])}
    try:
        factors = factor_map or {
            "technology": ["software", "semiconductor", "internet"],
            "financials": ["bank", "insurance", "financial"],
            "energy": ["energy", "oil", "gas"],
            "defensive": ["healthcare", "utilities", "consumer staples"],
        }
        totals: dict[str, float] = {}
        unexplained = 0.0
        for item in exposures:
            weight = float(item.get("weight", 0.0))
            sector_text = str(item.get("sector", "")).lower()
            matched = False
            for factor, sector_tokens in factors.items():
                if any(token in sector_text for token in sector_tokens):
                    totals[factor] = totals.get(factor, 0.0) + weight
                    matched = True
            if not matched:
                totals["other"] = totals.get("other", 0.0) + weight
                unexplained += weight
        ranked = sorted(
            [{"factor": key, "weight": round(value, 6)} for key, value in totals.items()],
            key=lambda item: item["weight"],
            reverse=True,
        )
        facts = {
            "factor_exposures": ranked,
            "largest_factor": ranked[0]["factor"] if ranked else None,
            "largest_factor_weight": ranked[0]["weight"] if ranked else 0.0,
            "unexplained_weight": round(unexplained, 6),
        }
        return _envelope("factor_exposure_summary", facts=facts, assumptions=assumptions)
    except Exception as exc:
        return _error("factor_exposure_summary", f"Error computing factor exposures: {exc}", assumptions=assumptions)


@mcp.tool()
def drawdown_risk_profile(returns: list[float]) -> dict[str, Any]:
    assumptions = {"period_count": len(returns or [])}
    try:
        if len(returns) < 2:
            return _error("drawdown_risk_profile", "Provide at least 2 return observations.", assumptions=assumptions)
        max_dd_result = calculate_max_drawdown(returns)
        if max_dd_result.get("errors"):
            return _error("drawdown_risk_profile", max_dd_result["errors"][0], assumptions=assumptions)
        dd = float(max_dd_result["facts"]["max_drawdown_decimal"])
        duration = int(max_dd_result["facts"]["drawdown_duration_periods"])
        recovery = int(max_dd_result["facts"]["recovery_periods"])
        severity = "elevated" if dd >= 0.20 else "moderate" if dd >= 0.10 else "contained"
        facts = {
            "max_drawdown_decimal": dd,
            "drawdown_duration_periods": duration,
            "recovery_periods": recovery,
            "drawdown_severity": severity,
        }
        return _envelope("drawdown_risk_profile", facts=facts, assumptions=assumptions)
    except Exception as exc:
        return _error("drawdown_risk_profile", f"Error computing drawdown risk profile: {exc}", assumptions=assumptions)


@mcp.tool()
def liquidity_stress(
    positions: list[dict[str, Any]],
    redemption_pct: float = 0.10,
) -> dict[str, Any]:
    assumptions = {"position_count": len(positions or []), "redemption_pct": redemption_pct}
    try:
        if redemption_pct < 0:
            return _error("liquidity_stress", "redemption_pct must be non-negative.", assumptions=assumptions)
        stressed_days = 0.0
        illiquid_weight = 0.0
        weighted_days = 0.0
        for item in positions:
            weight = float(item.get("weight", 0.0))
            adv = float(item.get("avg_daily_volume_weight", item.get("adv_weight", 0.2)) or 0.2)
            liquidation_days = float(item.get("liquidation_days", 0.0) or 0.0)
            if liquidation_days <= 0.0:
                liquidation_days = weight / max(adv, 1e-6)
            weighted_days += weight * liquidation_days
            stressed_days = max(stressed_days, liquidation_days)
            if liquidation_days > 5:
                illiquid_weight += weight
        liquidity_gap = redemption_pct - max(0.0, 1.0 - illiquid_weight)
        facts = {
            "weighted_liquidation_days": round(weighted_days, 4),
            "worst_position_liquidation_days": round(stressed_days, 4),
            "illiquid_weight": round(illiquid_weight, 6),
            "redemption_pct": float(redemption_pct),
            "liquidity_gap": round(liquidity_gap, 6),
            "stress_assessment": "tight" if liquidity_gap > 0 else "manageable",
        }
        return _envelope("liquidity_stress", facts=facts, assumptions=assumptions, is_estimated=True)
    except Exception as exc:
        return _error("liquidity_stress", f"Error computing liquidity stress: {exc}", assumptions=assumptions)


if __name__ == "__main__":
    mcp.run()
