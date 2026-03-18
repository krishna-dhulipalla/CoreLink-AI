import logging
import math
from statistics import mean, median
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("FinanceAnalytics")
logger = logging.getLogger(__name__)


def _result_envelope(
    tool_name: str,
    *,
    facts: dict[str, Any] | None = None,
    assumptions: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": tool_name,
        "facts": facts or {},
        "assumptions": assumptions or {},
        "source": {
            "tool": tool_name,
            "provider": "local_analytics",
        },
        "quality": {
            "is_synthetic": False,
            "is_estimated": False,
            "cache_hit": False,
            "missing_fields": [],
            **(quality or {}),
        },
        "errors": list(errors or []),
    }


def _error_result(tool_name: str, message: str, *, assumptions: dict[str, Any] | None = None) -> dict[str, Any]:
    return _result_envelope(tool_name, assumptions=assumptions, errors=[message])


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


@mcp.tool()
def weighted_average(values: list[float], weights: list[float]) -> dict:
    assumptions = {"values": values, "weights": weights}
    try:
        if len(values) != len(weights):
            return _error_result("weighted_average", "Lengths of values and weights must match.", assumptions=assumptions)
        total_weight = sum(weights)
        if total_weight == 0:
            return _error_result("weighted_average", "Sum of weights cannot be zero.", assumptions=assumptions)
        result = sum(v * w for v, w in zip(values, weights)) / total_weight
        return _result_envelope(
            "weighted_average",
            facts={"weighted_average": round(result, 6), "weight_sum": total_weight},
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in weighted_average: %s", exc)
        return _error_result("weighted_average", str(exc), assumptions=assumptions)


@mcp.tool()
def sum_values(values: list[float]) -> dict:
    assumptions = {"values": values}
    try:
        return _result_envelope(
            "sum_values",
            facts={"sum": sum(values), "count": len(values)},
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in sum_values: %s", exc)
        return _error_result("sum_values", str(exc), assumptions=assumptions)


@mcp.tool()
def pct_change(old_value: float, new_value: float) -> dict:
    assumptions = {"old_value": old_value, "new_value": new_value}
    try:
        if old_value == 0:
            return _error_result("pct_change", "old_value cannot be 0.", assumptions=assumptions)
        change = (new_value - old_value) / old_value
        return _result_envelope(
            "pct_change",
            facts={"percentage_change_decimal": round(change, 6), "percentage_change": round(change * 100, 4)},
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in pct_change: %s", exc)
        return _error_result("pct_change", str(exc), assumptions=assumptions)


@mcp.tool()
def cagr(beginning_value: float, ending_value: float, years: float) -> dict:
    assumptions = {
        "beginning_value": beginning_value,
        "ending_value": ending_value,
        "years": years,
    }
    try:
        if beginning_value <= 0 or ending_value <= 0:
            return _error_result("cagr", "Values must be positive for CAGR calculation.", assumptions=assumptions)
        if years <= 0:
            return _error_result("cagr", "Years must be greater than 0.", assumptions=assumptions)
        cagr_value = (ending_value / beginning_value) ** (1 / years) - 1
        return _result_envelope(
            "cagr",
            facts={"cagr_decimal": round(cagr_value, 6), "cagr_percent": round(cagr_value * 100, 4)},
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in cagr: %s", exc)
        return _error_result("cagr", str(exc), assumptions=assumptions)


@mcp.tool()
def annualize_return(period_return_decimal: float, days_held: float) -> dict:
    assumptions = {"period_return_decimal": period_return_decimal, "days_held": days_held}
    try:
        if days_held <= 0:
            return _error_result("annualize_return", "Days held must be positive.", assumptions=assumptions)
        annualized = (1 + period_return_decimal) ** (365 / days_held) - 1
        return _result_envelope(
            "annualize_return",
            facts={
                "annualized_return_decimal": round(annualized, 6),
                "annualized_return_percent": round(annualized * 100, 4),
            },
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in annualize_return: %s", exc)
        return _error_result("annualize_return", str(exc), assumptions=assumptions)


@mcp.tool()
def annualize_volatility(period_volatility_decimal: float, periods_per_year: int = 252) -> dict:
    assumptions = {
        "period_volatility_decimal": period_volatility_decimal,
        "periods_per_year": periods_per_year,
    }
    try:
        if periods_per_year <= 0:
            return _error_result("annualize_volatility", "Periods per year must be positive.", assumptions=assumptions)
        annualized = period_volatility_decimal * math.sqrt(periods_per_year)
        return _result_envelope(
            "annualize_volatility",
            facts={
                "annualized_volatility_decimal": round(annualized, 6),
                "annualized_volatility_percent": round(annualized * 100, 4),
            },
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in annualize_volatility: %s", exc)
        return _error_result("annualize_volatility", str(exc), assumptions=assumptions)


@mcp.tool()
def bond_price_yield(
    face_value: float,
    coupon_rate_decimal: float,
    periods_to_maturity: int,
    yield_to_maturity_decimal: float,
) -> dict:
    assumptions = {
        "face_value": face_value,
        "coupon_rate_decimal": coupon_rate_decimal,
        "periods_to_maturity": periods_to_maturity,
        "yield_to_maturity_decimal": yield_to_maturity_decimal,
    }
    try:
        coupon_payment = face_value * coupon_rate_decimal
        if yield_to_maturity_decimal == 0:
            pv_coupons = coupon_payment * periods_to_maturity
            pv_face = face_value
        else:
            pv_coupons = coupon_payment * (
                (1 - (1 + yield_to_maturity_decimal) ** -periods_to_maturity) / yield_to_maturity_decimal
            )
            pv_face = face_value / ((1 + yield_to_maturity_decimal) ** periods_to_maturity)
        price = pv_coupons + pv_face
        return _result_envelope(
            "bond_price_yield",
            facts={"bond_price": round(price, 4), "coupon_payment": round(coupon_payment, 4)},
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in bond_price_yield: %s", exc)
        return _error_result("bond_price_yield", str(exc), assumptions=assumptions)


@mcp.tool()
def duration_convexity(
    face_value: float,
    coupon_rate_decimal: float,
    periods_to_maturity: int,
    yield_to_maturity_decimal: float,
) -> dict:
    assumptions = {
        "face_value": face_value,
        "coupon_rate_decimal": coupon_rate_decimal,
        "periods_to_maturity": periods_to_maturity,
        "yield_to_maturity_decimal": yield_to_maturity_decimal,
    }
    try:
        coupon = face_value * coupon_rate_decimal
        ytm = yield_to_maturity_decimal
        periods = periods_to_maturity
        price = 0.0
        macaulay_numerator = 0.0
        convexity_numerator = 0.0

        for period in range(1, periods + 1):
            cash_flow = coupon if period < periods else coupon + face_value
            discounted = cash_flow / ((1 + ytm) ** period)
            price += discounted
            macaulay_numerator += period * discounted
            convexity_numerator += cash_flow * (period * (period + 1)) / ((1 + ytm) ** (period + 2))

        if price == 0:
            return _error_result("duration_convexity", "Bond price is 0.", assumptions=assumptions)

        macaulay_duration = macaulay_numerator / price
        modified_duration = macaulay_duration / (1 + ytm)
        convexity = convexity_numerator / price
        return _result_envelope(
            "duration_convexity",
            facts={
                "price": round(price, 4),
                "macaulay_duration": round(macaulay_duration, 4),
                "modified_duration": round(modified_duration, 4),
                "convexity": round(convexity, 4),
            },
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in duration_convexity: %s", exc)
        return _error_result("duration_convexity", str(exc), assumptions=assumptions)


@mcp.tool()
def du_pont_analysis(
    net_income: float | None = None,
    revenue: float | None = None,
    average_assets: float | None = None,
    average_equity: float | None = None,
    net_margin: float | None = None,
    asset_turnover: float | None = None,
    equity_multiplier: float | None = None,
) -> dict:
    assumptions = {
        "net_income": net_income,
        "revenue": revenue,
        "average_assets": average_assets,
        "average_equity": average_equity,
        "net_margin": net_margin,
        "asset_turnover": asset_turnover,
        "equity_multiplier": equity_multiplier,
    }
    try:
        missing_fields = []
        if net_margin is None:
            if net_income is None or revenue in (None, 0):
                missing_fields.append("net_margin or (net_income and revenue)")
            else:
                net_margin = net_income / revenue
        if asset_turnover is None:
            if revenue is None or average_assets in (None, 0):
                missing_fields.append("asset_turnover or (revenue and average_assets)")
            else:
                asset_turnover = revenue / average_assets
        if equity_multiplier is None:
            if average_assets is None or average_equity in (None, 0):
                missing_fields.append("equity_multiplier or (average_assets and average_equity)")
            else:
                equity_multiplier = average_assets / average_equity

        if missing_fields:
            return _error_result(
                "du_pont_analysis",
                "Insufficient inputs for DuPont analysis.",
                assumptions=assumptions,
            )

        roe = float(net_margin) * float(asset_turnover) * float(equity_multiplier)
        return _result_envelope(
            "du_pont_analysis",
            facts={
                "net_margin_decimal": round(float(net_margin), 6),
                "asset_turnover": round(float(asset_turnover), 6),
                "equity_multiplier": round(float(equity_multiplier), 6),
                "roe_decimal": round(roe, 6),
                "roe_percent": round(roe * 100, 4),
            },
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in du_pont_analysis: %s", exc)
        return _error_result("du_pont_analysis", str(exc), assumptions=assumptions)


@mcp.tool()
def liquidity_ratio_pack(
    current_assets: float,
    current_liabilities: float,
    inventory: float = 0.0,
    cash: float = 0.0,
    marketable_securities: float = 0.0,
    receivables: float = 0.0,
) -> dict:
    assumptions = {
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "inventory": inventory,
        "cash": cash,
        "marketable_securities": marketable_securities,
        "receivables": receivables,
    }
    try:
        if current_liabilities == 0:
            return _error_result("liquidity_ratio_pack", "current_liabilities cannot be 0.", assumptions=assumptions)
        working_capital = current_assets - current_liabilities
        current_ratio = current_assets / current_liabilities
        quick_assets = current_assets - inventory
        quick_ratio = quick_assets / current_liabilities
        cash_ratio = (cash + marketable_securities) / current_liabilities
        defensive_interval_assets = cash + marketable_securities + receivables
        return _result_envelope(
            "liquidity_ratio_pack",
            facts={
                "working_capital": round(working_capital, 4),
                "current_ratio": round(current_ratio, 6),
                "quick_ratio": round(quick_ratio, 6),
                "cash_ratio": round(cash_ratio, 6),
                "defensive_liquid_assets": round(defensive_interval_assets, 4),
            },
            assumptions=assumptions,
        )
    except Exception as exc:
        logger.error("Error in liquidity_ratio_pack: %s", exc)
        return _error_result("liquidity_ratio_pack", str(exc), assumptions=assumptions)


@mcp.tool()
def valuation_multiples_compare(
    subject: dict[str, float],
    peers: list[dict[str, float]],
    metrics: list[str] | None = None,
) -> dict:
    assumptions = {"subject": subject, "peers": peers, "metrics": metrics or []}
    try:
        if not peers:
            return _error_result("valuation_multiples_compare", "At least one peer is required.", assumptions=assumptions)

        available_metrics = sorted(
            {
                key
                for record in [subject, *peers]
                for key, value in record.items()
                if key not in {"name", "ticker"} and isinstance(value, (int, float))
            }
        )
        selected_metrics = metrics or available_metrics
        comparison = {}
        missing_fields = []

        for metric in selected_metrics:
            subject_value = subject.get(metric)
            peer_values = [float(record[metric]) for record in peers if isinstance(record.get(metric), (int, float))]
            if not isinstance(subject_value, (int, float)) or not peer_values:
                missing_fields.append(metric)
                continue
            peer_mean = _safe_mean(peer_values)
            peer_median = median(peer_values)
            premium_discount = (float(subject_value) / peer_mean - 1) if peer_mean else None
            comparison[metric] = {
                "subject": round(float(subject_value), 6),
                "peer_mean": round(peer_mean, 6),
                "peer_median": round(float(peer_median), 6),
                "peer_min": round(min(peer_values), 6),
                "peer_max": round(max(peer_values), 6),
                "premium_discount_to_mean": round(premium_discount, 6) if premium_discount is not None else None,
            }

        if not comparison:
            return _error_result(
                "valuation_multiples_compare",
                "No comparable numeric metrics were available across subject and peers.",
                assumptions=assumptions,
            )

        return _result_envelope(
            "valuation_multiples_compare",
            facts={
                "subject_name": subject.get("name") or subject.get("ticker"),
                "peer_count": len(peers),
                "metrics": comparison,
            },
            assumptions=assumptions,
            quality={"missing_fields": missing_fields},
        )
    except Exception as exc:
        logger.error("Error in valuation_multiples_compare: %s", exc)
        return _error_result("valuation_multiples_compare", str(exc), assumptions=assumptions)


@mcp.tool()
def dcf_sensitivity_grid(
    base_fcf: float,
    growth_rates: list[float],
    discount_rates: list[float],
    terminal_growth_rate: float = 0.02,
    years: int = 5,
) -> dict:
    assumptions = {
        "base_fcf": base_fcf,
        "growth_rates": growth_rates,
        "discount_rates": discount_rates,
        "terminal_growth_rate": terminal_growth_rate,
        "years": years,
    }
    try:
        if years <= 0:
            return _error_result("dcf_sensitivity_grid", "years must be positive.", assumptions=assumptions)
        if not growth_rates or not discount_rates:
            return _error_result("dcf_sensitivity_grid", "growth_rates and discount_rates cannot be empty.", assumptions=assumptions)

        grid = []
        for growth_rate in growth_rates:
            row = {"growth_rate": growth_rate, "values": []}
            projected_flows = []
            current_fcf = base_fcf
            for _ in range(years):
                current_fcf *= 1 + growth_rate
                projected_flows.append(current_fcf)
            for discount_rate in discount_rates:
                if discount_rate <= terminal_growth_rate:
                    row["values"].append({"discount_rate": discount_rate, "enterprise_value": None})
                    continue
                discounted = sum(
                    cash_flow / ((1 + discount_rate) ** period)
                    for period, cash_flow in enumerate(projected_flows, start=1)
                )
                terminal_value = projected_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
                discounted_terminal = terminal_value / ((1 + discount_rate) ** years)
                row["values"].append(
                    {
                        "discount_rate": discount_rate,
                        "enterprise_value": round(discounted + discounted_terminal, 4),
                    }
                )
            grid.append(row)

        return _result_envelope(
            "dcf_sensitivity_grid",
            facts={
                "base_fcf": base_fcf,
                "terminal_growth_rate": terminal_growth_rate,
                "years": years,
                "grid": grid,
            },
            assumptions=assumptions,
            quality={"is_estimated": True},
        )
    except Exception as exc:
        logger.error("Error in dcf_sensitivity_grid: %s", exc)
        return _error_result("dcf_sensitivity_grid", str(exc), assumptions=assumptions)


@mcp.tool()
def cashflow_waterfall(
    operating_cash_flow: float,
    capex: float,
    taxes: float = 0.0,
    debt_service: float = 0.0,
    working_capital_change: float = 0.0,
    dividends: float = 0.0,
) -> dict:
    assumptions = {
        "operating_cash_flow": operating_cash_flow,
        "capex": capex,
        "taxes": taxes,
        "debt_service": debt_service,
        "working_capital_change": working_capital_change,
        "dividends": dividends,
    }
    try:
        free_cash_flow = operating_cash_flow - capex - taxes - debt_service - working_capital_change
        residual_cash = free_cash_flow - dividends
        facts = {
            "operating_cash_flow": round(float(operating_cash_flow), 4),
            "capex": round(float(capex), 4),
            "taxes": round(float(taxes), 4),
            "debt_service": round(float(debt_service), 4),
            "working_capital_change": round(float(working_capital_change), 4),
            "free_cash_flow": round(float(free_cash_flow), 4),
            "dividends": round(float(dividends), 4),
            "residual_cash": round(float(residual_cash), 4),
            "coverage_ratio": round(float(free_cash_flow) / max(abs(float(dividends)), 1e-6), 6) if dividends else None,
        }
        return _result_envelope("cashflow_waterfall", facts=facts, assumptions=assumptions)
    except Exception as exc:
        logger.error("Error in cashflow_waterfall: %s", exc)
        return _error_result("cashflow_waterfall", str(exc), assumptions=assumptions)


@mcp.tool()
def bond_spread_duration(
    modified_duration: float,
    spread_duration: float,
    spread_change_bps: float,
    benchmark_yield_change_bps: float = 0.0,
) -> dict:
    assumptions = {
        "modified_duration": modified_duration,
        "spread_duration": spread_duration,
        "spread_change_bps": spread_change_bps,
        "benchmark_yield_change_bps": benchmark_yield_change_bps,
    }
    try:
        rate_impact = -float(modified_duration) * (float(benchmark_yield_change_bps) / 10000.0)
        spread_impact = -float(spread_duration) * (float(spread_change_bps) / 10000.0)
        total_impact = rate_impact + spread_impact
        facts = {
            "rate_price_impact_decimal": round(rate_impact, 6),
            "spread_price_impact_decimal": round(spread_impact, 6),
            "total_price_impact_decimal": round(total_impact, 6),
            "total_price_impact_percent": round(total_impact * 100, 4),
        }
        return _result_envelope("bond_spread_duration", facts=facts, assumptions=assumptions, quality={"is_estimated": True})
    except Exception as exc:
        logger.error("Error in bond_spread_duration: %s", exc)
        return _error_result("bond_spread_duration", str(exc), assumptions=assumptions)


if __name__ == "__main__":
    mcp.run()
