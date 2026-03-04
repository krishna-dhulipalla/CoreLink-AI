"""
Risk Metrics MCP Server
========================
Provides portfolio-level risk analytics: Greeks aggregation, Value at Risk (VaR),
stress testing, and performance ratio calculations.
"""

import math
import random
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Risk Metrics")

# ── Helpers ────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _norm_inv_approx(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer-Moro algorithm)."""
    if p < 0.5:
        return -_norm_inv_approx(1 - p)
    q = math.sqrt(-2 * math.log(1 - p))
    a = [2.515517, 0.802853, 0.010328]
    b = [1.432788, 0.189269, 0.001308]
    num = a[0] + a[1] * q + a[2] * q**2
    den = 1 + b[0] * q + b[1] * q**2 + b[2] * q**3
    return q - num / den

def _bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Return (delta, gamma, theta, vega, rho) tuple."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    e_rT = math.exp(-r * T)
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * _norm_pdf(d1) * math.sqrt(T) / 100.0

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = (-S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))
                 - r * K * e_rT * _norm_cdf(d2)) / 365.0
        rho   = K * T * e_rT * _norm_cdf(d2) / 100.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))
                 + r * K * e_rT * _norm_cdf(-d2)) / 365.0
        rho   = -K * T * e_rT * _norm_cdf(-d2) / 100.0
    return delta, gamma, theta, vega, rho


# ── MCP Tools ──────────────────────────────────────────────────────────────

@mcp.tool()
def calculate_portfolio_greeks(positions: list[dict]) -> str:
    """Calculate aggregated Greeks across a portfolio of options positions.

    Each position dict must contain:
        - S, K, T_days, r, sigma, option_type ('call'/'put'), contracts, action ('buy'/'sell')

    Args:
        positions: List of position dictionaries

    Returns:
        Aggregated portfolio Greeks and sensitivity profile.
    """
    try:
        total = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        for pos in positions:
            S, K = float(pos["S"]), float(pos["K"])
            T = float(pos["T_days"]) / 365.0
            r, sigma = float(pos["r"]), float(pos["sigma"])
            contracts = int(pos.get("contracts", 1)) * 100  # 100 shares per contract
            sign = 1 if pos.get("action", "buy").lower() == "buy" else -1
            d, g, th, v, rh = _bs_greeks(S, K, T, r, sigma, pos.get("option_type", "call"))
            total["delta"] += d * contracts * sign
            total["gamma"] += g * contracts * sign
            total["theta"] += th * contracts * sign
            total["vega"]  += v * contracts * sign
            total["rho"]   += rh * contracts * sign

        return (
            f"Aggregate Portfolio Greeks ({len(positions)} positions):\n"
            f"  Delta : {total['delta']:+.4f}  ($ move per $1 underlying move)\n"
            f"  Gamma : {total['gamma']:+.4f}  (Delta change per $1 underlying move)\n"
            f"  Theta : {total['theta']:+.4f}  ($ decay per calendar day)\n"
            f"  Vega  : {total['vega']:+.4f}  ($ change per 1% vol move)\n"
            f"  Rho   : {total['rho']:+.4f}  ($ change per 1% rate move)\n\n"
            f"  Directional Bias : {'BULLISH' if total['delta'] > 0 else 'BEARISH'} (Δ={total['delta']:+.2f})\n"
            f"  Time Decay       : {'POSITIVE' if total['theta'] > 0 else 'NEGATIVE'} (Θ={total['theta']:+.4f}/day)\n"
            f"  Vol Exposure     : {'LONG VOL' if total['vega'] > 0 else 'SHORT VOL'} (ν={total['vega']:+.4f})"
        )
    except Exception as e:
        return f"Error aggregating portfolio Greeks: {e}"


@mcp.tool()
def calculate_var(
    portfolio_value: float,
    daily_vol: float,
    confidence_level: float = 0.95,
    holding_period_days: int = 1,
    method: str = "parametric",
) -> str:
    """Calculate Value at Risk (VaR) for a portfolio.

    Args:
        portfolio_value: Total market value of the portfolio in USD
        daily_vol: Daily portfolio volatility as decimal (e.g. 0.02 for 2%)
        confidence_level: VaR confidence level (e.g. 0.95 for 95%, 0.99 for 99%)
        holding_period_days: Horizon in days (default 1 for daily VaR)
        method: 'parametric' (default) or 'monte_carlo'

    Returns:
        VaR estimate in dollar terms and as a percentage of portfolio.
    """
    try:
        T_sqrt = math.sqrt(holding_period_days)
        z = _norm_inv_approx(confidence_level)

        if method.lower() == "monte_carlo":
            random.seed(42)  # reproducible
            n_sims = 10_000
            returns = [random.gauss(0, daily_vol * T_sqrt) for _ in range(n_sims)]
            returns.sort()
            cutoff_idx = int((1 - confidence_level) * n_sims)
            pct_var = -returns[cutoff_idx]
            method_label = "Monte Carlo (10k sims)"
        else:
            pct_var = z * daily_vol * T_sqrt
            method_label = "Parametric (Normal)"

        dollar_var = pct_var * portfolio_value
        cvar_pct   = pct_var * 1.15  # approximate CVaR (Expected Shortfall)
        dollar_cvar = cvar_pct * portfolio_value

        return (
            f"Value at Risk (VaR) Report:\n"
            f"  Method           : {method_label}\n"
            f"  Portfolio Value  : ${portfolio_value:,.2f}\n"
            f"  Daily Volatility : {daily_vol:.2%}\n"
            f"  Confidence Level : {confidence_level:.0%}\n"
            f"  Holding Period   : {holding_period_days}d\n"
            f"  ──────────────────────────────────\n"
            f"  VaR ({confidence_level:.0%})     : ${dollar_var:,.2f} ({pct_var:.2%})\n"
            f"  CVaR (exp. loss) : ${dollar_cvar:,.2f} ({cvar_pct:.2%})\n\n"
            f"  Interpretation: With {confidence_level:.0%} confidence, the portfolio\n"
            f"  will NOT lose more than ${dollar_var:,.2f} in {holding_period_days} day(s)."
        )
    except Exception as e:
        return f"Error computing VaR: {e}"


@mcp.tool()
def calculate_risk_metrics(returns: list[float], risk_free_rate: float = 0.05) -> str:
    """Calculate key portfolio performance and risk ratios from a return series.

    Args:
        returns: List of periodic returns (e.g. daily returns as decimals)
        risk_free_rate: Annual risk-free rate as decimal (default 0.05)

    Returns:
        Sharpe, Sortino, Calmar, and max drawdown statistics.
    """
    try:
        if len(returns) < 2:
            return "Error: Provide at least 2 return observations."
        n = len(returns)
        mean_r = sum(returns) / n
        variance = sum((r - mean_r)**2 for r in returns) / (n - 1)
        std_r = math.sqrt(variance)

        # Annualize assuming daily
        ann_return = (1 + mean_r) ** 252 - 1
        ann_vol    = std_r * math.sqrt(252)
        daily_rf   = risk_free_rate / 252

        sharpe = (mean_r - daily_rf) / std_r * math.sqrt(252) if std_r > 0 else 0.0

        downside = [r for r in returns if r < daily_rf]
        down_std = math.sqrt(sum((r - daily_rf)**2 for r in downside) / max(len(downside), 1))
        sortino  = (mean_r - daily_rf) / down_std * math.sqrt(252) if down_std > 0 else 0.0

        # Max drawdown
        peak = cum = 1.0
        max_dd = 0.0
        for r in returns:
            cum *= (1 + r)
            if cum > peak:
                peak = cum
            dd = (peak - cum) / peak
            if dd > max_dd:
                max_dd = dd

        calmar = ann_return / max_dd if max_dd > 0 else float("inf")

        return (
            f"Portfolio Risk Metrics ({n} periods):\n"
            f"  Annualized Return : {ann_return:+.2%}\n"
            f"  Annualized Vol    : {ann_vol:.2%}\n"
            f"  Max Drawdown      : {max_dd:.2%}\n"
            f"  ──────────────────────────────\n"
            f"  Sharpe Ratio      : {sharpe:.3f}\n"
            f"  Sortino Ratio     : {sortino:.3f}\n"
            f"  Calmar Ratio      : {calmar:.3f}\n\n"
            f"  Assessment: {'Strong' if sharpe > 1.5 else 'Acceptable' if sharpe > 0.5 else 'Weak'} "
            f"risk-adjusted performance."
        )
    except Exception as e:
        return f"Error computing risk metrics: {e}"


@mcp.tool()
def run_stress_test(
    portfolio_value: float,
    portfolio_delta: float,
    portfolio_vega: float,
    portfolio_theta: float,
) -> str:
    """Run predefined scenario stress tests on a portfolio using top-level Greeks.

    Args:
        portfolio_value: Total market value of portfolio in USD
        portfolio_delta: Net portfolio delta (from calculate_portfolio_greeks)
        portfolio_vega: Net portfolio vega (per 1% vol move)
        portfolio_theta: Net portfolio theta (per day)

    Returns:
        P&L impact table across multiple market stress scenarios.
    """
    scenarios = [
        {"name": "Market +5%",     "spot_move": 0.05,  "vol_move": -0.02, "days": 1},
        {"name": "Market +10%",    "spot_move": 0.10,  "vol_move": -0.04, "days": 1},
        {"name": "Market -5%",     "spot_move": -0.05, "vol_move":  0.03, "days": 1},
        {"name": "Market -10%",    "spot_move": -0.10, "vol_move":  0.06, "days": 1},
        {"name": "Vol Spike +20%", "spot_move": 0.00,  "vol_move":  0.20, "days": 1},
        {"name": "Vol Crush -20%", "spot_move": 0.00,  "vol_move": -0.20, "days": 1},
        {"name": "1-Week Decay",   "spot_move": 0.00,  "vol_move":  0.00, "days": 7},
        {"name": "Crash -20%",     "spot_move": -0.20, "vol_move":  0.15, "days": 1},
    ]

    lines = [
        f"Stress Test Results (Portfolio Value: ${portfolio_value:,.2f}):",
        f"  Net Delta={portfolio_delta:+.2f}, Vega={portfolio_vega:+.2f}, Theta={portfolio_theta:+.4f}/day",
        "",
        f"  {'Scenario':<22} {'Spot Δ':>8} {'Vol Δ':>8} {'P&L':>12} {'P&L%':>8}",
        "  " + "-" * 60,
    ]

    S_ref = portfolio_value / max(abs(portfolio_delta), 0.01) * 0.01  # approximate
    for s in scenarios:
        spot_pnl  = portfolio_delta * s["spot_move"] * portfolio_value / 100
        vega_pnl  = portfolio_vega  * s["vol_move"] * 100
        theta_pnl = portfolio_theta * s["days"]
        total_pnl = spot_pnl + vega_pnl + theta_pnl
        pct = (total_pnl / portfolio_value) * 100
        lines.append(
            f"  {s['name']:<22} {s['spot_move']:>+7.0%} {s['vol_move']:>+7.0%} "
            f"{total_pnl:>+12,.2f} {pct:>+7.2f}%"
        )
    return "\n".join(lines)


@mcp.tool()
def calculate_max_drawdown(returns: list[float]) -> str:
    """Calculate maximum drawdown and recovery analysis from a return series.

    Args:
        returns: List of periodic returns (e.g. daily returns as decimals)

    Returns:
        Max drawdown %, duration, and recovery summary.
    """
    try:
        if not returns:
            return "Error: Empty returns list."
        peak = cum = 1.0
        max_dd = 0.0
        dd_start = dd_end = 0
        current_start = 0

        for i, r in enumerate(returns):
            cum *= (1 + r)
            if cum >= peak:
                peak = cum
                current_start = i
            else:
                dd = (peak - cum) / peak
                if dd > max_dd:
                    max_dd = dd
                    dd_start = current_start
                    dd_end = i

        recovery_periods = len(returns) - dd_end if max_dd > 0 else 0
        total_return = (cum - 1)

        return (
            f"Max Drawdown Analysis ({len(returns)} periods):\n"
            f"  Max Drawdown     : {max_dd:.2%}\n"
            f"  Drawdown Period  : periods {dd_start}–{dd_end} ({dd_end - dd_start} periods)\n"
            f"  Recovery Periods : {recovery_periods} (after trough)\n"
            f"  Total Return     : {total_return:+.2%}\n"
            f"  Underwater %     : {max_dd:.2%} of peak equity lost at worst"
        )
    except Exception as e:
        return f"Error computing max drawdown: {e}"


if __name__ == "__main__":
    mcp.run()
