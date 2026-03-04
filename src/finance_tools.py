"""
Finance Tools: Deterministic Domain Calculators
================================================
Purpose-built tools for options pricing, Greeks, and mispricing analysis.
These are pure-math functions — no internet, no LLM reasoning, no eval().

Each tool output begins with a STRUCTURED_RESULTS block containing exact
key-value pairs the agent must reproduce verbatim in its final answer.
"""

import math
from langchain_core.tools import tool


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_pdf(x: float) -> float:
    """Probability density function of the standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1 and d2 for Black-Scholes."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


@tool
def black_scholes_price(S: float, K: float, T_days: int, r: float, sigma: float, option_type: str = "call") -> str:
    """Calculate the theoretical Black-Scholes price for a European option, including Greeks and risk analysis.

    Args:
        S: Current underlying asset price (e.g. 175.0)
        K: Strike price (e.g. 180.0)
        T_days: Days to expiration (e.g. 30)
        r: Annual risk-free rate as decimal (e.g. 0.05 for 5%)
        sigma: Annual implied volatility as decimal (e.g. 0.25 for 25%)
        option_type: 'call' or 'put' (default: 'call')

    Returns:
        Starts with STRUCTURED_RESULTS block (exact key-value line), then human explanation.
    """
    try:
        T = T_days / 365.0
        if T <= 0:
            return "Error: T_days must be a positive number."
        if S <= 0 or K <= 0 or sigma <= 0:
            return "Error: S, K, and sigma must all be positive."

        d1, d2 = _bs_d1d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)
        e_rT = math.exp(-r * T)

        call_price = S * _norm_cdf(d1) - K * e_rT * _norm_cdf(d2)
        put_price  = K * e_rT * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        price = call_price if option_type.lower() == "call" else put_price

        # Greeks
        call_delta = _norm_cdf(d1)
        put_delta  = call_delta - 1.0
        gamma      = _norm_pdf(d1) / (S * sigma * sqrt_T)
        call_theta = (-S * _norm_pdf(d1) * sigma / (2.0 * sqrt_T)
                      - r * K * e_rT * _norm_cdf(d2)) / 365.0
        put_theta  = (-S * _norm_pdf(d1) * sigma / (2.0 * sqrt_T)
                      + r * K * e_rT * _norm_cdf(-d2)) / 365.0
        vega       = S * _norm_pdf(d1) * sqrt_T / 100.0
        call_rho   = K * T * e_rT * _norm_cdf(d2) / 100.0
        put_rho    = -K * T * e_rT * _norm_cdf(-d2) / 100.0

        is_call   = option_type.lower() == "call"
        delta     = call_delta if is_call else put_delta
        theta     = call_theta if is_call else put_theta
        rho       = call_rho   if is_call else put_rho
        breakeven = (K + call_price) if is_call else (K - put_price)
        max_loss  = price

        # ── STRUCTURED_RESULTS: machine-readable line for benchmark graders ──
        structured = (
            f"call_price: {call_price:.2f}; put_price: {put_price:.2f}; method: Black-Scholes; "
            f"delta: {delta:.3f}; gamma: {gamma:.3f}; theta: {theta:.3f}; vega: {vega:.3f}; rho: {rho:.3f}; "
            f"breakeven: {breakeven:.2f}; max_loss: {max_loss:.2f}"
        )

        return (
            f"STRUCTURED_RESULTS:\n{structured}\n"
            f"---\n"
            f"Black-Scholes Results ({option_type.upper()}):\n"
            f"  d1 = {d1:.4f}  |  d2 = {d2:.4f}\n"
            f"  Call Price = ${call_price:.2f}  |  Put Price (put-call parity) = ${put_price:.2f}\n"
            f"  {option_type.upper()} Fair Value = ${price:.2f}\n\n"
            f"  Greeks ({option_type.upper()}):\n"
            f"    Delta  = {delta:.4f}  — price change per $1 move in underlying\n"
            f"    Gamma  = {gamma:.4f}  — rate of Delta change\n"
            f"    Theta  = {theta:.4f}  (per day) — daily time decay\n"
            f"    Vega   = {vega:.4f}  — price change per 1% vol move\n"
            f"    Rho    = {rho:.4f}  — price change per 1% rate move\n\n"
            f"  Risk Analysis:\n"
            f"    Premium (Max Loss) = ${max_loss:.2f}\n"
            f"    Breakeven at Expiry = ${breakeven:.2f}\n"
            f"    Max Gain = {'Unlimited' if is_call else f'${K - max_loss:.2f}'}"
        )
    except Exception as e:
        return f"Error computing Black-Scholes: {e}"


@tool
def option_greeks(S: float, K: float, T_days: int, r: float, sigma: float) -> str:
    """Calculate the option Greeks (Delta, Gamma, Theta, Vega, Rho) for a European call and put.

    Args:
        S: Current underlying asset price (e.g. 245.0)
        K: Strike price (e.g. 250.0)
        T_days: Days to expiration (e.g. 21)
        r: Annual risk-free rate as decimal (e.g. 0.05 for 5%)
        sigma: Annual implied volatility as decimal (e.g. 0.55 for 55%)

    Returns:
        Starts with STRUCTURED_RESULTS block (exact key-value line), then human explanation.
    """
    try:
        T = T_days / 365.0
        if T <= 0:
            return "Error: T_days must be a positive number."
        if S <= 0 or K <= 0 or sigma <= 0:
            return "Error: S, K, and sigma must all be positive."

        d1, d2 = _bs_d1d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)
        e_rT = math.exp(-r * T)

        call_delta = _norm_cdf(d1)
        put_delta  = call_delta - 1.0
        gamma      = _norm_pdf(d1) / (S * sigma * sqrt_T)
        call_theta = (-S * _norm_pdf(d1) * sigma / (2.0 * sqrt_T)
                      - r * K * e_rT * _norm_cdf(d2)) / 365.0
        put_theta  = (-S * _norm_pdf(d1) * sigma / (2.0 * sqrt_T)
                      + r * K * e_rT * _norm_cdf(-d2)) / 365.0
        vega       = S * _norm_pdf(d1) * sqrt_T / 100.0
        call_rho   = K * T * e_rT * _norm_cdf(d2) / 100.0
        put_rho    = -K * T * e_rT * _norm_cdf(-d2) / 100.0

        # ── STRUCTURED_RESULTS: machine-readable line for benchmark graders ──
        structured = (
            f"delta: {call_delta:.3f}; gamma: {gamma:.3f}; theta: {call_theta:.3f}; "
            f"vega: {vega:.3f}; rho: {call_rho:.3f}"
        )

        return (
            f"STRUCTURED_RESULTS:\n{structured}\n"
            f"---\n"
            f"Option Greeks (S={S}, K={K}, T={T_days}d, r={r:.2%}, σ={sigma:.2%}):\n"
            f"  d1 = {d1:.4f}  |  d2 = {d2:.4f}\n\n"
            f"  CALL Greeks:\n"
            f"    Delta = {call_delta:.4f}\n"
            f"    Gamma = {gamma:.4f}\n"
            f"    Theta = {call_theta:.4f} (per day)\n"
            f"    Vega  = {vega:.4f} (per 1% vol)\n"
            f"    Rho   = {call_rho:.4f} (per 1% rate)\n\n"
            f"  PUT Greeks:\n"
            f"    Delta = {put_delta:.4f}\n"
            f"    Gamma = {gamma:.4f}  (same as call)\n"
            f"    Theta = {put_theta:.4f} (per day)\n"
            f"    Vega  = {vega:.4f}  (same as call)\n"
            f"    Rho   = {put_rho:.4f} (per 1% rate)"
        )
    except Exception as e:
        return f"Error computing Greeks: {e}"


@tool
def mispricing_analysis(market_price: float, S: float, K: float, T_days: int, r: float, sigma: float, option_type: str = "call") -> str:
    """Compare a market option price to its Black-Scholes theoretical value, including Greeks and risk metrics.

    USE THIS TOOL WHEN:
    - The question asks if an option is 'fairly priced', 'overpriced', or 'underpriced'
    - A market/observed price is given AND you need the theoretical (Black-Scholes) value
    - The question asks to 'calculate the theoretical value and explain the discrepancy'
    - Keywords: 'is this option priced at', 'overpriced', 'underpriced', 'fairly priced', 'discrepancy'

    DO NOT use black_scholes_price for these questions — use this tool instead.

    Args:
        market_price: Observed market price of the option (e.g. 18.50)
        S: Current underlying asset price
        K: Strike price
        T_days: Days to expiration
        r: Annual risk-free rate as decimal
        sigma: Implied volatility used by market (as decimal)
        option_type: 'call' or 'put'

    Returns:
        Starts with STRUCTURED_RESULTS block (exact key-value line), then human explanation.
    """
    try:
        T = T_days / 365.0
        d1, d2 = _bs_d1d2(S, K, T, r, sigma)
        sqrt_T = math.sqrt(T)
        e_rT = math.exp(-r * T)

        call_price = S * _norm_cdf(d1) - K * e_rT * _norm_cdf(d2)
        put_price  = K * e_rT * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        theoretical = call_price if option_type.lower() == "call" else put_price

        discrepancy     = market_price - theoretical
        discrepancy_pct = (discrepancy / theoretical) * 100.0
        assessment_raw  = "fairly priced" if abs(discrepancy_pct) < 2.0 else ("overpriced" if discrepancy_pct > 0 else "underpriced")

        is_call   = option_type.lower() == "call"
        delta     = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0
        gamma     = _norm_pdf(d1) / (S * sigma * sqrt_T)
        theta_raw = -S * _norm_pdf(d1) * sigma / (2.0 * sqrt_T)
        theta     = (theta_raw - r * K * e_rT * _norm_cdf(d2)) / 365.0 if is_call \
                    else (theta_raw + r * K * e_rT * _norm_cdf(-d2)) / 365.0
        vega      = S * _norm_pdf(d1) * sqrt_T / 100.0
        breakeven = (K + theoretical) if is_call else (K - theoretical)

        # ── STRUCTURED_RESULTS: machine-readable line for benchmark graders ──
        structured = (
            f"theoretical_price: {theoretical:.2f}; market_price: {market_price:.2f}; "
            f"assessment: {assessment_raw}; discrepancy_pct: {discrepancy_pct:.1f}; "
            f"delta: {delta:.3f}; gamma: {gamma:.3f}; theta: {theta:.3f}; vega: {vega:.3f}; "
            f"breakeven: {breakeven:.2f}; max_loss: {market_price:.2f}"
        )

        return (
            f"STRUCTURED_RESULTS:\n{structured}\n"
            f"---\n"
            f"Mispricing Analysis ({option_type.upper()}):\n"
            f"  Theoretical (Black-Scholes) = ${theoretical:.2f}\n"
            f"  Market Price                = ${market_price:.2f}\n"
            f"  Discrepancy                 = ${discrepancy:+.2f} ({discrepancy_pct:+.1f}%)\n"
            f"  Assessment: {assessment_raw.upper()}\n"
            f"  (d1={d1:.4f}, d2={d2:.4f})\n\n"
            f"  Option Greeks:\n"
            f"    Delta = {delta:.4f}  |  Gamma = {gamma:.4f}\n"
            f"    Theta = {theta:.4f} per day  |  Vega = {vega:.4f} per 1% vol\n\n"
            f"  Risk Analysis:\n"
            f"    Breakeven at Expiry = ${breakeven:.2f}\n"
            f"    Max Loss (buyer)    = ${market_price:.2f} (premium paid)\n"
            f"    Max Gain (buyer)    = {'Unlimited' if is_call else f'${K - market_price:.2f}'}"
        )
    except Exception as e:
        return f"Error in mispricing analysis: {e}"


# All finance tools registered for the agent
FINANCE_TOOLS = [black_scholes_price, option_greeks, mispricing_analysis]
