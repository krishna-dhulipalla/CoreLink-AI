"""
Finance Tools: Deterministic Domain Calculators
================================================
Purpose-built tools for options pricing, Greeks, and mispricing analysis.
These are pure-math functions — no internet, no LLM reasoning, no eval().
The agent calls them with structured numeric inputs and gets exact answers.
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
    """Calculate the theoretical Black-Scholes price for a European option.

    Args:
        S: Current underlying asset price (e.g. 175.0)
        K: Strike price (e.g. 180.0)
        T_days: Days to expiration (e.g. 30)
        r: Annual risk-free rate as decimal (e.g. 0.05 for 5%)
        sigma: Annual implied volatility as decimal (e.g. 0.25 for 25%)
        option_type: 'call' or 'put' (default: 'call')

    Returns:
        Formatted string with theoretical price, d1, d2, and put price via put-call parity.
    """
    try:
        T = T_days / 365.0
        if T <= 0:
            return "Error: T_days must be a positive number."
        if S <= 0 or K <= 0 or sigma <= 0:
            return "Error: S, K, and sigma must all be positive."

        d1, d2 = _bs_d1d2(S, K, T, r, sigma)

        call_price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        put_price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

        if option_type.lower() == "put":
            price = put_price
        else:
            price = call_price

        return (
            f"Black-Scholes Results:\n"
            f"  d1 = {d1:.4f}\n"
            f"  d2 = {d2:.4f}\n"
            f"  Call Price = ${call_price:.2f}\n"
            f"  Put Price  = ${put_price:.2f}\n"
            f"  Requested ({option_type}) = ${price:.2f}"
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
        Formatted string with all Greeks for both call and put options.
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

        # Delta
        call_delta = _norm_cdf(d1)
        put_delta = call_delta - 1.0

        # Gamma (same for call and put)
        gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)

        # Theta (per day, not annualized)
        call_theta = (
            -S * _norm_pdf(d1) * sigma / (2.0 * sqrt_T)
            - r * K * e_rT * _norm_cdf(d2)
        ) / 365.0
        put_theta = (
            -S * _norm_pdf(d1) * sigma / (2.0 * sqrt_T)
            + r * K * e_rT * _norm_cdf(-d2)
        ) / 365.0

        # Vega (per 1% change in vol → divide by 100)
        vega = S * _norm_pdf(d1) * sqrt_T / 100.0

        # Rho (per 1% change in rate → divide by 100)
        call_rho = K * T * e_rT * _norm_cdf(d2) / 100.0
        put_rho = -K * T * e_rT * _norm_cdf(-d2) / 100.0

        return (
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
    """Compare a market option price to its Black-Scholes theoretical value.

    Args:
        market_price: Observed market price of the option
        S: Current underlying asset price
        K: Strike price
        T_days: Days to expiration
        r: Annual risk-free rate as decimal
        sigma: Implied volatility used by market (as decimal)
        option_type: 'call' or 'put'

    Returns:
        Assessment of whether the option is fairly priced, overpriced, or underpriced.
    """
    try:
        T = T_days / 365.0
        d1, d2 = _bs_d1d2(S, K, T, r, sigma)

        call_price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        put_price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

        theoretical = call_price if option_type.lower() == "call" else put_price
        discrepancy = market_price - theoretical
        discrepancy_pct = (discrepancy / theoretical) * 100.0

        if abs(discrepancy_pct) < 2.0:
            assessment = "FAIRLY PRICED"
        elif discrepancy_pct > 0:
            assessment = "OVERPRICED"
        else:
            assessment = "UNDERPRICED"

        return (
            f"Mispricing Analysis ({option_type.upper()}):\n"
            f"  Theoretical (Black-Scholes) = ${theoretical:.2f}\n"
            f"  Market Price                = ${market_price:.2f}\n"
            f"  Discrepancy                 = ${discrepancy:+.2f} ({discrepancy_pct:+.1f}%)\n"
            f"  Assessment: {assessment}\n"
            f"  (d1={d1:.4f}, d2={d2:.4f})"
        )
    except Exception as e:
        return f"Error in mispricing analysis: {e}"


# All finance tools registered for the agent
FINANCE_TOOLS = [black_scholes_price, option_greeks, mispricing_analysis]
