"""
Paper Trading Simulator MCP Server
====================================
Maintains an in-memory simulated portfolio for executing and tracking
synthetic options trades. All state is per-process (ephemeral).
"""

import math
import uuid
from engine.mcp.server.fastmcp import FastMCP

mcp = FastMCP("Paper Trading Simulator")

# ── In-memory state ────────────────────────────────────────────────────────
_portfolios: dict[str, dict] = {}


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _bs_price(S, K, T_days, r, sigma, option_type="call") -> float:
    T = T_days / 365.0
    if T <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    e_rT = math.exp(-r * T)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * e_rT * _norm_cdf(d2)
    return K * e_rT * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# ── Tools ──────────────────────────────────────────────────────────────────

@mcp.tool()
def create_portfolio(name: str, initial_cash: float = 100_000.0) -> str:
    """Create a new paper trading portfolio with a given starting cash balance.

    Args:
        name: Descriptive name for the portfolio (e.g. 'test-run-1')
        initial_cash: Starting virtual USD balance (default: 100,000)

    Returns:
        Portfolio ID and confirmation.
    """
    pid = str(uuid.uuid4())[:8]
    _portfolios[pid] = {
        "name": name,
        "cash": initial_cash,
        "positions": {},
        "trades": [],
    }
    return (
        f"Portfolio created.\n"
        f"  ID   : {pid}\n"
        f"  Name : {name}\n"
        f"  Cash : ${initial_cash:,.2f}"
    )


@mcp.tool()
def execute_options_trade(
    portfolio_id: str,
    action: str,
    option_type: str,
    S: float,
    K: float,
    T_days: int,
    r: float,
    sigma: float,
    contracts: int = 1,
    slippage_pct: float = 0.01,
) -> str:
    """Execute a simulated options trade (buy or sell) with slippage.

    Args:
        portfolio_id: Portfolio ID from create_portfolio
        action: 'buy' or 'sell'
        option_type: 'call' or 'put'
        S: Current spot price
        K: Strike price
        T_days: Days to expiration
        r: Risk-free rate as decimal
        sigma: Implied volatility as decimal
        contracts: Number of contracts (1 contract = 100 shares)
        slippage_pct: Execution slippage as decimal (default 0.01 = 1%)

    Returns:
        Confirmation with fill price and updated cash balance.
    """
    if portfolio_id not in _portfolios:
        return f"Error: Portfolio '{portfolio_id}' not found. Use create_portfolio first."
    
    p = _portfolios[portfolio_id]
    fair_price = _bs_price(S, K, T_days, r, sigma, option_type)
    
    if action.lower() == "buy":
        fill = fair_price * (1 + slippage_pct)
        cost = fill * contracts * 100
        if cost > p["cash"]:
            return f"Error: Insufficient cash (have ${p['cash']:,.2f}, need ${cost:,.2f})"
        p["cash"] -= cost
    else:
        fill = fair_price * (1 - slippage_pct)
        cost = -fill * contracts * 100
        p["cash"] -= cost

    pos_key = f"{option_type.upper()}-K{K}-T{T_days}"
    prev = p["positions"].get(pos_key, {"contracts": 0, "avg_cost": 0.0})
    sign = 1 if action == "buy" else -1
    new_contracts = prev["contracts"] + sign * contracts
    if new_contracts == 0:
        p["positions"].pop(pos_key, None)
    else:
        p["positions"][pos_key] = {"contracts": new_contracts, "avg_cost": fill, "K": K, "T_days": T_days, "option_type": option_type}

    trade = {"action": action, "type": option_type, "K": K, "T_days": T_days,
             "contracts": contracts, "fill": fill, "cost": abs(cost)}
    p["trades"].append(trade)

    return (
        f"Trade Executed:\n"
        f"  {action.upper()} {contracts}x {option_type.upper()} K={K} T={T_days}d\n"
        f"  Fair Value  : ${fair_price:.4f}\n"
        f"  Fill Price  : ${fill:.4f} (slippage: {slippage_pct:.1%})\n"
        f"  Trade Value : ${abs(cost):,.2f}\n"
        f"  Cash After  : ${p['cash']:,.2f}"
    )


@mcp.tool()
def get_positions(portfolio_id: str, current_S: float, current_r: float = 0.05, current_sigma: float = 0.25) -> str:
    """View all open positions in a portfolio with current mark-to-market P&L.

    Args:
        portfolio_id: Portfolio ID from create_portfolio
        current_S: Current spot price of the underlying
        current_r: Current risk-free rate
        current_sigma: Current implied volatility for repricing

    Returns:
        Table of open positions with unrealized P&L.
    """
    if portfolio_id not in _portfolios:
        return f"Error: Portfolio '{portfolio_id}' not found."
    
    p = _portfolios[portfolio_id]
    if not p["positions"]:
        return f"Portfolio '{p['name']}' has no open positions.\nCash: ${p['cash']:,.2f}"

    lines = [f"Positions for '{p['name']}' (Cash: ${p['cash']:,.2f}):"]
    lines.append(f"{'Position':<22} {'Qty':>6} {'AvgCost':>10} {'MktPrice':>10} {'UnrealPnL':>12}")
    lines.append("-" * 65)

    total_pnl = 0.0
    for key, pos in p["positions"].items():
        mkt = _bs_price(current_S, pos["K"], pos["T_days"], current_r, current_sigma, pos["option_type"])
        unrealized = (mkt - pos["avg_cost"]) * pos["contracts"] * 100
        total_pnl += unrealized
        lines.append(
            f"{key:<22} {pos['contracts']:>6} {pos['avg_cost']:>10.4f} {mkt:>10.4f} {unrealized:>+12.2f}"
        )

    lines.append("-" * 65)
    lines.append(f"{'Total Unrealized P&L':>60} {total_pnl:>+12.2f}")
    return "\n".join(lines)


@mcp.tool()
def get_pnl_report(portfolio_id: str) -> str:
    """Get a comprehensive P&L trade history report from a portfolio.

    Args:
        portfolio_id: Portfolio ID from create_portfolio

    Returns:
        Full trade log with realized costs and portfolio summary.
    """
    if portfolio_id not in _portfolios:
        return f"Error: Portfolio '{portfolio_id}' not found."
    
    p = _portfolios[portfolio_id]
    lines = [f"P&L Report: {p['name']}"]
    lines.append(f"  Cash Balance : ${p['cash']:,.2f}")
    lines.append(f"  Open Positions: {len(p['positions'])}")
    lines.append(f"  Total Trades : {len(p['trades'])}")
    lines.append("")
    lines.append("Trade History:")
    lines.append(f"  {'#':<4} {'Action':<6} {'Type':<5} {'Strike':>8} {'Exp':>6} {'Qty':>5} {'Fill':>10} {'Value':>12}")
    lines.append("  " + "-" * 57)

    for i, t in enumerate(p["trades"], 1):
        lines.append(
            f"  {i:<4} {t['action'].upper():<6} {t['type'].upper():<5} "
            f"{t['K']:>8.2f} {t['T_days']:>5}d {t['contracts']:>5} "
            f"{t['fill']:>10.4f} {t['cost']:>+12.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
