import logging
import math
from typing import List, Dict, Union

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("FinanceAnalytics")

logger = logging.getLogger(__name__)


@mcp.tool()
def weighted_average(values: List[float], weights: List[float]) -> dict:
    """
    Calculate the weighted average of a list of values given a list of weights.
    The length of values and weights must be identical.
    
    Args:
        values: List of numeric values.
        weights: List of numeric weights corresponding to the values.
    """
    try:
        if len(values) != len(weights):
            return {"error": "Lengths of values and weights must match."}
        if sum(weights) == 0:
            return {"error": "Sum of weights cannot be zero."}
            
        wa = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        return {"weighted_average": round(wa, 6)}
    except Exception as e:
        logger.error(f"Error in weighted_average: {e}")
        return {"error": str(e)}


@mcp.tool()
def sum_values(values: List[float]) -> dict:
    """
    Calculate the simple sum of a list of numbers. Useful for exact arithmetic when reasoning over tables.
    
    Args:
        values: List of numeric values to sum.
    """
    try:
        return {"sum": sum(values), "count": len(values)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def pct_change(old_value: float, new_value: float) -> dict:
    """
    Calculate the percentage change from an old value to a new value.
    
    Args:
        old_value: The starting value.
        new_value: The ending value.
    """
    try:
        if old_value == 0:
            return {"error": "old_value cannot be 0."}
        change = ((new_value - old_value) / old_value) * 100
        return {"percentage_change": round(change, 4)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def cagr(beginning_value: float, ending_value: float, years: float) -> dict:
    """
    Calculate the Compound Annual Growth Rate (CAGR).
    
    Args:
        beginning_value: The initial investment or starting value.
        ending_value: The final investment or ending value.
        years: The number of years between the beginning and ending periods.
    """
    try:
        if beginning_value <= 0 or ending_value <= 0:
            return {"error": "Values must be positive for CAGR calculation."}
        if years <= 0:
            return {"error": "Years must be greater than 0."}
            
        cagr_val = ((ending_value / beginning_value) ** (1 / years)) - 1
        return {"cagr_decimal": round(cagr_val, 6), "cagr_percent": round(cagr_val * 100, 4)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def annualize_return(period_return_decimal: float, days_held: float) -> dict:
    """
    Convert a return over a specific holding period into an annualized return.
    
    Args:
        period_return_decimal: The return over the holding period as a decimal (e.g., 0.05 for 5%).
        days_held: The number of days the asset was held.
    """
    try:
        if days_held <= 0:
            return {"error": "Days held must be positive."}
            
        annualized = ((1 + period_return_decimal) ** (365 / days_held)) - 1
        return {"annualized_return_decimal": round(annualized, 6), "annualized_return_percent": round(annualized * 100, 4)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def annualize_volatility(period_volatility_decimal: float, periods_per_year: int = 252) -> dict:
    """
    Annualize a measured period volatility (e.g. daily or monthly standard deviation).
    
    Args:
        period_volatility_decimal: The standard deviation of the period returns as a decimal.
        periods_per_year: The number of periods in a year (e.g. 252 for daily, 12 for monthly).
    """
    try:
        if periods_per_year <= 0:
            return {"error": "Periods per year must be positive."}
            
        ann_vol = period_volatility_decimal * math.sqrt(periods_per_year)
        return {"annualized_volatility_decimal": round(ann_vol, 6), "annualized_volatility_percent": round(ann_vol * 100, 4)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def bond_price_yield(face_value: float, coupon_rate_decimal: float, periods_to_maturity: int, yield_to_maturity_decimal: float) -> dict:
    """
    Calculate the present value (price) of a traditional bond.
    
    Args:
        face_value: The par value of the bond.
        coupon_rate_decimal: The bond's coupon rate per period as a decimal.
        periods_to_maturity: Number of coupon periods remaining.
        yield_to_maturity_decimal: The market discount rate per period as a decimal.
    """
    try:
        coupon_payment = face_value * coupon_rate_decimal
        
        # Present value of annuity (the coupons)
        if yield_to_maturity_decimal == 0:
            pv_coupons = coupon_payment * periods_to_maturity
            pv_face = face_value
        else:
            pv_coupons = coupon_payment * ((1 - (1 + yield_to_maturity_decimal) ** -periods_to_maturity) / yield_to_maturity_decimal)
            pv_face = face_value / ((1 + yield_to_maturity_decimal) ** periods_to_maturity)
            
        price = pv_coupons + pv_face
        return {"bond_price": round(price, 4)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def duration_convexity(face_value: float, coupon_rate_decimal: float, periods_to_maturity: int, yield_to_maturity_decimal: float) -> dict:
    """
    Calculate the Macaulay Duration, Modified Duration, and Convexity of a traditional bond.
    
    Args:
        face_value: The par value of the bond.
        coupon_rate_decimal: The bond's coupon rate per period as a decimal.
        periods_to_maturity: Number of coupon periods remaining.
        yield_to_maturity_decimal: The market discount rate per period as a decimal.
    """
    try:
        coupon = face_value * coupon_rate_decimal
        y = yield_to_maturity_decimal
        n = periods_to_maturity
        
        # Calculate price, duration, and convexity components
        price = 0
        macaulay_numerator = 0
        convexity_numerator = 0
        
        for t in range(1, n + 1):
            cash_flow = coupon if t < n else (coupon + face_value)
            discounted_cf = cash_flow / ((1 + y) ** t)
            
            price += discounted_cf
            macaulay_numerator += t * discounted_cf
            convexity_numerator += cash_flow * (t * (t + 1)) / ((1 + y) ** (t + 2))
            
        if price == 0:
            return {"error": "Bond price is 0."}
            
        mac_duration = macaulay_numerator / price
        mod_duration = mac_duration / (1 + y)
        convexity = convexity_numerator / price
        
        return {
            "price": round(price, 4),
            "macaulay_duration": round(mac_duration, 4),
            "modified_duration": round(mod_duration, 4),
            "convexity": round(convexity, 4)
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
