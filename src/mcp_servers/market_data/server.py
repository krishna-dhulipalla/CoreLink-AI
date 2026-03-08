import logging
import json
from datetime import datetime
from typing import Optional, Dict

import yfinance as yf
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("MarketData")

logger = logging.getLogger(__name__)


@mcp.tool()
def get_price_history(ticker: str, period: str = "1mo", interval: str = "1d") -> dict:
    """
    Fetch historical OHLCV (Open, High, Low, Close, Volume) data for a given ticker.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        period: Time period to fetch. Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        interval: Data interval. Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return {"error": f"No price history found for {ticker} with period {period} and interval {interval}."}
            
        # Reset index to make Dates a column and convert timestamps to strings
        df.reset_index(inplace=True)
        # Handle both standard 'Date' and datetimes column
        date_col = df.columns[0]
        df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
        
        return {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "columns": df.columns.tolist(),
            "data": df.tail(100).to_dict(orient="records"), # limit output to last 100 rows to prevent context explosion
            "notice": "Output limited to most recent 100 periods." if len(df) > 100 else ""
        }
    except Exception as e:
        logger.error(f"Error fetching price history for {ticker}: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_company_fundamentals(ticker: str) -> dict:
    """
    Retrieve key fundamental metrics, ratios, and company information for a given ticker.
    Includes items like Market Cap, PE Ratio, PB Ratio, Debt to Equity, ROE, etc.
    
    Args:
        ticker: The stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or len(info) < 5:
            return {"error": f"Could not retrieve fundamentals for {ticker}."}
            
        # Extract the most critical financial ratios to prevent massive payload
        essential_keys = [
            "shortName", "sector", "industry", "marketCap", "enterpriseValue",
            "trailingPE", "forwardPE", "pegRatio", "priceToBook",
            "profitMargins", "operatingMargins", "returnOnAssets", "returnOnEquity",
            "revenueGrowth", "earningsGrowth", "totalDebt", "debtToEquity",
            "currentRatio", "quickRatio", "freeCashflow", "operatingCashflow",
            "dividendYield", "payoutRatio", "beta", "trailingEps", "forwardEps"
        ]
        
        fundamentals = {k: info.get(k) for k in essential_keys if k in info}
        
        return {
            "ticker": ticker.upper(),
            "fundamentals": fundamentals
        }
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_corporate_actions(ticker: str) -> dict:
    """
    Retrieve historical dividends and stock splits for a given ticker.
    
    Args:
        ticker: The stock ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)
        
        dividends = stock.dividends
        splits = stock.splits
        
        div_list = []
        if not dividends.empty:
            div_df = dividends.reset_index()
            div_col = div_df.columns[0]
            div_df[div_col] = div_df[div_col].dt.strftime('%Y-%m-%d')
            div_list = div_df.tail(20).to_dict(orient="records") # last 20 dividends
            
        split_list = []
        if not splits.empty:
            split_df = splits.reset_index()
            split_col = split_df.columns[0]
            split_df[split_col] = split_df[split_col].dt.strftime('%Y-%m-%d')
            split_list = split_df.tail(10).to_dict(orient="records") # last 10 splits
            
        return {
            "ticker": ticker.upper(),
            "recent_dividends": div_list,
            "recent_splits": split_list
        }
    except Exception as e:
        logger.error(f"Error fetching corporate actions for {ticker}: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_yield_curve() -> dict:
    """
    Retrieve the current US Treasury yield curve proxy using common indices.
    Fetches 13-week, 5-year, 10-year, and 30-year treasury yields.
    """
    try:
        # ^IRX (13-week), ^FVX (5-year), ^TNX (10-year), ^TYX (30-year)
        tickers = {
            "3_Month": "^IRX",
            "5_Year": "^FVX",
            "10_Year": "^TNX",
            "30_Year": "^TYX"
        }
        
        yields = {}
        for maturity, sym in tickers.items():
            stock = yf.Ticker(sym)
            history = stock.history(period="1d")
            if not history.empty:
                # The yield is usually the Close price for these indices
                yields[maturity] = history['Close'].iloc[-1]
                
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "yields_pct": yields
        }
    except Exception as e:
        logger.error(f"Error fetching yield curve: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_returns(ticker: str, period: str = "1y") -> dict:
    """
    Calculate the simple point-to-point percentage return of a ticker over a given period.
    
    Args:
        ticker: The stock ticker symbol.
        period: Time period (e.g., '1mo', '3mo', '6mo', '1y', '5y')
    """
    try:
        stock = yf.Ticker(ticker)
        # Fetch the start and end of the period
        df = stock.history(period=period, interval="1d")
        if df.empty or len(df) < 2:
            return {"error": f"Not enough data to calculate returns for {period}."}
            
        start_price = float(df['Close'].iloc[0])
        end_price = float(df['Close'].iloc[-1])
        
        pct_return = ((end_price - start_price) / start_price) * 100
        
        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_price": start_price,
            "end_price": end_price,
            "return_percent": round(pct_return, 4)
        }
    except Exception as e:
        logger.error(f"Error calculating returns for {ticker}: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
