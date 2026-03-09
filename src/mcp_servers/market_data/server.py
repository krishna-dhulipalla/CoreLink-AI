import logging
from datetime import datetime

import pandas as pd
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("MarketData")

logger = logging.getLogger(__name__)


def _to_python_scalar(value):
    """Convert pandas/numpy scalars to plain Python JSON-safe values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _format_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the leading date/datetime column to YYYY-MM-DD strings."""
    if df.empty:
        return df

    date_col = df.columns[0]
    series = df[date_col]
    if hasattr(series, "dt"):
        try:
            df[date_col] = series.dt.strftime("%Y-%m-%d")
            return df
        except Exception:
            pass

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        df[date_col] = parsed.dt.strftime("%Y-%m-%d").fillna(series.astype(str))
    else:
        df[date_col] = series.astype(str)
    return df


def _get_yfinance():
    try:
        import yfinance as yf_module
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run: uv add yfinance"
        ) from exc
    return yf_module


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
        yf = _get_yfinance()
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return {"error": f"No price history found for {ticker} with period {period} and interval {interval}."}
            
        # Reset index to make Dates a column and convert timestamps to strings
        df.reset_index(inplace=True)
        df = _format_date_column(df)
        
        start_close = float(df["Close"].iloc[0]) if "Close" in df.columns else None
        end_close = float(df["Close"].iloc[-1]) if "Close" in df.columns else None
        total_rows = len(df)
        if total_rows > 100:
            # Keep both ends of the series so long-period reasoning still sees
            # the opening and latest observations instead of only the tail.
            window = pd.concat([df.head(50), df.tail(50)], ignore_index=True)
            notice = (
                f"Output truncated from {total_rows} to 100 rows. "
                "Includes the earliest 50 rows and latest 50 rows."
            )
        else:
            window = df
            notice = ""

        return {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "columns": df.columns.tolist(),
            "total_rows": int(total_rows),
            "start_close": start_close,
            "end_close": end_close,
            "data": window.to_dict(orient="records"),
            "notice": notice,
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
        yf = _get_yfinance()
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
        
        fundamentals = {
            k: _to_python_scalar(info.get(k))
            for k in essential_keys
            if k in info
        }
        
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
        yf = _get_yfinance()
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
        yf = _get_yfinance()
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
                yields[maturity] = float(_to_python_scalar(history["Close"].iloc[-1]))
                
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
        yf = _get_yfinance()
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
