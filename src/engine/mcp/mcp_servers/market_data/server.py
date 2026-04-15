import copy
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from engine.mcp.server.fastmcp import FastMCP

mcp = FastMCP("MarketData")
logger = logging.getLogger(__name__)

_EVIDENCE_CACHE: dict[str, dict[str, Any]] = {}
_HISTORICAL_FUNDAMENTALS_ERROR = (
    "Historical fundamentals snapshots are not supported by the current provider. "
    "Use statement tools for as-of analysis."
)


def _to_python_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _utc_today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _normalize_as_of_date(as_of_date: str | None) -> str | None:
    if not as_of_date:
        return None
    parsed = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid as_of_date: {as_of_date!r}")
    return parsed.strftime("%Y-%m-%d")


def _normalize_as_of_or_error(as_of_date: str | None) -> tuple[str | None, str | None]:
    try:
        return _normalize_as_of_date(as_of_date), None
    except ValueError as exc:
        return None, str(exc)


def _index_mask_at_or_before(index: Any, as_of_date: str | None) -> Any:
    if not as_of_date:
        return slice(None)
    normalized = pd.to_datetime(index, errors="coerce", utc=True)
    cutoff = pd.Timestamp(as_of_date)
    return normalized.tz_localize(None).normalize() <= cutoff.normalize()


def _history_end_kwargs(as_of_date: str | None) -> dict[str, Any]:
    normalized = _normalize_as_of_date(as_of_date)
    if not normalized:
        return {}
    end_value = (pd.Timestamp(normalized) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return {"end": end_value}


def _cache_key(tool_name: str, payload: dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(payload, sort_keys=True, default=str)}"


def _cache_get(tool_name: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    key = _cache_key(tool_name, payload)
    cached = _EVIDENCE_CACHE.get(key)
    if not cached:
        return None
    result = copy.deepcopy(cached)
    quality = dict(result.get("quality", {}))
    quality["cache_hit"] = True
    result["quality"] = quality
    return result


def _cache_put(tool_name: str, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    stored = copy.deepcopy(result)
    quality = dict(stored.get("quality", {}))
    quality["cache_hit"] = False
    stored["quality"] = quality
    if not stored.get("errors"):
        _EVIDENCE_CACHE[_cache_key(tool_name, payload)] = stored
    return copy.deepcopy(stored)


def _result_envelope(
    tool_name: str,
    *,
    facts: dict[str, Any] | None = None,
    assumptions: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": tool_name,
        "facts": facts or {},
        "assumptions": assumptions or {},
        "source": {"tool": tool_name, **(source or {})},
        "quality": {
            "is_synthetic": False,
            "is_estimated": False,
            "cache_hit": False,
            "missing_fields": [],
            **(quality or {}),
        },
        "errors": list(errors or []),
    }


def _error_result(
    tool_name: str,
    message: str,
    *,
    assumptions: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _result_envelope(
        tool_name,
        facts={},
        assumptions=assumptions,
        source=source,
        quality=quality,
        errors=[message],
    )


def _normalize_corporate_action_records(records: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        cleaned = dict(record)
        if "Date" not in cleaned and cleaned:
            first_key = next(iter(cleaned))
            cleaned["Date"] = cleaned.pop(first_key)
        if "Date" in cleaned:
            cleaned["Date"] = _date_to_string(cleaned.get("Date"))
        normalized.append(cleaned)
    return normalized


def _get_yfinance():
    try:
        import yfinance as yf_module
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: uv add yfinance") from exc
    return yf_module


def _date_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d")


def _format_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    formatted = df.copy()
    date_col = formatted.columns[0]
    parsed = pd.to_datetime(formatted[date_col], errors="coerce")
    if parsed.notna().any():
        formatted[date_col] = parsed.dt.strftime("%Y-%m-%d").fillna(formatted[date_col].astype(str))
    else:
        formatted[date_col] = formatted[date_col].astype(str)
    return formatted


def _statement_attr_candidates(statement_type: str, frequency: str) -> list[str]:
    normalized_type = statement_type.lower().strip().replace(" ", "_")
    normalized_freq = frequency.lower().strip()
    mapping = {
        ("income", "annual"): ["income_stmt", "financials"],
        ("income", "quarterly"): ["quarterly_income_stmt", "quarterly_financials"],
        ("balance_sheet", "annual"): ["balance_sheet"],
        ("balance_sheet", "quarterly"): ["quarterly_balance_sheet"],
        ("cash_flow", "annual"): ["cashflow"],
        ("cash_flow", "quarterly"): ["quarterly_cashflow"],
    }
    if normalized_type not in {"income", "balance_sheet", "cash_flow"}:
        raise ValueError(f"Unsupported statement_type: {statement_type!r}")
    if normalized_freq not in {"annual", "quarterly"}:
        raise ValueError(f"Unsupported frequency: {frequency!r}")
    return mapping[(normalized_type, normalized_freq)]


def _load_statement_frame(stock: Any, statement_type: str, frequency: str) -> pd.DataFrame:
    for attr in _statement_attr_candidates(statement_type, frequency):
        value = getattr(stock, attr, None)
        if isinstance(value, pd.DataFrame) and not value.empty:
            return value.copy()
    return pd.DataFrame()


def _filtered_statement_columns(df: pd.DataFrame, as_of_date: str | None, limit: int) -> list[tuple[Any, pd.Timestamp]]:
    cutoff = pd.Timestamp(_normalize_as_of_date(as_of_date)) if as_of_date else None
    columns: list[tuple[Any, pd.Timestamp]] = []
    for original in df.columns:
        parsed = pd.to_datetime(original, errors="coerce")
        if pd.isna(parsed):
            continue
        if cutoff is not None and parsed.normalize() > cutoff.normalize():
            continue
        columns.append((original, parsed.normalize()))
    columns.sort(key=lambda item: item[1], reverse=True)
    return columns[: max(limit, 1)]


def _normalize_line_item(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _match_line_item(requested: str, available: list[str]) -> str | None:
    normalized_requested = _normalize_line_item(requested)
    exact = {_normalize_line_item(item): item for item in available}
    if normalized_requested in exact:
        return exact[normalized_requested]
    for item in available:
        normalized_item = _normalize_line_item(item)
        if normalized_requested in normalized_item or normalized_item in normalized_requested:
            return item
    return None


def _expected_entity_fields() -> list[str]:
    return ["ticker", "shortName", "quoteType", "currency", "exchange", "country"]


@mcp.tool()
def resolve_financial_entity(identifier: str, as_of_date: str | None = None) -> dict:
    normalized_as_of, date_error = _normalize_as_of_or_error(as_of_date)
    if date_error:
        return _error_result(
            "resolve_financial_entity",
            date_error,
            assumptions={"identifier": identifier, "as_of_date": as_of_date},
            source={"provider": "yfinance", "timestamp": _utc_today()},
        )
    assumptions = {"identifier": identifier, "as_of_date": normalized_as_of}
    cached = _cache_get("resolve_financial_entity", assumptions)
    if cached:
        return cached

    try:
        yf = _get_yfinance()
        stock = yf.Ticker(identifier)
        info = stock.info or {}
        fast_info = getattr(stock, "fast_info", {}) or {}
        resolved_ticker = str(info.get("symbol") or fast_info.get("symbol") or identifier).upper()
        facts = {
            "requested_identifier": identifier,
            "ticker": resolved_ticker,
            "shortName": info.get("shortName") or info.get("longName") or resolved_ticker,
            "quoteType": info.get("quoteType") or fast_info.get("quoteType"),
            "currency": info.get("currency") or fast_info.get("currency"),
            "exchange": info.get("exchange") or fast_info.get("exchange"),
            "country": info.get("country"),
            "industry": info.get("industry"),
            "sector": info.get("sector"),
            "market": info.get("market"),
            "jurisdiction": info.get("country"),
        }
        missing_fields = [field for field in _expected_entity_fields() if not facts.get(field)]
        if not facts.get("ticker"):
            return _error_result(
                "resolve_financial_entity",
                f"Could not resolve entity {identifier!r}.",
                assumptions=assumptions,
                source={"provider": "yfinance", "timestamp": _utc_today()},
                quality={"missing_fields": ["ticker"]},
            )
        result = _result_envelope(
            "resolve_financial_entity",
            facts=facts,
            assumptions=assumptions,
            source={
                "provider": "yfinance",
                "ticker": facts["ticker"],
                "timestamp": assumptions["as_of_date"] or _utc_today(),
                "jurisdiction": facts.get("jurisdiction"),
            },
            quality={"missing_fields": missing_fields},
        )
        return _cache_put("resolve_financial_entity", assumptions, result)
    except Exception as exc:
        logger.error("Error resolving entity %s: %s", identifier, exc)
        return _error_result(
            "resolve_financial_entity",
            str(exc),
            assumptions=assumptions,
            source={"provider": "yfinance", "timestamp": _utc_today()},
        )


@mcp.tool()
def get_price_history(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
    as_of_date: str | None = None,
) -> dict:
    normalized_as_of, date_error = _normalize_as_of_or_error(as_of_date)
    if date_error:
        return _error_result(
            "get_price_history",
            date_error,
            assumptions={"ticker": ticker.upper(), "period": period, "interval": interval, "as_of_date": as_of_date},
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": _utc_today()},
        )
    assumptions = {
        "ticker": ticker.upper(),
        "period": period,
        "interval": interval,
        "as_of_date": normalized_as_of,
    }
    cached = _cache_get("get_price_history", assumptions)
    if cached:
        return cached

    try:
        yf = _get_yfinance()
        stock = yf.Ticker(ticker)
        history_kwargs = {"period": period, "interval": interval, **_history_end_kwargs(normalized_as_of)}
        df = stock.history(**history_kwargs)
        if df.empty:
            return _error_result(
                "get_price_history",
                f"No price history found for {ticker} with period {period} and interval {interval}.",
                assumptions=assumptions,
                source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
            )

        df = df.reset_index()
        df = _format_date_column(df)
        total_rows = len(df)
        window = pd.concat([df.head(50), df.tail(50)], ignore_index=True) if total_rows > 100 else df
        notice = (
            f"Output truncated from {total_rows} to 100 rows. Includes the earliest 50 rows and latest 50 rows."
            if total_rows > 100
            else ""
        )
        date_col = window.columns[0]
        facts = {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "as_of_date": normalized_as_of,
            "columns": [str(column) for column in df.columns.tolist()],
            "total_rows": int(total_rows),
            "window_start": str(window[date_col].iloc[0]),
            "window_end": str(window[date_col].iloc[-1]),
            "start_close": float(window["Close"].iloc[0]) if "Close" in window.columns else None,
            "end_close": float(window["Close"].iloc[-1]) if "Close" in window.columns else None,
            "data": window.to_dict(orient="records"),
            "notice": notice,
        }
        result = _result_envelope(
            "get_price_history",
            facts=facts,
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": facts["window_end"]},
        )
        return _cache_put("get_price_history", assumptions, result)
    except Exception as exc:
        logger.error("Error fetching price history for %s: %s", ticker, exc)
        return _error_result(
            "get_price_history",
            str(exc),
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
        )


@mcp.tool()
def get_company_fundamentals(ticker: str, as_of_date: str | None = None) -> dict:
    normalized_as_of, date_error = _normalize_as_of_or_error(as_of_date)
    if date_error:
        return _error_result(
            "get_company_fundamentals",
            date_error,
            assumptions={"ticker": ticker.upper(), "as_of_date": as_of_date},
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": _utc_today()},
        )
    assumptions = {"ticker": ticker.upper(), "as_of_date": normalized_as_of}
    if normalized_as_of and normalized_as_of != _utc_today():
        return _error_result(
            "get_company_fundamentals",
            _HISTORICAL_FUNDAMENTALS_ERROR,
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of},
        )

    cached = _cache_get("get_company_fundamentals", assumptions)
    if cached:
        return cached

    try:
        yf = _get_yfinance()
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        if not info or len(info) < 5:
            return _error_result(
                "get_company_fundamentals",
                f"Could not retrieve fundamentals for {ticker}.",
                assumptions=assumptions,
                source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": _utc_today()},
            )

        essential_keys = [
            "shortName",
            "sector",
            "industry",
            "marketCap",
            "enterpriseValue",
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToBook",
            "profitMargins",
            "operatingMargins",
            "returnOnAssets",
            "returnOnEquity",
            "revenueGrowth",
            "earningsGrowth",
            "totalDebt",
            "debtToEquity",
            "currentRatio",
            "quickRatio",
            "freeCashflow",
            "operatingCashflow",
            "dividendYield",
            "payoutRatio",
            "beta",
            "trailingEps",
            "forwardEps",
            "currency",
        ]
        fundamentals = {
            key: _to_python_scalar(info.get(key))
            for key in essential_keys
            if key in info and info.get(key) is not None
        }
        missing_fields = [key for key in ("marketCap", "trailingPE", "priceToBook", "currency") if key not in fundamentals]
        result = _result_envelope(
            "get_company_fundamentals",
            facts={"ticker": ticker.upper(), "fundamentals": fundamentals},
            assumptions=assumptions,
            source={
                "provider": "yfinance",
                "ticker": ticker.upper(),
                "timestamp": _utc_today(),
                "currency": fundamentals.get("currency"),
            },
            quality={"missing_fields": missing_fields},
        )
        return _cache_put("get_company_fundamentals", assumptions, result)
    except Exception as exc:
        logger.error("Error fetching fundamentals for %s: %s", ticker, exc)
        return _error_result(
            "get_company_fundamentals",
            str(exc),
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": _utc_today()},
        )


@mcp.tool()
def get_corporate_actions(ticker: str, as_of_date: str | None = None) -> dict:
    normalized_as_of, date_error = _normalize_as_of_or_error(as_of_date)
    if date_error:
        return _error_result(
            "get_corporate_actions",
            date_error,
            assumptions={"ticker": ticker.upper(), "as_of_date": as_of_date},
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": _utc_today()},
        )
    assumptions = {"ticker": ticker.upper(), "as_of_date": normalized_as_of}
    cached = _cache_get("get_corporate_actions", assumptions)
    if cached:
        facts = dict(cached.get("facts", {}))
        facts["recent_dividends"] = _normalize_corporate_action_records(facts.get("recent_dividends"))
        facts["recent_splits"] = _normalize_corporate_action_records(facts.get("recent_splits"))
        cached["facts"] = facts
        return cached

    try:
        yf = _get_yfinance()
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        splits = stock.splits
        if normalized_as_of and not dividends.empty:
            dividends = dividends[_index_mask_at_or_before(dividends.index, normalized_as_of)]
        if normalized_as_of and not splits.empty:
            splits = splits[_index_mask_at_or_before(splits.index, normalized_as_of)]

        recent_dividends = []
        if not dividends.empty:
            div_df = dividends.reset_index()
            div_df.columns = ["Date", *div_df.columns[1:]]
            div_df.iloc[:, 0] = pd.to_datetime(div_df.iloc[:, 0], errors="coerce").dt.strftime("%Y-%m-%d")
            recent_dividends = _normalize_corporate_action_records(div_df.tail(20).to_dict(orient="records"))

        recent_splits = []
        if not splits.empty:
            split_df = splits.reset_index()
            split_df.columns = ["Date", *split_df.columns[1:]]
            split_df.iloc[:, 0] = pd.to_datetime(split_df.iloc[:, 0], errors="coerce").dt.strftime("%Y-%m-%d")
            recent_splits = _normalize_corporate_action_records(split_df.tail(10).to_dict(orient="records"))

        timestamp = normalized_as_of or _utc_today()
        if recent_dividends:
            timestamp = _date_to_string(recent_dividends[-1][next(iter(recent_dividends[-1]))])
        elif recent_splits:
            timestamp = _date_to_string(recent_splits[-1][next(iter(recent_splits[-1]))])

        result = _result_envelope(
            "get_corporate_actions",
            facts={
                "ticker": ticker.upper(),
                "as_of_date": normalized_as_of,
                "recent_dividends": recent_dividends,
                "recent_splits": recent_splits,
            },
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": timestamp},
        )
        return _cache_put("get_corporate_actions", assumptions, result)
    except Exception as exc:
        logger.error("Error fetching corporate actions for %s: %s", ticker, exc)
        return _error_result(
            "get_corporate_actions",
            str(exc),
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
        )


@mcp.tool()
def get_yield_curve(as_of_date: str | None = None) -> dict:
    normalized_as_of, date_error = _normalize_as_of_or_error(as_of_date)
    if date_error:
        return _error_result(
            "get_yield_curve",
            date_error,
            assumptions={"as_of_date": as_of_date},
            source={"provider": "yfinance", "timestamp": _utc_today()},
        )
    assumptions = {"as_of_date": normalized_as_of}
    cached = _cache_get("get_yield_curve", assumptions)
    if cached:
        return cached

    try:
        yf = _get_yfinance()
        tickers = {
            "3_Month": "^IRX",
            "5_Year": "^FVX",
            "10_Year": "^TNX",
            "30_Year": "^TYX",
        }
        yields = {}
        missing_fields = []
        timestamp = normalized_as_of or _utc_today()
        for maturity, symbol in tickers.items():
            history = yf.Ticker(symbol).history(period="7d", **_history_end_kwargs(normalized_as_of))
            if history.empty:
                missing_fields.append(maturity)
                continue
            yields[maturity] = float(_to_python_scalar(history["Close"].iloc[-1]))
            timestamp = _date_to_string(history.index[-1])

        if not yields:
            return _error_result(
                "get_yield_curve",
                "Could not retrieve any Treasury yield points.",
                assumptions=assumptions,
                source={"provider": "yfinance", "timestamp": timestamp},
            )

        result = _result_envelope(
            "get_yield_curve",
            facts={"date": timestamp, "as_of_date": normalized_as_of, "yields_pct": yields},
            assumptions=assumptions,
            source={"provider": "yfinance", "timestamp": timestamp},
            quality={"missing_fields": missing_fields},
        )
        return _cache_put("get_yield_curve", assumptions, result)
    except Exception as exc:
        logger.error("Error fetching yield curve: %s", exc)
        return _error_result(
            "get_yield_curve",
            str(exc),
            assumptions=assumptions,
            source={"provider": "yfinance", "timestamp": normalized_as_of or _utc_today()},
        )


@mcp.tool()
def get_returns(ticker: str, period: str = "1y", as_of_date: str | None = None) -> dict:
    normalized_as_of, date_error = _normalize_as_of_or_error(as_of_date)
    if date_error:
        return _error_result(
            "get_returns",
            date_error,
            assumptions={"ticker": ticker.upper(), "period": period, "as_of_date": as_of_date},
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": _utc_today()},
        )
    assumptions = {"ticker": ticker.upper(), "period": period, "as_of_date": normalized_as_of}
    cached = _cache_get("get_returns", assumptions)
    if cached:
        return cached

    try:
        yf = _get_yfinance()
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval="1d", **_history_end_kwargs(normalized_as_of))
        if history.empty or len(history) < 2:
            return _error_result(
                "get_returns",
                f"Not enough data to calculate returns for {ticker} over {period}.",
                assumptions=assumptions,
                source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
            )

        start_price = float(history["Close"].iloc[0])
        end_price = float(history["Close"].iloc[-1])
        return_decimal = (end_price - start_price) / start_price
        start_date = _date_to_string(history.index[0])
        end_date = _date_to_string(history.index[-1])
        result = _result_envelope(
            "get_returns",
            facts={
                "ticker": ticker.upper(),
                "period": period,
                "as_of_date": normalized_as_of,
                "start_date": start_date,
                "end_date": end_date,
                "start_price": start_price,
                "end_price": end_price,
                "return_decimal": round(return_decimal, 6),
                "return_percent": round(return_decimal * 100, 4),
            },
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": end_date},
        )
        return _cache_put("get_returns", assumptions, result)
    except Exception as exc:
        logger.error("Error calculating returns for %s: %s", ticker, exc)
        return _error_result(
            "get_returns",
            str(exc),
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
        )


@mcp.tool()
def get_financial_statements(
    ticker: str,
    statement_type: str = "income",
    frequency: str = "annual",
    limit: int = 4,
    as_of_date: str | None = None,
) -> dict:
    normalized_as_of, date_error = _normalize_as_of_or_error(as_of_date)
    if date_error:
        return _error_result(
            "get_financial_statements",
            date_error,
            assumptions={
                "ticker": ticker.upper(),
                "statement_type": statement_type,
                "frequency": frequency,
                "limit": limit,
                "as_of_date": as_of_date,
            },
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": _utc_today()},
        )
    assumptions = {
        "ticker": ticker.upper(),
        "statement_type": statement_type,
        "frequency": frequency,
        "limit": limit,
        "as_of_date": normalized_as_of,
    }
    cached = _cache_get("get_financial_statements", assumptions)
    if cached:
        return cached

    try:
        yf = _get_yfinance()
        stock = yf.Ticker(ticker)
        frame = _load_statement_frame(stock, statement_type, frequency)
        if frame.empty:
            return _error_result(
                "get_financial_statements",
                f"No {frequency} {statement_type} statement data found for {ticker}.",
                assumptions=assumptions,
                source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
            )

        selected_columns = _filtered_statement_columns(frame, normalized_as_of, limit)
        if not selected_columns:
            return _error_result(
                "get_financial_statements",
                f"No {statement_type} statement periods remain after applying as_of_date={normalized_as_of}.",
                assumptions=assumptions,
                source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
            )

        frame = frame.fillna(value=pd.NA)
        line_items = [str(item) for item in frame.index.tolist()]
        included_line_items = line_items[:40]
        periods = []
        for original_column, parsed_column in selected_columns:
            line_item_values = {}
            for line_item in included_line_items:
                value = frame.at[line_item, original_column]
                if pd.isna(value):
                    continue
                line_item_values[line_item] = _to_python_scalar(value)
            periods.append({"period_end": parsed_column.strftime("%Y-%m-%d"), "line_items": line_item_values})

        result = _result_envelope(
            "get_financial_statements",
            facts={
                "ticker": ticker.upper(),
                "statement_type": statement_type,
                "frequency": frequency,
                "as_of_date": normalized_as_of,
                "reported_period_count": len(selected_columns),
                "line_item_count": len(line_items),
                "line_items_included": included_line_items,
                "periods": periods,
                "truncated_line_items": len(line_items) > len(included_line_items),
            },
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": periods[0]["period_end"]},
        )
        return _cache_put("get_financial_statements", assumptions, result)
    except Exception as exc:
        logger.error("Error fetching financial statements for %s: %s", ticker, exc)
        return _error_result(
            "get_financial_statements",
            str(exc),
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
        )


@mcp.tool()
def get_statement_line_items(
    ticker: str,
    line_items: list[str],
    statement_type: str = "income",
    frequency: str = "annual",
    limit: int = 4,
    as_of_date: str | None = None,
) -> dict:
    normalized_as_of, date_error = _normalize_as_of_or_error(as_of_date)
    if date_error:
        return _error_result(
            "get_statement_line_items",
            date_error,
            assumptions={
                "ticker": ticker.upper(),
                "line_items": list(line_items),
                "statement_type": statement_type,
                "frequency": frequency,
                "limit": limit,
                "as_of_date": as_of_date,
            },
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": _utc_today()},
        )
    assumptions = {
        "ticker": ticker.upper(),
        "line_items": list(line_items),
        "statement_type": statement_type,
        "frequency": frequency,
        "limit": limit,
        "as_of_date": normalized_as_of,
    }
    cached = _cache_get("get_statement_line_items", assumptions)
    if cached:
        return cached

    try:
        yf = _get_yfinance()
        stock = yf.Ticker(ticker)
        frame = _load_statement_frame(stock, statement_type, frequency)
        if frame.empty:
            return _error_result(
                "get_statement_line_items",
                f"No {frequency} {statement_type} statement data found for {ticker}.",
                assumptions=assumptions,
                source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
            )

        selected_columns = _filtered_statement_columns(frame, normalized_as_of, limit)
        if not selected_columns:
            return _error_result(
                "get_statement_line_items",
                f"No statement periods remain after applying as_of_date={normalized_as_of}.",
                assumptions=assumptions,
                source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
            )

        available = [str(item) for item in frame.index.tolist()]
        matched_line_items: dict[str, str] = {}
        missing_fields: list[str] = []
        for requested in line_items:
            matched = _match_line_item(requested, available)
            if matched:
                matched_line_items[requested] = matched
            else:
                missing_fields.append(requested)

        if not matched_line_items:
            return _error_result(
                "get_statement_line_items",
                f"None of the requested line items matched the available {statement_type} rows.",
                assumptions=assumptions,
                source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
                quality={"missing_fields": missing_fields},
            )

        frame = frame.fillna(value=pd.NA)
        series_payload: dict[str, list[dict[str, Any]]] = {}
        for requested, matched in matched_line_items.items():
            series_payload[requested] = []
            for original_column, parsed_column in selected_columns:
                value = frame.at[matched, original_column]
                if pd.isna(value):
                    continue
                series_payload[requested].append(
                    {"period_end": parsed_column.strftime("%Y-%m-%d"), "value": _to_python_scalar(value)}
                )

        result = _result_envelope(
            "get_statement_line_items",
            facts={
                "ticker": ticker.upper(),
                "statement_type": statement_type,
                "frequency": frequency,
                "as_of_date": normalized_as_of,
                "requested_line_items": list(line_items),
                "matched_line_items": matched_line_items,
                "series": series_payload,
            },
            assumptions=assumptions,
            source={
                "provider": "yfinance",
                "ticker": ticker.upper(),
                "timestamp": selected_columns[0][1].strftime("%Y-%m-%d"),
            },
            quality={"missing_fields": missing_fields},
        )
        return _cache_put("get_statement_line_items", assumptions, result)
    except Exception as exc:
        logger.error("Error fetching statement line items for %s: %s", ticker, exc)
        return _error_result(
            "get_statement_line_items",
            str(exc),
            assumptions=assumptions,
            source={"provider": "yfinance", "ticker": ticker.upper(), "timestamp": normalized_as_of or _utc_today()},
        )


if __name__ == "__main__":
    mcp.run()
