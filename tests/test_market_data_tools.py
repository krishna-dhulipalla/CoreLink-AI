import pandas as pd
import pytest

pytest.importorskip("engine.mcp.server.fastmcp")

from engine.mcp.mcp_servers.market_data import server as market_server


class _DummyTicker:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.history_calls: list[dict] = []
        self.info = {
            "symbol": symbol,
            "shortName": f"{symbol} Corp",
            "quoteType": "EQUITY",
            "currency": "USD",
            "exchange": "NASDAQ",
            "country": "United States",
            "industry": "Software",
            "sector": "Technology",
            "marketCap": 3000000000,
            "trailingPE": 28.5,
            "priceToBook": 8.2,
            "currentRatio": 1.8,
            "quickRatio": 1.5,
        }
        self.fast_info = {
            "symbol": symbol,
            "currency": "USD",
            "exchange": "NASDAQ",
        }
        self.dividends = pd.Series(
            [0.75, 0.8],
            index=pd.to_datetime(["2024-06-30", "2024-09-30"]),
            name="Dividends",
        )
        self.splits = pd.Series(
            [2.0],
            index=pd.to_datetime(["2023-05-01"]),
            name="Stock Splits",
        )
        self.income_stmt = pd.DataFrame(
            {
                pd.Timestamp("2024-12-31"): [1000.0, 110.0],
                pd.Timestamp("2023-12-31"): [900.0, 95.0],
            },
            index=["Total Revenue", "Net Income"],
        )

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        index = pd.to_datetime(["2024-10-10", "2024-10-11", "2024-10-14"])
        if self.symbol.startswith("^"):
            return pd.DataFrame({"Close": [4.8, 4.82, 4.85]}, index=index)
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1200, 1300],
            },
            index=index,
        )


class _DummyYFinance:
    def __init__(self):
        self._tickers: dict[str, _DummyTicker] = {}

    def Ticker(self, symbol: str):
        if symbol not in self._tickers:
            self._tickers[symbol] = _DummyTicker(symbol)
        return self._tickers[symbol]


@pytest.fixture(autouse=True)
def _clear_market_cache():
    market_server._EVIDENCE_CACHE.clear()
    yield
    market_server._EVIDENCE_CACHE.clear()


def test_resolve_financial_entity_uses_cache(monkeypatch):
    provider = _DummyYFinance()
    monkeypatch.setattr(market_server, "_get_yfinance", lambda: provider)

    first = market_server.resolve_financial_entity("MSFT")
    second = market_server.resolve_financial_entity("MSFT")

    assert first["facts"]["ticker"] == "MSFT"
    assert first["quality"]["cache_hit"] is False
    assert second["quality"]["cache_hit"] is True


def test_get_price_history_binds_as_of_date(monkeypatch):
    provider = _DummyYFinance()
    monkeypatch.setattr(market_server, "_get_yfinance", lambda: provider)

    result = market_server.get_price_history("MSFT", period="1mo", interval="1d", as_of_date="2024-10-14")

    assert result["facts"]["as_of_date"] == "2024-10-14"
    assert result["source"]["timestamp"] == "2024-10-14"
    assert provider.Ticker("MSFT").history_calls[-1]["end"] == "2024-10-15"


def test_get_company_fundamentals_rejects_historical_as_of_date(monkeypatch):
    provider = _DummyYFinance()
    monkeypatch.setattr(market_server, "_get_yfinance", lambda: provider)

    result = market_server.get_company_fundamentals("MSFT", as_of_date="2023-01-01")

    assert result["errors"]
    assert "Historical fundamentals snapshots" in result["errors"][0]


def test_get_financial_statements_returns_structured_periods(monkeypatch):
    provider = _DummyYFinance()
    monkeypatch.setattr(market_server, "_get_yfinance", lambda: provider)

    result = market_server.get_financial_statements(
        "MSFT",
        statement_type="income",
        frequency="annual",
        limit=2,
        as_of_date="2024-12-31",
    )

    assert result["errors"] == []
    assert result["facts"]["periods"][0]["period_end"] == "2024-12-31"
    assert result["facts"]["periods"][0]["line_items"]["Total Revenue"] == 1000.0


def test_get_statement_line_items_matches_requested_labels(monkeypatch):
    provider = _DummyYFinance()
    monkeypatch.setattr(market_server, "_get_yfinance", lambda: provider)

    result = market_server.get_statement_line_items(
        "MSFT",
        line_items=["revenue", "net income", "ebitda"],
        statement_type="income",
        frequency="annual",
        limit=2,
    )

    assert result["facts"]["matched_line_items"]["revenue"] == "Total Revenue"
    assert result["facts"]["matched_line_items"]["net income"] == "Net Income"
    assert "ebitda" in result["quality"]["missing_fields"]


def test_invalid_as_of_date_returns_structured_error():
    result = market_server.get_returns("MSFT", period="1y", as_of_date="not-a-date")

    assert result["errors"]
    assert "Invalid as_of_date" in result["errors"][0]


def test_get_corporate_actions_handles_tz_aware_indexes(monkeypatch):
    provider = _DummyYFinance()
    ticker = provider.Ticker("MSFT")
    ticker.dividends.index = ticker.dividends.index.tz_localize("America/New_York")
    ticker.splits.index = ticker.splits.index.tz_localize("America/New_York")
    monkeypatch.setattr(market_server, "_get_yfinance", lambda: provider)

    result = market_server.get_corporate_actions("MSFT", as_of_date="2024-10-14")

    assert result["errors"] == []
    assert result["facts"]["as_of_date"] == "2024-10-14"
    assert result["facts"]["recent_dividends"][-1]["Date"] == "2024-09-30"
