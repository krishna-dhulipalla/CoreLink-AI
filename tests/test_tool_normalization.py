from agent.tool_normalization import normalize_tool_output


def test_normalize_tool_output_preserves_structured_envelope_dict():
    result = normalize_tool_output(
        "get_returns",
        {
            "type": "get_returns",
            "facts": {"return_percent": 12.5},
            "assumptions": {"ticker": "MSFT"},
            "source": {"tool": "get_returns", "provider": "yfinance"},
            "quality": {"cache_hit": True, "is_synthetic": False, "is_estimated": False, "missing_fields": []},
            "errors": [],
        },
        {"ticker": "MSFT"},
    )

    assert result.type == "get_returns"
    assert result.facts["return_percent"] == 12.5
    assert result.quality.cache_hit is True
    assert result.source["provider"] == "yfinance"


def test_normalize_tool_output_wraps_error_dict():
    result = normalize_tool_output("get_company_fundamentals", {"error": "upstream failed"}, {"ticker": "MSFT"})

    assert result.errors == ["upstream failed"]
    assert result.facts == {}


def test_normalize_tool_output_preserves_json_envelope_string():
    result = normalize_tool_output(
        "resolve_financial_entity",
        (
            '{"type":"resolve_financial_entity","facts":{"ticker":"MSFT"},'
            '"source":{"tool":"resolve_financial_entity","provider":"yfinance"},'
            '"quality":{"cache_hit":false,"is_synthetic":false,"is_estimated":false,"missing_fields":[]},'
            '"errors":[]}'
        ),
        {"identifier": "MSFT"},
    )

    assert result.type == "resolve_financial_entity"
    assert result.facts["ticker"] == "MSFT"
    assert result.quality.cache_hit is False
