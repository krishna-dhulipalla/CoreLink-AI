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


def test_normalize_tool_output_infers_retrieval_status_for_structured_fetch_dict():
    result = normalize_tool_output(
        "fetch_corpus_document",
        {
            "document_id": "treasury_1945_txt",
            "citation": "treasury_1945.txt",
            "metadata": {"file_name": "treasury_1945.txt", "format": "txt", "window": "chunks 1-1"},
            "chunks": [{"locator": "chunk 1", "kind": "text_excerpt", "text": "Total public debt outstanding in 1945 was 258.7 billion dollars."}],
            "tables": [],
            "numeric_summaries": [{"metric": "public_debt", "value": 258.7}],
        },
        {"document_id": "treasury_1945_txt"},
    )

    assert result.retrieval_status == "ok"
    assert result.evidence_quality_score > 0.0


def test_normalize_tool_output_marks_garbled_binary_reference_payload():
    raw = (
        "FILE: treasury_1945.bin\n"
        "FORMAT: BINARY | SIZE: 12.0 KB\n"
        "STATUS: GARBLED_BINARY\n"
        "ERROR: Binary payload was detected but could not be parsed as PDF, Office, CSV, JSON, or text.\n"
        "--------------------------------------------------\n"
        "Binary payload was detected but could not be parsed as PDF, Office, CSV, JSON, or text."
    )

    result = normalize_tool_output("fetch_reference_file", raw, {"url": "https://example.com/treasury_1945.bin"})

    assert result.retrieval_status == "garbled_binary"
    assert result.facts["metadata"]["status"] == "garbled_binary"
    assert result.evidence_quality_score <= 0.05
