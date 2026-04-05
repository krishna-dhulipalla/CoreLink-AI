import json

from agent.benchmarks.officeqa_index import (
    build_officeqa_index,
    resolve_source_files_to_manifest,
    search_officeqa_corpus_index,
    validate_officeqa_index,
)
from agent.benchmarks.officeqa_runtime import OfficeQACorpusBootstrapError, verify_officeqa_corpus_bundle
from agent.retrieval_tools import (
    _OfficeQATableExtractionTimeout,
    _extract_tables_from_html_string,
    fetch_officeqa_table,
    lookup_officeqa_cells,
    search_reference_corpus,
)


def test_build_officeqa_index_persists_manifest_and_metadata(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "page": 17,
                "section_title": "National Defense",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "100.0"], ["February", "101.5"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "notes.txt").write_text("Reference notes for an unrelated file.", encoding="utf-8")

    summary = build_officeqa_index(corpus_root=corpus_root)

    assert summary["document_count"] == 2
    manifest_path = corpus_root / ".officeqa_index" / "manifest.jsonl"
    metadata_path = corpus_root / ".officeqa_index" / "index_metadata.json"
    assert manifest_path.exists()
    assert metadata_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "national defense" in manifest_text.lower()
    assert "million dollars" in manifest_text.lower()
    assert '"years": ["1940"]' in manifest_text
    assert '"normalized_numeric_values"' in manifest_text


def test_search_reference_corpus_uses_officeqa_index_when_available(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "section_title": "National Defense",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "100.0"], ["February", "101.5"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_1953.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1953",
                "section_title": "Agriculture",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "50.0"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    result = search_reference_corpus.invoke({"query": "Treasury Bulletin 1940 national defense expenditures", "top_k": 2})

    assert result["index_mode"] == "officeqa_manifest"
    assert result["results"][0]["document_id"] == "treasury_1940_json"
    assert "1940" in result["results"][0]["metadata"]["years"]


def test_search_reference_corpus_can_filter_by_source_files(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps({"title": "Treasury Bulletin 1940", "section_title": "National Defense"}),
        encoding="utf-8",
    )
    (corpus_root / "treasury_1953.json").write_text(
        json.dumps({"title": "Treasury Bulletin 1953", "section_title": "Agriculture"}),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    result = search_reference_corpus.invoke(
        {
            "query": "Treasury Bulletin 1940",
            "top_k": 5,
            "source_files": ["treasury_1940.json"],
        }
    )

    assert result["source_files_filter_applied"] is True
    assert result["results"][0]["document_id"] == "treasury_1940_json"


def test_resolve_source_files_to_manifest_matches_relative_and_stem_names(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "Treasury_1940_National_Defense.json").write_text(
        json.dumps({"title": "Treasury Bulletin 1940"}),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)

    matches = resolve_source_files_to_manifest(
        ["Treasury_1940_National_Defense.json", "Treasury_1940_National_Defense"],
        corpus_root=corpus_root,
    )

    assert matches[0]["matched"] is True
    assert matches[1]["matched"] is True
    assert matches[0]["document_id"] == matches[1]["document_id"]


def test_search_officeqa_corpus_index_ranks_metadata_backed_match_first(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "section_title": "National Defense",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "100.0"]],
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "misc_1940.txt").write_text("General 1940 notes without the right section.", encoding="utf-8")
    build_officeqa_index(corpus_root=corpus_root)

    result = search_officeqa_corpus_index(
        "Treasury Bulletin 1940 national defense expenditures",
        corpus_root=corpus_root,
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_1940_json"


def test_validate_officeqa_index_reports_partially_parsed_documents(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "empty.pdf").write_bytes(b"%PDF-1.4\n%empty")
    build_officeqa_index(corpus_root=corpus_root)

    report = validate_officeqa_index(corpus_root=corpus_root)

    assert report["issue_count"] >= 1
    assert any(issue["flag"] == "pdf_extract_failed" for issue in report["issues"])


def test_verify_officeqa_corpus_bundle_succeeds_with_built_index(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps({"title": "Treasury Bulletin 1940", "section_title": "National Defense"}),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)

    summary = verify_officeqa_corpus_bundle(corpus_root=corpus_root)

    assert summary["document_count"] == 1
    assert summary["index_schema_version"] == 1


def test_verify_officeqa_corpus_bundle_requires_manifest_metadata(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps({"title": "Treasury Bulletin 1940"}),
        encoding="utf-8",
    )

    try:
        verify_officeqa_corpus_bundle(corpus_root=corpus_root)
    except OfficeQACorpusBootstrapError as exc:
        assert "requires a built corpus index" in str(exc)
    else:
        raise AssertionError("Expected OfficeQACorpusBootstrapError when manifest metadata is missing.")


def test_officeqa_table_and_cell_lookup_tools_extract_structured_values(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "section_title": "National Defense",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "100.0"], ["February", "101.5"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    table_result = fetch_officeqa_table.invoke({"document_id": "treasury_1940_json", "table_query": "national defense expenditures"})
    cell_result = lookup_officeqa_cells.invoke(
        {
            "document_id": "treasury_1940_json",
            "table_query": "national defense expenditures",
            "row_query": "February",
            "column_query": "Expenditures",
        }
    )

    assert table_result["metadata"]["officeqa_status"] == "ok"
    assert table_result["tables"][0]["headers"][0] == "Row"
    assert cell_result["metadata"]["officeqa_status"] == "ok"
    assert cell_result["cells"][0]["value"] == "101.5"


def test_officeqa_table_lookup_extracts_html_tables_from_parsed_json(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1945.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Public debt table",
                            "bbox": [{"page_id": 23}],
                            "content": (
                                "<table>"
                                "<tr><th>Year</th><th>Total public debt outstanding</th></tr>"
                                "<tr><td>1944</td><td>232,000</td></tr>"
                                "<tr><td>1945</td><td>278,000</td></tr>"
                                "</table>"
                            ),
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    table_result = fetch_officeqa_table.invoke(
        {"document_id": "treasury_1945_json", "table_query": "total public debt outstanding 1945"}
    )

    assert table_result["metadata"]["officeqa_status"] == "ok"
    assert table_result["tables"]
    assert table_result["tables"][0]["headers"] == ["Row", "Total public debt outstanding"]
    assert table_result["tables"][0]["rows"][1] == ["1945", "278,000"]
    assert table_result["tables"][0]["page_locator"] == "page 23"
    assert table_result["tables"][0]["canonical_table"]["column_paths"][1] == ["Total public debt outstanding"]
    assert table_result["tables"][0]["canonical_table"]["row_records"][1]["row_path"] == ["1945"]


def test_officeqa_tools_accept_relative_corpus_env_paths(monkeypatch, tmp_path):
    corpus_root = tmp_path / "data" / "officeqa" / "jsons"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "headers": ["Year", "Total expenditures"],
                "rows": [["1939", "450"], ["1940", "475"]],
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", "data/officeqa/jsons")

    table_result = fetch_officeqa_table.invoke({"document_id": "treasury_1940_json", "table_query": "total expenditures 1940"})

    assert table_result["metadata"]["officeqa_status"] == "ok"
    assert table_result["citation"] == "treasury_1940.json"
    assert "canonical_table" in table_result["tables"][0]


def test_extract_tables_from_html_string_handles_rowspans_without_hanging():
    html_table = (
        "<table>"
        "<tr><th rowspan='2'>Category</th><th rowspan='3'>Year</th><th>Value</th></tr>"
        "<tr><td>1945</td></tr>"
        "<tr><td>Public debt</td><td>278,000</td></tr>"
        "</table>"
    )

    extracted = _extract_tables_from_html_string(
        html_table,
        "treasury_1945.json",
        locator="Public debt table",
    )

    assert extracted
    assert extracted[0]["headers"][0] == "Row"


def test_fetch_officeqa_table_surfaces_timeout_as_structured_status(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    source_path = corpus_root / "treasury_1945.json"
    source_path.write_text(json.dumps({"title": "Treasury Bulletin 1945"}), encoding="utf-8")
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    def _raise_timeout(*args, **kwargs):
        raise _OfficeQATableExtractionTimeout("simulated extraction timeout")

    monkeypatch.setattr("agent.retrieval_tools._extract_document_tables", _raise_timeout)

    result = fetch_officeqa_table.invoke({"document_id": "treasury_1945_json", "table_query": "public debt"})

    assert result["metadata"]["officeqa_status"] == "table_timeout"
    assert "simulated extraction timeout" in result["error"]


def test_fetch_officeqa_table_prefers_analytical_table_over_contents_table(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1945.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Cumulative Table of Contents",
                            "bbox": [{"page_id": 6}],
                            "content": (
                                "<table>"
                                "<tr><th>Articles</th><th>Issue and page number</th></tr>"
                                "<tr><td>Public debt and guaranteed obligations outstanding</td><td>3</td></tr>"
                                "</table>"
                            ),
                        },
                        {
                            "type": "table",
                            "description": "Public debt statement",
                            "bbox": [{"page_id": 29}],
                            "content": (
                                "<table>"
                                "<tr><th>End of fiscal years, 1941 to 1945</th><th>1945</th></tr>"
                                "<tr><td>Total public debt outstanding</td><td>258682</td></tr>"
                                "</table>"
                            ),
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    table_result = fetch_officeqa_table.invoke(
        {"document_id": "treasury_1945_json", "table_query": "total public debt outstanding 1945"}
    )

    assert table_result["metadata"]["officeqa_status"] == "ok"
    assert table_result["tables"][0]["page_locator"] == "page 29"
    assert table_result["tables"][0]["rows"][0][1] == "258682"


def test_fetch_officeqa_table_prefers_monthly_series_over_annual_summary(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1953.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Annual summary",
                            "bbox": [{"page_id": 4}],
                            "content": (
                                "<table>"
                                "<tr><th>Total 9/</th><th>National defense and related activities</th></tr>"
                                "<tr><td>1953</td><td>900</td></tr>"
                                "</table>"
                            ),
                        },
                        {
                            "type": "table",
                            "description": "Receipts, expenditures, and balances by month",
                            "bbox": [{"page_id": 18}],
                            "content": (
                                "<table>"
                                "<tr><th>Month</th><th>Expenditures</th></tr>"
                                "<tr><td>January</td><td>100</td></tr>"
                                "<tr><td>February</td><td>101</td></tr>"
                                "<tr><td>March</td><td>102</td></tr>"
                                "<tr><td>April</td><td>103</td></tr>"
                                "</table>"
                            ),
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    table_result = fetch_officeqa_table.invoke(
        {"document_id": "treasury_1953_json", "table_query": "monthly expenditures 1953"}
    )

    assert table_result["metadata"]["officeqa_status"] == "ok"
    assert table_result["tables"][0]["table_family"] == "monthly_series"
    assert table_result["tables"][0]["page_locator"] == "page 18"
    assert table_result["tables"][0]["rows"][0][0] == "January"
