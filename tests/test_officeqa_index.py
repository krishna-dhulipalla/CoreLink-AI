import json

from engine.agent.benchmarks.officeqa_index import (
    build_officeqa_index,
    resolve_source_files_to_manifest,
    search_officeqa_corpus_index,
    validate_officeqa_index,
)
from engine.agent.benchmarks.officeqa_runtime import OfficeQACorpusBootstrapError, verify_officeqa_corpus_bundle
from engine.agent.retrieval_tools import (
    _OfficeQATableExtractionTimeout,
    _classify_table_family,
    _extract_tables_from_html_string,
    _rank_tables,
    _table_candidate_matches_query,
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
            "source_files_policy": "hard",
        }
    )

    assert result["source_files_filter_applied"] is True
    assert result["source_files_policy"] == "hard"
    assert result["results"][0]["document_id"] == "treasury_1940_json"


def test_search_officeqa_corpus_index_treats_source_files_as_soft_prior_by_default(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1940_01.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Current monthly statement",
                            "bbox": [{"page_id": 7}],
                            "content": (
                                "<table>"
                                "<tr><th>Month</th><th>Receipts</th></tr>"
                                "<tr><td>January</td><td>10</td></tr>"
                                "<tr><td>February</td><td>11</td></tr>"
                                "</table>"
                            ),
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_bulletin_1941_11.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "section_header", "content": "Summary of fiscal statistics"},
                        {
                            "type": "table",
                            "description": "Summary of expenditures for calendar year 1940",
                            "bbox": [{"page_id": 29}],
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>Calendar year 1940</th></tr>"
                                "<tr><td>U.S. national defense</td><td>4748</td></tr>"
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

    result = search_officeqa_corpus_index(
        "What were the total expenditures for U.S. national defense in the calendar year 1940?",
        corpus_root=corpus_root,
        source_files=["treasury_bulletin_1940_01.json"],
        source_files_policy="soft",
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1941", "1940", "1939"],
        period_type="calendar_year",
        granularity_requirement="calendar_year",
        entity="U.S. national defense",
        metric="total expenditures",
        top_k=2,
    )

    assert result["source_files_filter_applied"] is False
    assert result["source_files_prior_applied"] is True
    assert result["source_files_policy"] == "soft"
    assert result["results"][0]["document_id"] == "treasury_bulletin_1941_11_json"


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


def test_search_officeqa_corpus_index_prefers_temporal_neighbor_publication_with_stronger_table_unit(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1940_01.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Current monthly statement",
                            "bbox": [{"page_id": 7}],
                            "content": (
                                "<table>"
                                "<tr><th>Month</th><th>Receipts</th></tr>"
                                "<tr><td>January</td><td>10</td></tr>"
                                "<tr><td>February</td><td>11</td></tr>"
                                "</table>"
                            ),
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_bulletin_1941_11.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Summary of expenditures for calendar year 1940",
                            "bbox": [{"page_id": 29}],
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>Calendar year 1940</th></tr>"
                                "<tr><td>U.S. national defense</td><td>4748</td></tr>"
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

    result = search_officeqa_corpus_index(
        "What were the total expenditures for U.S. national defense in the calendar year 1940?",
        corpus_root=corpus_root,
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1941", "1940", "1939"],
        period_type="calendar_year",
        granularity_requirement="calendar_year",
        entity="U.S. national defense",
        metric="total expenditures",
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_bulletin_1941_11_json"
    assert result["results"][0]["metadata"]["publication_year"] == "1941"
    assert result["results"][0]["metadata"]["best_evidence_unit"]["table_family"] in {"category_breakdown", "annual_summary"}


def test_search_officeqa_corpus_index_penalizes_debt_tables_for_expenditure_questions(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1941_07.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Debt statement",
                            "bbox": [{"page_id": 20}],
                            "content": (
                                "<table>"
                                "<tr><th>Item</th><th>Calendar year 1940</th></tr>"
                                "<tr><td>Public debt outstanding</td><td>250000</td></tr>"
                                "</table>"
                            ),
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_bulletin_1941_10.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Expenditure summary",
                            "bbox": [{"page_id": 29}],
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>Calendar year 1940</th></tr>"
                                "<tr><td>U.S. national defense</td><td>4748</td></tr>"
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

    result = search_officeqa_corpus_index(
        "What were the total expenditures for U.S. national defense in the calendar year 1940?",
        corpus_root=corpus_root,
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1941", "1940", "1939"],
        period_type="calendar_year",
        granularity_requirement="calendar_year",
        entity="U.S. national defense",
        metric="total expenditures",
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_bulletin_1941_10_json"
    assert result["results"][0]["metadata"]["best_evidence_unit"]["table_family"] in {"category_breakdown", "annual_summary"}


def test_search_officeqa_corpus_index_penalizes_mixed_heading_body_units_for_debt_questions(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1946_12.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "section_header", "content": "Public Debt and Guaranteed Obligations Outstanding"},
                        {
                            "type": "table",
                            "description": "Table 2.- Analysis of Receipts from Internal Revenue",
                            "bbox": [{"page_id": 35}],
                            "content": (
                                "<table>"
                                "<tr><th>Fiscal year or month</th><th>Total</th><th>Income taxes</th></tr>"
                                "<tr><td>1945</td><td>2001</td><td>1000</td></tr>"
                                "</table>"
                            ),
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_bulletin_1946_08.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Public debt statement",
                            "bbox": [{"page_id": 20}],
                            "content": (
                                "<table>"
                                "<tr><th>Year</th><th>Total public debt outstanding</th></tr>"
                                "<tr><td>1945</td><td>258682</td></tr>"
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

    result = search_officeqa_corpus_index(
        "According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
        corpus_root=corpus_root,
        target_years=["1945"],
        publication_year_window=["1945", "1946"],
        preferred_publication_years=["1945", "1946"],
        period_type="point_lookup",
        granularity_requirement="point_lookup",
        metric="public debt outstanding",
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_bulletin_1946_08_json"
    assert result["results"][0]["metadata"]["best_evidence_unit"]["locator"] == "Public debt statement"


def test_search_officeqa_corpus_index_prefers_target_year_debt_statement_over_later_broad_debt_summary(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1945_08.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Table 3.- Public Debt Outstanding, June 30, 1944 and 1945",
                            "bbox": [{"page_id": 20}],
                            "content": (
                                "<table>"
                                "<tr><th>Year</th><th>Public debt outstanding</th></tr>"
                                "<tr><td>1944</td><td>251000</td></tr>"
                                "<tr><td>1945</td><td>258682</td></tr>"
                                "</table>"
                            ),
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_bulletin_1946_12.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "section_header", "content": "Public Debt and Guaranteed Obligations Outstanding"},
                        {
                            "type": "table",
                            "description": "Broad debt summary",
                            "bbox": [{"page_id": 35}],
                            "content": (
                                "<table>"
                                "<tr><th>Fiscal year</th><th>Total public debt outstanding</th></tr>"
                                "<tr><td>1945</td><td>258682</td></tr>"
                                "<tr><td>1946</td><td>269422</td></tr>"
                                "</table>"
                            ),
                        },
                        {
                            "type": "paragraph",
                            "content": "Treasury Bulletin discussion of debt, receipts, expenditures, and related financing.",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)

    result = search_officeqa_corpus_index(
        "According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
        corpus_root=corpus_root,
        target_years=["1945"],
        publication_year_window=["1944", "1945", "1946"],
        preferred_publication_years=["1945", "1946"],
        period_type="point_lookup",
        granularity_requirement="point_lookup",
        metric="public debt outstanding",
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_bulletin_1945_08_json"
    assert result["results"][0]["metadata"]["best_evidence_unit"]["locator"] == "Table 3.- Public Debt Outstanding, June 30, 1944 and 1945"


def test_search_officeqa_corpus_index_prefers_structured_surface_match_over_preview_only_overlap(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1940_08.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "General expenditure summary",
                            "bbox": [{"page_id": 16}],
                            "content": (
                                "<table>"
                                "<tr><th>Fiscal year or month</th><th>Total expenditures</th></tr>"
                                "<tr><td>1940</td><td>10000</td></tr>"
                                "</table>"
                            ),
                        },
                        {
                            "type": "paragraph",
                            "content": "National defense expenditures are discussed in surrounding commentary.",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_bulletin_1941_07.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Analysis of national defense expenditures",
                            "bbox": [{"page_id": 16}],
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>Calendar year 1940</th></tr>"
                                "<tr><td>National defense</td><td>4748</td></tr>"
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

    result = search_officeqa_corpus_index(
        "What were the total expenditures for U.S. national defense in the calendar year 1940?",
        corpus_root=corpus_root,
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1941", "1940", "1939"],
        period_type="calendar_year",
        granularity_requirement="calendar_year",
        entity="U.S. national defense",
        metric="total expenditures",
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_bulletin_1941_07_json"


def test_search_officeqa_corpus_index_prefers_entity_focused_later_year_table_over_broad_summary(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1940_08.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "title", "content": "Summary Table on Receipts, Expenditures and Public Debt", "bbox": [{"page_id": 13}]},
                        {"type": "section_header", "content": "Receipts and Expenditures", "bbox": [{"page_id": 13}]},
                        {
                            "type": "table",
                            "bbox": [{"page_id": 13}],
                            "content": (
                                "<table>"
                                "<tr><th></th><th>Actual 1939</th><th>Actual 1940</th><th>Estimated 1941</th></tr>"
                                "<tr><td>Income Tax</td><td>2,189</td><td>2,125</td><td>2,923</td></tr>"
                                "<tr><td>National defense and Veterans Adm</td><td>1,720</td><td>2,116</td><td>5,565</td></tr>"
                                "<tr><td>Total Expenditures</td><td>8,707</td><td>8,998</td><td>12,058</td></tr>"
                                "</table>"
                            ),
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_bulletin_1941_03.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "section_header", "content": "Budget Receipts and Expenditures", "bbox": [{"page_id": 16}]},
                        {
                            "type": "table",
                            "description": "Table 4.- Analysis of National Defense Expenditures",
                            "bbox": [{"page_id": 16}],
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>Calendar year 1940</th></tr>"
                                "<tr><td>U.S. national defense</td><td>4,748</td></tr>"
                                "<tr><td>Veterans' Administration</td><td>557</td></tr>"
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

    result = search_officeqa_corpus_index(
        "What were the total expenditures for U.S. national defense in the calendar year 1940?",
        corpus_root=corpus_root,
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1941", "1940", "1939"],
        period_type="calendar_year",
        granularity_requirement="calendar_year",
        entity="U.S. national defense",
        metric="total expenditures",
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_bulletin_1941_03_json"
    assert result["results"][0]["metadata"]["best_evidence_unit"]["table_family"] in {"category_breakdown", "annual_summary"}


def test_search_officeqa_corpus_index_prefers_retrospective_evidence_over_late_historical_mentions(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1939_03.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "section_header", "content": "SUMMARY OF FISCAL STATISTICS", "bbox": [{"page_id": 14}]},
                        {
                            "type": "table",
                            "description": "Veterans Administration expenditures by fiscal year",
                            "bbox": [{"page_id": 14}],
                            "content": (
                                "<table>"
                                "<tr><th>Agency</th><th>FY 1934</th></tr>"
                                "<tr><td>Veterans Administration</td><td>507</td></tr>"
                                "</table>"
                            ),
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (corpus_root / "treasury_bulletin_1959_09.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {
                            "type": "table",
                            "description": "Broad historical summary",
                            "bbox": [{"page_id": 22}],
                            "content": (
                                "<table>"
                                "<tr><th>Topic</th><th>Amount</th></tr>"
                                "<tr><td>Veterans Administration historical discussion</td><td>900</td></tr>"
                                "<tr><td>Fiscal year 1934 references</td><td>12</td></tr>"
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

    result = search_officeqa_corpus_index(
        "What were the total expenditures of the Veterans Administration in FY 1934 excluding trust accounts?",
        corpus_root=corpus_root,
        target_years=["1934"],
        publication_year_window=["1939", "1940", "1941", "1942"],
        preferred_publication_years=["1939", "1940", "1941", "1942"],
        acceptable_publication_lag_years=1,
        retrospective_evidence_allowed=True,
        retrospective_evidence_required=True,
        period_type="fiscal_year",
        granularity_requirement="fiscal_year",
        entity="Veterans Administration",
        metric="total expenditures",
        top_k=2,
    )

    assert result["results"][0]["document_id"] == "treasury_bulletin_1939_03_json"
    assert result["results"][0]["metadata"]["publication_year"] == "1939"


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
    assert summary["index_schema_version"] == 4


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

    monkeypatch.setattr("engine.agent.retrieval_tools._extract_document_tables", _raise_timeout)

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
    assert table_result["tables"][0]["period_type"] == "monthly_series"
    assert table_result["tables"][0]["typing_ambiguities"] == []
    assert table_result["tables"][0]["page_locator"] == "page 18"
    assert table_result["tables"][0]["rows"][0][0] == "January"


def test_fetch_officeqa_table_projects_sibling_section_header_context(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1941_11.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "title", "content": "Bulletin of the Treasury Department, November 1941", "bbox": [{"page_id": 11}]},
                        {"type": "section_header", "content": "SUMMARY OF FISCAL STATISTICS", "bbox": [{"page_id": 11}]},
                        {"type": "text", "content": "(in millions of dollars)", "bbox": [{"page_id": 11}]},
                        {
                            "type": "table",
                            "bbox": [{"page_id": 11}],
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>1940</th></tr>"
                                "<tr><td>Total expenditures</td><td>8,998</td></tr>"
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
        {"document_id": "treasury_bulletin_1941_11_json", "table_query": "summary of fiscal statistics"}
    )

    assert table_result["metadata"]["officeqa_status"] == "ok"
    assert "summary of fiscal statistics" in table_result["tables"][0]["context_text"].lower()
    assert "SUMMARY OF FISCAL STATISTICS" in table_result["tables"][0]["document_context"]["raw_heading_chain"]


def test_fetch_officeqa_table_stitches_continued_tables_across_pages(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1941_11.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "title", "content": "Bulletin of the Treasury Department, November 1941", "bbox": [{"page_id": 12}]},
                        {"type": "section_header", "content": "Budget Receipts and Expenditures", "bbox": [{"page_id": 12}]},
                        {"type": "text", "content": "(in millions of dollars)", "bbox": [{"page_id": 12}]},
                        {
                            "type": "table",
                            "bbox": [{"page_id": 12}],
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>1940</th></tr>"
                                "<tr><td>Total expenditures</td><td>8,998</td></tr>"
                                "</table>"
                            ),
                        },
                        {"type": "footnote", "content": "(Continued on following page)", "bbox": [{"page_id": 12}]},
                        {"type": "title", "content": "Budget Receipts and Expenditures - (Continued)", "bbox": [{"page_id": 13}]},
                        {
                            "type": "table",
                            "bbox": [{"page_id": 13}],
                            "content": (
                                "<table>"
                                "<tr><td>U.S. national defense</td><td>4,748</td></tr>"
                                "<tr><td>Veterans' Administration</td><td>557</td></tr>"
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
        {
            "document_id": "treasury_bulletin_1941_11_json",
            "table_query": "u.s. national defense total expenditures 1940",
        }
    )

    assert table_result["metadata"]["officeqa_status"] == "ok"
    assert table_result["tables"][0]["headers"] == ["Row", "1940"]
    assert any(row[0] == "U.S. national defense" and row[1] == "4,748" for row in table_result["tables"][0]["rows"])
    assert table_result["tables"][0]["document_context"]["is_continuation_group"] is True


def test_search_officeqa_corpus_index_uses_stateful_context_in_best_evidence_unit(tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_bulletin_1941_11.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "title", "content": "Bulletin of the Treasury Department, November 1941", "bbox": [{"page_id": 12}]},
                        {"type": "section_header", "content": "SUMMARY OF FISCAL STATISTICS", "bbox": [{"page_id": 12}]},
                        {"type": "section_header", "content": "Budget Receipts and Expenditures", "bbox": [{"page_id": 12}]},
                        {"type": "text", "content": "(in millions of dollars)", "bbox": [{"page_id": 12}]},
                        {
                            "type": "table",
                            "bbox": [{"page_id": 12}],
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>1940</th></tr>"
                                "<tr><td>Total expenditures</td><td>8,998</td></tr>"
                                "</table>"
                            ),
                        },
                        {"type": "footnote", "content": "(Continued on following page)", "bbox": [{"page_id": 12}]},
                        {"type": "title", "content": "Budget Receipts and Expenditures - (Continued)", "bbox": [{"page_id": 13}]},
                        {
                            "type": "table",
                            "bbox": [{"page_id": 13}],
                            "content": (
                                "<table>"
                                "<tr><td>U.S. national defense</td><td>4,748</td></tr>"
                                "<tr><td>Veterans' Administration</td><td>557</td></tr>"
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

    result = search_officeqa_corpus_index(
        "What were the total expenditures for U.S. national defense in the calendar year 1940?",
        corpus_root=corpus_root,
        target_years=["1940"],
        publication_year_window=["1940", "1941", "1942"],
        preferred_publication_years=["1941", "1940", "1942"],
        period_type="calendar_year",
        granularity_requirement="calendar_year",
        entity="U.S. national defense",
        metric="total expenditures",
        top_k=1,
    )

    assert result["results"][0]["document_id"] == "treasury_bulletin_1941_11_json"
    best_unit = result["results"][0]["metadata"]["best_evidence_unit"]
    assert "summary of fiscal statistics" in best_unit["context_text"].lower()
    assert "Budget Receipts and Expenditures" in best_unit["heading_chain"]
    assert best_unit["is_continuation_group"] is True


def test_table_candidate_matches_query_uses_structural_signature_not_fixed_preview():
    filler = "<tr><td>Filler category</td><td>000</td></tr>" * 220
    content = (
        "<table>"
        "<tr><th>Category</th><th>Calendar year 1940</th></tr>"
        f"{filler}"
        "<tr><td>Postal Savings System</td><td>125</td></tr>"
        "</table>"
    )

    assert _table_candidate_matches_query(
        "table 7",
        "Summary of fiscal statistics",
        content,
        "postal savings system expenditures 1940",
        page_locator="page 18",
        unit_hint="million dollars",
    ) is True


def test_classify_table_family_generalizes_beyond_benchmark_keyword_boosts():
    table = {
        "locator": "table 4",
        "headers": ["Row", "Calendar year 1940"],
        "rows": [
            ["Postal Savings System", "125"],
            ["Interior Department", "77"],
            ["Bureau of Mines", "18"],
        ],
        "context_text": "Summary of expenditures by department",
        "heading_chain": ["Summary of Fiscal Statistics", "Budget Expenditures"],
        "unit_hint": "million dollars",
        "canonical_table": {
            "row_records": [
                {"row_type": "data", "row_path": ["Postal Savings System"], "row_label": "Postal Savings System"},
                {"row_type": "data", "row_path": ["Interior Department"], "row_label": "Interior Department"},
                {"row_type": "data", "row_path": ["Bureau of Mines"], "row_label": "Bureau of Mines"},
            ],
            "normalization_metrics": {"header_data_separation_quality": 0.88},
        },
    }

    family, confidence = _classify_table_family(table)

    assert family == "category_breakdown"
    assert confidence >= 0.6


def test_rank_tables_prefers_generic_structural_fit_without_manual_category_boosts():
    query = "What were the total expenditures for Postal Savings System in the calendar year 1940?"
    ranked = _rank_tables(
        [
            {
                "locator": "table receipts",
                "headers": ["Row", "Calendar year 1940"],
                "rows": [["Postal Savings receipts", "800"]],
                "context_text": "Summary of receipts",
                "heading_chain": ["Budget Receipts"],
                "unit_hint": "million dollars",
                "canonical_table": {
                    "row_records": [
                        {"row_type": "data", "row_path": ["Postal Savings receipts"], "row_label": "Postal Savings receipts"},
                    ],
                    "normalization_metrics": {"header_data_separation_quality": 0.81},
                },
            },
            {
                "locator": "table expenditures",
                "headers": ["Row", "Calendar year 1940"],
                "rows": [["Postal Savings System", "125"], ["Interior Department", "77"]],
                "context_text": "Summary of expenditures by department",
                "heading_chain": ["Budget Expenditures"],
                "unit_hint": "million dollars",
                "canonical_table": {
                    "row_records": [
                        {"row_type": "data", "row_path": ["Postal Savings System"], "row_label": "Postal Savings System"},
                        {"row_type": "data", "row_path": ["Interior Department"], "row_label": "Interior Department"},
                    ],
                    "normalization_metrics": {"header_data_separation_quality": 0.9},
                },
            },
        ],
        query,
    )

    assert ranked[0]["locator"] == "table expenditures"
    assert ranked[0]["ranking_score"] > ranked[1]["ranking_score"]


def test_rank_tables_prefers_exact_phrase_alignment_over_generic_token_volume():
    query = "Treasury Bulletin national defense expenditures 1940 monthly series"
    generic_rows = [[f"Generic treasury bulletin expenditures row {idx}", str(100 + idx), str(120 + idx)] for idx in range(24)]
    ranked = _rank_tables(
        [
            {
                "locator": "table generic",
                "headers": ["Row", "January 1940", "February 1940"],
                "rows": generic_rows,
                "context_text": "Treasury Bulletin expenditures monthly series summary table",
                "heading_chain": ["Treasury Bulletin", "Monthly expenditures summary"],
                "unit_hint": "million dollars",
                "canonical_table": {
                    "row_records": [
                        {"row_type": "data", "row_path": [f"Generic treasury expenditures row {idx}"], "row_label": f"Generic treasury expenditures row {idx}"}
                        for idx in range(12)
                    ],
                    "normalization_metrics": {"header_data_separation_quality": 0.82},
                },
            },
            {
                "locator": "table exact",
                "headers": ["Row", "January 1940", "February 1940"],
                "rows": [["National defense", "210", "225"], ["Postal service", "120", "123"]],
                "context_text": "Treasury Bulletin monthly expenditures by category",
                "heading_chain": ["Treasury Bulletin", "Monthly expenditures by category"],
                "unit_hint": "million dollars",
                "canonical_table": {
                    "row_records": [
                        {"row_type": "data", "row_path": ["National defense"], "row_label": "National defense"},
                        {"row_type": "data", "row_path": ["Postal service"], "row_label": "Postal service"},
                    ],
                    "normalization_metrics": {"header_data_separation_quality": 0.9},
                },
            },
        ],
        query,
    )

    assert ranked[0]["locator"] == "table exact"


def test_fetch_officeqa_table_exposes_structural_candidates_for_table_rerank_llm(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_generic_1940.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"type": "section_header", "content": "Summary of Fiscal Statistics"},
                        {
                            "type": "table",
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>Calendar year 1940</th></tr>"
                                "<tr><td>Postal Savings System</td><td>125</td></tr>"
                                "<tr><td>Interior Department</td><td>77</td></tr>"
                                "</table>"
                            ),
                        },
                        {
                            "type": "table",
                            "content": (
                                "<table>"
                                "<tr><th>Category</th><th>Calendar year 1940</th></tr>"
                                "<tr><td>Postal Savings receipts</td><td>800</td></tr>"
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

    result = fetch_officeqa_table.invoke(
        {
            "document_id": "treasury_generic_1940_json",
            "table_query": "Postal Savings System total expenditures 1940",
        }
    )

    candidate = result["metadata"]["table_candidates"][0]
    assert candidate["structural_signature"]
    assert "Postal Savings System" in candidate["structural_signature"]
    assert "period_type" in candidate
    assert "table_confidence" in candidate
