from agent.benchmarks.officeqa_eval import (
    build_case_report,
    capture_officeqa_artifacts,
    classify_officeqa_trace,
    summarize_regression_report,
)
from test_utils import make_state


def _trace(state: dict, answer: str) -> dict:
    return {"final_state": state, "answer": answer}


def test_classify_officeqa_trace_flags_formatting_failure():
    state = make_state(
        "OfficeQA task",
        answer_contract={
            "format": "xml",
            "requires_adapter": True,
            "xml_root_tag": "FINAL_ANSWER",
            "value_rules": {"reasoning_tag": "REASONING", "final_answer_tag": "FINAL_ANSWER"},
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    classification = classify_officeqa_trace(_trace(state, "40.90"))

    assert classification["subsystem"] == "formatting"


def test_classify_officeqa_trace_flags_extraction_failure():
    state = make_state(
        "OfficeQA task",
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {"metadata": {"officeqa_status": "missing_table"}},
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
    )

    classification = classify_officeqa_trace(_trace(state, "<REASONING>x</REASONING><FINAL_ANSWER>Insufficient data</FINAL_ANSWER>"))

    assert classification["subsystem"] == "extraction"


def test_classify_officeqa_trace_flags_compute_failure():
    state = make_state(
        "OfficeQA task",
        answer_contract={
            "format": "xml",
            "requires_adapter": True,
            "xml_root_tag": "FINAL_ANSWER",
            "value_rules": {"reasoning_tag": "REASONING", "final_answer_tag": "FINAL_ANSWER"},
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        curated_context={
            "structured_evidence": {"values": [{"document_id": "treasury_1940_json"}], "value_count": 1},
            "compute_result": {"status": "insufficient", "validation_errors": ["Missing complete monthly coverage."]},
        },
    )

    classification = classify_officeqa_trace(_trace(state, "<REASONING>x</REASONING><FINAL_ANSWER>Insufficient data</FINAL_ANSWER>"))

    assert classification["subsystem"] == "compute"


def test_capture_officeqa_artifacts_collects_tables_and_ledger():
    state = make_state(
        "OfficeQA task",
        benchmark_overrides={
            "benchmark_adapter": "officeqa",
            "source_files_found": [{"document_id": "treasury_1940_json", "relative_path": "treasury_1940.json", "matched": True}],
        },
        source_bundle={
            "source_files_expected": ["treasury_1940.json"],
            "source_files_found": [{"document_id": "treasury_1940_json", "relative_path": "treasury_1940.json", "matched": True}],
        },
        curated_context={
            "structured_evidence": {
                "tables": [
                    {
                        "document_id": "treasury_1940_json",
                        "page_locator": "page 17",
                        "table_locator": "table 1",
                        "headers": ["Month", "Expenditures"],
                        "row_count": 12,
                        "column_count": 2,
                        "unit": "million dollars",
                    }
                ],
                "value_count": 12,
            },
            "compute_result": {
                "status": "ok",
                "ledger": [
                    {
                        "operator": "monthly_sum",
                        "description": "1940 monthly sum",
                        "output": {"value": 2602.0},
                    }
                ],
            },
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {
                        "document_id": "treasury_1940_json",
                        "citation": "treasury_1940.json#page=17",
                        "metadata": {"officeqa_status": "ok", "relative_path": "treasury_1940.json"},
                    },
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": ["treasury_1940.json#page=17"],
            "final_artifact_signature": "sig-1",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
    )

    artifacts = capture_officeqa_artifacts(_trace(state, "<REASONING>x</REASONING><FINAL_ANSWER>2602</FINAL_ANSWER>"))

    assert artifacts["source_files_expected"] == ["treasury_1940.json"]
    assert artifacts["structured_value_count"] == 12
    assert artifacts["extracted_tables"][0]["document_id"] == "treasury_1940_json"
    assert artifacts["compute_ledger"][0]["operator"] == "monthly_sum"
    assert artifacts["final_artifact_signature"] == "sig-1"


def test_summarize_regression_report_sets_go_no_go_threshold():
    good = {
        "classification": {"subsystem": "pass"},
        "artifacts": {"extracted_tables": [{"document_id": "x"}], "final_answer": "42"},
    }
    bad = {
        "classification": {"subsystem": "routing"},
        "artifacts": {"extracted_tables": [], "final_answer": ""},
    }

    summary = summarize_regression_report([good, good, bad])

    assert summary["counts_by_subsystem"]["routing"] == 1
    assert summary["go_for_full_benchmark"] is False
    assert summary["required_evidence_ready_cases"] == 2


def test_build_case_report_includes_classification_and_artifacts():
    state = make_state(
        "OfficeQA task",
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        task_intent={"task_family": "document_qa", "execution_mode": "document_grounded_analysis"},
        curated_context={"structured_evidence": {"values": [{"document_id": "treasury_1940_json"}], "value_count": 1}},
        execution_journal={
            "events": [],
            "tool_results": [
                {"type": "fetch_officeqa_table", "facts": {"metadata": {"officeqa_status": "ok"}, "document_id": "treasury_1940_json"}}
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
    )

    report = build_case_report(
        {"id": "case_1", "prompt": "OfficeQA task", "focus_subsystem": "routing", "smoke": True},
        _trace(state, "<REASONING>x</REASONING><FINAL_ANSWER>40.90</FINAL_ANSWER>"),
    )

    assert report["id"] == "case_1"
    assert report["classification"]["subsystem"] == "pass"
    assert report["execution_summary"]["execution_mode"] == "document_grounded_analysis"
