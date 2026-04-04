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
                "structure_confidence_summary": {
                    "min_confidence": 0.82,
                    "avg_confidence": 0.91,
                    "max_confidence": 0.97,
                    "low_confidence_value_count": 0,
                    "low_confidence_table_count": 0,
                    "table_confidence_gate_passed": True,
                },
                "value_count": 12,
            },
            "provenance_summary": {
                "retrieval_plan": {
                    "answer_mode": "hybrid_grounded",
                    "compute_policy": "preferred",
                },
                "retrieval_diagnostics": {
                    "retrieval_decision": {"tool_name": "fetch_officeqa_table", "strategy": "table_first"},
                    "strategy_reason": "primary metric is expected to be recoverable from structured table evidence",
                    "candidate_sources": [{"document_id": "treasury_1940_json", "score": 1.0}],
                    "rejected_candidates": [{"document_id": "treasury_1939_json", "reason": "lower-ranked than the selected candidates"}],
                },
                "evidence_plan_check": {"predictive_gaps": ["missing month coverage"]},
            },
            "compute_result": {
                "status": "ok",
                "selection_reasoning": "Selected monthly-sum compute because the task asks for a within-year monthly aggregation over 1940.",
                "rejected_alternatives": ["point_lookup rejected because the task requires aggregation or comparison, not a single isolated value"],
                "ledger": [
                    {
                        "operator": "monthly_sum",
                        "description": "1940 monthly sum",
                        "output": {"value": 2602.0},
                    }
                ],
            },
        },
        workpad={"solver_llm_decision": {"used_llm": False, "reason": "deterministic_compute_completed"}},
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
        review_packet={
            "validator_result": {
                "remediation_codes": ["RETRIEVE_EXACT_PERIOD"],
                "remediation_guidance": ["Re-extract evidence for the exact requested years."],
                "orchestration_strategy": "table_compute",
                "retry_allowed": True,
                "retry_stop_reason": "",
            }
        },
    )

    artifacts = capture_officeqa_artifacts(_trace(state, "<REASONING>x</REASONING><FINAL_ANSWER>2602</FINAL_ANSWER>"))

    assert artifacts["source_files_expected"] == ["treasury_1940.json"]
    assert artifacts["structured_value_count"] == 12
    assert artifacts["retrieval_decision"]["tool_name"] == "fetch_officeqa_table"
    assert artifacts["answer_mode"] == "hybrid_grounded"
    assert artifacts["compute_policy"] == "preferred"
    assert artifacts["strategy_reason"]
    assert artifacts["candidate_sources"]
    assert artifacts["rejected_candidates"]
    assert artifacts["evidence_gaps"] == ["missing month coverage"]
    assert artifacts["structure_confidence_summary"]["table_confidence_gate_passed"] is True
    assert artifacts["compute_selection_reasoning"]
    assert artifacts["rejected_aggregation_alternatives"]
    assert artifacts["validator_codes"] == ["RETRIEVE_EXACT_PERIOD"]
    assert artifacts["orchestration_strategy"] == "table_compute"
    assert artifacts["retry_allowed"] is True
    assert artifacts["validator_remediation"]
    assert artifacts["extracted_tables"][0]["document_id"] == "treasury_1940_json"
    assert artifacts["compute_ledger"][0]["operator"] == "monthly_sum"
    assert artifacts["solver_llm_decision"]["reason"] == "deterministic_compute_completed"
    assert artifacts["final_artifact_signature"] == "sig-1"


def test_summarize_regression_report_sets_go_no_go_threshold():
    good = {
        "case_kind": "qa",
        "classification": {"subsystem": "pass"},
        "retrieval_strategy": "table_first",
        "answer_mode": "deterministic_compute",
        "artifacts": {
            "chosen_sources": [{"document_id": "x"}],
            "extracted_tables": [{"document_id": "x"}],
            "final_answer": "42",
            "compute_policy": "required",
            "structure_confidence_summary": {"table_confidence_gate_passed": True},
        },
    }
    bad = {
        "case_kind": "routing_check",
        "classification": {"subsystem": "routing"},
        "retrieval_strategy": "hybrid",
        "answer_mode": "grounded_synthesis",
        "artifacts": {"extracted_tables": [], "final_answer": ""},
    }

    summary = summarize_regression_report([good, good, bad])

    assert summary["counts_by_subsystem"]["routing"] == 1
    assert summary["counts_by_strategy"]["table_first"] == 2
    assert summary["counts_by_strategy"]["hybrid"] == 1
    assert summary["counts_by_case_kind"]["routing_check"] == 1
    assert summary["counts_by_answer_mode"]["deterministic_compute"] == 2
    assert summary["counts_by_answer_mode"]["grounded_synthesis"] == 1
    assert summary["go_for_full_benchmark"] is False
    assert summary["evidence_ready_cases"] == 2
    assert summary["required_evidence_ready_cases"] == 2


def test_summarize_regression_report_blocks_on_validation_failures():
    validation_fail = {
        "case_kind": "qa",
        "classification": {"subsystem": "validation"},
        "retrieval_strategy": "table_first",
        "answer_mode": "hybrid_grounded",
        "artifacts": {
            "chosen_sources": [{"document_id": "x"}],
            "extracted_tables": [{"document_id": "x"}],
            "final_answer": "<FINAL_ANSWER>42</FINAL_ANSWER>",
            "compute_policy": "preferred",
            "structure_confidence_summary": {"table_confidence_gate_passed": True},
        },
    }

    summary = summarize_regression_report([validation_fail, validation_fail])

    assert summary["counts_by_subsystem"]["validation"] == 2
    assert summary["evidence_ready_cases"] == 0
    assert summary["go_for_full_benchmark"] is False


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
    state["retrieval_intent"] = {"strategy": "hybrid", "answer_mode": "hybrid_grounded"}

    report = build_case_report(
        {"id": "case_1", "prompt": "OfficeQA task", "focus_subsystem": "routing", "retrieval_strategy": "hybrid", "smoke": True, "case_kind": "routing_check"},
        _trace(state, "<REASONING>x</REASONING><FINAL_ANSWER>40.90</FINAL_ANSWER>"),
    )

    assert report["id"] == "case_1"
    assert report["case_kind"] == "routing_check"
    assert report["classification"]["subsystem"] == "pass"
    assert report["execution_summary"]["execution_mode"] == "document_grounded_analysis"
    assert report["execution_summary"]["retrieval_strategy"] == "hybrid"
    assert report["execution_summary"]["answer_mode"] == "hybrid_grounded"
