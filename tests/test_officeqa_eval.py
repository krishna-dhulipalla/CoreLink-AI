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
        workpad={
            "solver_llm_decision": {"used_llm": False, "reason": "deterministic_compute_completed"},
            "officeqa_llm_usage": [
                {
                    "category": "semantic_plan_llm",
                    "used": True,
                    "reason": "semantic_plan_llm",
                    "model_name": "semantic-plan-model",
                    "applied": True,
                }
            ],
            "officeqa_llm_repair_history": [
                {
                    "stage": "validator_repair",
                    "trigger": "table_compute",
                    "path_changed": True,
                    "decision": {"decision": "rewrite_query", "confidence": 0.92},
                }
            ],
            "officeqa_evidence_review": {
                "status": "ready",
                "predictive_gaps": [],
                "compute_policy": "preferred",
                "answer_mode": "hybrid_grounded",
                "strategy": "table_first",
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
    state["retrieval_intent"] = {
        "answer_mode": "hybrid_grounded",
        "compute_policy": "preferred",
        "semantic_plan": {
            "entity": "National defense",
            "metric": "total expenditures",
            "period": "1940",
            "granularity_requirement": "monthly_series",
            "confidence": 0.88,
            "used_llm": True,
            "model_name": "semantic-plan-model",
        },
    }

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
    assert artifacts["semantic_plan"]["used_llm"] is True
    assert artifacts["llm_usage"][0]["category"] == "semantic_plan_llm"
    assert artifacts["llm_repair_history"][0]["stage"] == "validator_repair"
    assert artifacts["repair_failures"] == []
    assert artifacts["evidence_review"]["status"] == "ready"
    assert artifacts["final_artifact_signature"] == "sig-1"


def test_summarize_regression_report_sets_go_no_go_threshold():
    good = {
        "case_kind": "qa",
        "classification": {"subsystem": "pass"},
        "retrieval_strategy": "table_first",
        "answer_mode": "deterministic_compute",
        "benchmark_analysis": {"semantic_verdict": "pass", "tags": [], "source_ranking_correct": True},
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
        "benchmark_analysis": {"semantic_verdict": "unknown", "tags": [], "source_ranking_correct": None},
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
        "benchmark_analysis": {"semantic_verdict": "fail", "tags": ["incomplete_evidence", "repair_stall"], "source_ranking_correct": True},
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


def test_summarize_regression_report_blocks_false_internal_passes_with_semantic_issues():
    false_pass = {
        "case_kind": "qa",
        "classification": {"subsystem": "pass", "compute_status": "ok"},
        "retrieval_strategy": "table_first",
        "answer_mode": "deterministic_compute",
        "benchmark_analysis": {"semantic_verdict": "fail", "tags": ["false_semantic_pass", "wrong_row_or_column_semantics"], "source_ranking_correct": True},
        "artifacts": {
            "chosen_sources": [{"document_id": "x"}],
            "extracted_tables": [{"document_id": "x"}],
            "final_answer": "4748",
            "compute_policy": "required",
            "structure_confidence_summary": {"table_confidence_gate_passed": True},
            "semantic_diagnostics": {"admissibility_passed": False, "issues": ["wrong row family"]},
        },
    }
    good = {
        "case_kind": "qa",
        "classification": {"subsystem": "pass", "compute_status": "ok"},
        "retrieval_strategy": "table_first",
        "answer_mode": "deterministic_compute",
        "benchmark_analysis": {"semantic_verdict": "pass", "tags": [], "source_ranking_correct": True},
        "artifacts": {
            "chosen_sources": [{"document_id": "y"}],
            "extracted_tables": [{"document_id": "y"}],
            "final_answer": "251286",
            "compute_policy": "required",
            "structure_confidence_summary": {"table_confidence_gate_passed": True},
            "semantic_diagnostics": {"admissibility_passed": True, "issues": []},
        },
    }

    summary = summarize_regression_report([good, false_pass])

    assert summary["semantic_compute_pass_cases"] == 1
    assert summary["compute_reliable_cases"] == 1
    assert summary["false_semantic_pass_cases"] == 1
    assert summary["counts_by_benchmark_failure"]["false_semantic_pass"] == 1
    assert summary["go_for_full_benchmark"] is False


def test_build_case_report_captures_benchmark_failure_taxonomy_and_source_expectations():
    state = make_state(
        "OfficeQA task",
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        task_intent={"task_family": "document_qa", "execution_mode": "document_grounded_analysis"},
        curated_context={
            "structured_evidence": {
                "tables": [
                    {
                        "document_id": "treasury_1945_json",
                        "page_locator": "page 3",
                        "table_locator": "table 1",
                        "headers": ["Issue and page number", "Description"],
                        "row_count": 10,
                        "column_count": 2,
                        "unit": "",
                    }
                ],
                "values": [{"document_id": "treasury_1945_json"}],
                "value_count": 1,
            },
            "compute_result": {
                "status": "ok",
                "semantic_diagnostics": {"admissibility_passed": False, "issues": ["wrong row family"]},
            },
            "provenance_summary": {
                "retrieval_diagnostics": {
                    "candidate_sources": [{"document_id": "treasury_1945_json", "score": 0.81}],
                }
            },
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {
                        "document_id": "treasury_1945_json",
                        "citation": "treasury_bulletin_1945_01.json#page=3",
                        "metadata": {"officeqa_status": "ok", "relative_path": "treasury_bulletin_1945_01.json"},
                    },
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

    report = build_case_report(
        {
            "id": "benchmark_case_1",
            "prompt": "OfficeQA task",
            "case_kind": "benchmark_regression",
            "expected_source_files": ["treasury_bulletin_1945_01.json"],
            "expected_answer": "251286",
        },
        _trace(state, "<REASONING>x</REASONING><FINAL_ANSWER>3</FINAL_ANSWER>"),
    )

    assert report["benchmark_analysis"]["semantic_verdict"] == "fail"
    assert "wrong_table_family" in report["benchmark_analysis"]["tags"]
    assert "wrong_row_or_column_semantics" in report["benchmark_analysis"]["tags"]
    assert report["benchmark_analysis"]["source_ranking_correct"] is True


def test_summarize_regression_report_tracks_source_ranking_and_repair_stalls():
    repair_stall = {
        "case_kind": "qa",
        "classification": {"subsystem": "validation", "compute_status": "insufficient"},
        "retrieval_strategy": "hybrid",
        "answer_mode": "deterministic_compute",
        "benchmark_analysis": {"semantic_verdict": "fail", "tags": ["wrong_source", "repair_stall", "incomplete_evidence"], "source_ranking_correct": False},
        "artifacts": {
            "chosen_sources": [{"document_id": "wrong"}],
            "extracted_tables": [{"document_id": "wrong"}],
            "final_answer": "",
            "compute_policy": "required",
            "structure_confidence_summary": {"table_confidence_gate_passed": True},
        },
    }
    good = {
        "case_kind": "qa",
        "classification": {"subsystem": "pass", "compute_status": "ok"},
        "retrieval_strategy": "table_first",
        "answer_mode": "deterministic_compute",
        "benchmark_analysis": {"semantic_verdict": "pass", "tags": [], "source_ranking_correct": True},
        "artifacts": {
            "chosen_sources": [{"document_id": "correct"}],
            "extracted_tables": [{"document_id": "correct"}],
            "final_answer": "42",
            "compute_policy": "required",
            "structure_confidence_summary": {"table_confidence_gate_passed": True},
            "semantic_diagnostics": {"admissibility_passed": True, "issues": []},
        },
    }

    summary = summarize_regression_report([good, repair_stall])

    assert summary["counts_by_benchmark_failure"]["wrong_source"] == 1
    assert summary["repair_stall_cases"] == 1
    assert summary["source_ranking_evaluable_cases"] == 2
    assert summary["source_ranking_correct_cases"] == 1
    assert summary["source_ranking_accuracy"] == 0.5
    assert summary["go_for_full_benchmark"] is False


def test_build_case_report_captures_explicit_repair_failure_tags():
    state = make_state(
        "OfficeQA task",
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        task_intent={"task_family": "document_qa", "execution_mode": "document_grounded_analysis"},
        curated_context={
            "structured_evidence": {"values": [], "value_count": 0},
            "provenance_summary": {
                "retrieval_diagnostics": {
                    "retrieval_decision": {"tool_name": "search_officeqa_documents", "strategy": "table_first"},
                    "candidate_sources": [{"document_id": "wrong"}],
                }
            },
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {"type": "search_officeqa_documents", "facts": {"metadata": {"officeqa_status": "ok"}, "document_id": "wrong"}}
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 2,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "repair_reused_stale_state",
            "contract_collapse_attempts": 0,
        },
        workpad={
            "officeqa_repair_failures": [
                {"code": "repair_applied_but_no_new_evidence", "details": {"tool_name": "search_officeqa_documents"}},
                {"code": "repair_reused_stale_state", "details": {"reason": "compute_or_review_reached_before_fresh_retrieval_hop"}},
            ],
            "officeqa_llm_repair_history": [
                {"stage": "retrieval_repair", "trigger": "wrong document", "path_changed": True, "decision": {"decision": "rewrite_query"}}
            ],
        },
        review_packet={
            "validator_result": {
                "verdict": "revise",
                "retry_allowed": True,
                "retry_stop_reason": "repair_reused_stale_state",
                "remediation_codes": ["RETRIEVE_EXACT_PERIOD"],
            }
        },
    )

    report = build_case_report(
        {"id": "case", "prompt": "OfficeQA task", "case_kind": "benchmark_regression"},
        _trace(state, "<FINAL_ANSWER>Cannot calculate</FINAL_ANSWER>"),
    )

    assert "repair_applied_but_no_new_evidence" in report["benchmark_analysis"]["tags"]
    assert "repair_reused_stale_state" in report["benchmark_analysis"]["tags"]
    assert "repair_stall" in report["benchmark_analysis"]["tags"]
    assert report["artifacts"]["repair_failures"][0]["code"] == "repair_applied_but_no_new_evidence"


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


def test_build_case_report_includes_strategy_exhaustion_proof():
    state = make_state(
        "OfficeQA task",
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        task_intent={"task_family": "document_qa", "execution_mode": "document_grounded_analysis"},
        workpad={
            "officeqa_strategy_exhaustion_proof": {
                "admissible_strategies": ["table_first", "hybrid"],
                "attempted_strategies_current_regime": ["table_first", "hybrid"],
                "untried_strategies": [],
                "strategies_exhausted": True,
                "benchmark_terminal_allowed": True,
            },
            "retrieval_strategy_attempts": [
                {"requested_strategy": "table_first", "applied_strategy": "table_first", "material_input_signature": "abc"},
                {"requested_strategy": "hybrid", "applied_strategy": "hybrid", "material_input_signature": "abc"},
            ],
        },
        execution_journal={
            "events": [],
            "tool_results": [],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 2,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "officeqa_retry_exhausted",
            "contract_collapse_attempts": 0,
        },
        review_packet={
            "validator_result": {
                "verdict": "revise",
                "retry_allowed": False,
                "retry_stop_reason": "officeqa_retry_exhausted",
                "remediation_codes": ["RETRIEVE_EXACT_PERIOD"],
            }
        },
    )

    report = build_case_report(
        {"id": "case", "prompt": "OfficeQA task", "case_kind": "benchmark_regression"},
        _trace(state, "<FINAL_ANSWER>Cannot calculate</FINAL_ANSWER>"),
    )

    assert report["artifacts"]["strategy_exhaustion_proof"]["strategies_exhausted"] is True
    assert len(report["artifacts"]["retrieval_strategy_attempts"]) == 2


def test_build_case_report_flags_premature_insufficiency_without_exhaustion():
    state = make_state(
        "OfficeQA task",
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        workpad={
            "officeqa_answerability_policy": {
                "insufficiency_requested": True,
                "benchmark_terminal_allowed": False,
                "policy_violation": True,
                "policy_violation_reason": "insufficiency_without_exhaustion",
                "insufficiency_answer_emitted": False,
                "low_confidence_compute": False,
            },
            "officeqa_strategy_exhaustion_proof": {
                "strategies_exhausted": False,
                "benchmark_terminal_allowed": False,
            },
        },
        execution_journal={
            "events": [],
            "tool_results": [],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "officeqa_no_retrieval_repair_path",
            "contract_collapse_attempts": 0,
        },
    )

    report = build_case_report(
        {"id": "case", "prompt": "OfficeQA task", "case_kind": "benchmark_regression"},
        _trace(state, "<FINAL_ANSWER>Cannot calculate</FINAL_ANSWER>"),
    )

    assert "premature_insufficiency" in report["benchmark_analysis"]["tags"]
    assert "insufficiency_without_exhaustion" in report["benchmark_analysis"]["tags"]
    assert report["artifacts"]["answerability_policy"]["policy_violation"] is True


def test_build_case_report_flags_low_confidence_compute_separately():
    state = make_state(
        "OfficeQA task",
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        curated_context={
            "structured_evidence": {
                "values": [{"document_id": "treasury_1940_json"}],
                "structure_confidence_summary": {"table_confidence_gate_passed": False},
            },
            "compute_result": {
                "status": "insufficient",
                "validation_errors": ["Low-confidence table structure prevents deterministic compute on the current evidence."],
            },
        },
        workpad={
            "officeqa_answerability_policy": {
                "insufficiency_requested": True,
                "benchmark_terminal_allowed": True,
                "policy_violation": False,
                "insufficiency_answer_emitted": True,
                "low_confidence_compute": True,
                "low_confidence_reason": "Low-confidence table structure prevents deterministic compute on the current evidence.",
            },
            "officeqa_strategy_exhaustion_proof": {
                "strategies_exhausted": True,
                "benchmark_terminal_allowed": True,
            },
        },
        execution_journal={
            "events": [],
            "tool_results": [],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 2,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "officeqa_low_confidence_structure",
            "contract_collapse_attempts": 0,
        },
    )

    report = build_case_report(
        {"id": "case", "prompt": "OfficeQA task", "case_kind": "benchmark_regression"},
        _trace(state, "<FINAL_ANSWER>Cannot calculate</FINAL_ANSWER>"),
    )

    assert "low_confidence_compute" in report["benchmark_analysis"]["tags"]
    assert report["artifacts"]["answerability_policy"]["low_confidence_compute"] is True


def test_summarize_regression_report_blocks_go_on_premature_insufficiency_and_low_confidence_compute():
    reports = [
        {
            "case_kind": "benchmark_regression",
            "classification": {"subsystem": "compute", "compute_status": "insufficient"},
            "retrieval_strategy": "hybrid",
            "answer_mode": "deterministic_compute",
            "benchmark_analysis": {
                "semantic_verdict": "fail",
                "tags": ["premature_insufficiency", "insufficiency_without_exhaustion", "low_confidence_compute"],
                "source_ranking_correct": None,
            },
            "artifacts": {
                "chosen_sources": [{"document_id": "x"}],
                "extracted_tables": [{"document_id": "x"}],
                "final_answer": "Cannot calculate",
                "compute_policy": "required",
                "structure_confidence_summary": {"table_confidence_gate_passed": False},
                "semantic_diagnostics": {"admissibility_passed": True},
            },
        }
    ]

    summary = summarize_regression_report(reports)

    assert summary["premature_insufficiency_cases"] == 1
    assert summary["low_confidence_compute_cases"] == 1
    assert summary["go_for_full_benchmark"] is False
