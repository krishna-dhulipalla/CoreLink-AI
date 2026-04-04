"""OfficeQA regression reporting helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

FailureSubsystem = Literal["pass", "routing", "retrieval", "extraction", "compute", "validation", "formatting"]

_EXTRACTION_STATUSES = {"missing_table", "partial_table", "missing_row", "missing_month_coverage", "unit_ambiguity"}
_ROUTING_STOP_REASONS = {"unsupported_capability", "no_bindable_capability", "recursion_limit"}


def load_regression_slice(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("OfficeQA regression slice must be a JSON list.")
    cases: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict) and str(item.get("id", "")).strip() and str(item.get("prompt", "")).strip():
            cases.append(dict(item))
    return cases


def _state_from_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    return dict((trace or {}).get("final_state") or {})


def _answer_from_trace(trace: dict[str, Any] | None) -> str:
    return str((trace or {}).get("answer", "") or "")


def _answer_contract(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("answer_contract") or {})


def _journal(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("execution_journal") or {})


def _quality_report(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("quality_report") or {})


def _curated_context(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("curated_context") or {})


def _benchmark_overrides(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("benchmark_overrides") or {})


def _workpad(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("workpad") or {})


def _retrieval_intent(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("retrieval_intent") or {})


def _structured_evidence(state: dict[str, Any]) -> dict[str, Any]:
    return dict(_curated_context(state).get("structured_evidence") or {})


def _compute_result(state: dict[str, Any]) -> dict[str, Any]:
    curated = _curated_context(state)
    if curated.get("compute_result"):
        return dict(curated.get("compute_result") or {})
    workpad = dict(state.get("workpad") or {})
    return dict(workpad.get("officeqa_compute") or {})


def _validator_result(state: dict[str, Any]) -> dict[str, Any]:
    review_packet = dict(state.get("review_packet") or {})
    return dict(review_packet.get("validator_result") or {})


def _tool_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_journal(state).get("tool_results") or [])


def _missing_dimensions(state: dict[str, Any]) -> list[str]:
    quality_missing = list(_quality_report(state).get("missing_dimensions") or [])
    sufficiency_missing = list(dict(state.get("evidence_sufficiency") or {}).get("missing_dimensions") or [])
    validator_missing = list(_validator_result(state).get("missing_dimensions") or [])
    return list(dict.fromkeys([*(str(item) for item in quality_missing), *(str(item) for item in sufficiency_missing), *(str(item) for item in validator_missing)]))


def _stop_reason(state: dict[str, Any]) -> str:
    quality_stop = str(_quality_report(state).get("stop_reason", "") or "").strip()
    if quality_stop:
        return quality_stop
    return str(_journal(state).get("stop_reason", "") or "").strip()


def _has_required_output_format(answer: str, answer_contract: dict[str, Any]) -> bool:
    if not answer_contract.get("requires_adapter"):
        return True
    if answer_contract.get("format") == "xml":
        reasoning_tag = str(dict(answer_contract.get("value_rules") or {}).get("reasoning_tag", "REASONING") or "REASONING")
        final_tag = str(dict(answer_contract.get("value_rules") or {}).get("final_answer_tag", answer_contract.get("xml_root_tag") or "FINAL_ANSWER") or "FINAL_ANSWER")
        return f"<{reasoning_tag}>" in answer and f"</{reasoning_tag}>" in answer and f"<{final_tag}>" in answer and f"</{final_tag}>" in answer
    if answer_contract.get("format") == "json":
        compact = answer.strip()
        return compact.startswith("{") and compact.endswith("}")
    return True


def classify_officeqa_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    state = _state_from_trace(trace)
    answer = _answer_from_trace(trace)
    answer_contract = _answer_contract(state)
    structured = _structured_evidence(state)
    compute = _compute_result(state)
    validator = _validator_result(state)
    stop_reason = _stop_reason(state)
    missing_dimensions = _missing_dimensions(state)
    tool_results = _tool_results(state)

    subsystem: FailureSubsystem = "pass"
    rationale = "Run completed without a classified failure."

    if not _has_required_output_format(answer, answer_contract):
        subsystem = "formatting"
        rationale = "Final answer did not satisfy the required output contract."
    elif stop_reason in _ROUTING_STOP_REASONS or stop_reason.endswith("preflight_failure"):
        subsystem = "routing"
        rationale = f"Run stopped before a viable OfficeQA execution path was established ({stop_reason or 'routing failure'})."
    elif validator.get("verdict") in {"fail", "revise"} or stop_reason.startswith("officeqa_"):
        subsystem = "validation"
        rationale = str(validator.get("reasoning", "") or _quality_report(state).get("reasoning", "") or "OfficeQA validator rejected the final answer.")
    elif compute and str(compute.get("status", "") or "") in {"insufficient", "unsupported"}:
        subsystem = "compute"
        rationale = "Structured evidence was present but deterministic compute could not complete."
    elif list(compute.get("validation_errors") or []):
        subsystem = "compute"
        rationale = "Deterministic compute emitted validation errors."
    else:
        statuses = {
            str(dict(result.get("facts") or {}).get("metadata", {}).get("officeqa_status", "") or "").strip().lower()
            for result in tool_results
            if isinstance(result, dict)
        }
        statuses.discard("")
        if statuses & _EXTRACTION_STATUSES:
            subsystem = "extraction"
            rationale = f"Document retrieval found candidate sources but extraction remained incomplete ({', '.join(sorted(statuses & _EXTRACTION_STATUSES))})."
        elif any("source family grounding" == item for item in missing_dimensions):
            subsystem = "retrieval"
            rationale = "Evidence did not ground to the required OfficeQA source family."
        elif not tool_results and not structured.get("values"):
            subsystem = "retrieval"
            rationale = "No OfficeQA retrieval artifacts were captured."

    return {
        "subsystem": subsystem,
        "stop_reason": stop_reason,
        "missing_dimensions": missing_dimensions,
        "validator_verdict": str(validator.get("verdict", "") or ""),
        "validator_codes": list(validator.get("remediation_codes", []) or []),
        "orchestration_strategy": str(validator.get("orchestration_strategy", "") or ""),
        "compute_status": str(compute.get("status", "") or ""),
        "rationale": rationale,
    }


def capture_officeqa_artifacts(trace: dict[str, Any] | None) -> dict[str, Any]:
    state = _state_from_trace(trace)
    answer = _answer_from_trace(trace)
    overrides = _benchmark_overrides(state)
    source_bundle = dict(state.get("source_bundle") or {})
    structured = _structured_evidence(state)
    compute = _compute_result(state)
    journal = _journal(state)
    provenance = dict(_curated_context(state).get("provenance_summary") or {})
    retrieval_plan = dict(provenance.get("retrieval_plan") or {})
    retrieval_diagnostics = dict(provenance.get("retrieval_diagnostics") or {})
    validator = _validator_result(state)
    citations = list(dict.fromkeys(str(item).strip() for item in journal.get("retrieved_citations", []) if str(item).strip()))
    workpad = _workpad(state)

    chosen_sources: list[dict[str, Any]] = []
    for result in _tool_results(state):
        facts = dict(result.get("facts") or {})
        metadata = dict(facts.get("metadata") or {})
        document_id = str(facts.get("document_id", "") or metadata.get("document_id", "") or "").strip()
        citation = str(facts.get("citation", "") or "").strip()
        path = str(facts.get("path", "") or metadata.get("relative_path", "") or metadata.get("file_name", "") or "").strip()
        if not any((document_id, citation, path)):
            continue
        entry = {
            "tool": str(result.get("type", "") or ""),
            "document_id": document_id,
            "path": path,
            "citation": citation,
            "status": str(metadata.get("officeqa_status", "") or ""),
        }
        if entry not in chosen_sources:
            chosen_sources.append(entry)

    extracted_tables = []
    for table in list(structured.get("tables", []))[:6]:
        if not isinstance(table, dict):
            continue
        extracted_tables.append(
            {
                "document_id": str(table.get("document_id", "") or ""),
                "page_locator": str(table.get("page_locator", "") or ""),
                "table_locator": str(table.get("table_locator", "") or ""),
                "headers": list(table.get("headers", []))[:8],
                "row_count": int(table.get("row_count", 0) or 0),
                "column_count": int(table.get("column_count", 0) or 0),
                "unit": str(table.get("unit", "") or ""),
            }
        )

    compute_ledger = []
    for step in list(compute.get("ledger", []))[:8]:
        if not isinstance(step, dict):
            continue
        compute_ledger.append(
            {
                "operator": str(step.get("operator", "") or ""),
                "description": str(step.get("description", "") or ""),
                "output": dict(step.get("output") or {}),
            }
        )

    return {
        "source_files_expected": list(source_bundle.get("source_files_expected", []) or overrides.get("source_files_expected", []) or []),
        "source_files_found": list(source_bundle.get("source_files_found", []) or overrides.get("source_files_found", []) or []),
        "source_files_missing": list(source_bundle.get("source_files_missing", []) or overrides.get("source_files_missing", []) or []),
        "chosen_sources": chosen_sources[:10],
        "retrieved_citations": citations[:10],
        "retrieval_decision": dict(retrieval_diagnostics.get("retrieval_decision") or {}),
        "strategy_reason": str(retrieval_diagnostics.get("strategy_reason", "") or ""),
        "answer_mode": str(retrieval_plan.get("answer_mode", "") or _retrieval_intent(state).get("answer_mode", "") or ""),
        "compute_policy": str(retrieval_plan.get("compute_policy", "") or _retrieval_intent(state).get("compute_policy", "") or ""),
        "validator_codes": list(validator.get("remediation_codes", []) or [])[:8],
        "orchestration_strategy": str(validator.get("orchestration_strategy", "") or ""),
        "retry_allowed": bool(validator.get("retry_allowed")),
        "retry_stop_reason": str(validator.get("retry_stop_reason", "") or ""),
        "candidate_sources": list(retrieval_diagnostics.get("candidate_sources", []) or [])[:6],
        "rejected_candidates": list(retrieval_diagnostics.get("rejected_candidates", []) or [])[:6],
        "evidence_gaps": list(dict(provenance.get("evidence_plan_check") or {}).get("predictive_gaps", []) or []),
        "extracted_tables": extracted_tables,
        "structured_value_count": int(structured.get("value_count", 0) or 0),
        "alignment_summary": dict(structured.get("alignment_summary", {}) or {}),
        "structure_confidence_summary": dict(structured.get("structure_confidence_summary", {}) or {}),
        "merged_series_count": len(list(structured.get("merged_series", [])) or []),
        "compute_selection_reasoning": str(compute.get("selection_reasoning", "") or ""),
        "rejected_aggregation_alternatives": list(compute.get("rejected_alternatives", []) or [])[:6],
        "compute_ledger": compute_ledger,
        "solver_llm_decision": dict(workpad.get("solver_llm_decision") or {}),
        "validator_remediation": list(validator.get("remediation_guidance", []) or [])[:6],
        "final_answer": answer,
        "final_artifact_signature": str(journal.get("final_artifact_signature", "") or ""),
    }


def build_case_report(case: dict[str, Any], trace: dict[str, Any] | None) -> dict[str, Any]:
    state = _state_from_trace(trace)
    classification = classify_officeqa_trace(trace)
    artifacts = capture_officeqa_artifacts(trace)
    quality_report = _quality_report(state)
    retrieval_intent = _retrieval_intent(state)
    retrieval_strategy = str(case.get("retrieval_strategy", "") or retrieval_intent.get("strategy", "") or "").strip()
    answer_mode = str(case.get("answer_mode", "") or retrieval_intent.get("answer_mode", "") or artifacts.get("answer_mode", "") or "").strip()
    case_kind = str(case.get("case_kind", "") or "qa").strip() or "qa"
    return {
        "id": str(case.get("id", "") or ""),
        "prompt": str(case.get("prompt", "") or ""),
        "case_kind": case_kind,
        "focus_subsystem": str(case.get("focus_subsystem", "") or ""),
        "retrieval_strategy": retrieval_strategy,
        "answer_mode": answer_mode,
        "smoke": bool(case.get("smoke")),
        "classification": classification,
        "artifacts": artifacts,
        "quality_report": {
            "verdict": str(quality_report.get("verdict", "") or ""),
            "score": float(quality_report.get("score", 0.0) or 0.0),
            "stop_reason": str(quality_report.get("stop_reason", "") or ""),
        },
        "execution_summary": {
            "task_family": str(dict(state.get("task_intent") or {}).get("task_family", "") or ""),
            "execution_mode": str(dict(state.get("task_intent") or {}).get("execution_mode", "") or ""),
            "retrieval_strategy": retrieval_strategy,
            "answer_mode": answer_mode,
            "solver_llm_used": bool(dict(artifacts.get("solver_llm_decision") or {}).get("used_llm")),
            "solver_llm_decision_reason": str(dict(artifacts.get("solver_llm_decision") or {}).get("reason", "") or ""),
            "retrieval_iterations": int(_journal(state).get("retrieval_iterations", 0) or 0),
            "tool_count": len(_tool_results(state)),
        },
    }


def summarize_regression_report(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {key: 0 for key in ("pass", "routing", "retrieval", "extraction", "compute", "validation", "formatting")}
    counts_by_strategy: dict[str, int] = {}
    counts_by_answer_mode: dict[str, int] = {}
    counts_by_case_kind: dict[str, int] = {}
    evidence_ready = 0
    extraction_ready = 0
    confidence_ready = 0
    compute_reliable = 0
    contract_success = 0
    qa_total = 0
    for item in case_reports:
        subsystem = str(dict(item.get("classification") or {}).get("subsystem", "") or "pass")
        counts[subsystem] = counts.get(subsystem, 0) + 1
        case_kind = str(item.get("case_kind", "") or "qa").strip() or "qa"
        counts_by_case_kind[case_kind] = counts_by_case_kind.get(case_kind, 0) + 1
        strategy = str(item.get("retrieval_strategy", "") or dict(item.get("execution_summary") or {}).get("retrieval_strategy", "") or "").strip()
        if strategy:
            counts_by_strategy[strategy] = counts_by_strategy.get(strategy, 0) + 1
        answer_mode = str(item.get("answer_mode", "") or dict(item.get("execution_summary") or {}).get("answer_mode", "") or "").strip()
        if answer_mode:
            counts_by_answer_mode[answer_mode] = counts_by_answer_mode.get(answer_mode, 0) + 1
        artifacts = dict(item.get("artifacts") or {})
        if subsystem == "pass" and list(artifacts.get("extracted_tables", [])) and str(artifacts.get("final_answer", "")).strip():
            evidence_ready += 1
        if case_kind != "qa":
            continue
        qa_total += 1
        if list(artifacts.get("chosen_sources", [])) and (list(artifacts.get("extracted_tables", [])) or str(artifacts.get("final_answer", "")).strip()):
            extraction_ready += 1
        confidence_summary = dict(artifacts.get("structure_confidence_summary", {}) or {})
        if confidence_summary:
            if bool(confidence_summary.get("table_confidence_gate_passed")):
                confidence_ready += 1
        else:
            confidence_ready += 1
        if subsystem != "formatting" and str(artifacts.get("final_answer", "")).strip():
            contract_success += 1
        compute_policy = str(artifacts.get("compute_policy", "") or "").strip()
        compute_status = str(dict(item.get("classification") or {}).get("compute_status", "") or "")
        if compute_policy in {"preferred", "not_applicable", ""} or compute_status == "ok":
            compute_reliable += 1

    total = len(case_reports)
    required_qa_threshold = math.ceil(qa_total * 0.6) if qa_total else 0
    qa_reports = [item for item in case_reports if str(item.get("case_kind", "") or "qa").strip() == "qa"]
    qa_critical_failures = sum(
        1
        for item in qa_reports
        if str(dict(item.get("classification") or {}).get("subsystem", "") or "") in {"routing", "formatting", "validation"}
    )
    go_for_full_benchmark = (
        qa_total > 0
        and qa_critical_failures == 0
        and extraction_ready >= required_qa_threshold
        and confidence_ready >= required_qa_threshold
        and compute_reliable >= required_qa_threshold
        and contract_success == qa_total
    )

    return {
        "total_cases": total,
        "qa_cases": qa_total,
        "counts_by_subsystem": counts,
        "counts_by_case_kind": counts_by_case_kind,
        "counts_by_strategy": counts_by_strategy,
        "counts_by_answer_mode": counts_by_answer_mode,
        "evidence_ready_cases": evidence_ready,
        "required_evidence_ready_cases": required_qa_threshold,
        "extraction_ready_cases": extraction_ready,
        "confidence_ready_cases": confidence_ready,
        "compute_reliable_cases": compute_reliable,
        "contract_success_cases": contract_success,
        "go_for_full_benchmark": go_for_full_benchmark,
        "go_no_go_reason": (
            "QA cases meet routing/formatting/validation, extraction, confidence, compute, and final-contract thresholds."
            if go_for_full_benchmark
            else "Hold full benchmark runs until QA cases have zero routing/formatting/validation failures and at least 60% satisfy extraction quality, evidence confidence, compute reliability, and final-answer contract success."
        ),
    }
