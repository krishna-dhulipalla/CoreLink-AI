"""Structured validator for OfficeQA finalization."""

from __future__ import annotations

from typing import Any

from agent.contracts import (
    CuratedContext,
    EvidenceSufficiency,
    OfficeQAComputeResult,
    OfficeQAStructuredEvidence,
    OfficeQAValidationResult,
    RetrievalIntent,
)

_HARD_MISSING_DIMENSIONS = {
    "retrieved evidence": "retrieved evidence",
    "source family grounding": "source family correctness",
    "period scope": "time scope correctness",
    "aggregation semantics": "aggregation correctness",
    "entity scope": "entity/category correctness",
    "metric scope": "entity/category correctness",
    "numeric or quoted support": "provenance presence",
    "missing table": "aggregation correctness",
    "partial table": "aggregation correctness",
    "missing month coverage": "time scope correctness",
    "unit ambiguity": "unit consistency",
    "missing fiscal month coverage": "time scope correctness",
    "missing inflation or monthly support": "aggregation correctness",
}
_HARD_SCOPE_STATUSES = {
    "source_family": {"unknown", "web_or_reference"},
    "entity_scope": {"mismatch"},
    "period_scope": {"mismatch", "fiscal_mismatch", "calendar_mismatch"},
    "aggregation_type": {"annual_total_mismatch", "missing_monthly_support", "missing_inflation_or_monthly_support", "fiscal_mismatch", "calendar_mismatch"},
}
_SOFT_SCOPE_STATUSES = {
    "entity_scope": {"partial"},
}
_SOFT_SYNTHESIS_MISSING_DIMENSIONS = {
    "aggregation semantics",
    "missing table",
    "partial table",
    "missing month coverage",
    "missing fiscal month coverage",
    "missing inflation or monthly support",
}
_NAVIGATION_FAILURE = "navigational table selection"
_FAILURE_REMEDIATION_CODES = {
    "structured evidence presence": "RETRIEVE_GROUNDED_EVIDENCE",
    "source family correctness": "RERANK_ALLOWED_SOURCES",
    "time scope correctness": "RETRIEVE_EXACT_PERIOD",
    "aggregation correctness": "RECOVER_AGGREGATION_SUPPORT",
    "entity/category correctness": "RETIGHTEN_ENTITY_SCOPE",
    "unit consistency": "NORMALIZE_UNITS",
    "provenance presence": "RECOVER_PROVENANCE",
    "deterministic compute support": "COLLECT_COMPUTE_INPUTS",
    "deterministic compute validation": "REPAIR_COMPUTE_VALIDATION",
    "deterministic compute ledger": "REBUILD_COMPUTE_LEDGER",
    _NAVIGATION_FAILURE: "RERANK_ANALYTICAL_TABLES",
}


def _requires_deterministic_compute(retrieval_intent: RetrievalIntent) -> bool:
    if retrieval_intent.compute_policy == "required":
        return True
    return False


def _append_unique(target: list[str], *items: str) -> None:
    for raw in items:
        item = str(raw or "").strip()
        if item and item not in target:
            target.append(item)


def _looks_navigational_evidence(structured: OfficeQAStructuredEvidence, compute: OfficeQAComputeResult) -> bool:
    table_markers = {
        "table of contents",
        "cumulative table of contents",
        "issue and page number",
        "articles",
    }
    for table in structured.tables:
        if not isinstance(table, dict):
            continue
        text = " ".join(
            [
                str(table.get("table_locator", "") or ""),
                " ".join(str(item) for item in list(table.get("headers", []))[:8]),
                " ".join(" ".join(str(cell) for cell in row[:6]) for row in list(table.get("header_rows", []))[:3]),
            ]
        ).lower()
        if any(marker in text for marker in table_markers):
            return True
    for step in compute.ledger:
        if not isinstance(step, dict):
            continue
        for ref in list(step.get("provenance_refs", []) or []):
            if not isinstance(ref, dict):
                continue
            ref_text = " ".join(
                [
                    str(ref.get("table_locator", "") or ""),
                    str(ref.get("row_label", "") or ""),
                    str(ref.get("column_label", "") or ""),
                    " ".join(str(item) for item in list(ref.get("row_path", []))[:4]),
                    " ".join(str(item) for item in list(ref.get("column_path", []))[:4]),
                    str(ref.get("raw_value", "") or ""),
                ]
            ).lower()
            if any(marker in ref_text for marker in table_markers):
                return True
    return False


def _insufficiency_statement(hard_failures: list[str]) -> str:
    if any("compute" in item for item in hard_failures):
        final_value = "Cannot calculate from the provided Treasury Bulletin evidence"
    else:
        final_value = "Cannot determine from the provided Treasury Bulletin evidence"
    reasoning = "Structured OfficeQA validation failed because " + ", ".join(hard_failures[:3]) + "."
    return f"{reasoning}\nFinal answer: {final_value}."


def _remediation_guidance(hard_failures: list[str]) -> list[str]:
    guidance: list[str] = []
    for failure in hard_failures:
        if failure == "structured evidence presence":
            _append_unique(guidance, "Retrieve grounded pages or tables before attempting compute or final synthesis.")
        elif failure == "source family correctness":
            _append_unique(guidance, "Re-rank or re-run retrieval against Treasury Bulletin or other allowed official government sources only.")
        elif failure == "time scope correctness":
            _append_unique(guidance, "Re-extract evidence for the exact requested years, months, or fiscal/calendar scope before compute.")
        elif failure == "aggregation correctness":
            _append_unique(guidance, "Switch retrieval strategy or extraction stage until the correct table shape and aggregation support are recovered.")
        elif failure == "entity/category correctness":
            _append_unique(guidance, "Tighten row, table, or page selection around the exact target entity or category.")
        elif failure == "unit consistency":
            _append_unique(guidance, "Normalize units or re-extract evidence until all compared values share a consistent unit basis.")
        elif failure == "provenance presence":
            _append_unique(guidance, "Recover citation-complete evidence with page, table, and source references before finalization.")
        elif failure == "deterministic compute support":
            _append_unique(guidance, "Collect the remaining required evidence so deterministic compute can run, or fall back to grounded synthesis only if the task is non-deterministic.")
        elif failure == "deterministic compute validation":
            _append_unique(guidance, "Inspect the compute ledger and repair missing inputs or invalid aggregation assumptions before retrying.")
        elif failure == "deterministic compute ledger":
            _append_unique(guidance, "Persist the chosen compute path and ledger before final formatting.")
        elif failure == _NAVIGATION_FAILURE:
            _append_unique(guidance, "Reject table-of-contents or page-reference tables and retrieve the analytical data table for the requested metric.")
    return guidance


def officeqa_orchestration_strategy(retrieval_intent: RetrievalIntent) -> str:
    plan = retrieval_intent.evidence_plan
    if plan.requires_cross_source_alignment or retrieval_intent.strategy == "multi_document":
        return "cross_document_comparison"
    if retrieval_intent.strategy in {"hybrid", "multi_table"} or plan.join_keys or (plan.requires_table_support and plan.requires_text_support):
        return "hybrid_join"
    if retrieval_intent.answer_mode == "grounded_synthesis" and plan.requires_text_support and not plan.requires_table_support:
        return "text_reasoning"
    if retrieval_intent.answer_mode in {"grounded_synthesis", "hybrid_grounded"} and plan.requires_text_support and retrieval_intent.compute_policy != "required":
        return "text_reasoning"
    return "table_compute"


def _remediation_codes(failures: list[str]) -> list[str]:
    codes: list[str] = []
    for failure in failures:
        code = _FAILURE_REMEDIATION_CODES.get(str(failure).strip(), "")
        if code and code not in codes:
            codes.append(code)
    return codes


def _recommended_repair_target(remediation_codes: list[str]) -> str:
    if any(code in remediation_codes for code in ("COLLECT_COMPUTE_INPUTS",)):
        return "gather"
    if any(code in remediation_codes for code in ("REPAIR_COMPUTE_VALIDATION", "REBUILD_COMPUTE_LEDGER")):
        return "compute"
    if remediation_codes:
        return "gather"
    return "none"


def validate_officeqa_final(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    curated_context: dict[str, Any] | CuratedContext,
    evidence_sufficiency: dict[str, Any] | EvidenceSufficiency,
    citations: list[str] | None = None,
) -> OfficeQAValidationResult:
    curated = curated_context if isinstance(curated_context, CuratedContext) else CuratedContext.model_validate(curated_context)
    sufficiency = evidence_sufficiency if isinstance(evidence_sufficiency, EvidenceSufficiency) else EvidenceSufficiency.model_validate(evidence_sufficiency)
    structured = OfficeQAStructuredEvidence.model_validate(curated.structured_evidence or {})
    compute = OfficeQAComputeResult.model_validate(curated.compute_result or {})
    citation_list = [str(item).strip() for item in citations or [] if str(item).strip()]
    orchestration_strategy = officeqa_orchestration_strategy(retrieval_intent)
    alignment_summary = dict(structured.alignment_summary or {})

    hard_failures: list[str] = []
    missing_dimensions: list[str] = []

    if not structured.tables and not structured.values and not structured.page_chunks:
        _append_unique(hard_failures, "structured evidence presence")
    if not structured.provenance_complete:
        _append_unique(hard_failures, "provenance presence")
    if not citation_list:
        _append_unique(hard_failures, "provenance presence")

    for field_name, blocked_statuses in _HARD_SCOPE_STATUSES.items():
        status = str(getattr(sufficiency, field_name, "") or "")
        if field_name == "aggregation_type" and retrieval_intent.compute_policy != "required":
            continue
        if status in blocked_statuses:
            label = field_name.replace("_", " ")
            if field_name == "source_family":
                label = "source family correctness"
            elif field_name == "period_scope":
                label = "time scope correctness"
            elif field_name == "aggregation_type":
                label = "aggregation correctness"
            elif field_name == "entity_scope":
                label = "entity/category correctness"
            _append_unique(hard_failures, label)

    for field_name, warned_statuses in _SOFT_SCOPE_STATUSES.items():
        status = str(getattr(sufficiency, field_name, "") or "")
        if status in warned_statuses:
            if field_name == "entity_scope":
                _append_unique(missing_dimensions, "entity/category correctness")

    for item in sufficiency.missing_dimensions:
        if retrieval_intent.compute_policy != "required" and str(item).strip() in _SOFT_SYNTHESIS_MISSING_DIMENSIONS:
            continue
        mapped = _HARD_MISSING_DIMENSIONS.get(str(item), "")
        if mapped:
            _append_unique(hard_failures, mapped)

    units_seen = {str(item).strip().lower() for item in structured.units_seen if str(item).strip()}
    if len(units_seen) > 1:
        _append_unique(hard_failures, "unit consistency")
    if retrieval_intent.evidence_plan.requires_cross_source_alignment:
        if int(alignment_summary.get("aligned_document_count", 0) or 0) < 2:
            _append_unique(hard_failures, "time scope correctness")
        if not bool(alignment_summary.get("unit_consistent", True)):
            _append_unique(hard_failures, "unit consistency")
    if _looks_navigational_evidence(structured, compute):
        _append_unique(hard_failures, _NAVIGATION_FAILURE)

    if _requires_deterministic_compute(retrieval_intent):
        if compute.status != "ok":
            _append_unique(hard_failures, "deterministic compute support")
        if compute.validation_errors:
            _append_unique(hard_failures, "deterministic compute validation")
        if compute.status == "ok" and not compute.provenance_complete:
            _append_unique(hard_failures, "provenance presence")
        if compute.status == "ok" and not compute.ledger:
            _append_unique(hard_failures, "deterministic compute ledger")
    elif compute.status == "ok":
        if compute.validation_errors:
            _append_unique(hard_failures, "deterministic compute validation")
        if not compute.provenance_complete:
            _append_unique(hard_failures, "provenance presence")

    if hard_failures:
        hard_failures = list(dict.fromkeys(hard_failures))
        remediation_codes = _remediation_codes(hard_failures)
        return OfficeQAValidationResult(
            verdict="revise",
            reasoning=(
                "OfficeQA validator found a structured grounding or compute gap that should drive a targeted repair pass "
                "instead of another generic synthesis loop."
            ),
            missing_dimensions=hard_failures,
            hard_failures=hard_failures,
            remediation_codes=remediation_codes,
            remediation_guidance=_remediation_guidance(hard_failures),
            recommended_repair_target=_recommended_repair_target(remediation_codes),
            orchestration_strategy=orchestration_strategy,
            retry_allowed=True,
            stop_reason="officeqa_structured_revision_required",
            insufficiency_answer=_insufficiency_statement(hard_failures),
            replace_answer=False,
        )

    if missing_dimensions:
        missing_dimensions = list(dict.fromkeys(missing_dimensions))
        remediation_codes = _remediation_codes(missing_dimensions)
        return OfficeQAValidationResult(
            verdict="revise",
            reasoning="OfficeQA validator found a remaining structured-alignment gap that should be corrected before formatting the final answer.",
            missing_dimensions=missing_dimensions,
            remediation_codes=remediation_codes,
            remediation_guidance=_remediation_guidance(missing_dimensions),
            recommended_repair_target=_recommended_repair_target(remediation_codes),
            orchestration_strategy=orchestration_strategy,
            retry_allowed=True,
            stop_reason="officeqa_structured_revision_required",
        )

    return OfficeQAValidationResult(
        verdict="pass",
        reasoning="OfficeQA validator accepted the structured evidence, provenance, and compute state.",
        orchestration_strategy=orchestration_strategy,
    )
