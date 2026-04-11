from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.contracts import (
    OfficeQALLLMUsageRecord,
    OfficeQALLMRepairDecision,
    OfficeQASourceRerankDecision,
    OfficeQATableAdmissibilityDecision,
    RetrievalIntent,
)
from agent.model_config import (
    get_model_name_for_officeqa_control,
    get_model_runtime_kwargs_for_officeqa_control,
    invoke_structured_output,
)
from agent.prompts import (
    FINANCIAL_EVIDENCE_COMMIT_SYSTEM,
    FINANCIAL_SOURCE_RERANK_SYSTEM,
    FINANCIAL_TABLE_ADMISSIBILITY_SYSTEM,
)

_MAX_SEMANTIC_PLAN_CALLS = 1
_BASE_RETRIEVAL_RERANK_CALLS = 2
_HARD_RETRIEVAL_RERANK_CALLS = 3
_BASE_TABLE_RERANK_CALLS = 2
_HARD_TABLE_RERANK_CALLS = 3
_MAX_EVIDENCE_COMMIT_CALLS = 1
_MAX_FINAL_SYNTHESIS_CALLS = 1


def _requires_hard_semantic_control(retrieval_intent: RetrievalIntent | None) -> bool:
    if retrieval_intent is None:
        return False
    if retrieval_intent.strategy in {"multi_table", "multi_document", "hybrid"}:
        return True
    if retrieval_intent.answer_mode != "deterministic_compute":
        return True
    if retrieval_intent.analysis_modes:
        return True
    if retrieval_intent.decomposition_confidence and retrieval_intent.decomposition_confidence < 0.78:
        return True
    if len(list(retrieval_intent.preferred_publication_years or [])) >= 4:
        return True
    if len(list(retrieval_intent.publication_year_window or [])) >= 5:
        return True
    return False


def officeqa_llm_control_budget(retrieval_intent: RetrievalIntent | None = None) -> dict[str, int]:
    hard_mode = _requires_hard_semantic_control(retrieval_intent)
    return {
        "semantic_plan_calls": _MAX_SEMANTIC_PLAN_CALLS,
        "retrieval_rerank_calls": _HARD_RETRIEVAL_RERANK_CALLS if hard_mode else _BASE_RETRIEVAL_RERANK_CALLS,
        "table_rerank_calls": _HARD_TABLE_RERANK_CALLS if hard_mode else _BASE_TABLE_RERANK_CALLS,
        "evidence_commit_calls": _MAX_EVIDENCE_COMMIT_CALLS,
        "final_synthesis_calls": _MAX_FINAL_SYNTHESIS_CALLS,
    }


def initial_officeqa_llm_control_state(retrieval_intent: RetrievalIntent) -> dict[str, int]:
    semantic_plan = retrieval_intent.semantic_plan
    return {
        "semantic_plan_calls": 1 if semantic_plan.used_llm else 0,
        "retrieval_rerank_calls": 0,
        "table_rerank_calls": 0,
        "evidence_commit_calls": 0,
        "final_synthesis_calls": 0,
    }


def record_officeqa_llm_usage(
    workpad: dict[str, Any],
    *,
    category: str,
    used: bool,
    reason: str,
    model_name: str = "",
    confidence: float = 0.0,
    applied: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(workpad)
    history = list(updated.get("officeqa_llm_usage", []) or [])
    payload = OfficeQALLLMUsageRecord(
        category=category,  # type: ignore[arg-type]
        used=used,
        reason=reason,
        model_name=model_name,
        confidence=confidence,
        applied=applied,
        details=dict(details or {}),
    ).model_dump()
    if payload not in history:
        history.append(payload)
    updated["officeqa_llm_usage"] = history
    updated["officeqa_latest_llm_usage"] = payload
    return updated


def _source_rerank_prompt(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    candidate_sources: list[dict[str, Any]],
    reason: str,
) -> str:
    compact_candidates: list[dict[str, Any]] = []
    for item in candidate_sources[:5]:
        if not isinstance(item, dict):
            continue
        compact_candidates.append(
            {
                "document_id": str(item.get("document_id", "") or ""),
                "title": str(item.get("title", "") or ""),
                "score": item.get("score", 0.0),
                "reason": str(item.get("reason", "") or ""),
                "best_evidence_unit": dict(item.get("best_evidence_unit") or {}),
                "metadata": dict(item.get("metadata", {}) or {}),
            }
        )
    return (
        f"TASK={task_text}\n"
        f"ENTITY={retrieval_intent.entity}\n"
        f"METRIC={retrieval_intent.metric}\n"
        f"PERIOD={retrieval_intent.period}\n"
        f"PERIOD_TYPE={retrieval_intent.period_type}\n"
        f"GRANULARITY={retrieval_intent.granularity_requirement}\n"
        f"SOURCE_CONSTRAINT_POLICY={retrieval_intent.source_constraint_policy}\n"
        f"PREFERRED_PUBLICATION_YEARS={list(retrieval_intent.preferred_publication_years)}\n"
        f"PUBLICATION_YEAR_WINDOW={list(retrieval_intent.publication_year_window)}\n"
        f"RETRIEVAL_REASON={reason}\n"
        f"CANDIDATE_SOURCES={compact_candidates}"
    )


def maybe_rerank_source_candidates(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    candidate_sources: list[dict[str, Any]],
    reason: str,
) -> OfficeQASourceRerankDecision | None:
    if len(candidate_sources) < 2:
        return None
    ranked_candidates = [dict(item) for item in list(candidate_sources or []) if isinstance(item, dict)]
    model_name = get_model_name_for_officeqa_control(
        "retrieval_rerank_llm",
        answer_mode=retrieval_intent.answer_mode,
        analysis_modes=retrieval_intent.analysis_modes,
    )
    runtime_kwargs = get_model_runtime_kwargs_for_officeqa_control(
        "retrieval_rerank_llm",
        answer_mode=retrieval_intent.answer_mode,
        analysis_modes=retrieval_intent.analysis_modes,
    )
    messages = [
        SystemMessage(content=FINANCIAL_SOURCE_RERANK_SYSTEM),
        HumanMessage(content=_source_rerank_prompt(task_text=task_text, retrieval_intent=retrieval_intent, candidate_sources=candidate_sources, reason=reason)),
    ]
    try:
        parsed, resolved_model = invoke_structured_output(
            "direct",
            OfficeQASourceRerankDecision,
            messages,
            temperature=0,
            max_tokens=220,
            model_name_override=model_name,
            runtime_kwargs_override=runtime_kwargs,
        )
        decision = OfficeQASourceRerankDecision.model_validate(parsed)
    except Exception:
        return None
    decision.model_name = resolved_model
    if decision.confidence < 0.52:
        return None
    if decision.decision == "select_candidate" and not decision.preferred_document_id.strip():
        return None
    if decision.decision == "select_candidate":
        selected = next(
            (
                dict(item)
                for item in ranked_candidates
                if str(item.get("document_id", "") or "").strip() == decision.preferred_document_id.strip()
            ),
            None,
        )
        if selected is None:
            return None
        top_score = float(ranked_candidates[0].get("score", 0.0) or 0.0)
        selected_score = float(selected.get("score", 0.0) or 0.0)
        selected_rank = int(selected.get("rank", 999) or 999)
        if selected_rank > 5:
            return None
        if reason not in {"wrong document", "incomplete evidence", "wrong row or column semantics"} and (top_score - selected_score) > 0.45:
            return None
    return decision


def _table_review_prompt(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    document_id: str,
    table_candidates: list[dict[str, Any]],
    reason: str,
) -> str:
    compact_candidates: list[dict[str, Any]] = []
    for item in table_candidates[:5]:
        if not isinstance(item, dict):
            continue
        compact_candidates.append(
            {
                "locator": str(item.get("locator", "") or ""),
                "table_family": str(item.get("table_family", "") or ""),
                "table_family_confidence": item.get("table_family_confidence", 0.0),
                "ranking_score": item.get("ranking_score", 0.0),
                "period_type": str(item.get("period_type", "") or ""),
                "table_confidence": item.get("table_confidence", 0.0),
                "page_locator": str(item.get("page_locator", "") or ""),
                "heading_chain": list(item.get("heading_chain", []))[:6],
                "headers": list(item.get("headers", []))[:8],
                "row_labels": list(item.get("row_labels", []))[:8],
                "unit_hint": str(item.get("unit_hint", "") or ""),
                "structural_signature": str(item.get("structural_signature", "") or ""),
            }
        )
    return (
        f"TASK={task_text}\n"
        f"DOCUMENT_ID={document_id}\n"
        f"ENTITY={retrieval_intent.entity}\n"
        f"METRIC={retrieval_intent.metric}\n"
        f"PERIOD={retrieval_intent.period}\n"
        f"PERIOD_TYPE={retrieval_intent.period_type}\n"
        f"GRANULARITY={retrieval_intent.granularity_requirement}\n"
        f"REVIEW_REASON={reason}\n"
        f"TABLE_CANDIDATES={compact_candidates}"
    )


def maybe_review_table_admissibility(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    document_id: str,
    table_candidates: list[dict[str, Any]],
    reason: str,
) -> OfficeQATableAdmissibilityDecision | None:
    if len(table_candidates) < 2:
        return None
    model_name = get_model_name_for_officeqa_control(
        "table_rerank_llm",
        answer_mode=retrieval_intent.answer_mode,
        analysis_modes=retrieval_intent.analysis_modes,
    )
    runtime_kwargs = get_model_runtime_kwargs_for_officeqa_control(
        "table_rerank_llm",
        answer_mode=retrieval_intent.answer_mode,
        analysis_modes=retrieval_intent.analysis_modes,
    )
    messages = [
        SystemMessage(content=FINANCIAL_TABLE_ADMISSIBILITY_SYSTEM),
        HumanMessage(content=_table_review_prompt(task_text=task_text, retrieval_intent=retrieval_intent, document_id=document_id, table_candidates=table_candidates, reason=reason)),
    ]
    try:
        parsed, resolved_model = invoke_structured_output(
            "direct",
            OfficeQATableAdmissibilityDecision,
            messages,
            temperature=0,
            max_tokens=220,
            model_name_override=model_name,
            runtime_kwargs_override=runtime_kwargs,
        )
        decision = OfficeQATableAdmissibilityDecision.model_validate(parsed)
    except Exception:
        return None
    decision.model_name = resolved_model
    if decision.confidence < 0.52:
        return None
    if decision.decision == "select_candidate" and not (
        decision.preferred_table_locator.strip() or decision.suggested_table_query.strip()
    ):
        return None
    return decision


def _current_structured_families(structured_evidence: dict[str, Any]) -> list[str]:
    families: list[str] = []
    for item in list(dict(structured_evidence or {}).get("tables", []) or [])[:6]:
        if not isinstance(item, dict):
            continue
        family = str(item.get("table_family", "") or "").strip().lower()
        if family:
            families.append(family)
    if not families:
        for item in list(dict(structured_evidence or {}).get("values", []) or [])[:12]:
            if not isinstance(item, dict):
                continue
            family = str(item.get("table_family", "") or "").strip().lower()
            if family:
                families.append(family)
    return list(dict.fromkeys(families))


def _evidence_commit_prompt(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    structured_evidence: dict[str, Any],
    candidate_sources: list[dict[str, Any]],
    evidence_review: dict[str, Any],
    reason: str,
) -> str:
    structured = dict(structured_evidence or {})
    compact_tables: list[dict[str, Any]] = []
    for item in list(structured.get("tables", []) or [])[:4]:
        if not isinstance(item, dict):
            continue
        compact_tables.append(
            {
                "document_id": str(item.get("document_id", "") or ""),
                "table_locator": str(item.get("table_locator", "") or item.get("locator", "") or ""),
                "page_locator": str(item.get("page_locator", "") or ""),
                "table_family": str(item.get("table_family", "") or ""),
                "table_family_confidence": item.get("table_family_confidence", 0.0),
                "period_type": str(item.get("period_type", "") or ""),
                "unit": str(item.get("unit", "") or ""),
            }
        )
    compact_sources: list[dict[str, Any]] = []
    for item in list(candidate_sources or [])[:5]:
        if not isinstance(item, dict):
            continue
        compact_sources.append(
            {
                "document_id": str(item.get("document_id", "") or ""),
                "title": str(item.get("title", "") or ""),
                "score": item.get("score", 0.0),
                "best_evidence_unit": dict(item.get("best_evidence_unit", {}) or {}),
                "metadata": dict(item.get("metadata", {}) or {}),
            }
        )
    return (
        f"TASK={task_text}\n"
        f"ENTITY={retrieval_intent.entity}\n"
        f"METRIC={retrieval_intent.metric}\n"
        f"PERIOD={retrieval_intent.period}\n"
        f"PERIOD_TYPE={retrieval_intent.period_type}\n"
        f"GRANULARITY={retrieval_intent.granularity_requirement}\n"
        f"ANSWER_MODE={retrieval_intent.answer_mode}\n"
        f"COMPUTE_POLICY={retrieval_intent.compute_policy}\n"
        f"REVIEW_REASON={reason}\n"
        f"EVIDENCE_REVIEW={dict(evidence_review or {})}\n"
        f"CURRENT_STRUCTURED_TABLES={compact_tables}\n"
        f"CURRENT_TYPING_SUMMARY={dict(structured.get('typing_consistency_summary', {}) or {})}\n"
        f"CURRENT_CONFIDENCE_SUMMARY={dict(structured.get('structure_confidence_summary', {}) or {})}\n"
        f"VISIBLE_CANDIDATE_SOURCES={compact_sources}"
    )


def maybe_review_evidence_commitment(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    structured_evidence: dict[str, Any],
    candidate_sources: list[dict[str, Any]],
    evidence_review: dict[str, Any],
    reason: str,
) -> OfficeQALLMRepairDecision | None:
    model_name = get_model_name_for_officeqa_control(
        "evidence_commit_llm",
        answer_mode=retrieval_intent.answer_mode,
        analysis_modes=retrieval_intent.analysis_modes,
    )
    runtime_kwargs = get_model_runtime_kwargs_for_officeqa_control(
        "evidence_commit_llm",
        answer_mode=retrieval_intent.answer_mode,
        analysis_modes=retrieval_intent.analysis_modes,
    )
    messages = [
        SystemMessage(content=FINANCIAL_EVIDENCE_COMMIT_SYSTEM),
        HumanMessage(
            content=_evidence_commit_prompt(
                task_text=task_text,
                retrieval_intent=retrieval_intent,
                structured_evidence=structured_evidence,
                candidate_sources=candidate_sources,
                evidence_review=evidence_review,
                reason=reason,
            )
        ),
    ]
    try:
        parsed, resolved_model = invoke_structured_output(
            "reviewer",
            OfficeQALLMRepairDecision,
            messages,
            temperature=0,
            max_tokens=240,
            model_name_override=model_name,
            runtime_kwargs_override=runtime_kwargs,
        )
        decision = OfficeQALLMRepairDecision.model_validate(parsed)
    except Exception:
        return None
    decision.model_name = resolved_model
    if decision.confidence < 0.55:
        return None
    if decision.decision == "retune_table_query" and not decision.revised_table_query.strip():
        return None
    if decision.restart_scope == "cross_document" and decision.decision not in {"rewrite_query", "widen_search_pool", "change_strategy"}:
        return None
    return decision


def _requested_period_kind(retrieval_intent: RetrievalIntent) -> str:
    return str(retrieval_intent.period_type or retrieval_intent.granularity_requirement or "").strip().lower()


def _candidate_family(candidate: dict[str, Any]) -> str:
    return str(dict(candidate.get("best_evidence_unit") or {}).get("table_family", "") or "").strip().lower()


def _required_family(retrieval_intent: RetrievalIntent) -> str:
    metric_tokens = {token for token in retrieval_intent.metric.lower().split() if token}
    if retrieval_intent.granularity_requirement == "monthly_series":
        return "monthly_series"
    if retrieval_intent.granularity_requirement == "fiscal_year":
        return "fiscal_year_comparison"
    if metric_tokens & {"debt", "outstanding", "obligations", "liabilities", "securities"}:
        return "debt_or_balance_sheet"
    if metric_tokens & {"expenditures", "receipts", "revenue", "revenues", "collections", "outlays", "spending"}:
        return "category_breakdown"
    return ""


def should_use_source_rerank_llm(
    *,
    retrieval_intent: RetrievalIntent,
    candidate_sources: list[dict[str, Any]],
    evidence_gap: str = "",
) -> tuple[bool, str]:
    ranked = [dict(item) for item in list(candidate_sources or []) if isinstance(item, dict)]
    if len(ranked) < 2:
        return False, "not_enough_candidates"

    normalized_gap = str(evidence_gap or "").strip().lower()
    if normalized_gap == "source pool too narrow":
        return False, "candidate_pool_requires_widening"
    if normalized_gap in {
        "wrong document",
        "incomplete evidence",
        "missing month coverage",
        "wrong period slice",
        "wrong row or column semantics",
        "repair_applied_but_no_new_evidence",
        "repair_reused_stale_state",
    }:
        return True, normalized_gap

    first = float(ranked[0].get("score", 0.0) or 0.0)
    second = float(ranked[1].get("score", 0.0) or 0.0)

    preferred_years = [str(item) for item in list(retrieval_intent.preferred_publication_years or []) if str(item)]
    top_year = ""
    runner_year = ""
    if preferred_years:
        top_year = str(dict(ranked[0].get("metadata") or {}).get("publication_year", "") or "")
        runner_year = str(dict(ranked[1].get("metadata") or {}).get("publication_year", "") or "")
        if top_year and top_year not in preferred_years[:2] and runner_year in preferred_years[:2]:
            return True, "publication_year_mismatch"

    requested_period = _requested_period_kind(retrieval_intent)
    top_unit = dict(ranked[0].get("best_evidence_unit") or {})
    runner_unit = dict(ranked[1].get("best_evidence_unit") or {})
    top_confidence = float(top_unit.get("table_confidence", 0.0) or 0.0)
    required_family = _required_family(retrieval_intent)
    if required_family:
        top_family = _candidate_family(ranked[0])
        if top_family and top_family != required_family:
            if any(_candidate_family(item) == required_family for item in ranked[1:5]):
                return True, "top_candidate_family_mismatch"
    if requested_period:
        top_period = str(top_unit.get("period_type", "") or "").strip().lower()
        runner_period = str(runner_unit.get("period_type", "") or "").strip().lower()
        if top_period and runner_period and top_period != requested_period and runner_period == requested_period:
            top_family = _candidate_family(ranked[0])
            runner_family = _candidate_family(ranked[1])
            if (first - second) < 0.38 or top_confidence < 0.72 or (top_family and runner_family and top_family != runner_family):
                return True, "period_type_mismatch"
            return False, "deterministic_top_candidate_stable"

    if top_confidence and top_confidence < 0.62:
        return True, "weak_evidence_unit_confidence"

    if first < 1.65:
        return True, "weak_top_source_score"
    if (first - second) < 0.35:
        top_family = _candidate_family(ranked[0])
        runner_family = _candidate_family(ranked[1])
        top_year_preferred = not preferred_years or top_year in preferred_years[:2]
        runner_year_reasonable = not preferred_years or runner_year in preferred_years[:3]
        if top_confidence >= 0.78 and top_family and top_family == runner_family and top_year_preferred and runner_year_reasonable:
            return False, "deterministic_top_candidate_stable"
        return True, "narrow_source_margin"

    if retrieval_intent.decomposition_confidence and retrieval_intent.decomposition_confidence < 0.78 and (first - second) < 0.4:
        return True, "low_decomposition_confidence"

    return False, "source_ranking_clear_enough"


def should_use_table_rerank_llm(
    *,
    retrieval_intent: RetrievalIntent,
    table_candidates: list[dict[str, Any]],
    evidence_gap: str = "",
) -> tuple[bool, str]:
    candidates = [dict(item) for item in list(table_candidates or []) if isinstance(item, dict)]
    if len(candidates) < 2:
        return False, "not_enough_candidates"

    normalized_gap = str(evidence_gap or "").strip().lower()
    if normalized_gap in {
        "wrong table family",
        "missing month coverage",
        "wrong row or column semantics",
        "wrong period slice",
        "incomplete evidence",
    }:
        return True, normalized_gap

    top = candidates[0]
    second = candidates[1]
    requested_period = _requested_period_kind(retrieval_intent)
    if requested_period:
        top_period = str(top.get("period_type", "") or "").strip().lower()
        runner_period = str(second.get("period_type", "") or "").strip().lower()
        if top_period and runner_period and top_period != requested_period and runner_period == requested_period:
            return True, "table_period_type_mismatch"

    top_score = float(top.get("ranking_score", 0.0) or 0.0)
    second_score = float(second.get("ranking_score", 0.0) or 0.0)
    top_table_conf = float(top.get("table_confidence", 0.0) or 0.0)
    top_family_conf = float(top.get("table_family_confidence", 0.0) or 0.0)
    if top_table_conf < 0.72 or top_family_conf < 0.7:
        return True, "low_table_confidence"
    if abs(top_score - second_score) < 0.35:
        return True, "narrow_table_margin"

    if retrieval_intent.decomposition_confidence and retrieval_intent.decomposition_confidence < 0.78 and top_table_conf < 0.82:
        return True, "low_decomposition_confidence"

    return False, "table_choice_clear_enough"


def should_use_evidence_commit_llm(
    *,
    retrieval_intent: RetrievalIntent,
    structured_evidence: dict[str, Any],
    candidate_sources: list[dict[str, Any]],
    evidence_review: dict[str, Any],
    repair_history: list[dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    if retrieval_intent.compute_policy != "required":
        return False, "compute_not_required"

    structured = dict(structured_evidence or {})
    tables = [dict(item) for item in list(structured.get("tables", []) or []) if isinstance(item, dict)]
    values = [dict(item) for item in list(structured.get("values", []) or []) if isinstance(item, dict)]
    if not tables and not values:
        return False, "no_structured_evidence"

    predictive_gaps = [
        str(item or "").strip().lower()
        for item in list(dict(evidence_review or {}).get("predictive_gaps", []) or [])
        if str(item or "").strip()
    ]
    if predictive_gaps:
        return False, "predictive_gaps_block_commit"

    typing_summary = dict(structured.get("typing_consistency_summary", {}) or {})
    if not bool(typing_summary.get("typing_consistent", True)):
        return True, "typing_drift_before_compute"

    confidence_summary = dict(structured.get("structure_confidence_summary", {}) or {})
    if not bool(confidence_summary.get("table_confidence_gate_passed", True)):
        return True, "low_structure_confidence_before_compute"

    current_families = _current_structured_families(structured)
    required_family = _required_family(retrieval_intent)
    ranked = [dict(item) for item in list(candidate_sources or []) if isinstance(item, dict)]
    if required_family and current_families and required_family not in current_families:
        if any(_candidate_family(item) == required_family for item in ranked):
            return True, "better_family_visible_in_candidate_pool"

    if repair_history and ranked:
        return True, "post_repair_commit_review"

    return False, "evidence_commit_stable"
