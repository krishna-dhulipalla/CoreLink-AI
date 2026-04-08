from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.contracts import (
    OfficeQALLLMUsageRecord,
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
    FINANCIAL_SOURCE_RERANK_SYSTEM,
    FINANCIAL_TABLE_ADMISSIBILITY_SYSTEM,
)

_MAX_SEMANTIC_PLAN_CALLS = 1
_MAX_RETRIEVAL_RERANK_CALLS = 2
_MAX_TABLE_RERANK_CALLS = 2
_MAX_FINAL_SYNTHESIS_CALLS = 1


def officeqa_llm_control_budget() -> dict[str, int]:
    return {
        "semantic_plan_calls": _MAX_SEMANTIC_PLAN_CALLS,
        "retrieval_rerank_calls": _MAX_RETRIEVAL_RERANK_CALLS,
        "table_rerank_calls": _MAX_TABLE_RERANK_CALLS,
        "final_synthesis_calls": _MAX_FINAL_SYNTHESIS_CALLS,
    }


def initial_officeqa_llm_control_state(retrieval_intent: RetrievalIntent) -> dict[str, int]:
    semantic_plan = retrieval_intent.semantic_plan
    return {
        "semantic_plan_calls": 1 if semantic_plan.used_llm else 0,
        "retrieval_rerank_calls": 0,
        "table_rerank_calls": 0,
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
            "reviewer",
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
                "page_locator": str(item.get("page_locator", "") or ""),
                "headers": list(item.get("headers", []))[:8],
                "row_labels": list(item.get("row_labels", []))[:8],
                "unit_hint": str(item.get("unit_hint", "") or ""),
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
            "reviewer",
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
