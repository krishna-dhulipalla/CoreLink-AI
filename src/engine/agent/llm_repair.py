from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from engine.agent.contracts import CuratedContext, ExecutionJournal, OfficeQALLMRepairDecision, RetrievalIntent, SourceBundle
from engine.agent.model_config import (
    get_model_name_for_officeqa_control,
    get_model_runtime_kwargs_for_officeqa_control,
    invoke_structured_output,
)
from engine.agent.prompts import OFFICEQA_STRUCTURED_REPAIR_SYSTEM, build_officeqa_structured_repair_prompt
from engine.agent.strategy_journal import recommend_strategy_order, strategy_journal_snapshot

_MAX_QUERY_REWRITE_CALLS = 2
_MAX_VALIDATOR_REPAIR_CALLS = 2
_REPAIR_JOURNAL_TASK_FAMILY = "document_qa"
_ALL_REPAIR_STRATEGIES = ["table_first", "text_first", "hybrid", "multi_table", "multi_document"]
_SUPPORTED_RETRIEVAL_GAPS = {
    "wrong document",
    "wrong table family",
    "missing month coverage",
    "year coverage",
    "narrative support",
    "inflation support",
    "join-ready evidence",
    "incomplete evidence",
    "wrong row or column semantics",
    "wrong period slice",
    "source pool too narrow",
}


def officeqa_llm_repair_budget() -> dict[str, int]:
    return {
        "query_rewrite_calls": _MAX_QUERY_REWRITE_CALLS,
        "validator_repair_calls": _MAX_VALIDATOR_REPAIR_CALLS,
    }


def initial_officeqa_llm_repair_state(retrieval_intent: RetrievalIntent) -> dict[str, int | bool]:
    return {
        "decomposition_llm_used": bool(retrieval_intent.decomposition_used_llm_fallback),
        "query_rewrite_calls": 0,
        "validator_repair_calls": 0,
    }


def _repairable_gap(evidence_gap: str) -> bool:
    normalized = str(evidence_gap or "").strip().lower()
    if not normalized:
        return False
    if normalized in _SUPPORTED_RETRIEVAL_GAPS:
        return True
    return any(fragment in normalized for fragment in _SUPPORTED_RETRIEVAL_GAPS)


def _repair_mutation_class(decision: OfficeQALLMRepairDecision) -> str:
    if decision.mutation_class != "none":
        return str(decision.mutation_class)
    if decision.decision == "retune_table_query":
        return "local_reselection"
    if decision.decision == "change_strategy":
        return "strategy_rotation"
    if decision.restart_scope == "same_document":
        return "same_document_restart"
    if decision.restart_scope == "cross_document":
        return "cross_document_restart"
    if decision.restart_scope == "semantic_plan_restart":
        return "semantic_plan_restart"
    if decision.decision == "widen_search_pool" or decision.publication_scope_action in {"widen_publication_horizon", "switch_to_retrospective"}:
        return "search_expansion"
    if decision.decision == "rewrite_query":
        return "cross_document_restart"
    return "none"


def officeqa_repair_universe_signature(
    *,
    retrieval_intent: RetrievalIntent,
    execution_journal: ExecutionJournal | None = None,
    current_query: str = "",
    current_table_query: str = "",
    candidate_sources: list[dict[str, Any]] | None = None,
    review_feedback: dict[str, Any] | None = None,
) -> str:
    journal = execution_journal or ExecutionJournal()
    feedback = dict(review_feedback or {})
    candidates = [dict(item) for item in list(candidate_sources or []) if isinstance(item, dict)]
    payload = {
        "strategy": str(retrieval_intent.strategy or "").strip().lower(),
        "source_constraint_policy": str(retrieval_intent.source_constraint_policy or "").strip().lower(),
        "target_years": list(retrieval_intent.target_years or []),
        "publication_year_window": list(retrieval_intent.publication_year_window or []),
        "preferred_publication_years": list(retrieval_intent.preferred_publication_years or []),
        "current_query": str(current_query or "").strip().lower(),
        "current_table_query": str(current_table_query or "").strip().lower(),
        "attempted_queries": list(dict.fromkeys(str(item or "").strip().lower() for item in list(journal.retrieval_queries or []) if str(item).strip()))[-6:],
        "candidate_documents": [
            {
                "document_id": str(item.get("document_id", "") or "").strip().lower(),
                "title": str(item.get("title", "") or "").strip().lower(),
                "table_family": str(dict(item.get("best_evidence_unit") or {}).get("table_family", "") or "").strip().lower(),
            }
            for item in candidates[:6]
        ],
        "repair_target": str(feedback.get("repair_target", "") or "").strip().lower(),
        "missing_dimensions": sorted(str(item or "").strip().lower() for item in list(feedback.get("missing_dimensions", []) or []) if str(item).strip()),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def officeqa_repair_universe_is_exhausted(workpad: dict[str, Any] | None, universe_signature: str) -> bool:
    signature = str(universe_signature or "").strip()
    if not signature:
        return False
    exhausted = [dict(item) for item in list(dict(workpad or {}).get("officeqa_exhausted_repair_universes", []) or []) if isinstance(item, dict)]
    return any(str(item.get("candidate_universe_signature", "") or "").strip() == signature for item in exhausted)


def _strategy_journal_context(retrieval_intent: RetrievalIntent) -> dict[str, Any]:
    recommendation = recommend_strategy_order(
        task_family=_REPAIR_JOURNAL_TASK_FAMILY,
        retrieval_intent=retrieval_intent,
        admissible_strategies=list(_ALL_REPAIR_STRATEGIES),
    )
    return {
        "recommendation": recommendation.model_dump(),
        "recent_entries": strategy_journal_snapshot(limit=6),
    }


def _structured_evidence_summary(curated_context: CuratedContext | None) -> dict[str, Any]:
    curated = curated_context or CuratedContext()
    structured = dict(curated.structured_evidence or {})
    values = [dict(item) for item in list(structured.get("values", []) or []) if isinstance(item, dict)]
    tables = [dict(item) for item in list(structured.get("tables", []) or []) if isinstance(item, dict)]
    families = [
        str(item.get("table_family", "") or "").strip()
        for item in [*values[:8], *tables[:8]]
        if str(item.get("table_family", "") or "").strip()
    ]
    period_types = [
        str(item.get("period_type", "") or "").strip()
        for item in [*values[:8], *tables[:8]]
        if str(item.get("period_type", "") or "").strip()
    ]
    return {
        "table_count": len(tables),
        "value_count": len(values),
        "table_families": list(dict.fromkeys(families))[:6],
        "period_types": list(dict.fromkeys(period_types))[:6],
        "typing_consistency_summary": dict(structured.get("typing_consistency_summary", {}) or {}),
    }


def _candidate_pools_seen(execution_journal: ExecutionJournal) -> list[dict[str, Any]]:
    pools: list[dict[str, Any]] = []
    for tool_result in list(execution_journal.tool_results or []):
        result = dict(tool_result or {})
        tool_type = str(result.get("type", "") or "").strip()
        if tool_type not in {"search_officeqa_documents", "search_reference_corpus", "internet_search", "list_reference_files"}:
            continue
        facts = dict(result.get("facts") or {})
        candidates = list(facts.get("results", []) or facts.get("documents", []) or [])[:4]
        pools.append(
            {
                "tool": tool_type,
                "documents": [
                    {
                        "document_id": str(dict(item).get("document_id", "") or "").strip(),
                        "title": str(dict(item).get("title", "") or "").strip(),
                        "score": dict(item).get("score", 0.0),
                    }
                    for item in candidates
                    if isinstance(item, dict)
                ],
            }
        )
    return pools[-3:]


def _compute_admissibility_failures(workpad: dict[str, Any], review_feedback: dict[str, Any] | None) -> list[str]:
    failures: list[str] = []
    compute_payload = dict(workpad.get("officeqa_compute") or {})
    failures.extend(str(item or "").strip() for item in list(compute_payload.get("semantic_issues", []) or []) if str(item or "").strip())
    validator_payload = dict(review_feedback or {})
    failures.extend(str(item or "").strip() for item in list(validator_payload.get("missing_dimensions", []) or []) if str(item or "").strip())
    failures.extend(str(item or "").strip() for item in list(validator_payload.get("remediation_codes", []) or []) if str(item or "").strip())
    return list(dict.fromkeys(failures))[:8]


def _rejected_evidence_families(
    candidate_sources: list[dict[str, Any]] | None,
    workpad: dict[str, Any],
    curated_context: CuratedContext | None,
) -> list[str]:
    families: list[str] = []
    for item in list(candidate_sources or [])[:8]:
        candidate = dict(item or {})
        best_unit = dict(candidate.get("best_evidence_unit", {}) or {})
        family = str(best_unit.get("table_family", "") or "").strip()
        if family:
            families.append(family)
    latest_repair = dict(workpad.get("officeqa_latest_repair_failure") or {})
    details = dict(latest_repair.get("details") or {})
    for key in ("table_family", "preferred_table_family", "admissibility_gap"):
        value = str(details.get(key, "") or "").strip()
        if value:
            families.append(value)
    structured_summary = _structured_evidence_summary(curated_context)
    families.extend(str(item or "").strip() for item in list(structured_summary.get("table_families", []) or []) if str(item or "").strip())
    return list(dict.fromkeys(families))[:8]


def _execution_snapshot(
    *,
    retrieval_intent: RetrievalIntent,
    execution_journal: ExecutionJournal,
    workpad: dict[str, Any],
    candidate_sources: list[dict[str, Any]] | None,
    review_feedback: dict[str, Any] | None,
    curated_context: CuratedContext | None,
) -> dict[str, Any]:
    candidate_universe_signature = officeqa_repair_universe_signature(
        retrieval_intent=retrieval_intent,
        execution_journal=execution_journal,
        current_query=str(retrieval_intent.query_plan.primary_semantic_query or ""),
        current_table_query=str(dict(workpad or {}).get("officeqa_override_table_query", "") or ""),
        candidate_sources=candidate_sources,
        review_feedback=review_feedback,
    )
    return {
        "attempted_queries": list(dict.fromkeys([str(item).strip() for item in list(execution_journal.retrieval_queries or []) if str(item).strip()]))[-6:],
        "candidate_pools_seen": _candidate_pools_seen(execution_journal),
        "rejected_evidence_families": _rejected_evidence_families(candidate_sources, workpad, curated_context),
        "compute_admissibility_failures": _compute_admissibility_failures(workpad, review_feedback),
        "repair_failures": [dict(item) for item in list(workpad.get("officeqa_repair_failures", []) or [])[-4:] if isinstance(item, dict)],
        "repair_history": [dict(item) for item in list(workpad.get("officeqa_llm_repair_history", []) or [])[-4:] if isinstance(item, dict)],
        "candidate_universe_signature": candidate_universe_signature,
        "structured_evidence_summary": _structured_evidence_summary(curated_context),
        "strategy_journal": _strategy_journal_context(retrieval_intent),
    }


def _regime_stall_detected(
    *,
    execution_journal: ExecutionJournal,
    workpad: dict[str, Any],
    evidence_gap: str,
    review_feedback: dict[str, Any] | None,
) -> bool:
    normalized_gap = str(evidence_gap or "").strip().lower()
    if not normalized_gap:
        return False
    severe_gap = any(
        fragment in normalized_gap
        for fragment in (
            "wrong document",
            "wrong table family",
            "source pool too narrow",
            "year coverage",
            "wrong row or column semantics",
            "wrong period slice",
            "incomplete evidence",
        )
    )
    if severe_gap and int(execution_journal.retrieval_iterations or 0) >= 1:
        return True
    if list(execution_journal.retrieval_queries or []):
        return True
    if list(workpad.get("officeqa_llm_repair_history", []) or []):
        return True
    if list(workpad.get("officeqa_repair_failures", []) or []):
        return True
    if list(workpad.get("officeqa_exhausted_repair_universes", []) or []):
        return True
    feedback = dict(review_feedback or {})
    if list(feedback.get("remediation_codes", []) or []) or list(feedback.get("missing_dimensions", []) or []):
        return True
    return False


def _invoke_repair_decision(prompt: str) -> OfficeQALLMRepairDecision | None:
    model_name = get_model_name_for_officeqa_control("repair_llm")
    runtime_kwargs = get_model_runtime_kwargs_for_officeqa_control("repair_llm")
    messages = [
        SystemMessage(content=OFFICEQA_STRUCTURED_REPAIR_SYSTEM),
        HumanMessage(content=prompt),
    ]
    try:
        parsed, resolved_model = invoke_structured_output(
            "solver",
            OfficeQALLMRepairDecision,
            messages,
            temperature=0,
            max_tokens=260,
            model_name_override=model_name,
            runtime_kwargs_override=runtime_kwargs,
        )
        decision = OfficeQALLMRepairDecision.model_validate(parsed)
    except Exception:
        return None
    decision.model_name = resolved_model
    if decision.confidence < 0.45:
        return None
    if decision.decision == "rewrite_query" and not decision.revised_query.strip():
        return None
    if decision.decision == "retune_table_query" and not decision.revised_table_query.strip():
        return None
    if decision.decision == "change_strategy" and not decision.preferred_strategy:
        return None
    decision.mutation_class = _repair_mutation_class(decision)  # type: ignore[assignment]
    if decision.mutation_class in {"strategy_rotation", "cross_document_restart", "search_expansion", "semantic_plan_restart"}:
        decision.prior_regime_exhausted = True
    return decision


def maybe_rewrite_retrieval_path(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    execution_journal: ExecutionJournal | None = None,
    workpad: dict[str, Any] | None = None,
    curated_context: CuratedContext | None = None,
    retrieval_strategy: str,
    evidence_gap: str,
    current_query: str = "",
    current_table_query: str = "",
    candidate_sources: list[dict[str, Any]] | None = None,
) -> OfficeQALLMRepairDecision | None:
    if not _repairable_gap(evidence_gap):
        return None
    journal = execution_journal or ExecutionJournal()
    workpad_payload = dict(workpad or {})
    if not _regime_stall_detected(
        execution_journal=journal,
        workpad=workpad_payload,
        evidence_gap=evidence_gap,
        review_feedback=None,
    ):
        return None
    universe_signature = officeqa_repair_universe_signature(
        retrieval_intent=retrieval_intent,
        execution_journal=journal,
        current_query=current_query or source_bundle.focus_query,
        current_table_query=current_table_query,
        candidate_sources=candidate_sources,
        review_feedback=None,
    )
    if officeqa_repair_universe_is_exhausted(workpad_payload, universe_signature):
        return None
    prompt = build_officeqa_structured_repair_prompt(
        task_text=task_text,
        retrieval_strategy=retrieval_strategy,
        evidence_gap=evidence_gap,
        source_constraint_policy=retrieval_intent.source_constraint_policy,
        target_years=list(retrieval_intent.target_years),
        publication_year_window=list(retrieval_intent.publication_year_window),
        preferred_publication_years=list(retrieval_intent.preferred_publication_years),
        current_query=current_query or source_bundle.focus_query,
        current_table_query=current_table_query,
        candidate_sources=candidate_sources,
        execution_snapshot=_execution_snapshot(
            retrieval_intent=retrieval_intent,
            execution_journal=journal,
            workpad=workpad_payload,
            candidate_sources=candidate_sources,
            review_feedback=None,
            curated_context=curated_context,
        ),
        review_feedback=None,
    )
    decision = _invoke_repair_decision(prompt)
    if decision is not None:
        decision.mutation_class = _repair_mutation_class(decision)  # type: ignore[assignment]
        if decision.mutation_class in {"strategy_rotation", "cross_document_restart", "search_expansion", "semantic_plan_restart"}:
            decision.prior_regime_exhausted = True
        decision.candidate_universe_signature = universe_signature
    return decision


def maybe_repair_from_validator(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    execution_journal: ExecutionJournal | None = None,
    workpad: dict[str, Any] | None = None,
    curated_context: CuratedContext | None = None,
    review_feedback: dict[str, Any],
    candidate_sources: list[dict[str, Any]] | None = None,
) -> OfficeQALLMRepairDecision | None:
    repair_target = str(review_feedback.get("repair_target", "") or "")
    if repair_target not in {"gather", "compute"}:
        return None
    journal = execution_journal or ExecutionJournal()
    workpad_payload = dict(workpad or {})
    if not _regime_stall_detected(
        execution_journal=journal,
        workpad=workpad_payload,
        evidence_gap=", ".join(str(item or "") for item in list(review_feedback.get("missing_dimensions", []) or [])),
        review_feedback=review_feedback,
    ):
        return None
    universe_signature = officeqa_repair_universe_signature(
        retrieval_intent=retrieval_intent,
        execution_journal=journal,
        current_query=(retrieval_intent.query_plan.primary_semantic_query or ""),
        current_table_query="",
        candidate_sources=candidate_sources,
        review_feedback=review_feedback,
    )
    if officeqa_repair_universe_is_exhausted(workpad_payload, universe_signature):
        return None
    missing_dimensions = [str(item or "") for item in list(review_feedback.get("missing_dimensions", []) or [])]
    evidence_gap = ", ".join(missing_dimensions[:4]) or ", ".join(str(item or "") for item in list(review_feedback.get("remediation_codes", []) or [])[:4])
    prompt = build_officeqa_structured_repair_prompt(
        task_text=task_text,
        retrieval_strategy=retrieval_intent.strategy,
        evidence_gap=evidence_gap,
        source_constraint_policy=retrieval_intent.source_constraint_policy,
        target_years=list(retrieval_intent.target_years),
        publication_year_window=list(retrieval_intent.publication_year_window),
        preferred_publication_years=list(retrieval_intent.preferred_publication_years),
        current_query=(retrieval_intent.query_plan.primary_semantic_query or ""),
        current_table_query="",
        candidate_sources=candidate_sources,
        execution_snapshot=_execution_snapshot(
            retrieval_intent=retrieval_intent,
            execution_journal=journal,
            workpad=workpad_payload,
            candidate_sources=candidate_sources,
            review_feedback=review_feedback,
            curated_context=curated_context,
        ),
        review_feedback=review_feedback,
    )
    decision = _invoke_repair_decision(prompt)
    if decision is not None:
        decision.mutation_class = _repair_mutation_class(decision)  # type: ignore[assignment]
        if decision.mutation_class in {"strategy_rotation", "cross_document_restart", "search_expansion", "semantic_plan_restart"}:
            decision.prior_regime_exhausted = True
        decision.candidate_universe_signature = universe_signature
    return decision
