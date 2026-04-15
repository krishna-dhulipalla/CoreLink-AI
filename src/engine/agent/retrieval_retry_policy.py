from __future__ import annotations

import hashlib
import json
from typing import Any

from engine.agent.contracts import ExecutionJournal, RetrievalAction, RetrievalIntent, SourceBundle


def retrieval_strategy_attempts(workpad: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(item) for item in list(dict(workpad or {}).get("retrieval_strategy_attempts", []) or []) if isinstance(item, dict)]


def retrieval_planning_signature(
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    workpad: dict[str, Any] | None = None,
) -> str:
    padded_workpad = dict(workpad or {})
    excluded_docs = sorted(
        str(doc).strip()
        for doc in list(padded_workpad.get("officeqa_excluded_documents", []) or [])
        if str(doc).strip()
    )
    payload = {
        "focus_query": source_bundle.focus_query,
        "target_period": source_bundle.target_period,
        "source_files_expected": list(source_bundle.source_files_expected or []),
        "entity": retrieval_intent.entity,
        "metric": retrieval_intent.metric,
        "period": retrieval_intent.period,
        "period_type": retrieval_intent.period_type,
        "target_years": list(retrieval_intent.target_years or []),
        "publication_year_window": list(retrieval_intent.publication_year_window or []),
        "preferred_publication_years": list(retrieval_intent.preferred_publication_years or []),
        "source_constraint_policy": retrieval_intent.source_constraint_policy,
        "granularity_requirement": retrieval_intent.granularity_requirement,
        "document_family": retrieval_intent.document_family,
        "aggregation_shape": retrieval_intent.aggregation_shape,
        "must_include_terms": list(retrieval_intent.must_include_terms or []),
        "must_exclude_terms": list(retrieval_intent.must_exclude_terms or []),
        "query_plan": retrieval_intent.query_plan.model_dump(),
        "override_query": str(padded_workpad.get("officeqa_override_query", "") or ""),
        "override_table_query": str(padded_workpad.get("officeqa_override_table_query", "") or ""),
        # Including the excluded-document set ensures that rotating away from
        # a bad document (P11.2) materialises as a genuinely new planning
        # signature.  Without this, the duplicate-detection guard in
        # _plan_retrieval_action sees the same hash as the first pass and
        # refuses to plan a second retrieval tool call after REVISE.
        "excluded_documents": excluded_docs,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]


def retrieval_action_material_input_signature(action: RetrievalAction) -> str:
    payload = {
        "requested_strategy": str(action.requested_strategy or ""),
        "strategy": str(action.strategy or ""),
        "stage": str(action.stage or ""),
        "tool_name": str(action.tool_name or ""),
        "query": str(action.query or ""),
        "document_id": str(action.document_id or ""),
        "path": str(action.path or ""),
        "page_start": int(action.page_start or 0),
        "page_limit": int(action.page_limit or 0),
        "row_offset": int(action.row_offset or 0),
        "row_limit": int(action.row_limit or 0),
        "chunk_start": int(action.chunk_start or 0),
        "chunk_limit": int(action.chunk_limit or 0),
        "evidence_gap": str(action.evidence_gap or ""),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]


def strategy_rotation_requested(workpad: dict[str, Any] | None) -> tuple[bool, str]:
    padded_workpad = dict(workpad or {})
    retry_policy = dict(padded_workpad.get("officeqa_retry_policy") or {})
    repair_transition = dict(
        padded_workpad.get("officeqa_pending_repair_transition")
        or padded_workpad.get("officeqa_latest_repair_transition")
        or {}
    )
    repair_failures = [dict(item) for item in list(padded_workpad.get("officeqa_repair_failures", []) or []) if isinstance(item, dict)]
    evidence_commit_review = dict(padded_workpad.get("officeqa_evidence_commit_review") or {})
    if bool(retry_policy.get("retry_allowed")) and str(retry_policy.get("recommended_repair_target", "") or "") == "gather":
        return True, "validator_gather_retry"
    if evidence_commit_review:
        return True, "evidence_commit_review"
    if repair_transition:
        return True, str(repair_transition.get("reroute_action", "") or "repair_transition")
    if any(str(item.get("code", "") or "") in {"repair_applied_but_no_new_evidence", "repair_reused_stale_state"} for item in repair_failures):
        return True, "repair_stall"
    return False, ""


def last_regime_change(workpad: dict[str, Any] | None) -> str:
    padded_workpad = dict(workpad or {})
    transition = dict(
        padded_workpad.get("officeqa_pending_repair_transition")
        or padded_workpad.get("officeqa_latest_repair_transition")
        or {}
    )
    if transition:
        return str(transition.get("reroute_action", "") or transition.get("status", "") or "")
    latest_attempt = dict(padded_workpad.get("latest_retrieval_strategy_attempt") or {})
    return str(latest_attempt.get("regime_change", "") or "")


def last_officeqa_statuses(journal: ExecutionJournal) -> set[str]:
    statuses: set[str] = set()
    for result in list(journal.tool_results or []):
        facts = dict(dict(result).get("facts") or {})
        metadata = dict(facts.get("metadata") or {})
        status = str(metadata.get("officeqa_status", "") or "").strip().lower()
        if status:
            statuses.add(status)
    return statuses


def build_retrieval_exhaustion_proof(
    *,
    admissible_strategies: list[str],
    journal: ExecutionJournal,
    retrieval_intent: RetrievalIntent,
    workpad: dict[str, Any] | None,
    planning_signature: str,
    terminal_reason: str = "",
) -> dict[str, Any]:
    attempts = retrieval_strategy_attempts(workpad)
    attempted_total = [
        str(item.get("applied_strategy", "") or item.get("requested_strategy", "") or "").strip()
        for item in attempts
        if str(item.get("applied_strategy", "") or item.get("requested_strategy", "") or "").strip()
    ]
    attempted_current = [
        str(item.get("applied_strategy", "") or item.get("requested_strategy", "") or "").strip()
        for item in attempts
        if str(item.get("material_input_signature", "") or "") == planning_signature
        and str(item.get("applied_strategy", "") or item.get("requested_strategy", "") or "").strip()
    ]
    untried_current = [item for item in admissible_strategies if item not in attempted_current]
    statuses = last_officeqa_statuses(journal)
    retry_policy = dict(dict(workpad or {}).get("officeqa_retry_policy") or {})
    repair_failures = [dict(item) for item in list(dict(workpad or {}).get("officeqa_repair_failures", []) or []) if isinstance(item, dict)]
    failure_codes = {str(item.get("code", "") or "").strip() for item in repair_failures if str(item.get("code", "") or "").strip()}
    retrieval_diagnostics = dict(dict(workpad or {}).get("retrieval_diagnostics") or {})
    candidate_sources = [dict(item) for item in list(retrieval_diagnostics.get("candidate_sources", []) or []) if isinstance(item, dict)]
    parser_or_extraction_gap = bool(statuses & {"missing_table", "partial_table", "missing_row", "missing_month_coverage", "unit_ambiguity"})
    candidate_universe_exhausted = bool(failure_codes & {"repair_applied_but_no_new_evidence", "repair_exhausted_candidate_universe"}) or (
        bool(candidate_sources) and bool(retry_policy.get("retry_allowed")) and not untried_current and not parser_or_extraction_gap
    )
    return {
        "admissible_strategies": admissible_strategies,
        "attempted_strategies_total": attempted_total,
        "attempted_strategies_current_regime": attempted_current,
        "untried_strategies_current_regime": untried_current,
        "strategies_exhausted": not bool(untried_current),
        "candidate_universe_exhausted": candidate_universe_exhausted,
        "compute_capability_missing": bool(failure_codes & {"compute_capability_missing"}),
        "parser_or_extraction_gap": parser_or_extraction_gap,
        "terminal_reason": str(terminal_reason or ""),
        "benchmark_terminal_allowed": bool(not untried_current),
        "current_planning_signature": planning_signature,
        "status_hints": sorted(statuses),
        "repair_failure_codes": sorted(failure_codes),
        "aggregation_shape": str(retrieval_intent.aggregation_shape or ""),
        "retry_allowed": bool(retry_policy.get("retry_allowed")),
    }
