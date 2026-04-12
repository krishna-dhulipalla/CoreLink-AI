from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from agent.benchmarks import benchmark_runtime_policy
from agent.contracts import ExecutionJournal, RetrievalAction, RetrievalIntent, SourceBundle, ToolPlan
from agent.officeqa_strategy_planner import (
    OfficeQAPlannerOps,
    OfficeQAPlanningContext,
    plan_cell_followup,
    plan_initial_action,
    plan_pages_followup,
    plan_row_followup,
    plan_search_followup,
    plan_table_followup,
)
from agent.retrieval_candidates import (
    candidate_best_evidence_unit as candidate_best_evidence_unit_impl,
    candidate_metadata_text as candidate_metadata_text_impl,
    dedupe_search_candidates as dedupe_search_candidates_impl,
    rank_search_candidates as rank_search_candidates_impl,
    retrieval_focus_tokens as retrieval_focus_tokens_impl,
    retrieval_tokens as retrieval_tokens_impl,
    search_candidate_score as search_candidate_score_impl,
    search_candidate_text as search_candidate_text_impl,
    search_result_candidates as search_result_candidates_impl,
    table_family_matches_intent as table_family_matches_intent_impl,
)
from agent.retrieval_strategy_kernel import FunctionRetrievalStrategyHandler, RetrievalStrategyContext, RetrievalStrategyKernel
from agent.retrieval_retry_policy import (
    build_retrieval_exhaustion_proof as build_retrieval_exhaustion_proof_impl,
    last_officeqa_statuses as last_officeqa_statuses_impl,
    last_regime_change as last_regime_change_impl,
    retrieval_action_material_input_signature as retrieval_action_material_input_signature_impl,
    retrieval_planning_signature as retrieval_planning_signature_impl,
    retrieval_strategy_attempts as retrieval_strategy_attempts_impl,
    strategy_rotation_requested as strategy_rotation_requested_impl,
)
from agent.retrieval_tool_runtime import (
    filter_args_for_tool as filter_args_for_tool_impl,
    generic_tool_args as generic_tool_args_impl,
    invoke_tool as invoke_tool_impl,
    run_tool_step as run_tool_step_impl,
    run_tool_step_with_args as run_tool_step_with_args_impl,
    structured_tool_args as structured_tool_args_impl,
    tool_args_from_retrieval_action as tool_args_from_retrieval_action_impl,
    tool_descriptor as tool_descriptor_impl,
    tool_family as tool_family_impl,
    tool_lookup as tool_lookup_impl,
    tool_role as tool_role_impl,
)
from agent.retrieval_reasoning import assess_evidence_sufficiency
from agent.strategy_journal import recommend_strategy_order
from agent.state import AgentState

RuntimeState = AgentState
IS_COMPETITION_MODE = os.getenv("COMPETITION_MODE", "").strip().lower() in {"1", "true", "yes", "on"} or os.getenv("BENCHMARK_NAME", "").strip().lower() == "officeqa"
MAX_RETRIEVAL_HOPS = max(2, int(os.getenv("MAX_RETRIEVAL_HOPS", "10" if IS_COMPETITION_MODE else "4")))
_RETRIEVAL_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "which",
    "according",
    "report",
    "document",
    "source",
    "using",
    "based",
}


def _tool_lookup(registry: dict[str, dict[str, Any]], tool_name: str) -> Any | None:
    return tool_lookup_impl(registry, tool_name)


def _tool_descriptor(registry: dict[str, dict[str, Any]], tool_name: str) -> dict[str, Any]:
    return tool_descriptor_impl(registry, tool_name)


def _tool_family(registry: dict[str, dict[str, Any]], tool_name: str) -> str:
    return tool_family_impl(registry, tool_name)


def _tool_role(registry: dict[str, dict[str, Any]], tool_name: str) -> str:
    return tool_role_impl(registry, tool_name)


def _filter_args_for_tool(registry: dict[str, dict[str, Any]], tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return filter_args_for_tool_impl(registry, tool_name, args)


def _generic_tool_args(
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent | None = None,
) -> dict[str, Any]:
    return generic_tool_args_impl(registry, tool_name, source_bundle, retrieval_intent, _derive_retrieval_seed_query)


def _structured_tool_args(state: RuntimeState, registry: dict[str, dict[str, Any]], tool_name: str) -> dict[str, Any]:
    source_bundle = SourceBundle.model_validate(state.get("source_bundle") or {})
    retrieval_intent = RetrievalIntent.model_validate(state.get("retrieval_intent") or {})
    return structured_tool_args_impl(
        state,
        registry,
        tool_name,
        source_bundle,
        retrieval_intent,
        _derive_retrieval_seed_query,
    )


async def _invoke_tool(tool_obj: Any, args: dict[str, Any]) -> Any:
    return await invoke_tool_impl(tool_obj, args)


async def _run_tool_step(state: RuntimeState, registry: dict[str, dict[str, Any]], tool_name: str) -> tuple[dict[str, Any], Any]:
    source_bundle = SourceBundle.model_validate(state.get("source_bundle") or {})
    retrieval_intent = RetrievalIntent.model_validate(state.get("retrieval_intent") or {})
    return await run_tool_step_impl(state, registry, tool_name, source_bundle, retrieval_intent, _derive_retrieval_seed_query)


async def _run_tool_step_with_args(
    state: RuntimeState,
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    args_override: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    _ = state
    return await run_tool_step_with_args_impl(registry, tool_name, args_override)


def _retrieval_tools_available(
    tool_plan: ToolPlan,
    registry: dict[str, dict[str, Any]],
) -> list[str]:
    retrieval_tools = []
    for tool_name in tool_plan.selected_tools:
        descriptor = _tool_descriptor(registry, tool_name)
        if descriptor.get("tool_family") in {"document_retrieval", "external_retrieval"}:
            retrieval_tools.append(tool_name)
    return retrieval_tools


def _retrieval_tools_by_role(
    tool_plan: ToolPlan,
    registry: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {"discover": [], "search": [], "fetch": [], "other": []}
    for tool_name in _retrieval_tools_available(tool_plan, registry):
        role = str(_tool_descriptor(registry, tool_name).get("tool_role", "") or "")
        bucket = role if role in grouped else "other"
        grouped[bucket].append(tool_name)
    return grouped


def _retrieval_query_candidates(retrieval_intent: RetrievalIntent | None = None) -> list[str]:
    if retrieval_intent is None:
        return []
    query_plan = retrieval_intent.query_plan
    candidates = [
        query_plan.temporal_query,
        query_plan.primary_semantic_query,
        query_plan.granularity_query,
        query_plan.qualifier_query,
        query_plan.alternate_lexical_query,
        *list(retrieval_intent.query_candidates or []),
    ]
    if retrieval_intent.source_constraint_policy == "hard" or not any(str(candidate or "").strip() for candidate in candidates):
        candidates.append(query_plan.source_file_query)
    return [
        item
        for item in dict.fromkeys(str(candidate).strip()[:280] for candidate in candidates if str(candidate or "").strip())
    ]


def _derive_retrieval_seed_query(source_bundle: SourceBundle, retrieval_intent: RetrievalIntent | None = None) -> str:
    candidates = _retrieval_query_candidates(retrieval_intent)
    if candidates:
        return candidates[0]
    parts = [source_bundle.focus_query or source_bundle.task_text]
    if source_bundle.entities:
        parts.append(" ".join(source_bundle.entities[:4]))
    if source_bundle.target_period:
        parts.append(source_bundle.target_period)
    seed = " ".join(part.strip() for part in parts if part and part.strip())
    return re.sub(r"\s+", " ", seed).strip()[:280]


def _next_retrieval_query(journal: ExecutionJournal, retrieval_intent: RetrievalIntent, source_bundle: SourceBundle) -> str:
    used = {re.sub(r"\s+", " ", query).strip().lower() for query in journal.retrieval_queries}
    for candidate in _retrieval_query_candidates(retrieval_intent):
        normalized = re.sub(r"\s+", " ", candidate).strip().lower()
        if normalized and normalized not in used:
            return candidate
    return _derive_retrieval_seed_query(source_bundle, retrieval_intent)


def _retrieval_tokens(text: str) -> list[str]:
    return retrieval_tokens_impl(text)


def _best_surface_match(text: str, phrase: str) -> float:
    normalized_text = " ".join(_retrieval_tokens(text))
    normalized_phrase = " ".join(_retrieval_tokens(phrase))
    if not normalized_text or not normalized_phrase:
        return 0.0
    if normalized_phrase in normalized_text:
        return 1.0
    text_tokens = set(normalized_text.split())
    phrase_tokens = set(normalized_phrase.split())
    if not phrase_tokens:
        return 0.0
    return float(len(text_tokens & phrase_tokens)) / max(1, len(phrase_tokens))


def _best_unit_text(best_unit: dict[str, Any]) -> str:
    return " ".join(
        [
            str(best_unit.get("locator", "") or ""),
            str(best_unit.get("context_text", "") or ""),
            " ".join(str(item or "") for item in list(best_unit.get("heading_chain", []))),
            " ".join(str(item or "") for item in list(best_unit.get("headers", []))),
            " ".join(str(item or "") for item in list(best_unit.get("row_labels", []))),
            " ".join(str(item or "") for item in list(best_unit.get("row_paths", []))),
            " ".join(str(item or "") for item in list(best_unit.get("column_paths", []))),
        ]
    ).strip()


def _family_terms(retrieval_intent: RetrievalIntent) -> tuple[set[str], set[str]]:
    metric_tokens = _query_metric_tokens(retrieval_intent)
    debt_terms = {"debt", "outstanding", "obligations", "liabilities", "securities", "guaranteed"}
    flow_terms = {"expenditures", "receipts", "revenue", "revenues", "collections", "outlays", "spending"}
    return metric_tokens & debt_terms, metric_tokens & flow_terms


def _table_family_preference_score(table_family: str, retrieval_intent: RetrievalIntent) -> float:
    family = str(table_family or "").strip().lower()
    if not family:
        return 0.0
    if family == "navigation_or_contents":
        return -1.5
    debt_terms, flow_terms = _family_terms(retrieval_intent)
    granularity = retrieval_intent.granularity_requirement
    if granularity == "monthly_series":
        if family == "monthly_series":
            return 0.95
        if family in {"annual_summary", "debt_or_balance_sheet"}:
            return -0.45
        return -0.12
    if granularity == "fiscal_year":
        if family == "fiscal_year_comparison":
            return 0.72
        if family == "category_breakdown":
            return 0.18
    if debt_terms:
        if family == "debt_or_balance_sheet":
            return 0.72
        if family in {"category_breakdown", "annual_summary"}:
            return -0.28
    if flow_terms:
        if family == "category_breakdown":
            return 0.82
        if family == "annual_summary":
            return 0.2
        if family == "debt_or_balance_sheet":
            return -0.55
    return 0.0


def _best_unit_alignment_score(best_unit: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    if not best_unit:
        return 0.0
    heading_text = " ".join(str(item or "") for item in list(best_unit.get("heading_chain", [])))
    row_text = " ".join(str(item or "") for item in list(best_unit.get("row_paths", [])) or list(best_unit.get("row_labels", [])))
    header_text = " ".join(str(item or "") for item in list(best_unit.get("headers", [])))
    column_text = " ".join(str(item or "") for item in list(best_unit.get("column_paths", [])))
    locator_text = str(best_unit.get("locator", "") or "")
    preview_text = str(best_unit.get("preview_text", "") or "")[:240]
    entity_text = str(retrieval_intent.entity or "").strip()
    metric_text = str(retrieval_intent.metric or "").strip()
    score = 0.0
    if entity_text:
        score += 0.78 * _best_surface_match(row_text, entity_text)
        score += 0.45 * _best_surface_match(heading_text, entity_text)
        score += 0.16 * _best_surface_match(locator_text, entity_text)
        if _best_surface_match(row_text, entity_text) < 0.34 and _best_surface_match(heading_text, entity_text) < 0.34:
            score -= 0.3
    if metric_text:
        score += 0.52 * _best_surface_match(header_text, metric_text)
        score += 0.46 * _best_surface_match(heading_text, metric_text)
        score += 0.28 * _best_surface_match(column_text, metric_text)
        score += 0.1 * _best_surface_match(preview_text, metric_text)
        if _best_surface_match(header_text, metric_text) < 0.25 and _best_surface_match(heading_text, metric_text) < 0.25:
            score -= 0.22
    score += 0.16 * _best_surface_match(_best_unit_text(best_unit), retrieval_intent.period)
    score += 0.1 * float(best_unit.get("ranking_score", 0.0) or 0.0)
    score += 0.18 * float(best_unit.get("table_confidence", 0.0) or 0.0)
    score += _table_family_preference_score(str(best_unit.get("table_family", "") or ""), retrieval_intent)
    return score


def _retrieval_focus_tokens(source_bundle: SourceBundle) -> list[str]:
    return retrieval_focus_tokens_impl(source_bundle)[:14]


def _officeqa_table_query(retrieval_intent: RetrievalIntent, source_bundle: SourceBundle) -> str:
    metric_identity = retrieval_intent.evidence_plan.metric_identity or retrieval_intent.metric
    parts = [
        retrieval_intent.entity,
        metric_identity,
        retrieval_intent.period,
    ]
    granularity = retrieval_intent.granularity_requirement
    period_type = retrieval_intent.period_type
    if granularity == "monthly_series":
        parts.append("monthly")
    elif granularity == "fiscal_year" or period_type == "fiscal_year":
        parts.append("fiscal year")
    elif granularity == "calendar_year" or period_type == "calendar_year":
        parts.append("calendar year")
    query = re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()[:280]
    if query:
        return query
    query_plan = retrieval_intent.query_plan
    fallback = query_plan.primary_semantic_query or query_plan.granularity_query or source_bundle.focus_query or source_bundle.task_text
    return re.sub(r"\s+", " ", str(fallback or "")).strip()[:280]


def _officeqa_row_query(retrieval_intent: RetrievalIntent, source_bundle: SourceBundle) -> str:
    return re.sub(
        r"\s+",
        " ",
        " ".join(part for part in [retrieval_intent.entity, source_bundle.focus_query, retrieval_intent.period] if part),
    ).strip()[:240]


def _officeqa_column_query(retrieval_intent: RetrievalIntent) -> str:
    parts = [retrieval_intent.metric]
    if retrieval_intent.aggregation_shape.startswith("monthly"):
        parts.append("month")
    if retrieval_intent.period:
        parts.append(retrieval_intent.period)
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()[:200]


def _strategy_chain(retrieval_intent: RetrievalIntent) -> list[str]:
    ordered = [retrieval_intent.strategy, *retrieval_intent.fallback_chain]
    return [item for item in dict.fromkeys(item for item in ordered if item)]


def _strategy_context(
    *,
    execution_mode: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    tool_plan: ToolPlan,
    journal: ExecutionJournal,
    registry: dict[str, dict[str, Any]],
    benchmark_overrides: dict[str, Any] | None,
    requested_strategy: str,
) -> RetrievalStrategyContext:
    return RetrievalStrategyContext(
        execution_mode=execution_mode,
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=journal,
        registry=registry,
        benchmark_overrides=benchmark_overrides,
        requested_strategy=requested_strategy,  # type: ignore[arg-type]
    )


def _strategy_prefers_text_first(retrieval_intent: RetrievalIntent, strategy: str) -> bool:
    if strategy == "text_first":
        return True
    if strategy != "hybrid":
        return False
    implicit_metric = not retrieval_intent.metric or retrieval_intent.metric.lower() in {"absolute percent change", "absolute difference"}
    return bool(retrieval_intent.evidence_plan.requires_text_support and implicit_metric)


def _document_ref_from_result(tool_result: dict[str, Any]) -> tuple[str, str]:
    facts = dict(tool_result.get("facts") or {})
    assumptions = dict(tool_result.get("assumptions") or {})
    document_id = str(assumptions.get("document_id", facts.get("document_id", "")) or "")
    path = str(assumptions.get("path", facts.get("citation", "")) or "")
    return document_id, path


def _used_document_ids(journal: ExecutionJournal) -> set[str]:
    used: set[str] = set()
    for result in journal.tool_results:
        document_id, _ = _document_ref_from_result(result)
        if document_id:
            used.add(document_id)
    return used


def _next_indexed_source_match(indexed_source_matches: list[dict[str, Any]], journal: ExecutionJournal) -> dict[str, Any] | None:
    used = _used_document_ids(journal)
    for match in indexed_source_matches:
        document_id = str(match.get("document_id", "") or "")
        if document_id and document_id in used:
            continue
        return match
    return None


def _used_table_queries(journal: ExecutionJournal) -> set[str]:
    used: set[str] = set()
    for result in journal.tool_results:
        assumptions = dict(result.get("assumptions") or {})
        query = re.sub(r"\s+", " ", str(assumptions.get("table_query", "") or "")).strip().lower()
        if query:
            used.add(query)
    return used


def _officeqa_table_query_variants(retrieval_intent: RetrievalIntent, source_bundle: SourceBundle) -> list[str]:
    plan = retrieval_intent.evidence_plan
    queries = [
        _officeqa_table_query(retrieval_intent, source_bundle),
        _normalize_query(" ".join(part for part in [retrieval_intent.entity, plan.metric_identity, retrieval_intent.period] if part)),
        _normalize_query(" ".join(part for part in [plan.metric_identity, retrieval_intent.period] if part)),
    ]
    queries.extend(_normalize_query(series) for series in plan.required_series[:3])
    if plan.requires_inflation_support:
        queries.append(_normalize_query(f"{retrieval_intent.period} CPI inflation price index"))
    return [query for query in dict.fromkeys(query for query in queries if query)]


def _candidate_table_query_hint(
    candidate: dict[str, Any],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
) -> str:
    best_unit = _candidate_best_evidence_unit(candidate)
    if not best_unit:
        return _officeqa_table_query(retrieval_intent, source_bundle)
    hinted = _normalize_query(
        " ".join(
            part
            for part in (
                retrieval_intent.entity,
                retrieval_intent.metric,
                retrieval_intent.period,
                " ".join(str(item or "") for item in list(best_unit.get("headers", []))[:4]),
                " ".join(str(item or "") for item in list(best_unit.get("row_labels", []))[:3]),
                str(best_unit.get("table_family", "") or ""),
            )
            if part
        )
    )
    return hinted or _officeqa_table_query(retrieval_intent, source_bundle)


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query or "").strip()[:280]


def _next_table_query(journal: ExecutionJournal, retrieval_intent: RetrievalIntent, source_bundle: SourceBundle) -> str:
    used = _used_table_queries(journal)
    for candidate in _officeqa_table_query_variants(retrieval_intent, source_bundle):
        normalized = candidate.lower()
        if normalized and normalized not in used:
            return candidate
    return _officeqa_table_query(retrieval_intent, source_bundle)


def _retrieval_content_text(facts: dict[str, Any]) -> str:
    parts: list[str] = []
    for chunk in facts.get("chunks", []):
        if isinstance(chunk, dict):
            parts.append(str(chunk.get("text", "")))
            parts.append(str(chunk.get("locator", "")))
            parts.append(str(chunk.get("citation", "")))
    for table in facts.get("tables", []):
        if isinstance(table, dict):
            parts.append(json.dumps(table, ensure_ascii=True))
    for summary in facts.get("numeric_summaries", []):
        if isinstance(summary, dict):
            parts.append(json.dumps(summary, ensure_ascii=True))
    metadata = dict(facts.get("metadata") or {})
    for key in ("file_name", "window"):
        if metadata.get(key):
            parts.append(str(metadata.get(key)))
    for key in ("citation", "document_id"):
        if facts.get(key):
            parts.append(str(facts.get(key)))
    return " ".join(part for part in parts if part).strip()


def _retrieved_evidence_is_sufficient(
    source_bundle: SourceBundle,
    tool_result: dict[str, Any],
    benchmark_overrides: dict[str, Any] | None = None,
) -> bool:
    if str(tool_result.get("retrieval_status", "") or "") in {"garbled_binary", "parse_error", "network_error", "unsupported_format", "empty", "irrelevant"}:
        return False
    sufficiency = assess_evidence_sufficiency(source_bundle.task_text, source_bundle, [tool_result], benchmark_overrides)
    return bool(sufficiency.is_sufficient)


def _retrieved_window_is_promising(
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    tool_result: dict[str, Any],
    benchmark_overrides: dict[str, Any] | None = None,
) -> bool:
    if str(tool_result.get("retrieval_status", "") or "") in {"garbled_binary", "parse_error", "network_error", "unsupported_format", "empty", "irrelevant"}:
        return False
    facts = dict(tool_result.get("facts") or {})
    text = _retrieval_content_text(facts).lower()
    entity_tokens = {token for token in _retrieval_tokens(retrieval_intent.entity) if token not in {"u", "s", "us"}}
    metric_basis = retrieval_intent.metric
    if retrieval_intent.aggregation_shape in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"} or retrieval_intent.metric in {"absolute percent change", "absolute difference"}:
        metric_basis = "expenditures"
    metric_tokens = set(_retrieval_tokens(metric_basis))
    period_tokens = set(_retrieval_tokens(retrieval_intent.period))
    overlap = len((entity_tokens | metric_tokens | period_tokens) & set(_retrieval_tokens(text)))
    citation = str(facts.get("citation", "") or dict(facts.get("metadata") or {}).get("file_name", "")).lower()
    if not (entity_tokens or metric_tokens or period_tokens) and any(token in citation for token in ("treasury", "bulletin", "statement", "budget")):
        return True
    if overlap >= 3:
        return True
    if overlap >= 2 and any(token in citation for token in ("treasury", "budget", "census", "annual", "statement")):
        return True
    if any(token in citation for token in ("treasury", "bulletin", "statement", "budget")) and overlap >= 1:
        return True
    return False


def _next_corpus_fetch_action(last_result: dict[str, Any]) -> RetrievalAction | None:
    facts = dict(last_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    assumptions = dict(last_result.get("assumptions") or {})
    if not metadata.get("has_more_chunks"):
        return None
    chunk_start = int(assumptions.get("chunk_start", metadata.get("chunk_start", 0)) or 0)
    chunk_limit = max(1, int(assumptions.get("chunk_limit", metadata.get("chunk_limit", 3)) or 3))
    document_id = str(assumptions.get("document_id", facts.get("document_id", "")) or "")
    path = str(assumptions.get("path", facts.get("citation", "")) or "")
    return RetrievalAction(
        action="tool",
        tool_name="fetch_corpus_document",
        stage="locate_pages",
        document_id=document_id,
        path=path,
        chunk_start=chunk_start + chunk_limit,
        chunk_limit=chunk_limit,
        rationale="Read the next document window because the current chunk is not sufficient yet.",
    )


def _next_officeqa_page_action(last_result: dict[str, Any], tool_name: str = "fetch_officeqa_pages") -> RetrievalAction | None:
    facts = dict(last_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    assumptions = dict(last_result.get("assumptions") or {})
    has_more = bool(metadata.get("has_more_windows") or metadata.get("has_more_chunks"))
    if not has_more:
        return None
    page_start = int(assumptions.get("page_start", metadata.get("page_start", metadata.get("chunk_start", 0))) or 0)
    page_limit = max(1, int(assumptions.get("page_limit", metadata.get("page_limit", metadata.get("chunk_limit", 5))) or 5))
    return RetrievalAction(
        action="tool",
        tool_name=tool_name,
        stage="locate_pages",
        document_id=str(assumptions.get("document_id", facts.get("document_id", "")) or ""),
        path=str(assumptions.get("path", facts.get("citation", "")) or ""),
        page_start=page_start + page_limit,
        page_limit=page_limit,
        rationale="Read the next OfficeQA page window because the current pages are still incomplete.",
    )


def _next_reference_fetch_action(last_result: dict[str, Any], tool_name: str) -> RetrievalAction | None:
    facts = dict(last_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    assumptions = dict(last_result.get("assumptions") or {})
    if not metadata.get("has_more_windows"):
        return None
    url = str(assumptions.get("url", facts.get("citation", "")) or "")
    document_id = str(assumptions.get("document_id", facts.get("document_id", "")) or "")
    path = str(assumptions.get("path", facts.get("citation", "")) or "")
    if not (url or document_id or path):
        return None
    window_kind = str(metadata.get("window_kind", "") or "").lower()
    if window_kind == "pages":
        page_start = int(assumptions.get("page_start", 0) or 0)
        page_limit = max(1, int(assumptions.get("page_limit", 5) or 5))
        return RetrievalAction(
            action="tool",
            tool_name=tool_name,
            url=url,
            document_id=document_id,
            path=path,
            page_start=page_start + page_limit,
            page_limit=page_limit,
            row_offset=int(assumptions.get("row_offset", 0) or 0),
            row_limit=max(100, int(assumptions.get("row_limit", 200) or 200)),
            rationale="Read the next page window because the current pages are not sufficient yet.",
        )
    if window_kind == "rows":
        row_offset = int(assumptions.get("row_offset", 0) or 0)
        row_limit = max(1, int(assumptions.get("row_limit", 200) or 200))
        return RetrievalAction(
            action="tool",
            tool_name=tool_name,
            url=url,
            document_id=document_id,
            path=path,
            page_start=int(assumptions.get("page_start", 0) or 0),
            page_limit=max(2, int(assumptions.get("page_limit", 5) or 5)),
            row_offset=row_offset + row_limit,
            row_limit=row_limit,
            rationale="Read the next row window because the current rows are not sufficient yet.",
        )
    return None


def _tool_result_citations(tool_result: dict[str, Any]) -> list[str]:
    facts = dict(tool_result.get("facts") or {})
    citations: list[str] = []
    direct = [facts.get("citation", "")]
    for item in facts.get("results", []):
        if isinstance(item, dict):
            direct.append(item.get("url", "") or item.get("citation", ""))
    for item in facts.get("documents", []):
        if isinstance(item, dict):
            direct.append(item.get("citation", "") or item.get("url", "") or item.get("path", ""))
    for item in facts.get("chunks", []):
        if isinstance(item, dict):
            direct.append(item.get("citation", ""))
    seen: set[str] = set()
    for raw in direct:
        citation = str(raw or "").strip()
        if not citation or citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
    return citations


def _search_result_candidates(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    return search_result_candidates_impl(tool_result)


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate.get("document_id", "") or "").strip().lower(),
        str(candidate.get("citation", "") or "").strip().lower(),
        str(candidate.get("path", "") or "").strip().lower(),
    )


def _merge_candidate_records(primary: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key in ("document_id", "citation", "path", "title", "snippet"):
        if not merged.get(key) and incoming.get(key):
            merged[key] = incoming.get(key)
    if incoming.get("title"):
        merged_title = str(merged.get("title", "") or "")
        merged_document_id = str(merged.get("document_id", "") or "")
        incoming_title = str(incoming.get("title", "") or "")
        if merged_title == merged_document_id and incoming_title != merged_document_id:
            merged["title"] = incoming_title
    primary_rank = int(merged.get("rank", 999) or 999)
    incoming_rank = int(incoming.get("rank", 999) or 999)
    merged["rank"] = min(primary_rank, incoming_rank)
    merged["score"] = max(float(merged.get("score", 0.0) or 0.0), float(incoming.get("score", 0.0) or 0.0))
    merged["metadata"] = {
        **dict(merged.get("metadata", {}) or {}),
        **dict(incoming.get("metadata", {}) or {}),
    }
    return merged


def _dedupe_search_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    index_by_identity: dict[tuple[str, str, str], int] = {}
    index_by_document_id: dict[str, int] = {}
    index_by_citation: dict[str, int] = {}

    for candidate in candidates:
        identity = _candidate_identity(candidate)
        document_id = identity[0]
        citation = identity[1]
        existing_index = index_by_identity.get(identity)
        if existing_index is None and document_id:
            existing_index = index_by_document_id.get(document_id)
        if existing_index is None and citation:
            existing_index = index_by_citation.get(citation)
        if existing_index is None:
            deduped.append(dict(candidate))
            current_index = len(deduped) - 1
        else:
            deduped[existing_index] = _merge_candidate_records(deduped[existing_index], candidate)
            current_index = existing_index
        merged = deduped[current_index]
        merged_identity = _candidate_identity(merged)
        index_by_identity[merged_identity] = current_index
        if merged_identity[0]:
            index_by_document_id[merged_identity[0]] = current_index
        if merged_identity[1]:
            index_by_citation[merged_identity[1]] = current_index
    return deduped


def _search_candidate_text(candidate: dict[str, Any]) -> str:
    return search_candidate_text_impl(candidate)


def _candidate_metadata_text(candidate: dict[str, Any]) -> str:
    return candidate_metadata_text_impl(candidate)


def _query_years(retrieval_intent: RetrievalIntent) -> set[str]:
    return {token for token in re.findall(r"\b((?:19|20)\d{2})\b", retrieval_intent.period or "")}


def _candidate_publication_years(candidate: dict[str, Any]) -> set[str]:
    metadata = dict(candidate.get("metadata", {}) or {})
    publication_year = str(metadata.get("publication_year", "") or "").strip()
    if re.fullmatch(r"(?:19|20)\d{2}", publication_year):
        return {publication_year}
    return set(
        re.findall(
            r"\b((?:19|20)\d{2})\b",
            " ".join(
                str(candidate.get(key, "") or "")
                for key in ("title", "citation", "path", "document_id")
            ),
        )
    )


def _candidate_best_evidence_unit(candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate_best_evidence_unit_impl(candidate)


def _query_entity_tokens(retrieval_intent: RetrievalIntent) -> set[str]:
    return {token for token in _retrieval_tokens(retrieval_intent.entity) if token not in {"u", "s", "us"}}


def _query_metric_tokens(retrieval_intent: RetrievalIntent) -> set[str]:
    metric_basis = retrieval_intent.metric
    if retrieval_intent.aggregation_shape in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"} or retrieval_intent.metric in {"absolute percent change", "absolute difference"}:
        metric_basis = "expenditures"
    return set(_retrieval_tokens(metric_basis))





def _granularity_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    metadata = dict(candidate.get("metadata", {}) or {})
    text = f"{_search_candidate_text(candidate)} {_candidate_metadata_text(candidate)}".lower()
    best_unit = _candidate_best_evidence_unit(candidate)
    granularity = retrieval_intent.granularity_requirement
    if granularity == "monthly_series":
        month_coverage = list(metadata.get("month_coverage", []))
        if str(best_unit.get("period_type", "") or "").lower() == "monthly_series":
            return 1.0
        if len(month_coverage) >= 6:
            return 0.9
        if any(token in text for token in ("monthly", "month", "receipts expenditures and balances", "january", "february", "march")):
            return 0.55
        if any(token in text for token in ("total 9/", "actual 6 months", "summary", "calendar year")):
            return -0.35
        return -0.15
    if granularity == "fiscal_year":
        if str(best_unit.get("period_type", "") or "").lower() == "fiscal_year":
            return 0.9
        if any(token in text for token in ("fiscal year", "fy ", "end of fiscal years")):
            return 0.65
        return -0.1
    if granularity == "calendar_year":
        if str(best_unit.get("period_type", "") or "").lower() == "calendar_year":
            return 0.55
        if any(token in text for token in ("calendar year", "annual", "summary", "actual 6 months", "estimate")):
            return 0.18
        return 0.0
    if granularity == "narrative_support":
        if any(token in text for token in ("discussion", "narrative", "commentary", "statement")):
            return 0.4
    return 0.0


def _category_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    metadata_text = _candidate_metadata_text(candidate).lower()
    best_unit = _candidate_best_evidence_unit(candidate)
    entity_tokens = _query_entity_tokens(retrieval_intent)
    metric_tokens = _query_metric_tokens(retrieval_intent)
    score = 0.0
    score += 0.18 * len(entity_tokens & set(_retrieval_tokens(metadata_text)))
    score += 0.12 * len(metric_tokens & set(_retrieval_tokens(metadata_text)))
    if entity_tokens and set(_retrieval_tokens(" ".join(str(item or "") for item in list(best_unit.get("row_labels", []))))) & entity_tokens:
        score += 0.18
    if metric_tokens and set(_retrieval_tokens(" ".join(str(item or "") for item in list(best_unit.get("headers", []))))) & metric_tokens:
        score += 0.14
    return score


def _year_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    metadata = dict(candidate.get("metadata", {}) or {})
    best_unit = _candidate_best_evidence_unit(candidate)
    candidate_years = {str(item) for item in list(metadata.get("years", [])) if str(item)}
    candidate_years.update(str(item) for item in list(best_unit.get("year_refs", [])) if str(item))
    required_years = set(retrieval_intent.target_years) or _query_years(retrieval_intent)
    publication_years = _candidate_publication_years(candidate)
    if not required_years:
        return 0.0
    preferred_publication_years = list(retrieval_intent.preferred_publication_years)
    publication_window = set(retrieval_intent.publication_year_window)
    explicit_scope = bool(retrieval_intent.publication_scope_explicit)
    score = 0.0
    for publication_year in publication_years:
        if publication_year in preferred_publication_years:
            position = preferred_publication_years.index(publication_year)
            if explicit_scope:
                score = max(score, max(0.2, 1.1 - (0.18 * position)))
            else:
                score = max(score, max(0.08, 0.34 - (0.04 * position)))
        elif publication_year in publication_window:
            score = max(score, 0.25 if explicit_scope else 0.06)
        elif publication_window:
            if retrieval_intent.retrospective_evidence_required:
                score -= 0.04
            elif retrieval_intent.retrospective_evidence_allowed:
                score -= 0.08
            else:
                score -= 0.22
    if candidate_years and required_years & candidate_years:
        score += 0.95
    if candidate_years and not (required_years & candidate_years):
        score -= 0.28
    if candidate_years and required_years & candidate_years and publication_years:
        acceptable_lag = max(0, int(retrieval_intent.acceptable_publication_lag_years or 0))
        required_year_ints = [int(year) for year in required_years if year.isdigit()]
        publication_year_ints = [int(year) for year in publication_years if year.isdigit()]
        if required_year_ints and publication_year_ints:
            target_max = max(required_year_ints)
            publication_min = min(publication_year_ints)
            if retrieval_intent.retrospective_evidence_required and publication_min >= target_max:
                lag = publication_min - target_max
                if lag <= max(acceptable_lag, 5):
                    score += max(0.18, 0.42 - (0.04 * lag))
            elif retrieval_intent.retrospective_evidence_allowed and publication_min >= target_max:
                lag = publication_min - target_max
                if lag <= max(acceptable_lag, 1):
                    score += max(0.08, 0.24 - (0.05 * lag))
    text = f"{_search_candidate_text(candidate)} {_candidate_metadata_text(candidate)}"
    text_years = set(re.findall(r"\b((?:19|20)\d{2})\b", text))
    if required_years & text_years:
        score += 0.35
    if text_years and not (required_years & text_years):
        score -= 0.12
    return score


def _exclusion_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    text = f"{_search_candidate_text(candidate)} {_candidate_metadata_text(candidate)}".lower()
    score = 0.0
    for term in retrieval_intent.exclude_constraints:
        tokens = _retrieval_tokens(term)
        if tokens and set(tokens).intersection(_retrieval_tokens(text)):
            score -= 0.45
    return score


def _historical_family_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    text = f"{_search_candidate_text(candidate)} {_candidate_metadata_text(candidate)}".lower()
    required_years = _query_years(retrieval_intent)
    if required_years and min(int(year) for year in required_years) <= 1945:
        if any(token in text for token in ("fiscal year", "end of fiscal years", "comparative", "statement", "veterans administration", "national defense")):
            return 0.35
    return 0.0


def _ranking_confidence(
    ranked: list[dict[str, Any]],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> tuple[float, float]:
    if not ranked:
        return 0.0, 0.0
    best = _search_candidate_score(ranked[0], retrieval_intent, source_bundle, benchmark_overrides)
    second = _search_candidate_score(ranked[1], retrieval_intent, source_bundle, benchmark_overrides) if len(ranked) > 1 else 0.0
    return best, best - second


def _candidate_pool_missing_preferred_publication_years(
    candidates: list[dict[str, Any]],
    retrieval_intent: RetrievalIntent,
) -> bool:
    preferred_years = [str(item) for item in list(retrieval_intent.preferred_publication_years or []) if str(item)]
    target_years = {str(item) for item in list(retrieval_intent.target_years or []) if str(item)}
    if not preferred_years or not target_years:
        return False
    preferred_head = preferred_years[0]
    if preferred_head in target_years:
        return False
    publication_years = {
        str(dict(candidate.get("metadata") or {}).get("publication_year", "") or "").strip()
        for candidate in list(candidates or [])[:5]
        if str(dict(candidate.get("metadata") or {}).get("publication_year", "") or "").strip()
    }
    if not publication_years:
        return False
    return preferred_head not in publication_years and publication_years.issubset(target_years)


def _latest_search_candidates(
    journal: ExecutionJournal,
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    for result in reversed(journal.tool_results):
        if str(result.get("type", "")) in {"search_officeqa_documents", "search_reference_corpus", "internet_search"}:
            return _rank_search_candidates(_search_result_candidates(result), retrieval_intent, source_bundle, benchmark_overrides)
    return []


def _next_ranked_source_candidate(
    journal: ExecutionJournal,
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    used_document_ids = _used_document_ids(journal)
    for candidate in _latest_search_candidates(journal, retrieval_intent, source_bundle, benchmark_overrides):
        document_id = str(candidate.get("document_id", "") or "")
        if document_id and document_id in used_document_ids:
            continue
        return candidate
    return None


def _table_family_matches_intent(table_family: str, retrieval_intent: RetrievalIntent) -> bool:
    return table_family_matches_intent_impl(table_family, retrieval_intent)


def _search_candidate_score(
    candidate: dict[str, Any],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> float:
    return search_candidate_score_impl(candidate, retrieval_intent, source_bundle, benchmark_overrides)


def _rank_search_candidates(
    candidates: list[dict[str, Any]],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return rank_search_candidates_impl(candidates, retrieval_intent, source_bundle, benchmark_overrides)


def _strategy_reason(retrieval_intent: RetrievalIntent) -> str:
    plan = retrieval_intent.evidence_plan
    reasons: list[str] = []
    if retrieval_intent.strategy == "table_first":
        reasons.append("primary metric is expected to be recoverable from structured table evidence")
    if retrieval_intent.strategy == "text_first":
        reasons.append("question asks for narrative or implicit metric support before table narrowing")
    if retrieval_intent.strategy == "hybrid":
        reasons.append("question needs both structured numeric evidence and narrative grounding")
    if retrieval_intent.strategy == "multi_table":
        reasons.append("question likely requires joining multiple table fragments before compute")
    if retrieval_intent.strategy == "multi_document":
        reasons.append("question spans multiple benchmark-linked source documents")
    if plan.requires_inflation_support:
        reasons.append("inflation support is required")
    if plan.requires_statistical_series:
        reasons.append("a complete series is required for statistical analysis")
    if plan.requires_forecast_support:
        reasons.append("a grounded time series is required for forecasting")
    if plan.join_keys:
        reasons.append(f"join keys: {', '.join(plan.join_keys[:4])}")
    return "; ".join(reasons) or "default retrieval strategy selected from the structured retrieval intent"


def _candidate_rejection_reason(
    candidate: dict[str, Any],
    score: float,
    retrieval_intent: RetrievalIntent,
) -> str:
    text = _search_candidate_text(candidate).lower()
    if any(term in text for term in retrieval_intent.must_exclude_terms):
        return "excluded retrieval term overlap"
    if any(
        bad in text
        for bad in (
            "monthly catalog",
            "public documents",
            "depository invoice",
            "federal register",
            "internal revenue bulletin",
            "cumulative bulletin",
            "flashcards",
            "quiz",
            "public law",
        )
    ):
        return "off-domain or noisy source"
    if score < 0.55:
        return "low semantic match score"
    return "lower-ranked than the selected candidates"


def _candidate_diagnostics(
    candidates: list[dict[str, Any]],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped_candidates = _dedupe_search_candidates(candidates)
    if not deduped_candidates:
        return [], []
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(deduped_candidates):
        score = round(_search_candidate_score(candidate, retrieval_intent, source_bundle, benchmark_overrides), 3)
        best_unit = _candidate_best_evidence_unit(candidate)
        metadata = dict(candidate.get("metadata", {}) or {})
        item = {
            "title": str(candidate.get("title", "") or ""),
            "citation": str(candidate.get("citation", "") or ""),
            "document_id": str(candidate.get("document_id", "") or ""),
            "rank": int(candidate.get("rank", 999) or 999),
            "score": score,
            "publication_year": str(metadata.get("publication_year", "") or ""),
            "best_table_family": str(best_unit.get("table_family", "") or ""),
            "metadata": metadata,
            "best_evidence_unit": best_unit,
        }
        if index < 5:
            kept.append(item)
        else:
            item["reason"] = _candidate_rejection_reason(candidate, score, retrieval_intent)
            rejected.append(item)
    return kept, rejected[:6]


def _source_file_candidate_diagnostics(indexed_source_matches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped_matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in indexed_source_matches:
        signature = (
            str(item.get("document_id", "") or "").strip().lower(),
            str(item.get("relative_path", "") or "").strip().lower(),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped_matches.append(item)
    kept = [
        {
            "title": str(item.get("relative_path", "") or item.get("document_id", "") or ""),
            "citation": str(item.get("relative_path", "") or ""),
            "document_id": str(item.get("document_id", "") or ""),
            "rank": index + 1,
            "score": 1.0,
        }
        for index, item in enumerate(deduped_matches[:3])
    ]
    rejected = [
        {
            "title": str(item.get("relative_path", "") or item.get("document_id", "") or ""),
            "citation": str(item.get("relative_path", "") or ""),
            "document_id": str(item.get("document_id", "") or ""),
            "rank": index + 4,
            "score": 0.95,
            "reason": "benchmark-linked source not selected in this hop",
        }
        for index, item in enumerate(deduped_matches[3:9])
    ]
    return kept, rejected


def _current_table_family(last_result: dict[str, Any]) -> str:
    facts = dict(last_result.get("facts") or {})
    tables = [item for item in list(facts.get("tables", [])) if isinstance(item, dict)]
    if not tables:
        return ""
    return str(tables[0].get("table_family", "") or dict(facts.get("metadata", {}) or {}).get("table_family", "") or "").strip()


def _table_candidates_from_result(last_result: dict[str, Any]) -> list[dict[str, Any]]:
    facts = dict(last_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    return [dict(item) for item in list(metadata.get("table_candidates", []) or []) if isinstance(item, dict)]


def _current_table_locator(last_result: dict[str, Any]) -> str:
    facts = dict(last_result.get("facts") or {})
    tables = [item for item in list(facts.get("tables", [])) if isinstance(item, dict)]
    if not tables:
        return ""
    current = dict(tables[0] or {})
    return str(current.get("table_locator", "") or current.get("locator", "") or "").strip()


def _best_same_document_table_candidate(
    last_result: dict[str, Any],
    retrieval_intent: RetrievalIntent,
) -> tuple[dict[str, Any] | None, float, float]:
    table_candidates = _table_candidates_from_result(last_result)
    if len(table_candidates) < 2:
        return None, 0.0, 0.0
    current_locator = _current_table_locator(last_result).lower()
    current_candidate = next(
        (
            item
            for item in table_candidates
            if str(item.get("locator", "") or "").strip().lower() == current_locator
        ),
        None,
    )
    current_score = _best_unit_alignment_score(current_candidate or {}, retrieval_intent)
    scored = sorted(
        (
            (_best_unit_alignment_score(item, retrieval_intent), item)
            for item in table_candidates
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not scored:
        return None, current_score, current_score
    best_score, best_candidate = scored[0]
    if str(best_candidate.get("locator", "") or "").strip().lower() == current_locator:
        return None, current_score, best_score
    if best_score <= max(0.75, current_score + 0.3):
        return None, current_score, best_score
    return dict(best_candidate), current_score, best_score


def _attach_retrieval_diagnostics(
    action: RetrievalAction,
    *,
    retrieval_intent: RetrievalIntent,
    journal: ExecutionJournal,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> RetrievalAction:
    if action.strategy:
        strategy = action.strategy
    else:
        strategy = retrieval_intent.strategy
        action.strategy = strategy
    action.strategy_reason = _strategy_reason(retrieval_intent)
    ranked = _rank_search_candidates(
        _search_result_candidates(journal.tool_results[-1] if journal.tool_results else {}),
        retrieval_intent,
        source_bundle,
        benchmark_overrides,
    )
    candidate_sources, rejected_candidates = _candidate_diagnostics(
        ranked,
        retrieval_intent,
        source_bundle,
        benchmark_overrides,
    )
    if not candidate_sources and source_bundle.source_files_found:
        candidate_sources, rejected_candidates = _source_file_candidate_diagnostics(list(source_bundle.source_files_found))
    action.candidate_sources = candidate_sources
    action.rejected_candidates = rejected_candidates
    return action


def _officeqa_planner_ops() -> OfficeQAPlannerOps:
    return OfficeQAPlannerOps(
        next_indexed_source_match=_next_indexed_source_match,
        ranking_confidence=_ranking_confidence,
        next_retrieval_query=_next_retrieval_query,
        candidate_pool_missing_preferred_publication_years=_candidate_pool_missing_preferred_publication_years,
        candidate_table_query_hint=_candidate_table_query_hint,
        retrieved_evidence_is_sufficient=_retrieved_evidence_is_sufficient,
        current_table_family=_current_table_family,
        table_family_matches_intent=_table_family_matches_intent,
        next_ranked_source_candidate=_next_ranked_source_candidate,
        best_same_document_table_candidate=_best_same_document_table_candidate,
        next_table_query=_next_table_query,
        next_officeqa_page_action=_next_officeqa_page_action,
        retrieved_window_is_promising=_retrieved_window_is_promising,
        normalize_query=_normalize_query,
    )


def _officeqa_planning_context(
    *,
    requested_strategy: str | None,
    active_strategy: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    journal: ExecutionJournal,
    benchmark_overrides: dict[str, Any] | None,
    overrides: dict[str, Any],
    officeqa_status: str,
    current_document_id: str,
    current_path: str,
    seed_query: str,
    indexed_source_matches: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    last_type: str,
    last_result: dict[str, Any],
    officeqa_search_tools: list[str],
    officeqa_table_tools: list[str],
    officeqa_row_tools: list[str],
    officeqa_cell_tools: list[str],
    officeqa_page_tools: list[str],
    document_search_tools: list[str],
    external_search_tools: list[str],
    allow_web_fallback: bool,
    prefer_text_first: bool,
    next_table_query: str,
    officeqa_table_query: str,
    officeqa_row_query: str,
    officeqa_column_query: str,
) -> OfficeQAPlanningContext:
    return OfficeQAPlanningContext(
        requested_strategy=requested_strategy,
        active_strategy=active_strategy,
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        journal=journal,
        benchmark_overrides=benchmark_overrides,
        overrides=overrides,
        officeqa_status=officeqa_status,
        current_document_id=current_document_id,
        current_path=current_path,
        seed_query=seed_query,
        indexed_source_matches=indexed_source_matches,
        candidates=candidates,
        last_type=last_type,
        last_result=last_result,
        officeqa_search_tools=officeqa_search_tools,
        officeqa_table_tools=officeqa_table_tools,
        officeqa_row_tools=officeqa_row_tools,
        officeqa_cell_tools=officeqa_cell_tools,
        officeqa_page_tools=officeqa_page_tools,
        document_search_tools=document_search_tools,
        external_search_tools=external_search_tools,
        allow_web_fallback=allow_web_fallback,
        prefer_text_first=prefer_text_first,
        next_table_query=next_table_query,
        officeqa_table_query=officeqa_table_query,
        officeqa_row_query=officeqa_row_query,
        officeqa_column_query=officeqa_column_query,
    )


def _fallback_retrieval_action(
    *,
    execution_mode: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    journal: ExecutionJournal,
    tool_plan: ToolPlan,
    registry: dict[str, dict[str, Any]],
    benchmark_overrides: dict[str, Any] | None = None,
    strategy_override: str | None = None,
) -> RetrievalAction:
    available = _retrieval_tools_available(tool_plan, registry)
    roles = _retrieval_tools_by_role(tool_plan, registry)
    document_search_tools = [name for name in roles["search"] if _tool_family(registry, name) == "document_retrieval"]
    document_fetch_tools = [name for name in roles["fetch"] if _tool_family(registry, name) == "document_retrieval"]
    discover_tools = [name for name in roles["discover"] if _tool_family(registry, name) == "document_retrieval"]
    external_search_tools = [name for name in roles["search"] if _tool_family(registry, name) == "external_retrieval"]
    last_result = journal.tool_results[-1] if journal.tool_results else {}
    last_type = str(last_result.get("type", ""))
    last_status = str(last_result.get("retrieval_status", "") or "")
    candidates = _rank_search_candidates(
        _search_result_candidates(last_result),
        retrieval_intent,
        source_bundle,
        benchmark_overrides,
    )
    seed_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
    overrides = dict(benchmark_overrides or {})
    officeqa_mode = str(overrides.get("benchmark_adapter") or "") == "officeqa"
    benchmark_policy = benchmark_runtime_policy(overrides)
    allow_web_fallback = bool(benchmark_policy.get("allow_web_fallback", True))
    corpus_grounded_only = execution_mode == "document_grounded_analysis" and officeqa_mode and not allow_web_fallback
    indexed_source_matches = list(source_bundle.source_files_found)
    officeqa_search_tools = [name for name in document_search_tools if name == "search_officeqa_documents"]
    officeqa_table_tools = [name for name in document_fetch_tools if name == "fetch_officeqa_table"]
    officeqa_row_tools = [name for name in document_fetch_tools if name == "lookup_officeqa_rows"]
    officeqa_cell_tools = [name for name in document_fetch_tools if name == "lookup_officeqa_cells"]
    officeqa_page_tools = [name for name in document_fetch_tools if name in {"fetch_officeqa_pages", "fetch_corpus_document"}]
    officeqa_table_query = _officeqa_table_query(retrieval_intent, source_bundle)
    officeqa_row_query = _officeqa_row_query(retrieval_intent, source_bundle)
    officeqa_column_query = _officeqa_column_query(retrieval_intent)

    if officeqa_mode:
        last_facts = dict(last_result.get("facts") or {})
        last_metadata = dict(last_facts.get("metadata") or {})
        officeqa_status = str(last_metadata.get("officeqa_status", "") or "")
        current_document_id, current_path = _document_ref_from_result(last_result)
        next_table_query = _next_table_query(journal, retrieval_intent, source_bundle) or officeqa_table_query
        strategy_chain = _strategy_chain(retrieval_intent)
        active_strategy = strategy_override or retrieval_intent.strategy or "table_first"
        if last_type in {"fetch_officeqa_table", "lookup_officeqa_rows", "lookup_officeqa_cells"} and officeqa_status in {
            "missing_table",
            "missing_row",
            "partial_table",
            "unit_ambiguity",
        }:
            if active_strategy == "table_first" and "hybrid" in strategy_chain:
                active_strategy = "hybrid"
            elif active_strategy in {"multi_table", "multi_document"} and "hybrid" in strategy_chain:
                active_strategy = "hybrid"
            elif "text_first" in strategy_chain:
                active_strategy = "text_first"
        elif last_type in {"fetch_officeqa_pages", "fetch_corpus_document"} and retrieval_intent.evidence_plan.requires_table_support:
            if active_strategy == "text_first" and "hybrid" in strategy_chain:
                active_strategy = "hybrid"
        prefer_text_first = _strategy_prefers_text_first(retrieval_intent, active_strategy)
        requested_strategy = strategy_override or retrieval_intent.strategy
        planner_ops = _officeqa_planner_ops()
        planner_context = _officeqa_planning_context(
            requested_strategy=requested_strategy,
            active_strategy=active_strategy,
            source_bundle=source_bundle,
            retrieval_intent=retrieval_intent,
            journal=journal,
            benchmark_overrides=benchmark_overrides,
            overrides=overrides,
            officeqa_status=officeqa_status,
            current_document_id=current_document_id,
            current_path=current_path,
            seed_query=seed_query,
            indexed_source_matches=indexed_source_matches,
            candidates=candidates,
            last_type=last_type,
            last_result=last_result,
            officeqa_search_tools=officeqa_search_tools,
            officeqa_table_tools=officeqa_table_tools,
            officeqa_row_tools=officeqa_row_tools,
            officeqa_cell_tools=officeqa_cell_tools,
            officeqa_page_tools=officeqa_page_tools,
            document_search_tools=document_search_tools,
            external_search_tools=external_search_tools,
            allow_web_fallback=allow_web_fallback,
            prefer_text_first=prefer_text_first,
            next_table_query=next_table_query,
            officeqa_table_query=officeqa_table_query,
            officeqa_row_query=officeqa_row_query,
            officeqa_column_query=officeqa_column_query,
        )

        if not journal.tool_results:
            return plan_initial_action(planner_context, planner_ops)

        if _tool_role(registry, last_type) in {"search", "discover"} and last_status in {"empty", "irrelevant"}:
            next_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
            if next_query and next_query != (journal.retrieval_queries[-1] if journal.retrieval_queries else ""):
                return RetrievalAction(
                    action="tool",
                    stage="identify_source",
                    requested_strategy=strategy_override or retrieval_intent.strategy,
                    strategy=active_strategy,
                    tool_name=last_type,
                    query=next_query,
                    rationale="Refine OfficeQA source search because the prior results were weak.",
                )

        if _tool_role(registry, last_type) in {"search", "discover"} and candidates:
            planned = plan_search_followup(planner_context, planner_ops)
            if planned is not None:
                return planned

        if last_type == "fetch_officeqa_table":
            planned = plan_table_followup(planner_context, planner_ops)
            if planned is not None:
                return planned

        if last_type == "lookup_officeqa_rows":
            planned = plan_row_followup(planner_context, planner_ops)
            if planned is not None:
                return planned

        if last_type == "lookup_officeqa_cells":
            planned = plan_cell_followup(planner_context, planner_ops)
            if planned is not None:
                return planned

        if last_type in {"fetch_officeqa_pages", "fetch_corpus_document"}:
            planned = plan_pages_followup(planner_context, planner_ops)
            if planned is not None:
                return planned

        if journal.retrieval_iterations >= MAX_RETRIEVAL_HOPS - 1:
            return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA retrieval hop budget exhausted.")
        if not corpus_grounded_only and external_search_tools and allow_web_fallback and last_type != external_search_tools[0]:
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                strategy=active_strategy,
                tool_name=external_search_tools[0],
                query=_next_retrieval_query(journal, retrieval_intent, source_bundle),
                rationale="Use explicit OfficeQA web fallback after corpus retrieval failed.",
            )
        
        next_ranked_candidate = _next_ranked_source_candidate(journal, retrieval_intent, source_bundle, benchmark_overrides)
        if next_ranked_candidate and officeqa_table_tools:
            return RetrievalAction(
                action="tool",
                stage="locate_table",
                strategy=active_strategy,
                tool_name=officeqa_table_tools[0],
                document_id=str(next_ranked_candidate.get("document_id", "")),
                path=str(next_ranked_candidate.get("path", "") or next_ranked_candidate.get("citation", "")),
                query=_candidate_table_query_hint(next_ranked_candidate, retrieval_intent, source_bundle) or next_table_query or officeqa_table_query,
                evidence_gap="validator rejection",
                rationale="Pivot to the next source because the current candidate failed evidence validation and no further deepening is available.",
            )

        return RetrievalAction(action="answer", stage="answer", rationale="No stronger OfficeQA retrieval action is available.")

    if not journal.tool_results:
        if indexed_source_matches and document_fetch_tools:
            first_match = indexed_source_matches[0]
            return RetrievalAction(
                action="tool",
                stage="locate_pages",
                tool_name=document_fetch_tools[0],
                document_id=str(first_match.get("document_id", "")),
                path=str(first_match.get("relative_path", "")),
                rationale="Read the benchmark-provided source file before broad corpus search.",
            )
        if source_bundle.urls and discover_tools:
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=discover_tools[0],
                query=source_bundle.task_text,
                rationale="Discover prompt-supplied reference files.",
            )
        if document_search_tools:
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=document_search_tools[0],
                query=seed_query,
                rationale="Search the grounded document source first.",
            )
        if source_bundle.urls and document_fetch_tools:
            return RetrievalAction(
                action="tool",
                stage="locate_pages",
                tool_name=document_fetch_tools[0],
                url=source_bundle.urls[0],
                rationale="Read the first supplied reference document.",
            )
        if external_search_tools and (not officeqa_mode or allow_web_fallback):
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=external_search_tools[0],
                query=seed_query,
                rationale="Search the web for a supporting source.",
            )
        return RetrievalAction(action="answer", stage="answer", rationale="No retrieval tools are available.")

    if _tool_role(registry, last_type) == "search" and last_status in {"empty", "irrelevant"}:
        next_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
        if next_query and next_query != (journal.retrieval_queries[-1] if journal.retrieval_queries else ""):
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=last_type,
                query=next_query,
                rationale="Refine the search because the prior results were not relevant enough.",
            )

    if _tool_role(registry, last_type) in {"search", "discover"} and candidates:
        first = candidates[0]
        best_score = _search_candidate_score(first, retrieval_intent, source_bundle, benchmark_overrides)
        next_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
        if best_score < 0.55 and next_query and next_query != (journal.retrieval_queries[-1] if journal.retrieval_queries else ""):
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=last_type,
                query=next_query,
                rationale="Refine the search because the top result is still weak or off-target.",
            )
        if _tool_family(registry, last_type) == "document_retrieval" and document_fetch_tools:
            return RetrievalAction(
                action="tool",
                stage="locate_pages",
                tool_name=document_fetch_tools[0],
                document_id=first.get("document_id", ""),
                path=first.get("path", "") or first.get("citation", ""),
                rationale="Read the top matching corpus document.",
            )
        if document_fetch_tools:
            return RetrievalAction(
                action="tool",
                stage="locate_pages",
                tool_name=document_fetch_tools[0],
                url=first.get("citation", ""),
                rationale="Read the top matching reference document.",
            )

    if _tool_family(registry, last_type) == "external_retrieval" and candidates and document_fetch_tools:
        return RetrievalAction(
            action="tool",
            stage="locate_pages",
            tool_name=document_fetch_tools[0],
            url=candidates[0].get("citation", ""),
            rationale="Open the top search result for grounded evidence.",
        )

    if _tool_role(registry, last_type) == "fetch":
        if _retrieved_evidence_is_sufficient(source_bundle, last_result, overrides):
            return RetrievalAction(action="answer", stage="answer", rationale="Retrieved document evidence is available for the final answer.")
        if (last_type == "fetch_reference_file" or _tool_family(registry, last_type) == "document_retrieval") and _retrieved_window_is_promising(
            source_bundle,
            retrieval_intent,
            last_result,
            overrides,
        ):
            next_window = _next_reference_fetch_action(last_result, last_type)
            if next_window is not None:
                return next_window
        if last_type == "fetch_corpus_document":
            next_window = _next_corpus_fetch_action(last_result)
            if next_window is not None:
                return next_window
        if corpus_grounded_only:
            return RetrievalAction(action="answer", stage="answer", rationale="No stronger grounded document evidence is available within the allowed retrieval budget.")

    if journal.retrieval_iterations >= MAX_RETRIEVAL_HOPS - 1:
        return RetrievalAction(action="answer", stage="answer", rationale="Retrieval hop budget exhausted.")

    if not corpus_grounded_only and external_search_tools and last_type != external_search_tools[0] and (not officeqa_mode or allow_web_fallback):
        query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
        return RetrievalAction(
            action="tool",
            stage="identify_source",
            tool_name=external_search_tools[0],
            query=query,
            rationale="Broaden search after insufficient local evidence.",
        )

    return RetrievalAction(action="answer", stage="answer", rationale="No better retrieval action is available.")


def _validate_retrieval_action(
    action: RetrievalAction,
    tool_plan: ToolPlan,
    registry: dict[str, dict[str, Any]],
) -> RetrievalAction:
    allowed_tools = set(_retrieval_tools_available(tool_plan, registry))
    if action.action != "tool":
        return action
    if action.tool_name not in allowed_tools:
        return RetrievalAction(action="answer", rationale="Planned retrieval tool is not available in the current tool plan.")
    if _tool_role(registry, action.tool_name) == "search" and not action.query.strip():
        return RetrievalAction(action="answer", rationale="Search action did not produce a usable query.")
    if _tool_role(registry, action.tool_name) == "fetch" and not (action.url.strip() or action.document_id.strip() or action.path.strip()):
        return RetrievalAction(action="answer", rationale="Corpus fetch action did not identify a document.")
    return action


def _plan_action_for_strategy(context: RetrievalStrategyContext) -> RetrievalAction:
    return _fallback_retrieval_action(
        execution_mode=context.execution_mode,
        source_bundle=context.source_bundle,
        retrieval_intent=context.retrieval_intent,
        journal=context.journal,
        tool_plan=context.tool_plan,
        registry=context.registry,
        benchmark_overrides=context.benchmark_overrides,
        strategy_override=context.requested_strategy,
    )


_RETRIEVAL_STRATEGY_KERNEL = RetrievalStrategyKernel(
    [
        FunctionRetrievalStrategyHandler(name="table_first", planner=_plan_action_for_strategy),
        FunctionRetrievalStrategyHandler(name="text_first", planner=_plan_action_for_strategy),
        FunctionRetrievalStrategyHandler(name="hybrid", planner=_plan_action_for_strategy),
        FunctionRetrievalStrategyHandler(name="multi_table", planner=_plan_action_for_strategy),
        FunctionRetrievalStrategyHandler(name="multi_document", planner=_plan_action_for_strategy),
    ]
)


def _plan_retrieval_action(
    *,
    execution_mode: str,
    task_family: str = "",
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    tool_plan: ToolPlan,
    journal: ExecutionJournal,
    registry: dict[str, dict[str, Any]],
    benchmark_overrides: dict[str, Any] | None = None,
    workpad: dict[str, Any] | None = None,
) -> RetrievalAction:
    available_tools = _retrieval_tools_available(tool_plan, registry)
    planning_signature = _retrieval_planning_signature(source_bundle, retrieval_intent, workpad)
    admissible_strategies = _RETRIEVAL_STRATEGY_KERNEL.admissible_strategies(retrieval_intent)
    journal_recommendation = recommend_strategy_order(
        task_family=task_family,
        retrieval_intent=retrieval_intent,
        admissible_strategies=admissible_strategies,
    )
    if journal_recommendation.ordered_strategies:
        admissible_strategies = [
            str(item)
            for item in journal_recommendation.ordered_strategies
            if str(item) in admissible_strategies
        ] + [item for item in admissible_strategies if item not in journal_recommendation.ordered_strategies]
    exhaustion_proof = _build_retrieval_exhaustion_proof(
        admissible_strategies=admissible_strategies,
        journal=journal,
        retrieval_intent=retrieval_intent,
        workpad=workpad,
        planning_signature=planning_signature,
    )
    exhaustion_proof["strategy_journal"] = journal_recommendation.model_dump()
    if not available_tools or journal.retrieval_iterations >= MAX_RETRIEVAL_HOPS:
        exhaustion_proof["terminal_reason"] = "no_remaining_retrieval_capacity"
        exhaustion_proof["candidate_universe_exhausted"] = True
        exhaustion_proof["benchmark_terminal_allowed"] = bool(exhaustion_proof.get("strategies_exhausted")) or journal.retrieval_iterations >= MAX_RETRIEVAL_HOPS
        return RetrievalAction(
            action="answer",
            rationale="No remaining retrieval capacity.",
            regime_change="strategy_exhausted",
            no_material_change=True,
            material_input_signature=planning_signature,
            exhaustion_proof=exhaustion_proof,
        )

    attempts = _retrieval_strategy_attempts(workpad)
    attempted_current_regime = {
        str(item.get("applied_strategy", "") or item.get("requested_strategy", "") or "").strip()
        for item in attempts
        if str(item.get("material_input_signature", "") or "") == planning_signature
        and str(item.get("applied_strategy", "") or item.get("requested_strategy", "") or "").strip()
    }
    rotation_requested, rotation_reason = _strategy_rotation_requested(workpad)
    preferred_strategy = _RETRIEVAL_STRATEGY_KERNEL.select_strategy(retrieval_intent)
    ordered_candidates = list(admissible_strategies)
    if rotation_requested and preferred_strategy in attempted_current_regime:
        rotated_strategy = _RETRIEVAL_STRATEGY_KERNEL.next_strategy(retrieval_intent, attempted_current_regime)
        if rotated_strategy:
            ordered_candidates = [rotated_strategy, *[item for item in admissible_strategies if item != rotated_strategy]]
    if not ordered_candidates:
        exhaustion_proof["terminal_reason"] = "no_admissible_retrieval_strategy"
        exhaustion_proof["strategies_exhausted"] = True
        exhaustion_proof["benchmark_terminal_allowed"] = True
        return RetrievalAction(
            action="answer",
            rationale="No admissible retrieval strategy is available.",
            regime_change="strategy_exhausted",
            no_material_change=True,
            material_input_signature=planning_signature,
            exhaustion_proof=exhaustion_proof,
        )

    duplicate_requested = False
    last_duplicate_signature = ""
    selected_action: RetrievalAction | None = None
    for index, requested_strategy in enumerate(ordered_candidates):
        heuristic = _RETRIEVAL_STRATEGY_KERNEL.plan_action(
            _strategy_context(
                execution_mode=execution_mode,
                source_bundle=source_bundle,
                retrieval_intent=retrieval_intent,
                tool_plan=tool_plan,
                journal=journal,
                registry=registry,
                benchmark_overrides=benchmark_overrides,
                requested_strategy=requested_strategy,
            )
        )
        planned = _validate_retrieval_action(heuristic, tool_plan, registry)
        if not planned.requested_strategy:
            planned.requested_strategy = requested_strategy
        if not planned.strategy:
            planned.strategy = requested_strategy or retrieval_intent.strategy
        planned.material_input_signature = _retrieval_action_material_input_signature(planned)
        strategy_name = str(planned.strategy or planned.requested_strategy or requested_strategy or "").strip()
        duplicate_requested = any(
            str(item.get("material_input_signature", "") or "") == planned.material_input_signature
            and str(item.get("applied_strategy", "") or item.get("requested_strategy", "") or "").strip() == strategy_name
            for item in attempts
        )
        if duplicate_requested:
            last_duplicate_signature = planned.material_input_signature
            continue
        if index > 0 or (rotation_requested and strategy_name != preferred_strategy):
            planned.regime_change = "strategy_rotation"
            planned.no_material_change = True
            if not planned.strategy_reason:
                planned.strategy_reason = rotation_reason or "Rotated to the next admissible retrieval strategy."
        elif not attempts:
            planned.regime_change = "initial_strategy"
        selected_action = planned
        break

    if selected_action is None:
        exhaustion_proof["terminal_reason"] = "strategies_exhausted_without_material_change"
        exhaustion_proof["strategies_exhausted"] = True
        exhaustion_proof["no_material_change"] = True
        exhaustion_proof["benchmark_terminal_allowed"] = True
        selected_action = RetrievalAction(
            action="answer",
            stage="answer",
            requested_strategy=ordered_candidates[0] if ordered_candidates else preferred_strategy,
            strategy=ordered_candidates[0] if ordered_candidates else preferred_strategy,
            strategy_reason=rotation_reason or "All admissible retrieval strategies would repeat materially identical inputs.",
            regime_change="strategy_exhausted",
            query=_derive_retrieval_seed_query(source_bundle, retrieval_intent),
            material_input_signature=last_duplicate_signature or planning_signature,
            no_material_change=True,
            exhaustion_proof=exhaustion_proof,
            rationale="No untried retrieval strategy remains for the current regime.",
        )

    if not selected_action.exhaustion_proof:
        selected_action.exhaustion_proof = exhaustion_proof
    if not selected_action.material_input_signature:
        selected_action.material_input_signature = planning_signature
    selected_action = _attach_retrieval_diagnostics(
        selected_action,
        retrieval_intent=retrieval_intent,
        journal=journal,
        source_bundle=source_bundle,
        benchmark_overrides=benchmark_overrides,
    )
    selected_action.exhaustion_proof = _build_retrieval_exhaustion_proof(
        admissible_strategies=admissible_strategies,
        journal=journal,
        retrieval_intent=retrieval_intent,
        workpad=workpad,
        planning_signature=planning_signature,
        terminal_reason=str(selected_action.exhaustion_proof.get("terminal_reason", "") or ""),
    )
    selected_action.exhaustion_proof["strategy_journal"] = journal_recommendation.model_dump()
    return selected_action


def _tool_args_from_retrieval_action(
    action: RetrievalAction,
    source_bundle: SourceBundle,
    registry: dict[str, dict[str, Any]],
    retrieval_intent: RetrievalIntent,
) -> dict[str, Any]:
    return tool_args_from_retrieval_action_impl(
        action,
        source_bundle,
        registry,
        retrieval_intent,
        _derive_retrieval_seed_query,
        _officeqa_table_query,
        _officeqa_row_query,
        _officeqa_column_query,
    )


def _retrieval_strategy_attempts(workpad: dict[str, Any] | None) -> list[dict[str, Any]]:
    return retrieval_strategy_attempts_impl(workpad)


def _retrieval_planning_signature(
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    workpad: dict[str, Any] | None = None,
) -> str:
    return retrieval_planning_signature_impl(source_bundle, retrieval_intent, workpad)


def _retrieval_action_material_input_signature(action: RetrievalAction) -> str:
    return retrieval_action_material_input_signature_impl(action)


def _strategy_rotation_requested(workpad: dict[str, Any] | None) -> tuple[bool, str]:
    return strategy_rotation_requested_impl(workpad)


def _last_regime_change(workpad: dict[str, Any] | None) -> str:
    return last_regime_change_impl(workpad)


def _last_officeqa_statuses(journal: ExecutionJournal) -> set[str]:
    return last_officeqa_statuses_impl(journal)


def _build_retrieval_exhaustion_proof(
    *,
    admissible_strategies: list[str],
    journal: ExecutionJournal,
    retrieval_intent: RetrievalIntent,
    workpad: dict[str, Any] | None,
    planning_signature: str,
    terminal_reason: str = "",
) -> dict[str, Any]:
    proof = build_retrieval_exhaustion_proof_impl(
        admissible_strategies=admissible_strategies,
        journal=journal,
        retrieval_intent=retrieval_intent,
        workpad=workpad,
        planning_signature=planning_signature,
        terminal_reason=terminal_reason,
    )
    proof["last_regime_change"] = last_regime_change_impl(workpad)
    proof["untried_strategies"] = list(proof.pop("untried_strategies_current_regime", proof.get("untried_strategies", [])))
    proof["no_material_change"] = bool(
        set(str(item.get("code", "") or "") for item in list(dict(workpad or {}).get("officeqa_repair_failures", []) or []) if isinstance(item, dict))
        & {"repair_applied_but_no_new_evidence", "repair_reused_stale_state"}
    )
    proof["recommended_repair_target"] = str(dict(dict(workpad or {}).get("officeqa_retry_policy") or {}).get("recommended_repair_target", "") or "")
    return proof
