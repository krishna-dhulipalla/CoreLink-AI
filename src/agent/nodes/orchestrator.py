"""Nodes for the active hybrid routing engine."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.contracts import (
    AnswerContract,
    CuratedContext,
    EvidenceSufficiency,
    ExecutionJournal,
    ProgressSignature,
    QualityReport,
    RetrievalAction,
    RetrievalIntent,
    ReviewPacket,
    SourceBundle,
    TaskIntent,
    ToolPlan,
    UnsupportedCapabilityReport,
)
from agent.context.evidence import _extract_policy_context
from agent.context.extraction import derive_market_snapshot
from agent.model_config import ChatOpenAI, get_client_kwargs, get_model_name, get_model_name_for_task, get_model_runtime_kwargs, invoke_structured_output
from agent.llm_control import (
    initial_officeqa_llm_control_state,
    maybe_review_evidence_commitment,
    maybe_rerank_source_candidates,
    maybe_review_table_admissibility,
    officeqa_llm_control_budget,
    record_officeqa_llm_usage,
    should_use_evidence_commit_llm,
    should_use_source_rerank_llm,
    should_use_table_rerank_llm,
)
from agent.llm_repair import (
    initial_officeqa_llm_repair_state,
    maybe_repair_from_validator,
    maybe_rewrite_retrieval_path,
    officeqa_llm_repair_budget,
)
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import latest_human_text
from agent.solver.options import (
    deterministic_options_final_answer,
    deterministic_policy_options_final_answer,
)
from agent.solver.quant import deterministic_quant_final_answer
from agent.tracer import format_messages_for_trace, get_tracer
from agent.capabilities import resolve_tool_plan
from agent.benchmarks.officeqa_validator import officeqa_orchestration_strategy
from agent.benchmarks import benchmark_compute_result, benchmark_task_intent, benchmark_validate_final
from agent.curated_context import (
    _compact_tool_findings,
    attach_compute_result,
    attach_structured_evidence,
    build_curated_context,
    build_review_packet,
    build_retrieval_bundle,
    build_source_bundle,
    solver_context_block,
)
from agent.nodes.orchestrator_intent import (
    _heuristic_intent,
    _normalize_task_intent,
    _supports_options_fast_path,
    _template_stub,
)
from agent.nodes.orchestrator_retrieval import (
    IS_COMPETITION_MODE as _IS_COMPETITION_MODE,
    MAX_RETRIEVAL_HOPS as _MAX_RETRIEVAL_HOPS,
    _derive_retrieval_seed_query,
    _next_corpus_fetch_action,
    _next_officeqa_page_action,
    _next_reference_fetch_action,
    _next_retrieval_query,
    _officeqa_column_query,
    _officeqa_row_query,
    _officeqa_table_query,
    _plan_retrieval_action,
    _rank_search_candidates,
    _retrieval_content_text,
    _retrieval_focus_tokens,
    _retrieval_tokens,
    _retrieval_tools_available,
    _retrieval_tools_by_role,
    _retrieved_evidence_is_sufficient,
    _retrieved_window_is_promising,
    _run_tool_step,
    _run_tool_step_with_args,
    _search_candidate_score,
    _search_result_candidates,
    _tool_args_from_retrieval_action,
    _tool_descriptor,
    _tool_family,
    _tool_lookup,
    _tool_result_citations,
    _tool_role,
)
from agent.prompts import (
    PLANNER_SYSTEM,
    EXECUTOR_SYSTEM,
    SELF_REFLECTION_SYSTEM,
    build_revision_prompt,
    contract_guidance,
    execution_guidance,
    heuristic_self_score,
    REUSABLE_TOOL_FAMILIES,
)
from agent.review_utils import looks_truncated, matches_exact_json_contract
from agent.retrieval_reasoning import assess_evidence_sufficiency, predictive_evidence_gaps
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)
RuntimeState = AgentState
_RETRIEVAL_EXECUTION_MODES = {"retrieval_augmented_analysis", "document_grounded_analysis"}
_OFFICEQA_SEARCH_DISCOVER_TOOLS = {
    "search_officeqa_documents",
    "search_reference_corpus",
    "internet_search",
    "list_reference_files",
}
_OFFICEQA_FETCH_TOOLS = {
    "fetch_officeqa_pages",
    "fetch_officeqa_table",
    "lookup_officeqa_rows",
    "lookup_officeqa_cells",
    "fetch_corpus_document",
    "fetch_reference_file",
}
_OFFICEQA_RETRIEVAL_TOOLS = _OFFICEQA_SEARCH_DISCOVER_TOOLS | _OFFICEQA_FETCH_TOOLS

# Prompts are centralized in prompts.py; orchestration code stays logic-only.


def _record_event(workpad: dict[str, Any], node: str, action: str) -> dict[str, Any]:
    updated = dict(workpad)
    events = list(updated.get("events", []))
    events.append({"node": node, "action": action})
    updated["events"] = events
    return updated


def _contract_status(answer_text: str, answer_contract: dict[str, Any], review_mode: str) -> str:
    if review_mode == "exact_quant":
        return "contract_ok" if matches_exact_json_contract(answer_text, answer_contract) else "contract_mismatch"
    if looks_truncated(answer_text):
        return "truncated"
    return "not_applicable"


def _build_progress_signature(
    *,
    execution_mode: str,
    selected_tools: list[str],
    missing_dimensions: list[str],
    artifact_signature: str,
    contract_status: str,
) -> ProgressSignature:
    payload = {
        "execution_mode": execution_mode,
        "selected_tools": sorted(selected_tools),
        "missing_dimensions": sorted(str(item) for item in missing_dimensions),
        "artifact_signature": artifact_signature,
        "contract_status": contract_status,
    }
    signature = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return ProgressSignature(signature=signature, **payload)


def fast_path_gate(state: RuntimeState) -> dict[str, Any]:
    step = increment_runtime_step()
    task_text = latest_human_text(state["messages"])
    answer_contract = state.get("answer_contract", {}) or {}
    benchmark_overrides = dict(state.get("benchmark_overrides") or {})
    intent, capability_flags, ambiguity_flags = _heuristic_intent(task_text, answer_contract, benchmark_overrides)
    workpad = dict(state.get("workpad", {}))
    workpad.setdefault("stage_history", [])
    workpad["stage_history"].append("FAST_PATH_GATE")
    workpad["routing_mode"] = "engine"
    workpad["fast_path_used"] = intent.planner_source == "fast_path"
    workpad = _record_event(
        workpad,
        "fast_path_gate",
        f"fast_path={workpad['fast_path_used']} family={intent.task_family} mode={intent.execution_mode}",
    )
    tracer = get_tracer()
    if tracer:
        tracer.record(
            "fast_path_gate",
            {
                "task_family": intent.task_family,
                "execution_mode": intent.execution_mode,
                "complexity_tier": intent.complexity_tier,
                "planner_source": intent.planner_source,
                "fast_path_used": workpad["fast_path_used"],
                "capability_flags": capability_flags,
                "ambiguity_flags": ambiguity_flags,
            },
        )
    logger.info("[Step %s] engine fast_path_gate -> fast_path=%s family=%s mode=%s", step, workpad["fast_path_used"], intent.task_family, intent.execution_mode)
    return {
        "task_intent": intent.model_dump() if intent.planner_source == "fast_path" else {},
        "task_profile": intent.task_family,
        "capability_flags": capability_flags,
        "ambiguity_flags": ambiguity_flags,
        "fast_path_used": intent.planner_source == "fast_path",
        "workpad": workpad,
    }


def task_planner(state: RuntimeState) -> dict[str, Any]:
    step = increment_runtime_step()
    existing = state.get("task_intent") or {}
    workpad = dict(state.get("workpad", {}))
    task_text = latest_human_text(state["messages"])
    answer_contract = state.get("answer_contract", {}) or {}
    benchmark_overrides = dict(state.get("benchmark_overrides") or {})
    capability_flags = list(state.get("capability_flags", []))
    ambiguity_flags = list(state.get("ambiguity_flags", []))

    if existing:
        intent = _normalize_task_intent(TaskIntent.model_validate(existing), task_text, TaskIntent.model_validate(existing))
        workpad["planner_output"] = intent.model_dump()
        workpad["review_mode"] = intent.review_mode
        workpad = _record_event(workpad, "task_planner", f"reused fast path intent -> {intent.execution_mode}")
        tracer = get_tracer()
        if tracer:
            tracer.record(
                "task_planner",
                {
                    "intent": intent.model_dump(),
                    "template_id": intent.execution_mode,
                    "planner_source": "fast_path_reuse",
                },
            )
        return {
            "task_intent": intent.model_dump(),
            "execution_template": _template_stub(intent),
            "workpad": workpad,
        }

    heuristic_intent, _, _ = _heuristic_intent(task_text, answer_contract, benchmark_overrides)
    intent = heuristic_intent
    if heuristic_intent.confidence < 0.9:
        messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(
                content=json.dumps(
                    {
                        "task": task_text,
                        "answer_contract": answer_contract,
                        "heuristic": heuristic_intent.model_dump(),
                        "capability_flags": capability_flags,
                        "ambiguity_flags": ambiguity_flags,
                    },
                    ensure_ascii=True,
                )
            ),
        ]
        try:
            parsed, _ = invoke_structured_output("profiler", TaskIntent, messages, temperature=0, max_tokens=220)
            candidate = TaskIntent.model_validate(parsed)
            if candidate.execution_mode == "exact_fast_path" and heuristic_intent.task_family != "finance_quant":
                candidate.execution_mode = heuristic_intent.execution_mode
            candidate.planner_source = "llm"
            intent = _normalize_task_intent(candidate, task_text, heuristic_intent)
        except Exception as exc:
            logger.info("Planner LLM fallback used heuristic intent: %s", exc)
    intent = _normalize_task_intent(intent, task_text, heuristic_intent)

    workpad["planner_output"] = intent.model_dump()
    workpad["review_mode"] = intent.review_mode
    workpad = _record_event(workpad, "task_planner", f"family={intent.task_family} mode={intent.execution_mode} source={intent.planner_source}")
    tracer = get_tracer()
    if tracer:
        tracer.record(
            "task_planner",
            {
                "intent": intent.model_dump(),
                "template_id": intent.execution_mode,
                "planner_source": intent.planner_source,
            },
        )
    logger.info("[Step %s] engine task_planner -> family=%s mode=%s", step, intent.task_family, intent.execution_mode)
    return {
        "task_intent": intent.model_dump(),
        "task_profile": intent.task_family,
        "execution_template": _template_stub(intent),
        "workpad": workpad,
    }


def make_capability_resolver(registry: dict[str, dict[str, Any]]):
    def capability_resolver(state: RuntimeState) -> dict[str, Any]:
        step = increment_runtime_step()
        task_text = latest_human_text(state["messages"])
        intent = TaskIntent.model_validate(state.get("task_intent") or {})
        benchmark_overrides = dict(state.get("benchmark_overrides") or {})
        source_bundle = SourceBundle.model_validate(
            state.get("source_bundle") or build_source_bundle(task_text, benchmark_overrides).model_dump()
        )
        tool_plan, _ = resolve_tool_plan(intent, source_bundle, registry, benchmark_overrides=benchmark_overrides)
        workpad = dict(state.get("workpad", {}))
        workpad["tool_plan"] = tool_plan.model_dump()
        workpad["ace_events"] = list(tool_plan.ace_events)
        unsupported_report: dict[str, Any] = {}
        if tool_plan.stop_reason == "unsupported_capability":
            unsupported_report = UnsupportedCapabilityReport(
                task_family=intent.task_family,
                requested_capability="non_finance_artifact_generation",
                reason="Task requires a capability outside the finance-first engine scope.",
            ).model_dump()
        workpad = _record_event(
            workpad,
            "capability_resolver",
            f"tools={','.join(tool_plan.selected_tools) if tool_plan.selected_tools else 'none'} blocked={','.join(tool_plan.blocked_families) if tool_plan.blocked_families else 'none'} stop={tool_plan.stop_reason or 'none'}",
        )
        tracer = get_tracer()
        if tracer:
            tracer.record(
                "capability_resolver",
                {
                    "selected_tools": tool_plan.selected_tools,
                    "pending_tools": tool_plan.pending_tools,
                    "blocked_families": tool_plan.blocked_families,
                    "widened_families": tool_plan.widened_families,
                    "ace_events": tool_plan.ace_events,
                    "stop_reason": tool_plan.stop_reason,
                },
            )
        template = _template_stub(intent, tool_plan.selected_tools)
        logger.info("[Step %s] engine capability_resolver -> selected=%s", step, tool_plan.selected_tools)
        return {
            "source_bundle": source_bundle.model_dump(),
            "tool_plan": tool_plan.model_dump(),
            "execution_template": template,
            "unsupported_capability_report": unsupported_report,
            "workpad": workpad,
        }

    return capability_resolver


def context_curator(state: RuntimeState) -> dict[str, Any]:
    step = increment_runtime_step()
    task_text = latest_human_text(state["messages"])
    answer_contract = state.get("answer_contract", {}) or {}
    benchmark_overrides = dict(state.get("benchmark_overrides") or {})
    intent = TaskIntent.model_validate(state.get("task_intent") or {})
    source_bundle = SourceBundle.model_validate(
        state.get("source_bundle") or build_source_bundle(task_text, benchmark_overrides).model_dump()
    )
    curated_context, evidence_stats = build_curated_context(
        task_text,
        answer_contract,
        intent,
        source_bundle,
        benchmark_overrides,
    )
    retrieval_intent, evidence_sufficiency = build_retrieval_bundle(task_text, source_bundle, benchmark_overrides)
    workpad = dict(state.get("workpad", {}))
    workpad["evidence_stats"] = evidence_stats
    workpad["task_complexity_tier"] = intent.complexity_tier
    workpad = _record_event(
        workpad,
        "context_curator",
        f"facts={len(curated_context.facts_in_use)} raw_tables={evidence_stats['raw_tables']} raw_formulas={evidence_stats['raw_formulas']}",
    )
    tracer = get_tracer()
    if tracer:
        tracer.record(
            "context_curator",
            {
                "objective_preview": curated_context.objective[:220],
                "fact_count": len(curated_context.facts_in_use),
                "evidence_stats": evidence_stats,
                "retrieval_intent": retrieval_intent.model_dump(),
                "evidence_sufficiency": evidence_sufficiency.model_dump(),
                "requested_output": curated_context.requested_output,
            },
        )
    budget = state.get("budget_tracker")
    if budget:
        budget.configure(
            complexity_tier=intent.complexity_tier,
            template_id=str((state.get("execution_template") or {}).get("template_id", "")),
            execution_mode=intent.execution_mode,
        )
    review_packet = build_review_packet(
        task_text=task_text,
        answer_text="",
        answer_contract=answer_contract,
        curated_context=curated_context.model_dump(),
        tool_results=[],
        evidence_sufficiency=evidence_sufficiency.model_dump(),
    )
    logger.info("[Step %s] engine context_curator -> facts=%s", step, len(curated_context.facts_in_use))
    return {
        "source_bundle": source_bundle.model_dump(),
        "benchmark_overrides": benchmark_overrides,
        "curated_context": curated_context.model_dump(),
        "review_packet": review_packet.model_dump(),
        "evidence_pack": {
            "curated_context": curated_context.model_dump(),
            "source_bundle_summary": {
                "entity_count": len(source_bundle.entities),
                "url_count": len(source_bundle.urls),
                "table_count": len(source_bundle.tables),
                "formula_count": len(source_bundle.formulas),
            },
        },
        "retrieval_intent": retrieval_intent.model_dump(),
        "evidence_sufficiency": evidence_sufficiency.model_dump(),
        "workpad": workpad,
    }



def _adaptive_completion_budget(
    mode: str,
    prompt_tokens: int,
    *,
    task_family: str = "",
    review_feedback: bool = False,
    task_text: str = "",
) -> int:
    normalized = (task_text or "").lower()
    derivation_heavy = task_family == "analytical_reasoning" or any(
        token in normalized for token in ("differentiate", "derivative", "integral", "marginal cost", "marginal revenue", "\\frac", "prove")
    )
    if mode == "scalar_or_json":
        return 600 if review_feedback else 400
    if mode == "long_form_derivation":
        if derivation_heavy or prompt_tokens >= 7000:
            return 4200 if _IS_COMPETITION_MODE else 3200
        if prompt_tokens >= 3200 or review_feedback:
            return 3600 if _IS_COMPETITION_MODE else 2800
        return 2800 if _IS_COMPETITION_MODE else 2200
    if mode == "document_grounded":
        if derivation_heavy or prompt_tokens >= 7000:
            return 4400 if _IS_COMPETITION_MODE else 3200
        if prompt_tokens >= 3500 or review_feedback:
            return 3800 if _IS_COMPETITION_MODE else 2800
        return 3000 if _IS_COMPETITION_MODE else 2400
    if mode == "advisory_memo":
        if task_family == "legal_transactional":
            if prompt_tokens >= 5000:
                return 4000 if _IS_COMPETITION_MODE else 3200
            if prompt_tokens >= 2400 or review_feedback:
                return 3400 if _IS_COMPETITION_MODE else 2800
            if prompt_tokens >= 1400:
                return 2800 if _IS_COMPETITION_MODE else 2400
            return 2400 if _IS_COMPETITION_MODE else 2200
        if prompt_tokens >= 3500 or review_feedback:
            return 3200 if _IS_COMPETITION_MODE else 2600
        if prompt_tokens >= 2000:
            return 2600 if _IS_COMPETITION_MODE else 2200
        return 2200 if _IS_COMPETITION_MODE else 1800
    if mode == "compact_sections":
        if task_family == "market_scenario":
            return 2000 if review_feedback or prompt_tokens >= 2400 else 1600
        if prompt_tokens >= 4000:
            return 2000
        if prompt_tokens >= 1800 or derivation_heavy:
            return 1800
        return 1200 if review_feedback else 1100
    if mode == "capability_gap":
        return 220
    return 400


def _artifact_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _officeqa_required_compute_insufficiency_answer(reasons: list[str]) -> str:
    compact_reasons = [str(item).strip() for item in reasons if str(item).strip()]
    rationale = "Structured evidence is insufficient for the required deterministic financial computation."
    if compact_reasons:
        rationale += " Missing or invalid support: " + ", ".join(compact_reasons[:3]) + "."
    return f"{rationale}\nFinal answer: Cannot calculate from the provided source evidence."


def _is_bounded_partial_answer(answer: str) -> bool:
    lowered = (answer or "").lower()
    support_markers = (
        "available evidence supports",
        "based on the retrieved evidence",
        "based on the available evidence",
        "grounded partial answer",
        "supported portion",
        "the retrieved evidence shows",
    )
    limitation_markers = (
        "cannot calculate exact",
        "cannot determine exact",
        "exact calculation is not supported",
        "partial answer",
        "remaining unsupported",
        "unsupported remainder",
        "unable to compute the exact",
    )
    return any(marker in lowered for marker in support_markers) and any(marker in lowered for marker in limitation_markers)


def _partial_answer_missing_dimensions(missing_dimensions: list[str]) -> list[str]:
    allowed = {
        "aggregation semantics",
        "exact grounded answer instead of an insufficiency placeholder",
        "exact numeric support from retrieved evidence",
        "monthly sum vs annual total alignment",
    }
    return [item for item in missing_dimensions if str(item).strip().lower() not in allowed]


def _orchestration_retrieval_strategy(orchestration_strategy: str, retrieval_intent: RetrievalIntent) -> str:
    if orchestration_strategy == "cross_document_comparison":
        return "multi_document"
    if orchestration_strategy == "hybrid_join":
        return retrieval_intent.strategy if retrieval_intent.strategy in {"hybrid", "multi_table", "multi_document"} else "hybrid"
    if orchestration_strategy == "text_reasoning":
        return "text_first"
    return "table_first"


def _apply_review_orchestration(retrieval_intent: RetrievalIntent, review_feedback: dict[str, Any] | None) -> RetrievalIntent:
    if not review_feedback:
        return retrieval_intent
    orchestration_strategy = str(review_feedback.get("orchestration_strategy", "") or "").strip()
    if not orchestration_strategy:
        return retrieval_intent
    updated = retrieval_intent.model_copy(deep=True)
    updated.strategy = _orchestration_retrieval_strategy(orchestration_strategy, retrieval_intent)  # type: ignore[assignment]
    if orchestration_strategy == "text_reasoning":
        updated.fallback_chain = [item for item in dict.fromkeys(["hybrid", "table_first", *updated.fallback_chain]) if item != updated.strategy]
    elif orchestration_strategy == "cross_document_comparison":
        updated.fallback_chain = [item for item in dict.fromkeys(["hybrid", "multi_table", "text_first", *updated.fallback_chain]) if item != updated.strategy]
    elif orchestration_strategy == "hybrid_join":
        updated.fallback_chain = [item for item in dict.fromkeys(["text_first", "table_first", *updated.fallback_chain]) if item != updated.strategy]
    return updated


def _officeqa_structured_value_text(value: dict[str, Any]) -> str:
    return " ".join(
        [
            " ".join(str(part) for part in list(value.get("row_path", []) or []) if str(part).strip()),
            str(value.get("row_label", "") or ""),
            " ".join(str(part) for part in list(value.get("column_path", []) or []) if str(part).strip()),
            str(value.get("column_label", "") or ""),
            str(value.get("table_locator", "") or ""),
            str(value.get("page_locator", "") or ""),
            str(value.get("document_id", "") or ""),
        ]
    ).strip()


def _officeqa_value_years(value: dict[str, Any]) -> set[str]:
    return set(re.findall(r"\b((?:19|20)\d{2})\b", _officeqa_structured_value_text(value)))


def _officeqa_value_has_month(value: dict[str, Any]) -> bool:
    lowered = _officeqa_structured_value_text(value).lower()
    return any(
        month in lowered
        for month in (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        )
    )


def _officeqa_value_semantic_score(value: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    text = _officeqa_structured_value_text(value).lower()
    tokens = set(_retrieval_tokens(text))
    entity_tokens = {
        token
        for token in _retrieval_tokens(retrieval_intent.entity)
        if token not in {"u", "s", "us"}
    }
    metric_basis = retrieval_intent.metric
    if retrieval_intent.metric in {"absolute percent change", "absolute difference"}:
        metric_basis = retrieval_intent.evidence_plan.metric_identity or "value"
    metric_tokens = set(_retrieval_tokens(metric_basis))
    required_years = set(re.findall(r"\b((?:19|20)\d{2})\b", retrieval_intent.period or ""))
    explicit_years = _officeqa_value_years(value)
    score = 0.0
    score += 0.45 * len(entity_tokens & tokens)
    score += 0.4 * len(metric_tokens & tokens)
    if required_years:
        if explicit_years & required_years:
            score += 0.7
        elif explicit_years:
            score -= 0.55
    if retrieval_intent.granularity_requirement == "monthly_series":
        score += 0.25 if _officeqa_value_has_month(value) else -0.4
    family = str(value.get("table_family", "") or "").lower()
    if family == "navigation_or_contents":
        score -= 2.0
    elif retrieval_intent.granularity_requirement == "monthly_series" and family == "monthly_series":
        score += 0.35
    elif retrieval_intent.granularity_requirement in {"calendar_year", "fiscal_year"} and family in {
        "category_breakdown",
        "annual_summary",
        "fiscal_year_comparison",
        "debt_or_balance_sheet",
    }:
        score += 0.2
    raw_value = str(value.get("raw_value", "") or "").strip().lower()
    column_label = str(value.get("column_label", "") or "").lower()
    if "issue and page number" in column_label and re.fullmatch(r"[a-z]?-?\d+(?:-\d+)?(?:\s+\d+(?:-\d+)?)*\.?", raw_value):
        score -= 2.0
    return score


def _reselect_officeqa_structured_evidence(
    structured_evidence: dict[str, Any] | None,
    retrieval_intent: RetrievalIntent,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    payload = dict(structured_evidence or {})
    values = [item for item in list(payload.get("values", []) or []) if isinstance(item, dict)]
    if len(values) < 2:
        return None, {}

    scored = sorted(
        (
            (_officeqa_value_semantic_score(value, retrieval_intent), value)
            for value in values
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score = scored[0][0]
    if best_score <= 0.15:
        return None, {}

    retained = [
        value
        for score, value in scored
        if score >= max(0.25, best_score - 0.75)
    ]
    if len(retained) == len(values):
        return None, {}

    retained_keys = {
        (
            str(item.get("document_id", "") or ""),
            str(item.get("citation", "") or ""),
            str(item.get("table_locator", "") or ""),
            str(item.get("page_locator", "") or ""),
        )
        for item in retained
    }
    filtered = dict(payload)
    filtered["values"] = retained
    filtered["value_count"] = len(retained)
    filtered["tables"] = [
        table
        for table in list(payload.get("tables", []) or [])
        if (
            str(table.get("document_id", "") or ""),
            str(table.get("citation", "") or ""),
            str(table.get("table_locator", table.get("locator", "")) or ""),
            str(table.get("page_locator", "") or ""),
        )
        in retained_keys
    ] or list(payload.get("tables", []) or [])
    diagnostics = {
        "attempted": True,
        "input_value_count": len(values),
        "retained_value_count": len(retained),
        "best_score": round(best_score, 4),
        "retention_threshold": round(max(0.25, best_score - 0.75), 4),
    }
    return filtered, diagnostics


def _record_llm_repair_decision(
    workpad: dict[str, Any],
    *,
    stage: str,
    trigger: str,
    decision: dict[str, Any],
    path_changed: bool,
    reroute_action: str = "",
    pre_retrieval_signature: str = "",
    post_retrieval_signature: str = "",
) -> dict[str, Any]:
    updated = dict(workpad)
    history = list(updated.get("officeqa_llm_repair_history", []))
    history.append(
        {
            "stage": stage,
            "trigger": trigger,
            "path_changed": path_changed,
            "llm_path_changed": path_changed,
            "document_pivot_triggered": False,
            "effective_retrieval_change": path_changed,
            "reroute_action": reroute_action,
            "pre_retrieval_signature": pre_retrieval_signature,
            "post_retrieval_signature": post_retrieval_signature,
            "decision": dict(decision or {}),
        }
    )
    updated["officeqa_llm_repair_history"] = history
    updated["officeqa_latest_llm_repair"] = history[-1]
    return updated


def _annotate_officeqa_repair_execution_outcome(
    workpad: dict[str, Any],
    *,
    retrieval_action: RetrievalAction,
    tool_result: dict[str, Any],
) -> dict[str, Any]:
    history = [dict(item) for item in list(workpad.get("officeqa_llm_repair_history", []) or []) if isinstance(item, dict)]
    if not history:
        return workpad
    latest = dict(history[-1] or {})
    if str(latest.get("stage", "") or "") not in {"retrieval_repair", "validator_repair"}:
        return workpad
    if bool(latest.get("execution_outcome_recorded", False)):
        return workpad

    facts = dict(tool_result.get("facts") or {})
    requested_document_id = str(retrieval_action.document_id or "").strip()
    resolved_document_id = str(facts.get("document_id", "") or "").strip()
    requested_path = str(retrieval_action.path or "").strip()
    resolved_path = str(facts.get("citation", "") or facts.get("path", "") or "").strip()
    document_pivot_triggered = False
    if requested_document_id and resolved_document_id:
        document_pivot_triggered = requested_document_id.lower() != resolved_document_id.lower()
    elif requested_path and resolved_path:
        document_pivot_triggered = requested_path.lower() != resolved_path.lower()

    llm_path_changed = bool(latest.get("path_changed", False))
    latest["llm_path_changed"] = llm_path_changed
    latest["requested_document_id"] = requested_document_id
    latest["resolved_document_id"] = resolved_document_id
    latest["requested_path"] = requested_path
    latest["resolved_path"] = resolved_path
    latest["document_pivot_triggered"] = document_pivot_triggered
    latest["effective_retrieval_change"] = bool(llm_path_changed or document_pivot_triggered)
    latest["execution_outcome_recorded"] = True

    updated = dict(workpad)
    history[-1] = latest
    updated["officeqa_llm_repair_history"] = history
    updated["officeqa_latest_llm_repair"] = latest
    for key in ("officeqa_latest_repair_transition", "officeqa_pending_repair_transition"):
        transition = dict(updated.get(key) or {})
        if transition:
            transition["requested_document_id"] = requested_document_id
            transition["resolved_document_id"] = resolved_document_id
            transition["requested_path"] = requested_path
            transition["resolved_path"] = resolved_path
            transition["document_pivot_triggered"] = document_pivot_triggered
            transition["effective_retrieval_change"] = bool(llm_path_changed or document_pivot_triggered)
            updated[key] = transition
    return updated


def _seed_officeqa_semantic_plan_usage(workpad: dict[str, Any], retrieval_intent: RetrievalIntent) -> dict[str, Any]:
    semantic_plan = retrieval_intent.semantic_plan
    if not semantic_plan.used_llm:
        return workpad
    existing = [dict(item) for item in list(workpad.get("officeqa_llm_usage", []) or []) if isinstance(item, dict)]
    if any(str(item.get("category", "") or "") == "semantic_plan_llm" for item in existing):
        return workpad
    return record_officeqa_llm_usage(
        workpad,
        category="semantic_plan_llm",
        used=True,
        reason=semantic_plan.rationale or "semantic_plan_llm",
        model_name=semantic_plan.model_name,
        confidence=semantic_plan.confidence,
        applied=True,
        details={
            "ambiguity_flags": list(semantic_plan.ambiguity_flags[:6]),
            "entity": semantic_plan.entity,
            "metric": semantic_plan.metric,
            "period": semantic_plan.period,
        },
    )


def _apply_source_rerank_decision(
    retrieval_action: RetrievalAction,
    candidate_sources: list[dict[str, Any]],
    decision_document_id: str,
) -> tuple[RetrievalAction, bool]:
    normalized_target = str(decision_document_id or "").strip().lower()
    if not normalized_target:
        return retrieval_action, False
    chosen = next(
        (
            dict(item)
            for item in candidate_sources
            if str(dict(item).get("document_id", "") or "").strip().lower() == normalized_target
        ),
        None,
    )
    if chosen is None:
        return retrieval_action, False
    if str(retrieval_action.document_id or "").strip().lower() == normalized_target:
        return retrieval_action, False
    updated = retrieval_action.model_copy(deep=True)
    updated.document_id = str(chosen.get("document_id", "") or "")
    updated.path = str(chosen.get("path", "") or chosen.get("citation", "") or "")
    updated.rationale = (
        f"{retrieval_action.rationale} LLM rerank selected a more semantically aligned source candidate."
    ).strip()
    return updated, True


def _table_candidates_from_last_result(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    facts = dict(tool_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    return [dict(item) for item in list(metadata.get("table_candidates", []) or []) if isinstance(item, dict)]


def _officeqa_trace_llm_usage(workpad: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(workpad.get("officeqa_llm_usage", []) or [])[:8] if isinstance(item, dict)]


def _apply_table_review_decision(
    retrieval_action: RetrievalAction,
    decision: dict[str, Any],
) -> tuple[RetrievalAction, dict[str, Any], bool]:
    locator = str(decision.get("preferred_table_locator", "") or "").strip()
    query = str(decision.get("suggested_table_query", "") or "").strip()
    if not query and locator:
        query = locator
    if not query:
        return retrieval_action, {}, False
    updated = retrieval_action.model_copy(deep=True)
    updated.action = "tool"
    updated.stage = "locate_table"
    updated.evidence_gap = str(decision.get("admissibility_gap", "") or retrieval_action.evidence_gap or "").strip()
    updated.query = query
    updated.rationale = (
        f"{retrieval_action.rationale} LLM table admissibility review requested a targeted table reselection."
    ).strip()
    return updated, {"officeqa_override_table_query": query}, True


def _officeqa_retrieval_input_signature(retrieval_intent: RetrievalIntent, workpad: dict[str, Any]) -> str:
    payload = {
        "strategy": retrieval_intent.strategy,
        "period_type": retrieval_intent.period_type,
        "target_years": list(retrieval_intent.target_years),
        "publication_year_window": list(retrieval_intent.publication_year_window),
        "preferred_publication_years": list(retrieval_intent.preferred_publication_years),
        "source_constraint_policy": retrieval_intent.source_constraint_policy,
        "granularity_requirement": retrieval_intent.granularity_requirement,
        "document_family": retrieval_intent.document_family,
        "aggregation_shape": retrieval_intent.aggregation_shape,
        "query_plan": retrieval_intent.query_plan.model_dump(),
        "include_constraints": list(retrieval_intent.include_constraints),
        "exclude_constraints": list(retrieval_intent.exclude_constraints),
        "must_include_terms": list(retrieval_intent.must_include_terms),
        "must_exclude_terms": list(retrieval_intent.must_exclude_terms),
        "override_query": str(workpad.get("officeqa_override_query", "") or ""),
        "override_table_query": str(workpad.get("officeqa_override_table_query", "") or ""),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]


def _expanded_officeqa_publication_scope(retrieval_intent: RetrievalIntent) -> tuple[list[str], list[str]]:
    seeds = {
        int(value)
        for value in [
            *list(retrieval_intent.target_years),
            *list(retrieval_intent.publication_year_window),
            *list(retrieval_intent.preferred_publication_years),
        ]
        if re.fullmatch(r"(?:19|20)\d{2}", str(value or "").strip())
    }
    if not seeds:
        return list(retrieval_intent.publication_year_window), list(retrieval_intent.preferred_publication_years)

    expanded_window = [str(year) for year in range(min(seeds) - 1, max(seeds) + 2)]
    preferred = [str(item) for item in list(retrieval_intent.preferred_publication_years) if str(item)]
    for year in expanded_window:
        if year not in preferred:
            preferred.append(year)
    return expanded_window, preferred


def _officeqa_tool_evidence_atoms(tool_result: dict[str, Any]) -> set[str]:
    tool_type = str(tool_result.get("type", "") or "").strip()
    if not tool_type:
        return set()
    facts = dict(tool_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    document_id = str(facts.get("document_id", "") or metadata.get("document_id", "") or "").strip()
    citation = str(facts.get("citation", "") or "").strip()
    atoms: set[str] = set()

    if tool_type in _OFFICEQA_SEARCH_DISCOVER_TOOLS:
        for candidate in _search_result_candidates(tool_result)[:8]:
            candidate_id = str(candidate.get("document_id", "") or "").strip()
            candidate_path = str(candidate.get("path", "") or candidate.get("citation", "") or "").strip()
            candidate_title = str(candidate.get("title", "") or "").strip()
            if candidate_id or candidate_path or candidate_title:
                atoms.add(f"search:{candidate_id}:{candidate_path}:{candidate_title}")
        return atoms

    if tool_type == "fetch_officeqa_table":
        table_locator = str(facts.get("table_locator", "") or metadata.get("table_locator", "") or "").strip()
        page_locator = str(facts.get("page_locator", "") or metadata.get("page_locator", "") or "").strip()
        atoms.add(f"table:{document_id}:{table_locator}:{page_locator}:{citation}")
        return atoms

    if tool_type == "lookup_officeqa_rows":
        table_locator = str(facts.get("table_locator", "") or metadata.get("table_locator", "") or "").strip()
        row_hits = list(facts.get("rows", []) or [])
        if row_hits:
            for row in row_hits[:12]:
                row_label = str(dict(row).get("row_label", "") or "").strip()
                atoms.add(f"row:{document_id}:{table_locator}:{row_label}")
        else:
            atoms.add(f"row:{document_id}:{table_locator}:{citation}")
        return atoms

    if tool_type == "lookup_officeqa_cells":
        table_locator = str(facts.get("table_locator", "") or metadata.get("table_locator", "") or "").strip()
        cells = list(facts.get("cells", []) or [])
        if cells:
            for cell in cells[:12]:
                cell_dict = dict(cell)
                atoms.add(
                    ":".join(
                        [
                            "cell",
                            document_id,
                            table_locator,
                            str(cell_dict.get("row_label", "") or "").strip(),
                            str(cell_dict.get("column_label", "") or "").strip(),
                            str(cell_dict.get("normalized_value", "") or cell_dict.get("value", "") or "").strip(),
                        ]
                    )
                )
        else:
            atoms.add(f"cell:{document_id}:{table_locator}:{citation}")
        return atoms

    if tool_type in _OFFICEQA_FETCH_TOOLS:
        page_locator = str(facts.get("page_locator", "") or metadata.get("page_locator", "") or "").strip()
        atoms.add(f"fetch:{tool_type}:{document_id}:{page_locator}:{citation}")
    return atoms


def _officeqa_structured_evidence_atoms(curated: CuratedContext) -> set[str]:
    structured = dict(curated.structured_evidence or {})
    atoms: set[str] = set()
    for table in list(structured.get("tables", []) or [])[:24]:
        table_dict = dict(table)
        atoms.add(
            ":".join(
                [
                    "table",
                    str(table_dict.get("document_id", "") or "").strip(),
                    str(table_dict.get("table_locator", "") or "").strip(),
                    str(table_dict.get("page_locator", "") or "").strip(),
                    str(table_dict.get("table_family", "") or "").strip(),
                ]
            )
        )
    for value in list(structured.get("values", []) or [])[:80]:
        value_dict = dict(value)
        atoms.add(
            ":".join(
                [
                    "value",
                    str(value_dict.get("document_id", "") or "").strip(),
                    str(value_dict.get("table_locator", "") or "").strip(),
                    str(value_dict.get("page_locator", "") or "").strip(),
                    " > ".join(str(part).strip() for part in list(value_dict.get("row_path", []) or []) if str(part).strip()),
                    " > ".join(str(part).strip() for part in list(value_dict.get("column_path", []) or []) if str(part).strip()),
                    str(value_dict.get("normalized_value", "") or value_dict.get("value", "") or "").strip(),
                ]
            )
        )
    return atoms


def _officeqa_evidence_signature(journal: ExecutionJournal, curated: CuratedContext) -> str:
    atoms: set[str] = set()
    atoms.update(str(item).strip() for item in list(journal.retrieved_citations) if str(item).strip())
    for tool_result in list(journal.tool_results or []):
        atoms.update(_officeqa_tool_evidence_atoms(dict(tool_result)))
    atoms.update(_officeqa_structured_evidence_atoms(curated))
    return hashlib.sha1(json.dumps(sorted(atoms), sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]


def _officeqa_reroute_action(stage: str, trigger: str, decision: dict[str, Any]) -> str:
    action = str(decision.get("decision", "") or "").strip()
    restart_scope = str(decision.get("restart_scope", "") or "").strip()
    normalized_trigger = " ".join([str(stage or "").strip().lower(), str(trigger or "").strip().lower()])
    if action == "widen_search_pool":
        return "search_pool_widening"
    if restart_scope == "semantic_plan_restart":
        return "semantic_plan_restart"
    if restart_scope == "cross_document":
        return "cross_document_restart"
    if restart_scope == "same_document" and action != "rewrite_query":
        return "same_document_restart"
    if action == "retune_table_query":
        if "unit" in normalized_trigger:
            return "unit_repair"
        return "table_query_rewrite"
    if action == "change_strategy":
        return "strategy_shift"
    if action == "rewrite_query":
        if any(fragment in normalized_trigger for fragment in ("wrong document", "wrong source", "source", "identify_source")):
            return "source_rerank"
        return "query_rewrite"
    return action or "keep"


def _filter_tool_results_for_reroute_action(
    tool_results: list[dict[str, Any]],
    reroute_action: str,
) -> list[dict[str, Any]]:
    if reroute_action in {
        "query_rewrite",
        "source_rerank",
        "strategy_shift",
        "search_pool_widening",
        "cross_document_restart",
        "semantic_plan_restart",
    }:
        return []
    if reroute_action in {"table_query_rewrite", "unit_repair", "same_document_restart"}:
        return [dict(item) for item in tool_results if str(dict(item).get("type", "") or "") in _OFFICEQA_SEARCH_DISCOVER_TOOLS]
    return [dict(item) for item in tool_results]


def _record_officeqa_repair_failure(
    workpad: dict[str, Any],
    *,
    code: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(workpad)
    failures = list(updated.get("officeqa_repair_failures", []) or [])
    payload = {
        "code": str(code or "").strip(),
        "details": dict(details or {}),
        "repair_epoch": int(updated.get("officeqa_repair_epoch", 0) or 0),
    }
    if payload["code"] and payload not in failures:
        failures.append(payload)
    updated["officeqa_repair_failures"] = failures
    updated["officeqa_latest_repair_failure"] = payload if payload["code"] else dict(updated.get("officeqa_latest_repair_failure") or {})
    return updated


def _invalidate_officeqa_repair_state(
    *,
    workpad: dict[str, Any],
    journal: ExecutionJournal,
    curated: CuratedContext,
    stage: str,
    trigger: str,
    decision: dict[str, Any],
    reroute_action: str,
    pre_retrieval_signature: str,
    post_retrieval_signature: str,
) -> tuple[dict[str, Any], ExecutionJournal, CuratedContext]:
    updated_workpad = dict(workpad)
    updated_journal = journal.model_copy(deep=True)
    updated_curated = curated.model_copy(deep=True)
    pre_evidence_signature = _officeqa_evidence_signature(journal, curated)
    retained_tool_results = _filter_tool_results_for_reroute_action(list(updated_journal.tool_results or []), reroute_action)
    updated_journal.tool_results = retained_tool_results
    updated_journal.retrieved_citations = []
    if reroute_action in {
        "query_rewrite",
        "source_rerank",
        "strategy_shift",
        "search_pool_widening",
        "cross_document_restart",
        "semantic_plan_restart",
    }:
        updated_journal.retrieval_queries = []
    updated_curated.structured_evidence = {}
    updated_curated.compute_result = {}
    provenance_summary = dict(updated_curated.provenance_summary or {})
    for key in ("retrieval_diagnostics", "evidence_plan_check", "compute_reselection", "answer_strategy"):
        provenance_summary.pop(key, None)
    updated_curated.provenance_summary = provenance_summary
    for key in (
        "retrieval_diagnostics",
        "officeqa_evidence_review",
        "officeqa_predictive_gaps",
        "officeqa_compute",
        "officeqa_answer_strategy",
        "officeqa_compute_reselection",
        "officeqa_compute_reselection_attempted",
        "review_ready",
        "completion_budget",
    ):
        updated_workpad.pop(key, None)
    repair_epoch = int(updated_workpad.get("officeqa_repair_epoch", 0) or 0) + 1
    transition = {
        "stage": stage,
        "trigger": trigger,
        "decision": str(decision.get("decision", "") or "").strip(),
        "reroute_action": reroute_action,
        "pre_retrieval_signature": pre_retrieval_signature,
        "post_retrieval_signature": post_retrieval_signature,
        "pre_evidence_signature": pre_evidence_signature,
        "retained_tool_result_count": len(retained_tool_results),
        "requires_fresh_retrieval": True,
        "status": "pending_fresh_retrieval",
    }
    updated_workpad["officeqa_repair_epoch"] = repair_epoch
    updated_workpad["officeqa_pending_repair_transition"] = transition
    updated_workpad["officeqa_latest_repair_transition"] = transition
    updated_workpad["officeqa_last_retrieval_input_signature"] = post_retrieval_signature
    if stage == "validator_repair":
        updated_workpad["officeqa_last_validator_repair_signature"] = post_retrieval_signature
    return updated_workpad, updated_journal, updated_curated


def _finalize_pending_officeqa_repair_transition(
    workpad: dict[str, Any],
    *,
    journal: ExecutionJournal,
    curated: CuratedContext,
    tool_name: str,
) -> dict[str, Any]:
    pending = dict(workpad.get("officeqa_pending_repair_transition") or {})
    if not pending or tool_name not in _OFFICEQA_RETRIEVAL_TOOLS:
        return workpad
    updated = dict(workpad)
    post_evidence_signature = _officeqa_evidence_signature(journal, curated)
    pending["completed_by_tool"] = tool_name
    pending["completed_retrieval_iteration"] = int(journal.retrieval_iterations or 0)
    pending["post_evidence_signature"] = post_evidence_signature
    if post_evidence_signature == str(pending.get("pre_evidence_signature", "") or ""):
        pending["status"] = "repair_applied_but_no_new_evidence"
        updated = _record_officeqa_repair_failure(
            updated,
            code="repair_applied_but_no_new_evidence",
            details={
                "reroute_action": pending.get("reroute_action", ""),
                "tool_name": tool_name,
                "stage": pending.get("stage", ""),
                "trigger": pending.get("trigger", ""),
            },
        )
    else:
        pending["status"] = "fresh_evidence_observed"
    updated["officeqa_latest_repair_transition"] = pending
    updated.pop("officeqa_pending_repair_transition", None)
    return updated


def _normalize_officeqa_provenance_term(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").replace("-", " ").strip().lower())


def _relax_officeqa_provenance_priors(retrieval_intent: RetrievalIntent) -> bool:
    changed = False
    normalized_source_file_query = _normalize_officeqa_provenance_term(retrieval_intent.query_plan.source_file_query)
    provenance_terms = {
        _normalize_officeqa_provenance_term(retrieval_intent.document_family),
        _normalize_officeqa_provenance_term(retrieval_intent.document_family.replace("_", " ")),
    }
    if retrieval_intent.document_family == "treasury_bulletin":
        provenance_terms.add("treasury bulletin")
    relaxed_terms: list[str] = []
    for term in list(retrieval_intent.must_include_terms or []):
        normalized = _normalize_officeqa_provenance_term(term)
        is_filename_hint = normalized.endswith("json") or ".json" in str(term or "").lower()
        if normalized and (normalized in provenance_terms or normalized == normalized_source_file_query or is_filename_hint):
            changed = True
            continue
        relaxed_terms.append(term)
    deduped_terms = list(dict.fromkeys([item for item in relaxed_terms if str(item).strip()]))
    if deduped_terms != list(retrieval_intent.must_include_terms):
        retrieval_intent.must_include_terms = deduped_terms
        changed = True
    source_file_query = str(retrieval_intent.query_plan.source_file_query or "").strip()
    if source_file_query:
        retrieval_intent.query_plan.source_file_query = ""
        retrieval_intent.query_candidates = [
            item
            for item in retrieval_intent.query_candidates
            if str(item).strip() and str(item).strip() != source_file_query
        ]
        changed = True
    return changed


def _restart_officeqa_query_universe_from_semantic_plan(retrieval_intent: RetrievalIntent) -> bool:
    base_queries = [
        str(retrieval_intent.query_plan.primary_semantic_query or "").strip(),
        str(retrieval_intent.query_plan.temporal_query or "").strip(),
        str(retrieval_intent.query_plan.granularity_query or "").strip(),
        str(retrieval_intent.query_plan.qualifier_query or "").strip(),
        str(retrieval_intent.query_plan.alternate_lexical_query or "").strip(),
    ]
    rebuilt = [item for item in dict.fromkeys(base_queries) if item]
    if rebuilt != list(retrieval_intent.query_candidates):
        retrieval_intent.query_candidates = rebuilt
        return True
    return False


def _apply_officeqa_llm_repair_decision(
    retrieval_intent: RetrievalIntent,
    workpad: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[RetrievalIntent, dict[str, Any], bool, str]:
    updated_intent = retrieval_intent.model_copy(deep=True)
    updated_workpad = dict(workpad)
    action = str(decision.get("decision", "") or "").strip()
    publication_scope_action = str(decision.get("publication_scope_action", "") or "").strip()
    restart_scope = str(decision.get("restart_scope", "") or "").strip()
    relax_provenance_priors = bool(decision.get("relax_provenance_priors", False))
    path_changed = False
    reroute_action = _officeqa_reroute_action("", "", decision)

    if action == "rewrite_query":
        revised_query = str(decision.get("revised_query", "") or "").strip()
        if revised_query:
            prior_primary = str(updated_intent.query_plan.primary_semantic_query or "").strip()
            updated_intent.query_plan.primary_semantic_query = revised_query
            updated_intent.query_candidates = [
                item
                for item in dict.fromkeys([revised_query, *updated_intent.query_candidates])
                if str(item).strip()
            ]
            updated_workpad["officeqa_override_query"] = revised_query
            path_changed = revised_query != prior_primary
    elif action == "retune_table_query":
        revised_table_query = str(decision.get("revised_table_query", "") or "").strip()
        if revised_table_query:
            prior_table_query = str(updated_workpad.get("officeqa_override_table_query", "") or "").strip()
            updated_workpad["officeqa_override_table_query"] = revised_table_query
            path_changed = revised_table_query != prior_table_query
    elif action == "change_strategy":
        preferred_strategy = str(decision.get("preferred_strategy", "") or "").strip()
        if preferred_strategy and preferred_strategy != updated_intent.strategy:
            updated_intent.strategy = preferred_strategy  # type: ignore[assignment]
            updated_intent.fallback_chain = [
                item
                for item in dict.fromkeys([*updated_intent.fallback_chain, retrieval_intent.strategy])
                if item and item != updated_intent.strategy
            ]
            path_changed = True
    elif action == "widen_search_pool":
        updated_workpad.pop("officeqa_override_query", None)
        updated_workpad.pop("officeqa_override_table_query", None)
        prior_policy = updated_intent.source_constraint_policy
        if prior_policy == "hard":
            updated_intent.source_constraint_policy = "soft"
            path_changed = True
        elif prior_policy != "off":
            updated_intent.source_constraint_policy = "off"
            path_changed = True
        expanded_window, expanded_preferred = _expanded_officeqa_publication_scope(updated_intent)
        if expanded_window != list(updated_intent.publication_year_window):
            updated_intent.publication_year_window = expanded_window
            path_changed = True
        if expanded_preferred != list(updated_intent.preferred_publication_years):
            updated_intent.preferred_publication_years = expanded_preferred
            path_changed = True
        source_file_query = str(updated_intent.query_plan.source_file_query or "").strip()
        if source_file_query:
            updated_intent.query_plan.source_file_query = ""
            updated_intent.query_candidates = [
                item
                for item in updated_intent.query_candidates
                if str(item).strip() and str(item).strip() != source_file_query
            ]
            path_changed = True

    if publication_scope_action == "widen_publication_horizon":
        expanded_window, expanded_preferred = _expanded_officeqa_publication_scope(updated_intent)
        if expanded_window != list(updated_intent.publication_year_window):
            updated_intent.publication_year_window = expanded_window
            path_changed = True
        if expanded_preferred != list(updated_intent.preferred_publication_years):
            updated_intent.preferred_publication_years = expanded_preferred
            path_changed = True
    elif publication_scope_action == "switch_to_retrospective":
        expanded_window, expanded_preferred = _expanded_officeqa_publication_scope(updated_intent)
        if expanded_window != list(updated_intent.publication_year_window):
            updated_intent.publication_year_window = expanded_window
            path_changed = True
        if expanded_preferred != list(updated_intent.preferred_publication_years):
            updated_intent.preferred_publication_years = expanded_preferred
            path_changed = True
        if not updated_intent.retrospective_evidence_allowed:
            updated_intent.retrospective_evidence_allowed = True
            path_changed = True
        if not updated_intent.retrospective_evidence_required:
            updated_intent.retrospective_evidence_required = True
            path_changed = True
        if updated_intent.publication_scope_explicit:
            updated_intent.publication_scope_explicit = False
            path_changed = True

    if relax_provenance_priors and _relax_officeqa_provenance_priors(updated_intent):
        path_changed = True

    if restart_scope == "semantic_plan_restart":
        updated_workpad.pop("officeqa_override_query", None)
        updated_workpad.pop("officeqa_override_table_query", None)
        updated_workpad["officeqa_restart_from_semantic_plan"] = True
        if _restart_officeqa_query_universe_from_semantic_plan(updated_intent):
            path_changed = True
    elif restart_scope == "cross_document":
        updated_workpad.pop("officeqa_override_table_query", None)
        if str(updated_workpad.get("officeqa_current_document_id", "") or "").strip():
            updated_workpad.pop("officeqa_current_document_id", None)
            path_changed = True
    elif restart_scope == "same_document":
        updated_workpad.pop("officeqa_override_table_query", None)

    return updated_intent, updated_workpad, path_changed, reroute_action


def _officeqa_retry_policy(
    officeqa_validation: Any,
    *,
    journal: ExecutionJournal,
    tool_plan: ToolPlan,
    retrieval_intent: RetrievalIntent,
) -> tuple[bool, str]:
    repair_target = str(getattr(officeqa_validation, "recommended_repair_target", "") or "none")
    orchestration_strategy = str(getattr(officeqa_validation, "orchestration_strategy", "") or officeqa_orchestration_strategy(retrieval_intent))
    if repair_target == "none":
        return False, "officeqa_no_repair_target"
    if repair_target == "gather":
        if journal.retrieval_iterations >= _MAX_RETRIEVAL_HOPS:
            return False, "officeqa_retry_exhausted"
        retrieval_tools = [
            tool_name
            for tool_name in tool_plan.selected_tools
            if tool_name in {"search_officeqa_documents", "fetch_officeqa_pages", "fetch_officeqa_table", "lookup_officeqa_rows", "lookup_officeqa_cells", "search_reference_corpus", "fetch_corpus_document", "fetch_reference_file", "list_reference_files", "internet_search"}
        ]
        if not retrieval_tools:
            return False, "officeqa_no_retrieval_repair_path"
        if orchestration_strategy == "cross_document_comparison" and not retrieval_intent.evidence_plan.requires_cross_source_alignment and len(retrieval_intent.fallback_chain) == 0:
            return False, "officeqa_cross_document_repair_not_supported"
    if repair_target == "compute":
        if not retrieval_intent.compute_policy == "required" and not retrieval_intent.evidence_plan.requires_table_support:
            return False, "officeqa_compute_repair_not_applicable"
    return True, ""


def _latest_public_answer(state: RuntimeState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return str(msg.content)
    return ""


def _exact_quant_answer(state: RuntimeState) -> str | None:
    source_bundle = SourceBundle.model_validate(state.get("source_bundle") or {})
    curated_context = CuratedContext.model_validate(state.get("curated_context") or {})
    relevant_rows = [fact["value"] for fact in curated_context.facts_in_use if fact.get("type") == "table_rows"]
    relevant_formulae = [fact["value"] for fact in curated_context.facts_in_use if fact.get("type") == "formula"]
    evidence_pack = {
        "relevant_rows": relevant_rows,
        "relevant_formulae": relevant_formulae,
        "tables": source_bundle.tables,
        "formulas": source_bundle.formulas,
        "entities": source_bundle.entities,
        "prompt_facts": source_bundle.inline_facts,
        "derived_facts": {},
        "policy_context": {},
    }
    adapter_state = {
        "messages": state.get("messages", []),
        "execution_template": {"template_id": "quant_inline_exact"},
        "evidence_pack": evidence_pack,
        "answer_contract": state.get("answer_contract", {}),
        "workpad": {"stage_outputs": {}, "tool_results": []},
        "assumption_ledger": state.get("assumption_ledger", []),
    }
    return deterministic_quant_final_answer(adapter_state)


def _options_final_answer(state: RuntimeState) -> str | None:
    source_bundle = SourceBundle.model_validate(state.get("source_bundle") or {})
    inline_facts = dict(source_bundle.inline_facts)
    market_snapshot, derived = derive_market_snapshot(source_bundle.task_text, inline_facts)
    policy_context = _extract_policy_context(source_bundle.task_text, "finance_options", ["needs_options_engine"])
    evidence_pack = {
        "derived_facts": derived,
        "prompt_facts": inline_facts,
        "market_snapshot": market_snapshot,
        "policy_context": policy_context,
    }
    option_state = {
        "messages": state.get("messages", []),
        "evidence_pack": evidence_pack,
        "assumption_ledger": state.get("assumption_ledger", []),
        "workpad": {
            "tool_results": ExecutionJournal.model_validate(state.get("execution_journal") or {}).tool_results,
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {
                "required_disclosures": [
                    "short-volatility / volatility-spike risk",
                    "tail loss / gap risk",
                    "downside scenario loss",
                ],
                "recommendation_class": "scenario_dependent_recommendation",
            },
        },
    }
    if policy_context.get("defined_risk_only") or policy_context.get("no_naked_options") or policy_context.get("retail_or_retirement_account"):
        return deterministic_policy_options_final_answer(option_state) or deterministic_options_final_answer(option_state)
    return deterministic_options_final_answer(option_state)


def make_executor(registry: dict[str, dict[str, Any]]):
    async def executor(state: RuntimeState) -> dict[str, Any]:
        step = increment_runtime_step()
        intent = TaskIntent.model_validate(state.get("task_intent") or {})
        tool_plan = ToolPlan.model_validate(state.get("tool_plan") or {})
        curated = CuratedContext.model_validate(state.get("curated_context") or {})
        source_bundle = SourceBundle.model_validate(state.get("source_bundle") or {})
        retrieval_intent = RetrievalIntent.model_validate(state.get("retrieval_intent") or {})
        evidence_sufficiency = EvidenceSufficiency.model_validate(state.get("evidence_sufficiency") or {})
        benchmark_overrides = dict(state.get("benchmark_overrides") or {})
        journal = ExecutionJournal.model_validate(state.get("execution_journal") or {})
        workpad = dict(state.get("workpad", {}))
        review_feedback = dict(state.get("review_feedback") or {})
        officeqa_mode = benchmark_overrides.get("benchmark_adapter") == "officeqa"
        targeted_retrieval_retry = bool(
            review_feedback
            and review_feedback.get("repair_target") == "gather"
            and review_feedback.get("retry_allowed", True)
        )
        targeted_compute_retry = bool(
            review_feedback
            and review_feedback.get("repair_target") == "compute"
            and review_feedback.get("retry_allowed", True)
        )
        active_retrieval_intent = _apply_review_orchestration(retrieval_intent, review_feedback)
        llm_control_state = (
            dict(workpad.get("officeqa_llm_control_state") or initial_officeqa_llm_control_state(active_retrieval_intent))
            if officeqa_mode
            else {}
        )
        llm_control_budget = (
            dict(workpad.get("officeqa_llm_control_budget") or officeqa_llm_control_budget(active_retrieval_intent))
            if officeqa_mode
            else {}
        )
        llm_repair_state = (
            dict(workpad.get("officeqa_llm_repair_state") or initial_officeqa_llm_repair_state(active_retrieval_intent))
            if officeqa_mode
            else {}
        )
        llm_repair_budget = (
            dict(workpad.get("officeqa_llm_repair_budget") or officeqa_llm_repair_budget())
            if officeqa_mode
            else {}
        )
        if officeqa_mode:
            workpad["officeqa_llm_control_state"] = llm_control_state
            workpad["officeqa_llm_control_budget"] = llm_control_budget
            workpad["officeqa_llm_repair_state"] = llm_repair_state
            workpad["officeqa_llm_repair_budget"] = llm_repair_budget
            workpad = _seed_officeqa_semantic_plan_usage(workpad, active_retrieval_intent)
        budget = state.get("budget_tracker")
        tracker = state.get("cost_tracker")
        tracer = get_tracer()
        tools_ran_this_call: list[str] = []
        task_text = latest_human_text(state["messages"])
        curated = attach_structured_evidence(curated, journal.tool_results, benchmark_overrides)

        if intent.task_family == "finance_options" and not _supports_options_fast_path(task_text):
            heuristic_intent, _, _ = _heuristic_intent(task_text, state.get("answer_contract", {}) or {}, benchmark_overrides)
            intent = _normalize_task_intent(heuristic_intent, task_text, heuristic_intent)
            tool_plan, _ = resolve_tool_plan(intent, source_bundle, registry, benchmark_overrides=benchmark_overrides)
            workpad = _record_event(workpad, "executor", f"rerouted before tool execution -> {intent.task_family}")

        if tool_plan.stop_reason == "unsupported_capability":
            evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
            journal.final_artifact_signature = _artifact_signature(
                "This task requires non-finance artifact generation that the active finance-first engine does not support."
            )
            journal.stop_reason = tool_plan.stop_reason
            content = (
                "This task requires non-finance artifact generation that the active finance-first engine does not support. "
                "I can analyze the request, but I cannot produce the requested media artifact."
            )
            workpad["completion_budget"] = 0
            workpad = _record_event(workpad, "executor", "unsupported capability -> clean stop")
            return {
                "messages": [AIMessage(content=content)],
                "task_intent": intent.model_dump(),
                "retrieval_intent": active_retrieval_intent.model_dump(),
                "tool_plan": tool_plan.model_dump(),
                "execution_journal": journal.model_dump(),
                "review_packet": build_review_packet(
                    task_text=task_text,
                    answer_text=content,
                    answer_contract=state.get("answer_contract", {}) or {},
                    curated_context=curated.model_dump(),
                    tool_results=[],
                    evidence_sufficiency=evidence_sufficiency.model_dump(),
                ).model_dump(),
                "evidence_sufficiency": evidence_sufficiency.model_dump(),
                "solver_stage": "COMPLETE",
                "workpad": workpad,
            }

        if tool_plan.stop_reason == "no_bindable_capability" and not tool_plan.pending_tools and not review_feedback:
            evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
            journal.final_artifact_signature = _artifact_signature(
                "I could not bind a safe retrieval or analysis capability for this task, so I am stopping instead of guessing."
            )
            journal.stop_reason = tool_plan.stop_reason
            content = (
                "I could not bind a safe retrieval or analysis capability for this task, so I am stopping instead of guessing."
            )
            workpad["completion_budget"] = 0
            workpad = _record_event(workpad, "executor", "no bindable capability -> clean stop")
            return {
                "messages": [AIMessage(content=content)],
                "task_intent": intent.model_dump(),
                "retrieval_intent": active_retrieval_intent.model_dump(),
                "tool_plan": tool_plan.model_dump(),
                "execution_journal": journal.model_dump(),
                "review_packet": build_review_packet(
                    task_text=task_text,
                    answer_text=content,
                    answer_contract=state.get("answer_contract", {}) or {},
                    curated_context=curated.model_dump(),
                    tool_results=[],
                    evidence_sufficiency=evidence_sufficiency.model_dump(),
                ).model_dump(),
                "evidence_sufficiency": evidence_sufficiency.model_dump(),
                "solver_stage": "COMPLETE",
                "workpad": workpad,
            }

        if intent.execution_mode == "exact_fast_path" and intent.task_family == "finance_quant":
            answer = _exact_quant_answer(state)
            if answer:
                evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                journal.final_artifact_signature = _artifact_signature(answer)
                workpad["review_ready"] = True
                workpad = _record_event(workpad, "executor", "exact_fast_path -> final draft ready")
                if tracer:
                    tracer.record(
                        "executor",
                        {
                            "intent": intent.model_dump(),
                            "used_llm": False,
                            "tools_ran": [],
                            "completion_budget": 0,
                            "output_preview": answer[:2000],
                        },
                    )
                return {
                    "messages": [AIMessage(content=answer)],
                    "task_intent": intent.model_dump(),
                    "execution_journal": journal.model_dump(),
                    "tool_plan": tool_plan.model_dump(),
                    "review_packet": build_review_packet(
                        task_text=task_text,
                        answer_text=answer,
                        answer_contract=state.get("answer_contract", {}) or {},
                        curated_context=curated.model_dump(),
                        tool_results=[],
                        evidence_sufficiency=evidence_sufficiency.model_dump(),
                    ).model_dump(),
                    "evidence_sufficiency": evidence_sufficiency.model_dump(),
                    "solver_stage": "SYNTHESIZE",
                "workpad": workpad,
            }

        if officeqa_mode and (targeted_retrieval_retry or targeted_compute_retry) and llm_repair_state.get("validator_repair_calls", 0) < llm_repair_budget.get("validator_repair_calls", 0):
            current_retrieval_signature = _officeqa_retrieval_input_signature(active_retrieval_intent, workpad)
            last_validator_repair_signature = str(workpad.get("officeqa_last_validator_repair_signature", "") or "").strip()
            validator_candidate_sources = list(dict(workpad.get("retrieval_diagnostics") or {}).get("candidate_sources") or [])
            repair_decision = None
            if current_retrieval_signature != last_validator_repair_signature:
                repair_decision = maybe_repair_from_validator(
                    task_text=task_text,
                    retrieval_intent=active_retrieval_intent,
                    execution_journal=journal,
                    workpad=workpad,
                    curated_context=curated,
                    review_feedback=review_feedback,
                    candidate_sources=validator_candidate_sources,
                )
            else:
                workpad = _record_officeqa_repair_failure(
                    workpad,
                    code="repair_reused_stale_state",
                    details={
                        "stage": "validator_repair",
                        "trigger": str(review_feedback.get("orchestration_strategy", "") or review_feedback.get("repair_target", "") or "validator_retry"),
                        "reason": "validator_repair_skipped_without_retrieval_input_change",
                    },
                )
            if repair_decision is not None:
                decision_payload = repair_decision.model_dump()
                validator_trigger = str(review_feedback.get("orchestration_strategy", "") or review_feedback.get("repair_target", "") or "validator_retry")
                active_retrieval_intent, workpad, path_changed, reroute_action = _apply_officeqa_llm_repair_decision(
                    active_retrieval_intent,
                    workpad,
                    decision_payload,
                )
                reroute_action = _officeqa_reroute_action("validator_repair", validator_trigger, decision_payload)
                post_retrieval_signature = _officeqa_retrieval_input_signature(active_retrieval_intent, workpad)
                if path_changed:
                    workpad, journal, curated = _invalidate_officeqa_repair_state(
                        workpad=workpad,
                        journal=journal,
                        curated=curated,
                        stage="validator_repair",
                        trigger=validator_trigger,
                        decision=decision_payload,
                        reroute_action=reroute_action,
                        pre_retrieval_signature=current_retrieval_signature,
                        post_retrieval_signature=post_retrieval_signature,
                    )
                    tool_plan.pending_tools = []
                llm_repair_state["validator_repair_calls"] = int(llm_repair_state.get("validator_repair_calls", 0) or 0) + 1
                workpad["officeqa_llm_repair_state"] = llm_repair_state
                workpad["solver_llm_decision"] = {
                    "used_llm": True,
                    "reason": "validator_directed_retrieval_repair",
                }
                workpad = _record_llm_repair_decision(
                    workpad,
                    stage="validator_repair",
                    trigger=validator_trigger,
                    decision=decision_payload,
                    path_changed=path_changed,
                    reroute_action=reroute_action,
                    pre_retrieval_signature=current_retrieval_signature,
                    post_retrieval_signature=post_retrieval_signature,
                )
                workpad = record_officeqa_llm_usage(
                    workpad,
                    category="repair_llm",
                    used=True,
                    reason="validator_directed_retrieval_repair",
                    model_name=str(decision_payload.get("model_name", "") or ""),
                    confidence=float(decision_payload.get("confidence", 0.0) or 0.0),
                    applied=path_changed,
                    details={"stage": "validator_repair", "trigger": validator_trigger, "decision": decision_payload.get("decision", "")},
                )
                workpad = _record_event(
                    workpad,
                    "executor",
                    f"structured validator repair -> {decision_payload.get('decision', 'keep')}",
                )

        if tool_plan.pending_tools and (not review_feedback or targeted_retrieval_retry):
            next_tool = tool_plan.pending_tools[0]
            if budget and budget.tool_calls_exhausted():
                budget.log_budget_exit("tool_budget_exhausted", f"Blocked tool '{next_tool}' after reaching tool-call cap.")
            else:
                tool_args, tool_result = await _run_tool_step(state, registry, next_tool)
                if tracker:
                    tracker.record_mcp_call()
                if budget:
                    budget.record_tool_call()
                tools_ran_this_call.append(next_tool)
                journal.tool_results.append(tool_result.model_dump())
                curated = attach_structured_evidence(curated, journal.tool_results, benchmark_overrides)
                if intent.execution_mode in _RETRIEVAL_EXECUTION_MODES:
                    journal.retrieval_iterations += 1
                    if next_tool in {"internet_search", "search_reference_corpus"}:
                        query = str(tool_args.get("query", "")).strip()
                        if query:
                            journal.retrieval_queries.append(query)
                    for citation in _tool_result_citations(tool_result.model_dump()):
                        if citation not in journal.retrieved_citations:
                            journal.retrieved_citations.append(citation)
                journal.routed_tool_families = list(dict.fromkeys([*journal.routed_tool_families, *tool_plan.tool_families_needed]))
                workpad = _record_event(workpad, "executor", f"ran tool {next_tool}")
                workpad = _finalize_pending_officeqa_repair_transition(
                    workpad,
                    journal=journal,
                    curated=curated,
                    tool_name=next_tool,
                )
                tool_plan.pending_tools = tool_plan.pending_tools[1:]
                evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                should_continue_after_tool = bool(tool_plan.pending_tools) or intent.execution_mode in _RETRIEVAL_EXECUTION_MODES
                if should_continue_after_tool and intent.execution_mode in {"advisory_analysis", "document_grounded_analysis", "tool_compute", "retrieval_augmented_analysis"}:
                    if not bool(dict(workpad.get("solver_llm_decision") or {}).get("used_llm")):
                        workpad["solver_llm_decision"] = {
                            "used_llm": False,
                            "reason": "llm_deferred_until_retrieval_complete",
                        }
                    if tracer:
                        tracer.record(
                            "executor",
                            {
                                "intent": intent.model_dump(),
                                "used_llm": bool(dict(workpad.get("solver_llm_decision") or {}).get("used_llm")),
                                "llm_decision_reason": str(dict(workpad.get("solver_llm_decision") or {}).get("reason", "") or "llm_deferred_until_retrieval_complete"),
                                "tools_ran": tools_ran_this_call,
                                "tool_results": _compact_tool_findings(journal.tool_results),
                                "output_preview": "",
                                "completion_budget": 0,
                                "officeqa_llm_usage": _officeqa_trace_llm_usage(workpad),
                                "llm_repair_history": list(workpad.get("officeqa_llm_repair_history", []) or []),
                            },
                        )
                    return {
                        "last_tool_result": tool_result.model_dump(),
                        "task_intent": intent.model_dump(),
                        "retrieval_intent": active_retrieval_intent.model_dump(),
                        "tool_plan": tool_plan.model_dump(),
                        "execution_journal": journal.model_dump(),
                        "curated_context": curated.model_dump(),
                        "evidence_sufficiency": evidence_sufficiency.model_dump(),
                        "solver_stage": "GATHER" if intent.execution_mode in _RETRIEVAL_EXECUTION_MODES else "COMPUTE",
                        "workpad": workpad,
                    }

        if intent.execution_mode in _RETRIEVAL_EXECUTION_MODES and (not review_feedback or targeted_retrieval_retry) and journal.retrieval_iterations < _MAX_RETRIEVAL_HOPS:
            retrieval_action = _plan_retrieval_action(
                execution_mode=intent.execution_mode,
                source_bundle=source_bundle,
                retrieval_intent=active_retrieval_intent,
                tool_plan=tool_plan,
                journal=journal,
                registry=registry,
                benchmark_overrides=benchmark_overrides,
            )
            if retrieval_action.action == "tool":
                if budget and budget.tool_calls_exhausted():
                    budget.log_budget_exit("tool_budget_exhausted", f"Blocked tool '{retrieval_action.tool_name}' after reaching tool-call cap.")
                else:
                  if (
                      officeqa_mode
                      and retrieval_action.candidate_sources
                      and retrieval_action.tool_name in {"fetch_officeqa_table", "fetch_officeqa_pages", "fetch_corpus_document"}
                      and llm_control_state.get("retrieval_rerank_calls", 0) < llm_control_budget.get("retrieval_rerank_calls", 0)
                  ):
                      source_review_needed, source_review_reason = should_use_source_rerank_llm(
                          retrieval_intent=active_retrieval_intent,
                          candidate_sources=retrieval_action.candidate_sources,
                          evidence_gap=retrieval_action.evidence_gap,
                      )
                      if source_review_needed:
                          rerank_decision = maybe_rerank_source_candidates(
                              task_text=task_text,
                              retrieval_intent=active_retrieval_intent,
                              candidate_sources=retrieval_action.candidate_sources,
                              reason=source_review_reason,
                          )
                          if rerank_decision is not None:
                              retrieval_action, rerank_applied = _apply_source_rerank_decision(
                                  retrieval_action,
                                  retrieval_action.candidate_sources,
                                  rerank_decision.preferred_document_id,
                              )
                              llm_control_state["retrieval_rerank_calls"] = int(llm_control_state.get("retrieval_rerank_calls", 0) or 0) + 1
                              workpad["officeqa_llm_control_state"] = llm_control_state
                              workpad = record_officeqa_llm_usage(
                                  workpad,
                                  category="retrieval_rerank_llm",
                                  used=True,
                                  reason=source_review_reason,
                                  model_name=rerank_decision.model_name,
                                  confidence=rerank_decision.confidence,
                                  applied=rerank_applied,
                                  details={
                                      "preferred_document_id": rerank_decision.preferred_document_id,
                                      "decision": rerank_decision.decision,
                                  },
                              )
                  if (
                      officeqa_mode
                      and journal.tool_results
                      and retrieval_action.tool_name in {"lookup_officeqa_rows", "lookup_officeqa_cells", "fetch_officeqa_pages"}
                      and llm_control_state.get("table_rerank_calls", 0) < llm_control_budget.get("table_rerank_calls", 0)
                  ):
                      last_tool_result = dict(journal.tool_results[-1] or {})
                      last_tool_type = str(last_tool_result.get("type", "") or "")
                      table_candidates = _table_candidates_from_last_result(last_tool_result)
                      if (
                          last_tool_type == "fetch_officeqa_table"
                      ):
                          table_review_needed, table_review_reason = should_use_table_rerank_llm(
                              retrieval_intent=active_retrieval_intent,
                              table_candidates=table_candidates,
                              evidence_gap=retrieval_action.evidence_gap,
                          )
                          if table_review_needed:
                              table_decision = maybe_review_table_admissibility(
                                  task_text=task_text,
                                  retrieval_intent=active_retrieval_intent,
                                  document_id=str(dict(last_tool_result.get("facts") or {}).get("document_id", "") or retrieval_action.document_id),
                                  table_candidates=table_candidates,
                                  reason=table_review_reason,
                              )
                              if table_decision is not None:
                                  retrieval_action, table_overrides, table_applied = _apply_table_review_decision(
                                      retrieval_action,
                                      table_decision.model_dump(),
                                  )
                                  if table_overrides:
                                      workpad.update(table_overrides)
                                  llm_control_state["table_rerank_calls"] = int(llm_control_state.get("table_rerank_calls", 0) or 0) + 1
                                  workpad["officeqa_llm_control_state"] = llm_control_state
                                  workpad = record_officeqa_llm_usage(
                                      workpad,
                                      category="table_rerank_llm",
                                      used=True,
                                      reason=table_review_reason,
                                      model_name=table_decision.model_name,
                                      confidence=table_decision.confidence,
                                      applied=table_applied,
                                      details={
                                          "decision": table_decision.decision,
                                          "preferred_table_locator": table_decision.preferred_table_locator,
                                          "preferred_table_family": table_decision.preferred_table_family,
                                      },
                                  )
                  if officeqa_mode and llm_repair_state.get("query_rewrite_calls", 0) < llm_repair_budget.get("query_rewrite_calls", 0):
                      current_retrieval_signature = _officeqa_retrieval_input_signature(active_retrieval_intent, workpad)
                      repair_decision = maybe_rewrite_retrieval_path(
                          task_text=task_text,
                          retrieval_intent=active_retrieval_intent,
                          source_bundle=source_bundle,
                          execution_journal=journal,
                          workpad=workpad,
                          curated_context=curated,
                          retrieval_strategy=retrieval_action.strategy,
                          evidence_gap=retrieval_action.evidence_gap,
                          current_query=retrieval_action.query,
                          current_table_query=_officeqa_table_query(active_retrieval_intent, source_bundle) if retrieval_action.tool_name in {"fetch_officeqa_table", "lookup_officeqa_rows", "lookup_officeqa_cells"} else "",
                          candidate_sources=retrieval_action.candidate_sources,
                      )
                      if repair_decision is not None:
                          decision_payload = repair_decision.model_dump()
                          repair_trigger = retrieval_action.evidence_gap or retrieval_action.stage or "retrieval_gap"
                          active_retrieval_intent, workpad, path_changed, reroute_action = _apply_officeqa_llm_repair_decision(
                              active_retrieval_intent,
                              workpad,
                              decision_payload,
                          )
                          reroute_action = _officeqa_reroute_action("retrieval_repair", repair_trigger, decision_payload)
                          post_retrieval_signature = _officeqa_retrieval_input_signature(active_retrieval_intent, workpad)
                          if path_changed:
                              workpad, journal, curated = _invalidate_officeqa_repair_state(
                                  workpad=workpad,
                                  journal=journal,
                                  curated=curated,
                                  stage="retrieval_repair",
                                  trigger=repair_trigger,
                                  decision=decision_payload,
                                  reroute_action=reroute_action,
                                  pre_retrieval_signature=current_retrieval_signature,
                                  post_retrieval_signature=post_retrieval_signature,
                              )
                              retrieval_action = _plan_retrieval_action(
                                  execution_mode=intent.execution_mode,
                                  source_bundle=source_bundle,
                                  retrieval_intent=active_retrieval_intent,
                                  tool_plan=tool_plan,
                                  journal=journal,
                                  registry=registry,
                                  benchmark_overrides=benchmark_overrides,
                              )
                          llm_repair_state["query_rewrite_calls"] = int(llm_repair_state.get("query_rewrite_calls", 0) or 0) + 1
                          workpad["officeqa_llm_repair_state"] = llm_repair_state
                          workpad["solver_llm_decision"] = {
                              "used_llm": True,
                              "reason": "structured_retrieval_repair",
                          }
                          workpad = _record_llm_repair_decision(
                              workpad,
                              stage="retrieval_repair",
                              trigger=repair_trigger,
                              decision=decision_payload,
                              path_changed=path_changed,
                              reroute_action=reroute_action,
                              pre_retrieval_signature=current_retrieval_signature,
                              post_retrieval_signature=post_retrieval_signature,
                          )
                          workpad = record_officeqa_llm_usage(
                              workpad,
                              category="repair_llm",
                              used=True,
                              reason="structured_retrieval_repair",
                              model_name=str(decision_payload.get("model_name", "") or ""),
                              confidence=float(decision_payload.get("confidence", 0.0) or 0.0),
                              applied=path_changed,
                              details={"stage": "retrieval_repair", "trigger": repair_trigger, "decision": decision_payload.get("decision", "")},
                          )
                          workpad = _record_event(
                              workpad,
                              "executor",
                              f"structured retrieval repair -> {decision_payload.get('decision', 'keep')}",
                          )
                  tool_args = _tool_args_from_retrieval_action(retrieval_action, source_bundle, registry, active_retrieval_intent)
                  override_query = str(workpad.get("officeqa_override_query", "") or "").strip()
                  if override_query and retrieval_action.tool_name in {"search_officeqa_documents", "search_reference_corpus", "internet_search"}:
                      tool_args["query"] = override_query
                      workpad.pop("officeqa_override_query", None)
                  override_table_query = str(workpad.get("officeqa_override_table_query", "") or "").strip()
                  if override_table_query and retrieval_action.tool_name in {"fetch_officeqa_table", "lookup_officeqa_rows", "lookup_officeqa_cells"}:
                      tool_args["table_query"] = override_table_query
                      workpad.pop("officeqa_override_table_query", None)
                  _, tool_result = await _run_tool_step_with_args(state, registry, retrieval_action.tool_name, tool_args)
                  if tracker:
                      tracker.record_mcp_call()
                  if budget:
                      budget.record_tool_call()
                  tools_ran_this_call.append(retrieval_action.tool_name)
                  journal.tool_results.append(tool_result.model_dump())
                  curated = attach_structured_evidence(curated, journal.tool_results, benchmark_overrides)
                  journal.retrieval_iterations += 1
                  if retrieval_action.tool_name in {"internet_search", "search_reference_corpus"} and tool_args.get("query"):
                      journal.retrieval_queries.append(str(tool_args["query"]))
                  for citation in _tool_result_citations(tool_result.model_dump()):
                      if citation not in journal.retrieved_citations:
                          journal.retrieved_citations.append(citation)
                  if retrieval_action.tool_name not in tool_plan.selected_tools:
                      tool_plan.selected_tools.append(retrieval_action.tool_name)
                  retrieval_diagnostics = {
                      "retrieval_decision": {
                          "tool_name": retrieval_action.tool_name,
                          "stage": retrieval_action.stage,
                          "strategy": retrieval_action.strategy,
                          "rationale": retrieval_action.rationale,
                          "evidence_gap": retrieval_action.evidence_gap,
                      },
                      "strategy_reason": retrieval_action.strategy_reason,
                      "candidate_sources": retrieval_action.candidate_sources,
                      "rejected_candidates": retrieval_action.rejected_candidates,
                  }
                  provenance_summary = dict(curated.provenance_summary or {})
                  provenance_summary["retrieval_diagnostics"] = retrieval_diagnostics
                  curated.provenance_summary = provenance_summary
                  workpad["retrieval_diagnostics"] = retrieval_diagnostics
                  if targeted_retrieval_retry:
                      workpad["officeqa_retry_path"] = {
                          "repair_target": review_feedback.get("repair_target", ""),
                          "orchestration_strategy": review_feedback.get("orchestration_strategy", ""),
                          "remediation_codes": list(review_feedback.get("remediation_codes", [])),
                      }
                  workpad = _record_event(workpad, "executor", f"retrieval hop -> {retrieval_action.tool_name}")
                  workpad = _annotate_officeqa_repair_execution_outcome(
                      workpad,
                      retrieval_action=retrieval_action,
                      tool_result=tool_result.model_dump(),
                  )
                  workpad = _finalize_pending_officeqa_repair_transition(
                      workpad,
                      journal=journal,
                      curated=curated,
                      tool_name=retrieval_action.tool_name,
                  )
                  if not bool(dict(workpad.get("solver_llm_decision") or {}).get("used_llm")):
                      workpad["solver_llm_decision"] = {
                          "used_llm": False,
                          "reason": "llm_deferred_until_retrieval_complete",
                      }
                  evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                  if tracer:
                      tracer.record(
                          "executor",
                          {
                              "intent": intent.model_dump(),
                              "used_llm": bool(dict(workpad.get("solver_llm_decision") or {}).get("used_llm")),
                              "llm_decision_reason": str(dict(workpad.get("solver_llm_decision") or {}).get("reason", "") or "llm_deferred_until_retrieval_complete"),
                              "tools_ran": tools_ran_this_call,
                              "tool_results": _compact_tool_findings(journal.tool_results),
                              "output_preview": "",
                              "completion_budget": 0,
                              "retrieval_action": retrieval_action.model_dump(),
                              "retrieval_decision": retrieval_diagnostics["retrieval_decision"],
                              "strategy_reason": retrieval_action.strategy_reason,
                              "candidate_sources": retrieval_action.candidate_sources,
                              "rejected_candidates": retrieval_action.rejected_candidates,
                              "evidence_gaps": [retrieval_action.evidence_gap] if retrieval_action.evidence_gap else [],
                              "orchestration_strategy": review_feedback.get("orchestration_strategy", "") if targeted_retrieval_retry else "",
                              "retry_path": dict(workpad.get("officeqa_retry_path") or {}),
                              "officeqa_llm_usage": _officeqa_trace_llm_usage(workpad),
                              "llm_repair_history": list(workpad.get("officeqa_llm_repair_history", []) or []),
                          },
                      )
                  return {
                      "last_tool_result": tool_result.model_dump(),
                      "task_intent": intent.model_dump(),
                      "retrieval_intent": active_retrieval_intent.model_dump(),
                      "tool_plan": tool_plan.model_dump(),
                      "execution_journal": journal.model_dump(),
                      "curated_context": curated.model_dump(),
                      "evidence_sufficiency": evidence_sufficiency.model_dump(),
                      "solver_stage": "GATHER",
                      "workpad": workpad,
                  }
            else:
                workpad = _record_event(workpad, "executor", f"retrieval ready to answer -> {retrieval_action.rationale or 'evidence collected'}")

        if benchmark_overrides.get("benchmark_adapter") == "officeqa" and workpad.get("officeqa_pending_repair_transition"):
            pending_transition = dict(workpad.get("officeqa_pending_repair_transition") or {})
            workpad = _record_officeqa_repair_failure(
                workpad,
                code="repair_reused_stale_state",
                details={
                    "stage": pending_transition.get("stage", ""),
                    "trigger": pending_transition.get("trigger", ""),
                    "reroute_action": pending_transition.get("reroute_action", ""),
                    "reason": "compute_or_review_reached_before_fresh_retrieval_hop",
                },
            )
            journal.stop_reason = "repair_reused_stale_state"
            answer = _officeqa_required_compute_insufficiency_answer(["repair reused stale state"])
            journal.final_artifact_signature = _artifact_signature(answer)
            workpad["completion_budget"] = 0
            workpad["review_ready"] = True
            workpad["solver_llm_decision"] = {
                "used_llm": False,
                "reason": "repair_reused_stale_state",
            }
            workpad = _record_event(workpad, "executor", "blocked stale evidence reuse after repair")
            return {
                "messages": [AIMessage(content=answer)],
                "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
                "task_intent": intent.model_dump(),
                "retrieval_intent": active_retrieval_intent.model_dump(),
                "tool_plan": tool_plan.model_dump(),
                "execution_journal": journal.model_dump(),
                "curated_context": curated.model_dump(),
                "review_packet": build_review_packet(
                    task_text=task_text,
                    answer_text=answer,
                    answer_contract=state.get("answer_contract", {}) or {},
                    curated_context=curated.model_dump(),
                    tool_results=journal.tool_results,
                    evidence_sufficiency=evidence_sufficiency.model_dump(),
                ).model_dump(),
                "evidence_sufficiency": evidence_sufficiency.model_dump(),
                "solver_stage": "SYNTHESIZE",
                "review_feedback": None,
                "workpad": workpad,
            }

        if benchmark_overrides.get("benchmark_adapter") == "officeqa" and journal.tool_results and curated.structured_evidence:
            predictive_gaps = predictive_evidence_gaps(active_retrieval_intent, curated.structured_evidence)
            workpad["officeqa_evidence_review"] = {
                "status": "insufficient" if predictive_gaps else "ready",
                "predictive_gaps": list(predictive_gaps),
                "compute_policy": active_retrieval_intent.compute_policy,
                "answer_mode": active_retrieval_intent.answer_mode,
                "strategy": active_retrieval_intent.strategy,
            }
            if predictive_gaps:
                merged_missing = list(dict.fromkeys([*evidence_sufficiency.missing_dimensions, *predictive_gaps]))
                evidence_sufficiency = EvidenceSufficiency(
                    source_family=evidence_sufficiency.source_family,
                    period_scope=evidence_sufficiency.period_scope,
                    aggregation_type=evidence_sufficiency.aggregation_type,
                    entity_scope=evidence_sufficiency.entity_scope,
                    is_sufficient=False,
                    missing_dimensions=merged_missing,
                    rationale="Structured evidence does not yet satisfy the planned retrieval requirements needed for deterministic compute.",
                )
                provenance_summary = dict(curated.provenance_summary or {})
                provenance_summary["evidence_plan_check"] = {
                    "status": "insufficient",
                    "predictive_gaps": predictive_gaps,
                    "strategy": active_retrieval_intent.strategy,
                }
                curated.provenance_summary = provenance_summary
                workpad["officeqa_predictive_gaps"] = predictive_gaps
                workpad["solver_llm_decision"] = {
                    "used_llm": False,
                    "reason": "required_compute_blocked_by_predictive_gaps",
                }
                workpad = _record_event(
                    workpad,
                    "executor",
                    f"predictive evidence gaps -> {', '.join(predictive_gaps[:4])}",
                )
                if tracer:
                    tracer.record(
                        "executor",
                        {
                            "intent": intent.model_dump(),
                            "used_llm": False,
                            "llm_decision_reason": "required_compute_blocked_by_predictive_gaps",
                            "tools_ran": tools_ran_this_call,
                            "output_preview": "",
                            "completion_budget": 0,
                            "evidence_gaps": predictive_gaps,
                            "strategy_reason": active_retrieval_intent.evidence_plan.objective or active_retrieval_intent.strategy,
                            "evidence_review": dict(workpad.get("officeqa_evidence_review") or {}),
                            "officeqa_llm_usage": _officeqa_trace_llm_usage(workpad),
                        },
                    )
                if active_retrieval_intent.compute_policy == "required":
                    if "low-confidence structure" in predictive_gaps:
                        journal.stop_reason = "officeqa_low_confidence_structure"
                    answer = _officeqa_required_compute_insufficiency_answer(predictive_gaps)
                    journal.final_artifact_signature = _artifact_signature(answer)
                    workpad["completion_budget"] = 0
                    workpad["review_ready"] = True
                    workpad = _record_event(
                        workpad,
                        "executor",
                        "required compute blocked by predictive evidence gaps",
                    )
                    return {
                        "messages": [AIMessage(content=answer)],
                        "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
                        "task_intent": intent.model_dump(),
                        "retrieval_intent": active_retrieval_intent.model_dump(),
                        "tool_plan": tool_plan.model_dump(),
                        "execution_journal": journal.model_dump(),
                        "curated_context": curated.model_dump(),
                        "review_packet": build_review_packet(
                            task_text=task_text,
                            answer_text=answer,
                            answer_contract=state.get("answer_contract", {}) or {},
                            curated_context=curated.model_dump(),
                            tool_results=journal.tool_results,
                            evidence_sufficiency=evidence_sufficiency.model_dump(),
                        ).model_dump(),
                        "evidence_sufficiency": evidence_sufficiency.model_dump(),
                        "solver_stage": "SYNTHESIZE",
                        "review_feedback": None,
                        "workpad": workpad,
                    }
            else:
                structured_for_compute = dict(curated.structured_evidence or {})
                if officeqa_mode and llm_control_state.get("evidence_commit_calls", 0) < llm_control_budget.get("evidence_commit_calls", 0):
                    evidence_commit_candidate_sources = list(dict(workpad.get("retrieval_diagnostics") or {}).get("candidate_sources") or [])
                    evidence_commit_needed, evidence_commit_reason = should_use_evidence_commit_llm(
                        retrieval_intent=active_retrieval_intent,
                        structured_evidence=structured_for_compute,
                        candidate_sources=evidence_commit_candidate_sources,
                        evidence_review=dict(workpad.get("officeqa_evidence_review") or {}),
                        repair_history=list(workpad.get("officeqa_llm_repair_history", []) or []),
                    )
                    if evidence_commit_needed:
                        evidence_commit_decision = maybe_review_evidence_commitment(
                            task_text=task_text,
                            retrieval_intent=active_retrieval_intent,
                            structured_evidence=structured_for_compute,
                            candidate_sources=evidence_commit_candidate_sources,
                            evidence_review=dict(workpad.get("officeqa_evidence_review") or {}),
                            reason=evidence_commit_reason,
                        )
                        if evidence_commit_decision is not None:
                            decision_payload = evidence_commit_decision.model_dump()
                            current_retrieval_signature = _officeqa_retrieval_input_signature(active_retrieval_intent, workpad)
                            active_retrieval_intent, workpad, path_changed, reroute_action = _apply_officeqa_llm_repair_decision(
                                active_retrieval_intent,
                                workpad,
                                decision_payload,
                            )
                            reroute_action = _officeqa_reroute_action("evidence_commit_review", evidence_commit_reason, decision_payload)
                            post_retrieval_signature = _officeqa_retrieval_input_signature(active_retrieval_intent, workpad)
                            llm_control_state["evidence_commit_calls"] = int(llm_control_state.get("evidence_commit_calls", 0) or 0) + 1
                            workpad["officeqa_llm_control_state"] = llm_control_state
                            workpad["officeqa_evidence_commit_review"] = {
                                "reason": evidence_commit_reason,
                                "decision": decision_payload,
                                "path_changed": path_changed,
                                "reroute_action": reroute_action,
                            }
                            workpad = _record_llm_repair_decision(
                                workpad,
                                stage="evidence_commit_review",
                                trigger=evidence_commit_reason,
                                decision=decision_payload,
                                path_changed=path_changed,
                                reroute_action=reroute_action,
                                pre_retrieval_signature=current_retrieval_signature,
                                post_retrieval_signature=post_retrieval_signature,
                            )
                            workpad = record_officeqa_llm_usage(
                                workpad,
                                category="evidence_commit_llm",
                                used=True,
                                reason=evidence_commit_reason,
                                model_name=str(decision_payload.get("model_name", "") or ""),
                                confidence=float(decision_payload.get("confidence", 0.0) or 0.0),
                                applied=path_changed,
                                details={
                                    "decision": decision_payload.get("decision", ""),
                                    "restart_scope": decision_payload.get("restart_scope", ""),
                                    "publication_scope_action": decision_payload.get("publication_scope_action", ""),
                                },
                            )
                            if path_changed:
                                workpad, journal, curated = _invalidate_officeqa_repair_state(
                                    workpad=workpad,
                                    journal=journal,
                                    curated=curated,
                                    stage="evidence_commit_review",
                                    trigger=evidence_commit_reason,
                                    decision=decision_payload,
                                    reroute_action=reroute_action,
                                    pre_retrieval_signature=current_retrieval_signature,
                                    post_retrieval_signature=post_retrieval_signature,
                                )
                                tool_plan.pending_tools = []
                                workpad["solver_llm_decision"] = {
                                    "used_llm": True,
                                    "reason": "evidence_commit_review_redirected_retrieval",
                                }
                                workpad = _record_event(
                                    workpad,
                                    "executor",
                                    f"evidence commit review -> {decision_payload.get('decision', 'keep')}",
                                )
                                evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                                return {
                                    "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
                                    "task_intent": intent.model_dump(),
                                    "retrieval_intent": active_retrieval_intent.model_dump(),
                                    "tool_plan": tool_plan.model_dump(),
                                    "execution_journal": journal.model_dump(),
                                    "curated_context": curated.model_dump(),
                                    "evidence_sufficiency": evidence_sufficiency.model_dump(),
                                    "solver_stage": "GATHER",
                                    "workpad": workpad,
                                }
                if officeqa_mode and targeted_compute_retry and not bool(workpad.get("officeqa_compute_reselection_attempted")):
                    reselected_structured, reselection_diagnostics = _reselect_officeqa_structured_evidence(
                        structured_for_compute,
                        active_retrieval_intent,
                    )
                    workpad["officeqa_compute_reselection_attempted"] = True
                    if reselected_structured is not None:
                        structured_for_compute = reselected_structured
                        workpad["officeqa_compute_reselection"] = reselection_diagnostics
                        provenance_summary = dict(curated.provenance_summary or {})
                        provenance_summary["compute_reselection"] = reselection_diagnostics
                        curated.provenance_summary = provenance_summary
                compute_result = benchmark_compute_result(
                    task_text,
                    active_retrieval_intent,
                    structured_for_compute,
                    benchmark_overrides,
                )
                if compute_result is None:
                    raise RuntimeError("Benchmark compute hook is unavailable for the active benchmark adapter.")
                curated = attach_compute_result(curated, compute_result.model_dump(), benchmark_overrides)
                workpad["officeqa_compute"] = compute_result.model_dump()
                provenance_summary = dict(curated.provenance_summary or {})
                provenance_summary["answer_strategy"] = {
                    "answer_mode": active_retrieval_intent.answer_mode,
                    "compute_policy": active_retrieval_intent.compute_policy,
                    "partial_answer_allowed": active_retrieval_intent.partial_answer_allowed,
                    "analysis_modes": list(active_retrieval_intent.analysis_modes[:6]),
                    "compute_status": compute_result.status,
                }
                curated.provenance_summary = provenance_summary
                workpad["officeqa_answer_strategy"] = provenance_summary["answer_strategy"]
                if compute_result.status == "ok" and compute_result.answer_text and active_retrieval_intent.answer_mode == "deterministic_compute":
                    answer = compute_result.answer_text
                    evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                    journal.final_artifact_signature = _artifact_signature(answer)
                    workpad["completion_budget"] = 0
                    workpad["review_ready"] = True
                    workpad["solver_llm_decision"] = {
                        "used_llm": False,
                        "reason": "deterministic_compute_completed",
                    }
                    workpad = _record_event(workpad, "executor", f"deterministic officeqa compute -> {compute_result.operation or 'answer'}")
                    if tracer:
                        tracer.record(
                            "executor",
                            {
                                "intent": intent.model_dump(),
                                "used_llm": False,
                                "llm_decision_reason": "deterministic_compute_completed",
                                "tools_ran": tools_ran_this_call,
                                "tool_results": _compact_tool_findings(journal.tool_results),
                                "output_preview": answer[:2000],
                                "completion_budget": 0,
                                "officeqa_compute": compute_result.model_dump(),
                                "aggregation_reason": compute_result.selection_reasoning,
                                "rejected_aggregation_alternatives": compute_result.rejected_alternatives,
                                "compute_reselection": dict(workpad.get("officeqa_compute_reselection") or {}),
                                "answer_mode": active_retrieval_intent.answer_mode,
                                "compute_policy": active_retrieval_intent.compute_policy,
                                "evidence_review": dict(workpad.get("officeqa_evidence_review") or {}),
                                "officeqa_llm_usage": _officeqa_trace_llm_usage(workpad),
                                "llm_repair_history": list(workpad.get("officeqa_llm_repair_history", []) or []),
                            },
                        )
                    return {
                        "messages": [AIMessage(content=answer)],
                        "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
                        "task_intent": intent.model_dump(),
                        "retrieval_intent": active_retrieval_intent.model_dump(),
                        "tool_plan": tool_plan.model_dump(),
                        "execution_journal": journal.model_dump(),
                        "curated_context": curated.model_dump(),
                        "review_packet": build_review_packet(
                            task_text=task_text,
                            answer_text=answer,
                            answer_contract=state.get("answer_contract", {}) or {},
                            curated_context=curated.model_dump(),
                            tool_results=journal.tool_results,
                            evidence_sufficiency=evidence_sufficiency.model_dump(),
                        ).model_dump(),
                        "evidence_sufficiency": evidence_sufficiency.model_dump(),
                        "solver_stage": "SYNTHESIZE",
                        "review_feedback": None,
                        "workpad": workpad,
                    }
                if compute_result.status == "ok" and active_retrieval_intent.answer_mode == "hybrid_grounded":
                    workpad = _record_event(
                        workpad,
                        "executor",
                        f"hybrid grounded synthesis using deterministic core -> {compute_result.operation or 'answer'}",
                    )
                elif compute_result.status != "ok" and active_retrieval_intent.compute_policy == "required":
                    answer = _officeqa_required_compute_insufficiency_answer(list(compute_result.validation_errors))
                    journal.final_artifact_signature = _artifact_signature(answer)
                    workpad["completion_budget"] = 0
                    workpad["review_ready"] = True
                    workpad["solver_llm_decision"] = {
                        "used_llm": False,
                        "reason": "required_compute_unavailable",
                    }
                    workpad = _record_event(
                        workpad,
                        "executor",
                        "required compute unavailable -> bounded insufficiency answer",
                    )
                    return {
                        "messages": [AIMessage(content=answer)],
                        "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
                        "task_intent": intent.model_dump(),
                        "retrieval_intent": active_retrieval_intent.model_dump(),
                        "tool_plan": tool_plan.model_dump(),
                        "execution_journal": journal.model_dump(),
                        "curated_context": curated.model_dump(),
                        "review_packet": build_review_packet(
                            task_text=task_text,
                            answer_text=answer,
                            answer_contract=state.get("answer_contract", {}) or {},
                            curated_context=curated.model_dump(),
                            tool_results=journal.tool_results,
                            evidence_sufficiency=evidence_sufficiency.model_dump(),
                        ).model_dump(),
                        "evidence_sufficiency": evidence_sufficiency.model_dump(),
                        "solver_stage": "SYNTHESIZE",
                        "review_feedback": None,
                        "workpad": workpad,
                    }
                elif compute_result.status != "ok" and active_retrieval_intent.compute_policy in {"preferred", "not_applicable"}:
                    workpad = _record_event(
                        workpad,
                        "executor",
                        f"grounded synthesis fallback -> compute {compute_result.status}",
                    )

        if intent.task_family == "finance_options" and journal.tool_results:
            answer = _options_final_answer({**state, "execution_journal": journal.model_dump()})
            if answer:
                evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                journal.final_artifact_signature = _artifact_signature(answer)
                workpad["completion_budget"] = 0
                workpad["review_ready"] = True
                workpad = _record_event(workpad, "executor", "deterministic options final ready")
                if tracer:
                    tracer.record(
                        "executor",
                        {
                            "intent": intent.model_dump(),
                            "used_llm": False,
                            "tools_ran": tools_ran_this_call,
                            "tool_results": _compact_tool_findings(journal.tool_results),
                            "output_preview": answer[:2000],
                            "completion_budget": 0,
                        },
                    )
                return {
                    "messages": [AIMessage(content=answer)],
                    "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
                    "task_intent": intent.model_dump(),
                    "retrieval_intent": active_retrieval_intent.model_dump(),
                    "tool_plan": tool_plan.model_dump(),
                    "execution_journal": journal.model_dump(),
                    "curated_context": curated.model_dump(),
                    "review_packet": build_review_packet(
                        task_text=task_text,
                        answer_text=answer,
                        answer_contract=state.get("answer_contract", {}) or {},
                        curated_context=curated.model_dump(),
                        tool_results=journal.tool_results,
                        evidence_sufficiency=evidence_sufficiency.model_dump(),
                    ).model_dump(),
                    "evidence_sufficiency": evidence_sufficiency.model_dump(),
                    "solver_stage": "SYNTHESIZE",
                    "review_feedback": None,
                    "workpad": workpad,
                }

        is_revision = bool(review_feedback)
        prompt_messages = [
            SystemMessage(content=EXECUTOR_SYSTEM),
        ]
        guidance = execution_guidance(
            intent.task_family,
            intent.execution_mode,
            benchmark_overrides=benchmark_overrides,
            task_text=task_text,
            answer_mode=active_retrieval_intent.answer_mode,
            compute_policy=active_retrieval_intent.compute_policy,
            partial_answer_allowed=active_retrieval_intent.partial_answer_allowed,
            analysis_modes=active_retrieval_intent.analysis_modes,
            compute_status=str(dict(curated.compute_result or {}).get("status", "") or ""),
        )
        if guidance:
            prompt_messages.append(SystemMessage(content=guidance))
        formatting_guidance = contract_guidance(state.get("answer_contract", {}) or {})
        if formatting_guidance:
            prompt_messages.append(SystemMessage(content=formatting_guidance))
        if is_revision:
            revision_text = build_revision_prompt(
                missing_dimensions=review_feedback.get("missing_dimensions", []),
                improve_hint=review_feedback.get("improve_hint", ""),
                reviewer_reasoning=review_feedback.get("reasoning", ""),
                task_family=intent.task_family,
                benchmark_overrides=benchmark_overrides,
                task_text=task_text,
            )
            prompt_messages.append(SystemMessage(content=revision_text))
            journal.revision_count += 1
        # On revision, skip tool findings for non-live-data tools (already in prior answer)
        skip_tools_on_revision = (
            is_revision
            and intent.execution_mode not in _RETRIEVAL_EXECUTION_MODES
            and all((r.get("type") or r.get("tool_name", "")) in REUSABLE_TOOL_FAMILIES for r in journal.tool_results)
        ) if journal.tool_results else False
        prompt_messages.append(
            SystemMessage(
                content=solver_context_block(
                    curated.model_dump(),
                    journal.tool_results,
                    include_objective=False,
                    revision_mode=skip_tools_on_revision,
                )
            )
        )
        prompt_messages.append(HumanMessage(content=task_text))

        prompt_tokens = count_tokens(prompt_messages)
        if budget:
            budget.record_context_tokens(prompt_tokens)
        max_tokens = _adaptive_completion_budget(
            intent.completion_mode,
            prompt_tokens,
            task_family=intent.task_family,
            review_feedback=bool(review_feedback),
            task_text=task_text,
        )
        model_name = get_model_name_for_task(
            "solver",
            execution_mode=intent.execution_mode,
            task_family=intent.task_family,
            prompt_tokens=prompt_tokens,
            answer_mode=active_retrieval_intent.answer_mode,
            analysis_modes=active_retrieval_intent.analysis_modes,
        )
        model = ChatOpenAI(
            model=model_name,
            **get_client_kwargs("solver"),
            **get_model_runtime_kwargs(
                "solver",
                execution_mode=intent.execution_mode,
                task_family=intent.task_family,
                prompt_tokens=prompt_tokens,
                answer_mode=active_retrieval_intent.answer_mode,
                analysis_modes=active_retrieval_intent.analysis_modes,
            ),
            temperature=0,
            max_tokens=max_tokens,
        )
        t0 = time.monotonic()
        response = model.invoke(prompt_messages)
        latency = (time.monotonic() - t0) * 1000
        content = str(getattr(response, "content", "") or "").strip()
        if tracker:
            tracker.record(
                operator="executor",
                model_name=model_name,
                tokens_in=prompt_tokens,
                tokens_out=count_tokens([AIMessage(content=content)]),
                latency_ms=latency,
                success=bool(content),
            )

        journal.final_artifact_signature = _artifact_signature(content)
        evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
        workpad["completion_budget"] = max_tokens
        workpad["review_ready"] = True
        workpad["solver_llm_decision"] = {
            "used_llm": True,
            "reason": "grounded_synthesis_required",
        }
        if officeqa_mode:
            llm_control_state["final_synthesis_calls"] = int(llm_control_state.get("final_synthesis_calls", 0) or 0) + 1
            workpad["officeqa_llm_control_state"] = llm_control_state
            workpad = record_officeqa_llm_usage(
                workpad,
                category="final_synthesis_llm",
                used=True,
                reason="grounded_synthesis_required",
                model_name=model_name,
                confidence=1.0,
                applied=True,
                details={"answer_mode": active_retrieval_intent.answer_mode, "compute_policy": active_retrieval_intent.compute_policy},
            )
        workpad = _record_event(workpad, "executor", "final draft ready")
        if tracer:
            tracer.record(
                "executor",
                {
                    "intent": intent.model_dump(),
                    "used_llm": True,
                    "llm_decision_reason": "grounded_synthesis_required",
                    "tools_ran": tools_ran_this_call,
                    "tool_results": _compact_tool_findings(journal.tool_results),
                    "prompt": format_messages_for_trace(prompt_messages),
                    "output_preview": content[:2000],
                    "completion_budget": max_tokens,
                    "answer_mode": active_retrieval_intent.answer_mode,
                    "compute_policy": active_retrieval_intent.compute_policy,
                    "tokens": {
                        "prompt": prompt_tokens,
                        "completion": count_tokens([AIMessage(content=content)]),
                    },
                    "officeqa_llm_usage": _officeqa_trace_llm_usage(workpad),
                    "llm_repair_history": list(workpad.get("officeqa_llm_repair_history", []) or []),
                },
            )
        logger.info("[Step %s] v4 executor -> final draft", step)
        return {
            "messages": [AIMessage(content=content)],
            "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
            "task_intent": intent.model_dump(),
            "retrieval_intent": active_retrieval_intent.model_dump(),
            "tool_plan": tool_plan.model_dump(),
            "execution_journal": journal.model_dump(),
            "curated_context": curated.model_dump(),
            "review_packet": build_review_packet(
                task_text=task_text,
                answer_text=content,
                answer_contract=state.get("answer_contract", {}) or {},
                curated_context=curated.model_dump(),
                tool_results=journal.tool_results,
                evidence_sufficiency=evidence_sufficiency.model_dump(),
            ).model_dump(),
            "evidence_sufficiency": evidence_sufficiency.model_dump(),
            "solver_stage": "SYNTHESIZE",
            "review_feedback": None,
            "workpad": workpad,
        }

    return executor


def route_from_executor(state: RuntimeState) -> str:
    workpad = state.get("workpad", {}) or {}
    if workpad.get("review_ready"):
        return "reviewer"
    if state.get("solver_stage") == "COMPLETE":
        if state.get("answer_contract", {}).get("requires_adapter"):
            return "output_adapter"
        return "reflect"
    return "executor"


def _targeted_fix_prompt(missing_dimensions: list[str]) -> str:
    normalized = [str(item).lower() for item in missing_dimensions]
    additions: list[str] = []
    if any("source" in item or "citation" in item or "provenance" in item for item in normalized):
        additions.append("the exact supporting Treasury citation, quote, or extracted row backing the answer")
    if any("period" in item or "calendar year" in item or "fiscal year" in item or "monthly" in item for item in normalized):
        additions.append("period alignment, including monthly versus annual and calendar-year versus fiscal-year scope")
    if any("aggregation" in item or "sum" in item or "difference" in item or "percent change" in item for item in normalized):
        additions.append("the correct aggregation path before finalizing the numeric result")
    if any("unit" in item for item in normalized):
        additions.append("unit normalization and magnitude alignment across all extracted values")
    if any("inflation" in item or "cpi" in item for item in normalized):
        additions.append("inflation support and CPI alignment for the requested comparison")
    if any("statistic" in item or "correlation" in item or "regression" in item or "standard deviation" in item for item in normalized):
        additions.append("the complete extracted series required for the requested statistical analysis")
    if any("forecast" in item or "projection" in item or "time series" in item for item in normalized):
        additions.append("the ordered time series evidence needed for the requested forecasting step")
    if any("risk" in item or "var" in item or "weighted average" in item for item in normalized):
        additions.append("the exact inputs and weighting assumptions needed for the requested financial metric")
    if not additions:
        additions.append("the missing evidence alignment before returning the final benchmark answer")
    return "Repair the answer by re-checking " + ", ".join(additions[:3]) + "."


def reviewer(state: RuntimeState) -> dict[str, Any]:
    step = increment_runtime_step()
    intent = TaskIntent.model_validate(state.get("task_intent") or {})
    journal = ExecutionJournal.model_validate(state.get("execution_journal") or {})
    curated = CuratedContext.model_validate(state.get("curated_context") or {})
    tool_plan = ToolPlan.model_validate(state.get("tool_plan") or {})
    benchmark_overrides = dict(state.get("benchmark_overrides") or {})
    retrieval_intent = RetrievalIntent.model_validate(state.get("retrieval_intent") or {})
    evidence_sufficiency = assess_evidence_sufficiency(
        latest_human_text(state["messages"]),
        SourceBundle.model_validate(state.get("source_bundle") or {}),
        journal.tool_results,
        benchmark_overrides,
    )
    workpad = dict(state.get("workpad", {}))
    answer = _latest_public_answer(state)
    answer_contract = state.get("answer_contract", {}) or {}
    validator_result: dict[str, Any] = {}
    officeqa_validation = None
    compute_status = str(dict(curated.compute_result or {}).get("status", "") or "")
    deterministic_structured_answer = (
        benchmark_overrides.get("benchmark_adapter") == "officeqa"
        and retrieval_intent.answer_mode == "deterministic_compute"
        and compute_status == "ok"
    )
    if benchmark_overrides.get("benchmark_adapter") == "officeqa" and intent.review_mode == "document_grounded":
        officeqa_validation = benchmark_validate_final(
            task_text=latest_human_text(state["messages"]),
            retrieval_intent=retrieval_intent,
            curated_context=curated,
            evidence_sufficiency=evidence_sufficiency,
            citations=[
                str(item)
                for item in (
                    list(curated.provenance_summary.get("source_bundle", {}).get("urls", []))
                    + list({citation for result in journal.tool_results for citation in _tool_result_citations(result)})
                )
                if str(item).strip()
            ],
            benchmark_overrides=benchmark_overrides,
        )
        if officeqa_validation.verdict == "revise":
            retry_allowed, retry_stop_reason = _officeqa_retry_policy(
                officeqa_validation,
                journal=journal,
                tool_plan=tool_plan,
                retrieval_intent=retrieval_intent,
            )
            officeqa_validation.retry_allowed = retry_allowed
            officeqa_validation.retry_stop_reason = retry_stop_reason
            workpad["officeqa_retry_policy"] = {
                "recommended_repair_target": officeqa_validation.recommended_repair_target,
                "orchestration_strategy": officeqa_validation.orchestration_strategy,
                "remediation_codes": list(officeqa_validation.remediation_codes),
                "retry_allowed": retry_allowed,
                "retry_stop_reason": retry_stop_reason,
            }
            if not retry_allowed and officeqa_validation.insufficiency_answer:
                answer = officeqa_validation.insufficiency_answer
        validator_result = officeqa_validation.model_dump()
        if officeqa_validation.replace_answer and officeqa_validation.insufficiency_answer:
            answer = officeqa_validation.insufficiency_answer
    review_packet = build_review_packet(
        task_text=latest_human_text(state["messages"]),
        answer_text=answer,
        answer_contract=answer_contract,
        curated_context=curated.model_dump(),
        tool_results=journal.tool_results,
        evidence_sufficiency=evidence_sufficiency.model_dump(),
        validator_result=validator_result,
    )

    missing: list[str] = []
    verdict = "pass"
    reasoning = "Reviewer accepted the final artifact."
    score = 0.9
    stop_reason = ""

    if officeqa_validation and officeqa_validation.verdict == "revise":
        missing.extend(list(officeqa_validation.missing_dimensions))
        reasoning = officeqa_validation.reasoning
        score = min(score, 0.64)

    if looks_truncated(answer):
        verdict = "revise" if journal.revision_count < 1 else "fail"
        reasoning = "Final answer appears truncated."
        missing = ["complete final answer"]
        score = 0.3 if verdict == "revise" else 0.24
    elif officeqa_validation and officeqa_validation.verdict == "fail":
        verdict = "fail"
        reasoning = officeqa_validation.reasoning
        missing = list(officeqa_validation.missing_dimensions)
        score = 0.34
        stop_reason = officeqa_validation.stop_reason
    elif intent.review_mode == "exact_quant":
        if not matches_exact_json_contract(answer, answer_contract):
            journal.contract_collapse_attempts += 1
            if answer_contract.get("requires_adapter"):
                verdict = "fail"
                reasoning = "Exact quantitative answer should collapse through the output adapter instead of another prose revision."
                missing = ["exact contract-compliant scalar output"]
                stop_reason = "exact_output_collapse"
                score = 0.52
            else:
                verdict = "revise"
                reasoning = "Exact quantitative answer does not satisfy the output contract."
                missing = ["exact contract-compliant scalar output"]
                score = 0.4
    elif intent.review_mode == "analytical_reasoning":
        if looks_truncated(answer):
            verdict = "revise" if journal.revision_count < 1 else "fail"
            reasoning = "Analytical derivation appears truncated or incomplete."
            missing = ["complete derivation and final result"]
            score = 0.46 if verdict == "revise" else 0.38
    elif intent.review_mode == "document_grounded":
        bounded_partial = retrieval_intent.partial_answer_allowed and _is_bounded_partial_answer(answer)
        if not journal.tool_results:
            missing.append("retrieved evidence")
        if not review_packet.citations:
            missing.append("source citations")
        if not evidence_sufficiency.is_sufficient:
            missing.extend(evidence_sufficiency.missing_dimensions)
        answer_lower = (answer or "").lower()
        has_inline_attribution = any(token in answer_lower for token in ("source:", "citation", "[source", "(source"))
        if any(
            token in answer_lower
            for token in (
                "insufficient data",
                "insufficient evidence",
                "cannot determine",
                "cannot calculate",
                "cannot compute",
                "not present in the provided evidence",
                "not available in the provided evidence",
            )
        ):
            if not bounded_partial:
                missing.append("exact grounded answer instead of an insufficiency placeholder")
        if any(
            token in answer_lower
            for token in (
                "i must rely on",
                "i recall",
                "should be",
                "highly promising",
                "would be the",
            )
        ):
            missing.append("unsupported inference from indirect evidence")
        if review_packet.citations and not has_inline_attribution and not deterministic_structured_answer:
            missing.append("inline source attribution")
        if review_packet.tool_findings and not has_inline_attribution and "\"" not in answer and "quote" not in answer_lower and not deterministic_structured_answer:
            missing.append("quoted supporting evidence or extracted table row")
        task_lower = latest_human_text(state["messages"]).lower()
        if "calendar year" in task_lower and "fiscal year" in answer_lower and "calendar year" not in answer_lower:
            missing.append("calendar year vs fiscal year alignment")
        if ("all individual calendar months" in task_lower or "total sum of these values" in task_lower) and any(
            token in answer_lower for token in ("annual total", "fiscal year total", "calendar year total")
        ):
            missing.append("monthly sum vs annual total alignment")
        if "should include" in task_lower or "shouldn't contain" in task_lower:
            if "include" not in answer_lower and "exclude" not in answer_lower and "excluding" not in answer_lower:
                missing.append("inclusion and exclusion qualifiers")
        if re.search(r"\d", answer or "") and "numeric or quoted support" in evidence_sufficiency.missing_dimensions:
            missing.append("exact numeric support from retrieved evidence")
        if bounded_partial:
            missing = _partial_answer_missing_dimensions(missing)
            if not missing:
                reasoning = "Reviewer accepted a bounded grounded partial answer because the supported portion is explicit and source-backed."
        elif deterministic_structured_answer and not missing:
            reasoning = "Reviewer accepted a deterministic structured answer backed by compute provenance and validator-approved evidence."
        if missing:
            verdict = "revise" if journal.revision_count < 1 else "fail"
            reasoning = "Grounded retrieval answer is missing evidence quality, attribution, or semantic alignment with the requested source, period, entity, or aggregation."
            score = 0.58 if verdict == "revise" else 0.48
    progress = _build_progress_signature(
        execution_mode=intent.execution_mode,
        selected_tools=tool_plan.selected_tools,
        missing_dimensions=missing,
        artifact_signature=journal.final_artifact_signature or _artifact_signature(answer),
        contract_status=_contract_status(answer, answer_contract, intent.review_mode),
    )
    if officeqa_validation and officeqa_validation.verdict == "revise" and not officeqa_validation.retry_allowed:
        verdict = "fail"
        reasoning = (
            "OfficeQA validator found a repairable gap, but the runtime has no remaining targeted repair path for this run."
        )
        stop_reason = officeqa_validation.retry_stop_reason or "officeqa_no_repair_path"
        score = min(score, 0.4)
    if journal.progress_signatures and journal.progress_signatures[-1].get("signature") == progress.signature and verdict == "revise":
        verdict = "fail"
        reasoning = "Progress stalled: the answer repeated the same unresolved gap without a materially different artifact."
        stop_reason = "progress_stalled"
        score = min(score, 0.45)
    if verdict == "fail" and officeqa_validation and officeqa_validation.insufficiency_answer:
        answer = officeqa_validation.insufficiency_answer
    journal.progress_signatures.append(progress.model_dump())

    report = QualityReport(
        verdict=verdict,  # type: ignore[arg-type]
        reasoning=reasoning,
        missing_dimensions=missing,
        targeted_fix_prompt=_targeted_fix_prompt(missing) if missing else "",
        score=score,
        stop_reason=stop_reason,
    )

    workpad["review_ready"] = False
    workpad["review_mode"] = intent.review_mode
    workpad = _record_event(workpad, "reviewer", f"{verdict.upper()}: {reasoning}")
    tracer = get_tracer()
    if tracer:
        tracer.record(
            "reviewer",
            {
                "review_mode": intent.review_mode,
                "task_family": intent.task_family,
                "verdict": verdict,
                "reasoning": reasoning,
                "missing_dimensions": missing,
                "stop_reason": stop_reason,
                "officeqa_validator": validator_result,
                "validator_remediation": dict(validator_result or {}).get("remediation_guidance", []),
                "validator_codes": dict(validator_result or {}).get("remediation_codes", []),
                "orchestration_strategy": dict(validator_result or {}).get("orchestration_strategy", ""),
                "retry_allowed": dict(validator_result or {}).get("retry_allowed", False),
                "retry_stop_reason": dict(validator_result or {}).get("retry_stop_reason", ""),
                "answer_mode": retrieval_intent.answer_mode,
                "compute_policy": retrieval_intent.compute_policy,
                "used_llm": False,
                "llm_decision_reason": "rule_based_validator_review",
            },
        )
    logger.info("[Step %s] engine reviewer -> %s", step, verdict.upper())

    if verdict == "pass":
        return {
            "quality_report": report.model_dump(),
            "execution_journal": journal.model_dump(),
            "review_packet": review_packet.model_dump(),
            "evidence_sufficiency": evidence_sufficiency.model_dump(),
            "progress_signature": progress.model_dump(),
            "review_feedback": None,
            "solver_stage": "COMPLETE",
            "workpad": workpad,
        }

    if verdict == "revise":
        return {
            "quality_report": report.model_dump(),
            "execution_journal": journal.model_dump(),
            "review_packet": review_packet.model_dump(),
            "evidence_sufficiency": evidence_sufficiency.model_dump(),
            "progress_signature": progress.model_dump(),
            "review_feedback": {
                "verdict": "revise",
                "reasoning": report.targeted_fix_prompt or reasoning,
                "missing_dimensions": missing,
                "improve_hint": report.targeted_fix_prompt,
                "repair_target": (
                    officeqa_validation.recommended_repair_target
                    if officeqa_validation and officeqa_validation.verdict == "revise"
                    else "final"
                ),
                "repair_class": (
                    "missing_evidence"
                    if officeqa_validation and officeqa_validation.recommended_repair_target == "gather"
                    else "scalar_only"
                    if officeqa_validation and officeqa_validation.recommended_repair_target == "compute"
                    else "missing_section"
                ),
                "orchestration_strategy": (
                    officeqa_validation.orchestration_strategy
                    if officeqa_validation and officeqa_validation.verdict == "revise"
                    else ""
                ),
                "remediation_codes": (
                    list(officeqa_validation.remediation_codes)
                    if officeqa_validation and officeqa_validation.verdict == "revise"
                    else []
                ),
                "retry_allowed": (
                    officeqa_validation.retry_allowed
                    if officeqa_validation and officeqa_validation.verdict == "revise"
                    else True
                ),
                "retry_stop_reason": (
                    officeqa_validation.retry_stop_reason
                    if officeqa_validation and officeqa_validation.verdict == "revise"
                    else ""
                ),
            },
            "solver_stage": "REVISE",
            "workpad": workpad,
        }

    workpad = _record_event(workpad, "reviewer", "quality ceiling reached -> stop without another loop")
    result = {
        "quality_report": report.model_dump(),
        "execution_journal": journal.model_dump(),
        "review_packet": review_packet.model_dump(),
        "evidence_sufficiency": evidence_sufficiency.model_dump(),
        "progress_signature": progress.model_dump(),
        "review_feedback": None,
        "solver_stage": "COMPLETE",
        "workpad": workpad,
    }
    if officeqa_validation and (
        officeqa_validation.replace_answer
        or (officeqa_validation.verdict == "revise" and not officeqa_validation.retry_allowed)
        or (verdict == "fail" and officeqa_validation.insufficiency_answer)
    ):
        result["messages"] = [AIMessage(content=answer)]
    return result


def route_from_reviewer(state: RuntimeState) -> str:
    if state.get("solver_stage") == "REVISE":
        return "executor"
    if state.get("solver_stage") == "COMPLETE":
        intent = TaskIntent.model_validate(state.get("task_intent") or {})
        report = QualityReport.model_validate(state.get("quality_report") or {})
        if report.stop_reason == "exact_output_collapse" and state.get("answer_contract", {}).get("requires_adapter"):
            return "output_adapter"
        if report.stop_reason.startswith("officeqa_"):
            if state.get("answer_contract", {}).get("requires_adapter"):
                return "output_adapter"
            return "reflect"
        if report.verdict == "fail":
            journal = ExecutionJournal.model_validate(state.get("execution_journal") or {})
            if intent.complexity_tier == "complex_qualitative" and journal.self_reflection_count < 1:
                return "self_reflection"
            if state.get("answer_contract", {}).get("requires_adapter"):
                return "output_adapter"
            return "reflect"
        if intent.complexity_tier == "complex_qualitative":
            return "self_reflection"
        if state.get("answer_contract", {}).get("requires_adapter"):
            return "output_adapter"
        return "reflect"
    return "executor"


def _llm_self_reflection(task_text: str, answer: str, tool_count: int) -> dict[str, Any]:
    """LLM-based self-reflection — cheap model, 3-question rubric.

    Returns {"score": float, "complete": bool, "missing": list, "improve_prompt": str}.
    Falls through gracefully on any error.
    """
    task_snippet = task_text[:500]
    answer_snippet = answer[:800]
    messages = [
        SystemMessage(content=SELF_REFLECTION_SYSTEM),
        HumanMessage(
            content=(
                f"Tools used: {tool_count}\n"
                f"Task: {task_snippet}\n"
                f"Answer: {answer_snippet}"
            )
        ),
    ]
    try:
        model_name = get_model_name("profiler")
        model = ChatOpenAI(
            model=model_name,
            **get_client_kwargs("profiler"),
            **get_model_runtime_kwargs("profiler"),
            temperature=0,
            max_tokens=200,
        )
        response = model.invoke(messages)
        raw = str(getattr(response, "content", "") or "").strip()
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            return {
                "score": float(parsed.get("score", 0.7)),
                "complete": bool(parsed.get("complete", True)),
                "missing": list(parsed.get("missing", [])),
                "improve_prompt": str(parsed.get("improve_prompt", "")),
            }
    except Exception as exc:
        logger.info("LLM self-reflection fallback: %s", exc)
    return {"score": 0.8, "complete": True, "missing": [], "improve_prompt": ""}


def self_reflection(state: RuntimeState) -> dict[str, Any]:
    step = increment_runtime_step()
    journal = ExecutionJournal.model_validate(state.get("execution_journal") or {})
    report = QualityReport.model_validate(state.get("quality_report") or {})
    intent = TaskIntent.model_validate(state.get("task_intent") or {})
    workpad = dict(state.get("workpad", {}))
    answer = _latest_public_answer(state)
    task_text = latest_human_text(state["messages"])

    journal.self_reflection_count += 1

    if report.verdict == "fail" and intent.complexity_tier == "complex_qualitative":
        workpad = _record_event(workpad, "self_reflection", "REVISE after failed review ceiling")
        tracer = get_tracer()
        if tracer:
            tracer.record(
                "self_reflection",
                {
                    "score": report.score,
                    "complete": False,
                    "missing_dimensions": list(report.missing_dimensions),
                    "improve_prompt": report.targeted_fix_prompt,
                    "used_llm": False,
                    "salvage_path": True,
                },
            )
        return {
            "execution_journal": journal.model_dump(),
            "reflection_feedback": {
                "score": report.score,
                "complete": False,
                "missing_dimensions": list(report.missing_dimensions),
                "improve_prompt": report.targeted_fix_prompt,
            },
            "review_feedback": {
                "verdict": "revise",
                "reasoning": report.targeted_fix_prompt or report.reasoning or "Add the missing legal detail and return the final answer.",
                "missing_dimensions": list(report.missing_dimensions) or ["completeness"],
                "improve_hint": report.targeted_fix_prompt,
                "repair_target": "final",
                "repair_class": "missing_section",
            },
            "solver_stage": "REVISE",
            "workpad": workpad,
        }

    # Non-complex or already has missing-dimensions from reviewer → skip LLM reflection
    if report.missing_dimensions or intent.complexity_tier != "complex_qualitative":
        flagged_complete = not bool(report.missing_dimensions)
        workpad = _record_event(
            workpad,
            "self_reflection",
            "PASS (non-complex)" if flagged_complete else "SKIP (reviewer-flagged gaps remain)",
        )
        tracer = get_tracer()
        if tracer:
            tracer.record(
                "self_reflection",
                {
                    "score": report.score,
                    "complete": flagged_complete,
                    "missing_dimensions": [] if flagged_complete else list(report.missing_dimensions),
                    "used_llm": False,
                },
            )
        return {
            "execution_journal": journal.model_dump(),
            "reflection_feedback": {
                "score": report.score,
                "complete": flagged_complete,
                "missing_dimensions": [] if flagged_complete else list(report.missing_dimensions),
                "improve_prompt": "",
            },
            "solver_stage": "COMPLETE" if flagged_complete else "REVISE",
            "workpad": workpad,
        }

    # Heuristic pre-check — skip LLM if clearly good
    tool_count = len(journal.tool_results)
    h_score = heuristic_self_score(answer, tool_count=tool_count, task_family=intent.task_family)
    used_llm = False

    if h_score >= 0.85:
        # Clearly good — skip LLM call
        score = h_score
        missing: list[str] = []
        improve_prompt = ""
    elif journal.self_reflection_count > 1:
        # Already reflected once — accept whatever we have
        score = h_score
        missing = []
        improve_prompt = ""
    else:
        # Call LLM for detailed rubric evaluation
        llm_result = _llm_self_reflection(task_text, answer, tool_count)
        score = llm_result["score"]
        missing = llm_result["missing"]
        improve_prompt = llm_result["improve_prompt"]
        used_llm = True

    verdict = "PASS" if score >= 0.75 else "REVISE"
    workpad = _record_event(workpad, "self_reflection", f"{verdict} score={score:.2f} llm={used_llm}")
    tracer = get_tracer()
    if tracer:
        tracer.record(
            "self_reflection",
            {
                "score": score,
                "complete": verdict == "PASS",
                "missing_dimensions": missing,
                "improve_prompt": improve_prompt,
                "used_llm": used_llm,
                "heuristic_score": h_score,
            },
        )
    logger.info("[Step %s] v4 self_reflection -> %s (score=%.2f, llm=%s)", step, verdict, score, used_llm)

    if verdict == "PASS":
        return {
            "execution_journal": journal.model_dump(),
            "reflection_feedback": {"score": score, "complete": True, "missing_dimensions": [], "improve_prompt": ""},
            "solver_stage": "COMPLETE",
            "workpad": workpad,
        }

    return {
        "execution_journal": journal.model_dump(),
        "reflection_feedback": {"score": score, "complete": False, "missing_dimensions": missing, "improve_prompt": improve_prompt},
        "review_feedback": {
            "verdict": "revise",
            "reasoning": improve_prompt or "Answer is incomplete — add the missing detail.",
            "missing_dimensions": missing or ["completeness"],
            "improve_hint": improve_prompt,
            "repair_target": "final",
            "repair_class": "missing_section",
        },
        "solver_stage": "REVISE",
        "workpad": workpad,
    }


def route_from_self_reflection(state: RuntimeState) -> str:
    if state.get("solver_stage") == "REVISE":
        return "executor"
    if state.get("answer_contract", {}).get("requires_adapter"):
        return "output_adapter"
    return "reflect"
