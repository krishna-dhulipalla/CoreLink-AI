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
from agent.model_config import ChatOpenAI, get_client_kwargs, get_model_name, get_model_name_for_task, get_model_runtime_kwargs, invoke_structured_output
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import latest_human_text
from agent.solver.options import (
    deterministic_options_final_answer,
    deterministic_policy_options_final_answer,
)
from agent.solver.quant import deterministic_quant_final_answer
from agent.tracer import format_messages_for_trace, get_tracer
from agent.capabilities import resolve_tool_plan
from agent.benchmarks.officeqa_compute import compute_officeqa_result
from agent.benchmarks.officeqa_validator import validate_officeqa_final
from agent.benchmarks import benchmark_task_intent
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

        if tool_plan.pending_tools and not review_feedback:
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
                tool_plan.pending_tools = tool_plan.pending_tools[1:]
                evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                should_continue_after_tool = bool(tool_plan.pending_tools) or intent.execution_mode in _RETRIEVAL_EXECUTION_MODES
                if should_continue_after_tool and intent.execution_mode in {"advisory_analysis", "document_grounded_analysis", "tool_compute", "retrieval_augmented_analysis"}:
                    if tracer:
                        tracer.record(
                            "executor",
                            {
                                "intent": intent.model_dump(),
                                "used_llm": False,
                                "tools_ran": tools_ran_this_call,
                                "tool_results": _compact_tool_findings(journal.tool_results),
                                "output_preview": "",
                                "completion_budget": 0,
                            },
                        )
                    return {
                        "last_tool_result": tool_result.model_dump(),
                        "task_intent": intent.model_dump(),
                        "tool_plan": tool_plan.model_dump(),
                        "execution_journal": journal.model_dump(),
                        "curated_context": curated.model_dump(),
                        "evidence_sufficiency": evidence_sufficiency.model_dump(),
                        "solver_stage": "GATHER" if intent.execution_mode in _RETRIEVAL_EXECUTION_MODES else "COMPUTE",
                        "workpad": workpad,
                    }

        if intent.execution_mode in _RETRIEVAL_EXECUTION_MODES and not review_feedback and journal.retrieval_iterations < _MAX_RETRIEVAL_HOPS:
            retrieval_action = _plan_retrieval_action(
                execution_mode=intent.execution_mode,
                source_bundle=source_bundle,
                retrieval_intent=retrieval_intent,
                tool_plan=tool_plan,
                journal=journal,
                registry=registry,
                benchmark_overrides=benchmark_overrides,
            )
            if retrieval_action.action == "tool":
                if budget and budget.tool_calls_exhausted():
                    budget.log_budget_exit("tool_budget_exhausted", f"Blocked tool '{retrieval_action.tool_name}' after reaching tool-call cap.")
                else:
                    tool_args = _tool_args_from_retrieval_action(retrieval_action, source_bundle, registry, retrieval_intent)
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
                    workpad = _record_event(workpad, "executor", f"retrieval hop -> {retrieval_action.tool_name}")
                    evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                    if tracer:
                        tracer.record(
                            "executor",
                            {
                                "intent": intent.model_dump(),
                                "used_llm": False,
                                "tools_ran": tools_ran_this_call,
                                "tool_results": _compact_tool_findings(journal.tool_results),
                                "output_preview": "",
                                "completion_budget": 0,
                                "retrieval_action": retrieval_action.model_dump(),
                            },
                        )
                    return {
                        "last_tool_result": tool_result.model_dump(),
                        "task_intent": intent.model_dump(),
                        "tool_plan": tool_plan.model_dump(),
                        "execution_journal": journal.model_dump(),
                        "curated_context": curated.model_dump(),
                        "evidence_sufficiency": evidence_sufficiency.model_dump(),
                        "solver_stage": "GATHER",
                        "workpad": workpad,
                    }
            else:
                workpad = _record_event(workpad, "executor", f"retrieval ready to answer -> {retrieval_action.rationale or 'evidence collected'}")

        if benchmark_overrides.get("benchmark_adapter") == "officeqa" and journal.tool_results and curated.structured_evidence:
            predictive_gaps = predictive_evidence_gaps(retrieval_intent, curated.structured_evidence)
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
                    "strategy": retrieval_intent.strategy,
                }
                curated.provenance_summary = provenance_summary
                workpad["officeqa_predictive_gaps"] = predictive_gaps
                workpad = _record_event(
                    workpad,
                    "executor",
                    f"predictive evidence gaps -> {', '.join(predictive_gaps[:4])}",
                )
            else:
                compute_result = compute_officeqa_result(task_text, retrieval_intent, curated.structured_evidence)
                curated = attach_compute_result(curated, compute_result.model_dump())
                workpad["officeqa_compute"] = compute_result.model_dump()
                if compute_result.status == "ok" and compute_result.answer_text:
                    answer = compute_result.answer_text
                    evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, journal.tool_results, benchmark_overrides)
                    journal.final_artifact_signature = _artifact_signature(answer)
                    workpad["completion_budget"] = 0
                    workpad["review_ready"] = True
                    workpad = _record_event(workpad, "executor", f"deterministic officeqa compute -> {compute_result.operation or 'answer'}")
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
                                "officeqa_compute": compute_result.model_dump(),
                            },
                        )
                    return {
                        "messages": [AIMessage(content=answer)],
                        "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
                        "task_intent": intent.model_dump(),
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
        )
        model = ChatOpenAI(
            model=model_name,
            **get_client_kwargs("solver"),
            **get_model_runtime_kwargs(
                "solver",
                execution_mode=intent.execution_mode,
                task_family=intent.task_family,
                prompt_tokens=prompt_tokens,
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
        workpad = _record_event(workpad, "executor", "final draft ready")
        if tracer:
            tracer.record(
                "executor",
                {
                    "intent": intent.model_dump(),
                    "used_llm": True,
                    "tools_ran": tools_ran_this_call,
                    "tool_results": _compact_tool_findings(journal.tool_results),
                    "prompt": format_messages_for_trace(prompt_messages),
                    "output_preview": content[:2000],
                    "completion_budget": max_tokens,
                    "tokens": {
                        "prompt": prompt_tokens,
                        "completion": count_tokens([AIMessage(content=content)]),
                    },
                },
            )
        logger.info("[Step %s] v4 executor -> final draft", step)
        return {
            "messages": [AIMessage(content=content)],
            "last_tool_result": journal.tool_results[-1] if journal.tool_results else None,
            "task_intent": intent.model_dump(),
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
    if benchmark_overrides.get("benchmark_adapter") == "officeqa" and intent.review_mode == "document_grounded":
        officeqa_validation = validate_officeqa_final(
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
        )
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
        if review_packet.citations and not has_inline_attribution:
            missing.append("inline source attribution")
        if review_packet.tool_findings and not has_inline_attribution and "\"" not in answer and "quote" not in answer_lower:
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
    if journal.progress_signatures and journal.progress_signatures[-1].get("signature") == progress.signature and verdict == "revise":
        verdict = "fail"
        reasoning = "Progress stalled: the answer repeated the same unresolved gap without a materially different artifact."
        stop_reason = "progress_stalled"
        score = min(score, 0.45)
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
                "used_llm": False,
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
                "repair_target": "final",
                "repair_class": "missing_section",
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
    if officeqa_validation and officeqa_validation.replace_answer:
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
