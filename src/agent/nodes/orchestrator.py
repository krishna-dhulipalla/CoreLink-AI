"""Nodes for the active hybrid routing engine."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.contracts import (
    AnswerContract,
    CuratedContext,
    ExecutionJournal,
    ProgressSignature,
    QualityReport,
    RetrievalAction,
    ReviewPacket,
    SourceBundle,
    TaskIntent,
    ToolPlan,
    UnsupportedCapabilityReport,
)
from agent.context.evidence import _extract_policy_context
from agent.context.extraction import derive_market_snapshot
from agent.model_config import ChatOpenAI, get_client_kwargs, get_model_name, get_model_name_for_task, invoke_structured_output
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import detect_ambiguity_flags, detect_capability_flags, infer_task_profile, latest_human_text
from agent.solver.options import (
    deterministic_options_final_answer,
    deterministic_policy_options_final_answer,
    deterministic_policy_options_tool_call,
    deterministic_standard_options_tool_call,
    scenario_args_from_primary_tool,
)
from agent.solver.quant import deterministic_quant_final_answer
from agent.tools.normalization import normalize_tool_output
from agent.tracer import format_messages_for_trace, get_tracer
from agent.capabilities import resolve_tool_plan
from agent.curated_context import _compact_tool_findings, build_curated_context, build_review_packet, build_source_bundle, solver_context_block
from agent.prompts import (
    PLANNER_SYSTEM,
    EXECUTOR_SYSTEM,
    RETRIEVAL_PLANNER_SYSTEM,
    SELF_REFLECTION_SYSTEM,
    build_revision_prompt,
    execution_guidance,
    heuristic_self_score,
    REUSABLE_TOOL_FAMILIES,
)
from agent.review_utils import legal_depth_gaps, looks_truncated, matches_exact_json_contract, options_gaps
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)
RuntimeState = AgentState
_RETRIEVAL_EXECUTION_MODES = {"retrieval_augmented_analysis", "document_grounded_analysis"}
_MAX_RETRIEVAL_HOPS = max(2, int(os.getenv("MAX_RETRIEVAL_HOPS", "4")))

# Prompts are centralized in runtime_prompts.py — no inline prompt strings here.


def _legal_structure_option_count(answer: str) -> int:
    normalized = re.sub(r"\s+", " ", (answer or "").lower())
    patterns = {
        "asset_purchase": r"\basset purchase\b|\basset deal\b",
        "stock_purchase": r"\bstock purchase\b|\bshare purchase\b|\bequity purchase\b",
        "merger": r"\bmerger\b|\bforward triangular merger\b|\breverse triangular merger\b|\brtm\b",
        "carve_out": r"\bcarve[- ]out\b",
        "hybrid_structure": r"\bhybrid\b|\bcontingent consideration\b|\bearn[- ]?out\b",
        "joint_venture": r"\bjoint venture\b|\bpartnership\b",
        "minority_or_staged": r"\bminority investment\b|\bstaged acquisition\b|\boption to acquire\b",
        "commercial_structure": r"\blicen[sc]e\b|\bcommercial agreement\b|\bchannel partnership\b",
    }
    return sum(1 for pattern in patterns.values() if re.search(pattern, normalized))


def _opening_summary_window(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return text[:2200]
    opening = "\n\n".join(paragraphs[:3]).strip()
    return opening[:2200]


def _legal_frontload_gap(answer: str) -> bool:
    preview = _opening_summary_window(answer)
    return _legal_structure_option_count(preview) < 2


def _record_event(workpad: dict[str, Any], node: str, action: str) -> dict[str, Any]:
    updated = dict(workpad)
    events = list(updated.get("events", []))
    events.append({"node": node, "action": action})
    updated["events"] = events
    return updated


def _market_options_signal_count(task_text: str) -> int:
    normalized = (task_text or "").lower()
    signals = [
        any(token in normalized for token in ("iv percentile", "implied volatility", "historical volatility", "skew", "term structure")),
        any(token in normalized for token in ("delta", "gamma", "theta", "vega", "greeks")),
        any(token in normalized for token in ("straddle", "strangle", "iron condor", "credit spread", "vertical spread", "covered call")),
        any(token in normalized for token in ("expiry", "expiration", "days to expiry", "dte", "calls and puts")),
        bool(re.search(r"\b[A-Z]{1,5}\b", task_text or "")),
        any(token in normalized for token in ("premium", "breakeven", "vol seller", "vol buyer")),
    ]
    return sum(1 for signal in signals if signal)


def _looks_like_insurance_or_nonlisted_option(task_text: str) -> bool:
    normalized = (task_text or "").lower()
    return any(
        token in normalized
        for token in (
            "crop",
            "frost",
            "weather insurance",
            "insurance payout",
            "insurance contract",
            "farmer",
            "freeze",
        )
    )


def _supports_options_fast_path(task_text: str) -> bool:
    if _looks_like_insurance_or_nonlisted_option(task_text):
        return False
    return _market_options_signal_count(task_text) >= 2


def _looks_like_analytical_reasoning(task_text: str, capability_flags: list[str]) -> bool:
    normalized = (task_text or "").lower()
    if "needs_analytical_reasoning" in capability_flags:
        return True
    return any(
        token in normalized
        for token in (
            "differentiate",
            "derivative",
            "marginal cost",
            "marginal revenue",
            "integral",
            "optimize",
            "maximise",
            "maximize",
            "minimise",
            "minimize",
        )
    )


def _looks_like_market_scenario(task_text: str, capability_flags: list[str]) -> bool:
    normalized = (task_text or "").lower()
    if "needs_market_scenario" in capability_flags:
        return True
    return any(
        token in normalized
        for token in (
            "crypto",
            "flash crash",
            "liquidity crisis",
            "stress scenario",
            "scenario validation",
            "drawdown scenario",
        )
    )


def _looks_like_unsupported_artifact(task_text: str, capability_flags: list[str]) -> bool:
    normalized = (task_text or "").lower()
    if "needs_artifact_generation" in capability_flags:
        return True
    return any(
        token in normalized
        for token in (".wav", ".mp3", "audio file", "music producer", "render audio", "generate a track", "zip file")
    )


def _source_requests_live_data(task_text: str) -> bool:
    normalized = (task_text or "").lower()
    return any(
        token in normalized
        for token in ("latest", "today", "recent", "look up", "search", "source-backed", "current ", "as of ", "what was", "what is")
    )


def _looks_like_grounded_document_query(task_text: str) -> bool:
    normalized = (task_text or "").lower()
    return any(
        token in normalized
        for token in (
            "according to",
            "treasury bulletin",
            "bulletin",
            "annual report",
            "10-k",
            "10q",
            "filing",
            "document",
            "report",
            "table",
            "page",
            "source document",
        )
    )


def _normalize_tool_families(intent: TaskIntent, task_text: str) -> list[str]:
    normalized: list[str] = []
    mapping = {
        "finance_quant": "market_data_retrieval" if _source_requests_live_data(task_text) else "exact_compute",
        "mathematical_analysis": "analytical_reasoning",
        "transaction_structure_checklist": "transaction_structure_analysis",
        "regulatory_execution_checklist": "regulatory_execution_analysis",
        "tax_structure_checklist": "tax_structure_analysis",
        "risk_analysis": "market_scenario_analysis",
    }
    for family in intent.tool_families_needed:
        value = mapping.get(str(family), str(family))
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_task_intent(intent: TaskIntent, task_text: str, heuristic_intent: TaskIntent) -> TaskIntent:
    candidate = intent.model_copy(deep=True)
    if candidate.task_family == "finance_options" and not _supports_options_fast_path(task_text):
        if _looks_like_market_scenario(task_text, []):
            candidate.task_family = "market_scenario"
            candidate.execution_mode = "advisory_analysis"
            candidate.review_mode = "qualitative_advisory"
            candidate.completion_mode = "compact_sections"
            candidate.tool_families_needed = ["market_scenario_analysis"]
        elif _looks_like_analytical_reasoning(task_text, []):
            candidate.task_family = "analytical_reasoning"
            candidate.execution_mode = "advisory_analysis"
            candidate.review_mode = "analytical_reasoning"
            candidate.completion_mode = "long_form_derivation"
            candidate.tool_families_needed = ["analytical_reasoning", "exact_compute"]
        else:
            candidate.task_family = heuristic_intent.task_family
            candidate.execution_mode = heuristic_intent.execution_mode
            candidate.review_mode = heuristic_intent.review_mode
            candidate.completion_mode = heuristic_intent.completion_mode
            candidate.tool_families_needed = list(heuristic_intent.tool_families_needed)
    if candidate.task_family == "unsupported_artifact":
        candidate.execution_mode = "advisory_analysis"
        candidate.review_mode = "qualitative_advisory"
        candidate.completion_mode = "capability_gap"
        candidate.tool_families_needed = []
    if candidate.task_family in {"external_retrieval", "document_qa"} and _looks_like_grounded_document_query(task_text):
        candidate.execution_mode = "document_grounded_analysis"
        candidate.evidence_strategy = "document_first"
        candidate.review_mode = "document_grounded"
        candidate.completion_mode = "document_grounded"
        candidate.tool_families_needed = list(dict.fromkeys(["document_retrieval", *candidate.tool_families_needed, "external_retrieval"]))
    candidate.tool_families_needed = _normalize_tool_families(candidate, task_text)
    if candidate.task_family == "legal_transactional":
        candidate.tool_families_needed = list(
            dict.fromkeys(
                [
                    *candidate.tool_families_needed,
                    "legal_playbook_retrieval",
                    "transaction_structure_analysis",
                    "regulatory_execution_analysis",
                    "tax_structure_analysis",
                ]
            )
        )
    if candidate.task_family == "external_retrieval":
        candidate.tool_families_needed = list(dict.fromkeys([*candidate.tool_families_needed, "market_data_retrieval", "external_retrieval"]))
    if candidate.task_family == "analytical_reasoning":
        candidate.review_mode = "analytical_reasoning"
        candidate.completion_mode = "long_form_derivation"
        candidate.tool_families_needed = list(dict.fromkeys([*candidate.tool_families_needed, "analytical_reasoning", "exact_compute"]))
    if candidate.task_family == "market_scenario":
        candidate.tool_families_needed = list(dict.fromkeys([*candidate.tool_families_needed, "market_scenario_analysis"]))
        candidate.evidence_strategy = "scenario_first"
    return candidate


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


def _template_stub(intent: TaskIntent, allowed_tools: list[str] | None = None) -> dict[str, Any]:
    review_stages = ["SYNTHESIZE"]
    if intent.execution_mode == "exact_fast_path":
        review_stages = ["COMPUTE", "SYNTHESIZE"]
    return {
        "template_id": intent.execution_mode,
        "description": f"Execution mode: {intent.execution_mode}",
        "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
        "default_initial_stage": "COMPUTE" if intent.execution_mode == "exact_fast_path" else "SYNTHESIZE",
        "allowed_tool_names": list(allowed_tools or []),
        "review_stages": review_stages,
        "review_cadence": "milestone_and_final" if intent.execution_mode == "exact_fast_path" else "final_only",
        "answer_focus": [intent.routing_rationale] if intent.routing_rationale else [],
        "ambiguity_safe": intent.execution_mode == "exact_fast_path",
    }


def _heuristic_intent(task_text: str, answer_contract: dict[str, Any]) -> tuple[TaskIntent, list[str], list[str]]:
    contract_obj = AnswerContract.model_validate(answer_contract or {})
    capability_flags = detect_capability_flags(task_text, contract_obj)
    ambiguity_flags = detect_ambiguity_flags(task_text, capability_flags)
    task_family = infer_task_profile(task_text, capability_flags)
    lowered = (task_text or "").lower()
    has_formula = "=" in task_text or "\\frac" in task_text
    has_table = "|---" in task_text
    exact_contract = bool(answer_contract.get("requires_adapter")) and answer_contract.get("format") == "json"

    if _looks_like_unsupported_artifact(task_text, capability_flags) or task_family == "unsupported_artifact":
        return (
            TaskIntent(
                task_family="unsupported_artifact",
                execution_mode="advisory_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=[],
                evidence_strategy="compact_prompt",
                review_mode="qualitative_advisory",
                completion_mode="capability_gap",
                routing_rationale="Prompt requests a non-finance artifact outside the active engine scope.",
                confidence=0.95,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "finance_quant" and has_formula and has_table and exact_contract and "needs_live_data" not in capability_flags:
        return (
            TaskIntent(
                task_family="finance_quant",
                execution_mode="exact_fast_path",
                complexity_tier="simple_exact",
                tool_families_needed=[],
                evidence_strategy="minimal_exact",
                review_mode="exact_quant",
                completion_mode="scalar_or_json",
                routing_rationale="Prompt contains formula, inline table evidence, and exact output contract.",
                confidence=0.98,
                planner_source="fast_path",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "finance_options" and "needs_options_engine" in capability_flags and _supports_options_fast_path(task_text):
        return (
            TaskIntent(
                task_family="finance_options",
                execution_mode="tool_compute",
                complexity_tier="structured_analysis",
                tool_families_needed=["options_strategy_analysis", "options_scenario_analysis"],
                evidence_strategy="compact_prompt",
                review_mode="tool_compute",
                completion_mode="compact_sections",
                routing_rationale="Options tasks should use structured analysis tools before final synthesis.",
                confidence=0.93,
                planner_source="fast_path",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if _looks_like_market_scenario(task_text, capability_flags) or task_family == "market_scenario":
        return (
            TaskIntent(
                task_family="market_scenario",
                execution_mode="advisory_analysis",
                complexity_tier="complex_qualitative",
                tool_families_needed=["market_scenario_analysis"],
                evidence_strategy="scenario_first",
                review_mode="qualitative_advisory",
                completion_mode="compact_sections",
                routing_rationale="Scenario-driven trading and stress tasks need scenario analysis rather than generic advisory prose.",
                confidence=0.85,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if _looks_like_analytical_reasoning(task_text, capability_flags) or task_family == "analytical_reasoning":
        return (
            TaskIntent(
                task_family="analytical_reasoning",
                execution_mode="advisory_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=["analytical_reasoning", "exact_compute"],
                evidence_strategy="compact_prompt",
                review_mode="analytical_reasoning",
                completion_mode="long_form_derivation",
                routing_rationale="The task requires stepwise symbolic or numerical derivation instead of generic advisory output.",
                confidence=0.84,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "legal_transactional":
        execution_mode = "document_grounded_analysis" if ("needs_files" in capability_flags or "http" in lowered) else "advisory_analysis"
        tool_families = [
            "legal_playbook_retrieval",
            "transaction_structure_analysis",
            "regulatory_execution_analysis",
            "tax_structure_analysis",
        ]
        if execution_mode == "document_grounded_analysis":
            tool_families.insert(0, "document_retrieval")
        return (
            TaskIntent(
                task_family="legal_transactional",
                execution_mode=execution_mode,
                complexity_tier="complex_qualitative",
                tool_families_needed=tool_families,
                evidence_strategy="document_first" if execution_mode == "document_grounded_analysis" else "compact_prompt",
                review_mode="qualitative_advisory",
                completion_mode="advisory_memo",
                routing_rationale="Legal structure/risk/compliance questions need broader checklist and retrieval capabilities.",
                confidence=0.88,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if _looks_like_grounded_document_query(task_text) and task_family not in {"finance_options", "legal_transactional"}:
        return (
            TaskIntent(
                task_family="document_qa" if task_family == "general" else task_family,
                execution_mode="document_grounded_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=["document_retrieval", "external_retrieval"],
                evidence_strategy="document_first",
                review_mode="document_grounded",
                completion_mode="document_grounded",
                routing_rationale="The task appears grounded in reports or source documents, so use retrieval before answering.",
                confidence=0.83,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "document_qa":
        return (
            TaskIntent(
                task_family="document_qa",
                execution_mode="document_grounded_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=["document_retrieval"],
                evidence_strategy="document_first",
                review_mode="document_grounded",
                completion_mode="document_grounded",
                routing_rationale="Document questions should ground the answer in retrieved file evidence.",
                confidence=0.82,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "external_retrieval":
        return (
            TaskIntent(
                task_family="external_retrieval",
                execution_mode="retrieval_augmented_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=["market_data_retrieval", "external_retrieval"],
                evidence_strategy="retrieval_first",
                review_mode="document_grounded",
                completion_mode="compact_sections",
                routing_rationale="The prompt explicitly requests current or source-backed information.",
                confidence=0.8,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    return (
        TaskIntent(
            task_family=task_family,  # type: ignore[arg-type]
            execution_mode="advisory_analysis",
            complexity_tier="complex_qualitative" if ambiguity_flags else "structured_analysis",
            tool_families_needed=["market_data_retrieval"] if _source_requests_live_data(task_text) else [],
            evidence_strategy="compact_prompt",
            review_mode="qualitative_advisory",
            completion_mode="compact_sections",
            routing_rationale="No high-confidence fast path was detected; use a general advisory flow.",
            confidence=0.68,
            planner_source="heuristic",
        ),
        capability_flags,
        ambiguity_flags,
    )


def fast_path_gate(state: RuntimeState) -> dict[str, Any]:
    step = increment_runtime_step()
    task_text = latest_human_text(state["messages"])
    answer_contract = state.get("answer_contract", {}) or {}
    intent, capability_flags, ambiguity_flags = _heuristic_intent(task_text, answer_contract)
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

    heuristic_intent, _, _ = _heuristic_intent(task_text, answer_contract)
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
        source_bundle = SourceBundle.model_validate(state.get("source_bundle") or build_source_bundle(task_text).model_dump())
        intent = TaskIntent.model_validate(state.get("task_intent") or {})
        tool_plan, _ = resolve_tool_plan(intent, source_bundle, registry)
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
    intent = TaskIntent.model_validate(state.get("task_intent") or {})
    source_bundle = SourceBundle.model_validate(state.get("source_bundle") or build_source_bundle(task_text).model_dump())
    curated_context, evidence_stats = build_curated_context(task_text, answer_contract, intent, source_bundle)
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
    )
    logger.info("[Step %s] engine context_curator -> facts=%s", step, len(curated_context.facts_in_use))
    return {
        "source_bundle": source_bundle.model_dump(),
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
        "workpad": workpad,
    }


def _tool_lookup(registry: dict[str, dict[str, Any]], tool_name: str) -> Any | None:
    payload = registry.get(tool_name) or {}
    return payload.get("tool")


def _structured_tool_args(state: RuntimeState, tool_name: str) -> dict[str, Any]:
    source_bundle = SourceBundle.model_validate(state.get("source_bundle") or {})
    task_text = source_bundle.task_text
    if tool_name == "legal_playbook_retrieval":
        return {"query": source_bundle.focus_query or task_text[:240], "deal_size_hint": "", "urgency": ""}
    if tool_name == "transaction_structure_checklist":
        return {
            "consideration_preference": "stock" if "stock" in task_text.lower() else "",
            "liability_goal": "minimize inherited liabilities" if "liabil" in task_text.lower() else "",
            "urgency": "accelerated" if "quick" in task_text.lower() else "",
        }
    if tool_name == "regulatory_execution_checklist":
        jurisdictions = []
        lowered = task_text.lower()
        if "eu" in lowered:
            jurisdictions.append("EU")
        if re.search(r"\bus\b|\bunited states\b", lowered):
            jurisdictions.append("US")
        return {"jurisdictions_json": json.dumps(jurisdictions), "regulatory_gaps": "regulatory compliance gaps" if "compliance gap" in lowered or "regulatory" in lowered else ""}
    if tool_name == "tax_structure_checklist":
        lowered = task_text.lower()
        return {
            "consideration_preference": "stock" if "stock" in lowered else "",
            "cross_border": bool("eu" in lowered and ("us" in lowered or "united states" in lowered)),
        }
    if tool_name == "internet_search":
        return {"query": source_bundle.focus_query or task_text[:240]}
    if tool_name == "search_reference_corpus":
        return {"query": source_bundle.focus_query or task_text[:240], "top_k": 5, "snippet_chars": 700}
    if tool_name == "fetch_reference_file":
        if source_bundle.urls:
            return {"url": source_bundle.urls[0], "page_start": 0, "page_limit": 5, "row_offset": 0, "row_limit": 200}
        return {}
    if tool_name == "list_reference_files":
        return {"prompt_text": source_bundle.task_text}
    if tool_name == "fetch_corpus_document":
        return {}
    if tool_name == "analyze_strategy":
        inline_facts = dict(source_bundle.inline_facts)
        market_snapshot, derived = derive_market_snapshot(source_bundle.task_text, inline_facts)
        prompt_facts = dict(inline_facts)
        if market_snapshot:
            prompt_facts["market_snapshot"] = market_snapshot
        compat_state = {
            "messages": state.get("messages", []),
            "evidence_pack": {
                "prompt_facts": prompt_facts,
                "derived_facts": derived,
                "policy_context": {},
            },
            "workpad": {"tool_results": []},
            "assumption_ledger": state.get("assumption_ledger", []),
        }
        tool_call = deterministic_policy_options_tool_call(compat_state) or deterministic_standard_options_tool_call(compat_state)
        if tool_call:
            return dict(tool_call.get("arguments", {}))
        return {}
    if tool_name == "scenario_pnl":
        journal = ExecutionJournal.model_validate(state.get("execution_journal") or {})
        for result in reversed(journal.tool_results):
            args = scenario_args_from_primary_tool(result)
            if args:
                return args
        return {}
    return {}


async def _invoke_tool(tool_obj: Any, args: dict[str, Any]) -> Any:
    if hasattr(tool_obj, "ainvoke"):
        return await tool_obj.ainvoke(args)
    return tool_obj.invoke(args)


async def _run_tool_step(state: RuntimeState, registry: dict[str, dict[str, Any]], tool_name: str) -> tuple[dict[str, Any], Any]:
    tool_obj = _tool_lookup(registry, tool_name)
    args = _structured_tool_args(state, tool_name)
    if tool_obj is None:
        return args, normalize_tool_output(tool_name, {"error": f"Tool '{tool_name}' is not registered."}, args)
    raw = await _invoke_tool(tool_obj, args)
    return args, normalize_tool_output(tool_name, raw, args)


async def _run_tool_step_with_args(
    state: RuntimeState,
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    args_override: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    tool_obj = _tool_lookup(registry, tool_name)
    if tool_obj is None:
        return args_override, normalize_tool_output(tool_name, {"error": f"Tool '{tool_name}' is not registered."}, args_override)
    raw = await _invoke_tool(tool_obj, args_override)
    return args_override, normalize_tool_output(tool_name, raw, args_override)


def _retrieval_tools_available(tool_plan: ToolPlan) -> list[str]:
    retrieval_tools = []
    for tool_name in tool_plan.selected_tools:
        if tool_name in {
            "internet_search",
            "search_reference_corpus",
            "fetch_corpus_document",
            "list_reference_files",
            "fetch_reference_file",
        }:
            retrieval_tools.append(tool_name)
    return retrieval_tools


def _derive_retrieval_seed_query(source_bundle: SourceBundle) -> str:
    parts = [source_bundle.focus_query or source_bundle.task_text]
    if source_bundle.entities:
        parts.append(" ".join(source_bundle.entities[:4]))
    if source_bundle.target_period:
        parts.append(source_bundle.target_period)
    seed = " ".join(part.strip() for part in parts if part and part.strip())
    return re.sub(r"\s+", " ", seed).strip()[:280]


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
    facts = dict(tool_result.get("facts") or {})
    candidates: list[dict[str, Any]] = []
    for item in facts.get("documents", []):
        if isinstance(item, dict):
            candidates.append(
                {
                    "document_id": str(item.get("document_id", "")),
                    "citation": str(item.get("citation", "") or item.get("url", "") or item.get("path", "")),
                    "path": str(item.get("path", "")),
                }
            )
    for item in facts.get("results", []):
        if isinstance(item, dict):
            candidates.append(
                {
                    "document_id": str(item.get("document_id", "")),
                    "citation": str(item.get("url", "") or item.get("citation", "")),
                    "path": str(item.get("path", "")),
                }
            )
    return [candidate for candidate in candidates if candidate.get("citation") or candidate.get("document_id") or candidate.get("path")]


def _fallback_retrieval_action(
    *,
    source_bundle: SourceBundle,
    journal: ExecutionJournal,
    tool_plan: ToolPlan,
) -> RetrievalAction:
    available = _retrieval_tools_available(tool_plan)
    last_result = journal.tool_results[-1] if journal.tool_results else {}
    last_type = str(last_result.get("type", ""))
    candidates = _search_result_candidates(last_result)
    seed_query = _derive_retrieval_seed_query(source_bundle)

    if not journal.tool_results:
        if source_bundle.urls and "list_reference_files" in available:
            return RetrievalAction(action="tool", tool_name="list_reference_files", query=source_bundle.task_text, rationale="Discover prompt-supplied reference files.")
        if "search_reference_corpus" in available:
            return RetrievalAction(action="tool", tool_name="search_reference_corpus", query=seed_query, rationale="Search the local document corpus first.")
        if source_bundle.urls and "fetch_reference_file" in available:
            return RetrievalAction(action="tool", tool_name="fetch_reference_file", url=source_bundle.urls[0], rationale="Read the first supplied reference file.")
        if "internet_search" in available:
            return RetrievalAction(action="tool", tool_name="internet_search", query=seed_query, rationale="Search the web for a supporting source.")
        return RetrievalAction(action="answer", rationale="No retrieval tools are available.")

    if last_type in {"search_reference_corpus", "list_reference_files"} and candidates:
        first = candidates[0]
        if last_type == "search_reference_corpus" and "fetch_corpus_document" in available:
            return RetrievalAction(
                action="tool",
                tool_name="fetch_corpus_document",
                document_id=first.get("document_id", ""),
                path=first.get("path", "") or first.get("citation", ""),
                rationale="Read the top matching corpus document.",
            )
        if "fetch_reference_file" in available:
            return RetrievalAction(
                action="tool",
                tool_name="fetch_reference_file",
                url=first.get("citation", ""),
                rationale="Read the top matching reference document.",
            )

    if last_type == "internet_search" and candidates and "fetch_reference_file" in available:
        return RetrievalAction(
            action="tool",
            tool_name="fetch_reference_file",
            url=candidates[0].get("citation", ""),
            rationale="Open the top search result for grounded evidence.",
        )

    if last_type in {"fetch_reference_file", "fetch_corpus_document"}:
        facts = dict(last_result.get("facts") or {})
        if facts.get("chunks") or facts.get("tables") or facts.get("numeric_summaries"):
            return RetrievalAction(action="answer", rationale="Retrieved document evidence is available for the final answer.")

    if journal.retrieval_iterations >= _MAX_RETRIEVAL_HOPS - 1:
        return RetrievalAction(action="answer", rationale="Retrieval hop budget exhausted.")

    if "internet_search" in available and last_type != "internet_search":
        query = journal.retrieval_queries[-1] if journal.retrieval_queries else seed_query
        return RetrievalAction(action="tool", tool_name="internet_search", query=query, rationale="Broaden search after insufficient local evidence.")

    return RetrievalAction(action="answer", rationale="No better retrieval action is available.")


def _validate_retrieval_action(action: RetrievalAction, tool_plan: ToolPlan) -> RetrievalAction:
    allowed_tools = set(_retrieval_tools_available(tool_plan))
    if action.action != "tool":
        return action
    if action.tool_name not in allowed_tools:
        return RetrievalAction(action="answer", rationale="Planned retrieval tool is not available in the current tool plan.")
    if action.tool_name in {"internet_search", "search_reference_corpus"} and not action.query.strip():
        return RetrievalAction(action="answer", rationale="Search action did not produce a usable query.")
    if action.tool_name == "fetch_reference_file" and not action.url.strip():
        return RetrievalAction(action="answer", rationale="Fetch action did not produce a usable URL.")
    if action.tool_name == "fetch_corpus_document" and not (action.document_id.strip() or action.path.strip()):
        return RetrievalAction(action="answer", rationale="Corpus fetch action did not identify a document.")
    return action


def _plan_retrieval_action(
    *,
    state: RuntimeState,
    source_bundle: SourceBundle,
    tool_plan: ToolPlan,
    journal: ExecutionJournal,
) -> RetrievalAction:
    """Deterministic retrieval planner — no inner LLM call.

    Uses the heuristic state machine in _fallback_retrieval_action to pick
    the next atomic tool action.  This keeps the executor node fast and
    avoids un-traced blocking API calls inside the graph step.
    """
    available_tools = _retrieval_tools_available(tool_plan)
    if not available_tools or journal.retrieval_iterations >= _MAX_RETRIEVAL_HOPS:
        return RetrievalAction(action="answer", rationale="No remaining retrieval capacity.")

    heuristic = _fallback_retrieval_action(
        source_bundle=source_bundle, journal=journal, tool_plan=tool_plan
    )
    return _validate_retrieval_action(heuristic, tool_plan)



def _tool_args_from_retrieval_action(action: RetrievalAction, source_bundle: SourceBundle) -> dict[str, Any]:
    if action.tool_name == "internet_search":
        return {"query": action.query or _derive_retrieval_seed_query(source_bundle)}
    if action.tool_name == "search_reference_corpus":
        return {"query": action.query or _derive_retrieval_seed_query(source_bundle), "top_k": 5, "snippet_chars": 700}
    if action.tool_name == "list_reference_files":
        return {"prompt_text": source_bundle.task_text}
    if action.tool_name == "fetch_reference_file":
        return {
            "url": action.url,
            "page_start": action.page_start,
            "page_limit": max(2, action.page_limit),
            "row_offset": action.row_offset,
            "row_limit": max(100, action.row_limit),
        }
    if action.tool_name == "fetch_corpus_document":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "chunk_start": action.chunk_start,
            "chunk_limit": max(1, action.chunk_limit),
        }
    return {}


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
        if derivation_heavy or prompt_tokens >= 5000:
            return 3200
        if prompt_tokens >= 2400 or review_feedback:
            return 2800
        return 2200
    if mode == "document_grounded":
        if derivation_heavy or prompt_tokens >= 7000:
            return 3200
        if prompt_tokens >= 3500 or review_feedback:
            return 2800
        return 2400
    if mode == "advisory_memo":
        if task_family == "legal_transactional":
            if prompt_tokens >= 5000:
                return 3200
            if prompt_tokens >= 2400 or review_feedback:
                return 2800
            if prompt_tokens >= 1400:
                return 2400
            return 2200
        if prompt_tokens >= 3500 or review_feedback:
            return 2600
        if prompt_tokens >= 2000:
            return 2200
        return 1800
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
        journal = ExecutionJournal.model_validate(state.get("execution_journal") or {})
        workpad = dict(state.get("workpad", {}))
        review_feedback = dict(state.get("review_feedback") or {})
        budget = state.get("budget_tracker")
        tracker = state.get("cost_tracker")
        tracer = get_tracer()
        tools_ran_this_call: list[str] = []
        task_text = latest_human_text(state["messages"])

        if intent.task_family == "finance_options" and not _supports_options_fast_path(task_text):
            heuristic_intent, _, _ = _heuristic_intent(task_text, state.get("answer_contract", {}) or {})
            intent = _normalize_task_intent(heuristic_intent, task_text, heuristic_intent)
            tool_plan, _ = resolve_tool_plan(intent, source_bundle, registry)
            workpad = _record_event(workpad, "executor", f"rerouted before tool execution -> {intent.task_family}")

        if tool_plan.stop_reason == "unsupported_capability":
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
                ).model_dump(),
                "solver_stage": "COMPLETE",
                "workpad": workpad,
            }

        if tool_plan.stop_reason == "no_bindable_capability" and not tool_plan.pending_tools and not review_feedback:
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
                ).model_dump(),
                "solver_stage": "COMPLETE",
                "workpad": workpad,
            }

        if intent.execution_mode == "exact_fast_path" and intent.task_family == "finance_quant":
            answer = _exact_quant_answer(state)
            if answer:
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
                    ).model_dump(),
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
                        "solver_stage": "GATHER" if intent.execution_mode in _RETRIEVAL_EXECUTION_MODES else "COMPUTE",
                        "workpad": workpad,
                    }

        if intent.execution_mode in _RETRIEVAL_EXECUTION_MODES and not review_feedback and journal.retrieval_iterations < _MAX_RETRIEVAL_HOPS:
            retrieval_action = _plan_retrieval_action(
                state=state,
                source_bundle=source_bundle,
                tool_plan=tool_plan,
                journal=journal,
            )
            if retrieval_action.action == "tool":
                if budget and budget.tool_calls_exhausted():
                    budget.log_budget_exit("tool_budget_exhausted", f"Blocked tool '{retrieval_action.tool_name}' after reaching tool-call cap.")
                else:
                    tool_args = _tool_args_from_retrieval_action(retrieval_action, source_bundle)
                    _, tool_result = await _run_tool_step_with_args(state, registry, retrieval_action.tool_name, tool_args)
                    if tracker:
                        tracker.record_mcp_call()
                    if budget:
                        budget.record_tool_call()
                    tools_ran_this_call.append(retrieval_action.tool_name)
                    journal.tool_results.append(tool_result.model_dump())
                    journal.retrieval_iterations += 1
                    if retrieval_action.tool_name in {"internet_search", "search_reference_corpus"} and tool_args.get("query"):
                        journal.retrieval_queries.append(str(tool_args["query"]))
                    for citation in _tool_result_citations(tool_result.model_dump()):
                        if citation not in journal.retrieved_citations:
                            journal.retrieved_citations.append(citation)
                    if retrieval_action.tool_name not in tool_plan.selected_tools:
                        tool_plan.selected_tools.append(retrieval_action.tool_name)
                    workpad = _record_event(workpad, "executor", f"retrieval hop -> {retrieval_action.tool_name}")
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
                        "solver_stage": "GATHER",
                        "workpad": workpad,
                    }
            else:
                workpad = _record_event(workpad, "executor", f"retrieval ready to answer -> {retrieval_action.rationale or 'evidence collected'}")

        if intent.task_family == "finance_options" and journal.tool_results:
            answer = _options_final_answer({**state, "execution_journal": journal.model_dump()})
            if answer:
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
                    "review_packet": build_review_packet(
                        task_text=task_text,
                        answer_text=answer,
                        answer_contract=state.get("answer_contract", {}) or {},
                        curated_context=curated.model_dump(),
                        tool_results=journal.tool_results,
                    ).model_dump(),
                    "solver_stage": "SYNTHESIZE",
                    "review_feedback": None,
                    "workpad": workpad,
                }

        is_revision = bool(review_feedback)
        prompt_messages = [
            SystemMessage(content=EXECUTOR_SYSTEM),
        ]
        guidance = execution_guidance(intent.task_family, intent.execution_mode)
        if guidance:
            prompt_messages.append(SystemMessage(content=guidance))
        if is_revision:
            revision_text = build_revision_prompt(
                missing_dimensions=review_feedback.get("missing_dimensions", []),
                improve_hint=review_feedback.get("improve_hint", ""),
                reviewer_reasoning=review_feedback.get("reasoning", ""),
                task_family=intent.task_family,
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
        model = ChatOpenAI(model=model_name, **get_client_kwargs("solver"), temperature=0, max_tokens=max_tokens)
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
            "review_packet": build_review_packet(
                task_text=task_text,
                answer_text=content,
                answer_contract=state.get("answer_contract", {}) or {},
                curated_context=curated.model_dump(),
                tool_results=journal.tool_results,
            ).model_dump(),
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
    if any("multiple structure alternatives" in item or "structure alternatives" in item for item in normalized):
        additions.append("multiple viable structures with tradeoffs and a clear recommendation")
    if any("liability allocation" in item for item in normalized):
        additions.append("explicit indemnities, escrow or holdback, caps or baskets, and survival periods")
    if any("regulatory execution" in item for item in normalized):
        additions.append("regulatory approvals, pre-closing remediation covenants, third-party consents, and closing-condition mechanics")
    if any("tax execution" in item for item in normalized):
        additions.append("who gets the economic or tax benefit, required elections or qualification conditions, and what breaks the intended treatment")
    if any("employee-transfer" in item or "workforce transfer" in item or "consultation" in item for item in normalized):
        additions.append("workforce transfer, consultation, retained-liability, and cross-border transition mechanics when relevant")
    if any("actionable next steps" in item or "next steps" in item for item in normalized):
        additions.append("a concrete first-week execution plan with owners and sequencing")
    if any("opening summary" in item or "front-loaded" in item for item in normalized):
        additions.append("an opening summary that names multiple viable paths, their tradeoffs, and the recommended path before deeper analysis")
    if not additions:
        additions.append("one concise but concrete layer of actionable detail for each missing dimension")
    return "Add concise execution detail covering " + ", ".join(additions[:3]) + "."


def reviewer(state: RuntimeState) -> dict[str, Any]:
    step = increment_runtime_step()
    intent = TaskIntent.model_validate(state.get("task_intent") or {})
    journal = ExecutionJournal.model_validate(state.get("execution_journal") or {})
    curated = CuratedContext.model_validate(state.get("curated_context") or {})
    tool_plan = ToolPlan.model_validate(state.get("tool_plan") or {})
    workpad = dict(state.get("workpad", {}))
    answer = _latest_public_answer(state)
    answer_contract = state.get("answer_contract", {}) or {}
    review_packet = build_review_packet(
        task_text=latest_human_text(state["messages"]),
        answer_text=answer,
        answer_contract=answer_contract,
        curated_context=curated.model_dump(),
        tool_results=journal.tool_results,
    )

    missing: list[str] = []
    verdict = "pass"
    reasoning = "Reviewer accepted the final artifact."
    score = 0.9
    stop_reason = ""

    if looks_truncated(answer):
        verdict = "revise" if journal.revision_count < 1 else "fail"
        reasoning = "Final answer appears truncated."
        missing = ["complete final answer"]
        score = 0.3 if verdict == "revise" else 0.24
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
        answer_lower = (answer or "").lower()
        has_inline_attribution = any(token in answer_lower for token in ("source:", "citation", "[source", "(source"))
        if review_packet.citations and not has_inline_attribution:
            missing.append("inline source attribution")
        if review_packet.tool_findings and not has_inline_attribution and "\"" not in answer and "quote" not in answer_lower:
            missing.append("quoted supporting evidence or extracted table row")
        if missing:
            verdict = "revise" if journal.revision_count < 1 else "fail"
            reasoning = "Grounded retrieval answer needs clearer evidence usage, attribution, or quoted support."
            score = 0.58 if verdict == "revise" else 0.48
    elif intent.task_family == "legal_transactional":
        missing = legal_depth_gaps(answer, latest_human_text(state["messages"]))
        if _legal_structure_option_count(answer) < 2:
            missing = sorted(set([*missing, "multiple structure alternatives with tradeoffs"]))
        if _legal_frontload_gap(answer):
            missing = sorted(set([*missing, "opening summary with multiple viable paths before the deep dive"]))
        if missing:
            verdict = "revise" if journal.revision_count < 1 else "fail"
            reasoning = "Final legal answer is directionally correct but still missing actionable execution depth or structure coverage."
            score = 0.62 if verdict == "revise" else 0.58
    elif intent.task_family == "finance_options":
        option_missing = options_gaps(answer)
        if not journal.tool_results:
            option_missing.insert(0, "tool-backed strategy analysis")
        missing = sorted(set(option_missing))
        if missing:
            verdict = "revise" if journal.revision_count < 1 else "fail"
            reasoning = "Options answer is missing benchmark-critical strategy or risk dimensions."
            score = 0.68 if verdict == "revise" else 0.6

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
                "used_llm": False,
            },
        )
    logger.info("[Step %s] engine reviewer -> %s", step, verdict.upper())

    if verdict == "pass":
        return {
            "quality_report": report.model_dump(),
            "execution_journal": journal.model_dump(),
            "review_packet": review_packet.model_dump(),
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
    return {
        "quality_report": report.model_dump(),
        "execution_journal": journal.model_dump(),
        "review_packet": review_packet.model_dump(),
        "progress_signature": progress.model_dump(),
        "review_feedback": None,
        "solver_stage": "COMPLETE",
        "workpad": workpad,
    }


def route_from_reviewer(state: RuntimeState) -> str:
    if state.get("solver_stage") == "REVISE":
        return "executor"
    if state.get("solver_stage") == "COMPLETE":
        intent = TaskIntent.model_validate(state.get("task_intent") or {})
        report = QualityReport.model_validate(state.get("quality_report") or {})
        if report.stop_reason == "exact_output_collapse" and state.get("answer_contract", {}).get("requires_adapter"):
            return "output_adapter"
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
        model = ChatOpenAI(model=model_name, **get_client_kwargs("profiler"), temperature=0, max_tokens=200)
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
