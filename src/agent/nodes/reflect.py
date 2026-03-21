"""
Reflect Node
============
Post-run persistence hook for the staged runtime.
"""

from __future__ import annotations

import logging

from agent.memory.curation import build_curation_signals
from agent.memory.schema import ReviewMemory, RunMemory, ToolMemory, infer_memory_family, normalize_memory_text, task_signature
from agent.memory.store import MemoryStore
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import latest_human_text
from agent.state import AgentState
from agent.tracer import get_tracer

logger = logging.getLogger(__name__)
_memory_store: MemoryStore | None = None


def _get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def _persist_run(state: AgentState, task_text: str, workpad: dict) -> None:
    store = state.get("memory_store") or _get_memory_store()
    tracker = state.get("cost_tracker")
    if store is None or tracker is None:
        return

    task_profile = state.get("task_profile", "general")
    capability_flags = list(state.get("capability_flags", []))
    ambiguity_flags = list(state.get("ambiguity_flags", []))
    execution_template = state.get("execution_template", {}) or {}
    template_id = str(execution_template.get("template_id", ""))
    assumption_ledger = list(state.get("assumption_ledger", []))
    provenance_map = dict(state.get("provenance_map", {}))
    route_path = list(dict.fromkeys(event.get("node", "") for event in workpad.get("events", []) if event.get("node")))
    tool_results = list(workpad.get("tool_results", []))
    review_results = list(workpad.get("review_results", []))
    risk_results = list(workpad.get("risk_results", []))
    compliance_results = list(workpad.get("compliance_results", []))
    record = RunMemory(
        task_signature=task_signature(task_text),
        task_summary=task_text[:160],
        semantic_text=normalize_memory_text(
            f"task: {task_text}\nprofile: {task_profile}\nflags: {' '.join(capability_flags)}\n"
            f"ambiguity: {' '.join(ambiguity_flags)}\ntemplate: {template_id}\nroute: {' > '.join(route_path)}\n"
            f"success: {state.get('solver_stage') == 'COMPLETE'}"
        ),
        task_profile=task_profile,
        task_family=infer_memory_family(task_profile, task_text),
        capability_flags=capability_flags,
        route_path=route_path,
        stage_history=list(workpad.get("stage_history", [])),
        answer_format=str((state.get("answer_contract") or {}).get("format", "text")),
        success=state.get("solver_stage") == "COMPLETE",
        tool_call_count=len(tool_results),
        review_cycle_count=len(review_results),
        cost_usd=tracker.total_cost() if tracker.pricing_status == "known" else 0.0,
        latency_ms=tracker.wall_clock_ms,
        tags=[task_profile, *capability_flags[:4]],
        metadata={
            "requires_adapter": bool((state.get("answer_contract") or {}).get("requires_adapter")),
            "final_stage": state.get("solver_stage", "PLAN"),
            "ambiguity_flags": ambiguity_flags,
            "template_id": template_id,
            "assumption_count": len(assumption_ledger),
            "provenance_count": len(provenance_map),
            "risk_result_count": len(risk_results),
            "compliance_result_count": len(compliance_results),
            "recommendation_class": str((workpad.get("risk_requirements") or {}).get("recommendation_class", "")),
            "cost_estimate_status": tracker.pricing_status,
            "unpriced_models": tracker.unpriced_models,
        },
    )
    store.store_run(record)


def _persist_tools(state: AgentState, task_text: str, workpad: dict) -> None:
    store = state.get("memory_store") or _get_memory_store()

    task_profile = state.get("task_profile", "general")
    task_family = infer_memory_family(task_profile, task_text)
    for result in workpad.get("tool_results", []):
        source = result.get("source", {}) if isinstance(result, dict) else {}
        assumptions = result.get("assumptions", {}) if isinstance(result, dict) else {}
        facts = result.get("facts", {}) if isinstance(result, dict) else {}
        errors = result.get("errors", []) if isinstance(result, dict) else []
        record = ToolMemory(
            task_signature=task_signature(task_text),
            task_profile=task_profile,
            task_family=task_family,
            solver_stage=str(source.get("solver_stage", workpad.get("review_stage") or state.get("solver_stage", "COMPUTE"))),
            tool_name=str(source.get("tool", result.get("type", "unknown"))),
            result_type=str(result.get("type", "unknown")),
            semantic_text=normalize_memory_text(
                f"tool: {source.get('tool', result.get('type', 'unknown'))}\n"
                f"result_type: {result.get('type', 'unknown')}\n"
                f"facts: {' '.join(sorted(facts.keys()))}"
            ),
            arguments_json=assumptions if isinstance(assumptions, dict) else {},
            fact_keys=sorted(facts.keys()) if isinstance(facts, dict) else [],
            error_count=len(errors) if isinstance(errors, list) else 0,
            success=not bool(errors),
            metadata={
                "source": source,
                "provenance_keys": [
                    key for key, value in (state.get("provenance_map") or {}).items()
                    if isinstance(value, dict) and str(value.get("tool_name", "")) == str(source.get("tool", result.get("type", "unknown")))
                ][:20],
            },
        )
        store.store_tool(record)


def _persist_reviews(state: AgentState, task_text: str, workpad: dict) -> None:
    store = state.get("memory_store") or _get_memory_store()

    task_profile = state.get("task_profile", "general")
    task_family = infer_memory_family(task_profile, task_text)
    for result in workpad.get("review_results", []):
        if not isinstance(result, dict):
            continue
        record = ReviewMemory(
            task_signature=task_signature(task_text),
            task_profile=task_profile,
            task_family=task_family,
            review_stage=str(result.get("review_stage", "SYNTHESIZE")),
            verdict=str(result.get("verdict", "revise")),
            repair_target=str(result.get("repair_target", "final")),
            missing_dimensions=list(result.get("missing_dimensions", [])),
            reasoning=normalize_memory_text(str(result.get("reasoning", "")), max_len=220),
            success=str(result.get("verdict", "revise")) == "pass",
            metadata={
                "is_final": bool(result.get("is_final")),
            },
        )
        store.store_review(record)


def _persist_curation(state: AgentState, task_text: str, workpad: dict) -> None:
    store = state.get("memory_store") or _get_memory_store()

    for signal in build_curation_signals(state, task_text, workpad):
        store.store_curation(signal)


def reflect(state: AgentState) -> dict:
    step = increment_runtime_step()
    workpad = dict(state.get("workpad", {}))
    workpad.setdefault("events", []).append({"node": "reflect", "action": "run complete"})
    task_text = latest_human_text(state.get("messages", []))

    try:
        _persist_run(state, task_text, workpad)
        _persist_tools(state, task_text, workpad)
        _persist_reviews(state, task_text, workpad)
        _persist_curation(state, task_text, workpad)
    except Exception as exc:
        logger.warning("[Memory] Failed to persist staged runtime memory: %s", exc)

    route_path = list(dict.fromkeys(
        event.get("node", "") for event in workpad.get("events", []) if event.get("node")
    ))
    tracer = get_tracer()
    if tracer:
        tracer.record("reflect", {
            "route_path": route_path,
            "final_stage": state.get("solver_stage", "PLAN"),
            "success": state.get("solver_stage") == "COMPLETE",
        })

    logger.info("[Step %s] reflect -> complete", step)
    return {"workpad": workpad}
