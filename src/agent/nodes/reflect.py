"""
Reflect Node
============
Post-run persistence hook for the staged runtime.
"""

from __future__ import annotations

import logging

from agent.memory.schema import ReviewMemory, RunMemory, ToolMemory, infer_memory_family, normalize_memory_text, task_signature
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import latest_human_text
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _persist_run(state: AgentState, task_text: str, workpad: dict) -> None:
    store = state.get("memory_store")
    tracker = state.get("cost_tracker")
    if store is None or tracker is None:
        return

    task_profile = state.get("task_profile", "general")
    capability_flags = list(state.get("capability_flags", []))
    ambiguity_flags = list(state.get("ambiguity_flags", []))
    route_path = list(dict.fromkeys(event.get("node", "") for event in workpad.get("events", []) if event.get("node")))
    tool_results = list(workpad.get("tool_results", []))
    review_results = list(workpad.get("review_results", []))
    record = RunMemory(
        task_signature=task_signature(task_text),
        task_summary=task_text[:160],
        semantic_text=normalize_memory_text(
            f"task: {task_text}\nprofile: {task_profile}\nflags: {' '.join(capability_flags)}\n"
            f"ambiguity: {' '.join(ambiguity_flags)}\nroute: {' > '.join(route_path)}\n"
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
        cost_usd=tracker.total_cost(),
        latency_ms=tracker.wall_clock_ms,
        tags=[task_profile, *capability_flags[:4]],
        metadata={
            "requires_adapter": bool((state.get("answer_contract") or {}).get("requires_adapter")),
            "final_stage": state.get("solver_stage", "PLAN"),
            "ambiguity_flags": ambiguity_flags,
        },
    )
    store.store_run(record)


def _persist_tools(state: AgentState, task_text: str, workpad: dict) -> None:
    store = state.get("memory_store")
    if store is None:
        return

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
            },
        )
        store.store_tool(record)


def _persist_reviews(state: AgentState, task_text: str, workpad: dict) -> None:
    store = state.get("memory_store")
    if store is None:
        return

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


def reflect(state: AgentState) -> dict:
    step = increment_runtime_step()
    workpad = dict(state.get("workpad", {}))
    workpad.setdefault("events", []).append({"node": "reflect", "action": "run complete"})
    task_text = latest_human_text(state.get("messages", []))

    try:
        _persist_run(state, task_text, workpad)
        _persist_tools(state, task_text, workpad)
        _persist_reviews(state, task_text, workpad)
    except Exception as exc:
        logger.warning("[Memory] Failed to persist staged runtime memory: %s", exc)

    logger.info("[Step %s] reflect -> complete", step)
    return {"workpad": workpad}
