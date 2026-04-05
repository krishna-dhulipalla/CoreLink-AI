from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.contracts import OfficeQALLMRepairDecision, RetrievalIntent, SourceBundle
from agent.model_config import invoke_structured_output
from agent.prompts import OFFICEQA_STRUCTURED_REPAIR_SYSTEM, build_officeqa_structured_repair_prompt

_MAX_QUERY_REWRITE_CALLS = 1
_MAX_VALIDATOR_REPAIR_CALLS = 1


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


def _invoke_repair_decision(prompt: str) -> OfficeQALLMRepairDecision | None:
    messages = [
        SystemMessage(content=OFFICEQA_STRUCTURED_REPAIR_SYSTEM),
        HumanMessage(content=prompt),
    ]
    try:
        parsed, _ = invoke_structured_output(
            "profiler",
            OfficeQALLMRepairDecision,
            messages,
            temperature=0,
            max_tokens=260,
        )
        decision = OfficeQALLMRepairDecision.model_validate(parsed)
    except Exception:
        return None
    if decision.confidence < 0.45:
        return None
    if decision.decision == "rewrite_query" and not decision.revised_query.strip():
        return None
    if decision.decision == "retune_table_query" and not decision.revised_table_query.strip():
        return None
    if decision.decision == "change_strategy" and not decision.preferred_strategy:
        return None
    return decision


def maybe_rewrite_retrieval_path(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    retrieval_strategy: str,
    evidence_gap: str,
    current_query: str = "",
    current_table_query: str = "",
    candidate_sources: list[dict[str, Any]] | None = None,
) -> OfficeQALLMRepairDecision | None:
    if evidence_gap not in {"wrong document", "wrong table family", "missing month coverage"}:
        return None
    prompt = build_officeqa_structured_repair_prompt(
        task_text=task_text,
        retrieval_strategy=retrieval_strategy,
        evidence_gap=evidence_gap,
        current_query=current_query or source_bundle.focus_query,
        current_table_query=current_table_query,
        candidate_sources=candidate_sources,
        review_feedback=None,
    )
    return _invoke_repair_decision(prompt)


def maybe_repair_from_validator(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    review_feedback: dict[str, Any],
    candidate_sources: list[dict[str, Any]] | None = None,
) -> OfficeQALLMRepairDecision | None:
    if str(review_feedback.get("repair_target", "") or "") != "gather":
        return None
    prompt = build_officeqa_structured_repair_prompt(
        task_text=task_text,
        retrieval_strategy=retrieval_intent.strategy,
        evidence_gap=", ".join(str(item) for item in list(review_feedback.get("missing_dimensions", []))[:4]),
        current_query=(retrieval_intent.query_plan.primary_semantic_query or ""),
        current_table_query="",
        candidate_sources=candidate_sources,
        review_feedback=review_feedback,
    )
    return _invoke_repair_decision(prompt)
