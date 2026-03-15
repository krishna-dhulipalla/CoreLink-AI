"""
Reviewer Node
=============
Milestone and final review only. The reviewer no longer sits in every tool hop.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.contracts import ReviewResult
from agent.document_evidence import has_extracted_document_evidence
from agent.cost import CostTracker
from agent.model_config import _extract_json_payload, get_client_kwargs, get_model_name
from agent.profile_packs import get_profile_pack
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import (
    artifact_checkpoint_from_state,
    latest_human_text,
    next_stage_after_review,
    selective_backtracking_allowed,
    should_checkpoint_stage,
)
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)

_REVIEW_PROMPT = """You are the stage reviewer.
Review the provided stage artifact only.
Return ONLY one JSON object with:
- verdict: pass | revise | backtrack
- reasoning: short explanation
- missing_dimensions: list of short strings
- repair_target: gather | compute | synthesize | final

Do not answer the task.
Do not produce user-facing prose.
"""

def _record_event(workpad: dict[str, Any], action: str) -> dict[str, Any]:
    updated = dict(workpad)
    events = list(updated.get("events", []))
    events.append({"node": "reviewer", "action": action})
    updated["events"] = events
    return updated


def _append_review_result(
    workpad: dict[str, Any],
    verdict: ReviewResult,
    review_stage: str,
    is_final: bool,
) -> dict[str, Any]:
    updated = dict(workpad)
    history = list(updated.get("review_results", []))
    history.append(
        {
            "review_stage": review_stage,
            "is_final": is_final,
            "verdict": verdict.verdict,
            "reasoning": verdict.reasoning,
            "missing_dimensions": list(verdict.missing_dimensions),
            "repair_target": verdict.repair_target,
        }
    )
    updated["review_results"] = history
    return updated


def _artifact_for_review(state: AgentState) -> tuple[str, bool]:
    workpad = state.get("workpad", {})
    review_stage = workpad.get("review_stage", state.get("solver_stage", "SYNTHESIZE"))
    if review_stage == "GATHER":
        artifact = json.dumps(state.get("last_tool_result") or workpad.get("stage_outputs", {}).get("GATHER", {}), ensure_ascii=True)
        return artifact, False
    if review_stage == "COMPUTE":
        artifact = str(workpad.get("stage_outputs", {}).get("COMPUTE", "")).strip()
        return artifact, False

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content), True
    return str(workpad.get("draft_answer", "")), True


def _looks_truncated(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith((":", ",", "(", "[", "{", "/", "-", "**")):
        return True
    last_line = stripped.splitlines()[-1].strip()
    if last_line.startswith(("-", "*")) and len(last_line) < 8:
        return True
    return False


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value.strip()))
    return False


def _matches_exact_json_contract(answer_text: str, answer_contract: dict[str, Any]) -> bool:
    stripped = (answer_text or "").strip()
    wrapper_key = answer_contract.get("wrapper_key")
    if _is_numeric_like(stripped):
        return True
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    if wrapper_key:
        return isinstance(parsed, dict) and wrapper_key in parsed and _is_numeric_like(parsed.get(wrapper_key))
    return isinstance(parsed, (dict, list, int, float, str))


def _keyword_gaps(answer_text: str, dimensions: dict[str, list[str]]) -> list[str]:
    normalized = re.sub(r"\s+", " ", (answer_text or "").lower()).strip()
    gaps = []
    for label, tokens in dimensions.items():
        if not any(token in normalized for token in tokens):
            gaps.append(label)
    return gaps


def _options_gaps(answer_text: str) -> list[str]:
    return _keyword_gaps(answer_text, get_profile_pack("finance_options").reviewer_dimensions)


def _has_undisclosed_required_assumption(state: AgentState, answer_text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (answer_text or "").lower()).strip()
    for record in state.get("assumption_ledger", []):
        if not isinstance(record, dict):
            continue
        if not record.get("requires_user_visible_disclosure"):
            continue
        key = str(record.get("key", "")).lower()
        if key == "spot_price" and "spot" not in normalized and "assum" not in normalized:
            return True
    return False


def _deterministic_review(state: AgentState, artifact: str, is_final: bool) -> ReviewResult | None:
    profile = state.get("task_profile", "general")
    workpad = state.get("workpad", {})
    review_stage = workpad.get("review_stage", state.get("solver_stage", "SYNTHESIZE"))
    last_tool_result = state.get("last_tool_result") or {}
    profile_pack = get_profile_pack(profile)
    tool_results = workpad.get("tool_results", [])
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    document_evidence = (state.get("evidence_pack") or {}).get("document_evidence", [])

    if review_stage == "GATHER" and last_tool_result.get("errors"):
        if selective_backtracking_allowed(state.get("execution_template"), "GATHER"):
            return ReviewResult(
                verdict="backtrack",
                reasoning="Gather stage produced invalid or unusable tool evidence.",
                missing_dimensions=["valid evidence"],
                repair_target="gather",
            )
        return ReviewResult(
            verdict="revise",
            reasoning="Gather stage produced invalid or unusable tool evidence.",
            missing_dimensions=["valid evidence"],
            repair_target="gather",
        )

    if review_stage == "COMPUTE" and last_tool_result.get("errors"):
        if selective_backtracking_allowed(state.get("execution_template"), "COMPUTE"):
            return ReviewResult(
                verdict="backtrack",
                reasoning="Compute stage is grounded in an invalid tool result and should revert to the last stable artifact state.",
                missing_dimensions=["valid compute evidence"],
                repair_target="compute",
            )
        return ReviewResult(
            verdict="revise",
            reasoning="Compute stage is grounded in an invalid tool result.",
            missing_dimensions=["valid compute evidence"],
            repair_target="compute",
        )

    if review_stage == "GATHER" and template_id in {"legal_with_document_evidence", "document_qa"}:
        if not has_extracted_document_evidence(document_evidence):
            return ReviewResult(
                verdict="revise",
                reasoning="Document gather stage found document references but not targeted extracted evidence yet.",
                missing_dimensions=["targeted document evidence"],
                repair_target="gather",
            )

    if review_stage == "COMPUTE" and not re.search(r"\d", artifact or ""):
        return ReviewResult(
            verdict="revise",
            reasoning="Compute stage did not produce concrete numerical or analytical output.",
            missing_dimensions=["concrete analytical output"],
            repair_target="compute",
        )

    if review_stage == "COMPUTE" and profile == "finance_options":
        has_structured_tool = any(result.get("facts") and not result.get("errors") for result in tool_results)
        if not has_structured_tool:
            return ReviewResult(
                verdict="revise",
                reasoning="Options compute stage requires tool-backed strategy analysis before synthesis.",
                missing_dimensions=["tool-backed strategy analysis"],
                repair_target="compute",
            )

    if is_final and _looks_truncated(artifact):
        return ReviewResult(
            verdict="revise",
            reasoning="Final answer appears truncated or cut off.",
            missing_dimensions=["complete final answer"],
            repair_target="final",
        )

    if is_final and profile == "finance_quant":
        answer_contract = state.get("answer_contract", {})
        if answer_contract.get("requires_adapter") and answer_contract.get("format") == "json":
            if not _matches_exact_json_contract(artifact, answer_contract):
                return ReviewResult(
                    verdict="revise",
                    reasoning="Final quantitative answer must be a scalar value or already match the JSON answer contract.",
                    missing_dimensions=["scalar answer matching output contract"],
                    repair_target="final",
                )

    if is_final and profile == "legal_transactional":
        gaps = _keyword_gaps(artifact, profile_pack.reviewer_dimensions)
        if gaps:
            return ReviewResult(
                verdict="revise",
                reasoning="Final legal answer is directionally correct but incomplete.",
                missing_dimensions=gaps,
                repair_target="final",
            )

    if is_final and profile == "finance_options":
        has_structured_tool = any(result.get("facts") and not result.get("errors") for result in tool_results)
        if not has_structured_tool:
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer must be grounded in at least one structured tool result.",
                missing_dimensions=["tool-backed strategy analysis"],
                repair_target="compute",
            )
        gaps = _options_gaps(artifact)
        if gaps:
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer is missing one or more benchmark-critical dimensions.",
                missing_dimensions=gaps,
                repair_target="final",
            )
        if _has_undisclosed_required_assumption(state, artifact):
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer must disclose material pricing assumptions introduced during compute.",
                missing_dimensions=["disclosed material assumptions"],
                repair_target="final",
            )

    if is_final and profile == "document_qa":
        if not has_extracted_document_evidence(document_evidence):
            return ReviewResult(
                verdict="revise",
                reasoning="Document QA final answer requires extracted document evidence, not URL discovery alone.",
                missing_dimensions=["extracted document evidence"],
                repair_target="gather",
            )

    if is_final and profile_pack.reviewer_dimensions and profile not in {"legal_transactional", "finance_options"}:
        gaps = _keyword_gaps(artifact, profile_pack.reviewer_dimensions)
        if gaps:
            return ReviewResult(
                verdict="revise",
                reasoning="Final answer is incomplete relative to the required profile dimensions.",
                missing_dimensions=gaps,
                repair_target="final",
            )

    return None


def _next_target_for_pass(state: AgentState) -> str:
    profile = state.get("task_profile", "general")
    template = state.get("execution_template", {}) or {}
    allowed_stages = set(template.get("allowed_stages", []))
    review_stage = state.get("workpad", {}).get("review_stage", state.get("solver_stage", "SYNTHESIZE"))
    if review_stage == "GATHER":
        if "COMPUTE" in allowed_stages and (
            profile in {"finance_quant", "finance_options"} or "needs_math" in set(state.get("capability_flags", []))
        ):
            return "compute"
        return "synthesize"
    if review_stage == "COMPUTE":
        return "synthesize"
    return "final"


def _fallback_review_on_parse_failure(state: AgentState, artifact: str, is_final: bool, error: Exception) -> ReviewResult:
    review_stage = state.get("workpad", {}).get("review_stage", state.get("solver_stage", "SYNTHESIZE"))
    logger.warning("Reviewer LLM parse failed: %s. Using deterministic fallback verdict.", error)

    if not str(artifact or "").strip():
        return ReviewResult(
            verdict="revise",
            reasoning="Reviewer fallback: empty artifact cannot be accepted.",
            missing_dimensions=["non-empty artifact"],
            repair_target="final" if is_final else "compute",
        )

    if review_stage == "GATHER":
        return ReviewResult(
            verdict="pass",
            reasoning="Reviewer fallback: gather artifact accepted after deterministic checks.",
            missing_dimensions=[],
            repair_target="compute",
        )
    if review_stage == "COMPUTE":
        return ReviewResult(
            verdict="pass",
            reasoning="Reviewer fallback: compute artifact accepted after deterministic checks.",
            missing_dimensions=[],
            repair_target="synthesize",
        )
    return ReviewResult(
        verdict="pass",
        reasoning="Reviewer fallback: final artifact accepted after deterministic checks.",
        missing_dimensions=[],
        repair_target="final",
    )


def _restore_checkpoint(state: AgentState) -> dict[str, Any]:
    stack = list(state.get("checkpoint_stack", []))
    if stack:
        checkpoint = stack[-1]
        restored_workpad = dict(state.get("workpad", {}))
        restored_workpad["stage_outputs"] = dict(checkpoint.get("stage_outputs", {}))
        draft_answer = str(checkpoint.get("draft_answer", "")).strip()
        if draft_answer:
            restored_workpad["draft_answer"] = draft_answer
        else:
            restored_workpad.pop("draft_answer", None)
        restored_workpad["review_ready"] = False
        restored_workpad["review_stage"] = None
        return {
            "evidence_pack": checkpoint.get("evidence_pack", {}),
            "assumption_ledger": checkpoint.get("assumption_ledger", []),
            "provenance_map": checkpoint.get("provenance_map", {}),
            "workpad": restored_workpad,
            "last_tool_result": checkpoint.get("last_tool_result"),
            "review_feedback": checkpoint.get("review_feedback"),
            "pending_tool_call": None,
        }
    return {
        "pending_tool_call": None,
    }


def route_from_reviewer(state: AgentState) -> str:
    if state.get("solver_stage") == "COMPLETE":
        if state.get("answer_contract", {}).get("requires_adapter"):
            return "output_adapter"
        return "reflect"
    return "solver"


def reviewer(state: AgentState) -> dict:
    step = increment_runtime_step()
    tracker: CostTracker = state.get("cost_tracker")
    artifact, is_final = _artifact_for_review(state)
    workpad = dict(state.get("workpad", {}))
    review_stage = workpad.get("review_stage", state.get("solver_stage", "SYNTHESIZE"))
    deterministic = _deterministic_review(state, artifact, is_final)
    task_text = latest_human_text(state.get("messages", []))

    verdict = deterministic
    latency = 0.0
    model_name = get_model_name("verifier")
    used_llm = False
    invocation_messages = [
        SystemMessage(content=_REVIEW_PROMPT),
        HumanMessage(
            content=json.dumps(
                {
                    "task_profile": state.get("task_profile", "general"),
                    "execution_template": state.get("execution_template", {}),
                    "capability_flags": state.get("capability_flags", []),
                    "is_final": is_final,
                    "review_stage": workpad.get("review_stage", state.get("solver_stage", "SYNTHESIZE")),
                    "task": task_text,
                    "artifact": artifact,
                    "answer_contract": state.get("answer_contract", {}),
                    "assumption_ledger": state.get("assumption_ledger", []),
                    "provenance_summary": {
                        key: {
                            "source_class": value.get("source_class"),
                            "source_id": value.get("source_id"),
                        }
                        for key, value in list((state.get("provenance_map") or {}).items())[:12]
                    },
                    "last_tool_result": state.get("last_tool_result"),
                },
                ensure_ascii=True,
            )
        ),
    ]

    if verdict is None:
        used_llm = True
        llm = ChatOpenAI(
            model=model_name,
            **get_client_kwargs("verifier"),
            temperature=0,
            max_tokens=240,
        )
        t0 = time.monotonic()
        try:
            raw = llm.invoke(invocation_messages)
            latency = (time.monotonic() - t0) * 1000
            payload = json.loads(_extract_json_payload(str(raw.content or "")))
            verdict = ReviewResult.model_validate(payload)
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            verdict = _fallback_review_on_parse_failure(state, artifact, is_final, exc)

    if tracker and used_llm:
        tracker.record(
            operator="reviewer",
            model_name=model_name,
            tokens_in=count_tokens(invocation_messages),
            tokens_out=count_tokens([AIMessage(content=verdict.model_dump_json())]),
            latency_ms=latency,
            success=verdict.verdict == "pass",
        )

    budget = state.get("budget_tracker")
    if verdict.verdict == "backtrack":
        if budget:
            budget.record_backtrack()
        restored = _restore_checkpoint(state)
        workpad = _append_review_result(workpad, verdict, str(review_stage), is_final)
        workpad = _record_event(workpad, f"BACKTRACK: {verdict.reasoning}")
        restored_workpad = dict(restored.get("workpad", {}))
        merged_events = list(restored_workpad.get("events", [])) + list(workpad.get("events", []))
        restored_workpad["events"] = merged_events
        restored_workpad["review_results"] = list(workpad.get("review_results", []))
        logger.info("[Step %s] reviewer -> BACKTRACK", step)
        return {
            **restored,
            "workpad": restored_workpad,
            "review_feedback": verdict.model_dump(),
            "solver_stage": "REVISE",
        }

    if verdict.verdict == "revise":
        if budget:
            budget.record_revise()
        workpad = _append_review_result(workpad, verdict, str(review_stage), is_final)
        workpad = _record_event(workpad, f"REVISE: {verdict.reasoning}")
        workpad["review_ready"] = False
        logger.info("[Step %s] reviewer -> REVISE missing=%s", step, verdict.missing_dimensions)
        return {
            "review_feedback": verdict.model_dump(),
            "solver_stage": "REVISE",
            "workpad": workpad,
        }

    checkpoint_stack = list(state.get("checkpoint_stack", []))
    if should_checkpoint_stage(state.get("execution_template"), str(review_stage)):
        checkpoint_stack.append(
            artifact_checkpoint_from_state(
                state,
                reason=f"stable_{str(review_stage).lower()}",
                stage=str(review_stage),
            )
        )
    next_target = _next_target_for_pass(state)
    next_stage = next_stage_after_review(
        str(workpad.get("review_stage", state.get("solver_stage", "SYNTHESIZE"))),
        next_target,
        "pass",
    )
    workpad = _append_review_result(workpad, verdict, str(review_stage), is_final)
    workpad = _record_event(workpad, f"PASS -> {next_stage}")
    workpad["review_ready"] = False
    workpad["review_stage"] = None
    logger.info("[Step %s] reviewer -> PASS next=%s", step, next_stage)
    return {
        "review_feedback": None,
        "solver_stage": next_stage,
        "checkpoint_stack": checkpoint_stack,
        "workpad": workpad,
    }
