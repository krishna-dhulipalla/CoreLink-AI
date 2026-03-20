"""
Reviewer Node
=============
Milestone and final review only. The reviewer no longer sits in every tool hop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.contracts import ReviewResult
from agent.context.legal_dimensions import (
    legal_allocation_groups,
    legal_employee_transfer_groups,
    legal_execution_groups,
    legal_regulatory_execution_groups,
    legal_tax_execution_groups,
    normalize_legal_task_text,
)
from agent.document_evidence import has_extracted_document_evidence
from agent.cost import CostTracker
from agent.nodes.compliance_guard import requires_compliance_guard
from agent.nodes.self_reflection import should_run_self_reflection
from agent.model_config import get_model_name, invoke_structured_output
from agent.profile_packs import get_profile_pack
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import (
    artifact_checkpoint_from_state,
    latest_human_text,
    next_stage_after_review,
    selective_backtracking_allowed,
    should_checkpoint_stage,
)
from agent.solver.quant import deterministic_quant_final_answer
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
- repair_class: generic | wrapper_only | scalar_only | missing_section | missing_evidence | missing_risk

Do not answer the task.
Do not produce user-facing prose.
"""


def _should_try_llm_review(state: AgentState, review_stage: str, is_final: bool) -> bool:
    profile = str(state.get("task_profile", "general"))
    ambiguity_flags = set(state.get("ambiguity_flags", []))

    # Intermediate gather/compute milestones are now primarily deterministic.
    # This keeps the reviewer focused on ambiguous final quality judgments
    # instead of adding churn to evidence and compute loops.
    if review_stage in {"GATHER", "COMPUTE"}:
        return False
    if not is_final:
        return False
    if ambiguity_flags:
        return True
    return profile in {"legal_transactional", "general", "external_retrieval"}

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
            "repair_class": verdict.repair_class,
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


def _artifact_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _update_repeat_diagnostics(workpad: dict[str, Any], verdict: ReviewResult, artifact: str) -> dict[str, Any]:
    updated = dict(workpad)
    current_signature = _artifact_signature(artifact)
    previous_signature = str(updated.get("repeat_signature", ""))
    current_missing_signature = "|".join(sorted(str(item).strip().lower() for item in verdict.missing_dimensions if str(item).strip()))
    previous_missing_signature = str(updated.get("last_missing_signature", ""))
    previous_reason = str(updated.get("last_review_reason", ""))
    previous_verdict = str(updated.get("last_review_verdict", ""))
    current_reason = str(verdict.reasoning or "").strip()
    if (
        current_reason
        and current_reason == previous_reason
        and verdict.verdict == previous_verdict
        and (
            (current_signature and current_signature == previous_signature)
            or (current_missing_signature and current_missing_signature == previous_missing_signature)
        )
    ):
        updated["repeat_count"] = int(updated.get("repeat_count", 1)) + 1
    else:
        updated["repeat_count"] = 1
    updated["repeat_signature"] = current_signature
    updated["last_missing_signature"] = current_missing_signature
    updated["last_repair_target"] = verdict.repair_target
    updated["last_review_reason"] = current_reason
    updated["last_review_verdict"] = verdict.verdict
    return updated


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


def _count_token_group_hits(answer_text: str, groups: list[list[str]]) -> int:
    normalized = re.sub(r"\s+", " ", (answer_text or "").lower()).strip()
    hits = 0
    for group in groups:
        if any(token in normalized for token in group):
            hits += 1
    return hits


def _legal_depth_gaps(answer_text: str, task_text: str = "") -> list[str]:
    gaps = _keyword_gaps(answer_text, get_profile_pack("legal_transactional").reviewer_dimensions)
    normalized_task = normalize_legal_task_text(task_text)
    allocation_groups = legal_allocation_groups()
    execution_groups = legal_execution_groups()
    if _count_token_group_hits(answer_text, allocation_groups) < 3:
        gaps.append("liability allocation mechanics")
    if _count_token_group_hits(answer_text, execution_groups) < 2:
        gaps.append("execution timing and closing mechanics")
    if any(token in normalized_task for token in ("stock consideration", "stock-for-stock", "tax reasons", "tax")):
        if _count_token_group_hits(answer_text, legal_tax_execution_groups()) < 2:
            gaps.append("tax execution mechanics")
    if any(token in normalized_task for token in ("eu", "us", "cross-border", "compliance")):
        if _count_token_group_hits(answer_text, legal_regulatory_execution_groups()) < 2:
            gaps.append("regulatory execution specifics")
        if _count_token_group_hits(answer_text, legal_employee_transfer_groups()) < 2:
            gaps.append("employee-transfer considerations")
    return sorted(set(gaps))


def _options_gaps(answer_text: str) -> list[str]:
    return _keyword_gaps(answer_text, get_profile_pack("finance_options").reviewer_dimensions)


_TEMPLATE_REVIEW_DIMENSIONS: dict[str, dict[str, list[str]]] = {
    "equity_research_report": {
        "thesis": ["thesis", "view", "recommendation"],
        "evidence": ["evidence", "fundamental", "revenue", "margin", "cash", "price"],
        "valuation": ["valuation", "multiple", "dcf", "upside", "downside", "target"],
        "risks": ["risk", "downside", "uncertainty", "watch"],
    },
    "portfolio_risk_review": {
        "exposures": ["exposure", "concentration", "factor", "sector", "weight"],
        "stress": ["stress", "scenario", "var", "drawdown", "liquidity"],
        "limits": ["limit", "breach", "cap", "budget"],
        "actions": ["rebalance", "trim", "hedge", "reduce", "hold", "action"],
    },
    "event_driven_finance": {
        "catalyst": ["event", "catalyst", "earnings", "guidance", "corporate action", "fed", "cpi"],
        "market context": ["price", "return", "volatility", "market", "timestamp", "as of"],
        "scenarios": ["base", "upside", "downside", "stress", "scenario"],
        "risk": ["risk", "uncertainty", "watch", "disclosure"],
    },
    "regulated_actionable_finance": {
        "recommendation": ["recommend", "buy", "sell", "hold", "overweight", "underweight"],
        "evidence": ["source", "timestamp", "as of", "evidence"],
        "risk": ["risk", "downside", "uncertainty", "limit"],
        "disclosures": ["recommendation class", "assumption", "disclosure"],
    },
}


def _template_gaps(template_id: str, answer_text: str) -> list[str]:
    dimensions = _TEMPLATE_REVIEW_DIMENSIONS.get(template_id, {})
    if not dimensions:
        return []
    return _keyword_gaps(answer_text, dimensions)


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


def _has_risk_required_disclosures(answer_text: str, disclosures: list[str]) -> bool:
    normalized = re.sub(r"[-/]+", " ", (answer_text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for disclosure in disclosures:
        lowered = re.sub(r"[-/]+", " ", str(disclosure).lower())
        lowered = re.sub(r"\s+", " ", lowered).strip()
        if "short volatility" in lowered or "volatility spike" in lowered:
            if not any(token in normalized for token in ("short vol", "vol spike", "volatility spike")):
                return False
        elif "tail loss" in lowered or "gap risk" in lowered or "unbounded" in lowered:
            if not any(token in normalized for token in ("tail risk", "tail loss", "gap risk", "unbounded")):
                return False
        elif "max loss" in lowered:
            if "max loss" not in normalized:
                return False
        elif "position sizing" in lowered:
            if not any(token in normalized for token in ("position size", "sizing")):
                return False
        elif "downside scenario loss" in lowered:
            if not any(token in normalized for token in ("scenario", "stress", "downside")):
                return False
    return True


def _deterministic_review(state: AgentState, artifact: str, is_final: bool) -> ReviewResult | None:
    profile = state.get("task_profile", "general")
    task_text = latest_human_text(state.get("messages", []))
    workpad = state.get("workpad", {})
    review_stage = workpad.get("review_stage", state.get("solver_stage", "SYNTHESIZE"))
    last_tool_result = state.get("last_tool_result") or {}
    profile_pack = get_profile_pack(profile)
    tool_results = workpad.get("tool_results", [])
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    document_evidence = (state.get("evidence_pack") or {}).get("document_evidence", [])
    capability_flags = set(state.get("capability_flags", []))
    has_structured_tool = any(result.get("facts") and not result.get("errors") for result in tool_results)
    risk_results = list(workpad.get("risk_results", []))
    risk_requirements = workpad.get("risk_requirements") or {}
    compliance_results = list(workpad.get("compliance_results", []))

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

    if review_stage == "GATHER" and profile == "finance_quant" and "needs_live_data" in capability_flags:
        if not (last_tool_result.get("facts") and not last_tool_result.get("errors")):
            return ReviewResult(
                verdict="revise",
                reasoning="Live-data finance gather stage requires a retrieval-backed tool result before synthesis.",
                missing_dimensions=["retrieval-backed finance evidence"],
                repair_target="gather",
            )

    if review_stage == "COMPUTE" and not re.search(r"\d", artifact or ""):
        return ReviewResult(
            verdict="revise",
            reasoning="Compute stage did not produce concrete numerical or analytical output.",
            missing_dimensions=["concrete analytical output"],
            repair_target="compute",
        )

    if review_stage == "COMPUTE" and profile == "finance_quant" and "needs_live_data" in capability_flags:
        if not has_structured_tool:
            return ReviewResult(
                verdict="revise",
                reasoning="Finance compute stage with live-data needs requires structured retrieval evidence before synthesis.",
                missing_dimensions=["tool-backed finance evidence"],
                repair_target="gather",
            )

    if review_stage == "COMPUTE" and profile == "finance_options":
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
            repair_class="generic",
        )

    if is_final and profile == "finance_quant":
        answer_contract = state.get("answer_contract", {})
        template_gaps = _template_gaps(template_id, artifact)
        if template_gaps:
            return ReviewResult(
                verdict="revise",
                reasoning="Final finance answer is incomplete relative to the selected execution template.",
                missing_dimensions=template_gaps,
                repair_target="final",
                repair_class="missing_section",
            )
        if "needs_live_data" in capability_flags and not has_structured_tool:
            return ReviewResult(
                verdict="revise",
                reasoning="Final finance answer with live-data needs must stay grounded in structured retrieval evidence.",
                missing_dimensions=["retrieval-backed finance evidence"],
                repair_target="gather",
                repair_class="missing_evidence",
            )
        if answer_contract.get("requires_adapter") and answer_contract.get("format") == "json":
            if not _matches_exact_json_contract(artifact, answer_contract):
                stripped = (artifact or "").strip()
                repair_class = "wrapper_only" if _is_numeric_like(stripped) else "scalar_only"
                return ReviewResult(
                    verdict="revise",
                    reasoning="Final quantitative answer must be a scalar value or already match the JSON answer contract.",
                    missing_dimensions=["scalar answer matching output contract"],
                    repair_target="final",
                    repair_class=repair_class,
                )

    if is_final and profile == "legal_transactional":
        gaps = _legal_depth_gaps(artifact, task_text)
        if gaps:
            return ReviewResult(
                verdict="revise",
                reasoning="Final legal answer is directionally correct but incomplete.",
                missing_dimensions=gaps,
                repair_target="final",
                repair_class="missing_section",
            )

    if is_final and profile == "finance_options":
        if not risk_results:
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer requires a risk-controller pass before acceptance.",
                missing_dimensions=["risk-controller validation"],
                repair_target="compute",
                repair_class="missing_risk",
            )
        if str(risk_results[-1].get("verdict", "pass")) != "pass":
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer cannot be accepted while the latest risk-controller result is unresolved.",
                missing_dimensions=["resolved risk-controller findings"],
                repair_target="compute",
                repair_class="missing_risk",
            )
        if requires_compliance_guard(state):
            if not compliance_results:
                return ReviewResult(
                    verdict="revise",
                    reasoning="Final options answer requires a compliance-guard pass before acceptance.",
                    missing_dimensions=["compliance-guard validation"],
                    repair_target="final",
                    repair_class="missing_risk",
                )
            if str(compliance_results[-1].get("verdict", "pass")) != "pass":
                return ReviewResult(
                    verdict="revise",
                    reasoning="Final options answer cannot be accepted while the latest compliance-guard result is unresolved.",
                    missing_dimensions=["resolved compliance findings"],
                    repair_target="final",
                    repair_class="missing_risk",
                )
        if not has_structured_tool:
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer must be grounded in at least one structured tool result.",
                missing_dimensions=["tool-backed strategy analysis"],
                repair_target="compute",
                repair_class="missing_evidence",
            )
        if risk_requirements.get("required_disclosures") and not _has_risk_required_disclosures(
            artifact,
            list(risk_requirements.get("required_disclosures", [])),
        ):
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer is missing one or more risk-controller-required disclosures.",
                missing_dimensions=["required risk disclosures"],
                repair_target="final",
                repair_class="missing_risk",
            )
        gaps = _options_gaps(artifact)
        if gaps:
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer is missing one or more benchmark-critical dimensions.",
                missing_dimensions=gaps,
                repair_target="final",
                repair_class="missing_section",
            )
        if _has_undisclosed_required_assumption(state, artifact):
            return ReviewResult(
                verdict="revise",
                reasoning="Final options answer must disclose material pricing assumptions introduced during compute.",
                missing_dimensions=["disclosed material assumptions"],
                repair_target="final",
                repair_class="missing_risk",
            )

    if is_final and profile == "document_qa":
        if not has_extracted_document_evidence(document_evidence):
            return ReviewResult(
                verdict="revise",
                reasoning="Document QA final answer requires extracted document evidence, not URL discovery alone.",
                missing_dimensions=["extracted document evidence"],
                repair_target="gather",
                repair_class="missing_evidence",
            )

    if is_final and profile_pack.reviewer_dimensions and profile not in {"legal_transactional", "finance_options"}:
        gaps = _keyword_gaps(artifact, profile_pack.reviewer_dimensions)
        if gaps:
            return ReviewResult(
                verdict="revise",
                reasoning="Final answer is incomplete relative to the required profile dimensions.",
                missing_dimensions=gaps,
                repair_target="final",
                repair_class="missing_section",
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
        if should_run_self_reflection(state):
            return "self_reflection"
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
    model_name = get_model_name("reviewer")
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

    if verdict is None and _should_try_llm_review(state, str(review_stage), is_final):
        used_llm = True
        t0 = time.monotonic()
        try:
            verdict, _ = invoke_structured_output(
                "reviewer",
                ReviewResult,
                invocation_messages,
                temperature=0,
                max_tokens=240,
            )
            latency = (time.monotonic() - t0) * 1000
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            verdict = _fallback_review_on_parse_failure(state, artifact, is_final, exc)
    elif verdict is None:
        verdict = ReviewResult(
            verdict="pass",
            reasoning="Deterministic reviewer accepted the artifact without LLM escalation.",
            missing_dimensions=[],
            repair_target=_next_target_for_pass(state),
        )

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
        workpad = _update_repeat_diagnostics(workpad, verdict, artifact)
        if budget and budget.backtrack_exhausted():
            budget.log_budget_exit("backtrack_budget_exhausted", verdict.reasoning)
            workpad = _record_event(workpad, "budget exit -> backtrack cap exhausted")
            return {
                "review_feedback": verdict.model_dump(),
                "solver_stage": "COMPLETE",
                "workpad": workpad,
            }
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
            "reflection_feedback": None,
            "solver_stage": "REVISE",
        }

    if verdict.verdict == "revise":
        if budget:
            budget.record_revise()
        workpad = _append_review_result(workpad, verdict, str(review_stage), is_final)
        workpad = _update_repeat_diagnostics(workpad, verdict, artifact)

        if (
            state.get("task_profile") == "finance_quant"
            and is_final
            and verdict.repair_class in {"wrapper_only", "scalar_only"}
            and int(workpad.get("repeat_count", 1)) >= 2
        ):
            repaired = deterministic_quant_final_answer(state)
            if repaired:
                if budget:
                    budget.log_budget_exit("repeat_review_loop", verdict.reasoning)
                workpad = _record_event(workpad, "repeat-review-loop -> deterministic final")
                workpad["review_ready"] = False
                workpad["review_stage"] = None
                logger.info("[Step %s] reviewer -> deterministic terminal repair", step)
                return {
                    "messages": [AIMessage(content=repaired)],
                    "review_feedback": None,
                    "reflection_feedback": None,
                    "solver_stage": "COMPLETE",
                    "workpad": workpad,
                }

        repeat_limit = 2 if is_final and state.get("task_profile") == "legal_transactional" else 3
        if is_final and int(workpad.get("repeat_count", 1)) >= repeat_limit:
            if budget:
                budget.log_budget_exit("repeat_review_loop", verdict.reasoning)
            workpad = _record_event(workpad, "repeat-review-loop -> terminate current finalization path")
            workpad["review_ready"] = False
            logger.info("[Step %s] reviewer -> terminate after repeated unchanged final review", step)
            return {
                "review_feedback": verdict.model_dump(),
                "reflection_feedback": None,
                "solver_stage": "COMPLETE",
                "workpad": workpad,
            }

        if budget and budget.revise_exhausted():
            repaired = None
            if state.get("task_profile") == "finance_quant" and is_final:
                repaired = deterministic_quant_final_answer(state)
            budget.log_budget_exit("revise_budget_exhausted", verdict.reasoning)
            workpad = _record_event(workpad, "budget exit -> revise cap exhausted")
            workpad["review_ready"] = False
            if repaired:
                return {
                    "messages": [AIMessage(content=repaired)],
                    "review_feedback": None,
                    "reflection_feedback": None,
                    "solver_stage": "COMPLETE",
                    "workpad": workpad,
                }
            return {
                "review_feedback": verdict.model_dump(),
                "reflection_feedback": None,
                "solver_stage": "COMPLETE",
                "workpad": workpad,
            }

        workpad = _record_event(workpad, f"REVISE: {verdict.reasoning}")
        workpad["review_ready"] = False
        logger.info("[Step %s] reviewer -> REVISE missing=%s", step, verdict.missing_dimensions)
        return {
            "review_feedback": verdict.model_dump(),
            "reflection_feedback": None,
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
    workpad = _update_repeat_diagnostics(workpad, verdict, artifact)
    workpad = _record_event(workpad, f"PASS -> {next_stage}")
    workpad["review_ready"] = False
    workpad["review_stage"] = None
    logger.info("[Step %s] reviewer -> PASS next=%s", step, next_stage)
    return {
        "review_feedback": None,
        "reflection_feedback": None,
        "solver_stage": next_stage,
        "checkpoint_stack": checkpoint_stack,
        "workpad": workpad,
    }
