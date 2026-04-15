"""
Offline Context-Pack Curation
=============================
Deterministic helpers for turning staged-runtime traces into store-only
curation signals and stable offline update proposals.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from engine.agent.memory.schema import CurationSignal, infer_memory_family, normalize_memory_text, task_signature


def _slug(text: str) -> str:
    normalized = normalize_memory_text(text, max_len=120).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized or "unknown"


def _final_answer_text(state: dict[str, Any], workpad: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if content:
            return normalize_memory_text(str(content), max_len=2000).lower()
    return normalize_memory_text(str(workpad.get("draft_answer", "")), max_len=2000).lower()


def _assumption_is_disclosed(record: dict[str, Any], final_answer_text: str) -> bool:
    key = str(record.get("key", "")).lower()
    if not final_answer_text:
        return False
    if key == "spot_price":
        return "spot" in final_answer_text and "assum" in final_answer_text
    assumption_text = normalize_memory_text(str(record.get("assumption", "")), max_len=120).lower()
    return bool(assumption_text and assumption_text[:24] in final_answer_text)


def build_curation_signals(state: dict[str, Any], task_text: str, workpad: dict[str, Any]) -> list[CurationSignal]:
    task_profile = str(state.get("task_profile", "general"))
    task_family = infer_memory_family(task_profile, task_text)
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    signature = task_signature(task_text)
    final_answer_text = _final_answer_text(state, workpad)
    signals: list[CurationSignal] = []

    for result in workpad.get("review_results", []):
        if not isinstance(result, dict):
            continue
        review_stage = str(result.get("review_stage", ""))
        verdict = str(result.get("verdict", "revise"))
        repair_target = str(result.get("repair_target", "final"))
        reasoning = normalize_memory_text(str(result.get("reasoning", "")), max_len=220)
        missing_dimensions = [str(item) for item in result.get("missing_dimensions", []) if str(item).strip()]

        for dimension in missing_dimensions:
            signals.append(
                CurationSignal(
                    task_signature=signature,
                    task_profile=task_profile,
                    task_family=task_family,
                    template_id=template_id,
                    signal_type="missing_dimension",
                    signal_key=_slug(dimension),
                    summary=f"{review_stage or 'review'} missing dimension: {dimension}",
                    stage=review_stage,
                    success=False,
                    metadata={
                        "dimension": dimension,
                        "repair_target": repair_target,
                        "verdict": verdict,
                        "is_final": bool(result.get("is_final")),
                    },
                )
            )

        lower_reasoning = reasoning.lower()
        if verdict != "pass" and (repair_target == "gather" or "evidence" in lower_reasoning or "source" in lower_reasoning):
            signals.append(
                CurationSignal(
                    task_signature=signature,
                    task_profile=task_profile,
                    task_family=task_family,
                    template_id=template_id,
                    signal_type="missing_evidence",
                    signal_key=_slug(review_stage or repair_target or "evidence_gap"),
                    summary=reasoning or "Reviewer requested additional evidence.",
                    stage=review_stage,
                    success=False,
                    metadata={
                        "repair_target": repair_target,
                        "verdict": verdict,
                    },
                )
            )

        if verdict == "backtrack":
            signals.append(
                CurationSignal(
                    task_signature=signature,
                    task_profile=task_profile,
                    task_family=task_family,
                    template_id=template_id,
                    signal_type="backtrack_pattern",
                    signal_key=_slug(f"{review_stage}_{repair_target}"),
                    summary=reasoning or "Reviewer forced a local backtrack.",
                    stage=review_stage,
                    success=False,
                    metadata={
                        "repair_target": repair_target,
                    },
                )
            )

    for record in state.get("assumption_ledger", []):
        if not isinstance(record, dict):
            continue
        if not record.get("requires_user_visible_disclosure"):
            continue
        assumption_key = str(record.get("key") or _slug(str(record.get("assumption", ""))))
        disclosed = _assumption_is_disclosed(record, final_answer_text)
        signals.append(
            CurationSignal(
                task_signature=signature,
                task_profile=task_profile,
                task_family=task_family,
                template_id=template_id,
                signal_type="assumption_issue",
                signal_key=_slug(assumption_key),
                summary=normalize_memory_text(str(record.get("assumption", "")), max_len=220),
                stage="COMPUTE",
                success=disclosed or str(record.get("review_status", "pending")) in {"accepted", "disclosed"},
                metadata={
                    "source": record.get("source", ""),
                    "confidence": record.get("confidence", ""),
                    "requires_user_visible_disclosure": True,
                    "review_status": record.get("review_status", "pending"),
                    "disclosed_in_final_answer": disclosed,
                },
            )
        )

    for result in workpad.get("tool_results", []):
        if not isinstance(result, dict):
            continue
        errors = [str(error) for error in result.get("errors", []) if str(error).strip()]
        if not errors:
            continue
        source = result.get("source", {}) if isinstance(result.get("source", {}), dict) else {}
        tool_name = str(source.get("tool", result.get("type", "unknown")))
        stage = str(source.get("solver_stage", ""))
        first_error = normalize_memory_text(errors[0], max_len=220)
        signals.append(
            CurationSignal(
                task_signature=signature,
                task_profile=task_profile,
                task_family=task_family,
                template_id=template_id,
                signal_type="tool_failure",
                signal_key=_slug(f"{tool_name}_{first_error}"),
                summary=f"{tool_name}: {first_error}",
                stage=stage,
                success=False,
                metadata={
                    "tool_name": tool_name,
                    "error_count": len(errors),
                    "result_type": result.get("type", "unknown"),
                },
            )
        )

    return signals


def summarize_curation_signals(
    signals: list[dict[str, Any]] | list[CurationSignal],
    *,
    min_count: int = 2,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "failure_count": 0, "examples": [], "task_family": "general"}
    )

    for entry in signals:
        payload = entry.model_dump() if isinstance(entry, CurationSignal) else dict(entry)
        key = (
            str(payload.get("task_profile", "general")),
            str(payload.get("template_id", "")),
            str(payload.get("signal_type", "")),
            str(payload.get("signal_key", "")),
        )
        grouped[key]["count"] += int(payload.get("count_hint", 1) or 1)
        if not bool(payload.get("success", False)):
            grouped[key]["failure_count"] += int(payload.get("count_hint", 1) or 1)
        grouped[key]["task_family"] = str(payload.get("task_family", "general"))
        example = {
            "summary": str(payload.get("summary", "")),
            "stage": str(payload.get("stage", "")),
            "metadata": dict(payload.get("metadata", {})),
        }
        if example not in grouped[key]["examples"]:
            grouped[key]["examples"].append(example)

    observations: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []

    for key in sorted(grouped.keys()):
        task_profile, template_id, signal_type, signal_key = key
        bucket = grouped[key]
        count = bucket["count"]
        examples = bucket["examples"][:3]
        observations.append(
            {
                "task_profile": task_profile,
                "task_family": bucket["task_family"],
                "template_id": template_id,
                "signal_type": signal_type,
                "signal_key": signal_key,
                "count": count,
                "failure_count": bucket["failure_count"],
                "examples": examples,
            }
        )
        if bucket["failure_count"] < min_count:
            continue

        if signal_type == "missing_dimension":
            target = "reviewer_dimensions"
            action = f"Tighten profile-pack coverage for recurring missing dimension '{signal_key}'."
        elif signal_type == "assumption_issue":
            target = "content_rules"
            action = f"Add or sharpen an explicit disclosure rule for recurring assumption '{signal_key}'."
        elif signal_type == "missing_evidence":
            target = "template_policy_defaults"
            action = f"Review gather-stage evidence policy for recurring gap '{signal_key}'."
        elif signal_type == "tool_failure":
            target = "template_policy_defaults"
            action = f"Review tool allowlist or normalization for recurring tool failure '{signal_key}'."
        else:
            target = "template_policy_defaults"
            action = f"Review selective checkpoint/backtrack policy for recurring pattern '{signal_key}'."

        recommendations.append(
            {
                "task_profile": task_profile,
                "template_id": template_id,
                "task_family": bucket["task_family"],
                "target": target,
                "signal_type": signal_type,
                "signal_key": signal_key,
                "count": count,
                "failure_count": bucket["failure_count"],
                "proposed_action": action,
                "examples": examples,
            }
        )

    return {
        "observation_count": len(observations),
        "recommendation_count": len(recommendations),
        "observations": observations,
        "recommendations": recommendations,
    }


def summarize_curation_as_json(signals: list[dict[str, Any]] | list[CurationSignal], *, min_count: int = 2) -> str:
    return json.dumps(summarize_curation_signals(signals, min_count=min_count), indent=2, ensure_ascii=True)
