"""
Stage policy and checkpoint helpers.
"""

from __future__ import annotations

from typing import Any

from engine.agent.contracts import ExecutionTemplate


def initial_solver_stage(task_profile: str, capability_flags: list[str], evidence_pack: dict[str, Any]) -> str:
    flags = set(capability_flags)
    if evidence_pack.get("citations") and "needs_files" in flags:
        return "GATHER"
    if task_profile in {"document_qa", "external_retrieval"} and evidence_pack.get("citations"):
        return "GATHER"
    if task_profile == "external_retrieval":
        return "GATHER"
    if task_profile in {"finance_quant", "finance_options"} or "needs_math" in flags or "needs_options_engine" in flags:
        return "COMPUTE"
    return "SYNTHESIZE"


def infer_task_complexity_tier(
    template: dict[str, Any] | ExecutionTemplate | None,
    task_profile: str,
    capability_flags: list[str],
    evidence_pack: dict[str, Any],
) -> str:
    if isinstance(template, dict):
        template_id = str(template.get("template_id", ""))
    elif isinstance(template, ExecutionTemplate):
        template_id = template.template_id
    else:
        template_id = ""

    flags = set(capability_flags)
    if template_id == "quant_inline_exact":
        return "simple_exact"
    if template_id in {"legal_reasoning_only", "legal_with_document_evidence", "live_retrieval"}:
        return "complex_qualitative"
    if task_profile == "legal_transactional":
        return "complex_qualitative"
    if {"needs_legal_reasoning", "needs_live_data"} <= flags and evidence_pack.get("document_evidence"):
        return "complex_qualitative"
    return "structured_analysis"


def initial_stage_for_template(
    template: dict[str, Any] | ExecutionTemplate | None,
    task_profile: str,
    capability_flags: list[str],
    evidence_pack: dict[str, Any],
) -> str:
    if isinstance(template, dict):
        default_stage = str(template.get("default_initial_stage", "SYNTHESIZE"))
        allowed_stages = set(template.get("allowed_stages", []))
        template_id = str(template.get("template_id", ""))
    elif isinstance(template, ExecutionTemplate):
        default_stage = template.default_initial_stage
        allowed_stages = set(template.allowed_stages)
        template_id = template.template_id
    else:
        default_stage = "SYNTHESIZE"
        allowed_stages = set()
        template_id = ""

    flags = set(capability_flags)
    if template_id in {
        "legal_with_document_evidence",
        "document_qa",
        "live_retrieval",
        "regulated_actionable_finance",
        "equity_research_report",
        "event_driven_finance",
    }:
        return "GATHER"
    if template_id in {"options_tool_backed", "quant_inline_exact"}:
        return "COMPUTE"
    if template_id == "portfolio_risk_review":
        return "GATHER" if ("needs_live_data" in flags or "needs_files" in flags) else "COMPUTE"
    if template_id == "quant_with_tool_compute":
        if "needs_live_data" in flags or "needs_files" in flags:
            return "GATHER"
        if evidence_pack.get("citations") and ("needs_files" in flags or task_profile in {"document_qa", "external_retrieval"}):
            return "GATHER"
        return "COMPUTE"
    if default_stage in allowed_stages or not allowed_stages:
        return default_stage
    return initial_solver_stage(task_profile, capability_flags, evidence_pack)


def selective_checkpoint_policy(template: dict[str, Any] | ExecutionTemplate | None) -> dict[str, Any]:
    if isinstance(template, dict):
        template_id = str(template.get("template_id", ""))
    elif isinstance(template, ExecutionTemplate):
        template_id = template.template_id
    else:
        template_id = ""

    if template_id == "quant_with_tool_compute":
        return {
            "enabled": True,
            "checkpoint_stages": {"GATHER", "COMPUTE"},
            "backtrack_stages": {"GATHER", "COMPUTE"},
        }
    if template_id == "options_tool_backed":
        return {
            "enabled": True,
            "checkpoint_stages": {"COMPUTE"},
            "backtrack_stages": {"COMPUTE"},
        }
    if template_id == "portfolio_risk_review":
        return {
            "enabled": True,
            "checkpoint_stages": {"COMPUTE"},
            "backtrack_stages": {"COMPUTE"},
        }
    if template_id in {"equity_research_report", "event_driven_finance", "regulated_actionable_finance"}:
        return {
            "enabled": True,
            "checkpoint_stages": {"GATHER", "COMPUTE"},
            "backtrack_stages": {"GATHER", "COMPUTE"},
        }
    if template_id in {"document_qa", "legal_with_document_evidence"}:
        return {
            "enabled": True,
            "checkpoint_stages": {"GATHER"},
            "backtrack_stages": {"GATHER"},
        }
    return {
        "enabled": False,
        "checkpoint_stages": set(),
        "backtrack_stages": set(),
    }


def selective_backtracking_allowed(template: dict[str, Any] | ExecutionTemplate | None, stage: str) -> bool:
    policy = selective_checkpoint_policy(template)
    return bool(policy["enabled"]) and stage in set(policy["backtrack_stages"])


def should_checkpoint_stage(template: dict[str, Any] | ExecutionTemplate | None, stage: str) -> bool:
    policy = selective_checkpoint_policy(template)
    return bool(policy["enabled"]) and stage in set(policy["checkpoint_stages"])


def stage_is_review_milestone(template: dict[str, Any] | ExecutionTemplate | None, stage: str) -> bool:
    if isinstance(template, dict):
        review_stages = set(template.get("review_stages", []))
    elif isinstance(template, ExecutionTemplate):
        review_stages = set(template.review_stages)
    else:
        review_stages = set()
    return stage in review_stages


def next_stage_after_review(stage: str, review_target: str, verdict: str) -> str:
    if verdict in {"revise", "backtrack"}:
        return "REVISE"
    if stage == "GATHER":
        return "COMPUTE" if review_target != "synthesize" else "SYNTHESIZE"
    if stage == "COMPUTE":
        return "SYNTHESIZE"
    if stage == "SYNTHESIZE":
        return "COMPLETE"
    return "COMPLETE"
