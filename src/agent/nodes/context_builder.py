"""
Context Builder Node
====================
Builds a compact typed evidence pack before the solver starts acting.
"""

from __future__ import annotations

import logging

from agent.contracts import AnswerContract, EvidencePack, ExecutionTemplate, ProfileDecision
from agent.profile_packs import get_profile_pack
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import (
    artifact_checkpoint_from_state,
    apply_profile_contract_rules,
    build_evidence_pack,
    infer_task_complexity_tier,
    initial_stage_for_template,
    latest_human_text,
    selective_checkpoint_policy,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)


def context_builder(state: AgentState) -> dict:
    step = increment_runtime_step()
    task_text = latest_human_text(state["messages"])
    answer_contract = AnswerContract.model_validate(state.get("answer_contract", {}))
    profile_decision = ProfileDecision.model_validate(
        state.get("profile_decision", {}) or {
            "primary_profile": state.get("task_profile", "general"),
            "capability_flags": state.get("capability_flags", []),
            "ambiguity_flags": state.get("ambiguity_flags", []),
            "needs_external_data": False,
            "needs_output_adapter": bool(answer_contract.requires_adapter),
        }
    )
    task_profile = profile_decision.primary_profile
    capability_flags = list(profile_decision.capability_flags)
    ambiguity_flags = list(profile_decision.ambiguity_flags)
    execution_template = ExecutionTemplate.model_validate(
        state.get("execution_template", {}) or {
            "template_id": "legal_reasoning_only",
            "description": "Fallback reasoning-only template.",
            "allowed_stages": ["SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "SYNTHESIZE",
            "allowed_tool_names": ["calculator"],
            "review_stages": ["SYNTHESIZE"],
            "review_cadence": "final_only",
            "answer_focus": [],
            "ambiguity_safe": True,
        }
    )
    profile_pack = get_profile_pack(task_profile)
    merged_contract = apply_profile_contract_rules(answer_contract, task_profile)

    evidence, assumption_ledger, provenance_map = build_evidence_pack(
        task_text=task_text,
        answer_contract=merged_contract,
        task_profile=task_profile,
        capability_flags=capability_flags,
        ambiguity_flags=ambiguity_flags,
    )

    workpad = dict(state.get("workpad", {}))
    workpad["profile_decision"] = profile_decision.model_dump()
    workpad["execution_template"] = execution_template.model_dump()
    workpad["profile_pack"] = profile_pack.model_dump()
    complexity_tier = infer_task_complexity_tier(
        execution_template,
        task_profile,
        capability_flags,
        evidence.model_dump(),
    )
    workpad["task_complexity_tier"] = complexity_tier
    workpad.setdefault("stage_history", [])
    workpad.setdefault("events", [])
    next_stage = initial_stage_for_template(
        execution_template,
        task_profile,
        capability_flags,
        evidence.model_dump(),
    )
    workpad["stage_history"].append(next_stage)
    workpad["events"].append(
        {
            "node": "context_builder",
            "action": (
                f"template={execution_template.template_id} "
                f"tier={workpad['task_complexity_tier']} "
                f"stage={next_stage} entities={len(evidence.entities)} "
                f"citations={len(evidence.citations)} documents={len(evidence.document_evidence)}"
            ),
        }
    )
    checkpoint_stack = list(state.get("checkpoint_stack", []))
    budget = state.get("budget_tracker")
    if budget is not None:
        budget.configure(complexity_tier=complexity_tier, template_id=execution_template.template_id)
    policy = selective_checkpoint_policy(execution_template)
    if not checkpoint_stack and policy["enabled"]:
        checkpoint_stack.append(
            artifact_checkpoint_from_state(
                {
                    **state,
                    "evidence_pack": evidence.model_dump(),
                    "assumption_ledger": assumption_ledger,
                    "provenance_map": provenance_map,
                    "workpad": workpad,
                    "execution_template": execution_template.model_dump(),
                    "last_tool_result": None,
                },
                reason="baseline_artifacts",
                stage=next_stage,
            )
        )

    logger.info(
        "[Step %s] context_builder -> profile=%s stage=%s entities=%s citations=%s documents=%s",
        step,
        task_profile,
        next_stage,
        evidence.entities,
        len(evidence.citations),
        len(evidence.document_evidence),
    )

    return {
        "evidence_pack": evidence.model_dump(),
        "solver_stage": next_stage,
        "workpad": workpad,
        "answer_contract": merged_contract.model_dump(),
        "task_profile": task_profile,
        "capability_flags": capability_flags,
        "ambiguity_flags": ambiguity_flags,
        "execution_template": execution_template.model_dump(),
        "assumption_ledger": assumption_ledger,
        "provenance_map": provenance_map,
        "checkpoint_stack": checkpoint_stack,
    }
