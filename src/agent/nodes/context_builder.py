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
    apply_profile_contract_rules,
    build_evidence_pack,
    initial_stage_for_template,
    latest_human_text,
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

    evidence: EvidencePack = build_evidence_pack(
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
                f"stage={next_stage} entities={len(evidence.entities)} files={len(evidence.file_refs)}"
            ),
        }
    )
    checkpoint_stack = list(state.get("checkpoint_stack", []))
    if not checkpoint_stack:
        checkpoint_stack.append(
            {
                "messages": list(state.get("messages", [])),
                "evidence_pack": evidence.model_dump(),
                "workpad": workpad,
                "solver_stage": next_stage,
                "last_tool_result": None,
            }
        )

    logger.info(
        "[Step %s] context_builder -> profile=%s stage=%s entities=%s files=%s",
        step,
        task_profile,
        next_stage,
        evidence.entities,
        len(evidence.file_refs),
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
        "checkpoint_stack": checkpoint_stack,
    }
