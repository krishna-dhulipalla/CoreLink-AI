"""
Template Selector Node
======================
Chooses one vetted execution template from the static template library.
"""

from __future__ import annotations

import logging

from agent.contracts import AnswerContract, ExecutionTemplate, ProfileDecision
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import latest_human_text, select_execution_template
from agent.state import AgentState
from agent.tracer import get_tracer

logger = logging.getLogger(__name__)


def template_selector(state: AgentState) -> dict:
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

    template: ExecutionTemplate = select_execution_template(task_text, profile_decision, answer_contract)
    workpad = dict(state.get("workpad", {}))
    workpad["execution_template"] = template.model_dump()
    workpad.setdefault("events", []).append(
        {
            "node": "template_selector",
            "action": (
                f"template={template.template_id} "
                f"initial={template.default_initial_stage} "
                f"reviews={','.join(template.review_stages) or 'final_only'}"
            ),
        }
    )

    logger.info(
        "[Step %s] template_selector -> template=%s profile=%s",
        step,
        template.template_id,
        profile_decision.primary_profile,
    )

    tracer = get_tracer()
    if tracer:
        tracer.record("template_selector", {
            "template_id": template.template_id,
            "default_initial_stage": template.default_initial_stage,
            "allowed_stages": template.allowed_stages,
            "review_stages": template.review_stages,
            "tool_policy": template.tool_policy,
        })

    return {
        "execution_template": template.model_dump(),
        "workpad": workpad,
    }
