from langchain_core.messages import HumanMessage

from agent.budget import BudgetTracker
from agent.cost import CostTracker


def make_state(
    prompt: str,
    *,
    profile_decision: dict | None = None,
    task_profile: str = "general",
    capability_flags: list[str] | None = None,
    ambiguity_flags: list[str] | None = None,
    execution_template: dict | None = None,
    answer_contract: dict | None = None,
    evidence_pack: dict | None = None,
    assumption_ledger: list[dict] | None = None,
    provenance_map: dict | None = None,
    solver_stage: str = "PLAN",
    workpad: dict | None = None,
    pending_tool_call: dict | None = None,
    last_tool_result: dict | None = None,
    risk_feedback: dict | None = None,
    review_feedback: dict | None = None,
    checkpoint_stack: list[dict] | None = None,
):
    return {
        "messages": [HumanMessage(content=prompt)],
        "profile_decision": profile_decision or {},
        "task_profile": task_profile,
        "capability_flags": capability_flags or [],
        "ambiguity_flags": ambiguity_flags or [],
        "execution_template": execution_template or {},
        "answer_contract": answer_contract or {},
        "evidence_pack": evidence_pack or {},
        "assumption_ledger": assumption_ledger or [],
        "provenance_map": provenance_map or {},
        "solver_stage": solver_stage,
        "workpad": workpad or {"events": [], "stage_outputs": {}, "tool_results": []},
        "pending_tool_call": pending_tool_call,
        "last_tool_result": last_tool_result,
        "risk_feedback": risk_feedback,
        "review_feedback": review_feedback,
        "checkpoint_stack": checkpoint_stack or [],
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "budget_tracker": BudgetTracker(),
        "cost_tracker": CostTracker(),
        "memory_store": None,
    }
