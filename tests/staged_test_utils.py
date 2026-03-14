from langchain_core.messages import HumanMessage

from agent.budget import BudgetTracker
from agent.cost import CostTracker


def make_state(
    prompt: str,
    *,
    task_profile: str = "general",
    capability_flags: list[str] | None = None,
    answer_contract: dict | None = None,
    evidence_pack: dict | None = None,
    solver_stage: str = "PLAN",
    workpad: dict | None = None,
    pending_tool_call: dict | None = None,
    last_tool_result: dict | None = None,
    review_feedback: dict | None = None,
):
    return {
        "messages": [HumanMessage(content=prompt)],
        "task_profile": task_profile,
        "capability_flags": capability_flags or [],
        "answer_contract": answer_contract or {},
        "evidence_pack": evidence_pack or {},
        "solver_stage": solver_stage,
        "workpad": workpad or {"events": [], "stage_outputs": {}, "tool_results": []},
        "pending_tool_call": pending_tool_call,
        "last_tool_result": last_tool_result,
        "review_feedback": review_feedback,
        "checkpoint_stack": [],
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "budget_tracker": BudgetTracker(),
        "cost_tracker": CostTracker(),
        "memory_store": None,
    }
