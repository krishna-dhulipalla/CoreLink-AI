from langchain_core.messages import HumanMessage

from agent.budget import BudgetTracker
from agent.cost import CostTracker
from agent.memory.store import MemoryStore
from agent.nodes.reflect import reflect


def test_reflect_persists_run_tool_and_review_memory():
    store = MemoryStore(":memory:")
    tracker = CostTracker()
    budget = BudgetTracker()

    state = {
        "messages": [HumanMessage(content="Calculate ROE from the inline data.")],
        "profile_decision": {
            "primary_profile": "finance_quant",
            "capability_flags": ["needs_math"],
            "ambiguity_flags": [],
            "needs_external_data": False,
            "needs_output_adapter": True,
        },
        "task_profile": "finance_quant",
        "capability_flags": ["needs_math"],
        "ambiguity_flags": [],
        "answer_contract": {"format": "json", "requires_adapter": True},
        "evidence_pack": {},
        "solver_stage": "COMPLETE",
        "workpad": {
            "events": [
                {"node": "intake", "action": "Detected output format=json"},
                {"node": "task_profiler", "action": "profile=finance_quant flags=needs_math"},
            ],
            "stage_history": ["PLAN", "COMPUTE", "SYNTHESIZE"],
            "tool_results": [
                {
                    "type": "calculator",
                    "facts": {"result": 0.9274},
                    "assumptions": {"expression": "(3.0433-1.579)/1.579"},
                    "source": {"tool": "calculator", "solver_stage": "COMPUTE"},
                    "errors": [],
                }
            ],
            "review_results": [
                {
                    "review_stage": "SYNTHESIZE",
                    "is_final": True,
                    "verdict": "pass",
                    "reasoning": "Complete answer.",
                    "missing_dimensions": [],
                    "repair_target": "final",
                }
            ],
        },
        "pending_tool_call": None,
        "last_tool_result": None,
        "review_feedback": None,
        "checkpoint_stack": [],
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "budget_tracker": budget,
        "cost_tracker": tracker,
        "memory_store": store,
    }

    update = reflect(state)
    stats = store.stats()

    assert update["workpad"]["events"][-1]["node"] == "reflect"
    assert stats["run_memory"] == 1
    assert stats["tool_memory"] == 1
    assert stats["review_memory"] == 1
