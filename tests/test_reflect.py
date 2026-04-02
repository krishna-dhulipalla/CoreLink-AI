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
        "execution_template": {
            "template_id": "quant_inline_exact",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["calculator"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        "answer_contract": {"format": "json", "requires_adapter": True},
        "evidence_pack": {},
        "assumption_ledger": [
            {
                "key": "spot_price",
                "assumption": "Spot price must be disclosed if introduced later.",
                "source": "context_curator_open_question",
                "confidence": "low",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            }
        ],
        "provenance_map": {
            "prompt_facts.roe": {
                "source_class": "prompt",
                "source_id": "user_prompt",
                "extraction_method": "inline_extraction",
                "tool_name": None,
            }
        },
        "solver_stage": "COMPLETE",
        "workpad": {
            "events": [
                {"node": "intake", "action": "Detected output format=json"},
                {"node": "task_planner", "action": "family=finance_quant flags=needs_math"},
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
    assert stats["curation_memory"] == 1
    signal = store.fetch_curation_signals(limit=10)[0]
    assert signal["signal_type"] == "assumption_issue"


def test_reflect_skips_persistence_when_memory_is_disabled(monkeypatch):
    monkeypatch.delenv("ENABLE_AGENT_MEMORY", raising=False)

    state = {
        "messages": [HumanMessage(content="OfficeQA benchmark task.")],
        "task_profile": "document_qa",
        "capability_flags": [],
        "ambiguity_flags": [],
        "execution_template": {"template_id": "document_grounded_analysis"},
        "answer_contract": {"format": "xml", "requires_adapter": True},
        "assumption_ledger": [],
        "provenance_map": {},
        "solver_stage": "COMPLETE",
        "workpad": {"events": [], "stage_history": [], "tool_results": [], "review_results": []},
        "cost_tracker": CostTracker(),
        "memory_store": None,
    }

    update = reflect(state)

    assert update["workpad"]["events"][-1]["node"] == "reflect"
