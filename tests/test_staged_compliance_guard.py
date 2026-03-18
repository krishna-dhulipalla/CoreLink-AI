from langchain_core.messages import AIMessage

from agent.nodes.compliance_guard import compliance_guard, requires_compliance_guard
from staged_test_utils import make_state


def test_compliance_guard_blocks_naked_options_for_retirement_account():
    state = make_state(
        "This is a retirement account. Defined-risk only. No naked options. Keep position risk to 2% of capital.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        evidence_pack={
            "policy_context": {
                "action_orientation": True,
                "defined_risk_only": True,
                "no_naked_options": True,
                "retail_or_retirement_account": True,
                "max_position_risk_pct": 2.0,
                "requires_recommendation_class": True,
            }
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(
        AIMessage(
            content=(
                "**Recommendation**\nBe a net seller of options.\n\n"
                "**Primary Strategy**\nShort straddle with credit premium.\n\n"
                "**Risk Management**\nUse 1-2% position sizing."
            )
        )
    )

    assert requires_compliance_guard(state) is True
    result = compliance_guard(state)

    assert result["solver_stage"] == "REVISE"
    assert result["compliance_feedback"]["verdict"] == "blocked"
    assert "NAKED_OPTIONS_PROHIBITED" in result["compliance_feedback"]["violation_codes"]


def test_compliance_guard_passes_defined_risk_recommendation():
    state = make_state(
        "Defined-risk only. No naked options.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        evidence_pack={
            "policy_context": {
                "action_orientation": True,
                "defined_risk_only": True,
                "no_naked_options": True,
                "requires_recommendation_class": True,
            }
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(
        AIMessage(
            content=(
                "**Recommendation**\nUse a defined-risk iron condor.\n\n"
                "**Primary Strategy**\nDefined-risk iron condor with capped downside.\n\n"
                "**Risk Management**\nKeep 1-2% position sizing.\n\n"
                "Recommendation class: scenario_dependent_recommendation."
            )
        )
    )

    result = compliance_guard(state)

    assert result["compliance_feedback"] is None
    assert result["workpad"]["compliance_results"][-1]["verdict"] == "pass"
