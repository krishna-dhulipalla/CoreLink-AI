import asyncio
import json

import agent.graph as graph_module
from langchain_core.messages import AIMessage

from agent.nodes.intake import intake
from agent.v4.capabilities import V4_BUILTIN_LEGAL_TOOLS, build_capability_registry
from agent.v4.nodes import (
    context_curator,
    fast_path_gate,
    make_capability_resolver,
    make_executor,
    reviewer,
    self_reflection,
    task_planner,
)
from staged_test_utils import make_state
from tools import CALCULATOR_TOOL, SEARCH_TOOL


def test_build_agent_graph_selects_v4_when_env_enabled(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_VERSION", "v4")
    monkeypatch.setattr("agent.v4.graph.build_v4_agent_graph", lambda external_tools=None: "v4_graph")

    assert graph_module.build_agent_graph() == "v4_graph"


def _exact_quant_prompt() -> str:
    return """
    ### <Formula List>
    Financial Leverage Effect = (ROE - ROA) / ROA

    ### <Related Data>
    | Stock Name | Return on Equity (ROE) (2024 Annual Report) (%) |
    |---|---|
    | China Overseas Grand Oceans Group | 3.0433 % |

    | Stock Name | Return on Assets (ROA) (2024 Annual Report) (%) |
    |---|---|
    | China Overseas Grand Oceans Group | 1.5790 % |

    ### <User Question>
    Please calculate: What is the Financial Leverage Effect (2024 Annual Report) for China Overseas Grand Oceans Group?

    ### Output Format
    {"answer": <value>}
    """


def test_v4_exact_quant_fast_path_curates_without_duplicate_solver_payload():
    state = make_state(_exact_quant_prompt())
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *V4_BUILTIN_LEGAL_TOOLS]))
    state.update(resolver(state))
    result = context_curator(state)

    assert result["curated_context"]["objective"]
    assert result["evidence_pack"].keys() == {"curated_context", "source_bundle_summary"}
    assert "constraints" not in json.dumps(result["evidence_pack"], ensure_ascii=True)
    assert "formulas" not in result["evidence_pack"]
    assert result["workpad"]["task_complexity_tier"] == "simple_exact"
    assert any(fact["type"] == "formula" for fact in result["curated_context"]["facts_in_use"])
    assert any(fact["type"] == "table_rows" for fact in result["curated_context"]["facts_in_use"])


def test_v4_legal_planner_binds_legal_capability_tools_not_calculator_only():
    prompt = (
        "target company we're acquiring has some clean IP but also regulatory compliance gaps in EU and US. "
        "their board wants stock consideration for tax reasons but we can't risk inheriting the compliance liabilities. "
        "deal size is ~$500M, what structure options do we have that could work for both sides? Need to move quickly here."
    )
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *V4_BUILTIN_LEGAL_TOOLS]))
    result = resolver(state)

    assert result["tool_plan"]["selected_tools"]
    assert "calculator" not in result["tool_plan"]["selected_tools"]
    assert "transaction_structure_checklist" in result["tool_plan"]["selected_tools"]
    assert "regulatory_execution_checklist" in result["tool_plan"]["selected_tools"]
    assert "tax_structure_checklist" in result["tool_plan"]["selected_tools"]
    assert result["execution_template"]["template_id"] == "v4_advisory_analysis"


def test_v4_executor_returns_deterministic_exact_quant_answer():
    state = make_state(_exact_quant_prompt())
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *V4_BUILTIN_LEGAL_TOOLS]))
    state.update(resolver(state))
    state.update(context_curator(state))

    executor = make_executor(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *V4_BUILTIN_LEGAL_TOOLS]))
    result = asyncio.run(executor(state))

    assert result["messages"]
    assert str(result["messages"][0].content).startswith('{"answer":')
    assert result["workpad"]["v4_review_ready"] is True


def test_v4_reviewer_revises_legal_once_then_stops_cleanly():
    prompt = (
        "Need acquisition structure options with stock consideration, liability protection, and cross-border compliance handling."
    )
    state = make_state(
        prompt,
        task_profile="legal_transactional",
        task_intent={
            "task_family": "legal_transactional",
            "execution_mode": "advisory_analysis",
            "complexity_tier": "complex_qualitative",
            "tool_families_needed": ["transaction_structure_checklist"],
            "evidence_strategy": "compact_prompt",
            "review_mode": "qualitative_advisory",
            "completion_mode": "advisory_memo",
            "routing_rationale": "",
            "confidence": 0.9,
            "planner_source": "heuristic",
        },
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": ""},
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "v4_review_ready": True},
    )
    state["messages"].append(AIMessage(content="Recommendation: use an asset deal. Tax: stock may help deferral. Liability: use indemnities. Next steps: move quickly."))

    first = reviewer(state)
    assert first["solver_stage"] == "REVISE"
    assert "liability allocation" in " ".join(first["review_feedback"]["missing_dimensions"]).lower()

    state["execution_journal"]["revision_count"] = 1
    second = reviewer(state)
    assert second["solver_stage"] == "COMPLETE"
    assert second["quality_report"]["verdict"] == "fail"


def test_v4_self_reflection_requests_one_extra_legal_deepen_pass():
    state = make_state(
        "Advise on acquisition structure.",
        task_profile="legal_transactional",
        task_intent={
            "task_family": "legal_transactional",
            "execution_mode": "advisory_analysis",
            "complexity_tier": "complex_qualitative",
            "tool_families_needed": [],
            "evidence_strategy": "compact_prompt",
            "review_mode": "qualitative_advisory",
            "completion_mode": "advisory_memo",
            "routing_rationale": "",
            "confidence": 0.9,
            "planner_source": "heuristic",
        },
        quality_report={"verdict": "pass", "reasoning": "", "missing_dimensions": [], "targeted_fix_prompt": "", "score": 0.9},
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": ""},
        workpad={"events": [], "stage_outputs": {}, "tool_results": []},
    )
    state["messages"].append(AIMessage(content="Recommendation: pursue an asset acquisition. Liability: lower inherited exposure."))

    result = self_reflection(state)

    assert result["solver_stage"] == "REVISE"
    assert "execution-specific next steps" in result["review_feedback"]["reasoning"].lower()
