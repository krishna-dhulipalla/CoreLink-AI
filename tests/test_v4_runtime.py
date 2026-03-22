import asyncio
import json

import agent.graph as graph_module
from langchain_core.messages import AIMessage

from agent.budget import BudgetTracker
from agent.nodes.intake import intake
from agent.tracer import RunTracer
from agent.v4.capabilities import V4_BUILTIN_LEGAL_TOOLS, build_capability_registry
from agent.v4.context import solver_context_block
from agent.v4.nodes import (
    context_curator,
    fast_path_gate,
    make_capability_resolver,
    make_executor,
    reviewer,
    route_from_reviewer,
    self_reflection,
    task_planner,
)
from agent.v4.v4_prompts import build_revision_prompt
from staged_test_utils import make_state
from tools import CALCULATOR_TOOL, SEARCH_TOOL


class _FakeModel:
    def __init__(self, response: AIMessage, captured: list | None = None):
        self._response = response
        self._captured = captured if captured is not None else []

    def invoke(self, messages):
        self._captured.append(messages)
        return self._response


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


def test_v4_solver_context_block_removes_redundant_objective_and_tool_query_noise():
    payload = solver_context_block(
        {
            "objective": "repeat me",
            "facts_in_use": [{"type": "jurisdictions", "value": ["EU", "US"]}],
            "open_questions": ["Confirm remediation timeline."],
            "assumptions": [],
            "requested_output": {"format": "text"},
        },
        [
            {
                "type": "legal_playbook_retrieval",
                "facts": {
                    "query": "repeat me",
                    "deal_size_hint": "",
                    "urgency": "",
                    "playbook_points": ["point 1", "point 2"],
                },
            }
        ],
        include_objective=False,
    )

    assert '"objective"' not in payload
    assert '"tool_results"' not in payload
    assert '"tool_findings"' in payload
    assert '"query"' not in payload
    assert "repeat me" not in payload


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


def test_v4_executor_dedupes_legal_prompt_and_uses_higher_legal_completion_budget(monkeypatch):
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
    state.update(resolver(state))
    state.update(context_curator(state))
    state["tool_plan"]["pending_tools"] = []
    state["execution_journal"]["tool_results"] = [
        {
            "type": "legal_playbook_retrieval",
            "facts": {"query": prompt, "playbook_points": ["point 1", "point 2"], "urgency": ""},
        },
        {
            "type": "transaction_structure_checklist",
            "facts": {"structures": ["asset purchase", "reverse triangular merger"], "allocation_mechanics": ["escrow"]},
        },
    ]

    captured: list = []
    monkeypatch.setattr(
        "agent.v4.nodes.ChatOpenAI",
        lambda **kwargs: _FakeModel(AIMessage(content="Structured legal answer with multiple options."), captured),
    )

    executor = make_executor(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *V4_BUILTIN_LEGAL_TOOLS]))
    result = asyncio.run(executor(state))

    assert result["workpad"]["completion_budget"] >= 1600
    prompt_messages = captured[0]
    serialized = json.dumps([str(msg.content) for msg in prompt_messages], ensure_ascii=True)
    assert serialized.count(prompt) == 1
    assert '"query"' not in serialized
    assert '"tool_results"' not in serialized


def test_v4_executor_uses_deterministic_options_final_without_llm(monkeypatch):
    prompt = (
        "META's current IV is 35% while its 30-day historical volatility is 28%. "
        "The IV percentile is 75%. Should you be a net buyer or seller of options? Design a strategy accordingly."
    )
    state = make_state(
        prompt,
        task_profile="finance_options",
        task_intent={
            "task_family": "finance_options",
            "execution_mode": "tool_compute",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["options_strategy_analysis", "options_scenario_analysis"],
            "evidence_strategy": "compact_prompt",
            "review_mode": "tool_compute",
            "completion_mode": "compact_sections",
            "routing_rationale": "",
            "confidence": 0.95,
            "planner_source": "fast_path",
        },
        tool_plan={"tool_families_needed": [], "selected_tools": [], "pending_tools": [], "blocked_families": [], "ace_events": [], "notes": []},
        source_bundle={"task_text": prompt, "focus_query": prompt, "target_period": "", "entities": ["META"], "urls": [], "inline_facts": {"implied_volatility": 0.35, "historical_volatility": 0.28, "iv_percentile": 75.0}, "tables": [], "formulas": []},
        curated_context={"objective": prompt, "facts_in_use": [], "open_questions": [], "assumptions": [], "requested_output": {"format": "text"}, "provenance_summary": {}},
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 23.98,
                        "premium_direction": "credit",
                        "total_delta": -0.073,
                        "total_gamma": -0.026,
                        "total_theta_per_day": 0.439,
                        "total_vega_per_vol_point": -0.683,
                        "max_loss": 9999.0,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "call", "action": "sell", "K": 300.0},
                            {"option_type": "put", "action": "sell", "K": 300.0},
                        ],
                        "reference_price": 300.0,
                    },
                },
                {
                    "type": "scenario_pnl",
                    "facts": {"worst_case_pnl": -22.26, "best_case_pnl": 0.44},
                    "assumptions": {"reference_price": 300.0},
                },
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "final_artifact_signature": "",
        },
    )
    monkeypatch.setattr("agent.v4.nodes.ChatOpenAI", lambda **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not run")))

    executor = make_executor(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *V4_BUILTIN_LEGAL_TOOLS]))
    result = asyncio.run(executor(state))
    answer = str(result["messages"][0].content)

    assert "defined-risk only" not in answer.lower()
    assert "- delta: -0.073" in answer.lower()
    assert "- gamma: -0.026" in answer.lower()
    assert "- vega: -0.683" in answer.lower()


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


def test_v4_reviewer_rejects_single_recommended_legal_structure():
    prompt = "Need structure options for a cross-border acquisition with stock consideration and compliance risk."
    state = make_state(
        prompt,
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
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": ""},
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "v4_review_ready": True},
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Recommended structure: reverse triangular merger. "
                "Use escrow and indemnities. Stock consideration helps tax deferral."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "multiple structure alternatives" in " ".join(result["review_feedback"]["missing_dimensions"]).lower()


def test_v4_reviewer_flags_when_option_snapshot_is_not_front_loaded():
    prompt = "Need structure options for a cross-border acquisition with stock consideration and compliance risk."
    repetitive_prefix = "Recommended structure: reverse triangular merger. " * 80
    state = make_state(
        prompt,
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
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": ""},
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "v4_review_ready": True},
    )
    state["messages"].append(
        AIMessage(
            content=(
                repetitive_prefix
                + "\n\nStructure options: reverse triangular merger, asset purchase, carve-out transaction."
                + "\nTax consequences: tax-free reorganization under section 368 can give target shareholders tax deferral, buyer basis step-up can arise in an asset deal, required elections matter, and tax treatment breaks if qualification conditions fail."
                + "\nLiability protection: indemnities, escrow, caps, baskets, survival periods, and disclosure schedules."
                + "\nRegulatory and diligence risks: EU and US approvals, employee-transfer consultation, closing conditions, and remediation timing."
                + "\nKey open questions and assumptions: severity of compliance gaps, cure feasibility, seller indemnity support."
                + "\nRecommended next steps: week-one workplan with owners, sequencing, and accelerated diligence."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "opening summary with multiple viable paths" in " ".join(result["review_feedback"]["missing_dimensions"]).lower()


def test_v4_failed_reviewer_path_skips_self_reflection():
    state = make_state(
        "Need acquisition structure advice.",
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
        solver_stage="COMPLETE",
        quality_report={
            "verdict": "fail",
            "reasoning": "still missing structure coverage",
            "missing_dimensions": ["multiple structure alternatives with tradeoffs"],
            "targeted_fix_prompt": "",
            "score": 0.58,
        },
    )

    assert route_from_reviewer(state) == "reflect"


def test_v4_self_reflection_requests_one_extra_legal_deepen_pass(monkeypatch):
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

    # LLM self-reflection finds missing items
    reflection_response = '{"score": 0.55, "complete": false, "missing": ["execution-specific next steps", "risk allocation detail"], "improve_prompt": "Add execution-specific next steps and risk-allocation detail."}'
    monkeypatch.setattr(
        "agent.v4.nodes.ChatOpenAI",
        lambda **kwargs: _FakeModel(AIMessage(content=reflection_response)),
    )

    result = self_reflection(state)

    assert result["solver_stage"] == "REVISE"
    assert "next steps" in " ".join(result["review_feedback"]["missing_dimensions"]).lower()


def test_v4_legal_revision_prompt_requires_front_loaded_snapshot():
    prompt = build_revision_prompt(
        ["opening summary with multiple viable paths before the deep dive", "regulatory execution specifics"],
        improve_hint="Add approvals, consultation timing, and closing conditions.",
        task_family="legal_transactional",
    )

    assert "snapshot" in prompt.lower()
    assert "multiple viable structures" in prompt.lower()


def test_v4_tracer_captures_v4_headers_and_counts(monkeypatch):
    tracer = RunTracer()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "agent.tracer._write_trace_file",
        lambda payload, profile, start_dt: captured.setdefault("payload", payload) or "ok",
    )

    tracer.set_task("Need legal structure options.")
    tracer.record("fast_path_gate", {"task_family": "legal_transactional", "execution_mode": "advisory_analysis", "complexity_tier": "complex_qualitative"})
    tracer.record("task_planner", {"intent": {"task_family": "legal_transactional", "execution_mode": "advisory_analysis", "complexity_tier": "complex_qualitative"}, "template_id": "v4_advisory_analysis"})
    tracer.record("capability_resolver", {"selected_tools": ["legal_playbook_retrieval"], "pending_tools": [], "blocked_families": []})
    tracer.record(
        "v4_executor",
        {
            "intent": {"task_family": "legal_transactional", "execution_mode": "advisory_analysis", "complexity_tier": "complex_qualitative"},
            "used_llm": True,
            "tools_ran": ["legal_playbook_retrieval", "transaction_structure_checklist"],
            "tokens": {"prompt": 100, "completion": 40},
            "output_preview": "answer",
        },
    )
    tracer.finalize("final answer", {"llm_calls": 1}, {"context_tokens_used": 1000})
    payload = captured["payload"]

    assert payload["final_profile"] == "legal_transactional"
    assert payload["final_template"] == "v4_advisory_analysis"
    assert payload["complexity_tier"] == "complex_qualitative"
    assert payload["total_llm_calls"] == 1
    assert payload["total_tool_calls"] == 2


def test_budget_tracker_context_summary_uses_peak_and_total():
    budget = BudgetTracker()
    budget.record_context_tokens(1200)
    budget.record_context_tokens(800)

    summary = budget.summary()

    assert summary["context_tokens_used"] == 1200
    assert summary["context_tokens_total"] == 2000
