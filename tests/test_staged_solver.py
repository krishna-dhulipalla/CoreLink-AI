from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agent.nodes.solver import make_solver
from agent.solver.quant import deterministic_inline_quant_value
from staged_test_utils import make_state


class _FakeModel:
    def __init__(self, response):
        self._response = response
        self.last_messages = None

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.last_messages = messages
        return self._response


class _DummyTool:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description


def test_solver_emits_one_tool_call_in_compute_stage(monkeypatch):
    response = AIMessage(content='{"name":"calculator","arguments":{"expression":"(3.0433-1.5791)/1.5791"}}')
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(response))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("calculator", "Safe arithmetic")])
    state = make_state(
        "Calculate the financial leverage effect.",
        task_profile="finance_quant",
        capability_flags=["needs_math"],
        solver_stage="COMPUTE",
        evidence_pack={"prompt_facts": {"roe": 0.030433, "roa": 0.015791}},
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "calculator"
    assert result["messages"][0].tool_calls[0]["name"] == "calculator"


def test_solver_deterministically_computes_inline_quant_formula(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        'Financial Leverage Effect = (ROE - ROA) / ROA. Calculate it with ROE = 3.0433%, ROA = 1.579%. Output Format: {"answer": <value>}',
        task_profile="finance_quant",
        capability_flags=["needs_math", "requires_exact_format"],
        execution_template={
            "template_id": "quant_inline_exact",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["calculator"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="COMPUTE",
        answer_contract={"format": "json", "requires_adapter": True, "wrapper_key": "answer"},
        evidence_pack={"formulas": ["Financial Leverage Effect = (ROE - ROA) / ROA"]},
    )

    result = solver(state)

    assert result["pending_tool_call"] is None
    assert result["workpad"]["review_stage"] == "COMPUTE"
    assert "0.9273" in result["workpad"]["stage_outputs"]["COMPUTE"]


def test_solver_deterministically_seeds_live_data_price_history_call(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("get_price_history", "Retrieve price history")])
    state = make_state(
        "As of 2024-10-14, retrieve MSFT price history and 1-month return.",
        task_profile="finance_quant",
        capability_flags=["needs_live_data"],
        execution_template={
            "template_id": "quant_with_tool_compute",
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["get_price_history", "pct_change"],
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        evidence_pack={"entities": ["MSFT"], "prompt_facts": {"as_of_date": "2024-10-14"}},
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "get_price_history"
    assert result["pending_tool_call"]["arguments"]["ticker"] == "MSFT"
    assert result["pending_tool_call"]["arguments"]["period"] == "1mo"


def test_solver_plan_uses_template_initial_stage(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Look up the latest SEC filing for META.",
        task_profile="external_retrieval",
        capability_flags=["needs_live_data"],
        solver_stage="PLAN",
        execution_template={
            "template_id": "live_retrieval",
            "allowed_stages": ["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["internet_search"],
            "review_stages": ["GATHER", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
    )

    result = solver(state)

    assert result["solver_stage"] == "GATHER"


def test_solver_revise_can_target_compute_without_final_answer(monkeypatch):
    response = AIMessage(content="IV premium is 0.07, which supports a net short-volatility bias.")
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(response))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("analyze_strategy", "Analyze a strategy")])
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="REVISE",
        evidence_pack={"derived_facts": {"iv_premium": 0.07}},
        review_feedback={"repair_target": "compute", "missing_dimensions": ["concrete comparative analysis"]},
    )

    result = solver(state)

    assert result["workpad"]["review_ready"] is True
    assert result["workpad"]["review_stage"] == "COMPUTE"
    assert "IV premium is 0.07" in result["workpad"]["stage_outputs"]["COMPUTE"]
    assert "messages" not in result


def test_solver_synthesize_generates_final_draft(monkeypatch):
    response = AIMessage(content="Use an asset purchase or carve-out SPV to isolate legacy compliance liabilities.")
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(response))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "What structure options do we have for this acquisition?",
        task_profile="legal_transactional",
        capability_flags=["needs_legal_reasoning"],
        solver_stage="SYNTHESIZE",
    )

    result = solver(state)

    assert result["workpad"]["review_ready"] is True
    assert result["messages"][0].content.startswith("Use an asset purchase")


def test_solver_deterministically_synthesizes_inline_quant_scalar(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        'Financial Leverage Effect = (ROE - ROA) / ROA. Calculate it with ROE = 3.0433%, ROA = 1.579%. Output Format: {"answer": <value>}',
        task_profile="finance_quant",
        capability_flags=["needs_math", "requires_exact_format"],
        execution_template={
            "template_id": "quant_inline_exact",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["calculator"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        answer_contract={"format": "json", "requires_adapter": True, "wrapper_key": "answer"},
        evidence_pack={"formulas": ["Financial Leverage Effect = (ROE - ROA) / ROA"]},
    )

    result = solver(state)

    assert result["workpad"]["review_stage"] == "SYNTHESIZE"
    assert result["messages"][0].content.startswith('{"answer": 0.9273')


def test_solver_revise_wrapper_only_quant_uses_terminal_deterministic_final(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        'Financial Leverage Effect = (ROE - ROA) / ROA. Calculate it with ROE = 3.0433%, ROA = 1.579%. Output Format: {"answer": <value>}',
        task_profile="finance_quant",
        capability_flags=["needs_math", "requires_exact_format"],
        execution_template={
            "template_id": "quant_inline_exact",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["calculator"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="REVISE",
        answer_contract={"format": "json", "requires_adapter": True, "wrapper_key": "answer"},
        evidence_pack={"relevant_formulae": ["Financial Leverage Effect = (ROE - ROA) / ROA"]},
        review_feedback={
            "repair_target": "final",
            "repair_class": "wrapper_only",
            "missing_dimensions": ["scalar answer matching output contract"],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"] is None
    assert result["messages"][0].content.startswith('{"answer": 0.9273')
    assert result["workpad"]["review_stage"] == "SYNTHESIZE"


def test_inline_quant_refuses_ambiguous_relevant_rows():
    state = make_state(
        "Calculate the financial leverage effect for Company A in 2024.",
        task_profile="finance_quant",
        capability_flags=["needs_math", "requires_exact_format"],
        execution_template={
            "template_id": "quant_inline_exact",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["calculator"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        evidence_pack={
            "relevant_formulae": ["Financial Leverage Effect = (ROE - ROA) / ROA"],
            "relevant_rows": [
                {
                    "headers": ["Company", "2024 ROE", "2024 ROA"],
                    "rows": [
                        {"Company": "Company A", "2024 ROE": 0.030433, "2024 ROA": 0.015791},
                        {"Company": "Company B", "2024 ROE": 0.01, "2024 ROA": 0.005},
                    ],
                }
            ],
        },
    )

    assert deterministic_inline_quant_value(state) is None


def test_solver_revise_compute_uses_existing_tool_result_before_more_tools(monkeypatch):
    response = AIMessage(content="Primary strategy is tool-backed: short straddle credit with negative vega and positive theta.")
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(response))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("analyze_strategy", "Analyze a strategy")])
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="REVISE",
        evidence_pack={"derived_facts": {"iv_premium": 0.07}},
        review_feedback={"repair_target": "compute", "missing_dimensions": ["tool-backed strategy analysis"]},
        last_tool_result={
            "type": "analyze_strategy",
            "facts": {"net_premium": 9.16, "total_theta_per_day": 0.04},
            "assumptions": {},
            "source": {"tool": "analyze_strategy"},
            "errors": [],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"] is None
    assert result["workpad"]["review_stage"] == "COMPUTE"
    assert "tool-backed" in result["workpad"]["stage_outputs"]["COMPUTE"].lower()


def test_solver_risk_revise_compute_can_emit_scenario_tool_call(monkeypatch):
    fake_model = _FakeModel(
        AIMessage(
            content='{"name":"scenario_pnl","arguments":{"premium":23.98,"delta":-0.0726,"gamma":-0.0264,"theta":0.4393,"vega":-0.6834,"spot":300}}'
        )
    )
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: fake_model)
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("scenario_pnl", "Run scenario P&L analysis")])
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="REVISE",
        review_feedback=None,
        risk_feedback={
            "repair_target": "compute",
            "violation_codes": ["MISSING_SCENARIO_ANALYSIS"],
            "risk_findings": ["No scenario evidence yet."],
            "required_disclosures": ["Explicitly disclose short-volatility / volatility-spike risk."],
            "reasoning": "Add scenario coverage before synthesis.",
        },
        last_tool_result={
            "type": "analyze_strategy",
            "facts": {
                "net_premium": 23.98,
                "total_delta": -0.0726,
                "total_gamma": -0.0264,
                "total_theta_per_day": 0.4393,
                "total_vega_per_vol_point": -0.6834,
            },
            "assumptions": {"legs": [{"S": 300}]},
            "source": {"tool": "analyze_strategy"},
            "errors": [],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "scenario_pnl"
    assert result["messages"][0].tool_calls[0]["name"] == "scenario_pnl"


def test_solver_deterministically_seeds_scenario_pnl_from_risk_feedback(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("scenario_pnl", "Run scenario P&L analysis")])
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="REVISE",
        risk_feedback={
            "repair_target": "compute",
            "violation_codes": ["MISSING_SCENARIO_ANALYSIS"],
            "risk_findings": [],
            "required_disclosures": [],
        },
        last_tool_result={
            "type": "analyze_strategy",
            "facts": {
                "net_premium": 23.98,
                "total_delta": -0.0726,
                "total_gamma": -0.0264,
                "total_theta_per_day": 0.4393,
                "total_vega_per_vol_point": -0.6834,
            },
            "assumptions": {"legs": [{"S": 300.0}]},
            "source": {"tool": "analyze_strategy"},
            "errors": [],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "scenario_pnl"
    assert result["pending_tool_call"]["arguments"]["net_premium"] == 23.98
    assert result["pending_tool_call"]["arguments"]["reference_price"] == 300.0


def test_solver_deterministically_seeds_scenario_pnl_from_black_scholes_result(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("scenario_pnl", "Run scenario P&L analysis")])
    state = make_state(
        "Should you be a net buyer or seller of options when IV is elevated?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="REVISE",
        risk_feedback={
            "repair_target": "compute",
            "violation_codes": ["MISSING_SCENARIO_ANALYSIS"],
            "risk_findings": [],
            "required_disclosures": [],
        },
        last_tool_result={
            "type": "black_scholes_price",
            "facts": {
                "call_price": 12.6,
                "put_price": 11.37,
                "delta": 0.536,
                "gamma": 0.013,
                "theta": -0.22,
                "vega": 0.342,
                "max_loss": 12.6,
            },
            "assumptions": {"S": 300.0, "K": 300.0, "option_type": "call"},
            "source": {"tool": "black_scholes_price"},
            "errors": [],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "scenario_pnl"
    assert result["pending_tool_call"]["arguments"]["net_premium"] == 12.6
    assert result["pending_tool_call"]["arguments"]["total_delta"] == 0.536
    assert result["pending_tool_call"]["arguments"]["reference_price"] == 300.0


def test_solver_builds_deterministic_options_compute_summary_after_scenario_tool(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Should you be a net buyer or seller of options when IV is elevated?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="COMPUTE",
        last_tool_result={
            "type": "scenario_pnl",
            "facts": {
                "worst_case_pnl": -18.4,
                "best_case_pnl": 0.44,
            },
            "assumptions": {"reference_price": 300.0},
            "source": {"tool": "scenario_pnl"},
            "errors": [],
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "black_scholes_price",
                    "facts": {
                        "call_price": 12.6,
                        "delta": 0.536,
                        "gamma": 0.013,
                        "theta": -0.22,
                        "vega": 0.342,
                        "max_loss": 12.6,
                    },
                    "assumptions": {"S": 300.0, "K": 300.0, "option_type": "call"},
                    "source": {"tool": "black_scholes_price"},
                    "errors": [],
                },
                {
                    "type": "scenario_pnl",
                    "facts": {"worst_case_pnl": -18.4, "best_case_pnl": 0.44},
                    "assumptions": {"reference_price": 300.0},
                    "source": {"tool": "scenario_pnl"},
                    "errors": [],
                },
            ],
            "risk_results": [
                {
                    "verdict": "revise",
                    "violation_codes": ["MISSING_RISK_CONTROLS"],
                    "repair_target": "compute",
                }
            ],
        },
    )

    result = solver(state)

    assert result["workpad"]["review_ready"] is True
    assert result["workpad"]["review_stage"] == "COMPUTE"
    compute_text = result["workpad"]["stage_outputs"]["COMPUTE"].lower()
    assert "position sizing" in compute_text
    assert "stop-loss" in compute_text
    assert "max loss" in compute_text


def test_solver_builds_primary_options_compute_milestone_after_first_strategy_tool(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Should you be a net buyer or seller of options when IV is elevated?",
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
        solver_stage="COMPUTE",
        last_tool_result={
            "type": "analyze_strategy",
            "facts": {
                "net_premium": 9.16,
                "premium_direction": "CREDIT",
                "total_delta": -0.12,
                "total_gamma": -0.0015,
                "total_theta_per_day": 0.04,
                "total_vega_per_vol_point": -0.06,
                "max_loss": 22.4,
            },
            "assumptions": {"legs": [{"S": 300.0}]},
            "source": {"tool": "analyze_strategy"},
            "errors": [],
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 9.16,
                        "premium_direction": "CREDIT",
                        "total_delta": -0.12,
                        "total_gamma": -0.0015,
                        "total_theta_per_day": 0.04,
                        "total_vega_per_vol_point": -0.06,
                        "max_loss": 22.4,
                    },
                    "assumptions": {"legs": [{"S": 300.0}]},
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                }
            ],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"] is None
    assert result["workpad"]["review_ready"] is True
    assert result["workpad"]["review_stage"] == "COMPUTE"
    compute_text = result["workpad"]["stage_outputs"]["COMPUTE"].lower()
    assert "tool-backed and ready for risk review" in compute_text
    assert "scenario analysis is still required" in compute_text


def test_solver_builds_deterministic_options_final_answer_after_risk_pass(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Should you be a net buyer or seller of options when IV is elevated?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="SYNTHESIZE",
        evidence_pack={"derived_facts": {"vol_bias": "short_vol"}},
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        assumption_ledger=[
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 300.0 from tool arguments because it was not explicit in prompt evidence.",
                "source": "tool_arguments:analyze_strategy",
                "confidence": "medium",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            }
        ],
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 15.33,
                        "premium_direction": "CREDIT",
                        "total_delta": -0.0729,
                        "total_gamma": -0.025,
                        "total_theta_per_day": 0.4177,
                        "total_vega_per_vol_point": -0.6467,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "put", "action": "sell", "S": 300.0, "K": 290.0},
                            {"option_type": "call", "action": "sell", "S": 300.0, "K": 310.0},
                        ]
                    },
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                {
                    "type": "scenario_pnl",
                    "facts": {"best_case_pnl": 0.42, "worst_case_pnl": -20.92},
                    "assumptions": {"reference_price": 300.0},
                    "source": {"tool": "scenario_pnl"},
                    "errors": [],
                },
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {
                "required_disclosures": [
                    "Explicitly disclose short-volatility / volatility-spike risk.",
                    "Explicitly disclose potentially unbounded tail loss and gap risk.",
                    "State downside scenario loss and the exit or sizing response.",
                ],
                "recommendation_class": "scenario_dependent_recommendation",
            },
        },
    )

    result = solver(state)

    assert result["workpad"]["review_ready"] is True
    assert result["workpad"]["review_stage"] == "SYNTHESIZE"
    content = result["messages"][0].content.lower()
    assert "recommendation" in content
    assert "alternative strategy comparison" in content
    assert "key greeks and breakevens" in content
    assert "risk management" in content
    assert "disclosures" in content
    assert "short-volatility / volatility-spike risk" in content


def test_solver_dedupes_equivalent_option_assumption_disclosures(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Should you be a net buyer or seller of options when IV is elevated?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="SYNTHESIZE",
        evidence_pack={"derived_facts": {"vol_bias": "short_vol"}},
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        assumption_ledger=[
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 300.0 from tool arguments because it was not explicit in prompt evidence.",
                "source": "tool_arguments:analyze_strategy",
                "confidence": "medium",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            },
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 300 from tool arguments because it was not explicit in prompt evidence.",
                "source": "tool_arguments:scenario_pnl",
                "confidence": "medium",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            },
        ],
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 15.33,
                        "premium_direction": "CREDIT",
                        "total_delta": -0.0729,
                        "total_gamma": -0.025,
                        "total_theta_per_day": 0.4177,
                        "total_vega_per_vol_point": -0.6467,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "put", "action": "sell", "S": 300.0, "K": 290.0},
                            {"option_type": "call", "action": "sell", "S": 300.0, "K": 310.0},
                        ]
                    },
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                {
                    "type": "scenario_pnl",
                    "facts": {"best_case_pnl": 0.42, "worst_case_pnl": -20.92},
                    "assumptions": {"reference_price": 300.0},
                    "source": {"tool": "scenario_pnl"},
                    "errors": [],
                },
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {
                "required_disclosures": ["Explicitly disclose short-volatility / volatility-spike risk."],
                "recommendation_class": "scenario_dependent_recommendation",
            },
        },
    )

    result = solver(state)

    content = result["messages"][0].content
    assert content.count("Spot price was assumed as") == 1


def test_solver_uses_deterministic_policy_tool_call_for_defined_risk_options():
    solver = make_solver([_DummyTool("analyze_strategy", "Analyze an options strategy")])
    state = make_state(
        "Design an options strategy, but use defined-risk only and no naked options.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="COMPUTE",
        evidence_pack={
            "derived_facts": {"vol_bias": "short_vol"},
            "policy_context": {"defined_risk_only": True, "no_naked_options": True, "requires_recommendation_class": True},
            "prompt_facts": {"implied_volatility": 0.35},
        },
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "analyze_strategy"
    legs = result["pending_tool_call"]["arguments"]["legs"]
    assert len(legs) == 4
    assert {leg["action"] for leg in legs} == {"buy", "sell"}
    assert {leg["option_type"] for leg in legs} == {"call", "put"}


def test_solver_uses_deterministic_primary_tool_call_for_standard_options():
    solver = make_solver([_DummyTool("analyze_strategy", "Analyze an options strategy")])
    state = make_state(
        "META's current IV is 35% while its 30-day historical volatility is 28%. The IV percentile is 75%. Should you be a net buyer or seller of options? Design a strategy accordingly.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="COMPUTE",
        evidence_pack={
            "derived_facts": {"vol_bias": "short_vol"},
            "prompt_facts": {"implied_volatility": 0.35, "historical_volatility": 0.28},
        },
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "analyze_strategy"
    legs = result["pending_tool_call"]["arguments"]["legs"]
    assert len(legs) == 2
    assert {leg["action"] for leg in legs} == {"sell"}
    assert {leg["option_type"] for leg in legs} == {"call", "put"}


def test_solver_uses_history_reference_price_for_standard_options_tool_call():
    solver = make_solver([_DummyTool("analyze_strategy", "Analyze an options strategy")])
    state = make_state(
        "META's IV is elevated versus realized volatility. Design a strategy accordingly.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="COMPUTE",
        evidence_pack={
            "derived_facts": {"vol_bias": "short_vol"},
            "prompt_facts": {"implied_volatility": 0.35, "historical_volatility": 0.28},
        },
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "get_price_history",
                    "facts": {"start_close": 405.0, "end_close": 412.3},
                    "assumptions": {"ticker": "META", "period": "3mo"},
                    "source": {"tool": "get_price_history", "timestamp": "2024-10-14"},
                    "errors": [],
                }
            ],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "analyze_strategy"
    legs = result["pending_tool_call"]["arguments"]["legs"]
    assert all(leg["S"] == 412.3 for leg in legs)


def test_solver_uses_deterministic_policy_final_when_primary_strategy_is_compliant():
    solver = make_solver([])
    state = make_state(
        "Design an options strategy, but use defined-risk only and no naked options.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="SYNTHESIZE",
        evidence_pack={
            "derived_facts": {"vol_bias": "short_vol"},
            "policy_context": {
                "defined_risk_only": True,
                "no_naked_options": True,
                "requires_recommendation_class": True,
                "max_position_risk_pct": 2.0,
            },
        },
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        assumption_ledger=[
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 300 from tool arguments because it was not explicit in prompt evidence.",
                "requires_user_visible_disclosure": True,
            }
        ],
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 4.15,
                        "premium_direction": "CREDIT",
                        "total_delta": -0.031,
                        "total_gamma": -0.011,
                        "total_theta_per_day": 0.125,
                        "total_vega_per_vol_point": -0.241,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "put", "action": "buy", "S": 300.0, "K": 290.0},
                            {"option_type": "put", "action": "sell", "S": 300.0, "K": 295.0},
                            {"option_type": "call", "action": "sell", "S": 300.0, "K": 305.0},
                            {"option_type": "call", "action": "buy", "S": 300.0, "K": 310.0},
                        ]
                    },
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                {
                    "type": "scenario_pnl",
                    "facts": {"best_case_pnl": 0.44, "worst_case_pnl": -3.1},
                    "assumptions": {"reference_price": 300.0},
                    "source": {"tool": "scenario_pnl"},
                    "errors": [],
                },
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {
                "required_disclosures": [
                    "Explicitly disclose short-volatility / volatility-spike risk.",
                    "State downside scenario loss and the exit or sizing response.",
                ],
                "recommendation_class": "scenario_dependent_recommendation",
            },
        },
    )

    result = solver(state)

    content = result["messages"][0].content.lower()
    assert "net seller of options" in content
    assert "defined-risk only" in content
    assert "naked options are not permitted" in content
    assert "volatility-spike risk" in content
    assert "recommendation class: scenario_dependent_recommendation." in content


def test_solver_appends_compliance_recommendation_class_fix(monkeypatch):
    fake_model = _FakeModel(AIMessage(content="Use a defined-risk iron condor with 1% position sizing."))
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: fake_model)
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Use a defined-risk-only options strategy.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="REVISE",
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        evidence_pack={"policy_context": {"defined_risk_only": True, "requires_recommendation_class": True}},
        compliance_feedback={
            "repair_target": "final",
            "violation_codes": ["MISSING_RECOMMENDATION_CLASS"],
            "required_disclosures": ["State the recommendation class explicitly."],
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "risk_requirements": {"recommendation_class": "scenario_dependent_recommendation"},
        },
    )

    result = solver(state)

    assert "Recommendation class: scenario_dependent_recommendation." in result["messages"][0].content


def test_solver_document_gather_prompt_requires_targeted_extraction(monkeypatch):
    fake_model = _FakeModel(AIMessage(content='{"name":"fetch_reference_file","arguments":{"url":"https://example.com/report.csv","row_limit":25}}'))
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: fake_model)
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("fetch_reference_file", "Fetch a reference file")])
    state = make_state(
        "Read the attached report and answer from the file at https://example.com/report.csv.",
        task_profile="document_qa",
        capability_flags=["needs_files"],
        execution_template={
            "template_id": "document_qa",
            "allowed_stages": ["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["fetch_reference_file", "list_reference_files"],
            "review_stages": ["GATHER", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        evidence_pack={
            "prompt_facts": {},
            "retrieved_facts": {},
            "derived_facts": {},
            "document_evidence": [
                {
                    "document_id": "report_csv",
                    "citation": "https://example.com/report.csv",
                    "status": "discovered",
                    "metadata": {"format": "csv"},
                    "chunks": [],
                    "tables": [],
                    "numeric_summaries": [],
                }
            ],
            "citations": ["https://example.com/report.csv"],
            "open_questions": ["Document evidence has not been extracted yet."],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "fetch_reference_file"
    assert "narrow page/row window first" in fake_model.last_messages[1].content


def test_solver_live_data_quant_gather_prompts_for_finance_evidence_tool(monkeypatch):
    fake_model = _FakeModel(AIMessage(content='{"name":"get_price_history","arguments":{"entity":"MSFT","time_frame":"1M","as_of":"2024-10-14"}}'))
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: fake_model)
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("get_price_history", "Retrieve market price history")])
    state = make_state(
        "As of 2024-10-14, use finance evidence tools to retrieve MSFT price history and 1-month return.",
        task_profile="finance_quant",
        capability_flags=["needs_live_data"],
        execution_template={
            "template_id": "quant_with_tool_compute",
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["get_price_history", "get_returns"],
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        evidence_pack={
            "prompt_facts": {"as_of_date": "2024-10-14"},
            "retrieved_facts": {},
            "derived_facts": {"time_sensitive": True},
            "document_evidence": [],
            "citations": [],
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "get_price_history"
    assert "emit one finance evidence tool call before any narrative" in fake_model.last_messages[1].content


def test_solver_compact_evidence_uses_document_summary_not_raw_blob(monkeypatch):
    long_text = "alpha beta gamma " * 200
    fake_model = _FakeModel(AIMessage(content="Answer: the document shows the covenant threshold in the extracted table."))
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: fake_model)
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Summarize the covenant threshold from the attached report.",
        task_profile="document_qa",
        capability_flags=["needs_files"],
        execution_template={
            "template_id": "document_qa",
            "allowed_stages": ["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "SYNTHESIZE",
            "allowed_tool_names": ["fetch_reference_file"],
            "review_stages": ["GATHER", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        evidence_pack={
            "prompt_facts": {},
            "retrieved_facts": {"fetch_reference_file": {"documents": [{"document_id": "report_pdf", "status": "extracted"}]}},
            "derived_facts": {},
            "document_evidence": [
                {
                    "document_id": "report_pdf",
                    "citation": "https://example.com/report.pdf",
                    "status": "extracted",
                    "metadata": {"file_name": "report.pdf", "format": "pdf", "window": "Pages 1-2"},
                    "chunks": [{"locator": "Pages 1-2", "kind": "text_excerpt", "text": long_text, "citation": "https://example.com/report.pdf"}],
                    "tables": [],
                    "numeric_summaries": [],
                }
            ],
            "citations": ["https://example.com/report.pdf"],
        },
    )

    solver(state)

    evidence_message = fake_model.last_messages[2].content
    assert '"document_evidence"' in evidence_message
    assert long_text[:400] not in evidence_message
    assert '"chunk_count": 1' in evidence_message


def test_solver_deterministically_seeds_equity_research_gather(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("get_company_fundamentals", "Retrieve company fundamentals")])
    state = make_state(
        "Write an equity research report on MSFT as of 2024-10-14 with thesis and valuation framing.",
        task_profile="finance_quant",
        capability_flags=["needs_equity_research", "needs_live_data"],
        execution_template={
            "template_id": "equity_research_report",
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["get_company_fundamentals"],
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        evidence_pack={"entities": ["MSFT"], "prompt_facts": {"as_of_date": "2024-10-14"}},
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "get_company_fundamentals"
    assert result["pending_tool_call"]["arguments"]["ticker"] == "MSFT"


def test_solver_deterministically_seeds_portfolio_risk_concentration_check(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("concentration_check", "Check concentration")])
    state = make_state(
        "Review this portfolio risk and recommend actions.",
        task_profile="finance_quant",
        capability_flags=["needs_portfolio_risk"],
        execution_template={
            "template_id": "portfolio_risk_review",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["concentration_check"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="COMPUTE",
        evidence_pack={
            "prompt_facts": {
                "portfolio_positions": [
                    {"ticker": "AAPL", "weight": 0.35, "sector": "Technology"},
                    {"ticker": "MSFT", "weight": 0.30, "sector": "Technology"},
                    {"ticker": "XOM", "weight": 0.20, "sector": "Energy"},
                ]
            }
        },
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "concentration_check"
    assert len(result["pending_tool_call"]["arguments"]["exposures"]) == 3


def test_solver_builds_richer_equity_research_final(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Write an equity research report on MSFT.",
        task_profile="finance_quant",
        capability_flags=["needs_equity_research", "needs_live_data"],
        execution_template={
            "template_id": "equity_research_report",
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["get_company_fundamentals", "get_price_history"],
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "get_company_fundamentals",
                    "facts": {
                        "fundamentals": {
                            "revenueGrowth": 0.12,
                            "operatingMargins": 0.31,
                            "returnOnEquity": 0.28,
                            "trailingPE": 29.4,
                            "forwardPE": 27.1,
                        }
                    },
                    "assumptions": {"ticker": "MSFT"},
                    "source": {"tool": "get_company_fundamentals", "timestamp": "2024-10-14"},
                    "errors": [],
                },
                {
                    "type": "get_price_history",
                    "facts": {"start_close": 410.0, "end_close": 432.0},
                    "assumptions": {"ticker": "MSFT", "period": "6mo"},
                    "source": {"tool": "get_price_history", "timestamp": "2024-10-14"},
                    "errors": [],
                },
            ],
        },
    )

    result = solver(state)

    content = result["messages"][0].content
    assert "**Recommendation**" in content
    assert "**What Would Change The View**" in content
    assert "**Catalysts and Watchpoints**" in content
    assert "Source timestamp: 2024-10-14." in content
    assert "Recommendation Class" in content


def test_solver_builds_richer_portfolio_risk_final(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Review this portfolio risk and recommend actions.",
        task_profile="finance_quant",
        capability_flags=["needs_portfolio_risk"],
        execution_template={
            "template_id": "portfolio_risk_review",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["concentration_check", "calculate_var", "portfolio_limit_check"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "concentration_check",
                    "facts": {
                        "has_breach": True,
                        "name_breaches": [{"name": "AAPL", "weight": 0.35}],
                        "sector_breaches": [{"sector": "Technology", "weight": 0.65}],
                    },
                    "assumptions": {},
                    "source": {"tool": "concentration_check", "timestamp": "2024-10-14"},
                    "errors": [],
                },
                {
                    "type": "factor_exposure_summary",
                    "facts": {"largest_factor": "technology_beta", "largest_factor_weight": 0.42},
                    "assumptions": {},
                    "source": {"tool": "factor_exposure_summary", "timestamp": "2024-10-14"},
                    "errors": [],
                },
                {
                    "type": "calculate_var",
                    "facts": {"var_amount": 125000.0, "var_decimal": 0.067},
                    "assumptions": {},
                    "source": {"tool": "calculate_var", "timestamp": "2024-10-14"},
                    "errors": [],
                },
                {
                    "type": "portfolio_limit_check",
                    "facts": {"hard_limit_breached": True},
                    "assumptions": {},
                    "source": {"tool": "portfolio_limit_check", "timestamp": "2024-10-14"},
                    "errors": [],
                },
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {
                "required_disclosures": ["State downside scenario loss and the exit or sizing response."],
                "risk_findings": ["Portfolio limit breach detected."],
                "recommendation_class": "scenario_dependent_recommendation",
            },
        },
    )

    result = solver(state)

    content = result["messages"][0].content
    assert "**Recommendation**" in content
    assert "Source timestamp: 2024-10-14." in content
    assert "**Immediate Actions**" in content
    assert "**Hedging / Rebalance Alternatives**" in content
    assert "**Monitoring Triggers**" in content
    assert "Portfolio limit breach detected." in content


def test_solver_builds_richer_event_driven_final(monkeypatch):
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(AIMessage(content="unused")))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([])
    state = make_state(
        "Assess this event-driven setup.",
        task_profile="finance_quant",
        capability_flags=["needs_event_driven_finance", "needs_live_data"],
        execution_template={
            "template_id": "event_driven_finance",
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["get_price_history", "get_corporate_actions"],
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "get_corporate_actions",
                    "facts": {"recent_dividends": [{"date": "2024-09-01", "amount": 0.75}], "recent_splits": []},
                    "assumptions": {"ticker": "MSFT"},
                    "source": {"tool": "get_corporate_actions", "timestamp": "2024-10-14"},
                    "errors": [],
                },
                {
                    "type": "get_price_history",
                    "facts": {"start_close": 410.0, "end_close": 447.0},
                    "assumptions": {"ticker": "MSFT", "period": "3mo"},
                    "source": {"tool": "get_price_history", "timestamp": "2024-10-14"},
                    "errors": [],
                },
            ],
        },
    )

    result = solver(state)

    content = result["messages"][0].content
    assert "**Recommendation**" in content
    assert "**Execution Discipline**" in content
    assert "**Scenarios**" in content
    assert "**What Would Change The View**" in content
    assert "Source timestamp: 2024-10-14." in content
