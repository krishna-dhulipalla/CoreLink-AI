from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agent.nodes.solver import make_solver
from staged_test_utils import make_state


class _FakeModel:
    def __init__(self, response):
        self._response = response

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
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
        evidence_pack={"inline_facts": {"roe": 0.030433, "roa": 0.015791}},
    )

    result = solver(state)

    assert result["pending_tool_call"]["name"] == "calculator"
    assert result["messages"][0].tool_calls[0]["name"] == "calculator"


def test_solver_revise_can_target_compute_without_final_answer(monkeypatch):
    response = AIMessage(content="IV premium is 0.07, which supports a net short-volatility bias.")
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(response))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("analyze_strategy", "Analyze a strategy")])
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="REVISE",
        evidence_pack={"derived_signals": {"iv_premium": 0.07}},
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


def test_solver_revise_compute_uses_existing_tool_result_before_more_tools(monkeypatch):
    response = AIMessage(content="Primary strategy is tool-backed: short straddle credit with negative vega and positive theta.")
    monkeypatch.setattr("agent.nodes.solver.ChatOpenAI", lambda **kwargs: _FakeModel(response))
    monkeypatch.setattr("agent.nodes.solver._tool_call_mode", lambda role: "prompt")

    solver = make_solver([_DummyTool("analyze_strategy", "Analyze a strategy")])
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="REVISE",
        evidence_pack={"derived_signals": {"iv_premium": 0.07}},
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
