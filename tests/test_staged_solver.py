from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agent.nodes.solver import make_solver
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
