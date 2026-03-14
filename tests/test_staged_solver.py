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
