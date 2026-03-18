import json
import asyncio

from langchain_core.messages import AIMessage, ToolMessage

from agent.nodes.tool_runner import make_tool_runner
from staged_test_utils import make_state


class _DummyTool:
    def __init__(self, name):
        self.name = name


class _DummyToolNode:
    def __init__(self, tool_name, content):
        self.tools_by_name = {tool_name: _DummyTool(tool_name)}
        self._tool_name = tool_name
        self._content = content

    async def ainvoke(self, state):
        pending = state["pending_tool_call"]
        return {
            "messages": [
                ToolMessage(
                    content=self._content,
                    tool_call_id="call_123",
                    name=pending["name"],
                )
            ]
        }


class _CapturingToolNode:
    def __init__(self, tool_name, content):
        self.tools_by_name = {tool_name: _DummyTool(tool_name)}
        self.seen_pending = None
        self._content = content

    async def ainvoke(self, state):
        self.seen_pending = state["pending_tool_call"]
        return {
            "messages": [
                ToolMessage(
                    content=self._content,
                    tool_call_id="call_123",
                    name=self.seen_pending["name"],
                )
            ]
        }


class _CapturingMessageToolNode:
    def __init__(self, tool_name, content):
        self.tools_by_name = {tool_name: _DummyTool(tool_name)}
        self.seen_tool_call = None
        self._content = content

    async def ainvoke(self, state):
        last_ai = next(msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage))
        self.seen_tool_call = last_ai.tool_calls[-1]
        return {
            "messages": [
                ToolMessage(
                    content=self._content,
                    tool_call_id=self.seen_tool_call["id"],
                    name=self.seen_tool_call["name"],
                )
            ]
        }


def test_tool_runner_blocks_disallowed_tool():
    runner = make_tool_runner(_DummyToolNode("internet_search", "unused"))
    state = make_state(
        "Advise on deal structure.",
        task_profile="legal_transactional",
        pending_tool_call={"name": "internet_search", "arguments": {"query": "M&A law"}},
    )

    result = asyncio.run(runner(state))

    assert result["tool_fail_count"] == 1
    assert "not allowed" in result["last_tool_result"]["errors"][0]


def test_tool_runner_blocks_profile_allowed_tool_when_template_disallows_it():
    runner = make_tool_runner(_DummyToolNode("fetch_reference_file", "unused"))
    state = make_state(
        "Advise on deal structure from the prompt only.",
        task_profile="legal_transactional",
        execution_template={
            "template_id": "legal_reasoning_only",
            "allowed_tool_names": ["calculator"],
            "allowed_stages": ["SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "SYNTHESIZE",
            "review_stages": ["SYNTHESIZE"],
            "review_cadence": "final_only",
            "answer_focus": [],
        },
        pending_tool_call={"name": "fetch_reference_file", "arguments": {"url": "https://example.com/deal.pdf"}},
    )

    result = asyncio.run(runner(state))

    assert result["tool_fail_count"] == 1
    assert "not allowed" in result["last_tool_result"]["errors"][0]


def test_tool_runner_normalizes_analyze_strategy_output():
    raw = (
        "Multi-Leg Strategy (2 legs):\n"
        "  Net Premium  : +9.16 (CREDIT)\n"
        "  Total Delta  : -0.1200\n"
        "  Total Gamma  : -0.0010\n"
        "  Total Theta  : +0.0400 /day\n"
        "  Total Vega   : -0.0600 per 1% vol\n"
    )
    runner = make_tool_runner(_DummyToolNode("analyze_strategy", raw))
    state = make_state(
        "Analyze a short premium strategy.",
        task_profile="finance_options",
        evidence_pack={"prompt_facts": {}, "retrieved_facts": {}, "derived_facts": {}, "citations": []},
        pending_tool_call={
            "name": "analyze_strategy",
            "arguments": {
                "legs": [
                    {"option_type": "call", "action": "sell", "S": 300.0, "K": 310.0, "T_days": 30, "sigma": 0.35}
                ]
            },
        },
    )

    result = asyncio.run(runner(state))
    normalized = result["last_tool_result"]

    assert normalized["facts"]["net_premium"] == 9.16
    assert normalized["facts"]["premium_direction"] == "CREDIT"
    assert normalized["facts"]["total_theta_per_day"] == 0.04
    assert result["messages"][0].name == "analyze_strategy"
    assert json.loads(result["messages"][0].content)["facts"]["total_delta"] == -0.12
    assert result["evidence_pack"]["derived_facts"]["analyze_strategy"]["net_premium"] == 9.16
    assert any(entry["key"] == "spot_price" for entry in result["assumption_ledger"])


def test_tool_runner_flags_unstructured_output():
    runner = make_tool_runner(_DummyToolNode("fetch_reference_file", "Narrative only without parseable metadata"))
    state = make_state(
        "Read the attached file.",
        task_profile="document_qa",
        pending_tool_call={"name": "fetch_reference_file", "arguments": {"url": "https://example.com/report.pdf"}},
    )

    result = asyncio.run(runner(state))

    assert result["last_tool_result"]["errors"]


def test_tool_runner_normalizes_options_chain_output():
    raw = (
        "Options Chain: S=300, σ=35%, T=30d, r=5.00%\n"
        "    Strike       Call        Put    CallDelta   PutDelta\n"
        "-------------------------------------------------------\n"
        "    270.00      34.85       3.74       0.8364    -0.1636\n"
        "    300.00      14.73      13.50       0.5402    -0.4598 ←ATM\n"
    )
    runner = make_tool_runner(_DummyToolNode("get_options_chain", raw))
    state = make_state(
        "Show an options chain.",
        task_profile="finance_options",
        pending_tool_call={"name": "get_options_chain", "arguments": {"S": 300, "r": 0.05, "sigma": 0.35, "T_days": 30}},
    )

    result = asyncio.run(runner(state))
    normalized = result["last_tool_result"]

    assert normalized["facts"]["spot"] == 300.0
    assert normalized["facts"]["chain"][1]["is_atm"] is True
    assert normalized["facts"]["chain"][1]["call_delta"] == 0.5402


def test_tool_runner_normalizes_reference_file_rows():
    raw = (
        "FILE: report.csv\n"
        "FORMAT: CSV | SIZE: 4.2 KB\n"
        "--------------------------------------------------\n"
        "[Rows 0-3 of ~3]\n"
        "metric,value\n"
        "roe,3.0433\n"
        "roa,1.5791\n"
    )
    runner = make_tool_runner(_DummyToolNode("fetch_reference_file", raw))
    state = make_state(
        "Read the attached file.",
        task_profile="document_qa",
        evidence_pack={"prompt_facts": {}, "retrieved_facts": {}, "derived_facts": {}, "citations": []},
        pending_tool_call={"name": "fetch_reference_file", "arguments": {"url": "https://example.com/report.csv"}},
    )

    result = asyncio.run(runner(state))
    normalized = result["last_tool_result"]

    assert normalized["facts"]["metadata"]["file_name"] == "report.csv"
    assert normalized["facts"]["metadata"]["format"] == "csv"
    assert normalized["facts"]["tables"][0]["headers"] == ["metric", "value"]
    assert normalized["facts"]["tables"][0]["rows"][0] == ["roe", "3.0433"]
    assert any(summary["metric"] == "row_count" for summary in normalized["facts"]["numeric_summaries"])
    assert result["evidence_pack"]["document_evidence"][0]["metadata"]["file_name"] == "report.csv"
    assert result["evidence_pack"]["document_evidence"][0]["status"] == "extracted"
    assert result["evidence_pack"]["retrieved_facts"]["fetch_reference_file"]["documents"][0]["document_id"] == "report_csv"
    assert result["provenance_map"]["document_evidence.report_csv.metadata.file_name"]["source_class"] == "retrieved"
    assert "preview" not in normalized["facts"]


def test_tool_runner_normalizes_reference_listing_into_document_placeholders():
    raw = (
        "REFERENCE FILES DETECTED:\n\n"
        "  1. [CSV] https://example.com/report.csv\n"
        "  2. [PDF] https://example.com/deal.pdf\n"
    )
    runner = make_tool_runner(_DummyToolNode("list_reference_files", raw))
    state = make_state(
        "Read the attached files.",
        task_profile="document_qa",
        evidence_pack={"prompt_facts": {}, "retrieved_facts": {}, "derived_facts": {}, "document_evidence": [], "citations": []},
        pending_tool_call={"name": "list_reference_files", "arguments": {"prompt_text": "REFERENCE FILES AVAILABLE"}} ,
    )

    result = asyncio.run(runner(state))

    assert result["last_tool_result"]["facts"]["document_count"] == 2
    assert result["evidence_pack"]["document_evidence"][0]["status"] == "discovered"
    assert result["evidence_pack"]["retrieved_facts"]["list_reference_files"]["document_count"] == 2


def test_tool_runner_normalizes_expiration_schedule():
    raw = (
        "Available Expirations:\n"
        "     7d  2026-03-11  [Weekly]\n"
        "    30d  2026-04-03  [Monthly]\n"
    )
    runner = make_tool_runner(_DummyToolNode("get_expirations", raw))
    state = make_state(
        "List available expirations.",
        task_profile="finance_options",
        pending_tool_call={"name": "get_expirations", "arguments": {"T_days_list": [7, 30]}},
    )

    result = asyncio.run(runner(state))
    normalized = result["last_tool_result"]

    assert normalized["facts"]["expirations"][0]["days"] == 7
    assert normalized["facts"]["expirations"][1]["label"] == "Monthly"


def test_tool_runner_normalizes_finance_argument_aliases_before_invoke():
    node = _CapturingToolNode(
        "get_price_history",
        {
            "type": "get_price_history",
            "facts": {"ticker": "MSFT", "period": "1mo"},
            "source": {"tool": "get_price_history", "provider": "yfinance"},
            "quality": {"cache_hit": False, "is_synthetic": False, "is_estimated": False, "missing_fields": []},
            "errors": [],
        },
    )
    runner = make_tool_runner(node)
    state = make_state(
        "As of 2024-10-14, retrieve MSFT price history.",
        task_profile="finance_quant",
        execution_template={
            "template_id": "quant_with_tool_compute",
            "allowed_tool_names": ["get_price_history"],
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        pending_tool_call={
            "name": "get_price_history",
            "arguments": {"entity": "MSFT", "time_frame": "1M", "as_of": "2024-10-14"},
        },
    )

    result = asyncio.run(runner(state))

    assert node.seen_pending["arguments"]["ticker"] == "MSFT"
    assert node.seen_pending["arguments"]["period"] == "1mo"
    assert node.seen_pending["arguments"]["as_of_date"] == "2024-10-14"


def test_tool_runner_rewrites_native_message_tool_args_before_invoke():
    node = _CapturingMessageToolNode(
        "get_price_history",
        {
            "type": "get_price_history",
            "facts": {"ticker": "MSFT", "period": "1mo"},
            "source": {"tool": "get_price_history", "provider": "yfinance"},
            "quality": {"cache_hit": False, "is_synthetic": False, "is_estimated": False, "missing_fields": []},
            "errors": [],
        },
    )
    runner = make_tool_runner(node)
    state = make_state(
        "As of 2024-10-14, retrieve MSFT price history.",
        task_profile="finance_quant",
        execution_template={
            "template_id": "quant_with_tool_compute",
            "allowed_tool_names": ["get_price_history"],
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        pending_tool_call={
            "name": "get_price_history",
            "arguments": {"symbol": "MSFT", "as_of": "2024-10-14", "time_frame": "1M"},
        },
    )
    state["messages"].append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_price_history",
                    "args": {"symbol": "MSFT", "as_of": "2024-10-14", "time_frame": "1M"},
                    "id": "call_123",
                    "type": "tool_call",
                }
            ],
        )
    )

    asyncio.run(runner(state))

    assert node.seen_tool_call["args"]["ticker"] == "MSFT"
    assert node.seen_tool_call["args"]["period"] == "1mo"
    assert node.seen_tool_call["args"]["as_of_date"] == "2024-10-14"


def test_tool_runner_normalizes_pct_change_aliases_before_invoke():
    node = _CapturingToolNode(
        "pct_change",
        {
            "type": "pct_change",
            "facts": {"percentage_change": -2.82},
            "source": {"tool": "pct_change", "provider": "local_analytics"},
            "quality": {"cache_hit": False, "is_synthetic": False, "is_estimated": False, "missing_fields": []},
            "errors": [],
        },
    )
    runner = make_tool_runner(node)
    state = make_state(
        "Compute the change between start and end price.",
        task_profile="finance_quant",
        execution_template={
            "template_id": "quant_with_tool_compute",
            "allowed_tool_names": ["pct_change"],
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        pending_tool_call={
            "name": "pct_change",
            "arguments": {"start": 426.35, "end": 414.29},
        },
    )

    asyncio.run(runner(state))

    assert node.seen_pending["arguments"]["old_value"] == 426.35
    assert node.seen_pending["arguments"]["new_value"] == 414.29
