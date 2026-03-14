import json
import asyncio

from langchain_core.messages import ToolMessage

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
        pending_tool_call={"name": "analyze_strategy", "arguments": {"legs": []}},
    )

    result = asyncio.run(runner(state))
    normalized = result["last_tool_result"]

    assert normalized["facts"]["net_premium"] == 9.16
    assert normalized["facts"]["premium_direction"] == "CREDIT"
    assert normalized["facts"]["total_theta_per_day"] == 0.04
    assert result["messages"][0].name == "analyze_strategy"
    assert json.loads(result["messages"][0].content)["facts"]["total_delta"] == -0.12


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
        pending_tool_call={"name": "fetch_reference_file", "arguments": {"url": "https://example.com/report.csv"}},
    )

    result = asyncio.run(runner(state))
    normalized = result["last_tool_result"]

    assert normalized["facts"]["file_name"] == "report.csv"
    assert normalized["facts"]["format"] == "csv"
    assert normalized["facts"]["rows"][0] == ["metric", "value"]


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
