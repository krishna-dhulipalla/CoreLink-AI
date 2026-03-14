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
