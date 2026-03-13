"""
Tool Executor Guardrail Tests
=============================
Focused regressions for malformed tool-call blocking and payload validation.
"""

import asyncio
import os
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.nodes.tool_executor import make_tool_executor


class _CreatePortfolioArgs(BaseModel):
    name: str
    initial_cash: float = 100000.0


class _DummyTool:
    def __init__(self, name, args_schema):
        self.name = name
        self.args_schema = args_schema


class _DummyToolNode:
    def __init__(self, tools_by_name):
        self.tools_by_name = tools_by_name
        self.called = False

    async def ainvoke(self, state):
        self.called = True
        return {"messages": [ToolMessage(content="unexpected", tool_call_id="call_x", name="dummy")]}


class _AnalyzeStrategyArgs(BaseModel):
    legs: list[dict]


def test_tool_executor_blocks_nested_tool_envelope_payload():
    tool = _DummyTool("create_portfolio", _CreatePortfolioArgs)
    node = _DummyToolNode({"create_portfolio": tool})
    tool_executor = make_tool_executor(node)

    state = {
        "messages": [
            HumanMessage(content="Create a portfolio and simulate an options trade."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_portfolio",
                        "args": {
                            "name": "internet_search",
                            "arguments": {"query": "acquisition structure"},
                        },
                        "id": "call_bad",
                        "type": "tool_call",
                    }
                ],
            ),
        ],
        "task_type": "options",
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "cost_tracker": None,
        "budget_tracker": None,
    }

    result = asyncio.run(tool_executor(state))

    assert node.called is False
    assert result["tool_fail_count"] == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert "Malformed tool payload" in result["messages"][0].content


def test_tool_executor_normalizes_common_finance_aliases_before_validation():
    tool = _DummyTool("analyze_strategy", _AnalyzeStrategyArgs)
    node = _DummyToolNode({"analyze_strategy": tool})
    tool_executor = make_tool_executor(node)

    state = {
        "messages": [
            HumanMessage(content="Analyze a short straddle."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_strategy",
                        "args": {
                            "legs": [
                                {
                                    "option_type": "call",
                                    "action": "sell",
                                    "spot": 300,
                                    "strike": 300,
                                    "days_to_expiration": 30,
                                    "risk_free_rate": 0.05,
                                    "volatility": 0.35,
                                }
                            ]
                        },
                        "id": "call_ok",
                        "type": "tool_call",
                    }
                ],
            ),
        ],
        "task_type": "options",
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "cost_tracker": None,
        "budget_tracker": None,
    }

    result = asyncio.run(tool_executor(state))

    assert node.called is True
    assert result["tool_fail_count"] == 0
    normalized_legs = state["messages"][-1].tool_calls[0]["args"]["legs"]
    assert normalized_legs[0]["S"] == 300
    assert normalized_legs[0]["K"] == 300
    assert normalized_legs[0]["T_days"] == 30
    assert normalized_legs[0]["r"] == 0.05
    assert normalized_legs[0]["sigma"] == 0.35
