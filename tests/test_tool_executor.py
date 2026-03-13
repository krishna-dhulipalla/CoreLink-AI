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
