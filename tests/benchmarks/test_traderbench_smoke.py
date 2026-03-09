import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

pytest.importorskip("mcp.server.fastmcp")

from agent.graph import build_agent_graph
from mcp_servers.market_data.server import (
    get_price_history,
    get_company_fundamentals,
)
from mcp_servers.finance_analytics.server import cagr


lc_get_price_history = tool(get_price_history)
lc_get_fundamentals = tool(get_company_fundamentals)
lc_cagr = tool(cagr)


_tb_reasoner_count = 0


def mock_llm_reasoner_traderbench(messages):
    global _tb_reasoner_count
    _tb_reasoner_count += 1

    if _tb_reasoner_count == 1:
        # Ask to fetch AAPL history
        return AIMessage(
            content="",
            tool_calls=[{"name": "get_price_history", "args": {"ticker": "AAPL", "period": "5y", "interval": "1wk"}, "id": "tcb1", "type": "tool_call"}],
        )
    if _tb_reasoner_count == 2:
        # Assume it saw prices and now calculates 5yr CAGR 
        # (Start price 100 -> End price 200)
        return AIMessage(
            content="",
            tool_calls=[{"name": "cagr", "args": {"beginning_value": 100.0, "ending_value": 200.0, "years": 5}, "id": "tcb2", "type": "tool_call"}],
        )
    # Give the final answer
    return AIMessage(
        content="The 5-year CAGR for AAPL is 14.8698%.",
        tool_calls=[]
    )


class TestTraderBenchSmoke:
    def setup_method(self):
        global _tb_reasoner_count
        _tb_reasoner_count = 0

    @pytest.mark.anyio
    @patch("agent.nodes.reasoner.ChatOpenAI")
    @patch("agent.nodes.verifier.ChatOpenAI")
    @patch("agent.nodes.coordinator.ChatOpenAI")
    @patch("mcp_servers.market_data.server._get_yfinance")
    async def test_traderbench_cagr_calculation(self, mock_get_yf, mock_coord, mock_verif, mock_reas):
        # 1. Mock the yfinance Ticker to return synthetic history
        mock_stock = MagicMock()
        # Create a tiny DataFrame that looks like history output
        dates = pd.date_range("2019-01-01", periods=2)
        df = pd.DataFrame({
            "Open": [99.0, 199.0],
            "High": [101.0, 201.0],
            "Low": [98.0, 198.0],
            "Close": [100.0, 200.0],  # Start 100, End 200
            "Volume": [1000, 2000]
        }, index=dates)
        
        mock_stock.history.return_value = df
        mock_get_yf.return_value = MagicMock(Ticker=MagicMock(return_value=mock_stock))
        
        # 2. Mock Agent Graph Routing
        mock_coord.return_value.with_structured_output.return_value.invoke.return_value = MagicMock(
            layers=["react_reason", "verifier_check"],
            needs_formatting=False,
            confidence=0.9,
            estimated_steps=4,
            early_exit_allowed=True,
        )
        mock_reas.return_value.bind_tools.return_value.invoke.side_effect = mock_llm_reasoner_traderbench
        mock_verif.return_value.with_structured_output.return_value.invoke.return_value = MagicMock(verdict="PASS", reasoning="OK")

        tools = [lc_get_price_history, lc_get_fundamentals, lc_cagr]
        graph = build_agent_graph(external_tools=tools)
        
        initial_state = {
            "messages": [HumanMessage(content="What is the 5-year CAGR of AAPL based on its actual history?")],
            "reflection_count": 0, "tool_fail_count": 0, "last_tool_signature": "",
            "selected_layers": [], "format_required": False, "policy_confidence": 0.0,
            "estimated_steps": 0, "early_exit_allowed": False, "architecture_trace": [],
            "checkpoint_stack": [], "cost_tracker": None, "budget_tracker": None,
            "memory_store": None, "pending_verifier_feedback": None,
        }

        final_state = await graph.ainvoke(initial_state)
        msgs = final_state["messages"]
        
        # Assertions
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].name == "get_price_history"
        assert tool_msgs[1].name == "cagr"
        
        # The CAGR is strictly calculated by python
        cagr_val = "14.8698"  # exactly what the formula produces for 100->200 over 5y
        assert cagr_val in tool_msgs[1].content
        
        # Final answer contains the exact numeric match
        final_ai = msgs[-1]
        assert isinstance(final_ai, AIMessage)
        assert cagr_val in final_ai.content
