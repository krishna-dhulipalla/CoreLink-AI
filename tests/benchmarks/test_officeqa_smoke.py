import os
import sys
import ast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

pytest.importorskip("mcp.server.fastmcp")

from agent.graph import build_agent_graph
from mcp_servers.document_analytics.server import (
    extract_pdf_tables,
    search_document_pages,
    sum_column,
    get_table_rows,
    _clear_table_cache,
)


# Re-wrap FastMCP tools as LangChain tools for immediate graph injection
lc_extract_pdf_tables = tool(extract_pdf_tables)
lc_search_document_pages = tool(search_document_pages)
lc_sum_column = tool(sum_column)
lc_get_table_rows = tool(get_table_rows)

_reasoner_call_count = 0


def mock_llm_reasoner_officeqa(messages):
    global _reasoner_call_count
    _reasoner_call_count += 1

    if _reasoner_call_count == 1:
        # Ask to extract tables from the document
        return AIMessage(
            content="",
            tool_calls=[{"name": "extract_pdf_tables", "args": {"file_path": "dummy_treasury.pdf", "pages": [1]}, "id": "tc1", "type": "tool_call"}],
        )
    if _reasoner_call_count == 2:
        tool_msg = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
        payload = ast.literal_eval(tool_msg.content)
        table_id = payload[0]["table_id"]
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "sum_column",
                "args": {"table_id": table_id, "column_matcher": "Total Debt"},
                "id": "tc2",
                "type": "tool_call",
            }],
        )
    tool_msg = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
    payload = ast.literal_eval(tool_msg.content)
    return AIMessage(
        content=(
            "Based on the Treasury Bulletin table matching 'Total Debt', "
            f"the exact aggregated value is {payload['sum']}."
        ),
        tool_calls=[]
    )


class TestOfficeQASmoke:
    def setup_method(self):
        global _reasoner_call_count
        _reasoner_call_count = 0
        _clear_table_cache()

    @pytest.mark.anyio
    @patch("agent.nodes.reasoner.ChatOpenAI")
    @patch("agent.nodes.verifier.ChatOpenAI")
    @patch("agent.nodes.coordinator.ChatOpenAI")
    @patch("mcp_servers.document_analytics.server._get_pdfplumber")
    async def test_officeqa_table_aggregation(self, mock_get_pdfplumber, mock_coord, mock_verif, mock_reas):
        # 1. Mock the PDF extractor to pretend we found a table
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        # Mock table: header row, data rows
        mock_page.extract_tables.return_value = [
            [["Category", "Total Debt", "Interest"], ["Public", "50000.0", "10"], ["Inter-gov", "4321.0", "5"]]
        ]
        mock_pdf.pages = [mock_page]
        mock_get_pdfplumber.return_value.open.return_value.__enter__.return_value = mock_pdf
        
        # Override the random UUID trick using side_effect or mocking uuid4, or easier: 
        # intercept the result of extract_pdf_tables in the graph? 
        with patch("mcp_servers.document_analytics.server.uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "112233445566"
            
            # 2. Mock Agent Graph Routing
            mock_coord.return_value.with_structured_output.return_value.invoke.return_value = MagicMock(
                layers=["react_reason", "verifier_check"],
                needs_formatting=False,
                confidence=0.9,
                estimated_steps=4,
                early_exit_allowed=True,
            )
            mock_reas.return_value.bind_tools.return_value.invoke.side_effect = mock_llm_reasoner_officeqa
            
            # Mock verifier to just PASS
            mock_verif.return_value.with_structured_output.return_value.invoke.return_value = MagicMock(verdict="PASS", reasoning="OK")
    
            tools = [lc_extract_pdf_tables, lc_search_document_pages, lc_sum_column, lc_get_table_rows]
            graph = build_agent_graph(external_tools=tools)
            
            initial_state = {
                "messages": [HumanMessage(content="What is the exact Total Debt aggregated value on page 1 of dummy_treasury.pdf?")],
                "reflection_count": 0, "tool_fail_count": 0, "last_tool_signature": "",
                "selected_layers": [], "format_required": False, "policy_confidence": 0.0,
                "estimated_steps": 0, "early_exit_allowed": False, "architecture_trace": [],
                "checkpoint_stack": [], "cost_tracker": None, "budget_tracker": None,
                "memory_store": None, "pending_verifier_feedback": None,
            }
    
            final_state = await graph.ainvoke(initial_state)
            msgs = final_state["messages"]
            
            # Assertions
            # Expect multiple tool messages (extract, sum)
            tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
            assert len(tool_msgs) == 2
            assert tool_msgs[0].name == "extract_pdf_tables"
            assert tool_msgs[1].name == "sum_column"
            
            # The sum calculation is deterministic and should be 54321.0
            assert "54321.0" in tool_msgs[1].content
            
            # Final AIMessage
            final_ai = msgs[-1]
            assert isinstance(final_ai, AIMessage)
            assert "54321.0" in final_ai.content
