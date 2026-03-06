"""
Graph Builder: Compiles the Multi-Agent StateGraph
====================================================
Wires together all nodes into the final LangGraph topology.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.nodes.reasoner import make_reasoner
from agent.nodes.coordinator import coordinator, route_task, direct_responder, format_normalizer
from agent.nodes.reflector import reflector, should_revise
from agent.nodes.tool_executor import should_use_tools, make_tool_executor
from agent.nodes.context import context_window
from tools import CALCULATOR_TOOL, SEARCH_TOOL

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in Tools (always available regardless of MCP servers)
# ---------------------------------------------------------------------------

@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


BUILTIN_TOOLS: list[Any] = [CALCULATOR_TOOL, SEARCH_TOOL, get_current_time]


# ---------------------------------------------------------------------------
# Graph Factory
# ---------------------------------------------------------------------------

def build_agent_graph(external_tools: list | None = None):
    """Construct and compile the LangGraph StateGraph (MaAS-Lite Architecture).

    Args:
        external_tools: Optional list of LangChain tools loaded from MCP.

    Graph topology (Sprint 1.5):
        coordinator ──(direct_answer)────▶ direct_responder ──▶ format_normalizer* ──▶ END
             │
             └─(react_reason)──▶ reasoner ──(tools?)──▶ tool_executor ──▶ context_window ──▶ reasoner
                                     │
                                     └─(no tools)───▶ reflector ──(PASS)──▶ format_normalizer* ──▶ END
                                                                   │
                                                               (REVISE) ──▶ reasoner

        * format_normalizer is CONDITIONAL: skips LLM call when format_required=False
    """
    all_tools = BUILTIN_TOOLS + (external_tools or [])
    raw_tool_node = ToolNode(all_tools)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("coordinator", coordinator)
    graph.add_node("direct_responder", direct_responder)
    graph.add_node("format_normalizer", format_normalizer)

    from agent.nodes.verifier import verifier, verify_routing
    graph.add_node("reasoner", make_reasoner(all_tools))
    graph.add_node("tool_executor", make_tool_executor(raw_tool_node))
    graph.add_node("context_window", context_window)
    graph.add_node("reflector", reflector) # Keep for older plans
    graph.add_node("verifier", verifier)

    # 1. Entry Point
    graph.set_entry_point("coordinator")

    # 2. Routing from Coordinator (layered policy → first operator)
    graph.add_conditional_edges("coordinator", route_task)

    # 3. Direct responders go to format normalizer (which may skip)
    graph.add_edge("direct_responder", "format_normalizer")

    # 4. Executor-Verifier Triad
    # reasoner emits tool calls or final answer
    graph.add_conditional_edges("reasoner", should_use_tools)
    
    # tools execute, context trims, then we verify the step
    graph.add_edge("tool_executor", "context_window")
    graph.add_edge("context_window", "verifier")

    # Verifier decides if PASS, REVISE, or BACKTRACK
    graph.add_conditional_edges("verifier", verify_routing)

    # 5. Reflection decides to loop or format (Legacy/Alternative operator)
    graph.add_conditional_edges("reflector", should_revise)

    # 6. Formatting is always the terminal node (but may no-op)
    graph.add_edge("format_normalizer", END)

    return graph.compile()
