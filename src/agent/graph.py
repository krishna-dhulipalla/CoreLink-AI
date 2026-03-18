"""
Graph Builder
=============
Compiles the staged finance-first runtime graph.
"""

from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes.context_builder import context_builder
from agent.nodes.intake import intake
from agent.nodes.output_adapter import output_adapter
from agent.nodes.reflect import reflect
from agent.nodes.risk_controller import risk_controller, route_from_risk_controller
from agent.nodes.reviewer import reviewer, route_from_reviewer
from agent.nodes.solver import make_solver, route_from_solver
from agent.nodes.task_profiler import task_profiler
from agent.nodes.template_selector import template_selector
from agent.nodes.tool_runner import make_tool_runner
from agent.state import AgentState
from tools import CALCULATOR_TOOL, SEARCH_TOOL

load_dotenv(override=False)

logger = logging.getLogger(__name__)


@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


BUILTIN_TOOLS: list[Any] = [CALCULATOR_TOOL, SEARCH_TOOL, get_current_time]


def build_agent_graph(external_tools: list | None = None):
    """Construct and compile the staged finance-first LangGraph."""
    all_tools = BUILTIN_TOOLS + (external_tools or [])
    raw_tool_node = ToolNode(all_tools)

    graph = StateGraph(AgentState)
    graph.add_node("intake", intake)
    graph.add_node("task_profiler", task_profiler)
    graph.add_node("template_selector", template_selector)
    graph.add_node("context_builder", context_builder)
    graph.add_node("solver", make_solver(all_tools))
    graph.add_node("tool_runner", make_tool_runner(raw_tool_node))
    graph.add_node("risk_controller", risk_controller)
    graph.add_node("reviewer", reviewer)
    graph.add_node("output_adapter", output_adapter)
    graph.add_node("reflect", reflect)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "task_profiler")
    graph.add_edge("task_profiler", "template_selector")
    graph.add_edge("template_selector", "context_builder")
    graph.add_edge("context_builder", "solver")
    graph.add_conditional_edges("solver", route_from_solver)
    graph.add_edge("tool_runner", "solver")
    graph.add_conditional_edges("risk_controller", route_from_risk_controller)
    graph.add_conditional_edges("reviewer", route_from_reviewer)
    graph.add_edge("output_adapter", "reflect")
    graph.add_edge("reflect", END)

    return graph.compile()
