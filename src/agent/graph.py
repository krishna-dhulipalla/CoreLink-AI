"""Graph builder for the active engine."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from agent.capabilities import BUILTIN_LEGAL_TOOLS, build_capability_registry
from agent.nodes.intake import intake
from agent.nodes.output_adapter import output_adapter
from agent.nodes.reflect import reflect
from agent.workflow_nodes import (
    context_curator,
    fast_path_gate,
    make_capability_resolver,
    make_executor,
    reviewer,
    route_from_executor,
    route_from_reviewer,
    route_from_self_reflection,
    self_reflection,
    task_planner,
)
from agent.workflow_state import RuntimeState
from tools import CALCULATOR_TOOL, SEARCH_TOOL

load_dotenv(override=False)


@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def build_agent_graph(external_tools: list | None = None):
    all_tools: list[Any] = [CALCULATOR_TOOL, SEARCH_TOOL, get_current_time, *BUILTIN_LEGAL_TOOLS, *(external_tools or [])]
    registry = build_capability_registry(all_tools)

    graph = StateGraph(RuntimeState)
    graph.add_node("intake", intake)
    graph.add_node("fast_path_gate", fast_path_gate)
    graph.add_node("task_planner", task_planner)
    graph.add_node("capability_resolver", make_capability_resolver(registry))
    graph.add_node("context_curator", context_curator)
    graph.add_node("executor", make_executor(registry))
    graph.add_node("reviewer", reviewer)
    graph.add_node("self_reflection", self_reflection)
    graph.add_node("output_adapter", output_adapter)
    graph.add_node("reflect", reflect)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "fast_path_gate")
    graph.add_edge("fast_path_gate", "task_planner")
    graph.add_edge("task_planner", "capability_resolver")
    graph.add_edge("capability_resolver", "context_curator")
    graph.add_edge("context_curator", "executor")
    graph.add_conditional_edges("executor", route_from_executor)
    graph.add_conditional_edges("reviewer", route_from_reviewer)
    graph.add_conditional_edges("self_reflection", route_from_self_reflection)
    graph.add_edge("output_adapter", "reflect")
    graph.add_edge("reflect", END)
    return graph.compile()
