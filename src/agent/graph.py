"""Graph builder for the active engine."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from agent.capabilities import BUILTIN_LEGAL_TOOLS, BUILTIN_RETRIEVAL_TOOLS, build_capability_registry, filter_registry_for_benchmark
from agent.langsmith_env import normalize_langsmith_env
from agent.nodes.intake import intake
from agent.nodes.orchestrator import (
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
from agent.nodes.output_adapter import output_adapter
from agent.nodes.reflect import reflect
from agent.retrieval_tools import local_corpus_available
from agent.state import AgentState
from tools import CALCULATOR_TOOL, SEARCH_TOOL

load_dotenv(override=False)
normalize_langsmith_env()


@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def build_agent_graph(external_tools: list | None = None):
    retrieval_tools = BUILTIN_RETRIEVAL_TOOLS if local_corpus_available() else []
    all_tools: list[Any] = [
        CALCULATOR_TOOL,
        get_current_time,
        *retrieval_tools,
        *BUILTIN_LEGAL_TOOLS,
        *(external_tools or []),
    ]
    if os.getenv("TAVILY_API_KEY", "").strip():
        all_tools.append(SEARCH_TOOL)
    registry = build_capability_registry(all_tools)
    registry = filter_registry_for_benchmark(
        registry,
        os.getenv("BENCHMARK_NAME", "").strip().lower(),
    )

    graph = StateGraph(AgentState)
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
