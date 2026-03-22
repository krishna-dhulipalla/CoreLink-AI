"""
Purple Agent: Multi-Agent Reasoning Engine
============================================
Public API for the LangGraph-based agent.

Usage:
    from agent import build_agent_graph, run_agent
"""

from agent.graph import build_agent_graph
from agent.runner import run_agent
from agent.state import AgentState

__all__ = ["build_agent_graph", "run_agent", "AgentState"]
