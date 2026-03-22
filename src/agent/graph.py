"""Graph builder for the active runtime."""

from __future__ import annotations

from dotenv import load_dotenv
from agent.v4.graph import build_runtime_graph

load_dotenv(override=False)

def build_agent_graph(external_tools: list | None = None):
    return build_runtime_graph(external_tools=external_tools)
