"""
Live Staged Runtime Smoke
=========================
Runs a few representative prompts through the active staged runtime with the
configured live LLM and tool surface, then prints compact trace summaries.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_RUNS_ENDPOINTS", "")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.graph import build_agent_graph
from agent.runner import run_agent_trace
from mcp_client import load_mcp_tools_from_env


PROMPTS = [
    (
        "finance_quant",
        "Financial Leverage Effect = (ROE - ROA) / ROA. "
        "Calculate the Financial Leverage Effect for China Overseas Grand Oceans Group. ROE = 3.0433%, ROA = 1.579%. "
        'Output Format: {"answer": <value>}',
    ),
    (
        "finance_options",
        "META's current IV is 35% while its 30-day historical volatility is 28%. "
        "The IV percentile is 75%. Should you be a net buyer or seller of options? Design a strategy accordingly.",
    ),
    (
        "legal_transactional",
        "Target company we're acquiring has clean IP but regulatory compliance gaps in EU and US. "
        "Board wants stock consideration for tax reasons but we cannot inherit the compliance liabilities. "
        "Deal size is about $500M. What structure options could work quickly for both sides?",
    ),
    (
        "mixed_legal_math",
        "We are acquiring a target with EU compliance gaps and need a fast structure recommendation, "
        "but also quantify a holdback formula tied to EBITDA and current liabilities. "
        "Do not browse unless the prompt truly requires it.",
    ),
]


def _summarize(trace: dict) -> dict:
    state = trace["final_state"]
    workpad = state.get("workpad", {})
    return {
        "answer": trace["answer"],
        "task_profile": state.get("task_profile"),
        "capability_flags": state.get("capability_flags"),
        "ambiguity_flags": state.get("ambiguity_flags"),
        "execution_template": (state.get("execution_template") or {}).get("template_id"),
        "solver_stage": state.get("solver_stage"),
        "events": workpad.get("events", []),
        "tool_results": workpad.get("tool_results", []),
        "cost_summary": state.get("cost_tracker").summary() if state.get("cost_tracker") else {},
    }


async def main() -> None:
    tools = await load_mcp_tools_from_env()
    graph = build_agent_graph(external_tools=tools)
    results = []
    for label, prompt in PROMPTS:
        try:
            trace = await run_agent_trace(graph, prompt)
            results.append({"label": label, **_summarize(trace)})
        except Exception as exc:
            results.append({"label": label, "error": str(exc)})
    print(json.dumps({"ok": True, "results": results}, ensure_ascii=True))


if __name__ == "__main__":
    asyncio.run(main())
