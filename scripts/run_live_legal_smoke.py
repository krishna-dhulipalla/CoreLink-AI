"""
Live Legal Smoke
================
Runs only the benchmark-shaped legal prompt variants through the active staged
runtime so legal reviewer/reflection changes can be debugged without running
the full live smoke suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_RUNS_ENDPOINTS", "")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TRACE_DIR = ROOT / "Results&traces"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.graph import build_agent_graph
from agent.runner import run_agent_trace
from agent.runtime_version import get_runtime_version
from mcp_client import load_mcp_tools_from_env


PROMPTS = [
    (
        "legal_transactional_benchmark",
        "target company we're acquiring has some clean IP but also regulatory compliance gaps in EU and US. "
        "ther board wants stock considerafton for tax reasons but we can't risk inheriting the compliance liabilities. "
        "deal size is ~$500M, what structure options do we have that could work for both sides? Need to move quicly here.",
    ),
    (
        "legal_transactional_clean",
        "Target company we're acquiring has some clean IP but also regulatory compliance gaps in EU and US. "
        "The board wants stock consideration for tax reasons, but we cannot risk inheriting the compliance liabilities. "
        "Deal size is about $500M. What structure options do we have that could work for both sides if we need to move quickly?",
    ),
]


def _summarize(trace: dict) -> dict:
    state = trace["final_state"]
    workpad = state.get("workpad", {})
    return {
        "runtime_version": get_runtime_version(),
        "answer": trace["answer"],
        "task_profile": state.get("task_profile"),
        "capability_flags": state.get("capability_flags"),
        "ambiguity_flags": state.get("ambiguity_flags"),
        "execution_template": (state.get("execution_template") or {}).get("template_id"),
        "task_intent": state.get("task_intent", {}),
        "tool_plan": state.get("tool_plan", {}),
        "curated_context": state.get("curated_context", {}),
        "quality_report": state.get("quality_report", {}),
        "solver_stage": state.get("solver_stage"),
        "review_results": workpad.get("review_results", []),
        "self_reflection_results": workpad.get("self_reflection_results", []),
        "events": workpad.get("events", []),
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
    payload = {
        "ok": True,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "runtime_version": get_runtime_version(),
        "loaded_tools": sorted(getattr(tool, "name", "") for tool in tools if getattr(tool, "name", "")),
        "results": results,
    }
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = TRACE_DIR / f"live_legal_smoke_{stamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "saved_path": str(output_path), "result_count": len(results)}, ensure_ascii=True))


if __name__ == "__main__":
    asyncio.run(main())
