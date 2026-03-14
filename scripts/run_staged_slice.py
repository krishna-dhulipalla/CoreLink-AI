"""
Replay Saved Task Prompts Through The Staged Runtime
===================================================
Uses the saved LangSmith task inputs as a local benchmark-aligned slice, but
runs them through the current staged runtime rather than the retired graph.
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


def _extract_prompt(task_path: Path) -> str:
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    messages = payload.get("inputs", {}).get("messages", [])
    for message in messages:
        if message.get("type") == "human" and message.get("content"):
            return str(message["content"])
    raise ValueError(f"No human prompt found in {task_path}")


def _final_state_summary(trace: dict) -> dict:
    state = trace.get("final_state", {})
    workpad = state.get("workpad", {})
    return {
        "task_profile": state.get("task_profile"),
        "capability_flags": state.get("capability_flags"),
        "solver_stage": state.get("solver_stage"),
        "stage_history": workpad.get("stage_history", []),
        "events": workpad.get("events", []),
        "tool_results": workpad.get("tool_results", []),
        "answer_contract": state.get("answer_contract", {}),
        "cost_summary": state.get("cost_tracker").summary() if state.get("cost_tracker") else {},
        "budget_summary": state.get("budget_tracker").summary() if state.get("budget_tracker") else {},
    }


async def main() -> None:
    tools = await load_mcp_tools_from_env()
    graph = build_agent_graph(external_tools=tools)
    task_files = [
        ROOT / "langsmith_runs" / "task1.json",
        ROOT / "langsmith_runs" / "task2.json",
        ROOT / "langsmith_runs" / "task3.json",
    ]

    results = []
    for task_file in task_files:
        prompt = _extract_prompt(task_file)
        try:
            trace = await run_agent_trace(graph, prompt)
            results.append(
                {
                    "task_file": task_file.name,
                    "answer": trace["answer"],
                    "trace": _final_state_summary(trace),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "task_file": task_file.name,
                    "error": str(exc),
                }
            )

    output_path = ROOT / "langsmith_runs" / "staged_slice_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"ok": True, "output_path": str(output_path), "tasks": [r["task_file"] for r in results]}))


if __name__ == "__main__":
    asyncio.run(main())
