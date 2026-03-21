"""
Benchmark/stateless smoke
=========================
Validates that BENCHMARK_STATELESS=1 forces fresh-run behavior and still
produces traceable budget output.
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
os.environ["BENCHMARK_STATELESS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langchain_core.messages import AIMessage, HumanMessage

from agent.runner import run_agent_trace
from agent.runtime_version import get_runtime_version

TRACE_DIR = ROOT / "Results&traces"


class _StaticGraph:
    def __init__(self, final_messages):
        self.final_messages = final_messages
        self.initial_state = None

    async def astream(self, initial_state, config=None, stream_mode=None):
        self.initial_state = initial_state
        state = dict(initial_state)
        state["messages"] = list(self.final_messages)
        yield state


async def main():
    graph = _StaticGraph([HumanMessage(content="What is ROE?"), AIMessage(content='{"answer": 0.12}')])
    trace = await run_agent_trace(
        graph,
        "What is ROE?",
        history=[HumanMessage(content="old benchmark item"), AIMessage(content="old answer")],
    )

    initial_messages = [str(msg.content) for msg in graph.initial_state["messages"]]
    if initial_messages != ["What is ROE?"]:
        raise SystemExit(f"benchmark stateless mode reused history unexpectedly: {initial_messages}")

    updated_history = [str(msg.content) for msg in trace["updated_history"]]
    if updated_history != ["What is ROE?", '{"answer": 0.12}']:
        raise SystemExit(f"updated history shape unexpected: {updated_history}")

    budget_summary = next((step for step in trace["steps"] if step.get("node") == "budget_summary"), None)
    if not budget_summary:
        raise SystemExit("missing budget_summary step")

    if budget_summary.get("complexity_tier") != "structured_analysis":
        raise SystemExit(f"unexpected default complexity tier: {budget_summary.get('complexity_tier')}")

    payload = {
        "ok": True,
        "benchmark_stateless": True,
        "runtime_version": get_runtime_version(),
        "initial_messages": initial_messages,
        "updated_history": updated_history,
        "budget_summary": budget_summary,
    }
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = TRACE_DIR / f"benchmark_stateless_smoke_{stamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "saved_path": str(output_path), "runtime_version": get_runtime_version()}, ensure_ascii=True))


if __name__ == "__main__":
    asyncio.run(main())
