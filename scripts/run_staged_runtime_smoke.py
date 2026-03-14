"""
Deterministic staged-runtime smoke
=================================
Exercises the staged graph end to end with mocked model responses and a small
tool surface so graph-path regressions can be caught without rerunning a full
benchmark slice.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_RUNS_ENDPOINTS", "")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent.graph import build_agent_graph
from agent.runner import run_agent
from agent.nodes import reviewer as reviewer_module
from agent.nodes import solver as solver_module


class _FakeModel:
    def __init__(self, queue):
        self._queue = queue

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if not self._queue:
            raise RuntimeError("Fake model queue exhausted.")
        return self._queue.pop(0)


@contextmanager
def patched_models(executor_queue, reviewer_queue):
    original_solver_chat = solver_module.ChatOpenAI
    original_reviewer_chat = reviewer_module.ChatOpenAI
    original_tool_mode = solver_module._tool_call_mode
    try:
        solver_module.ChatOpenAI = lambda **kwargs: _FakeModel(executor_queue)
        reviewer_module.ChatOpenAI = lambda **kwargs: _FakeModel(reviewer_queue)
        solver_module._tool_call_mode = lambda role: "prompt"
        yield
    finally:
        solver_module.ChatOpenAI = original_solver_chat
        reviewer_module.ChatOpenAI = original_reviewer_chat
        solver_module._tool_call_mode = original_tool_mode


@tool
def analyze_strategy(legs: list[dict]) -> str:
    """Return a deterministic options analysis block for smoke testing."""
    return (
        "Multi-Leg Strategy (2 legs):\n"
        "  Net Premium  : +9.16 (CREDIT)\n"
        "  Total Delta  : -0.1200\n"
        "  Total Gamma  : -0.0010\n"
        "  Total Theta  : +0.0400 /day\n"
        "  Total Vega   : -0.0600 per 1% vol\n"
    )


async def run_quant_scenario():
    executor_queue = [
        AIMessage(content='{"name":"calculator","arguments":{"expression":"round((3.0433-1.5791)/1.5791, 4)"}}'),
        AIMessage(content="Computed leverage effect is 0.9274 from prompt-contained ROE and ROA values."),
        AIMessage(content="0.9274"),
    ]
    reviewer_queue = [
        AIMessage(content='{"verdict":"pass","reasoning":"Compute output is concrete.","missing_dimensions":[],"repair_target":"synthesize"}'),
        AIMessage(content='{"verdict":"pass","reasoning":"Final answer is complete.","missing_dimensions":[],"repair_target":"final"}'),
    ]
    graph = build_agent_graph()
    prompt = (
        "Calculate the Financial Leverage Effect for China Overseas Grand Oceans Group.\n"
        "ROE = 3.0433 %, ROA = 1.5791 %.\n"
        'Output Format: {"answer": <value>}'
    )
    with patched_models(executor_queue, reviewer_queue):
        answer, steps, _ = await run_agent(graph, prompt)
    assert answer == '{"answer": 0.9274}'
    assert any(event["node"] == "tool_runner" for event in steps)
    return {"scenario": "finance_quant", "answer": answer, "steps": steps}


async def run_options_scenario():
    executor_queue = [
        AIMessage(
            content=json.dumps(
                {
                    "name": "analyze_strategy",
                    "arguments": {
                        "legs": [
                            {"option_type": "call", "action": "sell", "S": 300, "K": 310, "T_days": 30, "r": 0.05, "sigma": 0.35, "contracts": 1},
                            {"option_type": "put", "action": "sell", "S": 300, "K": 290, "T_days": 30, "r": 0.05, "sigma": 0.35, "contracts": 1},
                        ]
                    },
                }
            )
        ),
        AIMessage(content="Primary analysis: net premium +9.16 credit, delta -0.12, theta +0.04/day, and vega -0.06 support a short-volatility bias."),
        AIMessage(
            content=(
                "Recommendation: be a net seller of options.\n"
                "Primary strategy: short strangle with net credit and positive theta.\n"
                "Alternative strategy: iron condor for lower tail risk at the cost of lower premium.\n"
                "Key Greeks: delta near flat, negative gamma, positive theta, negative vega.\n"
                "Breakevens: manage around the short strikes plus or minus collected credit.\n"
                "Risk management: small sizing, defined stop-loss on large underlying moves, and max-loss awareness."
            )
        ),
    ]
    reviewer_queue = [
        AIMessage(content='{"verdict":"pass","reasoning":"Compute output is concrete.","missing_dimensions":[],"repair_target":"synthesize"}'),
        AIMessage(content='{"verdict":"pass","reasoning":"Final answer is complete.","missing_dimensions":[],"repair_target":"final"}'),
    ]
    graph = build_agent_graph(external_tools=[analyze_strategy])
    prompt = (
        "META's current IV is 35% while its 30-day historical volatility is 28%. "
        "The IV percentile is 75%. Should you be a net buyer or seller of options?"
    )
    with patched_models(executor_queue, reviewer_queue):
        answer, steps, _ = await run_agent(graph, prompt)
    assert "net seller" in answer.lower() or "seller" in answer.lower()
    assert any(event["action"].startswith("ran analyze_strategy") for event in steps if event["node"] == "tool_runner")
    return {"scenario": "finance_options", "answer": answer, "steps": steps}


async def main():
    quant = await run_quant_scenario()
    options = await run_options_scenario()
    print(json.dumps({"ok": True, "scenarios": [quant, options]}, ensure_ascii=True))


if __name__ == "__main__":
    asyncio.run(main())
