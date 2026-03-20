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
    original_tool_mode = solver_module._tool_call_mode
    original_reviewer_invoke = reviewer_module.invoke_structured_output
    try:
        solver_module.ChatOpenAI = lambda **kwargs: _FakeModel(executor_queue)
        solver_module._tool_call_mode = lambda role: "prompt"

        def _fake_reviewer_invoke(role, schema, messages, temperature=0, max_tokens=0):
            if not reviewer_queue:
                raise RuntimeError("Fake reviewer queue exhausted.")
            response = reviewer_queue.pop(0)
            content = str(getattr(response, "content", ""))
            return schema.model_validate_json(content), content

        reviewer_module.invoke_structured_output = _fake_reviewer_invoke
        yield
    finally:
        solver_module.ChatOpenAI = original_solver_chat
        solver_module._tool_call_mode = original_tool_mode
        reviewer_module.invoke_structured_output = original_reviewer_invoke


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


@tool
def scenario_pnl(
    net_premium: float,
    total_delta: float,
    total_gamma: float = 0.0,
    total_theta_per_day: float = 0.0,
    total_vega_per_vol_point: float = 0.0,
    reference_price: float = 100.0,
) -> dict:
    """Return a deterministic scenario P&L bundle for smoke testing."""
    return {
        "type": "scenario_pnl",
        "facts": {
            "scenarios": [
                {"name": "base", "approx_pnl": 2.5},
                {"name": "stress", "approx_pnl": -18.4},
            ],
            "worst_case_pnl": -18.4,
            "worst_case_scenario": "stress",
            "stress_loss_ratio": 2.0,
        },
        "assumptions": {
            "net_premium": net_premium,
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_theta_per_day": total_theta_per_day,
            "total_vega_per_vol_point": total_vega_per_vol_point,
            "reference_price": reference_price,
        },
        "source": {"tool": "scenario_pnl", "provider": "local_risk"},
        "quality": {"is_synthetic": False, "is_estimated": True, "cache_hit": False, "missing_fields": []},
        "errors": [],
    }


@tool
def fetch_reference_file(url: str, row_limit: int = 25, page_limit: int = 2) -> str:
    """Return a deterministic document preview for smoke testing."""
    if url.endswith(".csv"):
        return (
            "FILE: report.csv\n"
            "FORMAT: CSV | SIZE: 4.2 KB\n"
            "--------------------------------------------------\n"
            "[Rows 0-3 of ~3]\n"
            "metric,value\n"
            "roe,3.0433\n"
            "roa,1.5791\n"
        )
    return (
        "FILE: report.pdf\n"
        "FORMAT: PDF | SIZE: 120.5 KB\n"
        "--------------------------------------------------\n"
        "[Pages 1-2 of 10]\n"
        "Covenant threshold is 4.5x net leverage. Cure period is 30 days."
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
    assert any("template=quant_inline_exact" in event.get("action", "") for event in steps if event.get("node") == "template_selector")
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
        AIMessage(content="Primary analysis: net premium +9.16 credit, delta -0.12, theta +0.04/day, and vega -0.06 support a short-volatility bias. Use 1% position sizing and a stop-loss outside the breakeven range."),
        AIMessage(content="Updated risk analysis: stress scenario P&L is -18.4, so keep 1% position sizing, explicit stop-loss, and disclose short-volatility gap risk."),
        AIMessage(
            content=(
                "Recommendation: be a net seller of options.\n"
                "Primary strategy: short strangle with net credit and positive theta, assuming spot is 300.\n"
                "Alternative strategy: iron condor for lower tail risk at the cost of lower premium.\n"
                "Key Greeks: delta near flat, negative gamma, positive theta, negative vega.\n"
                "Breakevens: manage around the short strikes plus or minus collected credit.\n"
                "Risk management: small sizing, defined stop-loss on large underlying moves, and max-loss awareness.\n"
                "Short vol disclosure: short-volatility exposure can lose sharply in a vol spike and carries gap risk."
            )
        ),
        AIMessage(
            content=(
                "Recommendation: be a net seller of options.\n"
                "Primary strategy: short strangle with net credit and positive theta, assuming spot is 300.\n"
                "Alternative strategy: iron condor for lower tail risk at the cost of lower premium.\n"
                "Key Greeks: delta near flat, negative gamma, positive theta, negative vega.\n"
                "Breakevens: manage around the short strikes plus or minus collected credit.\n"
                "Risk management: 1% position sizing, defined stop-loss on large underlying moves, and max-loss awareness.\n"
                "Downside scenario: the stress scenario loses about $18.4, so this is a scenario-dependent recommendation.\n"
                "Short vol disclosure: short-volatility exposure can lose sharply in a vol spike and carries gap risk."
            )
        ),
    ]
    reviewer_queue = [
        AIMessage(content='{"verdict":"pass","reasoning":"Compute output is concrete.","missing_dimensions":[],"repair_target":"synthesize"}'),
        AIMessage(content='{"verdict":"pass","reasoning":"Final answer is complete.","missing_dimensions":[],"repair_target":"final"}'),
        AIMessage(content='{"verdict":"pass","reasoning":"Revised final answer is complete.","missing_dimensions":[],"repair_target":"final"}'),
    ]
    graph = build_agent_graph(external_tools=[analyze_strategy, scenario_pnl])
    prompt = (
        "META's current IV is 35% while its 30-day historical volatility is 28%. "
        "The IV percentile is 75%. Should you be a net buyer or seller of options?"
    )
    with patched_models(executor_queue, reviewer_queue):
        answer, steps, _ = await run_agent(graph, prompt)
    assert "net seller" in answer.lower() or "seller" in answer.lower()
    assert any(event["action"].startswith("ran analyze_strategy") for event in steps if event["node"] == "tool_runner")
    assert any("template=options_tool_backed" in event.get("action", "") for event in steps if event.get("node") == "template_selector")
    return {"scenario": "finance_options", "answer": answer, "steps": steps}


async def run_document_scenario():
    executor_queue = [
        AIMessage(
            content=json.dumps(
                {
                    "name": "fetch_reference_file",
                    "arguments": {
                        "url": "https://example.com/report.csv",
                        "row_limit": 25,
                    },
                }
            )
        ),
        AIMessage(
            content=(
                "Extracted document evidence: the CSV preview contains two metrics, ROE 3.0433 and ROA 1.5791, "
                "from the first table window of the source file."
            )
        ),
        AIMessage(
            content=(
                "Answer: the extracted CSV preview shows ROE 3.0433 and ROA 1.5791.\n"
                "Evidence summary: rows were extracted from the file preview rather than guessed from narrative.\n"
                "Source references: https://example.com/report.csv"
            )
        ),
    ]
    reviewer_queue = [
        AIMessage(content='{"verdict":"pass","reasoning":"Gather output contains structured document evidence.","missing_dimensions":[],"repair_target":"synthesize"}'),
        AIMessage(content='{"verdict":"pass","reasoning":"Final answer is grounded and complete.","missing_dimensions":[],"repair_target":"final"}'),
    ]
    graph = build_agent_graph(external_tools=[fetch_reference_file])
    prompt = "Read the attached CSV at https://example.com/report.csv and summarize the extracted values with a source reference."
    with patched_models(executor_queue, reviewer_queue):
        answer, steps, _ = await run_agent(graph, prompt)
    assert "source references" in answer.lower()
    assert any(event["action"].startswith("ran fetch_reference_file") for event in steps if event["node"] == "tool_runner")
    assert any("template=document_qa" in event.get("action", "") for event in steps if event.get("node") == "template_selector")
    return {"scenario": "document_qa", "answer": answer, "steps": steps}


async def main():
    quant = await run_quant_scenario()
    options = await run_options_scenario()
    document = await run_document_scenario()
    print(json.dumps({"ok": True, "scenarios": [quant, options, document]}, ensure_ascii=True))


if __name__ == "__main__":
    asyncio.run(main())
