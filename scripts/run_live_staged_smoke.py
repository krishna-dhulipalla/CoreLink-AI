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
        "finance_options_policy",
        "META's current IV is 35% while its 30-day historical volatility is 28%. "
        "The IV percentile is 75%. This is a retirement account mandate: defined-risk only, "
        "no naked options, and keep position risk to 2% of capital. "
        "Should you be a net buyer or seller of options? Design a compliant strategy accordingly.",
    ),
    (
        "legal_transactional",
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
    (
        "mixed_legal_math",
        "We are acquiring a target with EU compliance gaps and need a fast structure recommendation, "
        "but also quantify a holdback formula tied to EBITDA and current liabilities. "
        "Do not browse unless the prompt truly requires it.",
    ),
    (
        "document_qa",
        "Read the reference file at https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv. "
        "Summarize what structured evidence you extracted from it and include the source reference.",
    ),
    (
        "finance_evidence",
        "As of 2024-10-14, use finance evidence tools to retrieve MSFT price history and 1-month return, "
        "then summarize the result with the source timestamp and any missing-data caveats.",
    ),
    (
        "equity_research",
        "Write an equity research report on MSFT as of 2024-10-14. Include thesis, evidence, valuation framing, and key risks.",
    ),
    (
        "portfolio_risk_review",
        "Review this portfolio risk and recommend actions.\n"
        'Portfolio JSON: [{"ticker":"AAPL","weight":0.35,"sector":"Technology","liquidation_days":6},'
        '{"ticker":"MSFT","weight":0.30,"sector":"Technology","liquidation_days":4},'
        '{"ticker":"XOM","weight":0.20,"sector":"Energy","liquidation_days":2},'
        '{"ticker":"JNJ","weight":0.15,"sector":"Healthcare","liquidation_days":3}]\n'
        'Returns JSON: [0.012,-0.018,0.009,-0.021,0.011,-0.007,0.006,-0.013,0.008,-0.005]\n'
        'Limits JSON: {"max_loss_pct":0.12,"max_var_pct":0.05}',
    ),
    (
        "event_driven_finance",
        "Evaluate the event-driven setup for MSFT around its next earnings catalyst as of 2024-10-14. "
        "Explain the catalyst, market context, scenario framing, and risk factors.",
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
        "assumption_ledger": state.get("assumption_ledger", []),
        "provenance_keys": sorted((state.get("provenance_map") or {}).keys())[:20],
        "solver_stage": state.get("solver_stage"),
        "events": workpad.get("events", []),
        "tool_results": workpad.get("tool_results", []),
        "risk_results": workpad.get("risk_results", []),
        "risk_requirements": workpad.get("risk_requirements", {}),
        "compliance_results": workpad.get("compliance_results", []),
        "compliance_requirements": workpad.get("compliance_requirements", {}),
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
        "loaded_tools": sorted(getattr(tool, "name", "") for tool in tools if getattr(tool, "name", "")),
        "results": results,
    }
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = TRACE_DIR / f"live_staged_smoke_{stamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "saved_path": str(output_path), "result_count": len(results)}, ensure_ascii=True))


if __name__ == "__main__":
    asyncio.run(main())
