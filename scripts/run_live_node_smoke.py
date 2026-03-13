from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool as lc_tool

from agent.budget import BudgetTracker
from agent.cost import CostTracker
from agent.model_config import primary_runtime_model
from agent.nodes.context import context_window
from agent.nodes.coordinator import coordinator, direct_responder, format_normalizer
from agent.nodes.reasoner import get_step_counter, make_reasoner, reset_step_counter
from agent.nodes.tool_executor import make_tool_executor, should_use_tools
from agent.nodes.verifier import verifier
from agent.state import ReplaceMessages
from agent.graph import BUILTIN_TOOLS
from mcp_servers.finance.server import black_scholes_price, mispricing_analysis, option_greeks
from mcp_servers.options_chain.server import analyze_strategy, get_expirations, get_iv_surface, get_options_chain


EXTERNAL_TOOLS = [
    lc_tool(black_scholes_price),
    lc_tool(option_greeks),
    lc_tool(mispricing_analysis),
    lc_tool(analyze_strategy),
    lc_tool(get_options_chain),
    lc_tool(get_iv_surface),
    lc_tool(get_expirations),
]
ALL_TOOLS = BUILTIN_TOOLS + EXTERNAL_TOOLS


class _LocalToolNode:
    def __init__(self, tools: list[Any]):
        self.tools_by_name = {
            getattr(tool, "name", ""): tool
            for tool in tools
            if getattr(tool, "name", "")
        }

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        tool_messages: list[ToolMessage] = []
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    tool = self.tools_by_name[call["name"]]
                    try:
                        output = tool.invoke(call.get("args", {}))
                    except Exception as exc:
                        output = f"Error: {exc}"
                    tool_messages.append(
                        ToolMessage(
                            content=str(output),
                            tool_call_id=call.get("id", ""),
                            name=call["name"],
                        )
                    )
                break
        return {"messages": tool_messages}


@dataclass
class ProbeResult:
    name: str
    ok: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


def _base_state(prompt: str, *, task_type: str = "general", selected_layers: list[str] | None = None, format_required: bool = False) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=prompt)],
        "reflection_count": 0,
        "tool_fail_count": 0,
        "last_tool_signature": "",
        "selected_layers": selected_layers or ["react_reason", "verifier_check"],
        "format_required": format_required,
        "policy_confidence": 0.0,
        "estimated_steps": 0,
        "early_exit_allowed": False,
        "architecture_trace": [],
        "checkpoint_stack": [],
        "pending_verifier_feedback": None,
        "task_type": task_type,
        "cost_tracker": CostTracker(model_name=primary_runtime_model()),
        "memory_store": None,
        "budget_tracker": BudgetTracker(),
    }


def _apply_update(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    for key, value in update.items():
        if key == "messages":
            if isinstance(value, ReplaceMessages):
                merged["messages"] = list(value)
            else:
                merged["messages"] = list(merged.get("messages", [])) + list(value)
        else:
            merged[key] = value
    return merged


def _last_ai_content(state: dict[str, Any]) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def _message_type_tail(state: dict[str, Any]) -> list[str]:
    return [type(m).__name__ for m in state["messages"][-6:]]


def probe_coordinator_cases() -> list[ProbeResult]:
    prompts = [
        (
            "coordinator_legal",
            "target company we're acquiring has compliance gaps in EU and US and the board wants stock consideration for tax reasons. What deal structure options work?",
            "legal",
        ),
        (
            "coordinator_options",
            "META implied volatility is 35%, historical volatility is 28%, and IV percentile is 75%. Recommend an options strategy with Greeks and risk management.",
            "options",
        ),
        (
            "coordinator_quantitative",
            "Calculate the financial leverage effect using ROE 3.0433% and ROA 1.579%. Return JSON only.",
            "quantitative",
        ),
    ]

    results: list[ProbeResult] = []
    for name, prompt, expected in prompts:
        reset_step_counter()
        state = _base_state(prompt)
        update = coordinator(state)
        ok = update.get("task_type") == expected and bool(update.get("selected_layers"))
        results.append(
            ProbeResult(
                name=name,
                ok=ok,
                summary=f"task_type={update.get('task_type')} layers={update.get('selected_layers')}",
                details=update,
            )
        )
    return results


def probe_direct_responder() -> ProbeResult:
    reset_step_counter()
    prompt = "Hello. Give a one-sentence reply."
    state = _base_state(prompt, task_type="general", selected_layers=["direct_answer"])
    update = direct_responder(state)
    state = _apply_update(state, update)
    answer = _last_ai_content(state)
    ok = bool(answer.strip()) and '{"name"' not in answer and "<think>" not in answer.lower()
    return ProbeResult(
        name="direct_responder",
        ok=ok,
        summary=answer[:140],
        details={
            "answer": answer,
            "step_count": get_step_counter(),
        },
    )


def probe_format_normalizer() -> ProbeResult:
    reset_step_counter()
    prompt = 'Please answer in JSON as {"answer": "..."}'
    state = _base_state(prompt, task_type="quantitative", selected_layers=["react_reason", "verifier_check", "format_normalize"], format_required=True)
    state["messages"].append(AIMessage(content="The answer is 0.9274."))
    update = format_normalizer(state)
    state = _apply_update(state, update)
    answer = _last_ai_content(state)
    looks_structured = answer.strip().startswith("{") and "answer" in answer
    return ProbeResult(
        name="format_normalizer",
        ok=looks_structured,
        summary=answer[:140],
        details={
            "answer": answer,
            "step_count": get_step_counter(),
        },
    )


async def probe_legal_cycle() -> ProbeResult:
    reset_step_counter()
    prompt = (
        "target company we're acquiring has some clean IP but also regulatory compliance gaps in EU and US. "
        "the board wants stock consideration for tax reasons but we can't risk inheriting the compliance liabilities. "
        "deal size is ~$500M. What structure options could work for both sides? Move quickly."
    )
    state = _base_state(prompt, task_type="legal")
    reasoner = make_reasoner(ALL_TOOLS)
    tool_executor = make_tool_executor(_LocalToolNode(ALL_TOOLS))

    state = _apply_update(state, reasoner(state))
    route = should_use_tools(state)
    if route == "tool_executor":
        state = _apply_update(state, await tool_executor(state))
        state = _apply_update(state, context_window(state))
    verify_update = verifier(state)
    post_verify = _apply_update(state, verify_update)

    final_ai = _last_ai_content(post_verify)
    warning_tail = post_verify["messages"][-1].content if post_verify["messages"] else ""
    contains_raw_tool_json = '{"name"' in final_ai
    ok = (
        route in {"verifier", "tool_executor"}
        and not contains_raw_tool_json
        and "internet_search" not in (final_ai + warning_tail)
        and bool(final_ai.strip() or warning_tail)
    )
    return ProbeResult(
        name="legal_reasoner_verifier_cycle",
        ok=ok,
        summary=f"route={route} final_preview={final_ai[:120] or warning_tail[:120]}",
        details={
            "route": route,
            "message_tail": _message_type_tail(post_verify),
            "last_ai": final_ai,
            "last_message": getattr(post_verify['messages'][-1], "content", "") if post_verify.get("messages") else "",
            "budget": post_verify["budget_tracker"].summary(),
            "step_count": get_step_counter(),
        },
    )


async def _tool_cycle_probe(name: str, prompt: str, task_type: str) -> ProbeResult:
    reset_step_counter()
    state = _base_state(prompt, task_type=task_type)
    reasoner = make_reasoner(ALL_TOOLS)
    tool_executor = make_tool_executor(_LocalToolNode(ALL_TOOLS))

    state = _apply_update(state, reasoner(state))
    first_route = should_use_tools(state)

    tool_used = False
    verifier_update: dict[str, Any]

    if first_route == "tool_executor":
        tool_used = True
        state = _apply_update(state, await tool_executor(state))
        state = _apply_update(state, context_window(state))
        verifier_update = verifier(state)
        state = _apply_update(state, verifier_update)
    else:
        verifier_update = verifier(state)
        state = _apply_update(state, verifier_update)

    last_ai = _last_ai_content(state)
    tracker: CostTracker = state["cost_tracker"]
    ok = (
        first_route in {"tool_executor", "verifier"}
        and ("<think>" not in last_ai.lower())
        and ('{"name"' not in last_ai)
    )
    return ProbeResult(
        name=name,
        ok=ok,
        summary=f"route={first_route} tool_calls={tracker.mcp_calls} preview={last_ai[:120] or str(state['messages'][-1].content)[:120]}",
        details={
            "route": first_route,
            "tool_calls_recorded": tracker.mcp_calls,
            "message_tail": _message_type_tail(state),
            "last_ai": last_ai,
            "last_message": getattr(state['messages'][-1], "content", "") if state.get("messages") else "",
            "budget": state["budget_tracker"].summary(),
            "step_count": get_step_counter(),
            "tool_used": tool_used,
        },
    )


async def main() -> int:
    results: list[ProbeResult] = []
    results.extend(probe_coordinator_cases())
    results.append(probe_direct_responder())
    results.append(probe_format_normalizer())
    results.append(await probe_legal_cycle())
    results.append(
        await _tool_cycle_probe(
            "quantitative_tool_cycle",
            "Respond ONLY with a calculator tool call for the expression '(3.0433 - 1.579) / 1.579'. Do not include explanation or a final answer yet.",
            "quantitative",
        )
    )
    results.append(
        await _tool_cycle_probe(
            "options_tool_cycle",
            "Use analyze_strategy to analyze a short straddle with S=300, K=300 call and put, T_days=30, r=0.05, sigma=0.35, one contract each. "
            "Then recommend whether this short-vol trade makes sense when IV is 35%, historical volatility is 28%, and IV percentile is 75%. "
            "Include Greeks, breakevens, and one concrete alternative strategy.",
            "options",
        )
    )

    output = {
        "ok": all(r.ok for r in results),
        "results": [asdict(r) for r in results],
    }
    print(json.dumps(output, indent=2))
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
