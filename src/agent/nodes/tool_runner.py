"""
Tool Runner Node
================
Executes exactly one tool call and normalizes the result into a ToolResult
contract before returning control to the solver.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agent.contracts import ToolResult
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import (
    _merge_unique_assumptions,
    allowed_tools_for_template,
    derive_assumption_ledger_entries,
    merge_tool_result_into_evidence,
)
from agent.state import AgentState
from agent.tracer import get_tracer
from agent.tools.normalization import normalize_tool_output

logger = logging.getLogger(__name__)

_FINANCE_ARG_ALIASES: dict[str, dict[str, list[str]]] = {
    "resolve_financial_entity": {
        "identifier": ["entity", "ticker", "symbol", "security"],
        "as_of_date": ["as_of", "date"],
    },
    "get_price_history": {
        "ticker": ["entity", "symbol", "security", "underlying"],
        "period": ["time_frame", "window", "lookback"],
        "as_of_date": ["as_of", "date", "end_date"],
    },
    "get_company_fundamentals": {
        "ticker": ["entity", "symbol", "security"],
        "as_of_date": ["as_of", "date"],
    },
    "get_corporate_actions": {
        "ticker": ["entity", "symbol", "security"],
        "as_of_date": ["as_of", "date"],
    },
    "get_returns": {
        "ticker": ["entity", "symbol", "security", "underlying"],
        "period": ["time_frame", "window", "lookback"],
        "as_of_date": ["as_of", "date", "end_date"],
    },
    "get_financial_statements": {
        "ticker": ["entity", "symbol", "security"],
        "statement_type": ["statement", "statement_kind", "financial_statement"],
        "frequency": ["cadence"],
        "limit": ["period_limit", "num_periods"],
        "as_of_date": ["as_of", "date", "end_date"],
    },
    "get_statement_line_items": {
        "ticker": ["entity", "symbol", "security"],
        "line_items": ["metrics", "fields", "line_item_names"],
        "statement_type": ["statement", "statement_kind", "financial_statement"],
        "frequency": ["cadence"],
        "limit": ["period_limit", "num_periods"],
        "as_of_date": ["as_of", "date", "end_date"],
    },
    "get_yield_curve": {
        "as_of_date": ["as_of", "date", "end_date"],
    },
    "sum_values": {
        "values": ["numbers", "series", "items"],
    },
    "weighted_average": {
        "values": ["numbers", "series", "items"],
        "weights": ["weightings", "allocation_weights"],
    },
    "pct_change": {
        "old_value": ["start", "start_value", "start_price", "initial_value", "base"],
        "new_value": ["end", "end_value", "end_price", "final_value", "current"],
    },
    "annualize_return": {
        "period_return_decimal": ["return_decimal", "raw_return", "period_return", "return_value"],
        "days_held": ["holding_period_days", "days", "holding_days"],
    },
    "annualize_volatility": {
        "period_volatility_decimal": ["volatility", "vol", "sigma"],
        "periods_per_year": ["frequency", "freq", "observations_per_year"],
    },
    "bond_price_yield": {
        "face_value": ["par_value", "principal"],
        "coupon_rate_decimal": ["coupon_rate", "coupon"],
        "periods_to_maturity": ["maturity_periods", "n_periods"],
        "yield_to_maturity_decimal": ["ytm", "yield_rate"],
    },
    "duration_convexity": {
        "face_value": ["par_value", "principal"],
        "coupon_rate_decimal": ["coupon_rate", "coupon"],
        "periods_to_maturity": ["maturity_periods", "n_periods"],
        "yield_to_maturity_decimal": ["ytm", "yield_rate"],
    },
    "scenario_pnl": {
        "net_premium": ["premium", "credit", "debit"],
        "total_delta": ["delta"],
        "total_gamma": ["gamma"],
        "total_theta_per_day": ["theta", "theta_per_day"],
        "total_vega_per_vol_point": ["vega", "vega_per_vol_point"],
        "reference_price": ["spot", "spot_price", "underlying_price", "S"],
    },
    "calculate_var": {
        "portfolio_value": ["notional", "notional_value", "exposure"],
        "daily_vol": ["daily_volatility", "volatility", "vol", "sigma"],
        "holding_period_days": ["days", "horizon_days"],
    },
    "run_stress_test": {
        "portfolio_value": ["notional", "notional_value", "exposure"],
        "portfolio_delta": ["delta"],
        "portfolio_vega": ["vega"],
        "portfolio_theta": ["theta", "theta_per_day"],
    },
}

_STATEMENT_TYPE_ALIASES = {
    "income_statement": "income",
    "income": "income",
    "pnl": "income",
    "profit_and_loss": "income",
    "balance_sheet": "balance_sheet",
    "balance": "balance_sheet",
    "cashflow": "cash_flow",
    "cash_flow": "cash_flow",
    "cash_flow_statement": "cash_flow",
}

_FREQUENCY_ALIASES = {
    "annual": "annual",
    "yearly": "annual",
    "quarterly": "quarterly",
    "quarter": "quarterly",
    "q": "quarterly",
}


def _tool_signature(state: AgentState) -> str:
    pending = state.get("pending_tool_call") or {}
    return f"{pending.get('name', '')}:{json.dumps(pending.get('arguments', {}), sort_keys=True)}"


def _tool_registry(tool_node: ToolNode) -> dict[str, Any]:
    registry = getattr(tool_node, "tools_by_name", None)
    if isinstance(registry, dict):
        return registry
    tools = getattr(tool_node, "tools", []) or []
    return {getattr(tool, "name", ""): tool for tool in tools if getattr(tool, "name", "")}


def _extract_tool_call_id(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[-1].get("id", "unknown")
    return "unknown"


def _next_solver_stage_after_tool(state: AgentState) -> str | None:
    if str(state.get("solver_stage", "")).upper() != "REVISE":
        return None
    review_feedback = state.get("review_feedback") or {}
    risk_feedback = state.get("risk_feedback") or {}
    repair_target = str(risk_feedback.get("repair_target", review_feedback.get("repair_target", ""))).lower()
    if repair_target == "compute":
        return "COMPUTE"
    if repair_target == "gather":
        return "GATHER"
    return None


def _rewrite_messages_with_normalized_tool_args(
    messages: list[Any],
    *,
    tool_name: str,
    tool_args: dict[str, Any],
) -> list[Any]:
    updated_messages = list(messages or [])
    for index in range(len(updated_messages) - 1, -1, -1):
        message = updated_messages[index]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        rewritten = []
        changed = False
        for tool_call in message.tool_calls:
            if str(tool_call.get("name", "")).strip() == tool_name:
                rewritten.append(
                    {
                        **tool_call,
                        "args": dict(tool_args),
                    }
                )
                changed = True
            else:
                rewritten.append(tool_call)
        if changed:
            updated_messages[index] = AIMessage(
                content=message.content,
                additional_kwargs=message.additional_kwargs,
                response_metadata=message.response_metadata,
                tool_calls=rewritten,
                id=message.id,
            )
            break
    return updated_messages


def _normalize_period_alias(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower()
    mapping = {
        "1m": "1mo",
        "1mo": "1mo",
        "3m": "3mo",
        "3mo": "3mo",
        "6m": "6mo",
        "6mo": "6mo",
        "1y": "1y",
        "12m": "1y",
        "2y": "2y",
        "5y": "5y",
        "10y": "10y",
        "ytd": "ytd",
        "max": "max",
    }
    return mapping.get(normalized, value)


def _normalize_finance_tool_args(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool_args, dict):
        return {}

    normalized = dict(tool_args)
    alias_map = _FINANCE_ARG_ALIASES.get(tool_name, {})
    for canonical, aliases in alias_map.items():
        if canonical in normalized:
            continue
        for alias in aliases:
            if alias in normalized:
                normalized[canonical] = normalized[alias]
                break

    if tool_name == "scenario_pnl" and isinstance(normalized.get("strategy_facts"), dict):
        strategy_facts = normalized["strategy_facts"]
        for canonical in (
            "net_premium",
            "total_delta",
            "total_gamma",
            "total_theta_per_day",
            "total_vega_per_vol_point",
            "reference_price",
        ):
            if canonical in normalized:
                continue
            candidate_keys = [canonical] + list(_FINANCE_ARG_ALIASES.get("scenario_pnl", {}).get(canonical, []))
            for candidate in candidate_keys:
                if candidate in strategy_facts:
                    normalized[canonical] = strategy_facts[candidate]
                    break
        if "reference_price" not in normalized:
            for alias in ("spot", "spot_price", "S"):
                if alias in strategy_facts:
                    normalized["reference_price"] = strategy_facts[alias]
                    break

    if tool_name in {"get_price_history", "get_returns"} and "period" in normalized:
        normalized["period"] = _normalize_period_alias(normalized["period"])

    if tool_name in {"get_financial_statements", "get_statement_line_items"}:
        statement_type = normalized.get("statement_type")
        if isinstance(statement_type, str):
            normalized["statement_type"] = _STATEMENT_TYPE_ALIASES.get(
                statement_type.strip().lower(),
                statement_type,
            )
        frequency = normalized.get("frequency")
        if isinstance(frequency, str):
            normalized["frequency"] = _FREQUENCY_ALIASES.get(
                frequency.strip().lower(),
                frequency,
            )
        if tool_name == "get_statement_line_items":
            line_items = normalized.get("line_items")
            if isinstance(line_items, str):
                normalized["line_items"] = [line_items]

    return normalized


def _reference_price_from_tool_result(tool_result: dict[str, Any]) -> float | None:
    assumptions = tool_result.get("assumptions", {}) if isinstance(tool_result, dict) else {}
    if isinstance(assumptions, dict):
        for key in ("reference_price", "spot", "spot_price", "S"):
            value = assumptions.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        legs = assumptions.get("legs")
        if isinstance(legs, list):
            for leg in legs:
                if isinstance(leg, dict) and isinstance(leg.get("S"), (int, float)):
                    return float(leg["S"])
    return None


def _enrich_finance_tool_args_from_state(
    tool_name: str,
    tool_args: dict[str, Any],
    state: AgentState,
) -> dict[str, Any]:
    normalized = dict(tool_args)
    last_tool_result = state.get("last_tool_result") or {}

    if tool_name == "scenario_pnl" and str(last_tool_result.get("type", "")) == "analyze_strategy":
        facts = last_tool_result.get("facts", {}) if isinstance(last_tool_result, dict) else {}
        fallback_fields = {
            "net_premium": "net_premium",
            "total_delta": "total_delta",
            "total_gamma": "total_gamma",
            "total_theta_per_day": "total_theta_per_day",
            "total_vega_per_vol_point": "total_vega_per_vol_point",
        }
        for canonical, fact_key in fallback_fields.items():
            if canonical not in normalized and isinstance(facts.get(fact_key), (int, float)):
                normalized[canonical] = float(facts[fact_key])
        if "reference_price" not in normalized:
            reference_price = _reference_price_from_tool_result(last_tool_result)
            if isinstance(reference_price, (int, float)):
                normalized["reference_price"] = float(reference_price)

    return normalized



def make_tool_runner(tool_node: ToolNode):
    async def tool_runner(state: AgentState) -> dict:
        step = increment_runtime_step()
        profile = state.get("task_profile", "general")
        budget = state.get("budget_tracker")
        pending = state.get("pending_tool_call") or {}
        tool_name = str(pending.get("name", "")).strip()
        tool_args = _normalize_finance_tool_args(
            tool_name,
            pending.get("arguments", {}) if isinstance(pending.get("arguments", {}), dict) else {},
        )
        tool_args = _enrich_finance_tool_args_from_state(tool_name, tool_args, state)
        allowed = allowed_tools_for_template(state.get("execution_template"), profile)
        registry = _tool_registry(tool_node)
        workpad = dict(state.get("workpad", {}))
        normalized_pending = {"name": tool_name, "arguments": tool_args}

        if not tool_name:
            logger.warning("[Step %s] tool_runner -> missing pending tool call", step)
            tool_result = ToolResult(
                type="tool_runner",
                facts={},
                assumptions={},
                source={"tool": "tool_runner"},
                errors=["Missing pending tool call."],
            )
            return {
                "messages": [ToolMessage(content=json.dumps(tool_result.model_dump(), ensure_ascii=True), name="unknown", tool_call_id=_extract_tool_call_id(state))],
                "last_tool_result": tool_result.model_dump(),
                "pending_tool_call": None,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
            }

        if tool_name not in allowed:
            logger.warning("[Step %s] tool_runner -> blocked disallowed tool %s for profile=%s", step, tool_name, profile)
            tool_result = ToolResult(
                type=tool_name,
                facts={},
                assumptions=tool_args if isinstance(tool_args, dict) else {},
                source={"tool": tool_name},
                errors=[f"Tool '{tool_name}' is not allowed for task_profile '{profile}'."],
            )
            workpad.setdefault("tool_results", []).append(tool_result.model_dump())
            workpad.setdefault("events", []).append({"node": "tool_runner", "action": f"blocked {tool_name}"})
            return {
                "messages": [ToolMessage(content=json.dumps(tool_result.model_dump(), ensure_ascii=True), name=tool_name, tool_call_id=_extract_tool_call_id(state))],
                "last_tool_result": tool_result.model_dump(),
                "pending_tool_call": None,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
                "last_tool_signature": _tool_signature(state),
                "workpad": workpad,
            }

        if tool_name not in registry:
            tool_result = ToolResult(
                type=tool_name,
                facts={},
                assumptions=tool_args if isinstance(tool_args, dict) else {},
                source={"tool": tool_name},
                errors=[f"Tool '{tool_name}' is not registered in the current runtime."],
            )
            workpad.setdefault("tool_results", []).append(tool_result.model_dump())
            workpad.setdefault("events", []).append({"node": "tool_runner", "action": f"missing tool {tool_name}"})
            return {
                "messages": [ToolMessage(content=json.dumps(tool_result.model_dump(), ensure_ascii=True), name=tool_name, tool_call_id=_extract_tool_call_id(state))],
                "last_tool_result": tool_result.model_dump(),
                "pending_tool_call": None,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
                "last_tool_signature": _tool_signature(state),
                "workpad": workpad,
            }

        if budget and budget.tool_calls_exhausted():
            budget.log_budget_exit("tool_budget_exhausted", f"Blocked tool '{tool_name}' after reaching tool-call cap.")
            tool_result = ToolResult(
                type=tool_name,
                facts={},
                assumptions=tool_args if isinstance(tool_args, dict) else {},
                source={"tool": tool_name},
                errors=[f"Tool-call budget exhausted before running '{tool_name}'."],
            )
            workpad.setdefault("tool_results", []).append(tool_result.model_dump())
            workpad.setdefault("events", []).append({"node": "tool_runner", "action": f"budget blocked {tool_name}"})
            return {
                "messages": [ToolMessage(content=json.dumps(tool_result.model_dump(), ensure_ascii=True), name=tool_name, tool_call_id=_extract_tool_call_id(state))],
                "last_tool_result": tool_result.model_dump(),
                "pending_tool_call": None,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
                "last_tool_signature": _tool_signature(state),
                "workpad": workpad,
            }

        invoke_state = {
            **state,
            "pending_tool_call": normalized_pending,
            "messages": _rewrite_messages_with_normalized_tool_args(
                list(state.get("messages", [])),
                tool_name=tool_name,
                tool_args=tool_args,
            ),
        }
        result = await tool_node.ainvoke(invoke_state)
        messages = result.get("messages", [])
        tool_message = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None)
        tool_result = normalize_tool_output(
            tool_name,
            getattr(tool_message, "content", ""),
            tool_args if isinstance(tool_args, dict) else {},
        )
        tool_result.source.setdefault("solver_stage", state.get("solver_stage", "COMPUTE"))
        updated_evidence_pack, updated_provenance_map = merge_tool_result_into_evidence(
            state.get("evidence_pack", {}),
            tool_result,
            tool_args if isinstance(tool_args, dict) else {},
            state.get("provenance_map", {}),
        )
        added_assumptions = derive_assumption_ledger_entries(
            tool_name,
            tool_args if isinstance(tool_args, dict) else {},
            state.get("evidence_pack", {}),
        )
        updated_assumption_ledger = _merge_unique_assumptions(
            state.get("assumption_ledger", []),
            added_assumptions,
        )
        workpad.setdefault("tool_results", []).append(tool_result.model_dump())
        workpad.setdefault("events", []).append({"node": "tool_runner", "action": f"ran {tool_name}"})

        tracker = state.get("cost_tracker")
        if tracker:
            tracker.record_mcp_call()
        if budget:
            budget.record_tool_call()

        if tool_message is not None:
            normalized_message = ToolMessage(
                content=json.dumps(tool_result.model_dump(), ensure_ascii=True),
                tool_call_id=tool_message.tool_call_id,
                name=tool_name,
            )
            messages = [normalized_message]

        logger.info("[Step %s] tool_runner -> %s errors=%s", step, tool_name, bool(tool_result.errors))
        tracer = get_tracer()
        if tracer:
            tracer.record("tool_runner", {
                "tool_name": tool_name,
                "tool_args": tool_args if isinstance(tool_args, dict) else {},
                "result_type": tool_result.type,
                "errors": tool_result.errors,
                "fact_keys": sorted(tool_result.facts.keys()) if isinstance(tool_result.facts, dict) else [],
            })
        result_state = {
            "messages": messages,
            "last_tool_result": tool_result.model_dump(),
            "evidence_pack": updated_evidence_pack,
            "assumption_ledger": updated_assumption_ledger,
            "provenance_map": updated_provenance_map,
            "pending_tool_call": None,
            "risk_feedback": None,
            "tool_fail_count": state.get("tool_fail_count", 0) + (1 if tool_result.errors else 0),
            "last_tool_signature": _tool_signature(state),
            "workpad": workpad,
        }
        next_stage = _next_solver_stage_after_tool(state)
        if next_stage and not tool_result.errors:
            result_state["solver_stage"] = next_stage
        return result_state

    return tool_runner
