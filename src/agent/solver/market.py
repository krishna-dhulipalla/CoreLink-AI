"""
Market and cross-domain finance helpers for the solver.
"""

from __future__ import annotations

import re
from typing import Any

from agent.state import AgentState


def first_ticker_entity(entities: list[Any]) -> str | None:
    for entity in entities or []:
        token = str(entity).strip().upper()
        if re.fullmatch(r"[A-Z]{1,5}", token):
            return token
    return None


def infer_period_from_text(task_text: str) -> str:
    normalized = (task_text or "").lower()
    if any(token in normalized for token in ("1-month", "1 month", "1mo", "one month")):
        return "1mo"
    if any(token in normalized for token in ("3-month", "3 month", "3mo")):
        return "3mo"
    if any(token in normalized for token in ("6-month", "6 month", "6mo")):
        return "6mo"
    if any(token in normalized for token in ("1-year", "1 year", "12 month", "12-month", "1y")):
        return "1y"
    if any(token in normalized for token in ("performance", "trend", "history", "historical", "price action")):
        return "3mo"
    return "1mo"


def reference_price_from_tool(tool_result: dict[str, Any]) -> float | None:
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


def first_numeric(*candidates: Any) -> float | None:
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def latest_history_reference_price(tool_results: list[dict[str, Any]]) -> float | None:
    for result in reversed(tool_results):
        if not isinstance(result, dict) or result.get("errors"):
            continue
        if str(result.get("type", "")) != "get_price_history":
            continue
        facts = result.get("facts", {}) if isinstance(result.get("facts", {}), dict) else {}
        return first_numeric(facts.get("end_close"), facts.get("start_close"))
    return None


def infer_options_market_inputs(state: AgentState) -> tuple[float, float, float, int]:
    evidence_pack = state.get("evidence_pack", {}) or {}
    prompt_facts = evidence_pack.get("prompt_facts", {}) or {}
    market_snapshot = prompt_facts.get("market_snapshot", {}) if isinstance(prompt_facts.get("market_snapshot", {}), dict) else {}
    tool_results = list((state.get("workpad") or {}).get("tool_results", []))
    last_tool_result = state.get("last_tool_result") if isinstance(state.get("last_tool_result"), dict) else {}

    spot = first_numeric(
        prompt_facts.get("spot"),
        prompt_facts.get("spot_price"),
        prompt_facts.get("reference_price"),
        market_snapshot.get("spot"),
        market_snapshot.get("spot_price"),
        market_snapshot.get("reference_price"),
        market_snapshot.get("current_price"),
        reference_price_from_tool(last_tool_result),
        latest_history_reference_price(tool_results),
    )
    sigma = first_numeric(
        prompt_facts.get("implied_volatility"),
        market_snapshot.get("implied_volatility"),
    )
    r = first_numeric(
        prompt_facts.get("risk_free_rate"),
        market_snapshot.get("risk_free_rate"),
    )
    t_days_value = first_numeric(
        prompt_facts.get("days_to_expiry"),
        market_snapshot.get("days_to_expiry"),
    )
    t_days = int(t_days_value) if isinstance(t_days_value, (int, float)) else 30

    return (
        float(spot) if isinstance(spot, (int, float)) else 300.0,
        float(sigma) if isinstance(sigma, (int, float)) else 0.35,
        float(r) if isinstance(r, (int, float)) else 0.05,
        t_days,
    )


def latest_successful_tool_result(
    tool_results: list[dict[str, Any]],
    tool_names: set[str],
) -> dict[str, Any] | None:
    for result in reversed(tool_results):
        if not isinstance(result, dict):
            continue
        if str(result.get("type", "")) not in tool_names:
            continue
        if result.get("errors"):
            continue
        if isinstance(result.get("facts"), dict) and result.get("facts"):
            return result
    return None


def best_available_timestamp(state: AgentState) -> str | None:
    tool_results = list((state.get("workpad") or {}).get("tool_results", []))
    for result in reversed(tool_results):
        if not isinstance(result, dict) or result.get("errors"):
            continue
        source = result.get("source", {}) if isinstance(result.get("source", {}), dict) else {}
        facts = result.get("facts", {}) if isinstance(result.get("facts", {}), dict) else {}
        assumptions = result.get("assumptions", {}) if isinstance(result.get("assumptions", {}), dict) else {}
        for candidate in (
            source.get("timestamp"),
            source.get("as_of_date"),
            facts.get("as_of_date"),
            facts.get("window_end"),
            assumptions.get("as_of_date"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None
