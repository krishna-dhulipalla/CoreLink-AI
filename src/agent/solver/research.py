"""
Equity research, event-driven, and actionable finance deterministic helpers.
"""

from __future__ import annotations

from typing import Any

from agent.solver.market import best_available_timestamp, latest_successful_tool_result
from agent.state import AgentState


def fmt_pct(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.2f}%"
    return None


def classify_research_view(
    revenue_growth: Any,
    operating_margin: Any,
    price_change: float | None,
    trailing_pe: Any,
    forward_pe: Any,
) -> tuple[str, str]:
    growth = float(revenue_growth) if isinstance(revenue_growth, (int, float)) else None
    margin = float(operating_margin) if isinstance(operating_margin, (int, float)) else None
    fpe = float(forward_pe) if isinstance(forward_pe, (int, float)) else None
    tpe = float(trailing_pe) if isinstance(trailing_pe, (int, float)) else None

    if (
        growth is not None
        and growth > 0.05
        and margin is not None
        and margin > 0.15
        and (price_change is None or price_change > -0.08)
    ):
        return (
            "constructive_but_valuation_sensitive",
            "Fundamentals support a constructive view, but valuation discipline still matters.",
        )
    if (price_change is not None and price_change < -0.10) or (fpe is not None and fpe > 32) or (tpe is not None and tpe > 35):
        return (
            "cautious_wait_for_better_entry",
            "The setup is investable only with caution because either valuation or recent price damage raises execution risk.",
        )
    return (
        "scenario_dependent_recommendation",
        "The setup is scenario-dependent and should be framed as a watchlist or conditional view rather than a hard call.",
    )


def classify_event_view(price_change: float | None, has_actions: bool) -> tuple[str, str]:
    if price_change is not None and abs(price_change) >= 0.08:
        return (
            "heightened_event_risk",
            "The catalyst already sits in a higher-volatility regime, so execution should wait for confirmation rather than chase the move.",
        )
    if has_actions:
        return (
            "event_sensitive_watch",
            "There is enough catalyst context to keep an event-sensitive watch stance, but the recommendation should remain scenario-dependent until the event confirms direction.",
        )
    return (
        "scenario_dependent_recommendation",
        "The catalyst path is not strong enough for a directional call without further confirmation.",
    )


def price_change_from_history(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    facts = result.get("facts", {}) if isinstance(result.get("facts", {}), dict) else {}
    start_close = facts.get("start_close")
    end_close = facts.get("end_close")
    if isinstance(start_close, (int, float)) and isinstance(end_close, (int, float)) and start_close:
        return (float(end_close) - float(start_close)) / float(start_close)
    return None


def deterministic_research_compute_summary(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    fundamentals = latest_successful_tool_result(tool_results, {"get_company_fundamentals"})
    history = latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    if fundamentals is None and history is None:
        return None
    parts = ["Research evidence pack is now grounded in structured finance data."]
    if fundamentals:
        facts = fundamentals.get("facts", {}).get("fundamentals", {})
        revenue_growth = fmt_pct(facts.get("revenueGrowth"))
        operating_margin = fmt_pct(facts.get("operatingMargins"))
        trailing_pe = facts.get("trailingPE")
        forward_pe = facts.get("forwardPE")
        if revenue_growth:
            parts.append(f"Revenue growth is approximately {revenue_growth}.")
        if operating_margin:
            parts.append(f"Operating margin is approximately {operating_margin}.")
        if isinstance(trailing_pe, (int, float)):
            parts.append(f"Trailing P/E is about {float(trailing_pe):.2f}.")
        if isinstance(forward_pe, (int, float)):
            parts.append(f"Forward P/E is about {float(forward_pe):.2f}.")
    change = price_change_from_history(history)
    if isinstance(change, float):
        parts.append(f"Observed price change over the retrieved window is {change * 100:.2f}%.")
    return " ".join(parts)


def deterministic_research_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    fundamentals = latest_successful_tool_result(tool_results, {"get_company_fundamentals"})
    history = latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    if fundamentals is None and history is None:
        return None
    facts = fundamentals.get("facts", {}).get("fundamentals", {}) if fundamentals else {}
    timestamp = best_available_timestamp(state)
    revenue_growth = fmt_pct(facts.get("revenueGrowth"))
    operating_margin = fmt_pct(facts.get("operatingMargins"))
    roe = fmt_pct(facts.get("returnOnEquity"))
    trailing_pe = facts.get("trailingPE")
    forward_pe = facts.get("forwardPE")
    change = price_change_from_history(history)
    recommendation_class, recommendation_line = classify_research_view(
        facts.get("revenueGrowth"),
        facts.get("operatingMargins"),
        change,
        trailing_pe,
        forward_pe,
    )
    action_view = {
        "constructive_but_valuation_sensitive": "Action view: constructive on disciplined entries or pullbacks, not on blind momentum.",
        "cautious_wait_for_better_entry": "Action view: keep the name on a watchlist and wait for either valuation relief or cleaner operating evidence.",
    }.get(
        recommendation_class,
        "Action view: keep the recommendation conditional until evidence tightens on both operating trend and valuation support.",
    )
    thesis_points: list[str] = []
    if revenue_growth:
        thesis_points.append(f"Top-line growth is still supportive at {revenue_growth}.")
    if operating_margin:
        thesis_points.append(f"Margin profile remains informative at {operating_margin}, which is a key test of operating quality.")
    if isinstance(change, float):
        thesis_points.append(
            "Recent price action improves entry quality."
            if change < 0
            else "Recent price strength means entry discipline matters because sentiment may already reflect part of the good news."
        )
    if isinstance(trailing_pe, (int, float)) or isinstance(forward_pe, (int, float)):
        if isinstance(trailing_pe, (int, float)) and isinstance(forward_pe, (int, float)):
            thesis_points.append("Valuation should be judged through both trailing and forward multiples, not through growth alone.")
        else:
            thesis_points.append("Valuation context is available, but it still needs peer framing before a high-conviction call.")
    if not thesis_points:
        thesis_points.append(
            "The name should be judged through the interaction of operating quality, valuation, and near-term market follow-through rather than through a single metric."
        )

    catalysts: list[str] = ["- Watch the next reporting cycle for confirmation on revenue durability, margin trend, and guidance quality."]
    if isinstance(forward_pe, (int, float)) and isinstance(trailing_pe, (int, float)):
        if float(forward_pe) > float(trailing_pe):
            catalysts.append("- Forward multiple expectations are still demanding, so watch for execution that actually earns that valuation.")
        else:
            catalysts.append("- Forward multiple sits below trailing, so watch whether earnings delivery can justify a better entry or a rerating.")
    else:
        catalysts.append("- Add peer or DCF work before treating the current valuation framing as complete.")

    change_view: list[str] = [
        "- Upgrade the view only if the next update preserves growth quality while keeping valuation or price damage from worsening.",
        "- Cut the view quickly if growth slows, margin quality slips, or management guidance weakens materially.",
    ]
    if isinstance(change, float) and change <= -0.1:
        change_view.insert(0, "- A stabilizing post-drawdown price base would improve entry quality more than a reflexive bounce.")

    lines = ["**Recommendation**", recommendation_line, f"- {action_view}", "", "**Thesis**"]
    lines.extend([f"- {point}" for point in thesis_points])
    lines.extend(["", "**Evidence**"])
    if revenue_growth:
        lines.append(f"- Revenue growth: {revenue_growth}.")
    if operating_margin:
        lines.append(f"- Operating margin: {operating_margin}.")
    if roe:
        lines.append(f"- Return on equity: {roe}.")
    if isinstance(change, float):
        lines.append(f"- Retrieved price performance over the evidence window: {change * 100:.2f}%.")
    if timestamp:
        lines.append(f"- Source timestamp: {timestamp}.")
    lines.extend(["", "**Valuation**"])
    if isinstance(trailing_pe, (int, float)) or isinstance(forward_pe, (int, float)):
        lines.append(
            f"- Multiples framing: trailing P/E {float(trailing_pe):.2f} and forward P/E {float(forward_pe):.2f}."
            if isinstance(trailing_pe, (int, float)) and isinstance(forward_pe, (int, float))
            else f"- Multiples framing: trailing P/E {float(trailing_pe):.2f}."
            if isinstance(trailing_pe, (int, float))
            else f"- Multiples framing: forward P/E {float(forward_pe):.2f}."
        )
    else:
        lines.append("- Valuation framing is limited on the current evidence and should be expanded with peer or DCF work.")
    lines.extend(["", "**Catalysts and Watchpoints**"])
    lines.extend(catalysts)
    lines.extend(["", "**What Would Change The View**"])
    lines.extend(change_view)
    lines.extend(
        [
            "",
            "**Risks**",
            "- Thesis risk increases if growth decelerates faster than margins can absorb.",
            "- Market multiple compression can dominate even if the fundamental trend remains stable.",
            "",
            "**Recommendation Class**",
            f"- {recommendation_class}.",
        ]
    )
    return "\n".join(lines)


def deterministic_event_compute_summary(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    actions = latest_successful_tool_result(tool_results, {"get_corporate_actions"})
    history = latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    if actions is None and history is None:
        return None
    parts = ["Event-driven finance evidence is now grounded in structured catalyst and market context."]
    if actions:
        facts = actions.get("facts", {})
        dividend_count = len(facts.get("recent_dividends", []) or [])
        split_count = len(facts.get("recent_splits", []) or [])
        if dividend_count or split_count:
            parts.append(
                f"Retrieved catalyst context including {dividend_count} dividend records and {split_count} split records."
            )
    change = price_change_from_history(history)
    if isinstance(change, float):
        parts.append(f"Observed market move over the evidence window is {change * 100:.2f}%.")
    return " ".join(parts)


def deterministic_event_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    actions = latest_successful_tool_result(tool_results, {"get_corporate_actions"})
    history = latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    if actions is None and history is None:
        return None
    change = price_change_from_history(history)
    timestamp = best_available_timestamp(state)
    action_facts = actions.get("facts", {}) if isinstance(actions, dict) else {}
    dividend_count = len(action_facts.get("recent_dividends", []) or [])
    split_count = len(action_facts.get("recent_splits", []) or [])
    recommendation_class, action_view = classify_event_view(change, bool(dividend_count or split_count))
    execution_discipline = [
        "- Keep sizing smaller than a normal thesis trade until the catalyst confirms direction and post-event liquidity is visible.",
        "- Do not treat pre-event price drift as confirmation; wait for the catalyst and the first orderly post-event reaction.",
    ]
    if isinstance(change, float) and abs(change) >= 0.08:
        execution_discipline.append(
            "- Because the name is already moving materially, avoid chasing the gap; require confirmation that volatility and spreads are normalizing."
        )

    lines = ["**Recommendation**", action_view, "", "**Catalyst**", "The setup is event-driven and should be evaluated through explicit catalyst scenarios rather than a static valuation-only lens.", "", "**Market Context**"]
    if isinstance(change, float):
        lines.append(f"- Retrieved price move over the evidence window: {change * 100:.2f}%.")
    if timestamp:
        lines.append(f"- Source timestamp: {timestamp}.")
    if dividend_count or split_count:
        lines.append(f"- Corporate-action context retrieved: {dividend_count} dividend records and {split_count} split records.")
    lines.extend(
        [
            "",
            "**Scenarios**",
            "- Base case: the event lands near consensus and price reaction stays contained.",
            "- Upside case: the catalyst improves guidance or sentiment and supports upside follow-through.",
            "- Downside case: the event disappoints and reprices the name sharply lower.",
            "- Stress case: the catalyst coincides with a broader market or volatility shock.",
            "",
            "**Execution Discipline**",
        ]
    )
    lines.extend(execution_discipline)
    lines.extend(
        [
            "",
            "**What Would Change The View**",
            "- Upgrade the stance only if the event confirms direction and the post-event price action remains orderly.",
            "- Downgrade immediately if the catalyst triggers a gap move against the thesis or a broad volatility shock.",
            "",
            "**Risk**",
            "- Event timing, guidance uncertainty, and gap risk should be treated as the main risk drivers.",
            "",
            "**Recommendation Class**",
            f"- {recommendation_class}.",
        ]
    )
    return "\n".join(lines)


def deterministic_actionable_finance_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    history = latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    fundamentals = latest_successful_tool_result(tool_results, {"get_company_fundamentals"})
    if history is None and fundamentals is None:
        return None
    timestamp = best_available_timestamp(state)
    policy_context = (state.get("evidence_pack", {}) or {}).get("policy_context", {}) or {}
    price_change = price_change_from_history(history)
    lines = ["**Recommendation**", "Recommendation is scenario-dependent on the current retrieved evidence and should not be treated as a blind high-conviction action.", "", "**Evidence**"]
    if timestamp:
        lines.append(f"- Source timestamp: {timestamp}.")
    if isinstance(price_change, float):
        lines.append(f"- Observed price move over the retrieved window: {price_change * 100:.2f}%.")
    if fundamentals:
        facts = fundamentals.get("facts", {}).get("fundamentals", {})
        margin = fmt_pct(facts.get("operatingMargins"))
        growth = fmt_pct(facts.get("revenueGrowth"))
        if margin:
            lines.append(f"- Operating margin: {margin}.")
        if growth:
            lines.append(f"- Revenue growth: {growth}.")
    lines.extend(["", "**Risk**", "- Action remains sensitive to fresh evidence, market regime, and headline risk.", "", "**Disclosures**", "- Recommendation class: scenario_dependent_recommendation."])
    if policy_context.get("requires_recommendation_class"):
        lines.append("- Recommendation class is explicit because this is an actionable finance path.")
    return "\n".join(lines)
