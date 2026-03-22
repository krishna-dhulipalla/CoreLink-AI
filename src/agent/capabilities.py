"""Capability registry and dynamic tool resolution for the active runtime."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

from agent.legal_tools import (
    legal_playbook_retrieval,
    regulatory_execution_checklist,
    tax_structure_checklist,
    transaction_structure_checklist,
)
from agent.contracts import ACEEvent, CapabilityDescriptor, SourceBundle, TaskIntent, ToolPlan

BUILTIN_LEGAL_TOOLS = [
    legal_playbook_retrieval,
    transaction_structure_checklist,
    regulatory_execution_checklist,
    tax_structure_checklist,
]

_FAMILY_BY_TOOL: dict[str, tuple[str, int]] = {
    "calculator": ("exact_compute", 10),
    "sum_values": ("exact_compute", 20),
    "weighted_average": ("exact_compute", 20),
    "pct_change": ("exact_compute", 20),
    "cagr": ("exact_compute", 20),
    "annualize_return": ("exact_compute", 20),
    "annualize_volatility": ("exact_compute", 20),
    "bond_price_yield": ("exact_compute", 25),
    "duration_convexity": ("exact_compute", 25),
    "du_pont_analysis": ("fundamental_compute", 30),
    "valuation_multiples_compare": ("fundamental_compute", 30),
    "dcf_sensitivity_grid": ("fundamental_compute", 35),
    "cashflow_waterfall": ("fundamental_compute", 35),
    "bond_spread_duration": ("fundamental_compute", 35),
    "resolve_financial_entity": ("market_data_retrieval", 20),
    "get_price_history": ("market_data_retrieval", 15),
    "get_company_fundamentals": ("market_data_retrieval", 15),
    "get_corporate_actions": ("market_data_retrieval", 20),
    "get_returns": ("market_data_retrieval", 20),
    "get_financial_statements": ("market_data_retrieval", 25),
    "get_statement_line_items": ("market_data_retrieval", 15),
    "get_yield_curve": ("market_data_retrieval", 20),
    "analyze_strategy": ("options_strategy_analysis", 10),
    "black_scholes_price": ("options_strategy_analysis", 20),
    "option_greeks": ("options_strategy_analysis", 20),
    "mispricing_analysis": ("options_strategy_analysis", 20),
    "get_options_chain": ("options_strategy_analysis", 25),
    "get_iv_surface": ("options_strategy_analysis", 25),
    "get_expirations": ("options_strategy_analysis", 25),
    "scenario_pnl": ("options_scenario_analysis", 10),
    "calculate_portfolio_greeks": ("options_scenario_analysis", 20),
    "calculate_var": ("risk_analysis", 20),
    "run_stress_test": ("risk_analysis", 20),
    "portfolio_limit_check": ("risk_analysis", 25),
    "concentration_check": ("risk_analysis", 25),
    "factor_exposure_summary": ("risk_analysis", 25),
    "drawdown_risk_profile": ("risk_analysis", 25),
    "liquidity_stress": ("risk_analysis", 25),
    "calculate_risk_metrics": ("risk_analysis", 25),
    "fetch_reference_file": ("document_retrieval", 15),
    "list_reference_files": ("document_retrieval", 20),
    "internet_search": ("external_retrieval", 20),
    "legal_playbook_retrieval": ("legal_playbook_retrieval", 10),
    "transaction_structure_checklist": ("transaction_structure_checklist", 10),
    "regulatory_execution_checklist": ("regulatory_execution_checklist", 10),
    "tax_structure_checklist": ("tax_structure_checklist", 10),
    "execute_options_trade": ("execution_side_effect", 100),
}


def _domain_tags(tool_name: str, family: str) -> list[str]:
    if family.startswith("options_"):
        return ["finance", "options"]
    if family in {"risk_analysis"}:
        return ["finance", "risk"]
    if family in {"market_data_retrieval", "fundamental_compute"}:
        return ["finance", "market"]
    if family.startswith("legal_") or family.endswith("_checklist") or family == "legal_playbook_retrieval":
        return ["legal", "transactional"]
    if family == "document_retrieval":
        return ["documents"]
    if family == "external_retrieval":
        return ["retrieval"]
    return ["general"]


def _descriptor_for_tool(tool_obj: Any) -> CapabilityDescriptor:
    tool_name = str(getattr(tool_obj, "name", ""))
    family, priority = _FAMILY_BY_TOOL.get(tool_name, ("general", 50))
    side_effect = "blocked" if family == "execution_side_effect" else "read_only"
    return CapabilityDescriptor(
        tool_name=tool_name,
        tool_family=family,
        domain_tags=_domain_tags(tool_name, family),
        input_shape="structured_args",
        side_effect_level=side_effect,
        supports_live_data=family in {"market_data_retrieval", "external_retrieval"},
        supports_documents=family == "document_retrieval",
        supports_exact_compute=family == "exact_compute",
        priority=priority,
    )


def build_capability_registry(tools: list[Any]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for tool_obj in tools:
        descriptor = _descriptor_for_tool(tool_obj)
        registry[descriptor.tool_name] = {
            "descriptor": descriptor.model_dump(),
            "tool": tool_obj,
        }
    return registry


def _schema_bridge_tool() -> BaseTool:
    @tool
    def schema_bridge_transform(payload_json: str, key_map_json: str = "{}") -> str:
        """Transform a JSON object by remapping keys."""
        payload = json.loads(payload_json or "{}")
        key_map = json.loads(key_map_json or "{}")
        if not isinstance(payload, dict) or not isinstance(key_map, dict):
            return json.dumps({"error": "payload_json and key_map_json must decode to dict objects"}, ensure_ascii=True)
        transformed = {str(key_map.get(key, key)): value for key, value in payload.items()}
        return json.dumps(transformed, ensure_ascii=True)

    return schema_bridge_transform


def synthesize_capability(family: str) -> tuple[ACEEvent, Any | None]:
    if family in {"schema_bridge", "tabular_transform"}:
        tool_obj = _schema_bridge_tool()
        return ACEEvent(family=family, status="synthesized", reason="Generated safe schema-bridge helper.", tool_name=tool_obj.name), tool_obj
    if family in {"network_client", "file_writer", "browser_automation", "execution_side_effect"}:
        return ACEEvent(family=family, status="blocked", reason="Unsafe capability synthesis is disabled in the active engine."), None
    return ACEEvent(family=family, status="skipped", reason="No bounded synthesis path is defined for this family."), None


def _ordered_candidates(registry: dict[str, dict[str, Any]], family: str) -> list[tuple[dict[str, Any], Any]]:
    matches: list[tuple[dict[str, Any], Any]] = []
    for payload in registry.values():
        descriptor = dict(payload.get("descriptor", {}))
        if descriptor.get("tool_family") != family:
            continue
        if descriptor.get("side_effect_level") != "read_only":
            continue
        matches.append((descriptor, payload.get("tool")))
    return sorted(matches, key=lambda item: int(item[0].get("priority", 50)))


def resolve_tool_plan(
    intent: TaskIntent,
    source_bundle: SourceBundle,
    registry: dict[str, dict[str, Any]],
) -> tuple[ToolPlan, dict[str, dict[str, Any]]]:
    mutable_registry = dict(registry)
    selected: list[str] = []
    pending: list[str] = []
    blocked: list[str] = []
    notes: list[str] = []
    ace_events: list[dict[str, Any]] = []

    for family in intent.tool_families_needed:
        candidates = _ordered_candidates(mutable_registry, family)
        if not candidates:
            ace_event, synthesized = synthesize_capability(family)
            ace_events.append(ace_event.model_dump())
            if synthesized is not None:
                descriptor = _descriptor_for_tool(synthesized)
                mutable_registry[descriptor.tool_name] = {
                    "descriptor": descriptor.model_dump(),
                    "tool": synthesized,
                }
                candidates = _ordered_candidates(mutable_registry, family)
        if not candidates:
            blocked.append(family)
            notes.append(f"No safe tool binding found for {family}.")
            continue

        if intent.task_family == "legal_transactional" and family in {
            "legal_playbook_retrieval",
            "transaction_structure_checklist",
            "regulatory_execution_checklist",
            "tax_structure_checklist",
        }:
            family_tools = [descriptor["tool_name"] for descriptor, _ in candidates[:1]]
            selected.extend(family_tools)
            pending.extend(family_tools)
            continue

        descriptor, _ = candidates[0]
        tool_name = str(descriptor["tool_name"])
        selected.append(tool_name)
        if tool_name not in pending:
            pending.append(tool_name)

    if intent.task_family == "legal_transactional" and source_bundle.urls and "document_retrieval" not in blocked:
        doc_candidates = _ordered_candidates(mutable_registry, "document_retrieval")
        if doc_candidates:
            tool_name = str(doc_candidates[0][0]["tool_name"])
            if tool_name not in selected:
                selected.append(tool_name)
                pending.insert(0, tool_name)

    plan = ToolPlan(
        tool_families_needed=list(intent.tool_families_needed),
        selected_tools=list(dict.fromkeys(selected)),
        pending_tools=list(dict.fromkeys(pending)),
        blocked_families=blocked,
        ace_events=ace_events,
        notes=notes,
    )
    return plan, mutable_registry
