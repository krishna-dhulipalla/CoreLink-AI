"""Capability registry and dynamic tool resolution for the active runtime."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

from engine.agent.benchmarks import (
    benchmark_descriptor_allowed,
    benchmark_registry_policy,
    benchmark_runtime_policy,
    benchmark_tool_selection_active,
)
from engine.agent.legal_tools import (
    legal_playbook_retrieval,
    regulatory_execution_checklist,
    tax_structure_checklist,
    transaction_structure_checklist,
)
from engine.agent.retrieval_tools import (
    fetch_corpus_document,
    fetch_officeqa_pages,
    fetch_officeqa_table,
    lookup_officeqa_cells,
    lookup_officeqa_rows,
    search_officeqa_documents,
    search_reference_corpus,
)
from engine.agent.contracts import ACEEvent, CapabilityDescriptor, SourceBundle, TaskIntent, ToolPlan

BUILTIN_LEGAL_TOOLS = [
    legal_playbook_retrieval,
    transaction_structure_checklist,
    regulatory_execution_checklist,
    tax_structure_checklist,
]
BUILTIN_RETRIEVAL_TOOLS = [
    search_officeqa_documents,
    fetch_officeqa_pages,
    fetch_officeqa_table,
    lookup_officeqa_rows,
    lookup_officeqa_cells,
    search_reference_corpus,
    fetch_corpus_document,
]

_CANONICAL_FAMILIES = {
    "exact_compute",
    "market_data_retrieval",
    "document_retrieval",
    "external_retrieval",
    "options_strategy_analysis",
    "options_scenario_analysis",
    "legal_playbook_retrieval",
    "transaction_structure_analysis",
    "regulatory_execution_analysis",
    "tax_structure_analysis",
    "analytical_reasoning",
    "market_scenario_analysis",
}

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
    "du_pont_analysis": ("analytical_reasoning", 30),
    "valuation_multiples_compare": ("analytical_reasoning", 30),
    "dcf_sensitivity_grid": ("analytical_reasoning", 35),
    "cashflow_waterfall": ("analytical_reasoning", 35),
    "bond_spread_duration": ("analytical_reasoning", 35),
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
    "calculate_var": ("market_scenario_analysis", 20),
    "run_stress_test": ("market_scenario_analysis", 20),
    "portfolio_limit_check": ("market_scenario_analysis", 25),
    "concentration_check": ("market_scenario_analysis", 25),
    "factor_exposure_summary": ("market_scenario_analysis", 25),
    "drawdown_risk_profile": ("market_scenario_analysis", 25),
    "liquidity_stress": ("market_scenario_analysis", 25),
    "calculate_risk_metrics": ("market_scenario_analysis", 25),
    "fetch_reference_file": ("document_retrieval", 15),
    "list_reference_files": ("document_retrieval", 20),
    "search_officeqa_documents": ("document_retrieval", 5),
    "fetch_officeqa_table": ("document_retrieval", 6),
    "lookup_officeqa_rows": ("document_retrieval", 7),
    "lookup_officeqa_cells": ("document_retrieval", 8),
    "fetch_officeqa_pages": ("document_retrieval", 10),
    "search_reference_corpus": ("document_retrieval", 8),
    "fetch_corpus_document": ("document_retrieval", 9),
    "internet_search": ("external_retrieval", 20),
    "legal_playbook_retrieval": ("legal_playbook_retrieval", 10),
    "transaction_structure_checklist": ("transaction_structure_analysis", 10),
    "regulatory_execution_checklist": ("regulatory_execution_analysis", 10),
    "tax_structure_checklist": ("tax_structure_analysis", 10),
    "execute_options_trade": ("execution_side_effect", 100),
}
_ROLE_BY_TOOL: dict[str, str] = {
    "internet_search": "search",
    "search_officeqa_documents": "search",
    "fetch_officeqa_table": "fetch",
    "lookup_officeqa_rows": "fetch",
    "lookup_officeqa_cells": "fetch",
    "fetch_officeqa_pages": "fetch",
    "search_reference_corpus": "search",
    "list_reference_files": "discover",
    "fetch_reference_file": "fetch",
    "fetch_corpus_document": "fetch",
}


def _infer_external_family_and_role(tool_obj: Any) -> tuple[str, str, int]:
    tool_name = str(getattr(tool_obj, "name", "") or "")
    description = str(getattr(tool_obj, "description", "") or "")
    normalized = f"{tool_name} {description}".lower()

    explicit_doc_terms = (
        "document",
        "pdf",
        "bulletin",
        "filing",
        "corpus",
        "treasury",
        "reference file",
        "reference document",
        "source file",
        "table",
        "page",
        "10-k",
        "10q",
    )
    contextual_doc_terms = (
        "report",
        "annual report",
        "statement",
        "archive",
    )
    search_terms = ("search", "find", "lookup", "query", "retrieve")
    fetch_terms = ("fetch", "read", "open", "load", "download", "extract")
    discover_terms = ("list", "discover", "enumerate", "available files", "available documents")
    document_actions = (*search_terms, *fetch_terms, *discover_terms)

    if any(term in normalized for term in explicit_doc_terms) or (
        any(term in normalized for term in contextual_doc_terms)
        and any(term in normalized for term in document_actions)
    ):
        if any(term in normalized for term in discover_terms):
            return "document_retrieval", "discover", 8
        if any(term in normalized for term in search_terms):
            return "document_retrieval", "search", 8
        if any(term in normalized for term in fetch_terms):
            return "document_retrieval", "fetch", 9
        return "document_retrieval", "", 12

    if any(term in normalized for term in ("web search", "internet", "google", "serp", "browse", "browser search")):
        return "external_retrieval", "search", 20

    if any(term in normalized for term in ("black-scholes", "black scholes", "option greeks", "iv surface", "options chain")):
        return "options_strategy_analysis", "", 20

    if any(term in normalized for term in ("scenario pnl", "stress", "scenario analysis", "drawdown", "liquidity shock")):
        return "market_scenario_analysis", "", 25

    if any(term in normalized for term in ("amortization", "loan schedule", "npv", "irr", "present value", "discount rate", "pricing")):
        return "exact_compute", "", 25

    return "general", "", 50


def _domain_tags(tool_name: str, family: str) -> list[str]:
    if family.startswith("options_"):
        return ["finance", "options"]
    if family in {"market_scenario_analysis"}:
        return ["finance", "risk"]
    if family in {"market_data_retrieval", "analytical_reasoning"}:
        return ["finance", "market"]
    if family.startswith("legal_") or family.endswith("_analysis") or family == "legal_playbook_retrieval":
        return ["legal", "transactional"]
    if family == "document_retrieval":
        return ["documents"]
    if family == "external_retrieval":
        return ["retrieval"]
    return ["general"]


def _descriptor_for_tool(tool_obj: Any) -> CapabilityDescriptor:
    tool_name = str(getattr(tool_obj, "name", ""))
    family, priority = _FAMILY_BY_TOOL.get(tool_name, ("", 0))
    role = _ROLE_BY_TOOL.get(tool_name, "")
    if not family:
        family, role, priority = _infer_external_family_and_role(tool_obj)
    side_effect = "blocked" if family == "execution_side_effect" else "read_only"
    return CapabilityDescriptor(
        tool_name=tool_name,
        tool_family=family,
        tool_role=role,
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


def filter_registry_for_benchmark(
    registry: dict[str, dict[str, Any]],
    benchmark_name: str = "",
) -> dict[str, dict[str, Any]]:
    normalized = str(benchmark_name or "").strip().lower()
    if not normalized:
        return registry
    policy = benchmark_registry_policy(normalized)
    allowed_names = set(policy.get("allowed_tool_names", []))
    allowed_families = set(policy.get("allowed_families", []))
    if not allowed_names and not allowed_families:
        return registry

    filtered: dict[str, dict[str, Any]] = {}
    for tool_name, payload in registry.items():
        descriptor = dict(payload.get("descriptor", {}) or {})
        family = str(descriptor.get("tool_family", "") or "")
        if allowed_names and tool_name in allowed_names:
            filtered[tool_name] = payload
            continue
        if allowed_families and family in allowed_families:
            filtered[tool_name] = payload
    return filtered


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


def _source_requests_live_data(source_bundle: SourceBundle) -> bool:
    lowered = (source_bundle.task_text or "").lower()
    return any(
        token in lowered
        for token in (
            "latest",
            "today",
            "recent",
            "look up",
            "search",
            "source-backed",
            "current ",
            "as of ",
            "price of",
            "what was",
            "what is",
        )
    )


def _looks_like_document_corpus_query(source_bundle: SourceBundle) -> bool:
    lowered = (source_bundle.task_text or "").lower()
    return any(
        token in lowered
        for token in (
            "according to",
            "report",
            "bulletin",
            "filing",
            "document",
            "treasury",
            "annual report",
            "10-k",
            "10q",
            "table",
            "page",
            "pdf",
        )
    )


def _normalize_family(raw_family: str, intent: TaskIntent, source_bundle: SourceBundle) -> str:
    family = str(raw_family or "").strip()
    if family in _CANONICAL_FAMILIES:
        return family
    mapping = {
        "finance_quant": "market_data_retrieval" if _source_requests_live_data(source_bundle) else "exact_compute",
        "mathematical_analysis": "analytical_reasoning",
        "risk_analysis": "market_scenario_analysis",
        "transaction_structure_checklist": "transaction_structure_analysis",
        "regulatory_execution_checklist": "regulatory_execution_analysis",
        "tax_structure_checklist": "tax_structure_analysis",
        "market_data": "market_data_retrieval",
        "search": "external_retrieval",
    }
    if family in mapping:
        return mapping[family]
    if intent.task_family == "analytical_reasoning":
        return "analytical_reasoning"
    if intent.task_family == "market_scenario":
        return "market_scenario_analysis"
    return family


def _widen_families(
    intent: TaskIntent,
    source_bundle: SourceBundle,
    normalized: list[str],
    benchmark_overrides: dict[str, Any] | None = None,
) -> list[str]:
    widened = list(normalized)
    if benchmark_tool_selection_active(intent.task_family, benchmark_overrides):
        policy = benchmark_runtime_policy(benchmark_overrides)
        allowed_families = set(policy.get("allowed_families", []))
        for family in ("document_retrieval", "exact_compute"):
            if family not in widened:
                widened.append(family)
        if bool(policy.get("allow_web_fallback", True)):
            if "external_retrieval" not in widened:
                widened.append("external_retrieval")
        return [family for family in widened if not allowed_families or family in allowed_families]
    if intent.task_family in {"finance_quant", "external_retrieval"} and _source_requests_live_data(source_bundle):
        for family in ("market_data_retrieval", "external_retrieval"):
            if family not in widened:
                widened.append(family)
    if intent.task_family in {"document_qa", "external_retrieval"} and _looks_like_document_corpus_query(source_bundle):
        if "document_retrieval" not in widened:
            widened.insert(0, "document_retrieval")
    if intent.task_family == "legal_transactional":
        for family in (
            "legal_playbook_retrieval",
            "transaction_structure_analysis",
            "regulatory_execution_analysis",
            "tax_structure_analysis",
        ):
            if family not in widened:
                widened.append(family)
        if source_bundle.urls and "document_retrieval" not in widened:
            widened.insert(0, "document_retrieval")
    if intent.task_family == "market_scenario":
        for family in ("market_scenario_analysis", "market_data_retrieval"):
            if family not in widened:
                widened.append(family)
    if intent.task_family == "analytical_reasoning":
        for family in ("analytical_reasoning", "exact_compute"):
            if family not in widened:
                widened.append(family)
    if intent.task_family == "finance_options":
        for family in ("options_strategy_analysis", "options_scenario_analysis"):
            if family not in widened:
                widened.append(family)
    return widened


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


def _document_retrieval_candidates(
    registry: dict[str, dict[str, Any]],
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> list[str]:
    candidates = [descriptor for descriptor, _ in _ordered_candidates(registry, "document_retrieval")]
    search_tools = [item["tool_name"] for item in candidates if str(item.get("tool_role", "")) == "search"]
    discover_tools = [item["tool_name"] for item in candidates if str(item.get("tool_role", "")) == "discover"]
    fetch_tools = [item["tool_name"] for item in candidates if str(item.get("tool_role", "")) == "fetch"]
    other_tools = [
        item["tool_name"]
        for item in candidates
        if item["tool_name"] not in {*search_tools, *discover_tools, *fetch_tools}
    ]
    if str((benchmark_overrides or {}).get("benchmark_adapter", "") or "").strip().lower() == "officeqa":
        if "search_officeqa_documents" in search_tools:
            search_tools = ["search_officeqa_documents"]
        elif "search_reference_corpus" in search_tools:
            search_tools = ["search_reference_corpus"]
        other_tools = [tool_name for tool_name in other_tools if tool_name != "search_reference_corpus"]
    selected: list[str] = []
    if source_bundle.urls:
        for tool_name in [*discover_tools, *fetch_tools]:
            if tool_name not in selected:
                selected.append(tool_name)
    for tool_name in search_tools:
        if tool_name not in selected:
            selected.append(tool_name)
    for tool_name in fetch_tools:
        if tool_name not in selected:
            selected.append(tool_name)
    for tool_name in other_tools:
        if tool_name not in selected:
            selected.append(tool_name)
    return selected


def resolve_tool_plan(
    intent: TaskIntent,
    source_bundle: SourceBundle,
    registry: dict[str, dict[str, Any]],
    benchmark_overrides: dict[str, Any] | None = None,
) -> tuple[ToolPlan, dict[str, dict[str, Any]]]:
    mutable_registry = dict(registry)
    benchmark_policy_active = benchmark_tool_selection_active(intent.task_family, benchmark_overrides)
    requested = [_normalize_family(family, intent, source_bundle) for family in intent.tool_families_needed]
    requested = list(dict.fromkeys([family for family in requested if family]))
    widened = _widen_families(intent, source_bundle, requested, benchmark_overrides)
    selected: list[str] = []
    pending: list[str] = []
    blocked: list[str] = []
    notes: list[str] = []
    ace_events: list[dict[str, Any]] = []
    stop_reason = ""

    if intent.task_family == "unsupported_artifact":
        plan = ToolPlan(
            tool_families_needed=requested,
            widened_families=widened,
            selected_tools=[],
            pending_tools=[],
            blocked_families=[],
            ace_events=[],
            notes=["Task requires a non-finance artifact capability outside the active engine scope."],
            stop_reason="unsupported_capability",
        )
        return plan, mutable_registry

    for family in widened:
        if family == "document_retrieval":
            document_tools = _document_retrieval_candidates(mutable_registry, source_bundle, benchmark_overrides)
            if benchmark_policy_active:
                document_tools = [
                    tool_name
                    for tool_name in document_tools
                    if benchmark_descriptor_allowed(dict(mutable_registry.get(tool_name, {}).get("descriptor") or {}), benchmark_overrides)
                ]
            if not document_tools:
                ace_event, synthesized = synthesize_capability(family)
                ace_events.append(ace_event.model_dump())
                if synthesized is not None:
                    descriptor = _descriptor_for_tool(synthesized)
                    mutable_registry[descriptor.tool_name] = {
                        "descriptor": descriptor.model_dump(),
                        "tool": synthesized,
                    }
                    document_tools = _document_retrieval_candidates(mutable_registry, source_bundle, benchmark_overrides)
                    if benchmark_policy_active:
                        document_tools = [
                            tool_name
                            for tool_name in document_tools
                            if benchmark_descriptor_allowed(dict(mutable_registry.get(tool_name, {}).get("descriptor") or {}), benchmark_overrides)
                        ]
            if not document_tools:
                blocked.append(family)
                notes.append(f"No safe tool binding found for {family}.")
                continue
            for tool_name in document_tools:
                if tool_name not in selected:
                    selected.append(tool_name)
            initial_pending = [
                tool_name
                for tool_name in document_tools
                if str(mutable_registry.get(tool_name, {}).get("descriptor", {}).get("tool_role", "")) in {"search", "discover"}
                or (
                    str(mutable_registry.get(tool_name, {}).get("descriptor", {}).get("tool_role", "")) == "fetch"
                    and bool(source_bundle.urls)
                )
            ]
            for tool_name in initial_pending:
                if tool_name not in pending:
                    pending.append(tool_name)
            continue

        candidates = _ordered_candidates(mutable_registry, family)
        if benchmark_policy_active:
            candidates = [item for item in candidates if benchmark_descriptor_allowed(item[0], benchmark_overrides)]
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
                if benchmark_policy_active:
                    candidates = [item for item in candidates if benchmark_descriptor_allowed(item[0], benchmark_overrides)]
        if not candidates:
            blocked.append(family)
            notes.append(f"No safe tool binding found for {family}.")
            continue

        if intent.task_family == "legal_transactional" and family in {
            "legal_playbook_retrieval",
            "transaction_structure_analysis",
            "regulatory_execution_analysis",
            "tax_structure_analysis",
        }:
            family_tools = [descriptor["tool_name"] for descriptor, _ in candidates[:1]]
            selected.extend(family_tools)
            pending.extend(family_tools)
            continue

        descriptor, _ = candidates[0]
        tool_name = str(descriptor["tool_name"])
        selected.append(tool_name)
        if intent.execution_mode == "document_grounded_analysis" and family == "external_retrieval":
            continue
        if family in {
            "market_data_retrieval",
            "analytical_reasoning",
            "exact_compute",
            "market_scenario_analysis",
        }:
            continue
        if tool_name not in pending:
            pending.append(tool_name)

    if intent.task_family == "legal_transactional" and source_bundle.urls and "document_retrieval" not in blocked:
        doc_candidates = _ordered_candidates(mutable_registry, "document_retrieval")
        if doc_candidates:
            tool_name = str(doc_candidates[0][0]["tool_name"])
            if tool_name not in selected:
                selected.append(tool_name)
                pending.insert(0, tool_name)

    if not selected and any(family in {"market_data_retrieval", "external_retrieval", "document_retrieval"} for family in widened):
        stop_reason = "no_bindable_capability"
    elif blocked and len(blocked) == len(widened) and intent.task_family in {"external_retrieval", "finance_quant", "market_scenario"}:
        stop_reason = "no_bindable_capability"

    plan = ToolPlan(
        tool_families_needed=requested,
        widened_families=widened,
        selected_tools=list(dict.fromkeys(selected)),
        pending_tools=list(dict.fromkeys(pending)),
        blocked_families=blocked,
        ace_events=ace_events,
        notes=notes,
        stop_reason=stop_reason,
    )
    return plan, mutable_registry
