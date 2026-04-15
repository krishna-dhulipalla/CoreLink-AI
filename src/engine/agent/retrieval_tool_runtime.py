from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from engine.agent.context.extraction import derive_market_snapshot
from engine.agent.contracts import RetrievalAction, RetrievalIntent, SourceBundle
from engine.agent.solver.options import (
    deterministic_policy_options_tool_call,
    deterministic_standard_options_tool_call,
    scenario_args_from_primary_tool,
)
from engine.agent.state import AgentState
from engine.agent.tools.normalization import normalize_tool_output

RuntimeState = AgentState


def tool_lookup(registry: dict[str, dict[str, Any]], tool_name: str) -> Any | None:
    payload = registry.get(tool_name) or {}
    return payload.get("tool")


def tool_descriptor(registry: dict[str, dict[str, Any]], tool_name: str) -> dict[str, Any]:
    payload = registry.get(tool_name) or {}
    return dict(payload.get("descriptor") or {})


def tool_family(registry: dict[str, dict[str, Any]], tool_name: str) -> str:
    return str(tool_descriptor(registry, tool_name).get("tool_family", "") or "")


def tool_role(registry: dict[str, dict[str, Any]], tool_name: str) -> str:
    return str(tool_descriptor(registry, tool_name).get("tool_role", "") or "")


def filter_args_for_tool(registry: dict[str, dict[str, Any]], tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool_obj = tool_lookup(registry, tool_name)
    arg_schema = dict(getattr(tool_obj, "args", {}) or {})
    if not arg_schema:
        return dict(args)
    return {key: value for key, value in dict(args).items() if key in arg_schema}


def generic_tool_args(
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent | None,
    derive_seed_query: Callable[[SourceBundle, RetrievalIntent | None], str],
) -> dict[str, Any]:
    tool_obj = tool_lookup(registry, tool_name)
    descriptor = tool_descriptor(registry, tool_name)
    arg_schema = dict(getattr(tool_obj, "args", {}) or {})
    if not arg_schema:
        return {}

    task_text = source_bundle.task_text
    focus_query = derive_seed_query(source_bundle, retrieval_intent) or source_bundle.focus_query or task_text[:240]
    args: dict[str, Any] = {}
    active_role = str(descriptor.get("tool_role", "") or "")

    for field_name in arg_schema.keys():
        lowered = str(field_name).lower()
        if lowered in {"query", "search_query", "q", "question"}:
            args[field_name] = focus_query
        elif lowered in {"prompt_text", "task", "task_text", "text", "input"}:
            args[field_name] = task_text
        elif lowered in {"url", "document_url", "source_url", "file_url"} and source_bundle.urls:
            args[field_name] = source_bundle.urls[0]
        elif lowered in {"top_k", "k", "limit", "max_results"} and active_role == "search":
            args[field_name] = 5
        elif lowered in {"snippet_chars", "max_chars"} and active_role == "search":
            args[field_name] = 700
        elif lowered == "page_start":
            args[field_name] = 0
        elif lowered == "page_limit":
            args[field_name] = 5
        elif lowered == "row_offset":
            args[field_name] = 0
        elif lowered == "row_limit":
            args[field_name] = 200
    return args


def structured_tool_args(
    state: RuntimeState,
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    derive_seed_query: Callable[[SourceBundle, RetrievalIntent | None], str],
) -> dict[str, Any]:
    task_text = source_bundle.task_text
    if tool_name == "legal_playbook_retrieval":
        return {"query": source_bundle.focus_query or task_text[:240], "deal_size_hint": "", "urgency": ""}
    if tool_name == "transaction_structure_checklist":
        return {
            "consideration_preference": "stock" if "stock" in task_text.lower() else "",
            "liability_goal": "minimize inherited liabilities" if "liabil" in task_text.lower() else "",
            "urgency": "accelerated" if "quick" in task_text.lower() else "",
        }
    if tool_name == "regulatory_execution_checklist":
        jurisdictions = []
        lowered = task_text.lower()
        if "eu" in lowered:
            jurisdictions.append("EU")
        if re.search(r"\bus\b|\bunited states\b", lowered):
            jurisdictions.append("US")
        return {"jurisdictions_json": json.dumps(jurisdictions), "regulatory_gaps": "regulatory compliance gaps" if "compliance gap" in lowered or "regulatory" in lowered else ""}
    if tool_name == "tax_structure_checklist":
        lowered = task_text.lower()
        return {
            "consideration_preference": "stock" if "stock" in lowered else "",
            "cross_border": bool("eu" in lowered and ("us" in lowered or "united states" in lowered)),
        }
    if tool_name == "internet_search":
        return {"query": derive_seed_query(source_bundle, retrieval_intent) or source_bundle.focus_query or task_text[:240]}
    if tool_name in {"search_reference_corpus", "search_officeqa_documents"}:
        return {
            "query": derive_seed_query(source_bundle, retrieval_intent) or source_bundle.focus_query or task_text[:240],
            "top_k": 8 if tool_name == "search_officeqa_documents" else 5,
            "snippet_chars": 700,
            "source_files": list(source_bundle.source_files_expected),
            "source_files_policy": retrieval_intent.source_constraint_policy,
            "target_years": list(retrieval_intent.target_years),
            "publication_year_window": list(retrieval_intent.publication_year_window),
            "preferred_publication_years": list(retrieval_intent.preferred_publication_years),
            "acceptable_publication_lag_years": retrieval_intent.acceptable_publication_lag_years,
            "retrospective_evidence_allowed": retrieval_intent.retrospective_evidence_allowed,
            "retrospective_evidence_required": retrieval_intent.retrospective_evidence_required,
            "publication_scope_explicit": retrieval_intent.publication_scope_explicit,
            "period_type": retrieval_intent.period_type,
            "granularity_requirement": retrieval_intent.granularity_requirement,
            "entity": retrieval_intent.entity,
            "metric": retrieval_intent.metric,
        }
    if tool_name == "fetch_reference_file":
        if source_bundle.urls:
            return {"url": source_bundle.urls[0], "page_start": 0, "page_limit": 5, "row_offset": 0, "row_limit": 200}
        return {}
    if tool_name == "fetch_officeqa_pages":
        return {}
    if tool_name == "fetch_officeqa_table":
        query = " ".join(part for part in [retrieval_intent.entity, retrieval_intent.metric, retrieval_intent.period] if part).strip()
        return {"table_query": query, "row_offset": 0, "row_limit": 200}
    if tool_name == "lookup_officeqa_rows":
        table_query = " ".join(part for part in [retrieval_intent.entity, retrieval_intent.metric, retrieval_intent.period] if part).strip()
        return {
            "table_query": table_query,
            "row_query": retrieval_intent.entity or retrieval_intent.metric or table_query,
            "row_offset": 0,
            "row_limit": 120,
        }
    if tool_name == "lookup_officeqa_cells":
        table_query = " ".join(part for part in [retrieval_intent.entity, retrieval_intent.metric, retrieval_intent.period] if part).strip()
        return {
            "table_query": table_query,
            "row_query": retrieval_intent.entity or retrieval_intent.metric or table_query,
            "column_query": retrieval_intent.metric or retrieval_intent.period or table_query,
            "row_offset": 0,
            "row_limit": 60,
        }
    if tool_name == "list_reference_files":
        return {"prompt_text": source_bundle.task_text}
    if tool_name == "fetch_corpus_document":
        return {}
    if tool_name == "analyze_strategy":
        inline_facts = dict(source_bundle.inline_facts)
        market_snapshot, derived = derive_market_snapshot(source_bundle.task_text, inline_facts)
        prompt_facts = dict(inline_facts)
        if market_snapshot:
            prompt_facts["market_snapshot"] = market_snapshot
        compat_state = {
            "messages": state.get("messages", []),
            "evidence_pack": {"prompt_facts": prompt_facts, "derived_facts": derived, "policy_context": {}},
            "workpad": {"tool_results": []},
            "assumption_ledger": state.get("assumption_ledger", []),
        }
        tool_call = deterministic_policy_options_tool_call(compat_state) or deterministic_standard_options_tool_call(compat_state)
        if tool_call:
            return dict(tool_call.get("arguments", {}))
        return {}
    if tool_name == "scenario_pnl":
        journal = state.get("execution_journal", {}) or {}
        for result in reversed(list(dict(journal).get("tool_results", []) or [])):
            args = scenario_args_from_primary_tool(result)
            if args:
                return args
        return {}
    return generic_tool_args(registry, tool_name, source_bundle, retrieval_intent, derive_seed_query)


async def invoke_tool(tool_obj: Any, args: dict[str, Any]) -> Any:
    if hasattr(tool_obj, "ainvoke"):
        return await tool_obj.ainvoke(args)
    return tool_obj.invoke(args)


async def run_tool_step(
    state: RuntimeState,
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    derive_seed_query: Callable[[SourceBundle, RetrievalIntent | None], str],
) -> tuple[dict[str, Any], Any]:
    tool_obj = tool_lookup(registry, tool_name)
    args = filter_args_for_tool(registry, tool_name, structured_tool_args(state, registry, tool_name, source_bundle, retrieval_intent, derive_seed_query))
    if tool_obj is None:
        return args, normalize_tool_output(tool_name, {"error": f"Tool '{tool_name}' is not registered."}, args)
    try:
        raw = await invoke_tool(tool_obj, args)
    except Exception as exc:
        raw = {"error": f"Error executing tool {tool_name}: {exc}"}
    return args, normalize_tool_output(tool_name, raw, args)


async def run_tool_step_with_args(
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    args_override: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    tool_obj = tool_lookup(registry, tool_name)
    args_override = filter_args_for_tool(registry, tool_name, args_override)
    if tool_obj is None:
        return args_override, normalize_tool_output(tool_name, {"error": f"Tool '{tool_name}' is not registered."}, args_override)
    try:
        raw = await invoke_tool(tool_obj, args_override)
    except Exception as exc:
        raw = {"error": f"Error executing tool {tool_name}: {exc}"}
    return args_override, normalize_tool_output(tool_name, raw, args_override)


def tool_args_from_retrieval_action(
    action: RetrievalAction,
    source_bundle: SourceBundle,
    registry: dict[str, dict[str, Any]],
    retrieval_intent: RetrievalIntent,
    derive_seed_query: Callable[[SourceBundle, RetrievalIntent | None], str],
    officeqa_table_query: Callable[[RetrievalIntent, SourceBundle], str],
    officeqa_row_query: Callable[[RetrievalIntent, SourceBundle], str],
    officeqa_column_query: Callable[[RetrievalIntent], str],
) -> dict[str, Any]:
    if action.tool_name == "internet_search":
        return {"query": action.query or derive_seed_query(source_bundle, retrieval_intent)}
    if action.tool_name in {"search_officeqa_documents", "search_reference_corpus"}:
        return {
            "query": action.query or derive_seed_query(source_bundle, retrieval_intent),
            "top_k": 5,
            "snippet_chars": 700,
            "source_files": list(source_bundle.source_files_expected),
            "source_files_policy": retrieval_intent.source_constraint_policy,
            "target_years": list(retrieval_intent.target_years),
            "publication_year_window": list(retrieval_intent.publication_year_window),
            "preferred_publication_years": list(retrieval_intent.preferred_publication_years),
            "acceptable_publication_lag_years": retrieval_intent.acceptable_publication_lag_years,
            "retrospective_evidence_allowed": retrieval_intent.retrospective_evidence_allowed,
            "retrospective_evidence_required": retrieval_intent.retrospective_evidence_required,
            "publication_scope_explicit": retrieval_intent.publication_scope_explicit,
            "period_type": retrieval_intent.period_type,
            "granularity_requirement": retrieval_intent.granularity_requirement,
            "entity": retrieval_intent.entity,
            "metric": retrieval_intent.metric,
        }
    if action.tool_name == "list_reference_files":
        return {"prompt_text": source_bundle.task_text}
    if action.tool_name == "fetch_reference_file":
        args = {
            "url": action.url,
            "page_start": action.page_start,
            "page_limit": max(2, action.page_limit),
            "row_offset": action.row_offset,
            "row_limit": max(100, action.row_limit),
        }
        tool_obj = tool_lookup(registry, action.tool_name)
        arg_schema = {str(key).lower(): key for key in dict(getattr(tool_obj, "args", {}) or {}).keys()}
        hint = action.query or derive_seed_query(source_bundle, retrieval_intent)
        if "search_hint" in arg_schema and hint:
            args[arg_schema["search_hint"]] = hint
        return args
    if action.tool_name == "fetch_corpus_document":
        return {"document_id": action.document_id, "path": action.path, "chunk_start": action.chunk_start, "chunk_limit": max(1, action.chunk_limit)}
    if action.tool_name == "fetch_officeqa_pages":
        return {"document_id": action.document_id, "path": action.path, "page_start": action.page_start, "page_limit": max(1, action.page_limit)}
    if action.tool_name == "fetch_officeqa_table":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "table_query": action.query or officeqa_table_query(retrieval_intent, source_bundle),
            "row_offset": action.row_offset,
            "row_limit": max(50, action.row_limit or 200),
        }
    if action.tool_name == "lookup_officeqa_rows":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "table_query": officeqa_table_query(retrieval_intent, source_bundle),
            "row_query": action.query or officeqa_row_query(retrieval_intent, source_bundle),
            "row_offset": action.row_offset,
            "row_limit": max(20, action.row_limit or 120),
        }
    if action.tool_name == "lookup_officeqa_cells":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "table_query": officeqa_table_query(retrieval_intent, source_bundle),
            "row_query": officeqa_row_query(retrieval_intent, source_bundle),
            "column_query": action.query or officeqa_column_query(retrieval_intent),
            "row_offset": action.row_offset,
            "row_limit": max(10, action.row_limit or 60),
        }
    args = generic_tool_args(registry, action.tool_name, source_bundle, retrieval_intent, derive_seed_query)
    tool_obj = tool_lookup(registry, action.tool_name)
    arg_schema = dict(getattr(tool_obj, "args", {}) or {})
    for field_name in arg_schema.keys():
        lowered = str(field_name).lower()
        if lowered in {"query", "search_query", "q", "question"} and action.query:
            args[field_name] = action.query
        elif lowered in {"url", "document_url", "source_url", "file_url"} and action.url:
            args[field_name] = action.url
        elif lowered in {"document_id", "doc_id"} and action.document_id:
            args[field_name] = action.document_id
        elif lowered in {"path", "file_path", "citation"} and action.path:
            args[field_name] = action.path
        elif lowered == "page_start":
            args[field_name] = action.page_start
        elif lowered == "page_limit":
            args[field_name] = max(2, action.page_limit)
        elif lowered == "row_offset":
            args[field_name] = action.row_offset
        elif lowered == "row_limit":
            args[field_name] = max(100, action.row_limit)
        elif lowered == "chunk_start":
            args[field_name] = action.chunk_start
        elif lowered == "chunk_limit":
            args[field_name] = max(1, action.chunk_limit)
    return args
