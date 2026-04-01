from __future__ import annotations

import json
import os
import re
from typing import Any

from agent.benchmarks import benchmark_runtime_policy
from agent.context.extraction import derive_market_snapshot
from agent.contracts import ExecutionJournal, RetrievalAction, RetrievalIntent, SourceBundle, ToolPlan
from agent.retrieval_reasoning import assess_evidence_sufficiency
from agent.solver.options import (
    deterministic_policy_options_tool_call,
    deterministic_standard_options_tool_call,
    scenario_args_from_primary_tool,
)
from agent.state import AgentState
from agent.tools.normalization import normalize_tool_output

RuntimeState = AgentState
IS_COMPETITION_MODE = os.getenv("COMPETITION_MODE", "").strip().lower() in {"1", "true", "yes", "on"} or os.getenv("BENCHMARK_NAME", "").strip().lower() == "officeqa"
MAX_RETRIEVAL_HOPS = max(2, int(os.getenv("MAX_RETRIEVAL_HOPS", "10" if IS_COMPETITION_MODE else "4")))
_RETRIEVAL_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "which",
    "according",
    "report",
    "document",
    "source",
    "using",
    "based",
}


def _tool_lookup(registry: dict[str, dict[str, Any]], tool_name: str) -> Any | None:
    payload = registry.get(tool_name) or {}
    return payload.get("tool")


def _tool_descriptor(registry: dict[str, dict[str, Any]], tool_name: str) -> dict[str, Any]:
    payload = registry.get(tool_name) or {}
    return dict(payload.get("descriptor") or {})


def _tool_family(registry: dict[str, dict[str, Any]], tool_name: str) -> str:
    return str(_tool_descriptor(registry, tool_name).get("tool_family", "") or "")


def _tool_role(registry: dict[str, dict[str, Any]], tool_name: str) -> str:
    return str(_tool_descriptor(registry, tool_name).get("tool_role", "") or "")


def _generic_tool_args(
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent | None = None,
) -> dict[str, Any]:
    tool_obj = _tool_lookup(registry, tool_name)
    descriptor = _tool_descriptor(registry, tool_name)
    arg_schema = dict(getattr(tool_obj, "args", {}) or {})
    if not arg_schema:
        return {}

    task_text = source_bundle.task_text
    focus_query = (
        (retrieval_intent.query_candidates[0] if retrieval_intent and retrieval_intent.query_candidates else "")
        or source_bundle.focus_query
        or task_text[:240]
    )
    args: dict[str, Any] = {}
    tool_role = str(descriptor.get("tool_role", "") or "")

    for field_name in arg_schema.keys():
        lowered = str(field_name).lower()
        if lowered in {"query", "search_query", "q", "question"}:
            args[field_name] = focus_query
        elif lowered in {"prompt_text", "task", "task_text", "text", "input"}:
            args[field_name] = task_text
        elif lowered in {"url", "document_url", "source_url", "file_url"} and source_bundle.urls:
            args[field_name] = source_bundle.urls[0]
        elif lowered in {"top_k", "k", "limit", "max_results"} and tool_role == "search":
            args[field_name] = 5
        elif lowered in {"snippet_chars", "max_chars"} and tool_role == "search":
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


def _structured_tool_args(state: RuntimeState, registry: dict[str, dict[str, Any]], tool_name: str) -> dict[str, Any]:
    source_bundle = SourceBundle.model_validate(state.get("source_bundle") or {})
    retrieval_intent = RetrievalIntent.model_validate(state.get("retrieval_intent") or {})
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
        query = (retrieval_intent.query_candidates[0] if retrieval_intent.query_candidates else "") or source_bundle.focus_query or task_text[:240]
        return {"query": query}
    if tool_name == "search_reference_corpus":
        query = (retrieval_intent.query_candidates[0] if retrieval_intent.query_candidates else "") or source_bundle.focus_query or task_text[:240]
        return {
            "query": query,
            "top_k": 5,
            "snippet_chars": 700,
            "source_files": source_bundle.source_files_expected[:8],
        }
    if tool_name == "search_officeqa_documents":
        query = (retrieval_intent.query_candidates[0] if retrieval_intent.query_candidates else "") or source_bundle.focus_query or task_text[:240]
        return {
            "query": query,
            "top_k": 5,
            "snippet_chars": 700,
            "source_files": source_bundle.source_files_expected[:8],
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
            "evidence_pack": {
                "prompt_facts": prompt_facts,
                "derived_facts": derived,
                "policy_context": {},
            },
            "workpad": {"tool_results": []},
            "assumption_ledger": state.get("assumption_ledger", []),
        }
        tool_call = deterministic_policy_options_tool_call(compat_state) or deterministic_standard_options_tool_call(compat_state)
        if tool_call:
            return dict(tool_call.get("arguments", {}))
        return {}
    if tool_name == "scenario_pnl":
        journal = ExecutionJournal.model_validate(state.get("execution_journal") or {})
        for result in reversed(journal.tool_results):
            args = scenario_args_from_primary_tool(result)
            if args:
                return args
        return {}
    return _generic_tool_args(registry, tool_name, source_bundle, retrieval_intent)


async def _invoke_tool(tool_obj: Any, args: dict[str, Any]) -> Any:
    if hasattr(tool_obj, "ainvoke"):
        return await tool_obj.ainvoke(args)
    return tool_obj.invoke(args)


async def _run_tool_step(state: RuntimeState, registry: dict[str, dict[str, Any]], tool_name: str) -> tuple[dict[str, Any], Any]:
    tool_obj = _tool_lookup(registry, tool_name)
    args = _structured_tool_args(state, registry, tool_name)
    if tool_obj is None:
        return args, normalize_tool_output(tool_name, {"error": f"Tool '{tool_name}' is not registered."}, args)
    try:
        raw = await _invoke_tool(tool_obj, args)
    except Exception as exc:
        raw = {"error": f"Error executing tool {tool_name}: {exc}"}
    return args, normalize_tool_output(tool_name, raw, args)


async def _run_tool_step_with_args(
    state: RuntimeState,
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    args_override: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    tool_obj = _tool_lookup(registry, tool_name)
    if tool_obj is None:
        return args_override, normalize_tool_output(tool_name, {"error": f"Tool '{tool_name}' is not registered."}, args_override)
    try:
        raw = await _invoke_tool(tool_obj, args_override)
    except Exception as exc:
        raw = {"error": f"Error executing tool {tool_name}: {exc}"}
    return args_override, normalize_tool_output(tool_name, raw, args_override)


def _retrieval_tools_available(
    tool_plan: ToolPlan,
    registry: dict[str, dict[str, Any]],
) -> list[str]:
    retrieval_tools = []
    for tool_name in tool_plan.selected_tools:
        descriptor = _tool_descriptor(registry, tool_name)
        if descriptor.get("tool_family") in {"document_retrieval", "external_retrieval"}:
            retrieval_tools.append(tool_name)
    return retrieval_tools


def _retrieval_tools_by_role(
    tool_plan: ToolPlan,
    registry: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {"discover": [], "search": [], "fetch": [], "other": []}
    for tool_name in _retrieval_tools_available(tool_plan, registry):
        role = str(_tool_descriptor(registry, tool_name).get("tool_role", "") or "")
        bucket = role if role in grouped else "other"
        grouped[bucket].append(tool_name)
    return grouped


def _derive_retrieval_seed_query(source_bundle: SourceBundle, retrieval_intent: RetrievalIntent | None = None) -> str:
    if retrieval_intent and retrieval_intent.query_candidates:
        return retrieval_intent.query_candidates[0]
    parts = [source_bundle.focus_query or source_bundle.task_text]
    if source_bundle.entities:
        parts.append(" ".join(source_bundle.entities[:4]))
    if source_bundle.target_period:
        parts.append(source_bundle.target_period)
    seed = " ".join(part.strip() for part in parts if part and part.strip())
    return re.sub(r"\s+", " ", seed).strip()[:280]


def _next_retrieval_query(journal: ExecutionJournal, retrieval_intent: RetrievalIntent, source_bundle: SourceBundle) -> str:
    used = {re.sub(r"\s+", " ", query).strip().lower() for query in journal.retrieval_queries}
    for candidate in retrieval_intent.query_candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip().lower()
        if normalized and normalized not in used:
            return candidate
    return _derive_retrieval_seed_query(source_bundle, retrieval_intent)


def _retrieval_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 1 and token not in _RETRIEVAL_STOP_WORDS
    ]


def _retrieval_focus_tokens(source_bundle: SourceBundle) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for part in [source_bundle.focus_query, *source_bundle.entities[:6], source_bundle.target_period]:
        for token in _retrieval_tokens(str(part or "")):
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
    return ordered[:14]


def _officeqa_table_query(retrieval_intent: RetrievalIntent, source_bundle: SourceBundle) -> str:
    parts = [
        retrieval_intent.entity,
        retrieval_intent.metric,
        retrieval_intent.period,
        source_bundle.focus_query,
    ]
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()[:280]


def _officeqa_row_query(retrieval_intent: RetrievalIntent, source_bundle: SourceBundle) -> str:
    return re.sub(
        r"\s+",
        " ",
        " ".join(part for part in [retrieval_intent.entity, source_bundle.focus_query, retrieval_intent.period] if part),
    ).strip()[:240]


def _officeqa_column_query(retrieval_intent: RetrievalIntent) -> str:
    parts = [retrieval_intent.metric]
    if retrieval_intent.aggregation_shape.startswith("monthly"):
        parts.append("month")
    if retrieval_intent.period:
        parts.append(retrieval_intent.period)
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()[:200]


def _retrieval_content_text(facts: dict[str, Any]) -> str:
    parts: list[str] = []
    for chunk in facts.get("chunks", []):
        if isinstance(chunk, dict):
            parts.append(str(chunk.get("text", "")))
            parts.append(str(chunk.get("locator", "")))
            parts.append(str(chunk.get("citation", "")))
    for table in facts.get("tables", []):
        if isinstance(table, dict):
            parts.append(json.dumps(table, ensure_ascii=True))
    for summary in facts.get("numeric_summaries", []):
        if isinstance(summary, dict):
            parts.append(json.dumps(summary, ensure_ascii=True))
    metadata = dict(facts.get("metadata") or {})
    for key in ("file_name", "window"):
        if metadata.get(key):
            parts.append(str(metadata.get(key)))
    for key in ("citation", "document_id"):
        if facts.get(key):
            parts.append(str(facts.get(key)))
    return " ".join(part for part in parts if part).strip()


def _retrieved_evidence_is_sufficient(
    source_bundle: SourceBundle,
    tool_result: dict[str, Any],
    benchmark_overrides: dict[str, Any] | None = None,
) -> bool:
    if str(tool_result.get("retrieval_status", "") or "") in {"garbled_binary", "parse_error", "network_error", "unsupported_format", "empty", "irrelevant"}:
        return False
    sufficiency = assess_evidence_sufficiency(source_bundle.task_text, source_bundle, [tool_result], benchmark_overrides)
    return bool(sufficiency.is_sufficient)


def _retrieved_window_is_promising(
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    tool_result: dict[str, Any],
    benchmark_overrides: dict[str, Any] | None = None,
) -> bool:
    if str(tool_result.get("retrieval_status", "") or "") in {"garbled_binary", "parse_error", "network_error", "unsupported_format", "empty", "irrelevant"}:
        return False
    facts = dict(tool_result.get("facts") or {})
    text = _retrieval_content_text(facts).lower()
    entity_tokens = {token for token in _retrieval_tokens(retrieval_intent.entity) if token not in {"u", "s", "us"}}
    metric_basis = retrieval_intent.metric
    if retrieval_intent.aggregation_shape in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"} or retrieval_intent.metric in {"absolute percent change", "absolute difference"}:
        metric_basis = "expenditures"
    metric_tokens = set(_retrieval_tokens(metric_basis))
    period_tokens = set(_retrieval_tokens(retrieval_intent.period))
    overlap = len((entity_tokens | metric_tokens | period_tokens) & set(_retrieval_tokens(text)))
    citation = str(facts.get("citation", "") or dict(facts.get("metadata") or {}).get("file_name", "")).lower()
    if not (entity_tokens or metric_tokens or period_tokens) and any(token in citation for token in ("treasury", "bulletin", "statement", "budget")):
        return True
    if overlap >= 3:
        return True
    if overlap >= 2 and any(token in citation for token in ("treasury", "budget", "census", "annual", "statement")):
        return True
    if any(token in citation for token in ("treasury", "bulletin", "statement", "budget")) and overlap >= 1:
        return True
    return False


def _next_corpus_fetch_action(last_result: dict[str, Any]) -> RetrievalAction | None:
    facts = dict(last_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    assumptions = dict(last_result.get("assumptions") or {})
    if not metadata.get("has_more_chunks"):
        return None
    chunk_start = int(assumptions.get("chunk_start", metadata.get("chunk_start", 0)) or 0)
    chunk_limit = max(1, int(assumptions.get("chunk_limit", metadata.get("chunk_limit", 3)) or 3))
    document_id = str(assumptions.get("document_id", facts.get("document_id", "")) or "")
    path = str(assumptions.get("path", facts.get("citation", "")) or "")
    return RetrievalAction(
        action="tool",
        tool_name="fetch_corpus_document",
        stage="locate_pages",
        document_id=document_id,
        path=path,
        chunk_start=chunk_start + chunk_limit,
        chunk_limit=chunk_limit,
        rationale="Read the next document window because the current chunk is not sufficient yet.",
    )


def _next_officeqa_page_action(last_result: dict[str, Any], tool_name: str = "fetch_officeqa_pages") -> RetrievalAction | None:
    facts = dict(last_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    assumptions = dict(last_result.get("assumptions") or {})
    has_more = bool(metadata.get("has_more_windows") or metadata.get("has_more_chunks"))
    if not has_more:
        return None
    page_start = int(assumptions.get("page_start", metadata.get("page_start", metadata.get("chunk_start", 0))) or 0)
    page_limit = max(1, int(assumptions.get("page_limit", metadata.get("page_limit", metadata.get("chunk_limit", 5))) or 5))
    return RetrievalAction(
        action="tool",
        tool_name=tool_name,
        stage="locate_pages",
        document_id=str(assumptions.get("document_id", facts.get("document_id", "")) or ""),
        path=str(assumptions.get("path", facts.get("citation", "")) or ""),
        page_start=page_start + page_limit,
        page_limit=page_limit,
        rationale="Read the next OfficeQA page window because the current pages are still incomplete.",
    )


def _next_reference_fetch_action(last_result: dict[str, Any], tool_name: str) -> RetrievalAction | None:
    facts = dict(last_result.get("facts") or {})
    metadata = dict(facts.get("metadata") or {})
    assumptions = dict(last_result.get("assumptions") or {})
    if not metadata.get("has_more_windows"):
        return None
    url = str(assumptions.get("url", facts.get("citation", "")) or "")
    document_id = str(assumptions.get("document_id", facts.get("document_id", "")) or "")
    path = str(assumptions.get("path", facts.get("citation", "")) or "")
    if not (url or document_id or path):
        return None
    window_kind = str(metadata.get("window_kind", "") or "").lower()
    if window_kind == "pages":
        page_start = int(assumptions.get("page_start", 0) or 0)
        page_limit = max(1, int(assumptions.get("page_limit", 5) or 5))
        return RetrievalAction(
            action="tool",
            tool_name=tool_name,
            url=url,
            document_id=document_id,
            path=path,
            page_start=page_start + page_limit,
            page_limit=page_limit,
            row_offset=int(assumptions.get("row_offset", 0) or 0),
            row_limit=max(100, int(assumptions.get("row_limit", 200) or 200)),
            rationale="Read the next page window because the current pages are not sufficient yet.",
        )
    if window_kind == "rows":
        row_offset = int(assumptions.get("row_offset", 0) or 0)
        row_limit = max(1, int(assumptions.get("row_limit", 200) or 200))
        return RetrievalAction(
            action="tool",
            tool_name=tool_name,
            url=url,
            document_id=document_id,
            path=path,
            page_start=int(assumptions.get("page_start", 0) or 0),
            page_limit=max(2, int(assumptions.get("page_limit", 5) or 5)),
            row_offset=row_offset + row_limit,
            row_limit=row_limit,
            rationale="Read the next row window because the current rows are not sufficient yet.",
        )
    return None


def _tool_result_citations(tool_result: dict[str, Any]) -> list[str]:
    facts = dict(tool_result.get("facts") or {})
    citations: list[str] = []
    direct = [facts.get("citation", "")]
    for item in facts.get("results", []):
        if isinstance(item, dict):
            direct.append(item.get("url", "") or item.get("citation", ""))
    for item in facts.get("documents", []):
        if isinstance(item, dict):
            direct.append(item.get("citation", "") or item.get("url", "") or item.get("path", ""))
    for item in facts.get("chunks", []):
        if isinstance(item, dict):
            direct.append(item.get("citation", ""))
    seen: set[str] = set()
    for raw in direct:
        citation = str(raw or "").strip()
        if not citation or citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
    return citations


def _search_result_candidates(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    facts = dict(tool_result.get("facts") or {})
    candidates: list[dict[str, Any]] = []
    for item in facts.get("documents", []):
        if isinstance(item, dict):
            candidates.append(
                {
                    "document_id": str(item.get("document_id", "")),
                    "citation": str(item.get("citation", "") or item.get("url", "") or item.get("path", "")),
                    "path": str(item.get("path", "")),
                    "title": str(item.get("title", "") or item.get("document_id", "")),
                    "snippet": str(item.get("snippet", "")),
                    "rank": int(item.get("rank", 999) or 999),
                }
            )
    for item in facts.get("results", []):
        if isinstance(item, dict):
            candidates.append(
                {
                    "document_id": str(item.get("document_id", "")),
                    "citation": str(item.get("url", "") or item.get("citation", "")),
                    "path": str(item.get("path", "")),
                    "title": str(item.get("title", "")),
                    "snippet": str(item.get("snippet", "")),
                    "rank": int(item.get("rank", 999) or 999),
                }
            )
    return [candidate for candidate in candidates if candidate.get("citation") or candidate.get("document_id") or candidate.get("path")]


def _search_candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key, "") or "")
        for key in ("title", "snippet", "citation", "path", "document_id")
    ).strip()


def _search_candidate_score(
    candidate: dict[str, Any],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> float:
    text = _search_candidate_text(candidate).lower()
    tokens = set(_retrieval_tokens(text))
    title_tokens = set(_retrieval_tokens(str(candidate.get("title", ""))))
    score = 0.0
    rank = int(candidate.get("rank", 999) or 999)
    score += max(0.0, 0.4 - 0.05 * max(0, rank - 1))

    entity_tokens = {token for token in _retrieval_tokens(retrieval_intent.entity) if token not in {"u", "s", "us"}}
    metric_basis = retrieval_intent.metric
    if retrieval_intent.aggregation_shape in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"} or retrieval_intent.metric in {"absolute percent change", "absolute difference"}:
        metric_basis = "expenditures"
    metric_tokens = {token for token in _retrieval_tokens(metric_basis)}
    period_tokens = {token for token in _retrieval_tokens(retrieval_intent.period)}
    must_tokens = {token for term in retrieval_intent.must_include_terms for token in _retrieval_tokens(term)}
    query_tokens = set(_retrieval_focus_tokens(source_bundle))

    overlap = len((entity_tokens | metric_tokens | period_tokens | must_tokens | query_tokens) & tokens)
    score += 0.18 * overlap
    score += 0.12 * len(entity_tokens & title_tokens)
    score += 0.08 * len(metric_tokens & title_tokens)
    score += 0.06 * len(period_tokens & title_tokens)

    citation = str(candidate.get("citation", "")).lower()
    if any(host in citation for host in ("govinfo.gov", "census.gov", "va.gov", "fraser.stlouisfed.org", ".gov/")):
        score += 0.45
    if citation.endswith(".pdf"):
        score += 0.08

    if retrieval_intent.aggregation_shape.startswith("monthly"):
        if any(token in text for token in ("monthly", "month", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")):
            score += 0.45
        if "receipts expenditures and balances" in text or "monthly treasury statement" in text:
            score += 0.6
    if retrieval_intent.aggregation_shape == "inflation_adjusted_monthly_difference" and any(token in text for token in ("cpi", "inflation", "price index")):
        score += 0.4
    if retrieval_intent.document_family == "treasury_bulletin" and "treasury bulletin" in text:
        score += 0.5

    if any(term in text for term in retrieval_intent.must_exclude_terms):
        score -= 0.7
    if any(
        bad in text
        for bad in (
            "monthly catalog",
            "public documents",
            "depository invoice",
            "federal register",
            "internal revenue bulletin",
            "cumulative bulletin",
            "flashcards",
            "quiz",
            "public law",
        )
    ):
        score -= 1.1
    return score


def _rank_search_candidates(
    candidates: list[dict[str, Any]],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            _search_candidate_score(item, retrieval_intent, source_bundle, benchmark_overrides),
            -int(item.get("rank", 999) or 999),
        ),
        reverse=True,
    )


def _fallback_retrieval_action(
    *,
    execution_mode: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    journal: ExecutionJournal,
    tool_plan: ToolPlan,
    registry: dict[str, dict[str, Any]],
    benchmark_overrides: dict[str, Any] | None = None,
) -> RetrievalAction:
    available = _retrieval_tools_available(tool_plan, registry)
    roles = _retrieval_tools_by_role(tool_plan, registry)
    document_search_tools = [name for name in roles["search"] if _tool_family(registry, name) == "document_retrieval"]
    document_fetch_tools = [name for name in roles["fetch"] if _tool_family(registry, name) == "document_retrieval"]
    discover_tools = [name for name in roles["discover"] if _tool_family(registry, name) == "document_retrieval"]
    external_search_tools = [name for name in roles["search"] if _tool_family(registry, name) == "external_retrieval"]
    last_result = journal.tool_results[-1] if journal.tool_results else {}
    last_type = str(last_result.get("type", ""))
    last_status = str(last_result.get("retrieval_status", "") or "")
    candidates = _rank_search_candidates(
        _search_result_candidates(last_result),
        retrieval_intent,
        source_bundle,
        benchmark_overrides,
    )
    seed_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
    overrides = dict(benchmark_overrides or {})
    officeqa_mode = str(overrides.get("benchmark_adapter") or "") == "officeqa"
    benchmark_policy = benchmark_runtime_policy(overrides)
    allow_web_fallback = bool(benchmark_policy.get("allow_web_fallback", True))
    corpus_grounded_only = execution_mode == "document_grounded_analysis" and officeqa_mode and not allow_web_fallback
    indexed_source_matches = list(source_bundle.source_files_found[:4])
    officeqa_search_tools = [name for name in document_search_tools if name == "search_officeqa_documents"]
    officeqa_table_tools = [name for name in document_fetch_tools if name == "fetch_officeqa_table"]
    officeqa_row_tools = [name for name in document_fetch_tools if name == "lookup_officeqa_rows"]
    officeqa_cell_tools = [name for name in document_fetch_tools if name == "lookup_officeqa_cells"]
    officeqa_page_tools = [name for name in document_fetch_tools if name in {"fetch_officeqa_pages", "fetch_corpus_document"}]
    officeqa_table_query = _officeqa_table_query(retrieval_intent, source_bundle)
    officeqa_row_query = _officeqa_row_query(retrieval_intent, source_bundle)
    officeqa_column_query = _officeqa_column_query(retrieval_intent)

    if officeqa_mode:
        last_facts = dict(last_result.get("facts") or {})
        last_metadata = dict(last_facts.get("metadata") or {})
        officeqa_status = str(last_metadata.get("officeqa_status", "") or "")

        if not journal.tool_results:
            if indexed_source_matches:
                first_match = indexed_source_matches[0]
                if officeqa_table_tools:
                    return RetrievalAction(
                        action="tool",
                        stage="locate_table",
                        tool_name=officeqa_table_tools[0],
                        document_id=str(first_match.get("document_id", "")),
                        path=str(first_match.get("relative_path", "")),
                        query=officeqa_table_query,
                        rationale="Start from the benchmark-linked source file and extract the relevant table first.",
                    )
                if officeqa_page_tools:
                    return RetrievalAction(
                        action="tool",
                        stage="locate_pages",
                        tool_name=officeqa_page_tools[0],
                        document_id=str(first_match.get("document_id", "")),
                        path=str(first_match.get("relative_path", "")),
                        rationale="Start from the benchmark-linked source file and read the relevant pages.",
                    )
            if officeqa_search_tools:
                return RetrievalAction(
                    action="tool",
                    stage="identify_source",
                    tool_name=officeqa_search_tools[0],
                    query=seed_query,
                    rationale="Identify the best OfficeQA source document before extraction.",
                )
            if document_search_tools:
                return RetrievalAction(
                    action="tool",
                    stage="identify_source",
                    tool_name=document_search_tools[0],
                    query=seed_query,
                    rationale="Search the packaged OfficeQA corpus for a grounded source document.",
                )
            if source_bundle.urls and officeqa_page_tools:
                return RetrievalAction(
                    action="tool",
                    stage="locate_pages",
                    tool_name=officeqa_page_tools[0],
                    url=source_bundle.urls[0],
                    rationale="Read the first supplied reference document.",
                )
            if external_search_tools and allow_web_fallback:
                return RetrievalAction(
                    action="tool",
                    stage="identify_source",
                    tool_name=external_search_tools[0],
                    query=seed_query,
                    rationale="Use web search only as an explicit OfficeQA fallback.",
                )
            return RetrievalAction(action="answer", stage="answer", rationale="No OfficeQA retrieval tools are available.")

        if _tool_role(registry, last_type) in {"search", "discover"} and last_status in {"empty", "irrelevant"}:
            next_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
            if next_query and next_query != (journal.retrieval_queries[-1] if journal.retrieval_queries else ""):
                return RetrievalAction(
                    action="tool",
                    stage="identify_source",
                    tool_name=last_type,
                    query=next_query,
                    rationale="Refine OfficeQA source search because the prior results were weak.",
                )

        if _tool_role(registry, last_type) in {"search", "discover"} and candidates:
            first = candidates[0]
            best_score = _search_candidate_score(first, retrieval_intent, source_bundle, benchmark_overrides)
            next_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
            if best_score < 0.55 and next_query and next_query != (journal.retrieval_queries[-1] if journal.retrieval_queries else ""):
                return RetrievalAction(
                    action="tool",
                    stage="identify_source",
                    tool_name=last_type,
                    query=next_query,
                    rationale="Refine OfficeQA source search because the top result is still off-target.",
                )
            if officeqa_table_tools:
                return RetrievalAction(
                    action="tool",
                    stage="locate_table",
                    tool_name=officeqa_table_tools[0],
                    document_id=first.get("document_id", ""),
                    path=first.get("path", "") or first.get("citation", ""),
                    query=officeqa_table_query,
                    rationale="Open the best matching OfficeQA document and extract the relevant table first.",
                )
            if officeqa_page_tools:
                return RetrievalAction(
                    action="tool",
                    stage="locate_pages",
                    tool_name=officeqa_page_tools[0],
                    document_id=first.get("document_id", ""),
                    path=first.get("path", "") or first.get("citation", ""),
                    rationale="Open the best matching OfficeQA document and inspect the relevant pages.",
                )

        if last_type == "fetch_officeqa_table":
            if _retrieved_evidence_is_sufficient(source_bundle, last_result, overrides):
                return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA table evidence is sufficient for the final answer.")
            if officeqa_status == "ok" and officeqa_row_tools and last_facts.get("tables"):
                return RetrievalAction(
                    action="tool",
                    stage="extract_rows",
                    tool_name=officeqa_row_tools[0],
                    document_id=str((last_result.get("assumptions") or {}).get("document_id", last_facts.get("document_id", "")) or ""),
                    path=str((last_result.get("assumptions") or {}).get("path", last_facts.get("citation", "")) or ""),
                    query=officeqa_row_query,
                    rationale="Narrow the OfficeQA table down to the target rows before computing.",
                )
            if officeqa_page_tools and officeqa_status in {"missing_table", "partial_table", "unit_ambiguity"}:
                return RetrievalAction(
                    action="tool",
                    stage="locate_pages",
                    tool_name=officeqa_page_tools[0],
                    document_id=str((last_result.get("assumptions") or {}).get("document_id", last_facts.get("document_id", "")) or ""),
                    path=str((last_result.get("assumptions") or {}).get("path", last_facts.get("citation", "")) or ""),
                    rationale="Fallback to page inspection because OfficeQA table extraction was incomplete.",
                )

        if last_type == "lookup_officeqa_rows":
            if _retrieved_evidence_is_sufficient(source_bundle, last_result, overrides):
                return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA row evidence is sufficient for the final answer.")
            if officeqa_cell_tools and officeqa_status == "ok" and last_facts.get("tables"):
                return RetrievalAction(
                    action="tool",
                    stage="extract_cells",
                    tool_name=officeqa_cell_tools[0],
                    document_id=str((last_result.get("assumptions") or {}).get("document_id", last_facts.get("document_id", "")) or ""),
                    path=str((last_result.get("assumptions") or {}).get("path", last_facts.get("citation", "")) or ""),
                    query=officeqa_column_query,
                    rationale="Narrow the OfficeQA rows down to the target cells before computing.",
                )
            if officeqa_page_tools and officeqa_status in {"missing_row", "partial_table", "unit_ambiguity"}:
                return RetrievalAction(
                    action="tool",
                    stage="locate_pages",
                    tool_name=officeqa_page_tools[0],
                    document_id=str((last_result.get("assumptions") or {}).get("document_id", last_facts.get("document_id", "")) or ""),
                    path=str((last_result.get("assumptions") or {}).get("path", last_facts.get("citation", "")) or ""),
                    rationale="Fallback to page inspection because row extraction was incomplete.",
                )

        if last_type == "lookup_officeqa_cells":
            if _retrieved_evidence_is_sufficient(source_bundle, last_result, overrides):
                return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA cell evidence is sufficient for the final answer.")
            if officeqa_page_tools and officeqa_status in {"partial_table", "unit_ambiguity"}:
                return RetrievalAction(
                    action="tool",
                    stage="locate_pages",
                    tool_name=officeqa_page_tools[0],
                    document_id=str((last_result.get("assumptions") or {}).get("document_id", last_facts.get("document_id", "")) or ""),
                    path=str((last_result.get("assumptions") or {}).get("path", last_facts.get("citation", "")) or ""),
                    rationale="Inspect the OfficeQA pages directly because cell extraction is still ambiguous.",
                )

        if last_type in {"fetch_officeqa_pages", "fetch_corpus_document"}:
            if _retrieved_evidence_is_sufficient(source_bundle, last_result, overrides):
                return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA page evidence is sufficient for the final answer.")
            next_pages = _next_officeqa_page_action(last_result, last_type if last_type in {"fetch_officeqa_pages", "fetch_corpus_document"} else "fetch_officeqa_pages")
            if next_pages is not None and _retrieved_window_is_promising(source_bundle, retrieval_intent, last_result, overrides):
                return next_pages

        if journal.retrieval_iterations >= MAX_RETRIEVAL_HOPS - 1:
            return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA retrieval hop budget exhausted.")
        if not corpus_grounded_only and external_search_tools and allow_web_fallback and last_type != external_search_tools[0]:
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=external_search_tools[0],
                query=_next_retrieval_query(journal, retrieval_intent, source_bundle),
                rationale="Use explicit OfficeQA web fallback after corpus retrieval failed.",
            )
        return RetrievalAction(action="answer", stage="answer", rationale="No stronger OfficeQA retrieval action is available.")

    if not journal.tool_results:
        if indexed_source_matches and document_fetch_tools:
            first_match = indexed_source_matches[0]
            return RetrievalAction(
                action="tool",
                stage="locate_pages",
                tool_name=document_fetch_tools[0],
                document_id=str(first_match.get("document_id", "")),
                path=str(first_match.get("relative_path", "")),
                rationale="Read the benchmark-provided source file before broad corpus search.",
            )
        if source_bundle.urls and discover_tools:
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=discover_tools[0],
                query=source_bundle.task_text,
                rationale="Discover prompt-supplied reference files.",
            )
        if document_search_tools:
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=document_search_tools[0],
                query=seed_query,
                rationale="Search the grounded document source first.",
            )
        if source_bundle.urls and document_fetch_tools:
            return RetrievalAction(
                action="tool",
                stage="locate_pages",
                tool_name=document_fetch_tools[0],
                url=source_bundle.urls[0],
                rationale="Read the first supplied reference document.",
            )
        if external_search_tools and (not officeqa_mode or allow_web_fallback):
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=external_search_tools[0],
                query=seed_query,
                rationale="Search the web for a supporting source.",
            )
        return RetrievalAction(action="answer", stage="answer", rationale="No retrieval tools are available.")

    if _tool_role(registry, last_type) == "search" and last_status in {"empty", "irrelevant"}:
        next_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
        if next_query and next_query != (journal.retrieval_queries[-1] if journal.retrieval_queries else ""):
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=last_type,
                query=next_query,
                rationale="Refine the search because the prior results were not relevant enough.",
            )

    if _tool_role(registry, last_type) in {"search", "discover"} and candidates:
        first = candidates[0]
        best_score = _search_candidate_score(first, retrieval_intent, source_bundle, benchmark_overrides)
        next_query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
        if best_score < 0.55 and next_query and next_query != (journal.retrieval_queries[-1] if journal.retrieval_queries else ""):
            return RetrievalAction(
                action="tool",
                stage="identify_source",
                tool_name=last_type,
                query=next_query,
                rationale="Refine the search because the top result is still weak or off-target.",
            )
        if _tool_family(registry, last_type) == "document_retrieval" and document_fetch_tools:
            return RetrievalAction(
                action="tool",
                stage="locate_pages",
                tool_name=document_fetch_tools[0],
                document_id=first.get("document_id", ""),
                path=first.get("path", "") or first.get("citation", ""),
                rationale="Read the top matching corpus document.",
            )
        if document_fetch_tools:
            return RetrievalAction(
                action="tool",
                stage="locate_pages",
                tool_name=document_fetch_tools[0],
                url=first.get("citation", ""),
                rationale="Read the top matching reference document.",
            )

    if _tool_family(registry, last_type) == "external_retrieval" and candidates and document_fetch_tools:
        return RetrievalAction(
            action="tool",
            stage="locate_pages",
            tool_name=document_fetch_tools[0],
            url=candidates[0].get("citation", ""),
            rationale="Open the top search result for grounded evidence.",
        )

    if _tool_role(registry, last_type) == "fetch":
        if _retrieved_evidence_is_sufficient(source_bundle, last_result, overrides):
            return RetrievalAction(action="answer", stage="answer", rationale="Retrieved document evidence is available for the final answer.")
        if (last_type == "fetch_reference_file" or _tool_family(registry, last_type) == "document_retrieval") and _retrieved_window_is_promising(
            source_bundle,
            retrieval_intent,
            last_result,
            overrides,
        ):
            next_window = _next_reference_fetch_action(last_result, last_type)
            if next_window is not None:
                return next_window
        if last_type == "fetch_corpus_document":
            next_window = _next_corpus_fetch_action(last_result)
            if next_window is not None:
                return next_window
        if corpus_grounded_only:
            return RetrievalAction(action="answer", stage="answer", rationale="No stronger grounded document evidence is available within the allowed retrieval budget.")

    if journal.retrieval_iterations >= MAX_RETRIEVAL_HOPS - 1:
        return RetrievalAction(action="answer", stage="answer", rationale="Retrieval hop budget exhausted.")

    if not corpus_grounded_only and external_search_tools and last_type != external_search_tools[0] and (not officeqa_mode or allow_web_fallback):
        query = _next_retrieval_query(journal, retrieval_intent, source_bundle)
        return RetrievalAction(
            action="tool",
            stage="identify_source",
            tool_name=external_search_tools[0],
            query=query,
            rationale="Broaden search after insufficient local evidence.",
        )

    return RetrievalAction(action="answer", stage="answer", rationale="No better retrieval action is available.")


def _validate_retrieval_action(
    action: RetrievalAction,
    tool_plan: ToolPlan,
    registry: dict[str, dict[str, Any]],
) -> RetrievalAction:
    allowed_tools = set(_retrieval_tools_available(tool_plan, registry))
    if action.action != "tool":
        return action
    if action.tool_name not in allowed_tools:
        return RetrievalAction(action="answer", rationale="Planned retrieval tool is not available in the current tool plan.")
    if _tool_role(registry, action.tool_name) == "search" and not action.query.strip():
        return RetrievalAction(action="answer", rationale="Search action did not produce a usable query.")
    if _tool_role(registry, action.tool_name) == "fetch" and not (action.url.strip() or action.document_id.strip() or action.path.strip()):
        return RetrievalAction(action="answer", rationale="Corpus fetch action did not identify a document.")
    return action


def _plan_retrieval_action(
    *,
    execution_mode: str,
    source_bundle: SourceBundle,
    retrieval_intent: RetrievalIntent,
    tool_plan: ToolPlan,
    journal: ExecutionJournal,
    registry: dict[str, dict[str, Any]],
    benchmark_overrides: dict[str, Any] | None = None,
) -> RetrievalAction:
    available_tools = _retrieval_tools_available(tool_plan, registry)
    if not available_tools or journal.retrieval_iterations >= MAX_RETRIEVAL_HOPS:
        return RetrievalAction(action="answer", rationale="No remaining retrieval capacity.")

    heuristic = _fallback_retrieval_action(
        execution_mode=execution_mode,
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        journal=journal,
        tool_plan=tool_plan,
        registry=registry,
        benchmark_overrides=benchmark_overrides,
    )
    return _validate_retrieval_action(heuristic, tool_plan, registry)


def _tool_args_from_retrieval_action(
    action: RetrievalAction,
    source_bundle: SourceBundle,
    registry: dict[str, dict[str, Any]],
    retrieval_intent: RetrievalIntent,
) -> dict[str, Any]:
    if action.tool_name == "internet_search":
        return {"query": action.query or _derive_retrieval_seed_query(source_bundle, retrieval_intent)}
    if action.tool_name == "search_officeqa_documents":
        return {
            "query": action.query or _derive_retrieval_seed_query(source_bundle, retrieval_intent),
            "top_k": 5,
            "snippet_chars": 700,
            "source_files": source_bundle.source_files_expected[:8],
        }
    if action.tool_name == "search_reference_corpus":
        return {
            "query": action.query or _derive_retrieval_seed_query(source_bundle, retrieval_intent),
            "top_k": 5,
            "snippet_chars": 700,
            "source_files": source_bundle.source_files_expected[:8],
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
        tool_obj = _tool_lookup(registry, action.tool_name)
        arg_schema = {str(key).lower(): key for key in dict(getattr(tool_obj, "args", {}) or {}).keys()}
        hint = action.query or _derive_retrieval_seed_query(source_bundle, retrieval_intent)
        if "search_hint" in arg_schema and hint:
            args[arg_schema["search_hint"]] = hint
        return args
    if action.tool_name == "fetch_corpus_document":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "chunk_start": action.chunk_start,
            "chunk_limit": max(1, action.chunk_limit),
        }
    if action.tool_name == "fetch_officeqa_pages":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "page_start": action.page_start,
            "page_limit": max(1, action.page_limit),
        }
    if action.tool_name == "fetch_officeqa_table":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "table_query": action.query or _officeqa_table_query(retrieval_intent, source_bundle),
            "row_offset": action.row_offset,
            "row_limit": max(50, action.row_limit or 200),
        }
    if action.tool_name == "lookup_officeqa_rows":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "table_query": _officeqa_table_query(retrieval_intent, source_bundle),
            "row_query": action.query or _officeqa_row_query(retrieval_intent, source_bundle),
            "row_offset": action.row_offset,
            "row_limit": max(20, action.row_limit or 120),
        }
    if action.tool_name == "lookup_officeqa_cells":
        return {
            "document_id": action.document_id,
            "path": action.path,
            "table_query": _officeqa_table_query(retrieval_intent, source_bundle),
            "row_query": _officeqa_row_query(retrieval_intent, source_bundle),
            "column_query": action.query or _officeqa_column_query(retrieval_intent),
            "row_offset": action.row_offset,
            "row_limit": max(10, action.row_limit or 60),
        }
    args = _generic_tool_args(registry, action.tool_name, source_bundle, retrieval_intent)
    tool_obj = _tool_lookup(registry, action.tool_name)
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
