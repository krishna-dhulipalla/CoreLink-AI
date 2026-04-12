from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.contracts import ExecutionJournal, RetrievalAction, RetrievalIntent, SourceBundle


@dataclass(frozen=True)
class OfficeQAPlannerOps:
    next_indexed_source_match: Callable[[list[dict[str, Any]], ExecutionJournal], dict[str, Any] | None]
    ranking_confidence: Callable[[list[dict[str, Any]], RetrievalIntent, SourceBundle, dict[str, Any] | None], tuple[float, float]]
    next_retrieval_query: Callable[[ExecutionJournal, RetrievalIntent, SourceBundle], str]
    candidate_pool_missing_preferred_publication_years: Callable[[list[dict[str, Any]], RetrievalIntent], bool]
    candidate_table_query_hint: Callable[[dict[str, Any], RetrievalIntent, SourceBundle], str]
    retrieved_evidence_is_sufficient: Callable[[SourceBundle, dict[str, Any], dict[str, Any]], bool]
    current_table_family: Callable[[dict[str, Any]], str]
    table_family_matches_intent: Callable[[str, RetrievalIntent], bool]
    next_ranked_source_candidate: Callable[[ExecutionJournal, RetrievalIntent, SourceBundle, dict[str, Any] | None], dict[str, Any] | None]
    best_same_document_table_candidate: Callable[[dict[str, Any], RetrievalIntent], tuple[dict[str, Any] | None, float, float]]
    next_table_query: Callable[[ExecutionJournal, RetrievalIntent, SourceBundle], str]
    next_officeqa_page_action: Callable[[dict[str, Any], str], RetrievalAction | None]
    retrieved_window_is_promising: Callable[[SourceBundle, RetrievalIntent, dict[str, Any], dict[str, Any]], bool]
    normalize_query: Callable[[str], str]


@dataclass(frozen=True)
class OfficeQAPlanningContext:
    requested_strategy: str | None
    active_strategy: str
    source_bundle: SourceBundle
    retrieval_intent: RetrievalIntent
    journal: ExecutionJournal
    benchmark_overrides: dict[str, Any] | None
    overrides: dict[str, Any]
    officeqa_status: str
    current_document_id: str
    current_path: str
    seed_query: str
    indexed_source_matches: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    last_type: str
    last_result: dict[str, Any]
    officeqa_search_tools: list[str]
    officeqa_table_tools: list[str]
    officeqa_row_tools: list[str]
    officeqa_cell_tools: list[str]
    officeqa_page_tools: list[str]
    document_search_tools: list[str]
    external_search_tools: list[str]
    allow_web_fallback: bool
    prefer_text_first: bool
    next_table_query: str
    officeqa_table_query: str
    officeqa_row_query: str
    officeqa_column_query: str


def _officeqa_action(
    *,
    requested_strategy: str | None,
    active_strategy: str,
    action: str = "tool",
    stage: str = "",
    tool_name: str = "",
    query: str = "",
    document_id: str = "",
    path: str = "",
    url: str = "",
    evidence_gap: str = "",
    rationale: str = "",
) -> RetrievalAction:
    return RetrievalAction(
        action=action,
        stage=stage,
        requested_strategy=requested_strategy or active_strategy,
        strategy=active_strategy,
        tool_name=tool_name,
        query=query,
        document_id=document_id,
        path=path,
        url=url,
        evidence_gap=evidence_gap,
        rationale=rationale,
    )


def plan_initial_action(context: OfficeQAPlanningContext, ops: OfficeQAPlannerOps) -> RetrievalAction:
    next_source_match = ops.next_indexed_source_match(context.indexed_source_matches, ExecutionJournal())
    if context.officeqa_search_tools and context.retrieval_intent.source_constraint_policy != "hard" and len(context.indexed_source_matches) != 1:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="identify_source",
            tool_name=context.officeqa_search_tools[0],
            query=context.seed_query,
            rationale="Search the indexed OfficeQA corpus first, treating benchmark-linked source files as a soft prior rather than a hard fence.",
        )
    if next_source_match:
        if context.prefer_text_first and context.officeqa_page_tools:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_pages",
                tool_name=context.officeqa_page_tools[0],
                document_id=str(next_source_match.get("document_id", "")),
                path=str(next_source_match.get("relative_path", "")),
                rationale="Start from the benchmark-linked source file and inspect pages first for grounded context.",
            )
        if context.officeqa_table_tools:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=str(next_source_match.get("document_id", "")),
                path=str(next_source_match.get("relative_path", "")),
                query=context.next_table_query,
                rationale="Start from the benchmark-linked source file and extract the strongest table candidate first.",
            )
        if context.officeqa_page_tools:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_pages",
                tool_name=context.officeqa_page_tools[0],
                document_id=str(next_source_match.get("document_id", "")),
                path=str(next_source_match.get("relative_path", "")),
                rationale="Start from the benchmark-linked source file and read the relevant pages.",
            )
    if context.officeqa_search_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="identify_source",
            tool_name=context.officeqa_search_tools[0],
            query=context.seed_query,
            rationale="Identify the best OfficeQA source document before extraction.",
        )
    if context.document_search_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="identify_source",
            tool_name=context.document_search_tools[0],
            query=context.seed_query,
            rationale="Search the packaged OfficeQA corpus for a grounded source document.",
        )
    if context.source_bundle.urls and context.officeqa_page_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_pages",
            tool_name=context.officeqa_page_tools[0],
            url=context.source_bundle.urls[0],
            rationale="Read the first supplied reference document.",
        )
    if context.external_search_tools and context.allow_web_fallback:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="identify_source",
            tool_name=context.external_search_tools[0],
            query=context.seed_query,
            rationale="Use web search only as an explicit OfficeQA fallback.",
        )
    return RetrievalAction(action="answer", stage="answer", rationale="No OfficeQA retrieval tools are available.")


def plan_search_followup(context: OfficeQAPlanningContext, ops: OfficeQAPlannerOps) -> RetrievalAction | None:
    if not context.candidates:
        return None
    first = context.candidates[0]
    best_score, score_gap = ops.ranking_confidence(context.candidates, context.retrieval_intent, context.source_bundle, context.benchmark_overrides)
    next_query = ops.next_retrieval_query(context.journal, context.retrieval_intent, context.source_bundle)
    localized_pool = ops.candidate_pool_missing_preferred_publication_years(context.candidates, context.retrieval_intent)
    if localized_pool and next_query and next_query != (context.journal.retrieval_queries[-1] if context.journal.retrieval_queries else ""):
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="identify_source",
            tool_name=context.last_type,
            query=next_query,
            evidence_gap="source pool too narrow",
            rationale="Refine OfficeQA source search because the current candidate pool stays trapped in the target-year publication slice.",
        )
    if (best_score < 1.05 or (best_score < 1.45 and score_gap < 0.2)) and next_query and next_query != (context.journal.retrieval_queries[-1] if context.journal.retrieval_queries else ""):
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="identify_source",
            tool_name=context.last_type,
            query=next_query,
            evidence_gap="wrong document",
            rationale="Refine OfficeQA source search because the top candidate confidence is still too weak.",
        )
    if context.prefer_text_first and context.officeqa_page_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_pages",
            tool_name=context.officeqa_page_tools[0],
            document_id=first.get("document_id", ""),
            path=first.get("path", "") or first.get("citation", ""),
            evidence_gap="source pool too narrow" if localized_pool else "",
            rationale="Open the best matching OfficeQA document and inspect pages first for context.",
        )
    if context.officeqa_table_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_table",
            tool_name=context.officeqa_table_tools[0],
            document_id=first.get("document_id", ""),
            path=first.get("path", "") or first.get("citation", ""),
            query=ops.candidate_table_query_hint(first, context.retrieval_intent, context.source_bundle) or context.next_table_query or context.officeqa_table_query,
            evidence_gap="source pool too narrow" if localized_pool else "",
            rationale="Open the best matching OfficeQA document and extract the relevant table first.",
        )
    if context.officeqa_page_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_pages",
            tool_name=context.officeqa_page_tools[0],
            document_id=first.get("document_id", ""),
            path=first.get("path", "") or first.get("citation", ""),
            evidence_gap="source pool too narrow" if localized_pool else "",
            rationale="Open the best matching OfficeQA document and inspect the relevant pages.",
        )
    return None


def plan_table_followup(context: OfficeQAPlanningContext, ops: OfficeQAPlannerOps) -> RetrievalAction | None:
    last_facts = dict(context.last_result.get("facts") or {})
    if ops.retrieved_evidence_is_sufficient(context.source_bundle, context.last_result, context.overrides):
        return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA table evidence is sufficient for the final answer.")
    current_table_query = ops.normalize_query(str((context.last_result.get("assumptions") or {}).get("table_query", "") or ""))
    current_table_family = ops.current_table_family(context.last_result)
    wrong_table_family = bool(current_table_family) and not ops.table_family_matches_intent(current_table_family, context.retrieval_intent)
    next_ranked_candidate = ops.next_ranked_source_candidate(context.journal, context.retrieval_intent, context.source_bundle, context.benchmark_overrides)
    alternate_table_candidate, current_table_score, alternate_table_score = ops.best_same_document_table_candidate(context.last_result, context.retrieval_intent)
    if alternate_table_candidate and context.officeqa_table_tools:
        alternate_query = ops.normalize_query(
            " ".join(
                part
                for part in (
                    context.retrieval_intent.entity,
                    context.retrieval_intent.metric,
                    context.retrieval_intent.period,
                    str(alternate_table_candidate.get("locator", "") or ""),
                    " ".join(str(item or "") for item in list(alternate_table_candidate.get("headers", []))[:4]),
                    " ".join(str(item or "") for item in list(alternate_table_candidate.get("row_labels", []))[:4]),
                )
                if part
            )
        )
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_table",
            tool_name=context.officeqa_table_tools[0],
            document_id=context.current_document_id,
            path=context.current_path,
            query=alternate_query or str(alternate_table_candidate.get("locator", "") or "") or context.next_table_query or context.officeqa_table_query,
            evidence_gap="wrong row or column semantics",
            rationale=(
                "Retry table extraction inside the same document because another indexed table candidate has materially "
                f"stronger semantic alignment ({alternate_table_score:.2f} vs {current_table_score:.2f})."
            ),
        )
    if wrong_table_family and context.officeqa_table_tools:
        alternate_query = ops.next_table_query(context.journal, context.retrieval_intent, context.source_bundle)
        if alternate_query and alternate_query.lower() != current_table_query.lower():
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=context.current_document_id,
                path=context.current_path,
                query=alternate_query,
                evidence_gap="wrong table family",
                rationale="Retry table extraction with a more specific query because the selected table family does not match the question.",
            )
        if next_ranked_candidate and context.officeqa_table_tools:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=str(next_ranked_candidate.get("document_id", "")),
                path=str(next_ranked_candidate.get("path", "") or next_ranked_candidate.get("citation", "")),
                query=ops.candidate_table_query_hint(next_ranked_candidate, context.retrieval_intent, context.source_bundle) or context.next_table_query or context.officeqa_table_query,
                evidence_gap="wrong document",
                rationale="Reopen retrieval on the next ranked source because the current document keeps yielding the wrong table family.",
            )
    if context.retrieval_intent.strategy in {"multi_table", "multi_document"} and context.officeqa_table_tools:
        alternate_query = ops.next_table_query(context.journal, context.retrieval_intent, context.source_bundle)
        if alternate_query and alternate_query.lower() != current_table_query.lower():
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=context.current_document_id,
                path=context.current_path,
                query=alternate_query,
                evidence_gap="join-ready evidence",
                rationale="Try an alternate table candidate because the current table is not sufficient yet.",
            )
    if context.officeqa_status == "ok" and context.officeqa_row_tools and last_facts.get("tables"):
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="extract_rows",
            tool_name=context.officeqa_row_tools[0],
            document_id=context.current_document_id,
            path=context.current_path,
            query=context.officeqa_row_query,
            rationale="Narrow the OfficeQA table down to the target rows before computing.",
        )
    if context.officeqa_page_tools and (
        context.active_strategy in {"text_first", "hybrid", "multi_table", "multi_document"}
        or context.officeqa_status in {"missing_table", "partial_table", "unit_ambiguity"}
    ):
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_pages",
            tool_name=context.officeqa_page_tools[0],
            document_id=context.current_document_id,
            path=context.current_path,
            evidence_gap="narrative support",
            rationale="Fallback to page inspection because table extraction alone was incomplete or ambiguous.",
        )
    next_source_match = ops.next_indexed_source_match(list(context.source_bundle.source_files_found), context.journal)
    if context.active_strategy == "multi_document" and next_source_match:
        if context.prefer_text_first and context.officeqa_page_tools:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_pages",
                tool_name=context.officeqa_page_tools[0],
                document_id=str(next_source_match.get("document_id", "")),
                path=str(next_source_match.get("relative_path", "")),
                evidence_gap="cross-document alignment",
                rationale="Move to the next benchmark-linked source document to complete multi-document evidence.",
            )
        if context.officeqa_table_tools:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=str(next_source_match.get("document_id", "")),
                path=str(next_source_match.get("relative_path", "")),
                query=context.next_table_query or context.officeqa_table_query,
                evidence_gap="cross-document alignment",
                rationale="Move to the next benchmark-linked source document to complete multi-document evidence.",
            )
    return None


def plan_row_followup(context: OfficeQAPlanningContext, ops: OfficeQAPlannerOps) -> RetrievalAction | None:
    last_facts = dict(context.last_result.get("facts") or {})
    if ops.retrieved_evidence_is_sufficient(context.source_bundle, context.last_result, context.overrides):
        return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA row evidence is sufficient for the final answer.")
    current_table_family = ops.current_table_family(context.last_result)
    next_ranked_candidate = ops.next_ranked_source_candidate(context.journal, context.retrieval_intent, context.source_bundle, context.benchmark_overrides)
    if context.officeqa_status == "missing_row" and next_ranked_candidate and context.officeqa_table_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_table",
            tool_name=context.officeqa_table_tools[0],
            document_id=str(next_ranked_candidate.get("document_id", "")),
            path=str(next_ranked_candidate.get("path", "") or next_ranked_candidate.get("citation", "")),
            query=ops.candidate_table_query_hint(next_ranked_candidate, context.retrieval_intent, context.source_bundle) or context.next_table_query or context.officeqa_table_query,
            evidence_gap="wrong document",
            rationale="Reopen source search on the next ranked candidate because the current document did not contain the requested row.",
        )
    if current_table_family and not ops.table_family_matches_intent(current_table_family, context.retrieval_intent) and next_ranked_candidate and context.officeqa_table_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_table",
            tool_name=context.officeqa_table_tools[0],
            document_id=str(next_ranked_candidate.get("document_id", "")),
            path=str(next_ranked_candidate.get("path", "") or next_ranked_candidate.get("citation", "")),
            query=ops.candidate_table_query_hint(next_ranked_candidate, context.retrieval_intent, context.source_bundle) or context.next_table_query or context.officeqa_table_query,
            evidence_gap="wrong table family",
            rationale="Switch to the next ranked source because the current document produced a mismatched table family.",
        )
    if context.officeqa_cell_tools and context.officeqa_status == "ok" and last_facts.get("tables"):
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="extract_cells",
            tool_name=context.officeqa_cell_tools[0],
            document_id=context.current_document_id,
            path=context.current_path,
            query=context.officeqa_column_query,
            rationale="Narrow the OfficeQA rows down to the target cells before computing.",
        )
    if context.retrieval_intent.strategy in {"multi_table", "multi_document"} and context.officeqa_table_tools:
        alternate_query = ops.next_table_query(context.journal, context.retrieval_intent, context.source_bundle)
        current_table_query = ops.normalize_query(str((context.last_result.get("assumptions") or {}).get("table_query", "") or ""))
        if alternate_query and alternate_query.lower() != current_table_query.lower():
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=context.current_document_id,
                path=context.current_path,
                query=alternate_query,
                evidence_gap="join-ready evidence",
                rationale="Try a second table candidate because the current row extraction is incomplete.",
            )
    if context.officeqa_page_tools and (
        context.active_strategy in {"text_first", "hybrid", "multi_table", "multi_document"}
        or context.officeqa_status in {"missing_row", "partial_table", "unit_ambiguity"}
    ):
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_pages",
            tool_name=context.officeqa_page_tools[0],
            document_id=context.current_document_id,
            path=context.current_path,
            evidence_gap="narrative support",
            rationale="Fallback to page inspection because row extraction was incomplete.",
        )
    return None


def plan_cell_followup(context: OfficeQAPlanningContext, ops: OfficeQAPlannerOps) -> RetrievalAction | None:
    if ops.retrieved_evidence_is_sufficient(context.source_bundle, context.last_result, context.overrides):
        return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA cell evidence is sufficient for the final answer.")
    current_table_family = ops.current_table_family(context.last_result)
    next_ranked_candidate = ops.next_ranked_source_candidate(context.journal, context.retrieval_intent, context.source_bundle, context.benchmark_overrides)
    if current_table_family and not ops.table_family_matches_intent(current_table_family, context.retrieval_intent) and next_ranked_candidate and context.officeqa_table_tools:
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_table",
            tool_name=context.officeqa_table_tools[0],
            document_id=str(next_ranked_candidate.get("document_id", "")),
            path=str(next_ranked_candidate.get("path", "") or next_ranked_candidate.get("citation", "")),
            query=ops.candidate_table_query_hint(next_ranked_candidate, context.retrieval_intent, context.source_bundle) or context.next_table_query or context.officeqa_table_query,
            evidence_gap="wrong table family",
            rationale="Switch to the next ranked source because the current document still does not expose the required table family.",
        )
    if context.retrieval_intent.strategy in {"multi_table", "multi_document"} and context.officeqa_table_tools:
        alternate_query = ops.next_table_query(context.journal, context.retrieval_intent, context.source_bundle)
        current_table_query = ops.normalize_query(str((context.last_result.get("assumptions") or {}).get("table_query", "") or ""))
        if alternate_query and alternate_query.lower() != current_table_query.lower():
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=context.current_document_id,
                path=context.current_path,
                query=alternate_query,
                evidence_gap="join-ready evidence",
                rationale="Try another table candidate because the current cell extraction remains ambiguous.",
            )
    if context.officeqa_page_tools and (
        context.active_strategy in {"text_first", "hybrid", "multi_table", "multi_document"}
        or context.officeqa_status in {"partial_table", "unit_ambiguity"}
    ):
        return _officeqa_action(
            requested_strategy=context.requested_strategy,
            active_strategy=context.active_strategy,
            stage="locate_pages",
            tool_name=context.officeqa_page_tools[0],
            document_id=context.current_document_id,
            path=context.current_path,
            evidence_gap="narrative support",
            rationale="Inspect the OfficeQA pages directly because cell extraction is still ambiguous.",
        )
    return None


def plan_pages_followup(context: OfficeQAPlanningContext, ops: OfficeQAPlannerOps) -> RetrievalAction | None:
    last_facts = dict(context.last_result.get("facts") or {})
    if ops.retrieved_evidence_is_sufficient(context.source_bundle, context.last_result, context.overrides):
        return RetrievalAction(action="answer", stage="answer", rationale="OfficeQA page evidence is sufficient for the final answer.")
    if context.active_strategy in {"hybrid", "multi_table", "multi_document"} and context.officeqa_table_tools:
        current_page_path = context.current_path or str(last_facts.get("citation", "") or "")
        alternate_query = ops.next_table_query(context.journal, context.retrieval_intent, context.source_bundle)
        if context.current_document_id or current_page_path:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=context.current_document_id,
                path=current_page_path,
                query=alternate_query or context.officeqa_table_query,
                evidence_gap="table support",
                rationale="Use the page findings to pivot back into table extraction for structured values.",
            )
    next_pages = ops.next_officeqa_page_action(context.last_result, "fetch_officeqa_pages")
    if next_pages is not None and ops.retrieved_window_is_promising(context.source_bundle, context.retrieval_intent, context.last_result, context.overrides):
        next_pages.requested_strategy = next_pages.requested_strategy or context.requested_strategy or context.active_strategy
        next_pages.strategy = context.active_strategy
        return next_pages
    next_source_match = ops.next_indexed_source_match(list(context.source_bundle.source_files_found), context.journal)
    if context.active_strategy == "multi_document" and next_source_match:
        if context.prefer_text_first and context.officeqa_page_tools:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_pages",
                tool_name=context.officeqa_page_tools[0],
                document_id=str(next_source_match.get("document_id", "")),
                path=str(next_source_match.get("relative_path", "")),
                evidence_gap="cross-document alignment",
                rationale="Move to the next benchmark-linked source document after page-level evidence remained incomplete.",
            )
        if context.officeqa_table_tools:
            return _officeqa_action(
                requested_strategy=context.requested_strategy,
                active_strategy=context.active_strategy,
                stage="locate_table",
                tool_name=context.officeqa_table_tools[0],
                document_id=str(next_source_match.get("document_id", "")),
                path=str(next_source_match.get("relative_path", "")),
                query=context.next_table_query or context.officeqa_table_query,
                evidence_gap="cross-document alignment",
                rationale="Move to the next benchmark-linked source document after page-level evidence remained incomplete.",
            )
    return None
