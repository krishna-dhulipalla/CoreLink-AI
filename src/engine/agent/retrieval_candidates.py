from __future__ import annotations

import re
from typing import Any

from engine.agent.contracts import RetrievalCandidate, RetrievalIntent, SourceBundle

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


def retrieval_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 1 and token not in _RETRIEVAL_STOP_WORDS
    ]


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate.get("document_id", "") or "").strip().lower(),
        str(candidate.get("citation", "") or "").strip().lower(),
        str(candidate.get("path", "") or "").strip().lower(),
    )


def _merge_candidate_records(primary: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key in ("document_id", "citation", "path", "title", "snippet"):
        if not merged.get(key) and incoming.get(key):
            merged[key] = incoming.get(key)
    if incoming.get("title"):
        merged_title = str(merged.get("title", "") or "")
        merged_document_id = str(merged.get("document_id", "") or "")
        incoming_title = str(incoming.get("title", "") or "")
        if merged_title == merged_document_id and incoming_title != merged_document_id:
            merged["title"] = incoming_title
    primary_rank = int(merged.get("rank", 999) or 999)
    incoming_rank = int(incoming.get("rank", 999) or 999)
    merged["rank"] = min(primary_rank, incoming_rank)
    merged["score"] = max(float(merged.get("score", 0.0) or 0.0), float(incoming.get("score", 0.0) or 0.0))
    merged["metadata"] = {
        **dict(merged.get("metadata", {}) or {}),
        **dict(incoming.get("metadata", {}) or {}),
    }
    return RetrievalCandidate.model_validate(merged).model_dump()


def dedupe_search_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    index_by_identity: dict[tuple[str, str, str], int] = {}
    index_by_document_id: dict[str, int] = {}
    index_by_citation: dict[str, int] = {}

    for candidate in candidates:
        normalized = RetrievalCandidate.model_validate(candidate).model_dump()
        identity = _candidate_identity(normalized)
        document_id = identity[0]
        citation = identity[1]
        existing_index = index_by_identity.get(identity)
        if existing_index is None and document_id:
            existing_index = index_by_document_id.get(document_id)
        if existing_index is None and citation:
            existing_index = index_by_citation.get(citation)
        if existing_index is None:
            deduped.append(dict(normalized))
            current_index = len(deduped) - 1
        else:
            deduped[existing_index] = _merge_candidate_records(deduped[existing_index], normalized)
            current_index = existing_index
        merged = deduped[current_index]
        merged_identity = _candidate_identity(merged)
        index_by_identity[merged_identity] = current_index
        if merged_identity[0]:
            index_by_document_id[merged_identity[0]] = current_index
        if merged_identity[1]:
            index_by_citation[merged_identity[1]] = current_index
    return deduped


def search_result_candidates(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    facts = dict(tool_result.get("facts") or {})
    candidates: list[dict[str, Any]] = []
    for item in facts.get("documents", []):
        if not isinstance(item, dict):
            continue
        candidates.append(
            RetrievalCandidate(
                document_id=str(item.get("document_id", "")),
                citation=str(item.get("citation", "") or item.get("url", "") or item.get("path", "")),
                path=str(item.get("path", "")),
                title=str(item.get("title", "") or item.get("document_id", "")),
                snippet=str(item.get("snippet", "")),
                rank=int(item.get("rank", 999) or 999),
                score=float(item.get("score", 0.0) or 0.0),
                metadata=dict(item.get("metadata", {}) or {}),
            ).model_dump()
        )
    for item in facts.get("results", []):
        if not isinstance(item, dict):
            continue
        candidates.append(
            RetrievalCandidate(
                document_id=str(item.get("document_id", "")),
                citation=str(item.get("url", "") or item.get("citation", "")),
                path=str(item.get("path", "")),
                title=str(item.get("title", "")),
                snippet=str(item.get("snippet", "")),
                rank=int(item.get("rank", 999) or 999),
                score=float(item.get("score", 0.0) or 0.0),
                metadata=dict(item.get("metadata", {}) or {}),
            ).model_dump()
        )
    return dedupe_search_candidates(
        [
            candidate
            for candidate in candidates
            if candidate.get("citation") or candidate.get("document_id") or candidate.get("path")
        ]
    )


def search_candidate_text(candidate: dict[str, Any]) -> str:
    normalized = RetrievalCandidate.model_validate(candidate)
    return " ".join(
        str(getattr(normalized, key, "") or "")
        for key in ("title", "snippet", "citation", "path", "document_id")
    ).strip()


def candidate_best_evidence_unit(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(RetrievalCandidate.model_validate(candidate).metadata or {})
    return dict(metadata.get("best_evidence_unit", {}) or {})


def candidate_metadata_text(candidate: dict[str, Any]) -> str:
    normalized = RetrievalCandidate.model_validate(candidate)
    metadata = dict(normalized.metadata or {})
    best_unit = dict(metadata.get("best_evidence_unit", {}) or {})
    return " ".join(
        [
            " ".join(str(item or "") for item in list(metadata.get("years", []))),
            str(metadata.get("publication_year", "") or ""),
            str(metadata.get("publication_month", "") or ""),
            " ".join(str(item or "") for item in list(metadata.get("page_markers", []))),
            " ".join(str(item or "") for item in list(metadata.get("section_titles", []))),
            " ".join(str(item or "") for item in list(metadata.get("table_headers", []))),
            " ".join(str(item or "") for item in list(metadata.get("row_labels", []))),
            " ".join(str(item or "") for item in list(metadata.get("unit_hints", []))),
            " ".join(str(item or "") for item in list(metadata.get("month_coverage", []))),
            " ".join(str(item or "") for item in list(metadata.get("period_types", []))),
            str(best_unit.get("locator", "") or ""),
            str(best_unit.get("table_family", "") or ""),
            str(best_unit.get("period_type", "") or ""),
            " ".join(str(item or "") for item in list(best_unit.get("headers", []))),
            " ".join(str(item or "") for item in list(best_unit.get("row_labels", []))),
            " ".join(str(item or "") for item in list(best_unit.get("year_refs", []))),
            " ".join(str(item or "") for item in list(best_unit.get("month_coverage", []))),
            str(best_unit.get("preview_text", "") or ""),
        ]
    ).strip()


def _query_years(retrieval_intent: RetrievalIntent) -> set[str]:
    return {token for token in re.findall(r"\b((?:19|20)\d{2})\b", retrieval_intent.period or "")}


def _candidate_publication_years(candidate: dict[str, Any]) -> set[str]:
    metadata = dict(RetrievalCandidate.model_validate(candidate).metadata or {})
    publication_year = str(metadata.get("publication_year", "") or "").strip()
    if re.fullmatch(r"(?:19|20)\d{2}", publication_year):
        return {publication_year}
    normalized = RetrievalCandidate.model_validate(candidate)
    return set(
        re.findall(
            r"\b((?:19|20)\d{2})\b",
            " ".join(str(getattr(normalized, key, "") or "") for key in ("title", "citation", "path", "document_id")),
        )
    )


def _query_entity_tokens(retrieval_intent: RetrievalIntent) -> set[str]:
    return {token for token in retrieval_tokens(retrieval_intent.entity) if token not in {"u", "s", "us"}}


def _query_metric_tokens(retrieval_intent: RetrievalIntent) -> set[str]:
    metric_basis = retrieval_intent.metric
    if retrieval_intent.aggregation_shape in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"} or retrieval_intent.metric in {"absolute percent change", "absolute difference"}:
        metric_basis = "expenditures"
    return set(retrieval_tokens(metric_basis))


def _best_unit_text(best_unit: dict[str, Any]) -> str:
    return " ".join(
        [
            str(best_unit.get("locator", "") or ""),
            str(best_unit.get("context_text", "") or ""),
            " ".join(str(item or "") for item in list(best_unit.get("heading_chain", []))),
            " ".join(str(item or "") for item in list(best_unit.get("headers", []))),
            " ".join(str(item or "") for item in list(best_unit.get("row_labels", []))),
            " ".join(str(item or "") for item in list(best_unit.get("row_paths", []))),
            " ".join(str(item or "") for item in list(best_unit.get("column_paths", []))),
        ]
    ).strip()


def _best_surface_match(text: str, phrase: str) -> float:
    normalized_text = " ".join(retrieval_tokens(text))
    normalized_phrase = " ".join(retrieval_tokens(phrase))
    if not normalized_text or not normalized_phrase:
        return 0.0
    if normalized_phrase in normalized_text:
        return 1.0
    text_tokens = set(normalized_text.split())
    phrase_tokens = set(normalized_phrase.split())
    if not phrase_tokens:
        return 0.0
    return float(len(text_tokens & phrase_tokens)) / max(1, len(phrase_tokens))


def _family_terms(retrieval_intent: RetrievalIntent) -> tuple[set[str], set[str]]:
    metric_tokens = _query_metric_tokens(retrieval_intent)
    debt_terms = {"debt", "outstanding", "obligations", "liabilities", "securities", "guaranteed"}
    flow_terms = {"expenditures", "receipts", "revenue", "revenues", "collections", "outlays", "spending"}
    return metric_tokens & debt_terms, metric_tokens & flow_terms


def table_family_matches_intent(table_family: str, retrieval_intent: RetrievalIntent) -> bool:
    family = (table_family or "").strip().lower()
    if not family:
        return True
    if family == "navigation_or_contents":
        return False
    granularity = retrieval_intent.granularity_requirement
    if granularity == "monthly_series":
        return family == "monthly_series"
    if granularity == "fiscal_year":
        return family in {"fiscal_year_comparison", "category_breakdown"}
    return family in {"category_breakdown", "annual_summary", "fiscal_year_comparison", "debt_or_balance_sheet", "generic_financial_table"}


def _table_family_preference_score(table_family: str, retrieval_intent: RetrievalIntent) -> float:
    family = str(table_family or "").strip().lower()
    if not family:
        return 0.0
    if family == "navigation_or_contents":
        return -1.5
    debt_terms, flow_terms = _family_terms(retrieval_intent)
    granularity = retrieval_intent.granularity_requirement
    if granularity == "monthly_series":
        if family == "monthly_series":
            return 0.95
        if family in {"annual_summary", "debt_or_balance_sheet"}:
            return -0.45
        return -0.12
    if granularity == "fiscal_year":
        if family == "fiscal_year_comparison":
            return 0.72
        if family == "category_breakdown":
            return 0.18
    if debt_terms:
        if family == "debt_or_balance_sheet":
            return 0.72
        if family in {"category_breakdown", "annual_summary"}:
            return -0.22
    if flow_terms:
        if family == "category_breakdown":
            return 0.82
        if family == "annual_summary":
            return 0.28
        if family == "debt_or_balance_sheet":
            return -0.4
    if family == "generic_financial_table":
        return 0.08
    return 0.0


def _best_unit_alignment_score(best_unit: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    if not best_unit:
        return 0.0
    unit_text = _best_unit_text(best_unit)
    score = 0.0
    entity = retrieval_intent.entity or ""
    metric = retrieval_intent.metric or ""
    period = retrieval_intent.period or ""
    if entity:
        score += 0.32 * _best_surface_match(unit_text, entity)
    if metric:
        score += 0.26 * _best_surface_match(unit_text, metric)
    if period:
        score += 0.18 * _best_surface_match(unit_text, period)
    score += _table_family_preference_score(str(best_unit.get("table_family", "") or ""), retrieval_intent)
    return score


def retrieval_focus_tokens(source_bundle: SourceBundle) -> list[str]:
    text = " ".join(
        [
            source_bundle.focus_query,
            source_bundle.target_period,
            " ".join(source_bundle.entities[:4]),
            " ".join(source_bundle.inline_facts.keys()),
        ]
    )
    return retrieval_tokens(text)


def _granularity_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    metadata = dict(RetrievalCandidate.model_validate(candidate).metadata or {})
    text = f"{search_candidate_text(candidate)} {candidate_metadata_text(candidate)}".lower()
    best_unit = candidate_best_evidence_unit(candidate)
    granularity = retrieval_intent.granularity_requirement
    if granularity == "monthly_series":
        month_coverage = list(metadata.get("month_coverage", []))
        if str(best_unit.get("period_type", "") or "").lower() == "monthly_series":
            return 1.0
        if len(month_coverage) >= 6:
            return 0.9
        if any(token in text for token in ("monthly", "month", "receipts expenditures and balances", "january", "february", "march")):
            return 0.55
        if any(token in text for token in ("total 9/", "actual 6 months", "summary", "calendar year")):
            return -0.35
        return -0.15
    if granularity == "fiscal_year":
        if str(best_unit.get("period_type", "") or "").lower() == "fiscal_year":
            return 0.9
        if any(token in text for token in ("fiscal year", "fy ", "end of fiscal years")):
            return 0.65
        return -0.1
    if granularity == "calendar_year":
        if str(best_unit.get("period_type", "") or "").lower() == "calendar_year":
            return 0.55
        if any(token in text for token in ("calendar year", "annual", "summary", "actual 6 months", "estimate")):
            return 0.18
        return 0.0
    if granularity == "narrative_support":
        if any(token in text for token in ("discussion", "narrative", "commentary", "statement")):
            return 0.4
    return 0.0


def _category_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    metadata_text = candidate_metadata_text(candidate).lower()
    best_unit = candidate_best_evidence_unit(candidate)
    entity_tokens = _query_entity_tokens(retrieval_intent)
    metric_tokens = _query_metric_tokens(retrieval_intent)
    score = 0.0
    score += 0.18 * len(entity_tokens & set(retrieval_tokens(metadata_text)))
    score += 0.12 * len(metric_tokens & set(retrieval_tokens(metadata_text)))
    if entity_tokens and set(retrieval_tokens(" ".join(str(item or "") for item in list(best_unit.get("row_labels", []))))) & entity_tokens:
        score += 0.18
    if metric_tokens and set(retrieval_tokens(" ".join(str(item or "") for item in list(best_unit.get("headers", []))))) & metric_tokens:
        score += 0.14
    return score


def _year_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    metadata = dict(RetrievalCandidate.model_validate(candidate).metadata or {})
    best_unit = candidate_best_evidence_unit(candidate)
    candidate_years = {str(item) for item in list(metadata.get("years", [])) if str(item)}
    candidate_years.update(str(item) for item in list(best_unit.get("year_refs", [])) if str(item))
    required_years = set(retrieval_intent.target_years) or _query_years(retrieval_intent)
    publication_years = _candidate_publication_years(candidate)
    if not required_years:
        return 0.0
    preferred_publication_years = list(retrieval_intent.preferred_publication_years)
    publication_window = set(retrieval_intent.publication_year_window)
    explicit_scope = bool(retrieval_intent.publication_scope_explicit)
    score = 0.0
    for publication_year in publication_years:
        if publication_year in preferred_publication_years:
            position = preferred_publication_years.index(publication_year)
            if explicit_scope:
                score = max(score, max(0.2, 1.1 - (0.18 * position)))
            else:
                score = max(score, max(0.08, 0.34 - (0.04 * position)))
        elif publication_year in publication_window:
            score = max(score, 0.25 if explicit_scope else 0.06)
        elif publication_window:
            if retrieval_intent.retrospective_evidence_required:
                score -= 0.04
            elif retrieval_intent.retrospective_evidence_allowed:
                score -= 0.08
            else:
                score -= 0.22
    if candidate_years and required_years & candidate_years:
        score += 0.95
    if candidate_years and not (required_years & candidate_years):
        score -= 0.28
    if candidate_years and required_years & candidate_years and publication_years:
        acceptable_lag = max(0, int(retrieval_intent.acceptable_publication_lag_years or 0))
        required_year_ints = [int(year) for year in required_years if year.isdigit()]
        publication_year_ints = [int(year) for year in publication_years if year.isdigit()]
        if required_year_ints and publication_year_ints:
            target_max = max(required_year_ints)
            publication_min = min(publication_year_ints)
            if retrieval_intent.retrospective_evidence_required and publication_min >= target_max:
                lag = publication_min - target_max
                if lag <= max(acceptable_lag, 5):
                    score += max(0.18, 0.42 - (0.04 * lag))
            elif retrieval_intent.retrospective_evidence_allowed and publication_min >= target_max:
                lag = publication_min - target_max
                if lag <= max(acceptable_lag, 1):
                    score += max(0.08, 0.24 - (0.05 * lag))
    text = f"{search_candidate_text(candidate)} {candidate_metadata_text(candidate)}"
    text_years = set(re.findall(r"\b((?:19|20)\d{2})\b", text))
    if required_years & text_years:
        score += 0.35
    if text_years and not (required_years & text_years):
        score -= 0.12
    return score


def _exclusion_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    text = f"{search_candidate_text(candidate)} {candidate_metadata_text(candidate)}".lower()
    score = 0.0
    for term in retrieval_intent.exclude_constraints:
        tokens = retrieval_tokens(term)
        if tokens and set(tokens).intersection(retrieval_tokens(text)):
            score -= 0.45
    return score


def _historical_family_fit_score(candidate: dict[str, Any], retrieval_intent: RetrievalIntent) -> float:
    text = f"{search_candidate_text(candidate)} {candidate_metadata_text(candidate)}".lower()
    required_years = _query_years(retrieval_intent)
    if required_years and min(int(year) for year in required_years) <= 1945:
        if any(token in text for token in ("fiscal year", "end of fiscal years", "comparative", "statement", "veterans administration", "national defense")):
            return 0.35
    return 0.0


def search_candidate_score(
    candidate: dict[str, Any],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> float:
    text = search_candidate_text(candidate).lower()
    metadata_text = candidate_metadata_text(candidate).lower()
    best_unit = candidate_best_evidence_unit(candidate)
    combined_text = f"{text} {metadata_text}".strip()
    tokens = set(retrieval_tokens(combined_text))
    title_tokens = set(retrieval_tokens(str(RetrievalCandidate.model_validate(candidate).title)))
    score = 0.0
    rank = int(RetrievalCandidate.model_validate(candidate).rank or 999)
    score += max(0.0, 0.4 - 0.05 * max(0, rank - 1))
    score += min(1.0, float(RetrievalCandidate.model_validate(candidate).score or 0.0) * 0.18)

    entity_tokens = _query_entity_tokens(retrieval_intent)
    metric_tokens = _query_metric_tokens(retrieval_intent)
    period_tokens = {token for token in retrieval_tokens(retrieval_intent.period)}
    must_tokens = {token for term in retrieval_intent.must_include_terms for token in retrieval_tokens(term)}
    query_tokens = set(retrieval_focus_tokens(source_bundle))

    overlap = len((entity_tokens | metric_tokens | period_tokens | must_tokens | query_tokens) & tokens)
    score += 0.12 * overlap
    score += 0.12 * len(entity_tokens & title_tokens)
    score += 0.08 * len(metric_tokens & title_tokens)
    score += 0.06 * len(period_tokens & title_tokens)
    score += _year_fit_score(candidate, retrieval_intent)
    score += _granularity_fit_score(candidate, retrieval_intent)
    score += _category_fit_score(candidate, retrieval_intent)
    score += _exclusion_fit_score(candidate, retrieval_intent)
    score += _historical_family_fit_score(candidate, retrieval_intent)
    score += 0.12 * float(best_unit.get("table_confidence", 0.0) or 0.0)
    score += _best_unit_alignment_score(best_unit, retrieval_intent)

    citation = str(RetrievalCandidate.model_validate(candidate).citation).lower()
    if any(host in citation for host in ("govinfo.gov", "census.gov", "va.gov", "fraser.stlouisfed.org", ".gov/")):
        score += 0.45
    if citation.endswith(".pdf"):
        score += 0.08

    if retrieval_intent.aggregation_shape.startswith("monthly"):
        if any(token in combined_text for token in ("monthly", "month", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")):
            score += 0.45
        if "receipts expenditures and balances" in combined_text or "monthly treasury statement" in combined_text:
            score += 0.6
    if retrieval_intent.aggregation_shape == "inflation_adjusted_monthly_difference" and any(token in combined_text for token in ("cpi", "inflation", "price index")):
        score += 0.4
    if retrieval_intent.document_family == "treasury_bulletin" and "treasury bulletin" in combined_text:
        score += 0.5

    if any(term in combined_text for term in retrieval_intent.must_exclude_terms):
        score -= 0.7
    if any(
        bad in combined_text
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
    _ = benchmark_overrides
    return score


def rank_search_candidates(
    candidates: list[dict[str, Any]],
    retrieval_intent: RetrievalIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return sorted(
        [RetrievalCandidate.model_validate(item).model_dump() for item in candidates],
        key=lambda item: (
            search_candidate_score(item, retrieval_intent, source_bundle, benchmark_overrides),
            -int(item.get("rank", 999) or 999),
        ),
        reverse=True,
    )
