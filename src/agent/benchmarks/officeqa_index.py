"""Searchable local index for the OfficeQA corpus."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .officeqa_manifest import (
    build_manifest_entry,
    iter_officeqa_files,
    load_officeqa_manifest,
    match_source_files_to_records,
    normalize_source_name,
    resolve_officeqa_corpus_root,
    resolve_officeqa_index_dir,
    validate_manifest_records,
    write_officeqa_manifest,
)

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "for", "from", "with",
    "what", "which", "using", "specifically", "only", "reported", "values",
})
_SURFACE_NOISE_TOKENS = frozenset({
    "row", "rows", "table", "page", "pages", "note", "notes", "source",
    "analysis", "summary", "continued", "statement", "report",
})
_QUERY_PROVENANCE_TOKENS = frozenset({
    "according", "treasury", "bulletin", "department", "reported",
})
_QUERY_GENERIC_RANK_TOKENS = frozenset({
    "table", "row", "rows", "column", "columns", "page", "pages", "source",
    "total", "calendar", "year", "years", "monthly", "annual", "reported",
    "values", "figure", "figures",
})
_DEBT_FAMILY_TOKENS = frozenset({"debt", "obligations", "guaranteed", "outstanding", "liabilities", "securities"})
_FLOW_FAMILY_TOKENS = frozenset({"receipts", "expenditures", "revenue", "revenues", "collections", "outlays", "spending"})


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 1 and token not in _STOP_WORDS]


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values() if item is not None)
    return str(value or "")


def _query_years(query: str) -> list[str]:
    return re.findall(r"\b((?:19|20)\d{2})\b", query or "")


def _normalized_years(values: list[str] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        year = str(raw or "").strip()
        if not re.fullmatch(r"(?:19|20)\d{2}", year):
            continue
        if year in seen:
            continue
        seen.add(year)
        ordered.append(year)
    return ordered


def _semantic_phrases(query: str, entity: str, metric: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for raw in (entity, metric):
        normalized = re.sub(r"\s+", " ", str(raw or "").strip().lower())
        if len(_tokenize(normalized)) >= 2 and normalized not in seen:
            seen.add(normalized)
            phrases.append(normalized)
    query_tokens = _tokenize(query)
    for n in (2, 3):
        for index in range(0, max(0, len(query_tokens) - n + 1)):
            phrase = " ".join(query_tokens[index : index + n]).strip()
            if len(phrase) < 8 or phrase in seen or re.fullmatch(r"(?:19|20)\d{2}(?: (?:19|20)\d{2})*", phrase):
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= 8:
                return phrases
    return phrases


def _rank_tokens(query: str, entity: str, metric: str, target_years: list[str]) -> set[str]:
    query_tokens = set(_tokenize(query))
    entity_tokens = set(_tokenize(entity))
    metric_tokens = set(_tokenize(metric))
    year_tokens = {year for year in target_years if year}
    prioritized = (entity_tokens | metric_tokens | year_tokens) - _QUERY_PROVENANCE_TOKENS
    if prioritized:
        return prioritized
    return {
        token
        for token in query_tokens
        if token not in _QUERY_PROVENANCE_TOKENS and token not in _QUERY_GENERIC_RANK_TOKENS
    } or (query_tokens - _QUERY_PROVENANCE_TOKENS)


def _query_profile(
    *,
    query: str,
    target_years: list[str] | None = None,
    publication_year_window: list[str] | None = None,
    preferred_publication_years: list[str] | None = None,
    period_type: str = "",
    granularity_requirement: str = "",
    entity: str = "",
    metric: str = "",
) -> dict[str, Any]:
    normalized_target_years = _normalized_years(target_years) or _normalized_years(_query_years(query))
    semantic_tokens = set(_tokenize(" ".join(part for part in [query, entity, metric] if part)))
    return {
        "query": query,
        "query_tokens": set(_tokenize(query)),
        "semantic_tokens": semantic_tokens,
        "entity_tokens": set(_tokenize(entity)),
        "metric_tokens": set(_tokenize(metric)),
        "rank_tokens": _rank_tokens(query, entity, metric, normalized_target_years),
        "entity_text": str(entity or "").strip(),
        "metric_text": str(metric or "").strip(),
        "semantic_phrases": _semantic_phrases(query, entity, metric),
        "target_years": normalized_target_years,
        "publication_year_window": _normalized_years(publication_year_window),
        "preferred_publication_years": _normalized_years(preferred_publication_years),
        "period_type": str(period_type or "").strip().lower(),
        "granularity_requirement": str(granularity_requirement or "").strip().lower(),
    }


def _record_publication_year(record: dict[str, Any]) -> str:
    return str(record.get("publication_year", "") or "").strip()


def _normalize_source_files_policy(policy: str) -> str:
    normalized = str(policy or "").strip().lower()
    if normalized in {"hard", "soft", "off"}:
        return normalized
    return "soft"


def _table_units(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in list(record.get("table_units", [])) if isinstance(item, dict)]


def _unit_structured_text(unit: dict[str, Any]) -> str:
    return " ".join(
        [
            _flatten_text(unit.get("locator", "")),
            _flatten_text(unit.get("page_locator", "")),
            _flatten_text(unit.get("context_text", "")),
            " ".join(_flatten_text(item) for item in list(unit.get("heading_chain", []))),
            " ".join(_flatten_text(item) for item in list(unit.get("headers", []))),
            " ".join(_flatten_text(item) for item in list(unit.get("row_labels", []))),
            " ".join(_flatten_text(item) for item in list(unit.get("row_paths", []))),
            " ".join(_flatten_text(item) for item in list(unit.get("column_paths", []))),
            " ".join(_flatten_text(item) for item in list(unit.get("unit_hints", []))),
            _flatten_text(unit.get("table_family", "")),
            _flatten_text(unit.get("period_type", "")),
        ]
    ).strip()


def _unit_preview_text(unit: dict[str, Any]) -> str:
    return _flatten_text(unit.get("preview_text", "")).strip()


def _table_unit_text(unit: dict[str, Any]) -> str:
    return " ".join(
        [
            _unit_structured_text(unit),
            _unit_preview_text(unit),
        ]
    ).strip()


def _period_type_fit(unit: dict[str, Any], profile: dict[str, Any]) -> float:
    period_type = str(unit.get("period_type", "") or "").strip().lower()
    requested_period = str(profile.get("period_type", "") or "").strip().lower()
    granularity = str(profile.get("granularity_requirement", "") or "").strip().lower()
    if not requested_period and not granularity:
        return 0.0
    requested = requested_period or granularity
    if requested == "monthly_series":
        if period_type == "monthly_series":
            return 0.95
        if period_type in {"calendar_year", "fiscal_year"}:
            return -0.4
    if requested == "fiscal_year":
        if period_type == "fiscal_year":
            return 0.8
        if period_type == "calendar_year":
            return -0.2
    if requested == "calendar_year":
        if period_type == "calendar_year":
            return 0.65
        if period_type == "monthly_series":
            return 0.2
        if period_type in {"fiscal_year", "narrative_support"}:
            return -0.28
    if requested in {"point_lookup", "point_in_time"} and period_type == "point_lookup":
        return 0.35
    if requested in {"point_lookup", "point_in_time"} and period_type in {"calendar_year", "fiscal_year", "monthly_series"}:
        return -0.22
    return 0.0


def _required_table_family(profile: dict[str, Any]) -> str:
    requested = str(profile.get("granularity_requirement", "") or profile.get("period_type", "") or "").strip().lower()
    metric_tokens = set(profile.get("metric_tokens", set()))
    metric_text = " ".join(sorted(metric_tokens))
    if requested == "monthly_series":
        return "monthly_series"
    if requested == "fiscal_year":
        return "fiscal_year_comparison"
    if any(token in metric_text for token in ("debt", "outstanding", "obligations", "liabilities", "securities")):
        return "debt_or_balance_sheet"
    if any(token in metric_text for token in ("expenditures", "receipts", "revenue", "collections", "outlays", "spending")):
        return "category_breakdown"
    return ""


def _table_family_fit(unit: dict[str, Any], profile: dict[str, Any]) -> float:
    family = str(unit.get("table_family", "") or "").strip().lower()
    required = _required_table_family(profile)
    if not required or not family:
        return 0.0
    family_conf = float(unit.get("table_family_confidence", unit.get("table_confidence", 0.0)) or 0.0)
    if family == required:
        return 0.9 * max(0.4, family_conf)
    if required == "category_breakdown" and family in {"annual_summary", "fiscal_year_comparison"}:
        return 0.18
    if required == "fiscal_year_comparison" and family == "category_breakdown":
        return 0.22
    if required == "monthly_series" and family in {"annual_summary", "category_breakdown"}:
        return -0.45
    if required == "debt_or_balance_sheet" and family != "debt_or_balance_sheet":
        return -0.4
    if required == "category_breakdown" and family == "debt_or_balance_sheet":
        return -0.6
    return -0.18


def _publication_year_fit(publication_year: str, profile: dict[str, Any]) -> float:
    if not publication_year:
        return 0.0
    preferred = list(profile.get("preferred_publication_years", []))
    if publication_year in preferred:
        position = preferred.index(publication_year)
        return max(0.2, 1.15 - (0.18 * position))
    window = set(profile.get("publication_year_window", []))
    if publication_year in window:
        return 0.25
    return -0.18 if window else 0.0


def _surface_family(tokens: set[str]) -> str:
    debt_hits = len(tokens & _DEBT_FAMILY_TOKENS)
    flow_hits = len(tokens & _FLOW_FAMILY_TOKENS)
    if debt_hits >= 2 and debt_hits > flow_hits:
        return "debt"
    if flow_hits >= 2 and flow_hits > debt_hits:
        return "flow"
    return ""


def _surface_family_consistency_penalty(heading_tokens: set[str], body_tokens: set[str]) -> float:
    heading_core = {token for token in heading_tokens if token not in _SURFACE_NOISE_TOKENS}
    body_core = {token for token in body_tokens if token not in _SURFACE_NOISE_TOKENS}
    if not heading_core or not body_core:
        return 0.0
    heading_family = _surface_family(heading_core)
    body_family = _surface_family(body_core)
    if heading_family and body_family and heading_family != body_family:
        return -0.55
    overlap = heading_core & body_core
    if not overlap and len(heading_core) >= 2 and len(body_core) >= 4:
        return -0.18
    return 0.0


def _phrase_bonus(text: str, phrases: list[str], *, weight: float) -> float:
    lowered = text.lower()
    return weight * sum(1 for phrase in phrases if phrase and phrase in lowered)


def _best_surface_match(text: str, phrase: str) -> float:
    normalized_text = " ".join(_tokenize(text))
    normalized_phrase = " ".join(_tokenize(phrase))
    if not normalized_text or not normalized_phrase:
        return 0.0
    if normalized_phrase in normalized_text:
        return 1.0
    text_tokens = set(normalized_text.split())
    phrase_tokens = set(normalized_phrase.split())
    if not phrase_tokens:
        return 0.0
    return float(len(text_tokens & phrase_tokens)) / max(1, len(phrase_tokens))


def _row_focus_score(unit: dict[str, Any], entity_text: str) -> float:
    entity_phrase = " ".join(_tokenize(entity_text))
    if not entity_phrase:
        return 0.0
    row_candidates = [str(item or "") for item in list(unit.get("row_paths", []))]
    row_candidates.extend(str(item or "") for item in list(unit.get("row_labels", [])))
    normalized_rows = [row for row in row_candidates if row.strip()]
    if not normalized_rows:
        return 0.0
    row_scores = [_best_surface_match(row, entity_phrase) for row in normalized_rows]
    best = max(row_scores, default=0.0)
    if best <= 0.0:
        return 0.0
    strong_match_count = sum(1 for score in row_scores if score >= 0.72)
    row_count = max(1, len(normalized_rows))
    focus_ratio = 1.0 / max(1, strong_match_count)
    compactness = min(1.0, 6.0 / float(row_count))
    return best * ((0.7 * focus_ratio) + (0.3 * compactness))


def _table_unit_score(unit: dict[str, Any], profile: dict[str, Any]) -> float:
    structured_text = _unit_structured_text(unit)
    preview_text = _unit_preview_text(unit)
    tokens = set(_tokenize(structured_text))
    preview_tokens = set(_tokenize(preview_text))
    semantic_tokens = set(profile.get("semantic_tokens", set()))
    entity_tokens = set(profile.get("entity_tokens", set()))
    metric_tokens = set(profile.get("metric_tokens", set()))
    semantic_phrases = list(profile.get("semantic_phrases", []))
    target_years = set(profile.get("target_years", []))
    header_text = " ".join(_flatten_text(item) for item in list(unit.get("headers", [])))
    row_text = " ".join(_flatten_text(item) for item in list(unit.get("row_labels", [])))
    heading_text = " ".join(_flatten_text(item) for item in list(unit.get("heading_chain", [])))
    column_text = " ".join(_flatten_text(item) for item in list(unit.get("column_paths", [])))
    entity_text = str(profile.get("entity_text", "") or "")
    metric_text = str(profile.get("metric_text", "") or "")
    header_tokens = set(_tokenize(header_text))
    row_tokens = set(_tokenize(row_text))
    heading_tokens = set(_tokenize(heading_text))
    column_tokens = set(_tokenize(column_text))
    body_tokens = header_tokens | row_tokens | column_tokens
    year_refs = {str(item) for item in list(unit.get("year_refs", [])) if str(item)}
    month_coverage = list(unit.get("month_coverage", []))
    score = 0.0
    score += 0.22 * len(semantic_tokens & tokens)
    score += 0.03 * len(semantic_tokens & preview_tokens)
    score += 0.16 * len(entity_tokens & row_tokens)
    score += 0.14 * len(metric_tokens & header_tokens)
    score += 0.1 * len(metric_tokens & heading_tokens)
    score += 0.08 * len(entity_tokens & heading_tokens)
    score += 0.08 * len(metric_tokens & column_tokens)
    score += _phrase_bonus(heading_text, semantic_phrases, weight=0.18)
    score += _phrase_bonus(header_text, semantic_phrases, weight=0.16)
    score += _phrase_bonus(row_text, semantic_phrases, weight=0.16)
    score += _phrase_bonus(column_text, semantic_phrases, weight=0.14)
    score += _phrase_bonus(preview_text[:320], semantic_phrases, weight=0.05)
    if entity_text:
        score += 0.42 * _best_surface_match(heading_text, entity_text)
        score += 0.48 * _best_surface_match(row_text, entity_text)
        score += 0.4 * _row_focus_score(unit, entity_text)
    if metric_text:
        score += 0.34 * _best_surface_match(heading_text, metric_text)
        score += 0.26 * _best_surface_match(header_text, metric_text)
        score += 0.22 * _best_surface_match(column_text, metric_text)
    focus_tokens = sorted((entity_tokens | metric_tokens) - {"total"})
    if entity_text and metric_tokens and focus_tokens:
        focus_phrase = " ".join(focus_tokens)
        locator_text = _flatten_text(unit.get("locator", ""))
        score += 0.55 * _best_surface_match(heading_text, focus_phrase)
        score += 0.18 * _best_surface_match(locator_text, focus_phrase)
    if entity_tokens and not (entity_tokens & (row_tokens | column_tokens | heading_tokens)):
        score -= 0.35
    if metric_tokens and not (metric_tokens & (header_tokens | column_tokens | heading_tokens)):
        score -= 0.28
    score += _surface_family_consistency_penalty(heading_tokens, body_tokens)
    if target_years and year_refs & target_years:
        score += 0.9
    elif year_refs and target_years and not (year_refs & target_years):
        score -= 0.18
    if profile.get("granularity_requirement") == "monthly_series":
        if len(month_coverage) >= 6:
            score += 0.7
        elif month_coverage:
            score += 0.28
    score += _period_type_fit(unit, profile)
    score += _table_family_fit(unit, profile)
    score += 0.35 * float(unit.get("table_confidence", 0.0) or 0.0)
    return score


def _record_pool(records: list[dict[str, Any]], profile: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    target_years = set(profile.get("target_years", []))
    publication_window = set(profile.get("publication_year_window", []))
    if not target_years and not publication_window:
        return records
    filtered: list[dict[str, Any]] = []
    for record in records:
        publication_year = _record_publication_year(record)
        referenced_years = {str(item) for item in list(record.get("years", [])) if str(item)}
        for unit in _table_units(record):
            referenced_years.update(str(item) for item in list(unit.get("year_refs", [])) if str(item))
        if publication_year and publication_year in publication_window:
            filtered.append(record)
            continue
        if referenced_years & target_years:
            filtered.append(record)
    if len(filtered) >= max(top_k * 3, 18):
        return filtered
    return filtered or records


def build_officeqa_index(
    corpus_root: str | Path | None = None,
    index_dir: str | Path | None = None,
    max_files: int = 4000,
) -> dict[str, Any]:
    root = resolve_officeqa_corpus_root(str(corpus_root) if corpus_root is not None else None)
    if root is None:
        raise FileNotFoundError("No OfficeQA corpus directory found. Set OFFICEQA_CORPUS_DIR or pass --corpus-root.")
    resolved_index_dir = resolve_officeqa_index_dir(root, str(index_dir) if index_dir is not None else None, create=True)
    entries = [build_manifest_entry(path, root) for path in iter_officeqa_files(root, max_files=max_files)]
    summary = write_officeqa_manifest(entries, root, resolved_index_dir)
    summary["validation"] = validate_manifest_records(entries)
    summary["index_dir"] = str(resolved_index_dir)
    return summary


def officeqa_index_available(corpus_root: str | Path | None = None, index_dir: str | Path | None = None) -> bool:
    root = resolve_officeqa_corpus_root(str(corpus_root) if corpus_root is not None else None)
    if root is None:
        return False
    records = load_officeqa_manifest(root, resolve_officeqa_index_dir(root, str(index_dir) if index_dir is not None else None))
    return bool(records)


def _load_records(corpus_root: str | Path | None = None, index_dir: str | Path | None = None) -> tuple[Path | None, list[dict[str, Any]]]:
    root = resolve_officeqa_corpus_root(str(corpus_root) if corpus_root is not None else None)
    if root is None:
        return None, []
    resolved_index_dir = resolve_officeqa_index_dir(root, str(index_dir) if index_dir is not None else None)
    return root, load_officeqa_manifest(root, resolved_index_dir)


def _record_text(record: dict[str, Any]) -> str:
    table_unit_text = " ".join(_unit_structured_text(unit) for unit in _table_units(record)[:6])
    preview_text = " ".join(_unit_preview_text(unit)[:180] for unit in _table_units(record)[:3])
    parts: list[str] = [
        str(record.get("relative_path", "")),
        str(record.get("file_name", "")),
        str(record.get("publication_year", "")),
        str(record.get("publication_month", "")),
        " ".join(str(item) for item in record.get("section_titles", [])),
        " ".join(str(item) for item in record.get("table_headers", [])),
        " ".join(str(item) for item in record.get("row_labels", [])),
        " ".join(str(item) for item in record.get("unit_hints", [])),
        " ".join(str(item) for item in record.get("period_types", [])),
        table_unit_text,
        preview_text,
    ]
    return " ".join(part for part in parts if part)


def _score_record(record: dict[str, Any], profile: dict[str, Any]) -> tuple[float, dict[str, Any] | None]:
    query = str(profile.get("query", "") or "")
    text_tokens = set(_tokenize(_record_text(record)))
    alias_tokens = set(_tokenize(" ".join(str(item) for item in record.get("source_aliases", []))))
    query_tokens = set(profile.get("query_tokens", set()))
    rank_tokens = set(profile.get("rank_tokens", set())) or query_tokens
    if not rank_tokens:
        return 0.0, None
    overlap = rank_tokens & text_tokens
    if not overlap:
        return 0.0, None
    score = 0.18 * len(overlap)
    score += 0.2 * len(rank_tokens & alias_tokens)

    years = set(str(year) for year in record.get("years", []))
    for year in _query_years(query):
        if year in years:
            score += 0.7
    lowered_query = (query or "").lower()
    if "treasury bulletin" in lowered_query and record.get("is_treasury_bulletin"):
        score += 0.45
    if any(token in lowered_query for token in ("monthly", "calendar months")) and record.get("has_month_names"):
        score += 0.35
    if any(token in lowered_query for token in ("table", "row", "column", "reported values")) and record.get("has_table_like_rows"):
        score += 0.35
    if any(token in lowered_query for token in ("million", "billion", "percent", "nominal")):
        score += 0.12 * len(set(_tokenize(" ".join(record.get("unit_hints", [])))) & rank_tokens)
    publication_year = _record_publication_year(record)
    score += _publication_year_fit(publication_year, profile)

    best_unit: dict[str, Any] | None = None
    best_unit_score = 0.0
    for unit in _table_units(record):
        unit_score = _table_unit_score(unit, profile)
        if best_unit is None or unit_score > best_unit_score:
            best_unit = unit
            best_unit_score = unit_score
    if best_unit is not None:
        score += 1.25 * best_unit_score
    return score, best_unit


def _source_hint_score(document_id: str, allowed_document_ids: set[str], source_files_policy: str) -> float:
    if not allowed_document_ids or source_files_policy == "off":
        return 0.0
    if document_id in allowed_document_ids:
        return 0.45 if source_files_policy == "soft" else 0.8
    if source_files_policy == "hard":
        return -10.0
    return 0.0


def search_officeqa_corpus_index(
    query: str,
    corpus_root: str | Path | None = None,
    index_dir: str | Path | None = None,
    top_k: int = 5,
    snippet_chars: int = 700,
    source_files: list[str] | None = None,
    target_years: list[str] | None = None,
    publication_year_window: list[str] | None = None,
    preferred_publication_years: list[str] | None = None,
    period_type: str = "",
    granularity_requirement: str = "",
    entity: str = "",
    metric: str = "",
    source_files_policy: str = "soft",
) -> dict[str, Any]:
    root, records = _load_records(corpus_root, index_dir)
    if root is None or not records:
        return {"error": "No OfficeQA corpus index is available. Run scripts/build_officeqa_index.py first."}
    profile = _query_profile(
        query=query,
        target_years=target_years,
        publication_year_window=publication_year_window,
        preferred_publication_years=preferred_publication_years,
        period_type=period_type,
        granularity_requirement=granularity_requirement,
        entity=entity,
        metric=metric,
    )

    allowed_document_ids = {
        item.get("document_id", "")
        for item in resolve_source_files_to_manifest(source_files or [], corpus_root=corpus_root, index_dir=index_dir)
        if item.get("matched")
    }
    normalized_source_files_policy = _normalize_source_files_policy(source_files_policy)
    candidate_records = _record_pool(records, profile, top_k)
    scored = []
    for record in candidate_records:
        document_id = str(record.get("document_id", ""))
        if normalized_source_files_policy == "hard" and allowed_document_ids and document_id not in allowed_document_ids:
            continue
        score, best_unit = _score_record(record, profile)
        score += _source_hint_score(document_id, allowed_document_ids, normalized_source_files_policy)
        if score <= 0:
            continue
        scored.append((score, record, best_unit))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("relative_path", ""))))
    selected = scored[: max(1, min(top_k, 8))]

    results: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for rank, (score, record, best_unit) in enumerate(selected, start=1):
        preview_source = str((best_unit or {}).get("preview_text", "") or record.get("preview_text", ""))
        preview = preview_source[:snippet_chars]
        metadata = {
            "years": list(record.get("years", [])),
            "publication_year": str(record.get("publication_year", "") or ""),
            "publication_month": str(record.get("publication_month", "") or ""),
            "page_markers": list(record.get("page_markers", [])),
            "section_titles": list(record.get("section_titles", []))[:8],
            "table_headers": list(record.get("table_headers", []))[:8],
            "row_labels": list(record.get("row_labels", []))[:8],
            "unit_hints": list(record.get("unit_hints", [])),
            "month_coverage": list(record.get("month_coverage", [])),
            "period_types": list(record.get("period_types", []))[:6],
            "evidence_unit_count": len(list(record.get("table_units", []))),
            "best_evidence_unit": dict(best_unit or {}),
        }
        results.append(
            {
                "rank": rank,
                "title": str(record.get("relative_path", "")),
                "snippet": preview,
                "url": str(record.get("relative_path", "")),
                "score": round(score, 3),
                "document_id": str(record.get("document_id", "")),
                "metadata": metadata,
            }
        )
        documents.append(
            {
                "document_id": str(record.get("document_id", "")),
                "citation": str(record.get("relative_path", "")),
                "format": str(record.get("file_format", "")),
                "path": str(record.get("relative_path", "")),
                "metadata": metadata,
            }
        )
    return {
        "query": query,
        "corpus_root": str(root),
        "results": results,
        "documents": documents,
        "result_count": len(results),
        "index_mode": "officeqa_manifest",
        "candidate_pool_size": len(candidate_records),
        "source_files_filter_applied": bool(allowed_document_ids) and normalized_source_files_policy == "hard",
        "source_files_prior_applied": bool(allowed_document_ids) and normalized_source_files_policy in {"soft", "hard"},
        "source_files_policy": normalized_source_files_policy,
    }


def resolve_indexed_corpus_document(
    document_id: str = "",
    path: str = "",
    corpus_root: str | Path | None = None,
    index_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    _, records = _load_records(corpus_root, index_dir)
    if not records:
        return None
    normalized_path = normalize_source_name(path)
    for record in records:
        if document_id and str(record.get("document_id", "")) == document_id:
            return record
        if path and str(record.get("relative_path", "")) == path:
            return record
        if normalized_path and normalized_path in set(str(item) for item in record.get("source_aliases", [])):
            return record
    return None


def resolve_source_files_to_manifest(
    source_files: list[str],
    corpus_root: str | Path | None = None,
    index_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    _, records = _load_records(corpus_root, index_dir)
    if not records:
        return []
    return match_source_files_to_records(source_files, records)


def validate_officeqa_index(
    corpus_root: str | Path | None = None,
    index_dir: str | Path | None = None,
) -> dict[str, Any]:
    _, records = _load_records(corpus_root, index_dir)
    if not records:
        return {"document_count": 0, "issue_count": 0, "issues": []}
    return validate_manifest_records(records)
