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


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 1 and token not in _STOP_WORDS]


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
    semantic_tokens = set(_tokenize(" ".join(part for part in [query, entity, metric] if part)))
    return {
        "query": query,
        "query_tokens": set(_tokenize(query)),
        "semantic_tokens": semantic_tokens,
        "entity_tokens": set(_tokenize(entity)),
        "metric_tokens": set(_tokenize(metric)),
        "target_years": _normalized_years(target_years) or _normalized_years(_query_years(query)),
        "publication_year_window": _normalized_years(publication_year_window),
        "preferred_publication_years": _normalized_years(preferred_publication_years),
        "period_type": str(period_type or "").strip().lower(),
        "granularity_requirement": str(granularity_requirement or "").strip().lower(),
    }


def _record_publication_year(record: dict[str, Any]) -> str:
    return str(record.get("publication_year", "") or "").strip()


def _table_units(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in list(record.get("table_units", [])) if isinstance(item, dict)]


def _table_unit_text(unit: dict[str, Any]) -> str:
    return " ".join(
        [
            str(unit.get("locator", "") or ""),
            str(unit.get("page_locator", "") or ""),
            " ".join(str(item or "") for item in list(unit.get("headers", []))),
            " ".join(str(item or "") for item in list(unit.get("row_labels", []))),
            " ".join(str(item or "") for item in list(unit.get("column_paths", []))),
            " ".join(str(item or "") for item in list(unit.get("unit_hints", []))),
            str(unit.get("preview_text", "") or ""),
            str(unit.get("table_family", "") or ""),
            str(unit.get("period_type", "") or ""),
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
    if requested in {"point_lookup", "point_in_time"} and period_type == "point_lookup":
        return 0.35
    return 0.0


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


def _table_unit_score(unit: dict[str, Any], profile: dict[str, Any]) -> float:
    tokens = set(_tokenize(_table_unit_text(unit)))
    semantic_tokens = set(profile.get("semantic_tokens", set()))
    entity_tokens = set(profile.get("entity_tokens", set()))
    metric_tokens = set(profile.get("metric_tokens", set()))
    target_years = set(profile.get("target_years", []))
    header_tokens = set(_tokenize(" ".join(str(item or "") for item in list(unit.get("headers", [])))))
    row_tokens = set(_tokenize(" ".join(str(item or "") for item in list(unit.get("row_labels", [])))))
    year_refs = {str(item) for item in list(unit.get("year_refs", [])) if str(item)}
    month_coverage = list(unit.get("month_coverage", []))
    score = 0.0
    score += 0.18 * len(semantic_tokens & tokens)
    score += 0.16 * len(entity_tokens & row_tokens)
    score += 0.14 * len(metric_tokens & header_tokens)
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
    table_unit_text = " ".join(_table_unit_text(unit) for unit in _table_units(record)[:6])
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
        str(record.get("preview_text", "")),
    ]
    return " ".join(part for part in parts if part)


def _score_record(record: dict[str, Any], profile: dict[str, Any]) -> tuple[float, dict[str, Any] | None]:
    query = str(profile.get("query", "") or "")
    text_tokens = set(_tokenize(_record_text(record)))
    alias_tokens = set(_tokenize(" ".join(str(item) for item in record.get("source_aliases", []))))
    query_tokens = set(profile.get("query_tokens", set()))
    if not query_tokens:
        return 0.0, None
    overlap = query_tokens & text_tokens
    if not overlap:
        return 0.0, None
    score = 0.3 * len(overlap)
    score += 0.7 * len(query_tokens & alias_tokens)

    years = set(str(year) for year in record.get("years", []))
    for year in _query_years(query):
        if year in years:
            score += 1.0
    lowered_query = (query or "").lower()
    if "treasury bulletin" in lowered_query and record.get("is_treasury_bulletin"):
        score += 1.2
    if any(token in lowered_query for token in ("monthly", "calendar months")) and record.get("has_month_names"):
        score += 0.8
    if any(token in lowered_query for token in ("table", "row", "column", "reported values")) and record.get("has_table_like_rows"):
        score += 0.8
    if any(token in lowered_query for token in ("million", "billion", "percent", "nominal")):
        score += 0.2 * len(set(_tokenize(" ".join(record.get("unit_hints", [])))) & query_tokens)
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
        score += best_unit_score
    return score, best_unit


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
    candidate_records = _record_pool(records, profile, top_k)
    scored = []
    for record in candidate_records:
        if allowed_document_ids and str(record.get("document_id", "")) not in allowed_document_ids:
            continue
        score, best_unit = _score_record(record, profile)
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
        "source_files_filter_applied": bool(allowed_document_ids),
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
