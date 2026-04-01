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
    parts: list[str] = [
        str(record.get("relative_path", "")),
        str(record.get("file_name", "")),
        " ".join(str(item) for item in record.get("section_titles", [])),
        " ".join(str(item) for item in record.get("table_headers", [])),
        " ".join(str(item) for item in record.get("row_labels", [])),
        " ".join(str(item) for item in record.get("unit_hints", [])),
        str(record.get("preview_text", "")),
    ]
    return " ".join(part for part in parts if part)


def _score_record(record: dict[str, Any], query: str) -> float:
    text_tokens = set(_tokenize(_record_text(record)))
    alias_tokens = set(_tokenize(" ".join(str(item) for item in record.get("source_aliases", []))))
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    if not overlap:
        return 0.0
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
    return score


def search_officeqa_corpus_index(
    query: str,
    corpus_root: str | Path | None = None,
    index_dir: str | Path | None = None,
    top_k: int = 5,
    snippet_chars: int = 700,
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    root, records = _load_records(corpus_root, index_dir)
    if root is None or not records:
        return {"error": "No OfficeQA corpus index is available. Run scripts/build_officeqa_index.py first."}

    allowed_document_ids = {
        item.get("document_id", "")
        for item in resolve_source_files_to_manifest(source_files or [], corpus_root=corpus_root, index_dir=index_dir)
        if item.get("matched")
    }
    scored = []
    for record in records:
        if allowed_document_ids and str(record.get("document_id", "")) not in allowed_document_ids:
            continue
        score = _score_record(record, query)
        if score <= 0:
            continue
        scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("relative_path", ""))))
    selected = scored[: max(1, min(top_k, 8))]

    results: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for rank, (score, record) in enumerate(selected, start=1):
        preview = str(record.get("preview_text", ""))[:snippet_chars]
        metadata = {
            "years": list(record.get("years", [])),
            "page_markers": list(record.get("page_markers", [])),
            "section_titles": list(record.get("section_titles", []))[:8],
            "table_headers": list(record.get("table_headers", []))[:8],
            "row_labels": list(record.get("row_labels", []))[:8],
            "unit_hints": list(record.get("unit_hints", [])),
            "month_coverage": list(record.get("month_coverage", [])),
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
