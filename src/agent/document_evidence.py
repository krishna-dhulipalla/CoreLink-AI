"""
Document Evidence Helpers
=========================
Structured document-evidence helpers used by the active engine.
"""

from __future__ import annotations

import re
from typing import Any

from agent.contracts import DocumentEvidenceRecord, ToolResult


def guess_document_format(source: str) -> str:
    normalized = (source or "").lower().split("?", 1)[0]
    for ext, fmt in (
        (".pdf", "pdf"),
        (".xlsx", "excel"),
        (".xls", "excel"),
        (".csv", "csv"),
        (".json", "json"),
        (".docx", "word"),
        (".doc", "word"),
        (".txt", "text"),
        (".md", "text"),
    ):
        if normalized.endswith(ext):
            return fmt
    return "unknown"


def _document_id_from_citation(citation: str) -> str:
    tail = (citation or "").rstrip("/").rsplit("/", 1)[-1]
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", tail.split("?", 1)[0]).strip("_").lower()
    return cleaned or "document"


def _dedupe_dict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        signature = repr(sorted(row.items()))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped


def _dedupe_table_rows(rows: list[list[str]]) -> list[list[str]]:
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        signature = tuple(row)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped


def build_document_placeholders(citations: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, str) or not citation.startswith(("http://", "https://")):
            continue
        record = DocumentEvidenceRecord(
            document_id=_document_id_from_citation(citation),
            citation=citation,
            status="discovered",
            metadata={"format": guess_document_format(citation)},
        )
        records.append(record.model_dump())
    return records


def document_records_from_tool_result(
    tool_result: dict[str, Any] | ToolResult,
    tool_args: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    result = ToolResult.model_validate(tool_result)
    if result.errors or not result.facts:
        return []

    args = tool_args or {}
    citation = str(args.get("url") or result.facts.get("citation") or "")
    records: list[dict[str, Any]] = []

    if result.type == "list_reference_files":
        for entry in result.facts.get("documents", []):
            if not isinstance(entry, dict):
                continue
            entry_citation = str(entry.get("citation") or entry.get("url") or "")
            if not entry_citation:
                continue
            records.append(
                DocumentEvidenceRecord(
                    document_id=str(entry.get("document_id") or _document_id_from_citation(entry_citation)),
                    citation=entry_citation,
                    status="discovered",
                    metadata={
                        "format": str(entry.get("format") or guess_document_format(entry_citation)),
                    },
                ).model_dump()
            )
        return records

    if result.type != "fetch_reference_file":
        return []

    document_id = str(result.facts.get("document_id") or _document_id_from_citation(citation))
    metadata = dict(result.facts.get("metadata", {}))
    if citation:
        metadata.setdefault("source_url", citation)
    chunks = [
        dict(chunk)
        for chunk in result.facts.get("chunks", [])
        if isinstance(chunk, dict) and chunk.get("text")
    ]
    tables = []
    for table in result.facts.get("tables", []):
        if not isinstance(table, dict):
            continue
        tables.append(
            {
                "locator": table.get("locator", ""),
                "headers": list(table.get("headers", [])),
                "rows": _dedupe_table_rows([list(row) for row in table.get("rows", []) if isinstance(row, list)]),
                "citation": table.get("citation", citation),
            }
        )
    numeric_summaries = _dedupe_dict_rows(
        [
            dict(summary)
            for summary in result.facts.get("numeric_summaries", [])
            if isinstance(summary, dict) and "metric" in summary
        ]
    )
    status = "extracted" if (chunks or tables or numeric_summaries) else "indexed"

    records.append(
        DocumentEvidenceRecord(
            document_id=document_id,
            citation=citation,
            status=status,
            metadata=metadata,
            chunks=chunks,
            tables=tables,
            numeric_summaries=numeric_summaries,
        ).model_dump()
    )
    return records


def merge_document_evidence_records(
    existing: list[dict[str, Any]] | None,
    additions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for raw in [*(existing or []), *(additions or [])]:
        record = DocumentEvidenceRecord.model_validate(raw)
        key = record.document_id or _document_id_from_citation(record.citation)
        if key not in merged:
            merged[key] = record.model_dump()
            continue

        current = merged[key]
        current_record = DocumentEvidenceRecord.model_validate(current)
        higher_status = max(
            [current_record.status, record.status],
            key=lambda value: {"discovered": 0, "indexed": 1, "extracted": 2}.get(value, 0),
        )
        current_record.status = higher_status
        if record.citation:
            current_record.citation = record.citation
        current_record.metadata = {**current_record.metadata, **record.metadata}
        current_record.chunks = _dedupe_dict_rows([*current_record.chunks, *record.chunks])
        current_record.tables = _dedupe_dict_rows([*current_record.tables, *record.tables])
        current_record.numeric_summaries = _dedupe_dict_rows(
            [*current_record.numeric_summaries, *record.numeric_summaries]
        )
        merged[key] = current_record.model_dump()

    return list(merged.values())


def summarize_document_evidence(records: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for raw in records or []:
        record = DocumentEvidenceRecord.model_validate(raw)
        summaries.append(
            {
                "document_id": record.document_id,
                "citation": record.citation,
                "status": record.status,
                "metadata": {
                    key: value
                    for key, value in record.metadata.items()
                    if key in {"file_name", "format", "window", "sheet", "size_kb", "source_url"}
                },
                "chunk_count": len(record.chunks),
                "table_count": len(record.tables),
                "numeric_summary_count": len(record.numeric_summaries),
                "chunk_locators": [chunk.get("locator", "") for chunk in record.chunks[:3]],
                "table_locators": [table.get("locator", "") for table in record.tables[:3]],
            }
        )
    return summaries


def has_extracted_document_evidence(records: list[dict[str, Any]] | None) -> bool:
    for raw in records or []:
        record = DocumentEvidenceRecord.model_validate(raw)
        if record.status == "extracted" and (record.chunks or record.tables or record.numeric_summaries):
            return True
    return False
