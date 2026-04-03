"""Structured OfficeQA evidence projection with cell-level provenance."""

from __future__ import annotations

import re
from typing import Any

from agent.contracts import (
    DocumentMergedSeriesEvidence,
    DocumentEvidenceRecord,
    OfficeQAStructuredEvidence,
    OfficeQATableEvidence,
    OfficeQAValueEvidence,
    ToolResult,
)
from agent.document_evidence import document_records_from_tool_result, merge_document_evidence_records

_PAGE_LOCATOR_RE = re.compile(r"page\s+(\d+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_MONTH_INDEX = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _unit_profile(unit_hint: str, column_label: str = "") -> tuple[str, float, str]:
    text = f"{unit_hint} {column_label}".strip().lower()
    multiplier = 1.0
    unit_kind = "scalar"
    unit = ""
    if "percent" in text or "%" in text:
        multiplier = 0.01
        unit_kind = "percent"
        unit = "percent"
    elif "billion" in text:
        multiplier = 1_000_000_000.0
        unit_kind = "currency"
        unit = "billion"
    elif "million" in text:
        multiplier = 1_000_000.0
        unit_kind = "currency"
        unit = "million"
    elif "thousand" in text:
        multiplier = 1_000.0
        unit_kind = "currency"
        unit = "thousand"
    elif "dollar" in text or "cents" in text:
        multiplier = 1.0
        unit_kind = "currency"
        unit = "dollars"
    return unit, multiplier, unit_kind


def _numeric_value(raw: Any) -> float | None:
    compact = str(raw or "").replace(",", "").strip()
    if not compact:
        return None
    compact = compact.rstrip("%")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact):
        return None
    try:
        return float(compact)
    except ValueError:
        return None


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _extract_years(*parts: str) -> list[str]:
    text = " ".join(str(part or "") for part in parts)
    return list(dict.fromkeys(_YEAR_RE.findall(text)))


def _extract_month_index(*parts: str) -> int | None:
    text = " ".join(str(part or "") for part in parts).lower()
    for month, index in _MONTH_INDEX.items():
        if month in text:
            return index
    return None


def _series_key(value: dict[str, Any]) -> str:
    return "|".join(
        [
            _normalize_label(str(value.get("row_label", ""))),
            _normalize_label(str(value.get("column_label", ""))),
            _normalize_label(str(value.get("unit", ""))),
            _normalize_label(str(value.get("unit_kind", ""))),
        ]
    )


def _merged_series(values: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    all_document_ids: set[str] = set()
    all_years: set[str] = set()
    unit_consistent = True

    for value in values:
        if value.get("numeric_value") is None and value.get("normalized_value") is None:
            continue
        series_key = _series_key(value)
        if not series_key.strip("|"):
            continue
        document_id = str(value.get("document_id", "")).strip()
        years = _extract_years(
            str(value.get("row_label", "")),
            str(value.get("column_label", "")),
            str(value.get("table_locator", "")),
            str(value.get("page_locator", "")),
            str(value.get("citation", "")),
            document_id,
        )
        month_index = _extract_month_index(
            str(value.get("row_label", "")),
            str(value.get("column_label", "")),
            str(value.get("table_locator", "")),
        )
        series = grouped.setdefault(
            series_key,
            {
                "series_key": series_key,
                "row_label": str(value.get("row_label", "")),
                "column_label": str(value.get("column_label", "")),
                "unit": str(value.get("unit", "")),
                "unit_kind": str(value.get("unit_kind", "")),
                "document_ids": set(),
                "years": set(),
                "month_count_by_year": {},
                "value_count": 0,
                "provenance_refs": [],
                "_month_sets": {},
            },
        )
        series["value_count"] += 1
        if document_id:
            series["document_ids"].add(document_id)
            all_document_ids.add(document_id)
        for year in years:
            series["years"].add(year)
            all_years.add(year)
            if month_index is not None:
                series["_month_sets"].setdefault(year, set()).add(month_index)
        unit_text = str(value.get("unit", "")).strip().lower()
        if series["unit"] and unit_text and unit_text != str(series["unit"]).strip().lower():
            unit_consistent = False
        series["provenance_refs"].append(
            {
                "document_id": document_id,
                "citation": str(value.get("citation", "")),
                "page_locator": str(value.get("page_locator", "")),
                "table_locator": str(value.get("table_locator", "")),
                "row_label": str(value.get("row_label", "")),
                "column_label": str(value.get("column_label", "")),
                "raw_value": str(value.get("raw_value", "")),
            }
        )

    merged_series: list[dict[str, Any]] = []
    cross_document_series_count = 0
    aligned_document_count = 0
    for raw in grouped.values():
        document_ids = sorted(raw["document_ids"])
        years = sorted(raw["years"])
        month_count_by_year = {
            str(year): len(months)
            for year, months in sorted(raw["_month_sets"].items())
        }
        merged = DocumentMergedSeriesEvidence(
            series_key=str(raw["series_key"]),
            row_label=str(raw["row_label"]),
            column_label=str(raw["column_label"]),
            unit=str(raw["unit"]),
            unit_kind=str(raw["unit_kind"]),
            document_ids=document_ids,
            years=years,
            month_count_by_year=month_count_by_year,
            value_count=int(raw["value_count"]),
            provenance_refs=list(raw["provenance_refs"])[:24],
        )
        if len(document_ids) >= 2:
            cross_document_series_count += 1
            aligned_document_count = max(aligned_document_count, len(document_ids))
        merged_series.append(merged.model_dump())

    alignment_summary = {
        "document_count": len(all_document_ids),
        "aligned_document_count": aligned_document_count,
        "cross_document_series_count": cross_document_series_count,
        "aligned_years": sorted(all_years),
        "unit_consistent": unit_consistent,
        "series_count": len(merged_series),
    }
    return merged_series, alignment_summary


def _page_locator(chunk: dict[str, Any], metadata: dict[str, Any]) -> str:
    locator = str(chunk.get("locator", "")).strip()
    if locator:
        return locator
    page_start = metadata.get("page_start")
    if isinstance(page_start, int):
        return f"page {page_start + 1}"
    return ""


def _table_page_locator(table: dict[str, Any], chunks: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    locator = str(table.get("page_locator", "")).strip()
    if locator:
        return locator
    for chunk in chunks:
        page_locator = _page_locator(chunk, metadata)
        if page_locator:
            return page_locator
    return ""


def build_officeqa_structured_evidence(tool_results: list[dict[str, Any]] | None) -> dict[str, Any]:
    merged_records: list[dict[str, Any]] = []
    for raw in tool_results or []:
        record_additions = document_records_from_tool_result(raw)
        if record_additions:
            merged_records = merge_document_evidence_records(merged_records, record_additions)

    document_records = [DocumentEvidenceRecord.model_validate(item) for item in merged_records]
    tables: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    page_chunks: list[dict[str, Any]] = []
    units_seen: set[str] = set()
    provenance_complete = True

    for record in document_records:
        metadata = dict(record.metadata or {})
        for chunk in record.chunks:
            page_locator = _page_locator(chunk, metadata)
            page_chunks.append(
                {
                    "document_id": record.document_id,
                    "citation": record.citation,
                    "page_locator": page_locator,
                    "locator": str(chunk.get("locator", "")),
                    "kind": str(chunk.get("kind", "")),
                    "text": str(chunk.get("text", "")),
                }
            )
        for table in record.tables:
            headers = [str(item) for item in table.get("headers", [])]
            rows = [list(row) for row in table.get("rows", []) if isinstance(row, list)]
            unit_hint = str(table.get("unit_hint", "") or metadata.get("unit_hint", "") or "")
            table_locator = str(table.get("locator", "")).strip()
            page_locator = _table_page_locator(table, record.chunks, metadata)
            unit, multiplier, unit_kind = _unit_profile(unit_hint, " ".join(headers))
            if unit:
                units_seen.add(unit)
            table_record = OfficeQATableEvidence(
                document_id=record.document_id,
                citation=str(table.get("citation", "") or record.citation),
                page_locator=page_locator,
                table_locator=table_locator,
                headers=headers,
                unit=unit,
                unit_multiplier=multiplier,
                unit_kind=unit_kind,
                row_count=len(rows),
                column_count=len(headers),
            )
            tables.append(table_record.model_dump())
            for row_index, row in enumerate(rows):
                row_label = str(row[0]).strip() if row else ""
                for column_index, raw_value in enumerate(row):
                    column_label = headers[column_index] if column_index < len(headers) else f"column_{column_index + 1}"
                    unit_value, unit_multiplier, unit_kind_value = _unit_profile(unit_hint, column_label)
                    numeric_value = _numeric_value(raw_value)
                    normalized_value = numeric_value * unit_multiplier if numeric_value is not None else None
                    value_record = OfficeQAValueEvidence(
                        document_id=record.document_id,
                        citation=str(table.get("citation", "") or record.citation),
                        page_locator=page_locator,
                        table_locator=table_locator,
                        row_index=row_index,
                        row_label=row_label,
                        column_index=column_index,
                        column_label=column_label,
                        raw_value=str(raw_value),
                        numeric_value=numeric_value,
                        normalized_value=normalized_value,
                        unit=unit_value,
                        unit_multiplier=unit_multiplier,
                        unit_kind=unit_kind_value,
                    )
                    if not value_record.document_id or not value_record.citation or not value_record.table_locator or value_record.column_index < 0:
                        provenance_complete = False
                    values.append(value_record.model_dump())

    merged_series, alignment_summary = _merged_series(values)

    payload = OfficeQAStructuredEvidence(
        document_evidence=[record.model_dump() for record in document_records],
        tables=tables,
        values=values,
        page_chunks=page_chunks,
        merged_series=merged_series,
        alignment_summary=alignment_summary,
        units_seen=sorted(units_seen),
        value_count=len(values),
        provenance_complete=provenance_complete and bool(values or tables),
    )
    return payload.model_dump()


def compact_officeqa_structured_evidence(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(payload or {})
    if not data:
        return {}
    return {
        "table_count": len(list(data.get("tables", []))),
        "value_count": int(data.get("value_count", 0) or 0),
        "units_seen": list(data.get("units_seen", []))[:8],
        "provenance_complete": bool(data.get("provenance_complete")),
        "alignment_summary": {
            "document_count": int(dict(data.get("alignment_summary", {})).get("document_count", 0) or 0),
            "aligned_document_count": int(dict(data.get("alignment_summary", {})).get("aligned_document_count", 0) or 0),
            "cross_document_series_count": int(dict(data.get("alignment_summary", {})).get("cross_document_series_count", 0) or 0),
            "aligned_years": list(dict(data.get("alignment_summary", {})).get("aligned_years", []))[:8],
            "unit_consistent": bool(dict(data.get("alignment_summary", {})).get("unit_consistent", True)),
        },
        "tables": [
            {
                "document_id": item.get("document_id", ""),
                "page_locator": item.get("page_locator", ""),
                "table_locator": item.get("table_locator", ""),
                "headers": list(item.get("headers", []))[:6],
                "unit": item.get("unit", ""),
                "row_count": item.get("row_count", 0),
                "column_count": item.get("column_count", 0),
            }
            for item in list(data.get("tables", []))[:4]
            if isinstance(item, dict)
        ],
        "values": [
            {
                "document_id": item.get("document_id", ""),
                "page_locator": item.get("page_locator", ""),
                "table_locator": item.get("table_locator", ""),
                "row_label": item.get("row_label", ""),
                "column_label": item.get("column_label", ""),
                "raw_value": item.get("raw_value", ""),
                "normalized_value": item.get("normalized_value"),
                "unit": item.get("unit", ""),
            }
            for item in list(data.get("values", []))[:12]
            if isinstance(item, dict)
        ],
        "merged_series": [
            {
                "series_key": item.get("series_key", ""),
                "document_ids": list(item.get("document_ids", []))[:6],
                "years": list(item.get("years", []))[:6],
                "value_count": item.get("value_count", 0),
                "month_count_by_year": dict(item.get("month_count_by_year", {})),
            }
            for item in list(data.get("merged_series", []))[:6]
            if isinstance(item, dict)
        ],
    }
