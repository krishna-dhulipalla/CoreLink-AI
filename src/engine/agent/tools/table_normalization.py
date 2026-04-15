"""Canonical Treasury-style table normalization.

The goal of this module is not OCR-grade table structure recognition. It takes
already parsed table payloads and converts them into a stable internal table
representation that preserves:

- multi-row column headers
- row-header depth
- structural row types
- spanning-cell relationships
- resolved row/column paths for data cells
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_NUMERIC_RE = re.compile(r"^[+-]?\$?\d[\d,]*(?:\.\d+)?%?$")
_TOTAL_TOKEN_RE = re.compile(r"\b(total|subtotal|net|grand total)\b", re.IGNORECASE)
_UNIT_TOKEN_RE = re.compile(r"\b(percent|million|billion|thousand|dollar|dollars|cents)\b", re.IGNORECASE)


def _clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _raw_text(text: Any) -> str:
    return str(text or "").replace("\xa0", " ")


def _normalized_text(text: Any) -> str:
    return _clean_text(text).lower()


def _is_numeric_text(text: Any) -> bool:
    compact = _clean_text(text).replace(" ", "")
    if not compact:
        return False
    return bool(_NUMERIC_RE.fullmatch(compact))


def _leading_indent_units(text: Any) -> int:
    raw = _raw_text(text)
    if not raw:
        return 0
    match = re.match(r"^[ \t]+", raw)
    if not match:
        return 0
    width = len(match.group(0).replace("\t", "    "))
    return min(3, max(0, width // 2))


def _leading_empty_cells(row: list[dict[str, Any]], limit: int) -> int:
    count = 0
    for cell in row[:limit]:
        if _clean_text(cell.get("text", "")):
            break
        count += 1
    return count


def _first_numeric_index(texts: list[str]) -> int | None:
    for idx, text in enumerate(texts):
        if _is_numeric_text(text):
            return idx
    return None


def _dedupe_path(parts: list[str]) -> list[str]:
    deduped: list[str] = []
    for part in parts:
        cleaned = _clean_text(part)
        if not cleaned:
            continue
        if deduped and _normalized_text(deduped[-1]) == _normalized_text(cleaned):
            continue
        deduped.append(cleaned)
    return deduped


def _build_leaf_label(path: list[str], fallback: str) -> str:
    cleaned = _dedupe_path(path)
    if not cleaned:
        return fallback
    if len(cleaned) == 1:
        return cleaned[0]
    return " | ".join(cleaned)


def _row_type_for_values(values: list[str]) -> str:
    nonempty = [_clean_text(value) for value in values if _clean_text(value)]
    if not nonempty:
        return "empty"
    if len(nonempty) == 1 and not _is_numeric_text(nonempty[0]):
        return "section_divider"
    first = nonempty[0]
    numeric_count = sum(1 for value in nonempty if _is_numeric_text(value))
    if _TOTAL_TOKEN_RE.search(first):
        return "subtotal" if numeric_count else "section_divider"
    if numeric_count == 0:
        return "section_divider"
    return "data"


def _header_metrics(header_rows: list[list[dict[str, Any]]], data_rows: list[list[dict[str, Any]]], unit_hint: str) -> dict[str, Any]:
    before_tokens = [
        _normalized_text(cell.get("text", ""))
        for row in header_rows
        for cell in row
        if _clean_text(cell.get("text", ""))
    ]
    after_tokens = [
        _normalized_text(cell.get("text", ""))
        for row in header_rows
        for cell in row
        if _clean_text(cell.get("text", "")) and not cell.get("is_repeated_header")
    ]
    before_unique = len(set(before_tokens))
    after_unique = len(set(after_tokens))
    duplicate_collapse_score = 1.0
    if before_unique:
        duplicate_collapse_score = max(0.0, min(1.0, after_unique / before_unique))

    header_numeric = sum(1 for row in header_rows for cell in row if _is_numeric_text(cell.get("text", "")))
    header_total = max(1, sum(1 for row in header_rows for cell in row if _clean_text(cell.get("text", ""))))
    data_numeric = sum(1 for row in data_rows for cell in row if _is_numeric_text(cell.get("text", "")))
    data_total = max(1, sum(1 for row in data_rows for cell in row if _clean_text(cell.get("text", ""))))
    header_data_separation_quality = max(
        0.0,
        min(1.0, (1.0 - (header_numeric / header_total)) * (data_numeric / data_total if data_total else 0.0)),
    )

    recovered_unit_coverage = 1.0 if _UNIT_TOKEN_RE.search(unit_hint or "") else 0.0
    if not recovered_unit_coverage:
        recovered_unit_coverage = 1.0 if any(_UNIT_TOKEN_RE.search(cell.get("text", "")) for row in header_rows for cell in row) else 0.0

    return {
        "duplicate_header_collapse_score": round(duplicate_collapse_score, 4),
        "header_data_separation_quality": round(header_data_separation_quality, 4),
        "span_consistency": 1.0,
        "recovered_unit_coverage": recovered_unit_coverage,
    }


def _infer_header_row_count(grid: list[list[dict[str, Any]]]) -> int:
    header_rows = 0
    for row in grid[:6]:
        visible = [cell for cell in row if _clean_text(cell.get("text", ""))]
        if not visible:
            header_rows += 1
            continue
        header_ratio = sum(1 for cell in visible if cell.get("is_header")) / max(1, len(visible))
        numeric_ratio = sum(1 for cell in visible if _is_numeric_text(cell.get("text", ""))) / max(1, len(visible))
        year_ratio = sum(1 for cell in visible if _YEAR_RE.search(_clean_text(cell.get("text", "")))) / max(1, len(visible))
        if header_rows == 0 and (header_ratio >= 0.25 or numeric_ratio <= 0.2 or year_ratio >= 0.3):
            header_rows += 1
            continue
        if header_ratio >= 0.4:
            header_rows += 1
            continue
        if header_rows > 0 and numeric_ratio <= 0.25 and year_ratio >= 0.2:
            header_rows += 1
            continue
        break
    return max(1, header_rows)


def _header_alignment_depth(header_rows: list[list[dict[str, Any]]], column_count: int) -> int:
    candidates: list[int] = []
    for row in header_rows[-3:]:
        texts = [_clean_text(cell.get("text", "")) for cell in row[:column_count]]
        if not any(texts):
            continue
        leading_empty = 0
        for text in texts:
            if text:
                break
            leading_empty += 1
        if 0 < leading_empty < column_count:
            candidates.append(min(3, leading_empty))
    if not candidates:
        return 0
    return Counter(candidates).most_common(1)[0][0]


def _descriptive_row_depth_candidate(row: list[dict[str, Any]], column_count: int) -> int:
    texts = [_clean_text(cell.get("text", "")) for cell in row[:column_count]]
    if not any(texts):
        return 0
    if _first_numeric_index(texts) == 0:
        return 0
    first_nonempty = next((idx for idx, text in enumerate(texts) if text), None)
    if first_nonempty is None:
        return 0
    populated_positions = [idx for idx, text in enumerate(texts[: min(column_count, 4)]) if text]
    if len(populated_positions) >= 2 and all(not _is_numeric_text(texts[idx]) for idx in populated_positions):
        return min(3, populated_positions[-1] + 1)
    if first_nonempty > 0:
        return min(3, first_nonempty + 1)
    return 0


def _infer_row_header_depth(
    data_rows: list[list[dict[str, Any]]],
    header_rows: list[list[dict[str, Any]]],
    column_count: int,
) -> int:
    if column_count < 2:
        return 0
    vote_counter: Counter[int] = Counter()
    for row in data_rows[:160]:
        texts = [_clean_text(cell.get("text", "")) for cell in row[:column_count]]
        if not any(texts):
            continue
        first_numeric = _first_numeric_index(texts)
        if first_numeric is not None and first_numeric > 0:
            vote_counter[min(3, first_numeric)] += 3
            continue
        descriptive_depth = _descriptive_row_depth_candidate(row, column_count)
        if descriptive_depth > 0:
            vote_counter[min(3, descriptive_depth)] += 1

    aligned_depth = _header_alignment_depth(header_rows, column_count)
    if aligned_depth > 0:
        vote_counter[min(3, aligned_depth)] += 2

    if not vote_counter:
        return 1
    best_depth, _ = max(vote_counter.items(), key=lambda item: (item[1], item[0]))
    return max(1, min(3, best_depth))


def _column_paths(header_rows: list[list[dict[str, Any]]], column_count: int) -> list[list[str]]:
    paths: list[list[str]] = []
    for col_idx in range(column_count):
        parts: list[str] = []
        seen_origins: set[str] = set()
        for row in header_rows:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            origin_id = str(cell.get("origin_id", ""))
            text = _clean_text(cell.get("text", ""))
            if origin_id and origin_id in seen_origins:
                cell["is_repeated_header"] = True
                continue
            if origin_id:
                seen_origins.add(origin_id)
            if text:
                parts.append(text)
        paths.append(_dedupe_path(parts))
    return paths


def _row_records(
    data_rows: list[list[dict[str, Any]]],
    *,
    row_header_depth: int,
    column_paths: list[list[str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    section_stack: list[str] = []
    hierarchy_cap = max(1, row_header_depth + 1)
    for row_index, row in enumerate(data_rows):
        values = [_clean_text(cell.get("text", "")) for cell in row]
        row_type = _row_type_for_values(values)
        if row_type == "empty":
            continue

        header_cells = row[:row_header_depth] if row_header_depth > 0 else row[:1]
        leading_empty_cells = _leading_empty_cells(header_cells, len(header_cells))
        first_nonempty_header_index = next(
            (idx for idx, cell in enumerate(header_cells) if _clean_text(cell.get("text", ""))),
            None,
        )
        indentation_depth = (
            _leading_indent_units(header_cells[first_nonempty_header_index].get("text", ""))
            if first_nonempty_header_index is not None
            else 0
        )
        local_depth = min(hierarchy_cap, leading_empty_cells + indentation_depth)
        label_parts = _dedupe_path(values[leading_empty_cells:row_header_depth])
        if not label_parts and row_header_depth > 0:
            label_parts = _dedupe_path(values[:row_header_depth])
        if row_type == "section_divider":
            section_stack = _dedupe_path([*section_stack[:local_depth], *label_parts])
            records.append(
                {
                    "row_index": row_index,
                    "row_type": row_type,
                    "row_path": list(section_stack),
                    "row_depth": local_depth,
                    "leading_empty_cells": leading_empty_cells,
                    "indentation_depth": indentation_depth,
                    "cells": [],
                }
            )
            continue

        row_path = _dedupe_path([*section_stack[:local_depth], *label_parts])
        cells: list[dict[str, Any]] = []
        for col_idx in range(row_header_depth, min(len(row), len(column_paths))):
            cell_text = _clean_text(row[col_idx].get("text", ""))
            if not cell_text:
                continue
            column_path = _dedupe_path(column_paths[col_idx])
            cells.append(
                {
                    "column_index": col_idx,
                    "column_path": column_path,
                    "column_label": _build_leaf_label(column_path, f"column {col_idx + 1}"),
                    "value": cell_text,
                    "is_numeric": _is_numeric_text(cell_text),
                    "origin_id": row[col_idx].get("origin_id", ""),
                }
            )
        records.append(
            {
                "row_index": row_index,
                "row_type": row_type,
                "row_path": row_path,
                "row_label": _build_leaf_label(row_path, f"row {row_index + 1}"),
                "row_depth": local_depth,
                "leading_empty_cells": leading_empty_cells,
                "indentation_depth": indentation_depth,
                "cells": cells,
            }
        )
        section_stack = list(row_path)
    return records


def _display_projection(row_records: list[dict[str, Any]], column_paths: list[list[str]], row_header_depth: int) -> tuple[list[str], list[list[str]]]:
    data_column_indexes = [idx for idx in range(row_header_depth, len(column_paths))]
    headers = ["Row"] + [_build_leaf_label(column_paths[idx], f"column {idx + 1}") for idx in data_column_indexes]
    rows: list[list[str]] = []
    for record in row_records:
        if record.get("row_type") == "section_divider":
            continue
        value_map = {int(cell.get("column_index", -1)): str(cell.get("value", "")) for cell in record.get("cells", [])}
        row_label = str(record.get("row_label") or _build_leaf_label(list(record.get("row_path", [])), ""))
        rows.append([row_label] + [value_map.get(idx, "") for idx in data_column_indexes])
    return headers, rows


def normalize_dense_table_grid(
    grid: list[list[dict[str, Any]]],
    *,
    locator: str = "",
    page_locator: str = "",
    unit_hint: str = "",
    context_text: str = "",
) -> dict[str, Any]:
    if not grid:
        return {
            "header_rows": [],
            "column_paths": [],
            "row_records": [],
            "normalization_metrics": {
                "duplicate_header_collapse_score": 0.0,
                "header_data_separation_quality": 0.0,
                "span_consistency": 0.0,
                "recovered_unit_coverage": 0.0,
            },
            "row_header_depth": 0,
            "column_count": 0,
            "data_row_count": 0,
            "locator": locator,
            "page_locator": page_locator,
            "unit_hint": unit_hint,
            "context_text": context_text,
        }

    column_count = max(len(row) for row in grid)
    padded_grid: list[list[dict[str, Any]]] = []
    for row in grid:
        padded = list(row)
        while len(padded) < column_count:
            padded.append({"text": "", "is_header": False, "origin_id": ""})
        padded_grid.append(padded)

    header_row_count = min(len(padded_grid), _infer_header_row_count(padded_grid))
    header_rows = padded_grid[:header_row_count]
    raw_data_rows = padded_grid[header_row_count:]
    row_header_depth = _infer_row_header_depth(raw_data_rows, header_rows, column_count)
    column_paths = _column_paths(header_rows, column_count)
    row_records = _row_records(raw_data_rows, row_header_depth=row_header_depth, column_paths=column_paths)
    metrics = _header_metrics(header_rows, raw_data_rows, unit_hint)
    display_headers, display_rows = _display_projection(row_records, column_paths, row_header_depth)

    return {
        "locator": locator,
        "page_locator": page_locator,
        "unit_hint": unit_hint,
        "context_text": context_text,
        "column_count": column_count,
        "header_row_count": header_row_count,
        "row_header_depth": row_header_depth,
        "header_rows": [[_clean_text(cell.get("text", "")) for cell in row] for row in header_rows],
        "column_paths": column_paths,
        "row_records": row_records,
        "data_row_count": len([record for record in row_records if record.get("row_type") != "section_divider"]),
        "normalization_metrics": metrics,
        "display_headers": display_headers,
        "display_rows": display_rows,
    }


def normalize_flat_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    locator: str = "",
    page_locator: str = "",
    unit_hint: str = "",
    context_text: str = "",
) -> dict[str, Any]:
    grid: list[list[dict[str, Any]]] = []
    if headers:
        grid.append(
            [
                {"text": _raw_text(header), "is_header": True, "origin_id": f"h0_{idx}"}
                for idx, header in enumerate(headers)
            ]
        )
    for row_idx, row in enumerate(rows):
        grid.append(
            [
                {"text": _raw_text(cell), "is_header": False, "origin_id": f"r{row_idx}_{col_idx}"}
                for col_idx, cell in enumerate(row)
            ]
        )
    return normalize_dense_table_grid(
        grid,
        locator=locator,
        page_locator=page_locator,
        unit_hint=unit_hint,
        context_text=context_text,
    )
