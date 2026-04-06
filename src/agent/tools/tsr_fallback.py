from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from agent.tools.table_normalization import normalize_dense_table_grid, normalize_flat_table


def experimental_tsr_fallback_enabled() -> bool:
    return os.getenv("OFFICEQA_ENABLE_TSR_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _looks_numeric(text: Any) -> bool:
    compact = _clean_text(text).replace(",", "").replace("%", "")
    if not compact:
        return False
    try:
        float(compact)
        return True
    except ValueError:
        return False


def _infer_header_rows(grid: list[list[dict[str, Any]]]) -> int:
    header_rows = 0
    for row in grid[:6]:
        visible = [_clean_text(cell.get("text", "")) for cell in row if _clean_text(cell.get("text", ""))]
        if not visible:
            header_rows += 1
            continue
        numeric_ratio = sum(1 for value in visible if _looks_numeric(value)) / max(1, len(visible))
        if header_rows == 0 or numeric_ratio <= 0.25:
            header_rows += 1
            continue
        break
    return max(1, min(header_rows, len(grid)))


def _fill_header_blanks(grid: list[list[dict[str, Any]]], header_rows: int) -> list[list[dict[str, Any]]]:
    enriched = deepcopy(grid)
    for row_index in range(min(header_rows, len(enriched))):
        row = enriched[row_index]
        left_text = ""
        for col_index, cell in enumerate(row):
            text = _clean_text(cell.get("text", ""))
            if not text:
                upper_text = ""
                if row_index > 0 and col_index < len(enriched[row_index - 1]):
                    upper_text = _clean_text(enriched[row_index - 1][col_index].get("text", ""))
                replacement = left_text or upper_text
                if replacement:
                    cell["text"] = replacement
                    cell["is_header"] = True
                    if not str(cell.get("origin_id", "")).strip():
                        cell["origin_id"] = f"tsr_fill_r{row_index}_c{col_index}"
            else:
                left_text = text
    return enriched


def _variant_score(canonical_table: dict[str, Any]) -> float:
    metrics = dict(canonical_table.get("normalization_metrics", {}) or {})
    duplicate_score = float(metrics.get("duplicate_header_collapse_score", 0.0) or 0.0)
    separation = float(metrics.get("header_data_separation_quality", 0.0) or 0.0)
    span_consistency = float(metrics.get("span_consistency", 0.0) or 0.0)
    unit_coverage = float(metrics.get("recovered_unit_coverage", 0.0) or 0.0)
    row_records = list(canonical_table.get("row_records", []) or [])
    data_rows = [row for row in row_records if str(row.get("row_type", "")) != "section_divider"]
    column_paths = list(canonical_table.get("column_paths", []) or [])
    path_richness = sum(len([part for part in path if _clean_text(part)]) for path in column_paths) / max(1, len(column_paths))
    numeric_cell_count = sum(
        1
        for row in data_rows
        for cell in list(row.get("cells", []) or [])
        if cell.get("is_numeric")
    )
    return round(
        duplicate_score * 0.2
        + separation * 0.2
        + span_consistency * 0.2
        + unit_coverage * 0.1
        + min(1.0, path_richness / 2.0) * 0.15
        + min(1.0, numeric_cell_count / 12.0) * 0.15,
        4,
    )


def compare_dense_table_normalizers(
    grid: list[list[dict[str, Any]]],
    *,
    locator: str = "",
    page_locator: str = "",
    unit_hint: str = "",
    context_text: str = "",
) -> dict[str, Any]:
    default = normalize_dense_table_grid(
        grid,
        locator=locator,
        page_locator=page_locator,
        unit_hint=unit_hint,
        context_text=context_text,
    )
    header_rows = _infer_header_rows(grid)
    split_merge_grid = _fill_header_blanks(grid, header_rows)
    fallback = normalize_dense_table_grid(
        split_merge_grid,
        locator=locator,
        page_locator=page_locator,
        unit_hint=unit_hint,
        context_text=context_text,
    )
    default_score = _variant_score(default)
    fallback_score = _variant_score(fallback)
    selected_mode = "default"
    selected = default
    if fallback_score > default_score + 0.04:
        selected_mode = "tsr_split_merge_heuristic"
        selected = fallback
    diagnostics = {
        "enabled": experimental_tsr_fallback_enabled(),
        "selected_mode": selected_mode,
        "default_score": default_score,
        "fallback_score": fallback_score,
        "score_delta": round(fallback_score - default_score, 4),
        "header_rows_considered": header_rows,
    }
    return {
        "selected": selected,
        "default": default,
        "fallback": fallback,
        "diagnostics": diagnostics,
    }


def select_dense_table_normalization(
    grid: list[list[dict[str, Any]]],
    *,
    locator: str = "",
    page_locator: str = "",
    unit_hint: str = "",
    context_text: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparison = compare_dense_table_normalizers(
        grid,
        locator=locator,
        page_locator=page_locator,
        unit_hint=unit_hint,
        context_text=context_text,
    )
    selected = comparison["default"]
    diagnostics = dict(comparison["diagnostics"] or {})
    default_metrics = dict(comparison["default"].get("normalization_metrics", {}) or {})
    default_separation = float(default_metrics.get("header_data_separation_quality", 0.0) or 0.0)
    default_score = float(diagnostics.get("default_score", 0.0) or 0.0)
    fallback_score = float(diagnostics.get("fallback_score", 0.0) or 0.0)
    auto_promote = (
        fallback_score > default_score + 0.04
        or (
            default_separation < 0.5
            and fallback_score >= default_score + 0.01
        )
        or (
            default_score < 0.58
            and fallback_score >= default_score
        )
    )
    diagnostics["auto_promoted"] = auto_promote
    if experimental_tsr_fallback_enabled() or auto_promote:
        selected = comparison["selected"]
        diagnostics["selection_mode"] = "fallback_selected"
    else:
        diagnostics["selection_mode"] = "default_selected"
    selected = dict(selected or {})
    selected["experimental_tsr"] = diagnostics
    return selected, diagnostics


def compare_flat_table_normalizers(
    headers: list[str],
    rows: list[list[str]],
    *,
    locator: str = "",
    page_locator: str = "",
    unit_hint: str = "",
    context_text: str = "",
) -> dict[str, Any]:
    default = normalize_flat_table(
        headers,
        rows,
        locator=locator,
        page_locator=page_locator,
        unit_hint=unit_hint,
        context_text=context_text,
    )
    grid: list[list[dict[str, Any]]] = []
    if headers:
        grid.append([{"text": str(header), "is_header": True, "origin_id": f"h_{idx}"} for idx, header in enumerate(headers)])
    for row_index, row in enumerate(rows):
        grid.append([{"text": str(cell), "is_header": False, "origin_id": f"r{row_index}_{col_index}"} for col_index, cell in enumerate(row)])
    compared = compare_dense_table_normalizers(
        grid,
        locator=locator,
        page_locator=page_locator,
        unit_hint=unit_hint,
        context_text=context_text,
    )
    compared["default"] = default
    return compared
