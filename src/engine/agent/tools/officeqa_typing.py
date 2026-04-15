from __future__ import annotations

import re
from typing import Any

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_MONTH_TOKENS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _financial_metric_group(text: str) -> str:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("debt", "liabilities", "assets", "obligations", "securities", "balance sheet")):
        return "debt"
    if any(token in lowered for token in ("receipts", "revenue", "collections", "income")):
        return "revenue"
    if any(token in lowered for token in ("expenditures", "outlays", "disbursements", "spending", "expenses")):
        return "expenditures"
    return ""


def _table_text_parts(
    *,
    text: str = "",
    headers: list[str] | None = None,
    row_labels: list[str] | None = None,
    row_path_samples: list[str] | None = None,
    rows: list[list[str]] | None = None,
    heading_chain: list[str] | None = None,
) -> tuple[str, str, str, str, list[list[str]]]:
    header_values = [_normalize_space(item) for item in list(headers or []) if _normalize_space(item)]
    row_label_values = [_normalize_space(item) for item in list(row_labels or []) if _normalize_space(item)]
    row_path_values = [_normalize_space(item) for item in list(row_path_samples or []) if _normalize_space(item)]
    row_values = [list(row) for row in list(rows or []) if isinstance(row, list)]
    heading_values = [_normalize_space(item) for item in list(heading_chain or []) if _normalize_space(item)]
    full_text = " ".join(
        part
        for part in (
            _normalize_space(text),
            " ".join(heading_values),
            " ".join(header_values),
            " ".join(row_label_values),
            " ".join(row_path_values),
            " ".join(" ".join(_normalize_space(cell) for cell in row[:8]) for row in row_values[:30]),
        )
        if part
    )
    return full_text.lower(), " ".join(header_values).lower(), " ".join(row_label_values).lower(), " ".join(row_path_values).lower(), row_values


def infer_table_period_type(
    *,
    text: str = "",
    headers: list[str] | None = None,
    row_labels: list[str] | None = None,
    row_path_samples: list[str] | None = None,
    rows: list[list[str]] | None = None,
    heading_chain: list[str] | None = None,
    month_coverage: list[str] | None = None,
    years: list[str] | None = None,
    family: str = "",
    declared_period_type: str = "",
) -> str:
    explicit = _normalize_space(declared_period_type).lower()
    if explicit:
        return explicit
    if family == "monthly_series":
        return "monthly_series"
    if family == "fiscal_year_comparison":
        return "fiscal_year"
    lowered, headers_text, row_text, row_path_text, _rows = _table_text_parts(
        text=text,
        headers=headers,
        row_labels=row_labels,
        row_path_samples=row_path_samples,
        rows=rows,
        heading_chain=heading_chain,
    )
    months = list(month_coverage or [])
    if not months:
        months = [month for month in _MONTH_TOKENS if month in lowered or month in headers_text or month in row_text or month in row_path_text]
    years = list(years or _YEAR_RE.findall(lowered)[:12])
    if (
        len(months) >= 4
        or "monthly series" in lowered
        or "receipts expenditures and balances" in lowered
        or "month" in headers_text
        or (len(months) >= 2 and ("month" in row_text or "month" in row_path_text))
    ):
        return "monthly_series"
    if "fiscal year" in lowered or re.search(r"\bfy\s+\d{4}\b", lowered) or "end of fiscal years" in lowered:
        return "fiscal_year"
    if "calendar year" in lowered or "actual 6 months" in lowered or "estimate" in lowered or (years and "summary" in lowered):
        return "calendar_year"
    if "discussion" in lowered or "statement" in lowered or "commentary" in lowered:
        return "narrative_support"
    return "point_lookup"


def classify_table_typing(
    *,
    text: str = "",
    headers: list[str] | None = None,
    row_labels: list[str] | None = None,
    row_path_samples: list[str] | None = None,
    rows: list[list[str]] | None = None,
    heading_chain: list[str] | None = None,
    month_coverage: list[str] | None = None,
    years: list[str] | None = None,
    declared_family: str = "",
    declared_period_type: str = "",
) -> dict[str, Any]:
    lowered, headers_text, row_text, row_path_text, row_values = _table_text_parts(
        text=text,
        headers=headers,
        row_labels=row_labels,
        row_path_samples=row_path_samples,
        rows=rows,
        heading_chain=heading_chain,
    )
    month_hits = list(month_coverage or [])
    if not month_hits:
        month_hits = [month for month in _MONTH_TOKENS if month in lowered or month in headers_text or month in row_text or month in row_path_text]
    year_refs = list(years or list(dict.fromkeys(_YEAR_RE.findall(lowered)[:12])))
    row_labels_all = " ".join(part for part in (row_text, row_path_text) if part).strip()
    metric_group = _financial_metric_group(lowered)
    data_row_count = sum(1 for row in row_values[:40] if any(_normalize_space(cell) for cell in row))
    numeric_row_count = sum(1 for row in row_values[:20] if any(re.search(r"\d", _normalize_space(cell)) for cell in row[1:]))
    year_header_hits = sum(1 for header in list(headers or []) if _YEAR_RE.search(str(header or "")))
    total_like_headers = sum(
        1
        for header in list(headers or [])
        if any(token in str(header or "").lower() for token in ("total", "actual", "estimate", "calendar year", "fiscal year"))
    )

    family = "generic_financial_table"
    confidence = 0.45
    if any(token in lowered for token in ("table of contents", "issue and page number", "contents", "page number", "index")):
        family, confidence = "navigation_or_contents", 0.98
    elif any(token in lowered for token in ("public debt", "debt outstanding", "guaranteed obligations", "assets", "liabilities", "securities", "balance sheet")):
        family, confidence = "debt_or_balance_sheet", 0.9
    elif "fiscal year" in lowered or re.search(r"\bfy\s+\d{4}\b", lowered) or "end of fiscal years" in lowered:
        family, confidence = "fiscal_year_comparison", 0.88
    elif len(month_hits) >= 4 or "month" in headers_text or (len(month_hits) >= 2 and row_labels_all):
        family, confidence = "monthly_series", min(0.95, 0.55 + 0.04 * max(len(month_hits), data_row_count))
    elif (year_header_hits >= 1 or year_refs) and data_row_count >= 2 and numeric_row_count >= 1 and metric_group in {"expenditures", "revenue"}:
        family, confidence = "category_breakdown", 0.78
    elif (year_header_hits >= 1 or year_refs) and total_like_headers >= 1 and data_row_count <= 3:
        family, confidence = "annual_summary", 0.72
    elif (year_header_hits >= 1 or year_refs) and metric_group in {"expenditures", "revenue"} and row_labels_all:
        family, confidence = "category_breakdown", 0.64

    period_type = infer_table_period_type(
        text=text,
        headers=headers,
        row_labels=row_labels,
        row_path_samples=row_path_samples,
        rows=rows,
        heading_chain=heading_chain,
        month_coverage=month_hits,
        years=year_refs,
        family=family,
        declared_period_type=declared_period_type,
    )

    ambiguities: list[str] = []
    declared_family_norm = _normalize_space(declared_family).lower()
    if declared_family_norm and declared_family_norm != family:
        ambiguities.append(f"family_drift:{declared_family_norm}->{family}")
    declared_period_norm = _normalize_space(declared_period_type).lower()
    if declared_period_norm and declared_period_norm != period_type:
        ambiguities.append(f"period_drift:{declared_period_norm}->{period_type}")

    return {
        "table_family": family,
        "table_family_confidence": round(max(0.05, min(1.0, confidence)), 4),
        "period_type": period_type,
        "metric_group": metric_group,
        "year_refs": year_refs,
        "month_coverage": month_hits,
        "typing_ambiguities": ambiguities,
    }
