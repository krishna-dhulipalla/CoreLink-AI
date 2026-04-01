"""Deterministic OfficeQA compute over structured table evidence."""

from __future__ import annotations

import re
from typing import Any

from agent.contracts import OfficeQAComputeResult, OfficeQAComputeStep, RetrievalIntent
from agent.solver.common import format_scalar_number

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
_STOPWORDS = {
    "absolute",
    "adjusted",
    "all",
    "annual",
    "average",
    "calendar",
    "change",
    "difference",
    "fiscal",
    "for",
    "individual",
    "months",
    "nearest",
    "only",
    "percent",
    "reported",
    "round",
    "rounded",
    "sum",
    "the",
    "these",
    "total",
    "using",
    "value",
    "values",
    "what",
    "year",
}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_years(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text or "")))


def _extract_month_index(text: str) -> int | None:
    lowered = (text or "").lower()
    for month, index in _MONTH_INDEX.items():
        if month in lowered:
            return index
    return None


def _cell_text(value: dict[str, Any]) -> str:
    return _normalize_space(
        " ".join(
            [
                str(value.get("row_label", "")),
                str(value.get("column_label", "")),
                str(value.get("table_locator", "")),
                str(value.get("page_locator", "")),
                str(value.get("citation", "")),
                str(value.get("document_id", "")),
            ]
        )
    )


def _dedupe_values(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        signature = repr(
            (
                item.get("document_id", ""),
                item.get("citation", ""),
                item.get("page_locator", ""),
                item.get("table_locator", ""),
                item.get("row_index", -1),
                item.get("row_label", ""),
                item.get("column_index", -1),
                item.get("column_label", ""),
                item.get("raw_value", ""),
            )
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def _metric_basis(task_text: str, retrieval_intent: RetrievalIntent) -> str:
    metric = (retrieval_intent.metric or "").strip().lower()
    if metric in {"absolute percent change", "absolute difference"}:
        lowered = (task_text or "").lower()
        for candidate in ("receipts", "expenditures", "outlays", "public debt outstanding", "public debt", "cpi"):
            if candidate in lowered:
                return candidate
        return "value"
    if not metric:
        return "value"
    return metric


def _metric_tokens(task_text: str, retrieval_intent: RetrievalIntent) -> set[str]:
    basis = _metric_basis(task_text, retrieval_intent)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", basis)
        if len(token) > 2 and token not in _STOPWORDS
    }


def _matches_metric(value: dict[str, Any], metric_tokens: set[str]) -> bool:
    if not metric_tokens:
        return True
    text_tokens = set(re.findall(r"[a-z0-9]+", _cell_text(value).lower()))
    return bool(metric_tokens.intersection(text_tokens))


def _is_cpi_value(value: dict[str, Any]) -> bool:
    lowered = _cell_text(value).lower()
    return any(token in lowered for token in ("cpi", "consumer price index", "inflation"))


def _extract_value_years(value: dict[str, Any]) -> list[str]:
    years = _extract_years(_cell_text(value))
    if years:
        return years
    return _extract_years(str(value.get("document_id", "")))


def _pick_numeric_value(value: dict[str, Any]) -> float | None:
    normalized = value.get("normalized_value")
    if isinstance(normalized, (int, float)):
        return float(normalized)
    numeric = value.get("numeric_value")
    if isinstance(numeric, (int, float)):
        return float(numeric)
    return None


def _provenance_ref(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": str(value.get("document_id", "")),
        "citation": str(value.get("citation", "")),
        "page_locator": str(value.get("page_locator", "")),
        "table_locator": str(value.get("table_locator", "")),
        "row_label": str(value.get("row_label", "")),
        "column_label": str(value.get("column_label", "")),
        "raw_value": str(value.get("raw_value", "")),
    }


def _format_numeric(value: float, task_text: str, *, percent: bool = False) -> str:
    lowered = (task_text or "").lower()
    if "nearest hundredth" in lowered or "nearest hundredths" in lowered:
        return f"{value:.2f}"
    if "nearest tenth" in lowered:
        return f"{value:.1f}"
    if percent:
        return f"{value:.2f}"
    return format_scalar_number(value)


def _group_values(
    values: list[dict[str, Any]],
    *,
    metric_tokens: set[str],
    include_cpi: bool = False,
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    monthly: dict[str, dict[int, dict[str, Any]]] = {}
    annual: dict[str, list[dict[str, Any]]] = {}
    for item in _dedupe_values(values):
        numeric_value = _pick_numeric_value(item)
        if numeric_value is None:
            continue
        if include_cpi != _is_cpi_value(item):
            continue
        if not _matches_metric(item, metric_tokens) and not include_cpi:
            continue
        years = _extract_value_years(item)
        if not years:
            continue
        month_index = _extract_month_index(str(item.get("row_label", "")))
        for year in years[:1]:
            if month_index is not None:
                monthly.setdefault(year, {})
                monthly[year].setdefault(month_index, item)
            else:
                annual.setdefault(year, []).append(item)
    return monthly, annual


def _series_total_for_calendar_year(
    year: str,
    *,
    monthly: dict[str, dict[int, dict[str, Any]]],
    annual: dict[str, list[dict[str, Any]]],
) -> tuple[float | None, list[dict[str, Any]], str]:
    annual_candidates = list(annual.get(year, []))
    if annual_candidates:
        annual_candidates.sort(key=lambda item: ("total" not in _cell_text(item).lower(), _cell_text(item)))
        chosen = annual_candidates[0]
        total = _pick_numeric_value(chosen)
        if total is not None:
            return total, [_provenance_ref(chosen)], "annual_row"
    monthly_values = monthly.get(year, {})
    if len(monthly_values) >= 12:
        ordered = [monthly_values[index] for index in sorted(monthly_values)[:12]]
        total = sum(_pick_numeric_value(item) or 0.0 for item in ordered)
        return total, [_provenance_ref(item) for item in ordered], "monthly_sum"
    return None, [], "missing_year_support"


def _fiscal_month_keys(target_year: int) -> list[tuple[str, int]]:
    if target_year >= 1977:
        return [(str(target_year - 1), month) for month in range(10, 13)] + [(str(target_year), month) for month in range(1, 10)]
    return [(str(target_year - 1), month) for month in range(7, 13)] + [(str(target_year), month) for month in range(1, 7)]


def _series_total_for_fiscal_year(
    year: str,
    *,
    monthly: dict[str, dict[int, dict[str, Any]]],
    annual: dict[str, list[dict[str, Any]]],
) -> tuple[float | None, list[dict[str, Any]], str]:
    annual_candidates = [
        item
        for item in annual.get(year, [])
        if any(token in _cell_text(item).lower() for token in ("fiscal", "fy"))
    ]
    if annual_candidates:
        chosen = annual_candidates[0]
        total = _pick_numeric_value(chosen)
        if total is not None:
            return total, [_provenance_ref(chosen)], "fiscal_annual_row"
    month_keys = _fiscal_month_keys(int(year))
    values: list[dict[str, Any]] = []
    for year_key, month in month_keys:
        item = monthly.get(year_key, {}).get(month)
        if item is None:
            return None, [], "missing_fiscal_month_coverage"
        values.append(item)
    total = sum(_pick_numeric_value(item) or 0.0 for item in values)
    return total, [_provenance_ref(item) for item in values], "fiscal_month_sum"


def _series_average_for_year(
    year: str,
    *,
    monthly: dict[str, dict[int, dict[str, Any]]],
    annual: dict[str, list[dict[str, Any]]],
) -> tuple[float | None, list[dict[str, Any]], str]:
    annual_candidates = list(annual.get(year, []))
    if annual_candidates:
        chosen = annual_candidates[0]
        value = _pick_numeric_value(chosen)
        if value is not None:
            return value, [_provenance_ref(chosen)], "annual_average"
    monthly_values = monthly.get(year, {})
    if len(monthly_values) >= 12:
        ordered = [monthly_values[index] for index in sorted(monthly_values)[:12]]
        average = sum(_pick_numeric_value(item) or 0.0 for item in ordered) / len(ordered)
        return average, [_provenance_ref(item) for item in ordered], "monthly_average"
    return None, [], "missing_cpi_support"


def _operation_years(task_text: str, retrieval_intent: RetrievalIntent) -> list[str]:
    years = _extract_years(f"{retrieval_intent.period} {task_text}")
    return sorted(list(dict.fromkeys(years)))


def _build_answer_text(
    operation: str,
    display_value: str,
    ledger: list[OfficeQAComputeStep],
) -> str:
    lines = [f"Deterministic OfficeQA compute: {operation.replace('_', ' ')}."]
    for step in ledger:
        lines.append(f"- {step.description}")
    lines.append(f"Final answer: {display_value}")
    return "\n".join(lines)


def compact_officeqa_compute_result(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(payload or {})
    if not data:
        return {}
    return {
        "status": str(data.get("status", "")),
        "operation": str(data.get("operation", "")),
        "display_value": str(data.get("display_value", "")),
        "validation_errors": list(data.get("validation_errors", []))[:6],
        "provenance_complete": bool(data.get("provenance_complete")),
        "ledger": [
            {
                "operator": item.get("operator", ""),
                "description": item.get("description", ""),
                "output": dict(item.get("output", {})),
            }
            for item in list(data.get("ledger", []))[:4]
            if isinstance(item, dict)
        ],
    }


def compute_officeqa_result(
    task_text: str,
    retrieval_intent: RetrievalIntent,
    structured_evidence: dict[str, Any] | None,
) -> OfficeQAComputeResult:
    evidence = dict(structured_evidence or {})
    values = [item for item in list(evidence.get("values", [])) if isinstance(item, dict)]
    if not values:
        return OfficeQAComputeResult(
            status="insufficient",
            operation=retrieval_intent.aggregation_shape or "unknown",
            validation_errors=["No structured OfficeQA values are available for deterministic compute."],
        )

    operation = retrieval_intent.aggregation_shape or "point_lookup"
    years = _operation_years(task_text, retrieval_intent)
    metric_tokens = _metric_tokens(task_text, retrieval_intent)
    monthly, annual = _group_values(values, metric_tokens=metric_tokens, include_cpi=False)
    cpi_monthly, cpi_annual = _group_values(values, metric_tokens=set(), include_cpi=True)
    ledger: list[OfficeQAComputeStep] = []
    citations: list[str] = []
    provenance_complete = bool(evidence.get("provenance_complete"))

    def append_step(step: OfficeQAComputeStep) -> None:
        ledger.append(step)
        for ref in step.provenance_refs:
            citation = str(ref.get("citation", "")).strip()
            if citation and citation not in citations:
                citations.append(citation)

    if operation == "monthly_sum":
        if len(years) != 1:
            return OfficeQAComputeResult(
                status="insufficient",
                operation=operation,
                validation_errors=["Monthly sum compute requires exactly one target year."],
            )
        total, refs, mode = _series_total_for_calendar_year(years[0], monthly=monthly, annual={})
        if total is None:
            return OfficeQAComputeResult(
                status="insufficient",
                operation=operation,
                validation_errors=[f"Missing complete monthly coverage for calendar year {years[0]}."],
            )
        append_step(
            OfficeQAComputeStep(
                operator="monthly_sum",
                description=f"{years[0]} monthly sum = {format_scalar_number(total)} using 12 monthly rows [{mode}].",
                inputs={"year": years[0], "mode": mode},
                output={"value": total},
                provenance_refs=refs,
            )
        )
        display_value = _format_numeric(total, task_text)
        return OfficeQAComputeResult(
            status="ok",
            operation=operation,
            final_value=total,
            display_value=display_value,
            answer_text=_build_answer_text(operation, display_value, ledger),
            citations=citations,
            ledger=[step.model_dump() for step in ledger],
            provenance_complete=provenance_complete and bool(refs),
        )

    if operation == "calendar_year_total":
        if len(years) != 1:
            return OfficeQAComputeResult(
                status="insufficient",
                operation=operation,
                validation_errors=["Calendar year total compute requires exactly one target year."],
            )
        total, refs, mode = _series_total_for_calendar_year(years[0], monthly=monthly, annual=annual)
        if total is None:
            return OfficeQAComputeResult(
                status="insufficient",
                operation=operation,
                validation_errors=[f"Missing calendar-year support for {years[0]}."],
            )
        append_step(
            OfficeQAComputeStep(
                operator="calendar_year_total",
                description=f"{years[0]} calendar-year total = {format_scalar_number(total)} [{mode}].",
                inputs={"year": years[0], "mode": mode},
                output={"value": total},
                provenance_refs=refs,
            )
        )
        display_value = _format_numeric(total, task_text)
        return OfficeQAComputeResult(
            status="ok",
            operation=operation,
            final_value=total,
            display_value=display_value,
            answer_text=_build_answer_text(operation, display_value, ledger),
            citations=citations,
            ledger=[step.model_dump() for step in ledger],
            provenance_complete=provenance_complete and bool(refs),
        )

    if operation == "fiscal_year_total":
        if len(years) != 1:
            return OfficeQAComputeResult(
                status="insufficient",
                operation=operation,
                validation_errors=["Fiscal year total compute requires exactly one target year."],
            )
        total, refs, mode = _series_total_for_fiscal_year(years[0], monthly=monthly, annual=annual)
        if total is None:
            return OfficeQAComputeResult(
                status="insufficient",
                operation=operation,
                validation_errors=[f"Missing fiscal-year support for {years[0]}."],
            )
        append_step(
            OfficeQAComputeStep(
                operator="fiscal_year_total",
                description=f"FY {years[0]} total = {format_scalar_number(total)} [{mode}].",
                inputs={"year": years[0], "mode": mode},
                output={"value": total},
                provenance_refs=refs,
            )
        )
        display_value = _format_numeric(total, task_text)
        return OfficeQAComputeResult(
            status="ok",
            operation=operation,
            final_value=total,
            display_value=display_value,
            answer_text=_build_answer_text(operation, display_value, ledger),
            citations=citations,
            ledger=[step.model_dump() for step in ledger],
            provenance_complete=provenance_complete and bool(refs),
        )

    if operation in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"} or retrieval_intent.metric in {
        "absolute percent change",
        "absolute difference",
    }:
        if len(years) < 2:
            return OfficeQAComputeResult(
                status="insufficient",
                operation=operation,
                validation_errors=["Comparison compute requires two target years."],
            )
        base_year, target_year = years[0], years[-1]
        base_total, base_refs, base_mode = _series_total_for_calendar_year(base_year, monthly=monthly, annual=annual)
        target_total, target_refs, target_mode = _series_total_for_calendar_year(target_year, monthly=monthly, annual=annual)
        if base_total is None or target_total is None:
            return OfficeQAComputeResult(
                status="insufficient",
                operation=operation,
                validation_errors=[f"Missing comparable period totals for {base_year} and {target_year}."],
            )
        append_step(
            OfficeQAComputeStep(
                operator="calendar_year_total",
                description=f"{base_year} total = {format_scalar_number(base_total)} [{base_mode}].",
                inputs={"year": base_year, "mode": base_mode},
                output={"value": base_total},
                provenance_refs=base_refs,
            )
        )
        append_step(
            OfficeQAComputeStep(
                operator="calendar_year_total",
                description=f"{target_year} total = {format_scalar_number(target_total)} [{target_mode}].",
                inputs={"year": target_year, "mode": target_mode},
                output={"value": target_total},
                provenance_refs=target_refs,
            )
        )
        if operation == "inflation_adjusted_monthly_difference":
            base_cpi, base_cpi_refs, base_cpi_mode = _series_average_for_year(base_year, monthly=cpi_monthly, annual=cpi_annual)
            target_cpi, target_cpi_refs, target_cpi_mode = _series_average_for_year(target_year, monthly=cpi_monthly, annual=cpi_annual)
            if base_cpi is None or target_cpi is None:
                return OfficeQAComputeResult(
                    status="insufficient",
                    operation=operation,
                    validation_errors=["Inflation-adjusted compute requires CPI support for both comparison years."],
                )
            adjusted_base = base_total * (target_cpi / base_cpi)
            final_value = abs(target_total - adjusted_base)
            append_step(
                OfficeQAComputeStep(
                    operator="inflation_adjustment",
                    description=(
                        f"Adjusted {base_year} total into {target_year} dollars using CPI ratio "
                        f"{format_scalar_number(target_cpi)}/{format_scalar_number(base_cpi)} = {format_scalar_number(adjusted_base)} "
                        f"[{base_cpi_mode}, {target_cpi_mode}]."
                    ),
                    inputs={
                        "base_total": base_total,
                        "base_cpi": base_cpi,
                        "target_cpi": target_cpi,
                        "base_year": base_year,
                        "target_year": target_year,
                    },
                    output={"adjusted_base_value": adjusted_base},
                    provenance_refs=[*base_cpi_refs, *target_cpi_refs],
                )
            )
            append_step(
                OfficeQAComputeStep(
                    operator="absolute_difference",
                    description=f"Absolute inflation-adjusted difference = {format_scalar_number(final_value)}.",
                    inputs={"target_total": target_total, "adjusted_base_value": adjusted_base},
                    output={"value": final_value},
                    provenance_refs=[*base_refs, *target_refs],
                )
            )
            display_value = _format_numeric(final_value, task_text)
            return OfficeQAComputeResult(
                status="ok",
                operation=operation,
                final_value=final_value,
                display_value=display_value,
                answer_text=_build_answer_text(operation, display_value, ledger),
                citations=citations,
                ledger=[step.model_dump() for step in ledger],
                provenance_complete=provenance_complete and bool(base_refs and target_refs),
            )

        if retrieval_intent.metric == "absolute percent change" or operation == "monthly_sum_percent_change":
            if base_total == 0:
                return OfficeQAComputeResult(
                    status="insufficient",
                    operation=operation,
                    validation_errors=["Cannot compute percent change with a zero baseline."],
                )
            final_value = abs((target_total - base_total) / base_total) * 100.0
            append_step(
                OfficeQAComputeStep(
                    operator="absolute_percent_change",
                    description=f"Absolute percent change between {base_year} and {target_year} = {format_scalar_number(final_value)}.",
                    inputs={"base_value": base_total, "target_value": target_total},
                    output={"value": final_value},
                    provenance_refs=[*base_refs, *target_refs],
                )
            )
            display_value = _format_numeric(final_value, task_text, percent=True)
            return OfficeQAComputeResult(
                status="ok",
                operation=operation,
                final_value=final_value,
                display_value=display_value,
                answer_text=_build_answer_text(operation, display_value, ledger),
                unit="percent",
                citations=citations,
                ledger=[step.model_dump() for step in ledger],
                provenance_complete=provenance_complete and bool(base_refs and target_refs),
            )

        final_value = abs(target_total - base_total)
        append_step(
            OfficeQAComputeStep(
                operator="absolute_difference",
                description=f"Absolute difference between {base_year} and {target_year} = {format_scalar_number(final_value)}.",
                inputs={"base_value": base_total, "target_value": target_total},
                output={"value": final_value},
                provenance_refs=[*base_refs, *target_refs],
            )
        )
        display_value = _format_numeric(final_value, task_text)
        return OfficeQAComputeResult(
            status="ok",
            operation=operation or "absolute_difference",
            final_value=final_value,
            display_value=display_value,
            answer_text=_build_answer_text(operation or "absolute_difference", display_value, ledger),
            citations=citations,
            ledger=[step.model_dump() for step in ledger],
            provenance_complete=provenance_complete and bool(base_refs and target_refs),
        )

    if operation == "point_lookup":
        relevant = [item for item in _dedupe_values(values) if _matches_metric(item, metric_tokens)]
        if len(relevant) == 1:
            numeric_value = _pick_numeric_value(relevant[0])
            if numeric_value is None:
                return OfficeQAComputeResult(status="insufficient", operation=operation, validation_errors=["Point lookup value is not numeric."])
            refs = [_provenance_ref(relevant[0])]
            append_step(
                OfficeQAComputeStep(
                    operator="point_lookup",
                    description=f"Direct point lookup = {format_scalar_number(numeric_value)}.",
                    inputs={"match_count": 1},
                    output={"value": numeric_value},
                    provenance_refs=refs,
                )
            )
            display_value = _format_numeric(numeric_value, task_text)
            return OfficeQAComputeResult(
                status="ok",
                operation=operation,
                final_value=numeric_value,
                display_value=display_value,
                answer_text=_build_answer_text(operation, display_value, ledger),
                citations=citations,
                ledger=[step.model_dump() for step in ledger],
                provenance_complete=provenance_complete and bool(refs),
            )

    return OfficeQAComputeResult(
        status="unsupported",
        operation=operation,
        validation_errors=[f"Deterministic OfficeQA compute does not yet support aggregation shape '{operation}'."],
    )
