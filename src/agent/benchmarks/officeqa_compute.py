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
_ENTITY_STOPWORDS = _STOPWORDS | {"u", "s", "us", "adm", "administration", "related"}
_STRUCTURE_CONFIDENCE_AVG_THRESHOLD = 0.6
_STRUCTURE_CONFIDENCE_MAX_THRESHOLD = 0.7
_PAGE_REF_RE = re.compile(r"^[A-Z]?-?\d+(?:-\d+)?(?:\s+\d+(?:-\d+)?)*\.?$")
_PARTIAL_PERIOD_PATTERNS = (
    r"\bactual\s+\d+\s+months?\b",
    r"\bfirst\s+\d+\s+months?\b",
    r"\blast\s+\d+\s+months?\b",
    r"\b\d+\s+months?\b",
    r"\btotal\s+\d+/\b",
)


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
                " ".join(str(part) for part in value.get("row_path", []) if str(part).strip()),
                str(value.get("row_label", "")),
                " ".join(str(part) for part in value.get("column_path", []) if str(part).strip()),
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


def _entity_tokens(retrieval_intent: RetrievalIntent) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (retrieval_intent.entity or "").lower())
        if len(token) > 2 and token not in _ENTITY_STOPWORDS
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


def _is_page_reference_value_text(text: str) -> bool:
    compact = _normalize_space(text)
    if not compact:
        return False
    return bool(_PAGE_REF_RE.fullmatch(compact))


def _looks_navigational_value(value: dict[str, Any]) -> bool:
    lowered = _cell_text(value).lower()
    if any(token in lowered for token in ("table of contents", "cumulative table of contents", "issue and page number", "articles")):
        return True
    raw_value = str(value.get("raw_value", "") or "")
    column_label = str(value.get("column_label", "") or "").lower()
    if "issue and page number" in column_label and _is_page_reference_value_text(raw_value):
        return True
    return False


def _has_partial_period_marker(text: str) -> bool:
    lowered = _normalize_space(text).lower()
    return any(re.search(pattern, lowered) for pattern in _PARTIAL_PERIOD_PATTERNS)


def _explicit_value_years(value: dict[str, Any]) -> list[str]:
    row_path = [str(part) for part in value.get("row_path", []) if str(part).strip()]
    column_path = [str(part) for part in value.get("column_path", []) if str(part).strip()]
    terminal_parts: list[str] = []
    if row_path:
        terminal_parts.append(row_path[-1])
    if column_path:
        terminal_parts.append(column_path[-1])
    years = _extract_years(" ".join(terminal_parts))
    if years:
        return years
    fallback_parts = [str(value.get("row_label", "")), str(value.get("column_label", ""))]
    years = _extract_years(" ".join(fallback_parts))
    if years:
        return years
    return []


def _pick_numeric_value(value: dict[str, Any]) -> float | None:
    normalized = value.get("normalized_value")
    if isinstance(normalized, (int, float)):
        return float(normalized)
    numeric = value.get("numeric_value")
    if isinstance(numeric, (int, float)):
        return float(numeric)
    return None


def _target_years(task_text: str, retrieval_intent: RetrievalIntent) -> set[str]:
    return set(_operation_years(task_text, retrieval_intent))


def _point_lookup_score(
    value: dict[str, Any],
    *,
    metric_tokens: set[str],
    target_years: set[str],
) -> tuple[int, int, int, int, int, int, int, int, int]:
    text = _cell_text(value).lower()
    path_text = _normalize_space(
        " ".join(
            [
                " ".join(str(part) for part in value.get("row_path", []) if str(part).strip()),
                " ".join(str(part) for part in value.get("column_path", []) if str(part).strip()),
            ]
        )
    ).lower()
    explicit_years = set(_explicit_value_years(value))
    years = explicit_years or set(_extract_value_years(value))
    leaf_year_hits = len(target_years.intersection(explicit_years)) if target_years else 0
    token_hits = len(metric_tokens.intersection(set(re.findall(r"[a-z0-9]+", text))))
    path_token_hits = len(metric_tokens.intersection(set(re.findall(r"[a-z0-9]+", path_text))))
    year_hits = len(target_years.intersection(years)) if target_years else 0
    numeric_flag = 1 if _pick_numeric_value(value) is not None else 0
    has_total = 1 if "total" in text else 0
    row_named = 1 if str(value.get("row_label", "")).strip() else 0
    confidence_bucket = int(round(float(value.get("structure_confidence", 1.0) or 1.0) * 100))
    column_index = int(value.get("column_index", -1) or -1)
    penalty = 0
    if "interest-bearing" in text and "interest" not in metric_tokens:
        penalty -= 2
    if "guaranteed" in text and "guaranteed" not in metric_tokens:
        penalty -= 2
    if "change" in text and not any(token in metric_tokens for token in {"change", "difference"}):
        penalty -= 2
    if "first 5 months" in text and "month" not in metric_tokens:
        penalty -= 2
    if "table of contents" in text or "contents" in text:
        penalty -= 3
    if _looks_navigational_value(value):
        penalty -= 6
    return (leaf_year_hits, year_hits, numeric_flag, path_token_hits, token_hits, has_total, row_named, confidence_bucket, penalty - column_index)


def _provenance_ref(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": str(value.get("document_id", "")),
        "citation": str(value.get("citation", "")),
        "page_locator": str(value.get("page_locator", "")),
        "table_locator": str(value.get("table_locator", "")),
        "table_family": str(value.get("table_family", "")),
        "row_label": str(value.get("row_label", "")),
        "row_path": list(value.get("row_path", [])),
        "column_label": str(value.get("column_label", "")),
        "column_path": list(value.get("column_path", [])),
        "raw_value": str(value.get("raw_value", "")),
    }


def _semantic_status(issues: list[str], issue_name: str) -> str:
    return issue_name if issue_name in issues else "matched"


def _semantic_admissibility(
    values: list[dict[str, Any]],
    *,
    retrieval_intent: RetrievalIntent,
    task_text: str,
    operation: str,
    target_years: set[str],
    metric_tokens: set[str],
) -> dict[str, Any]:
    deduped = [item for item in _dedupe_values(values) if isinstance(item, dict)]
    if not deduped:
        return {
            "admissibility_passed": False,
            "issues": ["missing semantic support"],
            "row_family_status": "unknown",
            "column_family_status": "unknown",
            "period_slice_status": "unknown",
            "aggregation_grain_status": "unknown",
        }

    entity_tokens = _entity_tokens(retrieval_intent)
    issue_set: set[str] = set()
    value_texts = [_cell_text(item).lower() for item in deduped]
    families = {
        str(item.get("table_family", "") or "").strip().lower()
        for item in deduped
        if str(item.get("table_family", "") or "").strip()
    }

    if entity_tokens:
        entity_matched = False
        for text in value_texts:
            tokens = set(re.findall(r"[a-z0-9]+", text))
            overlap = entity_tokens.intersection(tokens)
            if len(overlap) >= min(2, len(entity_tokens)) or (len(entity_tokens) == 1 and bool(overlap)):
                entity_matched = True
                break
        if not entity_matched:
            issue_set.add("wrong row family")

    if metric_tokens and not any(_matches_metric(item, metric_tokens) for item in deduped):
        issue_set.add("wrong column family")

    if operation in {"calendar_year_total", "point_lookup"} or "calendar year" in (task_text or "").lower():
        if any(_has_partial_period_marker(text) for text in value_texts):
            issue_set.add("wrong period slice")

    if operation == "monthly_sum":
        if families and families != {"monthly_series"}:
            issue_set.add("wrong aggregation grain")
    elif operation in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"}:
        if families and any(family != "monthly_series" for family in families):
            issue_set.add("wrong aggregation grain")
    elif operation in {"calendar_year_total", "point_lookup"} and families and "navigation_or_contents" in families:
        issue_set.add("wrong aggregation grain")

    if target_years:
        year_supported = False
        for item in deduped:
            explicit_years = set(_explicit_value_years(item))
            years = explicit_years or set(_extract_value_years(item))
            if target_years.intersection(years):
                year_supported = True
                break
        if not year_supported:
            issue_set.add("wrong period slice")

    issues = sorted(issue_set)
    return {
        "admissibility_passed": not issues,
        "issues": issues,
        "row_family_status": _semantic_status(issues, "wrong row family"),
        "column_family_status": _semantic_status(issues, "wrong column family"),
        "period_slice_status": _semantic_status(issues, "wrong period slice"),
        "aggregation_grain_status": _semantic_status(issues, "wrong aggregation grain"),
        "table_families": sorted(families),
        "value_count": len(deduped),
    }


def _semantic_validation_errors(semantic_diagnostics: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    for issue in list(semantic_diagnostics.get("issues", []) or []):
        if issue == "wrong row family":
            messages.append("Wrong row family: selected evidence does not match the requested entity or category.")
        elif issue == "wrong column family":
            messages.append("Wrong column family: selected evidence does not match the requested metric column.")
        elif issue == "wrong period slice":
            messages.append("Wrong period slice: selected evidence reflects a partial or mismatched time scope.")
        elif issue == "wrong aggregation grain":
            messages.append("Wrong aggregation grain: selected evidence uses the wrong table or aggregation grain for the task.")
        elif issue == "missing semantic support":
            messages.append("Missing semantic support: no candidate values were available for admissibility checks.")
    return messages


def _structure_gate(evidence: dict[str, Any], values: list[dict[str, Any]]) -> tuple[bool, str]:
    summary = dict(evidence.get("structure_confidence_summary", {}) or {})
    avg_confidence = float(summary.get("avg_confidence", 0.0) or 0.0)
    max_confidence = float(summary.get("max_confidence", 0.0) or 0.0)
    if avg_confidence >= _STRUCTURE_CONFIDENCE_AVG_THRESHOLD and max_confidence >= _STRUCTURE_CONFIDENCE_MAX_THRESHOLD:
        return True, ""
    value_scores = [float(item.get("structure_confidence", 1.0) or 1.0) for item in values if isinstance(item, dict)]
    if value_scores and max(value_scores) >= _STRUCTURE_CONFIDENCE_MAX_THRESHOLD and (sum(value_scores) / len(value_scores)) >= _STRUCTURE_CONFIDENCE_AVG_THRESHOLD:
        return True, ""
    return False, "Low-confidence table structure prevents deterministic compute on the current evidence."


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
            return total, [chosen], "annual_row"
    monthly_values = monthly.get(year, {})
    if len(monthly_values) >= 12:
        ordered = [monthly_values[index] for index in sorted(monthly_values)[:12]]
        total = sum(_pick_numeric_value(item) or 0.0 for item in ordered)
        return total, ordered, "monthly_sum"
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
            return total, [chosen], "fiscal_annual_row"
    month_keys = _fiscal_month_keys(int(year))
    values: list[dict[str, Any]] = []
    for year_key, month in month_keys:
        item = monthly.get(year_key, {}).get(month)
        if item is None:
            return None, [], "missing_fiscal_month_coverage"
        values.append(item)
    total = sum(_pick_numeric_value(item) or 0.0 for item in values)
    return total, values, "fiscal_month_sum"


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
            return value, [chosen], "annual_average"
    monthly_values = monthly.get(year, {})
    if len(monthly_values) >= 12:
        ordered = [monthly_values[index] for index in sorted(monthly_values)[:12]]
        average = sum(_pick_numeric_value(item) or 0.0 for item in ordered) / len(ordered)
        return average, ordered, "monthly_average"
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


def _compute_selection_reasoning(operation: str, retrieval_intent: RetrievalIntent, years: list[str]) -> str:
    if operation == "monthly_sum":
        return f"Selected monthly-sum compute because the task asks for a within-year monthly aggregation over {', '.join(years) or 'the requested period'}."
    if operation == "calendar_year_total":
        return f"Selected calendar-year total compute because the task asks for a single-year total over {', '.join(years) or 'the requested year'}."
    if operation == "fiscal_year_total":
        return f"Selected fiscal-year total compute because the task explicitly targets a fiscal year ({', '.join(years) or retrieval_intent.period})."
    if operation == "monthly_sum_percent_change":
        return f"Selected cross-year monthly-sum percent-change compute because the task compares monthly totals across {', '.join(years[:2])}."
    if operation == "inflation_adjusted_monthly_difference":
        return f"Selected inflation-adjusted comparison compute because the task requires inflation support across {', '.join(years[:2])}."
    if operation == "point_lookup":
        return "Selected direct point lookup because the task appears to request a single grounded numeric value."
    return f"Selected deterministic compute path for aggregation shape '{operation}'."


def _rejected_aggregation_alternatives(operation: str, retrieval_intent: RetrievalIntent) -> list[str]:
    rejected: list[str] = []
    if operation != "point_lookup":
        rejected.append("point_lookup rejected because the task requires aggregation or comparison, not a single isolated value")
    if operation not in {"calendar_year_total", "fiscal_year_total"}:
        rejected.append("single-total path rejected because the task needs a more structured aggregation shape")
    if operation not in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"} and retrieval_intent.metric in {
        "absolute percent change",
        "absolute difference",
    }:
        rejected.append("simple difference path rejected because the task specifies a comparison metric that requires paired period totals")
    if operation != "inflation_adjusted_monthly_difference" and "inflation" in (retrieval_intent.metric or "").lower():
        rejected.append("non-inflation path rejected because the task explicitly asks for inflation support")
    return list(dict.fromkeys(rejected))


def _result_with_diagnostics(
    *,
    operation: str,
    retrieval_intent: RetrievalIntent,
    years: list[str],
    **kwargs: Any,
) -> OfficeQAComputeResult:
    kwargs.setdefault("selection_reasoning", _compute_selection_reasoning(operation, retrieval_intent, years))
    kwargs.setdefault("rejected_alternatives", _rejected_aggregation_alternatives(operation, retrieval_intent))
    return OfficeQAComputeResult(operation=operation, **kwargs)


def compact_officeqa_compute_result(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(payload or {})
    if not data:
        return {}
    return {
        "status": str(data.get("status", "")),
        "operation": str(data.get("operation", "")),
        "display_value": str(data.get("display_value", "")),
        "selection_reasoning": str(data.get("selection_reasoning", "")),
        "rejected_alternative_count": len(list(data.get("rejected_alternatives", []))),
        "validation_errors": list(data.get("validation_errors", []))[:6],
        "semantic_diagnostics": {
            "admissibility_passed": bool(dict(data.get("semantic_diagnostics", {}) or {}).get("admissibility_passed", False)),
            "issues": list(dict(data.get("semantic_diagnostics", {}) or {}).get("issues", []))[:6],
        },
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
        return _result_with_diagnostics(
            status="insufficient",
            operation=retrieval_intent.aggregation_shape or "unknown",
            retrieval_intent=retrieval_intent,
            years=[],
            validation_errors=["No structured OfficeQA values are available for deterministic compute."],
        )

    operation = retrieval_intent.aggregation_shape or "point_lookup"
    years = _operation_years(task_text, retrieval_intent)
    metric_tokens = _metric_tokens(task_text, retrieval_intent)
    structure_ok, structure_error = _structure_gate(evidence, values)
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
        if not structure_ok:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[structure_error],
            )
        if len(years) != 1:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=["Monthly sum compute requires exactly one target year."],
            )
        total, selected_values, mode = _series_total_for_calendar_year(years[0], monthly=monthly, annual={})
        if total is None:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[f"Missing complete monthly coverage for calendar year {years[0]}."],
            )
        semantic_diagnostics = _semantic_admissibility(
            selected_values,
            retrieval_intent=retrieval_intent,
            task_text=task_text,
            operation=operation,
            target_years=set(years),
            metric_tokens=metric_tokens,
        )
        semantic_errors = _semantic_validation_errors(semantic_diagnostics)
        if semantic_errors:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=semantic_errors,
                semantic_diagnostics=semantic_diagnostics,
            )
        refs = [_provenance_ref(item) for item in selected_values]
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
        return _result_with_diagnostics(
            status="ok",
            operation=operation,
            retrieval_intent=retrieval_intent,
            years=years,
            final_value=total,
            display_value=display_value,
            answer_text=_build_answer_text(operation, display_value, ledger),
            citations=citations,
            ledger=[step.model_dump() for step in ledger],
            semantic_diagnostics=semantic_diagnostics,
            provenance_complete=provenance_complete and bool(refs),
        )

    if operation == "calendar_year_total":
        if not structure_ok:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[structure_error],
            )
        if len(years) != 1:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=["Calendar year total compute requires exactly one target year."],
            )
        total, selected_values, mode = _series_total_for_calendar_year(years[0], monthly=monthly, annual=annual)
        if total is None:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[f"Missing calendar-year support for {years[0]}."],
            )
        semantic_diagnostics = _semantic_admissibility(
            selected_values,
            retrieval_intent=retrieval_intent,
            task_text=task_text,
            operation=operation,
            target_years=set(years),
            metric_tokens=metric_tokens,
        )
        semantic_errors = _semantic_validation_errors(semantic_diagnostics)
        if semantic_errors:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=semantic_errors,
                semantic_diagnostics=semantic_diagnostics,
            )
        refs = [_provenance_ref(item) for item in selected_values]
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
        return _result_with_diagnostics(
            status="ok",
            operation=operation,
            retrieval_intent=retrieval_intent,
            years=years,
            final_value=total,
            display_value=display_value,
            answer_text=_build_answer_text(operation, display_value, ledger),
            citations=citations,
            ledger=[step.model_dump() for step in ledger],
            semantic_diagnostics=semantic_diagnostics,
            provenance_complete=provenance_complete and bool(refs),
        )

    if operation == "fiscal_year_total":
        if not structure_ok:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[structure_error],
            )
        if len(years) != 1:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=["Fiscal year total compute requires exactly one target year."],
            )
        total, selected_values, mode = _series_total_for_fiscal_year(years[0], monthly=monthly, annual=annual)
        if total is None:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[f"Missing fiscal-year support for {years[0]}."],
            )
        semantic_diagnostics = _semantic_admissibility(
            selected_values,
            retrieval_intent=retrieval_intent,
            task_text=task_text,
            operation=operation,
            target_years=set(years),
            metric_tokens=metric_tokens,
        )
        semantic_errors = _semantic_validation_errors(semantic_diagnostics)
        if semantic_errors:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=semantic_errors,
                semantic_diagnostics=semantic_diagnostics,
            )
        refs = [_provenance_ref(item) for item in selected_values]
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
        return _result_with_diagnostics(
            status="ok",
            operation=operation,
            retrieval_intent=retrieval_intent,
            years=years,
            final_value=total,
            display_value=display_value,
            answer_text=_build_answer_text(operation, display_value, ledger),
            citations=citations,
            ledger=[step.model_dump() for step in ledger],
            semantic_diagnostics=semantic_diagnostics,
            provenance_complete=provenance_complete and bool(refs),
        )

    if operation in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"} or retrieval_intent.metric in {
        "absolute percent change",
        "absolute difference",
    }:
        if not structure_ok:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[structure_error],
            )
        if len(years) < 2:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=["Comparison compute requires two target years."],
            )
        base_year, target_year = years[0], years[-1]
        base_total, base_values, base_mode = _series_total_for_calendar_year(base_year, monthly=monthly, annual=annual)
        target_total, target_values, target_mode = _series_total_for_calendar_year(target_year, monthly=monthly, annual=annual)
        if base_total is None or target_total is None:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[f"Missing comparable period totals for {base_year} and {target_year}."],
            )
        semantic_diagnostics = _semantic_admissibility(
            [*base_values, *target_values],
            retrieval_intent=retrieval_intent,
            task_text=task_text,
            operation=operation,
            target_years=set(years),
            metric_tokens=metric_tokens,
        )
        semantic_errors = _semantic_validation_errors(semantic_diagnostics)
        if semantic_errors:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=semantic_errors,
                semantic_diagnostics=semantic_diagnostics,
            )
        base_refs = [_provenance_ref(item) for item in base_values]
        target_refs = [_provenance_ref(item) for item in target_values]
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
            base_cpi, base_cpi_values, base_cpi_mode = _series_average_for_year(base_year, monthly=cpi_monthly, annual=cpi_annual)
            target_cpi, target_cpi_values, target_cpi_mode = _series_average_for_year(target_year, monthly=cpi_monthly, annual=cpi_annual)
            if base_cpi is None or target_cpi is None:
                return _result_with_diagnostics(
                    status="insufficient",
                    operation=operation,
                    retrieval_intent=retrieval_intent,
                    years=years,
                    validation_errors=["Inflation-adjusted compute requires CPI support for both comparison years."],
                )
            base_cpi_refs = [_provenance_ref(item) for item in base_cpi_values]
            target_cpi_refs = [_provenance_ref(item) for item in target_cpi_values]
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
            return _result_with_diagnostics(
                status="ok",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                final_value=final_value,
                display_value=display_value,
                answer_text=_build_answer_text(operation, display_value, ledger),
                citations=citations,
                ledger=[step.model_dump() for step in ledger],
                semantic_diagnostics=semantic_diagnostics,
                provenance_complete=provenance_complete and bool(base_refs and target_refs),
            )

        if retrieval_intent.metric == "absolute percent change" or operation == "monthly_sum_percent_change":
            if base_total == 0:
                return _result_with_diagnostics(
                    status="insufficient",
                    operation=operation,
                    retrieval_intent=retrieval_intent,
                    years=years,
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
            return _result_with_diagnostics(
                status="ok",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                final_value=final_value,
                display_value=display_value,
                answer_text=_build_answer_text(operation, display_value, ledger),
                unit="percent",
                citations=citations,
                ledger=[step.model_dump() for step in ledger],
                semantic_diagnostics=semantic_diagnostics,
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
        return _result_with_diagnostics(
            status="ok",
            operation=operation or "absolute_difference",
            retrieval_intent=retrieval_intent,
            years=years,
            final_value=final_value,
            display_value=display_value,
            answer_text=_build_answer_text(operation or "absolute_difference", display_value, ledger),
            citations=citations,
            ledger=[step.model_dump() for step in ledger],
            semantic_diagnostics=semantic_diagnostics,
            provenance_complete=provenance_complete and bool(base_refs and target_refs),
        )

    if operation == "point_lookup":
        if not structure_ok:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[structure_error],
            )
        relevant = [item for item in _dedupe_values(values) if _matches_metric(item, metric_tokens)]
        relevant = [item for item in relevant if not _looks_navigational_value(item)]
        if not relevant:
            return _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=["Point lookup could not isolate a grounded financial value after excluding navigational page-reference cells."],
            )
        if relevant:
            numeric_relevant = [item for item in relevant if _pick_numeric_value(item) is not None]
            if numeric_relevant:
                relevant = numeric_relevant
            target_years = _target_years(task_text, retrieval_intent)
            ranked = sorted(
                relevant,
                key=lambda item: (_point_lookup_score(item, metric_tokens=metric_tokens, target_years=target_years), _cell_text(item)),
                reverse=True,
            )
            chosen = ranked[0]
            chosen_score = _point_lookup_score(chosen, metric_tokens=metric_tokens, target_years=target_years)
            second_score = (
                _point_lookup_score(ranked[1], metric_tokens=metric_tokens, target_years=target_years)
                if len(ranked) > 1
                else None
            )
            if chosen_score[0] <= 0 and target_years:
                return _result_with_diagnostics(
                    status="insufficient",
                    operation=operation,
                    retrieval_intent=retrieval_intent,
                    years=years,
                    validation_errors=["Point lookup could not isolate a value aligned to the requested year."],
                )
            if chosen_score[1] <= 0:
                return _result_with_diagnostics(
                    status="insufficient",
                    operation=operation,
                    retrieval_intent=retrieval_intent,
                    years=years,
                    validation_errors=["Point lookup could not isolate a value aligned to the requested metric."],
                )
            if second_score is not None and second_score == chosen_score:
                return _result_with_diagnostics(
                    status="insufficient",
                    operation=operation,
                    retrieval_intent=retrieval_intent,
                    years=years,
                    validation_errors=["Point lookup found multiple equally plausible values and could not disambiguate the correct one."],
                )
            if _looks_navigational_value(chosen):
                return _result_with_diagnostics(
                    status="insufficient",
                    operation=operation,
                    retrieval_intent=retrieval_intent,
                    years=years,
                    validation_errors=["Point lookup selected a navigational page-reference cell rather than a grounded financial value."],
                )
            semantic_diagnostics = _semantic_admissibility(
                [chosen],
                retrieval_intent=retrieval_intent,
                task_text=task_text,
                operation=operation,
                target_years=target_years,
                metric_tokens=metric_tokens,
            )
            semantic_errors = _semantic_validation_errors(semantic_diagnostics)
            if semantic_errors:
                return _result_with_diagnostics(
                    status="insufficient",
                    operation=operation,
                    retrieval_intent=retrieval_intent,
                    years=years,
                    validation_errors=semantic_errors,
                    semantic_diagnostics=semantic_diagnostics,
                )
            numeric_value = _pick_numeric_value(chosen)
            if numeric_value is None:
                return _result_with_diagnostics(status="insufficient", operation=operation, retrieval_intent=retrieval_intent, years=years, validation_errors=["Point lookup value is not numeric."])
            refs = [_provenance_ref(chosen)]
            append_step(
                OfficeQAComputeStep(
                    operator="point_lookup",
                    description=f"Direct point lookup = {format_scalar_number(numeric_value)}.",
                    inputs={"match_count": len(relevant)},
                    output={"value": numeric_value},
                    provenance_refs=refs,
                )
            )
            display_value = _format_numeric(numeric_value, task_text)
            return _result_with_diagnostics(
                status="ok",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                final_value=numeric_value,
                display_value=display_value,
                answer_text=_build_answer_text(operation, display_value, ledger),
                citations=citations,
                ledger=[step.model_dump() for step in ledger],
                semantic_diagnostics=semantic_diagnostics,
                provenance_complete=provenance_complete and bool(refs),
            )

    return _result_with_diagnostics(
        status="unsupported",
        operation=operation,
        retrieval_intent=retrieval_intent,
        years=years,
        validation_errors=[f"Deterministic OfficeQA compute does not yet support aggregation shape '{operation}'."],
    )
