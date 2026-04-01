from __future__ import annotations

import re
from typing import Any

from agent.contracts import EvidenceSufficiency, RetrievalIntent, SourceBundle

_MONTH_NAMES = (
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
_GENERIC_NUMERIC_SUMMARIES = {
    "row_count",
    "column_count",
    "numeric_cell_count",
}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(needle in lowered for needle in needles)


def _benchmark_policy(benchmark_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict((benchmark_overrides or {}).get("benchmark_policy") or {})


def _officeqa_active(benchmark_overrides: dict[str, Any] | None = None) -> bool:
    return str((benchmark_overrides or {}).get("benchmark_adapter") or "") == "officeqa"


def _extract_period(task_text: str, source_bundle: SourceBundle) -> str:
    if source_bundle.target_period:
        return source_bundle.target_period
    years = re.findall(r"\b((?:19|20)\d{2})\b", task_text or "")
    return " ".join(dict.fromkeys(years[:4]))


def _extract_entity(task_text: str, source_bundle: SourceBundle) -> str:
    if source_bundle.entities:
        return source_bundle.entities[0]
    lowered = task_text or ""
    for pattern in (
        r"for the ([A-Za-z0-9 .,'\-&]+?) in (?:fy|fiscal year|calendar year)\b",
        r"for ([A-Za-z0-9 .,'\-&]+?) in (?:fy|fiscal year|calendar year)\b",
        r"for the ([A-Za-z0-9 .,'\-&]+?)(?:,| specifically| rounded| using| what was| what is|\?)",
        r"for ([A-Za-z0-9 .,'\-&]+?)(?:,| specifically| rounded| using| what was| what is|\?)",
        r"for the ([A-Za-z0-9 .,'\-&]+?)\?",
        r"for ([A-Za-z0-9 .,'\-&]+?)\?",
    ):
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            entity = _normalize_space(match.group(1)).strip(" ,.")
            entity = re.sub(r"\s+in\s+(?:fy|fiscal year|calendar year)\b.*$", "", entity, flags=re.IGNORECASE)
            entity = re.sub(r"\s+(?:using|rounded)\b.*$", "", entity, flags=re.IGNORECASE)
            return entity.strip(" ,.")
    return ""


def _extract_metric(task_text: str) -> str:
    lowered = _normalize_space(task_text).lower()
    if "absolute percent change" in lowered:
        return "absolute percent change"
    if "absolute difference" in lowered:
        return "absolute difference"
    if "public debt outstanding" in lowered:
        return "public debt outstanding"
    if "total sum" in lowered:
        return "total sum of expenditures"
    if "total expenditures" in lowered:
        return "total expenditures"
    if "expenditures" in lowered:
        return "expenditures"
    return ""


def _document_family(task_text: str, source_bundle: SourceBundle, benchmark_overrides: dict[str, Any] | None = None) -> str:
    lowered = (task_text or "").lower()
    if _officeqa_active(benchmark_overrides):
        if "treasury bulletin" in lowered:
            return "treasury_bulletin"
        return "official_government_finance"
    if source_bundle.urls:
        return "reference_documents"
    if _contains_any(lowered, ("report", "filing", "document", "pdf")):
        return "finance_documents"
    return "general_retrieval"


def _aggregation_shape(task_text: str) -> str:
    lowered = (task_text or "").lower()
    if "all individual calendar months" in lowered and "percent change" in lowered:
        return "monthly_sum_percent_change"
    if "all individual calendar months" in lowered and "absolute difference" in lowered and "inflation" in lowered:
        return "inflation_adjusted_monthly_difference"
    if "all individual calendar months" in lowered or "total sum of these values" in lowered:
        return "monthly_sum"
    if "calendar year" in lowered:
        return "calendar_year_total"
    if "fiscal year" in lowered or re.search(r"\bfy\s+\d{4}\b", lowered):
        return "fiscal_year_total"
    return "point_lookup"


def _period_years(period: str, task_text: str) -> list[str]:
    years = re.findall(r"\b((?:19|20)\d{2})\b", f"{period} {task_text}")
    return list(dict.fromkeys(years[:4]))


def _primary_retrieval_metric(metric: str, aggregation_shape: str) -> str:
    normalized = (metric or "").strip().lower()
    if aggregation_shape in {"monthly_sum_percent_change", "inflation_adjusted_monthly_difference"}:
        return "expenditures"
    if normalized in {"absolute percent change", "absolute difference"}:
        return "expenditures"
    return metric


def _extract_qualifier_terms(task_text: str) -> list[str]:
    qualifiers: list[str] = []
    patterns = (
        r"should include\s+(.+?)(?:,| and | but |\.|$)",
        r"shouldn't contain\s+(.+?)(?:,| and | but |\.|$)",
        r"should not contain\s+(.+?)(?:,| and | but |\.|$)",
        r"excluding\s+(.+?)(?:,| and | but |\.|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, task_text or "", re.IGNORECASE):
            value = _normalize_space(match.group(1)).strip(" ,.")
            if value:
                qualifiers.append(value)
    return list(dict.fromkeys(qualifiers[:3]))


def _source_file_query_terms(source_bundle: SourceBundle) -> list[str]:
    terms: list[str] = []
    for match in source_bundle.source_files_found[:6]:
        relative_path = str(match.get("relative_path", "")).strip()
        if relative_path:
            terms.append(relative_path)
        document_id = str(match.get("document_id", "")).strip()
        if document_id:
            terms.append(document_id)
    for item in source_bundle.source_files_expected[:6]:
        compact = str(item).strip()
        if compact:
            terms.append(compact)
    return list(dict.fromkeys(terms))


def build_retrieval_intent(
    task_text: str,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> RetrievalIntent:
    entity = _extract_entity(task_text, source_bundle)
    metric = _extract_metric(task_text)
    period = _extract_period(task_text, source_bundle)
    document_family = _document_family(task_text, source_bundle, benchmark_overrides)
    aggregation_shape = _aggregation_shape(task_text)
    years = _period_years(period, task_text)
    retrieval_metric = _primary_retrieval_metric(metric, aggregation_shape)
    qualifier_terms = _extract_qualifier_terms(task_text)

    must_include_terms: list[str] = []
    if document_family == "treasury_bulletin":
        must_include_terms.append("Treasury Bulletin")
    elif document_family == "official_government_finance":
        must_include_terms.append("official government finance")
    if entity:
        must_include_terms.append(entity)
    if retrieval_metric:
        must_include_terms.append(retrieval_metric)
    if period:
        must_include_terms.extend(period.split())
    if "calendar year" in task_text.lower():
        must_include_terms.append("calendar year")
    if "fiscal year" in task_text.lower() or re.search(r"\bfy\s+\d{4}\b", task_text or "", re.IGNORECASE):
        must_include_terms.append("fiscal year")
    if aggregation_shape.startswith("monthly"):
        must_include_terms.extend(["monthly", "month"])
    if aggregation_shape == "inflation_adjusted_monthly_difference":
        must_include_terms.extend(["cpi", "inflation"])
    must_include_terms.extend(qualifier_terms)
    must_include_terms.extend(_source_file_query_terms(source_bundle))
    must_include_terms = list(dict.fromkeys([item for item in must_include_terms if item]))

    policy = _benchmark_policy(benchmark_overrides)
    must_exclude_terms = list(policy.get("excluded_retrieval_terms", [])) if _officeqa_active(benchmark_overrides) else []

    base_terms = [term for term in [document_family.replace("_", " "), entity, retrieval_metric, period] if term]
    base_query = _normalize_space(" ".join(base_terms))
    query_candidates: list[str] = []
    source_file_terms = _source_file_query_terms(source_bundle)
    if source_file_terms:
        query_candidates.append(_normalize_space(" ".join(source_file_terms[:2])))
    if document_family in {"treasury_bulletin", "official_government_finance"}:
        source_hint = "Treasury Bulletin" if document_family == "treasury_bulletin" else "official government finance"
        focused_terms = [source_hint, entity, retrieval_metric, period]
        query_candidates.append(_normalize_space(" ".join(term for term in focused_terms if term)))
        if entity or retrieval_metric or period:
            query_candidates.append(
                _normalize_space(
                    " ".join(
                        term
                        for term in (
                            f'"{entity}"' if entity else "",
                            f'"{period}"' if period else "",
                            f'"{retrieval_metric}"' if retrieval_metric else "",
                            f'"{source_hint}"',
                        )
                        if term
                    )
                )
            )
        if aggregation_shape.startswith("monthly"):
            monthly_terms = [source_hint, entity, retrieval_metric or "expenditures", period, "monthly"]
            query_candidates.append(_normalize_space(" ".join(term for term in monthly_terms if term)))
            for year in years[:2]:
                query_candidates.append(
                    _normalize_space(
                        " ".join(
                            term
                            for term in (
                                f'"{entity}"' if entity else "",
                                f'"{year}"',
                                '"monthly"',
                                f'"{source_hint}"',
                            )
                            if term
                        )
                    )
                )
        if aggregation_shape == "inflation_adjusted_monthly_difference":
            inflation_terms = [source_hint, entity, retrieval_metric or "expenditures", "CPI inflation", " ".join(years[:2])]
            query_candidates.append(_normalize_space(" ".join(term for term in inflation_terms if term)))
        for qualifier in qualifier_terms[:2]:
            query_candidates.append(
                _normalize_space(
                    " ".join(
                        term
                        for term in (
                            f'"{entity or retrieval_metric}"' if (entity or retrieval_metric) else "",
                            f'"{period}"' if period else "",
                            f'"{qualifier}"',
                            f'"{source_hint}"',
                        )
                        if term
                    )
                )
            )
    elif base_query:
        query_candidates.append(base_query)
        query_candidates.append(_normalize_space(f'"{base_query}" source document'))
    else:
        query_candidates.append(_normalize_space(source_bundle.focus_query or task_text))

    query_candidates = list(dict.fromkeys([query[:280] for query in query_candidates if query]))[:4]
    return RetrievalIntent(
        entity=entity,
        metric=metric,
        period=period,
        document_family=document_family,
        aggregation_shape=aggregation_shape,
        must_include_terms=must_include_terms,
        must_exclude_terms=must_exclude_terms,
        query_candidates=query_candidates,
    )


def _tool_result_text(tool_result: dict[str, Any]) -> str:
    facts = dict(tool_result.get("facts") or {})
    parts: list[str] = []
    for key in ("citation", "document_id"):
        if facts.get(key):
            parts.append(str(facts.get(key)))
    metadata = dict(facts.get("metadata") or {})
    for key in ("file_name", "window"):
        if metadata.get(key):
            parts.append(str(metadata.get(key)))
    for item in facts.get("chunks", []):
        if isinstance(item, dict):
            parts.append(str(item.get("text", "")))
    for item in facts.get("tables", []):
        if isinstance(item, dict):
            headers = " ".join(str(header) for header in item.get("headers", []))
            rows = " ".join(" ".join(str(cell) for cell in row) for row in item.get("rows", []))
            parts.extend([headers, rows])
    for item in facts.get("results", []):
        if isinstance(item, dict):
            parts.append(str(item.get("title", "")))
            parts.append(str(item.get("snippet", "")))
    return _normalize_space(" ".join(part for part in parts if part))


def _entity_scope(task_text: str, retrieval_intent: RetrievalIntent, combined_text: str) -> tuple[str, bool]:
    tokens = [
        token
        for token in _tokenize(retrieval_intent.entity or task_text)
        if len(token) > 2 and token not in {"united", "states"}
    ]
    if not tokens:
        return "unknown", True
    overlap = {token for token in tokens if token in _tokenize(combined_text)}
    if not overlap:
        return "mismatch", False
    if len(overlap) >= min(2, len(tokens)):
        return "matched", True
    return "partial", True


def _period_scope(task_text: str, retrieval_intent: RetrievalIntent, combined_text: str) -> tuple[str, bool]:
    lowered_task = (task_text or "").lower()
    lowered_text = (combined_text or "").lower()
    years = re.findall(r"\b((?:19|20)\d{2})\b", retrieval_intent.period or task_text)
    has_years = not years or all(year in lowered_text for year in years)
    if not has_years:
        return "mismatch", False
    if "calendar year" in lowered_task and "fiscal year" in lowered_text and "calendar year" not in lowered_text:
        return "fiscal_mismatch", False
    if ("fiscal year" in lowered_task or re.search(r"\bfy\s+\d{4}\b", lowered_task)) and "calendar year" in lowered_text and "fiscal year" not in lowered_text:
        return "calendar_mismatch", False
    if years:
        return "matched", True
    return "unknown", True


def _aggregation_scope(task_text: str, retrieval_intent: RetrievalIntent, combined_text: str) -> tuple[str, bool]:
    lowered_task = (task_text or "").lower()
    lowered_text = (combined_text or "").lower()
    if retrieval_intent.aggregation_shape == "monthly_sum":
        if any(month in lowered_text for month in _MONTH_NAMES):
            return "matched_monthly", True
        if "calendar year" in lowered_text or "annual total" in lowered_text:
            return "annual_total_mismatch", False
        return "missing_monthly_support", False
    if retrieval_intent.aggregation_shape == "monthly_sum_percent_change":
        if any(month in lowered_text for month in _MONTH_NAMES) and "percent" in lowered_text:
            return "matched_monthly_change", True
        return "missing_monthly_support", False
    if retrieval_intent.aggregation_shape == "inflation_adjusted_monthly_difference":
        if any(month in lowered_text for month in _MONTH_NAMES) and ("cpi" in lowered_text or "inflation" in lowered_text):
            return "matched_inflation_adjusted", True
        return "missing_inflation_or_monthly_support", False
    if "calendar year" in lowered_task and "fiscal year" in lowered_text and "calendar year" not in lowered_text:
        return "fiscal_mismatch", False
    if "fiscal year" in lowered_task and "calendar year" in lowered_text and "fiscal year" not in lowered_text:
        return "calendar_mismatch", False
    return "matched", True


def _source_family(tool_results: list[dict[str, Any]], combined_text: str) -> str:
    citations: list[str] = []
    for result in tool_results:
        facts = dict(result.get("facts") or {})
        direct = [facts.get("citation", "")]
        for item in facts.get("documents", []):
            if isinstance(item, dict):
                direct.append(item.get("citation", "") or item.get("path", ""))
        for item in facts.get("results", []):
            if isinstance(item, dict):
                direct.append(item.get("url", "") or item.get("citation", ""))
        citations.extend([str(item).lower() for item in direct if str(item).strip()])
    lowered = " ".join(citations + [combined_text.lower()])
    local_or_file_treasury = any(
        ("treasury_bulletin" in citation or "treasury_" in citation or citation.endswith((".json", ".txt", ".csv", ".tsv", ".pdf")))
        and not citation.startswith("http")
        for citation in citations
    )
    official_treasury = any(
        any(host in citation for host in ("govinfo.gov", "census.gov", "va.gov", "fraser.stlouisfed.org", ".gov/"))
        and any(token in citation for token in ("treasury", "bulletin", "statement", "budget"))
        for citation in citations
    )
    if "treasury bulletin" in combined_text.lower() or local_or_file_treasury or official_treasury:
        return "treasury_bulletin"
    if any(
        token in lowered
        for token in (
            "govinfo.gov",
            "census.gov",
            "va.gov",
            "fraser.stlouisfed.org",
            "federal reserve bank of minneapolis",
            "omb",
            "budget of the united states",
            "statistical abstract",
        )
    ):
        return "official_government_document"
    if any(item.endswith(".pdf") for item in citations):
        return "reference_file"
    if citations:
        return "web_or_reference"
    return "unknown"


def _metric_scope(retrieval_intent: RetrievalIntent, combined_text: str) -> tuple[str, bool]:
    metric_basis = _primary_retrieval_metric(retrieval_intent.metric, retrieval_intent.aggregation_shape)
    metric_tokens = [token for token in _tokenize(metric_basis) if len(token) > 2]
    if not metric_tokens:
        return "unknown", True
    overlap = {token for token in metric_tokens if token in _tokenize(combined_text)}
    if not overlap:
        if retrieval_intent.aggregation_shape == "inflation_adjusted_monthly_difference" and any(
            token in (combined_text or "").lower() for token in ("cpi", "inflation")
        ):
            return "matched", True
        return "missing_metric", False
    if len(overlap) >= min(2, len(metric_tokens)):
        return "matched", True
    return "partial", True


def _has_substantive_numeric_support(
    retrieval_intent: RetrievalIntent,
    relevant_results: list[dict[str, Any]],
) -> bool:
    metric_basis = _primary_retrieval_metric(retrieval_intent.metric, retrieval_intent.aggregation_shape)
    metric_tokens = {token for token in _tokenize(metric_basis) if len(token) > 2}
    period_years = set(re.findall(r"\b((?:19|20)\d{2})\b", retrieval_intent.period or ""))
    for result in relevant_results:
        facts = dict(result.get("facts") or {})
        for summary in facts.get("numeric_summaries", []):
            if not isinstance(summary, dict):
                continue
            metric_name = str(summary.get("metric", "") or "")
            if not metric_name or metric_name in _GENERIC_NUMERIC_SUMMARIES or metric_name.endswith("_range"):
                continue
            if metric_tokens and not metric_tokens.intersection(_tokenize(metric_name)):
                continue
            return True
        for table in facts.get("tables", []):
            if not isinstance(table, dict):
                continue
            headers = " ".join(str(item) for item in table.get("headers", []))
            rows = " ".join(" ".join(str(cell) for cell in row) for row in table.get("rows", []))
            table_text = _normalize_space(f"{headers} {rows}")
            numbers = re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", table_text)
            if any(number.rstrip("%") not in period_years for number in numbers):
                if not metric_tokens or metric_tokens.intersection(_tokenize(table_text)):
                    return True
        for chunk in facts.get("chunks", []):
            if not isinstance(chunk, dict):
                continue
            chunk_text = _normalize_space(str(chunk.get("text", "")))
            chunk_text = re.sub(r"^\[(?:Pages?|Rows?) [^\]]+\]\s*", "", chunk_text)
            if not chunk_text:
                continue
            numbers = re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", chunk_text)
            if not any(number.rstrip("%") not in period_years for number in numbers):
                continue
            if not metric_tokens or metric_tokens.intersection(_tokenize(chunk_text)):
                return True
    return False


def _officeqa_failure_dimensions(
    retrieval_intent: RetrievalIntent,
    relevant_results: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    combined_tables = 0
    combined_rows = 0
    combined_months: set[str] = set()
    unit_hints: set[str] = set()
    for result in relevant_results:
        facts = dict(result.get("facts") or {})
        metadata = dict(facts.get("metadata") or {})
        officeqa_status = str(metadata.get("officeqa_status", "") or "").lower()
        if officeqa_status == "missing_table":
            failures.append("missing table")
        elif officeqa_status in {"partial_table", "missing_row"}:
            failures.append("partial table")
        elif officeqa_status == "unit_ambiguity":
            failures.append("unit ambiguity")
        for table in facts.get("tables", []):
            if not isinstance(table, dict):
                continue
            combined_tables += 1
            rows = list(table.get("rows", []))
            combined_rows += len(rows)
            unit_hint = str(table.get("unit_hint", "")).strip().lower()
            if unit_hint:
                unit_hints.add(unit_hint)
            table_text = _normalize_space(
                " ".join(
                    [
                        " ".join(str(item) for item in table.get("headers", [])),
                        " ".join(" ".join(str(cell) for cell in row) for row in rows),
                    ]
                )
            ).lower()
            for month in _MONTH_NAMES:
                if month in table_text:
                    combined_months.add(month)
    if combined_tables == 0:
        failures.append("missing table")
    elif combined_rows < 1:
        failures.append("partial table")
    if retrieval_intent.aggregation_shape.startswith("monthly") and len(combined_months) < 6:
        failures.append("missing month coverage")
    if len(unit_hints) > 1:
        failures.append("unit ambiguity")
    return list(dict.fromkeys(failures))


def assess_evidence_sufficiency(
    task_text: str,
    source_bundle: SourceBundle,
    tool_results: list[dict[str, Any]] | None,
    benchmark_overrides: dict[str, Any] | None = None,
) -> EvidenceSufficiency:
    retrieval_intent = build_retrieval_intent(task_text, source_bundle, benchmark_overrides)
    results = list(tool_results or [])
    relevant_results = [result for result in results if str(result.get("retrieval_status", "") or "") not in {"", "empty", "irrelevant"}]
    combined_text = _normalize_space(" ".join(_tool_result_text(result) for result in relevant_results))

    source_family = _source_family(results, combined_text)
    period_scope, period_ok = _period_scope(task_text, retrieval_intent, combined_text)
    aggregation_type, aggregation_ok = _aggregation_scope(task_text, retrieval_intent, combined_text)
    entity_scope, entity_ok = _entity_scope(task_text, retrieval_intent, combined_text)
    metric_scope, metric_ok = _metric_scope(retrieval_intent, combined_text)
    policy = _benchmark_policy(benchmark_overrides)
    required_source_families = set(policy.get("required_source_families", []))

    missing_dimensions: list[str] = []
    if not relevant_results:
        missing_dimensions.append("retrieved evidence")
    if _officeqa_active(benchmark_overrides) and required_source_families and source_family not in required_source_families:
        missing_dimensions.append("source family grounding")
    if not period_ok:
        missing_dimensions.append("period scope")
    if not aggregation_ok:
        missing_dimensions.append("aggregation semantics")
    if not entity_ok:
        missing_dimensions.append("entity scope")
    if not metric_ok:
        missing_dimensions.append("metric scope")
    if not _has_substantive_numeric_support(retrieval_intent, relevant_results):
        missing_dimensions.append("numeric or quoted support")
    if _officeqa_active(benchmark_overrides):
        missing_dimensions.extend(_officeqa_failure_dimensions(retrieval_intent, relevant_results))

    for result in relevant_results:
        status = str(result.get("retrieval_status", "") or "")
        if status in {"garbled_binary", "parse_error", "network_error", "unsupported_format"}:
            missing_dimensions.append(status)

    missing_dimensions = list(dict.fromkeys(missing_dimensions))
    is_sufficient = not missing_dimensions
    rationale = "Evidence aligns with the requested source, entity, period, and aggregation." if is_sufficient else (
        "Evidence is missing or semantically misaligned with the requested source, entity, period, or aggregation."
    )
    return EvidenceSufficiency(
        source_family=source_family,
        period_scope=period_scope,
        aggregation_type=aggregation_type,
        entity_scope=entity_scope,
        is_sufficient=is_sufficient,
        missing_dimensions=missing_dimensions,
        rationale=rationale,
    )
