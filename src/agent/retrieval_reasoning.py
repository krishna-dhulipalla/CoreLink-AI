from __future__ import annotations

import re
from typing import Any

from agent.benchmarks.officeqa import officeqa_analysis_modes
from agent.context.extraction import build_question_semantic_plan
from agent.contracts import EvidencePlan, EvidenceRequirement, EvidenceSufficiency, QueryPlan, RetrievalIntent, SourceBundle

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
_STRUCTURE_CONFIDENCE_AVG_THRESHOLD = 0.6
_STRUCTURE_CONFIDENCE_MAX_THRESHOLD = 0.7


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
    generic_source_entities = {
        "treasury bulletin",
        "annual report",
        "narrative discussion",
        "bulletin",
        "report",
        "document",
    }
    if source_bundle.entities:
        first = _normalize_space(source_bundle.entities[0]).strip(" ,.")
        if first.lower() not in generic_source_entities:
            return first
    lowered = task_text or ""
    for pattern in (
        r"reason was given for (?:the )?([A-Za-z0-9 .,'\-&]+?)(?:\s+in\s+\b(?:19|20)\d{2}\b|\?)",
        r"reason was given for (?:the )?([A-Za-z0-9 .,'\-&]+?)(?:,|\.|$)",
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
    for pattern in (
        r"reason was given for (?:the )?([a-z0-9 .,'\-&]+?)(?:\s+in\s+\b(?:19|20)\d{2}\b|\?|$)",
        r"what (?:was|is|were) (?:the )?([a-z0-9 .,'\-&]+?)(?:\s+in\s+(?:fy|fiscal year|calendar year|\b(?:19|20)\d{2}\b)|\?|$)",
    ):
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            candidate = _normalize_space(match.group(1)).strip(" ,.")
            candidate = re.sub(r"\baccording to\b.*$", "", candidate, flags=re.IGNORECASE).strip(" ,.")
            if candidate and candidate not in {"treasury bulletin", "narrative discussion", "report", "document"}:
                return candidate
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
    if (
        "all individual calendar months" in lowered
        or "total sum of these values" in lowered
        or "total monthly expenditures" in lowered
        or "total monthly receipts" in lowered
        or "monthly expenditures" in lowered
        or "monthly receipts" in lowered
        or ("monthly" in lowered and any(token in lowered for token in ("total", "sum", "series", "values")))
    ):
        return "monthly_sum"
    if "calendar year" in lowered:
        return "calendar_year_total"
    if "fiscal year" in lowered or re.search(r"\bfy\s+\d{4}\b", lowered):
        return "fiscal_year_total"
    return "point_lookup"


def _task_analysis_modes(task_text: str, benchmark_overrides: dict[str, Any] | None = None) -> list[str]:
    if _officeqa_active(benchmark_overrides):
        return officeqa_analysis_modes(task_text)
    return []


def _metric_is_implicit(metric: str, aggregation_shape: str) -> bool:
    normalized = (metric or "").strip().lower()
    if not normalized:
        return True
    return normalized in {"absolute percent change", "absolute difference"} or aggregation_shape in {
        "monthly_sum_percent_change",
        "inflation_adjusted_monthly_difference",
    }


def _needs_narrative_support(task_text: str, analysis_modes: list[str]) -> bool:
    lowered = (task_text or "").lower()
    if any(
        token in lowered
        for token in (
            "narrative",
            "discussion",
            "reason was given",
            "trend",
            "forecast",
            "project",
            "projection",
            "explain",
            "describe",
            "why",
            "correlation",
            "regression",
            "standard deviation",
            "variance",
            "value at risk",
            "var ",
        )
    ):
        return True
    return any(mode in analysis_modes for mode in ("statistical_analysis", "time_series_forecasting", "risk_metric"))


def _supports_deterministic_compute(aggregation_shape: str, analysis_modes: list[str]) -> bool:
    unsupported_modes = {"statistical_analysis", "time_series_forecasting", "risk_metric", "weighted_average"}
    if unsupported_modes.intersection(set(analysis_modes)):
        return False
    return aggregation_shape in {
        "monthly_sum",
        "calendar_year_total",
        "fiscal_year_total",
        "monthly_sum_percent_change",
        "inflation_adjusted_monthly_difference",
        "point_lookup",
    }


def _needs_numeric_core(task_text: str, aggregation_shape: str, analysis_modes: list[str]) -> bool:
    lowered = (task_text or "").lower()
    if aggregation_shape != "point_lookup":
        return True
    if any(
        token in lowered
        for token in (
            "amount",
            "average",
            "calculate",
            "calculation",
            "compute",
            "correlation",
            "difference",
            "forecast",
            "percent change",
            "regression",
            "standard deviation",
            "sum",
            "total",
            "value at risk",
            "var ",
            "variance",
            "weighted average",
        )
    ):
        return True
    return any(
        mode in analysis_modes
        for mode in (
            "inflation_adjustment",
            "statistical_analysis",
            "time_series_forecasting",
            "weighted_average",
            "risk_metric",
            "numeric_compute",
        )
    )


def _classify_answer_mode(
    task_text: str,
    aggregation_shape: str,
    analysis_modes: list[str],
) -> tuple[str, str, bool]:
    narrative_support = _needs_narrative_support(task_text, analysis_modes)
    numeric_core = _needs_numeric_core(task_text, aggregation_shape, analysis_modes)
    deterministic_supported = _supports_deterministic_compute(aggregation_shape, analysis_modes)

    if aggregation_shape == "point_lookup":
        if deterministic_supported and numeric_core and not narrative_support:
            return "deterministic_compute", "required", False
        if narrative_support and numeric_core:
            return "hybrid_grounded", "preferred", True
        if numeric_core:
            synthesis_heavy = bool(
                {"weighted_average", "statistical_analysis", "time_series_forecasting", "risk_metric"}.intersection(set(analysis_modes))
            )
            return "grounded_synthesis", "preferred", synthesis_heavy
        return "grounded_synthesis", "not_applicable", False

    if deterministic_supported and numeric_core and narrative_support:
        return "hybrid_grounded", "required", False
    if deterministic_supported and numeric_core:
        return "deterministic_compute", "required", False
    if numeric_core and narrative_support:
        return "hybrid_grounded", "preferred", True
    if numeric_core:
        return "grounded_synthesis", "preferred", True
    return "grounded_synthesis", "not_applicable", False


def _select_retrieval_strategy(
    task_text: str,
    source_bundle: SourceBundle,
    metric: str,
    years: list[str],
    aggregation_shape: str,
    analysis_modes: list[str],
) -> tuple[str, float, list[str], list[str]]:
    lowered = (task_text or "").lower()
    implicit_metric = _metric_is_implicit(metric, aggregation_shape)
    narrative_support = _needs_narrative_support(task_text, analysis_modes)
    multi_document = len(years) >= 2 or any(
        token in lowered
        for token in (
            "across documents",
            "across reports",
            "between documents",
            "source files",
            "compare multiple bulletins",
            "combine multiple documents",
        )
    )
    multi_table = any(mode in analysis_modes for mode in ("inflation_adjustment", "weighted_average")) or any(
        token in lowered
        for token in (
            "weighted average",
            "inflation-adjusted",
            "adjusted for inflation",
            "join",
            "combine table",
        )
    )
    advanced_reasoning = any(
        mode in analysis_modes for mode in ("statistical_analysis", "time_series_forecasting", "risk_metric")
    )

    if multi_document:
        return "multi_document", 0.9, ["multi_table", "hybrid", "text_first"], ["document_id", "year", "month", "unit"]
    if multi_table:
        return "multi_table", 0.84, ["hybrid", "text_first", "table_first"], ["year", "month", "metric", "unit"]
    if advanced_reasoning:
        return "hybrid", 0.8, ["text_first", "table_first", "multi_table"], ["year", "month", "metric"]
    if implicit_metric and narrative_support:
        return "hybrid", 0.76, ["text_first", "table_first"], ["metric", "year"]
    if narrative_support and aggregation_shape == "point_lookup":
        return "text_first", 0.74, ["hybrid", "table_first"], ["metric", "year"]
    return "table_first", 0.88, ["hybrid", "text_first"], []


def _expected_unit_kind(task_text: str, metric: str, analysis_modes: list[str]) -> str:
    lowered = f"{task_text} {metric}".lower()
    if "percent" in lowered or "%" in lowered:
        return "percent"
    if any(token in lowered for token in ("dollar", "expenditure", "receipts", "debt", "balance", "outlay")):
        return "currency"
    if "risk_metric" in analysis_modes and "var" in lowered:
        return "currency"
    return "scalar"


def _build_evidence_plan(
    task_text: str,
    source_bundle: SourceBundle,
    metric: str,
    period: str,
    period_type: str,
    target_years: list[str],
    publication_year_window: list[str],
    preferred_publication_years: list[str],
    aggregation_shape: str,
    granularity_requirement: str,
    strategy: str,
    analysis_modes: list[str],
    join_requirements: list[str],
    include_constraints: list[str],
    exclude_constraints: list[str],
) -> EvidencePlan:
    years = list(target_years) or _period_years(period, task_text)
    metric_identity = _primary_retrieval_metric(metric, aggregation_shape) or "document-grounded financial value"
    required_month_coverage = aggregation_shape.startswith("monthly") or any(
        mode in analysis_modes for mode in ("statistical_analysis", "time_series_forecasting")
    )
    required_month_count = 12 if required_month_coverage else 0
    requires_inflation_support = aggregation_shape == "inflation_adjusted_monthly_difference" or "inflation_adjustment" in analysis_modes
    requires_statistical_series = "statistical_analysis" in analysis_modes
    requires_forecast_support = "time_series_forecasting" in analysis_modes
    requires_cross_source_alignment = strategy == "multi_document"
    implicit_metric = _metric_is_implicit(metric, aggregation_shape)
    narrative_support = _needs_narrative_support(task_text, analysis_modes)
    advanced_reasoning = any(
        mode in analysis_modes for mode in ("statistical_analysis", "time_series_forecasting", "risk_metric")
    )
    requires_text_support = (
        strategy in {"text_first", "hybrid"}
        or narrative_support
        or (strategy == "multi_document" and advanced_reasoning)
        or (strategy == "multi_document" and aggregation_shape == "point_lookup" and implicit_metric)
    )
    requires_table_support = strategy in {"table_first", "hybrid", "multi_table", "multi_document"} or aggregation_shape != "point_lookup"
    expected_value_count = max(1, len(years) or 1)
    if required_month_coverage:
        expected_value_count = max(expected_value_count, max(1, len(years) or 1) * 12)

    requirements: list[EvidenceRequirement] = [
        EvidenceRequirement(
            kind="primary_series",
            label=f"Ground the primary metric '{metric_identity}' in the source evidence.",
            target_count=max(1, len(years) or 1),
            metric=metric_identity,
            years=years,
            support_mode="table" if requires_table_support and not requires_text_support else "table_or_text",
            rationale="The target metric must be recovered before synthesis or compute can be trusted.",
        )
    ]
    if required_month_coverage:
        requirements.append(
            EvidenceRequirement(
                kind="monthly_coverage",
                label="Recover a complete monthly series for the requested period.",
                target_count=max(12, expected_value_count),
                metric=metric_identity,
                years=years,
                support_mode="table",
                rationale="Monthly aggregation, statistics, and forecasting all require broad month coverage.",
            )
        )
    if requires_inflation_support:
        requirements.append(
            EvidenceRequirement(
                kind="inflation_support",
                label="Recover CPI or inflation support aligned to the requested period.",
                target_count=max(1, len(years) or 1),
                metric="cpi",
                years=years,
                support_mode="table_or_text",
                rationale="Inflation-adjusted calculations require a grounded price-index series or explicit inflation support.",
            )
        )
    if requires_text_support:
        requirements.append(
            EvidenceRequirement(
                kind="narrative_support",
                label="Capture quoted or page-level narrative support for ambiguous or implicit metrics.",
                target_count=1,
                metric=metric_identity,
                years=years,
                support_mode="text",
                rationale="Narrative context helps disambiguate implicit metrics, trends, and forecasting questions.",
            )
        )
    if join_requirements:
        requirements.append(
            EvidenceRequirement(
                kind="join_ready_support",
                label="Align retrieved evidence across the join keys needed for compute.",
                target_count=max(1, len(years) or 1),
                metric=metric_identity,
                years=years,
                support_mode="table_and_text" if requires_text_support else "table",
                rationale="Multi-table and multi-document questions require aligned keys before deterministic compute.",
            )
        )
    if include_constraints:
        requirements.append(
            EvidenceRequirement(
                kind="include_constraints",
                label=f"Respect explicit question qualifiers: {', '.join(include_constraints[:3])}.",
                target_count=1,
                metric=metric_identity,
                years=years,
                support_mode="table_or_text",
                rationale="User-provided qualifiers should become retrieval constraints, not be buried inside entity text.",
            )
        )
    if exclude_constraints:
        requirements.append(
            EvidenceRequirement(
                kind="exclude_constraints",
                label=f"Avoid excluded scopes or evidence: {', '.join(exclude_constraints[:3])}.",
                target_count=1,
                metric=metric_identity,
                years=years,
                support_mode="table_or_text",
                rationale="Negative constraints should stay explicit so ranking and validation can reject mismatched evidence.",
            )
        )

    required_series = [f"{metric_identity} {year}".strip() for year in years] if years else [metric_identity]
    return EvidencePlan(
        objective=_normalize_space(source_bundle.focus_query or task_text)[:240],
        metric_identity=metric_identity,
        expected_unit_kind=_expected_unit_kind(task_text, metric_identity, analysis_modes),
        expected_value_count=expected_value_count,
        period_type=period_type,
        required_years=years,
        publication_year_window=publication_year_window,
        preferred_publication_years=preferred_publication_years,
        required_month_coverage=required_month_coverage,
        required_month_count=required_month_count,
        requires_table_support=requires_table_support,
        requires_text_support=requires_text_support,
        requires_cross_source_alignment=requires_cross_source_alignment,
        requires_inflation_support=requires_inflation_support,
        requires_statistical_series=requires_statistical_series,
        requires_forecast_support=requires_forecast_support,
        required_series=required_series,
        join_keys=list(dict.fromkeys(join_requirements)),
        requirements=requirements,
    )


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
    for match in source_bundle.source_files_found:
        relative_path = str(match.get("relative_path", "")).strip()
        if relative_path:
            terms.append(relative_path)
        document_id = str(match.get("document_id", "")).strip()
        if document_id:
            terms.append(document_id)
    for item in source_bundle.source_files_expected:
        compact = str(item).strip()
        if compact:
            terms.append(compact)
    return list(dict.fromkeys(terms))


def _active_query_candidates(
    query_plan: QueryPlan,
    *,
    source_constraint_policy: str,
) -> list[str]:
    candidates = [
        query_plan.temporal_query,
        query_plan.primary_semantic_query,
        query_plan.granularity_query,
        query_plan.qualifier_query or query_plan.alternate_lexical_query,
    ]
    if source_constraint_policy == "hard" or not any(str(item or "").strip() for item in candidates):
        candidates.append(query_plan.source_file_query)
    return list(dict.fromkeys([str(query).strip()[:280] for query in candidates if str(query or "").strip()]))[:4]


def _build_query_plan(
    *,
    document_family: str,
    entity: str,
    metric: str,
    retrieval_metric: str,
    period: str,
    period_type: str,
    publication_year_window: list[str],
    preferred_publication_years: list[str],
    granularity_requirement: str,
    include_constraints: list[str],
    exclude_constraints: list[str],
    source_bundle: SourceBundle,
) -> QueryPlan:
    source_hint = ""
    if document_family == "treasury_bulletin":
        source_hint = "Treasury Bulletin"
    elif document_family == "official_government_finance":
        source_hint = "official government finance"
    elif document_family:
        source_hint = document_family.replace("_", " ")

    monthly_hint = "monthly" if granularity_requirement == "monthly_series" else ""
    if granularity_requirement == "monthly_series" and "series" not in monthly_hint:
        monthly_hint = "monthly series"
    annual_hint = ""
    if granularity_requirement == "calendar_year":
        annual_hint = "calendar year"
    elif granularity_requirement == "fiscal_year":
        annual_hint = "fiscal year"
    elif granularity_requirement == "narrative_support":
        annual_hint = "narrative discussion"

    source_file_terms = _source_file_query_terms(source_bundle)
    source_file_query = _normalize_space(" ".join(source_file_terms))[:280] if source_file_terms else ""
    primary_semantic_query = _normalize_space(
        " ".join(
            part
            for part in (
                source_hint,
                entity,
                retrieval_metric or metric,
                period,
                monthly_hint or annual_hint,
            )
            if part
        )
    )[:280]
    alternate_lexical_query = _normalize_space(
        " ".join(
            part
            for part in (
                f'"{entity}"' if entity else "",
                f'"{period}"' if period else "",
                f'"{retrieval_metric or metric}"' if (retrieval_metric or metric) else "",
                f'"{source_hint}"' if source_hint else "",
            )
            if part
        )
    )[:280]
    temporal_query = _normalize_space(
        " ".join(
            part
            for part in (
                source_hint,
                entity,
                retrieval_metric or metric,
                " ".join(preferred_publication_years[:2]),
                period,
                period_type.replace("_", " ") if period_type else "",
                monthly_hint or annual_hint,
            )
            if part
        )
    )[:280]
    granularity_query = _normalize_space(
        " ".join(
            part
            for part in (
                source_hint,
                retrieval_metric or metric or source_bundle.focus_query,
                period,
                " ".join(publication_year_window[:3]) if publication_year_window else "",
                monthly_hint,
                annual_hint,
                "reported values" if "specifically only the reported values" in {item.lower() for item in include_constraints} else "",
            )
            if part
        )
    )[:280]
    qualifier_terms = [*include_constraints[:2], *[f"excluding {item}" for item in exclude_constraints[:2]]]
    qualifier_query = _normalize_space(
        " ".join(
            part
            for part in (
                source_hint,
                entity or retrieval_metric or metric,
                period,
                " ".join(qualifier_terms),
            )
            if part
        )
    )[:280]
    return QueryPlan(
        primary_semantic_query=primary_semantic_query,
        temporal_query=temporal_query,
        alternate_lexical_query=alternate_lexical_query,
        granularity_query=granularity_query,
        qualifier_query=qualifier_query,
        source_file_query=source_file_query,
    )


def build_retrieval_intent(
    task_text: str,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> RetrievalIntent:
    semantic_plan = build_question_semantic_plan(task_text, source_bundle)
    entity = semantic_plan.entity
    metric = semantic_plan.metric
    period = semantic_plan.period
    period_type = semantic_plan.period_type
    target_years = list(semantic_plan.target_years)
    publication_year_window = list(semantic_plan.publication_year_window)
    preferred_publication_years = list(semantic_plan.preferred_publication_years)
    granularity_requirement = semantic_plan.granularity_requirement
    document_family = _document_family(task_text, source_bundle, benchmark_overrides)
    aggregation_shape = _aggregation_shape(task_text)
    years = list(target_years) or _period_years(period, task_text)
    retrieval_metric = _primary_retrieval_metric(metric, aggregation_shape)
    qualifier_terms = list(semantic_plan.qualifier_terms)
    analysis_modes = _task_analysis_modes(task_text, benchmark_overrides)
    answer_mode, compute_policy, partial_answer_allowed = _classify_answer_mode(
        task_text,
        aggregation_shape,
        analysis_modes,
    )
    strategy, strategy_confidence, fallback_chain, join_requirements = _select_retrieval_strategy(
        task_text,
        source_bundle,
        metric,
        years,
        aggregation_shape,
        analysis_modes,
    )
    evidence_plan = _build_evidence_plan(
        task_text,
        source_bundle,
        metric,
        period,
        period_type,
        target_years,
        publication_year_window,
        preferred_publication_years,
        aggregation_shape,
        granularity_requirement,
        strategy,
        analysis_modes,
        join_requirements,
        semantic_plan.include_constraints,
        semantic_plan.exclude_constraints,
    )
    query_plan = _build_query_plan(
        document_family=document_family,
        entity=entity,
        metric=metric,
        retrieval_metric=retrieval_metric,
        period=period,
        period_type=period_type,
        publication_year_window=publication_year_window,
        preferred_publication_years=preferred_publication_years,
        granularity_requirement=granularity_requirement,
        include_constraints=semantic_plan.include_constraints,
        exclude_constraints=semantic_plan.exclude_constraints,
        source_bundle=source_bundle,
    )

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
    must_include_terms.extend(semantic_plan.include_constraints)
    must_include_terms = list(dict.fromkeys([item for item in must_include_terms if item]))
    source_constraint_policy = "soft" if (source_bundle.source_files_expected or source_bundle.source_files_found) else "off"

    policy = _benchmark_policy(benchmark_overrides)
    must_exclude_terms = list(policy.get("excluded_retrieval_terms", [])) if _officeqa_active(benchmark_overrides) else []
    must_exclude_terms.extend(semantic_plan.exclude_constraints)
    must_exclude_terms = list(dict.fromkeys([item for item in must_exclude_terms if item]))

    query_candidates = _active_query_candidates(
        query_plan,
        source_constraint_policy=source_constraint_policy,
    )
    return RetrievalIntent(
        entity=entity,
        metric=metric,
        period=period,
        period_type=period_type,
        target_years=target_years,
        publication_year_window=publication_year_window,
        preferred_publication_years=preferred_publication_years,
        source_constraint_policy=source_constraint_policy,
        granularity_requirement=granularity_requirement,
        document_family=document_family,
        aggregation_shape=aggregation_shape,
        analysis_modes=analysis_modes,
        answer_mode=answer_mode,
        compute_policy=compute_policy,
        partial_answer_allowed=partial_answer_allowed,
        strategy=strategy,
        strategy_confidence=strategy_confidence,
        evidence_requirements=[requirement.label for requirement in evidence_plan.requirements],
        fallback_chain=fallback_chain,
        join_requirements=join_requirements,
        evidence_plan=evidence_plan,
        include_constraints=semantic_plan.include_constraints,
        exclude_constraints=semantic_plan.exclude_constraints,
        decomposition_confidence=semantic_plan.confidence,
        decomposition_used_llm_fallback=semantic_plan.used_llm,
        semantic_plan=semantic_plan,
        query_plan=query_plan,
        must_include_terms=must_include_terms,
        must_exclude_terms=must_exclude_terms,
        query_candidates=query_candidates,
    )


def _structured_years_from_value(value: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            str(value.get("document_id", "")),
            str(value.get("citation", "")),
            str(value.get("page_locator", "")),
            str(value.get("table_locator", "")),
            str(value.get("row_label", "")),
            str(value.get("column_label", "")),
        ]
    )
    return re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)


def _structured_month_counts(values: list[dict[str, Any]]) -> dict[str, set[str]]:
    counts: dict[str, set[str]] = {}
    for value in values:
        lowered = str(value.get("row_label", "")).lower()
        month = next((name for name in _MONTH_NAMES if name in lowered), "")
        if not month:
            continue
        years = _structured_years_from_value(value)
        for year in years[:1]:
            counts.setdefault(year, set()).add(month)
    return counts


def predictive_evidence_gaps(
    retrieval_intent: RetrievalIntent,
    structured_evidence: dict[str, Any] | None,
) -> list[str]:
    payload = dict(structured_evidence or {})
    plan = retrieval_intent.evidence_plan
    values = [item for item in list(payload.get("values", [])) if isinstance(item, dict)]
    tables = [item for item in list(payload.get("tables", [])) if isinstance(item, dict)]
    page_chunks = [item for item in list(payload.get("page_chunks", [])) if isinstance(item, dict)]
    alignment_summary = dict(payload.get("alignment_summary", {}) or {})
    confidence_summary = dict(payload.get("structure_confidence_summary", {}) or {})
    gaps: list[str] = []

    if plan.requires_table_support and not (tables or values):
        gaps.append("table support")
    if plan.requires_text_support and not page_chunks:
        gaps.append("narrative support")
    if plan.required_years:
        found_years = {year for value in values for year in _structured_years_from_value(value)}
        if set(plan.required_years) - found_years:
            gaps.append("year coverage")
    if plan.required_month_coverage and plan.required_month_count:
        month_counts = _structured_month_counts(values)
        missing_months = [
            year
            for year in (plan.required_years or list(month_counts.keys()))
            if len(month_counts.get(year, set())) < min(12, plan.required_month_count)
        ]
        if missing_months:
            gaps.append("missing month coverage")
    if plan.requires_inflation_support:
        inflation_supported = any(
            any(token in " ".join(str(value.get(key, "")) for key in ("row_label", "column_label", "table_locator", "citation")).lower() for token in ("cpi", "inflation", "price index"))
            for value in values
        ) or any(
            any(token in str(chunk.get("text", "")).lower() for token in ("cpi", "inflation", "price index"))
            for chunk in page_chunks
        )
        if not inflation_supported:
            gaps.append("inflation support")
    if plan.join_keys and retrieval_intent.strategy in {"multi_table", "multi_document"} and len(tables) < 2:
        gaps.append("join-ready evidence")
    if plan.requires_cross_source_alignment:
        document_ids = {
            str(item.get("document_id", "")).strip()
            for item in [*values, *tables]
            if str(item.get("document_id", "")).strip()
        }
        aligned_document_count = int(alignment_summary.get("aligned_document_count", 0) or 0)
        if aligned_document_count < 2 and len(document_ids) < 2:
            gaps.append("cross-document alignment")
        if alignment_summary and not bool(alignment_summary.get("unit_consistent", True)):
            gaps.append("cross-document unit alignment")
        aligned_years = {str(year) for year in list(alignment_summary.get("aligned_years", []))}
        if plan.required_years and set(plan.required_years) - aligned_years:
            gaps.append("cross-document time alignment")
    has_confidence_signal = bool(confidence_summary) or any("structure_confidence" in value for value in values)
    avg_confidence = float(confidence_summary.get("avg_confidence", 1.0 if not has_confidence_signal else 0.0) or (1.0 if not has_confidence_signal else 0.0))
    max_confidence = float(confidence_summary.get("max_confidence", 1.0 if not has_confidence_signal else 0.0) or (1.0 if not has_confidence_signal else 0.0))
    if (tables or values) and has_confidence_signal:
        if avg_confidence < _STRUCTURE_CONFIDENCE_AVG_THRESHOLD or max_confidence < _STRUCTURE_CONFIDENCE_MAX_THRESHOLD:
            gaps.append("low-confidence structure")
    return list(dict.fromkeys(gaps))


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
