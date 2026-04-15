"""
Prompt extraction helpers used during intake and context assembly.
"""

from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from engine.agent.contracts import QueryPlan, QuestionDecomposition, QuestionSemanticPlan, SourceBundle
from engine.agent.model_config import (
    get_model_name_for_officeqa_control,
    get_model_runtime_kwargs_for_officeqa_control,
    invoke_structured_output,
)
from engine.agent.prompts import FINANCIAL_SEMANTIC_PLAN_SYSTEM

_URL_RE = re.compile(r"https?://[^\s\)\]\"',]+")
_TITLE_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&'()./-]*\s+){1,7}[A-Z][A-Za-z0-9&'()./-]*\b"
)
_MONTH_NAME_DATE_RE = re.compile(
    r"\b(?:as of|on|dated?|for)\s+"
    r"((?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(?:as of|on|dated?|for)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_GENERIC_SOURCE_ENTITIES = {
    "annual report",
    "bulletin",
    "document",
    "narrative discussion",
    "official report",
    "report",
    "treasury bulletin",
}
_ENTITY_TEXT_PATTERN = r"[A-Za-z0-9 .,'&/\-]+?"
_INCLUSION_PATTERNS = (
    r"(specifically only the reported values)",
    r"(all individual calendar months)",
    r"(monthly series)",
    r"(narrative discussion)",
    r"(reported values only)",
    r"should include\s+(.+?)(?:,| and | but |\.|$)",
    r"must include\s+(.+?)(?:,| and | but |\.|$)",
)
_EXCLUSION_PATTERNS = (
    r"excluding\s+(.+?)(?:,| but |\.|$)",
    r"except\s+(.+?)(?:,| but |\.|$)",
    r"not including\s+(.+?)(?:,| but |\.|$)",
    r"without\s+(.+?)(?:,| but |\.|$)",
    r"should(?:\s+not|n['’]t)\s+contain\s+(.+?)(?:,| but |\.|$)",
    r"should(?:\s+not|n['’]t)\s+include\s+(.+?)(?:,| but |\.|$)",
    r"must not contain\s+(.+?)(?:,| but |\.|$)",
)


def _officeqa_corpus_start_year() -> int:
    raw = str(os.getenv("OFFICEQA_CORPUS_START_YEAR", "1939") or "1939").strip()
    try:
        year = int(raw)
    except ValueError:
        return 1939
    return max(1900, min(2100, year))


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _dedupe_strings(values: list[str], *, limit: int | None = None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _normalize_space(raw).strip(" ,.;:!?")
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
        if limit is not None and len(ordered) >= limit:
            break
    return ordered


def _normalize_financial_phrase(value: str) -> str:
    cleaned = _normalize_space(value).strip(" ,.;:!?")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\baccording to\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:using|based on)\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s+in\s+(?:the\s+)?(?:fy|fiscal year|calendar year)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+\b(?:19|20)\d{2}\b.*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,.;:!?")


def _sanitize_source_cue_entity(entity: str, include_constraints: list[str]) -> str:
    cleaned = _normalize_financial_phrase(entity)
    if not cleaned:
        return ""
    if cleaned.lower() in _GENERIC_SOURCE_ENTITIES:
        return ""
    include_text = " ".join(str(item or "") for item in include_constraints).lower()
    if "according to the treasury bulletin" in include_text and cleaned.lower() == "treasury bulletin":
        return ""
    return cleaned


def _extract_year_scope(task_text: str, source_bundle: SourceBundle) -> str:
    if source_bundle.target_period:
        return _normalize_space(source_bundle.target_period)
    years = _YEAR_RE.findall(task_text or "")
    return " ".join(dict.fromkeys(years[:4]))


def _extract_granularity_requirement(task_text: str) -> str:
    lowered = _normalize_space(task_text).lower()
    if (
        "all individual calendar months" in lowered
        or "monthly series" in lowered
        or "calendar months" in lowered
        or "monthly expenditures" in lowered
        or "monthly receipts" in lowered
        or "monthly outlays" in lowered
        or "each month" in lowered
        or "for each month" in lowered
        or ("monthly" in lowered and re.search(r"\b(?:19|20)\d{2}\b", lowered))
    ):
        return "monthly_series"
    if "calendar year" in lowered:
        return "calendar_year"
    if "fiscal year" in lowered or re.search(r"\bfy\s+\d{4}\b", lowered):
        return "fiscal_year"
    if "narrative discussion" in lowered or "reason was given" in lowered:
        return "narrative_support"
    return "point_lookup"


def _period_type(task_text: str, granularity_requirement: str) -> str:
    lowered = _normalize_space(task_text).lower()
    if granularity_requirement == "monthly_series":
        return "monthly_series"
    if granularity_requirement == "calendar_year":
        return "calendar_year"
    if granularity_requirement == "fiscal_year":
        return "fiscal_year"
    if granularity_requirement == "narrative_support":
        return "narrative_support"
    if re.search(r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b", lowered):
        return "point_in_time"
    return "point_lookup"


def _target_years(period: str, task_text: str, source_bundle: SourceBundle) -> list[str]:
    years = _YEAR_RE.findall(" ".join([period or "", task_text or "", source_bundle.target_period or ""]))
    return list(dict.fromkeys(years[:4]))


def _publication_scope_explicit(task_text: str, source_bundle: SourceBundle) -> bool:
    lowered = _normalize_space(task_text).lower()
    if len(source_bundle.source_files_expected) == 1:
        return True
    if re.search(
        r"\b(?:in|from|using|according to)\s+(?:the\s+)?(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{4}\s+treasury bulletin\b",
        lowered,
    ):
        return True
    if re.search(r"\b(?:issue|edition)\s+of\s+(?:the\s+)?treasury bulletin\b", lowered):
        return True
    return False


def _acceptable_publication_lag_years(granularity_requirement: str, period_type: str) -> int:
    if granularity_requirement == "monthly_series" or period_type == "monthly_series":
        return 1
    if granularity_requirement in {"calendar_year", "fiscal_year"} or period_type in {"calendar_year", "fiscal_year"}:
        return 1
    return 0


def _publication_year_preferences(
    target_years: list[str],
    granularity_requirement: str,
    period_type: str,
    publication_scope_explicit: bool,
) -> tuple[list[str], list[str], int, bool, bool]:
    if not target_years:
        return [], [], 0, False, False
    years = []
    for year in target_years:
        try:
            years.append(int(year))
        except ValueError:
            continue
    if not years:
        return [], [], 0, False, False

    acceptable_lag = _acceptable_publication_lag_years(granularity_requirement, period_type)
    corpus_start_year = _officeqa_corpus_start_year()
    retrospective_evidence_allowed = bool(years) and not publication_scope_explicit
    retrospective_evidence_required = min(years) < corpus_start_year and retrospective_evidence_allowed

    if retrospective_evidence_required:
        preferred = [str(corpus_start_year + offset) for offset in range(0, max(acceptable_lag, 3) + 1)]
        window = [str(corpus_start_year + offset) for offset in range(0, max(acceptable_lag, 3) + 1)]
        return (
            _dedupe_strings(preferred, limit=12),
            _dedupe_strings(window, limit=12),
            acceptable_lag,
            retrospective_evidence_allowed,
            retrospective_evidence_required,
        )

    if granularity_requirement in {"calendar_year", "fiscal_year"} or period_type in {"calendar_year", "fiscal_year"}:
        preferred = [str(year + 1) for year in years] + [str(year) for year in years] + [str(year - 1) for year in years]
    elif granularity_requirement == "monthly_series" or period_type == "monthly_series":
        preferred = [str(year) for year in years] + [str(year + 1) for year in years] + [str(year - 1) for year in years]
    else:
        preferred = [str(year) for year in years] + [str(year + 1) for year in years] + [str(year - 1) for year in years]

    window = [str(year - 1) for year in years] + [str(year) for year in years] + [str(year + 1) for year in years]
    return (
        _dedupe_strings(preferred, limit=12),
        _dedupe_strings(window, limit=12),
        acceptable_lag,
        retrospective_evidence_allowed,
        retrospective_evidence_required,
    )


def _extract_include_constraints(task_text: str) -> list[str]:
    found: list[str] = []
    for pattern in _INCLUSION_PATTERNS:
        for match in re.finditer(pattern, task_text or "", re.IGNORECASE):
            found.append(match.group(1))
    if "according to the treasury bulletin" in (task_text or "").lower():
        found.append("according to the Treasury Bulletin")
    return _dedupe_strings(found, limit=6)


def _extract_exclude_constraints(task_text: str) -> list[str]:
    found: list[str] = []
    for pattern in _EXCLUSION_PATTERNS:
        for match in re.finditer(pattern, task_text or "", re.IGNORECASE):
            found.append(match.group(1))
    return _dedupe_strings(found, limit=6)


def _extract_expected_answer_unit_basis(task_text: str, metric: str) -> str:
    lowered = _normalize_space(f"{task_text} {metric}").lower()
    if "percent" in lowered or "%" in lowered:
        return "percent"
    if re.search(r"\bin\s+millions?\s+of\s+(?:nominal\s+)?dollars?\b", lowered):
        return "millions_nominal_dollars"
    return ""


def _extract_metric_identity(task_text: str) -> str:
    lowered = _normalize_space(task_text).lower()
    ordered_checks = (
        ("absolute percent change", "absolute percent change"),
        ("absolute difference", "absolute difference"),
        ("inflation-adjusted weighted average", "weighted average expenditures"),
        ("weighted average expenditures", "weighted average expenditures"),
        ("weighted average", "weighted average"),
        ("standard deviation", "standard deviation"),
        ("regression trend", "regression trend"),
        ("forecast", "forecast"),
        ("value at risk", "value at risk"),
        ("public debt outstanding", "public debt outstanding"),
        ("total monthly expenditures", "total expenditures"),
        ("monthly expenditures", "expenditures"),
        ("total monthly receipts", "total receipts"),
        ("total expenditures", "total expenditures"),
        ("expenditures", "expenditures"),
        ("receipts", "receipts"),
        ("debt outlook", "debt outlook"),
    )
    for needle, label in ordered_checks:
        if needle in lowered:
            return label

    for pattern in (
        rf"what (?:was|is|were) (?:the )?({_ENTITY_TEXT_PATTERN})(?:\s+in\s+(?:fy|fiscal year|calendar year|\b(?:19|20)\d{{2}}\b)|\?|$)",
        rf"what were the ({_ENTITY_TEXT_PATTERN}) of {_ENTITY_TEXT_PATTERN}(?:\s+in\s+(?:fy|fiscal year|calendar year|\b(?:19|20)\d{{2}}\b)|\?|$)",
        rf"reason was given for (?:the )?({_ENTITY_TEXT_PATTERN})(?:\s+in\s+\b(?:19|20)\d{{2}}\b|\?|$)",
    ):
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            return _normalize_financial_phrase(match.group(1))
    return ""


def _entity_from_source_bundle(source_bundle: SourceBundle) -> str:
    for raw in source_bundle.entities:
        entity = _normalize_financial_phrase(raw)
        if entity and entity.lower() not in _GENERIC_SOURCE_ENTITIES:
            return entity
    return ""


def _extract_entity_identity(task_text: str, source_bundle: SourceBundle, metric: str) -> str:
    bundled = _entity_from_source_bundle(source_bundle)
    if bundled:
        return bundled

    patterns = [
        rf"{re.escape(metric)}\s+(?:for|of)\s+({_ENTITY_TEXT_PATTERN})(?:\s+in\s+(?:fy|fiscal year|calendar year|\b(?:19|20)\d{{2}}\b)|\?|$)"
        for metric in (
            "total expenditures",
            "expenditures",
            "receipts",
            "public debt outstanding",
        )
    ]
    patterns.extend(
        (
            rf"(?:weighted average|regression trend|forecast)(?:\s+the)?\s+({_ENTITY_TEXT_PATTERN})\s+series(?:\s+for\s+\b(?:19|20)\d{{2}}\b|\?|$)",
            rf"for the ({_ENTITY_TEXT_PATTERN})(?:\s+in\s+(?:fy|fiscal year|calendar year|\b(?:19|20)\d{{2}}\b)|\?|$)",
            rf"of the ({_ENTITY_TEXT_PATTERN})(?:\s+in\s+(?:fy|fiscal year|calendar year|\b(?:19|20)\d{{2}}\b)|\?|$)",
        )
    )
    for pattern in patterns:
        match = re.search(pattern, task_text or "", re.IGNORECASE)
        if match:
            entity = _normalize_financial_phrase(match.group(1))
            if entity and entity.lower() not in _GENERIC_SOURCE_ENTITIES:
                return entity

    if metric == "debt outlook":
        return "debt outlook"
    return ""


def _rule_based_decomposition(task_text: str, source_bundle: SourceBundle) -> QuestionDecomposition:
    metric = _extract_metric_identity(task_text)
    period = _extract_year_scope(task_text, source_bundle)
    granularity_requirement = _extract_granularity_requirement(task_text)
    period_type = _period_type(task_text, granularity_requirement)
    target_years = _target_years(period, task_text, source_bundle)
    publication_scope_explicit = _publication_scope_explicit(task_text, source_bundle)
    (
        preferred_publication_years,
        publication_year_window,
        acceptable_publication_lag_years,
        retrospective_evidence_allowed,
        retrospective_evidence_required,
    ) = _publication_year_preferences(
        target_years,
        granularity_requirement,
        period_type,
        publication_scope_explicit,
    )
    include_constraints = _extract_include_constraints(task_text)
    exclude_constraints = _extract_exclude_constraints(task_text)
    expected_answer_unit_basis = _extract_expected_answer_unit_basis(task_text, metric)
    qualifier_terms = _dedupe_strings([*include_constraints, *exclude_constraints], limit=6)
    entity = _sanitize_source_cue_entity(
        _extract_entity_identity(task_text, source_bundle, metric),
        include_constraints,
    )

    confidence = 0.2
    if metric:
        confidence += 0.28
    if period:
        confidence += 0.22
    if entity or metric in {"public debt outstanding", "absolute percent change", "absolute difference"}:
        confidence += 0.18
    if granularity_requirement != "point_lookup":
        confidence += 0.12
    if include_constraints or exclude_constraints:
        confidence += 0.08
    if metric and entity and metric.lower() in entity.lower():
        confidence -= 0.08
    confidence = max(0.0, min(0.95, confidence))

    return QuestionDecomposition(
        entity=entity,
        metric=metric,
        period=period,
        period_type=period_type,
        target_years=target_years,
        publication_year_window=publication_year_window,
        preferred_publication_years=preferred_publication_years,
        acceptable_publication_lag_years=acceptable_publication_lag_years,
        retrospective_evidence_allowed=retrospective_evidence_allowed,
        retrospective_evidence_required=retrospective_evidence_required,
        publication_scope_explicit=publication_scope_explicit,
        granularity_requirement=granularity_requirement,
        expected_answer_unit_basis=expected_answer_unit_basis,
        include_constraints=include_constraints,
        exclude_constraints=exclude_constraints,
        qualifier_terms=qualifier_terms,
        confidence=confidence,
        query_plan=QueryPlan(),
    )

def _merge_decomposition(primary: QuestionDecomposition, fallback: QuestionDecomposition) -> QuestionDecomposition:
    return QuestionDecomposition(
        entity=_sanitize_source_cue_entity(primary.entity or fallback.entity, [*primary.include_constraints, *fallback.include_constraints]),
        metric=primary.metric or fallback.metric,
        period=primary.period or fallback.period,
        period_type=primary.period_type or fallback.period_type,
        target_years=list(dict.fromkeys([*primary.target_years, *fallback.target_years]))[:4],
        publication_year_window=list(dict.fromkeys([*primary.publication_year_window, *fallback.publication_year_window]))[:12],
        preferred_publication_years=list(dict.fromkeys([*primary.preferred_publication_years, *fallback.preferred_publication_years]))[:12],
        acceptable_publication_lag_years=max(primary.acceptable_publication_lag_years, fallback.acceptable_publication_lag_years),
        retrospective_evidence_allowed=bool(primary.retrospective_evidence_allowed or fallback.retrospective_evidence_allowed),
        retrospective_evidence_required=bool(primary.retrospective_evidence_required or fallback.retrospective_evidence_required),
        publication_scope_explicit=bool(primary.publication_scope_explicit or fallback.publication_scope_explicit),
        granularity_requirement=primary.granularity_requirement or fallback.granularity_requirement,
        expected_answer_unit_basis=primary.expected_answer_unit_basis or fallback.expected_answer_unit_basis,
        include_constraints=_dedupe_strings([*primary.include_constraints, *fallback.include_constraints], limit=6),
        exclude_constraints=_dedupe_strings([*primary.exclude_constraints, *fallback.exclude_constraints], limit=6),
        qualifier_terms=_dedupe_strings([*primary.qualifier_terms, *fallback.qualifier_terms], limit=6),
        confidence=max(primary.confidence, min(0.9, fallback.confidence)),
        used_llm_fallback=bool(fallback.used_llm_fallback),
        query_plan=fallback.query_plan if any(fallback.query_plan.model_dump().values()) else primary.query_plan,
    )


def _fallback_decomposition(task_text: str, source_bundle: SourceBundle) -> QuestionDecomposition | None:
    schema = QuestionDecomposition
    messages = [
        SystemMessage(
            content=(
                "Extract a typed financial-document question decomposition. "
                    "Return only the target entity or program, metric identity, period, granularity requirement, "
                    "expected answer unit basis, include constraints, exclude constraints, qualifier terms, and confidence. "
                    "Do not provide reasoning."
                )
        ),
        HumanMessage(
            content=_normalize_space(
                f"TASK={task_text}\nFOCUS_QUERY={source_bundle.focus_query}\nTARGET_PERIOD={source_bundle.target_period}\n"
                f"ENTITIES={source_bundle.entities}\nSOURCE_FILES={source_bundle.source_files_expected}"
            )
        ),
    ]
    try:
        parsed, _ = invoke_structured_output("profiler", schema, messages, temperature=0, max_tokens=240)
        candidate = schema.model_validate(parsed)
        candidate.used_llm_fallback = True
        return candidate
    except Exception:
        return None


def extract_question_decomposition(
    task_text: str,
    source_bundle: SourceBundle,
    *,
    allow_llm_fallback: bool = False,
) -> QuestionDecomposition:
    decomposition = _rule_based_decomposition(task_text, source_bundle)
    if not allow_llm_fallback:
        return decomposition
    if decomposition.confidence >= 0.58:
        return decomposition
    fallback = _fallback_decomposition(task_text, source_bundle)
    if fallback is None:
        return decomposition
    return _merge_decomposition(decomposition, fallback)


def _semantic_ambiguity_flags(task_text: str, decomposition: QuestionDecomposition) -> list[str]:
    task_lower = str(task_text or "").lower()
    flags: list[str] = []
    if decomposition.confidence < 0.8:
        flags.append("low_rule_confidence")
    if decomposition.granularity_requirement in {"calendar_year", "fiscal_year", "monthly_series"}:
        flags.append("temporal_publication_lag_risk")
    if decomposition.include_constraints or decomposition.exclude_constraints:
        flags.append("constraint_sensitive")
    if any(token in task_lower for token in ("forecast", "regression", "correlation", "standard deviation", "weighted average", "value at risk", "var ")):
        flags.append("advanced_financial_reasoning")
    if not decomposition.entity or not decomposition.metric:
        flags.append("missing_core_slot")
    return list(dict.fromkeys(flags))


def _needs_semantic_plan_llm(task_text: str, decomposition: QuestionDecomposition) -> bool:
    flags = _semantic_ambiguity_flags(task_text, decomposition)
    if any(
        flag in flags
        for flag in (
            "missing_core_slot",
            "advanced_financial_reasoning",
            "constraint_sensitive",
            "temporal_publication_lag_risk",
        )
    ):
        return True
    lowered = _normalize_space(task_text).lower()
    numeric_contract = any(
        token in lowered
        for token in (
            "calculate",
            "compute",
            "sum",
            "total",
            "average",
            "difference",
            "percent change",
            "standard deviation",
            "regression",
            "forecast",
            "weighted average",
            "value at risk",
            "variance",
        )
    )
    if numeric_contract and not decomposition.expected_answer_unit_basis and re.search(
        r"\bin\s+(?:millions?|billions?|thousands?)\b|\bpercent\b|%",
        lowered,
    ):
        return True
    return decomposition.confidence < 0.8 or len(flags) >= 2


def _semantic_contract_periods(
    decomposition: QuestionDecomposition,
) -> tuple[str, str, str, str]:
    evidence_period = decomposition.period or "unspecified"
    publication_period = " ".join(decomposition.preferred_publication_years[:3]) if decomposition.preferred_publication_years else ""
    aggregation_period = decomposition.granularity_requirement or decomposition.period_type or "point_lookup"
    display_unit_basis = decomposition.expected_answer_unit_basis or ""
    return evidence_period, publication_period, aggregation_period, display_unit_basis


def _semantic_completeness_audit(
    task_text: str,
    decomposition: QuestionDecomposition,
) -> tuple[bool, list[str]]:
    lowered = _normalize_space(task_text).lower()
    gaps: list[str] = []
    numeric_contract = any(
        token in lowered
        for token in (
            "calculate",
            "compute",
            "sum",
            "total",
            "average",
            "difference",
            "percent change",
            "standard deviation",
            "regression",
            "forecast",
            "weighted average",
            "value at risk",
            "variance",
        )
    )
    if not decomposition.metric:
        gaps.append("missing_metric")
    if not decomposition.period:
        gaps.append("missing_period")
    if numeric_contract and not decomposition.expected_answer_unit_basis and re.search(
        r"\bin\s+(?:millions?|billions?|thousands?)\b|\bpercent\b|%",
        lowered,
    ):
        gaps.append("missing_answer_unit_basis")
    if numeric_contract and any(mode in lowered for mode in ("weighted average", "regression", "forecast", "standard deviation", "variance")):
        if not decomposition.entity:
            gaps.append("missing_core_entity")
    if decomposition.include_constraints and not decomposition.qualifier_terms:
        gaps.append("include_constraints_not_promoted")
    if decomposition.exclude_constraints and not decomposition.qualifier_terms:
        gaps.append("exclude_constraints_not_promoted")
    return len(gaps) == 0, list(dict.fromkeys(gaps))


def _semantic_plan_from_decomposition(
    decomposition: QuestionDecomposition,
    *,
    task_text: str,
    rationale: str = "",
    used_llm: bool = False,
    model_name: str = "",
) -> QuestionSemanticPlan:
    evidence_period, publication_period, aggregation_period, display_unit_basis = _semantic_contract_periods(decomposition)
    completeness_ok, completeness_gaps = _semantic_completeness_audit(task_text, decomposition)
    return QuestionSemanticPlan(
        entity=_sanitize_source_cue_entity(decomposition.entity, decomposition.include_constraints),
        metric=decomposition.metric,
        period=decomposition.period,
        period_type=decomposition.period_type,
        evidence_period=evidence_period,
        publication_period=publication_period,
        aggregation_period=aggregation_period,
        display_unit_basis=display_unit_basis,
        target_years=list(decomposition.target_years),
        publication_year_window=list(decomposition.publication_year_window),
        preferred_publication_years=list(decomposition.preferred_publication_years),
        acceptable_publication_lag_years=decomposition.acceptable_publication_lag_years,
        retrospective_evidence_allowed=decomposition.retrospective_evidence_allowed,
        retrospective_evidence_required=decomposition.retrospective_evidence_required,
        publication_scope_explicit=decomposition.publication_scope_explicit,
        granularity_requirement=decomposition.granularity_requirement,
        expected_answer_unit_basis=decomposition.expected_answer_unit_basis,
        include_constraints=list(decomposition.include_constraints),
        exclude_constraints=list(decomposition.exclude_constraints),
        qualifier_terms=list(decomposition.qualifier_terms),
        ambiguity_flags=_semantic_ambiguity_flags("", decomposition),
        completeness_ok=completeness_ok,
        completeness_gaps=completeness_gaps,
        rationale=rationale,
        confidence=decomposition.confidence,
        used_llm=used_llm,
        model_name=model_name,
    )


def _merge_semantic_plan(primary: QuestionSemanticPlan, fallback: QuestionSemanticPlan) -> QuestionSemanticPlan:
    # P12.1: Pre-compute merged gaps so we can enforce completeness_ok contract.
    # Never allow completeness_ok=True to coexist with any missing-core-slot gap —
    # the LLM may optimistically claim completeness even when slots are unresolved.
    _merged_gaps = _dedupe_strings([*primary.completeness_gaps, *fallback.completeness_gaps], limit=8)
    _has_core_gap = any("missing" in g for g in _merged_gaps)
    _completeness_ok = bool((primary.completeness_ok or fallback.completeness_ok) and not _has_core_gap)
    return QuestionSemanticPlan(
        entity=_sanitize_source_cue_entity(primary.entity or fallback.entity, [*primary.include_constraints, *fallback.include_constraints]),
        metric=primary.metric or fallback.metric,
        period=primary.period or fallback.period,
        period_type=primary.period_type or fallback.period_type,
        evidence_period=primary.evidence_period or fallback.evidence_period,
        publication_period=primary.publication_period or fallback.publication_period,
        aggregation_period=primary.aggregation_period or fallback.aggregation_period,
        display_unit_basis=primary.display_unit_basis or fallback.display_unit_basis,
        target_years=list(dict.fromkeys([*primary.target_years, *fallback.target_years]))[:4],
        publication_year_window=list(dict.fromkeys([*primary.publication_year_window, *fallback.publication_year_window]))[:12],
        preferred_publication_years=list(dict.fromkeys([*primary.preferred_publication_years, *fallback.preferred_publication_years]))[:12],
        acceptable_publication_lag_years=max(primary.acceptable_publication_lag_years, fallback.acceptable_publication_lag_years),
        retrospective_evidence_allowed=bool(primary.retrospective_evidence_allowed or fallback.retrospective_evidence_allowed),
        retrospective_evidence_required=bool(primary.retrospective_evidence_required or fallback.retrospective_evidence_required),
        publication_scope_explicit=bool(primary.publication_scope_explicit or fallback.publication_scope_explicit),
        granularity_requirement=primary.granularity_requirement or fallback.granularity_requirement,
        expected_answer_unit_basis=primary.expected_answer_unit_basis or fallback.expected_answer_unit_basis,
        include_constraints=_dedupe_strings([*primary.include_constraints, *fallback.include_constraints], limit=6),
        exclude_constraints=_dedupe_strings([*primary.exclude_constraints, *fallback.exclude_constraints], limit=6),
        qualifier_terms=_dedupe_strings([*primary.qualifier_terms, *fallback.qualifier_terms], limit=6),
        ambiguity_flags=_dedupe_strings([*primary.ambiguity_flags, *fallback.ambiguity_flags], limit=8),
        completeness_ok=_completeness_ok,
        completeness_gaps=_merged_gaps,
        rationale=primary.rationale or fallback.rationale,
        confidence=max(primary.confidence, min(0.92, fallback.confidence)),
        used_llm=bool(primary.used_llm or fallback.used_llm),
        model_name=fallback.model_name or primary.model_name,
    )


def _fallback_semantic_plan(task_text: str, source_bundle: SourceBundle, decomposition: QuestionDecomposition) -> QuestionSemanticPlan | None:
    model_name = get_model_name_for_officeqa_control("semantic_plan_llm")
    runtime_kwargs = get_model_runtime_kwargs_for_officeqa_control("semantic_plan_llm")
    messages = [
        SystemMessage(content=FINANCIAL_SEMANTIC_PLAN_SYSTEM),
        HumanMessage(
            content=_normalize_space(
                f"TASK={task_text}\nFOCUS_QUERY={source_bundle.focus_query}\nTARGET_PERIOD={source_bundle.target_period}\n"
                f"ENTITIES={source_bundle.entities}\nSOURCE_FILES={source_bundle.source_files_expected}\n"
                f"RULE_ENTITY={decomposition.entity}\nRULE_METRIC={decomposition.metric}\nRULE_PERIOD={decomposition.period}\n"
                f"RULE_GRANULARITY={decomposition.granularity_requirement}\nRULE_ANSWER_UNIT_BASIS={decomposition.expected_answer_unit_basis}\nRULE_INCLUDE={decomposition.include_constraints}\n"
                f"RULE_EXCLUDE={decomposition.exclude_constraints}\nRULE_QUALIFIERS={decomposition.qualifier_terms}\n"
                f"RULE_COMPLETENESS={_semantic_completeness_audit(task_text, decomposition)}"
            )
        ),
    ]
    try:
        parsed, resolved_model = invoke_structured_output(
            "profiler",
            QuestionSemanticPlan,
            messages,
            temperature=0,
            max_tokens=260,
            model_name_override=model_name,
            runtime_kwargs_override=runtime_kwargs,
        )
        candidate = QuestionSemanticPlan.model_validate(parsed)
        candidate.used_llm = True
        candidate.model_name = resolved_model
        return candidate
    except Exception:
        return None


def build_question_semantic_plan(task_text: str, source_bundle: SourceBundle) -> QuestionSemanticPlan:
    decomposition = extract_question_decomposition(task_text, source_bundle, allow_llm_fallback=True)
    base_plan = _semantic_plan_from_decomposition(
        decomposition,
        task_text=task_text,
        rationale="rule_based_semantic_plan",
        used_llm=bool(decomposition.used_llm_fallback),
    )
    base_plan.ambiguity_flags = _semantic_ambiguity_flags(task_text, decomposition)
    if not _needs_semantic_plan_llm(task_text, decomposition):
        return base_plan
    fallback = _fallback_semantic_plan(task_text, source_bundle, decomposition)
    if fallback is None:
        return base_plan
    merged = _merge_semantic_plan(base_plan, fallback)
    if not merged.rationale:
        merged.rationale = "semantic_plan_llm"
    merged.completeness_ok, merged.completeness_gaps = _semantic_completeness_audit(
        task_text,
        QuestionDecomposition(
            entity=merged.entity,
            metric=merged.metric,
            period=merged.period,
            period_type=merged.period_type,
            target_years=list(merged.target_years),
            publication_year_window=list(merged.publication_year_window),
            preferred_publication_years=list(merged.preferred_publication_years),
            acceptable_publication_lag_years=merged.acceptable_publication_lag_years,
            retrospective_evidence_allowed=merged.retrospective_evidence_allowed,
            retrospective_evidence_required=merged.retrospective_evidence_required,
            publication_scope_explicit=merged.publication_scope_explicit,
            granularity_requirement=merged.granularity_requirement,
            expected_answer_unit_basis=merged.expected_answer_unit_basis,
            include_constraints=list(merged.include_constraints),
            exclude_constraints=list(merged.exclude_constraints),
            qualifier_terms=list(merged.qualifier_terms),
            confidence=merged.confidence,
        ),
    )
    # P12.1: Enforce contract rule — completeness_ok must be False if ANY core slot is missing.
    if merged.completeness_gaps and any("missing" in g for g in merged.completeness_gaps):
        merged.completeness_ok = False
    return merged


def extract_urls(text: str) -> list[str]:
    urls = []
    for match in _URL_RE.findall(text or ""):
        clean = match.rstrip(".,;)")
        if clean not in urls:
            urls.append(clean)
    return urls


def extract_as_of_date(text: str) -> str | None:
    match = _ISO_DATE_RE.search(text or "")
    if match:
        return match.group(1)

    match = _MONTH_NAME_DATE_RE.search(text or "")
    if not match:
        return None

    raw = match.group(1).strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_formulas(text: str) -> list[str]:
    formulas: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" in stripped and any(ch.isalpha() for ch in stripped):
            formulas.append(stripped[:300])
        elif stripped.startswith("\\text{") or stripped.startswith("$"):
            formulas.append(stripped[:300])
    deduped: list[str] = []
    for formula in formulas:
        if formula not in deduped:
            deduped.append(formula)
    return deduped[:20]


def parse_markdown_tables(text: str) -> list[dict[str, Any]]:
    lines = (text or "").splitlines()
    tables: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines) - 1:
        header = lines[idx].strip()
        separator = lines[idx + 1].strip()
        if "|" not in header or "|" not in separator or "---" not in separator:
            idx += 1
            continue
        headers = [col.strip() for col in header.strip("|").split("|")]
        rows: list[dict[str, str]] = []
        j = idx + 2
        while j < len(lines):
            row_line = lines[j].strip()
            if "|" not in row_line:
                break
            values = [col.strip() for col in row_line.strip("|").split("|")]
            if len(values) != len(headers):
                break
            rows.append(dict(zip(headers, values)))
            j += 1
        if rows:
            tables.append({"headers": headers, "rows": rows[:20]})
        idx = j
    return tables[:10]


def extract_entities(text: str) -> list[str]:
    candidates: list[str] = []
    normalized = text or ""

    for match in _TITLE_ENTITY_RE.findall(normalized):
        cleaned = re.sub(r"\s+", " ", match).strip(" ,.:;?()")
        if len(cleaned) < 6:
            continue
        lowered = cleaned.lower()
        if lowered in {
            "user question",
            "related data",
            "formula list",
            "output format",
            "annual report",
        }:
            continue
        if cleaned not in candidates:
            candidates.append(cleaned)
    for match in re.findall(r"\b[A-Z]{2,6}\b", text or ""):
        if match not in candidates:
            candidates.append(match)
    for match in re.findall(r"\b\d{4}\.HK\b", text or ""):
        if match not in candidates:
            candidates.append(match)
    return candidates[:10]


def extract_inline_facts(text: str, *, labeled_json_extractor=None) -> dict[str, Any]:
    lowered = (text or "").lower()
    facts: dict[str, Any] = {}
    as_of_date = extract_as_of_date(text)
    if as_of_date:
        facts["as_of_date"] = as_of_date

    if "iv percentile" in lowered:
        match = re.search(r"iv percentile[^0-9]*(\d+(?:\.\d+)?)", lowered)
        if match:
            facts["iv_percentile"] = float(match.group(1))

    iv_match = re.search(r"\biv\b[^0-9]*(\d+(?:\.\d+)?)\s*%", lowered)
    hv_match = re.search(r"historical volatility[^0-9]*(\d+(?:\.\d+)?)\s*%", lowered)
    if iv_match:
        facts["implied_volatility"] = float(iv_match.group(1)) / 100.0
    if hv_match:
        facts["historical_volatility"] = float(hv_match.group(1)) / 100.0

    for label, key in (
        ("Portfolio JSON", "portfolio_positions"),
        ("Returns JSON", "returns_series"),
        ("Metrics JSON", "risk_metrics_input"),
        ("Limits JSON", "limit_constraints"),
        ("Factors JSON", "factor_map"),
        ("Peers JSON", "peer_set"),
    ):
        parsed = labeled_json_extractor(text, label) if labeled_json_extractor else None
        if parsed is not None:
            facts[key] = parsed

    return facts


def derive_market_snapshot(task_text: str, inline_facts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot: dict[str, Any] = {}
    derived: dict[str, Any] = {}

    if "implied_volatility" in inline_facts:
        snapshot["implied_volatility"] = inline_facts["implied_volatility"]
    if "historical_volatility" in inline_facts:
        snapshot["historical_volatility"] = inline_facts["historical_volatility"]
    if "iv_percentile" in inline_facts:
        snapshot["iv_percentile"] = inline_facts["iv_percentile"]
    if "as_of_date" in inline_facts:
        snapshot["as_of_date"] = inline_facts["as_of_date"]

    if "implied_volatility" in snapshot and "historical_volatility" in snapshot:
        derived["iv_premium"] = round(
            float(snapshot["implied_volatility"]) - float(snapshot["historical_volatility"]),
            4,
        )
        derived["vol_bias"] = (
            "short_vol"
            if derived["iv_premium"] > 0 and float(snapshot.get("iv_percentile", 0)) >= 50
            else "neutral"
        )

    lowered = (task_text or "").lower()
    if inline_facts.get("as_of_date"):
        derived["time_sensitive"] = True
    elif any(
        token in lowered
        for token in (
            "latest",
            "today",
            "recent",
            "as of",
            "look up",
            "search",
            "source-backed",
            "current price",
            "current filing",
            "current market",
        )
    ):
        derived["time_sensitive"] = True

    return snapshot, derived
