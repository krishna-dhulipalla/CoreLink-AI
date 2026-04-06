"""
Prompt extraction helpers used during intake and context assembly.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.contracts import QueryPlan, QuestionDecomposition, SourceBundle
from agent.model_config import invoke_structured_output

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
)
_EXCLUSION_PATTERNS = (
    r"excluding\s+(.+?)(?:,| and | but |\.|$)",
    r"except\s+(.+?)(?:,| and | but |\.|$)",
    r"not including\s+(.+?)(?:,| and | but |\.|$)",
    r"without\s+(.+?)(?:,| and | but |\.|$)",
)


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
    include_constraints = _extract_include_constraints(task_text)
    exclude_constraints = _extract_exclude_constraints(task_text)
    qualifier_terms = _dedupe_strings([*include_constraints, *exclude_constraints], limit=6)
    entity = _extract_entity_identity(task_text, source_bundle, metric)

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
        granularity_requirement=granularity_requirement,
        include_constraints=include_constraints,
        exclude_constraints=exclude_constraints,
        qualifier_terms=qualifier_terms,
        confidence=confidence,
        query_plan=QueryPlan(),
    )

def _merge_decomposition(primary: QuestionDecomposition, fallback: QuestionDecomposition) -> QuestionDecomposition:
    return QuestionDecomposition(
        entity=primary.entity or fallback.entity,
        metric=primary.metric or fallback.metric,
        period=primary.period or fallback.period,
        granularity_requirement=primary.granularity_requirement or fallback.granularity_requirement,
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
                "include constraints, exclude constraints, qualifier terms, and confidence. "
                "Do not provide reasoning."
            )
        ),
        HumanMessage(
            content=_normalize_space(
                f"TASK={task_text}\nFOCUS_QUERY={source_bundle.focus_query}\nTARGET_PERIOD={source_bundle.target_period}\n"
                f"ENTITIES={source_bundle.entities}\nSOURCE_FILES={source_bundle.source_files_expected[:8]}"
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
