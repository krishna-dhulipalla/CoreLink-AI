"""
Prompt extraction helpers used during intake and context assembly.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

_URL_RE = re.compile(r"https?://[^\s\)\]\"',]+")
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9])")
_MONTH_NAME_DATE_RE = re.compile(
    r"\b(?:as of|on|dated?|for)\s+"
    r"((?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(?:as of|on|dated?|for)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


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

    percentages = _PERCENT_RE.findall(text or "")
    if percentages:
        facts["percentages"] = [float(value) / 100.0 for value in percentages[:12]]

    numbers = _NUMBER_RE.findall(text or "")
    if numbers:
        facts["numbers"] = [float(value) for value in numbers[:20]]

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
