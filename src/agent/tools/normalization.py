"""
Tool Result Normalization
=========================
Canonical normalization for staged-runtime tool outputs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.contracts import ToolQuality, ToolResult

_STRUCTURED_RESULTS_RE = re.compile(r"STRUCTURED_RESULTS:\s*(.+?)(?:\n---|\Z)", re.DOTALL)
_SEARCH_RESULT_RE = re.compile(
    r"\[(?P<rank>\d+)\]\s+(?P<title>.+?)\n\s*(?P<snippet>.+?)\n\s*URL:\s*(?P<url>https?://\S+)",
    re.DOTALL,
)


def _normalize_scalar(value: str) -> Any:
    raw = value.strip().strip("$")
    lowered = raw.lower()
    if lowered in {
        "credit",
        "debit",
        "fairly priced",
        "overpriced",
        "underpriced",
        "short_vol",
        "neutral",
        "weekly",
        "bi-weekly",
        "monthly",
        "quarterly",
        "leaps",
    }:
        return raw
    if raw.endswith("%"):
        try:
            return float(raw[:-1]) / 100.0
        except ValueError:
            return raw
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_structured_results(raw: str) -> dict[str, Any]:
    match = _STRUCTURED_RESULTS_RE.search(raw or "")
    if not match:
        return {}
    line = match.group(1).replace("\n", " ").strip()
    facts: dict[str, Any] = {}
    for part in line.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        facts[key.strip()] = _normalize_scalar(value)
    return facts


def _parse_strategy_analysis(raw: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    patterns = {
        "net_premium": r"Net Premium\s*:\s*([+-]?\d+(?:\.\d+)?)",
        "premium_direction": r"Net Premium\s*:\s*[+-]?\d+(?:\.\d+)?\s*\(([^)]+)\)",
        "total_delta": r"Total Delta\s*:\s*([+-]?\d+(?:\.\d+)?)",
        "total_gamma": r"Total Gamma\s*:\s*([+-]?\d+(?:\.\d+)?)",
        "total_theta_per_day": r"Total Theta\s*:\s*([+-]?\d+(?:\.\d+)?)",
        "total_vega_per_vol_point": r"Total Vega\s*:\s*([+-]?\d+(?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, raw or "", re.IGNORECASE)
        if match:
            facts[key] = _normalize_scalar(match.group(1))
    return facts


def _parse_reference_listing(raw: str) -> dict[str, Any]:
    documents = []
    for idx, match in enumerate(
        re.finditer(r"\d+\.\s+\[(?P<fmt>[A-Z]+)\]\s+(?P<url>https?://[^\s\)\]\"',]+)", raw or ""),
        start=1,
    ):
        url = match.group("url").strip()
        documents.append(
            {
                "document_id": re.sub(r"[^a-zA-Z0-9_]+", "_", url.rsplit("/", 1)[-1].split("?", 1)[0]).strip("_").lower()
                or f"document_{idx}",
                "citation": url,
                "format": match.group("fmt").strip().lower(),
            }
        )
    if not documents:
        urls = re.findall(r"https?://[^\s\)\]\"',]+", raw or "")
        documents = [
            {
                "document_id": re.sub(r"[^a-zA-Z0-9_]+", "_", url.rsplit("/", 1)[-1].split("?", 1)[0]).strip("_").lower()
                or f"document_{idx}",
                "citation": url,
                "format": "unknown",
            }
            for idx, url in enumerate(urls, start=1)
        ]
    return {"documents": documents, "document_count": len(documents)}


def _parse_document_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "\t" in stripped:
            rows.append([cell.strip() for cell in stripped.split("\t")])
        elif "," in stripped and len(stripped.split(",")) > 1:
            rows.append([cell.strip() for cell in stripped.split(",")])
        if len(rows) >= 20:
            break
    return rows


def _parse_file_fetch(raw: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    file_match = re.search(r"FILE:\s*(.+)", raw or "")
    format_match = re.search(r"FORMAT:\s*([A-Z0-9_]+)", raw or "")
    size_match = re.search(r"SIZE:\s*([0-9.]+)\s*KB", raw or "")
    window_match = re.search(r"\[(Pages|Rows)\s+([^\]]+)\]", raw or "")
    metadata: dict[str, Any] = {}
    if file_match:
        metadata["file_name"] = file_match.group(1).strip()
    if format_match:
        metadata["format"] = format_match.group(1).strip().lower()
    if size_match:
        metadata["size_kb"] = float(size_match.group(1))
    if window_match:
        metadata["window"] = f"{window_match.group(1)} {window_match.group(2)}"

    parts = re.split(r"\n[-\u2500]{10,}\n", raw or "", maxsplit=1)
    body = parts[1] if len(parts) > 1 else raw
    rows = _parse_document_rows(body)
    chunks: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    numeric_summaries: list[dict[str, Any]] = []
    if rows:
        headers = rows[0] if rows else []
        data_rows = rows[1:21] if len(rows) > 1 else []
        if headers:
            tables.append(
                {
                    "locator": metadata.get("window", "body"),
                    "headers": headers,
                    "rows": data_rows[:12],
                    "citation": "",
                }
            )
        numeric_cells = []
        numeric_columns: dict[int, list[float]] = {}
        for row in data_rows:
            for idx, cell in enumerate(row):
                if isinstance(cell, str) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cell.strip()):
                    value = float(cell)
                    numeric_cells.append(value)
                    numeric_columns.setdefault(idx, []).append(value)
        numeric_summaries.append({"metric": "row_count", "value": len(data_rows)})
        numeric_summaries.append({"metric": "column_count", "value": len(headers)})
        if numeric_cells:
            numeric_summaries.append({"metric": "numeric_cell_count", "value": len(numeric_cells)})
        for idx, values in list(numeric_columns.items())[:4]:
            header = headers[idx] if idx < len(headers) else f"col_{idx}"
            numeric_summaries.append(
                {
                    "metric": f"{header}_range",
                    "value": {"min": min(values), "max": max(values)},
                }
            )
        preview_rows = rows[: min(len(rows), 4)]
        chunks.append(
            {
                "locator": metadata.get("window", "body"),
                "kind": "table_preview",
                "text": "\n".join(",".join(row) for row in preview_rows)[:280].strip(),
                "citation": "",
            }
        )
    else:
        json_candidate = body.strip()
        if json_candidate.startswith("{") or json_candidate.startswith("["):
            try:
                parsed = json.loads(json_candidate)
                chunks.append(
                    {
                        "locator": metadata.get("window", "body"),
                        "kind": "json_preview",
                        "text": json.dumps(parsed, ensure_ascii=True)[:280],
                        "citation": "",
                    }
                )
            except Exception:
                pass
    excerpt = re.sub(r"\s+", " ", body).strip()
    if excerpt and not chunks:
        chunks.append(
            {
                "locator": metadata.get("window", "body"),
                "kind": "text_excerpt",
                "text": excerpt[:280],
                "citation": "",
            }
        )

    if not metadata and not tables and not numeric_summaries:
        return {}

    document_id = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        str(metadata.get("file_name", "document")).split("?", 1)[0],
    ).strip("_").lower() or "document"
    facts["document_id"] = document_id
    facts["metadata"] = metadata
    facts["chunks"] = chunks
    facts["tables"] = tables
    facts["numeric_summaries"] = numeric_summaries
    return facts


def _parse_search_results(raw: str) -> dict[str, Any]:
    results = []
    for match in _SEARCH_RESULT_RE.finditer(raw or ""):
        results.append(
            {
                "rank": int(match.group("rank")),
                "title": match.group("title").strip(),
                "snippet": re.sub(r"\s+", " ", match.group("snippet")).strip(),
                "url": match.group("url").strip(),
            }
        )
    return {"results": results}


def _parse_options_chain(raw: str) -> dict[str, Any]:
    header_match = re.search(
        r"Options Chain:\s*S=(?P<spot>[0-9.]+),\s*[\u03c3\u03c3]?=?(?P<sigma>[0-9.]+)%?,\s*T=(?P<days>\d+)d,\s*r=(?P<rate>[0-9.]+)%",
        raw or "",
    )
    chain = []
    for line in (raw or "").splitlines():
        if not re.match(r"^\s*\d", line):
            continue
        cleaned = line.replace("\u2190ATM", " ATM")
        parts = cleaned.split()
        if len(parts) < 5:
            continue
        record = {
            "strike": float(parts[0]),
            "call_price": float(parts[1]),
            "put_price": float(parts[2]),
            "call_delta": float(parts[3]),
            "put_delta": float(parts[4]),
            "is_atm": "ATM" in parts,
        }
        chain.append(record)
    facts: dict[str, Any] = {"chain": chain}
    if header_match:
        facts.update(
            {
                "spot": float(header_match.group("spot")),
                "implied_volatility": float(header_match.group("sigma").rstrip("%")) / 100.0,
                "days_to_expiry": int(header_match.group("days")),
                "risk_free_rate": float(header_match.group("rate")) / 100.0,
            }
        )
    return facts


def _parse_expirations(raw: str) -> dict[str, Any]:
    expirations = []
    for line in (raw or "").splitlines():
        match = re.search(r"(\d+)d\s+(\d{4}-\d{2}-\d{2})\s+\[([^\]]+)\]", line)
        if not match:
            continue
        expirations.append(
            {
                "days": int(match.group(1)),
                "date": match.group(2),
                "label": match.group(3),
            }
        )
    return {"expirations": expirations}


def _parse_iv_surface(raw: str) -> dict[str, Any]:
    rows = []
    for line in (raw or "").splitlines():
        match = re.match(r"^\s*(\d+)d\s+([0-9.]+)%\s+([0-9.]+)\s+([0-9.]+)\s+([+-]?[0-9.]+)", line)
        if not match:
            continue
        rows.append(
            {
                "days_to_expiry": int(match.group(1)),
                "implied_volatility": float(match.group(2)) / 100.0,
                "call_price": float(match.group(3)),
                "put_price": float(match.group(4)),
                "theta_per_day": float(match.group(5)),
            }
        )
    return {"surface": rows}


def _parse_json_payload(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def _default_quality(
    *,
    is_synthetic: bool = False,
    is_estimated: bool = False,
    cache_hit: bool = False,
    missing_fields: list[str] | None = None,
) -> ToolQuality:
    return ToolQuality(
        is_synthetic=is_synthetic,
        is_estimated=is_estimated,
        cache_hit=cache_hit,
        missing_fields=list(missing_fields or []),
    )


def _looks_like_tool_envelope(payload: dict[str, Any]) -> bool:
    return "facts" in payload and (
        "source" in payload or "quality" in payload or "errors" in payload
    )


def normalize_tool_output(tool_name: str, raw_content: Any, args: dict[str, Any]) -> ToolResult:
    if isinstance(raw_content, list):
        raw_content = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in raw_content
        )
    if isinstance(raw_content, dict):
        if _looks_like_tool_envelope(raw_content):
            payload = dict(raw_content)
            payload.setdefault("type", tool_name)
            payload.setdefault("facts", {})
            payload.setdefault("assumptions", args)
            payload.setdefault("source", {"tool": tool_name})
            payload.setdefault("quality", _default_quality().model_dump())
            payload.setdefault("errors", [])
            return ToolResult.model_validate(payload)
        if raw_content.get("error"):
            return ToolResult(
                type=tool_name,
                facts={},
                assumptions=args,
                source={"tool": tool_name},
                quality=_default_quality(),
                errors=[str(raw_content["error"])],
            )
        return ToolResult(
            type=tool_name,
            facts=raw_content,
            assumptions=args,
            source={"tool": tool_name},
            quality=_default_quality(),
            errors=[],
        )

    text = str(raw_content or "").strip()
    if not text:
        return ToolResult(
            type=tool_name,
            facts={},
            assumptions=args,
            source={"tool": tool_name},
            quality=_default_quality(),
            errors=["Empty tool output."],
        )
    if text.startswith("Error") or text.startswith("[HTTP") or text.startswith("[Network error]") or text.startswith("[Unexpected error"):
        return ToolResult(
            type=tool_name,
            facts={},
            assumptions=args,
            source={"tool": tool_name},
            quality=_default_quality(),
            errors=[text],
        )

    parsed_payload = _parse_json_payload(text)
    if parsed_payload and _looks_like_tool_envelope(parsed_payload):
        parsed_payload.setdefault("type", tool_name)
        parsed_payload.setdefault("facts", {})
        parsed_payload.setdefault("assumptions", args)
        parsed_payload.setdefault("source", {"tool": tool_name})
        parsed_payload.setdefault("quality", _default_quality().model_dump())
        parsed_payload.setdefault("errors", [])
        return ToolResult.model_validate(parsed_payload)

    facts = parsed_payload
    if not facts:
        facts = _parse_structured_results(text)
    if not facts and tool_name == "analyze_strategy":
        facts = _parse_strategy_analysis(text)
    elif not facts and tool_name == "get_options_chain":
        facts = _parse_options_chain(text)
    elif not facts and tool_name == "get_expirations":
        facts = _parse_expirations(text)
    elif not facts and tool_name == "get_iv_surface":
        facts = _parse_iv_surface(text)
    elif not facts and tool_name == "list_reference_files":
        facts = _parse_reference_listing(text)
    elif not facts and tool_name == "fetch_reference_file":
        facts = _parse_file_fetch(text)
    elif not facts and tool_name == "internet_search":
        facts = _parse_search_results(text)
    elif not facts and tool_name == "calculator":
        facts = {"result": _normalize_scalar(text)}

    errors = [] if facts else ["Unstructured tool output could not be normalized into machine-usable facts."]
    if tool_name == "fetch_reference_file":
        parse_error_match = re.search(r"\[(CSV|PDF|Excel|Word|JSON) parse error:[^\]]+\]", text)
        if parse_error_match:
            errors = [parse_error_match.group(0)]
    if facts and tool_name == "fetch_reference_file":
        citation = str(args.get("url", "")).strip()
        if citation:
            facts["citation"] = citation
            for chunk in facts.get("chunks", []):
                if isinstance(chunk, dict):
                    chunk.setdefault("citation", citation)
            for table in facts.get("tables", []):
                if isinstance(table, dict):
                    table.setdefault("citation", citation)
    return ToolResult(
        type=tool_name,
        facts=facts,
        assumptions=args,
        source={"tool": tool_name, "raw_preview": text[:240]},
        quality=_default_quality(),
        errors=errors,
    )
