"""Persistent manifest helpers for the local OfficeQA corpus."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

_CORPUS_ENV_NAMES = (
    "OFFICEQA_CORPUS_DIR",
    "REFERENCE_CORPUS_DIR",
    "DOCUMENT_CORPUS_DIR",
)
_CORPUS_CANDIDATES = (
    "treasury_bulletins_parsed",
    "officeqa/treasury_bulletins_parsed",
    "data/treasury_bulletins_parsed",
    "treasury_bulletin_pdfs",
    "officeqa/treasury_bulletin_pdfs",
    "reference_corpus",
    "documents",
)
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".html", ".xml", ".tsv", ".pdf"}
_INDEX_DIR_NAME = ".officeqa_index"
_MANIFEST_FILENAME = "manifest.jsonl"
_METADATA_FILENAME = "index_metadata.json"
_INDEX_SCHEMA_VERSION = 1
_MAX_FILES = 4000
_MONTHS = (
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
_UNIT_HINTS = ("thousand", "million", "billion", "percent", "dollars", "cents", "nominal")


def normalize_source_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def resolve_officeqa_corpus_root(raw: str | None = None) -> Path | None:
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate

    for env_name in _CORPUS_ENV_NAMES:
        env_value = os.getenv(env_name, "").strip()
        if not env_value:
            continue
        candidate = Path(env_value).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate

    cwd = Path.cwd()
    for candidate in _CORPUS_CANDIDATES:
        path = cwd / candidate
        if path.exists() and path.is_dir():
            return path
    return None


def resolve_officeqa_index_dir(corpus_root: Path, raw: str | None = None, create: bool = False) -> Path:
    target = raw or os.getenv("OFFICEQA_INDEX_DIR", "").strip()
    index_dir = Path(target).expanduser() if target else corpus_root / _INDEX_DIR_NAME
    if create:
        index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def manifest_path(index_dir: Path) -> Path:
    return index_dir / _MANIFEST_FILENAME


def metadata_path(index_dir: Path) -> Path:
    return index_dir / _METADATA_FILENAME


def officeqa_index_schema_version() -> int:
    return _INDEX_SCHEMA_VERSION


def iter_officeqa_files(corpus_root: Path, max_files: int = _MAX_FILES) -> list[Path]:
    files: list[Path] = []
    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        files.append(path)
        if len(files) >= max_files:
            break
    return files


def read_officeqa_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            parts = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(part for part in parts if part).strip()
        except Exception:
            return ""

    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, ensure_ascii=True, indent=2)
        except Exception:
            return raw
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows: list[list[str]] = []
        for line in raw.splitlines()[:400]:
            try:
                rows.append(next(csv.reader([line], delimiter=delimiter)))
            except Exception:
                rows.append([part.strip() for part in line.split(delimiter)])
        return "\n".join(delimiter.join(cell.strip() for cell in row) for row in rows)
    return raw


def _preview_text(text: str, limit: int = 5000) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _json_payload(path: Path) -> Any | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _append_limited(values: list[str], candidate: Any, limit: int = 40) -> None:
    if candidate is None:
        return
    normalized = re.sub(r"\s+", " ", str(candidate)).strip()
    if not normalized or normalized in values:
        return
    if len(values) < limit:
        values.append(normalized)


def _extract_json_metadata(payload: Any) -> dict[str, list[str]]:
    page_markers: list[str] = []
    section_titles: list[str] = []
    table_headers: list[str] = []
    row_labels: list[str] = []
    unit_hints: list[str] = []

    def visit(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                normalized_key = str(key).strip().lower()
                if normalized_key in {"page", "page_number", "page_num"} and isinstance(value, (int, float, str)):
                    _append_limited(page_markers, value)
                if normalized_key in {"section", "section_title", "title", "heading"}:
                    _append_limited(section_titles, value)
                if normalized_key in {"headers", "header", "columns", "column_headers"}:
                    if isinstance(value, list):
                        for item in value:
                            _append_limited(table_headers, item)
                    else:
                        _append_limited(table_headers, value)
                if normalized_key in {"row_label", "row_labels", "label"}:
                    if isinstance(value, list):
                        for item in value:
                            _append_limited(row_labels, item)
                    else:
                        _append_limited(row_labels, value)
                if normalized_key in {"unit", "units"}:
                    if isinstance(value, list):
                        for item in value:
                            _append_limited(unit_hints, item)
                    else:
                        _append_limited(unit_hints, value)
                if normalized_key == "rows" and isinstance(value, list):
                    for row in value[:30]:
                        if isinstance(row, list) and row:
                            _append_limited(row_labels, row[0])
                        elif isinstance(row, dict):
                            for row_key in ("label", "row_label", "name"):
                                if row_key in row:
                                    _append_limited(row_labels, row.get(row_key))
                visit(value, normalized_key)
        elif isinstance(node, list):
            for item in node[:200]:
                visit(item, parent_key)

    visit(payload)
    return {
        "page_markers": page_markers,
        "section_titles": section_titles,
        "table_headers": table_headers,
        "row_labels": row_labels,
        "unit_hints": unit_hints,
    }


def _extract_text_metadata(text: str, path: Path) -> dict[str, Any]:
    lowered_text = (text or "").lower()
    lowered_path = path.as_posix().lower()
    years = list(dict.fromkeys(re.findall(r"\b((?:19|20)\d{2})\b", f"{lowered_path} {lowered_text}")[:12]))
    month_coverage = [month for month in _MONTHS if month in lowered_text]
    unit_hints = [hint for hint in _UNIT_HINTS if hint in lowered_text]
    page_markers = list(dict.fromkeys(re.findall(r"\bpage[s]?\s+(\d{1,4})\b", lowered_text)[:40]))
    section_titles: list[str] = []
    table_headers: list[str] = []
    row_labels: list[str] = []
    for line in (text or "").splitlines()[:300]:
        compact = re.sub(r"\s+", " ", line).strip()
        if not compact:
            continue
        if re.match(r"^[A-Z][A-Z0-9 ,&()./%-]{8,}$", compact):
            _append_limited(section_titles, compact, limit=20)
        if "|" in compact:
            parts = [part.strip() for part in compact.split("|") if part.strip()]
            if len(parts) >= 2:
                for part in parts[:8]:
                    _append_limited(table_headers, part, limit=32)
        if "\t" in compact:
            parts = [part.strip() for part in compact.split("\t") if part.strip()]
            if len(parts) >= 2:
                _append_limited(row_labels, parts[0], limit=40)
        elif "," in compact and compact.count(",") >= 2:
            parts = [part.strip() for part in compact.split(",") if part.strip()]
            if len(parts) >= 2:
                _append_limited(row_labels, parts[0], limit=40)
    return {
        "years": years,
        "month_coverage": month_coverage,
        "page_markers": page_markers,
        "section_titles": section_titles,
        "table_headers": table_headers,
        "row_labels": row_labels,
        "unit_hints": unit_hints,
        "is_treasury_bulletin": "treasury bulletin" in lowered_text or "treasury" in lowered_path or "bulletin" in lowered_path,
        "has_month_names": bool(month_coverage),
        "has_table_like_rows": bool(table_headers or row_labels),
    }


def _normalized_numeric_values(text: str, unit_hints: list[str], limit: int = 40) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    multiplier_map = {
        "thousand": 1_000.0,
        "million": 1_000_000.0,
        "billion": 1_000_000_000.0,
    }
    for match in re.finditer(
        r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>thousand|million|billion|percent|dollars|cents)?",
        text or "",
        re.IGNORECASE,
    ):
        raw_value = str(match.group("value") or "").strip()
        if not raw_value:
            continue
        try:
            parsed = float(raw_value.replace(",", ""))
        except Exception:
            continue
        unit = str(match.group("unit") or "").strip().lower()
        canonical_unit = unit or (unit_hints[0] if unit_hints else "")
        normalized_value = parsed
        if unit in multiplier_map:
            normalized_value = parsed * multiplier_map[unit]
        elif unit == "cents":
            normalized_value = parsed / 100.0
        normalized.append(
            {
                "raw": raw_value,
                "unit": canonical_unit,
                "normalized_value": round(normalized_value, 6),
            }
        )
        if len(normalized) >= limit:
            break
    return normalized


def _validation_flags(entry: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if not entry.get("preview_text"):
        flags.append("empty_text")
    if entry.get("file_format") == "pdf" and not entry.get("preview_text"):
        flags.append("pdf_extract_failed")
    if entry.get("file_format") == "json" and not entry.get("table_headers") and not entry.get("row_labels"):
        flags.append("structured_json_without_tables")
    if entry.get("is_treasury_bulletin") and not entry.get("years"):
        flags.append("missing_years")
    if entry.get("has_table_like_rows") and not entry.get("normalized_numeric_values"):
        flags.append("table_without_numeric_values")
    if entry.get("has_month_names") and len(entry.get("month_coverage", [])) < 2:
        flags.append("partial_month_coverage")
    return flags


def build_manifest_entry(path: Path, corpus_root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(corpus_root).as_posix()
    text = read_officeqa_document_text(path)
    preview = _preview_text(text)
    payload = _json_payload(path)
    text_metadata = _extract_text_metadata(text, path)
    json_metadata = _extract_json_metadata(payload) if payload is not None else {
        "page_markers": [],
        "section_titles": [],
        "table_headers": [],
        "row_labels": [],
        "unit_hints": [],
    }
    source_aliases = sorted({
        normalize_source_name(relative_path),
        normalize_source_name(path.name),
        normalize_source_name(path.stem),
    })
    entry = {
        "document_id": normalize_source_name(relative_path) or "document",
        "relative_path": relative_path,
        "source_key": normalize_source_name(path.stem),
        "source_aliases": source_aliases,
        "file_name": path.name,
        "file_format": path.suffix.lower().lstrip(".") or "text",
        "size_bytes": path.stat().st_size,
        "years": text_metadata["years"],
        "month_coverage": text_metadata["month_coverage"],
        "page_markers": list(dict.fromkeys([*json_metadata["page_markers"], *text_metadata["page_markers"]]))[:40],
        "section_titles": list(dict.fromkeys([*json_metadata["section_titles"], *text_metadata["section_titles"]]))[:40],
        "table_headers": list(dict.fromkeys([*json_metadata["table_headers"], *text_metadata["table_headers"]]))[:60],
        "row_labels": list(dict.fromkeys([*json_metadata["row_labels"], *text_metadata["row_labels"]]))[:80],
        "unit_hints": list(dict.fromkeys([*json_metadata["unit_hints"], *text_metadata["unit_hints"]]))[:20],
        "is_treasury_bulletin": bool(text_metadata["is_treasury_bulletin"]),
        "has_month_names": bool(text_metadata["has_month_names"]),
        "has_table_like_rows": bool(text_metadata["has_table_like_rows"]),
        "normalized_numeric_values": _normalized_numeric_values(text, list(dict.fromkeys([*json_metadata["unit_hints"], *text_metadata["unit_hints"]]))[:20]),
        "preview_text": preview,
    }
    entry["validation_flags"] = _validation_flags(entry)
    entry["parse_status"] = "partial" if entry["validation_flags"] else "ok"
    return entry


def write_officeqa_manifest(entries: list[dict[str, Any]], corpus_root: Path, index_dir: Path) -> dict[str, Any]:
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path(index_dir)
    with manifest_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    metadata = {
        "index_schema_version": _INDEX_SCHEMA_VERSION,
        "corpus_root": str(corpus_root),
        "manifest_path": str(manifest_file),
        "document_count": len(entries),
        "years": sorted({year for entry in entries for year in entry.get("years", [])}),
        "partial_document_count": sum(1 for entry in entries if entry.get("parse_status") == "partial"),
    }
    metadata_file = metadata_path(index_dir)
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    return metadata


def load_officeqa_manifest(corpus_root: Path | None = None, index_dir: Path | None = None) -> list[dict[str, Any]]:
    root = corpus_root or resolve_officeqa_corpus_root()
    if root is None:
        return []
    resolved_index_dir = index_dir or resolve_officeqa_index_dir(root)
    manifest_file = manifest_path(resolved_index_dir)
    if not manifest_file.exists():
        return []
    records: list[dict[str, Any]] = []
    with manifest_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            compact = line.strip()
            if not compact:
                continue
            try:
                records.append(json.loads(compact))
            except Exception:
                continue
    return records


def load_officeqa_index_metadata(corpus_root: Path | None = None, index_dir: Path | None = None) -> dict[str, Any]:
    root = corpus_root or resolve_officeqa_corpus_root()
    if root is None:
        return {}
    resolved_index_dir = index_dir or resolve_officeqa_index_dir(root)
    metadata_file = metadata_path(resolved_index_dir)
    if not metadata_file.exists():
        return {}
    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def match_source_files_to_records(source_files: list[str], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    record_by_alias: dict[str, dict[str, Any]] = {}
    for record in records:
        for alias in record.get("source_aliases", []):
            record_by_alias[str(alias)] = record
        record_by_alias[str(record.get("source_key", ""))] = record

    for source_file in source_files:
        normalized = normalize_source_name(source_file)
        record = record_by_alias.get(normalized)
        if record is None and "." in source_file:
            record = record_by_alias.get(normalize_source_name(Path(source_file).stem))
        matches.append(
            {
                "source_file": source_file,
                "matched": record is not None,
                "document_id": str((record or {}).get("document_id", "")),
                "relative_path": str((record or {}).get("relative_path", "")),
            }
        )
    return matches


def validate_manifest_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for record in records:
        for flag in record.get("validation_flags", []):
            issues.append(
                {
                    "document_id": str(record.get("document_id", "")),
                    "relative_path": str(record.get("relative_path", "")),
                    "flag": str(flag),
                }
            )
    return {
        "document_count": len(records),
        "issue_count": len(issues),
        "issues": issues,
    }
