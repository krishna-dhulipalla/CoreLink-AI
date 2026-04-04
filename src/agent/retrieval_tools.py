"""Built-in corpus retrieval tools for document-heavy benchmarks."""

from __future__ import annotations

import csv
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from agent.benchmarks.officeqa_index import (
    officeqa_index_available,
    resolve_indexed_corpus_document,
    search_officeqa_corpus_index,
)
from agent.benchmarks.officeqa_manifest import (
    iter_officeqa_files,
    read_officeqa_document_text,
    resolve_officeqa_corpus_root,
)
from agent.tools.table_normalization import normalize_dense_table_grid, normalize_flat_table
from agent.tools.tsr_fallback import select_dense_table_normalization

_MAX_FILES = 4000
_MONTH_TOKENS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "its", "his", "her", "their",
    "this", "that", "these", "those",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "if", "than", "too",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "also", "just",
})
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_HTML_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(r"<(th|td)\b([^>]*)>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
_HTML_COLSPAN_RE = re.compile(r'colspan\s*=\s*["\']?(\d+)', re.IGNORECASE)
_HTML_ROWSPAN_RE = re.compile(r'rowspan\s*=\s*["\']?(\d+)', re.IGNORECASE)
_HTML_QUERY_PREVIEW_CHARS = 2500
_MAX_EXTRACTED_TABLES = 24
_PAGE_REF_RE = re.compile(r"^[A-Z]?-?\d+(?:-\d+)?(?:\s+\d+(?:-\d+)?)*\.?$")


class _OfficeQATableExtractionTimeout(RuntimeError):
    """Raised when OfficeQA table extraction exceeds the configured wall-clock budget."""


def local_corpus_available() -> bool:
    return _resolve_corpus_root() is not None


def _resolve_corpus_root() -> Path | None:
    return resolve_officeqa_corpus_root()


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alpha-numeric tokens, filtering stop words."""
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP_WORDS and len(t) > 1]


def _table_extraction_timeout_seconds() -> float:
    raw = os.getenv("OFFICEQA_TABLE_EXTRACTION_TIMEOUT_SECONDS", "20").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 20.0
    return max(1.0, value)


def _ensure_extraction_budget(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise _OfficeQATableExtractionTimeout(
            f"OfficeQA table extraction exceeded time budget during {stage}. "
            "Increase OFFICEQA_TABLE_EXTRACTION_TIMEOUT_SECONDS if needed."
        )


def _document_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return re.sub(r"[^a-z0-9_]+", "_", relative.lower()).strip("_") or "document"


def _read_file_text(path: Path) -> str:
    return read_officeqa_document_text(path)


def _iter_corpus_files(root: Path) -> list[Path]:
    return iter_officeqa_files(root, max_files=_MAX_FILES)


def _resolve_corpus_target(root: Path, document_id: str = "", path: str = "") -> Path | None:
    root = root.resolve()
    target: Path | None = None
    if officeqa_index_available(root):
        resolved = resolve_indexed_corpus_document(document_id=document_id, path=path, corpus_root=root)
        if resolved is not None:
            candidate = (root / str(resolved.get("relative_path", ""))).resolve()
            if candidate.exists() and candidate.is_file() and _is_within_root(candidate, root):
                target = candidate
    if path:
        candidate = (root / path).resolve()
        if candidate.exists() and candidate.is_file() and _is_within_root(candidate, root):
            target = candidate
    if target is None and document_id:
        for candidate in _iter_corpus_files(root):
            if _document_id(candidate, root) == document_id:
                target = candidate
                break
    return target


def _best_snippet(text: str, query: str, snippet_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return ""
    query_tokens = [token for token in _tokenize(query) if len(token) > 2][:6]
    if not query_tokens:
        return compact[:snippet_chars]

    lowered = compact.lower()
    first_hit = -1
    for token in query_tokens:
        idx = lowered.find(token)
        if idx != -1 and (first_hit == -1 or idx < first_hit):
            first_hit = idx
    if first_hit == -1:
        return compact[:snippet_chars]

    start = max(0, first_hit - snippet_chars // 4)
    end = min(len(compact), start + snippet_chars)
    return compact[start:end]


def _query_years(query: str) -> list[str]:
    return re.findall(r"\b((?:19|20)\d{2})\b", query or "")


def _query_has_monthly_shape(query: str) -> bool:
    lowered = (query or "").lower()
    return "all individual calendar months" in lowered or "monthly" in lowered or "total sum of these values" in lowered


def _document_metadata(text: str, path: Path) -> dict[str, Any]:
    lowered_text = (text or "").lower()
    lowered_path = path.as_posix().lower()
    years = list(dict.fromkeys(re.findall(r"\b((?:19|20)\d{2})\b", f"{lowered_path} {lowered_text}")[:8]))
    return {
        "years": years,
        "is_treasury_bulletin": "treasury" in lowered_path or "bulletin" in lowered_path or "treasury bulletin" in lowered_text,
        "has_month_names": any(month in lowered_text for month in _MONTH_TOKENS),
        "has_table_like_rows": "\t" in text or sum(1 for line in text.splitlines() if "," in line) >= 3,
    }


def _score_document(text: str, query: str, path: Path) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    path_text = path.as_posix().lower()
    document_tokens = set(_tokenize(text))
    path_tokens = set(_tokenize(path_text))
    metadata = _document_metadata(text, path)
    unique_hits = {token for token in query_tokens if token in document_tokens or token in path_tokens}
    if not unique_hits:
        return 0.0
    score = float(len(unique_hits))
    for token in query_tokens:
        if token in path_tokens:
            score += 0.5
    stem_tokens = set(_tokenize(path.stem))
    if stem_tokens and stem_tokens.intersection(document_tokens):
        score += 0.2
    for year in _query_years(query):
        if year in metadata["years"] or year in document_tokens or year in path_tokens:
            score += 1.0
    if metadata["is_treasury_bulletin"] and "treasury bulletin" in (query or "").lower():
        score += 1.4
    if _query_has_monthly_shape(query):
        if metadata["has_month_names"]:
            score += 1.2
        if metadata["has_table_like_rows"]:
            score += 0.8
    return score


def _text_chunks(text: str, max_chars: int = 1200) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    if not paragraphs:
        compact = re.sub(r"\s+", " ", text or "").strip()
        return [compact[i:i + max_chars] for i in range(0, len(compact), max_chars) if compact[i:i + max_chars]]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph[:max_chars]
    if current:
        chunks.append(current)
    return chunks


def _extract_numeric_summaries(text: str) -> list[dict[str, Any]]:
    values = [float(match) for match in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text or "")[:200]]
    if not values:
        return []
    return [
        {"metric": "numeric_value_count", "value": len(values)},
        {"metric": "numeric_range", "value": {"min": min(values), "max": max(values)}},
    ]


def _row_numeric_summaries(headers: list[str], rows: list[list[str]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    numeric_cells = 0
    for col_idx, header in enumerate(headers[:12]):
        values: list[float] = []
        for row in rows:
            if col_idx >= len(row):
                continue
            compact = str(row[col_idx]).replace(",", "").replace("%", "").strip()
            if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact):
                continue
            values.append(float(compact))
            numeric_cells += 1
        if values:
            metric = re.sub(r"[^a-z0-9]+", "_", header.lower()).strip("_") or f"column_{col_idx + 1}"
            summaries.append({"metric": metric, "value": {"min": min(values), "max": max(values)}})
    if rows:
        summaries.insert(0, {"metric": "row_count", "value": len(rows)})
    if headers:
        summaries.insert(1 if rows else 0, {"metric": "column_count", "value": len(headers)})
    if numeric_cells:
        summaries.append({"metric": "numeric_cell_count", "value": numeric_cells})
    return summaries


def _coerce_table(
    headers: list[str],
    rows: list[list[str]],
    citation: str,
    locator: str,
    unit_hint: str = "",
    *,
    page_locator: str = "",
    context_text: str = "",
    canonical_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_headers = [str(header).strip() for header in headers if str(header).strip()]
    normalized_rows = [[str(cell).strip() for cell in row] for row in rows if any(str(cell).strip() for cell in row)]
    payload = {
        "locator": locator,
        "headers": normalized_headers,
        "rows": normalized_rows,
        "citation": citation,
        "unit_hint": unit_hint,
    }
    if page_locator:
        payload["page_locator"] = page_locator
    if context_text:
        payload["context_text"] = context_text
    if canonical_table:
        payload["canonical_table"] = canonical_table
    return payload


def _html_cell_text(fragment: str) -> str:
    cleaned = re.sub(r"<br\s*/?>", "\n", fragment or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"</p\s*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _page_locator_from_payload(payload: dict[str, Any]) -> str:
    for entry in list(payload.get("bbox") or [])[:4]:
        if not isinstance(entry, dict):
            continue
        page_id = entry.get("page_id")
        if isinstance(page_id, int) and page_id > 0:
            return f"page {page_id}"
    page = payload.get("page")
    if isinstance(page, int) and page > 0:
        return f"page {page}"
    return ""


def _extract_tables_from_html_string(
    html_content: str,
    citation: str,
    *,
    locator: str,
    page_locator: str = "",
    unit_hint: str = "",
    context_text: str = "",
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for table_index, match in enumerate(_HTML_TABLE_RE.finditer(html_content or ""), start=1):
        _ensure_extraction_budget(deadline, "html_table_scan")
        row_blocks = _HTML_ROW_RE.findall(match.group(1))
        if len(row_blocks) < 2:
            continue
        raw_rows: list[list[dict[str, Any]]] = []
        pending_rowspans: dict[int, dict[str, Any]] = {}
        for row_index, row_html in enumerate(row_blocks):
            _ensure_extraction_budget(deadline, "html_row_scan")
            raw_cells = list(_HTML_CELL_RE.findall(row_html))
            row_cells: list[dict[str, Any]] = []
            col_index = 0
            cell_index = 0
            while cell_index < len(raw_cells) or pending_rowspans:
                _ensure_extraction_budget(deadline, "html_cell_expand")
                if col_index in pending_rowspans:
                    carried = dict(pending_rowspans[col_index]["cell"])
                    carried["from_rowspan"] = True
                    row_cells.append(carried)
                    pending_rowspans[col_index]["remaining"] -= 1
                    if pending_rowspans[col_index]["remaining"] <= 0:
                        pending_rowspans.pop(col_index, None)
                    col_index += 1
                    continue
                if cell_index >= len(raw_cells):
                    if pending_rowspans:
                        stale_indexes = [idx for idx in pending_rowspans if idx < col_index]
                        for idx in stale_indexes:
                            pending_rowspans.pop(idx, None)
                        if not pending_rowspans:
                            break
                        next_pending = min(pending_rowspans)
                        while col_index < next_pending:
                            row_cells.append({"text": "", "is_header": False, "origin_id": ""})
                            col_index += 1
                        if col_index in pending_rowspans:
                            continue
                    break
                tag_name, attrs, cell_html = raw_cells[cell_index]
                cell_index += 1
                cell_text = _html_cell_text(cell_html)
                colspan_match = _HTML_COLSPAN_RE.search(attrs or "")
                rowspan_match = _HTML_ROWSPAN_RE.search(attrs or "")
                colspan = max(1, int(colspan_match.group(1))) if colspan_match else 1
                rowspan = max(1, int(rowspan_match.group(1))) if rowspan_match else 1
                origin_id = f"t{table_index}_r{row_index}_c{col_index}"
                for offset in range(colspan):
                    cell = {
                        "text": cell_text,
                        "is_header": str(tag_name).lower() == "th",
                        "origin_id": origin_id,
                        "colspan": colspan,
                        "rowspan": rowspan,
                    }
                    row_cells.append(cell)
                    if rowspan > 1:
                        pending_rowspans[col_index + offset] = {"cell": cell, "remaining": rowspan - 1}
                col_index += colspan
            if any(_html_cell_text(cell.get("text", "")) for cell in row_cells):
                raw_rows.append(row_cells)
        if raw_rows:
            _ensure_extraction_budget(deadline, "dense_table_normalization")
            table_locator = locator if table_index == 1 else f"{locator}#{table_index}"
            canonical_table, tsr_diagnostics = select_dense_table_normalization(
                raw_rows,
                locator=table_locator,
                page_locator=page_locator,
                unit_hint=unit_hint,
                context_text=context_text,
            )
            headers = list(canonical_table.get("display_headers", []))
            rows = [list(row) for row in canonical_table.get("display_rows", [])]
            if headers and rows:
                extracted.append(
                    _coerce_table(
                        headers,
                        rows,
                        citation,
                        locator=table_locator,
                        unit_hint=unit_hint,
                        page_locator=page_locator,
                        context_text=context_text,
                        canonical_table={**canonical_table, "experimental_tsr": tsr_diagnostics},
                    )
                )
    return extracted


def _extract_tables_from_json_payload(
    payload: Any,
    citation: str,
    tables: list[dict[str, Any]],
    limit: int = 8,
    *,
    table_query: str = "",
    deadline: float | None = None,
) -> None:
    _ensure_extraction_budget(deadline, "json_payload_walk")
    if len(tables) >= limit:
        return
    if isinstance(payload, dict):
        page_locator = _page_locator_from_payload(payload)
        context_text = " ".join(
            str(part).strip()
            for part in (
                payload.get("section_title"),
                payload.get("title"),
                payload.get("description"),
                payload.get("type"),
            )
            if str(part).strip()
        )
        locator = str(payload.get("section_title") or payload.get("title") or payload.get("description") or f"table {len(tables) + 1}")
        content = payload.get("content")
        if (
            isinstance(content, str)
            and "<table" in content.lower()
            and _table_candidate_matches_query(locator, context_text, content, table_query)
        ):
            html_tables = _extract_tables_from_html_string(
                content,
                citation,
                locator=locator,
                page_locator=page_locator,
                unit_hint=str(payload.get("unit") or payload.get("units") or ""),
                context_text=context_text,
                deadline=deadline,
            )
            for item in html_tables:
                tables.append(item)
                if len(tables) >= limit:
                    return
        headers = payload.get("headers") or payload.get("header") or payload.get("columns") or payload.get("column_headers")
        rows = payload.get("rows")
        if isinstance(headers, list) and isinstance(rows, list):
            normalized_rows: list[list[str]] = []
            for row in rows[:200]:
                if isinstance(row, list):
                    normalized_rows.append([str(cell).strip() for cell in row])
                elif isinstance(row, dict):
                    normalized_rows.append([str(row.get(str(header), "")).strip() for header in headers])
            if normalized_rows:
                canonical_table = normalize_flat_table(
                    [str(header) for header in headers],
                    normalized_rows,
                    locator=locator,
                    page_locator=page_locator,
                    unit_hint=str(payload.get("unit") or payload.get("units") or ""),
                    context_text=context_text,
                )
                tables.append(
                    _coerce_table(
                        list(canonical_table.get("display_headers", []) or [str(header) for header in headers]),
                        [list(row) for row in canonical_table.get("display_rows", []) or normalized_rows],
                        citation,
                        locator=locator,
                        unit_hint=str(payload.get("unit") or payload.get("units") or ""),
                        page_locator=page_locator,
                        context_text=context_text,
                        canonical_table=canonical_table,
                    )
                )
        for value in payload.values():
            if len(tables) >= limit:
                return
            _extract_tables_from_json_payload(
                value,
                citation,
                tables,
                limit=limit,
                table_query=table_query,
                deadline=deadline,
            )
    elif isinstance(payload, list):
        for item in payload:
            if len(tables) >= limit:
                return
            _extract_tables_from_json_payload(
                item,
                citation,
                tables,
                limit=limit,
                table_query=table_query,
                deadline=deadline,
            )


def _extract_tables_from_delimited_text(text: str, citation: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return []
    delimiter = "\t" if any("\t" in line for line in lines[:80]) else ","
    table_lines = [line for line in lines if delimiter in line]
    if len(table_lines) < 2:
        return []
    parsed_rows: list[list[str]] = []
    for line in table_lines[:240]:
        try:
            parsed = next(csv.reader([line], delimiter=delimiter))
        except Exception:
            parsed = [part.strip() for part in line.split(delimiter)]
        parsed_rows.append([str(cell).strip() for cell in parsed])
    headers = parsed_rows[0]
    rows = parsed_rows[1:]
    canonical_table = normalize_flat_table(headers, rows, locator="table 1")
    return [
        _coerce_table(
            list(canonical_table.get("display_headers", []) or headers),
            [list(row) for row in canonical_table.get("display_rows", []) or rows],
            citation,
            locator="table 1",
            canonical_table=canonical_table,
        )
    ]


def _extract_tables_from_text_layout(text: str, citation: str) -> list[dict[str, Any]]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 3:
        return []
    table_rows: list[list[str]] = []
    for line in lines[:280]:
        if re.search(r"\s{2,}", line):
            cells = [part.strip() for part in re.split(r"\s{2,}", line) if part.strip()]
            if len(cells) >= 2:
                table_rows.append(cells)
    if len(table_rows) < 2:
        return []
    headers = table_rows[0]
    rows = table_rows[1:]
    canonical_table = normalize_flat_table(headers, rows, locator="table 1")
    return [
        _coerce_table(
            list(canonical_table.get("display_headers", []) or headers),
            [list(row) for row in canonical_table.get("display_rows", []) or rows],
            citation,
            locator="table 1",
            canonical_table=canonical_table,
        )
    ]


def _table_candidate_matches_query(locator: str, context_text: str, content: str, table_query: str) -> bool:
    if not (table_query or "").strip():
        return True
    preview = f"{locator}\n{context_text}\n{(content or '')[:_HTML_QUERY_PREVIEW_CHARS]}"
    return _match_score(preview, table_query) > 0.0


def _is_page_reference_value(value: Any) -> bool:
    compact = re.sub(r"\s+", " ", str(value or "")).strip()
    if not compact:
        return False
    return bool(_PAGE_REF_RE.fullmatch(compact))


def _is_navigational_table(table: dict[str, Any]) -> bool:
    locator = str(table.get("locator", "") or "")
    context = str(table.get("context_text", "") or "")
    headers = [str(item or "") for item in list(table.get("headers", []))]
    rows = [list(row) for row in list(table.get("rows", []))[:24]]
    lowered = " ".join([locator, context, *headers]).lower()
    if any(token in lowered for token in ("table of contents", "cumulative table of contents", "issue and page number", "articles")):
        return True
    page_ref_cells = 0
    inspected_cells = 0
    for row in rows:
        for cell in row[:8]:
            inspected_cells += 1
            if _is_page_reference_value(cell):
                page_ref_cells += 1
    if inspected_cells >= 8 and (page_ref_cells / max(1, inspected_cells)) >= 0.5:
        return True
    return False


def _extract_document_tables(target: Path, text: str, citation: str, *, table_query: str = "") -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    suffix = target.suffix.lower()
    deadline = time.perf_counter() + _table_extraction_timeout_seconds()
    if suffix == ".json":
        try:
            payload = json.loads(target.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            payload = None
        if payload is not None:
            _extract_tables_from_json_payload(
                payload,
                citation,
                tables,
                limit=_MAX_EXTRACTED_TABLES,
                table_query=table_query,
                deadline=deadline,
            )
    if not tables and suffix in {".csv", ".tsv"}:
        _ensure_extraction_budget(deadline, "delimited_table_parse")
        tables.extend(_extract_tables_from_delimited_text(text, citation))
    if not tables:
        _ensure_extraction_budget(deadline, "fallback_delimited_table_parse")
        tables.extend(_extract_tables_from_delimited_text(text, citation))
    if not tables:
        _ensure_extraction_budget(deadline, "text_layout_table_parse")
        tables.extend(_extract_tables_from_text_layout(text, citation))
    return tables[:_MAX_EXTRACTED_TABLES]


def _match_score(text: str, query: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    overlap = len(query_tokens & text_tokens)
    return float(overlap) / max(1, len(query_tokens))


def _rank_tables(tables: list[dict[str, Any]], table_query: str) -> list[dict[str, Any]]:
    if not table_query.strip():
        return tables

    def _score(table: dict[str, Any]) -> float:
        text = " ".join(
            [
                str(table.get("locator", "")),
                str(table.get("context_text", "")),
                " ".join(str(item) for item in table.get("headers", [])),
                " ".join(" ".join(str(cell) for cell in row) for row in table.get("rows", [])[:12]),
                str(table.get("unit_hint", "")),
            ]
        )
        score = _match_score(text, table_query)
        lowered = text.lower()
        if "table of contents" in lowered or "cumulative table of contents" in lowered:
            score -= 0.25
        if _is_navigational_table(table):
            score -= 0.75
        numeric_cells = sum(
            1
            for row in list(table.get("rows", []))[:24]
            for cell in row[:8]
            if re.search(r"\d", str(cell))
        )
        if numeric_cells >= 8:
            score += 0.05
        return score

    return sorted(tables, key=_score, reverse=True)


def _filter_rows(table: dict[str, Any], row_query: str) -> list[list[str]]:
    rows = list(table.get("rows", []))
    if not row_query.strip():
        return rows
    scored: list[tuple[float, list[str]]] = []
    for row in rows:
        row_text = " ".join(str(cell) for cell in row)
        score = _match_score(row_text, row_query)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[: min(40, len(scored))]]


def _matching_column_indexes(headers: list[str], column_query: str) -> list[int]:
    if not column_query.strip():
        return list(range(len(headers)))
    indexes: list[tuple[float, int]] = []
    for idx, header in enumerate(headers):
        score = _match_score(str(header), column_query)
        if score > 0:
            indexes.append((score, idx))
    indexes.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in indexes[:4]]


def _table_payload(
    *,
    document_id: str,
    citation: str,
    file_name: str,
    file_format: str,
    officeqa_status: str,
    tables: list[dict[str, Any]],
    chunks: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = tables[0] if tables else {}
    headers = list(primary.get("headers", [])) if isinstance(primary, dict) else []
    rows = list(primary.get("rows", [])) if isinstance(primary, dict) else []
    metadata = {
        "file_name": file_name,
        "format": file_format,
        "officeqa_status": officeqa_status,
        "window_kind": "table",
        "table_count": len(tables),
        "row_count": len(rows),
        "column_count": len(headers),
    }
    if extra:
        metadata.update(extra)
    return {
        "document_id": document_id,
        "citation": citation,
        "metadata": metadata,
        "chunks": list(chunks or []),
        "tables": tables,
        "numeric_summaries": _row_numeric_summaries(headers, rows),
    }


@tool
def search_reference_corpus(
    query: str,
    top_k: int = 5,
    snippet_chars: int = 700,
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    """Search a configured local document corpus for relevant files and snippets."""
    root = _resolve_corpus_root()
    if root is None:
        return {"error": "No local corpus directory is configured. Set OFFICEQA_CORPUS_DIR or REFERENCE_CORPUS_DIR."}
    if officeqa_index_available(root):
        return search_officeqa_corpus_index(
            query,
            corpus_root=root,
            top_k=top_k,
            snippet_chars=snippet_chars,
            source_files=source_files,
        )

    scored_results: list[tuple[float, Path, str]] = []
    for path in _iter_corpus_files(root):
        try:
            text = _read_file_text(path)
        except Exception:
            continue
        score = _score_document(text, query, path)
        if score <= 0:
            continue
        snippet = _best_snippet(text, query, snippet_chars)
        scored_results.append((score, path, snippet))

    scored_results.sort(key=lambda item: (-item[0], item[1].as_posix()))
    top = scored_results[: max(1, min(top_k, 8))]
    results: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for rank, (score, path, snippet) in enumerate(top, start=1):
        relative = path.relative_to(root).as_posix()
        doc_id = _document_id(path, root)
        citation = relative
        metadata = _document_metadata(_read_file_text(path), path)
        results.append(
            {
                "rank": rank,
                "title": relative,
                "snippet": snippet,
                "url": citation,
                "score": round(score, 3),
                "document_id": doc_id,
                "metadata": metadata,
            }
        )
        documents.append(
            {
                "document_id": doc_id,
                "citation": citation,
                "format": path.suffix.lower().lstrip(".") or "text",
                "path": relative,
                "metadata": metadata,
            }
        )

    return {
        "query": query,
        "corpus_root": str(root),
        "results": results,
        "documents": documents,
        "result_count": len(results),
    }


@tool
def search_officeqa_documents(
    query: str,
    top_k: int = 5,
    snippet_chars: int = 700,
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    """Search the indexed OfficeQA corpus for candidate Treasury source documents."""
    root = _resolve_corpus_root()
    if root is None:
        return {"error": "No local OfficeQA corpus directory is configured. Set OFFICEQA_CORPUS_DIR."}
    if not officeqa_index_available(root):
        return {"error": "No OfficeQA corpus index is available. Run scripts/build_officeqa_index.py first."}
    result = search_officeqa_corpus_index(
        query,
        corpus_root=root,
        top_k=top_k,
        snippet_chars=snippet_chars,
        source_files=source_files,
    )
    result["officeqa_stage"] = "identify_source"
    return result


@tool
def fetch_officeqa_pages(
    document_id: str = "",
    path: str = "",
    page_start: int = 0,
    page_limit: int = 5,
    max_chars: int = 4500,
) -> dict[str, Any]:
    """Read OfficeQA document windows using page-oriented semantics over the packaged corpus."""
    result = fetch_corpus_document.invoke(
        {
            "document_id": document_id,
            "path": path,
            "chunk_start": page_start,
            "chunk_limit": page_limit,
            "max_chars": max_chars,
        }
    )
    if not isinstance(result, dict) or result.get("error"):
        return result if isinstance(result, dict) else {"error": "Unable to read OfficeQA pages."}
    metadata = dict(result.get("metadata") or {})
    chunk_start = int(metadata.get("chunk_start", page_start) or page_start)
    returned_chunks = int(metadata.get("returned_chunks", len(result.get("chunks", []))) or len(result.get("chunks", [])))
    total_chunks = int(metadata.get("total_chunks", returned_chunks) or returned_chunks)
    metadata.update(
        {
            "window_kind": "pages",
            "window": f"pages {chunk_start + 1}-{chunk_start + returned_chunks}",
            "page_start": chunk_start,
            "page_limit": int(metadata.get("chunk_limit", page_limit) or page_limit),
            "total_pages": total_chunks,
            "has_more_windows": bool(metadata.get("has_more_chunks")),
            "officeqa_status": "ok" if result.get("chunks") else "empty",
        }
    )
    for idx, chunk in enumerate(result.get("chunks", []), start=1):
        if isinstance(chunk, dict):
            chunk["locator"] = f"page {chunk_start + idx}"
    result["metadata"] = metadata
    result["officeqa_stage"] = "locate_pages"
    return result


@tool
def fetch_officeqa_table(
    document_id: str = "",
    path: str = "",
    table_query: str = "",
    row_offset: int = 0,
    row_limit: int = 200,
) -> dict[str, Any]:
    """Extract the most relevant structured table from an OfficeQA corpus artifact."""
    root = _resolve_corpus_root()
    if root is None:
        return {"error": "No local OfficeQA corpus directory is configured. Set OFFICEQA_CORPUS_DIR."}
    target = _resolve_corpus_target(root, document_id=document_id, path=path)
    if target is None:
        return {"error": "OfficeQA document not found in configured corpus."}
    text = _read_file_text(target)
    citation = target.relative_to(root).as_posix()
    resolved_document_id = _document_id(target, root)
    try:
        tables = _rank_tables(_extract_document_tables(target, text, citation, table_query=table_query), table_query)
    except _OfficeQATableExtractionTimeout as exc:
        return _table_payload(
            document_id=resolved_document_id,
            citation=citation,
            file_name=target.name,
            file_format=target.suffix.lower().lstrip(".") or "text",
            officeqa_status="table_timeout",
            tables=[],
            chunks=[],
            extra={
                "row_offset": max(0, row_offset),
                "row_limit": max(1, row_limit),
                "error": str(exc),
            },
        ) | {"officeqa_stage": "locate_table", "error": str(exc)}
    if not tables:
        return _table_payload(
            document_id=resolved_document_id,
            citation=citation,
            file_name=target.name,
            file_format=target.suffix.lower().lstrip(".") or "text",
            officeqa_status="missing_table",
            tables=[],
            chunks=[],
        )
    selected = dict(tables[0])
    selected_rows = list(selected.get("rows", []))[max(0, row_offset): max(0, row_offset) + max(1, row_limit)]
    selected["rows"] = selected_rows
    preview = "\n".join(",".join(row) for row in selected_rows[:8])[:1200]
    officeqa_status = "partial_table" if not selected_rows else "ok"
    return _table_payload(
        document_id=resolved_document_id,
        citation=citation,
        file_name=target.name,
        file_format=target.suffix.lower().lstrip(".") or "text",
        officeqa_status=officeqa_status,
        tables=[selected],
        chunks=[{"locator": str(selected.get("locator", "table 1")), "kind": "table_preview", "text": preview, "citation": citation}],
        extra={"row_offset": max(0, row_offset), "row_limit": max(1, row_limit)},
    ) | {"officeqa_stage": "locate_table"}


@tool
def lookup_officeqa_rows(
    document_id: str = "",
    path: str = "",
    table_query: str = "",
    row_query: str = "",
    row_offset: int = 0,
    row_limit: int = 200,
) -> dict[str, Any]:
    """Filter an OfficeQA table down to the rows most relevant to the benchmark query."""
    table_result = fetch_officeqa_table.invoke(
        {
            "document_id": document_id,
            "path": path,
            "table_query": table_query,
            "row_offset": 0,
            "row_limit": max(row_limit, 200),
        }
    )
    if not isinstance(table_result, dict) or table_result.get("error"):
        return table_result if isinstance(table_result, dict) else {"error": "Unable to look up OfficeQA rows."}
    tables = list(table_result.get("tables", []))
    if not tables:
        table_result.setdefault("metadata", {})
        table_result["metadata"]["officeqa_status"] = "missing_table"
        table_result["officeqa_stage"] = "extract_rows"
        return table_result
    primary = dict(tables[0])
    filtered_rows = _filter_rows(primary, row_query)
    sliced_rows = filtered_rows[max(0, row_offset): max(0, row_offset) + max(1, row_limit)]
    primary["rows"] = sliced_rows
    status = "missing_row" if row_query.strip() and not filtered_rows else ("partial_table" if not sliced_rows else "ok")
    payload = _table_payload(
        document_id=str(table_result.get("document_id", "")),
        citation=str(table_result.get("citation", "")),
        file_name=str(dict(table_result.get("metadata") or {}).get("file_name", "")),
        file_format=str(dict(table_result.get("metadata") or {}).get("format", "")),
        officeqa_status=status,
        tables=[primary],
        chunks=[{"locator": str(primary.get("locator", "table 1")), "kind": "row_lookup", "text": "\n".join(",".join(row) for row in sliced_rows[:8])[:1200], "citation": str(table_result.get("citation", ""))}],
        extra={"row_query": row_query, "row_offset": max(0, row_offset), "row_limit": max(1, row_limit)},
    )
    payload["officeqa_stage"] = "extract_rows"
    return payload


@tool
def lookup_officeqa_cells(
    document_id: str = "",
    path: str = "",
    table_query: str = "",
    row_query: str = "",
    column_query: str = "",
    row_offset: int = 0,
    row_limit: int = 50,
) -> dict[str, Any]:
    """Filter an OfficeQA table down to the cells most relevant to the target row and column."""
    row_result = lookup_officeqa_rows.invoke(
        {
            "document_id": document_id,
            "path": path,
            "table_query": table_query,
            "row_query": row_query,
            "row_offset": row_offset,
            "row_limit": row_limit,
        }
    )
    if not isinstance(row_result, dict) or row_result.get("error"):
        return row_result if isinstance(row_result, dict) else {"error": "Unable to look up OfficeQA cells."}
    tables = list(row_result.get("tables", []))
    if not tables:
        row_result.setdefault("metadata", {})
        row_result["metadata"]["officeqa_status"] = "missing_table"
        row_result["officeqa_stage"] = "extract_cells"
        return row_result
    primary = dict(tables[0])
    headers = list(primary.get("headers", []))
    rows = list(primary.get("rows", []))
    matching_indexes = _matching_column_indexes(headers, column_query)
    if not matching_indexes:
        status = "unit_ambiguity" if column_query.strip() else "ok"
        matching_indexes = list(range(len(headers)))
    else:
        status = "ok"
    narrowed_headers = [headers[idx] for idx in matching_indexes if idx < len(headers)]
    narrowed_rows = [[row[idx] for idx in matching_indexes if idx < len(row)] for row in rows]
    cells: list[dict[str, Any]] = []
    for row_idx, row in enumerate(narrowed_rows[:40]):
        for col_idx, value in enumerate(row):
            header = narrowed_headers[col_idx] if col_idx < len(narrowed_headers) else f"col_{col_idx + 1}"
            cells.append({"row_index": row_idx, "column_label": header, "value": value})
    primary["headers"] = narrowed_headers
    primary["rows"] = narrowed_rows
    payload = _table_payload(
        document_id=str(row_result.get("document_id", "")),
        citation=str(row_result.get("citation", "")),
        file_name=str(dict(row_result.get("metadata") or {}).get("file_name", "")),
        file_format=str(dict(row_result.get("metadata") or {}).get("format", "")),
        officeqa_status=status if narrowed_rows else "partial_table",
        tables=[primary],
        chunks=[{"locator": str(primary.get("locator", "table 1")), "kind": "cell_lookup", "text": json.dumps(cells[:12], ensure_ascii=True), "citation": str(row_result.get("citation", ""))}],
        extra={"row_query": row_query, "column_query": column_query, "cell_count": len(cells)},
    )
    payload["cells"] = cells
    payload["officeqa_stage"] = "extract_cells"
    return payload


@tool
def fetch_corpus_document(
    document_id: str = "",
    path: str = "",
    chunk_start: int = 0,
    chunk_limit: int = 3,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Read a document window from the configured local corpus."""
    root = _resolve_corpus_root()
    if root is None:
        return {"error": "No local corpus directory is configured. Set OFFICEQA_CORPUS_DIR or REFERENCE_CORPUS_DIR."}

    root = root.resolve()
    target: Path | None = None
    if officeqa_index_available(root):
        resolved = resolve_indexed_corpus_document(document_id=document_id, path=path, corpus_root=root)
        if resolved is not None:
            candidate = (root / str(resolved.get("relative_path", ""))).resolve()
            if candidate.exists() and candidate.is_file() and _is_within_root(candidate, root):
                target = candidate
    if path:
        candidate = (root / path).resolve()
        if candidate.exists() and candidate.is_file() and _is_within_root(candidate, root):
            target = candidate
    if target is None and document_id:
        for candidate in _iter_corpus_files(root):
            if _document_id(candidate, root) == document_id:
                target = candidate
                break
    if target is None:
        return {"error": "Document not found in configured corpus."}

    try:
        text = _read_file_text(target)
    except Exception as exc:
        return {"error": f"Unable to read corpus document: {exc}"}

    relative = target.relative_to(root).as_posix()
    doc_id = _document_id(target, root)
    chunks = _text_chunks(text, max_chars=max(800, min(max_chars, 2400)))
    if not chunks:
        return {
            "document_id": doc_id,
            "citation": relative,
            "metadata": {
                "file_name": target.name,
                "format": target.suffix.lower().lstrip(".") or "text",
                "window": "chunks 0-0",
                "total_chunks": 0,
                "has_more_chunks": False,
                "chunk_start": 0,
                "chunk_limit": 0,
                "returned_chunks": 0,
            },
            "chunks": [],
            "tables": [],
            "numeric_summaries": [],
        }
    effective_limit = max(1, min(chunk_limit, 6))
    start = max(0, min(chunk_start, max(0, len(chunks) - 1)))
    selected_chunks = chunks[start: start + effective_limit]
    rendered_chunks = [
        {
            "locator": f"chunk {start + idx + 1}",
            "kind": "text_excerpt",
            "text": chunk[:max_chars],
            "citation": relative,
        }
        for idx, chunk in enumerate(selected_chunks)
    ]
    total_chunks = len(chunks)
    has_more = start + effective_limit < total_chunks
    excerpt = "\n\n".join(chunk["text"] for chunk in rendered_chunks)[:max_chars]
    return {
        "document_id": doc_id,
        "citation": relative,
        "metadata": {
            "file_name": target.name,
            "format": target.suffix.lower().lstrip(".") or "text",
            "window": f"chunks {start + 1}-{start + len(rendered_chunks)}",
            "total_chunks": total_chunks,
            "has_more_chunks": has_more,
            "chunk_start": start,
            "chunk_limit": effective_limit,
            "returned_chunks": len(rendered_chunks),
        },
        "chunks": rendered_chunks,
        "tables": [],
        "numeric_summaries": _extract_numeric_summaries(excerpt),
    }
