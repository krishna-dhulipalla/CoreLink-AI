"""Built-in corpus retrieval tools for document-heavy benchmarks."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_CORPUS_ENV_NAMES = (
    "OFFICEQA_CORPUS_DIR",
    "REFERENCE_CORPUS_DIR",
    "DOCUMENT_CORPUS_DIR",
)
_CORPUS_CANDIDATES = (
    "treasury_bulletins_parsed",
    "officeqa/treasury_bulletins_parsed",
    "data/treasury_bulletins_parsed",
    "reference_corpus",
    "documents",
)
_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".html", ".xml", ".tsv"}
_MAX_FILES = 4000


def _resolve_corpus_root() -> Path | None:
    for env_name in _CORPUS_ENV_NAMES:
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists() and path.is_dir():
            return path

    cwd = Path.cwd()
    for candidate in _CORPUS_CANDIDATES:
        path = cwd / candidate
        if path.exists() and path.is_dir():
            return path
    return None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _document_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return re.sub(r"[^a-z0-9_]+", "_", relative.lower()).strip("_") or "document"


def _read_file_text(path: Path) -> str:
    suffix = path.suffix.lower()
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
        for line in raw.splitlines()[:200]:
            try:
                rows.append(next(csv.reader([line], delimiter=delimiter)))
            except Exception:
                rows.append([part.strip() for part in line.split(delimiter)])
        return "\n".join(delimiter.join(cell.strip() for cell in row) for row in rows)
    return raw


def _iter_corpus_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        files.append(path)
        if len(files) >= _MAX_FILES:
            break
    return files


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


def _score_document(text: str, query: str, path: Path) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    lowered = (text or "").lower()
    path_text = path.as_posix().lower()
    unique_hits = {token for token in query_tokens if token in lowered or token in path_text}
    if not unique_hits:
        return 0.0
    score = float(len(unique_hits))
    for token in query_tokens:
        if token in path_text:
            score += 0.5
    if str(path.stem).lower() in lowered:
        score += 0.2
    year_matches = re.findall(r"\b((?:19|20)\d{2})\b", query)
    for year in year_matches:
        if year in lowered or year in path_text:
            score += 1.0
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


@tool
def search_reference_corpus(query: str, top_k: int = 5, snippet_chars: int = 700) -> dict[str, Any]:
    """Search a configured local document corpus for relevant files and snippets."""
    root = _resolve_corpus_root()
    if root is None:
        return {"error": "No local corpus directory is configured. Set OFFICEQA_CORPUS_DIR or REFERENCE_CORPUS_DIR."}

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
        results.append(
            {
                "rank": rank,
                "title": relative,
                "snippet": snippet,
                "url": citation,
                "score": round(score, 3),
                "document_id": doc_id,
            }
        )
        documents.append(
            {
                "document_id": doc_id,
                "citation": citation,
                "format": path.suffix.lower().lstrip(".") or "text",
                "path": relative,
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

    target: Path | None = None
    if path:
        candidate = (root / path).resolve()
        if candidate.exists() and candidate.is_file():
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
    selected_chunks = chunks[chunk_start: chunk_start + max(1, min(chunk_limit, 6))]
    rendered_chunks = [
        {
            "locator": f"chunk {chunk_start + idx + 1}",
            "kind": "text_excerpt",
            "text": chunk[:max_chars],
            "citation": relative,
        }
        for idx, chunk in enumerate(selected_chunks)
    ]
    excerpt = "\n\n".join(chunk["text"] for chunk in rendered_chunks)[:max_chars]
    return {
        "document_id": doc_id,
        "citation": relative,
        "metadata": {
            "file_name": target.name,
            "format": target.suffix.lower().lstrip(".") or "text",
            "window": f"chunks {chunk_start + 1}-{chunk_start + len(rendered_chunks)}",
        },
        "chunks": rendered_chunks,
        "tables": [],
        "numeric_summaries": _extract_numeric_summaries(excerpt),
    }
