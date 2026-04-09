"""State-aware traversal helpers for parsed OfficeQA Treasury JSON."""

from __future__ import annotations

import re
from typing import Any

_CONTINUED_RE = re.compile(r"\bcontinued\b", re.IGNORECASE)
_UNIT_RE = re.compile(
    r"\b(in|amounts in|figures in)\s+(millions?|billions?|thousands?|percent|dollars?|cents?)\b|\(in [^)]+\)",
    re.IGNORECASE,
)


def clean_context_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_continued_marker(text: Any) -> str:
    cleaned = clean_context_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\(?\bcontinued\b\)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*-\s*$", "", cleaned)
    return clean_context_text(cleaned)


def is_continuation_text(text: Any) -> bool:
    return bool(_CONTINUED_RE.search(str(text or "")))


def page_locator_from_node(node: dict[str, Any]) -> str:
    for page_key in ("page", "page_number", "page_num"):
        value = node.get(page_key)
        if isinstance(value, (int, float)) and int(value) > 0:
            return f"page {int(value)}"
        if isinstance(value, str) and value.strip().isdigit():
            return f"page {int(value.strip())}"
    bbox = node.get("bbox")
    if isinstance(bbox, list):
        for entry in bbox[:6]:
            if not isinstance(entry, dict):
                continue
            page_id = entry.get("page_id")
            if isinstance(page_id, int) and page_id > 0:
                return f"page {page_id}"
            if isinstance(page_id, str) and page_id.strip().isdigit():
                return f"page {int(page_id.strip())}"
    return ""


def looks_like_unit_text(text: Any) -> bool:
    cleaned = clean_context_text(text)
    if not cleaned:
        return False
    return bool(_UNIT_RE.search(cleaned))


def dedupe_context_parts(parts: list[str]) -> list[str]:
    deduped: list[str] = []
    lowered_seen: set[str] = set()
    for part in parts:
        cleaned = clean_context_text(part)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in lowered_seen:
            continue
        lowered_seen.add(lowered)
        deduped.append(cleaned)
    return deduped


def _normalized_heading_key(parts: list[str]) -> str:
    normalized = [strip_continued_marker(part).lower() for part in parts if strip_continued_marker(part)]
    return " | ".join(dict.fromkeys(normalized))


def iter_stateful_table_nodes(payload: Any) -> list[dict[str, Any]]:
    """Yield parsed Treasury table nodes with inherited document context."""
    if not isinstance(payload, dict):
        return []
    document = payload.get("document")
    if not isinstance(document, dict):
        return []
    elements = document.get("elements")
    if not isinstance(elements, list):
        return []

    results: list[dict[str, Any]] = []
    current_title_chain: list[str] = []
    section_chain: list[str] = []
    current_page = ""
    recent_unit_text = ""
    recent_notes: list[str] = []
    last_node_type = ""
    pending_continuation_anchor: dict[str, Any] | None = None
    last_table_anchor: dict[str, Any] | None = None

    for raw_node in elements[:5000]:
        if not isinstance(raw_node, dict):
            continue
        node = dict(raw_node)
        node_type = str(node.get("type", "") or "").strip().lower()
        if not node_type:
            continue

        page_locator = page_locator_from_node(node) or current_page
        if page_locator and page_locator != current_page:
            current_page = page_locator
            recent_unit_text = ""
            recent_notes = []
            section_chain = []

        content_text = clean_context_text(node.get("content", ""))
        description_text = clean_context_text(node.get("description", ""))
        primary_text = content_text or description_text

        if node_type in {"title", "section_header", "page_header"}:
            if primary_text:
                base_text = strip_continued_marker(primary_text)
                if base_text:
                    if node_type == "title":
                        current_title_chain = [base_text]
                    elif node_type == "section_header":
                        if last_node_type == "section_header" and section_chain:
                            section_chain = dedupe_context_parts([*section_chain, base_text])[:4]
                        else:
                            section_chain = [base_text]
                    elif not current_title_chain:
                        current_title_chain = [base_text]
            last_node_type = node_type
            continue

        if node_type in {"text", "footnote"}:
            if looks_like_unit_text(primary_text):
                recent_unit_text = primary_text
            elif primary_text:
                if node_type == "footnote":
                    recent_notes = dedupe_context_parts([*recent_notes, primary_text])[-4:]
                if is_continuation_text(primary_text) and last_table_anchor is not None:
                    pending_continuation_anchor = dict(last_table_anchor)
            last_node_type = node_type
            continue

        if node_type != "table":
            last_node_type = node_type
            continue

        local_label = strip_continued_marker(description_text or clean_context_text(node.get("heading", "")) or "")
        raw_heading_chain = [
            item
            for item in [*current_title_chain, *section_chain, local_label]
            if clean_context_text(item)
        ]
        cleaned_heading_chain = [strip_continued_marker(item) for item in raw_heading_chain if strip_continued_marker(item)]
        explicit_continuation = any(is_continuation_text(item) for item in raw_heading_chain)
        anchor = dict(pending_continuation_anchor or {}) if pending_continuation_anchor else None
        if explicit_continuation and last_table_anchor is not None and anchor is None:
            anchor = dict(last_table_anchor)
        inherited_heading_chain = list(anchor.get("heading_chain", [])) if anchor else []
        heading_chain = dedupe_context_parts([*inherited_heading_chain, *cleaned_heading_chain])
        if not heading_chain:
            heading_chain = dedupe_context_parts(cleaned_heading_chain)
        context_text = " | ".join(heading_chain)
        continuation_flag = bool(anchor) or explicit_continuation
        continuation_key = (
            str(anchor.get("continuation_key", "") or "")
            if anchor
            else _normalized_heading_key(heading_chain or raw_heading_chain)
        )
        raw_note_context = list(recent_notes)
        note_context = [clean_context_text(item) for item in raw_note_context if clean_context_text(item)]
        results.append(
            {
                "node": node,
                "locator": local_label or context_text or f"table {len(results) + 1}",
                "page_locator": page_locator,
                "heading_chain": heading_chain,
                "raw_heading_chain": raw_heading_chain,
                "context_text": context_text,
                "unit_hint": recent_unit_text,
                "raw_unit_hint": primary_text if looks_like_unit_text(primary_text) else recent_unit_text,
                "note_context": note_context,
                "raw_note_context": raw_note_context,
                "is_continuation": continuation_flag,
                "continuation_key": continuation_key,
                "continued_from_locator": str(anchor.get("locator", "") or "") if anchor else "",
                "continued_from_page_locator": str(anchor.get("page_locator", "") or "") if anchor else "",
                "continued_from_heading_chain": list(anchor.get("heading_chain", [])) if anchor else [],
            }
        )
        last_table_anchor = {
            "continuation_key": continuation_key,
            "heading_chain": heading_chain,
            "locator": local_label or context_text or f"table {len(results)}",
            "page_locator": page_locator,
        }
        pending_continuation_anchor = None
        recent_notes = []
        last_node_type = node_type
    return results
