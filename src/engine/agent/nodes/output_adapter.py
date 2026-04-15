"""
Output Adapter Node
===================
Final formatting adapter driven by the extracted answer contract.
"""

from __future__ import annotations

import logging
import json
import re

from langchain_core.messages import AIMessage

from engine.agent.runtime_clock import increment_runtime_step
from engine.agent.state import AgentState

logger = logging.getLogger(__name__)
_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_TAG_BLOCK_RE = re.compile(r"<(?P<tag>[A-Za-z][A-Za-z0-9_\-]*)>.*?</(?P=tag)>\s*", re.DOTALL)
_XMLISH_RE = re.compile(r"<[/!?]?[A-Za-z]")
_INSUFFICIENCY_RE = re.compile(
    r"\b(?:insufficient data|insufficient evidence|cannot determine|cannot calculate|cannot compute|cannot answer|not present in the provided evidence|not available in the provided evidence|data is insufficient)\b",
    re.IGNORECASE,
)


def _latest_answer(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content).strip()
    return ""


def _coerce_scalar(text: str):
    value = text.strip()
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        match = re.search(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", value)
        if match:
            candidate = match.group(0).replace("$", "").replace(",", "")
            try:
                if "." in candidate:
                    return float(candidate)
                return int(candidate)
            except ValueError:
                return candidate
        return value


def _extract_tag_contents(text: str, tag: str) -> str:
    if not tag:
        return ""
    match = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text or "", re.DOTALL)
    return str(match.group(1)).strip() if match else ""


def _escape_xml(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _render_final_answer_xml_value(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    escaped = value.replace("&", "&amp;")
    if _XMLISH_RE.search(escaped):
        escaped = escaped.replace("<", "&lt;").replace(">", "&gt;")
    return escaped


def _strip_xml_blocks(text: str) -> str:
    return _TAG_BLOCK_RE.sub("", text or "").strip()


def _strip_escaped_final_answer_markup(text: str) -> str:
    cleaned = re.sub(r"&lt;/?FINAL_ANSWER&gt;", "", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"&lt;/?REASONING&gt;", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_labeled_final_value(text: str) -> str:
    patterns = (
        r"(?im)^\s*(?:final answer|answer|conclusion|result)\s*:\s*(.+?)\s*$",
        r"(?im)^\s*\*\*(?:final answer|answer|conclusion|result)\*\*\s*:\s*(.+?)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "")
        if not match:
            continue
        candidate = str(match.group(1)).strip().strip("`").strip("*")
        if candidate:
            return candidate
    return ""


def _extract_insufficiency_statement(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    for line in reversed([part.strip() for part in source.splitlines() if part.strip()]):
        cleaned = _strip_escaped_final_answer_markup(line)
        if _INSUFFICIENCY_RE.search(cleaned):
            return cleaned.strip().strip("`")
    sentences = re.split(r"(?<=[.!?])\s+", source)
    for sentence in reversed([item.strip() for item in sentences if item.strip()]):
        cleaned = _strip_escaped_final_answer_markup(sentence)
        if _INSUFFICIENCY_RE.search(cleaned):
            return cleaned.strip().strip("`")
    return ""


def _infer_final_answer_value(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""

    explicit = _extract_labeled_final_value(source)
    if explicit:
        if _INSUFFICIENCY_RE.search(explicit):
            return explicit.rstrip(".")
        numeric_match = _NUMERIC_TOKEN_RE.search(explicit)
        if numeric_match and numeric_match.group(0) == explicit.strip().rstrip("."):
            return numeric_match.group(0)
        return explicit.rstrip(".")

    insufficiency = _extract_insufficiency_statement(source)
    if insufficiency:
        return insufficiency.rstrip(".")

    for line in reversed([part.strip() for part in source.splitlines() if part.strip()]):
        labeled = _extract_labeled_final_value(line)
        if labeled:
            return labeled.rstrip(".")
        numeric_line = line.strip().strip("`").replace("$", "")
        if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?%?", numeric_line):
            return numeric_line
        if re.search(r"\b(?:answer|result|total|equals|is|was)\b", line.lower()):
            line_numbers = _NUMERIC_TOKEN_RE.findall(line)
            if line_numbers:
                return line_numbers[-1]

    tail_lines = [part.strip() for part in source.splitlines() if part.strip()]
    if tail_lines:
        return tail_lines[-1].strip().strip("`").rstrip(".")
    return source


def _final_answer_needs_normalization(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return True
    if len(candidate) > 120:
        return True
    lowered = candidate.lower()
    if any(marker in lowered for marker in ("thus,", "therefore", "final answer", "reasoning", "<final_answer>", "&lt;final_answer&gt;")):
        return True
    if _XMLISH_RE.search(candidate):
        return True
    return False


def _adapt_final_answer_tags(source_text: str, contract: dict) -> str:
    value_rules = dict(contract.get("value_rules") or {})
    reasoning_tag = str(value_rules.get("reasoning_tag") or "REASONING")
    final_answer_tag = str(value_rules.get("final_answer_tag") or contract.get("xml_root_tag") or "FINAL_ANSWER")

    existing_final = _extract_tag_contents(source_text, final_answer_tag)
    existing_reasoning = _extract_tag_contents(source_text, reasoning_tag)
    if existing_final and not _final_answer_needs_normalization(existing_final):
        final_value = existing_final
    else:
        final_value = _infer_final_answer_value(existing_final or existing_reasoning or _strip_xml_blocks(source_text))

    if existing_reasoning:
        reasoning_text = _strip_escaped_final_answer_markup(existing_reasoning)
    else:
        reasoning_text = _strip_escaped_final_answer_markup(_strip_xml_blocks(source_text))
        if not reasoning_text:
            reasoning_text = f"The final answer is {final_value}."

    return (
        f"<{reasoning_tag}>\n{_escape_xml(reasoning_text)}\n</{reasoning_tag}>\n"
        f"<{final_answer_tag}>\n{_render_final_answer_xml_value(final_value)}\n</{final_answer_tag}>"
    )


def output_adapter(state: AgentState) -> dict:
    step = increment_runtime_step()
    contract = dict(state.get("answer_contract", {}))
    source_text = _latest_answer(state.get("messages", []))
    if not source_text or not contract.get("requires_adapter"):
        return {"messages": []}

    adapted = source_text
    fmt = contract.get("format", "text")
    if fmt == "json":
        try:
            parsed = json.loads(source_text)
            if contract.get("wrapper_key") and not (
                isinstance(parsed, dict)
                and set(parsed.keys()) == {contract["wrapper_key"]}
            ):
                adapted = json.dumps({contract["wrapper_key"]: parsed}, ensure_ascii=True)
            else:
                adapted = json.dumps(parsed, ensure_ascii=True)
        except Exception:
            wrapper = contract.get("wrapper_key") or "answer"
            adapted = json.dumps({wrapper: _coerce_scalar(source_text)}, ensure_ascii=True)
    elif fmt == "xml":
        value_rules = dict(contract.get("value_rules") or {})
        if str(value_rules.get("final_answer_tag") or contract.get("xml_root_tag") or "") == "FINAL_ANSWER":
            adapted = _adapt_final_answer_tags(source_text, contract)
        else:
            root = contract.get("xml_root_tag") or "answer"
            adapted = f"<{root}>{_escape_xml(source_text)}</{root}>"
    else:
        example = contract.get("exact_output_example")
        if example and re.fullmatch(r"\{.*\}", example.strip()):
            wrapper_match = re.search(r'"([A-Za-z0-9_]+)"\s*:', example)
            if wrapper_match:
                adapted = json.dumps({wrapper_match.group(1): _coerce_scalar(source_text)}, ensure_ascii=True)

    if adapted == source_text:
        logger.info("[Step %s] output_adapter -> no-op", step)
        return {"messages": []}

    logger.info("[Step %s] output_adapter -> formatted as %s", step, fmt)
    return {"messages": [AIMessage(content=adapted)]}
