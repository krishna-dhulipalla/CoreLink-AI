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

from agent.runtime_clock import increment_runtime_step
from agent.state import AgentState

logger = logging.getLogger(__name__)


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
        return value


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
        root = contract.get("xml_root_tag") or "answer"
        escaped = (
            source_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        adapted = f"<{root}>{escaped}</{root}>"
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
