"""
Tool Runner Node
================
Executes exactly one tool call and normalizes the result into a ToolResult
contract before returning control to the solver.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agent.contracts import ToolResult
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import PROFILE_TOOL_ALLOWLIST
from agent.state import AgentState

logger = logging.getLogger(__name__)

_STRUCTURED_RESULTS_RE = re.compile(r"STRUCTURED_RESULTS:\s*(.+?)(?:\n---|\Z)", re.DOTALL)


def _tool_signature(state: AgentState) -> str:
    pending = state.get("pending_tool_call") or {}
    return f"{pending.get('name', '')}:{json.dumps(pending.get('arguments', {}), sort_keys=True)}"


def _tool_registry(tool_node: ToolNode) -> dict[str, Any]:
    registry = getattr(tool_node, "tools_by_name", None)
    if isinstance(registry, dict):
        return registry
    tools = getattr(tool_node, "tools", []) or []
    return {getattr(tool, "name", ""): tool for tool in tools if getattr(tool, "name", "")}


def _normalize_scalar(value: str) -> Any:
    raw = value.strip().strip("$")
    lowered = raw.lower()
    if lowered in {"credit", "debit", "fairly priced", "overpriced", "underpriced", "short_vol", "neutral"}:
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
    urls = re.findall(r"https?://[^\s\)\]\"',]+", raw or "")
    return {"urls": urls}


def _parse_file_fetch(raw: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    file_match = re.search(r"FILE:\s*(.+)", raw or "")
    format_match = re.search(r"FORMAT:\s*([A-Z0-9_]+)", raw or "")
    if file_match:
        facts["file_name"] = file_match.group(1).strip()
    if format_match:
        facts["format"] = format_match.group(1).strip().lower()
    if file_match or format_match:
        facts["preview"] = raw[:400]
    return facts


def _normalize_tool_output(tool_name: str, raw_content: Any, args: dict[str, Any]) -> ToolResult:
    if isinstance(raw_content, list):
        raw_content = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in raw_content
        )
    if isinstance(raw_content, dict):
        return ToolResult(
            type=tool_name,
            facts=raw_content,
            assumptions=args,
            source={"tool": tool_name},
            errors=[],
        )

    text = str(raw_content or "").strip()
    if not text:
        return ToolResult(
            type=tool_name,
            facts={},
            assumptions=args,
            source={"tool": tool_name},
            errors=["Empty tool output."],
        )
    if text.startswith("Error"):
        return ToolResult(
            type=tool_name,
            facts={},
            assumptions=args,
            source={"tool": tool_name},
            errors=[text],
        )

    facts = _parse_structured_results(text)
    if not facts and tool_name == "analyze_strategy":
        facts = _parse_strategy_analysis(text)
    elif not facts and tool_name == "list_reference_files":
        facts = _parse_reference_listing(text)
    elif not facts and tool_name == "fetch_reference_file":
        facts = _parse_file_fetch(text)
    elif not facts and tool_name == "calculator":
        facts = {"result": _normalize_scalar(text)}

    errors = [] if facts else ["Unstructured tool output could not be normalized into machine-usable facts."]
    return ToolResult(
        type=tool_name,
        facts=facts,
        assumptions=args,
        source={"tool": tool_name, "raw_preview": text[:240]},
        errors=errors,
    )


def make_tool_runner(tool_node: ToolNode):
    async def tool_runner(state: AgentState) -> dict:
        step = increment_runtime_step()
        profile = state.get("task_profile", "general")
        pending = state.get("pending_tool_call") or {}
        tool_name = str(pending.get("name", "")).strip()
        tool_args = pending.get("arguments", {})
        allowed = PROFILE_TOOL_ALLOWLIST.get(profile, PROFILE_TOOL_ALLOWLIST["general"])
        registry = _tool_registry(tool_node)
        workpad = dict(state.get("workpad", {}))

        if not tool_name:
            logger.warning("[Step %s] tool_runner -> missing pending tool call", step)
            tool_result = ToolResult(
                type="tool_runner",
                facts={},
                assumptions={},
                source={"tool": "tool_runner"},
                errors=["Missing pending tool call."],
            )
            return {
                "last_tool_result": tool_result.model_dump(),
                "pending_tool_call": None,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
            }

        if tool_name not in allowed:
            logger.warning("[Step %s] tool_runner -> blocked disallowed tool %s for profile=%s", step, tool_name, profile)
            tool_result = ToolResult(
                type=tool_name,
                facts={},
                assumptions=tool_args if isinstance(tool_args, dict) else {},
                source={"tool": tool_name},
                errors=[f"Tool '{tool_name}' is not allowed for task_profile '{profile}'."],
            )
            workpad.setdefault("tool_results", []).append(tool_result.model_dump())
            workpad.setdefault("events", []).append({"node": "tool_runner", "action": f"blocked {tool_name}"})
            return {
                "last_tool_result": tool_result.model_dump(),
                "pending_tool_call": None,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
                "last_tool_signature": _tool_signature(state),
                "workpad": workpad,
            }

        if tool_name not in registry:
            tool_result = ToolResult(
                type=tool_name,
                facts={},
                assumptions=tool_args if isinstance(tool_args, dict) else {},
                source={"tool": tool_name},
                errors=[f"Tool '{tool_name}' is not registered in the current runtime."],
            )
            workpad.setdefault("tool_results", []).append(tool_result.model_dump())
            workpad.setdefault("events", []).append({"node": "tool_runner", "action": f"missing tool {tool_name}"})
            return {
                "last_tool_result": tool_result.model_dump(),
                "pending_tool_call": None,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
                "last_tool_signature": _tool_signature(state),
                "workpad": workpad,
            }

        result = await tool_node.ainvoke(state)
        messages = result.get("messages", [])
        tool_message = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None)
        tool_result = _normalize_tool_output(
            tool_name,
            getattr(tool_message, "content", ""),
            tool_args if isinstance(tool_args, dict) else {},
        )
        workpad.setdefault("tool_results", []).append(tool_result.model_dump())
        workpad.setdefault("events", []).append({"node": "tool_runner", "action": f"ran {tool_name}"})

        tracker = state.get("cost_tracker")
        if tracker:
            tracker.record_mcp_call()

        if tool_message is not None:
            normalized_message = ToolMessage(
                content=json.dumps(tool_result.model_dump(), ensure_ascii=True),
                tool_call_id=tool_message.tool_call_id,
                name=tool_name,
            )
            messages = [normalized_message]

        logger.info("[Step %s] tool_runner -> %s errors=%s", step, tool_name, bool(tool_result.errors))
        return {
            "messages": messages,
            "last_tool_result": tool_result.model_dump(),
            "pending_tool_call": None,
            "tool_fail_count": state.get("tool_fail_count", 0) + (1 if tool_result.errors else 0),
            "last_tool_signature": _tool_signature(state),
            "workpad": workpad,
        }

    return tool_runner
