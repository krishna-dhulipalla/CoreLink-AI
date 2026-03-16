"""
Tool Runner Node
================
Executes exactly one tool call and normalizes the result into a ToolResult
contract before returning control to the solver.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agent.contracts import ToolResult
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import (
    _merge_unique_assumptions,
    allowed_tools_for_template,
    derive_assumption_ledger_entries,
    merge_tool_result_into_evidence,
)
from agent.state import AgentState
from agent.tool_normalization import normalize_tool_output

logger = logging.getLogger(__name__)


def _tool_signature(state: AgentState) -> str:
    pending = state.get("pending_tool_call") or {}
    return f"{pending.get('name', '')}:{json.dumps(pending.get('arguments', {}), sort_keys=True)}"


def _tool_registry(tool_node: ToolNode) -> dict[str, Any]:
    registry = getattr(tool_node, "tools_by_name", None)
    if isinstance(registry, dict):
        return registry
    tools = getattr(tool_node, "tools", []) or []
    return {getattr(tool, "name", ""): tool for tool in tools if getattr(tool, "name", "")}


def _extract_tool_call_id(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[-1].get("id", "unknown")
    return "unknown"



def make_tool_runner(tool_node: ToolNode):
    async def tool_runner(state: AgentState) -> dict:
        step = increment_runtime_step()
        profile = state.get("task_profile", "general")
        pending = state.get("pending_tool_call") or {}
        tool_name = str(pending.get("name", "")).strip()
        tool_args = pending.get("arguments", {})
        allowed = allowed_tools_for_template(state.get("execution_template"), profile)
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
                "messages": [ToolMessage(content=json.dumps(tool_result.model_dump(), ensure_ascii=True), name="unknown", tool_call_id=_extract_tool_call_id(state))],
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
                "messages": [ToolMessage(content=json.dumps(tool_result.model_dump(), ensure_ascii=True), name=tool_name, tool_call_id=_extract_tool_call_id(state))],
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
                "messages": [ToolMessage(content=json.dumps(tool_result.model_dump(), ensure_ascii=True), name=tool_name, tool_call_id=_extract_tool_call_id(state))],
                "last_tool_result": tool_result.model_dump(),
                "pending_tool_call": None,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
                "last_tool_signature": _tool_signature(state),
                "workpad": workpad,
            }

        result = await tool_node.ainvoke(state)
        messages = result.get("messages", [])
        tool_message = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None)
        tool_result = normalize_tool_output(
            tool_name,
            getattr(tool_message, "content", ""),
            tool_args if isinstance(tool_args, dict) else {},
        )
        tool_result.source.setdefault("solver_stage", state.get("solver_stage", "COMPUTE"))
        updated_evidence_pack, updated_provenance_map = merge_tool_result_into_evidence(
            state.get("evidence_pack", {}),
            tool_result,
            tool_args if isinstance(tool_args, dict) else {},
            state.get("provenance_map", {}),
        )
        added_assumptions = derive_assumption_ledger_entries(
            tool_name,
            tool_args if isinstance(tool_args, dict) else {},
            state.get("evidence_pack", {}),
        )
        updated_assumption_ledger = _merge_unique_assumptions(
            state.get("assumption_ledger", []),
            added_assumptions,
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
            "evidence_pack": updated_evidence_pack,
            "assumption_ledger": updated_assumption_ledger,
            "provenance_map": updated_provenance_map,
            "pending_tool_call": None,
            "tool_fail_count": state.get("tool_fail_count", 0) + (1 if tool_result.errors else 0),
            "last_tool_signature": _tool_signature(state),
            "workpad": workpad,
        }

    return tool_runner
