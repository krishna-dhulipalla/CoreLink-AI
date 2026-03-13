"""
Tool Executor Node: Truncation, Failure Tracking, Deduplication
================================================================
Wraps LangGraph's ToolNode with output truncation and failure detection.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.cost import CostTracker
from agent.nodes.reasoner import _allowed_tool_names_for_task, _increment_step
from agent.memory.schema import _normalize_task_type
from agent.guardrails import sanitize_tool_output, tag_external_content
from context_manager import truncate_tool_output

logger = logging.getLogger(__name__)

MAX_TOOL_FAILURES = 2  # consecutive failures before forcing fallback

_FINANCE_ARG_ALIASES = {
    "spot": "S",
    "underlying_price": "S",
    "stock_price": "S",
    "current_price": "S",
    "strike": "K",
    "strike_price": "K",
    "days": "T_days",
    "days_to_expiration": "T_days",
    "days_to_expiry": "T_days",
    "expiry_days": "T_days",
    "rate": "r",
    "risk_free_rate": "r",
    "interest_rate": "r",
    "volatility": "sigma",
    "implied_volatility": "sigma",
    "iv": "sigma",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_signature(state: AgentState) -> str:
    """Build a string signature from the last AIMessage's tool calls (name + args).
    Used to detect duplicate identical tool calls.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            parts = []
            for tc in msg.tool_calls:
                args_str = str(sorted(tc.get("args", {}).items()))
                parts.append(f"{tc['name']}:{args_str}")
            return "|".join(parts)
    return ""


def _last_ai_tool_calls(state: AgentState) -> list[dict]:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return list(msg.tool_calls)
    return []


def _latest_human_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human" and getattr(msg, "content", None):
            return str(msg.content)
    return ""


def _tool_registry(tool_node: ToolNode) -> dict[str, Any]:
    registry = getattr(tool_node, "tools_by_name", None)
    if isinstance(registry, dict):
        return registry
    tools = getattr(tool_node, "tools", []) or []
    return {
        getattr(tool, "name", ""): tool
        for tool in tools
        if getattr(tool, "name", "")
    }


def _validate_tool_call_payload(tool_name: str, args: Any, tool: Any) -> str | None:
    if not isinstance(args, dict):
        return f"Invalid arguments for tool '{tool_name}': expected an object."

    # Catch wrapped tool envelopes like:
    # create_portfolio({name: 'internet_search', arguments: {...}})
    if set(args.keys()) == {"name", "arguments"} and isinstance(args.get("arguments"), dict):
        nested_name = str(args.get("name", "")).strip()
        if nested_name and nested_name != tool_name:
            return (
                f"Malformed tool payload for '{tool_name}': nested tool envelope "
                f"references '{nested_name}'."
            )

    args_schema = getattr(tool, "args_schema", None)
    if not args_schema:
        return None

    try:
        if isinstance(args_schema, dict):
            schema = args_schema
        else:
            schema = args_schema.model_json_schema()
    except Exception:
        schema = {}

    props = set(schema.get("properties", {}).keys())
    required = set(schema.get("required", []))
    provided = set(args.keys())

    if props:
        unexpected = sorted(provided - props)
        if unexpected:
            return f"Unexpected arguments for tool '{tool_name}': {', '.join(unexpected)}."
        missing = sorted(required - provided)
        if missing:
            return f"Missing required arguments for tool '{tool_name}': {', '.join(missing)}."

    if not isinstance(args_schema, dict):
        try:
            args_schema.model_validate(args)
        except Exception as exc:
            return f"Invalid arguments for tool '{tool_name}': {exc}"

    return None


def _normalize_finance_args(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_normalize_finance_args(item) for item in payload]
    if not isinstance(payload, dict):
        return payload

    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        mapped = _FINANCE_ARG_ALIASES.get(str(key), key)
        normalized[mapped] = _normalize_finance_args(value)
    return normalized


def _normalize_tool_call_args(tool_name: str, args: Any) -> Any:
    if tool_name in {
        "black_scholes_price",
        "option_greeks",
        "mispricing_analysis",
        "get_options_chain",
        "get_iv_surface",
        "analyze_strategy",
    }:
        return _normalize_finance_args(args)
    return args


def should_use_tools(state: AgentState) -> str:
    """Conditional edge: route to tool_executor or reflector.

    Before allowing another tool call, check if the agent is stuck in a
    repeated-failure loop. If tool_fail_count >= MAX_TOOL_FAILURES, force
    the agent to the reflector so it must produce a final answer.
    """
    selected_layers = state.get("selected_layers", [])
    # In Sprint 2, verifier_check replaces reflection_review in the loop
    wants_verification = not selected_layers or "verifier_check" in selected_layers

    # Sprint 4: Budget check — force through verifier for a clean exit
    budget = state.get("budget_tracker")
    if budget and budget.tool_calls_exhausted():
        budget.log_budget_exit("tool_calls", f"Reached {budget.tool_calls} tool calls. Forcing verifier exit.")
        return "verifier" if wants_verification else "format_normalizer"

    if state.get("tool_fail_count", 0) >= MAX_TOOL_FAILURES:
        logger.warning(
            f"[FailureGate] Tool failure limit ({MAX_TOOL_FAILURES}) reached. "
            "Forcing agent to verify final answer."
        )
        return "verifier" if wants_verification else "format_normalizer"

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_executor"
    return "verifier" if wants_verification else "format_normalizer"


# ---------------------------------------------------------------------------
# Wrapped Tool Executor
# ---------------------------------------------------------------------------

def make_tool_executor(tool_node: ToolNode):
    """Wrap the ToolNode to truncate verbose tool outputs and track failures."""

    async def tool_executor(state: AgentState) -> dict:
        step = _increment_step()
        tracker: CostTracker = state.get("cost_tracker")
        task_type = _normalize_task_type(state.get("task_type", "general"))
        task_text = _latest_human_text(state)
        allowed_tools = _allowed_tool_names_for_task(task_type, task_text)
        attempted_calls = _last_ai_tool_calls(state)
        registry = _tool_registry(tool_node)

        blocked_messages: list[ToolMessage] = []
        malformed_names: list[str] = []

        for tc in attempted_calls:
            tool_name = tc.get("name", "unknown")
            normalized_args = _normalize_tool_call_args(tool_name, tc.get("args", {}))
            tc["args"] = normalized_args
            if allowed_tools is not None and tool_name not in allowed_tools:
                malformed_names.append(tool_name)
                blocked_messages.append(
                    ToolMessage(
                        content=(
                            f"Error: Tool '{tool_name}' is not allowed for "
                            f"task_type '{task_type}'. Use a permitted tool or answer directly."
                        ),
                        tool_call_id=tc.get("id", ""),
                        name=tool_name,
                    )
                )
                continue

            tool = registry.get(tool_name)
            if tool is None:
                malformed_names.append(tool_name)
                blocked_messages.append(
                    ToolMessage(
                        content=f"Error: Tool '{tool_name}' is not registered in the current runtime.",
                        tool_call_id=tc.get("id", ""),
                        name=tool_name,
                    )
                )
                continue

            validation_error = _validate_tool_call_payload(tool_name, normalized_args, tool)
            if validation_error:
                malformed_names.append(tool_name)
                blocked_messages.append(
                    ToolMessage(
                        content=f"Error: {validation_error} Do not repeat this malformed call.",
                        tool_call_id=tc.get("id", ""),
                        name=tool_name,
                    )
                )

        if blocked_messages:
            logger.warning(
                "[Step %s] Blocked malformed/disallowed tool call(s) for task_type=%s: %s",
                step,
                task_type,
                ", ".join(malformed_names) or "unknown",
            )
            return {
                "messages": blocked_messages,
                "tool_fail_count": state.get("tool_fail_count", 0) + 1,
                "last_tool_signature": _make_tool_signature(state),
            }

        result = await tool_node.ainvoke(state)
        messages = result.get("messages", [])
        truncated_messages = []

        # Sprint 4: Record tool call in budget tracker
        budget = state.get("budget_tracker")
        if budget:
            budget.record_tool_call()

        call_signature = _make_tool_signature(state)
        previous_signature = state.get("last_tool_signature", "")

        any_error = False
        is_duplicate = (call_signature == previous_signature and call_signature != "")

        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Fix 3: Normalize non-string content (rich MCP responses) to str
                raw = msg.content
                if isinstance(raw, list):
                    raw = "\n".join(
                        item.get("text", str(item)) if isinstance(item, dict) else str(item)
                        for item in raw
                    )
                    msg = ToolMessage(
                        content=raw, tool_call_id=msg.tool_call_id, name=msg.name,
                    )
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                original_content = msg.content
                original_len = len(original_content)

                # Sprint 4 Fix: Guardrail scan on ORIGINAL content BEFORE truncation
                cleaned, was_sanitized = sanitize_tool_output(original_content)
                if was_sanitized:
                    msg = ToolMessage(
                        content=cleaned,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
                    truncated_messages.append(msg)
                    any_error = True  # treat sanitized content as error
                    if tracker:
                        tracker.record_mcp_call()
                    continue

                # Truncate after guardrail scan
                truncated_content = truncate_tool_output(original_content)
                if len(truncated_content) < original_len:
                    logger.info(
                        f"Truncated tool '{msg.name}' output: "
                        f"{original_len} → {len(truncated_content)} chars"
                    )
                if truncated_content.startswith("Error"):
                    any_error = True
                    logger.warning(
                        f"[Step {step}] tool_executor → {msg.name} ERROR: "
                        f"{truncated_content[:120]}"
                    )
                    if is_duplicate:
                        truncated_content += (
                            "\n\n[SYSTEM NOTE: You have called this tool with the same arguments before and it failed. "
                            "Do NOT repeat this call. Use a different tool or answer directly.]"
                        )
                else:
                    preview = truncated_content[:120].replace('\n', ' ')
                    logger.info(f"[Step {step}] tool_executor → {msg.name}: {preview}...")
                msg = ToolMessage(
                    content=truncated_content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )

                # Sprint 4: Tag external file content
                if msg.name and "file" in msg.name.lower():
                    msg = ToolMessage(
                        content=tag_external_content(str(msg.content)),
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )

                # Record MCP call in cost tracker
                if tracker:
                    tracker.record_mcp_call()

            truncated_messages.append(msg)

        current_fail_count = state.get("tool_fail_count", 0)
        new_fail_count = (current_fail_count + 1) if (any_error or is_duplicate) else 0

        return {
            "messages": truncated_messages,
            "tool_fail_count": new_fail_count,
            "last_tool_signature": call_signature,
        }

    return tool_executor
