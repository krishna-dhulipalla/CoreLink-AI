"""
Tool Executor Node: Truncation, Failure Tracking, Deduplication
================================================================
Wraps LangGraph's ToolNode with output truncation and failure detection.
"""

import logging

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.cost import CostTracker
from agent.nodes.reasoner import TOOL_FAMILIES, _increment_step
from agent.memory.schema import _normalize_task_type
from agent.guardrails import sanitize_tool_output, tag_external_content
from context_manager import truncate_tool_output

logger = logging.getLogger(__name__)

MAX_TOOL_FAILURES = 2  # consecutive failures before forcing fallback


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
        allowed_tools = TOOL_FAMILIES.get(task_type)
        attempted_calls = _last_ai_tool_calls(state)

        if allowed_tools is not None:
            blocked = [tc for tc in attempted_calls if tc.get("name") not in allowed_tools]
            if blocked:
                logger.warning(
                    "[Step %s] Blocked disallowed tool(s) for task_type=%s: %s",
                    step,
                    task_type,
                    ", ".join(tc.get("name", "unknown") for tc in blocked),
                )
                blocked_messages = [
                    ToolMessage(
                        content=(
                            f"Error: Tool '{tc.get('name', 'unknown')}' is not allowed for "
                            f"task_type '{task_type}'. Use a permitted tool or answer directly."
                        ),
                        tool_call_id=tc.get("id", ""),
                        name=tc.get("name", "blocked_tool"),
                    )
                    for tc in blocked
                ]
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
