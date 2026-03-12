"""
Reasoner Node: The core LLM "Brain"
=====================================
Calls the LLM with tool bindings, applies the OSS tool-call patcher.
"""

import json
import logging
import time
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.cost import CostTracker
from agent.model_config import get_client_kwargs, get_model_name, _tool_call_mode
from agent.pruning import prune_for_reasoner
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)

# Step counter for logging (reset per run_agent call)
_step_counter = 0


def reset_step_counter():
    """Reset the step counter (called at the start of each run)."""
    global _step_counter
    _step_counter = 0


def get_step_counter() -> int:
    return _step_counter


def _increment_step() -> int:
    global _step_counter
    _step_counter += 1
    return _step_counter


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def build_model(tools: list):
    """Instantiate the LLM, optionally with native tool bindings."""
    llm = ChatOpenAI(
        model=get_model_name("executor"),
        **get_client_kwargs("executor"),
        temperature=0,
        max_tokens=1000,
    )
    mode = _tool_call_mode("executor")
    if mode == "native":
        return llm.bind_tools(tools)
    # In prompt mode, we do NOT call bind_tools -- the tools are
    # injected into the system prompt and parsed from text output.
    logger.info(f"[ToolMode] Using prompt-based tool calling (mode={mode})")
    return llm


def _build_tool_prompt_block(tools: list) -> str:
    """Build a system-prompt block describing available tools for prompt-based calling."""
    lines = [
        "You have access to the following tools. To use a tool, respond with ONLY a JSON object "
        "(no markdown fences, no explanation before or after) in this exact format:",
        '{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}',
        "",
        "If you do NOT need a tool, respond with a normal text answer.",
        "",
        "Available tools:",
    ]
    for t in tools:
        name = getattr(t, "name", "unknown")
        desc = getattr(t, "description", "")
        # Extract parameter schema
        schema = {}
        if hasattr(t, "args_schema") and t.args_schema:
            if isinstance(t.args_schema, dict):
                schema = t.args_schema
            else:
                schema = t.args_schema.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])
        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            req_marker = " (required)" if pname in required else " (optional)"
            pdesc = pinfo.get("description", "")
            param_parts.append(f"    - {pname}: {ptype}{req_marker}. {pdesc}")
        params_block = "\n".join(param_parts) if param_parts else "    (no parameters)"
        lines.append(f"\n  {name}: {desc}")
        lines.append(f"  Parameters:\n{params_block}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OSS Model Patcher
# ---------------------------------------------------------------------------

def patch_oss_tool_calls(response: AIMessage, tools: list) -> AIMessage:
    """
    Middleware to fix 'leaked' JSON arguments from OSS models.
    Handles two patterns:
      1. {"name": "tool_name", "arguments": {...}}  (prompt-mode format)
      2. {arg1: val1, arg2: val2}  (leaked schema match)
    """
    if response.tool_calls or not response.content:
        return response

    content = str(response.content).strip()
    # Strip Qwen3 <think> reasoning blocks before JSON detection
    import re
    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()

    if content.startswith("{") and content.endswith("}"):
        try:
            payload = json.loads(content)

            # Pattern 1: Explicit {"name": ..., "arguments": ...}
            if "name" in payload and "arguments" in payload:
                tool_name = payload["name"]
                tool_args = payload["arguments"]
                valid_names = set()
                for t in tools:
                    n = getattr(t, "name", None) or (t.get("function", {}).get("name") if isinstance(t, dict) else None)
                    if n:
                        valid_names.add(n)
                if tool_name in valid_names:
                    logger.info(f"[OSS Patch] Converted prompt-mode JSON to tool call for '{tool_name}'")
                    response.tool_calls = [
                        {
                            "name": tool_name,
                            "args": tool_args if isinstance(tool_args, dict) else {},
                            "id": f"call_{uuid.uuid4().hex[:10]}",
                            "type": "tool_call"
                        }
                    ]
                    response.content = ""
                    return response

            # Pattern 2: Leaked schema match (original logic)
            payload_keys = set(payload.keys())
            best_tool = None
            best_match_count = 0

            for t in tools:
                if hasattr(t, "args_schema") and t.args_schema:
                    if isinstance(t.args_schema, dict):
                        schema_keys = set(t.args_schema.get("properties", {}).keys())
                    else:
                        schema_keys = set(t.args_schema.model_json_schema().get("properties", {}).keys())
                elif isinstance(t, dict) and "function" in t:
                    schema_keys = set(t["function"].get("parameters", {}).get("properties", {}).keys())
                else:
                    continue

                match_count = len(payload_keys.intersection(schema_keys))
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_tool = t.name if hasattr(t, "name") else t["function"]["name"]

            if best_tool and best_match_count > 0:
                logger.warning(f"[OSS Patch] Converted naked JSON to tool call for '{best_tool}'")
                response.tool_calls = [
                    {
                        "name": best_tool,
                        "args": payload,
                        "id": f"call_{uuid.uuid4().hex[:10]}",
                        "type": "tool_call"
                    }
                ]
                response.content = ""
                return response
        except json.JSONDecodeError:
            pass

    return response


def _estimate_response_tokens(response: AIMessage) -> int:
    """Approximate token usage for an LLM response, including tool-call payloads."""
    parts = []
    if response.content:
        parts.append(str(response.content))
    if response.tool_calls:
        parts.append(json.dumps(response.tool_calls))
    if not parts:
        return 0
    return count_tokens([AIMessage(content="\n".join(parts))])


# ---------------------------------------------------------------------------
# Prompt Helper
# ---------------------------------------------------------------------------

def with_system_prompt(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Ensure the base system prompt is always the first message."""
    if (
        messages
        and isinstance(messages[0], SystemMessage)
        and messages[0].content == SYSTEM_PROMPT
    ):
        return messages
    return [SystemMessage(content=SYSTEM_PROMPT)] + messages


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Reasoner Node Factory
# ---------------------------------------------------------------------------

def make_reasoner(tools: list):
    """Factory: returns a reasoner node that uses the given tool list."""

    # Pre-compute tool prompt block for prompt-mode calling
    use_prompt_tools = _tool_call_mode("executor") == "prompt"
    tool_prompt_block = _build_tool_prompt_block(tools) if use_prompt_tools else ""

    def reasoner(state: AgentState) -> dict:
        """The 'Brain' node -- calls the LLM with the current conversation."""
        step = _increment_step()
        tracker: CostTracker = state.get("cost_tracker")
        model_name = get_model_name("executor")
        model = build_model(tools)

        # Sprint 4: Prune state before building LLM prompt
        messages = prune_for_reasoner(state["messages"])
        messages = with_system_prompt(messages)

        # Prompt-mode: inject tool descriptions into system prompt
        if use_prompt_tools and tool_prompt_block:
            messages = messages[:1] + [SystemMessage(content=tool_prompt_block)] + messages[1:]

        # Sprint 3+4: Retrieve compact executor hints from memory (budget-capped)
        memory_store = state.get("memory_store")
        budget = state.get("budget_tracker")
        if memory_store:
            task_text = _latest_human_text(state["messages"])
            if task_text:
                hints = memory_store.retrieve_executor_hints(task_text)
                if hints:
                    hint_block = (
                        "TOOL-SELECTION MEMORY (compact hints from past runs):\n"
                        + "\n".join(f"- {h}" for h in hints)
                    )
                    hint_tokens = count_tokens([SystemMessage(content=hint_block)])
                    remaining = budget.hint_tokens_remaining() if budget else 200
                    if hint_tokens <= remaining:
                        messages = messages[:1] + [SystemMessage(content=hint_block)] + messages[1:]
                        if budget:
                            budget.record_hint_tokens(hint_tokens)
                        logger.info(f"[Memory] Injected executor hints ({hint_tokens} tokens).")
                    else:
                        logger.info(f"[Budget] Skipped executor hints ({hint_tokens} > {remaining} remaining).")

        t0 = time.monotonic()
        response = model.invoke(messages)
        latency = (time.monotonic() - t0) * 1000

        # Apply OSS patcher
        response = patch_oss_tool_calls(response, tools)

        if isinstance(response, AIMessage) and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info(f"[Step {step}] reasoner -> tool_call: {', '.join(tool_names)}")
        else:
            preview = (response.content or "")[:100]
            logger.info(f"[Step {step}] reasoner -> final answer: {preview}...")

        if tracker:
            tracker.record(
                operator="react_reason",
                model_name=model_name,
                tokens_in=count_tokens(messages),
                tokens_out=_estimate_response_tokens(response),
                latency_ms=latency,
                success=True,
            )

        return {"messages": [response]}

    return reasoner
