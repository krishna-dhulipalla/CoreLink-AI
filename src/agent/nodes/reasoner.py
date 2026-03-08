"""
Reasoner Node: The core LLM "Brain"
=====================================
Calls the LLM with tool bindings, applies the OSS tool-call patcher.
"""

import json
import logging
import os
"""
Reasoner Node: The core LLM "Brain"
=====================================
Calls the LLM with tool bindings, applies the OSS tool-call patcher.
"""

import json
import logging
import os
import time
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.cost import CostTracker
from agent.prompts import SYSTEM_PROMPT, MODEL_NAME
from agent.pruning import prune_for_reasoner
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
    """Instantiate the LLM with tool bindings."""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=1000,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    return llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# OSS Model Patcher
# ---------------------------------------------------------------------------

def patch_oss_tool_calls(response: AIMessage, tools: list) -> AIMessage:
    """
    Middleware to fix 'leaked' JSON arguments from OSS models.
    If the model output contains a valid JSON matching a tool's schema
    but lacks a formal `tool_calls` block, we synthetically wrap it.
    """
    if response.tool_calls or not response.content:
        return response

    content = str(response.content).strip()

    if content.startswith("{") and content.endswith("}"):
        try:
            payload = json.loads(content)
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

    def reasoner(state: AgentState) -> dict:
        """The 'Brain' node – calls the LLM with the current conversation."""
        step = _increment_step()
        tracker: CostTracker = state.get("cost_tracker")
        model = build_model(tools)

        # Sprint 4: Prune state before building LLM prompt
        messages = prune_for_reasoner(state["messages"])
        messages = with_system_prompt(messages)

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
            logger.info(f"[Step {step}] reasoner → tool_call: {', '.join(tool_names)}")
        else:
            preview = (response.content or "")[:100]
            logger.info(f"[Step {step}] reasoner → final answer: {preview}...")

        if tracker:
            tracker.record(
                operator="react_reason",
                tokens_in=count_tokens(messages),
                tokens_out=_estimate_response_tokens(response),
                latency_ms=latency,
                success=True,
            )

        return {"messages": [response]}

    return reasoner

