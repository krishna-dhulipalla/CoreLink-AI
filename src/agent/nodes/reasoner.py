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
from agent.memory.schema import _normalize_task_type
from agent.model_config import get_client_kwargs, get_model_name, _tool_call_mode
from agent.pruning import prune_for_reasoner
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task-family tool allowlists and domain prompts
# ---------------------------------------------------------------------------

TOOL_FAMILIES: dict[str, set[str] | None] = {
    "quantitative": {"calculator", "fetch_reference_file", "list_reference_files"},
    "legal": {"calculator", "fetch_reference_file", "list_reference_files"},
    "options": {
        "calculator", "black_scholes_price", "option_greeks", "mispricing_analysis",
        "analyze_strategy", "get_options_chain", "get_iv_surface", "get_expirations",
        "calculate_portfolio_greeks", "calculate_var", "run_stress_test",
        "execute_options_trade", "create_portfolio", "get_positions", "get_pnl_report",
        "calculate_risk_metrics", "calculate_max_drawdown",
    },
    "document": {"calculator", "fetch_reference_file", "list_reference_files"},
    "retrieval": {"calculator", "internet_search", "fetch_reference_file", "list_reference_files"},
    "general": None,  # None = all tools
}

_TRADING_SIM_TOOLS = {
    "create_portfolio",
    "execute_options_trade",
    "get_positions",
    "get_pnl_report",
}

DOMAIN_ADDENDA: dict[str, str] = {
    "quantitative": (
        "Extract all values from provided tables. Apply formulas step by step. "
        "Show intermediate calculations. Match the exact output format specified."
    ),
    "legal": (
        "Answer from domain knowledge first. Do NOT use internet_search "
        "for domain knowledge you already have.\n"
        "Structure your answer using these sections:\n"
        "1. STRUCTURE OPTIONS — deal/entity structuring alternatives\n"
        "2. TAX CONSEQUENCES — tax treatment of each alternative\n"
        "3. LIABILITY PROTECTION — liability isolation, indemnification, reps/warranties\n"
        "4. REGULATORY/DILIGENCE RISKS — cross-border regulatory issues, compliance, diligence\n"
        "5. RECOMMENDED NEXT STEPS — concrete execution actions\n"
        "No long preamble. Go directly into the sections. "
        "Each section should be 2-4 sentences, not multi-paragraph essays."
    ),
    "options": (
        "Include full Greeks analysis (Delta, Gamma, Theta, Vega) for every leg "
        "and aggregate them for the overall strategy position.\n"
        "Show P&L breakdown with breakevens.\n"
        "Your PRIMARY strategy must be fully analyzed using the provided tools "
        "(e.g., analyze_strategy, option_greeks).\n"
        "Provide at least one ALTERNATIVE strategy with a concrete quantitative "
        "tradeoff discussion (e.g., different max-loss, different Greeks profile, "
        "different breakeven). Use tools if the same input parameters apply; "
        "otherwise, provide a concrete numerical comparison from your analysis.\n"
        "Include risk management: max loss, position sizing, hedging.\n"
        "Verify that sell positions generate credit, buy positions generate debit."
    ),
    "document": (
        "Ground the answer in the provided files or tables. Extract the exact values "
        "needed from file content before answering. Prefer file tools over web search."
    ),
    "retrieval": (
        "Use search or file tools only when the prompt lacks the needed facts. "
        "Summarize the retrieved evidence directly instead of listing tools."
    ),
    "general": "",
}


def _looks_like_trade_simulation(task_text: str) -> bool:
    normalized = (task_text or "").lower()
    return any(
        token in normalized
        for token in (
            "portfolio",
            "position",
            "positions",
            "paper trade",
            "paper-trading",
            "simulate trade",
            "simulation",
            "execute trade",
            "trade execution",
            "fill price",
            "slippage",
            "p&l report",
            "pnl report",
            "cash balance",
            "mark-to-market",
            "unrealized pnl",
            "realized pnl",
        )
    )


def _allowed_tool_names_for_task(task_type: str, task_text: str = "") -> set[str] | None:
    """Return the effective tool allowlist for the task.

    This is stricter than the coarse family map alone. For example, generic
    options-strategy questions should not expose paper-trading tools unless the
    prompt is clearly about portfolio simulation or execution.
    """
    normalized_type = _normalize_task_type(task_type)
    allowed = TOOL_FAMILIES.get(normalized_type)
    if allowed is None:
        return None

    effective = set(allowed)
    if normalized_type == "options" and not _looks_like_trade_simulation(task_text):
        effective -= _TRADING_SIM_TOOLS
    return effective

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


def _executor_max_tokens(task_type: str) -> int:
    """Return a conservative per-task output budget for the executor.

    Legal synthesis needs a little more runway to finalize a structured answer.
    Most other task families should stay on the tighter budget to avoid paying
    for longer-than-necessary Qwen-style reasoning traces.
    """
    normalized_type = _normalize_task_type(task_type)
    if normalized_type == "legal":
        return 1500
    return 1000


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def build_model(tools: list, task_type: str = "general"):
    """Instantiate the LLM, optionally with native tool bindings."""
    llm = ChatOpenAI(
        model=get_model_name("executor"),
        **get_client_kwargs("executor"),
        temperature=0,
        max_tokens=_executor_max_tokens(task_type),
    )
    mode = _tool_call_mode("executor")
    if mode == "native":
        return llm.bind_tools(tools)
    # In prompt mode, we do NOT call bind_tools -- the tools are
    # injected into the system prompt and parsed from text output.
    logger.info(f"[ToolMode] Using prompt-based tool calling (mode={mode})")
    return llm


def _build_tool_prompt_block(tools: list, task_type: str = "general", task_text: str = "") -> str:
    """Build a system-prompt block describing available tools for prompt-based calling.

    Filters tools by task_type allowlist to reduce prompt noise.
    """
    # Filter tools by task family allowlist
    allowed = _allowed_tool_names_for_task(task_type, task_text)
    if allowed is not None:
        tools = [t for t in tools if getattr(t, "name", "") in allowed]

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


def _filter_tools_for_task_type(tools: list, task_type: str = "general", task_text: str = "") -> list:
    """Return the runtime tool set allowed for the given task type."""
    allowed = _allowed_tool_names_for_task(task_type, task_text)
    if allowed is None:
        return list(tools)
    return [t for t in tools if getattr(t, "name", "") in allowed]


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
                # Do not silently remap an explicit tool envelope to a different tool.
                logger.warning(
                    "[OSS Patch] Ignoring explicit tool envelope for hidden/unknown tool '%s'.",
                    tool_name,
                )
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
    # NOTE: tool_prompt_block is rebuilt per-call now based on task_type

    def reasoner(state: AgentState) -> dict:
        """The 'Brain' node -- calls the LLM with the current conversation."""
        step = _increment_step()
        tracker: CostTracker = state.get("cost_tracker")
        model_name = get_model_name("executor")
        task_type = _normalize_task_type(state.get("task_type", "general"))
        task_text = _latest_human_text(state["messages"])
        filtered_tools = _filter_tools_for_task_type(tools, task_type, task_text)
        model = build_model(filtered_tools, task_type=task_type)

        # Sprint 4: Prune state before building LLM prompt
        messages = prune_for_reasoner(state["messages"])
        messages = with_system_prompt(messages)

        # Sprint 5: Inject domain-specific addendum based on task type
        addendum = DOMAIN_ADDENDA.get(task_type, "")
        if addendum:
            messages = messages[:1] + [SystemMessage(content=addendum)] + messages[1:]

        # Prompt-mode: inject filtered tool descriptions into system prompt
        if use_prompt_tools:
            filtered_block = _build_tool_prompt_block(filtered_tools, task_type, task_text)
            if filtered_block:
                messages = messages[:1] + [SystemMessage(content=filtered_block)] + messages[1:]

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
        response = patch_oss_tool_calls(response, filtered_tools)

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
