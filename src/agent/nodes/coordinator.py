"""
Coordinator Node: MaAS-Lite Layered Policy Router
===================================================
Analyzes queries and produces a layered execution plan.
Also contains the direct_responder and format_normalizer nodes.
"""

import logging
import time
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.cost import CostTracker
from agent.memory.schema import _normalize_task_type
from agent.model_config import (
    _extract_json_payload,
    _structured_output_mode,
    get_client_kwargs,
    get_model_name,
)
from agent.operators import validate_layers, DEFAULT_PLANS
from agent.prompts import (
    COORDINATOR_PROMPT,
    COORDINATOR_JSON_FALLBACK_PROMPT,
    DIRECT_RESPONDER_PROMPT,
    FORMAT_NORMALIZATION_PROMPT,
    RouteDecision,
)
from agent.nodes.reasoner import _increment_step
from context_manager import count_tokens

logger = logging.getLogger(__name__)


def _contains_keyword(text: str, keyword: str) -> bool:
    """Match short keywords as whole words to avoid substring collisions.

    Example: `us` should not match `using`.
    """
    escaped = re.escape(keyword.lower())
    if re.fullmatch(r"[a-z0-9 ]+", keyword.lower()):
        pattern = rf"(?<!\w){escaped}(?!\w)"
        return re.search(pattern, text) is not None
    return keyword.lower() in text


def _is_reflection_message(msg) -> bool:
    return (
        isinstance(msg, AIMessage)
        and bool(msg.content)
        and msg.content.startswith("[Reflection]")
    )


def _latest_human_text(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return str(msg.content)
    return ""


def _looks_like_structured_output_request(text: str) -> bool:
    normalized = text.lower()
    return any(
        token in normalized for token in (
            "output format",
            "json format",
            "xml format",
            '{"answer"',
            "respond with json",
            "return json",
        )
    )


def _heuristic_task_type(task_text: str) -> str:
    normalized = (task_text or "").lower()
    if not normalized:
        return "general"
    if any(_contains_keyword(normalized, token) for token in ("acquisition", "merger", "compliance", "liability", "indemnification", "contract", "regulatory", "tax reasons", "deal structure", "eu", "us")):
        return "legal"
    if any(_contains_keyword(normalized, token) for token in ("implied volatility", "historical volatility", "iv percentile", "black-scholes", "greeks", "straddle", "strangle", "iron condor", "credit spread", "vega", "delta", "theta", "gamma", "call option", "put option")):
        return "options"
    if any(_contains_keyword(normalized, token) for token in ("reference file", "pdf", "page", "table", "document", "spreadsheet", "csv", "row", "column", "worksheet")):
        return "document"
    if any(_contains_keyword(normalized, token) for token in ("latest", "current news", "search the web", "search internet", "source", "citation", "look up")):
        return "retrieval"
    if any(_contains_keyword(normalized, token) for token in ("calculate", "formula", "ratio", "annual report", "roe", "roa", "numerical", "financial leverage effect", "inventory turnover", "equity multiplier")):
        return "quantitative"
    return "general"


def _heuristic_route_decision(task_text: str) -> dict:
    """Fallback route when structured coordinator output is not recoverable."""
    normalized = (task_text or "").strip().lower()
    if normalized in {"hi", "hello", "hey", "thanks", "thank you"}:
        return {
            "layers": ["direct_answer"],
            "needs_formatting": False,
            "confidence": 0.85,
            "estimated_steps": 1,
            "early_exit_allowed": True,
            "task_type": "general",
        }

    task_type = _heuristic_task_type(task_text)
    needs_formatting = _looks_like_structured_output_request(task_text)
    layers = ["react_reason", "verifier_check"]
    if needs_formatting:
        layers.append("format_normalize")

    estimated_steps = {
        "quantitative": 3,
        "legal": 4,
        "options": 4,
        "document": 4,
        "retrieval": 4,
        "general": 3,
    }.get(task_type, 3)

    return {
        "layers": layers,
        "needs_formatting": needs_formatting,
        "confidence": 0.35,
        "estimated_steps": estimated_steps,
        "early_exit_allowed": task_type in {"general", "quantitative"},
        "task_type": task_type,
    }


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def coordinator(state: AgentState) -> dict:
    """Graph node: MaAS-lite controller. Produces a layered execution plan."""
    step = _increment_step()
    tracker: CostTracker = state.get("cost_tracker")

    model_name = get_model_name("coordinator")
    llm = ChatOpenAI(
        model=model_name,
        **get_client_kwargs("coordinator"),
        temperature=0,
        max_tokens=300,
    )

    # Sprint 3: Retrieve compact route hints from memory
    memory_store = state.get("memory_store")
    memory_hint_block = ""
    if memory_store:
        task_text = ""
        for m in reversed(state["messages"]):
            if hasattr(m, "content") and m.content:
                task_text = m.content
                break
        if task_text:
            hints = memory_store.retrieve_router_hints(task_text)
            if hints:
                memory_hint_block = (
                    "\n\nPAST ROUTING MEMORY (use as guidance, not gospel):\n"
                    + "\n".join(f"- {h}" for h in hints)
                )

    messages = [SystemMessage(content=COORDINATOR_PROMPT + memory_hint_block)] + state["messages"]

    invocation_messages = messages
    t0 = time.monotonic()
    try:
        if _structured_output_mode("coordinator") == "native":
            verdict = llm.with_structured_output(RouteDecision).invoke(messages)
        else:
            invocation_messages = [
                SystemMessage(content=COORDINATOR_JSON_FALLBACK_PROMPT + memory_hint_block)
            ] + [m for m in state["messages"] if not isinstance(m, SystemMessage)]
            raw_response = llm.invoke(invocation_messages)
            verdict = RouteDecision.model_validate_json(
                _extract_json_payload(str(raw_response.content or ""))
            )
        latency = (time.monotonic() - t0) * 1000

        if isinstance(verdict, dict):
            layers = verdict.get("layers", ["react_reason", "verifier_check"])
            needs_fmt = verdict.get("needs_formatting", False)
            confidence = verdict.get("confidence", 0.5)
            estimated_steps = verdict.get("estimated_steps", 3)
            early_exit_allowed = verdict.get("early_exit_allowed", True)
            task_type = verdict.get("task_type", "general")
        else:
            layers = verdict.layers
            needs_fmt = verdict.needs_formatting
            confidence = verdict.confidence
            estimated_steps = verdict.estimated_steps
            early_exit_allowed = verdict.early_exit_allowed
            task_type = getattr(verdict, "task_type", "general")

        success = True
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        task_text = _latest_human_text(state["messages"])
        heuristic = _heuristic_route_decision(task_text)
        logger.warning(
            "Coordinator routing failed: %s. Using heuristic fallback plan task_type=%s layers=%s.",
            e,
            heuristic["task_type"],
            heuristic["layers"],
        )
        layers = heuristic["layers"]
        needs_fmt = heuristic["needs_formatting"]
        confidence = heuristic["confidence"]
        estimated_steps = heuristic["estimated_steps"]
        early_exit_allowed = heuristic["early_exit_allowed"]
        task_type = heuristic["task_type"]
        success = False

    # Validate layers against operator registry
    layers = validate_layers(layers)
    task_type = _normalize_task_type(task_type)

    # Record cost
    if tracker:
        verdict_payload = {
            "layers": layers,
            "needs_formatting": needs_fmt,
            "confidence": confidence,
            "estimated_steps": estimated_steps,
            "early_exit_allowed": early_exit_allowed,
            "task_type": task_type,
        }
        tracker.record(
            operator="coordinator",
            model_name=model_name,
            tokens_in=count_tokens(invocation_messages),
            tokens_out=count_tokens([HumanMessage(content=json.dumps(verdict_payload))]),
            latency_ms=latency,
            success=success,
        )

    logger.info(
        f"[Step {step}] coordinator -> layers={layers}, "
        f"confidence={confidence:.2f}, needs_formatting={needs_fmt}, task_type={task_type}"
    )

    return {
        "selected_layers": layers,
        "format_required": needs_fmt,
        "policy_confidence": confidence,
        "estimated_steps": estimated_steps,
        "early_exit_allowed": early_exit_allowed,
        "task_type": task_type,
    }


def route_task(state: AgentState) -> str:
    """Conditional edge: choose first execution node based on selected layers."""
    layers = state.get("selected_layers", [])

    if not layers:
        return "reasoner"

    first_layer = layers[0]
    if first_layer == "direct_answer":
        return "direct_responder"
    return "reasoner"


# ---------------------------------------------------------------------------
# Direct Responder (fast path -- no tools, lean prompt)
# ---------------------------------------------------------------------------

def direct_responder(state: AgentState) -> dict:
    """Graph node: Fast execution path for simple queries. No tool bindings."""
    step = _increment_step()
    tracker: CostTracker = state.get("cost_tracker")

    model_name = get_model_name("direct")
    llm = ChatOpenAI(
        model=model_name,
        **get_client_kwargs("direct"),
        temperature=0,
    )

    # Use the lean prompt that does NOT mention tools
    messages = [SystemMessage(content=DIRECT_RESPONDER_PROMPT)] + [
        m for m in state["messages"] if not isinstance(m, SystemMessage)
    ]

    t0 = time.monotonic()
    response = llm.invoke(messages)
    latency = (time.monotonic() - t0) * 1000
    if isinstance(response, AIMessage) and response.content:
        clean = re.sub(r"<think>.*?</think>\s*", "", str(response.content), flags=re.DOTALL).strip()
        clean = clean.replace("<think>", "").replace("</think>", "").strip()
        response = AIMessage(content=clean or str(response.content).strip())

    if tracker:
        tracker.record(
            operator="direct_answer",
            model_name=model_name,
            tokens_in=count_tokens(messages),
            tokens_out=count_tokens([response]),
            latency_ms=latency,
            success=True,
        )

    logger.info(f"[Step {step}] direct_responder -> fast answer generated.")
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Format Normalizer (conditional -- skips when not needed)
# ---------------------------------------------------------------------------

def format_normalizer(state: AgentState) -> dict:
    """Graph node: Strict JSON/XML formatting. Skips LLM call when not required."""
    step = _increment_step()
    tracker: CostTracker = state.get("cost_tracker")

    # CONDITIONAL: skip if coordinator said no formatting needed
    if not state.get("format_required", False):
        logger.info(f"[Step {step}] format_normalizer -> SKIPPED (format_required=False)")
        return {"messages": []}

    # Find the last real AI answer
    source_text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not _is_reflection_message(msg):
            source_text = msg.content
            break

    if not source_text:
        return {"messages": []}

    model_name = get_model_name("formatter")
    llm = ChatOpenAI(
        model=model_name,
        **get_client_kwargs("formatter"),
        temperature=0,
    )

    prompt = FORMAT_NORMALIZATION_PROMPT + f"\n\nSource Text to Format:\n{source_text}"

    t0 = time.monotonic()
    response = llm.invoke([HumanMessage(content=prompt)])
    latency = (time.monotonic() - t0) * 1000

    if tracker:
        tracker.record(
            operator="format_normalize",
            model_name=model_name,
            tokens_in=count_tokens([HumanMessage(content=prompt)]),
            tokens_out=count_tokens([response]),
            latency_ms=latency,
            success=True,
        )

    final_output = response.content.strip()
    # Strip Qwen3 <think> blocks if present
    final_output = re.sub(r"<think>.*?</think>\s*", "", final_output, flags=re.DOTALL).strip()
    if final_output.startswith("```json"):
        final_output = final_output[7:-3].strip()
    elif final_output.startswith("```xml"):
        final_output = final_output[6:-3].strip()
    elif final_output.startswith("```"):
        final_output = final_output[3:-3].strip()

    logger.info(f"[Step {step}] format_normalizer -> applied formatting check.")
    return {"messages": [AIMessage(content=final_output)]}
