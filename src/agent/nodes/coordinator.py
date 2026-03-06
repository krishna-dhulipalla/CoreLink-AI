"""
Coordinator Node: MaAS-Lite Layered Policy Router
===================================================
Analyzes queries and produces a layered execution plan.
Also contains the direct_responder and format_normalizer nodes.
"""

import logging
import os
import time
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.cost import CostTracker
from agent.operators import validate_layers, DEFAULT_PLANS
from agent.prompts import (
    COORDINATOR_PROMPT,
    DIRECT_RESPONDER_PROMPT,
    FORMAT_NORMALIZATION_PROMPT,
    MODEL_NAME,
    RouteDecision,
)
from agent.nodes.reasoner import _increment_step
from context_manager import count_tokens

logger = logging.getLogger(__name__)


def _is_reflection_message(msg) -> bool:
    return (
        isinstance(msg, AIMessage)
        and bool(msg.content)
        and msg.content.startswith("[Reflection]")
    )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def coordinator(state: AgentState) -> dict:
    """Graph node: MaAS-lite controller. Produces a layered execution plan."""
    step = _increment_step()
    tracker: CostTracker = state.get("cost_tracker")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=300,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    ).with_structured_output(RouteDecision)

    messages = [SystemMessage(content=COORDINATOR_PROMPT)] + state["messages"]

    t0 = time.monotonic()
    try:
        verdict = llm.invoke(messages)
        latency = (time.monotonic() - t0) * 1000

        if isinstance(verdict, dict):
            layers = verdict.get("layers", ["react_reason", "reflection_review"])
            needs_fmt = verdict.get("needs_formatting", False)
            confidence = verdict.get("confidence", 0.5)
            estimated_steps = verdict.get("estimated_steps", 3)
            early_exit_allowed = verdict.get("early_exit_allowed", True)
        else:
            layers = verdict.layers
            needs_fmt = verdict.needs_formatting
            confidence = verdict.confidence
            estimated_steps = verdict.estimated_steps
            early_exit_allowed = verdict.early_exit_allowed

        success = True
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        logger.warning(f"Coordinator routing failed: {e}. Using default heavy_research plan.")
        layers = DEFAULT_PLANS["heavy_research"]
        needs_fmt = False
        confidence = 0.0
        estimated_steps = 3
        early_exit_allowed = False
        success = False

    # Validate layers against operator registry
    layers = validate_layers(layers)

    # Record cost
    if tracker:
        verdict_payload = {
            "layers": layers,
            "needs_formatting": needs_fmt,
            "confidence": confidence,
            "estimated_steps": estimated_steps,
            "early_exit_allowed": early_exit_allowed,
        }
        tracker.record(
            operator="coordinator",
            tokens_in=count_tokens(messages),
            tokens_out=count_tokens([HumanMessage(content=json.dumps(verdict_payload))]),
            latency_ms=latency,
            success=success,
        )

    logger.info(
        f"[Step {step}] coordinator → layers={layers}, "
        f"confidence={confidence:.2f}, needs_formatting={needs_fmt}"
    )

    return {
        "selected_layers": layers,
        "format_required": needs_fmt,
        "policy_confidence": confidence,
        "estimated_steps": estimated_steps,
        "early_exit_allowed": early_exit_allowed,
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
# Direct Responder (fast path — no tools, lean prompt)
# ---------------------------------------------------------------------------

def direct_responder(state: AgentState) -> dict:
    """Graph node: Fast execution path for simple queries. No tool bindings."""
    step = _increment_step()
    tracker: CostTracker = state.get("cost_tracker")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    # Use the lean prompt that does NOT mention tools
    messages = [SystemMessage(content=DIRECT_RESPONDER_PROMPT)] + [
        m for m in state["messages"] if not isinstance(m, SystemMessage)
    ]

    t0 = time.monotonic()
    response = llm.invoke(messages)
    latency = (time.monotonic() - t0) * 1000

    if tracker:
        tracker.record(
            operator="direct_answer",
            tokens_in=count_tokens(messages),
            tokens_out=count_tokens([response]),
            latency_ms=latency,
            success=True,
        )

    logger.info(f"[Step {step}] direct_responder → fast answer generated.")
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Format Normalizer (conditional — skips when not needed)
# ---------------------------------------------------------------------------

def format_normalizer(state: AgentState) -> dict:
    """Graph node: Strict JSON/XML formatting. Skips LLM call when not required."""
    step = _increment_step()
    tracker: CostTracker = state.get("cost_tracker")

    # CONDITIONAL: skip if coordinator said no formatting needed
    if not state.get("format_required", False):
        logger.info(f"[Step {step}] format_normalizer → SKIPPED (format_required=False)")
        return {"messages": []}

    # Find the last real AI answer
    source_text = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not _is_reflection_message(msg):
            source_text = msg.content
            break

    if not source_text:
        return {"messages": []}

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    prompt = FORMAT_NORMALIZATION_PROMPT + f"\n\nSource Text to Format:\n{source_text}"

    t0 = time.monotonic()
    response = llm.invoke([HumanMessage(content=prompt)])
    latency = (time.monotonic() - t0) * 1000

    if tracker:
        tracker.record(
            operator="format_normalize",
            tokens_in=count_tokens([HumanMessage(content=prompt)]),
            tokens_out=count_tokens([response]),
            latency_ms=latency,
            success=True,
        )

    final_output = response.content.strip()
    if final_output.startswith("```json"):
        final_output = final_output[7:-3].strip()
    elif final_output.startswith("```xml"):
        final_output = final_output[6:-3].strip()
    elif final_output.startswith("```"):
        final_output = final_output[3:-3].strip()

    logger.info(f"[Step {step}] format_normalizer → applied formatting check.")
    return {"messages": [AIMessage(content=final_output)]}
