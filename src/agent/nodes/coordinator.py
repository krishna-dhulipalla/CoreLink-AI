"""
Coordinator Node: MaAS-Inspired Dynamic Router
================================================
Classifies queries into execution paths (direct vs heavy_research).
"""

import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.prompts import (
    COORDINATOR_PROMPT,
    FORMAT_NORMALIZATION_PROMPT,
    MODEL_NAME,
    RouteDecision,
    SYSTEM_PROMPT,
)
from agent.nodes.reasoner import _increment_step, with_system_prompt

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
    """Graph node: MaAS-inspired router. Classifies the query and sets the route."""
    step = _increment_step()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=200,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    ).with_structured_output(RouteDecision)

    messages = [SystemMessage(content=COORDINATOR_PROMPT)] + state["messages"]

    try:
        verdict = llm.invoke(messages)
        if isinstance(verdict, dict):
            route = verdict.get("route", "heavy_research")
        else:
            route = verdict.route
    except Exception as e:
        logger.warning(f"Coordinator routing failed: {e}. Defaulting to heavy_research.")
        route = "heavy_research"

    logger.info(f"[Step {step}] coordinator → selected route: {route}")
    return {"route": route}


def route_task(state: AgentState) -> str:
    """Conditional edge out of the coordinator."""
    route = state.get("route", "heavy_research")
    if route == "direct":
        return "direct_responder"
    return "reasoner"


# ---------------------------------------------------------------------------
# Direct Responder (fast path)
# ---------------------------------------------------------------------------

def direct_responder(state: AgentState) -> dict:
    """Graph node: Fast execution path for simple queries without tools."""
    step = _increment_step()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    messages = with_system_prompt(state["messages"])
    response = llm.invoke(messages)

    logger.info(f"[Step {step}] direct_responder → fast answer generated.")
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Format Normalizer (final gate)
# ---------------------------------------------------------------------------

def format_normalizer(state: AgentState) -> dict:
    """Graph node: The final pass guaranteeing JSON/XML shape matching."""
    step = _increment_step()

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
    response = llm.invoke([HumanMessage(content=prompt)])

    final_output = response.content.strip()
    if final_output.startswith("```json"):
        final_output = final_output[7:-3].strip()
    elif final_output.startswith("```xml"):
        final_output = final_output[6:-3].strip()
    elif final_output.startswith("```"):
        final_output = final_output[3:-3].strip()

    logger.info(f"[Step {step}] format_normalizer → applied formatting check.")
    return {"messages": [AIMessage(content=final_output)]}
