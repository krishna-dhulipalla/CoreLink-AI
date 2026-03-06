"""
Reflector Node: Self-Critique & Feedback Loop
===============================================
Evaluates the agent's draft answer and either passes or requests revision.
"""

import logging
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.prompts import REFLECTION_PROMPT, MAX_REFLECTIONS, MODEL_NAME

logger = logging.getLogger(__name__)


def _is_reflection_message(msg: BaseMessage) -> bool:
    return (
        isinstance(msg, AIMessage)
        and bool(msg.content)
        and msg.content.startswith("[Reflection]")
    )


def _build_reflection_context(
    messages: list[BaseMessage],
    keep_last: int = 6,
) -> list[BaseMessage]:
    """Build a protocol-safe context slice for the reflection LLM call.

    We deliberately exclude:
    - ``AIMessage`` entries that contain ``tool_calls`` (they require matching
      ``ToolMessage`` entries in OpenAI chat-completions)
    - ``ToolMessage`` entries (not needed for the reflection verdict)
    - internal ``[Reflection]`` messages from previous passes
    """
    filtered: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                continue
            if _is_reflection_message(msg):
                continue
            filtered.append(msg)
    return filtered[-keep_last:]


# ---------------------------------------------------------------------------
# Reflector Node
# ---------------------------------------------------------------------------

def reflector(state: AgentState) -> dict:
    """Graph node: critique the draft answer before submission."""
    messages = state["messages"]
    count = state.get("reflection_count", 0)

    if count >= MAX_REFLECTIONS:
        logger.info(
            f"Reflection limit reached ({MAX_REFLECTIONS}). "
            "Submitting answer as-is."
        )
        return {"reflection_count": count}

    reflection_messages = [SystemMessage(content=REFLECTION_PROMPT)]
    reflection_messages.extend(_build_reflection_context(messages))

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=500,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    verdict = llm.invoke(reflection_messages)
    verdict_text = verdict.content.strip() if verdict.content else "PASS: no verdict"

    logger.info(f"Reflection #{count + 1}: {verdict_text}")

    return {
        "messages": [AIMessage(content=f"[Reflection]: {verdict_text}")],
        "reflection_count": count + 1,
    }


# ---------------------------------------------------------------------------
# Conditional Edge
# ---------------------------------------------------------------------------

def should_revise(state: AgentState) -> str:
    """Conditional edge after reflector: REVISE loops back, PASS goes to format_normalizer."""
    count = state.get("reflection_count", 0)

    if count >= MAX_REFLECTIONS:
        return "format_normalizer"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.content:
        text = last_msg.content.upper()
        if "REVISE:" in text:
            logger.info("Reflector requested revision → looping back to reasoner")
            return "reasoner"

    return "format_normalizer"
