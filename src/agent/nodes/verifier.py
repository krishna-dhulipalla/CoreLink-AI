"""
Step-Level Verifier Node
========================
Evaluates the Executor's latest step and emits a structured verdict:
PASS, REVISE, or BACKTRACK. On BACKTRACK, the message state is restored
to the last verified checkpoint and a warning is injected for the Executor.
"""

import logging
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_openai import ChatOpenAI

from agent.cost import CostTracker
from agent.prompts import MODEL_NAME, VERIFIER_PROMPT, VerdictDecision
from agent.state import AgentState, ReplaceMessages
from context_manager import count_tokens

logger = logging.getLogger(__name__)

BACKTRACK_WARNING = (
    "BACKTRACK WARNING: Your previous approach was evaluated as fundamentally flawed "
    "or hallucinated. The state has been reverted to the last verified checkpoint. "
    "Do NOT repeat the same mistake. Try a different tool or strategy."
)


def _is_internal_warning(msg: BaseMessage) -> bool:
    return bool(getattr(msg, "additional_kwargs", {}).get("is_warning", False))


def _last_verified_messages(stack: list[dict]) -> list[BaseMessage]:
    if not stack:
        return []
    checkpoint = stack[-1]
    return messages_from_dict(checkpoint["messages"])


def verifier(state: AgentState) -> dict:
    """Verify the latest executor step and update checkpoint state."""
    selected_layers = state.get("selected_layers", [])
    if "verifier_check" not in selected_layers:
        return {}

    tracker: CostTracker | None = state.get("cost_tracker")
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.0,
    ).with_structured_output(VerdictDecision)

    history = [m for m in state["messages"] if not _is_internal_warning(m)]
    prompt_messages = [SystemMessage(content=VERIFIER_PROMPT)] + history

    t0 = time.monotonic()
    try:
        verdict: VerdictDecision = llm.invoke(prompt_messages)
        success = True
    except Exception as exc:
        logger.warning("Verifier LLM failed: %s. Defaulting to PASS.", exc)
        verdict = VerdictDecision(
            verdict="PASS",
            reasoning=f"Fallback due to verifier error: {exc}",
        )
        success = False
    latency = (time.monotonic() - t0) * 1000

    if tracker:
        tracker.record(
            operator="verifier_check",
            tokens_in=count_tokens(prompt_messages),
            tokens_out=count_tokens([AIMessage(content=verdict.model_dump_json())]),
            latency_ms=latency,
            success=success,
        )

    logger.info(
        "[Verifier] Verdict: %s | Reason: %s",
        verdict.verdict,
        verdict.reasoning[:120],
    )

    # Sprint 3: Retrieve repair hints from verifier memory
    repair_hint_block = ""
    memory_store = state.get("memory_store")
    if memory_store and verdict.verdict in ("REVISE", "BACKTRACK"):
        task_text = ""
        for m in state["messages"]:
            if isinstance(m, HumanMessage) and m.content:
                task_text = m.content
                break
        if task_text:
            repair_hints = memory_store.retrieve_verifier_hints(task_text)
            if repair_hints:
                repair_hint_block = (
                    "\n\nPAST REPAIR MEMORY:\n"
                    + "\n".join(f"- {h}" for h in repair_hints)
                )

    stack = list(state.get("checkpoint_stack", []))

    if verdict.verdict == "PASS":
        serialized_messages = messages_to_dict(state["messages"])
        if not stack or stack[-1]["messages"] != serialized_messages:
            stack.append({"messages": serialized_messages})
        return {"checkpoint_stack": stack}

    if verdict.verdict == "REVISE":
        warning_msg = SystemMessage(
            content=f"VERIFIER REVISION REQUIRED:\n{verdict.reasoning}{repair_hint_block}",
            additional_kwargs={"is_warning": True},
        )
        return {"messages": [warning_msg]}

    if not stack:
        warning_msg = SystemMessage(
            content=f"VERIFIER BACKTRACK (no checkpoints available):\n{verdict.reasoning}",
            additional_kwargs={"is_warning": True},
        )
        return {"messages": [warning_msg]}

    warning_msg = SystemMessage(
        content=f"{BACKTRACK_WARNING}\n\nReason for backtrack: {verdict.reasoning}{repair_hint_block}",
        additional_kwargs={"is_warning": True},
    )
    restored_messages = _last_verified_messages(stack) + [warning_msg]
    return {
        "messages": ReplaceMessages(restored_messages),
        "checkpoint_stack": stack,
    }


def verify_routing(state: AgentState) -> str:
    """Route after verifier based on the latest verified message state."""
    last_msg = state["messages"][-1]
    if _is_internal_warning(last_msg):
        return "reasoner"

    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
        return "format_normalizer"

    return "reasoner"
