"""
Step-Level Verifier Node
========================
Evaluates the Executor's latest step and emits a structured verdict:
PASS, REVISE, or BACKTRACK. On BACKTRACK, the message state is restored
to the last verified checkpoint and a warning is injected for the Executor.
"""

import logging
import time
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_openai import ChatOpenAI

from agent.cost import CostTracker
from agent.model_config import (
    _extract_json_payload,
    _structured_output_mode,
    get_client_kwargs,
    get_model_name,
)
from agent.memory.schema import ExecutorMemory, VerifierMemory, _task_signature
from agent.prompts import VERIFIER_PROMPT, VerdictDecision
from agent.state import AgentState, ReplaceMessages
from agent.pruning import truncate_memory_fields
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


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content
    return ""


def _last_non_warning_message(messages: list[BaseMessage]) -> BaseMessage | None:
    for msg in reversed(messages):
        if not _is_internal_warning(msg):
            return msg
    return None


def _extract_best_answer(messages: list[BaseMessage]) -> str:
    """Walk messages backwards to find the best answer text for budget-exit.

    Preference order:
    1. Last AIMessage with text content and no tool_calls (real answer).
    2. Last AIMessage with text content (even if it also had tool_calls).
    3. Fallback string.
    """
    best_with_calls = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not _is_internal_warning(msg):
            if not msg.tool_calls:
                return str(msg.content)
            if best_with_calls is None:
                best_with_calls = str(msg.content)
    if best_with_calls:
        return best_with_calls
    return "I was unable to complete this task within the allowed verification budget."


def _latest_tool_fragment(messages: list[BaseMessage]) -> tuple[str, str] | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_names = [tc["name"] for tc in msg.tool_calls]
            args = []
            for tc in msg.tool_calls:
                arg_pairs = tc.get("args", {})
                arg_text = ", ".join(f"{k}={v}" for k, v in sorted(arg_pairs.items()))
                args.append(f"{tc['name']}({arg_text})".strip())
            tool_used = tool_names[0] if len(tool_names) == 1 else ", ".join(tool_names)
            arguments_pattern = "; ".join(args)[:240] or "no arguments"
            return tool_used, arguments_pattern
    return None


def _store_executor_memory(
    state: AgentState,
    verdict: VerdictDecision,
    task_text: str,
) -> None:
    memory_store = state.get("memory_store")
    if not memory_store or not task_text:
        return

    last_msg = _last_non_warning_message(state["messages"])
    if last_msg is None or not isinstance(last_msg, ToolMessage):
        return

    fragment = _latest_tool_fragment(state["messages"])
    if fragment is None:
        return

    tool_used, arguments_pattern = fragment
    if verdict.verdict == "PASS":
        outcome_quality = "good"
        success = True
    elif verdict.verdict == "REVISE":
        outcome_quality = "acceptable"
        success = False
    else:
        outcome_quality = "poor"
        success = False

    rec = ExecutorMemory(
        task_signature=_task_signature(task_text),
        partial_context_summary=task_text[:120],
        tool_used=tool_used,
        arguments_pattern=arguments_pattern,
        outcome_quality=outcome_quality,
        success=success,
    )
    truncate_memory_fields(rec)
    memory_store.store_executor(rec)


def _repair_action_summary(messages: list[BaseMessage]) -> str:
    fragment = _latest_tool_fragment(messages)
    if fragment is not None:
        tool_used, arguments_pattern = fragment
        return f"Switched to {tool_used} with {arguments_pattern}"[:240]

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return f"Revised answer: {msg.content[:200]}"
    return "Adjusted the strategy and produced a passing step."


def _store_verifier_memory_on_repair_success(
    state: AgentState,
    task_text: str,
) -> None:
    memory_store = state.get("memory_store")
    pending_feedback = state.get("pending_verifier_feedback")
    if not memory_store or not task_text or not pending_feedback:
        return

    rec = VerifierMemory(
        task_signature=_task_signature(task_text),
        failure_pattern=str(pending_feedback.get("reasoning", ""))[:240],
        verdict=pending_feedback.get("verdict", "REVISE"),
        repair_action=_repair_action_summary(state["messages"]),
        repair_worked=True,
    )
    truncate_memory_fields(rec)
    memory_store.store_verifier(rec)


def verifier(state: AgentState) -> dict:
    """Verify the latest executor step and update checkpoint state."""
    selected_layers = state.get("selected_layers", [])
    if "verifier_check" not in selected_layers:
        return {}

    tracker: CostTracker | None = state.get("cost_tracker")
    model_name = get_model_name("verifier")
    llm = ChatOpenAI(
        model=model_name,
        **get_client_kwargs("verifier"),
        temperature=0.0,
    )

    history = [m for m in state["messages"] if not _is_internal_warning(m)]
    prompt_messages = [SystemMessage(content=VERIFIER_PROMPT)] + history

    t0 = time.monotonic()
    try:
        if _structured_output_mode("verifier") == "native":
            verdict: VerdictDecision = llm.with_structured_output(VerdictDecision).invoke(prompt_messages)
        else:
            schema_prompt = SystemMessage(
                content=(
                    "Return ONLY valid JSON matching this schema. "
                    "Do not include markdown fences or extra commentary.\n"
                    f"JSON_SCHEMA={json.dumps(VerdictDecision.model_json_schema(), ensure_ascii=True)}"
                )
            )
            raw_response = llm.invoke([schema_prompt] + prompt_messages)
            verdict = VerdictDecision.model_validate_json(
                _extract_json_payload(str(raw_response.content or ""))
            )
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
            model_name=model_name,
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

    task_text = _latest_human_text(state["messages"])

    # Sprint 3+4: Retrieve repair hints from verifier memory (budget-capped)
    repair_hint_block = ""
    memory_store = state.get("memory_store")
    budget = state.get("budget_tracker")
    if memory_store and verdict.verdict in ("REVISE", "BACKTRACK"):
        if task_text:
            repair_hints = memory_store.retrieve_verifier_hints(task_text)
            if repair_hints:
                from langchain_core.messages import SystemMessage as _SM
                hint_text = "\n".join(f"- {h}" for h in repair_hints)
                hint_tokens = count_tokens([_SM(content=hint_text)])
                remaining = budget.hint_tokens_remaining() if budget else 200
                if hint_tokens <= remaining:
                    repair_hint_block = "\n\nPAST REPAIR MEMORY:\n" + hint_text
                    if budget:
                        budget.record_hint_tokens(hint_tokens)
                    logger.info(f"[Memory] Injected verifier repair hints ({hint_tokens} tokens).")
                else:
                    logger.info(f"[Budget] Skipped verifier hints ({hint_tokens} > {remaining} remaining).")

    stack = list(state.get("checkpoint_stack", []))

    try:
        _store_executor_memory(state, verdict, task_text)
    except Exception as mem_err:
        logger.warning("[Memory] Failed to store executor memory: %s", mem_err)

    if verdict.verdict == "PASS":
        try:
            _store_verifier_memory_on_repair_success(state, task_text)
        except Exception as mem_err:
            logger.warning("[Memory] Failed to store verifier memory: %s", mem_err)
        serialized_messages = messages_to_dict(state["messages"])
        if not stack or stack[-1]["messages"] != serialized_messages:
            stack.append({"messages": serialized_messages})
        return {
            "checkpoint_stack": stack,
            "pending_verifier_feedback": None,
        }

    if verdict.verdict == "REVISE":
        # Sprint 4: Cycle cap
        if budget:
            budget.record_revise()
            if budget.revise_exhausted():
                budget.log_budget_exit("revise", f"Revise cycle cap reached ({budget.revise_cycles}). Accepting current answer.")
                # Extract best answer so far and append as clean AIMessage
                # so verify_routing sees a valid exit message.
                answer = _extract_best_answer(state["messages"])
                answer_msg = AIMessage(content=answer)
                serialized_messages = messages_to_dict(state["messages"])
                stack.append({"messages": serialized_messages})
                return {
                    "messages": [answer_msg],
                    "checkpoint_stack": stack,
                    "pending_verifier_feedback": None,
                }
        warning_msg = SystemMessage(
            content=f"VERIFIER REVISION REQUIRED:\n{verdict.reasoning}{repair_hint_block}",
            additional_kwargs={"is_warning": True},
        )
        return {
            "messages": [warning_msg],
            "pending_verifier_feedback": {
                "verdict": verdict.verdict,
                "reasoning": verdict.reasoning,
            },
        }

    if not stack:
        warning_msg = SystemMessage(
            content=f"VERIFIER BACKTRACK (no checkpoints available):\n{verdict.reasoning}",
            additional_kwargs={"is_warning": True},
        )
        return {
            "messages": [warning_msg],
            "pending_verifier_feedback": {
                "verdict": verdict.verdict,
                "reasoning": verdict.reasoning,
            },
        }

    # Sprint 4: Backtrack cycle cap
    if budget:
        budget.record_backtrack()
        if budget.backtrack_exhausted():
            budget.log_budget_exit("backtrack", f"Backtrack cycle cap reached ({budget.backtrack_cycles}). Accepting current answer.")
            answer = _extract_best_answer(state["messages"])
            answer_msg = AIMessage(content=answer)
            return {
                "messages": [answer_msg],
                "checkpoint_stack": stack,
                "pending_verifier_feedback": None,
            }

    warning_msg = SystemMessage(
        content=f"{BACKTRACK_WARNING}\n\nReason for backtrack: {verdict.reasoning}{repair_hint_block}",
        additional_kwargs={"is_warning": True},
    )
    restored_messages = _last_verified_messages(stack) + [warning_msg]
    return {
        "messages": ReplaceMessages(restored_messages),
        "checkpoint_stack": stack,
        "pending_verifier_feedback": {
            "verdict": verdict.verdict,
            "reasoning": verdict.reasoning,
        },
    }


def verify_routing(state: AgentState) -> str:
    """Route after verifier based on the latest verified message state."""
    last_msg = state["messages"][-1]
    if _is_internal_warning(last_msg):
        return "reasoner"

    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
        return "format_normalizer"

    return "reasoner"
