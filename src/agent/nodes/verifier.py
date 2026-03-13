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
import re
from difflib import SequenceMatcher

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
from agent.memory.schema import (
    ExecutorMemory,
    VerifierMemory,
    _infer_failure_family,
    _infer_tool_family,
    _normalize_task_type,
    _normalize_memory_text,
    _task_signature,
    _task_type_to_family,
)
from agent.nodes.reasoner import _allowed_tool_names_for_task
from agent.prompts import VERIFIER_FINAL_ANSWER_ADDENDUM, VERIFIER_JSON_FALLBACK_PROMPT, VERIFIER_PROMPT, VerdictDecision
from agent.state import AgentState, ReplaceMessages
from agent.pruning import truncate_memory_fields
from context_manager import count_tokens

logger = logging.getLogger(__name__)

BACKTRACK_WARNING = (
    "BACKTRACK WARNING: Your previous approach was evaluated as fundamentally flawed "
    "or hallucinated. The state has been reverted to the last verified checkpoint. "
    "Do NOT repeat the same mistake. Try a different tool or strategy."
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TEXTUAL_TOOL_ATTEMPT_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:',
    re.DOTALL,
)

_FINAL_ANSWER_CHECKS = {
    "quantitative": (
        "- For quantitative tasks: verify the correct values were extracted, the formula was applied correctly, "
        "and the answer matches the requested format."
    ),
    "legal": (
        "- For legal tasks: verify the answer covers structure options, liability allocation, tax implications, "
        "regulatory/compliance issues, diligence, key open questions/assumptions, and execution next steps when relevant. "
        "Do not PASS a generic memo that omits decision-driving unknowns or concrete risk-allocation mechanisms."
    ),
    "options": (
        "- For options tasks: verify the answer includes a primary strategy with "
        "tool-backed Greeks analysis and P&L/breakeven data. "
        "Verify that at least one alternative strategy is discussed with concrete "
        "quantitative tradeoffs (e.g., different max-loss, Greeks profile), "
        "not merely named. "
        "Reject final answers that are visibly truncated or stop mid-bullet. "
        "Do NOT reject an answer solely for lacking a second tool-backed analysis "
        "if the alternative includes concrete numerical comparison."
    ),
    "document": (
        "- For document tasks: verify the answer is grounded in file content and includes the key extracted figures "
        "or findings needed to answer the question."
    ),
    "retrieval": (
        "- For retrieval tasks: verify the answer uses retrieved evidence rather than generic filler and reflects "
        "the requested external facts or sources."
    ),
}

_LEGAL_COMPLETENESS_RULES: dict[str, tuple[str, ...]] = {
    "structure options": (
        "asset purchase",
        "stock purchase",
        "merger",
        "spv",
        "carve-out",
        "hybrid",
        "triangular",
    ),
    "tax consequences": (
        "tax",
        "capital gain",
        "basis",
        "deferral",
        "step-up",
        "tax-free",
        "section 355",
    ),
    "liability protection": (
        "indemn",
        "escrow",
        "holdback",
        "rep",
        "warrant",
        "insurance",
        "disclosure schedule",
        "cap",
    ),
    "regulatory/diligence risks": (
        "regulator",
        "compliance",
        "filing",
        "approval",
        "diligence",
        "audit",
        "hsr",
        "gdpr",
        "eu",
        "us",
    ),
    "key open questions / assumptions": (
        "need to determine",
        "need to assess",
        "need to confirm",
        "assumption",
        "willingness",
        "risk tolerance",
        "severity",
        "feasible",
        "timeline requirement",
        "unknown",
        "clarify",
    ),
    "recommended next steps": (
        "recommend",
        "next step",
        "engage",
        "draft",
        "file",
        "timeline",
        "week",
        "immediately",
    ),
}

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


def _strip_think_markup(text: str) -> str:
    clean = _THINK_BLOCK_RE.sub("", text)
    clean = clean.replace("<think>", "").replace("</think>", "")
    return clean.strip()


def _ai_finish_reason(msg: BaseMessage | None) -> str:
    if not isinstance(msg, AIMessage):
        return ""
    return str(getattr(msg, "response_metadata", {}).get("finish_reason", "") or "").lower()


def _looks_truncated_answer(msg: BaseMessage | None) -> bool:
    if not isinstance(msg, AIMessage) or not msg.content:
        return False
    if _ai_finish_reason(msg) == "length":
        return True

    text = _strip_think_markup(str(msg.content)).rstrip()
    if not text:
        return False
    if text.endswith(("(", "[", "{", ":", ",", "/", "-", "**")):
        return True

    last_line = text.splitlines()[-1].strip()
    if last_line.startswith(("-", "*")) and len(last_line) <= 6:
        return True
    return False


def _textual_tool_attempt_name(msg: BaseMessage | None) -> str | None:
    if not isinstance(msg, AIMessage) or msg.tool_calls or not msg.content:
        return None
    clean = _strip_think_markup(str(msg.content))
    match = _TEXTUAL_TOOL_ATTEMPT_RE.search(clean)
    if match:
        return match.group(1).strip()
    return None


def _requires_transactional_legal_depth(task_text: str) -> bool:
    normalized = (task_text or "").lower()
    return any(
        token in normalized
        for token in (
            "acquisition",
            "merger",
            "deal",
            "stock consideration",
            "liability",
            "compliance",
            "transaction",
            "target company",
        )
    )


def _legal_completeness_gaps(answer_text: str, task_text: str) -> list[str]:
    if not answer_text:
        return ["final answer text"]

    normalized = _normalize_memory_text(answer_text, max_len=5000)
    gaps: list[str] = []
    for label, keywords in _LEGAL_COMPLETENESS_RULES.items():
        if not any(keyword in normalized for keyword in keywords):
            gaps.append(label)

    if _requires_transactional_legal_depth(task_text):
        structure_hits = sum(
            1
            for keyword in ("asset purchase", "stock purchase", "merger", "spv", "carve-out", "hybrid", "triangular")
            if keyword in normalized
        )
        if structure_hits < 2 and "multiple structure alternatives" not in gaps:
            gaps.append("multiple structure alternatives")

        if "eu" in task_text.lower() and "eu" not in normalized:
            gaps.append("EU-specific regulatory touchpoint")
        if "us" in task_text.lower() and "us" not in normalized and "hsr" not in normalized:
            gaps.append("US-specific regulatory touchpoint")

    deduped: list[str] = []
    for gap in gaps:
        if gap not in deduped:
            deduped.append(gap)
    return deduped


def _baseline_checkpoint_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return the stable prefix up to the latest human turn.

    This preserves prior multi-turn context while excluding the current executor
    attempt that we may need to roll back.
    """
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return messages[: idx + 1]
    return messages[:1]


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


def _extract_best_answer_with_checkpoint_fallback(
    messages: list[BaseMessage],
    checkpoint_stack: list[dict],
) -> str:
    answer = _extract_best_answer(messages)
    if answer != "I was unable to complete this task within the allowed verification budget.":
        return answer

    checkpoint_answer = _extract_best_answer(_last_verified_messages(checkpoint_stack))
    if checkpoint_answer != "I was unable to complete this task within the allowed verification budget.":
        return checkpoint_answer
    return answer


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

    task_type = _normalize_task_type(state.get("task_type", "general"))
    rec = ExecutorMemory(
        task_signature=_task_signature(task_text),
        partial_context_summary=task_text[:120],
        semantic_text=_normalize_memory_text(
            f"task: {task_text}\n"
            f"tool: {tool_used}\n"
            f"arguments: {arguments_pattern}\n"
            f"quality: {outcome_quality}"
        ),
        task_family=_task_type_to_family(task_type, task_text),
        tool_used=tool_used,
        tool_family=_infer_tool_family(tool_used),
        arguments_pattern=arguments_pattern,
        outcome_quality=outcome_quality,
        success=success,
        tags=[tool_used, outcome_quality],
        metadata={
            "task_type": task_type,
            "verdict": verdict.verdict,
            "tool_message_name": last_msg.name,
            "tool_output_preview": str(last_msg.content)[:120],
        },
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

    task_type = _normalize_task_type(state.get("task_type", "general"))
    rec = VerifierMemory(
        task_signature=_task_signature(task_text),
        failure_pattern=str(pending_feedback.get("reasoning", ""))[:240],
        semantic_text=_normalize_memory_text(
            f"task: {task_text}\n"
            f"failure: {pending_feedback.get('reasoning', '')}\n"
            f"repair: {_repair_action_summary(state['messages'])}\n"
            f"verdict: {pending_feedback.get('verdict', 'REVISE')}"
        ),
        task_family=_task_type_to_family(task_type, task_text),
        failure_family=_infer_failure_family(str(pending_feedback.get("reasoning", ""))),
        verdict=pending_feedback.get("verdict", "REVISE"),
        repair_action=_repair_action_summary(state["messages"]),
        repair_worked=True,
        tags=[pending_feedback.get("verdict", "REVISE")],
        metadata={
            "task_type": task_type,
            "pending_reasoning_preview": str(pending_feedback.get("reasoning", ""))[:120],
        },
    )
    truncate_memory_fields(rec)
    memory_store.store_verifier(rec)


def _recent_nonwarning_ai_texts(messages: list[BaseMessage], limit: int = 2) -> list[str]:
    texts: list[str] = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls and not _is_internal_warning(msg):
            texts.append(str(msg.content))
            if len(texts) >= limit:
                break
    return list(reversed(texts))


def _is_stagnant_revise_loop(messages: list[BaseMessage]) -> bool:
    """Detect repeated text-only answer attempts with little material change."""
    texts = _recent_nonwarning_ai_texts(messages, limit=2)
    if len(texts) < 2:
        return False
    older = _normalize_memory_text(texts[0], max_len=500)
    newer = _normalize_memory_text(texts[1], max_len=500)
    if not older or not newer:
        return False
    if older == newer:
        return True
    return SequenceMatcher(a=older, b=newer).ratio() >= 0.92


def _verifier_prompt_for(task_type: str, is_final_answer: bool) -> str:
    """Build verifier prompt, preserving strictness in both native and fallback modes."""
    base_prompt = VERIFIER_PROMPT
    if not is_final_answer:
        return base_prompt
    task_specific = _FINAL_ANSWER_CHECKS.get(task_type, "")
    if task_specific:
        return base_prompt + VERIFIER_FINAL_ANSWER_ADDENDUM + "\n" + task_specific
    return base_prompt + VERIFIER_FINAL_ANSWER_ADDENDUM


def _verifier_fallback_prompt_for(task_type: str, is_final_answer: bool) -> str:
    base_prompt = VERIFIER_JSON_FALLBACK_PROMPT
    if not is_final_answer:
        return base_prompt
    task_specific = _FINAL_ANSWER_CHECKS.get(task_type, "")
    if task_specific:
        return base_prompt + VERIFIER_FINAL_ANSWER_ADDENDUM + "\n" + task_specific
    return base_prompt + VERIFIER_FINAL_ANSWER_ADDENDUM


def verifier(state: AgentState) -> dict:
    """Verify the latest executor step and update checkpoint state."""
    selected_layers = state.get("selected_layers", [])
    if "verifier_check" not in selected_layers:
        return {}

    history = [m for m in state["messages"] if not _is_internal_warning(m)]
    task_type = _normalize_task_type(state.get("task_type", "general"))
    task_text = _latest_human_text(state["messages"])

    # Sprint 5: Detect final answer vs intermediate step
    # Final answer = last real AI message has no tool_calls
    is_final_answer = False
    for msg in reversed(history):
        if isinstance(msg, AIMessage) and msg.content and not _is_internal_warning(msg):
            is_final_answer = not bool(msg.tool_calls)
            break

    last_msg = _last_non_warning_message(history)
    textual_tool_attempt = _textual_tool_attempt_name(last_msg)

    tracker: CostTracker | None = state.get("cost_tracker")
    model_name = get_model_name("verifier")
    used_llm = False
    invocation_messages: list[BaseMessage] = []

    verdict = None
    latency = 0.0
    success = False

    if textual_tool_attempt:
        allowed_tools = _allowed_tool_names_for_task(task_type, task_text)
        disallowed = allowed_tools is not None and textual_tool_attempt not in allowed_tools
        verdict = VerdictDecision(
            verdict="BACKTRACK" if disallowed else "REVISE",
            reasoning=(
                f"The executor emitted a textual tool attempt for '{textual_tool_attempt}' instead of a valid executed "
                f"tool call or a proper final answer. "
                + (
                    f"This tool is not allowed for task_type '{task_type}'. Revert and choose a permitted strategy."
                    if disallowed
                    else "Re-emit either a valid tool call or a complete final answer, but do not include raw tool JSON in text."
                )
            ),
        )
        latency = 0.0
        success = False
        logger.warning(
            "[Verifier] Detected textual tool attempt '%s' for task_type=%s; bypassing verifier LLM.",
            textual_tool_attempt,
            task_type,
        )
    elif is_final_answer and _looks_truncated_answer(last_msg):
        verdict = VerdictDecision(
            verdict="REVISE",
            reasoning=(
                "The final answer was truncated before completion. Re-synthesize a compact final answer. "
                "Do not include internal reasoning, and do not end with a partial bullet or unfinished sentence."
            ),
        )
        logger.warning(
            "[Verifier] Detected truncated final answer for task_type=%s; bypassing verifier LLM.",
            task_type,
        )
    elif is_final_answer and task_type == "legal":
        legal_answer = _strip_think_markup(str(getattr(last_msg, "content", "") or ""))
        gaps = _legal_completeness_gaps(legal_answer, task_text)
        if gaps:
            verdict = VerdictDecision(
                verdict="REVISE",
                reasoning=(
                    "The legal final answer is incomplete. Add the missing dimensions: "
                    + ", ".join(gaps[:6])
                    + ". Use the required legal sections and make the answer specific."
                ),
            )
            logger.warning(
                "[Verifier] Detected incomplete legal final answer; missing=%s. Bypassing verifier LLM.",
                ", ".join(gaps[:6]),
            )

    if verdict is not None:
        pass
    else:
        llm = ChatOpenAI(
            model=model_name,
            **get_client_kwargs("verifier"),
            temperature=0.0,
        )

        # Use stricter prompt for final answers only
        base_prompt = _verifier_prompt_for(task_type, is_final_answer)

        prompt_messages = [SystemMessage(content=base_prompt)] + history

        invocation_messages = prompt_messages
        t0 = time.monotonic()
        try:
            if _structured_output_mode("verifier") == "native":
                verdict = llm.with_structured_output(VerdictDecision).invoke(prompt_messages)
            else:
                invocation_messages = [SystemMessage(content=_verifier_fallback_prompt_for(task_type, is_final_answer))] + history
                raw_response = llm.invoke(invocation_messages)
                verdict = VerdictDecision.model_validate_json(
                    _extract_json_payload(str(raw_response.content or ""))
                )
            success = True
            used_llm = True
        except Exception as exc:
            logger.warning("Verifier LLM failed: %s. Defaulting to PASS.", exc)
            verdict = VerdictDecision(
                verdict="PASS",
                reasoning=f"Fallback due to verifier error: {exc}",
            )
            success = False
            used_llm = True
        latency = (time.monotonic() - t0) * 1000

    if tracker and used_llm:
        tracker.record(
            operator="verifier_check",
            model_name=model_name,
            tokens_in=count_tokens(invocation_messages),
            tokens_out=count_tokens([AIMessage(content=verdict.model_dump_json())]),
            latency_ms=latency,
            success=success,
        )

    logger.info(
        "[Verifier] Verdict: %s | Reason: %s",
        verdict.verdict,
        verdict.reasoning[:120],
    )

    # Sprint 3+4: Retrieve repair hints from verifier memory (budget-capped)
    repair_hint_block = ""
    memory_store = state.get("memory_store")
    budget = state.get("budget_tracker")
    current_failure_family = _infer_failure_family(verdict.reasoning)
    if (
        memory_store
        and verdict.verdict in ("REVISE", "BACKTRACK")
        and current_failure_family in {"schema", "format", "hallucination", "repetition", "tool_use"}
    ):
        if task_text:
            repair_hints = memory_store.retrieve_verifier_hints(
                task_text,
                failure_family=current_failure_family,
            )
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
    seeded_initial_checkpoint = False

    # Fix 2a: Create pre-step checkpoint if stack is empty
    # This ensures the very first tool call has a valid rollback point
    if not stack:
        initial_msgs = _baseline_checkpoint_messages(state["messages"])
        if initial_msgs:
            stack = [{"messages": messages_to_dict(initial_msgs)}]
            seeded_initial_checkpoint = True

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
        if seeded_initial_checkpoint and len(stack) == 1:
            stack = [{"messages": serialized_messages}]
        elif not stack or stack[-1]["messages"] != serialized_messages:
            stack.append({"messages": serialized_messages})
        return {
            "checkpoint_stack": stack,
            "pending_verifier_feedback": None,
        }

    if verdict.verdict == "REVISE":
        # Sprint 4: Cycle cap
        if budget:
            budget.record_revise()
            if budget.revise_cycles >= 3 and _is_stagnant_revise_loop(state["messages"]):
                logger.warning(
                    "[Verifier] Escalating stagnant revise loop to BACKTRACK after %s revise cycles.",
                    budget.revise_cycles,
                )
                verdict = VerdictDecision(
                    verdict="BACKTRACK",
                    reasoning=(
                        "Repeated revise cycles did not materially improve the answer. "
                        "Revert and use a different strategy. Do not restate the same draft."
                    ),
                )
            if budget.revise_exhausted():
                budget.log_budget_exit("revise", f"Revise cycle cap reached ({budget.revise_cycles}). Accepting current answer.")
                # Extract best answer so far and append as clean AIMessage
                # so verify_routing sees a valid exit message.
                answer = _extract_best_answer_with_checkpoint_fallback(state["messages"], stack)
                answer_msg = AIMessage(content=answer)
                return {
                    "messages": [answer_msg],
                    "checkpoint_stack": stack,
                    "pending_verifier_feedback": None,
                }
        if verdict.verdict == "REVISE":
            _LEGAL_CONSTRAINED_TEMPLATE = (
                "\n\nCONSTRAINED ANSWER MODE: Your previous answer was incomplete. "
                "You MUST now produce your final answer using ONLY these sections, "
                "providing HIGHLY DETAILED and exhaustive analytical depth for each:\n"
                "1. STRUCTURE OPTIONS\n2. TAX CONSEQUENCES\n3. LIABILITY PROTECTION\n"
                "4. REGULATORY/DILIGENCE RISKS\n5. KEY OPEN QUESTIONS / ASSUMPTIONS\n"
                "6. RECOMMENDED NEXT STEPS\n"
                "No preamble. No internal reasoning. Start directly with '1. STRUCTURE OPTIONS'."
            )
            _OPTIONS_CONSTRAINED_TEMPLATE = (
                "\n\nCONSTRAINED OPTIONS MODE: Your previous answer was incomplete or malformed. "
                "Output ONLY these blocks, in order:\n"
                "1. PRIMARY STRATEGY\n"
                "2. ALTERNATIVE STRATEGY\n"
                "3. KEY GREEKS / P&L / BREAKEVENS\n"
                "4. RISK MANAGEMENT / HEDGE / SIZING\n"
                "Use compact bullet points (2-4 bullets per block). No preamble. No internal reasoning. "
                "If you use a tool, emit ONLY the tool JSON. Do not embed tool JSON inside prose."
            )

            constrained_suffix = ""
            if task_type == "legal" and budget and budget.revise_cycles >= 1:
                constrained_suffix = _LEGAL_CONSTRAINED_TEMPLATE
                logger.info("[Verifier] Legal task: injecting constrained answer template after REVISE #%d", budget.revise_cycles)
            elif task_type == "options" and budget and budget.revise_cycles >= 1:
                constrained_suffix = _OPTIONS_CONSTRAINED_TEMPLATE
                logger.info("[Verifier] Options task: injecting constrained answer template after REVISE #%d", budget.revise_cycles)

            warning_msg = SystemMessage(
                content=f"VERIFIER REVISION REQUIRED:\n{verdict.reasoning}{repair_hint_block}{constrained_suffix}",
                additional_kwargs={"is_warning": True},
            )
            return {
                "messages": [warning_msg],
                "pending_verifier_feedback": {
                    "verdict": verdict.verdict,
                    "reasoning": verdict.reasoning,
                },
            }

    # Fix 2b: Real BACKTRACK — revert to last HumanMessage if no checkpoint
    if not stack:
        # Emergency fallback: should not happen now that we create pre-step checkpoints
        # but kept as a safety net
        reverted = []
        for i, msg in enumerate(state["messages"]):
            if isinstance(msg, HumanMessage):
                reverted = state["messages"][:i + 1]
        if not reverted:
            reverted = state["messages"][:1]
        warning_msg = SystemMessage(
            content=(
                f"BACKTRACK (clean slate): {verdict.reasoning}\n"
                "The previous tool call was irrelevant. Do NOT repeat it. "
                "Answer from your domain knowledge or choose a different tool."
                f"{repair_hint_block}"
            ),
            additional_kwargs={"is_warning": True},
        )
        return {
            "messages": ReplaceMessages(reverted + [warning_msg]),
            "checkpoint_stack": [],
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
            answer = _extract_best_answer_with_checkpoint_fallback(state["messages"], stack)
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
