"""Centralized prompt templates for the active document-grounded finance runtime."""

from __future__ import annotations

from typing import Any

_OFFICEQA_REASONING_SURFACES = (
    "simple extraction, inflation-adjusted multi-year comparisons, statistical analysis "
    "(regression, correlation, standard deviation), time-series forecasting, and complex "
    "financial metrics such as weighted averages and value-at-risk style calculations"
)


def _officeqa_active(benchmark_overrides: dict[str, Any] | None = None) -> bool:
    return str((benchmark_overrides or {}).get("benchmark_adapter") or "") == "officeqa"


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = (
    "You are a planner for document-grounded financial reasoning.\n"
    "Choose the execution mode and tool families for financial questions grounded in source documents such as Treasury Bulletins and similar official reports.\n\n"
    "The runtime should support question surfaces including "
    f"{_OFFICEQA_REASONING_SURFACES}.\n\n"
    "Prefer document-grounded analysis over generic advisory reasoning.\n"
    "Prefer retrieved document evidence before any synthesis.\n"
    "Use exact compute only when the retrieved evidence supports the operation.\n"
    "Return only JSON matching the schema."
)

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
EXECUTOR_SYSTEM = (
    "You are a document-grounded financial analyst answering financial questions from source documents.\n\n"
    "Ground every claim in the provided tool findings or curated facts.\n"
    "Preserve period scope, aggregation semantics, units, and source alignment.\n"
    "When the task is numeric, prefer extracted values and reproducible calculations over free-form estimation.\n"
    "If the evidence does not support the requested answer, say so directly instead of guessing."
)

RETRIEVAL_PLANNER_SYSTEM = (
    "You are a retrieval planner for financial questions grounded in source documents.\n"
    "Pick exactly ONE next action from the available tools, or 'answer' if evidence is sufficient.\n"
    "Prefer benchmark corpus retrieval and extracted table/page evidence over open-web search.\n"
    "Choose action='answer' only when the retrieved evidence supports the requested entity, period, metric, and aggregation.\n"
    "Return only JSON matching the schema."
)


# ---------------------------------------------------------------------------
# Guidance appended as additional system messages
# ---------------------------------------------------------------------------
OFFICEQA_FINANCIAL_DOCUMENT_GUIDANCE = (
    "Treat the question as document-grounded financial reasoning over official financial documents.\n"
    "Retrieve the right source before answering.\n"
    "Keep entity, period, unit, and aggregation alignment explicit.\n"
    "Support financial document question classes, including "
    f"{_OFFICEQA_REASONING_SURFACES}.\n"
    "If the task requires a numeric result, derive it from extracted evidence or the deterministic compute path.\n"
    "If deterministic compute is unavailable but evidence is still useful, keep the reasoning tightly grounded in retrieved values and citations.\n"
    "Wrong-source or wrong-period answers are worse than reporting insufficiency."
)

DOCUMENT_GROUNDED_GUIDANCE = (
    "Ground the answer in retrieved evidence.\n"
    "Use the retrieved documents or corpus content before outside knowledge.\n"
    "Quote the exact supporting phrase, number, or table row when it materially supports the answer.\n"
    "Cite the supporting source inline using the available citation or document label.\n"
    "Keep unsupported parts clearly marked as open questions."
)

RETRIEVAL_GUIDANCE = (
    "Answer from retrieved sources rather than model memory.\n"
    "Choose the next retrieval action from evidence quality, not from background knowledge.\n"
    "Prefer exact quotes or extracted numeric facts when they resolve the question.\n"
    "Cite the source used for the key conclusion."
)

GENERAL_GUIDANCE = (
    "Answer directly using the provided context and keep the structure aligned to the requested output."
)


def contract_guidance(answer_contract: dict[str, Any]) -> str:
    """Return contract-specific formatting guidance for the executor."""
    contract = dict(answer_contract or {})
    value_rules = dict(contract.get("value_rules") or {})
    final_answer_tag = str(value_rules.get("final_answer_tag", "") or contract.get("xml_root_tag", "") or "")
    reasoning_tag = str(value_rules.get("reasoning_tag", "") or "")

    if final_answer_tag == "FINAL_ANSWER":
        reasoning_tag = reasoning_tag or "REASONING"
        return (
            "Benchmark formatting is strict.\n"
            f"Put step-by-step reasoning inside <{reasoning_tag}>...</{reasoning_tag}>.\n"
            f"Put ONLY the final exact value or exact string inside <{final_answer_tag}>...</{final_answer_tag}>.\n"
            f"Inside <{final_answer_tag}> include no units, labels, prefixes, suffixes, or explanation.\n"
            "Preserve commas, decimals, and percent signs only when they are part of the exact answer."
        )

    if contract.get("requires_adapter") and contract.get("format") == "json":
        wrapper_key = str(contract.get("wrapper_key") or "answer")
        return (
            "The final answer must match the exact JSON contract.\n"
            f"Return the final result as a JSON object with the key \"{wrapper_key}\"."
        )

    return ""


def execution_guidance(
    task_family: str,
    execution_mode: str,
    *,
    benchmark_overrides: dict[str, Any] | None = None,
    task_text: str = "",
) -> str:
    """Return runtime guidance for the executor system prompt."""
    _ = task_family
    _ = task_text
    if _officeqa_active(benchmark_overrides):
        return OFFICEQA_FINANCIAL_DOCUMENT_GUIDANCE
    if execution_mode == "retrieval_augmented_analysis":
        return RETRIEVAL_GUIDANCE
    if execution_mode == "document_grounded_analysis":
        return DOCUMENT_GROUNDED_GUIDANCE
    return GENERAL_GUIDANCE


# ---------------------------------------------------------------------------
# Self-reflection
# ---------------------------------------------------------------------------
SELF_REFLECTION_SYSTEM = (
    "Evaluate this answer against the original task.\n"
    "1. Does it address ALL parts of the task?\n"
    "2. Are required fields present (amounts, structures, mechanics, values)?\n"
    "3. Does it show evidence of data or tool usage, not just reasoning?\n\n"
    "Reply JSON only:\n"
    '{"score": 0.0-1.0, "complete": true/false, '
    '"missing": ["item1", "item2"], '
    '"improve_prompt": "one sentence telling what to add"}'
)


# ---------------------------------------------------------------------------
# Revision
# ---------------------------------------------------------------------------
REVISION_TEMPLATE = (
    "Your previous answer was incomplete.\n"
    "Missing: {missing_items}\n"
    "Specifically: {improve_hint}\n"
    "Preserve all valid content. Add only the missing detail. "
    "Provide the complete, final answer now."
)


def build_revision_prompt(
    missing_dimensions: list[str],
    improve_hint: str = "",
    reviewer_reasoning: str = "",
    *,
    task_family: str = "",
    benchmark_overrides: dict[str, Any] | None = None,
    task_text: str = "",
) -> str:
    """Build a targeted revision prompt from reflection or reviewer output."""
    _ = task_family
    missing_str = ", ".join(missing_dimensions) if missing_dimensions else "see below"
    hint = improve_hint or reviewer_reasoning or "add the missing detail"
    prompt = REVISION_TEMPLATE.format(missing_items=missing_str, improve_hint=hint)
    if _officeqa_active(benchmark_overrides):
        prompt += (
            "\nRe-check source alignment, entity, period, aggregation, and unit normalization before finalizing."
            "\nIf the task is numeric, verify whether it requires simple extraction, inflation adjustment, statistical reasoning,"
            " forecasting logic, weighted averaging, risk-style analysis, or another grounded financial computation supported by the retrieved evidence."
        )
    elif task_text:
        prompt += "\nUse retrieved evidence directly when the task depends on source-backed support."
    return prompt


# ---------------------------------------------------------------------------
# Heuristic self-reflection
# ---------------------------------------------------------------------------
_ERROR_PHRASES = frozenset([
    "task failed", "error occurred", "unable to", "cannot access",
    "no data found", "tool unavailable", "token budget exhausted",
    "i cannot", "i don't have access",
])

_COMPLETION_MARKERS = frozenset([
    "recommendation:", "conclusion:", "summary:", "decision:", "action:",
    "next step:", "outcome:", "total:", "amount:", "score:", "rating:",
    "risk:", "status:", "approved", "rejected",
])


def heuristic_self_score(
    answer: str,
    tool_count: int = 0,
    task_family: str = "",
) -> float:
    """Fast zero-API quality score."""
    score = 0.50
    length = len((answer or "").strip())

    if length > 1200:
        score += 0.20
    elif length > 600:
        score += 0.15
    elif length > 250:
        score += 0.08
    elif length < 80:
        score -= 0.20

    if tool_count >= 4:
        score += 0.15
    elif tool_count >= 2:
        score += 0.10
    elif tool_count >= 1:
        score += 0.05
    elif tool_count == 0 and task_family not in ("finance_quant",):
        score -= 0.10

    answer_lower = (answer or "").lower()
    if any(marker in answer_lower for marker in _COMPLETION_MARKERS):
        score += 0.08
    if "{" in answer and "}" in answer:
        score += 0.05

    if any(phrase in answer_lower for phrase in _ERROR_PHRASES):
        score -= 0.25

    if answer and answer.rstrip()[-1] not in ".!?}]\n\"'":
        score -= 0.15

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Tools safe to skip on revision
# ---------------------------------------------------------------------------
REUSABLE_TOOL_FAMILIES = frozenset([
    "document_retrieval",
    "fetch_reference_file",
    "list_reference_files",
])
