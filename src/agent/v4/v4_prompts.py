"""Centralized prompt templates for V4 roles.

Every LLM-facing string lives here so that nodes.py stays logic-only.
Prompt design follows patterns from top-scoring finance agents:
  - Positive instructions (no "do not" blocks)
  - Domain-specific guidance per task family
  - Structured self-reflection rubric with heuristic pre-check
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = (
    "You are a finance task router.\n"
    "Given the task, select the best execution mode and tool families.\n\n"
    "Finance task families:\n"
    "- finance_quant: formulas, tables, exact numeric answers (NPV, IRR, amortization)\n"
    "- finance_options: Black-Scholes, Greeks, option strategies, IV analysis\n"
    "- legal_transactional: M&A structures, deal terms, regulatory compliance, tax treatment\n"
    "- document_qa: questions grounded in uploaded files or reference documents\n"
    "- external_retrieval: questions requiring current market or source-backed data\n"
    "- general: anything that does not clearly fit above\n\n"
    "Be conservative with exact fast paths. Use them only when the prompt already "
    "contains the formula, the relevant table, and an exact output contract.\n"
    "Prefer broader capability access for legal/advisory tasks instead of calculator-only reasoning.\n\n"
    "Return only JSON matching the schema."
)

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
EXECUTOR_SYSTEM = (
    "You are a senior finance analyst producing a decision-ready answer.\n\n"
    "Ground every claim in the provided tool findings or curated facts.\n"
    "Be specific about numbers, dates, entities, and mechanics.\n"
    "State material assumptions explicitly.\n"
    "If the data does not support a conclusion, say so rather than hedging."
)

# ---------------------------------------------------------------------------
# Family-specific guidance — appended as a second system message
# ---------------------------------------------------------------------------
LEGAL_GUIDANCE = (
    "Start with a compact opening snapshot before any deep dive.\n"
    "In the first section, list multiple viable structure alternatives with one-line tradeoffs, "
    "then name the recommended path.\n"
    "Make the opening summary self-sufficient for skimmers: it should already show the options, "
    "recommendation, economic or tax tradeoffs, liability mechanics, execution constraints, "
    "and a concrete next-step plan.\n"
    "For each structure cover:\n"
    "  (1) economic and tax treatment — who benefits and what breaks it,\n"
    "  (2) liability allocation — indemnities, escrow or holdback, caps, survival periods,\n"
    "  (3) execution constraints — required approvals, consultations, third-party consents, and timing,\n"
    "  (4) rapid next-step plan with owners and sequencing."
)

OPTIONS_GUIDANCE = (
    "Start with a direct buy/sell recommendation.\n"
    "Present the primary strategy with explicit Greek values "
    "(Delta, Gamma, Theta, Vega) as numbers.\n"
    "Show breakeven levels and max profit/loss.\n"
    "Compare at least one alternative strategy with defined-risk characteristics.\n"
    "Include concrete risk management rules "
    "(exit triggers, adjustment thresholds, position sizing)."
)

QUANT_GUIDANCE = (
    "Apply the specified formula to the correct data row.\n"
    "Show intermediate calculation steps.\n"
    "Return the result in the exact output format requested."
)

DOCUMENT_GROUNDED_GUIDANCE = (
    "Ground the answer in retrieved evidence. "
    "Keep unsupported parts clearly marked as open questions."
)

GENERAL_GUIDANCE = (
    "Answer directly using the provided context "
    "and keep the structure aligned to the requested output."
)


def execution_guidance(task_family: str, execution_mode: str) -> str:
    """Return family-specific guidance for the executor system prompt."""
    if task_family == "legal_transactional":
        return LEGAL_GUIDANCE
    if task_family == "finance_options":
        return OPTIONS_GUIDANCE
    if task_family == "finance_quant":
        return QUANT_GUIDANCE
    if execution_mode == "document_grounded_analysis":
        return DOCUMENT_GROUNDED_GUIDANCE
    return GENERAL_GUIDANCE


# ---------------------------------------------------------------------------
# Self-reflection — LLM rubric (for complex_qualitative tasks)
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
# Revision — targeted improvement prompt
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
) -> str:
    """Build a targeted revision prompt from reflection/reviewer output."""
    missing_str = ", ".join(missing_dimensions) if missing_dimensions else "see below"
    hint = improve_hint or reviewer_reasoning or "add the missing detail"
    prompt = REVISION_TEMPLATE.format(missing_items=missing_str, improve_hint=hint)
    if task_family == "legal_transactional":
        prompt += (
            "\nKeep the opening section compact. Start with a snapshot naming multiple viable structures, "
            "one-line tradeoffs, and the recommended path before any deep dive."
        )
    return prompt


# ---------------------------------------------------------------------------
# Heuristic self-reflection (pre-check before LLM call)
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
    """Fast zero-API quality score — skip LLM if clearly good (>= 0.85).

    Mirrors Purple Agent's computeAgentQuality() pattern:
    penalises empty data, no tool calls, very short answers, error phrases;
    rewards structured output, tool usage, completeness markers.
    """
    score = 0.50
    length = len((answer or "").strip())

    # Length signals
    if length > 1200:
        score += 0.20
    elif length > 600:
        score += 0.15
    elif length > 250:
        score += 0.08
    elif length < 80:
        score -= 0.20

    # Tool usage signals
    if tool_count >= 4:
        score += 0.15
    elif tool_count >= 2:
        score += 0.10
    elif tool_count >= 1:
        score += 0.05
    elif tool_count == 0 and task_family not in ("finance_quant",):
        score -= 0.10

    # Structure signals
    answer_lower = (answer or "").lower()
    if any(marker in answer_lower for marker in _COMPLETION_MARKERS):
        score += 0.08
    if "{" in answer and "}" in answer:
        score += 0.05

    # Error penalty
    if any(phrase in answer_lower for phrase in _ERROR_PHRASES):
        score -= 0.25

    # Truncation penalty
    if answer and answer.rstrip()[-1] not in ".!?}]\n\"'":
        score -= 0.15

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Tools whose results are safe to skip on revision (non-live-data)
# ---------------------------------------------------------------------------
REUSABLE_TOOL_FAMILIES = frozenset([
    "legal_playbook_retrieval",
    "transaction_structure_checklist",
    "regulatory_execution_checklist",
    "tax_structure_checklist",
    "document_retrieval",
    "fetch_reference_file",
    "list_reference_files",
])
