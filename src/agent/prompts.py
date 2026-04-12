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

OFFICEQA_STRUCTURED_REPAIR_SYSTEM = (
    "You are a bounded retrieval-repair controller for document-grounded financial reasoning.\n"
    "Your job is to improve retrieval or table selection without answering the user question.\n"
    "Return only a structured repair decision.\n"
    "Allowed actions are: keep, rewrite_query, retune_table_query, change_strategy, widen_search_pool.\n"
    "Use publication_scope_action, restart_scope, and relax_provenance_priors when the current retrieval regime is exhausted.\n"
    "Do not invent facts, numbers, or final answers.\n"
    "Only change the retrieval path if the current evidence is semantically weak or misaligned."
)

FINANCIAL_SEMANTIC_PLAN_SYSTEM = (
    "You are a semantic planner for document-grounded financial reasoning.\n"
    "Extract the financial target, period semantics, publication-year neighborhood, granularity, and constraints needed for retrieval.\n"
    "Return only a structured semantic plan.\n"
    "Do not answer the user question.\n"
    "Prefer precise temporal semantics over lexical echoing."
)

FINANCIAL_SOURCE_RERANK_SYSTEM = (
    "You are the authoritative source selector for document-grounded financial reasoning.\n"
    "Choose the best candidate document from a shortlisted ranked set.\n"
    "Your selection will be used directly — there is no secondary override.\n"
    "Use temporal alignment, publication lag, entity fit, metric fit, table family, period type, "
    "unit basis, evidence-period fit, provenance priors, and table dimensions (row_count, column_count).\n"
    "Return only a structured rerank decision.\n"
    "Do not answer the user question."
)

FINANCIAL_TABLE_ADMISSIBILITY_SYSTEM = (
    "You are the authoritative table selector for document-grounded financial reasoning.\n"
    "Choose the best table candidate from a shortlisted set based on entity, metric, period, aggregation, "
    "table family, heading chain, row labels, headers, period type, unit basis, and table dimensions (row_count, column_count).\n"
    "Your selection will be used directly — there is no secondary override.\n"
    "If no candidate is suitable, use reject_current.\n"
    "Return only a structured table-selection decision.\n"
    "Do not answer the user question."
)

FINANCIAL_EVIDENCE_COMMIT_SYSTEM = (
    "You are an evidence-commit reviewer for document-grounded financial reasoning.\n"
    "Decide whether the currently visible evidence is strong enough to commit to deterministic compute.\n"
    "If the visible evidence is semantically weaker than an already-visible alternative, do not keep the current path.\n"
    "Return only a structured repair-style decision.\n"
    "Use same-document restart when a better table is visible in the current document.\n"
    "Use cross-document restart when a better source is already visible.\n"
    "Use widen_search_pool only when the visible evidence universe is still semantically exhausted.\n"
    "Do not answer the user question."
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
    answer_mode: str = "",
    compute_policy: str = "",
    partial_answer_allowed: bool = False,
    analysis_modes: list[str] | None = None,
    compute_status: str = "",
) -> str:
    """Return runtime guidance for the executor system prompt."""
    _ = task_family
    _ = task_text
    analysis_markers = set(analysis_modes or [])
    if _officeqa_active(benchmark_overrides):
        guidance_parts = [OFFICEQA_FINANCIAL_DOCUMENT_GUIDANCE]
        if answer_mode == "deterministic_compute":
            guidance_parts.append(
                "This question requires an exact numeric answer. Use deterministic compute when available and do not replace it with free-form reasoning."
            )
        elif answer_mode == "hybrid_grounded":
            guidance_parts.append(
                "This question needs a grounded narrative wrapper around a numeric core."
                " If a deterministic compute result is present, treat that numeric result as authoritative and explain it from retrieved evidence without recomputing."
            )
        elif answer_mode == "grounded_synthesis":
            guidance_parts.append(
                "This question should be answered through grounded synthesis over retrieved evidence."
                " Do not invent an exact calculation that the evidence or deterministic compute path does not support."
            )
        if compute_policy == "preferred":
            guidance_parts.append(
                "If exact deterministic compute is unavailable but retrieved evidence still supports part of the task, provide a bounded grounded answer."
                " Clearly separate supported findings from unsupported remainder."
            )
        elif compute_policy == "not_applicable":
            guidance_parts.append(
                "Focus on a fully grounded answer from retrieved text and tables rather than forcing a numeric compute path."
            )
        if partial_answer_allowed:
            guidance_parts.append(
                "When only part of the task is supported, explicitly label the supported portion and the unsupported remainder instead of collapsing into a generic insufficiency response."
            )
        if compute_status == "ok":
            guidance_parts.append(
                "A deterministic compute result is already available in the context block. Reuse it exactly."
            )
        if analysis_markers.intersection({"statistical_analysis", "time_series_forecasting", "risk_metric"}):
            guidance_parts.append(
                "For advanced financial reasoning tasks, keep assumptions explicit, preserve ordered series logic, and ground every inference in retrieved evidence."
            )
        return "\n".join(guidance_parts)
    if execution_mode == "retrieval_augmented_analysis":
        return RETRIEVAL_GUIDANCE
    if execution_mode == "document_grounded_analysis":
        return DOCUMENT_GROUNDED_GUIDANCE
    return GENERAL_GUIDANCE


def build_officeqa_structured_repair_prompt(
    *,
    task_text: str,
    retrieval_strategy: str,
    evidence_gap: str,
    source_constraint_policy: str = "",
    target_years: list[str] | None = None,
    publication_year_window: list[str] | None = None,
    preferred_publication_years: list[str] | None = None,
    current_query: str = "",
    current_table_query: str = "",
    candidate_sources: list[dict[str, Any]] | None = None,
    execution_snapshot: dict[str, Any] | None = None,
    review_feedback: dict[str, Any] | None = None,
) -> str:
    candidates = []
    for item in list(candidate_sources or [])[:5]:
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "document_id": str(item.get("document_id", "") or ""),
                "title": str(item.get("title", "") or ""),
                "score": item.get("score", 0.0),
                "snippet": str(item.get("snippet", "") or "")[:220],
                "metadata": dict(item.get("metadata", {}) or {}),
                "best_evidence_unit": dict(item.get("best_evidence_unit", {}) or {}),
            }
        )
    feedback = dict(review_feedback or {})
    snapshot = dict(execution_snapshot or {})
    return (
        f"TASK={task_text}\n"
        f"CURRENT_STRATEGY={retrieval_strategy}\n"
        f"EVIDENCE_GAP={evidence_gap}\n"
        f"SOURCE_CONSTRAINT_POLICY={source_constraint_policy}\n"
        f"TARGET_YEARS={list(target_years or [])}\n"
        f"PUBLICATION_YEAR_WINDOW={list(publication_year_window or [])}\n"
        f"PREFERRED_PUBLICATION_YEARS={list(preferred_publication_years or [])}\n"
        f"CURRENT_QUERY={current_query}\n"
        f"CURRENT_TABLE_QUERY={current_table_query}\n"
        f"REVIEW_REPAIR_TARGET={feedback.get('repair_target', '')}\n"
        f"REVIEW_ORCHESTRATION={feedback.get('orchestration_strategy', '')}\n"
        f"REMEDIATION_CODES={list(feedback.get('remediation_codes', []) or [])}\n"
        f"ATTEMPTED_QUERIES={list(snapshot.get('attempted_queries', []) or [])}\n"
        f"CANDIDATE_POOLS_SEEN={list(snapshot.get('candidate_pools_seen', []) or [])}\n"
        f"REJECTED_EVIDENCE_FAMILIES={list(snapshot.get('rejected_evidence_families', []) or [])}\n"
        f"COMPUTE_ADMISSIBILITY_FAILURES={list(snapshot.get('compute_admissibility_failures', []) or [])}\n"
        f"REPAIR_FAILURES={list(snapshot.get('repair_failures', []) or [])}\n"
        f"REPAIR_HISTORY={list(snapshot.get('repair_history', []) or [])}\n"
        f"STRUCTURED_EVIDENCE_SUMMARY={dict(snapshot.get('structured_evidence_summary', {}) or {})}\n"
        f"CANDIDATE_SOURCES={candidates}"
    )


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
