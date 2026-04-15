from __future__ import annotations

from collections import deque
from typing import Any

from engine.agent.contracts import RetrievalIntent, StrategyJournalEntry, StrategyJournalRecommendation

_MAX_JOURNAL_ENTRIES = 256
_MAX_SUPPORTING_ENTRIES = 8

_STRATEGY_JOURNAL: deque[StrategyJournalEntry] = deque(maxlen=_MAX_JOURNAL_ENTRIES)
_STRATEGY_JOURNAL_SEQUENCE = 0


def clear_strategy_journal() -> None:
    global _STRATEGY_JOURNAL_SEQUENCE
    _STRATEGY_JOURNAL.clear()
    _STRATEGY_JOURNAL_SEQUENCE = 0


def _semantic_signature(retrieval_intent: RetrievalIntent) -> str:
    analysis_modes = ",".join(sorted(str(item or "").strip().lower() for item in list(retrieval_intent.analysis_modes or []) if str(item).strip()))
    constraint_shape = "constraints" if (retrieval_intent.include_constraints or retrieval_intent.exclude_constraints) else "plain"
    retrospective = "retro" if retrieval_intent.retrospective_evidence_allowed or retrieval_intent.retrospective_evidence_required else "direct"
    return "|".join(
        [
            str(retrieval_intent.metric or "").strip().lower(),
            str(retrieval_intent.period_type or "").strip().lower(),
            str(retrieval_intent.granularity_requirement or "").strip().lower(),
            str(retrieval_intent.answer_mode or "").strip().lower(),
            constraint_shape,
            retrospective,
            analysis_modes,
        ]
    )


def strategy_journal_key(
    *,
    task_family: str,
    retrieval_intent: RetrievalIntent,
    table_family: str = "",
) -> tuple[str, str]:
    aggregation_shape = str(retrieval_intent.aggregation_shape or "").strip().lower()
    semantic_signature = _semantic_signature(retrieval_intent)
    normalized_family = str(table_family or "").strip().lower()
    broader_key = "|".join(
        [
            str(task_family or "").strip().lower(),
            aggregation_shape,
            semantic_signature,
        ]
    )
    if normalized_family:
        return f"{broader_key}|{normalized_family}", broader_key
    return broader_key, broader_key


def _entry_weight(entry: StrategyJournalEntry) -> float:
    weight = 0.0
    if entry.success:
        weight += 2.0
    else:
        weight -= 1.25
    if entry.evidence_ready:
        weight += 0.45
    if entry.compute_status == "ok":
        weight += 0.4
    elif entry.compute_status in {"unsupported", "insufficient"}:
        weight -= 0.3
    if entry.validator_verdict == "pass":
        weight += 0.55
    elif entry.validator_verdict in {"revise", "fail"}:
        weight -= 0.5
    if entry.stop_reason in {"progress_stalled", "officeqa_no_repair_path"}:
        weight -= 0.35
    return weight


def recommend_strategy_order(
    *,
    task_family: str,
    retrieval_intent: RetrievalIntent,
    admissible_strategies: list[str],
) -> StrategyJournalRecommendation:
    if not admissible_strategies:
        return StrategyJournalRecommendation()
    journal_key, broader_key = strategy_journal_key(task_family=task_family, retrieval_intent=retrieval_intent)
    exact_entries = [entry for entry in list(_STRATEGY_JOURNAL) if entry.journal_key == journal_key]
    broad_entries = [entry for entry in list(_STRATEGY_JOURNAL) if entry.broader_key == broader_key]
    scores = {str(strategy): 0.0 for strategy in admissible_strategies}
    for entry in exact_entries:
        strategy = str(entry.applied_strategy or entry.requested_strategy or "").strip()
        if strategy in scores:
            scores[strategy] += _entry_weight(entry)
    if not exact_entries:
        for entry in broad_entries:
            strategy = str(entry.applied_strategy or entry.requested_strategy or "").strip()
            if strategy in scores:
                scores[strategy] += _entry_weight(entry) * 0.45
    ordered = sorted(
        admissible_strategies,
        key=lambda strategy: (-scores.get(str(strategy), 0.0), admissible_strategies.index(strategy)),
    )
    supporting = [
        entry.model_dump()
        for entry in sorted(exact_entries or broad_entries, key=lambda item: item.sequence_id, reverse=True)[:_MAX_SUPPORTING_ENTRIES]
    ]
    return StrategyJournalRecommendation(
        journal_key=journal_key,
        broader_key=broader_key,
        ordered_strategies=[str(item) for item in ordered],
        strategy_scores={str(key): round(float(value), 4) for key, value in scores.items()},
        supporting_entries=supporting,
    )


def record_strategy_outcome(
    *,
    task_family: str,
    retrieval_intent: RetrievalIntent,
    requested_strategy: str,
    applied_strategy: str,
    evidence_ready: bool,
    evidence_missing_count: int,
    compute_status: str,
    validator_verdict: str,
    final_verdict: str,
    success: bool,
    stop_reason: str = "",
    table_family: str = "",
) -> StrategyJournalEntry:
    global _STRATEGY_JOURNAL_SEQUENCE
    journal_key, broader_key = strategy_journal_key(
        task_family=task_family,
        retrieval_intent=retrieval_intent,
        table_family=table_family,
    )
    _STRATEGY_JOURNAL_SEQUENCE += 1
    entry = StrategyJournalEntry(
        journal_key=journal_key,
        broader_key=broader_key,
        task_family=str(task_family or "").strip().lower(),
        semantic_signature=_semantic_signature(retrieval_intent),
        aggregation_shape=str(retrieval_intent.aggregation_shape or "").strip().lower(),
        table_family=str(table_family or "").strip().lower(),
        requested_strategy=str(requested_strategy or "").strip(),
        applied_strategy=str(applied_strategy or "").strip(),
        evidence_ready=bool(evidence_ready),
        evidence_missing_count=max(0, int(evidence_missing_count or 0)),
        compute_status=str(compute_status or "").strip().lower(),
        validator_verdict=str(validator_verdict or "").strip().lower(),
        final_verdict=str(final_verdict or "").strip().lower(),
        success=bool(success),
        stop_reason=str(stop_reason or "").strip(),
        sequence_id=_STRATEGY_JOURNAL_SEQUENCE,
    )
    _STRATEGY_JOURNAL.append(entry)
    return entry


def strategy_journal_snapshot(limit: int = 12) -> list[dict[str, Any]]:
    capped = max(0, int(limit or 0))
    if capped == 0:
        return []
    return [entry.model_dump() for entry in list(_STRATEGY_JOURNAL)[-capped:]]
