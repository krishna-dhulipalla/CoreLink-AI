from engine.agent.contracts import RetrievalIntent
from engine.agent.strategy_journal import (
    clear_strategy_journal,
    record_strategy_outcome,
    recommend_strategy_order,
    strategy_journal_snapshot,
)


def _intent(**overrides):
    payload = {
        "entity": "National defense",
        "metric": "total expenditures",
        "period": "1940",
        "period_type": "calendar_year",
        "granularity_requirement": "calendar_year",
        "aggregation_shape": "calendar_year_total",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "analysis_modes": [],
    }
    payload.update(overrides)
    return RetrievalIntent(**payload)


def test_strategy_journal_recommends_recent_successful_strategy():
    clear_strategy_journal()
    retrieval_intent = _intent()
    record_strategy_outcome(
        task_family="document_qa",
        retrieval_intent=retrieval_intent,
        requested_strategy="table_first",
        applied_strategy="table_first",
        evidence_ready=False,
        evidence_missing_count=2,
        compute_status="insufficient",
        validator_verdict="revise",
        final_verdict="fail",
        success=False,
        stop_reason="progress_stalled",
        table_family="mixed_summary",
    )
    record_strategy_outcome(
        task_family="document_qa",
        retrieval_intent=retrieval_intent,
        requested_strategy="hybrid",
        applied_strategy="hybrid",
        evidence_ready=True,
        evidence_missing_count=0,
        compute_status="ok",
        validator_verdict="pass",
        final_verdict="pass",
        success=True,
        stop_reason="",
        table_family="category_breakdown",
    )

    recommendation = recommend_strategy_order(
        task_family="document_qa",
        retrieval_intent=retrieval_intent,
        admissible_strategies=["table_first", "hybrid", "text_first"],
    )

    assert recommendation.ordered_strategies[0] == "hybrid"
    assert recommendation.strategy_scores["hybrid"] > recommendation.strategy_scores["table_first"]


def test_strategy_journal_snapshot_is_bounded_and_traceable():
    clear_strategy_journal()
    retrieval_intent = _intent(aggregation_shape="weighted_average", analysis_modes=["weighted_average"])
    for index in range(10):
        record_strategy_outcome(
            task_family="document_qa",
            retrieval_intent=retrieval_intent,
            requested_strategy="hybrid",
            applied_strategy="hybrid",
            evidence_ready=True,
            evidence_missing_count=0,
            compute_status="ok",
            validator_verdict="pass",
            final_verdict="pass",
            success=True,
            stop_reason="",
            table_family="category_breakdown",
        )

    snapshot = strategy_journal_snapshot(limit=4)

    assert len(snapshot) == 4
    assert all(item["applied_strategy"] == "hybrid" for item in snapshot)
