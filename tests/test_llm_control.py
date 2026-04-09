from agent.contracts import RetrievalIntent
from agent.llm_control import (
    officeqa_llm_control_budget,
    should_use_source_rerank_llm,
    should_use_table_rerank_llm,
)


def _base_retrieval_intent(**overrides):
    payload = {
        "entity": "Postal Savings System",
        "metric": "total expenditures",
        "period": "1940",
        "period_type": "calendar_year",
        "target_years": ["1940"],
        "publication_year_window": ["1939", "1940", "1941"],
        "preferred_publication_years": ["1941", "1940", "1939"],
        "granularity_requirement": "calendar_year",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "calendar_year_total",
        "analysis_modes": [],
        "strategy": "table_first",
        "decomposition_confidence": 0.86,
    }
    payload.update(overrides)
    return RetrievalIntent(**payload)


def test_officeqa_llm_control_budget_expands_for_hard_semantic_cases():
    easy = _base_retrieval_intent()
    hard = _base_retrieval_intent(strategy="multi_table", analysis_modes=["inflation_adjustment"])

    easy_budget = officeqa_llm_control_budget(easy)
    hard_budget = officeqa_llm_control_budget(hard)

    assert easy_budget["retrieval_rerank_calls"] == 2
    assert easy_budget["table_rerank_calls"] == 2
    assert hard_budget["retrieval_rerank_calls"] == 3
    assert hard_budget["table_rerank_calls"] == 3


def test_source_rerank_llm_triggers_on_publication_year_mismatch():
    retrieval_intent = _base_retrieval_intent()
    needed, reason = should_use_source_rerank_llm(
        retrieval_intent=retrieval_intent,
        candidate_sources=[
            {
                "document_id": "treasury_bulletin_1938_12_json",
                "score": 2.1,
                "metadata": {"publication_year": "1938"},
                "best_evidence_unit": {"table_confidence": 0.84, "period_type": "calendar_year"},
            },
            {
                "document_id": "treasury_bulletin_1941_11_json",
                "score": 1.95,
                "metadata": {"publication_year": "1941"},
                "best_evidence_unit": {"table_confidence": 0.8, "period_type": "calendar_year"},
            },
        ],
        evidence_gap="",
    )

    assert needed is True
    assert reason == "publication_year_mismatch"


def test_table_rerank_llm_triggers_on_low_structural_confidence():
    retrieval_intent = _base_retrieval_intent()
    needed, reason = should_use_table_rerank_llm(
        retrieval_intent=retrieval_intent,
        table_candidates=[
            {
                "locator": "table 4",
                "ranking_score": 1.8,
                "table_confidence": 0.58,
                "table_family_confidence": 0.62,
                "period_type": "calendar_year",
            },
            {
                "locator": "table 7",
                "ranking_score": 1.45,
                "table_confidence": 0.88,
                "table_family_confidence": 0.82,
                "period_type": "calendar_year",
            },
        ],
        evidence_gap="",
    )

    assert needed is True
    assert reason == "low_table_confidence"


def test_table_rerank_llm_triggers_on_period_type_mismatch():
    retrieval_intent = _base_retrieval_intent(period_type="calendar_year", granularity_requirement="calendar_year")
    needed, reason = should_use_table_rerank_llm(
        retrieval_intent=retrieval_intent,
        table_candidates=[
            {
                "locator": "table annual",
                "ranking_score": 2.1,
                "table_confidence": 0.91,
                "table_family_confidence": 0.9,
                "period_type": "fiscal_year",
            },
            {
                "locator": "table calendar",
                "ranking_score": 1.95,
                "table_confidence": 0.88,
                "table_family_confidence": 0.87,
                "period_type": "calendar_year",
            },
        ],
        evidence_gap="",
    )

    assert needed is True
    assert reason == "table_period_type_mismatch"
