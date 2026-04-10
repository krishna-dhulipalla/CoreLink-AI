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


def test_source_rerank_llm_skips_when_candidate_pool_needs_widening():
    retrieval_intent = _base_retrieval_intent()
    needed, reason = should_use_source_rerank_llm(
        retrieval_intent=retrieval_intent,
        candidate_sources=[
            {
                "document_id": "treasury_bulletin_1940_07_json",
                "score": 2.1,
                "metadata": {"publication_year": "1940"},
                "best_evidence_unit": {"table_confidence": 0.78, "period_type": "calendar_year"},
            },
            {
                "document_id": "treasury_bulletin_1940_03_json",
                "score": 1.95,
                "metadata": {"publication_year": "1940"},
                "best_evidence_unit": {"table_confidence": 0.75, "period_type": "calendar_year"},
            },
        ],
        evidence_gap="source pool too narrow",
    )

    assert needed is False
    assert reason == "candidate_pool_requires_widening"


def test_source_rerank_llm_skips_narrow_margin_when_top_candidate_is_semantically_stable():
    retrieval_intent = _base_retrieval_intent(
        metric="public debt outstanding",
        preferred_publication_years=["1945", "1946", "1944"],
    )
    needed, reason = should_use_source_rerank_llm(
        retrieval_intent=retrieval_intent,
        candidate_sources=[
            {
                "document_id": "treasury_bulletin_1945_08_json",
                "score": 7.32,
                "metadata": {"publication_year": "1945"},
                "best_evidence_unit": {
                    "table_confidence": 0.88,
                    "period_type": "point_lookup",
                    "table_family": "debt_or_balance_sheet",
                },
            },
            {
                "document_id": "treasury_bulletin_1946_08_json",
                "score": 7.14,
                "metadata": {"publication_year": "1946"},
                "best_evidence_unit": {
                    "table_confidence": 0.85,
                    "period_type": "point_lookup",
                    "table_family": "debt_or_balance_sheet",
                },
            },
        ],
        evidence_gap="",
    )

    assert needed is False
    assert reason == "deterministic_top_candidate_stable"


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
