from agent.contracts import OfficeQALLMRepairDecision, RetrievalIntent
from agent.llm_repair import maybe_repair_from_validator, officeqa_llm_repair_budget


def test_officeqa_llm_repair_budget_is_explicit_and_deeper_than_single_retry():
    budget = officeqa_llm_repair_budget()

    assert budget["query_rewrite_calls"] == 2
    assert budget["validator_repair_calls"] == 2


def test_validator_repair_supports_compute_target(monkeypatch):
    monkeypatch.setattr(
        "agent.llm_repair._invoke_repair_decision",
        lambda prompt: OfficeQALLMRepairDecision(
            decision="retune_table_query",
            revised_table_query="Veterans Administration expenditures fiscal year 1934",
            confidence=0.82,
        ),
    )

    retrieval_intent = RetrievalIntent(
        entity="Veterans Administration",
        metric="total expenditures",
        period="1934",
        granularity_requirement="fiscal_year",
        strategy="table_first",
    )

    decision = maybe_repair_from_validator(
        task_text="What were the total expenditures of the Veterans Administration in FY 1934?",
        retrieval_intent=retrieval_intent,
        review_feedback={
            "repair_target": "compute",
            "missing_dimensions": ["wrong row or column semantics", "wrong period slice"],
            "remediation_codes": ["REPAIR_COMPUTE_VALIDATION"],
        },
        candidate_sources=[],
    )

    assert decision is not None
