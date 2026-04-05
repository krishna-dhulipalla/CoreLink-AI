from __future__ import annotations

from agent.context.extraction import extract_question_decomposition
from agent.contracts import QuestionDecomposition, SourceBundle
from agent.retrieval_reasoning import build_retrieval_intent


def test_decomposition_extracts_calendar_year_category_slots():
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="U.S. national defense total expenditures 1940",
        target_period="1940",
        entities=["U.S. national defense"],
    )

    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})

    assert retrieval_intent.entity == "U.S. national defense"
    assert retrieval_intent.metric == "total expenditures"
    assert retrieval_intent.period == "1940"
    assert retrieval_intent.granularity_requirement == "calendar_year"
    assert retrieval_intent.decomposition_confidence >= 0.7
    assert retrieval_intent.query_plan.primary_semantic_query.startswith(("Treasury Bulletin", "official government finance"))
    assert "calendar year" in retrieval_intent.query_plan.granularity_query.lower()


def test_decomposition_promotes_monthly_constraints_into_evidence_and_query_plan():
    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and "
        "all individual calendar months in 1940, what was the absolute percent change of these total sum values?"
    )
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="Treasury Bulletin expenditures 1953 1940",
        target_period="1953 1940",
        entities=[],
    )

    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})

    assert retrieval_intent.granularity_requirement == "monthly_series"
    assert "specifically only the reported values" in [item.lower() for item in retrieval_intent.include_constraints]
    assert "all individual calendar months" in [item.lower() for item in retrieval_intent.include_constraints]
    assert retrieval_intent.evidence_plan.required_month_coverage is True
    assert retrieval_intent.evidence_plan.required_month_count == 12
    assert any(requirement.kind == "include_constraints" for requirement in retrieval_intent.evidence_plan.requirements)
    assert "monthly" in retrieval_intent.query_plan.granularity_query.lower()


def test_decomposition_extracts_fiscal_year_entity_and_exclusion_constraints():
    prompt = "What were the total expenditures of the Veterans Administration in FY 1934 excluding trust accounts?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="Veterans Administration total expenditures FY 1934",
        target_period="1934",
        entities=["Veterans Administration"],
    )

    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})

    assert retrieval_intent.entity == "Veterans Administration"
    assert retrieval_intent.metric == "total expenditures"
    assert retrieval_intent.granularity_requirement == "fiscal_year"
    assert "trust accounts" in [item.lower() for item in retrieval_intent.exclude_constraints]
    assert "trust accounts" in [item.lower() for item in retrieval_intent.must_exclude_terms]
    assert any(requirement.kind == "exclude_constraints" for requirement in retrieval_intent.evidence_plan.requirements)


def test_decomposition_llm_fallback_merges_missing_fields(monkeypatch):
    prompt = "How much was it?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="",
        target_period="1945",
        entities=[],
    )

    monkeypatch.setattr(
        "agent.context.extraction.invoke_structured_output",
        lambda *args, **kwargs: (
            QuestionDecomposition(
                entity="Public debt",
                metric="public debt outstanding",
                period="1945",
                granularity_requirement="point_lookup",
                confidence=0.81,
                query_plan={},
            ),
            "fake-model",
        ),
    )

    decomposition = extract_question_decomposition(prompt, source_bundle, allow_llm_fallback=True)

    assert decomposition.used_llm_fallback is True
    assert decomposition.metric == "public debt outstanding"
    assert decomposition.period == "1945"
    assert decomposition.entity == "Public debt"
    assert decomposition.confidence >= 0.8
