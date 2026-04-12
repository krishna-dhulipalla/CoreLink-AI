from agent.contracts import CuratedContext, ExecutionJournal, OfficeQALLMRepairDecision, RetrievalIntent, SourceBundle
from agent.llm_repair import (
    maybe_repair_from_validator,
    maybe_rewrite_retrieval_path,
    officeqa_llm_repair_budget,
    officeqa_repair_universe_signature,
)


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


def test_retrieval_repair_prompt_includes_execution_journal_snapshot(monkeypatch):
    captured: dict[str, str] = {}

    def _fake_invoke(prompt: str):
        captured["prompt"] = prompt
        return OfficeQALLMRepairDecision(decision="keep", confidence=0.9)

    monkeypatch.setattr("agent.llm_repair._invoke_repair_decision", _fake_invoke)

    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        granularity_requirement="calendar_year",
        strategy="table_first",
    )

    decision = maybe_rewrite_retrieval_path(
        task_text="What were the total expenditures for U.S. national defense in 1940?",
        retrieval_intent=retrieval_intent,
        source_bundle=SourceBundle(
            task_text="What were the total expenditures for U.S. national defense in 1940?",
            focus_query="U.S. national defense expenditures 1940",
            target_period="1940",
            entities=["U.S. national defense"],
        ),
        execution_journal=ExecutionJournal(
            retrieval_iterations=2,
            retrieval_queries=["national defense expenditures 1940", "national defense calendar year 1940"],
            tool_results=[
                {
                    "type": "search_officeqa_documents",
                    "facts": {
                        "results": [
                            {"document_id": "treasury_bulletin_1940_01_json", "title": "Treasury Bulletin 1940-01", "score": 0.51}
                        ]
                    },
                }
            ],
        ),
        workpad={"officeqa_compute": {"semantic_issues": ["wrong row family"]}},
        curated_context=CuratedContext(
            structured_evidence={"tables": [{"table_family": "mixed_summary", "period_type": "calendar_year"}]}
        ),
        retrieval_strategy="table_first",
        evidence_gap="wrong document",
        current_query="national defense expenditures 1940",
        candidate_sources=[{"document_id": "treasury_bulletin_1940_01_json", "best_evidence_unit": {"table_family": "mixed_summary"}}],
    )

    assert decision is not None
    assert "ATTEMPTED_QUERIES=['national defense expenditures 1940', 'national defense calendar year 1940']" in captured["prompt"]
    assert "CANDIDATE_POOLS_SEEN=" in captured["prompt"]
    assert "REJECTED_EVIDENCE_FAMILIES=['mixed_summary']" in captured["prompt"]
    assert "COMPUTE_ADMISSIBILITY_FAILURES=['wrong row family']" in captured["prompt"]
    assert "STRATEGY_JOURNAL=" in captured["prompt"]
    assert "CANDIDATE_UNIVERSE_SIGNATURE=" in captured["prompt"]


def test_retrieval_repair_requires_stall_signals(monkeypatch):
    monkeypatch.setattr(
        "agent.llm_repair._invoke_repair_decision",
        lambda prompt: OfficeQALLMRepairDecision(decision="rewrite_query", revised_query="better query", confidence=0.9),
    )

    retrieval_intent = RetrievalIntent(
        entity="Public debt",
        metric="public debt outstanding",
        period="1945",
        granularity_requirement="point_lookup",
        strategy="table_first",
    )

    decision = maybe_rewrite_retrieval_path(
        task_text="What was public debt outstanding in 1945?",
        retrieval_intent=retrieval_intent,
        source_bundle=SourceBundle(
            task_text="What was public debt outstanding in 1945?",
            focus_query="public debt outstanding 1945",
            target_period="1945",
            entities=["Public debt"],
        ),
        execution_journal=ExecutionJournal(retrieval_iterations=0, retrieval_queries=[], tool_results=[]),
        workpad={},
        curated_context=CuratedContext(),
        retrieval_strategy="table_first",
        evidence_gap="wrong document",
        current_query="public debt outstanding 1945",
        candidate_sources=[],
    )

    assert decision is None


def test_retrieval_repair_skips_candidate_universe_already_marked_exhausted(monkeypatch):
    monkeypatch.setattr(
        "agent.llm_repair._invoke_repair_decision",
        lambda prompt: OfficeQALLMRepairDecision(decision="rewrite_query", revised_query="better query", confidence=0.9),
    )

    retrieval_intent = RetrievalIntent(
        entity="Public debt",
        metric="public debt outstanding",
        period="1945",
        granularity_requirement="point_lookup",
        strategy="table_first",
    )
    journal = ExecutionJournal(
        retrieval_iterations=2,
        retrieval_queries=["public debt outstanding 1945"],
    )
    universe_signature = officeqa_repair_universe_signature(
        retrieval_intent=retrieval_intent,
        execution_journal=journal,
        current_query="public debt outstanding 1945",
        candidate_sources=[],
    )

    decision = maybe_rewrite_retrieval_path(
        task_text="What was public debt outstanding in 1945?",
        retrieval_intent=retrieval_intent,
        source_bundle=SourceBundle(
            task_text="What was public debt outstanding in 1945?",
            focus_query="public debt outstanding 1945",
            target_period="1945",
            entities=["Public debt"],
        ),
        execution_journal=journal,
        workpad={"officeqa_exhausted_repair_universes": [{"candidate_universe_signature": universe_signature}]},
        curated_context=CuratedContext(),
        retrieval_strategy="table_first",
        evidence_gap="wrong document",
        current_query="public debt outstanding 1945",
        candidate_sources=[],
    )

    assert decision is None


def test_validator_repair_sets_typed_mutation_metadata(monkeypatch):
    monkeypatch.setattr(
        "agent.llm_repair._invoke_repair_decision",
        lambda prompt: OfficeQALLMRepairDecision(
            decision="change_strategy",
            preferred_strategy="hybrid",
            confidence=0.86,
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
        execution_journal=ExecutionJournal(retrieval_iterations=2, retrieval_queries=["Veterans Administration FY 1934"]),
        workpad={"officeqa_repair_failures": [{"code": "wrong_document"}]},
        review_feedback={
            "repair_target": "gather",
            "missing_dimensions": ["wrong document", "year coverage"],
            "remediation_codes": ["RETRIEVE_EXACT_PERIOD"],
        },
        candidate_sources=[{"document_id": "treasury_bulletin_1939_03_json"}],
    )

    assert decision is not None
    assert decision.mutation_class == "strategy_rotation"
    assert decision.prior_regime_exhausted is True
    assert decision.candidate_universe_signature
