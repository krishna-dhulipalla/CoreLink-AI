from agent.contracts import RetrievalIntent, SourceBundle
from agent.retrieval_candidates import rank_search_candidates, search_result_candidates


def test_search_result_candidates_uses_typed_candidate_schema_and_dedupes():
    tool_result = {
        "facts": {
            "documents": [
                {
                    "document_id": "doc-1",
                    "citation": "a.pdf",
                    "title": "Alpha",
                    "rank": 2,
                    "score": 0.3,
                    "metadata": {"publication_year": "1945"},
                }
            ],
            "results": [
                {
                    "document_id": "doc-1",
                    "citation": "a.pdf",
                    "title": "Alpha Better",
                    "rank": 1,
                    "score": 0.8,
                    "metadata": {"best_evidence_unit": {"table_family": "debt_or_balance_sheet"}},
                }
            ],
        }
    }

    candidates = search_result_candidates(tool_result)

    assert len(candidates) == 1
    assert candidates[0]["document_id"] == "doc-1"
    assert candidates[0]["rank"] == 1
    assert candidates[0]["score"] == 0.8
    assert candidates[0]["metadata"]["publication_year"] == "1945"


def test_rank_search_candidates_prefers_semantically_aligned_candidate():
    retrieval_intent = RetrievalIntent(
        entity="Public debt",
        metric="public debt outstanding",
        period="1945",
        granularity_requirement="point_lookup",
        strategy="table_first",
        document_family="treasury_bulletin",
    )
    source_bundle = SourceBundle(
        task_text="According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
        focus_query="public debt outstanding 1945",
        target_period="1945",
        entities=["Public debt"],
    )
    candidates = [
        {
            "document_id": "doc-generic",
            "citation": "generic.pdf",
            "title": "Treasury Bulletin Summary",
            "rank": 1,
            "score": 0.9,
            "metadata": {"best_evidence_unit": {"table_family": "annual_summary", "table_confidence": 0.5}},
        },
        {
            "document_id": "doc-debt",
            "citation": "debt.pdf",
            "title": "Public Debt Outstanding",
            "rank": 2,
            "score": 0.7,
            "metadata": {"best_evidence_unit": {"table_family": "debt_or_balance_sheet", "table_confidence": 0.9}},
        },
    ]

    ranked = rank_search_candidates(candidates, retrieval_intent, source_bundle, {"benchmark_adapter": "officeqa"})

    assert ranked[0]["document_id"] == "doc-debt"
