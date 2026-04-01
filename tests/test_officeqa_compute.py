import asyncio

from agent.benchmarks.officeqa_compute import compute_officeqa_result
from agent.contracts import RetrievalIntent, SourceBundle
from agent.nodes.orchestrator import make_executor
from agent.retrieval_reasoning import build_retrieval_intent
from test_utils import make_state


def _monthly_value(year: int, month: str, raw_value: float, *, metric: str = "Expenditures (million dollars)", citation: str | None = None) -> dict:
    return {
        "document_id": f"treasury_{year}_json",
        "citation": citation or f"treasury_{year}.json",
        "page_locator": "page 1",
        "table_locator": "National Defense",
        "row_index": 0,
        "row_label": month,
        "column_index": 1,
        "column_label": metric,
        "raw_value": str(raw_value),
        "numeric_value": raw_value,
        "normalized_value": raw_value * 1_000_000.0,
        "unit": "million",
        "unit_multiplier": 1_000_000.0,
        "unit_kind": "currency",
    }


def _annual_value(year: int, raw_value: float, *, row_label: str, metric: str) -> dict:
    return {
        "document_id": f"treasury_{year}_json",
        "citation": f"treasury_{year}.json",
        "page_locator": "page 1",
        "table_locator": "Annual Summary",
        "row_index": 0,
        "row_label": row_label,
        "column_index": 1,
        "column_label": metric,
        "raw_value": str(raw_value),
        "numeric_value": raw_value,
        "normalized_value": raw_value,
        "unit": "",
        "unit_multiplier": 1.0,
        "unit_kind": "scalar",
    }


def _structured(values: list[dict]) -> dict:
    return {
        "document_evidence": [],
        "tables": [],
        "values": values,
        "page_chunks": [],
        "units_seen": ["million"],
        "value_count": len(values),
        "provenance_complete": True,
    }


def _table_tool_result(year: int, monthly_value: float) -> dict:
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return {
        "type": "fetch_officeqa_table",
        "retrieval_status": "ok",
        "facts": {
            "document_id": f"treasury_{year}_json",
            "citation": f"treasury_{year}.json",
            "metadata": {"file_name": f"treasury_{year}.json", "format": "json", "officeqa_status": "ok"},
            "chunks": [],
            "tables": [
                {
                    "locator": "National Defense",
                    "headers": ["Month", "Expenditures (million dollars)"],
                    "rows": [[month, str(monthly_value)] for month in months],
                    "citation": f"treasury_{year}.json",
                    "unit_hint": "million dollars",
                }
            ],
            "numeric_summaries": [{"metric": "expenditures", "value": {"min": monthly_value, "max": monthly_value}}],
        },
    }


def test_compute_officeqa_monthly_sum_percent_change():
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    values = [*[_monthly_value(1940, month, 100.0) for month in months], *[_monthly_value(1953, month, 200.0) for month in months]]
    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all "
        "individual calendar months in 1940, what was the absolute percent change of these total sum values? "
        "Rounded to the nearest hundredths."
    )
    retrieval_intent = RetrievalIntent(
        entity="National Defense",
        metric="absolute percent change",
        period="1953 1940",
        document_family="treasury_bulletin",
        aggregation_shape="monthly_sum_percent_change",
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "ok"
    assert result.operation == "monthly_sum_percent_change"
    assert result.display_value == "100.00"
    assert len(result.ledger) == 3


def test_compute_officeqa_fiscal_year_total_uses_cross_year_months():
    values = []
    for month in ("July", "August", "September", "October", "November", "December"):
        values.append(_monthly_value(1939, month, 10.0))
    for month in ("January", "February", "March", "April", "May", "June"):
        values.append(_monthly_value(1940, month, 20.0))
    prompt = "What were the total expenditures for FY 1940?"
    retrieval_intent = RetrievalIntent(
        entity="National Defense",
        metric="total expenditures",
        period="1940",
        document_family="treasury_bulletin",
        aggregation_shape="fiscal_year_total",
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "ok"
    assert result.operation == "fiscal_year_total"
    assert result.final_value == 180_000_000.0
    assert result.display_value == "180000000"


def test_compute_officeqa_inflation_adjusted_difference_uses_cpi_support():
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    values = [*[_monthly_value(1940, month, 100.0) for month in months], *[_monthly_value(1953, month, 200.0) for month in months]]
    values.extend(
        [
            _annual_value(1940, 100.0, row_label="Annual average CPI", metric="CPI-U"),
            _annual_value(1953, 120.0, row_label="Annual average CPI", metric="CPI-U"),
        ]
    )
    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all individual "
        "calendar months in 1940, what was the inflation-adjusted absolute difference of these total sum values?"
    )
    retrieval_intent = RetrievalIntent(
        entity="National Defense",
        metric="absolute difference",
        period="1953 1940",
        document_family="treasury_bulletin",
        aggregation_shape="inflation_adjusted_monthly_difference",
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "ok"
    assert result.operation == "inflation_adjusted_monthly_difference"
    assert result.final_value == 960_000_000.0


def test_executor_prefers_deterministic_officeqa_compute_without_llm(monkeypatch):
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    structured = _structured([*[_monthly_value(1940, month, 100.0) for month in months], *[_monthly_value(1953, month, 200.0) for month in months]])
    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all "
        "individual calendar months in 1940, what was the absolute percent change of these total sum values?"
    )
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="monthly expenditures 1953 1940",
        target_period="1953 1940",
        entities=["National Defense"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})

    state = make_state(
        prompt,
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "routing_rationale": "officeqa deterministic compute",
            "confidence": 0.99,
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        tool_plan={"selected_tools": [], "pending_tools": [], "tool_families_needed": ["document_retrieval", "exact_compute"]},
        source_bundle=source_bundle.model_dump(),
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "xml"},
            "provenance_summary": {},
            "structured_evidence": structured,
        },
        execution_journal={
            "events": [],
            "tool_results": [_table_tool_result(1940, 100.0), _table_tool_result(1953, 200.0)],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 0,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
    )
    state["retrieval_intent"] = retrieval_intent.model_dump()

    def _should_not_build_model(**kwargs):
        raise AssertionError("LLM synthesis should not run for deterministic OfficeQA compute.")

    monkeypatch.setattr("agent.nodes.orchestrator.ChatOpenAI", _should_not_build_model)

    result = asyncio.run(make_executor({})(state))

    assert result["solver_stage"] == "SYNTHESIZE"
    assert "Final answer: 100.00" in str(result["messages"][0].content)
    assert result["curated_context"]["compute_result"]["status"] == "ok"
