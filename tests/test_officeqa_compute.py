import asyncio

from agent.benchmarks.officeqa_compute import compute_officeqa_result
from agent.contracts import RetrievalIntent, SourceBundle
from agent.nodes.orchestrator import make_executor
from agent.retrieval_reasoning import build_retrieval_intent
from test_utils import make_state


def _monthly_value(
    year: int,
    month: str,
    raw_value: float,
    *,
    metric: str = "Expenditures (million dollars)",
    citation: str | None = None,
    table_family: str = "monthly_series",
    table_locator: str = "National Defense",
) -> dict:
    return {
        "document_id": f"treasury_{year}_json",
        "citation": citation or f"treasury_{year}.json",
        "page_locator": "page 1",
        "table_locator": table_locator,
        "table_family": table_family,
        "row_index": 0,
        "row_label": month,
        "row_path": [month],
        "column_index": 1,
        "column_label": metric,
        "column_path": [metric],
        "raw_value": str(raw_value),
        "numeric_value": raw_value,
        "normalized_value": raw_value * 1_000_000.0,
        "unit": "million",
        "unit_multiplier": 1_000_000.0,
        "unit_kind": "currency",
        "structure_confidence": 0.95,
    }


def _annual_value(
    year: int,
    raw_value: float,
    *,
    row_label: str,
    metric: str,
    table_family: str = "category_breakdown",
    table_locator: str = "Annual Summary",
    column_path: list[str] | None = None,
) -> dict:
    return {
        "document_id": f"treasury_{year}_json",
        "citation": f"treasury_{year}.json",
        "page_locator": "page 1",
        "table_locator": table_locator,
        "table_family": table_family,
        "row_index": 0,
        "row_label": row_label,
        "row_path": [row_label],
        "column_index": 1,
        "column_label": metric,
        "column_path": column_path or [metric],
        "raw_value": str(raw_value),
        "numeric_value": raw_value,
        "normalized_value": raw_value,
        "unit": "",
        "unit_multiplier": 1.0,
        "unit_kind": "scalar",
        "structure_confidence": 0.95,
    }


def _structured(values: list[dict]) -> dict:
    return {
        "document_evidence": [],
        "tables": [],
        "values": values,
        "page_chunks": [],
        "structure_confidence_summary": {
            "min_confidence": 0.95,
            "avg_confidence": 0.95,
            "max_confidence": 0.95,
            "low_confidence_value_count": 0,
            "low_confidence_table_count": 0,
            "table_confidence_gate_passed": True,
        },
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
    assert result.selection_reasoning
    assert result.rejected_alternatives
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


def test_compute_officeqa_explicit_millions_contract_formats_display_in_millions():
    values = []
    for month in (
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
    ):
        values.append(_monthly_value(1940, month, 10.0))
    prompt = "What were the total expenditures (in millions of nominal dollars) for the calendar year 1940?"
    retrieval_intent = RetrievalIntent(
        entity="National Defense",
        metric="total expenditures",
        period="1940",
        granularity_requirement="calendar_year",
        expected_answer_unit_basis="millions_nominal_dollars",
        document_family="treasury_bulletin",
        aggregation_shape="calendar_year_total",
        answer_mode="deterministic_compute",
        compute_policy="required",
        evidence_plan={"expected_answer_unit_basis": "millions_nominal_dollars"},
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "ok"
    assert result.final_value == 120_000_000.0
    assert result.display_value == "120"
    assert result.answer_unit_basis == "millions_nominal_dollars"


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


def test_compute_officeqa_point_lookup_selects_best_year_and_metric_match():
    values = [
        {
            "document_id": "treasury_bulletin_1945_01_json",
            "citation": "treasury_bulletin_1945_01.json",
            "page_locator": "page 29",
            "table_locator": "table 19",
            "row_index": 10,
            "row_label": "Total public debt",
            "row_path": ["Total public debt outstanding"],
            "column_index": 8,
            "column_label": "Estimated 1/",
            "column_path": ["End of fiscal years, 1941 to 1945", "1945"],
            "raw_value": "258682",
            "numeric_value": 258682.0,
            "normalized_value": 258682.0,
            "unit": "",
            "unit_multiplier": 1.0,
            "unit_kind": "scalar",
            "structure_confidence": 0.95,
        },
        {
            "document_id": "treasury_bulletin_1945_01_json",
            "citation": "treasury_bulletin_1945_01.json",
            "page_locator": "page 29",
            "table_locator": "table 19",
            "row_index": 11,
            "row_label": "Total interest-bearing debt",
            "row_path": ["Total interest-bearing debt"],
            "column_index": 8,
            "column_label": "Estimated 1/",
            "column_path": ["End of fiscal years, 1941 to 1945", "1945"],
            "raw_value": "258682",
            "numeric_value": 258682.0,
            "normalized_value": 258682.0,
            "unit": "",
            "unit_multiplier": 1.0,
            "unit_kind": "scalar",
            "structure_confidence": 0.95,
        },
        {
            "document_id": "treasury_bulletin_1945_01_json",
            "citation": "treasury_bulletin_1945_01.json",
            "page_locator": "page 29",
            "table_locator": "table 19",
            "row_index": 10,
            "row_label": "Total public debt",
            "row_path": ["Total public debt outstanding"],
            "column_index": 7,
            "column_label": "Actual",
            "column_path": ["End of fiscal years, 1941 to 1945", "1944"],
            "raw_value": "201003",
            "numeric_value": 201003.0,
            "normalized_value": 201003.0,
            "unit": "",
            "unit_multiplier": 1.0,
            "unit_kind": "scalar",
            "structure_confidence": 0.95,
        },
    ]
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    retrieval_intent = RetrievalIntent(
        entity="Public debt",
        metric="public debt outstanding",
        period="1945",
        document_family="treasury_bulletin",
        aggregation_shape="point_lookup",
        answer_mode="deterministic_compute",
        compute_policy="required",
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "ok"
    assert result.operation == "point_lookup"
    assert result.display_value == "258682"
    assert result.ledger[0]["operator"] == "point_lookup"


def test_compute_officeqa_refuses_low_confidence_structure():
    values = [
        {
            "document_id": "treasury_bulletin_1945_01_json",
            "citation": "treasury_bulletin_1945_01.json",
            "page_locator": "page 29",
            "table_locator": "table 19",
            "row_index": 10,
            "row_label": "Total public debt",
            "row_path": ["Total public debt outstanding"],
            "column_index": 8,
            "column_label": "Estimated 1/",
            "column_path": ["End of fiscal years, 1941 to 1945", "1945"],
            "raw_value": "258682",
            "numeric_value": 258682.0,
            "normalized_value": 258682.0,
            "unit": "",
            "unit_multiplier": 1.0,
            "unit_kind": "scalar",
            "structure_confidence": 0.42,
        }
    ]
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    retrieval_intent = RetrievalIntent(
        entity="Public debt",
        metric="public debt outstanding",
        period="1945",
        document_family="treasury_bulletin",
        aggregation_shape="point_lookup",
        answer_mode="deterministic_compute",
        compute_policy="required",
    )

    result = compute_officeqa_result(
        prompt,
        retrieval_intent,
        {
            "tables": [],
            "values": values,
            "structure_confidence_summary": {
                "min_confidence": 0.42,
                "avg_confidence": 0.42,
                "max_confidence": 0.42,
                "low_confidence_value_count": 1,
                "low_confidence_table_count": 1,
                "table_confidence_gate_passed": False,
            },
            "provenance_complete": True,
        },
    )

    assert result.status == "insufficient"
    assert "Low-confidence table structure" in result.validation_errors[0]


def test_compute_officeqa_point_lookup_rejects_navigational_page_reference_cells():
    values = [
        {
            "document_id": "treasury_bulletin_1945_01_json",
            "citation": "treasury_bulletin_1945_01.json",
            "page_locator": "page 6",
            "table_locator": "table 1",
            "row_index": 8,
            "row_label": "Public debt and guaranteed obligations outstanding",
            "row_path": ["Public debt and guaranteed obligations outstanding"],
            "column_index": 2,
            "column_label": "Issue and page number | Jan.",
            "column_path": ["Issue and page number", "Jan."],
            "raw_value": "3",
            "numeric_value": 3.0,
            "normalized_value": 3.0,
            "unit": "",
            "unit_multiplier": 1.0,
            "unit_kind": "scalar",
            "structure_confidence": 0.95,
        }
    ]
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    retrieval_intent = RetrievalIntent(
        entity="Public debt",
        metric="public debt outstanding",
        period="1945",
        document_family="treasury_bulletin",
        aggregation_shape="point_lookup",
        answer_mode="deterministic_compute",
        compute_policy="required",
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "insufficient"
    assert "navigational" in result.validation_errors[0].lower()


def test_compute_officeqa_calendar_year_total_rejects_partial_year_column():
    values = [
        _annual_value(
            1940,
            4748.0,
            row_label="National defense and related activities",
            metric="Actual 6 months 1940",
            table_family="annual_summary",
            table_locator="Summary of expenditures",
            column_path=["Actual 6 months 1940"],
        )
    ]
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    retrieval_intent = RetrievalIntent(
        entity="National defense",
        metric="total expenditures",
        period="1940",
        granularity_requirement="calendar_year",
        document_family="treasury_bulletin",
        aggregation_shape="calendar_year_total",
        answer_mode="deterministic_compute",
        compute_policy="required",
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "insufficient"
    assert any("wrong period slice" in item.lower() for item in result.validation_errors)
    assert result.semantic_diagnostics["period_slice_status"] == "wrong period slice"


def test_compute_officeqa_calendar_year_total_rejects_wrong_row_family():
    values = [
        _annual_value(
            1940,
            4748.0,
            row_label="Total Expenditures",
            metric="Calendar year 1940",
            table_family="category_breakdown",
            table_locator="Summary of budget receipts and expenditures",
            column_path=["Calendar year 1940"],
        )
    ]
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    retrieval_intent = RetrievalIntent(
        entity="National defense",
        metric="total expenditures",
        period="1940",
        granularity_requirement="calendar_year",
        document_family="treasury_bulletin",
        aggregation_shape="calendar_year_total",
        answer_mode="deterministic_compute",
        compute_policy="required",
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "insufficient"
    assert any("wrong row family" in item.lower() for item in result.validation_errors)
    assert result.semantic_diagnostics["row_family_status"] == "wrong row family"


def test_compute_officeqa_rejects_typing_ambiguity_on_selected_value():
    values = [
        {
            "document_id": "treasury_bulletin_1940_12_json",
            "citation": "treasury_bulletin_1940_12.json",
            "page_locator": "page 22",
            "table_locator": "table 4",
            "table_family": "category_breakdown",
            "table_family_confidence": 0.72,
            "period_type": "calendar_year",
            "typing_ambiguities": ["family_drift:annual_summary->category_breakdown"],
            "row_index": 2,
            "row_label": "National defense and related activities",
            "row_path": ["National defense and related activities"],
            "column_index": 1,
            "column_label": "Expenditures, calendar year 1940",
            "column_path": ["Expenditures", "Calendar year 1940"],
            "raw_value": "2602",
            "numeric_value": 2602.0,
            "normalized_value": 2602_000_000.0,
            "unit": "million",
            "unit_multiplier": 1_000_000.0,
            "unit_kind": "currency",
            "structure_confidence": 0.95,
        }
    ]
    prompt = "What were the total expenditures (in millions of nominal dollars) for U.S national defense in the calendar year of 1940?"
    retrieval_intent = RetrievalIntent(
        entity="U.S national defense",
        metric="total expenditures",
        period="1940",
        period_type="calendar_year",
        granularity_requirement="calendar_year",
        expected_answer_unit_basis="millions_nominal_dollars",
        document_family="treasury_bulletin",
        aggregation_shape="calendar_year_total",
        answer_mode="deterministic_compute",
        compute_policy="required",
        evidence_plan={"expected_answer_unit_basis": "millions_nominal_dollars"},
    )

    result = compute_officeqa_result(prompt, retrieval_intent, _structured(values))

    assert result.status == "insufficient"
    assert any("typing ambiguity" in item.lower() for item in result.validation_errors)


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
