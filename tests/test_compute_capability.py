from agent.compute_capability import (
    clear_compute_capability_cache,
    maybe_acquire_officeqa_compute_result,
)
from agent.contracts import RetrievalIntent


def _monthly_value(year: int, month: str, raw_value: float) -> dict:
    return {
        "document_id": f"treasury_{year}_json",
        "citation": f"treasury_{year}.json",
        "page_locator": "page 1",
        "table_locator": "Monthly Expenditures",
        "table_family": "monthly_series",
        "row_index": 0,
        "row_label": month,
        "row_path": [month],
        "column_index": 1,
        "column_label": "Expenditures",
        "column_path": ["Expenditures"],
        "raw_value": str(raw_value),
        "numeric_value": raw_value,
        "normalized_value": raw_value,
        "unit": "million",
        "unit_multiplier": 1.0,
        "unit_kind": "currency",
        "structure_confidence": 0.95,
    }


def _structured(values: list[dict]) -> dict:
    return {
        "values": values,
        "tables": [],
        "page_chunks": [],
        "provenance_complete": True,
        "structure_confidence_summary": {
            "min_confidence": 0.95,
            "avg_confidence": 0.95,
            "max_confidence": 0.95,
            "low_confidence_value_count": 0,
            "low_confidence_table_count": 0,
            "table_confidence_gate_passed": True,
        },
    }


def test_compute_capability_acquires_and_caches_validated_function(monkeypatch):
    clear_compute_capability_cache()
    prompt = "Using Treasury Bulletin data, calculate the standard deviation of monthly expenditures for 1953."
    retrieval_intent = RetrievalIntent(
        entity="Expenditures",
        metric="standard deviation of expenditures",
        period="1953",
        period_type="calendar_year",
        target_years=["1953"],
        aggregation_shape="point_lookup",
        analysis_modes=["statistical_analysis"],
        answer_mode="deterministic_compute",
        compute_policy="required",
    )
    structured = _structured(
        [
            _monthly_value(1953, "January", 10.0),
            _monthly_value(1953, "February", 20.0),
            _monthly_value(1953, "March", 30.0),
        ]
    )
    call_counter = {"count": 0}

    def _fake_invoke(*args, **kwargs):
        call_counter["count"] += 1
        return (
            {
                "function_name": "compute_capability",
                "function_code": (
                    "def compute_capability(records, context):\n"
                    "    target_years = set(context.get('target_years', []))\n"
                    "    selected = [record for record in records if target_years.intersection(set(record.get('year_refs', [])))]\n"
                    "    values = [float(record.get('normalized_value')) for record in selected if record.get('normalized_value') is not None]\n"
                    "    if len(values) < 2:\n"
                    "        return {'error': 'insufficient'}\n"
                    "    return {\n"
                    "        'final_value': statistics.pstdev(values),\n"
                    "        'selected_record_ids': [record['record_id'] for record in selected],\n"
                    "        'explanation': 'population standard deviation'\n"
                    "    }\n"
                ),
                "rationale": "Compute the population standard deviation over the selected monthly values.",
                "required_record_fields": ["record_id", "normalized_value", "year_refs"],
                "validation_checks": ["order_invariant"],
            },
            "fake-capability-model",
        )

    monkeypatch.setattr("agent.compute_capability.invoke_structured_output", _fake_invoke)

    result, workpad, llm_state = maybe_acquire_officeqa_compute_result(
        task_text=prompt,
        retrieval_intent=retrieval_intent,
        structured_evidence=structured,
        workpad={},
        llm_control_state={"compute_capability_calls": 0},
        llm_control_budget={"compute_capability_calls": 1},
    )

    assert result is not None
    assert result.status == "ok"
    assert result.capability_source == "synthesized"
    assert result.capability_validated is True
    assert result.final_value == 8.16496580927726
    assert call_counter["count"] == 1
    assert workpad["officeqa_latest_llm_usage"]["category"] == "compute_capability_llm"
    assert llm_state["compute_capability_calls"] == 1

    cached_result, cached_workpad, cached_state = maybe_acquire_officeqa_compute_result(
        task_text=prompt,
        retrieval_intent=retrieval_intent,
        structured_evidence=structured,
        workpad={},
        llm_control_state={"compute_capability_calls": 0},
        llm_control_budget={"compute_capability_calls": 1},
    )

    assert cached_result is not None
    assert cached_result.status == "ok"
    assert cached_result.capability_source == "cached"
    assert call_counter["count"] == 1
    assert cached_workpad["officeqa_latest_llm_usage"]["reason"] == "cache_hit"
    assert cached_state["compute_capability_calls"] == 0


def test_compute_capability_rejects_unsafe_generated_code(monkeypatch):
    clear_compute_capability_cache()
    retrieval_intent = RetrievalIntent(
        entity="Expenditures",
        metric="standard deviation of expenditures",
        period="1953",
        target_years=["1953"],
        aggregation_shape="point_lookup",
        analysis_modes=["statistical_analysis"],
        answer_mode="deterministic_compute",
        compute_policy="required",
    )

    def _fake_invoke(*args, **kwargs):
        return (
            {
                "function_name": "compute_capability",
                "function_code": (
                    "def compute_capability(records, context):\n"
                    "    import os\n"
                    "    return {'final_value': 1.0, 'selected_record_ids': []}\n"
                ),
                "rationale": "unsafe",
                "required_record_fields": [],
                "validation_checks": [],
            },
            "fake-capability-model",
        )

    monkeypatch.setattr("agent.compute_capability.invoke_structured_output", _fake_invoke)

    result, workpad, llm_state = maybe_acquire_officeqa_compute_result(
        task_text="calculate the standard deviation",
        retrieval_intent=retrieval_intent,
        structured_evidence=_structured([_monthly_value(1953, "January", 10.0), _monthly_value(1953, "February", 20.0)]),
        workpad={},
        llm_control_state={"compute_capability_calls": 0},
        llm_control_budget={"compute_capability_calls": 1},
    )

    assert result is not None
    assert result.status == "unsupported"
    assert "Compute capability acquisition failed" in " ".join(result.validation_errors)
    assert any("unsupported capability syntax" in error.lower() or "blocked capability name" in error.lower() for error in result.validation_errors)
    assert workpad["officeqa_latest_llm_usage"]["category"] == "compute_capability_llm"
    assert workpad["officeqa_latest_llm_usage"]["used"] is True
    assert llm_state["compute_capability_calls"] == 1
