from agent.budget import BudgetTracker
from agent.cost import CostTracker


def test_budget_tracker_configures_caps_by_complexity_tier():
    budget = BudgetTracker()
    budget.configure(complexity_tier="simple_exact", template_id="quant_inline_exact")

    summary = budget.summary()

    assert summary["complexity_tier"] == "simple_exact"
    assert summary["template_id"] == "quant_inline_exact"
    assert summary["tool_calls_cap"] == 1
    assert summary["backtrack_cap"] == 0


def test_cost_tracker_marks_unknown_model_pricing_as_unpriced():
    tracker = CostTracker()
    tracker.record(
        operator="solver_compute",
        model_name="Qwen/Qwen3-32B-fast",
        tokens_in=100,
        tokens_out=20,
        latency_ms=10.0,
        success=True,
    )

    summary = tracker.summary()
    payload = tracker.trace_payload()

    assert summary["total_cost_usd"] is None
    assert summary["cost_estimate_status"] == "unpriced"
    assert "Qwen/Qwen3-32B-fast" in summary["unpriced_models"]
    assert payload["traces"][0]["estimated_cost_usd"] is None
    assert payload["traces"][0]["pricing_known"] is False
