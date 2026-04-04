import json
from pathlib import Path

from agent.tools.tsr_fallback import compare_dense_table_normalizers, select_dense_table_normalization


def _fixture(name: str) -> dict:
    payload = json.loads(Path("eval/officeqa_tsr_fixture_set.json").read_text(encoding="utf-8"))
    for item in payload:
        if item.get("id") == name:
            return dict(item)
    raise AssertionError(f"Missing fixture: {name}")


def test_tsr_fallback_comparison_prefers_split_merge_on_sparse_parent_headers():
    fixture = _fixture("sparse_parent_header_years")

    comparison = compare_dense_table_normalizers(
        fixture["grid"],
        locator=fixture["locator"],
        unit_hint=fixture["unit_hint"],
    )

    assert comparison["diagnostics"]["selected_mode"] == "tsr_split_merge_heuristic"
    selected = comparison["selected"]
    assert len(selected["column_paths"][1]) >= 2
    assert comparison["diagnostics"]["fallback_score"] > comparison["diagnostics"]["default_score"]


def test_tsr_fallback_selection_remains_default_when_experiment_disabled(monkeypatch):
    fixture = _fixture("sparse_month_header_band")
    monkeypatch.delenv("OFFICEQA_ENABLE_TSR_FALLBACK", raising=False)

    selected, diagnostics = select_dense_table_normalization(
        fixture["grid"],
        locator=fixture["locator"],
        unit_hint=fixture["unit_hint"],
    )

    assert diagnostics["selected_mode"] == "tsr_split_merge_heuristic"
    assert selected["experimental_tsr"]["enabled"] is False
    assert selected["column_paths"][1] == ["January"]


def test_tsr_fallback_selection_uses_fallback_when_enabled(monkeypatch):
    fixture = _fixture("sparse_month_header_band")
    monkeypatch.setenv("OFFICEQA_ENABLE_TSR_FALLBACK", "1")

    selected, diagnostics = select_dense_table_normalization(
        fixture["grid"],
        locator=fixture["locator"],
        unit_hint=fixture["unit_hint"],
    )

    assert diagnostics["selected_mode"] == "tsr_split_merge_heuristic"
    assert selected["experimental_tsr"]["enabled"] is True
    assert len(selected["column_paths"][1]) >= 2
