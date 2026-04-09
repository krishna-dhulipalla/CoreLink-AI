from agent.tools.table_normalization import normalize_dense_table_grid, normalize_flat_table
from agent.tools.tsr_fallback import select_dense_table_normalization


def test_normalize_flat_table_builds_column_and_row_paths():
    normalized = normalize_flat_table(
        ["Month", "Expenditures (million dollars)"],
        [["January", "100.0"], ["February", "101.5"]],
        locator="table 1",
        unit_hint="million dollars",
    )

    assert normalized["row_header_depth"] == 1
    assert normalized["display_headers"] == ["Row", "Expenditures (million dollars)"]
    assert normalized["row_records"][0]["row_path"] == ["January"]
    assert normalized["row_records"][0]["cells"][0]["column_path"] == ["Expenditures (million dollars)"]
    assert normalized["normalization_metrics"]["recovered_unit_coverage"] == 1.0


def test_normalize_dense_table_grid_collapses_repeated_hierarchical_headers():
    grid = [
        [
            {"text": "End of fiscal years, 1941 to 1945", "is_header": True, "origin_id": "h0"},
            {"text": "End of fiscal years, 1941 to 1945", "is_header": True, "origin_id": "h0"},
            {"text": "End of fiscal years, 1941 to 1945", "is_header": True, "origin_id": "h0"},
        ],
        [
            {"text": "", "is_header": True, "origin_id": "h1"},
            {"text": "1944", "is_header": True, "origin_id": "h2"},
            {"text": "1945", "is_header": True, "origin_id": "h3"},
        ],
        [
            {"text": "Public debt", "is_header": False, "origin_id": "r0c0"},
            {"text": "232,000", "is_header": False, "origin_id": "r0c1"},
            {"text": "278,000", "is_header": False, "origin_id": "r0c2"},
        ],
    ]

    normalized = normalize_dense_table_grid(grid, locator="table 19")

    assert normalized["column_paths"][1] == ["End of fiscal years, 1941 to 1945", "1944"]
    assert normalized["column_paths"][2] == ["End of fiscal years, 1941 to 1945", "1945"]
    assert normalized["row_records"][0]["row_path"] == ["Public debt"]
    assert normalized["display_headers"][1] == "End of fiscal years, 1941 to 1945 | 1944"
    assert normalized["normalization_metrics"]["duplicate_header_collapse_score"] <= 1.0


def test_tsr_fallback_auto_promotes_when_default_header_quality_is_low(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.tsr_fallback.compare_dense_table_normalizers",
        lambda *args, **kwargs: {
            "default": {"normalization_metrics": {"header_data_separation_quality": 0.32}},
            "fallback": {"normalization_metrics": {"header_data_separation_quality": 0.74}},
            "selected": {"normalization_metrics": {"header_data_separation_quality": 0.74}},
            "diagnostics": {
                "enabled": False,
                "default_score": 0.42,
                "fallback_score": 0.58,
                "score_delta": 0.16,
                "header_rows_considered": 2,
            },
        },
    )

    selected, diagnostics = select_dense_table_normalization([[{"text": "x", "is_header": True}]])

    assert diagnostics["auto_promoted"] is True
    assert diagnostics["selection_mode"] == "fallback_selected"
    assert selected["experimental_tsr"]["auto_promoted"] is True


def test_normalize_dense_table_grid_uses_empty_leading_cells_as_depth_signal():
    grid = [
        [
            {"text": "", "is_header": True, "origin_id": "h0"},
            {"text": "", "is_header": True, "origin_id": "h1"},
            {"text": "Amount", "is_header": True, "origin_id": "h2"},
        ],
        [
            {"text": "Internal revenue", "is_header": False, "origin_id": "r0c0"},
            {"text": "", "is_header": False, "origin_id": "r0c1"},
            {"text": "", "is_header": False, "origin_id": "r0c2"},
        ],
        [
            {"text": "", "is_header": False, "origin_id": "r1c0"},
            {"text": "Income Tax", "is_header": False, "origin_id": "r1c1"},
            {"text": "100", "is_header": False, "origin_id": "r1c2"},
        ],
    ]

    normalized = normalize_dense_table_grid(grid, locator="table 1")

    assert normalized["row_header_depth"] == 2
    assert normalized["row_records"][1]["row_path"] == ["Internal revenue", "Income Tax"]
    assert normalized["row_records"][1]["row_depth"] == 1


def test_normalize_flat_table_preserves_inline_indentation_as_hierarchy():
    normalized = normalize_flat_table(
        ["Category", "Amount"],
        [["Internal revenue", ""], ["  Income Tax", "100"], ["    Corporate Tax", "80"]],
        locator="table 2",
    )

    assert normalized["row_header_depth"] == 1
    assert normalized["row_records"][1]["row_path"] == ["Internal revenue", "Income Tax"]
    assert normalized["row_records"][2]["row_path"] == ["Internal revenue", "Income Tax", "Corporate Tax"]


def test_normalize_dense_table_grid_uses_header_alignment_when_numeric_rows_appear_late():
    grid = [
        [
            {"text": "", "is_header": True, "origin_id": "h0"},
            {"text": "", "is_header": True, "origin_id": "h1"},
            {"text": "Calendar year 1940", "is_header": True, "origin_id": "h2"},
        ]
    ]
    for idx in range(55):
        grid.append(
            [
                {"text": f"Descriptive intro row {idx}", "is_header": False, "origin_id": f"r{idx}c0"},
                {"text": "", "is_header": False, "origin_id": f"r{idx}c1"},
                {"text": "", "is_header": False, "origin_id": f"r{idx}c2"},
            ]
        )
    grid.append(
        [
            {"text": "", "is_header": False, "origin_id": "r_final_c0"},
            {"text": "U.S. national defense", "is_header": False, "origin_id": "r_final_c1"},
            {"text": "4,748", "is_header": False, "origin_id": "r_final_c2"},
        ]
    )

    normalized = normalize_dense_table_grid(grid, locator="table 3")

    assert normalized["row_header_depth"] == 2
    assert normalized["row_records"][-1]["row_path"][-1] == "U.S. national defense"
