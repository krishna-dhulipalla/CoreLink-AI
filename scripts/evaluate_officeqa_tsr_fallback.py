from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.tools.tsr_fallback import compare_dense_table_normalizers, compare_flat_table_normalizers

DEFAULT_FIXTURE_PATH = ROOT / "eval" / "officeqa_tsr_fixture_set.json"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the experimental OfficeQA TSR fallback on fixture tables.")
    parser.add_argument("--fixture-path", default=str(DEFAULT_FIXTURE_PATH), help="Path to the OfficeQA TSR fixture set JSON.")
    return parser.parse_args()


def _column_path_depth(candidate: dict[str, Any]) -> int:
    column_paths = list(candidate.get("column_paths", []) or [])
    return max((len([part for part in path if str(part).strip()]) for path in column_paths), default=0)


def _numeric_cell_count(candidate: dict[str, Any]) -> int:
    return sum(
        1
        for row in list(candidate.get("row_records", []) or [])
        for cell in list(row.get("cells", []) or [])
        if cell.get("is_numeric")
    )


def _evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    fixture_type = str(fixture.get("type", "") or "dense_grid")
    locator = str(fixture.get("locator", "") or "")
    unit_hint = str(fixture.get("unit_hint", "") or "")
    if fixture_type == "flat_table":
        comparison = compare_flat_table_normalizers(
            list(fixture.get("headers", []) or []),
            list(fixture.get("rows", []) or []),
            locator=locator,
            unit_hint=unit_hint,
        )
    else:
        comparison = compare_dense_table_normalizers(
            list(fixture.get("grid", []) or []),
            locator=locator,
            unit_hint=unit_hint,
        )
    selected = dict(comparison.get("selected") or {})
    diagnostics = dict(comparison.get("diagnostics") or {})
    expectations = dict(fixture.get("expectations") or {})
    preferred_mode = str(expectations.get("preferred_mode", "") or "")
    min_column_path_depth = int(expectations.get("min_column_path_depth", 0) or 0)
    min_numeric_cells = int(expectations.get("min_numeric_cells", 0) or 0)
    selected_mode = str(diagnostics.get("selected_mode", "") or "default")
    result = {
        "id": str(fixture.get("id", "") or ""),
        "selected_mode": selected_mode,
        "default_score": float(diagnostics.get("default_score", 0.0) or 0.0),
        "fallback_score": float(diagnostics.get("fallback_score", 0.0) or 0.0),
        "score_delta": float(diagnostics.get("score_delta", 0.0) or 0.0),
        "column_path_depth": _column_path_depth(selected),
        "numeric_cell_count": _numeric_cell_count(selected),
    }
    checks = {
        "preferred_mode": (not preferred_mode) or selected_mode == preferred_mode,
        "column_path_depth": result["column_path_depth"] >= min_column_path_depth,
        "numeric_cell_count": result["numeric_cell_count"] >= min_numeric_cells,
    }
    result["checks"] = checks
    result["pass"] = all(checks.values())
    return result


def main() -> int:
    args = _args()
    fixtures = json.loads(Path(args.fixture_path).read_text(encoding="utf-8"))
    reports = [_evaluate_fixture(dict(item)) for item in fixtures if isinstance(item, dict)]
    total = len(reports)
    passing = sum(1 for item in reports if item["pass"])
    fallback_wins = sum(1 for item in reports if item["selected_mode"] == "tsr_split_merge_heuristic")
    avg_delta = round(sum(float(item["score_delta"]) for item in reports) / max(1, total), 4)
    recommendation = (
        "keep_optional"
        if fallback_wins < max(1, total // 2) or avg_delta < 0.05
        else "candidate_for_promotion"
    )
    payload = {
        "fixture_count": total,
        "passing_fixtures": passing,
        "fallback_wins": fallback_wins,
        "avg_score_delta": avg_delta,
        "recommendation": recommendation,
        "reports": reports,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
