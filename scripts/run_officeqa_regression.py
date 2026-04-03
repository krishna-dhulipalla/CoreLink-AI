"""Run the curated OfficeQA regression slice and emit a classified report."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_RUNS_ENDPOINTS", "")
os.environ.setdefault("BENCHMARK_NAME", "officeqa")
os.environ.setdefault("BENCHMARK_STATELESS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import build_agent_graph
from agent.benchmarks.officeqa_eval import build_case_report, load_regression_slice, summarize_regression_report
from agent.runner import run_agent_trace

DEFAULT_SLICE_PATH = ROOT / "eval" / "officeqa_regression_slice.json"
DEFAULT_OUTPUT_DIR = ROOT / "Results&traces"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the curated OfficeQA regression slice.")
    parser.add_argument("--slice-path", default=str(DEFAULT_SLICE_PATH), help="Path to the OfficeQA regression slice JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where the regression report JSON will be written.")
    parser.add_argument("--smoke", action="store_true", help="Run only smoke-tagged cases.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of cases to run after filtering.")
    return parser.parse_args()


def _selected_cases(cases: list[dict[str, Any]], smoke: bool, limit: int) -> list[dict[str, Any]]:
    selected = [item for item in cases if (not smoke or bool(item.get("smoke")))]
    if limit > 0:
        selected = selected[:limit]
    return selected


async def _run_case(graph: Any, case: dict[str, Any]) -> dict[str, Any]:
    overrides = {
        "benchmark_name": "officeqa",
        "benchmark_adapter": "officeqa",
        "source_files": list(case.get("source_files", [])),
    }
    trace = await run_agent_trace(
        graph,
        str(case.get("prompt", "") or ""),
        benchmark_overrides=overrides,
        trace_identity={
            "benchmark_uid": str(case.get("id", "") or ""),
            "task_id": str(case.get("id", "") or ""),
            "request_id": str(case.get("id", "") or ""),
        },
    )
    return build_case_report(case, trace)


async def _main() -> int:
    args = _parse_args()
    cases = _selected_cases(load_regression_slice(args.slice_path), smoke=args.smoke, limit=args.limit)
    if not cases:
        raise SystemExit("No OfficeQA regression cases selected.")

    graph = build_agent_graph()
    case_reports = []
    total = len(cases)
    print(json.dumps({"event": "officeqa_regression_start", "mode": "smoke" if args.smoke else "full", "case_count": total}, ensure_ascii=True), flush=True)
    for index, case in enumerate(cases, start=1):
        print(
            json.dumps(
                {
                    "event": "officeqa_case_start",
                    "index": index,
                    "total": total,
                    "id": str(case.get("id", "") or ""),
                    "focus_subsystem": str(case.get("focus_subsystem", "") or ""),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        report = await _run_case(graph, case)
        case_reports.append(report)
        print(
            json.dumps(
                {
                    "event": "officeqa_case_done",
                    "index": index,
                    "total": total,
                    "id": str(case.get("id", "") or ""),
                    "classification": str(dict(report.get("classification") or {}).get("subsystem", "") or ""),
                    "stop_reason": str(dict(report.get("quality_report") or {}).get("stop_reason", "") or ""),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    summary = summarize_regression_report(case_reports)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "smoke" if args.smoke else "full"
    output_path = output_dir / f"officeqa_regression_{mode}_{stamp}.json"
    payload = {
        "mode": mode,
        "slice_path": str(Path(args.slice_path).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "cases": case_reports,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"saved_path": str(output_path), "summary": summary}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
