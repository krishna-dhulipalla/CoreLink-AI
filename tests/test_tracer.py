import asyncio
import os

import agent.tracer as tracer_module
from agent.tracer import finalize_tracer, get_tracer, start_tracer


def test_tracer_is_request_scoped_under_concurrent_tasks(monkeypatch):
    monkeypatch.setenv("ENABLE_RUN_TRACER", "1")
    monkeypatch.setenv("BENCHMARK_STATELESS", "1")
    captured: list[tuple[str, dict]] = []

    def _capture(payload, profile, trace_identity):
        captured.append((str(trace_identity.get("request_id", "")), payload))
        return "ok"

    monkeypatch.setattr("agent.tracer._write_trace_file", _capture)

    async def _run(request_id: str, question: str) -> None:
        tracer = start_tracer({"request_id": request_id, "task_id": request_id, "benchmark_uid": request_id})
        assert tracer is not None
        active = get_tracer()
        assert active is tracer
        active.set_task(question)
        await asyncio.sleep(0)
        active.record(
            "executor",
            {
                "intent": {
                    "task_family": "document_qa",
                    "execution_mode": "document_grounded_analysis",
                    "complexity_tier": "structured_analysis",
                },
                "used_llm": False,
                "tools_ran": ["search_reference_corpus"],
                "output_preview": "",
            },
        )
        finalize_tracer(f"final answer for {request_id}")

    async def _main() -> None:
        await asyncio.gather(
            _run("req-a", "Question A"),
            _run("req-b", "Question B"),
        )

    asyncio.run(_main())

    assert len(captured) == 2
    payloads = {request_id: payload for request_id, payload in captured}
    assert payloads["req-a"]["request_id"] == "req-a"
    assert payloads["req-b"]["request_id"] == "req-b"
    assert payloads["req-a"]["task_preview"] == "Question A"
    assert payloads["req-b"]["task_preview"] == "Question B"


def test_tracer_evicts_old_trace_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_MAX_RECENT", "2")
    monkeypatch.setattr(tracer_module, "_TRACES_DIR", tmp_path)

    stale_a = tmp_path / "2026-03-01_00-00-00"
    stale_b = tmp_path / "2026-03-02_00-00-00"
    fresh = tmp_path / "2026-03-03_00-00-00"
    newest = tmp_path / "2026-03-04_00-00-00"
    for index, path in enumerate((stale_a, stale_b, fresh, newest), start=1):
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "task_001.json"
        marker.write_text("{}", encoding="utf-8")
        ts = 1000 + index
        os.utime(marker, (ts, ts))
        os.utime(path, (ts, ts))

    tracer_module._cleanup_old_traces()

    remaining = sorted(item.name for item in tmp_path.iterdir())
    assert remaining == ["2026-03-03_00-00-00", "2026-03-04_00-00-00"]


def test_tracer_preserves_structured_diagnostic_artifacts(monkeypatch):
    monkeypatch.setenv("ENABLE_RUN_TRACER", "1")
    captured: list[dict] = []

    def _capture(payload, profile, trace_identity):
        captured.append(payload)
        return "ok"

    monkeypatch.setattr("agent.tracer._write_trace_file", _capture)

    tracer = start_tracer({"request_id": "diag-1"})
    assert tracer is not None
    tracer.set_task("diagnostic task")
    tracer.record(
        "executor",
        {
            "intent": {"task_family": "document_qa", "execution_mode": "document_grounded_analysis", "complexity_tier": "structured_analysis"},
            "used_llm": False,
            "llm_decision_reason": "deterministic_compute_completed",
            "llm_repair_history": [
                {
                    "stage": "retrieval_repair",
                    "trigger": "wrong document",
                    "path_changed": True,
                    "decision": {"decision": "rewrite_query", "confidence": 0.88},
                }
            ],
            "officeqa_llm_usage": [
                {
                    "category": "semantic_plan_llm",
                    "reason": "semantic_plan_llm",
                    "model_name": "semantic-plan-model",
                    "applied": True,
                }
            ],
            "tools_ran": ["fetch_officeqa_table"],
            "retrieval_decision": {"tool_name": "fetch_officeqa_table", "strategy": "table_first"},
            "strategy_reason": "primary metric is expected to be recoverable from structured table evidence",
            "candidate_sources": [{"document_id": "treasury_1940_json", "score": 1.0}],
            "rejected_candidates": [{"document_id": "treasury_1939_json", "reason": "lower-ranked than the selected candidates"}],
            "aggregation_reason": "Selected monthly-sum compute because the task asks for a within-year monthly aggregation.",
            "evidence_gaps": ["missing month coverage"],
            "tool_results": [
                {
                    "tool": "fetch_officeqa_table",
                    "facts": {
                        "document_id": "treasury_1940_json",
                        "citation": "treasury_1940.json#page=17",
                        "metadata": {"officeqa_status": "ok"},
                        "tables": [{"locator": "table 1"}],
                    },
                }
            ],
            "output_preview": "",
        },
    )
    finalize_tracer("answer")

    assert captured
    node = captured[0]["nodes"][0]
    assert node["retrieval_decision"]["tool_name"] == "fetch_officeqa_table"
    assert node["strategy_reason"]
    assert node["candidate_sources"]
    assert node["rejected_candidates"]
    assert node["aggregation_reason"]
    assert node["evidence_gaps"] == ["missing month coverage"]
    execution_summary = captured[0]["execution_summary"]
    assert execution_summary
    assert execution_summary[0]["llm_decision_reason"] == "deterministic_compute_completed"
    assert execution_summary[0]["llm_repair"]["count"] == 1
    assert execution_summary[0]["llm_repair"]["decision"] == "rewrite_query"
    assert execution_summary[0]["llm_usage"][0]["category"] == "semantic_plan_llm"
    assert execution_summary[0]["retrieval"]["tool_name"] == "fetch_officeqa_table"
    assert execution_summary[0]["evidence_gaps"] == ["missing month coverage"]
    assert execution_summary[0]["tool_results"][0]["tool"] == "fetch_officeqa_table"


def test_tracer_execution_summary_compacts_candidate_lists(monkeypatch):
    monkeypatch.setenv("ENABLE_RUN_TRACER", "1")
    captured: list[dict] = []

    def _capture(payload, profile, trace_identity):
        captured.append(payload)
        return "ok"

    monkeypatch.setattr("agent.tracer._write_trace_file", _capture)

    tracer = start_tracer({"request_id": "diag-compact"})
    assert tracer is not None
    tracer.set_task("compact diagnostic task")
    tracer.record(
        "executor",
        {
            "intent": {"task_family": "document_qa", "execution_mode": "document_grounded_analysis", "complexity_tier": "structured_analysis"},
            "used_llm": False,
            "tools_ran": ["search_officeqa_documents"],
            "retrieval_decision": {"tool_name": "search_officeqa_documents", "strategy": "table_first", "document_id": "treasury_1945_json"},
            "candidate_sources": [
                {"title": "Treasury Bulletin Jan 1945", "document_id": "treasury_1945_json", "score": 0.98},
                {"title": "Treasury Bulletin Feb 1945", "document_id": "treasury_1945_feb_json", "score": 0.87},
            ],
            "rejected_candidates": [{"document_id": "treasury_1944_json", "reason": "year mismatch"}],
            "tool_results": [],
            "output_preview": "",
        },
    )
    finalize_tracer("answer")

    execution_summary = captured[0]["execution_summary"]
    assert execution_summary
    assert execution_summary[0]["candidate_source_count"] == 2
    assert execution_summary[0]["rejected_candidate_count"] == 1
    assert execution_summary[0]["top_candidate"]["document_id"] == "treasury_1945_json"
    assert "candidate_sources" not in execution_summary[0]


def test_tracer_prefers_cost_tracker_counts_for_llm_and_tool_totals(monkeypatch):
    monkeypatch.setenv("ENABLE_RUN_TRACER", "1")
    captured: list[dict] = []

    def _capture(payload, profile, trace_identity):
        captured.append(payload)
        return "ok"

    monkeypatch.setattr("agent.tracer._write_trace_file", _capture)

    tracer = start_tracer({"request_id": "count-sync"})
    assert tracer is not None
    tracer.set_task("count sync task")
    tracer.record(
        "executor",
        {
            "used_llm": True,
            "tools_ran": ["search_officeqa_documents", "fetch_officeqa_table"],
            "output_preview": "",
        },
    )
    finalize_tracer(
        "answer",
        cost_summary={"llm_calls": 1, "mcp_calls": 2},
        budget_summary={},
    )

    assert captured[0]["total_llm_calls"] == 1
    assert captured[0]["total_tool_calls"] == 2
