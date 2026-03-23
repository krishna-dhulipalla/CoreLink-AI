import asyncio

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
