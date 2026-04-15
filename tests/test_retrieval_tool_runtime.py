import asyncio

from engine.agent.contracts import RetrievalAction, RetrievalIntent, SourceBundle
from engine.agent.retrieval_tool_runtime import tool_args_from_retrieval_action


def test_tool_args_from_retrieval_action_builds_officeqa_table_fetch_args():
    action = RetrievalAction(
        tool_name="fetch_officeqa_table",
        document_id="doc-1",
        path="path/to/doc",
        query="public debt outstanding 1945",
        row_limit=10,
    )
    source_bundle = SourceBundle(
        task_text="What was public debt outstanding in 1945?",
        focus_query="public debt outstanding 1945",
        target_period="1945",
        entities=["Public debt"],
    )
    retrieval_intent = RetrievalIntent(
        entity="Public debt",
        metric="public debt outstanding",
        period="1945",
        granularity_requirement="point_lookup",
    )

    args = tool_args_from_retrieval_action(
        action,
        source_bundle,
        {},
        retrieval_intent,
        lambda bundle, intent: bundle.focus_query,
        lambda intent, bundle: "fallback table query",
        lambda intent, bundle: "fallback row query",
        lambda intent: "fallback column query",
    )

    assert args["document_id"] == "doc-1"
    assert args["table_query"] == "public debt outstanding 1945"
    assert args["row_limit"] == 50


def test_run_tool_step_with_args_filters_unknown_fields():
    class _FakeTool:
        args = {"query": {"type": "string"}}

        async def ainvoke(self, args):
            return {"ok": True, "args": args}

    async def _run():
        from engine.agent.retrieval_tool_runtime import run_tool_step_with_args

        args, result = await run_tool_step_with_args(
            {"search_reference_corpus": {"tool": _FakeTool(), "descriptor": {}}},
            "search_reference_corpus",
            {"query": "hello", "junk": "ignored"},
        )
        return args, result

    args, result = asyncio.run(_run())

    assert args == {"query": "hello"}
    assert result.facts["ok"] is True
