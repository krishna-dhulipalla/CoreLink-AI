from engine.agent.contracts import ExecutionJournal, RetrievalAction, RetrievalIntent, SourceBundle, ToolPlan
from engine.agent.retrieval_strategy_kernel import FunctionRetrievalStrategyHandler, RetrievalStrategyContext, RetrievalStrategyKernel


def test_retrieval_strategy_kernel_dispatches_to_requested_strategy():
    calls: list[str] = []

    def _planner(context: RetrievalStrategyContext) -> RetrievalAction:
        calls.append(context.requested_strategy)
        return RetrievalAction(action="tool", tool_name="search_officeqa_documents", requested_strategy=context.requested_strategy)

    kernel = RetrievalStrategyKernel(
        [
            FunctionRetrievalStrategyHandler(name="table_first", planner=_planner),
            FunctionRetrievalStrategyHandler(name="text_first", planner=_planner),
            FunctionRetrievalStrategyHandler(name="hybrid", planner=_planner),
            FunctionRetrievalStrategyHandler(name="multi_table", planner=_planner),
            FunctionRetrievalStrategyHandler(name="multi_document", planner=_planner),
        ]
    )
    retrieval_intent = RetrievalIntent(strategy="hybrid", fallback_chain=["text_first", "table_first"])
    context = RetrievalStrategyContext(
        execution_mode="document_grounded_analysis",
        source_bundle=SourceBundle(task_text="query"),
        retrieval_intent=retrieval_intent,
        tool_plan=ToolPlan(),
        journal=ExecutionJournal(),
        registry={},
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        requested_strategy="hybrid",
    )

    action = kernel.plan_action(context)

    assert calls == ["hybrid"]
    assert action.requested_strategy == "hybrid"
    assert action.strategy == "hybrid"


def test_retrieval_strategy_kernel_uses_primary_strategy_from_intent():
    kernel = RetrievalStrategyKernel(
        [
            FunctionRetrievalStrategyHandler(
                name="table_first",
                planner=lambda context: RetrievalAction(action="tool", tool_name="fetch_officeqa_table"),
            )
        ]
    )

    retrieval_intent = RetrievalIntent(strategy="table_first", fallback_chain=["hybrid", "text_first"])

    assert kernel.select_strategy(retrieval_intent) == "table_first"


def test_retrieval_strategy_kernel_infers_multi_document_from_semantic_shape():
    kernel = RetrievalStrategyKernel(
        [
            FunctionRetrievalStrategyHandler(name="table_first", planner=lambda context: RetrievalAction(action="tool", tool_name="fetch_officeqa_table")),
            FunctionRetrievalStrategyHandler(name="multi_document", planner=lambda context: RetrievalAction(action="tool", tool_name="search_officeqa_documents")),
        ]
    )

    retrieval_intent = RetrievalIntent(
        strategy="table_first",
        aggregation_shape="cross_document_alignment",
        analysis_modes=["cross_document_alignment"],
    )

    assert kernel.select_strategy(retrieval_intent) == "multi_document"


def test_retrieval_strategy_kernel_infers_hybrid_from_constraints():
    kernel = RetrievalStrategyKernel(
        [
            FunctionRetrievalStrategyHandler(name="table_first", planner=lambda context: RetrievalAction(action="tool", tool_name="fetch_officeqa_table")),
            FunctionRetrievalStrategyHandler(name="hybrid", planner=lambda context: RetrievalAction(action="tool", tool_name="search_officeqa_documents")),
        ]
    )

    retrieval_intent = RetrievalIntent(
        strategy="table_first",
        include_constraints=["include public works"],
        exclude_constraints=["exclude revolving funds"],
    )

    assert kernel.select_strategy(retrieval_intent) == "hybrid"


def test_retrieval_strategy_kernel_returns_next_untried_strategy_in_admissible_order():
    kernel = RetrievalStrategyKernel(
        [
            FunctionRetrievalStrategyHandler(name="table_first", planner=lambda context: RetrievalAction(action="tool", tool_name="fetch_officeqa_table")),
            FunctionRetrievalStrategyHandler(name="hybrid", planner=lambda context: RetrievalAction(action="tool", tool_name="search_officeqa_documents")),
            FunctionRetrievalStrategyHandler(name="multi_table", planner=lambda context: RetrievalAction(action="tool", tool_name="fetch_officeqa_table")),
            FunctionRetrievalStrategyHandler(name="text_first", planner=lambda context: RetrievalAction(action="tool", tool_name="fetch_officeqa_pages")),
            FunctionRetrievalStrategyHandler(name="multi_document", planner=lambda context: RetrievalAction(action="tool", tool_name="search_officeqa_documents")),
        ]
    )

    retrieval_intent = RetrievalIntent(
        strategy="table_first",
        fallback_chain=["multi_table", "hybrid"],
    )

    assert kernel.admissible_strategies(retrieval_intent)[:3] == ["table_first", "multi_table", "hybrid"]
    assert kernel.next_strategy(retrieval_intent, {"table_first"}) == "multi_table"
