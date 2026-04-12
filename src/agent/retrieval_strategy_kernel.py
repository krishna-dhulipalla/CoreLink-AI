from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from agent.contracts import ExecutionJournal, RetrievalAction, RetrievalIntent, RetrievalStrategy, SourceBundle, ToolPlan


@dataclass(frozen=True)
class RetrievalStrategyContext:
    execution_mode: str
    source_bundle: SourceBundle
    retrieval_intent: RetrievalIntent
    tool_plan: ToolPlan
    journal: ExecutionJournal
    registry: dict[str, dict[str, Any]]
    benchmark_overrides: dict[str, Any] | None = None
    requested_strategy: RetrievalStrategy = "table_first"


class RetrievalStrategyHandler(Protocol):
    name: RetrievalStrategy

    def plan_action(self, context: RetrievalStrategyContext) -> RetrievalAction:
        ...


@dataclass(frozen=True)
class FunctionRetrievalStrategyHandler:
    name: RetrievalStrategy
    planner: Callable[[RetrievalStrategyContext], RetrievalAction]

    def plan_action(self, context: RetrievalStrategyContext) -> RetrievalAction:
        return self.planner(context)


class RetrievalStrategyKernel:
    def __init__(self, handlers: list[RetrievalStrategyHandler]) -> None:
        self._handlers = {handler.name: handler for handler in handlers}
        if "table_first" not in self._handlers:
            raise ValueError("RetrievalStrategyKernel requires a table_first handler.")

    def _infer_strategy(self, retrieval_intent: RetrievalIntent) -> RetrievalStrategy:
        analysis_modes = {str(item or "").strip().lower() for item in retrieval_intent.analysis_modes or [] if str(item or "").strip()}
        aggregation_shape = str(retrieval_intent.aggregation_shape or "").strip().lower()
        period_type = str(retrieval_intent.period_type or "").strip().lower()
        has_constraints = bool(retrieval_intent.include_constraints or retrieval_intent.exclude_constraints)

        if aggregation_shape in {"cross_document", "cross_document_alignment", "document_comparison", "document_join"}:
            return "multi_document"
        if {"cross_document_alignment", "document_join", "cross_source_join"} & analysis_modes:
            return "multi_document"
        if aggregation_shape in {"series_aggregation", "multi_table", "table_join"}:
            return "multi_table"
        if period_type == "monthly_series" and {"series_aggregation", "monthly_rollup", "monthly_sum"} & analysis_modes:
            return "multi_table"
        if has_constraints or {"narrative_support", "text_support", "quote_grounding", "policy_context"} & analysis_modes:
            return "hybrid"
        if {"narrative_extraction", "text_lookup", "quote_lookup"} & analysis_modes:
            return "text_first"
        return "table_first"

    def strategy_chain(self, retrieval_intent: RetrievalIntent) -> list[RetrievalStrategy]:
        inferred = self._infer_strategy(retrieval_intent)
        preferred = retrieval_intent.strategy or inferred
        if preferred == "table_first" and inferred != "table_first":
            preferred = inferred
        ordered = [preferred, inferred, *list(retrieval_intent.fallback_chain or [])]
        deduped: list[RetrievalStrategy] = []
        for item in ordered:
            if item and item not in deduped:
                deduped.append(item)
        return deduped or ["table_first"]

    def select_strategy(self, retrieval_intent: RetrievalIntent) -> RetrievalStrategy:
        return self.strategy_chain(retrieval_intent)[0]

    def plan_action(self, context: RetrievalStrategyContext) -> RetrievalAction:
        strategy = context.requested_strategy or self.select_strategy(context.retrieval_intent)
        handler = self._handlers.get(strategy) or self._handlers["table_first"]
        action = handler.plan_action(context)
        if not action.requested_strategy:
            action.requested_strategy = strategy
        if not action.strategy:
            action.strategy = strategy
        return action
