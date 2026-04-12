import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path

import agent.graph as graph_module
import agent.benchmarks as benchmark_module
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent.budget import BudgetTracker
from agent.benchmarks import (
    benchmark_build_structured_evidence,
    benchmark_compute_result,
    benchmark_validate_final,
    register_benchmark_document_adapter,
)
from agent.benchmarks.officeqa_validator import validate_officeqa_final
from agent.benchmarks.officeqa_index import build_officeqa_index
from agent.contracts import EvidenceSufficiency, ExecutionJournal, OfficeQAComputeResult, OfficeQALLMRepairDecision, OfficeQASourceRerankDecision, OfficeQAValidationResult, ProgressSignature, RetrievalAction, RetrievalIntent, SourceBundle, TaskIntent, ToolPlan
from agent.officeqa_structured_evidence import build_officeqa_structured_evidence
from agent.retrieval_tools import fetch_corpus_document as fetch_corpus_document_tool
from agent.retrieval_tools import fetch_officeqa_table, lookup_officeqa_cells, search_reference_corpus
from agent.retrieval_reasoning import assess_evidence_sufficiency, build_retrieval_intent, predictive_evidence_gaps
from agent.tools.normalization import normalize_tool_output
from agent.nodes.intake import intake
from agent.nodes.output_adapter import output_adapter
from agent.tracer import RunTracer
from agent.capabilities import (
    BUILTIN_LEGAL_TOOLS,
    BUILTIN_RETRIEVAL_TOOLS,
    build_capability_registry,
    filter_registry_for_benchmark,
    resolve_tool_plan,
)
from agent.curated_context import attach_structured_evidence, build_curated_context, build_source_bundle, solver_context_block
from agent.nodes.orchestrator_retrieval import _plan_retrieval_action, _search_result_candidates
from agent.nodes.orchestrator_retrieval import _tool_args_from_retrieval_action
from agent.nodes.orchestrator import (
    _MAX_RETRIEVAL_HOPS,
    _apply_officeqa_llm_repair_decision,
    _record_retrieval_strategy_attempt,
    _officeqa_retrieval_input_signature,
    context_curator,
    fast_path_gate,
    make_capability_resolver,
    make_executor,
    reviewer,
    route_from_reviewer,
    self_reflection,
    task_planner,
    _rank_search_candidates,
)
from test_utils import make_state
from tools import CALCULATOR_TOOL, SEARCH_TOOL


class _FakeModel:
    def __init__(self, response: AIMessage, captured: list | None = None):
        self._response = response
        self._captured = captured if captured is not None else []

    def invoke(self, messages):
        self._captured.append(messages)
        return self._response


def test_build_agent_graph_uses_active_runtime(monkeypatch):
    monkeypatch.setattr("agent.graph.build_agent_graph", lambda external_tools=None: "engine_graph")

    assert graph_module.build_agent_graph() == "engine_graph"


def test_build_source_bundle_dedupes_benchmark_source_files():
    source_bundle = build_source_bundle(
        "According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
        {
            "source_files_expected": [
                "treasury_bulletin_1945_01.json",
                "TREASURY_BULLETIN_1945_01.JSON",
                "treasury_bulletin_1945_02.json",
            ],
            "source_files_found": [
                {"document_id": "treasury_bulletin_1945_01_json", "relative_path": "treasury_bulletin_1945_01.json"},
                {"document_id": "TREASURY_BULLETIN_1945_01_JSON", "relative_path": "TREASURY_BULLETIN_1945_01.JSON"},
                {"document_id": "treasury_bulletin_1945_02_json", "relative_path": "treasury_bulletin_1945_02.json"},
            ],
        },
    )

    assert source_bundle.source_files_expected == [
        "treasury_bulletin_1945_01.json",
        "treasury_bulletin_1945_02.json",
    ]
    assert source_bundle.source_files_found == [
        {"document_id": "treasury_bulletin_1945_01_json", "relative_path": "treasury_bulletin_1945_01.json"},
        {"document_id": "treasury_bulletin_1945_02_json", "relative_path": "treasury_bulletin_1945_02.json"},
    ]


def test_search_result_candidates_dedupes_documents_and_results_views():
    tool_result = {
        "facts": {
            "documents": [
                {
                    "document_id": "treasury_bulletin_1945_01_json",
                    "citation": "treasury_bulletin_1945_01.json",
                    "path": "treasury_bulletin_1945_01.json",
                    "title": "",
                    "rank": 999,
                }
            ],
            "results": [
                {
                    "document_id": "treasury_bulletin_1945_01_json",
                    "url": "treasury_bulletin_1945_01.json",
                    "title": "treasury_bulletin_1945_01.json",
                    "snippet": "Treasury Bulletin table for public debt outstanding.",
                    "rank": 1,
                }
            ],
        }
    }

    candidates = _search_result_candidates(tool_result)

    assert len(candidates) == 1
    assert candidates[0]["document_id"] == "treasury_bulletin_1945_01_json"
    assert candidates[0]["citation"] == "treasury_bulletin_1945_01.json"
    assert candidates[0]["title"] == "treasury_bulletin_1945_01.json"
    assert candidates[0]["rank"] == 1


def test_search_ranking_prefers_publication_year_match_over_historical_mentions():
    retrieval_intent = RetrievalIntent(
        entity="Veterans Administration",
        metric="total expenditures",
        period="1934",
        granularity_requirement="fiscal_year",
        document_family="official_government_finance",
        aggregation_shape="fiscal_year_total",
    )
    source_bundle = SourceBundle(
        task_text="What were the total expenditures of the Veterans Administration in FY 1934?",
        focus_query="Veterans Administration total expenditures fiscal year 1934",
        target_period="1934",
        entities=["Veterans Administration"],
    )
    candidates = [
        {
            "document_id": "treasury_bulletin_1974_02_json",
            "citation": "treasury_bulletin_1974_02.json",
            "path": "treasury_bulletin_1974_02.json",
            "title": "Treasury Bulletin 1974-02",
            "snippet": "Historical comparison table mentioning Veterans Administration fiscal year 1934.",
            "rank": 1,
            "score": 1.0,
            "metadata": {"years": ["1934", "1974"], "table_headers": ["Veterans Administration"]},
        },
        {
            "document_id": "treasury_bulletin_1934_06_json",
            "citation": "treasury_bulletin_1934_06.json",
            "path": "treasury_bulletin_1934_06.json",
            "title": "Treasury Bulletin 1934-06",
            "snippet": "Veterans Administration expenditures in fiscal year 1934.",
            "rank": 2,
            "score": 0.8,
            "metadata": {"years": ["1934"], "table_headers": ["Veterans Administration", "Expenditures"]},
        },
    ]

    ranked = _rank_search_candidates(candidates, retrieval_intent, source_bundle, {"benchmark_adapter": "officeqa"})

    assert ranked[0]["document_id"] == "treasury_bulletin_1934_06_json"


def test_search_ranking_can_prefer_next_year_publication_for_calendar_year_summary_questions():
    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        period_type="calendar_year",
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1941", "1940", "1939"],
        granularity_requirement="calendar_year",
        document_family="official_government_finance",
        aggregation_shape="calendar_year_total",
    )
    source_bundle = SourceBundle(
        task_text="What were the total expenditures for U.S. national defense in the calendar year 1940?",
        focus_query="U.S. national defense total expenditures calendar year 1940",
        target_period="1940",
        entities=["U.S. national defense"],
    )
    candidates = [
        {
            "document_id": "treasury_bulletin_1940_01_json",
            "citation": "treasury_bulletin_1940_01.json",
            "path": "treasury_bulletin_1940_01.json",
            "title": "Treasury Bulletin 1940-01",
            "snippet": "Current monthly Treasury statement.",
            "rank": 1,
            "score": 0.95,
            "metadata": {
                "publication_year": "1940",
                "years": ["1940"],
                "month_coverage": ["january", "february"],
                "best_evidence_unit": {
                    "table_family": "monthly_series",
                    "period_type": "monthly_series",
                    "headers": ["Month", "Receipts"],
                    "row_labels": ["January", "February"],
                    "year_refs": ["1940"],
                    "table_confidence": 0.82,
                },
            },
        },
        {
            "document_id": "treasury_bulletin_1941_11_json",
            "citation": "treasury_bulletin_1941_11.json",
            "path": "treasury_bulletin_1941_11.json",
            "title": "Treasury Bulletin 1941-11",
            "snippet": "Summary of expenditures for calendar year 1940.",
            "rank": 2,
            "score": 0.9,
            "metadata": {
                "publication_year": "1941",
                "years": ["1940", "1941"],
                "best_evidence_unit": {
                    "table_family": "category_breakdown",
                    "period_type": "calendar_year",
                    "headers": ["Category", "Calendar year 1940"],
                    "row_labels": ["U.S. national defense"],
                    "year_refs": ["1940"],
                    "table_confidence": 0.88,
                },
            },
        },
    ]

    ranked = _rank_search_candidates(candidates, retrieval_intent, source_bundle, {"benchmark_adapter": "officeqa"})

    assert ranked[0]["document_id"] == "treasury_bulletin_1941_11_json"


def test_plan_retrieval_action_records_requested_strategy_from_kernel():
    registry = build_capability_registry(BUILTIN_RETRIEVAL_TOOLS)
    tool_plan = ToolPlan(selected_tools=["search_officeqa_documents"])
    source_bundle = SourceBundle(
        task_text="What were the total expenditures for U.S. national defense in the calendar year 1940?",
        focus_query="U.S. national defense total expenditures calendar year 1940",
        target_period="1940",
        entities=["U.S. national defense"],
    )
    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        strategy="text_first",
        fallback_chain=["hybrid", "table_first"],
        source_constraint_policy="off",
    )

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=ExecutionJournal(),
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    assert action.requested_strategy == "text_first"
    assert action.strategy in {"text_first", "hybrid"}


def test_record_retrieval_strategy_attempt_persists_requested_and_applied_strategy():
    workpad = {}
    journal = ExecutionJournal(retrieval_iterations=2)
    action = RetrievalAction(
        action="tool",
        requested_strategy="text_first",
        strategy="hybrid",
        stage="locate_pages",
        tool_name="fetch_officeqa_pages",
        evidence_gap="narrative support",
        strategy_reason="hybrid strategy selected because narrative support is likely necessary",
        regime_change="strategy_rotation",
        query="national defense 1940",
        document_id="treasury_bulletin_1941_11_json",
        candidate_sources=[{"document_id": "treasury_bulletin_1941_11_json"}],
        material_input_signature="sig-123",
        no_material_change=True,
        exhaustion_proof={"strategies_exhausted": False, "untried_strategies": ["multi_table"]},
    )

    updated = _record_retrieval_strategy_attempt(workpad, journal, action)

    latest = dict(updated.get("latest_retrieval_strategy_attempt") or {})
    assert latest["iteration"] == 3
    assert latest["requested_strategy"] == "text_first"
    assert latest["applied_strategy"] == "hybrid"
    assert latest["tool_name"] == "fetch_officeqa_pages"
    assert latest["regime_change"] == "strategy_rotation"
    assert latest["material_input_signature"] == "sig-123"
    assert latest["no_material_change"] is True
    assert updated["officeqa_strategy_exhaustion_proof"]["strategies_exhausted"] is False


def test_plan_retrieval_action_rotates_to_next_strategy_after_validator_retry_request():
    registry = build_capability_registry(BUILTIN_RETRIEVAL_TOOLS)
    tool_plan = ToolPlan(selected_tools=["search_officeqa_documents", "fetch_officeqa_table", "fetch_officeqa_pages"])
    source_bundle = SourceBundle(
        task_text="What were the total expenditures for U.S. national defense in the calendar year 1940?",
        focus_query="U.S. national defense total expenditures calendar year 1940",
        target_period="1940",
        entities=["U.S. national defense"],
    )
    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        strategy="table_first",
        fallback_chain=["hybrid", "text_first"],
        source_constraint_policy="off",
    )

    first_action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=ExecutionJournal(),
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        workpad={},
    )

    rotated = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=ExecutionJournal(),
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        workpad={
            "officeqa_retry_policy": {"retry_allowed": True, "recommended_repair_target": "gather"},
            "retrieval_strategy_attempts": [
                {
                    "applied_strategy": first_action.strategy,
                    "requested_strategy": first_action.requested_strategy,
                    "material_input_signature": first_action.exhaustion_proof["material_input_signature"],
                }
            ],
        },
    )

    assert rotated.requested_strategy == "hybrid"
    assert rotated.regime_change == "strategy_rotation"
    assert rotated.no_material_change is True


def test_plan_retrieval_action_emits_exhaustion_proof_when_all_strategies_repeat_same_regime():
    registry = build_capability_registry(BUILTIN_RETRIEVAL_TOOLS)
    tool_plan = ToolPlan(selected_tools=["search_officeqa_documents", "fetch_officeqa_table", "fetch_officeqa_pages"])
    source_bundle = SourceBundle(
        task_text="What were the total expenditures for U.S. national defense in the calendar year 1940?",
        focus_query="U.S. national defense total expenditures calendar year 1940",
        target_period="1940",
        entities=["U.S. national defense"],
    )
    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        strategy="table_first",
        fallback_chain=["hybrid", "text_first"],
        source_constraint_policy="off",
    )

    first_action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=ExecutionJournal(),
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        workpad={},
    )
    duplicate_signature = first_action.exhaustion_proof["material_input_signature"]
    exhausted = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=ExecutionJournal(retrieval_iterations=2),
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        workpad={
            "officeqa_retry_policy": {"retry_allowed": True, "recommended_repair_target": "gather"},
            "officeqa_repair_failures": [{"code": "repair_applied_but_no_new_evidence"}],
            "retrieval_strategy_attempts": [
                {"applied_strategy": "table_first", "material_input_signature": duplicate_signature},
                {"applied_strategy": "hybrid", "material_input_signature": duplicate_signature},
                {"applied_strategy": "text_first", "material_input_signature": duplicate_signature},
            ],
        },
    )

    assert exhausted.action == "answer"
    assert exhausted.regime_change == "strategy_exhausted"
    assert exhausted.exhaustion_proof["strategies_exhausted"] is True
    assert exhausted.exhaustion_proof["benchmark_terminal_allowed"] is True


def test_search_ranking_ignores_generic_domain_tokens_and_prefers_entity_focused_table_family():
    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        period_type="calendar_year",
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1940", "1941"],
        granularity_requirement="calendar_year",
        document_family="official_government_finance",
        aggregation_shape="calendar_year_total",
        must_include_terms=["official government finance"],
    )
    source_bundle = SourceBundle(
        task_text="What were the total expenditures for U.S. national defense in the calendar year 1940?",
        focus_query="official government finance U.S. national defense total expenditures 1940 calendar year",
        target_period="1940",
        entities=["U.S. national defense"],
    )
    candidates = [
        {
            "document_id": "treasury_bulletin_1940_08_json",
            "citation": "treasury_bulletin_1940_08.json",
            "path": "treasury_bulletin_1940_08.json",
            "title": "Treasury Bulletin 1940-08",
            "snippet": "Summary table on receipts, expenditures and public debt.",
            "rank": 2,
            "score": 9.676,
            "metadata": {
                "publication_year": "1940",
                "years": ["1940"],
                "best_evidence_unit": {
                    "locator": "Summary Table on Receipts, Expenditures and Public Debt | Receipts and Expenditures",
                    "table_family": "debt_or_balance_sheet",
                    "period_type": "calendar_year",
                    "headers": ["Actual 1939", "Actual 1940", "Estimated 1941"],
                    "row_labels": ["Income Tax", "National defense and Veterans Adm", "Total Expenditures"],
                    "year_refs": ["1940", "1941"],
                    "table_confidence": 0.91,
                },
            },
        },
        {
            "document_id": "treasury_bulletin_1942_04_json",
            "citation": "treasury_bulletin_1942_04.json",
            "path": "treasury_bulletin_1942_04.json",
            "title": "Treasury Bulletin 1942-04",
            "snippet": "Table 4.- Analysis of National Defense Expenditures.",
            "rank": 1,
            "score": 10.268,
            "metadata": {
                "publication_year": "1942",
                "years": ["1940", "1941", "1942"],
                "best_evidence_unit": {
                    "locator": "Table 1.- Summary by Major Classifications | Table 4.- Analysis of National Defense Expenditures",
                    "table_family": "category_breakdown",
                    "period_type": "calendar_year",
                    "headers": ["1939", "1940", "1941"],
                    "row_labels": ["Total", "War Department", "Navy Department", "Total miscellaneous national defense"],
                    "heading_chain": ["Table 1.- Summary by Major Classifications", "Table 4.- Analysis of National Defense Expenditures"],
                    "year_refs": ["1940", "1941"],
                    "table_confidence": 0.85,
                },
            },
        },
    ]

    ranked = _rank_search_candidates(candidates, retrieval_intent, source_bundle, {"benchmark_adapter": "officeqa"})

    assert ranked[0]["document_id"] == "treasury_bulletin_1942_04_json"


def test_officeqa_soft_source_hints_trigger_search_first_and_keep_full_source_list():
    source_files = [f"treasury_bulletin_1940_{month:02d}.json" for month in range(1, 13)]
    source_bundle = SourceBundle(
        task_text="What were the total expenditures for U.S. national defense in the calendar year 1940?",
        focus_query="U.S. national defense total expenditures calendar year 1940",
        target_period="1940",
        entities=["U.S. national defense"],
        source_files_expected=source_files,
        source_files_found=[
            {"document_id": name.replace(".json", "_json"), "relative_path": name}
            for name in source_files
        ],
    )
    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        period_type="calendar_year",
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1941", "1940", "1939"],
        source_constraint_policy="soft",
        granularity_requirement="calendar_year",
        document_family="treasury_bulletin",
        aggregation_shape="calendar_year_total",
        strategy="table_first",
    )
    tool_plan = ToolPlan(selected_tools=["search_officeqa_documents", "fetch_officeqa_table"])
    registry = build_capability_registry(BUILTIN_RETRIEVAL_TOOLS)

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=ExecutionJournal(),
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )
    args = _tool_args_from_retrieval_action(action, source_bundle, registry, retrieval_intent)

    assert action.tool_name == "search_officeqa_documents"
    assert args["source_files_policy"] == "soft"
    assert args["source_files"] == source_files
    assert "national defense" in args["query"].lower()
    assert "treasury_bulletin_1940_01.json" not in args["query"].lower()


def test_officeqa_repair_can_widen_search_pool_without_task_specific_rules():
    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        period_type="calendar_year",
        target_years=["1940"],
        publication_year_window=["1940"],
        preferred_publication_years=["1940"],
        source_constraint_policy="soft",
        granularity_requirement="calendar_year",
        document_family="treasury_bulletin",
        aggregation_shape="calendar_year_total",
        strategy="table_first",
    )

    updated_intent, updated_workpad, changed, reroute_action = _apply_officeqa_llm_repair_decision(
        retrieval_intent,
        {"officeqa_override_query": "stale", "officeqa_override_table_query": "stale table"},
        {"decision": "widen_search_pool"},
    )

    assert changed is True
    assert reroute_action == "search_pool_widening"
    assert updated_intent.source_constraint_policy == "off"
    assert updated_intent.publication_year_window == ["1939", "1940", "1941"]
    assert updated_intent.preferred_publication_years[:1] == ["1940"]
    assert "officeqa_override_query" not in updated_workpad
    assert "officeqa_override_table_query" not in updated_workpad


def test_officeqa_repair_can_switch_to_retrospective_and_relax_provenance_priors():
    retrieval_intent = RetrievalIntent(
        entity="Veterans Administration",
        metric="total expenditures",
        period="1934",
        period_type="fiscal_year",
        target_years=["1934"],
        publication_year_window=["1939", "1940"],
        preferred_publication_years=["1939", "1940"],
        source_constraint_policy="soft",
        granularity_requirement="fiscal_year",
        document_family="treasury_bulletin",
        aggregation_shape="fiscal_year_total",
        strategy="table_first",
        must_include_terms=["Treasury Bulletin", "Veterans Administration", "treasury_bulletin_1940_01.json"],
        query_plan={"source_file_query": "treasury_bulletin_1940_01.json"},
    )

    updated_intent, _, changed, _ = _apply_officeqa_llm_repair_decision(
        retrieval_intent,
        {},
        {
            "decision": "keep",
            "publication_scope_action": "switch_to_retrospective",
            "relax_provenance_priors": True,
        },
    )

    assert changed is True
    assert updated_intent.retrospective_evidence_allowed is True
    assert updated_intent.retrospective_evidence_required is True
    assert updated_intent.query_plan.source_file_query == ""
    assert "Treasury Bulletin" not in updated_intent.must_include_terms
    assert "Veterans Administration" in updated_intent.must_include_terms


def test_officeqa_repair_can_restart_query_universe_from_semantic_plan():
    retrieval_intent = RetrievalIntent(
        entity="Public debt",
        metric="public debt outstanding",
        period="1945",
        period_type="point_lookup",
        target_years=["1945"],
        publication_year_window=["1944", "1945", "1946"],
        preferred_publication_years=["1945", "1946", "1944"],
        granularity_requirement="point_lookup",
        document_family="treasury_bulletin",
        aggregation_shape="point_lookup",
        strategy="table_first",
        query_plan={
            "primary_semantic_query": "Treasury Bulletin public debt outstanding 1945",
            "temporal_query": "Treasury Bulletin public debt outstanding 1945 1945 1946",
            "alternate_lexical_query": "\"Public debt\" \"1945\"",
            "granularity_query": "public debt outstanding 1945 point lookup",
            "qualifier_query": "public debt 1945",
            "source_file_query": "treasury_bulletin_1945_08.json",
        },
        query_candidates=["bad override", "Treasury Bulletin public debt outstanding 1945"],
    )

    updated_intent, updated_workpad, changed, reroute_action = _apply_officeqa_llm_repair_decision(
        retrieval_intent,
        {"officeqa_override_query": "bad override", "officeqa_override_table_query": "bad table"},
        {
            "decision": "keep",
            "restart_scope": "semantic_plan_restart",
        },
    )

    assert changed is True
    assert reroute_action == "semantic_plan_restart"
    assert updated_workpad["officeqa_restart_from_semantic_plan"] is True
    assert "officeqa_override_query" not in updated_workpad
    assert "bad override" not in updated_intent.query_candidates
    assert updated_intent.query_candidates[0] == "Treasury Bulletin public debt outstanding 1945"


def test_officeqa_search_marks_temporally_local_candidate_pool_for_widening():
    source_bundle = SourceBundle(
        task_text="What were the total expenditures for U.S. national defense in the calendar year 1940?",
        focus_query="U.S. national defense total expenditures calendar year 1940",
        target_period="1940",
        entities=["U.S. national defense"],
        source_files_expected=[f"treasury_bulletin_1940_{month:02d}.json" for month in range(1, 13)],
    )
    retrieval_intent = RetrievalIntent(
        entity="U.S. national defense",
        metric="total expenditures",
        period="1940",
        period_type="calendar_year",
        target_years=["1940"],
        publication_year_window=["1939", "1940", "1941"],
        preferred_publication_years=["1941", "1940", "1939"],
        source_constraint_policy="soft",
        granularity_requirement="calendar_year",
        document_family="treasury_bulletin",
        aggregation_shape="calendar_year_total",
        strategy="table_first",
        query_candidates=[
            "Treasury Bulletin U.S. national defense total expenditures 1941 1940 calendar year",
            "Treasury Bulletin U.S. national defense total expenditures 1940 calendar year",
        ],
    )
    tool_plan = ToolPlan(selected_tools=["search_officeqa_documents", "fetch_officeqa_table"])
    registry = build_capability_registry(BUILTIN_RETRIEVAL_TOOLS)
    journal = ExecutionJournal(
        retrieval_iterations=1,
        retrieval_queries=["Treasury Bulletin U.S. national defense total expenditures 1941 1940 calendar year"],
        tool_results=[
            {
                "type": "search_officeqa_documents",
                "retrieval_status": "ok",
                "facts": {
                    "results": [
                        {
                            "document_id": "treasury_bulletin_1940_07_json",
                            "title": "Treasury Bulletin 1940-07",
                            "url": "treasury_bulletin_1940_07.json",
                            "snippet": "Budget receipts and expenditures summary.",
                            "rank": 1,
                            "score": 1.2,
                            "metadata": {
                                "publication_year": "1940",
                                "best_evidence_unit": {
                                    "table_family": "annual_summary",
                                    "period_type": "calendar_year",
                                    "table_confidence": 0.72,
                                },
                            },
                        },
                        {
                            "document_id": "treasury_bulletin_1940_03_json",
                            "title": "Treasury Bulletin 1940-03",
                            "url": "treasury_bulletin_1940_03.json",
                            "snippet": "General expenditures summary.",
                            "rank": 2,
                            "score": 1.1,
                            "metadata": {
                                "publication_year": "1940",
                                "best_evidence_unit": {
                                    "table_family": "annual_summary",
                                    "period_type": "calendar_year",
                                    "table_confidence": 0.7,
                                },
                            },
                        },
                    ]
                },
            }
        ],
    )

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=journal,
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    assert action.tool_name == "search_officeqa_documents"
    assert action.evidence_gap == "source pool too narrow"
    assert action.query == retrieval_intent.query_candidates[1]


def test_officeqa_plan_reselects_better_table_within_same_document():
    source_bundle = SourceBundle(
        task_text="According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
        focus_query="public debt outstanding 1945",
        target_period="1945",
        entities=[],
    )
    retrieval_intent = RetrievalIntent(
        entity="",
        metric="public debt outstanding",
        period="1945",
        period_type="point_lookup",
        target_years=["1945"],
        publication_year_window=["1944", "1945", "1946"],
        preferred_publication_years=["1945", "1946", "1944"],
        granularity_requirement="point_lookup",
        document_family="treasury_bulletin",
        aggregation_shape="point_lookup",
        strategy="table_first",
        answer_mode="deterministic_compute",
        compute_policy="required",
    )
    tool_plan = ToolPlan(selected_tools=["fetch_officeqa_table", "lookup_officeqa_rows"])
    registry = build_capability_registry(BUILTIN_RETRIEVAL_TOOLS)
    journal = ExecutionJournal(
        retrieval_iterations=1,
        tool_results=[
            {
                "type": "fetch_officeqa_table",
                "retrieval_status": "ok",
                "assumptions": {
                    "document_id": "treasury_bulletin_1945_08_json",
                    "path": "treasury_bulletin_1945_08.json",
                    "table_query": "public debt outstanding 1945",
                },
                "facts": {
                    "document_id": "treasury_bulletin_1945_08_json",
                    "citation": "treasury_bulletin_1945_08.json",
                    "tables": [
                        {
                            "table_locator": "Treasury Bulletin | Table 5.- United States Savings Bonds Issued and Redeemed Through June 30, 1945",
                            "table_family": "debt_or_balance_sheet",
                            "headers": ["Row", "Amount issued 1/", "Amount redeemed 1/", "Amount outstanding 2/"],
                            "row_labels": ["Series A-1935", "Series B-1936", "Series D-1941"],
                        }
                    ],
                    "metadata": {
                        "officeqa_status": "ok",
                        "table_candidates": [
                            {
                                "locator": "Treasury Bulletin | Table 5.- United States Savings Bonds Issued and Redeemed Through June 30, 1945",
                                "table_family": "debt_or_balance_sheet",
                                "table_confidence": 0.91,
                                "ranking_score": 5.17,
                                "heading_chain": ["Treasury Bulletin", "Table 5.- United States Savings Bonds Issued and Redeemed Through June 30, 1945"],
                                "headers": ["Row", "Amount issued 1/", "Amount redeemed 1/", "Amount outstanding 2/"],
                                "row_labels": ["Series A-1935", "Series B-1936", "Series D-1941"],
                                "column_paths": ["Amount issued 1/", "Amount redeemed 1/", "Amount outstanding 2/"],
                            },
                            {
                                "locator": "Financial Operations of the United States Government During the Fiscal Year 1946 | Table 4.- Public Debt Outstanding, June 30, 1945 and 1946",
                                "table_family": "debt_or_balance_sheet",
                                "table_confidence": 0.88,
                                "ranking_score": 6.4,
                                "heading_chain": ["Financial Operations of the United States Government During the Fiscal Year 1946", "Table 4.- Public Debt Outstanding, June 30, 1945 and 1946"],
                                "headers": ["June 30, 1945", "June 30, 1946", "Change"],
                                "row_labels": ["Total public debt outstanding", "Marketable", "Nonmarketable"],
                                "column_paths": ["June 30, 1945", "June 30, 1946", "Change"],
                            },
                        ],
                    },
                },
            }
        ],
    )

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=journal,
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    assert action.tool_name == "fetch_officeqa_table"
    assert action.document_id == "treasury_bulletin_1945_08_json"
    assert action.evidence_gap == "wrong row or column semantics"
    assert "Public Debt Outstanding" in action.query


def test_engine_solver_context_block_removes_redundant_objective_and_tool_query_noise():
    payload = solver_context_block(
        {
            "objective": "repeat me",
            "facts_in_use": [{"type": "jurisdictions", "value": ["EU", "US"]}],
            "open_questions": ["Confirm remediation timeline."],
            "assumptions": [],
            "requested_output": {"format": "text"},
        },
        [
            {
                "type": "legal_playbook_retrieval",
                "facts": {
                    "query": "repeat me",
                    "deal_size_hint": "",
                    "urgency": "",
                    "playbook_points": ["point 1", "point 2"],
                },
            }
        ],
        include_objective=False,
    )

    assert '"objective"' not in payload
    assert '"tool_results"' not in payload
    assert '"tool_findings"' in payload
    assert '"query"' not in payload
    assert "repeat me" not in payload


def test_officeqa_document_tasks_route_document_first_without_pending_calculator(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")

    @tool
    def search_treasury_bulletins(query: str, top_k: int = 5) -> dict:
        """Search Treasury Bulletin documents."""
        return {}

    @tool
    def read_treasury_bulletin(document_id: str = "", url: str = "", page_start: int = 0, page_limit: int = 5) -> dict:
        """Read a Treasury Bulletin document."""
        return {}

    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all "
        "individual calendar months in 1940, what was the absolute percent change of these total sum values?"
    )
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    assert state["task_intent"]["execution_mode"] == "document_grounded_analysis"
    resolver = make_capability_resolver(
        build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, search_treasury_bulletins, read_treasury_bulletin])
    )

    result = resolver(state)

    assert result["tool_plan"]["selected_tools"][0] == "search_treasury_bulletins"
    assert "calculator" in result["tool_plan"]["selected_tools"]
    assert "calculator" not in result["tool_plan"]["pending_tools"]


def test_officeqa_financial_reasoning_questions_still_route_to_document_qa(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")

    prompt = (
        "Using Treasury Bulletin values, compute the inflation-adjusted weighted average expenditures for 1953 "
        "and report the standard deviation of the monthly series."
    )
    state = make_state(prompt)
    state.update(intake(state))
    fast_path = fast_path_gate(state)
    state.update(fast_path)
    state.update(task_planner(state))

    intent = state["task_intent"]
    assert intent["task_family"] == "document_qa"
    assert intent["execution_mode"] == "document_grounded_analysis"
    assert "exact_compute" in intent["tool_families_needed"]
    assert "analytical_reasoning" in intent["tool_families_needed"]
    assert "needs_math" in state["capability_flags"]
    assert "needs_analytical_reasoning" in state["capability_flags"]


def test_officeqa_context_curator_carries_financial_analysis_modes(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = (
        "Using Treasury Bulletin data, compute the inflation-adjusted weighted average expenditures for 1953 "
        "and explain the forecast trend for the monthly series."
    )
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS]))
    state.update(resolver(state))

    result = context_curator(state)
    analysis_fact = next(
        fact for fact in result["curated_context"]["facts_in_use"] if fact["type"] == "officeqa_analysis_modes"
    )

    assert "inflation_adjustment" in analysis_fact["value"]
    assert "weighted_average" in analysis_fact["value"]
    assert "time_series_forecasting" in analysis_fact["value"]


def test_officeqa_document_tasks_do_not_infer_pnl_report_as_document_tool(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")

    @tool
    def search_treasury_bulletins(query: str, top_k: int = 5) -> dict:
        """Search Treasury Bulletin documents."""
        return {}

    @tool
    def read_treasury_bulletin(document_id: str = "", url: str = "", page_start: int = 0, page_limit: int = 5) -> dict:
        """Read a Treasury Bulletin document."""
        return {}

    @tool
    def get_pnl_report(portfolio_id: str) -> str:
        """Get a comprehensive P&L trade history report from a portfolio."""
        return ""

    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(
        build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, get_pnl_report, search_treasury_bulletins, read_treasury_bulletin])
    )

    result = resolver(state)

    assert "get_pnl_report" not in result["tool_plan"]["selected_tools"]


def test_officeqa_registry_filter_prunes_irrelevant_competition_tools():
    @tool
    def get_pnl_report(portfolio_id: str) -> str:
        """Get a comprehensive P&L trade history report from a portfolio."""
        return ""

    @tool
    def search_treasury_bulletins(query: str, top_k: int = 5) -> dict:
        """Search Treasury Bulletin documents."""
        return {}

    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, get_pnl_report, search_treasury_bulletins, *BUILTIN_LEGAL_TOOLS])
    filtered = filter_registry_for_benchmark(registry, "officeqa")

    assert "calculator" in filtered
    assert "internet_search" in filtered
    assert "search_treasury_bulletins" in filtered
    assert "get_pnl_report" not in filtered
    assert "legal_playbook_retrieval" not in filtered


def test_officeqa_search_ranking_prefers_semantically_relevant_sources():
    source_bundle = SourceBundle(
        task_text="What were the total expenditures of the Veterans Administration in FY 1934?",
        focus_query="Veterans Administration total expenditures 1934",
        target_period="1934",
        entities=["Veterans Administration"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(
        source_bundle.task_text,
        source_bundle,
        {"benchmark_adapter": "officeqa"},
    )

    ranked = _rank_search_candidates(
        [
            {
                "title": "Depository Invoice No. 1666 - GovInfo",
                "snippet": "Monthly Statement of Capital ... VETERANS' ADMINISTRATION Medical Bulletin",
                "citation": "https://www.govinfo.gov/content/pkg/GOVPUB-GP3-dbacb11a808d6c8ebadba8bc449dd18a/html/GOVPUB-GP3-dbacb11a808d6c8ebadba8bc449dd18a.htm",
                "path": "",
                "document_id": "",
                "rank": 1,
                "metadata": {"years": ["1959"], "section_titles": ["Public debt"], "table_headers": ["Total Federal securities"], "row_labels": []},
            },
            {
                "title": "[PDF] annual report - administrator of veterans' affairs",
                "snippet": "Annual report for fiscal year 1934.",
                "citation": "https://www.va.gov/vetdata/docs/FY1934.pdf",
                "path": "",
                "document_id": "",
                "rank": 3,
                "metadata": {"years": ["1934"], "section_titles": ["Veterans Administration"], "table_headers": ["Fiscal year 1934"], "row_labels": ["Total expenditures"]},
            },
        ],
        retrieval_intent,
        source_bundle,
        {"benchmark_adapter": "officeqa"},
    )

    assert "veterans" in ranked[0]["title"].lower()
    assert "depository invoice" in ranked[-1]["title"].lower()


def test_officeqa_search_requeries_when_top_candidate_confidence_is_weak():
    prompt = "What were the total expenditures of the Veterans Administration in FY 1934?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="Veterans Administration total expenditures 1934",
        target_period="1934",
        entities=["Veterans Administration"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})
    registry = build_capability_registry([CALCULATOR_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    tool_plan = ToolPlan(
        tool_families_needed=["document_retrieval"],
        selected_tools=["search_officeqa_documents", "fetch_officeqa_table", "fetch_officeqa_pages"],
    )
    journal = ExecutionJournal(
        tool_results=[
            {
                "type": "search_officeqa_documents",
                "retrieval_status": "ok",
                "facts": {
                    "results": [
                        {
                            "document_id": "treasury_bulletin_1959_09_json",
                            "title": "treasury_bulletin_1959_09.json",
                            "snippet": "Public debt and budget receipts",
                            "citation": "treasury_bulletin_1959_09.json",
                            "rank": 1,
                            "score": 0.58,
                            "metadata": {"years": ["1959"], "section_titles": ["Public debt"], "table_headers": ["Budget receipts"], "row_labels": []},
                        }
                    ]
                },
            }
        ],
        retrieval_iterations=1,
        retrieval_queries=["Veterans Administration total expenditures 1934"],
    )

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=journal,
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    assert action.tool_name == "search_officeqa_documents"
    assert action.stage == "identify_source"
    assert action.evidence_gap == "wrong document"


def test_officeqa_planner_reopens_source_search_after_missing_row_in_wrong_document():
    prompt = "What were the total expenditures of the Veterans Administration in FY 1934?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="Veterans Administration total expenditures 1934",
        target_period="1934",
        entities=["Veterans Administration"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})
    registry = build_capability_registry([CALCULATOR_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    tool_plan = ToolPlan(
        tool_families_needed=["document_retrieval"],
        selected_tools=["search_officeqa_documents", "fetch_officeqa_table", "lookup_officeqa_rows", "fetch_officeqa_pages"],
    )
    journal = ExecutionJournal(
        tool_results=[
            {
                "type": "search_officeqa_documents",
                "retrieval_status": "ok",
                "facts": {
                    "results": [
                        {
                            "document_id": "treasury_bulletin_1959_09_json",
                            "title": "treasury_bulletin_1959_09.json",
                            "snippet": "Public debt and budget receipts",
                            "citation": "treasury_bulletin_1959_09.json",
                            "rank": 1,
                            "score": 0.58,
                            "metadata": {"years": ["1959"], "section_titles": ["Public debt"], "table_headers": ["Budget receipts"], "row_labels": []},
                        },
                        {
                            "document_id": "treasury_bulletin_1939_01_json",
                            "title": "treasury_bulletin_1939_01.json",
                            "snippet": "Veterans Administration expenditures statement",
                            "citation": "treasury_bulletin_1939_01.json",
                            "rank": 2,
                            "score": 1.41,
                            "metadata": {"years": ["1934", "1939"], "section_titles": ["Veterans Administration"], "table_headers": ["Fiscal year 1934"], "row_labels": ["Total expenditures"]},
                        },
                    ]
                },
            },
            {
                "type": "lookup_officeqa_rows",
                "retrieval_status": "ok",
                "assumptions": {"document_id": "treasury_bulletin_1959_09_json", "path": "treasury_bulletin_1959_09.json", "table_query": "Veterans Administration total expenditures 1934"},
                "facts": {
                    "document_id": "treasury_bulletin_1959_09_json",
                    "citation": "treasury_bulletin_1959_09.json",
                    "metadata": {"officeqa_status": "missing_row"},
                    "tables": [{"locator": "table 2", "table_family": "debt_or_balance_sheet", "headers": ["Budget receipts"], "rows": []}],
                },
            },
        ],
        retrieval_iterations=2,
        retrieval_queries=["Veterans Administration total expenditures 1934"],
    )

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=journal,
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    assert action.tool_name == "fetch_officeqa_table"
    assert action.document_id == "treasury_bulletin_1939_01_json"
    assert action.evidence_gap == "wrong document"


def test_officeqa_structured_evidence_projects_normalized_table_values(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1940",
                "page": 17,
                "section_title": "National Defense",
                "headers": ["Month", "Expenditures (million dollars)"],
                "rows": [["January", "100.0"], ["February", "101.5"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    table_args = {"document_id": "treasury_1940_json", "table_query": "national defense expenditures"}
    cell_args = {
        "document_id": "treasury_1940_json",
        "table_query": "national defense expenditures",
        "row_query": "February",
        "column_query": "Expenditures",
    }
    table_result = normalize_tool_output("fetch_officeqa_table", fetch_officeqa_table.invoke(table_args), table_args)
    cell_result = normalize_tool_output("lookup_officeqa_cells", lookup_officeqa_cells.invoke(cell_args), cell_args)

    structured = build_officeqa_structured_evidence([table_result.model_dump(), cell_result.model_dump()])

    assert structured["provenance_complete"] is True
    assert structured["value_count"] >= 2
    assert structured["units_seen"] == ["million"]
    assert structured["structure_confidence_summary"]["table_confidence_gate_passed"] is True
    assert structured["typing_consistency_summary"]["typing_consistent"] is True
    february_value = next(
        item
        for item in structured["values"]
        if item["row_label"] == "February" and item["column_label"] == "Expenditures (million dollars)"
    )
    assert february_value["numeric_value"] == 101.5
    assert february_value["normalized_value"] == 101_500_000.0
    assert february_value["period_type"] == "monthly_series"
    assert february_value["row_path"] == ["February"]
    assert february_value["column_path"] == ["Expenditures (million dollars)"]
    assert february_value["table_locator"]
    assert february_value["page_locator"]


def test_officeqa_structured_evidence_builds_cross_document_alignment_summary():
    structured = build_officeqa_structured_evidence(
        [
            {
                "type": "fetch_officeqa_table",
                "facts": {
                    "document_id": "treasury_1940_json",
                    "citation": "https://govinfo.gov/treasury_1940.pdf",
                    "metadata": {"page_start": 4, "file_name": "treasury_1940.pdf"},
                    "tables": [
                        {
                            "locator": "table 1",
                            "citation": "https://govinfo.gov/treasury_1940.pdf",
                            "headers": ["Month", "Receipts (million dollars)"],
                            "rows": [["January", "10.0"], ["February", "12.0"]],
                            "unit_hint": "million dollars",
                        }
                    ],
                },
            },
            {
                "type": "fetch_officeqa_table",
                "facts": {
                    "document_id": "treasury_1953_json",
                    "citation": "https://govinfo.gov/treasury_1953.pdf",
                    "metadata": {"page_start": 6, "file_name": "treasury_1953.pdf"},
                    "tables": [
                        {
                            "locator": "table 1",
                            "citation": "https://govinfo.gov/treasury_1953.pdf",
                            "headers": ["Month", "Receipts (million dollars)"],
                            "rows": [["January", "15.0"], ["February", "18.0"]],
                            "unit_hint": "million dollars",
                        }
                    ],
                },
            },
        ]
    )

    assert structured["alignment_summary"]["aligned_document_count"] == 2
    assert structured["alignment_summary"]["cross_document_series_count"] >= 1
    assert set(structured["alignment_summary"]["aligned_years"]) == {"1940", "1953"}
    assert any(len(item["document_ids"]) == 2 for item in structured["merged_series"])


def test_officeqa_evidence_commit_review_redirects_to_gather_before_compute(monkeypatch):
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)

    monkeypatch.setattr("agent.nodes.orchestrator.predictive_evidence_gaps", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "agent.nodes.orchestrator.should_use_evidence_commit_llm",
        lambda **kwargs: (True, "better_family_visible_in_candidate_pool"),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator.maybe_review_evidence_commitment",
        lambda **kwargs: OfficeQALLMRepairDecision(
            decision="retune_table_query",
            restart_scope="same_document",
            revised_table_query="national defense expenditures calendar year 1940",
            confidence=0.84,
            model_name="mock-reviewer",
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator.assess_evidence_sufficiency",
        lambda *args, **kwargs: EvidenceSufficiency(
            source_family="official_government_finance",
            period_scope="matched",
            aggregation_type="matched",
            entity_scope="matched",
            is_sufficient=True,
            missing_dimensions=[],
            rationale="ok",
        ),
    )

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {
                        "document_id": "treasury_bulletin_1940_07_json",
                        "citation": "treasury_bulletin_1940_07.json",
                        "table_locator": "table 5",
                        "page_locator": "page 18",
                    },
                }
            ],
            "routed_tool_families": ["document_retrieval"],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 2,
            "retrieval_queries": ["national defense expenditures 1940"],
            "retrieved_citations": ["treasury_bulletin_1940_07.json"],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        curated_context={
            "objective": "Find the exact national defense expenditures value.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "structured_evidence": {
                "tables": [
                    {
                        "document_id": "treasury_bulletin_1940_07_json",
                        "citation": "treasury_bulletin_1940_07.json",
                        "table_locator": "table 5",
                        "page_locator": "page 18",
                        "table_family": "mixed_summary",
                    }
                ],
                "values": [
                    {
                        "document_id": "treasury_bulletin_1940_07_json",
                        "citation": "treasury_bulletin_1940_07.json",
                        "table_locator": "table 5",
                        "page_locator": "page 18",
                        "table_family": "mixed_summary",
                        "row_label": "Expenditures",
                        "column_label": "Calendar year 1940",
                        "raw_value": "4748",
                    }
                ],
                "typing_consistency_summary": {"typing_consistent": True},
                "structure_confidence_summary": {"table_confidence_gate_passed": True},
            },
            "compute_result": {},
            "provenance_summary": {},
        },
        workpad={
            "retrieval_diagnostics": {
                "candidate_sources": [
                    {
                        "document_id": "treasury_bulletin_1940_07_json",
                        "best_evidence_unit": {"table_family": "mixed_summary"},
                    },
                    {
                        "document_id": "treasury_bulletin_1941_12_json",
                        "best_evidence_unit": {"table_family": "category_breakdown"},
                    },
                ]
            },
            "officeqa_evidence_review": {
                "status": "ready",
                "predictive_gaps": [],
                "compute_policy": "required",
                "answer_mode": "deterministic_compute",
                "strategy": "table_first",
            },
            "officeqa_llm_control_state": {},
            "officeqa_llm_repair_history": [{"stage": "retrieval_repair"}],
        },
    )
    state["evidence_sufficiency"] = {
        "source_family": "official_government_finance",
        "period_scope": "matched",
        "aggregation_type": "matched",
        "entity_scope": "matched",
        "is_sufficient": True,
        "missing_dimensions": [],
        "rationale": "ok",
    }
    state["retrieval_intent"] = {
        "entity": "U.S. national defense",
        "metric": "total expenditures",
        "period": "1940",
        "period_type": "calendar_year",
        "target_years": ["1940"],
        "publication_year_window": ["1939", "1940", "1941"],
        "preferred_publication_years": ["1941", "1940", "1939"],
        "granularity_requirement": "calendar_year",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "calendar_year_total",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "strategy_confidence": 0.6,
        "analysis_modes": [],
        "semantic_plan": {"used_llm": False},
        "query_candidates": ["national defense expenditures 1940"],
        "query_plan": {"primary_semantic_query": "national defense expenditures 1940"},
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "GATHER"
    assert result["workpad"]["officeqa_evidence_commit_review"]["path_changed"] is True
    assert result["workpad"]["officeqa_llm_usage"][-1]["category"] == "evidence_commit_llm"


def test_solver_context_block_includes_compact_structured_evidence_for_officeqa(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1953.json").write_text(
        json.dumps(
            {
                "title": "Treasury Bulletin 1953",
                "page": 4,
                "section_title": "Agriculture",
                "headers": ["Month", "Receipts (million dollars)"],
                "rows": [["January", "50.0"], ["February", "55.25"]],
                "unit": "million dollars",
            }
        ),
        encoding="utf-8",
    )
    build_officeqa_index(corpus_root=corpus_root)
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    table_args = {"document_id": "treasury_1953_json", "table_query": "agriculture receipts"}
    tool_result = normalize_tool_output("fetch_officeqa_table", fetch_officeqa_table.invoke(table_args), table_args)
    curated = attach_structured_evidence(
        {
            "objective": "Compute change across Treasury Bulletin monthly receipts.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "json"},
            "provenance_summary": {},
        },
        [tool_result.model_dump()],
        {"benchmark_adapter": "officeqa"},
    )

    payload = json.loads(solver_context_block(curated.model_dump(), [tool_result.model_dump()], include_objective=False))

    assert "structured_evidence" in payload
    assert payload["structured_evidence"]["table_count"] == 1
    assert payload["structured_evidence"]["value_count"] >= 2
    assert payload["structured_evidence"]["provenance_complete"] is True
    assert any(item["type"] == "structured_sample_values" for item in curated.facts_in_use)
    assert "tool_findings" in payload


def test_benchmark_document_adapter_hooks_allow_second_document_runtime_without_prompt_hacks(monkeypatch):
    adapter_name = "dummy_docs"
    previous = benchmark_module._DOCUMENT_ADAPTERS.get(adapter_name)
    register_benchmark_document_adapter(
        adapter_name,
        {
            "build_structured_evidence": lambda tool_results: {"kind": "dummy_structured", "count": len(tool_results or [])},
            "compute_result": lambda task_text, retrieval_intent, structured_evidence: {
                "status": "ok",
                "operation": "dummy_compute",
                "display_value": "42",
                "answer_text": "dummy",
                "structured_kind": structured_evidence.get("kind", ""),
            },
            "validate_final": lambda **kwargs: {
                "verdict": "pass",
                "reasoning": "dummy validator",
                "orchestration_strategy": "table_compute",
            },
        },
    )

    try:
        overrides = {"benchmark_adapter": adapter_name}
        structured = benchmark_build_structured_evidence([{"type": "dummy_tool"}], overrides)
        compute_result = benchmark_compute_result(
            "dummy task",
            build_retrieval_intent("dummy task", SourceBundle(task_text="dummy task"), None),
            structured,
            overrides,
        )
        validation = benchmark_validate_final(
            task_text="dummy task",
            retrieval_intent=build_retrieval_intent("dummy task", SourceBundle(task_text="dummy task"), None),
            curated_context={"structured_evidence": structured, "compute_result": compute_result},
            evidence_sufficiency={"is_sufficient": True},
            citations=[],
            benchmark_overrides=overrides,
        )

        assert structured["kind"] == "dummy_structured"
        assert compute_result["operation"] == "dummy_compute"
        assert validation["verdict"] == "pass"
    finally:
        if previous is None:
            benchmark_module._DOCUMENT_ADAPTERS.pop(adapter_name, None)
        else:
            benchmark_module._DOCUMENT_ADAPTERS[adapter_name] = previous


def test_officeqa_benchmark_env_forces_document_grounded_runtime_without_prompt_keywords(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")

    prompt = "Use the provided source files and compute the exact answer."
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))

    assert state["benchmark_overrides"]["benchmark_adapter"] == "officeqa"
    assert state["benchmark_overrides"]["officeqa_mode"] is True
    assert state["task_intent"]["task_family"] == "document_qa"
    assert state["task_intent"]["execution_mode"] == "document_grounded_analysis"


def test_evidence_sufficiency_ignores_generic_numeric_summary_metadata():
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="U.S. national defense total expenditures 1940",
        target_period="1940",
        entities=["U.S. national defense"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )

    sufficiency = assess_evidence_sufficiency(
        prompt,
        source_bundle,
        [
            {
                "type": "fetch_reference_file",
                "retrieval_status": "ok",
                "facts": {
                    "citation": "https://example.com/catalog.pdf",
                    "metadata": {"file_name": "catalog.pdf", "format": "pdf", "status": "ok"},
                    "chunks": [{"locator": "Pages 1-5", "text": "Monthly catalog of publications.", "citation": "https://example.com/catalog.pdf"}],
                    "tables": [],
                    "numeric_summaries": [
                        {"metric": "row_count", "value": 19},
                        {"metric": "column_count", "value": 2},
                        {"metric": "numeric_cell_count", "value": 9},
                    ],
                },
            }
        ],
        {"benchmark_adapter": "officeqa"},
    )

    assert sufficiency.is_sufficient is False
    assert "numeric or quoted support" in sufficiency.missing_dimensions


def test_officeqa_retrieval_intent_avoids_legacy_web_query_templates(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")

    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all "
        "individual calendar months in 1940, what was the absolute percent change of these total sum values?"
    )
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="",
        target_period="",
        entities=[],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )

    retrieval_intent = build_retrieval_intent(prompt, source_bundle, intake(make_state(prompt))["benchmark_overrides"])
    joined_queries = " || ".join(retrieval_intent.query_candidates).lower()

    assert "site:govinfo.gov" not in joined_queries
    assert "federal reserve bank of minneapolis" not in joined_queries
    assert "national defense and associated activities" not in joined_queries


def test_officeqa_retrieval_intent_selects_hybrid_strategy_for_statistical_forecast_questions(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = (
        "Using Treasury Bulletin data, calculate the regression trend and forecast the monthly expenditures series "
        "for 1953."
    )
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="monthly expenditures 1953 regression forecast",
        target_period="1953",
        entities=["Treasury Bulletin"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )

    retrieval_intent = build_retrieval_intent(prompt, source_bundle, intake(make_state(prompt))["benchmark_overrides"])

    assert retrieval_intent.strategy == "hybrid"
    assert retrieval_intent.answer_mode == "hybrid_grounded"
    assert retrieval_intent.compute_policy == "required"
    assert retrieval_intent.partial_answer_allowed is False
    assert retrieval_intent.evidence_plan.requires_table_support is True
    assert retrieval_intent.evidence_plan.requires_text_support is True
    assert retrieval_intent.evidence_plan.requires_forecast_support is True
    assert "Capture quoted or page-level narrative support for ambiguous or implicit metrics." in retrieval_intent.evidence_requirements


def test_officeqa_retrieval_intent_marks_calendar_year_total_as_deterministic(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="national defense expenditures 1940",
        target_period="1940",
        entities=["National Defense"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )

    retrieval_intent = build_retrieval_intent(prompt, source_bundle, intake(make_state(prompt))["benchmark_overrides"])

    assert retrieval_intent.answer_mode == "deterministic_compute"
    assert retrieval_intent.compute_policy == "required"
    assert retrieval_intent.partial_answer_allowed is False


def test_officeqa_retrieval_intent_selects_multi_table_for_inflation_adjusted_weighted_average(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = "Using Treasury Bulletin data, compute the inflation-adjusted weighted average expenditures for 1953."
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="inflation adjusted weighted average expenditures 1953",
        target_period="1953",
        entities=["Treasury Bulletin"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )

    retrieval_intent = build_retrieval_intent(prompt, source_bundle, intake(make_state(prompt))["benchmark_overrides"])

    assert retrieval_intent.strategy == "multi_table"
    assert retrieval_intent.answer_mode == "deterministic_compute"
    assert retrieval_intent.compute_policy == "required"
    assert retrieval_intent.partial_answer_allowed is False
    assert retrieval_intent.evidence_plan.requires_inflation_support is True
    assert retrieval_intent.evidence_plan.join_keys
    assert "join_ready_support" in {item.kind for item in retrieval_intent.evidence_plan.requirements}


def test_officeqa_executor_uses_compute_capability_acquisition_before_required_insufficiency(monkeypatch):
    prompt = "Using Treasury Bulletin data, calculate the standard deviation of monthly expenditures for 1953."
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    monkeypatch.setattr(
        "agent.nodes.orchestrator.assess_evidence_sufficiency",
        lambda *args, **kwargs: EvidenceSufficiency(
            source_family="official_government_document",
            period_scope="match",
            aggregation_type="match",
            entity_scope="match",
            is_sufficient=True,
            missing_dimensions=[],
            rationale="Structured evidence is sufficient.",
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator.benchmark_compute_result",
        lambda *args, **kwargs: OfficeQAComputeResult(
            status="unsupported",
            operation="point_lookup",
            validation_errors=["Deterministic OfficeQA compute does not yet support this aggregation shape."],
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator.maybe_acquire_officeqa_compute_result",
        lambda **kwargs: (
            OfficeQAComputeResult(
                status="ok",
                operation="point_lookup",
                final_value=42.0,
                display_value="42",
                answer_text="Deterministic OfficeQA compute: point lookup.\n- Validated synthesized compute = 42.\nFinal answer: 42",
                citations=["treasury_1953.json"],
                capability_source="synthesized",
                capability_signature="cap-stddev",
                capability_validated=True,
            ),
            dict(kwargs["workpad"]),
            dict(kwargs["llm_control_state"]),
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator._plan_retrieval_action",
        lambda **kwargs: RetrievalAction(action="answer", stage="compute_ready", strategy="multi_table"),
    )
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "routing_rationale": "",
            "confidence": 0.95,
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["fetch_officeqa_table"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        source_bundle={
            "task_text": prompt,
            "focus_query": "standard deviation monthly expenditures 1953",
            "target_period": "1953",
            "entities": ["Expenditures"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
            "structured_evidence": {
                "tables": [],
                "values": [
                    {
                        "document_id": "treasury_1953_json",
                        "citation": "treasury_1953.json",
                        "page_locator": "page 1",
                        "table_locator": "Monthly Expenditures",
                        "row_label": "January",
                        "row_path": ["January"],
                        "column_label": "Expenditures",
                        "column_path": ["Expenditures"],
                        "raw_value": "10",
                        "numeric_value": 10.0,
                        "normalized_value": 10.0,
                        "unit": "million",
                        "unit_multiplier": 1.0,
                        "unit_kind": "currency",
                        "structure_confidence": 0.95,
                    },
                    {
                        "document_id": "treasury_1953_json",
                        "citation": "treasury_1953.json",
                        "page_locator": "page 1",
                        "table_locator": "Monthly Expenditures",
                        "row_label": "February",
                        "row_path": ["February"],
                        "column_label": "Expenditures",
                        "column_path": ["Expenditures"],
                        "raw_value": "20",
                        "numeric_value": 20.0,
                        "normalized_value": 20.0,
                        "unit": "million",
                        "unit_multiplier": 1.0,
                        "unit_kind": "currency",
                        "structure_confidence": 0.95,
                    },
                ],
                "page_chunks": [],
                "provenance_complete": True,
                "structure_confidence_summary": {
                    "min_confidence": 0.95,
                    "avg_confidence": 0.95,
                    "max_confidence": 0.95,
                    "low_confidence_value_count": 0,
                    "low_confidence_table_count": 0,
                    "table_confidence_gate_passed": True,
                },
            },
            "compute_result": {},
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {"document_id": "treasury_1953_json", "citation": "treasury_1953.json", "metadata": {"officeqa_status": "ok"}},
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": ["treasury_1953.json"],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
    )
    state["retrieval_intent"] = {
        "entity": "Expenditures",
        "metric": "standard deviation of expenditures",
        "period": "1953",
        "period_type": "calendar_year",
        "target_years": ["1953"],
        "document_family": "official_government_finance",
        "aggregation_shape": "point_lookup",
        "analysis_modes": ["statistical_analysis"],
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "partial_answer_allowed": False,
        "strategy": "multi_table",
        "strategy_confidence": 0.82,
        "evidence_requirements": [],
        "fallback_chain": ["hybrid", "text_first"],
        "join_requirements": [],
        "evidence_plan": {
            "requires_table_support": True,
            "requires_text_support": False,
            "requires_cross_source_alignment": False,
            "required_years": ["1953"],
            "join_keys": [],
        },
        "must_include_terms": ["expenditures", "1953"],
        "must_exclude_terms": [],
        "query_candidates": ["standard deviation monthly expenditures 1953"],
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "SYNTHESIZE"
    assert "Final answer: 42" in str(result["messages"][0].content)
    assert result["workpad"]["officeqa_compute"]["capability_source"] == "synthesized"
    assert result["workpad"]["officeqa_compute"]["capability_validated"] is True


def test_officeqa_context_curator_carries_retrieval_plan_summary(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = "Using Treasury Bulletin data, calculate the regression trend and forecast the monthly expenditures series for 1953."
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS]))
    state.update(resolver(state))

    result = context_curator(state)
    retrieval_plan = result["curated_context"]["provenance_summary"]["retrieval_plan"]

    assert retrieval_plan["strategy"] == "hybrid"
    assert retrieval_plan["required_years"] == ["1953"]
    assert retrieval_plan["evidence_requirements"]


def test_officeqa_planner_prefers_text_first_pages_for_narrative_document_questions():
    prompt = "According to the Treasury Bulletin narrative discussion, what reason was given for the 1945 debt outlook?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="debt outlook 1945 narrative discussion",
        target_period="1945",
        entities=["Treasury Bulletin"],
        urls=[],
        source_files_found=[{"document_id": "treasury_1945_json", "relative_path": "treasury_1945.json"}],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})
    registry = build_capability_registry([CALCULATOR_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    tool_plan = ToolPlan(
        tool_families_needed=["document_retrieval", "exact_compute"],
        selected_tools=[
            "search_officeqa_documents",
            "fetch_officeqa_pages",
            "fetch_officeqa_table",
            "lookup_officeqa_rows",
            "lookup_officeqa_cells",
            "calculator",
        ],
    )

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=ExecutionJournal(),
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    assert action.tool_name == "fetch_officeqa_pages"
    assert action.strategy in {"text_first", "hybrid"}
    assert action.strategy_reason
    assert action.candidate_sources


def test_officeqa_single_year_questions_do_not_force_multi_document_alignment():
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="total public debt outstanding 1945",
        target_period="1945",
        entities=["Treasury Bulletin"],
        urls=[],
        source_files_expected=[
            "treasury_bulletin_1945_01.json",
            "treasury_bulletin_1945_02.json",
            "treasury_bulletin_1945_03.json",
        ],
        source_files_found=[
            {"document_id": "treasury_bulletin_1945_01_json", "relative_path": "treasury_bulletin_1945_01.json"},
            {"document_id": "treasury_bulletin_1945_02_json", "relative_path": "treasury_bulletin_1945_02.json"},
            {"document_id": "treasury_bulletin_1945_03_json", "relative_path": "treasury_bulletin_1945_03.json"},
        ],
        inline_facts={},
        tables=[],
        formulas=[],
    )

    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})

    assert retrieval_intent.strategy != "multi_document"
    assert retrieval_intent.evidence_plan.requires_cross_source_alignment is False
    assert retrieval_intent.answer_mode == "deterministic_compute"
    assert retrieval_intent.compute_policy == "required"


def test_officeqa_planner_tries_alternate_table_query_for_multi_table_questions():
    prompt = "Using Treasury Bulletin data, compute the inflation-adjusted weighted average expenditures for 1953."
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="inflation adjusted weighted average expenditures 1953",
        target_period="1953",
        entities=["Treasury Bulletin"],
        urls=[],
        source_files_found=[{"document_id": "treasury_1953_json", "relative_path": "treasury_1953.json"}],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})
    registry = build_capability_registry([CALCULATOR_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    tool_plan = ToolPlan(
        tool_families_needed=["document_retrieval", "exact_compute"],
        selected_tools=["fetch_officeqa_table", "fetch_officeqa_pages", "lookup_officeqa_rows", "lookup_officeqa_cells"],
    )
    journal = ExecutionJournal(
        tool_results=[
            {
                "type": "fetch_officeqa_table",
                "retrieval_status": "ok",
                "assumptions": {"document_id": "treasury_1953_json", "path": "treasury_1953.json", "table_query": "Treasury Bulletin expenditures 1953"},
                "facts": {
                    "document_id": "treasury_1953_json",
                    "citation": "treasury_1953.json",
                    "metadata": {"officeqa_status": "partial_table"},
                    "tables": [{"locator": "table 1", "headers": ["Month", "Expenditures"], "rows": [["January", "10.0"]]}],
                },
            }
        ],
        retrieval_iterations=1,
    )

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=journal,
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    assert action.tool_name == "fetch_officeqa_table"
    assert action.strategy in {"multi_table", "hybrid"}
    assert action.query
    assert action.query.lower() != "treasury bulletin expenditures 1953"
    assert action.strategy_reason


def test_officeqa_planner_retries_table_extraction_when_monthly_question_hits_annual_summary():
    prompt = "Using specifically only the reported values for all individual calendar months in 1953, what was the total sum of these values?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="monthly expenditures 1953",
        target_period="1953",
        entities=["Treasury Bulletin"],
        urls=[],
        source_files_found=[{"document_id": "treasury_1953_json", "relative_path": "treasury_1953.json"}],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(prompt, source_bundle, {"benchmark_adapter": "officeqa"})
    registry = build_capability_registry([CALCULATOR_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    tool_plan = ToolPlan(
        tool_families_needed=["document_retrieval", "exact_compute"],
        selected_tools=["fetch_officeqa_table", "fetch_officeqa_pages", "lookup_officeqa_rows", "lookup_officeqa_cells"],
    )
    journal = ExecutionJournal(
        tool_results=[
            {
                "type": "fetch_officeqa_table",
                "retrieval_status": "ok",
                "assumptions": {"document_id": "treasury_1953_json", "path": "treasury_1953.json", "table_query": "Treasury Bulletin expenditures 1953"},
                "facts": {
                    "document_id": "treasury_1953_json",
                    "citation": "treasury_1953.json",
                    "metadata": {"officeqa_status": "ok"},
                    "tables": [{"locator": "table 4", "table_family": "annual_summary", "headers": ["Total 9/", "National defense and related activities"], "rows": [["1953", "900"]]}],
                },
            }
        ],
        retrieval_iterations=1,
    )

    action = _plan_retrieval_action(
        execution_mode="document_grounded_analysis",
        source_bundle=source_bundle,
        retrieval_intent=retrieval_intent,
        tool_plan=tool_plan,
        journal=journal,
        registry=registry,
        benchmark_overrides={"benchmark_adapter": "officeqa"},
    )

    assert action.tool_name == "fetch_officeqa_table"
    assert action.document_id == "treasury_1953_json"
    assert action.evidence_gap == "wrong table family"
    assert "monthly" in action.query.lower()


def test_officeqa_predictive_evidence_gaps_require_month_coverage_before_compute(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all "
        "individual calendar months in 1940, what was the absolute percent change of these total sum values?"
    )
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="monthly expenditures 1953 1940",
        target_period="1953 1940",
        entities=["Treasury Bulletin"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(prompt, source_bundle, intake(make_state(prompt))["benchmark_overrides"])

    gaps = predictive_evidence_gaps(
        retrieval_intent,
        {
            "tables": [{"document_id": "treasury_1953_json", "table_locator": "table 1"}],
            "values": [
                {
                    "document_id": "treasury_1953_json",
                    "citation": "treasury_1953.json",
                    "page_locator": "page 1",
                    "table_locator": "table 1",
                    "row_label": "January 1953",
                    "column_label": "Expenditures",
                    "raw_value": "10.0",
                    "numeric_value": 10.0,
                    "normalized_value": 10000000.0,
                }
            ],
            "page_chunks": [],
        },
    )

    assert "missing month coverage" in gaps
    assert "year coverage" in gaps


def test_officeqa_predictive_evidence_gaps_flag_low_confidence_structure(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="total public debt outstanding 1945",
        target_period="1945",
        entities=["Treasury Bulletin"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )
    retrieval_intent = build_retrieval_intent(prompt, source_bundle, intake(make_state(prompt))["benchmark_overrides"])

    gaps = predictive_evidence_gaps(
        retrieval_intent,
        {
            "tables": [{"document_id": "treasury_1945_json", "table_locator": "table 19"}],
            "values": [
                {
                    "document_id": "treasury_1945_json",
                    "citation": "treasury_1945.json",
                    "page_locator": "page 29",
                    "table_locator": "table 19",
                    "row_label": "Total public debt",
                    "row_path": ["Total public debt outstanding"],
                    "column_label": "Estimated 1/",
                    "column_path": ["End of fiscal years, 1941 to 1945", "1945"],
                    "raw_value": "258682",
                    "numeric_value": 258682.0,
                    "normalized_value": 258682.0,
                    "structure_confidence": 0.42,
                }
            ],
            "page_chunks": [],
            "structure_confidence_summary": {
                "min_confidence": 0.42,
                "avg_confidence": 0.42,
                "max_confidence": 0.42,
                "low_confidence_value_count": 1,
                "low_confidence_table_count": 1,
                "table_confidence_gate_passed": False,
            },
        },
    )

    assert "low-confidence structure" in gaps


def test_officeqa_evidence_sufficiency_rejects_wrong_source_family(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="total public debt outstanding 1945",
        target_period="1945",
        entities=["Treasury Bulletin"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )

    sufficiency = assess_evidence_sufficiency(
        prompt,
        source_bundle,
        [
            {
                "type": "fetch_officeqa_pages",
                "retrieval_status": "ok",
                "facts": {
                    "citation": "https://quizlet.com/treasury_1945",
                    "metadata": {"file_name": "treasury_1945.html", "format": "html", "officeqa_status": "ok"},
                    "chunks": [{"locator": "page 1", "text": "In 1945 total public debt outstanding was 258.7 billion dollars.", "citation": "https://quizlet.com/treasury_1945"}],
                    "tables": [],
                    "numeric_summaries": [{"metric": "public_debt_outstanding", "value": 258.7}],
                },
            }
        ],
        intake(make_state(prompt))["benchmark_overrides"],
    )

    assert sufficiency.is_sufficient is False
    assert "source family grounding" in sufficiency.missing_dimensions


def test_officeqa_evidence_sufficiency_flags_missing_month_coverage(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all "
        "individual calendar months in 1940, what was the absolute percent change of these total sum values?"
    )
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="monthly expenditures 1953 1940",
        target_period="1953 1940",
        entities=["Treasury Bulletin"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
    )

    sufficiency = assess_evidence_sufficiency(
        prompt,
        source_bundle,
        [
            {
                "type": "fetch_officeqa_table",
                "retrieval_status": "ok",
                "facts": {
                    "citation": "treasury_1953.json",
                    "metadata": {"file_name": "treasury_1953.json", "format": "json", "officeqa_status": "ok"},
                    "chunks": [],
                    "tables": [
                        {
                            "locator": "table 1",
                            "headers": ["Month", "Expenditures"],
                            "rows": [["January", "100.0"], ["February", "102.0"]],
                            "citation": "treasury_1953.json",
                            "unit_hint": "million dollars",
                        }
                    ],
                    "numeric_summaries": [{"metric": "expenditures", "value": {"min": 100.0, "max": 102.0}}],
                },
            }
        ],
        intake(make_state(prompt))["benchmark_overrides"],
    )

    assert sufficiency.is_sufficient is False
    assert "missing month coverage" in sufficiency.missing_dimensions


def test_intake_merges_source_files_from_benchmark_metadata(monkeypatch, tmp_path):
    corpus_root = tmp_path / "treasury_bulletins_parsed"
    corpus_root.mkdir(parents=True)
    (corpus_root / "treasury_1940.json").write_text(json.dumps({"title": "Treasury Bulletin 1940"}), encoding="utf-8")
    monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

    from agent.benchmarks.officeqa_index import build_officeqa_index

    build_officeqa_index(corpus_root=corpus_root)

    state = make_state(
        "Use the provided source files to compute the exact answer.",
        benchmark_overrides={"source_files": ["treasury_1940.json"]},
    )

    result = intake(state)

    assert result["benchmark_overrides"]["source_files_expected"] == ["treasury_1940.json"]
    assert result["benchmark_overrides"]["source_files_found"][0]["document_id"] == "treasury_1940_json"


def test_engine_executor_runs_retrieval_search_then_fetch_before_final_answer(monkeypatch):
    @tool
    def search_reference_corpus(query: str, top_k: int = 5, snippet_chars: int = 700) -> dict:
        """Fake corpus search tool."""
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "treasury_1945.txt",
                    "snippet": "In 1945, total public debt outstanding was 258.7 billion dollars.",
                    "url": "treasury_1945.txt",
                    "document_id": "treasury_1945_txt",
                }
            ],
            "documents": [
                {
                    "document_id": "treasury_1945_txt",
                    "citation": "treasury_1945.txt",
                    "format": "txt",
                    "path": "treasury_1945.txt",
                }
            ],
        }

    @tool
    def fetch_corpus_document(document_id: str = "", path: str = "", chunk_start: int = 0, chunk_limit: int = 3) -> dict:
        """Fake corpus fetch tool."""
        return {
            "document_id": document_id or "treasury_1945_txt",
            "citation": path or "treasury_1945.txt",
            "metadata": {"file_name": "treasury_1945.txt", "format": "txt", "window": "chunks 1-1"},
            "chunks": [
                {
                    "locator": "chunk 1",
                    "kind": "text_excerpt",
                    "text": "Total public debt outstanding in 1945 was 258.7 billion dollars.",
                    "citation": path or "treasury_1945.txt",
                }
            ],
            "tables": [],
            "numeric_summaries": [{"metric": "debt", "value": 258.7}],
        }

    captured: list = []
    monkeypatch.setattr(
        "agent.nodes.orchestrator.ChatOpenAI",
        lambda **kwargs: _FakeModel(AIMessage(content='The total public debt outstanding in 1945 was 258.7 billion dollars. [Source: treasury_1945.txt]'), captured),
    )
    monkeypatch.setattr("agent.nodes.orchestrator.invoke_structured_output", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("fallback")))

    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, search_reference_corpus, fetch_corpus_document, *BUILTIN_LEGAL_TOOLS])
    executor = make_executor(registry)
    state = make_state(
        "According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "external_retrieval"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "routing_rationale": "",
            "confidence": 0.85,
            "planner_source": "heuristic",
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "external_retrieval"],
            "widened_families": ["document_retrieval", "external_retrieval"],
            "selected_tools": ["search_reference_corpus", "fetch_corpus_document", "internet_search"],
            "pending_tools": ["search_reference_corpus"],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        source_bundle={
            "task_text": "According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
            "focus_query": "total public debt outstanding in 1945",
            "target_period": "1945",
            "entities": ["Treasury Bulletin"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        curated_context={
            "objective": "total public debt outstanding in 1945",
            "facts_in_use": [{"type": "focus_query", "value": "total public debt outstanding in 1945"}],
            "open_questions": ["Find the exact supporting quote before finalizing."],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
        },
    )

    first = asyncio.run(executor(state))
    state.update(first)
    assert first["solver_stage"] == "GATHER"
    assert first["last_tool_result"]["type"] == "search_reference_corpus"

    second = asyncio.run(executor(state))
    state.update(second)
    assert second["solver_stage"] == "GATHER"
    assert second["last_tool_result"]["type"] == "fetch_corpus_document"

    third = asyncio.run(executor(state))
    assert third["solver_stage"] == "SYNTHESIZE"
    assert "treasury_1945.txt" in str(third["messages"][0].content)
    assert third["workpad"]["completion_budget"] >= 2400
    serialized_prompt = json.dumps([str(msg.content) for msg in captured[0]], ensure_ascii=True)
    assert "treasury_1945.txt" in serialized_prompt
    assert "258.7 billion" in serialized_prompt


def test_engine_executor_prefers_officeqa_table_first_retrieval(monkeypatch):
    @tool
    def search_officeqa_documents(query: str, top_k: int = 5, snippet_chars: int = 700, source_files: list[str] | None = None) -> dict:
        """Search the indexed OfficeQA corpus for candidate Treasury source documents."""
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "treasury_1945.json",
                    "snippet": "Treasury Bulletin table for public debt outstanding.",
                    "url": "treasury_1945.json",
                    "document_id": "treasury_1945_json",
                }
            ],
            "documents": [
                {
                    "document_id": "treasury_1945_json",
                    "citation": "treasury_1945.json",
                    "format": "json",
                    "path": "treasury_1945.json",
                }
            ],
        }

    @tool
    def fetch_officeqa_table(document_id: str = "", path: str = "", table_query: str = "", row_offset: int = 0, row_limit: int = 200) -> dict:
        """Extract the most relevant structured table from an OfficeQA corpus artifact."""
        return {
            "document_id": document_id or "treasury_1945_json",
            "citation": path or "treasury_1945.json",
            "metadata": {
                "file_name": "treasury_1945.json",
                "format": "json",
                "officeqa_status": "ok",
                "table_count": 1,
            },
                "chunks": [
                    {
                        "locator": "table 1",
                        "kind": "table_preview",
                        "text": "Category,1945\nTotal public debt outstanding,258.7",
                        "citation": path or "treasury_1945.json",
                    }
                ],
                "tables": [
                    {
                        "locator": "table 1",
                        "headers": ["Category", "1945"],
                        "rows": [["Total public debt outstanding", "258.7"]],
                        "citation": path or "treasury_1945.json",
                        "unit_hint": "billion dollars",
                    }
            ],
            "numeric_summaries": [{"metric": "public_debt_outstanding", "value": 258.7}],
        }

    captured: list = []
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    monkeypatch.setattr(
        "agent.nodes.orchestrator.ChatOpenAI",
        lambda **kwargs: _FakeModel(AIMessage(content="The total public debt outstanding in 1945 was 258.7. [Source: treasury_1945.json]"), captured),
    )

    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, search_officeqa_documents, fetch_officeqa_table, *BUILTIN_LEGAL_TOOLS])
    executor = make_executor(registry)
    state = make_state(
        "According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
        benchmark_overrides={"benchmark_name": "officeqa", "benchmark_adapter": "officeqa"},
    )
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(registry)
    state.update(resolver(state))
    state.update(context_curator(state))

    first = asyncio.run(executor(state))
    state.update(first)
    assert first["solver_stage"] == "GATHER"
    assert first["last_tool_result"]["type"] == "search_officeqa_documents"

    second = asyncio.run(executor(state))
    state.update(second)
    assert second["solver_stage"] == "GATHER"
    assert second["last_tool_result"]["type"] == "fetch_officeqa_table"

    third = asyncio.run(executor(state))
    assert third["solver_stage"] == "SYNTHESIZE"
    assert "Deterministic OfficeQA compute: point lookup." in str(third["messages"][0].content)
    assert "Final answer:" in str(third["messages"][0].content)
    assert not captured


def test_officeqa_executor_uses_llm_wrapper_for_hybrid_answer_mode(monkeypatch):
    prompt = "What was total public debt outstanding in 1945, and what trend does the retrieved evidence suggest?"
    captured: list = []
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    monkeypatch.setattr(
        "agent.nodes.orchestrator.ChatOpenAI",
        lambda **kwargs: _FakeModel(
            AIMessage(
                content=(
                    "Based on the retrieved evidence, the 1945 total public debt outstanding was 258.7 and the nearby narrative indicates continued elevated debt pressure. "
                    "[Source: treasury_1945.json]"
                )
            ),
            captured,
        ),
    )

    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute", "analytical_reasoning"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "routing_rationale": "",
            "confidence": 0.95,
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["fetch_officeqa_table", "fetch_officeqa_pages", "lookup_officeqa_cells"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        source_bundle={
            "task_text": prompt,
            "focus_query": "public debt outstanding 1945 trend",
            "target_period": "1945",
            "entities": ["Public debt outstanding"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        curated_context={
            "objective": prompt,
            "facts_in_use": [{"type": "answer_mode", "value": "hybrid_grounded"}],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
            "structured_evidence": {
                "tables": [
                    {
                        "document_id": "treasury_1945_json",
                        "citation": "treasury_1945.json",
                        "page_locator": "page 2",
                        "table_locator": "table 1",
                        "headers": ["Category", "Amount"],
                        "row_count": 1,
                        "column_count": 2,
                        "unit": "billion dollars",
                    }
                ],
                "values": [
                    {
                        "document_id": "treasury_1945_json",
                        "citation": "treasury_1945.json",
                        "page_locator": "page 2",
                        "table_locator": "table 1",
                        "row_label": "Total public debt outstanding",
                        "column_label": "Amount",
                        "raw_value": "258.7",
                        "numeric_value": 258.7,
                        "normalized_value": 258.7,
                        "unit": "billion",
                        "unit_multiplier": 1.0,
                        "unit_kind": "currency",
                    }
                ],
                "page_chunks": [
                    {
                        "document_id": "treasury_1945_json",
                        "citation": "treasury_1945.json",
                        "page_locator": "page 2",
                        "text": "Debt remained elevated through 1945 because of war financing pressures.",
                    }
                ],
                "units_seen": ["billion"],
                "value_count": 1,
                "provenance_complete": True,
            },
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {"document_id": "treasury_1945_json", "citation": "treasury_1945.json", "metadata": {"officeqa_status": "ok"}},
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 99,
            "retrieval_queries": [],
            "retrieved_citations": ["treasury_1945.json"],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
    )
    state["retrieval_intent"] = {
        "entity": "Public debt outstanding",
        "metric": "public debt outstanding",
        "period": "1945",
        "document_family": "official_government_finance",
        "aggregation_shape": "point_lookup",
        "analysis_modes": ["time_series_forecasting"],
        "answer_mode": "hybrid_grounded",
        "compute_policy": "preferred",
        "partial_answer_allowed": True,
        "strategy": "hybrid",
        "strategy_confidence": 0.82,
        "evidence_requirements": ["Ground the primary metric in the source evidence."],
        "fallback_chain": ["text_first", "table_first"],
        "join_requirements": [],
        "evidence_plan": {
            "objective": prompt,
            "metric_identity": "public debt outstanding",
            "expected_unit_kind": "currency",
            "expected_value_count": 1,
            "required_years": ["1945"],
            "required_month_coverage": False,
            "required_month_count": 0,
            "requires_table_support": True,
            "requires_text_support": True,
            "requires_cross_source_alignment": False,
            "requires_inflation_support": False,
            "requires_statistical_series": False,
            "requires_forecast_support": True,
            "required_series": ["public debt outstanding 1945"],
            "join_keys": [],
            "requirements": [],
        },
        "must_include_terms": ["public debt outstanding", "1945"],
        "must_exclude_terms": [],
        "query_candidates": ["public debt outstanding 1945 Treasury Bulletin"],
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "SYNTHESIZE"
    assert "retrieved evidence" in str(result["messages"][0].content)
    assert captured
    serialized_prompt = json.dumps([str(msg.content) for msg in captured[0]], ensure_ascii=True)
    assert "compute_result" in serialized_prompt
    assert "selection_reasoning" in serialized_prompt


def test_officeqa_executor_honors_validator_directed_gather_retry(monkeypatch):
    @tool
    def search_officeqa_documents(query: str, top_k: int = 5, snippet_chars: int = 700, source_files: list[str] | None = None) -> dict:
        """Search the indexed OfficeQA corpus for candidate Treasury source documents."""
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "treasury_1940.json",
                    "snippet": "Treasury Bulletin expenditures table.",
                    "url": "treasury_1940.json",
                    "document_id": "treasury_1940_json",
                }
            ],
            "documents": [
                {
                    "document_id": "treasury_1940_json",
                    "citation": "treasury_1940.json",
                    "format": "json",
                    "path": "treasury_1940.json",
                }
            ],
        }

    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, search_officeqa_documents, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    state = make_state(
        "What were the total expenditures for U.S. national defense in the calendar year 1940?",
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "routing_rationale": "",
            "confidence": 0.95,
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents", "fetch_officeqa_table"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        source_bundle={
            "task_text": "What were the total expenditures for U.S. national defense in the calendar year 1940?",
            "focus_query": "national defense expenditures 1940",
            "target_period": "1940",
            "entities": ["National Defense"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        curated_context={
            "objective": "national defense expenditures 1940",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
            "structured_evidence": {},
            "compute_result": {},
        },
        execution_journal={
            "events": [],
            "tool_results": [],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 0,
            "retrieval_queries": [],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        review_feedback={
            "verdict": "revise",
            "reasoning": "Recover the exact table support.",
            "missing_dimensions": ["aggregation correctness"],
            "improve_hint": "Recover the exact table support.",
            "repair_target": "gather",
            "repair_class": "missing_evidence",
            "orchestration_strategy": "table_compute",
            "remediation_codes": ["RECOVER_AGGREGATION_SUPPORT"],
            "retry_allowed": True,
            "retry_stop_reason": "",
        },
    )
    state["retrieval_intent"] = {
        "entity": "National Defense",
        "metric": "total expenditures",
        "period": "1940",
        "document_family": "official_government_finance",
        "aggregation_shape": "calendar_year_total",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "partial_answer_allowed": False,
        "strategy": "table_first",
        "analysis_modes": [],
        "evidence_plan": {
            "requires_table_support": True,
            "requires_text_support": False,
            "requires_cross_source_alignment": False,
            "required_years": ["1940"],
            "join_keys": [],
        },
        "must_include_terms": ["National Defense", "1940"],
        "must_exclude_terms": [],
        "query_candidates": ["National Defense expenditures 1940 Treasury Bulletin"],
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "GATHER"
    assert result["last_tool_result"]["type"] == "search_officeqa_documents"
    assert result["workpad"]["officeqa_retry_path"]["repair_target"] == "gather"
    assert result["workpad"]["officeqa_retry_path"]["orchestration_strategy"] == "table_compute"


def test_officeqa_executor_applies_structured_validator_repair_before_gather(monkeypatch):
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.nodes.orchestrator._plan_retrieval_action",
        lambda **kwargs: RetrievalAction(
            action="tool",
            tool_name="search_officeqa_documents",
            stage="identify_source",
            strategy="table_first",
            query="public debt outstanding 1945",
            evidence_gap="wrong document",
            rationale="Need a stronger source match.",
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator.maybe_repair_from_validator",
        lambda **kwargs: OfficeQALLMRepairDecision(
            decision="rewrite_query",
            revised_query="Treasury Bulletin total public debt outstanding 1945 year-end",
            rationale="Use an explicit source-grounded query.",
            confidence=0.91,
        ),
    )
    monkeypatch.setattr("agent.nodes.orchestrator.maybe_rewrite_retrieval_path", lambda **kwargs: None)

    async def _fake_run_tool_step_with_args(state, registry, tool_name, tool_args):
        captured["tool_name"] = tool_name
        captured["tool_args"] = dict(tool_args)
        return tool_args, normalize_tool_output(
            tool_name,
            {"results": [], "documents": [], "metadata": {"officeqa_status": "ok"}},
            tool_args,
        )

    monkeypatch.setattr("agent.nodes.orchestrator._run_tool_step_with_args", _fake_run_tool_step_with_args)

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        review_feedback={
            "verdict": "revise",
            "repair_target": "gather",
            "retry_allowed": True,
            "orchestration_strategy": "table_compute",
            "missing_dimensions": ["time scope correctness"],
            "remediation_codes": ["RETRIEVE_EXACT_PERIOD"],
        },
        source_bundle={
            "task_text": prompt,
            "focus_query": "public debt outstanding 1945",
            "target_period": "1945",
            "entities": ["Public debt"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        curated_context={
            "objective": "Find the exact public debt value for 1945.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
        },
    )
    state["retrieval_intent"] = {
        "entity": "Public debt",
        "metric": "public debt outstanding",
        "period": "1945",
        "granularity_requirement": "point_lookup",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "point_lookup",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "strategy_confidence": 0.7,
        "analysis_modes": [],
        "query_candidates": ["public debt outstanding 1945"],
        "query_plan": {"primary_semantic_query": "public debt outstanding 1945"},
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "GATHER"
    assert captured["tool_name"] == "search_officeqa_documents"
    assert dict(captured["tool_args"])["query"] == "Treasury Bulletin total public debt outstanding 1945 year-end"
    assert result["workpad"]["solver_llm_decision"]["reason"] == "validator_directed_retrieval_repair"
    assert result["workpad"]["officeqa_llm_repair_history"][0]["stage"] == "validator_repair"
    assert result["retrieval_intent"]["query_plan"]["primary_semantic_query"] == "Treasury Bulletin total public debt outstanding 1945 year-end"


def test_officeqa_executor_uses_structured_query_rewrite_for_wrong_document_gap(monkeypatch):
    prompt = "What were the total expenditures of the Veterans Administration in FY 1934?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.nodes.orchestrator._plan_retrieval_action",
        lambda **kwargs: RetrievalAction(
            action="tool",
            tool_name="search_officeqa_documents",
            stage="identify_source",
            strategy="table_first",
            query="Veterans Administration total expenditures 1934",
            evidence_gap="wrong document",
            rationale="Current source match is weak.",
            candidate_sources=[{"document_id": "treasury_bulletin_1959_09_json", "score": 0.58}],
        ),
    )
    monkeypatch.setattr("agent.nodes.orchestrator.maybe_repair_from_validator", lambda **kwargs: None)
    monkeypatch.setattr(
        "agent.nodes.orchestrator.maybe_rewrite_retrieval_path",
        lambda **kwargs: OfficeQALLMRepairDecision(
            decision="rewrite_query",
            revised_query="Treasury Bulletin Veterans Administration total expenditures fiscal year 1934 excluding trust accounts",
            rationale="Clarify entity, year, and exclusion.",
            confidence=0.88,
        ),
    )

    async def _fake_run_tool_step_with_args(state, registry, tool_name, tool_args):
        captured["tool_name"] = tool_name
        captured["tool_args"] = dict(tool_args)
        return tool_args, normalize_tool_output(
            tool_name,
            {"results": [], "documents": [], "metadata": {"officeqa_status": "ok"}},
            tool_args,
        )

    monkeypatch.setattr("agent.nodes.orchestrator._run_tool_step_with_args", _fake_run_tool_step_with_args)

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        source_bundle={
            "task_text": prompt,
            "focus_query": "Veterans Administration total expenditures 1934",
            "target_period": "1934",
            "entities": ["Veterans Administration"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        curated_context={
            "objective": "Find the Veterans Administration total expenditures for FY 1934.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
        },
    )
    state["retrieval_intent"] = {
        "entity": "Veterans Administration",
        "metric": "total expenditures",
        "period": "1934",
        "granularity_requirement": "fiscal_year",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "point_lookup",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "strategy_confidence": 0.61,
        "analysis_modes": [],
        "exclude_constraints": ["trust accounts"],
        "query_candidates": ["Veterans Administration total expenditures 1934"],
        "query_plan": {"primary_semantic_query": "Veterans Administration total expenditures 1934"},
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "GATHER"
    assert captured["tool_name"] == "search_officeqa_documents"
    assert dict(captured["tool_args"])["query"] == "Treasury Bulletin Veterans Administration total expenditures fiscal year 1934 excluding trust accounts"
    assert result["workpad"]["solver_llm_decision"]["reason"] == "structured_retrieval_repair"
    assert result["workpad"]["officeqa_llm_repair_history"][-1]["stage"] == "retrieval_repair"
    assert result["retrieval_intent"]["query_plan"]["primary_semantic_query"] == "Treasury Bulletin Veterans Administration total expenditures fiscal year 1934 excluding trust accounts"


def test_officeqa_validator_repair_invalidates_stale_state_before_new_search(monkeypatch):
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.nodes.orchestrator._plan_retrieval_action",
        lambda **kwargs: RetrievalAction(
            action="tool",
            tool_name="search_officeqa_documents",
            stage="identify_source",
            strategy="table_first",
            query="public debt outstanding 1945",
            evidence_gap="wrong document",
            rationale="Need a stronger source match.",
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator.maybe_repair_from_validator",
        lambda **kwargs: OfficeQALLMRepairDecision(
            decision="rewrite_query",
            revised_query="Treasury Bulletin total public debt outstanding 1945 year-end",
            rationale="Use an explicit source-grounded query.",
            confidence=0.91,
        ),
    )
    monkeypatch.setattr("agent.nodes.orchestrator.maybe_rewrite_retrieval_path", lambda **kwargs: None)

    async def _fake_run_tool_step_with_args(state, registry, tool_name, tool_args):
        captured["tool_name"] = tool_name
        captured["tool_args"] = dict(tool_args)
        return tool_args, normalize_tool_output(
            tool_name,
            {
                "results": [
                    {
                        "document_id": "treasury_bulletin_1945_12_json",
                        "url": "treasury_bulletin_1945_12.json",
                        "title": "Treasury Bulletin 1945-12",
                        "snippet": "Debt outstanding at year end.",
                    }
                ],
                "documents": [],
                "metadata": {"officeqa_status": "ok"},
            },
            tool_args,
        )

    monkeypatch.setattr("agent.nodes.orchestrator._run_tool_step_with_args", _fake_run_tool_step_with_args)

    old_search = normalize_tool_output(
        "search_officeqa_documents",
        {
            "results": [
                {
                    "document_id": "treasury_bulletin_1945_01_json",
                    "url": "treasury_bulletin_1945_01.json",
                    "title": "Treasury Bulletin 1945-01",
                    "snippet": "Older debt bulletin match.",
                }
            ],
            "documents": [],
            "metadata": {"officeqa_status": "ok"},
        },
        {"query": "public debt outstanding 1945"},
    )

    old_table = normalize_tool_output(
        "fetch_officeqa_table",
        {
            "document_id": "treasury_bulletin_1945_01_json",
            "table_locator": "table 1",
            "page_locator": "page 3",
            "headers": ["Issue and page number", "Description"],
            "rows": [["1", "contents"]],
            "metadata": {"officeqa_status": "ok", "document_id": "treasury_bulletin_1945_01_json"},
        },
        {"document_id": "treasury_bulletin_1945_01_json", "table_query": "public debt outstanding"},
    )

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        review_feedback={
            "verdict": "revise",
            "repair_target": "gather",
            "retry_allowed": True,
            "orchestration_strategy": "table_compute",
            "missing_dimensions": ["time scope correctness"],
            "remediation_codes": ["RETRIEVE_EXACT_PERIOD"],
        },
        execution_journal={
            "events": [],
            "tool_results": [old_search.model_dump(), old_table.model_dump()],
            "routed_tool_families": ["document_retrieval"],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 2,
            "retrieval_queries": ["public debt outstanding 1945"],
            "retrieved_citations": ["treasury_bulletin_1945_01.json#page=3"],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        source_bundle={
            "task_text": prompt,
            "focus_query": "public debt outstanding 1945",
            "target_period": "1945",
            "entities": ["Public debt"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        curated_context={
            "objective": "Find the exact public debt value for 1945.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {
                "retrieval_diagnostics": {
                    "retrieval_decision": {"tool_name": "fetch_officeqa_table"},
                    "candidate_sources": [{"document_id": "treasury_bulletin_1945_01_json"}],
                }
            },
            "structured_evidence": {
                "tables": [{"document_id": "treasury_bulletin_1945_01_json", "table_locator": "table 1"}],
                "values": [{"document_id": "treasury_bulletin_1945_01_json", "table_locator": "table 1", "value": "3"}],
                "value_count": 1,
            },
            "compute_result": {"status": "ok", "answer_text": "3"},
        },
        workpad={
            "retrieval_diagnostics": {"candidate_sources": [{"document_id": "treasury_bulletin_1945_01_json"}]},
            "officeqa_compute": {"status": "ok", "answer_text": "3"},
            "officeqa_evidence_review": {"status": "ready"},
        },
    )
    state["retrieval_intent"] = {
        "entity": "Public debt",
        "metric": "public debt outstanding",
        "period": "1945",
        "granularity_requirement": "point_lookup",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "point_lookup",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "strategy_confidence": 0.7,
        "analysis_modes": [],
        "query_candidates": ["public debt outstanding 1945"],
        "query_plan": {"primary_semantic_query": "public debt outstanding 1945"},
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "GATHER"
    assert captured["tool_name"] == "search_officeqa_documents"
    assert dict(captured["tool_args"])["query"] == "Treasury Bulletin total public debt outstanding 1945 year-end"
    assert len(result["execution_journal"]["tool_results"]) == 1
    assert result["execution_journal"]["tool_results"][0]["type"] == "search_officeqa_documents"
    assert result["curated_context"]["compute_result"] == {}
    assert result["curated_context"]["structured_evidence"] == {}
    assert result["workpad"]["officeqa_latest_repair_transition"]["reroute_action"] == "query_rewrite"


def test_officeqa_retrieval_repair_retunes_table_query_and_replaces_stale_table_state(monkeypatch):
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.nodes.orchestrator._plan_retrieval_action",
        lambda **kwargs: RetrievalAction(
            action="tool",
            tool_name="fetch_officeqa_table",
            stage="locate_table",
            strategy="table_first",
            query="expenditures 1940",
            document_id="treasury_bulletin_1941_11_json",
            path="treasury_bulletin_1941_11.json",
            evidence_gap="wrong table family",
            rationale="Current table family is too generic.",
        ),
    )
    monkeypatch.setattr("agent.nodes.orchestrator.maybe_repair_from_validator", lambda **kwargs: None)
    monkeypatch.setattr(
        "agent.nodes.orchestrator.maybe_rewrite_retrieval_path",
        lambda **kwargs: OfficeQALLMRepairDecision(
            decision="retune_table_query",
            revised_table_query="national defense expenditures calendar year 1940",
            rationale="Retarget the exact table family.",
            confidence=0.87,
        ),
    )

    async def _fake_run_tool_step_with_args(state, registry, tool_name, tool_args):
        captured["tool_name"] = tool_name
        captured["tool_args"] = dict(tool_args)
        return tool_args, normalize_tool_output(
            tool_name,
            {
                "document_id": "treasury_bulletin_1941_11_json",
                "table_locator": "table 19",
                "page_locator": "page 29",
                "headers": ["Category", "Calendar year 1940"],
                "rows": [["U.S. national defense", "4748"]],
                "metadata": {"officeqa_status": "ok", "document_id": "treasury_bulletin_1941_11_json"},
            },
            tool_args,
        )

    monkeypatch.setattr("agent.nodes.orchestrator._run_tool_step_with_args", _fake_run_tool_step_with_args)

    search_result = normalize_tool_output(
        "search_officeqa_documents",
        {
            "results": [
                {
                    "document_id": "treasury_bulletin_1941_11_json",
                    "url": "treasury_bulletin_1941_11.json",
                    "title": "Treasury Bulletin 1941-11",
                    "snippet": "Summary expenditures tables.",
                }
            ],
            "documents": [],
            "metadata": {"officeqa_status": "ok"},
        },
        {"query": "national defense expenditures 1940"},
    )
    stale_table = normalize_tool_output(
        "fetch_officeqa_table",
        {
            "document_id": "treasury_bulletin_1941_11_json",
            "table_locator": "table 2",
            "page_locator": "page 4",
            "headers": ["Issue and page number", "Description"],
            "rows": [["2", "summary"]],
            "metadata": {"officeqa_status": "ok", "document_id": "treasury_bulletin_1941_11_json"},
        },
        {"document_id": "treasury_bulletin_1941_11_json", "table_query": "expenditures 1940"},
    )

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        execution_journal={
            "events": [],
            "tool_results": [search_result.model_dump(), stale_table.model_dump()],
            "routed_tool_families": ["document_retrieval"],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 2,
            "retrieval_queries": ["national defense expenditures 1940"],
            "retrieved_citations": ["treasury_bulletin_1941_11.json#page=4"],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        source_bundle={
            "task_text": prompt,
            "focus_query": "national defense expenditures 1940",
            "target_period": "1940",
            "entities": ["U.S. national defense"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents", "fetch_officeqa_table"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        curated_context={
            "objective": "Find the calendar year 1940 national defense expenditures.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "structured_evidence": {
                "tables": [{"document_id": "treasury_bulletin_1941_11_json", "table_locator": "table 2"}],
                "values": [{"document_id": "treasury_bulletin_1941_11_json", "table_locator": "table 2", "value": "2"}],
                "value_count": 1,
            },
            "compute_result": {"status": "ok", "answer_text": "2"},
            "provenance_summary": {},
        },
    )
    state["retrieval_intent"] = {
        "entity": "U.S. national defense",
        "metric": "total expenditures",
        "period": "1940",
        "granularity_requirement": "calendar_year",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "calendar_year_total",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "strategy_confidence": 0.8,
        "analysis_modes": [],
        "query_candidates": ["national defense expenditures 1940"],
        "query_plan": {"primary_semantic_query": "national defense expenditures 1940"},
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "GATHER"
    assert captured["tool_name"] == "fetch_officeqa_table"
    assert dict(captured["tool_args"])["table_query"] == "national defense expenditures calendar year 1940"
    assert [item["type"] for item in result["execution_journal"]["tool_results"]] == ["search_officeqa_documents", "fetch_officeqa_table"]
    assert result["workpad"]["officeqa_latest_repair_transition"]["reroute_action"] == "table_query_rewrite"


def test_officeqa_executor_applies_llm_source_rerank_before_fetch(monkeypatch):
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "agent.nodes.orchestrator._plan_retrieval_action",
        lambda **kwargs: RetrievalAction(
            action="tool",
            tool_name="fetch_officeqa_table",
            stage="locate_table",
            strategy="table_first",
            query="national defense expenditures calendar year 1940",
            document_id="treasury_bulletin_1940_01_json",
            path="treasury_bulletin_1940_01.json",
            evidence_gap="wrong document",
            candidate_sources=[
                {
                    "document_id": "treasury_bulletin_1940_01_json",
                    "path": "treasury_bulletin_1940_01.json",
                    "score": 1.11,
                },
                {
                    "document_id": "treasury_bulletin_1941_11_json",
                    "path": "treasury_bulletin_1941_11.json",
                    "score": 1.02,
                },
            ],
            rationale="Fetch the top-ranked candidate table first.",
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator.select_source_candidate",
        lambda **kwargs: type(
            "ArbiterResult",
            (),
            {
                "used_llm": True,
                "reason": "wrong document",
                "shortlist_count": 2,
                "decision": OfficeQASourceRerankDecision(
                    decision="select_candidate",
                    preferred_document_id="treasury_bulletin_1941_11_json",
                    rationale="Later publication-year summary is a better semantic fit.",
                    confidence=0.83,
                    model_name="rerank-model",
                ),
            },
        )(),
    )
    monkeypatch.setattr("agent.nodes.orchestrator.maybe_rewrite_retrieval_path", lambda **kwargs: None)
    monkeypatch.setattr("agent.nodes.orchestrator.maybe_repair_from_validator", lambda **kwargs: None)

    async def _fake_run_tool_step_with_args(state, registry, tool_name, tool_args):
        captured["tool_name"] = tool_name
        captured["tool_args"] = dict(tool_args)
        return tool_args, normalize_tool_output(
            tool_name,
            {
                "document_id": "treasury_bulletin_1941_11_json",
                "table_locator": "table 19",
                "page_locator": "page 29",
                "headers": ["Category", "Calendar year 1940"],
                "rows": [["U.S. national defense", "4748"]],
                "metadata": {"officeqa_status": "ok", "document_id": "treasury_bulletin_1941_11_json"},
            },
            tool_args,
        )

    monkeypatch.setattr("agent.nodes.orchestrator._run_tool_step_with_args", _fake_run_tool_step_with_args)

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        execution_journal={
            "events": [],
            "tool_results": [],
            "routed_tool_families": ["document_retrieval"],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": ["national defense expenditures 1940"],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        source_bundle={
            "task_text": prompt,
            "focus_query": "national defense expenditures 1940",
            "target_period": "1940",
            "entities": ["U.S. national defense"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        curated_context={
            "objective": "Find the calendar year 1940 national defense expenditures.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "structured_evidence": {},
            "compute_result": {},
            "provenance_summary": {},
        },
    )
    state["retrieval_intent"] = {
        "entity": "U.S. national defense",
        "metric": "total expenditures",
        "period": "1940",
        "period_type": "calendar_year",
        "target_years": ["1940"],
        "publication_year_window": ["1939", "1940", "1941"],
        "preferred_publication_years": ["1941", "1940", "1939"],
        "granularity_requirement": "calendar_year",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "calendar_year_total",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "strategy_confidence": 0.6,
        "analysis_modes": [],
        "semantic_plan": {"used_llm": False},
        "query_candidates": ["national defense expenditures 1940"],
        "query_plan": {"primary_semantic_query": "national defense expenditures 1940"},
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "GATHER"
    assert captured["tool_name"] == "fetch_officeqa_table"
    assert dict(captured["tool_args"])["document_id"] == "treasury_bulletin_1941_11_json"
    assert result["workpad"]["officeqa_llm_usage"][-1]["category"] == "source_arbiter_llm"
    assert result["workpad"]["officeqa_llm_usage"][-1]["applied"] is True


def test_officeqa_retrieval_repair_records_document_pivot_separately_from_llm_path_change(monkeypatch):
    prompt = "What were the total expenditures for U.S. national defense in the calendar year 1940?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)

    monkeypatch.setattr(
        "agent.nodes.orchestrator._plan_retrieval_action",
        lambda **kwargs: RetrievalAction(
            action="tool",
            tool_name="fetch_officeqa_table",
            stage="locate_table",
            strategy="table_first",
            query="national defense expenditures 1940",
            document_id="treasury_bulletin_1940_01_json",
            path="treasury_bulletin_1940_01.json",
            evidence_gap="wrong document",
            rationale="Need a better source candidate.",
        ),
    )
    monkeypatch.setattr("agent.nodes.orchestrator.maybe_repair_from_validator", lambda **kwargs: None)
    monkeypatch.setattr(
        "agent.nodes.orchestrator.maybe_rewrite_retrieval_path",
        lambda **kwargs: OfficeQALLMRepairDecision(
            decision="change_strategy",
            preferred_strategy="table_first",
            rationale="Keep the current retrieval strategy but allow the next table fetch to try a better source.",
            confidence=0.74,
        ),
    )

    async def _fake_run_tool_step_with_args(state, registry, tool_name, tool_args):
        return tool_args, normalize_tool_output(
            tool_name,
            {
                "document_id": "treasury_bulletin_1940_03_json",
                "table_locator": "table 9",
                "page_locator": "page 21",
                "headers": ["Category", "Calendar year 1940"],
                "rows": [["U.S. national defense", "4748"]],
                "metadata": {"officeqa_status": "ok", "document_id": "treasury_bulletin_1940_03_json"},
            },
            tool_args,
        )

    monkeypatch.setattr("agent.nodes.orchestrator._run_tool_step_with_args", _fake_run_tool_step_with_args)

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        execution_journal={
            "events": [],
            "tool_results": [],
            "routed_tool_families": ["document_retrieval"],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": ["national defense expenditures 1940"],
            "retrieved_citations": [],
            "final_artifact_signature": "",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        source_bundle={
            "task_text": prompt,
            "focus_query": "national defense expenditures 1940",
            "target_period": "1940",
            "entities": ["U.S. national defense"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["fetch_officeqa_table"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        curated_context={
            "objective": "Find the calendar year 1940 national defense expenditures.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "structured_evidence": {},
            "compute_result": {},
            "provenance_summary": {},
        },
    )
    state["retrieval_intent"] = {
        "entity": "U.S. national defense",
        "metric": "total expenditures",
        "period": "1940",
        "period_type": "calendar_year",
        "target_years": ["1940"],
        "publication_year_window": ["1939", "1940", "1941"],
        "preferred_publication_years": ["1941", "1940", "1939"],
        "granularity_requirement": "calendar_year",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "calendar_year_total",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "strategy_confidence": 0.7,
        "analysis_modes": [],
        "semantic_plan": {"used_llm": False},
        "query_candidates": ["national defense expenditures 1940"],
        "query_plan": {"primary_semantic_query": "national defense expenditures 1940"},
    }

    result = asyncio.run(executor(state))

    repair_entry = result["workpad"]["officeqa_llm_repair_history"][-1]
    assert repair_entry["stage"] == "retrieval_repair"
    assert repair_entry["path_changed"] is False
    assert repair_entry["llm_path_changed"] is False
    assert repair_entry["document_pivot_triggered"] is True
    assert repair_entry["effective_retrieval_change"] is True
    assert repair_entry["requested_document_id"] == "treasury_bulletin_1940_01_json"
    assert repair_entry["resolved_document_id"] == "treasury_bulletin_1940_03_json"


def test_officeqa_validator_repair_does_not_repeat_without_retrieval_input_change(monkeypatch):
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    executor = make_executor(registry)
    call_count = {"validator": 0}

    monkeypatch.setattr(
        "agent.nodes.orchestrator._plan_retrieval_action",
        lambda **kwargs: RetrievalAction(
            action="tool",
            tool_name="search_officeqa_documents",
            stage="identify_source",
            strategy="table_first",
            query="Treasury Bulletin total public debt outstanding 1945 year-end",
            evidence_gap="wrong document",
            rationale="Need a stronger source match.",
        ),
    )

    def _fake_validator_repair(**kwargs):
        call_count["validator"] += 1
        return OfficeQALLMRepairDecision(
            decision="rewrite_query",
            revised_query="Treasury Bulletin total public debt outstanding 1945 year-end",
            rationale="Use an explicit source-grounded query.",
            confidence=0.91,
        )

    monkeypatch.setattr("agent.nodes.orchestrator.maybe_repair_from_validator", _fake_validator_repair)
    monkeypatch.setattr("agent.nodes.orchestrator.maybe_rewrite_retrieval_path", lambda **kwargs: None)

    async def _fake_run_tool_step_with_args(state, registry, tool_name, tool_args):
        return tool_args, normalize_tool_output(
            tool_name,
            {"results": [], "documents": [], "metadata": {"officeqa_status": "ok"}},
            tool_args,
        )

    monkeypatch.setattr("agent.nodes.orchestrator._run_tool_step_with_args", _fake_run_tool_step_with_args)

    repair_signature = _officeqa_retrieval_input_signature(
        RetrievalIntent(
            entity="Public debt",
            metric="public debt outstanding",
            period="1945",
            granularity_requirement="point_lookup",
            document_family="treasury_bulletin",
            aggregation_shape="point_lookup",
            answer_mode="deterministic_compute",
            compute_policy="required",
            strategy="table_first",
            query_candidates=["Treasury Bulletin total public debt outstanding 1945 year-end"],
            query_plan={"primary_semantic_query": "Treasury Bulletin total public debt outstanding 1945 year-end"},
        ),
        {},
    )

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        review_feedback={
            "verdict": "revise",
            "repair_target": "gather",
            "retry_allowed": True,
            "orchestration_strategy": "table_compute",
            "missing_dimensions": ["time scope correctness"],
            "remediation_codes": ["RETRIEVE_EXACT_PERIOD"],
        },
        source_bundle={
            "task_text": prompt,
            "focus_query": "public debt outstanding 1945",
            "target_period": "1945",
            "entities": ["Public debt"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "exact_compute"],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        workpad={"officeqa_last_validator_repair_signature": repair_signature},
        curated_context={
            "objective": "Find the exact public debt value for 1945.",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
        },
    )
    state["retrieval_intent"] = {
        "entity": "Public debt",
        "metric": "public debt outstanding",
        "period": "1945",
        "granularity_requirement": "point_lookup",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "point_lookup",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "strategy": "table_first",
        "strategy_confidence": 0.7,
        "analysis_modes": [],
        "query_candidates": ["Treasury Bulletin total public debt outstanding 1945 year-end"],
        "query_plan": {"primary_semantic_query": "Treasury Bulletin total public debt outstanding 1945 year-end"},
    }

    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "GATHER"
    assert call_count["validator"] == 0
    assert result["workpad"]["officeqa_repair_failures"][0]["code"] == "repair_reused_stale_state"


def test_officeqa_tool_plan_prefers_native_search_over_generic_reference_search():
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS])
    intent = TaskIntent(
        task_family="document_qa",
        execution_mode="document_grounded_analysis",
        complexity_tier="structured_analysis",
        tool_families_needed=["document_retrieval", "exact_compute"],
        evidence_strategy="document_first",
        review_mode="document_grounded",
        completion_mode="document_grounded",
        confidence=0.95,
        planner_source="heuristic",
    )
    source_bundle = SourceBundle(
        task_text="What was the total public debt outstanding in 1945 according to the Treasury Bulletin?",
        focus_query="total public debt outstanding in 1945",
        target_period="1945",
        entities=["public debt outstanding"],
    )

    plan, _ = resolve_tool_plan(intent, source_bundle, registry, {"benchmark_adapter": "officeqa"})

    assert "search_officeqa_documents" in plan.selected_tools
    assert "search_reference_corpus" not in plan.selected_tools
    assert "search_officeqa_documents" in plan.pending_tools
    assert "search_reference_corpus" not in plan.pending_tools


def test_officeqa_tool_plan_falls_back_to_reference_search_when_native_search_unavailable():
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, search_reference_corpus, fetch_corpus_document_tool])
    intent = TaskIntent(
        task_family="document_qa",
        execution_mode="document_grounded_analysis",
        complexity_tier="structured_analysis",
        tool_families_needed=["document_retrieval"],
        evidence_strategy="document_first",
        review_mode="document_grounded",
        completion_mode="document_grounded",
        confidence=0.95,
        planner_source="heuristic",
    )
    source_bundle = SourceBundle(
        task_text="What was the total public debt outstanding in 1945 according to the Treasury Bulletin?",
        focus_query="total public debt outstanding in 1945",
        target_period="1945",
        entities=["public debt outstanding"],
    )

    plan, _ = resolve_tool_plan(intent, source_bundle, registry, {"benchmark_adapter": "officeqa"})

    assert "search_reference_corpus" in plan.selected_tools
    assert "search_reference_corpus" in plan.pending_tools


def test_officeqa_curated_context_keeps_authoritative_retrieval_state_in_provenance():
    prompt = "What was the total public debt outstanding in 1945 according to the Treasury Bulletin?"
    source_bundle = build_source_bundle(
        prompt,
        {
            "benchmark_adapter": "officeqa",
            "source_files_expected": ["treasury_bulletin_1945_01.json"],
            "source_files_found": [{"requested": "treasury_bulletin_1945_01.json", "matched": True, "document_id": "treasury_bulletin_1945_01_json"}],
        },
    )
    intent = TaskIntent(
        task_family="document_qa",
        execution_mode="document_grounded_analysis",
        complexity_tier="structured_analysis",
        tool_families_needed=["document_retrieval", "exact_compute"],
        evidence_strategy="document_first",
        review_mode="document_grounded",
        completion_mode="document_grounded",
        confidence=0.95,
        planner_source="heuristic",
    )

    curated, _ = build_curated_context(
        prompt,
        {"format": "xml", "requires_adapter": True, "wrapper_key": None, "section_requirements": ["REASONING", "FINAL_ANSWER"]},
        intent,
        source_bundle,
        {"benchmark_adapter": "officeqa"},
    )

    fact_types = {fact.get("type") for fact in curated.facts_in_use}
    assert "source_files_expected" not in fact_types
    assert "source_files_found" not in fact_types
    assert "document_family" not in fact_types
    assert "retrieval_strategy" not in fact_types
    assert "query_candidates" not in fact_types
    assert "evidence_requirements" not in fact_types
    assert curated.provenance_summary["source_bundle"]["source_files_expected"] == ["treasury_bulletin_1945_01.json"]
    assert curated.provenance_summary["retrieval_plan"]["document_family"] == "treasury_bulletin"
    assert curated.provenance_summary["retrieval_plan"]["query_plan"]["primary_semantic_query"]
    assert curated.provenance_summary["retrieval_plan"]["retrieval_seed"]


def test_officeqa_runtime_schema_uses_authoritative_field_owners():
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    source_bundle = SourceBundle(
        task_text=prompt,
        focus_query="total public debt outstanding 1945",
        target_period="1945",
        entities=["Treasury Bulletin"],
        urls=[],
        inline_facts={},
        tables=[],
        formulas=[],
        source_files_expected=["treasury_bulletin_1945_01.json"],
    )
    intent = TaskIntent(
        task_family="document_qa",
        execution_mode="document_grounded_analysis",
        complexity_tier="structured_analysis",
        tool_families_needed=["document_retrieval", "exact_compute"],
        evidence_strategy="document_first",
        review_mode="document_grounded",
        completion_mode="document_grounded",
        routing_rationale="Document questions should ground the answer in retrieved file evidence.",
        planner_source="heuristic",
    )

    curated, _ = build_curated_context(
        prompt,
        {"format": "xml", "requires_adapter": True, "wrapper_key": None, "section_requirements": ["REASONING", "FINAL_ANSWER"]},
        intent,
        source_bundle,
        {"benchmark_adapter": "officeqa"},
    )
    template = task_planner(
        {
            **make_state(prompt, task_intent=intent.model_dump(), benchmark_overrides={"benchmark_adapter": "officeqa"}),
            "task_intent": intent.model_dump(),
        }
    )["execution_template"]

    fact_types = {fact.get("type") for fact in curated.facts_in_use}
    retrieval_plan = curated.provenance_summary["retrieval_plan"]

    assert "focus_query" not in fact_types
    assert "query_candidates" not in retrieval_plan
    assert "answer_focus" not in template
    assert retrieval_plan["query_plan"]["primary_semantic_query"]
    assert retrieval_plan["retrieval_seed"] == retrieval_plan["query_plan"]["primary_semantic_query"]
    assert curated.objective == prompt


def test_fetch_corpus_document_rejects_paths_outside_corpus_root(monkeypatch):
    scratch_parent = Path("results") / "pytest_scratch"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    tmp_root = scratch_parent / f"engine_runtime_{uuid.uuid4().hex}"
    try:
        corpus_root = tmp_root / "corpus"
        corpus_root.mkdir(parents=True)
        (corpus_root / "inside.txt").write_text("inside", encoding="utf-8")
        outside = tmp_root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        monkeypatch.setenv("OFFICEQA_CORPUS_DIR", str(corpus_root))

        result = fetch_corpus_document_tool.invoke({"path": "..\\outside.txt"})

        assert "error" in result
        assert "not found" in result["error"].lower()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_normalize_fetch_reference_file_extracts_pagination_metadata():
    raw = (
        "FILE: treasury_1945.pdf\n"
        "FORMAT: PDF | SIZE: 12.3 KB\n"
        "--------------------------------------------------\n"
        "[Pages 1-2 of 5]\n"
        "Treasury Bulletin overview."
    )

    result = normalize_tool_output(
        "fetch_reference_file",
        raw,
        {"url": "https://example.com/treasury_1945.pdf", "page_start": 0, "page_limit": 2, "row_offset": 0, "row_limit": 200},
    )

    assert result.facts["metadata"]["window_kind"] == "pages"
    assert result.facts["metadata"]["total_pages"] == 5
    assert result.facts["metadata"]["has_more_windows"] is True


def test_engine_executor_paginates_corpus_before_answering(monkeypatch):
    @tool
    def search_reference_corpus(query: str, top_k: int = 5, snippet_chars: int = 700) -> dict:
        """Fake corpus search tool."""
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "treasury_1945.txt",
                    "snippet": "Treasury Bulletin overview and publication notes.",
                    "url": "treasury_1945.txt",
                    "document_id": "treasury_1945_txt",
                }
            ],
            "documents": [
                {
                    "document_id": "treasury_1945_txt",
                    "citation": "treasury_1945.txt",
                    "format": "txt",
                    "path": "treasury_1945.txt",
                }
            ],
        }

    @tool
    def fetch_corpus_document(document_id: str = "", path: str = "", chunk_start: int = 0, chunk_limit: int = 3) -> dict:
        """Fake paginated corpus fetch tool."""
        if chunk_start <= 0:
            return {
                "document_id": document_id or "treasury_1945_txt",
                "citation": path or "treasury_1945.txt",
                "metadata": {
                    "file_name": "treasury_1945.txt",
                    "format": "txt",
                    "window": "chunks 1-1",
                    "total_chunks": 2,
                    "has_more_chunks": True,
                    "chunk_start": 0,
                    "chunk_limit": chunk_limit,
                    "returned_chunks": 1,
                },
                "chunks": [
                    {
                        "locator": "chunk 1",
                        "kind": "text_excerpt",
                        "text": "Treasury Bulletin introduction and editorial notes.",
                        "citation": path or "treasury_1945.txt",
                    }
                ],
                "tables": [],
                "numeric_summaries": [],
            }
        return {
            "document_id": document_id or "treasury_1945_txt",
            "citation": path or "treasury_1945.txt",
            "metadata": {
                "file_name": "treasury_1945.txt",
                "format": "txt",
                "window": "chunks 2-2",
                "total_chunks": 2,
                "has_more_chunks": False,
                "chunk_start": chunk_start,
                "chunk_limit": chunk_limit,
                "returned_chunks": 1,
            },
            "chunks": [
                {
                    "locator": "chunk 2",
                    "kind": "text_excerpt",
                    "text": "In 1945, total public debt outstanding was 258.7 billion dollars.",
                    "citation": path or "treasury_1945.txt",
                }
            ],
            "tables": [],
            "numeric_summaries": [{"metric": "public_debt", "value": 258.7}],
        }

    captured: list = []
    monkeypatch.setattr(
        "agent.nodes.orchestrator.ChatOpenAI",
        lambda **kwargs: _FakeModel(
            AIMessage(content="The total public debt outstanding in 1945 was 258.7 billion dollars. [Source: treasury_1945.txt]"),
            captured,
        ),
    )

    registry = build_capability_registry(
        [CALCULATOR_TOOL, SEARCH_TOOL, search_reference_corpus, fetch_corpus_document, *BUILTIN_LEGAL_TOOLS]
    )
    executor = make_executor(registry)
    state = make_state(
        "According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval", "external_retrieval"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "routing_rationale": "",
            "confidence": 0.85,
            "planner_source": "heuristic",
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval", "external_retrieval"],
            "widened_families": ["document_retrieval", "external_retrieval"],
            "selected_tools": ["search_reference_corpus", "fetch_corpus_document", "internet_search"],
            "pending_tools": ["search_reference_corpus"],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        source_bundle={
            "task_text": "According to the Treasury Bulletin, what was total public debt outstanding in 1945?",
            "focus_query": "total public debt outstanding in 1945",
            "target_period": "1945",
            "entities": ["Treasury Bulletin"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        curated_context={
            "objective": "total public debt outstanding in 1945",
            "facts_in_use": [{"type": "focus_query", "value": "total public debt outstanding in 1945"}],
            "open_questions": ["Find the exact supporting quote before finalizing."],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
        },
    )

    first = asyncio.run(executor(state))
    state.update(first)
    assert first["solver_stage"] == "GATHER"
    assert first["last_tool_result"]["type"] == "search_reference_corpus"

    second = asyncio.run(executor(state))
    state.update(second)
    assert second["solver_stage"] == "GATHER"
    assert second["last_tool_result"]["type"] == "fetch_corpus_document"
    assert second["last_tool_result"]["facts"]["metadata"]["has_more_chunks"] is True

    third = asyncio.run(executor(state))
    state.update(third)
    assert third["solver_stage"] == "GATHER"
    assert third["last_tool_result"]["type"] == "fetch_corpus_document"
    assert third["last_tool_result"]["facts"]["metadata"]["chunk_start"] >= 1

    fourth = asyncio.run(executor(state))
    assert fourth["solver_stage"] == "SYNTHESIZE"
    assert "treasury_1945.txt" in str(fourth["messages"][0].content)
    assert captured


def test_engine_executor_paginates_reference_file_before_answering(monkeypatch):
    @tool
    def fetch_reference_file(
        url: str,
        page_start: int = 0,
        page_limit: int = 5,
        row_offset: int = 0,
        row_limit: int = 200,
        sheet: str | None = None,
        format_hint: str | None = None,
    ) -> str:
        """Fake paginated reference file tool."""
        if page_start <= 0:
            return (
                "FILE: treasury_1945.pdf\n"
                "FORMAT: PDF | SIZE: 12.3 KB\n"
                "--------------------------------------------------\n"
                "[Pages 1-1 of 2]\n"
                "Treasury Bulletin introduction and publication notes."
            )
        return (
            "FILE: treasury_1945.pdf\n"
            "FORMAT: PDF | SIZE: 12.3 KB\n"
            "--------------------------------------------------\n"
            "[Pages 2-2 of 2]\n"
            "In 1945, total public debt outstanding was 258.7 billion dollars."
        )

    captured: list = []
    monkeypatch.setattr(
        "agent.nodes.orchestrator.ChatOpenAI",
        lambda **kwargs: _FakeModel(
            AIMessage(content="The total public debt outstanding in 1945 was 258.7 billion dollars. [Source: treasury_1945.pdf]"),
            captured,
        ),
    )

    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, fetch_reference_file, *BUILTIN_LEGAL_TOOLS])
    executor = make_executor(registry)
    state = make_state(
        "According to the provided Treasury Bulletin PDF, what was total public debt outstanding in 1945?",
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "routing_rationale": "",
            "confidence": 0.85,
            "planner_source": "heuristic",
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval"],
            "widened_families": ["document_retrieval"],
            "selected_tools": ["fetch_reference_file"],
            "pending_tools": ["fetch_reference_file"],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        source_bundle={
            "task_text": "According to the provided Treasury Bulletin PDF, what was total public debt outstanding in 1945?",
            "focus_query": "total public debt outstanding in 1945",
            "target_period": "1945",
            "entities": ["Treasury Bulletin"],
            "urls": ["https://example.com/treasury_1945.pdf"],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        curated_context={
            "objective": "total public debt outstanding in 1945",
            "facts_in_use": [{"type": "focus_query", "value": "total public debt outstanding in 1945"}],
            "open_questions": ["Find the exact supporting quote before finalizing."],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
        },
    )

    first = asyncio.run(executor(state))
    state.update(first)
    assert first["solver_stage"] == "GATHER"
    assert first["last_tool_result"]["type"] == "fetch_reference_file"
    assert first["last_tool_result"]["facts"]["metadata"]["has_more_windows"] is True

    second = asyncio.run(executor(state))
    state.update(second)
    assert second["solver_stage"] == "GATHER"
    assert second["last_tool_result"]["type"] == "fetch_reference_file"
    assert second["last_tool_result"]["assumptions"]["page_start"] >= 1

    third = asyncio.run(executor(state))
    assert third["solver_stage"] == "SYNTHESIZE"
    assert "treasury_1945.pdf" in str(third["messages"][0].content)
    assert captured


def test_engine_executor_stops_cleanly_for_unsupported_artifact_tasks():
    prompt = "Create a .wav file synced to the uploaded drum loop."
    state = make_state(
        prompt,
        task_profile="unsupported_artifact",
        task_intent={
            "task_family": "unsupported_artifact",
            "execution_mode": "advisory_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": [],
            "evidence_strategy": "compact_prompt",
            "review_mode": "qualitative_advisory",
            "completion_mode": "capability_gap",
            "routing_rationale": "",
            "confidence": 0.95,
            "planner_source": "heuristic",
        },
        tool_plan={
            "tool_families_needed": [],
            "widened_families": [],
            "selected_tools": [],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "unsupported_capability",
        },
        curated_context={"objective": prompt, "facts_in_use": [], "open_questions": [], "assumptions": [], "requested_output": {"format": "text"}, "provenance_summary": {}},
        source_bundle={"task_text": prompt, "focus_query": prompt, "target_period": "", "entities": [], "urls": [], "inline_facts": {}, "tables": [], "formulas": []},
    )
    executor = make_executor(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_LEGAL_TOOLS]))
    result = asyncio.run(executor(state))

    assert result["solver_stage"] == "COMPLETE"
    assert "does not support" in str(result["messages"][0].content).lower()


def test_engine_reviewer_collapses_exact_output_to_adapter_instead_of_looping():
    prompt = "Use Gordon Growth Model and return JSON as {\"answer\": <value>}."
    state = make_state(
        prompt,
        task_profile="analytical_reasoning",
        task_intent={
            "task_family": "analytical_reasoning",
            "execution_mode": "advisory_analysis",
            "complexity_tier": "simple_exact",
            "tool_families_needed": ["analytical_reasoning", "exact_compute"],
            "evidence_strategy": "compact_prompt",
            "review_mode": "exact_quant",
            "completion_mode": "scalar_or_json",
            "routing_rationale": "",
            "confidence": 0.9,
            "planner_source": "heuristic",
        },
        answer_contract={"format": "json", "requires_adapter": True, "wrapper_key": "answer"},
        tool_plan={
            "tool_families_needed": [],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents", "fetch_officeqa_table", "lookup_officeqa_rows"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": "abc", "progress_signatures": [], "stop_reason": "", "contract_collapse_attempts": 0},
        curated_context={"objective": prompt, "facts_in_use": [], "open_questions": [], "assumptions": [], "requested_output": {"format": "json", "requires_adapter": True, "wrapper_key": "answer"}, "provenance_summary": {}},
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["messages"].append(AIMessage(content="The fair value under the Gordon Growth Model is $53.00 per share."))

    reviewed = reviewer(state)
    state.update(reviewed)

    assert reviewed["solver_stage"] == "COMPLETE"
    assert reviewed["quality_report"]["stop_reason"] == "exact_output_collapse"
    assert route_from_reviewer(state) == "output_adapter"

    adapted = output_adapter({**state, "messages": state["messages"]})
    assert str(adapted["messages"][0].content).startswith('{"answer":')


def test_engine_executor_dedupes_legal_prompt_and_uses_higher_legal_completion_budget(monkeypatch):
    prompt = (
        "target company we're acquiring has some clean IP but also regulatory compliance gaps in EU and US. "
        "their board wants stock consideration for tax reasons but we can't risk inheriting the compliance liabilities. "
        "deal size is ~$500M, what structure options do we have that could work for both sides? Need to move quickly here."
    )
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_LEGAL_TOOLS]))
    state.update(resolver(state))
    state.update(context_curator(state))
    state["tool_plan"]["pending_tools"] = []
    state["execution_journal"]["tool_results"] = [
        {
            "type": "legal_playbook_retrieval",
            "facts": {"query": prompt, "playbook_points": ["point 1", "point 2"], "urgency": ""},
        },
        {
            "type": "transaction_structure_checklist",
            "facts": {"structures": ["asset purchase", "reverse triangular merger"], "allocation_mechanics": ["escrow"]},
        },
    ]

    captured: list = []
    monkeypatch.setattr(
        "agent.nodes.orchestrator.ChatOpenAI",
        lambda **kwargs: _FakeModel(AIMessage(content="Structured legal answer with multiple options."), captured),
    )

    executor = make_executor(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_LEGAL_TOOLS]))
    result = asyncio.run(executor(state))

    assert result["workpad"]["completion_budget"] >= 1600
    prompt_messages = captured[0]
    serialized = json.dumps([str(msg.content) for msg in prompt_messages], ensure_ascii=True)
    assert serialized.count(prompt) == 1
    assert '"query"' not in serialized
    assert '"tool_results"' not in serialized


def test_engine_reviewer_flags_missing_grounding_for_document_answers():
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["document_retrieval"],
            "evidence_strategy": "document_first",
            "review_mode": "document_grounded",
            "completion_mode": "document_grounded",
            "routing_rationale": "",
            "confidence": 0.84,
            "planner_source": "heuristic",
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_corpus_document",
                    "facts": {
                        "citation": "treasury_1945.txt",
                        "chunks": [{"locator": "chunk 1", "text": "Debt was 258.7 billion dollars.", "citation": "treasury_1945.txt"}],
                    },
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 2,
            "retrieval_queries": ["public debt 1945"],
            "retrieved_citations": ["treasury_1945.txt"],
            "final_artifact_signature": "abc",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        curated_context={"objective": prompt, "facts_in_use": [], "open_questions": [], "assumptions": [], "requested_output": {"format": "text"}, "provenance_summary": {}},
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["messages"].append(AIMessage(content="The total public debt outstanding in 1945 was 258.7 billion dollars."))

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "source attribution" in " ".join(result["review_feedback"]["missing_dimensions"]).lower()


def test_officeqa_reviewer_routes_structured_failure_into_targeted_gather_retry_and_records_validator_result():
    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all "
        "individual calendar months in 1940, what was the absolute percent change of these total sum values?"
    )
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "complex_qualitative",
            "review_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa", "officeqa_xml_contract": True},
        answer_contract={"format": "xml", "requires_adapter": True, "xml_root_tag": "FINAL_ANSWER", "value_rules": {"reasoning_tag": "REASONING", "final_answer_tag": "FINAL_ANSWER"}},
        source_bundle={
            "task_text": prompt,
            "focus_query": "Treasury Bulletin expenditures 1953 1940",
            "target_period": "1953 1940",
            "entities": ["National Defense"],
            "urls": ["https://govinfo.gov/treasury_1953.pdf"],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": [],
            "widened_families": [],
            "selected_tools": ["search_officeqa_documents", "fetch_officeqa_table", "lookup_officeqa_rows"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {
                        "document_id": "treasury_1953_json",
                        "citation": "https://govinfo.gov/treasury_1953.pdf",
                        "tables": [
                            {
                                "citation": "https://govinfo.gov/treasury_1953.pdf",
                                "locator": "table 1",
                                "headers": ["Month", "Expenditures (million dollars)"],
                                "rows": [["January", "10.0"], ["February", "12.0"]],
                                "unit_hint": "million dollars",
                            }
                        ],
                        "metadata": {"officeqa_status": "ok", "file_name": "treasury_1953.pdf"},
                    },
                    "assumptions": {"document_id": "treasury_1953_json", "path": "treasury_1953.json"},
                    "source": {},
                    "quality": {},
                    "retrieval_status": "ok",
                    "evidence_quality_score": 0.9,
                    "errors": [],
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": ["national defense expenditures 1953 1940"],
            "retrieved_citations": ["https://govinfo.gov/treasury_1953.pdf"],
            "final_artifact_signature": "abc",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "xml"},
            "provenance_summary": {
                "source_bundle": {"urls": ["https://govinfo.gov/treasury_1953.pdf"]},
                "structured_evidence": {"table_count": 1, "value_count": 2, "units_seen": ["million"], "provenance_complete": True},
                "compute_result": {"status": "insufficient", "operation": "monthly_sum_percent_change", "validation_errors": ["Missing comparable period totals for 1940 and 1953."], "provenance_complete": False},
            },
            "structured_evidence": {
                "document_evidence": [],
                "tables": [
                    {
                        "document_id": "treasury_1953_json",
                        "citation": "https://govinfo.gov/treasury_1953.pdf",
                        "page_locator": "page 1",
                        "table_locator": "table 1",
                        "headers": ["Month", "Expenditures (million dollars)"],
                        "unit": "million",
                        "unit_multiplier": 1000000.0,
                        "unit_kind": "currency",
                        "row_count": 2,
                        "column_count": 2,
                    }
                ],
                "values": [
                    {
                        "document_id": "treasury_1953_json",
                        "citation": "https://govinfo.gov/treasury_1953.pdf",
                        "page_locator": "page 1",
                        "table_locator": "table 1",
                        "row_index": 0,
                        "row_label": "January",
                        "column_index": 1,
                        "column_label": "Expenditures (million dollars)",
                        "raw_value": "10.0",
                        "numeric_value": 10.0,
                        "normalized_value": 10000000.0,
                        "unit": "million",
                        "unit_multiplier": 1000000.0,
                        "unit_kind": "currency",
                    }
                ],
                "page_chunks": [],
                "units_seen": ["million"],
                "value_count": 1,
                "provenance_complete": True,
            },
            "compute_result": {
                "status": "insufficient",
                "operation": "monthly_sum_percent_change",
                "validation_errors": ["Missing comparable period totals for 1940 and 1953."],
                "citations": [],
                "ledger": [],
                "provenance_complete": False,
            },
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "officeqa_strategy_exhaustion_proof": {
                "strategies_exhausted": True,
                "benchmark_terminal_allowed": True,
                "candidate_universe_exhausted": True,
            },
        },
    )
    state["retrieval_intent"] = {
        "entity": "National Defense",
        "metric": "absolute percent change",
        "period": "1953 1940",
        "document_family": "official_government_finance",
        "aggregation_shape": "monthly_sum_percent_change",
        "must_include_terms": [],
        "must_exclude_terms": [],
        "query_candidates": [],
    }
    state["messages"].append(AIMessage(content="The absolute percent change was 18.2."))

    result = reviewer(state)
    state.update(result)

    assert result["solver_stage"] == "REVISE"
    assert result["quality_report"]["verdict"] == "revise"
    assert result["review_feedback"]["repair_target"] == "gather"
    assert result["review_feedback"]["repair_class"] == "missing_evidence"
    assert result["review_feedback"]["orchestration_strategy"] == "table_compute"
    assert result["review_feedback"]["remediation_codes"]
    assert "deterministic compute support" in result["review_packet"]["validator_result"]["hard_failures"]
    assert result["review_packet"]["validator_result"]["recommended_repair_target"] == "gather"
    assert result["review_packet"]["validator_result"]["orchestration_strategy"] == "table_compute"
    assert result["review_packet"]["validator_result"]["retry_allowed"] is True
    assert result["review_packet"]["validator_result"]["remediation_guidance"]
    assert result["review_packet"]["diagnostic_artifacts"]["compute_diagnostics"]
    assert result["review_packet"]["diagnostic_artifacts"]["validator_codes"]
    assert result["review_packet"]["diagnostic_artifacts"]["validator_orchestration"]
    assert result["review_packet"]["diagnostic_artifacts"]["validator_remediation"]
    assert route_from_reviewer(state) == "executor"


def test_officeqa_reviewer_stops_when_validator_retry_path_is_exhausted():
    prompt = (
        "Using specifically only the reported values for all individual calendar months in 1953 and all "
        "individual calendar months in 1940, what was the absolute percent change of these total sum values?"
    )
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "complex_qualitative",
            "review_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa", "officeqa_xml_contract": True},
        answer_contract={"format": "xml", "requires_adapter": True, "xml_root_tag": "FINAL_ANSWER", "value_rules": {"reasoning_tag": "REASONING", "final_answer_tag": "FINAL_ANSWER"}},
        source_bundle={
            "task_text": prompt,
            "focus_query": "Treasury Bulletin expenditures 1953 1940",
            "target_period": "1953 1940",
            "entities": ["National Defense"],
            "urls": ["https://govinfo.gov/treasury_1953.pdf"],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={"tool_families_needed": [], "widened_families": [], "selected_tools": ["fetch_officeqa_table"], "pending_tools": [], "blocked_families": [], "ace_events": [], "notes": [], "stop_reason": ""},
        execution_journal={
            "events": [],
            "tool_results": [],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": _MAX_RETRIEVAL_HOPS,
            "retrieval_queries": ["national defense expenditures 1953 1940"],
            "retrieved_citations": ["https://govinfo.gov/treasury_1953.pdf"],
            "final_artifact_signature": "abc",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "xml"},
            "provenance_summary": {"source_bundle": {"urls": ["https://govinfo.gov/treasury_1953.pdf"]}},
            "structured_evidence": {},
            "compute_result": {"status": "insufficient", "operation": "monthly_sum_percent_change", "validation_errors": ["Missing comparable period totals for 1940 and 1953."]},
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "officeqa_strategy_exhaustion_proof": {
                "strategies_exhausted": True,
                "benchmark_terminal_allowed": True,
                "parser_or_extraction_gap": True,
            },
        },
    )
    state["retrieval_intent"] = {
        "entity": "National Defense",
        "metric": "absolute percent change",
        "period": "1953 1940",
        "document_family": "official_government_finance",
        "aggregation_shape": "monthly_sum_percent_change",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "partial_answer_allowed": False,
        "strategy": "table_first",
        "analysis_modes": [],
        "evidence_plan": {
            "requires_table_support": True,
            "requires_text_support": False,
            "requires_cross_source_alignment": False,
            "join_keys": [],
        },
        "must_include_terms": [],
        "must_exclude_terms": [],
        "query_candidates": [],
    }
    state["messages"].append(AIMessage(content="The absolute percent change was 18.2."))

    result = reviewer(state)
    state.update(result)

    assert result["solver_stage"] == "COMPLETE"
    assert result["quality_report"]["verdict"] == "fail"
    assert result["quality_report"]["stop_reason"] == "officeqa_retry_exhausted"
    assert route_from_reviewer(state) == "output_adapter"


def test_officeqa_reviewer_emits_safe_insufficiency_answer_for_adapter():
    prompt = "What was the calendar year total for U.S. national defense expenditures in 1940?"
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "review_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa", "officeqa_xml_contract": True},
        answer_contract={"format": "xml", "requires_adapter": True, "xml_root_tag": "FINAL_ANSWER", "value_rules": {"reasoning_tag": "REASONING", "final_answer_tag": "FINAL_ANSWER"}},
        source_bundle={
            "task_text": prompt,
            "focus_query": "national defense expenditures 1940",
            "target_period": "1940",
            "entities": ["National Defense"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={"tool_families_needed": [], "widened_families": [], "selected_tools": [], "pending_tools": [], "blocked_families": [], "ace_events": [], "notes": [], "stop_reason": ""},
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": "abc", "progress_signatures": [], "stop_reason": "", "contract_collapse_attempts": 0},
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "xml"},
            "provenance_summary": {},
            "structured_evidence": {},
            "compute_result": {},
        },
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["retrieval_intent"] = {
        "entity": "National Defense",
        "metric": "total expenditures",
        "period": "1940",
        "document_family": "official_government_finance",
        "aggregation_shape": "calendar_year_total",
        "must_include_terms": [],
        "must_exclude_terms": [],
        "query_candidates": [],
    }
    state["messages"].append(AIMessage(content="The answer was 123.4 million dollars."))

    reviewed = reviewer(state)
    state.update(reviewed)
    adapted = output_adapter({**state, "messages": state["messages"]})
    rendered = str(adapted["messages"][0].content)

    assert reviewed["quality_report"]["stop_reason"] == "officeqa_no_retrieval_repair_path"
    assert "provided Treasury Bulletin evidence" in rendered
    assert ("Cannot calculate" in rendered) or ("Cannot determine" in rendered)
    assert "<FINAL_ANSWER>" in rendered


def test_officeqa_reviewer_replaces_progress_stalled_answer_with_safe_insufficiency(monkeypatch):
    prompt = "What was total public debt outstanding in 1945?"
    monkeypatch.setattr(
        "agent.nodes.orchestrator.assess_evidence_sufficiency",
        lambda *args, **kwargs: EvidenceSufficiency(
            source_family="official_government_document",
            period_scope="match",
            aggregation_type="match",
            entity_scope="match",
            is_sufficient=True,
            missing_dimensions=[],
            rationale="Grounded evidence is otherwise aligned.",
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator.benchmark_validate_final",
        lambda *args, **kwargs: OfficeQAValidationResult(
            verdict="revise",
            reasoning="Structured validation still needs exact period repair.",
            missing_dimensions=["time scope correctness"],
            hard_failures=["time scope correctness"],
            remediation_codes=["RETRIEVE_EXACT_PERIOD"],
            remediation_guidance=["Re-extract the exact requested year before finalization."],
            recommended_repair_target="gather",
            orchestration_strategy="table_compute",
            retry_allowed=True,
            stop_reason="officeqa_structured_revision_required",
            insufficiency_answer=(
                "Structured OfficeQA validation failed because time scope correctness.\n"
                "Final answer: Cannot determine from the provided Treasury Bulletin evidence."
            ),
            replace_answer=False,
        ),
    )
    monkeypatch.setattr(
        "agent.nodes.orchestrator._build_progress_signature",
        lambda **kwargs: ProgressSignature(
            signature="same-signature",
            execution_mode=str(kwargs.get("execution_mode", "")),
            selected_tools=list(kwargs.get("selected_tools", [])),
            missing_dimensions=list(kwargs.get("missing_dimensions", [])),
            artifact_signature=str(kwargs.get("artifact_signature", "")),
            contract_status=str(kwargs.get("contract_status", "")),
        ),
    )

    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "review_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        source_bundle={
            "task_text": prompt,
            "focus_query": "total public debt outstanding 1945",
            "target_period": "1945",
            "entities": ["Treasury Bulletin"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        tool_plan={
            "tool_families_needed": ["document_retrieval"],
            "widened_families": [],
            "selected_tools": ["fetch_officeqa_table"],
            "pending_tools": [],
            "blocked_families": [],
            "ace_events": [],
            "notes": [],
            "stop_reason": "",
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {
                        "document_id": "treasury_bulletin_1945_01_json",
                        "citation": "treasury_bulletin_1945_01.json#page=29",
                        "metadata": {"officeqa_status": "ok"},
                    },
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": ["treasury_bulletin_1945_01.json#page=29"],
            "final_artifact_signature": "same-artifact",
            "progress_signatures": [{"signature": "same-signature"}],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
            "structured_evidence": {"tables": [{"document_id": "treasury_bulletin_1945_01_json"}], "values": []},
            "compute_result": {},
        },
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["retrieval_intent"] = {
        "entity": "Public debt",
        "metric": "total public debt outstanding",
        "period": "1945",
        "document_family": "official_government_finance",
        "aggregation_shape": "point_lookup",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "partial_answer_allowed": False,
        "strategy": "table_first",
        "analysis_modes": [],
        "evidence_plan": {
            "requires_table_support": True,
            "requires_text_support": False,
            "requires_cross_source_alignment": False,
            "join_keys": [],
        },
        "must_include_terms": [],
        "must_exclude_terms": [],
        "query_candidates": [],
    }
    state["messages"].append(AIMessage(content="The answer is 258682."))

    result = reviewer(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["quality_report"]["verdict"] == "fail"
    assert result["quality_report"]["stop_reason"] == "progress_stalled"
    assert "Cannot determine from the provided Treasury Bulletin evidence." in str(result["messages"][0].content)
    assert "258682" not in str(result["messages"][0].content)


def test_officeqa_reviewer_accepts_bounded_partial_answer_when_compute_is_only_preferred(monkeypatch):
    prompt = "Using Treasury Bulletin data, compute the weighted average expenditures for 1953 and explain the supported trend."
    monkeypatch.setattr(
        "agent.nodes.orchestrator.assess_evidence_sufficiency",
        lambda *args, **kwargs: EvidenceSufficiency(
            source_family="official_government_document",
            period_scope="match",
            aggregation_type="missing_monthly_support",
            entity_scope="match",
            is_sufficient=False,
            missing_dimensions=["aggregation semantics"],
            rationale="Weighted average support is incomplete, but grounded trend evidence is present.",
        ),
    )
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "review_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa"},
        source_bundle={
            "task_text": prompt,
            "focus_query": "weighted average expenditures 1953 trend",
            "target_period": "1953",
            "entities": ["Expenditures"],
            "urls": ["treasury_1953.json"],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_pages",
                    "facts": {
                        "document_id": "treasury_1953_json",
                        "citation": "treasury_1953.json",
                        "chunks": [
                            {
                                "locator": "page 4",
                                "text": "Monthly expenditures rose through mid-year before flattening.",
                                "citation": "treasury_1953.json",
                            }
                        ],
                        "metadata": {"officeqa_status": "ok"},
                    },
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": ["treasury_1953.json"],
            "final_artifact_signature": "sig-partial",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "text"},
            "provenance_summary": {},
            "structured_evidence": {
                "tables": [],
                "values": [],
                "page_chunks": [{"document_id": "treasury_1953_json", "citation": "treasury_1953.json", "page_locator": "page 4"}],
                "units_seen": [],
                "value_count": 0,
                "provenance_complete": True,
            },
            "compute_result": {
                "status": "unsupported",
                "operation": "point_lookup",
                "validation_errors": ["Deterministic OfficeQA compute does not yet support aggregation shape 'point_lookup'."],
                "citations": [],
                "ledger": [],
                "provenance_complete": False,
            },
        },
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["retrieval_intent"] = {
        "entity": "Expenditures",
        "metric": "weighted average expenditures",
        "period": "1953",
        "document_family": "official_government_finance",
        "aggregation_shape": "point_lookup",
        "analysis_modes": ["weighted_average", "time_series_forecasting"],
        "answer_mode": "hybrid_grounded",
        "compute_policy": "preferred",
        "partial_answer_allowed": True,
        "must_include_terms": [],
        "must_exclude_terms": [],
        "query_candidates": [],
    }
    state["messages"].append(
        AIMessage(
            content=(
                "Grounded partial answer: based on the retrieved evidence, monthly expenditures rose through mid-year before flattening. "
                "An exact weighted average calculation is not supported by the currently retrieved values, so the remaining unsupported remainder is the exact weighted-average amount. "
                "[Source: treasury_1953.json]"
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["quality_report"]["verdict"] == "pass"
    assert result["review_packet"]["validator_result"]["verdict"] == "pass"


def test_officeqa_reviewer_accepts_deterministic_compute_without_inline_quote(monkeypatch):
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    monkeypatch.setattr(
        "agent.nodes.orchestrator.assess_evidence_sufficiency",
        lambda *args, **kwargs: EvidenceSufficiency(
            source_family="official_government_document",
            period_scope="match",
            aggregation_type="match",
            entity_scope="match",
            is_sufficient=True,
            missing_dimensions=[],
            rationale="Structured evidence is aligned.",
        ),
    )
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "review_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa", "officeqa_xml_contract": True},
        answer_contract={"format": "xml", "requires_adapter": True, "xml_root_tag": "FINAL_ANSWER", "value_rules": {"reasoning_tag": "REASONING", "final_answer_tag": "FINAL_ANSWER"}},
        source_bundle={
            "task_text": prompt,
            "focus_query": "public debt outstanding 1945",
            "target_period": "1945",
            "entities": ["Public debt"],
            "urls": [],
            "inline_facts": {},
            "tables": [],
            "formulas": [],
        },
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {
                        "document_id": "treasury_bulletin_1945_01_json",
                        "citation": "treasury_bulletin_1945_01.json#page=29",
                        "metadata": {"officeqa_status": "ok"},
                    },
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": ["treasury_bulletin_1945_01.json#page=29"],
            "final_artifact_signature": "sig-det",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "xml"},
            "provenance_summary": {"source_bundle": {"urls": []}},
            "structured_evidence": {
                "tables": [{"document_id": "treasury_bulletin_1945_01_json"}],
                "values": [{"document_id": "treasury_bulletin_1945_01_json", "numeric_value": 258682.0}],
                "page_chunks": [],
                "value_count": 1,
                "provenance_complete": True,
            },
            "compute_result": {
                "status": "ok",
                "operation": "point_lookup",
                "answer_text": "258682",
                "citations": ["treasury_bulletin_1945_01.json#page=29"],
                "ledger": [{"operator": "point_lookup", "description": "Direct point lookup", "output": {"value": 258682.0}}],
                "provenance_complete": True,
            },
        },
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["retrieval_intent"] = {
        "entity": "Public debt",
        "metric": "total public debt outstanding",
        "period": "1945",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "point_lookup",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "partial_answer_allowed": False,
        "strategy": "table_first",
        "analysis_modes": [],
        "evidence_plan": {
            "requires_table_support": True,
            "requires_text_support": False,
            "requires_cross_source_alignment": False,
            "join_keys": [],
        },
        "must_include_terms": [],
        "must_exclude_terms": [],
        "query_candidates": [],
    }
    state["messages"].append(AIMessage(content="258682"))

    result = reviewer(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["quality_report"]["verdict"] == "pass"
    assert result["review_packet"]["validator_result"]["verdict"] == "pass"


def test_officeqa_reviewer_rejects_navigational_table_evidence():
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    state = make_state(
        prompt,
        task_profile="document_qa",
        task_intent={
            "task_family": "document_qa",
            "execution_mode": "document_grounded_analysis",
            "complexity_tier": "structured_analysis",
            "review_mode": "document_grounded",
            "planner_source": "heuristic",
        },
        benchmark_overrides={"benchmark_adapter": "officeqa", "officeqa_xml_contract": True},
        answer_contract={"format": "xml", "requires_adapter": True, "xml_root_tag": "FINAL_ANSWER", "value_rules": {"reasoning_tag": "REASONING", "final_answer_tag": "FINAL_ANSWER"}},
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "fetch_officeqa_table",
                    "facts": {
                        "document_id": "treasury_bulletin_1945_01_json",
                        "citation": "treasury_bulletin_1945_01.json#page=6",
                        "metadata": {"officeqa_status": "ok"},
                    },
                }
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "retrieval_iterations": 1,
            "retrieval_queries": [],
            "retrieved_citations": ["treasury_bulletin_1945_01.json#page=6"],
            "final_artifact_signature": "sig-toc",
            "progress_signatures": [],
            "stop_reason": "",
            "contract_collapse_attempts": 0,
        },
        curated_context={
            "objective": prompt,
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "xml"},
            "provenance_summary": {"source_bundle": {"urls": []}},
            "structured_evidence": {
                "tables": [
                    {
                        "document_id": "treasury_bulletin_1945_01_json",
                        "table_locator": "table 1",
                        "headers": ["Row", "Issue and page number | Jan."],
                        "header_rows": [["Articles", "Issue and page number"]],
                    }
                ],
                "values": [
                    {
                        "document_id": "treasury_bulletin_1945_01_json",
                        "citation": "treasury_bulletin_1945_01.json#page=6",
                        "page_locator": "page 6",
                        "table_locator": "table 1",
                        "row_label": "Public debt and guaranteed obligations outstanding",
                        "row_path": ["Public debt and guaranteed obligations outstanding"],
                        "column_label": "Issue and page number | Jan.",
                        "column_path": ["Issue and page number", "Jan."],
                        "raw_value": "3",
                        "numeric_value": 3.0,
                    }
                ],
                "page_chunks": [],
                "value_count": 1,
                "provenance_complete": True,
            },
            "compute_result": {
                "status": "ok",
                "operation": "point_lookup",
                "answer_text": "3",
                "citations": ["treasury_bulletin_1945_01.json#page=6"],
                "ledger": [
                    {
                        "operator": "point_lookup",
                        "description": "Direct point lookup",
                        "output": {"value": 3.0},
                        "provenance_refs": [
                            {
                                "table_locator": "table 1",
                                "row_label": "Public debt and guaranteed obligations outstanding",
                                "column_label": "Issue and page number | Jan.",
                                "column_path": ["Issue and page number", "Jan."],
                                "raw_value": "3",
                            }
                        ],
                    }
                ],
                "provenance_complete": True,
            },
        },
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["retrieval_intent"] = {
        "entity": "Public debt",
        "metric": "total public debt outstanding",
        "period": "1945",
        "document_family": "treasury_bulletin",
        "aggregation_shape": "point_lookup",
        "answer_mode": "deterministic_compute",
        "compute_policy": "required",
        "partial_answer_allowed": False,
        "strategy": "table_first",
        "analysis_modes": [],
        "evidence_plan": {
            "requires_table_support": True,
            "requires_text_support": False,
            "requires_cross_source_alignment": False,
            "join_keys": [],
        },
        "must_include_terms": [],
        "must_exclude_terms": [],
        "query_candidates": [],
    }
    state["messages"].append(AIMessage(content="3"))

    result = reviewer(state)

    assert result["solver_stage"] in {"REVISE", "COMPLETE"}
    assert result["review_packet"]["validator_result"]["verdict"] == "revise"
    assert result["quality_report"]["verdict"] in {"revise", "fail"}
    assert "navigational table selection" in result["review_packet"]["validator_result"]["missing_dimensions"]


def test_officeqa_validator_rejects_semantically_wrong_but_numeric_compute():
    result = validate_officeqa_final(
        task_text="What were the total expenditures for U.S. national defense in the calendar year 1940?",
        retrieval_intent=RetrievalIntent(
            entity="National defense",
            metric="total expenditures",
            period="1940",
            granularity_requirement="calendar_year",
            document_family="treasury_bulletin",
            aggregation_shape="calendar_year_total",
            answer_mode="deterministic_compute",
            compute_policy="required",
            strategy="table_first",
            analysis_modes=[],
            evidence_plan={
                "requires_table_support": True,
                "requires_text_support": False,
                "requires_cross_source_alignment": False,
                "join_keys": [],
            },
        ),
        curated_context={
            "objective": "What were the total expenditures for U.S. national defense in the calendar year 1940?",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "xml"},
            "structured_evidence": {
                "tables": [{"document_id": "treasury_1940_json", "table_locator": "Summary of expenditures", "headers": ["Calendar year 1940"], "header_rows": []}],
                "values": [{"document_id": "treasury_1940_json"}],
                "page_chunks": [],
                "provenance_complete": True,
                "units_seen": ["million dollars"],
            },
            "compute_result": {
                "status": "ok",
                "operation": "calendar_year_total",
                "display_value": "4748",
                "provenance_complete": True,
                "ledger": [{"operator": "calendar_year_total", "description": "1940 calendar-year total", "provenance_refs": [{"citation": "treasury_1940.json"}]}],
                "semantic_diagnostics": {
                    "admissibility_passed": False,
                    "issues": ["wrong row family", "wrong period slice"],
                    "row_family_status": "wrong row family",
                    "period_slice_status": "wrong period slice",
                    "column_family_status": "matched",
                    "aggregation_grain_status": "matched",
                },
            },
        },
        evidence_sufficiency={
            "source_family": "official_government_finance",
            "period_scope": "matched",
            "aggregation_type": "matched",
            "entity_scope": "matched",
            "is_sufficient": True,
            "missing_dimensions": [],
            "rationale": "Looks structurally sufficient.",
        },
        citations=["treasury_1940.json"],
    )

    assert result.verdict == "revise"
    assert "entity/category correctness" in result.hard_failures
    assert "time scope correctness" in result.hard_failures


def test_officeqa_validator_rejects_benchmark_unit_basis_mismatch():
    result = validate_officeqa_final(
        task_text="What were the total expenditures (in millions of nominal dollars) for U.S. national defense in the calendar year 1940?",
        retrieval_intent=RetrievalIntent(
            entity="National defense",
            metric="total expenditures",
            period="1940",
            granularity_requirement="calendar_year",
            expected_answer_unit_basis="millions_nominal_dollars",
            document_family="treasury_bulletin",
            aggregation_shape="calendar_year_total",
            answer_mode="deterministic_compute",
            compute_policy="required",
            strategy="table_first",
            analysis_modes=[],
            evidence_plan={
                "requires_table_support": True,
                "requires_text_support": False,
                "requires_cross_source_alignment": False,
                "join_keys": [],
                "expected_answer_unit_basis": "millions_nominal_dollars",
            },
        ),
        curated_context={
            "objective": "What were the total expenditures (in millions of nominal dollars) for U.S. national defense in the calendar year 1940?",
            "facts_in_use": [],
            "open_questions": [],
            "assumptions": [],
            "requested_output": {"format": "xml"},
            "structured_evidence": {
                "tables": [{"document_id": "treasury_1940_json", "table_locator": "Summary of expenditures", "headers": ["Calendar year 1940"], "header_rows": []}],
                "values": [{"document_id": "treasury_1940_json"}],
                "page_chunks": [],
                "provenance_complete": True,
                "units_seen": ["million dollars"],
            },
            "compute_result": {
                "status": "ok",
                "operation": "calendar_year_total",
                "display_value": "1657000000",
                "answer_unit_basis": "",
                "provenance_complete": True,
                "ledger": [{"operator": "calendar_year_total", "description": "1940 calendar-year total", "provenance_refs": [{"citation": "treasury_1941.json"}]}],
                "semantic_diagnostics": {
                    "admissibility_passed": True,
                    "issues": [],
                    "row_family_status": "matched",
                    "period_slice_status": "matched",
                    "column_family_status": "matched",
                    "aggregation_grain_status": "matched",
                },
            },
        },
        evidence_sufficiency={
            "source_family": "official_government_finance",
            "period_scope": "matched",
            "aggregation_type": "matched",
            "entity_scope": "matched",
            "is_sufficient": True,
            "missing_dimensions": [],
            "rationale": "Looks structurally sufficient.",
        },
        citations=["treasury_1941.json"],
    )

    assert result.verdict == "revise"
    assert "unit consistency" in result.hard_failures
    assert "NORMALIZE_UNITS" in result.remediation_codes


def test_failed_reviewer_path_uses_one_bounded_salvage_pass():
    state = make_state(
        "Need acquisition structure advice.",
        task_intent={
            "task_family": "legal_transactional",
            "execution_mode": "advisory_analysis",
            "complexity_tier": "complex_qualitative",
            "tool_families_needed": [],
            "evidence_strategy": "compact_prompt",
            "review_mode": "qualitative_advisory",
            "completion_mode": "advisory_memo",
            "routing_rationale": "",
            "confidence": 0.9,
            "planner_source": "heuristic",
        },
        solver_stage="COMPLETE",
        quality_report={
            "verdict": "fail",
            "reasoning": "still missing structure coverage",
            "missing_dimensions": ["multiple structure alternatives with tradeoffs"],
            "targeted_fix_prompt": "",
            "score": 0.58,
        },
    )

    assert route_from_reviewer(state) == "self_reflection"


def test_engine_self_reflection_requests_one_extra_legal_deepen_pass(monkeypatch):
    state = make_state(
        "Advise on acquisition structure.",
        task_profile="legal_transactional",
        task_intent={
            "task_family": "legal_transactional",
            "execution_mode": "advisory_analysis",
            "complexity_tier": "complex_qualitative",
            "tool_families_needed": [],
            "evidence_strategy": "compact_prompt",
            "review_mode": "qualitative_advisory",
            "completion_mode": "advisory_memo",
            "routing_rationale": "",
            "confidence": 0.9,
            "planner_source": "heuristic",
        },
        quality_report={"verdict": "pass", "reasoning": "", "missing_dimensions": [], "targeted_fix_prompt": "", "score": 0.9},
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": ""},
        workpad={"events": [], "stage_outputs": {}, "tool_results": []},
    )
    state["messages"].append(AIMessage(content="Recommendation: pursue an asset acquisition. Liability: lower inherited exposure."))

    # LLM self-reflection finds missing items
    reflection_response = '{"score": 0.55, "complete": false, "missing": ["execution-specific next steps", "risk allocation detail"], "improve_prompt": "Add execution-specific next steps and risk-allocation detail."}'
    monkeypatch.setattr(
        "agent.nodes.orchestrator.ChatOpenAI",
        lambda **kwargs: _FakeModel(AIMessage(content=reflection_response)),
    )

    result = self_reflection(state)

    assert result["solver_stage"] == "REVISE"
    assert "next steps" in " ".join(result["review_feedback"]["missing_dimensions"]).lower()


def test_engine_tracer_captures_engine_headers_and_counts(monkeypatch):
    tracer = RunTracer()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "agent.tracer._write_trace_file",
        lambda payload, profile, start_dt: captured.setdefault("payload", payload) or "ok",
    )

    tracer.set_task("Need legal structure options.")
    tracer.record("fast_path_gate", {"task_family": "legal_transactional", "execution_mode": "advisory_analysis", "complexity_tier": "complex_qualitative"})
    tracer.record("task_planner", {"intent": {"task_family": "legal_transactional", "execution_mode": "advisory_analysis", "complexity_tier": "complex_qualitative"}, "template_id": "advisory_analysis"})
    tracer.record("capability_resolver", {"selected_tools": ["legal_playbook_retrieval"], "pending_tools": [], "blocked_families": []})
    tracer.record(
        "executor",
        {
            "intent": {"task_family": "legal_transactional", "execution_mode": "advisory_analysis", "complexity_tier": "complex_qualitative"},
            "used_llm": True,
            "tools_ran": ["legal_playbook_retrieval", "transaction_structure_checklist"],
            "tokens": {"prompt": 100, "completion": 40},
            "output_preview": "answer",
        },
    )
    tracer.finalize("final answer", {"llm_calls": 1}, {"context_tokens_used": 1000})
    payload = captured["payload"]

    assert payload["final_profile"] == "legal_transactional"
    assert payload["final_template"] == "advisory_analysis"
    assert payload["complexity_tier"] == "complex_qualitative"
    assert payload["total_llm_calls"] == 1
    assert payload["total_tool_calls"] == 2


def test_budget_tracker_context_summary_uses_peak_and_total():
    budget = BudgetTracker()
    budget.record_context_tokens(1200)
    budget.record_context_tokens(800)

    summary = budget.summary()

    assert summary["context_tokens_used"] == 1200
    assert summary["context_tokens_total"] == 2000
