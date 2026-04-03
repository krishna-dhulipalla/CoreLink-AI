import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path

import agent.graph as graph_module
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent.budget import BudgetTracker
from agent.benchmarks.officeqa_index import build_officeqa_index
from agent.contracts import EvidenceSufficiency, ExecutionJournal, SourceBundle, ToolPlan
from agent.officeqa_structured_evidence import build_officeqa_structured_evidence
from agent.retrieval_tools import fetch_corpus_document as fetch_corpus_document_tool
from agent.retrieval_tools import fetch_officeqa_table, lookup_officeqa_cells
from agent.retrieval_reasoning import assess_evidence_sufficiency, build_retrieval_intent, predictive_evidence_gaps
from agent.tools.normalization import normalize_tool_output
from agent.nodes.intake import intake
from agent.nodes.output_adapter import output_adapter
from agent.tracer import RunTracer
from agent.capabilities import BUILTIN_LEGAL_TOOLS, BUILTIN_RETRIEVAL_TOOLS, build_capability_registry, filter_registry_for_benchmark
from agent.curated_context import attach_structured_evidence, solver_context_block
from agent.nodes.orchestrator_retrieval import _plan_retrieval_action
from agent.nodes.orchestrator import (
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
from agent.prompts import build_revision_prompt
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


def _exact_quant_prompt() -> str:
    return """
    ### <Formula List>
    Financial Leverage Effect = (ROE - ROA) / ROA

    ### <Related Data>
    | Stock Name | Return on Equity (ROE) (2024 Annual Report) (%) |
    |---|---|
    | China Overseas Grand Oceans Group | 3.0433 % |

    | Stock Name | Return on Assets (ROA) (2024 Annual Report) (%) |
    |---|---|
    | China Overseas Grand Oceans Group | 1.5790 % |

    ### <User Question>
    Please calculate: What is the Financial Leverage Effect (2024 Annual Report) for China Overseas Grand Oceans Group?

    ### Output Format
    {"answer": <value>}
    """


def test_engine_exact_quant_fast_path_curates_without_duplicate_solver_payload():
    state = make_state(_exact_quant_prompt())
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_LEGAL_TOOLS]))
    state.update(resolver(state))
    result = context_curator(state)

    assert result["curated_context"]["objective"]
    assert result["evidence_pack"].keys() == {"curated_context", "source_bundle_summary"}
    assert "constraints" not in json.dumps(result["evidence_pack"], ensure_ascii=True)
    assert "formulas" not in result["evidence_pack"]
    assert result["workpad"]["task_complexity_tier"] == "simple_exact"
    assert any(fact["type"] == "formula" for fact in result["curated_context"]["facts_in_use"])
    assert any(fact["type"] == "table_rows" for fact in result["curated_context"]["facts_in_use"])


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


def test_engine_legal_planner_binds_legal_capability_tools_not_calculator_only():
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
    result = resolver(state)

    assert result["tool_plan"]["selected_tools"]
    assert "calculator" not in result["tool_plan"]["selected_tools"]


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
            },
            {
                "title": "[PDF] annual report - administrator of veterans' affairs",
                "snippet": "Annual report for fiscal year 1934.",
                "citation": "https://www.va.gov/vetdata/docs/FY1934.pdf",
                "path": "",
                "document_id": "",
                "rank": 3,
            },
        ],
        retrieval_intent,
        source_bundle,
        {"benchmark_adapter": "officeqa"},
    )

    assert "veterans" in ranked[0]["title"].lower()
    assert "depository invoice" in ranked[-1]["title"].lower()


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
    february_value = next(
        item
        for item in structured["values"]
        if item["row_label"] == "February" and item["column_label"] == "Expenditures (million dollars)"
    )
    assert february_value["numeric_value"] == 101.5
    assert february_value["normalized_value"] == 101_500_000.0
    assert february_value["table_locator"]
    assert february_value["page_locator"]


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


def test_engine_executor_returns_deterministic_exact_quant_answer():
    state = make_state(_exact_quant_prompt())
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_LEGAL_TOOLS]))
    state.update(resolver(state))
    state.update(context_curator(state))

    executor = make_executor(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_LEGAL_TOOLS]))
    result = asyncio.run(executor(state))

    assert result["messages"]
    assert str(result["messages"][0].content).startswith('{"answer":')
    assert result["workpad"]["review_ready"] is True


def test_engine_capability_resolver_widens_live_finance_retrieval_to_search_tool():
    prompt = "What was AAPL's EBITDA in fiscal year 2024? Use current source-backed data."
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_LEGAL_TOOLS]))
    result = resolver(state)

    assert "market_data_retrieval" in result["tool_plan"]["widened_families"]
    assert "external_retrieval" in result["tool_plan"]["widened_families"]
    assert "internet_search" in result["tool_plan"]["selected_tools"]
    assert result["tool_plan"]["stop_reason"] == ""


def test_engine_document_query_prefers_document_grounded_retrieval_tools():
    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(
        build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_RETRIEVAL_TOOLS, *BUILTIN_LEGAL_TOOLS])
    )
    result = resolver(state)

    assert result["execution_template"]["template_id"] == "document_grounded_analysis"
    assert "document_retrieval" in result["tool_plan"]["widened_families"]
    assert "search_reference_corpus" in result["tool_plan"]["selected_tools"]


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
    assert retrieval_intent.compute_policy == "preferred"
    assert retrieval_intent.partial_answer_allowed is True
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
    assert retrieval_intent.answer_mode == "grounded_synthesis"
    assert retrieval_intent.compute_policy == "preferred"
    assert retrieval_intent.partial_answer_allowed is True
    assert retrieval_intent.evidence_plan.requires_inflation_support is True
    assert retrieval_intent.evidence_plan.join_keys
    assert "join_ready_support" in {item.kind for item in retrieval_intent.evidence_plan.requirements}


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


def test_engine_document_query_uses_external_document_tools_with_inferred_roles(monkeypatch):
    @tool
    def search_treasury_bulletins(query: str, top_k: int = 5) -> dict:
        """Search the Treasury Bulletin corpus for matching documents."""
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "treasury_1945.pdf",
                    "snippet": "Treasury Bulletin result for 1945 debt.",
                    "url": "https://example.com/treasury_1945.pdf",
                    "document_id": "treasury_1945_pdf",
                }
            ],
            "documents": [
                {
                    "document_id": "treasury_1945_pdf",
                    "citation": "https://example.com/treasury_1945.pdf",
                    "format": "pdf",
                }
            ],
        }

    @tool
    def read_treasury_bulletin(
        document_id: str = "",
        url: str = "",
        page_start: int = 0,
        page_limit: int = 5,
        row_offset: int = 0,
        row_limit: int = 200,
    ) -> str:
        """Read a Treasury Bulletin PDF by document id or URL."""
        if page_start <= 0:
            return (
                "FILE: treasury_1945.pdf\n"
                "FORMAT: PDF | SIZE: 12.3 KB\n"
                "--------------------------------------------------\n"
                "[Pages 1-1 of 2]\n"
                "Treasury Bulletin overview and publication notes."
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

    prompt = "According to the Treasury Bulletin, what was total public debt outstanding in 1945?"
    registry = build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, search_treasury_bulletins, read_treasury_bulletin, *BUILTIN_LEGAL_TOOLS])
    state = make_state(prompt)
    state.update(intake(state))
    state.update(fast_path_gate(state))
    state.update(task_planner(state))
    resolver = make_capability_resolver(registry)
    state.update(resolver(state))
    state.update(context_curator(state))

    assert "search_treasury_bulletins" in state["tool_plan"]["selected_tools"]
    assert "read_treasury_bulletin" in state["tool_plan"]["selected_tools"]
    assert "search_treasury_bulletins" in state["tool_plan"]["pending_tools"]

    executor = make_executor(registry)

    first = asyncio.run(executor(state))
    state.update(first)
    assert first["solver_stage"] == "GATHER"
    assert first["last_tool_result"]["type"] == "search_treasury_bulletins"

    second = asyncio.run(executor(state))
    state.update(second)
    assert second["solver_stage"] == "GATHER"
    assert second["last_tool_result"]["type"] == "read_treasury_bulletin"
    assert second["last_tool_result"]["assumptions"]["document_id"] == "treasury_1945_pdf"

    third = asyncio.run(executor(state))
    state.update(third)
    assert third["solver_stage"] == "GATHER"
    assert third["last_tool_result"]["type"] == "read_treasury_bulletin"
    assert third["last_tool_result"]["assumptions"]["page_start"] >= 1

    result = None
    for _ in range(3):
        result = asyncio.run(executor(state))
        state.update(result)
        if result["solver_stage"] == "SYNTHESIZE":
            break

    assert result is not None
    assert result["solver_stage"] == "SYNTHESIZE"
    assert "treasury_1945.pdf" in str(result["messages"][0].content)
    assert captured


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
                    "text": "Category,Amount\nTotal public debt outstanding,258.7",
                    "citation": path or "treasury_1945.json",
                }
            ],
            "tables": [
                {
                    "locator": "table 1",
                    "headers": ["Category", "Amount"],
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
    assert "treasury_1945.json" in str(third["messages"][0].content)
    assert captured


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
        tool_plan={"tool_families_needed": [], "widened_families": [], "selected_tools": [], "pending_tools": [], "blocked_families": [], "ace_events": [], "notes": [], "stop_reason": ""},
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


def test_engine_executor_uses_deterministic_options_final_without_llm(monkeypatch):
    prompt = (
        "META's current IV is 35% while its 30-day historical volatility is 28%. "
        "The IV percentile is 75%. Should you be a net buyer or seller of options? Design a strategy accordingly."
    )
    state = make_state(
        prompt,
        task_profile="finance_options",
        task_intent={
            "task_family": "finance_options",
            "execution_mode": "tool_compute",
            "complexity_tier": "structured_analysis",
            "tool_families_needed": ["options_strategy_analysis", "options_scenario_analysis"],
            "evidence_strategy": "compact_prompt",
            "review_mode": "tool_compute",
            "completion_mode": "compact_sections",
            "routing_rationale": "",
            "confidence": 0.95,
            "planner_source": "fast_path",
        },
        tool_plan={"tool_families_needed": [], "selected_tools": [], "pending_tools": [], "blocked_families": [], "ace_events": [], "notes": []},
        source_bundle={"task_text": prompt, "focus_query": prompt, "target_period": "", "entities": ["META"], "urls": [], "inline_facts": {"implied_volatility": 0.35, "historical_volatility": 0.28, "iv_percentile": 75.0}, "tables": [], "formulas": []},
        curated_context={"objective": prompt, "facts_in_use": [], "open_questions": [], "assumptions": [], "requested_output": {"format": "text"}, "provenance_summary": {}},
        execution_journal={
            "events": [],
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 23.98,
                        "premium_direction": "credit",
                        "total_delta": -0.073,
                        "total_gamma": -0.026,
                        "total_theta_per_day": 0.439,
                        "total_vega_per_vol_point": -0.683,
                        "max_loss": 9999.0,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "call", "action": "sell", "K": 300.0},
                            {"option_type": "put", "action": "sell", "K": 300.0},
                        ],
                        "reference_price": 300.0,
                    },
                },
                {
                    "type": "scenario_pnl",
                    "facts": {"worst_case_pnl": -22.26, "best_case_pnl": 0.44},
                    "assumptions": {"reference_price": 300.0},
                },
            ],
            "routed_tool_families": [],
            "revision_count": 0,
            "self_reflection_count": 0,
            "final_artifact_signature": "",
        },
    )
    monkeypatch.setattr("agent.nodes.orchestrator.ChatOpenAI", lambda **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not run")))

    executor = make_executor(build_capability_registry([CALCULATOR_TOOL, SEARCH_TOOL, *BUILTIN_LEGAL_TOOLS]))
    result = asyncio.run(executor(state))
    answer = str(result["messages"][0].content)

    assert "defined-risk only" not in answer.lower()
    assert "- delta: -0.073" in answer.lower()
    assert "- gamma: -0.026" in answer.lower()
    assert "- vega: -0.683" in answer.lower()


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


def test_officeqa_reviewer_blocks_structured_failure_before_reflection_and_records_validator_result():
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
        tool_plan={"tool_families_needed": [], "widened_families": [], "selected_tools": [], "pending_tools": [], "blocked_families": [], "ace_events": [], "notes": [], "stop_reason": ""},
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
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
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

    assert result["solver_stage"] == "COMPLETE"
    assert result["quality_report"]["verdict"] == "fail"
    assert result["quality_report"]["stop_reason"] == "officeqa_structured_validation_failed"
    assert "deterministic compute support" in result["review_packet"]["validator_result"]["hard_failures"]
    assert result["review_packet"]["validator_result"]["remediation_guidance"]
    assert result["review_packet"]["diagnostic_artifacts"]["compute_diagnostics"]
    assert result["review_packet"]["diagnostic_artifacts"]["validator_remediation"]
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

    assert reviewed["quality_report"]["stop_reason"] == "officeqa_structured_validation_failed"
    assert "provided Treasury Bulletin evidence" in rendered
    assert ("Cannot calculate" in rendered) or ("Cannot determine" in rendered)
    assert "<FINAL_ANSWER>" in rendered


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


def test_engine_reviewer_revises_legal_once_then_stops_cleanly():
    prompt = (
        "Need acquisition structure options with stock consideration, liability protection, and cross-border compliance handling."
    )
    state = make_state(
        prompt,
        task_profile="legal_transactional",
        task_intent={
            "task_family": "legal_transactional",
            "execution_mode": "advisory_analysis",
            "complexity_tier": "complex_qualitative",
            "tool_families_needed": ["transaction_structure_checklist"],
            "evidence_strategy": "compact_prompt",
            "review_mode": "qualitative_advisory",
            "completion_mode": "advisory_memo",
            "routing_rationale": "",
            "confidence": 0.9,
            "planner_source": "heuristic",
        },
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": ""},
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["messages"].append(AIMessage(content="Recommendation: use an asset deal. Tax: stock may help deferral. Liability: use indemnities. Next steps: move quickly."))

    first = reviewer(state)
    assert first["solver_stage"] == "REVISE"
    assert "liability allocation" in " ".join(first["review_feedback"]["missing_dimensions"]).lower()

    state["execution_journal"]["revision_count"] = 1
    second = reviewer(state)
    assert second["solver_stage"] == "COMPLETE"
    assert second["quality_report"]["verdict"] == "fail"


def test_engine_reviewer_rejects_single_recommended_legal_structure():
    prompt = "Need structure options for a cross-border acquisition with stock consideration and compliance risk."
    state = make_state(
        prompt,
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
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": ""},
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Recommended structure: reverse triangular merger. "
                "Use escrow and indemnities. Stock consideration helps tax deferral."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "multiple structure alternatives" in " ".join(result["review_feedback"]["missing_dimensions"]).lower()


def test_engine_reviewer_flags_when_option_snapshot_is_not_front_loaded():
    prompt = "Need structure options for a cross-border acquisition with stock consideration and compliance risk."
    repetitive_prefix = "Recommended structure: reverse triangular merger. " * 80
    state = make_state(
        prompt,
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
        execution_journal={"events": [], "tool_results": [], "routed_tool_families": [], "revision_count": 0, "self_reflection_count": 0, "final_artifact_signature": ""},
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "review_ready": True},
    )
    state["messages"].append(
        AIMessage(
            content=(
                repetitive_prefix
                + "\n\nStructure options: reverse triangular merger, asset purchase, carve-out transaction."
                + "\nTax consequences: tax-free reorganization under section 368 can give target shareholders tax deferral, buyer basis step-up can arise in an asset deal, required elections matter, and tax treatment breaks if qualification conditions fail."
                + "\nLiability protection: indemnities, escrow, caps, baskets, survival periods, and disclosure schedules."
                + "\nRegulatory and diligence risks: EU and US approvals, employee-transfer consultation, closing conditions, and remediation timing."
                + "\nKey open questions and assumptions: severity of compliance gaps, cure feasibility, seller indemnity support."
                + "\nRecommended next steps: week-one workplan with owners, sequencing, and accelerated diligence."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "opening summary with multiple viable paths" in " ".join(result["review_feedback"]["missing_dimensions"]).lower()


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


def test_engine_legal_revision_prompt_requires_front_loaded_snapshot():
    prompt = build_revision_prompt(
        ["opening summary with multiple viable paths before the deep dive", "regulatory execution specifics"],
        improve_hint="Add approvals, consultation timing, and closing conditions.",
        task_family="legal_transactional",
    )

    assert "snapshot" in prompt.lower()
    assert "multiple viable structures" in prompt.lower()


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
