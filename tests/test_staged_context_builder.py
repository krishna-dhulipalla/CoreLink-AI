import json

from agent.nodes.context_builder import context_builder
from agent.nodes.intake import intake
from agent.nodes.task_profiler import task_profiler
from agent.nodes.template_selector import template_selector
from staged_test_utils import make_state


class TestContextBuilder:
    def test_extracts_tables_formulas_and_urls(self):
        prompt = """
        ### Formula
        Financial Leverage Effect = (ROE - ROA) / ROA

        ### Related Data
        | Company | ROE | ROA |
        |---|---|---|
        | China Overseas Grand Oceans Group | 3.0433 % | 1.5791 % |

        Output Format: {"answer": <value>}
        Reference file: https://example.com/report.pdf
        """
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))
        state.update(template_selector(state))

        result = context_builder(state)
        evidence = result["evidence_pack"]

        assert result["solver_stage"] == "GATHER"
        assert result["execution_template"]["template_id"] == "quant_with_tool_compute"
        assert evidence["tables"]
        assert any("Financial Leverage Effect" in formula for formula in evidence["formulas"])
        assert evidence["citations"] == ["https://example.com/report.pdf"]
        assert evidence["document_evidence"][0]["status"] == "discovered"
        assert evidence["document_evidence"][0]["metadata"]["format"] == "pdf"
        assert evidence["answer_contract"]["format"] == "json"
        assert evidence["relevant_rows"]
        first_row = evidence["relevant_rows"][0]["rows"][0]
        assert "China Overseas Grand Oceans Group" in json.dumps(first_row)
        assert result["workpad"]["task_complexity_tier"] == "structured_analysis"
        assert "Use formulas and inline tables from the prompt before calling tools." in evidence["constraints"]
        assert any("metadata or a narrow page/row window first" in item for item in evidence["constraints"])
        assert "relevant_rows" in result["provenance_map"]
        assert "relevant_formulae" in result["provenance_map"]
        assert "document_evidence.report_pdf.metadata.citation" in result["provenance_map"]
        assert result["workpad"]["profile_pack"]["profile"] == "finance_quant"
        assert state["budget_tracker"].summary()["complexity_tier"] == "structured_analysis"

    def test_derives_options_market_signals(self):
        prompt = (
            "META's current IV is 35% while its 30-day historical volatility is 28%. "
            "The IV percentile is 75%."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))
        state.update(template_selector(state))

        result = context_builder(state)
        evidence = result["evidence_pack"]

        assert evidence["prompt_facts"]["market_snapshot"]["implied_volatility"] == 0.35
        assert evidence["prompt_facts"]["market_snapshot"]["historical_volatility"] == 0.28
        assert evidence["derived_facts"]["iv_premium"] == 0.07
        assert evidence["derived_facts"]["vol_bias"] == "short_vol"
        assert result["execution_template"]["template_id"] == "options_tool_backed"
        assert any("Spot price is not explicit" in question for question in evidence["open_questions"])
        assert "Recommendation" in result["answer_contract"]["section_requirements"]

    def test_extracts_finance_policy_context_from_prompt(self):
        prompt = (
            "META's current IV is 35% while its 30-day historical volatility is 28%. "
            "The IV percentile is 75%. This is a retirement account mandate: defined-risk only, "
            "no naked options, and keep position risk to 2% of capital. "
            "Should you be a net buyer or seller of options? Design a compliant strategy accordingly."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))
        state.update(template_selector(state))

        result = context_builder(state)
        policy = result["evidence_pack"]["policy_context"]

        assert policy["action_orientation"] is True
        assert policy["defined_risk_only"] is True
        assert policy["no_naked_options"] is True
        assert policy["retail_or_retirement_account"] is True
        assert policy["max_position_risk_pct"] == 2.0
        assert "requires_timestamped_evidence" not in policy
        assert "policy_context.defined_risk_only" in result["provenance_map"]

    def test_extracts_as_of_date_into_prompt_facts(self):
        prompt = (
            "As of Oct 14, 2022, compare the implied volatility setup for META and keep any "
            "market-data tool usage bound to that date."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))
        state.update(template_selector(state))

        result = context_builder(state)
        evidence = result["evidence_pack"]

        assert evidence["prompt_facts"]["as_of_date"] == "2022-10-14"
        assert evidence["prompt_facts"]["market_snapshot"]["as_of_date"] == "2022-10-14"
        assert evidence["derived_facts"]["time_sensitive"] is True

    def test_live_data_finance_quant_starts_in_gather(self):
        prompt = (
            "As of 2024-10-14, use finance evidence tools to retrieve MSFT price history and 1-month return, "
            "then summarize the result with the source timestamp and any missing-data caveats."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))
        state.update(template_selector(state))

        result = context_builder(state)

        assert result["task_profile"] == "finance_quant"
        assert "needs_live_data" in result["capability_flags"]
        assert result["execution_template"]["template_id"] == "quant_with_tool_compute"
        assert result["solver_stage"] == "GATHER"

    def test_ambiguous_profile_adds_conservative_constraint(self):
        prompt = (
            "We need acquisition structure advice and also a quick valuation ratio calculation from a file "
            "at https://example.com/deal.pdf."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))
        state.update(template_selector(state))

        result = context_builder(state)
        evidence = result["evidence_pack"]

        assert "Task profile is partially ambiguous" in " ".join(evidence["constraints"])
        assert result["ambiguity_flags"]
        assert "time_sensitive" not in evidence["derived_facts"]

    def test_legal_prompt_without_source_document_does_not_fabricate_retrieved_provenance(self):
        prompt = (
            "We need acquisition structure advice for a fast stock-for-stock deal, "
            "but there is no file or external source attached."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))
        state.update(template_selector(state))

        result = context_builder(state)

        assert result["evidence_pack"]["retrieved_facts"] == {}
        assert all(
            value["source_class"] != "retrieved"
            for value in result["provenance_map"].values()
        )
        assert result["checkpoint_stack"] == []

    def test_document_template_seeds_selective_artifact_checkpoint(self):
        prompt = "Read the attached report at https://example.com/report.pdf and summarize it."
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))
        state.update(template_selector(state))

        result = context_builder(state)

        assert len(result["checkpoint_stack"]) == 1
        checkpoint = result["checkpoint_stack"][0]
        assert checkpoint["template_id"] == "document_qa"
        assert checkpoint["checkpoint_stage"] == "GATHER"
