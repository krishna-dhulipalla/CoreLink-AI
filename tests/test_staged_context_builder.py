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
        assert evidence["file_refs"] == ["https://example.com/report.pdf"]
        assert evidence["answer_contract"]["format"] == "json"
        assert "Use formulas and inline tables from the prompt before calling tools." in evidence["constraints"]
        assert result["workpad"]["profile_pack"]["profile"] == "finance_quant"

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

        assert evidence["market_snapshot"]["implied_volatility"] == 0.35
        assert evidence["market_snapshot"]["historical_volatility"] == 0.28
        assert evidence["derived_signals"]["iv_premium"] == 0.07
        assert evidence["derived_signals"]["vol_bias"] == "short_vol"
        assert result["execution_template"]["template_id"] == "options_tool_backed"
        assert "Recommendation" in result["answer_contract"]["section_requirements"]

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
