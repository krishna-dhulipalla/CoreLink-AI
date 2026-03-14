from agent.nodes.context_builder import context_builder
from agent.nodes.intake import intake
from agent.nodes.task_profiler import task_profiler
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

        result = context_builder(state)
        evidence = result["evidence_pack"]

        assert result["solver_stage"] == "GATHER"
        assert evidence["tables"]
        assert any("Financial Leverage Effect" in formula for formula in evidence["formulas"])
        assert evidence["file_refs"] == ["https://example.com/report.pdf"]
        assert evidence["answer_contract"]["format"] == "json"

    def test_derives_options_market_signals(self):
        prompt = (
            "META's current IV is 35% while its 30-day historical volatility is 28%. "
            "The IV percentile is 75%."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))

        result = context_builder(state)
        evidence = result["evidence_pack"]

        assert evidence["market_snapshot"]["implied_volatility"] == 0.35
        assert evidence["market_snapshot"]["historical_volatility"] == 0.28
        assert evidence["derived_signals"]["iv_premium"] == 0.07
        assert evidence["derived_signals"]["vol_bias"] == "short_vol"
