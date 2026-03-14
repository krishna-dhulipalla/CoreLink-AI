from agent.nodes.intake import intake
from agent.nodes.task_profiler import task_profiler
from staged_test_utils import make_state


class TestStagedProfiler:
    def test_legal_structure_options_stays_legal(self):
        prompt = (
            "Target company we're acquiring has compliance gaps in EU and US. "
            "The board wants stock consideration for tax reasons. What structure options do we have?"
        )
        state = make_state(prompt)
        state.update(intake(state))

        result = task_profiler(state)

        assert result["task_profile"] == "legal_transactional"
        assert "needs_legal_reasoning" in result["capability_flags"]
        assert "needs_options_engine" not in result["capability_flags"]

    def test_options_prompt_maps_to_finance_options(self):
        prompt = (
            "META's current IV is 35% while its 30-day historical volatility is 28%. "
            "The IV percentile is 75%. Should you be a net buyer or seller of options?"
        )
        state = make_state(prompt)
        state.update(intake(state))

        result = task_profiler(state)

        assert result["task_profile"] == "finance_options"
        assert "needs_options_engine" in result["capability_flags"]
        assert "needs_math" not in result["capability_flags"]

    def test_current_data_prompt_maps_to_external_retrieval(self):
        prompt = "Look up the latest SEC filing for META and cite the source."
        state = make_state(prompt)
        state.update(intake(state))

        result = task_profiler(state)

        assert result["task_profile"] == "external_retrieval"
        assert "needs_live_data" in result["capability_flags"]

    def test_quant_tables_do_not_false_positive_as_legal_or_live_data(self):
        prompt = (
            "Financial Leverage Effect = (ROE - ROA) / ROA\n"
            "| Company | ROE | ROA |\n"
            "|---|---|---|\n"
            "| China Overseas Grand Oceans Group | 3.0433 % | 1.5791 % |\n"
            "Quick Ratio = (Total Current Assets - Inventory) / Total Current Liabilities\n"
            'Output Format: {"answer": <value>}'
        )
        state = make_state(prompt)
        state.update(intake(state))

        result = task_profiler(state)

        assert result["task_profile"] == "finance_quant"
        assert "needs_legal_reasoning" not in result["capability_flags"]
        assert "needs_live_data" not in result["capability_flags"]
