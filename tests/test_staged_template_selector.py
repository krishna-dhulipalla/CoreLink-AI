from agent.nodes.intake import intake
from agent.nodes.task_profiler import task_profiler
from agent.nodes.template_selector import template_selector
from staged_test_utils import make_state


class TestTemplateSelector:
    def test_selects_quant_inline_exact_for_exact_output_prompt(self):
        prompt = (
            "Calculate operating margin given revenue 100 and operating income 20. "
            'Output Format: {"answer": <value>}'
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))

        result = template_selector(state)

        assert result["execution_template"]["template_id"] == "quant_inline_exact"

    def test_same_quant_profile_can_select_tool_compute_without_exact_contract(self):
        prompt = "Calculate operating margin given revenue 100 and operating income 20."
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))

        result = template_selector(state)

        assert state["task_profile"] == "finance_quant"
        assert result["execution_template"]["template_id"] == "quant_with_tool_compute"

    def test_legal_with_file_reference_selects_document_evidence_template(self):
        prompt = (
            "Review the acquisition structure options and the compliance schedule in "
            "https://example.com/deal.pdf before advising on liability protection."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))

        result = template_selector(state)

        assert state["task_profile"] == "legal_transactional"
        assert result["execution_template"]["template_id"] == "legal_with_document_evidence"

    def test_ambiguous_legal_finance_prompt_chooses_safe_reasoning_template(self):
        prompt = (
            "We need acquisition structure advice and also a quick valuation ratio calculation. "
            "Keep it practical and do not browse."
        )
        state = make_state(prompt)
        state.update(intake(state))
        state.update(task_profiler(state))

        result = template_selector(state)

        assert "legal_finance_overlap" in state["ambiguity_flags"]
        assert result["execution_template"]["template_id"] == "legal_reasoning_only"
