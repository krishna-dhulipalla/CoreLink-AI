import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.runner import _extract_final_answer


class TestRunnerAnswerExtraction:
    def test_extract_final_answer_strips_unclosed_think_tag(self):
        text = "<think>\nReasoning about the deal structure.\nUse an asset purchase with carve-outs."
        result = _extract_final_answer(text)
        assert "<think>" not in result
        assert "asset purchase" in result

    def test_extract_final_answer_uses_reasoning_for_orphan_tool_json(self):
        text = (
            "<think>\nNet seller because IV exceeds realized volatility.\n</think>\n"
            '{"name":"analyze_strategy","arguments":{"legs":[]}}'
        )
        result = _extract_final_answer(text)
        assert "Net seller" in result
        assert '"name"' not in result
