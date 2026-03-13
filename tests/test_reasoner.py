import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.nodes.reasoner import _executor_max_tokens
from agent.nodes.reasoner import patch_oss_tool_calls
from langchain_core.messages import AIMessage


def test_executor_max_tokens_is_raised_only_for_legal_tasks():
    assert _executor_max_tokens("legal") == 1500
    assert _executor_max_tokens("options") == 1300
    assert _executor_max_tokens("quantitative") == 1000
    assert _executor_max_tokens("general") == 1000


def test_patch_oss_tool_calls_extracts_explicit_tool_json_with_trailing_text():
    class _Tool:
        def __init__(self, name):
            self.name = name

    response = AIMessage(
        content=(
            '<think>use tool</think>\n'
            '{"name":"calculator","arguments":{"expression":"1+1"}}\n\n'
            "2"
        )
    )
    patched = patch_oss_tool_calls(response, [_Tool("calculator")])
    assert patched.tool_calls
    assert patched.tool_calls[0]["name"] == "calculator"
    assert patched.content == ""
