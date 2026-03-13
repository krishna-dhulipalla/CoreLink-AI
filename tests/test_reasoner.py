import os
import sys
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.nodes.reasoner import _executor_max_tokens
from agent.nodes.reasoner import make_reasoner
from agent.nodes.reasoner import patch_oss_tool_calls
from langchain_core.messages import AIMessage, HumanMessage


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


@patch("agent.nodes.reasoner.ChatOpenAI")
def test_reasoner_injects_legal_revision_guidance(mock_chat_openai):
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Revised legal answer")
    mock_chat_openai.return_value = mock_model

    reasoner = make_reasoner([])
    reasoner(
        {
            "messages": [HumanMessage(content="Advise on acquisition structure with compliance liabilities.")],
            "task_type": "legal",
            "pending_verifier_feedback": {
                "verdict": "REVISE",
                "reasoning": "The legal final answer is incomplete. Add key open questions / assumptions.",
            },
        }
    )

    invocation_messages = mock_model.invoke.call_args.args[0]
    joined = "\n".join(str(m.content) for m in invocation_messages if getattr(m, "content", None))
    assert "VERIFIER FEEDBACK" in joined
    assert "KEY OPEN QUESTIONS / ASSUMPTIONS" in joined


@patch("agent.nodes.reasoner.ChatOpenAI")
def test_reasoner_injects_compact_options_revision_guidance(mock_chat_openai):
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Revised options answer")
    mock_chat_openai.return_value = mock_model

    reasoner = make_reasoner([])
    reasoner(
        {
            "messages": [HumanMessage(content="IV is high. Recommend an options strategy.")],
            "task_type": "options",
            "pending_verifier_feedback": {
                "verdict": "REVISE",
                "reasoning": "The final answer was truncated before completion.",
            },
        }
    )

    invocation_messages = mock_model.invoke.call_args.args[0]
    joined = "\n".join(str(m.content) for m in invocation_messages if getattr(m, "content", None))
    assert "VERIFIER FEEDBACK" in joined
    assert "PRIMARY STRATEGY" in joined
    assert "KEY GREEKS / P&L / BREAKEVENS" in joined


@patch("agent.nodes.reasoner.ChatOpenAI")
def test_reasoner_injects_first_turn_legal_no_tool_policy(mock_chat_openai):
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = AIMessage(content="Structured legal answer")
    mock_chat_openai.return_value = mock_model

    reasoner = make_reasoner([])
    reasoner(
        {
            "messages": [HumanMessage(content="Target company has EU and US compliance gaps and we need an acquisition structure quickly.")],
            "task_type": "legal",
            "pending_verifier_feedback": None,
        }
    )

    invocation_messages = mock_model.invoke.call_args.args[0]
    joined = "\n".join(str(m.content) for m in invocation_messages if getattr(m, "content", None))
    assert "FIRST LEGAL TURN POLICY" in joined
    assert "Do NOT call any tool on this turn" in joined
