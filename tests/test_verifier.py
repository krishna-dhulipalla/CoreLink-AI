"""
Sprint 2: Verifier Node Tests
=============================
Tests the step-level verification loop, PASS/REVISE/BACKTRACK logic, and
checkpoint stack behavior.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.state import AgentState, ReplaceMessages
from agent.nodes.verifier import verifier, verify_routing, BACKTRACK_WARNING
from agent.prompts import VerdictDecision

class TestVerifierNode:

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_pass(self, mock_chat_openai):
        """A PASS verdict adds the current state to the checkpoint stack."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(verdict="PASS", reasoning="Looks good")
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        state = {
            "selected_layers": ["verifier_check"],
            "messages": [HumanMessage(content="Query"), AIMessage(content="Answer")],
            "checkpoint_stack": [],
        }

        result = verifier(state)
        
        # We expect a new checkpoint stack to be returned
        assert "checkpoint_stack" in result
        stack = result["checkpoint_stack"]
        assert len(stack) == 1
        assert len(stack[0]["messages"]) == 2

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_revise(self, mock_chat_openai):
        """A REVISE verdict injects a warning SystemMessage but doesn't change the stack."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(verdict="REVISE", reasoning="Fix the syntax")
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        state = {
            "selected_layers": ["verifier_check"],
            "messages": [AIMessage(content="Answer with bad syntax")],
            "checkpoint_stack": [],
        }

        result = verifier(state)
        
        assert "messages" in result
        assert isinstance(result["messages"][0], SystemMessage)
        assert "VERIFIER REVISION REQUIRED" in result["messages"][0].content
        assert result["messages"][0].additional_kwargs.get("is_warning") is True
        assert "checkpoint_stack" not in result

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_backtrack_with_stack(self, mock_chat_openai):
        """BACKTRACK reverts to the top of the checkpoint stack and appends a warning."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(verdict="BACKTRACK", reasoning="Bad path")
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        # Top of stack has one message
        checkpoint = {"messages": [{"type": "human", "data": {"content": "Base Query"}}]}
        
        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="Base Query"), 
                AIMessage(content="Hallucination")
            ],
            "checkpoint_stack": [checkpoint],
        }

        result = verifier(state)
        
        assert "messages" in result
        msgs = result["messages"]
        assert isinstance(msgs, ReplaceMessages)
        # Should be Base Query + System Warning
        assert len(msgs) == 2
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], SystemMessage)
        assert getattr(msgs[1], "additional_kwargs", {}).get("is_warning") is True
        assert "BACKTRACK WARNING" in msgs[1].content or "BACKTRACK" in msgs[1].content

    def test_verify_routing_warning(self):
        """If the last message is a warning, route back to reasoner."""
        state = {
            "messages": [SystemMessage(content="Warning", additional_kwargs={"is_warning": True})]
        }
        assert verify_routing(state) == "reasoner"

    def test_verify_routing_final_answer(self):
        """If the last message is a plain AIMessage (final answer), route to format_normalizer."""
        state = {
            "messages": [AIMessage(content="Final Answer", tool_calls=[])]
        }
        assert verify_routing(state) == "format_normalizer"

    def test_verify_routing_tool_call(self):
        """If the last message (before verification) was a tool results, route to reasoner."""
        state = {
            "messages": [ToolMessage(content="Results", tool_call_id="123")]
        }
        assert verify_routing(state) == "reasoner"
