"""
Test End-to-End Verifier Loop
=============================
Exercises the full graph with mocked LLM nodes and a real built-in tool call.
The sequence is:
1. Executor emits a valid tool call.
2. Verifier PASS saves a checkpoint.
3. Executor emits a hallucinated final answer.
4. Verifier BACKTRACK restores the checkpoint and injects a warning.
5. Executor emits a corrected final answer.
6. Verifier PASS allows exit to format_normalizer.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.graph import build_agent_graph
from agent.prompts import VerdictDecision


@pytest.fixture
def anyio_backend():
    return "asyncio"


def mock_llm_reasoner(messages):
    warnings = [
        msg for msg in messages if getattr(msg, "additional_kwargs", {}).get("is_warning", False)
    ]
    tool_results = [msg for msg in messages if isinstance(msg, ToolMessage)]

    if not warnings and not tool_results:
        return AIMessage(
            content="",
            tool_calls=[{"name": "get_current_time", "args": {}, "id": "tc1", "type": "tool_call"}],
        )
    if not warnings and tool_results:
        return AIMessage(content="Final hallucinated answer", tool_calls=[])
    return AIMessage(content="Actually, here is the verified correct answer.", tool_calls=[])


def mock_llm_verifier(messages):
    last_msg = messages[-1]

    if isinstance(last_msg, ToolMessage) and last_msg.name == "get_current_time":
        return VerdictDecision(verdict="PASS", reasoning="Valid tool result.")

    if isinstance(last_msg, AIMessage) and "hallucinated" in last_msg.content:
        return VerdictDecision(
            verdict="BACKTRACK",
            reasoning="The final answer is hallucinated. Revert to the last verified checkpoint.",
        )

    return VerdictDecision(verdict="PASS", reasoning="Looks correct.")


class TestEndToEndVerifier:
    @pytest.mark.anyio
    @patch("agent.nodes.reasoner.ChatOpenAI")
    @patch("agent.nodes.verifier.ChatOpenAI")
    @patch("agent.nodes.coordinator.ChatOpenAI")
    async def test_algorithmic_failure_backtrack(self, mock_coord, mock_verif, mock_reas):
        mock_coord.return_value.with_structured_output.return_value.invoke.return_value = MagicMock(
            layers=["react_reason", "verifier_check"],
            needs_formatting=False,
            confidence=0.9,
            estimated_steps=4,
            early_exit_allowed=True,
        )

        mock_reas.return_value.bind_tools.return_value.invoke.side_effect = mock_llm_reasoner
        mock_verif.return_value.with_structured_output.return_value.invoke.side_effect = mock_llm_verifier

        graph = build_agent_graph([])
        initial_state = {
            "messages": [HumanMessage(content="Solve this problem")],
            "reflection_count": 0,
            "tool_fail_count": 0,
            "last_tool_signature": "",
            "selected_layers": [],
            "format_required": False,
            "policy_confidence": 0.0,
            "estimated_steps": 0,
            "early_exit_allowed": False,
            "architecture_trace": [],
            "checkpoint_stack": [],
            "cost_tracker": None,
        }

        final_state = await graph.ainvoke(initial_state)
        msgs = final_state["messages"]

        assert any(
            isinstance(msg, SystemMessage) and "BACKTRACK WARNING" in msg.content
            for msg in msgs
        )
        assert any(
            isinstance(msg, ToolMessage) and msg.name == "get_current_time"
            for msg in msgs
        )
        assert isinstance(msgs[-1], AIMessage)
        assert "Actually, here is the verified correct answer." in msgs[-1].content
        assert final_state["checkpoint_stack"], "Verifier should have saved at least one checkpoint."
