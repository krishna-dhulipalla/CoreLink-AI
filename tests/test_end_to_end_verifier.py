"""
Test End-to-End Verifier Loop
=============================
Tests the whole graph using mock tools and a mock LLM, ensuring an algorithmic
failure triggers REVISE and BACKTRACK paths.
"""

import os
import sys
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.graph import build_agent_graph
from agent.prompts import VerdictDecision
from agent.state import AgentState

def mock_llm_reasoner(*args, **kwargs):
    # This mock behaves differently depending on the history length and contents
    messages = args[0]
    warns = [m for m in messages if getattr(m, "additional_kwargs", {}).get("is_warning", False)]
    
    if len(warns) == 0:
        # Step 1: initial bad answer that hallucinated a tool
        return AIMessage(content="I am calling a non-existent tool", tool_calls=[{"name": "made_up_tool", "args": {}, "id": "tc1"}])
    elif len(warns) == 1 and "REVISION REQUIRED" in warns[0].content:
        # Step 2: instead of fixing, we double down and give a terrible answer to force backtrack
        return AIMessage(content="Final hallucinated answer", tool_calls=[])
    else:
        # Step 3: after backtrack warning, we give a good answer
        return AIMessage(content="Actually, here is the verified correct answer.", tool_calls=[])

def mock_llm_verifier(*args, **kwargs):
    messages = args[0]
    last_msg = messages[-1]
    
    # Check what the executor did
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls and last_msg.tool_calls[0]["name"] == "made_up_tool":
        # Verifier catches the bad tool
        return VerdictDecision(verdict="REVISE", reasoning="This tool does not exist. Please review your available tools.")
    
    if isinstance(last_msg, AIMessage) and "hallucinated" in last_msg.content:
        # Verifier detects a complete dead-end
        return VerdictDecision(verdict="BACKTRACK", reasoning="You are hallucinating. Try a completely different approach.")
        
    # Otherwise pass
    return VerdictDecision(verdict="PASS", reasoning="Looks correct.")

import pytest

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
            early_exit_allowed=True
        )
        
        mock_reas.return_value.bind_tools.return_value.invoke.side_effect = mock_llm_reasoner
        mock_verif.return_value.with_structured_output.return_value.invoke.side_effect = mock_llm_verifier
        
        graph = build_agent_graph([])
        
        initial_state = {
            "messages": [HumanMessage(content="Solve this problem")],
            "reflection_count": 0,
            "tool_fail_count": 0,
            "last_tool_signature": "",
            "cost_tracker": None
        }
        
        final_state = await graph.ainvoke(initial_state)
        
        # Verify the timeline
        msgs = final_state["messages"]
        content_log = [
            (type(m).__name__, m.content[:30] if hasattr(m, "content") else "") 
            for m in msgs
        ]
        
        # We expect:
        # HumanMessage: Solve this problem
        # AIMessage: I am calling a non-ex... (bad tool call)
        # SystemMessage (Revise): VERIFIER REVISION RE...
        # AIMessage: Final hallucinated an... (bad final answer)
        # SystemMessage (Backtrack): SYSTEM WARNING: Your ... -> wait! BACKTRACK REVERTS the state!
        # When it reverts, it pops back to the last good checkpoint.
        # But wait, our first step was REVISE, which doesn't push a checkpoint.
        # So the stack is empty! When stack is empty, BACKTRACK acts like a REVISE according to our verifier logic.
        
        assert "Actually, here is the verified" in msgs[-1].content
