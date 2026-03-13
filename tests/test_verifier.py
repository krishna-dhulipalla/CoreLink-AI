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
from agent.budget import BudgetTracker
from agent.nodes.verifier import verifier, verify_routing, BACKTRACK_WARNING
from agent.memory.store import MemoryStore
from agent.prompts import VERIFIER_JSON_FALLBACK_PROMPT, VerdictDecision


@pytest.fixture(autouse=True)
def _force_native_structured_output(monkeypatch):
    monkeypatch.setenv("STRUCTURED_OUTPUT_MODE", "native")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("COORDINATOR_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("VERIFIER_OPENAI_BASE_URL", raising=False)


class TestVerifierNode:

    def test_verdict_schema_defaults_to_revise_on_junk_payload(self):
        """Schema drift like {'answer': ...} should not silently become PASS."""
        verdict = VerdictDecision.model_validate({"answer": "looks fine"})
        assert verdict.verdict == "REVISE"

    def test_json_fallback_prompt_bans_answer_key(self):
        assert "Do not output keys like answer" in VERIFIER_JSON_FALLBACK_PROMPT

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_fallback_keeps_final_answer_strictness(self, mock_chat_openai, monkeypatch):
        monkeypatch.setenv("STRUCTURED_OUTPUT_MODE", "local_json")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content='{"verdict":"PASS","reasoning":"Looks complete"}')
        mock_chat_openai.return_value = mock_llm

        state = {
            "selected_layers": ["verifier_check"],
            "messages": [HumanMessage(content="Query"), AIMessage(content="Final answer")],
            "checkpoint_stack": [],
            "task_type": "options",
        }

        verifier(state)

        invocation_messages = mock_llm.invoke.call_args.args[0]
        assert "FINAL ANSWER MODE" in invocation_messages[0].content
        assert "options tasks" in invocation_messages[0].content

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
    def test_verifier_legal_revise_injects_constrained_template(self, mock_chat_openai):
        """Legal tasks should switch into constrained final-answer mode after REVISE."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(verdict="REVISE", reasoning="Too verbose")
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="Advise on acquisition structure with EU/US compliance risk."),
                AIMessage(content="Long wandering legal draft"),
            ],
            "checkpoint_stack": [],
            "task_type": "legal",
            "budget_tracker": BudgetTracker(),
        }

        result = verifier(state)

        assert "messages" in result
        warning = result["messages"][0]
        assert isinstance(warning, SystemMessage)
        assert "CONSTRAINED ANSWER MODE" in warning.content
        assert "1. STRUCTURE OPTIONS" in warning.content

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_nonlegal_revise_does_not_inject_constrained_template(self, mock_chat_openai):
        """The constrained legal template should not bleed into other task families."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(verdict="REVISE", reasoning="Needs fuller risk analysis")
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="IV is elevated. Recommend an options strategy."),
                AIMessage(content="Short strangle is appropriate."),
            ],
            "checkpoint_stack": [],
            "task_type": "options",
            "budget_tracker": BudgetTracker(),
        }

        result = verifier(state)

        warning = result["messages"][0]
        assert "CONSTRAINED ANSWER MODE" not in warning.content

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_backtracks_on_textual_disallowed_tool_attempt(self, mock_chat_openai):
        """A blocked tool envelope rendered as plain text should not pass as a final answer."""
        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="Advise on acquisition structure with compliance liabilities."),
                AIMessage(
                    content=(
                        "<think>Need external search.</think>\n"
                        '{"name": "internet_search", "arguments": {"query": "M&A structure"}}'
                    )
                ),
            ],
            "checkpoint_stack": [],
            "task_type": "legal",
        }

        result = verifier(state)

        mock_chat_openai.assert_not_called()
        assert isinstance(result["messages"], ReplaceMessages)
        assert "internet_search" in result["messages"][-1].content
        assert "not allowed" in result["messages"][-1].content

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_revises_on_textual_truncated_tool_attempt(self, mock_chat_openai):
        """A truncated textual tool attempt should be revised before it is mistaken for a final answer."""
        budget = BudgetTracker()
        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="IV is high. Recommend an options strategy."),
                AIMessage(
                    content=(
                        "<think>Need tool data.</think>\n"
                        '{"name": "analyze_strategy", "arguments": {"legs": [{"option_type": "call"'
                    )
                ),
            ],
            "checkpoint_stack": [],
            "task_type": "options",
            "budget_tracker": budget,
        }

        result = verifier(state)

        mock_chat_openai.assert_not_called()
        warning = result["messages"][0]
        assert isinstance(warning, SystemMessage)
        assert "raw tool JSON" in warning.content
        assert "CONSTRAINED OPTIONS MODE" in warning.content

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_uses_latest_user_turn_for_repair_lookup(self, mock_chat_openai):
        """Repair lookup should use the current turn, not the oldest human message."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(
            verdict="REVISE",
            reasoning="The tool call repeats a failed action and should change strategy.",
        )
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        memory_store = MagicMock()
        memory_store.retrieve_verifier_hints.return_value = []

        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="old task"),
                AIMessage(content="old answer"),
                HumanMessage(content="current task"),
                AIMessage(content="current draft"),
            ],
            "checkpoint_stack": [],
            "memory_store": memory_store,
        }

        verifier(state)

        memory_store.retrieve_verifier_hints.assert_called_once_with(
            "current task",
            failure_family="repetition",
        )

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_skips_generic_repair_hints_for_content_incompleteness(self, mock_chat_openai):
        """Generic completeness REVISE signals should not inject stale tool-repair memory."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(
            verdict="REVISE",
            reasoning="The answer is directionally correct but critically incomplete.",
        )
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        memory_store = MagicMock()
        memory_store.retrieve_verifier_hints.return_value = ["irrelevant hint"]

        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="legal task"),
                AIMessage(content="first incomplete answer"),
            ],
            "checkpoint_stack": [],
            "memory_store": memory_store,
        }

        verifier(state)

        memory_store.retrieve_verifier_hints.assert_not_called()

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

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_pass_stores_executor_memory_for_tool_step(self, mock_chat_openai, tmp_path):
        """A passing tool step should populate live executor memory."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(verdict="PASS", reasoning="Tool step valid")
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        store = MemoryStore(db_path=str(tmp_path / "agent_memory.db"))
        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="older task"),
                HumanMessage(content="current task"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "black_scholes_price",
                            "args": {"spot": 175, "strike": 180},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="call_price: 3.22", tool_call_id="call_1", name="black_scholes_price"),
            ],
            "checkpoint_stack": [],
            "memory_store": store,
            "pending_verifier_feedback": None,
        }

        result = verifier(state)

        assert result["pending_verifier_feedback"] is None
        assert store.stats()["executor_memory"] == 1
        hints = store.retrieve_executor_hints("current task")
        assert len(hints) == 1
        assert "black_scholes_price" in hints[0]

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_pass_stores_successful_repair_memory(self, mock_chat_openai, tmp_path):
        """A later PASS should convert pending verifier feedback into verifier memory."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(verdict="PASS", reasoning="Recovered")
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        store = MemoryStore(db_path=str(tmp_path / "agent_memory.db"))
        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="current task"),
                SystemMessage(
                    content="VERIFIER REVISION REQUIRED:\nMissing field",
                    additional_kwargs={"is_warning": True},
                ),
                AIMessage(content="Revised answer with the required field"),
            ],
            "checkpoint_stack": [],
            "memory_store": store,
            "pending_verifier_feedback": {
                "verdict": "REVISE",
                "reasoning": "Missing required field",
            },
        }

        result = verifier(state)

        assert result["pending_verifier_feedback"] is None
        assert store.stats()["verifier_memory"] == 1
        hints = store.retrieve_verifier_hints("current task")
        assert len(hints) == 1
        assert "Missing required field" in hints[0]

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_verifier_escalates_stagnant_revise_loop_to_backtrack(self, mock_chat_openai):
        """Repeated near-identical revise attempts should stop looping and backtrack."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(
            verdict="REVISE",
            reasoning="The answer remains incomplete.",
        )
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        budget = BudgetTracker()
        budget.revise_cycles = 2

        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="legal task"),
                AIMessage(content="Draft structure answer"),
                SystemMessage(content="VERIFIER REVISION REQUIRED", additional_kwargs={"is_warning": True}),
                AIMessage(content="Draft structure answer"),
            ],
            "checkpoint_stack": [],
            "budget_tracker": budget,
        }

        result = verifier(state)

        assert isinstance(result["messages"], ReplaceMessages)
        assert "Repeated revise cycles" in result["messages"][-1].content

    @patch("agent.nodes.verifier.ChatOpenAI")
    def test_revise_budget_exit_does_not_checkpoint_poisoned_state(self, mock_chat_openai):
        """Budget exits should not save the current failed attempt as a verified checkpoint."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = VerdictDecision(
            verdict="REVISE",
            reasoning="The executor repeated a malformed tool path.",
        )
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm

        budget = BudgetTracker()
        budget.revise_cycles = 7

        original_stack = [{"messages": [{"type": "human", "data": {"content": "Base Query"}}]}]
        state = {
            "selected_layers": ["verifier_check"],
            "messages": [
                HumanMessage(content="Base Query"),
                AIMessage(content="", tool_calls=[{"name": "create_portfolio", "args": {"name": "internet_search"}, "id": "call_1", "type": "tool_call"}]),
                ToolMessage(content="Error: malformed payload", tool_call_id="call_1", name="create_portfolio"),
            ],
            "checkpoint_stack": list(original_stack),
            "budget_tracker": budget,
        }

        result = verifier(state)

        assert len(result["checkpoint_stack"]) == len(original_stack)
        assert result["checkpoint_stack"] == original_stack

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
