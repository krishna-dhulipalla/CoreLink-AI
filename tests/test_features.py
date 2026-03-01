"""
Unit tests for observation masking, reflection hygiene, and conversation store.

These run without a live server — they test the modules directly.
"""

import sys
import os
import time

import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from context_manager import (
    count_tokens,
    truncate_tool_output,
    summarize_and_window,
    _group_messages,
)
from conversation_store import ConversationStore
from agent import (
    SYSTEM_PROMPT,
    _build_reflection_context,
    _with_system_prompt,
)


# ── Truncation Tests ──────────────────────────────────────────────────────


class TestTruncation:
    def test_short_content_unchanged(self):
        """Content under the limit should pass through unchanged."""
        short = "Hello, world!"
        assert truncate_tool_output(short, max_chars=100) == short

    def test_long_content_truncated(self):
        """Content over the limit should be truncated with a marker."""
        long_text = "x" * 500
        result = truncate_tool_output(long_text, max_chars=100)
        assert len(result) < len(long_text)
        assert "[TRUNCATED" in result
        assert "400 characters removed" in result

    def test_exact_limit_unchanged(self):
        """Content exactly at the limit should pass through unchanged."""
        exact = "y" * 100
        assert truncate_tool_output(exact, max_chars=100) == exact


# ── Token Counting Tests ─────────────────────────────────────────────────


class TestTokenCounting:
    def test_empty_messages(self):
        assert count_tokens([]) == 0

    def test_single_message(self):
        tokens = count_tokens([HumanMessage(content="Hello")])
        assert tokens > 0

    def test_more_messages_more_tokens(self):
        one = count_tokens([HumanMessage(content="Hi")])
        two = count_tokens([
            HumanMessage(content="Hi"),
            AIMessage(content="Hello there, how can I help?"),
        ])
        assert two > one


# ── Windowing Tests ──────────────────────────────────────────────────────


class TestWindowing:
    def _make_messages(self, n: int) -> list:
        """Build a conversation with n HumanMessage/AIMessage pairs."""
        msgs = [SystemMessage(content="You are a helpful agent.")]
        for i in range(n):
            msgs.append(HumanMessage(content=f"Question {i}: " + "x" * 200))
            msgs.append(AIMessage(content=f"Answer {i}: " + "y" * 200))
        return msgs

    def test_under_budget_unchanged(self):
        """Messages under the token budget should not be compressed."""
        msgs = self._make_messages(2)
        result = summarize_and_window(msgs, max_tokens=999999)
        assert len(result) == len(msgs)

    def test_over_budget_compressed(self):
        """Messages over the token budget should be compressed."""
        msgs = self._make_messages(20)  # ~20 pairs = lots of tokens
        result = summarize_and_window(msgs, max_tokens=500, keep_recent=4)
        assert len(result) < len(msgs)
        # Should have: SystemMessage + summary SystemMessage + 4 recent
        assert len(result) <= 6

    def test_system_message_preserved(self):
        """The original SystemMessage should always be preserved."""
        msgs = self._make_messages(20)
        result = summarize_and_window(msgs, max_tokens=500, keep_recent=4)
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are a helpful agent."

    def test_summary_marker_present(self):
        """Compressed output should contain the context summary marker."""
        msgs = self._make_messages(20)
        result = summarize_and_window(msgs, max_tokens=500, keep_recent=4)
        summary_msgs = [
            m for m in result
            if isinstance(m, SystemMessage) and "CONTEXT SUMMARY" in m.content
        ]
        assert len(summary_msgs) == 1

    def test_tool_call_bundle_kept_together(self):
        """Compression must not orphan ToolMessages from their tool-call AI message."""
        msgs = [
            SystemMessage(content="You are a helpful agent."),
            HumanMessage(content="Do the tool call."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_magic",
                        "args": {"text": "hello"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="MOCK_MCP_OK::hello",
                tool_call_id="call_1",
                name="echo_magic",
            ),
            AIMessage(content="Done."),
            HumanMessage(content="Next question."),
        ]

        result = summarize_and_window(msgs, max_tokens=1, keep_recent=3)

        tool_idx = next(
            i for i, msg in enumerate(result) if isinstance(msg, ToolMessage)
        )
        assert tool_idx > 0
        assert isinstance(result[tool_idx - 1], AIMessage)
        assert result[tool_idx - 1].tool_calls

    def test_skips_compression_if_bundle_starts_immediately(self):
        """If compression can only split a leading tool bundle, leave history unchanged."""
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_magic",
                        "args": {"text": "hello"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="MOCK_MCP_OK::hello",
                tool_call_id="call_1",
                name="echo_magic",
            ),
            AIMessage(content="Done."),
            HumanMessage(content="Next question."),
        ]

        result = summarize_and_window(msgs, max_tokens=1, keep_recent=3)
        # Should skip compression because not enough groups to split
        assert result == msgs


# ── Tool-Call Adjacency Tests ─────────────────────────────────────────────


class TestToolCallAdjacency:
    """Verify that windowing NEVER separates AIMessage(tool_calls) from ToolMessage."""

    def _tool_pair(self, call_id: str, tool_name: str = "echo_magic"):
        """Create a matched AIMessage(tool_calls) + ToolMessage pair."""
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": {"text": "hello"},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )
        tool = ToolMessage(
            content=f"result for {call_id}",
            tool_call_id=call_id,
            name=tool_name,
        )
        return ai, tool

    def test_group_messages_basic(self):
        """AIMessage(tool_calls) + ToolMessage should form one group."""
        ai, tool = self._tool_pair("c1")
        msgs = [HumanMessage(content="hi"), ai, tool, AIMessage(content="done")]
        groups = _group_messages(msgs)
        assert len(groups) == 3  # [Human], [AI+Tool], [AI]
        assert len(groups[1]) == 2  # the tool pair
        assert groups[1][0] is ai
        assert groups[1][1] is tool

    def test_group_messages_multiple_tools(self):
        """AIMessage with multiple tool_calls groups all following ToolMessages."""
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "a", "args": {}, "id": "c1", "type": "tool_call"},
                {"name": "b", "args": {}, "id": "c2", "type": "tool_call"},
            ],
        )
        t1 = ToolMessage(content="r1", tool_call_id="c1", name="a")
        t2 = ToolMessage(content="r2", tool_call_id="c2", name="b")
        groups = _group_messages([ai, t1, t2])
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_windowing_preserves_adjacency(self):
        """After windowing with tool groups, every ToolMessage must follow its AI."""
        # Build a long conversation with alternating tool-call turns
        msgs = [SystemMessage(content="sys")]
        for i in range(10):
            msgs.append(HumanMessage(content=f"Q{i}: " + "x" * 200))
            ai, tool = self._tool_pair(f"call_{i}")
            msgs.append(ai)
            msgs.append(tool)
            msgs.append(AIMessage(content=f"A{i}: " + "y" * 200))

        result = summarize_and_window(msgs, max_tokens=500, keep_recent=4)
        assert len(result) < len(msgs)

        # Verify the invariant: every ToolMessage must be preceded by its AIMessage
        for idx, msg in enumerate(result):
            if isinstance(msg, ToolMessage):
                prev = result[idx - 1]
                assert isinstance(prev, AIMessage) or isinstance(prev, ToolMessage), (
                    f"ToolMessage at index {idx} is not preceded by AIMessage or ToolMessage"
                )

    def test_no_orphaned_tool_messages_in_recent(self):
        """The recent set must not start with orphaned ToolMessages."""
        msgs = [SystemMessage(content="sys")]
        for i in range(8):
            msgs.append(HumanMessage(content=f"Q{i}: " + "x" * 300))
            ai, tool = self._tool_pair(f"call_{i}")
            msgs.append(ai)
            msgs.append(tool)
            msgs.append(AIMessage(content=f"A{i}"))

        result = summarize_and_window(msgs, max_tokens=200, keep_recent=3)

        # Find where the summary ends and recent starts
        summary_end = 0
        for i, msg in enumerate(result):
            if isinstance(msg, SystemMessage):
                summary_end = i

        # First non-system message should NOT be an orphaned ToolMessage
        if summary_end + 1 < len(result):
            first_after_summary = result[summary_end + 1]
            assert not isinstance(first_after_summary, ToolMessage), (
                "Recent set starts with an orphaned ToolMessage"
            )


# ── ConversationStore Tests ──────────────────────────────────────────────


class TestConversationStore:
    def test_get_empty(self):
        store = ConversationStore()
        assert store.get("nonexistent") == []

    def test_save_and_get(self):
        store = ConversationStore()
        msgs = [HumanMessage(content="Hello"), AIMessage(content="Hi")]
        store.save("ctx1", msgs)
        result = store.get("ctx1")
        assert len(result) == 2
        assert result[0].content == "Hello"

    def test_clear(self):
        store = ConversationStore()
        store.save("ctx1", [HumanMessage(content="Hello")])
        store.clear("ctx1")
        assert store.get("ctx1") == []

    def test_ttl_expiry(self):
        """Entries older than TTL should be purged."""
        store = ConversationStore(ttl=1)  # 1 second TTL
        store.save("ctx1", [HumanMessage(content="Hello")])
        assert len(store.get("ctx1")) == 1

        time.sleep(1.5)  # Wait for expiry
        assert store.get("ctx1") == []

    def test_independent_contexts(self):
        store = ConversationStore()
        store.save("a", [HumanMessage(content="from A")])
        store.save("b", [HumanMessage(content="from B")])
        assert store.get("a")[0].content == "from A"
        assert store.get("b")[0].content == "from B"


# ── Reflection Hygiene Tests ─────────────────────────────────────────────


class TestReflectionHygiene:
    def test_reflection_messages_identified(self):
        """Verify the reflection prefix pattern used for stripping."""
        reflection = AIMessage(content="[Reflection]: PASS: looks good")
        normal = AIMessage(content="The answer is 42.")

        assert reflection.content.startswith("[Reflection]")
        assert not normal.content.startswith("[Reflection]")

    def test_strip_reflections_from_history(self):
        """Simulate the stripping logic from run_agent."""
        messages = [
            HumanMessage(content="What is 2+2?"),
            AIMessage(content="The answer is 4."),
            AIMessage(content="[Reflection]: PASS: correct"),
        ]
        cleaned = [
            msg for msg in messages
            if not (
                isinstance(msg, AIMessage)
                and msg.content
                and msg.content.startswith("[Reflection]")
            )
        ]
        assert len(cleaned) == 2
        assert all(not m.content.startswith("[Reflection]") for m in cleaned)

    def test_reflection_context_excludes_tool_call_messages(self):
        """Reflection input must not include orphaned tool-call assistant messages."""
        messages = [
            HumanMessage(content="Use the tool."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_magic",
                        "args": {"text": "hello"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="MOCK_MCP_OK::hello",
                tool_call_id="call_1",
                name="echo_magic",
            ),
            AIMessage(content="MOCK_MCP_OK::hello"),
            AIMessage(content="[Reflection]: PASS: good"),
        ]

        result = _build_reflection_context(messages)

        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert not result[1].tool_calls
        assert not result[1].content.startswith("[Reflection]")


class TestSystemPromptHandling:
    def test_system_prompt_added_before_summary_message(self):
        """A summary SystemMessage must not suppress the core agent prompt."""
        messages = [SystemMessage(content="[CONTEXT SUMMARY]"), HumanMessage(content="Hi")]
        result = _with_system_prompt(messages)

        assert isinstance(result[0], SystemMessage)
        assert result[0].content == SYSTEM_PROMPT
        assert isinstance(result[1], SystemMessage)
        assert result[1].content == "[CONTEXT SUMMARY]"

    def test_system_prompt_not_duplicated(self):
        result = _with_system_prompt([SystemMessage(content=SYSTEM_PROMPT)])
        assert len(result) == 1
