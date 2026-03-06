"""
Sprint 1.5: Coordinator Policy & Cost Tracker Tests
=====================================================
Tests routing policy decisions, operator validation,
cost tracking, and conditional format normalization.
All tests mock the LLM — no live API calls required.
"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.operators import Operator, OPERATOR_REGISTRY, validate_layers, DEFAULT_PLANS
from agent.cost import CostTracker, OperatorTrace
from agent.prompts import RouteDecision, DIRECT_RESPONDER_PROMPT, SYSTEM_PROMPT
from agent.nodes.coordinator import coordinator, route_task, direct_responder, format_normalizer


# ── Operator Registry Tests ──────────────────────────────────────────────────


class TestOperatorRegistry:

    def test_all_operators_have_required_fields(self):
        """Every operator must have name, description, cost_class, node_name."""
        for name, op in OPERATOR_REGISTRY.items():
            assert op.name == name
            assert op.description
            assert op.cost_class in ("cheap", "moderate", "expensive")
            assert op.node_name
            assert op.failure_policy in ("retry_once", "fallback_to_reflector", "abort")

    def test_validate_layers_valid(self):
        """Valid operator names pass through."""
        result = validate_layers(["direct_answer", "reflection_review"])
        assert result == ["direct_answer", "reflection_review"]

    def test_validate_layers_invalid_falls_back(self):
        """Invalid operator names are removed; empty → default plan."""
        result = validate_layers(["nonexistent_op", "made_up"])
        assert result == ["react_reason", "reflection_review"]

    def test_validate_layers_mixed(self):
        """Mix of valid and invalid keeps only valid ones."""
        result = validate_layers(["direct_answer", "nonexistent_op"])
        assert result == ["direct_answer"]

    def test_default_plans_exist(self):
        """Default plans cover direct and heavy_research."""
        assert "direct" in DEFAULT_PLANS
        assert "heavy_research" in DEFAULT_PLANS
        assert DEFAULT_PLANS["direct"] == ["direct_answer"]

    def test_operator_is_frozen(self):
        """Operators should be immutable."""
        op = OPERATOR_REGISTRY["direct_answer"]
        with pytest.raises(AttributeError):
            op.name = "hacked"


# ── Cost Tracker Tests ───────────────────────────────────────────────────────


class TestCostTracker:

    def test_empty_tracker(self):
        """Fresh tracker starts at zero."""
        tracker = CostTracker(model_name="gpt-4o-mini")
        assert tracker.total_tokens == 0
        assert tracker.total_cost() == 0.0
        assert tracker.llm_calls == 0
        assert tracker.mcp_calls == 0

    def test_record_accumulates(self):
        """Multiple records accumulate tokens and cost."""
        tracker = CostTracker(model_name="gpt-4o-mini")
        tracker.record(operator="coordinator", tokens_in=100, tokens_out=50, latency_ms=150)
        tracker.record(operator="react_reason", tokens_in=500, tokens_out=200, latency_ms=1200)
        assert tracker.llm_calls == 2
        assert tracker.total_tokens == 850  # 100+50+500+200

    def test_cost_calculation(self):
        """Cost = (tokens_in/1K)*input_rate + (tokens_out/1K)*output_rate."""
        tracker = CostTracker(model_name="gpt-4o-mini")
        # gpt-4o-mini: input=0.00015, output=0.0006
        trace = tracker.record(operator="test", tokens_in=1000, tokens_out=1000)
        expected = (1000/1000)*0.00015 + (1000/1000)*0.0006  # 0.00075
        assert abs(trace.cost_usd - expected) < 1e-8

    def test_free_model_zero_cost(self):
        """Competition endpoint (gpt-oss-20b) should have zero cost."""
        tracker = CostTracker(model_name="openai/gpt-oss-20b")
        tracker.record(operator="test", tokens_in=5000, tokens_out=2000)
        assert tracker.total_cost() == 0.0

    def test_mcp_call_counter(self):
        """MCP calls are tracked separately from LLM calls."""
        tracker = CostTracker()
        tracker.record_mcp_call()
        tracker.record_mcp_call()
        tracker.record_mcp_call()
        assert tracker.mcp_calls == 3
        assert tracker.llm_calls == 0  # MCP calls don't count as LLM calls

    def test_summary_structure(self):
        """Summary dict has expected keys."""
        tracker = CostTracker()
        tracker.record(operator="coordinator", tokens_in=100, tokens_out=50, latency_ms=150)
        s = tracker.summary()
        assert "llm_calls" in s
        assert "mcp_calls" in s
        assert "total_tokens" in s
        assert "total_cost_usd" in s
        assert "wall_clock_ms" in s
        assert "operators_used" in s
        assert "any_failure" in s

    def test_architecture_trace_serialization(self):
        """Architecture trace produces list of dicts."""
        tracker = CostTracker()
        tracker.record(operator="coordinator", tokens_in=100, tokens_out=50, latency_ms=150)
        trace = tracker.architecture_trace()
        assert len(trace) == 1
        assert trace[0]["operator"] == "coordinator"
        assert isinstance(trace[0]["tokens_in"], int)

    def test_failure_tracking(self):
        """Failures are counted correctly."""
        tracker = CostTracker()
        tracker.record(operator="react_reason", success=True)
        tracker.record(operator="react_reason", success=False)
        assert tracker.summary()["any_failure"] is True


# ── RouteDecision Schema Tests ───────────────────────────────────────────────


class TestRouteDecision:

    def test_valid_decision(self):
        """A well-formed RouteDecision parses correctly."""
        d = RouteDecision(
            layers=["react_reason", "reflection_review"],
            confidence=0.85,
            needs_formatting=True,
            estimated_steps=5,
            early_exit_allowed=False,
        )
        assert d.layers == ["react_reason", "reflection_review"]
        assert d.confidence == 0.85
        assert d.needs_formatting is True

    def test_defaults(self):
        """RouteDecision has sensible defaults."""
        d = RouteDecision(layers=["direct_answer"])
        assert d.confidence == 0.5
        assert d.needs_formatting is False
        assert d.early_exit_allowed is True


# ── Routing Policy Tests ─────────────────────────────────────────────────────


class TestRoutingPolicy:

    def test_direct_route_goes_to_direct_responder(self):
        """When layers = ['direct_answer'], route_task returns 'direct_responder'."""
        state = {"selected_layers": ["direct_answer"]}
        assert route_task(state) == "direct_responder"

    def test_heavy_route_goes_to_reasoner(self):
        """When layers start with 'react_reason', route_task returns 'reasoner'."""
        state = {"selected_layers": ["react_reason", "reflection_review"]}
        assert route_task(state) == "reasoner"

    def test_empty_layers_defaults_to_reasoner(self):
        """Empty layers = safe default → reasoner."""
        state = {"selected_layers": []}
        assert route_task(state) == "reasoner"

    def test_missing_layers_defaults_to_reasoner(self):
        """Missing key = safe default → reasoner."""
        state = {}
        assert route_task(state) == "reasoner"


# ── Format Normalizer Conditional Tests ──────────────────────────────────────


class TestFormatNormalizerConditional:

    def test_skips_when_not_required(self):
        """format_normalizer returns empty messages when format_required=False."""
        state = {
            "messages": [
                HumanMessage(content="hello"),
                AIMessage(content="world"),
            ],
            "format_required": False,
            "cost_tracker": CostTracker(),
        }
        result = format_normalizer(state)
        assert result["messages"] == []

    def test_skips_when_format_required_missing(self):
        """Default = skip formatting."""
        state = {
            "messages": [AIMessage(content="test")],
            "cost_tracker": CostTracker(),
        }
        result = format_normalizer(state)
        assert result["messages"] == []


# ── Direct Responder Prompt Tests ────────────────────────────────────────────


class TestDirectResponderPrompt:

    def test_direct_prompt_has_no_tool_claims(self):
        """DIRECT_RESPONDER_PROMPT must NOT claim tools are available."""
        # The prompt may mention tools to disclaim them ("you do NOT have tools").
        # It must NOT contain phrases that claim tool access exists.
        positive_claims = [
            "you have access to",
            "use the calculator",
            "use internet_search",
            "call list_reference_files",
            "call fetch_reference_file",
            "tools available to you",
        ]
        prompt_lower = DIRECT_RESPONDER_PROMPT.lower()
        for phrase in positive_claims:
            assert phrase not in prompt_lower, \
                f"DIRECT_RESPONDER_PROMPT claims tool access: '{phrase}'"

    def test_system_prompt_has_tool_mentions(self):
        """SYSTEM_PROMPT (for reasoner) should mention tools — sanity check."""
        assert "tool" in SYSTEM_PROMPT.lower()
