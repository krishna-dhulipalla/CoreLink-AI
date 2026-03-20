"""
Budget Tracker (Sprint 4C)
===========================
Enforces per-run hard caps on tool calls, verifier cycles, and hint tokens.

Inspired by MaAS cost-constrained early exits — prevents runaway loops
without relying solely on LangGraph's recursive depth limit.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


_BASE_TOOL_CALLS = _env_int("MAX_TOOL_CALLS", 30)
_BASE_REVISE_CYCLES = _env_int("MAX_REVISE_CYCLES", 8)
_BASE_BACKTRACK_CYCLES = _env_int("MAX_BACKTRACK_CYCLES", 5)
_BASE_CONTEXT_TOKENS = _env_int("MAX_CONTEXT_TOKENS", _env_int("MAX_HINT_TOKENS", 800))

_TIER_CAPS: dict[str, dict[str, int]] = {
    "simple_exact": {
        "tool_calls": _env_int("SIMPLE_EXACT_MAX_TOOL_CALLS", 1),
        "revise_cycles": _env_int("SIMPLE_EXACT_MAX_REVISE_CYCLES", 2),
        "backtrack_cycles": _env_int("SIMPLE_EXACT_MAX_BACKTRACK_CYCLES", 0),
        "context_tokens": _env_int("SIMPLE_EXACT_MAX_CONTEXT_TOKENS", 400),
    },
    "structured_analysis": {
        "tool_calls": _env_int("STRUCTURED_ANALYSIS_MAX_TOOL_CALLS", 12),
        "revise_cycles": _env_int("STRUCTURED_ANALYSIS_MAX_REVISE_CYCLES", 5),
        "backtrack_cycles": _env_int("STRUCTURED_ANALYSIS_MAX_BACKTRACK_CYCLES", 2),
        "context_tokens": _env_int("STRUCTURED_ANALYSIS_MAX_CONTEXT_TOKENS", 900),
    },
    "complex_qualitative": {
        "tool_calls": _env_int("COMPLEX_QUALITATIVE_MAX_TOOL_CALLS", 14),
        "revise_cycles": _env_int("COMPLEX_QUALITATIVE_MAX_REVISE_CYCLES", 6),
        "backtrack_cycles": _env_int("COMPLEX_QUALITATIVE_MAX_BACKTRACK_CYCLES", 3),
        "context_tokens": _env_int("COMPLEX_QUALITATIVE_MAX_CONTEXT_TOKENS", 1400),
    },
}


class BudgetTracker:
    """Tracks per-run resource consumption and enforces hard caps."""

    def __init__(self):
        self.tool_calls: int = 0
        self.revise_cycles: int = 0
        self.backtrack_cycles: int = 0
        self.context_tokens_used: int = 0
        self.budget_exits: list[dict] = []
        self.complexity_tier: str = "structured_analysis"
        self.template_id: str = ""
        self.tool_calls_cap: int = _BASE_TOOL_CALLS
        self.revise_cap: int = _BASE_REVISE_CYCLES
        self.backtrack_cap: int = _BASE_BACKTRACK_CYCLES
        self.context_tokens_cap: int = _BASE_CONTEXT_TOKENS

    def configure(self, *, complexity_tier: str = "structured_analysis", template_id: str = "") -> None:
        caps = _TIER_CAPS.get(complexity_tier, _TIER_CAPS["structured_analysis"])
        self.complexity_tier = complexity_tier
        self.template_id = template_id
        self.tool_calls_cap = caps["tool_calls"]
        self.revise_cap = caps["revise_cycles"]
        self.backtrack_cap = caps["backtrack_cycles"]
        self.context_tokens_cap = caps["context_tokens"]

    # ---- recording ----

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_revise(self) -> None:
        self.revise_cycles += 1

    def record_backtrack(self) -> None:
        self.backtrack_cycles += 1

    def record_context_tokens(self, n: int) -> None:
        self.context_tokens_used += n

    def record_hint_tokens(self, n: int) -> None:
        self.record_context_tokens(n)

    # ---- cap checks ----

    def tool_calls_exhausted(self) -> bool:
        return self.tool_calls >= self.tool_calls_cap

    def revise_exhausted(self) -> bool:
        return self.revise_cycles >= self.revise_cap

    def backtrack_exhausted(self) -> bool:
        return self.backtrack_cycles >= self.backtrack_cap

    def context_tokens_remaining(self) -> int:
        return max(0, self.context_tokens_cap - self.context_tokens_used)

    def hint_tokens_remaining(self) -> int:
        return self.context_tokens_remaining()

    # ---- budget-exit logging ----

    def log_budget_exit(self, category: str, reason: str) -> None:
        """Record a budget-exit event."""
        event = {"category": category, "reason": reason}
        self.budget_exits.append(event)
        logger.warning(f"[Budget] {category}: {reason}")

    # ---- summary ----

    def summary(self) -> dict:
        return {
            "complexity_tier": self.complexity_tier,
            "template_id": self.template_id,
            "tool_calls": self.tool_calls,
            "tool_calls_cap": self.tool_calls_cap,
            "revise_cycles": self.revise_cycles,
            "revise_cap": self.revise_cap,
            "backtrack_cycles": self.backtrack_cycles,
            "backtrack_cap": self.backtrack_cap,
            "context_tokens_used": self.context_tokens_used,
            "context_tokens_cap": self.context_tokens_cap,
            "budget_exits": self.budget_exits,
        }

    def __repr__(self) -> str:
        summary = self.summary()
        return (
            "BudgetTracker("
            f"tier={summary['complexity_tier']}, "
            f"tools={summary['tool_calls']}/{summary['tool_calls_cap']}, "
            f"revise={summary['revise_cycles']}/{summary['revise_cap']}, "
            f"backtrack={summary['backtrack_cycles']}/{summary['backtrack_cap']})"
        )

    __str__ = __repr__
