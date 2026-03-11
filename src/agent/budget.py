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

# ---------------------------------------------------------------------------
# Configurable caps (via environment)
# ---------------------------------------------------------------------------

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "30"))
MAX_REVISE_CYCLES = int(os.getenv("MAX_REVISE_CYCLES", "8"))
MAX_BACKTRACK_CYCLES = int(os.getenv("MAX_BACKTRACK_CYCLES", "5"))
MAX_HINT_TOKENS = int(os.getenv("MAX_HINT_TOKENS", "800"))


class BudgetTracker:
    """Tracks per-run resource consumption and enforces hard caps."""

    def __init__(self):
        self.tool_calls: int = 0
        self.revise_cycles: int = 0
        self.backtrack_cycles: int = 0
        self.hint_tokens_used: int = 0
        self.budget_exits: list[dict] = []

    # ---- recording ----

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_revise(self) -> None:
        self.revise_cycles += 1

    def record_backtrack(self) -> None:
        self.backtrack_cycles += 1

    def record_hint_tokens(self, n: int) -> None:
        self.hint_tokens_used += n

    # ---- cap checks ----

    def tool_calls_exhausted(self) -> bool:
        return self.tool_calls >= MAX_TOOL_CALLS

    def revise_exhausted(self) -> bool:
        return self.revise_cycles >= MAX_REVISE_CYCLES

    def backtrack_exhausted(self) -> bool:
        return self.backtrack_cycles >= MAX_BACKTRACK_CYCLES

    def hint_tokens_remaining(self) -> int:
        return max(0, MAX_HINT_TOKENS - self.hint_tokens_used)

    # ---- budget-exit logging ----

    def log_budget_exit(self, category: str, reason: str) -> None:
        """Record a budget-exit event."""
        event = {"category": category, "reason": reason}
        self.budget_exits.append(event)
        logger.warning(f"[Budget] {category}: {reason}")

    # ---- summary ----

    def summary(self) -> dict:
        return {
            "tool_calls": self.tool_calls,
            "tool_calls_cap": MAX_TOOL_CALLS,
            "revise_cycles": self.revise_cycles,
            "revise_cap": MAX_REVISE_CYCLES,
            "backtrack_cycles": self.backtrack_cycles,
            "backtrack_cap": MAX_BACKTRACK_CYCLES,
            "hint_tokens_used": self.hint_tokens_used,
            "hint_tokens_cap": MAX_HINT_TOKENS,
            "budget_exits": self.budget_exits,
        }
