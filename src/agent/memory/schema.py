"""
Memory Schemas (Sprint 3)
=========================
Pydantic models for the three role-specific memory units:
  - RouterMemory:   What operator plan worked for a given task signature.
  - ExecutorMemory: Which tool + args pattern produced a quality outcome.
  - VerifierMemory: Which failure pattern was recovered via REVISE or BACKTRACK.

Each schema is compact by design. We store *fragments*, not full trajectories.
"""

from __future__ import annotations

import hashlib
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field


def _task_signature(text: str) -> str:
    """Create a stable, short hash of the task text for dedup / retrieval."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


class RouterMemory(BaseModel):
    """Compact record of a Coordinator routing decision and its outcome."""

    task_signature: str = Field(
        description="SHA-256 prefix of the normalized task text."
    )
    task_summary: str = Field(
        description="One-line summary of the task (max ~120 chars)."
    )
    selected_layers: list[str] = Field(
        description="Operator plan chosen by the Coordinator."
    )
    success: bool = Field(
        description="Whether the run ended with a verified PASS final answer."
    )
    cost_usd: float = Field(default=0.0, description="Total run cost in USD.")
    latency_ms: float = Field(default=0.0, description="Total wall-clock ms.")
    timestamp: float = Field(default_factory=time.time)
    tags: list[str] = Field(default_factory=list)


class ExecutorMemory(BaseModel):
    """Compact record of a single tool-selection fragment."""

    task_signature: str = Field(
        description="SHA-256 prefix of the normalized task text."
    )
    partial_context_summary: str = Field(
        description="One-line summary of what the executor was trying to do."
    )
    tool_used: str = Field(description="Name of the tool selected.")
    arguments_pattern: str = Field(
        description=(
            "Normalized hint about the arguments used, e.g. "
            "'spot=180, strike=175, rate=0.05, vol=0.25, T=0.5'."
        )
    )
    outcome_quality: Literal["good", "acceptable", "poor"] = Field(
        description="Subjective quality based on verifier verdict."
    )
    success: bool = Field(description="True if the tool call ultimately PASSed.")
    timestamp: float = Field(default_factory=time.time)
    tags: list[str] = Field(default_factory=list)


class VerifierMemory(BaseModel):
    """Compact record of a failure-recovery pattern."""

    task_signature: str = Field(
        description="SHA-256 prefix of the normalized task text."
    )
    failure_pattern: str = Field(
        description="One-line description of what went wrong."
    )
    verdict: Literal["REVISE", "BACKTRACK"] = Field(
        description="Which verdict was issued."
    )
    repair_action: str = Field(
        description="What the executor did differently after the verdict."
    )
    repair_worked: bool = Field(
        description="Whether the repair ultimately led to PASS."
    )
    timestamp: float = Field(default_factory=time.time)
    tags: list[str] = Field(default_factory=list)
