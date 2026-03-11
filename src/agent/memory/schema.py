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
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field


def _task_signature(text: str) -> str:
    """Create a stable, short hash of the task text for dedup / retrieval."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def _normalize_memory_text(text: str, max_len: int = 400) -> str:
    """Normalize free-text fields so future semantic indexing has clean input."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:max_len]


def _infer_task_family(text: str) -> str:
    """Assign a coarse task family for future filtered retrieval."""
    normalized = _normalize_memory_text(text, max_len=800).lower()
    if not normalized:
        return "general"
    if any(
        token in normalized for token in (
            "option", "greeks", "volatility", "iv", "p&l", "black-scholes",
            "portfolio", "yield", "bond", "stock", "market", "trade", "var",
            "risk", "roe", "roa", "financial", "price", "valuation",
        )
    ):
        return "finance"
    if any(
        token in normalized for token in (
            "acquisition", "merger", "regulatory", "compliance", "liability",
            "indemnification", "contract", "legal", "tax", "board", "eu", "us law",
        )
    ):
        return "legal"
    if any(
        token in normalized for token in (
            "pdf", "document", "reference data", "table", "page", "treasury",
            "bulletin", "extract", "file", "row", "column",
        )
    ):
        return "document"
    if any(
        token in normalized for token in (
            "search", "retrieve", "look up", "reference file", "source", "citation",
        )
    ):
        return "retrieval"
    return "general"


def _infer_tool_family(tool_name: str) -> str:
    """Assign a coarse tool family for later executor-memory filtering."""
    normalized = (tool_name or "").strip().lower()
    if any(token in normalized for token in ("search", "tavily", "internet", "lookup")):
        return "search"
    if any(token in normalized for token in ("file", "pdf", "document", "table", "fetch_reference", "extract")):
        return "document"
    if any(
        token in normalized for token in (
            "black_scholes", "greek", "option", "risk", "portfolio", "yield",
            "volatility", "price", "cagr", "finance", "market_data",
        )
    ):
        return "finance"
    if any(token in normalized for token in ("calculator", "math", "python", "compute")):
        return "calculator"
    return "generic"


def _infer_failure_family(text: str) -> str:
    """Assign a coarse verifier-failure family for later repair filtering."""
    normalized = _normalize_memory_text(text, max_len=800).lower()
    if any(token in normalized for token in ("json", "schema", "field required", "structured output")):
        return "schema"
    if any(token in normalized for token in ("format", "missing field", "invalid format", "output format")):
        return "format"
    if any(token in normalized for token in ("hallucinat", "made up", "unsupported claim", "not grounded")):
        return "hallucination"
    if any(token in normalized for token in ("repeat", "same mistake", "loop", "again")):
        return "repetition"
    if any(token in normalized for token in ("lack of data", "missing data", "unable to find", "not enough data")):
        return "missing_data"
    if any(token in normalized for token in ("tool", "call", "argument", "calculator", "mcp")):
        return "tool_use"
    return "generic"


class RouterMemory(BaseModel):
    """Compact record of a Coordinator routing decision and its outcome."""

    task_signature: str = Field(
        description="SHA-256 prefix of the normalized task text."
    )
    task_summary: str = Field(
        description="One-line summary of the task (max ~120 chars)."
    )
    semantic_text: str = Field(
        default="",
        description="Normalized text chunk intended for future semantic retrieval."
    )
    task_family: str = Field(
        default="generic",
        description="Coarse task bucket such as finance, legal, retrieval, document, or general."
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutorMemory(BaseModel):
    """Compact record of a single tool-selection fragment."""

    task_signature: str = Field(
        description="SHA-256 prefix of the normalized task text."
    )
    partial_context_summary: str = Field(
        description="One-line summary of what the executor was trying to do."
    )
    semantic_text: str = Field(
        default="",
        description="Normalized retrieval text for future semantic memory lookup."
    )
    task_family: str = Field(
        default="generic",
        description="Coarse task bucket such as finance, legal, retrieval, document, or general."
    )
    tool_used: str = Field(description="Name of the tool selected.")
    tool_family: str = Field(
        default="generic",
        description="Coarse tool bucket such as search, calculator, finance, file, or generic."
    )
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierMemory(BaseModel):
    """Compact record of a failure-recovery pattern."""

    task_signature: str = Field(
        description="SHA-256 prefix of the normalized task text."
    )
    failure_pattern: str = Field(
        description="One-line description of what went wrong."
    )
    semantic_text: str = Field(
        default="",
        description="Normalized retrieval text for future semantic repair lookup."
    )
    task_family: str = Field(
        default="generic",
        description="Coarse task bucket such as finance, legal, retrieval, document, or general."
    )
    failure_family: str = Field(
        default="generic",
        description="Coarse failure bucket such as schema, hallucination, repetition, format, or missing_data."
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
    metadata: dict[str, Any] = Field(default_factory=dict)
