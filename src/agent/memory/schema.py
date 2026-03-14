"""
Memory Schemas
==============
Versioned persistence records for the staged finance-first runtime.

The active runtime persists three compact record types:
  - RunMemory:    one record per completed graph run
  - ToolMemory:   one record per normalized tool execution
  - ReviewMemory: one record per reviewer verdict

These records are intentionally small and schema-versioned so future runtime
changes can replace the on-disk format without inheriting stale coordinator-era
data.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.contracts import TaskProfile

MEMORY_SCHEMA_VERSION = 2


def task_signature(text: str) -> str:
    """Create a stable, short hash of the task text for dedup / retrieval."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def normalize_memory_text(text: str, max_len: int = 500) -> str:
    """Normalize free-text fields so future retrieval/indexing has clean input."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:max_len]


def infer_memory_family(task_profile: str | None, text: str = "") -> str:
    """Map runtime profile into a broader persistence bucket."""
    normalized = (task_profile or "").strip().lower()
    if normalized in {"finance_quant", "finance_options"}:
        return "finance"
    if normalized == "legal_transactional":
        return "legal"
    if normalized == "document_qa":
        return "document"
    if normalized == "external_retrieval":
        return "retrieval"

    content = normalize_memory_text(text, max_len=800).lower()
    if any(token in content for token in ("option", "greeks", "iv", "volatility", "roe", "roa", "valuation")):
        return "finance"
    if any(token in content for token in ("merger", "acquisition", "compliance", "liability", "indemnity", "tax")):
        return "legal"
    if any(token in content for token in ("pdf", "document", "table", "page", "extract", "file")):
        return "document"
    if any(token in content for token in ("latest", "current", "search", "look up", "citation", "source")):
        return "retrieval"
    return "general"


class RunMemory(BaseModel):
    """Compact record of one staged-runtime execution."""

    task_signature: str
    task_summary: str
    semantic_text: str = ""
    task_profile: TaskProfile | str = "general"
    task_family: str = "general"
    capability_flags: list[str] = Field(default_factory=list)
    route_path: list[str] = Field(default_factory=list)
    stage_history: list[str] = Field(default_factory=list)
    answer_format: Literal["text", "json", "xml"] | str = "text"
    success: bool
    tool_call_count: int = 0
    review_cycle_count: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolMemory(BaseModel):
    """Compact record of one structured tool execution."""

    task_signature: str
    task_profile: TaskProfile | str = "general"
    task_family: str = "general"
    solver_stage: str = "COMPUTE"
    tool_name: str
    result_type: str
    semantic_text: str = ""
    arguments_json: dict[str, Any] = Field(default_factory=dict)
    fact_keys: list[str] = Field(default_factory=list)
    error_count: int = 0
    success: bool = True
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewMemory(BaseModel):
    """Compact record of one reviewer decision."""

    task_signature: str
    task_profile: TaskProfile | str = "general"
    task_family: str = "general"
    review_stage: str = "SYNTHESIZE"
    verdict: Literal["pass", "revise", "backtrack"]
    repair_target: Literal["gather", "compute", "synthesize", "final"] = "final"
    missing_dimensions: list[str] = Field(default_factory=list)
    reasoning: str = ""
    success: bool = True
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
