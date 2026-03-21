"""Contracts for the V4 hybrid routing runtime."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskFamily = Literal[
    "finance_quant",
    "finance_options",
    "legal_transactional",
    "document_qa",
    "external_retrieval",
    "general",
]

ExecutionMode = Literal[
    "exact_fast_path",
    "tool_compute",
    "retrieval_augmented_analysis",
    "document_grounded_analysis",
    "advisory_analysis",
]

ComplexityTier = Literal["simple_exact", "structured_analysis", "complex_qualitative"]
ReviewMode = Literal["exact_quant", "tool_compute", "qualitative_advisory", "document_grounded"]
CompletionMode = Literal["scalar_or_json", "compact_sections", "advisory_memo", "document_grounded"]
EvidenceStrategy = Literal["minimal_exact", "compact_prompt", "document_first", "retrieval_first"]
SideEffectLevel = Literal["read_only", "transactional", "blocked"]


class TaskIntent(BaseModel):
    task_family: TaskFamily = "general"
    execution_mode: ExecutionMode = "advisory_analysis"
    complexity_tier: ComplexityTier = "structured_analysis"
    tool_families_needed: list[str] = Field(default_factory=list)
    evidence_strategy: EvidenceStrategy = "compact_prompt"
    review_mode: ReviewMode = "qualitative_advisory"
    completion_mode: CompletionMode = "compact_sections"
    routing_rationale: str = ""
    confidence: float = 0.7
    planner_source: Literal["fast_path", "heuristic", "llm"] = "heuristic"


class CapabilityDescriptor(BaseModel):
    tool_name: str
    tool_family: str
    domain_tags: list[str] = Field(default_factory=list)
    input_shape: str = "generic"
    side_effect_level: SideEffectLevel = "read_only"
    supports_live_data: bool = False
    supports_documents: bool = False
    supports_exact_compute: bool = False
    priority: int = 50


class ACEEvent(BaseModel):
    family: str
    status: Literal["synthesized", "blocked", "skipped"] = "skipped"
    reason: str = ""
    tool_name: str | None = None


class ToolPlan(BaseModel):
    tool_families_needed: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    pending_tools: list[str] = Field(default_factory=list)
    blocked_families: list[str] = Field(default_factory=list)
    ace_events: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SourceBundle(BaseModel):
    task_text: str = ""
    focus_query: str = ""
    target_period: str = ""
    entities: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    inline_facts: dict[str, Any] = Field(default_factory=dict)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)


class CuratedContext(BaseModel):
    objective: str = ""
    facts_in_use: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requested_output: dict[str, Any] = Field(default_factory=dict)
    provenance_summary: dict[str, Any] = Field(default_factory=dict)


class ExecutionJournal(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    routed_tool_families: list[str] = Field(default_factory=list)
    revision_count: int = 0
    self_reflection_count: int = 0
    final_artifact_signature: str = ""


class QualityReport(BaseModel):
    verdict: Literal["pass", "revise", "fail"] = "pass"
    reasoning: str = ""
    missing_dimensions: list[str] = Field(default_factory=list)
    targeted_fix_prompt: str = ""
    score: float = 0.9
