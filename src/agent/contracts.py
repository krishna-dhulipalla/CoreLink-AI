"""
Typed Runtime Contracts
=======================
Typed artifacts shared across the staged finance-first runtime.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


CapabilityFlag = Literal[
    "needs_math",
    "needs_tables",
    "needs_files",
    "needs_live_data",
    "needs_options_engine",
    "needs_legal_reasoning",
    "requires_exact_format",
]

AmbiguityFlag = Literal[
    "legal_finance_overlap",
    "legal_options_overlap",
    "document_math_overlap",
    "document_live_overlap",
    "broad_multi_capability",
]

TaskProfile = Literal[
    "finance_quant",
    "finance_options",
    "legal_transactional",
    "document_qa",
    "external_retrieval",
    "general",
]

SolverStage = Literal[
    "PLAN",
    "GATHER",
    "COMPUTE",
    "SYNTHESIZE",
    "REVISE",
    "COMPLETE",
]

ReviewVerdict = Literal["pass", "revise", "backtrack"]
ExecutionTemplateId = Literal[
    "quant_inline_exact",
    "quant_with_tool_compute",
    "options_tool_backed",
    "legal_reasoning_only",
    "legal_with_document_evidence",
    "document_qa",
    "live_retrieval",
]
ReviewCadence = Literal["final_only", "milestone_and_final"]


class ProfileDecision(BaseModel):
    primary_profile: TaskProfile = "general"
    capability_flags: list[CapabilityFlag | str] = Field(default_factory=list)
    ambiguity_flags: list[AmbiguityFlag | str] = Field(default_factory=list)
    needs_external_data: bool = False
    needs_output_adapter: bool = False


class ExecutionTemplate(BaseModel):
    template_id: ExecutionTemplateId
    description: str
    allowed_stages: list[SolverStage] = Field(default_factory=list)
    default_initial_stage: SolverStage = "SYNTHESIZE"
    allowed_tool_names: list[str] = Field(default_factory=list)
    review_stages: list[SolverStage] = Field(default_factory=list)
    review_cadence: ReviewCadence = "final_only"
    answer_focus: list[str] = Field(default_factory=list)
    ambiguity_safe: bool = False


class AnswerContract(BaseModel):
    format: Literal["text", "json", "xml"] = "text"
    requires_adapter: bool = False
    raw_instruction: str = ""
    wrapper_key: str | None = None
    xml_root_tag: str | None = None
    schema_hint: dict[str, Any] = Field(default_factory=dict)
    exact_output_example: str | None = None
    content_rules: list[str] = Field(default_factory=list)
    section_requirements: list[str] = Field(default_factory=list)
    value_rules: dict[str, Any] = Field(default_factory=dict)


class ProfileContextPack(BaseModel):
    profile: TaskProfile
    domain_summary: str
    content_rules: list[str] = Field(default_factory=list)
    section_requirements: list[str] = Field(default_factory=list)
    required_evidence_types: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    reviewer_dimensions: dict[str, list[str]] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    task_brief: str = ""
    answer_contract: dict[str, Any] = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    inline_facts: dict[str, Any] = Field(default_factory=dict)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
    file_refs: list[str] = Field(default_factory=list)
    market_snapshot: dict[str, Any] = Field(default_factory=dict)
    derived_signals: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ToolCallEnvelope(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    type: str
    facts: dict[str, Any] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    verdict: ReviewVerdict = "revise"
    reasoning: str = ""
    missing_dimensions: list[str] = Field(default_factory=list)
    repair_target: Literal["gather", "compute", "synthesize", "final"] = "final"
