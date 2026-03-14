"""
Typed Runtime Contracts
=======================
Typed artifacts shared across the staged finance-first runtime.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
