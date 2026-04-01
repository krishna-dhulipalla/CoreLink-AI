"""Typed contracts shared across the active engine."""

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
    "needs_analytical_reasoning",
    "needs_market_scenario",
    "needs_artifact_generation",
    "requires_exact_format",
    "needs_equity_research",
    "needs_portfolio_risk",
    "needs_event_analysis",
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
    "analytical_reasoning",
    "market_scenario",
    "document_qa",
    "external_retrieval",
    "unsupported_artifact",
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
RepairClass = Literal[
    "generic",
    "wrapper_only",
    "scalar_only",
    "missing_section",
    "missing_evidence",
    "missing_risk",
]
RiskVerdict = Literal["pass", "revise", "blocked"]
ComplianceVerdict = Literal["pass", "revise", "blocked"]
SourceClass = Literal["prompt", "retrieved", "derived", "assumption"]
AssumptionConfidence = Literal["low", "medium", "high"]
AssumptionReviewStatus = Literal["pending", "accepted", "rejected", "disclosed"]
RecommendationClass = Literal[
    "high_confidence_analytical_result",
    "scenario_dependent_recommendation",
    "insufficient_evidence_no_action",
]
ExecutionTemplateId = Literal[
    "quant_inline_exact",
    "quant_with_tool_compute",
    "options_tool_backed",
    "regulated_actionable_finance",
    "equity_research_report",
    "portfolio_risk_review",
    "event_driven_finance",
    "legal_reasoning_only",
    "legal_with_document_evidence",
    "document_qa",
    "live_retrieval",
]
ReviewCadence = Literal["final_only", "milestone_and_final"]
RetrievalOutcomeStatus = Literal[
    "",
    "ok",
    "empty",
    "garbled_binary",
    "irrelevant",
    "parse_error",
    "network_error",
    "unsupported_format",
]


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


class AssumptionRecord(BaseModel):
    key: str = ""
    assumption: str
    source: str
    confidence: AssumptionConfidence = "medium"
    requires_user_visible_disclosure: bool = False
    review_status: AssumptionReviewStatus = "pending"


class ProvenanceRecord(BaseModel):
    source_class: SourceClass
    source_id: str = ""
    extraction_method: str = ""
    tool_name: str | None = None


class DocumentEvidenceRecord(BaseModel):
    document_id: str
    citation: str = ""
    status: Literal["discovered", "indexed", "extracted"] = "discovered"
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    numeric_summaries: list[dict[str, Any]] = Field(default_factory=list)


class EvidencePack(BaseModel):
    task_constraints: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    target_period: str = ""
    relevant_rows: list[dict[str, Any]] = Field(default_factory=list)
    relevant_formulae: list[str] = Field(default_factory=list)
    required_output: dict[str, Any] = Field(default_factory=dict)
    unused_table_count: int = 0
    answer_contract: dict[str, Any] = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    prompt_facts: dict[str, Any] = Field(default_factory=dict)
    retrieved_facts: dict[str, Any] = Field(default_factory=dict)
    derived_facts: dict[str, Any] = Field(default_factory=dict)
    policy_context: dict[str, Any] = Field(default_factory=dict)
    document_evidence: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ToolCallEnvelope(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolQuality(BaseModel):
    is_synthetic: bool = False
    is_estimated: bool = False
    cache_hit: bool = False
    missing_fields: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    type: str
    facts: dict[str, Any] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)
    source: dict[str, Any] = Field(default_factory=dict)
    quality: ToolQuality = Field(default_factory=ToolQuality)
    retrieval_status: RetrievalOutcomeStatus | str = ""
    evidence_quality_score: float = 0.0
    errors: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    verdict: ReviewVerdict = "revise"
    reasoning: str = ""
    missing_dimensions: list[str] = Field(default_factory=list)
    repair_target: Literal["gather", "compute", "synthesize", "final"] = "final"
    repair_class: RepairClass = "generic"


class RiskResult(BaseModel):
    verdict: RiskVerdict = "revise"
    reasoning: str = ""
    violation_codes: list[str] = Field(default_factory=list)
    risk_findings: list[str] = Field(default_factory=list)
    required_disclosures: list[str] = Field(default_factory=list)
    recommendation_class: RecommendationClass = "scenario_dependent_recommendation"
    repair_target: Literal["compute", "final"] = "compute"


class ComplianceResult(BaseModel):
    verdict: ComplianceVerdict = "revise"
    reasoning: str = ""
    violation_codes: list[str] = Field(default_factory=list)
    policy_findings: list[str] = Field(default_factory=list)
    required_disclosures: list[str] = Field(default_factory=list)
    repair_target: Literal["final"] = "final"


class ArtifactCheckpoint(BaseModel):
    template_id: str = ""
    checkpoint_stage: str = ""
    reason: str = ""
    evidence_pack: dict[str, Any] = Field(default_factory=dict)
    assumption_ledger: list[dict[str, Any]] = Field(default_factory=list)
    provenance_map: dict[str, dict[str, Any]] = Field(default_factory=dict)
    last_tool_result: dict[str, Any] | None = None
    draft_answer: str = ""
    stage_outputs: dict[str, Any] = Field(default_factory=dict)
    review_feedback: dict[str, Any] | None = None


ExecutionMode = Literal[
    "exact_fast_path",
    "tool_compute",
    "retrieval_augmented_analysis",
    "document_grounded_analysis",
    "advisory_analysis",
]

ComplexityTier = Literal["simple_exact", "structured_analysis", "complex_qualitative"]
ReviewMode = Literal["exact_quant", "tool_compute", "qualitative_advisory", "document_grounded", "analytical_reasoning"]
CompletionMode = Literal["scalar_or_json", "compact_sections", "advisory_memo", "document_grounded", "long_form_derivation", "capability_gap"]
EvidenceStrategy = Literal["minimal_exact", "compact_prompt", "document_first", "retrieval_first", "scenario_first"]
SideEffectLevel = Literal["read_only", "transactional", "blocked"]


class TaskIntent(BaseModel):
    task_family: TaskProfile = "general"
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
    tool_role: str = ""
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
    widened_families: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    pending_tools: list[str] = Field(default_factory=list)
    blocked_families: list[str] = Field(default_factory=list)
    ace_events: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    stop_reason: str = ""


class SourceBundle(BaseModel):
    task_text: str = ""
    focus_query: str = ""
    target_period: str = ""
    entities: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    source_files_expected: list[str] = Field(default_factory=list)
    source_files_found: list[dict[str, Any]] = Field(default_factory=list)
    inline_facts: dict[str, Any] = Field(default_factory=dict)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)


class RetrievalIntent(BaseModel):
    entity: str = ""
    metric: str = ""
    period: str = ""
    document_family: str = ""
    aggregation_shape: str = ""
    must_include_terms: list[str] = Field(default_factory=list)
    must_exclude_terms: list[str] = Field(default_factory=list)
    query_candidates: list[str] = Field(default_factory=list)


class CuratedContext(BaseModel):
    objective: str = ""
    facts_in_use: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requested_output: dict[str, Any] = Field(default_factory=dict)
    provenance_summary: dict[str, Any] = Field(default_factory=dict)


class EvidenceSufficiency(BaseModel):
    source_family: str = ""
    period_scope: str = ""
    aggregation_type: str = ""
    entity_scope: str = ""
    is_sufficient: bool = False
    missing_dimensions: list[str] = Field(default_factory=list)
    rationale: str = ""


class ReviewPacket(BaseModel):
    task_text: str = ""
    answer_text: str = ""
    answer_contract: dict[str, Any] = Field(default_factory=dict)
    tool_findings: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    evidence_sufficiency: dict[str, Any] = Field(default_factory=dict)


class RetrievalAction(BaseModel):
    action: Literal["tool", "answer"] = "tool"
    tool_name: str = ""
    query: str = ""
    url: str = ""
    document_id: str = ""
    path: str = ""
    page_start: int = 0
    page_limit: int = 5
    row_offset: int = 0
    row_limit: int = 200
    chunk_start: int = 0
    chunk_limit: int = 3
    rationale: str = ""


class ProgressSignature(BaseModel):
    execution_mode: str = ""
    selected_tools: list[str] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    artifact_signature: str = ""
    contract_status: str = ""
    signature: str = ""


class UnsupportedCapabilityReport(BaseModel):
    task_family: str = ""
    requested_capability: str = ""
    reason: str = ""
    suggested_scope: str = "finance-first"


class TraceIdentity(BaseModel):
    request_id: str = ""
    task_id: str = ""
    context_id: str = ""
    benchmark_uid: str = ""


class ExecutionJournal(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    routed_tool_families: list[str] = Field(default_factory=list)
    revision_count: int = 0
    self_reflection_count: int = 0
    retrieval_iterations: int = 0
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_citations: list[str] = Field(default_factory=list)
    final_artifact_signature: str = ""
    progress_signatures: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: str = ""
    contract_collapse_attempts: int = 0


class QualityReport(BaseModel):
    verdict: Literal["pass", "revise", "fail"] = "pass"
    reasoning: str = ""
    missing_dimensions: list[str] = Field(default_factory=list)
    targeted_fix_prompt: str = ""
    score: float = 0.9
    stop_reason: str = ""
