"""
Profile Context Packs
=====================
Explicit domain guidance and review rules for the staged finance-first runtime.
"""

from __future__ import annotations

from agent.contracts import ProfileContextPack, TaskProfile


PROFILE_PACKS: dict[TaskProfile, ProfileContextPack] = {
    "finance_quant": ProfileContextPack(
        profile="finance_quant",
        domain_summary=(
            "Finance quantitative task. Extract prompt-contained values first, keep units and signs exact, "
            "and prefer exact computation over narrative explanation."
        ),
        content_rules=[
            "Use formulas and inline tables from the prompt before calling tools.",
            "Keep the analytical summary compact and exact.",
            "Do not introduce external assumptions unless explicitly stated.",
        ],
        required_evidence_types=["inline_facts", "tables", "formulas"],
        allowed_tools=["calculator", "fetch_reference_file", "list_reference_files"],
        failure_modes=[
            "Missing or misread values from inline tables.",
            "Dropping the exact output contract when a JSON wrapper is required.",
        ],
    ),
    "finance_options": ProfileContextPack(
        profile="finance_options",
        domain_summary=(
            "Options and volatility task. Ground the recommendation in volatility context, strategy structure, "
            "Greeks, breakevens, premium direction, and practical risk controls."
        ),
        content_rules=[
            "Use tool-backed calculations for the primary strategy whenever the needed parameters are available.",
            "Compare at least one alternative strategy with concrete tradeoffs, not just name-dropping.",
            "Keep credit/debit direction, breakevens, and risk controls explicit.",
        ],
        section_requirements=[
            "Recommendation",
            "Primary strategy",
            "Alternative strategy comparison",
            "Key Greeks and breakevens",
            "Risk management",
        ],
        required_evidence_types=["market_snapshot", "derived_signals", "tool_results"],
        allowed_tools=[
            "calculator",
            "black_scholes_price",
            "option_greeks",
            "mispricing_analysis",
            "analyze_strategy",
            "get_options_chain",
            "get_iv_surface",
            "get_expirations",
            "fetch_reference_file",
            "list_reference_files",
        ],
        failure_modes=[
            "Pure prose comparison without tool-backed quantitative analysis.",
            "Missing Greeks, breakevens, or risk management.",
            "Using an options tool with hidden assumptions that are not disclosed.",
        ],
        reviewer_dimensions={
            "recommendation": ["seller", "short vol", "buy", "net seller", "net buyer"],
            "primary strategy": ["strategy", "strangle", "straddle", "condor", "spread"],
            "alternative strategy comparison": ["alternative", "compared", "instead", "iron condor", "credit spread", "strangle"],
            "key Greeks and breakevens": ["delta", "gamma", "theta", "vega", "breakeven", "break-even"],
            "risk management": ["risk", "hedge", "sizing", "max loss", "stop-loss"],
        },
    ),
    "legal_transactional": ProfileContextPack(
        profile="legal_transactional",
        domain_summary=(
            "Transactional legal task. Focus on structure alternatives, tax consequences, liability allocation, "
            "regulatory and diligence risks, key assumptions, and next steps."
        ),
        content_rules=[
            "Do not invent statutes, filings, or source documents.",
            "Prioritize practical structure tradeoffs and execution constraints.",
            "Use tools only when the prompt explicitly requires files, current facts, or exact calculations.",
        ],
        section_requirements=[
            "Structure options",
            "Tax consequences",
            "Liability protection",
            "Regulatory and diligence risks",
            "Key open questions and assumptions",
            "Recommended next steps",
        ],
        required_evidence_types=["inline_facts", "file_refs"],
        allowed_tools=["calculator", "fetch_reference_file", "list_reference_files"],
        failure_modes=[
            "Generic memo that omits diligence or liability allocation.",
            "Hallucinated legal authorities or invented file references.",
            "First-step tool detours when prompt-contained reasoning is sufficient.",
        ],
        reviewer_dimensions={
            "structure options": ["asset", "stock", "merger", "triangular", "carve-out", "hybrid"],
            "tax consequences": ["tax", "deferral", "step-up", "capital gain", "basis"],
            "liability protection": ["indemn", "escrow", "reps", "warrant", "insurance", "holdback"],
            "regulatory and diligence risks": ["regulatory", "compliance", "diligence", "approval", "eu", "us", "hsr"],
            "key open questions and assumptions": ["assumption", "confirm", "determine", "assess", "willingness", "risk tolerance"],
            "recommended next steps": ["next step", "engage", "draft", "timeline", "recommend", "immediately"],
        },
    ),
    "document_qa": ProfileContextPack(
        profile="document_qa",
        domain_summary=(
            "Document question answering task. Ground the answer in extracted file evidence and keep source references visible."
        ),
        content_rules=[
            "Summarize extracted evidence instead of repeating raw document bodies.",
            "Call file tools only when a file or URL is actually present.",
        ],
        section_requirements=["Answer", "Evidence summary", "Source references"],
        required_evidence_types=["file_refs", "tables", "citations"],
        allowed_tools=["calculator", "fetch_reference_file", "list_reference_files"],
        failure_modes=[
            "Answering from raw guesses without extracted evidence.",
            "Dumping large file excerpts instead of a grounded summary.",
        ],
        reviewer_dimensions={
            "answer": ["answer", "therefore", "based on"],
            "evidence summary": ["evidence", "document", "table", "page", "row"],
            "source references": ["source", "page", "url", "file"],
        },
    ),
    "external_retrieval": ProfileContextPack(
        profile="external_retrieval",
        domain_summary=(
            "External retrieval task. Fetch current or source-backed facts only when the prompt explicitly asks for them."
        ),
        content_rules=[
            "Do not browse when prompt-contained evidence is enough.",
            "Cite retrieved facts directly and keep unsupported extrapolation out of the answer.",
        ],
        section_requirements=["Answer", "Retrieved evidence", "Sources"],
        required_evidence_types=["citations", "tool_results"],
        allowed_tools=["calculator", "internet_search", "fetch_reference_file", "list_reference_files"],
        failure_modes=[
            "Using stale model knowledge where explicit retrieval was required.",
            "Returning ungrounded prose without identifiable sources.",
        ],
        reviewer_dimensions={
            "answer": ["answer", "based on", "current"],
            "retrieved evidence": ["retrieved", "found", "result", "evidence"],
            "sources": ["source", "url", "http"],
        },
    ),
    "general": ProfileContextPack(
        profile="general",
        domain_summary="General reasoning task. Answer directly using prompt-contained facts first.",
        content_rules=[
            "Keep the answer focused on the user's actual request.",
            "Use tools only when they add evidence that is clearly missing.",
        ],
        required_evidence_types=[],
        allowed_tools=["calculator", "fetch_reference_file", "list_reference_files"],
        failure_modes=["Tool use without a clear evidence gap."],
    ),
}


def get_profile_pack(profile: str) -> ProfileContextPack:
    return PROFILE_PACKS.get(profile, PROFILE_PACKS["general"])
