"""
Execution Template Library
==========================
Static, versioned execution templates selected deterministically at runtime.
"""

from __future__ import annotations

from agent.contracts import ExecutionTemplate, ExecutionTemplateId


TEMPLATE_LIBRARY: dict[ExecutionTemplateId, ExecutionTemplate] = {
    "quant_inline_exact": ExecutionTemplate(
        template_id="quant_inline_exact",
        description="Inline quantitative reasoning with exact computation from prompt-contained evidence.",
        allowed_stages=["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
        default_initial_stage="COMPUTE",
        allowed_tool_names=["calculator"],
        review_stages=["COMPUTE", "SYNTHESIZE"],
        review_cadence="milestone_and_final",
        answer_focus=[
            "Extract prompt-contained values exactly.",
            "Keep numeric computation compact and precise.",
            "Honor exact output contracts without narrative drift.",
        ],
        ambiguity_safe=True,
    ),
    "quant_with_tool_compute": ExecutionTemplate(
        template_id="quant_with_tool_compute",
        description="Quantitative reasoning that may require one evidence-gather or compute tool step before synthesis.",
        allowed_stages=["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
        default_initial_stage="COMPUTE",
        allowed_tool_names=[
            "calculator",
            "fetch_reference_file",
            "list_reference_files",
            "resolve_financial_entity",
            "get_price_history",
            "get_company_fundamentals",
            "get_corporate_actions",
            "get_yield_curve",
            "get_returns",
            "get_financial_statements",
            "get_statement_line_items",
            "weighted_average",
            "sum_values",
            "pct_change",
            "cagr",
            "annualize_return",
            "annualize_volatility",
            "bond_price_yield",
            "duration_convexity",
            "du_pont_analysis",
            "liquidity_ratio_pack",
            "valuation_multiples_compare",
            "dcf_sensitivity_grid",
        ],
        review_stages=["GATHER", "COMPUTE", "SYNTHESIZE"],
        review_cadence="milestone_and_final",
        answer_focus=[
            "Use prompt evidence first, then targeted compute or file evidence if justified.",
            "Keep intermediate compute artifacts compact and reviewable.",
        ],
        ambiguity_safe=True,
    ),
    "options_tool_backed": ExecutionTemplate(
        template_id="options_tool_backed",
        description="Options workflow requiring structured tool-backed compute before final synthesis.",
        allowed_stages=["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
        default_initial_stage="COMPUTE",
        allowed_tool_names=[
            "calculator",
            "black_scholes_price",
            "option_greeks",
            "mispricing_analysis",
            "analyze_strategy",
            "get_options_chain",
            "get_iv_surface",
            "get_expirations",
            "resolve_financial_entity",
            "get_price_history",
            "get_returns",
            "fetch_reference_file",
            "list_reference_files",
        ],
        review_stages=["COMPUTE", "SYNTHESIZE"],
        review_cadence="milestone_and_final",
        answer_focus=[
            "Ground the primary strategy in structured tool results.",
            "Compare at least one alternative with concrete tradeoffs.",
            "Keep Greeks, premium direction, breakevens, and risk controls explicit.",
        ],
    ),
    "legal_reasoning_only": ExecutionTemplate(
        template_id="legal_reasoning_only",
        description="Transactional legal reasoning with no routine external retrieval or file evidence.",
        allowed_stages=["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
        default_initial_stage="SYNTHESIZE",
        allowed_tool_names=["calculator"],
        review_stages=["SYNTHESIZE"],
        review_cadence="final_only",
        answer_focus=[
            "Cover structure, tax, liability, diligence, assumptions, and next steps.",
            "Do not invent authorities or source documents.",
        ],
        ambiguity_safe=True,
    ),
    "legal_with_document_evidence": ExecutionTemplate(
        template_id="legal_with_document_evidence",
        description="Transactional legal reasoning supported by targeted document evidence.",
        allowed_stages=["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
        default_initial_stage="GATHER",
        allowed_tool_names=["calculator", "fetch_reference_file", "list_reference_files"],
        review_stages=["GATHER", "SYNTHESIZE"],
        review_cadence="milestone_and_final",
        answer_focus=[
            "Retrieve only targeted document evidence and preserve source references.",
            "Start with metadata or a narrow extraction window before deeper reads.",
            "Synthesize legal structure tradeoffs without raw document dumping.",
        ],
        ambiguity_safe=True,
    ),
    "document_qa": ExecutionTemplate(
        template_id="document_qa",
        description="Document-grounded QA with targeted evidence gathering first.",
        allowed_stages=["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
        default_initial_stage="GATHER",
        allowed_tool_names=["calculator", "fetch_reference_file", "list_reference_files"],
        review_stages=["GATHER", "SYNTHESIZE"],
        review_cadence="milestone_and_final",
        answer_focus=[
            "Ground the answer in extracted evidence.",
            "Start with a narrow extraction window instead of pulling the whole document body.",
            "Prefer evidence summaries and source references over raw excerpts.",
        ],
        ambiguity_safe=True,
    ),
    "live_retrieval": ExecutionTemplate(
        template_id="live_retrieval",
        description="Explicit current-data retrieval followed by concise synthesis with sources.",
        allowed_stages=["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
        default_initial_stage="GATHER",
        allowed_tool_names=[
            "internet_search",
            "fetch_reference_file",
            "list_reference_files",
            "calculator",
            "resolve_financial_entity",
            "get_price_history",
            "get_company_fundamentals",
            "get_corporate_actions",
            "get_yield_curve",
            "get_returns",
            "get_financial_statements",
            "get_statement_line_items",
        ],
        review_stages=["GATHER", "SYNTHESIZE"],
        review_cadence="milestone_and_final",
        answer_focus=[
            "Retrieve current facts only when explicitly requested.",
            "Keep sources visible and avoid unsupported extrapolation.",
        ],
        ambiguity_safe=True,
    ),
}


def get_execution_template(template_id: str) -> ExecutionTemplate:
    return TEMPLATE_LIBRARY.get(template_id, TEMPLATE_LIBRARY["legal_reasoning_only"])
