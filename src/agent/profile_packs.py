"""Minimal profile packs retained for OfficeQA-era compatibility helpers.

The active runtime is benchmark-driven and OfficeQA-specific. This module now
keeps only a small document-grounded profile surface for compatibility code
that still expects profile packs.
"""

from __future__ import annotations

from agent.contracts import ProfileContextPack


PROFILE_PACKS: dict[str, ProfileContextPack] = {
    "document_qa": ProfileContextPack(
        profile="document_qa",
        domain_summary=(
            "Document-grounded financial question answering task. Retrieve Treasury or official-source evidence "
            "first, preserve period and unit alignment, and prefer explicit numeric support over narrative guesses."
        ),
        content_rules=[
            "Ground the answer in retrieved document evidence before synthesis.",
            "Preserve period scope, aggregation semantics, and unit normalization.",
            "Use extracted values or quoted document support for the final answer.",
        ],
        section_requirements=["Answer", "Evidence summary", "Source references"],
        required_evidence_types=["document_evidence", "citations", "financial_numeric_support"],
        allowed_tools=[
            "calculator",
            "sum_values",
            "weighted_average",
            "pct_change",
            "cagr",
            "search_officeqa_documents",
            "fetch_officeqa_pages",
            "fetch_officeqa_table",
            "lookup_officeqa_rows",
            "lookup_officeqa_cells",
            "search_reference_corpus",
            "fetch_corpus_document",
            "fetch_reference_file",
            "list_reference_files",
        ],
        failure_modes=[
            "Answering from unsupported memory instead of extracted evidence.",
            "Confusing calendar-year, fiscal-year, monthly, or inflation-adjusted scopes.",
            "Dropping unit alignment or numeric provenance in the final answer.",
        ],
    ),
    "external_retrieval": ProfileContextPack(
        profile="external_retrieval",
        domain_summary="Source-backed retrieval task. Prefer official or benchmark-provided sources over broad web search.",
        content_rules=[
            "Retrieve source-backed evidence before answering.",
            "Keep unsupported extrapolation out of the final answer.",
        ],
        section_requirements=["Answer", "Retrieved evidence", "Sources"],
        required_evidence_types=["retrieved_facts", "citations"],
        allowed_tools=[
            "internet_search",
            "search_reference_corpus",
            "fetch_corpus_document",
            "fetch_reference_file",
            "list_reference_files",
        ],
        failure_modes=[
            "Using stale model knowledge where explicit retrieval was required.",
            "Returning an answer without identifiable source support.",
        ],
    ),
    "general": ProfileContextPack(
        profile="general",
        domain_summary="General fallback reasoning task. Keep the answer aligned to the requested output and available evidence.",
        content_rules=[
            "Answer directly from the provided context when possible.",
            "Use tools only when they close a clear evidence gap.",
        ],
        required_evidence_types=[],
        allowed_tools=["calculator", "fetch_reference_file", "list_reference_files"],
        failure_modes=["Tool use without a clear evidence gap."],
    ),
}


def get_profile_pack(profile: str) -> ProfileContextPack:
    return PROFILE_PACKS.get(profile, PROFILE_PACKS["general"])
