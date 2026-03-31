"""OfficeQA benchmark adapter."""

from __future__ import annotations

import os
import re
from typing import Any

from agent.contracts import AnswerContract, TaskIntent

from .base import truthy_env

OFFICEQA_REGISTRY_TOOL_NAME_ALLOWLIST = {
    "calculator",
    "sum_values",
    "weighted_average",
    "pct_change",
    "cagr",
    "annualize_return",
    "annualize_volatility",
    "bond_price_yield",
    "duration_convexity",
    "internet_search",
    "search_reference_corpus",
    "fetch_corpus_document",
    "fetch_reference_file",
    "list_reference_files",
}
OFFICEQA_REGISTRY_ALLOWED_FAMILIES = {
    "document_retrieval",
    "external_retrieval",
    "exact_compute",
}
OFFICEQA_RUNTIME_ALLOWED_FAMILIES = {
    "document_retrieval",
    "external_retrieval",
    "analytical_reasoning",
    "exact_compute",
}
OFFICEQA_ALLOWED_EXACT_COMPUTE_TOOLS = {
    "calculator",
    "sum_values",
    "weighted_average",
    "pct_change",
    "cagr",
}
OFFICEQA_REQUIRED_SOURCE_FAMILIES = {
    "treasury_bulletin",
    "reference_file",
    "official_government_document",
}
OFFICEQA_EXCLUDED_RETRIEVAL_TERMS = {
    "stock price",
    "share price",
    "earnings call",
    "quarterly results",
    "10-k",
    "10q",
    "etf",
    "wikipedia",
    "blog",
    "quiz",
    "flashcards",
}


def looks_like_officeqa_prompt(text: str) -> bool:
    lowered = re.sub(r"[^a-z0-9%]+", " ", (text or "").lower())
    if not lowered:
        return False

    score = 0
    if re.search(r"\b(19[3-9]\d|20[0-2]\d)\b", lowered):
        score += 1
    if any(
        token in lowered
        for token in (
            "calendar year",
            "fiscal year",
            "individual calendar months",
            "annual average",
            "reported values",
        )
    ):
        score += 1
    if any(
        token in lowered
        for token in (
            "expenditures",
            "outlays",
            "receipts",
            "public debt",
            "nominal dollars",
            "cpi u",
            "cpi-u",
            "percent change",
        )
    ):
        score += 1
    if any(
        token in lowered
        for token in (
            "using specifically only",
            "according to",
            "rounded to the nearest hundredths",
            "reported as a percent value",
            "million",
        )
    ):
        score += 1
    if any(
        token in lowered
        for token in (
            "treasury bulletin",
            "monthly treasury statement",
            "federal reserve bank of minneapolis",
            "veterans administration",
            "national defense",
        )
    ):
        score += 1
    return score >= 3


def _xml_contract_requested(task_text: str, benchmark_name: str) -> bool:
    if benchmark_name == "officeqa":
        return True
    if truthy_env("OFFICEQA_FINAL_ANSWER_TAGS") or truthy_env("OFFICEQA_XML_OUTPUT"):
        return looks_like_officeqa_prompt(task_text)
    if truthy_env("BENCHMARK_STATELESS") and os.getenv("OFFICEQA_CORPUS_DIR", "").strip():
        return looks_like_officeqa_prompt(task_text)
    return False


def build_officeqa_overrides(task_text: str, benchmark_name: str) -> dict[str, Any]:
    officeqa_like = looks_like_officeqa_prompt(task_text)
    xml_contract = _xml_contract_requested(task_text, benchmark_name)
    explicit_benchmark = benchmark_name == "officeqa"
    allow_web_fallback = os.getenv("OFFICEQA_ALLOW_WEB_FALLBACK", "last_fallback").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "never",
    }
    adapter_active = explicit_benchmark or xml_contract or officeqa_like
    return {
        "benchmark_adapter": "officeqa" if adapter_active else "",
        "officeqa_mode": explicit_benchmark or officeqa_like,
        "officeqa_like_prompt": officeqa_like,
        "officeqa_xml_contract": xml_contract,
        "officeqa_allow_web_fallback": allow_web_fallback,
        "benchmark_policy": officeqa_runtime_policy(allow_web_fallback) if adapter_active else {},
    }


def officeqa_answer_contract() -> AnswerContract:
    return AnswerContract(
        format="xml",
        requires_adapter=True,
        raw_instruction="OfficeQA benchmark requires <REASONING> and <FINAL_ANSWER> XML tags.",
        xml_root_tag="FINAL_ANSWER",
        section_requirements=["REASONING", "FINAL_ANSWER"],
        content_rules=[
            "Place step-by-step reasoning inside <REASONING> tags.",
            "Place only the final exact value or exact string answer inside <FINAL_ANSWER> tags.",
            "Do not include units, labels, or extra explanation inside <FINAL_ANSWER>.",
        ],
        value_rules={
            "reasoning_tag": "REASONING",
            "final_answer_tag": "FINAL_ANSWER",
            "final_answer_only": True,
            "preserve_numeric_format": True,
        },
    )


def officeqa_registry_policy() -> dict[str, Any]:
    return {
        "allowed_tool_names": sorted(OFFICEQA_REGISTRY_TOOL_NAME_ALLOWLIST),
        "allowed_families": sorted(OFFICEQA_REGISTRY_ALLOWED_FAMILIES),
    }


def officeqa_runtime_policy(allow_web_fallback: bool = True) -> dict[str, Any]:
    return {
        "answer_contract": {
            "format": "xml",
            "xml_root_tag": "FINAL_ANSWER",
        },
        "allowed_families": sorted(OFFICEQA_RUNTIME_ALLOWED_FAMILIES),
        "allowed_exact_compute_tools": sorted(OFFICEQA_ALLOWED_EXACT_COMPUTE_TOOLS),
        "allowed_external_search_tools": ["internet_search"],
        "required_source_families": sorted(OFFICEQA_REQUIRED_SOURCE_FAMILIES),
        "excluded_retrieval_terms": sorted(OFFICEQA_EXCLUDED_RETRIEVAL_TERMS),
        "validation_dimensions": [
            "source family grounding",
            "period scope",
            "aggregation semantics",
            "entity scope",
            "metric scope",
            "numeric or quoted support",
        ],
        "output_normalization": {
            "reasoning_tag": "REASONING",
            "final_answer_tag": "FINAL_ANSWER",
            "final_answer_only": True,
            "preserve_numeric_format": True,
        },
        "allow_web_fallback": allow_web_fallback,
    }


def officeqa_tool_selection_active(task_family: str, benchmark_overrides: dict[str, Any] | None = None) -> bool:
    overrides = dict(benchmark_overrides or {})
    if overrides.get("benchmark_adapter") != "officeqa":
        return False
    if overrides.get("officeqa_mode") is True:
        return True
    return bool(overrides.get("officeqa_like_prompt")) and task_family == "document_qa"


def officeqa_descriptor_allowed(descriptor: dict[str, Any], benchmark_overrides: dict[str, Any] | None = None) -> bool:
    policy = dict((benchmark_overrides or {}).get("benchmark_policy") or {})
    family = str(descriptor.get("tool_family", "") or "")
    tool_name = str(descriptor.get("tool_name", "") or "")
    role = str(descriptor.get("tool_role", "") or "")
    allowed_families = set(policy.get("allowed_families", []))
    if allowed_families and family not in allowed_families:
        return False
    if family == "document_retrieval":
        return True
    if family == "external_retrieval":
        allowed_search_tools = set(policy.get("allowed_external_search_tools", []))
        return role == "search" and (not allowed_search_tools or tool_name in allowed_search_tools)
    if family == "exact_compute":
        allowed_exact_compute_tools = set(policy.get("allowed_exact_compute_tools", []))
        return not allowed_exact_compute_tools or tool_name in allowed_exact_compute_tools
    if family == "analytical_reasoning":
        return False
    return False


def officeqa_task_intent(task_text: str, capability_flags: list[str], benchmark_overrides: dict[str, Any] | None = None) -> TaskIntent | None:
    overrides = dict(benchmark_overrides or {})
    if overrides.get("benchmark_adapter") != "officeqa":
        return None
    normalized = (task_text or "").lower()
    needs_compute = "needs_math" in capability_flags or "needs_analytical_reasoning" in capability_flags or any(
        token in normalized
        for token in (
            "sum",
            "total sum",
            "difference",
            "absolute difference",
            "percent change",
            "rounded",
            "inflation",
            "adjusted",
        )
    )
    tool_families = ["document_retrieval"]
    if needs_compute:
        tool_families.extend(["analytical_reasoning", "exact_compute"])
    return TaskIntent(
        task_family="document_qa",
        execution_mode="document_grounded_analysis",
        complexity_tier="structured_analysis",
        tool_families_needed=tool_families,
        evidence_strategy="document_first",
        review_mode="document_grounded",
        completion_mode="document_grounded",
        routing_rationale="OfficeQA runs must retrieve grounded Treasury evidence before synthesis and only compute from extracted document values.",
        confidence=0.97 if overrides.get("benchmark_name") == "officeqa" else 0.95,
        planner_source="heuristic",
    )
