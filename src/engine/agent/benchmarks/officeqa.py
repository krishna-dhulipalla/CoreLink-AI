"""OfficeQA benchmark adapter."""

from __future__ import annotations

import os
from typing import Any

from engine.agent.contracts import AnswerContract, TaskIntent

OFFICEQA_REGISTRY_TOOL_NAME_ALLOWLIST = {
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

_OFFICEQA_ANALYSIS_PATTERNS: dict[str, tuple[str, ...]] = {
    "inflation_adjustment": (
        "inflation adjusted",
        "inflation-adjusted",
        "adjusted for inflation",
        "cpi",
        "consumer price index",
        "constant dollars",
        "real dollars",
    ),
    "statistical_analysis": (
        "regression",
        "correlation",
        "standard deviation",
        "std dev",
        "std. dev",
        "variance",
        "covariance",
    ),
    "time_series_forecasting": (
        "forecast",
        "forecasting",
        "projected",
        "projection",
        "predict",
        "trend",
        "time series",
    ),
    "weighted_average": (
        "weighted average",
        "weighted mean",
    ),
    "risk_metric": (
        "value at risk",
        "var",
        "volatility",
    ),
}


def officeqa_analysis_modes(task_text: str, capability_flags: list[str] | None = None) -> list[str]:
    lowered = (task_text or "").lower()
    flags = set(capability_flags or [])
    modes: list[str] = []
    if any(token in lowered for token in _OFFICEQA_ANALYSIS_PATTERNS["inflation_adjustment"]):
        modes.append("inflation_adjustment")
    if any(token in lowered for token in _OFFICEQA_ANALYSIS_PATTERNS["statistical_analysis"]):
        modes.append("statistical_analysis")
    if any(token in lowered for token in _OFFICEQA_ANALYSIS_PATTERNS["time_series_forecasting"]):
        modes.append("time_series_forecasting")
    if any(token in lowered for token in _OFFICEQA_ANALYSIS_PATTERNS["weighted_average"]):
        modes.append("weighted_average")
    if any(token in lowered for token in _OFFICEQA_ANALYSIS_PATTERNS["risk_metric"]):
        modes.append("risk_metric")
    if "needs_math" in flags and not modes:
        modes.append("numeric_compute")
    if not modes:
        modes.append("simple_extraction")
    return list(dict.fromkeys(modes))


def build_officeqa_overrides(task_text: str, benchmark_name: str) -> dict[str, Any]:
    explicit_benchmark = benchmark_name == "officeqa"
    allow_web_fallback = os.getenv("OFFICEQA_ALLOW_WEB_FALLBACK", "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
        "never",
    }
    return {
        "benchmark_adapter": "officeqa" if explicit_benchmark else "",
        "officeqa_mode": explicit_benchmark,
        "officeqa_xml_contract": explicit_benchmark,
        "benchmark_policy": officeqa_runtime_policy(allow_web_fallback) if explicit_benchmark else {},
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
    _ = task_family
    overrides = dict(benchmark_overrides or {})
    return overrides.get("benchmark_adapter") == "officeqa"


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
    analysis_modes = officeqa_analysis_modes(task_text, capability_flags)
    needs_compute = (
        "needs_math" in capability_flags
        or "needs_analytical_reasoning" in capability_flags
        or any(
            token in normalized
            for token in (
                "sum",
                "total sum",
                "difference",
                "absolute difference",
                "percent change",
                "rounded",
                "average",
            )
        )
        or any(
            mode in analysis_modes
            for mode in (
                "inflation_adjustment",
                "statistical_analysis",
                "time_series_forecasting",
                "weighted_average",
                "risk_metric",
                "numeric_compute",
            )
        )
    )
    needs_reasoning = (
        "needs_analytical_reasoning" in capability_flags
        or any(
            mode in analysis_modes
            for mode in ("statistical_analysis", "time_series_forecasting", "risk_metric")
        )
    )
    tool_families = ["document_retrieval"]
    if needs_reasoning:
        tool_families.append("analytical_reasoning")
    if needs_compute:
        tool_families.append("exact_compute")
    return TaskIntent(
        task_family="document_qa",
        execution_mode="document_grounded_analysis",
        complexity_tier="structured_analysis",
        tool_families_needed=list(dict.fromkeys(tool_families)),
        evidence_strategy="document_first",
        review_mode="document_grounded",
        completion_mode="document_grounded",
        routing_rationale=(
            "OfficeQA runs must retrieve grounded Treasury evidence before synthesis. "
            "Question types may include extraction, inflation-adjusted comparisons, statistical analysis, "
            "forecasting, weighted averages, and risk metrics, but all must stay document-grounded."
        ),
        confidence=0.97 if overrides.get("benchmark_name") == "officeqa" else 0.95,
        planner_source="heuristic",
    )
