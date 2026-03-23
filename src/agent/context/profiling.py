"""
Profiling and answer-contract helpers.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from agent.context.extraction import extract_urls
from agent.contracts import AnswerContract, ExecutionTemplate, ProfileDecision, TaskProfile
from agent.profile_packs import get_profile_pack
from agent.template_library import get_execution_template

_XML_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_\-]*)>")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_officeqa_prompt(text: str) -> bool:
    lowered = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return any(
        token in lowered
        for token in (
            "treasury bulletin",
            "u.s national defense",
            "u s national defense",
            "veterans administration",
            "individual calendar months",
            "bls cpi-u",
            "bls cpi u",
            "federal reserve bank of minneapolis",
        )
    )


def _officeqa_xml_contract_enabled(task_text: str = "") -> bool:
    officeqa_env = _truthy_env("OFFICEQA_FINAL_ANSWER_TAGS") or _truthy_env("OFFICEQA_XML_OUTPUT") or os.getenv("BENCHMARK_NAME", "").strip().lower() == "officeqa"
    if officeqa_env and _looks_like_officeqa_prompt(task_text):
        return True
    if _truthy_env("BENCHMARK_STATELESS") and os.getenv("OFFICEQA_CORPUS_DIR", "").strip() and _looks_like_officeqa_prompt(task_text):
        return True
    return False


def infer_benchmark_overrides(task_text: str) -> dict[str, Any]:
    benchmark_name = os.getenv("BENCHMARK_NAME", "").strip().lower()
    officeqa_like = _looks_like_officeqa_prompt(task_text)
    officeqa_mode = benchmark_name == "officeqa" and officeqa_like
    allow_web_fallback = os.getenv("OFFICEQA_ALLOW_WEB_FALLBACK", "last_fallback").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "never",
    }
    return {
        "benchmark_name": benchmark_name,
        "officeqa_mode": officeqa_mode,
        "officeqa_like_prompt": officeqa_like,
        "officeqa_xml_contract": _officeqa_xml_contract_enabled(task_text),
        "officeqa_allow_web_fallback": allow_web_fallback,
    }


def _extract_labeled_json_block(text: str, label: str) -> Any | None:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*", re.IGNORECASE)
    match = pattern.search(text or "")
    if not match:
        return None
    tail = (text or "")[match.end():].lstrip()
    if not tail or tail[0] not in "[{":
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(tail)
        return parsed
    except Exception:
        return None


def allowed_tools_for_profile(profile: str) -> set[str]:
    return set(get_profile_pack(profile).allowed_tools)


def allowed_tools_for_template(template: dict[str, Any] | ExecutionTemplate | None, profile: str) -> set[str]:
    if isinstance(template, dict):
        allowed = set(template.get("allowed_tool_names", []))
    elif isinstance(template, ExecutionTemplate):
        allowed = set(template.allowed_tool_names)
    else:
        allowed = set()
    if not allowed:
        return allowed_tools_for_profile(profile)
    return allowed & allowed_tools_for_profile(profile)


def apply_profile_contract_rules(answer_contract: AnswerContract, task_profile: str) -> AnswerContract:
    pack = get_profile_pack(task_profile)
    contract = answer_contract.model_copy(deep=True)

    if pack.content_rules:
        merged_rules = list(dict.fromkeys([*contract.content_rules, *pack.content_rules]))
        contract.content_rules = merged_rules

    has_strict_wrapper = contract.requires_adapter and contract.format in {"json", "xml"}
    if not has_strict_wrapper and pack.section_requirements:
        contract.section_requirements = list(
            dict.fromkeys([*contract.section_requirements, *pack.section_requirements])
        )

    if pack.required_evidence_types:
        required = list(
            dict.fromkeys(
                [*(contract.schema_hint.get("required_evidence_types", []) or []), *pack.required_evidence_types]
            )
        )
        contract.schema_hint["required_evidence_types"] = required

    return contract


def latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return str(msg.content)
    return ""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_answer_contract(task_text: str, benchmark_overrides: dict[str, Any] | None = None) -> AnswerContract:
    text = task_text or ""
    lowered = text.lower()
    overrides = dict(benchmark_overrides or {})

    if overrides.get("officeqa_xml_contract") or _officeqa_xml_contract_enabled(text):
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

    if "output format" in lowered or "json format" in lowered or '{"answer"' in text:
        wrapper = None
        wrapper_match = re.search(r'\{\s*"([A-Za-z0-9_]+)"\s*:', text)
        if wrapper_match:
            wrapper = wrapper_match.group(1)
        example_match = re.search(r"(\{.*?\})", text, flags=re.DOTALL)
        example = example_match.group(1).strip() if example_match else None
        return AnswerContract(
            format="json",
            requires_adapter=True,
            raw_instruction="JSON output required by the task prompt.",
            wrapper_key=wrapper,
            exact_output_example=example,
        )

    if "xml" in lowered or "</" in text:
        tag_match = _XML_TAG_RE.search(text)
        return AnswerContract(
            format="xml",
            requires_adapter=True,
            raw_instruction="XML output required by the task prompt.",
            xml_root_tag=tag_match.group(1) if tag_match else None,
        )

    return AnswerContract()


def detect_capability_flags(task_text: str, answer_contract: AnswerContract) -> list[str]:
    normalized = (task_text or "").lower()
    flags: set[str] = set()

    if any(
        re.search(pattern, normalized)
        for pattern in (
            r"\bcalculate\b",
            r"\bformula\b",
            r"\bratio\b",
            r"\bnumerical\b",
            r"\bcompute\b",
        )
    ):
        flags.add("needs_math")
    if "|---" in task_text or ("row" in normalized and "column" in normalized):
        flags.add("needs_tables")
    if extract_urls(task_text) or any(ext in normalized for ext in (".pdf", ".csv", ".xlsx", ".xls", ".docx", ".json")):
        flags.add("needs_files")
    if any(
        token in normalized
        for token in ("latest", "today", "recent", "look up", "search", "source-backed", "source citation", "cite sources", "as of")
    ):
        flags.add("needs_live_data")
    if any(token in normalized for token in ("iv percentile", "implied volatility", "historical volatility", "greeks", "straddle", "strangle", "iron condor", "credit spread", "call option", "put option")):
        flags.add("needs_options_engine")
    if any(
        token in normalized
        for token in (
            "acquisition",
            "merger",
            "transaction structure",
            "deal structure",
            "compliance",
            "regulatory",
            "stock consideration",
            "tax reasons",
            "indemnification",
            "indemnity",
            "compliance liabilities",
            "liability protection",
            "liability isolation",
            "escrow",
            "warranties",
            "reverse triangular",
            "asset purchase",
        )
    ):
        flags.add("needs_legal_reasoning")
    if any(
        token in normalized
        for token in (
            "equity research",
            "research note",
            "investment thesis",
            "target price",
            "bull case",
            "bear case",
            "catalyst",
            "research report",
        )
    ):
        flags.add("needs_equity_research")
    if any(
        token in normalized
        for token in (
            "portfolio risk",
            "risk review",
            "concentration",
            "risk budget",
            "drawdown",
            "exposure review",
            "factor exposure",
            "var limit",
            "portfolio exposures",
            "position weights",
        )
    ):
        flags.add("needs_portfolio_risk")
    if any(
        token in normalized
        for token in (
            "event-driven",
            "earnings",
            "guidance",
            "macro event",
            "cpi release",
            "fed meeting",
            "corporate action",
            "catalyst trade",
            "merger arb",
        )
    ):
        flags.add("needs_event_analysis")
    if any(
        token in normalized
        for token in (
            "derivative of",
            "differentiate",
            "marginal cost",
            "marginal revenue",
            "integral",
            "optimize",
            "maximise",
            "maximize",
            "minimise",
            "minimize",
            "prove that",
        )
    ):
        flags.add("needs_analytical_reasoning")
    if any(
        token in normalized
        for token in (
            "flash crash",
            "liquidity crisis",
            "stress scenario",
            "scenario validation",
            "crypto",
            "drawdown scenario",
        )
    ):
        flags.add("needs_market_scenario")
    if any(
        token in normalized
        for token in (
            ".wav",
            ".mp3",
            "audio file",
            "music producer",
            "render audio",
            "generate audio",
            "create a track",
            "zip file",
            "video file",
            "image file",
        )
    ):
        flags.add("needs_artifact_generation")
    if answer_contract.requires_adapter:
        flags.add("requires_exact_format")

    return sorted(flags)


def detect_ambiguity_flags(task_text: str, capability_flags: list[str]) -> list[str]:
    normalized = (task_text or "").lower()
    flags = set(capability_flags)
    ambiguity: set[str] = set()

    if "needs_legal_reasoning" in flags and (
        "needs_math" in flags
        or any(token in normalized for token in ("roe", "roa", "yield", "valuation", "financial"))
    ):
        ambiguity.add("legal_finance_overlap")

    if "needs_legal_reasoning" in flags and "needs_options_engine" in flags:
        ambiguity.add("legal_options_overlap")

    if "needs_files" in flags and "needs_math" in flags:
        ambiguity.add("document_math_overlap")

    if "needs_files" in flags and "needs_live_data" in flags:
        ambiguity.add("document_live_overlap")

    domain_markers = 0
    if "needs_legal_reasoning" in flags:
        domain_markers += 1
    if "needs_options_engine" in flags or "needs_math" in flags:
        domain_markers += 1
    if "needs_files" in flags or "needs_tables" in flags:
        domain_markers += 1
    if "needs_live_data" in flags:
        domain_markers += 1
    if domain_markers >= 3:
        ambiguity.add("broad_multi_capability")

    return sorted(ambiguity)


def infer_task_profile(task_text: str, capability_flags: list[str]) -> TaskProfile:
    normalized = (task_text or "").lower()
    flags = set(capability_flags)
    finance_data_markers = (
        "price history",
        "historical prices",
        "1-month return",
        "monthly return",
        "return over",
        "fundamentals",
        "yield curve",
        "income statement",
        "balance sheet",
        "cash flow",
        "financial statements",
        "statement line item",
        "corporate actions",
    )

    if "needs_artifact_generation" in flags:
        return "unsupported_artifact"
    if "needs_market_scenario" in flags:
        return "market_scenario"
    if "needs_options_engine" in flags:
        return "finance_options"
    if "needs_legal_reasoning" in flags:
        return "legal_transactional"
    if "needs_analytical_reasoning" in flags and "needs_live_data" not in flags:
        return "analytical_reasoning"
    if {"needs_equity_research", "needs_portfolio_risk", "needs_event_analysis"} & flags:
        return "finance_quant"
    if any(marker in normalized for marker in finance_data_markers):
        return "finance_quant"
    if "needs_math" in flags and any(
        token in normalized
        for token in ("annual report", "roe", "roa", "financial leverage", "inventory turnover", "equity multiplier", "valuation", "yield", "p&l")
    ):
        return "finance_quant"
    if "needs_live_data" in flags and not {"needs_legal_reasoning", "needs_options_engine"} & flags:
        return "external_retrieval"
    if "needs_files" in flags or "needs_tables" in flags:
        return "document_qa"
    if "needs_math" in flags:
        return "finance_quant"
    return "general"


def build_profile_decision(task_text: str, answer_contract: AnswerContract) -> ProfileDecision:
    capability_flags = detect_capability_flags(task_text, answer_contract)
    primary_profile = infer_task_profile(task_text, capability_flags)
    ambiguity_flags = detect_ambiguity_flags(task_text, capability_flags)

    if "legal_options_overlap" in ambiguity_flags:
        primary_profile = "general"

    return ProfileDecision(
        primary_profile=primary_profile,
        capability_flags=capability_flags,
        ambiguity_flags=ambiguity_flags,
        needs_external_data="needs_live_data" in capability_flags,
        needs_output_adapter=answer_contract.requires_adapter,
    )


def select_execution_template(
    task_text: str,
    profile_decision: ProfileDecision,
    answer_contract: AnswerContract,
) -> ExecutionTemplate:
    flags = set(profile_decision.capability_flags)
    ambiguity = set(profile_decision.ambiguity_flags)
    profile = profile_decision.primary_profile
    has_files = bool(extract_urls(task_text)) or "needs_files" in flags
    has_live = "needs_live_data" in flags
    exact_contract = answer_contract.requires_adapter and answer_contract.format in {"json", "xml"}
    lowered = (task_text or "").lower()
    has_inline_quant_evidence = any(token in lowered for token in ("=", "|---", "roe", "roa", "ebitda", "yield"))
    action_orientation = any(
        token in lowered
        for token in ("should i", "should we", "recommend", "buy or sell", "allocate", "overweight", "underweight", "action")
    )

    template_id = "legal_reasoning_only"

    if profile == "finance_options":
        template_id = "options_tool_backed"
    elif profile == "legal_transactional":
        template_id = "legal_with_document_evidence" if has_files else "legal_reasoning_only"
    elif profile == "document_qa":
        template_id = "document_qa"
    elif profile == "external_retrieval":
        template_id = "live_retrieval"
    elif profile == "finance_quant":
        if "needs_portfolio_risk" in flags:
            template_id = "portfolio_risk_review"
        elif "needs_event_analysis" in flags:
            template_id = "event_driven_finance"
        elif "needs_equity_research" in flags:
            template_id = "equity_research_report"
        elif action_orientation and (has_live or "needs_live_data" in flags):
            template_id = "regulated_actionable_finance"
        elif has_files or has_live:
            template_id = "quant_with_tool_compute"
        elif exact_contract or has_inline_quant_evidence:
            template_id = "quant_inline_exact"
        else:
            template_id = "quant_with_tool_compute"
    else:
        if has_live:
            template_id = "live_retrieval"
        elif has_files and "needs_legal_reasoning" in flags:
            template_id = "legal_with_document_evidence"
        elif has_files:
            template_id = "document_qa"
        elif "needs_portfolio_risk" in flags:
            template_id = "portfolio_risk_review"
        elif "needs_event_analysis" in flags:
            template_id = "event_driven_finance"
        elif "needs_equity_research" in flags:
            template_id = "equity_research_report"
        elif "needs_options_engine" in flags and "legal_options_overlap" not in ambiguity:
            template_id = "options_tool_backed"
        elif action_orientation and ("needs_live_data" in flags or "needs_math" in flags):
            template_id = "regulated_actionable_finance"
        elif "needs_legal_reasoning" in flags:
            template_id = "legal_reasoning_only"
        elif "needs_math" in flags:
            template_id = "quant_inline_exact" if (exact_contract or has_inline_quant_evidence) else "quant_with_tool_compute"

    if ambiguity and template_id == "options_tool_backed" and "needs_legal_reasoning" in flags:
        template_id = "legal_reasoning_only"

    if ambiguity and template_id == "quant_with_tool_compute" and "needs_legal_reasoning" in flags and not has_files:
        template_id = "legal_reasoning_only"

    return get_execution_template(template_id)
