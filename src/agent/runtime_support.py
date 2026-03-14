"""
Runtime Support Helpers
=======================
Shared helpers for profiling, intake, context assembly, and structured output.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from agent.contracts import AnswerContract, EvidencePack, ExecutionTemplate, ProfileDecision, TaskProfile
from agent.profile_packs import get_profile_pack
from agent.template_library import get_execution_template

_URL_RE = re.compile(r"https?://[^\s\)\]\"',]+")
_JSON_WRAPPER_RE = re.compile(r"\{\s*\"([A-Za-z0-9_]+)\"\s*:\s*<")
_XML_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_\-]*)>")
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9])")

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


def extract_answer_contract(task_text: str) -> AnswerContract:
    text = task_text or ""
    lowered = text.lower()

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
    if _URL_RE.search(task_text) or any(ext in normalized for ext in (".pdf", ".csv", ".xlsx", ".xls", ".docx", ".json")):
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

    if "needs_options_engine" in flags:
        return "finance_options"
    if "needs_legal_reasoning" in flags:
        return "legal_transactional"
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
        if has_files or has_live:
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
        elif "needs_options_engine" in flags and "legal_options_overlap" not in ambiguity:
            template_id = "options_tool_backed"
        elif "needs_legal_reasoning" in flags:
            template_id = "legal_reasoning_only"
        elif "needs_math" in flags:
            template_id = "quant_inline_exact" if (exact_contract or has_inline_quant_evidence) else "quant_with_tool_compute"

    if ambiguity and template_id == "options_tool_backed" and "needs_legal_reasoning" in flags:
        template_id = "legal_reasoning_only"

    if ambiguity and template_id == "quant_with_tool_compute" and "needs_legal_reasoning" in flags and not has_files:
        template_id = "legal_reasoning_only"

    return get_execution_template(template_id)


def extract_urls(text: str) -> list[str]:
    urls = []
    for match in _URL_RE.findall(text or ""):
        clean = match.rstrip(".,;)")
        if clean not in urls:
            urls.append(clean)
    return urls


def extract_formulas(text: str) -> list[str]:
    formulas: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" in stripped and any(ch.isalpha() for ch in stripped):
            formulas.append(stripped[:300])
        elif stripped.startswith("\\text{") or stripped.startswith("$"):
            formulas.append(stripped[:300])
    deduped: list[str] = []
    for formula in formulas:
        if formula not in deduped:
            deduped.append(formula)
    return deduped[:20]


def parse_markdown_tables(text: str) -> list[dict[str, Any]]:
    lines = (text or "").splitlines()
    tables: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines) - 1:
        header = lines[idx].strip()
        separator = lines[idx + 1].strip()
        if "|" not in header or "|" not in separator or "---" not in separator:
            idx += 1
            continue
        headers = [col.strip() for col in header.strip("|").split("|")]
        rows: list[dict[str, str]] = []
        j = idx + 2
        while j < len(lines):
            row_line = lines[j].strip()
            if "|" not in row_line:
                break
            values = [col.strip() for col in row_line.strip("|").split("|")]
            if len(values) != len(headers):
                break
            rows.append(dict(zip(headers, values)))
            j += 1
        if rows:
            tables.append({"headers": headers, "rows": rows[:20]})
        idx = j
    return tables[:10]


def extract_entities(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.findall(r"\b[A-Z]{2,6}\b", text or ""):
        if match not in candidates:
            candidates.append(match)
    for match in re.findall(r"\b\d{4}\.HK\b", text or ""):
        if match not in candidates:
            candidates.append(match)
    return candidates[:10]


def extract_inline_facts(text: str) -> dict[str, Any]:
    lowered = (text or "").lower()
    facts: dict[str, Any] = {}

    if "iv percentile" in lowered:
        match = re.search(r"iv percentile[^0-9]*(\d+(?:\.\d+)?)", lowered)
        if match:
            facts["iv_percentile"] = float(match.group(1))

    percentages = _PERCENT_RE.findall(text or "")
    if percentages:
        facts["percentages"] = [float(value) / 100.0 for value in percentages[:12]]

    numbers = _NUMBER_RE.findall(text or "")
    if numbers:
        facts["numbers"] = [float(value) for value in numbers[:20]]

    iv_match = re.search(r"\biv\b[^0-9]*(\d+(?:\.\d+)?)\s*%", lowered)
    hv_match = re.search(r"historical volatility[^0-9]*(\d+(?:\.\d+)?)\s*%", lowered)
    if iv_match:
        facts["implied_volatility"] = float(iv_match.group(1)) / 100.0
    if hv_match:
        facts["historical_volatility"] = float(hv_match.group(1)) / 100.0

    return facts


def derive_market_snapshot(task_text: str, inline_facts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot: dict[str, Any] = {}
    derived: dict[str, Any] = {}

    if "implied_volatility" in inline_facts:
        snapshot["implied_volatility"] = inline_facts["implied_volatility"]
    if "historical_volatility" in inline_facts:
        snapshot["historical_volatility"] = inline_facts["historical_volatility"]
    if "iv_percentile" in inline_facts:
        snapshot["iv_percentile"] = inline_facts["iv_percentile"]

    if "implied_volatility" in snapshot and "historical_volatility" in snapshot:
        derived["iv_premium"] = round(
            float(snapshot["implied_volatility"]) - float(snapshot["historical_volatility"]),
            4,
        )
        derived["vol_bias"] = (
            "short_vol"
            if derived["iv_premium"] > 0 and float(snapshot.get("iv_percentile", 0)) >= 50
            else "neutral"
        )

    if "latest" in (task_text or "").lower() or "current" in (task_text or "").lower():
        derived["time_sensitive"] = True

    return snapshot, derived


def build_evidence_pack(
    task_text: str,
    answer_contract: AnswerContract,
    task_profile: str,
    capability_flags: list[str],
    ambiguity_flags: list[str] | None = None,
) -> EvidencePack:
    pack = get_profile_pack(task_profile)
    urls = extract_urls(task_text)
    inline_facts = extract_inline_facts(task_text)
    market_snapshot, derived = derive_market_snapshot(task_text, inline_facts)

    constraints: list[str] = []
    if "requires_exact_format" in capability_flags:
        constraints.append("Must satisfy the exact output contract from the prompt.")
    if "needs_live_data" in capability_flags:
        constraints.append("External retrieval is allowed only if the prompt explicitly requests current data.")
    if ambiguity_flags:
        constraints.append("Task profile is partially ambiguous; avoid unsupported domain assumptions or premature tool use.")
    for rule in pack.content_rules[:3]:
        constraints.append(rule)

    open_questions: list[str] = []
    if task_profile == "finance_options" and "spot" not in json.dumps(inline_facts).lower():
        open_questions.append("Spot price is not explicit in the prompt; any strategy pricing may require a stated assumption.")

    return EvidencePack(
        task_brief=normalize_whitespace(task_text)[:280],
        answer_contract=answer_contract.model_dump(),
        entities=extract_entities(task_text),
        constraints=constraints,
        inline_facts=inline_facts,
        tables=parse_markdown_tables(task_text),
        formulas=extract_formulas(task_text),
        file_refs=urls,
        market_snapshot=market_snapshot,
        derived_signals=derived,
        citations=urls[:],
        assumptions=list(pack.failure_modes[:2]),
        open_questions=open_questions,
    )


def initial_solver_stage(task_profile: str, capability_flags: list[str], evidence_pack: dict[str, Any]) -> str:
    flags = set(capability_flags)
    if evidence_pack.get("file_refs") and "needs_files" in flags:
        return "GATHER"
    if task_profile in {"document_qa", "external_retrieval"} and evidence_pack.get("file_refs"):
        return "GATHER"
    if task_profile == "external_retrieval":
        return "GATHER"
    if task_profile in {"finance_quant", "finance_options"} or "needs_math" in flags or "needs_options_engine" in flags:
        return "COMPUTE"
    return "SYNTHESIZE"


def initial_stage_for_template(
    template: dict[str, Any] | ExecutionTemplate | None,
    task_profile: str,
    capability_flags: list[str],
    evidence_pack: dict[str, Any],
) -> str:
    if isinstance(template, dict):
        default_stage = str(template.get("default_initial_stage", "SYNTHESIZE"))
        allowed_stages = set(template.get("allowed_stages", []))
        template_id = str(template.get("template_id", ""))
    elif isinstance(template, ExecutionTemplate):
        default_stage = template.default_initial_stage
        allowed_stages = set(template.allowed_stages)
        template_id = template.template_id
    else:
        default_stage = "SYNTHESIZE"
        allowed_stages = set()
        template_id = ""

    flags = set(capability_flags)
    if template_id in {"legal_with_document_evidence", "document_qa", "live_retrieval"}:
        return "GATHER"
    if template_id in {"options_tool_backed", "quant_inline_exact"}:
        return "COMPUTE"
    if template_id == "quant_with_tool_compute":
        if evidence_pack.get("file_refs") and ("needs_files" in flags or task_profile in {"document_qa", "external_retrieval"}):
            return "GATHER"
        return "COMPUTE"
    if default_stage in allowed_stages or not allowed_stages:
        return default_stage
    return initial_solver_stage(task_profile, capability_flags, evidence_pack)


def stage_is_review_milestone(template: dict[str, Any] | ExecutionTemplate | None, stage: str) -> bool:
    if isinstance(template, dict):
        review_stages = set(template.get("review_stages", []))
    elif isinstance(template, ExecutionTemplate):
        review_stages = set(template.review_stages)
    else:
        review_stages = set()
    return stage in review_stages


def next_stage_after_review(stage: str, review_target: str, verdict: str) -> str:
    if verdict in {"revise", "backtrack"}:
        return "REVISE"
    if stage == "GATHER":
        return "COMPUTE" if review_target != "synthesize" else "SYNTHESIZE"
    if stage == "COMPUTE":
        return "SYNTHESIZE"
    if stage == "SYNTHESIZE":
        return "COMPLETE"
    return "COMPLETE"
