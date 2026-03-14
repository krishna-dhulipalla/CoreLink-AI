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

from agent.contracts import AnswerContract, EvidencePack, TaskProfile

_URL_RE = re.compile(r"https?://[^\s\)\]\"',]+")
_JSON_WRAPPER_RE = re.compile(r"\{\s*\"([A-Za-z0-9_]+)\"\s*:\s*<")
_XML_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_\-]*)>")
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9])")

PROFILE_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "finance_quant": {"calculator", "fetch_reference_file", "list_reference_files"},
    "finance_options": {
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
    },
    "legal_transactional": {"calculator", "fetch_reference_file", "list_reference_files"},
    "document_qa": {"calculator", "fetch_reference_file", "list_reference_files"},
    "external_retrieval": {
        "calculator",
        "internet_search",
        "fetch_reference_file",
        "list_reference_files",
    },
    "general": {"calculator", "fetch_reference_file", "list_reference_files"},
}

PROFILE_CONTEXT: dict[str, str] = {
    "finance_quant": (
        "Focus on exact extraction and exact math. Use prompt-contained formulas and tables first. "
        "Keep units, sign conventions, and requested output formatting precise."
    ),
    "finance_options": (
        "Focus on volatility view, strategy structure, Greeks, breakevens, credit/debit direction, "
        "and risk management. Prefer tool-backed calculations to prose estimates."
    ),
    "legal_transactional": (
        "Focus on transactional structure alternatives, tax consequences, liability allocation, "
        "regulatory and diligence risks, and practical next steps. Avoid unsupported legal citations."
    ),
    "document_qa": (
        "Ground the answer in extracted file or table evidence. Summarize the evidence rather than "
        "repeating raw document text."
    ),
    "external_retrieval": (
        "Use retrieval only to fetch explicitly requested current or source-backed facts. "
        "Cite retrieved evidence directly."
    ),
    "general": "Answer directly using prompt-contained facts first.",
}


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

    if any(token in normalized for token in ("calculate", "formula", "ratio", "numerical", "compute")):
        flags.add("needs_math")
    if "|---" in task_text or ("row" in normalized and "column" in normalized):
        flags.add("needs_tables")
    if _URL_RE.search(task_text) or any(ext in normalized for ext in (".pdf", ".csv", ".xlsx", ".xls", ".docx", ".json")):
        flags.add("needs_files")
    if any(token in normalized for token in ("latest", "current", "today", "recent", "source", "citation", "look up", "search")):
        flags.add("needs_live_data")
    if any(token in normalized for token in ("iv percentile", "implied volatility", "historical volatility", "greeks", "straddle", "strangle", "iron condor", "credit spread", "call option", "put option")):
        flags.add("needs_options_engine")
    if any(token in normalized for token in ("acquisition", "merger", "liability", "indemnification", "transaction", "compliance", "regulatory", "stock consideration", "tax reasons")):
        flags.add("needs_legal_reasoning")
    if answer_contract.requires_adapter:
        flags.add("requires_exact_format")

    return sorted(flags)


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
) -> EvidencePack:
    urls = extract_urls(task_text)
    inline_facts = extract_inline_facts(task_text)
    market_snapshot, derived = derive_market_snapshot(task_text, inline_facts)

    constraints: list[str] = []
    if "requires_exact_format" in capability_flags:
        constraints.append("Must satisfy the exact output contract from the prompt.")
    if "needs_live_data" in capability_flags:
        constraints.append("External retrieval is allowed only if the prompt explicitly requests current data.")

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
        assumptions=[],
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
