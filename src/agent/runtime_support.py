"""
Runtime Support Helpers
=======================
Shared helpers for profiling, intake, context assembly, and structured output.
"""

from __future__ import annotations

from datetime import datetime
import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from agent.contracts import (
    AnswerContract,
    ArtifactCheckpoint,
    AssumptionRecord,
    EvidencePack,
    ExecutionTemplate,
    ProfileDecision,
    ProvenanceRecord,
    TaskProfile,
    ToolResult,
)
from agent.document_evidence import (
    build_document_placeholders,
    document_records_from_tool_result,
    guess_document_format,
    merge_document_evidence_records,
)
from agent.profile_packs import get_profile_pack
from agent.template_library import get_execution_template

_URL_RE = re.compile(r"https?://[^\s\)\]\"',]+")
_JSON_WRAPPER_RE = re.compile(r"\{\s*\"([A-Za-z0-9_]+)\"\s*:\s*<")
_XML_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_\-]*)>")
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9])")
_MONTH_NAME_DATE_RE = re.compile(
    r"\b(?:as of|on|dated?|for)\s+"
    r"((?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(?:as of|on|dated?|for)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_RISK_CAP_RE = re.compile(
    r"(?:(?:max(?:imum)?|keep|cap(?:ped)?|limit(?:ed)?)\s+)?(?:position\s+)?risk(?:\s+(?:to|at|under))?(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

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


def _flatten_provenance(
    prefix: str,
    payload: dict[str, Any],
    *,
    source_class: str,
    source_id: str,
    extraction_method: str,
    tool_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for key, value in (payload or {}).items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            records.update(
                _flatten_provenance(
                    path,
                    value,
                    source_class=source_class,
                    source_id=source_id,
                    extraction_method=extraction_method,
                    tool_name=tool_name,
                )
            )
            continue
        if isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                for idx, item in enumerate(value[:20]):
                    records.update(
                        _flatten_provenance(
                            f"{path}[{idx}]",
                            item,
                            source_class=source_class,
                            source_id=source_id,
                            extraction_method=extraction_method,
                            tool_name=tool_name,
                        )
                    )
            else:
                records[path] = ProvenanceRecord(
                    source_class=source_class,
                    source_id=source_id,
                    extraction_method=extraction_method,
                    tool_name=tool_name,
                ).model_dump()
            continue
        records[path] = ProvenanceRecord(
            source_class=source_class,
            source_id=source_id,
            extraction_method=extraction_method,
            tool_name=tool_name,
        ).model_dump()
    return records


def _has_prompt_fact(prompt_facts: dict[str, Any], *keys: str) -> bool:
    haystack = json.dumps(prompt_facts or {}, ensure_ascii=True).lower()
    return any(key.lower() in haystack for key in keys)


def _merge_unique_assumptions(
    existing: list[dict[str, Any]] | list[AssumptionRecord],
    additions: list[dict[str, Any]] | list[AssumptionRecord],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in [*(existing or []), *(additions or [])]:
        payload = record.model_dump() if isinstance(record, AssumptionRecord) else dict(record)
        signature = (str(payload.get("key", "")), str(payload.get("assumption", "")))
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(payload)
    return merged


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

    if "needs_options_engine" in flags:
        return "finance_options"
    if "needs_legal_reasoning" in flags:
        return "legal_transactional"
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


def extract_as_of_date(text: str) -> str | None:
    match = _ISO_DATE_RE.search(text or "")
    if match:
        return match.group(1)

    match = _MONTH_NAME_DATE_RE.search(text or "")
    if not match:
        return None

    raw = match.group(1).strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


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
    as_of_date = extract_as_of_date(text)
    if as_of_date:
        facts["as_of_date"] = as_of_date

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
    if "as_of_date" in inline_facts:
        snapshot["as_of_date"] = inline_facts["as_of_date"]

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

    lowered = (task_text or "").lower()
    if inline_facts.get("as_of_date"):
        derived["time_sensitive"] = True
    elif any(
        token in lowered
        for token in (
            "latest",
            "today",
            "recent",
            "as of",
            "look up",
            "search",
            "source-backed",
            "current price",
            "current filing",
            "current market",
        )
    ):
        derived["time_sensitive"] = True

    return snapshot, derived


def _initial_assumption_ledger(
    task_profile: str,
    prompt_facts: dict[str, Any],
    derived_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    assumptions: list[dict[str, Any]] = []
    if derived_facts.get("time_sensitive"):
        assumptions.append(
            AssumptionRecord(
                key="time_sensitive_context",
                assumption="Current-data interpretation depends on the retrieval timestamp and should be source-backed.",
                source="prompt_time_sensitivity",
                confidence="medium",
                requires_user_visible_disclosure=False,
                review_status="pending",
            ).model_dump()
        )
    return assumptions


def _extract_policy_context(
    task_text: str,
    task_profile: str,
    capability_flags: list[str],
) -> dict[str, Any]:
    normalized = (task_text or "").lower()
    policy: dict[str, Any] = {}

    action_orientation = any(
        token in normalized
        for token in (
            "should i",
            "should we",
            "recommend",
            "design a strategy",
            "net buyer or seller",
            "buy or sell",
            "allocate",
            "position",
            "trade",
            "execute",
        )
    )
    if action_orientation:
        policy["action_orientation"] = True

    if any(token in normalized for token in ("defined-risk only", "defined risk only", "defined-risk", "defined risk")):
        policy["defined_risk_only"] = True

    if any(token in normalized for token in ("no naked options", "no naked option", "no naked short", "avoid naked options")):
        policy["no_naked_options"] = True

    if any(token in normalized for token in ("retirement account", "ira", "401k", "retail account")):
        policy["retail_or_retirement_account"] = True

    match = _RISK_CAP_RE.search(task_text or "")
    if match:
        try:
            policy["max_position_risk_pct"] = float(match.group(1))
        except ValueError:
            pass

    jurisdictions = []
    if re.search(r"\busa?\b|\bunited states\b", normalized):
        jurisdictions.append("US")
    if re.search(r"\beu\b|\beuropean union\b", normalized):
        jurisdictions.append("EU")
    if re.search(r"\buk\b|\bunited kingdom\b", normalized):
        jurisdictions.append("UK")
    if jurisdictions:
        policy["jurisdictions"] = sorted(set(jurisdictions))

    if action_orientation and (
        "needs_live_data" in set(capability_flags)
        or any(token in normalized for token in ("today", "latest", "as of", "source-backed"))
    ):
        policy["requires_timestamped_evidence"] = True

    if task_profile in {"finance_quant", "finance_options"} and action_orientation:
        policy["requires_recommendation_class"] = True

    return policy


def build_evidence_pack(
    task_text: str,
    answer_contract: AnswerContract,
    task_profile: str,
    capability_flags: list[str],
    ambiguity_flags: list[str] | None = None,
) -> tuple[EvidencePack, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pack = get_profile_pack(task_profile)
    urls = extract_urls(task_text)
    document_placeholders = build_document_placeholders(urls) if "needs_files" in capability_flags else []
    inline_facts = extract_inline_facts(task_text)
    market_snapshot, derived = derive_market_snapshot(task_text, inline_facts)
    policy_context = _extract_policy_context(task_text, task_profile, capability_flags)
    prompt_facts: dict[str, Any] = dict(inline_facts)
    if market_snapshot:
        prompt_facts["market_snapshot"] = market_snapshot
        if task_profile == "finance_options":
            policy_context.pop("requires_timestamped_evidence", None)

    constraints: list[str] = []
    if "requires_exact_format" in capability_flags:
        constraints.append("Must satisfy the exact output contract from the prompt.")
    if "needs_live_data" in capability_flags:
        constraints.append("External retrieval is allowed only if the prompt explicitly requests current data.")
    if ambiguity_flags:
        constraints.append("Task profile is partially ambiguous; avoid unsupported domain assumptions or premature tool use.")
    if document_placeholders:
        constraints.append("For file-backed tasks, gather document metadata or a narrow page/row window first; do not dump raw document bodies.")
    if policy_context.get("defined_risk_only"):
        constraints.append("Recommendations must respect a defined-risk-only mandate.")
    if policy_context.get("no_naked_options"):
        constraints.append("Recommendations must not use naked options.")
    if policy_context.get("max_position_risk_pct") is not None:
        constraints.append(
            f"Position risk must stay within approximately {policy_context['max_position_risk_pct']}% of capital."
        )
    for rule in pack.content_rules[:3]:
        constraints.append(rule)

    open_questions: list[str] = []
    if task_profile == "finance_options" and not _has_prompt_fact(prompt_facts, "spot", "spot_price", '"spot"'):
        open_questions.append("Spot price is not explicit in the prompt; any strategy pricing may require a stated assumption.")
    if policy_context.get("defined_risk_only") and task_profile == "finance_options":
        open_questions.append("A defined-risk alternative may be required even if the first tool-backed strategy is naked short premium.")
    if document_placeholders:
        open_questions.append("Document evidence has not been extracted yet; start with metadata or a targeted fetch before answering.")

    evidence = EvidencePack(
        task_brief=normalize_whitespace(task_text)[:280],
        answer_contract=answer_contract.model_dump(),
        entities=extract_entities(task_text),
        constraints=constraints,
        prompt_facts=prompt_facts,
        retrieved_facts={},
        derived_facts=derived,
        policy_context=policy_context,
        document_evidence=document_placeholders,
        tables=parse_markdown_tables(task_text),
        formulas=extract_formulas(task_text),
        citations=urls[:],
        open_questions=open_questions,
    )
    provenance_map: dict[str, dict[str, Any]] = {}
    provenance_map.update(
        _flatten_provenance(
            "prompt_facts",
            prompt_facts,
            source_class="prompt",
            source_id="user_prompt",
            extraction_method="inline_extraction",
        )
    )
    provenance_map.update(
        _flatten_provenance(
            "derived_facts",
            derived,
            source_class="derived",
            source_id="context_builder",
            extraction_method="derive_market_snapshot",
        )
    )
    provenance_map.update(
        _flatten_provenance(
            "policy_context",
            policy_context,
            source_class="prompt",
            source_id="user_prompt",
            extraction_method="policy_extraction",
        )
    )
    for record in document_placeholders:
        document_id = str(record.get("document_id", "document"))
        metadata = dict(record.get("metadata", {}))
        metadata["citation"] = record.get("citation", "")
        provenance_map.update(
            _flatten_provenance(
                f"document_evidence.{document_id}.metadata",
                metadata,
                source_class="prompt",
                source_id="user_prompt",
                extraction_method="url_discovery",
            )
        )
    assumption_ledger = _initial_assumption_ledger(task_profile, prompt_facts, derived)
    return evidence, assumption_ledger, provenance_map


def _tool_result_source_class(tool_name: str) -> str:
    if tool_name in {"fetch_reference_file", "list_reference_files", "internet_search"}:
        return "retrieved"
    return "derived"


def _tool_result_source_id(tool_result: ToolResult, tool_args: dict[str, Any]) -> str:
    if tool_name := str(tool_result.source.get("tool", tool_result.type)):
        if tool_name == "fetch_reference_file":
            return str(tool_args.get("url") or tool_result.facts.get("file_name") or tool_name)
        if tool_name == "internet_search":
            return str(tool_args.get("query") or tool_name)
    return str(tool_result.type)


def derive_assumption_ledger_entries(
    tool_name: str,
    tool_args: dict[str, Any],
    evidence_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    prompt_facts = dict((evidence_pack or {}).get("prompt_facts", {}))
    records: list[dict[str, Any]] = []
    option_tools = {"analyze_strategy", "black_scholes_price", "option_greeks", "mispricing_analysis", "get_options_chain"}

    if tool_name in option_tools:
        spot_value = None
        if isinstance(tool_args.get("S"), (int, float)):
            spot_value = tool_args.get("S")
        elif tool_name == "analyze_strategy":
            legs = tool_args.get("legs", [])
            if isinstance(legs, list):
                for leg in legs:
                    if isinstance(leg, dict) and isinstance(leg.get("S"), (int, float)):
                        spot_value = leg.get("S")
                        break
        if spot_value is not None and not _has_prompt_fact(prompt_facts, "spot", "spot_price", '"spot"'):
            records.append(
                AssumptionRecord(
                    key="spot_price",
                    assumption=f"Spot price was assumed as {spot_value} from tool arguments because it was not explicit in prompt evidence.",
                    source=f"tool_arguments:{tool_name}",
                    confidence="medium",
                    requires_user_visible_disclosure=True,
                    review_status="pending",
                ).model_dump()
            )

    return records


def merge_tool_result_into_evidence(
    evidence_pack: dict[str, Any],
    tool_result: dict[str, Any] | ToolResult,
    tool_args: dict[str, Any] | None = None,
    provenance_map: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence = EvidencePack.model_validate(evidence_pack or {})
    result = ToolResult.model_validate(tool_result)
    updated = evidence.model_copy(deep=True)
    provenance = dict(provenance_map or {})
    if result.errors or not result.facts:
        return updated.model_dump(), provenance

    if result.type in {"list_reference_files", "fetch_reference_file"}:
        document_records = document_records_from_tool_result(result, tool_args or {})
        if document_records:
            updated.document_evidence = merge_document_evidence_records(updated.document_evidence, document_records)
        if result.type == "list_reference_files":
            updated.retrieved_facts = dict(updated.retrieved_facts)
            updated.retrieved_facts[result.type] = {
                "document_count": len(document_records),
                "documents": [
                    {
                        "document_id": record.get("document_id", ""),
                        "citation": record.get("citation", ""),
                        "format": (record.get("metadata", {}) or {}).get("format", guess_document_format(record.get("citation", ""))),
                    }
                    for record in document_records
                ],
            }
        else:
            updated.retrieved_facts = dict(updated.retrieved_facts)
            updated.retrieved_facts[result.type] = {
                "document_count": len(document_records),
                "documents": [
                    {
                        "document_id": record.get("document_id", ""),
                        "citation": record.get("citation", ""),
                        "status": record.get("status", ""),
                        "table_count": len(record.get("tables", []) or []),
                        "chunk_count": len(record.get("chunks", []) or []),
                        "numeric_summary_count": len(record.get("numeric_summaries", []) or []),
                    }
                    for record in document_records
                ],
            }
        updated.citations = list(
            dict.fromkeys(
                [
                    *updated.citations,
                    *[
                        str(record.get("citation", ""))
                        for record in document_records
                        if str(record.get("citation", "")).strip()
                    ],
                ]
            )
        )
        for record in document_records:
            document_id = str(record.get("document_id", "document"))
            metadata = dict(record.get("metadata", {}))
            metadata["citation"] = record.get("citation", "")
            provenance.update(
                _flatten_provenance(
                    f"document_evidence.{document_id}.metadata",
                    metadata,
                    source_class="retrieved",
                    source_id=str(record.get("citation", "") or document_id),
                    extraction_method="document_evidence_merge",
                    tool_name=str(result.source.get("tool", result.type)),
                )
            )
            for section in ("chunks", "tables", "numeric_summaries"):
                items = record.get(section, [])
                if isinstance(items, list) and items:
                    provenance[f"document_evidence.{document_id}.{section}"] = ProvenanceRecord(
                        source_class="retrieved",
                        source_id=str(record.get("citation", "") or document_id),
                        extraction_method="document_evidence_merge",
                        tool_name=str(result.source.get("tool", result.type)),
                    ).model_dump()
        return updated.model_dump(), provenance

    source_class = _tool_result_source_class(result.type)
    fact_bucket = dict(updated.retrieved_facts if source_class == "retrieved" else updated.derived_facts)
    fact_bucket[result.type] = result.facts
    if source_class == "retrieved":
        updated.retrieved_facts = fact_bucket
    else:
        updated.derived_facts = fact_bucket

    if result.type == "internet_search":
        urls = [entry.get("url") for entry in result.facts.get("results", []) if isinstance(entry, dict) and entry.get("url")]
        updated.citations = list(dict.fromkeys([*updated.citations, *urls]))
    elif result.type == "fetch_reference_file":
        source_id = str((tool_args or {}).get("url") or result.facts.get("file_name") or "")
        if source_id:
            updated.citations = list(dict.fromkeys([*updated.citations, source_id]))

    prefix = f"{'retrieved_facts' if source_class == 'retrieved' else 'derived_facts'}.{result.type}"
    provenance.update(
        _flatten_provenance(
            prefix,
            result.facts,
            source_class=source_class,
            source_id=_tool_result_source_id(result, tool_args or {}),
            extraction_method="tool_normalization",
            tool_name=str(result.source.get("tool", result.type)),
        )
    )
    return updated.model_dump(), provenance


def initial_solver_stage(task_profile: str, capability_flags: list[str], evidence_pack: dict[str, Any]) -> str:
    flags = set(capability_flags)
    if evidence_pack.get("citations") and "needs_files" in flags:
        return "GATHER"
    if task_profile in {"document_qa", "external_retrieval"} and evidence_pack.get("citations"):
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
        if "needs_live_data" in flags or "needs_files" in flags:
            return "GATHER"
        if evidence_pack.get("citations") and ("needs_files" in flags or task_profile in {"document_qa", "external_retrieval"}):
            return "GATHER"
        return "COMPUTE"
    if default_stage in allowed_stages or not allowed_stages:
        return default_stage
    return initial_solver_stage(task_profile, capability_flags, evidence_pack)


def selective_checkpoint_policy(template: dict[str, Any] | ExecutionTemplate | None) -> dict[str, Any]:
    if isinstance(template, dict):
        template_id = str(template.get("template_id", ""))
    elif isinstance(template, ExecutionTemplate):
        template_id = template.template_id
    else:
        template_id = ""

    if template_id == "quant_with_tool_compute":
        return {
            "enabled": True,
            "checkpoint_stages": {"GATHER", "COMPUTE"},
            "backtrack_stages": {"GATHER", "COMPUTE"},
        }
    if template_id == "options_tool_backed":
        return {
            "enabled": True,
            "checkpoint_stages": {"COMPUTE"},
            "backtrack_stages": {"COMPUTE"},
        }
    if template_id in {"document_qa", "legal_with_document_evidence"}:
        return {
            "enabled": True,
            "checkpoint_stages": {"GATHER"},
            "backtrack_stages": {"GATHER"},
        }
    return {
        "enabled": False,
        "checkpoint_stages": set(),
        "backtrack_stages": set(),
    }


def selective_backtracking_allowed(template: dict[str, Any] | ExecutionTemplate | None, stage: str) -> bool:
    policy = selective_checkpoint_policy(template)
    return bool(policy["enabled"]) and stage in set(policy["backtrack_stages"])


def should_checkpoint_stage(template: dict[str, Any] | ExecutionTemplate | None, stage: str) -> bool:
    policy = selective_checkpoint_policy(template)
    return bool(policy["enabled"]) and stage in set(policy["checkpoint_stages"])


def artifact_checkpoint_from_state(
    state: dict[str, Any],
    *,
    reason: str,
    stage: str,
) -> dict[str, Any]:
    workpad = dict(state.get("workpad", {}))
    template = state.get("execution_template", {}) or {}
    checkpoint = ArtifactCheckpoint(
        template_id=str(template.get("template_id", "")),
        checkpoint_stage=stage,
        reason=reason,
        evidence_pack=dict(state.get("evidence_pack", {})),
        assumption_ledger=list(state.get("assumption_ledger", [])),
        provenance_map=dict(state.get("provenance_map", {})),
        last_tool_result=state.get("last_tool_result"),
        draft_answer=str(workpad.get("draft_answer", "")),
        stage_outputs=dict(workpad.get("stage_outputs", {})),
        review_feedback=state.get("review_feedback"),
    )
    return checkpoint.model_dump()


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
