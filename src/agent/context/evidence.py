"""
Evidence-pack assembly, assumptions, and provenance helpers.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.context.extraction import (
    derive_market_snapshot,
    extract_entities,
    extract_formulas,
    extract_inline_facts,
    extract_urls,
    parse_markdown_tables,
)
from agent.contracts import (
    AnswerContract,
    ArtifactCheckpoint,
    AssumptionRecord,
    EvidencePack,
    ProvenanceRecord,
    ToolResult,
)
from agent.document_evidence import (
    build_document_placeholders,
    document_records_from_tool_result,
    guess_document_format,
    merge_document_evidence_records,
)
from agent.profile_packs import get_profile_pack

_RISK_CAP_RE = re.compile(
    r"(?:(?:max(?:imum)?|keep|cap(?:ped)?|limit(?:ed)?)\s+)?(?:position\s+)?risk(?:\s+(?:to|at|under))?(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_USER_QUESTION_RE = re.compile(
    r"(?im)^\s*(?:###\s*)?(?:<User Question>|User Question)\s*:?\s*(.*?)\s*(?=^\s*###|\Z)",
    re.DOTALL,
)
_FOCUS_VERB_RE = re.compile(r"\b(calculate|compute|derive|determine|evaluate|find|what is|what's|for)\b", re.IGNORECASE)
_QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "then",
    "than",
    "over",
    "under",
    "using",
    "show",
    "need",
    "needs",
    "should",
    "would",
    "could",
    "about",
    "compare",
    "company",
    "companies",
    "calculate",
    "compute",
    "derive",
    "determine",
    "output",
    "format",
    "answer",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _extract_focus_query(task_text: str) -> str:
    match = _USER_QUESTION_RE.search(task_text or "")
    if match:
        return _normalize_text(match.group(1))
    candidate_lines: list[str] = []
    for raw_line in (task_text or "").splitlines():
        line = _normalize_text(raw_line)
        if not line:
            continue
        lowered = line.lower()
        if line.startswith("|"):
            continue
        if lowered.startswith("output format") or lowered.startswith("reference file") or lowered.startswith("reference:"):
            continue
        if lowered.startswith("###") and "question" not in lowered:
            continue
        if "=" in line and not line.endswith("?"):
            continue
        candidate_lines.append(line)
    for line in reversed(candidate_lines):
        if "?" in line or _FOCUS_VERB_RE.search(line):
            return line
    return ""


def _extract_target_period(task_text: str) -> str:
    years = _YEAR_RE.findall(task_text or "")
    if years:
        return str(years[-1])
    return ""


def _coerce_row_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip().replace(",", "")
    percent_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*%", stripped)
    if percent_match:
        return float(percent_match.group(1)) / 100.0
    number_match = re.fullmatch(r"-?\d+(?:\.\d+)?", stripped)
    if number_match:
        return float(stripped)
    return value


def _query_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return {token for token in tokens if len(token) >= 3 and token not in _QUERY_STOPWORDS}


def _row_label(row: dict[str, Any]) -> str:
    for value in row.values():
        if isinstance(value, str):
            text = _normalize_text(value)
            if text and not re.fullmatch(r"-?\d+(?:\.\d+)?%?", text.replace(",", "")):
                return text.lower()
    return json.dumps(row, ensure_ascii=True, sort_keys=True).lower()


def _row_match_score(row: dict[str, Any], query: str) -> int:
    query_tokens = _query_tokens(query)
    if not query_tokens:
        return 0
    best_score = 0
    for value in row.values():
        text = _normalize_text(str(value or "").lower())
        if not text:
            continue
        value_tokens = _query_tokens(text)
        if not value_tokens:
            continue
        overlap = len(query_tokens & value_tokens)
        if overlap > best_score:
            best_score = overlap
        if len(value_tokens) >= 2 and value_tokens.issubset(query_tokens):
            best_score = max(best_score, len(value_tokens) + 2)
    return best_score


def _select_relevant_formulas(formulas: list[str], focus_query: str) -> list[str]:
    if not formulas:
        return []
    normalized_query = _normalize_text(focus_query).lower()
    query_tokens = _query_tokens(focus_query)
    scored: list[tuple[int, str]] = []
    for formula in formulas:
        normalized_formula = _normalize_text(formula).lower()
        score = 0
        if normalized_query and normalized_query in normalized_formula:
            score += 8
        formula_tokens = _query_tokens(formula)
        score += min(len(query_tokens & formula_tokens), 5)
        formula_name = normalized_formula.split("=", 1)[0].strip()
        formula_name_tokens = _query_tokens(formula_name)
        if formula_name_tokens and len(formula_name_tokens & query_tokens) >= min(2, len(formula_name_tokens)):
            score += 4
        scored.append((score, formula))
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    selected = [formula for score, formula in ranked if score > 0][:4]
    if selected:
        return selected
    return formulas[:2]


def _select_relevant_table_rows(
    tables: list[dict[str, Any]],
    *,
    focus_query: str,
    target_entities: list[str],
    target_period: str,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    unused = 0
    normalized_entities = [str(item).strip().lower() for item in target_entities if str(item).strip()]
    for table in tables:
        headers = list(table.get("headers", []))
        rows = list(table.get("rows", []))
        scored_rows: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_text = " ".join(str(value or "").lower() for value in row.values())
            entity_match = any(entity and entity in row_text for entity in normalized_entities)
            score = _row_match_score(row, focus_query)
            if entity_match:
                score += 3
            if score > 0:
                scored_rows.append((score, row))
        if not scored_rows:
            if not focus_query and len(rows) == 1 and isinstance(rows[0], dict):
                scored_rows.append((1, rows[0]))
            else:
                unused += 1
                continue
        top_score = max(score for score, _ in scored_rows)
        if focus_query and top_score < 2:
            unused += 1
            continue
        matched_rows = [row for score, row in scored_rows if score == top_score]
        unique_labels = {_row_label(row) for row in matched_rows}
        if len(unique_labels) > 1:
            unused += 1
            continue
        normalized_rows: list[dict[str, Any]] = []
        for row in matched_rows[:1]:
            normalized_row: dict[str, Any] = {}
            for key, value in row.items():
                key_text = str(key)
                if target_period and target_period not in key_text and any(year in key_text for year in re.findall(r"\b(?:19|20)\d{2}\b", key_text)):
                    continue
                normalized_row[key_text] = _coerce_row_value(value)
            if normalized_row:
                normalized_rows.append(normalized_row)
        if normalized_rows:
            selected.append({"headers": headers, "rows": normalized_rows})
        else:
            unused += 1
    return selected, unused


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
    def _normalize_assumption_signature(value: str) -> str:
        compact = re.sub(r"\s+", " ", str(value or "")).strip().lower()

        def _normalize_numeric(match: re.Match[str]) -> str:
            raw = match.group(0)
            try:
                return f"{float(raw):g}"
            except Exception:
                return raw

        return re.sub(r"-?\d+(?:\.\d+)?", _normalize_numeric, compact)

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in [*(existing or []), *(additions or [])]:
        payload = record.model_dump() if isinstance(record, AssumptionRecord) else dict(record)
        signature = (
            str(payload.get("key", "")),
            _normalize_assumption_signature(str(payload.get("assumption", ""))),
        )
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(payload)
    return merged


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

    if task_profile == "document_qa" and action_orientation:
        policy["requires_recommendation_class"] = True

    return policy


def build_evidence_pack(
    task_text: str,
    answer_contract: AnswerContract,
    task_profile: str,
    capability_flags: list[str],
    ambiguity_flags: list[str] | None = None,
    *,
    labeled_json_extractor,
) -> tuple[EvidencePack, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pack = get_profile_pack(task_profile)
    urls = extract_urls(task_text)
    document_placeholders = build_document_placeholders(urls) if "needs_files" in capability_flags else []
    inline_facts = extract_inline_facts(task_text, labeled_json_extractor=labeled_json_extractor)
    market_snapshot, derived = derive_market_snapshot(task_text, inline_facts)
    policy_context = _extract_policy_context(task_text, task_profile, capability_flags)
    prompt_facts: dict[str, Any] = dict(inline_facts)
    focus_query = _extract_focus_query(task_text)
    target_period = _extract_target_period(task_text)
    target_entities = extract_entities(focus_query) or extract_entities(task_text)
    formulas = extract_formulas(task_text)
    parsed_tables = parse_markdown_tables(task_text)
    relevant_formulae = _select_relevant_formulas(formulas, focus_query)
    relevant_rows, unused_table_count = _select_relevant_table_rows(
        parsed_tables,
        focus_query=focus_query,
        target_entities=target_entities,
        target_period=target_period,
    )
    if market_snapshot:
        prompt_facts["market_snapshot"] = market_snapshot

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
    if document_placeholders:
        open_questions.append("Document evidence has not been extracted yet; start with metadata or a targeted fetch before answering.")

    evidence = EvidencePack(
        task_constraints=constraints,
        target_entities=target_entities[:6],
        target_period=target_period,
        relevant_rows=relevant_rows,
        relevant_formulae=relevant_formulae,
        required_output={
            "format": answer_contract.format,
            "requires_adapter": answer_contract.requires_adapter,
            "wrapper_key": answer_contract.wrapper_key,
            "section_requirements": answer_contract.section_requirements,
        },
        unused_table_count=unused_table_count,
        answer_contract=answer_contract.model_dump(),
        entities=extract_entities(task_text),
        constraints=constraints,
        prompt_facts=prompt_facts,
        retrieved_facts={},
        derived_facts=derived,
        policy_context=policy_context,
        document_evidence=document_placeholders,
        tables=relevant_rows if relevant_rows else parsed_tables[:2],
        formulas=relevant_formulae if relevant_formulae else formulas[:4],
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
            source_id="evidence_builder",
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
    if relevant_rows:
        provenance_map["relevant_rows"] = ProvenanceRecord(
            source_class="prompt",
            source_id="user_prompt",
            extraction_method="table_row_selection",
        ).model_dump()
    if relevant_formulae:
        provenance_map["relevant_formulae"] = ProvenanceRecord(
            source_class="prompt",
            source_id="user_prompt",
            extraction_method="formula_selection",
        ).model_dump()
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
