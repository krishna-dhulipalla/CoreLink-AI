"""Source-bundle and curated-context helpers for the active runtime."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.context.evidence import (
    _extract_focus_query,
    _extract_policy_context,
    _extract_target_period,
    _select_relevant_formulas,
    _select_relevant_table_rows,
)
from agent.context.extraction import extract_entities, extract_formulas, extract_inline_facts, extract_urls, parse_markdown_tables
from agent.context.profiling import _extract_labeled_json_block
from agent.benchmarks.officeqa import officeqa_analysis_modes
from agent.benchmarks.officeqa_compute import compact_officeqa_compute_result
from agent.contracts import CuratedContext, EvidenceSufficiency, RetrievalIntent, ReviewPacket, SourceBundle, TaskIntent
from agent.officeqa_structured_evidence import build_officeqa_structured_evidence, compact_officeqa_structured_evidence
from agent.retrieval_reasoning import assess_evidence_sufficiency, build_retrieval_intent

_DEAL_SIZE_RE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)\s*(M|MM|B|BN|million|billion)\b", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _deal_size_hint(task_text: str) -> str:
    match = _DEAL_SIZE_RE.search(task_text or "")
    if not match:
        return ""
    return f"{match.group(1)} {match.group(2)}"


def _consideration_preference(task_text: str) -> str:
    normalized = (task_text or "").lower()
    if any(token in normalized for token in ("stock consideration", "stock-for-stock", "stock deal", "equity consideration", "rollover equity", "share consideration")):
        return "equity"
    if "asset purchase" in normalized or "asset deal" in normalized:
        return "asset"
    if any(token in normalized for token in ("cash consideration", "all-cash", "cash deal")):
        return "cash"
    if any(token in normalized for token in ("earnout", "contingent consideration", "hybrid")):
        return "hybrid"
    return ""


def _liability_goal(task_text: str) -> str:
    normalized = (task_text or "").lower()
    if any(token in normalized for token in ("can't risk inheriting", "avoid inheriting", "mitigating the risk", "ring-fence liabilities", "limit liability carryover", "avoid successor liability")):
        return "minimize inherited liabilities"
    if "liability" in normalized:
        return "manage inherited and contingent liabilities"
    return ""


def _urgency_hint(task_text: str) -> str:
    normalized = (task_text or "").lower()
    if any(token in normalized for token in ("quickly", "move quickly", "accelerated", "urgent", "tight timeline")):
        return "accelerated"
    return ""


def _jurisdictions(task_text: str) -> list[str]:
    normalized = (task_text or "").lower()
    jurisdictions: list[str] = []
    if "eu" in normalized or "european union" in normalized:
        jurisdictions.append("EU")
    if re.search(r"\bus\b|\bunited states\b", normalized):
        jurisdictions.append("US")
    if "uk" in normalized or "united kingdom" in normalized:
        jurisdictions.append("UK")
    return jurisdictions


def build_source_bundle(task_text: str, benchmark_overrides: dict[str, Any] | None = None) -> SourceBundle:
    focus_query = _extract_focus_query(task_text)
    target_period = _extract_target_period(task_text)
    entities = extract_entities(focus_query) or extract_entities(task_text)
    urls = extract_urls(task_text)
    inline_facts = extract_inline_facts(task_text, labeled_json_extractor=_extract_labeled_json_block)
    tables = parse_markdown_tables(task_text)
    formulas = extract_formulas(task_text)
    return SourceBundle(
        task_text=task_text,
        focus_query=focus_query,
        target_period=target_period,
        entities=entities,
        urls=urls,
        source_files_expected=list((benchmark_overrides or {}).get("source_files_expected", [])),
        source_files_found=list((benchmark_overrides or {}).get("source_files_found", [])),
        inline_facts=inline_facts,
        tables=tables,
        formulas=formulas,
    )


def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for fact in facts:
        signature = json.dumps(fact, ensure_ascii=True, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(fact)
    return deduped


def _quant_facts_in_use(source_bundle: SourceBundle) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    relevant_formulae = _select_relevant_formulas(source_bundle.formulas, source_bundle.focus_query)
    relevant_rows, unused_table_count = _select_relevant_table_rows(
        source_bundle.tables,
        focus_query=source_bundle.focus_query,
        target_entities=source_bundle.entities,
        target_period=source_bundle.target_period,
    )
    facts: list[dict[str, Any]] = []
    for formula in relevant_formulae:
        facts.append({"type": "formula", "value": formula})
    for table in relevant_rows:
        facts.append({"type": "table_rows", "value": table})
    for key, value in list(source_bundle.inline_facts.items())[:8]:
        facts.append({"type": "inline_fact", "key": key, "value": value})
    return _dedupe_facts(facts), {
        "relevant_formulae": relevant_formulae,
        "relevant_rows": relevant_rows,
        "unused_table_count": unused_table_count,
    }


def _legal_facts_in_use(task_text: str, source_bundle: SourceBundle) -> list[dict[str, Any]]:
    policy_context = _extract_policy_context(task_text, "legal_transactional", ["needs_legal_reasoning"])
    facts: list[dict[str, Any]] = []
    if source_bundle.entities:
        facts.append({"type": "entities", "value": source_bundle.entities[:6]})
    if _deal_size_hint(task_text):
        facts.append({"type": "deal_size_hint", "value": _deal_size_hint(task_text)})
    if _consideration_preference(task_text):
        facts.append({"type": "consideration_preference", "value": _consideration_preference(task_text)})
    if _liability_goal(task_text):
        facts.append({"type": "liability_goal", "value": _liability_goal(task_text)})
    if _urgency_hint(task_text):
        facts.append({"type": "urgency", "value": _urgency_hint(task_text)})
    if _jurisdictions(task_text):
        facts.append({"type": "jurisdictions", "value": _jurisdictions(task_text)})
    if policy_context and set(policy_context.keys()) - {"jurisdictions"}:
        facts.append({"type": "policy_context", "value": policy_context})
    return _dedupe_facts(facts)


def _analytical_facts_in_use(source_bundle: SourceBundle) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for formula in _select_relevant_formulas(source_bundle.formulas, source_bundle.focus_query or source_bundle.task_text):
        facts.append({"type": "formula", "value": formula})
    for key, value in list(source_bundle.inline_facts.items())[:10]:
        facts.append({"type": "inline_fact", "key": key, "value": value})
    if source_bundle.entities:
        facts.append({"type": "entities", "value": source_bundle.entities[:6]})
    return _dedupe_facts(facts)


def _market_scenario_facts(task_text: str, source_bundle: SourceBundle) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    if source_bundle.entities:
        facts.append({"type": "entities", "value": source_bundle.entities[:6]})
    for key, value in list(source_bundle.inline_facts.items())[:10]:
        facts.append({"type": "inline_fact", "key": key, "value": value})
    lowered = (task_text or "").lower()
    scenario_markers = []
    for marker in ("stress", "flash crash", "liquidity", "drawdown", "hedge", "earnings", "volatility", "scenario"):
        if marker in lowered:
            scenario_markers.append(marker)
    if scenario_markers:
        facts.append({"type": "scenario_markers", "value": scenario_markers[:6]})
    return _dedupe_facts(facts)


def _retrieval_facts_in_use(task_text: str, source_bundle: SourceBundle) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    if source_bundle.focus_query:
        facts.append({"type": "focus_query", "value": source_bundle.focus_query})
    if source_bundle.entities:
        facts.append({"type": "entities", "value": source_bundle.entities[:8]})
    if source_bundle.target_period:
        facts.append({"type": "target_period", "value": source_bundle.target_period})
    if source_bundle.urls:
        facts.append({"type": "reference_urls", "value": source_bundle.urls[:6]})
    if source_bundle.source_files_expected:
        facts.append({"type": "source_files_expected", "value": source_bundle.source_files_expected[:8]})
    if source_bundle.source_files_found:
        facts.append({"type": "source_files_found", "value": source_bundle.source_files_found[:6]})
    for key, value in list(source_bundle.inline_facts.items())[:8]:
        facts.append({"type": "inline_fact", "key": key, "value": value})
    lowered = (task_text or "").lower()
    if any(token in lowered for token in ("according to", "report", "bulletin", "filing", "document", "source")):
        facts.append({"type": "grounding_required", "value": True})
    return _dedupe_facts(facts)


def _officeqa_document_facts_in_use(
    task_text: str,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    retrieval_intent = build_retrieval_intent(task_text, source_bundle, benchmark_overrides)
    analysis_modes = officeqa_analysis_modes(task_text)
    facts = _retrieval_facts_in_use(task_text, source_bundle)
    facts.append({"type": "officeqa_analysis_modes", "value": analysis_modes})
    if retrieval_intent.metric:
        facts.append({"type": "retrieval_metric", "value": retrieval_intent.metric})
    if retrieval_intent.entity:
        facts.append({"type": "retrieval_entity", "value": retrieval_intent.entity})
    if retrieval_intent.period:
        facts.append({"type": "retrieval_period", "value": retrieval_intent.period})
    if retrieval_intent.aggregation_shape:
        facts.append({"type": "aggregation_shape", "value": retrieval_intent.aggregation_shape})
    if retrieval_intent.document_family:
        facts.append({"type": "document_family", "value": retrieval_intent.document_family})
    if retrieval_intent.strategy:
        facts.append({"type": "retrieval_strategy", "value": retrieval_intent.strategy})
    if retrieval_intent.fallback_chain:
        facts.append({"type": "retrieval_fallback_chain", "value": retrieval_intent.fallback_chain[:4]})
    if retrieval_intent.evidence_requirements:
        facts.append({"type": "evidence_requirements", "value": retrieval_intent.evidence_requirements[:5]})
    if retrieval_intent.evidence_plan.metric_identity:
        facts.append({"type": "evidence_metric_identity", "value": retrieval_intent.evidence_plan.metric_identity})
    if retrieval_intent.evidence_plan.required_years:
        facts.append({"type": "evidence_required_years", "value": retrieval_intent.evidence_plan.required_years[:4]})
    if retrieval_intent.query_candidates:
        facts.append({"type": "query_candidates", "value": retrieval_intent.query_candidates[:3]})

    open_questions = [
        "Confirm the exact Treasury source, entity, period, aggregation, and unit support before finalizing the answer.",
    ]
    if "inflation_adjustment" in analysis_modes:
        open_questions.append("Check whether CPI or inflation support is required alongside the target financial series.")
    if "statistical_analysis" in analysis_modes:
        open_questions.append("Verify the extracted series is complete enough for the requested statistical analysis.")
    if "time_series_forecasting" in analysis_modes:
        open_questions.append("Verify the retrieved series is ordered and complete enough for grounded forecasting or trend projection.")
    if "risk_metric" in analysis_modes:
        open_questions.append("Verify the risk metric inputs and units are grounded in the retrieved Treasury evidence.")
    return _dedupe_facts(facts), list(dict.fromkeys(open_questions))


def build_curated_context(
    task_text: str,
    answer_contract: dict[str, Any],
    intent: TaskIntent,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> tuple[CuratedContext, dict[str, Any]]:
    evidence_stats: dict[str, Any] = {
        "raw_tables": len(source_bundle.tables),
        "raw_formulas": len(source_bundle.formulas),
        "raw_urls": len(source_bundle.urls),
        "raw_entities": len(source_bundle.entities),
    }
    retrieval_intent_obj: RetrievalIntent | None = None
    overrides = dict(benchmark_overrides or {})
    if str(overrides.get("benchmark_adapter", "") or "") == "officeqa":
        retrieval_intent_obj = build_retrieval_intent(task_text, source_bundle, overrides)
        facts_in_use, open_questions = _officeqa_document_facts_in_use(task_text, source_bundle, overrides)
        assumptions: list[str] = []
    elif intent.task_family == "finance_quant":
        facts_in_use, quant_stats = _quant_facts_in_use(source_bundle)
        evidence_stats.update(quant_stats)
        open_questions: list[str] = []
        assumptions: list[str] = []
        if not quant_stats.get("relevant_rows"):
            open_questions.append("Target row selection is ambiguous; avoid pretending the exact compute is resolved.")
    elif intent.task_family == "analytical_reasoning":
        facts_in_use = _analytical_facts_in_use(source_bundle)
        open_questions = []
        assumptions = []
        if not facts_in_use:
            open_questions.append("No explicit structured evidence was extracted; derivation must stay close to the task text.")
    elif intent.task_family == "market_scenario":
        facts_in_use = _market_scenario_facts(task_text, source_bundle)
        open_questions = [
            "Confirm position sizing, mandate limits, and stress constraints if they are not explicit in the prompt."
        ]
        assumptions = []
    elif intent.task_family == "legal_transactional":
        facts_in_use = _legal_facts_in_use(task_text, source_bundle)
        open_questions = [
            "Confirm the severity of the core execution risks and whether they are curable pre-closing.",
            "Confirm willingness to provide indemnities, escrow, holdback, insurance, or other downside protection.",
            "Confirm whether workforce transfer, stakeholder approvals, consultations, or third-party consents create signing-to-closing constraints.",
        ]
        assumptions = [
            "Exact severity and curability of the execution risks are not specified.",
            "Seller support for indemnity, escrow, or other downside protection is not yet confirmed.",
        ]
    elif intent.execution_mode in {"retrieval_augmented_analysis", "document_grounded_analysis"}:
        facts_in_use = _retrieval_facts_in_use(task_text, source_bundle)
        retrieval_intent_obj = build_retrieval_intent(task_text, source_bundle)
        if retrieval_intent_obj.entity:
            facts_in_use.append({"type": "retrieval_entity", "value": retrieval_intent_obj.entity})
        if retrieval_intent_obj.period:
            facts_in_use.append({"type": "retrieval_period", "value": retrieval_intent_obj.period})
        if retrieval_intent_obj.aggregation_shape:
            facts_in_use.append({"type": "aggregation_shape", "value": retrieval_intent_obj.aggregation_shape})
        if retrieval_intent_obj.document_family:
            facts_in_use.append({"type": "document_family", "value": retrieval_intent_obj.document_family})
        if retrieval_intent_obj.query_candidates:
            facts_in_use.append({"type": "query_candidates", "value": retrieval_intent_obj.query_candidates[:3]})
        open_questions = [
            "Find the exact supporting quote, table row, or document window before finalizing the answer.",
        ]
        assumptions = []
    else:
        facts_in_use = _dedupe_facts(
            [{"type": "inline_fact", "key": key, "value": value} for key, value in list(source_bundle.inline_facts.items())[:10]]
        )
        open_questions = []
        assumptions = []

    facts_in_use = _dedupe_facts(facts_in_use)
    curated = CuratedContext(
        objective=source_bundle.focus_query or _normalize_text(task_text)[:300],
        facts_in_use=facts_in_use,
        open_questions=[item for item in open_questions if item],
        assumptions=[item for item in assumptions if item],
        requested_output={
            "format": answer_contract.get("format", "text"),
            "requires_adapter": bool(answer_contract.get("requires_adapter")),
            "wrapper_key": answer_contract.get("wrapper_key"),
            "section_requirements": list(answer_contract.get("section_requirements", [])),
        },
        provenance_summary={
            "source_bundle": {
                "entities": source_bundle.entities[:6],
                "urls": source_bundle.urls[:4],
                "source_files_expected": source_bundle.source_files_expected[:8],
                "source_files_found": source_bundle.source_files_found[:8],
                "target_period": source_bundle.target_period,
            },
            "retrieval_plan": {
                "strategy": retrieval_intent_obj.strategy if retrieval_intent_obj else "",
                "strategy_confidence": retrieval_intent_obj.strategy_confidence if retrieval_intent_obj else 0.0,
                "fallback_chain": list(retrieval_intent_obj.fallback_chain[:4]) if retrieval_intent_obj else [],
                "evidence_requirements": list(retrieval_intent_obj.evidence_requirements[:5]) if retrieval_intent_obj else [],
                "required_years": list(retrieval_intent_obj.evidence_plan.required_years[:4]) if retrieval_intent_obj else [],
                "join_keys": list(retrieval_intent_obj.evidence_plan.join_keys[:6]) if retrieval_intent_obj else [],
            },
            "fact_count": len(facts_in_use),
        },
    )
    return curated, evidence_stats


def _compact_prompt_value(value: Any) -> Any:
    if isinstance(value, str):
        compact = _normalize_text(value)
        return compact[:240] + "..." if len(compact) > 240 else compact
    if isinstance(value, list):
        compact_items = [_compact_prompt_value(item) for item in value]
        compact_items = [item for item in compact_items if item not in ("", [], {}, None)]
        return compact_items[:6]
    if isinstance(value, dict):
        compact_dict = {str(key): _compact_prompt_value(val) for key, val in value.items()}
        return {key: val for key, val in compact_dict.items() if val not in ("", [], {}, None)}
    return value


def _compact_tool_findings(tool_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in tool_results or []:
        if not isinstance(result, dict):
            continue
        facts = dict(result.get("facts") or {})
        retrieval_status = str(result.get("retrieval_status", "") or facts.get("retrieval_status", "") or "")
        evidence_quality = float(result.get("evidence_quality_score", 0.0) or 0.0)
        facts.pop("query", None)
        if not facts.get("deal_size_hint"):
            facts.pop("deal_size_hint", None)
        if not facts.get("urgency"):
            facts.pop("urgency", None)
        if retrieval_status in {"garbled_binary", "parse_error", "network_error", "unsupported_format"}:
            facts = {
                "retrieval_status": retrieval_status,
                "metadata": _compact_prompt_value(facts.get("metadata", {})),
                "errors": _compact_prompt_value(result.get("errors", [])),
            }
        if "results" in facts and isinstance(facts["results"], list):
            facts["results"] = [
                {
                    "title": item.get("title", ""),
                    "snippet": _compact_prompt_value(item.get("snippet", "")),
                    "citation": item.get("url", "") or item.get("citation", ""),
                }
                for item in facts["results"][:4]
                if isinstance(item, dict)
            ]
        if "documents" in facts and isinstance(facts["documents"], list):
            facts["documents"] = [
                {
                    "document_id": item.get("document_id", ""),
                    "citation": item.get("citation", "") or item.get("url", ""),
                    "format": item.get("format", ""),
                }
                for item in facts["documents"][:6]
                if isinstance(item, dict)
            ]
        if "chunks" in facts and isinstance(facts["chunks"], list):
            facts["chunks"] = [
                {
                    "locator": item.get("locator", ""),
                    "text": _compact_prompt_value(item.get("text", "")),
                    "citation": item.get("citation", ""),
                }
                for item in facts["chunks"][:4]
                if isinstance(item, dict)
            ]
        if "tables" in facts and isinstance(facts["tables"], list):
            facts["tables"] = [
                {
                    "locator": item.get("locator", ""),
                    "headers": list(item.get("headers", []))[:8],
                    "citation": item.get("citation", ""),
                }
                for item in facts["tables"][:2]
                if isinstance(item, dict)
            ]
        finding = {
            "tool": str(result.get("type", "") or result.get("tool_name", "")),
            "facts": _compact_prompt_value(facts),
        }
        if retrieval_status:
            finding["retrieval_status"] = retrieval_status
        if evidence_quality:
            finding["evidence_quality_score"] = round(evidence_quality, 3)
        if finding["tool"]:
            findings.append(finding)
    return findings


def _extract_tool_citations(tool_results: list[dict[str, Any]] | None) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for result in tool_results or []:
        if not isinstance(result, dict):
            continue
        facts = dict(result.get("facts") or {})
        direct = [facts.get("citation", "")]
        for item in facts.get("results", []):
            if isinstance(item, dict):
                direct.append(item.get("url", "") or item.get("citation", ""))
        for item in facts.get("documents", []):
            if isinstance(item, dict):
                direct.append(item.get("citation", "") or item.get("url", ""))
        for item in facts.get("chunks", []):
            if isinstance(item, dict):
                direct.append(item.get("citation", ""))
        for item in direct:
            citation = str(item or "").strip()
            if not citation or citation in seen:
                continue
            seen.add(citation)
            citations.append(citation)
    return citations


def attach_structured_evidence(
    curated_context: dict[str, Any] | CuratedContext,
    tool_results: list[dict[str, Any]] | None = None,
    benchmark_overrides: dict[str, Any] | None = None,
) -> CuratedContext:
    curated = curated_context if isinstance(curated_context, CuratedContext) else CuratedContext.model_validate(curated_context)
    overrides = dict(benchmark_overrides or {})
    if str(overrides.get("benchmark_adapter", "") or "") != "officeqa":
        return curated
    structured = build_officeqa_structured_evidence(tool_results)
    if not structured.get("tables") and not structured.get("values"):
        return curated

    facts = list(curated.facts_in_use)
    facts.append({"type": "structured_value_count", "value": int(structured.get("value_count", 0) or 0)})
    if structured.get("units_seen"):
        facts.append({"type": "structured_units", "value": list(structured.get("units_seen", []))[:8]})
    if structured.get("values"):
        facts.append(
            {
                "type": "structured_sample_values",
                "value": compact_officeqa_structured_evidence(structured).get("values", [])[:6],
            }
        )

    provenance_summary = dict(curated.provenance_summary or {})
    provenance_summary["structured_evidence"] = {
        "table_count": len(list(structured.get("tables", []))),
        "value_count": int(structured.get("value_count", 0) or 0),
        "units_seen": list(structured.get("units_seen", []))[:8],
        "provenance_complete": bool(structured.get("provenance_complete")),
    }

    curated.facts_in_use = _dedupe_facts(facts)
    curated.provenance_summary = provenance_summary
    curated.structured_evidence = structured
    return curated


def attach_compute_result(
    curated_context: dict[str, Any] | CuratedContext,
    compute_result: dict[str, Any] | None = None,
) -> CuratedContext:
    curated = curated_context if isinstance(curated_context, CuratedContext) else CuratedContext.model_validate(curated_context)
    compact = compact_officeqa_compute_result(compute_result)
    if not compact:
        return curated

    facts = list(curated.facts_in_use)
    if compact.get("operation"):
        facts.append({"type": "compute_operation", "value": compact.get("operation", "")})
    if compact.get("display_value"):
        facts.append({"type": "computed_value", "value": compact.get("display_value", "")})

    provenance_summary = dict(curated.provenance_summary or {})
    provenance_summary["compute_result"] = {
        "status": compact.get("status", ""),
        "operation": compact.get("operation", ""),
        "validation_errors": list(compact.get("validation_errors", [])),
        "provenance_complete": bool(compact.get("provenance_complete")),
    }

    curated.facts_in_use = _dedupe_facts(facts)
    curated.provenance_summary = provenance_summary
    curated.compute_result = dict(compute_result or {})
    return curated


def build_review_packet(
    *,
    task_text: str,
    answer_text: str,
    answer_contract: dict[str, Any],
    curated_context: dict[str, Any],
    tool_results: list[dict[str, Any]] | None = None,
    evidence_sufficiency: dict[str, Any] | None = None,
    validator_result: dict[str, Any] | None = None,
) -> ReviewPacket:
    provenance = dict(curated_context.get("provenance_summary") or {})
    source_summary = dict(provenance.get("source_bundle") or {})
    citations = [str(item) for item in source_summary.get("urls", []) if str(item).strip()]
    citations.extend([item for item in _extract_tool_citations(tool_results) if item not in citations])
    return ReviewPacket(
        task_text=task_text,
        answer_text=answer_text,
        answer_contract=answer_contract,
        tool_findings=_compact_tool_findings(tool_results),
        citations=citations,
        assumptions=[str(item) for item in curated_context.get("assumptions", []) if str(item).strip()],
        open_questions=[str(item) for item in curated_context.get("open_questions", []) if str(item).strip()],
        evidence_sufficiency=dict(evidence_sufficiency or {}),
        structured_evidence=compact_officeqa_structured_evidence(curated_context.get("structured_evidence", {})),
        compute_result=compact_officeqa_compute_result(curated_context.get("compute_result", {})),
        validator_result=dict(validator_result or {}),
    )


def solver_context_block(
    curated_context: dict[str, Any],
    tool_results: list[dict[str, Any]] | None = None,
    *,
    include_objective: bool = False,
    revision_mode: bool = False,
) -> str:
    payload = {
        "facts_in_use": _compact_prompt_value(curated_context.get("facts_in_use", [])),
        "requested_output": _compact_prompt_value(curated_context.get("requested_output", {})),
    }
    open_questions = _compact_prompt_value(curated_context.get("open_questions", []))
    assumptions = _compact_prompt_value(curated_context.get("assumptions", []))
    if open_questions:
        payload["open_questions"] = open_questions
    if assumptions:
        payload["assumptions"] = assumptions
    if include_objective and curated_context.get("objective"):
        payload["objective"] = _compact_prompt_value(curated_context.get("objective", ""))
    structured_evidence = compact_officeqa_structured_evidence(curated_context.get("structured_evidence", {}))
    if structured_evidence:
        payload["structured_evidence"] = structured_evidence
    compute_result = compact_officeqa_compute_result(curated_context.get("compute_result", {}))
    if compute_result:
        payload["compute_result"] = compute_result
    # On revision, skip tool findings — they are already reflected in the prior answer
    if not revision_mode:
        tool_findings = _compact_tool_findings(tool_results)
        if tool_findings:
            payload["tool_findings"] = tool_findings
    return json.dumps(payload, ensure_ascii=True)


def build_retrieval_bundle(
    task_text: str,
    source_bundle: SourceBundle,
    benchmark_overrides: dict[str, Any] | None = None,
) -> tuple[RetrievalIntent, EvidenceSufficiency]:
    retrieval_intent = build_retrieval_intent(task_text, source_bundle, benchmark_overrides)
    evidence_sufficiency = assess_evidence_sufficiency(task_text, source_bundle, [], benchmark_overrides)
    return retrieval_intent, evidence_sufficiency
