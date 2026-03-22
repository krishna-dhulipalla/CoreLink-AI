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
from agent.workflow_models import CuratedContext, SourceBundle, TaskIntent

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


def build_source_bundle(task_text: str) -> SourceBundle:
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


def build_curated_context(
    task_text: str,
    answer_contract: dict[str, Any],
    intent: TaskIntent,
    source_bundle: SourceBundle,
) -> tuple[CuratedContext, dict[str, Any]]:
    evidence_stats: dict[str, Any] = {
        "raw_tables": len(source_bundle.tables),
        "raw_formulas": len(source_bundle.formulas),
        "raw_urls": len(source_bundle.urls),
        "raw_entities": len(source_bundle.entities),
    }
    if intent.task_family == "finance_quant":
        facts_in_use, quant_stats = _quant_facts_in_use(source_bundle)
        evidence_stats.update(quant_stats)
        open_questions: list[str] = []
        assumptions: list[str] = []
        if not quant_stats.get("relevant_rows"):
            open_questions.append("Target row selection is ambiguous; avoid pretending the exact compute is resolved.")
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
    else:
        facts_in_use = _dedupe_facts(
            [{"type": "inline_fact", "key": key, "value": value} for key, value in list(source_bundle.inline_facts.items())[:10]]
        )
        open_questions = []
        assumptions = []

    curated = CuratedContext(
        objective=source_bundle.focus_query or _normalize_text(task_text)[:300],
        facts_in_use=facts_in_use,
        open_questions=open_questions,
        assumptions=assumptions,
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
                "target_period": source_bundle.target_period,
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
        facts.pop("query", None)
        if not facts.get("deal_size_hint"):
            facts.pop("deal_size_hint", None)
        if not facts.get("urgency"):
            facts.pop("urgency", None)
        finding = {
            "tool": str(result.get("type", "") or result.get("tool_name", "")),
            "facts": _compact_prompt_value(facts),
        }
        if finding["tool"]:
            findings.append(finding)
    return findings


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
    # On revision, skip tool findings — they are already reflected in the prior answer
    if not revision_mode:
        tool_findings = _compact_tool_findings(tool_results)
        if tool_findings:
            payload["tool_findings"] = tool_findings
    return json.dumps(payload, ensure_ascii=True)
