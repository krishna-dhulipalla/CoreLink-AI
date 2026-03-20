"""
Common solver helpers shared across stage implementations.
"""

from __future__ import annotations

import ast
import json
import logging
import operator as _op
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage

from agent.document_evidence import summarize_document_evidence
from agent.profile_packs import get_profile_pack
from agent.runtime_support import allowed_tools_for_template, latest_human_text
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_ASSIGNMENT_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*(%)?")
_FORMULA_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_ ]{1,80})\s*=\s*([^\n.;]+)")
_SAFE_OPERATORS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
    ast.Mod: _op.mod,
    ast.FloorDiv: _op.floordiv,
}


def strip_think_markup(text: str) -> str:
    clean = _THINK_BLOCK_RE.sub("", text)
    clean = clean.replace("<think>", "").replace("</think>", "")
    return clean.strip()


def allowed_tools_for_state(all_tools: list, state: AgentState) -> list:
    profile = state.get("task_profile", "general")
    flags = set(state.get("capability_flags", []))
    ambiguity = set(state.get("ambiguity_flags", []))
    stage = state.get("solver_stage", "SYNTHESIZE")
    template = state.get("execution_template", {})
    task_text = latest_human_text(state["messages"]).lower()

    allowed_names = allowed_tools_for_template(template, profile)

    if profile == "legal_transactional" and stage in {"COMPUTE", "SYNTHESIZE"}:
        if not ({"needs_files", "needs_live_data", "needs_math"} & flags):
            allowed_names = set()

    if stage == "SYNTHESIZE":
        allowed_names = set()

    if stage == "GATHER" and template.get("template_id") in {"legal_with_document_evidence", "document_qa"}:
        allowed_names &= {"fetch_reference_file", "list_reference_files"}

    review_feedback = state.get("review_feedback") or {}
    risk_feedback = state.get("risk_feedback") or {}
    repair_target = str(review_feedback.get("repair_target", risk_feedback.get("repair_target", "final")))
    if stage == "REVISE" and repair_target not in {"gather", "compute"}:
        allowed_names = set()
    if (
        stage == "REVISE"
        and repair_target == "compute"
        and (state.get("last_tool_result") or {}).get("facts")
        and not (state.get("last_tool_result") or {}).get("errors")
        and not any(
            token in " ".join(
                [
                    str(risk_feedback.get("reasoning", "")),
                    " ".join(str(item) for item in risk_feedback.get("violation_codes", [])),
                    " ".join(str(item) for item in risk_feedback.get("risk_findings", [])),
                    " ".join(str(item) for item in risk_feedback.get("required_disclosures", [])),
                    " ".join(str(item) for item in review_feedback.get("missing_dimensions", [])),
                ]
            ).lower()
            for token in ("scenario", "stress", "var", "risk", "limit", "exposure")
        )
    ):
        allowed_names = set()

    if "external_retrieval" != profile and "internet_search" in allowed_names and "latest" not in task_text and "current" not in task_text:
        allowed_names.discard("internet_search")

    if ambiguity and profile == "general":
        allowed_names &= {"calculator", "fetch_reference_file", "list_reference_files"}

    return [tool for tool in all_tools if getattr(tool, "name", "") in allowed_names]


def build_tool_prompt_block(tools: list) -> str:
    if not tools:
        return ""
    lines = [
        "Allowed tools for this stage:",
        'To call a tool, respond with ONLY: {"name":"tool_name","arguments":{...}}',
    ]
    for tool in tools:
        lines.append(f"- {getattr(tool, 'name', 'unknown')}: {getattr(tool, 'description', '')}")
    return "\n".join(lines)


def patch_prompt_tool_call(response: AIMessage, tools: list, extract_json_payload) -> AIMessage:
    if response.tool_calls or not response.content:
        return response

    content = strip_think_markup(str(response.content))
    try:
        payload = json.loads(extract_json_payload(content))
    except Exception:
        return response

    if "name" not in payload or "arguments" not in payload:
        return response

    tool_name = str(payload.get("name", "")).strip()
    valid_names = {getattr(tool, "name", "") for tool in tools}
    if tool_name:
        if tool_name not in valid_names:
            logger.warning(
                "[Solver] Preserving explicit tool envelope for downstream validation of hidden/unknown tool '%s'.",
                tool_name,
            )
        response.tool_calls = [
            {
                "name": tool_name,
                "args": payload.get("arguments", {}) if isinstance(payload.get("arguments", {}), dict) else {},
                "id": f"call_{uuid.uuid4().hex[:10]}",
                "type": "tool_call",
            }
        ]
        response.content = ""
    return response


def estimate_response_tokens(response: AIMessage) -> int:
    parts = []
    if response.content:
        parts.append(str(response.content))
    if response.tool_calls:
        parts.append(json.dumps(response.tool_calls))
    if not parts:
        return 0
    return count_tokens([AIMessage(content="\n".join(parts))])


def _filtered_prompt_facts(evidence: dict[str, Any], complexity_tier: str) -> dict[str, Any]:
    prompt_facts = dict(evidence.get("prompt_facts", {}))
    if complexity_tier == "simple_exact":
        keep_keys = {"as_of_date", "structured_inputs"}
        return {key: value for key, value in prompt_facts.items() if key in keep_keys}
    return prompt_facts


def _solver_facing_evidence(evidence: dict[str, Any], complexity_tier: str) -> dict[str, Any]:
    relevant_rows = evidence.get("relevant_rows", [])
    relevant_formulae = evidence.get("relevant_formulae", [])
    payload = {
        "task_constraints": evidence.get("task_constraints", []),
        "target_entities": evidence.get("target_entities", []),
        "target_period": evidence.get("target_period", ""),
        "relevant_rows": relevant_rows,
        "relevant_formulae": relevant_formulae,
        "required_output": evidence.get("required_output", {}),
        "constraints": evidence.get("constraints", []),
        "entities": evidence.get("entities", []),
        "prompt_facts": _filtered_prompt_facts(evidence, complexity_tier),
        "retrieved_facts": {
            key: value
            for key, value in evidence.get("retrieved_facts", {}).items()
            if key not in {"fetch_reference_file"}
        },
        "derived_facts": evidence.get("derived_facts", {}),
        "policy_context": evidence.get("policy_context", {}),
        "document_evidence": summarize_document_evidence(evidence.get("document_evidence", [])),
        "tables": [] if relevant_rows else evidence.get("tables", [])[:2],
        "formulas": [] if relevant_formulae else evidence.get("formulas", [])[:2],
        "citations": evidence.get("citations", []),
        "open_questions": evidence.get("open_questions", []),
        "unused_table_count": evidence.get("unused_table_count", 0),
    }
    if complexity_tier == "simple_exact":
        payload["retrieved_facts"] = {}
        payload["document_evidence"] = []
        payload["open_questions"] = []
        payload["citations"] = []
        payload["derived_facts"] = {}
        payload["entities"] = payload.get("target_entities", [])[:4]
    return payload


def solver_user_message(state: AgentState) -> str:
    evidence = state.get("evidence_pack", {}) or {}
    workpad = state.get("workpad", {}) or {}
    complexity_tier = str(workpad.get("task_complexity_tier", "structured_analysis"))
    if complexity_tier != "simple_exact":
        return latest_human_text(state["messages"])
    if not evidence.get("target_entities") and not evidence.get("relevant_rows"):
        return latest_human_text(state["messages"])

    compact_task = {
        "task_profile": state.get("task_profile", "general"),
        "target_entities": evidence.get("target_entities", [])[:4],
        "target_period": evidence.get("target_period", ""),
        "required_output": evidence.get("required_output", {}),
    }
    return json.dumps(compact_task, ensure_ascii=True)


def compact_evidence_block(state: AgentState) -> str:
    evidence = state.get("evidence_pack", {})
    workpad = state.get("workpad", {})
    review_feedback = state.get("review_feedback") or {}
    assumption_ledger = list(state.get("assumption_ledger", []))
    provenance_map = dict(state.get("provenance_map", {}))
    profile_pack = workpad.get("profile_pack") or get_profile_pack(state.get("task_profile", "general")).model_dump()
    answer_contract = state.get("answer_contract", {})
    execution_template = state.get("execution_template", {})
    complexity_tier = str(workpad.get("task_complexity_tier", "structured_analysis"))
    payload = {
        "task_profile": state.get("task_profile", "general"),
        "capability_flags": state.get("capability_flags", []),
        "task_complexity_tier": complexity_tier,
        "execution_template": {
            "template_id": execution_template.get("template_id"),
            "description": execution_template.get("description", ""),
            "answer_focus": execution_template.get("answer_focus", []),
        },
        "profile_pack": {
            "domain_summary": profile_pack.get("domain_summary", ""),
            "content_rules": profile_pack.get("content_rules", []),
            "section_requirements": profile_pack.get("section_requirements", []),
            "required_evidence_types": profile_pack.get("required_evidence_types", []),
            "failure_modes": profile_pack.get("failure_modes", []),
        },
        "answer_contract": {
            "format": answer_contract.get("format", "text"),
            "requires_adapter": answer_contract.get("requires_adapter", False),
            "wrapper_key": answer_contract.get("wrapper_key"),
            "xml_root_tag": answer_contract.get("xml_root_tag"),
            "content_rules": answer_contract.get("content_rules", []),
            "section_requirements": answer_contract.get("section_requirements", []),
            "schema_hint": answer_contract.get("schema_hint", {}),
            "value_rules": answer_contract.get("value_rules", {}),
        },
        "evidence_pack": _solver_facing_evidence(evidence, complexity_tier),
        "assumption_ledger": assumption_ledger[:8],
        "provenance_summary": {
            key: {
                "source_class": value.get("source_class"),
                "source_id": value.get("source_id"),
                "tool_name": value.get("tool_name"),
            }
            for key, value in list(provenance_map.items())[: (8 if complexity_tier == "simple_exact" else 20)]
        },
        "last_tool_result": state.get("last_tool_result"),
        "stage_outputs": workpad.get("stage_outputs", {}),
        "review_feedback": review_feedback,
    }
    if complexity_tier == "simple_exact":
        payload["profile_pack"] = {
            "domain_summary": profile_pack.get("domain_summary", ""),
            "content_rules": profile_pack.get("content_rules", [])[:2],
            "section_requirements": [],
            "required_evidence_types": profile_pack.get("required_evidence_types", [])[:2],
            "failure_modes": [],
        }
        payload["review_feedback"] = {
            "verdict": review_feedback.get("verdict"),
            "repair_target": review_feedback.get("repair_target"),
            "repair_class": review_feedback.get("repair_class"),
        }
    return json.dumps(payload, ensure_ascii=True, default=str)


def solver_max_tokens(stage: str, profile: str) -> int:
    if stage in {"SYNTHESIZE", "REVISE"}:
        if profile == "legal_transactional":
            return 1400
        if profile == "finance_options":
            return 1200
    if stage == "COMPUTE":
        return 900
    return 700


def safe_arithmetic_eval(expression: str) -> float | None:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
            return float(_SAFE_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
            return float(_SAFE_OPERATORS[type(node.op)](_eval(node.operand)))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        return _eval(parsed)
    except Exception:
        return None


def extract_inline_assignments(text: str) -> dict[str, float]:
    assignments: dict[str, float] = {}
    for match in _ASSIGNMENT_RE.finditer(text or ""):
        key = match.group(1).strip()
        value = float(match.group(2))
        assignments[key] = value
        assignments[key.upper()] = value
    return assignments


def format_scalar_number(value: float) -> str:
    return f"{float(value):.10g}"


def record_event(workpad: dict[str, Any], node: str, action: str) -> dict[str, Any]:
    updated = dict(workpad)
    events = list(updated.get("events", []))
    events.append({"node": node, "action": action})
    updated["events"] = events
    return updated


def merge_stage_output(workpad: dict[str, Any], stage: str, text: str) -> dict[str, Any]:
    updated = dict(workpad)
    outputs = dict(updated.get("stage_outputs", {}))
    outputs[stage] = text
    updated["stage_outputs"] = outputs
    return updated
