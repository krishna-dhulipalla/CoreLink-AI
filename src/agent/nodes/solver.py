"""
Solver Node
===========
Stage-based solver for the finance-first runtime.
"""

from __future__ import annotations

import ast
import json
import logging
import operator as _op
import re
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.contracts import ReviewResult, ToolCallEnvelope, ToolResult
from agent.document_evidence import has_extracted_document_evidence, summarize_document_evidence
from agent.cost import CostTracker
from agent.nodes.compliance_guard import requires_compliance_guard
from agent.model_config import _extract_json_payload, _tool_call_mode, get_client_kwargs, get_model_name
from agent.nodes.risk_controller import requires_risk_control
from agent.profile_packs import get_profile_pack
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import (
    allowed_tools_for_template,
    latest_human_text,
    next_stage_after_review,
    stage_is_review_milestone,
)
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

SOLVER_PROMPT = """You are the staged solver in a finance-first runtime.
Work only on the current stage.
Use the provided evidence pack and workpad summary.
Do not restate the whole task.
If a tool is allowed and truly needed, call exactly one tool.
If no tool is needed, return only the stage output requested.
"""

_STAGE_INSTRUCTIONS = {
    "GATHER": (
        "Current stage: GATHER.\n"
        "Goal: gather one missing piece of evidence only if it is justified by the task profile and flags.\n"
        "If a tool is needed, emit one tool call.\n"
        "If prompt-contained evidence is already sufficient, return a short evidence summary only."
    ),
    "COMPUTE": (
        "Current stage: COMPUTE.\n"
        "Goal: produce a compact analytical summary from the evidence and any tool result.\n"
        "Call one exact compute or finance tool only if it materially improves precision.\n"
        "Otherwise return a compact computation summary that can be reviewed before synthesis."
    ),
    "SYNTHESIZE": (
        "Current stage: SYNTHESIZE.\n"
        "Goal: write the final user-facing answer.\n"
        "Respect the answer contract, but do not add formatting wrappers unless the output adapter is required later.\n"
        "Do not narrate your reasoning process or include planning filler.\n"
        "Use compact sections or bullets only when they improve clarity."
    ),
    "REVISE": (
        "Current stage: REVISE.\n"
        "Goal: fix only the missing dimensions identified by the reviewer.\n"
        "When repairing a final answer, return a complete replacement final answer that preserves valid sections and adds the missing ones.\n"
        "Do not include planning filler or self-talk.\n"
        "Use tools again only if the reviewer explicitly points to missing evidence or missing computation."
    ),
}


def _strip_think_markup(text: str) -> str:
    clean = _THINK_BLOCK_RE.sub("", text)
    clean = clean.replace("<think>", "").replace("</think>", "")
    return clean.strip()


def _allowed_tools_for_state(all_tools: list, state: AgentState) -> list:
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


def _build_tool_prompt_block(tools: list) -> str:
    if not tools:
        return ""
    lines = [
        "Allowed tools for this stage:",
        'To call a tool, respond with ONLY: {"name":"tool_name","arguments":{...}}',
    ]
    for tool in tools:
        lines.append(f"- {getattr(tool, 'name', 'unknown')}: {getattr(tool, 'description', '')}")
    return "\n".join(lines)


def _patch_prompt_tool_call(response: AIMessage, tools: list) -> AIMessage:
    if response.tool_calls or not response.content:
        return response

    content = _strip_think_markup(str(response.content))
    try:
        payload = json.loads(_extract_json_payload(content))
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


def _estimate_response_tokens(response: AIMessage) -> int:
    parts = []
    if response.content:
        parts.append(str(response.content))
    if response.tool_calls:
        parts.append(json.dumps(response.tool_calls))
    if not parts:
        return 0
    return count_tokens([AIMessage(content="\n".join(parts))])


def _compact_evidence_block(state: AgentState) -> str:
    evidence = state.get("evidence_pack", {})
    workpad = state.get("workpad", {})
    review_feedback = state.get("review_feedback") or {}
    assumption_ledger = list(state.get("assumption_ledger", []))
    provenance_map = dict(state.get("provenance_map", {}))
    profile_pack = workpad.get("profile_pack") or get_profile_pack(state.get("task_profile", "general")).model_dump()
    answer_contract = state.get("answer_contract", {})
    execution_template = state.get("execution_template", {})
    payload = {
        "task_profile": state.get("task_profile", "general"),
        "capability_flags": state.get("capability_flags", []),
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
        "evidence_pack": {
            "task_brief": evidence.get("task_brief", ""),
            "constraints": evidence.get("constraints", []),
            "entities": evidence.get("entities", []),
            "prompt_facts": evidence.get("prompt_facts", {}),
            "retrieved_facts": {
                key: value
                for key, value in evidence.get("retrieved_facts", {}).items()
                if key not in {"fetch_reference_file"}
            },
            "derived_facts": evidence.get("derived_facts", {}),
            "policy_context": evidence.get("policy_context", {}),
            "document_evidence": summarize_document_evidence(evidence.get("document_evidence", [])),
            "tables": evidence.get("tables", []),
            "formulas": evidence.get("formulas", []),
            "citations": evidence.get("citations", []),
            "open_questions": evidence.get("open_questions", []),
        },
        "assumption_ledger": assumption_ledger[:8],
        "provenance_summary": {
            key: {
                "source_class": value.get("source_class"),
                "source_id": value.get("source_id"),
                "tool_name": value.get("tool_name"),
            }
            for key, value in list(provenance_map.items())[:20]
        },
        "last_tool_result": state.get("last_tool_result"),
        "stage_outputs": workpad.get("stage_outputs", {}),
        "review_feedback": review_feedback,
    }
    return json.dumps(payload, ensure_ascii=True, default=str)


def _solver_max_tokens(stage: str, profile: str) -> int:
    if stage in {"SYNTHESIZE", "REVISE"}:
        if profile == "legal_transactional":
            return 1400
        if profile == "finance_options":
            return 1200
    if stage == "COMPUTE":
        return 900
    return 700


def _safe_arithmetic_eval(expression: str) -> float | None:
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


def _extract_inline_assignments(text: str) -> dict[str, float]:
    assignments: dict[str, float] = {}
    for match in _ASSIGNMENT_RE.finditer(text or ""):
        key = match.group(1).strip()
        value = float(match.group(2))
        assignments[key] = value
        assignments[key.upper()] = value
    return assignments


def _deterministic_inline_quant_value(state: AgentState) -> float | None:
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))
    if template_id != "quant_inline_exact":
        return None

    task_text = latest_human_text(state.get("messages", []))
    evidence_pack = state.get("evidence_pack", {}) or {}
    formulas = list(evidence_pack.get("formulas", []))
    assignments = _extract_inline_assignments(task_text)
    if not assignments:
        return None

    candidates: list[str] = []
    for source in [*formulas, task_text]:
        for match in _FORMULA_RE.finditer(source or ""):
            rhs = match.group(2).strip()
            if re.fullmatch(r"-?\d+(?:\.\d+)?%?", rhs):
                continue
            if any(ch.isalpha() for ch in rhs) or any(op in rhs for op in "+-*/"):
                candidates.append(rhs)

    for expression in candidates:
        rewritten = expression.replace("^", "**")
        for name in sorted(assignments, key=len, reverse=True):
            rewritten = re.sub(rf"\b{re.escape(name)}\b", str(assignments[name]), rewritten)
        if re.search(r"[A-Za-z]", rewritten):
            continue
        value = _safe_arithmetic_eval(rewritten)
        if value is not None:
            return float(value)
    return None


def _format_scalar_number(value: float) -> str:
    return f"{float(value):.10g}"


def _record_event(workpad: dict[str, Any], node: str, action: str) -> dict[str, Any]:
    updated = dict(workpad)
    events = list(updated.get("events", []))
    events.append({"node": node, "action": action})
    updated["events"] = events
    return updated


def _merge_stage_output(workpad: dict[str, Any], stage: str, text: str) -> dict[str, Any]:
    updated = dict(workpad)
    outputs = dict(updated.get("stage_outputs", {}))
    outputs[stage] = text
    updated["stage_outputs"] = outputs
    return updated


def _first_ticker_entity(entities: list[Any]) -> str | None:
    for entity in entities or []:
        token = str(entity).strip().upper()
        if re.fullmatch(r"[A-Z]{1,5}", token):
            return token
    return None


def _infer_period_from_text(task_text: str) -> str:
    normalized = (task_text or "").lower()
    if any(token in normalized for token in ("1-month", "1 month", "1mo", "one month")):
        return "1mo"
    if any(token in normalized for token in ("3-month", "3 month", "3mo")):
        return "3mo"
    if any(token in normalized for token in ("6-month", "6 month", "6mo")):
        return "6mo"
    if any(token in normalized for token in ("1-year", "1 year", "12 month", "12-month", "1y")):
        return "1y"
    return "1mo"


def _reference_price_from_tool(tool_result: dict[str, Any]) -> float | None:
    assumptions = tool_result.get("assumptions", {}) if isinstance(tool_result, dict) else {}
    if isinstance(assumptions, dict):
        for key in ("reference_price", "spot", "spot_price", "S"):
            value = assumptions.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        legs = assumptions.get("legs")
        if isinstance(legs, list):
            for leg in legs:
                if isinstance(leg, dict) and isinstance(leg.get("S"), (int, float)):
                    return float(leg["S"])
    return None


def _latest_successful_tool_result(
    tool_results: list[dict[str, Any]],
    tool_names: set[str],
) -> dict[str, Any] | None:
    for result in reversed(tool_results):
        if not isinstance(result, dict):
            continue
        if str(result.get("type", "")) not in tool_names:
            continue
        if result.get("errors"):
            continue
        if isinstance(result.get("facts"), dict) and result.get("facts"):
            return result
    return None


def _scenario_args_from_primary_tool(tool_result: dict[str, Any]) -> dict[str, Any] | None:
    tool_type = str(tool_result.get("type", ""))
    facts = tool_result.get("facts", {}) if isinstance(tool_result, dict) else {}
    assumptions = tool_result.get("assumptions", {}) if isinstance(tool_result, dict) else {}

    if tool_type == "analyze_strategy":
        net_premium = facts.get("net_premium")
        total_delta = facts.get("total_delta")
        if not isinstance(net_premium, (int, float)) or not isinstance(total_delta, (int, float)):
            return None
        args = {
            "net_premium": float(net_premium),
            "total_delta": float(total_delta),
            "total_gamma": float(facts.get("total_gamma", 0.0) or 0.0),
            "total_theta_per_day": float(facts.get("total_theta_per_day", 0.0) or 0.0),
            "total_vega_per_vol_point": float(facts.get("total_vega_per_vol_point", 0.0) or 0.0),
        }
        reference_price = _reference_price_from_tool(tool_result)
        if isinstance(reference_price, (int, float)):
            args["reference_price"] = float(reference_price)
        return args

    if tool_type in {"black_scholes_price", "mispricing_analysis"}:
        option_type = str(assumptions.get("option_type", "call")).lower()
        premium = facts.get("market_price")
        if not isinstance(premium, (int, float)):
            if option_type == "put" and isinstance(facts.get("put_price"), (int, float)):
                premium = facts.get("put_price")
            elif isinstance(facts.get("call_price"), (int, float)):
                premium = facts.get("call_price")
            elif isinstance(facts.get("theoretical_price"), (int, float)):
                premium = facts.get("theoretical_price")
        delta = facts.get("delta")
        if not isinstance(premium, (int, float)) or not isinstance(delta, (int, float)):
            return None
        args = {
            "net_premium": float(premium),
            "total_delta": float(delta),
            "total_gamma": float(facts.get("gamma", 0.0) or 0.0),
            "total_theta_per_day": float(facts.get("theta", 0.0) or 0.0),
            "total_vega_per_vol_point": float(facts.get("vega", 0.0) or 0.0),
        }
        reference_price = _reference_price_from_tool(tool_result)
        if isinstance(reference_price, (int, float)):
            args["reference_price"] = float(reference_price)
        return args

    return None


def _deterministic_options_compute_summary(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    latest_risk_result = (workpad.get("risk_results") or [])[-1] if workpad.get("risk_results") else {}
    if str(latest_risk_result.get("verdict", "")) == "pass":
        return None

    scenario_result = state.get("last_tool_result") or {}
    if str(scenario_result.get("type", "")) != "scenario_pnl" or scenario_result.get("errors"):
        return None

    primary_tool = _latest_successful_tool_result(
        tool_results,
        {"analyze_strategy", "black_scholes_price", "option_greeks", "mispricing_analysis"},
    )
    primary_facts = primary_tool.get("facts", {}) if isinstance(primary_tool, dict) else {}
    scenario_facts = scenario_result.get("facts", {}) if isinstance(scenario_result, dict) else {}
    scenario_assumptions = scenario_result.get("assumptions", {}) if isinstance(scenario_result, dict) else {}

    max_loss = primary_facts.get("max_loss")
    delta = primary_facts.get("total_delta", primary_facts.get("delta"))
    gamma = primary_facts.get("total_gamma", primary_facts.get("gamma"))
    theta = primary_facts.get("total_theta_per_day", primary_facts.get("theta"))
    vega = primary_facts.get("total_vega_per_vol_point", primary_facts.get("vega"))
    worst_case_pnl = scenario_facts.get("worst_case_pnl")
    best_case_pnl = scenario_facts.get("best_case_pnl")

    summary_lines = [
        "Primary risk summary is now tool-backed and ready for review.",
    ]
    greeks_bits: list[str] = []
    if isinstance(delta, (int, float)):
        greeks_bits.append(f"delta {float(delta):.3f}")
    if isinstance(gamma, (int, float)):
        greeks_bits.append(f"gamma {float(gamma):.3f}")
    if isinstance(theta, (int, float)):
        greeks_bits.append(f"theta {float(theta):.3f}/day")
    if isinstance(vega, (int, float)):
        greeks_bits.append(f"vega {float(vega):.3f} per vol point")
    if greeks_bits:
        summary_lines.append("Key Greeks: " + ", ".join(greeks_bits) + ".")
    if isinstance(max_loss, (int, float)):
        summary_lines.append(f"Max loss is approximately {float(max_loss):.2f}.")
    if isinstance(worst_case_pnl, (int, float)):
        summary_lines.append(f"Downside scenario loss is approximately {float(worst_case_pnl):.2f}.")
    if isinstance(best_case_pnl, (int, float)):
        summary_lines.append(f"Base or favorable scenario P&L is approximately {float(best_case_pnl):.2f}.")

    summary_lines.append(
        "Risk controls: use 1-2% position sizing, place a stop-loss near a 1x premium loss or a breakeven breach, "
        "and hedge or reduce exposure if delta or gamma expands materially."
    )

    reference_price = scenario_assumptions.get("reference_price")
    if isinstance(reference_price, (int, float)):
        summary_lines.append(f"Reference spot for the scenario grid is {float(reference_price):.2f}.")

    return " ".join(summary_lines)


def _deterministic_quant_compute_summary(state: AgentState) -> str | None:
    value = _deterministic_inline_quant_value(state)
    if value is None:
        return None
    return f"Exact inline computation result: {_format_scalar_number(value)}"


def _deterministic_quant_final_answer(state: AgentState) -> str | None:
    value = _deterministic_inline_quant_value(state)
    if value is None:
        return None
    return _format_scalar_number(value)


def _table_positions_from_evidence(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    exposures: list[dict[str, Any]] = []
    for table in (evidence_pack.get("tables", []) or []):
        rows = table.get("rows", []) if isinstance(table, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = {str(key).strip().lower(): value for key, value in row.items()}
            weight_value = None
            for key in ("weight", "portfolio weight", "position weight", "% weight", "pct weight"):
                if key in normalized:
                    weight_value = normalized[key]
                    break
            if weight_value is None:
                continue
            try:
                if isinstance(weight_value, str) and weight_value.strip().endswith("%"):
                    weight = float(weight_value.strip().rstrip("%")) / 100.0
                else:
                    weight = float(weight_value)
                    if weight > 1.0:
                        weight /= 100.0
            except Exception:
                continue
            exposures.append(
                {
                    "ticker": str(normalized.get("ticker", normalized.get("name", normalized.get("asset", "unknown")))),
                    "name": str(normalized.get("name", normalized.get("ticker", normalized.get("asset", "unknown")))),
                    "sector": str(normalized.get("sector", normalized.get("industry", "unknown"))),
                    "weight": weight,
                    "avg_daily_volume_weight": float(normalized.get("adv_weight", normalized.get("avg daily volume weight", 0.2)) or 0.2),
                }
            )
    return exposures


def _portfolio_positions_from_evidence(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_facts = (evidence_pack or {}).get("prompt_facts", {}) or {}
    positions = prompt_facts.get("portfolio_positions")
    if isinstance(positions, list) and positions and all(isinstance(item, dict) for item in positions):
        return [dict(item) for item in positions]
    return _table_positions_from_evidence(evidence_pack or {})


def _returns_series_from_evidence(evidence_pack: dict[str, Any]) -> list[float]:
    prompt_facts = (evidence_pack or {}).get("prompt_facts", {}) or {}
    returns = prompt_facts.get("returns_series")
    if isinstance(returns, list):
        parsed: list[float] = []
        for item in returns:
            try:
                parsed.append(float(item))
            except Exception:
                continue
        if parsed:
            return parsed
    values: list[float] = []
    for table in (evidence_pack.get("tables", []) or []):
        rows = table.get("rows", []) if isinstance(table, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = {str(key).strip().lower(): value for key, value in row.items()}
            for key in ("return", "returns", "daily return", "monthly return"):
                if key not in normalized:
                    continue
                raw = normalized[key]
                try:
                    if isinstance(raw, str) and raw.strip().endswith("%"):
                        values.append(float(raw.strip().rstrip("%")) / 100.0)
                    else:
                        values.append(float(raw))
                except Exception:
                    pass
    return values


def _limit_constraints_from_evidence(evidence_pack: dict[str, Any], policy_context: dict[str, Any]) -> dict[str, Any]:
    prompt_facts = (evidence_pack or {}).get("prompt_facts", {}) or {}
    limits = prompt_facts.get("limit_constraints")
    if isinstance(limits, dict):
        return dict(limits)
    derived: dict[str, Any] = {}
    if isinstance(policy_context.get("max_position_risk_pct"), (int, float)):
        derived["max_loss_pct"] = float(policy_context["max_position_risk_pct"]) / 100.0
    return derived


def _portfolio_limit_metrics(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    var_result = _latest_successful_tool_result(tool_results, {"calculate_var"})
    if var_result:
        facts = var_result.get("facts", {})
        if isinstance(facts.get("var_decimal"), (int, float)):
            metrics["var_decimal"] = float(facts["var_decimal"])
        if isinstance(facts.get("var_amount"), (int, float)):
            metrics["var_amount"] = float(facts["var_amount"])
    drawdown_result = _latest_successful_tool_result(tool_results, {"drawdown_risk_profile"})
    if drawdown_result:
        facts = drawdown_result.get("facts", {})
        if isinstance(facts.get("max_drawdown_decimal"), (int, float)):
            metrics["worst_case_pnl_pct"] = -abs(float(facts["max_drawdown_decimal"]))
    factor_result = _latest_successful_tool_result(tool_results, {"factor_exposure_summary"})
    if factor_result:
        facts = factor_result.get("facts", {})
        if isinstance(facts.get("largest_factor_weight"), (int, float)):
            metrics["largest_factor_weight"] = float(facts["largest_factor_weight"])
    return metrics


def _deterministic_portfolio_compute_summary(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    findings: list[str] = []
    actions: list[str] = []

    concentration = _latest_successful_tool_result(tool_results, {"concentration_check"})
    factor = _latest_successful_tool_result(tool_results, {"factor_exposure_summary"})
    drawdown = _latest_successful_tool_result(tool_results, {"drawdown_risk_profile"})
    var_result = _latest_successful_tool_result(tool_results, {"calculate_var"})
    liquidity = _latest_successful_tool_result(tool_results, {"liquidity_stress"})
    limits = _latest_successful_tool_result(tool_results, {"portfolio_limit_check"})

    if concentration:
        facts = concentration.get("facts", {})
        if facts.get("name_breaches"):
            top = facts["name_breaches"][0]
            findings.append(f"Single-name concentration breach in {top.get('name')} at {float(top.get('weight', 0.0)) * 100:.1f}%.")
            actions.append("Trim the oversized name exposure toward the portfolio cap.")
        if facts.get("sector_breaches"):
            top = facts["sector_breaches"][0]
            findings.append(f"Sector concentration breach in {top.get('sector')} at {float(top.get('weight', 0.0)) * 100:.1f}%.")
            actions.append("Rebalance sector weight or offset it with lower-correlated positions.")

    if factor:
        facts = factor.get("facts", {})
        if facts.get("largest_factor"):
            findings.append(
                f"Largest factor exposure is {facts.get('largest_factor')} at about {float(facts.get('largest_factor_weight', 0.0)) * 100:.1f}%."
            )

    if drawdown:
        facts = drawdown.get("facts", {})
        if isinstance(facts.get("max_drawdown_decimal"), (int, float)):
            findings.append(f"Historical max drawdown is approximately {float(facts['max_drawdown_decimal']) * 100:.2f}%.")
            actions.append("Set rebalance thresholds and reduce risk if drawdown tolerance is exceeded.")

    if var_result:
        facts = var_result.get("facts", {})
        if isinstance(facts.get("var_amount"), (int, float)):
            findings.append(f"Estimated VaR is approximately {float(facts['var_amount']):.2f}.")

    if liquidity:
        facts = liquidity.get("facts", {})
        if str(facts.get("stress_assessment", "")) == "tight":
            findings.append("Liquidity stress is tight under the current redemption assumption.")
            actions.append("Stage reductions and avoid forced liquidation in the least-liquid positions.")

    if limits:
        facts = limits.get("facts", {})
        if facts.get("hard_limit_breached"):
            findings.append("At least one hard portfolio limit is breached.")
            actions.append("Do not increase risk until the limit breach is remediated.")

    if not findings:
        return None

    unique_actions = list(dict.fromkeys(actions or ["Maintain current sizing and continue monitoring key limits."]))
    summary = ["Portfolio risk review is now grounded in structured risk evidence."]
    summary.extend(findings[:4])
    summary.append("Recommended actions: " + " ".join(unique_actions[:3]))
    return " ".join(summary)


def _deterministic_portfolio_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    risk_results = list(workpad.get("risk_results", []))
    if not risk_results or str(risk_results[-1].get("verdict", "")) != "pass":
        return None

    tool_results = list(workpad.get("tool_results", []))
    concentration = _latest_successful_tool_result(tool_results, {"concentration_check"})
    factor = _latest_successful_tool_result(tool_results, {"factor_exposure_summary"})
    drawdown = _latest_successful_tool_result(tool_results, {"drawdown_risk_profile"})
    var_result = _latest_successful_tool_result(tool_results, {"calculate_var"})
    liquidity = _latest_successful_tool_result(tool_results, {"liquidity_stress"})
    limits = _latest_successful_tool_result(tool_results, {"portfolio_limit_check"})
    risk_requirements = workpad.get("risk_requirements") or {}
    timestamp = _best_available_timestamp(state)

    concentration_breach = bool(concentration and concentration.get("facts", {}).get("has_breach"))
    hard_limit_breach = bool(limits and limits.get("facts", {}).get("hard_limit_breached"))
    liquidity_tight = bool(liquidity and str(liquidity.get("facts", {}).get("stress_assessment", "")) == "tight")

    if hard_limit_breach:
        recommendation_line = (
            "Recommendation: reduce risk before adding any new exposure because the current evidence shows a hard-limit breach."
        )
    elif concentration_breach or liquidity_tight:
        recommendation_line = (
            "Recommendation: de-risk or rebalance the dominant exposures first, then reassess whether incremental risk is justified."
        )
    else:
        recommendation_line = (
            "Recommendation: keep the core allocation only if the current risk budget is acceptable, and use hedges or rebalancing triggers instead of discretionary timing."
        )

    lines = [
        "**Recommendation**",
        recommendation_line,
        "",
        "**Portfolio Risk Summary**",
    ]
    if timestamp:
        lines.append(f"Source timestamp: {timestamp}.")
    if factor:
        facts = factor.get("facts", {})
        lines.append(
            f"Largest factor exposure: {facts.get('largest_factor', 'n/a')} "
            f"at about {float(facts.get('largest_factor_weight', 0.0)) * 100:.1f}%."
        )
    if concentration:
        facts = concentration.get("facts", {})
        lines.append(
            f"Concentration check: {'breach detected' if facts.get('has_breach') else 'within stated limits on the available evidence'}."
        )

    lines.extend(["", "**Stress and Scenario Evidence**"])
    if var_result:
        facts = var_result.get("facts", {})
        if isinstance(facts.get("var_amount"), (int, float)):
            lines.append(f"Estimated VaR: {float(facts['var_amount']):.2f} ({float(facts.get('var_decimal', 0.0)) * 100:.2f}% of portfolio).")
    if drawdown:
        facts = drawdown.get("facts", {})
        if isinstance(facts.get("max_drawdown_decimal"), (int, float)):
            lines.append(f"Historical max drawdown: {float(facts['max_drawdown_decimal']) * 100:.2f}%.")
    if liquidity:
        facts = liquidity.get("facts", {})
        lines.append(f"Liquidity stress: {facts.get('stress_assessment', 'n/a')}.")

    lines.extend(["", "**Limit Status**"])
    if limits:
        facts = limits.get("facts", {})
        lines.append("Hard limit breached." if facts.get("hard_limit_breached") else "No hard limit breach detected from current limit checks.")
    else:
        lines.append("No explicit hard-limit check was available.")
    if risk_requirements.get("risk_findings"):
        for finding in list(dict.fromkeys(risk_requirements.get("risk_findings", [])))[:3]:
            lines.append(f"- {finding}")

    immediate_actions: list[str] = []
    hedge_actions: list[str] = []
    monitoring_actions: list[str] = []

    if concentration and concentration.get("facts", {}).get("has_breach"):
        immediate_actions.append("Trim the breached name or sector concentration toward policy limits.")
        hedge_actions.append("Offset the dominant concentration with lower-correlated exposure or a sector hedge while reductions are staged.")
    if liquidity and str(liquidity.get("facts", {}).get("stress_assessment", "")) == "tight":
        immediate_actions.append("Stage reductions in the least-liquid positions before adding new risk.")
        monitoring_actions.append("Track liquidation-days assumptions against actual trading capacity before any rebalance.")
    if drawdown:
        hedge_actions.append("Tie rebalance and hedge actions to drawdown or VaR thresholds rather than discretionary timing.")
        monitoring_actions.append("Escalate if drawdown or VaR trends move beyond the current tolerance band.")
    if factor and factor.get("facts", {}).get("largest_factor"):
        hedge_actions.append(
            f"Reduce or hedge the dominant factor exposure in {factor.get('facts', {}).get('largest_factor', 'the main factor bucket')} rather than relying only on name-level trims."
        )
    if not any([immediate_actions, hedge_actions, monitoring_actions]):
        monitoring_actions.append("Maintain current positioning but keep the main risk indicators on a tighter monitoring cadence.")

    lines.extend(["", "**Immediate Actions**"])
    for item in list(dict.fromkeys(immediate_actions)) or ["- No immediate hard-risk intervention is required on the current evidence."]:
        lines.append(item if item.startswith("-") else f"- {item}")

    lines.extend(["", "**Hedging / Rebalance Alternatives**"])
    for item in list(dict.fromkeys(hedge_actions)) or ["- No hedge or rebalance alternative is required beyond the current allocation discipline."]:
        lines.append(item if item.startswith("-") else f"- {item}")

    lines.extend(["", "**Monitoring Triggers**"])
    for item in list(dict.fromkeys(monitoring_actions)):
        lines.append(item if item.startswith("-") else f"- {item}")

    lines.extend(["", "**Disclosures**"])
    for item in risk_requirements.get("required_disclosures", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Recommendation class: {risk_requirements.get('recommendation_class', 'scenario_dependent_recommendation')}.")
    return "\n".join(lines)


def _fmt_pct(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.2f}%"
    return None


def _classify_research_view(
    revenue_growth: Any,
    operating_margin: Any,
    price_change: float | None,
    trailing_pe: Any,
    forward_pe: Any,
) -> tuple[str, str]:
    growth = float(revenue_growth) if isinstance(revenue_growth, (int, float)) else None
    margin = float(operating_margin) if isinstance(operating_margin, (int, float)) else None
    fpe = float(forward_pe) if isinstance(forward_pe, (int, float)) else None
    tpe = float(trailing_pe) if isinstance(trailing_pe, (int, float)) else None

    if (
        growth is not None
        and growth > 0.05
        and margin is not None
        and margin > 0.15
        and (price_change is None or price_change > -0.08)
    ):
        return (
            "constructive_but_valuation_sensitive",
            "Fundamentals support a constructive view, but valuation discipline still matters.",
        )
    if (price_change is not None and price_change < -0.10) or (fpe is not None and fpe > 32) or (tpe is not None and tpe > 35):
        return (
            "cautious_wait_for_better_entry",
            "The setup is investable only with caution because either valuation or recent price damage raises execution risk.",
        )
    return (
        "scenario_dependent_recommendation",
        "The setup is scenario-dependent and should be framed as a watchlist or conditional view rather than a hard call.",
    )


def _classify_event_view(price_change: float | None, has_actions: bool) -> tuple[str, str]:
    if price_change is not None and abs(price_change) >= 0.08:
        return (
            "heightened_event_risk",
            "The catalyst already sits in a higher-volatility regime, so execution should wait for confirmation rather than chase the move.",
        )
    if has_actions:
        return (
            "event_sensitive_watch",
            "There is enough catalyst context to keep an event-sensitive watch stance, but the recommendation should remain scenario-dependent until the event confirms direction.",
        )
    return (
        "scenario_dependent_recommendation",
        "The catalyst path is not strong enough for a directional call without further confirmation.",
    )


def _price_change_from_history(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    facts = result.get("facts", {}) if isinstance(result.get("facts", {}), dict) else {}
    start_close = facts.get("start_close")
    end_close = facts.get("end_close")
    if isinstance(start_close, (int, float)) and isinstance(end_close, (int, float)) and start_close:
        return (float(end_close) - float(start_close)) / float(start_close)
    return None


def _deterministic_research_compute_summary(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    fundamentals = _latest_successful_tool_result(tool_results, {"get_company_fundamentals"})
    history = _latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    if fundamentals is None and history is None:
        return None
    parts = ["Research evidence pack is now grounded in structured finance data."]
    if fundamentals:
        facts = fundamentals.get("facts", {}).get("fundamentals", {})
        revenue_growth = _fmt_pct(facts.get("revenueGrowth"))
        operating_margin = _fmt_pct(facts.get("operatingMargins"))
        trailing_pe = facts.get("trailingPE")
        forward_pe = facts.get("forwardPE")
        if revenue_growth:
            parts.append(f"Revenue growth is approximately {revenue_growth}.")
        if operating_margin:
            parts.append(f"Operating margin is approximately {operating_margin}.")
        if isinstance(trailing_pe, (int, float)):
            parts.append(f"Trailing P/E is about {float(trailing_pe):.2f}.")
        if isinstance(forward_pe, (int, float)):
            parts.append(f"Forward P/E is about {float(forward_pe):.2f}.")
    change = _price_change_from_history(history)
    if isinstance(change, float):
        parts.append(f"Observed price change over the retrieved window is {change * 100:.2f}%.")
    return " ".join(parts)


def _deterministic_research_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    fundamentals = _latest_successful_tool_result(tool_results, {"get_company_fundamentals"})
    history = _latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    if fundamentals is None and history is None:
        return None
    facts = fundamentals.get("facts", {}).get("fundamentals", {}) if fundamentals else {}
    timestamp = _best_available_timestamp(state)
    revenue_growth = _fmt_pct(facts.get("revenueGrowth"))
    operating_margin = _fmt_pct(facts.get("operatingMargins"))
    roe = _fmt_pct(facts.get("returnOnEquity"))
    trailing_pe = facts.get("trailingPE")
    forward_pe = facts.get("forwardPE")
    change = _price_change_from_history(history)
    recommendation_class, recommendation_line = _classify_research_view(
        facts.get("revenueGrowth"),
        facts.get("operatingMargins"),
        change,
        trailing_pe,
        forward_pe,
    )
    action_view = {
        "constructive_but_valuation_sensitive": "Action view: constructive on disciplined entries or pullbacks, not on blind momentum.",
        "cautious_wait_for_better_entry": "Action view: keep the name on a watchlist and wait for either valuation relief or cleaner operating evidence.",
    }.get(
        recommendation_class,
        "Action view: keep the recommendation conditional until evidence tightens on both operating trend and valuation support.",
    )
    thesis_points: list[str] = []
    if revenue_growth:
        thesis_points.append(f"Top-line growth is still supportive at {revenue_growth}.")
    if operating_margin:
        thesis_points.append(f"Margin profile remains informative at {operating_margin}, which is a key test of operating quality.")
    if isinstance(change, float):
        thesis_points.append(
            "Recent price action improves entry quality."
            if change < 0
            else "Recent price strength means entry discipline matters because sentiment may already reflect part of the good news."
        )
    if isinstance(trailing_pe, (int, float)) or isinstance(forward_pe, (int, float)):
        if isinstance(trailing_pe, (int, float)) and isinstance(forward_pe, (int, float)):
            thesis_points.append(
                "Valuation should be judged through both trailing and forward multiples, not through growth alone."
            )
        else:
            thesis_points.append("Valuation context is available, but it still needs peer framing before a high-conviction call.")
    if not thesis_points:
        thesis_points.append(
            "The name should be judged through the interaction of operating quality, valuation, and near-term market follow-through rather than through a single metric."
        )

    catalysts: list[str] = [
        "- Watch the next reporting cycle for confirmation on revenue durability, margin trend, and guidance quality.",
    ]
    if isinstance(forward_pe, (int, float)) and isinstance(trailing_pe, (int, float)):
        if float(forward_pe) > float(trailing_pe):
            catalysts.append("- Forward multiple expectations are still demanding, so watch for execution that actually earns that valuation.")
        else:
            catalysts.append("- Forward multiple sits below trailing, so watch whether earnings delivery can justify a better entry or a rerating.")
    else:
        catalysts.append("- Add peer or DCF work before treating the current valuation framing as complete.")

    change_view: list[str] = [
        "- Upgrade the view only if the next update preserves growth quality while keeping valuation or price damage from worsening.",
        "- Cut the view quickly if growth slows, margin quality slips, or management guidance weakens materially.",
    ]
    if isinstance(change, float) and change <= -0.1:
        change_view.insert(0, "- A stabilizing post-drawdown price base would improve entry quality more than a reflexive bounce.")

    lines = [
        "**Recommendation**",
        recommendation_line,
        f"- {action_view}",
        "",
        "**Thesis**",
    ]
    lines.extend([f"- {point}" for point in thesis_points])
    lines.extend([
        "",
        "**Evidence**",
    ])
    if revenue_growth:
        lines.append(f"- Revenue growth: {revenue_growth}.")
    if operating_margin:
        lines.append(f"- Operating margin: {operating_margin}.")
    if roe:
        lines.append(f"- Return on equity: {roe}.")
    if isinstance(change, float):
        lines.append(f"- Retrieved price performance over the evidence window: {change * 100:.2f}%.")
    if timestamp:
        lines.append(f"- Source timestamp: {timestamp}.")
    lines.extend(["", "**Valuation**"])
    if isinstance(trailing_pe, (int, float)) or isinstance(forward_pe, (int, float)):
        lines.append(
            f"- Multiples framing: trailing P/E {float(trailing_pe):.2f} and forward P/E {float(forward_pe):.2f}."
            if isinstance(trailing_pe, (int, float)) and isinstance(forward_pe, (int, float))
            else f"- Multiples framing: trailing P/E {float(trailing_pe):.2f}."
            if isinstance(trailing_pe, (int, float))
            else f"- Multiples framing: forward P/E {float(forward_pe):.2f}."
        )
    else:
        lines.append("- Valuation framing is limited on the current evidence and should be expanded with peer or DCF work.")
    lines.extend(
        [
            "",
            "**Catalysts and Watchpoints**",
        ]
    )
    lines.extend(catalysts)
    lines.extend(
        [
            "",
            "**What Would Change The View**",
        ]
    )
    lines.extend(change_view)
    lines.extend(
        [
            "",
            "**Risks**",
            "- Thesis risk increases if growth decelerates faster than margins can absorb.",
            "- Market multiple compression can dominate even if the fundamental trend remains stable.",
            "",
            "**Recommendation Class**",
            f"- {recommendation_class}.",
        ]
    )
    return "\n".join(lines)


def _deterministic_event_compute_summary(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    actions = _latest_successful_tool_result(tool_results, {"get_corporate_actions"})
    history = _latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    if actions is None and history is None:
        return None
    parts = ["Event-driven finance evidence is now grounded in structured catalyst and market context."]
    if actions:
        facts = actions.get("facts", {})
        dividend_count = len(facts.get("recent_dividends", []) or [])
        split_count = len(facts.get("recent_splits", []) or [])
        if dividend_count or split_count:
            parts.append(
                f"Retrieved catalyst context including {dividend_count} dividend records and {split_count} split records."
            )
    change = _price_change_from_history(history)
    if isinstance(change, float):
        parts.append(f"Observed market move over the evidence window is {change * 100:.2f}%.")
    return " ".join(parts)


def _deterministic_event_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    actions = _latest_successful_tool_result(tool_results, {"get_corporate_actions"})
    history = _latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    if actions is None and history is None:
        return None
    change = _price_change_from_history(history)
    timestamp = _best_available_timestamp(state)
    action_facts = actions.get("facts", {}) if isinstance(actions, dict) else {}
    dividend_count = len(action_facts.get("recent_dividends", []) or [])
    split_count = len(action_facts.get("recent_splits", []) or [])
    recommendation_class, action_view = _classify_event_view(change, bool(dividend_count or split_count))
    execution_discipline = [
        "- Keep sizing smaller than a normal thesis trade until the catalyst confirms direction and post-event liquidity is visible.",
        "- Do not treat pre-event price drift as confirmation; wait for the catalyst and the first orderly post-event reaction.",
    ]
    if isinstance(change, float) and abs(change) >= 0.08:
        execution_discipline.append(
            "- Because the name is already moving materially, avoid chasing the gap; require confirmation that volatility and spreads are normalizing."
        )

    lines = [
        "**Recommendation**",
        action_view,
        "",
        "**Catalyst**",
        "The setup is event-driven and should be evaluated through explicit catalyst scenarios rather than a static valuation-only lens.",
        "",
        "**Market Context**",
    ]
    if isinstance(change, float):
        lines.append(f"- Retrieved price move over the evidence window: {change * 100:.2f}%.")
    if timestamp:
        lines.append(f"- Source timestamp: {timestamp}.")
    if dividend_count or split_count:
        lines.append(f"- Corporate-action context retrieved: {dividend_count} dividend records and {split_count} split records.")
    lines.extend(
        [
            "",
            "**Scenarios**",
            "- Base case: the event lands near consensus and price reaction stays contained.",
            "- Upside case: the catalyst improves guidance or sentiment and supports upside follow-through.",
            "- Downside case: the event disappoints and reprices the name sharply lower.",
            "- Stress case: the catalyst coincides with a broader market or volatility shock.",
            "",
            "**Execution Discipline**",
        ]
    )
    lines.extend(execution_discipline)
    lines.extend(
        [
            "",
            "**What Would Change The View**",
            "- Upgrade the stance only if the event confirms direction and the post-event price action remains orderly.",
            "- Downgrade immediately if the catalyst triggers a gap move against the thesis or a broad volatility shock.",
            "",
            "**Risk**",
            "- Event timing, guidance uncertainty, and gap risk should be treated as the main risk drivers.",
            "",
            "**Recommendation Class**",
            f"- {recommendation_class}.",
        ]
    )
    return "\n".join(lines)


def _deterministic_actionable_finance_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    history = _latest_successful_tool_result(tool_results, {"get_price_history", "get_returns"})
    fundamentals = _latest_successful_tool_result(tool_results, {"get_company_fundamentals"})
    if history is None and fundamentals is None:
        return None
    timestamp = _best_available_timestamp(state)
    policy_context = (state.get("evidence_pack", {}) or {}).get("policy_context", {}) or {}
    price_change = _price_change_from_history(history)
    lines = [
        "**Recommendation**",
        "Recommendation is scenario-dependent on the current retrieved evidence and should not be treated as a blind high-conviction action.",
        "",
        "**Evidence**",
    ]
    if timestamp:
        lines.append(f"- Source timestamp: {timestamp}.")
    if isinstance(price_change, float):
        lines.append(f"- Observed price move over the retrieved window: {price_change * 100:.2f}%.")
    if fundamentals:
        facts = fundamentals.get("facts", {}).get("fundamentals", {})
        margin = _fmt_pct(facts.get("operatingMargins"))
        growth = _fmt_pct(facts.get("revenueGrowth"))
        if margin:
            lines.append(f"- Operating margin: {margin}.")
        if growth:
            lines.append(f"- Revenue growth: {growth}.")
    lines.extend(
        [
            "",
            "**Risk**",
            "- Action remains sensitive to fresh evidence, market regime, and headline risk.",
            "",
            "**Disclosures**",
            "- Recommendation class: scenario_dependent_recommendation.",
        ]
    )
    if policy_context.get("requires_recommendation_class"):
        lines.append("- Recommendation class is explicit because this is an actionable finance path.")
    return "\n".join(lines)


def _infer_options_strategy_label(primary_tool: dict[str, Any]) -> str:
    assumptions = primary_tool.get("assumptions", {}) if isinstance(primary_tool, dict) else {}
    legs = assumptions.get("legs")
    if isinstance(legs, list) and legs:
        short_calls: list[float] = []
        short_puts: list[float] = []
        long_calls = 0
        long_puts = 0
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            option_type = str(leg.get("option_type", "")).lower()
            action = str(leg.get("action", "")).lower()
            strike = leg.get("K")
            if action == "sell" and option_type == "call" and isinstance(strike, (int, float)):
                short_calls.append(float(strike))
            elif action == "sell" and option_type == "put" and isinstance(strike, (int, float)):
                short_puts.append(float(strike))
            elif action == "buy" and option_type == "call":
                long_calls += 1
            elif action == "buy" and option_type == "put":
                long_puts += 1
        if short_calls and short_puts and long_calls and long_puts:
            return "iron condor"
        if short_calls and short_puts:
            if len(short_calls) == 1 and len(short_puts) == 1 and abs(short_calls[0] - short_puts[0]) < 1e-9:
                return "short straddle"
            return "short strangle"
    if str(primary_tool.get("type", "")) == "black_scholes_price":
        return "single-option premium sale"
    return "options premium strategy"


def _infer_breakeven_text(primary_tool: dict[str, Any]) -> str:
    facts = primary_tool.get("facts", {}) if isinstance(primary_tool, dict) else {}
    assumptions = primary_tool.get("assumptions", {}) if isinstance(primary_tool, dict) else {}
    if isinstance(facts.get("breakeven"), (int, float)):
        return f"{float(facts['breakeven']):.2f}"
    net_premium = facts.get("net_premium")
    legs = assumptions.get("legs")
    if isinstance(net_premium, (int, float)) and isinstance(legs, list):
        short_calls: list[float] = []
        short_puts: list[float] = []
        for leg in legs:
            if not isinstance(leg, dict) or str(leg.get("action", "")).lower() != "sell":
                continue
            strike = leg.get("K")
            if not isinstance(strike, (int, float)):
                continue
            option_type = str(leg.get("option_type", "")).lower()
            if option_type == "call":
                short_calls.append(float(strike))
            elif option_type == "put":
                short_puts.append(float(strike))
        if short_calls and short_puts:
            lower = min(short_puts) - float(net_premium)
            upper = max(short_calls) + float(net_premium)
            return f"{lower:.2f} / {upper:.2f}"
    return "manage around the short strikes adjusted by collected premium"


def _primary_tool_is_policy_compliant(primary_tool: dict[str, Any], policy_context: dict[str, Any]) -> bool:
    if not primary_tool:
        return False
    strategy_label = _infer_options_strategy_label(primary_tool)
    assumptions = primary_tool.get("assumptions", {}) if isinstance(primary_tool, dict) else {}
    legs = assumptions.get("legs")
    if policy_context.get("defined_risk_only"):
        if strategy_label == "iron condor":
            pass
        elif isinstance(legs, list):
            has_buy = any(isinstance(leg, dict) and str(leg.get("action", "")).lower() == "buy" for leg in legs)
            has_sell = any(isinstance(leg, dict) and str(leg.get("action", "")).lower() == "sell" for leg in legs)
            if not (has_buy and has_sell):
                return False
        else:
            return False
    if policy_context.get("no_naked_options"):
        normalized_label = strategy_label.lower()
        if normalized_label in {"short straddle", "short strangle", "single-option premium sale"}:
            return False
        if isinstance(legs, list):
            has_long_call = any(
                isinstance(leg, dict)
                and str(leg.get("action", "")).lower() == "buy"
                and str(leg.get("option_type", "")).lower() == "call"
                for leg in legs
            )
            has_long_put = any(
                isinstance(leg, dict)
                and str(leg.get("action", "")).lower() == "buy"
                and str(leg.get("option_type", "")).lower() == "put"
                for leg in legs
            )
            short_calls = any(
                isinstance(leg, dict)
                and str(leg.get("action", "")).lower() == "sell"
                and str(leg.get("option_type", "")).lower() == "call"
                for leg in legs
            )
            short_puts = any(
                isinstance(leg, dict)
                and str(leg.get("action", "")).lower() == "sell"
                and str(leg.get("option_type", "")).lower() == "put"
                for leg in legs
            )
            if short_calls and not has_long_call:
                return False
            if short_puts and not has_long_put:
                return False
    return True


def _deterministic_policy_options_tool_call(state: AgentState) -> dict[str, Any] | None:
    evidence_pack = state.get("evidence_pack", {}) or {}
    prompt_facts = evidence_pack.get("prompt_facts", {}) or {}
    derived_facts = evidence_pack.get("derived_facts", {}) or {}
    policy_context = evidence_pack.get("policy_context", {}) or {}
    if not (policy_context.get("defined_risk_only") or policy_context.get("no_naked_options")):
        return None

    spot = 300.0
    for candidate in (
        prompt_facts.get("spot"),
        prompt_facts.get("spot_price"),
        prompt_facts.get("reference_price"),
    ):
        if isinstance(candidate, (int, float)):
            spot = float(candidate)
            break
    sigma = float(prompt_facts.get("implied_volatility", 0.35) or 0.35)
    r = float(prompt_facts.get("risk_free_rate", 0.05) or 0.05)
    t_days = int(prompt_facts.get("days_to_expiry", 30) or 30)
    width = max(5.0, round(spot * 0.03 / 5.0) * 5.0)
    inner = max(5.0, round(spot * 0.015 / 5.0) * 5.0)
    vol_bias = str(derived_facts.get("vol_bias", "short_vol"))

    if vol_bias == "short_vol":
        legs = [
            {"option_type": "put", "action": "buy", "S": spot, "K": spot - inner - width, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "put", "action": "sell", "S": spot, "K": spot - inner, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "call", "action": "sell", "S": spot, "K": spot + inner, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "call", "action": "buy", "S": spot, "K": spot + inner + width, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
        ]
    else:
        legs = [
            {"option_type": "call", "action": "buy", "S": spot, "K": spot, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "call", "action": "sell", "S": spot, "K": spot + width, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
        ]
    return {"name": "analyze_strategy", "arguments": {"legs": legs}}


def _deterministic_standard_options_tool_call(state: AgentState) -> dict[str, Any] | None:
    evidence_pack = state.get("evidence_pack", {}) or {}
    prompt_facts = evidence_pack.get("prompt_facts", {}) or {}
    derived_facts = evidence_pack.get("derived_facts", {}) or {}

    spot = 300.0
    for candidate in (
        prompt_facts.get("spot"),
        prompt_facts.get("spot_price"),
        prompt_facts.get("reference_price"),
    ):
        if isinstance(candidate, (int, float)):
            spot = float(candidate)
            break

    sigma = float(prompt_facts.get("implied_volatility", 0.35) or 0.35)
    r = float(prompt_facts.get("risk_free_rate", 0.05) or 0.05)
    t_days = int(prompt_facts.get("days_to_expiry", 30) or 30)
    vol_bias = str(derived_facts.get("vol_bias", "short_vol"))
    atm = round(spot / 5.0) * 5.0

    if vol_bias == "short_vol":
        legs = [
            {"option_type": "call", "action": "sell", "S": spot, "K": atm, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "put", "action": "sell", "S": spot, "K": atm, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
        ]
    else:
        legs = [
            {"option_type": "call", "action": "buy", "S": spot, "K": atm, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
            {"option_type": "put", "action": "buy", "S": spot, "K": atm, "T_days": t_days, "r": r, "sigma": sigma, "contracts": 1},
        ]
    return {"name": "analyze_strategy", "arguments": {"legs": legs}}


def _deterministic_options_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    risk_results = list(workpad.get("risk_results", []))
    if not risk_results or str(risk_results[-1].get("verdict", "")) != "pass":
        return None

    primary_tool = _latest_successful_tool_result(
        list(workpad.get("tool_results", [])),
        {"analyze_strategy", "black_scholes_price", "option_greeks", "mispricing_analysis"},
    )
    scenario_result = _latest_successful_tool_result(
        list(workpad.get("tool_results", [])),
        {"scenario_pnl", "run_stress_test", "calculate_var", "portfolio_limit_check", "concentration_check", "calculate_portfolio_greeks"},
    )
    if primary_tool is None or scenario_result is None:
        return None

    primary_facts = primary_tool.get("facts", {}) if isinstance(primary_tool, dict) else {}
    primary_assumptions = primary_tool.get("assumptions", {}) if isinstance(primary_tool, dict) else {}
    scenario_facts = scenario_result.get("facts", {}) if isinstance(scenario_result, dict) else {}
    derived_facts = (state.get("evidence_pack", {}) or {}).get("derived_facts", {}) or {}
    risk_requirements = workpad.get("risk_requirements") or {}
    strategy_label = _infer_options_strategy_label(primary_tool)
    recommendation = "net seller of options" if derived_facts.get("vol_bias") == "short_vol" else "options seller with scenario-dependent conviction"
    net_premium = primary_facts.get("net_premium")
    premium_direction = str(primary_facts.get("premium_direction", "credit")).upper()
    delta = primary_facts.get("total_delta", primary_facts.get("delta"))
    gamma = primary_facts.get("total_gamma", primary_facts.get("gamma"))
    theta = primary_facts.get("total_theta_per_day", primary_facts.get("theta"))
    vega = primary_facts.get("total_vega_per_vol_point", primary_facts.get("vega"))
    max_loss = primary_facts.get("max_loss")
    breakevens = _infer_breakeven_text(primary_tool)
    worst_case_pnl = scenario_facts.get("worst_case_pnl")
    best_case_pnl = scenario_facts.get("best_case_pnl")
    reference_price = _reference_price_from_tool(primary_tool)

    alternative = "iron condor with defined wings" if strategy_label in {"short straddle", "short strangle"} else "defined-risk spread"
    tradeoff = "lower premium but better tail-risk control" if alternative == "iron condor with defined wings" else "more controlled downside at the cost of smaller carry"

    disclosures: list[str] = []
    for item in risk_requirements.get("required_disclosures", []):
        normalized = str(item).lower()
        if "short-volatility" in normalized or "volatility-spike" in normalized:
            disclosures.append("Short-volatility / volatility-spike risk is material: losses can accelerate if implied volatility expands.")
        elif "tail loss" in normalized or "gap risk" in normalized or "unbounded" in normalized:
            disclosures.append("Tail loss and gap risk are material, especially if the underlying gaps through the short strikes.")
        elif "downside scenario loss" in normalized:
            if isinstance(worst_case_pnl, (int, float)):
                disclosures.append(f"Downside scenario loss is approximately {float(worst_case_pnl):.2f}; the exit / sizing response is to keep exposure at 1-2% of capital and cut risk on a breach.")
            else:
                disclosures.append("Downside scenario loss should be treated as the hard sizing and exit reference.")
    if not disclosures and isinstance(worst_case_pnl, (int, float)):
        disclosures.append(f"Downside scenario loss is approximately {float(worst_case_pnl):.2f}.")

    assumption_lines: list[str] = []
    for record in state.get("assumption_ledger", []):
        if not isinstance(record, dict) or not record.get("requires_user_visible_disclosure"):
            continue
        assumption_lines.append(str(record.get("assumption", "")))

    lines = [
        "**Recommendation**",
        f"Be a {recommendation}.",
        "",
        "**Primary Strategy**",
        f"{strategy_label.title()} with {premium_direction.lower()} premium"
        + (f" of {float(net_premium):.2f}." if isinstance(net_premium, (int, float)) else "."),
        "",
        "**Alternative Strategy Comparison**",
        f"{alternative.title()} is the cleaner alternative when you want {tradeoff}.",
        "",
        "**Key Greeks and Breakevens**",
    ]
    greeks_line = []
    if isinstance(delta, (int, float)):
        greeks_line.append(f"delta {float(delta):.3f}")
    if isinstance(gamma, (int, float)):
        greeks_line.append(f"gamma {float(gamma):.3f}")
    if isinstance(theta, (int, float)):
        greeks_line.append(f"theta {float(theta):.3f}/day")
    if isinstance(vega, (int, float)):
        greeks_line.append(f"vega {float(vega):.3f} per vol point")
    if greeks_line:
        lines.append(", ".join(greeks_line) + ".")
    lines.append(f"Breakevens: {breakevens}.")
    if isinstance(max_loss, (int, float)):
        lines.append(f"Max loss reference: {float(max_loss):.2f}.")
    lines.extend(
        [
            "",
            "**Risk Management**",
            "Use 1-2% position sizing, predefine a stop-loss at a breakeven breach or roughly a 1x premium loss, and hedge or reduce exposure if delta/gamma expands.",
        ]
    )
    if isinstance(best_case_pnl, (int, float)) or isinstance(worst_case_pnl, (int, float)):
        parts = []
        if isinstance(best_case_pnl, (int, float)):
            parts.append(f"base-case P&L about {float(best_case_pnl):.2f}")
        if isinstance(worst_case_pnl, (int, float)):
            parts.append(f"stress downside about {float(worst_case_pnl):.2f}")
        lines.append("Scenario summary: " + "; ".join(parts) + ".")
    lines.extend(["", "**Disclosures**"])
    if assumption_lines:
        for item in assumption_lines[:3]:
            lines.append(f"- Assumption: {item}")
    elif isinstance(reference_price, (int, float)):
        lines.append(f"- Assumption: spot/reference price treated as {float(reference_price):.2f}.")
    for item in disclosures:
        lines.append(f"- {item}")
    if risk_requirements.get("recommendation_class"):
        lines.append("")
        lines.append(f"Recommendation class: {risk_requirements.get('recommendation_class')}.")

    return "\n".join(lines)


def _deterministic_policy_options_final_answer(state: AgentState) -> str | None:
    workpad = state.get("workpad", {}) or {}
    risk_results = list(workpad.get("risk_results", []))
    if not risk_results or str(risk_results[-1].get("verdict", "")) != "pass":
        return None

    policy_context = (state.get("evidence_pack", {}) or {}).get("policy_context", {}) or {}
    primary_tool = _latest_successful_tool_result(
        list(workpad.get("tool_results", [])),
        {"analyze_strategy", "black_scholes_price", "option_greeks", "mispricing_analysis"},
    )
    scenario_result = _latest_successful_tool_result(
        list(workpad.get("tool_results", [])),
        {"scenario_pnl", "run_stress_test", "calculate_var", "portfolio_limit_check", "concentration_check", "calculate_portfolio_greeks"},
    )
    if primary_tool is None or scenario_result is None:
        return None
    if not _primary_tool_is_policy_compliant(primary_tool, policy_context):
        return None

    primary_facts = primary_tool.get("facts", {}) if isinstance(primary_tool, dict) else {}
    scenario_facts = scenario_result.get("facts", {}) if isinstance(scenario_result, dict) else {}
    risk_requirements = workpad.get("risk_requirements") or {}
    strategy_label = _infer_options_strategy_label(primary_tool)
    short_vol = str((state.get("evidence_pack", {}) or {}).get("derived_facts", {}).get("vol_bias", "")) == "short_vol"
    recommendation = (
        "Be a net seller of options through a defined-risk structure"
        if short_vol
        else "Be a net buyer of options through a defined-risk spread"
    )
    alternative = "put credit spread" if strategy_label == "iron condor" else "iron condor"
    tradeoff = "retains short-vol carry with simpler one-sided management" if alternative == "put credit spread" else "diversifies tail exposure across both sides at the cost of more moving parts"
    net_premium = primary_facts.get("net_premium")
    premium_direction = str(primary_facts.get("premium_direction", "credit")).upper()
    delta = primary_facts.get("total_delta", primary_facts.get("delta"))
    gamma = primary_facts.get("total_gamma", primary_facts.get("gamma"))
    theta = primary_facts.get("total_theta_per_day", primary_facts.get("theta"))
    vega = primary_facts.get("total_vega_per_vol_point", primary_facts.get("vega"))
    breakevens = _infer_breakeven_text(primary_tool)
    worst_case_pnl = scenario_facts.get("worst_case_pnl")
    best_case_pnl = scenario_facts.get("best_case_pnl")
    risk_cap = policy_context.get("max_position_risk_pct")
    disclosures: list[str] = []
    for item in risk_requirements.get("required_disclosures", []):
        normalized = str(item).lower()
        if "short-volatility" in normalized or "volatility-spike" in normalized:
            disclosures.append("Short-volatility / volatility-spike risk is still material even with defined wings.")
        elif "tail loss" in normalized or "gap risk" in normalized or "unbounded" in normalized:
            disclosures.append("Tail loss is capped by the long wings, but gap risk into the short strikes still matters.")
        elif "downside scenario loss" in normalized:
            if isinstance(worst_case_pnl, (int, float)):
                disclosures.append(f"Stress downside scenario loss is approximately {float(worst_case_pnl):.2f}, which defines the sizing and exit response.")
        elif "max loss" in normalized:
            max_loss = primary_facts.get("max_loss")
            if isinstance(max_loss, (int, float)):
                disclosures.append(f"Max loss reference is approximately {float(max_loss):.2f}.")

    lines = [
        "**Recommendation**",
        f"{recommendation}. This mandate requires defined-risk only and prohibits naked options.",
        "",
        "**Primary Strategy**",
        f"Defined-risk {strategy_label} with {premium_direction.lower()} premium"
        + (f" of {float(net_premium):.2f}." if isinstance(net_premium, (int, float)) else "."),
        "",
        "**Alternative Strategy Comparison**",
        f"{alternative.title()} is the cleaner backup when you want {tradeoff}.",
        "",
        "**Key Greeks and Breakevens**",
    ]
    greeks_line = []
    if isinstance(delta, (int, float)):
        greeks_line.append(f"delta {float(delta):.3f}")
    if isinstance(gamma, (int, float)):
        greeks_line.append(f"gamma {float(gamma):.3f}")
    if isinstance(theta, (int, float)):
        greeks_line.append(f"theta {float(theta):.3f}/day")
    if isinstance(vega, (int, float)):
        greeks_line.append(f"vega {float(vega):.3f} per vol point")
    if greeks_line:
        lines.append(", ".join(greeks_line) + ".")
    lines.append(f"Breakevens: {breakevens}.")
    lines.extend(
        [
            "",
            "**Risk Management**",
            (
                f"Cap position risk at about {float(risk_cap):g}% of capital, use defined exit points near the short strikes "
                "or at roughly a 1x premium-loss threshold, and reduce exposure if gamma or vol expands sharply."
            )
            if isinstance(risk_cap, (int, float))
            else "Use defined exit points near the short strikes or at roughly a 1x premium-loss threshold, and reduce exposure if gamma or vol expands sharply."
        ]
    )
    if isinstance(best_case_pnl, (int, float)) or isinstance(worst_case_pnl, (int, float)):
        parts = []
        if isinstance(best_case_pnl, (int, float)):
            parts.append(f"base-case P&L about {float(best_case_pnl):.2f}")
        if isinstance(worst_case_pnl, (int, float)):
            parts.append(f"stress downside about {float(worst_case_pnl):.2f}")
        lines.append("Scenario summary: " + "; ".join(parts) + ".")
    lines.extend(["", "**Disclosures**"])
    lines.append("- Mandate: defined-risk only; naked options are not permitted in this account.")
    if isinstance(risk_cap, (int, float)):
        lines.append(f"- Position-risk cap: keep exposure around {float(risk_cap):g}% of capital or lower.")
    for item in disclosures:
        lines.append(f"- {item}")
    if isinstance(worst_case_pnl, (int, float)) and not disclosures:
        lines.append(f"- Stress downside scenario loss is approximately {float(worst_case_pnl):.2f}.")
    for record in state.get("assumption_ledger", []):
        if isinstance(record, dict) and record.get("requires_user_visible_disclosure"):
            lines.append(f"- Assumption: {record.get('assumption', '')}")
    lines.append("")
    lines.append(f"Recommendation class: {risk_requirements.get('recommendation_class', 'scenario_dependent_recommendation')}.")
    return "\n".join(lines)


def _best_available_timestamp(state: AgentState) -> str | None:
    tool_results = list((state.get("workpad") or {}).get("tool_results", []))
    for result in reversed(tool_results):
        if not isinstance(result, dict) or result.get("errors"):
            continue
        source = result.get("source", {}) if isinstance(result.get("source", {}), dict) else {}
        facts = result.get("facts", {}) if isinstance(result.get("facts", {}), dict) else {}
        assumptions = result.get("assumptions", {}) if isinstance(result.get("assumptions", {}), dict) else {}
        for candidate in (
            source.get("timestamp"),
            source.get("as_of_date"),
            facts.get("as_of_date"),
            facts.get("window_end"),
            assumptions.get("as_of_date"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _apply_compliance_final_fixes(
    text: str,
    *,
    state: AgentState,
    compliance_feedback: dict[str, Any],
    policy_context: dict[str, Any],
    risk_requirements: dict[str, Any],
) -> str:
    content = (text or "").strip()
    if not content:
        return content

    normalized = re.sub(r"\s+", " ", content.lower()).strip()
    append_lines: list[str] = []
    required_disclosures = list(compliance_feedback.get("required_disclosures", []))

    if (
        (policy_context.get("requires_recommendation_class") or any("recommendation class" in str(item).lower() for item in required_disclosures))
        and "recommendation class" not in normalized
    ):
        append_lines.append(
            f"Recommendation class: {risk_requirements.get('recommendation_class', 'scenario_dependent_recommendation')}."
        )

    if (
        (policy_context.get("requires_timestamped_evidence") or any("timestamp" in str(item).lower() for item in required_disclosures))
        and "timestamp" not in normalized
        and "as of" not in normalized
    ):
        timestamp = _best_available_timestamp(state)
        if timestamp:
            append_lines.append(f"Source timestamp: {timestamp}.")

    if not append_lines:
        return content

    separator = "\n\n" if "\n" in content else " "
    return content + separator + "\n".join(append_lines)


def _build_tool_call_message(tool_name: str, tool_args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": tool_args,
                "id": f"call_{uuid.uuid4().hex[:10]}",
                "type": "tool_call",
            }
        ],
    )


def _deterministic_finance_tool_call(
    state: AgentState,
    *,
    profile: str,
    effective_stage: str,
    allowed_tool_names: set[str],
    risk_feedback: dict[str, Any],
) -> dict[str, Any] | None:
    task_text = latest_human_text(state["messages"])
    normalized = task_text.lower()
    evidence_pack = state.get("evidence_pack", {}) or {}
    prompt_facts = evidence_pack.get("prompt_facts", {}) or {}
    last_tool_result = state.get("last_tool_result") or {}
    workpad = state.get("workpad", {}) or {}
    tool_results = list(workpad.get("tool_results", []))
    template_id = str((state.get("execution_template") or {}).get("template_id", ""))

    if profile == "finance_quant" and effective_stage == "GATHER" and "needs_live_data" in set(state.get("capability_flags", [])):
        if (
            str(last_tool_result.get("type", "")) == "get_price_history"
            and "return" in normalized
            and "pct_change" in allowed_tool_names
            and not any(str(result.get("type", "")) == "pct_change" for result in tool_results if isinstance(result, dict))
        ):
            facts = last_tool_result.get("facts", {}) if isinstance(last_tool_result, dict) else {}
            start_close = facts.get("start_close")
            end_close = facts.get("end_close")
            if isinstance(start_close, (int, float)) and isinstance(end_close, (int, float)):
                return {
                    "name": "pct_change",
                    "arguments": {"old_value": float(start_close), "new_value": float(end_close)},
                }

        if not last_tool_result and "get_price_history" in allowed_tool_names and any(
            token in normalized for token in ("price history", "historical prices", "return", "monthly return", "1-month")
        ):
            ticker = _first_ticker_entity(evidence_pack.get("entities", []))
            if ticker:
                args = {"ticker": ticker, "period": _infer_period_from_text(task_text)}
                as_of_date = prompt_facts.get("as_of_date")
                if isinstance(as_of_date, str) and as_of_date:
                    args["as_of_date"] = as_of_date
                return {"name": "get_price_history", "arguments": args}

    if (
        profile == "finance_options"
        and effective_stage == "COMPUTE"
        and "analyze_strategy" in allowed_tool_names
        and not state.get("last_tool_result")
        and not state.get("review_feedback")
        and not risk_feedback
    ):
        policy_call = _deterministic_policy_options_tool_call(state)
        if policy_call:
            return policy_call
        return _deterministic_standard_options_tool_call(state)

    if (
        profile == "finance_options"
        and effective_stage == "COMPUTE"
        and "scenario_pnl" in allowed_tool_names
        and "MISSING_SCENARIO_ANALYSIS" in set(risk_feedback.get("violation_codes", []))
    ):
        args = _scenario_args_from_primary_tool(last_tool_result)
        if args:
            return {"name": "scenario_pnl", "arguments": args}

    if template_id == "equity_research_report" and effective_stage == "GATHER":
        ticker = _first_ticker_entity(evidence_pack.get("entities", []))
        as_of_date = prompt_facts.get("as_of_date")
        tool_types = {str(result.get("type", "")) for result in tool_results if isinstance(result, dict)}
        if ticker and "get_company_fundamentals" in allowed_tool_names and "get_company_fundamentals" not in tool_types:
            args = {"ticker": ticker}
            if isinstance(as_of_date, str) and as_of_date:
                args["as_of_date"] = as_of_date
            return {"name": "get_company_fundamentals", "arguments": args}
        if ticker and "get_price_history" in allowed_tool_names and "get_price_history" not in tool_types:
            args = {"ticker": ticker, "period": "6mo"}
            if isinstance(as_of_date, str) and as_of_date:
                args["as_of_date"] = as_of_date
            return {"name": "get_price_history", "arguments": args}

    if template_id == "event_driven_finance" and effective_stage == "GATHER":
        ticker = _first_ticker_entity(evidence_pack.get("entities", []))
        as_of_date = prompt_facts.get("as_of_date")
        tool_types = {str(result.get("type", "")) for result in tool_results if isinstance(result, dict)}
        if ticker and "get_corporate_actions" in allowed_tool_names and "get_corporate_actions" not in tool_types:
            args = {"ticker": ticker}
            if isinstance(as_of_date, str) and as_of_date:
                args["as_of_date"] = as_of_date
            return {"name": "get_corporate_actions", "arguments": args}
        if ticker and "get_price_history" in allowed_tool_names and "get_price_history" not in tool_types:
            args = {"ticker": ticker, "period": "3mo"}
            if isinstance(as_of_date, str) and as_of_date:
                args["as_of_date"] = as_of_date
            return {"name": "get_price_history", "arguments": args}

    if template_id == "regulated_actionable_finance" and effective_stage == "GATHER":
        ticker = _first_ticker_entity(evidence_pack.get("entities", []))
        as_of_date = prompt_facts.get("as_of_date")
        tool_types = {str(result.get("type", "")) for result in tool_results if isinstance(result, dict)}
        if ticker and "get_company_fundamentals" in allowed_tool_names and "get_company_fundamentals" not in tool_types:
            args = {"ticker": ticker}
            if isinstance(as_of_date, str) and as_of_date:
                args["as_of_date"] = as_of_date
            return {"name": "get_company_fundamentals", "arguments": args}
        if ticker and "get_price_history" in allowed_tool_names and "get_price_history" not in tool_types:
            args = {"ticker": ticker, "period": _infer_period_from_text(task_text)}
            if isinstance(as_of_date, str) and as_of_date:
                args["as_of_date"] = as_of_date
            return {"name": "get_price_history", "arguments": args}

    if template_id == "portfolio_risk_review" and effective_stage == "COMPUTE":
        positions = _portfolio_positions_from_evidence(evidence_pack)
        returns_series = _returns_series_from_evidence(evidence_pack)
        limits = _limit_constraints_from_evidence(
            evidence_pack,
            (evidence_pack.get("policy_context", {}) if isinstance(evidence_pack.get("policy_context", {}), dict) else {}),
        )
        tool_types = {str(result.get("type", "")) for result in tool_results if isinstance(result, dict)}
        if positions and "concentration_check" in allowed_tool_names and "concentration_check" not in tool_types:
            return {"name": "concentration_check", "arguments": {"exposures": positions}}
        if positions and "factor_exposure_summary" in allowed_tool_names and "factor_exposure_summary" not in tool_types:
            factor_map = prompt_facts.get("factor_map")
            args = {"exposures": positions}
            if isinstance(factor_map, dict) and factor_map:
                args["factor_map"] = factor_map
            return {"name": "factor_exposure_summary", "arguments": args}
        if returns_series and "drawdown_risk_profile" in allowed_tool_names and "drawdown_risk_profile" not in tool_types:
            return {"name": "drawdown_risk_profile", "arguments": {"returns": returns_series}}
        if returns_series and "calculate_var" in allowed_tool_names and "calculate_var" not in tool_types:
            daily_vol = 0.0
            if len(returns_series) >= 2:
                mean_r = sum(returns_series) / len(returns_series)
                variance = sum((item - mean_r) ** 2 for item in returns_series) / max(len(returns_series) - 1, 1)
                daily_vol = max(variance, 0.0) ** 0.5
            portfolio_value = float(prompt_facts.get("portfolio_value", 1_000_000.0) or 1_000_000.0)
            return {
                "name": "calculate_var",
                "arguments": {"portfolio_value": portfolio_value, "daily_vol": daily_vol, "confidence_level": 0.95},
            }
        if positions and "liquidity_stress" in allowed_tool_names and "liquidity_stress" not in tool_types:
            redemption_pct = float(prompt_facts.get("redemption_pct", 0.10) or 0.10)
            return {"name": "liquidity_stress", "arguments": {"positions": positions, "redemption_pct": redemption_pct}}
        if limits and "portfolio_limit_check" in allowed_tool_names and "portfolio_limit_check" not in tool_types:
            metrics = _portfolio_limit_metrics(tool_results)
            if metrics:
                return {"name": "portfolio_limit_check", "arguments": {"metrics": metrics, "limits": limits}}

    return None


def route_from_solver(state: AgentState) -> str:
    if state.get("pending_tool_call"):
        return "tool_runner"
    workpad = state.get("workpad", {})
    if workpad.get("review_ready"):
        if requires_risk_control(state):
            return "risk_controller"
        if requires_compliance_guard(state):
            return "compliance_guard"
        return "reviewer"
    if state.get("solver_stage") == "COMPLETE":
        if state.get("answer_contract", {}).get("requires_adapter"):
            return "output_adapter"
        return "reflect"
    return "solver"


def make_solver(tools: list):
    use_prompt_tools = _tool_call_mode("executor") == "prompt"

    def solver(state: AgentState) -> dict:
        step = increment_runtime_step()
        tracker: CostTracker = state.get("cost_tracker")
        profile = state.get("task_profile", "general")
        stage = state.get("solver_stage", "SYNTHESIZE")
        workpad = dict(state.get("workpad", {}))
        review_feedback = state.get("review_feedback") or {}
        risk_feedback = state.get("risk_feedback") or {}
        compliance_feedback = state.get("compliance_feedback") or {}
        profile_pack = workpad.get("profile_pack") or get_profile_pack(profile).model_dump()
        execution_template = state.get("execution_template", {}) or workpad.get("execution_template", {})
        allowed_stages = set(execution_template.get("allowed_stages", []))

        if stage == "PLAN":
            next_stage = str(execution_template.get("default_initial_stage", "SYNTHESIZE"))
            if allowed_stages and next_stage not in allowed_stages:
                next_stage = "COMPUTE" if "COMPUTE" in allowed_stages else "GATHER" if "GATHER" in allowed_stages else "SYNTHESIZE"
            workpad = _record_event(workpad, "solver", f"Plan complete -> {next_stage}")
            return {"solver_stage": next_stage, "workpad": workpad}

        effective_stage = stage
        if stage == "REVISE":
            repair_target = str(review_feedback.get("repair_target", "final"))
            if risk_feedback:
                repair_target = str(risk_feedback.get("repair_target", repair_target))
            if compliance_feedback:
                repair_target = str(compliance_feedback.get("repair_target", repair_target))
            if repair_target == "gather":
                effective_stage = "GATHER"
            elif repair_target == "compute":
                effective_stage = "COMPUTE"
            else:
                effective_stage = "SYNTHESIZE"

        stage_prompt = _STAGE_INSTRUCTIONS.get(stage, _STAGE_INSTRUCTIONS["SYNTHESIZE"])
        if profile_pack.get("domain_summary"):
            stage_prompt += f"\nDomain summary: {profile_pack['domain_summary']}"
        if execution_template.get("answer_focus"):
            stage_prompt += "\nTemplate focus:\n- " + "\n- ".join(execution_template["answer_focus"])
        contract = state.get("answer_contract", {})
        if contract.get("content_rules"):
            stage_prompt += "\nContent rules:\n- " + "\n- ".join(contract["content_rules"])
        if effective_stage == "SYNTHESIZE" and contract.get("section_requirements"):
            stage_prompt += "\nRequired sections:\n- " + "\n- ".join(contract["section_requirements"])
        disclosure_assumptions = [
            entry.get("assumption", "")
            for entry in state.get("assumption_ledger", [])
            if isinstance(entry, dict) and entry.get("requires_user_visible_disclosure")
        ]
        as_of_date = (
            (state.get("evidence_pack", {}) or {})
            .get("prompt_facts", {})
            .get("as_of_date")
        )
        if effective_stage in {"SYNTHESIZE", "COMPUTE"} and disclosure_assumptions:
            stage_prompt += "\nDisclose these assumptions if they affect the answer:\n- " + "\n- ".join(disclosure_assumptions[:4])
        risk_requirements = workpad.get("risk_requirements") or {}
        compliance_requirements = workpad.get("compliance_requirements") or {}
        policy_context = (state.get("evidence_pack", {}) or {}).get("policy_context", {}) or {}
        if effective_stage == "SYNTHESIZE" and risk_requirements.get("required_disclosures"):
            stage_prompt += "\nRisk-required disclosures:\n- " + "\n- ".join(
                str(item) for item in risk_requirements.get("required_disclosures", [])[:5]
            )
        if effective_stage == "SYNTHESIZE" and risk_requirements.get("recommendation_class"):
            stage_prompt += (
                "\nRecommendation class for this finance answer: "
                f"{risk_requirements.get('recommendation_class')}. "
                "Make the answer explicitly reflect this confidence posture."
            )
        policy_lines: list[str] = []
        if policy_context.get("defined_risk_only"):
            policy_lines.append("Use a defined-risk structure only.")
        if policy_context.get("no_naked_options"):
            policy_lines.append("Do not recommend naked options.")
        if isinstance(policy_context.get("max_position_risk_pct"), (int, float)):
            policy_lines.append(
                f"Carry a max position-risk cap of about {float(policy_context['max_position_risk_pct']):g}% of capital."
            )
        if policy_context.get("requires_recommendation_class"):
            policy_lines.append("State the recommendation class explicitly.")
        if policy_lines and effective_stage == "SYNTHESIZE":
            stage_prompt += "\nPolicy constraints:\n- " + "\n- ".join(policy_lines)
        if effective_stage == "SYNTHESIZE" and compliance_requirements.get("required_disclosures"):
            stage_prompt += "\nCompliance-required disclosures:\n- " + "\n- ".join(
                str(item) for item in compliance_requirements.get("required_disclosures", [])[:5]
            )
        if as_of_date and effective_stage in {"GATHER", "COMPUTE"}:
            stage_prompt += (
                f"\nIf you use market or statement tools, pass as_of_date=\"{as_of_date}\" "
                "unless the task explicitly asks for current data instead."
            )
        document_evidence = state.get("evidence_pack", {}).get("document_evidence", [])
        if effective_stage == "GATHER" and execution_template.get("template_id") in {
            "legal_with_document_evidence",
            "document_qa",
        }:
            if not has_extracted_document_evidence(document_evidence):
                stage_prompt += (
                    "\nFor document gathering, fetch metadata plus a narrow page/row window first. "
                    "Prefer targeted extraction over raw document dumps. "
                    "If the prompt already includes a URL, use fetch_reference_file directly with small limits."
                )
            else:
                stage_prompt += (
                    "\nDocument evidence already exists. Gather only one additional targeted window if the current "
                    "evidence still leaves a material question unanswered."
                )
        if (
            profile == "finance_quant"
            and "needs_live_data" in set(state.get("capability_flags", []))
            and effective_stage == "GATHER"
            and not state.get("last_tool_result")
        ):
            stage_prompt += (
                "\nFor finance_quant gather stage with live-data needs, emit one finance evidence tool call before "
                "any narrative. Prefer get_price_history, get_returns, get_company_fundamentals, get_yield_curve, "
                "or get_statement_line_items depending on the request."
            )
        if profile == "finance_options" and effective_stage == "COMPUTE" and not state.get("last_tool_result"):
            stage_prompt += (
                "\nFor finance_options compute stage, emit one options-analysis tool call before narrative "
                "unless the evidence pack already contains concrete tool-backed strategy facts."
            )
        if profile == "finance_options" and effective_stage == "COMPUTE" and risk_feedback:
            stage_prompt += (
                "\nRisk controller feedback is active. If structured primary strategy facts already exist, "
                "emit one risk/scenario tool call next, preferably scenario_pnl. "
                "Use the latest strategy facts for net_premium, delta, gamma, theta, vega, and reference spot "
                "before writing more narrative."
            )
        if stage == "REVISE" and review_feedback:
            stage_prompt += (
                "\nReviewer feedback:\n"
                f"{json.dumps(review_feedback, ensure_ascii=True)}\n"
                f"Repair target stage: {effective_stage}."
            )
        if stage == "REVISE" and risk_feedback:
            stage_prompt += (
                "\nRisk controller feedback:\n"
                f"{json.dumps(risk_feedback, ensure_ascii=True)}\n"
                f"Repair target stage: {effective_stage}."
            )
        if stage == "REVISE" and compliance_feedback:
            stage_prompt += (
                "\nCompliance guard feedback:\n"
                f"{json.dumps(compliance_feedback, ensure_ascii=True)}\n"
                f"Repair target stage: {effective_stage}. "
                "If the current recommendation violates mandate or product constraints, rewrite it into a compliant alternative "
                "or explicitly recommend no action."
            )
        allowed_tools = _allowed_tools_for_state(tools, state)
        allowed_tool_names = {getattr(tool, "name", "") for tool in allowed_tools}
        deterministic_call = _deterministic_finance_tool_call(
            state,
            profile=profile,
            effective_stage=effective_stage,
            allowed_tool_names=allowed_tool_names,
            risk_feedback=risk_feedback,
        )
        if deterministic_call:
            tool_name = str(deterministic_call["name"])
            tool_args = dict(deterministic_call.get("arguments", {}))
            message = _build_tool_call_message(tool_name, tool_args)
            pending = ToolCallEnvelope(name=tool_name, arguments=tool_args).model_dump()
            workpad = _record_event(workpad, "solver", f"{stage}: tool_call {tool_name}")
            logger.info("[Step %s] solver(%s) -> deterministic tool_call %s", step, stage, tool_name)
            return {
                "messages": [message],
                "pending_tool_call": pending,
                "workpad": workpad,
            }

        deterministic_compute_text = None
        if execution_template.get("template_id") == "quant_inline_exact" and effective_stage == "COMPUTE":
            deterministic_compute_text = _deterministic_quant_compute_summary(state)
        elif profile == "finance_options" and effective_stage == "COMPUTE":
            deterministic_compute_text = _deterministic_options_compute_summary(state)
        elif execution_template.get("template_id") == "portfolio_risk_review" and effective_stage == "COMPUTE":
            deterministic_compute_text = _deterministic_portfolio_compute_summary(state)
        elif execution_template.get("template_id") == "equity_research_report" and effective_stage == "COMPUTE":
            deterministic_compute_text = _deterministic_research_compute_summary(state)
        elif execution_template.get("template_id") in {"event_driven_finance", "regulated_actionable_finance"} and effective_stage == "COMPUTE":
            deterministic_compute_text = _deterministic_event_compute_summary(state)
        if deterministic_compute_text:
            workpad = _merge_stage_output(workpad, effective_stage, deterministic_compute_text)
            if stage_is_review_milestone(execution_template, effective_stage):
                workpad["review_ready"] = True
                workpad["review_stage"] = effective_stage
                workpad = _record_event(workpad, "solver", f"{effective_stage}: deterministic milestone draft ready")
                logger.info("[Step %s] solver(%s) -> deterministic milestone ready", step, effective_stage)
                return {
                    "workpad": workpad,
                    "pending_tool_call": None,
                    "risk_feedback": None,
                }

        deterministic_final_text = None
        if execution_template.get("template_id") == "quant_inline_exact" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = _deterministic_quant_final_answer(state)
        elif profile == "finance_options" and effective_stage == "SYNTHESIZE" and not compliance_feedback:
            if policy_context.get("defined_risk_only") or policy_context.get("no_naked_options"):
                deterministic_final_text = _deterministic_policy_options_final_answer(state)
            else:
                deterministic_final_text = _deterministic_options_final_answer(state)
        elif execution_template.get("template_id") == "portfolio_risk_review" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = _deterministic_portfolio_final_answer(state)
        elif execution_template.get("template_id") == "equity_research_report" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = _deterministic_research_final_answer(state)
        elif execution_template.get("template_id") == "event_driven_finance" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = _deterministic_event_final_answer(state)
        elif execution_template.get("template_id") == "regulated_actionable_finance" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = _deterministic_actionable_finance_final_answer(state)
        if deterministic_final_text:
            final_message = AIMessage(content=deterministic_final_text)
            workpad["draft_answer"] = deterministic_final_text
            workpad["review_ready"] = True
            workpad["review_stage"] = "SYNTHESIZE"
            workpad = _record_event(workpad, "solver", f"{stage}: deterministic final draft ready")
            logger.info("[Step %s] solver(%s) -> deterministic final draft", step, stage)
            return {
                "messages": [final_message],
                "workpad": workpad,
                "pending_tool_call": None,
                "risk_feedback": None,
            }

        messages = [
            SystemMessage(content=SOLVER_PROMPT),
            SystemMessage(content=stage_prompt),
            SystemMessage(content=_compact_evidence_block(state)),
        ]
        if use_prompt_tools:
            tool_prompt = _build_tool_prompt_block(allowed_tools)
            if tool_prompt:
                messages.append(SystemMessage(content=tool_prompt))
        messages.append(HumanMessage(content=latest_human_text(state["messages"])))

        model_name = get_model_name("executor")
        model = ChatOpenAI(
            model=model_name,
            **get_client_kwargs("executor"),
            temperature=0,
            max_tokens=_solver_max_tokens(stage, profile),
        )
        if _tool_call_mode("executor") == "native" and allowed_tools:
            model = model.bind_tools(allowed_tools)

        t0 = time.monotonic()
        response = model.invoke(messages)
        latency = (time.monotonic() - t0) * 1000
        if isinstance(response, AIMessage):
            response = _patch_prompt_tool_call(response, allowed_tools)
        else:
            response = AIMessage(content=str(getattr(response, "content", "")))

        if tracker:
            tracker.record(
                operator=f"solver_{stage.lower()}",
                model_name=model_name,
                tokens_in=count_tokens(messages),
                tokens_out=_estimate_response_tokens(response),
                latency_ms=latency,
                success=True,
            )

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            pending = ToolCallEnvelope(
                name=tool_call["name"],
                arguments=tool_call.get("args", {}),
            ).model_dump()
            workpad = _record_event(workpad, "solver", f"{stage}: tool_call {tool_call['name']}")
            logger.info("[Step %s] solver(%s) -> tool_call %s", step, stage, tool_call["name"])
            return {
                "messages": [response],
                "pending_tool_call": pending,
                "workpad": workpad,
            }

        content = _strip_think_markup(str(response.content or ""))
        if effective_stage == "SYNTHESIZE":
            content = _apply_compliance_final_fixes(
                content,
                state=state,
                compliance_feedback=compliance_feedback,
                policy_context=policy_context,
                risk_requirements=risk_requirements,
            )
        workpad = _merge_stage_output(workpad, effective_stage, content)

        if effective_stage in {"GATHER", "COMPUTE"}:
            if stage_is_review_milestone(execution_template, effective_stage):
                workpad["review_ready"] = True
                workpad["review_stage"] = effective_stage
                workpad = _record_event(workpad, "solver", f"{effective_stage}: milestone draft ready")
                logger.info("[Step %s] solver(%s) -> milestone ready", step, effective_stage)
                return {
                    "workpad": workpad,
                    "pending_tool_call": None,
                    "risk_feedback": None,
                }
            next_target = "compute"
            if effective_stage == "COMPUTE":
                next_target = "synthesize"
            elif effective_stage == "GATHER":
                if not ("needs_math" in set(state.get("capability_flags", [])) or profile in {"finance_quant", "finance_options"}):
                    next_target = "synthesize"
            next_stage = next_stage_after_review(effective_stage, next_target, "pass")
            workpad = _record_event(workpad, "solver", f"{effective_stage}: auto-advance -> {next_stage}")
            logger.info("[Step %s] solver(%s) -> auto-advance %s", step, effective_stage, next_stage)
            return {
                "solver_stage": next_stage,
                "workpad": workpad,
                "pending_tool_call": None,
                "risk_feedback": None,
            }

        final_message = AIMessage(content=content)
        workpad["draft_answer"] = content
        workpad["review_ready"] = True
        workpad["review_stage"] = "SYNTHESIZE"
        workpad = _record_event(workpad, "solver", f"{stage}: final draft ready")
        logger.info("[Step %s] solver(%s) -> final draft", step, stage)
        return {
            "messages": [final_message],
            "workpad": workpad,
            "pending_tool_call": None,
            "risk_feedback": None,
        }

    return solver
