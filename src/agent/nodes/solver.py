"""
Solver Node
===========
Stage-based solver for the finance-first runtime.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.contracts import ToolCallEnvelope
from agent.cost import CostTracker
from agent.document_evidence import has_extracted_document_evidence
from agent.model_config import _extract_json_payload, _tool_call_mode, get_client_kwargs, get_model_name
from agent.nodes.compliance_guard import requires_compliance_guard
from agent.nodes.risk_controller import requires_risk_control
from agent.profile_packs import get_profile_pack
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import latest_human_text, next_stage_after_review, stage_is_review_milestone
from agent.solver.common import (
    allowed_tools_for_state,
    build_tool_prompt_block,
    compact_evidence_block,
    estimate_response_tokens,
    merge_stage_output,
    patch_prompt_tool_call,
    record_event,
    solver_max_tokens,
    strip_think_markup,
)
from agent.solver.market import best_available_timestamp, first_ticker_entity, infer_period_from_text
from agent.solver.options import (
    deterministic_options_compute_summary,
    deterministic_options_final_answer,
    deterministic_policy_options_final_answer,
    deterministic_policy_options_tool_call,
    deterministic_standard_options_tool_call,
    scenario_args_from_primary_tool,
)
from agent.solver.portfolio import (
    deterministic_portfolio_compute_summary,
    deterministic_portfolio_final_answer,
    limit_constraints_from_evidence,
    portfolio_limit_metrics,
    portfolio_positions_from_evidence,
    returns_series_from_evidence,
)
from agent.solver.quant import deterministic_quant_compute_summary, deterministic_quant_final_answer
from agent.solver.research import (
    deterministic_actionable_finance_final_answer,
    deterministic_event_compute_summary,
    deterministic_event_final_answer,
    deterministic_research_compute_summary,
    deterministic_research_final_answer,
)
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)

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
        timestamp = best_available_timestamp(state)
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
                return {"name": "pct_change", "arguments": {"old_value": float(start_close), "new_value": float(end_close)}}

        if not last_tool_result and "get_price_history" in allowed_tool_names and any(
            token in normalized for token in ("price history", "historical prices", "return", "monthly return", "1-month")
        ):
            ticker = first_ticker_entity(evidence_pack.get("entities", []))
            if ticker:
                args = {"ticker": ticker, "period": infer_period_from_text(task_text)}
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
        policy_call = deterministic_policy_options_tool_call(state)
        if policy_call:
            return policy_call
        return deterministic_standard_options_tool_call(state)

    if (
        profile == "finance_options"
        and effective_stage == "COMPUTE"
        and "scenario_pnl" in allowed_tool_names
        and "MISSING_SCENARIO_ANALYSIS" in set(risk_feedback.get("violation_codes", []))
    ):
        args = scenario_args_from_primary_tool(last_tool_result)
        if args:
            return {"name": "scenario_pnl", "arguments": args}

    if template_id == "equity_research_report" and effective_stage == "GATHER":
        ticker = first_ticker_entity(evidence_pack.get("entities", []))
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
        ticker = first_ticker_entity(evidence_pack.get("entities", []))
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
        ticker = first_ticker_entity(evidence_pack.get("entities", []))
        as_of_date = prompt_facts.get("as_of_date")
        tool_types = {str(result.get("type", "")) for result in tool_results if isinstance(result, dict)}
        if ticker and "get_company_fundamentals" in allowed_tool_names and "get_company_fundamentals" not in tool_types:
            args = {"ticker": ticker}
            if isinstance(as_of_date, str) and as_of_date:
                args["as_of_date"] = as_of_date
            return {"name": "get_company_fundamentals", "arguments": args}
        if ticker and "get_price_history" in allowed_tool_names and "get_price_history" not in tool_types:
            args = {"ticker": ticker, "period": infer_period_from_text(task_text)}
            if isinstance(as_of_date, str) and as_of_date:
                args["as_of_date"] = as_of_date
            return {"name": "get_price_history", "arguments": args}

    if template_id == "portfolio_risk_review" and effective_stage == "COMPUTE":
        positions = portfolio_positions_from_evidence(evidence_pack)
        returns_series = returns_series_from_evidence(evidence_pack)
        limits = limit_constraints_from_evidence(
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
            return {"name": "calculate_var", "arguments": {"portfolio_value": portfolio_value, "daily_vol": daily_vol, "confidence_level": 0.95}}
        if positions and "liquidity_stress" in allowed_tool_names and "liquidity_stress" not in tool_types:
            redemption_pct = float(prompt_facts.get("redemption_pct", 0.10) or 0.10)
            return {"name": "liquidity_stress", "arguments": {"positions": positions, "redemption_pct": redemption_pct}}
        if limits and "portfolio_limit_check" in allowed_tool_names and "portfolio_limit_check" not in tool_types:
            metrics = portfolio_limit_metrics(tool_results)
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
            workpad = record_event(workpad, "solver", f"Plan complete -> {next_stage}")
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
        as_of_date = (state.get("evidence_pack", {}) or {}).get("prompt_facts", {}).get("as_of_date")
        if effective_stage in {"SYNTHESIZE", "COMPUTE"} and disclosure_assumptions:
            stage_prompt += "\nDisclose these assumptions if they affect the answer:\n- " + "\n- ".join(disclosure_assumptions[:4])
        risk_requirements = workpad.get("risk_requirements") or {}
        compliance_requirements = workpad.get("compliance_requirements") or {}
        policy_context = (state.get("evidence_pack", {}) or {}).get("policy_context", {}) or {}
        if effective_stage == "SYNTHESIZE" and risk_requirements.get("required_disclosures"):
            stage_prompt += "\nRisk-required disclosures:\n- " + "\n- ".join(str(item) for item in risk_requirements.get("required_disclosures", [])[:5])
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
            policy_lines.append(f"Carry a max position-risk cap of about {float(policy_context['max_position_risk_pct']):g}% of capital.")
        if policy_context.get("requires_recommendation_class"):
            policy_lines.append("State the recommendation class explicitly.")
        if policy_lines and effective_stage == "SYNTHESIZE":
            stage_prompt += "\nPolicy constraints:\n- " + "\n- ".join(policy_lines)
        if effective_stage == "SYNTHESIZE" and compliance_requirements.get("required_disclosures"):
            stage_prompt += "\nCompliance-required disclosures:\n- " + "\n- ".join(str(item) for item in compliance_requirements.get("required_disclosures", [])[:5])
        if as_of_date and effective_stage in {"GATHER", "COMPUTE"}:
            stage_prompt += f'\nIf you use market or statement tools, pass as_of_date="{as_of_date}" unless the task explicitly asks for current data instead.'
        document_evidence = state.get("evidence_pack", {}).get("document_evidence", [])
        if effective_stage == "GATHER" and execution_template.get("template_id") in {"legal_with_document_evidence", "document_qa"}:
            if not has_extracted_document_evidence(document_evidence):
                stage_prompt += "\nFor document gathering, fetch metadata plus a narrow page/row window first. Prefer targeted extraction over raw document dumps. If the prompt already includes a URL, use fetch_reference_file directly with small limits."
            else:
                stage_prompt += "\nDocument evidence already exists. Gather only one additional targeted window if the current evidence still leaves a material question unanswered."
        if profile == "finance_quant" and "needs_live_data" in set(state.get("capability_flags", [])) and effective_stage == "GATHER" and not state.get("last_tool_result"):
            stage_prompt += "\nFor finance_quant gather stage with live-data needs, emit one finance evidence tool call before any narrative. Prefer get_price_history, get_returns, get_company_fundamentals, get_yield_curve, or get_statement_line_items depending on the request."
        if profile == "finance_options" and effective_stage == "COMPUTE" and not state.get("last_tool_result"):
            stage_prompt += "\nFor finance_options compute stage, emit one options-analysis tool call before narrative unless the evidence pack already contains concrete tool-backed strategy facts."
        if profile == "finance_options" and effective_stage == "COMPUTE" and risk_feedback:
            stage_prompt += "\nRisk controller feedback is active. If structured primary strategy facts already exist, emit one risk/scenario tool call next, preferably scenario_pnl. Use the latest strategy facts for net_premium, delta, gamma, theta, vega, and reference spot before writing more narrative."
        if stage == "REVISE" and review_feedback:
            stage_prompt += "\nReviewer feedback:\n" + f"{json.dumps(review_feedback, ensure_ascii=True)}\n" + f"Repair target stage: {effective_stage}."
        if stage == "REVISE" and risk_feedback:
            stage_prompt += "\nRisk controller feedback:\n" + f"{json.dumps(risk_feedback, ensure_ascii=True)}\n" + f"Repair target stage: {effective_stage}."
        if stage == "REVISE" and compliance_feedback:
            stage_prompt += "\nCompliance guard feedback:\n" + f"{json.dumps(compliance_feedback, ensure_ascii=True)}\n" + f"Repair target stage: {effective_stage}. If the current recommendation violates mandate or product constraints, rewrite it into a compliant alternative or explicitly recommend no action."

        allowed_tools = allowed_tools_for_state(tools, state)
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
            workpad = record_event(workpad, "solver", f"{stage}: tool_call {tool_name}")
            logger.info("[Step %s] solver(%s) -> deterministic tool_call %s", step, stage, tool_name)
            return {"messages": [message], "pending_tool_call": pending, "workpad": workpad}

        deterministic_compute_text = None
        if execution_template.get("template_id") == "quant_inline_exact" and effective_stage == "COMPUTE":
            deterministic_compute_text = deterministic_quant_compute_summary(state)
        elif profile == "finance_options" and effective_stage == "COMPUTE":
            deterministic_compute_text = deterministic_options_compute_summary(state)
        elif execution_template.get("template_id") == "portfolio_risk_review" and effective_stage == "COMPUTE":
            deterministic_compute_text = deterministic_portfolio_compute_summary(state)
        elif execution_template.get("template_id") == "equity_research_report" and effective_stage == "COMPUTE":
            deterministic_compute_text = deterministic_research_compute_summary(state)
        elif execution_template.get("template_id") in {"event_driven_finance", "regulated_actionable_finance"} and effective_stage == "COMPUTE":
            deterministic_compute_text = deterministic_event_compute_summary(state)
        if deterministic_compute_text:
            workpad = merge_stage_output(workpad, effective_stage, deterministic_compute_text)
            if stage_is_review_milestone(execution_template, effective_stage):
                workpad["review_ready"] = True
                workpad["review_stage"] = effective_stage
                workpad = record_event(workpad, "solver", f"{effective_stage}: deterministic milestone draft ready")
                logger.info("[Step %s] solver(%s) -> deterministic milestone ready", step, effective_stage)
                return {"workpad": workpad, "pending_tool_call": None, "risk_feedback": None}

        deterministic_final_text = None
        if execution_template.get("template_id") == "quant_inline_exact" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = deterministic_quant_final_answer(state)
        elif profile == "finance_options" and effective_stage == "SYNTHESIZE" and not compliance_feedback:
            if policy_context.get("defined_risk_only") or policy_context.get("no_naked_options"):
                deterministic_final_text = deterministic_policy_options_final_answer(state)
            else:
                deterministic_final_text = deterministic_options_final_answer(state)
        elif execution_template.get("template_id") == "portfolio_risk_review" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = deterministic_portfolio_final_answer(state)
        elif execution_template.get("template_id") == "equity_research_report" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = deterministic_research_final_answer(state)
        elif execution_template.get("template_id") == "event_driven_finance" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = deterministic_event_final_answer(state)
        elif execution_template.get("template_id") == "regulated_actionable_finance" and effective_stage == "SYNTHESIZE":
            deterministic_final_text = deterministic_actionable_finance_final_answer(state)
        if deterministic_final_text:
            final_message = AIMessage(content=deterministic_final_text)
            workpad["draft_answer"] = deterministic_final_text
            workpad["review_ready"] = True
            workpad["review_stage"] = "SYNTHESIZE"
            workpad = record_event(workpad, "solver", f"{stage}: deterministic final draft ready")
            logger.info("[Step %s] solver(%s) -> deterministic final draft", step, stage)
            return {"messages": [final_message], "workpad": workpad, "pending_tool_call": None, "risk_feedback": None}

        messages = [SystemMessage(content=SOLVER_PROMPT), SystemMessage(content=stage_prompt), SystemMessage(content=compact_evidence_block(state))]
        if use_prompt_tools:
            tool_prompt = build_tool_prompt_block(allowed_tools)
            if tool_prompt:
                messages.append(SystemMessage(content=tool_prompt))
        messages.append(HumanMessage(content=latest_human_text(state["messages"])))

        model_name = get_model_name("executor")
        model = ChatOpenAI(model=model_name, **get_client_kwargs("executor"), temperature=0, max_tokens=solver_max_tokens(stage, profile))
        if _tool_call_mode("executor") == "native" and allowed_tools:
            model = model.bind_tools(allowed_tools)

        t0 = time.monotonic()
        response = model.invoke(messages)
        latency = (time.monotonic() - t0) * 1000
        if isinstance(response, AIMessage):
            response = patch_prompt_tool_call(response, allowed_tools, _extract_json_payload)
        else:
            response = AIMessage(content=str(getattr(response, "content", "")))

        if tracker:
            tracker.record(
                operator=f"solver_{stage.lower()}",
                model_name=model_name,
                tokens_in=count_tokens(messages),
                tokens_out=estimate_response_tokens(response),
                latency_ms=latency,
                success=True,
            )

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            pending = ToolCallEnvelope(name=tool_call["name"], arguments=tool_call.get("args", {})).model_dump()
            workpad = record_event(workpad, "solver", f"{stage}: tool_call {tool_call['name']}")
            logger.info("[Step %s] solver(%s) -> tool_call %s", step, stage, tool_call["name"])
            return {"messages": [response], "pending_tool_call": pending, "workpad": workpad}

        content = strip_think_markup(str(response.content or ""))
        if effective_stage == "SYNTHESIZE":
            content = _apply_compliance_final_fixes(
                content,
                state=state,
                compliance_feedback=compliance_feedback,
                policy_context=policy_context,
                risk_requirements=risk_requirements,
            )
        workpad = merge_stage_output(workpad, effective_stage, content)

        if effective_stage in {"GATHER", "COMPUTE"}:
            if stage_is_review_milestone(execution_template, effective_stage):
                workpad["review_ready"] = True
                workpad["review_stage"] = effective_stage
                workpad = record_event(workpad, "solver", f"{effective_stage}: milestone draft ready")
                logger.info("[Step %s] solver(%s) -> milestone ready", step, effective_stage)
                return {"workpad": workpad, "pending_tool_call": None, "risk_feedback": None}
            next_target = "compute"
            if effective_stage == "COMPUTE":
                next_target = "synthesize"
            elif effective_stage == "GATHER":
                if not ("needs_math" in set(state.get("capability_flags", [])) or profile in {"finance_quant", "finance_options"}):
                    next_target = "synthesize"
            next_stage = next_stage_after_review(effective_stage, next_target, "pass")
            workpad = record_event(workpad, "solver", f"{effective_stage}: auto-advance -> {next_stage}")
            logger.info("[Step %s] solver(%s) -> auto-advance %s", step, effective_stage, next_stage)
            return {"solver_stage": next_stage, "workpad": workpad, "pending_tool_call": None, "risk_feedback": None}

        final_message = AIMessage(content=content)
        workpad["draft_answer"] = content
        workpad["review_ready"] = True
        workpad["review_stage"] = "SYNTHESIZE"
        workpad = record_event(workpad, "solver", f"{stage}: final draft ready")
        logger.info("[Step %s] solver(%s) -> final draft", step, stage)
        return {"messages": [final_message], "workpad": workpad, "pending_tool_call": None, "risk_feedback": None}

    return solver
