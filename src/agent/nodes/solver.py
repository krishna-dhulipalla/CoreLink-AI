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

from agent.contracts import ReviewResult, ToolCallEnvelope, ToolResult
from agent.cost import CostTracker
from agent.model_config import _extract_json_payload, _tool_call_mode, get_client_kwargs, get_model_name
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

    review_feedback = state.get("review_feedback") or {}
    repair_target = str(review_feedback.get("repair_target", "final"))
    if stage == "REVISE" and repair_target not in {"gather", "compute"}:
        allowed_names = set()
    if (
        stage == "REVISE"
        and repair_target == "compute"
        and (state.get("last_tool_result") or {}).get("facts")
        and not (state.get("last_tool_result") or {}).get("errors")
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
        "evidence_pack": evidence,
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


def route_from_solver(state: AgentState) -> str:
    if state.get("pending_tool_call"):
        return "tool_runner"
    workpad = state.get("workpad", {})
    if workpad.get("review_ready"):
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
        if profile == "finance_options" and effective_stage == "COMPUTE" and not state.get("last_tool_result"):
            stage_prompt += (
                "\nFor finance_options compute stage, emit one options-analysis tool call before narrative "
                "unless the evidence pack already contains concrete tool-backed strategy facts."
            )
        if stage == "REVISE" and review_feedback:
            stage_prompt += (
                "\nReviewer feedback:\n"
                f"{json.dumps(review_feedback, ensure_ascii=True)}\n"
                f"Repair target stage: {effective_stage}."
            )
        allowed_tools = _allowed_tools_for_state(tools, state)

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
        }

    return solver
