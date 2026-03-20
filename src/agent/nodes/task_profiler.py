"""
Task Profiler Node
==================
Profiles the task into a coarse task profile plus additive capability flags.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.contracts import AnswerContract, ProfileDecision, TaskProfile
from agent.cost import CostTracker
from agent.model_config import get_model_name, invoke_structured_output
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import (
    build_profile_decision,
    latest_human_text,
)
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)

_PROFILE_SCHEMA_PROMPT = """You are a task profiler.
Return ONLY one JSON object with these keys:
- task_profile
- capability_flags
- ambiguity_flags

Valid task_profile values:
- finance_quant
- finance_options
- legal_transactional
- document_qa
- external_retrieval
- general

Valid capability_flags values:
- needs_math
- needs_tables
- needs_files
- needs_live_data
- needs_options_engine
- needs_legal_reasoning
- requires_exact_format
- needs_equity_research
- needs_portfolio_risk
- needs_event_analysis

Valid ambiguity_flags values:
- legal_finance_overlap
- legal_options_overlap
- document_math_overlap
- document_live_overlap
- broad_multi_capability

Do not answer the task. Do not add any other keys."""


class _ProfileDecisionPayload(BaseModel):
    task_profile: TaskProfile = "general"
    capability_flags: list[str] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)


def _should_try_llm_profile(profile_decision: ProfileDecision) -> bool:
    flags = set(profile_decision.capability_flags)
    ambiguity = set(profile_decision.ambiguity_flags)
    if ambiguity:
        return True
    if len(flags) < 2:
        return False
    return (
        "needs_legal_reasoning" in flags and "needs_math" in flags
    ) or (
        "needs_live_data" in flags and ("needs_files" in flags or "needs_legal_reasoning" in flags)
    )


def _llm_profile(task_text: str) -> tuple[TaskProfile, list[str], list[str]] | None:
    parsed, _ = invoke_structured_output(
        "profiler",
        _ProfileDecisionPayload,
        [
            SystemMessage(content=_PROFILE_SCHEMA_PROMPT),
            HumanMessage(content=task_text),
        ],
        temperature=0,
        max_tokens=120,
    )
    task_profile = str(parsed.task_profile or "general")
    valid_profiles = {
        "finance_quant",
        "finance_options",
        "legal_transactional",
        "document_qa",
        "external_retrieval",
        "general",
    }
    if task_profile not in valid_profiles:
        task_profile = "general"
    flags = parsed.capability_flags if isinstance(parsed.capability_flags, list) else []
    ambiguity = parsed.ambiguity_flags if isinstance(parsed.ambiguity_flags, list) else []
    return task_profile, sorted({str(flag) for flag in flags}), sorted({str(flag) for flag in ambiguity})


def task_profiler(state: AgentState) -> dict:
    step = increment_runtime_step()
    tracker: CostTracker = state.get("cost_tracker")
    task_text = latest_human_text(state["messages"])
    answer_contract = state.get("answer_contract", {})
    answer_contract_model = AnswerContract.model_validate(answer_contract if isinstance(answer_contract, dict) else {})

    profile_decision = build_profile_decision(task_text, answer_contract_model)
    task_profile: TaskProfile = profile_decision.primary_profile
    capability_flags = list(profile_decision.capability_flags)
    ambiguity_flags = list(profile_decision.ambiguity_flags)

    invocation_messages = [HumanMessage(content=task_text)]
    model_name = get_model_name("profiler")
    success = True
    latency = 0.0
    used_llm = False
    if _should_try_llm_profile(profile_decision):
        used_llm = True
        t0 = time.monotonic()
        try:
            llm_profile = _llm_profile(task_text)
            if llm_profile is not None:
                task_profile, llm_flags, llm_ambiguity = llm_profile
                capability_flags = sorted(set(capability_flags) | set(llm_flags))
                ambiguity_flags = sorted(set(ambiguity_flags) | set(llm_ambiguity))
                if "legal_options_overlap" in ambiguity_flags:
                    task_profile = "general"
                profile_decision = ProfileDecision(
                    primary_profile=task_profile,
                    capability_flags=capability_flags,
                    ambiguity_flags=ambiguity_flags,
                    needs_external_data="needs_live_data" in capability_flags,
                    needs_output_adapter=answer_contract_model.requires_adapter,
                )
        except Exception as exc:
            logger.warning("Task profiler LLM fallback failed: %s. Using deterministic profile.", exc)
            success = False
        latency = (time.monotonic() - t0) * 1000

    workpad = dict(state.get("workpad", {}))
    workpad["profile_decision"] = profile_decision.model_dump()
    workpad["task_profile"] = task_profile
    workpad["capability_flags"] = capability_flags
    workpad["ambiguity_flags"] = ambiguity_flags
    workpad.setdefault("events", []).append(
        {
            "node": "task_profiler",
            "action": (
                f"profile={task_profile} "
                f"flags={','.join(capability_flags) or 'none'} "
                f"ambiguity={','.join(ambiguity_flags) or 'none'}"
            ),
        }
    )

    if tracker and used_llm:
        tracker.record(
            operator="task_profiler",
            model_name=model_name,
            tokens_in=count_tokens(invocation_messages),
            tokens_out=0,
            latency_ms=latency,
            success=success,
        )

    logger.info(
        "[Step %s] task_profiler -> profile=%s flags=%s ambiguity=%s",
        step,
        task_profile,
        capability_flags,
        ambiguity_flags,
    )
    return {
        "profile_decision": profile_decision.model_dump(),
        "task_profile": task_profile,
        "capability_flags": capability_flags,
        "ambiguity_flags": ambiguity_flags,
        "workpad": workpad,
    }
