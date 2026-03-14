"""
Task Profiler Node
==================
Profiles the task into a coarse task profile plus additive capability flags.
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.contracts import AnswerContract, TaskProfile
from agent.cost import CostTracker
from agent.model_config import _extract_json_payload, get_client_kwargs, get_model_name
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import (
    detect_capability_flags,
    infer_task_profile,
    latest_human_text,
)
from agent.state import AgentState
from context_manager import count_tokens

logger = logging.getLogger(__name__)

_PROFILE_SCHEMA_PROMPT = """You are a task profiler.
Return ONLY one JSON object with these keys:
- task_profile
- capability_flags

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

Do not answer the task. Do not add any other keys."""


def _should_try_llm_profile(task_text: str, capability_flags: list[str]) -> bool:
    flags = set(capability_flags)
    if len(flags) < 2:
        return False
    return (
        "needs_legal_reasoning" in flags and "needs_math" in flags
    ) or (
        "needs_live_data" in flags and ("needs_files" in flags or "needs_legal_reasoning" in flags)
    )


def _llm_profile(task_text: str) -> tuple[TaskProfile, list[str]] | None:
    model_name = get_model_name("coordinator")
    llm = ChatOpenAI(
        model=model_name,
        **get_client_kwargs("coordinator"),
        temperature=0,
        max_tokens=120,
    )
    raw = llm.invoke(
        [
            SystemMessage(content=_PROFILE_SCHEMA_PROMPT),
            HumanMessage(content=task_text),
        ]
    )
    payload = json.loads(_extract_json_payload(str(raw.content or "")))
    task_profile = str(payload.get("task_profile", "general"))
    flags = payload.get("capability_flags", [])
    if not isinstance(flags, list):
        flags = []
    return task_profile, sorted({str(flag) for flag in flags})


def task_profiler(state: AgentState) -> dict:
    step = increment_runtime_step()
    tracker: CostTracker = state.get("cost_tracker")
    task_text = latest_human_text(state["messages"])
    answer_contract = state.get("answer_contract", {})

    capability_flags = detect_capability_flags(
        task_text,
        AnswerContract.model_validate(answer_contract if isinstance(answer_contract, dict) else {}),
    )
    task_profile: TaskProfile = infer_task_profile(task_text, capability_flags)

    invocation_messages = [HumanMessage(content=task_text)]
    model_name = get_model_name("coordinator")
    success = True
    latency = 0.0
    used_llm = False
    if _should_try_llm_profile(task_text, capability_flags):
        used_llm = True
        t0 = time.monotonic()
        try:
            llm_profile = _llm_profile(task_text)
            if llm_profile is not None:
                task_profile, llm_flags = llm_profile
                capability_flags = sorted(set(capability_flags) | set(llm_flags))
        except Exception as exc:
            logger.warning("Task profiler LLM fallback failed: %s. Using deterministic profile.", exc)
            success = False
        latency = (time.monotonic() - t0) * 1000

    workpad = dict(state.get("workpad", {}))
    workpad["task_profile"] = task_profile
    workpad["capability_flags"] = capability_flags
    workpad.setdefault("events", []).append(
        {"node": "task_profiler", "action": f"profile={task_profile} flags={','.join(capability_flags) or 'none'}"}
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
        "[Step %s] task_profiler -> profile=%s flags=%s",
        step,
        task_profile,
        capability_flags,
    )
    return {
        "task_profile": task_profile,
        "capability_flags": capability_flags,
        "workpad": workpad,
    }
