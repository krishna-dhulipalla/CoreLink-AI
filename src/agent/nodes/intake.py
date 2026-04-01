"""
Intake Node
===========
Normalizes the incoming task and extracts the first-class answer contract.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agent.benchmarks import merge_benchmark_overrides
from agent.contracts import AnswerContract
from agent.runtime_clock import increment_runtime_step
from agent.runtime_support import extract_answer_contract, latest_human_text
from agent.state import AgentState, ReplaceMessages
from agent.tracer import get_tracer

logger = logging.getLogger(__name__)


def intake(state: AgentState) -> dict:
    step = increment_runtime_step()
    task_text = latest_human_text(state["messages"])
    runtime_overrides = dict(state.get("benchmark_overrides") or {})
    benchmark_overrides = merge_benchmark_overrides(task_text, runtime_overrides)
    contract: AnswerContract = extract_answer_contract(task_text, benchmark_overrides=benchmark_overrides)

    workpad = dict(state.get("workpad", {}))
    workpad.setdefault("stage_history", [])
    workpad.setdefault("events", [])
    workpad["answer_contract_detected"] = contract.requires_adapter
    workpad["stage_history"].append("PLAN")
    workpad["events"].append({"node": "intake", "action": f"Detected output format={contract.format}"})

    logger.info(
        "[Step %s] intake -> format=%s adapter=%s",
        step,
        contract.format,
        contract.requires_adapter,
    )

    # Keep messages clean: only the conversation itself stays in message history.
    clean_messages = [
        msg for msg in state["messages"] if isinstance(msg, HumanMessage) or getattr(msg, "type", "") != "system"
    ]

    tracer = get_tracer()
    if tracer:
        tracer.record("intake", {
            "answer_contract": contract.model_dump(),
            "benchmark_overrides": benchmark_overrides,
            "message_count": len(clean_messages),
        })

    return {
        "messages": ReplaceMessages(clean_messages),
        "answer_contract": contract.model_dump(),
        "benchmark_overrides": benchmark_overrides,
        "solver_stage": "PLAN",
        "workpad": workpad,
    }
