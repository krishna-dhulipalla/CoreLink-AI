"""Benchmark adapter entrypoints."""

from __future__ import annotations

from typing import Any

from agent.contracts import AnswerContract, TaskIntent

from .base import benchmark_name_from_env
from .officeqa import (
    build_officeqa_overrides,
    officeqa_answer_contract,
    officeqa_descriptor_allowed,
    officeqa_registry_policy,
    officeqa_task_intent,
    officeqa_tool_selection_active,
)


def build_benchmark_overrides(task_text: str) -> dict[str, Any]:
    benchmark_name = benchmark_name_from_env()
    overrides: dict[str, Any] = {
        "benchmark_name": benchmark_name,
        "benchmark_adapter": "",
    }
    overrides.update(build_officeqa_overrides(task_text, benchmark_name))
    return overrides


def benchmark_answer_contract(
    task_text: str,
    benchmark_overrides: dict[str, Any] | None = None,
) -> AnswerContract | None:
    overrides = dict(benchmark_overrides or build_benchmark_overrides(task_text))
    if overrides.get("benchmark_adapter") == "officeqa" and overrides.get("officeqa_xml_contract"):
        return officeqa_answer_contract()
    return None


def benchmark_registry_policy(benchmark_name: str) -> dict[str, Any]:
    normalized = str(benchmark_name or "").strip().lower()
    if normalized == "officeqa":
        return officeqa_registry_policy()
    return {}


def benchmark_tool_selection_active(task_family: str, benchmark_overrides: dict[str, Any] | None = None) -> bool:
    overrides = dict(benchmark_overrides or {})
    if overrides.get("benchmark_adapter") == "officeqa":
        return officeqa_tool_selection_active(task_family, overrides)
    return False


def benchmark_descriptor_allowed(descriptor: dict[str, Any], benchmark_overrides: dict[str, Any] | None = None) -> bool:
    overrides = dict(benchmark_overrides or {})
    if overrides.get("benchmark_adapter") == "officeqa":
        return officeqa_descriptor_allowed(descriptor, overrides)
    return True


def benchmark_task_intent(
    task_text: str,
    capability_flags: list[str],
    benchmark_overrides: dict[str, Any] | None = None,
) -> TaskIntent | None:
    overrides = dict(benchmark_overrides or build_benchmark_overrides(task_text))
    if overrides.get("benchmark_adapter") == "officeqa":
        return officeqa_task_intent(task_text, capability_flags, overrides)
    return None


__all__ = [
    "benchmark_answer_contract",
    "benchmark_descriptor_allowed",
    "benchmark_registry_policy",
    "benchmark_task_intent",
    "benchmark_tool_selection_active",
    "build_benchmark_overrides",
]
