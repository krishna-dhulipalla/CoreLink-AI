"""Benchmark adapter entrypoints."""

from __future__ import annotations

import json
from typing import Any

from agent.contracts import AnswerContract, TaskIntent

from .base import benchmark_name_from_env
from .officeqa_index import resolve_source_files_to_manifest
from .officeqa_runtime import (
    OfficeQACorpusBootstrapError,
    officeqa_competition_corpus_required,
    verify_officeqa_competition_bootstrap,
)
from .officeqa import (
    build_officeqa_overrides,
    officeqa_answer_contract,
    officeqa_descriptor_allowed,
    officeqa_registry_policy,
    officeqa_runtime_policy,
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


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        compact = value.strip()
        if not compact:
            return []
        if compact.startswith("[") and compact.endswith("]"):
            try:
                parsed = json.loads(compact)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [part.strip() for part in compact.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def merge_benchmark_overrides(task_text: str, runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = build_benchmark_overrides(task_text)
    incoming = dict(runtime_overrides or {})
    merged = dict(base)

    for key, value in incoming.items():
        if key == "benchmark_policy":
            continue
        merged[key] = value

    if not merged.get("benchmark_name"):
        merged["benchmark_name"] = benchmark_name_from_env()
    if not merged.get("benchmark_adapter") and merged.get("benchmark_name") == "officeqa":
        merged["benchmark_adapter"] = "officeqa"

    source_files = _coerce_string_list(incoming.get("source_files"))
    source_files.extend(_coerce_string_list(incoming.get("source_docs")))
    source_files = list(dict.fromkeys(source_files))
    merged["source_files_expected"] = source_files
    if merged.get("benchmark_adapter") == "officeqa" and source_files:
        matches = resolve_source_files_to_manifest(source_files)
        merged["source_files_found"] = [item for item in matches if item.get("matched")]
        merged["source_files_missing"] = [item for item in matches if not item.get("matched")]
    else:
        merged["source_files_found"] = []
        merged["source_files_missing"] = []
    return merged


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


def benchmark_runtime_policy(benchmark_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = dict(benchmark_overrides or {})
    policy = dict(overrides.get("benchmark_policy") or {})
    if policy:
        return policy
    if overrides.get("benchmark_adapter") == "officeqa":
        return officeqa_runtime_policy()
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
    "benchmark_runtime_policy",
    "benchmark_task_intent",
    "benchmark_tool_selection_active",
    "build_benchmark_overrides",
    "merge_benchmark_overrides",
    "officeqa_competition_corpus_required",
    "OfficeQACorpusBootstrapError",
    "verify_officeqa_competition_bootstrap",
]
