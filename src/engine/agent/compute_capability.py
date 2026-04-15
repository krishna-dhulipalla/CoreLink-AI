from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import statistics
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from engine.agent.benchmarks.officeqa_compute import (
    _build_answer_text,
    _display_contract_value,
    _extract_month_index,
    _extract_years,
    _metric_tokens,
    _normalize_space,
    _pick_numeric_value,
    _provenance_ref,
    _result_with_diagnostics,
    _semantic_admissibility,
    _semantic_validation_errors,
    _structure_gate,
)
from engine.agent.contracts import (
    OfficeQAComputeCapabilitySpec,
    OfficeQAComputeResult,
    OfficeQAComputeStep,
    RetrievalIntent,
)
from engine.agent.llm_control import record_officeqa_llm_usage
from engine.agent.model_config import (
    get_model_name_for_officeqa_control,
    get_model_runtime_kwargs_for_officeqa_control,
    invoke_structured_output,
)
from engine.agent.prompts import FINANCIAL_COMPUTE_CAPABILITY_SYSTEM
from engine.agent.solver.common import format_scalar_number

_COMPUTE_CAPABILITY_ENABLED = os.getenv("OFFICEQA_COMPUTE_CAPABILITY_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_MAX_SAMPLE_RECORDS = 24
_MAX_CODE_CHARS = 5000
_ALLOWED_GLOBAL_NAMES = {
    "abs",
    "all",
    "any",
    "Decimal",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "math",
    "max",
    "min",
    "range",
    "round",
    "set",
    "sorted",
    "statistics",
    "str",
    "sum",
    "tuple",
    "zip",
}
_BLOCKED_LOAD_NAMES = {
    "__import__",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "os",
    "pathlib",
    "setattr",
    "subprocess",
    "sys",
    "vars",
}
_BLOCKED_AST_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
_SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": _SAFE_BUILTINS,
    "math": math,
    "statistics": statistics,
    "Decimal": Decimal,
}
_CAPABILITY_CACHE: dict[str, OfficeQAComputeCapabilitySpec] = {}


def clear_compute_capability_cache() -> None:
    _CAPABILITY_CACHE.clear()


def _operation_signature_payload(task_text: str, retrieval_intent: RetrievalIntent) -> dict[str, Any]:
    _ = task_text
    return {
        "aggregation_shape": str(retrieval_intent.aggregation_shape or ""),
        "analysis_modes": sorted(str(item) for item in list(retrieval_intent.analysis_modes or []) if str(item).strip()),
        "expected_answer_unit_basis": str(retrieval_intent.expected_answer_unit_basis or ""),
        "granularity_requirement": str(retrieval_intent.granularity_requirement or ""),
        "metric": str(retrieval_intent.metric or "").strip().lower(),
        "period_type": str(retrieval_intent.period_type or "").strip().lower(),
    }


def _operation_signature(task_text: str, retrieval_intent: RetrievalIntent) -> str:
    payload = _operation_signature_payload(task_text, retrieval_intent)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _record_identity(value: dict[str, Any], index: int) -> str:
    signature = (
        str(value.get("document_id", "") or ""),
        str(value.get("citation", "") or ""),
        str(value.get("page_locator", "") or ""),
        str(value.get("table_locator", "") or ""),
        int(value.get("row_index", -1) or -1),
        str(value.get("row_label", "") or ""),
        int(value.get("column_index", -1) or -1),
        str(value.get("column_label", "") or ""),
        str(value.get("raw_value", "") or ""),
    )
    encoded = repr(signature).encode("utf-8")
    return f"r{index}_{hashlib.sha1(encoded).hexdigest()[:8]}"


def _build_compute_records(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        row_path = [str(part) for part in list(value.get("row_path", [])) if str(part).strip()]
        column_path = [str(part) for part in list(value.get("column_path", [])) if str(part).strip()]
        year_refs = _extract_years(
            " ".join(
                [
                    str(value.get("row_label", "") or ""),
                    " ".join(row_path),
                    str(value.get("column_label", "") or ""),
                    " ".join(column_path),
                    str(value.get("table_locator", "") or ""),
                    str(value.get("page_locator", "") or ""),
                    str(value.get("document_id", "") or ""),
                ]
            )
        )
        record = {
            "record_id": _record_identity(value, index),
            "document_id": str(value.get("document_id", "") or ""),
            "citation": str(value.get("citation", "") or ""),
            "page_locator": str(value.get("page_locator", "") or ""),
            "table_locator": str(value.get("table_locator", "") or ""),
            "table_family": str(value.get("table_family", "") or ""),
            "period_type": str(value.get("period_type", "") or ""),
            "row_label": str(value.get("row_label", "") or ""),
            "row_path": row_path,
            "column_label": str(value.get("column_label", "") or ""),
            "column_path": column_path,
            "raw_value": str(value.get("raw_value", "") or ""),
            "numeric_value": _pick_numeric_value(value),
            "normalized_value": value.get("normalized_value"),
            "unit": str(value.get("unit", "") or ""),
            "unit_multiplier": float(value.get("unit_multiplier", 1.0) or 1.0),
            "unit_kind": str(value.get("unit_kind", "") or ""),
            "structure_confidence": float(value.get("structure_confidence", 0.0) or 0.0),
            "month_index": _extract_month_index(
                " ".join(
                    [
                        str(value.get("row_label", "") or ""),
                        " ".join(row_path),
                        str(value.get("column_label", "") or ""),
                        " ".join(column_path),
                    ]
                )
            ),
            "year_refs": year_refs,
        }
        records.append(record)
    return records


def _build_capability_prompt(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    records: list[dict[str, Any]],
    signature: str,
    prior_error: str = "",
    attempt_index: int = 1,
) -> str:
    sample = []
    for record in records[:_MAX_SAMPLE_RECORDS]:
        sample.append(
            {
                "record_id": record["record_id"],
                "row_label": record["row_label"],
                "row_path": record["row_path"][:4],
                "column_label": record["column_label"],
                "column_path": record["column_path"][:4],
                "numeric_value": record["numeric_value"],
                "normalized_value": record["normalized_value"],
                "year_refs": record["year_refs"],
                "month_index": record["month_index"],
                "table_family": record["table_family"],
                "unit": record["unit"],
                "unit_multiplier": record["unit_multiplier"],
            }
        )
    return (
        f"TASK={task_text}\n"
        f"OPERATION_SIGNATURE={signature}\n"
        f"ATTEMPT_INDEX={attempt_index}\n"
        f"AGGREGATION_SHAPE={retrieval_intent.aggregation_shape}\n"
        f"ANALYSIS_MODES={list(retrieval_intent.analysis_modes)}\n"
        f"ENTITY={retrieval_intent.entity}\n"
        f"METRIC={retrieval_intent.metric}\n"
        f"PERIOD={retrieval_intent.period}\n"
        f"PERIOD_TYPE={retrieval_intent.period_type}\n"
        f"TARGET_YEARS={list(retrieval_intent.target_years)}\n"
        f"EXPECTED_ANSWER_UNIT_BASIS={retrieval_intent.expected_answer_unit_basis}\n"
        f"GRANULARITY={retrieval_intent.granularity_requirement}\n"
        f"RECORD_COUNT={len(records)}\n"
        f"RECORD_SCHEMA=['record_id','row_label','row_path','column_label','column_path','numeric_value','normalized_value','year_refs','month_index','unit','unit_multiplier','table_family']\n"
        f"PRIOR_ERROR={prior_error}\n"
        f"RECORD_SAMPLE={sample}"
    )


def _validate_function_tree(tree: ast.AST) -> ast.FunctionDef:
    if not isinstance(tree, ast.Module):
        raise ValueError("Capability code must parse as a module.")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Capability code must contain exactly one function.")
    function_def = tree.body[0]
    if function_def.name != "compute_capability":
        raise ValueError("Capability function must be named compute_capability.")
    arg_names = [arg.arg for arg in function_def.args.args]
    if arg_names != ["records", "context"]:
        raise ValueError("Capability function signature must be compute_capability(records, context).")
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_AST_NODES):
            raise ValueError(f"Unsupported capability syntax: {type(node).__name__}.")
        if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
            raise ValueError("Dunder attribute access is not allowed in compute capabilities.")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in _BLOCKED_LOAD_NAMES:
            raise ValueError(f"Blocked capability name: {node.id}.")
    return function_def


def _compile_capability(spec: OfficeQAComputeCapabilitySpec) -> Any:
    code = str(spec.function_code or "").strip()
    if not code:
        raise ValueError("Capability code is empty.")
    if len(code) > _MAX_CODE_CHARS:
        raise ValueError("Capability code exceeds the maximum allowed length.")
    tree = ast.parse(code, mode="exec")
    _validate_function_tree(tree)
    namespace: dict[str, Any] = {}
    exec(compile(tree, "<officeqa_compute_capability>", "exec"), dict(_SAFE_GLOBALS), namespace)
    function = namespace.get("compute_capability")
    if not callable(function):
        raise ValueError("Capability code did not define a callable compute_capability.")
    return function


def _normalize_capability_output(payload: Any, valid_record_ids: set[str]) -> tuple[float | None, list[str], str, str | None]:
    if isinstance(payload, dict):
        error = str(payload.get("error", "") or "").strip()
        if error:
            return None, [], "", error
        final_value = payload.get("final_value", payload.get("value"))
        explanation = str(payload.get("explanation", "") or "").strip()
        selected_record_ids = [str(item).strip() for item in list(payload.get("selected_record_ids", []) or []) if str(item).strip()]
    else:
        final_value = payload
        explanation = ""
        selected_record_ids = []
    if final_value is None:
        return None, [], explanation, "missing_final_value"
    if not isinstance(final_value, (int, float)):
        return None, [], explanation, "non_numeric_final_value"
    final_value = float(final_value)
    if not math.isfinite(final_value):
        return None, [], explanation, "non_finite_final_value"
    if not selected_record_ids:
        selected_record_ids = sorted(valid_record_ids)
    if any(record_id not in valid_record_ids for record_id in selected_record_ids):
        return None, [], explanation, "invalid_selected_record_ids"
    return final_value, selected_record_ids, explanation, None


def _execute_capability(
    spec: OfficeQAComputeCapabilitySpec,
    *,
    records: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[float | None, list[str], str, str | None]:
    function = _compile_capability(spec)
    valid_record_ids = {str(record["record_id"]) for record in records}
    payload = function(copy.deepcopy(records), dict(context))
    value, selected_ids, explanation, error = _normalize_capability_output(payload, valid_record_ids)
    if error:
        return value, selected_ids, explanation, error
    reversed_payload = function(list(reversed(copy.deepcopy(records))), dict(context))
    re_value, re_selected_ids, _re_explanation, re_error = _normalize_capability_output(reversed_payload, valid_record_ids)
    if re_error:
        return None, [], explanation, f"order_invariance_failed:{re_error}"
    if re_value is None or value is None:
        return None, [], explanation, "missing_validation_value"
    if abs(re_value - value) > 1e-9:
        return None, [], explanation, "order_invariance_failed:value_changed"
    if set(re_selected_ids) != set(selected_ids):
        return None, [], explanation, "order_invariance_failed:selection_changed"
    return value, selected_ids, explanation, None


def _capability_context(task_text: str, retrieval_intent: RetrievalIntent) -> dict[str, Any]:
    return {
        "task_text": task_text,
        "aggregation_shape": str(retrieval_intent.aggregation_shape or ""),
        "analysis_modes": list(retrieval_intent.analysis_modes or []),
        "entity": str(retrieval_intent.entity or ""),
        "metric": str(retrieval_intent.metric or ""),
        "period": str(retrieval_intent.period or ""),
        "period_type": str(retrieval_intent.period_type or ""),
        "target_years": list(retrieval_intent.target_years or []),
        "expected_answer_unit_basis": str(retrieval_intent.expected_answer_unit_basis or ""),
        "granularity_requirement": str(retrieval_intent.granularity_requirement or ""),
    }


def _supports_compute_capability_repair(retrieval_intent: RetrievalIntent) -> bool:
    analysis_modes = {
        str(item or "").strip().lower()
        for item in list(retrieval_intent.analysis_modes or [])
        if str(item).strip()
    }
    if analysis_modes & {"statistical_analysis", "risk_metric", "weighted_average", "regression", "forecast"}:
        return True
    aggregation_shape = str(retrieval_intent.aggregation_shape or "").strip().lower()
    return aggregation_shape in {
        "weighted_average",
        "distribution_summary",
        "statistical_summary",
        "risk_metric",
    }


def maybe_acquire_officeqa_compute_result(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    structured_evidence: dict[str, Any] | None,
    workpad: dict[str, Any],
    llm_control_state: dict[str, int],
    llm_control_budget: dict[str, int],
) -> tuple[OfficeQAComputeResult | None, dict[str, Any], dict[str, int]]:
    updated_workpad = dict(workpad)
    updated_llm_state = dict(llm_control_state)
    if not _COMPUTE_CAPABILITY_ENABLED:
        updated_workpad = record_officeqa_llm_usage(
            updated_workpad,
            category="compute_capability_llm",
            used=False,
            reason="capability_acquisition_disabled",
        )
        return None, updated_workpad, updated_llm_state

    evidence = dict(structured_evidence or {})
    values = [item for item in list(evidence.get("values", [])) if isinstance(item, dict)]
    if not values:
        updated_workpad = record_officeqa_llm_usage(
            updated_workpad,
            category="compute_capability_llm",
            used=False,
            reason="no_structured_values",
        )
        return None, updated_workpad, updated_llm_state

    operation = retrieval_intent.aggregation_shape or "point_lookup"
    years = sorted(list(dict.fromkeys(list(retrieval_intent.target_years or []) or _extract_years(f"{retrieval_intent.period} {task_text}"))))
    structure_ok, structure_error = _structure_gate(evidence, values)
    if not structure_ok:
        return (
            _result_with_diagnostics(
                status="insufficient",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[structure_error],
            ),
            updated_workpad,
            updated_llm_state,
        )

    signature = _operation_signature(task_text, retrieval_intent)
    records = _build_compute_records(values)
    context = _capability_context(task_text, retrieval_intent)

    cached_spec = _CAPABILITY_CACHE.get(signature)
    if cached_spec is not None:
        final_value, selected_ids, explanation, error = _execute_capability(cached_spec, records=records, context=context)
        updated_workpad = record_officeqa_llm_usage(
            updated_workpad,
            category="compute_capability_llm",
            used=False,
            reason="cache_hit",
            model_name=cached_spec.model_name,
            applied=error is None,
            details={"operation_signature": signature},
        )
        if error:
            return (
                _result_with_diagnostics(
                    status="unsupported",
                    operation=operation,
                    retrieval_intent=retrieval_intent,
                    years=years,
                    validation_errors=[f"Cached compute capability validation failed: {error}."],
                    capability_source="cached",
                    capability_signature=signature,
                ),
                updated_workpad,
                updated_llm_state,
            )
        return (
            _build_compute_result_from_capability(
                task_text=task_text,
                retrieval_intent=retrieval_intent,
                values=values,
                final_value=final_value,
                selected_ids=selected_ids,
                explanation=explanation,
                operation=operation,
                years=years,
                capability_source="cached",
                capability_signature=signature,
            ),
            updated_workpad,
            updated_llm_state,
        )

    call_count = int(updated_llm_state.get("compute_capability_calls", 0) or 0)
    call_budget = int(llm_control_budget.get("compute_capability_calls", 0) or 0)
    if call_count >= call_budget:
        updated_workpad = record_officeqa_llm_usage(
            updated_workpad,
            category="compute_capability_llm",
            used=False,
            reason="budget_exhausted",
            details={"operation_signature": signature},
        )
        return None, updated_workpad, updated_llm_state

    model_name = get_model_name_for_officeqa_control(
        "compute_capability_llm",
        answer_mode=retrieval_intent.answer_mode,
        analysis_modes=retrieval_intent.analysis_modes,
    )
    runtime_kwargs = get_model_runtime_kwargs_for_officeqa_control(
        "compute_capability_llm",
        answer_mode=retrieval_intent.answer_mode,
        analysis_modes=retrieval_intent.analysis_modes,
    )
    allow_repair_attempt = _supports_compute_capability_repair(retrieval_intent)
    prior_error = ""
    spec: OfficeQAComputeCapabilitySpec | None = None
    final_value: float | None = None
    selected_ids: list[str] = []
    explanation = ""
    error: str | None = None
    last_failure_message = ""
    max_attempts = min(call_budget - call_count, 2 if allow_repair_attempt else 1)
    for attempt_offset in range(max_attempts):
        updated_llm_state["compute_capability_calls"] = call_count + attempt_offset + 1
        try:
            parsed, resolved_model = invoke_structured_output(
                "solver",
                OfficeQAComputeCapabilitySpec,
                [
                    SystemMessage(content=FINANCIAL_COMPUTE_CAPABILITY_SYSTEM),
                    HumanMessage(
                        content=_build_capability_prompt(
                            task_text=task_text,
                            retrieval_intent=retrieval_intent,
                            records=records,
                            signature=signature,
                            prior_error=prior_error,
                            attempt_index=attempt_offset + 1,
                        )
                    ),
                ],
                temperature=0,
                max_tokens=900,
                model_name_override=model_name,
                runtime_kwargs_override=runtime_kwargs,
            )
            spec = OfficeQAComputeCapabilitySpec.model_validate(parsed)
            spec.model_name = resolved_model
            spec.operation_signature = signature
            final_value, selected_ids, explanation, error = _execute_capability(spec, records=records, context=context)
            if error is None:
                updated_workpad = record_officeqa_llm_usage(
                    updated_workpad,
                    category="compute_capability_llm",
                    used=True,
                    reason="capability_acquired" if attempt_offset == 0 else "capability_repaired",
                    model_name=spec.model_name,
                    applied=True,
                    details={
                        "operation_signature": signature,
                        "validation_checks": list(spec.validation_checks[:6]),
                        "attempt_index": attempt_offset + 1,
                    },
                )
                break
            last_failure_message = f"Validated compute capability was rejected: {error}."
            updated_workpad = record_officeqa_llm_usage(
                updated_workpad,
                category="compute_capability_llm",
                used=True,
                reason="capability_rejected",
                model_name=spec.model_name,
                applied=False,
                details={
                    "operation_signature": signature,
                    "validation_checks": list(spec.validation_checks[:6]),
                    "attempt_index": attempt_offset + 1,
                    "error": error,
                },
            )
            prior_error = error
        except Exception as exc:
            last_failure_message = f"Compute capability acquisition failed: {str(exc)[:240]}"
            updated_workpad = record_officeqa_llm_usage(
                updated_workpad,
                category="compute_capability_llm",
                used=True,
                reason="capability_generation_failed",
                model_name=model_name,
                applied=False,
                details={
                    "operation_signature": signature,
                    "attempt_index": attempt_offset + 1,
                    "error": str(exc)[:240],
                },
            )
            prior_error = str(exc)[:240]
            spec = None
            error = prior_error
        if attempt_offset + 1 >= max_attempts:
            break
        if not allow_repair_attempt:
            break
    else:
        spec = None

    if spec is None or error is not None:
        return (
            _result_with_diagnostics(
                status="unsupported",
                operation=operation,
                retrieval_intent=retrieval_intent,
                years=years,
                validation_errors=[last_failure_message or "Compute capability acquisition failed."],
                capability_source="native",
                capability_signature=signature,
            ),
            updated_workpad,
            updated_llm_state,
        )

    _CAPABILITY_CACHE[signature] = spec
    return (
        _build_compute_result_from_capability(
            task_text=task_text,
            retrieval_intent=retrieval_intent,
            values=values,
            final_value=final_value,
            selected_ids=selected_ids,
            explanation=explanation or spec.rationale,
            operation=operation,
            years=years,
            capability_source="synthesized",
            capability_signature=signature,
        ),
        updated_workpad,
        updated_llm_state,
    )


def _build_compute_result_from_capability(
    *,
    task_text: str,
    retrieval_intent: RetrievalIntent,
    values: list[dict[str, Any]],
    final_value: float | None,
    selected_ids: list[str],
    explanation: str,
    operation: str,
    years: list[str],
    capability_source: str,
    capability_signature: str,
) -> OfficeQAComputeResult:
    if final_value is None:
        return _result_with_diagnostics(
            status="unsupported",
            operation=operation,
            retrieval_intent=retrieval_intent,
            years=years,
            validation_errors=["Compute capability returned no numeric value."],
            capability_source=capability_source,
            capability_signature=capability_signature,
        )

    record_map = {_record_identity(value, index): value for index, value in enumerate(values)}
    selected_values = [record_map[record_id] for record_id in selected_ids if record_id in record_map]
    if not selected_values:
        return _result_with_diagnostics(
            status="insufficient",
            operation=operation,
            retrieval_intent=retrieval_intent,
            years=years,
            validation_errors=["Synthesized compute capability did not select any admissible evidence rows."],
            capability_source=capability_source,
            capability_signature=capability_signature,
            capability_validated=True,
        )

    metric_tokens = _metric_tokens(task_text, retrieval_intent)
    semantic_diagnostics = _semantic_admissibility(
        selected_values,
        retrieval_intent=retrieval_intent,
        task_text=task_text,
        operation=operation,
        target_years=set(years),
        metric_tokens=metric_tokens,
    )
    semantic_errors = _semantic_validation_errors(semantic_diagnostics)
    if semantic_errors:
        return _result_with_diagnostics(
            status="insufficient",
            operation=operation,
            retrieval_intent=retrieval_intent,
            years=years,
            validation_errors=semantic_errors,
            semantic_diagnostics=semantic_diagnostics,
            capability_source=capability_source,
            capability_signature=capability_signature,
            capability_validated=True,
        )

    refs = [_provenance_ref(item) for item in selected_values]
    ledger_step = OfficeQAComputeStep(
        operator="compute_capability",
        description=(
            f"Validated synthesized compute = {format_scalar_number(final_value)} "
            f"[{capability_source}:{capability_signature}]"
            + (f" ({explanation})" if explanation else "")
        ),
        inputs={
            "operation_signature": capability_signature,
            "capability_source": capability_source,
            "selected_value_count": len(selected_values),
            "analysis_modes": list(retrieval_intent.analysis_modes[:6]),
        },
        output={"value": final_value},
        provenance_refs=refs,
    )
    display_value, answer_unit_basis = _display_contract_value(
        final_value,
        task_text,
        retrieval_intent,
        selected_values=selected_values,
    )
    citations = []
    for ref in refs:
        citation = str(ref.get("citation", "") or "").strip()
        if citation and citation not in citations:
            citations.append(citation)
    return _result_with_diagnostics(
        status="ok",
        operation=operation,
        retrieval_intent=retrieval_intent,
        years=years,
        final_value=final_value,
        display_value=display_value,
        answer_unit_basis=answer_unit_basis,
        answer_text=_build_answer_text(operation, display_value, [ledger_step]),
        citations=citations,
        ledger=[ledger_step.model_dump()],
        semantic_diagnostics=semantic_diagnostics,
        provenance_complete=bool(refs),
        selection_reasoning=(
            f"Selected validated synthesized compute capability for operation signature '{capability_signature}' "
            "because native deterministic compute does not yet support the requested calculation directly."
        ),
        capability_source=capability_source,
        capability_signature=capability_signature,
        capability_validated=True,
    )
