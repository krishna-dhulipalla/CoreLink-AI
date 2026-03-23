"""Run tracer with request-scoped state."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent.contracts import TraceIdentity

_CST = timezone(timedelta(hours=-6))
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRACES_DIR = _PROJECT_ROOT / "traces"


class _Session:
    """Tracks a multi-task benchmark session for folder grouping."""

    def __init__(self) -> None:
        dt = datetime.now(_CST)
        self.folder_name = dt.strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_path = _TRACES_DIR / self.folder_name
        self.task_counter = 0
        self._lock = threading.Lock()

    def next_task_number(self) -> int:
        with self._lock:
            self.task_counter += 1
            return self.task_counter


_active_tracer_var: ContextVar["RunTracer | None"] = ContextVar("active_run_tracer", default=None)
_active_session: _Session | None = None
_session_lock = threading.Lock()


def _tracer_enabled() -> bool:
    return os.getenv("ENABLE_RUN_TRACER", "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in (value or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned[:80]


def _trace_identity(raw: dict[str, Any] | None) -> TraceIdentity:
    return TraceIdentity.model_validate(raw or {})


class RunTracer:
    """Collects per-node records during a single graph run."""

    def __init__(self, trace_identity: dict[str, Any] | None = None) -> None:
        self._run_id = ""
        self._start_time = time.monotonic()
        self._start_dt = datetime.now(_CST).strftime("%Y-%m-%dT%H:%M:%S")
        self._task_preview = ""
        self._profile = ""
        self._template_id = ""
        self._complexity_tier = ""
        self._nodes: list[dict[str, Any]] = []
        self._stop_reason = ""
        self._trace_identity = _trace_identity(trace_identity).model_dump()

    def set_task(self, task_text: str, run_id: str = "") -> None:
        self._task_preview = (task_text or "")[:200]
        self._run_id = run_id

    def record(self, node: str, data: dict[str, Any]) -> None:
        entry: dict[str, Any] = {
            "node": node,
            "ts": datetime.now(_CST).strftime("%H:%M:%S.%f")[:-3],
        }
        entry.update(data)
        self._nodes.append(entry)

        if not self._stop_reason and data.get("stop_reason"):
            self._stop_reason = str(data.get("stop_reason", ""))

        if node == "fast_path_gate":
            if not self._profile:
                self._profile = data.get("task_family", "")
            if not self._template_id and data.get("execution_mode"):
                self._template_id = data.get("execution_mode", "")
            if not self._complexity_tier:
                self._complexity_tier = data.get("complexity_tier", "")
        elif node == "task_planner":
            intent = data.get("intent", {})
            if isinstance(intent, dict):
                if not self._profile:
                    self._profile = intent.get("task_family", "")
                if not self._template_id:
                    self._template_id = data.get("template_id", "") or intent.get("execution_mode", "")
                if not self._complexity_tier:
                    self._complexity_tier = intent.get("complexity_tier", "")
        elif node == "executor":
            intent = data.get("intent", {})
            if isinstance(intent, dict):
                if not self._profile:
                    self._profile = intent.get("task_family", "")
                if not self._template_id:
                    self._template_id = intent.get("execution_mode", "")
                if not self._complexity_tier:
                    self._complexity_tier = intent.get("complexity_tier", "")

    def finalize(self, final_answer: str = "", cost_summary: dict | None = None, budget_summary: dict | None = None) -> str | None:
        duration = round(time.monotonic() - self._start_time, 2)
        total_prompt = 0
        total_completion = 0
        llm_calls = 0
        tool_calls = 0
        for entry in self._nodes:
            tokens = entry.get("tokens")
            if isinstance(tokens, dict):
                total_prompt += tokens.get("prompt", 0)
                total_completion += tokens.get("completion", 0)
            if entry.get("node") == "solver" and entry.get("llm_call"):
                llm_calls += 1
            elif entry.get("node") in {"reviewer", "self_reflection"} and entry.get("used_llm"):
                llm_calls += 1
            elif entry.get("node") == "executor" and entry.get("used_llm"):
                llm_calls += 1
            if entry.get("node") == "executor":
                tools_ran = entry.get("tools_ran")
                if isinstance(tools_ran, list):
                    tool_calls += len(tools_ran)

        payload: dict[str, Any] = {
            "run_id": self._run_id,
            "timestamp": self._start_dt,
            "duration_seconds": duration,
            "task_preview": self._task_preview,
            "final_profile": self._profile,
            "final_template": self._template_id,
            "complexity_tier": self._complexity_tier,
            "stop_reason": self._stop_reason,
            "request_id": self._trace_identity.get("request_id", ""),
            "task_id": self._trace_identity.get("task_id", ""),
            "context_id": self._trace_identity.get("context_id", ""),
            "benchmark_uid": self._trace_identity.get("benchmark_uid", ""),
            "total_llm_calls": llm_calls,
            "total_tool_calls": tool_calls,
            "total_tokens": {
                "prompt": total_prompt,
                "completion": total_completion,
                "total": total_prompt + total_completion,
            },
            "final_answer_preview": (final_answer or "")[:4000],
            "cost_summary": cost_summary or {},
            "budget_summary": budget_summary or {},
            "nodes": _make_readable(self._nodes),
        }
        return _write_trace_file(payload, self._profile, self._trace_identity)


def _make_readable(value: Any) -> Any:
    if isinstance(value, str) and ("\n" in value or len(value) > 300):
        return value.split("\n")
    if isinstance(value, dict):
        return {k: _make_readable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_readable(item) for item in value]
    return value


def format_messages_for_trace(messages: list) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "").lower()
        content = str(getattr(msg, "content", ""))
        if "\n" in content or len(content) > 300:
            result.append({"role": role, "content": content.split("\n")})
        else:
            result.append({"role": role, "content": content})
    return result


def _ensure_session() -> _Session | None:
    global _active_session
    if os.getenv("BENCHMARK_STATELESS", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    with _session_lock:
        if _active_session is None:
            _active_session = _Session()
            _active_session.folder_path.mkdir(parents=True, exist_ok=True)
            logger.info("[RunTracer] Session started -> %s", _active_session.folder_path)
        return _active_session


def _write_trace_file(payload: dict[str, Any], profile: str, trace_identity: dict[str, Any]) -> str | None:
    global _active_session
    try:
        session = _ensure_session()
        safe_profile = (profile or "general").replace(" ", "_")
        label = (
            trace_identity.get("benchmark_uid")
            or trace_identity.get("task_id")
            or trace_identity.get("request_id")
            or ""
        )
        safe_label = _safe_slug(str(label))
        if session is not None:
            target_dir = session.folder_path
            target_dir.mkdir(parents=True, exist_ok=True)
            task_num = session.next_task_number()
            if safe_label:
                filename = f"task_{task_num:03d}__{safe_label}.json"
            else:
                filename = f"task_{task_num:03d}.json"
        else:
            target_dir = _TRACES_DIR
            target_dir.mkdir(parents=True, exist_ok=True)
            dt = datetime.now(_CST)
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H-%M-%S")
            suffix = f"_{safe_label}" if safe_label else ""
            filename = f"{date_str}_{safe_profile}_{time_str}{suffix}.json"

        filepath = target_dir / filename
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        logger.info("[RunTracer] Saved trace to %s", filepath)
        return str(filepath)
    except Exception as exc:
        logger.warning("[RunTracer] Failed to save trace: %s", exc)
        return None


def get_tracer() -> RunTracer | None:
    return _active_tracer_var.get()


def start_tracer(trace_identity: dict[str, Any] | None = None) -> RunTracer | None:
    if not _tracer_enabled():
        _active_tracer_var.set(None)
        return None
    tracer = RunTracer(trace_identity=trace_identity)
    _active_tracer_var.set(tracer)
    _ensure_session()
    return tracer


def finalize_tracer(final_answer: str = "", cost_summary: dict | None = None, budget_summary: dict | None = None) -> str | None:
    tracer = _active_tracer_var.get()
    if tracer is None:
        return None
    path = tracer.finalize(final_answer, cost_summary, budget_summary)
    _active_tracer_var.set(None)
    return path


def write_preflight_failure_trace(
    task_text: str,
    error: str,
    *,
    profile: str = "preflight_failure",
    node: str = "executor",
    details: dict[str, Any] | None = None,
    trace_identity: dict[str, Any] | None = None,
) -> str | None:
    if not _tracer_enabled():
        return None
    _ensure_session()
    identity = _trace_identity(trace_identity or details or {}).model_dump()
    payload: dict[str, Any] = {
        "run_id": "",
        "timestamp": datetime.now(_CST).strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": 0.0,
        "task_preview": (task_text or "")[:200],
        "final_profile": profile,
        "final_template": "",
        "complexity_tier": "",
        "stop_reason": "preflight_failure",
        "request_id": identity.get("request_id", ""),
        "task_id": identity.get("task_id", ""),
        "context_id": identity.get("context_id", ""),
        "benchmark_uid": identity.get("benchmark_uid", ""),
        "total_llm_calls": 0,
        "total_tool_calls": 0,
        "total_tokens": {"prompt": 0, "completion": 0, "total": 0},
        "final_answer_preview": f"Agent error: {error}",
        "cost_summary": {},
        "budget_summary": {},
        "nodes": _make_readable(
            [
                {
                    "node": node,
                    "ts": datetime.now(_CST).strftime("%H:%M:%S.%f")[:-3],
                    "preflight_failure": True,
                    "error": error,
                    **(details or {}),
                }
            ]
        ),
    }
    return _write_trace_file(payload, profile, identity)


def start_session() -> str:
    session = _ensure_session()
    return session.folder_name if session is not None else ""


def end_session() -> dict[str, Any] | None:
    global _active_session
    with _session_lock:
        if _active_session is None:
            return None
        summary = {
            "session_folder": _active_session.folder_name,
            "total_tasks": _active_session.task_counter,
            "path": str(_active_session.folder_path),
        }
        logger.info("[RunTracer] Session ended - %d tasks in %s", _active_session.task_counter, _active_session.folder_path)
        _active_session = None
        return summary


def get_session_info() -> dict[str, Any] | None:
    if _active_session is None:
        return None
    return {
        "session_folder": _active_session.folder_name,
        "tasks_so_far": _active_session.task_counter,
        "path": str(_active_session.folder_path),
    }
