"""Run tracer with request-scoped state."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
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


def _trace_retention_limit() -> int:
    raw = os.getenv("TRACE_MAX_RECENT", "5").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 5
    return max(1, value)


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
        summary = dict(cost_summary or {})
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

        if isinstance(summary.get("llm_calls"), int):
            llm_calls = int(summary.get("llm_calls", 0) or 0)
        if isinstance(summary.get("mcp_calls"), int):
            tool_calls = int(summary.get("mcp_calls", 0) or 0)

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
            "cost_summary": summary,
            "budget_summary": budget_summary or {},
            "execution_summary": _make_readable(_execution_summary(self._nodes)),
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


def _short_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _tool_results_summary(tool_results: Any) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in list(tool_results or [])[:6]:
        if not isinstance(item, dict):
            continue
        facts = dict(item.get("facts") or {})
        metadata = dict(facts.get("metadata") or {})
        entry: dict[str, Any] = {
            "tool": str(item.get("type", "") or item.get("tool_name", "") or item.get("tool", "") or ""),
        }
        retrieval_status = str(item.get("retrieval_status", "") or "")
        if retrieval_status:
            entry["retrieval_status"] = retrieval_status
        officeqa_status = str(metadata.get("officeqa_status", "") or "")
        if officeqa_status:
            entry["officeqa_status"] = officeqa_status
        if facts.get("document_id"):
            entry["document_id"] = str(facts.get("document_id", ""))
        if facts.get("citation"):
            entry["citation"] = str(facts.get("citation", ""))
        if isinstance(facts.get("tables"), list):
            entry["table_count"] = len(facts.get("tables", []))
        if isinstance(facts.get("chunks"), list):
            entry["chunk_count"] = len(facts.get("chunks", []))
        if isinstance(facts.get("cells"), list):
            entry["cell_count"] = len(facts.get("cells", []))
        summary.append(entry)
    return summary


def _execution_summary(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for entry in nodes:
        if not isinstance(entry, dict):
            continue
        node = str(entry.get("node", "") or "")
        item: dict[str, Any] = {"node": node, "ts": entry.get("ts", "")}
        if entry.get("stop_reason"):
            item["stop_reason"] = str(entry.get("stop_reason", ""))
        if node == "executor":
            tools_ran = [str(tool) for tool in list(entry.get("tools_ran") or [])[:8] if str(tool).strip()]
            if tools_ran:
                item["tools_ran"] = tools_ran
            if entry.get("used_llm") is not None:
                item["used_llm"] = bool(entry.get("used_llm"))
            if entry.get("llm_decision_reason"):
                item["llm_decision_reason"] = _short_text(entry.get("llm_decision_reason", ""))
            repair_history = [repair for repair in list(entry.get("llm_repair_history") or []) if isinstance(repair, dict)]
            if repair_history:
                last_repair = dict(repair_history[-1])
                decision = dict(last_repair.get("decision") or {})
                item["llm_repair"] = {
                    key: value
                    for key, value in {
                        "count": len(repair_history),
                        "stage": last_repair.get("stage", ""),
                        "trigger": last_repair.get("trigger", ""),
                        "path_changed": bool(last_repair.get("path_changed")),
                        "decision": decision.get("decision", ""),
                        "confidence": decision.get("confidence", 0.0),
                    }.items()
                    if value not in ("", [], {}, None)
                }
            llm_usage = [dict(item) for item in list(entry.get("officeqa_llm_usage") or []) if isinstance(item, dict)]
            if llm_usage:
                item["llm_usage"] = [
                    {
                        key: value
                        for key, value in {
                            "category": record.get("category", ""),
                            "reason": record.get("reason", ""),
                            "model_name": record.get("model_name", ""),
                            "applied": record.get("applied"),
                        }.items()
                        if value not in ("", [], {}, None)
                    }
                    for record in llm_usage[-4:]
                ]
            retrieval = dict(entry.get("retrieval_decision") or entry.get("retrieval_action") or {})
            if retrieval:
                item["retrieval"] = {
                    key: value
                    for key, value in {
                        "tool_name": retrieval.get("tool_name", ""),
                        "stage": retrieval.get("stage", ""),
                        "strategy": retrieval.get("strategy", ""),
                        "document_id": retrieval.get("document_id", ""),
                        "path": retrieval.get("path", ""),
                        "query": _short_text(retrieval.get("query", ""), 140),
                    }.items()
                    if value not in ("", [], {}, None)
                }
            if entry.get("strategy_reason"):
                item["strategy_reason"] = _short_text(entry.get("strategy_reason", ""))
            candidate_sources = [source for source in list(entry.get("candidate_sources") or []) if isinstance(source, dict)]
            if candidate_sources:
                item["candidate_source_count"] = len(candidate_sources)
                top_candidate = {
                    key: value
                    for key, value in {
                        "title": candidate_sources[0].get("title", ""),
                        "document_id": candidate_sources[0].get("document_id", ""),
                        "score": candidate_sources[0].get("score"),
                    }.items()
                    if value not in ("", [], {}, None)
                }
                if top_candidate:
                    item["top_candidate"] = top_candidate
            rejected_candidates = [candidate for candidate in list(entry.get("rejected_candidates") or []) if isinstance(candidate, dict)]
            if rejected_candidates:
                item["rejected_candidate_count"] = len(rejected_candidates)
            if entry.get("aggregation_reason"):
                item["aggregation_reason"] = _short_text(entry.get("aggregation_reason", ""))
            if entry.get("evidence_gaps"):
                item["evidence_gaps"] = list(entry.get("evidence_gaps", []))[:6]
            tool_summary = _tool_results_summary(entry.get("tool_results"))
            if tool_summary:
                item["tool_results"] = tool_summary
        elif node == "reviewer":
            if entry.get("used_llm") is not None:
                item["used_llm"] = bool(entry.get("used_llm"))
            if entry.get("llm_decision_reason"):
                item["llm_decision_reason"] = _short_text(entry.get("llm_decision_reason", ""))
            validator = dict(entry.get("validator_result") or {})
            if validator:
                item["validator"] = {
                    key: value
                    for key, value in {
                        "status": validator.get("status", ""),
                        "remediation_codes": list(validator.get("remediation_codes", []))[:6],
                        "orchestration_strategy": validator.get("orchestration_strategy", ""),
                        "stop_reason": validator.get("stop_reason", ""),
                    }.items()
                    if value not in ("", [], {}, None)
                }
        elif node == "solver":
            if entry.get("llm_call") is not None:
                item["llm_call"] = bool(entry.get("llm_call"))
            if entry.get("output_preview"):
                item["output_preview"] = _short_text(entry.get("output_preview", ""), 220)
        elif node == "output_adapter":
            if entry.get("output_preview"):
                item["output_preview"] = _short_text(entry.get("output_preview", ""), 220)
        elif node == "self_reflection":
            if entry.get("used_llm") is not None:
                item["used_llm"] = bool(entry.get("used_llm"))
            if entry.get("decision"):
                item["decision"] = _short_text(entry.get("decision", ""))
        summary.append(item)
    return summary


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


def _write_trace_file_sync(payload: dict[str, Any], profile: str, trace_identity: dict[str, Any]) -> str | None:
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
        _cleanup_old_traces()
        logger.info("[RunTracer] Saved trace to %s", filepath)
        return str(filepath)
    except Exception as exc:
        logger.warning("[RunTracer] Failed to save trace: %s", exc)
        return None


def _cleanup_old_traces() -> None:
    if not _TRACES_DIR.exists():
        return
    limit = _trace_retention_limit()
    entries = [entry for entry in _TRACES_DIR.iterdir() if entry.exists()]
    if len(entries) <= limit:
        return
    entries.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    active_session_path = _active_session.folder_path.resolve() if _active_session is not None else None
    retained: list[Path] = []
    stale_entries: list[Path] = []
    for entry in entries:
        resolved = entry.resolve()
        if active_session_path is not None and resolved == active_session_path:
            retained.append(entry)
            continue
        if len(retained) < limit:
            retained.append(entry)
            continue
        stale_entries.append(entry)
    for stale in stale_entries:
        try:
            if not stale.exists():
                continue
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("[RunTracer] Failed to evict stale trace artifact %s: %s", stale, exc)


def _write_trace_file(payload: dict[str, Any], profile: str, trace_identity: dict[str, Any]) -> str | None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _write_trace_file_sync(payload, profile, trace_identity)

    thread = threading.Thread(
        target=_write_trace_file_sync,
        args=(dict(payload), profile, dict(trace_identity)),
        name="run-tracer-writer",
        daemon=True,
    )
    thread.start()
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
