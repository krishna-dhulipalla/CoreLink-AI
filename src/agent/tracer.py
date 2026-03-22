"""RunTracer — Lightweight per-run trace files
============================================
Captures one structured JSON file per graph run with only the debugging-
critical details for each node: profile decisions, template selection,
evidence pack, exact LLM prompts/outputs, reviewer verdicts, tool calls,
and cost.

Enable with  ENABLE_RUN_TRACER=1  in .env.
Files are saved to  traces/  in the project root.

Benchmark session grouping
--------------------------
Call ``start_session()`` before a multi-task benchmark run.
All subsequent traces will be saved into a single folder:
  traces/2026-03-21_21-10-44/task_001.json
  traces/2026-03-21_21-10-44/task_002.json
  ...
Call ``end_session()`` when the benchmark finishes.
If no session is active, traces save to ``traces/`` as before.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

_CST = timezone(timedelta(hours=-6))  # US Central Standard Time

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/agent → src → root
_TRACES_DIR = _PROJECT_ROOT / "traces"

# ──────────────────────────── singleton ────────────────────────────
_active_tracer: RunTracer | None = None


# ──────────────────────── session state ────────────────────────
class _Session:
    """Tracks a multi-task benchmark session for folder grouping."""

    def __init__(self) -> None:
        dt = datetime.now(_CST)
        self.folder_name: str = dt.strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_path: Path = _TRACES_DIR / self.folder_name
        self.task_counter: int = 0

    def next_task_number(self) -> int:
        self.task_counter += 1
        return self.task_counter


_active_session: _Session | None = None


def _tracer_enabled() -> bool:
    return os.getenv("ENABLE_RUN_TRACER", "").strip().lower() in {"1", "true", "yes", "on"}


class RunTracer:
    """Collects per-node records during a single graph run."""

    def __init__(self) -> None:
        self._run_id: str = ""
        self._start_time: float = time.monotonic()
        self._start_dt: str = datetime.now(_CST).strftime("%Y-%m-%dT%H:%M:%S")
        self._task_preview: str = ""
        self._profile: str = ""
        self._template_id: str = ""
        self._complexity_tier: str = ""
        self._nodes: list[dict[str, Any]] = []

    # ── public recording API ──

    def set_task(self, task_text: str, run_id: str = "") -> None:
        self._task_preview = (task_text or "")[:200]
        self._run_id = run_id

    def record(self, node: str, data: dict[str, Any]) -> None:
        """Append a node record with a timestamp."""
        entry: dict[str, Any] = {
            "node": node,
            "ts": datetime.now(_CST).strftime("%H:%M:%S.%f")[:-3],
        }
        entry.update(data)
        self._nodes.append(entry)

        # Cache top-level summaries so the header doesn't need a second pass.
        if node == "fast_path_gate":
            if not self._profile:
                self._profile = data.get("task_family", "")
            if not self._template_id and data.get("execution_mode"):
                self._template_id = data.get("execution_mode", "")
            if not self._complexity_tier:
                self._complexity_tier = data.get("complexity_tier", "")
        elif node == "task_planner":
            intent = data.get("intent", {})
            if intent and isinstance(intent, dict):
                if not self._profile:
                    self._profile = intent.get("task_family", "")
                if not self._template_id:
                    self._template_id = data.get("template_id", "") or intent.get("execution_mode", "")
                if not self._complexity_tier:
                    self._complexity_tier = intent.get("complexity_tier", "")
        elif node == "executor":
            intent = data.get("intent", {})
            if intent and isinstance(intent, dict):
                if not self._profile:
                    self._profile = intent.get("task_family", "")
                if not self._template_id:
                    self._template_id = intent.get("execution_mode", "")
                if not self._complexity_tier:
                    self._complexity_tier = intent.get("complexity_tier", "")

    def finalize(self, final_answer: str = "", cost_summary: dict | None = None, budget_summary: dict | None = None) -> str | None:
        """Build the final JSON payload and write it to disk. Returns the file path or None."""
        duration = round(time.monotonic() - self._start_time, 2)

        # Aggregate token counts from all nodes
        total_prompt = 0
        total_completion = 0
        llm_calls = 0
        tool_calls = 0
        for entry in self._nodes:
            tokens = entry.get("tokens")
            if tokens and isinstance(tokens, dict):
                total_prompt += tokens.get("prompt", 0)
                total_completion += tokens.get("completion", 0)
            # Count LLM calls from solver, reviewer, self_reflection, and executor
            if entry.get("node") == "solver" and entry.get("llm_call"):
                llm_calls += 1
            elif entry.get("node") in {"reviewer", "self_reflection"} and entry.get("used_llm"):
                llm_calls += 1
            elif entry.get("node") == "executor" and entry.get("used_llm"):
                llm_calls += 1
            if entry.get("node") == "executor":
                tools_ran = entry.get("tools_ran")
                if tools_ran and isinstance(tools_ran, list):
                    tool_calls += len(tools_ran)

        payload: dict[str, Any] = {
            "run_id": self._run_id,
            "timestamp": self._start_dt,
            "duration_seconds": duration,
            "task_preview": self._task_preview,
            "final_profile": self._profile,
            "final_template": self._template_id,
            "complexity_tier": self._complexity_tier,
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

        return _write_trace_file(payload, self._profile, self._start_dt)


# ──────────────────────── helpers for LLM message capture ─────────────────

def _make_readable(value: Any) -> Any:
    """Split long strings into arrays of lines for vertical JSON formatting."""
    if isinstance(value, str) and ("\n" in value or len(value) > 300):
        return value.split("\n")
    if isinstance(value, dict):
        return {k: _make_readable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_readable(item) for item in value]
    return value


def format_messages_for_trace(messages: list) -> list[dict[str, Any]]:
    """Convert LangChain messages into a compact list of {role, content} dicts.
    Long content is split into line arrays for readability."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "").lower()
        content = str(getattr(msg, "content", ""))
        # Split long content into lines so JSON indents vertically
        if "\n" in content or len(content) > 300:
            result.append({"role": role, "content": content.split("\n")})
        else:
            result.append({"role": role, "content": content})
    return result


# ──────────────────────── file writer ─────────────────────────

def _write_trace_file(payload: dict[str, Any], profile: str, start_dt: str) -> str | None:
    """Write the trace JSON to traces/ directory (or session subfolder)."""
    global _active_session
    try:
        if _active_session is not None:
            # Session mode: save as task_NNN.json inside the session folder
            target_dir = _active_session.folder_path
            target_dir.mkdir(parents=True, exist_ok=True)
            task_num = _active_session.next_task_number()
            filename = f"task_{task_num:03d}.json"
        else:
            # Standalone mode: save to traces/ with date_profile_time.json
            target_dir = _TRACES_DIR
            target_dir.mkdir(parents=True, exist_ok=True)
            dt = datetime.now(_CST)
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H-%M-%S")
            safe_profile = (profile or "general").replace(" ", "_")
            filename = f"{date_str}_{safe_profile}_{time_str}.json"

        filepath = target_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        logger.info("[RunTracer] Saved trace to %s", filepath)
        return str(filepath)
    except Exception as exc:
        logger.warning("[RunTracer] Failed to save trace: %s", exc)
        return None


# ──────────────────────── singleton access ─────────────────────

def get_tracer() -> RunTracer | None:
    """Return the active tracer if enabled, else None."""
    global _active_tracer
    if _active_tracer is not None:
        return _active_tracer
    return None


def start_tracer() -> RunTracer | None:
    """Create a new RunTracer instance if tracing is enabled.

    In benchmark mode (BENCHMARK_STATELESS=1), auto-starts a session folder
    on the first call so all tasks land in one dated folder.
    """
    global _active_tracer, _active_session
    if not _tracer_enabled():
        _active_tracer = None
        return None
    # Auto-start session for benchmark runs
    if _active_session is None and os.getenv("BENCHMARK_STATELESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        start_session()
    _active_tracer = RunTracer()
    return _active_tracer


def finalize_tracer(final_answer: str = "", cost_summary: dict | None = None, budget_summary: dict | None = None) -> str | None:
    """Finalize and save the current trace, then clear the singleton."""
    global _active_tracer
    if _active_tracer is None:
        return None
    path = _active_tracer.finalize(final_answer, cost_summary, budget_summary)
    _active_tracer = None
    return path


# ──────────────────────── session management ──────────────────

def start_session() -> str:
    """Start a benchmark session. All subsequent traces go into one folder.

    Returns the session folder name (e.g. '2026-03-21_21-10-44').
    """
    global _active_session
    _active_session = _Session()
    _active_session.folder_path.mkdir(parents=True, exist_ok=True)
    logger.info("[RunTracer] Session started → %s", _active_session.folder_path)
    return _active_session.folder_name


def end_session() -> dict[str, Any] | None:
    """End the current benchmark session. Returns a summary dict or None."""
    global _active_session
    if _active_session is None:
        return None
    summary = {
        "session_folder": _active_session.folder_name,
        "total_tasks": _active_session.task_counter,
        "path": str(_active_session.folder_path),
    }
    logger.info("[RunTracer] Session ended — %d tasks in %s", _active_session.task_counter, _active_session.folder_path)
    _active_session = None
    return summary


def get_session_info() -> dict[str, Any] | None:
    """Return current session info without ending it."""
    if _active_session is None:
        return None
    return {
        "session_folder": _active_session.folder_name,
        "tasks_so_far": _active_session.task_counter,
        "path": str(_active_session.folder_path),
    }
