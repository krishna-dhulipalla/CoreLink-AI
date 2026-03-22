"""
RunTracer — Lightweight per-run trace files
============================================
Captures one structured JSON file per graph run with only the debugging-
critical details for each node: profile decisions, template selection,
evidence pack, exact LLM prompts/outputs, reviewer verdicts, tool calls,
and cost.

Enable with  ENABLE_RUN_TRACER=1  in .env.
Files are saved to  traces/  in the project root.
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

        # Cache top-level summaries so the header doesn't need a second pass
        if node == "task_profiler":
            self._profile = data.get("profile", "")
            self._complexity_tier = data.get("complexity_tier", "")
        elif node == "template_selector":
            self._template_id = data.get("template_id", "")
        elif node == "fast_path_gate":
            if not self._profile:
                self._profile = data.get("task_family", "")
            if not self._template_id and data.get("execution_mode"):
                self._template_id = f"v4_{data.get('execution_mode', '')}"
            if not self._complexity_tier:
                self._complexity_tier = data.get("complexity_tier", "")
        elif node == "task_planner":
            intent = data.get("intent", {})
            if intent and isinstance(intent, dict):
                if not self._profile:
                    self._profile = intent.get("task_family", "")
                if not self._template_id:
                    self._template_id = data.get("template_id", "") or f"v4_{intent.get('execution_mode', '')}"
                if not self._complexity_tier:
                    self._complexity_tier = intent.get("complexity_tier", "")
        elif node == "v4_executor":
            intent = data.get("intent", {})
            if intent and isinstance(intent, dict):
                if not self._profile:
                    self._profile = intent.get("task_family", "")
                if not self._template_id:
                    self._template_id = f"v4_{intent.get('execution_mode', '')}"
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
            # Count LLM calls from solver, reviewer, self_reflection, and v4_executor
            if entry.get("node") == "solver" and entry.get("llm_call"):
                llm_calls += 1
            elif entry.get("node") in {"reviewer", "self_reflection"} and entry.get("used_llm"):
                llm_calls += 1
            elif entry.get("node") == "v4_executor" and entry.get("used_llm"):
                llm_calls += 1
            # Count tool calls from tool_runner (V3) or v4_executor tool_results (V4)
            if entry.get("node") == "tool_runner":
                tool_calls += 1
            elif entry.get("node") == "v4_executor":
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
    """Write the trace JSON to traces/ directory."""
    try:
        _TRACES_DIR.mkdir(parents=True, exist_ok=True)
        # Filename: date_profile_time.json  e.g. 2026-03-21_finance_quant_00-06-44.json
        dt = datetime.now(_CST)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H-%M-%S")
        safe_profile = (profile or "general").replace(" ", "_")
        filename = f"{date_str}_{safe_profile}_{time_str}.json"
        filepath = _TRACES_DIR / filename

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
    """Create a new RunTracer instance if tracing is enabled."""
    global _active_tracer
    if not _tracer_enabled():
        _active_tracer = None
        return None
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
