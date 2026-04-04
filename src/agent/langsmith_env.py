from __future__ import annotations

import os


def _is_truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_langsmith_env() -> None:
    project_name = os.getenv("LANGSMITH_PROJECT_NAME", "").strip()
    if project_name and not os.getenv("LANGSMITH_PROJECT", "").strip():
        os.environ["LANGSMITH_PROJECT"] = project_name
    project = os.getenv("LANGSMITH_PROJECT", "").strip()
    if project and not os.getenv("LANGCHAIN_PROJECT", "").strip():
        os.environ["LANGCHAIN_PROJECT"] = project
    if _is_truthy(os.getenv("LANGSMITH_TRACING", "")) and not os.getenv("LANGCHAIN_TRACING_V2", "").strip():
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
