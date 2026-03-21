"""Runtime version selection helpers."""

from __future__ import annotations

import os


def get_runtime_version() -> str:
    value = os.getenv("AGENT_RUNTIME_VERSION", "v3").strip().lower()
    if value not in {"v3", "v4"}:
        return "v3"
    return value


def use_v4_runtime() -> bool:
    return get_runtime_version() == "v4"
