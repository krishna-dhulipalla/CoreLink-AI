"""Benchmark adapter primitives."""

from __future__ import annotations

import os


def truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def benchmark_name_from_env() -> str:
    return os.getenv("BENCHMARK_NAME", "").strip().lower()
