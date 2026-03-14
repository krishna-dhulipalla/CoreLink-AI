"""
Runtime Step Clock
==================
Small per-run counter used for logging staged-node progress.
"""

_step_counter = 0


def reset_runtime_steps() -> None:
    global _step_counter
    _step_counter = 0


def increment_runtime_step() -> int:
    global _step_counter
    _step_counter += 1
    return _step_counter
