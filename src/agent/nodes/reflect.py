"""
Reflect Node
============
Post-run node. Memory remains store-only in runner for v1.
"""

from __future__ import annotations

import logging

from agent.runtime_clock import increment_runtime_step
from agent.state import AgentState

logger = logging.getLogger(__name__)


def reflect(state: AgentState) -> dict:
    step = increment_runtime_step()
    workpad = dict(state.get("workpad", {}))
    workpad.setdefault("events", []).append({"node": "reflect", "action": "run complete"})
    logger.info("[Step %s] reflect -> complete", step)
    return {"workpad": workpad}
