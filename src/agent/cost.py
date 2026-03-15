"""
Cost Tracker: Per-Run Token, Latency & Cost Accounting
======================================================
Tracks every LLM call and MCP tool invocation during a graph run.
"""

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table: estimated $/1K tokens per model
# ---------------------------------------------------------------------------

MODEL_COST_TABLE: dict[str, dict[str, float]] = {
    # model_name -> {"input": $/1K tok, "output": $/1K tok}
    "gpt-4.1-mini":         {"input": 0.0004,  "output": 0.0016},
    "gpt-4.1":              {"input": 0.002,   "output": 0.008},
    "gpt-4o-mini":          {"input": 0.00015, "output": 0.0006},
    "gpt-4o":               {"input": 0.0025,  "output": 0.01},
    "openai/gpt-oss-20b":   {"input": 0.0,     "output": 0.0},   # free competition endpoint
}

# Fallback for unknown models
_DEFAULT_COST = {"input": 0.001, "output": 0.002}


def _get_cost_rates(model_name: str) -> dict[str, float]:
    return MODEL_COST_TABLE.get(model_name, _DEFAULT_COST)


# ---------------------------------------------------------------------------
# Trace dataclass
# ---------------------------------------------------------------------------

@dataclass
class OperatorTrace:
    """Record of a single operator invocation."""
    operator: str
    model_name: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    success: bool

    def to_dict(self) -> dict:
        return {
            "operator": self.operator,
            "model_name": self.model_name,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    """Accumulates cost and trace data for a single graph run."""

    model_name: str = "openai/gpt-oss-20b"
    traces: list[OperatorTrace] = field(default_factory=list)
    llm_calls: int = 0
    mcp_calls: int = 0
    _run_start: float = field(default_factory=time.monotonic)

    def record(
        self,
        operator: str,
        model_name: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> OperatorTrace:
        """Record an operator invocation and return its trace."""
        effective_model = model_name or self.model_name
        rates = _get_cost_rates(effective_model)
        cost = (tokens_in / 1000.0) * rates["input"] + \
               (tokens_out / 1000.0) * rates["output"]

        trace = OperatorTrace(
            operator=operator,
            model_name=effective_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
            success=success,
        )
        self.traces.append(trace)
        self.llm_calls += 1
        return trace

    def record_mcp_call(self) -> None:
        """Increment MCP tool call counter."""
        self.mcp_calls += 1

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens_in + t.tokens_out for t in self.traces)

    def total_cost(self) -> float:
        return sum(t.cost_usd for t in self.traces)

    @property
    def wall_clock_ms(self) -> float:
        return (time.monotonic() - self._run_start) * 1000.0

    def summary(self) -> dict:
        """Produce a summary dict for logging and dashboards."""
        return {
            "llm_calls": self.llm_calls,
            "mcp_calls": self.mcp_calls,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost(), 6),
            "wall_clock_ms": round(self.wall_clock_ms, 1),
            "models_used": sorted({t.model_name for t in self.traces}),
            "operators_used": [t.operator for t in self.traces],
            "any_failure": any(not t.success for t in self.traces),
        }

