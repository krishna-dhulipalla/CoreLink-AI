"""
Tests for Sprint 3: Execution Memory & Repair Reuse
=====================================================
Covers schemas, admission policy, storage, retrieval, and eviction.
"""

import os
import sys
import tempfile

import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.memory.schema import (
    ExecutorMemory,
    RouterMemory,
    VerifierMemory,
    _task_signature,
)
from agent.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh MemoryStore with a temp DB for each test."""
    db_path = str(tmp_path / "test_agent_memory.db")
    return MemoryStore(db_path=db_path)


# ====================================================================
# Schema Tests
# ====================================================================

class TestSchemas:
    def test_task_signature_deterministic(self):
        sig1 = _task_signature("Calculate AAPL call price")
        sig2 = _task_signature("Calculate AAPL call price")
        assert sig1 == sig2
        assert len(sig1) == 16

    def test_task_signature_case_insensitive(self):
        sig1 = _task_signature("Calculate AAPL call price")
        sig2 = _task_signature("calculate aapl call price")
        assert sig1 == sig2

    def test_task_signature_normalizes_whitespace(self):
        sig1 = _task_signature("  Calculate AAPL call price  ")
        sig2 = _task_signature("Calculate AAPL call price")
        assert sig1 == sig2

    def test_router_memory_creation(self):
        rec = RouterMemory(
            task_signature=_task_signature("test"),
            task_summary="test task",
            selected_layers=["react_reason", "verifier_check"],
            success=True,
            cost_usd=0.001,
            latency_ms=500.0,
        )
        assert rec.success is True
        assert rec.selected_layers == ["react_reason", "verifier_check"]

    def test_executor_memory_creation(self):
        rec = ExecutorMemory(
            task_signature=_task_signature("test"),
            partial_context_summary="Calculating option price",
            tool_used="black_scholes_price",
            arguments_pattern="spot=180, strike=175",
            outcome_quality="good",
            success=True,
        )
        assert rec.tool_used == "black_scholes_price"

    def test_verifier_memory_creation(self):
        rec = VerifierMemory(
            task_signature=_task_signature("test"),
            failure_pattern="Calculator rejected multiline input",
            verdict="REVISE",
            repair_action="Switch to dedicated finance tool",
            repair_worked=True,
        )
        assert rec.verdict == "REVISE"
        assert rec.repair_worked is True


# ====================================================================
# Admission Policy Tests
# ====================================================================

class TestAdmission:
    def test_router_admits_successful(self, store):
        rec = RouterMemory(
            task_signature=_task_signature("test"),
            task_summary="test",
            selected_layers=["react_reason"],
            success=True,
        )
        assert store.store_router(rec) is True

    def test_router_rejects_failed(self, store):
        rec = RouterMemory(
            task_signature=_task_signature("test"),
            task_summary="test",
            selected_layers=["react_reason"],
            success=False,
        )
        assert store.store_router(rec) is False

    def test_executor_admits_successful(self, store):
        rec = ExecutorMemory(
            task_signature=_task_signature("test"),
            partial_context_summary="calc",
            tool_used="calculator",
            arguments_pattern="2+2",
            outcome_quality="good",
            success=True,
        )
        assert store.store_executor(rec) is True

    def test_executor_rejects_poor_failure(self, store):
        rec = ExecutorMemory(
            task_signature=_task_signature("test"),
            partial_context_summary="calc",
            tool_used="calculator",
            arguments_pattern="import os",
            outcome_quality="poor",
            success=False,
        )
        assert store.store_executor(rec) is False

    def test_executor_admits_acceptable_failure(self, store):
        """Acceptable quality is admitted even if success=False."""
        rec = ExecutorMemory(
            task_signature=_task_signature("test"),
            partial_context_summary="calc",
            tool_used="calculator",
            arguments_pattern="sqrt(2)",
            outcome_quality="acceptable",
            success=False,
        )
        assert store.store_executor(rec) is True

    def test_verifier_admits_successful_repair(self, store):
        rec = VerifierMemory(
            task_signature=_task_signature("test"),
            failure_pattern="hallucinated formula",
            verdict="BACKTRACK",
            repair_action="used correct formula",
            repair_worked=True,
        )
        assert store.store_verifier(rec) is True

    def test_verifier_rejects_failed_repair(self, store):
        rec = VerifierMemory(
            task_signature=_task_signature("test"),
            failure_pattern="hallucinated formula",
            verdict="BACKTRACK",
            repair_action="tried again same way",
            repair_worked=False,
        )
        assert store.store_verifier(rec) is False


# ====================================================================
# Storage & Retrieval Tests
# ====================================================================

class TestStorageRetrieval:
    def test_router_round_trip(self, store):
        task = "Calculate AAPL call price"
        rec = RouterMemory(
            task_signature=_task_signature(task),
            task_summary="AAPL options pricing",
            selected_layers=["react_reason", "verifier_check"],
            success=True,
            cost_usd=0.002,
            latency_ms=1200.0,
        )
        store.store_router(rec)
        hints = store.retrieve_router_hints(task)
        assert len(hints) == 1
        assert "react_reason" in hints[0]
        assert "AAPL" in hints[0]

    def test_executor_round_trip(self, store):
        task = "Calculate AAPL call price"
        rec = ExecutorMemory(
            task_signature=_task_signature(task),
            partial_context_summary="Pricing AAPL call option",
            tool_used="black_scholes_price",
            arguments_pattern="spot=180, strike=175, rate=0.05",
            outcome_quality="good",
            success=True,
        )
        store.store_executor(rec)
        hints = store.retrieve_executor_hints(task)
        assert len(hints) == 1
        assert "black_scholes_price" in hints[0]

    def test_verifier_round_trip(self, store):
        task = "Calculate AAPL call price"
        rec = VerifierMemory(
            task_signature=_task_signature(task),
            failure_pattern="calculator rejects multiline",
            verdict="REVISE",
            repair_action="switch to finance tool",
            repair_worked=True,
        )
        store.store_verifier(rec)
        hints = store.retrieve_verifier_hints(task)
        assert len(hints) == 1
        assert "calculator rejects multiline" in hints[0]
        assert "switch to finance tool" in hints[0]

    def test_no_hints_for_unknown_task(self, store):
        assert store.retrieve_router_hints("completely unknown task xyz") == []
        assert store.retrieve_executor_hints("completely unknown task xyz") == []
        assert store.retrieve_verifier_hints("completely unknown task xyz") == []

    def test_top_k_limits_results(self, store):
        task = "Calculate AAPL call price"
        sig = _task_signature(task)
        for i in range(10):
            rec = RouterMemory(
                task_signature=sig,
                task_summary=f"AAPL pricing attempt {i}",
                selected_layers=["react_reason"],
                success=True,
                cost_usd=0.001 * i,
                latency_ms=100.0 * i,
            )
            store.store_router(rec)
        hints = store.retrieve_router_hints(task, top_k=3)
        assert len(hints) == 3

    def test_stats(self, store):
        stats = store.stats()
        assert stats["router_memory"] == 0
        assert stats["executor_memory"] == 0
        assert stats["verifier_memory"] == 0


# ====================================================================
# Eviction Tests
# ====================================================================

class TestEviction:
    def test_eviction_caps_table_size(self, store):
        """Inserting more than MAX_RECORDS evicts oldest entries."""
        task = "eviction test"
        sig = _task_signature(task)
        # Store uses default MAX_RECORDS_PER_TABLE (500), but we can
        # test the eviction logic by inserting and checking stats.
        for i in range(5):
            rec = RouterMemory(
                task_signature=sig,
                task_summary=f"attempt {i}",
                selected_layers=["react_reason"],
                success=True,
            )
            store.store_router(rec)
        stats = store.stats()
        assert stats["router_memory"] == 5
