"""
Memory Store (Sprint 3)
=======================
Bounded SQLite-backed store for role-specific execution memory.

Key design decisions:
- SQLite for simplicity and zero-dependency persistence.
- Strict admission policy: only verified-successful or high-signal repair records.
- Bounded: each table is capped at MAX_RECORDS; oldest entries are evicted.
- Retrieval returns compact structured hints, never raw transcript dumps.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from agent.memory.schema import (
    ExecutorMemory,
    RouterMemory,
    VerifierMemory,
    _task_signature,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RECORDS_PER_TABLE = int(os.getenv("MEMORY_MAX_RECORDS", "500"))
TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))
DEFAULT_DB_PATH = os.getenv(
    "MEMORY_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "agent_memory.db"),
)

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS router_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_sig    TEXT NOT NULL,
    task_summary TEXT NOT NULL,
    layers      TEXT NOT NULL,
    success     INTEGER NOT NULL,
    cost_usd    REAL DEFAULT 0.0,
    latency_ms  REAL DEFAULT 0.0,
    timestamp   REAL NOT NULL,
    tags        TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS executor_memory (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_sig                TEXT NOT NULL,
    partial_context_summary TEXT NOT NULL,
    tool_used               TEXT NOT NULL,
    arguments_pattern       TEXT NOT NULL,
    outcome_quality         TEXT NOT NULL,
    success                 INTEGER NOT NULL,
    timestamp               REAL NOT NULL,
    tags                    TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS verifier_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_sig        TEXT NOT NULL,
    failure_pattern TEXT NOT NULL,
    verdict         TEXT NOT NULL,
    repair_action   TEXT NOT NULL,
    repair_worked   INTEGER NOT NULL,
    timestamp       REAL NOT NULL,
    tags            TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_router_sig   ON router_memory(task_sig);
CREATE INDEX IF NOT EXISTS idx_executor_sig ON executor_memory(task_sig);
CREATE INDEX IF NOT EXISTS idx_verifier_sig ON verifier_memory(task_sig);
"""


class MemoryStore:
    """Thread-safe, bounded SQLite store for agent execution memory."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    # ---- connection management (per-thread) ----

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_DDL)
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ====================================================================
    # ADMISSION POLICY
    # ====================================================================

    def _admit_router(self, rec: RouterMemory) -> bool:
        """Only store successful routing decisions."""
        return rec.success

    def _admit_executor(self, rec: ExecutorMemory) -> bool:
        """Only store successful tool calls or acceptable quality ones."""
        return rec.success or rec.outcome_quality == "acceptable"

    def _admit_verifier(self, rec: VerifierMemory) -> bool:
        """Only store repair patterns that actually worked."""
        return rec.repair_worked

    # ====================================================================
    # STORAGE (with admission + eviction)
    # ====================================================================

    def _evict_oldest(self, table: str) -> None:
        conn = self._get_conn()
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count >= MAX_RECORDS_PER_TABLE:
            overflow = count - MAX_RECORDS_PER_TABLE + 1
            conn.execute(
                f"DELETE FROM {table} WHERE id IN "
                f"(SELECT id FROM {table} ORDER BY timestamp ASC LIMIT ?)",
                (overflow,),
            )

    def store_router(self, rec: RouterMemory) -> bool:
        """Store a router memory record if it passes admission. Returns True if stored."""
        if not self._admit_router(rec):
            logger.debug("RouterMemory rejected by admission policy.")
            return False
        conn = self._get_conn()
        self._evict_oldest("router_memory")
        conn.execute(
            "INSERT INTO router_memory (task_sig, task_summary, layers, success, cost_usd, latency_ms, timestamp, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.task_signature,
                rec.task_summary,
                json.dumps(rec.selected_layers),
                int(rec.success),
                rec.cost_usd,
                rec.latency_ms,
                rec.timestamp,
                json.dumps(rec.tags),
            ),
        )
        conn.commit()
        return True

    def store_executor(self, rec: ExecutorMemory) -> bool:
        """Store an executor memory record if it passes admission."""
        if not self._admit_executor(rec):
            logger.debug("ExecutorMemory rejected by admission policy.")
            return False
        conn = self._get_conn()
        self._evict_oldest("executor_memory")
        conn.execute(
            "INSERT INTO executor_memory (task_sig, partial_context_summary, tool_used, arguments_pattern, outcome_quality, success, timestamp, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.task_signature,
                rec.partial_context_summary,
                rec.tool_used,
                rec.arguments_pattern,
                rec.outcome_quality,
                int(rec.success),
                rec.timestamp,
                json.dumps(rec.tags),
            ),
        )
        conn.commit()
        return True

    def store_verifier(self, rec: VerifierMemory) -> bool:
        """Store a verifier memory record if it passes admission."""
        if not self._admit_verifier(rec):
            logger.debug("VerifierMemory rejected by admission policy.")
            return False
        conn = self._get_conn()
        self._evict_oldest("verifier_memory")
        conn.execute(
            "INSERT INTO verifier_memory (task_sig, failure_pattern, verdict, repair_action, repair_worked, timestamp, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                rec.task_signature,
                rec.failure_pattern,
                rec.verdict,
                rec.repair_action,
                int(rec.repair_worked),
                rec.timestamp,
                json.dumps(rec.tags),
            ),
        )
        conn.commit()
        return True

    # ====================================================================
    # RETRIEVAL (compact hints, never raw dumps)
    # ====================================================================

    def retrieve_router_hints(self, task_text: str, top_k: int = TOP_K) -> list[str]:
        """Return compact text hints for the Coordinator based on similar past routes.

        Returns lines like:
          'For similar tasks, layers ["react_reason","verifier_check"] succeeded (cost $0.002, 1200ms).'
        """
        sig = _task_signature(task_text)
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT task_summary, layers, cost_usd, latency_ms "
            "FROM router_memory WHERE task_sig = ? AND success = 1 "
            "ORDER BY timestamp DESC LIMIT ?",
            (sig, top_k),
        ).fetchall()

        hints = []
        for r in rows:
            layers = json.loads(r["layers"])
            hints.append(
                f"For similar task \"{r['task_summary'][:80]}\", "
                f"layers {layers} succeeded "
                f"(cost ${r['cost_usd']:.4f}, {r['latency_ms']:.0f}ms)."
            )
        return hints

    def retrieve_executor_hints(self, task_text: str, top_k: int = TOP_K) -> list[str]:
        """Return compact tool-selection hints for the Executor.

        Returns lines like:
          'For similar context, use black_scholes_price with spot=180, strike=175. Quality: good.'
        """
        sig = _task_signature(task_text)
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT partial_context_summary, tool_used, arguments_pattern, outcome_quality "
            "FROM executor_memory WHERE task_sig = ? "
            "AND (success = 1 OR outcome_quality = 'acceptable') "
            "ORDER BY timestamp DESC LIMIT ?",
            (sig, top_k),
        ).fetchall()

        hints = []
        for r in rows:
            hints.append(
                f"For \"{r['partial_context_summary'][:60]}\", "
                f"use {r['tool_used']} with {r['arguments_pattern'][:80]}. "
                f"Quality: {r['outcome_quality']}."
            )
        return hints

    def retrieve_verifier_hints(self, task_text: str, top_k: int = TOP_K) -> list[str]:
        """Return compact repair-strategy hints for the Verifier.

        Returns lines like:
          'Known failure: calculator rejects multiline. Repair: switch to finance tool. Worked: True.'
        """
        sig = _task_signature(task_text)
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT failure_pattern, verdict, repair_action, repair_worked "
            "FROM verifier_memory WHERE task_sig = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (sig, top_k),
        ).fetchall()

        hints = []
        for r in rows:
            hints.append(
                f"Known failure: {r['failure_pattern'][:80]}. "
                f"Repair ({r['verdict']}): {r['repair_action'][:80]}. "
                f"Worked: {bool(r['repair_worked'])}."
            )
        return hints

    # ---- Stats ----

    def stats(self) -> dict:
        """Return record counts per table."""
        conn = self._get_conn()
        return {
            "router_memory": conn.execute("SELECT COUNT(*) FROM router_memory").fetchone()[0],
            "executor_memory": conn.execute("SELECT COUNT(*) FROM executor_memory").fetchone()[0],
            "verifier_memory": conn.execute("SELECT COUNT(*) FROM verifier_memory").fetchone()[0],
        }
