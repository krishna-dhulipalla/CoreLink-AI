"""
Memory Store
============
Versioned SQLite-backed store for staged-runtime execution memory.

Design goals:
- Persist compact staged-runtime artifacts, not legacy coordinator telemetry.
- Reset incompatible schemas automatically so stale DB files do not poison the
  active runtime after architecture changes.
- Keep the on-disk format simple enough to evolve again later.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

from agent.memory.schema import MEMORY_SCHEMA_VERSION, ReviewMemory, RunMemory, ToolMemory

logger = logging.getLogger(__name__)

MAX_RECORDS_PER_TABLE = int(os.getenv("MEMORY_MAX_RECORDS", "1000"))
DEFAULT_DB_PATH = os.getenv(
    "MEMORY_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "agent_memory.db"),
)

_DDL = """
CREATE TABLE IF NOT EXISTS memory_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_memory (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    task_sig         TEXT NOT NULL,
    task_summary     TEXT NOT NULL,
    semantic_text    TEXT DEFAULT '',
    task_profile     TEXT NOT NULL,
    task_family      TEXT DEFAULT 'general',
    capability_flags TEXT DEFAULT '[]',
    route_path       TEXT DEFAULT '[]',
    stage_history    TEXT DEFAULT '[]',
    answer_format    TEXT DEFAULT 'text',
    success          INTEGER NOT NULL,
    tool_call_count  INTEGER DEFAULT 0,
    review_cycles    INTEGER DEFAULT 0,
    cost_usd         REAL DEFAULT 0.0,
    latency_ms       REAL DEFAULT 0.0,
    timestamp        REAL NOT NULL,
    tags             TEXT DEFAULT '[]',
    metadata_json    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tool_memory (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_sig       TEXT NOT NULL,
    task_profile   TEXT NOT NULL,
    task_family    TEXT DEFAULT 'general',
    solver_stage   TEXT DEFAULT 'COMPUTE',
    tool_name      TEXT NOT NULL,
    result_type    TEXT NOT NULL,
    semantic_text  TEXT DEFAULT '',
    arguments_json TEXT DEFAULT '{}',
    fact_keys      TEXT DEFAULT '[]',
    error_count    INTEGER DEFAULT 0,
    success        INTEGER NOT NULL,
    timestamp      REAL NOT NULL,
    metadata_json  TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS review_memory (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    task_sig           TEXT NOT NULL,
    task_profile       TEXT NOT NULL,
    task_family        TEXT DEFAULT 'general',
    review_stage       TEXT DEFAULT 'SYNTHESIZE',
    verdict            TEXT NOT NULL,
    repair_target      TEXT DEFAULT 'final',
    missing_dimensions TEXT DEFAULT '[]',
    reasoning          TEXT DEFAULT '',
    success            INTEGER NOT NULL,
    timestamp          REAL NOT NULL,
    metadata_json      TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_run_sig    ON run_memory(task_sig);
CREATE INDEX IF NOT EXISTS idx_tool_sig   ON tool_memory(task_sig);
CREATE INDEX IF NOT EXISTS idx_review_sig ON review_memory(task_sig);
"""

_EXPECTED_COLUMNS = {
    "memory_meta": {"key", "value"},
    "run_memory": {
        "id",
        "task_sig",
        "task_summary",
        "semantic_text",
        "task_profile",
        "task_family",
        "capability_flags",
        "route_path",
        "stage_history",
        "answer_format",
        "success",
        "tool_call_count",
        "review_cycles",
        "cost_usd",
        "latency_ms",
        "timestamp",
        "tags",
        "metadata_json",
    },
    "tool_memory": {
        "id",
        "task_sig",
        "task_profile",
        "task_family",
        "solver_stage",
        "tool_name",
        "result_type",
        "semantic_text",
        "arguments_json",
        "fact_keys",
        "error_count",
        "success",
        "timestamp",
        "metadata_json",
    },
    "review_memory": {
        "id",
        "task_sig",
        "task_profile",
        "task_family",
        "review_stage",
        "verdict",
        "repair_target",
        "missing_dimensions",
        "reasoning",
        "success",
        "timestamp",
        "metadata_json",
    },
}


class MemoryStore:
    """Thread-safe, versioned SQLite store for staged-runtime memory."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        self._local = threading.local()
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_DDL)
        conn.commit()
        if not self._schema_matches():
            logger.warning("[Memory] Resetting staged memory DB because the stored schema is incompatible.")
            self.reset()
            conn = self._get_conn()
        self._write_schema_version(conn)

    def _schema_matches(self) -> bool:
        conn = self._get_conn()
        for table, expected in _EXPECTED_COLUMNS.items():
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            columns = {row["name"] for row in rows}
            if columns != expected:
                return False
        row = conn.execute("SELECT value FROM memory_meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            # Existing DB with staged tables but no meta version is still treated as stale.
            return False
        return str(row["value"]) == str(MEMORY_SCHEMA_VERSION)

    def _write_schema_version(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO memory_meta(key, value) VALUES('schema_version', ?)",
            (str(MEMORY_SCHEMA_VERSION),),
        )
        conn.commit()

    def reset(self) -> None:
        """Reset the DB to the active staged-runtime schema."""
        if self._db_path == ":memory:":
            conn = self._get_conn()
            conn.executescript(
                """
                DROP TABLE IF EXISTS memory_meta;
                DROP TABLE IF EXISTS run_memory;
                DROP TABLE IF EXISTS tool_memory;
                DROP TABLE IF EXISTS review_memory;
                """
            )
            conn.executescript(_DDL)
            self._write_schema_version(conn)
            return

        db_file = Path(self._db_path)
        self.close()
        if db_file.exists():
            db_file.unlink()
        self._local = threading.local()
        conn = self._get_conn()
        conn.executescript(_DDL)
        self._write_schema_version(conn)

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def _evict_oldest(self, table: str) -> None:
        conn = self._get_conn()
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count >= MAX_RECORDS_PER_TABLE:
            overflow = count - MAX_RECORDS_PER_TABLE + 1
            conn.execute(
                f"DELETE FROM {table} WHERE id IN (SELECT id FROM {table} ORDER BY timestamp ASC LIMIT ?)",
                (overflow,),
            )

    def store_run(self, rec: RunMemory) -> bool:
        conn = self._get_conn()
        self._evict_oldest("run_memory")
        conn.execute(
            """
            INSERT INTO run_memory (
                task_sig, task_summary, semantic_text, task_profile, task_family,
                capability_flags, route_path, stage_history, answer_format, success,
                tool_call_count, review_cycles, cost_usd, latency_ms, timestamp, tags, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.task_signature,
                rec.task_summary,
                rec.semantic_text,
                str(rec.task_profile),
                rec.task_family,
                json.dumps(rec.capability_flags),
                json.dumps(rec.route_path),
                json.dumps(rec.stage_history),
                rec.answer_format,
                int(rec.success),
                rec.tool_call_count,
                rec.review_cycle_count,
                rec.cost_usd,
                rec.latency_ms,
                rec.timestamp,
                json.dumps(rec.tags),
                json.dumps(rec.metadata),
            ),
        )
        conn.commit()
        return True

    def store_tool(self, rec: ToolMemory) -> bool:
        conn = self._get_conn()
        self._evict_oldest("tool_memory")
        conn.execute(
            """
            INSERT INTO tool_memory (
                task_sig, task_profile, task_family, solver_stage, tool_name, result_type,
                semantic_text, arguments_json, fact_keys, error_count, success, timestamp, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.task_signature,
                str(rec.task_profile),
                rec.task_family,
                rec.solver_stage,
                rec.tool_name,
                rec.result_type,
                rec.semantic_text,
                json.dumps(rec.arguments_json),
                json.dumps(rec.fact_keys),
                rec.error_count,
                int(rec.success),
                rec.timestamp,
                json.dumps(rec.metadata),
            ),
        )
        conn.commit()
        return True

    def store_review(self, rec: ReviewMemory) -> bool:
        conn = self._get_conn()
        self._evict_oldest("review_memory")
        conn.execute(
            """
            INSERT INTO review_memory (
                task_sig, task_profile, task_family, review_stage, verdict, repair_target,
                missing_dimensions, reasoning, success, timestamp, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.task_signature,
                str(rec.task_profile),
                rec.task_family,
                rec.review_stage,
                rec.verdict,
                rec.repair_target,
                json.dumps(rec.missing_dimensions),
                rec.reasoning,
                int(rec.success),
                rec.timestamp,
                json.dumps(rec.metadata),
            ),
        )
        conn.commit()
        return True

    def stats(self) -> dict[str, int]:
        conn = self._get_conn()
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "run_memory": conn.execute("SELECT COUNT(*) FROM run_memory").fetchone()[0],
            "tool_memory": conn.execute("SELECT COUNT(*) FROM tool_memory").fetchone()[0],
            "review_memory": conn.execute("SELECT COUNT(*) FROM review_memory").fetchone()[0],
        }
