import sqlite3
import uuid
from pathlib import Path

from agent.memory.schema import CurationSignal, ReviewMemory, RunMemory, ToolMemory, task_signature
from agent.memory.store import MemoryStore


def _workspace_temp_db_path() -> Path:
    return Path.cwd() / f".tmp_memory_store_{uuid.uuid4().hex}.db"


def test_memory_store_resets_legacy_schema():
    db_path = _workspace_temp_db_path()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE router_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_sig TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    store = MemoryStore(str(db_path))
    stats = store.stats()

    assert stats["schema_version"] == 3
    assert stats["run_memory"] == 0
    assert stats["tool_memory"] == 0
    assert stats["review_memory"] == 0
    assert stats["curation_memory"] == 0


def test_memory_store_persists_staged_records():
    store = MemoryStore(str(_workspace_temp_db_path()))
    sig = task_signature("test prompt")

    assert store.store_run(
        RunMemory(
            task_signature=sig,
            task_summary="test prompt",
            task_profile="finance_quant",
            task_family="finance",
            capability_flags=["needs_math"],
            route_path=["intake", "task_profiler", "context_builder", "solver", "reviewer", "reflect"],
            stage_history=["PLAN", "COMPUTE", "SYNTHESIZE"],
            answer_format="json",
            success=True,
        )
    )
    assert store.store_tool(
        ToolMemory(
            task_signature=sig,
            task_profile="finance_quant",
            task_family="finance",
            solver_stage="COMPUTE",
            tool_name="calculator",
            result_type="calculator",
            arguments_json={"expression": "1+1"},
            fact_keys=["result"],
            success=True,
        )
    )
    assert store.store_review(
        ReviewMemory(
            task_signature=sig,
            task_profile="finance_quant",
            task_family="finance",
            review_stage="SYNTHESIZE",
            verdict="pass",
            reasoning="Complete answer.",
            success=True,
        )
    )
    assert store.store_curation(
        CurationSignal(
            task_signature=sig,
            task_profile="finance_quant",
            task_family="finance",
            template_id="quant_inline_exact",
            signal_type="missing_dimension",
            signal_key="scalar_answer",
            summary="Final answer missed the scalar answer shape.",
            stage="SYNTHESIZE",
        )
    )

    stats = store.stats()
    assert stats["run_memory"] == 1
    assert stats["tool_memory"] == 1
    assert stats["review_memory"] == 1
    assert stats["curation_memory"] == 1

    fetched = store.fetch_curation_signals(limit=10)
    assert fetched[0]["signal_type"] == "missing_dimension"
