"""
Generate Offline Context-Pack Curation Summary
=============================================
Reads stored curation signals from the staged runtime SQLite DB and emits a
deterministic JSON summary for offline profile-pack/template review.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.memory.curation import summarize_curation_signals
from agent.memory.store import MemoryStore


def main() -> None:
    store = MemoryStore(os.getenv("MEMORY_DB_PATH"))
    signals = store.fetch_curation_signals(limit=5000)
    summary = summarize_curation_signals(signals, min_count=2)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
