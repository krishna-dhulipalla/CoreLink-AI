"""
Conversation Store: In-Memory Session Persistence
===================================================
Persists LangGraph message history across A2A requests sharing the
same ``context_id``, enabling multi-turn conversations.

Messages are stored in a plain dict with TTL-based auto-cleanup so
completed conversations don't leak memory.

Configuration via .env:
    CONVERSATION_TTL_SECONDS=3600   (default 1 hour)
"""

import logging
import os
import time
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

load_dotenv()

logger = logging.getLogger(__name__)

CONVERSATION_TTL_SECONDS = int(os.getenv("CONVERSATION_TTL_SECONDS", "3600"))


class ConversationStore:
    """In-memory store keyed by ``context_id``.

    Each entry holds ``(messages, last_access_timestamp)``.
    Stale entries are lazily purged on every ``get`` / ``save`` call.
    """

    def __init__(self, ttl: int | None = None):
        self._ttl = ttl or CONVERSATION_TTL_SECONDS
        self._store: dict[str, tuple[list[BaseMessage], float]] = {}

    # ── Public API ────────────────────────────────────────────────────

    def get(self, context_id: str) -> list[BaseMessage]:
        """Return stored messages for *context_id*, or an empty list."""
        self._purge_stale()
        entry = self._store.get(context_id)
        if entry is None:
            return []
        messages, _ = entry
        # Touch timestamp on read
        self._store[context_id] = (messages, time.time())
        logger.debug(
            "Loaded %d messages for context_id=%s", len(messages), context_id
        )
        return list(messages)  # shallow copy

    def save(self, context_id: str, messages: Sequence[BaseMessage]) -> None:
        """Persist *messages* for *context_id*, overwriting any prior entry."""
        self._purge_stale()
        self._store[context_id] = (list(messages), time.time())
        logger.debug(
            "Saved %d messages for context_id=%s", len(messages), context_id
        )

    def clear(self, context_id: str) -> None:
        """Remove the conversation for *context_id*."""
        self._store.pop(context_id, None)

    # ── Internal ──────────────────────────────────────────────────────

    def _purge_stale(self) -> None:
        """Remove entries older than TTL."""
        now = time.time()
        stale = [
            cid
            for cid, (_, ts) in self._store.items()
            if now - ts > self._ttl
        ]
        for cid in stale:
            del self._store[cid]
            logger.info("Purged stale conversation context_id=%s", cid)
