"""Local journal — SQLite, no migrations, no ORM.

A reflection is a small JSON-y record. We persist enough to render the
history list and reconstruct any past reflection in full: the question, the
chart hash, the system, the moment's transits, the headline + body, and the
citations. We do *not* persist the underlying chart or the corpus snippets;
both are deterministic given the user's birth data and chart_hash, and the
corpus chunks are immutable references the retriever can re-fetch.

Storage path is configurable via SUNDIAL_JOURNAL_PATH. Tests use ":memory:".
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any

DEFAULT_PATH = os.environ.get("SUNDIAL_JOURNAL_PATH", "sundial.db")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reflections (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    system        TEXT NOT NULL,
    chart_hash    TEXT NOT NULL,
    question      TEXT NOT NULL,
    headline      TEXT,
    body          TEXT,
    citations     TEXT NOT NULL DEFAULT '[]',
    transits      TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS reflections_chart_created
    ON reflections (chart_hash, created_at DESC);
"""


class JournalStore:
    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self._path = path
        # One connection guarded by a lock. Reflections write at human pace —
        # contention is theoretical, but the lock keeps `:memory:` databases
        # usable across threads (FastAPI TestClient runs handlers in a worker).
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save(
        self,
        *,
        system: str,
        chart_hash: str,
        question: str,
        headline: str | None,
        body: str | None,
        citations: list[dict[str, Any]],
        transits: list[dict[str, Any]],
    ) -> str:
        rid = f"refl_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO reflections "
                "(id, created_at, system, chart_hash, question, headline, body, citations, transits) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rid, time.time(), system, chart_hash, question,
                    headline, body,
                    json.dumps(citations), json.dumps(transits),
                ),
            )
            self._conn.commit()
        return rid

    def recent(self, *, chart_hash: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            if chart_hash:
                rows = self._conn.execute(
                    "SELECT * FROM reflections WHERE chart_hash = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (chart_hash, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "system": row["system"],
        "chart_hash": row["chart_hash"],
        "question": row["question"],
        "headline": row["headline"],
        "body": row["body"],
        "citations": json.loads(row["citations"]),
        "transits": json.loads(row["transits"]),
    }
