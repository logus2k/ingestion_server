"""Run persistence.

A run must survive a restart: it can suspend on a judge flag and wait for a human
decision that arrives minutes or hours later, from a frontend, another agent, or
the bus. In-memory state would lose it.

SQLite: one file, no extra service, and runs are low-volume (one per document
batch). The whole Run is stored as JSON — it is a document, not a relational
model, and nothing queries inside it except by id and state.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from .config import settings
from .models import Run, RunState

_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or settings.state_db
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with _LOCK, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id      TEXT PRIMARY KEY,
                    state       TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    doc         TEXT NOT NULL
                )""")
            c.execute("CREATE INDEX IF NOT EXISTS runs_state ON runs (state)")

    def save(self, run: Run) -> Run:
        run.updated_at = now_iso()
        if not run.created_at:
            run.created_at = run.updated_at
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO runs (run_id, state, created_at, updated_at, doc) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET state=excluded.state, "
                "updated_at=excluded.updated_at, doc=excluded.doc",
                (run.run_id, run.state.value, run.created_at, run.updated_at,
                 run.model_dump_json()))
        return run

    def get(self, run_id: str) -> Optional[Run]:
        with _LOCK, self._conn() as c:
            row = c.execute("SELECT doc FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return Run.model_validate_json(row["doc"]) if row else None

    def list(self, state: RunState | None = None, limit: int = 50) -> list[Run]:
        q = "SELECT doc FROM runs"
        args: tuple = ()
        if state:
            q += " WHERE state = ?"
            args = (state.value,)
        q += " ORDER BY created_at DESC LIMIT ?"
        args = args + (limit,)
        with _LOCK, self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [Run.model_validate_json(r["doc"]) for r in rows]

    def reap_orphans(self) -> int:
        """A run left `running` by a crash can never resume — nothing is driving
        it. Mark those failed at startup so they aren't mistaken for live work."""
        with _LOCK, self._conn() as c:
            rows = c.execute("SELECT doc FROM runs WHERE state = ?",
                             (RunState.running.value,)).fetchall()
        n = 0
        for r in rows:
            run = Run.model_validate_json(r["doc"])
            run.state = RunState.failed
            run.error = "abandoned: the service restarted while this run was in flight"
            self.save(run)
            n += 1
        return n
