"""Run history in SQLite.

Every CLI run is recorded so a project can audit what the system did and evals
can diff outputs between versions.
"""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(os.environ.get("RUN_DB", "data/runs.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app TEXT NOT NULL,
    task TEXT NOT NULL,
    output TEXT NOT NULL,
    steps INTEGER NOT NULL,
    transcript TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class RunStore:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(
        self, app: str, task: str, output: str, steps: int, transcript: list[dict[str, str]]
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (app, task, output, steps, transcript, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    app,
                    task,
                    output,
                    steps,
                    json.dumps(transcript),
                    datetime.now(UTC).isoformat(),
                ),
            )
            return int(cur.lastrowid or 0)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, app, task, output, steps, created_at FROM runs"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "app": r[1],
                "task": r[2],
                "output": r[3],
                "steps": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]
