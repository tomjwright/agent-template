"""Run history in SQLite.

Every CLI run is recorded so a project can audit what the system did and evals
can diff outputs between versions. Each row also records the run's status,
model, tokens, cost, latency, trace id and who ran it, so the table can answer
"why did the bill spike?", "who ran this?" and "did it hit the step cap?".
"""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_template.orchestrator import RunResult

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

# Added after the first release; added to older tables on open.
_TELEMETRY_COLUMNS = {
    "thread_id": "TEXT",
    "status": "TEXT",
    "model_name": "TEXT",
    "input_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "cost_usd": "REAL",
    "latency_ms": "INTEGER",
    "trace_id": "TEXT",
    "user_id": "TEXT",
    "tenant_id": "TEXT",
}


_INSERT = """
INSERT INTO runs (
    app, task, output, steps, transcript, created_at, thread_id, status, model_name,
    input_tokens, output_tokens, cost_usd, latency_ms, trace_id, user_id, tenant_id
) VALUES (
    :app, :task, :output, :steps, :transcript, :created_at, :thread_id, :status, :model_name,
    :input_tokens, :output_tokens, :cost_usd, :latency_ms, :trace_id, :user_id, :tenant_id
)
"""


class RunStore:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # Hold the write lock while checking columns, so two processes
            # opening an older table don't both try to add the same column.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(_SCHEMA)
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
            for name, column_type in _TELEMETRY_COLUMNS.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {column_type}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(
        self,
        app: str,
        result: RunResult,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> int:
        row = {
            "app": app,
            "task": result.task,
            "output": result.output,
            "steps": result.steps,
            "transcript": json.dumps(result.transcript),
            "created_at": datetime.now(UTC).isoformat(),
            "thread_id": result.thread_id,
            "status": result.status,
            "model_name": result.model_name,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "trace_id": result.trace_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
        }
        with self._connect() as conn:
            cur = conn.execute(_INSERT, row)
            return int(cur.lastrowid or 0)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """The newest runs first, every column except the transcript."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
            runs = [dict(row) for row in rows]
        for run in runs:
            del run["transcript"]
        return runs
