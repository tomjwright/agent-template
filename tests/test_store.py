import sqlite3
import threading
from pathlib import Path

from agent_template.orchestrator import RunResult
from agent_template.store import RunStore


def run_result(task: str = "task") -> RunResult:
    return RunResult(
        task=task,
        thread_id="thread-1",
        status="step_cap_hit",
        output="out",
        steps=3,
        transcript=[{"agent": "A", "text": "hi"}],
        model_name="claude-haiku-4-5",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=0.002,
        latency_ms=850,
        trace_id="ab" * 16,
    )


def test_record_and_recent(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = store.record("app", run_result(), user_id="u1", tenant_id="t1")
    assert run_id == 1
    [row] = store.recent()
    assert row == {
        "id": 1,
        "app": "app",
        "task": "task",
        "output": "out",
        "steps": 3,
        "created_at": row["created_at"],
        "thread_id": "thread-1",
        "status": "step_cap_hit",
        "model_name": "claude-haiku-4-5",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cost_usd": 0.002,
        "latency_ms": 850,
        "trace_id": "ab" * 16,
        "user_id": "u1",
        "tenant_id": "t1",
    }


def test_recent_orders_newest_first(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.record("app", run_result("first"))
    store.record("app", run_result("second"))
    assert [r["task"] for r in store.recent()] == ["second", "first"]


def test_adds_telemetry_columns_to_an_older_table(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, app TEXT NOT NULL,"
            " task TEXT NOT NULL, output TEXT NOT NULL, steps INTEGER NOT NULL,"
            " transcript TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO runs (app, task, output, steps, transcript, created_at)"
            " VALUES ('app', 'old', 'out', 1, '[]', '2026-01-01')"
        )
    store = RunStore(path)
    store.record("app", run_result("new"))
    old, new = reversed(store.recent())
    assert old["status"] is None
    assert new["status"] == "step_cap_hit"


def test_processes_can_open_an_older_table_together(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, app TEXT NOT NULL,"
            " task TEXT NOT NULL, output TEXT NOT NULL, steps INTEGER NOT NULL,"
            " transcript TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
    errors: list[Exception] = []

    def open_store() -> None:
        try:
            RunStore(path)
        except Exception as error:
            errors.append(error)

    openers = [threading.Thread(target=open_store) for _ in range(8)]
    for opener in openers:
        opener.start()
    for opener in openers:
        opener.join()
    assert errors == []
