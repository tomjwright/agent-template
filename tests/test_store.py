"""Run history in SQLite, and in Postgres when a test database is set.

The Postgres tests need TEST_DATABASE_URL; see tests/test_checkpoint.py.
"""

import os
import sqlite3
import threading
from pathlib import Path

import psycopg
import pytest

from agent_template.orchestrator import RunResult
from agent_template.store import PostgresRunStore, RunStore, open_run_store

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
needs_postgres = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")


def empty_postgres_store() -> PostgresRunStore:
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS runs")
    return PostgresRunStore(TEST_DATABASE_URL)


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=needs_postgres)])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> RunStore | PostgresRunStore:
    if request.param == "postgres":
        return empty_postgres_store()
    return RunStore(tmp_path / "runs.db")


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


def test_record_and_recent(store: RunStore | PostgresRunStore) -> None:
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


def test_recent_orders_newest_first(store: RunStore | PostgresRunStore) -> None:
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


def test_sqlite_without_database_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("RUN_DB", str(tmp_path / "runs.db"))
    store = open_run_store()
    assert isinstance(store, RunStore)
    assert store.path == tmp_path / "runs.db"


@needs_postgres
def test_postgres_with_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    assert isinstance(open_run_store(), PostgresRunStore)


@needs_postgres
def test_postgres_stores_the_transcript_as_json() -> None:
    empty_postgres_store().record("app", run_result())
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        row = conn.execute("SELECT transcript->0->>'text' FROM runs").fetchone()
    assert row == ("hi",)


@needs_postgres
def test_replicas_can_create_the_table_together() -> None:
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS runs")
    errors: list[Exception] = []

    def open_store() -> None:
        try:
            PostgresRunStore(TEST_DATABASE_URL).pool.close()
        except Exception as error:
            errors.append(error)

    replicas = [threading.Thread(target=open_store) for _ in range(8)]
    for replica in replicas:
        replica.start()
    for replica in replicas:
        replica.join()
    assert errors == []
