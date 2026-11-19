from pathlib import Path

from agent_template.store import RunStore


def test_record_and_recent(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = store.record("app", "task", "out", 3, [{"agent": "A", "text": "hi"}])
    assert run_id == 1
    rows = store.recent()
    assert rows[0]["app"] == "app"
    assert rows[0]["output"] == "out"
    assert rows[0]["steps"] == 3


def test_recent_orders_newest_first(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.record("app", "first", "1", 1, [])
    store.record("app", "second", "2", 1, [])
    assert [r["task"] for r in store.recent()] == ["second", "first"]
