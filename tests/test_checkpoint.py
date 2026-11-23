"""Checkpointer selection, and Postgres durability when a test database is set.

The Postgres tests need TEST_DATABASE_URL (CI starts a Postgres service), e.g.
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:17
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres uv run pytest
"""

import os
import threading

import psycopg
import pytest
from conftest import FakeModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_template.agents import Agent
from agent_template.checkpoint import default_checkpointer, postgres_checkpointer
from agent_template.orchestrator import FINISH, Orchestrator, Route

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
needs_postgres = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")


def test_in_memory_without_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "")
    assert isinstance(default_checkpointer(), InMemorySaver)


def server(reply: str, routes: list[Route]) -> Orchestrator:
    """One API replica: its own model and process memory, a shared database."""
    model = FakeModel(messages=iter([AIMessage(content=reply)]))
    model.routes = routes
    agent = Agent(name="WorkerAgent", description="does a thing", system_prompt="help")
    return Orchestrator([agent], model=model, checkpointer=postgres_checkpointer(TEST_DATABASE_URL))


@needs_postgres
def test_another_replica_continues_the_conversation() -> None:
    first_server = server(
        "first answer", [Route(next="WorkerAgent", reason="a"), Route(next=FINISH, reason="done")]
    )
    first = first_server.run("task one")

    second_server = server(
        "second answer", [Route(next="WorkerAgent", reason="b"), Route(next=FINISH, reason="done")]
    )
    second = second_server.run("follow up", thread_id=first.thread_id)

    texts = [t["text"] for t in second.transcript]
    assert texts == ["task one", "first answer", "follow up", "second answer"]


@needs_postgres
def test_threads_do_not_share_state() -> None:
    a = server("answer a", [Route(next="WorkerAgent", reason="a"), Route(next=FINISH, reason="x")])
    b = server("answer b", [Route(next="WorkerAgent", reason="b"), Route(next=FINISH, reason="x")])
    result_a = a.run("user a's task")
    result_b = b.run("user b's task")
    assert [t["text"] for t in result_b.transcript] == ["user b's task", "answer b"]
    assert result_a.thread_id != result_b.thread_id


@needs_postgres
def test_replicas_can_start_together_on_a_fresh_database() -> None:
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
        admin.execute("DROP DATABASE IF EXISTS startup_race WITH (FORCE)")
        admin.execute("CREATE DATABASE startup_race")
    fresh_url = TEST_DATABASE_URL.rsplit("/", 1)[0] + "/startup_race"
    errors: list[Exception] = []

    def start_replica() -> None:
        try:
            postgres_checkpointer(fresh_url)
        except Exception as error:
            errors.append(error)

    replicas = [threading.Thread(target=start_replica) for _ in range(4)]
    for replica in replicas:
        replica.start()
    for replica in replicas:
        replica.join()
    assert errors == []
