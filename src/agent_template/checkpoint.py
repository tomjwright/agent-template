"""Where LangGraph saves each run's state after every step (checkpoints).

With DATABASE_URL set, checkpoints go to Postgres: a run survives a crash or a
restart, and any API replica can continue any conversation. Without it they
stay in memory, which is fine for the CLI and tests but lost when the process
exits.
"""

import atexit
import os
import time

from dotenv import load_dotenv
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

SETUP_LOCK_ID = 7_413_201  # any constant; shared by every replica


def default_checkpointer() -> BaseCheckpointSaver[str]:
    """Postgres when DATABASE_URL is set, else in memory."""
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        return InMemorySaver()
    return postgres_checkpointer(url)


def postgres_checkpointer(url: str) -> PostgresSaver:
    """A Postgres checkpointer on a connection pool, with its tables created."""
    pool = ConnectionPool(
        url,
        connection_class=Connection[DictRow],
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    atexit.register(pool.close)
    # Replicas starting together would race to create the tables; the lock
    # makes them take turns, and setup() skips work that is already done.
    with pool.connection() as conn:
        _wait_for_setup_lock(conn)
        try:
            PostgresSaver(conn).setup()
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (SETUP_LOCK_ID,))
    return PostgresSaver(pool)


def _wait_for_setup_lock(conn: Connection[DictRow]) -> None:
    # Poll rather than block in pg_advisory_lock: a blocked statement is an
    # open transaction, and setup()'s CREATE INDEX CONCURRENTLY would wait on it.
    while True:
        row = conn.execute("SELECT pg_try_advisory_lock(%s) AS locked", (SETUP_LOCK_ID,)).fetchone()
        if row and row["locked"]:
            return
        time.sleep(0.1)
