"""Where LangGraph saves each run's state after every step (checkpoints).

With DATABASE_URL set, checkpoints go to Postgres: a run survives a crash or a
restart, and any API replica can continue any conversation. Without it they
stay in memory, which is fine for the CLI and tests but lost when the process
exits.
"""

import atexit
import os

from dotenv import load_dotenv
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool


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
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer
