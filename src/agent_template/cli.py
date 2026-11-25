"""Command line: run an app, list past runs.

python -m agent_template run --app examples.job_finder.app "find me jobs"
python -m agent_template history
"""

import argparse
import importlib
import logging
import sys
from typing import Any

from agent_template.store import open_run_store
from agent_template.telemetry import setup_logging, setup_tracing

log = logging.getLogger(__name__)


def _load_app(app_module: str) -> Any:
    module = importlib.import_module(app_module)
    return module.build_orchestrator()


def cmd_run(args: argparse.Namespace) -> int:
    setup_logging()
    setup_tracing()
    orchestrator = _load_app(args.app)
    result = orchestrator.run(args.task, thread_id=args.thread_id)
    run_id = open_run_store().record(
        args.app, result, user_id=args.user_id, tenant_id=args.tenant_id
    )
    print(result.output)
    # "tokens" here are LLM token counts, not credentials.
    log.info(  # nosemgrep
        "run %s %s (%d steps, %d ms, %d+%d tokens, thread %s)",
        run_id,
        result.status,
        result.steps,
        result.latency_ms,
        result.input_tokens,
        result.output_tokens,
        result.thread_id,
    )
    return 1 if result.status == "failed" else 0


def cmd_history(args: argparse.Namespace) -> int:
    for row in open_run_store().recent(args.limit):
        status = row["status"] or "-"
        print(f"[{row['id']}] {row['created_at'][:19]} {status} {row['app']}: {row['task'][:60]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-template", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an app on a task")
    run.add_argument("--app", required=True, help="app module, e.g. examples.job_finder.app")
    run.add_argument("task")
    run.add_argument(
        "--thread-id", help="continue this conversation (needs DATABASE_URL to outlive a process)"
    )
    run.add_argument("--user-id", help="who ran it, recorded with the run")
    run.add_argument("--tenant-id", help="their organisation, recorded with the run")
    run.set_defaults(func=cmd_run)

    hist = sub.add_parser("history", help="list recent runs")
    hist.add_argument("--limit", type=int, default=20)
    hist.set_defaults(func=cmd_history)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
