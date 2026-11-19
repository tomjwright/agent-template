"""Command line: run an app, list past runs.

python -m agent_template run --app examples.job_finder.app "find me jobs"
python -m agent_template history
"""

import argparse
import importlib
import logging
import sys
from typing import Any

from agent_template.store import RunStore
from agent_template.telemetry import setup_logging, setup_tracing

log = logging.getLogger(__name__)


def _load_app(app_module: str) -> Any:
    module = importlib.import_module(app_module)
    return module.build_orchestrator()


def cmd_run(args: argparse.Namespace) -> int:
    setup_logging()
    setup_tracing()
    orchestrator = _load_app(args.app)
    result = orchestrator.run(args.task)
    run_id = RunStore().record(
        app=args.app,
        task=args.task,
        output=result.output,
        steps=result.steps,
        transcript=result.transcript,
    )
    print(result.output)
    log.info("run %s recorded (%d steps)", run_id, result.steps)
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    for row in RunStore().recent(args.limit):
        print(f"[{row['id']}] {row['created_at'][:19]} {row['app']}: {row['task'][:60]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-template", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an app on a task")
    run.add_argument("--app", required=True, help="app module, e.g. examples.job_finder.app")
    run.add_argument("task")
    run.set_defaults(func=cmd_run)

    hist = sub.add_parser("history", help="list recent runs")
    hist.add_argument("--limit", type=int, default=20)
    hist.set_defaults(func=cmd_history)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
