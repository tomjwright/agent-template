"""Evals: run the app on fixed cases and fail when a score drops.

Cases are YAML:

```yaml
cases:
  - input: "find me python jobs in Berlin"
    expect_contains: ["job"]          # all must appear in the output
    expect_not_contains: ["sorry"]    # none may appear
```

Run: `python -m agent_template.evals --app examples.job_finder.app [--cases path]`
Every case runs through the real app with the real model, so evals need an
API key in .env. CI uses this as a release gate.
"""

import argparse
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_template.orchestrator import RunResult


class Case(BaseModel):
    # A typo in cases.yaml (e.g. `expect_contain`) must fail, not be ignored.
    model_config = ConfigDict(extra="forbid")

    input: str
    expect_contains: list[str] = Field(default_factory=list)
    expect_not_contains: list[str] = Field(default_factory=list)

    def check(self, output: str) -> list[str]:
        """The reasons this output fails, or an empty list when it passes."""
        lower = output.lower()
        failures = [f"missing {s!r}" for s in self.expect_contains if s.lower() not in lower]
        failures += [f"contains {s!r}" for s in self.expect_not_contains if s.lower() in lower]
        return failures


class EvalReport(BaseModel):
    passed: int
    total: int
    failures: list[tuple[Case, list[str]]] = Field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


def load_cases(path: Path | str) -> list[Case]:
    raw = yaml.safe_load(Path(path).read_text())
    return [Case(**c) for c in raw["cases"]]


def run_cases(run: Callable[[str], RunResult], cases: list[Case]) -> EvalReport:
    """A case fails if its run did not complete, even when the output passes the checks."""
    failures: list[tuple[Case, list[str]]] = []
    for case in cases:
        result = run(case.input)
        reasons = case.check(result.output)
        if result.status != "completed":
            reasons.insert(0, f"run {result.status}")
        if reasons:
            failures.append((case, reasons))
    return EvalReport(passed=len(cases) - len(failures), total=len(cases), failures=failures)


def _load_runner(app_module: str) -> Callable[[str], RunResult]:
    """`run(task)` for an app module exposing `build_orchestrator`."""
    module = importlib.import_module(app_module)
    orchestrator = module.build_orchestrator()
    return cast(Callable[[str], RunResult], orchestrator.run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, help="app module, e.g. examples.job_finder.app")
    parser.add_argument(
        "--cases", default=None, help="cases.yaml (default: <app dir>/evals/cases.yaml)"
    )
    parser.add_argument("--min-score", type=float, default=1.0)
    args = parser.parse_args(argv)

    app_file = importlib.import_module(args.app).__file__
    if app_file is None:
        raise SystemExit(f"cannot locate module {args.app!r}")
    cases_path = Path(args.cases) if args.cases else Path(app_file).parent / "evals" / "cases.yaml"
    report = run_cases(_load_runner(args.app), load_cases(cases_path))
    print(f"evals: {report.passed}/{report.total} passed (score {report.score:.2f})")
    for case, reasons in report.failures:
        print(f"  FAIL {case.input!r}: {', '.join(reasons)}")
    return 0 if report.score >= args.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
