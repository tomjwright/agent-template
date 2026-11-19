from pathlib import Path

from agent_template.evals import Case, load_cases, run_cases


def test_case_check_passes() -> None:
    case = Case(input="q", expect_contains=["ok"], expect_not_contains=["bad"])
    assert case.check("all is ok") == []


def test_case_check_reports_missing_and_forbidden() -> None:
    case = Case(input="q", expect_contains=["yes"], expect_not_contains=["no"])
    failures = case.check("no way")
    assert failures == ["missing 'yes'", "contains 'no'"]


def test_check_is_case_insensitive() -> None:
    assert Case(input="q", expect_contains=["Berlin"]).check("berlin") == []


def test_run_cases_score() -> None:
    cases = [Case(input="a", expect_contains=["a"]), Case(input="b", expect_contains=["b"])]
    report = run_cases(lambda task: "a", cases)
    assert report.passed == 1 and report.total == 2
    assert report.score == 0.5


def test_load_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text(
        "cases:\n  - input: hi\n    expect_contains: [hello]\n"
        "  - input: bye\n    expect_not_contains: [hi]\n"
    )
    cases = load_cases(path)
    assert len(cases) == 2
    assert cases[0].expect_contains == ["hello"]
    assert cases[1].expect_not_contains == ["hi"]
