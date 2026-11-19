"""The example app's tools, exercised offline."""

from examples.job_finder.tools import improve_resume_section, search_jobs, submit_application


def test_search_jobs_matches_query_and_location() -> None:
    jobs = search_jobs.invoke({"query": "python", "location": "Berlin"})
    assert jobs
    assert all("Berlin" in j["location"] for j in jobs)


def test_search_jobs_falls_back_when_nothing_matches() -> None:
    jobs = search_jobs.invoke({"query": "underwater-basket-weaving", "location": ""})
    assert len(jobs) == 2


def test_improve_resume_section_appends_tailored_line() -> None:
    out = improve_resume_section.invoke(
        {"resume_text": "Built APIs.", "job_title": "Backend Engineer", "job_skills": "python, go"}
    )
    assert "Built APIs." in out
    assert "Tailored for Backend Engineer" in out
    assert "python" in out


def test_submit_application_is_dry_run() -> None:
    result = submit_application.invoke(
        {"job_url": "https://jobs.example/j1", "resume": "r", "cover_letter": "letter"}
    )
    assert result["status"] == "dry-run-saved"
