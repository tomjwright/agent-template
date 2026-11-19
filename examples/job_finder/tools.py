"""Tools for the job finder agents.

These are plain functions wrapped as LangChain tools. The search tool returns
canned listings so the example runs offline; replace it with real job-board
APIs (Adzuna, Reed, LinkedIn) when the project is real.
"""

from typing import Any

from langchain_core.tools import tool

_LISTINGS: list[dict[str, Any]] = [
    {
        "id": "J1",
        "title": "Senior Python Engineer",
        "company": "Nordlicht GmbH",
        "location": "Berlin",
        "skills": ["python", "sql", "docker"],
        "url": "https://jobs.example/j1",
    },
    {
        "id": "J2",
        "title": "Backend Engineer (Go/Python)",
        "company": "Klarheit AG",
        "location": "Berlin",
        "skills": ["python", "go", "kubernetes"],
        "url": "https://jobs.example/j2",
    },
    {
        "id": "J3",
        "title": "Data Engineer",
        "company": "Fernblick Ltd",
        "location": "Remote (UK)",
        "skills": ["python", "spark", "airflow"],
        "url": "https://jobs.example/j3",
    },
]


@tool
def search_jobs(query: str, location: str = "") -> list[dict[str, Any]]:
    """Search job listings for `query`, optionally filtered to `location`.

    Returns a list of jobs with id, title, company, location, skills and url.
    """
    query_words = query.lower().split()
    jobs = [
        j
        for j in _LISTINGS
        if any(w in (str(j["title"]) + " " + " ".join(j["skills"])).lower() for w in query_words)
    ]
    if location:
        jobs = [j for j in jobs if location.lower() in str(j["location"]).lower()]
    return jobs or _LISTINGS[:2]


@tool
def improve_resume_section(resume_text: str, job_title: str, job_skills: str) -> str:
    """Rewrite `resume_text` to emphasise experience matching `job_skills`.

    `job_skills` is a comma-separated string. Returns the improved section.
    """
    skills = [s.strip() for s in job_skills.split(",") if s.strip()]
    match = ", ".join(skills[:3])
    return f"{resume_text.strip()}\n- Tailored for {job_title}: delivered projects using {match}."


@tool
def submit_application(job_url: str, resume: str, cover_letter: str) -> dict[str, Any]:
    """Submit an application to `job_url`. Dry-run only: records intent, sends nothing.

    Returns the job url and a status. Wire a real submission endpoint here when
    the job board has an API, or keep this a human checkpoint.
    """
    return {"job_url": job_url, "status": "dry-run-saved", "cover_letter_chars": len(cover_letter)}
