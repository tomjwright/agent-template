# Example: job application finder

A complete multi-agent app built on the template:

- **JobFinderAgent** searches listings (`search_jobs` — canned data).
- **ResumeImproverAgent** tailors a resume section to the chosen job.
- **JobApplicationAgent** drafts a cover letter and submits (dry-run).

The supervisor decides who acts next; each agent only sees the tools it owns.

## Run

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run python -m agent_template run --app examples.job_finder.app \
    "Find me a Python job in Berlin and apply to the best one."
```

## Evals

```bash
uv run python -m agent_template.evals --app examples.job_finder.app
```

## Make it yours

Swap the canned listings in `tools.py` for a real job-board API, write a real
submission endpoint in `submit_application`, and add agents in `app.py`.
