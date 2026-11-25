# AGENTS.md

Guidance for coding agents working in this repo. Humans: start with
[README.md](README.md).

## Standards

The workflow rules shared by all of Tom's repos (branches, PR size and
descriptions, testing, secrets) live in his home repo, `tomjwright/home`.
Rules for this repo:

- This is a template: keep the skeleton thin. Project-specific behaviour goes
  in `examples/`, never in `src/agent_template/`.
- Agents communicate through the shared message thread and the `data` state
  dict. Do not add private channels between agents.
- The supervisor is the only router. Do not hard-code agent-to-agent edges.
- `.env` is git-ignored; CI reads GitHub secrets. `ANTHROPIC_API_KEY` is the
  default secret (`LLM_MODEL` selects the provider), but the evals workflow
  runs on Gemini (`GOOGLE_API_KEY`) via `LLM_MODEL`.

## Checks

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest --cov    # fails below the coverage floor (statements and branches combined) in pyproject.toml
uv run pre-commit run --all-files
```

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues in this repo; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical labels, applied as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` at the root and `docs/adr/` (created lazily by `/domain-modeling`). See `docs/agents/domain.md`.

## Commands

```bash
uv run python -m agent_template run --app examples.job_finder.app "task"
uv run python -m agent_template history
uv run python -m agent_template.evals --app examples.job_finder.app   # needs API key
```

## Architecture in one line

Task → supervisor (LLM router) → agent subgraph (model + tools loop) →
supervisor → … → FINISH; every message lands on one shared thread (one
`thread_id` per conversation, checkpointed to Postgres when `DATABASE_URL` is
set), runs are recorded in Postgres when `DATABASE_URL` is set (else SQLite),
logs are JSON with trace IDs.

## Gotchas

- `examples/` is importable from the repo root (`python -m` puts the root on
  `sys.path`); it is intentionally not in the wheel.
- Tests never call an LLM: fake models come from
  `langchain_core.language_models.fake_chat_models`; script supervisor routes
  with `FakeModel.routes` in `tests/conftest.py`.
- Postgres tests skip unless `TEST_DATABASE_URL` is set; CI sets it. Locally:
  `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:17`, then
  `TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres uv run pytest`.
- The supervisor picks agents by `description`, not name — vague descriptions
  cause misrouting.
