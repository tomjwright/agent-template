# agent-template

A template for building multi-agent applications. Clone it, define your agents
as Python classes, and the orchestrator routes work between them.

- **Agents as subgraphs.** Each agent is a LangChain agent (model + tools +
  system prompt), compiled with `create_agent`, so it can call its tools in a
  loop before handing control back.
- **Supervisor routing.** A LangGraph supervisor reads the conversation and
  picks the next agent (or FINISH). Agents keep their own tools; the router
  only sees names and descriptions. Its answer is limited to the registered
  names; a run whose router misses twice ends with `status="failed"`.
- **Shared state.** One message thread plus a `data` dict for structured
  hand-off values.
- **Provider-agnostic.** `LLM_MODEL` is a `provider:model` spec
  (`anthropic:claude-haiku-4-5` by default); any LangChain provider works.
- **Checked in.** JSON logs with trace IDs, optional OTLP tracing, run history
  (SQLite locally, Postgres when `DATABASE_URL` is set), and a YAML-driven eval
  suite that gates merges in CI. Each run row records its status, model,
  tokens, cost, latency, trace id, user and tenant.

## Quick start

```bash
git clone <this repo> my-project && cd my-project
uv sync
cp .env.example .env    # add ANTHROPIC_API_KEY
uv run python -m agent_template run --app examples.job_finder.app \
    "Find me a Python job in Berlin and apply to the best one."
```

## Layout

```
src/agent_template/
  agents.py        Agent: name, description, system prompt, tools -> subgraph
  orchestrator.py  supervisor node + agent nodes, step cap, route retry, RunResult, resume
  checkpoint.py    run checkpoints: Postgres when DATABASE_URL is set, else memory
  state.py         shared AgentState (messages, data, routing fields)
  llm.py           provider-agnostic chat model from LLM_MODEL
  cli.py           `run` an app, `history` of past runs
  evals.py         run evals/cases.yaml, fail below --min-score
  store.py         run history: Postgres when DATABASE_URL is set, else SQLite
  pricing.py       model prices for each run's cost_usd
  telemetry.py     JSON logs + optional OTLP tracing
examples/job_finder/   a complete app: finder, resume improver, applicant
tests/              offline tests (fake models, no LLM calls)
```

## Write your own app

1. Write tools: plain functions with `@tool` (see `examples/job_finder/tools.py`).
2. Define agents in an `app.py`: an `Agent` per specialist, each with a
   `description` written for the router (see `build_orchestrator`).
3. `uv run python -m agent_template run --app your_package.app "your task"`.

## Checks

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run pre-commit run --all-files
```

See [AGENTS.md](AGENTS.md) for the workflow rules and [GLOSSARY.md](GLOSSARY.md)
for the project's vocabulary.
