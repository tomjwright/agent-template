# Agent template

A codebase template for building agentic applications: clone it, define agents
as Python classes, and assemble them into a tool.

## Language

**Template**:
The repo a new agentic project is cloned from. It holds the skeleton, conventions and one complete example app.
_Avoid_: boilerplate, scaffold, starter

**Skeleton**:
The thin part of the template meant to be copied into a new project: package layout, agent base class, CLI entrypoint, tests.
_Avoid_: core, framework

**Example app**:
A complete working multi-agent application shipped inside the template (the job-application finder) that shows how agents, tools and the orchestrator fit together.
_Avoid_: demo, sample, tutorial

**Agent**:
A Python class that does one kind of work for the system, with its own prompt and tools, registered with the orchestrator.
_Avoid_: bot, worker, session

**Orchestrator**:
The LangGraph layer that routes work between agents and holds the shared state they pass through.
_Avoid_: manager, supervisor, controller
