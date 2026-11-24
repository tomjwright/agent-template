"""The orchestrator: a supervisor that routes work between agent subgraphs.

Graph shape::

    START ──► supervisor ──► agent ──► supervisor ──► ... ──► END
                  │            ▲           │
                  └── FINISH ──┴───────────┘ (step cap)

The supervisor is an LLM call that reads the conversation and picks the next
agent (or FINISH). Each agent is a `create_agent` subgraph: it may call its
tools in a loop, then returns control. `max_steps` caps the loop so a confused
router can't run forever.

The router's answer is limited to the registered agent names plus FINISH. If it
still names no registered agent, it gets one retry with the valid names. A
second miss ends the run with status "failed" and counts toward the
`router.invalid_routes` metric, instead of passing for a finished run.

Every run has its own `thread_id`, the key its checkpoints are saved under, so
concurrent runs never share state. Pass a result's `thread_id` back to `run` to
continue that conversation, or to `resume` to finish a run that crashed.
"""

import logging
from collections.abc import Hashable, Sequence
from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from opentelemetry import metrics
from pydantic import BaseModel, Field, ValidationError, create_model

from agent_template.agents import Agent
from agent_template.checkpoint import default_checkpointer
from agent_template.llm import chat_model
from agent_template.state import AgentState

FINISH = "FINISH"

log = logging.getLogger(__name__)
INVALID_ROUTES = metrics.get_meter(__name__).create_counter(
    "router.invalid_routes", description="Router answers that named no registered agent."
)

Status = Literal["running", "completed", "failed", "step_cap_hit"]


class Route(BaseModel):
    """The supervisor's decision: which agent acts next."""

    next: str = Field(description=f"An agent name, or {FINISH!r} when the task is done.")
    reason: str = Field(description="One short sentence for the run log.")


def route_schema(agent_names: Sequence[str]) -> type[Route]:
    """`Route` with `next` limited to these agents or FINISH, so the model can't invent one."""
    choices: Any = Literal[tuple([*agent_names, FINISH])]
    next_field = Field(description=f"An agent name, or {FINISH!r} when the task is done.")
    return create_model("Route", __base__=Route, next=(choices, next_field))


ROUTER_PROMPT = """You route work between specialist agents.

Agents:
{roster}

Conversation so far:
{conversation}

Pick the agent best placed to make progress on the task, or {finish} if the
task is complete or cannot proceed. Never pick the same agent twice in a row
unless it is still doing useful work."""

RETRY_PROMPT = """

Your last answer, {answer}, is not a valid choice. Reply with exactly one of: {choices}."""


class RunResult(BaseModel):
    """What one run produced: the final answer plus the full transcript."""

    task: str
    thread_id: str
    status: Status
    output: str
    steps: int
    transcript: list[dict[str, str]] = Field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        agents: Sequence[Agent],
        model: BaseChatModel | None = None,
        max_steps: int = 12,
        checkpointer: Any | None = None,
    ) -> None:
        if not agents:
            raise ValueError("an orchestrator needs at least one agent")
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate agent names: {names}")
        self.agents = {a.name: a for a in agents}
        self.model = model or chat_model()
        self.max_steps = max_steps
        self.route_schema = route_schema(names)
        self.checkpointer = checkpointer if checkpointer is not None else default_checkpointer()
        self._graph: CompiledStateGraph[Any, Any, Any, Any] | None = None

    @property
    def graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        if self._graph is None:
            self._graph = self._build()
        return self._graph

    def run(
        self, task: str, data: dict[str, Any] | None = None, thread_id: str | None = None
    ) -> RunResult:
        """Run the system on `task`. `data` seeds the shared state dict.

        Without `thread_id` this starts a new conversation with a fresh id.
        With one, `task` is added to that conversation's thread.
        """
        thread_id = thread_id or str(uuid4())
        brief = task if not data else f"{task}\n\nContext:\n{data}"
        try:
            state = self.graph.invoke(
                {
                    "task": task,
                    "messages": [HumanMessage(content=brief)],
                    "data": data or {},
                    "current_agent": "",
                    "steps": 0,
                    "status": "running",
                },
                config=_config(thread_id),
            )
        except Exception as error:
            error.add_note(f"Resume this run with Orchestrator.resume({thread_id!r}).")
            raise
        return _result(thread_id, state)

    def resume(self, thread_id: str) -> RunResult:
        """Finish a run that stopped part-way (crash, restart) from its last checkpoint."""
        config = _config(thread_id)
        if not self.graph.get_state(config).values:
            raise ValueError(f"no run with thread_id {thread_id!r}")
        state = self.graph.invoke(None, config=config)
        return _result(thread_id, state)

    # -- graph construction -------------------------------------------------

    def _build(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        builder = StateGraph(AgentState)
        builder.add_node("supervisor", self._supervisor)
        for agent in self.agents.values():
            builder.add_node(agent.name, agent.build(self.model))
        builder.add_edge(START, "supervisor")
        routes: dict[Hashable, str] = {name: name for name in self.agents}
        routes[FINISH] = END
        builder.add_conditional_edges("supervisor", self._route_edge, routes)
        for name in self.agents:
            builder.add_edge(name, "supervisor")
        return builder.compile(checkpointer=self.checkpointer)

    def _supervisor(self, state: AgentState) -> dict[str, Any]:
        steps = state["steps"] + 1
        if steps > self.max_steps:
            return {"current_agent": FINISH, "steps": steps, "status": "step_cap_hit"}
        route = self._pick_agent(state)
        if route is None:
            return {"current_agent": FINISH, "steps": steps, "status": "failed"}
        if route.next == FINISH:
            return {"current_agent": FINISH, "steps": steps, "status": "completed"}
        # The handoff is a user-role turn, so the agent's model never receives a
        # thread that ends with another agent's reply (Gemini rejects that).
        handoff = HumanMessage(
            content=f"{route.next}, over to you: {route.reason}", name="supervisor"
        )
        return {"current_agent": route.next, "steps": steps, "messages": [handoff]}

    def _pick_agent(self, state: AgentState) -> Route | None:
        """The router's next route, or None if it named no registered agent twice."""
        roster = "\n".join(f"- {a.name}: {a.description}" for a in self.agents.values())
        # Leave out the supervisor's own handoffs, so they don't push the task
        # out of the 20-message window.
        messages = [m for m in state["messages"] if m.name != "supervisor"]
        conversation = "\n".join(f"{m.type}: {_text(m)}" for m in messages[-20:])
        prompt = ROUTER_PROMPT.format(roster=roster, conversation=conversation, finish=FINISH)
        structured = self.model.with_structured_output(self.route_schema)
        choices = ", ".join([*self.agents, FINISH])
        for attempt in (1, 2):
            try:
                route = cast(Route | None, structured.invoke(prompt))
            except (OutputParserException, ValidationError):
                route = None
            if route is not None and (route.next == FINISH or route.next in self.agents):
                return route
            answer = repr(route.next) if route is not None else "an answer that did not parse"
            INVALID_ROUTES.add(1, {"attempt": attempt})
            log.warning("router gave an invalid route: %s (attempt %d of 2)", answer, attempt)
            prompt += RETRY_PROMPT.format(answer=answer, choices=choices)
        return None

    @staticmethod
    def _route_edge(state: AgentState) -> str:
        return state["current_agent"]


def _text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(str(part) for part in content)


def _last_agent_text(messages: list[Any]) -> str:
    """The latest agent reply to the latest task, or "" if no agent answered it."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return ""
        if isinstance(message, AIMessage):
            return _text(message)
    return ""


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _result(thread_id: str, state: dict[str, Any]) -> RunResult:
    transcript = [
        {"agent": m.name or "user", "text": _text(m)}
        for m in state["messages"]
        if isinstance(m, HumanMessage | AIMessage)
    ]
    return RunResult(
        task=state["task"],
        thread_id=thread_id,
        status=state["status"],
        output=_last_agent_text(state["messages"]),
        steps=state["steps"],
        transcript=transcript,
    )
