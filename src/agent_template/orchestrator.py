"""The orchestrator: a supervisor that routes work between agent subgraphs.

Graph shape::

    START ──► supervisor ──► agent ──► supervisor ──► ... ──► END
                  │            ▲           │
                  └── FINISH ──┴───────────┘ (step cap)

The supervisor is an LLM call that reads the conversation and picks the next
agent (or FINISH). Each agent is a `create_agent` subgraph: it may call its
tools in a loop, then returns control. `max_steps` caps the loop so a confused
router can't run forever.

Every run has its own `thread_id`, the key its checkpoints are saved under, so
concurrent runs never share state. Pass a result's `thread_id` back to `run` to
continue that conversation, or to `resume` to finish a run that crashed.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from agent_template.agents import Agent
from agent_template.checkpoint import default_checkpointer
from agent_template.llm import chat_model
from agent_template.state import AgentState

FINISH = "FINISH"


class Route(BaseModel):
    """The supervisor's decision: which agent acts next."""

    next: str = Field(description=f"An agent name, or {FINISH!r} when the task is done.")
    reason: str = Field(description="One short sentence for the run log.")


ROUTER_PROMPT = """You route work between specialist agents.

Agents:
{roster}

Conversation so far:
{conversation}

Pick the agent best placed to make progress on the task, or {finish} if the
task is complete or cannot proceed. Never pick the same agent twice in a row
unless it is still doing useful work."""


@dataclass
class RunResult:
    task: str
    thread_id: str
    output: str
    steps: int
    transcript: list[dict[str, str]] = field(default_factory=list)


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
            return {"current_agent": FINISH, "steps": steps}
        route = self._pick_agent(state)
        return {"current_agent": route.next, "steps": steps}

    def _pick_agent(self, state: AgentState) -> Route:
        roster = "\n".join(f"- {a.name}: {a.description}" for a in self.agents.values())
        conversation = "\n".join(f"{m.type}: {_text(m)}" for m in state["messages"][-20:])
        prompt = ROUTER_PROMPT.format(roster=roster, conversation=conversation, finish=FINISH)
        structured = self.model.with_structured_output(Route)
        route = cast(Route, structured.invoke(prompt))
        if route.next != FINISH and route.next not in self.agents:
            route = Route(next=FINISH, reason=f"unknown agent {route.next!r}")
        return route

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
        output=_last_agent_text(state["messages"]),
        steps=state["steps"],
        transcript=transcript,
    )
