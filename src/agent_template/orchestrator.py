"""The orchestrator: a supervisor that routes work between agent subgraphs.

Graph shape::

    START ──► supervisor ──► agent ──► supervisor ──► ... ──► END
                  │            ▲           │
                  └── FINISH ──┴───────────┘ (step cap)

The supervisor is an LLM call that reads the conversation and picks the next
agent (or FINISH). Each agent is a `create_agent` subgraph: it may call its
tools in a loop, then returns control. `max_steps` caps the loop so a confused
router can't run forever.
"""

from collections.abc import Hashable, Sequence
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from agent_template.agents import Agent
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


class RunResult(BaseModel):
    """What one run produced: the final answer plus the full transcript."""

    task: str
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
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self._graph: CompiledStateGraph[Any, Any, Any, Any] | None = None

    @property
    def graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        if self._graph is None:
            self._graph = self._build()
        return self._graph

    def run(self, task: str, data: dict[str, Any] | None = None) -> RunResult:
        """Run the system on `task`. `data` seeds the shared state dict."""
        brief = task if not data else f"{task}\n\nContext:\n{data}"
        state = self.graph.invoke(
            {
                "task": task,
                "messages": [HumanMessage(content=brief)],
                "data": data or {},
                "current_agent": "",
                "steps": 0,
            },
            config={"configurable": {"thread_id": "run"}},
        )
        transcript = [
            {"agent": m.name or "user", "text": _text(m)}
            for m in state["messages"]
            if isinstance(m, HumanMessage | AIMessage)
        ]
        return RunResult(
            task=task,
            output=_last_agent_text(state["messages"]),
            steps=state["steps"],
            transcript=transcript,
        )

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
        if route.next == FINISH:
            return {"current_agent": FINISH, "steps": steps}
        # The handoff is a user-role turn, so the agent's model never receives a
        # thread that ends with another agent's reply (Gemini rejects that).
        handoff = HumanMessage(
            content=f"{route.next}, over to you: {route.reason}", name="supervisor"
        )
        return {"current_agent": route.next, "steps": steps, "messages": [handoff]}

    def _pick_agent(self, state: AgentState) -> Route:
        roster = "\n".join(f"- {a.name}: {a.description}" for a in self.agents.values())
        # Leave out the supervisor's own handoffs, so they don't push the task
        # out of the 20-message window.
        messages = [m for m in state["messages"] if m.name != "supervisor"]
        conversation = "\n".join(f"{m.type}: {_text(m)}" for m in messages[-20:])
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
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _text(message)
    return ""
