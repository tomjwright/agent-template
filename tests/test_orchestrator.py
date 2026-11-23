"""Orchestrator routing, tested with fake models — no LLM calls."""

from typing import Any

import pytest
from conftest import FakeModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import Field

from agent_template.agents import Agent
from agent_template.orchestrator import FINISH, Orchestrator, Route


def make_model(replies: list[str], routes: list[Route] | None = None) -> FakeModel:
    model = FakeModel(messages=iter([AIMessage(content=r) for r in replies]))
    model.routes = list(routes or [])
    model.seen_prompts = []
    return model


def echo_agent(name: str, description: str = "does a thing") -> Agent:
    return Agent(name=name, description=description, system_prompt="help")


def test_rejects_empty_agent_list() -> None:
    with pytest.raises(ValueError, match="at least one agent"):
        Orchestrator([], model=GenericFakeChatModel(messages=iter([])))


def test_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate agent names"):
        Orchestrator(
            [echo_agent("A"), echo_agent("A")],
            model=GenericFakeChatModel(messages=iter([])),
        )


def test_supervisor_routes_to_named_agent_then_finishes() -> None:
    model = make_model(
        replies=["I did the work."],
        routes=[
            Route(next="WorkerAgent", reason="to work"),
            Route(next=FINISH, reason="done"),
        ],
    )
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=model)
    result = orchestrator.run("do something")
    assert result.output == "I did the work."
    assert result.steps == 2


def test_unknown_agent_name_finishes() -> None:
    model = make_model(replies=[], routes=[Route(next="NoSuchAgent", reason="lost")])
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=model)
    result = orchestrator.run("do something")
    assert result.steps == 1


def test_step_cap_stops_a_looping_router() -> None:
    model = make_model(
        replies=["tick", "tick", "tick", "tick"],
        routes=[Route(next="WorkerAgent", reason="loop") for _ in range(10)],
    )
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=model, max_steps=3)
    result = orchestrator.run("do something")
    assert result.steps == 4  # one past the cap triggers FINISH


def test_router_sees_agent_descriptions() -> None:
    model = make_model(
        replies=["done"],
        routes=[Route(next="WorkerAgent", reason="x"), Route(next=FINISH, reason="y")],
    )
    orchestrator = Orchestrator([echo_agent("WorkerAgent", "files paperwork")], model=model)
    orchestrator.run("file the forms")
    prompt = model.seen_prompts[0]
    assert "WorkerAgent" in prompt and "files paperwork" in prompt
    assert "file the forms" in prompt


class RecordingModel(FakeModel):
    """Remembers the last message of every request it receives."""

    last_messages: list[BaseMessage] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.last_messages.append(messages[-1])
        return super()._generate(messages, *args, **kwargs)


def test_every_agent_turn_starts_from_a_user_role_message() -> None:
    replies = [AIMessage(content="found"), AIMessage(content="applied")]
    model = RecordingModel(messages=iter(replies))
    model.routes = [
        Route(next="FinderAgent", reason="find jobs"),
        Route(next="ApplyAgent", reason="apply to the best one"),
        Route(next=FINISH, reason="done"),
    ]
    orchestrator = Orchestrator([echo_agent("FinderAgent"), echo_agent("ApplyAgent")], model=model)
    result = orchestrator.run("find and apply")
    assert [type(m) for m in model.last_messages] == [HumanMessage, HumanMessage]
    assert model.last_messages[1].content == "ApplyAgent, over to you: apply to the best one"
    assert result.output == "applied"
    assert "over to you" not in model.seen_prompts[-1]
