"""Orchestrator routing, tested with fake models — no LLM calls."""

import re
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
    assert result.model_dump()["transcript"][-1] == {
        "agent": "WorkerAgent",
        "text": "I did the work.",
    }


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


def test_each_run_gets_its_own_thread() -> None:
    model = make_model(
        replies=["first answer", "second answer"],
        routes=[
            Route(next="WorkerAgent", reason="a"),
            Route(next=FINISH, reason="done"),
            Route(next="WorkerAgent", reason="b"),
            Route(next=FINISH, reason="done"),
        ],
    )
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=model)
    first = orchestrator.run("task one")
    second = orchestrator.run("task two")
    assert first.thread_id != second.thread_id
    assert [t["text"] for t in second.transcript if t["agent"] != "supervisor"] == [
        "task two",
        "second answer",
    ]


def test_same_thread_id_continues_the_conversation() -> None:
    model = make_model(
        replies=["first answer", "second answer"],
        routes=[
            Route(next="WorkerAgent", reason="a"),
            Route(next=FINISH, reason="done"),
            Route(next="WorkerAgent", reason="b"),
            Route(next=FINISH, reason="done"),
        ],
    )
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=model)
    first = orchestrator.run("task one")
    second = orchestrator.run("follow up", thread_id=first.thread_id)
    assert second.thread_id == first.thread_id
    texts = [t["text"] for t in second.transcript if t["agent"] != "supervisor"]
    assert texts == ["task one", "first answer", "follow up", "second answer"]
    assert second.steps == 2  # the step cap counts per run, not per conversation


def test_follow_up_that_no_agent_answers_has_empty_output() -> None:
    model = make_model(
        replies=["first answer"],
        routes=[
            Route(next="WorkerAgent", reason="a"),
            Route(next=FINISH, reason="done"),
            Route(next=FINISH, reason="nothing to do"),
        ],
    )
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=model)
    first = orchestrator.run("task one")
    second = orchestrator.run("follow up", thread_id=first.thread_id)
    assert second.output == ""  # not the previous turn's answer


class CrashOnceModel(FakeModel):
    """Raises on its first reply, like a process dying mid-run."""

    crashed: bool = False

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        if not self.crashed:
            self.crashed = True
            raise RuntimeError("process died")
        return super()._generate(*args, **kwargs)


def test_resume_finishes_a_run_that_crashed() -> None:
    model = CrashOnceModel(messages=iter([AIMessage(content="done after restart")]))
    model.routes = [Route(next="WorkerAgent", reason="work"), Route(next=FINISH, reason="done")]
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=model)
    with pytest.raises(RuntimeError, match="process died"):
        orchestrator.run("do something", thread_id="t-1")
    result = orchestrator.resume("t-1")
    assert result.output == "done after restart"
    assert result.steps == 2  # the supervisor's first step was not repeated


def test_a_crashed_run_reports_its_generated_thread_id() -> None:
    model = CrashOnceModel(messages=iter([AIMessage(content="done after restart")]))
    model.routes = [Route(next="WorkerAgent", reason="work"), Route(next=FINISH, reason="done")]
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=model)
    with pytest.raises(RuntimeError, match="process died") as crash:
        orchestrator.run("do something")
    found = re.search(r"resume\('([^']+)'\)", "\n".join(crash.value.__notes__))
    assert found
    thread_id = found.group(1)
    assert orchestrator.resume(thread_id).output == "done after restart"


def test_resume_unknown_thread_raises() -> None:
    orchestrator = Orchestrator([echo_agent("WorkerAgent")], model=make_model([]))
    with pytest.raises(ValueError, match="no run with thread_id"):
        orchestrator.resume("missing")


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
