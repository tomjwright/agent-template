"""An Agent: one specialist in the system.

Each agent is a LangChain agent (model + tools + system prompt) compiled to a
graph, so it can call its tools in a loop before handing control back to the
orchestrator. The `description` is what the supervisor reads to decide who
should act next, so write it for the router, not for humans.
"""

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: Sequence[BaseTool | Any] = (),
        model: BaseChatModel | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = list(tools)
        self.model = model

    def build(self, default_model: BaseChatModel) -> CompiledStateGraph[Any, Any, Any, Any]:
        """The agent as a runnable graph node for the orchestrator.

        Uses the agent's own `model` when set, else the orchestrator's.
        """
        return create_agent(
            model=self.model or default_model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            name=self.name,
        )
