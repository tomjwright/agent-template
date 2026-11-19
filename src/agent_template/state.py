"""The shared state every agent in a run sees.

`messages` is the conversation thread: the task is injected as the first
HumanMessage and every agent appends to it, so later agents read what earlier
ones did. `data` is a plain dict for structured hand-off values (a list of
jobs, a drafted letter); keys merge shallowly, later writes win.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_data(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class AgentState(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    data: Annotated[dict[str, Any], merge_data]
    current_agent: str
    steps: int
