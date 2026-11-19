from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from pydantic import Field


class FakeModel(GenericFakeChatModel):
    """A chat model that cycles canned replies and fakes structured output.

    `with_structured_output` returns an object whose `.invoke` pops the next
    Route from `routes`, so tests can script the supervisor's decisions.
    """

    routes: list[Any] = Field(default_factory=list)
    seen_prompts: list[str] = Field(default_factory=list)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        model = self

        class Structured:
            def invoke(self, prompt: Any) -> Any:
                model.seen_prompts.append(str(prompt))
                if model.routes:
                    return model.routes.pop(0)
                raise AssertionError("FakeModel ran out of routes")

        return Structured()
