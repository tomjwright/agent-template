"""Chat model construction, provider-agnostic.

`LLM_MODEL` is a "provider:model" spec, e.g. "anthropic:claude-haiku-4-5".
Any LangChain provider works once its package (e.g. langchain-anthropic) and
API key are installed.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "anthropic:claude-haiku-4-5"


def chat_model(model_spec: str | None = None) -> BaseChatModel:
    """A chat model from LLM_MODEL, or `model_spec`, or the default."""
    load_dotenv()
    spec = model_spec or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    return init_chat_model(spec, temperature=0)
