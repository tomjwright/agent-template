from langchain_core.messages.ai import UsageMetadata

from agent_template.pricing import cost_usd


def tokens(input_tokens: int, output_tokens: int) -> UsageMetadata:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def test_cost_of_a_dated_model_name() -> None:
    # $1 per million input tokens and $5 per million output tokens.
    usage = {"claude-haiku-4-5-20251001": tokens(1_000_000, 100_000)}
    assert cost_usd(usage) == 1.5


def test_cost_adds_up_every_model() -> None:
    usage = {
        "claude-haiku-4-5": tokens(1_000_000, 0),
        "gemini-3.5-flash-lite": tokens(1_000_000, 0),
    }
    assert cost_usd(usage) == 1.3


def test_unpriced_model_has_no_cost() -> None:
    usage = {"claude-haiku-4-5": tokens(10, 10), "some-new-model": tokens(10, 10)}
    assert cost_usd(usage) is None
