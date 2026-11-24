"""Model prices, so each run can record what it cost.

USD per million tokens, (input, cached input, output), from the providers'
pricing pages. Cached input is the part of the input read from the prompt cache.
Add the models your app uses; a run on an unlisted model records no cost.
"""

from langchain_core.messages.ai import UsageMetadata

PRICES_PER_MILLION = {
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
    "gemini-3.5-flash-lite": (0.30, 0.03, 2.50),
}


def cost_usd(usage: dict[str, UsageMetadata]) -> float | None:
    """The cost of the token usage per model, or None if a model has no price."""
    total = 0.0
    for model_name, tokens in usage.items():
        price = _price(model_name)
        if price is None:
            return None
        input_price, cached_price, output_price = price
        cached = tokens.get("input_token_details", {}).get("cache_read", 0)
        total += (tokens["input_tokens"] - cached) * input_price
        total += cached * cached_price
        total += tokens["output_tokens"] * output_price
    return round(total / 1_000_000, 6)


def _price(model_name: str) -> tuple[float, float, float] | None:
    """Match on the longest listed prefix, so dated names like
    claude-haiku-4-5-20251001 find their model."""
    matches = [name for name in PRICES_PER_MILLION if model_name.startswith(name)]
    if not matches:
        return None
    return PRICES_PER_MILLION[max(matches, key=len)]
