"""What a model turn costs, in one place.

Two things in this application spend tokens: an agent authoring session, and a
healer re-finding a control mid-replay. Both need to price what they used, and
before this module they were the only two places that could have -- so pricing
lived in the agent package, where the healer could not reach it without
importing an optional dependency group it has nothing to do with.

**These figures are estimates and the AWS bill is the authority.** They exist
so a person can see roughly what a session cost before it finishes and set a
limit in the unit they actually budget in. A ceiling built on them is a
guard rail, not an accounting control.
"""

from __future__ import annotations

from typing import Any

#: USD per million tokens, as (input, output). Matched loosely on the model id
#: because a real Bedrock id carries a region prefix and a version suffix --
#: `us.anthropic.claude-sonnet-5-20260514-v1:0` -- so an exact lookup would
#: fall through to the default for every actual deployment.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
}

#: What an unlisted model is assumed to cost. Deliberately not zero: an
#: estimate of $0.00 for a four-thousand-row batch is the most expensive kind
#: of wrong, and a model missing from the table above is far more likely to be
#: new than to be free.
DEFAULT_PRICE: tuple[float, float] = (3.0, 15.0)

#: What Bedrock's prompt cache changes about the input rate. A cache read is
#: nearly free because the provider skips reprocessing that prefix; a cache
#: write costs a little more than an ordinary token, because writing the cache
#: is itself work. Anthropic publishes these as fixed multipliers of the base
#: input rate, the same ratio for every model, which is why they live here
#: rather than in the per-model table above.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def rates_for(model: str) -> tuple[float, float]:
    for known, rates in PRICES.items():
        if known in (model or ""):
            return rates
    return DEFAULT_PRICE


def price_of(model: str, usage: dict[str, Any]) -> float:
    """Dollars for one turn, from its own input/output split.

    ``input_tokens`` already includes any cache read and cache write -- see
    ``chat.usage_of`` -- so the fresh (regular-priced) portion is what is left
    after taking those back out. Without this, enabling the cache would have
    made every session look no cheaper than before it was turned on: the same
    total token count, priced as if none of it had been a cache hit.
    """
    read, written = rates_for(model)
    tokens_in = int(usage.get("input_tokens") or 0)
    tokens_out = int(usage.get("output_tokens") or 0)
    cache_read = int(usage.get("cache_read_tokens") or 0)
    cache_write = int(usage.get("cache_creation_tokens") or 0)
    fresh = max(0, tokens_in - cache_read - cache_write)
    return (
        fresh * read
        + cache_read * read * CACHE_READ_MULTIPLIER
        + cache_write * read * CACHE_WRITE_MULTIPLIER
        + tokens_out * written
    ) / 1_000_000


def price_of_total(model: str, tokens: int, *, output_share: float = 0.2) -> float:
    """Dollars for a token count whose split was not kept.

    A worse answer than :func:`price_of` and used only where the split is
    genuinely gone. The share is stated rather than hidden because output
    tokens cost roughly five times input ones, so the assumption is most of the
    answer -- and it is deliberately generous, since a governance number that
    under-reports is the one that lets a bill through.
    """
    read, written = rates_for(model)
    tokens = max(0, int(tokens))
    out = tokens * output_share
    return ((tokens - out) * read + out * written) / 1_000_000


__all__ = ["DEFAULT_PRICE", "PRICES", "price_of", "price_of_total", "rates_for"]
