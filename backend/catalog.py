"""What models this deployment can actually reach, for the picker.

Two providers answer this question in completely different ways, and the
difference is not incidental.

**Bedrock is a list from configuration.** Not a discovery call, deliberately:
``ListFoundationModels`` returns what the *region* carries rather than what
this account may invoke, so it offers models that then fail with an access
error -- and it needs an IAM permission beyond invoking a model, which a
deployment granted only ``bedrock:InvokeModel`` does not have. An operator who
wants another one adds it to ``BEDROCK_MODELS``, where the region prefix is
visible and deliberate.

**OpenRouter is a live fetch.** It fronts several hundred models and changes
them weekly, so a hand-maintained list would be wrong within a fortnight --
and the whole reason to point at OpenRouter is to reach models nobody wrote
down in advance. The catalogue is cached (``OPENROUTER_CATALOG_TTL_SECONDS``)
because the alternative is a third-party HTTP request in front of a dashboard
page load.

The fetch also brings back **prices**, per token, per model. That matters more
than it sounds: ``pricing.py``'s table is a hand-maintained estimate of a
handful of Claude models, and it cannot possibly cover hundreds. Prices that
come from the provider are the provider's own numbers.

Never raises. A catalogue that cannot be fetched degrades to "Bedrock only,
and here is why OpenRouter is missing" -- which is a usable dashboard, whereas
a 500 on the model picker is not.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pricing import known_rates_for

log = logging.getLogger(__name__)

#: How long a failed fetch is remembered before trying again. Short enough
#: that fixing a key is noticed quickly, long enough that a dashboard left
#: open on a broken key does not hammer somebody else's API.
ERROR_TTL_SECONDS = 60

#: Hard cap on how many OpenRouter models are returned. The list is a dropdown
#: in a browser, and several hundred entries of JSON per page load is a cost
#: nobody chose. Ordered by the provider, which puts the interesting ones first.
MAX_MODELS = 400


@dataclass(slots=True)
class ModelInfo:
    """One model a person can pick."""

    provider: str
    id: str
    name: str = ""
    #: USD per million tokens, as (input, output). ``None`` where the provider
    #: does not say -- which is different from free, and rendered differently.
    prices: tuple[float, float] | None = None
    context: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "id": self.id,
            "name": self.name or self.id,
            "input_per_million": None if self.prices is None else self.prices[0],
            "output_per_million": None if self.prices is None else self.prices[1],
            "context": self.context,
        }


@dataclass(slots=True)
class Catalogue:
    """Everything the picker needs, in one reply."""

    models: list[ModelInfo] = field(default_factory=list)
    #: Per provider: why it is unavailable, or "" when it is fine. Shown in the
    #: picker rather than logged, because "OpenRouter is missing" with no
    #: reason is the kind of gap a person fills in with a guess.
    problems: dict[str, str] = field(default_factory=dict)

    def to_dict(self, default: Any) -> dict[str, Any]:
        return {
            "models": [model.to_dict() for model in self.models],
            "problems": self.problems,
            "default": {"provider": default.provider, "model": default.model},
        }


class ModelCatalogue:
    """Reads and caches what each provider offers."""

    def __init__(self, settings: Any, fetch: Any = None) -> None:
        self._settings = settings
        #: A test injects a fetcher here rather than a transport, so what is
        #: under test is the parsing and the caching rather than httpx.
        self._fetch = fetch
        self._openrouter: list[ModelInfo] | None = None
        self._problem = ""
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    async def read(self) -> Catalogue:
        # Bedrock quotes no prices over its API, so these come from the
        # hand-maintained table -- and only when it actually lists the model.
        # Showing the table's default as though it were this model's price is
        # how somebody budgets a batch against a figure nobody checked.
        models = [
            ModelInfo(
                provider="bedrock",
                id=model,
                name=_pretty_bedrock(model),
                prices=known_rates_for(model),
            )
            for model in self._settings.bedrock_models
        ]
        problems: dict[str, str] = {}

        if not self._settings.openrouter_api_key:
            problems["openrouter"] = (
                "OPENROUTER_API_KEY is not set, so no OpenRouter model can be "
                "reached. Add it to the environment (or .env) and restart the backend."
            )
            return Catalogue(models=models, problems=problems)

        fetched = await self._openrouter_models()
        if self._problem:
            problems["openrouter"] = self._problem
        return Catalogue(models=[*models, *fetched], problems=problems)

    async def prices_for(self, model: str) -> tuple[float, float] | None:
        """What OpenRouter says this model costs, if it said anything.

        Read by the pricing layer so a session on a model nobody hand-listed
        is still costed from real numbers rather than from a default.
        """
        for info in await self._openrouter_models():
            if info.id == model:
                return info.prices
        return None

    # -- internals ----------------------------------------------------------
    async def _openrouter_models(self) -> list[ModelInfo]:
        if not self._settings.openrouter_api_key:
            return []
        ttl = (
            ERROR_TTL_SECONDS
            if self._problem
            else self._settings.openrouter_catalog_ttl_seconds
        )
        if self._openrouter is not None and time.monotonic() - self._fetched_at < ttl:
            return self._openrouter

        async with self._lock:
            # Re-checked under the lock: several dashboard tabs opening at once
            # would otherwise each fetch the same several hundred models.
            if self._openrouter is not None and time.monotonic() - self._fetched_at < ttl:
                return self._openrouter
            self._openrouter, self._problem = await self._load()
            self._fetched_at = time.monotonic()
            return self._openrouter

    async def _load(self) -> tuple[list[ModelInfo], str]:
        try:
            payload = await (self._fetch or self._http_get)()
        except Exception as exc:  # noqa: BLE001 - a picker must still render
            log.warning("could not read the OpenRouter catalogue", exc_info=True)
            return [], f"OpenRouter did not answer: {str(exc)[:200]}"
        try:
            return _parse(payload), ""
        except Exception as exc:  # noqa: BLE001
            log.warning("could not parse the OpenRouter catalogue", exc_info=True)
            return [], f"OpenRouter's model list could not be read: {str(exc)[:200]}"

    async def _http_get(self) -> dict[str, Any]:
        import httpx

        url = f"{self._settings.openrouter_base_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {self._settings.openrouter_api_key}"},
            )
            response.raise_for_status()
            return response.json()


def _parse(payload: dict[str, Any]) -> list[ModelInfo]:
    """OpenRouter's ``/models`` reply as this codebase's shape.

    Prices arrive as USD *per token*, as strings, which is a number so small
    that reading it as anything but a string loses precision. Converted to per
    million here, which is the unit every other price in this codebase and on
    every provider's pricing page is quoted in.
    """
    models: list[ModelInfo] = []
    for entry in (payload.get("data") or [])[:MAX_MODELS]:
        model_id = str(entry.get("id") or "").strip()
        if not model_id:
            continue
        models.append(
            ModelInfo(
                provider="openrouter",
                id=model_id,
                name=str(entry.get("name") or model_id),
                prices=_prices(entry.get("pricing") or {}),
                context=_int_or_none(entry.get("context_length")),
            )
        )
    return models


def _prices(pricing: dict[str, Any]) -> tuple[float, float] | None:
    prompt = _float_or_none(pricing.get("prompt"))
    completion = _float_or_none(pricing.get("completion"))
    if prompt is None or completion is None:
        return None
    return (prompt * 1_000_000, completion * 1_000_000)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pretty_bedrock(model: str) -> str:
    """A Bedrock id as something readable, without losing the id itself.

    ``us.anthropic.claude-haiku-4-5-20251001`` says four useful things and
    three that only matter when something breaks, and a dropdown is not where
    a person wants to read a region prefix and a date stamp.
    """
    name = model.split(".")[-1]
    for vendor in ("anthropic-", "amazon-", "meta-", "mistral-"):
        name = name.removeprefix(vendor)
    return name.replace("-", " ")


__all__ = ["Catalogue", "ModelCatalogue", "ModelInfo"]
