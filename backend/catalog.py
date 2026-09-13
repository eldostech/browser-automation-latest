"""What models this deployment can actually reach, for the picker.

Two providers answer this question in completely different ways, and the
difference is not incidental.

**Bedrock is discovered, and the configured list is kept.** This was
configuration only, on two objections that were true and are now handled rather
than avoided. ``ListFoundationModels`` reports what the *region* carries rather
than what the account may invoke -- so discovery is additive, and the picker's
check button, which makes one real call, is the only thing that can tell
"listed" from "invokable". And it needs an IAM permission beyond invoking a
model -- so a refusal degrades to ``BEDROCK_MODELS`` and names the permission,
rather than emptying the picker.

What that cost while it stood: a dashboard offering three Claude models on an
account that can reach eighty-nine, across seventeen providers.

Two calls, not one. A model that supports only ``INFERENCE_PROFILE`` cannot be
invoked by its own id -- the profile id, ``us.anthropic...``, is what works --
so ``ListInferenceProfiles`` is read too and the picker offers the id that will
actually run.

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
        #: Discovered Bedrock models, cached apart from OpenRouter's: the two
        #: change on different schedules, and a failure in one must not empty
        #: the other.
        self._bedrock: list[ModelInfo] | None = None
        self._bedrock_problem = ""
        self._bedrock_at = 0.0
        self._bedrock_lock = asyncio.Lock()

    async def read(self) -> Catalogue:
        problems: dict[str, str] = {}
        models = await self._bedrock_models()
        if self._bedrock_problem:
            problems["bedrock"] = self._bedrock_problem

        if not self._settings.openrouter_enabled:
            # Said rather than omitted. A picker with no OpenRouter in it and
            # no reason given looks broken, and the reason is one line in the
            # environment.
            problems["openrouter"] = (
                "OpenRouter is switched off in this deployment "
                "(OPENROUTER_ENABLED=false), so nothing here will call it."
            )
            return Catalogue(models=models, problems=problems)
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

    # -- Bedrock ------------------------------------------------------------
    def _configured(self) -> list[ModelInfo]:
        """The models named in BEDROCK_MODELS, offered whatever discovery says.

        Bedrock quotes no prices over its API, so these come from the
        hand-maintained table -- and only when it actually lists the model.
        Showing the table's default as though it were this model's price is how
        somebody budgets a batch against a figure nobody checked.
        """
        return [
            ModelInfo(
                provider="bedrock",
                id=model,
                name=_pretty_bedrock(model),
                prices=known_rates_for(model),
            )
            for model in self._settings.bedrock_models
        ]

    async def _bedrock_models(self) -> list[ModelInfo]:
        """The configured models first, then whatever the account can list."""
        configured = self._configured()
        if not self._settings.bedrock_discover:
            return configured

        ttl = (
            ERROR_TTL_SECONDS
            if self._bedrock_problem
            else self._settings.bedrock_catalog_ttl_seconds
        )
        if self._bedrock is None or time.monotonic() - self._bedrock_at >= ttl:
            async with self._bedrock_lock:
                if self._bedrock is None or time.monotonic() - self._bedrock_at >= ttl:
                    self._bedrock, self._bedrock_problem = await self._discover()
                    self._bedrock_at = time.monotonic()

        known = {info.id for info in configured}
        return [*configured, *(m for m in self._bedrock or [] if m.id not in known)]

    async def _discover(self) -> tuple[list[ModelInfo], str]:
        """Ask Bedrock what it carries. Never raises.

        On a thread, because both calls are blocking boto3 and this runs inside
        a request handler.
        """
        try:
            return await asyncio.to_thread(self._discover_blocking)
        except Exception as exc:  # noqa: BLE001 - a picker must still render
            log.warning("could not list Bedrock models", exc_info=True)
            return [], _bedrock_refusal(exc)

    def _discover_blocking(self) -> tuple[list[ModelInfo], str]:
        import boto3

        settings = self._settings
        region = (
            getattr(settings, "bedrock_region", "")
            or settings.aws_region
            or "us-east-1"
        )
        session = (
            boto3.Session(profile_name=settings.aws_profile)
            if settings.aws_profile
            else boto3.Session()
        )
        client = session.client("bedrock", region_name=region)
        profiles = _inference_profiles(client)

        listed = client.list_foundation_models(byOutputModality="TEXT")
        found: list[ModelInfo] = []
        for summary in listed.get("modelSummaries") or []:
            info = _bedrock_info(summary, profiles)
            if info is not None:
                found.append(info)
        found.sort(key=lambda info: info.name.casefold())
        return found, ""

    # -- OpenRouter ---------------------------------------------------------

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
        # The switch is checked here as well as in `read`, because this is the
        # path `prices_for` takes -- and a price lookup that fetched the
        # catalogue would be a call to openrouter.ai from a deployment that
        # had switched OpenRouter off.
        if not self._settings.openrouter_available:
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


def _inference_profiles(client: Any) -> dict[str, str]:
    """Base model id -> the profile id that can invoke it.

    Read because "what can I run" is not a property of the model alone: a
    model whose only inference type is INFERENCE_PROFILE is invoked by the
    profile, and offering its bare id would put an entry in the picker that
    fails the moment somebody chooses it.

    A failure here is swallowed. The profiles improve the list; they are not
    the list, and a deployment without permission to read them should still see
    the on-demand models.
    """
    profiles: dict[str, str] = {}
    try:
        token = None
        while True:
            page = (
                client.list_inference_profiles(maxResults=100, nextToken=token)
                if token
                else client.list_inference_profiles(maxResults=100)
            )
            for profile in page.get("inferenceProfileSummaries") or []:
                if (profile.get("status") or "ACTIVE") != "ACTIVE":
                    continue
                profile_id = profile.get("inferenceProfileId") or ""
                for model in profile.get("models") or []:
                    base = str(model.get("modelArn") or "").split("/")[-1]
                    if base and base not in profiles:
                        profiles[base] = profile_id
            token = page.get("nextToken")
            if not token:
                break
    except Exception:  # noqa: BLE001 - see the docstring
        log.info("could not list Bedrock inference profiles", exc_info=True)
    return profiles


def _bedrock_info(summary: dict[str, Any], profiles: dict[str, str]) -> "ModelInfo | None":
    """One listed model as an entry in the picker, or nothing.

    Nothing when it cannot be invoked on demand at all -- a model available
    only through a provisioned throughput commitment is not something a person
    should be offered in a dropdown, because choosing it fails with a billing
    error nobody reading the dropdown could have predicted.
    """
    model_id = str(summary.get("modelId") or "")
    if not model_id:
        return None
    kinds = set(summary.get("inferenceTypesSupported") or [])
    profile_id = profiles.get(model_id, "")
    if "ON_DEMAND" in kinds:
        offered = model_id
    elif "INFERENCE_PROFILE" in kinds and profile_id:
        offered = profile_id
    else:
        return None

    status = str((summary.get("modelLifecycle") or {}).get("status") or "")
    vendor = str(summary.get("providerName") or "").strip()
    name = str(summary.get("modelName") or "").strip() or model_id
    label = f"{vendor} {name}".strip()
    if status == "LEGACY":
        label = f"{label} (legacy)"
    return ModelInfo(
        provider="bedrock",
        id=offered,
        name=label,
        # Bedrock's API quotes no prices. Only the hand-maintained table can
        # answer, and for most of eighty-nine models it cannot -- which is the
        # honest answer rather than a default dressed as a quote.
        prices=known_rates_for(offered),
    )


def _bedrock_refusal(exc: Exception) -> str:
    """Why the list could not be read, in terms an operator can act on."""
    text = str(exc)
    name = type(exc).__name__
    if "AccessDenied" in text or "not authorized" in text:
        return (
            "This account cannot list Bedrock models: it needs "
            "bedrock:ListFoundationModels (and bedrock:ListInferenceProfiles for "
            "the cross-region ones). The models in BEDROCK_MODELS are still "
            "offered, and invoking them needs no extra permission. Set "
            "BEDROCK_DISCOVER=false to stop asking."
        )
    if "NoCredentialsError" in name or "Unable to locate credentials" in text:
        return (
            "No AWS credentials were found, so Bedrock could not be asked what "
            "it carries. The models in BEDROCK_MODELS are still listed."
        )
    return f"Bedrock did not answer: {text[:200]}"


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
