"""The LLM seam: one protocol, a LangChain-backed Bedrock implementation.

Callers depend on :class:`LLMClient`, not on a provider, so the tests can
substitute a scripted implementation and never touch the network. That seam is
the reason the provider underneath could be swapped for LangChain without
touching distillation, healing or repair.

Two providers, and the second one earned its place. **Bedrock** via
``langchain-aws`` needs no API key: credentials resolve through the standard AWS
chain -- environment, ``~/.aws``, an attached role, or
``AWS_BEARER_TOKEN_BEDROCK`` -- so the same build runs on a laptop and under an
IAM role unchanged. **OpenRouter** via ``langchain-openai`` needs one key and
fronts most of the models anybody would want to compare.

This file used to say "a second provider is a second code path to keep working,
and nothing here needs one". That was right about a deployment and wrong about
the reason a second one gets asked for: choosing a model *is* the work when you
are measuring accuracy, and you cannot choose between models you cannot reach.
OpenRouter is the cheap version of that -- one wire format for hundreds of
models rather than one integration each.

A model is therefore a :class:`ModelChoice`, not a string, and a process holds
as many clients as it is asked for (:class:`ModelPool`). Which one a run uses
can be decided per request, because comparing two models means having both.

Streaming matters here for UX, not for tokens: the dashboard shows the model's
prose as it is produced, so a 6-second turn does not look like a hang.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

log = logging.getLogger(__name__)

TextDeltaHandler = Callable[[str], Awaitable[None]]

#: The only provider. Named rather than configured, so /healthz and the client
#: still report it without a setting that has one legal value.
PROVIDER = "bedrock"


#: The providers this build can reach.
PROVIDERS: tuple[str, ...] = ("bedrock", "openrouter")

#: Said the same way everywhere it is said. An operator reading it needs the
#: setting's name, not a paraphrase of it.
OPENROUTER_OFF = (
    "OpenRouter is switched off in this deployment (OPENROUTER_ENABLED=false). "
    "Nothing here will call it. Pick a Bedrock model, or enable OpenRouter in "
    "the environment and restart the backend."
)


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """Which model, from which provider.

    Frozen and hashable so it can key the pool: two runs asking for the same
    model share one client, and a run asking for a different one does not wait
    for somebody else's.
    """

    provider: str
    model: str

    @classmethod
    def default(cls, settings: Any) -> "ModelChoice":
        provider = settings.llm_provider
        if provider == "openrouter":
            return cls("openrouter", settings.openrouter_model)
        return cls("bedrock", settings.llm_repair_model)

    @classmethod
    def resolve(
        cls, settings: Any, provider: str | None, model: str | None
    ) -> "ModelChoice":
        """What a request asked for, filled in from the defaults.

        Either half may be omitted. Naming a provider without a model is the
        useful case for Bedrock, where there is a configured default; on
        OpenRouter it fails with that as the reason, because a provider
        fronting hundreds of models has no sensible default to pick for you.
        """
        if not provider and not model:
            return cls.default(settings)
        chosen = provider or settings.llm_provider
        if chosen not in PROVIDERS:
            raise ValueError(
                f"{chosen!r} is not a provider this build can reach. "
                f"Available: {', '.join(PROVIDERS)}."
            )
        if chosen == "openrouter" and not getattr(settings, "openrouter_enabled", True):
            # The first of three refusals on this path, and each one is worth
            # having: this is where a request names a provider, `build_llm` is
            # where a client would be constructed, and `catalog` is where the
            # model list would be fetched. A deployment that must not reach a
            # third party should not have to reason about which of those a
            # given feature happens to go through.
            raise ValueError(OPENROUTER_OFF)
        if model:
            return cls(chosen, model)
        fallback = cls.default(settings)
        return cls(chosen, fallback.model if fallback.provider == chosen else "")

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


class ModelPool:
    """The clients this process has been asked for, built once each.

    This was ``RepairModel``, which held exactly one client because there was
    exactly one model. Now a person comparing two models has two in flight, so
    the cache is keyed by the choice rather than being a single slot -- and a
    client is still built lazily, because building one validates credentials
    and a process that never calls a model should never have to have them.
    """

    def __init__(self, settings, client: "LLMClient | None" = None) -> None:
        self._settings = settings
        #: A test injects a scripted client here, by name -- `_client` is an
        #: established seam (`# noqa: SLF001 - test seam` at its call sites)
        #: and renaming it would silently start making real provider calls in
        #: every test that uses it.
        #:
        #: It answers for *every* choice, deliberately, for the same reason: a
        #: pool that honoured the injection only for the default would make a
        #: real call the moment a test named another model.
        self._client = client
        self._clients: dict[ModelChoice, LLMClient] = {}

    @property
    def default_choice(self) -> ModelChoice:
        return ModelChoice.default(self._settings)

    @property
    def client(self) -> "LLMClient":
        """The configured default. What everything used before this existed."""
        return self.for_choice(self.default_choice)

    def for_choice(self, choice: "ModelChoice | None") -> "LLMClient":
        if self._client is not None:
            return self._client
        resolved = choice or self.default_choice
        if resolved not in self._clients:
            self._clients[resolved] = build_llm(
                self._settings, resolved.model, resolved.provider
            )
        return self._clients[resolved]

    def factory(self, choice: "ModelChoice | None" = None):
        """A zero-argument callable, for anything that wants one."""
        return lambda: self.for_choice(choice)

    def __call__(self) -> "LLMClient":
        """So it can be passed anywhere a zero-argument factory is wanted."""
        return self.client


#: The old name, kept so nothing that imports it breaks on the way through.
RepairModel = ModelPool


class LLMAccessError(RuntimeError):
    """The configured model cannot be used: no access, no such model, bad key.

    A configuration problem wearing a provider exception's clothes. Raised as
    its own type so a run fails with something an operator can act on, instead
    of the generic "agent run crashed" and a stack trace ending in a 403.
    """


#: Provider status codes that mean "your configuration is wrong", not "the
#: request was bad" -- retrying cannot help, and the fix is in .env.
_ACCESS_STATUSES = frozenset({401, 403, 404})

#: Environment variable holding a Bedrock API key (bearer token). When it is
#: set, the Anthropic SDK authenticates with it *instead of* SigV4 -- and
#: rejects the request outright if AWS credential arguments are also passed.
BEDROCK_BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"


@dataclass(slots=True)
class ToolCallRequest:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMTurn:
    """One assistant turn, normalised across providers."""

    text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = "end_turn"
    #: Provider-native assistant content blocks, appended verbatim to history.
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    model: str

    async def run_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text_delta: TextDeltaHandler | None = None,
        timeout: float | None = None,
    ) -> LLMTurn: ...

    def describe(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# The client: a thin adapter over a LangChain chat model
# ---------------------------------------------------------------------------


class LangChainLLM:
    """Implements :class:`LLMClient` on top of a LangChain chat model.

    Provider wiring -- endpoints, auth, request shapes, model IDs -- is the
    part that rots, and ``langchain-aws`` tracks it.
    What stays here is the small amount this codebase actually needs on top:
    a normalised :class:`LLMTurn`, streamed text for the dashboard, and access
    failures translated into something an operator can act on.
    """

    def __init__(self, model: Any, model_name: str, provider: str = PROVIDER) -> None:
        self._model = model
        self.model = model_name
        self.provider = provider

    @property
    def raw(self) -> Any:
        """The underlying LangChain chat model, unwrapped.

        `create_agent` wants an actual `BaseChatModel` rather than this
        codebase's own `run_turn` protocol -- this is the seam that lets the
        authoring graph pass one through without every other caller of
        `LLMClient` (healing, repair, the recover/explore loop) knowing or
        caring that it exists.
        """
        return self._model

    async def run_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text_delta: TextDeltaHandler | None = None,
        timeout: float | None = None,
    ) -> LLMTurn:
        from chat import (
            text_of,
            to_anthropic_blocks,
            to_langchain,
            tool_calls_of,
            usage_of,
        )

        model = self._model.bind_tools(tools) if tools else self._model
        history = to_langchain(system, messages)

        # Bedrock's prompt cache. The system prompt and the tool schemas are
        # identical on every turn of a session -- an agent loop resends its
        # whole history every time, because the Messages API is stateless --
        # and without this each of those turns is billed and reprocessed from
        # nothing. A real session measured 127,000 tokens across six turns
        # against a real page, almost all of it the ~20-tool schema list
        # repeated verbatim; caching turns everything after the first hit into
        # roughly a tenth of the price and skips reprocessing the cached
        # prefix, which is also most of what a slow turn is spending time on.
        # `ChatBedrockConverse` inserts the cache breakpoints; this only says
        # to use them.
        # ...and only Bedrock inserts them. `ChatOpenAI` rejects an unknown
        # keyword outright, so sending it to OpenRouter would fail every call
        # rather than merely miss a saving.
        options: dict[str, Any] = (
            {"cache_control": {"ttl": "5m"}} if self.provider == "bedrock" else {}
        )

        started = time.monotonic()
        try:
            final: Any = None
            async for chunk in model.astream(history, **options):
                if on_text_delta is not None:
                    piece = text_of(chunk)
                    if piece:
                        await on_text_delta(piece)
                final = chunk if final is None else final + chunk
        except Exception as exc:  # noqa: BLE001 - narrowed by _translate
            raise self._translate(exc) from exc
        finally:
            # The only place this call's wall time is recorded. Without it, a
            # slow turn is invisible until somebody reconstructs it from the
            # gap between two events -- which is how the 95-second stall
            # above was actually found.
            log.info(
                "model turn",
                extra={
                    "model": self.model,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "tool_count": len(tools),
                },
            )

        if final is None:
            return LLMTurn()

        return LLMTurn(
            text=text_of(final).strip(),
            tool_calls=[ToolCallRequest(**call) for call in tool_calls_of(final)],
            stop_reason="tool_use" if tool_calls_of(final) else "end_turn",
            raw_content=to_anthropic_blocks(final),
            usage=usage_of(final),
        )

    def _translate(self, exc: Exception) -> Exception:
        return translate_access_error(exc, model=self.model, provider=self.provider)

    def describe(self) -> dict[str, Any]:
        described: dict[str, Any] = {"provider": self.provider, "model": self.model}
        if self.provider == "bedrock":
            described["auth"] = bedrock_auth_status()
        return described

    async def check_access(self) -> dict[str, Any]:
        """Can this client actually call its model? One tiny request.

        Used by the deep health probe so an unusable model is found before a
        run rather than one step into one.
        """
        try:
            await self._model.ainvoke("hi")
        except Exception as exc:  # noqa: BLE001 - a probe must never raise
            translated = self._translate(exc)
            return {
                "model": self.model,
                "ok": False,
                "access_problem": isinstance(translated, LLMAccessError),
                "error": str(translated)[:300],
            }
        return {"model": self.model, "ok": True}


def translate_access_error(exc: Exception, *, model: str, provider: str = PROVIDER) -> Exception:
    """Turn a provider access failure into something an operator can fix.

    Bedrock's own wording -- "anthropic.claude-sonnet-5 is not available for
    this account" -- names a model ID the operator never typed (the inference
    profile's region prefix is stripped), and says nothing about which of
    three configured models it was or where to change it.

    A module-level function rather than only a method on `LangChainLLM`
    because the authoring graph's model-call middleware calls the raw chat
    model directly (`create_agent` needs a `BaseChatModel`, not this
    codebase's `run_turn` wrapper) and still needs the same translation.
    """
    status = getattr(exc, "status_code", None) or _status_from_message(str(exc))
    if status not in _ACCESS_STATUSES:
        return exc

    if status == 404:
        reason = f"the provider has no model {model!r}"
    elif status == 401:
        reason = "the credentials were rejected"
    else:
        reason = f"this account cannot use {model!r}"

    return LLMAccessError(
        f"{reason} on {provider}. Nothing will run until the model or the "
        f"credentials change. Provider said: {str(exc)[:300]}"
    )


def _status_from_message(message: str) -> int | None:
    """Recover an HTTP status a provider only reported in prose.

    botocore raises ``AccessDeniedException`` / ``ValidationException`` rather
    than anything carrying a status code, so the access check would miss the
    very failure it exists to catch.

    The OpenAI SDK -- which is how OpenRouter is reached -- puts the status in
    the message as ``Error code: 401``, and wraps it in exception types that do
    not always survive LangChain's own re-raising. Reading it out of the text
    is what makes "your key was rejected" an actionable message rather than a
    stack trace ending in a 401.
    """
    lowered = message.lower()

    coded = re.search(r"error code:\s*(\d{3})", lowered)
    if coded:
        return int(coded.group(1))
    if "no auth credentials" in lowered or "invalid api key" in lowered:
        return 401
    if "accessdenied" in lowered or "not available for this account" in lowered:
        return 403
    if "unrecognizedclient" in lowered or "invalid" in lowered and "token" in lowered:
        return 401
    if "resourcenotfound" in lowered or "could not be found" in lowered:
        return 404
    return None



def bedrock_auth_status(profile: str | None = None) -> dict[str, Any]:
    """Resolve how Bedrock will authenticate, without making a network call.

    Used by ``/healthz`` so an operator can tell a credential problem from a
    model problem before running anything. Never raises.
    """
    try:
        from botocore.session import Session
    except ImportError:
        return {
            "ok": False,
            "error": "botocore is not installed -- run: pip install botocore",
        }

    def resolve_region() -> str | None:
        try:
            session = Session(profile=profile) if profile else Session()
            return session.get_config_variable("region")
        except Exception:  # noqa: BLE001
            return None

    # A bearer token short-circuits SigV4 entirely -- the SDK uses it instead
    # of AWS credentials, so report that plainly rather than the IAM identity
    # the operator may believe is in play.
    if os.environ.get(BEDROCK_BEARER_TOKEN_ENV):
        return {
            "ok": True,
            "method": "bearer_token",
            "source": BEDROCK_BEARER_TOKEN_ENV,
            "region": resolve_region(),
            "note": (
                f"{BEDROCK_BEARER_TOKEN_ENV} takes precedence over IAM credentials. "
                "Unset it to authenticate with a role or ~/.aws instead."
            ),
        }

    try:
        session = Session(profile=profile) if profile else Session()
        credentials = session.get_credentials()
        if credentials is None:
            return {
                "ok": False,
                "error": (
                    "No AWS credentials found. Configure ~/.aws/credentials, run "
                    "'aws sso login', or attach an IAM role to this host."
                ),
            }
        return {
            "ok": True,
            "method": "sigv4",
            # botocore's label for where the credentials came from, e.g.
            # 'shared-credentials-file', 'env', 'iam-role', 'sso'.
            "source": credentials.method,
            "region": session.get_config_variable("region"),
            "profile": profile,
        }
    except Exception as exc:  # noqa: BLE001 - a health probe must never raise
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def llm_health(settings: Any) -> dict[str, Any]:
    """Report whether Bedrock could authenticate.

    Cheap and offline: this resolves the credential chain locally without
    calling AWS, so ``/healthz`` stays safe to poll.
    """
    auth = bedrock_auth_status(settings.aws_profile)
    return {
        "provider": PROVIDER,
        # One role left. The driver and the distiller went with the agent.
        "model": settings.llm_repair_model,
        "region": settings.aws_region or auth.get("region"),
        "configured": bool(auth.get("ok")),
        "auth": auth,
    }


def build_llm(
    settings: Any, model: str | None = None, provider: str | None = None
) -> LLMClient:
    """One model, wrapped in this codebase's client protocol."""
    from chat import chat_model

    chosen = provider or settings.llm_provider
    if chosen == "openrouter":
        if not getattr(settings, "openrouter_enabled", True):
            raise ValueError(OPENROUTER_OFF)
        resolved = model or settings.openrouter_model
        return LangChainLLM(
            chat_model(settings, resolved, "openrouter"), resolved, "openrouter"
        )

    resolved = model or settings.llm_repair_model

    # The bearer token short-circuits SigV4 entirely, and boto3 rejects a
    # request carrying both. Catch it here with an actionable message rather
    # than letting a generic credentials error surface mid-run.
    if os.environ.get(BEDROCK_BEARER_TOKEN_ENV) and settings.aws_profile:
        raise ValueError(
            f"Both {BEDROCK_BEARER_TOKEN_ENV} and AWS_PROFILE are set. Bedrock accepts one "
            f"or the other, not both. Either unset {BEDROCK_BEARER_TOKEN_ENV} to authenticate "
            "with the IAM profile, or clear AWS_PROFILE to use the Bedrock API key."
        )

    return LangChainLLM(chat_model(settings, resolved, "bedrock"), resolved, "bedrock")
