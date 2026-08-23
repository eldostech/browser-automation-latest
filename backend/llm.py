"""LLM provider adapter.

The agent loop depends on the :class:`LLMClient` protocol, not on a specific
provider, so adding one means writing a class with a ``run_turn`` method that
returns an :class:`LLMTurn`. The tests substitute a scripted implementation and
never touch the network.

Two providers ship here, both speaking the Anthropic Messages API:

``bedrock`` (default)
    Claude on Amazon Bedrock. **No API key.** Credentials come from the
    standard AWS chain, so the same code works from a developer laptop
    (``~/.aws/credentials`` or SSO) and from an EC2/ECS/EKS/Lambda role with no
    configuration change.

``anthropic``
    The first-party Anthropic API, authenticated with ``ANTHROPIC_API_KEY``.

Streaming matters here for UX, not for tokens: the dashboard shows the model's
prose as it is produced, so a 6-second turn does not look like a hang.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

log = logging.getLogger(__name__)

TextDeltaHandler = Callable[[str], Awaitable[None]]

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
# Shared Messages API implementation
# ---------------------------------------------------------------------------


class _MessagesAPILLM:
    """Streaming turn logic shared by every Anthropic Messages API client.

    Subclasses differ only in how they build and authenticate the client, so
    the request/response handling lives here exactly once.
    """

    provider = "unknown"

    def __init__(self, client: Any, model: str, max_tokens: int, temperature: float) -> None:
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def run_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text_delta: TextDeltaHandler | None = None,
        timeout: float | None = None,
    ) -> LLMTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if timeout is not None:
            kwargs["timeout"] = timeout

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if (
                    on_text_delta is not None
                    and event.type == "content_block_delta"
                    and getattr(event.delta, "type", None) == "text_delta"
                ):
                    await on_text_delta(event.delta.text)
            final = await stream.get_final_message()

        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        raw_content: list[dict[str, Any]] = []

        for block in final.content:
            raw_content.append(block.model_dump(exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input) if isinstance(block.input, dict) else {},
                    )
                )

        usage = {
            "input_tokens": getattr(final.usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(final.usage, "output_tokens", 0) or 0,
        }
        log.debug(
            "llm turn complete",
            extra={"stop_reason": final.stop_reason, "tool_calls": len(tool_calls), **usage},
        )

        return LLMTurn(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=final.stop_reason or "end_turn",
            raw_content=raw_content,
            usage=usage,
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model}


# ---------------------------------------------------------------------------
# Amazon Bedrock
# ---------------------------------------------------------------------------


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
            "error": "botocore is not installed -- run: pip install 'anthropic[bedrock]'",
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


class BedrockLLM(_MessagesAPILLM):
    """Claude on Amazon Bedrock, authenticated by the standard AWS chain.

    No API key is passed. Credentials resolve in botocore's usual order --
    environment variables, then ``~/.aws/credentials`` / SSO, then the
    instance/task/container IAM role -- so a laptop and a production role need
    identical configuration.

    ``api`` selects the endpoint:

    ``invoke`` (default)
        ``bedrock-runtime.{region}.amazonaws.com``. Model IDs are the
        Bedrock-native, version-suffixed form, and for current models that
        means a **cross-region inference profile** ID such as
        ``us.anthropic.claude-haiku-4-5-20251001-v1:0``.

    ``mantle``
        The newer Messages-API Bedrock endpoint, whose model IDs are the short
        ``anthropic.claude-haiku-4-5`` form.
    """

    provider = "bedrock"

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        region: str | None = None,
        profile: str | None = None,
        api: str = "invoke",
        max_retries: int = 3,
    ) -> None:
        try:
            from anthropic import AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "The Bedrock client requires the AWS extra: pip install 'anthropic[bedrock]'"
            ) from exc

        # The SDK refuses a request that carries both a bearer token and AWS
        # credential arguments. Catch it here with an actionable message
        # instead of letting the SDK raise a generic ValueError at startup.
        bearer_token_set = bool(os.environ.get(BEDROCK_BEARER_TOKEN_ENV))
        if bearer_token_set and profile:
            raise ValueError(
                f"Both {BEDROCK_BEARER_TOKEN_ENV} and AWS_PROFILE are set. The Anthropic "
                "SDK accepts one or the other, not both. Either unset "
                f"{BEDROCK_BEARER_TOKEN_ENV} to authenticate with the IAM profile, or "
                "clear AWS_PROFILE to authenticate with the Bedrock API key."
            )

        client_cls = AsyncAnthropicBedrockMantle if api == "mantle" else AsyncAnthropicBedrock

        # Only pass what was explicitly configured. Passing nothing lets the
        # SDK run its own resolution, which is what makes IAM roles work with
        # no configuration at all.
        kwargs: dict[str, Any] = {"max_retries": max_retries}
        if region:
            kwargs["aws_region"] = region
        if profile:
            kwargs["aws_profile"] = profile

        self.api = api
        self.region = region
        self.profile = profile
        self._auth = bedrock_auth_status(profile)

        super().__init__(client_cls(**kwargs), model, max_tokens, temperature)

        log.info(
            "bedrock client ready",
            extra={
                "model": model,
                "api": api,
                "region": region or self._auth.get("region") or "(from AWS chain)",
                "auth_method": self._auth.get("method"),
                "auth_source": self._auth.get("source"),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api": self.api,
            "region": self.region or self._auth.get("region"),
            "profile": self.profile,
            "auth": self._auth,
        }


# ---------------------------------------------------------------------------
# First-party Anthropic API
# ---------------------------------------------------------------------------


class AnthropicLLM(_MessagesAPILLM):
    """The first-party Anthropic API, authenticated with an API key."""

    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY. Set it in .env "
                "(backend only -- never in the frontend or a URL), or switch to "
                "LLM_PROVIDER=bedrock to use AWS credentials instead."
            )
        from anthropic import AsyncAnthropic

        super().__init__(
            AsyncAnthropic(api_key=api_key, max_retries=max_retries),
            model,
            max_tokens,
            temperature,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def llm_health(settings: Any) -> dict[str, Any]:
    """Report whether the configured provider could authenticate.

    Cheap and offline: for Bedrock this resolves the credential chain locally
    without calling AWS, so ``/healthz`` stays safe to poll.
    """
    provider = (settings.llm_provider or "bedrock").lower()
    base: dict[str, Any] = {
        "provider": provider,
        # `model` stays the driver, so existing consumers keep working.
        "model": settings.llm_model,
        # Which model does what -- three roles can differ.
        "models": getattr(settings, "models_in_use", {"driver": settings.llm_model}),
    }

    if provider == "bedrock":
        auth = bedrock_auth_status(settings.aws_profile)
        return {
            **base,
            "api": settings.bedrock_api,
            "region": settings.aws_region or auth.get("region"),
            "configured": bool(auth.get("ok")),
            "auth": auth,
        }

    if provider == "anthropic":
        return {**base, "configured": bool(settings.anthropic_api_key)}

    return {**base, "configured": False, "error": f"unknown provider {provider!r}"}


def build_llm(settings: Any, model: str | None = None) -> LLMClient:
    """Construct the configured provider. Add new providers here.

    ``model`` overrides ``settings.llm_model`` so one process can run several
    models at once -- a fast driver for the agent loop, a more capable one for
    the rare repair calls -- without a second Settings object.
    """
    provider = (settings.llm_provider or "bedrock").lower()
    model = model or settings.llm_model

    if provider == "bedrock":
        return BedrockLLM(
            model=model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            region=settings.aws_region,
            profile=settings.aws_profile,
            api=settings.bedrock_api,
        )

    if provider == "anthropic":
        return AnthropicLLM(
            api_key=settings.anthropic_api_key,
            model=model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER {provider!r}. Use 'bedrock' or 'anthropic'.")
