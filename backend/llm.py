"""The LLM seam: one protocol, a LangChain-backed Bedrock implementation.

Callers depend on :class:`LLMClient`, not on a provider, so the tests can
substitute a scripted implementation and never touch the network. That seam is
the reason the provider underneath could be swapped for LangChain without
touching distillation, healing or repair.

Claude on Amazon Bedrock via ``langchain-aws``, and only that. **No API key
required:** credentials resolve through the standard AWS chain -- environment,
``~/.aws``, an attached role, or ``AWS_BEARER_TOKEN_BEDROCK`` -- so the same
build runs on a laptop and under an IAM role unchanged. A second provider is a
second code path to keep working, and nothing here needs one.

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

#: The only provider. Named rather than configured, so /healthz and the client
#: still report it without a setting that has one legal value.
PROVIDER = "bedrock"


class RepairModel:
    """The one model this application still uses, built once and cached.

    There used to be three roles -- a driver for the agent loop, a distiller
    for turning a recording into a use case, and this one. The first two went
    with the agent: a workflow is recorded by watching someone do it now, and a
    codegen script is parsed rather than interpreted.

    What is left is the model that looks at a page when a step breaks. It is
    held here rather than on a manager so that the thing which owns *runs* is
    not also the thing which owns a model client -- that coupling is how the
    replay path ended up one attribute away from an LLM.
    """

    def __init__(self, settings, client: "LLMClient | None" = None) -> None:
        self._settings = settings
        #: A test injects a scripted client here.
        self._client = client

    @property
    def client(self) -> "LLMClient":
        if self._client is None:
            self._client = build_llm(self._settings, self._settings.llm_repair_model)
        return self._client

    def __call__(self) -> "LLMClient":
        """So it can be passed anywhere a zero-argument factory is wanted."""
        return self.client


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

    def __init__(self, model: Any, model_name: str) -> None:
        self._model = model
        self.model = model_name
        self.provider = PROVIDER

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

        try:
            final: Any = None
            async for chunk in model.astream(history):
                if on_text_delta is not None:
                    piece = text_of(chunk)
                    if piece:
                        await on_text_delta(piece)
                final = chunk if final is None else final + chunk
        except Exception as exc:  # noqa: BLE001 - narrowed by _translate
            raise self._translate(exc) from exc

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
        """Turn a provider access failure into something an operator can fix.

        Bedrock's own wording -- "anthropic.claude-sonnet-5 is not available
        for this account" -- names a model ID the operator never typed (the
        inference profile's region prefix is stripped), and says nothing about
        which of three configured models it was or where to change it.
        """
        status = getattr(exc, "status_code", None) or _status_from_message(str(exc))
        if status not in _ACCESS_STATUSES:
            return exc

        if status == 404:
            reason = f"the provider has no model {self.model!r}"
        elif status == 401:
            reason = "the credentials were rejected"
        else:
            reason = f"this account cannot use {self.model!r}"

        return LLMAccessError(
            f"{reason} on {self.provider}. Nothing will run until the model or the "
            f"credentials change. Provider said: {str(exc)[:300]}"
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": PROVIDER, "model": self.model, "auth": bedrock_auth_status()}

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


def _status_from_message(message: str) -> int | None:
    """Recover an HTTP status a provider only reported in prose.

    botocore raises ``AccessDeniedException`` / ``ValidationException`` rather
    than anything carrying a status code, so the access check would miss the
    very failure it exists to catch.
    """
    lowered = message.lower()
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


def build_llm(settings: Any, model: str | None = None) -> LLMClient:
    """The configured model, wrapped in this codebase's client protocol.

    ``model`` overrides the configured one, which is what let a single process
    run several at once when there were several roles to run.
    """
    from chat import chat_model

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

    return LangChainLLM(chat_model(settings, resolved), resolved)
