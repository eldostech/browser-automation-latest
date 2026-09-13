"""The Bedrock chat model, and the bridge from this codebase's message shape.

Why LangChain here
------------------
Provider wiring is the part of an LLM integration that rots: model IDs move,
auth mechanisms are added, request parameters change shape. ``langchain-aws``
tracks Bedrock so this project does not have to, and it already understands the
credential paths that matter here -- SigV4, a named profile, and the
``AWS_BEARER_TOKEN_BEDROCK`` API key this deployment actually uses.

What is deliberately NOT delegated
----------------------------------
Everything that is domain logic rather than plumbing stays: the domain
allowlist, secret redaction, loop detection, retry policy, and the event stream
the dashboard renders. A framework has no opinion about those, and pretending
otherwise would scatter them.

Two surfaces
------------
:func:`chat_model` returns the LangChain model itself, for the graph, which
speaks LangChain messages natively.

:func:`as_client` wraps one in the :class:`llm.LLMClient` protocol this
codebase already uses for its *one-shot* calls -- distillation, healing,
repair. Those are single request/response pairs with no loop to orchestrate,
so a graph would add ceremony without removing any.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

log = logging.getLogger(__name__)


def chat_model(
    settings: Any,
    model: str | None = None,
    provider: str | None = None,
    **overrides: Any,
):
    """Build a chat model for ``provider``.

    ``model`` overrides the configured one so one process can run several --
    which is now the ordinary case rather than a leftover: a person comparing
    models has two open at once by definition.
    """
    chosen = provider or settings.llm_provider
    if chosen == "openrouter":
        return _openrouter_model(settings, model, **overrides)
    return _bedrock_model(settings, model, **overrides)


def _bedrock_model(settings: Any, model: str | None, **overrides: Any):
    """Claude on Bedrock.

    Credentials are left entirely to the AWS chain: environment variables,
    ``~/.aws``, an attached IAM role, or ``AWS_BEARER_TOKEN_BEDROCK``. That is
    what lets the same build run on a laptop and under a role unchanged.
    """
    from langchain_aws import ChatBedrockConverse

    kwargs: dict[str, Any] = {
        "model": model or settings.llm_repair_model,
        "max_tokens": settings.llm_max_tokens,
        "temperature": settings.llm_temperature,
        **overrides,
    }
    _apply_thinking(kwargs, settings)
    if settings.aws_region:
        kwargs["region_name"] = settings.aws_region
    if settings.aws_profile:
        kwargs["credentials_profile_name"] = settings.aws_profile
    return ChatBedrockConverse(**kwargs)


def _openrouter_model(settings: Any, model: str | None, **overrides: Any):
    """Anything OpenRouter fronts, over its OpenAI-compatible endpoint.

    ``ChatOpenAI`` rather than a client written here: OpenRouter speaks the
    OpenAI wire format deliberately, and the part of a provider integration
    that rots is the wire format.

    Two things are deliberately *not* carried over from the Bedrock path.
    Extended thinking is an Anthropic-shaped request parameter and means
    nothing to most of what OpenRouter fronts -- a model that reasons does it
    on its own terms. And the prompt cache is Bedrock's; see
    ``LangChainLLM.run_turn`` for why the flag cannot simply be sent anyway.
    """
    if not settings.openrouter_api_key:
        raise ValueError(
            "OpenRouter was selected and OPENROUTER_API_KEY is not set, so there is "
            "nothing to authenticate with. Add it to the environment (or .env) and "
            "restart the backend."
        )
    resolved = model or settings.openrouter_model
    if not resolved:
        raise ValueError(
            "OpenRouter was selected with no model named, and there is no default: "
            "OPENROUTER_MODEL is blank. Pick a model in the dashboard, or set one."
        )

    from langchain_openai import ChatOpenAI

    # OpenRouter reads these for its activity page, which is how an operator
    # tells this application's spend apart from everything else on one key.
    headers = {"X-Title": settings.openrouter_app_name}
    if settings.openrouter_app_url:
        headers["HTTP-Referer"] = settings.openrouter_app_url

    kwargs: dict[str, Any] = {
        "model": resolved,
        "api_key": settings.openrouter_api_key,
        "base_url": settings.openrouter_base_url,
        "max_tokens": settings.llm_max_tokens,
        "temperature": settings.llm_temperature,
        "default_headers": headers,
        **overrides,
    }
    return ChatOpenAI(**kwargs)


def _apply_thinking(kwargs: dict[str, Any], settings: Any) -> None:
    """Turn on extended thinking, if configured -- in place, on ``kwargs``.

    Two things confirmed against the real model rather than assumed from
    documentation: Bedrock rejects the request outright ("`temperature` may
    only be set to 1 when thinking is enabled") if a non-default temperature
    rides along, so it is dropped here rather than left to fail at call time;
    and a reply's thinking block does *not* need to be replayed on the next
    turn for the conversation to continue -- dropping it (which
    :func:`to_anthropic_blocks` already does, since it only keeps ``text``
    and ``tool_use``) is safe, not a shortcut taken under time pressure.
    """
    budget = settings.llm_thinking_budget_tokens
    if not budget:
        return

    max_tokens = kwargs.get("max_tokens") or 0
    # Thinking tokens draw from the same ceiling as the response. Left alone,
    # a misconfigured budget at or above max_tokens does not fail loudly --
    # it silently starves the actual tool call of room to exist.
    if max_tokens and budget >= max_tokens:
        clamped = max(1024, max_tokens // 2)
        log.warning(
            "llm_thinking_budget_tokens (%d) leaves no room under "
            "llm_max_tokens (%d); using %d instead",
            budget, max_tokens, clamped,
        )
        budget = clamped

    fields = dict(kwargs.get("additional_model_request_fields") or {})
    fields.setdefault("thinking", {"type": "enabled", "budget_tokens": budget})
    kwargs["additional_model_request_fields"] = fields
    kwargs.pop("temperature", None)


# ---------------------------------------------------------------------------
# Message bridging
# ---------------------------------------------------------------------------


def to_langchain(system: str, messages: Iterable[dict[str, Any]]) -> list[BaseMessage]:
    """Convert this codebase's Anthropic-shaped history into LangChain messages.

    The shapes that actually occur are narrow -- a text user turn, an assistant
    turn of content blocks, and a user turn of tool results -- so this handles
    those precisely rather than trying to be a general translator.
    """
    converted: list[BaseMessage] = [SystemMessage(content=system)] if system else []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "assistant":
            converted.append(_assistant(content))
            continue

        # A user turn is either prose or a batch of tool results. Tool results
        # become one ToolMessage each, which is how LangChain pairs them back
        # to the call that produced them.
        if isinstance(content, list):
            results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if results:
                converted.extend(
                    ToolMessage(
                        content=str(block.get("content") or ""),
                        tool_call_id=str(block.get("tool_use_id") or ""),
                        status="error" if block.get("is_error") else "success",
                    )
                    for block in results
                )
                continue
            converted.append(HumanMessage(content=_text_of(content)))
            continue

        converted.append(HumanMessage(content=str(content or "")))

    return converted


def _assistant(content: Any) -> AIMessage:
    """Rebuild an assistant turn from stored Anthropic content blocks."""
    if isinstance(content, str):
        return AIMessage(content=content)

    blocks = content if isinstance(content, list) else []
    tool_calls = [
        {
            "name": str(block.get("name") or ""),
            "args": dict(block.get("input") or {}),
            "id": str(block.get("id") or ""),
        }
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    return AIMessage(content=_text_of(blocks), tool_calls=tool_calls)


def _text_of(content: Any) -> str:
    """The human-readable text of a content value, whatever shape it arrives in.

    LangChain returns block lists for models that interleave text and tool use,
    so ``.content`` is not reliably a string.

    A ``reasoning_content`` block -- confirmed by calling the real model with
    extended thinking on -- carries the model's deliberation *before* whatever
    it says out loud, in the same content list a plain ``text`` block would
    be. Included here, ahead of any ``text``, because dropping it is exactly
    how a session that reasoned at length still looked silent: nothing else
    in this codebase reads any other block type out of an assistant turn.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            _reasoning_text(block) if block.get("type") == "reasoning_content" else block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in ("text", "reasoning_content")
        ]
        return "".join(part for part in parts if part)
    return str(content or "")


def _reasoning_text(block: dict[str, Any]) -> str:
    return str((block.get("reasoning_content") or {}).get("text") or "")


def text_of(message: Any) -> str:
    """Public alias: the text of a LangChain message or raw content value."""
    return _text_of(getattr(message, "content", message))


def tool_calls_of(message: Any) -> list[dict[str, Any]]:
    """Normalised tool calls from a LangChain AI message."""
    return [
        {"id": call.get("id") or "", "name": call.get("name") or "", "input": call.get("args") or {}}
        for call in (getattr(message, "tool_calls", None) or [])
    ]


def usage_of(message: Any) -> dict[str, int]:
    """Token usage in this codebase's shape, whatever the provider reported.

    ``input_tokens`` already includes any cache read and cache write --
    ``langchain_aws`` sums them in, because a cached token is still a token
    Bedrock had to be told about, and the budget cares about that total. The
    two are broken out as well because they are not priced the same: a cache
    read costs a tenth of a fresh input token, and pricing that ignores the
    difference overstates the cost of exactly the thing caching exists to cut.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(details.get("cache_read", 0) or 0),
        "cache_creation_tokens": int(details.get("cache_creation", 0) or 0),
    }


def to_anthropic_blocks(message: AIMessage) -> list[dict[str, Any]]:
    """An assistant turn as content blocks, for storing in history.

    Kept because the stored shape is what the rest of this codebase and its
    tests already read, and because it round-trips through :func:`to_langchain`.
    """
    blocks: list[dict[str, Any]] = []
    text = _text_of(message.content)
    if text:
        blocks.append({"type": "text", "text": text})
    for call in getattr(message, "tool_calls", None) or []:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": call.get("name") or "",
                "input": call.get("args") or {},
            }
        )
    return blocks
