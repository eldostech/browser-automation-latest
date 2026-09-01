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


def chat_model(settings: Any, model: str | None = None, **overrides: Any):
    """Build the configured Bedrock chat model.

    ``model`` overrides the driver model so one process can run several -- a
    fast driver for the agent loop, a more capable one for repair.

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
    if settings.aws_region:
        kwargs["region_name"] = settings.aws_region
    if settings.aws_profile:
        kwargs["credentials_profile_name"] = settings.aws_profile
    return ChatBedrockConverse(**kwargs)


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
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)
    return str(content or "")


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
    """Token usage in this codebase's shape, whatever the provider reported."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
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
