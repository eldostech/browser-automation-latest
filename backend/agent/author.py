"""The authoring session's request/wiring shapes, and its system prompt.

`create_agent` (`graph.py`) and its middleware (`middleware.py`) provide the
model-calling and tool-dispatch loop now; a hand-written `StateGraph` with
its own `decide`/`act`/`ask`/`resolve`/`finish` nodes used to live here and
was deleted once the rewrite made it entirely dead. What is left is what
every part of the new loop still needs: the request shape a caller builds,
the live objects (`Wiring`) closed over by the middleware rather than
checkpointed, and the system prompt.

Nothing in this module imports LangGraph, FastAPI or ``store``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from llm import LLMClient
from prompt_loader import AUTHOR, render

from .budget import Budget, Spend
from .session import AgentToolSession


@dataclass
class AuthorRequest:
    """Everything a session needs to start. No FastAPI types, deliberately."""

    task: str
    start_url: str
    allowed_domains: tuple[str, ...] = ()
    #: Credential slot names. Never values: the model is shown the slot and the
    #: tool layer substitutes, which is how replay already works.
    secrets: tuple[str, ...] = ()
    #: One record to work through, if the caller has one.
    sample: dict[str, Any] = field(default_factory=dict)
    may_write: bool = False
    budget: Budget = field(default_factory=Budget)
    run_id: str = ""
    workspace_id: str = ""


@dataclass
class Wiring:
    """The live objects, which a checkpoint must never hold.

    `create_agent`'s state is a plain `messages` list plus the few outcome
    fields `middleware.py`'s `AuthorGraphState` adds -- everything that
    cannot be written to Postgres (the open tool session, the model client)
    is closed over by the middleware instances instead, built fresh from one
    of these per session.
    """

    request: AuthorRequest
    tools: AgentToolSession
    llm: LLMClient
    spend: Spend
    emit: "Emit"


#: Where events go. The same sink a replay writes to -- see ``runner.py`` --
#: so an agent run appears in the existing run view, on the existing stream,
#: with the same sequence numbers and the same resume token. A second streaming
#: path would be a second thing to keep working.
Emit = Callable[[Any], Awaitable[None]]


def system_prompt(request: AuthorRequest) -> str:
    return render(
        AUTHOR,
        allowed_domains=", ".join(request.allowed_domains) or "(nothing configured)",
        secrets=", ".join(request.secrets) or "(none bound to this session)",
    )


#: How many page snapshots stay in the model's context in full. Everything
#: older is replaced by one line. `middleware.py` carries its own copy of this
#: idea for `create_agent`'s `BaseMessage` history; `for_model` below is the
#: same rule for `agent/operate.py`'s recover/explore loop, which still talks
#: to a model through the older `LLMClient`/provider-native-dict protocol and
#: is out of scope for the `create_agent` migration.
SNAPSHOTS_KEPT = 2


def for_model(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation, with stale page snapshots taken out.

    Every tool result here is a full accessibility tree -- thousands of tokens
    -- and it stayed in the history forever, so turn N carried N snapshots and
    the cost of a session grew with the square of its length. A real session
    measured at ~7,700 tokens per call and ran out of budget *after* doing the
    task correctly but *before* calling `end_row`, which lost the recording.

    Dropping them costs nothing, and not for a subtle reason: a ref from an
    older snapshot is stale, and the guard refuses it. Keeping those pages in
    context was paying to send the model information it is forbidden to act
    on -- and, worse, tempting it to try.

    The replacement says *why* it is gone, so a model reading back does not
    conclude the page went blank.
    """
    kept = 0
    out: list[dict[str, Any]] = []
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, list) and content and _is_page(content[0]):
            kept += 1
            if kept > SNAPSHOTS_KEPT:
                out.append(
                    {
                        **message,
                        "content": [
                            {
                                **content[0],
                                "content": (
                                    "(page omitted: this is no longer the current "
                                    "page and its refs are stale. Take a snapshot "
                                    "if you need to look again.)"
                                ),
                            }
                        ],
                    }
                )
                continue
        out.append(message)
    out.reverse()
    return out


def _is_page(block: dict[str, Any]) -> bool:
    return block.get("type") == "tool_result" and "### Page" in str(
        block.get("content") or ""
    )


__all__ = ["AuthorRequest", "Emit", "Wiring", "for_model", "system_prompt"]
