"""The single event schema shared by the agent loop, the database, the
WebSocket stream and the frontend.

``frontend/src/lib/events.ts`` is the hand-maintained TypeScript mirror of this
file. ``tests/test_events.py`` asserts the two stay in sync by checking that
every ``type`` literal here appears there.

Contract
--------
* Every event carries ``run_id``, a monotonically increasing ``seq`` (unique
  per run) and an ISO-8601 UTC ``ts``.
* ``seq`` is the resume token: a reconnecting client sends the last ``seq`` it
  saw and the server replays everything after it.
* Events are append-only with one exception: ``thinking`` events are upserted
  on ``(run_id, seq)`` while the model streams, so a thinking block occupies
  exactly one ``seq`` no matter how many deltas arrive.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

RunStatus = Literal["pending", "running", "awaiting_approval", "succeeded", "failed", "cancelled"]

TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseEvent(BaseModel):
    run_id: str
    seq: int
    ts: str = Field(default_factory=utc_now)


class RunStarted(BaseEvent):
    type: Literal["run_started"] = "run_started"
    task: str
    start_url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    #: Tool names discovered from the MCP server for this run.
    tools: list[str] = Field(default_factory=list)


class Thinking(BaseEvent):
    """Assistant prose. Streamed: the same ``seq`` is re-sent as text grows."""

    type: Literal["thinking"] = "thinking"
    step: int
    text: str = ""
    done: bool = False


class ToolCall(BaseEvent):
    type: Literal["tool_call"] = "tool_call"
    step: int
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    sensitive: bool = False
    #: Populated when the call was gated and a human approved it.
    approved_by_human: bool = False


class ToolResult(BaseEvent):
    type: Literal["tool_result"] = "tool_result"
    step: int
    call_id: str
    name: str
    ok: bool
    duration_ms: int
    #: Text handed back to the model (already truncated to the context budget).
    text: str = ""
    truncated: bool = False
    attempts: int = 1


class Screenshot(BaseEvent):
    """Human-facing observation only -- screenshots are never added to the
    model's history, which observes the accessibility snapshot instead."""

    type: Literal["screenshot"] = "screenshot"
    step: int
    artifact_id: str
    url: str
    caption: str | None = None
    page_url: str | None = None


class ApprovalRequired(BaseEvent):
    type: Literal["approval_required"] = "approval_required"
    step: int
    approval_id: str
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str
    categories: list[str] = Field(default_factory=list)
    expires_at: str


class ApprovalResolved(BaseEvent):
    type: Literal["approval_resolved"] = "approval_resolved"
    step: int
    approval_id: str
    decision: Literal["approved", "rejected", "timeout"]
    note: str | None = None


class ErrorEvent(BaseEvent):
    type: Literal["error"] = "error"
    step: int | None = None
    kind: str
    message: str
    recoverable: bool = False
    #: Screenshot captured at the moment of failure, when one could be taken.
    artifact_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class RunFinished(BaseEvent):
    type: Literal["run_finished"] = "run_finished"
    status: RunStatus
    steps: int
    duration_ms: int
    summary: str | None = None
    #: ``{"answer": str, "data": <parsed JSON or None>}``
    result: dict[str, Any] | None = None
    error: str | None = None


AgentEvent = Annotated[
    Union[
        RunStarted,
        Thinking,
        ToolCall,
        ToolResult,
        Screenshot,
        ApprovalRequired,
        ApprovalResolved,
        ErrorEvent,
        RunFinished,
    ],
    Field(discriminator="type"),
]

EVENT_ADAPTER: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)

EVENT_TYPES: tuple[str, ...] = (
    "run_started",
    "thinking",
    "tool_call",
    "tool_result",
    "screenshot",
    "approval_required",
    "approval_resolved",
    "error",
    "run_finished",
)


def parse_event(data: dict[str, Any]) -> AgentEvent:
    """Validate an untyped dict (e.g. a row read back out of SQLite)."""
    return EVENT_ADAPTER.validate_python(data)


def dump_event(event: AgentEvent) -> dict[str, Any]:
    return EVENT_ADAPTER.dump_python(event, mode="json")
