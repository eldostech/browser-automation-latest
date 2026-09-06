"""The authoring session: what each node of the graph actually does.

The graph is LangGraph and lives in ``graph.py``. Everything it calls is here,
as plain async functions over a plain state object, and that split is
deliberate rather than tidy. A LangGraph node *should* be an ordinary function
-- but more usefully, it means the behaviour of the loop can be tested with no
graph library, no browser and no model, while the wiring, the checkpointing and
the interrupt are tested separately with all three.

The shape, and where the money goes:

    load_context   no model    what site, whose credentials, which record
    plan           MODEL x1    an outline, to notice deviation against
    perceive       no model    browser_snapshot
    decide         MODEL x1    one tool call
    act            no model    guard, dispatch, record   (see session.py)
    finish         no model    stop, and say why

Three of six touch a model, and only ``decide`` runs more than once. The
guard -- allowlist, ref discipline, write gate, irreversibility -- is not a node
here because it is inside ``AgentToolSession.call``, where it cannot be
forgotten by a future node that dispatches a tool some other way.

Nothing in this module imports LangGraph, FastAPI or ``store``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypedDict

from events import (
    ApprovalRequired,
    ApprovalResolved,
    ErrorEvent,
    RunFinished,
    RunStarted,
    Thinking,
    ToolCall,
    ToolResult as ToolResultEvent,
)
from llm import LLMClient
from prompt_loader import AUTHOR, AUTHOR_TASK, render

from .budget import Budget, BudgetExhausted, Spend
from .session import AgentToolSession

log = logging.getLogger(__name__)

#: The tool that ends a session. Ours, like the marks, and it is a tool rather
#: than a stop-word so that ending is a deliberate act with a reason attached.
FINISH = "finish"

FINISH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "What you did, and anything a reviewer should check.",
        },
        "complete": {
            "type": "boolean",
            "description": "False if something stopped you before the task was done.",
        },
    },
    "required": ["summary"],
}

#: How much of a tool result the model is shown. A full accessibility snapshot
#: of a large page is thousands of tokens, and the tail of one is rarely what
#: decides the next action -- but the *head* carries the page URL and the top
#: of the tree, which usually does.
RESULT_BUDGET_CHARS = 6000


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


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


class AuthorState(TypedDict, total=False):
    """What the graph carries between nodes, and what a checkpoint holds.

    A TypedDict of plain values, and that is the whole design constraint: this
    is what LangGraph serialises after every node, so anything in here has to
    survive being written to Postgres and read back in another process.

    The live objects a node needs -- the open tool session, the model client --
    are therefore *not* here. They are closed over by the graph, exactly as
    `emit` is, for exactly the same reason. An earlier version of this carried
    them in the state and the checkpointer rejected the whole object, which is
    the checkpointer being right.
    """

    #: The model's own history. Provider-native blocks, appended verbatim.
    messages: list[dict[str, Any]]
    #: What `plan` produced. Not steps -- a map to notice deviation against.
    outline: str
    #: Set once the session is over, and why.
    done: bool
    status: str
    summary: str
    stopped_by: str
    #: The call waiting on a human, if any.
    pending: dict[str, Any] | None
    step: int
    #: Mirrored from the live Spend after each node, so a checkpoint records
    #: what has been spent and a resume in a fresh process does not hand the
    #: session a full budget again.
    spend: dict[str, Any]


def initial_state() -> AuthorState:
    return AuthorState(
        messages=[],
        outline="",
        done=False,
        status="running",
        summary="",
        stopped_by="",
        pending=None,
        step=0,
        spend={},
    )


@dataclass
class Wiring:
    """The live objects, which a checkpoint must never hold.

    Passed to every node beside the state. Kept in one object rather than
    three arguments so that adding one later is not a change to every
    signature in the graph.
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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _mirror(state: AuthorState, w: Wiring) -> AuthorState:
    """Copy what was spent into the state, so the checkpoint records it.

    The live Spend cannot be checkpointed and does not need to be: it is
    primitives, and writing them here means a session resumed in a fresh
    process is not handed a full budget again.
    """
    state["spend"] = w.spend.as_dict()
    return state


async def load_context(state: AuthorState, w: Wiring) -> AuthorState:
    """Announce the run and seed the conversation. No model.

    The system prompt names the allowlist because the guard enforces it either
    way and an agent that knows the rule wastes fewer turns discovering it.
    """
    request = w.request
    await w.emit(
        RunStarted(
            run_id=request.run_id,
            seq=0,
            task=request.task,
            start_url=request.start_url,
            options={
                "mode": "author",
                "may_write": request.may_write,
                "budget": request.budget.as_dict(),
            },
            tools=[spec.name for spec in w.tools.tools] + [FINISH],
        )
    )
    state["messages"] = [
        {
            "role": "user",
            "content": render(
                AUTHOR_TASK,
                task=request.task,
                start_url=request.start_url,
                secrets=", ".join(request.secrets) or "(none)",
                sample=json.dumps(request.sample) if request.sample else "(none given)",
            ),
        }
    ]
    return state


def system_prompt(request: AuthorRequest) -> str:
    return render(
        AUTHOR,
        allowed_domains=", ".join(request.allowed_domains) or "(nothing configured)",
        secrets=", ".join(request.secrets) or "(none bound to this session)",
    )


async def plan(state: AuthorState, w: Wiring) -> AuthorState:
    """One model call, producing an outline rather than steps.

    Not a plan to execute -- the page decides that. It is a map: having said
    "sign in, search, open the record, read the balance" the agent notices when
    it is on a page that fits none of those, which is the cheapest available
    signal that something has gone wrong.

    Emits before it calls the model, not after. This is the only step in the
    whole session where nothing touches the browser -- a person watching a
    headed window sees it sit still while this one network call happens, and a
    slow model turn here read as "the browser is taking a long time to open"
    when the browser had, in fact, already opened. One line of visible
    progress is the fix; the round trip itself is a provider's to speed up.
    """
    w.spend.check()
    await w.emit(
        Thinking(
            run_id=w.request.run_id, seq=0, step=0,
            text="Working out an approach before touching the browser…",
            done=True,
        )
    )
    turn = await w.llm.run_turn(
        system=system_prompt(w.request),
        messages=[
            *state["messages"],
            {
                "role": "user",
                "content": (
                    "Before you touch the browser: in three or four lines, what "
                    "phases do you expect this task to have? Do not list steps "
                    "or guess at element names -- you have not seen the page."
                ),
            },
        ],
        tools=[],
    )
    w.spend.turn(turn.usage, w.llm.model)
    state["outline"] = turn.text.strip()
    if state["outline"]:
        await w.emit(
            Thinking(run_id=w.request.run_id, seq=0, step=0, text=state["outline"], done=True)
        )
        state["messages"].append({"role": "assistant", "content": state["outline"]})
    state["messages"].append(
        {
            "role": "user",
            "content": (
                "The outline is enough. Now inspect the current page and use the "
                "available tools to carry out the task."
            ),
        }
    )
    return state


async def decide(state: AuthorState, w: Wiring) -> AuthorState:
    """One model call, one tool call. Temperature is the client's business.

    The budget is checked *before* the call rather than after, so a limit is a
    ceiling rather than something noticed on the way past.
    """
    w.spend.check()
    # Counted on the Spend, not only in the state. An earlier version
    # incremented the state's step and left `spend.steps` at zero, so the step
    # budget could never trip -- the limit was there, checked, and always
    # satisfied. A loop with a budget that counts nothing is a loop.
    w.spend.step()
    state["step"] += 1

    turn = await w.llm.run_turn(
        system=system_prompt(w.request),
        messages=for_model(state["messages"]),
        tools=_tool_schemas(w),
    )
    w.spend.turn(turn.usage, w.llm.model)

    if turn.text.strip():
        await w.emit(
            Thinking(
                run_id=w.request.run_id,
                seq=0,
                step=state["step"],
                text=turn.text.strip(),
                done=True,
            )
        )
    if turn.raw_content:
        state["messages"].append({"role": "assistant", "content": turn.raw_content})
    elif turn.text:
        state["messages"].append({"role": "assistant", "content": turn.text})

    if not turn.tool_calls:
        # A turn with no tool call is the model talking to itself. Said once it
        # is harmless; said repeatedly it is a loop that spends the whole budget
        # producing prose, so it is nudged rather than ignored.
        state["messages"].append(
            {
                "role": "user",
                "content": (
                    "Call a tool, or call finish. Prose alone does not move the "
                    "browser and does not record anything."
                ),
            }
        )
        return state

    call = turn.tool_calls[0]
    state["pending"] = {"id": call.id, "name": call.name, "input": dict(call.input)}
    return state


async def act(state: AuthorState, w: Wiring) -> AuthorState:
    """Dispatch the pending call through the guard, and tell the model.

    ``finish`` is answered here rather than in a node of its own because it is
    a tool call like any other from the model's side, and giving it a node
    would mean two places that end a session.
    """
    pending = state["pending"]
    state["pending"] = None
    if pending is None:
        return state

    name, arguments, call_id = pending["name"], pending["input"], pending["id"]

    if name == FINISH:
        state["done"] = True
        state["summary"] = str(arguments.get("summary") or "")
        problem = w.tools.marks.unfinished()
        if problem:
            # Refused, and the session continues. A recording with no row
            # boundary cannot be distilled at all, so accepting it here would
            # mean discovering that on the review screen with nothing to do
            # about it.
            state["done"] = False
            await _answer(state, w, call_id, name, False, problem)
            return state
        state["status"] = "succeeded" if arguments.get("complete", True) else "partial"
        return state

    await w.emit(
        ToolCall(
            run_id=w.request.run_id,
            seq=0,
            step=state["step"],
            call_id=call_id,
            name=name,
            arguments=w.tools.redactor.structure(arguments),
        )
    )

    started = time.monotonic()
    result = await w.tools.call(name, arguments)
    duration = int((time.monotonic() - started) * 1000)

    await w.emit(
        ToolResultEvent(
            run_id=w.request.run_id,
            seq=0,
            step=state["step"],
            call_id=call_id,
            name=name,
            ok=not result.is_error,
            duration_ms=duration,
            text=_trim(result.text),
            truncated=len(result.text) > RESULT_BUDGET_CHARS,
        )
    )
    await _answer(state, w, call_id, name, not result.is_error, result.text)
    return state


async def needs_approval(state: AuthorState, w: Wiring) -> bool:
    """Whether the pending call has to stop and ask a person.

    Decided by the guard from the call's own arguments -- submit, delete, pay,
    send -- rather than by asking the model what it thinks of its own next
    action. That is the only version of this check worth having.
    """
    if state["pending"] is None or state["pending"]["name"] == FINISH:
        return False
    from .tools import guard

    verdict = guard(state["pending"]["name"], state["pending"]["input"], w.tools.context())
    return verdict.allowed and verdict.needs_approval


async def ask(state: AuthorState, w: Wiring) -> dict[str, Any]:
    """The payload a human is shown when the graph interrupts.

    The interrupt itself belongs to LangGraph and lives in ``graph.py``; what
    is *in* it is decided here, so that the question a person is asked does not
    depend on which orchestrator is driving.
    """
    pending = state["pending"] or {}
    from .tools import guard

    verdict = guard(pending.get("name", ""), pending.get("input", {}), w.tools.context())
    approval_id = uuid.uuid4().hex
    await w.emit(
        ApprovalRequired(
            run_id=w.request.run_id,
            seq=0,
            step=state["step"],
            approval_id=approval_id,
            call_id=pending.get("id", ""),
            name=pending.get("name", ""),
            arguments=w.tools.redactor.structure(pending.get("input", {})),
            reason=(
                "This looks irreversible. It was classified from what the call "
                "actually does, not from the agent's opinion of it."
            ),
            categories=[c for c in verdict.category.split(", ") if c],
            expires_at="",
        )
    )
    return {
        "approval_id": approval_id,
        "call": {k: pending.get(k) for k in ("id", "name", "input")},
        "categories": verdict.category,
    }


async def resolve(
    state: AuthorState, decision: str, w: Wiring, approval_id: str = ""
) -> AuthorState:
    """Apply a human's answer to the waiting call."""
    await w.emit(
        ApprovalResolved(
            run_id=w.request.run_id,
            seq=0,
            step=state["step"],
            approval_id=approval_id,
            decision="approved" if decision == "approved" else "rejected",
        )
    )
    if decision == "approved":
        return await act(state, w)

    pending, state["pending"] = state["pending"], None
    if pending is not None:
        # Told *why* it was refused rather than merely that it was, because an
        # agent that knows a person declined can try the read-only route; one
        # that only sees a failure retries the same thing.
        await _answer(
            state,
            w,
            pending["id"],
            pending["name"],
            False,
            "A person declined this action. Do not try it again. Continue with "
            "what you can do without it, or call finish and say what is missing.",
        )
    return state


async def finish(state: AuthorState, w: Wiring) -> AuthorState:
    """Close the session and say what it cost."""
    if state["status"] == "running":
        state["status"] = "succeeded" if state["done"] else "failed"
    await w.emit(
        RunFinished(
            run_id=w.request.run_id,
            seq=0,
            status="succeeded" if state["status"] in {"succeeded", "partial"} else "failed",
            steps=state["step"],
            duration_ms=int(w.spend.elapsed * 1000),
            summary=state["summary"] or None,
            result={"answer": state["summary"], "data": None},
            error=state["stopped_by"] or None,
        )
    )
    return state


async def stop_for_budget(
    state: AuthorState, exc: BudgetExhausted, w: Wiring
) -> AuthorState:
    """A limit is a stop, not a crash.

    Everything the session did is kept, and often it is a complete recording
    with the agent merely about to tidy up.
    """
    state["done"] = True
    state["status"] = "partial"
    state["stopped_by"] = str(exc)
    await w.emit(
        ErrorEvent(
            run_id=w.request.run_id,
            seq=0,
            step=state["step"],
            kind=f"budget_{exc.limit}",
            message=str(exc),
            recoverable=True,
        )
    )
    return await finish(state, w)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_schemas(w: Wiring) -> list[dict[str, Any]]:
    """What the model is shown: the session's tools, plus ``finish``."""
    schemas = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema or {"type": "object", "properties": {}},
        }
        for spec in w.tools.tools
    ]
    schemas.append(
        {
            "name": FINISH,
            "description": "End the session. Call this when the task is done, or "
            "when something has stopped you and you cannot continue.",
            "input_schema": FINISH_SCHEMA,
        }
    )
    return schemas


#: How many page snapshots stay in the model's context in full. Everything
#: older is replaced by one line.
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


def _trim(text: str) -> str:
    """Keep the head. The page URL and the top of the tree live there."""
    if len(text) <= RESULT_BUDGET_CHARS:
        return text
    return text[:RESULT_BUDGET_CHARS] + "\n... (truncated)"


async def _answer(
    state: AuthorState, w: Wiring, call_id: str, name: str, ok: bool, text: str
) -> None:
    """Hand a tool's outcome back to the model, in its own message shape."""
    state["messages"].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": _trim(w.tools.redactor.text(text)),
                    "is_error": not ok,
                }
            ],
        }
    )


__all__ = [
    "AuthorRequest",
    "AuthorState",
    "Emit",
    "FINISH",
    "act",
    "ask",
    "decide",
    "finish",
    "load_context",
    "needs_approval",
    "plan",
    "resolve",
    "stop_for_budget",
    "system_prompt",
]
