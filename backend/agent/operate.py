"""Running a recorded workflow, with an agent on call rather than in the loop.

The shape, and where the money goes:

    load  ->  replay  ->  done  ->  learn        no model at all
                 |
                 v  (a step failed, and only then)
             recover  ->  replay from that step   MODEL

**The happy path contains no model call.** That is the whole point and it is
worth stating as a property rather than an aspiration: a row where nothing
breaks costs exactly what a Strict row costs, because it *is* a Strict row --
the same `engine.py`, the same steps, the same zero.

## What this adds to what already worked

Guided mode has existed since phase A: the engine runs the plan and an injected
healer re-finds a control that moved. That handles the common failure -- a
button was renamed -- cheaply and well, and it stays the first line of defence.
It is also all it can do. A healer takes a step whose locator stopped matching
and returns a better locator; it cannot dismiss a cookie banner, go back from a
page the site redirected to, or wait for something slow.

This graph is the second line: when the row fails anyway, an agent looks at the
page, does the smallest thing that clears the obstacle, and hands control back
to the engine, which **resumes from the step that failed** rather than starting
the row again. Resuming is not an optimisation -- re-running a partial row is
how a form gets submitted twice.

## The browser

There is one, and it is the engine's. The recovery agent reaches it through
``EngineBrowser``, an MCP-shaped session over the page the replay is already
on, so the guard, the ref discipline, the redaction and the audit all apply
unchanged. Starting a second browser would be a second blank tab looking at
nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypedDict

from events import ErrorEvent, Thinking, ToolCall, ToolResult as ToolResultEvent
from llm import LLMClient
from prompt_loader import RECOVER, render

from .budget import Budget, BudgetExhausted, Spend
from .inprocess import EngineBrowser
from .session import AgentToolSession

log = logging.getLogger(__name__)

#: Our two tools. The agent ends its own turn deliberately, either because the
#: page is ready or because it cannot get there -- both are answers, and a
#: recovery that simply runs out of steps is neither.
RESUME = "resume"
GIVE_UP = "give_up"

RECOVERY_TOOLS: dict[str, dict[str, Any]] = {
    RESUME: {
        "description": (
            "The page is where the failed step expects it. Hand control back "
            "to the workflow, which carries on from that step."
        ),
        "properties": {
            "note": {"type": "string", "description": "What was in the way."}
        },
        "required": [],
    },
    GIVE_UP: {
        "description": (
            "You cannot get the page there. Say what is in the way; the row "
            "fails with your reason attached."
        ),
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
    },
}


class OperateState(TypedDict, total=False):
    """What the graph carries. Plain values, because it is checkpointed."""

    #: Where in ``row_steps`` to start. Non-zero after a recovery.
    start_at: int
    #: What earlier steps already read. A value extracted before the failure is
    #: not extracted again.
    outputs: dict[str, Any]
    ok: bool
    done: bool
    error: str
    failed_step_id: str
    #: How many times an agent has been asked to clear the way on this row.
    attempts: int
    #: The recovery conversation, kept across the turns of one attempt only.
    messages: list[dict[str, Any]]
    #: What the recovery said it did, for the run trail.
    notes: list[str]


def initial_state(inputs: dict[str, Any] | None = None) -> OperateState:
    return OperateState(
        start_at=0,
        outputs={},
        ok=False,
        done=False,
        error="",
        failed_step_id="",
        attempts=0,
        messages=[],
        notes=[],
    )


@dataclass
class OperateWiring:
    """The live objects. Never checkpointed -- see author.py for why."""

    executor: Any
    inputs: dict[str, Any]
    tools: AgentToolSession
    llm: LLMClient
    spend: Spend
    emit: Callable[[Any], Awaitable[None]]
    run_id: str = ""
    allowed_domains: tuple[str, ...] = ()
    #: How many times a row may be rescued before it is allowed to fail. Small
    #: on purpose: a row needing four rescues is a use case that needs
    #: re-recording, and spending four model calls per row across four thousand
    #: rows is the bill this whole design exists to avoid.
    max_attempts: int = 2
    #: Tool calls within one rescue.
    max_actions: int = 6
    #: The last RowResult. On the wiring rather than in the state for the same
    #: reason the tool session is: the state is what gets checkpointed, and a
    #: result object full of StepOutcomes is not something to write to Postgres
    #: after every node.
    result: Any = None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def replay(state: OperateState, w: OperateWiring) -> OperateState:
    """Run the plan. No model, and the only node most rows ever touch."""
    result = await w.executor.run_row(
        w.inputs, start_at=state.get("start_at", 0), outputs=state.get("outputs") or {}
    )
    state["outputs"] = dict(result.outputs or {})
    state["ok"] = result.ok
    state["error"] = result.error or ""
    state["failed_step_id"] = result.failed_step_id or ""
    if result.ok:
        state["done"] = True
    else:
        # Where to carry on from, if a rescue clears the way. None means the
        # failure was not in a row step -- a missing input, say -- and there is
        # nothing to resume.
        state["start_at"] = (
            result.failed_index if result.failed_index is not None else -1
        )
    w.result = result
    return state


async def recover(state: OperateState, w: OperateWiring) -> OperateState:
    """Ask an agent to clear the way. The only node that spends anything.

    Bounded twice: by the number of rescues a row may have, and by the number
    of tool calls within one. Both are small, because a row that needs a lot of
    rescuing is a use case that needs re-recording, and the alternative is
    paying for that discovery once per row across a whole batch.
    """
    state["attempts"] = state.get("attempts", 0) + 1
    await w.emit(
        Thinking(
            run_id=w.run_id,
            seq=0,
            step=state.get("start_at", 0),
            text=(
                f"Step {state.get('failed_step_id')} failed. Looking at the page "
                "to see whether it can be cleared."
            ),
            done=True,
        )
    )

    system = render(
        RECOVER,
        step=state.get("failed_step_id") or "(unknown)",
        error=state.get("error") or "(no reason recorded)",
        allowed_domains=", ".join(w.allowed_domains) or "(nothing configured)",
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Take a snapshot and decide what is in the way.",
        }
    ]

    for _ in range(w.max_actions):
        try:
            w.spend.check()
        except BudgetExhausted as exc:
            state["error"] = str(exc)
            return state

        turn = await w.llm.run_turn(
            system=system, messages=messages, tools=_schemas(w)
        )
        w.spend.turn(turn.usage, w.llm.model)
        if turn.raw_content:
            messages.append({"role": "assistant", "content": turn.raw_content})
        elif turn.text:
            messages.append({"role": "assistant", "content": turn.text})

        if not turn.tool_calls:
            messages.append(
                {"role": "user", "content": "Call a tool, or call resume or give_up."}
            )
            continue

        call = turn.tool_calls[0]
        if call.name == RESUME:
            note = str(call.input.get("note") or "cleared the way")
            state.setdefault("notes", []).append(note)
            await w.emit(
                Thinking(run_id=w.run_id, seq=0, step=state.get("start_at", 0),
                         text=f"Resuming: {note}", done=True)
            )
            return state
        if call.name == GIVE_UP:
            reason = str(call.input.get("reason") or "could not clear the way")
            state["error"] = f"{state.get('error', '')} The agent could not help: {reason}"
            state["done"] = True
            return state

        await w.emit(
            ToolCall(
                run_id=w.run_id, seq=0, step=state.get("start_at", 0),
                call_id=call.id, name=call.name,
                arguments=w.tools.redactor.structure(dict(call.input)),
            )
        )
        result = await w.tools.call(call.name, dict(call.input))
        await w.emit(
            ToolResultEvent(
                run_id=w.run_id, seq=0, step=state.get("start_at", 0),
                call_id=call.id, name=call.name, ok=not result.is_error,
                duration_ms=0, text=result.text[:2000],
            )
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": w.tools.redactor.text(result.text)[:6000],
                        "is_error": result.is_error,
                    }
                ],
            }
        )

    # Out of actions without saying either way. Resuming anyway would be a
    # guess about a page nobody looked at the end state of.
    state["notes"].append("ran out of recovery steps without reaching the page")
    state["done"] = True
    return state


async def learn(state: OperateState, w: OperateWiring) -> OperateState:
    """Record what a rescue found, so the next row does not pay for it again.

    Only the *engine's* repairs go to healing memory -- it is a store of
    locators, and that is what a heal produces. A recovery produces a sequence
    of actions on a page, which is not a locator and must not be written there
    pretending to be one. What it produces here is a note on the run, which is
    what a person needs in order to decide whether to re-record.
    """
    notes = state.get("notes") or []
    if notes:
        await w.emit(
            ErrorEvent(
                run_id=w.run_id,
                seq=0,
                step=state.get("start_at", 0),
                kind="recovered",
                message=(
                    f"This row needed {state.get('attempts', 0)} rescue(s): "
                    + "; ".join(notes)
                    + ". A use case needing these regularly is one to re-record."
                ),
                recoverable=True,
            )
        )
    return state


def _schemas(w: OperateWiring) -> list[dict[str, Any]]:
    schemas = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema or {"type": "object", "properties": {}},
        }
        for spec in w.tools.tools
    ]
    for name, spec in RECOVERY_TOOLS.items():
        schemas.append(
            {
                "name": name,
                "description": spec["description"],
                "input_schema": {
                    "type": "object",
                    "properties": spec["properties"],
                    "required": spec["required"],
                },
            }
        )
    return schemas


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def build(w: OperateWiring, checkpointer: Any = None):
    """The compiled operate graph. LangGraph imported here, not at module scope."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(OperateState)

    def node(func):
        # A real coroutine function, not a lambda returning one: LangGraph
        # awaits an async node and calls a sync one, and a sync callable that
        # hands back a coroutine falls between the two -- it is never awaited
        # and arrives as an unusable state update.
        async def run(state: OperateState) -> OperateState:
            return await func(state, w)

        run.__name__ = func.__name__
        return run

    graph.add_node("replay", node(replay))
    graph.add_node("recover", node(recover))
    graph.add_node("learn", node(learn))

    graph.add_edge(START, "replay")

    def after_replay(state: OperateState) -> str:
        if state.get("ok"):
            return "learn"
        if state.get("done"):
            return "learn"
        if state.get("start_at", -1) < 0:
            # Nothing to resume from: the row failed before any step ran, or
            # outside the row steps. An agent cannot help with that.
            return "learn"
        if state.get("attempts", 0) >= w.max_attempts:
            return "learn"
        return "recover"

    graph.add_conditional_edges(
        "replay", after_replay, {"recover": "recover", "learn": "learn"}
    )
    graph.add_conditional_edges(
        "recover",
        lambda state: "learn" if state.get("done") else "replay",
        {"replay": "replay", "learn": "learn"},
    )
    graph.add_edge("learn", END)
    return graph.compile(checkpointer=checkpointer)


async def run_row_with_agent(
    executor: Any,
    inputs: dict[str, Any],
    *,
    llm: LLMClient,
    emit: Callable[[Any], Awaitable[None]],
    run_id: str = "",
    allowed_domains: tuple[str, ...] = (),
    budget: Budget | None = None,
    redactor: Any = None,
    max_attempts: int = 2,
) -> Any:
    """One row, replay-first, with an agent on call. Returns the RowResult.

    The browser is the executor's throughout: the recovery agent is a guest on
    the page the replay is already on, reached through an MCP-shaped adapter so
    that everything above it -- guard, refs, redaction, audit -- is the code
    that already exists.
    """
    session = AgentToolSession(
        EngineBrowser(executor),
        allowed_domains=allowed_domains,
        # A recovery exists to change the page. Read-only would leave it able
        # to look at the obstacle and nothing else.
        may_write=True,
        redactor=redactor,
    )
    async with session:
        wiring = OperateWiring(
            executor=executor,
            inputs=inputs,
            tools=session,
            llm=llm,
            spend=Spend(budget=budget or Budget(steps=None, seconds=None)),
            emit=emit,
            run_id=run_id,
            allowed_domains=allowed_domains,
            max_attempts=max_attempts,
        )
        from . import graph as graph_module

        compiled = build(wiring, graph_module.memory_checkpointer())
        final: OperateState = initial_state()
        async for state in compiled.astream(
            initial_state(),
            config={"configurable": {"thread_id": run_id or "row"}, "recursion_limit": 60},
            stream_mode="values",
        ):
            final = state

    result = wiring.result
    if result is not None and not result.ok and final.get("error"):
        result.error = final["error"]
    return result


__all__ = [
    "GIVE_UP",
    "RESUME",
    "OperateState",
    "OperateWiring",
    "build",
    "initial_state",
    "learn",
    "recover",
    "replay",
    "run_row_with_agent",
]
