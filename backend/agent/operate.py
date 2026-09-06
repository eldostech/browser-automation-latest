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
from prompt_loader import EXPLORE, RECOVER, render

from .author import for_model
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


#: Explore's terminal tools, plus the one that reports a value.
FINISH = "finish"
RECORD = "record_value"

EXPLORE_TOOLS: dict[str, dict[str, Any]] = {
    FINISH: {
        "description": (
            "This record is done and every value asked for has been recorded."
        ),
        "properties": {"note": {"type": "string"}},
        "required": [],
    },
    GIVE_UP: {
        "description": (
            "You cannot complete this record. Say what stopped you; the row "
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
    #: The use case, for Explore -- its task text and the values it declares.
    usecase: Any = None
    #: Tool calls one Explore row may make. Larger than a rescue's, because it
    #: is doing the work rather than clearing an obstacle, and still bounded.
    max_explore_actions: int = 24
    #: Set by the explore node so `record_value` can reach it.
    on_record: Any = None
    #: Whether this row is worked out rather than replayed. A property of the
    #: use case's mode, decided by the caller: this module does not read
    #: settings, and a graph that decided its own shape from configuration
    #: would be a graph nobody could test one branch of.
    explore: bool = False
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

    ending = await agent_loop(
        w,
        system=render(
            RECOVER,
            step=state.get("failed_step_id") or "(unknown)",
            error=state.get("error") or "(no reason recorded)",
            allowed_domains=", ".join(w.allowed_domains) or "(nothing configured)",
        ),
        opening="Take a snapshot and decide what is in the way.",
        terminals=RECOVERY_TOOLS,
        step=state.get("start_at", 0),
    )

    if ending.tool == RESUME:
        note = str(ending.arguments.get("note") or "cleared the way")
        state.setdefault("notes", []).append(note)
        await w.emit(
            Thinking(run_id=w.run_id, seq=0, step=state.get("start_at", 0),
                     text=f"Resuming: {note}", done=True)
        )
        return state

    if ending.tool == GIVE_UP:
        reason = str(ending.arguments.get("reason") or "could not clear the way")
        state["error"] = f"{state.get('error', '')} The agent could not help: {reason}"
        state["done"] = True
        return state

    # Out of actions without saying either way. Resuming anyway would be a
    # guess about a page nobody looked at the end state of.
    state.setdefault("notes", []).append(
        "ran out of recovery steps without reaching the page"
    )
    state["done"] = True
    return state


@dataclass
class Ending:
    """How an agent turn ended: which terminal tool, and what it said."""

    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Set when the loop ran out of actions without the agent saying either
    #: way. Not a success and not a refusal -- a third thing, and one that must
    #: not be mistaken for the page being ready.
    exhausted: bool = False


async def agent_loop(
    w: OperateWiring,
    *,
    system: str,
    opening: str,
    terminals: dict[str, dict[str, Any]],
    step: int = 0,
    max_actions: int | None = None,
) -> Ending:
    """Perceive, decide, act -- until the agent calls one of ``terminals``.

    One implementation for both things that drive a browser here: clearing an
    obstacle mid-replay, and working a row out from scratch. They differ in
    their prompt and in which tools end them, and in nothing else -- so a
    second copy of this would be a second place for the budget check, the
    redaction and the tool-result plumbing to drift.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": opening}]
    schemas = _schemas(w, terminals)

    for _ in range(max_actions if max_actions is not None else w.max_actions):
        try:
            w.spend.check()
        except BudgetExhausted as exc:
            return Ending(tool=GIVE_UP, arguments={"reason": str(exc)})

        turn = await w.llm.run_turn(
            system=system, messages=for_model(messages), tools=schemas
        )
        w.spend.turn(turn.usage, w.llm.model)
        if turn.text.strip():
            await w.emit(
                Thinking(
                    run_id=w.run_id, seq=0, step=step, text=turn.text.strip(), done=True
                )
            )
        if turn.raw_content:
            messages.append({"role": "assistant", "content": turn.raw_content})
        elif turn.text:
            messages.append({"role": "assistant", "content": turn.text})

        if not turn.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Call a tool, or call "
                        + " or ".join(sorted(terminals))
                        + ". Prose alone does not move the browser."
                    ),
                }
            )
            continue

        call = turn.tool_calls[0]
        if call.name in terminals:
            return Ending(tool=call.name, arguments=dict(call.input))

        if call.name == RECORD and w.on_record is not None:
            # Ours, and it never reaches the browser. Answered inline so a
            # value is banked the moment it is seen rather than at the end,
            # when the page it came from is several pages back.
            name = str(call.input.get("name") or "")
            value = str(call.input.get("value") or "")
            said = (
                await w.on_record(name, value)
                if name
                else "record_value needs a name."
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": w.tools.redactor.text(said),
                            "is_error": not name,
                        }
                    ],
                }
            )
            continue

        await w.emit(
            ToolCall(
                run_id=w.run_id, seq=0, step=step, call_id=call.id, name=call.name,
                arguments=w.tools.redactor.structure(dict(call.input)),
            )
        )
        result = await w.tools.call(call.name, dict(call.input))
        await w.emit(
            ToolResultEvent(
                run_id=w.run_id, seq=0, step=step, call_id=call.id, name=call.name,
                ok=not result.is_error, duration_ms=0, text=result.text[:2000],
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

    return Ending(exhausted=True)


async def explore(state: OperateState, w: OperateWiring) -> OperateState:
    """Work one row out from the page, with no plan to follow.

    The expensive mode, and the product should say so rather than hide it: this
    is a model call per decision, per row, and four thousand rows is four
    thousand times. It exists because some work genuinely cannot be recorded --
    a page that differs per record, a task whose next step depends on what the
    last one said -- and refusing to offer it does not make that work go away,
    it just makes somebody do it by hand.

    Values are reported as they are seen rather than collected at the end. The
    page a value was on is three pages back by the time a row finishes, and a
    value the agent meant to report and did not is a row somebody redoes.
    """
    usecase = w.usecase
    wanted = list(getattr(usecase, "outputs", []) or [])

    recorded: dict[str, Any] = dict(state.get("outputs") or {})

    async def record(name: str, value: str) -> str:
        recorded[name] = value
        return f"recorded {name}"

    w.on_record = record

    ending = await agent_loop(
        w,
        system=render(
            EXPLORE,
            task=getattr(usecase, "description", "") or getattr(usecase, "name", ""),
            inputs=_as_lines(w.inputs) or "(no values given)",
            outputs=", ".join(wanted) or "(nothing -- just do the task)",
            allowed_domains=", ".join(w.allowed_domains) or "(nothing configured)",
        ),
        opening="Take a snapshot and begin.",
        terminals=EXPLORE_TOOLS,
        max_actions=w.max_explore_actions,
    )

    state["outputs"] = recorded
    missing = [name for name in wanted if name not in recorded]

    if ending.tool == GIVE_UP:
        reason = str(ending.arguments.get("reason") or "could not complete the record")
        state["ok"] = False
        state["error"] = f"The agent stopped: {reason}"
    elif ending.exhausted:
        state["ok"] = False
        state["error"] = (
            "The agent ran out of steps for this row without finishing. Raise the "
            "per-row budget, or record this workflow so it does not have to be "
            "worked out every time."
        )
    elif missing:
        # Finishing without the values is a failed row, not a successful one:
        # a results file with blank columns is worse than a row marked failed,
        # because nobody goes looking for it.
        state["ok"] = False
        state["error"] = (
            "The agent finished without reporting: " + ", ".join(missing) + "."
        )
    else:
        state["ok"] = True
        state["error"] = ""

    w.result = _as_row_result(state, w)
    state["done"] = True
    return state


def _as_lines(values: dict[str, Any]) -> str:
    return chr(10).join(f"- {key}: {value}" for key, value in (values or {}).items())


def _as_row_result(state: OperateState, w: OperateWiring) -> Any:
    from engine import RowResult

    return RowResult(
        ok=bool(state.get("ok")),
        outputs=dict(state.get("outputs") or {}),
        error=state.get("error") or None,
        llm_calls=w.spend.llm_calls,
        llm_tokens=w.spend.tokens,
        llm_usd=w.spend.usd,
    )


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


def _schemas(
    w: OperateWiring, terminals: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    schemas = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema or {"type": "object", "properties": {}},
        }
        for spec in w.tools.tools
    ]
    if w.on_record is not None:
        schemas.append(
            {
                "name": RECORD,
                "description": (
                    "Report one of the values this row was asked for. Call it "
                    "the moment you can see the value, on the page it is on."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The column."},
                        "value": {"type": "string"},
                    },
                    "required": ["name", "value"],
                },
            }
        )
    for name, spec in terminals.items():
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
    graph.add_node("explore", node(explore))
    graph.add_node("learn", node(learn))

    def entry(_state: OperateState) -> str:
        # Explore has no plan to replay, so it does not pass through the
        # engine at all. Everything else starts where it always did.
        return "explore" if w.explore else "replay"

    graph.add_conditional_edges(
        START, entry, {"explore": "explore", "replay": "replay"}
    )
    graph.add_edge("explore", "learn")

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
    explore: bool = False,
    usecase: Any = None,
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
            explore=explore,
            usecase=usecase if usecase is not None else getattr(executor, "usecase", None),
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
