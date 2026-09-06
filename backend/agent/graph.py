"""The authoring loop as a LangGraph graph.

Everything the nodes *do* is in ``author.py``. This file is the wiring, and it
is separate for a reason worth stating: LangGraph is an optional dependency, so
importing it at module scope would make the whole agent package unimportable in
a deployment that installed only ``requirements.txt``. The import happens
inside :func:`build`, and a test asserts it.

What the graph buys, beyond a `while` loop:

**A checkpoint per node.** A session that ends because a person went home, or
because the process was restarted, resumes rather than starts again. That
matters more here than in most agent products: an authoring session is the
expensive part, and losing one at step thirty is losing real money.

**A real interrupt.** ``approve`` is not a poll on a database flag. The graph
stops, the state is persisted, and resuming with a decision continues from
exactly there -- whether that is four seconds later in a browser tab or the
next morning. That is the rendezvous ``RUN_APPROVE`` was defined for and never
got.

**One place the loop shape lives.** The conditional edges are the whole control
flow, visible in twenty lines, rather than spread through a function with
`break`s in it.

    load_context -> plan -> decide -> ┬─ approve ─┬─ act ─┬─ decide  (not done)
                                      └───────────┴───────┴─ finish  (done)
"""

from __future__ import annotations

import logging
from typing import Any

from .author import (
    AuthorState,
    Wiring,
    _mirror,
    act,
    ask,
    decide,
    finish,
    load_context,
    needs_approval,
    plan,
    resolve,
    stop_for_budget,
)
from .budget import BudgetExhausted

log = logging.getLogger(__name__)

#: The one place the node names are written down, so the graph and anything
#: reading a checkpoint agree.
NODES = ("load_context", "plan", "decide", "approve", "act", "finish")


def available() -> bool:
    """Whether this deployment has the graph library installed."""
    try:
        import langgraph.graph  # noqa: F401
    except ImportError:
        return False
    return True


def build(w: Wiring, checkpointer: Any = None):
    """The compiled graph. LangGraph is imported here, never at module scope.

    The wiring -- the open tool session, the model client, the emit callable --
    is closed over rather than carried in the state, because the state is the
    thing LangGraph serialises after every node. A live browser session cannot
    be written to Postgres, and a checkpoint that tried would be a checkpoint
    that could not be read back.
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    graph = StateGraph(AuthorState)

    # Every node is wrapped so that a budget stop ends the session cleanly
    # rather than propagating as an exception through the graph. A limit
    # reached is an outcome, not a failure, and the trajectory is still worth
    # keeping -- often it is a complete recording.
    def guarded(func):
        async def node(state: AuthorState) -> AuthorState:
            # No early return on `done`. The conditional edges already route a
            # finished session to `finish`, and skipping a node because the
            # session is over would skip `finish` itself -- which is the node
            # that says it is over.
            try:
                state = await func(state, w)
            except BudgetExhausted as exc:
                log.info(
                    "agent session stopped by budget",
                    extra={"run_id": w.request.run_id, "limit": exc.limit},
                )
                state = await stop_for_budget(state, exc, w)
            # One place the spend is copied into the checkpoint, rather than
            # the end of every node, where the next node added would forget.
            return _mirror(state, w)

        node.__name__ = func.__name__
        return node

    async def approve(state: AuthorState) -> AuthorState:
        """Stop, and wait for a person.

        `interrupt` suspends the graph and persists the state. Resuming with a
        `Command(resume=...)` returns that value *from this call* -- so the
        code below reads as though the human answered inline, which is what
        makes the rendezvous survivable across a restart.
        """
        question = await ask(state, w)
        decision = interrupt(question)
        answer = decision if isinstance(decision, str) else str(
            (decision or {}).get("decision", "rejected")
        )
        return _mirror(await resolve(state, answer, w, question["approval_id"]), w)

    graph.add_node("load_context", guarded(load_context))
    graph.add_node("plan", guarded(plan))
    graph.add_node("decide", guarded(decide))
    graph.add_node("approve", approve)
    graph.add_node("act", guarded(act))
    graph.add_node("finish", guarded(finish))

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "plan")
    graph.add_edge("plan", "decide")

    async def after_decide(state: AuthorState) -> str:
        if state.get("done"):
            return "finish"
        if state.get("pending") is None:
            # The model produced prose and was nudged. Go round again rather
            # than ending: said once this is harmless.
            return "decide"
        return "approve" if await needs_approval(state, w) else "act"

    graph.add_conditional_edges(
        "decide", after_decide,
        {"approve": "approve", "act": "act", "decide": "decide", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "approve", lambda state: "finish" if state.get("done") else "decide",
        {"decide": "decide", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "act", lambda state: "finish" if state.get("done") else "decide",
        {"decide": "decide", "finish": "finish"},
    )
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=checkpointer)


def memory_checkpointer():
    """An in-process checkpointer, for a single-machine install and for tests.

    Postgres is the one a deployment wants -- it is what makes a session
    survive a restart -- and it is a different class from the same library over
    the database this application already owns.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


__all__ = ["NODES", "available", "build", "memory_checkpointer"]
