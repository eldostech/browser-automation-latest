"""The authoring loop, built on `langchain.agents.create_agent`.

This used to be a hand-built `StateGraph`: a `decide` node calling the model,
an `act` node dispatching tools, and an `approve` node that suspended for a
person. `create_agent` now provides the model-calling and tool-dispatch nodes,
and this file's job shrinks to configuring four middleware:

* :class:`~agent.middleware.BudgetMiddleware` -- the budget check, the
  prompt-cache hint, and the snapshot-pruning that used to live in `decide`
  and `LangChainLLM.run_turn`.
* :class:`~agent.middleware.ThinkingMiddleware` -- emits whatever the model
  said before it acted as the same `Thinking` event the older hand-rolled
  loop always produced. `create_agent`'s own nodes never looked at a turn's
  prose, only its tool calls, so without this a session could reason at
  length and still look silent end to end.
* :class:`~agent.middleware.FinishMiddleware` -- answers the `finish` call
  and ends the graph, replacing `act`'s special case for it.
* ``HumanInTheLoopMiddleware`` -- LangChain's own approval flow, not a
  hand-rolled one. Its `after_model` hook recomputes the interrupt payload
  fresh from already-checkpointed state every time, with nothing minted and
  nothing emitted before the `interrupt()` call -- which is the safe form of
  the pattern this file's previous `approve` node got wrong. See
  ``run.py`` for how a person's decision actually reaches it: this file only
  configures *which* calls stop to ask, via `guard()`, the same function that
  already decides it for the deterministic dispatch path underneath.

LangChain is imported here, never at module scope -- it is an optional extra
(`pip install -r backend/requirements-agent.txt`), and a deployment that only
replays must not need it installed. A test asserts the absence.
"""

from __future__ import annotations

import logging
from typing import Any

from .author import Wiring, system_prompt
from .guardrails import guard
from .tool_adapter import as_langchain_tools
from .tools import TOOLS

log = logging.getLogger(__name__)


def available() -> bool:
    """Whether this deployment has `create_agent` installed."""
    try:
        import langchain.agents  # noqa: F401
    except ImportError:
        return False
    return True


def build(w: Wiring, checkpointer: Any = None):
    """The compiled graph.

    The wiring -- the open tool session, the model client, the emit callable
    -- is closed over by the middleware instances built here rather than
    carried in the state, for the same reason it always was: the state is
    what LangGraph checkpoints, and a live browser session cannot be written
    to Postgres.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.tools import StructuredTool, ToolException

    from .middleware import AuthorGraphState, BudgetMiddleware, FinishMiddleware, ThinkingMiddleware

    async def _unreachable_finish(**kwargs: Any) -> str:
        # `FinishMiddleware.aafter_model` answers every `finish` call itself
        # -- either by ending the graph or by injecting a refusal -- and
        # strips it from the AI message either way, so the tool node never
        # actually dispatches this. It still has to exist: the model is shown
        # its schema, and a tool offered but never registered is a harder
        # bug to explain than a coroutine that is never called.
        raise ToolException("finish is answered by the graph, not dispatched.")

    finish_def = TOOLS["finish"]
    finish_tool = StructuredTool(
        name=finish_def.name,
        description=finish_def.description,
        args_schema=finish_def.input_schema,
        coroutine=_unreachable_finish,
        handle_tool_error=True,
    )
    tools = [*as_langchain_tools(w), finish_tool]

    def needs_approval(request: Any) -> bool:
        """The same question `guard()` already answers for the deterministic
        dispatch path -- decided from the call's own arguments, not asked of
        the model."""
        name, args = request.tool_call["name"], request.tool_call["args"]
        return guard(name, args, w.tools.context()).needs_approval

    interrupt_on = {
        spec.name: {"allowed_decisions": ["approve", "reject"], "when": needs_approval}
        for spec in w.tools.tools
    }

    return create_agent(
        model=w.llm.raw,
        tools=tools,
        system_prompt=system_prompt(w.request),
        middleware=[
            BudgetMiddleware(w.spend, model_name=w.llm.model, marks=w.tools.marks),
            ThinkingMiddleware(w.emit, run_id=w.request.run_id, spend=w.spend),
            FinishMiddleware(w.tools),
            HumanInTheLoopMiddleware(interrupt_on=interrupt_on),
        ],
        state_schema=AuthorGraphState,
        checkpointer=checkpointer,
    )


def memory_checkpointer():
    """An in-process checkpointer, for a single-machine install and for tests.

    Postgres is the one a deployment wants -- it is what makes a session
    survive a restart -- and it is a different class from the same library over
    the database this application already owns.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


__all__ = ["available", "build", "memory_checkpointer"]
