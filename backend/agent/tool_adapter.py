"""Bridges `AgentToolSession`'s tools to LangChain's tool-calling convention.

`AgentToolSession.call()` already does everything that matters: the guard,
the ref discipline, secret substitution, marks, redaction, audit. This file's
job is making a `ToolSpec` look like a `StructuredTool` so `create_agent` can
offer it to a model -- nothing about dispatch changes, and nothing here knows
whether a tool came from the browser, a mark, or a registered MCP server.
That is deliberate: Phase 1 already made the tool list itself data-driven,
and this is what lets `create_agent` see the result of that without caring
how it was assembled.

It also emits the two events a call produces -- `ToolCall` before dispatch,
`ToolResult` after -- because nothing else does. The old `act()` node used to
own both the dispatch *and* this pair of emissions; here they are separated,
since `AgentToolSession.call()` is the dispatch this codebase already trusts
and this file's only addition is telling the dashboard about it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool, ToolException

from events import ToolCall
from events import ToolResult as ToolResultEvent

from .providers.base import ToolSpec

if TYPE_CHECKING:
    from .author import Wiring

#: How much of a tool result the model is shown. Ported from `author.py`'s
#: own constant of the same name and the same value -- kept local rather than
#: imported so this module still only ever imports `author` under
#: `TYPE_CHECKING`, matching the rest of the deferred-import discipline here.
RESULT_BUDGET_CHARS = 6000


def as_langchain_tools(w: "Wiring") -> list[StructuredTool]:
    """One `StructuredTool` per tool `w.tools` currently offers.

    Built fresh per session rather than cached anywhere, because
    `w.tools.tools` reflects whatever the browser and any registered MCP
    servers actually advertised for *this* session -- Phase 1's whole point,
    and something that can differ session to session as the registry changes.
    """
    return [_tool_for(w, spec) for spec in w.tools.tools]


def _tool_for(w: "Wiring", spec: ToolSpec) -> StructuredTool:
    async def call(**kwargs: Any) -> str:
        session = w.tools
        call_id = f"tc{session._seq + 1}"  # noqa: SLF001 - about to become the real one
        await w.emit(
            ToolCall(
                run_id=w.request.run_id, seq=0, step=w.spend.steps,
                call_id=call_id, name=spec.name,
                arguments=session.redactor.structure(kwargs),
            )
        )
        started = time.monotonic()
        result = await session.call(spec.name, kwargs)
        duration = int((time.monotonic() - started) * 1000)
        text = result.text
        await w.emit(
            ToolResultEvent(
                run_id=w.request.run_id, seq=0, step=w.spend.steps,
                call_id=call_id, name=spec.name, ok=not result.is_error,
                duration_ms=duration,
                text=text[:RESULT_BUDGET_CHARS],
                truncated=len(text) > RESULT_BUDGET_CHARS,
            )
        )
        if result.is_error:
            # A refusal is not a crash. `session.call` already put the
            # guard's reason in `result.text`; `ToolException` is what turns
            # that into a `status="error"` ToolMessage the model reads and
            # can act on, rather than an exception that ends the run.
            raise ToolException(text)
        return text

    return StructuredTool(
        name=spec.name,
        description=spec.description,
        args_schema=spec.input_schema or {"type": "object", "properties": {}},
        coroutine=call,
        handle_tool_error=True,
    )


__all__ = ["as_langchain_tools"]
