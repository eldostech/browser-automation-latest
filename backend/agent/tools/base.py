"""The shape every Python-implemented tool declares itself in.

Playwright's own tools are advertised by Playwright MCP and never touch this
package -- `guardrails/catalog.py` classifies those by name, since nobody
here implements them. These are the ones this codebase answers itself: the
marks that declare a recording's shape, and `finish`. One file per tool, each
exporting a `ToolDef`, is what makes adding one "add a file and one line in
`__init__.py`" rather than "edit a schema dict in one file and an if/elif
chain in `session.py`".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..marks import Described, Marks
from ..providers.base import ToolResult

#: `handler(marks, described, seq, arguments) -> ToolResult`. `described` is
#: `None` unless the tool's `needs_ref` asked the dispatcher to resolve one
#: first, and `seq` is the call's own sequence number -- what `Marks`'
#: bookkeeping keys rows and marks by. `None` for `finish`: it is answered by
#: `FinishMiddleware` before dispatch would ever reach it.
Handler = Callable[[Marks, "Described | None", int, dict[str, Any]], ToolResult]


@dataclass(slots=True)
class ToolDef:
    """One tool this codebase implements itself, wherever it is dispatched
    from -- an agent session today, whatever else calls `Marks` later."""

    name: str
    description: str
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    #: Whether the dispatcher should resolve `arguments["ref"]` into a
    #: `Described` (durable locator + ambiguity check) before calling the
    #: handler. Every tool but the three plain state transitions needs one.
    needs_ref: bool = True
    handler: Handler | None = None

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.properties, "required": self.required}


def value_marking_handler(tool_name: str, field_key: str) -> Handler:
    """A handler for a tool that names a value on the current element.

    `mark_as_input`, `mark_as_output` and `mark_as_secret` are the same
    handler in every way but which argument key carries the name and which
    mark they record it as -- a factory here is what keeps three files from
    each carrying their own copy of the same four lines.
    """
    def handler(marks: Marks, described: "Described | None", seq: int, arguments: dict[str, Any]) -> ToolResult:
        assert described is not None  # needs_ref=True for every caller of this
        value = str(arguments.get(field_key) or "")
        if not value:
            return ToolResult.failed(f"{tool_name} needs a {field_key}.")
        problem = marks.mark_value(tool_name, seq, described.ref, value, described)
        if problem:
            return ToolResult.failed(problem)
        # Told what the name *became*, not what was asked for. A model that
        # said "Account number" and is answered "recorded" will use its own
        # spelling in the next call and in its summary, and then two names
        # for one column are loose in the session.
        recorded = marks.entries[-1].name
        return ToolResult(text=f"{described.describe_first()} recorded as {recorded!r}.")

    return handler


__all__ = ["Handler", "ToolDef", "value_marking_handler"]
