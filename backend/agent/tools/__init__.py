"""Tools this codebase implements itself: the marks that declare a
recording's shape, and `finish`. Playwright's own tools never appear here --
they are advertised by Playwright MCP and classified by name in
`agent/guardrails/catalog.py`, since nobody here implements them.

Adding a tool is adding a file in this directory and one line in the tuple
below -- an explicit list rather than a filesystem scan, so a mistyped
filename fails to import instead of silently leaving a tool the model was
never actually offered.
"""

from __future__ import annotations

from . import (
    begin_row,
    describe_element,
    end_row,
    finish,
    mark_as_input,
    mark_as_output,
    mark_as_secret,
    mark_setup_complete,
)
from .base import Handler, ToolDef

#: Every tool this package answers itself, by name. `finish` is here too --
#: for the schema `graph.py` reads -- but its `handler` is `None`, which is
#: also how `session.py` knows not to advertise or dispatch it as one of its
#: own: it never runs through this dispatch at all.
TOOLS: dict[str, ToolDef] = {
    module.TOOL.name: module.TOOL
    for module in (
        mark_setup_complete,
        begin_row,
        end_row,
        mark_as_input,
        mark_as_output,
        mark_as_secret,
        describe_element,
        finish,
    )
}

__all__ = ["Handler", "ToolDef", "TOOLS"]
