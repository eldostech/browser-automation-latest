"""Everything before this call was setup; the rest is once-per-row work."""

from __future__ import annotations

from typing import Any

from ..marks import Described, Marks
from ..providers.base import ToolResult
from .base import ToolDef

NAME = "mark_setup_complete"


def handler(marks: Marks, described: Described | None, seq: int, arguments: dict[str, Any]) -> ToolResult:
    problem = marks.setup_complete(seq)
    return ToolResult.failed(problem) if problem else ToolResult(text="Setup ends here.")


TOOL = ToolDef(
    name=NAME,
    description=(
        "Everything done so far was setup -- signing in, choosing a "
        "workspace -- and runs once per batch rather than once per row. "
        "Call this exactly once, when the per-record work is about to start."
    ),
    needs_ref=False,
    handler=handler,
)
