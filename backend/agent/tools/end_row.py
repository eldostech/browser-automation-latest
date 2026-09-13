"""The work for one record is finished."""

from __future__ import annotations

from typing import Any

from ..marks import Described, Marks
from ..providers.base import ToolResult
from .base import ToolDef

NAME = "end_row"


def handler(marks: Marks, described: Described | None, seq: int, arguments: dict[str, Any]) -> ToolResult:
    problem = marks.end_row(seq)
    return ToolResult.failed(problem) if problem else ToolResult(text="Row finished.")


TOOL = ToolDef(
    name=NAME,
    description="The work for one record is finished.",
    needs_ref=False,
    handler=handler,
)
