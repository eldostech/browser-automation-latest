"""The work for one record starts here."""

from __future__ import annotations

from typing import Any

from ..marks import Described, Marks
from ..providers.base import ToolResult
from .base import ToolDef

NAME = "begin_row"


def handler(marks: Marks, described: Described | None, seq: int, arguments: dict[str, Any]) -> ToolResult:
    key = str(arguments.get("key") or "")
    problem = marks.begin_row(seq, key)
    return ToolResult.failed(problem) if problem else ToolResult(text=f"Row {key!r} started.")


TOOL = ToolDef(
    name=NAME,
    description=(
        "The work for one record starts here. Everything until end_row "
        "becomes the steps that repeat, once per row of the spreadsheet."
    ),
    properties={
        "key": {
            "type": "string",
            "description": "What identifies this record, e.g. 'A-1001'.",
        }
    },
    required=["key"],
    needs_ref=False,
    handler=handler,
)
