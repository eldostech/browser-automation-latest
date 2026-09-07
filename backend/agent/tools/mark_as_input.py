"""A value that changes per record -- becomes a spreadsheet column."""

from __future__ import annotations

from .base import ToolDef, value_marking_handler

NAME = "mark_as_input"

TOOL = ToolDef(
    name=NAME,
    description=(
        "This value changes per record and should come from a spreadsheet "
        "column. The step that types it becomes a template. Only valid "
        "inside a row: call begin_row first."
    ),
    properties={
        "ref": {"type": "string"},
        "name": {"type": "string", "description": "The column name."},
    },
    required=["ref", "name"],
    needs_ref=True,
    handler=value_marking_handler(NAME, "name"),
)
