"""Read this element's value into the results file."""

from __future__ import annotations

from .base import ToolDef, value_marking_handler

NAME = "mark_as_output"

TOOL = ToolDef(
    name=NAME,
    description=(
        "Read this element's value into the results file. Becomes an "
        "extract step at this point in the run, on the page it was seen "
        "on. Only valid inside a row: call begin_row first."
    ),
    properties={"ref": {"type": "string"}, "column": {"type": "string"}},
    required=["ref", "column"],
    needs_ref=True,
    handler=value_marking_handler(NAME, "column"),
)
