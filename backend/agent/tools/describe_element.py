"""Resolve a ref into the durable locator a recorded step would carry, and
say how many elements it matches. Call before marking anything: an ambiguous
locator is far cheaper to fix now than to discover in a failed batch."""

from __future__ import annotations

from typing import Any

from ..marks import Described, Marks
from ..providers.base import ToolResult
from .base import ToolDef

NAME = "describe_element"


def handler(marks: Marks, described: Described | None, seq: int, arguments: dict[str, Any]) -> ToolResult:
    assert described is not None  # needs_ref=True
    return ToolResult(text=described.as_text(), is_error=described.matches == 0)


TOOL = ToolDef(
    name=NAME,
    description=(
        "Resolve an element reference from the current snapshot into the "
        "durable locator a recorded step would use, and report how many "
        "elements each way of finding it matches. Call this before marking "
        "anything: an ambiguous locator is far cheaper to fix now."
    ),
    properties={"ref": {"type": "string", "description": "A ref such as 'e12'."}},
    required=["ref"],
    needs_ref=True,
    handler=handler,
)
