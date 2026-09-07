"""This is a credential -- binds to a stored slot, never written verbatim."""

from __future__ import annotations

from .base import ToolDef, value_marking_handler

NAME = "mark_as_secret"

TOOL = ToolDef(
    name=NAME,
    description=(
        "This is a credential. It binds to a stored credential slot; the "
        "value itself is never written into the use case."
    ),
    properties={
        "ref": {"type": "string"},
        "slot": {"type": "string", "description": "The credential slot name."},
    },
    required=["ref", "slot"],
    needs_ref=True,
    handler=value_marking_handler(NAME, "slot"),
)
