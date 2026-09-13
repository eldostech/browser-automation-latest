"""Ends the session. Schema only -- unlike every other tool here, this one
has no `handler`: `agent/middleware.py`'s `FinishMiddleware` answers every
`finish` call itself, either by ending the graph or by injecting a refusal,
and strips it from the model's turn either way, so dispatch never reaches
this file. It still needs a home: the model has to be shown its schema, and
`graph.py` builds the `finish` tool from it.
"""

from __future__ import annotations

from .base import ToolDef

NAME = "finish"

TOOL = ToolDef(
    name=NAME,
    description=(
        "Call this when the task is done and the marks describe it, or "
        "when you cannot complete it and need to say why."
    ),
    properties={
        "summary": {
            "type": "string",
            "description": "What you did, and anything a reviewer should check.",
        },
        "complete": {
            "type": "boolean",
            "description": "False if something stopped you before the task was done.",
        },
    },
    required=["summary"],
    needs_ref=False,
    handler=None,
)
