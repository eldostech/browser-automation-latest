"""The agent: a browser driven by a model, beside an engine that repeats.

The platform's guarantee is that a replay costs nothing and does the same thing
every time. That guarantee is mechanical -- ``engine.py`` does not import
``llm`` and has no parameter that could accept a model client -- and it is why
this package exists as a separate thing that *calls* the engine rather than as
a branch inside it. The engine never learns that an agent exists.

Everything here is optional. The client library and the Node server it starts
live in ``requirements-agent.txt``, not ``requirements.txt``, so a deployment
that only replays installs neither and behaves exactly as it did before this
package was written. ``AGENT_ENABLED`` reports that the way ``RECORDER_ENABLED``
already reports a machine with no display: as an answer, rather than as a spawn
failing with a message nobody can act on.

Nothing in this package imports FastAPI or ``store``. Its entry point takes a
request and yields results, so the HTTP surface -- ours today, AgentCore's
later -- is a thin adapter over it.
"""

from __future__ import annotations

from .provider import (
    Availability,
    BrowserProvider,
    LocalPlaywrightMCP,
    MCPSession,
    ToolResult,
    ToolSpec,
    local_availability,
)
from .author import AuthorRequest, AuthorState
from .budget import Budget, BudgetExhausted, Spend
from .marks import MARK_TOOLS, Described, Mark, Marks, describe_element
from .run import AuthorResult, run_agent_session
from .session import AgentToolSession, Recorder, ToolCallRecord
from .tools import DISTILS_TO, GuardContext, Guarded, PERCEPTION, REFUSED, guard, offered

__all__ = [
    "AgentToolSession",
    "Availability",
    "MARK_TOOLS",
    "AuthorRequest",
    "AuthorResult",
    "AuthorState",
    "Budget",
    "BudgetExhausted",
    "Spend",
    "run_agent_session",
    "Described",
    "Mark",
    "Marks",
    "describe_element",
    "BrowserProvider",
    "DISTILS_TO",
    "GuardContext",
    "Guarded",
    "LocalPlaywrightMCP",
    "MCPSession",
    "PERCEPTION",
    "REFUSED",
    "Recorder",
    "ToolCallRecord",
    "ToolResult",
    "ToolSpec",
    "guard",
    "local_availability",
    "offered",
]
