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

Four things a reader can name on sight, one directory each:
``tools/`` (what this codebase implements itself), ``guardrails/`` (what's
allowed and what needs a person), ``providers/`` (where the browser/MCP
connection comes from), and everything else here -- the session that ties
them together, the graph that drives it, and the pipeline that turns a
session into a document.
"""

from __future__ import annotations

from .author import AuthorRequest
from .budget import Budget, BudgetExhausted, Spend
from .distil import Draft, distil
from .guardrails import DISTILS_TO, GuardContext, Guarded, PERCEPTION, REFUSED, guard, offered
from .marks import Described, Mark, Marks, describe_element
from .providers import (
    Availability,
    BrowserProvider,
    LocalPlaywrightMCP,
    MCPSession,
    StdioMCPProvider,
    ToolResult,
    ToolSpec,
    local_availability,
)
from .run import AgentSession, AuthorResult, run_agent_session
from .session import AgentToolSession, Recorder, ToolCallRecord
from .tools import TOOLS
from .verify import Verification, verify

__all__ = [
    "AgentToolSession",
    "Availability",
    "TOOLS",
    "AuthorRequest",
    "AuthorResult",
    "Budget",
    "BudgetExhausted",
    "Spend",
    "Draft",
    "Verification",
    "distil",
    "AgentSession",
    "run_agent_session",
    "verify",
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
    "StdioMCPProvider",
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
