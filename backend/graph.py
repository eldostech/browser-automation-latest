"""The agent loop as a LangGraph state graph.

Why a graph rather than a ``while`` loop
----------------------------------------
The loop itself was never the hard part -- it was about thirty lines. What
LangGraph adds is the thing a hand-written loop cannot have without a lot more
code: **a checkpointer**. Every superstep is persisted, so a run that dies
mid-flight has state to come back to. Before this, a backend restart could only
mark an in-flight run failed and throw its progress away.

What is deliberately still ours
-------------------------------
The graph orchestrates; it does not decide. The domain allowlist, secret
redaction, loop detection, retry policy, approval gating and the event stream
the dashboard renders all stay in :mod:`agent`, and the nodes below call into
them. Those are the parts a framework has no opinion about, and moving them
into node bodies would only scatter them.

Two nodes and one decision:

    think ──(tool calls?)──► act ──► think
      │
      └──(no)──► END

``think`` is one model turn. ``act`` runs every tool the turn asked for,
through the same gates the loop used. The conditional edge is the only control
flow, which is why the graph is worth reading rather than merely obeying.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

log = logging.getLogger(__name__)


class AgentState(TypedDict):
    """What flows between supersteps.

    ``messages`` uses LangGraph's ``add_messages`` reducer, so a node returns
    only what it adds and the framework handles appending and de-duplication by
    id -- which is most of what the old manual history juggling was doing.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    #: Steps taken. Carried in state so the budget survives a resume.
    step: int
    #: Set when the agent produced a final answer rather than a tool call.
    answer: str | None
    #: Set when a guardrail stopped the run: budget, deadline, or a loop.
    halted: str | None


#: Reasons a run stops that are not "the agent finished". Kept as data so the
#: caller can map them to the right failure kind without string matching.
BUDGET_EXHAUSTED = "budget_exhausted"
DEADLINE_EXCEEDED = "deadline_exceeded"


def build_agent_graph(
    *,
    think: Callable[[AgentState], Any],
    act: Callable[[AgentState], Any],
    should_continue: Callable[[AgentState], str],
    checkpointer: Any | None = None,
):
    """Wire the two nodes together and compile.

    The callables are supplied by :class:`agent.BrowserAgent` so the graph
    stays a description of control flow and nothing else -- no policy, no
    events, no MCP.
    """
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)

    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", should_continue, {"act": "act", END: END})
    graph.add_edge("act", "think")

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def initial_state(prompt: str) -> AgentState:
    return {
        "messages": [HumanMessage(content=prompt)],
        "step": 0,
        "answer": None,
        "halted": None,
    }


def tool_message(call_id: str, content: str, *, is_error: bool = False) -> ToolMessage:
    """A tool result in the shape LangGraph pairs back to its call."""
    return ToolMessage(
        content=content or "(the tool returned nothing)",
        tool_call_id=call_id,
        status="error" if is_error else "success",
    )


def last_ai_message(state: AgentState) -> AIMessage | None:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage):
            return message
    return None
