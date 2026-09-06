"""The entry point: one function, three transports.

    async def run_agent_session(request, *, llm, provider, emit, ...) -> AuthorResult

Everything above this is an adapter. In-process it is called directly; in a
worker it is called by a job; on AgentCore it is called by the runtime's
``/invocations`` handler. That is the rule from the design -- *the agent
package imports nothing from FastAPI and nothing from store* -- and it is what
keeps the eventual port a matter of packaging rather than rewriting.

Two consequences visible here. Events go to a callable, not a database: the
caller decides what persistence means, and on AgentCore the caller is on
another machine. And the browser arrives as a ``BrowserProvider``, because the
agent must not assume it owns one.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from llm import LLMClient
from redaction import Redactor

from .author import AuthorRequest, AuthorState, Wiring, initial_state
from .budget import Spend
from .distil import Draft, distil
from .provider import BrowserProvider
from .session import AgentToolSession, Recorder
from .verify import Replayer, Verification, verify

log = logging.getLogger(__name__)


@dataclass
class AuthorResult:
    """What a session produced, whatever transport carried it."""

    run_id: str
    status: str
    summary: str = ""
    #: Set when the graph suspended for a human. The caller needs to know a
    #: person is needed, which is a fact about the session rather than an
    #: error -- so it is a field and not an exception.
    awaiting: dict[str, Any] | None = None
    stopped_by: str = ""
    steps: int = 0
    spend: dict[str, Any] = field(default_factory=dict)
    #: Ordered tool calls: the input to distillation, and the audit evidence.
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    #: What the agent declared about the shape of the use case.
    marks: list[dict[str, Any]] = field(default_factory=list)
    #: Set when the session cannot be distilled and why. Reported rather than
    #: discovered on the review screen with nothing to do about it.
    unfinished: str = ""
    #: The draft use case, as a document. None when the session produced
    #: nothing worth drafting.
    use_case: dict[str, Any] | None = None
    #: What the reviewer needs in order to judge it.
    draft_warnings: list[str] = field(default_factory=list)
    #: Whether the draft replays. The whole point: the agent does not get to
    #: claim it recorded something.
    verification: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "awaiting": self.awaiting,
            "summary": self.summary,
            "stopped_by": self.stopped_by,
            "steps": self.steps,
            "spend": self.spend,
            "trajectory": self.trajectory,
            "marks": self.marks,
            "unfinished": self.unfinished,
            "use_case": self.use_case,
            "draft_warnings": self.draft_warnings,
            "verification": self.verification,
        }


async def run_agent_session(
    request: AuthorRequest,
    *,
    llm: LLMClient,
    provider: BrowserProvider,
    emit: Any,
    recorder: Recorder | None = None,
    secrets: dict[str, str] | None = None,
    checkpointer: Any = None,
    thread_id: str = "",
    replay: Replayer | None = None,
    name: str = "",
) -> AuthorResult:
    """Run one authoring session to completion, or to its budget.

    The redactor is built from the run's secret *values* before the tool
    session exists, which is the existing invariant: registered before anything
    is emitted. An agent makes it easier to break than a replay does, because
    typing credentials into pages is most of what it does.
    """
    request.run_id = request.run_id or uuid.uuid4().hex
    redactor = Redactor((secrets or {}).values())

    session = AgentToolSession(
        provider,
        allowed_domains=request.allowed_domains,
        may_write=request.may_write,
        redactor=redactor,
        recorder=recorder,
    )

    async with session:
        wiring = Wiring(
            request=request,
            tools=session,
            llm=llm,
            spend=Spend(budget=request.budget),
            emit=emit,
        )
        state, awaiting = await _drive(
            wiring, checkpointer, thread_id or request.run_id
        )

    if awaiting:
        # Nothing to distil yet: the session is paused mid-recording, not over.
        return _result(request, state, session, wiring, awaiting=awaiting)

    return await _draft_and_verify(
        request, state, session, wiring, name=name, replay=replay, secrets=secrets or {}
    )


async def _draft_and_verify(
    request: AuthorRequest,
    state: AuthorState,
    session: AgentToolSession,
    wiring: Wiring,
    *,
    name: str,
    replay: Replayer | None,
    secrets: dict[str, str],
) -> AuthorResult:
    """Turn the session into a draft, then prove the draft replays.

    Verification happens after the agent's browser is closed and opens its own,
    which is the point rather than an accident -- see verify.py. A use case that
    only works inside the session that recorded it is not a use case.
    """
    result = _result(request, state, session, wiring)
    draft: Draft = distil(
        session.calls,
        session.marks,
        name=name or (request.task[:80] or "Recorded by the agent"),
        task=request.task,
        start_url=request.start_url,
        allowed_domains=request.allowed_domains,
    )
    result.use_case = draft.use_case.model_dump(mode="json", by_alias=True)
    result.draft_warnings = list(draft.warnings)

    report: Verification = await verify(
        draft.use_case, draft.sample_inputs, secrets, replay=replay
    )
    result.verification = report.as_dict()
    if report.ran and not report.ok:
        # Said on the draft as well as in the report, because this is the line
        # a reviewer reads first and it must not be somewhere else.
        result.draft_warnings.insert(0, report.as_text())
    return result


def _result(
    request: AuthorRequest,
    state: AuthorState,
    session: AgentToolSession,
    wiring: Wiring,
    *,
    awaiting: dict[str, Any] | None = None,
) -> AuthorResult:
    return AuthorResult(
        run_id=request.run_id,
        status="awaiting_approval" if awaiting else state.get("status", "failed"),
        awaiting=awaiting,
        summary=state.get("summary", ""),
        stopped_by=state.get("stopped_by", ""),
        steps=state.get("step", 0),
        spend=wiring.spend.as_dict(),
        trajectory=[call.as_dict() for call in session.calls],
        marks=session.marks.as_dicts(),
        unfinished=session.marks.unfinished(),
    )


async def _drive(
    wiring: Wiring, checkpointer: Any, thread_id: str
) -> tuple[AuthorState, dict[str, Any] | None]:
    """Run the graph, or say plainly that this deployment cannot.

    There is no fallback loop. A hand-rolled `while` beside the graph would be
    a second implementation of the control flow, drifting from the one that is
    actually shipped -- and the two would disagree exactly where it matters,
    on approval and on resume.
    """
    from . import graph as graph_module

    if not graph_module.available():
        raise RuntimeError(
            "LangGraph is not installed here. The agent is an optional extra: "
            "pip install -r backend/requirements-agent.txt"
        )

    compiled = graph_module.build(
        wiring, checkpointer or graph_module.memory_checkpointer()
    )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 400}
    # `astream` rather than `invoke` so a cancelled session stops between nodes
    # rather than only at the end, and so the checkpoint after each node is
    # written as the session goes rather than in one write at the finish.
    final: AuthorState = initial_state()
    async for state in compiled.astream(
        initial_state(), config=config, stream_mode="values"
    ):
        final = state

    # A graph that stopped for a human does not raise; it simply has somewhere
    # left to go. Reporting that as a status rather than an exception is what
    # lets a caller hold the session open and resume it with a decision -- see
    # `resume_agent_session` below.
    return final, await _awaiting(compiled, config)


async def _awaiting(compiled: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    """The question a suspended graph is waiting on, if it is waiting."""
    snapshot = await compiled.aget_state(config)
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        for pending in getattr(task, "interrupts", ()) or ():
            return dict(pending.value) if isinstance(pending.value, dict) else {
                "question": str(pending.value)
            }
    return None


async def resume_agent_session(
    request: AuthorRequest,
    decision: str,
    *,
    llm: LLMClient,
    provider: BrowserProvider,
    emit: Any,
    checkpointer: Any,
    recorder: Recorder | None = None,
    secrets: dict[str, str] | None = None,
    thread_id: str = "",
) -> AuthorResult:
    """Continue a session a person was asked about.

    The checkpointer must be the same one -- that is the whole mechanism. The
    browser is opened again because the old one belonged to the process that
    suspended, which is exactly the case a durable checkpoint exists for: the
    conversation and the marks survive, the live objects do not.
    """
    redactor = Redactor((secrets or {}).values())
    session = AgentToolSession(
        provider,
        allowed_domains=request.allowed_domains,
        may_write=request.may_write,
        redactor=redactor,
        recorder=recorder,
    )

    async with session:
        wiring = Wiring(
            request=request,
            tools=session,
            llm=llm,
            spend=Spend(budget=request.budget),
            emit=emit,
        )
        from . import graph as graph_module
        from langgraph.types import Command

        compiled = graph_module.build(wiring, checkpointer)
        config = {
            "configurable": {"thread_id": thread_id or request.run_id},
            "recursion_limit": 400,
        }
        final: AuthorState = initial_state()
        async for state in compiled.astream(
            Command(resume={"decision": decision}), config=config, stream_mode="values"
        ):
            final = state
        awaiting = await _awaiting(compiled, config)

    return AuthorResult(
        run_id=request.run_id,
        status="awaiting_approval" if awaiting else final.get("status", "failed"),
        awaiting=awaiting,
        summary=final.get("summary", ""),
        stopped_by=final.get("stopped_by", ""),
        steps=final.get("step", 0),
        spend=wiring.spend.as_dict(),
        trajectory=[call.as_dict() for call in session.calls],
        marks=session.marks.as_dicts(),
        unfinished=session.marks.unfinished(),
    )


__all__ = ["AuthorResult", "resume_agent_session", "run_agent_session"]
