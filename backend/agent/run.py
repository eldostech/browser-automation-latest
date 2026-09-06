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


class AgentSession:
    """One authoring session's whole lifetime: browser, graph, checkpoint.

    An object rather than a function because of the interrupt. When the graph
    stops to ask a person, **the browser has to stay open**: the agent is three
    pages into a workflow and the refs it is holding belong to the page in
    front of it. Closing and reopening between ``start`` and ``resume`` would
    leave the resumed graph acting on a blank tab -- the opposite of what a
    durable checkpoint is for.

    So the caller owns the lifetime. :func:`run_agent_session` is the one-shot
    wrapper for a caller that never needs to resume.
    """

    def __init__(
        self,
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
    ) -> None:
        request.run_id = request.run_id or uuid.uuid4().hex
        self.request = request
        self.secrets = dict(secrets or {})
        self.replay = replay
        self.name = name
        self.thread_id = thread_id or request.run_id
        self._checkpointer = checkpointer
        self._compiled: Any = None
        self._config: dict[str, Any] = {}
        self._state: AuthorState = initial_state()
        # The redactor is built from the secret *values* before the tool
        # session exists: the existing invariant is that it is registered
        # before anything is emitted, and an agent makes that easier to break
        # than a replay does, because typing credentials into pages is most of
        # what it does.
        self.tools = AgentToolSession(
            provider,
            allowed_domains=request.allowed_domains,
            may_write=request.may_write,
            redactor=Redactor(self.secrets.values()),
            recorder=recorder,
            # The route from a bound credential to a typed character. The
            # model is told to type the literal `{{secret.slot}}`; this is
            # what turns that into the real value in the one call that
            # reaches a browser, and it is the whole reason a session can
            # sign in at all rather than fabricating "admin" / "password".
            secret_values=self.secrets,
        )
        self.wiring = Wiring(
            request=request,
            tools=self.tools,
            llm=llm,
            spend=Spend(budget=request.budget),
            emit=emit,
        )

    async def __aenter__(self) -> "AgentSession":
        from . import graph as graph_module

        if not graph_module.available():
            raise RuntimeError(
                "LangGraph is not installed here. The agent is an optional "
                "extra: pip install -r backend/requirements-agent.txt"
            )
        await self.tools.__aenter__()
        self._checkpointer = self._checkpointer or graph_module.memory_checkpointer()
        self._compiled = graph_module.build(self.wiring, self._checkpointer)
        self._config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": 400,
        }
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.tools.__aexit__(*exc)

    async def start(self) -> AuthorResult:
        return await self._run(initial_state())

    async def resume(self, decision: str) -> AuthorResult:
        """Continue after a person answered. Same browser, same checkpoint."""
        from langgraph.types import Command

        return await self._run(Command(resume={"decision": decision}))

    async def _run(self, entry: Any) -> AuthorResult:
        # `astream` rather than `invoke` so a cancelled session stops between
        # nodes, and so the checkpoint after each node is written as the
        # session goes rather than in one write at the end.
        async for state in self._compiled.astream(
            entry, config=self._config, stream_mode="values"
        ):
            self._state = state

        awaiting = await _awaiting(self._compiled, self._config)
        if awaiting:
            # Paused mid-recording, not over. Nothing to distil yet.
            return _result(
                self.request, self._state, self.tools, self.wiring, awaiting=awaiting
            )
        return await _draft_and_verify(
            self.request,
            self._state,
            self.tools,
            self.wiring,
            name=self.name,
            replay=self.replay,
            secrets=self.secrets,
        )


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
    """Run one session to completion, or to its budget, and close the browser.

    The one-shot form. A caller that has to answer an approval and carry on
    wants :class:`AgentSession`, which keeps the browser open across the pause.
    """
    async with AgentSession(
        request,
        llm=llm,
        provider=provider,
        emit=emit,
        recorder=recorder,
        secrets=secrets,
        checkpointer=checkpointer,
        thread_id=thread_id,
        replay=replay,
        name=name,
    ) as session:
        return await session.start()


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



__all__ = ["AgentSession", "AuthorResult", "run_agent_session"]
