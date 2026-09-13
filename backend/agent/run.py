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

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from events import ApprovalRequired, ApprovalResolved, RunFinished, RunStarted, Thinking
from llm import LLMClient
from prompt_loader import AUTHOR_TASK, render
from redaction import Redactor

from .author import AuthorRequest, Wiring
from .brief import (
    TaskBrief,
    apply_walkthrough,
    write_brief,
    write_walkthrough,
)
from .budget import Spend
from .distil import Draft, distil
from .guardrails import guard
from .providers.base import BrowserProvider
from .session import AgentToolSession, Recorder
from .tools.finish import NAME as FINISH
from .verify import Replayer, Verification, verify

log = logging.getLogger(__name__)

NEWLINE = "\n"


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
    #: The request, restated before the browser opened: goal, per-row values,
    #: what proves a row worked, and what the request did not say. None when
    #: no scribe was injected, which is every caller that had none before.
    brief: dict[str, Any] | None = None
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
            "brief": self.brief,
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
        healer: Any = None,
        scribe: Any = None,
        verify_draft: bool = True,
        name: str = "",
        extra: dict[str, BrowserProvider] | None = None,
    ) -> None:
        request.run_id = request.run_id or uuid.uuid4().hex
        self.request = request
        self.secrets = dict(secrets or {})
        self.replay = replay
        #: Mends a draft that does not replay, after which the draft is
        #: replayed again without it. None keeps the old single-pass
        #: behaviour, which is what a deployment with healing off should get.
        self.healer = healer
        #: Writes the brief before the browser opens and the walkthrough after
        #: the steps exist. Injected, like the healer, and for the same two
        #: reasons: a caller that passes none makes no extra model calls, and a
        #: deployment turns the passes off by not passing one.
        self.scribe = scribe
        #: Kept from `start` so the walkthrough pass can be shown what the
        #: recording was trying to achieve, not only what it did.
        self._brief: TaskBrief | None = None
        #: Whether to replay the finished draft once, cold. See
        #: `Settings.agent_verify_draft`: the replay spends no tokens, so what
        #: turning it off buys is wall-clock time, and what it costs is finding
        #: out on the first real run instead of before publishing.
        self.verify_draft = verify_draft
        self.name = name
        self.thread_id = thread_id or request.run_id
        self._checkpointer = checkpointer
        self._compiled: Any = None
        self._config: dict[str, Any] = {}
        self._state: dict[str, Any] = {"messages": []}
        #: Ids handed out for the approval(s) currently paused on, so `resume`
        #: can announce them resolved with the same ids a person was shown.
        #: Emitted and consumed entirely by this class -- never by a graph
        #: node -- so unlike the interrupt payload itself, minting a fresh one
        #: per real pause is safe: `_run` runs exactly once per actual pause.
        self._pending_approvals: list[str] = []
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
            extra=extra,
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
                "langchain is not installed here. The agent is an optional "
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
        from langchain_core.messages import HumanMessage

        await self.wiring.emit(
            RunStarted(
                run_id=self.request.run_id,
                seq=0,
                task=self.request.task,
                start_url=self.request.start_url,
                options={
                    "mode": "author",
                    "may_write": self.request.may_write,
                    "budget": self.request.budget.as_dict(),
                },
                tools=[spec.name for spec in self.tools.tools] + [FINISH],
            )
        )
        # Before the first snapshot, and before the model is asked to do
        # anything: the request restated as an outcome, the per-row values, and
        # what proves a row worked. See `brief.py` for why this is not a plan
        # of clicks.
        self._brief = await self._write_brief()
        task_message = render(
            AUTHOR_TASK,
            brief=self._brief.as_prompt() if self._brief else "",
            task=self.request.task,
            start_url=self.request.start_url,
            secrets=", ".join(self.request.secrets) or "(none)",
            sample=json.dumps(self.request.sample) if self.request.sample else "(none given)",
        )
        return await self._run({"messages": [HumanMessage(content=task_message)]})

    async def _write_brief(self) -> TaskBrief | None:
        """The pre-recording pass, or nothing at all.

        Emitted as prose on the run's own event stream rather than logged,
        because the point of naming what the request left unsaid is that a
        person reads it -- and they are watching this stream while the browser
        works. It arrives before the first tool call, which is the only moment
        at which correcting an assumption is cheap.
        """
        if self.scribe is None:
            return None
        brief = await write_brief(
            self.scribe,
            task=self.request.task,
            start_url=self.request.start_url,
            allowed_domains=self.request.allowed_domains,
            secrets=self.request.secrets,
            sample=self.request.sample,
        )
        if brief is None:
            return None
        self.wiring.spend.turn(brief.usage, getattr(self.scribe, "model", ""))
        await self.wiring.emit(
            Thinking(
                run_id=self.request.run_id,
                seq=0,
                step=0,
                text="Before opening the browser:" + NEWLINE + NEWLINE + brief.as_prompt(),
                done=True,
            )
        )
        return brief

    async def resume(self, decision: str) -> AuthorResult:
        """Continue after a person answered. Same browser, same checkpoint."""
        from langgraph.types import Command

        outcome = "approved" if decision == "approved" else "rejected"
        for approval_id in self._pending_approvals:
            await self.wiring.emit(
                ApprovalResolved(
                    run_id=self.request.run_id, seq=0,
                    step=self.wiring.spend.steps,
                    approval_id=approval_id, decision=outcome,
                )
            )
        decisions = [
            {"type": "approve" if outcome == "approved" else "reject"}
            for _ in self._pending_approvals
        ]
        self._pending_approvals = []
        return await self._run(Command(resume={"decisions": decisions}))

    async def _run(self, entry: Any) -> AuthorResult:
        # `astream` rather than `invoke` so a cancelled session stops between
        # nodes, and so the checkpoint after each node is written as the
        # session goes rather than in one write at the end.
        async for state in self._compiled.astream(
            entry, config=self._config, stream_mode="values"
        ):
            self._state = state

        requests = await _pending_action_requests(self._compiled, self._config)
        if requests:
            # Paused mid-recording, not over. Nothing to distil yet. Ids are
            # minted here, outside the graph, exactly once per real pause --
            # see `_pending_approvals`'s own comment for why that is safe.
            self._pending_approvals = [uuid.uuid4().hex for _ in requests]
            step = self.wiring.spend.steps
            first_category_str = ""
            for idx, (approval_id, request) in enumerate(zip(self._pending_approvals, requests)):
                name, args = request.get("name", ""), request.get("args", {})
                verdict = guard(name, args, self.tools.context())
                if idx == 0:
                    first_category_str = verdict.category
                await self.wiring.emit(
                    ApprovalRequired(
                        run_id=self.request.run_id, seq=0, step=step,
                        approval_id=approval_id, call_id="", name=name,
                        arguments=self.tools.redactor.structure(args),
                        reason=request.get("description") or "This looks irreversible.",
                        categories=[c for c in verdict.category.split(", ") if c],
                        expires_at="",
                    )
                )
            first = requests[0]
            awaiting = {
                "call": {"name": first.get("name", ""), "input": first.get("args", {})},
                # A joined string, matching the shape the dashboard has always
                # read this field as -- `ApprovalRequired.categories` above is
                # the list form, for anything that wants to iterate it.
                "categories": first_category_str,
                "pending_count": len(requests),
            }
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
            healer=self.healer,
            scribe=self.scribe,
            brief=self._brief,
            verify_draft=self.verify_draft,
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
    healer: Any = None,
    scribe: Any = None,
    verify_draft: bool = True,
    name: str = "",
    extra: dict[str, BrowserProvider] | None = None,
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
        healer=healer,
        scribe=scribe,
        verify_draft=verify_draft,
        name=name,
        extra=extra,
    ) as session:
        return await session.start()


async def _draft_and_verify(
    request: AuthorRequest,
    state: dict[str, Any],
    session: AgentToolSession,
    wiring: Wiring,
    *,
    name: str,
    replay: Replayer | None,
    secrets: dict[str, str],
    healer: Any = None,
    scribe: Any = None,
    brief: TaskBrief | None = None,
    verify_draft: bool = True,
) -> AuthorResult:
    """Turn the session into a draft, then prove the draft replays.

    Verification happens after the agent's browser is closed and opens its own,
    which is the point rather than an accident -- see verify.py. A use case that
    only works inside the session that recorded it is not a use case.
    """
    result = _result(request, state, session, wiring)
    # The one place this run announces it is over. `create_agent`'s own nodes
    # have no equivalent of the old `finish` node to attach this to, and
    # distillation/verification below can take real time (a cold-browser
    # replay), so this fires first rather than after -- exactly when the
    # graph itself is done, not when everything downstream of it is.
    await wiring.emit(
        RunFinished(
            run_id=request.run_id, seq=0,
            status="succeeded" if result.status in {"succeeded", "partial"} else "failed",
            steps=result.steps,
            duration_ms=int(wiring.spend.elapsed * 1000),
            summary=result.summary or None,
            result={"answer": result.summary, "data": None},
            error=result.stopped_by or None,
        )
    )
    draft: Draft = distil(
        session.calls,
        session.marks,
        name=name or (request.task[:80] or "Recorded by the agent"),
        task=request.task,
        start_url=request.start_url,
        allowed_domains=request.allowed_domains,
    )
    # Before the dump, and before verification: the walkthrough writes onto
    # the same object verification then copies, so the prose travels with the
    # mended definition instead of having to be stitched onto both.
    result.draft_warnings = list(draft.warnings)
    if scribe is not None:
        result.draft_warnings.extend(
            await _write_walkthrough(scribe, draft, wiring, request=request, brief=brief)
        )
    result.use_case = draft.use_case.model_dump(mode="json", by_alias=True)
    if brief is not None:
        result.brief = brief.as_dict()

    not_worth_it = _why_not_verify(result, draft, verify_draft=verify_draft)
    if not_worth_it:
        result.verification = Verification(ran=False, skipped=not_worth_it).as_dict()
        return result

    report: Verification = await verify(
        draft.use_case,
        draft.sample_inputs,
        secrets,
        replay=replay,
        healer=healer,
    )
    result.verification = report.as_dict()

    # The mended draft replaces the recorded one, because it is the version
    # that has been proved to run. Keeping the original and reporting the
    # repair separately would hand somebody a recording known not to work
    # alongside a note saying how to fix it -- which is the thing this whole
    # pass exists to stop doing.
    if report.patched is not None:
        result.use_case = report.patched

    if report.ran and (not report.ok or report.repairs):
        # Said on the draft as well as in the report, because this is the line
        # a reviewer reads first and it must not be somewhere else.
        result.draft_warnings.insert(0, report.as_text())
    return result


def _why_not_verify(
    result: AuthorResult,
    draft: Draft,
    *,
    verify_draft: bool,
) -> str:
    """Why replaying this draft would tell nobody anything, or "".

    Two cases, and the second is the one that was being paid for repeatedly.

    A session that **stopped early without finishing a record** -- out of
    budget, out of time, stopped by a person -- has already reported that it
    did not get through. Replaying what it managed spends a browser launch,
    and a whole flow's wall clock, to confirm the thing the session's own
    message says. Four real sessions in a row hit their token ceiling
    mid-record and each one was then replayed anyway.

    And the deployment may simply not want it. Off is a legitimate choice:
    what it buys is time, and what it costs is finding out on the first real
    run rather than before publishing. Saying which of the two happened
    matters, because "not verified" with no reason reads as a failure.
    """
    if not verify_draft:
        return (
            "verification is switched off for this deployment (AGENT_VERIFY_DRAFT). "
            "Nothing has replayed this draft, so the first real run is the first "
            "time anybody will know whether it works."
        )
    if result.stopped_by and not draft.rows_recorded:
        return (
            "the session stopped before it finished a record, so there is nothing "
            f"whole to replay: {result.stopped_by}"
        )
    return ""


async def _write_walkthrough(
    scribe: Any,
    draft: Draft,
    wiring: Wiring,
    *,
    request: AuthorRequest,
    brief: TaskBrief | None,
) -> list[str]:
    """Describe the recording in plain language, onto the draft itself.

    Runs here rather than on the review screen because this is the last moment
    the session's own context is still assembled -- the task, the brief, and a
    step list that has just been cut on the agent's declared boundaries. A
    person opening the draft tomorrow has the steps and nothing else.
    """
    walkthrough = await write_walkthrough(
        scribe,
        draft.use_case,
        task=request.task,
        brief=brief,
    )
    if walkthrough is None:
        return []
    wiring.spend.turn(walkthrough.usage, getattr(scribe, "model", ""))
    return apply_walkthrough(draft.use_case, walkthrough)


def _result(
    request: AuthorRequest,
    state: dict[str, Any],
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
        steps=wiring.spend.steps,
        spend=wiring.spend.as_dict(),
        trajectory=[call.as_dict() for call in session.calls],
        marks=session.marks.as_dicts(),
        unfinished=session.marks.unfinished(),
    )


async def _pending_action_requests(compiled: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    """What a suspended graph is waiting on, if it is waiting.

    `HumanInTheLoopMiddleware` batches every tool call one model turn needed
    approval for into a single interrupt -- `{"action_requests": [...],
    "review_configs": [...]}`, no id of its own attached to either. `AgentSession`
    is what turns each entry into this codebase's own `ApprovalRequired`
    event and its own minted id; this function only reads the payload back.
    """
    snapshot = await compiled.aget_state(config)
    if not snapshot.next:
        return []
    for task in snapshot.tasks:
        for pending in getattr(task, "interrupts", ()) or ():
            value = pending.value
            if isinstance(value, dict):
                return list(value.get("action_requests") or ())
    return []



__all__ = ["AgentSession", "AuthorResult", "run_agent_session"]
