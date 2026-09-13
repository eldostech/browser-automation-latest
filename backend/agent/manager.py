"""Agent sessions, held while they run.

The adapter between an HTTP request and :class:`agent.run.AgentSession`.
Everything application-shaped lives here -- the store, the event bus,
settings, the workspace scoping -- because the rest of the agent package
deliberately knows about none of it, and that is what keeps the eventual move
to another runtime a matter of packaging. It lives inside the package rather
than beside it for the same reason: nothing else in this codebase needs to
know an agent session's lifecycle exists without also knowing what an agent
session is.

**Sessions are in memory, like recordings.** `recorder.Recorder` holds its
sessions in a dict and the recordings router works entirely off that; an agent
session gets the same treatment for the same reason. What is *not* lost on a
restart is the part that matters: every tool call is an event, and events are
persisted as they happen, so the transcript survives even when the live session
does not. The design's ``agent_sessions`` table is deferred until something
actually queries across sessions -- building a query table before there is a
query is how a schema fills up with columns nobody reads.

What is lost on a restart is an in-flight session's draft. That is the same
trade the recorder already makes with an open codegen window, and it is
survivable because an authoring session is minutes rather than hours.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from events import AgentEvent
from lifecycle import RunLifecycle, Terminal
from store import Store, WorkspaceStore

from .author import AuthorRequest
from .budget import Budget
from .providers import BrowserProvider, LocalPlaywrightMCP, StdioMCPProvider, local_availability
from .run import AgentSession, AuthorResult

log = logging.getLogger(__name__)

#: How many locators verification may re-find before giving up on a draft.
#:
#: Small on purpose. One or two bad locators in a recording is ordinary -- the
#: agent acted on refs and described what it acted on, and description is the
#: lossy step. Ten is not a recording that needs mending, it is one that needs
#: redoing, and spending a model call on each to reach that conclusion is the
#: expensive way to find out.
VERIFY_REPAIR_ATTEMPTS = 4


class AgentUnavailable(RuntimeError):
    """This deployment cannot run an agent, and the message says why."""


@dataclass
class Session:
    """One live authoring session, as the API sees it."""

    id: str
    run_id: str
    workspace_id: str
    task: str
    start_url: str
    status: str = "running"
    owner_id: str | None = None
    owner_email: str = ""
    error: str = ""
    result: AuthorResult | None = None
    #: Set while the graph is suspended: the call a person is being asked about.
    awaiting: dict[str, Any] | None = None
    #: Which model drove this session, as "provider:model". Reported rather
    #: than inferred: the whole point of choosing one is being able to tell
    #: afterwards which one produced the recording you are looking at.
    model: str = ""
    _task: asyncio.Task | None = field(default=None, repr=False)
    _session: AgentSession | None = field(default=None, repr=False)
    _finished: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def summary(self) -> dict[str, Any]:
        """What the session screen polls for. Never the trajectory: that is on
        the event stream already, and sending it on every poll would send the
        whole recording again every two seconds."""
        result = self.result
        return {
            "id": self.id,
            "run_id": self.run_id,
            "task": self.task,
            "start_url": self.start_url,
            "status": self.status,
            "model": self.model,
            "error": self.error or None,
            "awaiting": self.awaiting,
            "summary": result.summary if result else "",
            "stopped_by": result.stopped_by if result else "",
            "spend": result.spend if result else {},
            "steps": result.steps if result else 0,
            "marks": result.marks if result else [],
            # What the request was taken to mean, before the browser opened.
            # On the summary as well as on the event stream, because the
            # session screen is reloadable and an assumption worth correcting
            # must not be a message that scrolled past.
            "brief": result.brief if result else None,
            "unfinished": result.unfinished if result else "",
            "use_case": result.use_case if result else None,
            "draft_warnings": result.draft_warnings if result else [],
            "verification": result.verification if result else {},
        }


class AgentSessions:
    """Starts sessions, holds them, and answers for them.

    Deliberately shaped like ``recorder.Recorder``: start, get, list, cancel.
    Two products that both produce a draft use case should not have two
    different session lifecycles, and a person moving between them should not
    have to learn a second set of words.
    """

    def __init__(
        self,
        store: Store,
        bus: Any,
        settings: Any,
        llm_factory: Any,
        replay: Any = None,
    ) -> None:
        self.store = store
        self.bus = bus
        self.settings = settings
        self.llm_factory = llm_factory
        #: How a draft gets verified. None means "open a browser and use the
        #: engine", which is what a single-machine install wants. It is an
        #: injection point because the design has verification running
        #: somewhere else eventually -- an agent on a managed runtime calling
        #: back into a replay worker -- and because a test should not have to
        #: start Chromium to check that the router saves a draft.
        self.replay = replay
        self._sessions: dict[str, Session] = {}

    @property
    def verify_draft(self) -> bool:
        """Whether a finished recording is replayed before anybody sees it.

        Read from settings each time rather than captured, so an operator who
        changes it does not have to restart to find out whether they meant it.
        """
        return bool(getattr(self.settings, "agent_verify_draft", True))

    # -- availability -------------------------------------------------------
    def check_available(self) -> None:
        """Raise with something a person can act on, rather than failing later.

        Three separate reasons, because they need three different fixes: the
        deployment said no, the extras are not installed, or there is no model
        configured. Collapsing them would tell an operator who enabled the
        agent that it is disabled.
        """
        if not getattr(self.settings, "agent_enabled", False):
            raise AgentUnavailable(
                "The agent is switched off in this deployment (AGENT_ENABLED)."
            )
        available = local_availability()
        if not available.available:
            raise AgentUnavailable(available.reason)
        if self.llm_factory is None:
            raise AgentUnavailable(
                "No model is configured, and an agent session needs one. "
                "Recording with codegen does not, and still works."
            )

    def provider(self, headless: bool | None = None) -> Any:
        """The agent's browser, per this deployment's configuration.

        `cdp` attaches to a browser somebody else runs -- a managed session --
        rather than launching one. It is a different argv and nothing else,
        which is the whole point of the provider being an interface.

        ``headless`` is a per-session override of the deployment default, for
        the same reason a replay has "Show the browser while it runs": whether
        a person wants to watch is a choice made when the session starts, not
        a thing the deployment should decide for everyone. It has no effect on
        `cdp` -- the browser there is somebody else's to configure, and this
        session did not start it.
        """
        endpoint = ""
        provider = getattr(self.settings, "agent_browser_provider", "local")
        if provider == "cdp":
            endpoint = getattr(self.settings, "agent_cdp_endpoint", "")
            if not endpoint:
                raise AgentUnavailable(
                    "AGENT_BROWSER_PROVIDER is 'cdp' but AGENT_CDP_ENDPOINT is "
                    "empty, so there is no browser to attach to."
                )
        return LocalPlaywrightMCP(
            headless=(
                getattr(self.settings, "agent_headless", True)
                if headless is None or provider == "cdp"
                else headless
            ),
            version=getattr(self.settings, "agent_mcp_version", None),
            cdp_endpoint=endpoint,
        )

    # -- lifecycle ----------------------------------------------------------
    async def start(
        self,
        *,
        task: str,
        start_url: str,
        allowed_domains: tuple[str, ...],
        workspace_id: str,
        may_write: bool = False,
        budget: Budget | None = None,
        secrets: dict[str, str] | None = None,
        sample: dict[str, Any] | None = None,
        name: str = "",
        owner_id: str | None = None,
        owner_email: str = "",
        headless: bool | None = None,
        model: Any = None,
    ) -> Session:
        self.check_available()
        budget = await self._within_the_ceiling(workspace_id, budget or Budget())
        session_id = uuid.uuid4().hex
        run_id = uuid.uuid4().hex

        record = Session(
            id=session_id,
            run_id=run_id,
            workspace_id=workspace_id,
            task=task,
            start_url=start_url,
            owner_id=owner_id,
            owner_email=owner_email,
        )
        self._sessions[session_id] = record

        request = AuthorRequest(
            task=task,
            start_url=start_url,
            allowed_domains=allowed_domains,
            secrets=tuple(sorted(secrets or {})),
            sample=dict(sample or {}),
            may_write=may_write,
            budget=budget,
            run_id=run_id,
            workspace_id=workspace_id,
        )
        record.model = _describe_model(model, self.llm_factory)
        record._task = asyncio.create_task(
            self._drive(record, request, dict(secrets or {}), name, headless, model)
        )
        return record

    def _healer_for(self, model: Any) -> Any:
        """A healer for verification, or None when nothing can heal.

        Its own small budget rather than the session's. A draft that needs
        three locators re-found is worth three calls; the session's remaining
        step allowance is about driving a browser and says nothing about that.
        """
        if self.llm_factory is None:
            return None
        try:
            from healing import HealingBudget, StepHealer
        except Exception:  # noqa: BLE001 - healing is part of the same extra
            return None
        return StepHealer(
            _client_for(self.llm_factory, model),
            HealingBudget(max_attempts=VERIFY_REPAIR_ATTEMPTS, max_tokens=40_000),
        )

    def _scribe_for(self, model: Any) -> Any:
        """The client for the brief and walkthrough passes, or None.

        Two calls per session against a session that makes dozens, so this is
        not budgeted separately the way the healer is -- both are charged to
        the session's own spend, because both are part of recording.
        """
        if self.llm_factory is None:
            return None
        return _client_for(self.llm_factory, model)

    async def _within_the_ceiling(self, workspace_id: str, budget: Budget) -> Budget:
        """Fold the workspace's monthly ceiling into this session's own budget.

        Rather than adding a second enforcement path. The session budget is
        already checked before every model call and every tool call, so
        lowering it to what is left in the month means one mechanism stops the
        session and one message says which limit bit.

        A workspace already at its ceiling is refused here instead, because
        starting a session with nothing to spend would open a browser, take a
        snapshot and stop -- which looks like a failure rather than a budget.
        """
        spend = await self.store.workspace(workspace_id).spend_this_month()
        remaining = spend.get("remaining_usd")
        if remaining is None:
            return budget
        if remaining <= 0:
            raise AgentUnavailable(
                f"This workspace has spent ${spend['usd']:.2f} of its "
                f"${spend['limit_usd']:.2f} monthly limit. An administrator can "
                "raise it, or it resets at the start of next month."
            )
        if budget.usd is None or budget.usd > remaining:
            log.info(
                "session budget lowered to the workspace ceiling",
                extra={"workspace_id": workspace_id, "remaining_usd": remaining},
            )
            return replace(budget, usd=round(remaining, 4))
        return budget

    async def _registered_providers(self, data: "WorkspaceStore") -> dict[str, BrowserProvider]:
        """This workspace's enabled MCP servers, resolved into providers.

        Resolved fresh per session rather than cached: a server disabled or
        edited between sessions must take effect on the next one without a
        restart, the same way a credential edit does.
        """
        rows = await data.list_tool_servers(enabled_only=True)
        providers: dict[str, BrowserProvider] = {}
        for row in rows:
            if row["transport"] != "stdio":
                # Only stdio exists today; a row saved by a future version
                # with a transport this one does not understand is skipped
                # rather than crashing every session in the workspace.
                log.warning(
                    "skipping a tool server with an unsupported transport",
                    extra={"server_name": row["name"], "transport": row["transport"]},
                )
                continue
            connection = row["connection"] or {}
            providers[row["name"]] = StdioMCPProvider(
                row["name"],
                connection.get("command", ""),
                tuple(connection.get("args") or ()),
                dict(connection.get("env") or {}),
            )
        return providers

    async def _drive(
        self,
        record: Session,
        request: AuthorRequest,
        secrets: dict[str, str],
        name: str,
        headless: bool | None,
        model: Any = None,
    ) -> None:
        """Run the session, writing its events where every run writes them."""
        data = self.store.workspace(record.workspace_id)
        try:
            await data.create_run(
                record.run_id,
                request.task,
                request.start_url,
                {"agent": True, "session_id": record.id},
                owner_id=record.owner_id,
                owner_email=record.owner_email,
            )
            async with RunLifecycle(
                record.run_id, data, self.bus, secrets=list(secrets.values())
            ) as run:
                async with AgentSession(
                    request,
                    # The model this session was started with. `llm_factory`
                    # is the pool, which answers for a choice; calling it bare
                    # would drive on the configured model while the dashboard
                    # said otherwise, which is the one thing a model picker
                    # must not do.
                    llm=_client_for(self.llm_factory, model),
                    provider=self.provider(headless),
                    emit=_stamped(run.sink),
                    secrets=secrets,
                    name=name,
                    replay=self.replay,
                    # Built from *this session's* model, so a draft is mended
                    # by whatever the person chose to record with rather than
                    # by the deployment default. Budgeted separately from the
                    # session: mending a recording is not what the session's
                    # steps were for.
                    healer=self._healer_for(model),
                    # The two passes that bracket the recording: the request
                    # restated before the browser opens, the flow written down
                    # after the steps exist. Same model the person chose to
                    # record with, because a brief in one model's words feeding
                    # another model's session is a translation nobody asked for.
                    scribe=self._scribe_for(model),
                    # A cold replay of the finished draft, unless the
                    # deployment has turned it off. It spends no tokens; what
                    # it costs is a browser launch and one pass through the
                    # flow, and what it buys is knowing the recording runs
                    # before somebody publishes it.
                    verify_draft=self.verify_draft,
                    extra=await self._registered_providers(data),
                ) as session:
                    record._session = session
                    result = await session.start()
                    # Flushed *before* the status changes, not after. `_absorb`
                    # is what makes the new status visible to a client, and
                    # nothing may be visible before the events explaining it
                    # are readable -- see `_flush`. Doing it the other way round
                    # also puts an await between the status becoming
                    # "awaiting_approval" and the decision future existing,
                    # which is a window for an answer to arrive and be refused.
                    await _flush(run.sink)
                    self._absorb(record, result)

                    # Hold the browser open while a person decides. The agent is
                    # mid-workflow and the refs it holds belong to the page in
                    # front of it, so this cannot become a poll that reopens.
                    while record.status == "awaiting_approval":
                        decision = await self._wait_for_decision(record)
                        result = await session.resume(decision)
                        await _flush(run.sink)
                        self._absorb(record, result)

                run.finish(
                    Terminal(
                        # "partial" is a budget stop that kept everything it
                        # did -- worth distilling, not a failure -- and
                        # `run.py`'s `_draft_and_verify` already announced it
                        # as "succeeded" for exactly that reason. Matching
                        # that rule here, rather than treating anything but a
                        # literal "succeeded" as failed, is what stops this
                        # run's terminal *row* disagreeing with the terminal
                        # *event* the session already emitted.
                        status="succeeded" if record.status in {"succeeded", "partial"} else "failed",
                        steps=result.steps,
                        summary=result.summary or None,
                        error=result.stopped_by or None,
                        tokens=int(result.spend.get("tokens") or 0),
                        cost_usd=float(result.spend.get("usd") or 0.0),
                        # The session already emitted this run's `run_finished`
                        # through this same sink -- every completed
                        # `session.start()`/`session.resume()` goes through
                        # it. Announcing again here, with a status computed by
                        # a different rule, is the second `run_finished` a
                        # single run produced.
                        announced=True,
                    )
                )
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.error = "Stopped."
            raise
        except Exception as exc:  # noqa: BLE001 - the session's own failure
            log.exception("agent session failed", extra={"session_id": record.id})
            record.status = "failed"
            record.error = str(exc)
        finally:
            record._session = None
            record._finished.set()

    def _absorb(self, record: Session, result: AuthorResult) -> None:
        record.result = result
        record.awaiting = result.awaiting
        record.status = result.status

    async def _wait_for_decision(self, record: Session) -> str:
        """Block until someone answers.

        A bare future rather than a poll, so the browser sits still and costs
        nothing while a person thinks. Cancelling the task cancels the await,
        which is what lets shutdown finish rather than wait for an answer that
        is never coming.
        """
        decision: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        record._decision = decision  # type: ignore[attr-defined]
        try:
            return await decision
        finally:
            record._decision = None  # type: ignore[attr-defined]

    async def decide(self, session_id: str, workspace_id: str, decision: str) -> bool:
        """Answer the question a suspended session is waiting on."""
        record = self.get(session_id, workspace_id)
        if record is None or record.status != "awaiting_approval":
            return False
        future = getattr(record, "_decision", None)
        if future is None or future.done():
            return False
        future.set_result(decision)
        return True

    async def cancel(self, session_id: str, workspace_id: str) -> bool:
        record = self.get(session_id, workspace_id)
        if record is None or record._task is None or record._task.done():
            return False
        record._task.cancel()
        return True

    # -- reading ------------------------------------------------------------
    def get(self, session_id: str, workspace_id: str) -> Session | None:
        """Scoped by workspace, always. An id is not an authorisation."""
        record = self._sessions.get(session_id)
        if record is None or record.workspace_id != workspace_id:
            return None
        return record

    def list(self, workspace_id: str) -> list[Session]:
        return [s for s in self._sessions.values() if s.workspace_id == workspace_id]

    def discard(self, session_id: str, workspace_id: str) -> None:
        if self.get(session_id, workspace_id) is not None:
            self._sessions.pop(session_id, None)

    async def shutdown(self) -> None:
        """Stop everything in flight, and **wait for it to actually stop**.

        Cancelling without awaiting is the bug this docstring exists to
        prevent. A cancelled session still has to run its `finally`: close the
        browser, and let ``RunLifecycle`` write the run's ending and return its
        database connection. Walking away at that moment leaks a connection out
        of a pool of ten, and the symptom is not an error -- it is the *next*
        thing that wants a connection waiting forever. That is exactly what it
        did, and it hung the test suite one file later rather than anywhere
        near here.
        """
        running = [
            record._task
            for record in self._sessions.values()
            if record._task is not None and not record._task.done()
        ]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        self._sessions.clear()


async def _flush(sink: Any) -> None:
    """Write out whatever this session's events are still buffered.

    Called every time the session's *observable* status changes, and that is
    the point: `record.status` is what a client polls, and it is set inside the
    run's lifecycle rather than at the end of it. Without this a session could
    report "awaiting_approval" -- or "succeeded" -- while the events explaining
    why had not reached the database, so the approval card a person was being
    asked to answer would not be in the run they were reading.

    A status must never outrun its own events. `RunLifecycle` enforces that at
    the end of a run; this enforces it at each of the points in between.
    """
    flush = getattr(sink, "flush", None)
    if flush is not None:
        await flush()


def _client_for(factory: Any, model: Any) -> Any:
    """The client for this session's chosen model, or the configured one."""
    chooser = getattr(factory, "for_choice", None)
    return chooser(model) if chooser is not None else factory()


def _describe_model(model: Any, factory: Any) -> str:
    """"provider:model" for whatever this session will actually drive on.

    Resolved at start rather than read off the client later, so the session
    screen can say which model is running before the first turn comes back.
    """
    if model is not None:
        return getattr(model, "describe", lambda: str(model))()
    default = getattr(factory, "default_choice", None)
    return default.describe() if default is not None else ""


def _stamped(sink: Any):
    """Give each event its sequence number on the way to the sink.

    The agent package emits events with ``seq=0`` because it has no sink and no
    opinion about ordering; the run's sink owns the counter, and that counter is
    the client's resume token. Stamping here is what lets an agent run appear on
    the same stream as a replay, with the same guarantees, and no second
    streaming path to keep working.
    """

    async def emit(event: AgentEvent) -> None:
        await sink.emit(event.model_copy(update={"seq": sink.reserve_seq()}))

    return emit


__all__ = ["AgentSessions", "AgentUnavailable", "Session"]
