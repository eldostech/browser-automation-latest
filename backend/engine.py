"""Executes a recorded use case. **No LLM, ever.**

This is ``replay.py`` with the browser layer swapped: the structure, the
recovery rules and the guarantees are the ones that module established, and
what changed underneath is that a step now calls Playwright directly instead of
sending JSON-RPC to a Node process that then calls Playwright.

The zero-token guarantee is unchanged and still structural. This module does
not import :mod:`llm`, and :class:`UseCaseExecutor` has no parameter that could
accept a model client. ``tests/test_engine.py`` asserts both. Threading a
no-LLM flag through an agent instead would have left the expensive path one bug
away from being re-entered.

What the direct route changed
-----------------------------
**The locator ladder is executed rather than approximated.** ``role`` rungs
used to be resolved by parsing a snapshot and handing back a ``ref=e17``
string. Now a rung becomes ``page.get_by_role(role, name=...)`` and Playwright
resolves it against the live page, waiting for the element to be actionable and
refusing ambiguity instead of silently taking the first match. The ladder still
leads with semantic rungs, still reports which one matched, and still treats
falling through as drift worth warning about.

**Waiting is the library's job.** The hand-written settle loop existed because
resolving refs from our own snapshots meant the waiting was ours to do. Actions
auto-wait now. What remains is a bounded retry of the *ladder*, for the case
where an element is genuinely not there yet.

**A snapshot is still free, and still useful.** ``locator.aria_snapshot()``
gives the same accessibility tree the MCP server returned, so assertions and
the failure context that a repair reads later are unchanged.

Setup versus rows
-----------------
:meth:`UseCaseExecutor.run_setup` runs once per browser session and
:meth:`UseCaseExecutor.run_row` once per input row, because a batch shares one
session and must not sign in a thousand times. A single-row execution is
``run_setup`` followed by one ``run_row``, so there is no second code path for
the batch runner to drift away from.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import imagediff
from browser import BrowserConfig, BrowserError, PlaywrightSession
from events import (
    AgentEvent,
    ErrorEvent,
    Screenshot,
    StepFinished,
    StepStarted,
    ToolCall,
    ToolResult,
)
from policy import check_navigation
from redaction import Redactor
from snapshot import Snapshot
from usecase import (
    render_code,
    OPTIONAL_LOCATOR_ACTIONS,
    Assertion,
    Locator,
    MissingValue,
    NAMED_STRATEGIES,
    Step,
    UseCase,
    render_template,
)

log = logging.getLogger(__name__)

Phase = Literal["setup", "row", "reset", "teardown"]

#: A ladder that matches nothing is retried for the whole step timeout, the
#: same budget Playwright would give a single locator.
#:
#: There used to be a five-second cap here whenever a weak rung existed to fall
#: back to, on the reasoning that a renamed element should not cost the full
#: timeout on every row of a batch. It is the wrong trade. "Not there yet" and
#: "not there any more" are indistinguishable without waiting -- no signal
#: separates them, since a page blocked on a slow request is as quiet as a
#: finished one -- so capping the wait turns every site slower than five
#: seconds into a hard failure that reads as a missing element. A slow batch is
#: recoverable; a batch that reports the wrong reason is not.
#:
#: The renamed case is cheaper than it looks anyway: the weak rung matches on
#: the first pass and returns immediately. Only a genuinely absent element
#: spends the budget. Raise REPLAY_STEP_TIMEOUT for a site slower than this.
POLL_INTERVAL = 0.25

#: How many matching elements the locator check names back. Enough to see
#: whether an ambiguous rung is finding the right kind of thing; not so many
#: that checking a rung matching a whole table reads the whole table.
PROBE_SAMPLES = 5

#: How much of the page to keep on a failure event. A repair reads this later,
#: and the interactive nodes it needs are near the top.
FAILURE_SNAPSHOT_CHARS = 8_000


class EventSink(Protocol):
    def reserve_seq(self) -> int: ...
    async def emit(self, event: AgentEvent) -> None: ...
    async def save_screenshot(
        self, data: bytes, *, seq: int, mime: str = "image/png"
    ) -> tuple[str, str] | None: ...

    async def save_download(
        self, data: bytes, *, seq: int, filename: str, mime: str
    ) -> tuple[str, str] | None:
        """Keep a downloaded file. Optional, like ``record_step``.

        A sink with no storage behind it -- a test's, or one used somewhere
        with no database -- simply does not have this, and the executor checks
        once rather than guarding every call.
        """
        raise NotImplementedError

    async def record_step(self, **fields: Any) -> None:
        """Persist one step as a row, for the timeline and the visual diff.

        Optional. A sink that only collects events -- a test's, or one used
        somewhere with no database behind it -- may simply not have this, and
        the executor checks once rather than guarding every call.
        """


class Healer(Protocol):
    """Repairs one broken step by looking at the page as it is now.

    Implemented where an LLM client is imported -- never here. A healer is
    injected only where healing is deliberately enabled, so with none passed
    there is no path to a model at all. That is what keeps "zero tokens"
    structural rather than configured.
    """

    async def repair(self, step: Step, snapshot: Any) -> Any: ...

    @property
    def tokens_used(self) -> int: ...

    async def confirm(self, repair: Any, worked: bool) -> None:
        """How that repair turned out, once the step has been retried.

        Optional, the way ``record_step`` is optional on the sink. It exists so
        that a healer which remembers fixes can record only the ones that
        worked -- which is knowable here and nowhere earlier -- without this
        module knowing that a memory exists at all.
        """


class StepFailed(RuntimeError):
    """A step did not succeed and its ``on_failure`` says to stop."""

    def __init__(self, step_id: str, message: str) -> None:
        super().__init__(message)
        self.step_id = step_id


class ScriptBlocked(RuntimeError):
    """A ``script`` step was reached without ``allow_scripts``."""


class NavigationBlocked(RuntimeError):
    """A step tried to leave the use case's allowed domains."""


@dataclass(slots=True)
class StepOutcome:
    step_id: str
    ok: bool
    duration_ms: int
    message: str = ""
    matched_locator: str | None = None
    locator_rung: int | None = None
    skipped: bool = False


@dataclass(slots=True)
class RowResult:
    """What one input row produced."""

    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    failed_step_id: str | None = None
    #: Where in ``row_steps`` it stopped, so a caller that fixes the page can
    #: carry on from there rather than from the top. Re-running a partial row
    #: from the beginning is how a form gets submitted twice.
    failed_index: int | None = None
    error: str | None = None
    duration_ms: int = 0
    steps: list[StepOutcome] = field(default_factory=list)
    #: Always 0 for a pure replay. Recorded so the dashboard can prove it.
    llm_calls: int = 0
    llm_tokens: int = 0
    #: What those calls cost. Zero on a Strict row by construction -- the
    #: engine cannot reach a model -- and non-zero on a Guided row that had to
    #: repair itself. Reported rather than assumed: "free" and "not measured"
    #: look identical on a dashboard, and until this existed every replay
    #: claimed to be free whether or not it had healed.
    llm_usd: float = 0.0
    #: Set by `agent/operate.py` when an agent recovery gave up on this row
    #: and could still produce a diagnosis -- an untyped `repair.PendingRepair`
    #: rather than importing that type here, so that this module (used by
    #: every replay, agent installed or not) never has to import anything
    #: from the agent's own optional dependency tree just to *hold* a value it
    #: never inspects. Never applied by anything that only sees a `RowResult`:
    #: turning it into a draft version is the caller's job, once, and only the
    #: caller (`runner.py`) has the store this needs. Not included in
    #: `to_dict()` deliberately -- nothing serializes a `RowResult` wholesale
    #: for storage; a caller that wants this reads the attribute directly.
    repair_proposal: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "outputs": self.outputs,
            "failed_step_id": self.failed_step_id,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "llm_usd": round(self.llm_usd, 4),
        }


class UseCaseExecutor:
    """Runs a :class:`UseCase` against a live browser.

    Note the constructor signature: there is no ``llm`` parameter, and there is
    no code path here that could use one.
    """

    #: Used when the caller names no timeout. Production always names one, from
    #: ``replay_step_timeout``; this is the fallback, and the single seam the
    #: test suite shortens so that a deliberate miss costs milliseconds rather
    #: than the production wait.
    DEFAULT_STEP_TIMEOUT = 30.0

    def __init__(
        self,
        usecase: UseCase,
        session: PlaywrightSession,
        sink: EventSink,
        *,
        run_id: str,
        secrets: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        redactor: Redactor | None = None,
        screenshots: str = "final",
        step_timeout: float | None = None,
        healer: "Healer | None" = None,
        baselines: dict[str, str] | None = None,
        read_artifact: Any = None,
    ) -> None:
        self.usecase = usecase
        self.browser = session
        self.sink = sink
        self.run_id = run_id
        self.secrets = dict(secrets or {})
        #: What this deployment answers ``{{env.x}}`` with. The schema has
        #: always matched that namespace and nothing ever filled it, so every
        #: such reference raised MissingValue at replay -- an advertised seam
        #: with no other side. See config.usecase_env.
        # The document's own origin is the default; the deployment overrides
        # it. That ordering is the whole promotion story: dev runs the
        # recording as recorded, and UAT answers with its own address without
        # the document differing by a byte.
        # Resolved once: a sink without storage behind it -- a test's -- simply
        # does not have this, and a download step says so rather than failing
        # per call. Same shape as `record_step`.
        self._save_download = getattr(sink, "save_download", None)
        self.env = {
            **({"base_url": usecase.base_url} if usecase.base_url else {}),
            **(env or {}),
        }
        self.redactor = redactor or Redactor(self.secrets.values())
        #: How much of a replay to photograph.
        #:
        #:   off        nothing, not even failures
        #:   failure    only where a step failed
        #:   final      one per row, showing the end state  (default)
        #:   every_step everything, for troubleshooting
        #:
        #: "final" answers "what happened to record 700" at one image per row.
        #: "every_step" multiplies that by the step count, which over a
        #: thousand-row batch is gigabytes.
        self.screenshots = screenshots
        self.step_timeout = (
            self.DEFAULT_STEP_TIMEOUT if step_timeout is None else step_timeout
        )
        #: Optional and off by default. See :class:`Healer`.
        self.healer = healer
        #: ``{step_id: screenshot_id}`` from the last run of this version that
        #: worked. Empty on the first run, which is why a diff is ``None``
        #: rather than zero when there is nothing to compare against.
        self.baselines = dict(baselines or {})
        #: Reads an artifact back, for the comparison. Left None -- as a test
        #: does -- steps are still recorded, just without a diff.
        self.read_artifact = read_artifact
        #: Which row of a batch is running, stamped onto each step row.
        self.row_index: int | None = None
        #: Where step rows go, or None for a sink that only collects events.
        #: Resolved once rather than guarded at every call site: recording a
        #: step is bookkeeping about work already done, and a sink that does
        #: not persist has nothing to record it into.
        self._record = getattr(sink, "record_step", None)

        self.step_number = 0
        #: The artifact the last capture produced, so the step row can point at
        #: the picture of itself.
        self._last_shot: str | None = None
        #: The bytes of that screenshot, still in hand.
        #:
        #: The visual diff used to fetch them back out of storage immediately
        #: after writing them -- a database row lookup and an object-store GET,
        #: per step, for bytes that had not left memory. Against S3 in another
        #: rack that was most of what an `every_step` run spent its time on.
        self._last_shot_bytes: bytes | None = None
        #: Baseline images already fetched, by artifact id. The baseline for a
        #: step is the *same image on every row*, so a thousand-row batch was
        #: fetching one identical object a thousand times.
        self._baseline_cache: dict[str, bytes | None] = {}
        self._last_snapshot: Snapshot | None = None
        self._last_snapshot_text = ""
        self._last_page_url: str | None = None
        self._last_node = None
        #: Rungs that matched more than one element on the last resolve, and
        #: how many. Read only to explain a failure.
        self._ambiguous: dict[str, int] = {}
        #: Playwright's own account of the last failed action: what it waited
        #: for, and what stopped it. The only place the cause of a timeout is
        #: written down -- see `_reason`.
        self._last_call_log: str = ""
        #: Rungs that matched exactly one element which is not the element the
        #: step was recorded against. Kept for the failure message: "no
        #: element matched" said of a rung that matched something else sends
        #: somebody looking for a locator problem rather than a page change.
        self._mismatched: dict[str, str] = {}
        #: The row being processed, for rendering a template into a locator, an
        #: assertion or a condition. Empty rather than unset: a session check
        #: and a setup step are evaluated outside any row, and reading this
        #: before the first row must give "no values" rather than raise.
        self._row_values: dict[str, Any] = {}
        #: Rungs deeper than the first, per step id. Surfaced as drift.
        self.locator_drift: dict[str, int] = {}
        #: Repairs accepted this session, for the version bump afterwards.
        self.healed: list[Any] = []

    # -- what this run has spent --------------------------------------------
    @property
    def spent(self) -> tuple[int, int, float]:
        """(calls, tokens, usd) for this session, cumulative.

        Read off the injected healer, which is the only object in reach that
        can spend anything -- and read by attribute rather than by importing
        it, because this module must not import ``llm`` and the healer is the
        thing that does. Zero when there is no healer, which is the Strict
        case, and it is a measured zero rather than an assumed one.
        """
        healer = self.healer
        if healer is None:
            return 0, 0, 0.0
        return (
            int(getattr(healer, "calls_used", 0) or 0),
            int(getattr(healer, "tokens_used", 0) or 0),
            float(getattr(healer, "usd_used", 0.0) or 0.0),
        )

    def _since(self, before: tuple[int, int, float]) -> dict[str, Any]:
        calls, tokens, usd = self.spent
        return {
            "llm_calls": calls - before[0],
            "llm_tokens": tokens - before[1],
            "llm_usd": usd - before[2],
        }

    # -- public API ---------------------------------------------------------
    async def run_setup(self) -> RowResult:
        """Run ``setup_steps`` once for this session. Signing in happens here."""
        started = time.monotonic()
        outcomes: list[StepOutcome] = []
        try:
            for step in self.usecase.setup_steps:
                outcomes.append(await self._run_step(step, {}, phase="setup"))
        except StepFailed as exc:
            return RowResult(
                ok=False,
                failed_step_id=exc.step_id,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
                steps=outcomes,
            )
        return RowResult(
            ok=True, duration_ms=int((time.monotonic() - started) * 1000), steps=outcomes
        )

    async def run_row(
        self,
        inputs: dict[str, Any],
        *,
        start_at: int = 0,
        outputs: dict[str, Any] | None = None,
    ) -> RowResult:
        """Run ``row_reset`` then ``row_steps`` for one input row.

        Never raises for an ordinary step failure -- the caller is a batch that
        must keep going. Only a dead browser propagates.

        ``start_at`` continues a row that stopped part way, for a caller that
        has put the page back where the next step expects it. It skips
        ``row_reset``, deliberately: a reset returns the browser to the start,
        which is the one thing a resume must not do. Re-running a partial row
        from the top is how a form gets submitted twice, so this is not an
        optimisation -- it is the only safe way to carry on at all.

        ``outputs`` carries forward what earlier steps already read, because a
        value extracted before the failure is not extracted again.
        """
        started = time.monotonic()
        # What the healer had spent before this row, so the row reports its own
        # cost rather than the session's running total.
        before = self.spent
        values = self.usecase.with_defaults(inputs)
        outputs = dict(outputs or {})
        outcomes: list[StepOutcome] = []

        missing = self.usecase.missing_inputs(values)
        if missing:
            return RowResult(
                ok=False,
                error=f"missing required input(s): {', '.join(missing)}",
                duration_ms=0,
            )

        index = start_at
        try:
            if self.usecase.row_reset is not None and start_at == 0:
                outcomes.append(
                    await self._run_step(self.usecase.row_reset, values, phase="reset")
                )
            for index, step in enumerate(
                self.usecase.row_steps[start_at:], start=start_at
            ):
                outcomes.append(
                    await self._run_step(step, values, phase="row", outputs=outputs)
                )
        except StepFailed as exc:
            return RowResult(
                ok=False,
                outputs=outputs,
                failed_step_id=exc.step_id,
                failed_index=index,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
                steps=outcomes,
                **self._since(before),
            )

        await self._capture_row(ok=True)
        return RowResult(
            ok=True,
            outputs=outputs,
            duration_ms=int((time.monotonic() - started) * 1000),
            steps=outcomes,
            **self._since(before),
        )

    async def run_teardown(self) -> RowResult:
        started = time.monotonic()
        outcomes: list[StepOutcome] = []
        try:
            for step in self.usecase.teardown_steps:
                outcomes.append(await self._run_step(step, {}, phase="teardown"))
        except StepFailed as exc:
            return RowResult(ok=False, failed_step_id=exc.step_id, error=str(exc), steps=outcomes)
        return RowResult(
            ok=True, duration_ms=int((time.monotonic() - started) * 1000), steps=outcomes
        )

    async def check_session(self) -> bool:
        """Is the shared session still authenticated?

        Cheap, and run between rows. A false answer tells the batch runner to
        sign in again rather than failing every remaining row.
        """
        if self.usecase.session_check is None:
            return True
        await self._refresh_snapshot()
        ok, _ = await self._evaluate(self._for_this_row(self.usecase.session_check))
        return ok

    # -- one step -----------------------------------------------------------
    async def _run_step(
        self,
        step: Step,
        values: dict[str, Any],
        *,
        phase: Phase,
        outputs: dict[str, Any] | None = None,
    ) -> StepOutcome:
        self.step_number += 1
        started = time.monotonic()
        self._row_values = values

        await self.sink.emit(
            StepStarted(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step_number,
                step_id=step.id,
                action=step.action,
                description=step.summary(),
                phase=phase,
            )
        )

        try:
            skip = await self._condition_says_skip(step)
            if skip is not None:
                outcome = skip
            else:
                outcome = await self._perform(
                    step, values, outputs if outputs is not None else {}
                )
        except (StepFailed, ScriptBlocked):
            raise
        except (BrowserError, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 - one step must not kill the batch
            outcome = StepOutcome(
                step_id=step.id, ok=False, duration_ms=0, message=_reason(exc)
            )

        # One repair attempt, only for a step that asked for it and only when a
        # healer was deliberately injected.
        if not outcome.ok and step.on_failure == "heal" and self.healer is not None:
            repaired = await self._heal(step, values, outputs if outputs is not None else {})
            if repaired is not None:
                outcome = repaired

        outcome.duration_ms = int((time.monotonic() - started) * 1000)

        await self.sink.emit(
            StepFinished(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step_number,
                step_id=step.id,
                ok=outcome.ok,
                duration_ms=outcome.duration_ms,
                matched_locator=outcome.matched_locator,
                locator_rung=outcome.locator_rung,
                assertion=step.assertion.describe() if step.assertion else None,
                message=self.redactor.text(outcome.message),
                skipped=outcome.skipped,
            )
        )

        if outcome.ok or outcome.skipped:
            await self._capture_step(step)
            await self._write_step_row(step, outcome, phase)
            return outcome

        # An optional step, or one told to continue, is a recorded failure that
        # does not stop the row.
        if step.optional or step.on_failure == "continue":
            outcome.skipped = True
            await self._write_step_row(step, outcome, phase)
            return outcome

        await self._record_failure_context(step.id, outcome.message)
        await self._capture_failure()
        await self._write_step_row(step, outcome, phase)
        raise StepFailed(step.id, f"step {step.id!r} ({step.summary()}) failed: {outcome.message}")

    async def _condition_says_skip(self, step: Step) -> StepOutcome | None:
        """``None`` to run the step, or the outcome recording that it was not.

        Evaluated **once**, with no retry loop, against the page as it stands.
        That is the opposite of how `assert` is evaluated and the difference is
        deliberate: an assertion is a claim that something *will* be true and
        is worth waiting for, while a condition asks what is on the page now.
        Retrying would make "the cookie banner is absent" cost the full
        timeout on every row of a batch -- the cheapest possible way to make a
        thousand rows slow.

        A skip is a success with a reason, not a failure. It is written to the
        step row and the event stream exactly like the skip an optional step
        produces, so the run view already knows how to show it.
        """
        if step.when is None:
            return None
        # The snapshot is refreshed after every action, so at step start it is
        # already the current page -- except for the very first step of a run,
        # which has never had one taken.
        if self._last_snapshot is None:
            await self._refresh_snapshot()
        check = self._for_this_row(step.when)
        held, detail = await self._evaluate(check)
        if held:
            return None
        return StepOutcome(
            step_id=step.id,
            ok=True,
            duration_ms=0,
            message=f"skipped: {check.describe()} did not hold{detail}",
            skipped=True,
        )

    async def _heal(
        self, step: Step, values: dict[str, Any], outputs: dict[str, Any]
    ) -> StepOutcome | None:
        """Ask the healer for a new locator, then retry the step once.

        The repaired locator is *prepended* to the ladder rather than replacing
        it, so a repair that turns out to be wrong degrades to what the
        recording already knew instead of losing it.
        """
        assert self.healer is not None
        # Settled, for the same reason the failure context is: the candidate
        # list the healer chooses from *is* this snapshot, and offering it a
        # half-rendered page is how a repair names a control that was on its
        # way out.
        await self.browser.settle()
        await self._refresh_snapshot()
        repair = await self.healer.repair(step, self._last_snapshot)
        if repair is None:
            return None

        await self._emit_error(
            "healed",
            f"step {step.id!r} was repaired: now looks for {repair.locator.describe()} "
            f"({repair.confidence} confidence). {repair.reason}",
            recoverable=True,
        )

        step.locators = [repair.locator, *[l for l in step.locators if l != repair.locator]]
        self.healed.append(repair)

        try:
            outcome = await self._perform(step, values, outputs)
        except Exception as exc:  # noqa: BLE001 - a failed repair is a failed row
            log.warning("retry after healing failed", extra={"step_id": step.id, "error": str(exc)})
            await self._confirm_repair(repair, False)
            return None

        # Whether the repair was right is knowable *here* and nowhere earlier,
        # so this is where the healer is told. A healer that keeps a memory
        # used to write the fix down when it proposed one, which recorded what
        # the model believed rather than what turned out to be true -- and a
        # confident wrong answer then came back as evidence every time that
        # site broke again.
        await self._confirm_repair(repair, outcome.ok)

        # The repair may have left a dialog open or the page part-way through
        # something, so put the browser back before the next row inherits it.
        if self.usecase.row_reset is not None:
            try:
                await self._do_navigate(self.usecase.row_reset, values)
            except Exception:  # noqa: BLE001 - best effort
                log.debug("row_reset after healing failed", extra={"step_id": step.id})

        return outcome if outcome.ok else None

    async def _confirm_repair(self, repair: Any, worked: bool) -> None:
        """Tell the healer how its repair turned out, if it wants to know.

        Optional on the protocol, the way ``record_step`` is optional on the
        sink: a healer with no memory behind it -- a test's, or one built
        without an embedder -- simply does not have this, and a healer that
        raises here must not turn a repaired row into a failed one.
        """
        confirm = getattr(self.healer, "confirm", None)
        if confirm is None:
            return
        try:
            await confirm(repair, worked)
        except Exception:  # noqa: BLE001 - bookkeeping, never the run
            log.debug("could not confirm a repair", exc_info=True)

    async def _emit_error(self, kind: str, message: str, *, recoverable: bool = False) -> None:
        await self.sink.emit(
            ErrorEvent(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step_number,
                kind=kind,
                message=self.redactor.text(message),
                recoverable=recoverable,
            )
        )

    async def _perform(
        self, step: Step, values: dict[str, Any], outputs: dict[str, Any]
    ) -> StepOutcome:
        # The one funnel every action passes through, and therefore the place
        # to say which row is being performed. `_resolve` needs it because a
        # locator may now name its element by a value from the row -- a search
        # whose dropdown is filled from the data has no other way to say which
        # suggestion it means. Set here rather than threaded through eight
        # signatures, and cleared nowhere: the next step overwrites it, and a
        # locator that references an input while no row is in hand fails loudly
        # through `_render` rather than resolving against stale values.
        self._row_values = values

        if step.action == "script" and not self.usecase.allow_scripts:
            raise ScriptBlocked(
                f"step {step.id!r} is raw JavaScript and this use case has allow_scripts "
                "turned off. Review the code and enable it deliberately."
            )

        if step.action == "assert":
            return await self._do_assert(step)
        if step.action == "wait":
            return await self._do_wait(step, values)
        if step.action == "navigate":
            return await self._do_navigate(step, values)
        if step.action == "fill_form":
            return await self._do_fill_form(step, values)
        if step.action == "extract":
            return await self._do_extract(step, outputs)
        if step.action == "extract_rows":
            return await self._do_extract_rows(step, outputs)
        if step.action == "download":
            return await self._do_download(step, outputs)
        if step.action == "script":
            return await self._do_script(step, values)
        return await self._do_element_action(step, values)

    # -- actions ------------------------------------------------------------
    async def _do_navigate(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        url = self._render(step.url or "", values)
        self._enforce_allowlist(url)

        async def go():
            await self.browser.page.goto(url, timeout=int(self.step_timeout * 1000))
            return f"navigated to {url}"

        ok, message, duration = await self._act("navigate", {"url": url}, step, go)
        if ok:
            await self._after_action()
        return StepOutcome(step.id, ok, duration, message)

    async def _do_element_action(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        """One action, through the first rung that can actually perform it.

        Resolving used to be the end of the ladder's job: the first rung
        matching one visible element won, the action ran against it, and a
        failure there failed the step. That is wrong for a whole class of page,
        and the class is common.

        A profile picker was recorded from the accessibility tree as
        `role=radio name="Nayra Asati"`. The rung resolves -- there is exactly
        one such radio -- and then `click` waits thirty seconds and gives up,
        because the site draws a styled radio whose input is not clickable and
        whose label is. The server had clicked the label; the tree knew only
        about the radio. Nothing was wrong with *finding* the element and the
        step failed anyway, twice, and a repair offered a differently spelled
        name because the error said "timeout" and named a locator.

        So an action that fails is a reason to try the next rung, not a reason
        to stop. It is safe: Playwright's actionability timeout means the
        action never dispatched, so there is nothing to have happened twice.
        """
        timeout = int(self.step_timeout * 1000)

        # `press` sends a key to whatever has focus and `upload` answers an open
        # file chooser; neither needs a target, and neither is recorded with
        # one. Every other element action does.
        if not step.locators and step.action in OPTIONAL_LOCATOR_ACTIONS:
            return await self._act_on(step, values, None, None, None, timeout)

        attempts: list[tuple[str, str]] = []
        tried: set[int] = set()
        while True:
            resolved = await self._resolve(
                step.locators, step.id, expect_text=step.expect_text, skip=tried
            )
            if resolved is None:
                break
            locator, rung, described = resolved
            tried.add(rung)
            # Each attempt gets a share of the step rather than the whole of
            # it. Three rungs at thirty seconds each is a ninety-second step,
            # which turns one slow failure into three.
            share = max(2_000, timeout // max(1, len(step.locators)))
            outcome = await self._act_on(step, values, locator, rung, described, share)
            if outcome.ok:
                if attempts:
                    log.info(
                        "a later rung performed what an earlier one could not",
                        extra={"step_id": step.id, "rung": rung, "tried": len(attempts)},
                    )
                return outcome
            attempts.append((described or "", outcome.message))

        if not attempts:
            return StepOutcome(step.id, False, 0, self._not_found_message(step))
        if len(attempts) == 1:
            described, message = attempts[0]
            return StepOutcome(step.id, False, 0, message, matched_locator=described)
        detail = "; ".join(f"{where}: {why}" for where, why in attempts)
        return StepOutcome(
            step.id,
            False,
            0,
            f"every recorded locator was found and none could be acted on -- {detail}",
            matched_locator=attempts[0][0],
        )

    async def _act_on(
        self,
        step: Step,
        values: dict[str, Any],
        locator: Any,
        rung: int | None,
        described: str | None,
        timeout: int,
    ) -> StepOutcome:
        """The action itself, against one already-resolved rung."""
        arguments: dict[str, Any] = {"target": described}
        if step.description:
            arguments["element"] = step.description
        value = self._render(step.value, values) if step.value is not None else None
        if value is not None:
            arguments["key" if step.action == "press" else "text"] = value

        async def perform():
            if step.action == "click":
                await locator.click(timeout=timeout)
            elif step.action == "fill":
                await locator.fill(value or "", timeout=timeout)
            elif step.action == "select":
                await locator.select_option(value or "", timeout=timeout)
            elif step.action == "hover":
                await locator.hover(timeout=timeout)
            elif step.action == "press":
                if locator is not None:
                    await locator.press(value or "", timeout=timeout)
                else:
                    await self.browser.page.keyboard.press(value or "")
            elif step.action == "upload":
                await locator.set_input_files([value or ""], timeout=timeout)
            else:  # pragma: no cover - the schema constrains `action`
                raise RuntimeError(f"unsupported action {step.action!r}")
            return f"{step.action} ok"

        ok, message, duration = await self._act(step.action, arguments, step, perform)
        if ok:
            await self._after_action()
        return StepOutcome(
            step.id, ok, duration, message, matched_locator=described, locator_rung=rung
        )

    async def _do_fill_form(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        """Fill every field of a form, one at a time.

        MCP had a ``browser_fill_form`` that took the whole form at once, and
        this degraded to per-field fills when it was missing. There is no
        batched equivalent in the library and there is no need for one: the
        round trip that made batching worthwhile is gone.
        """
        deepest = 0
        for item in step.fields:
            resolved = await self._resolve(item.locators, step.id)
            if resolved is None:
                return StepOutcome(
                    step.id, False, 0, f"could not find the {item.name!r} field on the page"
                )
            locator, rung, described = resolved
            deepest = max(deepest, rung)
            value = self._render(item.value, values)

            async def fill(locator=locator, value=value):
                await locator.fill(value, timeout=int(self.step_timeout * 1000))
                return "filled"

            ok, message, _ = await self._act(
                "fill", {"target": described, "element": item.name, "text": value}, step, fill
            )
            if not ok:
                return StepOutcome(
                    step.id, False, 0, f"filling {item.name!r} failed: {message}"
                )

        await self._after_action()
        return StepOutcome(
            step.id, True, 0, f"filled {len(step.fields)} field(s)", locator_rung=deepest
        )

    async def _do_wait(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        spec = step.wait_for
        if spec is None:
            return StepOutcome(step.id, True, 0, "nothing to wait for")

        timeout = int(self.step_timeout * 1000)
        try:
            if spec.kind == "time" and spec.seconds is not None:
                await asyncio.sleep(min(spec.seconds, self.step_timeout))
                return StepOutcome(step.id, True, 0, f"waited {spec.seconds}s")
            if spec.kind == "text" and spec.value:
                text = self._render(spec.value, values)
                await self.browser.page.get_by_text(text).first.wait_for(
                    state="visible", timeout=timeout
                )
                return StepOutcome(step.id, True, 0, f"{text!r} appeared")
            if spec.kind == "text_gone" and spec.value:
                text = self._render(spec.value, values)
                await self.browser.page.get_by_text(text).first.wait_for(
                    state="hidden", timeout=timeout
                )
                return StepOutcome(step.id, True, 0, f"{text!r} went away")
            await self.browser.page.wait_for_load_state("networkidle", timeout=timeout)
            return StepOutcome(step.id, True, 0, "page settled")
        except Exception as exc:  # noqa: BLE001
            return StepOutcome(step.id, False, 0, _reason(exc))
        finally:
            await self._refresh_snapshot()

    async def _do_assert(self, step: Step) -> StepOutcome:
        if step.assertion is None:
            return StepOutcome(step.id, True, 0, "no assertion")
        check = self._for_this_row(step.assertion)

        # If the allowlist makes this check impossible, say so instead of
        # waiting out the timeout and then blaming the page.
        impossible = check.unsatisfiable_reason(self.usecase.allowed_domains)
        if impossible:
            return StepOutcome(
                step.id,
                False,
                0,
                f"this assertion can never pass: it {impossible}. Fix the assertion rather "
                "than the page.",
            )

        deadline = time.monotonic() + max(check.timeout_ms, 0) / 1000.0
        detail = ""
        while True:
            await self._refresh_snapshot()
            ok, detail = await self._evaluate(check)
            if ok or time.monotonic() >= deadline:
                break
            await asyncio.sleep(POLL_INTERVAL)

        return StepOutcome(
            step.id,
            ok,
            0,
            f"{check.describe()} -- {'held' if ok else 'did not hold'}{detail}",
        )

    async def _do_download(self, step: Step, outputs: dict[str, Any]) -> StepOutcome:
        """Click something that yields a file, and keep the file.

        A download is a click with a consequence, not a kind of navigation:
        Playwright only surfaces one through ``expect_download`` around the
        action that triggers it, so the click and the capture cannot be
        separate steps.

        The file is stored exactly where screenshots and traces are -- locally
        or in S3, per deployment -- and the row's output records what it was
        called, how big it was, and the id to fetch it back by. That is what
        makes a migration's second half possible: the documents are addressable
        per record rather than sitting in a folder nobody can join to anything.

        The vendor's own filename is kept. It is the deliverable's identity, and
        the system it gets uploaded into will expect it.
        """
        resolved = await self._resolve(step.locators, step.id)
        if resolved is None:
            return StepOutcome(step.id, False, 0, self._not_found_message(step))
        locator, rung, described = resolved

        if self._save_download is None:
            return StepOutcome(
                step.id,
                False,
                0,
                "this deployment has nowhere to keep a downloaded file, so the step "
                "cannot run. Artifact storage is what holds them.",
            )

        started = time.monotonic()
        try:
            async with self.browser.page.expect_download(
                timeout=int(self.step_timeout * 1000)
            ) as info:
                await locator.click(timeout=int(self.step_timeout * 1000))
            download = await info.value
            path = await download.path()
            if path is None:
                # Playwright refused the download -- usually the context was
                # closing. Saying so beats an empty file that looks like a
                # document until somebody opens it.
                return StepOutcome(
                    step.id, False, 0, "the download did not complete"
                )
            data = Path(path).read_bytes()
            name = download.suggested_filename or f"{step.output or step.id}.bin"
        except Exception as exc:  # noqa: BLE001
            return StepOutcome(step.id, False, 0, _reason(exc))

        duration = int((time.monotonic() - started) * 1000)
        saved = await self._save_download(
            data,
            seq=self.step_number,
            filename=name,
            mime=_mime_for(name),
        )
        if saved is None:
            return StepOutcome(
                step.id, False, duration, f"{name!r} downloaded but could not be stored"
            )
        artifact_id, url = saved

        outputs[step.output or step.id] = {
            "filename": name,
            "artifact_id": artifact_id,
            "url": url,
            "bytes": len(data),
        }
        return StepOutcome(
            step.id,
            True,
            duration,
            f"downloaded {name!r} ({len(data)} bytes)",
            matched_locator=described,
            locator_rung=rung,
        )

    async def _do_extract_rows(
        self, step: Step, outputs: dict[str, Any]
    ) -> StepOutcome:
        """Read a list page into many rows -- the first pass of a migration.

        The row-driven model everything else uses assumes you already know the
        four thousand account numbers. Against a vendor who will not open their
        back end, you do not: the list page *is* the index, and this is how it
        becomes a dataset the detail pass can run against.

        Unlike every other step, matching more than one element is the point,
        so the strictness that protects the others is deliberately not applied
        here. Matching none is not an error either -- the last page of a
        paginated list is legitimately empty, and failing on it would break
        every crawl at its final step.
        """
        resolved = await self._resolve(step.locators, step.id, single=False)
        if resolved is None:
            # The rows may simply not be there yet, or there may be none. Both
            # are ordinary; an empty list says so without stopping the run.
            outputs[step.output or step.id] = []
            return StepOutcome(step.id, True, 0, "no rows matched")
        locator, rung, described = resolved

        try:
            rows = await locator.all()
        except Exception as exc:  # noqa: BLE001
            return StepOutcome(step.id, False, 0, _reason(exc))

        collected: list[dict[str, str]] = []
        for row in rows:
            record: dict[str, str] = {}
            for column in step.columns:
                try:
                    cell = row.locator(column.selector)
                    if column.attribute:
                        value = await cell.first.get_attribute(column.attribute)
                    else:
                        value = await cell.first.inner_text()
                    record[column.name] = (value or "").strip()
                except Exception:  # noqa: BLE001
                    # One missing cell must not lose the other columns of the
                    # row, nor the rest of the page. A blank is the honest
                    # answer and shows up in the dataset as one.
                    record[column.name] = ""
            collected.append(record)

        outputs[step.output or step.id] = collected
        return StepOutcome(
            step.id,
            True,
            0,
            f"extracted {len(collected)} row(s)",
            matched_locator=described,
            locator_rung=rung,
        )

    async def _do_extract(self, step: Step, outputs: dict[str, Any]) -> StepOutcome:
        resolved = await self._resolve(step.locators, step.id)
        if resolved is None:
            return StepOutcome(step.id, False, 0, self._not_found_message(step))
        locator, rung, described = resolved

        # The element's text, from the element -- rather than the accessible
        # name the snapshot happened to record for it, which is what the MCP
        # path had to settle for.
        try:
            if step.attribute:
                # The identifier a later pass needs is usually in the link
                # rather than in the words a person sees.
                raw = await locator.get_attribute(
                    step.attribute, timeout=int(self.step_timeout * 1000)
                )
                value = (raw or "").strip()
            else:
                value = (
                    await locator.inner_text(timeout=int(self.step_timeout * 1000))
                ).strip()
        except Exception as exc:  # noqa: BLE001
            return StepOutcome(step.id, False, 0, _reason(exc))

        outputs[step.output or step.id] = value
        return StepOutcome(
            step.id, True, 0, f"extracted {value!r}", matched_locator=described, locator_rung=rung
        )

    async def _do_script(self, step: Step, inputs: dict[str, Any] | None = None) -> StepOutcome:
        # Values are substituted as JSON literals rather than spliced into
        # source: a spreadsheet cell is untrusted input, and splicing it into
        # JavaScript is code injection.
        code = self._render_code(step, inputs or {})

        async def evaluate():
            return f"script returned {await self.browser.page.evaluate(code)!r}"[:400]

        ok, message, duration = await self._act("script", {"code": code}, step, evaluate)
        if ok:
            await self._after_action()
        return StepOutcome(step.id, ok, duration, message)

    def _render_code(self, step: Step, inputs: dict[str, Any]) -> str:
        try:
            return render_code(step.code or "", inputs=inputs, secrets=self.secrets)
        except MissingValue as exc:
            raise StepFailed(
                step.id,
                f"{exc.args[0]} was referenced by a script step but not supplied. "
                "Refusing to run it: a script with a missing value would either "
                "throw or silently submit nothing.",
            ) from exc

    # -- locator ladder -----------------------------------------------------
    async def locate(self, locators: list[Locator], step_id: str = "ad-hoc"):
        """Find an element the way a recorded step would.

        Public so that something outside a step can borrow the ladder -- the
        recovery agent, which is handed the same page mid-replay and must not
        get a second, weaker way of finding things. Sharing this is what makes
        its ambiguity refusal apply there too.
        """
        return await self._resolve(locators, step_id)

    async def _resolve(
        self,
        locators: list[Locator],
        step_id: str,
        *,
        single: bool = True,
        expect_text: str = "",
        skip: set[int] | None = None,
    ):
        """Walk the ladder and return ``(locator, rung, description)``.

        Semantic rungs lead, because ``role``/``label``/``placeholder``/
        ``alt_text`` describe what an element *means* and survive a redesign.
        Each becomes the Playwright call it was recorded as, which is the whole
        argument for recording them that way.

        A rung is accepted when it matches exactly one element. Zero is not
        there yet -- worth retrying, because a single-page app renders after it
        navigates. More than one is ambiguous, and picking the first silently
        is how a batch fills the wrong row of a table a thousand times; the
        recorded ``nth`` is how a recording says which one it meant.

        This used to accept any rung matching *at least* one and collapse it
        with ``.first``, which is the opposite of the paragraph above. It
        failed on a page with a "+ Invite User" button and an "Invite" button
        in the dialog that button opens: Playwright matches an accessible name
        as a substring, so the dialog's rung found both, took the one behind
        the dialog, and spent the whole step budget waiting for an element the
        dialog was covering. A thirty-second timeout was the only symptom.

        **Counting is done over what is visible.** See :meth:`_visible_only`
        for why that is not a refinement but a correction. A rung that matches
        one visible element is taken even when hidden duplicates exist, and the
        locator returned is the narrowed one, so the action lands on the
        element that was counted rather than on a sibling of it.

        A rung matching exactly one element that is *not* visible is held back
        rather than discarded. The page may still be rendering, so the loop
        keeps polling; if the deadline passes with nothing better, that rung is
        returned so Playwright raises its own actionability error, which says
        what the element is and why it could not be used. Failing with "no
        element matched" when one plainly did sends somebody looking for the
        wrong problem.
        """
        self._last_node = None
        self._ambiguous = {}
        self._mismatched = {}
        if not locators:
            return None

        # Rendered before anything looks at it, so everything downstream --
        # the ladder order, the match count, the failure message -- is about
        # the locator this row actually uses rather than about the template.
        # That is the answer to the standing objection to templating a
        # locator at all: "when it stops matching you cannot tell whether the
        # site changed or the input did." A failure names the rendered form.
        ordered = [
            pair
            for pair in self._rungs([self._render_locator(spec) for spec in locators])
            # Rungs an earlier attempt already resolved and failed to act on.
            # Walked by index rather than removed from the list, so the number
            # reported as drift stays the recording's own ordering.
            if not skip or pair[0] not in skip
        ]
        deadline = time.monotonic() + self.step_timeout
        #: The best rung that matched exactly one element nobody can see yet.
        held: tuple[Any, int, str] | None = None

        while True:
            held = None
            for rung, spec in ordered:
                locator = self._build(spec)
                if locator is None:
                    continue

                narrowed = self._visible_only(locator)
                try:
                    count = await (narrowed or locator).count()
                except Exception:  # noqa: BLE001 - mid-navigation; try again
                    continue

                # `extract_rows` passes single=False, because matching many is
                # the whole point there.
                if count == 1 or (count > 1 and not single):
                    if not await self._still_says_what_it_said(
                        narrowed or locator, spec, expect_text
                    ):
                        continue
                    if rung > 0:
                        self._note_drift(step_id, rung)
                    return (narrowed or locator), rung, spec.describe()
                if count > 1:
                    # Kept for the failure message. A step that dies saying
                    # "no element matched" when three of them did sends
                    # somebody looking for the wrong problem entirely.
                    self._ambiguous[spec.describe()] = count
                    continue

                if narrowed is None or held is not None:
                    continue
                # Nothing visible matched. If the DOM holds exactly one, keep
                # it as the answer of last resort and carry on looking.
                try:
                    if await locator.count() == 1:
                        held = (locator, rung, spec.describe())
                except Exception:  # noqa: BLE001 - mid-navigation; try again
                    continue

            if time.monotonic() >= deadline:
                if held is not None:
                    if held[1] > 0:
                        self._note_drift(step_id, held[1])
                    return held
                return None
            await asyncio.sleep(POLL_INTERVAL)

    async def _still_says_what_it_said(
        self, locator: Any, spec: Locator, expect_text: str
    ) -> bool:
        """Whether the element found is still the one that was recorded.

        Asked only of a rung that did **not** match on text. A rung that found
        its element by accessible name has already proved the wording; a CSS
        path, a test id or a bare role has proved only that something occupies
        that position, and those are the rungs that quietly land on a
        different control when a page is rebuilt. A step that "succeeds"
        against the wrong control is the worst outcome this system has: it is
        recorded as a success and does the wrong thing on every row after it.

        Borrowed from how locator caches elsewhere decide a cached path is
        still good -- check that the element there still says what it said --
        and narrowed to the case where it cannot produce a false refusal.

        Containment either way, not equality, and casefolded. A wrapper's text
        includes its children's, a button may have gained an icon's label or a
        count beside it, and a recorded name is often a trimmed version of
        what the DOM holds. Equality here would refuse correct steps, which is
        the failure this must not introduce.

        A read that raises, or an element with nothing to read, passes. The
        check exists to catch a control that is demonstrably something else,
        not to add a second way for a step to fail.
        """
        if not expect_text or spec.matches_on_text:
            return True
        try:
            found = (await locator.first.inner_text(timeout=1_000) or "").strip()
        except Exception:  # noqa: BLE001 - see the docstring
            return True
        if not found:
            return True

        wanted = expect_text.strip().casefold()
        seen = " ".join(found.split()).casefold()
        if wanted in seen or seen in wanted:
            return True
        self._mismatched[spec.describe()] = found[:120]
        log.info(
            "a positional rung matched something else",
            extra={"rung": spec.describe(), "recorded": expect_text, "found": found[:120]},
        )
        return False

    def _rungs(self, locators: list[Locator]) -> list[tuple[int, Locator]]:
        """The ladder in the order it is walked, semantic rungs first.

        A named rung that was *not* recorded as exact is walked twice: once
        requiring the whole accessible name, then as recorded. Playwright's
        default is a case-insensitive substring, so "Invite" also finds "+
        Invite User" -- and a recording made before ``exact`` was carried
        through the parser has no way left to say which was meant.

        The narrowed twin is not a guess about the page. It is the same
        recorded name, read more strictly, and it is taken only when it matches
        exactly one element; when it matches nothing the rung as recorded is
        tried immediately after. It shares its parent's rung number, so
        resolving through it is not reported as drift.
        """
        expanded: list[tuple[int, Locator]] = []
        for index, spec in enumerate(locators):
            named = spec.name if spec.strategy == "role" else spec.text
            if named and not spec.exact and spec.strategy in NAMED_STRATEGIES:
                expanded.append((index, spec.model_copy(update={"exact": True})))
            expanded.append((index, spec))
        # Semantic before markup, and within each of those, scoped before
        # unscoped. A rung that says *where* to look was the recording being
        # more specific, and specificity is what turns an ambiguous page into
        # a resolvable one -- so trying it first is the point of recording it.
        # A stable sort, so rungs that tie stay in the order the ladder lists.
        return sorted(expanded, key=lambda pair: (not pair[1].semantic, not pair[1].scoped))

    def _build(self, spec: Locator):
        """One rung as a Playwright locator, against the page this run is on."""
        return build_locator(self.browser.page, spec)

    @staticmethod
    def _visible_only(locator):
        return visible_only(locator)

    def _note_drift(self, step_id: str, rung: int) -> None:
        """Record that the preferred locator no longer matched.

        A use case that starts falling through to later rungs is drifting, and
        saying so early is the difference between a warning and a broken batch.
        """
        self.locator_drift[step_id] = max(self.locator_drift.get(step_id, 0), rung)
        log.info("locator fell through", extra={"step_id": step_id, "rung": rung})

    def _not_found_message(self, step: Step) -> str:
        if self._mismatched:
            found = "; ".join(
                f"{described} now holds {text!r}" for described, text in self._mismatched.items()
            )
            return (
                f"the element found is not the one that was recorded: {found}, but this step "
                f"was recorded against {step.expect_text!r}. A locator that says only where "
                "to look has landed on something else, which is how a batch acts on the "
                "wrong control. Re-record this step, or repair it to name what it wants."
            )
        if self._ambiguous:
            # Naming the count is most of the fix: two matches on a name means
            # the page holds a second control whose name contains this one, and
            # re-recording that step gets an exact locator written for it.
            found = "; ".join(
                f"{described} matched {count}" for described, count in self._ambiguous.items()
            )
            return (
                f"the locator is ambiguous: {found}. The recording does not say which one "
                "was meant, and clicking whichever comes first is how a batch acts on the "
                "wrong element. Re-record this step, or repair it to pick the right control."
            )
        tried = (
            "; ".join(self._render_locator(loc).describe() for loc in step.locators)
            or "(no locators recorded)"
        )
        available = ""
        if self._last_snapshot is not None:
            roles = self._last_snapshot.roles()
            top = ", ".join(f"{k}x{v}" for k, v in sorted(roles.items(), key=lambda kv: -kv[1])[:6])
            if top:
                available = f" The page has: {top}."
        return f"no element matched. Tried: {tried}.{available}"

    # -- assertions ---------------------------------------------------------
    def _for_this_row(self, check: Assertion) -> Assertion:
        """One check with its templates made real for the row in hand.

        At the edge rather than inside `_evaluate`, so everything downstream --
        the evaluation, the locator it builds, and the message a failure
        carries -- is about what this row actually looked for. That is the same
        rule `_resolve` follows for a locator rung, and the same reason: "it
        did not hold" printed against a template tells nobody whether the site
        changed or the input did.
        """
        return check.render(lambda value: self._render(value, self._row_values))

    async def _evaluate(self, check: Assertion) -> tuple[bool, str]:
        snapshot = self._last_snapshot
        value = check.value or ""

        if check.kind == "url_contains":
            actual = self._last_page_url or (snapshot.page_url if snapshot else "") or ""
            held = value in actual
            detail = f" (url is {actual!r})" if not held or check.negate else ""
        elif check.kind == "title_contains":
            actual = (snapshot.page_title if snapshot else "") or ""
            held = value in actual
            detail = f" (title is {actual!r})" if not held or check.negate else ""
        elif check.kind == "text_present":
            held = value.casefold() in self._last_snapshot_text.casefold()
            detail = ""
        elif check.kind == "element_visible":
            held = await self._visible(check.locator)
            detail = ""
        elif check.kind == "element_count":
            found = await self._count(check.locator)
            held = found == (check.count or 0)
            detail = f" (found {found})" if not held else ""
        elif check.kind == "attribute_contains":
            actual = await self._attribute(check.locator, check.attribute)
            held = actual is not None and value in actual
            # The distinction matters to whoever reads the failure: an element
            # that was not there at all is a different problem from one whose
            # href changed, and "does not contain" said of nothing is the
            # message that sends somebody to check the wrong thing.
            if actual is None:
                detail = " (no element matched, so the attribute could not be read)"
            elif not held or check.negate:
                detail = f" ({check.attribute} is {actual!r})"
            else:
                detail = ""
        else:  # pragma: no cover - the schema constrains `kind`
            return False, f" (unknown assertion kind {check.kind!r})"

        return (not held if check.negate else held), detail

    async def _visible(self, spec: Locator | None) -> bool:
        """Is this element on the page, right now?

        Asked of the live page rather than of the snapshot, and every strategy
        is answered the same way. Evaluating from the snapshot could only
        resolve the rungs whose value happens to be an accessible name, which
        silently made an ``element_visible`` assertion on a ``text`` or ``css``
        rung *unable to hold* -- a check that always fails is worse than no
        check, because it fails a row that actually worked.

        ``count`` and ``is_visible`` do not auto-wait, which is what makes this
        a check rather than a wait. ``_do_assert`` owns the retry loop and its
        timeout.
        """
        locator = self._build(spec) if spec is not None else None
        if locator is None:
            return False
        try:
            if await locator.count() == 0:
                return False
            return await locator.first.is_visible()
        except Exception:  # noqa: BLE001 - mid-navigation reads as "not there"
            return False

    async def _attribute(self, spec: Locator | None, name: str) -> str | None:
        """One attribute of the first match, or ``None`` if there is no match.

        ``None`` rather than the empty string, because an attribute that is
        absent and an element that is absent are different failures and an
        empty string cannot tell them apart.
        """
        locator = self._build(spec) if spec is not None else None
        if locator is None or not name:
            return None
        try:
            if await locator.count() == 0:
                return None
            return await locator.first.get_attribute(name) or ""
        except Exception:  # noqa: BLE001 - mid-navigation reads as "not there"
            return None

    async def _count(self, spec: Locator | None) -> int:
        """How many elements this locator matches, for ``element_count``."""
        locator = self._build(spec) if spec is not None else None
        if locator is None:
            return 0
        try:
            return await locator.count()
        except Exception:  # noqa: BLE001
            return 0

    async def _act(self, name: str, arguments: dict[str, Any], step: Step, action):
        """Perform one browser action, emitted as a tool call.

        There are no "tools" any more -- Playwright is called directly -- but
        the *events* are worth keeping exactly as they were. They are what the
        timeline, the WebSocket replay and the run view are built on, so a
        replay renders with no frontend change at all. They are also where a
        typed value passes through the redactor on its way to the event log,
        which is the mechanism that keeps a credential out of history.

        The name is the action rather than an MCP tool, which is what it always
        described.
        """
        call_id = f"{step.id}-{uuid.uuid4().hex[:6]}"
        await self.sink.emit(
            ToolCall(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step_number,
                call_id=call_id,
                name=name,
                arguments=self.redactor.structure(arguments),
            )
        )

        started = time.monotonic()
        ok, message = True, ""
        try:
            message = (await action()) or ""
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            ok, message = False, _reason(exc)
            # Held for `_record_failure_context`, which runs after the outcome
            # has travelled back up through three functions that have no reason
            # to carry a diagnostic string.
            self._last_call_log = call_log_of(exc)

        duration = int((time.monotonic() - started) * 1000)
        await self.sink.emit(
            ToolResult(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step_number,
                call_id=call_id,
                name=name,
                ok=ok,
                duration_ms=duration,
                text=self.redactor.text(message[:4000]),
                attempts=1,
            )
        )
        return ok, message, duration

    async def _write_step_row(self, step: Step, outcome: StepOutcome, phase: Phase) -> None:
        """Record what this step did, as a row.

        Beside the event, not instead of it: the event log is what the live
        timeline streams, and this is what the finished timeline and the visual
        diff query. Screenshot ids are carried across so the two pictures of a
        step -- this run's and the baseline's -- can be shown side by side
        without decoding anything at render time.
        """
        if self._record is None:
            return

        status = "skipped" if outcome.skipped else ("succeeded" if outcome.ok else "failed")
        if outcome.ok and any(h for h in self.healed if getattr(h, "step_id", None) == step.id):
            status = "healed"

        shot = self._last_shot
        shot_bytes = self._last_shot_bytes
        self._last_shot = None
        self._last_shot_bytes = None
        baseline = self.baselines.get(step.id)
        diff = await self._compare(baseline, shot, shot_bytes)

        await self._record(
            run_id=self.run_id,
            usecase_id=self.usecase.id,
            version=self.usecase.version,
            row_index=self.row_index,
            seq=self.step_number,
            step_id=step.id,
            phase=phase,
            action=step.action,
            locator=outcome.matched_locator or "",
            locator_rung=outcome.locator_rung,
            page_url=self._last_page_url or "",
            status=status,
            duration_ms=outcome.duration_ms,
            error=self.redactor.text(outcome.message) if not outcome.ok else None,
            screenshot_id=shot,
            baseline_id=baseline,
            pixel_diff=diff,
        )

    async def _compare(
        self, baseline: str | None, shot: str | None, shot_bytes: bytes | None = None
    ) -> float | None:
        """How much this step's page differs from the last time it worked.

        Neither image is fetched if it does not have to be. The *after* side is
        the screenshot this step just took and still holds, and the *before*
        side is the same object on every row of a batch, so it is read once and
        kept. Both used to be read back from storage on every step, which on a
        remote object store is two round trips per step to learn something
        about bytes that were already in memory.
        """
        if not baseline or not shot:
            return None
        after = shot_bytes
        if after is None:
            if self.read_artifact is None:
                return None
            try:
                after = await self.read_artifact(shot)
            except Exception:  # noqa: BLE001 - a diff must never fail a row
                return None
        before = await self._baseline_bytes(baseline)
        if before is None or after is None:
            return None
        return imagediff.ratio(before, after)

    async def _baseline_bytes(self, artifact_id: str) -> bytes | None:
        """The baseline image, fetched at most once per run.

        A failed read is cached too, as ``None``: a baseline that cannot be
        read will not become readable on row 700, and retrying it every row is
        a round trip spent to fail again.
        """
        if artifact_id in self._baseline_cache:
            return self._baseline_cache[artifact_id]
        data: bytes | None = None
        if self.read_artifact is not None:
            try:
                data = await self.read_artifact(artifact_id)
            except Exception:  # noqa: BLE001 - a diff must never fail a row
                data = None
        self._baseline_cache[artifact_id] = data
        return data

    # -- browser plumbing ---------------------------------------------------
    async def _refresh_snapshot(self) -> None:
        """Read the page, for assertions and for the failure context.

        Free here in a way it never was for the agent: a snapshot is expensive
        only when it enters a model's context window, and there is no model.
        """
        self._last_snapshot = await self.browser.snapshot()
        self._last_page_url = self._last_snapshot.page_url or self.browser.url
        self._last_snapshot_text = await self.browser.text_content()

    async def _after_action(self) -> None:
        """Let the page finish what the action started, then look at it.

        Three things in one place because they are one thing: an action can
        leave the browser somewhere other than where it found it, and every
        observation between here and the next action is wrong until that has
        resolved.

        The order is forced. Settle first, because a page mid-navigation
        reports neither its old state nor its new one. Adopt second, because a
        click that opened a tab means the page worth reading is not the page
        the click happened on. Snapshot last, because it is the thing the other
        two exist to make truthful.
        """
        await self.browser.settle()
        opened = await self.browser.adopt_new_page()
        if opened:
            # Worth an event: a run that silently changed which page it was
            # driving is the hardest kind of failure to read afterwards.
            await self._emit_error(
                "tab_opened",
                f"the site opened a new tab and the run followed it to {opened}",
                recoverable=True,
            )
            await self.browser.settle()
        await self._refresh_snapshot()

    async def _record_failure_context(self, step_id: str, message: str) -> None:
        """Persist the page as it was when a step failed.

        Repairing a use case later needs to know what was actually on screen,
        and the snapshots taken while resolving locators are internal -- they
        never reach the event log. Recording one here means a repair can be
        proposed from history alone, with no second browser session.

        Settled first, and that is the whole reason :meth:`settle` exists. This
        snapshot is not a diagnostic nicety: it is what a repair is proposed
        from and what goes into healing memory to be recalled the next time
        something breaks on this site. Capturing a page mid-navigation writes
        something false into that memory, and a false memory is recalled
        forever.
        """
        await self.browser.settle()
        await self._refresh_snapshot()
        # The page as text, not as a rendering of the parsed nodes: a repair
        # parses this again, and anything but the original shape gives it
        # nothing to work with.
        aria = self._last_snapshot.raw if self._last_snapshot is not None else ""
        await self.sink.emit(
            ErrorEvent(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step_number,
                kind="step_failed",
                message=self.redactor.text(message),
                recoverable=True,
                detail={
                    "step_id": step_id,
                    "page_url": self._last_page_url,
                    # Capped: a huge page must not bloat the event log, and the
                    # interactive nodes a repair needs are near the top.
                    "snapshot": self.redactor.text(aria[:FAILURE_SNAPSHOT_CHARS]),
                    # What Playwright tried and what stopped it. A repair
                    # proposed without this answers a timeout by re-spelling
                    # the locator, because a locator is the only thing the
                    # message it was given mentions.
                    "call_log": self.redactor.text(self._last_call_log),
                },
            )
        )

    async def _capture_failure(self) -> str | None:
        if self.screenshots == "off":
            return None
        return await self._capture("failure")

    async def _capture_step(self, step: Step) -> None:
        if self.screenshots == "every_step":
            await self._capture(f"after {step.id}")

    async def _capture_row(self, ok: bool) -> None:
        """Screenshot the end state of a row.

        This is the audit artifact: one image per record showing what the page
        looked like when the work finished. A failure is captured separately at
        the point it happened, which is more useful than the state afterwards.
        """
        if self.screenshots in ("final", "every_step") and ok:
            await self._capture("row finished")

    async def _capture(self, caption: str) -> str | None:
        """Take one screenshot. Never fails the row.

        A screenshot is a diagnostic aid; the row is the work. Every failure
        path here returns None rather than raising -- losing an image must not
        turn a successful record into a failed one.
        """
        data = await self.browser.screenshot()
        if not data:
            return None
        seq = self.sink.reserve_seq()
        saved = await self.sink.save_screenshot(data, seq=seq, mime="image/png")
        if saved is None:
            return None
        artifact_id, url = saved
        self._last_shot = artifact_id
        self._last_shot_bytes = data
        await self.sink.emit(
            Screenshot(
                run_id=self.run_id,
                seq=seq,
                step=self.step_number,
                artifact_id=artifact_id,
                url=url,
                caption=caption,
                page_url=self._last_page_url,
            )
        )
        return artifact_id

    # -- values -------------------------------------------------------------
    def _render_locator(self, spec: Locator) -> Locator:
        """One rung with ``{{input.x}}`` made real for the row in hand.

        Returns the rung unchanged when it holds no template, which is every
        rung of every recording made before a locator could carry one.
        """
        return spec.render(lambda value: self._render(value, self._row_values))

    def _render(self, value: str, inputs: dict[str, Any]) -> str:
        try:
            return render_template(
                value, inputs=inputs, secrets=self.secrets, env=self.env
            )
        except MissingValue as exc:
            raise StepFailed(
                "?",
                f"{exc.args[0]} was referenced but not supplied. Refusing to continue: "
                "typing an empty value and reporting success is worse than stopping.",
            ) from exc

    def _enforce_allowlist(self, url: str) -> None:
        """The domain allowlist applies to a replay exactly as to the agent.

        A recorded step is not automatically trusted: a use case whose input is
        a URL column takes that URL from a spreadsheet, and a spreadsheet is
        untrusted input.
        """
        # Rendered, not raw: a use case that navigates to {{env.base_url}} has
        # an allowlist naming the same thing, and comparing a live URL against
        # the literal text "{{env.base_url}}" blocks every navigation it was
        # meant to permit.
        allowed = [
            _host_of(render_template(domain, inputs={}, secrets={}, env=self.env))
            for domain in self.usecase.allowed_domains
        ]
        decision = check_navigation("navigate", {"url": url}, allowed)
        if not decision.allowed:
            raise NavigationBlocked(decision.reason or f"{url} is not an allowed domain")



#: Content types for the files a migration actually pulls off a vendor site.
#:
#: Pinned rather than left to ``mimetypes`` because that reads the Windows
#: registry: the same .csv is ``text/csv`` on a Linux worker and
#: ``application/vnd.ms-excel`` on a developer's laptop. An artifact stored for
#: years, and re-uploaded into another system, should not have a content type
#: that depends on which machine happened to fetch it.
_DOCUMENT_TYPES = {
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".json": "application/json",
    ".zip": "application/zip",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


# ---------------------------------------------------------------------------
# Composing a locator
# ---------------------------------------------------------------------------
#
# Module-level rather than methods, and that is not tidying. The review UI
# offers a "check this locator against a page" button, and a check that
# resolved locators through *different* code than the executor would be
# answering a different question -- confidently, in a screen whose entire
# purpose is to tell somebody whether their edit will work.


def build_locator(page: Any, spec: Locator):
    """``spec`` as a Playwright locator on ``page``, or ``None``.

    One call each, deliberately: the recorder wrote what it saw, and this
    performs it rather than reinterpreting it. What composition there is -- a
    frame chain, a scope, a text filter -- performs the *same* calls Playwright
    would, in the order ``playwright codegen`` writes them, so a recorded chain
    and the chain rebuilt here are the same locator.
    """
    try:
        root = _frame_root(page, spec.frames)
        if root is None:
            return None
        return _compose(root, spec)
    except Exception:  # noqa: BLE001 - a malformed selector is a dead rung
        return None


def _frame_root(page: Any, frames: list[str]):
    """The page, or the innermost frame of a chain of iframes.

    An element inside an iframe is not merely harder to find from the page: it
    is not there at all. Every rung of a locator that names frames is resolved
    against the frame, which is why this is the first thing
    :func:`build_locator` does rather than a special case inside each strategy.
    """
    root = page
    for selector in frames:
        if not selector:
            return None
        root = root.frame_locator(selector)
    return root


def _compose(root: Any, spec: Locator):
    """``spec`` as a locator under ``root``, scope first.

    Recursive because ``within`` is: the button inside the cell inside the row
    is three calls, and each one is the same call Playwright would have been
    given directly.
    """
    parent = root
    if spec.within is not None:
        parent = _compose(root, spec.within)
        if parent is None:
            return None

    # `exact` decides whether the recorded name has to be the whole accessible
    # name or merely part of it, and Playwright's default is the loose one.
    # Every strategy that matches by name takes it; `get_by_test_id` matches an
    # attribute and has no such parameter.
    if spec.strategy == "role":
        base = parent.get_by_role(spec.role or "", name=spec.name or None, exact=spec.exact)
    elif spec.strategy == "label":
        base = parent.get_by_label(spec.text or "", exact=spec.exact)
    elif spec.strategy == "placeholder":
        base = parent.get_by_placeholder(spec.text or "", exact=spec.exact)
    elif spec.strategy == "test_id":
        base = parent.get_by_test_id(spec.text or "")
    elif spec.strategy == "alt_text":
        base = parent.get_by_alt_text(spec.text or "", exact=spec.exact)
    elif spec.strategy == "text":
        base = parent.get_by_text(spec.text or "", exact=spec.exact)
    elif spec.strategy == "css":
        base = parent.locator(spec.selector or "")
    elif spec.strategy == "nth":
        return None
    else:  # pragma: no cover - the schema constrains `strategy`
        return None

    # Filter before indexing. `nth` counts among the matches that survive the
    # filter, which is what "the second row mentioning Acme" means and is the
    # order codegen writes the two calls in.
    if spec.has_text:
        base = base.filter(has_text=spec.has_text)
    if spec.nth < 0:
        # How ``.last`` is held; see the locator chain reader in codegen.py.
        return base.last
    return base.nth(spec.nth) if spec.nth else base


def visible_only(locator):
    """``locator`` narrowed to what a person could actually see, or None.

    ``locator.count()`` reports every match in the DOM, visible or not, so one
    visible control plus one hidden duplicate counts as two and the step is
    refused as ambiguous on a page a person would call unambiguous.

    **Which rungs this affects is not the obvious answer.** ``get_by_role``
    resolves against the accessibility tree, and a ``display:none`` element is
    not in it, so a role rung never saw the duplicate. ``text`` and ``css``
    rungs match against the DOM, and do. Those are the *fallback* rungs --
    ``codegen._ladder`` puts a text rung under every named role rung -- so a
    hidden duplicate costs nothing until the day the role rung stops matching.
    Then the ladder falls through, and the step fails as ambiguous because of
    an element nobody can see, on the one occasion the fallback existed for.

    It is not a visibility check in general. Playwright counts an element with
    a box as visible even when it is off-screen or covered, so an overlay still
    has to be handled by the step that opens it.

    Returns ``None`` when the installed Playwright predates
    ``filter(visible=)``, which is inside the supported range: the caller then
    behaves exactly as it did before this existed.
    """
    try:
        return locator.filter(visible=True)
    except TypeError:  # Playwright < 1.51
        return None
    except Exception:  # noqa: BLE001 - mid-navigation; the caller retries
        return None


async def probe_locators(
    locators: list[Locator], *, url: str, settings: Any, timeout_ms: int = 10_000
) -> dict[str, Any]:
    """Open ``url`` and say what each rung matches, right now.

    The answer a person editing a locator needs, and could not get before: a
    rung that reads perfectly well matches nothing, or matches four things, and
    until now the only way to find that out was to run the use case -- where an
    ambiguous rung shows up as a thirty-second timeout on row one of a batch.

    Both counts are reported. Total is what is in the DOM, visible is what a
    person could reach, and the gap between them is worth showing rather than
    hiding: "matches 2, one of them visible" tells somebody their page has a
    hidden duplicate, which is a thing they may want to know about their site
    as much as about their locator.

    Read-only. It navigates and counts; it never clicks, fills or submits.
    """
    config = BrowserConfig.from_settings(settings, headless=True)
    config.timeout_ms = timeout_ms

    results: list[dict[str, Any]] = []
    async with PlaywrightSession(config) as browser:
        await browser.page.goto(url, timeout=timeout_ms)
        await browser.settle(timeout_ms)

        for spec in locators:
            results.append(await _probe_one(browser.page, spec))

        return {
            "page_url": browser.url,
            "page_title": await browser.title(),
            "results": results,
        }


async def _probe_one(page: Any, spec: Locator) -> dict[str, Any]:
    """One rung's verdict: how many it matches, and what they are."""
    entry: dict[str, Any] = {
        "describe": spec.describe(),
        "total": 0,
        "visible": 0,
        "matches": [],
        "ok": False,
        "reason": "",
    }

    locator = build_locator(page, spec)
    if locator is None:
        entry["reason"] = (
            "this rung cannot be turned into a Playwright call -- a 'nth' rung has no "
            "meaning on its own, and a blank frame selector has nowhere to go"
        )
        return entry

    try:
        entry["total"] = await locator.count()
        narrowed = visible_only(locator)
        entry["visible"] = await narrowed.count() if narrowed is not None else entry["total"]
    except Exception as exc:  # noqa: BLE001 - a bad selector is an answer
        entry["reason"] = _reason(exc)
        return entry

    # Named so a person can tell whether the thing found is the thing meant.
    # A count alone says "1 match" for the wrong element just as happily.
    countable = (visible_only(locator) or locator) if entry["visible"] else locator
    for index in range(min(entry["visible"] or entry["total"], PROBE_SAMPLES)):
        try:
            one = countable.nth(index)
            role = await one.get_attribute("role") or ""
            text = ((await one.inner_text()) or "").strip()
        except Exception:  # noqa: BLE001 - it was counted, it may still go
            continue
        label = " ".join(text.split())[:120]
        entry["matches"].append(f"{role} {label}".strip() or "(no text)")

    counted = entry["visible"] if visible_only(locator) is not None else entry["total"]
    if counted == 1:
        entry["ok"] = True
    elif counted == 0 and entry["total"]:
        entry["reason"] = (
            f"matches {entry['total']} element(s) in the page, none of them visible. A "
            "step would wait for one to appear and then fail. This is usually a control "
            "that only exists once a dialog or a tab is opened."
        )
    elif counted == 0:
        entry["reason"] = "matches nothing on this page"
    else:
        entry["reason"] = (
            f"matches {counted} elements, so a step using it is refused as ambiguous "
            "rather than acting on whichever one comes first. Say which by scoping it "
            "to the row or dialog it sits in, or by filtering on text."
        )
    return entry



def _mime_for(filename: str) -> str:
    """The content type to store a downloaded file under."""
    suffix = Path(filename).suffix.lower()
    if suffix in _DOCUMENT_TYPES:
        return _DOCUMENT_TYPES[suffix]
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _host_of(pattern: str) -> str:
    """An allowlist entry as a host, which is what the matcher compares.

    ``{{env.base_url}}`` has to carry a scheme -- it builds URLs -- so once
    rendered it reads ``https://uat.example.com`` while the matcher wants
    ``uat.example.com``. A pattern that is already a host, wildcard or not, is
    returned untouched.
    """
    if "://" not in pattern:
        return pattern
    host = urlparse(pattern).hostname or ""
    return host or pattern


#: What Playwright's call log says, and what it means in a sentence.
#:
#: The log is the only place the *cause* of a timeout is written down, and this
#: is the list of causes it actually reports. Order matters: interception is
#: checked before visibility because a covered element is often reported as
#: both, and "covered by something" is the one a person can act on.
_CALL_LOG_CAUSES: tuple[tuple[str, str], ...] = (
    ("intercepts pointer events", "something else on the page is covering it"),
    ("element is not stable", "it kept moving, so a click was never safe to send"),
    ("element is not enabled", "the control is disabled"),
    ("element is not visible", "it is in the page but not visible"),
    ("element is not editable", "the field is read-only"),
    ("element does not receive pointer events", "it cannot be clicked where it is"),
)

#: How much of Playwright's call log is kept on the failure event. Enough for
#: the interception line and the few before it; not the whole retry history,
#: which repeats the same three lines for thirty seconds.
CALL_LOG_CHARS = 2_000


def _cause_in(text: str) -> str:
    """The plain-language cause out of a Playwright call log, or "".

    Where the interception line names the offending element, that name comes
    with it: "something else is covering it" sends somebody looking, and
    "a div with id onetrust-consent-sdk is covering it" sends them to the
    cookie banner.
    """
    lowered = text.casefold()
    for phrase, meaning in _CALL_LOG_CAUSES:
        if phrase not in lowered:
            continue
        if phrase == "intercepts pointer events":
            for line in text.splitlines():
                if "intercepts pointer events" in line.casefold():
                    culprit = line.strip().split(" intercepts")[0].strip("- ").strip()
                    if culprit:
                        return f"{meaning}: {culprit[:160]}"
            return meaning
        return meaning
    return ""


def _reason(exc: Exception) -> str:
    """A step failure in one line a person can act on.

    Playwright's own errors are a first line and then several paragraphs of
    call log. This used to keep the first line only, on the stated grounds
    that it "carries the actual cause" -- which is true of every error except
    the one that matters most. For a timeout the first line says
    ``Locator.click: Timeout 30000ms exceeded`` and the cause is in the log:
    the element was found, and something was covering it, or it would not hold
    still, or it was disabled.

    A real recording failed this way on a column header and the stored reason
    said only "timeout". Nobody could act on that, a repair answered it by
    re-spelling the locator, and the next run happened to work -- so the
    diagnosis was never made and the same failure came back. The cause is now
    read out of the log and put on the end of the line, and the log itself is
    kept on the failure event for a repair to read.
    """
    text = str(exc).strip()
    first = text.splitlines()[0] if text else type(exc).__name__
    cause = _cause_in(text)
    line = f"{type(exc).__name__}: {first}"
    if cause:
        line = f"{line} -- {cause}"
    return line[:500]


def call_log_of(exc: Exception) -> str:
    """Playwright's own account of what it tried, capped.

    Kept verbatim rather than summarised: the lines are already terse, and a
    summary of a diagnosis is a second place for the diagnosis to be wrong.
    """
    text = str(exc).strip()
    lines = text.splitlines()
    return "\n".join(lines[1:]).strip()[:CALL_LOG_CHARS] if len(lines) > 1 else ""


async def emit_replay_error(
    sink: EventSink, run_id: str, kind: str, message: str
) -> None:
    """One error event, for the paths that fail before an executor exists."""
    await sink.emit(
        ErrorEvent(
            run_id=run_id, seq=sink.reserve_seq(), step=0, kind=kind, message=message
        )
    )


__all__ = [
    "Healer",
    "NavigationBlocked",
    "RowResult",
    "ScriptBlocked",
    "StepFailed",
    "StepOutcome",
    "UseCaseExecutor",
    "emit_replay_error",
]
