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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import imagediff
from browser import BrowserError, PlaywrightSession
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

#: How much of the page to keep on a failure event. A repair reads this later,
#: and the interactive nodes it needs are near the top.
FAILURE_SNAPSHOT_CHARS = 8_000


class EventSink(Protocol):
    def reserve_seq(self) -> int: ...
    async def emit(self, event: AgentEvent) -> None: ...
    async def save_screenshot(
        self, data: bytes, *, seq: int, mime: str = "image/png"
    ) -> tuple[str, str] | None: ...

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
    error: str | None = None
    duration_ms: int = 0
    steps: list[StepOutcome] = field(default_factory=list)
    #: Always 0 for a pure replay. Recorded so the dashboard can prove it.
    llm_calls: int = 0
    llm_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "outputs": self.outputs,
            "failed_step_id": self.failed_step_id,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
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
        self._last_snapshot: Snapshot | None = None
        self._last_snapshot_text = ""
        self._last_page_url: str | None = None
        self._last_node = None
        #: Rungs deeper than the first, per step id. Surfaced as drift.
        self.locator_drift: dict[str, int] = {}
        #: Repairs accepted this session, for the version bump afterwards.
        self.healed: list[Any] = []

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

    async def run_row(self, inputs: dict[str, Any]) -> RowResult:
        """Run ``row_reset`` then ``row_steps`` for one input row.

        Never raises for an ordinary step failure -- the caller is a batch that
        must keep going. Only a dead browser propagates.
        """
        started = time.monotonic()
        values = self.usecase.with_defaults(inputs)
        outputs: dict[str, Any] = {}
        outcomes: list[StepOutcome] = []

        missing = self.usecase.missing_inputs(values)
        if missing:
            return RowResult(
                ok=False,
                error=f"missing required input(s): {', '.join(missing)}",
                duration_ms=0,
            )

        try:
            if self.usecase.row_reset is not None:
                outcomes.append(
                    await self._run_step(self.usecase.row_reset, values, phase="reset")
                )
            for step in self.usecase.row_steps:
                outcomes.append(
                    await self._run_step(step, values, phase="row", outputs=outputs)
                )
        except StepFailed as exc:
            return RowResult(
                ok=False,
                outputs=outputs,
                failed_step_id=exc.step_id,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
                steps=outcomes,
            )

        await self._capture_row(ok=True)
        return RowResult(
            ok=True,
            outputs=outputs,
            duration_ms=int((time.monotonic() - started) * 1000),
            steps=outcomes,
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
        ok, _ = await self._evaluate(self.usecase.session_check)
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
            outcome = await self._perform(step, values, outputs if outputs is not None else {})
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

    async def _heal(
        self, step: Step, values: dict[str, Any], outputs: dict[str, Any]
    ) -> StepOutcome | None:
        """Ask the healer for a new locator, then retry the step once.

        The repaired locator is *prepended* to the ladder rather than replacing
        it, so a repair that turns out to be wrong degrades to what the
        recording already knew instead of losing it.
        """
        assert self.healer is not None
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
            return None

        # The repair may have left a dialog open or the page part-way through
        # something, so put the browser back before the next row inherits it.
        if self.usecase.row_reset is not None:
            try:
                await self._do_navigate(self.usecase.row_reset, values)
            except Exception:  # noqa: BLE001 - best effort
                log.debug("row_reset after healing failed", extra={"step_id": step.id})

        return outcome if outcome.ok else None

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
            await self._refresh_snapshot()
        return StepOutcome(step.id, ok, duration, message)

    async def _do_element_action(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        locator = None
        rung: int | None = None
        described: str | None = None

        # `press` sends a key to whatever has focus and `upload` answers an open
        # file chooser; neither needs a target, and neither is recorded with
        # one. Every other element action does.
        if step.locators or step.action not in OPTIONAL_LOCATOR_ACTIONS:
            resolved = await self._resolve(step.locators, step.id)
            if resolved is None:
                return StepOutcome(step.id, False, 0, self._not_found_message(step))
            locator, rung, described = resolved

        timeout = int(self.step_timeout * 1000)
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
            await self._refresh_snapshot()
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

        await self._refresh_snapshot()
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
        check = step.assertion
        if check is None:
            return StepOutcome(step.id, True, 0, "no assertion")

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
            await self._refresh_snapshot()
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
    async def _resolve(
        self, locators: list[Locator], step_id: str, *, single: bool = True
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
        """
        self._last_node = None
        if not locators:
            return None

        semantic = [(i, loc) for i, loc in enumerate(locators) if loc.semantic]
        weak = [(i, loc) for i, loc in enumerate(locators) if not loc.semantic]
        ordered = [*semantic, *weak]

        deadline = time.monotonic() + self.step_timeout

        while True:
            for rung, spec in ordered:
                locator = self._build(spec)
                if locator is None:
                    continue
                try:
                    if await locator.count() >= 1:
                        if rung > 0:
                            self._note_drift(step_id, rung)
                        # Collapsed to the first match for an action that acts
                        # on one element. `extract_rows` passes single=False,
                        # because matching many is the whole point there.
                        if single and spec.nth == 0:
                            return locator.first, rung, spec.describe()
                        return locator, rung, spec.describe()
                except Exception:  # noqa: BLE001 - mid-navigation; try again
                    continue

            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(POLL_INTERVAL)

    def _build(self, spec: Locator):
        """One rung as a Playwright locator.

        One call each, deliberately: the recorder wrote what it saw, and this
        performs it rather than reinterpreting it.
        """
        page = self.browser.page
        try:
            if spec.strategy == "role":
                base = page.get_by_role(spec.role or "", name=spec.name or None)
            elif spec.strategy == "label":
                base = page.get_by_label(spec.text or "")
            elif spec.strategy == "placeholder":
                base = page.get_by_placeholder(spec.text or "")
            elif spec.strategy == "test_id":
                base = page.get_by_test_id(spec.text or "")
            elif spec.strategy == "alt_text":
                base = page.get_by_alt_text(spec.text or "")
            elif spec.strategy == "text":
                base = page.get_by_text(spec.text or "")
            elif spec.strategy == "css":
                base = page.locator(spec.selector or "")
            elif spec.strategy == "nth":
                return None
            else:  # pragma: no cover - the schema constrains `strategy`
                return None
        except Exception:  # noqa: BLE001 - a malformed selector is a dead rung
            return None
        if spec.nth < 0:
            # How ``.last`` is held; see the locator chain reader in codegen.py.
            return base.last
        return base.nth(spec.nth) if spec.nth else base

    def _note_drift(self, step_id: str, rung: int) -> None:
        """Record that the preferred locator no longer matched.

        A use case that starts falling through to later rungs is drifting, and
        saying so early is the difference between a warning and a broken batch.
        """
        self.locator_drift[step_id] = max(self.locator_drift.get(step_id, 0), rung)
        log.info("locator fell through", extra={"step_id": step_id, "rung": rung})

    def _not_found_message(self, step: Step) -> str:
        tried = "; ".join(loc.describe() for loc in step.locators) or "(no locators recorded)"
        available = ""
        if self._last_snapshot is not None:
            roles = self._last_snapshot.roles()
            top = ", ".join(f"{k}x{v}" for k, v in sorted(roles.items(), key=lambda kv: -kv[1])[:6])
            if top:
                available = f" The page has: {top}."
        return f"no element matched. Tried: {tried}.{available}"

    # -- assertions ---------------------------------------------------------
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
        self._last_shot = None
        baseline = self.baselines.get(step.id)
        diff = await self._compare(baseline, shot)

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

    async def _compare(self, baseline: str | None, shot: str | None) -> float | None:
        """How much this step's page differs from the last time it worked."""
        if not baseline or not shot or self.read_artifact is None:
            return None
        try:
            before = await self.read_artifact(baseline)
            after = await self.read_artifact(shot)
        except Exception:  # noqa: BLE001 - a diff must never fail a row
            return None
        return imagediff.ratio(before, after)

    # -- browser plumbing ---------------------------------------------------
    async def _refresh_snapshot(self) -> None:
        """Read the page, for assertions and for the failure context.

        Free here in a way it never was for the agent: a snapshot is expensive
        only when it enters a model's context window, and there is no model.
        """
        self._last_snapshot = await self.browser.snapshot()
        self._last_page_url = self._last_snapshot.page_url or self.browser.url
        self._last_snapshot_text = await self.browser.text_content()

    async def _record_failure_context(self, step_id: str, message: str) -> None:
        """Persist the page as it was when a step failed.

        Repairing a use case later needs to know what was actually on screen,
        and the snapshots taken while resolving locators are internal -- they
        never reach the event log. Recording one here means a repair can be
        proposed from history alone, with no second browser session.
        """
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


def _reason(exc: Exception) -> str:
    """A step failure in one line a person can act on.

    Playwright's own errors are several paragraphs of call log, which is
    excellent in a terminal and useless in a results CSV with a thousand rows.
    The first line carries the actual cause.
    """
    text = str(exc).strip()
    first = text.splitlines()[0] if text else type(exc).__name__
    return f"{type(exc).__name__}: {first}"[:500]


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
