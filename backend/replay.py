"""Executes a recorded use case. **No LLM, ever.**

This module deliberately does not import :mod:`llm`, and
:class:`UseCaseExecutor` has no parameter that could accept a model client.
``tests/test_replay.py`` asserts both. That is what makes "zero tokens" a
structural property rather than a promise someone has to keep remembering:
threading a no-LLM flag through :class:`agent.BrowserAgent` instead would have
left the expensive path one bug away from being re-entered.

What replaces the model's judgement
-----------------------------------
The agent decides what to click by looking at a page. A replay cannot, so two
mechanisms stand in:

**The locator ladder.** Each step carries ranked ways to find its element. The
``role`` rung is resolved against a *snapshot taken right now*, which is what
makes a recording survive a redesign -- and snapshots are free here, because
nothing is paying to put them in a context window. Which rung matched is
reported on every step, so drift shows up before it becomes breakage.

**Assertions.** Evaluated locally against the snapshot or the URL. Without them
a batch has no way to notice that row 12 silently did nothing.

Setup versus rows
-----------------
:meth:`UseCaseExecutor.run_setup` runs once per browser session and
:meth:`UseCaseExecutor.run_row` once per input row, because a batch shares one
session and must not sign in a thousand times. A single-row execution is just
``run_setup`` followed by one ``run_row``, so there is no second code path for
the batch runner to drift away from.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from events import (
    AgentEvent,
    ErrorEvent,
    Screenshot,
    StepFinished,
    StepStarted,
    ToolCall,
    ToolResult,
)
from mcp_client import MCPBrowserSession, MCPConnectionError, MCPToolError
from policy import check_navigation
from redaction import Redactor
from snapshot import Snapshot, parse as parse_snapshot
from usecase import (
    Assertion,
    Locator,
    MissingValue,
    Step,
    UseCase,
    render_template,
)

log = logging.getLogger(__name__)

Phase = Literal["setup", "row", "reset", "teardown"]

#: Retry schedule for transient tool failures, mirroring the agent loop.
RETRY_BACKOFF = (0.4, 1.2)

#: How often an assertion re-checks while waiting for its timeout.
POLL_INTERVAL = 0.5

_URL_RE = re.compile(r"(?:Page URL|url)\s*:\s*(\S+)", re.IGNORECASE)
_TITLE_RE = re.compile(r"Page Title:\s*(.*)", re.IGNORECASE)


class EventSink(Protocol):
    def reserve_seq(self) -> int: ...
    async def emit(self, event: AgentEvent) -> None: ...
    async def save_screenshot(
        self, data: bytes, *, seq: int, mime: str = "image/png"
    ) -> tuple[str, str] | None: ...


class StepFailed(RuntimeError):
    """A step did not succeed and its ``on_failure`` says to stop."""

    def __init__(self, step_id: str, message: str) -> None:
        super().__init__(message)
        self.step_id = step_id


class SessionLost(RuntimeError):
    """The shared session is no longer authenticated.

    Raised by :meth:`UseCaseExecutor.check_session` so the batch runner can
    re-run setup once rather than failing every remaining row.
    """


class ScriptBlocked(RuntimeError):
    """A ``script`` step was reached without ``allow_scripts``."""


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
    """Runs a :class:`UseCase` against a live MCP session.

    Note the constructor signature: there is no ``llm`` parameter, and there is
    no code path here that could use one.
    """

    def __init__(
        self,
        usecase: UseCase,
        mcp: MCPBrowserSession,
        sink: EventSink,
        *,
        run_id: str,
        secrets: dict[str, str] | None = None,
        redactor: Redactor | None = None,
        screenshot_on_failure: bool = True,
        step_timeout: float = 30.0,
    ) -> None:
        self.usecase = usecase
        self.mcp = mcp
        self.sink = sink
        self.run_id = run_id
        self.secrets = dict(secrets or {})
        self.redactor = redactor or Redactor(self.secrets.values())
        self.screenshot_on_failure = screenshot_on_failure
        self.step_timeout = step_timeout

        self.step_number = 0
        self._last_snapshot: Snapshot | None = None
        self._last_page_url: str | None = None
        #: Rungs deeper than the first, per step id. Surfaced as drift.
        self.locator_drift: dict[str, int] = {}

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
        must keep going. Only a dead transport propagates.
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
            await self._capture_failure()
            return RowResult(
                ok=False,
                outputs=outputs,
                failed_step_id=exc.step_id,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
                steps=outcomes,
            )

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
        ok, _ = self._evaluate(self.usecase.session_check)
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
        except StepFailed:
            raise
        except (MCPConnectionError, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 - one step must not kill the batch
            outcome = StepOutcome(
                step_id=step.id, ok=False, duration_ms=0, message=f"{type(exc).__name__}: {exc}"
            )

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
            return outcome

        # An optional step, or one told to continue, is a recorded failure that
        # does not stop the row.
        if step.optional or step.on_failure == "continue":
            outcome.skipped = True
            return outcome

        raise StepFailed(step.id, f"step {step.id!r} ({step.summary()}) failed: {outcome.message}")

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
        if step.action == "script":
            return await self._do_script(step)
        return await self._do_element_action(step, values)

    # -- actions ------------------------------------------------------------
    async def _do_navigate(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        url = self._render(step.url or "", values)
        self._enforce_allowlist("browser_navigate", {"url": url})

        tool = self.mcp.find_tool("browser_navigate", contains=("navigate", "goto"))
        if tool is None:
            return StepOutcome(step.id, False, 0, "the MCP server exposes no navigation tool")

        outcome = await self._call(tool, {"url": url}, step)
        if outcome.ok:
            await self._absorb(outcome.message)
        return outcome

    async def _do_element_action(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        tool = self._tool_for(step.action)
        if tool is None:
            return StepOutcome(step.id, False, 0, f"no MCP tool for action {step.action!r}")

        resolved = await self._resolve(step.locators, step.id)
        if resolved is None:
            return StepOutcome(step.id, False, 0, self._not_found_message(step))
        target, rung, described = resolved

        arguments: dict[str, Any] = {"target": target}
        if step.description:
            arguments["element"] = step.description
        if step.action in {"fill", "select", "press"} and step.value is not None:
            key = "key" if step.action == "press" else "text"
            arguments[key] = self._render(step.value, values)
        if step.action == "upload" and step.value is not None:
            arguments["paths"] = [self._render(step.value, values)]

        self._enforce_allowlist(tool, arguments)
        outcome = await self._call(tool, arguments, step)
        outcome.matched_locator = described
        outcome.locator_rung = rung
        if outcome.ok:
            await self._absorb(outcome.message)
        return outcome

    async def _do_fill_form(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        tool = self.mcp.find_tool("browser_fill_form", contains=("fill_form",))
        if tool is None:
            # Degrade to one fill per field rather than failing the step.
            return await self._fill_fields_individually(step, values)

        fields: list[dict[str, Any]] = []
        deepest = 0
        for item in step.fields:
            resolved = await self._resolve(item.locators, step.id)
            if resolved is None:
                return StepOutcome(
                    step.id, False, 0, f"could not find the {item.name!r} field on the page"
                )
            target, rung, _ = resolved
            deepest = max(deepest, rung)
            fields.append(
                {
                    "target": target,
                    "name": item.name,
                    "type": item.type,
                    "value": self._render(item.value, values),
                }
            )

        self._enforce_allowlist(tool, {"fields": fields})
        outcome = await self._call(tool, {"fields": fields}, step)
        outcome.locator_rung = deepest
        if outcome.ok:
            await self._absorb(outcome.message)
        return outcome

    async def _fill_fields_individually(
        self, step: Step, values: dict[str, Any]
    ) -> StepOutcome:
        tool = self._tool_for("fill")
        if tool is None:
            return StepOutcome(step.id, False, 0, "no MCP tool can fill a form field")
        for item in step.fields:
            resolved = await self._resolve(item.locators, step.id)
            if resolved is None:
                return StepOutcome(step.id, False, 0, f"could not find the {item.name!r} field")
            target, _, _ = resolved
            arguments = {
                "target": target,
                "element": item.name,
                "text": self._render(item.value, values),
            }
            self._enforce_allowlist(tool, arguments)
            outcome = await self._call(tool, arguments, step)
            if not outcome.ok:
                return outcome
        return StepOutcome(step.id, True, 0, "filled each field individually")

    async def _do_wait(self, step: Step, values: dict[str, Any]) -> StepOutcome:
        spec = step.wait_for
        if spec is None:
            return StepOutcome(step.id, True, 0, "nothing to wait for")

        tool = self.mcp.find_tool("browser_wait_for", contains=("wait",))
        arguments: dict[str, Any] = {}
        if spec.kind == "time" and spec.seconds is not None:
            arguments["time"] = spec.seconds
        elif spec.kind == "text" and spec.value:
            arguments["text"] = self._render(spec.value, values)
        elif spec.kind == "text_gone" and spec.value:
            arguments["textGone"] = self._render(spec.value, values)
        else:
            # load_state has no MCP equivalent; a fresh snapshot is the
            # cheapest proof the page responded.
            await self._refresh_snapshot()
            return StepOutcome(step.id, True, 0, "page responded")

        if tool is None:
            await asyncio.sleep(min(spec.seconds or 1.0, 5.0))
            return StepOutcome(step.id, True, 0, "waited locally")
        return await self._call(tool, arguments, step)

    async def _do_assert(self, step: Step) -> StepOutcome:
        check = step.assertion
        if check is None:
            return StepOutcome(step.id, True, 0, "no assertion")

        deadline = time.monotonic() + max(check.timeout_ms, 0) / 1000.0
        detail = ""
        while True:
            await self._refresh_snapshot()
            ok, detail = self._evaluate(check)
            if ok or time.monotonic() >= deadline:
                break
            await asyncio.sleep(POLL_INTERVAL)

        return StepOutcome(
            step.id,
            ok,
            0,
            f"{check.describe()} -- {'held' if ok else 'did not hold'}{detail}",
        )

    async def _do_extract(self, step: Step, outputs: dict[str, Any]) -> StepOutcome:
        resolved = await self._resolve(step.locators, step.id)
        if resolved is None:
            return StepOutcome(step.id, False, 0, self._not_found_message(step))
        _, rung, described = resolved

        node = self._last_node
        value = (node.text or node.name) if node else ""
        outputs[step.output or step.id] = value
        return StepOutcome(
            step.id, True, 0, f"extracted {value!r}", matched_locator=described, locator_rung=rung
        )

    async def _do_script(self, step: Step) -> StepOutcome:
        tool = self.mcp.find_tool(
            "browser_run_code_unsafe", "browser_evaluate", contains=("run_code", "evaluate")
        )
        if tool is None:
            return StepOutcome(step.id, False, 0, "the MCP server exposes no script tool")
        return await self._call(tool, {"code": step.code or ""}, step)

    # -- locator ladder -----------------------------------------------------
    async def _resolve(
        self, locators: list[Locator], step_id: str
    ) -> tuple[str, int, str] | None:
        """Walk the ladder. Returns ``(target, rung_index, description)``.

        The ``role`` rung is resolved against a snapshot taken *now*, which is
        what lets a use case recorded months ago survive a redesign. Later
        rungs are handed to the server as recorded.
        """
        self._last_node = None
        for rung, locator in enumerate(locators):
            if locator.strategy == "role":
                await self._refresh_snapshot()
                if self._last_snapshot is None:
                    continue
                node = self._last_snapshot.locate(locator.role or "", locator.name, locator.nth)
                if node is None:
                    continue
                self._last_node = node
                if rung > 0:
                    self._note_drift(step_id, rung)
                return f"ref={node.ref}", rung, locator.describe()

            if locator.strategy == "css" and locator.selector:
                if rung > 0:
                    self._note_drift(step_id, rung)
                return locator.selector, rung, locator.describe()

            if locator.strategy == "text" and locator.text:
                if rung > 0:
                    self._note_drift(step_id, rung)
                # Playwright's text engine, which the MCP server accepts.
                return f"text={locator.text}", rung, locator.describe()

            if locator.strategy == "nth":
                if rung > 0:
                    self._note_drift(step_id, rung)
                return f"nth={locator.nth}", rung, locator.describe()

        return None

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
            available = f". The page currently shows: {top}"
        return f"no element matched. Tried: {tried}{available}"

    # -- assertions ---------------------------------------------------------
    def _evaluate(self, check: Assertion) -> tuple[bool, str]:
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
            haystack = self._last_snapshot_text
            held = value.casefold() in haystack.casefold()
            detail = ""
        elif check.kind == "element_visible":
            node = self._locate_for_assertion(check)
            held = node is not None
            detail = ""
        elif check.kind == "element_count":
            matches = (
                snapshot.find(check.locator.role or "", check.locator.name)
                if snapshot and check.locator
                else []
            )
            held = len(matches) == (check.count or 0)
            detail = f" (found {len(matches)})" if not held else ""
        else:  # pragma: no cover - the schema constrains `kind`
            return False, f" (unknown assertion kind {check.kind!r})"

        return (not held if check.negate else held), detail

    def _locate_for_assertion(self, check: Assertion):
        if self._last_snapshot is None or check.locator is None:
            return None
        locator = check.locator
        if locator.strategy == "role":
            return self._last_snapshot.locate(locator.role or "", locator.name, locator.nth)
        return None

    # -- MCP plumbing -------------------------------------------------------
    def _tool_for(self, action: str) -> str | None:
        return {
            "click": self.mcp.find_tool("browser_click", contains=("click",)),
            "fill": self.mcp.find_tool("browser_type", contains=("type", "fill")),
            "select": self.mcp.find_tool("browser_select_option", contains=("select",)),
            "press": self.mcp.find_tool("browser_press_key", contains=("press", "key")),
            "hover": self.mcp.find_tool("browser_hover", contains=("hover",)),
            "upload": self.mcp.find_tool("browser_file_upload", contains=("upload",)),
            "extract": self.mcp.find_tool("browser_snapshot", contains=("snapshot",)),
        }.get(action)

    async def _call(self, tool: str, arguments: dict[str, Any], step: Step) -> StepOutcome:
        """One MCP tool call with bounded retries, emitted as normal events.

        Reusing ``tool_call``/``tool_result`` means the existing timeline, the
        WebSocket replay and the run view all work on a replay with no frontend
        changes at all.
        """
        call_id = f"{step.id}-{uuid.uuid4().hex[:6]}"
        await self.sink.emit(
            ToolCall(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step_number,
                call_id=call_id,
                name=tool,
                arguments=self.redactor.structure(arguments),
            )
        )

        timeout = min(step.timeout_ms / 1000.0 or self.step_timeout, self.step_timeout)
        last_error = ""
        for attempt in range(len(RETRY_BACKOFF) + 1):
            try:
                result = await self.mcp.call_tool(tool, arguments, timeout=timeout)
            except MCPToolError as exc:
                last_error = str(exc)
                if attempt < len(RETRY_BACKOFF):
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue
                break
            else:
                text = result.text or ""
                await self.sink.emit(
                    ToolResult(
                        run_id=self.run_id,
                        seq=self.sink.reserve_seq(),
                        step=self.step_number,
                        call_id=call_id,
                        name=tool,
                        ok=not result.is_error,
                        duration_ms=result.duration_ms,
                        text=self.redactor.text(text[:4000]),
                        attempts=attempt + 1,
                    )
                )
                return StepOutcome(
                    step.id, not result.is_error, result.duration_ms, text[:500]
                )

        await self.sink.emit(
            ToolResult(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step_number,
                call_id=call_id,
                name=tool,
                ok=False,
                duration_ms=0,
                text=self.redactor.text(last_error),
                attempts=len(RETRY_BACKOFF) + 1,
            )
        )
        return StepOutcome(step.id, False, 0, last_error)

    async def _refresh_snapshot(self) -> None:
        """Take a fresh accessibility snapshot.

        Free in this context: a snapshot only costs money when it is put into a
        model's context window, and there is no model here.
        """
        tool = self.mcp.find_tool("browser_snapshot", contains=("snapshot",))
        if tool is None:
            return
        try:
            result = await self.mcp.call_tool(tool, {}, timeout=20.0)
        except (MCPToolError, MCPConnectionError) as exc:
            log.debug("snapshot failed", extra={"error": str(exc)})
            return
        await self._absorb(result.text or "")

    async def _absorb(self, text: str) -> None:
        """Update the cached page view from any tool result that carries one."""
        if not text:
            return
        self._last_snapshot_text = text
        parsed = parse_snapshot(text)
        if len(parsed):
            self._last_snapshot = parsed
        match = _URL_RE.search(text)
        if match:
            self._last_page_url = match.group(1)
        elif parsed.page_url:
            self._last_page_url = parsed.page_url

    async def _capture_failure(self) -> str | None:
        """Screenshot at the moment of failure. Never fails the row."""
        if not self.screenshot_on_failure:
            return None
        tool = self.mcp.find_tool("browser_take_screenshot", contains=("screenshot",))
        if tool is None:
            return None
        try:
            result = await self.mcp.call_tool(tool, {}, timeout=15.0)
        except (MCPToolError, MCPConnectionError):
            return None
        for mime, data in result.images:
            seq = self.sink.reserve_seq()
            saved = await self.sink.save_screenshot(data, seq=seq, mime=mime)
            if saved is None:
                return None
            artifact_id, url = saved
            await self.sink.emit(
                Screenshot(
                    run_id=self.run_id,
                    seq=seq,
                    step=self.step_number,
                    artifact_id=artifact_id,
                    url=url,
                    caption="failure",
                    page_url=self._last_page_url,
                )
            )
            return artifact_id
        return None

    # -- values -------------------------------------------------------------
    def _render(self, value: str, inputs: dict[str, Any]) -> str:
        try:
            return render_template(value, inputs=inputs, secrets=self.secrets)
        except MissingValue as exc:
            raise StepFailed(
                "?",
                f"{exc.args[0]} was referenced but not supplied. Refusing to continue: "
                "typing an empty value and reporting success is worse than stopping.",
            ) from exc

    def _enforce_allowlist(self, tool: str, arguments: dict[str, Any]) -> None:
        """The domain allowlist applies to a replay exactly as to the agent.

        A use case cannot escape it just because a human approved the original
        recording once.
        """
        check = check_navigation(tool, arguments, self.usecase.allowed_domains)
        if not check.allowed:
            raise StepFailed("?", check.reason)

    # -- attributes assigned lazily ----------------------------------------
    _last_snapshot_text: str = ""
    _last_node: Any = None


async def emit_replay_error(
    sink: EventSink, run_id: str, kind: str, message: str, *, step: int | None = None
) -> None:
    await sink.emit(
        ErrorEvent(
            run_id=run_id,
            seq=sink.reserve_seq(),
            step=step,
            kind=kind,
            message=message,
            recoverable=False,
        )
    )
