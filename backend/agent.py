"""The agentic loop.

One iteration is: send history + the latest page observation to the LLM, get
back either a tool call or a final answer, run the tool through MCP, append the
result, emit events, repeat.

Observation strategy: **snapshot-first, screenshot-second.** The accessibility
tree returned by the MCP snapshot/interaction tools is what the model sees --
it is far cheaper than an image, and its element refs are stable enough to
click reliably. Screenshots are captured for the human watching the dashboard
and are deliberately *never* appended to the model's history. Vision-first
would cost more per step and give the model refs it cannot act on.

Guardrails enforced here:
  * hard step ceiling
  * wall-clock deadline covering LLM calls and tool calls alike
  * domain allowlist checked before every tool call, not just navigation
  * repeated-identical-action detection (nudge, then abort)
  * human approval for sensitive actions (see ``policy.py``)
  * bounded retries with backoff on transient browser failures
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol

from events import (
    AgentEvent,
    ApprovalRequired,
    ApprovalResolved,
    ErrorEvent,
    RunStarted,
    Screenshot,
    Thinking,
    ToolCall,
    ToolResult,
)
from llm import LLMAccessError, LLMClient, ToolCallRequest
from mcp_client import MCPBrowserSession, MCPConnectionError, MCPToolError
from policy import Decision, check_navigation, classify
from prompt_loader import (
    APPROVAL_REJECTED,
    EMPTY_TOOL_RESULT,
    LOOP_NUDGE,
    SYSTEM,
    TASK,
    load,
    render,
)

log = logging.getLogger(__name__)

#: Consecutive identical actions before the agent is nudged, then aborted.
LOOP_NUDGE_AT = 3
LOOP_ABORT_AT = 5

#: Consecutive transport-level tool failures before the session is presumed dead.
TRANSPORT_FAILURE_LIMIT = 3

#: Retry schedule for transient tool failures, in seconds.
RETRY_BACKOFF = (0.5, 1.5, 3.0)

#: Minimum interval between streamed `thinking` updates, to keep the WebSocket
#: from carrying one message per token.
THINKING_FLUSH_INTERVAL = 0.2


PAGE_URL_RE = re.compile(r"(?:Page URL|url)\s*:\s*(https?://\S+)", re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Collaborator protocols (implemented in runner.py, faked in tests)
# ---------------------------------------------------------------------------


class EventSink(Protocol):
    def reserve_seq(self) -> int: ...
    async def emit(self, event: AgentEvent) -> None: ...
    async def save_screenshot(
        self, data: bytes, *, seq: int, mime: str = "image/png"
    ) -> tuple[str, str] | None:
        """Persist an image; return ``(artifact_id, url)`` or ``None``."""
        ...


class ApprovalGate(Protocol):
    async def request(
        self, approval_id: str, timeout: float
    ) -> tuple[Literal["approved", "rejected", "timeout"], str | None]: ...

    async def on_pause(self) -> None: ...
    async def on_resume(self) -> None: ...


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunOptions:
    max_steps: int = 30
    timeout_seconds: float = 300.0
    allowed_domains: list[str] = field(default_factory=list)
    require_approval: bool = True
    approval_timeout_seconds: float = 300.0
    screenshot_every_step: bool = True
    max_tool_result_chars: int = 20_000
    max_history_messages: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "allowed_domains": self.allowed_domains,
            "require_approval": self.require_approval,
            "approval_timeout_seconds": self.approval_timeout_seconds,
            "screenshot_every_step": self.screenshot_every_step,
        }


@dataclass(slots=True)
class AgentSpec:
    run_id: str
    task: str
    start_url: str | None = None
    options: RunOptions = field(default_factory=RunOptions)


@dataclass(slots=True)
class AgentOutcome:
    status: Literal["succeeded", "failed", "cancelled"]
    steps: int
    duration_ms: int
    summary: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class BudgetExhausted(RuntimeError):
    """Step ceiling or wall-clock deadline hit."""


class LoopDetected(RuntimeError):
    """The agent repeated the same action too many times."""


class PolicyViolation(RuntimeError):
    """A hard policy gate (currently the domain allowlist) refused an action."""


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class BrowserAgent:
    def __init__(
        self,
        spec: AgentSpec,
        mcp: MCPBrowserSession,
        llm: LLMClient,
        sink: EventSink,
        approvals: ApprovalGate,
    ) -> None:
        self.spec = spec
        self.mcp = mcp
        self.llm = llm
        self.sink = sink
        self.approvals = approvals

        self.messages: list[dict[str, Any]] = []
        self.step = 0
        self._recent_actions: deque[str] = deque(maxlen=LOOP_ABORT_AT + 1)
        self._transport_failures = 0
        self._last_page_url: str | None = None
        self._started_at = 0.0
        self._deadline = float("inf")

    # -- public entry point -------------------------------------------------
    async def run(self) -> AgentOutcome:
        options = self.spec.options
        loop = asyncio.get_running_loop()
        self._started_at = loop.time()
        self._deadline = self._started_at + options.timeout_seconds

        await self.sink.emit(
            RunStarted(
                run_id=self.spec.run_id,
                seq=self.sink.reserve_seq(),
                task=self.spec.task,
                start_url=self.spec.start_url,
                options=options.to_dict(),
                tools=self.mcp.tool_names,
            )
        )

        try:
            if self.spec.start_url:
                await self._navigate_to_start()

            self.messages = [{"role": "user", "content": self._initial_prompt()}]
            return await self._loop()

        except asyncio.CancelledError:
            raise
        except BudgetExhausted as exc:
            return await self._fail("budget_exhausted", str(exc))
        except LoopDetected as exc:
            return await self._fail("loop_detected", str(exc))
        except PolicyViolation as exc:
            return await self._fail("allowlist_blocked", str(exc))
        except LLMAccessError as exc:
            # A configuration problem, not a crash. Say so plainly.
            return await self._fail("llm_unavailable", str(exc))
        except MCPConnectionError as exc:
            return await self._fail("mcp_unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001 - anything else fails the run cleanly
            log.exception("agent run crashed", extra={"run_id": self.spec.run_id})
            return await self._fail("internal_error", f"{type(exc).__name__}: {exc}")

    # -- main loop ----------------------------------------------------------
    async def _loop(self) -> AgentOutcome:
        options = self.spec.options

        while True:
            self._check_deadline()
            if self.step >= options.max_steps:
                raise BudgetExhausted(
                    f"Step budget of {options.max_steps} exhausted before the task finished."
                )
            self.step += 1

            turn = await self._llm_turn()
            self.messages.append({"role": "assistant", "content": turn.raw_content})

            if not turn.wants_tools:
                return await self._succeed(turn.text)

            tool_results: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                tool_results.append(await self._handle_tool_call(call))

            self.messages.append({"role": "user", "content": tool_results})
            self._trim_history()

            if options.screenshot_every_step:
                await self._capture_screenshot()

    async def _llm_turn(self):
        """One streamed LLM turn, with `thinking` events pushed as text arrives."""
        seq = self.sink.reserve_seq()
        buffer: list[str] = []
        last_flush = 0.0
        loop = asyncio.get_running_loop()

        async def on_delta(chunk: str) -> None:
            nonlocal last_flush
            buffer.append(chunk)
            now = loop.time()
            if now - last_flush >= THINKING_FLUSH_INTERVAL:
                last_flush = now
                await self.sink.emit(
                    Thinking(
                        run_id=self.spec.run_id,
                        seq=seq,
                        step=self.step,
                        text="".join(buffer),
                        done=False,
                    )
                )

        turn = await self.llm.run_turn(
            system=load(SYSTEM),
            messages=self.messages,
            tools=self.mcp.anthropic_tools(),
            on_text_delta=on_delta,
            timeout=self._remaining(),
        )

        # The final emit reuses `seq`, so the store upserts a single row and a
        # reconnecting client replays one complete thinking block.
        await self.sink.emit(
            Thinking(
                run_id=self.spec.run_id,
                seq=seq,
                step=self.step,
                text=turn.text or "".join(buffer),
                done=True,
            )
        )
        return turn

    # -- tool execution -----------------------------------------------------
    async def _handle_tool_call(self, call: ToolCallRequest) -> dict[str, Any]:
        options = self.spec.options
        decision = classify(call.name, call.input, allowlist=options.allowed_domains)

        await self.sink.emit(
            ToolCall(
                run_id=self.spec.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step,
                call_id=call.id,
                name=call.name,
                arguments=call.input,
                sensitive=decision.sensitive,
            )
        )

        repeats = self._count_repeats(call)
        if repeats >= LOOP_ABORT_AT:
            raise LoopDetected(
                f"The agent called {call.name!r} with identical arguments "
                f"{repeats} times in a row; aborting to avoid an infinite loop."
            )
        if repeats >= LOOP_NUDGE_AT:
            return await self._refuse(
                call, render(LOOP_NUDGE, tool_name=call.name, repeats=repeats)
            )

        # Hard allowlist gate. With approvals enabled a human may override it;
        # with approvals disabled it is an outright refusal.
        navigation = check_navigation(call.name, call.input, options.allowed_domains)
        if not navigation.allowed and not options.require_approval:
            await self._emit_error("allowlist_blocked", navigation.reason, recoverable=True)
            return await self._refuse(call, navigation.reason)

        if decision.sensitive and options.require_approval:
            approved = await self._await_approval(call, decision)
            if approved is not True:
                return await self._refuse(
                    call, render(APPROVAL_REJECTED, decision=approved)
                )

        return await self._invoke_with_retry(call)

    async def _invoke_with_retry(self, call: ToolCallRequest) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, len(RETRY_BACKOFF) + 1):
            self._check_deadline()
            try:
                outcome = await self.mcp.call_tool(
                    call.name, call.input, timeout=min(self._remaining(), self.mcp.config.tool_timeout)
                )
            except MCPToolError as exc:
                last_error = exc
                self._transport_failures += 1
                if self._transport_failures >= TRANSPORT_FAILURE_LIMIT:
                    raise MCPConnectionError(
                        "The MCP server stopped responding after "
                        f"{self._transport_failures} consecutive transport failures: {exc}"
                    ) from exc
                if attempt < len(RETRY_BACKOFF):
                    delay = RETRY_BACKOFF[attempt - 1]
                    log.warning(
                        "tool call failed, retrying",
                        extra={"tool": call.name, "attempt": attempt, "delay": delay,
                               "error": str(exc), "run_id": self.spec.run_id},
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            else:
                self._transport_failures = 0
                return await self._finish_tool_call(call, outcome, attempt)

        # Every attempt failed at the transport level. Capture what we can and
        # hand the failure to the model as a tool error rather than dying.
        message = f"Tool {call.name!r} failed after {len(RETRY_BACKOFF)} attempts: {last_error}"
        artifact_id = await self._capture_failure_context()
        await self._emit_error("tool_failed", message, recoverable=True, artifact_id=artifact_id)
        return await self._refuse(call, message, attempts=len(RETRY_BACKOFF))

    async def _finish_tool_call(
        self, call: ToolCallRequest, outcome: Any, attempts: int
    ) -> dict[str, Any]:
        options = self.spec.options
        text = outcome.text
        if outcome.structured and not text:
            text = json.dumps(outcome.structured)[: options.max_tool_result_chars]

        truncated = len(text) > options.max_tool_result_chars
        if truncated:
            text = (
                text[: options.max_tool_result_chars]
                + f"\n\n[... truncated at {options.max_tool_result_chars} characters ...]"
            )

        match = PAGE_URL_RE.search(text)
        if match:
            self._last_page_url = match.group(1)

        await self.sink.emit(
            ToolResult(
                run_id=self.spec.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step,
                call_id=call.id,
                name=call.name,
                ok=not outcome.is_error,
                duration_ms=outcome.duration_ms,
                text=text,
                truncated=truncated,
                attempts=attempts,
            )
        )

        # Images returned by a tool are shown to the human, not fed to the model.
        for mime, data in outcome.images:
            await self._emit_screenshot(data, mime=mime, caption=f"result of {call.name}")

        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": text or load(EMPTY_TOOL_RESULT),
            "is_error": bool(outcome.is_error),
        }

    async def _refuse(
        self, call: ToolCallRequest, message: str, *, attempts: int = 0
    ) -> dict[str, Any]:
        """Report a tool call that was never executed (or that failed for good).

        The refusal is emitted as a failed ``tool_result`` so the timeline
        always pairs a call with an outcome, and is handed back to the model as
        a tool error so it can adapt.
        """
        await self.sink.emit(
            ToolResult(
                run_id=self.spec.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step,
                call_id=call.id,
                name=call.name,
                ok=False,
                duration_ms=0,
                text=message,
                attempts=attempts,
            )
        )
        return self._error_result(call, message)

    def _error_result(self, call: ToolCallRequest, message: str) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": message,
            "is_error": True,
        }

    # -- approvals ----------------------------------------------------------
    async def _await_approval(self, call: ToolCallRequest, decision: Decision) -> bool | str:
        options = self.spec.options
        approval_id = uuid.uuid4().hex
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=options.approval_timeout_seconds)
        ).isoformat()

        await self.approvals.on_pause()
        await self.sink.emit(
            ApprovalRequired(
                run_id=self.spec.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step,
                approval_id=approval_id,
                call_id=call.id,
                name=call.name,
                arguments=call.input,
                reason=decision.reason or "This action was classified as sensitive.",
                categories=decision.category_values,
                expires_at=expires_at,
            )
        )

        decision_value, note = await self.approvals.request(
            approval_id, options.approval_timeout_seconds
        )

        await self.sink.emit(
            ApprovalResolved(
                run_id=self.spec.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step,
                approval_id=approval_id,
                decision=decision_value,
                note=note,
            )
        )
        await self.approvals.on_resume()

        if decision_value == "approved":
            return True
        return "rejected" if decision_value == "rejected" else "did not respond to"

    # -- observation --------------------------------------------------------
    async def _navigate_to_start(self) -> None:
        url = self.spec.start_url or ""
        navigation = check_navigation(
            "browser_navigate", {"url": url}, self.spec.options.allowed_domains
        )
        if not navigation.allowed:
            raise PolicyViolation(f"Refusing to open the start URL. {navigation.reason}")

        tool = self.mcp.find_tool("browser_navigate", "navigate", contains=("navigate", "goto"))
        if tool is None:
            log.warning("no navigation tool exposed by the MCP server; skipping start URL")
            return

        call = ToolCallRequest(id=f"bootstrap-{uuid.uuid4().hex[:8]}", name=tool, input={"url": url})
        self.step = 0
        await self.sink.emit(
            ToolCall(
                run_id=self.spec.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step,
                call_id=call.id,
                name=call.name,
                arguments=call.input,
                sensitive=False,
            )
        )
        await self._invoke_with_retry(call)
        if self.spec.options.screenshot_every_step:
            await self._capture_screenshot()

    async def _capture_screenshot(self, caption: str | None = None) -> str | None:
        """Take a dashboard screenshot. Never fails the run."""
        tool = self.mcp.find_tool(
            "browser_take_screenshot", "browser_screenshot", contains=("screenshot",)
        )
        if tool is None:
            return None
        try:
            outcome = await self.mcp.call_tool(tool, {}, timeout=min(20.0, max(self._remaining(), 1.0)))
        except (MCPToolError, MCPConnectionError) as exc:
            log.debug("screenshot failed", extra={"error": str(exc), "run_id": self.spec.run_id})
            return None

        for mime, data in outcome.images:
            return await self._emit_screenshot(data, mime=mime, caption=caption)
        return None

    async def _emit_screenshot(
        self, data: bytes, *, mime: str = "image/png", caption: str | None = None
    ) -> str | None:
        seq = self.sink.reserve_seq()
        saved = await self.sink.save_screenshot(data, seq=seq, mime=mime)
        if saved is None:
            return None
        artifact_id, url = saved
        await self.sink.emit(
            Screenshot(
                run_id=self.spec.run_id,
                seq=seq,
                step=self.step,
                artifact_id=artifact_id,
                url=url,
                caption=caption,
                page_url=self._last_page_url,
            )
        )
        return artifact_id

    async def _capture_failure_context(self) -> str | None:
        """Best-effort screenshot + console log at the moment of failure."""
        artifact_id = await self._capture_screenshot(caption="failure")
        console_tool = self.mcp.find_tool(
            "browser_console_messages", contains=("console", "network_requests")
        )
        if console_tool:
            try:
                outcome = await self.mcp.call_tool(console_tool, {}, timeout=10.0)
                log.warning(
                    "captured page diagnostics after tool failure",
                    extra={"run_id": self.spec.run_id, "console": outcome.text[:2000]},
                )
            except (MCPToolError, MCPConnectionError):
                pass
        return artifact_id

    # -- bookkeeping --------------------------------------------------------
    def _initial_prompt(self) -> str:
        """The first user turn. Wording lives in ``prompts/task.md``."""
        options = self.spec.options
        return render(
            TASK,
            task=self.spec.task,
            allowed_domains=", ".join(options.allowed_domains)
            or "(none -- navigation is blocked)",
            max_steps=options.max_steps,
            timeout_seconds=f"{options.timeout_seconds:.0f}s",
            start_url_line=(
                f"The browser has already been opened at {self.spec.start_url}."
                if self.spec.start_url
                else ""
            ),
        )

    def _count_repeats(self, call: ToolCallRequest) -> int:
        signature = f"{call.name}:{json.dumps(call.input, sort_keys=True, default=str)}"
        self._recent_actions.append(signature)
        count = 0
        for previous in reversed(self._recent_actions):
            if previous == signature:
                count += 1
            else:
                break
        return count

    def _trim_history(self) -> None:
        """Drop the oldest assistant/tool-result pairs once history gets long.

        Message 0 (the task) is always kept, and messages are only ever removed
        in assistant+tool_result pairs so no ``tool_use`` block is ever left
        without its matching ``tool_result``.
        """
        limit = self.spec.options.max_history_messages
        while len(self.messages) > limit and len(self.messages) > 3:
            del self.messages[1:3]
            log.debug("trimmed agent history", extra={"run_id": self.spec.run_id})

    def _remaining(self) -> float:
        return max(self._deadline - asyncio.get_running_loop().time(), 0.0)

    def _check_deadline(self) -> None:
        if self._remaining() <= 0:
            raise BudgetExhausted(
                f"Wall-clock budget of {self.spec.options.timeout_seconds:.0f}s exhausted."
            )

    def _elapsed_ms(self) -> int:
        return int((asyncio.get_running_loop().time() - self._started_at) * 1000)

    async def _emit_error(
        self, kind: str, message: str, *, recoverable: bool = False, artifact_id: str | None = None
    ) -> None:
        await self.sink.emit(
            ErrorEvent(
                run_id=self.spec.run_id,
                seq=self.sink.reserve_seq(),
                step=self.step,
                kind=kind,
                message=message,
                recoverable=recoverable,
                artifact_id=artifact_id,
            )
        )

    # -- terminal states ----------------------------------------------------
    async def _succeed(self, text: str) -> AgentOutcome:
        summary, data = _split_answer(text)
        return AgentOutcome(
            status="succeeded",
            steps=self.step,
            duration_ms=self._elapsed_ms(),
            summary=summary,
            result={"answer": summary, "data": data},
        )

    async def _fail(self, kind: str, message: str) -> AgentOutcome:
        artifact_id = None
        if kind in ("tool_failed", "loop_detected", "budget_exhausted", "internal_error"):
            artifact_id = await self._capture_failure_context()
        await self._emit_error(kind, message, recoverable=False, artifact_id=artifact_id)
        return AgentOutcome(
            status="failed",
            steps=self.step,
            duration_ms=self._elapsed_ms(),
            error=message,
        )


def _split_answer(text: str) -> tuple[str, Any]:
    """Split a final answer into prose plus any fenced JSON payload."""
    text = (text or "").strip()
    match = JSON_FENCE_RE.search(text)
    if not match:
        return text, None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return text, None
    prose = (text[: match.start()] + text[match.end():]).strip()
    return prose or text, data
