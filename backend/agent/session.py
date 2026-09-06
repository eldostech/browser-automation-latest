"""One agent session's tool layer: dispatch, refs, redaction, audit.

This is everything an agent needs except the agent. There is no model here and
no loop -- a graph arrives in a later phase and drives this object -- which is
deliberate: the part that decides what may happen is testable on its own, and a
test of the guard should not need a model or a browser.

Three responsibilities, in the order a call meets them:

1. **Track what the page reported.** Every result carries refs, not just an
   explicit snapshot's, because an action's result includes the page as it now
   stands and a ref that just appeared is the one the next call needs.
2. **Guard.** See ``tools.py``. A refusal comes back as a tool error so the
   agent can choose differently, and is audited exactly like a call that ran.
3. **Redact, then record.** The redactor is registered with the run's secrets
   before anything is emitted -- an existing invariant of this codebase, and
   one an agent makes easier to break because it types credentials into pages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

from redaction import NULL_REDACTOR, Redactor

from snapshot import Snapshot, parse as parse_snapshot

from .marks import MARK_TOOLS, Described, Marks, describe_element
from .provider import BrowserProvider, MCPSession, ToolResult, ToolSpec
from .tools import DISTILS_TO, GuardContext, PERCEPTION, guard, offered

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolCallRecord:
    """One attempted call, whether or not it ran.

    A trajectory is built from these, and so is the audit trail. Refusals are
    kept: "the agent tried to leave the allowlist and was stopped" is the entry
    somebody will actually want, and an audit that records only what succeeded
    describes a system nobody has to trust.
    """

    seq: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    #: The server's own rendering, or the refusal. Redacted.
    detail: str = ""
    refused: bool = False
    needs_approval: bool = False
    category: str = ""
    duration_ms: int = 0
    #: Which ``Step.action`` this would distil into, or "" for perception.
    action: str = ""
    #: The durable locator ladder for whatever this call acted on, resolved
    #: **before** it ran. That timing is the whole point: a ref is an index
    #: into the snapshot it came from, and by the time the call returns the
    #: page has moved on and the ref means something else or nothing. Captured
    #: for every acting call rather than only for marked ones, because
    #: distillation needs a locator for each step and not just the interesting
    #: ones.
    locators: list[dict[str, Any]] = field(default_factory=list)
    #: How a person would read that locator, for the review screen.
    element: str = ""
    #: Where the browser was when this call finished. Read off the server's own
    #: reply, which carries it. Needed because `browser_navigate_back` records
    #: no destination -- the page it landed on is the only thing that says
    #: where a replay should go.
    page_url: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "tool": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "refused": self.refused,
            "needs_approval": self.needs_approval,
            "category": self.category,
            "duration_ms": self.duration_ms,
            "action": self.action,
            "locators": self.locators,
            "element": self.element,
            "page_url": self.page_url,
            "detail": self.detail,
        }


#: Where a record goes. A callable rather than a store, because this package
#: imports neither FastAPI nor ``store``: the caller decides what persistence
#: means, and on AgentCore the caller is on a different machine.
Recorder = Callable[[ToolCallRecord], Awaitable[None]]


class AgentToolSession:
    """The browser, the tools, and the rules -- for the length of one session.

    Use it as an async context manager. The provider is opened on entry and
    closed on exit including on cancellation, because a Node subprocess and a
    Chromium behind it are not things to leak per run.
    """

    def __init__(
        self,
        provider: BrowserProvider,
        *,
        allowed_domains: Iterable[str] = (),
        may_write: bool = False,
        redactor: Redactor | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        self.provider = provider
        self.allowed_domains = tuple(allowed_domains)
        self.may_write = may_write
        self.redactor = redactor or NULL_REDACTOR
        self.recorder = recorder
        self.calls: list[ToolCallRecord] = []

        self._session: MCPSession | None = None
        self._specs: list[ToolSpec] = []
        #: What the agent has declared about the shape of the use case.
        self.marks = Marks()
        #: The page as it was last reported, parsed. Held because every mark
        #: resolves a ref against it, and because re-snapshotting to answer
        #: `describe_element` would both cost a round trip and risk describing
        #: a different page than the one the agent is looking at.
        self._snapshot: Snapshot | None = None
        #: Refs the page has reported. Replaced rather than accumulated: a ref
        #: is only valid for the snapshot it came from, and remembering old
        #: ones would let a stale target through the check that exists to catch
        #: exactly that.
        self._refs: frozenset[str] = frozenset()
        self._seq = 0

    # -- lifecycle ----------------------------------------------------------
    async def __aenter__(self) -> "AgentToolSession":
        self._session = await self.provider.open()
        self._specs = offered(await self._session.list_tools())
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._session = None
        await self.provider.close()

    # -- what the model is shown -------------------------------------------
    @property
    def tools(self) -> list[ToolSpec]:
        """The tool list: the server's, minus refusals, plus ours.

        The marking tools are advertised beside the browser's own because from
        the model's side there is no difference -- it calls a tool and gets an
        answer. That they never reach the browser is this object's business.
        """
        ours = [
            ToolSpec(
                name=name,
                description=spec["description"],
                input_schema={
                    "type": "object",
                    "properties": spec["properties"],
                    "required": spec["required"],
                },
            )
            for name, spec in MARK_TOOLS.items()
        ]
        return [*self._specs, *ours]

    @property
    def snapshot(self) -> Snapshot | None:
        """The page as it was last reported."""
        return self._snapshot

    @property
    def known_refs(self) -> frozenset[str]:
        return self._refs

    def context(self) -> GuardContext:
        return GuardContext(
            allowed_domains=self.allowed_domains,
            known_refs=self._refs,
            may_write=self.may_write,
            available=frozenset(spec.name for spec in self._specs) | frozenset(MARK_TOOLS),
        )

    # -- the one method that matters ---------------------------------------
    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Guard, dispatch, record. Never raises for a refusal.

        A refused call returns an error-shaped result carrying the reason,
        because that is what lets an agent adapt: told it may not leave the
        allowlist, it can look for the link it actually wanted. Told nothing,
        it retries the same call until its budget is gone.
        """
        arguments = dict(arguments or {})
        self._seq += 1
        started = time.monotonic()

        verdict = guard(name, arguments, self.context())
        if not verdict.allowed:
            return await self._finish(
                name, arguments, ToolResult.failed(verdict.reason), verdict,
                started, refused=True,
            )

        # Ours are answered here and never reach the browser. They still pass
        # the guard first, because the ref discipline applies to a mark as much
        # as to a click: marking an element the page is not showing would
        # record a step nothing can replay.
        if name in MARK_TOOLS:
            result = self._mark(name, arguments)
            return await self._finish(name, arguments, result, verdict, started)

        if self._session is None:
            raise RuntimeError(
                "The tool session is not open. Use it as an async context manager."
            )

        # Resolved here, before dispatch, and not afterwards. After the call
        # the page has re-rendered and this ref names something else or
        # nothing at all.
        described = self._describe(str(arguments.get("target") or ""))

        try:
            result = await self._session.call(name, arguments)
        except Exception as exc:  # noqa: BLE001 - a dead server is a tool error
            log.warning("tool call raised", extra={"tool": name, "error": str(exc)})
            result = ToolResult.failed(f"{name} failed: {exc}")

        # Refs are replaced from whatever the page just reported. A result that
        # carries none -- a console dump, say -- leaves the previous set alone,
        # because it did not re-render the page and did not invalidate them.
        if result.refs:
            self._refs = frozenset(result.refs)
            # Parsed once, here, rather than by each mark that needs it. The
            # snapshot the agent is looking at and the one a mark resolves
            # against have to be the same page.
            self._snapshot = parse_snapshot(result.text)

        return await self._finish(
            name, arguments, result, verdict, started, described=described
        )

    # -- our own tools ------------------------------------------------------
    def _mark(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Answer a marking tool. No browser, no model, no I/O."""
        after = self._seq

        if name == "mark_setup_complete":
            return self._say(self.marks.setup_complete(after), "Setup ends here.")
        if name == "begin_row":
            key = str(arguments.get("key") or "")
            return self._say(self.marks.begin_row(after, key), f"Row {key!r} started.")
        if name == "end_row":
            return self._say(self.marks.end_row(after), "Row finished.")

        described = self._describe(str(arguments.get("ref") or ""))
        if name == "describe_element":
            return ToolResult(text=described.as_text(), is_error=described.matches == 0)

        field = {"mark_as_input": "name", "mark_as_output": "column"}.get(name, "slot")
        value = str(arguments.get(field) or "")
        if not value:
            return ToolResult.failed(f"{name} needs a {field}.")
        problem = self.marks.mark_value(name, after, described.ref, value, described)
        return self._say(problem, f"{described.describe_first()} marked as {value!r}.")

    def _describe(self, ref: str) -> Described:
        if self._snapshot is None:
            return Described(ref=ref, role="", name="", matches=0)
        return describe_element(self._snapshot, ref)

    @staticmethod
    def _say(problem: str, done: str) -> ToolResult:
        """A mark either happened or did not, and the reason is the answer."""
        return ToolResult.failed(problem) if problem else ToolResult(text=done)

    async def _finish(
        self,
        name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        verdict: Any,
        started: float,
        *,
        refused: bool = False,
        described: "Described | None" = None,
    ) -> ToolResult:
        record = ToolCallRecord(
            seq=self._seq,
            name=name,
            # Redacted here rather than at the edge: this is the single point
            # every call passes through, and a secret typed into a page would
            # otherwise reach both the trajectory and the audit log.
            arguments=self.redactor.structure(arguments),
            ok=not result.is_error,
            detail=self.redactor.text(result.text)[:4000],
            refused=refused,
            needs_approval=bool(getattr(verdict, "needs_approval", False)),
            category=getattr(verdict, "category", "") or "",
            duration_ms=int((time.monotonic() - started) * 1000),
            action="" if name in PERCEPTION else DISTILS_TO.get(name, ""),
            locators=[
                loc.model_dump(mode="json", exclude_none=True)
                for loc in (described.ladder if described else [])
            ],
            element=described.describe_first() if described and described.ladder else "",
            page_url=self._snapshot.page_url if self._snapshot else "",
        )
        self.calls.append(record)
        if self.recorder is not None:
            await self.recorder(record)
        return result


__all__ = ["AgentToolSession", "Recorder", "ToolCallRecord"]
