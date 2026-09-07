"""One agent session's tool layer: dispatch, refs, redaction, audit.

This is everything an agent needs except the agent. There is no model here and
no loop -- a graph arrives in a later phase and drives this object -- which is
deliberate: the part that decides what may happen is testable on its own, and a
test of the guard should not need a model or a browser.

Three responsibilities, in the order a call meets them:

1. **Track what the page reported.** Every result carries refs, not just an
   explicit snapshot's, because an action's result includes the page as it now
   stands and a ref that just appeared is the one the next call needs.
2. **Guard.** See ``guardrails/guard.py``. A refusal comes back as a tool error so the
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
from usecase import MissingValue, render_template

from snapshot import Snapshot, parse as parse_snapshot

from .guardrails import DISTILS_TO, GuardContext, PERCEPTION, guard, offered
from .marks import Described, Marks, describe_element
from .providers.base import BrowserProvider, MCPSession, ToolResult, ToolSpec
from .tools import TOOLS, ToolDef

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
    #: How many elements the locator's leading rung matched *at record time*.
    #: A ref click always runs -- refs are position-specific, not name-based,
    #: so the agent's own click is never ambiguous -- but the durable locator
    #: distilled from it (role/name, no ref) is checked the same way a mark
    #: already is. A step built from a call where this is not 1 replays
    #: against a live page the same locator will match many of, and the
    #: engine will correctly refuse to guess which one was meant; the point of
    #: carrying this through to `distil.py` is saying so *now*, on the review
    #: screen, rather than leaving it to be discovered days later when a batch
    #: fails on a step nobody flagged.
    match_count: int = 1

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
            "match_count": self.match_count,
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
        secret_values: dict[str, str] | None = None,
        extra: dict[str, BrowserProvider] | None = None,
    ) -> None:
        self.provider = provider
        self.allowed_domains = tuple(allowed_domains)
        self.may_write = may_write
        self.redactor = redactor or NULL_REDACTOR
        self.recorder = recorder
        #: Other MCP servers, keyed by a short name -- from the tool-server
        #: registry, resolved by whoever starts this session. Their tools are
        #: shown to the model beside the browser's and dispatched the same
        #: way, but never distilled into a step: `DISTILS_TO` only knows
        #: Playwright's tool names, and a prefixed name can never match it.
        self.extra = dict(extra or {})
        #: Real credential values, keyed by slot. Never shown to the model and
        #: never stored: the model is told to type the literal text
        #: ``{{secret.slot}}``, and this is what turns that placeholder into a
        #: real value in the one call that actually reaches a browser. Without
        #: this there was no route from a bound credential to a typed
        #: character at all -- an agent asked to "log in with the credentials
        #: provided" had nothing to type and fabricated "admin" / "password",
        #: twice, in two different real sessions.
        self.secret_values = dict(secret_values or {})
        self.calls: list[ToolCallRecord] = []

        self._session: MCPSession | None = None
        self._specs: list[ToolSpec] = []
        #: Extra servers' live sessions, keyed the same as `self.extra`.
        self._extra_sessions: dict[str, MCPSession] = {}
        #: Extra servers' tools, keyed by their prefixed name --
        #: "{server}.{tool}" -- so a name can never collide with a browser
        #: tool or a mark tool.
        self._extra_specs: dict[str, ToolSpec] = {}
        #: Which server a prefixed name belongs to, for dispatch.
        self._extra_owner: dict[str, str] = {}
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
        try:
            for server_name, provider in self.extra.items():
                session = await provider.open()
                self._extra_sessions[server_name] = session
                for spec in offered(await session.list_tools()):
                    prefixed = f"{server_name}.{spec.name}"
                    self._extra_specs[prefixed] = ToolSpec(
                        name=prefixed,
                        description=spec.description,
                        input_schema=spec.input_schema,
                        annotations=spec.annotations,
                    )
                    self._extra_owner[prefixed] = server_name
        except Exception:
            # A later server failing to open must not leak an earlier one, or
            # the browser this session already started.
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._session = None
        self._extra_sessions = {}
        for provider in self.extra.values():
            await provider.close()
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
            ToolSpec(name=t.name, description=t.description, input_schema=t.input_schema)
            for t in TOOLS.values()
            if t.handler is not None  # `finish` is answered elsewhere; see tools/finish.py
        ]
        return [*self._specs, *self._extra_specs.values(), *ours]

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
            available=(
                frozenset(spec.name for spec in self._specs)
                | frozenset(name for name, t in TOOLS.items() if t.handler is not None)
                | frozenset(self._extra_specs)
            ),
            annotations={
                name: spec.annotations for name, spec in self._extra_specs.items()
            },
        )

    def _substitute_secrets(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """A copy of ``arguments`` with every ``{{secret.x}}`` made real.

        Scoped to the argument keys that actually carry typed text --
        ``browser_type``'s ``text``, ``browser_select_option``'s ``values``,
        and each field's ``value`` in ``browser_fill_form``. A raw literal
        with no ``{{...}}`` in it -- a guessed value, or ordinary text --
        passes through unchanged: this only ever narrows what a placeholder
        means, it never invents one.

        Raises ``MissingValue`` -- caught by the caller -- when the model
        names a slot this session has no value for, rather than typing the
        literal placeholder text into a live page.
        """
        if not self.secret_values:
            return arguments

        def sub(value: Any) -> Any:
            if not isinstance(value, str) or "{{" not in value:
                return value
            return render_template(value, inputs={}, secrets=self.secret_values, env={})

        out = dict(arguments)
        if "text" in out:
            out["text"] = sub(out["text"])
        if isinstance(out.get("values"), list):
            out["values"] = [sub(v) for v in out["values"]]
        if isinstance(out.get("fields"), list):
            out["fields"] = [
                {**f, "value": sub(f.get("value"))} if isinstance(f, dict) else f
                for f in out["fields"]
            ]
        return out

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
        tool = TOOLS.get(name)
        if tool is not None and tool.handler is not None:
            result = self._mark(tool, arguments)
            return await self._finish(name, arguments, result, verdict, started)

        if name in self._extra_owner:
            # A tool from a registered server rather than the browser: same
            # guard, same secret substitution, same audit trail -- but no ref
            # or snapshot bookkeeping, since nothing about this call resolves
            # against a page. `_finish` below leaves `action`/`locators` at
            # their defaults for exactly that reason, which is also what
            # keeps a call like this out of a distilled use case.
            return await self._call_extra(name, arguments, verdict, started)

        if self._session is None:
            raise RuntimeError(
                "The tool session is not open. Use it as an async context manager."
            )

        # Resolved here, before dispatch, and not afterwards. After the call
        # the page has re-rendered and this ref names something else or
        # nothing at all.
        described = self._describe(str(arguments.get("target") or ""))

        # `arguments` -- the placeholder-bearing version -- is what gets
        # recorded, redacted and shown back to the model. `dispatched` is a
        # copy with every `{{secret.x}}` turned into the real value, and it is
        # the only one that ever reaches a browser. Keeping the two apart is
        # what lets the trajectory read "typed {{secret.email}}" -- which is
        # exactly the step a replay should carry -- while the actual sign-in
        # still succeeds against a real account.
        try:
            dispatched = self._substitute_secrets(arguments)
        except MissingValue as exc:
            known = ", ".join(sorted(self.secret_values)) or "(none bound to this session)"
            return await self._finish(
                name, arguments,
                ToolResult.failed(
                    f"{{{{{exc.args[0]}}}}} is not a credential this session has. "
                    f"Bound slots: {known}. Use one of those, exactly as "
                    "{{secret.slot}} -- do not type a guessed value."
                ),
                verdict, started, refused=True,
            )

        try:
            result = await self._session.call(name, dispatched)
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

    async def _call_extra(
        self, name: str, arguments: dict[str, Any], verdict: Any, started: float
    ) -> ToolResult:
        """Dispatch to a registered server rather than the browser.

        Secret substitution is scoped to the same argument keys the browser
        path knows (`text`, `values`, `fields`) -- see `_substitute_secrets`.
        A server whose own argument shape carries a credential under some
        other key gets the model's literal `{{secret.x}}` text unsubstituted,
        which fails that call rather than leaking anything: nothing here
        invents a value it was not told to use.
        """
        server = self._extra_owner[name]
        tool_name = name.split(".", 1)[1]

        try:
            dispatched = self._substitute_secrets(arguments)
        except MissingValue as exc:
            known = ", ".join(sorted(self.secret_values)) or "(none bound to this session)"
            return await self._finish(
                name, arguments,
                ToolResult.failed(
                    f"{{{{{exc.args[0]}}}}} is not a credential this session has. "
                    f"Bound slots: {known}. Use one of those, exactly as "
                    "{{secret.slot}} -- do not type a guessed value."
                ),
                verdict, started, refused=True,
            )

        try:
            result = await self._extra_sessions[server].call(tool_name, dispatched)
        except Exception as exc:  # noqa: BLE001 - a dead server is a tool error
            log.warning("tool call raised", extra={"tool": name, "error": str(exc)})
            result = ToolResult.failed(f"{name} failed: {exc}")

        return await self._finish(name, arguments, result, verdict, started)

    # -- our own tools ------------------------------------------------------
    def _mark(self, tool: ToolDef, arguments: dict[str, Any]) -> ToolResult:
        """Answer one of our own tools. No browser, no model, no I/O.

        Each tool's own logic lives in `agent/tools/<name>.py`; this is only
        the dispatch -- resolve a ref first when the tool needs one, then hand
        off to the handler it registered.
        """
        described = self._describe(str(arguments.get("ref") or "")) if tool.needs_ref else None
        assert tool.handler is not None  # `finish` never reaches here
        return tool.handler(self.marks, described, self._seq, arguments)

    def _describe(self, ref: str) -> Described:
        if self._snapshot is None:
            return Described(ref=ref, role="", name="", matches=0)
        return describe_element(self._snapshot, ref)

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
            match_count=described.matches if described else 1,
        )
        self.calls.append(record)
        if self.recorder is not None:
            await self.recorder(record)
        return result


__all__ = ["AgentToolSession", "Recorder", "ToolCallRecord"]
