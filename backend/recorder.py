"""Recording a workflow by watching someone do it once.

``npx playwright codegen`` opens a real browser window, the user performs the
task by hand, and closing the window leaves a script behind. This module owns
that subprocess; :mod:`codegen` reads what it wrote.

Replacing the agent recorder with this is the change the whole redesign turns
on. A model driving a browser to *discover* a workflow cost ~247,000 input
tokens for one recording, spent most of them re-sending accessibility
snapshots, and failed a third of its actions on the way. The user already knows
how to do the task -- they do it every day -- so the software's job is to watch,
not to work it out.

Three properties this has to keep
---------------------------------
**The subprocess never outlives the process that spawned it.** It is killed in
a ``finally`` that also runs on cancellation, and again at shutdown. A leaked
headed browser is a window nobody owns, holding a session somebody was signed
into.

**Recording is local-only.** A codegen window needs a display, so this is off
by default in any deployment that has none, and ``RECORDER_ENABLED`` says so
explicitly rather than leaving it to fail at spawn time with an X11 error. The
EKS deployment runs replay workers; recording is not part of it.

**The script it writes is untrusted.** :mod:`codegen` parses it with ``ast``
and never executes it. This module only reads bytes off disk.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codegen import CodegenError, Recording, parse, summarise, urls_of

log = logging.getLogger(__name__)

Status = Literal["recording", "parsing", "ready", "failed", "cancelled"]

#: How often the poller checks whether the window has been closed.
POLL_SECONDS = 0.5


class RecorderUnavailable(RuntimeError):
    """Recording is switched off, or the tooling is not installed."""


@dataclass
class Session:
    """One recording in progress, or one that has finished."""

    id: str
    workspace_id: str
    owner_id: str | None
    owner_email: str
    start_url: str
    name: str
    status: Status = "recording"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    recording: Recording | None = None
    #: The directory holding codegen's output, removed when the session is
    #: discarded. Kept out of the response.
    workdir: Path | None = None
    process: subprocess.Popen | None = None
    task: asyncio.Task | None = None

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "recording_id": self.id,
            "status": self.status,
            "name": self.name,
            "start_url": self.start_url,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "owner_email": self.owner_email,
        }
        if self.recording is not None:
            body.update(
                {
                    "summary": summarise(self.recording),
                    "domains": urls_of(self.recording),
                    "steps": [
                        {
                            "id": step.id,
                            "action": step.action,
                            "url": step.url,
                            "value": step.value,
                            "locator": step.locators[0].describe() if step.locators else "",
                        }
                        for step in self.recording.steps
                    ],
                    "assertions": [
                        check.model_dump(mode="json") for check in self.recording.assertions
                    ],
                    "typed": list(self.recording.typed),
                    "unsupported": [item.to_dict() for item in self.recording.unsupported],
                }
            )
        return body


class Recorder:
    """Spawns codegen, waits for the window to close, parses the result.

    One instance per process, held on ``app.state``. Sessions live in memory:
    a recording is a person sitting in front of a browser, so it does not
    outlive the process they started it from, and pretending otherwise by
    persisting it would promise a resume that cannot work.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        command: str = "",
        browser: str = "chromium",
        timeout_seconds: float = 1800.0,
    ) -> None:
        self.enabled = enabled
        #: Blank means "the Playwright that is already installed here", which
        #: is the right answer and was not the original one. This used to shell
        #: out to `npx playwright`, which needed Node, needed a separate
        #: install, and could resolve to a *different* Playwright version than
        #: the one the parser was written against -- codegen's output is not a
        #: stable contract, so a version skew between the recorder and the
        #: parser is a real failure mode rather than a theoretical one.
        #:
        #: The Python package ships the same CLI. Using this interpreter to run
        #: it means the recorder and the engine are the same Playwright, by
        #: construction, and recording needs no Node at all.
        self.command = command or f"{sys.executable} -m playwright"
        self.browser = browser
        self.timeout_seconds = timeout_seconds
        self._sessions: dict[str, Session] = {}

    def _argv_prefix(self) -> list[str]:
        r"""The configured command, split into argv.

        ``shlex`` rather than ``str.split`` because a Windows install puts npx
        under ``C:\Program Files\nodejs``, and splitting that on spaces
        produces two arguments and a confusing "not on PATH". ``posix=False``
        keeps backslashes intact -- they are path separators here, not escapes
        -- and the surrounding quotes it leaves behind are stripped after.
        """
        return [token.strip('"') for token in shlex.split(self.command, posix=False)]

    # -- lifecycle ----------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        """Whether a recording could start, and why not if it could not."""
        if not self.enabled:
            return False, (
                "Recording is switched off here. It opens a browser window, so it only "
                "works on a machine with a display -- run the backend locally to record."
            )
        executable = self._argv_prefix()[0]
        if shutil.which(executable) is None:
            return False, (
                f"{executable!r} could not be found. Playwright is missing from this "
                "environment: install the requirements and run "
                "`python -m playwright install chromium`."
            )
        return True, ""

    async def start(
        self,
        *,
        start_url: str,
        name: str,
        workspace_id: str,
        owner_id: str | None,
        owner_email: str,
    ) -> Session:
        ok, reason = self.available()
        if not ok:
            raise RecorderUnavailable(reason)

        session = Session(
            id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            owner_id=owner_id,
            owner_email=owner_email,
            start_url=start_url,
            name=name,
        )
        session.workdir = Path(tempfile.mkdtemp(prefix="recording-"))
        self._sessions[session.id] = session

        output = session.workdir / "recorded.py"
        argv = [
            *self._argv_prefix(),
            "codegen",
            "--target=python-async",
            f"--output={output}",
            f"--browser={self.browser}",
            start_url,
        ]

        log.info(
            "starting recorder",
            extra={"recording_id": session.id, "start_url": start_url},
        )
        # A plain Popen, waited on in a thread. See this module's note on why
        # the asyncio flavour is not worth its platform baggage here.
        session.process = subprocess.Popen(  # noqa: S603 - argv, never a shell
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        session.task = asyncio.create_task(
            self._wait(session, output), name=f"recorder-{session.id}"
        )
        # Nothing awaits that task -- the request returns as soon as the window
        # is open. So if it dies, its exception is stored on the Task and
        # retrieved by nobody, and the session sits at "recording" for ever
        # while the UI polls it. This turns that silence into a failed status
        # with the reason attached.
        session.task.add_done_callback(lambda task: self._finished(session, task))
        return session

    def _finished(self, session: Session, task: "asyncio.Task") -> None:
        """Make sure a watcher that died leaves a session that says so."""
        if task.cancelled():
            session.status = "cancelled"
        else:
            error = task.exception()
            if error is not None:
                log.error(
                    "the recorder watcher failed",
                    extra={"recording_id": session.id},
                    exc_info=error,
                )
                session.status = "failed"
                session.error = f"{type(error).__name__}: {error}"

        if session.status == "recording":
            # It returned without deciding anything, which should be
            # impossible; saying so beats leaving the UI polling for ever.
            session.status = "failed"
            session.error = "the recorder stopped without saying what happened"
        if session.finished_at is None:
            session.finished_at = time.time()

    async def _wait(self, session: Session, output: Path) -> None:
        """Wait for the window to close, then read what it left behind."""
        process = session.process
        assert process is not None
        try:
            try:
                # The blocking wait happens on a worker thread, so the event
                # loop keeps serving the status requests the UI is polling with.
                _, stderr = await asyncio.wait_for(
                    asyncio.to_thread(process.communicate),
                    timeout=self.timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError):
                session.status = "failed"
                session.error = (
                    f"the recording window was still open after "
                    f"{int(self.timeout_seconds / 60)} minutes, so it was closed."
                )
                return

            if session.status == "cancelled":
                return

            session.status = "parsing"
            if not output.exists():
                session.status = "failed"
                detail = (stderr or "").strip().splitlines()
                session.error = (
                    "the recorder wrote nothing. "
                    + (detail[-1] if detail else "The window may have been closed immediately.")
                )
                return

            try:
                session.recording = parse(output.read_text(encoding="utf-8"))
            except CodegenError as exc:
                session.status = "failed"
                session.error = str(exc)
                return

            session.status = "ready"
            log.info(
                "recording ready",
                extra={
                    "recording_id": session.id,
                    "summary": summarise(session.recording),
                },
            )
        except asyncio.CancelledError:
            session.status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - a recorder crash is not a server crash
            log.exception("recorder failed", extra={"recording_id": session.id})
            session.status = "failed"
            session.error = f"{type(exc).__name__}: {exc}"
        finally:
            session.finished_at = time.time()
            await self._terminate(session)

    async def cancel(self, recording_id: str, workspace_id: str) -> bool:
        session = self.get(recording_id, workspace_id)
        if session is None or session.status != "recording":
            return False
        session.status = "cancelled"
        await self._terminate(session)
        return True

    async def _terminate(self, session: Session) -> None:
        """Close the window if it is still open. Safe to call twice."""
        process = session.process
        if process is None or process.poll() is not None:
            return

        def stop() -> None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except (ProcessLookupError, OSError):
                pass

        await asyncio.to_thread(stop)

    # -- reading ------------------------------------------------------------
    def get(self, recording_id: str, workspace_id: str) -> Session | None:
        """One session, scoped. A recording belongs to the tenant that started
        it, so an id from another workspace is simply not found."""
        session = self._sessions.get(recording_id)
        if session is None or session.workspace_id != workspace_id:
            return None
        return session

    def list(self, workspace_id: str) -> list[Session]:
        return [s for s in self._sessions.values() if s.workspace_id == workspace_id]

    def discard(self, recording_id: str, workspace_id: str) -> bool:
        session = self.get(recording_id, workspace_id)
        if session is None:
            return False
        self._sessions.pop(recording_id, None)
        if session.workdir is not None:
            shutil.rmtree(session.workdir, ignore_errors=True)
        return True

    async def shutdown(self) -> None:
        """Close every window this process opened.

        A headed browser that outlives its backend is a window nobody owns,
        very possibly signed into something.
        """
        for session in list(self._sessions.values()):
            if session.task is not None and not session.task.done():
                session.task.cancel()
            await self._terminate(session)
            if session.workdir is not None:
                shutil.rmtree(session.workdir, ignore_errors=True)
        self._sessions.clear()


__all__ = ["Recorder", "RecorderUnavailable", "Session", "Status"]
