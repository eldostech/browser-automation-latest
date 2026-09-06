"""Replay the draft before anyone is told it exists.

The agent does not get to *claim* it recorded something. What it produced is
handed to ``engine.py`` -- the same code that will run the four thousand rows,
which cannot call a model -- and either replays or does not. A draft that does
not replay is a draft that says so, on the review screen, before anybody
publishes it.

**A correction to the design.** It said this runs "in the same browser session,
against the same row". The first half is not possible and, once you see why,
not desirable. The agent's browser lives behind Playwright MCP in a Node
subprocess; the engine drives a Playwright ``Page`` directly. There is no
shared session to reuse.

Verifying from a **cold start** is the stronger check anyway. A use case that
only replays inside the session that recorded it is not a use case: the batch
will open a fresh browser, sign in, and work through a row, and that is exactly
what this does. The half that was right -- *against the same row* -- is kept,
because the record the agent worked through is the only one whose answer is
known.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from usecase import UseCase

log = logging.getLogger(__name__)


@dataclass
class Verification:
    """What happened when the draft was replayed."""

    ran: bool
    ok: bool = False
    #: Which step failed, when one did.
    failed_step: str = ""
    error: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    #: Why it was not attempted, when it was not.
    skipped: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "ok": self.ok,
            "failed_step": self.failed_step,
            "error": self.error,
            "outputs": self.outputs,
            "duration_ms": self.duration_ms,
            "skipped": self.skipped,
        }

    def as_text(self) -> str:
        if not self.ran:
            return f"Not verified: {self.skipped}"
        if self.ok:
            read = ", ".join(self.outputs) or "nothing"
            return (
                f"Replayed cleanly by the engine in {self.duration_ms / 1000:.1f}s, "
                f"no model involved. It read: {read}."
            )
        return (
            f"Did not replay. Step {self.failed_step or '?'} failed: {self.error} "
            "The recording is kept -- this is what a reviewer needs to see."
        )


#: How a draft gets replayed. Injected so a test can drive it without a
#: browser, and so a deployment whose engine lives elsewhere -- an AgentCore
#: runtime calling back into a replay worker -- can supply its own.
Replayer = Callable[[UseCase, dict[str, Any], dict[str, str]], Awaitable[Verification]]


async def verify(
    use_case: UseCase,
    inputs: dict[str, Any],
    secrets: dict[str, str] | None = None,
    *,
    replay: Replayer | None = None,
) -> Verification:
    """Run the draft once, and say plainly whether it worked."""
    missing = [name for name in use_case.input_names if name not in inputs]
    if missing:
        return Verification(
            ran=False,
            skipped=(
                f"the recording declares {', '.join(sorted(missing))} but no "
                "value was captured for it, so there is nothing to replay with"
            ),
        )
    if not use_case.row_steps:
        return Verification(ran=False, skipped="the draft has no row steps")

    runner = replay or replay_with_engine
    try:
        return await runner(use_case, inputs, secrets or {})
    except Exception as exc:  # noqa: BLE001 - a failure to verify is a result
        log.warning("verification could not run", extra={"error": str(exc)})
        return Verification(ran=False, skipped=f"{type(exc).__name__}: {exc}")


class QuietSink:
    """Absorbs the verification run's events.

    They are deliberately not put on the session's stream. A replay inside an
    authoring session would appear in the trail as a second run under the same
    id, and a person reading it would see every step twice. What reaches the
    stream is the *verdict*, as one event.
    """

    def __init__(self) -> None:
        self.events: list[Any] = []
        self._seq = 0

    def reserve_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def emit(self, event: Any) -> None:
        self.events.append(event)

    async def save_screenshot(self, data: bytes, *, seq: int, mime: str = "image/png"):
        return None


async def replay_with_engine(
    use_case: UseCase, inputs: dict[str, Any], secrets: dict[str, str]
) -> Verification:
    """The real thing: a fresh browser, and the engine that runs batches.

    Imported here rather than at module scope so that importing this module
    does not pull in Playwright, which matters for the deployment that installs
    the agent extras but never verifies anything.
    """
    import time

    from browser import BrowserConfig, PlaywrightSession
    from engine import UseCaseExecutor

    # `ready` rather than the draft's own status: `runnable` gates on it, and
    # this is precisely the check that decides whether it *should* be
    # publishable. Validating a copy also means a draft that cannot even be
    # constructed at `ready` fails here rather than on somebody's screen.
    runnable = UseCase.model_validate({**use_case.model_dump(mode="json"), "status": "ready"})

    sink = QuietSink()
    started = time.monotonic()
    async with PlaywrightSession(BrowserConfig(headless=True)) as browser:
        executor = UseCaseExecutor(
            runnable,
            browser,
            sink,
            run_id="verify",
            secrets=secrets,
            env={"base_url": runnable.base_url} if runnable.base_url else {},
            screenshots="off",
        )
        setup = await executor.run_setup()
        if not setup.ok:
            return Verification(
                ran=True,
                ok=False,
                failed_step=setup.failed_step_id or "",
                error=setup.error or "setup failed",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        row = await executor.run_row(inputs)

    return Verification(
        ran=True,
        ok=row.ok,
        failed_step=row.failed_step_id or "",
        error=row.error or "",
        outputs=dict(row.outputs or {}),
        duration_ms=int((time.monotonic() - started) * 1000),
    )


__all__ = ["QuietSink", "Replayer", "Verification", "replay_with_engine", "verify"]
