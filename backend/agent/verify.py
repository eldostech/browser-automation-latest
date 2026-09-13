"""Replay the draft before anyone is told it exists.

The agent does not get to *claim* it recorded something. What it produced is
handed to ``engine.py`` -- the same code that will run the four thousand rows,
which cannot call a model -- and either replays or does not. A draft that does
not replay is a draft that says so, on the review screen, before anybody
publishes it.

**The loop closes here.** Verification used to end at a verdict: a draft that
did not replay came back with a warning saying so, and a person fixed it by
hand. That is the gap between "the agent did the task perfectly" and "the
recording does not work", and it is the one users actually feel -- the agent
acts on ``ref=e12``, an index into a snapshot seconds old that always names one
element, while a replay acts on a *description*. Recording therefore never
exercises the thing replay depends on and cannot fail the way replay fails.

So the first pass replays with a **healer attached**. A step whose locator does
not resolve is re-found against the page as it actually is, by the same
machinery that mends a locator mid-batch -- the model choosing from controls
that are really there, never inventing one. Whatever it mends is written back
into the draft, and then the draft is replayed **again, with no healer at all**.
That second pass is the one that decides: a repair nobody has proved is a
guess, and the whole point of this file is not to hand over guesses.

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

from healing import apply_repairs
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
    #: What the first pass had to mend before it would replay, as
    #: ``{step_id: "now looks for ..."}``. Reported rather than hidden: a
    #: reviewer deciding whether to trust this needs to know the recording did
    #: not work as recorded, even though what they are being handed does.
    repairs: dict[str, str] = field(default_factory=dict)
    #: Whether the *repaired* draft was replayed again to prove it. A repair
    #: nobody re-ran is a guess, and this says which kind you have.
    reproved: bool = False
    #: The mended definition, when mending happened. The caller writes this
    #: back as the draft: a repair proved by the second pass and then thrown
    #: away would be the most expensive possible way to learn nothing.
    patched: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "ok": self.ok,
            "failed_step": self.failed_step,
            "error": self.error,
            "outputs": self.outputs,
            "duration_ms": self.duration_ms,
            "skipped": self.skipped,
            "repairs": self.repairs,
            "reproved": self.reproved,
        }

    def as_text(self) -> str:
        if not self.ran:
            return f"Not verified: {self.skipped}"
        if self.ok and self.repairs:
            mended = "; ".join(f"{step}: {what}" for step, what in self.repairs.items())
            proof = (
                "then replayed again with no model involved at all, and worked"
                if self.reproved
                else "not replayed again, so treat this as unproven"
            )
            return (
                f"Replayed in {self.duration_ms / 1000:.1f}s, after mending "
                f"{len(self.repairs)} step(s) that did not work as recorded -- "
                f"{mended}. It was {proof}. Worth a look before publishing: the "
                "recording needed help, and the mend is in the draft."
            )
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
    healer: Any = None,
) -> Verification:
    """Replay the draft, mend what does not work, and prove the mend.

    ``healer`` closes the loop. Without one this behaves exactly as it always
    did -- one pass, one verdict -- which is what a deployment with healing
    switched off should get, and what every existing test gets.

    With one, a draft that fails is mended *in place* and replayed again with
    no healer at all. The second pass is the answer: it is the only evidence
    that what a reviewer is being handed actually runs.
    """
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
    first = await _attempt(runner, use_case, inputs, secrets or {}, healer=healer)
    if first.ok or not first.ran or healer is None:
        return first

    # The healer ran and mended nothing, so there is nothing new to prove and
    # the verdict stands. A step that failed for a reason healing cannot touch
    # -- an assertion that can never hold, a page that will not load -- lands
    # here, and saying "did not replay" is the honest answer.
    repairs = _repairs_of(healer)
    if not repairs:
        return first

    # Both steps inside one guard. Applying a repair reaches into whatever the
    # healer handed back, and validating the result reaches into the schema;
    # either can refuse a malformed mend, and neither is worth failing the
    # whole verification over. A truthful "did not replay" beats a crash and
    # beats a draft nobody can construct.
    try:
        mended = apply_repairs(
            use_case.model_dump(mode="json", by_alias=True), repairs
        )
        patched = UseCase.model_validate(mended)
    except Exception as exc:  # noqa: BLE001 - a mend that cannot be applied
        log.warning(
            "a verification repair could not be applied", extra={"error": str(exc)}
        )
        return first

    # No healer this time, deliberately. The first pass proves a model can get
    # through; only a pass with nothing model-shaped in it proves the *draft*
    # can. Anything else would hand somebody a recording that works when a
    # model is watching and fails at three in the morning.
    second = await _attempt(runner, patched, inputs, secrets or {}, healer=None)
    second.repairs = {
        repair.step_id: repair.locator.describe() for repair in repairs
    }
    second.reproved = True
    second.patched = mended
    return second


async def _attempt(
    runner: Replayer,
    use_case: UseCase,
    inputs: dict[str, Any],
    secrets: dict[str, str],
    *,
    healer: Any,
) -> Verification:
    """One replay. Never raises: a failure to verify is itself a result."""
    try:
        if healer is None:
            return await runner(use_case, inputs, secrets)
        return await runner(use_case, inputs, secrets, healer=healer)
    except TypeError:
        # An injected replayer from before healing was threaded through here.
        # Falling back rather than failing: the verdict is worth more than the
        # repair, and a test's stub is not obliged to know about this.
        try:
            return await runner(use_case, inputs, secrets)
        except Exception as exc:  # noqa: BLE001
            log.warning("verification could not run", extra={"error": str(exc)})
            return Verification(ran=False, skipped=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - a failure to verify is a result
        log.warning("verification could not run", extra={"error": str(exc)})
        return Verification(ran=False, skipped=f"{type(exc).__name__}: {exc}")


def _repairs_of(healer: Any) -> list[Any]:
    """What the healer mended, if it keeps a list. Empty if it does not."""
    return list(getattr(healer, "repairs", ()) or ())


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
    use_case: UseCase,
    inputs: dict[str, Any],
    secrets: dict[str, str],
    *,
    healer: Any = None,
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
    definition = {**use_case.model_dump(mode="json"), "status": "ready"}
    if healer is not None:
        # Healing is offered per *step*, by its own `on_failure`, and a
        # recording does not set that -- there is nowhere in an authoring
        # session for a person to have asked for it. Asking on this copy is
        # what lets the first pass mend anything at all; the draft itself is
        # untouched, so nothing about how it runs later is decided here.
        definition = {**definition, "mode": "guided"}
        for phase in ("setup_steps", "row_steps"):
            definition[phase] = [
                {**step, "on_failure": "heal"} for step in definition.get(phase) or []
            ]
    runnable = UseCase.model_validate(definition)

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
            healer=healer,
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
