"""How accurate and how consistent, as four numbers rather than an impression.

Everything else in this suite asks "does this behave correctly". This asks "how
often", which is a different question and the one that was missing. Every
accuracy change in this codebase so far has been argued from a failure somebody
hit and verified against a fixture -- which is good evidence that a specific
thing is fixed and no evidence at all about the whole.

Run it:

    cd backend && RUN_EVAL=1 ../.venv/Scripts/python -m pytest tests/test_eval_replay.py -q -s

Opt-in like the end-to-end tests, and for the same reason: it drives a real
Chromium against a real HTTP server, many times over.

The four numbers, and why each one is separate
----------------------------------------------
They pull against each other, which is the whole reason for measuring more than
one. Make the resolver bolder and refusals fall while wrong clicks rise; make
it stricter and the reverse. A single "success rate" hides that trade
completely.

**Step success.** The headline. Moves for any reason, so never read alone.

**False refusals.** Steps that failed as ambiguous where a person looking at
the page would say one candidate was obviously right. This is what scoping,
visible-only counting and the locator editor were built to reduce.

**Wrong element.** A step that "succeeded" against the wrong control. Counted
by having each fixture page record what was actually clicked, so this is
measured rather than assumed -- and it is the number that must not rise when
the others improve.

**Run-to-run variance.** The same use case, the same rows, several times. A
mean says how good it is; the spread says whether it can be trusted, and
"trust" was the actual complaint.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import statistics
import threading
from dataclasses import dataclass, field

import pytest

from browser import BrowserConfig, PlaywrightSession
from engine import UseCaseExecutor
from usecase import Locator, Step, UseCase

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.eval,
    pytest.mark.skipif(
        os.environ.get("RUN_EVAL") != "1",
        reason="set RUN_EVAL=1 to measure accuracy; it drives a real browser many times",
    ),
]

#: How many times each case runs. Three is the smallest number that can show a
#: spread at all; raise it when a change looks like an improvement and you need
#: to know whether it is one.
REPEATS = int(os.environ.get("EVAL_REPEATS", "3"))


# ---------------------------------------------------------------------------
# The site
# ---------------------------------------------------------------------------
#
# Every page records what was actually acted on, in `#log`, so "did the right
# thing happen" is read off the page rather than inferred from the step not
# raising. That is the only way to measure a wrong click, and a wrong click is
# the failure that matters most and shows up least.

LOGGER = """
<script>
  function note(what) {
    var log = document.getElementById('log');
    log.textContent = (log.textContent ? log.textContent + '|' : '') + what;
  }
</script>
<p id="log"></p>
"""

#: Three rows, one control each, distinguishable only by the row they are in.
#: The shape `within` exists for, and the shape that used to be refused.
ROWS = f"""<!doctype html>
<html><head><title>Customers</title></head><body>
  <table><tbody>
    <tr><td>Acme Ltd</td><td><button type="button" onclick="note('acme')">Edit</button></td></tr>
    <tr><td>Globex</td><td><button type="button" onclick="note('globex')">Edit</button></td></tr>
    <tr><td>Initech</td><td><button type="button" onclick="note('initech')">Edit</button></td></tr>
  </tbody></table>
  {LOGGER}
</body></html>
"""

#: A visible control with a hidden twin in the DOM. The role rung never saw the
#: twin; the text rung did, and the text rung is what a ladder falls through to.
HIDDEN_TWIN = f"""<!doctype html>
<html><head><title>Twin</title></head><body>
  <div style="display:none"><span>Continue</span></div>
  <main><button type="button" onclick="note('continue')">Continue</button></main>
  {LOGGER}
</body></html>
"""

#: A control inside an iframe: absent from the page as far as every rung that
#: does not name the frame is concerned.
FRAME_INNER = """<!doctype html>
<html><body><button type="button" onclick="parent.note('inner')">Pay</button></body></html>
"""

FRAMED = f"""<!doctype html>
<html><head><title>Checkout</title></head><body>
  <iframe id="pay" src="/frame-inner.html" title="Payment"></iframe>
  {LOGGER}
</body></html>
"""

#: A control whose wording differs from what was recorded. Nothing here can
#: resolve it, so this case measures that a failure is *reported honestly*
#: rather than silently landing on the nearest button.
RENAMED = f"""<!doctype html>
<html><head><title>Renamed</title></head><body>
  <button type="button" onclick="note('logon')">Log on</button>
  <button type="button" onclick="note('cancel')">Cancel</button>
  {LOGGER}
</body></html>
"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval-site")
    (root / "rows.html").write_text(ROWS, encoding="utf-8")
    (root / "hidden-twin.html").write_text(HIDDEN_TWIN, encoding="utf-8")
    (root / "framed.html").write_text(FRAMED, encoding="utf-8")
    (root / "frame-inner.html").write_text(FRAME_INNER, encoding="utf-8")
    (root / "renamed.html").write_text(RENAMED, encoding="utf-8")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One thing a replay is expected to be able to do.

    ``expect`` is what the page must report having done. ``""`` means the step
    is expected to fail -- and a case that is *meant* to fail is as much a part
    of accuracy as one meant to pass: a resolver that never refuses anything
    scores perfectly here and is useless.
    """

    name: str
    page: str
    locators: list[Locator]
    expect: str
    #: True where a person looking at the page would say one candidate is
    #: obviously right. A failure here is a *false* refusal.
    resolvable_by_eye: bool = True


def cases() -> list[Case]:
    return [
        Case(
            name="unambiguous button",
            page="hidden-twin.html",
            locators=[Locator(strategy="role", role="button", name="Continue")],
            expect="continue",
        ),
        Case(
            name="text rung with a hidden twin",
            page="hidden-twin.html",
            locators=[Locator(strategy="text", text="Continue", exact=True)],
            expect="continue",
        ),
        Case(
            name="one of three rows, scoped",
            page="rows.html",
            locators=[
                Locator(
                    strategy="role",
                    role="button",
                    name="Edit",
                    within=Locator(strategy="role", role="row", has_text="Globex"),
                )
            ],
            expect="globex",
        ),
        Case(
            name="one of three rows, unscoped",
            page="rows.html",
            locators=[Locator(strategy="role", role="button", name="Edit")],
            expect="",
            # Nothing on the page says which row was meant, so refusing is the
            # right answer and this must not be counted as a false refusal.
            resolvable_by_eye=False,
        ),
        Case(
            name="control inside an iframe",
            page="framed.html",
            locators=[
                Locator(strategy="role", role="button", name="Pay", frames=["iframe#pay"])
            ],
            expect="inner",
        ),
        Case(
            name="renamed control",
            page="renamed.html",
            locators=[Locator(strategy="role", role="button", name="Sign in")],
            expect="",
            resolvable_by_eye=False,
        ),
    ]


class Sink:
    def __init__(self) -> None:
        self._seq = 0

    def reserve_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def emit(self, event) -> None:
        return None

    async def save_screenshot(self, data, *, seq, mime="image/png"):
        return None


@dataclass
class Outcome:
    ok: bool
    acted_on: str
    error: str = ""


@dataclass
class Tally:
    attempted: int = 0
    succeeded: int = 0
    false_refusals: list[str] = field(default_factory=list)
    wrong_element: list[str] = field(default_factory=list)
    honest_refusals: int = 0

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.attempted if self.attempted else 0.0


async def _run_case(case: Case, base: str) -> Outcome:
    use_case = UseCase(
        name=case.name,
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{base}/{case.page}"),
            Step(id="s2", action="click", locators=case.locators),
        ],
    )
    sink = Sink()
    async with PlaywrightSession(BrowserConfig(headless=True, timeout_ms=3_000)) as browser:
        executor = UseCaseExecutor(
            use_case, browser, sink, run_id="eval", secrets={},
            step_timeout=3.0, screenshots="off",
        )
        result = await executor.run_row({})
        try:
            acted = await browser.page.locator("#log").inner_text()
        except Exception:  # noqa: BLE001 - a page that never loaded logged nothing
            acted = ""
    return Outcome(ok=result.ok, acted_on=acted.strip(), error=result.error or "")


def _score(case: Case, outcome: Outcome, tally: Tally) -> None:
    tally.attempted += 1

    if case.expect == "":
        # Meant to fail. Succeeding here means a wrong element was acted on,
        # which is the worst outcome and the one a bare success rate rewards.
        if outcome.ok:
            tally.wrong_element.append(f"{case.name}: acted on {outcome.acted_on!r}")
        else:
            tally.honest_refusals += 1
            tally.succeeded += 1
        return

    if outcome.ok and outcome.acted_on == case.expect:
        tally.succeeded += 1
        return
    if outcome.ok:
        tally.wrong_element.append(
            f"{case.name}: expected {case.expect!r}, acted on {outcome.acted_on!r}"
        )
        return
    if case.resolvable_by_eye:
        tally.false_refusals.append(f"{case.name}: {outcome.error[:120]}")


async def test_accuracy_and_consistency(site):
    """Run every case several times and print the four numbers."""
    runs: list[Tally] = []
    for _ in range(REPEATS):
        tally = Tally()
        for case in cases():
            _score(case, await _run_case(case, site), tally)
        runs.append(tally)

    rates = [run.success_rate for run in runs]
    report = {
        "repeats": REPEATS,
        "cases_per_run": len(cases()),
        "step_success_mean": round(statistics.mean(rates), 3),
        "step_success_spread": round(max(rates) - min(rates), 3),
        "false_refusals": sorted({item for run in runs for item in run.false_refusals}),
        "wrong_element": sorted({item for run in runs for item in run.wrong_element}),
        "honest_refusals_per_run": runs[0].honest_refusals,
    }
    print("\n" + json.dumps(report, indent=2))

    # The two that are not judgement calls. A wrong element is always a defect,
    # and a spread means the same input gave different answers -- which is what
    # "I cannot trust it" actually means, measured.
    assert report["wrong_element"] == [], "a step acted on the wrong control"
    assert report["step_success_spread"] == 0.0, "the same input gave different answers"

    # And the bar the accuracy work was for. Stated as a number so a change
    # that lowers it fails here rather than being noticed in production.
    assert report["step_success_mean"] == 1.0, report
