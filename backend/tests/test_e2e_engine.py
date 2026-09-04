"""Record, then replay, against a real browser and a real page.

Every other test in this suite fakes the browser session. That was tolerable
while the browser layer was stable and only the logic above it changed; it
stopped being tolerable the moment the browser layer itself was replaced, which
is why the design document makes P4 conditional on this file existing.

What it proves, end to end and with nothing stubbed:

* a ``playwright codegen`` script parses into a use case;
* every locator strategy the parser can emit resolves against a live page;
* a parameterised replay types per-row values into a real form;
* assertions, extraction, drift, screenshots and traces all work;
* the allowlist stops a row whose input URL leaves the recorded domains.

The site is served from a temp directory over ``http.server`` on a random
port, so the test needs no network and cannot be flaky because of one.

Opt in with ``RUN_E2E=1`` -- it launches Chromium, which is slower than the
rest of the suite put together and needs a browser installed.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import os
import threading
from pathlib import Path

import pytest

from browser import BrowserConfig, PlaywrightSession
from codegen import parse
from engine import UseCaseExecutor
from fields import FieldSet
from routers.recordings import build_usecase
from usecase import Locator, Step

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("RUN_E2E") != "1",
        reason="set RUN_E2E=1 to run the real-browser test",
    ),
]

#: The first four bytes of any PNG, so "it saved something" is not the
#: whole assertion.
PNG_MAGIC = b"\x89PNG"

SIGN_IN = """<!doctype html>
<html><head><title>Sign in</title></head><body>
  <h1>Sign in</h1>
  <form action="/orders.html">
    <label>Username <input name="u"></label>
    <label>Password <input name="p" type="password"></label>
    <button type="submit">Sign in</button>
  </form>
</body></html>
"""

ORDERS = """<!doctype html>
<html><head><title>Orders</title></head><body>
  <h1>Orders</h1>
  <p>Signed in as Ada</p>
  <form id="order-form">
    <input id="reference" placeholder="Order reference">
    <img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" alt="Company logo">
    <button data-testid="submit-order" type="submit">Submit</button>
  </form>
  <div id="done" style="display:none">
    <h2>Thank you</h2>
    <span id="ref"></span>
  </div>
  <script>
    document.getElementById('order-form').addEventListener('submit', function (event) {
      event.preventDefault();
      document.getElementById('ref').textContent =
        document.getElementById('reference').value;
      document.getElementById('done').style.display = 'block';
    });
  </script>
</body></html>
"""


#: A vendor's list page: the index a migration has to read before it can do
#: anything, and the identifier is in the href rather than in the visible text.
ACCOUNTS = """<!doctype html>
<html><head><title>Accounts</title></head><body>
  <h1>Accounts</h1>
  <table><tbody>
    <tr><td><a href="/account/A-1001">Ada Lovelace</a></td><td>Active</td></tr>
    <tr><td><a href="/account/A-1002">Grace Hopper</a></td><td>Closed</td></tr>
    <tr><td><a href="/account/A-1003">Karen Sparck Jones</a></td><td>Active</td></tr>
  </tbody></table>
</body></html>
"""


#: Renders nothing for seven seconds, then puts the control on the page. This
#: is a slow site reduced to the one property that matters: the element the
#: recording asks for is not there when the step starts, and is there later.
#: Seven is past the wait the ladder used to allow and inside the step timeout.
LATE = """<!doctype html>
<html><head><title>Slow</title></head><body>
  <div id="app"></div>
  <script>
    setTimeout(function () {
      document.getElementById('app').innerHTML =
        '<button id="go">Continue</button>';
    }, 7000);
  </script>
</body></html>
"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A two-page static site on a random port."""
    root = tmp_path_factory.mktemp("site")
    (root / "index.html").write_text(SIGN_IN, encoding="utf-8")
    (root / "late.html").write_text(LATE, encoding="utf-8")
    (root / "accounts.html").write_text(ACCOUNTS, encoding="utf-8")
    (root / "orders.html").write_text(ORDERS, encoding="utf-8")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def codegen_script(base: str) -> str:
    """What `playwright codegen` writes for the task, by hand.

    Every locator strategy the parser supports appears at least once, so this
    doubles as the proof that each one resolves against a real page.
    """
    return f'''import asyncio
from playwright.async_api import Playwright, async_playwright, expect


async def run(playwright: Playwright) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto("{base}/index.html")
    await page.get_by_label("Username").fill("ada")
    await page.get_by_label("Password").fill("s3cret-Example-Pw")
    await page.get_by_role("button", name="Sign in").click()
    await page.get_by_placeholder("Order reference").fill("A-1024")
    await page.get_by_alt_text("Company logo").hover()
    await page.get_by_test_id("submit-order").click()
    await expect(page.get_by_text("Thank you")).to_be_visible()

    # ---------------------
    await context.close()
    await browser.close()
'''


class Sink:
    """Collects events instead of persisting them."""

    def __init__(self) -> None:
        self.events: list = []
        self.shots: list[bytes] = []
        self._seq = 0

    def reserve_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def emit(self, event) -> None:
        self.events.append(event)

    async def save_screenshot(self, data: bytes, *, seq: int, mime: str = "image/png"):
        self.shots.append(data)
        return f"artifact-{seq}", f"/artifacts/artifact-{seq}"

    def failures(self) -> list:
        return [e for e in self.events if getattr(e, "type", "") == "step_finished" and not e.ok]


def recorded(base: str):
    """The codegen script, parsed and parameterised as the API would."""
    declared = FieldSet.from_payload(
        [
            {"name": "username", "value": "ada", "secret": True},
            {"name": "password", "value": "s3cret-Example-Pw", "secret": True},
            {"name": "reference", "value": "A-1024", "secret": False},
        ]
    )
    use_case = build_usecase(
        parse(codegen_script(base)), name="Submit orders", description="", declared=declared
    )
    # Published by a person in the product; done here so it can run.
    use_case.status = "ready"
    use_case.allowed_domains = ["127.0.0.1"]
    # A recording cannot know what "back to the start" means, so codegen emits
    # no reset and the draft carries no `row_reset`. Adding one is a review
    # step, and without it row 2 begins wherever row 1 happened to end -- which
    # is exactly the coupling the row/reset/setup split exists to manage.
    use_case.row_reset = Step(
        id="reset", action="navigate", url=f"{base}/orders.html"
    )
    return use_case


async def execute(
    use_case, base, rows, *, screenshots="final", trace_dir=None, step_timeout=10.0
):
    sink = Sink()
    config = BrowserConfig(
        headless=True, timeout_ms=int(step_timeout * 1000), trace_dir=trace_dir
    )
    results = []
    async with PlaywrightSession(config) as browser:
        executor = UseCaseExecutor(
            use_case,
            browser,
            sink,
            run_id="e2e",
            secrets={"username": "ada", "password": "s3cret-Example-Pw"},
            screenshots=screenshots,
            step_timeout=step_timeout,
        )
        setup = await executor.run_setup()
        assert setup.ok, setup.error
        for row in rows:
            results.append(await executor.run_row(row))
    return executor, sink, results


# --- the whole loop --------------------------------------------------------


async def test_a_recording_replays_against_a_real_page(site):
    """The claim the product is built on, exercised without a stub in sight."""
    use_case = recorded(site)
    executor, sink, [result] = await execute(use_case, site, [{"reference": "A-1024"}])

    assert result.ok, result.error
    assert sink.failures() == []
    # Zero tokens is the product. Nothing in this path can spend one.
    assert result.llm_tokens == 0 and result.llm_calls == 0


async def test_every_recorded_locator_strategy_resolves(site):
    """role, label, placeholder, alt_text and test_id in one recording.

    Each becomes exactly one Playwright call. If a strategy stopped mapping
    correctly this is where it shows, rather than on somebody's thousand-row
    batch.
    """
    use_case = recorded(site)
    strategies = {
        locator.strategy for step in use_case.all_steps for locator in step.locators
    }
    assert {"role", "label", "placeholder", "alt_text", "test_id"} <= strategies

    _, sink, [result] = await execute(use_case, site, [{"reference": "A-1024"}])
    assert result.ok, result.error


async def test_each_row_types_its_own_value(site):
    """Parameterisation, against a page that echoes what it was given.

    This is the difference between a recording and an automation: the same
    steps, different data, and the page proving it received the right one.
    """
    use_case = recorded(site)
    # Read back what the form echoed, so the assertion is about the page.
    use_case.row_steps.append(
        Step(
            id="extract-ref",
            action="extract",
            output="echoed",
            locators=[Locator(strategy="css", selector="#ref")],
        )
    )

    _, _, results = await execute(
        use_case, site, [{"reference": "A-1024"}, {"reference": "B-2048"}]
    )
    assert [r.ok for r in results] == [True, True]
    assert [r.outputs["echoed"] for r in results] == ["A-1024", "B-2048"]


async def test_the_login_runs_once_for_the_whole_batch(site):
    """A shared session signs in once. Running setup per row is the failure
    this split exists to prevent."""
    use_case = recorded(site)
    assert [s.action for s in use_case.setup_steps] == ["navigate", "fill", "fill", "click"]

    _, sink, results = await execute(
        use_case, site, [{"reference": "A-1"}, {"reference": "A-2"}, {"reference": "A-3"}]
    )
    assert all(r.ok for r in results)
    signins = [
        e for e in sink.events
        if getattr(e, "type", "") == "step_started" and e.phase == "setup"
    ]
    assert len(signins) == 4, "setup ran more than once"


# --- what happens when the page changes ------------------------------------


async def test_a_renamed_control_falls_through_to_the_next_rung(site):
    """The durability claim, made concrete.

    The recorded rung names a button that no longer exists; the free name-only
    rung recorded beside it still finds the control. The fall-through is
    reported as drift rather than passing silently.
    """
    use_case = recorded(site)
    click = next(s for s in use_case.setup_steps if s.action == "click")
    # The recorded rung names a control that is not on this page. The free text
    # rung recorded beside it still is.
    click.locators[0].name = "Log in instead"

    executor, _, [result] = await execute(use_case, site, [{"reference": "A-1024"}])
    assert result.ok, result.error
    assert executor.locator_drift.get(click.id) == 1, "the drift was not reported"


async def test_a_step_that_matches_nothing_fails_with_what_was_on_the_page(site):
    use_case = recorded(site)
    click = next(s for s in use_case.row_steps if s.action == "click")
    click.locators = [Locator(strategy="css", selector="#not-here")]

    _, _, [result] = await execute(use_case, site, [{"reference": "A-1024"}])
    assert not result.ok
    assert "no element matched" in (result.error or "")
    # The failure message names what the page did have, so a person can see
    # what changed without opening the site.
    assert "The page has" in (result.error or "")


async def test_an_assertion_that_does_not_hold_fails_the_row(site):
    """Without this a batch of a thousand rows fails silently on row 12 and
    reports success on all of them."""
    use_case = recorded(site)
    check = next(s for s in use_case.row_steps if s.action == "assert")
    check.assertion.value = "Something that is not on the page"
    check.assertion.kind = "text_present"
    check.assertion.locator = None
    check.assertion.timeout_ms = 1_000

    _, _, [result] = await execute(use_case, site, [{"reference": "A-1024"}])
    assert not result.ok
    assert result.failed_step_id == check.id


async def test_navigation_off_the_allowlist_is_refused(site):
    """A URL column comes from a spreadsheet, and a spreadsheet is untrusted
    input. The allowlist applies to a replay exactly as it did to the agent --
    including to the sign-in, which is why this cannot use the helper that
    expects setup to succeed.
    """
    use_case = recorded(site)
    use_case.allowed_domains = ["example.com"]

    sink = Sink()
    async with PlaywrightSession(BrowserConfig(headless=True, timeout_ms=5_000)) as browser:
        executor = UseCaseExecutor(
            use_case, browser, sink, run_id="e2e", secrets={}, step_timeout=5.0
        )
        setup = await executor.run_setup()

    assert not setup.ok
    assert "not in the allowed domain list" in (setup.error or "")


# --- the audit trail -------------------------------------------------------


async def test_a_screenshot_is_kept_for_every_row(site):
    use_case = recorded(site)
    _, sink, _ = await execute(
        use_case, site, [{"reference": "A-1"}, {"reference": "A-2"}], screenshots="final"
    )
    assert len(sink.shots) == 2
    assert all(shot[:4] == PNG_MAGIC for shot in sink.shots)


async def test_a_trace_is_written_when_asked_for(site, tmp_path):
    """Playwright's own trace, with a DOM snapshot per action.

    Not reachable through the MCP tool surface at all, and the single biggest
    debugging win of driving the library directly.
    """
    use_case = recorded(site)
    await execute(use_case, site, [{"reference": "A-1024"}], trace_dir=tmp_path)

    trace = tmp_path / "trace.zip"
    assert trace.exists() and trace.stat().st_size > 0


async def test_the_engine_cannot_reach_a_model():
    """The zero-token guarantee, asserted structurally rather than promised.

    Threading a no-LLM flag through an agent would have left the expensive path
    one bug away from being re-entered. A module that cannot import a model
    client has no such path.
    """
    import inspect

    import engine

    source = inspect.getsource(engine)
    assert "import llm" not in source
    assert "from llm" not in source
    assert "llm" not in inspect.signature(UseCaseExecutor.__init__).parameters


async def test_an_element_that_renders_late_is_waited_for(site):
    """The failure a slow site produces, and the reason the wait is dynamic.

    The ladder here has a semantic rung and a weak one, which is what almost
    every recorded step has -- and that combination used to cap the wait at
    five seconds regardless of the step timeout. Against a page that takes
    seven seconds to render, the step gave up at five and reported "no element
    matched" for an element that was simply not there yet.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="late",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/late.html"),
            Step(
                id="s2",
                action="click",
                locators=[
                    Locator(strategy="role", role="button", name="Continue"),
                    Locator(strategy="css", selector="#go"),
                ],
            ),
        ],
    )

    started = asyncio.get_running_loop().time()
    _, _, results = await execute(use_case, site, [{}], step_timeout=20.0)
    elapsed = asyncio.get_running_loop().time() - started

    assert results[0].ok, results[0].error
    assert elapsed > 7, "it cannot have passed without actually waiting"


async def test_one_document_runs_against_two_different_addresses(site):
    """Promotion, proved rather than asserted.

    The same use case -- not a copy, the same object -- is run twice: once
    letting it fall back to the origin it was recorded against, and once with a
    deployment naming a different one. Both navigate successfully, and the
    allowlist permits each in turn without being widened to both.

    The second address is the same server reached through a hostname the
    recording never saw, which is exactly the shape of promoting dev to UAT.
    """
    from usecase import UseCase

    recorded_at = site  # http://127.0.0.1:PORT
    promoted_to = site.replace("127.0.0.1", "localhost")

    use_case = UseCase(
        name="promotable",
        status="ready",
        base_url=recorded_at,
        allowed_domains=["{{env.base_url}}"],
        row_steps=[
            Step(id="s1", action="navigate", url="{{env.base_url}}/index.html"),
            # A real interaction, so this proves the page was actually served
            # at the bound address rather than merely navigated to.
            Step(
                id="s2",
                action="fill",
                value="ada",
                locators=[Locator(strategy="label", text="Username")],
            ),
        ],
    )

    sink = Sink()
    config = BrowserConfig(headless=True, timeout_ms=10_000)
    async with PlaywrightSession(config) as browser:
        # Dev: nothing configured, so the recorded origin answers.
        as_recorded = UseCaseExecutor(
            use_case, browser, sink, run_id="dev", secrets={}, step_timeout=10.0
        )
        assert (await as_recorded.run_row({})).ok
        assert browser.url.startswith(recorded_at)

        # UAT: the deployment names its own address, document untouched.
        as_promoted = UseCaseExecutor(
            use_case,
            browser,
            sink,
            run_id="uat",
            secrets={},
            step_timeout=10.0,
            env={"base_url": promoted_to},
        )
        assert (await as_promoted.run_row({})).ok
        assert browser.url.startswith(promoted_to)


async def test_a_list_page_becomes_rows(site):
    """The first pass of a migration, against a real page.

    A vendor who will not open their back end still has a list page, and that
    page is the index. This reads it into rows -- including the identifier out
    of the href, which is the part that is never in the visible text.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="discover",
        status="ready",
        allowed_domains=["127.0.0.1"],
        outputs=["accounts"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/accounts.html"),
            Step(
                id="s2",
                action="extract_rows",
                output="accounts",
                locators=[Locator(strategy="css", selector="table tbody tr")],
                columns=[
                    {"name": "account_id", "selector": "td a", "attribute": "href"},
                    {"name": "customer", "selector": "td a"},
                    {"name": "status", "selector": "td:nth-child(2)"},
                ],
            ),
        ],
    )

    _, _, results = await execute(use_case, site, [{}])

    assert results[0].ok, results[0].error
    found = results[0].outputs["accounts"]
    assert found == [
        {"account_id": "/account/A-1001", "customer": "Ada Lovelace", "status": "Active"},
        {"account_id": "/account/A-1002", "customer": "Grace Hopper", "status": "Closed"},
        {
            "account_id": "/account/A-1003",
            "customer": "Karen Sparck Jones",
            "status": "Active",
        },
    ]


async def test_a_list_page_with_no_rows_is_not_a_failure(site):
    """The last page of a paginated crawl is legitimately empty."""
    from usecase import UseCase

    use_case = UseCase(
        name="discover nothing",
        status="ready",
        allowed_domains=["127.0.0.1"],
        outputs=["accounts"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/accounts.html"),
            Step(
                id="s2",
                action="extract_rows",
                output="accounts",
                locators=[Locator(strategy="css", selector="table tbody tr.missing")],
                columns=[{"name": "account_id", "selector": "td"}],
            ),
        ],
    )

    _, _, results = await execute(use_case, site, [{}])

    assert results[0].ok, "an empty page must not break the crawl at its final step"
    assert results[0].outputs["accounts"] == []
