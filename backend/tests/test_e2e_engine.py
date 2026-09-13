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
from usecase import InputSpec, Locator, Step

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

#: The heading deliberately does *not* repeat the button's wording. It used to,
#: and `get_by_text("Sign in")` then matched the heading as well as the button
#: -- so the fall-through test below was passing while clicking an <h1>. That
#: is the bug the resolver now refuses to commit, and a fixture that contains
#: it cannot demonstrate a working fall-through.
SIGN_IN = """<!doctype html>
<html><head><title>Sign in</title></head><body>
  <h1>Welcome back</h1>
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


#: A vendor's document page. `download` on the anchor is what makes the browser
#: treat it as a file rather than navigating to it, which is how these pages
#: actually behave.
STATEMENT_PAGE = """<!doctype html>
<html><head><title>Statement</title></head><body>
  <h1>Statement</h1>
  <a id="dl" href="/statement-A-1001.csv" download>Download statement</a>
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


#: The page that broke a real run. A "+ Invite User" button opens a dialog
#: holding an "Invite" button, and the dialog's backdrop covers the first one.
#: Playwright matches an accessible name as a substring, so "Invite" names both
#: -- and the one it finds first is the one nobody can click.
INVITE = """<!doctype html>
<html><head><title>Users</title>
<style>
  .backdrop { position: fixed; inset: 0; background: rgba(0,0,0,.4); display: none; }
  .backdrop.open { display: block; }
  .dialog { position: fixed; top: 25%; left: 30%; background: #fff; padding: 24px; }
</style></head>
<body>
  <h1>Users</h1>
  <button id="open">+ Invite User</button>
  <div class="backdrop" id="backdrop"><div class="dialog" role="dialog">
    <label>Email <input type="email"></label>
    <button id="send">Invite</button>
  </div></div>
  <p id="outcome"></p>
  <script>
    document.getElementById('open').onclick = function () {
      document.getElementById('backdrop').classList.add('open');
    };
    document.getElementById('send').onclick = function () {
      document.getElementById('outcome').textContent = 'invited';
    };
  </script>
</body></html>
"""


#: Two controls with the same accessible name and nothing to tell them apart.
#: No locator can mean one of them, which is a different problem from the one
#: above and has to fail differently.
TWINS = """<!doctype html>
<html><head><title>Twins</title></head><body>
  <form><button type="button">Save</button></form>
  <form><button type="button">Save</button></form>
</body></html>
"""


#: A hidden twin of a visible control -- the shape every responsive site has,
#: where a nav is rendered twice and one copy is display:none at the current
#: width.
#:
#: Which rung this breaks is worth being precise about, because the obvious
#: guess is wrong. `get_by_role` reads the accessibility tree, and a
#: display:none element is not in it, so a *role* rung never saw the twin. A
#: `text` or `css` rung is matched against the DOM, and does. Those are the
#: fallback rungs -- `_ladder` puts a text rung under every named role rung --
#: so the twin costs nothing until the day the role rung stops matching, and
#: then the ladder falls through to a rung that is refused as ambiguous by an
#: element nobody can see.
HIDDEN_TWIN = """<!doctype html>
<html><head><title>Twin</title></head><body>
  <div id="mobile-nav" style="display:none"><span>Continue</span></div>
  <main>
    <button type="button" id="real">Continue</button>
    <p id="outcome"></p>
  </main>
  <script>
    document.getElementById('real').onclick = function () {
      document.getElementById('outcome').textContent = 'continued';
    };
  </script>
</body></html>
"""

#: The same button name on every row of a table. Nothing but the row it sits
#: in tells them apart, which is what `within` is for.
ROWS = """<!doctype html>
<html><head><title>Customers</title></head><body>
  <table>
    <tbody>
      <tr><td>Acme Ltd</td><td><button type="button" class="edit">Edit</button></td></tr>
      <tr><td>Globex</td><td><button type="button" class="edit">Edit</button></td></tr>
      <tr><td>Initech</td><td><button type="button" class="edit">Edit</button></td></tr>
    </tbody>
  </table>
  <p id="outcome"></p>
  <script>
    document.querySelectorAll('tr').forEach(function (row) {
      row.querySelector('button').onclick = function () {
        document.getElementById('outcome').textContent =
          'editing ' + row.querySelector('td').textContent;
      };
    });
  </script>
</body></html>
"""

#: A form inside an iframe. Before frames were recordable this was not a hard
#: page to automate -- it was an impossible one.
PAYMENT_FRAME = """<!doctype html>
<html><head><title>Card</title></head><body>
  <label>Card number <input id="card" name="card"></label>
</body></html>
"""

CHECKOUT = """<!doctype html>
<html><head><title>Checkout</title></head><body>
  <h1>Checkout</h1>
  <iframe id="pay" src="/payment-frame.html" title="Payment"></iframe>
</body></html>
"""

#: A link that opens a second tab. The run has to follow it, or every step
#: after this one runs against a page nobody is looking at.
NEW_TAB = """<!doctype html>
<html><head><title>Reports</title></head><body>
  <a id="open" href="/orders.html" target="_blank">Open the order form</a>
</body></html>
"""


#: A search box whose dropdown shows something *other* than what was typed.
#:
#: Typing a customer number opens a list of customer *names*, and the name has
#: no textual relationship to the number at all. `playwright codegen` records
#: the click on that suggestion by its accessible name, so the recording ends
#: up carrying row one's answer -- "Acme Ltd" -- as the thing to look for on
#: every subsequent row.
CUSTOMER_SEARCH = """<!doctype html>
<html><head><title>Customer search</title></head><body>
  <h1>Customers</h1>
  <label for="q">Customer number</label>
  <input id="q" role="combobox" aria-controls="results" aria-expanded="false"
         aria-autocomplete="list" placeholder="Customer number">
  <ul id="results" role="listbox" hidden></ul>
  <p id="chosen"></p>
  <script>
    var CUSTOMERS = [
      {number: 'C-1001', name: 'Acme Ltd'},
      {number: 'C-1002', name: 'Globex Corporation'},
      {number: 'C-1003', name: 'Initech'}
    ];
    var box = document.getElementById('q');
    var list = document.getElementById('results');
    box.addEventListener('input', function () {
      var typed = box.value.trim().toUpperCase();
      list.innerHTML = '';
      var hits = typed ? CUSTOMERS.filter(function (c) {
        return c.number.toUpperCase().indexOf(typed) === 0;
      }) : [];
      hits.forEach(function (c) {
        var item = document.createElement('li');
        item.setAttribute('role', 'option');
        item.textContent = c.name;
        item.onclick = function () {
          document.getElementById('chosen').textContent = 'Selected ' + c.name;
          list.hidden = true;
          box.setAttribute('aria-expanded', 'false');
        };
        list.appendChild(item);
      });
      list.hidden = hits.length === 0;
      box.setAttribute('aria-expanded', String(hits.length > 0));
    });
  </script>
</body></html>
"""


#: A table header under a cookie banner, which is the shape a real recording
#: failed on: the header is found, visible and enabled, and a fixed overlay
#: sits on top of it, so every click is intercepted until the timeout.
COVERED = """<!doctype html>
<html><head><title>Recalls</title>
<style>
  #consent { position: fixed; inset: 0; background: rgba(0,0,0,0.35); z-index: 9; }
</style></head><body>
  <div id="consent"></div>
  <table>
    <thead><tr>
      <th onclick="document.getElementById('log').textContent='sorted'">Make</th>
    </tr></thead>
    <!-- A body row, because Chrome exposes a `columnheader` only for a table
         that has one -- without it the header is not in the tree at all and
         this fixture would prove something else entirely. -->
    <tbody><tr><td>RAM</td></tr></tbody>
  </table>
  <p id="log"></p>
</body></html>
"""

#: A styled radio: the input is in the accessibility tree with its name, and
#: it is the label that can be clicked. Real, and the shape that cost a whole
#: recording -- a profile picker drawn exactly this way.
#:
#: `width: 0` rather than `display: none`, because a radio that is display:none
#: is absent from the accessibility tree too and the recording would never have
#: named it. Nothing catches this except trying the action.
STYLED_RADIO = """<!doctype html>
<html><head><title>Who are you?</title>
<style>
  input[type=radio] { position: absolute; width: 0; height: 0; opacity: 0; }
  label { display: block; padding: 12px; border: 1px solid #ccc; }
</style></head><body>
  <form>
    <label for="nayra">Nayra Asati
      <input type="radio" id="nayra" name="who" aria-label="Nayra Asati">
    </label>
  </form>
  <p id="log"></p>
  <script>
    document.getElementById('nayra').addEventListener('click', function () {
      document.getElementById('log').textContent = 'chosen';
    });
  </script>
</body></html>
"""

#: The same markup as when it was recorded, with a different control in the
#: same place. A CSS path still matches exactly one element here, which is
#: precisely why it is dangerous: the step succeeds against the wrong button.
REBUILT = """<!doctype html>
<html><head><title>Rebuilt</title></head><body>
  <div class="toolbar">
    <button type="button" onclick="document.title='WRONG'">Delete permanently</button>
  </div>
  <a id="receipt" href="/receipt/A-1001.pdf">Receipt</a>
  <p id="log"></p>
</body></html>
"""


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A two-page static site on a random port."""
    root = tmp_path_factory.mktemp("site")
    (root / "index.html").write_text(SIGN_IN, encoding="utf-8")
    (root / "late.html").write_text(LATE, encoding="utf-8")
    (root / "accounts.html").write_text(ACCOUNTS, encoding="utf-8")
    (root / "statement.html").write_text(STATEMENT_PAGE, encoding="utf-8")
    (root / "statement-A-1001.csv").write_text(
        "account,balance" + chr(10) + "A-1001,1240.55" + chr(10), encoding="utf-8"
    )
    (root / "orders.html").write_text(ORDERS, encoding="utf-8")
    (root / "invite.html").write_text(INVITE, encoding="utf-8")
    (root / "twins.html").write_text(TWINS, encoding="utf-8")
    (root / "hidden-twin.html").write_text(HIDDEN_TWIN, encoding="utf-8")
    (root / "rows.html").write_text(ROWS, encoding="utf-8")
    (root / "checkout.html").write_text(CHECKOUT, encoding="utf-8")
    (root / "payment-frame.html").write_text(PAYMENT_FRAME, encoding="utf-8")
    (root / "new-tab.html").write_text(NEW_TAB, encoding="utf-8")
    (root / "search.html").write_text(CUSTOMER_SEARCH, encoding="utf-8")
    (root / "rebuilt.html").write_text(REBUILT, encoding="utf-8")
    (root / "styled-radio.html").write_text(STYLED_RADIO, encoding="utf-8")
    (root / "covered.html").write_text(COVERED, encoding="utf-8")

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
        self.downloads: list[tuple[str, bytes, str]] = []
        self._seq = 0

    def reserve_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def emit(self, event) -> None:
        self.events.append(event)

    async def save_screenshot(self, data: bytes, *, seq: int, mime: str = "image/png"):
        self.shots.append(data)
        return f"artifact-{seq}", f"/artifacts/artifact-{seq}"

    async def save_download(self, data: bytes, *, seq: int, filename: str, mime: str):
        self.downloads.append((filename, data, mime))
        return f"download-{seq}", f"/artifacts/download-{seq}"

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

    # Without this the row would begin by navigating to the orders page, and
    # would succeed whether or not the sign-in click found anything at all.
    # Dropping it makes the row's own steps the proof: they are on the page
    # that only a real click on the sign-in button reaches.
    use_case.row_reset = None

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


async def test_a_document_is_downloaded_and_kept(site):
    """The other half of a migration: the files, not just the fields.

    A vendor who will not open their back end still lets you click "download",
    and that file is the deliverable. It is kept where every other artifact
    goes, under the name it arrived with, addressable per row.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="fetch a statement",
        status="ready",
        allowed_domains=["127.0.0.1"],
        outputs=["statement"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/statement.html"),
            Step(
                id="s2",
                action="download",
                output="statement",
                locators=[Locator(strategy="css", selector="#dl")],
            ),
        ],
    )

    _, sink, results = await execute(use_case, site, [{}])

    assert results[0].ok, results[0].error
    kept = results[0].outputs["statement"]
    assert kept["filename"] == "statement-A-1001.csv", "the vendor's own name is kept"
    assert kept["bytes"] > 0
    assert kept["artifact_id"], "it is addressable"

    assert len(sink.downloads) == 1
    filename, data, mime = sink.downloads[0]
    assert filename == "statement-A-1001.csv"
    assert b"A-1001,1240.55" in data, "the bytes are the real file"
    assert mime == "text/csv"


# --- a name that contains another name -------------------------------------


def invite_use_case(site, *, exact: bool):
    """Open the dialog, then click the button inside it."""
    from usecase import UseCase

    return UseCase(
        name="invite",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/invite.html"),
            Step(
                id="s2",
                action="click",
                locators=[Locator(strategy="role", role="button", name="+ Invite User")],
            ),
            Step(
                id="s3",
                action="click",
                locators=[
                    Locator(strategy="role", role="button", name="Invite", exact=exact),
                    Locator(strategy="text", text="Invite", exact=exact),
                ],
            ),
            Step(
                id="s4",
                action="extract",
                output="outcome",
                locators=[Locator(strategy="css", selector="#outcome")],
            ),
        ],
    )


async def test_the_button_in_a_dialog_is_clicked_and_not_the_one_behind_it(site):
    """The failure this whole arrangement exists to prevent.

    ``get_by_role("button", name="Invite")`` finds "+ Invite User" as well, and
    the resolver used to take ``.first`` of whatever matched. That is the
    button the dialog's backdrop is covering, so the click waited for it to
    become actionable and died on the step timeout -- reporting a timeout on a
    button that was on the page, enabled, and one match away.
    """
    started = asyncio.get_running_loop().time()
    _, _, [result] = await execute(
        invite_use_case(site, exact=True), site, [{}], step_timeout=8.0
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert result.ok, result.error
    assert result.outputs["outcome"] == "invited", "the dialog's button, not the page's"
    assert elapsed < 8, "it cannot have spent the step budget waiting"


async def test_a_recording_made_before_exact_was_kept_still_resolves(site):
    """Recordings already saved have no ``exact`` to read, and must still run.

    The ladder narrows a named rung to the whole accessible name before trying
    it as recorded. That is not a guess about the page: it is the same name,
    read strictly, and it is taken only when it matches exactly one element.
    """
    _, _, [result] = await execute(
        invite_use_case(site, exact=False), site, [{}], step_timeout=8.0
    )

    assert result.ok, result.error
    assert result.outputs["outcome"] == "invited"


async def test_two_identical_controls_fail_as_ambiguous_rather_than_as_missing(site):
    """When no reading of the name picks one, say so.

    Clicking whichever came first is how a batch acts on the wrong element a
    thousand times over, and "no element matched" for two elements that did
    sends somebody looking for the wrong problem.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="twins",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/twins.html"),
            Step(
                id="s2",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Save")],
            ),
        ],
    )

    _, _, [result] = await execute(use_case, site, [{}], step_timeout=3.0)

    assert not result.ok
    assert "ambiguous" in result.error
    assert "matched 2" in result.error


# ---------------------------------------------------------------------------
# What only a real browser can show
# ---------------------------------------------------------------------------
#
# Every test below covers something an accessibility-snapshot fake cannot
# express, which is why each one had to be a real-browser test or no test at
# all. Visibility, stacking, frame boundaries and tabs are properties of a
# rendered document; a parsed YAML tree has none of them.


async def test_a_hidden_twin_does_not_make_a_visible_control_ambiguous(site):
    """The responsive-site shape: a nav rendered twice, one copy display:none.

    The rung here is `text`, which is what every recorded ladder falls through
    to once its role rung stops matching -- and text is matched against the
    DOM, so `count()` reports the copy nobody can see. This used to be refused
    as ambiguous, and the row failed on a page a person would call
    unambiguous. Counting what is visible is the whole fix.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="hidden twin",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/hidden-twin.html"),
            Step(
                id="s2",
                action="click",
                locators=[Locator(strategy="text", text="Continue", exact=True)],
            ),
            Step(
                id="s3",
                action="assert",
                **{"assert": {"kind": "text_present", "value": "continued"}},
            ),
        ],
    )

    _, _, results = await execute(use_case, site, [{}], step_timeout=5.0)
    assert results[0].ok, results[0].error


async def test_a_row_scoped_locator_clicks_the_row_it_names(site):
    """Three "Edit" buttons, and the recording says which one by saying where.

    Without `within` the only expressible answers were "an Edit button", which
    is refused as ambiguous, and "the second Edit button", which is a claim
    about ordering. This is the third answer, and it is the one a person means.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="scoped",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/rows.html"),
            Step(
                id="s2",
                action="click",
                locators=[
                    Locator(
                        strategy="role",
                        role="button",
                        name="Edit",
                        within=Locator(strategy="role", role="row", has_text="Globex"),
                    )
                ],
            ),
            Step(
                id="s3",
                action="assert",
                **{"assert": {"kind": "text_present", "value": "editing Globex"}},
            ),
        ],
    )

    _, _, results = await execute(use_case, site, [{}], step_timeout=5.0)
    assert results[0].ok, results[0].error


async def test_an_unscoped_locator_on_the_same_page_is_still_refused(site):
    """The companion to the test above, and the reason it is worth having.

    Scoping must be what resolves the ambiguity, not a general loosening. Three
    identical buttons with nothing said about which one is still three.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="unscoped",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/rows.html"),
            Step(
                id="s2",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Edit")],
            ),
        ],
    )

    _, _, results = await execute(use_case, site, [{}], step_timeout=3.0)
    assert not results[0].ok
    assert "3" in (results[0].error or ""), results[0].error


async def test_a_field_inside_an_iframe_can_be_filled(site):
    """An element in a frame is not hard to find from the page: it is absent.

    Both spellings codegen writes for the hop are covered -- the recorded form
    here, and `page.locator(...).content_frame` through the parser test.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="iframe",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/checkout.html"),
            Step(
                id="s2",
                action="fill",
                value="4242424242424242",
                locators=[
                    Locator(
                        strategy="role",
                        role="textbox",
                        name="Card number",
                        frames=["iframe#pay"],
                    )
                ],
            ),
        ],
    )

    _, browser_url, results = await execute(use_case, site, [{}], step_timeout=5.0)
    assert results[0].ok, results[0].error


async def test_an_iframes_contents_appear_in_the_snapshot(site):
    """The other half of the frame work, and the one healing depends on.

    A repair proposed from a failure context can only name controls the
    snapshot contains. While `aria_snapshot` was read from the main frame's
    body alone, every control in a frame was invisible to it -- so a step that
    broke inside a payment form could not be repaired at all.
    """
    config = BrowserConfig(headless=True, timeout_ms=10_000)
    async with PlaywrightSession(config) as browser:
        await browser.page.goto(f"{site}/checkout.html")
        await browser.settle()
        snapshot = await browser.snapshot()

    names = [node.name for node in snapshot]
    assert "Card number" in names, names


async def test_a_click_that_opens_a_tab_is_followed(site):
    """Before this, the run went on driving the page underneath the new tab.

    The failure that produced was the hardest kind to read: the click
    succeeded, and a step several later failed with "could not find" on a page
    that had never been wrong.
    """
    from usecase import UseCase

    use_case = UseCase(
        name="new tab",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/new-tab.html"),
            Step(
                id="s2",
                action="click",
                locators=[
                    Locator(strategy="role", role="link", name="Open the order form")
                ],
            ),
            # Only reachable on the page the link opened.
            Step(
                id="s3",
                action="fill",
                value="A-2048",
                locators=[Locator(strategy="placeholder", text="Order reference")],
            ),
        ],
    )

    _, _, results = await execute(use_case, site, [{}], step_timeout=5.0)
    assert results[0].ok, results[0].error


# ---------------------------------------------------------------------------
# Checking a locator before saving it
# ---------------------------------------------------------------------------


async def test_the_locator_check_says_what_each_rung_actually_matches(site):
    """The answer somebody editing a locator could not get before.

    A rung reads perfectly well and matches nothing, or matches four things.
    Without this the only way to find out was to run the use case, where an
    ambiguous rung shows up as a thirty-second timeout on row one of a batch.
    """
    from engine import probe_locators

    class Settings:
        browser_engine = "chromium"
        browser_headless = True
        replay_step_timeout = 10.0

    report = await probe_locators(
        [
            Locator(strategy="role", role="button", name="Edit"),
            Locator(
                strategy="role",
                role="button",
                name="Edit",
                within=Locator(strategy="role", role="row", has_text="Globex"),
            ),
            Locator(strategy="role", role="button", name="Archive"),
        ],
        url=f"{site}/rows.html",
        settings=Settings(),
        timeout_ms=10_000,
    )

    ambiguous, scoped, missing = report["results"]

    assert ambiguous["visible"] == 3 and not ambiguous["ok"]
    assert "ambiguous" in ambiguous["reason"]

    assert scoped["visible"] == 1 and scoped["ok"]
    assert scoped["reason"] == ""

    assert missing["total"] == 0 and not missing["ok"]
    assert "matches nothing" in missing["reason"]


async def test_the_locator_check_reports_both_counts(site):
    """The gap between the two is worth showing rather than hiding.

    "Matches 2, one of them visible" tells somebody their page carries a hidden
    duplicate -- which they may want to know about their site as much as about
    their locator, and which decides whether the rung is safe to keep.
    """
    from engine import probe_locators

    class Settings:
        browser_engine = "chromium"
        browser_headless = True
        replay_step_timeout = 10.0

    report = await probe_locators(
        [Locator(strategy="text", text="Continue", exact=True)],
        url=f"{site}/hidden-twin.html",
        settings=Settings(),
        timeout_ms=10_000,
    )

    only = report["results"][0]
    assert only["total"] == 2, "the DOM holds both copies"
    assert only["visible"] == 1, "a person sees one"
    assert only["ok"], "and that is not ambiguous"


# ---------------------------------------------------------------------------
# A locator made of row one's data
# ---------------------------------------------------------------------------


def search_script(base: str) -> str:
    """What `playwright codegen` writes for "search by number, pick the hit".

    The last line is the whole problem: the suggestion is addressed by the
    customer *name*, which is not what was typed and is not the same on the
    next row.
    """
    return f'''import asyncio
from playwright.async_api import Playwright, async_playwright, expect


async def run(playwright: Playwright) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto("{base}/search.html")
    await page.get_by_placeholder("Customer number").fill("C-1001")
    await page.get_by_role("option", name="Acme Ltd").click()

    # ---------------------
    await context.close()
    await browser.close()
'''


def searching(base: str):
    """The recording above, parameterised on the customer number as the
    product would parameterise it: the typed value is the input column."""
    from usecase import UseCase

    declared = FieldSet.from_payload(
        [{"name": "customer_number", "value": "C-1001", "secret": False}]
    )
    use_case = build_usecase(
        parse(search_script(base)),
        name="Open a customer",
        description="",
        declared=declared,
    )
    use_case.status = "ready"
    use_case.allowed_domains = ["127.0.0.1"]
    use_case.row_reset = Step(id="reset", action="navigate", url=f"{base}/search.html")
    return use_case


async def test_a_suggestion_picked_by_name_replays_for_a_different_customer(site):
    """The defect, stated as the behaviour that has to hold.

    Row one searches C-1001 and picks "Acme Ltd". Row two searches C-1002,
    whose suggestion reads "Globex Corporation" -- a name the recording has
    never seen and cannot contain. A step that looks for "Acme Ltd" fails on
    every row but the one it was recorded on, which makes the whole recording
    good for exactly one customer.
    """
    use_case = searching(site)

    _, _, results = await execute(
        use_case,
        site,
        [{"customer_number": "C-1001"}, {"customer_number": "C-1002"}],
        step_timeout=5.0,
    )

    assert results[0].ok, results[0].error
    assert results[1].ok, f"row two must not look for row one's customer: {results[1].error}"

    # Not passing by accident: the recorded name is gone from what executes,
    # kept where a reviewer can see it, and the draft said so.
    click = use_case.row_steps[-1]
    assert [loc.describe() for loc in click.locators] == ["role=option"]
    assert any("Acme Ltd" in loc.describe() for loc in click.rejected_locators)
    assert any("row one's answer" in warning for warning in use_case.warnings)


async def test_a_search_that_narrows_to_several_refuses_rather_than_picking_one(site):
    """The other half, and the reason the rewrite is safe.

    Searching "C-100" matches all three customers. The recording does not say
    which one a different row should take -- it only ever saw one -- so the
    honest answer is to stop. Clicking the first would be the same silent
    wrong-record failure in a new costume.

    This is also why the recorded name is deleted rather than demoted to a
    fallback rung: a ladder takes the first rung that matches exactly one, so a
    demoted "Acme Ltd" would sit unused on every row that works and fire on
    exactly the rows that are ambiguous -- acting only when it is certainly
    wrong.
    """
    use_case = searching(site)

    _, _, results = await execute(
        use_case, site, [{"customer_number": "C-100"}], step_timeout=3.0
    )

    assert not results[0].ok
    assert "ambiguous" in (results[0].error or ""), results[0].error


async def test_a_column_from_the_file_can_name_the_suggestion(site):
    """The complete fix, for somebody whose file has the name as well.

    "The option called {{input.customer_name}}" is the locator a person would
    write, and the blanket ban on templating inside a locator made it
    unexpressible -- which is why the recorded name was the only option and the
    recorded name was wrong.
    """
    use_case = searching(site)
    use_case.inputs.append(InputSpec(name="customer_name", required=True))
    use_case.row_steps[-1].locators = [
        Locator(strategy="role", role="option", name="{{input.customer_name}}", exact=True)
    ]

    _, _, results = await execute(
        use_case,
        site,
        [
            {"customer_number": "C-100", "customer_name": "Initech"},
            {"customer_number": "C-100", "customer_name": "Globex Corporation"},
        ],
        step_timeout=5.0,
    )

    assert results[0].ok, results[0].error
    assert results[1].ok, results[1].error


async def test_a_failed_templated_locator_reports_what_it_looked_for(site):
    """The standing objection to templating a locator at all: "when it stops
    matching you cannot tell whether the site changed or the input did."

    Answered by reporting both. The step is named by its definition, which is
    the template, and what was tried is named by the rendered form -- so the
    failure reads "looking for {{input.customer_name}}, tried 'Umbrella PLC'"
    and the reader can see at a glance that the substitution happened and the
    page did not have it. Either half alone leaves the question open.
    """
    use_case = searching(site)
    use_case.inputs.append(InputSpec(name="customer_name", required=True))
    use_case.row_steps[-1].locators = [
        Locator(strategy="role", role="option", name="{{input.customer_name}}", exact=True)
    ]

    _, _, results = await execute(
        use_case,
        site,
        [{"customer_number": "C-1001", "customer_name": "Umbrella PLC"}],
        step_timeout=3.0,
    )

    error = results[0].error or ""
    assert not results[0].ok
    assert "Tried: role=option name=\"Umbrella PLC\"" in error, error
    assert "{{input.customer_name}}" in error, "and which template produced it"


# --- a positional rung that finds the wrong control ------------------------


def rebuilt_use_case(site, *, expect_text: str):
    """One step, addressed by a CSS path, with what it used to say."""
    from usecase import Locator, Step, UseCase

    return UseCase(
        name="rebuilt",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/rebuilt.html"),
            Step(
                id="s2",
                action="click",
                locators=[Locator(strategy="css", selector=".toolbar > button")],
                expect_text=expect_text,
            ),
        ],
    )


async def test_a_css_rung_that_lands_on_a_renamed_control_refuses_to_act(site):
    """The failure that is worse than a failure: the rung matches exactly one
    element, so before this the step clicked it and reported success.

    A snapshot fake cannot express this -- an accessibility tree has no CSS --
    so the check that catches it has to be proved here.
    """
    _, _, results = await execute(
        rebuilt_use_case(site, expect_text="Remove from list"), site, [{}], step_timeout=2.0
    )

    assert not results[0].ok
    assert "not the one that was recorded" in results[0].error
    assert "Delete permanently" in results[0].error


async def test_the_same_step_acts_when_the_control_still_says_the_same_thing(site):
    _, _, results = await execute(
        rebuilt_use_case(site, expect_text="Delete permanently"), site, [{}], step_timeout=2.0
    )

    assert results[0].ok, results[0].error


# --- asserting on an attribute ---------------------------------------------


async def test_an_assertion_can_read_an_attribute(site):
    """The identifier a later step needs is in the link, not in the words a
    person sees."""
    from usecase import Assertion, Locator, Step, UseCase

    def checking(value: str) -> UseCase:
        return UseCase(
            name="attr",
            status="ready",
            allowed_domains=["127.0.0.1"],
            row_steps=[
                Step(id="s1", action="navigate", url=f"{site}/rebuilt.html"),
                Step(
                    id="s2",
                    action="assert",
                    **{
                        "assert": Assertion(
                            kind="attribute_contains",
                            locator=Locator(strategy="role", role="link", name="Receipt"),
                            attribute="href",
                            value=value,
                            timeout_ms=2_000,
                        )
                    },
                ),
            ],
        )

    _, _, held = await execute(checking("/receipt/A-1001"), site, [{}], step_timeout=3.0)
    _, _, failed = await execute(checking("/receipt/A-9999"), site, [{}], step_timeout=3.0)

    assert held[0].ok, held[0].error
    assert not failed[0].ok
    assert "A-1001" in failed[0].error, "the failure says what the attribute actually holds"


# --- a rung that resolves and cannot be acted on ---------------------------


def styled_radio_use_case(site, *, with_the_servers_rung: bool):
    """The recording as it was made, with and without the rung this adds.

    The first locator is what the accessibility tree said. The second is what
    the MCP server reported having run, which is the label -- and the only
    thing on the page a person can actually click.
    """
    from usecase import Locator, Step, UseCase

    ladder = [Locator(strategy="role", role="radio", name="Nayra Asati", exact=True)]
    if with_the_servers_rung:
        ladder.append(Locator(strategy="css", selector="label", has_text="Nayra Asati"))

    return UseCase(
        name="who are you",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/styled-radio.html"),
            Step(id="s2", action="click", locators=ladder, timeout_ms=2_000),
        ],
    )


async def test_a_recording_of_only_the_tree_rung_fails_the_way_it_did_in_production(site):
    """The bug, reproduced. The rung resolves -- there is exactly one radio
    with that name -- and the click cannot be performed, so the step fails
    having found the element it was looking for."""
    _, _, results = await execute(
        styled_radio_use_case(site, with_the_servers_rung=False),
        site, [{}], step_timeout=3.0,
    )

    assert not results[0].ok
    assert "Timeout" in results[0].error or "timeout" in results[0].error


async def test_the_rung_the_server_ran_performs_it(site):
    """And the step succeeds, against the element a person clicks."""
    executor, _, results = await execute(
        styled_radio_use_case(site, with_the_servers_rung=True),
        site, [{}], step_timeout=6.0,
    )

    assert results[0].ok, results[0].error
    # The fall-through is reported as drift, because it is: the recording's
    # preferred locator no longer performs and somebody should know.
    assert executor.locator_drift.get("s2") == 1


async def test_a_failure_on_every_rung_says_they_were_all_found(site):
    """"no element matched" said of three rungs that all matched sends
    somebody looking for a locator problem they do not have."""
    from usecase import Locator, Step, UseCase

    use_case = UseCase(
        name="who are you",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/styled-radio.html"),
            Step(
                id="s2",
                action="click",
                locators=[
                    Locator(strategy="role", role="radio", name="Nayra Asati", exact=True),
                    Locator(strategy="css", selector="input#nayra"),
                ],
                timeout_ms=2_000,
            ),
        ],
    )

    _, _, results = await execute(use_case, site, [{}], step_timeout=6.0)

    assert not results[0].ok
    assert "none could be acted on" in results[0].error


# --- a failure that says why, not only that ---------------------------------


async def test_a_covered_control_reports_what_is_covering_it(site):
    """The permanent fix for a failure that kept coming back. Before this the
    stored reason was "TimeoutError: Locator.click: Timeout 30000ms exceeded"
    and nothing else: the element had been found and something was on top of
    it, and the record did not say so."""
    from usecase import Locator, Step, UseCase

    use_case = UseCase(
        name="recalls",
        status="ready",
        allowed_domains=["127.0.0.1"],
        row_steps=[
            Step(id="s1", action="navigate", url=f"{site}/covered.html"),
            Step(
                id="s2",
                action="click",
                locators=[Locator(strategy="role", role="columnheader", name="Make")],
                timeout_ms=2_000,
            ),
        ],
    )

    _, sink, results = await execute(use_case, site, [{}], step_timeout=3.0)

    assert not results[0].ok
    assert "covering it" in results[0].error, results[0].error
    assert "consent" in results[0].error, "and names what is covering it"

    # And the log itself is on the failure event, for a repair to read.
    failed = [
        event
        for event in sink.events
        if getattr(event, "kind", "") == "step_failed"
    ]
    assert failed, "a failed step records its context"
    assert "intercepts pointer events" in (failed[-1].detail or {}).get("call_log", "")
