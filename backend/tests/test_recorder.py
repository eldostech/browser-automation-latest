"""Recording by hand: the subprocess, and what a recording becomes.

No real browser opens here. ``codegen`` is stood in for by a short Python
script that writes the same file a real one would and then exits, which is the
whole contract this module depends on: a process that runs until the window
closes, and a script on disk afterwards.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest

from codegen import parse
from fields import FieldSet
from recorder import Recorder, RecorderUnavailable
from routers.recordings import build_usecase

pytestmark = pytest.mark.anyio

RECORDED = '''import asyncio
from playwright.async_api import Playwright, async_playwright, expect


async def run(playwright: Playwright) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto("https://example.com/login")
    await page.get_by_label("Username").fill("nitin")
    await page.get_by_label("Password").fill("s3cret-Example-Pw")
    await page.get_by_role("button", name="Sign in").click()
    await page.get_by_placeholder("Order reference").fill("A-1024")
    await page.get_by_role("button", name="Submit").click()
    await expect(page.get_by_text("Thank you")).to_be_visible()

    # ---------------------
    await context.close()
    await browser.close()
'''


@pytest.fixture
def fake_codegen(tmp_path):
    """A stand-in for `npx playwright`, as a command Recorder can spawn.

    It reads `--output=` off its own argv exactly as codegen does, so the
    argument construction is under test rather than assumed, and it exits as
    soon as it has written -- which is what closing the window does.

    A script file rather than `python -c`: the command is a string that has to
    survive being split into argv, and an inline program full of spaces and
    quotes would be testing the splitting rather than the recorder.
    """

    def build(script: str = RECORDED, *, exit_code: int = 0, write: bool = True) -> str:
        program = tmp_path / f"fake_codegen_{abs(hash((script, write, exit_code)))}.py"
        program.write_text(
            textwrap.dedent(
                f"""
                import sys

                SCRIPT = {script!r}
                for arg in sys.argv[1:]:
                    if arg.startswith("--output=") and {write!r}:
                        with open(arg.split("=", 1)[1], "w", encoding="utf-8") as handle:
                            handle.write(SCRIPT)
                sys.exit({exit_code})
                """
            ).strip(),
            encoding="utf-8",
        )
        return f'"{sys.executable}" "{program}"'

    return build


async def finish(recorder: Recorder, session) -> None:
    """Wait for the recorder's own watcher task, as the poller would."""
    assert session.task is not None
    await asyncio.wait_for(session.task, timeout=15)


# --- the subprocess --------------------------------------------------------


async def test_a_closed_window_leaves_a_parsed_recording(fake_codegen):
    recorder = Recorder(command=fake_codegen())
    session = await recorder.start(
        start_url="https://example.com/login",
        name="Orders",
        workspace_id="ws",
        owner_id="u",
        owner_email="u@example.com",
    )
    await finish(recorder, session)

    assert session.status == "ready"
    assert session.recording is not None
    assert [step.action for step in session.recording.steps] == [
        "navigate",
        "fill",
        "fill",
        "click",
        "fill",
        "click",
    ]
    await recorder.shutdown()


async def test_a_recording_that_wrote_nothing_says_so(fake_codegen):
    """Closing the window immediately is the common way this goes wrong, and
    an empty file is indistinguishable from a crash unless it is named."""
    recorder = Recorder(command=fake_codegen(write=False))
    session = await recorder.start(
        start_url="https://example.com/",
        name="x",
        workspace_id="ws",
        owner_id=None,
        owner_email="",
    )
    await finish(recorder, session)

    assert session.status == "failed"
    assert "wrote nothing" in session.error
    await recorder.shutdown()


async def test_a_script_with_no_actions_is_a_failure_not_an_empty_use_case(fake_codegen):
    empty = """import asyncio
from playwright.async_api import Playwright, async_playwright, expect


async def run(playwright: Playwright) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()

    # ---------------------
    await context.close()
    await browser.close()
"""
    recorder = Recorder(command=fake_codegen(empty))
    session = await recorder.start(
        start_url="https://example.com/",
        name="x",
        workspace_id="ws",
        owner_id=None,
        owner_email="",
    )
    await finish(recorder, session)

    assert session.status == "failed"
    assert "no actions were recorded" in session.error
    await recorder.shutdown()


async def test_a_recording_belongs_to_the_workspace_that_started_it(fake_codegen):
    recorder = Recorder(command=fake_codegen())
    session = await recorder.start(
        start_url="https://example.com/",
        name="x",
        workspace_id="ws-a",
        owner_id=None,
        owner_email="",
    )
    await finish(recorder, session)

    assert recorder.get(session.id, "ws-a") is not None
    assert recorder.get(session.id, "ws-b") is None, "another tenant must not see it"
    assert recorder.list("ws-b") == []
    await recorder.shutdown()


async def test_shutdown_closes_windows_and_forgets_sessions(fake_codegen):
    """A headed browser that outlives its backend is a window nobody owns."""
    recorder = Recorder(command=fake_codegen())
    session = await recorder.start(
        start_url="https://example.com/",
        name="x",
        workspace_id="ws",
        owner_id=None,
        owner_email="",
    )
    await finish(recorder, session)
    await recorder.shutdown()

    assert recorder.list("ws") == []
    assert session.workdir is not None and not session.workdir.exists()


async def test_recording_can_be_switched_off():
    recorder = Recorder(enabled=False)
    ok, reason = recorder.available()
    assert not ok
    assert "display" in reason

    with pytest.raises(RecorderUnavailable):
        await recorder.start(
            start_url="https://example.com/",
            name="x",
            workspace_id="ws",
            owner_id=None,
            owner_email="",
        )


async def test_missing_tooling_is_reported_rather_than_crashing_at_spawn():
    recorder = Recorder(command="definitely-not-a-real-binary playwright")
    ok, reason = recorder.available()
    assert not ok
    assert "could not be found" in reason


# --- what a recording becomes ----------------------------------------------


def declared(*items: tuple[str, str, bool]) -> FieldSet:
    return FieldSet.from_payload(
        [{"name": name, "value": value, "secret": secret} for name, value, secret in items]
    )


def built(fields: FieldSet):
    return build_usecase(
        parse(RECORDED), name="Orders", description="", declared=fields
    )


def test_typed_values_become_templates():
    use_case = built(
        declared(
            ("username", "nitin", True),
            ("password", "s3cret-Example-Pw", True),
            ("reference", "A-1024", False),
        )
    )
    values = [step.value for step in use_case.all_steps if step.value]
    assert "{{secret.password}}" in values
    assert "{{input.reference}}" in values
    assert "s3cret-Example-Pw" not in str(values), "a credential must not survive in the steps"


def test_secrets_become_slots_and_inputs_become_columns():
    use_case = built(
        declared(("password", "s3cret-Example-Pw", True), ("reference", "A-1024", False))
    )
    assert [s.name for s in use_case.secrets] == ["password"]
    assert [i.name for i in use_case.inputs] == ["reference"]


def test_the_sign_in_becomes_setup_and_the_rest_is_per_row():
    """A batch shares one session, so the login must not run once per row.

    The boundary is the last step that types a declared secret, plus the click
    that submits it -- which is where a person stops supplying credentials and
    starts doing the work.
    """
    use_case = built(
        declared(
            ("username", "nitin", True),
            ("password", "s3cret-Example-Pw", True),
            ("reference", "A-1024", False),
        )
    )
    assert [step.action for step in use_case.setup_steps] == [
        "navigate",
        "fill",
        "fill",
        "click",
    ]
    assert [step.action for step in use_case.row_steps] == ["fill", "click", "assert"]


def test_with_no_secrets_nothing_is_setup():
    """Putting the first few steps in setup anyway would skip them on every row
    but the first."""
    use_case = built(declared(("reference", "A-1024", False)))
    assert use_case.setup_steps == []
    assert any("once per row" in w for w in use_case.warnings)


def test_a_recorded_check_becomes_an_assertion_step():
    use_case = built(declared(("reference", "A-1024", False)))
    assert use_case.row_steps[-1].action == "assert"
    assert use_case.row_steps[-1].assertion.kind == "element_visible"


def test_a_recording_with_no_checks_is_warned_about():
    without = RECORDED.replace(
        '    await expect(page.get_by_text("Thank you")).to_be_visible()\n', ""
    )
    use_case = build_usecase(
        parse(without), name="x", description="", declared=declared(("a", "A-1024", False))
    )
    assert any("fail silently" in w for w in use_case.warnings)


def test_visited_domains_become_the_allowlist():
    """Still derived from what was visited, and still closed by default.

    The recording's own origin is now held as ``{{env.base_url}}`` rather than
    as its literal host, so promoting the document does not have to remember to
    widen the allowlist -- and cannot widen it to two environments at once. It
    resolves to exactly one host at replay: the recorded one unless the
    deployment says otherwise.
    """
    use_case = built(declared(("reference", "A-1024", False)))
    assert use_case.allowed_domains == ["{{env.base_url}}"]
    assert use_case.base_url == "https://example.com"


def test_a_saved_recording_is_a_draft():
    """The split, the parameterisation and the assertions are all inferred, so
    a person publishes it -- the same gate a distilled recording passes."""
    assert built(declared(("reference", "A-1024", False))).status == "draft"
    assert built(declared(("reference", "A-1024", False))).runnable is False


def test_dropped_lines_are_carried_into_the_use_case():
    """A shape the parser cannot represent becomes a line a person reads.

    The example here used to be an iframe, which is recordable now. It is a
    `filter(has=...)` instead -- that takes a locator rather than a string, and
    the schema has no rung for it. The point of the test is unchanged: what
    cannot be represented is *reported*, never approximated, because an
    approximation is a step that clicks something adjacent on row one.
    """
    unrepresentable = RECORDED.replace(
        '    await page.get_by_placeholder("Order reference").fill("A-1024")\n',
        '    await page.get_by_role("row").filter(has=page.get_by_text("A-1024"))'
        '.get_by_role("textbox").fill("A-1024")\n',
    )
    use_case = build_usecase(
        parse(unrepresentable), name="x", description="", declared=declared(("a", "nitin", False))
    )
    assert use_case.dropped
    assert any("could not be represented" in w for w in use_case.warnings)


def test_a_sign_in_value_left_unmarked_names_itself_in_the_refusal():
    """The failure a person actually hits when they record a login.

    A field that is not a secret becomes a per-row input, and the sign-in steps
    become setup -- so an unmarked value typed while signing in lands in setup
    as ``{{input.x}}``, which setup cannot have. pydantic catches it, but
    reports it against a generated step id; the message has to name the field
    the person chose, and say what to do instead.
    """
    with pytest.raises(ValueError) as caught:
        built(
            declared(
                ("username", "nitin", False),  # left unticked
                ("password", "s3cret-Example-Pw", True),
            )
        )

    message = str(caught.value)
    assert "'username'" in message
    assert "secret" in message
    assert "s6" not in message, "a step id is not something the person chose"


def test_a_sign_in_value_marked_secret_converts_cleanly():
    """The fix the message recommends has to actually work."""
    use_case = built(
        declared(
            ("username", "nitin", True),
            ("password", "s3cret-Example-Pw", True),
        )
    )

    assert use_case.setup_steps, "signing in belongs to setup"
    assert {s.name for s in use_case.secrets} == {"username", "password"}
    assert not [
        name
        for step in use_case.setup_steps
        for kind, name in step.references()
        if kind == "input"
    ]


# --- telling two fields apart ----------------------------------------------

SAME_TEXT_TWICE = '''import asyncio
from playwright.async_api import Playwright, async_playwright, expect


async def run(playwright: Playwright) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto("https://example.com/new")
    await page.get_by_label("Name").fill("test")
    await page.get_by_label("Description").fill("test")
    await page.get_by_role("button", name="Create").click()
'''


def positioned(*items: tuple[str, int, str]) -> FieldSet:
    return FieldSet.from_payload(
        [
            {"name": name, "index": index, "value": value, "secret": False}
            for name, index, value in items
        ]
    )


def test_two_fields_typed_with_the_same_text_stay_separate():
    """The bug that made a recording unpublishable, and wrong before that.

    Substitutions used to be keyed by the recorded value, so typing "test" into
    both the name and the description collapsed them onto one entry: both steps
    took whichever field was declared last, and the other was left declared but
    referenced by nothing. That surfaced much later, as a publish refusing an
    input no step reads -- and had it published, it would have typed the
    description into the name box.
    """
    use_case = build_usecase(
        parse(SAME_TEXT_TWICE),
        name="Create",
        description="",
        declared=positioned(("project_name", 0, "test"), ("summary", 1, "test")),
    )

    typed = [step.value for step in use_case.all_steps if step.value]
    assert typed == ["{{input.project_name}}", "{{input.summary}}"]

    referenced = {
        name
        for step in use_case.all_steps
        for kind, name in step.references()
        if kind == "input"
    }
    assert referenced == {"project_name", "summary"}, "neither field is left unread"


def test_a_typed_value_nobody_declared_keeps_what_was_recorded():
    """Leaving a value undeclared means "use this literal", not "drop it"."""
    use_case = build_usecase(
        parse(SAME_TEXT_TWICE),
        name="Create",
        description="",
        declared=positioned(("summary", 1, "test")),
    )

    assert [s.value for s in use_case.all_steps if s.value] == [
        "test",
        "{{input.summary}}",
    ]


def test_without_positions_two_fields_sharing_a_value_are_refused():
    """A client that sends no positions cannot have this resolved for it.

    Silently letting one field win both steps is what produced the bug above,
    so the ambiguity is named instead.
    """
    with pytest.raises(ValueError) as caught:
        build_usecase(
            parse(SAME_TEXT_TWICE),
            name="Create",
            description="",
            declared=declared(("project_name", "test", False), ("summary", "test", False)),
        )

    assert "same recorded value" in str(caught.value)


# --- promoting one document between environments ---------------------------


def test_the_recorded_origin_is_bound_so_the_document_can_move():
    """What makes dev, UAT and production one document rather than three.

    A recording holds the addresses of the environment it was made in. Binding
    them to ``{{env.base_url}}`` at save is what lets the same bytes run
    everywhere; keeping the recorded origin on the use case is what lets dev
    run with no configuration at all.
    """
    use_case = built(declared(("password", "s3cret-Example-Pw", True)))

    assert use_case.base_url == "https://example.com"
    navigations = [s.url for s in use_case.all_steps if s.action == "navigate"]
    assert navigations == ["{{env.base_url}}/login"]
    assert use_case.allowed_domains == ["{{env.base_url}}"]


def test_a_third_party_host_is_left_alone():
    """Only the thing being promoted moves; an identity provider does not."""
    recording = parse(
        RECORDED.replace(
            'await page.goto("https://example.com/login")',
            'await page.goto("https://example.com/login")\n'
            '    await page.goto("https://sso.vendor.com/authorize")',
        )
    )
    use_case = build_usecase(
        recording, name="x", description="", declared=declared()
    )

    navigations = [s.url for s in use_case.all_steps if s.action == "navigate"]
    assert navigations == ["{{env.base_url}}/login", "https://sso.vendor.com/authorize"]
    assert use_case.allowed_domains == ["{{env.base_url}}", "sso.vendor.com"]


# ---------------------------------------------------------------------------
# Recording behind single sign-on
# ---------------------------------------------------------------------------
#
# The whole flow, because the three defects here compound: the identity
# provider becomes the use case's own address, the sign-in request keeps a
# `state` the provider will refuse the second time it sees it, and the page it
# redirected back to is recorded as a step that replays a spent code.


SSO = '''import asyncio
from playwright.async_api import Playwright, async_playwright, expect


async def run(playwright: Playwright) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto("https://login.example.com/oauth2/authorize?client_id=8b21&response_type=code&redirect_uri=https%3A%2F%2Fcrm.example.com%2Fcb&scope=openid&state=Ab9xQ2zKp&nonce=Nn41Kd")
    await page.get_by_label("Email").fill("ada@example.com")
    await page.get_by_label("Password").fill("s3cret-Example-Pw")
    await page.get_by_role("button", name="Sign in").click()
    await page.goto("https://crm.example.com/cb?code=0.AXkAr9&state=Ab9xQ2zKp&session_state=4f1c")
    await page.goto("https://crm.example.com/orders?ref=A-1024&sessionDataKey=91ab")
    await page.get_by_placeholder("Order reference").fill("A-1024")

    # ---------------------
    await context.close()
    await browser.close()
'''


def signed_in():
    return build_usecase(
        parse(SSO),
        name="Orders",
        description="",
        declared=declared(
            ("email", "ada@example.com", True),
            ("password", "s3cret-Example-Pw", True),
            ("reference", "A-1024", False),
        ),
    )


def test_the_use_case_belongs_to_the_application_not_the_identity_provider():
    """`{{env.base_url}}` used to bind to login.example.com, because that is
    where the address bar was when recording started.

    Promoting the use case to UAT then repointed the *identity provider* at the
    UAT address -- which nobody meant, and which is very hard to see in a diff.
    """
    use_case = signed_in()

    assert use_case.base_url == "https://crm.example.com"
    assert use_case.target == "crm"


def test_the_identity_provider_is_still_allowed_to_be_visited():
    """Fixing which host is *ours* must not stop the sign-in reaching theirs."""
    use_case = signed_in()

    assert "login.example.com" in use_case.allowed_domains
    assert "{{env.base_url}}" in use_case.allowed_domains


def test_the_sign_in_request_keeps_its_address_and_loses_its_single_use_parts():
    use_case = signed_in()

    authorize = use_case.setup_steps[0]
    assert authorize.url is not None
    assert authorize.url.startswith("https://login.example.com/oauth2/authorize")
    assert "client_id=8b21" in authorize.url, "still a valid request"
    assert "state=" not in authorize.url and "nonce=" not in authorize.url


def test_the_page_the_provider_redirected_back_to_is_not_a_step():
    """Nobody types a callback address, and everything in it is spent. The
    sign-in steps above it put the browser there again by themselves."""
    use_case = signed_in()

    urls = [step.url or "" for step in use_case.all_steps]
    assert not any("/cb?" in url for url in urls)
    assert any("one-time code" in warning for warning in use_case.warnings)


def test_a_session_key_is_removed_while_the_row_value_survives():
    """The two live in the same query string, and only one of them is data
    from the sign-in."""
    use_case = signed_in()

    orders = [step for step in use_case.row_steps if step.action == "navigate"][0]
    assert orders.url == "{{env.base_url}}/orders?ref={{input.reference}}"


def test_the_draft_says_what_it_took_out_of_the_addresses():
    use_case = signed_in()

    notes = [w for w in use_case.warnings if "address" in w]
    assert notes, "a person has to be told the recording was edited"
    assert any("'state'" in note for note in notes)
