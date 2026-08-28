"""The executor: a recorded use case replayed with no model in the loop.

The first two tests are the load-bearing ones. Everything else in this feature
exists to make them true.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import FakeMCPSession, FakeTool, RecordingSink
from mcp_client import ToolOutcome
from replay import ScriptBlocked, StepFailed, UseCaseExecutor
from usecase import (
    Assertion,
    FormField,
    InputSpec,
    Locator,
    SecretSpec,
    Step,
    UseCase,
    WaitFor,
)

SIGNED_OUT = """### Page
- Page URL: https://example.com/signin
### Snapshot
```yaml
- textbox "Username" [ref=e1]
- textbox "Password" [ref=e2]
- button "Sign in" [ref=e3]
```"""

SIGNED_IN = """### Page
- Page URL: https://example.com/dashboard
- Page Title: Dashboard
### Snapshot
```yaml
- heading "Welcome back" [ref=e9]
- button "Submit" [ref=e10]
- status "Score" [ref=e11]: 92%
```"""


def role(name: str, kind: str = "button", nth: int = 0) -> Locator:
    return Locator(strategy="role", role=kind, name=name, nth=nth)


class ScriptedMCP(FakeMCPSession):
    """A fake session whose page changes the way a real one would.

    ``routes`` maps a URL substring to the page it serves, so navigating to
    ``/signin`` shows the signed-out page however many times you do it. Without
    that, a fake that blindly advances on every navigation makes tests pass or
    fail for reasons the real system would never produce.
    """

    def __init__(
        self,
        pages: list[str] | None = None,
        routes: dict[str, str] | None = None,
        advance_on: set[str] | None = None,
        **handlers,
    ) -> None:
        self.pages = pages or [SIGNED_OUT]
        self.routes = routes or {}
        #: Tools that move the session to the next page, e.g. a sign-in click.
        self.advance_on = advance_on or set()
        self.page_index = 0
        tools = [
            FakeTool("browser_snapshot", handler=self._snapshot),
            FakeTool("browser_navigate", handler=self._navigate),
            FakeTool("browser_click", handler=handlers.get("click") or self._act("browser_click")),
            FakeTool("browser_type", handler=handlers.get("type") or self._ok),
            FakeTool("browser_fill_form", handler=handlers.get("fill_form") or self._ok),
            FakeTool("browser_press_key", handler=self._ok),
            FakeTool("browser_wait_for", handler=self._ok),
            FakeTool("browser_take_screenshot", handler=self._shot),
            FakeTool("browser_run_code_unsafe", handler=self._ok),
        ]
        super().__init__(tools)

    _pinned: str | None = None

    @property
    def page(self) -> str:
        return self._pinned or self.pages[min(self.page_index, len(self.pages) - 1)]

    def advance(self) -> None:
        self._pinned = None
        self.page_index += 1

    def _snapshot(self, arguments):
        return ToolOutcome(name="browser_snapshot", text=self.page, duration_ms=1)

    def _navigate(self, arguments):
        url = str(arguments.get("url") or "")
        for fragment, page in self.routes.items():
            if fragment in url:
                self._pinned = page
                return ToolOutcome(name="browser_navigate", text=page, duration_ms=2)
        self._pinned = None
        self.advance()
        return ToolOutcome(name="browser_navigate", text=self.page, duration_ms=2)

    def _ok(self, arguments):
        return ToolOutcome(name="ok", text="done", duration_ms=1)

    def _act(self, tool: str):
        """A handler that advances the page when this tool is one that should."""

        def handler(arguments):
            if tool in self.advance_on:
                self.advance()
                return ToolOutcome(name=tool, text=self.page, duration_ms=1)
            return ToolOutcome(name=tool, text="done", duration_ms=1)

        return handler

    def _shot(self, arguments):
        return ToolOutcome(name="shot", images=[("image/png", b"PNG")], duration_ms=1)

    def calls_to(self, tool: str) -> list[dict]:
        return [args for name, args in self.calls if name == tool]


def executor(usecase: UseCase, mcp: ScriptedMCP, **kwargs) -> UseCaseExecutor:
    kwargs.setdefault("step_timeout", 1.0)
    return UseCaseExecutor(usecase, mcp, RecordingSink(), run_id="r1", **kwargs)


# --- the two guarantees the whole feature rests on -------------------------


def test_the_executor_module_never_imports_an_llm():
    """Structural, not a promise someone has to remember."""
    source = (Path(__file__).parent.parent / "replay.py").read_text(encoding="utf-8")
    assert "import llm" not in source
    assert "from llm" not in source


def test_the_executor_cannot_be_given_a_model():
    import inspect

    params = inspect.signature(UseCaseExecutor.__init__).parameters
    assert not any("llm" in name or "model" in name for name in params)

    with pytest.raises(TypeError):
        UseCaseExecutor(
            UseCase(name="x"), ScriptedMCP(), RecordingSink(), run_id="r", llm=object()
        )


async def test_a_full_replay_costs_zero_tokens():
    """The point of the feature, asserted end to end."""
    if "llm" in sys.modules:
        del sys.modules["llm"]

    use_case = UseCase(
        name="Sign in and read the score",
        secrets=[SecretSpec(name="username"), SecretSpec(name="password")],
        inputs=[InputSpec(name="record_url", type="url")],
        allowed_domains=["example.com"],
        setup_steps=[
            Step(id="u1", action="navigate", url="https://example.com/signin"),
            Step(
                id="u2",
                action="fill_form",
                fields=[
                    FormField(name="Username", value="{{secret.username}}",
                              locators=[role("Username", "textbox")]),
                    FormField(name="Password", value="{{secret.password}}",
                              locators=[role("Password", "textbox")]),
                ],
            ),
            Step(id="u3", action="click", locators=[role("Sign in")]),
        ],
        row_reset=Step(id="reset", action="navigate", url="{{input.record_url}}"),
        row_steps=[
            Step(id="s1", action="assert",
                 assertion=Assertion(kind="url_contains", value="/signin", negate=True)),
            Step(id="s2", action="extract", locators=[role("Score", "status")], output="score"),
        ],
        outputs=["score"],
    )

    # /signin always serves the signed-out page; clicking "Sign in" is what
    # moves the session on, exactly as the real site behaves.
    mcp = ScriptedMCP(
        [SIGNED_OUT, SIGNED_IN],
        routes={"/signin": SIGNED_OUT},
        advance_on={"browser_click"},
    )
    runner = executor(use_case, mcp, secrets={"username": "u", "password": "pw"})

    setup = await runner.run_setup()
    assert setup.ok, setup.error

    row = await runner.run_row({"record_url": "https://example.com/record/1"})

    assert row.ok, row.error
    assert row.llm_calls == 0 and row.llm_tokens == 0
    assert row.outputs == {"score": "92%"}
    assert "llm" not in sys.modules, "nothing in the replay path may import a model client"


# --- the locator ladder ----------------------------------------------------


async def test_a_role_locator_is_resolved_against_a_live_snapshot():
    """The recorded ref is long dead; the live one is looked up now."""
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    result = await executor(use_case, mcp).run_row({})

    assert result.ok
    assert mcp.calls_to("browser_click")[0]["target"] == "ref=e3"


async def test_the_ladder_falls_through_and_reports_drift():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(
                id="s1",
                action="click",
                locators=[role("Button That Moved"), Locator(strategy="css", selector="#submit")],
            )
        ],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    runner = executor(use_case, mcp)
    result = await runner.run_row({})

    assert result.ok
    assert mcp.calls_to("browser_click")[0]["target"] == "#submit"
    assert runner.locator_drift == {"s1": 1}, "falling to rung 1 is the drift warning"
    assert result.steps[-1].locator_rung == 1


async def test_the_first_rung_matching_reports_no_drift():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(id="s1", action="click",
                 locators=[role("Sign in"), Locator(strategy="css", selector="#submit")])
        ],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    runner = executor(use_case, mcp)
    await runner.run_row({})
    assert runner.locator_drift == {}


async def test_nothing_matching_names_what_was_tried_and_what_is_there():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Nonexistent")])],
    )
    result = await executor(use_case, ScriptedMCP([SIGNED_OUT])).run_row({})

    assert not result.ok
    assert result.failed_step_id == "s1"
    assert "Nonexistent" in result.error
    assert "page currently shows" in result.error


# --- assertions ------------------------------------------------------------


async def test_a_satisfied_assertion_passes():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(id="s1", action="assert",
                 assertion=Assertion(kind="url_contains", value="/dashboard"))
        ],
    )
    assert (await executor(use_case, ScriptedMCP([SIGNED_IN])).run_row({})).ok


async def test_a_failing_assertion_fails_the_row_rather_than_reporting_success():
    """Without this a batch reports success on 1,000 rows that did nothing."""
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(
                id="s1",
                action="assert",
                assertion=Assertion(kind="url_contains", value="/dashboard", timeout_ms=0),
            )
        ],
    )
    result = await executor(use_case, ScriptedMCP([SIGNED_OUT])).run_row({})

    assert not result.ok
    assert result.failed_step_id == "s1"
    assert "did not hold" in result.error


async def test_a_negated_assertion_works():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(id="s1", action="assert",
                 assertion=Assertion(kind="url_contains", value="/signin", negate=True))
        ],
    )
    assert (await executor(use_case, ScriptedMCP([SIGNED_IN])).run_row({})).ok


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("text_present", "Welcome back", True),
        ("text_present", "Not on this page", False),
        ("title_contains", "Dashboard", True),
        ("title_contains", "Nope", False),
    ],
)
async def test_assertion_kinds(kind: str, value: str, expected: bool):
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(id="s1", action="assert",
                 assertion=Assertion(kind=kind, value=value, timeout_ms=0))
        ],
    )
    assert (await executor(use_case, ScriptedMCP([SIGNED_IN])).run_row({})).ok is expected


async def test_element_count_assertion():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(
                id="s1",
                action="assert",
                assertion=Assertion(
                    kind="element_count", locator=role("Sign in"), count=1, timeout_ms=0
                ),
            )
        ],
    )
    assert (await executor(use_case, ScriptedMCP([SIGNED_OUT])).run_row({})).ok


# --- secrets ---------------------------------------------------------------


async def test_secrets_are_rendered_into_the_call_but_redacted_from_events():
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        secrets=[SecretSpec(name="password")],
        row_steps=[
            Step(id="s1", action="fill", locators=[role("Password", "textbox")],
                 value="{{secret.password}}")
        ],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    sink = RecordingSink()
    runner = UseCaseExecutor(
        use_case, mcp, sink, run_id="r1", secrets={"password": "s3cret-Example-Pw!"}
    )
    result = await runner.run_row({})

    assert result.ok
    assert mcp.calls_to("browser_type")[0]["text"] == "s3cret-Example-Pw!", "the page gets the real value"
    assert "s3cret-Example-Pw!" not in str([e.model_dump() for e in sink.events])


async def test_a_missing_secret_stops_rather_than_typing_nothing():
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        secrets=[SecretSpec(name="password")],
        row_steps=[
            Step(id="s1", action="fill", locators=[role("Password", "textbox")],
                 value="{{secret.password}}")
        ],
    )
    result = await executor(use_case, ScriptedMCP([SIGNED_OUT]), secrets={}).run_row({})

    assert not result.ok
    assert "secret.password" in result.error


async def test_a_missing_required_input_fails_before_the_browser_is_touched():
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        inputs=[InputSpec(name="record_url", type="url")],
        row_steps=[Step(id="s1", action="navigate", url="{{input.record_url}}")],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    result = await executor(use_case, mcp).run_row({})

    assert not result.ok
    assert "record_url" in result.error
    assert mcp.calls == [], "not a single tool call was made"


# --- policy ----------------------------------------------------------------


async def test_the_domain_allowlist_still_applies_to_a_replay():
    """A use case cannot escape the allowlist because a recording was approved."""
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="navigate", url="https://evil.example.net/x")],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    result = await executor(use_case, mcp).run_row({})

    assert not result.ok
    assert "not in the allowed domain list" in result.error
    assert mcp.calls_to("browser_navigate") == []


async def test_a_script_step_refuses_to_run_without_the_opt_in():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="script", code="await page.evaluate('1')")],
    )
    result = await executor(use_case, ScriptedMCP([SIGNED_OUT])).run_row({})

    assert not result.ok
    assert "allow_scripts" in result.error


async def test_a_script_step_runs_once_opted_in():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"], allow_scripts=True,
        row_steps=[Step(id="s1", action="script", code="await page.evaluate('1')")],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    assert (await executor(use_case, mcp).run_row({})).ok
    assert mcp.calls_to("browser_run_code_unsafe")[0]["code"] == "await page.evaluate('1')"


# --- failure handling ------------------------------------------------------


async def test_an_optional_step_that_fails_does_not_stop_the_row():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(id="s1", action="click", locators=[role("Nonexistent")], optional=True),
            Step(id="s2", action="click", locators=[role("Sign in")]),
        ],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    result = await executor(use_case, mcp).run_row({})

    assert result.ok
    assert result.steps[0].skipped is True
    assert len(mcp.calls_to("browser_click")) == 1


async def test_on_failure_continue_behaves_the_same_way():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(id="s1", action="click", locators=[role("Nonexistent")], on_failure="continue"),
            Step(id="s2", action="click", locators=[role("Sign in")]),
        ],
    )
    assert (await executor(use_case, ScriptedMCP([SIGNED_OUT])).run_row({})).ok


async def test_a_failing_row_captures_a_screenshot():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Nonexistent")])],
    )
    sink = RecordingSink()
    runner = UseCaseExecutor(use_case, ScriptedMCP([SIGNED_OUT]), sink, run_id="r1")
    await runner.run_row({})

    assert [e.caption for e in sink.of_type("screenshot")] == ["failure"]


async def test_a_tool_error_fails_the_step_with_the_server_message():
    def boom(arguments):
        return ToolOutcome(name="browser_click", text="element is not visible", is_error=True)

    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )
    result = await executor(use_case, ScriptedMCP([SIGNED_OUT], click=boom)).run_row({})

    assert not result.ok
    assert "not visible" in result.error


# --- setup vs rows ---------------------------------------------------------


async def test_setup_runs_once_and_rows_run_repeatedly():
    """The reason the schema is split in three."""
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        secrets=[SecretSpec(name="password")],
        setup_steps=[
            Step(id="u1", action="fill", locators=[role("Password", "textbox")],
                 value="{{secret.password}}")
        ],
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    runner = executor(use_case, mcp, secrets={"password": "pw"})

    await runner.run_setup()
    for _ in range(3):
        assert (await runner.run_row({})).ok

    assert len(mcp.calls_to("browser_type")) == 1, "signed in once, not once per row"
    assert len(mcp.calls_to("browser_click")) == 3


async def test_row_reset_runs_before_every_row():
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        row_reset=Step(id="reset", action="navigate", url="https://example.com/start"),
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )
    mcp = ScriptedMCP([SIGNED_OUT, SIGNED_OUT, SIGNED_OUT, SIGNED_OUT])
    runner = executor(use_case, mcp)
    for _ in range(3):
        await runner.run_row({})

    assert len(mcp.calls_to("browser_navigate")) == 3


async def test_a_failing_setup_reports_the_step_rather_than_raising():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        setup_steps=[Step(id="u1", action="click", locators=[role("Nonexistent")])],
    )
    result = await executor(use_case, ScriptedMCP([SIGNED_OUT])).run_setup()

    assert not result.ok
    assert result.failed_step_id == "u1"


# --- session check ---------------------------------------------------------


async def test_the_session_check_passes_while_signed_in():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        session_check=Assertion(kind="url_contains", value="/signin", negate=True),
        row_steps=[Step(id="s1", action="click", locators=[role("Submit")])],
    )
    assert await executor(use_case, ScriptedMCP([SIGNED_IN])).check_session() is True


async def test_the_session_check_fails_once_signed_out():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        session_check=Assertion(kind="url_contains", value="/signin", negate=True),
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )
    assert await executor(use_case, ScriptedMCP([SIGNED_OUT])).check_session() is False


async def test_no_session_check_means_always_healthy():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )
    assert await executor(use_case, ScriptedMCP([SIGNED_OUT])).check_session() is True


# --- events ----------------------------------------------------------------


async def test_a_replay_emits_the_same_events_the_dashboard_already_renders():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")],
                        description="Sign in button")],
    )
    sink = RecordingSink()
    await UseCaseExecutor(use_case, ScriptedMCP([SIGNED_OUT]), sink, run_id="r1").run_row({})

    assert [e.step_id for e in sink.of_type("step_started")] == ["s1"]
    assert sink.of_type("step_started")[0].phase == "row"
    assert sink.of_type("step_started")[0].description == "Sign in button"

    finished = sink.of_type("step_finished")[0]
    assert finished.ok is True
    assert finished.matched_locator == 'role=button name="Sign in"'
    assert finished.locator_rung == 0

    # The existing timeline works unchanged because these are the same events.
    assert sink.of_type("tool_call") and sink.of_type("tool_result")


async def test_a_wait_step_is_honoured():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="wait", wait_for=WaitFor(kind="time", seconds=0.01))],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    assert (await executor(use_case, mcp).run_row({})).ok
    assert mcp.calls_to("browser_wait_for")[0]["time"] == 0.01


# --- screenshots during replay ---------------------------------------------
#
# Executing a use case used to photograph only failures, so a successful run --
# the overwhelming majority -- left nothing to audit or to look at when a
# result was questioned later. Headless was never the reason: no screenshot
# code anywhere consults it.


def _one_step_usecase() -> UseCase:
    return UseCase(
        name="x",
        allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )


def _captions(runner: UseCaseExecutor) -> list[str]:
    return [e.caption for e in runner.sink.events if e.type == "screenshot"]


async def test_a_successful_row_is_photographed_by_default():
    """The default answers "what happened to record 700", at one image per row."""
    runner = executor(_one_step_usecase(), ScriptedMCP([SIGNED_OUT]))
    result = await runner.run_row({})

    assert result.ok, result.error
    assert _captions(runner) == ["row finished"]


async def test_off_captures_nothing():
    runner = executor(_one_step_usecase(), ScriptedMCP([SIGNED_OUT]), screenshots="off")
    assert (await runner.run_row({})).ok
    assert _captions(runner) == []


async def test_failure_only_mode_leaves_a_successful_row_unphotographed():
    """The behaviour that made a successful replay invisible, kept as an option."""
    runner = executor(_one_step_usecase(), ScriptedMCP([SIGNED_OUT]), screenshots="failure")
    assert (await runner.run_row({})).ok
    assert _captions(runner) == []


async def test_every_step_photographs_each_step_and_the_end():
    runner = executor(
        _one_step_usecase(), ScriptedMCP([SIGNED_OUT]), screenshots="every_step"
    )
    assert (await runner.run_row({})).ok

    captions = _captions(runner)
    assert "after s1" in captions
    assert captions[-1] == "row finished"


async def test_a_failed_row_is_photographed_where_it_failed():
    """A failure shot beats an end-of-row shot: it shows the moment, not the aftermath."""
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Nothing Like This")])],
    )
    runner = executor(use_case, ScriptedMCP([SIGNED_OUT]))
    result = await runner.run_row({})

    assert not result.ok
    assert _captions(runner) == ["failure"]


# --- the page must be given time to settle ----------------------------------
#
# The failure these guard: a single-page app that has just navigated exposes an
# EMPTY accessibility tree for the first moment. Resolution used to run once,
# immediately, so every element step failed with "no element matched" before
# the page had drawn anything. Recording never showed it, because the model's
# think-time between actions is an accidental sleep of several seconds; replay
# has no model and no pause.

BLANK_PAGE = """### Page
- Page URL: https://example.com/signin
### Snapshot
```yaml
```"""


def _slow_render(mcp: ScriptedMCP, blank_snapshots: int) -> None:
    """Make the first N snapshot calls return an empty page, as a real SPA does."""
    remaining = {"n": blank_snapshots}

    def handler(arguments):
        if remaining["n"] > 0:
            remaining["n"] -= 1
            return ToolOutcome(name="browser_snapshot", text=BLANK_PAGE, duration_ms=1)
        return ToolOutcome(name="browser_snapshot", text=mcp.page, duration_ms=1)

    for tool in mcp._tools:  # noqa: SLF001 - rewiring the fake
        if tool.name == "browser_snapshot":
            tool.handler = handler


async def test_an_element_that_renders_late_is_still_found():
    """Replay must wait for the page the way Playwright's own actions do."""
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(id="s1", action="fill", value="{{secret.userid}}",
                 locators=[Locator(strategy="role", role="textbox", name="Username")]),
        ],
        secrets=[SecretSpec(name="userid")],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    _slow_render(mcp, blank_snapshots=2)  # ~1s of blank page at POLL_INTERVAL=0.5

    runner = executor(use_case, mcp, secrets={"userid": "someone"}, step_timeout=5.0)
    result = await runner.run_row({})

    assert result.ok, result.error
    typed = mcp.calls_to("browser_type")
    assert typed and typed[0]["text"] == "someone"


async def test_a_truly_missing_element_still_fails_after_the_deadline():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Never Appears")])],
    )
    runner = executor(use_case, ScriptedMCP([SIGNED_OUT]), step_timeout=1.2)
    result = await runner.run_row({})

    assert not result.ok
    assert "no element matched" in (result.error or "")


async def test_weak_rungs_wait_for_the_role_grace_not_the_full_timeout(monkeypatch):
    """A renamed element with a css fallback costs seconds per row, not the
    whole step timeout -- and the fallback is still taken."""
    import replay as replay_module

    monkeypatch.setattr(replay_module, "ROLE_GRACE_SECONDS", 0.3)
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[
            Step(id="s1", action="click",
                 locators=[role("Renamed Button"),
                           Locator(strategy="css", selector="#submit")]),
        ],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    runner = executor(use_case, mcp, step_timeout=30.0)

    import time as time_module
    started = time_module.monotonic()
    result = await runner.run_row({})
    elapsed = time_module.monotonic() - started

    assert result.ok, result.error
    assert mcp.calls_to("browser_click")[0]["target"] == "#submit"
    assert elapsed < 5, f"fallback took {elapsed:.1f}s; the grace should be ~0.3s"
    # Falling through to a weaker rung is drift, and it is still recorded.
    assert runner.locator_drift.get("s1")


async def test_an_empty_snapshot_does_not_leave_stale_refs_matchable():
    """A page that currently exposes nothing must not 'match' the old page.

    A stale ref handed to the server is rejected as an error the retry loop
    cannot see past -- worse than honestly reporting no match.
    """
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )
    mcp = ScriptedMCP([SIGNED_OUT])
    runner = executor(use_case, mcp, step_timeout=1.2)

    # Seed the cache with a real page, then make every snapshot come back blank.
    await runner._refresh_snapshot()  # noqa: SLF001
    assert runner._last_snapshot is not None  # noqa: SLF001
    _slow_render(mcp, blank_snapshots=10_000)

    result = await runner.run_row({})

    assert not result.ok
    assert "no element matched" in (result.error or "")
    # The click was never attempted with a ref from the vanished page.
    assert not mcp.calls_to("browser_click")
