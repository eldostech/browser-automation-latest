"""The agent's tool layer: what it may call, and what has to be true first.

No model and no browser anywhere in this file. That is the point of building
the tool layer before the graph -- the part that decides what is allowed to
happen is exactly the part that should be testable without either.

The centrepiece is the ref discipline. The design document claimed that
"the model cannot invent a locator" came free with Playwright MCP, because the
model picks an opaque handle out of a snapshot. Probing a real server showed
that was wrong: ``target`` also accepts a raw selector, and ``#o`` clicks the
element. These tests pin the property back on by enforcement.
"""

from __future__ import annotations

import pytest

from agent import (
    AgentToolSession,
    GuardContext,
    LocalPlaywrightMCP,
    ToolResult,
    ToolSpec,
    guard,
    offered,
)
from agent.providers import REF_IN_SNAPSHOT

pytestmark = pytest.mark.anyio


# --- a browser that is not a browser ---------------------------------------


SNAPSHOT = """### Page
- Page URL: https://vendor.test/users
### Snapshot
```yaml
- generic [active] [ref=e1]:
  - heading "Users" [level=1] [ref=e2]
  - button "+ Invite User" [ref=e3]
  - textbox "Email" [ref=e4]
```
"""

#: What the real server replies with after a click on a ref: the Playwright
#: code it ran. That line is a durable locator handed over for free, and it is
#: the reason forcing refs matters beyond safety.
AFTER_CLICK = """### Ran Playwright code
```js
await page.getByRole('button', { name: '+ Invite User' }).click();
```
### Page
- Page URL: https://vendor.test/users
### Snapshot
```yaml
- generic [active] [ref=e7]:
  - dialog "Invite" [ref=e8]:
    - textbox "Email" [ref=e9]
    - button "Invite" [ref=e10]
```
"""

#: The shape that broke a real session: an interactive control inside an
#: iframe, which Playwright MCP addresses with a frame-scoped ref -- one
#: `f<N>` segment per level of nesting before the element's own `e<N>`.
IFRAME_SNAPSHOT = """### Page
- Page URL: https://vendor.test/widget
### Snapshot
```yaml
- generic [ref=e1]:
  - iframe [ref=e2]:
    - textbox "Answer" [ref=f1e3]
```
"""

TOOLS = [
    ToolSpec("browser_navigate", "", {"required": ["url"]}),
    # Advertised by the real server, so the fake advertises it too. A fake
    # narrower than the thing it stands in for silently narrows every test
    # written against it: this one was missing, so `browser_navigate_back` was
    # refused as an unknown tool and never reached distillation.
    ToolSpec("browser_navigate_back", "", {}),
    ToolSpec("browser_snapshot", "", {}),
    ToolSpec("browser_click", "", {"required": ["target"]}),
    ToolSpec("browser_type", "", {"required": ["target", "text"]}),
    ToolSpec("browser_select_option", "", {"required": ["target", "values"]}),
    ToolSpec("browser_fill_form", "", {"required": ["fields"]}),
    ToolSpec("browser_evaluate", "", {"required": ["function"]}),
    ToolSpec("browser_run_code_unsafe", "", {}),
]


class FakeMCP:
    """Answers like Playwright MCP does, without Node or a browser."""

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.replies = replies or {}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def list_tools(self):
        return list(TOOLS)

    async def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        reply = self.replies.get(name, SNAPSHOT)
        # A test simulating a genuine dispatch failure (not merely a
        # different page) passes a `ToolResult` directly rather than a page
        # of text there is no honest way to mark as an error.
        if isinstance(reply, ToolResult):
            return reply
        return ToolResult(
            text=reply, refs=tuple(dict.fromkeys(REF_IN_SNAPSHOT.findall(reply)))
        )

    # -- as a provider ---------------------------------------------------
    async def open(self):
        return self

    async def close(self):
        self.closed = True


async def session(**kwargs) -> AgentToolSession:
    """A session that can act, because most of these tests are about what
    happens when it does. The read-only default is exercised on its own."""
    fake = FakeMCP(kwargs.pop("replies", None))
    kwargs.setdefault("may_write", True)
    return AgentToolSession(
        fake, allowed_domains=kwargs.pop("allowed_domains", ("vendor.test",)), **kwargs
    )


# --- what is offered -------------------------------------------------------


def test_a_refused_tool_is_removed_rather_than_forbidden():
    """A prompt saying "do not use browser_evaluate" is a request.

    A tool absent from the list cannot be called by a model that decides the
    rules do not apply to it.
    """
    names = [spec.name for spec in offered(TOOLS)]

    assert "browser_evaluate" not in names
    assert "browser_run_code_unsafe" not in names
    assert "browser_click" in names


def test_asking_for_a_refused_tool_anyway_says_why():
    """It cannot be called, and the reason is worth more than "unknown tool"."""
    verdict = guard("browser_evaluate", {"function": "() => 1"}, GuardContext())

    assert not verdict.allowed
    assert "allow_scripts" in verdict.reason


# --- the ref discipline ----------------------------------------------------


def context(**kwargs) -> GuardContext:
    return GuardContext(
        allowed_domains=kwargs.pop("allowed_domains", ("vendor.test",)),
        known_refs=frozenset(kwargs.pop("known_refs", {"e3", "e4"})),
        may_write=kwargs.pop("may_write", True),
        available=frozenset(kwargs.pop("available", {s.name for s in TOOLS})),
        annotations=kwargs.pop("annotations", {}),
    )


def test_a_selector_the_model_composed_is_refused():
    """The finding that made this file necessary.

    `browser_click`'s target is documented as "Exact target element reference
    from the page snapshot, or a unique element selector", and passing '#o'
    really does click the element. The property the design claimed came free
    has to be enforced.
    """
    verdict = guard("browser_click", {"target": "#invite-button"}, context())

    assert not verdict.allowed
    assert "not an element reference" in verdict.reason
    assert "snapshot" in verdict.reason, "it has to say what to do instead"


@pytest.mark.parametrize(
    "target", ["button >> nth=0", "text=Invite", "//button[1]", "e", "3", ""]
)
def test_nothing_that_is_not_a_ref_gets_through(target):
    assert not guard("browser_click", {"target": target}, context()).allowed


def test_a_ref_from_a_stale_snapshot_is_refused_differently():
    """A different failure needing a different fix: take a fresh snapshot."""
    verdict = guard("browser_click", {"target": "e99"}, context())

    assert not verdict.allowed
    assert "as it now stands" in verdict.reason


def test_a_ref_the_page_reported_is_allowed():
    assert guard("browser_click", {"target": "e3"}, context()).allowed


def test_a_frame_scoped_ref_is_a_ref_not_a_composed_selector():
    """The bug this session was built to fix: `REF_FORMAT` used to be
    `^e\\d+$`, so an element inside an iframe -- `f1e3`, one `f<N>` segment
    per level of frame nesting before the element's own `e<N>` -- was refused
    outright as "not an element reference", the same refusal a composed CSS
    selector gets. Not "stale", not "ambiguous": unreachable, regardless of
    whether it was ever valid. Found for real, on a page whose only
    interactive control happened to be inside one."""
    verdict = guard("browser_click", {"target": "f1e3"}, context(known_refs={"f1e3"}))
    assert verdict.allowed


def test_a_frame_scoped_ref_the_page_never_reported_is_still_refused():
    """The fix widens what counts as a ref; it must not widen what counts as
    a known one -- a made-up frame-scoped string is exactly as refusable as a
    made-up bare one."""
    verdict = guard("browser_click", {"target": "f1e3"}, context(known_refs={"e3"}))
    assert not verdict.allowed
    assert "as it now stands" in verdict.reason


async def test_a_frame_scoped_ref_reaches_the_browser_end_to_end():
    """Not just accepted by the guard in isolation -- reachable through the
    whole path a real session takes: a snapshot naming an iframe's control,
    `known_refs` picking it up, and a click on it actually dispatching."""
    async with await session(replies={"browser_snapshot": IFRAME_SNAPSHOT}) as tools:
        await tools.call("browser_snapshot")
        assert "f1e3" in tools.known_refs

        result = await tools.call("browser_click", {"target": "f1e3"})
        assert not result.is_error


async def test_refs_are_replaced_by_what_the_page_last_reported():
    """Accumulating them would let exactly the stale target through.

    A ref is only valid for the snapshot it came from. After a click opens a
    dialog, the refs that named the page behind it are gone.
    """
    async with await session(replies={"browser_click": AFTER_CLICK}) as tools:
        await tools.call("browser_snapshot")
        assert tools.known_refs == {"e1", "e2", "e3", "e4"}

        await tools.call("browser_click", {"target": "e3"})
        assert tools.known_refs == {"e7", "e8", "e9", "e10"}
        assert "e3" not in tools.known_refs, "the page behind the dialog is gone"


# --- retrying a target that already failed ----------------------------------


async def test_a_target_that_already_failed_is_refused_without_retrying_it():
    """Found for real: a session retried the identical invalid ref across four
    calls and two different tools, twice after being told in as many words
    that the ref did not exist, and burned its whole budget on the loop
    before it ever reached the task. `guard()` accepting a call is not the
    same as that call succeeding, and nothing before this stopped the second
    identical mistake -- only the third or fourth. This is the backstop."""
    async with await session(
        replies={"browser_click": ToolResult.failed("element is not visible")},
    ) as tools:
        await tools.call("browser_snapshot")

        first = await tools.call("browser_click", {"target": "e3"})
        assert first.is_error
        assert "element is not visible" in first.text

        second = await tools.call("browser_click", {"target": "e3"})
        assert second.is_error
        assert "already failed" in second.text
        assert "element is not visible" in second.text, "the original reason travels with it"
        assert len([c for c in tools.provider.calls if c[0] == "browser_click"]) == 1, (
            "the second attempt must not reach the browser at all"
        )


async def test_a_different_target_after_a_failure_is_not_affected():
    """The guard is keyed on the exact target string, not "this tool failed
    recently" -- a different, unrelated ref must still reach the browser."""
    async with await session(
        replies={"browser_click": ToolResult.failed("element is not visible")},
    ) as tools:
        await tools.call("browser_snapshot")
        await tools.call("browser_click", {"target": "e3"})

        other = await tools.call("browser_click", {"target": "e4"})
        assert "already failed" not in other.text
        assert len(tools.provider.calls) == 3, "e4 reached the browser, unlike a repeat of e3 would"


async def test_the_failed_target_memory_is_bounded():
    """Ref strings are not unique forever -- a much later snapshot generation
    reusing one must be judged on its own result, not refused for a mistake
    an earlier, unrelated snapshot made many calls ago."""
    from agent.session import FAILED_TARGET_MEMORY

    async with await session() as tools:
        for i in range(FAILED_TARGET_MEMORY + 2):
            tools._failed_targets[f"e{i}"] = "stale reason"

        assert len(tools._failed_targets) == FAILED_TARGET_MEMORY + 2
        tools._remember_failed_target("e999", "newest")
        assert len(tools._failed_targets) == FAILED_TARGET_MEMORY
        assert "e0" not in tools._failed_targets, "the oldest entries are the ones dropped"
        assert "e999" in tools._failed_targets


# --- the hard gate ---------------------------------------------------------


def test_leaving_the_allowlist_is_refused():
    verdict = guard(
        "browser_navigate", {"url": "https://elsewhere.test/"}, context()
    )
    assert not verdict.allowed


def test_an_empty_allowlist_blocks_everything():
    """Deny by default. A session with no domains configured is not a session
    with every domain configured."""
    verdict = guard(
        "browser_navigate", {"url": "https://vendor.test/"}, context(allowed_domains=())
    )
    assert not verdict.allowed


def test_the_allowlist_is_checked_on_every_tool_not_just_navigation():
    """A tool we have never seen could still take a URL."""
    verdict = guard(
        "browser_type", {"target": "e4", "text": "https://elsewhere.test/"}, context()
    )
    assert not verdict.allowed


# --- the write gate --------------------------------------------------------


def test_a_read_only_session_cannot_change_the_page():
    """An agent sent to find out how a form works must not submit it."""
    read_only = context(may_write=False)

    assert not guard("browser_click", {"target": "e3"}, read_only).allowed
    assert not guard("browser_type", {"target": "e4", "text": "x"}, read_only).allowed
    assert guard("browser_snapshot", {}, read_only).allowed, "reading is still fine"


def test_an_irreversible_action_is_flagged_without_being_refused():
    """Classified in code from the arguments, not from the model's opinion of
    its own next action. The rendezvous itself arrives with the graph."""
    verdict = guard(
        "browser_click", {"target": "e3", "element": "Delete account"}, context()
    )

    assert verdict.allowed


def test_a_form_submit_no_longer_needs_approval():
    """Deliberately different from `payment`/`destructive` just below: this
    deployment's operator asked for submit specifically to stop pausing on a
    person, after being told what that means for every session, not only
    this one -- see `catalog.IRREVERSIBLE`'s own comment. It is still
    detected and still allowed either way; only the pause is gone."""
    verdict = guard(
        "browser_click", {"target": "e3", "element": "Submit form"}, context()
    )

    assert verdict.allowed
    assert not verdict.needs_approval
    assert "form_submit" in verdict.category


def test_payment_and_destructive_still_need_approval():
    for element in ("Pay now", "Delete account"):
        verdict = guard(
            "browser_click", {"target": "e3", "element": element}, context()
        )
        assert verdict.allowed
        assert verdict.needs_approval, element
    assert verdict.needs_approval
    assert verdict.category


# --- a tool from a registered server, classified with no name to go on -----


def test_an_unannotated_tool_is_blocked_without_write_access():
    """No annotation means "unknown", and unknown is the conservative answer
    -- the same "deny-by-default" rule the allowlist already applies."""
    verdict = guard(
        "crm.delete_account", {},
        context(may_write=False, available={"crm.delete_account"}),
    )
    assert not verdict.allowed


def test_an_unannotated_tool_needs_approval_even_with_write_access():
    """Write access is not the same question as "should a person see this
    first" -- a tool nobody here has read the source of gets both gates."""
    verdict = guard(
        "crm.delete_account", {},
        context(may_write=True, available={"crm.delete_account"}),
    )
    assert verdict.allowed
    assert verdict.needs_approval


def test_a_declared_read_only_tool_skips_both_gates():
    """The one way a registered server's tool is trusted: it says so itself,
    via the MCP protocol's own annotations -- not by a name this file has an
    opinion about, since nobody here has read that server's source."""
    ctx = context(
        may_write=False,
        available={"crm.lookup_account"},
        annotations={"crm.lookup_account": {"readOnlyHint": True}},
    )
    verdict = guard("crm.lookup_account", {}, ctx)
    assert verdict.allowed
    assert not verdict.needs_approval


def test_a_declared_destructive_tool_still_needs_write_access_and_approval():
    ctx = context(
        may_write=True,
        available={"crm.delete_account"},
        annotations={"crm.delete_account": {"destructiveHint": True}},
    )
    verdict = guard("crm.delete_account", {}, ctx)
    assert verdict.allowed
    assert verdict.needs_approval


def test_a_mark_tool_is_unaffected_by_the_unannotated_classification():
    """Mark tools are not in DISTILS_TO either, and must not fall into the
    "unknown server tool" path -- they would suddenly need write access and
    approval just to record a row boundary."""
    verdict = guard(
        "begin_row", {"key": "A-1"},
        context(may_write=False, available={s.name for s in TOOLS} | {"begin_row"}),
    )
    assert verdict.allowed
    assert not verdict.needs_approval


# --- dispatch, recording, redaction ---------------------------------------


async def test_a_refusal_comes_back_as_a_tool_error_rather_than_an_exception():
    """An agent told why it may not do something can choose differently.

    An agent that crashes cannot, and one told nothing retries the same call
    until its budget is gone.
    """
    async with await session() as tools:
        result = await tools.call("browser_navigate", {"url": "https://elsewhere.test/"})

    assert result.is_error
    assert "elsewhere.test" in result.text
    assert tools.provider.calls == [], "it must not have reached the browser"


async def test_a_refusal_is_audited_exactly_like_a_call_that_ran():
    """"It tried to leave the allowlist and was stopped" is the entry somebody
    will actually want. An audit of only what succeeded describes a system
    nobody has to trust."""
    recorded = []

    async with await session(recorder=lambda r: _collect(recorded, r)) as tools:
        await tools.call("browser_snapshot")
        await tools.call("browser_navigate", {"url": "https://elsewhere.test/"})

    assert [r.name for r in recorded] == ["browser_snapshot", "browser_navigate"]
    assert [r.refused for r in recorded] == [False, True]
    assert [r.seq for r in recorded] == [1, 2]


async def test_a_navigate_call_is_not_reported_as_ambiguous():
    """`browser_navigate` has no `target` at all -- it goes to a URL, not an
    element -- but describing one was called anyway, on whatever an absent
    `target` argument stringified to (an empty ref), which found nothing to
    describe and reported back as "matched 0 elements". Distillation then
    warned that a navigate step needed re-recording for being "ambiguous",
    which it was never at risk of being. Only a call that actually names a
    `target` should be described at all."""
    recorded = []

    async with await session(recorder=lambda r: _collect(recorded, r)) as tools:
        await tools.call("browser_navigate", {"url": "https://vendor.test/"})

    navigate = next(r for r in recorded if r.name == "browser_navigate")
    assert navigate.match_count == 1
    assert navigate.locators == []


async def _collect(bucket, record):
    bucket.append(record)


async def test_a_typed_secret_never_reaches_the_trajectory():
    """The redactor is registered before anything is emitted -- an existing
    invariant, and one an agent makes easier to break because it types
    credentials into pages for a living."""
    from redaction import Redactor

    async with await session(redactor=Redactor(["hunter2-not-a-real-password"])) as tools:
        await tools.call("browser_snapshot")
        await tools.call(
            "browser_type", {"target": "e4", "text": "hunter2-not-a-real-password"}
        )

    typed = tools.calls[-1]
    assert "hunter2" not in typed.arguments["text"]
    assert typed.arguments["text"] != "hunter2-not-a-real-password"


async def test_every_call_records_what_it_would_distil_into():
    """The trajectory is the input to distillation, so the mapping is carried
    at the moment the call is made rather than inferred from a name later."""
    async with await session() as tools:
        await tools.call("browser_snapshot")
        await tools.call("browser_click", {"target": "e3"})

    assert [(c.name, c.action) for c in tools.calls] == [
        ("browser_snapshot", ""),
        ("browser_click", "click"),
    ]


async def test_the_browser_is_closed_even_when_the_body_raises():
    """A Node subprocess and a Chromium behind it are not things to leak."""
    fake = FakeMCP()
    tools = AgentToolSession(fake, allowed_domains=("vendor.test",))

    with pytest.raises(RuntimeError, match="deliberate"):
        async with tools:
            raise RuntimeError("deliberate")

    assert fake.closed


# --- the deployment role ---------------------------------------------------


def test_the_local_provider_pins_the_server_and_isolates_the_profile():
    """A batch that signed in as one tenant must not leave a cookie behind for
    the next, and `@latest` would let a release change the tool contract inside
    somebody's session rather than in CI."""
    argv = LocalPlaywrightMCP(version="0.0.80").argv()

    assert argv[0] == "@playwright/mcp@0.0.80"
    assert "--isolated" in argv
    assert "--headless" in argv


def test_attaching_to_a_managed_browser_is_a_different_argv_not_a_rewrite():
    """The AgentCore seam. Playwright MCP can attach over CDP rather than
    launch, which is what keeps that port small."""
    argv = LocalPlaywrightMCP(cdp_endpoint="ws://managed/abc").argv()

    assert "--cdp-endpoint" in argv
    assert "ws://managed/abc" in argv
    assert "--headless" not in argv, "the browser is somebody else's to configure"


def test_the_agent_package_imports_no_optional_dependency_at_module_scope():
    """The compatibility contract, made mechanical.

    With the extras uninstalled the platform must be exactly the platform it
    was before this package existed. That is only true if importing it does not
    reach for `mcp` or `langgraph` -- in the spirit of the two tests asserting
    `engine.py` does not import `llm`.
    """
    import inspect
    import pkgutil

    import agent
    import agent.guardrails
    import agent.providers
    import agent.session
    import agent.tools

    # Every submodule of every package below, not just the four top-level
    # names -- the compatibility contract broke once already from a file one
    # level down (`tool_adapter.py`) that nothing here was checking.
    modules = [agent, agent.session]
    for package in (agent.guardrails, agent.providers, agent.tools):
        modules.append(package)
        modules.extend(
            __import__(f"{package.__name__}.{info.name}", fromlist=["_"])
            for info in pkgutil.iter_modules(package.__path__)
        )

    for module in modules:
        source = inspect.getsource(module)
        top_level = [
            line
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "#" not in line.split("import")[0]
        ]
        joined = "\n".join(top_level)
        assert "import mcp" not in joined, module.__name__
        assert "from mcp" not in joined, module.__name__
        assert "langgraph" not in joined, module.__name__
        assert "langchain" not in joined, module.__name__


def test_the_config_endpoint_tells_a_deployment_apart_from_a_broken_one():
    """An operator who enabled the agent must not be told it is disabled.

    Two independent answers -- "this deployment says no" and "this deployment
    says yes but cannot" -- need two different fixes, so they get two different
    messages.
    """
    from config import Settings
    from routers.health import _agent_status

    off = _agent_status(Settings(_env_file=None, agent_enabled=False))
    assert off["enabled"] is False
    assert "AGENT_ENABLED" in off["reason"]

    on = _agent_status(Settings(_env_file=None, agent_enabled=True))
    # Whether it *can* run depends on the machine; what must be true is that it
    # answers about the machine rather than about the setting.
    assert "AGENT_ENABLED" not in (on["reason"] or "")


# --- credentials: the route from a bound slot to a typed character ---------
#
# Found from real sessions against a real app. An agent told "log in with the
# credentials provided" and given no real value to type fabricated "admin" and
# "password" -- twice, in two separate sessions -- because nothing existed to
# turn a bound credential into a character on the page. These tests pin the
# fix: the model types a placeholder, and only the dispatch that actually
# reaches the browser ever sees the real value.


async def test_a_placeholder_becomes_the_real_value_only_in_what_reaches_the_browser():
    async with await session(secret_values={"email": "ada@vendor.test"}) as tools:
        await tools.call("browser_type", {"target": "e4", "text": "{{secret.email}}"})

    dispatched_name, dispatched_args = tools.provider.calls[-1]
    assert dispatched_args["text"] == "ada@vendor.test"

    # What is kept -- the trajectory, the audit trail, what a reviewer sees --
    # is the placeholder. That is also what a replay of this step should carry,
    # so this is not a compromise made for secrecy; it is the correct value to
    # store either way.
    recorded = tools.calls[-1]
    assert recorded.arguments["text"] == "{{secret.email}}"
    assert "ada@vendor.test" not in str(recorded.arguments)
    assert "ada@vendor.test" not in recorded.detail


async def test_a_literal_guess_is_left_alone_rather_than_rejected():
    """Substitution only ever narrows what a placeholder means. A model that
    ignores the instruction and types a guess anyway gets exactly what it
    typed -- wrong, but not silently rewritten into something else wrong."""
    async with await session(secret_values={"email": "ada@vendor.test"}) as tools:
        await tools.call("browser_type", {"target": "e4", "text": "admin"})

    assert tools.provider.calls[-1][1]["text"] == "admin"


async def test_an_unknown_slot_is_refused_before_it_reaches_the_browser():
    """Typing the literal, unresolved placeholder into a real page is worse
    than refusing: the page silently rejects garbage and nothing explains why
    the rest of the task became impossible. This is caught first."""
    async with await session(secret_values={"email": "ada@vendor.test"}) as tools:
        before = len(tools.provider.calls)
        result = await tools.call("browser_type", {"target": "e4", "text": "{{secret.password}}"})

    assert result.is_error
    assert "email" in result.text, "it names what is actually bound, to fix the call"
    assert len(tools.provider.calls) == before, "nothing reached the browser"


async def test_select_option_values_are_substituted_too():
    async with await session(secret_values={"role": "administrator"}) as tools:
        await tools.call(
            "browser_select_option", {"target": "e4", "values": ["{{secret.role}}"]}
        )

    assert tools.provider.calls[-1][1]["values"] == ["administrator"]


async def test_fill_form_substitutes_each_fields_own_value():
    """`browser_fill_form` batches several fields in one call, so each one's
    `value` has to be handled on its own rather than as a single string."""
    async with await session(secret_values={"email": "ada@vendor.test", "password": "s3cret"}) as tools:
        await tools.call(
            "browser_fill_form",
            {
                "fields": [
                    {"name": "Email", "target": "e4", "value": "{{secret.email}}"},
                    {"name": "Password", "target": "e9", "value": "{{secret.password}}"},
                ]
            },
        )

    sent = tools.provider.calls[-1][1]["fields"]
    assert [f["value"] for f in sent] == ["ada@vendor.test", "s3cret"]


async def test_a_session_with_no_bound_credentials_substitutes_nothing():
    """The common case, and it must cost nothing: most calls carry no
    placeholder at all, and a session recording a workflow with no sign-in
    should not pay for a lookup that can never match."""
    async with await session() as tools:  # no secret_values
        await tools.call("browser_type", {"target": "e4", "text": "ordinary text"})

    assert tools.provider.calls[-1][1]["text"] == "ordinary text"


# --- tools from a registered server: prefixing, dispatch, distillation ----
#
# Phase 1's whole point: an agent session used to have exactly one tool
# source. These pin the three things that had to be true for a second one to
# be safe -- its tools cannot collide with the browser's, a call to one
# cannot be mistaken for a browser action, and it never becomes a step.


class FakeExtraServer:
    """A server that is not the browser: tools and answers, no refs, no page."""

    def __init__(self, specs: list[ToolSpec], replies: dict[str, ToolResult] | None = None) -> None:
        self._specs = specs
        self.replies = replies or {}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._specs)

    async def call(self, name: str, arguments: dict) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        return self.replies.get(name, ToolResult(text="ok"))

    async def open(self):
        return self

    async def close(self):
        self.closed = True


async def test_an_extra_servers_tools_are_shown_prefixed_by_its_name():
    """Unprefixed, "lookup_account" could collide with a browser tool added
    later. Prefixed, it never can."""
    extra = FakeExtraServer([ToolSpec("lookup_account", "Look up a CRM account")])
    async with await session(extra={"crm": extra}) as tools:
        names = [spec.name for spec in tools.tools]

    assert "crm.lookup_account" in names
    assert "lookup_account" not in names


async def test_calling_an_extra_tool_dispatches_to_its_own_server():
    extra = FakeExtraServer([ToolSpec("lookup_account", "")])
    async with await session(extra={"crm": extra}) as tools:
        result = await tools.call("crm.lookup_account", {"id": "A-1"})

    assert not result.is_error
    assert extra.calls == [("lookup_account", {"id": "A-1"})]
    assert tools.provider.calls == [], "it must not have reached the browser"


async def test_an_extra_tool_call_never_carries_a_locator_or_distils_to_a_step():
    """The invariant the whole design rests on: nothing from a registered
    server can become a replay step."""
    extra = FakeExtraServer([ToolSpec("lookup_account", "")])
    async with await session(extra={"crm": extra}) as tools:
        await tools.call("crm.lookup_account", {"id": "A-1"})

    recorded = tools.calls[-1]
    assert recorded.action == ""
    assert recorded.locators == []


async def test_an_extra_tool_does_not_disturb_the_browsers_known_refs():
    extra = FakeExtraServer([ToolSpec("lookup_account", "")])
    async with await session(extra={"crm": extra}) as tools:
        await tools.call("browser_snapshot")
        before = tools.known_refs
        await tools.call("crm.lookup_account", {"id": "A-1"})

        assert tools.known_refs == before


async def test_an_unadvertised_extra_tool_name_is_refused_as_unknown():
    extra = FakeExtraServer([ToolSpec("lookup_account", "")])
    async with await session(extra={"crm": extra}) as tools:
        result = await tools.call("crm.delete_account", {})

    assert result.is_error
    assert extra.calls == []


async def test_a_secret_placeholder_is_substituted_for_an_extra_tool_too():
    extra = FakeExtraServer([ToolSpec("lookup_account", "")])
    async with await session(extra={"crm": extra}, secret_values={"key": "sk-real"}) as tools:
        await tools.call("crm.lookup_account", {"text": "{{secret.key}}"})

    assert extra.calls[-1][1]["text"] == "sk-real"


async def test_closing_the_session_closes_every_extra_provider_too():
    extra = FakeExtraServer([ToolSpec("lookup_account", "")])
    async with await session(extra={"crm": extra}):
        pass

    assert extra.closed


# --- looping on a call that is allowed but is not working ------------------


async def test_the_same_click_three_times_is_refused_as_a_loop():
    """The stale-ref rule covers a *dead* reference retried. This is the other
    loop, and until now only the budget stopped it: a live reference clicked
    over and over because the click is not having the effect expected.

    The budget is a bad backstop for this. It stops the agent eventually, after
    the session has spent everything it had on one button.
    """
    async with await session() as tools:
        await tools.call("browser_snapshot", {})
        first = await tools.call("browser_click", {"target": "e3"})
        second = await tools.call("browser_click", {"target": "e3"})
        third = await tools.call("browser_click", {"target": "e3"})

    assert not first.is_error and not second.is_error
    assert third.is_error
    assert "already been called" in third.text
    assert "browser_snapshot" in third.text, "says what to do instead"


async def test_acting_on_a_different_element_is_never_a_repeat():
    """Counted per (tool, arguments). Clicking two things is doing two things,
    however many times the page has been clicked overall."""
    async with await session() as tools:
        await tools.call("browser_snapshot", {})
        for _ in range(3):
            assert not (await tools.call("browser_click", {"target": "e3"})).is_error or True
        result = await tools.call("browser_type", {"target": "e4", "text": "a@b.test"})

    assert not result.is_error


async def test_taking_the_same_snapshot_repeatedly_is_not_a_loop():
    """Re-reading the page after every change is exactly what the ref
    discipline asks for. Refusing the third would break the loop this rule
    exists to protect."""
    async with await session() as tools:
        results = [await tools.call("browser_snapshot", {}) for _ in range(5)]

    assert not any(result.is_error for result in results)


# --- saying what you expect, as a schema rather than a request -------------


async def test_every_offered_tool_requires_an_observation():
    """The prompts already ask for this in prose, and a prompt cannot make it
    happen: a model under pressure drops the sentence and calls the tool, and
    nothing notices. As a required argument it cannot be dropped."""
    from agent.session import OBSERVATION

    async with await session() as tools:
        specs = tools.tools

    assert specs, "the fake advertises tools"
    for spec in specs:
        assert OBSERVATION in spec.input_schema["properties"], spec.name
        assert OBSERVATION in spec.input_schema["required"], spec.name


async def test_the_observation_never_reaches_the_browser():
    """It is a note for the trail, not an argument. A server told about it
    would refuse the call for a parameter it has never heard of."""
    fake = FakeMCP()
    tools = AgentToolSession(fake, allowed_domains=("vendor.test",), may_write=True)
    async with tools:
        await tools.call("browser_snapshot", {"observation": "reading the page"})
        await tools.call(
            "browser_click",
            {"target": "e3", "observation": "the Invite dialog should open"},
        )

    for _, arguments in fake.calls:
        assert "observation" not in arguments


async def test_the_observation_is_kept_beside_the_call_it_describes():
    """In the trail rather than in loose prose above it, which is what makes a
    trail read afterwards say what was intended as well as what happened."""
    async with await session() as tools:
        await tools.call("browser_snapshot", {})
        await tools.call(
            "browser_click",
            {"target": "e3", "observation": "the Invite dialog should open"},
        )

    click = next(record for record in tools.calls if record.name == "browser_click")
    assert click.observation == "the Invite dialog should open"


async def test_a_call_with_no_observation_is_still_dispatched():
    """The schema asks; this does not add a second refusal on top of it.

    A model that omits a required argument has already been told so by its own
    provider, and turning that into a browser-level refusal would spend a turn
    on bookkeeping instead of on the page.
    """
    async with await session() as tools:
        await tools.call("browser_snapshot", {})
        result = await tools.call("browser_click", {"target": "e3"})

    assert not result.is_error


# --- told while the page is still in front of it ---------------------------
#
# The asymmetry: an agent acts on `ref=e12`, which always names exactly one
# element, and a replay acts on a description. So the moment a
# badly-describable element gets clicked is both the cheapest moment to act on
# that and the last moment anybody has the page. Left alone, the click was
# recorded silently and surfaced as a draft warning nobody could act on.

WRAPPERS = """### Page
- Page URL: https://vendor.test/projects
### Snapshot
```yaml
- list [ref=e1]:
  - generic [ref=e2]
  - generic [ref=e3]
  - generic [ref=e4]
```
"""


async def test_clicking_an_undescribable_wrapper_is_refused_the_first_time():
    async with await session(replies={"browser_snapshot": WRAPPERS}) as tools:
        await tools.call("browser_snapshot", {})
        result = await tools.call("browser_click", {"target": "e3"})

    assert result.is_error
    assert "cannot be *recorded*" in result.text
    assert "role and name" in result.text, "says why a replay cannot use it"
    assert "button, a link, a heading" in result.text, "says what to do instead"


async def test_repeating_it_goes_through():
    """Not a hard refusal, deliberately. The click itself is fine -- it is the
    recording of it that is not -- and blocking it outright would stop the
    agent completing a task it can plainly do. A repeat is the agent saying
    there is nothing better, and on this page there genuinely is not."""
    async with await session(replies={"browser_snapshot": WRAPPERS}) as tools:
        await tools.call("browser_snapshot", {})
        first = await tools.call("browser_click", {"target": "e3"})
        second = await tools.call("browser_click", {"target": "e3"})

    assert first.is_error
    assert not second.is_error


async def test_a_nameable_element_is_never_refused():
    async with await session() as tools:
        await tools.call("browser_snapshot", {})
        result = await tools.call("browser_click", {"target": "e3"})

    assert not result.is_error


async def test_looking_at_the_page_is_never_refused():
    """Perception is how the agent finds something better. Refusing a snapshot
    would break the very loop this asks for."""
    async with await session(replies={"browser_snapshot": WRAPPERS}) as tools:
        for _ in range(3):
            assert not (await tools.call("browser_snapshot", {})).is_error


async def test_the_refused_attempt_never_reaches_the_browser():
    fake = FakeMCP({"browser_snapshot": WRAPPERS})
    tools = AgentToolSession(fake, allowed_domains=("vendor.test",), may_write=True)
    async with tools:
        await tools.call("browser_snapshot", {})
        await tools.call("browser_click", {"target": "e3"})

    assert [name for name, _ in fake.calls] == ["browser_snapshot"]


# --- the page a step was recorded against ---------------------------------


async def test_a_call_keeps_the_page_it_was_made_against():
    """The evidence a repair months later has no other way to get: the page as
    it *was*. Given only the page as it is now, choosing a replacement is
    guessing which of forty controls somebody meant."""
    async with await session() as tools:
        await tools.call("browser_snapshot", {})
        await tools.call("browser_click", {"target": "e3"})

    click = next(record for record in tools.calls if record.name == "browser_click")
    assert "+ Invite User" in click.page, "the tree it acted against"


async def test_the_recorded_page_is_capped():
    """Review material, not the recording. An uncapped blob per step would put
    a megabyte of markup into every definition."""
    from agent.session import RECORDED_PAGE_CHARS

    huge = "### Page\n### Snapshot\n```yaml\n" + "\n".join(
        f'- button "Button {n}" [ref=e{n}]' for n in range(4000)
    ) + "\n```\n"

    async with await session(replies={"browser_snapshot": huge}) as tools:
        await tools.call("browser_snapshot", {})
        await tools.call("browser_click", {"target": "e3"})

    click = next(record for record in tools.calls if record.name == "browser_click")
    assert 0 < len(click.page) <= RECORDED_PAGE_CHARS


# --- what the element said, for a replay to check against -----------------


async def test_a_call_records_what_the_element_said():
    """So a replay can tell that a rung which says only *where* to look has
    landed on the same control. The recorder already knew this and threw it
    away -- see `Step.expect_text`."""
    async with await session() as tools:
        await tools.call("browser_snapshot", {})
        await tools.call("browser_click", {"target": "e3"})

    click = next(record for record in tools.calls if record.name == "browser_click")
    assert click.expect_text, "the element's own words"
    assert click.expect_text in click.page, "read off the page it acted against"


async def test_an_unnamed_wrapper_records_the_text_of_the_control_inside_it():
    """The rung is borrowed from a named descendant, so the expectation has to
    be that descendant's words -- the wrapper's own name is empty, and checking
    an empty expectation checks nothing."""
    page = """### Page
- Page URL: https://vendor.test/projects
### Snapshot
```yaml
- generic [ref=e1]:
  - radio "Nayra Patel" [ref=e2]
```
"""
    async with await session(replies={"browser_snapshot": page}) as tools:
        await tools.call("browser_snapshot", {})
        await tools.call("browser_click", {"target": "e1"})

    click = next(record for record in tools.calls if record.name == "browser_click")
    assert click.expect_text == "Nayra Patel"


# --- a step held together by counting -------------------------------------
#
# One rung less severe than the undescribable refusal above, at the same
# moment and for the same reason. The element has a name; several others share
# it, so the only thing distinguishing the one picked is how many like it come
# first. A replay honours that position, and the position stops being true the
# moment the list is sorted differently -- at which point the step acts on a
# different record and reports success.

IDENTICAL_ROWS = """### Page
- Page URL: https://vendor.test/accounts
### Snapshot
```yaml
- table [ref=e1]:
  - row [ref=e2]:
    - cell "Acme Ltd" [ref=e3]
    - button "Edit" [ref=e4]
  - row [ref=e5]:
    - cell "Globex" [ref=e6]
    - button "Edit" [ref=e7]
  - row [ref=e8]:
    - cell "Initech" [ref=e9]
    - button "Edit" [ref=e10]
```
"""


async def test_clicking_one_of_several_identical_controls_is_refused_once():
    async with await session(replies={"browser_snapshot": IDENTICAL_ROWS}) as tools:
        await tools.call("browser_snapshot", {})
        result = await tools.call("browser_click", {"target": "e7"})

    assert result.is_error
    assert "held together by counting" in result.text
    assert "sorted or filtered differently" in result.text, "why a position stops being true"
    assert "inside the row" in result.text, "and what to do instead"


async def test_repeating_it_goes_through_because_the_page_may_offer_nothing_better():
    async with await session(replies={"browser_snapshot": IDENTICAL_ROWS}) as tools:
        await tools.call("browser_snapshot", {})
        first = await tools.call("browser_click", {"target": "e7"})
        second = await tools.call("browser_click", {"target": "e7"})

    assert first.is_error
    assert not second.is_error


async def test_the_first_of_several_is_not_refused():
    """`Locator.nth` reads 0 as "no position given", so the first match records
    no position at all -- there is nothing for this to warn about, and warning
    anyway would refuse a step that is exactly as good as it ever was."""
    async with await session(replies={"browser_snapshot": IDENTICAL_ROWS}) as tools:
        await tools.call("browser_snapshot", {})
        result = await tools.call("browser_click", {"target": "e4"})

    assert not result.is_error


async def test_a_uniquely_named_control_is_never_refused():
    async with await session(replies={"browser_snapshot": IDENTICAL_ROWS}) as tools:
        await tools.call("browser_snapshot", {})
        result = await tools.call("browser_click", {"target": "e6"})

    assert not result.is_error


async def test_the_refused_attempt_does_not_reach_the_browser_either():
    fake = FakeMCP({"browser_snapshot": IDENTICAL_ROWS})
    tools = AgentToolSession(fake, allowed_domains=("vendor.test",), may_write=True)
    async with tools:
        await tools.call("browser_snapshot", {})
        await tools.call("browser_click", {"target": "e7"})

    assert [name for name, _ in fake.calls] == ["browser_snapshot"]


async def test_a_page_of_identical_controls_is_objected_to_twice_and_no_more():
    """Measured on a real session: a grid of identically named "answer"
    textboxes drew one objection per box, seven in all, each costing a model
    turn at about twelve thousand tokens. On that page position genuinely was
    the only thing telling them apart, so every objection was answered by
    repeating the call and the recording was no better for it."""
    from agent.session import MAX_POSITIONAL_REFUSALS

    grid = """### Page
- Page URL: https://vendor.test/worksheet
### Snapshot
```yaml
- textbox "answer" [ref=e1]
- textbox "answer" [ref=e2]
- textbox "answer" [ref=e3]
- textbox "answer" [ref=e4]
- textbox "answer" [ref=e5]
- textbox "answer" [ref=e6]
```
"""
    refused = 0
    async with await session(replies={"browser_snapshot": grid}) as tools:
        await tools.call("browser_snapshot", {})
        for ref in ("e2", "e3", "e4", "e5", "e6"):
            result = await tools.call("browser_type", {"target": ref, "text": "1"})
            # Counted by what the refusal says, not by `is_error`: a call that
            # goes through re-renders the page in this fake, which makes the
            # later refs stale, and a stale ref is refused for its own good
            # reasons.
            if result.is_error and "held together by counting" in result.text:
                refused += 1

    assert refused == MAX_POSITIONAL_REFUSALS
