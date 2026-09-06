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
from agent.provider import REF_IN_SNAPSHOT

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

TOOLS = [
    ToolSpec("browser_navigate", "", {"required": ["url"]}),
    ToolSpec("browser_snapshot", "", {}),
    ToolSpec("browser_click", "", {"required": ["target"]}),
    ToolSpec("browser_type", "", {"required": ["target", "text"]}),
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
        text = self.replies.get(name, SNAPSHOT)
        return ToolResult(
            text=text, refs=tuple(dict.fromkeys(REF_IN_SNAPSHOT.findall(text)))
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
    assert verdict.needs_approval
    assert verdict.category


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

    import agent
    import agent.provider
    import agent.session
    import agent.tools

    for module in (agent, agent.provider, agent.session, agent.tools):
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
