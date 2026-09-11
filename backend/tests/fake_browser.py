"""A Playwright-shaped browser double, driven by snapshot fixtures.

The API tests used to inject an MCP-shaped fake, because that is what the
executor talked to. The executor talks to Playwright now, so the double had to
change shape -- but not substance: it serves the same ``SIGNED_OUT`` /
``SIGNED_IN`` accessibility snapshots those tests were already written against,
so what they assert is unchanged and only the surface underneath moved.

It is a fake and not a mock: locators resolve against the parsed snapshot by
the same rules the real engine uses, a click advances the page the way signing
in does, and an element that is not in the snapshot is not found. That is
enough for every test above the browser, and the real thing is covered by
``test_e2e_engine.py`` against a real Chromium.
"""

from __future__ import annotations

from typing import Any

from snapshot import parse as parse_snapshot

#: A one-pixel PNG, so "a screenshot was saved" can be asserted on its bytes.
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
    b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeTimeout(Exception):
    """Stands in for Playwright's TimeoutError."""


class FakeLocator:
    """Zero or more nodes from the current snapshot, with actions on them."""

    def __init__(self, page: "FakePage", nodes: list, describe: str) -> None:
        self._page = page
        self._nodes = nodes
        self._describe = describe

    # -- shape --------------------------------------------------------------
    async def count(self) -> int:
        return len(self._nodes)

    @property
    def first(self) -> "FakeLocator":
        return FakeLocator(self._page, self._nodes[:1], self._describe)

    @property
    def last(self) -> "FakeLocator":
        return FakeLocator(self._page, self._nodes[-1:], self._describe)

    def nth(self, index: int) -> "FakeLocator":
        chosen = self._nodes[index : index + 1] if index < len(self._nodes) else []
        return FakeLocator(self._page, chosen, f"{self._describe}[{index}]")

    def filter(self, has_text: str | None = None, visible: bool | None = None) -> "FakeLocator":
        """`visible=True` is the identity here, and that is not a shortcut.

        An accessibility snapshot only contains what is in the accessibility
        tree, so every node this fake holds is one a person could reach. The
        engine's visible-first counting therefore behaves here exactly as it
        does on a page with no hidden duplicates, which is the case the
        existing tests were written against. The case it exists *for* -- a
        hidden twin of a visible control -- cannot be expressed in a snapshot
        at all, and is covered against a real browser in `test_e2e_engine.py`.
        """
        nodes = self._nodes
        if has_text is not None:
            wanted = has_text.casefold()
            nodes = [
                node
                for node in nodes
                if wanted in (node.name or "").casefold()
                or wanted in (node.text or "").casefold()
            ]
        described = self._describe
        if has_text is not None:
            described = f"{described} has_text={has_text!r}"
        return FakeLocator(self._page, nodes, described)

    # -- as a scope ---------------------------------------------------------
    #
    # A locator is also a place to search inside. The fake resolves a scope by
    # searching the whole snapshot and then keeping what falls under the
    # scope's node, which is what the indentation of an accessibility tree
    # means. Enough to tell "the button in the dialog" from "the button in the
    # sidebar", which is the whole point of `within`.
    def _descendants(self) -> list:
        snapshot = self._page._snapshot()
        nodes = list(snapshot)
        kept: list = []
        for scope in self._nodes:
            start = nodes.index(scope) if scope in nodes else -1
            if start < 0:
                continue
            for node in nodes[start + 1 :]:
                if node.depth <= scope.depth:
                    break
                kept.append(node)
        return kept

    def _scoped(self, produce) -> "FakeLocator":
        allowed = self._descendants()
        found = produce(self._page)
        return FakeLocator(
            self._page,
            [node for node in found._nodes if node in allowed],
            f"{found._describe} in {self._describe}",
        )

    def get_by_role(self, role: str, name: str | None = None, exact: bool = False):
        return self._scoped(lambda page: page.get_by_role(role, name, exact))

    def get_by_label(self, text: str, exact: bool = False, **_: Any):
        return self._scoped(lambda page: page.get_by_label(text, exact))

    def get_by_placeholder(self, text: str, exact: bool = False, **_: Any):
        return self._scoped(lambda page: page.get_by_placeholder(text, exact))

    def get_by_alt_text(self, text: str, exact: bool = False, **_: Any):
        return self._scoped(lambda page: page.get_by_alt_text(text, exact))

    def get_by_test_id(self, text: str):
        return self._scoped(lambda page: page.get_by_test_id(text))

    def get_by_text(self, text: str, exact: bool = False, **_: Any):
        return self._scoped(lambda page: page.get_by_text(text, exact))

    def locator(self, selector: str):
        return self._scoped(lambda page: page.locator(selector))

    def _require(self):
        if not self._nodes:
            raise FakeTimeout(f"Timeout: no element matches {self._describe}")
        return self._nodes[0]

    # -- actions ------------------------------------------------------------
    async def click(self, **_: Any) -> None:
        self._require()
        self._page.clicks.append(self._describe)
        self._page.calls.append(("click", {"target": self._describe}))
        self._page.advance()

    async def fill(self, value: str, **_: Any) -> None:
        node = self._require()
        name = node.name or self._describe
        self._page.filled[name] = value
        self._page.calls.append(("fill", {"element": name, "text": value}))

    async def select_option(self, value: str, **_: Any) -> None:
        node = self._require()
        self._page.filled[node.name or self._describe] = value

    async def hover(self, **_: Any) -> None:
        self._require()

    async def press(self, key: str, **_: Any) -> None:
        self._require()
        self._page.keys.append(key)

    async def set_input_files(self, paths, **_: Any) -> None:
        self._require()
        self._page.uploads.extend(paths)

    async def is_visible(self, **_: Any) -> bool:
        return bool(self._nodes)

    async def inner_text(self, **_: Any) -> str:
        node = self._require()
        return node.text or node.name or ""

    async def wait_for(self, **_: Any) -> None:
        self._require()

    async def aria_snapshot(self, **_: Any) -> str:
        return self._page.text


class _Keyboard:
    def __init__(self, page: "FakePage") -> None:
        self._page = page

    async def press(self, key: str) -> None:
        self._page.keys.append(key)


class FakePage:
    """The current page, as an accessibility snapshot plus a little state."""

    def __init__(self, pages: list[str], routes: dict[str, str] | None = None) -> None:
        self.pages = pages
        self.routes = routes or {}
        self.index = 0
        self._pinned: str | None = None
        self.url = "about:blank"
        self.clicks: list[str] = []
        self.keys: list[str] = []
        self.uploads: list[str] = []
        self.filled: dict[str, str] = {}
        self.visited: list[str] = []
        #: (action, arguments) for every action performed, so a test can assert
        #: what the engine actually did rather than only what came back.
        self.calls: list[tuple[str, dict]] = []
        self.keyboard = _Keyboard(self)
        #: What `document.readyState` reports. A test showing a page that is
        #: still arriving sets this to "loading".
        self.ready_state = "complete"

    @property
    def text(self) -> str:
        return self._pinned or self.pages[min(self.index, len(self.pages) - 1)]

    def advance(self) -> None:
        """A click moves to the next page, which is how signing in behaves."""
        self._pinned = None
        self.index += 1

    def _snapshot(self):
        return parse_snapshot(self.text)

    # -- navigation ---------------------------------------------------------
    async def goto(self, url: str, **_: Any) -> None:
        self.url = url
        self.visited.append(url)
        # A route pins its page however many times you navigate to it, so a
        # fake that blindly advances does not make tests pass for reasons the
        # real system would never produce.
        for fragment, page in self.routes.items():
            if fragment in url:
                self._pinned = page
                return
        # An unrouted navigation moves on, the way following a link does. This
        # is the semantics the snapshot fixtures were written against, and
        # keeping it is what lets the tests above the browser stay unchanged.
        self._pinned = None
        self.advance()

    async def title(self) -> str:
        return self._snapshot().page_title or ""

    async def wait_for_load_state(self, *_: Any, **__: Any) -> None:
        return None

    async def evaluate(self, code: str) -> Any:
        self.scripts = getattr(self, "scripts", [])
        self.scripts.append(code)
        if "readyState" in code:
            # The engine's "is this page still changing?" probe. Answering it
            # from the fake's own state is what lets a test show a page that
            # renders late, which is the case the locator wait exists for.
            return [self.ready_state, len(list(self._snapshot()))]
        return None

    async def screenshot(self, **_: Any) -> bytes:
        return PNG

    def is_closed(self) -> bool:
        return False

    async def bring_to_front(self) -> None:
        return None

    @property
    def frames(self) -> list:
        """No child frames. A fake that invented one would let a frame rung
        pass here and fail against a real page."""
        return []

    def frame_locator(self, selector: str) -> FakeLocator:
        """Matches nothing, for the same reason `locator` does for CSS.

        A frame cannot be resolved against an accessibility snapshot, so
        pretending otherwise would let a test pass on a rung the real engine
        would have to fall through. Frames are covered against a real browser
        in `test_e2e_engine.py`.
        """
        return FakeLocator(self, [], f"frame={selector!r}")

    # -- locators -----------------------------------------------------------
    def get_by_role(
        self, role: str, name: str | None = None, exact: bool = False
    ) -> FakeLocator:
        """Careful: the loose path here is kinder than Playwright's.

        ``Snapshot.find`` tries an exact name, then a normalised one, then a
        substring, and stops at the first tier that yields anything. Playwright
        goes straight to the substring. So a page holding both "Invite" and
        "+ Invite User" resolves to one node here and to two there -- which is
        why an ambiguous locator can only be caught in the real-browser tests,
        and why one lived in `_resolve` long enough to break a run.
        """
        if exact and name is not None:
            nodes = [
                node
                for node in self._snapshot().find(role)
                if (node.name or "").strip() == name
            ]
        else:
            nodes = self._snapshot().find(role, name)
        return FakeLocator(self, nodes, f'role={role} name="{name}"')

    def _by_name(self, value: str, label: str, exact: bool = False) -> FakeLocator:
        snapshot = self._snapshot()
        if exact:
            nodes = [node for node in snapshot if (node.name or "").strip() == value]
            return FakeLocator(self, nodes[:1], f"{label}={value!r}")
        node = snapshot.by_name(value)
        return FakeLocator(self, [node] if node else [], f"{label}={value!r}")

    def get_by_label(self, text: str, exact: bool = False, **_: Any) -> FakeLocator:
        return self._by_name(text, "label", exact)

    def get_by_placeholder(self, text: str, exact: bool = False, **_: Any) -> FakeLocator:
        return self._by_name(text, "placeholder", exact)

    def get_by_alt_text(self, text: str, exact: bool = False, **_: Any) -> FakeLocator:
        return self._by_name(text, "alt_text", exact)

    def get_by_test_id(self, text: str) -> FakeLocator:
        return self._by_name(text, "test_id")

    def get_by_text(self, text: str, exact: bool = False, **_: Any) -> FakeLocator:
        """Any node whose name or text contains this, which is what
        `get_by_text` does on a real page -- unless `exact`, which is the whole
        string and is why a recorded `exact=True` has to reach this far."""
        wanted = text.casefold()

        def holds(node) -> bool:
            name, body = (node.name or ""), (node.text or "")
            if exact:
                return name.strip() == text or body.strip() == text
            return wanted in name.casefold() or wanted in body.casefold()

        return FakeLocator(self, [n for n in self._snapshot() if holds(n)], f"text={text!r}")

    def locator(self, selector: str) -> FakeLocator:
        """`body` is the whole page; anything else matches nothing.

        A CSS selector cannot be resolved against an accessibility tree, and a
        fake that pretended otherwise would let a test pass on a rung the real
        engine would have to fall through.
        """
        if selector == "body":
            return FakeLocator(self, [_Body(self.text)], "css=body")
        return FakeLocator(self, [], f"css={selector}")


class _Body:
    """Stands in for the page body when the engine reads whole-page text."""

    def __init__(self, text: str) -> None:
        self.name = ""
        self.text = text


class FakeBrowserSession:
    """What ``PlaywrightSession`` is replaced with in an API test."""

    #: Set on entry so a test can inspect what happened.
    current: "FakeBrowserSession | None" = None

    pages: list[str] = []
    routes: dict[str, str] = {}

    def __init__(self, config: Any) -> None:
        self.config = config
        self._page = FakePage(list(self.pages), dict(self.routes))

    async def __aenter__(self) -> "FakeBrowserSession":
        type(self).current = self
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    @property
    def page(self) -> FakePage:
        return self._page

    @property
    def calls(self) -> list[tuple[str, dict]]:
        """Every action the engine performed, for tests that assert on it."""
        return self._page.calls

    @property
    def url(self) -> str:
        return self._page.url

    async def settle(self, timeout_ms: int | None = None) -> None:
        """Already settled. The fake serves whole pages, never half of one."""
        return None

    async def adopt_new_page(self) -> str:
        """No popups. A fake that opened tabs would be testing itself."""
        return ""

    async def snapshot(self):
        snap = parse_snapshot(self._page.text)
        snap.page_url = self._page.url
        return snap

    async def screenshot(self) -> bytes:
        return PNG

    async def title(self) -> str:
        return await self._page.title()

    async def text_content(self) -> str:
        return self._page.text


def session_serving(pages: list[str], routes: dict[str, str] | None = None):
    """A session class serving these pages, for `build_app(session_cls=...)`."""
    return type(
        "ConfiguredFakeSession",
        (FakeBrowserSession,),
        {"pages": list(pages), "routes": dict(routes or {})},
    )




# ---------------------------------------------------------------------------
# The pages the tests replay against
# ---------------------------------------------------------------------------
#
# These moved here from ``test_replay.py`` when that module went. They are
# written in Playwright MCP's snapshot dialect, refs and all, because the
# parser reads both and rewriting them would have changed what a dozen tests
# assert for no reason.

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


def role(name: str, kind: str = "button", nth: int = 0):
    """A role locator, which is what most recorded steps carry."""
    from usecase import Locator

    return Locator(strategy="role", role=kind, name=name, nth=nth)


__all__ = [
    "SIGNED_IN",
    "SIGNED_OUT",
    "FakeBrowserSession",
    "FakeLocator",
    "FakePage",
    "FakeTimeout",
    "PNG",
    "role",
    "session_serving",
]
