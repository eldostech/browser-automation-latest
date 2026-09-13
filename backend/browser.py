"""The browser, driven directly.

This replaces ``mcp_client.py``. That module's opening line was "the only place
in this codebase that talks to a browser", and its central claim was that
routing everything through MCP meant the agent and a human debugging a run saw
the same tool surface. That was true, and it stopped being the point once the
agent went: there is no model in the loop to share a tool surface with, and a
JSON-RPC hop to a Node process that then calls Playwright is a translation
layer between us and the library we actually want.

What the direct route buys, beyond the hop
------------------------------------------
**Auto-waiting.** ``locator.click()`` waits for the element to be attached,
visible, stable and enabled. The hand-rolled settle loop in ``replay.py``
existed to approximate that against a snapshot taken through MCP.

**Real locators.** ``get_by_role(role, name=...)`` is resolved by Playwright
against the live page, with strictness: two matches is an error rather than a
silent first-one-wins. That is the durability argument the recording format was
built on, finally executed by the engine that invented it.

**Traces.** ``context.tracing`` produces a file that opens in Playwright's own
viewer, with a DOM snapshot per action. For a batch that failed on row 412 this
is the difference between a screenshot and being able to look around the page.
It is not reachable through the MCP tool surface at all.

Lifetime
--------
One browser and one context per session, closed in a ``finally`` that also runs
on cancellation -- the discipline ``MCPBrowserSession`` established and the one
thing from it worth keeping verbatim. A leaked Chromium is worse here than it
was there, because nothing else will reap it.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snapshot import Snapshot, parse as parse_snapshot

log = logging.getLogger(__name__)

#: How often :meth:`PlaywrightSession.settle` re-reads the page while waiting
#: for it to stop changing.
SETTLE_INTERVAL = 0.1

#: How many child frames a snapshot reads. An ad-heavy page carries dozens,
#: and reading every one turns a free snapshot into the slowest thing in the
#: step. Real workflows use one or two.
MAX_FRAMES_READ = 8


class BrowserError(RuntimeError):
    """The browser could not be started, or died under us."""


@dataclass(slots=True)
class BrowserConfig:
    browser: str = "chromium"
    headless: bool = True
    #: Written per execution when set, and stored as an artifact.
    trace_dir: Path | None = None
    #: Applies to every action and every navigation.
    timeout_ms: int = 30_000
    storage_state: str | None = None

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> "BrowserConfig":
        config = cls(
            browser=settings.browser_engine,
            headless=settings.browser_headless,
            timeout_ms=int(settings.replay_step_timeout * 1000),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        return config



def _require_a_loop_that_can_spawn() -> None:
    """Refuse early on an event loop that cannot start the Playwright driver.

    Playwright's async API launches its Node driver as a subprocess through
    asyncio, and on Windows only a ``ProactorEventLoop`` can do that -- a
    ``SelectorEventLoop`` raises a bare ``NotImplementedError`` from deep inside
    the library, with nothing in the traceback to suggest what to do about it.

    You get a Selector loop by accident: uvicorn chooses one whenever
    ``--reload`` or ``--workers`` is set, because its own reloader needs one.
    So the ordinary development command is exactly the one that breaks
    replaying, which is worth saying out loud rather than letting a run fail.
    """
    if sys.platform != "win32":
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - always called from a coroutine
        return
    if isinstance(loop, asyncio.ProactorEventLoop):
        return
    raise BrowserError(
        f"this event loop ({type(loop).__name__}) cannot start a browser on "
        "Windows. uvicorn picks it whenever --reload or --workers is set; add "
        "--loop none so it leaves the loop to asyncio, which chooses one that "
        "can spawn processes."
    )


class PlaywrightSession:
    """One browser, one context, one page, for the length of a run.

    An async context manager, because the only safe way to own a browser is to
    have its close in a ``finally`` the language writes for you.
    """

    def __init__(self, config: BrowserConfig) -> None:
        self.config = config
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._tracing = False
        #: Pages the site opened that this session has not adopted yet. A
        #: listener appends here; :meth:`adopt_new_page` decides.
        self._opened: list[Any] = []

    async def __aenter__(self) -> "PlaywrightSession":
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - a deployment problem
            raise BrowserError(
                "Playwright is not installed. Run `pip install -r backend/requirements.txt` "
                "and `python -m playwright install chromium`."
            ) from exc

        _require_a_loop_that_can_spawn()

        try:
            self._playwright = await async_playwright().start()
            launcher = getattr(self._playwright, self.config.browser, None)
            if launcher is None:
                raise BrowserError(f"unknown browser {self.config.browser!r}")
            self._browser = await launcher.launch(headless=self.config.headless)
            self._context = await self._browser.new_context(
                storage_state=self.config.storage_state or None
            )
            self._context.set_default_timeout(self.config.timeout_ms)
            self._context.set_default_navigation_timeout(self.config.timeout_ms)

            if self.config.trace_dir is not None:
                await self._context.tracing.start(screenshots=True, snapshots=True)
                self._tracing = True

            self._page = await self._context.new_page()

            # A click on a target=_blank link opens a page this session never
            # asked for. Before this listener existed the session went on
            # driving the page underneath it, so every remaining step of the
            # row ran against the wrong document and failed several steps
            # later with "could not find" -- on a step that had nothing wrong
            # with it. Registered *after* the first page, which this session
            # opened deliberately and must not treat as a popup. Collected
            # rather than adopted here, because a listener cannot know whether
            # the step that caused it has finished.
            #
            # A named function rather than `self._opened.append`: Playwright
            # stores bookkeeping on the handler it is given, and a builtin
            # method has no attributes to store it on.
            self._context.on("page", self._note_opened)
        except BrowserError:
            await self.__aexit__(None, None, None)
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as one actionable error
            await self.__aexit__(None, None, None)
            raise BrowserError(f"the browser could not be started: {exc}") from exc
        return self

    async def __aexit__(self, *exc: Any) -> None:
        # Every step guarded separately: a failure closing the context must not
        # leave the browser and the driver running.
        if self._tracing and self._context is not None:
            try:
                await self._context.tracing.stop(path=str(self.trace_path))
            except Exception:  # noqa: BLE001 - a trace is a diagnostic, not the run
                log.warning("could not write the trace", exc_info=True)
            self._tracing = False
        for closer in (self._context, self._browser):
            if closer is not None:
                try:
                    await closer.close()
                except Exception:  # noqa: BLE001
                    log.warning("error closing the browser", exc_info=True)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # noqa: BLE001
                log.warning("error stopping playwright", exc_info=True)
        self._context = self._browser = self._playwright = self._page = None

    @property
    def trace_path(self) -> Path | None:
        if self.config.trace_dir is None:
            return None
        return self.config.trace_dir / "trace.zip"

    @property
    def page(self):
        if self._page is None:
            raise BrowserError("the browser session is not open")
        return self._page

    def _note_opened(self, page: Any) -> None:
        """A page the site opened, kept until someone decides what it means."""
        self._opened.append(page)

    # -- what the executor needs -------------------------------------------
    async def adopt_new_page(self) -> str:
        """Make the newest page the site opened the current one.

        Returns the URL adopted, or "" when nothing opened. Called after an
        action rather than from the listener, because during the click itself
        the new page has no URL yet and the old one is still the right answer.

        The *newest* rather than the first: a site that opens two tabs has put
        the person in front of the last one, and following that is what makes
        the recording and the replay agree about which page they are on.
        """
        live = [page for page in self._opened if not page.is_closed()]
        self._opened.clear()
        if not live:
            return ""
        adopted = live[-1]
        for stale in live[:-1]:
            # Left open rather than closed: a site that opens two tabs may
            # still be using the other one, and closing somebody's page is not
            # this session's call to make.
            log.info("a second page was opened and not adopted", extra={"url": stale.url})
        self._page = adopted
        try:
            await adopted.wait_for_load_state("domcontentloaded", timeout=self.config.timeout_ms)
        except Exception:  # noqa: BLE001 - adopted anyway; the caller retries
            log.debug("adopted page did not finish loading", exc_info=True)
        try:
            await adopted.bring_to_front()
        except Exception:  # noqa: BLE001 - headless, or already frontmost
            pass
        return self.url

    async def settle(self, timeout_ms: int | None = None) -> None:
        """Wait for the page to stop arriving, within a bound.

        Never raises and never waits long. This is not the step's wait -- the
        locator resolver polls, and actions auto-wait -- it is there so that
        what gets *observed* after an action is the page the action produced.

        Reading a page mid-navigation is not merely a stale answer. The
        snapshot taken when a step fails is the evidence a repair is proposed
        from and the text that goes into healing memory, so observing the
        outgoing page teaches the system something false about the site, and
        keeps teaching it every time that memory is recalled.
        """
        budget = timeout_ms if timeout_ms is not None else min(self.config.timeout_ms, 5_000)
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=budget)
        except Exception:  # noqa: BLE001 - a slow page is not a failed step
            return

        # Then wait for the tree itself to stop changing, which is what a
        # single-page app does after the document is already "loaded". Two
        # identical reads in a row is enough; the bound matters more than the
        # precision, because the resolver retries anyway.
        deadline = asyncio.get_running_loop().time() + budget / 1000.0
        previous = -1
        while asyncio.get_running_loop().time() < deadline:
            try:
                state, size = await self.page.evaluate(
                    "() => [document.readyState, document.getElementsByTagName('*').length]"
                )
            except Exception:  # noqa: BLE001 - mid-navigation; that is an answer
                return
            if state == "complete" and size == previous:
                return
            previous = size
            await asyncio.sleep(SETTLE_INTERVAL)

    async def snapshot(self) -> Snapshot:
        """The accessibility tree, parsed.

        The same YAML Playwright MCP returned, from the library rather than
        from a subprocess -- minus the ``[ref=eN]`` markers, which the MCP
        server added for its own addressing and which nothing here needs now
        that elements are addressed by locator.

        Frames are read too, and appended under a header naming the frame. An
        iframe's contents are a separate document: ``aria_snapshot`` on the
        main frame's body does not descend into one, so before this every
        element inside a payment form, a chat widget or an embedded editor was
        invisible to assertions, to the failure context a repair reads, and to
        the healer's candidate list -- not hard to find, absent.

        Never raises: a page mid-navigation can refuse to be read, and a
        snapshot is an input to locator resolution, which retries.
        """
        try:
            text = await self.page.locator("body").aria_snapshot()
        except Exception:  # noqa: BLE001 - mid-navigation, or no body yet
            return Snapshot()
        snapshot = parse_snapshot(text + await self._frame_snapshots())
        try:
            snapshot.page_url = self.page.url
            snapshot.page_title = await self.page.title()
        except Exception:  # noqa: BLE001
            pass
        return snapshot

    async def _frame_snapshots(self) -> str:
        """Every child frame's tree, indented under a line naming the frame.

        Bounded by :data:`MAX_FRAMES_READ`: an ad-heavy page can carry dozens
        of frames, and reading all of them would turn a free snapshot into the
        slowest thing in the step.
        """
        try:
            frames = [frame for frame in self.page.frames if frame.parent_frame is not None]
        except Exception:  # noqa: BLE001 - mid-navigation
            return ""

        parts: list[str] = []
        for index, frame in enumerate(frames[:MAX_FRAMES_READ]):
            try:
                body = await frame.locator("body").aria_snapshot()
            except Exception:  # noqa: BLE001 - a cross-origin or dead frame
                continue
            if not body.strip():
                continue
            # `iframe` is a real ARIA role, so this parses as a node like any
            # other and the name says which frame it was -- which is what a
            # person reading a failure context needs, and what tells the
            # healer that a candidate is not on the main page.
            label = frame.name or frame.url or f"frame {index}"
            indented = "\n".join(f"  {line}" for line in body.splitlines())
            parts.append(f'- iframe "{label}":\n{indented}')
        return ("\n" + "\n".join(parts)) if parts else ""

    async def screenshot(self) -> bytes | None:
        """A screenshot, or None. Never fails the run it belongs to."""
        try:
            return await self.page.screenshot(full_page=False)
        except Exception:  # noqa: BLE001 - a diagnostic aid, not the work
            log.debug("screenshot failed", exc_info=True)
            return None

    @property
    def url(self) -> str:
        try:
            return self.page.url
        except Exception:  # noqa: BLE001
            return ""

    async def title(self) -> str:
        try:
            return await self.page.title()
        except Exception:  # noqa: BLE001
            return ""

    async def text_content(self) -> str:
        """The visible text of the page, for `text_present` assertions."""
        try:
            return await self.page.locator("body").inner_text()
        except Exception:  # noqa: BLE001
            return ""


__all__ = ["BrowserConfig", "BrowserError", "PlaywrightSession"]
