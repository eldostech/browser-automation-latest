"""Start the API on an event loop that can launch a browser.

Why this exists rather than a documented uvicorn command
--------------------------------------------------------
Playwright launches its driver as a subprocess through asyncio, and on Windows
only a ``ProactorEventLoop`` can do that. uvicorn switches to a
``SelectorEventLoop`` whenever ``--reload`` or ``--workers`` is set -- so the
obvious `uvicorn main:app --reload` is the one command under which nothing can
be recorded or replayed.

``--loop none`` leaves the loop to asyncio, which picks the one that works. But
that alone trades one failure for another, and a worse one: uvicorn's reloader
binds the listening socket in the *parent* and hands it to the child, and on
Windows an inherited socket cannot be registered with the child's IOCP. The
Proactor loop then fails every accept with

    OSError: [WinError 87] The parameter is incorrect

which arrives *after* "Application startup complete", so the server looks up
and answers nothing. That is what uvicorn's Selector loop is protecting against,
and it is why the two settings cannot simply be combined.

So reloading happens a level up. ``watchfiles`` restarts this whole process,
and each new process binds its own socket in the loop that will use it. Nothing
is inherited, both constraints are satisfied, and edits still take effect.

    python serve.py                 # reload on, port 8000
    PORT=8002 RELOAD=false python serve.py
"""

from __future__ import annotations

import os
from pathlib import Path


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def serve() -> None:
    """One server, binding its own socket. The target of a reload restart."""
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        # Both of these are load-bearing; see the note above before changing
        # either. uvicorn must not reload (its socket handover breaks the
        # Proactor loop) and must not choose the loop (its choice cannot spawn
        # a browser).
        reload=False,
        loop="none",
    )


def main() -> None:
    if not _flag("RELOAD", True):
        serve()
        return

    try:
        from watchfiles import PythonFilter, run_process
    except ImportError:
        # Reloading is a convenience; serving is not. Losing the watcher must
        # not cost you the server.
        print(
            "watchfiles is not installed, so edits will not restart the server. "
            "Install it, or set RELOAD=false to stop seeing this.",
            flush=True,
        )
        serve()
        return

    here = Path(__file__).resolve().parent
    print(f"watching {here} for changes; set RELOAD=false to stop", flush=True)
    run_process(here, target=serve, watch_filter=PythonFilter())


if __name__ == "__main__":
    main()
