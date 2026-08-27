"""Where LangGraph keeps the state of a run in progress.

Until now this was ``InMemorySaver``, which meant the one concrete benefit the
graph was adopted for -- a run surviving a restart -- did not actually exist.
State died with the process, and ``reap_orphaned_runs`` marked the run failed
because there was nothing left to resume from.

**Postgres where it can run, SQLite where it cannot.** LangGraph's Postgres
saver is built on psycopg, and on Windows psycopg's async mode requires a
``SelectorEventLoop`` while ``asyncio`` subprocesses require a
``ProactorEventLoop``. This process needs both -- it spawns
``npx @playwright/mcp`` over stdio and talks to the database in the same loop --
so on Windows the Postgres saver cannot be used at all. See ``db/engine.py``,
where the same constraint chose asyncpg for the application's own queries.

The fallback is a real file, not memory, so the durability property holds on
both platforms. What differs is reach: a SQLite checkpoint is visible only to
the worker that wrote it, so a run interrupted on one machine resumes on that
machine. Since a run is pinned to the worker holding its job lease, that is the
common case anyway; the Postgres saver is what makes recovery possible when the
*machine* is gone, which is why production (Linux) gets it.

This is a platform adaptation with one code path, not a configurable backend.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from config import REPO_ROOT, Settings
from db.engine import database_url

log = logging.getLogger(__name__)

#: psycopg's async mode cannot run on Windows' default event loop policy, and
#: the browser subprocess cannot run on the alternative.
POSTGRES_SAVER_USABLE = sys.platform != "win32"


def checkpoint_db_path(settings: Settings) -> Path:
    path = Path(settings.checkpoint_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class Checkpointer:
    """Owns the saver and its connection, opened and closed with the app.

    An async context manager because both savers hold a connection that must
    be released; the application enters it in ``lifespan``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cm: Any = None
        self.saver: Any = None
        self.backend: str = "none"

    async def __aenter__(self) -> "Checkpointer":
        if not self._settings.checkpoint_enabled:
            # Explicitly off: the graph falls back to InMemorySaver, which is
            # right for tests and for anyone who does not want the extra
            # connection.
            self.backend = "memory"
            return self

        if POSTGRES_SAVER_USABLE:
            try:
                await self._open_postgres()
                return self
            except Exception:  # noqa: BLE001 - fall back rather than fail to boot
                log.exception("could not open the Postgres checkpointer; using SQLite")

        await self._open_sqlite()
        return self

    async def _open_postgres(self) -> None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # psycopg's own URL form, not SQLAlchemy's.
        url = database_url(self._settings, driver="postgresql+psycopg").render_as_string(
            hide_password=False
        ).replace("postgresql+psycopg://", "postgresql://")

        self._cm = AsyncPostgresSaver.from_conn_string(url)
        self.saver = await self._cm.__aenter__()
        await self.saver.setup()
        self.backend = "postgres"
        log.info("checkpointer ready", extra={"backend": self.backend})

    async def _open_sqlite(self) -> None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = checkpoint_db_path(self._settings)
        self._cm = AsyncSqliteSaver.from_conn_string(str(path))
        self.saver = await self._cm.__aenter__()
        await self.saver.setup()
        self.backend = "sqlite"
        log.info("checkpointer ready", extra={"backend": self.backend, "path": str(path)})

    async def __aexit__(self, *exc_info) -> bool:
        if self._cm is not None:
            try:
                await self._cm.__aexit__(*exc_info)
            except Exception:  # noqa: BLE001 - shutdown must not raise
                log.debug("checkpointer close failed", exc_info=True)
            self._cm = None
            self.saver = None
        return False


__all__ = ["Checkpointer", "POSTGRES_SAVER_USABLE", "checkpoint_db_path"]
