"""Engine and session construction.

**The URL is built, never formatted.** A password containing ``@`` -- and this
project's does -- terminates the userinfo section of a URL, so
``f"postgresql://{user}:{password}@{host}"`` silently produces a DSN pointing
at whatever followed it rather than at the database. ``URL.create()`` escapes
each component itself, which is why the credential parts stay separate
settings rather than one connection string.

**Two drivers, for two jobs.** The application runs on asyncpg; Alembic, which
is synchronous, runs on psycopg. That is not an accident of taste -- on Windows
psycopg's async mode requires a ``SelectorEventLoop``, while ``asyncio``
subprocesses require a ``ProactorEventLoop``, and this process must do both: it
spawns ``npx @playwright/mcp`` over stdio and talks to Postgres in the same
loop. asyncpg works under either policy, so it is the only driver that lets the
browser and the database coexist. Alembic never touches an event loop, so the
constraint does not reach it.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import Settings
from db.base import Base, DEFAULT_SCHEMA

log = logging.getLogger(__name__)


def _assert_schema_matches(settings: Settings) -> None:
    """Refuse to run if the configured schema is not the one the models use.

    ``MetaData`` fixes the schema when the model modules are imported, from the
    ``DB_SCHEMA`` environment variable. If ``Settings.db_schema`` says something
    else -- because it was passed programmatically, or a ``.env`` disagrees with
    the process environment -- then DDL lands in one schema while queries read
    another, and the symptom is an empty database rather than an error. Fail
    here instead, where the message can say what to fix.
    """
    if settings.db_schema != DEFAULT_SCHEMA:
        raise RuntimeError(
            f"Configured db_schema is {settings.db_schema!r} but the models were built "
            f"for {DEFAULT_SCHEMA!r}. The schema is fixed when db.base is imported, so "
            f"set the DB_SCHEMA environment variable to {settings.db_schema!r} before "
            "importing the application."
        )

#: The application driver. See the module docstring for why it is not psycopg.
ASYNC_DRIVER = "postgresql+asyncpg"
#: The migration driver: synchronous, so the event-loop constraint is moot.
SYNC_DRIVER = "postgresql+psycopg"


def database_url(settings: Settings, *, driver: str = ASYNC_DRIVER) -> URL:
    """The SQLAlchemy URL for this configuration."""
    return URL.create(
        drivername=driver,
        username=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
    )


def sync_database_url(settings: Settings) -> str:
    """Render for Alembic."""
    return database_url(settings, driver=SYNC_DRIVER).render_as_string(hide_password=False)


def create_engine(settings: Settings) -> AsyncEngine:
    """An async engine with the pool sized for a web process.

    ``pool_pre_ping`` is not optional here. A connection idle in the pool
    across a Postgres restart, a failover, or an idle-timeout looks fine until
    it is used, and the resulting error surfaces as a random 500 on whichever
    request happened to draw it.
    """
    _assert_schema_matches(settings)
    return create_async_engine(
        database_url(settings),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
        echo=settings.db_echo,
        # Every connection lands in the application schema without any query
        # naming it. SQLAlchemy already qualifies its own SQL from the metadata
        # schema; this covers the raw SQL -- the queue's SKIP LOCKED claim and
        # the LISTEN/NOTIFY channel -- which does not know about our MetaData.
        #
        # asyncpg takes this as a connection parameter rather than a statement
        # on connect: it is applied before the first query on a fresh
        # connection, so there is no window in which a pooled connection is
        # pointing at the wrong schema.
        connect_args={
            "server_settings": {
                "search_path": f"{settings.db_schema},public",
                # Shows up in pg_stat_activity, which is the difference between
                # diagnosing a lock and guessing at one.
                "application_name": "browser-automation",
            }
        },
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        expire_on_commit=False,  # objects stay usable after commit
        autoflush=False,
    )


async def ensure_schema(engine: AsyncEngine, schema: str) -> None:
    """Create the schema if it is missing.

    Tables are Alembic's job. This is only the container they live in, which
    must exist before the first migration can run.
    """
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


async def create_all(engine: AsyncEngine) -> None:
    """Build the schema directly from the models, skipping Alembic.

    For tests. Production goes through migrations so that upgrades are
    reproducible; a test wants the current shape in one round trip.
    """
    import db.models  # noqa: F401 - registers the tables on Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all(engine: AsyncEngine) -> None:
    import db.models  # noqa: F401 - registers the tables on Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Unit of work: one transaction, committed on success, rolled back on error.

    Every request handler and every job execution runs inside exactly one of
    these, which is what makes "the run row and its first event either both
    exist or neither does" true rather than hopeful.
    """
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
