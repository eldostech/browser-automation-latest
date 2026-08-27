"""Alembic environment.

Two things here are not boilerplate:

* The URL comes from ``Settings``, not from alembic.ini, so `alembic upgrade`
  and the running application cannot disagree about which database they mean.
* ``include_schemas`` plus ``version_table_schema`` keep Alembic's own bookkeeping
  table inside the application schema. Left at the default it lands in
  ``public``, and then "drop the browser schema" leaves a version stamp behind
  claiming migrations that no longer exist have been applied.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool, text

# The backend is a flat module tree, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from db.base import Base  # noqa: E402
from db.engine import database_url  # noqa: E402
import db.models  # noqa: E402,F401 - import for the side effect of registering tables

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()

# The URL is handed to create_engine() directly and never written into the
# alembic config. Two reasons, both learned the hard way:
#   * configparser interpolates '%', and a URL-escaped password ('@' -> '%40')
#     therefore raises ValueError before a single migration runs;
#   * a password in the config object ends up in Alembic's error output.
# Passing a URL object keeps it escaped exactly once and unprinted.
URL = database_url(settings, driver="postgresql+psycopg")

target_metadata = Base.metadata
SCHEMA = settings.db_schema


def include_object(obj, name, type_, reflected, compare_to):  # noqa: ANN001 - alembic hook
    """Ignore everything that is not ours.

    This is a safety device, not a tidiness one. The database this deploys
    against is shared: autogenerate reflects *every* table it can see,
    including other applications', and writes DROP statements for the ones
    missing from our metadata.

    The comparison must be exact. An earlier version allowed ``None``
    alongside the schema name, reasoning that unqualified meant ours -- but
    reflected ``public`` tables report ``schema=None``, so the filter passed
    another application's ``alembic_version`` straight through into a
    ``drop_table``. Anything not explicitly in our schema is somebody else's.
    """
    if type_ == "table":
        return obj.schema == SCHEMA
    # Indexes and constraints inherit the decision made about their table.
    parent = getattr(obj, "table", None)
    if parent is not None:
        return parent.schema == SCHEMA
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        include_object=include_object,
        version_table_schema=SCHEMA,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        # The schema must exist before the version table can be created in it.
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
        connection.execute(text(f'SET search_path TO "{SCHEMA}", public'))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            include_object=include_object,
            version_table_schema=SCHEMA,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
