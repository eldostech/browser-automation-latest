"""Database layer: schema, engine, and session lifecycle.

``store.py`` sits on top of this and exposes the task-shaped surface the rest
of the backend calls. Nothing outside this package should build a URL, hold an
engine, or import a model directly for a query -- go through the Store.
"""

from db.base import Base, iso, metadata_obj, new_id, utcnow
from db.engine import (
    create_all,
    create_engine,
    create_session_factory,
    database_url,
    drop_all,
    ensure_schema,
    session_scope,
    sync_database_url,
)

__all__ = [
    "Base",
    "create_all",
    "create_engine",
    "create_session_factory",
    "database_url",
    "drop_all",
    "ensure_schema",
    "iso",
    "metadata_obj",
    "new_id",
    "session_scope",
    "sync_database_url",
    "utcnow",
]
