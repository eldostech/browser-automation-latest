"""The physical schema.

Ported from the hand-written SQLite DDL that used to live in ``store.py``,
with four changes that the single-user version did not need:

* **Tenancy.** Every row a user can reach carries ``workspace_id``. This is the
  column that makes a second user safe, and it is deliberately ``NOT NULL``
  with a foreign key -- an orphan row is a row nobody can see and nobody can
  clean up.
* **Ownership.** Resources also carry ``owner_id``, which answers "who made
  this" for display and for owner-scoped permissions. It is nullable and
  ``ON DELETE SET NULL``: deleting a user must not delete their team's work.
* **Real types.** ``timestamptz`` instead of ISO strings, ``JSONB`` instead of
  serialized TEXT. The API still emits ISO strings; see ``base.iso``.
* **Identity, audit and queueing tables**, which have no SQLite ancestor.

Cascade rules follow one principle: *containment cascades, reference nulls*.
Events belong to a run and die with it; a run's ``owner_id`` merely points at a
user and survives them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, created_at_column, id_column, new_id, utcnow


def _json_column(default: Any = dict, nullable: bool = False):
    return mapped_column(JSONB, nullable=nullable, default=default)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class Workspace(Base):
    """The tenant boundary. Every other table is scoped to one of these."""

    __tablename__ = "workspaces"

    id: Mapped[str] = id_column()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    created_at: Mapped[datetime] = created_at_column()


class User(Base):
    """A local account.

    There is no SSO here yet, so ``password_hash`` is the credential. When an
    OIDC provider is added it becomes nullable and a ``federated_identities``
    table joins alongside -- the ``role`` column and everything that reads it
    stay exactly as they are. That separation is the point: authentication is
    replaceable, authorization is not.
    """

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        Index("ix_users_workspace", "workspace_id"),
    )

    id: Mapped[str] = id_column()
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    #: bcrypt. Never selected into any response model; see auth/service.py.
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="operator")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = created_at_column()
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Set when a password is changed; every session issued before this instant
    #: is refused. Cheaper and more reliable than hunting down session rows.
    credentials_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class UserSession(Base):
    """An opaque bearer token.

    Opaque rather than a JWT, deliberately: revocation is a DELETE that takes
    effect on the next request. A stateless JWT cannot be withdrawn before it
    expires without building exactly this table to check against anyway.

    Only the SHA-256 of the token is stored, so a database leak does not hand
    over live sessions.
    """

    __tablename__ = "user_sessions"
    __table_args__ = (
        Index("ix_user_sessions_user", "user_id"),
        Index("ix_user_sessions_expires", "expires_at"),
    )

    id: Mapped[str] = id_column()
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = created_at_column()
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str] = mapped_column(String(300), nullable=False, default="")


class AuditLogEntry(Base):
    """Append-only record of who did what.

    ``actor_email`` is denormalized on purpose: the audit trail has to stay
    readable after the user row is gone, and a join that can return NULL is not
    an audit trail.
    """

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_workspace_created", "workspace_id", "created_at"),
        Index("ix_audit_log_resource", "resource_type", "resource_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    actor_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    actor_email: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    resource_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    detail: Mapped[dict] = _json_column()
    created_at: Mapped[datetime] = created_at_column()


# ---------------------------------------------------------------------------
# Runs, events, artifacts
# ---------------------------------------------------------------------------


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_workspace_status_created", "workspace_id", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    #: Denormalized, like ``audit_log.actor_email`` and for the same reason: a
    #: record of who did something has to stay readable after the account is
    #: deleted, and ``owner_id`` is ON DELETE SET NULL. An id alone is also
    #: unreadable in a UI without a join on every row.
    #: ``server_default`` as well as ``default``: the column was added to
    #: populated tables, so the database needs a value for the rows that
    #: predate it. Blank means "nobody recorded who", which is the truth about
    #: those rows rather than a placeholder pretending otherwise.
    owner_email: Mapped[str] = mapped_column(
        String(320), nullable=False, default="", server_default=""
    )
    task: Mapped[str] = mapped_column(Text, nullable=False)
    start_url: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    options: Mapped[dict] = _json_column()
    created_at: Mapped[datetime] = created_at_column()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    summary: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict | None] = _json_column(default=None, nullable=True)
    error: Mapped[str | None] = mapped_column(Text)


class Event(Base):
    """The run timeline, replayed to the UI from ``seq`` on reconnect.

    Not the system of record -- the run/execution rows are. This is a log, and
    keeping the distinction is what stops the project drifting into
    event-sourcing it does not need.
    """

    __tablename__ = "events"
    __table_args__ = (Index("ix_events_run_seq", "run_id", "seq"),)

    run_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True
    )
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = _json_column()


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (Index("ix_artifacts_run_seq", "run_id", "seq"),)

    id: Mapped[str] = id_column()
    run_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int | None] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    mime: Mapped[str] = mapped_column(String(100), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = created_at_column()


# ---------------------------------------------------------------------------
# Use cases
# ---------------------------------------------------------------------------


class UseCase(Base):
    __tablename__ = "usecases"
    __table_args__ = (
        Index("ix_usecases_workspace_status_updated", "workspace_id", "status", "updated_at"),
    )

    id: Mapped[str] = id_column()
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="draft")
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source_run_id: Mapped[str | None] = mapped_column(String(32))
    #: A5. A ``script`` step is arbitrary JavaScript running inside a session
    #: that may hold someone else's credentials, so enabling it is a privileged
    #: act recorded on the resource -- not a per-request flag a caller can set.
    scripts_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    scripts_enabled_by: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    scripts_enabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at_column()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )


class UseCaseVersion(Base):
    """Immutable. An edit appends a row, so a batch already running cannot have
    its definition changed underneath it."""

    __tablename__ = "usecase_versions"

    usecase_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("usecases.id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    definition: Mapped[dict] = _json_column()
    created_at: Mapped[datetime] = created_at_column()
    #: Free text describing provenance ("distilled", "repair v3"). Kept
    #: alongside the structured actor below rather than replaced by it.
    created_by: Mapped[str | None] = mapped_column(String(100))
    created_by_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class Credential(Base):
    """Fernet ciphertext. No code path returns ``ciphertext`` over HTTP.

    The uniqueness constraint is per workspace, not global. Globally unique
    names were the single clearest proof that the old schema could not hold two
    users: they could not both have a credential called "IXL account".
    """

    __tablename__ = "credentials"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_credentials_workspace_id_name"),
    )

    id: Mapped[str] = id_column()
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slots: Mapped[list] = _json_column(default=list)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = created_at_column()
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ---------------------------------------------------------------------------
# Batches and executions
# ---------------------------------------------------------------------------


class Batch(Base):
    __tablename__ = "batches"
    __table_args__ = (
        Index("ix_batches_workspace_usecase_created", "workspace_id", "usecase_id", "created_at"),
    )

    id: Mapped[str] = id_column()
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    #: Denormalized, like ``audit_log.actor_email`` and for the same reason: a
    #: record of who did something has to stay readable after the account is
    #: deleted, and ``owner_id`` is ON DELETE SET NULL. An id alone is also
    #: unreadable in a UI without a join on every row.
    #: ``server_default`` as well as ``default``: the column was added to
    #: populated tables, so the database needs a value for the rows that
    #: predate it. Blank means "nobody recorded who", which is the truth about
    #: those rows rather than a placeholder pretending otherwise.
    owner_email: Mapped[str] = mapped_column(
        String(320), nullable=False, default="", server_default=""
    )
    usecase_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("usecases.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    succeeded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    credential_id: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_column()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Execution(Base):
    __tablename__ = "executions"
    __table_args__ = (
        Index("ix_executions_batch_row", "batch_id", "row_index"),
        Index("ix_executions_workspace_usecase_created", "workspace_id", "usecase_id", "created_at"),
    )

    id: Mapped[str] = id_column()
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    #: Who ran it. A single-row execution recorded nobody at all before this,
    #: so "who ran that record and with what" had no answer.
    owner_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    #: Denormalized, like ``audit_log.actor_email`` and for the same reason: a
    #: record of who did something has to stay readable after the account is
    #: deleted, and ``owner_id`` is ON DELETE SET NULL. An id alone is also
    #: unreadable in a UI without a join on every row.
    #: ``server_default`` as well as ``default``: the column was added to
    #: populated tables, so the database needs a value for the rows that
    #: predate it. Blank means "nobody recorded who", which is the truth about
    #: those rows rather than a placeholder pretending otherwise.
    owner_email: Mapped[str] = mapped_column(
        String(320), nullable=False, default="", server_default=""
    )
    batch_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("batches.id", ondelete="CASCADE")
    )
    usecase_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("usecases.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(32))
    row_index: Mapped[int | None] = mapped_column(Integer)
    inputs: Mapped[dict] = _json_column()
    outputs: Mapped[dict | None] = _json_column(default=None, nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    failed_step_id: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = created_at_column()


# ---------------------------------------------------------------------------
# Work queue
# ---------------------------------------------------------------------------


class Job(Base):
    """A unit of work claimed with ``FOR UPDATE SKIP LOCKED``.

    This replaces the in-process dicts (``ReplayManager._slot``,
    ``RunManager._tasks``) that survived neither a restart nor a second worker.
    Postgres is the broker: it is already here, it is transactional with the
    rows the job is about, and a claim that crashes is released by a lease
    expiry rather than lost.

    ``lease_expires_at`` rather than a lock flag, because the failure that
    matters is a worker dying mid-job. A flag would strand the row forever; a
    lease is reclaimed by the next sweeper.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_claimable", "status", "run_after"),
        Index("ix_jobs_workspace_created", "workspace_id", "created_at"),
        UniqueConstraint("dedupe_key", name="uq_jobs_dedupe_key"),
        CheckConstraint(
            "status in ('queued','running','succeeded','failed','cancelled')",
            name="status_valid",
        ),
    )

    id: Mapped[str] = id_column()
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    payload: Mapped[dict] = _json_column()
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Lets a caller say "this job already exists" without a race. NULL means
    #: no deduplication, and Postgres treats every NULL as distinct.
    dedupe_key: Mapped[str | None] = mapped_column(String(200))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    run_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    claimed_by: Mapped[str | None] = mapped_column(String(100))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_column()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


__all__ = [
    "Artifact",
    "AuditLogEntry",
    "Batch",
    "Credential",
    "Event",
    "Execution",
    "Job",
    "Run",
    "UseCase",
    "UseCaseVersion",
    "User",
    "UserSession",
    "Workspace",
    "new_id",
]
