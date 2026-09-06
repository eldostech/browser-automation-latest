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
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base, created_at_column, id_column, new_id, utcnow

#: Titan Text Embeddings V2's default width. A vector column is fixed, so
#: changing the embedding model is a migration.
EMBEDDING_DIMENSIONS = 1024


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
    #: What this workspace may spend with a model in a calendar month, or NULL
    #: for no ceiling. Nullable rather than a large default because "unlimited"
    #: and "limited to a number somebody picked" are different states, and an
    #: install that has never thought about it should not be told it has a
    #: budget it did not set.
    #:
    #: A guard rail, not an accounting control: it is enforced against this
    #: application's own estimate of what a turn costs, and the AWS bill is the
    #: authority. See pricing.py.
    monthly_spend_limit_usd: Mapped[float | None] = mapped_column(Float)
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
    #: What this run spent with a model. Columns rather than fields inside
    #: ``result`` because the question they exist to answer -- what has this
    #: workspace spent this month -- is a SUM, and a SUM over JSONB is a
    #: question nobody asks twice.
    #:
    #: Zero is the ordinary value and it is *measured*, not assumed: a Strict
    #: replay cannot reach a model at all, and saying so with a number is what
    #: lets a dashboard tell "free" from "not recorded".
    tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    cost_usd: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, server_default="0"
    )


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


class RunStep(Base):
    """One step of one execution, as a row rather than as a log line.

    ``events`` already carries this. It is a *log*, though, and the model
    docstring above is emphatic that it is not the system of record -- keeping
    that distinction is what stops this project drifting into event sourcing it
    does not need. Two screens want the same thing as an indexed query instead
    of a JSON scan:

    * the run timeline, asking "every step of this execution with its status,
      duration and screenshot";
    * the visual diff, asking "this step, and the same ``step_id`` from the
      baseline run", which is a join the event log cannot express.

    ``baseline_artifact_id`` and ``pixel_diff`` are written when a baseline
    exists, so the comparison is computed once on write rather than on every
    render of the timeline.
    """

    __tablename__ = "run_steps"
    __table_args__ = (
        Index("ix_run_steps_run_seq", "run_id", "seq"),
        Index("ix_run_steps_usecase_step", "usecase_id", "version", "step_id"),
    )

    id: Mapped[str] = id_column()
    run_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    #: Which recipe this step came from, so a baseline can be found without
    #: walking back through runs.
    usecase_id: Mapped[str | None] = mapped_column(String(32))
    version: Mapped[int | None] = mapped_column(Integer)
    #: Which row of a batch, when it was one.
    row_index: Mapped[int | None] = mapped_column(Integer)

    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    step_id: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False, default="row")
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    #: The rung that matched, rendered for a person.
    locator: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Which rung it was. Above zero means the recording is drifting.
    locator_rung: Mapped[int | None] = mapped_column(Integer)

    #: Where the step happened. The trail shows it, and the domain is what a
    #: fix recorded from here gets filed under.
    page_url: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    #: succeeded | failed | skipped | healed
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    screenshot_id: Mapped[str | None] = mapped_column(String(32))
    baseline_id: Mapped[str | None] = mapped_column(String(32))
    #: Fraction of pixels that differ from the baseline, 0.0 to 1.0. Null when
    #: there was no baseline to compare against.
    pixel_diff: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = created_at_column()


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
    #: The name the file arrived with. Empty for a screenshot, which is
    #: identified by its step; a downloaded document is identified by its name,
    #: and that name is what the next system expects on the way back in.
    filename: Mapped[str] = mapped_column(
        String(400), nullable=False, default="", server_default=""
    )
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
    #: Which :class:`Target` supplies this use case's base URL. Empty falls
    #: back to the URL recorded into the definition, which is what lets a
    #: single-environment install run with nothing configured.
    target: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    #: How much a model may do while this runs -- "strict", "guided", or empty
    #: for "whatever the deployment says". Mirrored from the definition, which
    #: stays the source of truth because the document has to carry it for
    #: promotion; the column exists so a list can show a mode chip and filter
    #: on it without reading every definition.
    #:
    #: Empty rather than a "strict" default on purpose: a use case published
    #: before this column existed has not chosen, and saying it chose strict
    #: would turn healing off underneath a deployment that has it on.
    mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default="", server_default=""
    )
    #: "person" or "agent". Which authoring path produced this document.
    authored_by: Mapped[str] = mapped_column(
        String(16), nullable=False, default="person", server_default="person"
    )
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


class HealingMemory(Base):
    """A locator that broke, and what fixed it.

    Written when a repair is confirmed -- by the model with high confidence, or
    by a person -- and read the next time something breaks on the same domain.
    That is the whole loop: the same site redesign costs one model call the
    first time a workflow hits it and none afterwards.

    Three rules govern reading it, and each is enforced somewhere different:

    **Scoped to a workspace**, by ``WorkspaceStore`` like everything else. A
    tenant must not be shown another tenant's selectors: they describe the
    shape of another company's internal tooling.

    **Filtered by domain before it is ranked.** Nearest-neighbour over every
    fix ever recorded will cheerfully return a plausible button from an
    unrelated site. ``domain`` is a hard ``WHERE`` and the vector distance only
    orders what survives it.

    **Never authoritative.** What comes back is context in a prompt. The model
    still chooses from the controls on the page *now*, and the choice is still
    validated against them -- so a stale or poisoned memory can make healing
    worse, but it cannot make it unsafe.
    """

    __tablename__ = "healing_memory"
    __table_args__ = (
        Index("ix_healing_memory_scope", "workspace_id", "domain", "created_at"),
        # Declared here as well as in the migration, or `alembic check` reports
        # the models and the schema as disagreeing on every run. Untuned on
        # purpose: this table holds tens to hundreds of rows per domain, where
        # a sequential scan is already fast.
        Index(
            "ix_healing_memory_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[str] = id_column()
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    usecase_id: Mapped[str | None] = mapped_column(String(32))
    #: The host the failure happened on. The hard filter on retrieval.
    domain: Mapped[str] = mapped_column(String(253), nullable=False, default="")
    step_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    #: not_found | assertion | value | other -- what kind of thing broke.
    error_kind: Mapped[str] = mapped_column(String(24), nullable=False, default="not_found")

    #: What was on the page when it broke, capped. This is what was embedded.
    dom_context: Mapped[str] = mapped_column(Text, nullable=False, default="")
    old_locator: Mapped[dict | None] = _json_column(default=None, nullable=True)
    new_locator: Mapped[dict | None] = _json_column(default=None, nullable=True)
    #: The plain-English account a person reads. "The Submit Order button is
    #: now labelled Confirm Purchase and sits inside a dialog."
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: "model" when it healed itself, or the email of whoever confirmed it. A
    #: fix a person approved is worth more than one nobody looked at, and this
    #: is what lets that distinction be made later.
    confirmed_by: Mapped[str] = mapped_column(String(320), nullable=False, default="")

    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=True)
    created_at: Mapped[datetime] = created_at_column()


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



class Target(Base):
    """A name, and the address it means in *this* deployment.

    The base URL a use case runs against used to come from a single map in the
    environment. That works while every workflow in an environment shares one
    site, and stops working the moment they do not: a second site means a
    second variable, set in the environment, needing a release to add.

    Holding it as data instead makes the common shapes cheap. Twenty use cases
    against one site share one row. Moving that site's UAT host is one edit,
    not twenty. Onboarding a new site is a row.

    Each deployment's own database holds its own URLs, so a use case promoted
    from dev to production carries no address with it -- it names a target, and
    each environment answers that name for itself. A name with no target here
    is refused at run time rather than guessed at: running the right workflow
    against the wrong site is the failure this whole arrangement exists to
    prevent.
    """

    __tablename__ = "targets"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_targets_workspace_name"),
        Index("ix_targets_workspace", "workspace_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str] = mapped_column(String(2000), nullable=False)
    description: Mapped[str] = mapped_column(
        String(300), nullable=False, default="", server_default=""
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_by: Mapped[str] = mapped_column(
        String(320), nullable=False, default="", server_default=""
    )


class Dataset(Base):
    """An uploaded file of input rows, parsed once and kept.

    Uploads used to be anonymous: the rows arrived in the request that started
    a batch, were used, and were gone. Making them a resource is what the
    mapping step needs -- you upload a file, look at what is in it, agree how
    its columns line up with a use case, and only then run something. That is
    three round trips over the same rows, and re-parsing the file on each one
    means holding it in the browser and posting it three times.

    It also makes the common case cheap: the same customer list is run against
    a workflow every week, and nothing about it changed.

    ``rows`` is the parsed content, ``columns`` the profile of each column that
    the mapper reasons over -- kind, blanks, distinct count, examples. Both are
    JSONB for the same reason ``batches.input_rows`` is: they are read whole,
    by one owner, and never joined on.
    """

    __tablename__ = "datasets"
    __table_args__ = (
        Index("ix_datasets_workspace_created", "workspace_id", "created_at"),
    )

    id: Mapped[str] = id_column()
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL")
    )
    owner_email: Mapped[str] = mapped_column(
        String(320), nullable=False, default="", server_default=""
    )
    #: What the user called it, defaulting to the filename they uploaded.
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    filename: Mapped[str] = mapped_column(String(400), nullable=False, default="")
    #: csv | xlsx | text | json -- how it was read, for the UI and for errors.
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="csv")
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    columns: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    rows: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    warnings: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = created_at_column()


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
    #: The address this batch actually ran against, resolved once when it was
    #: queued. Kept so a resume or a retry goes back to the same site: a target
    #: edited in between must not silently move the second half of a batch to a
    #: different deployment from the first.
    base_url: Mapped[str] = mapped_column(
        String(2000), nullable=False, default="", server_default=""
    )
    #: The rows this batch runs, snapshotted when it was queued.
    #:
    #: They live here rather than in the API process's memory because the work
    #: is claimed by a worker that may be another process on another machine,
    #: and because a resume after a restart has nowhere else to read them from.
    #: Reconstructing them from ``executions.inputs`` -- what resume used to do
    #: -- silently substitutes an empty row for anything that was never
    #: attempted, which is precisely the set of rows a resume exists to run.
    #:
    #: Secrets are never among them. Whoever runs the batch resolves those from
    #: the vault via ``credential_id``, so a password is not written to a table
    #: that a batch listing reads.
    input_rows: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    #: Which dataset this batch was started from, when it was started from
    #: one. Nullable because rows may still be posted directly, and because
    #: every batch that predates datasets was.
    dataset_id: Mapped[str | None] = mapped_column(String(32))
    #: The queue entry that will run, or is running, this batch. Written in the
    #: same transaction as the row itself, so a queued batch always has a job
    #: and a job always has its batch.
    job_id: Mapped[str | None] = mapped_column(String(32))
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
