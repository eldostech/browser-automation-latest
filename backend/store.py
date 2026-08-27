"""Persistence for runs, events, use cases, credentials and executions.

Postgres via SQLAlchemy 2.0. History has to survive a restart and be visible to
every worker at once, which is what ruled out the SQLite file this used to be:
a second process could not see the first one's rows, and a job queue needs
``SELECT ... FOR UPDATE SKIP LOCKED``, which SQLite has no answer for.

Screenshots stay on the filesystem with only a reference in the database.
Binary blobs would bloat the tables the WebSocket replay reads from.

**How tenancy is enforced.** Scoped operations do not live on :class:`Store`.
They live on :class:`WorkspaceStore`, which you obtain with
``store.workspace(workspace_id)``, and which puts ``workspace_id`` into the
WHERE clause of every statement it issues. The point is that forgetting the
filter is not possible: there is no method on the scoped object that can reach
another tenant's row, so a missing check cannot leak data -- it can only fail
to compile. The handful of genuinely cross-tenant operations (startup reaping,
health probes, workspace creation) stay on the unscoped :class:`Store` where
they are easy to enumerate and review.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from config import Settings
from db.base import iso, utcnow
from db.engine import create_engine, create_session_factory, ensure_schema
from db.models import (
    Artifact,
    AuditLogEntry,
    Batch,
    Credential,
    Event,
    Execution,
    Run,
    UseCase,
    UseCaseVersion,
    Workspace,
)
from events import AgentEvent, RunFinished, RunStatus, dump_event, parse_event
from storage import ArtifactStorage, LocalStorage, artifact_key, storage_for

log = logging.getLogger(__name__)

ORPHAN_MESSAGE = "Backend restarted while this run was in flight."

#: Statuses that mean "this run believed it was still going".
UNFINISHED_STATUSES = ("pending", "running", "awaiting_approval")


@dataclass(slots=True)
class RunRecord:
    id: str
    task: str
    start_url: str | None
    status: RunStatus
    options: dict[str, Any]
    created_at: str
    started_at: str | None
    finished_at: str | None
    steps: int
    duration_ms: int | None
    summary: str | None
    result: dict[str, Any] | None
    error: str | None
    owner_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "start_url": self.start_url,
            "status": self.status,
            "options": self.options,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": self.steps,
            "duration_ms": self.duration_ms,
            "summary": self.summary,
            "result": self.result,
            "error": self.error,
            "owner_id": self.owner_id,
        }


@dataclass(slots=True)
class ArtifactRecord:
    id: str
    run_id: str
    seq: int | None
    kind: str
    mime: str
    path: str
    bytes: int
    created_at: str


def _run_record(row: Run) -> RunRecord:
    return RunRecord(
        id=row.id,
        task=row.task,
        start_url=row.start_url,
        status=row.status,  # type: ignore[arg-type]
        options=row.options or {},
        created_at=iso(row.created_at) or "",
        started_at=iso(row.started_at),
        finished_at=iso(row.finished_at),
        steps=row.steps or 0,
        duration_ms=row.duration_ms,
        summary=row.summary,
        result=row.result,
        error=row.error,
        owner_id=row.owner_id,
    )


def _artifact_record(row: Artifact) -> ArtifactRecord:
    return ArtifactRecord(
        id=row.id,
        run_id=row.run_id,
        seq=row.seq,
        kind=row.kind,
        mime=row.mime,
        path=row.path,
        bytes=row.bytes,
        created_at=iso(row.created_at) or "",
    )


def _execution_dict(row: Execution) -> dict[str, Any]:
    return {
        "id": row.id,
        "batch_id": row.batch_id,
        "usecase_id": row.usecase_id,
        "version": row.version,
        "run_id": row.run_id,
        "row_index": row.row_index,
        "inputs": row.inputs or {},
        "outputs": row.outputs,
        "status": row.status,
        "failed_step_id": row.failed_step_id,
        "error": row.error,
        "llm_calls": row.llm_calls,
        "llm_tokens": row.llm_tokens,
        "duration_ms": row.duration_ms,
        "created_at": iso(row.created_at),
    }


def _batch_dict(row: Batch) -> dict[str, Any]:
    return {
        "id": row.id,
        "usecase_id": row.usecase_id,
        "version": row.version,
        "status": row.status,
        "total": row.total,
        "succeeded": row.succeeded,
        "failed": row.failed,
        "credential_id": row.credential_id,
        "error": row.error,
        "created_at": iso(row.created_at),
        "finished_at": iso(row.finished_at),
        "owner_id": row.owner_id,
    }


class Store:
    """Connection lifecycle, and the few operations that cross tenants."""

    def __init__(
        self,
        settings: Settings,
        artifacts_dir: Path | None = None,
        storage: ArtifactStorage | None = None,
    ) -> None:
        self._settings = settings
        self.artifacts_dir = artifacts_dir or settings.artifacts_path
        #: Built here when not supplied, so a Store constructed in a test or a
        #: script gets working artifact storage without extra wiring.
        self.storage = storage or LocalStorage(self.artifacts_dir)
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[AsyncSession] | None = None

    # -- lifecycle ----------------------------------------------------------
    async def connect(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(self._settings)
        self._sessions = create_session_factory(self._engine)
        await ensure_schema(self._engine, self._settings.db_schema)
        log.info(
            "store connected",
            extra={
                "db_host": self._settings.db_host,
                "db_name": self._settings.db_name,
                "db_schema": self._settings.db_schema,
            },
        )

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessions = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Store.connect() has not been awaited")
        return self._engine

    @property
    def sessions(self) -> async_sessionmaker[AsyncSession]:
        if self._sessions is None:
            raise RuntimeError("Store.connect() has not been awaited")
        return self._sessions

    def workspace(self, workspace_id: str) -> "WorkspaceStore":
        """A view of this store confined to one tenant.

        Cheap -- it holds a session factory and a string -- so callers make one
        per request or per operation rather than caching it.
        """
        return WorkspaceStore(
            self.sessions,
            self.artifacts_dir,
            workspace_id,
            storage=self.storage,
            settings=self._settings,
        )

    async def ping(self) -> bool:
        try:
            async with self.sessions() as session:
                await session.execute(select(1))
            return True
        except Exception:  # noqa: BLE001 - a health probe must never raise
            return False

    # -- workspaces ---------------------------------------------------------
    async def ensure_workspace(self, name: str, slug: str) -> str:
        async with self.sessions() as session:
            existing = await session.scalar(select(Workspace).where(Workspace.slug == slug))
            if existing is not None:
                return existing.id
            workspace = Workspace(name=name, slug=slug)
            session.add(workspace)
            await session.commit()
            return workspace.id

    async def default_workspace_id(self) -> str | None:
        async with self.sessions() as session:
            return await session.scalar(
                select(Workspace.id).order_by(Workspace.created_at).limit(1)
            )

    async def list_workspaces(self) -> list[dict[str, Any]]:
        async with self.sessions() as session:
            rows = (await session.scalars(select(Workspace).order_by(Workspace.created_at))).all()
            return [
                {"id": w.id, "name": w.name, "slug": w.slug, "created_at": iso(w.created_at)}
                for w in rows
            ]

    # -- startup recovery ---------------------------------------------------
    async def reap_orphaned_runs(self) -> int:
        """Fail runs left mid-flight by a crash or restart, across all tenants.

        Deliberately unscoped: it runs at startup, before any request has
        established who is calling, and a tenant filter here would leave other
        workspaces' runs spinning forever in the UI.

        Note that this *fails* orphans rather than resuming them. Resumption is
        the job of the queue, which re-leases work whose lease expired; this
        covers runs that were never queue-managed.
        """
        async with self.sessions() as session:
            orphans = (
                await session.scalars(
                    select(Run.id).where(Run.status.in_(UNFINISHED_STATUSES))
                )
            ).all()
            if not orphans:
                return 0

            for run_id in orphans:
                next_seq = (
                    await session.scalar(
                        select(func.coalesce(func.max(Event.seq), 0)).where(
                            Event.run_id == run_id
                        )
                    )
                ) + 1
                event = RunFinished(
                    run_id=run_id,
                    seq=next_seq,
                    status="failed",
                    steps=0,
                    duration_ms=0,
                    error=ORPHAN_MESSAGE,
                )
                payload = dump_event(event)
                session.add(
                    Event(
                        run_id=run_id,
                        seq=next_seq,
                        type=payload["type"],
                        payload=payload,
                    )
                )

            await session.execute(
                update(Run)
                .where(Run.status.in_(UNFINISHED_STATUSES))
                .values(status="failed", finished_at=utcnow(), error=ORPHAN_MESSAGE)
            )
            await session.commit()

        log.warning("reaped orphaned runs", extra={"count": len(orphans)})
        return len(orphans)


class WorkspaceStore:
    """Every operation, confined to one workspace.

    Read the WHERE clauses as a set: each one carries
    ``workspace_id == self._ws``. That repetition is the security property, not
    boilerplate to factor away -- the moment it becomes implicit is the moment a
    new method can forget it.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        artifacts_dir: Path,
        workspace_id: str,
        storage: ArtifactStorage | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._sessions = sessions
        self.artifacts_dir = artifacts_dir
        self._ws = workspace_id
        self._settings = settings
        #: Defaults to the local filesystem, so a WorkspaceStore built without
        #: one behaves exactly as it did before storage was pluggable.
        self._storage = storage or LocalStorage(artifacts_dir)

    @property
    def workspace_id(self) -> str:
        return self._ws

    # -- runs ---------------------------------------------------------------
    async def create_run(
        self,
        run_id: str,
        task: str,
        start_url: str | None,
        options: dict[str, Any],
        *,
        owner_id: str | None = None,
    ) -> RunRecord:
        async with self._sessions() as session:
            run = Run(
                id=run_id,
                workspace_id=self._ws,
                owner_id=owner_id,
                task=task,
                start_url=start_url,
                status="pending",
                options=options or {},
            )
            session.add(run)
            await session.commit()
            return _run_record(run)

    async def mark_started(self, run_id: str) -> None:
        await self._update_run(run_id, status="running", started_at=utcnow())

    async def set_status(self, run_id: str, status: RunStatus) -> None:
        await self._update_run(run_id, status=status)

    async def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        steps: int,
        duration_ms: int,
        summary: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        await self._update_run(
            run_id,
            status=status,
            finished_at=utcnow(),
            steps=steps,
            duration_ms=duration_ms,
            summary=summary,
            result=result,
            error=error,
        )

    async def _update_run(self, run_id: str, **values: Any) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(Run)
                .where(Run.id == run_id, Run.workspace_id == self._ws)
                .values(**values)
            )
            await session.commit()

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(Run).where(Run.id == run_id, Run.workspace_id == self._ws)
            )
            return _run_record(row) if row else None

    async def list_runs(
        self, *, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[RunRecord]:
        async with self._sessions() as session:
            stmt = select(Run).where(Run.workspace_id == self._ws)
            if status:
                stmt = stmt.where(Run.status == status)
            stmt = stmt.order_by(Run.created_at.desc()).limit(limit).offset(offset)
            return [_run_record(r) for r in (await session.scalars(stmt)).all()]

    async def count_runs(self, status: str | None = None) -> int:
        async with self._sessions() as session:
            stmt = select(func.count()).select_from(Run).where(Run.workspace_id == self._ws)
            if status:
                stmt = stmt.where(Run.status == status)
            return int(await session.scalar(stmt) or 0)

    # -- events -------------------------------------------------------------
    async def append_event(self, event: AgentEvent) -> None:
        """Insert an event, or replace it if that ``seq`` already exists.

        Replacement is what makes streaming ``thinking`` events work: the agent
        reserves one ``seq`` per prose block and rewrites it as text arrives,
        so replay after a reconnect yields the final text exactly once.
        """
        payload = dump_event(event)
        stmt = pg_insert(Event).values(
            run_id=event.run_id,
            seq=event.seq,
            ts=utcnow(),
            type=payload["type"],
            payload=payload,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[Event.run_id, Event.seq],
            set_={"ts": stmt.excluded.ts, "payload": stmt.excluded.payload},
        )
        async with self._sessions() as session:
            await session.execute(stmt)
            await session.commit()

    async def get_events(
        self, run_id: str, after_seq: int = 0, limit: int = 5000
    ) -> list[AgentEvent]:
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(Event)
                    .join(Run, Run.id == Event.run_id)
                    .where(
                        Event.run_id == run_id,
                        Event.seq > after_seq,
                        Run.workspace_id == self._ws,
                    )
                    .order_by(Event.seq.asc())
                    .limit(limit)
                )
            ).all()
            return [parse_event(row.payload) for row in rows]

    async def next_seq(self, run_id: str) -> int:
        async with self._sessions() as session:
            highest = await session.scalar(
                select(func.coalesce(func.max(Event.seq), 0)).where(Event.run_id == run_id)
            )
            return int(highest or 0) + 1

    # -- use cases ----------------------------------------------------------
    async def save_usecase(
        self,
        definition: dict[str, Any],
        *,
        created_by: str | None = None,
        created_by_id: str | None = None,
        owner_id: str | None = None,
    ) -> tuple[str, int]:
        """Insert a use case, or append a new version of an existing one.

        Returns ``(usecase_id, version)``. Versions are append-only: an edit
        never rewrites the row a running batch is reading from.
        """
        usecase_id = str(definition["id"])
        name = str(definition.get("name") or "Untitled")
        description = str(definition.get("description") or "")
        status = str(definition.get("status") or "draft")
        now = utcnow()

        async with self._sessions() as session:
            # Scoped: appending a version to another tenant's use case would
            # otherwise be possible by supplying its id.
            owner_ws = await session.scalar(
                select(UseCase.workspace_id).where(UseCase.id == usecase_id)
            )
            if owner_ws is not None and owner_ws != self._ws:
                raise PermissionError("That use case belongs to another workspace.")

            latest = await session.scalar(
                select(func.coalesce(func.max(UseCaseVersion.version), 0)).where(
                    UseCaseVersion.usecase_id == usecase_id
                )
            )
            version = int(latest or 0) + 1
            stored = {**definition, "version": version, "updated_at": iso(now)}

            upsert = pg_insert(UseCase).values(
                id=usecase_id,
                workspace_id=self._ws,
                owner_id=owner_id,
                name=name,
                description=description,
                status=status,
                current_version=version,
                source_run_id=definition.get("source_run_id"),
                created_at=now,
                updated_at=now,
            )
            await session.execute(
                upsert.on_conflict_do_update(
                    index_elements=[UseCase.id],
                    set_={
                        "name": upsert.excluded.name,
                        "description": upsert.excluded.description,
                        "status": upsert.excluded.status,
                        "current_version": upsert.excluded.current_version,
                        "updated_at": upsert.excluded.updated_at,
                    },
                )
            )
            session.add(
                UseCaseVersion(
                    usecase_id=usecase_id,
                    version=version,
                    definition=stored,
                    created_at=now,
                    created_by=created_by,
                    created_by_id=created_by_id,
                )
            )
            await session.commit()
            return usecase_id, version

    async def get_usecase(
        self, usecase_id: str, version: int | None = None
    ) -> dict[str, Any] | None:
        """One stored definition. ``version=None`` means the current one."""
        async with self._sessions() as session:
            if version is None:
                version = await session.scalar(
                    select(UseCase.current_version).where(
                        UseCase.id == usecase_id, UseCase.workspace_id == self._ws
                    )
                )
                if version is None:
                    return None
            else:
                # A caller naming an explicit version still may not read
                # across the tenant boundary.
                exists = await session.scalar(
                    select(UseCase.id).where(
                        UseCase.id == usecase_id, UseCase.workspace_id == self._ws
                    )
                )
                if exists is None:
                    return None

            return await session.scalar(
                select(UseCaseVersion.definition).where(
                    UseCaseVersion.usecase_id == usecase_id,
                    UseCaseVersion.version == version,
                )
            )

    async def get_usecase_row(self, usecase_id: str) -> dict[str, Any] | None:
        """The summary row, including the script-permission fields."""
        async with self._sessions() as session:
            row = await session.scalar(
                select(UseCase).where(
                    UseCase.id == usecase_id, UseCase.workspace_id == self._ws
                )
            )
            return _usecase_dict(row) if row else None

    async def list_usecases(
        self, *, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Summary rows for the list view -- not the full definitions."""
        async with self._sessions() as session:
            stmt = select(UseCase).where(UseCase.workspace_id == self._ws)
            if status:
                stmt = stmt.where(UseCase.status == status)
            stmt = stmt.order_by(UseCase.updated_at.desc()).limit(limit).offset(offset)
            return [_usecase_dict(row) for row in (await session.scalars(stmt)).all()]

    async def list_usecase_versions(self, usecase_id: str) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(
                        UseCaseVersion.version,
                        UseCaseVersion.created_at,
                        UseCaseVersion.created_by,
                    )
                    .join(UseCase, UseCase.id == UseCaseVersion.usecase_id)
                    .where(
                        UseCaseVersion.usecase_id == usecase_id,
                        UseCase.workspace_id == self._ws,
                    )
                    .order_by(UseCaseVersion.version.desc())
                )
            ).all()
            return [
                {"version": v, "created_at": iso(c), "created_by": by} for v, c, by in rows
            ]

    async def set_usecase_status(self, usecase_id: str, status: str) -> bool:
        """Move a use case between draft / ready / archived.

        Also rewrites the status inside the current stored definition, so a
        definition read back on its own still reports the truth.
        """
        async with self._sessions() as session:
            usecase = await session.scalar(
                select(UseCase).where(
                    UseCase.id == usecase_id, UseCase.workspace_id == self._ws
                )
            )
            if usecase is None:
                return False

            version_row = await session.scalar(
                select(UseCaseVersion).where(
                    UseCaseVersion.usecase_id == usecase_id,
                    UseCaseVersion.version == usecase.current_version,
                )
            )
            if version_row is not None:
                # JSONB columns are only marked dirty on rebind, so replace the
                # dict rather than mutating it in place.
                version_row.definition = {**(version_row.definition or {}), "status": status}

            usecase.status = status
            usecase.updated_at = utcnow()
            await session.commit()
            return True

    async def rename_usecase(
        self, usecase_id: str, name: str, description: str | None = None
    ) -> bool:
        """Change the label, in place, without creating a version.

        Versions exist so a running batch cannot have its *recipe* changed
        underneath it. A name is not part of the recipe -- nothing executes
        differently because of it -- so renaming appends no version and the
        history stays a record of behaviour rather than of typos.
        """
        async with self._sessions() as session:
            usecase = await session.scalar(
                select(UseCase).where(
                    UseCase.id == usecase_id, UseCase.workspace_id == self._ws
                )
            )
            if usecase is None:
                return False

            version_row = await session.scalar(
                select(UseCaseVersion).where(
                    UseCaseVersion.usecase_id == usecase_id,
                    UseCaseVersion.version == usecase.current_version,
                )
            )
            if version_row is not None:
                definition = {**(version_row.definition or {}), "name": name}
                if description is not None:
                    definition["description"] = description
                version_row.definition = definition
                usecase.description = definition.get("description", "")

            usecase.name = name
            usecase.updated_at = utcnow()
            await session.commit()
            return True

    async def set_scripts_enabled(
        self, usecase_id: str, enabled: bool, *, actor_id: str | None
    ) -> bool:
        """Permit (or withdraw permission for) script steps on a use case.

        Recorded on the resource with who and when, because this is the one
        setting that lets a use case run arbitrary JavaScript inside a session
        holding somebody else's credentials.
        """
        async with self._sessions() as session:
            usecase = await session.scalar(
                select(UseCase).where(
                    UseCase.id == usecase_id, UseCase.workspace_id == self._ws
                )
            )
            if usecase is None:
                return False
            usecase.scripts_enabled = enabled
            usecase.scripts_enabled_by = actor_id if enabled else None
            usecase.scripts_enabled_at = utcnow() if enabled else None
            usecase.updated_at = utcnow()
            await session.commit()
            return True

    async def delete_usecase(self, usecase_id: str) -> bool:
        """Archive: hide it from the active list but keep every reference intact."""
        return await self.set_usecase_status(usecase_id, "archived")

    async def purge_usecase(self, usecase_id: str) -> dict[str, int] | None:
        """Delete a use case, its versions, and its execution records for good.

        The runs and events those executions produced are deliberately left
        alone: they are the timeline of things that actually happened to a
        browser, and they stay meaningful -- and auditable -- after the recipe
        that caused them is gone.

        Returns the row counts removed, or ``None`` if there was no such use
        case in this workspace.
        """
        async with self._sessions() as session:
            exists = await session.scalar(
                select(UseCase.id).where(
                    UseCase.id == usecase_id, UseCase.workspace_id == self._ws
                )
            )
            if exists is None:
                return None

            removed: dict[str, int] = {}
            for table, model, column in (
                ("executions", Execution, Execution.usecase_id),
                ("batches", Batch, Batch.usecase_id),
                ("usecase_versions", UseCaseVersion, UseCaseVersion.usecase_id),
            ):
                result = await session.execute(delete(model).where(column == usecase_id))
                removed[table] = int(result.rowcount or 0)
            result = await session.execute(
                delete(UseCase).where(
                    UseCase.id == usecase_id, UseCase.workspace_id == self._ws
                )
            )
            removed["usecases"] = int(result.rowcount or 0)
            await session.commit()

        log.info("purged a use case", extra={"usecase_id": usecase_id, **removed})
        return removed

    # -- credentials --------------------------------------------------------
    async def save_credential(
        self,
        credential_id: str,
        name: str,
        slots: list[str],
        ciphertext: bytes,
        *,
        owner_id: str | None = None,
    ) -> str:
        """Store an encrypted credential bundle. Re-saving a name replaces it.

        Only ciphertext lands here; see ``credentials.py`` for why there is no
        method to read a value back out over HTTP.
        """
        async with self._sessions() as session:
            stmt = pg_insert(Credential).values(
                id=credential_id,
                workspace_id=self._ws,
                owner_id=owner_id,
                name=name,
                slots=slots,
                ciphertext=ciphertext,
                created_at=utcnow(),
            )
            stmt = stmt.on_conflict_do_update(
                # Per workspace, so two tenants may both have "IXL account".
                index_elements=[Credential.workspace_id, Credential.name],
                set_={"slots": stmt.excluded.slots, "ciphertext": stmt.excluded.ciphertext},
            ).returning(Credential.id)
            stored_id = await session.scalar(stmt)
            await session.commit()
            return stored_id or credential_id

    async def get_credential_ciphertext(self, credential_id: str) -> bytes | None:
        async with self._sessions() as session:
            return await session.scalar(
                select(Credential.ciphertext).where(
                    Credential.id == credential_id, Credential.workspace_id == self._ws
                )
            )

    async def list_credentials(self) -> list[dict[str, Any]]:
        """Names and slot lists only -- never a value."""
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(Credential)
                    .where(Credential.workspace_id == self._ws)
                    .order_by(Credential.name)
                )
            ).all()
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "slots": row.slots or [],
                    "created_at": iso(row.created_at),
                    "last_used_at": iso(row.last_used_at),
                }
                for row in rows
            ]

    async def touch_credential(self, credential_id: str) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(Credential)
                .where(Credential.id == credential_id, Credential.workspace_id == self._ws)
                .values(last_used_at=utcnow())
            )
            await session.commit()

    async def delete_credential(self, credential_id: str) -> bool:
        async with self._sessions() as session:
            result = await session.execute(
                delete(Credential).where(
                    Credential.id == credential_id, Credential.workspace_id == self._ws
                )
            )
            await session.commit()
            return bool(result.rowcount)

    # -- executions ---------------------------------------------------------
    async def create_execution(
        self,
        execution_id: str,
        usecase_id: str,
        version: int,
        *,
        run_id: str | None = None,
        batch_id: str | None = None,
        row_index: int | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        async with self._sessions() as session:
            session.add(
                Execution(
                    id=execution_id,
                    workspace_id=self._ws,
                    batch_id=batch_id,
                    usecase_id=usecase_id,
                    version=version,
                    run_id=run_id,
                    row_index=row_index,
                    inputs=inputs or {},
                    status="pending",
                )
            )
            await session.commit()

    async def finish_execution(
        self,
        execution_id: str,
        status: str,
        *,
        outputs: dict[str, Any] | None = None,
        failed_step_id: str | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
        llm_calls: int = 0,
        llm_tokens: int = 0,
    ) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(Execution)
                .where(Execution.id == execution_id, Execution.workspace_id == self._ws)
                .values(
                    status=status,
                    outputs=outputs,
                    failed_step_id=failed_step_id,
                    error=error,
                    duration_ms=duration_ms,
                    llm_calls=llm_calls,
                    llm_tokens=llm_tokens,
                )
            )
            await session.commit()

    async def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(Execution).where(
                    Execution.id == execution_id, Execution.workspace_id == self._ws
                )
            )
            return _execution_dict(row) if row else None

    async def list_executions(
        self, *, batch_id: str | None = None, usecase_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            stmt = select(Execution).where(Execution.workspace_id == self._ws)
            if batch_id:
                stmt = stmt.where(Execution.batch_id == batch_id)
            if usecase_id:
                stmt = stmt.where(Execution.usecase_id == usecase_id)
            stmt = stmt.order_by(
                func.coalesce(Execution.row_index, 0).asc(), Execution.created_at.asc()
            ).limit(limit)
            return [_execution_dict(row) for row in (await session.scalars(stmt)).all()]

    # -- batches ------------------------------------------------------------
    async def create_batch(
        self,
        batch_id: str,
        usecase_id: str,
        version: int,
        *,
        total: int,
        credential_id: str | None = None,
        owner_id: str | None = None,
    ) -> None:
        async with self._sessions() as session:
            session.add(
                Batch(
                    id=batch_id,
                    workspace_id=self._ws,
                    owner_id=owner_id,
                    usecase_id=usecase_id,
                    version=version,
                    status="pending",
                    total=total,
                    credential_id=credential_id,
                )
            )
            await session.commit()

    async def update_batch(
        self,
        batch_id: str,
        *,
        status: str | None = None,
        succeeded: int | None = None,
        failed: int | None = None,
        error: str | None = None,
        finished: bool = False,
    ) -> None:
        values: dict[str, Any] = {
            column: value
            for column, value in (
                ("status", status),
                ("succeeded", succeeded),
                ("failed", failed),
                ("error", error),
            )
            if value is not None
        }
        if finished:
            values["finished_at"] = utcnow()
        if not values:
            return

        async with self._sessions() as session:
            await session.execute(
                update(Batch)
                .where(Batch.id == batch_id, Batch.workspace_id == self._ws)
                .values(**values)
            )
            await session.commit()

    async def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(Batch).where(Batch.id == batch_id, Batch.workspace_id == self._ws)
            )
            return _batch_dict(row) if row else None

    async def list_batches(
        self, *, usecase_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            stmt = select(Batch).where(Batch.workspace_id == self._ws)
            if usecase_id:
                stmt = stmt.where(Batch.usecase_id == usecase_id)
            stmt = stmt.order_by(Batch.created_at.desc()).limit(limit)
            return [_batch_dict(row) for row in (await session.scalars(stmt)).all()]

    # -- artifacts ----------------------------------------------------------
    async def save_artifact(
        self,
        run_id: str,
        data: bytes,
        *,
        kind: str = "screenshot",
        mime: str = "image/png",
        seq: int | None = None,
        suffix: str = ".png",
    ) -> ArtifactRecord:
        """Write the bytes, then record where they went.

        The ``path`` column now holds a *locator* -- a filesystem path or an
        ``s3://`` URL -- so a row says which backend wrote it. Reading follows
        the row rather than the current setting, which is what lets a
        deployment switch backends without orphaning what it already has.
        """
        artifact_id = uuid.uuid4().hex
        stored = await self._storage.put(
            artifact_key(run_id, artifact_id, suffix), data, content_type=mime
        )

        async with self._sessions() as session:
            row = Artifact(
                id=artifact_id,
                run_id=run_id,
                workspace_id=self._ws,
                seq=seq,
                kind=kind,
                mime=mime,
                path=stored.locator,
                bytes=stored.bytes,
            )
            session.add(row)
            await session.commit()
            return _artifact_record(row)

    async def read_artifact(self, record: ArtifactRecord) -> bytes:
        """The bytes behind a record, from whichever backend holds them."""
        backend = storage_for(record.path, self._storage, self._settings)
        return await backend.get(record.path)

    def artifact_url(self, record: ArtifactRecord) -> str | None:
        """A URL the browser can fetch directly, when the backend offers one."""
        backend = storage_for(record.path, self._storage, self._settings)
        return backend.presigned_url(record.path)

    async def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(Artifact).where(
                    Artifact.id == artifact_id, Artifact.workspace_id == self._ws
                )
            )
            return _artifact_record(row) if row else None

    async def list_artifacts(self, run_id: str) -> Sequence[ArtifactRecord]:
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(Artifact)
                    .where(Artifact.run_id == run_id, Artifact.workspace_id == self._ws)
                    .order_by(Artifact.created_at.asc())
                )
            ).all()
            return [_artifact_record(row) for row in rows]

    # -- audit --------------------------------------------------------------
    async def audit(
        self,
        action: str,
        *,
        actor_id: str | None,
        actor_email: str = "",
        resource_type: str = "",
        resource_id: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Record something a person did.

        Never raises. An audit write that fails must not take down the
        operation it was describing -- losing one line of the trail is bad, but
        rolling back a completed publish because the log was unavailable is
        worse, and would make the log a single point of failure for the whole
        application.
        """
        try:
            async with self._sessions() as session:
                session.add(
                    AuditLogEntry(
                        workspace_id=self._ws,
                        actor_id=actor_id,
                        actor_email=actor_email,
                        action=action,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        detail=detail or {},
                    )
                )
                await session.commit()
        except Exception:  # noqa: BLE001 - see docstring
            log.exception("failed to write an audit entry", extra={"action": action})

    async def list_audit(
        self,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            stmt = select(AuditLogEntry).where(AuditLogEntry.workspace_id == self._ws)
            if resource_type:
                stmt = stmt.where(AuditLogEntry.resource_type == resource_type)
            if resource_id:
                stmt = stmt.where(AuditLogEntry.resource_id == resource_id)
            stmt = stmt.order_by(AuditLogEntry.created_at.desc()).limit(limit)
            rows = (await session.scalars(stmt)).all()
            return [
                {
                    "id": row.id,
                    "actor_id": row.actor_id,
                    "actor_email": row.actor_email,
                    "action": row.action,
                    "resource_type": row.resource_type,
                    "resource_id": row.resource_id,
                    "detail": row.detail or {},
                    "created_at": iso(row.created_at),
                }
                for row in rows
            ]


def _usecase_dict(row: UseCase) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "status": row.status,
        "current_version": row.current_version,
        "source_run_id": row.source_run_id,
        "scripts_enabled": row.scripts_enabled,
        "scripts_enabled_by": row.scripts_enabled_by,
        "scripts_enabled_at": iso(row.scripts_enabled_at),
        "owner_id": row.owner_id,
        "created_at": iso(row.created_at),
        "updated_at": iso(row.updated_at),
    }


__all__ = [
    "ArtifactRecord",
    "ORPHAN_MESSAGE",
    "RunRecord",
    "Store",
    "WorkspaceStore",
]
