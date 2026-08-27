"""A durable work queue on Postgres.

This replaces the in-process dicts (``ReplayManager._slot``,
``RunManager._tasks``) that survived neither a restart nor a second worker.

**Why Postgres and not a broker.** The work here is measured in minutes per
job, dozens per hour -- not thousands per second. At that rate the useful
property is not throughput, it is that enqueueing a job and writing the rows
the job is about happen in *one transaction*: a batch row and its job either
both exist or neither does, with no window where the UI shows a queued batch no
worker will ever run. A separate broker cannot offer that without an outbox
table, which is to say without reimplementing this.

The pattern is ``SELECT ... FOR UPDATE SKIP LOCKED``: each worker locks the
rows it claims, and other workers step over locked rows instead of blocking.

**Leases, not flags.** The failure that matters is a worker dying mid-job. A
boolean ``locked`` column strands that row forever; a lease expires and the
next sweep reclaims it. Every running job therefore has a deadline it must keep
renewing, and ``reclaim_expired`` is what makes a crashed worker's work
somebody else's work.

**Concurrency is per workspace.** "One execution at a time" was the right
product decision for a single operator; as a platform it becomes a limit each
tenant gets on its own, so one workspace's thousand-row batch cannot starve
another's single run.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Awaitable, Callable

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.base import iso, utcnow
from db.models import Job

log = logging.getLogger(__name__)

#: How long a claim is good for before another worker may take it. Long enough
#: that a slow step does not lose its own job, short enough that a crash is
#: noticed in a reasonable time. Renewed by the heartbeat while work proceeds.
DEFAULT_LEASE_SECONDS = 120

#: How often a running job pushes its lease forward.
HEARTBEAT_SECONDS = 30

TERMINAL = ("succeeded", "failed", "cancelled")


def worker_identity() -> str:
    """Something a human can trace back to a process when a lease goes stale."""
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass(slots=True)
class ClaimedJob:
    id: str
    workspace_id: str
    owner_id: str | None
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int

    @property
    def is_final_attempt(self) -> bool:
        return self.attempts >= self.max_attempts


def _job_dict(row: Job) -> dict[str, Any]:
    return {
        "id": row.id,
        "workspace_id": row.workspace_id,
        "kind": row.kind,
        "payload": row.payload or {},
        "status": row.status,
        "priority": row.priority,
        "attempts": row.attempts,
        "max_attempts": row.max_attempts,
        "run_after": iso(row.run_after),
        "claimed_by": row.claimed_by,
        "lease_expires_at": iso(row.lease_expires_at),
        "error": row.error,
        "created_at": iso(row.created_at),
        "finished_at": iso(row.finished_at),
    }


class JobQueue:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        worker_id: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> None:
        self._sessions = sessions
        self._worker = worker_id or worker_identity()
        self._lease = lease_seconds

    @property
    def worker_id(self) -> str:
        return self._worker

    # -- producing ----------------------------------------------------------
    async def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        workspace_id: str,
        owner_id: str | None = None,
        priority: int = 0,
        max_attempts: int = 1,
        dedupe_key: str | None = None,
        delay_seconds: float = 0.0,
        session: AsyncSession | None = None,
    ) -> str | None:
        """Queue a job. Returns its id, or None if ``dedupe_key`` already exists.

        Pass ``session`` to enlist in a caller's transaction -- that is what
        makes "create the batch and queue its work" atomic. Without it the job
        gets its own transaction.
        """
        values = dict(
            workspace_id=workspace_id,
            owner_id=owner_id,
            kind=kind,
            payload=payload,
            status="queued",
            priority=priority,
            max_attempts=max_attempts,
            dedupe_key=dedupe_key,
            run_after=utcnow() + timedelta(seconds=delay_seconds),
        )
        stmt = pg_insert(Job).values(**values)
        if dedupe_key is not None:
            # Let the unique index decide, so two callers racing produce one
            # job rather than one job and one error.
            stmt = stmt.on_conflict_do_nothing(index_elements=[Job.dedupe_key])
        stmt = stmt.returning(Job.id)

        if session is not None:
            return await session.scalar(stmt)
        async with self._sessions() as own:
            job_id = await own.scalar(stmt)
            await own.commit()
            return job_id

    # -- consuming ----------------------------------------------------------
    async def claim(
        self, kinds: list[str] | None = None, *, workspace_concurrency: int | None = None
    ) -> ClaimedJob | None:
        """Take one runnable job, or return None.

        The whole claim is a single statement so that selecting and marking
        cannot be interleaved by another worker. ``SKIP LOCKED`` is what makes
        several workers safe: a row another worker is claiming is stepped over,
        not waited on.
        """
        async with self._sessions() as session:
            now = utcnow()
            candidates = (
                select(Job.id)
                .where(
                    Job.status == "queued",
                    Job.run_after <= now,
                )
                .order_by(Job.priority.desc(), Job.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if kinds:
                candidates = candidates.where(Job.kind.in_(kinds))
            if workspace_concurrency is not None:
                candidates = candidates.where(
                    Job.workspace_id.notin_(await self._saturated(session, workspace_concurrency))
                )

            job_id = await session.scalar(candidates)
            if job_id is None:
                await session.commit()
                return None

            row = await session.scalar(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status="running",
                    attempts=Job.attempts + 1,
                    claimed_by=self._worker,
                    claimed_at=now,
                    lease_expires_at=now + timedelta(seconds=self._lease),
                )
                .returning(Job)
            )
            await session.commit()
            if row is None:
                return None
            return ClaimedJob(
                id=row.id,
                workspace_id=row.workspace_id,
                owner_id=row.owner_id,
                kind=row.kind,
                payload=row.payload or {},
                attempts=row.attempts,
                max_attempts=row.max_attempts,
            )

    async def _saturated(self, session: AsyncSession, limit: int) -> list[str]:
        """Workspaces already running their allowance."""
        rows = (
            await session.execute(
                select(Job.workspace_id)
                .where(Job.status == "running")
                .group_by(Job.workspace_id)
                .having(func.count() >= limit)
            )
        ).all()
        return [r[0] for r in rows]

    async def heartbeat(self, job_id: str) -> bool:
        """Push the lease forward. False means we no longer hold this job."""
        async with self._sessions() as session:
            result = await session.execute(
                update(Job)
                .where(
                    Job.id == job_id,
                    Job.claimed_by == self._worker,
                    Job.status == "running",
                )
                .values(lease_expires_at=utcnow() + timedelta(seconds=self._lease))
            )
            await session.commit()
            return bool(result.rowcount)

    async def succeed(self, job_id: str) -> None:
        await self._finish(job_id, "succeeded")

    async def fail(self, job_id: str, error: str, *, retry: bool = False) -> None:
        """Mark a job failed, or put it back for another attempt.

        Retry is the caller's decision, not the queue's: only the caller knows
        whether the failure was a flaky network or a selector that will never
        match again. Retrying the latter just burns the attempt budget slowly.
        """
        async with self._sessions() as session:
            job = await session.get(Job, job_id)
            if job is None:
                return
            if retry and job.attempts < job.max_attempts:
                job.status = "queued"
                job.claimed_by = None
                job.claimed_at = None
                job.lease_expires_at = None
                # Back off on the square of the attempt count: a dependency
                # that is down stays down for a while, and hammering it makes
                # the outage worse for everyone.
                job.run_after = utcnow() + timedelta(seconds=min(300, 5 * job.attempts**2))
            else:
                job.status = "failed"
                job.finished_at = utcnow()
            job.error = error[:2000]
            await session.commit()

    async def _finish(self, job_id: str, status: str) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status=status, finished_at=utcnow(), lease_expires_at=None)
            )
            await session.commit()

    async def cancel(self, job_id: str, workspace_id: str) -> bool:
        """Cancel a job that has not started. A running job must be stopped by
        its worker, so this only takes queued ones."""
        async with self._sessions() as session:
            result = await session.execute(
                update(Job)
                .where(
                    Job.id == job_id,
                    Job.workspace_id == workspace_id,
                    Job.status == "queued",
                )
                .values(status="cancelled", finished_at=utcnow())
            )
            await session.commit()
            return bool(result.rowcount)

    # -- recovery -----------------------------------------------------------
    async def reclaim_expired(self) -> int:
        """Return jobs whose worker stopped renewing the lease.

        This is the piece that makes a crashed worker recoverable rather than a
        permanently stuck row. A job that has exhausted its attempts is failed
        outright instead of looping.
        """
        now = utcnow()
        async with self._sessions() as session:
            stale = (
                await session.scalars(
                    select(Job).where(
                        Job.status == "running",
                        Job.lease_expires_at.is_not(None),
                        Job.lease_expires_at < now,
                    )
                )
            ).all()
            for job in stale:
                if job.attempts < job.max_attempts:
                    job.status = "queued"
                    job.claimed_by = None
                    job.lease_expires_at = None
                    job.run_after = now
                else:
                    job.status = "failed"
                    job.finished_at = now
                    job.error = "The worker holding this job stopped responding."
            await session.commit()

        if stale:
            log.warning("reclaimed expired job leases", extra={"count": len(stale)})
        return len(stale)

    async def purge_finished(self, older_than_days: int = 14) -> int:
        cutoff = utcnow() - timedelta(days=older_than_days)
        async with self._sessions() as session:
            result = await session.execute(
                delete(Job).where(Job.status.in_(TERMINAL), Job.finished_at < cutoff)
            )
            await session.commit()
            return int(result.rowcount or 0)

    # -- inspection ---------------------------------------------------------
    async def get(self, job_id: str, workspace_id: str | None = None) -> dict[str, Any] | None:
        async with self._sessions() as session:
            stmt = select(Job).where(Job.id == job_id)
            if workspace_id is not None:
                stmt = stmt.where(Job.workspace_id == workspace_id)
            row = await session.scalar(stmt)
            return _job_dict(row) if row else None

    async def list_jobs(
        self, workspace_id: str, *, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            stmt = select(Job).where(Job.workspace_id == workspace_id)
            if status:
                stmt = stmt.where(Job.status == status)
            stmt = stmt.order_by(Job.created_at.desc()).limit(limit)
            return [_job_dict(r) for r in (await session.scalars(stmt)).all()]

    async def depth(self) -> dict[str, int]:
        """Queued and running counts, for the health endpoint and metrics."""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(Job.status, func.count())
                    .where(Job.status.in_(("queued", "running")))
                    .group_by(Job.status)
                )
            ).all()
            return {status: int(count) for status, count in rows}


class Worker:
    """Polls the queue and runs one job at a time.

    Polling rather than LISTEN/NOTIFY for job *arrival*: a notification can be
    missed while a worker is busy, so a correct implementation has to poll as a
    backstop anyway -- and once it polls, the notification only saves latency
    that nobody here can perceive. Events use NOTIFY because a dropped UI
    update is recoverable; a dropped job is not.
    """

    def __init__(
        self,
        queue: JobQueue,
        handlers: dict[str, Callable[[ClaimedJob], Awaitable[None]]],
        *,
        poll_interval: float = 2.0,
        workspace_concurrency: int | None = 1,
    ) -> None:
        self._queue = queue
        self._handlers = handlers
        self._poll = poll_interval
        self._workspace_concurrency = workspace_concurrency
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop(), name="job-worker")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                job = await self._queue.claim(
                    kinds=list(self._handlers),
                    workspace_concurrency=self._workspace_concurrency,
                )
                if job is None:
                    await asyncio.sleep(self._poll)
                    continue
                await self._run(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the worker loop must not die
                log.exception("job worker loop error")
                await asyncio.sleep(self._poll)

    async def _run(self, job: ClaimedJob) -> None:
        handler = self._handlers.get(job.kind)
        if handler is None:
            await self._queue.fail(job.id, f"No handler registered for {job.kind!r}.")
            return

        beat = asyncio.create_task(self._heartbeat(job.id), name=f"heartbeat-{job.id}")
        try:
            await handler(job)
            await self._queue.succeed(job.id)
        except asyncio.CancelledError:
            # A shutdown mid-job: release it rather than leaving it running
            # until the lease expires, so a restart picks it straight back up.
            await self._queue.fail(job.id, "Worker shut down mid-job.", retry=True)
            raise
        except Exception as exc:  # noqa: BLE001 - one bad job must not stop the worker
            log.exception("job failed", extra={"job_id": job.id, "job_kind": job.kind})
            await self._queue.fail(job.id, str(exc), retry=not job.is_final_attempt)
        finally:
            beat.cancel()

    async def _heartbeat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            if not await self._queue.heartbeat(job_id):
                log.warning("lost the lease on a running job", extra={"job_id": job_id})
                return


__all__ = ["ClaimedJob", "JobQueue", "Worker", "worker_identity"]
