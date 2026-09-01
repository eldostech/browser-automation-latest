"""The durable job queue.

``jobs.py`` was written, reviewed and merged without a single caller or a
single test, and it stayed that way long enough for the code that was supposed
to use it to be described as "not yet wired" in the backlog. It is wired now,
and these are the tests that were missing -- written against the properties the
module's docstring claims, because those are the claims a future change is
liable to break quietly:

* a claim is exclusive, so two workers never run the same job;
* a lease expires, so a worker that dies releases its work;
* concurrency is per workspace, so one tenant cannot starve another;
* enqueueing a job and writing the row it is about happen in one transaction.
"""

from __future__ import annotations

import pytest

from db.base import utcnow
from db.models import Job
from jobs import JobQueue
from sqlalchemy import select, update

pytestmark = pytest.mark.anyio


@pytest.fixture
def queue(root_store) -> JobQueue:
    return JobQueue(root_store.sessions, worker_id="worker-a")


@pytest.fixture
def other_queue(root_store) -> JobQueue:
    """A second worker against the same table, which is the whole point."""
    return JobQueue(root_store.sessions, worker_id="worker-b")


async def test_a_queued_job_comes_back_once(queue, other_queue, store):
    job_id = await queue.enqueue("batch", {"n": 1}, workspace_id=store.workspace_id)
    assert job_id

    first = await queue.claim(["batch"], workspace_concurrency=None)
    assert first is not None
    assert first.id == job_id
    assert first.payload == {"n": 1}
    assert first.attempts == 1

    # SKIP LOCKED is what makes this the interesting assertion: the second
    # worker steps over the claimed row rather than blocking on it.
    assert await other_queue.claim(["batch"], workspace_concurrency=None) is None


async def test_a_job_of_another_kind_is_left_alone(queue, store):
    await queue.enqueue("recording", {}, workspace_id=store.workspace_id)
    assert await queue.claim(["batch"], workspace_concurrency=None) is None


async def test_succeeding_finishes_the_row(queue, store, root_store):
    job_id = await queue.enqueue("batch", {}, workspace_id=store.workspace_id)
    claimed = await queue.claim(["batch"], workspace_concurrency=None)
    await queue.succeed(claimed.id)

    row = await queue.get(job_id)
    assert row["status"] == "succeeded"
    assert row["finished_at"] is not None


async def test_failing_without_retry_is_final(queue, store):
    job_id = await queue.enqueue("batch", {}, workspace_id=store.workspace_id)
    claimed = await queue.claim(["batch"], workspace_concurrency=None)
    await queue.fail(claimed.id, "the selector is gone")

    row = await queue.get(job_id)
    assert row["status"] == "failed"
    assert "selector" in row["error"]
    assert await queue.claim(["batch"], workspace_concurrency=None) is None


async def test_failing_with_retry_puts_it_back(queue, store):
    job_id = await queue.enqueue(
        "batch", {}, workspace_id=store.workspace_id, max_attempts=2
    )
    claimed = await queue.claim(["batch"], workspace_concurrency=None)
    await queue.fail(claimed.id, "the network blinked", retry=True)

    row = await queue.get(job_id)
    assert row["status"] == "queued"
    # Backed off, so a dependency that is down is not hammered while it is.
    assert row["run_after"] > row["created_at"]


async def test_an_expired_lease_is_reclaimed(queue, other_queue, store, root_store):
    """The failure that matters is a worker dying mid-job.

    A boolean `locked` column strands that row forever. A lease expires, and
    this is the sweep that makes a dead worker's work somebody else's.
    """
    await queue.enqueue("batch", {}, workspace_id=store.workspace_id, max_attempts=2)
    claimed = await queue.claim(["batch"], workspace_concurrency=None)

    # Simulate the worker dying: nothing renews the lease and it falls behind.
    async with root_store.sessions() as session:
        await session.execute(
            update(Job).where(Job.id == claimed.id).values(lease_expires_at=utcnow())
        )
        await session.commit()

    assert await queue.reclaim_expired() == 1
    retaken = await other_queue.claim(["batch"], workspace_concurrency=None)
    assert retaken is not None
    assert retaken.id == claimed.id
    assert retaken.attempts == 2


async def test_a_reclaimed_job_out_of_attempts_fails_instead_of_looping(
    queue, store, root_store
):
    await queue.enqueue("batch", {}, workspace_id=store.workspace_id, max_attempts=1)
    claimed = await queue.claim(["batch"], workspace_concurrency=None)
    async with root_store.sessions() as session:
        await session.execute(
            update(Job).where(Job.id == claimed.id).values(lease_expires_at=utcnow())
        )
        await session.commit()

    await queue.reclaim_expired()
    row = await queue.get(claimed.id)
    assert row["status"] == "failed"
    assert "stopped responding" in row["error"]


async def test_concurrency_is_counted_per_workspace(queue, other_queue, root_store, store):
    """One workspace at its limit must not stop another one working.

    This is the property that replaced the single in-process execution slot,
    and it is the one that would silently regress into a global lock.
    """
    other_ws = await root_store.ensure_workspace("Other", "other")

    await queue.enqueue("batch", {"ws": "test"}, workspace_id=store.workspace_id)
    await queue.enqueue("batch", {"ws": "test"}, workspace_id=store.workspace_id)
    await queue.enqueue("batch", {"ws": "other"}, workspace_id=other_ws)

    first = await queue.claim(["batch"], workspace_concurrency=1)
    assert first.workspace_id == store.workspace_id

    # The second job in that workspace is held back; the other tenant's is not.
    second = await other_queue.claim(["batch"], workspace_concurrency=1)
    assert second is not None
    assert second.workspace_id == other_ws


async def test_a_queued_job_can_be_cancelled_but_a_running_one_cannot(queue, store):
    queued = await queue.enqueue("batch", {}, workspace_id=store.workspace_id)
    assert await queue.cancel(queued, store.workspace_id) is True
    assert (await queue.get(queued))["status"] == "cancelled"

    running = await queue.enqueue("batch", {}, workspace_id=store.workspace_id)
    await queue.claim(["batch"], workspace_concurrency=None)
    # A running job has a worker driving a browser; only that worker can stop
    # it, which is what ReplayManager.cancel_batch falls through to.
    assert await queue.cancel(running, store.workspace_id) is False


async def test_another_tenant_cannot_cancel_your_job(queue, root_store, store):
    other_ws = await root_store.ensure_workspace("Other", "other")
    job_id = await queue.enqueue("batch", {}, workspace_id=store.workspace_id)
    assert await queue.cancel(job_id, other_ws) is False
    assert (await queue.get(job_id))["status"] == "queued"


async def test_a_dedupe_key_makes_enqueueing_idempotent(queue, store):
    first = await queue.enqueue(
        "batch", {}, workspace_id=store.workspace_id, dedupe_key="batch:abc"
    )
    second = await queue.enqueue(
        "batch", {}, workspace_id=store.workspace_id, dedupe_key="batch:abc"
    )
    assert first is not None
    assert second is None


async def test_enqueue_can_join_the_callers_transaction(queue, root_store, store):
    """The reason this queue is a table and not a broker.

    A rolled-back transaction must leave no job behind, or the UI shows queued
    work that nothing will ever claim.
    """
    async with root_store.sessions() as session:
        await queue.enqueue(
            "batch", {}, workspace_id=store.workspace_id, session=session
        )
        await session.rollback()

    async with root_store.sessions() as session:
        assert (await session.scalars(select(Job))).all() == []


async def test_depth_reports_what_keda_would_scale_on(queue, store):
    await queue.enqueue("batch", {}, workspace_id=store.workspace_id)
    await queue.enqueue("batch", {}, workspace_id=store.workspace_id)
    await queue.claim(["batch"], workspace_concurrency=None)

    assert await queue.depth() == {"queued": 1, "running": 1}
