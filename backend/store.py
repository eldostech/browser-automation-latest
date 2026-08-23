"""SQLite persistence for runs, events and artifacts.

Why SQLite: a single-node control plane with modest write volume, and history
that must survive a restart. It needs no extra service, which keeps
clone-to-first-run short. Swap in Postgres when you want several backend
replicas sharing one history -- the ``Store`` surface below is intentionally
small enough that a Postgres implementation is a drop-in.

Screenshots are written to the filesystem and only referenced from the
database; binary blobs in SQLite would bloat the file and slow down the event
queries that the WebSocket replay depends on.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import aiosqlite

from events import AgentEvent, RunFinished, RunStatus, dump_event, parse_event

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    task          TEXT NOT NULL,
    start_url     TEXT,
    status        TEXT NOT NULL,
    options       TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    steps         INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER,
    summary       TEXT,
    result        TEXT,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS events (
    run_id  TEXT NOT NULL,
    seq     INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    type    TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    seq        INTEGER,
    kind       TEXT NOT NULL,
    mime       TEXT NOT NULL,
    path       TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_runs_status    ON runs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_run  ON artifacts (run_id, seq);
"""

ORPHAN_MESSAGE = "Backend restarted while this run was in flight."


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class Store:
    def __init__(self, db_path: Path, artifacts_dir: Path) -> None:
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        self._db: aiosqlite.Connection | None = None

    # -- lifecycle ----------------------------------------------------------
    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        # WAL lets the WebSocket replay read while the agent loop writes.
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        log.info("store connected", extra={"db_path": str(self.db_path)})

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store.connect() has not been awaited")
        return self._db

    async def ping(self) -> bool:
        try:
            await self.db.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 - a health probe must never raise
            return False

    # -- runs ---------------------------------------------------------------
    async def create_run(
        self, run_id: str, task: str, start_url: str | None, options: dict[str, Any]
    ) -> RunRecord:
        created = _now()
        await self.db.execute(
            "INSERT INTO runs (id, task, start_url, status, options, created_at)"
            " VALUES (?, ?, ?, 'pending', ?, ?)",
            (run_id, task, start_url, json.dumps(options), created),
        )
        await self.db.commit()
        return RunRecord(
            id=run_id,
            task=task,
            start_url=start_url,
            status="pending",
            options=options,
            created_at=created,
            started_at=None,
            finished_at=None,
            steps=0,
            duration_ms=None,
            summary=None,
            result=None,
            error=None,
        )

    async def mark_started(self, run_id: str) -> None:
        await self.db.execute(
            "UPDATE runs SET status='running', started_at=? WHERE id=?", (_now(), run_id)
        )
        await self.db.commit()

    async def set_status(self, run_id: str, status: RunStatus) -> None:
        await self.db.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))
        await self.db.commit()

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
        await self.db.execute(
            "UPDATE runs SET status=?, finished_at=?, steps=?, duration_ms=?,"
            " summary=?, result=?, error=? WHERE id=?",
            (
                status,
                _now(),
                steps,
                duration_ms,
                summary,
                json.dumps(result) if result is not None else None,
                error,
                run_id,
            ),
        )
        await self.db.commit()

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)) as cursor:
            row = await cursor.fetchone()
        return self._row_to_run(row) if row else None

    async def list_runs(
        self, *, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[RunRecord]:
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        async with self.db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_run(row) for row in rows]

    async def count_runs(self, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM runs"
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        async with self.db.execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return int(row["n"])

    async def reap_orphaned_runs(self) -> int:
        """Fail runs left mid-flight by a backend crash or restart.

        Nothing is resuming them, so leaving them 'running' would make the
        history list lie and the live run view spin forever.
        """
        async with self.db.execute(
            "SELECT id FROM runs WHERE status IN ('pending','running','awaiting_approval')"
        ) as cursor:
            rows = await cursor.fetchall()
        orphans = [row["id"] for row in rows]
        if not orphans:
            return 0

        for run_id in orphans:
            next_seq = await self.next_seq(run_id)
            await self.append_event(
                RunFinished(
                    run_id=run_id,
                    seq=next_seq,
                    status="failed",
                    steps=0,
                    duration_ms=0,
                    error=ORPHAN_MESSAGE,
                )
            )
        await self.db.execute(
            "UPDATE runs SET status='failed', finished_at=?, error=?"
            " WHERE status IN ('pending','running','awaiting_approval')",
            (_now(), ORPHAN_MESSAGE),
        )
        await self.db.commit()
        log.warning("reaped orphaned runs", extra={"count": len(orphans)})
        return len(orphans)

    @staticmethod
    def _row_to_run(row: aiosqlite.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            task=row["task"],
            start_url=row["start_url"],
            status=row["status"],
            options=_loads(row["options"], {}),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            steps=row["steps"] or 0,
            duration_ms=row["duration_ms"],
            summary=row["summary"],
            result=_loads(row["result"]),
            error=row["error"],
        )

    # -- events -------------------------------------------------------------
    async def append_event(self, event: AgentEvent) -> None:
        """Insert an event, or replace it if that ``seq`` already exists.

        Replacement is what makes streaming ``thinking`` events work: the agent
        reserves one ``seq`` per prose block and rewrites it as text arrives,
        so replay after a reconnect yields the final text exactly once.
        """
        payload = dump_event(event)
        await self.db.execute(
            "INSERT INTO events (run_id, seq, ts, type, payload) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(run_id, seq) DO UPDATE SET ts=excluded.ts, payload=excluded.payload",
            (event.run_id, event.seq, event.ts, payload["type"], json.dumps(payload)),
        )
        await self.db.commit()

    async def get_events(
        self, run_id: str, after_seq: int = 0, limit: int = 5000
    ) -> list[AgentEvent]:
        async with self.db.execute(
            "SELECT payload FROM events WHERE run_id=? AND seq>? ORDER BY seq ASC LIMIT ?",
            (run_id, after_seq, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [parse_event(json.loads(row["payload"])) for row in rows]

    async def next_seq(self, run_id: str) -> int:
        async with self.db.execute(
            "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM events WHERE run_id=?", (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["max_seq"]) + 1

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
        artifact_id = uuid.uuid4().hex
        directory = self.artifacts_dir / run_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{artifact_id}{suffix}"
        path.write_bytes(data)

        record = ArtifactRecord(
            id=artifact_id,
            run_id=run_id,
            seq=seq,
            kind=kind,
            mime=mime,
            path=str(path),
            bytes=len(data),
            created_at=_now(),
        )
        await self.db.execute(
            "INSERT INTO artifacts (id, run_id, seq, kind, mime, path, bytes, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.run_id,
                record.seq,
                record.kind,
                record.mime,
                record.path,
                record.bytes,
                record.created_at,
            ),
        )
        await self.db.commit()
        return record

    async def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        async with self.db.execute(
            "SELECT * FROM artifacts WHERE id=?", (artifact_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_artifact(row)

    async def list_artifacts(self, run_id: str) -> Sequence[ArtifactRecord]:
        async with self.db.execute(
            "SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at ASC", (run_id,)
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_artifact(row) for row in rows]

    @staticmethod
    def _row_to_artifact(row: aiosqlite.Row) -> ArtifactRecord:
        return ArtifactRecord(
            id=row["id"],
            run_id=row["run_id"],
            seq=row["seq"],
            kind=row["kind"],
            mime=row["mime"],
            path=row["path"],
            bytes=row["bytes"],
            created_at=row["created_at"],
        )
