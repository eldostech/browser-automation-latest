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

CREATE TABLE IF NOT EXISTS usecases (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'draft',
    current_version INTEGER NOT NULL DEFAULT 1,
    source_run_id   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- Versions are immutable: an edit appends a row rather than rewriting one, so
-- a batch already running cannot have its definition changed underneath it.
CREATE TABLE IF NOT EXISTS usecase_versions (
    usecase_id TEXT NOT NULL,
    version    INTEGER NOT NULL,
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT,
    PRIMARY KEY (usecase_id, version)
);

-- Values are Fernet ciphertext; there is no code path that returns them over
-- HTTP. See credentials.py.
CREATE TABLE IF NOT EXISTS credentials (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    slots        TEXT NOT NULL DEFAULT '[]',
    ciphertext   BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    last_used_at TEXT
);

CREATE TABLE IF NOT EXISTS batches (
    id            TEXT PRIMARY KEY,
    usecase_id    TEXT NOT NULL,
    version       INTEGER NOT NULL,
    status        TEXT NOT NULL,
    total         INTEGER NOT NULL DEFAULT 0,
    succeeded     INTEGER NOT NULL DEFAULT 0,
    failed        INTEGER NOT NULL DEFAULT 0,
    credential_id TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    finished_at   TEXT
);

CREATE TABLE IF NOT EXISTS executions (
    id             TEXT PRIMARY KEY,
    batch_id       TEXT,
    usecase_id     TEXT NOT NULL,
    version        INTEGER NOT NULL,
    run_id         TEXT,
    row_index      INTEGER,
    inputs         TEXT NOT NULL DEFAULT '{}',
    outputs        TEXT,
    status         TEXT NOT NULL,
    failed_step_id TEXT,
    error          TEXT,
    llm_calls      INTEGER NOT NULL DEFAULT 0,
    llm_tokens     INTEGER NOT NULL DEFAULT 0,
    duration_ms    INTEGER,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_runs_status    ON runs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_run  ON artifacts (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_usecases_status ON usecases (status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_executions_batch ON executions (batch_id, row_index);
CREATE INDEX IF NOT EXISTS idx_executions_usecase ON executions (usecase_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batches_usecase ON batches (usecase_id, created_at DESC);
"""

#: Schema revision this build expects. ``CREATE TABLE IF NOT EXISTS`` above
#: handles *adding* tables to an existing database, but it silently does
#: nothing when a table exists with an older shape -- so an ``ALTER`` or a
#: backfill needs somewhere to hang. :data:`MIGRATIONS` is that place, and
#: ``user_version`` records how far a given file has been brought forward.
SCHEMA_VERSION = 3

#: ``{target_version: (sql_statement, ...)}``, applied in ascending order to
#: any database whose ``user_version`` is below the target. Statements must be
#: idempotent where SQLite allows it, and must never drop user data.
MIGRATIONS: dict[int, tuple[str, ...]] = {
    # v1 is the original runs/events/artifacts schema.
    1: (),
    # v2 adds usecases + usecase_versions. ``CREATE TABLE IF NOT EXISTS`` in
    # SCHEMA already creates them on connect, so there is nothing to run --
    # the entry exists to record that this database has been seen by a build
    # that knows about those tables.
    2: (),
    # v3 adds credentials, batches and executions. Same reasoning as v2.
    3: (),
}

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
        await self.migrate()
        log.info("store connected", extra={"db_path": str(self.db_path)})

    async def schema_version(self) -> int:
        async with self.db.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def migrate(self) -> int:
        """Bring the database up to :data:`SCHEMA_VERSION`. Returns the version reached.

        A fresh file is stamped at the current version without running anything
        -- ``SCHEMA`` already created it in its final shape. An existing file
        runs only the migrations above its recorded version.
        """
        current = await self.schema_version()
        if current >= SCHEMA_VERSION:
            return current

        for target in sorted(MIGRATIONS):
            if target <= current:
                continue
            for statement in MIGRATIONS[target]:
                await self.db.execute(statement)
            # PRAGMA does not accept a bound parameter.
            await self.db.execute(f"PRAGMA user_version = {int(target)}")
            await self.db.commit()
            log.info("applied schema migration", extra={"from": current, "to": target})
            current = target

        return current

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


    # -- use cases ----------------------------------------------------------
    async def save_usecase(self, definition: dict[str, Any], *, created_by: str | None = None) -> tuple[str, int]:
        """Insert a use case, or append a new version of an existing one.

        Returns ``(usecase_id, version)``. Versions are append-only: an edit
        never rewrites the row a running batch is reading from.
        """
        usecase_id = str(definition["id"])
        name = str(definition.get("name") or "Untitled")
        description = str(definition.get("description") or "")
        status = str(definition.get("status") or "draft")
        now = _now()

        async with self.db.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM usecase_versions WHERE usecase_id=?",
            (usecase_id,),
        ) as cursor:
            row = await cursor.fetchone()
        version = int(row["v"]) + 1

        stored = {**definition, "version": version, "updated_at": now}
        await self.db.execute(
            "INSERT INTO usecase_versions (usecase_id, version, definition, created_at, created_by)"
            " VALUES (?, ?, ?, ?, ?)",
            (usecase_id, version, json.dumps(stored), now, created_by),
        )
        await self.db.execute(
            "INSERT INTO usecases (id, name, description, status, current_version, source_run_id,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET name=excluded.name, description=excluded.description,"
            " status=excluded.status, current_version=excluded.current_version,"
            " updated_at=excluded.updated_at",
            (
                usecase_id,
                name,
                description,
                status,
                version,
                definition.get("source_run_id"),
                now,
                now,
            ),
        )
        await self.db.commit()
        return usecase_id, version

    async def get_usecase(self, usecase_id: str, version: int | None = None) -> dict[str, Any] | None:
        """One stored definition. ``version=None`` means the current one."""
        if version is None:
            async with self.db.execute(
                "SELECT current_version AS v FROM usecases WHERE id=?", (usecase_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            version = int(row["v"])

        async with self.db.execute(
            "SELECT definition FROM usecase_versions WHERE usecase_id=? AND version=?",
            (usecase_id, version),
        ) as cursor:
            row = await cursor.fetchone()
        return _loads(row["definition"]) if row else None

    async def list_usecases(
        self, *, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Summary rows for the list view -- not the full definitions."""
        sql = (
            "SELECT id, name, description, status, current_version, source_run_id,"
            " created_at, updated_at FROM usecases"
        )
        params: list[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        async with self.db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def list_usecase_versions(self, usecase_id: str) -> list[dict[str, Any]]:
        async with self.db.execute(
            "SELECT version, created_at, created_by FROM usecase_versions"
            " WHERE usecase_id=? ORDER BY version DESC",
            (usecase_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def set_usecase_status(self, usecase_id: str, status: str) -> bool:
        """Move a use case between draft / ready / archived.

        Also rewrites the status inside the current stored definition, so a
        definition read back on its own still reports the truth.
        """
        definition = await self.get_usecase(usecase_id)
        if definition is None:
            return False

        definition["status"] = status
        async with self.db.execute(
            "SELECT current_version AS v FROM usecases WHERE id=?", (usecase_id,)
        ) as cursor:
            row = await cursor.fetchone()
        version = int(row["v"])

        await self.db.execute(
            "UPDATE usecase_versions SET definition=? WHERE usecase_id=? AND version=?",
            (json.dumps(definition), usecase_id, version),
        )
        await self.db.execute(
            "UPDATE usecases SET status=?, updated_at=? WHERE id=?", (status, _now(), usecase_id)
        )
        await self.db.commit()
        return True

    async def delete_usecase(self, usecase_id: str) -> bool:
        """Archive rather than delete: a batch's history references the id."""
        return await self.set_usecase_status(usecase_id, "archived")


    # -- credentials --------------------------------------------------------
    async def save_credential(
        self, credential_id: str, name: str, slots: list[str], ciphertext: bytes
    ) -> str:
        """Store an encrypted credential bundle. Re-saving a name replaces it.

        Only ciphertext lands here; see ``credentials.py`` for why there is no
        method to read a value back out over HTTP.
        """
        await self.db.execute(
            "INSERT INTO credentials (id, name, slots, ciphertext, created_at)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET slots=excluded.slots,"
            " ciphertext=excluded.ciphertext",
            (credential_id, name, json.dumps(slots), ciphertext, _now()),
        )
        await self.db.commit()
        async with self.db.execute("SELECT id FROM credentials WHERE name=?", (name,)) as cursor:
            row = await cursor.fetchone()
        return row["id"]

    async def get_credential_ciphertext(self, credential_id: str) -> bytes | None:
        async with self.db.execute(
            "SELECT ciphertext FROM credentials WHERE id=?", (credential_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return row["ciphertext"] if row else None

    async def list_credentials(self) -> list[dict[str, Any]]:
        """Names and slot lists only -- never a value."""
        async with self.db.execute(
            "SELECT id, name, slots, created_at, last_used_at FROM credentials ORDER BY name"
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "slots": _loads(row["slots"], []),
                "created_at": row["created_at"],
                "last_used_at": row["last_used_at"],
            }
            for row in rows
        ]

    async def touch_credential(self, credential_id: str) -> None:
        await self.db.execute(
            "UPDATE credentials SET last_used_at=? WHERE id=?", (_now(), credential_id)
        )
        await self.db.commit()

    async def delete_credential(self, credential_id: str) -> bool:
        cursor = await self.db.execute("DELETE FROM credentials WHERE id=?", (credential_id,))
        await self.db.commit()
        return cursor.rowcount > 0

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
        await self.db.execute(
            "INSERT INTO executions (id, batch_id, usecase_id, version, run_id, row_index,"
            " inputs, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (
                execution_id,
                batch_id,
                usecase_id,
                version,
                run_id,
                row_index,
                json.dumps(inputs or {}),
                _now(),
            ),
        )
        await self.db.commit()

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
        await self.db.execute(
            "UPDATE executions SET status=?, outputs=?, failed_step_id=?, error=?,"
            " duration_ms=?, llm_calls=?, llm_tokens=? WHERE id=?",
            (
                status,
                json.dumps(outputs) if outputs is not None else None,
                failed_step_id,
                error,
                duration_ms,
                llm_calls,
                llm_tokens,
                execution_id,
            ),
        )
        await self.db.commit()

    async def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_execution(row) if row else None

    async def list_executions(
        self, *, batch_id: str | None = None, usecase_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM executions"
        clauses: list[str] = []
        params: list[Any] = []
        if batch_id:
            clauses.append("batch_id=?")
            params.append(batch_id)
        if usecase_id:
            clauses.append("usecase_id=?")
            params.append(usecase_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY COALESCE(row_index, 0) ASC, created_at ASC LIMIT ?"
        params.append(limit)
        async with self.db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_execution(row) for row in rows]

    @staticmethod
    def _row_to_execution(row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "batch_id": row["batch_id"],
            "usecase_id": row["usecase_id"],
            "version": row["version"],
            "run_id": row["run_id"],
            "row_index": row["row_index"],
            "inputs": _loads(row["inputs"], {}),
            "outputs": _loads(row["outputs"]),
            "status": row["status"],
            "failed_step_id": row["failed_step_id"],
            "error": row["error"],
            "llm_calls": row["llm_calls"],
            "llm_tokens": row["llm_tokens"],
            "duration_ms": row["duration_ms"],
            "created_at": row["created_at"],
        }


    # -- batches ------------------------------------------------------------
    async def create_batch(
        self,
        batch_id: str,
        usecase_id: str,
        version: int,
        *,
        total: int,
        credential_id: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO batches (id, usecase_id, version, status, total, credential_id,"
            " created_at) VALUES (?, ?, ?, 'pending', ?, ?, ?)",
            (batch_id, usecase_id, version, total, credential_id, _now()),
        )
        await self.db.commit()

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
        assignments: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("status", status),
            ("succeeded", succeeded),
            ("failed", failed),
            ("error", error),
        ):
            if value is not None:
                assignments.append(f"{column}=?")
                params.append(value)
        if finished:
            assignments.append("finished_at=?")
            params.append(_now())
        if not assignments:
            return
        params.append(batch_id)
        await self.db.execute(
            f"UPDATE batches SET {', '.join(assignments)} WHERE id=?", params
        )
        await self.db.commit()

    async def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        async with self.db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_batches(
        self, *, usecase_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM batches"
        params: list[Any] = []
        if usecase_id:
            sql += " WHERE usecase_id=?"
            params.append(usecase_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        async with self.db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

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
