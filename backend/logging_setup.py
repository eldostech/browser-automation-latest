"""Structured JSON logging with the current run id attached to every record.

Every log line is one JSON object, so a run can be reconstructed from a log
aggregator with ``run_id == "..."`` and nothing else.
"""

from __future__ import annotations

import contextvars
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any

run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        run_id = getattr(record, "run_id", None) or run_id_var.get()
        if run_id:
            payload["run_id"] = run_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_") and key != "run_id":
                payload[key] = _safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def configure_logging(
    level: str = "INFO",
    *,
    log_dir: Path | None = None,
    file_name: str = "backend.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 10,
) -> None:
    """Send logs to stdout, and to a rotating file when ``log_dir`` is given.

    Both destinations get the same JSON, so a line grepped out of a file and a
    line scraped from a container's stdout parse identically.

    Rotation is by size rather than by time. A run that goes wrong produces far
    more output than a quiet day, and a daily file is either too big on the bad
    day or pointless on the good one.

    A file that cannot be opened is a warning, not a failure. Losing the log is
    bad; refusing to start the backend because a directory is read-only is
    worse, and stdout still works.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())
    root.addHandler(stream)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            rotating = RotatingFileHandler(
                log_dir / file_name,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            rotating.setFormatter(JsonFormatter())
            root.addHandler(rotating)
        except OSError as exc:
            root.warning(
                "could not open the log file; logging to stdout only",
                extra={"log_dir": str(log_dir), "error": str(exc)},
            )

    # uvicorn installs its own colourised handlers; route them through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # The MCP SDK is chatty at DEBUG about every JSON-RPC frame.
    logging.getLogger("mcp").setLevel(max(logging.INFO, root.level))
    logging.getLogger("httpx").setLevel(logging.WARNING)


def bind_run_id(run_id: str | None) -> None:
    """Attach ``run_id`` to every log record emitted by the current task."""
    run_id_var.set(run_id)
