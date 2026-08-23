"""Structured JSON logging with the current run id attached to every record.

Every log line is one JSON object, so a run can be reconstructed from a log
aggregator with ``run_id == "..."`` and nothing else.
"""

from __future__ import annotations

import contextvars
import json
import logging
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


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

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
