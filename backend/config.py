"""Process-wide configuration, loaded from the environment / .env file.

Everything the operator can tune lives here. Per-run overrides (max steps,
allowed domains, headless) arrive on the ``POST /api/runs`` body and are merged
on top of these defaults in ``runner.py``.
"""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


def _split_csv(value: str | list[str] | None) -> list[str]:
    """Parse a list setting from a comma-separated string, a JSON array, or a list.

    The fields using this are annotated ``NoDecode``, so pydantic-settings hands
    over the raw .env string untouched. Accepting the JSON form too means a
    value written either way behaves the same.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = value.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", Path(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---------------------------------------------------------------
    # Claude on Amazon Bedrock, authenticated by the standard AWS chain. There
    # is no provider setting: adding one back means adding a second code path
    # to keep working, and nothing here needs it.

    #: The DRIVER model: the agent loop that records a use case by actually
    #: driving the browser. Every step of a recording costs tokens here, so it
    #: is the one worth keeping fast.
    #:
    #: On Bedrock this must be a Bedrock model ID. Current Claude models are
    #: only offered through cross-region inference profiles, so the ID carries
    #: a region prefix ("us." / "eu." / "apac." / "global."). Naming the bare
    #: foundation model fails with "on-demand throughput isn't supported".
    llm_model: str = "us.anthropic.claude-sonnet-5"

    #: The REPAIR model: self-healing mid-run, and repairing a failed use case
    #: afterwards. Both are one-shot judgement calls on a page the model has
    #: never seen, run rarely, where a wrong answer gets written back into a
    #: use case and then repeats silently on every future row -- so they are
    #: worth more capability than the driver.
    llm_repair_model: str = "us.anthropic.claude-opus-5"

    #: The DISTILLER model: the single call that turns a recording into a use
    #: case. Blank means "use the driver model". Split out because it is one
    #: call per use case rather than per step, so it can be pointed at a
    #: stronger model without materially changing cost.
    llm_distill_model: str = ""

    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    #: Both are optional. Left unset, the AWS SDK resolves them itself from the
    #: environment, ~/.aws, or the attached IAM role -- which is what lets the
    #: same build run on a laptop and on an EC2/ECS/Lambda role unchanged.
    #: These map to the standard AWS_REGION / AWS_PROFILE variables.
    aws_region: str | None = None
    aws_profile: str | None = None

    # --- MCP ---------------------------------------------------------------
    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_npx_package: str = "@playwright/mcp@latest"
    mcp_browser: str = "chromium"
    mcp_headless: bool = True
    mcp_isolated: bool = True
    mcp_storage_state: str | None = None
    mcp_extra_args: str = ""
    mcp_server_url: str = "http://localhost:8931/sse"
    mcp_handshake_timeout: float = 45.0
    mcp_tool_timeout: float = 60.0

    # --- Agent guardrails --------------------------------------------------
    agent_max_steps: int = 30
    agent_timeout_seconds: float = 300.0
    #: NoDecode is required: without it pydantic-settings tries json.loads() on
    #: the raw .env string before any validator runs, so a comma-separated
    #: value raises SettingsError at import time. NoDecode hands the raw
    #: string to the _csv validator below instead.
    agent_allowed_domains: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["example.com", "*.example.com"]
    )
    agent_require_approval: bool = True
    agent_approval_timeout_seconds: float = 300.0
    agent_screenshot_every_step: bool = True
    # Tool results are fed back to the model verbatim; cap them so one enormous
    # accessibility snapshot cannot blow the context window.
    agent_max_tool_result_chars: int = 20_000
    # Oldest tool results are trimmed once history exceeds this many messages.
    agent_max_history_messages: int = 60

    # --- Credentials -------------------------------------------------------
    #: Fernet key encrypting stored credentials. Generate one with:
    #:     python -c "from cryptography.fernet import Fernet;
    #:                print(Fernet.generate_key().decode())"
    #: Left unset, credential storage is DISABLED -- deliberately, rather than
    #: falling back to writing passwords in the clear. Losing the key makes
    #: existing stored credentials unreadable.
    credentials_key: str = ""

    # --- Replay ------------------------------------------------------------
    #: Rows per batch run in sequence on one shared browser session; this is
    #: the delay between them. Politeness, and it keeps a fast use case from
    #: looking like a denial-of-service to the target site.
    replay_row_delay_seconds: float = 0.3
    #: Abort a batch after this many consecutive row failures. Ten minutes of a
    #: broken selector failing 400 rows is worse than stopping and saying so.
    replay_failure_streak_limit: int = 5
    #: Per-step wall clock ceiling inside a replay.
    replay_step_timeout: float = 30.0

    #: How much of an executed use case to photograph.
    #:
    #:   off        nothing at all
    #:   failure    only where a step failed
    #:   final      one per row, showing the end state  (default)
    #:   every_step everything -- for troubleshooting, not for a large batch
    #:
    #: Executing a use case used to capture only failures, so a *successful*
    #: run left no visual record: nothing to audit, and nothing to look at when
    #: a result was questioned afterwards. "final" answers "what actually
    #: happened to record 700" at one image per row. "every_step" multiplies
    #: that by the step count, which over a thousand rows is gigabytes -- put
    #: STORAGE_BACKEND on S3 before choosing it for a large batch.
    replay_screenshots: Literal["off", "failure", "final", "every_step"] = "final"

    #: Self-healing. OFF by default, and deliberately so: it is the best
    #: defence against a site redesign and also the easiest way to turn a free
    #: batch back into an expensive one. Only steps whose `on_failure` is
    #: "heal" are ever offered a repair, and the caps below bound a whole batch.
    replay_healing_enabled: bool = False
    replay_heal_max_attempts: int = 3
    replay_heal_max_tokens: int = 20_000

    # --- Database ----------------------------------------------------------
    #: Postgres, in parts rather than as one DSN string. A password containing
    #: '@', ':' or '/' cannot be safely interpolated into a URL, and this
    #: project's own password does; db/engine.py escapes each part with
    #: URL.create(). See docs/operations/database.md.
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "postgres"
    db_user: str = "postgres"
    db_password: str = ""
    #: A named schema rather than 'public', so this application can share a
    #: database without colliding, and so its tables can be dropped as a unit.
    db_schema: str = "browser"

    #: Pool sizing. The defaults suit a single web process; a deployment with
    #: several workers should keep (pool_size + max_overflow) * workers below
    #: Postgres max_connections.
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_pool_timeout: float = 30.0
    #: LangGraph's checkpointer, which is what lets a run in flight survive a
    #: restart. Off in tests, where an in-memory saver is correct and a second
    #: connection is waste. See checkpoints.py for why the backend differs by
    #: platform.
    checkpoint_enabled: bool = True
    checkpoint_path: str = "./data/checkpoints.sqlite"
    #: Recycle below any proxy/firewall idle timeout, which is what turns a
    #: pooled connection into a mystery 500 hours after it was opened.
    db_pool_recycle: int = 1800
    db_echo: bool = False

    # --- Authentication ----------------------------------------------------
    #: Bearer sessions are opaque tokens hashed in the database, so there is no
    #: signing secret to configure or rotate.
    auth_session_ttl_hours: int = 12
    #: bcrypt work factor. 12 is roughly 250ms on current hardware -- slow
    #: enough to matter for offline cracking, fast enough for a login.
    auth_bcrypt_rounds: int = 12
    #: Bootstrap: created on first startup if no user exists at all, so a fresh
    #: deployment is reachable. Leave the password blank and one is generated
    #: and printed once to the log.
    bootstrap_admin_email: str = "admin@localhost"
    bootstrap_admin_password: str = ""
    bootstrap_workspace_name: str = "Default"

    # --- Artifact storage --------------------------------------------------
    #: Where screenshots and other run artifacts are kept. "local" writes under
    #: artifacts_dir; "s3" uploads to a bucket and serves presigned URLs.
    #: Rows written under one backend stay readable after switching to the
    #: other -- see storage.py.
    storage_backend: Literal["local", "s3"] = "local"
    s3_bucket: str = ""
    #: Key prefix inside the bucket, so several deployments can share one.
    s3_prefix: str = "artifacts"
    #: Falls back to aws_region when blank.
    s3_region: str = ""
    #: For MinIO or any S3-compatible store. Blank means real AWS.
    s3_endpoint_url: str = ""
    #: How long a presigned artifact URL stays valid. Short: it is a bearer
    #: URL for otherwise access-controlled content.
    s3_url_expiry_seconds: int = 900

    # --- Server ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    artifacts_dir: str = "./artifacts"
    log_level: str = "INFO"

    # --- Logging -----------------------------------------------------------
    #: Write logs to a rotating file as well as stdout. Containers usually want
    #: this off (the platform collects stdout); anyone running the backend
    #: directly wants it on, because stdout scrolls away.
    log_to_file: bool = True
    log_dir: str = "./logs"
    log_file_name: str = "backend.log"
    #: Rotation. Ten files of 10MB is enough to cover a few days of a busy
    #: instance without needing a cron job to tidy up.
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 10
    #: Same NoDecode reasoning as agent_allowed_domains above.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    @field_validator("agent_allowed_domains", "cors_origins", mode="before")
    @classmethod
    def _csv(cls, value):  # noqa: ANN001 - pydantic hook
        return _split_csv(value)

    @field_validator(
        "mcp_storage_state", "aws_region", "aws_profile", mode="before"
    )
    @classmethod
    def _blank_to_none(cls, value):  # noqa: ANN001 - pydantic hook
        return value or None

    # --- Derived helpers ---------------------------------------------------
    @property
    def log_path(self) -> Path:
        path = Path(self.log_dir)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    @property
    def artifacts_path(self) -> Path:
        path = Path(self.artifacts_dir)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    @property
    def extra_mcp_args(self) -> list[str]:
        return self.mcp_extra_args.split() if self.mcp_extra_args else []

    @property
    def distill_model(self) -> str:
        """Model for the one distillation call. Falls back to the driver."""
        return self.llm_distill_model.strip() or self.llm_model

    @property
    def models_in_use(self) -> dict[str, str]:
        """Which model does what. Surfaced by /healthz and /api/config."""
        return {
            "driver": self.llm_model,
            "distiller": self.distill_model,
            "repair": self.llm_repair_model,
        }

    def resolve_npx(self) -> str:
        """Absolute path to npx.

        Resolving explicitly matters on Windows, where the executable is
        ``npx.cmd`` and a bare ``npx`` will not be found by ``CreateProcess``.
        """
        return shutil.which("npx") or shutil.which("npx.cmd") or "npx"


@lru_cache
def get_settings() -> Settings:
    """The process's settings, built once.

    There is deliberately no module-level ``settings`` object. One used to
    exist, and because it was constructed at *import* time, tests that meant to
    supply their own configuration silently picked up the developer's ``.env``
    instead -- a bug this project shipped three separate times (an API key, the
    default model, then the repair model). A function that must be called can
    be overridden; an object that already exists by the time your test runs
    cannot.

    FastAPI handlers should depend on ``deps.get_config`` rather than calling
    this, so a test can override the dependency for one app instance.
    """
    settings = Settings()
    settings.artifacts_path.mkdir(parents=True, exist_ok=True)
    return settings
