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
    #: "bedrock" (AWS credential chain, no API key) or "anthropic" (API key).
    llm_provider: Literal["bedrock", "anthropic"] = "bedrock"

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

    #: Only required when llm_provider == "anthropic".
    anthropic_api_key: str = ""

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

    #: Self-healing. OFF by default, and deliberately so: it is the best
    #: defence against a site redesign and also the easiest way to turn a free
    #: batch back into an expensive one. Only steps whose `on_failure` is
    #: "heal" are ever offered a repair, and the caps below bound a whole batch.
    replay_healing_enabled: bool = False
    replay_heal_max_attempts: int = 3
    replay_heal_max_tokens: int = 20_000

    # --- Server ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    database_path: str = "./data/runs.db"
    artifacts_dir: str = "./artifacts"
    log_level: str = "INFO"
    #: Same NoDecode reasoning as agent_allowed_domains above.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    @field_validator("agent_allowed_domains", "cors_origins", mode="before")
    @classmethod
    def _csv(cls, value):  # noqa: ANN001 - pydantic hook
        return _split_csv(value)

    @field_validator("mcp_storage_state", "aws_region", "aws_profile", mode="before")
    @classmethod
    def _blank_to_none(cls, value):  # noqa: ANN001 - pydantic hook
        return value or None

    # --- Derived helpers ---------------------------------------------------
    @property
    def db_path(self) -> Path:
        path = Path(self.database_path)
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
    settings = Settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.artifacts_path.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
