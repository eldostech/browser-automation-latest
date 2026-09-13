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

from pydantic import Field, field_validator, model_validator
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

    #: The one model setting. A use case is recorded by watching somebody do
    #: it and distilled by parsing the recording, not by asking a model, so
    #: neither of those costs a token -- this is asked only two things: how a
    #: spreadsheet column maps onto a recorded field (mapping fallback), and,
    #: rarely, which control a broken locator now means (healing/repair). Both
    #: are one-shot judgement calls whose answer gets written back and then
    #: repeats silently on every future row, so it is worth capability over
    #: speed. The agent path (recording *with* an agent instead of by hand)
    #: reads this too.
    #:
    #: On Bedrock this must be a Bedrock model ID. Current Claude models are
    #: only offered through cross-region inference profiles, so the ID carries
    #: a region prefix ("us." / "eu." / "apac." / "global."). Naming the bare
    #: foundation model fails with "on-demand throughput isn't supported".
    llm_repair_model: str = "us.anthropic.claude-opus-5"

    #: Which provider answers a model call by default.
    #:
    #: This module's own docstring used to say Bedrock "and only that", on the
    #: reasoning that a second provider is a second code path to keep working.
    #: That reasoning holds for a *deployment* and did not survive contact with
    #: the reason anyone wants a second one: comparing models for accuracy.
    #: OpenRouter is one endpoint in front of most of them, so it buys a great
    #: many models for one code path rather than one model for each -- and it
    #: is OpenAI-shaped, which `langchain-openai` already speaks.
    #:
    #: A run may override this per request; see `llm.ModelChoice`. This is the
    #: fallback for everything that does not.
    llm_provider: Literal["bedrock", "openrouter"] = "bedrock"

    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # --- OpenRouter --------------------------------------------------------
    #: Whether this deployment may reach OpenRouter **at all**.
    #:
    #: A kill switch rather than a preference, and the reason is a deployment
    #: leaving one company for another: an installation that must not send a
    #: byte to a third party needs a single line it can point at, not an
    #: argument about which code paths happen to be reachable. False means the
    #: provider is not offered, not resolvable, not buildable, and its
    #: catalogue is never fetched -- which is itself an HTTP request to
    #: openrouter.ai and the one most easily forgotten.
    #:
    #: `tests/test_models.py` asserts that every one of those paths refuses.
    openrouter_enabled: bool = True

    #: The key. Without it the provider is offered nowhere and selecting it is
    #: refused with that as the reason, rather than failing at the first call.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    #: Used when a run asks for OpenRouter without naming a model. Blank means
    #: "no default": the picker has to choose, which is honest for a provider
    #: whose whole point is that there are hundreds.
    openrouter_model: str = ""

    #: Sent as OpenRouter's attribution headers. They show up on their activity
    #: page, which is how an operator tells this application's spend apart from
    #: everything else on the same key.
    openrouter_app_name: str = "TRACE"
    openrouter_app_url: str = ""

    #: How long the fetched model catalogue is reused before asking again.
    #: OpenRouter lists hundreds and changes them weekly; refetching per page
    #: load would put a third-party request in front of the dashboard.
    openrouter_catalog_ttl_seconds: int = 900

    #: Whether to ask Bedrock what models the account can reach.
    #:
    #: On, now that the objections to it are handled rather than avoided. They
    #: were real: `ListFoundationModels` reports what the *region* carries
    #: rather than what this account may invoke, and it needs an IAM permission
    #: beyond invoking a model. So the discovery is *additive* -- it degrades to
    #: `BEDROCK_MODELS` and says why when the permission is missing -- and the
    #: picker has a check button that proves one model by calling it, which is
    #: the only thing that can distinguish "listed" from "invokable".
    #:
    #: Off for a deployment that would rather make no control-plane call.
    bedrock_discover: bool = True

    #: How long the discovered Bedrock list is reused. Longer than
    #: OpenRouter's: a region's model list changes on AWS's schedule, not
    #: weekly, and this is a control-plane call in front of a page load.
    bedrock_catalog_ttl_seconds: int = 3600

    #: Bedrock models always offered in the picker, whether or not discovery
    #: works. The configured default belongs here so it is present even when
    #: the account cannot list models at all.
    bedrock_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "us.anthropic.claude-opus-5",
            "us.anthropic.claude-sonnet-5",
            "us.anthropic.claude-haiku-4-5-20251001",
        ]
    )

    #: Claude's extended-thinking budget, in tokens. 0 disables it. A model
    #: asked to act with no room to reason first will happily emit a tool
    #: call with no text at all -- a real session on this model produced 62
    #: events and not one word of reasoning, then spent its whole budget
    #: retrying a target ref it had already been told twice did not exist.
    #: This is the fix for the *first* half of that: room to think before
    #: acting. Must be less than ``llm_max_tokens``, since thinking tokens are
    #: drawn from the same budget as the response; a value that leaves no
    #: room for an actual tool call is clamped down with a warning rather
    #: than left to fail the request outright.
    #:
    #: Anthropic's API rejects a non-default ``temperature`` while thinking is
    #: enabled, so ``llm_temperature`` above is ignored for calls made while
    #: this is greater than 0 -- confirmed against the real model, not assumed.
    llm_thinking_budget_tokens: int = 4096

    #: Both are optional. Left unset, the AWS SDK resolves them itself from the
    #: environment, ~/.aws, or the attached IAM role -- which is what lets the
    #: same build run on a laptop and on an EC2/ECS/Lambda role unchanged.
    #: These map to the standard AWS_REGION / AWS_PROFILE variables.
    aws_region: str | None = None
    aws_profile: str | None = None

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

    #: Values this deployment answers ``{{env.x}}`` with, as JSON:
    #: ``USECASE_ENV={"base_url": "https://uat.example.com"}``.
    #:
    #: This is the seam a use case is promoted through. A recording made
    #: against dev holds dev's URLs, and the same document has to run against
    #: UAT and production without being edited -- so the parts that differ per
    #: environment are named in the document and answered by the deployment.
    #: An input cannot do this job: inputs are per row, and filling a base URL
    #: from a spreadsheet column is how a UAT dataset ends up pointed at
    #: production.
    usecase_env: dict[str, str] = {}

    #: What to call this deployment in the interface -- "Dev", "UAT",
    #: "Production". Shown beside the product name, because a use case runs
    #: against whatever `usecase_env` names and nobody should have to guess
    #: which one they are about to start a batch against. Blank hides it,
    #: which is right for a single-environment install.
    environment: str = ""

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

    #: How long a run's events may wait before being written, in seconds.
    #:
    #: Events are written in batches (see `eventbuffer.py`). This is the bound
    #: on how stale the live view may be, not a bound on loss: a batch is also
    #: flushed at the end of every row and on the way out, and an event is
    #: published to watchers only after it is durable.
    #:
    #: The default is well under the threshold where a person reads a delay as
    #: a stall. Raise it for a database several hops away and a run with very
    #: chatty steps; set it near zero to get the old behaviour of a write per
    #: event, which is what you want only when debugging this machinery.
    event_flush_interval: float = Field(default=0.2, ge=0.0, le=5.0)

    #: How many events may accumulate before a flush happens regardless of the
    #: interval. Bounds memory and keeps one batch from growing large enough
    #: that writing it becomes its own latency problem.
    event_flush_max_batch: int = Field(default=200, ge=1, le=5_000)

    #: Self-healing. OFF by default, and deliberately so: it is the best
    #: defence against a site redesign and also the easiest way to turn a free
    #: batch back into an expensive one. Only steps whose `on_failure` is
    #: "heal" are ever offered a repair, and the caps below bound a whole batch.
    # --- Healing memory ----------------------------------------------------
    #: Whether a confirmed fix is remembered and recalled. Off makes healing
    #: behave exactly as it did before there was a memory, which is the point
    #: of the flag: the memory is an optimisation, not a dependency.
    healing_memory_enabled: bool = True
    #: How a page is turned into a vector. "bedrock" is Titan; "hash" is the
    #: deterministic stand-in, for a deployment with no Bedrock access and for
    #: the tests -- similar text still scores closer than unrelated text, which
    #: is the only property retrieval depends on.
    embedding_backend: Literal["bedrock", "hash"] = "bedrock"
    embedding_model: str = "amazon.titan-embed-text-v2:0"

    # --- Browser -----------------------------------------------------------
    #: The browser a replay drives. Playwright is called directly now, so this
    #: is the engine name it knows: chromium, firefox or webkit.
    browser_engine: Literal["chromium", "firefox", "webkit"] = "chromium"
    #: Headless is the default because a batch runs unattended. A single-row
    #: execution can ask for a window per request -- watching it work is the
    #: fastest way to understand why a step fails.
    browser_headless: bool = True
    #: Write a Playwright trace per execution and keep it as an artifact. It
    #: opens in Playwright's own viewer with a DOM snapshot per action, which
    #: for a batch that failed on row 412 is the difference between a
    #: screenshot and being able to look around the page. Off by default: a
    #: trace is a few megabytes per run.
    browser_trace: bool = False

    # --- Recorder ----------------------------------------------------------
    #: Whether this deployment can record. A codegen window needs a display, so
    #: this is a local-development capability; an EKS pod runs replay workers
    #: and answers 501 here rather than failing at spawn time with an X11 error.
    recorder_enabled: bool = True
    #: How codegen is invoked. Blank -- the default -- runs the Playwright
    #: installed alongside this application, which is what keeps the recorder
    #: and the replay engine on one version. Set it only to point at a
    #: different install deliberately.
    recorder_command: str = ""
    recorder_browser: str = "chromium"
    #: How long a window may stay open before it is closed for you. Generous:
    #: a person working through a real form is slow, and losing their recording
    #: to a timeout costs more than a stray browser process does.
    recorder_timeout_seconds: float = 1800.0

    # --- Agent -------------------------------------------------------------
    #: Whether this deployment offers the agent at all. Off by default: it
    #: needs an optional dependency group and a Node runtime, and a feature
    #: that fails when pressed is worse than one that says it is not here.
    #:
    #: This is the third deployment role, beside RECORDER_ENABLED and
    #: WORKER_ENABLED. It exists for the same reason they do -- capabilities
    #: differ per machine, and a laptop, a replay pod and an AgentCore runtime
    #: are not the same machine.
    agent_enabled: bool = False
    #: Where the agent's browser comes from.
    #:
    #:   local      `npx @playwright/mcp`, owning its own Chromium.
    #:   cdp        attach to a browser somebody else runs, named by
    #:              AGENT_CDP_ENDPOINT. This is the AgentCore Browser path,
    #:              and it is why the provider is an interface at all.
    agent_browser_provider: str = "local"
    #: The CDP endpoint to attach to when the provider is `cdp`. Blank
    #: otherwise. A managed browser session's address, never a secret.
    agent_cdp_endpoint: str = ""
    #: The @playwright/mcp version to start. Pinned for the reason `playwright`
    #: is pinned: its tool names and argument shapes are the contract the tool
    #: registry is written against, and `@latest` would let a release change
    #: them inside somebody's session rather than in CI.
    agent_mcp_version: str = "0.0.80"
    #: Whether the agent's browser is headed. Headless by default; a person
    #: watching an authoring session wants to see it, and that is a per-session
    #: choice made where the session starts rather than here.
    agent_headless: bool = True
    #: Whether a finished recording is replayed once, cold, before anybody
    #: sees the draft.
    #:
    #: On, because it is the only thing that distinguishes "the agent got
    #: through the task" from "this recording runs" -- the two are not the same
    #: and the gap between them is what a reviewer cannot see. The replay
    #: itself spends **no tokens**: it is the ordinary engine, which has no
    #: path to a model. What it costs is a browser launch and one pass through
    #: the flow.
    #:
    #: Turn it off if that wall-clock cost is not worth it to you. What you
    #: give up is finding out that a draft does not replay *before* publishing
    #: it rather than on the first real run.
    agent_verify_draft: bool = True

    # --- Job worker --------------------------------------------------------
    #: Whether this process claims queued batches as well as serving HTTP. True
    #: is what makes a single-machine install work with nothing else started.
    #: An API pod that should only serve requests sets this false and leaves the
    #: work to a worker Deployment.
    worker_enabled: bool = True
    #: How long the worker waits before asking for work again when the queue is
    #: empty. Polling rather than LISTEN/NOTIFY because a notification can be
    #: missed while the worker is busy, so a correct implementation polls as a
    #: backstop anyway.
    worker_poll_seconds: float = 2.0
    #: Batches running at once **per workspace**. One is the old single-slot
    #: behaviour, now a limit each tenant gets on its own rather than a global
    #: lock, so one workspace cannot starve another.
    worker_workspace_concurrency: int = 1

    #: The ceiling on healing, not the decision. A use case's own ``mode``
    #: chooses within it; see :func:`usecase.effective_mode`. False here means
    #: no use case in this deployment can reach a model however it is marked.
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
    #: Recycle a pooled connection after this many seconds, so one that a
    #: firewall or the server silently dropped is never handed back stale.
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
    #: Comma-separated in the environment; see NoDecode above.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _known_provider(cls, value):  # noqa: ANN001 - pydantic hook
        """Accept a stale value from an older .env rather than refusing to start.

        This setting existed once, was removed when the codebase went
        Bedrock-only, and is back. An operator upgrading across all of that may
        still have `LLM_PROVIDER=anthropic` in a file nobody has opened in
        months, and a backend that will not start because of one dead line is a
        worse outcome than one that starts on its default and says so.

        A *typo* still fails, which is the house rule -- see
        `test_a_typo_in_a_typed_setting_fails_at_load`. Only the one value this
        codebase itself used to accept is forgiven.
        """
        if isinstance(value, str) and value.strip().lower() == "anthropic":
            import logging

            logging.getLogger(__name__).warning(
                "LLM_PROVIDER=anthropic is from an older version of this "
                "application and no longer exists; using bedrock. Set it to "
                "'bedrock' or 'openrouter' to say which you meant."
            )
            return "bedrock"
        return value

    @model_validator(mode="after")
    def _openrouter_default_needs_openrouter(self) -> "Settings":
        """Refuse the one combination that cannot work.

        `LLM_PROVIDER=openrouter` with `OPENROUTER_ENABLED=false` is a
        deployment configured to default to a provider it has forbidden. Every
        model call would then fail, one at a time, with a message about a flag
        -- so it is said once, at startup, where a person is looking.
        """
        if self.llm_provider == "openrouter" and not self.openrouter_enabled:
            raise ValueError(
                "LLM_PROVIDER is 'openrouter' but OPENROUTER_ENABLED is false, so "
                "nothing could run. Enable OpenRouter, or set LLM_PROVIDER=bedrock."
            )
        return self

    @property
    def openrouter_available(self) -> bool:
        """Whether an OpenRouter call may be made at all.

        The one predicate every path asks, so "disabled" and "no key" cannot
        drift into two different answers in two different files.
        """
        return bool(self.openrouter_enabled and self.openrouter_api_key)

    @field_validator("cors_origins", "bedrock_models", mode="before")
    @classmethod
    def _csv(cls, value):  # noqa: ANN001 - pydantic hook
        return _split_csv(value)

    @field_validator(
        "aws_region", "aws_profile", mode="before"
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
