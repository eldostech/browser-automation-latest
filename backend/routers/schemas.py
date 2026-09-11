"""Request and response bodies shared by more than one router.

Kept in one module so that a shape used by both ``/execute`` and ``/batch``
cannot drift into two subtly different versions.
"""

from __future__ import annotations

from typing import Any, Literal

import re

from pydantic import BaseModel, Field, field_validator

from auth.passwords import MIN_PASSWORD_LENGTH


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict[str, Any]


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=200)
    new_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=200)


class CreateUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=200)
    role: Literal["viewer", "operator", "author", "admin"] = "operator"
    display_name: str = Field(default="", max_length=200)


class UpdateUserRequest(BaseModel):
    role: Literal["viewer", "operator", "author", "admin"] | None = None
    is_active: bool | None = None
    new_password: str | None = Field(default=None, min_length=MIN_PASSWORD_LENGTH, max_length=200)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class DeclaredFieldPayload(BaseModel):
    """One named value the recording will use.

    Marking a field `secret` changes three things at once: the model is shown a
    placeholder instead of the value, the value is registered for redaction on
    the way to storage, and the use case gets a credential *slot* rather than
    an input column. See `fields.py`.
    """

    name: str = Field(min_length=1, max_length=64)
    #: Write-only. No endpoint returns this for a secret field, and the value
    #: of a secret is never persisted at all.
    value: str = Field(max_length=4000)
    #: Which of the recording's typed values this field names, as a position in
    #: ``Recording.typed``. Values alone cannot identify a field: type the same
    #: text into two boxes -- a name and a description, say -- and both steps
    #: collapse onto whichever field was declared last, leaving the other
    #: declared but unreferenced. The position tells them apart. Optional, so a
    #: client that does not send it keeps the older value-matching behaviour.
    index: int | None = Field(default=None, ge=0)
    secret: bool = False
    description: str = Field(default="", max_length=300)
    example: str = Field(default="", max_length=200)

    @field_validator("name")
    @classmethod
    def _usable_as_a_column(cls, value: str) -> str:
        stripped = value.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", stripped):
            raise ValueError(
                "a field name becomes a CSV column and a template name, so it must "
                "be letters, digits and underscores, starting with a letter"
            )
        return stripped


class CreateRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)
    start_url: str | None = None

    #: The values the task refers to, named. Keeping them out of the prose is
    #: what lets the recording be parameterised deterministically afterwards --
    #: and what keeps a password from being pasted into a stored task string.
    fields: list[DeclaredFieldPayload] = Field(default_factory=list, max_length=50)

    # Guardrail overrides; anything omitted falls back to the server defaults.
    max_steps: int | None = Field(default=None, ge=1, le=200)
    timeout_seconds: float | None = Field(default=None, ge=10, le=3600)
    allowed_domains: list[str] | None = None
    require_approval: bool | None = None
    screenshot_every_step: bool | None = None

    # Browser overrides.
    headless: bool | None = None
    browser: str | None = None

    #: Values to keep out of the event log, the database and the logs. Anything
    #: listed here is replaced with a placeholder wherever it appears -- in the
    #: task text, in a tool argument, in a tool result echoing it back, or in
    #: the model's own prose. Write-only: never returned by any endpoint.
    secrets: list[str] | None = None

    @field_validator("start_url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("start_url must begin with http:// or https://")
        return value

    @field_validator("allowed_domains")
    @classmethod
    def _clean_domains(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [d.strip() for d in value if d and d.strip()]


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]
    approval_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)


class CreateRunResponse(BaseModel):
    run_id: str
    status: str


# ---------------------------------------------------------------------------
# Use cases
# ---------------------------------------------------------------------------


class RenameRequest(BaseModel):
    """A label change. Deliberately not the definition."""

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        # min_length counts characters, so "   " passes it and then strips to
        # nothing -- leaving a use case with no name at all in the list.
        stripped = value.strip()
        if not stripped:
            raise ValueError("a name cannot be blank")
        return stripped


class ScriptsRequest(BaseModel):
    """Permit, or withdraw permission for, script steps on one use case."""

    enabled: bool
    #: Free text recorded in the audit entry: why this was considered safe.
    reason: str = Field(default="", max_length=1000)


class DistillRequest(BaseModel):
    """Turn a finished recording into a use case.

    `save_credential_as` decides the fate of any credentials the recording
    used. Naming one seals them into the vault under that name and binds the
    slots to the use case; leaving it unset discards them. There is no third
    option -- they are held in memory only until this call resolves.
    """

    name: str | None = Field(default=None, max_length=200)
    save_credential_as: str | None = Field(default=None, max_length=120)


class RepairRequest(BaseModel):
    """Which failure to mend. Either is enough to find the rest."""

    execution_id: str | None = None
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class CredentialRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    #: ``{slot: value}`` matching the use case's declared secrets. Write-only:
    #: no endpoint returns these, and nothing in the dashboard needs them back.
    values: dict[str, str] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Agent tool servers
# ---------------------------------------------------------------------------


class StdioConnection(BaseModel):
    """What it takes to open one stdio MCP server.

    Its own model rather than a bare dict, so a malformed registration is
    refused at the boundary rather than surfacing as a subprocess that will
    not start, three steps into an agent session nobody can debug from there.
    """

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ToolServerRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    #: Only stdio exists today; the field exists so an sse/http server later
    #: is a new value here rather than a new endpoint.
    transport: Literal["stdio"] = "stdio"
    connection: StdioConnection
    enabled: bool = True


class ToolServerPreviewRequest(BaseModel):
    """Same shape as registering one, but nothing is saved -- see the
    ``/preview`` route: this is what a person answers *before* deciding
    whether a server is worth registering at all."""

    transport: Literal["stdio"] = "stdio"
    connection: StdioConnection


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExtractionChoice(BaseModel):
    """One element the person pointed at, and what they want done with it.

    ``line`` identifies which captured element this is about -- it is the line
    of the recording it came from, which is stable and does not depend on the
    client keeping a list in order.
    """

    line: int
    #: The spreadsheet column to put it under. Empty means "not a value" --
    #: keep it as a check, or drop it.
    name: str = Field(default="", max_length=64)


class DatasetFromRunRequest(BaseModel):
    """Make a dataset out of what a discovery run extracted."""

    #: Exactly one of these. A batch, because discovery is often itself a batch
    #: -- one row per page of a paginated list -- and their rows are one list.
    execution_id: str = ""
    batch_id: str = ""
    #: The name the ``extract_rows`` step landed its rows under.
    output: str = Field(min_length=1, max_length=64)
    name: str = Field(default="", max_length=200)


class TargetRequest(BaseModel):
    """The address this deployment gives one target."""

    base_url: str = Field(min_length=1, max_length=2000)
    description: str = Field(default="", max_length=300)


class ExecuteRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: Run against this address instead of the use case's target. For a one-off
    #: against a branch deployment or one customer's tenant, where a standing
    #: target would be ceremony for a single run.
    base_url: str = Field(default="", max_length=2000)
    #: Bind stored credentials by id, or pass values inline for a one-off.
    credential_id: str | None = None
    secrets: dict[str, str] | None = None
    version: int | None = None
    headless: bool | None = None
    browser: str | None = None


class StartRecordingRequest(BaseModel):
    """Open a browser window at a URL and record what happens in it."""

    start_url: str
    name: str = ""


class SaveRecordingRequest(BaseModel):
    """Turn a finished recording into a draft use case.

    ``fields`` names the values that were typed: which become per-row inputs,
    and which are credentials. It is the same payload the distil path takes,
    because it answers the same question about the same kind of recording.
    """

    name: str = ""
    description: str = ""
    fields: list[DeclaredFieldPayload] = []
    #: Which of the pointed-at elements are values to read out, and what to
    #: call each. Anything not named here stays a check, as codegen recorded it.
    extractions: list[ExtractionChoice] = []


class RememberFixRequest(BaseModel):
    """A repair a person worked out, recorded so the next run recalls it.

    ``page`` and ``page_url`` are what the failure looked like: the URL scopes
    recall to a domain, and the page is what gets embedded. Both come back from
    the failure the user is looking at, so the UI does not ask them to type
    anything it already knows.
    """

    step_id: str
    page_url: str
    page: str = ""
    step_summary: str = ""
    wanted: str = ""
    explanation: str = Field(min_length=1, max_length=2000)
    usecase_id: str | None = None
    old_locator: dict[str, Any] | None = None
    new_locator: dict[str, Any] | None = None
    error_kind: str = "not_found"


class MappingRequest(BaseModel):
    """Which dataset to line up against which version of a use case."""

    dataset_id: str
    version: int | None = None


class BatchRequestBody(BaseModel):
    """Rows arrive as an uploaded dataset, CSV text, a base64 .xlsx workbook,
    or JSON objects.

    ``dataset_id`` is the route the UI takes, because it is the only one where
    the columns were profiled and mapped before anything ran. The others remain
    for callers driving the API directly.
    """

    #: An already-uploaded dataset. With it, ``mapping`` says which column
    #: fills which declared input; without a mapping the column names must
    #: already be the field names.
    dataset_id: str | None = None
    mapping: dict[str, str] | None = None
    #: Run every row against this address instead of the use case's target.
    base_url: str = Field(default="", max_length=2000)
    csv: str | None = None
    #: A base64-encoded .xlsx. Spreadsheets are how people actually keep lists
    #: of records, and re-saving one as CSV silently mangles leading zeros,
    #: dates, and anything containing a comma.
    xlsx_base64: str | None = None
    sheet: str | None = None
    rows: list[dict[str, Any]] | None = None
    credential_id: str | None = None
    secrets: dict[str, str] | None = None
    version: int | None = None
    headless: bool | None = None
    browser: str | None = None


__all__ = [
    "ApprovalRequest",
    "BatchRequestBody",
    "MappingRequest",
    "RememberFixRequest",
    "SaveRecordingRequest",
    "StartRecordingRequest",
    "ChangePasswordRequest",
    "CreateRunRequest",
    "CreateRunResponse",
    "CreateUserRequest",
    "CredentialRequest",
    "ExecuteRequest",
    "LoginRequest",
    "LoginResponse",
    "RenameRequest",
    "RepairRequest",
    "ScriptsRequest",
    "UpdateUserRequest",
]


class StartAgentSessionRequest(BaseModel):
    """Start an agent working on a task in a browser.

    Budgets are on the request rather than only in configuration because they
    are a per-session judgement: exploring an unfamiliar site is worth more
    steps than re-recording one somebody already knows.
    """

    task: str = Field(min_length=1, max_length=4000)
    #: Where to start. A target is preferred -- the deployment says where a
    #: site lives, so nothing about the address is baked into what is recorded.
    target: str = ""
    start_url: str = ""
    #: What to call the use case this produces. Optional; the task is used.
    name: str = Field(default="", max_length=200)
    #: A stored credential to bind. The slot names reach the model; the values
    #: never do.
    credential_id: str | None = None
    #: Inline values, for a one-off. Same shape as a single-row execute.
    secrets: dict[str, str] | None = None
    #: One record to work through, so the agent has something concrete to do
    #: and the draft has a row to be verified against.
    sample: dict[str, Any] = Field(default_factory=dict)
    #: Off by default. An agent sent to find out how a form works must not
    #: submit it on the way.
    may_write: bool = False
    #: Show a real window instead of running headless. None follows the
    #: deployment default; a person starting a session chooses for themselves
    #: the same way they choose it for a replay, under "Show the browser while
    #: it runs" -- watching is how trust in this gets built the first few
    #: times, and nobody should have to ask an administrator for that.
    headless: bool | None = None

    budget_steps: int | None = Field(default=40, ge=1, le=500)
    #: A backstop, not the everyday limit -- see agent.budget.Budget. The
    #: frontend does not ask for this; ``budget_usd`` is the number a person
    #: actually sets, and this stays out of its way.
    budget_tokens: int | None = Field(default=400_000, ge=1000)
    budget_seconds: float | None = Field(default=600.0, ge=10, le=3600)
    budget_usd: float | None = Field(default=1.0, ge=0.01, le=100)


class AgentDecisionRequest(BaseModel):
    """A person's answer to a session that stopped to ask."""

    decision: Literal["approved", "rejected"]


class SaveAgentSessionRequest(BaseModel):
    """Turn a finished session's draft into a use case."""

    name: str = Field(default="", max_length=200)


class SpendLimitRequest(BaseModel):
    """The monthly ceiling, or None to remove it.

    None rather than zero for "no limit": zero is a perfectly reasonable
    ceiling to set deliberately, and conflating the two would make "stop all
    spending" unexpressible.
    """

    limit_usd: float | None = Field(default=None, ge=0, le=1_000_000)


class LocatorCheckRequest(BaseModel):
    """Try some locators against a real page and say what each one matches.

    A person editing a locator is otherwise guessing: the rung reads fine and
    only a batch discovers it matched nothing, or matched four things. This is
    the difference between editing a locator and editing a string.
    """

    #: Where to look. Must be inside the use case's own allowlist -- the same
    #: gate a run passes, for the same reason.
    url: str = Field(min_length=1, max_length=2_000)
    #: The ladder as it would be saved. Validated as `Locator` in the handler,
    #: so a malformed rung comes back as a message rather than a 422 on a body
    #: the editor cannot map back to a field.
    locators: list[dict] = Field(min_length=1, max_length=12)
    #: A step's own timeout, so a check on a slow page behaves like the step
    #: it is checking rather than failing faster than the real thing would.
    timeout_ms: int = Field(default=10_000, ge=1_000, le=60_000)
