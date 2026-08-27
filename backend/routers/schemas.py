"""Request and response bodies shared by more than one router.

Kept in one module so that a shape used by both ``/execute`` and ``/batch``
cannot drift into two subtly different versions.
"""

from __future__ import annotations

from typing import Any, Literal

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


class CreateRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)
    start_url: str | None = None

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
# Execution
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: Bind stored credentials by id, or pass values inline for a one-off.
    credential_id: str | None = None
    secrets: dict[str, str] | None = None
    version: int | None = None
    headless: bool | None = None
    browser: str | None = None


class BatchRequestBody(BaseModel):
    """Rows arrive as CSV text, a base64 .xlsx workbook, or JSON objects."""

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
