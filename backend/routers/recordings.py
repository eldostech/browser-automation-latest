"""Recording a workflow by hand, and turning it into a draft use case.

The subprocess is :mod:`recorder`'s job and the parse is :mod:`codegen`'s. What
lives here is the HTTP shape and the one piece of judgement in between:
deciding which recorded steps are *setup* and which are *per-row*.

**Why that split cannot be skipped.** A batch shares one browser session, so
signing in must happen once and not once per row. A flat list of steps cannot
express that, and getting it wrong is not a small mistake -- it is a thousand
sign-ins, or a thousand rows that were never signed in.

**The heuristic, and why it is not a guess.** Everything up to and including
the last step that types a *declared secret* is setup. That is not pattern
matching on the word "login": a credential is the thing a person only supplies
once, and the user has just told us which values those are. Everything after is
row work. A reviewer can move the boundary before publishing, which is the same
gate every draft passes through.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request

from auth.rbac import Permission
from auth.service import Principal
from codegen import Recording, urls_of
# The parser's own ladder builder: a pointed-at element deserves the same
# semantic-then-weak fallback chain every recorded action gets.
from codegen import _ladder  # noqa: PLC2701
from deps import WorkspaceData, require
from fields import FieldSet, parameterise
from recorder import Recorder, RecorderUnavailable
from routers.schemas import SaveRecordingRequest, StartRecordingRequest
from usecase import Assertion, InputSpec, Locator, SecretSpec, Step, UseCase

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


def get_recorder(request: Request) -> Recorder:
    return request.app.state.recorder


RecorderDep = Annotated[Recorder, Depends(get_recorder)]


@router.post("", status_code=201)
async def start_recording(
    body: StartRecordingRequest,
    recorder: RecorderDep,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    """Open a browser window and record what the user does in it."""
    try:
        session = await recorder.start(
            start_url=str(body.start_url),
            name=body.name or "Untitled recording",
            workspace_id=principal.workspace_id,
            owner_id=principal.user_id,
            owner_email=principal.email,
        )
    except RecorderUnavailable as exc:
        # 501 rather than 503: this deployment does not implement recording, and
        # retrying will not change that.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return session.to_dict()


@router.get("")
async def list_recordings(
    recorder: RecorderDep,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
) -> dict[str, Any]:
    return {
        "available": recorder.available()[0],
        "reason": recorder.available()[1],
        "recordings": [s.to_dict() for s in recorder.list(principal.workspace_id)],
    }


@router.get("/{recording_id}")
async def get_recording(
    recording_id: str,
    recorder: RecorderDep,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
) -> dict[str, Any]:
    session = recorder.get(recording_id, principal.workspace_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No such recording.")
    return session.to_dict()


@router.post("/{recording_id}/cancel")
async def cancel_recording(
    recording_id: str,
    recorder: RecorderDep,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    if not await recorder.cancel(recording_id, principal.workspace_id):
        raise HTTPException(status_code=409, detail="That recording is not in progress.")
    return {"recording_id": recording_id, "cancelled": True}


@router.delete("/{recording_id}")
async def discard_recording(
    recording_id: str,
    recorder: RecorderDep,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    if not recorder.discard(recording_id, principal.workspace_id):
        raise HTTPException(status_code=404, detail="No such recording.")
    return {"discarded": recording_id}


@router.post("/{recording_id}/save", status_code=201)
async def save_recording(
    recording_id: str,
    body: SaveRecordingRequest,
    data: WorkspaceData,
    recorder: RecorderDep,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    """Turn a finished recording into a draft use case.

    A draft, never a published one: the split between setup and row steps is a
    heuristic, the parameterisation is derived from values the user typed, and
    both deserve a person's eye before anything runs a thousand times. That is
    the same gate a distilled recording passes through.
    """
    session = recorder.get(recording_id, principal.workspace_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No such recording.")
    if session.status != "ready" or session.recording is None:
        raise HTTPException(
            status_code=409,
            detail=f"That recording is {session.status}, so there is nothing to save yet.",
        )

    declared = FieldSet.from_payload([f.model_dump() for f in body.fields])
    try:
        use_case = build_usecase(
            session.recording,
            name=body.name or session.name,
            description=body.description,
            declared=declared,
            extractions={c.line: c.name for c in body.extractions},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    usecase_id, version = await data.save_usecase(
        use_case.model_dump(mode="json"), created_by=principal.email
    )
    await data.audit(
        "usecase.record",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={
            "recording_id": recording_id,
            "steps": len(use_case.all_steps),
            "unsupported": len(session.recording.unsupported),
        },
    )
    recorder.discard(recording_id, principal.workspace_id)
    log.info(
        "recording saved as a use case",
        extra={"usecase_id": usecase_id, "version": version},
    )
    return {"usecase_id": usecase_id, "version": version, "status": use_case.status}


#: The actions whose value lands in ``Recording.typed``, in the same order.
#: The nth such step is the nth typed value, which is what lets a field name a
#: position rather than a value. Keep in step with ``codegen._consume``.
_RECORDS_A_TYPED_VALUE = frozenset({"fill", "select"})


def _refuse_ambiguous_values(declared: FieldSet) -> None:
    """Two fields, one recorded value, and no way to tell which step is which.

    Only reachable from a client that sends no positions. The alternative is
    what used to happen: one of them silently wins both steps.
    """
    seen: dict[str, str] = {}
    for field in declared.fields:
        if not field.value:
            continue
        first = seen.get(field.value)
        if first is not None:
            raise ValueError(
                f"{first!r} and {field.name!r} were both given the same recorded "
                "value, so there is no way to tell which step belongs to which. "
                "Re-record with a different value in each, or declare only one "
                "of them."
            )
        seen[field.value] = field.name


def build_usecase(
    recording: Recording,
    *,
    name: str,
    description: str,
    declared: FieldSet,
    extractions: dict[int, str] | None = None,
) -> UseCase:
    """A draft use case from a parsed recording and the values the user named."""
    substitutions = {
        field.value: field.template for field in declared.fields if field.value
    }
    # Where the client said *which* typed value each field names, use that.
    # Matching on the value itself cannot tell two fields apart when the same
    # text was typed into both -- a name and a description that both read
    # "test" collapse onto whichever was declared last, so one step gets the
    # wrong template and the other field ends up declared but unread. That
    # surfaces much later, as a publish refusing an input nothing references.
    by_position = {
        field.index: field.template
        for field in declared.fields
        if field.index is not None
    }
    if not by_position:
        _refuse_ambiguous_values(declared)

    steps: list[Step] = []
    typed_so_far = 0
    for step in recording.steps:
        clone = step.model_copy(deep=True)
        if clone.action in _RECORDS_A_TYPED_VALUE and clone.value is not None:
            position = typed_so_far
            typed_so_far += 1
            if by_position:
                # A position with no field declared for it keeps the value that
                # was recorded, which is what leaving it undeclared means.
                template = by_position.get(position)
                if template is not None:
                    clone.value = template
            else:
                clone.value = parameterise(clone.value, substitutions)
        elif clone.value is not None:
            clone.value = parameterise(clone.value, substitutions)
        if clone.url is not None:
            clone.url = parameterise(clone.url, substitutions)
        steps.append(clone)

    # The recording holds the addresses of the environment it was made in. Bind
    # them to {{env.base_url}} so the same document runs in dev, UAT and
    # production, and keep the recorded origin on the use case so dev needs no
    # configuration at all. See §8.4 of the design document.
    origin = _origin_of(recording.start_url or "")
    if origin:
        for step in steps:
            if step.url:
                step.url = _bind_origin(step.url, origin)

    # Elements the person pointed at and named become reading steps, placed
    # where they were pointed at rather than appended. A value read after the
    # browser has moved on is read from the wrong page, which is the failure
    # this ordering exists to avoid.
    named = extractions or {}
    outputs: list[str] = []
    if named:
        for captured in sorted(recording.captured, key=lambda c: -c.after_step):
            column = (named.get(captured.line) or "").strip()
            if not column:
                continue
            steps.insert(
                captured.after_step,
                Step(
                    id=f"x{captured.line}",
                    action="extract",
                    output=column,
                    locators=_ladder_for(captured.locator),
                    # An input holds its text in `value`, and inner_text on one
                    # returns nothing at all -- so which button was used decides
                    # how it has to be read.
                    attribute="value" if captured.kind == "value" else "",
                ),
            )
            outputs.append(column)

    setup, row = _split(steps, declared)

    # A field that is not a secret becomes a per-row input, and the sign-in
    # steps become setup -- so a non-secret value typed while signing in lands
    # in setup as {{input.x}}, which setup cannot have: it runs once for the
    # whole batch, and a per-row value has no meaning there.
    #
    # pydantic catches this, but it reports it against a generated step id and
    # leaves the person to work out which of the fields they declared caused
    # it. Naming the field, and what to do about it, is the whole difference
    # between a message that helps and one that does not.
    stranded = sorted(
        {name for step in setup for kind, name in step.references() if kind == "input"}
    )
    if stranded:
        named = ", ".join(repr(name) for name in stranded)
        raise ValueError(
            f"{named} was typed while signing in, and signing in happens once for the "
            "whole batch rather than once per row -- so it cannot be a per-row input. "
            "Mark it as a secret: that is the bucket for values supplied once per run, "
            "whether or not they are confidential, and a sign-in identity is one. "
            "Alternatively, leave it undeclared to keep the value that was recorded."
        )

    # Assertions go at the end of the row work, which is where codegen writes
    # them: a person records a check after doing the thing they are checking.
    for index, check in enumerate(recording.assertions):
        row.append(Step(id=f"a{index + 1}", action="assert", **{"assert": check}))

    return UseCase(
        name=name.strip()[:200] or "Untitled recording",
        description=description,
        status="draft",
        base_url=origin,
        # Named after the host it was recorded on, so a use case arrives
        # already pointing at something a person can recognise and repoint.
        # Each deployment answers this name with its own address; see
        # routers/targets.py.
        target=_target_name(origin),
        # The recorded origin becomes the template too, so promotion never has
        # to remember to widen the allowlist -- and never widens it to *both*
        # environments, which is one bad input away from acting on the wrong
        # one. Other hosts the recording visited stay literal: they are
        # third parties, and they do not move between environments.
        allowed_domains=[
            "{{env.base_url}}" if _same_host(host, origin) else host
            for host in urls_of(recording)
        ],
        inputs=[
            InputSpec(name=field.name, description=field.description)
            for field in declared.fields
            if not field.secret
        ],
        secrets=[
            SecretSpec(name=field.name, description=field.description)
            for field in declared.fields
            if field.secret
        ],
        setup_steps=setup,
        row_steps=row,
        # Declared in the order they were pointed at, which is the order the
        # results file gets its columns in.
        outputs=list(reversed(outputs)),
        warnings=_warnings(recording, setup),
        dropped=[
            f"line {item.line}: {item.source} -- {item.reason}"
            for item in recording.unsupported
        ],
    )


def _target_name(origin: str) -> str:
    """A handle for the site this was recorded on.

    The registrable-looking part of the host, so ``https://dev.schemora.ai``
    and ``https://uat.schemora.ai`` both suggest ``schemora`` -- the same
    target, which is the whole point: one name, one address per deployment.
    """
    host = (urlparse(origin).hostname or "").lower()
    if not host:
        return ""
    # A loopback recording has no site to name. "localhost" is where the
    # browser was, not what it was looking at, and naming a target after it
    # forces every locally recorded use case to be pointed somewhere before it
    # will run -- against the machine it was just recorded on. Empty means "the
    # URL in the recording", which is exactly right here.
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".localhost"):
        return ""
    parts = [p for p in host.split(".") if p]
    # Drop a leading environment-ish label and the public suffix, leaving the
    # name people actually use for the site.
    if len(parts) >= 3 and parts[0] in {"dev", "uat", "test", "stage", "staging", "qa", "www"}:
        parts = parts[1:]
    stem = parts[0] if parts else host
    return "".join(c for c in stem if c.isalnum() or c in "-_")[:64]



def _ladder_for(locator: Locator) -> list[Locator]:
    """A pointed-at element as a locator ladder.

    The recorder gives one locator, and one locator is brittle. `_ladder` is
    what the parser already builds for every recorded action -- a semantic rung
    first, a weaker one behind it -- and a reading step deserves the same.
    """
    return _ladder(locator)



def _origin_of(url: str) -> str:
    """Scheme and host, which is the part that changes between environments."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _bind_origin(url: str, origin: str) -> str:
    """Replace the recorded origin with the template, leaving the path alone.

    Only an exact origin match is rewritten. A recording that also visits an
    identity provider or a third-party host keeps those URLs literal, because
    they are not the thing being promoted.
    """
    if url.startswith(origin):
        return "{{env.base_url}}" + url[len(origin) :]
    return url


def _same_host(host: str, origin: str) -> bool:
    return bool(origin) and host.lower() == (urlparse(origin).hostname or "").lower()


def _split(steps: list[Step], declared: FieldSet) -> tuple[list[Step], list[Step]]:
    """Setup steps and row steps.

    The boundary is the last step that types a declared secret. Signing in is
    the part a shared session does once, and a credential is exactly the value
    a person supplies once -- so the user has already told us where the line
    is, without being asked a question they would find hard to answer.

    With no secrets declared, nothing is setup: a workflow that does not sign
    in has no per-session work, and putting the first few steps there anyway
    would mean skipping them on every row but the first.
    """
    secret_templates = {f.template for f in declared.fields if f.secret}
    if not secret_templates:
        return [], steps

    boundary = -1
    for index, step in enumerate(steps):
        if step.value and any(template in step.value for template in secret_templates):
            boundary = index

    if boundary < 0:
        return [], steps

    # The submit that follows the password belongs with the sign-in, not with
    # the first row's work. Anything up to the next navigation or the next
    # click is close enough, and a reviewer moves the line if it is not.
    end = boundary + 1
    if end < len(steps) and steps[end].action == "click":
        end += 1
    return steps[:end], steps[end:]


def _warnings(recording: Recording, setup: list[Step]) -> list[str]:
    warnings: list[str] = []
    if recording.unsupported:
        warnings.append(
            f"{len(recording.unsupported)} recorded line(s) could not be represented and "
            "were dropped. Check the list before publishing."
        )
    if not recording.assertions:
        warnings.append(
            "Nothing verifies that a row succeeded. Without a check, a batch of a "
            "thousand rows can fail silently on row 12 and report success on all of "
            "them -- add an assertion before publishing."
        )
    if not setup:
        warnings.append(
            "Every step runs once per row. If this workflow signs in, declare the "
            "login as a secret so it runs once per session instead."
        )
    return warnings
