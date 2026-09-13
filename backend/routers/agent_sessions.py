"""Recording a workflow by describing it, over HTTP.

Deliberately the same shape as ``recordings.py``: start, watch, save. Two ways
of producing the same artifact should not be two products, and the review
screen a draft lands in is the same one either way.

What is different is what happens in the middle. A codegen recording is a
person doing the work; an agent session is a model doing it, which means it
costs money, can be stopped, and sometimes has to ask. So this router has three
things the other does not: a budget on the way in, a decision endpoint, and a
verification report on the way out.

The live view needs nothing new here. Agent sessions write to the same
``events`` table with the same sequence numbers, so the existing
``/api/runs/{run_id}/stream`` shows one exactly as it shows a replay.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request

from agent import Budget
from agent.manager import AgentSessions, AgentUnavailable
from auth.rbac import Permission
from llm import ModelChoice
from auth.service import Principal
from credentials import Vault
from deps import WorkspaceData, get_vault, require
from services import resolve_secrets
from redaction import scrub_credentials
from routers.schemas import (
    AgentDecisionRequest,
    SaveAgentSessionRequest,
    StartAgentSessionRequest,
)
from usecase import TargetMissing, UseCase, resolve_base_url

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent-sessions", tags=["agent"])


def get_sessions(request: Request) -> AgentSessions:
    return request.app.state.agent_sessions


SessionsDep = Annotated[AgentSessions, Depends(get_sessions)]
VaultDep = Annotated[Vault, Depends(get_vault)]


@router.post("", status_code=201)
async def start_session(
    body: StartAgentSessionRequest,
    request: Request,
    sessions: SessionsDep,
    data: WorkspaceData,
    vault: VaultDep,
    principal: Annotated[Principal, Depends(require(Permission.AGENT_AUTHOR))],
) -> dict[str, Any]:
    """Start an agent working on a task, in a browser you can watch."""
    start_url = await _start_url(body, data)
    if not start_url:
        raise HTTPException(
            status_code=422,
            detail="An agent needs somewhere to start: give a target or an address.",
        )

    # Deny-by-default, and derived rather than asked for. A person describing a
    # task should not also have to write an allowlist, and one they typed by
    # hand would be the thing that is wrong when a session goes somewhere
    # unexpected.
    allowed = _host(start_url)
    # The same resolution a replay uses, so an agent session and a batch bind
    # a credential the same way and there is one place that decrypts. What
    # reaches the model is the slot name; the value is substituted by the tool
    # layer at the moment of typing and redacted out of every event.
    secrets = await resolve_secrets(body, data, vault)

    # The other half of keeping credentials out of the log, and the half no
    # redactor could do: a value nobody declared. A task pasted in as "User: x
    # / Password : y" becomes the run row, the `run_started` event, the use
    # case's own description and the first message sent to a model, and none of
    # those know that string is a password. So it is recognised here, before
    # the first write, and replaced -- with `{{secret.slot}}` when a bound
    # credential has a slot for it, which is the literal the recorder is
    # already instructed to type.
    scrubbed = scrub_credentials(body.task.strip(), slots=secrets.keys())
    if scrubbed.secrets and not secrets:
        # Refused rather than started, because this session could not sign in
        # anyway: the value is not going to be stored or sent, and there is no
        # credential bound to use instead. Better to say so now than to spend a
        # budget discovering it.
        raise HTTPException(
            status_code=422,
            detail=(
                "This task has a "
                + ", ".join(scrubbed.labels)
                + " typed into it. Credentials do not belong in a task: the text is "
                "stored with the run, shown in the timeline and sent to the model. "
                "Save it once under Credentials, pick it under 'Sign in as', and "
                "write the task without it -- the recorder is given the slot names "
                "and types the values itself."
            ),
        )

    try:
        session = await sessions.start(
            task=scrubbed.text,
            start_url=start_url,
            allowed_domains=(allowed,),
            workspace_id=principal.workspace_id,
            may_write=body.may_write,
            headless=body.headless,
            budget=Budget(
                steps=body.budget_steps,
                tokens=body.budget_tokens,
                seconds=body.budget_seconds,
                usd=body.budget_usd,
            ),
            secrets=secrets,
            sample=body.sample,
            name=body.name.strip(),
            owner_id=principal.user_id,
            owner_email=principal.email,
            model=_chosen_model(request, body),
        )
    except AgentUnavailable as exc:
        # 501, like the recorder answers on a machine with no display: this
        # deployment cannot do it, which is not the caller's mistake.
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    await data.audit(
        "agent.session.start",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="agent_session",
        resource_id=session.id,
        detail={
            "task": scrubbed.text[:200],
            "may_write": body.may_write,
            "credentials_removed": scrubbed.labels or None,
        },
    )
    summary = session.summary()
    if scrubbed.found:
        # Said back, because a person who typed a password into a task has a
        # habit to change and a value to rotate. Silently cleaning it up would
        # leave them believing it was never stored anywhere.
        summary["notice"] = (
            "A credential typed into the task was replaced with the bound "
            "credential's slots before anything was stored. Nothing here kept the "
            "value, but whatever you pasted it from still has it."
        )
    return summary


@router.get("")
async def list_sessions(
    sessions: SessionsDep,
    principal: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> dict[str, Any]:
    return {"sessions": [s.summary() for s in sessions.list(principal.workspace_id)]}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    sessions: SessionsDep,
    principal: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> dict[str, Any]:
    session = sessions.get(session_id, principal.workspace_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No such session.")
    return session.summary()


@router.post("/{session_id}/decide")
async def decide(
    session_id: str,
    body: AgentDecisionRequest,
    sessions: SessionsDep,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.RUN_APPROVE))],
) -> dict[str, Any]:
    """Answer the question a suspended session is waiting on.

    RUN_APPROVE rather than AGENT_AUTHOR: deciding whether an irreversible
    action may happen is the operator's call, and it is deliberately not the
    same authority as starting the session. The permission has existed since
    the first agent and this is the first thing to use it.
    """
    if not await sessions.decide(session_id, principal.workspace_id, body.decision):
        raise HTTPException(
            status_code=409,
            detail="That session is not waiting for a decision.",
        )
    await data.audit(
        "agent.session.decide",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="agent_session",
        resource_id=session_id,
        detail={"decision": body.decision},
    )
    return {"session_id": session_id, "decision": body.decision}


@router.post("/{session_id}/cancel")
async def cancel(
    session_id: str,
    sessions: SessionsDep,
    principal: Annotated[Principal, Depends(require(Permission.RUN_CANCEL))],
) -> dict[str, Any]:
    stopped = await sessions.cancel(session_id, principal.workspace_id)
    if not stopped:
        raise HTTPException(status_code=409, detail="That session is not running.")
    return {"session_id": session_id, "status": "cancelling"}


@router.post("/{session_id}/save", status_code=201)
async def save_session(
    session_id: str,
    body: SaveAgentSessionRequest,
    sessions: SessionsDep,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    """Turn the distilled draft into a use case.

    A draft, never a published one -- the same gate a codegen recording passes
    through, and for a stronger reason here: a model chose these steps. The
    verification report travels with it so a reviewer sees whether it replays
    before deciding anything.
    """
    session = sessions.get(session_id, principal.workspace_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No such session.")
    if session.result is None or not session.result.use_case:
        raise HTTPException(
            status_code=409,
            detail=(
                f"That session is {session.status} and has produced no draft yet."
            ),
        )

    document = {**session.result.use_case}
    if body.name.strip():
        document["name"] = body.name.strip()
    try:
        use_case = UseCase.model_validate(document)
    except Exception as exc:  # noqa: BLE001 - shown to the review UI verbatim
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
            "agent_session_id": session_id,
            "run_id": session.run_id,
            "steps": len(use_case.all_steps),
            "verified": bool(session.result.verification.get("ok")),
            "spend": session.result.spend,
        },
    )
    sessions.discard(session_id, principal.workspace_id)
    log.info(
        "agent session saved as a use case",
        extra={"usecase_id": usecase_id, "version": version, "session_id": session_id},
    )
    return {
        "usecase_id": usecase_id,
        "version": version,
        "status": use_case.status,
        "warnings": session.result.draft_warnings,
        "verification": session.result.verification,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _start_url(body: StartAgentSessionRequest, data: WorkspaceData) -> str:
    """Where to begin: a named target, or an address given for this session.

    A target is preferred for the reason it is preferred everywhere else --
    the deployment says where a site lives, so nothing about the address ends
    up baked into what gets recorded.
    """
    if body.target:
        try:
            return resolve_base_url(
                target=body.target,
                targets=await data.target_urls(),
                recorded="",
                override="",
            )
        except TargetMissing as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return body.start_url.strip()


def _host(url: str) -> str:
    parts = urlsplit(url if "//" in url else f"//{url}")
    return (parts.netloc or url).split("@")[-1].split(":")[0] or url


def _chosen_model(request: Request, body: Any) -> "ModelChoice | None":
    """Which model this session should drive on, or ``None`` for the default."""
    if not body.provider and not body.model:
        return None
    try:
        return ModelChoice.resolve(request.app.state.settings, body.provider, body.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
