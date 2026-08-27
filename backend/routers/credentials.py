"""Stored credentials. Write-only over HTTP: no endpoint returns a value.

Encryption is a static Fernet key from the environment rather than a KMS, by
explicit decision. That means the key sits beside the ciphertext on the same
host, and anyone who can read the environment can read the credentials -- so
the deployment must treat process environment and backups as sensitive. The
data-shape work for envelope encryption is done (values are already opaque
bytes behind one seal/open interface), so moving to a KMS later changes
``credentials.py`` and nothing else.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from auth.rbac import Permission
from auth.service import Principal
from credentials import NO_KEY_MESSAGE, Vault, new_credential_id
from deps import WorkspaceData, get_vault, require
from routers.schemas import CredentialRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/credentials", tags=["credentials"])

VaultDep = Annotated[Vault, Depends(get_vault)]


@router.post("", status_code=201)
async def create_credential(
    body: CredentialRequest,
    data: WorkspaceData,
    vault: VaultDep,
    principal: Annotated[Principal, Depends(require(Permission.CREDENTIAL_WRITE))],
) -> dict[str, Any]:
    if not vault.available:
        raise HTTPException(status_code=503, detail=NO_KEY_MESSAGE)
    try:
        ciphertext = vault.seal(body.values)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    credential_id = await data.save_credential(
        new_credential_id(),
        body.name,
        Vault.slots_of(body.values),
        ciphertext,
        owner_id=principal.user_id,
    )
    await data.audit(
        "credential.save",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="credential",
        resource_id=credential_id,
        detail={"name": body.name, "slots": Vault.slots_of(body.values)},
    )
    log.info(
        "stored a credential",
        extra={"credential_id": credential_id, "slots": len(body.values)},
    )
    return {"id": credential_id, "name": body.name, "slots": Vault.slots_of(body.values)}


@router.get("")
async def list_credentials(
    data: WorkspaceData,
    vault: VaultDep,
    _: Annotated[Principal, Depends(require(Permission.CREDENTIAL_READ))],
) -> dict[str, Any]:
    """Names and slot lists. Never a value."""
    return {"credentials": await data.list_credentials(), "vault_available": vault.available}


@router.delete("/{credential_id}")
async def delete_credential(
    credential_id: str,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.CREDENTIAL_DELETE))],
) -> dict[str, Any]:
    if not await data.delete_credential(credential_id):
        raise HTTPException(status_code=404, detail="No such credential.")
    await data.audit(
        "credential.delete",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="credential",
        resource_id=credential_id,
    )
    return {"id": credential_id, "deleted": True}
