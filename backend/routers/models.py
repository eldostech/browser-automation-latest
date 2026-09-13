"""Which models this deployment can reach, and what they cost.

One endpoint, read by the model picker. It exists because the answer is not
static: OpenRouter fronts several hundred models, changes them weekly, and
quotes its own prices -- so a dropdown built from a constant in the frontend
would be wrong within a fortnight and silent about the cost of being wrong.

Nothing here *sets* anything. A choice of model is not server state: two people
comparing two models at the same time is the whole point, so the choice travels
on the request that uses it (see ``ModelChoice`` and the ``provider``/``model``
fields on the run and session requests). This endpoint only says what is on
offer.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from auth.rbac import Permission
from auth.service import Principal
from deps import require
from llm import PROVIDERS, ModelChoice

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def list_models(
    request: Request,
    _: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
) -> dict[str, Any]:
    """Every model on offer, with its provider, price and context window.

    Behind ``usecase:read`` rather than an admin permission: choosing which
    model to try is ordinary work for anybody allowed to run a use case, and
    this returns no credential and no account detail -- only a catalogue the
    provider publishes openly.

    A provider that cannot be reached comes back under ``problems`` with the
    reason rather than being silently absent. "OpenRouter is missing" with no
    explanation is a gap a person fills in with a guess.
    """
    catalogue = await request.app.state.model_catalogue.read()
    settings = request.app.state.settings
    # Every provider this build knows, including one that is switched off. The
    # picker renders a provider with a stated problem as a disabled tab
    # carrying the reason, which is better than silent absence: "OpenRouter is
    # missing" with no explanation is a gap a person fills in with a guess.
    # The refusal is enforced in `ModelChoice.resolve` and below it, not by
    # hiding the name.
    return {
        "providers": list(PROVIDERS),
        **catalogue.to_dict(ModelChoice.default(settings)),
    }


@router.post("/models/check")
async def check_model(
    body: dict[str, Any],
    request: Request,
    _: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
) -> dict[str, Any]:
    """Can this deployment actually call that model? One tiny request.

    The picker offers hundreds of models and this deployment can reach some of
    them. A key without credit, a model that needs its own provider agreement,
    an id that has been retired -- none of those are visible in a catalogue,
    and all of them look identical to a broken workflow when they surface three
    steps into a run. One "hi" costs a fraction of a cent and answers it.
    """
    settings = request.app.state.settings
    try:
        choice = ModelChoice.resolve(
            settings, body.get("provider"), body.get("model")
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not choice.model:
        raise HTTPException(
            status_code=422,
            detail="Name a model to check; this provider has no default to fall back on.",
        )

    try:
        client = request.app.state.repair_model.for_choice(choice)
    except ValueError as exc:
        # A missing key or a blank model: a configuration problem, reported as
        # the reason rather than as a failed call.
        return {"provider": choice.provider, "model": choice.model, "ok": False, "error": str(exc)}

    result = await client.check_access()
    return {"provider": choice.provider, **result}
