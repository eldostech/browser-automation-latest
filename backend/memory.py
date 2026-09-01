"""What broke before, and what fixed it.

Healing asks a model to find a control again. This is what makes it ask less
often: a fix confirmed once is recalled the next time the same site breaks the
same way, so a redesign costs one model call across every workflow that hits
it rather than one per workflow per row.

Three rules, and they are the whole design
------------------------------------------
**Scoped to a workspace.** Retrieval goes through :class:`WorkspaceStore`, not
:class:`Store`. A tenant must not be shown another tenant's selectors — they
describe the shape of another company's internal tooling — and the scoped
object has no method that can reach across the boundary.

**Domain-filtered before it is ranked.** Nearest-neighbour over every fix ever
recorded will cheerfully return a plausible-looking button from an unrelated
site. ``domain`` is a hard ``WHERE``; the vector distance only orders what
survives it.

**Never authoritative.** What comes back is *context in a prompt*. The model
still chooses from the controls present on the page now, and the choice is
still validated against them. A stale or poisoned memory can make healing
worse; it cannot make it unsafe. That is the same invariant distillation had
and the reason a hallucinated selector has no route into a use case.

What is worth remembering
-------------------------
Only a fix somebody stands behind: one the model proposed with high confidence
and that then *worked*, or one a person confirmed. Writing down every attempt
would fill the table with the guesses that failed, and those are exactly the
answers not to give next time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from embeddings import Embedder, page_signature

log = logging.getLogger(__name__)

#: How many past fixes to put in front of the model. Enough to show a pattern,
#: few enough that they do not crowd out the page itself.
TOP_K = 5

#: Cosine distance beyond which a "match" is not one. Two pages from the same
#: site share so much chrome that almost anything scores somewhat close, and an
#: unrelated fix in the prompt is worse than none.
MAX_DISTANCE = 0.45

_HOST = re.compile(r"^[a-z][a-z0-9+.\-]*://([^/:]+)", re.IGNORECASE)


def domain_of(url: str | None) -> str:
    """The host a failure happened on, which is what scopes recall."""
    if not url:
        return ""
    match = _HOST.match(url)
    return match.group(1).lower() if match else ""


@dataclass(slots=True)
class PastFix:
    """One remembered repair, as the prompt will see it."""

    step_id: str
    explanation: str
    old_locator: dict[str, Any] | None
    new_locator: dict[str, Any] | None
    confirmed_by: str
    distance: float

    @property
    def human(self) -> bool:
        """Whether a person confirmed this, rather than the model alone."""
        return bool(self.confirmed_by) and self.confirmed_by != "model"

    def describe(self) -> str:
        old = (self.old_locator or {}).get("name") or (self.old_locator or {}).get("selector") or "?"
        new = (self.new_locator or {}).get("name") or (self.new_locator or {}).get("selector") or "?"
        who = "confirmed by a person" if self.human else "applied automatically"
        return f"- {old!r} became {new!r} ({who}). {self.explanation}".strip()


class HealingMemory:
    """Reads and writes the record of past fixes for one workspace."""

    def __init__(self, store: Any, embedder: Embedder | None) -> None:
        self.store = store
        self.embedder = embedder

    @property
    def available(self) -> bool:
        return self.embedder is not None

    async def _embed(
        self, step_summary: str, wanted: str, page_url: str, page: str
    ) -> list[float] | None:
        """The page as a vector, or None.

        The embedder is a network call to somebody else's service. It can be
        slow, rate-limited, or refused by an account that lacks the model --
        and none of that is a reason to fail a row, because everything this
        module does is an optimisation over healing that already worked without
        it. So a raised exception is caught here and reads as "no vector".
        """
        if self.embedder is None:
            return None
        try:
            return await self.embedder.embed(
                page_signature(step_summary, wanted, page_url, page)
            )
        except Exception:  # noqa: BLE001 - an optimisation, never a failure
            log.debug("could not embed the page", exc_info=True)
            return None

    async def recall(
        self, *, step_summary: str, wanted: str, page_url: str, page: str
    ) -> list[PastFix]:
        """Fixes made before on this domain for something like this failure.

        Returns an empty list for every ordinary reason -- no embedder, nothing
        recorded yet, the embedding call failing. Retrieval is an optimisation,
        so its failure mode is "heal as before", not "fail the row".
        """
        domain = domain_of(page_url)
        if not self.available or not domain:
            return []

        vector = await self._embed(step_summary, wanted, page_url, page)
        if vector is None:
            return []

        try:
            rows = await self.store.similar_fixes(
                domain=domain, embedding=vector, limit=TOP_K
            )
        except Exception:  # noqa: BLE001 - an optimisation, never a failure
            log.debug("could not read healing memory", exc_info=True)
            return []

        fixes = [
            PastFix(
                step_id=row["step_id"],
                explanation=row["explanation"],
                old_locator=row["old_locator"],
                new_locator=row["new_locator"],
                confirmed_by=row["confirmed_by"],
                distance=row["distance"],
            )
            for row in rows
            if row["distance"] <= MAX_DISTANCE
        ]
        if fixes:
            log.info(
                "recalled past fixes",
                extra={"domain": domain, "count": len(fixes)},
            )
        return fixes

    async def remember(
        self,
        *,
        usecase_id: str | None,
        step_id: str,
        page_url: str,
        page: str,
        step_summary: str,
        wanted: str,
        old_locator: dict[str, Any] | None,
        new_locator: dict[str, Any] | None,
        explanation: str,
        confirmed_by: str,
        error_kind: str = "not_found",
    ) -> None:
        """Record a fix that worked. Never raises into the run that made it."""
        domain = domain_of(page_url)
        if not self.available or not domain:
            return

        vector = await self._embed(step_summary, wanted, page_url, page)
        if vector is None:
            return

        try:
            await self.store.remember_fix(
                usecase_id=usecase_id,
                domain=domain,
                step_id=step_id,
                error_kind=error_kind,
                dom_context=page[:4000],
                old_locator=old_locator,
                new_locator=new_locator,
                explanation=explanation,
                confirmed_by=confirmed_by,
                embedding=vector,
            )
        except Exception:  # noqa: BLE001
            log.debug("could not write healing memory", exc_info=True)
            return
        log.info(
            "remembered a fix",
            extra={"domain": domain, "step_id": step_id, "by": confirmed_by},
        )


def as_prompt(fixes: list[PastFix]) -> str:
    """Past fixes as the lines that go into the prompt.

    Human-confirmed ones first regardless of distance: somebody looked at that
    one and said yes, which outranks a closer match nobody checked.
    """
    if not fixes:
        return "(nothing similar has been fixed on this site before)"
    ordered = sorted(fixes, key=lambda fix: (not fix.human, fix.distance))
    return "\n".join(fix.describe() for fix in ordered)


__all__ = ["HealingMemory", "MAX_DISTANCE", "PastFix", "TOP_K", "as_prompt", "domain_of"]
