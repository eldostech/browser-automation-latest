"""What a batch will cost, before anybody commits to it.

An agent loop over a spreadsheet is the easiest way to spend a lot of money in
this product. The defence is a limit checked before every call, which phase H
built -- but a limit is what stops a mistake, and an estimate is what prevents
one. They are different jobs and the estimate is the cheaper of the two.

The numbers here are deliberately coarse. Precision is not what makes an
estimate useful: the difference a person needs to see is between "nothing",
"a few dollars if the site has changed" and "three hundred and twenty dollars",
and those are three orders of magnitude apart. A figure to two decimal places
would imply an accuracy this cannot have and does not need.

**The estimate is honest about what it does not know.** Explore is a range,
not a number, because how many turns a row takes depends on the site. Guided is
a range starting at zero, because the usual answer for a healthy site *is*
zero and quoting an average would misrepresent the common case as the expected
one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pricing import rates_for
from usecase import UseCase, effective_mode

#: Rough token cost of one model turn in each mode. An agent turn carries a
#: page snapshot, which is most of it; a repair carries a candidate list, which
#: is much smaller.
TOKENS_PER_EXPLORE_TURN = 9_000
TOKENS_PER_REPAIR = 4_000

#: Turns a row takes in Explore, low and high. Wide on purpose -- a two-field
#: form and a five-page workflow are both "a row".
EXPLORE_TURNS = (4, 14)

#: How often a Guided row needs a repair at all, low and high. Zero is the
#: usual answer on a site that has not changed, and the whole point of the mode
#: is that it stays zero until something moves.
GUIDED_REPAIR_RATE = (0.0, 0.05)


@dataclass
class Estimate:
    """What a batch is expected to cost, as a range."""

    mode: str
    rows: int
    low_usd: float = 0.0
    high_usd: float = 0.0
    #: A sentence a person reads, which is the part that actually lands.
    note: str = ""
    #: Set when the range is above what the workspace has left this month.
    over_budget: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rows": self.rows,
            "low_usd": round(self.low_usd, 2),
            "high_usd": round(self.high_usd, 2),
            "note": self.note,
            "over_budget": self.over_budget,
        }


def estimate_batch(
    usecase: UseCase,
    rows: int,
    *,
    model: str = "",
    healing_enabled: bool = True,
    remaining_usd: float | None = None,
) -> Estimate:
    """What running ``rows`` rows of this use case is likely to cost."""
    mode = effective_mode(usecase.mode, healing_enabled=healing_enabled)
    read, written = rates_for(model)
    # Output tokens are a small share of an agent turn -- the snapshot going in
    # dwarfs the tool call coming out -- so a blended rate weighted towards
    # input is closer than either rate alone.
    per_token = (read * 0.85 + written * 0.15) / 1_000_000

    if mode == "strict":
        return Estimate(
            mode=mode,
            rows=rows,
            note=(
                "Nothing. Strict cannot reach a model at all, so this is a "
                "property of the code rather than an estimate."
            ),
        )

    if mode == "guided":
        low = rows * GUIDED_REPAIR_RATE[0] * TOKENS_PER_REPAIR * per_token
        high = rows * GUIDED_REPAIR_RATE[1] * TOKENS_PER_REPAIR * per_token
        return Estimate(
            mode=mode,
            rows=rows,
            low_usd=low,
            high_usd=high,
            note=(
                "Nothing unless something on the site has moved. Guided pays "
                "only for the rows that break, and on a site that has not "
                f"changed that is none of them -- at worst about ${high:.2f}."
            ),
            over_budget=_over(high, remaining_usd),
        )

    low = rows * EXPLORE_TURNS[0] * TOKENS_PER_EXPLORE_TURN * per_token
    high = rows * EXPLORE_TURNS[1] * TOKENS_PER_EXPLORE_TURN * per_token
    note = (
        f"Roughly ${low:.0f} to ${high:.0f}. Explore works every row out from "
        "the page, so this is paid on all "
        f"{rows:,} of them."
    )
    if rows > 50:
        note += (
            " Consider running twenty rows this way, saving what it learns as "
            "a use case, and running the rest in Strict for nothing."
        )
    return Estimate(
        mode=mode,
        rows=rows,
        low_usd=low,
        high_usd=high,
        note=note,
        over_budget=_over(high, remaining_usd),
    )


def _over(high: float, remaining: float | None) -> bool:
    return remaining is not None and high > remaining


__all__ = [
    "EXPLORE_TURNS",
    "GUIDED_REPAIR_RATE",
    "Estimate",
    "estimate_batch",
]
