"""What an agent session may spend, and the moment it stops.

An agent loop is the easiest way to spend a lot of money in this product, and
the only honest defence is a limit checked *before* each thing that costs
rather than reported after it. Four of them, because they fail differently:

``steps``
    Calls made. Catches the loop that is making progress in its own opinion
    and none in anybody else's -- snapshot, click, snapshot, click.
``tokens``
    What the model consumed. The one that actually maps to the bill.
``seconds``
    Wall clock. Catches the page that never finishes loading, which spends no
    tokens at all and would otherwise hold a browser open until the process
    dies.
``usd``
    Tokens priced. Kept separate from ``tokens`` because it is the number a
    person budgets in, and because the rate is a property of the model rather
    than of the run.

Exhausting a budget is a **stop, not a crash**: the session ends, says which
limit it hit, and keeps everything it did. A trajectory that stopped at the
limit is still worth distilling -- often it is a complete recording and the
agent was merely about to tidy up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pricing import price_of

# Pricing lives in `pricing.py`, not here. The healer spends tokens too, and
# it cannot import this package -- the agent is an optional dependency group
# and a replay must work without it installed.
class BudgetExhausted(RuntimeError):
    """A limit was reached. Carries which one, because it is the whole answer."""

    def __init__(self, limit: str, message: str) -> None:
        super().__init__(message)
        self.limit = limit


@dataclass(slots=True)
class Budget:
    """The caps. Every field is optional; None means "no limit of this kind".

    The defaults are deliberately small. A person who wants a longer session
    can say so, and the failure mode of a generous default is a bill.
    """

    steps: int | None = 40
    tokens: int | None = 120_000
    seconds: float | None = 600.0
    usd: float | None = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "seconds": self.seconds,
            "usd": self.usd,
        }


@dataclass(slots=True)
class Spend:
    """What has been used so far, and the check that stops it going further."""

    budget: Budget = field(default_factory=Budget)
    steps: int = 0
    tokens: int = 0
    usd: float = 0.0
    llm_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # -- recording ----------------------------------------------------------
    def step(self) -> None:
        self.steps += 1

    def turn(self, usage: dict[str, int], model: str = "") -> None:
        """One model turn: its tokens, and what they cost."""
        self.llm_calls += 1
        self.tokens += int(usage.get("input_tokens") or 0) + int(
            usage.get("output_tokens") or 0
        )
        self.usd += price_of(model, usage)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    # -- the check ----------------------------------------------------------
    def exceeded(self) -> tuple[str, str] | None:
        """Which limit has been reached, and what to tell the person.

        Checked before each model call and each tool call rather than after,
        so the limit is a ceiling rather than a thing noticed on the way past.
        """
        caps = self.budget
        if caps.steps is not None and self.steps >= caps.steps:
            return (
                "steps",
                f"Stopped after {self.steps} steps, the limit for this session. "
                "If it was still making progress, raise the step budget and run "
                "it again from here.",
            )
        if caps.tokens is not None and self.tokens >= caps.tokens:
            return (
                "tokens",
                f"Stopped at {self.tokens:,} tokens, the limit for this session.",
            )
        if caps.usd is not None and self.usd >= caps.usd:
            return (
                "usd",
                f"Stopped at ${self.usd:.2f}, the spending limit for this session.",
            )
        if caps.seconds is not None and self.elapsed >= caps.seconds:
            return (
                "seconds",
                f"Stopped after {int(self.elapsed)}s, the time limit for this "
                "session. A page that never finishes loading spends no tokens "
                "at all, which is why this limit exists separately.",
            )
        return None

    def check(self) -> None:
        """Raise if a limit has been reached."""
        hit = self.exceeded()
        if hit is not None:
            raise BudgetExhausted(*hit)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "usd": round(self.usd, 4),
            "llm_calls": self.llm_calls,
            "seconds": round(self.elapsed, 1),
            "budget": self.budget.as_dict(),
        }


__all__ = ["Budget", "BudgetExhausted", "Spend"]
