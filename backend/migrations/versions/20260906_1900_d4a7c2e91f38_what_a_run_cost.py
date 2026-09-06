"""what a run cost, and what a workspace may spend

Two things in this application spend tokens: an agent authoring session, and a
healer re-finding a control mid-replay. Neither was recorded anywhere that
could be summed.

``runs.tokens`` and ``runs.cost_usd`` are columns rather than fields inside the
existing ``result`` JSON because the question they exist to answer -- what has
this workspace spent this month -- is a SUM, and a SUM over JSONB is a question
nobody asks twice. Zero is the ordinary value and it is measured rather than
assumed: a Strict replay cannot reach a model at all, and recording that as a
number is what lets a dashboard tell "free" apart from "not recorded".

``workspaces.monthly_spend_limit_usd`` is nullable on purpose. "Unlimited" and
"limited to a number somebody chose" are different states, and an installation
that has never thought about spending should not be told it has a budget it did
not set.

Revision ID: d4a7c2e91f38
Revises: c3f9a1d84e26
Create Date: 2026-09-06 19:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4a7c2e91f38"
down_revision: Union[str, None] = "c3f9a1d84e26"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "runs",
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "workspaces",
        sa.Column("monthly_spend_limit_usd", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("workspaces", "monthly_spend_limit_usd")
    op.drop_column("runs", "cost_usd")
    op.drop_column("runs", "tokens")
