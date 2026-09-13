"""a use case says how it runs

Whether a run may spend a token was a deployment-wide switch,
``REPLAY_HEALING_ENABLED``, and nobody using the product could see it. That is
the wrong place for the decision twice over: it is invisible, and it is a
property of the *workflow* rather than of the installation. One use case runs
against a site rebuilt every sprint and another against a form that has not
changed in four years; a single variable cannot be right for both.

``mode`` moves that choice onto the use case. It is deliberately nullable --
stored as the empty string rather than defaulting to "strict" -- because a use
case published before this column existed never got to choose, and recording
that it chose strict would switch healing off underneath a deployment that has
it on today. Empty means "follow the deployment", which is exactly what every
existing row was doing.

``authored_by`` is here for the same reason ``target`` is: the definition is
the source of truth and has to carry it for promotion, but the list view needs
to show it without reading every stored document.

Revision ID: c3f9a1d84e26
Revises: b8e4a2f70c53
Create Date: 2026-09-06 15:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3f9a1d84e26"
down_revision: Union[str, None] = "b8e4a2f70c53"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "usecases",
        sa.Column(
            "mode",
            sa.String(length=16),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column(
        "usecases",
        sa.Column(
            "authored_by",
            sa.String(length=16),
            nullable=False,
            server_default="person",
        ),
    )


def downgrade() -> None:
    op.drop_column("usecases", "authored_by")
    op.drop_column("usecases", "mode")
