"""artifacts remember their filename

A screenshot needs no name -- it is identified by the step it belongs to. A
document downloaded from a vendor does: the file is the deliverable, and
"statement-A-1001.pdf" is what a person looks for and what the next system
expects when it is uploaded again. Losing it leaves a bucket of opaque ids.

Revision ID: b8e4a2f70c53
Revises: a7d3f5c81b29
Create Date: 2026-09-04 12:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b8e4a2f70c53'
down_revision: Union[str, None] = 'a7d3f5c81b29'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'artifacts',
        sa.Column('filename', sa.String(length=400), nullable=False, server_default=''),
    )


def downgrade() -> None:
    op.drop_column('artifacts', 'filename')
