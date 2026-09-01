"""a step records where it happened

`run_steps` said what each step did but not what page it did it on, which the
trail needs for two things: showing a person where they were, and scoping a
fix they record to the right site. The domain is what healing memory filters
on, so without this the "I know what changed" box has nothing to file the
answer under.

Revision ID: f4c2b8e63d17
Revises: e5b1c7d24a80
Create Date: 2026-08-30 22:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f4c2b8e63d17'
down_revision: Union[str, None] = 'e5b1c7d24a80'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'run_steps',
        sa.Column('page_url', sa.Text(), nullable=False, server_default=''),
    )


def downgrade() -> None:
    op.drop_column('run_steps', 'page_url')
