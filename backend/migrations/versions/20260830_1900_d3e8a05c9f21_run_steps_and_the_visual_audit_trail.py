"""run_steps and the visual audit trail

The events table already carries every step, but it is a log: reading "every
step of this execution with its status, duration and screenshot" out of it
means scanning JSON, and the visual diff needs something the log cannot express
at all -- this step joined to the same `step_id` from the run that last worked.

So this is a projection, written alongside the events rather than instead of
them. `baseline_id` and `pixel_diff` are filled in on write, so comparing two
runs is a read rather than an image operation per render.

Revision ID: d3e8a05c9f21
Revises: c8f2a71b4d33
Create Date: 2026-08-30 19:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd3e8a05c9f21'
down_revision: Union[str, None] = 'c8f2a71b4d33'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'run_steps',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('run_id', sa.String(length=32), nullable=False),
        sa.Column('workspace_id', sa.String(length=32), nullable=False),
        sa.Column('usecase_id', sa.String(length=32), nullable=True),
        sa.Column('version', sa.Integer(), nullable=True),
        sa.Column('row_index', sa.Integer(), nullable=True),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('step_id', sa.String(length=64), nullable=False),
        sa.Column('phase', sa.String(length=16), nullable=False),
        sa.Column('action', sa.String(length=24), nullable=False),
        sa.Column('locator', sa.Text(), nullable=False),
        sa.Column('locator_rung', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('duration_ms', sa.Integer(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('screenshot_id', sa.String(length=32), nullable=True),
        sa.Column('baseline_id', sa.String(length=32), nullable=True),
        sa.Column('pixel_diff', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['run_id'], ['runs.id'], name=op.f('fk_run_steps_run_id'), ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['workspace_id'],
            ['workspaces.id'],
            name=op.f('fk_run_steps_workspace_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_run_steps')),
    )
    op.create_index('ix_run_steps_run_seq', 'run_steps', ['run_id', 'seq'], unique=False)
    op.create_index(
        'ix_run_steps_usecase_step',
        'run_steps',
        ['usecase_id', 'version', 'step_id'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_run_steps_usecase_step', table_name='run_steps')
    op.drop_index('ix_run_steps_run_seq', table_name='run_steps')
    op.drop_table('run_steps')
