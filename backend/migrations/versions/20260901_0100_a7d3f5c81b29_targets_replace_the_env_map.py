"""targets replace the environment map

The base URL a use case runs against used to come from ``USECASE_ENV``, one map
per deployment. That only scales while every use case in an environment shares
a base URL. Run workflows against several sites and it becomes a variable per
site, set in the environment and requiring a redeploy to add one -- which is
configuration standing in for data.

A target is that data: a name and the URL it means *here*. Twenty use cases
against one site share one row, moving that site's UAT host is one edit, and
onboarding a new site is a row rather than a release. Each deployment's own
database holds its own URLs, so promoting a use case still moves nothing about
where it points.

Revision ID: a7d3f5c81b29
Revises: f4c2b8e63d17
Create Date: 2026-09-01 01:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a7d3f5c81b29'
down_revision: Union[str, None] = 'f4c2b8e63d17'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'targets',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('workspace_id', sa.String(length=32), nullable=False),
        # The handle a use case refers to, e.g. "schemora". Unique per
        # workspace: a use case names one, and two rows answering to the same
        # name would make which site it reaches a coin toss.
        sa.Column('name', sa.String(length=64), nullable=False),
        sa.Column('base_url', sa.String(length=2000), nullable=False),
        sa.Column('description', sa.String(length=300), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_by', sa.String(length=320), nullable=False, server_default=''),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('workspace_id', 'name', name='uq_targets_workspace_name'),
    )
    op.create_index('ix_targets_workspace', 'targets', ['workspace_id'])

    # Which target a use case runs against. Empty means the one recorded into
    # the definition, which is what keeps a single-environment install working
    # with nothing configured at all.
    op.add_column(
        'usecases',
        sa.Column('target', sa.String(length=64), nullable=False, server_default=''),
    )

    # Where a batch actually ran, resolved once when it was queued. A resume
    # or a retry reads it back rather than resolving again, so a target edited
    # mid-batch cannot move the second half to a different deployment from the
    # first.
    op.add_column(
        'batches',
        sa.Column('base_url', sa.String(length=2000), nullable=False, server_default=''),
    )


def downgrade() -> None:
    op.drop_column('batches', 'base_url')
    op.drop_column('usecases', 'target')
    op.drop_index('ix_targets_workspace', table_name='targets')
    op.drop_table('targets')
