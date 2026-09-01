"""datasets as a resource

An uploaded file used to be anonymous: rows arrived in the request that started
a batch, were used, and were gone. Mapping needs them to outlive one request --
you upload a file, look at what is in it, agree how its columns line up with a
use case, and only then run something. That is three round trips over the same
rows, and without a table the browser has to hold the file and post it each
time.

`columns` holds one profile per column (kind, blanks, distinct count, example
values), which is what the mapper reasons over. `rows` holds the parsed
content. Both are JSONB rather than child tables because they are read whole,
by one owner, and never joined on -- the same argument `batches.input_rows`
makes.

Revision ID: c8f2a71b4d33
Revises: b71c4d905e12
Create Date: 2026-08-30 16:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'c8f2a71b4d33'
down_revision: Union[str, None] = 'b71c4d905e12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'datasets',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('workspace_id', sa.String(length=32), nullable=False),
        sa.Column('owner_id', sa.String(length=32), nullable=True),
        sa.Column(
            'owner_email', sa.String(length=320), nullable=False, server_default=''
        ),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('filename', sa.String(length=400), nullable=False),
        sa.Column('source', sa.String(length=16), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=False),
        sa.Column(
            'columns',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default='[]',
        ),
        sa.Column(
            'rows',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default='[]',
        ),
        sa.Column(
            'warnings',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default='[]',
        ),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['owner_id'], ['users.id'], name=op.f('fk_datasets_owner_id'), ondelete='SET NULL'
        ),
        sa.ForeignKeyConstraint(
            ['workspace_id'],
            ['workspaces.id'],
            name=op.f('fk_datasets_workspace_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_datasets')),
    )
    op.create_index(
        'ix_datasets_workspace_created', 'datasets', ['workspace_id', 'created_at'], unique=False
    )
    # Which dataset a batch came from, so a run can be traced back to the file
    # it was started with. Nullable: a batch may still be started by posting
    # rows directly, and every batch that predates this one was.
    op.add_column('batches', sa.Column('dataset_id', sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column('batches', 'dataset_id')
    op.drop_index('ix_datasets_workspace_created', table_name='datasets')
    op.drop_table('datasets')
