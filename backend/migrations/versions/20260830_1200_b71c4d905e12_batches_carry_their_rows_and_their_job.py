"""batches carry their rows and their job

A batch used to exist only as a counter: the rows it was running lived in the
API process's memory, held by the asyncio task driving them. That made two
things impossible, and this migration is what unblocks both.

`input_rows` is the snapshot of what the batch runs. A worker claiming the job
may be a different process on a different machine, so it cannot inherit the
rows from whoever accepted the upload. Resume needed them too: it used to
rebuild rows from `executions.inputs`, which substitutes an empty row for
anything never attempted -- exactly the set of rows a resume exists to run.

`job_id` ties the batch to its queue entry. Both are written in one
transaction, so there is never a queued batch no worker will run, and
cancelling a batch that has not started is a lookup rather than a search.

Secrets are deliberately not here. They are resolved from the vault via
`credential_id` by whoever runs the batch, so a password is never written to a
table that a batch listing reads.

`server_default='[]'` because the column is NOT NULL and these tables have
rows. An existing batch gets an empty row set, which is the truth about it: its
rows are gone, and only a resume rebuilt from executions could ever have run
them.

Revision ID: b71c4d905e12
Revises: a4006c8e38c0
Create Date: 2026-08-30 12:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'b71c4d905e12'
down_revision: Union[str, None] = 'a4006c8e38c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'batches',
        sa.Column(
            'input_rows',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default='[]',
        ),
    )
    op.add_column('batches', sa.Column('job_id', sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column('batches', 'job_id')
    op.drop_column('batches', 'input_rows')
