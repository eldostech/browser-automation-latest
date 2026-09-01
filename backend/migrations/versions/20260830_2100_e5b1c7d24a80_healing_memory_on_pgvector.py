"""healing memory on pgvector

A locator that broke, and what fixed it, so the same site redesign costs one
model call the first time a workflow hits it and none afterwards.

`CREATE EXTENSION vector` needs to succeed before the table can exist. Aurora
PostgreSQL ships it; a stock local Postgres needs the pgvector package
installed on the server. If that is missing this migration stops here with a
message saying so, which is a far better failure than a driver error about an
unknown type.

The index is IVFFlat with cosine distance. It is deliberately created without
`lists` tuning: this table holds tens to hundreds of rows per domain, where a
sequential scan is already fast and a carefully tuned index would be
premature. It exists so the query plan does not change shape later.

Revision ID: e5b1c7d24a80
Revises: d3e8a05c9f21
Create Date: 2026-08-30 21:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = 'e5b1c7d24a80'
down_revision: Union[str, None] = 'd3e8a05c9f21'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DIMENSIONS = 1024


def upgrade() -> None:
    connection = op.get_bind()
    available = connection.execute(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
    ).scalar()
    if not available:
        raise RuntimeError(
            "The 'vector' extension is not available on this PostgreSQL server, so "
            "healing memory cannot be created. Install pgvector on the server "
            "(Aurora PostgreSQL and the pgvector/pgvector Docker image ship it), "
            "then run `make db-upgrade` again."
        )
    # Extensions are database-wide, not schema-scoped, so this is idempotent
    # and safe when several schemas share one database.
    connection.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    op.create_table(
        'healing_memory',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('workspace_id', sa.String(length=32), nullable=False),
        sa.Column('usecase_id', sa.String(length=32), nullable=True),
        sa.Column('domain', sa.String(length=253), nullable=False),
        sa.Column('step_id', sa.String(length=64), nullable=False),
        sa.Column('error_kind', sa.String(length=24), nullable=False),
        sa.Column('dom_context', sa.Text(), nullable=False),
        sa.Column('old_locator', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('new_locator', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=False),
        sa.Column('confirmed_by', sa.String(length=320), nullable=False),
        sa.Column('embedding', Vector(DIMENSIONS), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['workspace_id'],
            ['workspaces.id'],
            name=op.f('fk_healing_memory_workspace_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_healing_memory')),
    )
    op.create_index(
        'ix_healing_memory_scope',
        'healing_memory',
        ['workspace_id', 'domain', 'created_at'],
        unique=False,
    )

    op.execute(
        sa.text(
            "CREATE INDEX ix_healing_memory_embedding ON healing_memory "
            "USING ivfflat (embedding vector_cosine_ops)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ix_healing_memory_embedding"))
    op.drop_index('ix_healing_memory_scope', table_name='healing_memory')
    op.drop_table('healing_memory')
    # The extension is left in place: another schema in the same database may
    # be using it, and dropping it would take their tables with it.
