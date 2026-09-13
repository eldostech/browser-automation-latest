"""a workspace registers mcp servers for its agent

Before this, the only tool source an agent session could have was one
hardcoded browser provider. This table is what lets a workspace add more --
each row a server to open beside the browser, its tools shown to the model
under that server's name so they can never collide with a browser tool or a
mark tool. See ``agent/session.py``'s ``AgentToolSession.extra``.

``connection`` is JSONB rather than columns because what it takes to open a
server is a property of ``transport``, which today is only ``"stdio"``
(``{"command", "args", "env"}``) -- an ``"sse"``/``"http"`` transport later
takes a different shape and should not need a migration to add.

None of this touches the replay engine. A server's tools are for the agent's
own use while it works; ``agent/tools.py``'s ``DISTILS_TO`` only ever names
Playwright's own tool names, so nothing registered here can become a step.

Revision ID: b0bf6e3c74cf
Revises: d4a7c2e91f38
Create Date: 2026-09-06 20:05:08.959660
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'b0bf6e3c74cf'
down_revision: Union[str, None] = 'd4a7c2e91f38'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # No `schema=` here, and the foreign keys below name `users`/`workspaces`
    # unqualified -- matching every sibling migration. `env.py` sets
    # `search_path` before running, so unqualified DDL lands in `DB_SCHEMA`
    # regardless of which one this was generated against.
    op.create_table(
        'agent_tool_servers',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('workspace_id', sa.String(length=32), nullable=False),
        sa.Column('owner_id', sa.String(length=32), nullable=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('transport', sa.String(length=20), nullable=False),
        sa.Column('connection', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_agent_tool_servers_owner_id'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], name=op.f('fk_agent_tool_servers_workspace_id'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_agent_tool_servers')),
        sa.UniqueConstraint('workspace_id', 'name', name='uq_agent_tool_servers_workspace_id_name'),
    )


def downgrade() -> None:
    op.drop_table('agent_tool_servers')
