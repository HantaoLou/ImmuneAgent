"""init

Initialize session table

Revision ID: 7d771b541320
Revises:
Create Date: 2025-08-14 17:46:20.968973

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7d771b541320"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create session table
    op.create_table(
        "session",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True, default=""),
        sa.Column("usecase", sa.String(), nullable=True, default=""),
        sa.Column("configuration", sa.Text(), nullable=True, default="{}"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "session_chat_history",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True, default=""),
        sa.Column("message", sa.Text(), nullable=True, default=""),
        sa.Column("timestamp", sa.BigInteger(), nullable=True, default=0),
        sa.Column("role", sa.String(), nullable=True, default=""),
        sa.Column("type", sa.String(), nullable=True, default=""),
        sa.Column("data", sa.Text(), nullable=True, default="{}"),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["session_id"], ["session.id"], ondelete="CASCADE"),
    )

    # Create session_artifact table
    op.create_table(
        "session_artifact",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("original_file_name", sa.String(), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=True, default=0),
        sa.Column(
            "mime_type", sa.String(), nullable=True, default="application/octet-stream"
        ),
        sa.Column("description", sa.Text(), nullable=True, default=""),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["session_id"], ["session.id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop session_artifact table
    op.drop_table("session_artifact")
    # Drop session chat history table
    op.drop_table("session_chat_history")
    # Drop session table
    op.drop_table("session")
