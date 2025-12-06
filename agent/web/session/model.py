import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import (
    UUID,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from web.utils.uuid import get_uuid


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "session"
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=get_uuid)
    user_id: Mapped[str] = mapped_column(String, default="")
    usecase: Mapped[str] = mapped_column(String, default="")
    configuration: Mapped[str] = mapped_column(Text, default="{}")

    # Relationships
    chat_history: Mapped[list["SessionChatHistory"]] = relationship(
        "SessionChatHistory", back_populates="session", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list["SessionArtifact"]] = relationship(
        "SessionArtifact", back_populates="session", cascade="all, delete-orphan"
    )


class SessionChatHistory(Base):
    __tablename__ = "session_chat_history"
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=get_uuid)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("session.id", ondelete="CASCADE"), nullable=False
    )
    message: Mapped[str] = mapped_column(Text, default="")
    timestamp: Mapped[int] = mapped_column(BigInteger, default=0)
    role: Mapped[str] = mapped_column(String, default="")
    type: Mapped[str] = mapped_column(String, default="")
    # https://ts.llamaindex.ai/docs/chat-ui/parts
    # object resresentation for data like image, video, audio, etc.
    data: Mapped[str] = mapped_column(Text, default="{}")

    # Relationship
    session: Mapped["Session"] = relationship("Session", back_populates="chat_history")


class SessionArtifact(Base):
    __tablename__ = "session_artifact"
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=get_uuid)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("session.id", ondelete="CASCADE"), nullable=False
    )
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    original_file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)
    mime_type: Mapped[str] = mapped_column(String, default="application/octet-stream")
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationship
    session: Mapped["Session"] = relationship("Session", back_populates="artifacts")


# Pydantic schemas for API
class SessionArtifactCreate(BaseModel):
    session_id: uuid.UUID
    file_name: str
    original_file_name: str
    file_size: int
    mime_type: Optional[str] = "application/octet-stream"
    description: Optional[str] = ""
