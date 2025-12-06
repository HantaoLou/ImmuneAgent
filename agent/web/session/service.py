import json
import logging
import os
import shutil
import time
import uuid
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from web.session.model import (
    Session,
    SessionArtifact,
    SessionArtifactCreate,
    SessionChatHistory,
)
from web.session.usecases import Usecase, Usecases
from web.settings import settings
from web.storage.storage_factory import get_storage_service

from .model import Session, SessionChatHistory

logger = logging.getLogger(__name__)


class SessionService:
    def __init__(self):
        self.sessions = {}

    @staticmethod
    def _ensure_uuid(value: UUID | str) -> UUID:
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            return UUID(value)
        raise ValueError(f"Invalid UUID value: {value!r}")

    @classmethod
    async def create_session(cls, db: AsyncSession, usecase: str):
        sess = Session(usecase=usecase)
        uc: Usecase = Usecases.get_usecase(usecase)
        sess.configuration = json.dumps(uc.default_configuration)
        db.add(sess)
        await db.commit()
        await db.refresh(sess)
        return sess

    @classmethod
    async def get_session(cls, db: AsyncSession, session_id: UUID):
        normalized_id = cls._ensure_uuid(session_id)
        result = await db.execute(select(Session).where(Session.id == normalized_id))
        return result.scalar_one()

    @classmethod
    async def delete_session(cls, db: AsyncSession, session_id: UUID):
        normalized_id = cls._ensure_uuid(session_id)
        await db.execute(delete(Session).where(Session.id == normalized_id))

    @classmethod
    async def list_sessions(cls, db: AsyncSession):
        result = await db.execute(select(Session))
        return result.scalars().all()

    @classmethod
    async def update_session(
        cls, db: AsyncSession, session_id: UUID, config: dict
    ) -> Session:
        normalized_id = cls._ensure_uuid(session_id)
        result = await db.execute(select(Session).where(Session.id == normalized_id))
        sess = result.scalar_one()
        sess.configuration = json.dumps(config)
        await db.commit()
        return sess

    @classmethod
    async def save_chat_history(
        cls, db: AsyncSession, session_id: UUID, message: str, role: str = "assistant"
    ):
        if not message:
            return
        timestamp = int(time.time() * 1000)
        chat_history = SessionChatHistory(
            session_id=session_id, message=message, role=role, timestamp=timestamp
        )
        db.add(chat_history)
        await db.commit()
        return chat_history

    @classmethod
    async def get_chat_history(cls, db: AsyncSession, session_id: UUID):
        normalized_id = cls._ensure_uuid(session_id)
        result = await db.execute(
            select(SessionChatHistory).where(
                SessionChatHistory.session_id == normalized_id
            )
        )
        return result.scalars().all()


class SessionArtifactService:
    artifact_path = settings.artifact_path

    @classmethod
    def get_artifact_url(cls, session_id: UUID, file_name: str) -> str:
        return f"{cls.artifact_path}/{session_id.hex}/{file_name}"

    @classmethod
    async def create_artifact_with_binary(
        cls,
        db: AsyncSession,
        session_id: UUID,
        file_name: str,
        file_content: bytes,
        original_file_name: str,
        mime_type: str = "application/octet-stream",
        description: str = "",
    ):
        storage_service = get_storage_service()
        
        # 如果使用OSS，上传到OSS；否则保存到本地
        if storage_service.use_oss:
            try:
                # 上传到OSS
                object_key = storage_service.upload_file(
                    session_id=session_id,
                    file_name=file_name,
                    file_content=file_content,
                    content_type=mime_type,
                )
                # 在数据库中存储OSS对象键（object_key）而不是本地路径
                file_path = object_key
                logger.info(f"文件已上传到OSS: {object_key}")
            except Exception as e:
                logger.error(f"OSS上传失败，回退到本地存储: {e}")
                # 回退到本地存储
                session_dir = cls.ensure_session_artifact_dir(session_id)
                file_path = os.path.join(session_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(file_content)
        else:
            # 本地存储
            session_dir = cls.ensure_session_artifact_dir(session_id)
            file_path = os.path.join(session_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(file_content)
        
        return await cls.create_artifact(
            db,
            SessionArtifactCreate(
                session_id=session_id,
                file_name=file_path,  # 可能是OSS object_key或本地路径
                original_file_name=original_file_name,
                file_size=len(file_content),
                mime_type=mime_type,
                description=description,
            ),
        )

    @classmethod
    async def create_artifact(
        cls, db_session: AsyncSession, artifact_data: SessionArtifactCreate
    ) -> SessionArtifact:
        """Create a new session artifact record."""
        artifact = SessionArtifact(
            session_id=artifact_data.session_id,
            file_name=artifact_data.file_name,
            original_file_name=artifact_data.original_file_name,
            file_size=artifact_data.file_size,
            mime_type=artifact_data.mime_type,
            description=artifact_data.description,
        )

        db_session.add(artifact)
        await db_session.flush()
        await db_session.refresh(artifact)
        await db_session.commit()

        return artifact

    @classmethod
    async def get_artifact_by_id(
        cls, db_session: AsyncSession, artifact_id: uuid.UUID
    ) -> Optional[SessionArtifact]:
        """Get artifact by ID."""
        result = await db_session.execute(
            select(SessionArtifact).where(SessionArtifact.id == artifact_id)
        )
        artifact = result.scalar_one_or_none()

        if artifact:
            return artifact
        logger.error("artifact not found: {}", artifact_id)
        return None

    @classmethod
    async def get_artifacts_by_session(
        cls, db_session: AsyncSession, session_id: uuid.UUID
    ) -> List[SessionArtifact]:
        """Get all artifacts for a specific session."""
        result = await db_session.execute(
            select(SessionArtifact)
            .where(SessionArtifact.session_id == session_id)
            .order_by(SessionArtifact.created_at.desc())
        )
        artifacts = result.scalars().all()

        return artifacts

    @classmethod
    async def delete_artifact(
        cls, db_session: AsyncSession, artifact_id: uuid.UUID
    ) -> bool:
        """Delete an artifact record and its file."""
        result = await db_session.execute(
            select(SessionArtifact).where(SessionArtifact.id == artifact_id)
        )
        artifact = result.scalar_one_or_none()

        if not artifact:
            return False

        # Delete the physical file (OSS or local)
        storage_service = get_storage_service()
        
        if storage_service.use_oss:
            # 如果file_name是OSS object_key格式（以artifacts/开头），则从OSS删除
            if artifact.file_name.startswith("artifacts/"):
                try:
                    # 从OSS删除
                    # 从object_key中提取文件名
                    file_name = os.path.basename(artifact.file_name)
                    storage_service.delete_file(artifact.session_id, file_name)
                    logger.info(f"文件已从OSS删除: {artifact.file_name}")
                except Exception as e:
                    logger.error(f"从OSS删除文件失败: {e}")
            else:
                # 可能是旧数据，尝试本地删除
                file_path = cls.get_artifact_file_path(artifact.session_id, artifact.file_name)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except OSError:
                    pass
        else:
            # 本地存储删除
            file_path = cls.get_artifact_file_path(artifact.session_id, artifact.file_name)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass  # File might already be deleted

        # Delete the database record
        await db_session.delete(artifact)
        return True

    @classmethod
    def get_artifact_file_path(cls, session_id: uuid.UUID, file_name: str) -> str:
        """Get the full file path for an artifact."""
        return os.path.join(cls.artifact_path, str(session_id), file_name)

    @classmethod
    def ensure_session_artifact_dir(cls, session_id: uuid.UUID) -> str:
        """Ensure the session artifact directory exists."""
        session_dir = os.path.join(cls.artifact_path, str(session_id))
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    @classmethod
    async def save_artifact_file(
        cls,
        db_session: AsyncSession,
        session_id: uuid.UUID,
        file_name: str,
        file_content: bytes,
        original_file_name: str,
        mime_type: str = None,
        description: str = "",
    ) -> SessionArtifact:
        """Save an artifact file and create the database record."""
        storage_service = get_storage_service()
        
        # 如果使用OSS，上传到OSS；否则保存到本地
        if storage_service.use_oss:
            try:
                # 上传到OSS
                object_key = storage_service.upload_file(
                    session_id=session_id,
                    file_name=file_name,
                    file_content=file_content,
                    content_type=mime_type or "application/octet-stream",
                )
                file_path = object_key
                logger.info(f"文件已上传到OSS: {object_key}")
            except Exception as e:
                logger.error(f"OSS上传失败，回退到本地存储: {e}")
                # 回退到本地存储
                session_dir = cls.ensure_session_artifact_dir(session_id)
                file_path = os.path.join(session_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(file_content)
        else:
            # 本地存储
            session_dir = cls.ensure_session_artifact_dir(session_id)
            file_path = os.path.join(session_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(file_content)

        # Create database record
        artifact_data = SessionArtifactCreate(
            session_id=session_id,
            file_name=file_path,  # 可能是OSS object_key或本地路径
            original_file_name=original_file_name,
            file_size=len(file_content),
            mime_type=mime_type or "application/octet-stream",
            description=description,
        )

        return await cls.create_artifact(db_session, artifact_data)
