import json
import os
import mimetypes
from pathlib import Path
from sys import prefix
from typing import Annotated, Optional
from uuid import UUID
from loguru import logger
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from web.app import app
from web.db.db import get_db
from web.session.service import SessionArtifactService, SessionService
from web.session.usecases import Usecases
from web.settings import settings
from web.storage.storage_factory import get_storage_service
from fastapi.responses import Response

from .model import Session, SessionArtifact, SessionChatHistory, SessionArtifactCreate

router = APIRouter(prefix="/api/sessions")
usecase_router = APIRouter(prefix="/api/usecases")


class SessionCreationRequest(BaseModel):
    usecase: str


class SessionUpdateRequest(BaseModel):
    configuration: Optional[dict] = None
    name: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    usecase: str
    configuration: dict
    name: Optional[str] = None

    @classmethod
    def from_model(cls, model: Session):
        config = json.loads(model.configuration) if model.configuration else {}
        # 从configuration中提取name字段
        name = config.get("name") if isinstance(config, dict) else None
        return SessionResponse(
            id=model.id.hex,
            usecase=model.usecase,
            configuration=config,
            name=name,
        )


class UsecaseInfo(BaseModel):
    name: str
    description: str


class ChatHistoryResponse(BaseModel):
    session_id: UUID
    message: str
    timestamp: int
    role: str

    @classmethod
    def from_model(cls, model: SessionChatHistory):
        return ChatHistoryResponse(
            session_id=model.session_id,
            message=model.message,
            timestamp=model.timestamp,
            role=model.role,
        )


@router.post("/")
@router.post("")
async def create_session(
    request: SessionCreationRequest, db: AsyncSession = Depends(get_db)
):
    sess = await SessionService.create_session(db, request.usecase)
    return SessionResponse.from_model(sess)


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    sess = await SessionService.get_session(db, session_id)
    return SessionResponse.from_model(sess)


@router.delete("/{session_id}")
async def delete_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    await SessionService.delete_session(db, session_id)
    return {"message": "Session deleted"}


@router.get("/")
@router.get("")
async def get_sessions(db: AsyncSession = Depends(get_db)):
    sess = await SessionService.list_sessions(db)
    return [SessionResponse.from_model(s) for s in sess]


@router.get("/{session_id}/chat-history")
async def get_chat_history(session_id: UUID, db: AsyncSession = Depends(get_db)):
    chat_history = await SessionService.get_chat_history(db, session_id)
    return [ChatHistoryResponse.from_model(m) for m in chat_history]


# 注意：更具体的路由必须在更通用的路由之前定义
# artifacts 相关的路由按从具体到通用的顺序排列

@router.get("/{session_id}/artifacts/upload-url")
async def get_upload_url(
    session_id: str,
    file_name: str = Query(..., description="文件名"),
    content_type: Optional[str] = Query(None, description="文件MIME类型"),
    db: AsyncSession = Depends(get_db),
):
    """获取OSS预签名上传URL（用于前端直传）"""
    try:
        # Convert session_id to UUID
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            try:
                session_uuid = UUID(hex=session_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
        
        # Verify session exists
        session = await SessionService.get_session(db, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        storage_service = get_storage_service()
        
        # 如果未启用OSS，返回错误
        if not storage_service.use_oss:
            raise HTTPException(
                status_code=400, 
                detail="OSS未启用，无法生成上传URL"
            )
        
        # 生成预签名上传URL
        upload_url = storage_service.get_upload_url(
            session_id=session_uuid,
            file_name=file_name,
            content_type=content_type,
            expires=3600 * 8,  # 8小时有效期
        )
        download_url = storage_service.get_download_url(
            session_id=session_uuid,
            file_name=file_name,
            expires=3600 * 8,  # 8小时有效期
        )
        
        # 返回上传URL和相关信息
        return {
            "upload_url": upload_url,
            "download_url": download_url,
            "object_key": storage_service.get_object_key(session_uuid, file_name),
            "expires_in": 3600 * 8,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成上传URL失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate upload URL: {str(e)}"
        )


@router.post("/{session_id}/artifacts/initiate-multipart-upload")
async def initiate_multipart_upload(
    session_id: str,
    file_name: str = Form(...),
    content_type: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """初始化分片上传"""
    try:
        # Convert session_id to UUID
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            try:
                session_uuid = UUID(hex=session_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
        
        # Verify session exists
        session = await SessionService.get_session(db, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        storage_service = get_storage_service()
        
        if not storage_service.use_oss:
            raise HTTPException(status_code=400, detail="OSS未启用，无法使用分片上传")
        
        upload_id = storage_service.initiate_multipart_upload(
            session_id=session_uuid,
            file_name=file_name,
            content_type=content_type,
        )
        
        download_url = storage_service.get_download_url(session_uuid, file_name)
        
        return {
            "upload_id": upload_id,
            "download_url": download_url,
            "object_key": storage_service.get_object_key(session_uuid, file_name),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"初始化分片上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate multipart upload: {str(e)}")


@router.post("/{session_id}/artifacts/upload-part")
async def upload_part(
    session_id: str,
    file_name: str = Form(...),
    upload_id: str = Form(...),
    part_number: int = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """上传分片"""
    try:
        # Convert session_id to UUID
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            try:
                session_uuid = UUID(hex=session_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
        
        # Verify session exists
        session = await SessionService.get_session(db, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        storage_service = get_storage_service()
        
        if not storage_service.use_oss:
            raise HTTPException(status_code=400, detail="OSS未启用，无法上传分片")
        
        part_data = await file.read()
        etag = storage_service.upload_part(
            session_id=session_uuid,
            file_name=file_name,
            upload_id=upload_id,
            part_number=part_number,
            part_data=part_data,
        )
        
        return {
            "etag": etag,
            "part_number": part_number,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传分片失败: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload part: {str(e)}")


@router.get("/{session_id}/artifacts/list-uploaded-parts")
async def list_uploaded_parts(
    session_id: str,
    file_name: str = Query(..., description="文件名"),
    upload_id: str = Query(..., description="上传ID"),
    db: AsyncSession = Depends(get_db),
):
    """列出已上传的分片（用于续传）"""
    try:
        # Convert session_id to UUID
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            try:
                session_uuid = UUID(hex=session_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
        
        # Verify session exists
        session = await SessionService.get_session(db, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        storage_service = get_storage_service()
        
        if not storage_service.use_oss:
            raise HTTPException(status_code=400, detail="OSS未启用，无法列出分片")
        
        parts = storage_service.list_uploaded_parts(
            session_id=session_uuid,
            file_name=file_name,
            upload_id=upload_id,
        )
        
        return {
            "parts": parts,
            "upload_id": upload_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"列出分片失败: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list parts: {str(e)}")


@router.post("/{session_id}/artifacts/complete-multipart-upload")
async def complete_multipart_upload(
    session_id: str,
    file_name: str = Form(...),
    upload_id: str = Form(...),
    parts: str = Form(..., description="JSON格式的分片列表: [{\"part_number\": 1, \"etag\": \"...\"}, ...]"),
    original_file_name: str = Form(...),
    file_size: int = Form(...),
    mime_type: Optional[str] = Form("application/octet-stream"),
    description: Optional[str] = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """完成分片上传并创建artifact记录"""
    try:
        # Convert session_id to UUID
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            try:
                session_uuid = UUID(hex=session_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
        
        # Verify session exists
        session = await SessionService.get_session(db, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        storage_service = get_storage_service()
        
        if not storage_service.use_oss:
            raise HTTPException(status_code=400, detail="OSS未启用，无法完成分片上传")
        
        # 解析分片列表
        import json
        parts_list = json.loads(parts)
        parts_tuple = [(part["part_number"], part["etag"]) for part in parts_list]
        
        # 完成分片上传
        object_key = storage_service.complete_multipart_upload(
            session_id=session_uuid,
            file_name=file_name,
            upload_id=upload_id,
            parts=parts_tuple,
        )
        
        # 创建artifact记录
        artifact_data = SessionArtifactCreate(
            session_id=session_uuid,
            file_name=object_key,
            original_file_name=original_file_name,
            file_size=file_size,
            mime_type=mime_type or "application/octet-stream",
            description=description or "",
        )
        
        artifact = await SessionArtifactService.create_artifact(db, artifact_data)
        
        # 生成下载URL
        download_url = f"/api/sessions/artifacts/{artifact.id}/download"
        
        result = {
            "id": str(artifact.id),
            "session_id": str(artifact.session_id),
            "file_name": artifact.file_name,
            "original_file_name": artifact.original_file_name,
            "file_size": artifact.file_size,
            "mime_type": artifact.mime_type,
            "description": artifact.description,
            "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            "updated_at": artifact.updated_at.isoformat() if artifact.updated_at else None,
            "download_url": download_url,
        }
        
        # 添加预签名下载URL（8小时有效期）
        try:
            file_name_only = os.path.basename(artifact.file_name)
            oss_url = storage_service.get_download_url(artifact.session_id, file_name_only)
            result["oss_direct_url"] = oss_url
            logger.info(f"分片上传完成，生成预签名下载URL（8小时有效）: {oss_url[:100]}...")
        except Exception as e:
            logger.warning(f"无法生成OSS预签名URL: {e}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"完成分片上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to complete multipart upload: {str(e)}")


@router.post("/{session_id}/artifacts/abort-multipart-upload")
async def abort_multipart_upload(
    session_id: str,
    file_name: str = Form(...),
    upload_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """取消分片上传"""
    try:
        # Convert session_id to UUID
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            try:
                session_uuid = UUID(hex=session_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
        
        # Verify session exists
        session = await SessionService.get_session(db, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        storage_service = get_storage_service()
        
        if not storage_service.use_oss:
            raise HTTPException(status_code=400, detail="OSS未启用，无法取消分片上传")
        
        success = storage_service.abort_multipart_upload(
            session_id=session_uuid,
            file_name=file_name,
            upload_id=upload_id,
        )
        
        return {"success": success}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消分片上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to abort multipart upload: {str(e)}")


@router.post("/{session_id}/artifacts/confirm-upload")
async def confirm_upload(
    session_id: str,
    file_name: str = Form(...),
    original_file_name: str = Form(...),
    file_size: int = Form(...),
    mime_type: Optional[str] = Form("application/octet-stream"),
    description: Optional[str] = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """确认文件已上传到OSS，创建artifact记录"""
    try:
        # Convert session_id to UUID
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            try:
                session_uuid = UUID(hex=session_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
        
        # Verify session exists
        session = await SessionService.get_session(db, session_uuid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        storage_service = get_storage_service()
        
        # 验证文件是否存在于OSS
        if storage_service.use_oss:
            object_key = storage_service.get_object_key(session_uuid, file_name)
            if not storage_service.file_exists(session_uuid, file_name):
                raise HTTPException(
                    status_code=404, 
                    detail=f"File not found in OSS: {object_key}. Please upload the file first."
                )
        
        # 创建artifact记录
        artifact_data = SessionArtifactCreate(
            session_id=session_uuid,
            file_name=storage_service.get_object_key(session_uuid, file_name) if storage_service.use_oss else file_name,
            original_file_name=original_file_name,
            file_size=file_size,
            mime_type=mime_type or "application/octet-stream",
            description=description or "",
        )
        
        artifact = await SessionArtifactService.create_artifact(db, artifact_data)
        
        # 生成下载URL
        download_url = f"/api/sessions/artifacts/{artifact.id}/download"
        
        result = {
            "id": str(artifact.id),
            "session_id": str(artifact.session_id),
            "file_name": artifact.file_name,
            "original_file_name": artifact.original_file_name,
            "file_size": artifact.file_size,
            "mime_type": artifact.mime_type,
            "description": artifact.description,
            "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            "updated_at": artifact.updated_at.isoformat() if artifact.updated_at else None,
            "download_url": download_url,
        }
        
        # 如果是OSS存储，添加预签名下载URL（8小时有效期）
        if storage_service.use_oss and artifact.file_name.startswith("artifacts/"):
            try:
                file_name_only = os.path.basename(artifact.file_name)
                # 使用预签名下载URL，默认8小时有效期
                oss_url = storage_service.get_download_url(artifact.session_id, file_name_only)
                result["oss_direct_url"] = oss_url
                logger.info(f"生成预签名下载URL（8小时有效）: {oss_url[:100]}...")
            except Exception as e:
                logger.warning(f"无法生成OSS预签名URL: {e}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"确认上传失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to confirm upload: {str(e)}"
        )


@router.get("/{session_id}/artifacts")
async def get_session_artifacts(session_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get all artifacts for a specific session."""
    try:
        artifacts = await SessionArtifactService.get_artifacts_by_session(
            db, session_id
        )
        
        # 如果使用OSS，为每个artifact添加OSS直接下载URL
        storage_service = get_storage_service()
        result = []
        for artifact in artifacts:
            artifact_dict = {
                "id": str(artifact.id),
                "session_id": str(artifact.session_id),
                "file_name": artifact.file_name,
                "original_file_name": artifact.original_file_name,
                "file_size": artifact.file_size,
                "mime_type": artifact.mime_type,
                "description": artifact.description,
                "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
                "updated_at": artifact.updated_at.isoformat() if artifact.updated_at else None,
                "download_url": f"/api/sessions/artifacts/{artifact.id}/download",
            }
            
            # 如果是OSS存储，添加预签名下载URL（8小时有效）
            if storage_service.use_oss and artifact.file_name.startswith("artifacts/"):
                try:
                    file_name = os.path.basename(artifact.file_name)
                    # 使用预签名下载URL（8小时有效期）
                    oss_url = storage_service.get_download_url(
                        session_id=artifact.session_id,
                        file_name=file_name,
                        expires=3600 * 8,  # 8小时有效期
                    )
                    artifact_dict["oss_direct_url"] = oss_url
                    logger.debug(f"生成预签名下载URL（8小时有效）: {oss_url[:100]}...")
                except Exception as e:
                    logger.warning(f"无法生成OSS预签名下载URL: {e}")
            
            result.append(artifact_dict)
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get artifacts: {str(e)}"
        )


@router.put("/{session_id}")
async def update_session(
    session_id: UUID, request: SessionUpdateRequest, db: AsyncSession = Depends(get_db)
):
    # 如果提供了name，需要将其合并到configuration中
    if request.name is not None:
        # 获取当前session的configuration
        sess = await SessionService.get_session(db, session_id)
        current_config = json.loads(sess.configuration) if sess.configuration else {}
        
        # 更新name字段
        if not isinstance(current_config, dict):
            current_config = {}
        current_config["name"] = request.name
        
        # 如果也提供了configuration，合并它们
        if request.configuration:
            current_config.update(request.configuration)
        
        sess = await SessionService.update_session(db, session_id, current_config)
    elif request.configuration is not None:
        sess = await SessionService.update_session(db, session_id, request.configuration)
    else:
        # 如果没有提供任何更新，返回当前session
        sess = await SessionService.get_session(db, session_id)
    
    return SessionResponse.from_model(sess)


@router.get("/artifacts/{artifact_id}")
async def get_artifact_info(artifact_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get artifact information by ID."""
    try:
        artifact = await SessionArtifactService.get_artifact_by_id(db, artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        # 如果使用OSS，尝试添加OSS直接下载URL
        storage_service = get_storage_service()
        result = {
            "id": str(artifact.id),
            "session_id": str(artifact.session_id),
            "file_name": artifact.file_name,
            "original_file_name": artifact.original_file_name,
            "file_size": artifact.file_size,
            "mime_type": artifact.mime_type,
            "description": artifact.description,
            "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            "updated_at": artifact.updated_at.isoformat() if artifact.updated_at else None,
        }
        
        # 如果是OSS存储，添加直接下载URL
        if storage_service.use_oss and artifact.file_name.startswith("artifacts/"):
            try:
                file_name = os.path.basename(artifact.file_name)
                if settings.oss_use_public_url:
                    # 使用公共URL
                    oss_url = storage_service.get_public_url(artifact.session_id, file_name)
                else:
                    # 使用预签名URL（1小时有效）
                    oss_url = storage_service.get_file_url(artifact.session_id, file_name, expires=3600)
                result["oss_direct_url"] = oss_url
            except Exception as e:
                logger.warning(f"无法生成OSS直接URL: {e}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get artifact: {str(e)}")


@router.get("/artifacts/{artifact_id}/download")
async def download_artifact(artifact_id: UUID, db: AsyncSession = Depends(get_db)):
    """Download an artifact file."""
    try:
        artifact = await SessionArtifactService.get_artifact_by_id(db, artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        storage_service = get_storage_service()
        
        # 判断文件是存储在OSS还是本地
        if storage_service.use_oss and artifact.file_name.startswith("artifacts/"):
            # 从OSS下载
            try:
                # 从object_key中提取文件名
                file_name = os.path.basename(artifact.file_name)
                file_content = storage_service.download_file(
                    artifact.session_id,
                    file_name
                )
                
                # 返回文件内容
                return Response(
                    content=file_content,
                    media_type=artifact.mime_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{artifact.original_file_name}"'
                    }
                )
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Artifact file not found in OSS")
            except Exception as e:
                logger.error(f"从OSS下载文件失败: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to download artifact from OSS: {str(e)}"
                )
        else:
            # 从本地文件系统读取
            # artifact.file_name may be a full path (from create_artifact_with_binary) or relative path
            if os.path.isabs(artifact.file_name) and os.path.exists(artifact.file_name):
                file_path = artifact.file_name
            else:
                file_path = SessionArtifactService.get_artifact_file_path(
                    artifact.session_id, artifact.file_name
                )

            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Artifact file not found")

            # Return file response
            return FileResponse(
                path=file_path,
                filename=artifact.original_file_name,
                media_type=artifact.mime_type,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to download artifact: {str(e)}"
        )


def validate_file_path(file_path: str) -> Path:
    """
    Validate and normalize file path for security.
    Prevents path traversal attacks and ensures path is within allowed directories.
    
    Args:
        file_path: The file path to validate
        
    Returns:
        Normalized Path object
        
    Raises:
        HTTPException: If path is invalid or outside allowed directories
    """
    try:
        # Decode URL-encoded path
        decoded_path = unquote(file_path)
        
        # Convert to absolute path and normalize
        abs_path = Path(os.path.abspath(decoded_path)).resolve()
        
        # Define allowed directories
        allowed_dirs = [
            Path(settings.artifact_path).resolve(),  # Primary artifact directory
            Path("/data_new/workspace").resolve(),  # Additional allowed directory (entire workspace directory)
        ]
        
        # Check if path is within any allowed directory
        for allowed_dir in allowed_dirs:
            try:
                abs_path.relative_to(allowed_dir)
                # Path is within this allowed directory, allow it
                return abs_path
            except ValueError:
                # Path is not within this directory, try next one
                continue
        
        # Path is not within any allowed directory
        logger.warning(
            f"Attempted to access file outside allowed directories: {decoded_path}"
        )
        allowed_dirs_str = ", ".join(str(d) for d in allowed_dirs)
        raise HTTPException(
            status_code=403,
            detail=f"File path must be within allowed directory: {allowed_dirs_str}"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating file path {file_path}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid file path: {str(e)}"
        )


@router.get("/files/download")
async def download_file_by_path(
    path: str = Query(..., description="Absolute file path to download"),
    db: AsyncSession = Depends(get_db),
):
    """
    Download a file by its absolute path.
    
    Security: Only files within the artifact_path directory are allowed.
    This prevents path traversal attacks.
    
    Args:
        path: Absolute file path (URL-encoded if needed)
        db: Database session
        
    Returns:
        FileResponse with the requested file
    """
    try:
        logger.info(f"Download file request received for path: {path}")
        
        # Validate and normalize the path
        file_path = validate_file_path(path)
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if it's actually a file (not a directory)
        if not file_path.is_file():
            logger.warning(f"Path is not a file: {file_path}")
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        # Get filename for download
        filename = file_path.name
        
        logger.info(f"Returning file: {file_path} (MIME: {mime_type})")
        
        # Return file response
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=mime_type,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file by path: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to download file: {str(e)}"
        )


@router.delete("/artifacts/{artifact_id}")
async def delete_artifact(artifact_id: UUID, db: AsyncSession = Depends(get_db)):
    """Delete an artifact."""
    try:
        success = await SessionArtifactService.delete_artifact(db, artifact_id)

        if not success:
            raise HTTPException(status_code=404, detail="Artifact not found")

        return {"message": "Artifact deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete artifact: {str(e)}"
        )


@usecase_router.get("")
async def list_usecases():
    """List all available usecases"""
    usecases = []
    for usecase in Usecases.list_usecases():
        usecases.append(UsecaseInfo(name=usecase.name, description=""))
    return usecases
