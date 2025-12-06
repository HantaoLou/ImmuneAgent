"""
OSS存储服务API
提供HTTP接口供各个MCP服务和本项目调用，实现OSS文件的统一访问
"""
import os
import logging
from typing import Optional
from uuid import UUID
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from web.db.db import get_db
from web.storage.storage_factory import get_storage_service
from web.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oss", tags=["OSS Storage"])


class ResolveUrlRequest(BaseModel):
    """解析OSS URL请求"""
    oss_url: str
    session_id: str


class ResolveUrlResponse(BaseModel):
    """解析OSS URL响应"""
    object_key: Optional[str] = None
    session_id: Optional[str] = None
    file_name: Optional[str] = None
    success: bool
    message: Optional[str] = None


@router.get("/file/{session_id}/{file_name:path}")
async def download_file(
    session_id: str,
    file_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    下载OSS文件
    
    Args:
        session_id: 会话ID
        file_name: 文件名（支持路径，如 subdir/file.txt）
        
    Returns:
        文件内容（二进制流）
    """
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
    
    storage_service = get_storage_service()
    
    if not storage_service.use_oss:
        raise HTTPException(status_code=503, detail="OSS service is not enabled")
    
    try:
        # 从OSS下载文件
        file_content = storage_service.download_file(session_uuid, file_name)
        
        # 获取文件MIME类型
        from mimetypes import guess_type
        mime_type, _ = guess_type(file_name)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        # 返回文件内容
        return Response(
            content=file_content,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.basename(file_name)}"',
                "X-OSS-Object-Key": f"artifacts/{session_id}/{file_name}",
            }
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_name}")
    except Exception as e:
        logger.error(f"从OSS下载文件失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to download file from OSS: {str(e)}"
        )


@router.get("/file/{session_id}/{file_name:path}/exists")
async def check_file_exists(
    session_id: str,
    file_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    检查OSS文件是否存在
    
    Args:
        session_id: 会话ID
        file_name: 文件名
        
    Returns:
        {"exists": bool, "object_key": str}
    """
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
    
    storage_service = get_storage_service()
    
    if not storage_service.use_oss:
        return {"exists": False, "object_key": None, "message": "OSS service is not enabled"}
    
    try:
        exists = storage_service.file_exists(session_uuid, file_name)
        object_key = storage_service.get_object_key(session_uuid, file_name)
        return {
            "exists": exists,
            "object_key": object_key,
        }
    except Exception as e:
        logger.error(f"检查OSS文件存在性失败: {e}")
        return {"exists": False, "object_key": None, "message": str(e)}


@router.get("/file/{session_id}/{file_name:path}/info")
async def get_file_info(
    session_id: str,
    file_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    获取OSS文件信息
    
    Args:
        session_id: 会话ID
        file_name: 文件名
        
    Returns:
        文件信息（大小、MIME类型等）
    """
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid session ID format: {session_id}")
    
    storage_service = get_storage_service()
    
    if not storage_service.use_oss:
        raise HTTPException(status_code=503, detail="OSS service is not enabled")
    
    try:
        object_key = storage_service.get_object_key(session_uuid, file_name)
        
        # 检查文件是否存在
        exists = storage_service.file_exists(session_uuid, file_name)
        if not exists:
            raise HTTPException(status_code=404, detail=f"File not found: {file_name}")
        
        # 获取文件URL
        if settings.oss_use_public_url:
            file_url = storage_service.get_public_url(session_uuid, file_name)
        else:
            file_url = storage_service.get_file_url(session_uuid, file_name, expires=3600)
        
        # 获取文件大小（需要从数据库获取）
        try:
            from web.session.model import SessionArtifact
            from sqlalchemy import select
            
            stmt = select(SessionArtifact).where(
                SessionArtifact.session_id == session_uuid
            ).where(
                SessionArtifact.file_name == object_key
            )
            result = await db.execute(stmt)
            artifact = result.scalar_one_or_none()
            
            file_size = artifact.file_size if artifact else None
            mime_type = artifact.mime_type if artifact else None
        except Exception:
            file_size = None
            mime_type = None
        
        return {
            "object_key": object_key,
            "file_name": file_name,
            "file_url": file_url,
            "file_size": file_size,
            "mime_type": mime_type,
            "session_id": str(session_uuid),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取OSS文件信息失败: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get file info: {str(e)}"
        )


@router.post("/resolve-url", response_model=ResolveUrlResponse)
async def resolve_oss_url(
    request: ResolveUrlRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    解析OSS URL为object_key
    
    将OSS URL（公共URL或预签名URL）转换为object_key格式，供工具使用
    
    Args:
        request: 包含OSS URL和session_id的请求
        
    Returns:
        解析结果（object_key, session_id, file_name）
    """
    try:
        session_uuid = UUID(request.session_id)
    except ValueError:
        return ResolveUrlResponse(
            success=False,
            message=f"Invalid session ID format: {request.session_id}"
        )
    
    try:
        # 从URL中提取信息
        parsed = urlparse(request.oss_url)
        path = parsed.path.lstrip('/')
        path_parts = path.split('/')
        
        # 检查路径中是否包含artifacts/
        artifacts_index = None
        for i, part in enumerate(path_parts):
            if part == 'artifacts':
                artifacts_index = i
                break
        
        if artifacts_index is not None:
            # 找到artifacts/，提取object_key
            object_key = '/'.join(path_parts[artifacts_index:])
            # 验证object_key格式：artifacts/{session_id}/{file_name}
            key_parts = object_key.split('/')
            if len(key_parts) >= 3 and key_parts[0] == 'artifacts':
                # 验证session_id是否匹配
                try:
                    url_session_id = UUID(key_parts[1])
                    if url_session_id == session_uuid:
                        file_name = '/'.join(key_parts[2:])
                        return ResolveUrlResponse(
                            success=True,
                            object_key=object_key,
                            session_id=str(session_uuid),
                            file_name=file_name,
                        )
                except (ValueError, IndexError):
                    pass
        
        # 如果路径中没有artifacts/，尝试通过数据库查询匹配
        from sqlalchemy import select
        from web.session.model import SessionArtifact
        
        # 从URL路径提取可能的文件名
        if path_parts:
            file_name = path_parts[-1]
            
            # 查询匹配的artifact记录
            stmt = select(SessionArtifact).where(
                SessionArtifact.session_id == session_uuid
            ).where(
                (SessionArtifact.original_file_name == file_name) |
                (SessionArtifact.file_name.endswith(f"/{file_name}")) |
                (SessionArtifact.file_name == f"artifacts/{session_uuid}/{file_name}")
            )
            result = await db.execute(stmt)
            artifact = result.scalar_one_or_none()
            
            if artifact and artifact.file_name.startswith("artifacts/"):
                file_name_only = os.path.basename(artifact.file_name)
                return ResolveUrlResponse(
                    success=True,
                    object_key=artifact.file_name,
                    session_id=str(session_uuid),
                    file_name=file_name_only,
                )
        
        # 如果无法匹配，尝试从URL路径构建object_key
        if path_parts:
            file_name = path_parts[-1]
            object_key = f"artifacts/{session_uuid}/{file_name}"
            return ResolveUrlResponse(
                success=True,
                object_key=object_key,
                session_id=str(session_uuid),
                file_name=file_name,
            )
        
        return ResolveUrlResponse(
            success=False,
            message="Unable to resolve OSS URL to object_key"
        )
            
    except Exception as e:
        logger.warning(f"解析OSS URL失败: {e}")
        return ResolveUrlResponse(
            success=False,
            message=f"Failed to resolve OSS URL: {str(e)}"
        )


@router.get("/proxy/{session_id}/{file_name:path}")
async def proxy_file(
    session_id: str,
    file_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    代理OSS文件访问（用于工具直接访问）
    
    这个接口允许工具通过HTTP请求访问OSS文件，返回文件内容
    工具可以使用这个URL代替本地文件路径
    
    Args:
        session_id: 会话ID
        file_name: 文件名
        
    Returns:
        文件内容（二进制流）
    """
    return await download_file(session_id, file_name, db)

