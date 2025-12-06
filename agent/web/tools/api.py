from typing import Any, Dict

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from common.util.mcp_utils import mcp_tool_async
from web.db.db import get_db
from web.session.service import SessionArtifactService
from uuid import UUID
from langchain_core.runnables import RunnableConfig
from common.factory import get_mcp_client
from config.config import ApplicationConfig
from urllib.parse import urlparse
import os
import json
from fastapi import HTTPException
from fastapi.responses import FileResponse
from web.storage.storage_factory import get_storage_service

# 简要介绍（可按需补充/调整）
SERVICE_ABOUT: dict[str, str] = {
    "af3": "AlphaFold3 structure prediction and complex modeling for proteins and antibodies/ligands.",
    "imm": "Immunology analysis service integrating retrieval, planning, and experiment design.",
    "fdg": "Energy/stability evaluation (e.g., ΔΔG) for mutation stability and affinity changes.",
    "metabcr": "MetaBCR for antibody BCR modeling and prediction, supporting affinity/epitope tasks.",
    "airr": "AIRR Data Commons access to millions of BCR sequences from multiple repositories, with IgBLAST-compatible output format.",
    "anarci": "Antibody numbering service using ANARCI for multiple numbering schemes (Chothia, Kabat, IMGT, etc.) and germline assignment.",
    "sabdab": "Structural Antibody Database download service for CSV summaries, PDB structures, and dataset exports with various numbering schemes.",
    "bcell": "B cell single-cell RNA-seq analysis service for clonotypes, SHM, lineage, cell type identification, and visualization.",
    "geo": "Gene Expression Omnibus (GEO) database access service for downloading and querying gene expression datasets.",
    "lgblast": "Light chain IgBLAST analysis service for V(D)J recombination analysis of light chain sequences.",
    "oas": "Observed Antibody Space database service for accessing and analyzing large-scale antibody sequence datasets.",
    "scrna": "Single-cell RNA sequencing (scRNA-seq) analysis service for quality control, preprocessing, dimensionality reduction, clustering, differential expression, marker detection, and pathway enrichment.",
    "annotation": "Gene and sequence annotation service for functional annotation, gene ontology analysis, and pathway mapping.",
    "communication": "Inter-service communication and data exchange service for coordinating multi-tool workflows.",
    "multimodal": "Multi-modal analysis service integrating genomics, transcriptomics, and proteomics data for comprehensive analysis.",
    "bioinformatics": "Bioinformatics analysis including sequence alignment, gene annotation, DE analysis, pathway enrichment, and visualization.",
}


router = APIRouter(prefix="/api/tools")


class InvokeRequest(BaseModel):
    service_id: str
    tool_name: str
    params: Dict[str, Any] = {}
    session_id: UUID | None = None
    filename: str | None = None


@router.post("/invoke")
async def invoke_tool(req: InvokeRequest, db: AsyncSession = Depends(get_db)):
    # 工具可以直接使用预签名URL，无需转换
    # pandas等库支持直接读取HTTP URL
    result = await mcp_tool_async(req.service_id, req.tool_name, req.params)

    artifact = None
    try:
        if req.session_id is not None:
            # 序列化结果
            if isinstance(result, (dict, list)):
                content_bytes = (json.dumps(result, ensure_ascii=False, indent=2)).encode()
                fname = req.filename or f"{req.tool_name}_result.json"
                content_type = "application/json"
                art = await SessionArtifactService.create_artifact_with_binary(
                    db,
                    req.session_id,
                    fname,
                    content_bytes,
                    fname,
                    content_type,
                )
                artifact = {
                    "id": str(art.id),
                    "filename": fname,
                    "url": f"/api/sessions/artifacts/{art.id}/download",
                    "content_type": content_type,
                    "size": len(content_bytes),
                }
            else:
                # 若工具返回的是文件路径，则直接以二进制创建会话附件
                if isinstance(result, str) and os.path.isfile(result):
                    fname = req.filename or os.path.basename(result)
                    content_type = "application/octet-stream"
                    with open(result, "rb") as f:
                        content_bytes = f.read()
                    art = await SessionArtifactService.create_artifact_with_binary(
                        db,
                        req.session_id,
                        fname,
                        content_bytes,
                        fname,
                        content_type,
                    )
                    artifact = {
                        "id": str(art.id),
                        "filename": fname,
                        "url": f"/api/sessions/artifacts/{art.id}/download",
                        "content_type": content_type,
                        "size": len(content_bytes),
                    }
                else:
                    # 其他纯文本结果
                    content_bytes = str(result).encode()
                    fname = req.filename or f"{req.tool_name}_result.txt"
                    content_type = "text/plain"
                    art = await SessionArtifactService.create_artifact_with_binary(
                        db,
                        req.session_id,
                        fname,
                        content_bytes,
                        fname,
                        content_type,
                    )
                    artifact = {
                        "id": str(art.id),
                        "filename": fname,
                        "url": f"/api/sessions/artifacts/{art.id}/download",
                        "content_type": content_type,
                        "size": len(content_bytes),
                    }
    except Exception:
        # 失败不影响主结果返回
        artifact = None

    # 如果没有 session 也希望前端可下载文件：
    if artifact is None and isinstance(result, str) and os.path.isfile(result):
        artifact = {
            "id": None,
            "filename": os.path.basename(result),
            "url": f"/api/tools/download?path={result}",
            "content_type": "application/octet-stream",
            "size": os.path.getsize(result),
        }

    return {"ok": True, "result": result, "artifact": artifact}


@router.get("/services")
async def list_services():
    """列出可用的 MCP 服务（包含元信息）"""
    cfg = ApplicationConfig.get_instance()
    items = []
    for sid, meta in cfg.mcp_servers.items():
        items.append({
            "id": sid,
            "transport": meta.get("transport"),
            "url": meta.get("url"),
            "timeout": meta.get("timeout"),
            "about": SERVICE_ABOUT.get(sid, "MCP 服务"),
            "host": (lambda u: (urlparse(u).netloc if isinstance(u, str) else None))(meta.get("url")),
        })
    return {"ok": True, "services": items}


def extract_enhanced_schema(args_schema) -> dict:
    """
    从工具的 args_schema 中提取增强的 schema，保留 Pydantic Field 中通过 json_schema_extra 定义的元数据
    
    Args:
        args_schema: 工具的 args_schema（可能是 Pydantic BaseModel 或 None）
        
    Returns:
        增强后的 JSON Schema 字典，保留所有原始元数据（包括 json_schema_extra 中的信息）
    """
    if args_schema is None:
        return None
    
    # 将 Pydantic Model 转换为 JSON Schema（这会自动保留 json_schema_extra 中的信息）
    if hasattr(args_schema, "model_json_schema"):
        schema_dict = args_schema.model_json_schema()
    elif isinstance(args_schema, dict):
        schema_dict = args_schema.copy()
    else:
        # 尝试其他方式获取 schema
        try:
            schema_dict = args_schema.model_dump() if hasattr(args_schema, "model_dump") else {}
        except:
            return None
    
    if not isinstance(schema_dict, dict) or "properties" not in schema_dict:
        return schema_dict
    
    # json_schema_extra 中的内容会被自动保留在 schema 的 properties 中
    # 不需要额外处理，因为 Pydantic 的 model_json_schema() 已经包含了这些信息
    
    return schema_dict


@router.get("/list")
async def list_tools(service_id: str = Query(..., description="MCP 服务ID")):
    """列出某个服务的工具列表（名称与可选的描述/参数架构）"""
    rc = RunnableConfig(configurable={"mcp_config": {"service_ids": [service_id]}})
    client = await get_mcp_client(rc)
    tools = await client.get_tools()
    items = []
    for t in tools:
        # 获取原始 args_schema
        raw_schema = getattr(t, "args_schema", None)
        
        # 提取增强的 schema（保留 json_schema_extra 中的元数据）
        enhanced_schema = extract_enhanced_schema(raw_schema)
        
        items.append({
            "name": getattr(t, "name", "unknown"),
            "description": getattr(t, "description", ""),
            "args_schema": enhanced_schema,
        })
    return {"ok": True, "service_id": service_id, "tools": items}


@router.get("/download")
async def download_tool_file(path: str):
    """下载工具产生的文件（仅内部使用）。
    安全考虑：仅允许绝对路径且文件存在；可按需增加白名单目录校验。
    """
    # 简单安全校验
    if not isinstance(path, str) or not os.path.isabs(path):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    filename = os.path.basename(path)
    return FileResponse(path, filename=filename)


