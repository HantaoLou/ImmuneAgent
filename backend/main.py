import json
import os
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import asyncio
import threading

from config import CORS_ORIGINS, API_HOST, API_PORT, API_RELOAD
from agent_service import (
    create_global_state,
    invoke_agent_sync,
    format_agent_response,
    generate_session_id,
    collect_sandbox_output_files,
)
from progress_tracker import (
    create_progress_tracker,
    get_progress_tracker,
    remove_progress_tracker,
)
from session_storage import get_session_storage

_session_metadata_cache = {}


def save_session_metadata(session_id: str, metadata: dict):
    metadata_file = Path(f"./sandbox/sessions/{session_id}/metadata.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    _session_metadata_cache[session_id] = metadata


def get_session_metadata(session_id: str) -> dict:
    if session_id in _session_metadata_cache:
        return _session_metadata_cache[session_id]

    metadata_file = Path(f"./sandbox/sessions/{session_id}/metadata.json")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            _session_metadata_cache[session_id] = metadata
            return metadata
    return {}


app = FastAPI(
    title="Bio-Agent API",
    description="Backend API for Bio-Agent Demo System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    task_type: str
    result: dict


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "bio-agent-backend",
        "version": "1.0.0",
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with SSE streaming"""

    async def event_generator():
        session_id = request.session_id or generate_session_id()
        tracker = None
        agent_result = None

        try:
            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "init",
                        "message": "Initializing agent...",
                        "session_id": session_id,
                    }
                ),
            }

            # 创建进度跟踪器
            tracker = create_progress_tracker(session_id)
            progress_callback = tracker.create_callback()

            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "creating_state",
                        "message": "Creating agent state...",
                        "session_id": session_id,
                    }
                ),
            }

            await asyncio.sleep(0.1)

            # 创建状态，传入进度回调
            state = create_global_state(request.message, session_id, progress_callback)

            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "invoking_agent",
                        "message": "Starting agent execution...",
                        "session_id": session_id,
                    }
                ),
            }

            # 在后台线程中执行 agent
            agent_task = asyncio.create_task(
                asyncio.to_thread(invoke_agent_sync, state)
            )

            # 同时监听进度事件和 agent 完成
            while not agent_task.done():
                # 检查是否有进度事件
                progress_event = await tracker.get_event(timeout=0.1)
                if progress_event:
                    yield {
                        "event": "progress",
                        "data": json.dumps(progress_event.model_dump()),
                    }

                await asyncio.sleep(0.05)

            # 获取 agent 结果
            agent_result = agent_task.result()

            # 发送剩余的进度事件
            while True:
                progress_event = await tracker.get_event(timeout=0.01)
                if not progress_event:
                    break
                yield {
                    "event": "progress",
                    "data": json.dumps(progress_event.model_dump()),
                }

            yield {
                "event": "status",
                "data": json.dumps(
                    {
                        "stage": "formatting_response",
                        "message": "Formatting response...",
                    }
                ),
            }

            response = format_agent_response(agent_result)

            if response.get("session_id"):
                session_metadata = {
                    "opensandbox_id": response.get("result", {})
                    .get("merged_result", {})
                    .get("opensandbox_id"),
                    "sandbox_data_dir": response.get("result", {})
                    .get("merged_result", {})
                    .get("sandbox_data_dir"),
                    "sandbox_output_dir": response.get("result", {})
                    .get("merged_result", {})
                    .get("sandbox_output_dir"),
                    "session_id": response.get("session_id"),
                }
                save_session_metadata(response["session_id"], session_metadata)

            # 发送输出文件列表
            if response.get("output_files"):
                yield {
                    "event": "output_files",
                    "data": json.dumps(
                        {
                            "files": response["output_files"],
                            "count": response["output_files_count"],
                            "session_id": session_id,
                        }
                    ),
                }

            yield {"event": "done", "data": json.dumps(response)}

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                ),
            }
        finally:
            # 清理进度跟踪器
            if session_id and tracker:
                remove_progress_tracker(session_id)

    return EventSourceResponse(event_generator())


@app.get("/api/sessions")
async def get_sessions():
    """Get list of all sessions"""
    storage = get_session_storage()
    sessions = storage.get_session_list()
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get all messages for a session"""
    storage = get_session_storage()
    messages = storage.get_messages(session_id)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages"""
    storage = get_session_storage()
    success = storage.delete_session(session_id)
    if success:
        return {"status": "success", "message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.get("/api/sessions/{session_id}/files")
async def get_session_files(session_id: str):
    """Get list of output files for a session"""
    sandbox_dir = f"./sandbox/sessions/{session_id}"

    if not os.path.exists(sandbox_dir):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    files = collect_sandbox_output_files(sandbox_dir)

    return {"session_id": session_id, "files": files, "count": len(files)}


@app.get("/api/sessions/{session_id}/files/download")
async def download_session_file(session_id: str, file_path: str):
    """
    Download a specific file from sandbox output directory

    Supports both local and remote (OpenSandbox) file retrieval.

    Args:
        session_id: Session ID
        file_path: Relative path to the file (e.g., "output/results.csv")
    """
    sandbox_dir = Path(f"./sandbox/sessions/{session_id}")
    full_path = (sandbox_dir / file_path).resolve() if sandbox_dir.exists() else None

    # 安全检查
    if full_path:
        try:
            full_path.relative_to(sandbox_dir.resolve())
        except ValueError:
            raise HTTPException(
                status_code=403, detail="Access denied: file path outside sandbox"
            )

    # 尝试从本地读取
    if full_path and full_path.exists() and full_path.is_file():
        ext = full_path.suffix.lower()
        content_type = _get_content_type(ext)

        return FileResponse(
            path=str(full_path),
            media_type=content_type,
            filename=full_path.name,
            headers={"Content-Disposition": f'attachment; filename="{full_path.name}"'},
        )

    # 本地不存在，尝试从远程沙盒读取
    metadata = get_session_metadata(session_id)
    opensandbox_id = metadata.get("opensandbox_id")
    sandbox_data_dir = metadata.get("sandbox_data_dir") or metadata.get(
        "sandbox_output_dir", ""
    ).rstrip("/output")

    if opensandbox_id and sandbox_data_dir:
        try:
            remote_file_path = f"{sandbox_data_dir}/{file_path}"

            agent_utils_dir = os.path.join(
                os.path.dirname(__file__), "..", "agent", "utils"
            )
            import sys

            if agent_utils_dir not in sys.path:
                sys.path.insert(0, agent_utils_dir)

            from opensandbox_helper import OpenSandboxHelper

            helper = OpenSandboxHelper()
            file_content = await helper.read_file(
                remote_file_path, sandbox_id=opensandbox_id
            )

            if file_content:
                filename = os.path.basename(file_path)
                ext = os.path.splitext(filename)[1].lower()
                content_type = _get_content_type(ext)

                from fastapi.responses import Response

                return Response(
                    content=file_content
                    if isinstance(file_content, bytes)
                    else file_content.encode("utf-8"),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"'
                    },
                )
        except Exception as e:
            print(f"[download_session_file] Failed to fetch from remote sandbox: {e}")
            import traceback

            traceback.print_exc()

    # 既没有本地文件也没有远程沙盒
    if not sandbox_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")


def _get_content_type(ext: str) -> str:
    """Get content type based on file extension"""
    content_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tsv": "text/tab-separated-values",
        ".fasta": "text/plain",
        ".fa": "text/plain",
    }
    return content_types.get(ext, "application/octet-stream")


@app.get("/api/sessions/{session_id}/files/preview")
async def preview_session_file(session_id: str, file_path: str, max_lines: int = 100):
    """
    Preview a text file from sandbox output directory

    Args:
        session_id: Session ID
        file_path: Relative path to the file
        max_lines: Maximum number of lines to preview (default 100)
    """
    sandbox_dir = Path(f"./sandbox/sessions/{session_id}")

    if not sandbox_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # 安全检查
    full_path = (sandbox_dir / file_path).resolve()

    try:
        full_path.relative_to(sandbox_dir.resolve())
    except ValueError:
        raise HTTPException(
            status_code=403, detail="Access denied: file path outside sandbox"
        )

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if not full_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_path}")

    # 检查文件大小，限制预览大文件
    file_size = full_path.stat().st_size
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=413, detail="File too large to preview")

    ext = full_path.suffix.lower()

    # 二进制文件不支持预览
    binary_extensions = {".h5ad", ".rds", ".pdf", ".png", ".jpg", ".jpeg"}
    if ext in binary_extensions:
        return {
            "session_id": session_id,
            "file_path": file_path,
            "preview": f"[Binary file: {ext}, use download endpoint]",
            "file_type": ext,
            "file_size": file_size,
        }

    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"... (truncated, total {i}+ lines)")
                    break
                lines.append(line.rstrip("\n\r"))

        return {
            "session_id": session_id,
            "file_path": file_path,
            "preview": "\n".join(lines),
            "line_count": len(lines),
            "file_type": ext,
            "file_size": file_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bio-Agent API",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )
