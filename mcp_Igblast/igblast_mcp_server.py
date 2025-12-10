"""
IgBLAST MCP Server - V(D)J Analysis Tool Wrapper

This server exposes the IgBLAST + ChangeO pipeline via MCP protocol.
V(D)J recombination analysis using IgBLAST + ChangeO pipeline.
Returns AIRR format output - NO hardcoded V(D)J logic.
"""

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import StreamingResponse
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
from collections.abc import AsyncIterator
import uuid
import time
import os
from pydantic import BaseModel, Field, field_validator
import logging
import re
import asyncio
import threading
import json
import inspect
from mcp.types import TextContent, CallToolRequest, ServerResult, CallToolResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('IgBLAST_MCP')
# Import configuration
from config.config import TEMP_DIR as TEMP_FILES_DIR, IGBLAST_ROOT, IGBLAST_DB, IGBLAST_OPTIONAL, OUTPUT_DIR

# Create MCP server
mcp = FastMCP("IgBLAST V(D)J Analysis Server")

# 全局任务状态存储（用于自定义 SSE 端点）
_task_streams: dict[str, asyncio.Queue] = {}
_task_streams_lock = threading.Lock()

# 添加自定义 SSE 流式传输端点
@mcp.custom_route("/stream/{task_id}", methods=["GET"])
async def stream_task_progress(request: Request):
    """
    自定义 SSE 端点，用于流式传输任务进度
    
    客户端通过 GET /stream/{task_id} 连接，服务器会通过 SSE 推送进度消息
    """
    # 从路径参数获取 task_id
    task_id = request.path_params.get("task_id")
    if not task_id:
        from starlette.responses import JSONResponse
        return JSONResponse(
            {"error": "task_id is required in path"},
            status_code=400
        )
    
    # 检查任务是否存在
    with _task_streams_lock:
        if task_id not in _task_streams:
            from starlette.responses import JSONResponse
            return JSONResponse(
                {"error": f"Task {task_id} not found"},
                status_code=404
            )
        queue = _task_streams[task_id]
    
    async def event_generator():
        """SSE 事件生成器"""
        try:
            while True:
                try:
                    # 从队列获取消息（超时 1 秒）
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # 检查是否是结束标记
                    if message is None:
                        # 发送结束事件
                        yield f"data: {json.dumps({'type': 'end'})}\n\n"
                        break
                    
                    # 发送消息
                    yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
                    
                except asyncio.TimeoutError:
                    # 发送心跳，保持连接
                    yield ": heartbeat\n\n"
                    continue
        except Exception as e:
            logger.error(f"[SSE流式传输] 错误 (任务: {task_id}): {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # 清理任务队列
            with _task_streams_lock:
                if task_id in _task_streams:
                    del _task_streams[task_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# 实现真正的流式传输：通过猴子补丁修改底层服务器的 handler
if CallToolRequest in mcp._mcp_server.request_handlers:
    _original_handler = mcp._mcp_server.request_handlers[CallToolRequest]
    
    # Monkey patch RequestResponder.respond 来追踪响应发送
    from mcp.shared.session import RequestResponder
    _original_respond = RequestResponder.respond
    
    async def _patched_respond(self, response):
        """追踪响应发送的 monkey patch"""
        request_id = getattr(self, 'request_id', 'unknown')
        respond_start_time = time.time()
        logger.info(f"[流式传输] 🔵 RequestResponder.respond 被调用 (request_id: {request_id}, response_type: {type(response)})")
        try:
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
                logger.info(f"[流式传输] 🔵 响应内容预览: {json.dumps(response_dict, ensure_ascii=False, default=str)[:500]}...")
            else:
                logger.info(f"[流式传输] 🔵 响应内容: {str(response)[:500]}...")
        except Exception as e:
            logger.warning(f"[流式传输] 🔵 无法序列化响应: {str(e)}")
        
        result = await _original_respond(self, response)
        respond_elapsed = time.time() - respond_start_time
        logger.info(f"[流式传输] ✅ RequestResponder.respond 完成 (request_id: {request_id}, 耗时: {respond_elapsed:.3f}秒)")
        if respond_elapsed > 0.1:
            logger.warning(f"[流式传输] ⚠️ RequestResponder.respond 耗时过长 (request_id: {request_id}, 耗时: {respond_elapsed:.3f}秒)，可能被阻塞")
        return result
    
    RequestResponder.respond = _patched_respond
    logger.info("[流式传输] ✅ 已安装 RequestResponder.respond monkey patch")
    
    # Monkey patch _send_response 来追踪消息发送
    from mcp.shared.session import BaseSession
    _original_send_response = BaseSession._send_response
    
    async def _patched_send_response(self, request_id, response):
        """追踪消息发送的 monkey patch"""
        send_start_time = time.time()
        logger.info(f"[流式传输] 🔵 BaseSession._send_response 被调用 (request_id: {request_id}, response_type: {type(response)})")
        
        try:
            try:
                result = await asyncio.wait_for(
                    _original_send_response(self, request_id, response),
                    timeout=1.0
                )
                send_elapsed = time.time() - send_start_time
                logger.info(f"[流式传输] ✅ BaseSession._send_response 完成 (request_id: {request_id}, 耗时: {send_elapsed:.3f}秒)")
                if send_elapsed > 0.1:
                    logger.warning(f"[流式传输] ⚠️ BaseSession._send_response 耗时过长 (request_id: {request_id}, 耗时: {send_elapsed:.3f}秒)，可能被阻塞")
                return result
            except asyncio.TimeoutError:
                logger.error(f"[流式传输] ❌ BaseSession._send_response 超时 (request_id: {request_id})")
                result = await _original_send_response(self, request_id, response)
                send_elapsed = time.time() - send_start_time
                logger.info(f"[流式传输] ✅ BaseSession._send_response 最终完成 (request_id: {request_id}, 总耗时: {send_elapsed:.3f}秒)")
                return result
        except Exception as e:
            logger.error(f"[流式传输] ❌ BaseSession._send_response 异常 (request_id: {request_id}): {str(e)}", exc_info=True)
            raise
    
    try:
        from mcp.server.lowlevel.server import ServerSession
        ServerSession._send_response = _patched_send_response
        logger.info("[流式传输] ✅ 已安装 ServerSession._send_response monkey patch")
    except Exception as e:
        logger.warning(f"[流式传输] ⚠️ 无法安装 ServerSession._send_response monkey patch: {str(e)}")
    
    # Monkey patch _handle_request 来追踪响应传递
    _original_handle_request = mcp._mcp_server._handle_request
    
    async def _patched_handle_request(self, message, req, session, lifespan_context, raise_exceptions):
        """追踪请求处理的 monkey patch"""
        request_id = getattr(message, 'request_id', 'unknown')
        logger.info(f"[流式传输] 🔵 _handle_request 开始 (request_id: {request_id}, request_type: {type(req).__name__})")
        
        try:
            result = await _original_handle_request(message, req, session, lifespan_context, raise_exceptions)
            logger.info(f"[流式传输] ✅ _handle_request 完成 (request_id: {request_id})")
            return result
        except Exception as e:
            logger.error(f"[流式传输] ❌ _handle_request 异常 (request_id: {request_id}): {str(e)}", exc_info=True)
            raise
    
    import types
    mcp._mcp_server._handle_request = types.MethodType(_patched_handle_request, mcp._mcp_server)
    logger.info("[流式传输] ✅ 已安装 _handle_request monkey patch")
    
    # 创建新的 handler，它能够处理异步生成器并实现真正的流式传输
    async def _streaming_handler(req: CallToolRequest):
        """支持异步生成器的 handler，实现真正的流式传输"""
        import jsonschema
        
        handler_start_time = time.time()
        logger.info(f"[流式传输] Handler 开始处理请求 (工具: {req.params.name if hasattr(req, 'params') else 'unknown'}, 请求ID: {req.id if hasattr(req, 'id') else 'unknown'})")
        
        try:
            tool_name = req.params.name
            arguments = req.params.arguments or {}
            tool = await mcp._mcp_server._get_cached_tool_definition(tool_name)
            
            # input validation
            if tool:
                try:
                    jsonschema.validate(instance=arguments, schema=tool.inputSchema)
                except jsonschema.ValidationError as e:
                    return mcp._mcp_server._make_error_result(f"Input validation error: {e.message}")
            
            # tool call - 使用原始的 call_tool，但不转换结果
            context = mcp.get_context()
            results = await mcp._tool_manager.call_tool(tool_name, arguments, context=context, convert_result=False)
            
            logger.info(f"[流式传输] 工具调用完成 (工具: {tool_name}), 返回类型: {type(results)}, 是否为异步生成器: {inspect.isasyncgen(results)}")
            
            # 检查是否是异步生成器
            if inspect.isasyncgen(results):
                logger.info(f"[流式传输] 检测到异步生成器 (工具: {tool_name})")
                
                # 1. 生成任务 ID
                task_id = str(uuid.uuid4())
                
                # 2. 创建消息队列
                message_queue = asyncio.Queue()
                with _task_streams_lock:
                    _task_streams[task_id] = message_queue
                
                logger.info(f"[流式传输] 创建任务流 (工具: {tool_name}, 任务ID: {task_id})")
                
                # 3. 在后台任务中收集消息并推送到队列
                async def collect_and_push():
                    try:
                        logger.info(f"[流式传输] 开始收集异步生成器的消息 (工具: {tool_name}, 任务ID: {task_id})")
                        async for item in results:
                            if isinstance(item, dict):
                                await message_queue.put(item)
                                logger.info(f"[流式传输] 推送消息到队列 (工具: {tool_name}, 任务ID: {task_id}): {json.dumps(item, ensure_ascii=False)[:200]}...")
                            else:
                                await message_queue.put({"type": "data", "content": str(item)})
                        await message_queue.put(None)
                        logger.info(f"[流式传输] 消息收集完成 (工具: {tool_name}, 任务ID: {task_id})")
                    except Exception as e:
                        logger.error(f"[流式传输] 收集消息时出错 (工具: {tool_name}, 任务ID: {task_id}): {str(e)}", exc_info=True)
                        await message_queue.put({"type": "error", "message": str(e)})
                        await message_queue.put(None)
                
                # 4. 在新线程中启动后台任务
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        logger.info(f"[流式传输] 在新线程中启动事件循环 (工具: {tool_name}, 任务ID: {task_id})")
                        new_loop.run_until_complete(collect_and_push())
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()
                logger.info(f"[流式传输] 已在新线程中启动收集任务 (工具: {tool_name}, 任务ID: {task_id})")
                
                # 5. 返回任务 ID 和流式端点 URL
                host = mcp.settings.host or "localhost"
                port = mcp.settings.port or 8110
                
                if host == "0.0.0.0":
                    actual_host = os.getenv("MCP_SERVER_HOST", "localhost")
                    host = actual_host
                
                base_url = f"http://{host}:{port}"
                stream_url = f"{base_url}/stream/{task_id}"
                
                logger.info(f"[流式传输] 返回流式端点 (工具: {tool_name}, 任务ID: {task_id}, URL: {stream_url})")
                
                # 立即返回任务 ID 和流式端点 URL
                try:
                    result = ServerResult(
                        CallToolResult(
                            content=[TextContent(
                                type="text",
                                text=json.dumps({
                                    "type": "streaming_task",
                                    "task_id": task_id,
                                    "stream_url": stream_url,
                                    "message": "任务已启动，请通过 SSE 端点接收进度更新。客户端应连接到 stream_url 来接收实时进度消息。"
                                }, ensure_ascii=False)
                            )],
                            structuredContent={
                                "type": "streaming_task",
                                "task_id": task_id,
                                "stream_url": stream_url,
                                "message": "任务已启动，请通过 SSE 端点接收进度更新"
                            },
                            isError=False,
                        )
                    )
                    
                    handler_elapsed = time.time() - handler_start_time
                    logger.info(f"[流式传输] 正在返回 ServerResult (工具: {tool_name}, 任务ID: {task_id}, Handler耗时: {handler_elapsed:.3f}秒)")
                    logger.info(f"[流式传输] ✅ Handler 返回结果，应该立即发送给客户端 (工具: {tool_name}, 任务ID: {task_id})")
                    
                    return result
                except Exception as e:
                    logger.error(f"[流式传输] 创建返回结果时出错 (工具: {tool_name}, 任务ID: {task_id}): {str(e)}", exc_info=True)
                    return mcp._mcp_server._make_error_result(f"Failed to create streaming task result: {str(e)}")
            else:
                # 非异步生成器，使用原始 handler
                return await _original_handler(req)
        except Exception as e:
            logger.error(f"[流式传输] Handler 处理请求时出错: {str(e)}", exc_info=True)
            return mcp._mcp_server._make_error_result(f"Handler error: {str(e)}")
    
    # 替换原始 handler
    mcp._mcp_server.request_handlers[CallToolRequest] = _streaming_handler
    logger.info("[流式传输] ✅ 已安装自定义 streaming handler")

# Configuration paths
IGBLAST_BIN = "igblastn"  # Use conda-installed igblastn
# Use ChangeO scripts - try multiple locations
# Check conda installation first, then local copy
_conda_makedb = Path("/data_new/lht/.conda/envs/antibody_venv/bin/MakeDb.py")
_local_makedb = Path("/data_new/hd/server/mcp_Igblast/igblast_changeO/MakeDb.py")

if _conda_makedb.exists():
    CHANGEO_MAKEDB = str(_conda_makedb)
elif _local_makedb.exists():
    CHANGEO_MAKEDB = str(_local_makedb)
else:
    CHANGEO_MAKEDB = "MakeDb.py"  # Fall back to PATH

def sanitize_fasta_file(input_fasta: Path, output_fasta: Path, session_id: str) -> Tuple[Path, int]:
    """
    清理FASTA文件，移除无效的核苷酸字符。
    
    Args:
        input_fasta: 输入FASTA文件路径
        output_fasta: 输出FASTA文件路径
        session_id: 会话ID用于日志
        
    Returns:
        (清理后的文件路径, 被替换的字符数)
    """
    valid_nucleotides = set('ATCGN-')
    replaced_count = 0
    sequences_processed = 0
    sequences_cleaned = 0
    
    with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
        current_seq_id = None
        current_seq = []
        
        for line in infile:
            line = line.rstrip()
            if line.startswith('>'):
                # 保存前一个序列
                if current_seq_id is not None:
                    sequences_processed += 1
                    seq = ''.join(current_seq).upper()
                    # 清理序列：将无效字符替换为N
                    cleaned_seq = ''.join(c if c in valid_nucleotides else 'N' for c in seq)
                    if cleaned_seq != seq:
                        sequences_cleaned += 1
                        replaced_count += sum(1 for c1, c2 in zip(seq, cleaned_seq) if c1 != c2)
                    outfile.write(f"{current_seq_id}\n{cleaned_seq}\n")
                
                # 开始新序列
                current_seq_id = line
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一个序列
        if current_seq_id is not None:
            sequences_processed += 1
            seq = ''.join(current_seq).upper()
            cleaned_seq = ''.join(c if c in valid_nucleotides else 'N' for c in seq)
            if cleaned_seq != seq:
                sequences_cleaned += 1
                replaced_count += sum(1 for c1, c2 in zip(seq, cleaned_seq) if c1 != c2)
            outfile.write(f"{current_seq_id}\n{cleaned_seq}\n")
    
    if replaced_count > 0:
        logger.warning(f"[{session_id}] Sanitized FASTA file: replaced {replaced_count} invalid characters in {sequences_cleaned}/{sequences_processed} sequences")
    else:
        logger.debug(f"[{session_id}] FASTA file is clean, no invalid characters found")
    
    return output_fasta, replaced_count

def split_fasta_file(input_fasta: Path, batch_size: int, output_dir: Path, session_id: str) -> List[Path]:
    """
    Split a FASTA file into smaller batches.
    
    Args:
        input_fasta: Input FASTA file path
        batch_size: Number of sequences per batch
        output_dir: Directory to save batch files
        session_id: Session ID for naming
        
    Returns:
        List of batch file paths
    """
    batch_files = []
    current_batch = []
    current_batch_num = 0
    
    with open(input_fasta, 'r') as f:
        current_seq_id = None
        current_seq = []
        
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_seq_id is not None:
                    current_batch.append((current_seq_id, ''.join(current_seq)))
                    
                    # Check if batch is full
                    if len(current_batch) >= batch_size:
                        # Write batch file
                        batch_file = output_dir / f"batch_{session_id}_{current_batch_num}.fasta"
                        with open(batch_file, 'w') as bf:
                            for seq_id, seq in current_batch:
                                bf.write(f"{seq_id}\n{seq}\n")
                        batch_files.append(batch_file)
                        current_batch = []
                        current_batch_num += 1
                
                # Start new sequence
                current_seq_id = line
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_seq_id is not None:
            current_batch.append((current_seq_id, ''.join(current_seq)))
        
        # Write last batch if not empty
        if current_batch:
            batch_file = output_dir / f"batch_{session_id}_{current_batch_num}.fasta"
            with open(batch_file, 'w') as bf:
                for seq_id, seq in current_batch:
                    bf.write(f"{seq_id}\n{seq}\n")
            batch_files.append(batch_file)
    
    return batch_files


def save_results_to_csv(results: List[Dict[str, Any]], session_id: str, output_dir: Path) -> str:
    """
    Save AIRR results to CSV file.
    
    Args:
        results: List of AIRR format dictionaries
        session_id: Session ID for filename
        output_dir: Output directory path
        
    Returns:
        Absolute path to the saved CSV file, or None if no results
    """
    if not results:
        logger.warning(f"[{session_id}] No results to save")
        return None
    
    try:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"airr_results_{session_id}_{timestamp}.csv"
        csv_path = output_dir / csv_filename
        
        df.to_csv(csv_path, index=False)
        logger.info(f"[{session_id}] Results saved to CSV: {csv_path} ({len(results)} records)")
        
        return str(csv_path.absolute())
    except Exception as e:
        logger.error(f"[{session_id}] Failed to save results to CSV: {str(e)}")
        return None


class AnalyzeVdjBatchArgs(BaseModel):
    """Parameters for V(D)J analysis"""
    sequences: Union[List[Dict[str, str]], str] = Field(
        ...,
        description="Sequences input: (1) List of sequences with id and sequence fields, or (2) Local FASTA file path",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["fasta"],
            "placeholder": 'Enter sequences like: [{"id": "seq1", "sequence": "ATGC..."}], or FASTA file path',
            "help_text": "NUCLEOTIDE sequences! You can Upload a FASTA file",
            "demo_urls": "/data_new/workspace/antibody_gen/mcp_Igblast/igblast_changeO/input/rsvH.fasta"
        }
    )
    
    @field_validator('sequences', mode='before')
    @classmethod
    def normalize_sequences(cls, v) -> Union[List[Dict[str, str]], str]:
        """规范化 sequences 输入，处理各种格式"""
        # 如果是列表
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("sequences cannot be an empty list")
            
            # 如果列表中只有一个元素
            if len(v) == 1:
                item = v[0]
                # 如果是字符串（URL 或路径），提取它
                if isinstance(item, str):
                    logger.debug(f"Normalizing sequences: list with single string -> string: {item[:50]}...")
                    return item
                # 如果是字典，返回字典列表格式
                if isinstance(item, dict):
                    return v
            
            # 如果列表中都是字符串（URL 或路径），提取第一个
            if all(isinstance(item, str) for item in v):
                logger.info(f"Normalizing sequences: list of {len(v)} strings -> using first string: {v[0][:50]}...")
                return v[0]
            
            # 如果列表中都是字典，保持不变
            if all(isinstance(item, dict) for item in v):
                return v
            
            # 混合类型，无法处理
            raise ValueError(f"sequences list contains mixed types. Expected all strings or all dicts, got: {type(v[0]).__name__}")
        
        # 如果是字符串，直接返回
        if isinstance(v, str):
            return v
        
        # 其他类型
        raise ValueError(f"sequences must be a string, list of strings, or list of dicts. Got: {type(v).__name__}")
    organism: str = Field(
        default="human",
        description="Organism type for germline database",
        json_schema_extra={
            "ui_type": "select",
            "options": ["human", "mouse", "rabbit", "rat", "rhesus", "pig"],
            "placeholder": "Select organism"
        }
    )
    receptor_type: str = Field(
        default="Ig",
        description="Receptor type",
        json_schema_extra={
            "ui_type": "select",
            "options": ["Ig", "TCR"],
            "placeholder": "Select receptor type"
        }
    )
    locus: str = Field(
        default="IGH",
        description="Locus type",
        json_schema_extra={
            "ui_type": "select",
            "options": ["IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD"],
            "placeholder": "Select locus"
        }
    )
    timeout: int = Field(
        default=7200,
        description="Timeout in seconds for IgBLAST and ChangeO execution (default: 7200 seconds = 2 hours). Increase for very large files.",
        json_schema_extra={
            "ui_type": "number",
            "min": 300,
            "max": 14400,
            "placeholder": "Timeout in seconds (default: 7200)"
        }
    )


# Internal batch processing configuration
_AUTO_BATCH_THRESHOLD = 500  # Automatically batch if more than this many sequences
_BATCH_SIZE = 500  # Number of sequences per batch


def _process_single_batch(
    fasta_file: Path,
    organism: str,
    receptor_type: str,
    locus: str,
    timeout: int,
    session_id: str
) -> List[Dict[str, Any]]:
    """
    Process a single batch of sequences.
    
    Args:
        fasta_file: FASTA file to process
        organism: Organism type
        receptor_type: Receptor type
        locus: Locus type
        timeout: Timeout in seconds
        session_id: Unique session identifier
        
    Returns:
        List of AIRR format results
        
    Raises:
        subprocess.CalledProcessError: If IgBLAST or ChangeO fails
        subprocess.TimeoutExpired: If processing times out
    """
    logger.debug(f"[{session_id}] Starting batch processing: {fasta_file}")
    batch_start_time = time.time()
    
    # 清理FASTA文件（移除无效字符）
    cleaned_fasta = None
    replaced_chars = 0
    try:
        cleaned_fasta_path = TEMP_FILES_DIR / f"cleaned_{session_id}.fasta"
        cleaned_fasta_path, replaced_chars = sanitize_fasta_file(fasta_file, cleaned_fasta_path, session_id)
        if replaced_chars > 0:
            logger.info(f"[{session_id}] Using cleaned FASTA file (replaced {replaced_chars} invalid characters)")
            fasta_file = cleaned_fasta_path
            cleaned_fasta = cleaned_fasta_path
    except Exception as e:
        logger.warning(f"[{session_id}] Failed to sanitize FASTA file, using original: {str(e)}")
        # 如果清理失败，继续使用原始文件
    
    # Run IgBLAST
    igblast_out = TEMP_FILES_DIR / f"igblast_output_{session_id}.txt"
    
    germline_v = IGBLAST_DB / f"imgt_{organism}_ig_v"
    germline_d = IGBLAST_DB / f"imgt_{organism}_ig_d"
    germline_j = IGBLAST_DB / f"imgt_{organism}_ig_j"
    aux_file = IGBLAST_OPTIONAL / f"{organism}_gl.aux"
    
    igblast_cmd = [
        IGBLAST_BIN,
        "-germline_db_V", str(germline_v),
        "-germline_db_D", str(germline_d),
        "-germline_db_J", str(germline_j),
        "-organism", organism,
        "-domain_system", "imgt",
        "-ig_seqtype", receptor_type,
        "-auxiliary_data", str(aux_file),
        "-query", str(fasta_file),
        "-show_translation",
        "-outfmt", "7 std qseq sseq btop",
        "-out", str(igblast_out)
    ]
    
    logger.info(f"[{session_id}] Running IgBLAST (organism={organism}, timeout={timeout}s)")
    logger.debug(f"[{session_id}] IgBLAST command: {' '.join(igblast_cmd)}")
    igblast_start = time.time()
    try:
        result = subprocess.run(igblast_cmd, check=True, timeout=timeout, capture_output=True, text=True)
        igblast_time = time.time() - igblast_start
        logger.info(f"[{session_id}] IgBLAST completed (elapsed: {igblast_time:.2f}s)")
        if result.stdout:
            logger.debug(f"[{session_id}] IgBLAST stdout: {result.stdout[:500]}")  # 只记录前500字符
        if result.stderr:
            logger.warning(f"[{session_id}] IgBLAST stderr: {result.stderr[:500]}")
    except subprocess.CalledProcessError as e:
        igblast_time = time.time() - igblast_start
        error_msg = f"IgBLAST failed with return code {e.returncode}"
        if e.stdout:
            logger.error(f"[{session_id}] IgBLAST stdout: {e.stdout[:1000]}")
        if e.stderr:
            logger.error(f"[{session_id}] IgBLAST stderr: {e.stderr[:1000]}")
        raise Exception(f"{error_msg}: {e.stderr[:500] if e.stderr else 'No error message'}")
    
    # Check IgBLAST output file
    if not igblast_out.exists():
        raise Exception(f"IgBLAST output file not created: {igblast_out}")
    if igblast_out.stat().st_size == 0:
        logger.warning(f"[{session_id}] IgBLAST output file is empty: {igblast_out}")
        # Read first few lines to see what's in the file
        try:
            with open(igblast_out, 'r') as f:
                first_lines = ''.join(f.readlines()[:10])
                logger.debug(f"[{session_id}] First 10 lines of IgBLAST output:\n{first_lines}")
        except:
            pass
    
    # Run ChangeO MakeDb
    changeo_prefix = f"changeo_{session_id}"
    v_fasta = IGBLAST_DB / f"imgt_{organism}_ig_v.fasta"
    d_fasta = IGBLAST_DB / f"imgt_{organism}_ig_d.fasta"
    j_fasta = IGBLAST_DB / f"imgt_{organism}_ig_j.fasta"
    
    # Check if reference files exist
    for ref_file in [v_fasta, d_fasta, j_fasta]:
        if not ref_file.exists():
            logger.warning(f"[{session_id}] Reference file not found: {ref_file}")
    
    # 验证输入文件存在
    if not igblast_out.exists():
        raise Exception(f"IgBLAST output file does not exist: {igblast_out}")
    if igblast_out.stat().st_size == 0:
        raise Exception(f"IgBLAST output file is empty: {igblast_out}")
    if not fasta_file.exists():
        raise Exception(f"FASTA file does not exist: {fasta_file}")
    
    # 验证参考文件存在
    missing_refs = []
    for ref_file in [v_fasta, d_fasta, j_fasta]:
        if not ref_file.exists():
            missing_refs.append(str(ref_file))
    if missing_refs:
        logger.warning(f"[{session_id}] Missing reference files: {missing_refs}")
        # 不抛出异常，ChangeO可能会处理这种情况
    
    # 检查IgBLAST输出是否有任何匹配
    has_hits = False
    if igblast_out.exists():
        try:
            with open(igblast_out, 'r') as f:
                content = f.read()
                # 检查是否有非注释行的匹配结果（不是"0 hits found"）
                lines = content.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        has_hits = True
                        break
                    # 检查是否有匹配的序列（不是"0 hits found"）
                    if "#" in line and "hits found" in line and "0 hits" not in line:
                        has_hits = True
                        break
        except:
            pass
    
    if not has_hits:
        logger.warning(f"[{session_id}] IgBLAST output appears to have no hits - ChangeO may not generate output files")
    
    makedb_cmd = [
        CHANGEO_MAKEDB,
        "igblast",
        "-i", str(igblast_out),
        "-r", str(v_fasta), str(d_fasta), str(j_fasta),
        "-s", str(fasta_file),
        "--format", "airr",
        "--partial",
        "--infer-junction",
        "--outdir", str(TEMP_FILES_DIR),
        "--outname", changeo_prefix
    ]
    
    # 如果可能没有匹配，添加--extend参数可能会帮助（但需要检查ChangeO版本）
    # 暂时不添加，先看看能否通过其他方式解决
    
    # 记录完整的命令和文件信息
    logger.info(f"[{session_id}] ChangeO command: {' '.join(makedb_cmd)}")
    logger.info(f"[{session_id}] IgBLAST output: {igblast_out} (size: {igblast_out.stat().st_size} bytes)")
    logger.info(f"[{session_id}] FASTA input: {fasta_file} (size: {fasta_file.stat().st_size} bytes)")
    logger.info(f"[{session_id}] Output directory: {TEMP_FILES_DIR}")
    logger.info(f"[{session_id}] Output prefix: {changeo_prefix}")
    
    logger.info(f"[{session_id}] Running ChangeO MakeDb")
    changeo_start = time.time()
    try:
        result = subprocess.run(makedb_cmd, check=True, timeout=timeout, capture_output=True, text=True)
        changeo_time = time.time() - changeo_start
        logger.info(f"[{session_id}] ChangeO completed (elapsed: {changeo_time:.2f}s)")
        # 始终记录完整的stdout和stderr用于调试
        changeo_stdout = result.stdout if result.stdout else ""
        changeo_stderr = result.stderr if result.stderr else ""
        
        if changeo_stdout:
            logger.info(f"[{session_id}] ChangeO stdout (full): {changeo_stdout}")
        else:
            logger.warning(f"[{session_id}] ChangeO stdout is empty")
        if changeo_stderr:
            logger.warning(f"[{session_id}] ChangeO stderr (full): {changeo_stderr}")
        else:
            logger.debug(f"[{session_id}] ChangeO stderr is empty")
        
        # 检查ChangeO是否真的生成了输出（从stdout中解析）
        output_none = "OUTPUT> None" in changeo_stdout or "OUTPUT>" in changeo_stdout and "None" in changeo_stdout.split("OUTPUT>")[1].split("\n")[0]
        if output_none:
            logger.warning(f"[{session_id}] ChangeO reported OUTPUT> None - no output file was generated")
            # 尝试从stdout中提取PASS和FAIL计数
            pass_count_match = None
            fail_count_match = None
            for line in changeo_stdout.split('\n'):
                if 'PASS>' in line:
                    try:
                        pass_count_match = int(line.split('PASS>')[1].strip().split()[0])
                    except:
                        pass
                if 'FAIL>' in line:
                    try:
                        fail_count_match = int(line.split('FAIL>')[1].strip().split()[0])
                    except:
                        pass
            if fail_count_match is not None:
                logger.warning(f"[{session_id}] ChangeO processed {fail_count_match} sequences but all failed (PASS: {pass_count_match or 0})")
                logger.warning(f"[{session_id}] This usually means IgBLAST found no matches for the input sequences")
    except subprocess.CalledProcessError as e:
        changeo_time = time.time() - changeo_start
        error_msg = f"ChangeO MakeDb failed with return code {e.returncode}"
        if e.stdout:
            logger.error(f"[{session_id}] ChangeO stdout: {e.stdout[:1000]}")
            if len(e.stdout) > 1000:
                logger.error(f"[{session_id}] ChangeO stdout (full): {e.stdout}")
        if e.stderr:
            logger.error(f"[{session_id}] ChangeO stderr: {e.stderr[:1000]}")
            if len(e.stderr) > 1000:
                logger.error(f"[{session_id}] ChangeO stderr (full): {e.stderr}")
        raise Exception(f"{error_msg}: {e.stderr[:500] if e.stderr else 'No error message'}")
    
    # Read results - ChangeO generates two files:
    # 1. db-pass.tsv: sequences that passed quality filters
    # 2. db-fail.tsv: sequences that failed quality filters
    # We should read both to include ALL sequences
    airr_pass_file = TEMP_FILES_DIR / f"{changeo_prefix}_db-pass.tsv"
    airr_fail_file = TEMP_FILES_DIR / f"{changeo_prefix}_db-fail.tsv"
    
    # List all files in temp directory for debugging
    logger.info(f"[{session_id}] Checking output directory: {TEMP_FILES_DIR}")
    try:
        temp_files = list(TEMP_FILES_DIR.glob(f"{changeo_prefix}*"))
        logger.info(f"[{session_id}] Files matching prefix '{changeo_prefix}': {[str(f.name) for f in temp_files]}")
        # 如果找不到预期的输出文件，列出所有文件
        if not airr_pass_file.exists() and not airr_fail_file.exists():
            all_files = list(TEMP_FILES_DIR.glob("*"))
            logger.warning(f"[{session_id}] ChangeO output files not found. All files in temp dir: {[str(f.name) for f in all_files[:20]]}")
            # 检查是否有任何新创建的文件（在ChangeO运行后）
            logger.info(f"[{session_id}] Checking for any recently created files in {TEMP_FILES_DIR}")
            try:
                import time as time_module
                current_time = time_module.time()
                recent_files = []
                for f in all_files:
                    try:
                        if f.is_file():
                            mtime = f.stat().st_mtime
                            if current_time - mtime < 10:  # 最近10秒内创建的文件
                                recent_files.append((f.name, current_time - mtime))
                    except:
                        pass
                if recent_files:
                    logger.info(f"[{session_id}] Recently created files: {recent_files}")
            except Exception as e:
                logger.warning(f"[{session_id}] Could not check file timestamps: {str(e)}")
    except Exception as e:
        logger.warning(f"[{session_id}] Could not list temp directory: {str(e)}")
    
    results = []
    pass_count = 0
    fail_count = 0
    
    # 检查是否有其他可能的输出文件名（ChangeO可能使用不同的命名）
    possible_pass_files = [
        airr_pass_file,
        TEMP_FILES_DIR / f"{changeo_prefix}_db-pass.tsv",
        TEMP_FILES_DIR / f"{changeo_prefix}_db-pass.txt",
        TEMP_FILES_DIR / f"db-pass.tsv",
        TEMP_FILES_DIR / f"db-pass.txt",
    ]
    possible_fail_files = [
        airr_fail_file,
        TEMP_FILES_DIR / f"{changeo_prefix}_db-fail.tsv",
        TEMP_FILES_DIR / f"{changeo_prefix}_db-fail.txt",
        TEMP_FILES_DIR / f"db-fail.tsv",
        TEMP_FILES_DIR / f"db-fail.txt",
    ]
    
    # 检查ChangeO的stdout中是否有OUTPUT>None的指示
    changeo_output_none = False
    if 'changeo_stdout' in locals() and changeo_stdout:
        if "OUTPUT> None" in changeo_stdout:
            changeo_output_none = True
            logger.warning(f"[{session_id}] ChangeO reported OUTPUT> None in stdout - no output file generated")
    
    # 尝试找到实际的输出文件
    actual_pass_file = None
    actual_fail_file = None
    for pf in possible_pass_files:
        if pf.exists():
            file_size = pf.stat().st_size
            if file_size > 0:
                actual_pass_file = pf
                logger.info(f"[{session_id}] Found db-pass file: {pf} (size: {file_size} bytes)")
                break
            else:
                logger.debug(f"[{session_id}] Found empty db-pass file: {pf}")
    
    for ff in possible_fail_files:
        if ff.exists():
            file_size = ff.stat().st_size
            if file_size > 0:
                actual_fail_file = ff
                logger.info(f"[{session_id}] Found db-fail file: {ff} (size: {file_size} bytes)")
                break
            else:
                logger.debug(f"[{session_id}] Found empty db-fail file: {ff}")
    
    # 如果ChangeO显示OUTPUT>None且没有找到文件，这是正常情况（所有序列都失败且无匹配）
    if changeo_output_none and not actual_pass_file and not actual_fail_file:
        logger.warning(f"[{session_id}] ChangeO did not generate output files because all sequences failed (likely no IgBLAST hits)")
    
    # Read passed sequences
    if actual_pass_file:
        file_size = actual_pass_file.stat().st_size
        logger.debug(f"[{session_id}] Reading passed sequences: {actual_pass_file} (size: {file_size} bytes)")
        if file_size == 0:
            logger.warning(f"[{session_id}] db-pass file exists but is empty: {actual_pass_file}")
        else:
            try:
                # 尝试不同的分隔符
                try:
                    df_pass = pd.read_csv(actual_pass_file, sep='\t')
                except:
                    try:
                        df_pass = pd.read_csv(actual_pass_file, sep=',')
                    except:
                        df_pass = pd.read_csv(actual_pass_file, sep=None, engine='python')
                
                logger.debug(f"[{session_id}] db-pass file has {len(df_pass)} rows, {len(df_pass.columns)} columns")
                pass_results = df_pass.to_dict('records')
                results.extend(pass_results)
                pass_count = len(pass_results)
                logger.info(f"[{session_id}] Loaded {pass_count} sequences that passed quality filters")
            except Exception as e:
                logger.error(f"[{session_id}] Failed to read db-pass file: {str(e)}", exc_info=True)
    else:
        logger.warning(f"[{session_id}] ChangeO did not produce db-pass file: {airr_pass_file}")
        # Check if there are any other output files
        try:
            all_changeo_files = list(TEMP_FILES_DIR.glob(f"{changeo_prefix}*"))
            if all_changeo_files:
                logger.info(f"[{session_id}] Found other ChangeO output files: {[str(f.name) for f in all_changeo_files]}")
        except:
            pass
    
    # Read failed sequences (optional, but we include them to preserve all input sequences)
    if actual_fail_file:
        file_size = actual_fail_file.stat().st_size
        logger.debug(f"[{session_id}] Reading failed sequences: {actual_fail_file} (size: {file_size} bytes)")
        if file_size == 0:
            logger.debug(f"[{session_id}] db-fail file exists but is empty")
        else:
            try:
                # 尝试不同的分隔符
                try:
                    df_fail = pd.read_csv(actual_fail_file, sep='\t')
                except:
                    try:
                        df_fail = pd.read_csv(actual_fail_file, sep=',')
                    except:
                        df_fail = pd.read_csv(actual_fail_file, sep=None, engine='python')
                
                logger.debug(f"[{session_id}] db-fail file has {len(df_fail)} rows, {len(df_fail.columns)} columns")
                fail_results = df_fail.to_dict('records')
                results.extend(fail_results)
                fail_count = len(fail_results)
                logger.info(f"[{session_id}] Loaded {fail_count} sequences that failed quality filters")
            except Exception as e:
                logger.warning(f"[{session_id}] Failed to read db-fail file: {str(e)}", exc_info=True)
    else:
        logger.debug(f"[{session_id}] No db-fail file found (this is normal if all sequences passed)")
    
    if not results:
        logger.warning(f"[{session_id}] No results found in either db-pass or db-fail files")
        # Additional diagnostics
        logger.warning(f"[{session_id}] IgBLAST output file: {igblast_out} (exists: {igblast_out.exists()}, size: {igblast_out.stat().st_size if igblast_out.exists() else 0} bytes)")
        logger.warning(f"[{session_id}] Input FASTA file: {fasta_file} (exists: {fasta_file.exists()}, size: {fasta_file.stat().st_size if fasta_file.exists() else 0} bytes)")
        
        # 检查IgBLAST输出文件内容，统计"0 hits found"的数量
        if igblast_out.exists() and igblast_out.stat().st_size > 0:
            try:
                with open(igblast_out, 'r') as f:
                    content = f.read()
                    zero_hits_count = content.count("# 0 hits found")
                    total_queries = content.count("# Query:")
                    logger.warning(f"[{session_id}] IgBLAST analysis: {zero_hits_count}/{total_queries} sequences had 0 hits found")
                    if zero_hits_count == total_queries and total_queries > 0:
                        logger.error(f"[{session_id}] ALL sequences had 0 hits - this indicates:")
                        logger.error(f"[{session_id}]   1. Sequences may not be valid Ig/TCR sequences")
                        logger.error(f"[{session_id}]   2. Sequences may be amino acid instead of nucleotide")
                        logger.error(f"[{session_id}]   3. Database path may be incorrect")
                        logger.error(f"[{session_id}]   4. Sequences may be too short or low quality")
            except Exception as e:
                logger.warning(f"[{session_id}] Could not analyze IgBLAST output: {str(e)}")
        
        # 尝试检查ChangeO是否真的执行了
        logger.error(f"[{session_id}] ChangeO MakeDb appears to have run successfully but produced no output files.")
        logger.error(f"[{session_id}] This may indicate a problem with ChangeO configuration or input data.")
        logger.error(f"[{session_id}] Please check ChangeO stdout/stderr logs above for details.")
        
        # 即使没有结果，也返回一个空列表，但记录警告
        return []
    
    batch_total_time = time.time() - batch_start_time
    logger.info(f"[{session_id}] Batch processing completed: {len(results)} total results ({pass_count} passed, {fail_count} failed, elapsed: {batch_total_time:.2f}s)")
    
    # Cleanup
    logger.debug(f"[{session_id}] Cleaning up temporary files")
    cleanup_files = [igblast_out, airr_pass_file, airr_fail_file]
    if cleaned_fasta is not None and cleaned_fasta.exists():
        cleanup_files.append(cleaned_fasta)
    for temp_file in cleanup_files:
        try:
            Path(temp_file).unlink(missing_ok=True)
        except:
            pass
    
    return results


@mcp.tool()
async def analyze_vdj_batch(args: AnalyzeVdjBatchArgs):
    """
    V(D)J recombination analysis using IgBLAST + ChangeO.
    Returns AIRR format results.
    
    Automatically splits large files into batches for efficient processing.

    Args:
        args: AnalyzeVdjBatchArgs - Parameters for V(D)J analysis

    Returns:
        AIRR format results from ChangeO
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    # 发送初始化进度
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting V(D)J analysis",
            "timestamp": time.time()
        }
    }
    
    logger.info(f"="*60)
    logger.info(f"[{session_id}] Starting V(D)J analysis")
    logger.info(f"[{session_id}] Parameters: organism={args.organism}, receptor_type={args.receptor_type}, locus={args.locus}, timeout={args.timeout}s")

    batch_files = []  # Track batch files for cleanup
    
    try:
        sequences = args.sequences
        organism = args.organism
        receptor_type = args.receptor_type
        locus = args.locus
        timeout = args.timeout

        # Handle different input types
        if isinstance(sequences, str):
            # If sequences is a string, treat it as file path
            input_file = Path(sequences)
            
            # Check if it's a URL
            if sequences.startswith(('http://', 'https://')):
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "unsupported_input",
                    "message": f"URL input is not supported. Please use the File Utils MCP service (download_url tool) to download the file first, then provide the local file path.",
                    "session_id": session_id
                }
                return
            
            # Check if it's CSV/Excel
            if input_file.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "unsupported_input",
                    "message": f"CSV/Excel file input is not supported. Please use the File Utils MCP service (convert_to_fasta tool) to convert the file to FASTA format first, then provide the FASTA file path.",
                    "session_id": session_id
                }
                return
            
            # Check if file exists
            if not input_file.exists():
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "file_not_found",
                    "message": f"File not found: {sequences}",
                    "session_id": session_id
                }
                return
            
            # It's a FASTA file
            fasta_file = input_file
        else:
            # If sequences is a list, create temporary FASTA file
            fasta_file = TEMP_FILES_DIR / f"igblast_input_{session_id}.fasta"
            logger.info(f"[{session_id}] Creating temporary FASTA file from list: {len(sequences)} sequences")
            with open(fasta_file, 'w') as f:
                for seq in sequences:
                    f.write(f">{seq['id']}\n{seq['sequence']}\n")
            logger.debug(f"[{session_id}] Temporary file created: {fasta_file}")

        # Count sequences and determine if batching is needed
        file_size = fasta_file.stat().st_size / (1024 * 1024)  # Size in MB
        num_seqs = 0
        if fasta_file.exists():
            with open(fasta_file, 'r') as f:
                num_seqs = sum(1 for line in f if line.startswith('>'))
        
        logger.info(f"[{session_id}] File info: {num_seqs} sequences, {file_size:.2f} MB")
        
        # Decide whether to use batch processing (automatically enabled for large files)
        use_batching = num_seqs > _AUTO_BATCH_THRESHOLD
        logger.info(f"[{session_id}] Processing mode: {'batched' if use_batching else 'single'} (threshold: {_AUTO_BATCH_THRESHOLD})")
        
        if use_batching:
            # === BATCH PROCESSING MODE ===
            logger.info(f"[{session_id}] " + "="*50)
            logger.info(f"[{session_id}] Large file detected: {num_seqs} sequences ({file_size:.2f} MB)")
            logger.info(f"[{session_id}] Enabling automatic batch processing (batch_size={_BATCH_SIZE})")
            
            # Split file into batches
            logger.info(f"[{session_id}] Starting file splitting...")
            split_start = time.time()
            batch_files = split_fasta_file(fasta_file, _BATCH_SIZE, TEMP_FILES_DIR, session_id)
            num_batches = len(batch_files)
            split_time = time.time() - split_start
            logger.info(f"[{session_id}] File splitting completed: {num_batches} batches (elapsed: {split_time:.2f}s)")
            
            # Process each batch
            all_results = []
            logger.info(f"[{session_id}] Starting to process {num_batches} batches")
            for batch_idx, batch_file in enumerate(batch_files, 1):
                batch_session_id = f"{session_id}_b{batch_idx}"
                logger.info(f"[{session_id}] >>> Processing batch {batch_idx}/{num_batches}...")
                
                # 发送进度更新
                progress_percent = (batch_idx / num_batches) * 100
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / batch_idx if batch_idx > 0 else 0
                estimated_remaining = avg_time_per_batch * (num_batches - batch_idx)
                
                yield {
                    "type": "progress",
                    "data": {
                        "session_id": session_id,
                        "status": "processing",
                        "batch_current": batch_idx,
                        "batch_total": num_batches,
                        "progress_percent": round(progress_percent, 1),
                        "elapsed_seconds": round(elapsed_time, 1),
                        "elapsed_minutes": round(elapsed_time / 60, 1),
                        "eta_seconds": round(estimated_remaining, 1),
                        "eta_minutes": round(estimated_remaining / 60, 1),
                        "results_so_far": len(all_results),
                        "message": f"Processing batch {batch_idx}/{num_batches} ({progress_percent:.1f}%)",
                        "timestamp": time.time()
                    }
                }
                
                try:
                    batch_results = _process_single_batch(
                        batch_file,
                        organism,
                        receptor_type,
                        locus,
                        timeout,
                        batch_session_id
                    )
                    all_results.extend(batch_results)
                    logger.info(f"[{session_id}] <<< Batch {batch_idx}/{num_batches} completed: {len(batch_results)} results (total: {len(all_results)})")
                    
                except subprocess.TimeoutExpired:
                    logger.error(f"[{session_id}] Batch {batch_idx}/{num_batches} timed out (>{timeout}s)")
                    # Save partial results if any
                    output_file = save_results_to_csv(all_results, session_id, OUTPUT_DIR) if all_results else None
                    yield {
                        "type": "error",
                        "status": "error",
                        "error_type": "timeout",
                        "message": f"Batch {batch_idx}/{num_batches} timed out after {timeout}s. "
                                  f"Try increasing timeout parameter (current: {timeout}s, max: 14400s).",
                        "output_file": output_file,
                        "batch_failed": batch_idx,
                        "total_batches": num_batches,
                        "processed_so_far": len(all_results),
                        "session_id": session_id
                    }
                    return
                except subprocess.CalledProcessError as e:
                    logger.error(f"[{session_id}] Batch {batch_idx}/{num_batches} processing failed: {str(e)}")
                    # Save partial results if any
                    output_file = save_results_to_csv(all_results, session_id, OUTPUT_DIR) if all_results else None
                    yield {
                        "type": "error",
                        "status": "error",
                        "error_type": "subprocess_failed",
                        "message": f"Batch {batch_idx}/{num_batches} failed: {str(e)}",
                        "stderr": e.stderr.decode() if e.stderr else "",
                        "output_file": output_file,
                        "batch_failed": batch_idx,
                        "total_batches": num_batches,
                        "processed_so_far": len(all_results),
                        "session_id": session_id
                    }
                    return
            
            # Clean up batch files
            logger.debug(f"[{session_id}] Cleaning up {len(batch_files)} batch files")
            for batch_file in batch_files:
                try:
                    batch_file.unlink(missing_ok=True)
                except:
                    pass
            
            # Clean up original file if created from list
            if isinstance(sequences, list) and fasta_file.exists():
                try:
                    fasta_file.unlink()
                    logger.debug(f"[{session_id}] Cleaning up temporary input file")
                except:
                    pass
            
            total_time = time.time() - start_time
            logger.info(f"[{session_id}] Batch processing completed: {len(all_results)}/{num_seqs} results, total elapsed: {total_time:.2f}s")
            
            # Save results to CSV file
            output_file = save_results_to_csv(all_results, session_id, OUTPUT_DIR)
            
            logger.info(f"="*60)
            
            # 发送完成进度
            total_time = time.time() - start_time
            yield {
                "type": "progress",
                "data": {
                    "session_id": session_id,
                    "status": "completed",
                    "progress_percent": 100.0,
                    "elapsed_seconds": round(total_time, 1),
                    "elapsed_minutes": round(total_time / 60, 1),
                    "message": f"Completed: {len(all_results)}/{num_seqs} sequences processed",
                    "timestamp": time.time()
                }
            }
            
            # 返回最终结果
            yield {
                "type": "result",
                "status": "success",
                "session_id": session_id,
                "output_file": output_file,
                "total_sequences": num_seqs,
                "processed": len(all_results),
                "format": "AIRR",
                "processing_mode": "batched",
                "num_batches": num_batches,
                "batch_size": _BATCH_SIZE,
                "processing_time_ms": total_time * 1000
            }
        
        else:
            # === SINGLE FILE PROCESSING MODE ===
            logger.info(f"[{session_id}] Single batch processing mode: {num_seqs} sequences ({file_size:.2f} MB)")
            
            # Estimate processing time
            estimated_time = num_seqs * 3
            if estimated_time > timeout * 0.8:
                logger.warning(f"[{session_id}] Estimated processing time ({estimated_time}s) approaches timeout limit ({timeout}s)")
                logger.warning(f"[{session_id}] Suggestion: increase timeout parameter or use smaller input file")
            
            try:
                results = _process_single_batch(
                    fasta_file,
                    organism,
                    receptor_type,
                    locus,
                    timeout,
                    session_id
                )
                
                # Clean up if created from list
                if isinstance(sequences, list) and fasta_file.exists():
                    try:
                        fasta_file.unlink()
                        logger.debug(f"[{session_id}] Cleaning up temporary input file")
                    except:
                        pass
                
                total_time = time.time() - start_time
                logger.info(f"[{session_id}] Single batch processing completed: {len(results)}/{num_seqs} results, total elapsed: {total_time:.2f}s")
                
                # Save results to CSV file
                output_file = save_results_to_csv(results, session_id, OUTPUT_DIR)
                
                logger.info(f"="*60)
                
                # 发送完成进度
                total_time = time.time() - start_time
                yield {
                    "type": "progress",
                    "data": {
                        "session_id": session_id,
                        "status": "completed",
                        "progress_percent": 100.0,
                        "elapsed_seconds": round(total_time, 1),
                        "elapsed_minutes": round(total_time / 60, 1),
                        "message": f"Completed: {len(results)}/{len(sequences) if isinstance(sequences, list) else num_seqs} sequences processed",
                        "timestamp": time.time()
                    }
                }
                
                # 返回最终结果
                yield {
                    "type": "result",
                    "status": "success",
                    "session_id": session_id,
                    "output_file": output_file,
                    "total_sequences": len(sequences) if isinstance(sequences, list) else num_seqs,
                    "processed": len(results),
                    "format": "AIRR",
                    "processing_mode": "single",
                    "processing_time_ms": total_time * 1000
                }
                
            except subprocess.TimeoutExpired as e:
                logger.error(f"[{session_id}] Processing timed out: {timeout}s")
                # Clean up if created from list
                if isinstance(sequences, list) and fasta_file.exists():
                    try:
                        fasta_file.unlink()
                    except:
                        pass
                
                recommended_timeout = max(num_seqs * 5, 1800)
                logger.info(f"[{session_id}] Recommended timeout: {recommended_timeout}s")
                
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "timeout",
                    "message": f"Processing timed out after {timeout}s ({timeout/60:.1f} min). "
                              f"File: {num_seqs} sequences ({file_size:.2f} MB). "
                              f"Recommended: Increase timeout to {recommended_timeout}s or split file into smaller batches.",
                    "output_file": None,
                    "sequences_count": num_seqs,
                    "file_size_mb": round(file_size, 2),
                    "timeout_used": timeout,
                    "recommended_timeout": recommended_timeout,
                    "auto_batch_enabled": False,
                    "session_id": session_id
                }
                return
            except subprocess.CalledProcessError as e:
                logger.error(f"[{session_id}] Subprocess execution failed: {str(e)}")
                # Clean up if created from list
                if isinstance(sequences, list) and fasta_file.exists():
                    try:
                        fasta_file.unlink()
                    except:
                        pass
                
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "subprocess_failed",
                    "message": str(e),
                    "stderr": e.stderr.decode() if e.stderr else "",
                    "output_file": None,
                    "session_id": session_id
                }
                return

    except subprocess.CalledProcessError as e:
        logger.error(f"[{session_id}] Outer subprocess error caught: {str(e)}")
        yield {
            "type": "error",
            "status": "error",
            "error_type": "subprocess_failed",
            "message": str(e),
            "stderr": e.stderr.decode() if e.stderr else "",
            "output_file": None,
            "session_id": session_id
        }
        return
    except Exception as e:
        logger.error(f"[{session_id}] Unknown error: {str(e)}", exc_info=True)
        yield {
            "type": "error",
            "status": "error",
            "error_type": "unknown",
            "message": str(e),
            "output_file": None,
            "session_id": session_id
        }
        return
    finally:
        # 清理批次文件
        if batch_files:
            logger.debug(f"[{session_id}] Final cleanup: {len(batch_files)} batch files")
            for batch_file in batch_files:
                try:
                    if batch_file.exists():
                        batch_file.unlink()
                except:
                    pass


class ExtractCdr3FromAirrArgs(BaseModel):
    """Parameters for CDR3 extraction from AIRR results"""
    airr_results: Union[List[Dict[str, Any]], str] = Field(
        ...,
        description="AIRR format results from V(D)J analysis. Can be: (1) Array of AIRR records, or (2) Local file path (CSV/TSV/JSON). For URLs, please use the File Utils service to download first.",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["csv", "tsv", "json"],
            "placeholder": "Paste AIRR results array, or provide file path/URL",
            "help_text": "AIRR format results containing V(D)J analysis data. You can provide: array of records, or local file path. For URLs, please use the File Utils MCP service (download_url tool) first."
        }
    )
    
    @field_validator('airr_results', mode='before')
    @classmethod
    def normalize_airr_results(cls, v) -> Union[List[Dict[str, Any]], str]:
        """规范化 airr_results 输入，处理各种格式"""
        # 如果是列表，直接返回
        if isinstance(v, list):
            return v
        
        # 如果是字符串，直接返回（文件路径或 URL）
        if isinstance(v, str):
            return v
        
        # 其他类型
        raise ValueError(f"airr_results must be a list of dicts or file path string. Got: {type(v).__name__}")


def _load_airr_results(airr_input: Union[List[Dict[str, Any]], str]) -> Tuple[List[Dict[str, Any]], None]:
    """
    从各种来源加载 AIRR 格式结果
    
    Args:
        airr_input: AIRR 结果数组或本地文件路径
        
    Returns:
        (AIRR 结果数组, None) - 不再返回临时文件路径，因为不再支持 URL 下载
        
    Raises:
        ValueError: 如果无法加载数据或输入是 URL
        FileNotFoundError: 如果文件不存在
    """
    # 如果已经是数组，直接返回
    if isinstance(airr_input, list):
        logger.debug(f"Input is already an array with {len(airr_input)} records")
        return airr_input, None
    
    # 如果是字符串，可能是文件路径或 URL
    if isinstance(airr_input, str):
        input_path_or_url = airr_input.strip()
        
        # 检查是否是 URL
        if input_path_or_url.startswith(('http://', 'https://')):
            error_msg = f"URL input is not supported. Please use the File Utils MCP service (download_url tool) to download the file first, then provide the local file path."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        input_path = Path(input_path_or_url)
        if not input_path.exists():
            error_msg = f"AIRR results file not found: {input_path_or_url}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 根据文件扩展名确定读取方式
        file_ext = input_path.suffix.lower()
        logger.info(f"Loading AIRR results from file: {input_path} (format: {file_ext})")
        
        try:
            if file_ext == '.json':
                # JSON 格式
                import json
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 如果数据是字典且包含数组，尝试提取数组
                if isinstance(data, dict):
                    # 尝试常见的键名
                    for key in ['airr_results', 'results', 'data', 'records']:
                        if key in data and isinstance(data[key], list):
                            data = data[key]
                            break
                if not isinstance(data, list):
                    raise ValueError(f"JSON file does not contain an array of records. Root type: {type(data).__name__}")
                logger.info(f"Loaded {len(data)} AIRR records from JSON file")
                return data, None
            elif file_ext in ['.csv', '.tsv']:
                # CSV/TSV 格式
                sep = '\t' if file_ext == '.tsv' else ','
                df = pd.read_csv(input_path, sep=sep, encoding='utf-8')
                # 转换为字典列表
                records = df.to_dict('records')
                logger.info(f"Loaded {len(records)} AIRR records from {file_ext.upper()} file")
                return records, None
            else:
                # 尝试作为 CSV 读取
                logger.warning(f"Unknown file extension {file_ext}, attempting to read as CSV")
                df = pd.read_csv(input_path, sep=None, engine='python', encoding='utf-8')
                records = df.to_dict('records')
                logger.info(f"Loaded {len(records)} AIRR records (auto-detected as CSV)")
                return records, None
        except Exception as e:
            error_msg = f"Failed to load AIRR results from file {input_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    raise ValueError(f"Invalid airr_results input type: {type(airr_input).__name__}")


@mcp.tool()
async def extract_cdr3_from_airr(args: ExtractCdr3FromAirrArgs):
    """
    Extract CDR3 information from AIRR format results.
    
    Supports multiple input formats:
    - Array of AIRR records: [{"sequence_id": "...", "junction": "...", ...}, ...]
    - Local file path: "/path/to/airr_results.csv" or "/path/to/airr_results.json"
    - HTTP/HTTPS URL: "https://example.com/airr_results.csv"
    
    通过 SSE 流式推送提取进度，支持大文件处理。

    Args:
        args: ExtractCdr3FromAirrArgs - Parameters for CDR3 extraction

    Yields:
        Progress updates and final result through SSE stream.
    """
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Starting CDR3 extraction from AIRR results")
    
    # 发送初始化进度
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting CDR3 extraction from AIRR results",
            "timestamp": time.time()
        }
    }
    
    try:
        # 加载 AIRR 结果（可能是数组或文件路径）
        yield {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "status": "loading",
                "message": "Loading AIRR results...",
                "timestamp": time.time()
            }
        }
        
        airr_results, _ = _load_airr_results(args.airr_results)
        
        if not airr_results:
            logger.warning(f"[{session_id}] No AIRR results found")
            yield {
                "type": "error",
                "data": {
                    "session_id": session_id,
                    "status": "error",
                    "message": "No AIRR results found in the input"
                }
            }
            return
        
        logger.info(f"[{session_id}] Processing {len(airr_results)} AIRR records")
        yield {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "status": "processing",
                "message": f"Processing {len(airr_results)} AIRR records...",
                "timestamp": time.time()
            }
        }
        
        cdr3_data = []

        for record in airr_results:
            if not isinstance(record, dict):
                logger.warning(f"[{session_id}] Skipping invalid record (not a dict): {type(record).__name__}")
                continue
                
            cdr3_data.append({
                "sequence_id": record.get("sequence_id"),
                "junction": record.get("junction"),  # CDR3 nucleotide
                "junction_aa": record.get("junction_aa"),  # CDR3 amino acid
                "junction_length": record.get("junction_length"),
                "productive": record.get("productive"),
                "v_call": record.get("v_call"),
                "j_call": record.get("j_call"),
                "stop_codon": record.get("stop_codon"),
                "vj_in_frame": record.get("vj_in_frame")
            })

        logger.info(f"[{session_id}] CDR3 extraction completed: {len(cdr3_data)} CDR3 sequences extracted")
        
        # 返回最终结果
        yield {
            "type": "result",
            "data": {
                "status": "success",
                "cdr3_results": cdr3_data,
                "total": len(cdr3_data)
            }
        }

    except FileNotFoundError as e:
        error_msg = f"[{session_id}] File not found: {str(e)}"
        logger.error(error_msg)
        yield {
            "type": "error",
            "data": {
                "session_id": session_id,
                "status": "error",
                "error_type": "file_not_found",
                "message": error_msg
            }
        }
    except ValueError as e:
        error_msg = f"[{session_id}] Invalid input: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield {
            "type": "error",
            "data": {
                "session_id": session_id,
                "status": "error",
                "error_type": "invalid_input",
                "message": error_msg
            }
        }
    except Exception as e:
        error_msg = f"[{session_id}] Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield {
            "type": "error",
            "data": {
                "session_id": session_id,
                "status": "error",
                "error_type": "unknown",
                "message": error_msg
            }
        }


# 添加生命周期管理
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def igblast_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("IgBLAST MCP Server 正在初始化...")
    
    try:
        yield {"initialized": True}
    finally:
        print("IgBLAST MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = igblast_lifespan

if __name__ == "__main__":
    print("启动IgBLAST MCP服务器...")
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8088
    
    # 使用SSE模式启动
    mcp.run(transport="sse")
