"""
FDG MCP Server - Core FDG Tool Wrapper

This server exposes the core FDG (Foldx, DDG, GearBind) process via MCP protocol.
"""

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Any
from collections.abc import AsyncIterator
import os
import tempfile
import urllib.request
from urllib.parse import urlparse
import time
import uuid
import asyncio
import threading
import logging
import json
import inspect
from mcp.types import TextContent, CallToolRequest, ServerResult, CallToolResult, ErrorData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MetaBcr_MCP')

# Create MCP server
mcp = FastMCP("MetaBcr Core Server")

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
# 确保 _mcp_server 已初始化
if hasattr(mcp, '_mcp_server') and CallToolRequest in mcp._mcp_server.request_handlers:
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
            
            # 尝试获取工具定义（如果方法存在）
            tool = None
            try:
                if hasattr(mcp._mcp_server, '_get_cached_tool_definition'):
                    tool = await mcp._mcp_server._get_cached_tool_definition(tool_name)
            except Exception as e:
                logger.warning(f"[流式传输] 无法获取工具定义: {str(e)}")
            
            # input validation（如果工具定义可用）
            if tool and hasattr(tool, 'inputSchema'):
                try:
                    jsonschema.validate(instance=arguments, schema=tool.inputSchema)
                except jsonschema.ValidationError as e:
                    return ServerResult(ErrorData(code=-32602, message=f"Input validation error: {e.message}"))
            
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
                
                # 5. 返回任务 ID 和服务 ID
                service_id = "metabcr"
                
                logger.info(f"[流式传输] 返回任务信息 (工具: {tool_name}, 任务ID: {task_id}, 服务ID: {service_id})")
                
                # 立即返回任务 ID 和服务 ID
                try:
                    result = ServerResult(
                        CallToolResult(
                            content=[TextContent(
                                type="text",
                                text=json.dumps({
                                    "type": "streaming_task",
                                    "task_id": task_id,
                                    "service_id": service_id,
                                    "message": "任务已启动，请通过 service_id 和 task_id 构建 SSE 端点 URL 来接收实时进度消息。URL 格式: http://{host}:{port}/stream/{task_id}，其中 host 和 port 由客户端根据 service_id 确定。"
                                }, ensure_ascii=False)
                            )],
                            structuredContent={
                                "type": "streaming_task",
                                "task_id": task_id,
                                "service_id": service_id,
                                "message": "任务已启动，请通过 service_id 和 task_id 构建 SSE 端点 URL"
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
                    return ServerResult(ErrorData(code=-32603, message=f"Failed to create streaming task result: {str(e)}"))
            else:
                # 非异步生成器，使用原始 handler
                return await _original_handler(req)
        except Exception as e:
            logger.error(f"[流式传输] Handler 处理请求时出错: {str(e)}", exc_info=True)
            return ServerResult(ErrorData(code=-32603, message=f"Handler error: {str(e)}"))
    
    # 替换原始 handler
    mcp._mcp_server.request_handlers[CallToolRequest] = _streaming_handler
    logger.info("[流式传输] ✅ 已安装自定义 streaming handler")

def download_url_to_temp_file(url: str) -> str:
    """
    下载 HTTP/HTTPS URL 到临时文件
    
    Args:
        url: HTTP/HTTPS URL
        
    Returns:
        临时文件路径
        
    Raises:
        Exception: 如果下载失败
    """
    try:
        # 从 URL 获取文件扩展名
        parsed_url = urlparse(url)
        url_path = parsed_url.path
        # 获取文件扩展名，如果没有则使用 .csv 作为默认扩展名
        ext = os.path.splitext(url_path)[1] or '.csv'
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # 下载文件
        urllib.request.urlretrieve(url, temp_file_path)
        
        return temp_file_path
    except Exception as e:
        raise Exception(f"Failed to download URL {url}: {str(e)}")

class MetabcrArgs(BaseModel):
    """Parameters for MetaBCR prediction"""
    
    antibody_file: str = Field(
        ...,
        description="Antibody CSV file path or HTTP/HTTPS URL. This file contains antibody sequences.",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["csv"],
            "placeholder": "Enter antibody CSV file path or URL",
            "help_text": "Antibody CSV file containing antibody sequences (Heavy and Light chains). Each row will be combined with each antigen sequence.",
            "demo_urls": "/data_new/workspace/AgeB_BCR_standardized.csv"
        }
    )
    
    antigen_file: str = Field(
        ...,
        description="Antigen CSV file path or HTTP/HTTPS URL. This file contains antigen sequences that will be combined with antibodies.",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["csv"],
            "placeholder": "Enter antigen CSV file path or URL",
            "help_text": "Antigen CSV file containing antigen sequences. Each antigen sequence will be combined with each antibody row, creating variant_seq_1, variant_seq_2, ... columns."
        }
    )
    
    antigen_name: str = Field(
        default="flu",
        description="Antigen name for prediction (model selection)",
        json_schema_extra={
            "ui_type": "select",
            "options": ["flu", "sars", "rsv", "hiv"],
            "placeholder": "Select antigen name",
            "help_text": "Select the antigen type for binding prediction. Options: flu, sars, rsv, hiv"
        }
    )
    
    output_file_path: Optional[str] = Field(
        default=None,
        description="Optional output directory path. If provided, results will be saved to this directory; otherwise, the default output path will be used",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": False,
            "placeholder": "Enter output directory path",
            "help_text": "Directory where prediction results will be saved"
        }
    )

@mcp.tool()
async def metabcr(args: MetabcrArgs):
    """MetaBCR: A Deep Learning Framework for Antibody-Antigen Interaction Prediction
    MetaBCR is designed to predict the binding affinity between antibodies and antigens using deep learning models.
    It supports multiple model architectures, including CNN, GNN, and BERT-based models, and can be configured
    for various tasks and datasets through command-line arguments and configuration files.
    
    Args:
        args: MetabcrArgs - Parameters for MetaBCR prediction
    
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    # 发送初始化进度
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting MetaBCR prediction",
            "timestamp": time.time()
        }
    }
    
    # 处理 URL 下载
    temp_file_paths = []
    antibody_file = args.antibody_file
    antigen_file = args.antigen_file
    output_file_path = args.output_file_path
    import os
    import sys
    import torch
    import numpy as np
    import glob
    import pandas as pd
    
    try:
        # 处理抗体文件：可能是 URL 或本地路径
        if antibody_file.startswith(('http://', 'https://')):
            try:
                temp_ab_file = download_url_to_temp_file(antibody_file)
                temp_file_paths.append(temp_ab_file)
                actual_antibody_file = temp_ab_file
            except Exception as e:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "download_failed",
                    "message": f"Failed to download antibody file from URL {antibody_file}: {str(e)}",
                    "session_id": session_id
                }
                return
        else:
            if not os.path.exists(antibody_file):
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "file_not_found",
                    "message": f"Antibody file not found: {antibody_file}",
                    "session_id": session_id
                }
                return
            actual_antibody_file = antibody_file
        
        # 处理抗原文件：可能是 URL 或本地路径
        if antigen_file.startswith(('http://', 'https://')):
            try:
                temp_ag_file = download_url_to_temp_file(antigen_file)
                temp_file_paths.append(temp_ag_file)
                actual_antigen_file = temp_ag_file
            except Exception as e:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "download_failed",
                    "message": f"Failed to download antigen file from URL {antigen_file}: {str(e)}",
                    "session_id": session_id
                }
                return
        else:
            if not os.path.exists(antigen_file):
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "file_not_found",
                    "message": f"Antigen file not found: {antigen_file}",
                    "session_id": session_id
                }
                return
            actual_antigen_file = antigen_file
        
        # 读取抗体文件和抗原文件
        yield {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "status": "reading_files",
                "message": "Reading antibody and antigen files",
                "timestamp": time.time()
            }
        }
        
        try:
            antibody_df = pd.read_csv(actual_antibody_file)
        except:
            try:
                antibody_df = pd.read_excel(actual_antibody_file)
            except Exception as e:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "file_read_failed",
                    "message": f"Failed to read antibody file: {str(e)}",
                    "session_id": session_id
                }
                return
        
        try:
            antigen_df = pd.read_csv(actual_antigen_file)
        except:
            try:
                antigen_df = pd.read_excel(actual_antigen_file)
            except Exception as e:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "file_read_failed",
                    "message": f"Failed to read antigen file: {str(e)}",
                    "session_id": session_id
                }
                return
        
        if antibody_df.empty:
            yield {
                "type": "error",
                "status": "error",
                "error_type": "empty_file",
                "message": "Antibody file is empty",
                "session_id": session_id
            }
            return
        
        if antigen_df.empty:
            yield {
                "type": "error",
                "status": "error",
                "error_type": "empty_file",
                "message": "Antigen file is empty",
                "session_id": session_id
            }
            return
        
        # 在抗原文件中找到 variant_seq 相关列
        variant_seq_col = None
        for col in antigen_df.columns:
            if col in ['variant_seq', 'antigen_seq', 'variant']:
                variant_seq_col = col
                break
        
        if variant_seq_col is None:
            # 尝试查找包含 variant_seq 的列
            for col in antigen_df.columns:
                if 'variant' in col.lower() or 'antigen' in col.lower() or 'seq' in col.lower():
                    variant_seq_col = col
                    logger.info(f"[{session_id}] Using column '{col}' as variant_seq column")
                    break
        
        if variant_seq_col is None:
            yield {
                "type": "error",
                "status": "error",
                "error_type": "no_variant_seq_column",
                "message": f"No variant_seq column found in antigen file. Expected columns: variant_seq, antigen_seq, variant, or columns containing 'variant', 'antigen', or 'seq'",
                "session_id": session_id
            }
            return
        
        logger.info(f"[{session_id}] Found variant_seq column in antigen file: {variant_seq_col}")
        logger.info(f"[{session_id}] Antibody file: {len(antibody_df)} rows, Antigen file: {len(antigen_df)} rows")
        
        # 将抗原文件的每一行数据填充到抗体文件中，创建 variant_seq_1, variant_seq_2, ... 列
        # 这是笛卡尔积操作：每一行抗体 × 每一行抗原
        yield {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "status": "merging",
                "message": f"Merging antibody and antigen data (creating variant_seq_1 to variant_seq_{len(antigen_df)})",
                "timestamp": time.time()
            }
        }
        
        # 为每个抗原序列创建一列
        merged_df = antibody_df.copy()
        for idx, antigen_row in antigen_df.iterrows():
            col_name = f"variant_seq_{idx + 1}"
            # 将当前抗原序列的值填充到抗体文件的每一行
            merged_df[col_name] = antigen_row[variant_seq_col]
        
        logger.info(f"[{session_id}] Created {len(antigen_df)} variant_seq columns in merged dataframe")
        
        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        
        # 设置基础路径
        METABCR_ROOT = "/data/lht/meta_bcr"
        
        # 添加到Python路径以便导入模块
        if METABCR_ROOT not in sys.path:
            sys.path.append(METABCR_ROOT)
        
        # 设置与原始脚本完全相同的参数
        antigen_name = args.antigen_name
        task_name = "bind"
        config_date = "250312"
        
        # 验证 antigen_name 是否有效
        valid_antigens = ['flu', 'sars', 'rsv', 'hiv']
        if antigen_name not in valid_antigens:
            yield {
                "type": "error",
                "status": "error",
                "error_type": "invalid_argument",
                "message": f"Invalid antigen_name '{antigen_name}'. Must be one of: {', '.join(valid_antigens)}",
                "session_id": session_id
            }
            return
        
        # 处理输出路径
        
        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        
        # 设置基础路径
        METABCR_ROOT = "/data/lht/meta_bcr"
        
        # 添加到Python路径以便导入模块
        if METABCR_ROOT not in sys.path:
            sys.path.append(METABCR_ROOT)
        
        # 设置与原始脚本完全相同的参数
        antigen_name = args.antigen_name
        task_name = "bind"
        config_date = "250312"
        
        # 验证 antigen_name 是否有效
        valid_antigens = ['flu', 'sars', 'rsv', 'hiv']
        if antigen_name not in valid_antigens:
            yield {
                "type": "error",
                "status": "error",
                "error_type": "invalid_argument",
                "message": f"Invalid antigen_name '{antigen_name}'. Must be one of: {', '.join(valid_antigens)}",
                "session_id": session_id
            }
            return
        
        # 处理输出路径
        if not output_file_path:
            # 为每次调用创建唯一的输出目录（使用时间戳），避免结果累积
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            random_id = str(uuid.uuid4())[:8]
            unique_dir_name = f"metabcr_{timestamp}_{random_id}"
            output_file_path = os.path.join(METABCR_ROOT, "output", unique_dir_name)
            print(f"使用自动生成的唯一输出目录: {output_file_path}")
        else:
            # 标准化路径分隔符（将Windows的\转换为Linux的/）
            output_file_path = output_file_path.replace('\\', '/')
            # 如果路径以文件扩展名结尾，取其父目录
            if output_file_path.endswith(('.csv', '.xlsx', '.xls')):
                output_file_path = os.path.dirname(output_file_path)
        
        # 检查并创建输出目录（仅在目录不存在时创建）
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path, exist_ok=True)
            print(f"创建输出目录: {output_file_path}")
        
        output_base_dir = os.path.join(output_file_path, "MetaBcr")
        # 检查并创建MetaBcr目录（仅在目录不存在时创建）
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir, exist_ok=True)
            print(f"创建MetaBcr目录: {output_base_dir}")
        
        output_dir = os.path.join(output_base_dir, task_name)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存合并后的文件（包含所有 variant_seq_n 列）
        merged_file_path = os.path.join(output_dir, "merged_antibody_antigen.csv")
        merged_df.to_csv(merged_file_path, index=False)
        logger.info(f"[{session_id}] Saved merged file with {len(antigen_df)} variant_seq columns to {merged_file_path}")
        
        # 导入必要的模块
        from Config.config import get_config
        
        # 构建配置文件的完整路径
        config_path = os.path.join(METABCR_ROOT, 
                                  f"Config/config_five_fold_{antigen_name}_{task_name}_meta_{config_date}_semi_supervise.json")
        
        # 加载配置
        configure = get_config(config_path)
        
        # 导入predict_metabcr模块
        import predict_metabcr
        
        # 保存原始glob函数
        original_glob = glob.glob
        
        # 创建一个自定义glob函数，确保从METABCR_ROOT开始搜索
        def custom_glob(pattern):
            if not os.path.isabs(pattern) and pattern.startswith('Results/'):
                absolute_pattern = os.path.join(METABCR_ROOT, pattern)
                return original_glob(absolute_pattern)
            return original_glob(pattern)
        
        # 替换glob.glob函数为自定义版本
        glob.glob = custom_glob
        
        # 设置测试参数
        fold_set = [0]
        label_str = None
        
        # 检测所有 variant_seq 相关的列（在合并后的文件中）
        variant_seq_columns = []
        for col in merged_df.columns:
            # 匹配 variant_seq_1, variant_seq_2, ... 等（我们刚创建的列）
            if col.startswith('variant_seq_') and col[13:].isdigit():
                variant_seq_columns.append(col)
        
        if not variant_seq_columns:
            yield {
                "type": "error",
                "status": "error",
                "error_type": "no_variant_seq_column",
                "message": f"No variant_seq columns found in merged file. Expected columns: variant_seq_1, variant_seq_2, ...",
                "session_id": session_id
            }
            return
        
        logger.info(f"[{session_id}] Found {len(variant_seq_columns)} variant_seq columns: {variant_seq_columns}")
        
        # 为每个 variant_seq 列运行预测
        total_variant_tasks = len(variant_seq_columns) * len(fold_set)
        completed_variant_tasks = 0
        
        # 创建结果DataFrame，从合并后的数据开始
        result_df = merged_df.copy()
        
        for variant_seq_col in variant_seq_columns:
                # 确定后缀（如果是 variant_seq，后缀为空；如果是 variant_seq_n，后缀为 _n）
                if variant_seq_col == 'variant_seq':
                    suffix = ''
                elif variant_seq_col.startswith('variant_seq_'):
                    # 提取数字部分，例如 variant_seq_1 -> _1
                    suffix = '_' + variant_seq_col[13:]
                elif variant_seq_col == 'antigen_seq':
                    suffix = '_antigen'
                elif variant_seq_col == 'variant':
                    suffix = '_variant'
                else:
                    # 对于其他情况，使用列名作为后缀（去除 variant_seq 前缀）
                    suffix = '_' + variant_seq_col.replace('variant_seq', '').replace('antigen_seq', 'antigen').replace('variant', 'var')
                
                for fold in fold_set:
                    completed_variant_tasks += 1
                    progress_percent = (completed_variant_tasks / total_variant_tasks) * 100
                    elapsed_time = time.time() - start_time
                    
                    yield {
                        "type": "progress",
                        "data": {
                            "session_id": session_id,
                            "status": "processing",
                            "progress_percent": round(progress_percent, 1),
                            "current_task": completed_variant_tasks,
                            "total_tasks": total_variant_tasks,
                            "elapsed_seconds": round(elapsed_time, 1),
                            "elapsed_minutes": round(elapsed_time / 60, 1),
                            "message": f"Processing {variant_seq_col} ({completed_variant_tasks}/{total_variant_tasks}, fold {fold})",
                            "timestamp": time.time()
                        }
                    }
                    
                    # 创建临时文件，将当前 variant_seq_n 列重命名为 variant_seq
                    temp_input_df = merged_df.copy()
                    
                    # 重命名当前列为标准名称 variant_seq
                    temp_input_df = temp_input_df.rename(columns={variant_seq_col: 'variant_seq'})
                    
                    # 如果还有其他 variant_seq 列，先删除它们（避免混淆）
                    cols_to_drop = [c for c in temp_input_df.columns if c.startswith('variant_seq') and c != 'variant_seq']
                    if cols_to_drop:
                        temp_input_df = temp_input_df.drop(columns=cols_to_drop)
                    
                    # 保存临时文件
                    temp_input_file = os.path.join(output_dir, f"temp_input_{variant_seq_col}_fold{fold}.csv")
                    temp_input_df.to_csv(temp_input_file, index=False)
                    temp_file_paths.append(temp_input_file)
                    
                    # 为这次运行创建独立的输出目录
                    variant_output_dir = os.path.join(output_dir, f"variant_{variant_seq_col}_fold{fold}")
                    os.makedirs(variant_output_dir, exist_ok=True)
                    
                    try:
                        # 执行预测
                        predict_metabcr.test(
                            _cfg_=configure,
                            antigen_name=antigen_name,
                            fold=fold,
                            fdir_tst=temp_input_file,
                            output_dir=variant_output_dir,
                            label_str=label_str,
                            date=config_date,
                            task_name=task_name
                        )
                        
                        # 查找生成的结果文件
                        result_file = None
                        if os.path.exists(variant_output_dir):
                            for file in os.listdir(variant_output_dir):
                                if file.endswith('.xlsx') and 'test_results' in file:
                                    result_file = os.path.join(variant_output_dir, file)
                                    break
                        
                        if result_file and os.path.exists(result_file):
                            # 读取预测结果
                            try:
                                pred_df = pd.read_excel(result_file)
                                
                                # 提取预测结果列（bind_output 和 bind_predict）
                                output_col = f"{task_name}_output"
                                predict_col = f"{task_name}_predict"
                                
                                if output_col in pred_df.columns and predict_col in pred_df.columns:
                                    # 重命名列，添加后缀
                                    new_output_col = f"{task_name}_output{suffix}" if suffix else f"{task_name}_output"
                                    new_predict_col = f"{task_name}_predict{suffix}" if suffix else f"{task_name}_predict"
                                    
                                    # 确保列名唯一
                                    if new_output_col in result_df.columns:
                                        new_output_col = f"{task_name}_output{suffix}_fold{fold}"
                                    if new_predict_col in result_df.columns:
                                        new_predict_col = f"{task_name}_predict{suffix}_fold{fold}"
                                    
                                    # 将预测结果添加到结果DataFrame
                                    # 使用索引对齐（假设行数相同且顺序一致）
                                    if len(pred_df) == len(result_df):
                                        result_df[new_output_col] = pred_df[output_col].values
                                        result_df[new_predict_col] = pred_df[predict_col].values
                                        logger.info(f"[{session_id}] Added predictions for {variant_seq_col}: {new_output_col}, {new_predict_col}")
                                    else:
                                        logger.warning(f"[{session_id}] Row count mismatch: original={len(result_df)}, prediction={len(pred_df)}")
                                else:
                                    logger.warning(f"[{session_id}] Expected columns {output_col} and {predict_col} not found in {result_file}")
                                    
                            except Exception as e:
                                logger.error(f"[{session_id}] Failed to read prediction results from {result_file}: {str(e)}", exc_info=True)
                        else:
                            logger.warning(f"[{session_id}] No result file found for {variant_seq_col} fold {fold}")
                            
                    except Exception as e:
                        logger.error(f"[{session_id}] Error processing {variant_seq_col} fold {fold}: {str(e)}", exc_info=True)
                        yield {
                            "type": "progress",
                            "data": {
                                "session_id": session_id,
                                "status": "warning",
                                "message": f"Warning: Failed to process {variant_seq_col} fold {fold}: {str(e)}",
                                "timestamp": time.time()
                            }
                        }
            
        # 保存最终结果文件（包含所有预测结果）
        antibody_basename = os.path.basename(actual_antibody_file).split(".")[0]
        antigen_basename = os.path.basename(actual_antigen_file).split(".")[0]
        final_output_file = os.path.join(output_dir, f"final_results_{antibody_basename}_{antigen_basename}_{task_name}_{config_date}_fold{fold_set[0]}.csv")
        result_df.to_csv(final_output_file, index=False)
        
        logger.info(f"[{session_id}] Saved final results with all predictions to {final_output_file}")
        logger.info(f"[{session_id}] Final result file contains {len(result_df)} rows and {len(result_df.columns)} columns")
        
        # 恢复原始glob函数
        glob.glob = original_glob
        
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
                "message": f"MetaBCR prediction completed for all {len(variant_seq_columns)} variant_seq columns",
                "timestamp": time.time()
            }
        }
        
        # 返回最终结果
        yield {
            "type": "result",
            "status": "success",
            "session_id": session_id,
            "output_file": final_output_file,
            "num_variant_seq_columns": len(variant_seq_columns),
            "num_antibody_rows": len(antibody_df),
            "num_antigen_rows": len(antigen_df),
            "processing_time_ms": total_time * 1000,
            "message": f"All {len(variant_seq_columns)} variant_seq columns have been processed and results merged into the output file"
        }
    finally:
        # 清理从 URL 下载的临时文件
        for temp_file_path in temp_file_paths:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass


# 添加生命周期管理
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def fdg_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("MetaBcr MCP Server 正在初始化...")
    
    try:
        yield {"initialized": True}
    finally:
        print("MetaBcr MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = fdg_lifespan

if __name__ == "__main__":
    print("启动MetaBcr MCP服务器...")
    # 设置MCP标准路径
    # mcp.settings.sse_path = "/_mcp/v1/sse"
    # mcp.settings.message_path = "/_mcp/v1/messages/"
    # 设置网络参数
    metabcr(MetabcrArgs(input_file_path="/data/lht/meta_bcr/Data/FLU_infer/0322_ddg_datasets.csv"))    
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8082
    
    # 使用SSE模式启动
    mcp.run(transport="sse")
