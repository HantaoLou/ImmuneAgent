"""
Bioinformatics Analysis Modular MCP Server

This server provides Figure2-Figure5 related bioinformatics analysis tools, 
with each tool corresponding to a specific analysis function.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections.abc import AsyncIterator
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import StreamingResponse
import urllib.request
from urllib.parse import urlparse
from urllib.error import URLError, HTTPError
import time
import uuid
import asyncio
import threading
import logging
import json
import inspect
from mcp.types import TextContent, CallToolRequest, ServerResult, CallToolResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('Bioinformatics_MCP')

# Create MCP server
mcp = FastMCP("Bioinformatics Analysis Modular Server")

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
                
                # 5. 返回任务 ID 和服务 ID
                service_id = "bioinformatics"
                
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
        # 获取文件扩展名，如果没有则使用 .rds 作为默认扩展名
        ext = os.path.splitext(url_path)[1] or '.rds'
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # 下载文件
        urllib.request.urlretrieve(url, temp_file_path)
        
        return temp_file_path
    except Exception as e:
        raise Exception(f"Failed to download URL {url}: {str(e)}")

def run_bioinformatics_module_script(module_name: str, input_file: str, base_dir: str, figure_type: str = "figure2", **kwargs) -> str:
    """
    Generic function for executing bioinformatics analysis modular R scripts
    
    Args:
        module_name: Module name (e.g., "Figure2_A2_Binding", "Figure3_A_Density")
        input_file: Input file path containing Seurat object and single-cell RNA-seq data (支持本地路径或 HTTP/HTTPS URL)
        base_dir: Output directory base path
        figure_type: Figure type ("figure2", "figure3", "figure4", "figure5")
        **kwargs: Additional parameters
        
    Returns:
        Analysis execution result string, including generated file paths
    """

    # 如果输入是 URL，先下载到临时文件
    temp_file_path = None
    actual_input_file = input_file
    
    if input_file.startswith(('http://', 'https://')):
        try:
            temp_file_path = download_url_to_temp_file(input_file)
            actual_input_file = temp_file_path
        except Exception as e:
            return f"Error: Failed to download file from URL {input_file}: {str(e)}"

    # Check if input file exists (support both local path and URL)
    if not os.path.exists(actual_input_file):
        return f"Error: Input file does not exist or is not accessible: {actual_input_file}"
    
    working_dir = Path(__file__).parent
    base_dir = Path(base_dir)
    
    # R script path
    r_script_path = working_dir / f"scripts/common/{figure_type}_modules" / f"{module_name}.R"
    
    # Check if R script exists
    if not r_script_path.exists():
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return f"Error: R script does not exist: {r_script_path}"
    
    try:
        # Build command arguments
        cmd_args = ["Rscript", str(r_script_path), actual_input_file, str(base_dir)]
        
        # Add additional parameters
        for key, value in kwargs.items():
            if value is not None:
                cmd_args.append(str(value))
        
        # Execute R script
        result = subprocess.run(
            cmd_args,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800  # 30 minutes timeout
        )
        
        # Check execution result
        if result.returncode != 0:
            error_msg = f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}"
            # 清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            return error_msg
        
        # Collect generated files
        generated_files = []
        
        # Find generated files (support Figure2 and Figure3 etc.)
        figure_pattern = figure_type.replace("figure", "Figure").capitalize()
        for output_dir in base_dir.glob(f"{figure_pattern}*"):
            if output_dir.is_dir():
                # CSV files
                csv_files = list((output_dir / "files").glob("*.csv")) if (output_dir / "files").exists() else []
                generated_files.extend([str(f) for f in csv_files])
                
                # PDF files
                pdf_files = list((output_dir / "plots").glob("*.pdf")) if (output_dir / "plots").exists() else []
                generated_files.extend([str(f) for f in pdf_files])
                
                # Other files
                other_files = list(output_dir.glob("*.txt")) + list(output_dir.glob("*.RData"))
                generated_files.extend([str(f) for f in other_files])
        
        success_msg = f"{module_name} bioinformatics analysis executed successfully!\n"
        if generated_files:
            success_msg += f"Generated files ({len(generated_files)} files):\n"
            for file in generated_files:
                success_msg += f"  - {file}\n"
        else:
            success_msg += f"Analysis completed, please check output directory: {base_dir}\n"
        
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return success_msg
        
    except subprocess.TimeoutExpired:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return f"R script execution timeout (exceeded 1800 seconds)"
    except Exception as e:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return f"Error occurred during R script execution: {str(e)}"

def run_figure2_module_script(module_name: str, input_file: str, base_dir: str, **kwargs) -> str:
    """Function for executing Figure2 modular R scripts"""
    return run_bioinformatics_module_script(module_name, input_file, base_dir, "figure2", **kwargs)

def run_figure3_module_script(module_name: str, input_file: str, base_dir: str, **kwargs) -> str:
    """Function for executing Figure3 modular R scripts"""
    return run_bioinformatics_module_script(module_name, input_file, base_dir, "figure3", **kwargs)

def run_figure4_module_script(module_name: str, input_file: str, base_dir: str, **kwargs) -> str:
    """Function for executing Figure4 modular R scripts"""
    return run_bioinformatics_module_script(module_name, input_file, base_dir, "figure4", **kwargs)

def run_figure5_module_script(module_name: str, input_file: str, base_dir: str, **kwargs) -> str:
    """Function for executing Figure5 modular R scripts"""
    return run_bioinformatics_module_script(module_name, input_file, base_dir, "figure5", **kwargs)


# ============================================================================
# Pydantic Parameter Models
# ============================================================================

class AntigenBindingPredictionVisualizationArgs(BaseModel):
    """Parameters for antigen binding prediction visualization"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing Seurat object and binding prediction data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with Seurat object",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results and charts",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    binding_threshold: float = Field(
        default=0.5,
        description="Broad reactivity threshold (between 0-1)",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter threshold value",
            "help_text": "Threshold for binding classification",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.5
        }
    )


class BcellCelltypeDistributionAnalysisArgs(BaseModel):
    """Parameters for B cell celltype distribution analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing Seurat object and cell type annotations",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with cell annotations",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results and charts",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )


class BindingPredictionIntervalDistributionAnalysisArgs(BaseModel):
    """Parameters for binding prediction interval distribution analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing antigen binding values",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with binding data",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving statistics analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    interval_step: float = Field(
        default=0.1,
        description="Interval step",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter interval step",
            "help_text": "Step size for distribution intervals",
            "min": 0.01,
            "max": 1.0,
            "demo_value": 0.1
        }
    )
    data_min: float = Field(
        default=0.0,
        description="Data minimum value",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter minimum value",
            "help_text": "Minimum value for data range",
            "demo_value": 0.0
        }
    )
    data_max: float = Field(
        default=1.0,
        description="Data maximum value",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter maximum value",
            "help_text": "Maximum value for data range",
            "demo_value": 1.0
        }
    )


class DifferentialGeneExpressionVolcanoAnalysisArgs(BaseModel):
    """Parameters for differential gene expression volcano analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing gene expression data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with gene expression",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    logfc_threshold: float = Field(
        default=0.0,
        description="log2 fold change threshold",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter log2FC threshold",
            "help_text": "Minimum log2 fold change for significance",
            "demo_value": 0.0
        }
    )
    min_pct: float = Field(
        default=0.2,
        description="Minimum expression percent (0.2 = 20% cells)",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter minimum percentage",
            "help_text": "Minimum fraction of cells expressing the gene",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.2
        }
    )
    analysis_strategy: str = Field(
        default="both",
        description="Analysis strategy",
        json_schema_extra={
            "ui_type": "select",
            "options": ["both", "broad", "specific"],
            "placeholder": "Select analysis strategy",
            "help_text": "Strategy for differential expression analysis"
        }
    )


class UmapDimensionalityReductionVisualizationArgs(BaseModel):
    """Parameters for UMAP dimensionality reduction visualization"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing UMAP coordinates and cell annotations (支持本地路径或 HTTP/HTTPS URL)",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path or HTTP/HTTPS URL",
            "help_text": "Single-cell RNA-seq RDS file with UMAP data (支持本地路径或 HTTP/HTTPS URL)",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving visualization results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )


class BcellMarkerGeneDotplotAnalysisArgs(BaseModel):
    """Parameters for B cell marker gene dotplot analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing gene expression and cell type data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with marker genes",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    min_pct: float = Field(
        default=0.1,
        description="Minimum expression percent threshold (0.1 = 10% cells)",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter minimum percentage",
            "help_text": "Minimum fraction of cells expressing the gene",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.1
        }
    )
    min_expression: float = Field(
        default=0.25,
        description="Minimum average expression level",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter minimum expression",
            "help_text": "Minimum average expression threshold",
            "demo_value": 0.25
        }
    )


class AntigenBindingNeutralizationDensityVisualizationArgs(BaseModel):
    """Parameters for antigen binding and neutralization density visualization"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing UMAP coordinates and prediction data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with prediction data",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving visualization results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    prediction_keywords: str = Field(
        default="neut,bind,average,predict,output",
        description="Prediction field detection keywords, comma-separated",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter keywords separated by comma",
            "help_text": "Keywords for detecting prediction fields",
            "demo_value": "neut,bind,average,predict,output"
        }
    )
    na_strategy: str = Field(
        default="exclude_cells",
        description="NA value handling strategy",
        json_schema_extra={
            "ui_type": "select",
            "options": ["exclude_cells", "replace_zero", "replace_median"],
            "placeholder": "Select NA handling strategy",
            "help_text": "How to handle NA values in data"
        }
    )
    feature_priority: str = Field(
        default="neutralization_first",
        description="Feature selection priority",
        json_schema_extra={
            "ui_type": "select",
            "options": ["neutralization_first", "binding_first", "highest_value"],
            "placeholder": "Select feature priority",
            "help_text": "Priority for feature selection"
        }
    )


class BcellCelltypeUmapVisualizationArgs(BaseModel):
    """Parameters for B cell celltype UMAP visualization"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing UMAP coordinates and cell type annotations",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with UMAP data",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving visualization results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    celltype_column: str = Field(
        default="CellType",
        description="Cell type field name",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter cell type column name",
            "help_text": "Name of the cell type column in data",
            "demo_value": "CellType"
        }
    )


class BcellMarkerGeneExpressionDotplotArgs(BaseModel):
    """Parameters for B cell marker gene expression dotplot"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing gene expression and cell type data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with gene expression",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    celltype_column: str = Field(
        default="CellType",
        description="Cell type field name",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter cell type column name",
            "help_text": "Name of the cell type column in data",
            "demo_value": "CellType"
        }
    )


class DifferentialGeneCorrelationPipelineArgs(BaseModel):
    """Parameters for integrated differential gene expression and correlation analysis pipeline"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing gene expression data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with gene expression",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving all analysis results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    logfc_threshold: float = Field(
        default=0.0,
        description="log2 fold change threshold for volcano analysis",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter log2FC threshold",
            "help_text": "Minimum log2 fold change for DEG significance",
            "demo_value": 0.0
        }
    )
    min_pct: float = Field(
        default=0.2,
        description="Minimum expression percent for volcano analysis (0.2 = 20% cells)",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter minimum percentage",
            "help_text": "Minimum fraction of cells expressing the gene",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.2
        }
    )
    p_value_threshold: float = Field(
        default=0.05,
        description="Significance p value threshold for correlation analysis",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter p-value threshold",
            "help_text": "Threshold for filtering significant DEGs in correlation",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.05
        }
    )
    min_common_genes: int = Field(
        default=10,
        description="Minimum common gene count for correlation analysis",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter minimum gene count",
            "help_text": "Minimum overlapping genes required for correlation",
            "min": 1,
            "demo_value": 10
        }
    )
    highlight_genes: str = Field(
        default="ITGAX,FGR,FCRL4,FCRL5,CD68,TNFRSF1B,JCHAIN,MZB1,XBP1,MARCKSL1",
        description="Highlight genes list for correlation plot, comma separated",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter gene names separated by comma",
            "help_text": "Genes to highlight in the correlation scatter plot",
            "demo_value": "ITGAX,FGR,FCRL4,FCRL5"
        }
    )


class DifferentialGeneCorrelationAnalysisArgs(BaseModel):
    """Parameters for differential gene correlation analysis"""
    deg_file1: str = Field(
        ...,
        description="First DEG result file path",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["csv", "tsv", "txt"],
            "placeholder": "Enter first DEG file path",
            "help_text": "CSV file with differential expression results",
            "demo_value": "/data_new/workspace/deg1.csv"
        }
    )
    deg_file2: str = Field(
        ...,
        description="Second DEG result file path",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["csv", "tsv", "txt"],
            "placeholder": "Enter second DEG file path",
            "help_text": "CSV file with differential expression results",
            "demo_value": "/data_new/workspace/deg2.csv"
        }
    )
    base_dir: str = Field(
        ...,
        description="Output directory absolute path",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    dataset1_name: str = Field(
        ...,
        description="First data set name",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter dataset name",
            "help_text": "Name for the first dataset",
            "demo_value": "Dataset1"
        }
    )
    dataset2_name: str = Field(
        ...,
        description="Second data set name",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter dataset name",
            "help_text": "Name for the second dataset",
            "demo_value": "Dataset2"
        }
    )
    p_value_threshold: float = Field(
        default=0.05,
        description="Significance p value threshold",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter p-value threshold",
            "help_text": "Threshold for statistical significance",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.05
        }
    )
    min_common_genes: int = Field(
        default=10,
        description="Minimum common gene count",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter minimum gene count",
            "help_text": "Minimum number of overlapping genes required",
            "min": 1,
            "demo_value": 10
        }
    )
    highlight_genes: str = Field(
        default="ITGAX,FGR,FCRL4,FCRL5,CD68,TNFRSF1B,JCHAIN,MZB1,XBP1,MARCKSL1",
        description="Highlight genes list, comma separated",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter gene names separated by comma",
            "help_text": "Genes to highlight in the plot",
            "demo_value": "ITGAX,FGR,FCRL4,FCRL5"
        }
    )


class PredictionValueDensityVisualizationArgs(BaseModel):
    """Parameters for prediction value density visualization"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing prediction value and UMAP coordinates",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with prediction values",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving visualization results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    prediction_keywords: str = Field(
        default="bind,predict,output,average,score",
        description="Prediction field detection keywords, comma-separated",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter keywords separated by comma",
            "help_text": "Keywords for detecting prediction fields",
            "demo_value": "bind,predict,output,average,score"
        }
    )
    prediction_threshold: float = Field(
        default=0.5,
        description="Prediction value classification threshold",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter threshold value",
            "help_text": "Threshold for prediction classification",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.5
        }
    )


class PseudotimeTrajectoryAnalysisArgs(BaseModel):
    """Parameters for pseudotime trajectory analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing Seurat objects and cell type annotations",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with Seurat object",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    num_dim: int = Field(
        default=50,
        description="Principal component dimension",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter number of dimensions",
            "help_text": "Number of PCA dimensions for trajectory analysis",
            "min": 10,
            "max": 100,
            "demo_value": 50
        }
    )
    cluster_resolution: float = Field(
        default=0.001,
        description="Cluster resolution",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter cluster resolution",
            "help_text": "Resolution for clustering (lower = fewer clusters)",
            "min": 0.0001,
            "max": 2.0,
            "demo_value": 0.001
        }
    )
    min_gene_cells: int = Field(
        default=3,
        description="Gene filtering threshold",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter minimum cells",
            "help_text": "Minimum number of cells expressing a gene",
            "min": 1,
            "demo_value": 3
        }
    )
    root_celltype: str = Field(
        default="Naive",
        description="Root cell type for trajectory",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter root cell type",
            "help_text": "Starting cell type for trajectory analysis",
            "demo_value": "Naive"
        }
    )


class PseudotimeCelltypeBoxplotAnalysisArgs(BaseModel):
    """Parameters for pseudotime celltype boxplot analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing pseudotime and cell type data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with pseudotime",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    celltype_column: str = Field(
        default="",
        description="Cell type field name (empty for auto-detection)",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter cell type column name",
            "help_text": "Name of cell type column (leave empty for auto-detection)",
            "demo_value": ""
        }
    )


class TrajectoryPolynomialRegressionAnalysisArgs(BaseModel):
    """Parameters for trajectory polynomial regression analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing trajectory and gene expression data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with trajectory data",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )


class TrajectorySupplementaryAnalysisArgs(BaseModel):
    """Parameters for trajectory supplementary analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing trajectory and gene expression data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with trajectory data",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )


class BcrIsotypeDistributionShmAnalysisArgs(BaseModel):
    """Parameters for BCR isotype distribution and SHM analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing BCR isotype and binding prediction data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with BCR data",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    binding_threshold: float = Field(
        default=0.5,
        description="Binding threshold for classification",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter threshold value",
            "help_text": "Threshold for defining broadly reactive BCRs",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.5
        }
    )


class NeutralizingAntibodyShmComparisonAnalysisArgs(BaseModel):
    """Parameters for neutralizing antibody SHM comparison analysis"""
    input_file: str = Field(
        ...,
        description="Complete path to input RDS file containing neutralization prediction and cell type data",
        json_schema_extra={
            "ui_type": "text",
            "support_upload": True,
            "support_file_types": ["rds", "RDS"],
            "placeholder": "Enter RDS file path",
            "help_text": "Single-cell RNA-seq RDS file with neutralization data",
            "demo_urls": "/data_new/workspace/Age_Bcells.rds"
        }
    )
    base_dir: str = Field(
        ...,
        description="Absolute path to output directory for saving analysis results",
        json_schema_extra={
            "ui_type": "text",
            "placeholder": "Enter output directory path",
            "help_text": "Directory for saving results",
            "demo_value": "/data_new/workspace/output"
        }
    )
    binding_threshold: float = Field(
        default=0.5,
        description="Binding threshold for classification",
        json_schema_extra={
            "ui_type": "number",
            "placeholder": "Enter threshold value",
            "help_text": "Threshold consistent with isotype analysis",
            "min": 0.0,
            "max": 1.0,
            "demo_value": 0.5
        }
    )

# MCP Tool Functions
# ============================================================================

@mcp.tool()
async def antigen_binding_prediction_visualization(args: AntigenBindingPredictionVisualizationArgs):
    """Single-cell B cell antigen binding prediction visualization analysis
    
    Performs visualization analysis of antigen binding prediction for single-cell B cell data:
    - Automatically detects and processes multiple binding prediction column formats (bind_predict, bind_output, etc.)
    - Numerical conversion and NA value handling to ensure data quality
    - Broad reactivity threshold classification and statistical analysis
    - Binding prediction value distribution visualization and density plot generation
    - Cell type-specific binding pattern analysis
    - Export binding prediction statistical results to CSV files
    
    Bioinformatics domains: ["single-cell", "B-cell", "antigen binding", "prediction analysis", "visualization"]
    Input data: ["Single-cell RNA-seq RDS files", "Seurat objects", "Binding prediction data"]
    Output results: ["Binding prediction plots", "Statistical analysis", "CSV files", "Visualization charts"]
    
    Args:
        args: AntigenBindingPredictionVisualizationArgs - Parameters for antigen binding prediction visualization
        
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
            "message": "Starting antigen binding prediction visualization",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    # 发送处理进度
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure2_module_script(
        "Figure2_A2_Binding", 
        args.input_file, 
        args.base_dir, 
        binding_threshold=args.binding_threshold
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure2"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_2A2_flu_bind_prediction.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
    
    # 发送完成进度
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    # 返回最终结果
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def bcell_celltype_distribution_analysis(args: BcellCelltypeDistributionAnalysisArgs):
    """Single-cell B cell subtype distribution visualization analysis
    
    Performs visualization analysis of cell type distribution for single-cell B cell data:
    - King dataset cell type mapping and standardized annotation
    - B cell subtype classification statistics (Naive, Memory, Germinal Center, Plasma, etc.)
    - Cell type proportion distribution calculation and visualization
    - Multi-color palette cell type coloring scheme
    - Cell type distribution pie charts and bar chart generation
    - Export cell type statistical data to CSV files
    
    Bioinformatics domains: ["single-cell", "B-cell", "cell type", "distribution analysis", "visualization"]
    Input data: ["Single-cell RNA-seq RDS files", "Seurat objects", "Cell type annotations"]
    Output results: ["Distribution plots", "Statistical charts", "CSV files", "Cell type analysis"]
    
    Args:
        args: BcellCelltypeDistributionAnalysisArgs - Parameters for B cell celltype distribution analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting B cell celltype distribution analysis",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure2_module_script("Figure2_B1_CellType", args.input_file, args.base_dir)
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure2"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_2B1_celltype_distribution.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def binding_prediction_interval_distribution_analysis(args: BindingPredictionIntervalDistributionAnalysisArgs):
    """Single-cell antigen binding prediction value interval distribution analysis
    
    Analyzes antigen binding prediction value in single-cell data:
    - Customize interval step and data range flexibility
    - Generate antigen binding prediction value interval distribution histogram
    - Calculate number of cells and percentage in each interval
    - Cumulative distribution function(CDF) calculation and visualization
    - Quantile analysis and outlier detection
    - Export interval statistics to CSV file for further analysis
    
    Bioinformatics domains: ["single-cell", "statistics analysis", "distribution analysis", "data mining", "visualization"]
    Input data: ["single-cell RNA-seq RDS files", "antigen binding data", "numerical prediction score"]
    Output results: ["distribution histogram", "statistics analysis", "CSV data", "quantile analysis"]
    
    Args:
        args: BindingPredictionIntervalDistributionAnalysisArgs - Parameters for binding prediction interval analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting binding prediction interval distribution analysis",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure2_module_script(
        "Figure2_B2_Intervals", 
        args.input_file, 
        args.base_dir, 
        interval_step=args.interval_step,
        data_min=args.data_min,
        data_max=args.data_max
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure2"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_2B2_binding_intervals.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

# @mcp.tool()
def differential_gene_expression_volcano_analysis(args: DifferentialGeneExpressionVolcanoAnalysisArgs) -> dict:
    """Single-cell differential gene expression and volcano plot visualization
    
    Analyzes single-cell B cell data for differential gene expression and volcano plot visualization:
    - Smart threshold setting, based on data distribution dynamics classification
    - Broad reaction vs specific B cell differential expression analysis
    - Seurat FindMarkers function for statistical test
    - Volcano plot generation, containing significant gene annotation and statistical information
    - Multiple analysis strategy support (broad, specific, both)
    - P value adjustment and multiple change threshold filtering
    - Export differential gene list to CSV file
    
    Bioinformatics domains: ["single-cell", "differential expression", "statistics analysis", "gene expression", "visualization"]
    Input data: ["single-cell RNA-seq RDS files", "gene expression matrix", "cell division information"]
    Output results: ["volcano plot", "differential gene list", "statistics results", "CSV files"]
    
    Args:
        args: DifferentialGeneExpressionVolcanoAnalysisArgs - Parameters for differential gene expression analysis
        
    Returns:
        Differential expression analysis result summary, including significant gene count and generated visualization files
    """
    from pathlib import Path
    
    message = run_figure2_module_script(
        "Figure2_C_Volcano", 
        args.input_file, 
        args.base_dir, 
        logfc_threshold=args.logfc_threshold,
        min_pct=args.min_pct,
        analysis_strategy=args.analysis_strategy
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure2"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_2C.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
        
        if files_dir.exists():
            import glob
            csv_files = glob.glob(str(files_dir / "DEG_analysis_threshold=*_strategy=*.csv"))
            generated_files.extend(csv_files)
    
    # 返回最终结果
    yield {
        "type": "result",
        "data": {
            "message": message,
            "result_path": generated_files
        }
    }

@mcp.tool()
async def umap_dimensionality_reduction_visualization(args: UmapDimensionalityReductionVisualizationArgs):
    """Single-cell B cell UMAP reduction and cell type visualization analysis
    
    Analyzes single-cell B cell data for UMAP reduction and cell type visualization:
    - UMAP coordinate extraction and two-dimensional space mapping
    - B cell type in UMAP space distribution visualization
    - Cell type specific color encoding and figure legend
    - High quality UMAP plot generation suitable for publication use
    - Cell density distribution and cluster boundary visualization
    - Support King dataset's cell type mapping
    - Export UMAP coordinate and cell type information to CSV file
    
    Bioinformatics domains: ["single-cell", "reduction analysis", "UMAP", "visualization", "cell group"]
    Input data: ["single-cell RNA-seq RDS files", "UMAP coordinates", "cell type annotations"]
    Output results: ["UMAP plot", "cell distribution plot", "coordinate data", "visualization file"]
    
    Args:
        args: UmapDimensionalityReductionVisualizationArgs - Parameters for UMAP visualization
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting UMAP dimensionality reduction visualization",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure2_module_script("Figure2_S2A_UMAP", args.input_file, args.base_dir)
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure2"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_S2A_fluBcells_UMAP.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def bcell_marker_gene_dotplot_analysis(args: BcellMarkerGeneDotplotAnalysisArgs):
    """B cell type specific gene expression dotplot analysis
    
    Analyzes B cell type specific gene expression dotplot:
    - B cell type specific gene expression set definition and detection
    - Gene expression level and expression ratio's double visualization
    - Dotplot size represents expression ratio, color represents expression strength
    - Expression threshold filtering, ensuring biological significance
    - Multiple B cell type specific gene expression comparison
    - Auto detect data available gene markers
    - Export gene expression statistics to CSV file
    
    Bioinformatics domains: ["single-cell", "gene expression", "gene expression", "cell type", "visualization"]
    Input data: ["single-cell RNA-seq RDS files", "gene expression matrix", "cell type annotations"]
    Output results: ["dotplot visualization", "expression statistics", "gene markers list", "CSV files"]
    
    Args:
        args: BcellMarkerGeneDotplotAnalysisArgs - Parameters for B cell marker gene dotplot analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting B cell marker gene dotplot analysis",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure2_module_script(
        "Figure2_S2C_DotPlot", 
        args.input_file, 
        args.base_dir, 
        min_pct=args.min_pct,
        min_expression=args.min_expression
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure2"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_S2C_marker_genes_dotplot.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def antigen_binding_neutralization_density_visualization(args: AntigenBindingNeutralizationDensityVisualizationArgs):
    """Single-cell antigen binding and neutralization prediction density plot visualization analysis
    
    Performs UMAP density plot visualization of antigen binding and neutralization predictions for single-cell data:
    - Automatically detects multiple prediction field formats (neut, bind, predict, etc.)
    - Flexible NA value handling strategies (exclude cells, replace with zero, replace with median)
    - Feature selection priority configuration (neutralization first, binding first, highest value first)
    - Nebulosa density plot generation showing prediction value distribution in UMAP space
    - Gradient color mapping visualization (transparent→coral→brown)
    - Supports King dataset cell type mapping
    - Export prediction value statistics and UMAP coordinate data
    
    Bioinformatics domains: ["single-cell", "UMAP", "density visualization", "antigen binding", "neutralization prediction"]
    Input data: ["single-cell RNA-seq RDS files", "UMAP coordinates", "Prediction value data"]
    Output results: ["Density plots", "UMAP visualization", "Prediction statistics", "PDF files"]
    
    Args:
        args: AntigenBindingNeutralizationDensityVisualizationArgs - Parameters for antigen binding neutralization density visualization
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting antigen binding neutralization density visualization",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure3_module_script(
        "Figure3_A_Density", 
        args.input_file, 
        args.base_dir, 
        prediction_keywords=args.prediction_keywords,
        na_strategy=args.na_strategy,
        feature_priority=args.feature_priority
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure3"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_3A.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
            
            session_info = plots_dir / "Figure3A_session_info.txt"
            if session_info.exists():
                generated_files.append(str(session_info))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def bcell_celltype_umap_visualization(args: BcellCelltypeUmapVisualizationArgs):
    """Single-cell B cell type UMAP space distribution visualization analysis
    
    Analyzes single-cell B cell data for cell type in UMAP space distribution visualization:
    - King data set cell type mapping and standardized annotation
    - B cell type in UMAP two-dimensional space distribution visualization
    - 36 tone color palette for cell type specific reactivity
    - High quality UMAP plot generation suitable for publication use
    - Cell type cluster boundary and density distribution visualization
    - Support custom cell type field name
    - Export UMAP coordinate and cell type statistics data
    
    Bioinformatics domains: ["single-cell", "UMAP", "cell type", "space distribution", "visualization"]
    Input data: ["single-cell RNA-seq RDS files", "UMAP coordinates", "cell type annotations"]
    Output results: ["UMAP plot", "cell type distribution", "statistics data", "PDF files"]
    
    Args:
        args: BcellCelltypeUmapVisualizationArgs - Parameters for B cell celltype UMAP visualization
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting B cell celltype UMAP visualization",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure3_module_script(
        "Figure3_C_CellType", 
        args.input_file, 
        args.base_dir, 
        celltype_column=args.celltype_column
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure3"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_3C.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
            
            session_info = plots_dir / "Figure3C_session_info.txt"
            if session_info.exists():
                generated_files.append(str(session_info))
        
        if files_dir.exists():
            stats_file = files_dir / "Figure3C_celltype_stats.csv"
            if stats_file.exists():
                generated_files.append(str(stats_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def bcell_marker_gene_expression_dotplot(args: BcellMarkerGeneExpressionDotplotArgs):
    """B cell type specific marker gene expression dotplot visualization analysis
    
    Analyzes B cell type specific marker gene expression dotplot:
    - B cell type specific marker gene expression set definition and detection
    - Gene expression level and expression ratio's double visualization
    - Dotplot size represents expression ratio, color represents expression strength
    - Multiple B cell type specific marker gene expression comparison
    - Auto detect data available gene markers
    - Support custom cell type field name
    - Export marker gene expression statistics and visualization result
    
    Bioinformatics domains: ["single-cell", "marker gene", "gene expression", "dotplot visualization", "cell type"]
    Input data: ["single-cell RNA-seq RDS files", "gene expression matrix", "cell type annotations"]
    Output results: ["dotplot visualization", "expression statistics", "marker gene information", "PDF files"]
    
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting B cell marker gene expression dotplot",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure3_module_script(
        "Figure3_D_DotPlot", 
        args.input_file, 
        args.base_dir, 
        celltype_column=args.celltype_column
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure3"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_3D.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
            
            session_info = plots_dir / "Figure3D_session_info.txt"
            if session_info.exists():
                generated_files.append(str(session_info))
        
        if files_dir.exists():
            for csv_name in ["Figure3D_marker_genes.csv", "Figure3D_markers_by_celltype.csv", "Figure3D_missing_genes.csv"]:
                csv_file = files_dir / csv_name
                if csv_file.exists():
                    generated_files.append(str(csv_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def differential_gene_correlation_analysis(args: DifferentialGeneCorrelationAnalysisArgs):
    """Differential gene correlation analysis and scatter plot visualization
    
    Analyzes two data sets for differential gene correlation:
    - Automatically validate input DEG file format and necessary fields
    - Filter significant differential genes with p value threshold
    - Compute Pearson correlation coefficient between two data sets
    - Generate correlation scatter plot, containing statistical significant information
    - Support custom highlight genes annotation and visualization
    - Ensure statistical significance of minimum common genes requirement
    - Export correlation data and statistical results
    
    Bioinformatics domains: ["differential expression", "correlation analysis", "statistics analysis", "gene expression", "comparison analysis"]
    Input data: ["DEG result CSV files", "differential gene list", "statistics test results"]
    Output results: ["correlation scatter plot", "statistics results", "correlation data", "PDF files"]
    
    通过 SSE 流式推送分析进度，支持大文件分析。
    
    Args:
        args: DifferentialGeneCorrelationAnalysisArgs - Parameters for differential gene correlation analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    # Figure3_F_Correlation需要特殊的参数传递方式
    # 直接调用R脚本，因为它需要两个DEG文件作为输入
    working_dir = Path(__file__).parent
    base_dir_path = Path(args.base_dir)
    
    # R脚本路径
    r_script_path = working_dir / "scripts/common/figure3_modules/Figure3_F_Correlation.R"
    
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting differential gene expression correlation analysis",
            "timestamp": time.time()
        }
    }
    
    # Check if R script exists
    if not r_script_path.exists():
        yield {
            "type": "error",
            "status": "error",
            "error_type": "file_not_found",
            "message": f"R script does not exist: {r_script_path}",
            "session_id": session_id
        }
        return
    
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting differential gene expression correlation analysis",
            "timestamp": time.time()
        }
    }
    
    # 处理 URL 下载
    temp_file_paths = []
    actual_deg_file1 = args.deg_file1
    actual_deg_file2 = args.deg_file2
    
    try:
        yield {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "status": "processing",
                "message": "Running R script analysis...",
                "timestamp": time.time()
            }
        }
        # 如果 deg_file1 是 URL，先下载到临时文件
        if args.deg_file1.startswith(('http://', 'https://')):
            try:
                temp_file1 = download_url_to_temp_file(args.deg_file1)
                temp_file_paths.append(temp_file1)
                actual_deg_file1 = temp_file1
            except Exception as e:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "download_failed",
                    "message": f"Failed to download first DEG file from URL {args.deg_file1}: {str(e)}",
                    "session_id": session_id
                }
                return
        
        # 如果 deg_file2 是 URL，先下载到临时文件
        if args.deg_file2.startswith(('http://', 'https://')):
            try:
                temp_file2 = download_url_to_temp_file(args.deg_file2)
                temp_file_paths.append(temp_file2)
                actual_deg_file2 = temp_file2
            except Exception as e:
                # 清理已下载的临时文件
                for temp_path in temp_file_paths:
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    except:
                        pass
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "download_failed",
                    "message": f"Failed to download second DEG file from URL {args.deg_file2}: {str(e)}",
                    "session_id": session_id
                }
                return
        
        # Check if input files exist
        if not os.path.exists(actual_deg_file1):
            # 清理临时文件
            for temp_path in temp_file_paths:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass
            yield {
                "type": "error",
                "status": "error",
                "error_type": "file_not_found",
                "message": f"First DEG file does not exist: {actual_deg_file1}",
                "session_id": session_id
            }
            return
        if not os.path.exists(actual_deg_file2):
            # 清理临时文件
            for temp_path in temp_file_paths:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass
            yield {
                "type": "error",
                "status": "error",
                "error_type": "file_not_found",
                "message": f"Second DEG file does not exist: {actual_deg_file2}",
                "session_id": session_id
            }
            return
        
        # Build command arguments - in the order required by R script
        cmd_args = [
            "Rscript", str(r_script_path), 
            actual_deg_file1, actual_deg_file2, str(base_dir_path),
            args.dataset1_name, args.dataset2_name
        ]
        
        # Add optional parameters
        if args.p_value_threshold is not None:
            cmd_args.append(str(args.p_value_threshold))
        if args.min_common_genes is not None:
            cmd_args.append(str(args.min_common_genes))
        if args.highlight_genes is not None:
            cmd_args.append(str(args.highlight_genes))
        
        # Execute R script
        result = subprocess.run(
            cmd_args,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800  # 30 minutes timeout
        )
        
        # Check execution result
        if result.returncode != 0:
            yield {
                "type": "error",
                "status": "error",
                "error_type": "r_script_failed",
                "message": f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}",
                "session_id": session_id
            }
            return
        
        # Collect generated files
        generated_files = []
        
        # Find generated files
        for output_dir in base_dir_path.glob("Figure3*"):
            if output_dir.is_dir():
                # CSV files
                csv_files = list((output_dir / "files").glob("*.csv")) if (output_dir / "files").exists() else []
                generated_files.extend([str(f) for f in csv_files])
                
                # PDF files
                pdf_files = list((output_dir / "plots").glob("*.pdf")) if (output_dir / "plots").exists() else []
                generated_files.extend([str(f) for f in pdf_files])
                
                # Other files
                other_files = list(output_dir.glob("*.txt")) + list(output_dir.glob("*.RData"))
                generated_files.extend([str(f) for f in other_files])
        
        success_msg = f"Figure3_F_Correlation differential gene correlation analysis executed successfully!\n"
        if generated_files:
            success_msg += f"Generated files ({len(generated_files)} files):\n"
            for file in generated_files:
                success_msg += f"  - {file}\n"
        else:
            success_msg += f"Analysis completed, please check output directory: {base_dir_path}\n"
        
        total_time = time.time() - start_time
        yield {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "status": "completed",
                "progress_percent": 100.0,
                "elapsed_seconds": round(total_time, 1),
                "message": "Analysis completed",
                "timestamp": time.time()
            }
        }
        
        yield {
            "type": "result",
            "status": "success",
            "session_id": session_id,
            "message": success_msg,
            "result_path": generated_files,
            "processing_time_ms": total_time * 1000
        }
        
    except subprocess.TimeoutExpired:
        yield {
            "type": "error",
            "data": {
                "session_id": session_id,
                "status": "error",
                "error_type": "timeout",
                "message": f"R script execution timeout (exceeded 1800 seconds)"
            }
        }
    except Exception as e:
        yield {
            "type": "error",
            "data": {
                "session_id": session_id,
                "status": "error",
                "error_type": "unknown",
                "message": f"Error occurred during R script execution: {str(e)}"
            }
        }
    finally:
        # 清理临时文件
        for temp_path in temp_file_paths:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass

@mcp.tool()
async def differential_gene_expression_correlation_pipeline(args: DifferentialGeneCorrelationPipelineArgs):
    """Integrated pipeline: differential gene expression volcano analysis + correlation analysis
    
    This pipeline automatically orchestrates two sequential bioinformatics analyses:
    
    **Step 1: Volcano Plot Analysis**
    - Identifies differentially expressed genes between broad reactive and specific B cells
    - Uses Seurat FindMarkers with 'both' strategy to generate two DEG lists
    - Applies smart threshold setting based on data distribution
    - Generates volcano plot with significant gene annotations
    - Exports two DEG CSV files (broad_control and specific_control)
    
    **Step 2: Correlation Analysis**
    - Automatically extracts the two DEG files from step 1 as input
    - Validates DEG file format and required columns (avg_log2FC, p_val_adj)
    - Filters significant genes using p-value threshold
    - Computes Pearson correlation coefficient between the two datasets
    - Generates correlation scatter plot with highlighted genes
    - Exports correlation statistics and detailed correlation data
    
    **Key Features:**
    - Seamless data flow: DEG files automatically passed between steps
    - Ensures statistical significance with minimum common genes requirement
    - Comprehensive output: Returns all 6 files from both analyses
    - Detailed execution summary with step-by-step status
    
    Bioinformatics domains: ["single-cell", "differential expression", "volcano plot", "correlation analysis", "statistics analysis", "automated pipeline"]
    Input data: ["Single-cell RNA-seq RDS files", "gene expression matrix", "cell type annotations"]
    Output results: ["Volcano plot PDF", "Two DEG CSV files", "Correlation scatter plot PDF", "Correlation statistics CSV", "Correlation data CSV"]
    
    Args:
        args: DifferentialGeneCorrelationPipelineArgs - Parameters for the integrated pipeline
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting differential gene expression correlation pipeline",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    import glob
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Step 1: Running volcano analysis...",
            "timestamp": time.time()
        }
    }
    
    # Step 1: Run volcano analysis with "both" strategy
    volcano_message = run_figure2_module_script(
        "Figure2_C_Volcano", 
        args.input_file, 
        args.base_dir, 
        logfc_threshold=args.logfc_threshold,
        min_pct=args.min_pct,
        analysis_strategy="both"  # Force "both" strategy
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure2"
    all_generated_files = []
    
    # Collect volcano analysis output files
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_2C.pdf"
            if pdf_file.exists():
                all_generated_files.append(str(pdf_file))
        
        # Find generated DEG CSV files
        deg_files = []
        if files_dir.exists():
            deg_files = glob.glob(str(files_dir / "DEG_analysis_threshold=*_strategy=*.csv"))
            all_generated_files.extend(deg_files)
    
    # Step 2: Check if we have exactly 2 DEG files
    broad_deg_file = None
    specific_deg_file = None
    
    for deg_file in deg_files:
        if "broad_control" in deg_file:
            broad_deg_file = deg_file
        elif "specific_control" in deg_file:
            specific_deg_file = deg_file
    
    correlation_message = ""
    if broad_deg_file and specific_deg_file:
        # Step 3: Run correlation analysis
        working_dir = Path(__file__).parent
        r_script_path = working_dir / "scripts/common/figure3_modules/Figure3_F_Correlation.R"
        
        if r_script_path.exists():
            try:
                cmd_args = [
                    "Rscript", str(r_script_path), 
                    broad_deg_file, specific_deg_file, str(base_dir),
                    "Broad_Control", "Specific_Control"
                ]
                
                # Add optional parameters
                if args.p_value_threshold is not None:
                    cmd_args.append(str(args.p_value_threshold))
                if args.min_common_genes is not None:
                    cmd_args.append(str(args.min_common_genes))
                if args.highlight_genes:
                    cmd_args.append(args.highlight_genes)
                
                # Execute R script
                result = subprocess.run(
                    cmd_args,
                    cwd=str(working_dir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1800
                )

                if result.returncode == 0:
                    correlation_message = "Correlation analysis completed successfully"

                    # Collect ONLY the files generated by this correlation analysis
                    # Based on dataset names: "Broad_Control" vs "Specific_Control"
                    dataset1_name = "Broad_Control"
                    dataset2_name = "Specific_Control"

                    # Check both possible locations: base_dir/Figure3* and base_dir/output/Figure3
                    figure3_dirs_to_check = []

                    # Method 1: Search for Figure3* directories in base_dir
                    for output_dir in base_dir.glob("Figure3*"):
                        if output_dir.is_dir():
                            figure3_dirs_to_check.append(output_dir)

                    # Method 2: Check fixed path base_dir/output/Figure3
                    standard_fig3_path = base_dir / "output" / "Figure3"
                    if standard_fig3_path.exists() and standard_fig3_path not in figure3_dirs_to_check:
                        figure3_dirs_to_check.append(standard_fig3_path)

                    # Collect only files matching this run's dataset names
                    for fig3_dir in figure3_dirs_to_check:
                        plots_dir = fig3_dir / "plots"
                        files_dir = fig3_dir / "files"

                        # Collect PDF with exact name match
                        if plots_dir.exists():
                            pdf_file = plots_dir / f"Figure_3F_{dataset1_name}_vs_{dataset2_name}.pdf"
                            if pdf_file.exists():
                                all_generated_files.append(str(pdf_file))

                        # Collect CSV files with exact name match
                        if files_dir.exists():
                            stats_csv = files_dir / f"correlation_stats_{dataset1_name}_vs_{dataset2_name}.csv"
                            if stats_csv.exists():
                                all_generated_files.append(str(stats_csv))

                            data_csv = files_dir / f"correlation_data_{dataset1_name}_vs_{dataset2_name}.csv"
                            if data_csv.exists():
                                all_generated_files.append(str(data_csv))
                else:
                    correlation_message = f"Correlation analysis failed: {result.stderr[:200]}"

            except subprocess.TimeoutExpired:
                correlation_message = "Correlation analysis timeout (exceeded 1800 seconds)"
            except Exception as e:
                correlation_message = f"Correlation analysis error: {str(e)}"
        else:
            correlation_message = f"Correlation R script not found: {r_script_path}"
    else:
        correlation_message = f"Insufficient DEG files for correlation analysis (found {len(deg_files)}files, need 2)"
    
    # Compile final message
    final_message = f"Pipeline execution summary:\n"
    final_message += f"Step 1 - Volcano analysis: {volcano_message.split(chr(10))[0]}\n"
    final_message += f"Step 2 - Correlation analysis: {correlation_message}\n"
    final_message += f"Total generated files: {len(all_generated_files)}"
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Pipeline completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": final_message,
        "result_path": all_generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def prediction_value_density_visualization(args: PredictionValueDensityVisualizationArgs):
    """Prediction value UMAP density plot visualization analysis
    
    Analyzes single-cell data for prediction value density plot visualization:
    - Automatically detect multiple prediction field formats (bind, predict, output etc.)
    - Based on prediction value threshold for cell classification and statistics
    - Nebulosa density plot generation, showing prediction value space distribution
    - Gradient color mapping visualization prediction strength
    - Support custom prediction field detection keywords
    - Prediction value distribution statistics and threshold analysis
    - Export prediction value data and UMAP coordinate information
    
    Bioinformatics domains: ["single-cell", "prediction analysis", "UMAP", "density visualization", "threshold analysis"]
    Input data: ["single-cell RNA-seq RDS files", "prediction value data", "UMAP coordinates"]
    Output results: ["density plot", "prediction distribution plot", "statistics analysis", "PDF files"]
    
    Args:
        args: PredictionValueDensityVisualizationArgs - Parameters for prediction value density visualization
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting prediction value density visualization",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure3_module_script(
        "Figure3_G_Prediction", 
        args.input_file, 
        args.base_dir, 
        prediction_keywords=args.prediction_keywords,
        prediction_threshold=args.prediction_threshold
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure3"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_3G.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
            
            session_info = plots_dir / "Figure3G_session_info.txt"
            if session_info.exists():
                generated_files.append(str(session_info))
        
        if files_dir.exists():
            stats_file = files_dir / "Figure3G_prediction_stats.csv"
            if stats_file.exists():
                generated_files.append(str(stats_file))
            
            columns_file = files_dir / "Figure3G_available_columns.csv"
            if columns_file.exists():
                generated_files.append(str(columns_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def pseudotime_trajectory_analysis(args: PseudotimeTrajectoryAnalysisArgs):
    """Single-cell B cell pseudotime trajectory and UMAP visualization
    
    Analyzes single-cell B cell data for pseudotime trajectory and UMAP visualization:
    - Use monocle3 for trajectory segmentation and pseudotime calculation
    - Automatically select root cell type as trajectory start (default Naive B cell)
    - Principal component analysis and reduction quality control
    - Cluster resolution optimization, suitable for trajectory analysis low resolution setting
    - Gene quality control and filtering, ensuring trajectory segmentation accuracy
    - Generate high quality pseudotime trajectory plot, suitable for publication use
    - Save monocle3 CDS object for subsequent analysis
    
    Bioinformatics domains: ["single-cell", "trajectory analysis", "pseudotime", "monocle3", "development trajectory"]
    Input data: ["single-cell RNA-seq RDS files", "Seurat objects", "cell type annotations"]
    Output results: ["trajectory plot", "CDS objects", "pseudotime data", "PDF files"]
    
    Args:
        args: PseudotimeTrajectoryAnalysisArgs - Parameters for pseudotime trajectory analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting pseudotime trajectory analysis",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    # 执行R脚本分析
    message = run_figure4_module_script(
        "Figure4_A_Trajectory", 
        args.input_file, 
        args.base_dir, 
        num_dim=args.num_dim,
        cluster_resolution=args.cluster_resolution,
        min_gene_cells=args.min_gene_cells,
        root_celltype=args.root_celltype
    )
    
    # 收集实际生成的文件路径
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure4"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_4A.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
        
        if files_dir.exists():
            rdata_file = files_dir / "flu_B_monocle_cds.RData"
            if rdata_file.exists():
                generated_files.append(str(rdata_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def pseudotime_celltype_boxplot_analysis(args: PseudotimeCelltypeBoxplotAnalysisArgs):
    """Pseudotime and cell type distribution boxplot analysis
    
    Analyzes single-cell data for pseudotime and cell type distribution boxplot analysis:
    - Depends on trajectory analysis generated CDS objects and pseudotime data
    - Automatically detect cell type field, supporting various naming formats
    - Calculate different cell types' pseudotime distribution statistics
    - Generate boxplot to show cell type along trajectory's distribution mode
    - Statistical significance test and multiple comparisons adjustment
    - Recognize developmental stage specific cell type
    - Export pseudotime statistical data and visualization result
    
    Bioinformatics domains: ["single-cell", "pseudotime", "cell type", "statistics analysis", "development stage"]
    Input data: ["single-cell RNA-seq RDS files", "pseudotime data", "cell type annotations"]
    Output results: ["boxplot", "statistics data", "pseudotime distribution", "PDF files"]
    
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting pseudotime celltype boxplot analysis",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure4_module_script(
        "Figure4_C_Boxplot", 
        args.input_file, 
        args.base_dir, 
        celltype_column=args.celltype_column
    )
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure4"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            pdf_file = plots_dir / "Figure_4C.pdf"
            if pdf_file.exists():
                generated_files.append(str(pdf_file))
        
        if files_dir.exists():
            csv_file = files_dir / "Figure4C_pseudotime_stats.csv"
            if csv_file.exists():
                generated_files.append(str(csv_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def trajectory_polynomial_regression_analysis(args: TrajectoryPolynomialRegressionAnalysisArgs):
    """Trajectory polynomial regression analysis and gene module scoring
    
    Analyzes single-cell trajectory data for polynomial regression analysis and gene module scoring:
    - Calculate B cell feature gene module scores (activation, memory, germinal center, etc.)
    - Estimate somatic hypermutation (SHM) levels based on gene expression features
    - Polynomial regression fitting and trend analysis along pseudotime trajectory
    - Identify key trajectory turning points and developmental stage markers
    - Generate combined plots showing trajectory change patterns of multiple features
    - Statistical significance testing and regression model evaluation
    - Export trajectory data and regression analysis results
    
    Bioinformatics domains: ["single-cell", "trajectory analysis", "polynomial regression", "gene modules", "SHM analysis"]
    Input data: ["single-cell RNA-seq RDS files", "pseudotime data", "gene expression matrix"]
    Output results: ["regression plots", "trajectory data", "module scores", "PDF files"]
    
    Args:
        args: TrajectoryPolynomialRegressionAnalysisArgs - Parameters for trajectory polynomial regression analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting trajectory polynomial regression analysis",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure4_module_script("Figure4_DEFG_Polynomial", args.input_file, args.base_dir)
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure4"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            combined_pdf = plots_dir / "Figure4DEFG_combined.pdf"
            if combined_pdf.exists():
                generated_files.append(str(combined_pdf))
            else:
                for i in range(1, 5):
                    part_pdf = plots_dir / f"Figure4D_E_F_G_part{i}.pdf"
                    if part_pdf.exists():
                        generated_files.append(str(part_pdf))
        
        if files_dir.exists():
            csv_file = files_dir / "Figure4DEFG_trajectory_data.csv"
            if csv_file.exists():
                generated_files.append(str(csv_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def trajectory_supplementary_analysis(args: TrajectorySupplementaryAnalysisArgs):
    """Trajectory analysis supplementary figure generation and transcriptional marker analysis
    
    Performs supplementary analysis and transcriptional marker visualization on single-cell trajectory data:
    - S6A: Expression patterns of B cell activation-related transcriptional markers along trajectory
    - S6B: Dynamic changes of atypical B cell-related transcriptional markers
    - S6C: Immunoglobulin expression dynamics and isotype switching analysis
    - S6D: Key transcription factor expression patterns and regulatory networks
    - Multi-gene expression heatmaps and trajectory visualization
    - Gene expression correlation analysis and co-expression module identification
    - Export gene expression data and statistical analysis results
    
    Bioinformatics domains: ["single-cell", "transcriptional markers", "gene expression", "trajectory analysis", "supplementary analysis"]
    Input data: ["single-cell RNA-seq RDS files", "trajectory data", "gene expression matrix"]
    Output results: ["supplementary figures", "gene expression data", "correlation analysis", "PDF files"]
    
    Args:
        args: TrajectorySupplementaryAnalysisArgs - Parameters for trajectory supplementary analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting trajectory supplementary analysis",
            "timestamp": time.time()
        }
    }
    
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    message = run_figure4_module_script("Figure4_S6_Supplementary", args.input_file, args.base_dir)
    
    base_dir = Path(args.base_dir)
    output_base = base_dir / "output" / "Figure4"
    generated_files = []
    
    if output_base.exists():
        plots_dir = output_base / "plots"
        files_dir = output_base / "files"
        
        if plots_dir.exists():
            for suffix in ['A', 'B', 'C', 'D']:
                pdf_file = plots_dir / f"FigureS6{suffix}.pdf"
                if pdf_file.exists():
                    generated_files.append(str(pdf_file))
        
        if files_dir.exists():
            for suffix in ['A', 'B', 'C', 'D']:
                csv_file = files_dir / f"FigureS6{suffix}_gene_expression.csv"
                if csv_file.exists():
                    generated_files.append(str(csv_file))
            
            availability_file = files_dir / "FigureS6_gene_availability.csv"
            if availability_file.exists():
                generated_files.append(str(availability_file))
    
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def bcr_isotype_distribution_shm_analysis(args: BcrIsotypeDistributionShmAnalysisArgs):
    """B cell receptor isotype distribution and somatic hypermutation rate analysis
    
    Performs comprehensive analysis of B cell receptor isotype distribution and somatic hypermutation (SHM) rates:
    - Analyze isotype distribution differences between broadly reactive BCRs and specific/non-binding BCRs
    - Compare SHM rates across different binding levels (broadly reactive, specific, non-binding)
    - Automatically detect and standardize isotype annotation formats from different datasets
    - Estimate SHM levels and affinity maturation degree based on gene expression features
    - Generate combined plots: isotype distribution bar chart + SHM level distribution + SHM boxplot
    - Statistical significance testing and multiple comparison correction
    - Export detailed analysis data and statistical results
    
    Bioinformatics domains: ["B cells", "antibodies", "isotype switching", "SHM analysis", "affinity maturation"]
    Input data: ["single-cell RNA-seq RDS files", "isotype annotations", "binding prediction data"]
    Output results: ["combined plots", "statistical analysis", "isotype distribution data", "PDF files"]
    
    Args:
        args: BcrIsotypeDistributionShmAnalysisArgs - Parameters for BCR isotype distribution and SHM analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting BCR isotype distribution and SHM analysis",
            "timestamp": time.time()
        }
    }
    
    import os
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    # 执行R脚本分析
    message = run_figure5_module_script(
        "Figure5_C_Isotype", 
        args.input_file, 
        args.base_dir, 
        binding_threshold=args.binding_threshold
    )
    
    # 收集实际生成的文件路径
    # 根据R脚本分析，输出路径为: base_dir/output/Figure5/
    base_dir = Path(args.base_dir)
    generated_files = []
    
    # R脚本的实际输出目录结构
    output_base = base_dir / "output" / "Figure5"
    
    if output_base.exists() and output_base.is_dir():
        # CSV数据文件 (在 files/ 子目录)
        files_dir = output_base / "files"
        if files_dir.exists():
            # Figure5C生成的具体文件
            csv_files = [
                files_dir / "Figure5C_analysis_data.csv",
                files_dir / "Figure5C_statistics.csv"
            ]
            generated_files.extend([str(f) for f in csv_files if f.exists()])
        
        # PDF图表和txt文件 (在 plots/ 子目录)
        plots_dir = output_base / "plots"
        if plots_dir.exists():
            plot_files = [
                plots_dir / "Figure5C.pdf",
                plots_dir / "Figure5C_session_info.txt"
            ]
            generated_files.extend([str(f) for f in plot_files if f.exists()])
    
    # 发送完成进度
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    # 发送最终结果
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

@mcp.tool()
async def neutralizing_antibody_shm_comparison_analysis(args: NeutralizingAntibodyShmComparisonAnalysisArgs):
    """Neutralizing antibody versus non-neutralizing antibody SHM rate comparison analysis
    
    Performs SHM rate comparison analysis between predicted neutralizing and non-neutralizing antibodies:
    - Compare SHM rate differences between predicted neutralizing and non-neutralizing antibodies
    - Focus specifically on antibody characteristics from FCRL5+ atypical B cells
    - Analyze correlation between neutralization capacity and somatic hypermutation levels
    - Isotype distribution analysis to identify dominant isotypes of neutralizing antibodies
    - Generate combined plots: isotype distribution + SHM level distribution + SHM comparison boxplot
    - Statistical significance testing and effect size calculation
    - Export neutralizing antibody characteristic data and comparative analysis results
    
    Bioinformatics domains: ["neutralizing antibodies", "SHM analysis", "antibody function", "immune protection", "viral neutralization"]
    Input data: ["single-cell RNA-seq RDS files", "neutralization prediction data", "cell type annotations"]
    Output results: ["comparison plots", "statistical analysis", "neutralizing antibody data", "PDF files"]
    
    Args:
        args: NeutralizingAntibodyShmComparisonAnalysisArgs - Parameters for neutralizing antibody SHM comparison analysis
        
    Yields:
        Progress updates and final result through SSE stream.
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting neutralizing antibody SHM comparison analysis",
            "timestamp": time.time()
        }
    }
    
    import os
    from pathlib import Path
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "processing",
            "message": "Running R script analysis...",
            "timestamp": time.time()
        }
    }
    
    # 执行R脚本分析
    message = run_figure5_module_script(
        "Figure5_D_Neutralization", 
        args.input_file, 
        args.base_dir, 
        binding_threshold=args.binding_threshold
    )
    
    # 收集实际生成的文件路径
    # 根据R脚本分析，输出路径为: base_dir/output/Figure5/
    base_dir = Path(args.base_dir)
    generated_files = []
    
    # R脚本的实际输出目录结构
    output_base = base_dir / "output" / "Figure5"
    
    if output_base.exists() and output_base.is_dir():
        # CSV数据文件 (在 files/ 子目录)
        files_dir = output_base / "files"
        if files_dir.exists():
            # Figure5D生成的具体文件
            csv_files = [
                files_dir / "Figure5D_analysis_data.csv",
                files_dir / "Figure5D_statistics.csv"
            ]
            generated_files.extend([str(f) for f in csv_files if f.exists()])
        
        # PDF图表和txt文件 (在 plots/ 子目录)
        plots_dir = output_base / "plots"
        if plots_dir.exists():
            plot_files = [
                plots_dir / "Figure5D.pdf",
                plots_dir / "Figure5D_session_info.txt"
            ]
            generated_files.extend([str(f) for f in plot_files if f.exists()])
    
    # 发送完成进度
    total_time = time.time() - start_time
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "completed",
            "progress_percent": 100.0,
            "elapsed_seconds": round(total_time, 1),
            "message": "Analysis completed",
            "timestamp": time.time()
        }
    }
    
    # 发送最终结果
    yield {
        "type": "result",
        "status": "success",
        "session_id": session_id,
        "message": message,
        "result_path": generated_files,
        "processing_time_ms": total_time * 1000
    }

# Add lifecycle management
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def bioinformatics_modules_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Handle server startup and shutdown lifecycle"""
    print("Bioinformatics modular MCP server is initializing...")
    
    try:
        yield {"initialized": True}
    finally:
        print("Bioinformatics modular MCP server is shutting down...")

# Set lifecycle
mcp.lifespan = bioinformatics_modules_lifespan

if __name__ == "__main__":
    print("Starting bioinformatics modular MCP server...")
    
    # Set network parameters
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8090
    
    # Start using SSE mode
    mcp.run(transport="sse")