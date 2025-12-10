"""
R Analysis MCP Server

This server provides R analysis tools for Figure 2-5 RSV data analysis.
"""

import subprocess
import tempfile
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
import urllib.request
from urllib.parse import urlparse
from urllib.error import URLError, HTTPError
import time
import uuid
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
logger = logging.getLogger('R_Data_Integration_MCP')

# Create MCP server
mcp = FastMCP("R Analysis Server")

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
                service_id = "r_data_integration"
                
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

def download_url_to_temp_file(url: str, default_ext: str = None) -> str:
    """
    下载 HTTP/HTTPS URL 到临时文件
    
    Args:
        url: HTTP/HTTPS URL
        default_ext: 默认文件扩展名（如果 URL 中没有扩展名）
        
    Returns:
        临时文件路径
        
    Raises:
        Exception: 如果下载失败
    """
    try:
        # 从 URL 获取文件扩展名
        parsed_url = urlparse(url)
        url_path = parsed_url.path
        # 获取文件扩展名
        ext = os.path.splitext(url_path)[1]
        if not ext and default_ext:
            ext = default_ext
        elif not ext:
            ext = '.tmp'
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # 下载文件
        urllib.request.urlretrieve(url, temp_file_path)
        
        return temp_file_path
    except Exception as e:
        raise Exception(f"Failed to download URL {url}: {str(e)}")

# # Pydantic参数模型定义
# class RunBcrStandardizeArgs(BaseModel):
#     """Parameters for BCR data standardization"""
#     bcr_file_path: str = Field(
#         ...,
#         description="Path to BCR data file (支持本地路径或 HTTP/HTTPS URL)",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/data/bcr_raw.csv or https://example.com/data.csv",
#             "help_text": "Path to the input BCR data file (CSV format, 支持本地路径或 HTTP/HTTPS URL)"
#         }
#     )
#     combine_fields: List[str] = Field(
#         ...,
#         description="Fields to combine for barcode",
#         json_schema_extra={
#             "ui_type": "array_input",
#             "placeholder": '["orig.ident", "cell", "sample_id"]',
#             "help_text": "List of column names to combine (e.g., ['orig.ident', 'cell', 'sample_id'])"
#         }
#     )
#     output_path: str = Field(
#         ...,
#         description="Output file path",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/output/bcr_standardized.csv",
#             "help_text": "Path where the standardized file will be saved"
#         }
#     )


# class RunRdsStandardizeArgs(BaseModel):
#     """Parameters for RDS data standardization"""
#     rds_file_path: str = Field(
#         ...,
#         description="Path to RDS file (支持本地路径或 HTTP/HTTPS URL)",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/data/seurat_object.rds or https://example.com/data.rds",
#             "help_text": "Path to the input RDS file containing Seurat object (支持本地路径或 HTTP/HTTPS URL)"
#         }
#     )
#     combine_fields: List[str] = Field(
#         ...,
#         description="Fields to combine for barcode",
#         json_schema_extra={
#             "ui_type": "array_input",
#             "placeholder": '["orig.ident", "seurat_clusters"]',
#             "help_text": "List of metadata fields to combine"
#         }
#     )
#     output_path: str = Field(
#         ...,
#         description="Output file path",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/output/seurat_standardized.rds",
#             "help_text": "Path where the standardized RDS file will be saved"
#         }
#     )


# class CombineCsvFilesArgs(BaseModel):
#     """Parameters for combining CSV files"""
#     csv_file_paths: List[str] = Field(
#         ...,
#         description="List of CSV file paths to combine",
#         json_schema_extra={
#             "ui_type": "array_input",
#             "placeholder": '["/data/file1.csv", "/data/file2.csv"]',
#             "help_text": "List of CSV file paths to combine"
#         }
#     )
#     output_path: str = Field(
#         ...,
#         description="Output file path",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/output/combined.csv",
#             "help_text": "Path where the combined CSV file will be saved"
#         }
#     )


# class FilterCsvByBarcodeArgs(BaseModel):
#     """Parameters for filtering CSV by barcode"""
#     csv_file_path: str = Field(
#         ...,
#         description="Path to CSV file",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/data/input.csv",
#             "help_text": "Path to the input CSV file"
#         }
#     )
#     barcode_list: List[str] = Field(
#         ...,
#         description="List of barcodes to filter",
#         json_schema_extra={
#             "ui_type": "array_input",
#             "placeholder": '["barcode1", "barcode2", "barcode3"]',
#             "help_text": "List of barcodes to keep in the filtered file"
#         }
#     )
#     output_path: str = Field(
#         ...,
#         description="Output file path",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/output/filtered.csv",
#             "help_text": "Path where the filtered CSV file will be saved"
#         }
#     )


# class ConvertCsvToRdsArgs(BaseModel):
#     """Parameters for converting CSV to RDS"""
#     csv_file_path: str = Field(
#         ...,
#         description="Path to CSV file",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/data/input.csv",
#             "help_text": "Path to the input CSV file"
#         }
#     )
#     output_path: str = Field(
#         ...,
#         description="Output RDS file path",
#         json_schema_extra={
#             "ui_type": "text",
#             "placeholder": "/output/output.rds",
#             "help_text": "Path where the RDS file will be saved"
#         }
#     )


# @mcp.tool()
# def run_bcr_standardize(args: RunBcrStandardizeArgs) -> str:

#     """
#     Standardize B-cell receptor (BCR) data files by combining specified fields to create a unified barcode identifier.
    
#     This tool is essential for BCR repertoire analysis workflows where multiple identifier fields need to be 
#     consolidated into a single 'combine_barcode' field for downstream analysis. Commonly used to merge cell 
#     identifiers, sample origins, and experimental conditions into a standardized format.
    
#     Use cases:
#     - Preparing BCR data for single-cell analysis pipelines
#     - Standardizing multi-sample BCR datasets
#     - Creating unified cell identifiers across experimental batches
    
#     Args:
#         bcr_file_path (str): Path to the input BCR data file (CSV format, 支持本地路径或 HTTP/HTTPS URL)
#         combine_fields (List[str]): List of column names to combine (e.g., ['orig.ident', 'cell', 'sample_id'])
#         output_path (str): Path where the standardized file will be saved

#     Returns:
#         str: Path to the standardized output file with combine_barcode field added
        
#     Example:
#         run_bcr_standardize("/data/bcr_raw.csv", ["orig.ident", "cell"], "/output/bcr_standardized.csv")
#     """
#     temp_file_path = None
#     try:
#         from scripts.combine.standardize_csv import standardize_csv
        
#         # 处理 URL 下载
#         actual_bcr_path = args.bcr_file_path
#         if args.bcr_file_path.startswith(('http://', 'https://')):
#             actual_bcr_path = download_url_to_temp_file(args.bcr_file_path, '.csv')
#             temp_file_path = actual_bcr_path
        
#         if not os.path.exists(actual_bcr_path):
#             return f"Error: Input file does not exist or is not accessible: {args.bcr_file_path}"
        
#         output_path = standardize_csv(bcr_file_path=actual_bcr_path, combine_fields=args.combine_fields, output_path=args.output_path)
        
#         return output_path
#     finally:
#         # 清理临时文件
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.unlink(temp_file_path)
#             except:
#                 pass


# @mcp.tool()
# def run_rds_standardize(args: RunRdsStandardizeArgs) -> str:


#     """
#     Standardize R data structure (RDS) files containing single-cell or bulk sequencing data by creating unified cell identifiers.
    
#     This tool processes Seurat objects or other R data structures stored in RDS format, combining multiple metadata 
#     fields into a standardized 'combine_barcode' identifier. Essential for integrating datasets from different 
#     experiments, batches, or processing pipelines in immunology research.
    
#     Use cases:
#     - Standardizing Seurat objects for multi-sample integration
#     - Preparing single-cell RNA-seq data for BCR/TCR analysis
#     - Creating consistent cell identifiers across experimental conditions
#     - Preprocessing data for spatial transcriptomics analysis
    
#     Args:
#         rds_file_path (str): Path to input RDS file containing single-cell or bulk data
#         combine_fields (List[str]): Metadata column names to combine (typically ['orig.ident', 'cell'])
#         output_path (str): Path where the standardized RDS file will be saved

        
#     Returns:
#         str: Detailed execution status with success/error messages and R script output
        
#     Note:
#         Commonly uses 'orig.ident,cell' combination to create unique cell identifiers across samples
        
#     Example:
#         run_rds_standardize("/data/seurat_obj.rds", ["orig.ident", "cell"], "/output/standardized.rds")
#     """
#     temp_file_path = None
#     try:
#         # 处理 URL 下载
#         actual_rds_path = args.rds_file_path
#         if args.rds_file_path.startswith(('http://', 'https://')):
#             actual_rds_path = download_url_to_temp_file(args.rds_file_path, '.rds')
#             temp_file_path = actual_rds_path
        
#         if not os.path.exists(actual_rds_path):
#             return f"Error: Input file does not exist or is not accessible: {args.rds_file_path}"
        
#         # Call standardize_rds function with default field combination
#         working_dir = Path(__file__).parent
#         result = subprocess.run(
#                 ["Rscript", "scripts/combine/standardize_rds.R", actual_rds_path, ",".join(args.combine_fields), args.output_path],        
#                 capture_output=True,
#                 text=True,
#                 encoding="utf-8",
#                 errors="replace",
#                 timeout=7200,
#                 cwd=str(working_dir)
#             )
#         # Check execution result
#         if result.returncode != 0:
#             return f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}"
        
#         # Execution successful, return output result
#         return f"R script executed successfully\nOutput information: {result.stdout}"
#     except subprocess.TimeoutExpired:
#         return f"R script execution timeout (exceeded 7200 seconds)"
#     except Exception as e:
#         return f"Error occurred while executing R script: {str(e)}"
#     finally:
#         # 清理临时文件
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.unlink(temp_file_path)
#             except:
#                 pass


# @mcp.tool()
# def run_extract_bcr_info(bcr_file_path: str, n_rows: int = 5) -> Dict[str, Any]:
#     """
#     Intelligently extract and identify B-cell receptor (BCR) data structure using large language model analysis.
    
#     This tool automatically analyzes BCR data files to identify key columns and data structure patterns. 
#     It uses AI to recognize common BCR data formats and extract essential field mappings including barcode 
#     identifiers, heavy chain sequences, and light chain sequences. Particularly useful when working with 
#     datasets from different sources or with non-standard column naming conventions.
    
#     Use cases:
#     - Analyzing unknown BCR dataset structures before processing
#     - Identifying column mappings in multi-source BCR data
#     - Quality assessment of BCR sequencing data
#     - Automated field detection for downstream analysis pipelines
    
#     Args:
#         bcr_file_path (str): Path to BCR data file (CSV, Excel, or TSV format)
#         n_rows (int): Number of sample rows to analyze for structure detection (default: 5)
        
#     Returns:
#         Dict[str, Any]: Structured analysis result containing:
#             - status: "success" or "error"
#             - message: Human-readable status description
#             - result: Extracted field mappings (bar_code, Heavy, Light chain columns) if successful
#             - error: Detailed error information if failed
            
#     Example:
#         run_extract_bcr_info("/data/bcr_sequences.csv", 10)
#     """
#     temp_file_path = None
#     try:
#         # 处理 URL 下载
#         actual_bcr_path = bcr_file_path
#         if bcr_file_path.startswith(('http://', 'https://')):
#             actual_bcr_path = download_url_to_temp_file(bcr_file_path, '.csv')
#             temp_file_path = actual_bcr_path

#         if not os.path.exists(actual_bcr_path):
#             return {
#                 "status": "error",
#                 "message": "BCR information extraction failed",
#                 "error": f"Input file does not exist or is not accessible: {bcr_file_path}"
#             }
        
#         from scripts.combine.bcr_extractor import extract_bcr_info_with_llm
#         # Call extract_bcr_info_with_llm function
#         result = extract_bcr_info_with_llm(actual_bcr_path, n_rows)
#         return {
#             "status": "success",
#             "message": "BCR information extraction successful",
#             "result": result
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": "BCR information extraction failed",
#             "error": str(e)
#         }
#     finally:
#         # 清理临时文件
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.unlink(temp_file_path)
#             except:
#                 pass


# @mcp.tool()
# def run_process_csv_to_standard(csv_file_path: str, bar_code: str, heavy: str, light: str, 
#                                    variant_seq: str, experiment: str, output_path: str = None) -> str:
#     """
#     Extract and standardize BCR sequence data with quality filtering and metadata annotation.
    
#     This tool processes raw BCR CSV files by extracting specified columns (barcode, heavy chain, light chain), 
#     renaming them to standard format (combine_barcode, Heavy, Light), and applying quality filters. It removes 
#     records with missing sequences, filters out sequences longer than 235 characters, and adds experimental 
#     metadata fields. The output is a clean, standardized CSV ready for downstream BCR analysis tools.
    
#     Quality control features:
#     - Removes rows with empty heavy or light chain sequences
#     - Filters sequences exceeding 235 characters (quality threshold)
#     - Standardizes column names for consistent downstream processing
#     - Adds experimental metadata (variant_seq, experiment, Label fields)
    
#     Use cases:
#     - Preparing raw BCR data for MetaBCR antigen specificity analysis
#     - Standardizing 10x Genomics VDJ output for repertoire analysis
#     - Quality filtering BCR sequences before clonotype analysis
#     - Creating analysis-ready datasets with experimental annotations
    
#     Args:
#         csv_file_path: Path to input CSV file with raw BCR sequence data
#         bar_code: Source column name for cell/sequence identifiers
#         heavy: Source column name for heavy chain variable region sequences
#         light: Source column name for light chain variable region sequences
#         variant_seq: Experimental variant identifier (added as metadata)
#         experiment: Experimental condition/batch identifier (added as metadata)
#         output_path: Output file path (auto-generated as *_processed.csv if None)
    
#     Returns:
#         str: Processing status with output file path, record count, and column information
        
#     Example:
#         run_process_csv_to_standard("/data/raw_bcr.csv", "cell_barcode", "VH_sequence", "VL_sequence", "variant_A", "batch_1")
#     """
#     temp_file_path = None
#     try:
#         # 处理 URL 下载
#         actual_csv_path = csv_file_path
#         if csv_file_path.startswith(('http://', 'https://')):
#             actual_csv_path = download_url_to_temp_file(csv_file_path, '.csv')
#             temp_file_path = actual_csv_path
        
#         if not os.path.exists(actual_csv_path):
#             return f"Error: Input file does not exist or is not accessible: {csv_file_path}"
        
#         from scripts.combine.bcr_extractor import process_csv_to_standard_format
#         # Call process_csv_to_standard_format function
#         result = process_csv_to_standard_format(actual_csv_path, bar_code, heavy, light, variant_seq, experiment, output_path)
#         return f"CSV processing successful\nExtraction result: {result}"
#     except Exception as e:
#         return f"CSV processing failed: {str(e)}"
#     finally:
#         # 清理临时文件
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.unlink(temp_file_path)
#             except:
#                 pass


# @mcp.tool()
# def run_integrate_rds_bcr_data(bcr_file_path: str, rds_file_path: str, output_path: str) -> str:
#     """
#     Integrate B-cell receptor repertoire data with single-cell RNA sequencing data for comprehensive immunological analysis.
    
#     This tool performs sophisticated data integration by matching BCR sequence information with corresponding 
#     single-cell transcriptomic profiles. It links BCR clonotype data (heavy/light chain sequences, CDR3 regions) 
#     with gene expression profiles from the same cells, enabling paired BCR-transcriptome analysis essential for 
#     understanding B-cell differentiation, activation states, and antigen specificity.
    
#     Use cases:
#     - Linking BCR repertoire with single-cell gene expression
#     - Creating paired datasets for B-cell functional analysis
#     - Integrating 10x Genomics VDJ and gene expression data
#     - Preparing data for clonal evolution and lineage tracing studies
#     - Combining BCR specificity with transcriptional states
    
#     Args:
#         bcr_file_path (str): Path to standardized BCR data file (CSV format with combine_barcode)
#         rds_file_path (str): Path to single-cell RNA-seq data (RDS format, typically Seurat object)
#         output_path (str): Path for the integrated output file (RDS format with BCR annotations)

        
#     Returns:
#         str: Detailed integration status including cell matching statistics, data quality metrics, and file paths
        
#     Example:
#         run_integrate_rds_bcr_data("/data/bcr_standardized.csv", "/data/scrna_seurat.rds", "/output/integrated_bcr_scrna.rds")
#     """
#     temp_bcr_path = None
#     temp_rds_path = None
#     try:
#         # 处理 URL 下载
#         actual_bcr_path = bcr_file_path
#         actual_rds_path = rds_file_path
        
#         if bcr_file_path.startswith(('http://', 'https://')):
#             actual_bcr_path = download_url_to_temp_file(bcr_file_path, '.csv')
#             temp_bcr_path = actual_bcr_path
        
#         if rds_file_path.startswith(('http://', 'https://')):
#             actual_rds_path = download_url_to_temp_file(rds_file_path, '.rds')
#             temp_rds_path = actual_rds_path
        
#         if not os.path.exists(actual_bcr_path):
#             return f"Error: BCR file does not exist or is not accessible: {bcr_file_path}"
#         if not os.path.exists(actual_rds_path):
#             return f"Error: RDS file does not exist or is not accessible: {rds_file_path}"
        
#         working_dir = Path(__file__).parent
#         result = subprocess.run(
#                 ["Rscript", "scripts/combine/integrate_bcr_data.R", actual_bcr_path, actual_rds_path, output_path],        
#                 capture_output=True,
#                 text=True,
#                 encoding="utf-8",
#                 errors="replace",
#                 timeout=7200,
#                 cwd=str(working_dir)
#             )
#         # Check execution result
#         if result.returncode != 0:
#             return f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}"
        
#         # Execution successful, return output result
#         return f"R script executed successfully\nOutput information: {result.stdout}"
#     except subprocess.TimeoutExpired:
#         return f"R script execution timeout (exceeded 7200 seconds)"
#     except Exception as e:
#         return f"Error occurred while executing R script: {str(e)}"
#     finally:
#         # 清理临时文件
#         for temp_path in [temp_bcr_path, temp_rds_path]:
#             if temp_path and os.path.exists(temp_path):
#                 try:
#                     os.unlink(temp_path)
#                 except:
#                     pass


# ============================================================================
# Integrate All - 异步版本的完整BCR数据整合工具
# ============================================================================

class IntegrateAllArgs(BaseModel):
    """Parameters for integrate_all.R tool"""
    csv_file: str = Field(
        ...,
        description="Path to CSV/Excel file containing BCR prediction data"
    )
    rds_file: str = Field(
        ...,
        description="Path to RDS file containing Seurat single-cell RNA-seq data"
    )
    output_file: str = Field(
        ...,
        description="Path for output integrated RDS file"
    )
    csv_fields: Optional[str] = Field(
        default=None,
        description="Comma-separated CSV fields to combine for barcode (e.g., 'BarCode' or 'Sample,Barcode')"
    )
    rds_fields: Optional[str] = Field(
        default=None,
        description="Comma-separated RDS fields to combine for barcode (e.g., 'rownames' or 'orig.ident,barcode')"
    )
    separator: Optional[str] = Field(
        default="_",
        description="Separator for combining fields into barcode"
    )
    skip_umap: bool = Field(
        default=False,
        description="Skip UMAP dimensionality reduction analysis"
    )
    skip_annotation: bool = Field(
        default=False,
        description="Skip cell type annotation based on marker genes"
    )


@mcp.tool()
async def integrate_bcr_data_complete(args: IntegrateAllArgs):
    """
    Complete BCR data integration pipeline with UMAP, clustering, and cell type annotation.
    
    This advanced tool performs comprehensive integration of BCR prediction data with single-cell
    RNA-seq data, including:
    - Automatic detection and conversion of Excel files
    - Intelligent field version control (protects Heavy/Light chains, versions prediction fields)
    - UMAP dimensionality reduction and visualization
    - FindClusters for cell clustering analysis
    - Marker gene-based cell type annotation with confidence scoring
    
    The tool automatically handles:
    - Excel (.xlsx, .xls) to CSV conversion
    - Field standardization and barcode matching
    - Smart version control for repeated integrations
    - Complete B-cell subset annotation (Naive, Memory, Plasma, GC, etc.)
    
    Use cases:
    - One-click BCR prediction data integration
    - Complete single-cell B-cell analysis pipeline
    - Version-controlled data updates
    - Reproducible B-cell immunology analysis
    
    Args:
        args: IntegrateAllArgs containing all parameters
        
    Yields:
        Progress updates and final result through SSE stream.
        
    Example:
        integrate_bcr_data_complete(IntegrateAllArgs(
            csv_file="/data/predictions.xlsx",
            rds_file="/data/Age_Bcells.rds", 
            output_file="/output/integrated_complete.rds",
            csv_fields="BarCode",
            rds_fields="rownames"
        ))
    """
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    yield {
        "type": "progress",
        "data": {
            "session_id": session_id,
            "status": "initializing",
            "message": "Starting complete BCR data integration",
            "timestamp": time.time()
        }
    }
    
    temp_csv_path = None
    temp_rds_path = None
    
    try:
        yield {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "status": "processing",
                "message": "Processing files and running R script...",
                "timestamp": time.time()
            }
        }
        
        # 处理 URL 下载
        actual_csv_path = args.csv_file
        actual_rds_path = args.rds_file
        
        if args.csv_file.startswith(('http://', 'https://')):
            actual_csv_path = download_url_to_temp_file(args.csv_file, '.csv')
            temp_csv_path = actual_csv_path
        
        if args.rds_file.startswith(('http://', 'https://')):
            actual_rds_path = download_url_to_temp_file(args.rds_file, '.rds')
            temp_rds_path = actual_rds_path
        
        # 验证文件存在
        if not os.path.exists(actual_csv_path):
            yield {
                "type": "error",
                "status": "error",
                "error_type": "file_not_found",
                "message": f"CSV/Excel file does not exist or is not accessible: {args.csv_file}",
                "session_id": session_id
            }
            return
        if not os.path.exists(actual_rds_path):
            yield {
                "type": "error",
                "status": "error",
                "error_type": "file_not_found",
                "message": f"RDS file does not exist or is not accessible: {args.rds_file}",
                "session_id": session_id
            }
            return
        
        # 构建命令行参数
        working_dir = Path(__file__).parent
        cmd = [
            "Rscript",
            "scripts/combine/integrate_all.R",
            f"--csv={actual_csv_path}",
            f"--rds={actual_rds_path}",
            f"--output={args.output_file}"
        ]
        
        # 添加可选参数
        if args.csv_fields:
            cmd.append(f"--csv-fields={args.csv_fields}")
        if args.rds_fields:
            cmd.append(f"--rds-fields={args.rds_fields}")
        if args.separator and args.separator != "_":
            cmd.append(f"--separator={args.separator}")
        if args.skip_umap:
            cmd.append("--skip-umap")
        if args.skip_annotation:
            cmd.append("--skip-annotation")
        
        # 执行 R 脚本
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=7200,
            cwd=str(working_dir)
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
        
        # Execution successful
        total_time = time.time() - start_time
        yield {
            "type": "progress",
            "data": {
                "session_id": session_id,
                "status": "completed",
                "progress_percent": 100.0,
                "elapsed_seconds": round(total_time, 1),
                "message": "Integration completed",
                "timestamp": time.time()
            }
        }
        
        yield {
            "type": "result",
            "status": "success",
            "session_id": session_id,
            "message": f"R script executed successfully\nOutput information: {result.stdout}",
            "processing_time_ms": total_time * 1000
        }
        
    except subprocess.TimeoutExpired:
        yield {
            "type": "error",
            "status": "error",
            "error_type": "timeout",
            "message": "R script execution timeout (exceeded 7200 seconds)",
            "session_id": session_id
        }
        return
    except Exception as e:
        yield {
            "type": "error",
            "status": "error",
            "error_type": "execution_error",
            "message": f"Error occurred while executing R script: {str(e)}",
            "session_id": session_id
        }
        return
    finally:
        # 清理临时文件
        for temp_path in [temp_csv_path, temp_rds_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass


# Add lifecycle management
from contextlib import asynccontextmanager

@asynccontextmanager
async def figure_analysis_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Handle server startup and shutdown"""
    print("R Analysis MCP Server is initializing...")
    
    try:
        yield {"initialized": True}
    finally:
        print("R Analysis MCP Server is shutting down...")

# Set lifecycle
mcp.lifespan = figure_analysis_lifespan

if __name__ == "__main__":
    print("Starting R Analysis MCP Server...")
    
    # Set network parameters
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 18105
    
    # Start using SSE mode
    mcp.run(transport="sse")
