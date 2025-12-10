"""
FDG MCP Server - Core FDG Tool Wrapper

This server exposes the core FDG (Foldx, DDG, GearBind) process via MCP protocol.
"""
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import StreamingResponse
from Bio.PDB import MMCIFParser, PDBIO
from tqdm import tqdm
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import os
import sys
import pandas as pd
import re
import time
import uuid
# Add project root directory to Python path - ensure this line is before all import statements
# 注意：这行代码在模块级别执行时，__file__ 可能不可用，需要处理
try:
    _file_path = __file__
except NameError:
    _file_path = os.path.abspath('af3_mcp_server.py')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(_file_path))))
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import csv_run_af3
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
logger = logging.getLogger('Alphafold3_MCP')

# Create MCP server
mcp = FastMCP("Alphafold3 Core Server")

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
                service_id = "af3"
                
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


def read_csv_robust(file_path: str, encodings: list = None) -> pd.DataFrame:
    """Robustly read CSV file with multiple strategies to handle format issues
    
    Args:
        file_path: Path to CSV file
        encodings: List of encodings to try (default: common encodings)
        
    Returns:
        DataFrame if successful, None otherwise
    """
    if encodings is None:
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252', 'iso-8859-1']
    
    # Strategy 1: Try with default settings and error handling
    for encoding in encodings:
        try:
            # Try with on_bad_lines='skip' (pandas >= 1.3) or error_bad_lines=False (pandas < 1.3)
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', engine='python')
            except TypeError:
                # Fallback for older pandas versions
                try:
                    df = pd.read_csv(file_path, encoding=encoding, error_bad_lines=False, warn_bad_lines=False, engine='python')
                except TypeError:
                    # Fallback for even older versions
                    df = pd.read_csv(file_path, encoding=encoding, error_bad_lines=False, engine='python')
            
            if df is not None and not df.empty:
                return df
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    # Strategy 2: Try with different separators
    separators = [',', '\t', ';', '|']
    for encoding in encodings:
        for sep in separators:
            try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip', engine='python', quotechar='"', skipinitialspace=True)
                except TypeError:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep, error_bad_lines=False, warn_bad_lines=False, engine='python', quotechar='"', skipinitialspace=True)
                    except TypeError:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep, error_bad_lines=False, engine='python', quotechar='"', skipinitialspace=True)
                
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
    
    # Strategy 3: Try with quoting and escape characters
    for encoding in encodings:
        try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', engine='python', 
                               quoting=1, escapechar='\\', doublequote=True)
            except TypeError:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, error_bad_lines=False, warn_bad_lines=False, engine='python',
                                   quoting=1, escapechar='\\', doublequote=True)
                except TypeError:
                    df = pd.read_csv(file_path, encoding=encoding, error_bad_lines=False, engine='python',
                                   quoting=1, escapechar='\\', doublequote=True)
            
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    
    # Strategy 4: Try reading with low_memory=False and dtype=str to handle mixed types
    for encoding in encodings:
        try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', engine='python', 
                               low_memory=False, dtype=str)
            except TypeError:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, error_bad_lines=False, warn_bad_lines=False, engine='python',
                                   low_memory=False, dtype=str)
                except TypeError:
                    df = pd.read_csv(file_path, encoding=encoding, error_bad_lines=False, engine='python',
                                   low_memory=False, dtype=str)
            
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    
    return None


def detect_file_type(file_path: str) -> str:
    """Detect file type by reading magic bytes
    
    Args:
        file_path: File path
        
    Returns:
        File type: 'xlsx', 'xls', 'csv', or 'unknown'
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)  # Read more bytes for better detection
        
        # Excel 2007+ (XLSX) is a ZIP file - check for ZIP signature
        if header.startswith(b'PK\x03\x04'):
            # Additional check: XLSX files typically have specific structure
            # Check if it contains Office document structure indicators
            f.seek(0)
            content = f.read(1024)
            if b'[Content_Types].xml' in content or b'xl/' in content or b'workbook' in content.lower():
                return 'xlsx'
            # Could still be a ZIP file, but might be XLSX
            return 'xlsx'
        # Excel 97-2003 (XLS) - OLE2 compound document format
        elif header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            return 'xls'
        # Check if file appears to be binary (contains null bytes or high percentage of non-printable)
        elif b'\x00' in header[:16]:
            # Likely binary file, try to determine if it's Excel
            f.seek(0)
            sample = f.read(512)
            if b'PK\x03\x04' in sample:
                return 'xlsx'
            elif b'\xd0\xcf\x11\xe0' in sample:
                return 'xls'
            else:
                return 'unknown'  # Binary but not recognized Excel format
        # Try to read as text (CSV)
        else:
            return 'csv'
    except Exception:
        return 'unknown'


def validate_input_file(file_path: str) -> dict:
    """Validate if the input file meets standard format requirements
    
    Standard field requirements:
    - ID: Antibody identifier (required)
    - Heavy_Chain: Heavy chain sequence (required)
    - Light_Chain: Light chain sequence (required)
    - Antigen: Antigen name (optional, uses default value if not provided)
    
    Args:
        file_path: Input file path
        
    Returns:
        Dictionary containing validation results
    """
    # Define required fields
    REQUIRED_FIELDS = ['clone_id', 'Heavy', 'Light']  # Adapted to original field names in csv_run_af3.py
    OPTIONAL_FIELDS = ['Antigen']
    ALL_VALID_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS
    
    try:
        # Detect actual file type first
        detected_type = detect_file_type(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        df = None
        read_error = None
        
        # Try to read file based on detected type (prioritize detected type over extension)
        if detected_type == 'xlsx':
            # File is actually XLSX, even if extension is .csv
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except Exception as e:
                read_error = f'Failed to read Excel file (XLSX): {str(e)}'
                # Try alternative engine
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                    read_error = None
                except Exception:
                    pass
        elif detected_type == 'xls':
            # File is actually XLS, even if extension is .csv
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                read_error = f'Failed to read Excel file (XLS): {str(e)}'
        elif detected_type == 'unknown':
            # Unknown binary file - try Excel formats first, then CSV
            read_error = 'File appears to be binary but format is unknown. Trying Excel formats...'
            # Try XLSX first
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                read_error = None
            except:
                # Try XLS
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                    read_error = None
                except:
                    # Last resort: try CSV
                    df = read_csv_robust(file_path)
                    if df is None:
                        read_error = f'File appears to be binary but could not be read as Excel or CSV. File extension: {file_ext}, Detected type: {detected_type}'
        else:
            # Try to read as CSV with robust reading function
            df = read_csv_robust(file_path)
            if df is None:
                read_error = f'Failed to read CSV file with any strategy. File may be binary, corrupted, or have severe format issues. Detected type: {detected_type}'
            else:
                # Validate that CSV columns look reasonable (not binary garbage)
                columns = df.columns.tolist()
                # Check if columns contain mostly non-printable characters (binary garbage)
                # Note: We allow Unicode characters (including Chinese, Japanese, etc.)
                # Only flag true binary control characters (0-31 except common whitespace)
                garbage_columns = []
                for col in columns:
                    if isinstance(col, str):
                        # Count true binary control characters (excluding common whitespace: space, tab, newline)
                        # Control characters are 0-31, but we exclude common ones: 9 (tab), 10 (LF), 13 (CR), 32 (space)
                        binary_chars = sum(1 for c in col if (ord(c) < 9 or (9 < ord(c) < 32)) and c not in ['\t', '\n', '\r', ' '])
                        # Also check for NULL bytes and other obvious binary indicators
                        has_null = '\x00' in col
                        # If more than 30% are binary control chars or has NULL bytes, it's likely garbage
                        if len(col) > 0 and (has_null or (binary_chars / len(col) > 0.3)):
                            garbage_columns.append(col)
                
                if garbage_columns:
                    # Likely binary file misread as CSV - try Excel formats
                    read_error = f'CSV columns appear to contain binary data (garbage): {garbage_columns[:3]}. File may actually be Excel format. Trying Excel readers...'
                    df = None
                    # Try XLSX
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                        read_error = None
                    except:
                        # Try XLS
                        try:
                            df = pd.read_excel(file_path, engine='xlrd')
                            read_error = None
                        except:
                            read_error = f'File appears to be binary (detected garbage columns: {garbage_columns[:3]}), but could not be read as Excel. Please check file format.'
        
        if read_error or df is None:
            # Provide more helpful error message
            error_msg = read_error or 'File could not be read'
            if 'binary' in error_msg.lower() or 'garbage' in error_msg.lower() or detected_type in ['xlsx', 'xls', 'unknown']:
                error_msg += f' The file extension is {file_ext}, but the actual file type appears to be {detected_type}. '
                error_msg += 'If the file is an Excel file (.xlsx or .xls), please ensure it is downloaded correctly and not corrupted.'
            return {
                'valid': False,
                'error': error_msg,
                'missing_fields': REQUIRED_FIELDS,
                'extra_fields': [],
                'empty_required_fields': [],
                'row_count': 0,
                'antigen_name': "H5N1_TEXAS",
                'columns': [],
                'detected_type': detected_type,
                'file_extension': file_ext
            }
        
        if df.empty:
            return {
                'valid': False,
                'error': 'File is empty',
                'missing_fields': REQUIRED_FIELDS,
                'extra_fields': [],
                'row_count': 0
            }
        
        columns = df.columns.tolist()
        
        # Map alternative column names to standard names
        # This allows flexibility in input file formats
        column_mapping = {}
        found_required_fields = {}
        
        # Define alternative column names for each required field
        field_alternatives = {
            'clone_id': ['clone_id', 'ID', 'id', 'Id', 'clone_ID', 'Clone_ID', 'Clone_id', 'main_name', 'Main_name', 'MAIN_NAME'],
            'Heavy': ['Heavy', 'Heavy_Chain', 'heavy', 'HEAVY', 'Heavy_chain', 'H', 'Heavy_DNA', 'heavy_DNA', 'HEAVY_DNA'],
            'Light': ['Light', 'Light_Chain', 'light', 'LIGHT', 'Light_chain', 'L', 'Light_DNA', 'light_DNA', 'LIGHT_DNA']
        }
        
        # Find matching columns for each required field
        for required_field, alternatives in field_alternatives.items():
            found = False
            for alt in alternatives:
                if alt in columns:
                    column_mapping[required_field] = alt
                    found_required_fields[required_field] = alt
                    found = True
                    break
            if not found:
                found_required_fields[required_field] = None
        
        # Check required fields (using mapped names)
        missing_fields = [field for field in REQUIRED_FIELDS if found_required_fields[field] is None]
        
        # Check extra fields (fields not in standard field list or mapped fields)
        mapped_columns = list(column_mapping.values())
        extra_fields = [col for col in columns if col not in ALL_VALID_FIELDS and col not in mapped_columns]
        
        # Check data integrity - only flag if ALL rows are empty (not just some)
        empty_required_fields = []
        for field in REQUIRED_FIELDS:
            if found_required_fields[field] is not None:
                actual_col = found_required_fields[field]
                # Check if all values are empty (not just some)
                if actual_col in df.columns:
                    # Count non-empty values (not NaN and not empty string)
                    non_empty_count = df[actual_col].notna().sum()
                    # Also check for non-empty strings
                    if non_empty_count > 0:
                        non_empty_str_count = df[actual_col].astype(str).str.strip().ne('').sum()
                        if non_empty_str_count == 0:
                            # All values are empty strings
                            empty_required_fields.append(field)
                    else:
                        # All values are NaN
                        empty_required_fields.append(field)
        
        # Get antigen name
        antigen_name = "H5N1_TEXAS"  # Default value
        if 'Antigen' in columns and not df['Antigen'].isna().all():
            # Get the first non-empty antigen name
            first_antigen = df['Antigen'].dropna().iloc[0] if not df['Antigen'].dropna().empty else antigen_name
            antigen_name = str(first_antigen).strip()
        
        # Determine if file is valid
        is_valid = (len(missing_fields) == 0 and len(empty_required_fields) == 0)
        
        result = {
            'valid': is_valid,
            'missing_fields': missing_fields,
            'extra_fields': extra_fields,
            'empty_required_fields': empty_required_fields,
            'row_count': len(df),
            'antigen_name': antigen_name,
            'columns': columns,
            'column_mapping': column_mapping  # Add mapping info for later use
        }
        
        if not is_valid:
            error_msgs = []
            if missing_fields:
                error_msgs.append(f"Missing required fields: {', '.join(missing_fields)}")
            if empty_required_fields:
                error_msgs.append(f"The following required fields contain empty values: {', '.join(empty_required_fields)}")
            result['error'] = '; '.join(error_msgs)
        
        if extra_fields:
            print(f"Warning: Found extra fields {extra_fields}, will be ignored")
        
        return result
        
    except UnicodeDecodeError as e:
        return {
            'valid': False,
            'error': f'UTF-8 decoding error: File appears to be binary or corrupted. {str(e)}',
            'missing_fields': REQUIRED_FIELDS,
            'extra_fields': [],
            'empty_required_fields': [],
            'row_count': 0,
            'antigen_name': "H5N1_TEXAS",
            'columns': []
        }
    except Exception as e:
        return {
            'valid': False,
            'error': f'File reading failed: {str(e)}',
            'missing_fields': REQUIRED_FIELDS,
            'extra_fields': [],
            'empty_required_fields': [],
            'row_count': 0,
            'antigen_name': "H5N1_TEXAS",
            'columns': []
        }


class AlphaFold3Args(BaseModel):
    """Parameters for AlphaFold3 structure prediction"""
    
    input_file_path: str = Field(
        ...,
        description="Path to the input Excel file (.xlsx) or CSV file (.csv) containing antibody sequences.",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["xlsx", "csv"],
            "placeholder": "Enter file path or upload Excel/CSV file",
            "help_text": "File must contain columns: clone_id, Heavy, Light, and optionally Antigen",
            "demo_urls": "/data_new/workspace/20250401_AF3.xlsx"
        }
    )
    
    antigen_name: Optional[str] = Field(
        default="H5N1_TEXAS",
        description="Name of the antigen. Optional, will use file content or default value if not provided.",
        json_schema_extra={
            "ui_type": "text_input",
            "placeholder": "Enter antigen name"
        }
    )
    
    gpu_device: Optional[str] = Field(
        default="3",
        description="GPU device ID to use. Default '3'.",
        json_schema_extra={
            "ui_type": "text_input",
            "placeholder": "Enter GPU device ID"
        }
    )


@mcp.tool()
async def alphafold3(args: AlphaFold3Args):
    """Uses AlphaFold3 to predict the 3D structure of antibody sequences from an input Excel file and saves the result as a PDB file.

    This function reads an Excel file containing antibody sequences (heavy and light chains), 
    uses AlphaFold3 to predict the 3D structure of each antibody, and writes the predicted structures to a PDB file.
    AlphaFold3 is a state-of-the-art deep learning model for protein structure prediction.

    Args:
        args: AlphaFold3Args - Parameters for AlphaFold3 structure prediction

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
            "message": "Starting AlphaFold3 prediction",
            "timestamp": time.time()
        }
    }
    
    # Set base path
    root_dir = '/data_new/workspace/antibody_gen/mcp_af3'
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    
    try:
        # First validate input file format
        validation_result = validate_input_file(args.input_file_path)
        
        if not validation_result['valid']:
            error_msg = f"Input file format does not meet requirements: {validation_result['error']}"
            print(error_msg)
            yield {
                "type": "error",
                "status": "error",
                "error_type": "invalid_file_format",
                "message": error_msg,
                "details": validation_result,
                "session_id": session_id
            }
            return
        
        print(f"✓ File validation passed, {validation_result['row_count']} rows of data")
        
        # Use antigen name from validation result
        antigen_name = args.antigen_name if args.antigen_name else validation_result['antigen_name']
        
        print(f"Using antigen name: {antigen_name}")
        
        # 如果数据行数超过100，筛选出100条合适的数据
        MAX_ROWS = 100
        original_row_count = validation_result['row_count']
        
        if original_row_count > MAX_ROWS:
            yield {
                "type": "progress",
                "data": {
                    "session_id": session_id,
                    "status": "filtering",
                    "message": f"Input file contains {original_row_count} rows, filtering to {MAX_ROWS} rows for processing",
                    "timestamp": time.time()
                }
            }
            
            # 读取完整文件
            detected_type = validation_result.get('detected_type', 'csv')
            file_ext = os.path.splitext(args.input_file_path)[1].lower()
            
            df_full = None
            if detected_type == 'xlsx' or (file_ext in ['.xlsx', '.xls'] and detected_type != 'csv'):
                try:
                    df_full = pd.read_excel(args.input_file_path, engine='openpyxl')
                except:
                    try:
                        df_full = pd.read_excel(args.input_file_path, engine='xlrd')
                    except:
                        df_full = read_csv_robust(args.input_file_path)
            elif detected_type == 'xls':
                try:
                    df_full = pd.read_excel(args.input_file_path, engine='xlrd')
                except:
                    df_full = read_csv_robust(args.input_file_path)
            else:
                df_full = read_csv_robust(args.input_file_path)
            
            if df_full is None or df_full.empty:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "file_read_failed",
                    "message": "Failed to read input file for filtering",
                    "session_id": session_id
                }
                return
            
            # 筛选策略：优先选择数据完整、质量好的行
            # 1. 首先过滤掉 Heavy 或 Light 为空的行
            df_filtered = df_full.copy()
            required_cols = ['Heavy', 'Light']
            
            # 检查列名是否存在（支持变体）
            heavy_col = None
            light_col = None
            for col in df_filtered.columns:
                if col in ['Heavy', 'Heavy_Chain', 'heavy', 'HEAVY']:
                    heavy_col = col
                if col in ['Light', 'Light_Chain', 'light', 'LIGHT']:
                    light_col = col
            
            if heavy_col and light_col:
                # 过滤掉空值行
                df_filtered = df_filtered.dropna(subset=[heavy_col, light_col])
                # 过滤掉空字符串
                df_filtered = df_filtered[
                    (df_filtered[heavy_col].astype(str).str.strip() != '') &
                    (df_filtered[light_col].astype(str).str.strip() != '')
                ]
            
            # 2. 如果过滤后仍超过100行，按以下优先级选择：
            #    - 优先选择序列长度合理的（Heavy 和 Light 长度在合理范围内）
            if len(df_filtered) > MAX_ROWS and heavy_col and light_col:
                # 计算序列长度并筛选
                df_filtered['_heavy_len'] = df_filtered[heavy_col].astype(str).str.len()
                df_filtered['_light_len'] = df_filtered[light_col].astype(str).str.len()
                
                # 过滤掉序列长度异常的行（太短或太长）
                # 抗体序列通常在 100-500 氨基酸之间
                df_filtered = df_filtered[
                    (df_filtered['_heavy_len'] >= 50) & (df_filtered['_heavy_len'] <= 600) &
                    (df_filtered['_light_len'] >= 50) & (df_filtered['_light_len'] <= 400)
                ]
                
                # 如果仍然超过100行，选择序列长度最接近平均值的行
                if len(df_filtered) > MAX_ROWS:
                    heavy_mean = df_filtered['_heavy_len'].mean()
                    light_mean = df_filtered['_light_len'].mean()
                    df_filtered['_distance'] = (
                        (df_filtered['_heavy_len'] - heavy_mean).abs() +
                        (df_filtered['_light_len'] - light_mean).abs()
                    )
                    df_filtered = df_filtered.nsmallest(MAX_ROWS, '_distance')
                
                # 删除临时列
                df_filtered = df_filtered.drop(columns=['_heavy_len', '_light_len', '_distance'], errors='ignore')
            
            # 3. 如果仍然超过100行，随机采样
            if len(df_filtered) > MAX_ROWS:
                df_filtered = df_filtered.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
            
            # 4. 保存筛选后的文件到临时位置
            import tempfile
            temp_dir = tempfile.gettempdir()
            filtered_file_name = f"filtered_{os.path.basename(args.input_file_path)}"
            filtered_file_path = os.path.join(temp_dir, filtered_file_name)
            
            # 根据原文件格式保存
            if file_ext in ['.xlsx', '.xls']:
                df_filtered.to_excel(filtered_file_path, index=False, engine='openpyxl')
            else:
                df_filtered.to_csv(filtered_file_path, index=False)
            
            # 使用筛选后的文件路径
            filtered_input_path = filtered_file_path
            print(f"✓ Filtered {original_row_count} rows to {len(df_filtered)} rows, saved to {filtered_file_path}")
            
            yield {
                "type": "progress",
                "data": {
                    "session_id": session_id,
                    "status": "filtered",
                    "message": f"Filtered {original_row_count} rows to {len(df_filtered)} rows for processing",
                    "original_row_count": original_row_count,
                    "filtered_row_count": len(df_filtered),
                    "timestamp": time.time()
                }
            }
            
            # 重新验证筛选后的文件（快速验证）
            validation_result = validate_input_file(filtered_input_path)
            if not validation_result['valid']:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "filtering_failed",
                    "message": f"Filtered file validation failed: {validation_result.get('error', 'Unknown error')}",
                    "session_id": session_id
                }
                return
            
            # 更新输入文件路径为筛选后的文件
            args.input_file_path = filtered_input_path
            print(f"✓ Using filtered file with {validation_result['row_count']} rows")
        else:
            print(f"✓ File has {original_row_count} rows, no filtering needed")
        
        # Get input file name
        input_file_name = os.path.basename(args.input_file_path)
        input_name, suffix_ext = os.path.splitext(input_file_name)
        # Keep the dot in suffix for convert2afformat (it expects '.csv' or '.xlsx')
        suffix = suffix_ext if suffix_ext else '.csv'
        
        # Copy input file to AF3 directory for processing
        csv_dir = os.path.join(root_dir, 'af3_inputs', 'csv_files')
        os.makedirs(csv_dir, exist_ok=True)
        target_path = os.path.join(csv_dir, input_file_name)
        
        if args.input_file_path != target_path and not os.path.exists(target_path):
            import shutil
            shutil.copy2(args.input_file_path, target_path)
            print(f"Copied input file to {target_path}")
        
        # Set output name
        output_name = f"{input_name}_{antigen_name}"
        
        # Add root_dir to system path for imports
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        
        # Dynamically set csv_run_af3 parameters
        print(f"Calling convert2afformat to process input file...")
        
        # Generate dynamic letter mapping - using fixed letter mapping
        antigen_letter = antigen_name[:3].upper() if len(antigen_name) >= 3 else "AGN"
        heavy_letter = "H"
        light_letter = "L"
        
        # Get actual columns from validation result
        actual_columns = validation_result['columns']
        
        # Get column mapping from validation result (if available)
        column_mapping = validation_result.get('column_mapping', {})
        
        # Build HEADER format expected by csv_run_af3
        # Build csv_header mapping: map from file column names to internal column names
        # Use the mapping from validation result if available
        csv_header = {}
        
        # Map clone_id to ID using validation mapping or fallback to direct check
        if 'clone_id' in column_mapping:
            csv_header[column_mapping['clone_id']] = 'ID'
        elif 'clone_id' in actual_columns:
            csv_header['clone_id'] = 'ID'
        else:
            # Check for alternative column names (fallback)
            id_alternatives = ['ID', 'id', 'Id', 'clone_ID', 'Clone_ID', 'Clone_id', 'main_name', 'Main_name', 'MAIN_NAME']
            found_id_col = None
            for alt in id_alternatives:
                if alt in actual_columns:
                    found_id_col = alt
                    csv_header[alt] = 'ID'
                    break
            
            if found_id_col is None:
                error_msg = f"Required column 'clone_id' (or alternatives) not found. "
                error_msg += f"Found columns: {', '.join(actual_columns[:20])}{'...' if len(actual_columns) > 20 else ''}"
                print(f"Error: {error_msg}")
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "invalid_file_format",
                    "message": error_msg,
                    "type": "missing_columns",
                    "missing_columns": ['clone_id'],
                    "found_columns": actual_columns,
                    "session_id": session_id
                }
                return
        
        # Map Heavy to heavy chain letter using validation mapping or fallback
        if 'Heavy' in column_mapping:
            csv_header[column_mapping['Heavy']] = heavy_letter
        elif 'Heavy' in actual_columns:
            csv_header['Heavy'] = heavy_letter
        else:
            # Check for alternative column names (fallback)
            heavy_alternatives = ['Heavy_Chain', 'heavy', 'HEAVY', 'Heavy_chain', 'H', 'Heavy_DNA', 'heavy_DNA', 'HEAVY_DNA']
            found_heavy_col = None
            for alt in heavy_alternatives:
                if alt in actual_columns:
                    found_heavy_col = alt
                    csv_header[alt] = heavy_letter
                    break
            
            if found_heavy_col is None:
                error_msg = f"Required column 'Heavy' (or alternatives) not found. "
                error_msg += f"Found columns: {', '.join(actual_columns[:20])}{'...' if len(actual_columns) > 20 else ''}"
                print(f"Error: {error_msg}")
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "invalid_file_format",
                    "message": error_msg,
                    "type": "missing_columns",
                    "missing_columns": ['Heavy'],
                    "found_columns": actual_columns,
                    "session_id": session_id
                }
                return
        
        # Map Light to light chain letter using validation mapping or fallback
        if 'Light' in column_mapping:
            csv_header[column_mapping['Light']] = light_letter
        elif 'Light' in actual_columns:
            csv_header['Light'] = light_letter
        else:
            # Check for alternative column names (fallback)
            light_alternatives = ['Light_Chain', 'light', 'LIGHT', 'Light_chain', 'L', 'Light_DNA', 'light_DNA', 'LIGHT_DNA']
            found_light_col = None
            for alt in light_alternatives:
                if alt in actual_columns:
                    found_light_col = alt
                    csv_header[alt] = light_letter
                    break
            
            if found_light_col is None:
                error_msg = f"Required column 'Light' (or alternatives) not found. "
                error_msg += f"Found columns: {', '.join(actual_columns[:20])}{'...' if len(actual_columns) > 20 else ''}"
                print(f"Error: {error_msg}")
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "invalid_file_format",
                    "message": error_msg,
                    "type": "missing_columns",
                    "missing_columns": ['Light'],
                    "found_columns": actual_columns,
                    "session_id": session_id
                }
                return
        
        # If file has Antigen column, add to header mapping
        if 'Antigen' in actual_columns:
            csv_header['Antigen'] = antigen_letter
        elif antigen_name in actual_columns:
            csv_header[antigen_name] = antigen_letter
        
        # Build header parameter to pass to convert2afformat (should be list of mapped values)
        header = list(csv_header.values())
        
        print(f"Using header mapping: {csv_header}")
        print(f"Passing header: {header}")
        print(f"Actual file columns: {', '.join(actual_columns[:20])}{'...' if len(actual_columns) > 20 else ''}")
        
        # Set csv_run_af3 global variables
        csv_run_af3.ANTIG_LETTER = antigen_letter
        csv_run_af3.HEAVY_LETTER = heavy_letter
        csv_run_af3.LIGHT_LETTER = light_letter
        csv_run_af3.ANTIGEN_NAME = antigen_name
        
        # Set HEADER variable to match csv_run_af3 expected format
        csv_run_af3.HEADER = csv_header
        
        # Set ROOT_DIR and other necessary variables
        csv_run_af3.ROOT_DIR = root_dir
        csv_run_af3.PDB_DIR = os.path.join(root_dir, 'af3_inputs', 'pdb_files')
        csv_run_af3.CSV_DIR = csv_dir
        csv_run_af3.JSON_DIR = os.path.join(root_dir, 'json_files')
        csv_run_af3.MODEL_DIR = os.path.join(root_dir, 'af3_model')
        csv_run_af3.OUT_DIR = os.path.join(root_dir, 'af3_outputs')
        csv_run_af3.PUBLIC_DATA_DIR = os.path.join(root_dir, 'public_databases')
        
        # Set JSON file directory and output directory (before calling convert2afformat)
        json_dir = os.path.join(root_dir, 'json_files', output_name)
        output_dir = os.path.join(root_dir, 'af3_outputs', output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Call convert2afformat function
        try:
            # Verify input file exists and is readable before calling convert2afformat
            if not os.path.exists(target_path):
                    error_msg = f"Input file not found at expected path: {target_path}"
                    print(f"Error: {error_msg}")
                    yield {
                        "type": "error",
                        "status": "error",
                        "error_type": "preprocessing_failed",
                        "message": error_msg,
                        "type": "file_not_found",
                        "session_id": session_id
                    }
                    return
            
            # Detect actual file type first (may differ from extension)
            detected_type = detect_file_type(target_path)
            file_ext = os.path.splitext(target_path)[1].lower()
            
            # Verify required columns exist in the file (use robust reading like validate_input_file)
            df_check = None
            read_error_msg = None
            
            # Prioritize detected type over file extension
            # Note: suffix includes the dot (e.g., '.csv', '.xlsx')
            suffix_no_dot = suffix.lstrip('.')
            if detected_type == 'xlsx' or (suffix_no_dot in ['xlsx', 'xls'] and detected_type != 'csv'):
                # File is actually Excel, even if extension suggests CSV
                try:
                    df_check = pd.read_excel(target_path, engine='openpyxl')
                except:
                    try:
                        df_check = pd.read_excel(target_path, engine='xlrd')
                    except Exception as e:
                        read_error_msg = f"Failed to read Excel file: {str(e)}. Detected type: {detected_type}, Extension: {file_ext}"
            elif detected_type == 'xls':
                try:
                    df_check = pd.read_excel(target_path, engine='xlrd')
                except Exception as e:
                    read_error_msg = f"Failed to read Excel file (XLS): {str(e)}. Detected type: {detected_type}, Extension: {file_ext}"
            elif detected_type == 'unknown':
                # Unknown binary - try Excel first
                try:
                    df_check = pd.read_excel(target_path, engine='openpyxl')
                except:
                    try:
                        df_check = pd.read_excel(target_path, engine='xlrd')
                    except:
                        # Last resort: try CSV
                        df_check = read_csv_robust(target_path)
                        if df_check is None:
                            read_error_msg = f"File appears to be binary (detected type: {detected_type}) but could not be read as Excel or CSV. Extension: {file_ext}"
            else:
                # Try CSV reading
                df_check = read_csv_robust(target_path)
                if df_check is None:
                    read_error_msg = f"Failed to read CSV file with any strategy. Detected type: {detected_type}, Extension: {file_ext}"
                else:
                    # Validate columns are not binary garbage
                    columns = df_check.columns.tolist()
                    garbage_columns = []
                    for col in columns:
                        if isinstance(col, str):
                            # Count true binary control characters (excluding common whitespace)
                            binary_chars = sum(1 for c in col if (ord(c) < 9 or (9 < ord(c) < 32)) and c not in ['\t', '\n', '\r', ' '])
                            has_null = '\x00' in col
                            # If more than 30% are binary control chars or has NULL bytes, it's likely garbage
                            if len(col) > 0 and (has_null or (binary_chars / len(col) > 0.3)):
                                garbage_columns.append(col)
                    
                    if garbage_columns:
                        # Binary file misread as CSV - try Excel
                        df_check = None
                        try:
                            df_check = pd.read_excel(target_path, engine='openpyxl')
                        except:
                            try:
                                df_check = pd.read_excel(target_path, engine='xlrd')
                            except:
                                read_error_msg = f"File appears to be binary (garbage columns detected: {garbage_columns[:3]}), but could not be read as Excel. Detected type: {detected_type}, Extension: {file_ext}"
            
            if read_error_msg or df_check is None:
                error_msg = read_error_msg or "File could not be read"
                if detected_type in ['xlsx', 'xls', 'unknown'] and file_ext == '.csv':
                    error_msg += f" The file has .csv extension but appears to be {detected_type} format. Please check if the file was downloaded correctly."
                print(f"Error: {error_msg}")
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "preprocessing_failed",
                    "message": error_msg,
                    "type": "file_read_error",
                    "detected_type": detected_type,
                    "file_extension": file_ext,
                    "session_id": session_id
                }
                return
            missing_cols = [col for col in ['clone_id', 'Heavy', 'Light'] if col not in df_check.columns]
            if missing_cols:
                error_msg = f"Required columns missing in input file: {', '.join(missing_cols)}. Found columns: {', '.join(df_check.columns.tolist())}"
                print(f"Error: {error_msg}")
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "preprocessing_failed",
                    "message": error_msg,
                    "type": "missing_columns",
                    "found_columns": df_check.columns.tolist(),
                    "missing_columns": missing_cols,
                    "session_id": session_id
                }
                return
            
            # Check for empty values in required columns
            empty_rows = []
            for idx, row in df_check.iterrows():
                if pd.isna(row.get('clone_id')) or pd.isna(row.get('Heavy')) or pd.isna(row.get('Light')):
                    empty_rows.append(idx + 1)  # 1-indexed for user
            if empty_rows:
                error_msg = f"Found empty values in required columns at rows: {', '.join(map(str, empty_rows[:10]))}{'...' if len(empty_rows) > 10 else ''}"
                print(f"Warning: {error_msg}")
            
            print(f"Input file verified: {len(df_check)} rows, columns: {', '.join(df_check.columns.tolist())}")
            
            # Verify file path matches what convert2afformat expects
            # convert2afformat expects: CSV_DIR/input_name+suffix
            expected_path_by_convert = os.path.join(csv_dir, input_name + suffix)
            if not os.path.exists(expected_path_by_convert):
                # If the expected path doesn't exist, check if target_path exists and matches
                if os.path.exists(target_path):
                    # If target_path has a different name, we need to ensure convert2afformat can find it
                    # The issue might be that input_name doesn't match the actual file name
                    actual_input_name = os.path.splitext(input_file_name)[0]
                    if actual_input_name != input_name:
                        print(f"Warning: input_name mismatch. Expected: {input_name}, Actual: {actual_input_name}")
                        # Use actual file name for convert2afformat
                        input_name = actual_input_name
                        expected_path_by_convert = os.path.join(csv_dir, input_name + suffix)
                        if not os.path.exists(expected_path_by_convert):
                            # Create symlink or copy to expected name
                            import shutil
                            shutil.copy2(target_path, expected_path_by_convert)
                            print(f"Copied file to expected path: {expected_path_by_convert}")
                else:
                    error_msg = f"Input file not found at target path: {target_path}, and expected path by convert2afformat: {expected_path_by_convert}"
                    print(f"Error: {error_msg}")
                    yield {
                        "type": "error",
                        "status": "error",
                        "error_type": "preprocessing_failed",
                        "message": error_msg,
                        "type": "file_not_found",
                        "session_id": session_id
                    }
                    return
            
            print(f"Calling convert2afformat with: suffix={suffix}, input_name={input_name}, output_name={output_name}, header={header}")
            print(f"Expected input file path: {os.path.join(csv_dir, input_name + suffix)}")
            print(f"HEADER mapping: {csv_run_af3.HEADER}")
            
            # 发送预处理进度
            yield {
                "type": "progress",
                "data": {
                    "session_id": session_id,
                    "status": "processing",
                    "message": "Running convert2afformat preprocessing...",
                    "timestamp": time.time()
                }
            }
            
            # Call convert2afformat
            csv_run_af3.convert2afformat(
                suffix=suffix,
                input_name=input_name,
                output_name=output_name,
                header=header
            )
            print(f"convert2afformat execution completed")
            
            # Immediately check if JSON directory was created
            if not os.path.exists(json_dir):
                error_msg = f"JSON directory was not created: {json_dir}. convert2afformat completed without errors but did not create output directory. "
                error_msg += f"Please check: 1) Input file format is correct, 2) Required columns (clone_id, Heavy, Light) exist and have valid data, "
                error_msg += f"3) Column mapping is correct (current mapping: {csv_header}). "
                error_msg += f"Input file has {len(df_check)} rows with columns: {', '.join(df_check.columns.tolist()[:10])}"
                print(f"Error: {error_msg}")
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "preprocessing_failed",
                    "message": error_msg,
                    "type": "missing_directory",
                    "input_file": target_path,
                    "expected_json_dir": json_dir,
                    "column_mapping": csv_header,
                    "session_id": session_id
                }
                return
            
            # Check if JSON files were generated
            json_files_check = [f for f in os.listdir(json_dir) if f.endswith('.json')] if os.path.exists(json_dir) else []
            if not json_files_check:
                error_msg = f"No JSON files found in {json_dir}. convert2afformat created the directory but did not generate any JSON files. "
                error_msg += f"This may indicate that the input data format is incorrect or the column mapping is wrong. "
                error_msg += f"Current column mapping: {csv_header}. Input file has {len(df_check)} rows."
                print(f"Error: {error_msg}")
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "preprocessing_failed",
                    "message": error_msg,
                    "type": "no_json_files",
                    "json_dir": json_dir,
                    "column_mapping": csv_header,
                    "row_count": len(df_check),
                    "session_id": session_id
                }
                return
            
            print(f"✓ Preprocessing successful: {len(json_files_check)} JSON files generated")
            
        except Exception as e:
            import traceback
            error_msg = f"Input preprocessing (convert2afformat) failed: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"Error: {error_msg}")
            print(f"Traceback: {traceback_str}")
            yield {
                "type": "error",
                "status": "error",
                "error_type": "preprocessing_failed",
                "message": error_msg,
                "type": "preprocessing_error",
                "traceback": traceback_str,
                "session_id": session_id
            }
            return
        
            # Process JSON files and call AlphaFold3
        inference_successful = False
        inference_errors = []
        processed_count = 0
        failed_count = 0
        gpu_oom_errors = []
        
        try:
            # Process JSON files
            json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
            print(f"Found {len(json_files)} JSON files")
            
            total_files = len(json_files)
            
            for idx, json_file in enumerate(json_files, 1):
                # 发送进度更新
                progress_percent = (idx / total_files) * 100 if total_files > 0 else 0
                elapsed_time = time.time() - start_time
                
                yield {
                    "type": "progress",
                    "data": {
                        "session_id": session_id,
                        "status": "processing",
                        "progress_percent": round(progress_percent, 1),
                        "current": idx,
                        "total": total_files,
                        "elapsed_seconds": round(elapsed_time, 1),
                        "elapsed_minutes": round(elapsed_time / 60, 1),
                        "message": f"Processing JSON file {idx}/{total_files}: {json_file}",
                        "timestamp": time.time()
                    }
                }
                json_path = os.path.join(json_dir, json_file)
                sample_name = os.path.splitext(json_file)[0]
                output_subdir = os.path.join(output_dir, sample_name)
                
                # Check if already processed
                if not os.path.exists(output_subdir):
                    print(f"Processing JSON file: {json_path}")
                    
                    # Record directory state before execution
                    existing_dirs = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
                    
                    # Call AlphaFold3 using system command
                    af3_script = os.path.join(root_dir, 'alphafold3', 'run_alphafold.py')
                    cmd = f"CUDA_VISIBLE_DEVICES={args.gpu_device} /data_new/lht/.conda/envs/alphafold3_venv/bin/python {af3_script} --json_path={json_path} --model_dir={os.path.join(root_dir, 'af3_model')} --db_dir={os.path.join(root_dir, 'public_databases')} --gpu_device=0 --output_dir={output_dir}"
                    print(f"Executing command: {cmd}")
                    
                    # Execute AlphaFold3 inference
                    import subprocess
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode != 0:
                        # Extract key error information
                        stderr_lines = result.stderr.split('\n')
                        key_errors = []
                        
                        # Check for GPU out of memory
                        if 'RESOURCE_EXHAUSTED' in result.stderr or 'Out of memory' in result.stderr:
                            oom_msg = f"GPU out of memory for {json_file}. The GPU (device {args.gpu_device}) does not have enough memory to run AlphaFold3. Try: 1) Use a GPU with more memory, 2) Free up GPU memory by closing other processes, 3) Reduce batch size or sequence length."
                            key_errors.append(oom_msg)
                            gpu_oom_errors.append(json_file)
                        
                        # Extract other key errors (first few lines of traceback)
                        for line in stderr_lines[-20:]:  # Last 20 lines usually contain the key error
                            if any(keyword in line for keyword in ['Error:', 'Exception:', 'Traceback', 'Failed', 'ERROR']):
                                if len(key_errors) < 3:  # Limit to 3 key errors
                                    key_errors.append(line.strip())
                        
                        if key_errors:
                            error_msg = f"AlphaFold3 execution failed for {json_file}: {'; '.join(key_errors[:2])}"
                        else:
                            # Fallback to first 500 chars of stderr
                            error_msg = f"AlphaFold3 execution failed for {json_file}: {result.stderr[:500]}"
                        
                        print(f"Error: {error_msg}")
                        inference_errors.append(error_msg)
                        failed_count += 1
                    else:
                        print(f"AlphaFold3 execution successful: {json_file}")
                        
                        # Detect newly generated directories
                        current_dirs = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
                        new_dirs = current_dirs - existing_dirs
                        
                        # Find new directories related to current sample
                        sample_related_dirs = [d for d in new_dirs if sample_name.lower() in d.lower()]
                        if sample_related_dirs:
                            # Select the newest directory (usually only one)
                            actual_output_dir = sorted(sample_related_dirs)[-1]
                            actual_output_path = os.path.join(output_dir, actual_output_dir)
                            
                            # Store mapping of actual generated directory paths
                            mapping_file = os.path.join(output_dir, '.af3_output_mapping.json')
                            mapping = {}
                            if os.path.exists(mapping_file):
                                import json
                                with open(mapping_file, 'r') as f:
                                    mapping = json.load(f)
                            
                            mapping[sample_name] = actual_output_path
                            
                            import json
                            with open(mapping_file, 'w') as f:
                                json.dump(mapping, f, indent=2)
                            
                            print(f"Recorded directory mapping: {sample_name} -> {actual_output_path}")
                            inference_successful = True
                            processed_count += 1
                        else:
                            warning_msg = f"No newly generated directory found for sample {sample_name}"
                            print(f"Warning: {warning_msg}")
                            inference_errors.append(warning_msg)
                            failed_count += 1
                else:
                    print(f"Skipping already processed file: {json_file}")
                    processed_count += 1
                    inference_successful = True
        
        except Exception as e:
            error_msg = f"Error occurred during JSON processing: {str(e)}"
            print(f"Error: {error_msg}")
            inference_errors.append(error_msg)
            failed_count += 1
        
        # Check if any inference was successful
        if not inference_successful:
            # Build comprehensive error message
            error_msg = f"AlphaFold3 inference failed for all {len(json_files)} samples. "
            
            # Add specific guidance based on error type
            if gpu_oom_errors:
                error_msg += f"GPU out of memory errors detected for {len(gpu_oom_errors)} sample(s). "
                error_msg += "The GPU (device {}) does not have enough memory. ".format(args.gpu_device)
                error_msg += "Solutions: 1) Use a GPU with more memory, 2) Free up GPU memory, 3) Try a different GPU device. "
            elif inference_errors:
                # Show first 2 most relevant errors
                key_errors = [err for err in inference_errors if len(err) < 500][:2]
                if key_errors:
                    error_msg += f"Key errors: {'; '.join(key_errors)}. "
            
            error_msg += "Please check the input data and AlphaFold3 environment setup."
            print(f"Error: {error_msg}")
            yield {
                "type": "error",
                "status": "error",
                "error_type": "inference_failed",
                "message": error_msg,
                "processed": processed_count,
                "failed": failed_count,
                "total": len(json_files),
                "gpu_oom_count": len(gpu_oom_errors) if gpu_oom_errors else 0,
                "session_id": session_id
            }
            return
        
        # Unified file collection and conversion logic
        # Only proceed if at least one inference was successful
        print(f"Inference completed: {processed_count} successful, {failed_count} failed")
        
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
                "message": f"AlphaFold3 prediction completed: {processed_count} successful, {failed_count} failed",
                "timestamp": time.time()
            }
        }
        
        # 收集和转换文件
        result = collect_and_convert_files(output_dir, json_dir)
        
        # 返回最终结果
        yield {
            "type": "result",
            "status": "success" if "error" not in result else "error",
            "session_id": session_id,
            "result": result,
            "processing_time_ms": total_time * 1000
        }
        
    except ImportError as e:
        error_msg = f"Failed to import csv_run_af3 module: {str(e)}"
        print(f"Error: {error_msg}")
        yield {
            "type": "error",
            "status": "error",
            "error_type": "import_error",
            "message": error_msg,
            "session_id": session_id
        }
        return
    
    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        print(f"Error: {error_msg}")
        yield {
            "type": "error",
            "status": "error",
            "error_type": "file_not_found",
            "message": error_msg,
            "session_id": session_id
        }
        return
    
    except pd.errors.EmptyDataError:
        error_msg = "Input file is empty or format is incorrect"
        print(f"Error: {error_msg}")
        yield {
            "type": "error",
            "status": "error",
            "error_type": "empty_data",
            "message": error_msg,
            "session_id": session_id
        }
        return
    
    except pd.errors.ParserError as e:
        error_msg = f"File parsing error: {str(e)}"
        print(f"Error: {error_msg}")
        yield {
            "type": "error",
            "status": "error",
            "error_type": "parser_error",
            "message": error_msg,
            "session_id": session_id
        }
        return
    
    except Exception as e:
        error_msg = f"Error occurred during processing: {str(e)}"
        print(f"Error: {error_msg}")
        yield {
            "type": "error",
            "status": "error",
            "error_type": "general_error",
            "message": error_msg,
            "session_id": session_id
        }
        return


def collect_and_convert_files(output_dir, json_dir):
    """
    Collect and convert AlphaFold3 output files
    Use stored directory mapping to directly locate files, avoiding complex matching logic
    """
    import json
    
    try:
        # Check if output directory exists
        if not os.path.exists(output_dir):
            error_msg = f"Output directory does not exist: {output_dir}. AlphaFold3 inference may not have completed successfully."
            print(f"Error: {error_msg}")
            return {"error": "no_output_directory", "message": error_msg, "type": "missing_directory"}
        
        # Check if mapping file exists
        mapping_file = os.path.join(output_dir, '.af3_output_mapping.json')
        if not os.path.exists(mapping_file):
            # Check if there are any output directories (maybe inference completed but mapping wasn't created)
            output_contents = os.listdir(output_dir) if os.path.exists(output_dir) else []
            if not output_contents:
                error_msg = f"Directory mapping file not found and output directory is empty: {output_dir}. AlphaFold3 inference may have failed. Please check the inference logs and ensure the inference completed successfully before collecting files."
            else:
                error_msg = f"Directory mapping file not found: {mapping_file}. AlphaFold3 inference may not have completed successfully, or the mapping file was not created. Found {len(output_contents)} items in output directory."
            print(f"Error: {error_msg}")
            return {"error": "no_mapping_file", "message": error_msg, "type": "missing_mapping"}
        
        # Read directory mapping
        try:
            with open(mapping_file, 'r') as f:
                directory_mapping = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse mapping file {mapping_file}: {str(e)}. The file may be corrupted."
            print(f"Error: {error_msg}")
            return {"error": "invalid_mapping_file", "message": error_msg, "type": "corrupted_mapping"}
        
        # Get JSON file list
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        if not json_files:
            return {"error": "no_json_files", "message": f"No JSON files found in {json_dir}"}
        
        pdb_files = []
        
        for json_file in json_files:
            sample_name = os.path.splitext(json_file)[0]
            
            # Get actual output directory from mapping
            if sample_name in directory_mapping:
                actual_output_path = directory_mapping[sample_name]
                print(f"Using mapped directory: {sample_name} -> {actual_output_path}")
            else:
                # If not in mapping, try to find traditional non-timestamped directory
                fallback_dir = os.path.join(output_dir, sample_name)
                if os.path.exists(fallback_dir):
                    actual_output_path = fallback_dir
                    print(f"Using traditional directory: {sample_name} -> {actual_output_path}")
                else:
                    print(f"Warning: Output directory not found for sample {sample_name}")
                    continue
            
            # Find CIF files
            cif_files = []
            if os.path.exists(actual_output_path):
                for file in os.listdir(actual_output_path):
                    if file.endswith('.cif'):
                        cif_files.append(os.path.join(actual_output_path, file))
            
            if not cif_files:
                print(f"Warning: No CIF files found in {actual_output_path}")
                continue
            
            # Convert CIF files to PDB format
            for cif_file in cif_files:
                try:
                    base_name = os.path.splitext(os.path.basename(cif_file))[0]
                    pdb_file_path = os.path.join(actual_output_path, f"{base_name}.pdb")
                    
                    # If PDB file already exists, add directly to list
                    if os.path.exists(pdb_file_path):
                        pdb_files.append(pdb_file_path)
                        print(f"PDB file already exists: {pdb_file_path}")
                        continue
                    
                    # Use Bio.PDB to convert CIF to PDB
                    parser = MMCIFParser()
                    structure = parser.get_structure("structure_id", cif_file)
                    
                    io = PDBIO()
                    io.set_structure(structure)
                    io.save(pdb_file_path)
                    
                    pdb_files.append(pdb_file_path)
                    print(f"Successfully converted: {cif_file} -> {pdb_file_path}")
                except Exception as e:
                    print(f"Error converting CIF file {cif_file}: {str(e)}")
        
        if not pdb_files:
            return {"error": "no_pdb_files", "message": "No PDB files generated"}
        
        return {"pdb_files": pdb_files, "type": "success"}
        
    except Exception as e:
        return {"error": "collection_failed", "message": f"File collection failed: {str(e)}"}

@asynccontextmanager
async def fdg_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Handle server startup and shutdown"""
    print("Alphafold3 MCP Server is initializing...")
    try:
        yield {"initialized": True}
    finally:
        print("Alphafold3 MCP Server is shutting down...")

# Set lifecycle
mcp.lifespan = fdg_lifespan

if __name__ == "__main__":
    print("Starting Alphafold3 MCP server...")
    # Set network parameters
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8084
    
    # Start using SSE mode
    mcp.run(transport="sse")
