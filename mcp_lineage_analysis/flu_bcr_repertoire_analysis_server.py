"""
Lineage Analysis Server

Integrated analysis pipeline for B cell receptor lineage analysis.
Encompasses VDJ annotation, clonal clustering, lineage tracing, and functional characterization.

Core capabilities:
1. Single-cell metadata extraction (UMAP coordinates, cell type annotations)
2. Multi-omics data integration (scBCR-seq + bulk BCR-seq)
3. VDJ annotation and clonal clustering (ChangeO + ANARCI)
4. Functional assay integration (binding/neutralization experiments)
5. Machine learning prediction integration
6. Comprehensive repertoire visualization
7. Phylogenetic lineage tree data preparation
8. B cell clonal evolution and lineage reconstruction
"""

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncIterator
import os
import subprocess
import json
import tempfile
import traceback
from pathlib import Path
from collections import Counter
import time
import uuid
import asyncio
import threading
import logging
import inspect
from mcp.types import TextContent, CallToolRequest, ServerResult, CallToolResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('Lineage_Analysis_MCP')

# Create MCP server
mcp = FastMCP("Lineage Analysis Server")

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
                service_id = "lineage_analysis"
                
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

# Base directory configuration
FLU_BASE_DIR = "/data_new/workspace/antibody_gen/flu"
TEMP_DIR = os.path.join(FLU_BASE_DIR, "temp")
INPUT_DIR = os.path.join(FLU_BASE_DIR, "input")
OUTPUT_DIR = os.path.join(FLU_BASE_DIR, "output")

# Ensure directories exist
for dir_path in [TEMP_DIR, INPUT_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def run_r_script(script_path: str, args: dict = None) -> dict:
    """Execute R script and return results"""
    try:
        # Build R command
        r_cmd = ["Rscript", script_path]
        
        if args:
            # Pass arguments as JSON to R script
            args_json = json.dumps(args)
            r_cmd.extend(["--args", args_json])
        
        result = subprocess.run(
            r_cmd,
            capture_output=True,
            text=True,
            cwd=FLU_BASE_DIR
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr,
                "output": result.stdout
            }
        
        return {
            "success": True,
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def run_python_script(script_path: str, args: dict = None) -> dict:
    """Execute Python script and return results"""
    try:
        import sys
        sys.path.insert(0, FLU_BASE_DIR)
        
        # Read and execute Python script
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Create execution environment
        exec_globals = {
            "__name__": "__main__",
            "__file__": script_path,
            "os": os,
            "sys": sys,
            "args": args or {}
        }
        
        exec_globals.update(__builtins__ if isinstance(__builtins__, dict) else {__builtins__: __builtins__})
        
        # Execute code
        exec(code, exec_globals)
        
        return {
            "success": True,
            "output": "Script executed successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ==================== Tool 1: Extract Seurat UMAP Metadata ====================

class ExtractSeuratUmapMetadataArgs(BaseModel):
    """Arguments for extracting UMAP metadata from Seurat objects"""
    
    rds_file_path: Optional[str] = Field(
        default=None,
        description="Path to RDS file containing single-cell data",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["rds"],
            "placeholder": "Enter RDS file path or upload file",
            "help_text": "RDS file of Seurat object",
            "demo_paths": "../input/rds/20240923_flu_B_annotation.rds"
        }
    )
    
    output_path: Optional[str] = Field(
        default=None,
        description="Output CSV file path",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": False,
            "placeholder": "Output file path, defaults to temp/umap_coordinates.csv"
        }
    )
    
    focus_genes: List[str] = Field(
        default_factory=lambda: ["DUSP4","ZBTB38","LGMN","KPNA2","LRMP",
                                 "SSPN","MRPL36","IGHG1","ELK3",
                                 "FCRL5","ITGAX"],
        description="List of genes of interest for expression extraction"
    )


@mcp.tool()
async def extract_seurat_umap_metadata(args: ExtractSeuratUmapMetadataArgs):
    """Extract UMAP coordinates and cellular metadata from Seurat RDS files
    
    This tool corresponds to the functionality in notebook 1.get_cell_location(R).ipynb.
    Extracts from Seurat object RDS files:
    - UMAP coordinates (dimensionality-reduced cell positions)
    - Cell type annotation information
    - Expression values for genes of interest
    
    Bioinformatics domains: ["single-cell", "dimensionality reduction", "UMAP", "metadata extraction"]
    Input data: ["Seurat RDS files", "single-cell RNA-seq data"]
    Output results: ["UMAP coordinates CSV", "cell type annotations", "gene expression values"]
    
    Args:
        args: ExtractSeuratUmapMetadataArgs - Extraction parameters
        
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
            "message": "Starting Seurat UMAP metadata extraction",
            "timestamp": time.time()
        }
    }
    
    try:
        # Determine input file path
        if args.rds_file_path:
            rds_path = args.rds_file_path
        else:
            rds_path = os.path.join(INPUT_DIR, "rds", "20240923_flu_B_annotation.rds")
        
        if not os.path.exists(rds_path):
            yield {
                "type": "error",
                "status": "error",
                "error_type": "file_not_found",
                "message": f"RDS file not found: {rds_path}",
                "session_id": session_id
            }
            return
        
        # Determine output path
        if args.output_path:
            output_path = args.output_path
        else:
            output_path = os.path.join(TEMP_DIR, "umap_coordinates.csv")
        
        # Create temporary R script
        r_script = f"""
library(Seurat)
library(ggplot2)

# Load RDS file
sc_data <- readRDS("{rds_path}")

# Extract UMAP coordinates
umap_coords <- sc_data@reductions$umap@cell.embeddings

# Extract cell type information
X <- sc_data@meta.data[,c("main_name","annotation_final")]
Y <- data.frame(umap_coords)

# Extract expression values for genes of interest
focus_gene <- c({', '.join(f'"{gene}"' for gene in args.focus_genes)})
Z <- t(as.data.frame(GetAssayData(sc_data, assay = "RNA", layer = "data")[focus_gene,]))

# Merge data
XYZ <- cbind(X, Y, Z)
colnames(XYZ) <- c("main_name","celltype","umap_1","umap_2", focus_gene)

# Save results
write.csv(XYZ, file = "{output_path}", quote = FALSE, row.names = FALSE)
"""
        
        # Save temporary R script
        temp_r_script = os.path.join(TEMP_DIR, "temp_get_cell_location.R")
        with open(temp_r_script, 'w', encoding='utf-8') as f:
            f.write(r_script)
        
        # Execute R script
        result = run_r_script(temp_r_script)
        
        if result["success"]:
            # Clean up temporary files
            if os.path.exists(temp_r_script):
                os.remove(temp_r_script)
            
            if os.path.exists(output_path):
                total_time = time.time() - start_time
                yield {
                    "type": "progress",
                    "data": {
                        "session_id": session_id,
                        "status": "completed",
                        "progress_percent": 100.0,
                        "elapsed_seconds": round(total_time, 1),
                        "message": "Extraction completed",
                        "timestamp": time.time()
                    }
                }
                
                yield {
                    "type": "result",
                    "status": "success",
                    "session_id": session_id,
                    "message": f"Successfully extracted cellular metadata, results saved to: {output_path}",
                    "output_path": output_path,
                    "processing_time_ms": total_time * 1000
                }
            else:
                yield {
                    "type": "error",
                    "status": "error",
                    "error_type": "output_not_found",
                    "message": f"Script executed successfully, but output file does not exist: {output_path}\nOutput: {result.get('output', '')}",
                    "session_id": session_id
                }
                return
        else:
            yield {
                "type": "error",
                "status": "error",
                "error_type": "r_script_failed",
                "message": f"R script execution failed: {result.get('error', 'Unknown error')}\nOutput: {result.get('output', '')}",
                "session_id": session_id
            }
            return
            
    except Exception as e:
        yield {
            "type": "error",
            "status": "error",
            "error_type": "execution_error",
            "message": f"Error occurred during execution: {str(e)}",
            "session_id": session_id
        }
        return


# ==================== Tool 2: Integrate Single-cell and Bulk BCR Data ====================

class IntegrateScbcrBulkBcrDataArgs(BaseModel):
    """Arguments for integrating single-cell and bulk BCR data"""
    
    sc_rna_csv_path: Optional[str] = Field(
        default=None,
        description="Path to single-cell RNA data CSV file",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["csv"]
        }
    )
    
    bulk_raw_data_dir: Optional[str] = Field(
        default=None,
        description="Directory containing bulk raw data",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": False
        }
    )
    
    umap_coordinates_path: Optional[str] = Field(
        default=None,
        description="Path to UMAP coordinates file (from extract_seurat_umap_metadata)",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["csv"]
        }
    )
    
    output_path: Optional[str] = Field(
        default=None,
        description="Output file path",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": False
        }
    )


@mcp.tool()
async def integrate_scbcr_bulk_bcr_data(args: IntegrateScbcrBulkBcrDataArgs):
    """Integrate single-cell BCR and bulk BCR sequencing data
    
    This tool corresponds to the functionality in notebook 2.flu_dataset_collect.ipynb.
    Performs the following data integration steps:
    - Load single-cell RNA-seq BCR data
    - Parse FASTQ files from bulk BCR sequencing
    - Merge single-cell and bulk BCR sequence data
    - Append UMAP coordinates and cell type annotations
    - Standardize timepoint and cell type information
    
    Bioinformatics domains: ["BCR repertoire", "data integration", "single-cell", "bulk sequencing"]
    Input data: ["Single-cell BCR CSV", "Bulk BCR FASTQ files", "UMAP coordinates"]
    Output results: ["Integrated BCR dataset CSV", "merged sequence data"]
    
    Args:
        args: IntegrateScbcrBulkBcrDataArgs - Integration parameters
        
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
            "message": "Starting single-cell and bulk BCR data integration",
            "timestamp": time.time()
        }
    }
    
    try:
        import pandas as pd
        import numpy as np
        
        # Determine input paths
        sc_rna_path = args.sc_rna_csv_path or os.path.join(INPUT_DIR, "scBCR", "flu_raw_data", "scRNA.csv")
        bulk_dir = args.bulk_raw_data_dir or os.path.join(INPUT_DIR, "scBCR", "flu_raw_data", "bulk_raw_data")
        umap_path = args.umap_coordinates_path or os.path.join(TEMP_DIR, "umap_coordinates.csv")
        
        # Determine output path
        output_path = args.output_path or os.path.join(TEMP_DIR, "all_data.csv")
        
        # Load single-cell data
        data_v1 = pd.read_csv(sc_rna_path)
        if "Unnamed: 0" in data_v1.columns:
            del data_v1["Unnamed: 0"]
        
        # Extract Timepoint from main_name with error handling
        def extract_timepoint(name):
            """Extract timepoint from main_name, handling various formats"""
            if pd.isna(name) or name == '':
                return None
            try:
                name_str = str(name)
                parts = name_str.split("_")
                if len(parts) >= 2:
                    return parts[1]
                elif len(parts) == 1:
                    # If no underscore, try to extract from the name itself
                    # Return the original name or a default value
                    return name_str
                else:
                    return None
            except (AttributeError, IndexError, TypeError) as e:
                logger.warning(f"Failed to extract timepoint from '{name}': {str(e)}")
                return None
        
        data_v1["Timepoint"] = data_v1["main_name"].apply(extract_timepoint)
        
        # Helper dictionary (extracted from notebook)
        Helper_Dict = {
            'SRR11233621': ("321-11", "12", "d0", "PBMC"),
            'SRR11233630': ("321-11", "9", "d6", "Plasmblast"),
            'SRR11233641': ("321-05", "11", "d0", "PBMC"),
            'SRR11233652': ("321-05", "8", "d5", "Plasmblast"),
            'SRR11233663': ("321-04", "10", "d0", "PBMC"),
            'SRR11233664': ("321-04", "7", "d5", "Plasmblast")
        }
        
        # Load bulk data (FASTQ conversion)
        def q2dict(fastq_dir):
            """Convert FASTQ file to dictionary"""
            fq2fa_dict = {}
            with open(fastq_dir, 'r') as fq:
                i = 0
                for line in fq:
                    i += 1
                    if i % 4 == 1:
                        seq_name = '>' + line.replace('\n', '')[1:]
                        fq2fa_dict[seq_name] = ''
                    elif i % 4 == 2:
                        seq = line.replace('\n', '')
                        fq2fa_dict[seq_name] = seq
            return fq2fa_dict
        
        Total_data = pd.DataFrame()
        if os.path.exists(bulk_dir):
            for i in os.listdir(bulk_dir):
                if "SRR" not in i or Helper_Dict.get(i, ("", "", "", ""))[1] not in ["8", "11"]:
                    continue
                fastq_files = [os.path.join(bulk_dir, i, j) for j in os.listdir(os.path.join(bulk_dir, i)) if ".fastq" in j]
                if not fastq_files:
                    continue
                fastq = fastq_files[0]
                D = q2dict(fastq)
                Data = pd.DataFrame()
                Data["main_name"] = list(D.keys())
                Data["Heavy_DNA"] = list(D.values())
                Data["Timepoint"] = Helper_Dict[i][2]
                Data["('celltype', '')"] = Helper_Dict[i][3]
                Total_data = pd.concat([Total_data, Data])
        
        # Merge single-cell and bulk data
        if not Total_data.empty:
            data_v2 = pd.concat([data_v1, Total_data]).reset_index()
            if "index" in data_v2.columns:
                del data_v2["index"]
        else:
            data_v2 = data_v1.copy()
        
        # Append UMAP coordinates
        if os.path.exists(umap_path):
            data_locate = pd.read_csv(umap_path)
            data_locate = data_locate.rename(columns={"umap_1": "locate_x", "umap_2": "locate_y"})
            data_v2 = pd.merge(data_v2, data_locate, on="main_name", how="left")
        
        # Save results
        data_v2.to_csv(output_path, index=False)
        
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
            "message": f"Successfully integrated datasets, results saved to: {output_path}\nData shape: {data_v2.shape}",
            "output_path": output_path,
            "processing_time_ms": total_time * 1000
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "status": "error",
            "error_type": "execution_error",
            "message": f"Error occurred during execution: {str(e)}\n{traceback.format_exc()}",
            "session_id": session_id
        }
        return


# ==================== Tool 3: BCR Clonal Clustering and Feature Extraction ====================

class BcrClonalClusteringAndFeatureExtractionArgs(BaseModel):
    """Arguments for BCR clonal clustering and antibody feature extraction"""
    
    input_data_path: Optional[str] = Field(
        default=None,
        description="Path to input data CSV file",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": True,
            "support_file_types": ["csv"]
        }
    )
    
    changeo_helper_dir: Optional[str] = Field(
        default=None,
        description="ChangeO toolkit directory",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": False
        }
    )
    
    anarci_dir: Optional[str] = Field(
        default=None,
        description="ANARCI toolkit directory",
        json_schema_extra={
            "ui_type": "file_input",
            "support_upload": False
        }
    )
    
    output_path: Optional[str] = Field(
        default=None,
        description="Output file path"
    )


# @mcp.tool()
def bcr_clonal_clustering_and_feature_extraction(args: BcrClonalClusteringAndFeatureExtractionArgs) -> str:
    """BCR clonal clustering and antibody feature extraction
    
    This tool corresponds to the functionality in notebook 3.ChangeO+ANARCI.ipynb.
    Performs the following analytical workflow:
    
    **Step 1: ChangeO Clonal Clustering**
    - IgBLAST V(D)J gene alignment
    - MakeDb database construction
    - DefineClones clonal definition (based on V gene and CDR3 sequence similarity)
    - CreateGermlines germline sequence reconstruction
    
    **Step 2: ANARCI Antibody Feature Extraction**
    - V/D/J gene usage frequencies
    - CDR1/2/3 region sequences
    - Somatic hypermutation (SHM) statistics
    - IMGT numbering and structural features
    
    Bioinformatics domains: ["BCR repertoire", "clonal clustering", "antibody annotation", "germline reconstruction"]
    Input data: ["BCR sequences", "V(D)J segments", "germline databases"]
    Output results: ["Clonal assignments", "CDR sequences", "SHM statistics", "gene usage"]
    
    Args:
        args: BcrClonalClusteringAndFeatureExtractionArgs - Analysis parameters
        
    Returns:
        Path to CSV file containing clonal information and antibody features
    """
    try:
        # This tool requires external ChangeO and ANARCI toolkits
        # Due to complex command-line tool invocations, guidance information is returned here
        return """ChangeO + ANARCI Analysis Tool
    
Note: This tool requires execution in a WSL environment and cannot run on Windows.
Required dependencies:
1. ChangeO toolkit (igblast, MakeDb, DefineClones, CreateGermlines)
2. ANARCI tool
3. Relevant reference databases

It is recommended to use the original notebook 3.ChangeO+ANARCI.ipynb to perform this analysis.
Alternatively, extract the relevant code as a standalone Python script for invocation.

Please let me know if you need assistance extracting the code into a callable script."""
        
    except Exception as e:
        return f"Error occurred during execution: {str(e)}"


# ==================== Tool 4: Integrate Binding and Neutralization Experiments ====================

class IntegrateBindingNeutralizationExperimentsArgs(BaseModel):
    """Arguments for integrating binding and neutralization experimental data"""
    
    first_experiment_path: Optional[str] = Field(
        default=None,
        description="Path to first batch experimental data Excel file"
    )
    
    second_experiment_path: Optional[str] = Field(
        default=None,
        description="Path to second batch experimental data Excel file"
    )
    
    output_path: Optional[str] = Field(
        default=None,
        description="Output CSV file path"
    )


@mcp.tool()
async def integrate_binding_neutralization_experiments(args: IntegrateBindingNeutralizationExperimentsArgs):
    """Integrate antibody binding and neutralization experimental measurements
    
    This tool corresponds to the functionality in notebook 3.5.clone_result.ipynb.
    Performs the following data processing steps:
    
    **Experimental Data Standardization**
    - Load two batches of antibody functional assay data
    - Apply thresholds to convert continuous measurements into binary labels (binding+/-, neutralization+/-)
    - Process binding and neutralization data for multiple influenza strains (H1N1, H3N2)
    - Standardize antibody nomenclature and batch information
    
    **Data Integration**
    - Merge replicate measurements from two experimental batches
    - Prioritize results from more recent batches
    - Handle missing values and conflicting data
    
    Bioinformatics domains: ["antibody characterization", "functional assays", "data integration"]
    Input data: ["Binding assay Excel", "Neutralization assay Excel", "antibody annotations"]
    Output results: ["Standardized binding/neutralization results CSV", "binary classifications"]
    
    Args:
        args: IntegrateBindingNeutralizationExperimentsArgs - Processing parameters
        
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
            "message": "Starting binding and neutralization experiments integration",
            "timestamp": time.time()
        }
    }
    
    try:
        import pandas as pd
        import numpy as np
        
        # Determine file paths
        first_path = args.first_experiment_path or os.path.join(INPUT_DIR, "raw_doc", "first-time_Inf", "flu_simple(origin_flu-binding_neutralizations).xlsx")
        second_path = args.second_experiment_path or os.path.join(INPUT_DIR, "raw_doc", "second-time_Inf", "flu_second_simple.xlsx")
        output_path = args.output_path or os.path.join(TEMP_DIR, "0220_Flu_cAb.csv")
        
        # Load first batch experimental data
        flu_simple = pd.read_excel(first_path).rename(columns={"Name": "mAb"})
        flu_simple = flu_simple.dropna(subset=["mAb"])
        
        # Define thresholds and rename columns
        D1 = {
            'LPS': (0.069, "LPS(poly)"),
            'dsDNA': (0.187, "dsDNA(poly)"),
            'H1N1 Michigan Monomer(neat)': (0.13, "H1N1_Michigan(bind)(experiment)"),
            'H1N1 trimer (Victoria)': (0.11, "H1N1_Victoria(bind)(experiment)"),
            'H1N1 trimer (Wisconsin)': (0.12, "H1N1_Wisconsin(bind)(experiment)"),
            'H3N2 monomer (neat)': (0.11, "H3N2_Singapore(bind)(experiment)"),
            'Neu H1N1': ("+", "H1N1_Jiangsu(neu)(experiment)"),
        }
        
        for origin_name in D1.keys():
            limitation, new_name = D1[origin_name]
            if type(limitation) == str:
                flu_simple[origin_name] = flu_simple[origin_name].apply(lambda x: 1 if x == limitation else 0)
            else:
                flu_simple[origin_name] = flu_simple[origin_name].apply(
                    lambda x: "nan" if pd.isna(x) else (1 if x > limitation else 0)
                )
            flu_simple = flu_simple.rename(columns={origin_name: new_name})
        
        flu_simple["mAb"] = flu_simple["mAb"].apply(lambda x: x.replace("Inf", "QIV"))
        
        # Load second batch experimental data
        flu_simple2 = pd.read_excel(second_path).rename(columns={"Name": "mAb"})
        flu_simple2 = flu_simple2.dropna(subset=["mAb"])
        
        normal_limit = 0.125
        D2 = {
            'A/Michigan/45/2015 Monomer(20)': (normal_limit, "H1N1_Michigan(bind)(experiment)"),
            'H1N1 Victoria Trimer(20)': (normal_limit, "H1N1_Victoria(bind)(experiment)"),
            'H1N1 Wisconsin Trimer(20)': (normal_limit, "H1N1_Wisconsin(bind)(experiment)"),
            'H3N2 A/Singapore/INF/MH-16-0019/2016(20)': (normal_limit, "H3N2_Singapore(bind)(experiment)"),
            'A/Puerto Rico/8/34 Monomer(20)': (normal_limit, "H1N1_PuertoRico(bind)(experiment)"),
            'H3N2 A/Hong Kong/45/2019 Trimer(20)': (normal_limit, "H3N2_HongKong(bind)(experiment)"),
            'H1N1/Jiangsu/2018': (75, "H1N1_Jiangsu(neu)(experiment)"),
            'H1N1/FM1': (75, "H1N1_FortMonmouth(neu)(experiment)"),
            'H1N1/A/Michigan/45/2015 ': (75, "H1N1_Michigan(neu)(experiment)"),
            'H1N1/CA07': (75, "H1N1_California(neu)(experiment)"),
            'H1N1/A/Puerto Rico/8/34 ': (75, "H1N1_PuertoRico(neu)(experiment)")
        }
        
        def f(x, limitation):
            if pd.isna(x):
                return "nan"
            if x == "NB":
                return 0
            if x > limitation:
                return 1
            return 0
        
        for origin_name in D2.keys():
            limitation, new_name = D2[origin_name]
            flu_simple2[origin_name] = flu_simple2[origin_name].apply(lambda x: f(x, limitation))
            flu_simple2 = flu_simple2.rename(columns={origin_name: new_name})
        
        # Filter columns
        flu_simple2 = flu_simple2[[x for x in flu_simple2.columns if ("(experiment)" in x) or (x in ['mAb', 'main_name', 'Heavy', 'Light'])]]
        flu_simple2 = flu_simple2.loc[flu_simple2["mAb"].apply(lambda x: "Negative" not in str(x))]
        
        # ID mapping
        ID = "1,3,4,5,8,10,14,21,22,23,25,26,27,28,29,34,36,39,41,42,44,45,47,48,49,51,52,54,55,56,57,60,62,64,65,66,68,69,70,74,75,76,78,80,82,85,86,87,95,96,104,115,126,132"
        ID_list = ID.split(",")
        D_mapping = {}
        for i in range(0, len(ID_list)):
            D_mapping[f"Inf-2-{i+1}"] = f"Inf-{ID_list[i]}"
        
        flu_simple2["mAb"] = flu_simple2["mAb"].apply(lambda x: D_mapping.get(x, x.replace("Inf", "QIV")))
        
        # Merge two experimental batches
        flu_simple["batch"] = 1
        flu_simple2["batch"] = 2
        flu_clone = pd.concat([flu_simple, flu_simple2]).fillna("nan")
        
        # Handle duplicates (prioritize batch 2 results)
        temp = flu_clone.melt(id_vars=["mAb", 'Heavy', 'Light', 'main_name', "batch"])
        temp = temp[["mAb", "batch", "variable", "value"]].drop_duplicates()
        
        temp0 = temp[temp["value"] == 0]
        temp1 = temp[temp["value"] == 1]
        temp01 = pd.concat([temp0, temp1]).sort_values("batch", ascending=False).drop_duplicates(["mAb", "variable"])
        
        tempnan = temp[temp["value"] == "nan"]
        tempnan["t"] = -1
        temp01["t"] = 0
        
        temp_final = pd.concat([temp01, tempnan]).sort_values("t", ascending=False).drop_duplicates(["mAb", "variable"])
        del temp_final["t"]
        del temp_final["batch"]
        
        ttemp = temp_final.pivot(index='mAb', columns='variable', values='value').reset_index()
        ttemp = pd.merge(ttemp, flu_clone[["mAb", 'Heavy', 'Light', 'main_name']].drop_duplicates("mAb"), on="mAb")
        ttemp["batch"] = ttemp["mAb"].apply(lambda x: 2 if "Inf" in str(x) else 1)
        ttemp["bb"] = ttemp["mAb"].apply(lambda x: int(str(x).split("-")[-1]))
        ttemp = ttemp.sort_values(["batch", "bb"])
        del ttemp["bb"]
        
        # Save results
        ttemp.to_csv(output_path, index=False)
        
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
            "message": f"Successfully processed experimental results, saved to: {output_path}\nData shape: {ttemp.shape}",
            "output_path": output_path,
            "processing_time_ms": total_time * 1000
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "status": "error",
            "error_type": "execution_error",
            "message": f"Error occurred during execution: {str(e)}\n{traceback.format_exc()}",
            "session_id": session_id
        }
        return


# ==================== Tool 5: Integrate ML Predictions with Experimental Data ====================

class IntegratePredictionsWithExperimentalDataArgs(BaseModel):
    """Arguments for integrating machine learning predictions with experimental measurements"""
    
    feature_data_path: Optional[str] = Field(
        default=None,
        description="Path to feature data CSV file"
    )
    
    bind_predict_dir: Optional[str] = Field(
        default=None,
        description="Directory containing binding prediction results"
    )
    
    neu_predict_dir: Optional[str] = Field(
        default=None,
        description="Directory containing neutralization prediction results"
    )
    
    clone_results_path: Optional[str] = Field(
        default=None,
        description="Path to clonal experimental results CSV file"
    )
    
    output_path: Optional[str] = Field(
        default=None,
        description="Output file path"
    )


@mcp.tool()
async def integrate_predictions_with_experimental_data(args: IntegratePredictionsWithExperimentalDataArgs):
    """Integrate machine learning prediction results with laboratory measurements
    
    This tool corresponds to the functionality in notebook 4.experiment+prediction.ipynb.
    Performs the following data integration workflow:
    
    **Machine Learning Prediction Data**
    - Load ensemble prediction results from multiple folds
    - Binding predictions: MetaBCR model predictions for H1N1/H3N2 strain binding
    - Neutralization predictions: Multi-fold cross-validated neutralization activity predictions
    - Pivot prediction scores into wide-format tables
    
    **Experimental Measurement Data**
    - Load laboratory-measured binding and neutralization activities
    - Standardize experimental results into binary classifications
    
    **Data Merging**
    - Merge prediction and experimental data into BCR feature dataset
    - Add single-cell/bulk data type labels
    - Retain all BCR sequence features, UMAP coordinates, and cell types
    
    Bioinformatics domains: ["machine learning", "antibody prediction", "experimental validation", "data integration"]
    Input data: ["ML prediction CSVs", "experimental measurements", "BCR features"]
    Output results: ["Integrated dataset CSV", "predictions + experiments + features"]
    
    Args:
        args: IntegratePredictionsWithExperimentalDataArgs - Merging parameters
        
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
            "message": "Starting predictions and experimental data integration",
            "timestamp": time.time()
        }
    }
    
    try:
        import pandas as pd
        import os
        
        # Determine file paths
        feature_path = args.feature_data_path or os.path.join(TEMP_DIR, "all_data_with_feature.csv")
        bind_dir = args.bind_predict_dir or os.path.join(INPUT_DIR, "predict_data", "ensemble_predict", "bind")
        neu_dir = args.neu_predict_dir or os.path.join(INPUT_DIR, "predict_data", "ensemble_predict", "neut")
        clone_path = args.clone_results_path or os.path.join(TEMP_DIR, "0220_Flu_cAb.csv")
        output_path = args.output_path or os.path.join(TEMP_DIR, "all_data_with_predict_and_feature.csv")
        
        # Load feature data
        data_v4 = pd.read_csv(feature_path, low_memory=False)
        if "Unnamed: 0" in data_v4.columns:
            del data_v4["Unnamed: 0"]
        if "Unnamed: 0.1" in data_v4.columns:
            del data_v4["Unnamed: 0.1"]
        
        # Load binding prediction results
        pb = pd.DataFrame()
        if os.path.exists(bind_dir):
            D_bind = {
                'BCR_0621MetaBCR-finetuned-2211_fold1_flu_binding_test__.csv': "0621_fold1-bind",
                "BCR_0822semi-bigdata_fold0_93_flu_binding_test__.csv": "0822_fold0-bind"
            }
            
            for xlsx in os.listdir(bind_dir):
                if xlsx not in D_bind:
                    continue
                pBind = pd.read_csv(os.path.join(bind_dir, xlsx))[['main_name', 'Antig Name', 'variant_seq', 'output']]
                pBind = pBind.pivot_table(values="output", index="main_name", columns="Antig Name").reset_index()
                pBind.columns = [f"{i}({D_bind[xlsx]})(predict)" if i != "main_name" else i for i in pBind.columns]
                
                if pb.empty:
                    pb = pBind
                else:
                    pb = pd.merge(pb, pBind, on="main_name")
        
        # Load neutralization prediction results
        pn = pd.DataFrame()
        if os.path.exists(neu_dir):
            D_neu = {
                'BCR_0620-nopretrain-2131_65_fold2_flu_neut_test__.csv': "0620_fold2-neu",
                'BCR_0620-nopretrain-2131_76_fold3_flu_neut_test__.csv': "0620_fold3-neu",
                'BCR_0620-nopretrain-2131_86_fold4_flu_neut_test__.csv': "0620_fold4-neu",
            }
            
            for xlsx in os.listdir(neu_dir):
                if xlsx not in D_neu:
                    continue
                pNeu = pd.read_csv(os.path.join(neu_dir, xlsx))[['main_name', 'Antig Name', 'variant_seq', 'output']]
                pNeu = pNeu.pivot_table(values="output", index="main_name", columns="Antig Name").reset_index()
                pNeu.columns = [f"{i}({D_neu[xlsx]})(predict)" if i != "main_name" else i for i in pNeu.columns]
                
                if pn.empty:
                    pn = pNeu
                else:
                    pn = pd.merge(pn, pNeu, on="main_name")
        
        # Merge prediction data
        if not pb.empty:
            data_v5 = pd.merge(data_v4, pb, on="main_name", how="left")
        else:
            data_v5 = data_v4.copy()
        
        if not pn.empty:
            data_v6 = pd.merge(data_v5, pn, on="main_name", how="left")
        else:
            data_v6 = data_v5.copy()
        
        # Merge clonal experimental results
        if os.path.exists(clone_path):
            clone_data = pd.read_csv(clone_path)
            if "Heavy" in clone_data.columns:
                del clone_data["Heavy"]
            if "Light" in clone_data.columns:
                del clone_data["Light"]
            Final_data = pd.merge(data_v6, clone_data, on="main_name", how="left")
        else:
            Final_data = data_v6.copy()
        
        # Add data type column
        Final_data["type"] = Final_data["main_name"].apply(lambda x: "bulk" if "_" not in str(x) else "sc")
        
        # Save results
        Final_data.to_csv(output_path, index=False)
        
        # 发送完成进度
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
        
        # 发送最终结果
        yield {
            "type": "result",
            "status": "success",
            "session_id": session_id,
            "message": f"Successfully integrated predictions and experimental results, saved to: {output_path}\nData shape: {Final_data.shape}",
            "output_path": output_path,
            "data_shape": list(Final_data.shape),
            "processing_time_ms": total_time * 1000
        }
        
    except Exception as e:
        error_msg = f"Error occurred during execution: {str(e)}\n{traceback.format_exc()}"
        yield {
            "type": "error",
            "status": "error",
            "error_type": "execution_error",
            "message": error_msg,
            "session_id": session_id
        }
        return


# ==================== Tools 6-8: Visualization and Tree Construction (Simplified) ====================

# @mcp.tool()
def visualize_antibody_repertoire_analysis(data_path: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """Visualize antibody repertoire analysis results
    
    This tool corresponds to the functionality in notebook 5.draw_for_mainfig.ipynb.
    Generates comprehensive analytical figures for influenza-specific antibody repertoires.
    
    Bioinformatics domains: ["antibody repertoire", "data visualization", "clonal analysis"]
    Input data: ["Integrated BCR dataset", "predictions", "experimental results"]
    Output results: ["Heatmaps", "UMAP plots", "clonal distribution plots", "gene usage plots"]
    
    Args:
        data_path: Input data path
        output_dir: Output directory
        
    Returns:
        Instructional information
    """
    return """Visualization Tool
    
Note: Due to the complexity of the visualization code, it is recommended to use the original 
notebook 5.draw_for_mainfig.ipynb directly, or extract the plotting code as a standalone Python 
script for invocation.

Included figures:
- FigS2E: Correlation coefficient heatmap and clustering plot
- Fig2F: Experimental score distribution on UMAP
- FigS3C: Heavy and light chain usage heatmap
- Fig2E: Clonal distribution scatter plot
- Fig5A: Clonal tree and sequence logo plot"""


# @mcp.tool()
def prepare_bcell_phylogenetic_tree_data(data_path: Optional[str] = None, output_path: Optional[str] = None) -> str:
    """Prepare data for B cell phylogenetic tree construction
    
    This tool corresponds to the functionality in notebook 6.cell_tree_preliminary.ipynb.
    
    Bioinformatics domains: ["phylogenetic analysis", "clonal lineage", "sequence alignment"]
    Input data: ["BCR clonal data", "CDR sequences", "germline sequences"]
    Output results: ["MSA-ready sequences", "ChangeO format data", "filtered clones"]
    
    Args:
        data_path: Input data path
        output_path: Output path
        
    Returns:
        Instructional information
    """
    return """Phylogenetic Tree Data Preparation Tool
    
This tool prepares the data required for phylogenetic tree construction.
It is recommended to use the original notebook 6.cell_tree_preliminary.ipynb directly.

Key functionalities:
- Filter clones with broadly neutralizing antibodies
- Extract CDR sequences for multiple sequence alignment (MSA) analysis
- Prepare data in ChangeO format"""


# @mcp.tool()
def construct_bcell_phylogenetic_tree(data_path: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """Construct and visualize B cell clonal phylogenetic trees
    
    This tool corresponds to the functionality in notebook 7.cell_tree(R).ipynb.
    Requires execution in an R environment.
    
    Bioinformatics domains: ["phylogenetic tree", "clonal evolution", "lineage tracing"]
    Input data: ["Clonal sequences", "germline references", "CDR regions"]
    Output results: ["Phylogenetic trees", "sequence logos", "evolution visualization"]
    
    Args:
        data_path: Input data path
        output_dir: Output directory
        
    Returns:
        Instructional information
    """
    return """Phylogenetic Tree Construction Tool
    
Note: This tool requires execution in an R environment.
Required R packages:
- ggplot2
- Biostrings
- alakazam
- igraph
- dplyr
- stringr
- ggpubr
- ggtree
- treeio
- tidyverse
- seqinr

It is recommended to use the original notebook 7.cell_tree(R).ipynb directly for tree construction.

Key functionalities:
- Construct phylogenetic trees
- Visualize clonal evolution
- Generate CDR sequence logo plots"""


# ==================== Lifecycle Management ====================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lineage_analysis_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Handle server startup and shutdown"""
    print("Lineage Analysis Server initializing...")
    
    try:
        yield {"initialized": True}
    finally:
        print("Lineage Analysis Server shutting down...")

# Set lifecycle handler
mcp.lifespan = lineage_analysis_lifespan


if __name__ == "__main__":
    print("Starting Lineage Analysis Server...")
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8109  # Use available port
    
    # Start with SSE transport mode
    mcp.run(transport="sse")

