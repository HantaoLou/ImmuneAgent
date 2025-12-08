from functools import wraps
from typing import Any, Callable, Optional
import asyncio
import time
import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_core.runnables.config import RunnableConfig
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing_extensions import Annotated
from urllib.parse import urlparse, urlunparse

from usecases.deepagents.prompts import (
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
    WRITE_TODOS_DESCRIPTION,
)
from usecases.deepagents.state import DeepAgentState, Todo
from common.factory import get_mcp_client, get_all_mcp_servers
from common.util.mcp_utils import MCPStreamingClient
from config.config import ApplicationConfig


async def _get_tool_schema_from_mcp(tool_name: str) -> Optional[dict]:
    """
    从MCP服务获取工具的schema
    
    Args:
        tool_name: 工具名称
        
    Returns:
        工具的JSON schema字典，如果找不到则返回None
    """
    try:
        all_servers = get_all_mcp_servers()
        
        # 遍历所有MCP服务器，查找包含该工具的服务
        for service_id in all_servers.keys():
            try:
                config = RunnableConfig(configurable={"mcp_config": {"service_ids": [service_id]}})
                client = await get_mcp_client(config)
                tools = await client.get_tools()
                
                # 查找匹配的工具
                for tool in tools:
                    if tool.name.lower() == tool_name.lower():
                        # 获取工具的args_schema
                        args_schema = getattr(tool, "args_schema", None)
                        
                        if args_schema:
                            # 尝试转换为JSON schema
                            if hasattr(args_schema, "model_json_schema"):
                                schema = args_schema.model_json_schema()
                                print(f"[FileValidation] 成功获取工具 {tool_name} 的schema (来自服务 {service_id})")
                                return schema
                            elif hasattr(args_schema, "schema"):
                                schema = args_schema.schema()
                                print(f"[FileValidation] 成功获取工具 {tool_name} 的schema (来自服务 {service_id})")
                                return schema
                            elif isinstance(args_schema, dict):
                                # 如果已经是字典，直接返回
                                print(f"[FileValidation] 成功获取工具 {tool_name} 的schema (来自服务 {service_id})")
                                return args_schema
            except Exception as e:
                # 某个服务器连接失败，继续尝试下一个
                continue
        
        return None
    except Exception as e:
        return None


@tool(description=WRITE_TODOS_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files"""
    return list(state.get("files", {}).keys())


@tool(description=TOOL_DESCRIPTION)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file."""
    mock_filesystem = state.get("files", {})
    if file_path not in mock_filesystem:
        return f"Error: File '{file_path}' not found"

    # Get file content
    content = mock_filesystem[file_path]

    # Handle empty file
    if not content or content.strip() == "":
        return "System reminder: File exists but has empty contents"

    # Split content into lines
    lines = content.splitlines()

    # Apply line offset and limit
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    # Handle case where offset is beyond file length
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    # Format output with line numbers (cat -n format)
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i]

        # Truncate lines longer than 2000 characters
        if len(line_content) > 2000:
            line_content = line_content[:2000]

        # Line numbers start at 1, so add 1 to the index
        line_number = i + 1
        result_lines.append(f"{line_number:6d}\t{line_content}")

    return "\n".join(result_lines)


def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write to a file."""
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(description=EDIT_DESCRIPTION)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    replace_all: bool = False,
) -> str:
    """Write to a file."""
    mock_filesystem = state.get("files", {})
    # Check if file exists in mock filesystem
    if file_path not in mock_filesystem:
        return f"Error: File '{file_path}' not found"

    # Get current file content
    content = mock_filesystem[file_path]

    # Check if old_string exists in the file
    if old_string not in content:
        return f"Error: String not found in file: '{old_string}'"

    # If not replace_all, check for uniqueness
    if not replace_all:
        occurrences = content.count(old_string)
        if occurrences > 1:
            return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        elif occurrences == 0:
            return f"Error: String not found in file: '{old_string}'"

    # Perform the replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacement_count = content.count(old_string)
        result_msg = f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
    else:
        new_content = content.replace(
            old_string, new_string, 1
        )  # Replace only first occurrence
        result_msg = f"Successfully replaced string in '{file_path}'"

    # Update the mock filesystem
    mock_filesystem[file_path] = new_content
    return Command(
        update={
            "files": mock_filesystem,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


class UserInput(BaseModel):
    accept: bool
    args: Optional[dict[str, Any]] = None


async def _invoke_with_progress_and_retry(
    tool: BaseTool,
    args: dict[str, Any],
    max_retries: int = 3,
    retry_delay: float = 2.0,
    progress_interval: int = 30,
) -> Any:
    """
    带进度提示和重试机制的工具调用函数
    
    Args:
        tool: 要调用的工具
        args: 工具参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        progress_interval: 进度提示间隔（秒）
        
    Returns:
        工具调用的结果
        
    Raises:
        Exception: 所有重试失败后抛出最后一个异常
    """
    import time
    import sys
    from datetime import datetime
    
    async def show_progress(tool_name: str, start_time: float):
        """定期显示进度信息"""
        while True:
            await asyncio.sleep(progress_interval)
            elapsed = time.time() - start_time
            elapsed_minutes = int(elapsed // 60)
            elapsed_seconds = int(elapsed % 60)
            
            if elapsed_minutes > 0:
                time_str = f"{elapsed_minutes}分{elapsed_seconds}秒"
            else:
                time_str = f"{elapsed_seconds}秒"
            
            progress_msg = (
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"⏳ 工具 '{tool_name}' 仍在运行中... "
                f"(已用时: {time_str})"
            )
            print(progress_msg, file=sys.stderr, flush=True)
    
    last_exception = None
    current_delay = retry_delay
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # 创建进度提示任务
            progress_task = asyncio.create_task(
                show_progress(tool.name, start_time)
            )
            
            try:
                # 执行工具调用
                if hasattr(tool, "ainvoke"):
                    result = await tool.ainvoke(args)
                else:
                    result = tool.invoke(args)
                
                # 取消进度提示任务
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
                
                # 计算总耗时
                total_time = time.time() - start_time
                total_minutes = int(total_time // 60)
                total_seconds = int(total_time % 60)
                
                if total_minutes > 0:
                    total_time_str = f"{total_minutes}分{total_seconds}秒"
                else:
                    total_time_str = f"{total_seconds}秒"
                
                # 检查结果是否是流式任务响应（收到响应不代表完成）
                is_streaming_response = False
                streaming_info = None
                
                if isinstance(result, str):
                    try:
                        result_dict = json.loads(result)
                        if isinstance(result_dict, dict) and result_dict.get("type") == "streaming_task":
                            is_streaming_response = True
                            streaming_info = {
                                "task_id": result_dict.get("task_id"),
                                "service_id": result_dict.get("service_id"),  # 新格式
                                "stream_url": result_dict.get("stream_url"),  # 旧格式兼容
                                "message": result_dict.get("message", "")
                            }
                    except:
                        pass
                elif isinstance(result, dict) and result.get("type") == "streaming_task":
                    is_streaming_response = True
                    streaming_info = {
                        "task_id": result.get("task_id"),
                        "service_id": result.get("service_id"),  # 新格式
                        "stream_url": result.get("stream_url"),  # 旧格式兼容
                        "message": result.get("message", "")
                    }
                
                # 如果检测到流式任务响应，需要连接 SSE 等待完成
                if is_streaming_response and streaming_info:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"⚠️ 工具 '{tool.name}' 返回了流式任务响应（工具执行尚未完成）",
                        flush=True
                    )
                    
                    # 提取 task_id 和 service_id（新格式）
                    task_id = streaming_info.get("task_id")
                    stream_service_id = streaming_info.get("service_id")
                    existing_stream_url = streaming_info.get("stream_url")  # 旧格式兼容
                    
                    stream_url = None
                    base_url = None
                    
                    # 新格式：根据 service_id 和 task_id 拼接 stream_url
                    if stream_service_id and task_id:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"🔗 使用新格式：service_id={stream_service_id}, task_id={task_id}",
                            flush=True
                        )
                        
                        # 获取服务器配置
                        all_servers = ApplicationConfig.get_instance().mcp_servers
                        server_config = all_servers.get(stream_service_id, {})
                        server_url = server_config.get("url", "")
                        
                        # 使用 urlparse 提取基础 URL（只包含 协议://ip:port）
                        server_url_parsed = urlparse(server_url)
                        base_url = f"{server_url_parsed.scheme}://{server_url_parsed.netloc}"
                        
                        # 拼接 stream_url: {base_url}/stream/{task_id}
                        stream_url = f"{base_url}/stream/{task_id}"
                        
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"   服务器配置 URL: {server_url}",
                            flush=True
                        )
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"   基础 URL: {base_url}",
                            flush=True
                        )
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"   拼接的 Stream URL: {stream_url}",
                            flush=True
                        )
                    
                    # 旧格式兼容：如果已有 stream_url，使用它（但需要处理 localhost 等）
                    elif existing_stream_url:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"⚠️ 使用旧格式的 stream_url（兼容模式）",
                            flush=True
                        )
                        stream_url = existing_stream_url
                        original_stream_url = stream_url
                        
                        # 从 stream_url 推断服务器配置
                        stream_url_parsed = urlparse(stream_url)
                        base_url = f"{stream_url_parsed.scheme}://{stream_url_parsed.netloc}"
                        
                        # 尝试从配置中找到匹配的 service_id
                        detected_service_id = None
                        all_servers = ApplicationConfig.get_instance().mcp_servers
                        for sid, server_config in all_servers.items():
                            server_url = server_config.get("url", "").rstrip('/')
                            if server_url.endswith("/sse"):
                                server_url = server_url[:-4]
                            
                            try:
                                server_url_parsed = urlparse(server_url)
                                if server_url_parsed.netloc == stream_url_parsed.netloc:
                                    detected_service_id = sid
                                    break
                            except:
                                pass
                        
                        # 处理 localhost 地址替换（如果需要）
                        if "localhost" in stream_url or "127.0.0.1" in stream_url:
                            if detected_service_id:
                                server_config = all_servers.get(detected_service_id, {})
                                server_url = server_config.get("url", "").rstrip('/')
                                if server_url.endswith("/sse"):
                                    server_url = server_url[:-4]
                                try:
                                    server_url_parsed = urlparse(server_url)
                                    new_stream_url = urlunparse((
                                        server_url_parsed.scheme,
                                        server_url_parsed.netloc,
                                        stream_url_parsed.path,
                                        stream_url_parsed.params,
                                        stream_url_parsed.query,
                                        stream_url_parsed.fragment
                                    ))
                                    stream_url = new_stream_url
                                    base_url = f"{server_url_parsed.scheme}://{server_url_parsed.netloc}"
                                    print(
                                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                                        f"🔄 将 localhost 替换为实际服务器地址: {stream_url}",
                                        flush=True
                                    )
                                except Exception as e:
                                    print(
                                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                                        f"⚠️ 替换 localhost 失败: {e}，继续使用原始 stream_url",
                                        flush=True
                                    )
                        elif stream_url.startswith("/"):
                            # 使用检测到的 service_id 的 base_url，或使用默认
                            if detected_service_id:
                                server_config = all_servers.get(detected_service_id, {})
                                server_url = server_config.get("url", "").rstrip('/')
                                if server_url.endswith("/sse"):
                                    server_url = server_url[:-4]
                                base_url = server_url
                                stream_url = f"{base_url}{stream_url}"
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"   补全相对路径 Stream URL: {original_stream_url} -> {stream_url}",
                                flush=True
                            )
                    
                    # 如果既没有 service_id 也没有 stream_url，无法继续
                    if not stream_url or not base_url:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"❌ 流式任务响应中缺少 service_id/task_id 或 stream_url，无法连接 SSE",
                            flush=True
                        )
                        return result
                    
                    # 创建流式客户端并连接 SSE
                    try:
                        # 确定使用的 service_id
                        actual_service_id = stream_service_id
                        if not actual_service_id and existing_stream_url:
                            # 尝试从 stream_url 推断 service_id
                            all_servers = ApplicationConfig.get_instance().mcp_servers
                            stream_url_parsed = urlparse(stream_url)
                            for sid, server_config in all_servers.items():
                                server_url = server_config.get("url", "").rstrip('/')
                                if server_url.endswith("/sse"):
                                    server_url = server_url[:-4]
                                try:
                                    server_url_parsed = urlparse(server_url)
                                    if server_url_parsed.netloc == stream_url_parsed.netloc:
                                        actual_service_id = sid
                                        break
                                except:
                                    pass
                        
                        timeout = 36000  # 默认10小时
                        if actual_service_id:
                            all_servers = ApplicationConfig.get_instance().mcp_servers
                            server_config = all_servers.get(actual_service_id, {})
                            timeout = server_config.get("timeout", timeout)
                        
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"   使用的 Service ID: {actual_service_id or '未知'}",
                            flush=True
                        )
                        streaming_client = MCPStreamingClient(base_url, timeout=timeout)
                        
                        def on_progress(msg):
                            data = msg.get("data", {})
                            progress = data.get("progress_percent", 0)
                            msg_text = data.get("message", "")
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"📊 [SSE进度] {progress:.1f}% - {msg_text}",
                                flush=True
                            )
                        
                        def on_result(msg):
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"✅ [SSE结果] 任务完成: {msg.get('status', 'success')}",
                                flush=True
                            )
                        
                        def on_error(msg):
                            error_msg = msg.get("message", "Unknown error")
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"❌ [SSE错误] 任务错误: {error_msg}",
                                flush=True
                            )
                        
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"🌊 开始连接到 SSE 端点接收实时进度...",
                            flush=True
                        )
                        
                        # 接收流式进度和最终结果
                        stream_result = await streaming_client.receive_stream(
                            stream_url,
                            on_progress=on_progress,
                            on_result=on_result,
                            on_error=on_error
                        )
                        
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"✅ SSE 流式传输完成，工具执行已真正完成",
                            flush=True
                        )
                        
                        # 从流式结果中提取最终结果
                        if stream_result.get("type") == "result":
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"   从 SSE 结果消息中提取最终数据",
                                flush=True
                            )
                            # 检查是否有嵌套的 data 字段
                            if "data" in stream_result and isinstance(stream_result["data"], dict):
                                # 如果有 data 字段，使用 data 字段的内容
                                result_data = stream_result["data"]
                                # 优先使用 data.result，否则使用 data 本身
                                result = result_data.get("result") or result_data
                            else:
                                # 如果没有 data 字段，直接使用 stream_result 本身
                                # 排除 type 字段，保留其他所有字段（file_path, file_size_mb 等）
                                result = {k: v for k, v in stream_result.items() if k != "type"}
                            
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"   最终结果类型: {type(result)}, 大小: {len(str(result)) if result else 0} 字符",
                                flush=True
                            )
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"   最终结果内容: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}",
                                flush=True
                            )
                            
                            # 计算最终耗时
                            final_total_time = time.time() - start_time
                            final_total_minutes = int(final_total_time // 60)
                            final_total_seconds = int(final_total_time % 60)
                            if final_total_minutes > 0:
                                final_total_time_str = f"{final_total_minutes}分{final_total_seconds}秒"
                            else:
                                final_total_time_str = f"{final_total_seconds}秒"
                            
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"✅ 工具 '{tool.name}' 执行完成 (总耗时: {final_total_time_str})",
                                flush=True
                            )
                        elif stream_result.get("type") == "end":
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"✅ 收到 SSE 结束消息，工具执行完成",
                                flush=True
                            )
                            # 如果只有结束消息，保持原始结果
                        
                        # 关闭流式客户端
                        await streaming_client.close()
                        
                    except Exception as stream_error:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"⚠️ SSE 流式传输失败，使用原始结果: {str(stream_error)}",
                            flush=True
                        )
                        # 如果流式传输失败，返回原始结果
                        pass
                else:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"✅ 工具 '{tool.name}' 执行完成 (总耗时: {total_time_str})",
                        flush=True
                    )
                
                return result
                
            except Exception as e:
                # 取消进度提示任务
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
                
                # 重新抛出异常以便重试逻辑处理
                raise
                
        except Exception as e:
            last_exception = e
            error_str = str(e)
            error_type = type(e).__name__
            
            # 检查是否是MCP连接错误
            is_mcp_error = (
                "McpError" in error_type or
                "Connection closed" in error_str or
                "Connection close" in error_str or
                "RemoteProtocolError" in error_type or
                "peer closed connection" in error_str or
                "incomplete chunked read" in error_str
            )
            
            if is_mcp_error:
                if attempt < max_retries - 1:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"[MCP重试] 工具 {tool.name} 调用失败 (尝试 {attempt + 1}/{max_retries}): "
                        f"MCP连接错误 - {error_str[:200]}"
                    )
                    print(f"[MCP重试] 等待 {current_delay:.1f} 秒后重试...")
                    await asyncio.sleep(current_delay)
                    # 指数退避：每次重试延迟时间增加
                    current_delay *= 1.5
                else:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"[MCP错误] 工具 {tool.name} 调用失败，已达到最大重试次数"
                    )
                    print(f"[MCP错误] 错误详情: {error_str}")
                    print(
                        "[MCP错误] 提示: 请检查MCP服务器是否正在运行，"
                        "以及网络连接是否正常"
                    )
            else:
                # 非MCP错误，不重试，直接抛出
                raise
    
    # 所有重试都失败
    raise last_exception


async def _invoke_with_retry(
    tool: BaseTool,
    args: dict[str, Any],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Any:
    """
    带重试机制的工具调用函数（使用进度提示版本）
    
    Args:
        tool: 要调用的工具
        args: 工具参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        
    Returns:
        工具调用的结果
        
    Raises:
        Exception: 所有重试失败后抛出最后一个异常
    """
    return await _invoke_with_progress_and_retry(
        tool, args, max_retries, retry_delay, progress_interval=30
    )


def hil(t: BaseTool) -> BaseTool:
    def wrapper_fn(**tool_input):
        print(f"call tool: {t.name} with {tool_input}")
        print(f"[DEBUG] hil() 包装器：准备触发 interrupt，工具: {t.name}")
        import traceback
        print(f"[DEBUG] hil() 调用栈:\n{''.join(traceback.format_stack()[-5:-1])}")
        user_input = interrupt(f"call tool: {t.name} with {tool_input}")
        print(f"[DEBUG] hil() 包装器：interrupt() 返回，工具: {t.name}")

        try:
            print(f"[DEBUG] 接收到的user_input: {user_input}")
            print(f"[DEBUG] user_input类型: {type(user_input)}")
            print(f"[DEBUG] ⚠️ 警告：这个 user_input 是从 interrupt() 返回的，不是用户输入！")
            user_input_model = UserInput.model_validate_json(user_input)
            print(
                f"[DEBUG] 解析后的模型: accept={user_input_model.accept}, args={user_input_model.args}"
            )
            if user_input_model.accept:
                # 确定最终使用的参数
                final_args = user_input_model.args if user_input_model.args is not None else tool_input
                
                try:
                    from usecases.immunity.common.file_param_validator import validate_file_params_for_tool
                    
                    # 获取工具的schema
                    tool_schema = None
                    
                    # 首先尝试从工具对象本身获取schema
                    if hasattr(t, "args_schema") and t.args_schema:
                        # 将Pydantic schema转换为字典格式
                        if hasattr(t.args_schema, "model_json_schema"):
                            tool_schema = t.args_schema.model_json_schema()
                        elif hasattr(t.args_schema, "schema"):
                            tool_schema = t.args_schema.schema()
                    
                    # 如果工具对象没有schema，尝试从MCP服务获取
                    if not tool_schema:
                        print(f"[FileValidation] 工具 {t.name} 没有args_schema，尝试从MCP服务获取...")
                        tool_schema = asyncio.run(_get_tool_schema_from_mcp(t.name))
                    
                    # 异步校验文件参数（在调用工具前）
                    print(f"[FileValidation] 开始校验工具 {t.name} 的文件参数...")
                    validated_args = asyncio.run(
                        validate_file_params_for_tool(
                            tool_args=final_args,
                            tool_schema=tool_schema
                        )
                    )
                    
                    if validated_args != final_args:
                        print(f"[FileValidation] 文件参数已更新，使用校验后的参数")
                        final_args = validated_args
                    else:
                        print(f"[FileValidation] 文件参数校验完成，无需转换")
                        
                except Exception as validation_error:
                    # 文件校验失败不应该阻止工具执行，只记录警告
                    import traceback
                    print(f"[FileValidation] 警告：文件参数校验失败，继续使用原始参数: {validation_error}")
                    print(f"[FileValidation] 详细错误: {traceback.format_exc()}")
                
                if user_input_model.args is not None:
                    print(
                        f"user accept to call tool: {t.name} with {final_args}"
                    )
                    # 使用带重试机制的调用
                    return asyncio.run(_invoke_with_retry(t, final_args))
                else:
                    print(f"calling with default args: {final_args}")
                    # 使用带重试机制的调用
                    return asyncio.run(_invoke_with_retry(t, final_args))
            else:
                return f"user reject to call tool: {t.name} with {tool_input}"
        except Exception as e:
            import traceback

            stack_trace = traceback.format_exc()
            error_str = str(e)
            error_type = type(e).__name__
            
            # 检查是否是MCP连接错误，提供更友好的错误消息
            is_mcp_error = (
                "McpError" in error_type or
                "Connection closed" in error_str or
                "Connection close" in error_str or
                "RemoteProtocolError" in error_type or
                "peer closed connection" in error_str or
                "incomplete chunked read" in error_str
            )
            
            if is_mcp_error:
                friendly_error = (
                    f"MCP连接错误: 无法连接到MCP服务器。"
                    f"工具 '{t.name}' 调用失败。"
                    f"错误信息: {error_str[:300]}"
                    f"\n\n建议："
                    f"\n1. 检查MCP服务器是否正在运行"
                    f"\n2. 检查网络连接是否正常"
                    f"\n3. 检查配置文件中的MCP服务器地址是否正确"
                    f"\n4. 查看服务器日志以获取更多信息"
                )
                print(f"error: {friendly_error}\n{stack_trace}")
                return f"error: {friendly_error}"
            else:
                print(f"error: {e}\n{stack_trace}")
                return f"error: {e}"

    wrapper = tool(t.name, description=t.description)(wrapper_fn)
    wrapper.args_schema = t.args_schema
    wrapper.return_direct = t.return_direct
    wrapper.description = t.description
    return wrapper
