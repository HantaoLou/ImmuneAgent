import asyncio
import json
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from langchain_core.runnables.config import RunnableConfig
import httpx

from common.factory import get_mcp_client
from config.config import ApplicationConfig


class MCPStreamingClient:
    """
    MCP 流式传输客户端
    支持建立 SSE 连接、调用工具、接收响应和连接自定义 SSE 端点
    """
    
    def __init__(self, server_url: str, timeout: int = 36000):
        """
        初始化 MCP 流式客户端
        
        Args:
            server_url: 服务器地址，例如 "http://localhost:8110"
            timeout: 超时时间（秒），默认10小时
        """
        self.server_url = server_url.rstrip('/')
        self.sse_url = f"{self.server_url}/sse"
        self.messages_url = f"{self.server_url}/messages"
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self.session_id: Optional[str] = None
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 [MCPStreamingClient] 初始化客户端")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   服务器地址: {self.server_url}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   SSE 端点: {self.sse_url}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   消息端点: {self.messages_url}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   超时时间: {timeout}秒")
    
    async def connect(self) -> str:
        """
        建立 MCP SSE 连接
        
        Returns:
            session_id: 会话 ID，用于后续请求
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔌 [MCPStreamingClient] 开始建立 SSE 连接")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   连接 URL: {self.sse_url}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   请求头: Accept: text/event-stream")
        
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        # 建立 SSE 连接
        try:
            response = await self.client.get(
                self.sse_url,
                headers={"Accept": "text/event-stream"},
            )
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 [MCPStreamingClient] 收到 SSE 连接响应")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   状态码: {response.status_code}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [MCPStreamingClient] SSE 连接失败: {response.status_code}")
                raise Exception(f"Failed to establish SSE connection: {response.status_code}")
            
            # 从响应头或第一个消息中获取 session_id
            self.session_id = response.headers.get("X-Session-Id") or "default_session"
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [MCPStreamingClient] SSE 连接已建立")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Session ID: {self.session_id}")
            return self.session_id
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [MCPStreamingClient] SSE 连接异常: {str(e)}")
            raise
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        调用工具并接收响应
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            session_id: 会话 ID（如果未提供，使用默认的）
        
        Returns:
            工具调用响应，包含 task_id 和 stream_url（如果是流式工具）
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 [MCPStreamingClient] 发起工具调用请求")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   工具名称: {tool_name}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   工具参数: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
        
        if not self.client:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [MCPStreamingClient] 未建立连接，请先调用 connect()")
            raise Exception("Not connected. Call connect() first.")
        
        session_id = session_id or self.session_id
        if not session_id:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [MCPStreamingClient] 无可用 session_id")
            raise Exception("No session_id available")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   使用 Session ID: {session_id}")
        
        # 发送工具调用请求
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        request_url = f"{self.messages_url}/?session_id={session_id}"
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   请求 URL: {request_url}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   请求体: {json.dumps(request, ensure_ascii=False, indent=2)}")
        
        try:
            response = await self.client.post(
                request_url,
                json=request,
                headers={"Content-Type": "application/json"},
            )
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📨 [MCPStreamingClient] 收到工具调用响应")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   状态码: {response.status_code}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   响应头: {dict(response.headers)}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   响应体长度: {len(response.text) if response.text else 0} 字符")
            
            if response.status_code != 202:  # FastMCP 返回 202 Accepted
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [MCPStreamingClient] 工具调用失败")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   错误响应: {response.text[:500]}")
                raise Exception(f"Tool call failed: {response.status_code} - {response.text}")
            
            # 尝试解析响应
            result = {}
            try:
                if response.text:
                    result = response.json()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [MCPStreamingClient] 成功解析响应")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   响应内容: {json.dumps(result, ensure_ascii=False, indent=2)[:1000]}")
                    return result
            except json.JSONDecodeError as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [MCPStreamingClient] JSON 解析失败: {e}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   原始响应: {response.text[:500]}")
            
            # 如果响应为空，需要通过 SSE 流接收（由 langchain_mcp_adapters 处理）
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ [MCPStreamingClient] 响应为空，可能需要通过 SSE 流接收")
            return {}
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [MCPStreamingClient] 工具调用异常: {str(e)}")
            raise
    
    @staticmethod
    def parse_streaming_response(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        解析工具调用响应，提取流式传输信息
        
        Args:
            response: 工具调用响应
        
        Returns:
            包含 task_id 和 stream_url 的字典，如果不是流式工具则返回 None
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 [parse_streaming_response] 开始解析响应")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   响应类型: {type(response)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   响应键: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
        
        if not response or "result" not in response:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ [parse_streaming_response] 响应不包含 result 字段")
            return None
        
        result = response["result"]
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果类型: {type(result)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        # 首先检查 result 本身是否就是流式任务信息（最直接的情况）
        if isinstance(result, dict) and result.get("type") == "streaming_task":
            streaming_info = {
                "task_id": result.get("task_id"),
                "service_id": result.get("service_id"),  # 新格式：包含 service_id
                "stream_url": result.get("stream_url"),  # 旧格式兼容：可能还包含 stream_url
                "message": result.get("message", "")
            }
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [parse_streaming_response] 在 result 中直接找到流式任务信息")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Task ID: {streaming_info.get('task_id')}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Service ID: {streaming_info.get('service_id')}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Stream URL: {streaming_info.get('stream_url')}")
            return streaming_info
        
        # 检查 structuredContent
        structured_content = result.get("structuredContent")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   检查 structuredContent: {structured_content is not None}")
        if structured_content:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   structuredContent 内容: {json.dumps(structured_content, ensure_ascii=False, indent=2)[:500]}")
            if structured_content.get("type") == "streaming_task":
                streaming_info = {
                    "task_id": structured_content.get("task_id"),
                    "service_id": structured_content.get("service_id"),  # 新格式：包含 service_id
                    "stream_url": structured_content.get("stream_url"),  # 旧格式兼容：可能还包含 stream_url
                    "message": structured_content.get("message", "")
                }
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [parse_streaming_response] 在 structuredContent 中找到流式任务信息")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Task ID: {streaming_info.get('task_id')}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Service ID: {streaming_info.get('service_id')}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Stream URL: {streaming_info.get('stream_url')}")
                return streaming_info
        
        # 检查 content 中的文本
        content = result.get("content", [])
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   检查 content: 类型={type(content)}, 长度={len(content) if isinstance(content, (list, str)) else 'N/A'}")
        
        if isinstance(content, list):
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   遍历 content 列表，共 {len(content)} 项")
            for i, item in enumerate(content):
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   检查 content[{i}]: 类型={item.get('type') if isinstance(item, dict) else type(item)}")
                if item.get("type") == "text":
                    try:
                        text_content = item.get("text", "{}")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   解析文本内容: {text_content[:200]}...")
                        text_data = json.loads(text_content)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   解析后的类型: {text_data.get('type') if isinstance(text_data, dict) else 'N/A'}")
                        if text_data.get("type") == "streaming_task":
                            streaming_info = {
                                "task_id": text_data.get("task_id"),
                                "service_id": text_data.get("service_id"),  # 新格式：包含 service_id
                                "stream_url": text_data.get("stream_url"),  # 旧格式兼容：可能还包含 stream_url
                                "message": text_data.get("message", "")
                            }
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [parse_streaming_response] 在 content[{i}] 中找到流式任务信息")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Task ID: {streaming_info.get('task_id')}")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Service ID: {streaming_info.get('service_id')}")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Stream URL: {streaming_info.get('stream_url')}")
                            return streaming_info
                    except json.JSONDecodeError as e:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ content[{i}] JSON 解析失败: {e}")
                        pass
        elif isinstance(content, str):
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   content 是字符串，尝试解析")
            try:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   字符串内容: {content[:200]}...")
                text_data = json.loads(content)
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   解析后的类型: {text_data.get('type') if isinstance(text_data, dict) else 'N/A'}")
                if text_data.get("type") == "streaming_task":
                    streaming_info = {
                        "task_id": text_data.get("task_id"),
                        "service_id": text_data.get("service_id"),  # 新格式：包含 service_id
                        "stream_url": text_data.get("stream_url"),  # 旧格式兼容：可能还包含 stream_url
                        "message": text_data.get("message", "")
                    }
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [parse_streaming_response] 在字符串 content 中找到流式任务信息")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Task ID: {streaming_info.get('task_id')}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Service ID: {streaming_info.get('service_id')}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Stream URL: {streaming_info.get('stream_url')}")
                    return streaming_info
            except json.JSONDecodeError as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 字符串 content JSON 解析失败: {e}")
                pass
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ [parse_streaming_response] 未找到流式任务信息")
        return None
    
    async def receive_stream(
        self,
        stream_url: str,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        连接到自定义 SSE 端点接收进度消息
        
        Args:
            stream_url: 流式传输端点 URL
            on_progress: 进度消息回调
            on_result: 结果消息回调
            on_error: 错误消息回调
        
        Returns:
            最终结果消息
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌐 [receive_stream] 开始连接到流式传输端点")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   流式端点 URL: {stream_url}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   请求方法: GET")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   请求头: Accept: text/event-stream")
        
        final_result = None
        message_count = 0
        
        try:
            async with httpx.AsyncClient(timeout=None) as stream_client:
                async with stream_client.stream(
                    "GET",
                    stream_url,
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 [receive_stream] 收到连接响应")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   状态码: {response.status_code}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   响应头: {dict(response.headers)}")
                    
                    if response.status_code != 200:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [receive_stream] 连接失败: {response.status_code}")
                        # 尝试读取错误响应体
                        error_text = "无错误详情"
                        try:
                            # 由于使用的是 stream，需要先读取内容
                            error_body = b""
                            async for chunk in response.aiter_bytes():
                                error_body += chunk
                                if len(error_body) > 1000:  # 限制读取长度
                                    break
                            if error_body:
                                error_text = error_body.decode('utf-8', errors='ignore')
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   错误响应体: {error_text}")
                        except Exception as read_error:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]   无法读取错误响应: {read_error}")
                        
                        error_msg = f"Failed to connect to stream endpoint: {response.status_code}"
                        if error_text and error_text != "无错误详情":
                            error_msg += f"\n服务器错误信息: {error_text}"
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   完整错误: {error_msg}")
                        raise Exception(error_msg)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [receive_stream] 已成功连接到流式传输端点")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 [receive_stream] 开始接收 SSE 消息流...")
                    
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        
                        # SSE 格式: "data: {...}\n\n"
                        if line.startswith("data: "):
                            message_count += 1
                            data = line[6:]  # 移除 "data: " 前缀
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📨 [receive_stream] 收到第 {message_count} 条 SSE 消息")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]   原始数据长度: {len(data)} 字符")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]   原始数据预览: {data[:200]}...")
                            
                            try:
                                message = json.loads(data)
                                msg_type = message.get("type", "unknown")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [receive_stream] 消息解析成功")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   消息类型: {msg_type}")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   完整消息: {json.dumps(message, ensure_ascii=False, indent=2)[:500]}")
                                
                                if msg_type == "end":
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 [receive_stream] 收到结束消息，流式传输结束")
                                    break
                                elif msg_type == "error":
                                    error_msg = message.get("message", "Unknown error")
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [receive_stream] 收到错误消息")
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   错误信息: {error_msg}")
                                    if on_error:
                                        on_error(message)
                                    raise Exception(error_msg)
                                elif msg_type == "progress":
                                    data_content = message.get("data", {})
                                    progress = data_content.get("progress_percent", 0)
                                    msg_text = data_content.get("message", "")
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 [receive_stream] 收到进度消息")
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   进度: {progress:.1f}%")
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   消息: {msg_text}")
                                    if on_progress:
                                        on_progress(message)
                                elif msg_type == "result":
                                    final_result = message
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [receive_stream] 收到结果消息")
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   状态: {message.get('status', 'success')}")
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果数据: {json.dumps(message.get('data', {}), ensure_ascii=False, indent=2)[:500]}")
                                    if on_result:
                                        on_result(message)
                                else:
                                    # 未知消息类型，尝试作为进度消息处理
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📨 [receive_stream] 收到未知类型消息: {msg_type}")
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   消息内容: {json.dumps(message, ensure_ascii=False, indent=2)[:500]}")
                                    if on_progress:
                                        on_progress(message)
                            except json.JSONDecodeError as e:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [receive_stream] SSE 消息 JSON 解析失败")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   解析错误: {str(e)}")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   原始数据: {data[:500]}...")
                        elif line.startswith(": "):
                            # 心跳消息，忽略
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] 💓 [receive_stream] 收到心跳消息（忽略）")
                            continue
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 [receive_stream] 收到未知格式的行: {line[:100]}")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 [receive_stream] 流式传输统计")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   共收到消息数: {message_count}")
            
            if final_result is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [receive_stream] 未收到最终结果消息")
                raise Exception("未收到最终结果消息")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [receive_stream] 流式传输完成，返回最终结果")
            return final_result
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [receive_stream] 流式传输异常")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   异常类型: {type(e).__name__}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   异常信息: {str(e)}")
            raise
    
    async def close(self):
        """关闭连接"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔌 [MCPStreamingClient] 关闭连接")
        if self.client:
            await self.client.aclose()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [MCPStreamingClient] 连接已关闭")


async def _invoke_with_progress(
    tool, params: dict, tool_name: str, progress_interval: int = 30
):
    """
    带进度提示的工具调用包装函数
    
    Args:
        tool: 要调用的工具对象
        params: 工具参数
        tool_name: 工具名称（用于显示）
        progress_interval: 进度提示间隔（秒），默认30秒
        
    Returns:
        工具调用结果
    """
    import sys
    
    start_time = time.time()
    last_progress_time = start_time
    connection_healthy = True
    
    async def show_progress():
        """定期显示进度信息"""
        nonlocal last_progress_time, connection_healthy
        while True:
            await asyncio.sleep(progress_interval)
            elapsed = time.time() - start_time
            elapsed_minutes = int(elapsed // 60)
            elapsed_seconds = int(elapsed % 60)
            
            # 格式化时间
            if elapsed_minutes > 0:
                time_str = f"{elapsed_minutes}分{elapsed_seconds}秒"
            else:
                time_str = f"{elapsed_seconds}秒"
            
            # 输出进度信息（使用stderr避免干扰输出）
            progress_msg = (
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"⏳ 工具 '{tool_name}' 仍在运行中... "
                f"(已用时: {time_str})"
            )
            print(progress_msg, file=sys.stderr, flush=True)
            
            last_progress_time = time.time()
    
    # 创建进度提示任务
    progress_task = asyncio.create_task(show_progress())
    
    try:
        # 执行工具调用（这里可能会因为连接断开而抛出异常）
        result = await tool.ainvoke(params)
        
        # 取消进度提示任务
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        
        connection_healthy = True
        
        # 计算总耗时
        total_time = time.time() - start_time
        total_minutes = int(total_time // 60)
        total_seconds = int(total_time % 60)
        
        if total_minutes > 0:
            total_time_str = f"{total_minutes}分{total_seconds}秒"
        else:
            total_time_str = f"{total_seconds}秒"
        
        # 检查结果是否有效
        result_type = type(result).__name__
        result_size = len(str(result)) if result is not None else 0
        
        # 检查结果是否是流式任务响应（收到响应不代表完成）
        is_streaming_response = False
        if isinstance(result, str):
            try:
                result_dict = json.loads(result)
                if isinstance(result_dict, dict) and result_dict.get("type") == "streaming_task":
                    is_streaming_response = True
            except:
                pass
        elif isinstance(result, dict) and result.get("type") == "streaming_task":
            is_streaming_response = True
        
        if is_streaming_response:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"⚠️ 工具 '{tool_name}' 返回了流式任务响应（工具执行尚未完成）",
                flush=True
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"📋 响应类型: {result_type}, 大小: {result_size} 字符",
                flush=True
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"ℹ️ 工具实际执行状态：正在执行中，需要等待 SSE 流中的结束消息",
                flush=True
            )
            # 注意：这里只是检测，实际处理在 mcp_tool_async 中完成
            # 因为 _invoke_with_progress 没有 service_id 信息
            # 完整的处理逻辑应该在调用者中实现
        else:
            print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"✅ 工具 '{tool_name}' 执行完成 (总耗时: {total_time_str})",
            flush=True
            )
            print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"📋 结果类型: {result_type}, 大小: {result_size} 字符",
            flush=True
        )
        
        # 明确通知结果已准备好返回
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"🔄 结果已准备好，即将返回给调用者...",
            flush=True
        )
        
        return result
        
    except Exception as e:
        # 取消进度提示任务
        connection_healthy = False
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        
        # 检查是否是连接相关错误
        error_str = str(e)
        error_type_name = type(e).__name__
        elapsed_time = time.time() - start_time
        
        is_connection_lost = (
            "Connection closed" in error_str or
            "Connection close" in error_str or
            "peer closed connection" in error_str or
            "incomplete chunked read" in error_str or
            "RemoteProtocolError" in error_type_name or
            "Connection" in error_str and ("closed" in error_str.lower() or "lost" in error_str.lower())
        )
        
        # 如果连接断开且运行时间较长，工具可能已完成
        if is_connection_lost and elapsed_time > 300:  # 运行超过5分钟
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)
            time_str = f"{elapsed_minutes}分{elapsed_seconds}秒" if elapsed_minutes > 0 else f"{elapsed_seconds}秒"
            
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"⚠️ 检测到连接断开，但工具已运行 {time_str}",
                flush=True
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"💡 工具可能已完成但连接断开，请检查：",
                flush=True
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"   1. MCP服务器日志，确认工具是否完成",
                flush=True
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"   2. 输出文件目录，查看结果文件",
                flush=True
            )
        
        # 重新抛出异常以便上层处理
        raise


async def mcp_tool_async(service_id: str, tool_name: str, params: dict):
    """通用异步MCP工具函数 - 适配0.1.7版本API

    Args:
        service_id: 服务ID (如 'metabcr', 'r_analysis')
        tool_name: 工具名称 (如 'metabcr', 'run_figure2_analysis')
        params: 工具参数字典
    """
    # 在函数开始就定义变量，确保在整个函数作用域内可用
    max_retries = 3
    retry_delay = 2  # 秒
    tools = None
    client = None
    server_url = "未知"

    # 打印当前配置
    from common.factory import get_all_mcp_servers

    all_servers = get_all_mcp_servers()
    print(f"所有可用服务器: {list(all_servers.keys())}")
    if service_id in all_servers:
        print(f"{service_id}服务器配置: {all_servers[service_id]}")

    # 使用封装好的get_mcp_client函数
    config = RunnableConfig(configurable={"mcp_config": {"service_ids": [service_id]}})
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 [mcp_tool_async] 创建 MCP 客户端")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   服务 ID: {service_id}")
    
    # 获取服务器配置信息
    try:
        server_config = ApplicationConfig.get_instance().mcp_servers.get(service_id, {})
        server_url = server_config.get("url", "未知")
    except Exception:
        server_url = "未知（无法获取配置）"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   服务器 URL: {server_url}")
    
    # 添加重试机制获取工具列表
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 [mcp_tool_async] 尝试连接服务器 (第 {attempt}/{max_retries} 次)")
            client = await get_mcp_client(config)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [mcp_tool_async] 客户端创建成功")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 [mcp_tool_async] 开始获取工具列表...")
            tools = await client.get_tools()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [mcp_tool_async] 成功获取工具列表")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   可用工具数量: {len(tools)}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   工具名称列表: {[t.name for t in tools]}")
            break  # 成功，退出重试循环
            
        except Exception as connection_error:
            error_str = str(connection_error)
            error_type = type(connection_error).__name__
            
            # 检查是否是连接相关错误
            is_connection_error = (
                "Connection closed" in error_str or
                "Connection close" in error_str or
                "peer closed connection" in error_str or
                "incomplete chunked read" in error_str or
                "RemoteProtocolError" in error_type or
                "McpError" in error_type or
                "TaskGroup" in error_str or
                "Connection" in error_str and ("closed" in error_str.lower() or "failed" in error_str.lower())
            )
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [mcp_tool_async] 连接失败 (第 {attempt}/{max_retries} 次)")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   错误类型: {error_type}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   错误信息: {error_str[:300]}")
            
            if is_connection_error:
                if attempt < max_retries:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 [mcp_tool_async] 等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [mcp_tool_async] 所有重试尝试均失败")
                    # 抛出详细错误信息
                    raise Exception(
                        f"MCP 服务器连接失败（已重试 {max_retries} 次）\n"
                        f"服务 ID: {service_id}\n"
                        f"服务器 URL: {server_url}\n"
                        f"错误类型: {error_type}\n"
                        f"错误信息: {error_str[:500]}\n\n"
                        f"可能的原因：\n"
                        f"1. MCP 服务器未运行或无法访问\n"
                        f"2. 网络连接问题\n"
                        f"3. SSE 连接被服务器关闭\n"
                        f"4. 服务器资源不足\n\n"
                        f"建议：\n"
                        f"1. 检查 MCP 服务器是否正在运行\n"
                        f"2. 检查网络连接和防火墙设置\n"
                        f"3. 检查服务器日志\n"
                        f"4. 尝试稍后重试"
                    ) from connection_error
            else:
                # 非连接错误，直接抛出
                raise

    if tools is None:
        raise Exception(f"无法获取工具列表，服务 ID: {service_id}")

    try:

        # 查找匹配的工具
        tool = None
        for t in tools:
            if t.name.lower() == tool_name.lower():
                tool = t
                break

        if not tool:
            available_tools_str = ", ".join([t.name for t in tools]) if tools else "无"
            error_msg = (
                f"错误: 找不到工具 '{tool_name}'\n"
                f"服务 ID: {service_id}\n"
                f"可用工具: {available_tools_str}"
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [mcp_tool_async] {error_msg}")
            return error_msg

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 [mcp_tool_async] 开始调用工具")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   工具名称: {tool.name}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   工具参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   服务 ID: {service_id}")
        
        # 首先使用常规方式调用工具
        result = None
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📞 [mcp_tool_async] 使用常规方式调用工具（langchain_mcp_adapters）")
            # 使用带进度提示的工具调用
            initial_result = await _invoke_with_progress(tool, params, tool_name, progress_interval=30)
            result = initial_result
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 [mcp_tool_async] 收到工具调用初始响应")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   ⚠️ 注意：收到响应不代表工具已完成，需要检查是否为流式任务")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果类型: {type(result)}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果是否为 None: {result is None}")
            if result is not None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果大小: {len(str(result))} 字符")
                
                # 检查结果对象的属性
                if hasattr(result, '__dict__'):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果对象属性: {list(result.__dict__.keys())}")
                if hasattr(result, '__class__'):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果对象类: {result.__class__.__name__}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果对象方法: {[m for m in dir(result) if not m.startswith('_')][:10]}")
                
                # 尝试访问可能的属性
                if hasattr(result, 'content'):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   result.content: {result.content}")
                if hasattr(result, 'structuredContent'):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   result.structuredContent: {result.structuredContent}")
                
                # 打印结果的完整内容（限制长度）
                result_preview = str(result)[:2000] if len(str(result)) > 2000 else str(result)
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果内容预览（前2000字符）:\n{result_preview}")
            
            # 检查结果是否包含流式传输信息
            # 结果可能是字符串（JSON 格式）或字典
            streaming_info = None
            result_dict = None
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 [mcp_tool_async] 开始检查结果中是否包含流式传输信息")
            
            # 尝试解析结果为字典
            result_dict = None
            if isinstance(result, str):
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果为字符串类型，尝试解析 JSON")
                try:
                    result_dict = json.loads(result)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   ✅ JSON 解析成功")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   解析后的字典键: {list(result_dict.keys()) if isinstance(result_dict, dict) else 'N/A'}")
                except json.JSONDecodeError as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   ⚠️ JSON 解析失败: {e}")
                    # 尝试检查字符串中是否包含流式信息
                    if "streaming_task" in result or "stream_url" in result or "task_id" in result:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   🔍 字符串中包含流式关键字，尝试提取")
                        # 尝试使用正则或简单字符串解析提取流式信息
                        try:
                            import re
                            # 尝试提取 JSON 对象
                            json_match = re.search(r'\{[^{}]*"type"\s*:\s*"streaming_task"[^{}]*\}', result)
                            if json_match:
                                potential_json = json_match.group(0)
                                result_dict = json.loads(potential_json)
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   ✅ 从字符串中提取到流式信息")
                        except:
                            pass
            elif isinstance(result, dict):
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果为字典类型，直接使用")
                result_dict = result
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   字典键: {list(result_dict.keys())}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果类型为 {type(result)}，尝试转换为字符串后再解析")
                # 尝试将其他类型转换为字符串再解析
                try:
                    result_str = str(result)
                    if "streaming_task" in result_str or "stream_url" in result_str:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   🔍 字符串化结果中包含流式关键字")
                        # 尝试提取 JSON
                        import re
                        json_match = re.search(r'\{.*?"type"\s*:\s*"streaming_task".*?\}', result_str, re.DOTALL)
                        if json_match:
                            try:
                                result_dict = json.loads(json_match.group(0))
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   ✅ 从字符串化结果中提取到流式信息")
                            except:
                                pass
                except:
                    pass
            
            # 检查是否包含流式信息
            if result_dict:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   调用 parse_streaming_response 解析流式信息")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   传入的响应结构: {json.dumps({'result': result_dict}, ensure_ascii=False, indent=2)[:500]}")
                streaming_info = MCPStreamingClient.parse_streaming_response({"result": result_dict})
                if streaming_info:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   ✅ 解析到流式信息: {streaming_info}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   ℹ️ 未解析到流式信息")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   ⚠️ 无法获取字典格式的结果，跳过流式信息检查")
                # 即使没有字典，也尝试在原始结果字符串中搜索
                if result and isinstance(result, str):
                    if "stream_url" in result:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   🔍 在原始字符串中发现 stream_url，尝试提取")
                        try:
                            import re
                            # 提取完整的流式任务 JSON
                            pattern = r'\{"type"\s*:\s*"streaming_task"[^}]+\}'
                            match = re.search(pattern, result)
                            if match:
                                streaming_json = json.loads(match.group(0))
                                streaming_info = {
                                    "task_id": streaming_json.get("task_id"),
                                    "service_id": streaming_json.get("service_id"),  # 新格式
                                    "stream_url": streaming_json.get("stream_url"),  # 旧格式兼容
                                    "message": streaming_json.get("message", "")
                                }
                                print(f"[{datetime.now().strftime('%H:%M:%S')}]   ✅ 从字符串中提取到流式信息: {streaming_info}")
                        except Exception as e:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}]   ⚠️ 从字符串提取流式信息失败: {e}")
            
            # 如果结果中包含流式信息，连接到流式端点接收进度
            if streaming_info:
                task_id = streaming_info.get("task_id")
                stream_service_id = streaming_info.get("service_id")
                existing_stream_url = streaming_info.get("stream_url")  # 旧格式兼容
                
                # 优先使用 service_id + task_id 拼接 URL（新格式）
                # 如果没有 service_id，尝试使用现有的 stream_url（旧格式兼容）
                if stream_service_id or existing_stream_url:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [mcp_tool_async] 检测到流式任务信息")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   ⚠️ 重要：工具调用尚未完成，需要等待 SSE 流中的结束消息")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Task ID: {task_id}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Service ID: {stream_service_id}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   消息: {streaming_info.get('message', '')}")
                    
                    stream_url = None
                    
                    # 新格式：根据 service_id 和 task_id 拼接 stream_url
                    if stream_service_id and task_id:
                        # 获取服务器配置
                        server_config = ApplicationConfig.get_instance().mcp_servers.get(stream_service_id, {})
                        server_url = server_config.get("url", "")
                        
                        # 使用 urlparse 提取基础 URL（只包含 协议://ip:port）
                        from urllib.parse import urlparse
                        server_url_parsed = urlparse(server_url)
                        base_url = f"{server_url_parsed.scheme}://{server_url_parsed.netloc}"
                        
                        # 拼接 stream_url: {base_url}/stream/{task_id}
                        stream_url = f"{base_url}/stream/{task_id}"
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   服务器配置 URL: {server_url}")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   基础 URL: {base_url}")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   拼接的 Stream URL: {stream_url}")
                
                # 旧格式兼容：如果已有 stream_url，使用它（但需要处理 localhost 等）
                elif existing_stream_url:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   ⚠️ 使用旧格式的 stream_url（兼容模式）")
                    stream_url = existing_stream_url
                    original_stream_url = stream_url
                    
                    # 获取服务器配置（使用传入的 service_id 作为后备）
                    server_config = ApplicationConfig.get_instance().mcp_servers.get(service_id, {})
                    server_url = server_config.get("url", "")
                    # 使用 urlparse 提取基础 URL（只包含 协议://ip:port）
                    from urllib.parse import urlparse
                    server_url_parsed = urlparse(server_url)
                    base_url = f"{server_url_parsed.scheme}://{server_url_parsed.netloc}"
                    
                    # 处理 localhost 地址或相对路径
                    if "localhost" in stream_url or "127.0.0.1" in stream_url:
                        # 从服务器配置 URL 中提取主机和端口
                        from urllib.parse import urlparse, urlunparse
                        server_url_parsed = urlparse(server_url)
                        base_scheme = server_url_parsed.scheme
                        base_netloc = server_url_parsed.netloc
                        
                        stream_url_parsed = urlparse(stream_url)
                        new_stream_url = urlunparse((
                            base_scheme,
                            base_netloc,
                            stream_url_parsed.path,
                            stream_url_parsed.params,
                            stream_url_parsed.query,
                            stream_url_parsed.fragment
                        ))
                        stream_url = new_stream_url
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   替换 localhost 为实际服务器地址")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   原始 URL: {original_stream_url}")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   新 URL: {stream_url}")
                    elif stream_url.startswith("/"):
                        stream_url = f"{base_url}{stream_url}"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   补全相对路径 Stream URL: {original_stream_url} -> {stream_url}")
                
                    # 如果既没有 service_id 也没有 stream_url，无法继续
                    if not stream_url:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"❌ 流式任务响应中缺少 service_id/task_id 或 stream_url，无法连接 SSE",
                            flush=True
                        )
                        return result
                
                # 创建流式客户端并接收进度
                try:
                    # 确定使用的 service_id（优先使用响应中的 service_id）
                    actual_service_id = stream_service_id or service_id
                    server_config = ApplicationConfig.get_instance().mcp_servers.get(actual_service_id, {})
                    server_url = server_config.get("url", "")
                    # 使用 urlparse 提取基础 URL（只包含 协议://ip:port）
                    from urllib.parse import urlparse
                    server_url_parsed = urlparse(server_url)
                    base_url = f"{server_url_parsed.scheme}://{server_url_parsed.netloc}"
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 [mcp_tool_async] 创建流式客户端")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   使用的 Service ID: {actual_service_id}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   基础 URL: {base_url}")
                    streaming_client = MCPStreamingClient(base_url, timeout=server_config.get("timeout", 36000))
                    
                    def on_progress(msg):
                        data = msg.get("data", {})
                        progress = data.get("progress_percent", 0)
                        msg_text = data.get("message", "")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 [SSE进度] {progress:.1f}% - {msg_text}", flush=True)
                    
                    def on_result(msg):
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [SSE结果] 任务完成: {msg.get('status', 'success')}", flush=True)
                    
                    def on_error(msg):
                        error_msg = msg.get("message", "Unknown error")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [SSE错误] 任务错误: {error_msg}", flush=True)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🌊 [mcp_tool_async] 开始连接到 SSE 端点接收实时进度")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   ⚠️ 工具执行正在进行中，等待 SSE 流中的结束消息...")
                    
                    # 接收流式进度和最终结果
                    # 只有在收到 SSE 流中的结束消息（type: "end" 或 "result"）时，才认为工具真正完成
                    stream_result = await streaming_client.receive_stream(
                        stream_url,
                        on_progress=on_progress,
                        on_result=on_result,
                        on_error=on_error
                    )
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [mcp_tool_async] SSE 流式传输完成，工具执行已真正完成")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   流式结果类型: {type(stream_result)}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   流式结果内容: {json.dumps(stream_result, ensure_ascii=False, indent=2)[:500]}")
                    
                    # 从流式结果中提取最终结果
                    if stream_result.get("type") == "result":
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   从 SSE 结果消息中提取最终数据")
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
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   最终结果类型: {type(result)}, 大小: {len(str(result)) if result else 0} 字符")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   最终结果内容: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}")
                    elif stream_result.get("type") == "end":
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   收到 SSE 结束消息，工具执行完成")
                        # 如果只有结束消息，保持原始结果
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   使用初始响应中的结果")
                    else:
                        # 保持原有结果
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   流式结果类型: {stream_result.get('type')}，保持原有结果")
                        pass
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔌 [mcp_tool_async] 关闭流式客户端连接")
                    await streaming_client.close()
                except Exception as stream_error:
                    # 如果流式传输失败，使用原始结果
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [mcp_tool_async] 流式传输失败，使用原始结果")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   异常类型: {type(stream_error).__name__}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   异常信息: {str(stream_error)}")
                    # result 已经包含在 initial_result 中，继续使用
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ [mcp_tool_async] 未检测到流式任务信息，使用常规结果")
                if streaming_info:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   流式信息: {streaming_info}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   未找到流式信息")
                    
        except Exception as invoke_error:
            # 如果工具调用抛出异常，检查是否是连接断开
            error_str = str(invoke_error)
            error_type_name = type(invoke_error).__name__
            
            is_connection_lost = (
                "Connection closed" in error_str or
                "Connection close" in error_str or
                "peer closed connection" in error_str or
                "incomplete chunked read" in error_str or
                "RemoteProtocolError" in error_type_name
            )
            
            if is_connection_lost:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"⚠️ 工具执行过程中连接断开，但工具可能在服务器端已完成",
                    flush=True
                )
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"💡 建议：检查MCP服务器日志和输出文件目录，确认工具是否真正完成",
                    flush=True
                )
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"📋 从日志看，如果工具已完成，结果文件路径应该包含在服务器日志中",
                    flush=True
                )
            
            # 重新抛出异常，让上层错误处理
            raise
        
        # 结果确认和处理
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 [mcp_tool_async] 开始处理工具调用结果")
        
        if result is None:
            error_msg = f"工具 '{tool_name}' 返回了 None 值，这可能表示执行失败"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [mcp_tool_async] {error_msg}", flush=True)
            return error_msg
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [mcp_tool_async] 工具调用成功，结果不为空")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果类型: {type(result)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果是否为字典: {isinstance(result, dict)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果是否为字符串: {isinstance(result, str)}")
        
        # 检查结果类型并格式化
        try:
            if isinstance(result, dict):
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果已格式化为 JSON 字符串")
            elif isinstance(result, str):
                result_str = result
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果已是字符串类型")
            else:
                result_str = str(result)
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   结果转换为字符串")
        except Exception as e:
            result_str = str(result)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [mcp_tool_async] 结果序列化警告: {e}", flush=True)
        
        # 显示结果摘要（完整结果会通过返回值传递）
        result_preview = result_str[:500] if len(result_str) > 500 else result_str
        result_length = len(result_str)
        
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] ✅ [mcp_tool_async] 工具 '{tool_name}' 执行完成，结果已返回",
            flush=True
        )
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 📊 [mcp_tool_async] 结果统计: 长度={result_length} 字符",
            flush=True
        )
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 📦 [mcp_tool_async] 结果预览:\n{result_preview}",
            flush=True
        )
        if result_length > 500:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ [mcp_tool_async] 结果已截断显示，完整结果将通过返回值传递",
                flush=True
            )
        
        # 确保结果被正确返回
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 [mcp_tool_async] 正在将结果返回给 agent...",
            flush=True
        )
        
        return result
    except Exception as e:
        import sys
        import traceback
        from httpx import ConnectError, RemoteProtocolError
        from httpcore import ConnectError as HttpCoreConnectError

        # 获取完整的异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_str = str(e)
        error_type_name = type(e).__name__
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ [mcp_tool_async] 工具调用过程发生异常")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   异常类型: {error_type_name}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   异常信息: {error_str[:500]}")
        
        # 检查是否是连接相关的错误（包括超时和连接关闭）
        is_connection_error = (
            isinstance(e, (ConnectError, HttpCoreConnectError, RemoteProtocolError)) or
            "ConnectError" in error_type_name or
            "RemoteProtocolError" in error_type_name or
            "McpError" in error_type_name or
            "TaskGroup" in error_str or
            "ExceptionGroup" in error_type_name or
            "All connection attempts failed" in error_str or
            "Connection" in error_str and ("failed" in error_str.lower() or "closed" in error_str.lower()) or
            "peer closed connection" in error_str.lower() or
            "incomplete chunked read" in error_str.lower() or
            "Connection close" in error_str or
            "Connection closed" in error_str or
            "timeout" in error_str.lower() or
            "Timeout" in error_type_name
        )
        
        if is_connection_error:
            # 获取服务器配置信息（使用顶部已导入的 ApplicationConfig）
            # 如果之前没有获取到 server_url，尝试再次获取
            if server_url == "未知":
                try:
                    config_obj = ApplicationConfig.get_instance()
                    server_config = config_obj.mcp_servers.get(service_id, {})
                    server_url = server_config.get("url", "未知")
                except Exception:
                    server_url = "未知（无法获取配置）"
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 [mcp_tool_async] 检测到连接错误")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   服务器 URL: {server_url}")
            
            # 检查是否是 TaskGroup/ExceptionGroup 错误（SSE 连接问题）
            if "TaskGroup" in error_str or "ExceptionGroup" in error_type_name:
                error_msg = (
                    f"⚠️ MCP SSE 连接错误\n"
                    f"服务ID: {service_id}\n"
                    f"工具名称: {tool_name}\n"
                    f"服务器URL: {server_url}\n"
                    f"错误类型: {error_type_name}\n"
                    f"错误信息: {error_str[:500]}\n\n"
                    f"📋 诊断信息:\n"
                    f"- SSE 连接在建立或读取时被服务器关闭\n"
                    f"- 可能是服务器端主动关闭连接或网络问题\n"
                    f"- 已尝试自动重试 {max_retries} 次但均失败\n\n"
                    f"💡 建议:\n"
                    f"1. 检查 MCP 服务器是否正在运行: {server_url}\n"
                    f"2. 检查服务器日志，查看是否有错误信息\n"
                    f"3. 检查网络连接和防火墙设置\n"
                    f"4. 尝试手动访问服务器 URL 确认服务可用性\n"
                    f"5. 如果服务器正在运行，可能是资源不足或配置问题\n"
                    f"6. 等待一段时间后重试"
                )
            else:
            # 检查是否是长时间运行后的连接断开（可能是工具已完成但连接断开）
                error_msg = (
                f"⚠️ MCP 连接错误（工具可能已完成但连接断开）\n"
                f"服务ID: {service_id}\n"
                f"工具名称: {tool_name}\n"
                f"服务器URL: {server_url}\n"
                f"错误类型: {error_type_name}\n"
                f"错误信息: {error_str[:300]}\n\n"
                f"📋 诊断信息:\n"
                f"- 如果工具运行时间较长（>30分钟），可能已完成但连接已断开\n"
                f"- 请检查 MCP 服务器日志确认工具是否真正完成\n"
                f"- 如果工具已完成，结果可能已保存到输出文件\n"
                f"- 请检查工具的输出目录（通常在日志中会有输出路径）\n\n"
                f"💡 建议:\n"
                f"1. 检查 MCP 服务器日志，查看工具是否已完成\n"
                f"2. 如果工具已完成，检查输出文件目录\n"
                f"3. 检查网络连接和防火墙设置\n"
                f"4. 检查服务器是否仍在运行\n"
                f"5. 考虑增加超时时间或使用轮询机制检查工具状态"
            )
        else:
            error_msg = f"工具调用出错: {error_str}"
        
        detailed_error = traceback.format_exc()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {error_msg}", flush=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [详细] 异常类型: {exc_type}", flush=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [详细] 异常值: {exc_value}", flush=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [详细] 完整堆栈:\n{detailed_error}", flush=True)

        return error_msg
