# -*- coding: utf-8 -*-
"""
Common Tools MCP Server - 将 Bio-Agent 的 Common Tools 暴露为 MCP 服务

这个服务器将 agent/tools/ 中的 52 个工具包装成 MCP 协议，
让 OpenCode 可以在沙盒中通过 MCP 协议调用这些工具。

使用方式:
    # 启动服务器
    python -m mcp_servers.common_tools_server --port 40002
    
    # 或使用 uvicorn
    uvicorn mcp_servers.common_tools_server:app --host 0.0.0.0 --port 40002

配置到 OpenCode:
    # opencode.json
    {
        "mcp": {
            "common_tools": {
                "type": "remote",
                "url": "http://your-host:40002/mcp/sse",
                "enabled": true
            }
        }
    }
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MCP 协议数据结构
# ============================================================================

class MCPToolParameter(BaseModel):
    """MCP 工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class MCPToolDefinition(BaseModel):
    """MCP 工具定义"""
    name: str
    description: str
    parameters: List[MCPToolParameter]


class MCPToolCall(BaseModel):
    """MCP 工具调用请求"""
    tool_name: str
    arguments: Dict[str, Any]


class MCPToolResult(BaseModel):
    """MCP 工具调用结果"""
    content: str
    is_error: bool = False


# ============================================================================
# Common Tools 加载器
# ============================================================================

class CommonToolsLoader:
    """加载和包装 Common Tools"""
    
    def __init__(self):
        self._tools_cache: Dict[str, Callable] = {}
        self._tools_definitions: List[MCPToolDefinition] = []
        self._loaded = False
    
    def load_tools(self) -> None:
        """加载所有 common tools"""
        if self._loaded:
            return
        
        try:
            from tools import get_all_common_tools, get_tools_by_category
            self._tools_cache = get_all_common_tools()
            self._build_definitions()
            self._loaded = True
            logger.info(f"已加载 {len(self._tools_cache)} 个 common tools")
        except Exception as e:
            logger.error(f"加载 common tools 失败: {e}")
            raise
    
    def _build_definitions(self) -> None:
        """构建 MCP 工具定义"""
        # 分类映射
        category_info = {
            "search": "搜索工具 - 网页搜索和知识库检索",
            "memory": "记忆工具 - Agent 状态持久化",
            "biomedical": "生物医学数据库 - 基因/疾病/药物/蛋白质查询",
            "literature": "文献搜索 - 学术论文搜索和获取",
            "clinical": "临床数据 - 临床试验和药物基因组学",
            "reference": "参考数据库 - 蛋白质结构、通路、本体论",
            "docx": "文档解析 - DOCX 文件处理",
        }
        
        for tool_name, tool_func in self._tools_cache.items():
            # 提取文档字符串作为描述
            doc = tool_func.__doc__ or "无描述"
            description = doc.strip().split('\n')[0]
            
            # 解析参数（从函数签名）
            parameters = self._extract_parameters(tool_func)
            
            self._tools_definitions.append(MCPToolDefinition(
                name=tool_name,
                description=description,
                parameters=parameters
            ))
    
    def _extract_parameters(self, func: Callable) -> List[MCPToolParameter]:
        """从函数签名提取参数"""
        import inspect
        
        parameters = []
        sig = inspect.signature(func)
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            # 判断是否必需
            required = param.default is inspect.Parameter.empty
            
            # 判断类型
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_name = getattr(param.annotation, "__name__", str(param.annotation))
                type_map = {
                    "str": "string",
                    "int": "integer", 
                    "float": "number",
                    "bool": "boolean",
                    "List": "array",
                    "Dict": "object",
                }
                param_type = type_map.get(type_name, "string")
            
            parameters.append(MCPToolParameter(
                name=param_name,
                type=param_type,
                description=f"{param_name} 参数",
                required=required,
                default=None if required else param.default
            ))
        
        return parameters
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """获取指定工具"""
        return self._tools_cache.get(name)
    
    def get_all_definitions(self) -> List[MCPToolDefinition]:
        """获取所有工具定义"""
        if not self._loaded:
            self.load_tools()
        return self._tools_definitions
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """调用工具"""
        tool_func = self.get_tool(name)
        if not tool_func:
            return MCPToolResult(
                content=f"错误: 未找到工具 '{name}'",
                is_error=True
            )
        
        try:
            # 大多数 common tools 是同步的
            result = tool_func(**arguments)
            
            # 处理不同类型的返回值
            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, ensure_ascii=False, indent=2)
            
            # 截断过长的结果
            max_length = 10000
            if len(content) > max_length:
                content = content[:max_length] + "\n... (结果已截断)"
            
            return MCPToolResult(content=content)
            
        except Exception as e:
            logger.exception(f"工具 {name} 执行失败")
            return MCPToolResult(
                content=f"工具执行错误: {str(e)}",
                is_error=True
            )


# 全局加载器实例
tools_loader = CommonToolsLoader()


# ============================================================================
# MCP SSE 服务器实现
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载工具
    tools_loader.load_tools()
    logger.info(f"Common Tools MCP Server 已启动，共 {len(tools_loader.get_all_definitions())} 个工具")
    yield
    # 关闭时清理
    logger.info("Common Tools MCP Server 已关闭")


app = FastAPI(
    title="Common Tools MCP Server",
    description="将 Bio-Agent Common Tools 暴露为 MCP 服务",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/mcp/sse")
async def mcp_sse_endpoint(request: Request):
    """
    MCP SSE 端点
    
    OpenCode 通过此端点连接 MCP 服务器
    """
    async def event_generator():
        # 发送服务器信息
        yield f"data: {json.dumps({'type': 'server_info', 'name': 'common_tools', 'version': '1.0.0'})}\n\n"
        
        # 发送工具列表
        tools = tools_loader.get_all_definitions()
        tools_data = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": [p.model_dump() for p in t.parameters]
            }
            for t in tools
        ]
        yield f"data: {json.dumps({'type': 'tools_list', 'tools': tools_data})}\n\n"
        
        # 保持连接并监听客户端请求
        try:
            async for line in request.stream():
                if line:
                    try:
                        message = json.loads(line.decode('utf-8').strip())
                        if message.get('type') == 'tool_call':
                            # 处理工具调用
                            tool_name = message.get('tool_name')
                            arguments = message.get('arguments', {})
                            
                            result = await tools_loader.call_tool(tool_name, arguments)
                            
                            yield f"data: {json.dumps({'type': 'tool_result', 'result': result.model_dump()})}\n\n"
                    except json.JSONDecodeError:
                        continue
        except asyncio.CancelledError:
            pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/mcp/call")
async def mcp_call_endpoint(call: MCPToolCall):
    """
    MCP HTTP 调用端点 (非 SSE 模式)
    
    用于简单的工具调用场景
    """
    result = await tools_loader.call_tool(call.tool_name, call.arguments)
    return result.model_dump()


@app.get("/mcp/tools")
async def list_tools():
    """列出所有可用工具"""
    tools = tools_loader.get_all_definitions()
    return {
        "count": len(tools),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": [p.model_dump() for p in t.parameters]
            }
            for t in tools
        ]
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "tools_loaded": len(tools_loader.get_all_definitions())
    }


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Common Tools MCP Server")
    parser.add_argument("--port", type=int, default=40002, help="服务器端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

