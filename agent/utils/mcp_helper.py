"""
MCP工具调用辅助模块

提供统一的MCP工具调用接口，供生成的代码使用。
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# 添加agent目录到路径
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))


# 全局MCP客户端缓存
_mcp_client_cache: Optional[Any] = None
_mcp_tools_cache: Dict[str, Any] = {}
# 工具名称到服务ID的映射（从 mcp_tools.json 加载）
_tool_to_service_map: Dict[str, str] = {}
# 单服务器客户端缓存（按服务ID索引）
_single_server_clients: Dict[str, Any] = {}


def _load_tool_to_service_map() -> Dict[str, str]:
    """
    加载工具名称到服务ID的映射
    
    Returns:
        工具名称到服务ID的字典映射
    """
    global _tool_to_service_map
    
    if _tool_to_service_map:
        return _tool_to_service_map
    
    try:
        mcp_tools_path = agent_dir / "config" / "mcp_tools.json"
        if not mcp_tools_path.exists():
            print(f"  ⚠ mcp_tools.json 不存在，无法建立工具到服务的映射")
            return {}
        
        with open(mcp_tools_path, "r", encoding="utf-8") as f:
            mcp_tools = json.load(f)
        
        # 处理两种格式：直接列表或包含 "mcp_tools" 键的字典
        if isinstance(mcp_tools, dict) and "mcp_tools" in mcp_tools:
            tools_list = mcp_tools["mcp_tools"]
        elif isinstance(mcp_tools, list):
            tools_list = mcp_tools
        else:
            print(f"  ⚠ mcp_tools.json 格式不正确")
            return {}
        
        # 建立映射：工具名称 -> 服务ID
        for tool in tools_list:
            tool_name = tool.get("name", "")
            service_id = tool.get("service", "")
            
            if tool_name and service_id:
                # 支持多种工具名称格式：
                # 1. 直接工具名称：search_airr_repertoires
                # 2. 带服务前缀：airr_search_airr_repertoires
                _tool_to_service_map[tool_name] = service_id
                _tool_to_service_map[f"{service_id}_{tool_name}"] = service_id
        
        print(f"  ✓ 已加载 {len(_tool_to_service_map)} 个工具到服务的映射")
        return _tool_to_service_map
    except Exception as e:
        print(f"  ⚠ 加载工具到服务映射失败: {e}")
        return {}


def _get_service_id_for_tool(tool_name: str) -> Optional[str]:
    """
    根据工具名称获取对应的服务ID
    
    Args:
        tool_name: 工具名称（可能是 search_airr_repertoires 或 airr_search_airr_repertoires）
    
    Returns:
        服务ID，如果未找到则返回 None
    """
    # 确保映射已加载
    if not _tool_to_service_map:
        _load_tool_to_service_map()
    
    # 1. 先尝试精确匹配
    if tool_name in _tool_to_service_map:
        return _tool_to_service_map[tool_name]
    
    # 2. 尝试匹配带服务前缀的格式（service_tool_name）
    # 例如：airr_search_airr_repertoires -> search_airr_repertoires -> airr
    if "_" in tool_name:
        parts = tool_name.split("_", 1)  # 只分割一次，保留后面的部分
        if len(parts) == 2:
            potential_service = parts[0]
            base_tool_name = parts[1]
            
            # 检查 base_tool_name 是否在映射中
            if base_tool_name in _tool_to_service_map:
                return _tool_to_service_map[base_tool_name]
            
            # 检查 potential_service 是否是有效的服务ID
            if potential_service in _tool_to_service_map.values():
                return potential_service
    
    # 3. 尝试部分匹配（工具名称可能包含服务前缀）
    # 例如：airr_search_airr_repertoires 包含 search_airr_repertoires
    for mapped_name, service_id in _tool_to_service_map.items():
        # 检查工具名称是否以映射名称结尾，或者映射名称是否在工具名称中
        if tool_name.endswith(mapped_name) or mapped_name in tool_name:
            # 确保不是误匹配（例如 "search" 匹配到 "search_airr_repertoires"）
            if len(mapped_name) > 5:  # 只匹配较长的工具名称
                return service_id
    
    # 4. 尝试从工具名称开头提取服务ID（格式：service_...）
    if "_" in tool_name:
        parts = tool_name.split("_")
        potential_service = parts[0]
        # 验证是否是有效的服务ID（在映射的值中）
        if potential_service in _tool_to_service_map.values():
            return potential_service
    
    return None


async def _get_single_server_client(service_id: str) -> Any:
    """
    获取单个服务器的MCP客户端（按需创建和缓存）
    
    Args:
        service_id: 服务ID
    
    Returns:
        MultiServerMCPClient实例（只连接该服务器）
    """
    global _single_server_clients
    
    # 检查缓存
    if service_id in _single_server_clients:
        return _single_server_clients[service_id]
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        # 加载MCP服务器配置
        mcp_servers_path = agent_dir / "config" / "mcp_servers.json"
        if not mcp_servers_path.exists():
            raise FileNotFoundError(f"MCP服务器配置文件不存在: {mcp_servers_path}")
        
        with open(mcp_servers_path, "r", encoding="utf-8") as f:
            all_servers = json.load(f)
        
        # 只获取指定服务器的配置
        if service_id not in all_servers:
            raise ValueError(f"服务 {service_id} 不在配置中")
        
        server_config = {service_id: all_servers[service_id]}
        
        # 创建单服务器客户端
        client = MultiServerMCPClient(connections=server_config)
        _single_server_clients[service_id] = client
        
        print(f"  ✓ 已创建单服务器客户端: {service_id}")
        return client
    except Exception as e:
        raise RuntimeError(f"创建单服务器客户端失败 ({service_id}): {e}")


async def _get_mcp_client():
    """
    获取MCP客户端（单例模式）
    
    Returns:
        MultiServerMCPClient实例
    """
    global _mcp_client_cache
    
    if _mcp_client_cache is not None:
        return _mcp_client_cache
    
    try:
        # 正确的导入路径：从 langchain_mcp_adapters.client 导入
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        # 提供更详细的错误信息，包括 Python 路径和已安装的包
        import sys
        error_msg = (
            f"无法导入 langchain-mcp-adapters。\n"
            f"错误详情: {str(e)}\n"
            f"Python 可执行文件: {sys.executable}\n"
            f"Python 路径: {sys.path[:3]}...\n"
            f"请确保已安装: pip install langchain-mcp-adapters 或 uv pip install langchain-mcp-adapters"
        )
        raise ImportError(error_msg)
    
    try:
        # 加载MCP服务器配置
        mcp_servers_path = agent_dir / "config" / "mcp_servers.json"
        if not mcp_servers_path.exists():
            raise FileNotFoundError(f"MCP服务器配置文件不存在: {mcp_servers_path}")
        
        with open(mcp_servers_path, "r", encoding="utf-8") as f:
            mcp_servers = json.load(f)
        
        # 创建多服务器MCP客户端
        _mcp_client_cache = MultiServerMCPClient(connections=mcp_servers)
        
        return _mcp_client_cache
    except Exception as e:
        raise RuntimeError(f"初始化MCP客户端失败: {e}")


async def _get_all_tools():
    """
    获取所有MCP工具
    
    Returns:
        工具列表（BaseTool实例列表）
    """
    global _mcp_tools_cache
    
    if _mcp_tools_cache:
        return list(_mcp_tools_cache.values())
    
    try:
        client = await _get_mcp_client()
        print(f"  ℹ MCP客户端已创建，准备获取工具...")
        
        # 获取工具（可能需要配置，但先尝试无配置方式）
        tools = []
        try:
            print(f"  ℹ 尝试批量获取所有MCP工具...")
            tools = await client.get_tools()
            print(f"  ✓ 成功批量获取 {len(tools)} 个工具")
        except TypeError as type_err:
            # 如果 get_tools 需要参数，尝试使用默认配置
            print(f"  ℹ get_tools 需要参数，尝试使用默认配置...")
            try:
                from langchain_core.runnables import RunnableConfig
                config = RunnableConfig()
                tools = await client.get_tools(config)
                print(f"  ✓ 使用配置成功获取 {len(tools)} 个工具")
            except Exception as config_err:
                # 即使使用配置也失败，继续到降级方案
                print(f"  ⚠ 使用配置也失败: {type(config_err).__name__}: {str(config_err)}")
                raise type_err from config_err
        except Exception as e:
            # 处理 TaskGroup 异常和其他连接错误
            error_msg = str(e)
            error_type = type(e).__name__
            
            # 如果是 TaskGroup 或 ExceptionGroup 异常，尝试提取子异常信息
            if "TaskGroup" in error_type or "TaskGroup" in error_msg or "ExceptionGroup" in error_type:
                import traceback
                tb = traceback.format_exc()
                
                # 尝试提取更详细的错误信息
                # Python 3.11+ 使用 ExceptionGroup
                if hasattr(e, 'exceptions'):
                    sub_exceptions = e.exceptions
                    detailed_errors = []
                    for i, sub_e in enumerate(sub_exceptions):
                        detailed_errors.append(f"  [{i+1}] {type(sub_e).__name__}: {str(sub_e)}")
                    error_msg = f"TaskGroup/ExceptionGroup错误，{len(sub_exceptions)} 个子异常:\n" + "\n".join(detailed_errors)
                    print(f"  ⚠ 检测到 {len(sub_exceptions)} 个子异常，详细信息:")
                    for i, sub_e in enumerate(sub_exceptions):
                        print(f"     子异常 [{i+1}]: {type(sub_e).__name__}: {str(sub_e)[:200]}")
                # 检查是否有 __cause__ 或 __context__
                elif hasattr(e, '__cause__') and e.__cause__:
                    error_msg = f"TaskGroup错误，原因: {type(e.__cause__).__name__}: {str(e.__cause__)}"
                    print(f"  ⚠ 错误原因: {type(e.__cause__).__name__}: {str(e.__cause__)[:200]}")
                elif hasattr(e, '__context__') and e.__context__:
                    error_msg = f"TaskGroup错误，上下文: {type(e.__context__).__name__}: {str(e.__context__)}"
                    print(f"  ⚠ 错误上下文: {type(e.__context__).__name__}: {str(e.__context__)[:200]}")
                else:
                    error_msg = f"TaskGroup错误: {error_msg}"
                    print(f"  ⚠ 完整错误堆栈:\n{tb[:500]}")
            
            # 如果某些服务器连接失败，尝试逐个服务器获取工具（优雅降级）
            print(f"  ⚠ 批量获取工具失败 ({error_type}): {error_msg}")
            print(f"  ℹ 尝试逐个服务器连接...")
            
            # 尝试从配置中获取服务器列表，逐个连接
            try:
                mcp_servers_path = agent_dir / "config" / "mcp_servers.json"
                if mcp_servers_path.exists():
                    import json
                    with open(mcp_servers_path, "r", encoding="utf-8") as f:
                        mcp_servers = json.load(f)
                    
                    # 逐个服务器尝试连接
                    for server_id, server_config in mcp_servers.items():
                        try:
                            from langchain_mcp_adapters.client import MultiServerMCPClient
                            single_client = MultiServerMCPClient(connections={server_id: server_config})
                            server_tools = await single_client.get_tools()
                            tools.extend(server_tools)
                            print(f"  ✓ 成功连接服务器 {server_id}，获取 {len(server_tools)} 个工具")
                        except Exception as server_e:
                            print(f"  ⚠ 服务器 {server_id} 连接失败: {type(server_e).__name__}: {str(server_e)}")
                            continue
                    
                    if tools:
                        print(f"  ✓ 总共获取 {len(tools)} 个工具（来自部分服务器）")
                    else:
                        raise RuntimeError(f"所有MCP服务器连接失败。最后一个错误: {error_msg}")
                else:
                    raise RuntimeError(f"获取MCP工具失败: {error_msg}")
            except Exception as fallback_e:
                raise RuntimeError(f"获取MCP工具失败: {error_msg}。降级方案也失败: {str(fallback_e)}")
        
        # 缓存工具（按名称索引）
        for tool in tools:
            _mcp_tools_cache[tool.name] = tool
        
        if not tools:
            raise RuntimeError("未获取到任何MCP工具")
        
        return tools
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        raise RuntimeError(f"获取MCP工具失败 ({error_type}): {error_msg}")


def get_mcp_tool(tool_name: str) -> Optional[Any]:
    """
    同步获取MCP工具（从缓存中）
    
    Args:
        tool_name: 工具名称（可以是完整名称，如 "airr_search_airr_repertoires"）
    
    Returns:
        BaseTool实例，如果未找到则返回None
    """
    # 先尝试精确匹配
    if tool_name in _mcp_tools_cache:
        return _mcp_tools_cache[tool_name]
    
    # 尝试部分匹配（工具名称可能包含service前缀）
    for cached_name, tool in _mcp_tools_cache.items():
        if cached_name.endswith(tool_name) or tool_name in cached_name:
            return tool
    
    return None


async def invoke_mcp_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    调用MCP工具（异步）
    
    优化：只连接包含该工具的特定服务器，而不是所有服务器。
    
    Args:
        tool_name: 工具名称
        parameters: 工具参数
        config: RunnableConfig配置（可选）
    
    Returns:
        执行结果字典，格式：
        {
            "status": "success" 或 "failed",
            "output": <执行结果>,
            "error": <错误信息，如果有>
        }
    """
    try:
        # 1. 根据工具名称确定服务ID
        service_id = _get_service_id_for_tool(tool_name)
        
        if service_id:
            print(f"  ℹ 工具 {tool_name} 属于服务 {service_id}，只连接该服务器")
            
            # 2. 只连接该服务器
            try:
                client = await _get_single_server_client(service_id)
                
                # 3. 从该服务器获取工具
                try:
                    tools = await client.get_tools()
                except TypeError:
                    from langchain_core.runnables import RunnableConfig
                    runnable_config = RunnableConfig()
                    tools = await client.get_tools(runnable_config)
                
                # 4. 查找目标工具
                tool = None
                for t in tools:
                    # 支持多种工具名称格式匹配
                    if (t.name == tool_name or 
                        t.name.endswith(tool_name) or 
                        tool_name in t.name):
                        tool = t
                        break
                
                if tool is None:
                    return {
                        "status": "failed",
                        "output": None,
                        "error": f"在服务 {service_id} 中未找到工具: {tool_name}。可用工具: {[t.name for t in tools][:10]}..."
                    }
                
                # 5. 调用工具
                from langchain_core.runnables import RunnableConfig
                runnable_config = config or RunnableConfig()
                
                # 所有 MCP 工具统一使用 args 字段包装参数
                wrapped_params = {"args": parameters}
                result = await tool.ainvoke(wrapped_params, runnable_config)
                
                # 6. 缓存工具（供后续使用）
                _mcp_tools_cache[tool.name] = tool
                
                return {
                    "status": "success",
                    "output": result,
                    "error": None
                }
            except Exception as server_e:
                # 如果单服务器连接失败，降级到全服务器连接
                print(f"  ⚠ 单服务器连接失败: {type(server_e).__name__}: {str(server_e)}")
                print(f"  ℹ 降级到全服务器连接...")
                # 继续到降级方案
        else:
            print(f"  ⚠ 无法确定工具 {tool_name} 的服务ID，使用全服务器连接")
        
        # 降级方案：使用全服务器连接（原有逻辑）
        # 确保工具已加载
        if not _mcp_tools_cache:
            await _get_all_tools()
        
        # 获取工具
        tool = get_mcp_tool(tool_name)
        if tool is None:
            # 尝试重新加载工具
            await _get_all_tools()
            tool = get_mcp_tool(tool_name)
        
        if tool is None:
            return {
                "status": "failed",
                "output": None,
                "error": f"未找到工具: {tool_name}。可用工具: {list(_mcp_tools_cache.keys())[:10]}..."
            }
        
        # 准备配置
        from langchain_core.runnables import RunnableConfig
        runnable_config = config or RunnableConfig()
        
        # 所有 MCP 工具统一使用 args 字段包装参数
        wrapped_params = {"args": parameters}
        result = await tool.ainvoke(wrapped_params, runnable_config)
        
        return {
            "status": "success",
            "output": result,
            "error": None
        }
    except Exception as e:
        return {
            "status": "failed",
            "output": None,
            "error": f"调用工具 {tool_name} 失败: {str(e)}"
        }


def invoke_mcp_tool_sync(
    tool_name: str,
    parameters: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    同步调用MCP工具（在同步代码中使用）
    
    Args:
        tool_name: 工具名称
        parameters: 工具参数
        config: RunnableConfig配置（可选）
    
    Returns:
        执行结果字典
    """
    try:
        # 在新的事件循环中运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                invoke_mcp_tool(tool_name, parameters, config)
            )
            return result
        finally:
            loop.close()
    except RuntimeError:
        # 如果已有事件循环在运行，尝试使用它
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环正在运行，创建任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        invoke_mcp_tool(tool_name, parameters, config)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    invoke_mcp_tool(tool_name, parameters, config)
                )
        except Exception as e:
            return {
                "status": "failed",
                "output": None,
                "error": f"同步调用工具失败: {str(e)}"
            }

