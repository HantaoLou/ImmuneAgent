from langchain_core.runnables.config import RunnableConfig

from common.factory import get_mcp_client


async def mcp_tool_async(service_id: str, tool_name: str, params: dict):
    """通用异步MCP工具函数 - 适配0.1.7版本API

    Args:
        service_id: 服务ID (如 'metabcr', 'r_analysis')
        tool_name: 工具名称 (如 'metabcr', 'run_figure2_analysis')
        params: 工具参数字典
    """

    # 打印当前配置
    from common.factory import get_all_mcp_servers

    all_servers = get_all_mcp_servers()
    print(f"所有可用服务器: {list(all_servers.keys())}")
    if service_id in all_servers:
        print(f"{service_id}服务器配置: {all_servers[service_id]}")

    # 使用封装好的get_mcp_client函数
    config = RunnableConfig(configurable={"mcp_config": {"service_ids": [service_id]}})
    client = await get_mcp_client(config)
    print(f"工具配置: 使用get_mcp_client获取{service_id}客户端")

    try:
        tools = await client.get_tools()
        print(f"可用工具: {[t.name for t in tools]}")

        # 查找匹配的工具
        tool = None
        for t in tools:
            if t.name.lower() == tool_name.lower():
                tool = t
                break

        if not tool:
            return f"错误: 找不到工具 {tool_name}"

        print(f"开始调用工具 {tool.name}")
        result = await tool.ainvoke(params)
        print(f"工具调用完成: {result}")
        return result
    except Exception as e:
        import sys
        import traceback
        from httpx import ConnectError
        from httpcore import ConnectError as HttpCoreConnectError

        # 获取完整的异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_str = str(e)
        
        # 检查是否是连接错误
        is_connection_error = (
            isinstance(e, (ConnectError, HttpCoreConnectError)) or
            "ConnectError" in str(type(e)) or
            "All connection attempts failed" in error_str or
            "Connection" in error_str and "failed" in error_str.lower()
        )
        
        if is_connection_error:
            # 获取服务器配置信息
            from config.config import ApplicationConfig
            config = ApplicationConfig.get_instance()
            server_config = config.mcp_servers.get(service_id, {})
            server_url = server_config.get("url", "未知")
            
            error_msg = (
                f"MCP 服务器连接失败\n"
                f"服务ID: {service_id}\n"
                f"服务器URL: {server_url}\n"
                f"错误: {error_str}\n\n"
                f"可能的原因:\n"
                f"1. 远程服务器 {server_url.split('//')[1].split('/')[0] if '//' in server_url else '未知'} 不可访问\n"
                f"2. 网络连接问题或防火墙阻止\n"
                f"3. MCP 服务器未运行或已停止\n"
                f"4. 服务器地址配置错误\n\n"
                f"建议:\n"
                f"- 检查网络连接\n"
                f"- 确认 MCP 服务器是否正在运行\n"
                f"- 验证服务器地址和端口是否正确\n"
                f"- 检查防火墙设置"
            )
        else:
            error_msg = f"工具调用出错: {error_str}"
        
        detailed_error = traceback.format_exc()

        print(f"[错误] {error_msg}")
        print(f"[详细] 异常类型: {exc_type}")
        print(f"[详细] 异常值: {exc_value}")
        print(f"[详细] 完整堆栈:\n{detailed_error}")

        return error_msg
