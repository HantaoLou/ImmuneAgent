# -*- coding: utf-8 -*-
"""
MCP Servers - Bio-Agent 的 MCP 服务器模块

提供将本地功能暴露为 MCP 服务的服务器实现。

可用服务器:
    - common_tools_server: 将 agent/tools/ 中的 52 个工具暴露为 MCP 服务
"""

from mcp_servers.common_tools_server import app as common_tools_app

