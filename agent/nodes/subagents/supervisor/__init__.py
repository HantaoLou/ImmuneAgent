"""Supervisor subagent module

提供两个版本:
1. graph_refactored.py - 重构版本（推荐，~600行，通过 CodeAct 子图执行）
2. graph.py - 原始版本（已废弃，2249行，直接执行 OpenSandbox 代码）

架构原则：
- supervisor 不直接与 OpenSandbox 沟通
- 所有沙盒操作通过 CodeAct 子图执行
- main_graph.py 默认使用重构版本

使用方式:
    # 推荐方式（main_graph.py 已使用）
    from nodes.subagents.supervisor.graph_refactored import (
        build_supervisor_subgraph,
        supervisor_input_mapper,
        supervisor_output_mapper,
        SupervisorState,
    )
    
    # 废弃方式（仅用于向后兼容）
    from nodes.subagents.supervisor import build_supervisor_subgraph  # 会触发废弃警告
"""

import warnings

# 默认使用重构版本（遵循架构原则：唯一与 OpenSandbox 沟通的入口是 CodeAct）
try:
    from .graph_refactored import (
        build_supervisor_subgraph,
        supervisor_input_mapper,
        supervisor_output_mapper,
        SupervisorState,
    )
    _USING_REFACTORED = True
except ImportError:
    # 回退到原始版本（并发出警告）
    warnings.warn(
        "graph_refactored.py 导入失败，回退到已废弃的 graph.py。"
        "原始版本直接调用 OpenSandbox，违反架构原则。"
        "请确保 graph_refactored.py 可用。",
        DeprecationWarning,
        stacklevel=2
    )
    from .graph import (
        build_supervisor_subgraph,
        supervisor_input_mapper,
        supervisor_output_mapper,
        SupervisorState,
    )
    _USING_REFACTORED = False

__all__ = [
    # 当前使用的版本（默认是重构版本）
    "build_supervisor_subgraph",
    "supervisor_input_mapper",
    "supervisor_output_mapper",
    "SupervisorState",
    # 版本标识
    "_USING_REFACTORED",
]
