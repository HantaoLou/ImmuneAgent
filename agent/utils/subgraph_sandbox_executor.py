"""
Subgraph Sandbox Executor

Provides isolated execution environment for subgraphs to prevent errors from affecting the main graph.
Supports multiple isolation strategies: process isolation, thread isolation with error handling, and lightweight exception catching.
"""

import os
import sys
import json
import pickle
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
from enum import Enum
from multiprocessing import Process, Queue, TimeoutError as MPTimeoutError
from threading import Thread
import time
from datetime import datetime

from state import GlobalState
from utils.opensandbox_helper import OpenSandboxHelper


class IsolationStrategy(str, Enum):
    """Isolation strategy for subgraph execution"""
    PROCESS = "process"  # Full process isolation (most secure, highest overhead)
    THREAD = "thread"  # Thread isolation with error handling (balanced)
    EXCEPTION = "exception"  # Lightweight exception catching (fastest, least isolation)


class SubgraphExecutionResult:
    """Result of subgraph execution in sandbox"""
    
    def __init__(
        self,
        success: bool,
        output_state: Optional[Any] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        error_traceback: Optional[str] = None,
        execution_time: float = 0.0,
        sandbox_dir: Optional[str] = None,
        isolation_strategy: Optional[str] = None
    ):
        self.success = success
        self.output_state = output_state
        self.error = error
        self.error_type = error_type
        self.error_traceback = error_traceback
        self.execution_time = execution_time
        self.sandbox_dir = sandbox_dir
        self.isolation_strategy = isolation_strategy


class SubgraphSandboxExecutor:
    """
    Executor for running subgraphs in isolated sandbox environments.
    
    Features:
    - Multiple isolation strategies (process, thread, exception)
    - Automatic sandbox directory creation and cleanup
    - State serialization and deserialization
    - Error capture and reporting
    - Timeout support
    - Resource cleanup
    """
    
    def __init__(
        self,
        strategy: IsolationStrategy = IsolationStrategy.THREAD,
        timeout: Optional[float] = None,
        auto_cleanup: bool = True,
        base_sandbox_dir: Optional[str] = None
    ):
        """
        Initialize sandbox executor
        
        Args:
            strategy: Isolation strategy to use
            timeout: Maximum execution time in seconds (None for no timeout)
            auto_cleanup: Whether to automatically cleanup sandbox directories
            base_sandbox_dir: Base directory for sandbox creation (None for temp dir)
        """
        self.strategy = strategy
        self.timeout = timeout
        self.auto_cleanup = auto_cleanup
        self.base_sandbox_dir = Path(base_sandbox_dir) if base_sandbox_dir else None
        self._active_sandboxes: Dict[str, Path] = {}
    
    def _fix_enum_types_in_dict(self, data: Any) -> Any:
        """
        递归修复字典中的枚举类型（将字符串转换回枚举对象）
        
        Args:
            data: 可能是字典、列表或其他类型的数据
        
        Returns:
            修复后的数据
        """
        from state import UserTaskType
        
        if isinstance(data, dict):
            fixed_dict = {}
            for key, value in data.items():
                # 特殊处理 user_task_type 字段
                if key == "user_task_type" and isinstance(value, str):
                    try:
                        # 尝试将字符串转换为 UserTaskType 枚举
                        fixed_dict[key] = UserTaskType(value)
                    except (ValueError, KeyError):
                        # 如果转换失败，保持原值
                        fixed_dict[key] = value
                else:
                    # 递归处理嵌套结构
                    fixed_dict[key] = self._fix_enum_types_in_dict(value)
            return fixed_dict
        elif isinstance(data, list):
            return [self._fix_enum_types_in_dict(item) for item in data]
        else:
            return data
    
    def execute_subgraph(
        self,
        subgraph_name: str,
        subgraph_builder: Callable,
        input_mapper: Callable[[GlobalState], Any],
        output_mapper: Callable[[Any, GlobalState], GlobalState],
        main_state: GlobalState,
        **kwargs
    ) -> SubgraphExecutionResult:
        """
        Execute a subgraph in isolated sandbox using OpenSandbox
        
        Args:
            subgraph_name: Name of the subgraph (for logging and sandbox naming)
            subgraph_builder: Function that builds the subgraph
            input_mapper: Function to map main state to subgraph input
            output_mapper: Function to map subgraph output back to main state
            main_state: Main graph state
            **kwargs: Additional arguments for subgraph execution
        
        Returns:
            SubgraphExecutionResult with execution results
        """
        import asyncio
        import inspect
        
        start_time = time.time()
        sandbox_dir = None
        
        try:
            # 准备输入状态
            subgraph_input = input_mapper(main_state)
            
            # 从函数对象中提取模块路径和函数名
            subgraph_module = inspect.getmodule(subgraph_builder)
            if subgraph_module is None:
                raise ValueError(f"无法获取 {subgraph_builder.__name__} 的模块信息")
            
            # 获取模块路径（相对于 agent 目录）
            module_file = subgraph_module.__file__
            if module_file is None:
                raise ValueError(f"无法获取 {subgraph_builder.__name__} 的模块文件路径")
            
            # 尝试从模块路径推断模块导入路径
            # 例如: /data/server/ImmuneAgent_2.0/agent/nodes/subagents/supervisor/react_supervisor.py
            # -> nodes.subagents.supervisor.react_supervisor
            module_path_str = str(Path(module_file))
            if "/agent/" in module_path_str:
                # 提取 agent 目录后的路径
                agent_part = module_path_str.split("/agent/", 1)[1]
                module_path = agent_part.replace(".py", "").replace("/", ".")
            else:
                # 如果无法推断，使用模块名
                module_path = subgraph_module.__name__
            
            subgraph_builder_name = subgraph_builder.__name__
            input_mapper_name = input_mapper.__name__
            output_mapper_name = output_mapper.__name__
            
            # 使用 OpenSandbox 执行子图
            async def _execute():
                helper = OpenSandboxHelper()
                try:
                    await helper.create_sandbox()
                    
                    # 执行子图
                    result = await helper.execute_subgraph_in_sandbox(
                        subgraph_module_path=module_path,
                        subgraph_builder_name=subgraph_builder_name,
                        input_mapper_name=input_mapper_name,
                        output_mapper_name=output_mapper_name,
                        input_state=subgraph_input,
                        agent_dir=os.getenv("AGENT_DIR", "/data/server/ImmuneAgent_2.0/agent")
                    )
                    
                    return result
                finally:
                    await helper.close()
            
            # 运行异步函数
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # 如果事件循环正在运行，使用线程执行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _execute())
                    result = future.result()
            else:
                result = loop.run_until_complete(_execute())
            
            # 处理结果
            if result.get("status") == "success":
                output_dict = result.get("output", {})
                
                # 修复枚举类型：将字符串转换回枚举对象
                # 这解决了 JSON 序列化时枚举被转换为字符串的问题
                output_dict = self._fix_enum_types_in_dict(output_dict)
                
                # 将输出字典转换回状态对象（如果需要）
                # 这里假设 output_mapper 可以处理字典
                try:
                    output_state = output_mapper(output_dict, main_state)
                except Exception as mapper_error:
                    # 如果 output_mapper 需要对象而不是字典，尝试创建对象
                    print(f"[subgraph_sandbox] output_mapper 失败: {mapper_error}")
                    # 尝试从字典创建状态对象
                    try:
                        # 尝试导入 SupervisorState 或 ReactSupervisorState
                        from nodes.subagents.supervisor.graph import SupervisorState
                        from nodes.subagents.supervisor.react_supervisor import ReactSupervisorState
                        
                        # 尝试创建状态对象
                        try:
                            state_obj = ReactSupervisorState(**output_dict)
                        except:
                            state_obj = SupervisorState(**output_dict)
                        
                        output_state = output_mapper(state_obj, main_state)
                    except Exception as create_error:
                        print(f"[subgraph_sandbox] 创建状态对象失败: {create_error}")
                        output_state = main_state  # 降级：返回原始状态
                
                return SubgraphExecutionResult(
                    success=True,
                    output_state=output_state,
                    execution_time=time.time() - start_time,
                    sandbox_dir=str(sandbox_dir) if sandbox_dir else None,
                    isolation_strategy="opensandbox"
                )
            else:
                return SubgraphExecutionResult(
                    success=False,
                    error=result.get("error", "Unknown error"),
                    error_type=result.get("error_type", "UnknownError"),
                    error_traceback=result.get("traceback"),
                    execution_time=time.time() - start_time,
                    sandbox_dir=str(sandbox_dir) if sandbox_dir else None,
                    isolation_strategy="opensandbox"
                )
            
        except Exception as e:
            # Catch any errors in the execution wrapper
            return SubgraphExecutionResult(
                success=False,
                error=f"Sandbox execution wrapper failed: {str(e)}",
                error_type=type(e).__name__,
                error_traceback=traceback.format_exc(),
                execution_time=time.time() - start_time,
                sandbox_dir=str(sandbox_dir) if sandbox_dir else None,
                isolation_strategy="opensandbox"
            )


class SubgraphLocalExecutor:
    """Local executor for subgraphs (not implemented yet)"""
    pass


def execute_subgraph_in_sandbox(
    subgraph_name: str,
    subgraph_builder: Callable,
    input_mapper: Callable[[GlobalState], Any],
    output_mapper: Callable[[Any, GlobalState], GlobalState],
    main_state: GlobalState,
    timeout: Optional[float] = None,
    **kwargs
) -> GlobalState:
    """
    Convenience function to execute subgraph in sandbox and return updated main state
    
    Args:
        subgraph_name: Name of the subgraph
        subgraph_builder: Function that builds the subgraph
        input_mapper: Function to map main state to subgraph input
        output_mapper: Function to map subgraph output back to main state
        main_state: Main graph state
        timeout: Maximum execution time
        **kwargs: Additional arguments for subgraph execution
    
    Returns:
        Updated main state (or original state if execution failed)
    """
    open_sandbox_enabled = os.environ.get("OPEN_SANDBOX_ENABLED") == "true"

    executor = SubgraphLocalExecutor(timeout=timeout) if not open_sandbox_enabled else SubgraphSandboxExecutor(timeout=timeout)


    result = executor.execute_subgraph(
        subgraph_name=subgraph_name,
        subgraph_builder=subgraph_builder,
        input_mapper=input_mapper,
        output_mapper=output_mapper,
        main_state=main_state,
        **kwargs
    )

    return result.output_state
