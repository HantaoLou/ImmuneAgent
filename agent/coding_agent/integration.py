"""
Coding Agent 集成模块

提供与 Bio-Agent main_graph 的集成接口，支持：
1. 完整的 Coding Agent 执行流程
2. 同步和异步执行接口
3. 与 GlobalState 的无缝集成

使用方式：
    from coding_agent import run_coding_agent_in_sandbox
    
    result = await run_coding_agent_in_sandbox(
        tasks_md_content=tasks_md,
        context=context,
        session_id=state.session_id,
    )
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from coding_agent.config import (
    ExecutionResult,
    ExecutionStatus,
    OpenCodeConfig,
    OpenCodeMode,
    TaskContext,
)
from coding_agent.opencode_executor import OpenCodeExecutor
from coding_agent.tasks_md_generator import generate_tasks_md_content

if TYPE_CHECKING:
    from state import GlobalState, SubTask


async def run_coding_agent_in_sandbox(
    tasks_md_content: str,
    context: TaskContext,
    session_id: str,
    model_provider: str = "glm-4.7",
    api_key: Optional[str] = None,
    config: Optional[OpenCodeConfig] = None,
    sandbox_image: Optional[str] = None,
    keep_alive: bool = False,
    output_base_dir: str = "/data/sessions",
    download_results: bool = True,
) -> Dict[str, Any]:
    """
    完整的 Coding Agent 执行流程
    
    Args:
        tasks_md_content: tasks.md 文件内容
        context: 任务上下文（参数表、文件路径等）
        session_id: 会话 ID
        model_provider: 模型提供商（glm-4.7, claude-sonnet-4, gpt-4o 等）
        api_key: API 密钥
        config: OpenCode 配置（如果提供，其他参数将被忽略）
        sandbox_image: Docker 镜像名称
        keep_alive: 是否保持沙盒存活
        output_base_dir: 输出文件的基础目录（统一位置）
        download_results: 是否下载结果到统一目录
    
    Returns:
        执行结果字典，包含:
        - status: 执行状态
        - output_dir: 统一输出目录
        - output_files: 输出文件列表
        - summary: 执行摘要
    """
    # 准备配置
    if not config:
        config = OpenCodeConfig(
            model_provider=model_provider,
            api_key=api_key or os.getenv("GLM_API_KEY"),
            sandbox_image=sandbox_image or os.getenv(
                "OPENSANDBOX_IMAGE",
                "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.1"
            ),
        )
    
    executor = OpenCodeExecutor(config)
    
    # 统一输出目录
    unified_output_dir = f"{output_base_dir}/{session_id}/output"
    
    try:
        # 1. 创建沙盒
        print(f"[Coding Agent] 创建沙盒 (session: {session_id})...")
        sandbox = await executor.create_sandbox(config.sandbox_image)
        
        # 2. 准备工作目录结构
        workspace = f"/tmp/sessions/{session_id}"
        await sandbox.commands.run(f"mkdir -p {workspace}/{{input,output,.agent,reports}}")
        
        # 3. 上传 tasks.md
        tasks_path = f"{workspace}/.agent/tasks.md"
        await sandbox.files.write_file(tasks_path, tasks_md_content)
        print(f"[Coding Agent] tasks.md 已上传: {tasks_path}")
        
        # 4. 上传上下文（参数表）
        context_path = f"{workspace}/.agent/context.json"
        await executor.upload_context(context.to_dict(), context_path)
        
        # 5. 上传输入文件（如果有）
        if context.file_paths:
            for file_path in context.file_paths:
                # 假设文件路径是服务器路径，转换为容器路径
                container_path = file_path.replace("/data/sessions/", "/tmp/sessions/", 1)
                # 如果文件在本地，上传；否则假设已经在沙盒中
                if Path(file_path).exists():
                    await executor.upload_file(file_path, container_path)
        
        # 6. 执行任务
        print(f"[Coding Agent] 开始执行任务...")
        result = await executor.execute_tasks(
            tasks_md_path=tasks_path,
            workspace_dir=workspace,
            mode=config.opencode_mode,
        )
        
        # 7. 下载结果到统一目录
        downloaded_files = []
        if download_results and result.output_files:
            print(f"[Coding Agent] 下载结果到统一目录: {unified_output_dir}")
            
            # 确保统一输出目录存在
            Path(unified_output_dir).mkdir(parents=True, exist_ok=True)
            
            for sandbox_file in result.output_files:
                try:
                    # 提取文件名
                    file_name = Path(sandbox_file).name
                    local_path = f"{unified_output_dir}/{file_name}"
                    
                    # 下载文件
                    await executor.download_file(sandbox_file, local_path)
                    downloaded_files.append(local_path)
                    print(f"  - 下载: {sandbox_file} -> {local_path}")
                except Exception as e:
                    print(f"  - 下载失败 {sandbox_file}: {e}")
        
        # 8. 读取执行摘要
        summary = result.summary or {}
        
        # 9. 添加下载文件信息到摘要
        summary["downloaded_files"] = downloaded_files
        summary["unified_output_dir"] = unified_output_dir
        
        print(f"[Coding Agent] 执行完成: {result.status.value}")
        print(f"[Coding Agent] 输出目录: {unified_output_dir}")
        
        return {
            "status": result.status.value,
            "output": summary,
            "output_dir": unified_output_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
            "sandbox_id": result.sandbox_id,
            "execution_time_ms": result.execution_time_ms,
            "output_files": result.output_files,
            "downloaded_files": downloaded_files,
            "completed_tasks": result.completed_tasks,
            "failed_tasks": result.failed_tasks,
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "sandbox_id": getattr(executor.sandbox, 'id', None) if executor.sandbox else None,
            "output_dir": unified_output_dir,
        }
        
    finally:
        if not keep_alive:
            await executor.cleanup()


def run_coding_agent_sync(
    tasks_md_content: str,
    context: TaskContext,
    session_id: str,
    model_provider: str = "glm-4.7",
    api_key: Optional[str] = None,
    config: Optional[OpenCodeConfig] = None,
    sandbox_image: Optional[str] = None,
    keep_alive: bool = False,
) -> Dict[str, Any]:
    """
    同步版本的 Coding Agent 执行
    
    Args:
        同 run_coding_agent_in_sandbox
    
    Returns:
        执行结果字典
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    run_coding_agent_in_sandbox(
                        tasks_md_content=tasks_md_content,
                        context=context,
                        session_id=session_id,
                        model_provider=model_provider,
                        api_key=api_key,
                        config=config,
                        sandbox_image=sandbox_image,
                        keep_alive=keep_alive,
                    ),
                )
                return future.result()
    except RuntimeError:
        pass
    
    return asyncio.run(
        run_coding_agent_in_sandbox(
            tasks_md_content=tasks_md_content,
            context=context,
            session_id=session_id,
            model_provider=model_provider,
            api_key=api_key,
            config=config,
            sandbox_image=sandbox_image,
            keep_alive=keep_alive,
        )
    )


async def run_coding_agent_from_state(
    state: "GlobalState",
    config: Optional[OpenCodeConfig] = None,
    keep_alive: bool = False,
) -> Dict[str, Any]:
    """
    从 GlobalState 运行 Coding Agent
    
    这是与 main_graph.py 集成的主要接口。
    
    Args:
        state: GlobalState 实例
        config: OpenCode 配置
        keep_alive: 是否保持沙盒存活
    
    Returns:
        执行结果字典
    """
    # 1. 生成 tasks.md 内容
    tasks_md = generate_tasks_md_content(
        subtasks=state.subtasks or [],
        parameter_table=state.extracted_parameters or {},
        session_id=state.session_id or "default",
        user_input=state.user_input or "",
        execution_plan=getattr(state, 'execution_plan', '') or "",
        file_paths=state.file_paths or [],
    )
    
    # 2. 准备上下文
    context = TaskContext(
        session_id=state.session_id or "default",
        user_input=state.user_input or "",
        execution_plan=getattr(state, 'execution_plan', '') or "",
        parameter_table=state.extracted_parameters or {},
        file_paths=state.file_paths or [],
        sandbox_id=getattr(state, 'sandbox_id', None),
        sandbox_dir=getattr(state, 'sandbox_dir', None),
    )
    
    # 3. 执行 Coding Agent
    result = await run_coding_agent_in_sandbox(
        tasks_md_content=tasks_md,
        context=context,
        session_id=state.session_id or "default",
        config=config,
        keep_alive=keep_alive,
    )
    
    return result


# ============================================================================
# main_graph.py 集成节点函数
# ============================================================================

async def coding_agent_node(state: "GlobalState") -> "GlobalState":
    """
    Coding Agent 节点（用于 main_graph.py）
    
    在 OpenSandbox 中启动 OpenCode，执行 tasks.md。
    
    Args:
        state: GlobalState 实例
    
    Returns:
        更新后的 GlobalState
    """
    # 执行 Coding Agent
    result = await run_coding_agent_from_state(state)
    
    # 更新状态
    if result.get("status") == "success":
        # 成功
        state.completed_tasks = result.get("completed_tasks", [])
        if hasattr(state, 'merged_result'):
            state.merged_result["coding_agent_results"] = result.get("output", {})
    else:
        # 失败
        if hasattr(state, 'merged_result'):
            state.merged_result["coding_agent_error"] = result.get("error", "Unknown error")
    
    # 保存沙盒 ID
    if result.get("sandbox_id"):
        state.sandbox_id = result.get("sandbox_id")
    
    return state


def coding_agent_node_sync(state: "GlobalState") -> "GlobalState":
    """
    Coding Agent 节点的同步版本
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    coding_agent_node(state),
                )
                return future.result()
    except RuntimeError:
        pass
    
    return asyncio.run(coding_agent_node(state))


# ============================================================================
# 便捷函数
# ============================================================================

async def execute_simple_tasks(
    tasks: List[Dict[str, Any]],
    session_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    model_provider: str = "glm-4.7",
) -> Dict[str, Any]:
    """
    执行简单任务列表
    
    便捷函数，用于快速执行简单的任务列表。
    
    Args:
        tasks: 任务列表
        session_id: 会话 ID
        parameters: 参数表
        model_provider: 模型提供商
    
    Returns:
        执行结果
    
    Example:
        result = await execute_simple_tasks([
            {"id": "task_1", "description": "分析数据"},
            {"id": "task_2", "description": "生成报告"},
        ], session_id="test_session")
    """
    from coding_agent.tasks_md_generator import create_simple_tasks_md
    
    # 生成 tasks.md
    tasks_md = create_simple_tasks_md(
        tasks=tasks,
        session_id=session_id,
        parameters=parameters,
    )
    
    # 创建上下文
    context = TaskContext(
        session_id=session_id,
        user_input="执行任务列表",
        parameter_table=parameters or {},
    )
    
    # 执行
    return await run_coding_agent_in_sandbox(
        tasks_md_content=tasks_md,
        context=context,
        session_id=session_id,
        model_provider=model_provider,
    )


async def execute_mcp_tool_in_sandbox(
    tool_name: str,
    service_name: str,
    parameters: Dict[str, Any],
    session_id: str,
    model_provider: str = "glm-4.7",
) -> Dict[str, Any]:
    """
    在沙盒中执行单个 MCP 工具
    
    Args:
        tool_name: 工具名称
        service_name: 服务名称
        parameters: 工具参数
        session_id: 会话 ID
        model_provider: 模型提供商
    
    Returns:
        执行结果
    """
    task = {
        "id": f"{service_name}_{tool_name}",
        "description": f"调用 {service_name} 服务的 {tool_name} 工具",
        "type": "MCP_TOOL",
        "service_id": service_name,
        "tool_name": tool_name,
        "parameters": parameters,
    }
    
    return await execute_simple_tasks(
        tasks=[task],
        session_id=session_id,
        parameters=parameters,
        model_provider=model_provider,
    )


__all__ = [
    # 主要接口
    "run_coding_agent_in_sandbox",
    "run_coding_agent_sync",
    "run_coding_agent_from_state",
    # main_graph 节点
    "coding_agent_node",
    "coding_agent_node_sync",
    # 便捷函数
    "execute_simple_tasks",
    "execute_mcp_tool_in_sandbox",
]

