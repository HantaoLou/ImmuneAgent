# -*- coding: utf-8 -*-
"""
Iterative Executor 输入映射器

负责将 GlobalState 映射到 IterativeExecutorState。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state import GlobalState

from nodes.subagents.iterative_executor.state import (
    IterativeExecutorState,
    IterationStatus,
    EvaluationLevel,
)


def iterative_executor_input_mapper(global_state: "GlobalState") -> IterativeExecutorState:
    """
    将 GlobalState 映射到 IterativeExecutorState
    
    数据流:
    GlobalState
    ├── session_id              →  state.session_id
    ├── user_input              →  state.user_input
    ├── execution_plan          →  state.execution_plan
    ├── extracted_parameters    →  state.extracted_parameters
    ├── file_paths              →  state.file_paths
    ├── opensandbox_id          →  state.opensandbox_id
    └── sandbox_data_dir        →  state.sandbox_data_dir
    
    Args:
        global_state: 全局状态
        
    Returns:
        IterativeExecutorState: 迭代执行器状态
    """
    # 从 GlobalState 提取必要信息
    session_id = global_state.session_id or ""
    user_input = global_state.user_input
    execution_plan = global_state.execution_plan
    extracted_parameters = global_state.extracted_parameters or {}
    file_paths = global_state.file_paths or {}
    opensandbox_id = global_state.opensandbox_id
    sandbox_data_dir = global_state.sandbox_data_dir
    
    # 推断需要的 MCP 服务
    mcp_services = _infer_mcp_services(global_state)
    
    # 从 merged_result 获取额外信息
    merged_result = global_state.merged_result or {}
    
    # 创建 IterativeExecutorState
    state = IterativeExecutorState(
        # 基本信息
        session_id=session_id,
        user_input=user_input,
        
        # 可选输入
        execution_plan=execution_plan,
        extracted_parameters=extracted_parameters,
        file_paths=file_paths,
        mcp_services=mcp_services,
        
        # 配置
        max_iterations=3,
        early_stop_on_success=True,
        opensandbox_id=opensandbox_id,
        sandbox_data_dir=sandbox_data_dir,
        
        # 初始状态
        current_iteration=0,
        iteration_status=IterationStatus.NEEDS_IMPROVEMENT,
        
        # 元数据
        start_time=None,  # 将在执行时设置
    )
    
    return state


def _infer_mcp_services(global_state: "GlobalState") -> list:
    """
    从 GlobalState 推断需要的 MCP 服务
    
    推断策略:
    1. 从 extracted_parameters 中的工具名称推断
    2. 从 execution_plan 中的关键词推断
    3. 从 file_paths 中的文件类型推断
    
    Args:
        global_state: 全局状态
        
    Returns:
        list: MCP 服务名称列表
    """
    services = set()
    
    # 1. 从 extracted_parameters 中的工具名称推断
    extracted_params = global_state.extracted_parameters or {}
    for key, value in extracted_params.items():
        key_lower = key.lower()
        # 检查是否指定了服务
        if "nettcr" in key_lower or "tcr" in key_lower:
            services.add("nettcr")
        if "igblast" in key_lower or "antibody" in key_lower:
            services.add("igblast")
        if "blast" in key_lower:
            services.add("blast")
    
    # 2. 从 execution_plan 中的关键词推断
    execution_plan = global_state.execution_plan or ""
    plan_lower = execution_plan.lower()
    
    if "nettcr" in plan_lower or "tcr" in plan_lower:
        services.add("nettcr")
    if "igblast" in plan_lower or "antibody" in plan_lower:
        services.add("igblast")
    if "blast" in plan_lower:
        services.add("blast")
    
    # 3. 从 user_input 中的关键词推断
    user_input_lower = global_state.user_input.lower()
    
    if "nettcr" in user_input_lower or "tcr" in user_input_lower or "肽" in global_state.user_input:
        services.add("nettcr")
    if "igblast" in user_input_lower or "antibody" in user_input_lower or "抗体" in global_state.user_input:
        services.add("igblast")
    if "blast" in user_input_lower:
        services.add("blast")
    
    # 4. 从 file_paths 中的文件类型推断
    file_paths = global_state.file_paths or {}
    for path in file_paths.values():
        path_lower = path.lower()
        if ".fasta" in path_lower or ".fa" in path_lower:
            # FASTA 文件通常需要 blast 或 igblast
            if "igblast" not in services:
                services.add("igblast")
        if ".csv" in path_lower:
            # CSV 文件可能包含序列数据
            pass  # 不确定具体服务，保持默认
    
    # 如果没有推断出任何服务，添加默认的 nettcr
    if not services:
        # 检查 merged_result 中是否有服务信息
        merged_result = global_state.merged_result or {}
        supervisor_result = merged_result.get("supervisor", {})
        if isinstance(supervisor_result, dict):
            detected_services = supervisor_result.get("mcp_services", [])
            if detected_services:
                services.update(detected_services)
    
    return list(services)


def prepare_executor_input(state: IterativeExecutorState) -> dict:
    """
    准备 IterativeOpenCodeExecutor 的输入数据
    
    将 IterativeExecutorState 转换为 IterativeOpenCodeExecutor.execute() 需要的格式。
    
    Args:
        state: 迭代执行器状态
        
    Returns:
        dict: 执行器输入数据
    """
    input_data = {
        # 基本信息
        "session_id": state.session_id,
        "user_input": state.user_input,
        
        # 可选输入
        "execution_plan": state.execution_plan,
        "params": state.extracted_parameters,
        "input_files": list(state.file_paths.values()) if state.file_paths else [],
        "mcp_tools": state.mcp_services,
        
        # 配置
        "max_iterations": state.max_iterations,
        "early_stop_on_success": state.early_stop_on_success,
    }
    
    return input_data

