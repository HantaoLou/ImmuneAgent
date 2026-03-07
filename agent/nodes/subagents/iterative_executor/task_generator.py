# -*- coding: utf-8 -*-
"""
Iterative Executor 任务生成器

负责根据 planning 和参数生成 tasks.md 内容。
"""

from typing import Dict, List, Any, Optional
import os

from nodes.subagents.iterative_executor.state import IterativeExecutorState
from nodes.subagents.iterative_executor.prompts import (
    TASK_GENERATION_SYSTEM_PROMPT,
    TASK_GENERATION_USER_PROMPT,
    FILE_PROCESSING_TASKS_TEMPLATE,
    NETTCR_ANALYSIS_TASKS_TEMPLATE,
    IGBLAST_ANALYSIS_TASKS_TEMPLATE,
    format_mcp_services_info,
    format_params_info,
    format_file_paths_info,
)


def generate_tasks_md(
    state: IterativeExecutorState,
    use_llm: bool = False,
    llm=None,
) -> str:
    """
    根据状态生成 tasks.md 内容
    
    生成策略:
    1. 如果有 execution_plan，使用 LLM 生成（如果可用）
    2. 否则根据推断的任务类型使用模板
    
    Args:
        state: 迭代执行器状态
        use_llm: 是否使用 LLM 生成
        llm: LLM 实例（如果 use_llm=True）
        
    Returns:
        str: tasks.md 内容
    """
    # 策略 1: 如果有 execution_plan 且 LLM 可用，使用 LLM 生成
    if use_llm and llm and state.execution_plan:
        return _generate_tasks_with_llm(state, llm)
    
    # 策略 2: 根据推断的任务类型使用模板
    task_type = _infer_task_type(state)
    
    if task_type == "nettcr_analysis":
        return _generate_nettcr_tasks(state)
    elif task_type == "igblast_analysis":
        return _generate_igblast_tasks(state)
    elif task_type == "file_processing":
        return _generate_file_processing_tasks(state)
    else:
        # 默认：通用任务模板
        return _generate_generic_tasks(state)


def _infer_task_type(state: IterativeExecutorState) -> str:
    """
    推断任务类型
    
    Args:
        state: 迭代执行器状态
        
    Returns:
        str: 任务类型
    """
    # 检查 MCP 服务
    services = state.mcp_services
    user_input_lower = state.user_input.lower()
    execution_plan_lower = (state.execution_plan or "").lower()
    
    # NetTCR 任务
    if "nettcr" in services or "nettcr" in user_input_lower or "nettcr" in execution_plan_lower:
        if "肽" in state.user_input or "peptide" in user_input_lower or "tcr" in user_input_lower:
            return "nettcr_analysis"
    
    # IgBLAST 任务
    if "igblast" in services or "igblast" in user_input_lower or "igblast" in execution_plan_lower:
        if "抗体" in state.user_input or "antibody" in user_input_lower or "vdj" in user_input_lower:
            return "igblast_analysis"
    
    # 文件处理任务
    if state.file_paths and not state.execution_plan:
        return "file_processing"
    
    # 默认
    return "generic"


def _generate_tasks_with_llm(state: IterativeExecutorState, llm) -> str:
    """
    使用 LLM 生成 tasks.md
    
    Args:
        state: 迭代执行器状态
        llm: LLM 实例
        
    Returns:
        str: tasks.md 内容
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError:
        # 如果 langchain 不可用，回退到模板
        return _generate_generic_tasks(state)
    
    # 准备提示词
    system_prompt = TASK_GENERATION_SYSTEM_PROMPT.format(
        mcp_services_info=format_mcp_services_info(state.mcp_services)
    )
    
    user_prompt = TASK_GENERATION_USER_PROMPT.format(
        user_input=state.user_input,
        execution_plan=state.execution_plan or "无",
        extracted_parameters=format_params_info(state.extracted_parameters),
        file_paths=format_file_paths_info(state.file_paths),
    )
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        response = llm.invoke(messages)
        tasks_md = response.content
        
        # 确保输出是字符串
        if not isinstance(tasks_md, str):
            tasks_md = str(tasks_md)
        
        return tasks_md
        
    except Exception as e:
        print(f"[TaskGenerator] LLM 生成失败: {e}")
        return _generate_generic_tasks(state)


def _generate_nettcr_tasks(state: IterativeExecutorState) -> str:
    """
    生成 NetTCR 分析任务
    
    Args:
        state: 迭代执行器状态
        
    Returns:
        str: tasks.md 内容
    """
    # 从参数中提取肽段信息
    params_info = format_params_info(state.extracted_parameters)
    
    tasks_md = NETTCR_ANALYSIS_TASKS_TEMPLATE.format(
        params_info=params_info,
        session_id=state.session_id,
    )
    
    return tasks_md


def _generate_igblast_tasks(state: IterativeExecutorState) -> str:
    """
    生成 IgBLAST 分析任务
    
    Args:
        state: 迭代执行器状态
        
    Returns:
        str: tasks.md 内容
    """
    file_paths_info = format_file_paths_info(state.file_paths)
    
    tasks_md = IGBLAST_ANALYSIS_TASKS_TEMPLATE.format(
        file_paths=file_paths_info,
        session_id=state.session_id,
    )
    
    return tasks_md


def _generate_file_processing_tasks(state: IterativeExecutorState) -> str:
    """
    生成文件处理任务
    
    Args:
        state: 迭代执行器状态
        
    Returns:
        str: tasks.md 内容
    """
    file_list = format_file_paths_info(state.file_paths)
    
    tasks_md = FILE_PROCESSING_TASKS_TEMPLATE.format(
        file_list=file_list,
        session_id=state.session_id,
    )
    
    return tasks_md


def _generate_generic_tasks(state: IterativeExecutorState) -> str:
    """
    生成通用任务模板
    
    Args:
        state: 迭代执行器状态
        
    Returns:
        str: tasks.md 内容
    """
    tasks_md = f"""# 任务执行计划

## 会话信息
- **Session ID**: {state.session_id}
- **用户输入**: {state.user_input}

## 可用参数

{format_params_info(state.extracted_parameters)}

## 输入文件

{format_file_paths_info(state.file_paths)}

## 可用 MCP 服务

{format_mcp_services_info(state.mcp_services)}

## 执行步骤

### 任务 1: 环境准备
- 检查沙盒环境
- 确认输入文件可用
- 初始化工作目录

### 任务 2: 数据预处理
- 读取输入文件
- 验证数据格式
- 进行必要的数据转换

### 任务 3: 核心分析
- 根据用户需求执行分析
- 使用可用的 MCP 工具
- 生成分析结果

### 任务 4: 结果处理
- 整理分析结果
- 生成输出文件
- 保存到沙盒目录: `/data/sessions/{state.session_id}/output/`

### 任务 5: 生成报告
- 汇总分析结果
- 生成 Markdown 报告
- 保存到: `/data/sessions/{state.session_id}/output/reports/analysis_report.md`

## 输出要求

所有输出文件应保存到以下目录:
- 数据文件: `/data/sessions/{state.session_id}/output/`
- 报告文件: `/data/sessions/{state.session_id}/output/reports/`
"""
    
    return tasks_md


def optimize_tasks_md(
    original_tasks: str,
    evaluation_result: Dict[str, Any],
    execution_log: str,
    use_llm: bool = False,
    llm=None,
) -> str:
    """
    根据评估结果优化 tasks.md
    
    Args:
        original_tasks: 原始 tasks.md 内容
        evaluation_result: 评估结果
        execution_log: 执行日志
        use_llm: 是否使用 LLM 优化
        llm: LLM 实例
        
    Returns:
        str: 优化后的 tasks.md
    """
    if not use_llm or not llm:
        # 简单优化：添加错误处理说明
        return _add_error_handling(original_tasks, evaluation_result)
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError:
        return _add_error_handling(original_tasks, evaluation_result)
    
    from nodes.subagents.iterative_executor.prompts import TASK_OPTIMIZATION_PROMPT
    
    optimization_prompt = TASK_OPTIMIZATION_PROMPT.format(
        original_tasks=original_tasks,
        evaluation_result=str(evaluation_result),
        execution_log=execution_log[:2000] if len(execution_log) > 2000 else execution_log,
    )
    
    try:
        messages = [
            SystemMessage(content="你是一个任务优化专家，负责根据执行反馈优化任务列表。"),
            HumanMessage(content=optimization_prompt),
        ]
        
        response = llm.invoke(messages)
        optimized_tasks = response.content
        
        if not isinstance(optimized_tasks, str):
            optimized_tasks = str(optimized_tasks)
        
        return optimized_tasks
        
    except Exception as e:
        print(f"[TaskGenerator] LLM 优化失败: {e}")
        return _add_error_handling(original_tasks, evaluation_result)


def _add_error_handling(tasks_md: str, evaluation_result: Dict[str, Any]) -> str:
    """
    在 tasks.md 中添加错误处理说明
    
    Args:
        tasks_md: 原始 tasks.md
        evaluation_result: 评估结果
        
    Returns:
        str: 添加了错误处理的 tasks.md
    """
    issues = evaluation_result.get("issues", [])
    suggestions = evaluation_result.get("suggestions", [])
    
    if not issues and not suggestions:
        return tasks_md
    
    error_section = "\n## 错误处理要点\n\n"
    
    if issues:
        error_section += "### 已知问题\n"
        for issue in issues:
            error_section += f"- {issue}\n"
        error_section += "\n"
    
    if suggestions:
        error_section += "### 改进建议\n"
        for suggestion in suggestions:
            error_section += f"- {suggestion}\n"
        error_section += "\n"
    
    error_section += "### 注意事项\n"
    error_section += "- 每个步骤执行前检查输入数据\n"
    error_section += "- 捕获并记录所有异常\n"
    error_section += "- 验证输出文件格式\n"
    error_section += "- 确保文件路径正确\n\n"
    
    # 在文件开头添加错误处理部分
    lines = tasks_md.split("\n")
    
    # 找到第一个任务步骤的位置
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith("### 任务") or line.startswith("## 执行步骤"):
            insert_pos = i
            break
    
    # 插入错误处理部分
    lines.insert(insert_pos, error_section)
    
    return "\n".join(lines)


def get_required_output_files(state: IterativeExecutorState) -> List[str]:
    """
    根据任务类型获取必需的输出文件列表
    
    Args:
        state: 迭代执行器状态
        
    Returns:
        list: 必需的输出文件路径列表
    """
    task_type = _infer_task_type(state)
    session_id = state.session_id
    output_dir = f"/data/sessions/{session_id}/output"
    
    if task_type == "nettcr_analysis":
        return [
            f"{output_dir}/nettcr_predictions.csv",
            f"{output_dir}/reports/nettcr_analysis_report.md",
        ]
    elif task_type == "igblast_analysis":
        return [
            f"{output_dir}/igblast_results.csv",
            f"{output_dir}/reports/igblast_analysis_report.md",
        ]
    elif task_type == "file_processing":
        return [
            f"/data/sessions/{session_id}/.agent/file_params.json",
        ]
    else:
        return [
            f"{output_dir}/analysis_results.csv",
            f"{output_dir}/reports/analysis_report.md",
        ]

