# -*- coding: utf-8 -*-
"""
Iterative Executor 输出映射器

负责将 IterativeExecutorState 执行结果映射回 GlobalState。
"""

from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from state import GlobalState

from nodes.subagents.iterative_executor.state import (
    IterativeExecutorState,
    IterationStatus,
    EvaluationLevel,
)


def iterative_executor_output_mapper(
    executor_state: IterativeExecutorState, global_state: "GlobalState"
) -> "GlobalState":
    """
    将 IterativeExecutorState 结果映射回 GlobalState

    数据流:
    IterativeExecutorState
    ├── iteration_status       →  state.merged_result["iterative_executor"]["status"]
    ├── current_iteration      →  state.merged_result["iterative_executor"]["total_iterations"]
    ├── output_files           →  state.completed_output_files (新字段)
    ├── mcp_calls              →  state.mcp_call_records (新字段)
    └── iteration_history      →  state.iteration_history (新字段)

    Args:
        executor_state: 迭代执行器状态（执行后）
        global_state: 全局状态（将被更新）

    Returns:
        GlobalState: 更新后的全局状态
    """
    # 确保 merged_result 存在
    if not global_state.merged_result:
        global_state.merged_result = {}

    # 1. 更新 merged_result 中的 iterative_executor 部分
    global_state.merged_result["iterative_executor"] = {
        "status": executor_state.iteration_status,
        "total_iterations": executor_state.current_iteration,
        "output_files": executor_state.output_files,
        "mcp_calls_count": len(executor_state.mcp_calls),
        "quality_score": executor_state.quality_score,
        "evaluation_level": executor_state.evaluation_level,
        "errors": executor_state.errors,
        "report_path": executor_state.report_path,
    }

    # 2. 更新 iteration_history (GlobalState 已定义此字段)
    global_state.iteration_history = executor_state.iteration_history

    # 3. 更新 completed_output_files (GlobalState 已定义此字段)
    global_state.completed_output_files = executor_state.output_files

    # 4. 更新 mcp_call_records (GlobalState 已定义此字段)
    global_state.mcp_call_records = executor_state.mcp_calls

    # 5. 更新 completed_tasks（将输出文件作为完成的任务）
    _update_completed_tasks(global_state, executor_state)

    # 6. 更新 file_paths（添加新生成的输出文件）
    _update_file_paths(global_state, executor_state)

    return global_state


def _update_completed_tasks(
    global_state: "GlobalState", executor_state: IterativeExecutorState
) -> None:
    """
    更新 completed_tasks

    将迭代执行的结果添加到 completed_tasks 中。

    Args:
        global_state: 全局状态
        executor_state: 迭代执行器状态
    """
    from state import SubTask, UserTaskType

    # 如果执行成功，创建一个表示整体执行的 SubTask
    if executor_state.iteration_status == IterationStatus.SUCCESS:
        # 不覆盖已有的 completed_tasks，只添加一个汇总任务
        summary_task = SubTask(
            task_id="iterative_executor_summary",
            task_type=UserTaskType.EXECUTE_PLAN,
            content=f"迭代执行完成: {executor_state.current_iteration} 次迭代",
            dependencies=[],
            parallel_group_id=None,
            result={
                "status": "success",
                "total_iterations": executor_state.current_iteration,
                "output_files": executor_state.output_files,
                "quality_score": executor_state.quality_score,
            },
        )
        global_state.completed_tasks[summary_task.task_id] = summary_task


def _update_file_paths(
    global_state: "GlobalState", executor_state: IterativeExecutorState
) -> None:
    """
    更新 file_paths

    将新生成的输出文件添加到 file_paths 中。

    Args:
        global_state: 全局状态
        executor_state: 迭代执行器状态
    """
    if not global_state.file_paths:
        global_state.file_paths = {}

    # 添加输出文件到 file_paths
    for i, output_file in enumerate(executor_state.output_files):
        # 从路径中提取文件名
        file_name = output_file.split("/")[-1]

        # 避免重复
        if file_name not in global_state.file_paths:
            global_state.file_paths[f"output_{i}_{file_name}"] = output_file


def extract_mcp_call_summary(executor_state: IterativeExecutorState) -> Dict[str, Any]:
    """
    从执行状态中提取 MCP 调用摘要

    Args:
        executor_state: 迭代执行器状态

    Returns:
        dict: MCP 调用摘要
    """
    if not executor_state.mcp_calls:
        return {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "services_used": [],
            "tools_used": [],
        }

    total_calls = len(executor_state.mcp_calls)
    successful_calls = sum(
        1 for call in executor_state.mcp_calls if call.get("success", False)
    )
    failed_calls = total_calls - successful_calls

    services_used = set()
    tools_used = set()

    for call in executor_state.mcp_calls:
        if call.get("service_name"):
            services_used.add(call["service_name"])
        if call.get("tool_name"):
            tools_used.add(call["tool_name"])

    return {
        "total_calls": total_calls,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "services_used": list(services_used),
        "tools_used": list(tools_used),
    }


def format_iteration_summary(executor_state: IterativeExecutorState) -> str:
    """
    格式化迭代执行摘要

    Args:
        executor_state: 迭代执行器状态

    Returns:
        str: 格式化的摘要文本
    """
    lines = [
        "# 迭代执行摘要",
        "",
        f"**会话 ID**: {executor_state.session_id}",
        f"**最终状态**: {executor_state.iteration_status}",
        f"**总迭代次数**: {executor_state.current_iteration}",
        f"**质量分数**: {executor_state.quality_score:.2f}",
        f"**评估等级**: {executor_state.evaluation_level}",
        "",
    ]

    # 输出文件
    if executor_state.output_files:
        lines.append("## 输出文件")
        for f in executor_state.output_files:
            lines.append(f"- {f}")
        lines.append("")

    # MCP 调用
    if executor_state.mcp_calls:
        mcp_summary = extract_mcp_call_summary(executor_state)
        lines.append("## MCP 调用统计")
        lines.append(f"- 总调用次数: {mcp_summary['total_calls']}")
        lines.append(f"- 成功: {mcp_summary['successful_calls']}")
        lines.append(f"- 失败: {mcp_summary['failed_calls']}")
        lines.append(f"- 使用的服务: {', '.join(mcp_summary['services_used']) or '无'}")
        lines.append("")

    # 错误信息
    if executor_state.errors:
        lines.append("## 错误信息")
        for error in executor_state.errors:
            lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines)
