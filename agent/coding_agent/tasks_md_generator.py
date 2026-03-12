"""
Tasks.md 生成器

将 SubTask 列表和参数表转换为 OpenCode 可读取的 tasks.md 格式。

tasks.md 格式说明：
- OpenCode 读取 markdown 格式的任务列表
- 每个任务包含类型、描述、参数等信息
- 支持 MCP_TOOL、CODE_GENERATION 等任务类型
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from coding_agent.config import TasksMDConfig, TaskType

if TYPE_CHECKING:
    from state import SubTask, GlobalState


# 任务类型映射
TASK_TYPE_ICONS = {
    TaskType.MCP_TOOL: "[TOOL]",
    TaskType.CODE_GENERATION: "💻",
    TaskType.FILE_OPERATION: "📁",
    TaskType.DATA_TRANSFORM: "[RUN]",
    TaskType.ANALYSIS: "[STAT]",
    TaskType.REPORT: "📝",
}

TASK_TYPE_LABELS = {
    TaskType.MCP_TOOL: "MCP 工具调用",
    TaskType.CODE_GENERATION: "代码生成",
    TaskType.FILE_OPERATION: "文件操作",
    TaskType.DATA_TRANSFORM: "数据转换",
    TaskType.ANALYSIS: "数据分析",
    TaskType.REPORT: "报告生成",
}


def generate_tasks_md_content(
    subtasks: List["SubTask"],
    parameter_table: Dict[str, Any],
    session_id: str,
    config: Optional[TasksMDConfig] = None,
    user_input: str = "",
    execution_plan: str = "",
    file_paths: Optional[List[str]] = None,
) -> str:
    """
    生成 tasks.md 内容
    
    Args:
        subtasks: 子任务列表
        parameter_table: 参数表（从 supervisor 收集）
        session_id: 会话 ID
        config: 生成配置
        user_input: 用户原始输入
        execution_plan: 执行计划
        file_paths: 文件路径列表
    
    Returns:
        tasks.md 文件内容
    """
    config = config or TasksMDConfig.default()
    
    lines = []
    
    # ========== 文件头 ==========
    lines.append("# 任务列表")
    lines.append("")
    lines.append(f"> 会话 ID: `{session_id}`")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # ========== 用户输入 ==========
    if user_input and config.include_context:
        lines.append("## 用户输入")
        lines.append("")
        lines.append(f"```text")
        lines.append(user_input[:config.max_task_description_length])
        lines.append("```")
        lines.append("")
    
    # ========== 执行计划 ==========
    if execution_plan and config.include_context:
        lines.append("## 执行计划")
        lines.append("")
        lines.append(execution_plan[:config.max_task_description_length * 2])
        lines.append("")
    
    # ========== 参数表 ==========
    if parameter_table and config.include_parameters:
        lines.append("## 参数表")
        lines.append("")
        lines.append("以下是系统收集的参数，可在任务执行中使用：")
        lines.append("")
        lines.append("```json")
        # 格式化参数表，避免过长
        formatted_params = _format_parameter_table(parameter_table)
        lines.append(json.dumps(formatted_params, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    
    # ========== 文件信息 ==========
    if file_paths and config.include_file_info:
        lines.append("## 输入文件")
        lines.append("")
        for file_path in file_paths:
            lines.append(f"- `{file_path}`")
        lines.append("")
    
    # ========== 任务列表 ==========
    lines.append("## 任务列表")
    lines.append("")
    lines.append("请按顺序执行以下任务：")
    lines.append("")
    
    for idx, task in enumerate(subtasks, 1):
        task_md = _format_task(task, idx, parameter_table, config)
        lines.append(task_md)
        lines.append("")
    
    # ========== 执行说明 ==========
    lines.append("---")
    lines.append("")
    lines.append("## 执行说明")
    lines.append("")
    lines.append("### 任务类型说明")
    lines.append("")
    for task_type, label in TASK_TYPE_LABELS.items():
        icon = TASK_TYPE_ICONS.get(task_type, "📌")
        lines.append(f"- **{task_type.value}** {icon} {label}")
    lines.append("")
    
    lines.append("### MCP 工具调用")
    lines.append("")
    lines.append("对于 MCP_TOOL 类型任务，使用 `call_tool()` 函数调用对应工具：")
    lines.append("")
    lines.append("```python")
    lines.append("from core.tool_interface import call_tool")
    lines.append("")
    lines.append("result = call_tool(")
    lines.append('    tool_name="工具名称",')
    lines.append('    service_name="服务名称",')
    lines.append('    parameters={"参数名": "参数值"}')
    lines.append(")")
    lines.append("```")
    lines.append("")
    
    lines.append("### 完成标记")
    lines.append("")
    lines.append("所有任务完成后，请生成 `execution_summary.json` 到 `output/` 目录：")
    lines.append("")
    lines.append("```json")
    lines.append('{')
    lines.append('  "status": "success",')
    lines.append('  "completed_tasks": ["task_1", "task_2"],')
    lines.append('  "output_files": ["output/result.csv"],')
    lines.append('  "summary": "任务执行摘要..."')
    lines.append('}')
    lines.append("```")
    lines.append("")
    
    return "\n".join(lines)


def _format_task(
    task: "SubTask",
    index: int,
    parameter_table: Dict[str, Any],
    config: TasksMDConfig,
) -> str:
    """格式化单个任务"""
    lines = []
    
    # 确定任务类型
    task_type = _determine_task_type(task)
    icon = TASK_TYPE_ICONS.get(task_type, "📌")
    
    # 任务标题 - 使用 content 或 description（兼容两种格式）
    task_content = getattr(task, 'content', None) or getattr(task, 'description', '') or ''
    task_title = task_content[:config.max_task_description_length]
    lines.append(f"### 任务 {index}: {icon} {task_title}")
    lines.append("")
    
    # 任务元数据 - 兼容 task_id 和 id
    task_id = getattr(task, 'task_id', None) or getattr(task, 'id', 'unknown')
    lines.append(f"- **任务 ID**: `{task_id}`")
    lines.append(f"- **类型**: `{task_type.value}`")
    
    # 服务和工具信息（如果存在）
    service_id = getattr(task, 'service_id', None)
    tool_name = getattr(task, 'tool_name', None)
    if service_id:
        lines.append(f"- **服务**: `{service_id}`")
    if tool_name:
        lines.append(f"- **工具**: `{tool_name}`")
    
    # 依赖关系
    if task.dependencies:
        deps = ", ".join(f"`{d}`" for d in task.dependencies)
        lines.append(f"- **依赖**: {deps}")
    
    lines.append("")
    
    # 任务描述
    if task_content:
        lines.append("**描述：**")
        lines.append("")
        lines.append(task_content)
        lines.append("")
    
    # 任务参数（如果存在）
    parameters = getattr(task, 'parameters', None)
    if parameters:
        lines.append("**参数：**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(parameters, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    
    # 参数推断提示
    if parameter_table and service_id:
        relevant_params = _find_relevant_params(task, parameter_table)
        if relevant_params:
            lines.append("**可用参数（从参数表匹配）：**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(relevant_params, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
    
    # 预期输出（如果存在）
    expected_output = getattr(task, 'expected_output', None)
    if expected_output:
        lines.append("**预期输出：**")
        lines.append("")
        lines.append(expected_output)
        lines.append("")
    
    return "\n".join(lines)


def _determine_task_type(task: "SubTask") -> TaskType:
    """确定任务类型"""
    # 获取服务和工具信息（兼容两种字段名）
    service_id = getattr(task, 'service_id', None)
    tool_name = getattr(task, 'tool_name', None)
    # 兼容 content 和 description 两种字段名
    content = getattr(task, 'content', None) or getattr(task, 'description', '') or ''
    
    # 根据服务 ID 和工具名称推断任务类型
    service_lower = (service_id or "").lower()
    tool_lower = (tool_name or "").lower()
    content_lower = content.lower()
    
    # MCP 工具
    if service_id in ["nettcr", "igblast", "metabcr", "bcell", "tcell"]:
        return TaskType.MCP_TOOL
    
    # 代码生成
    if "code" in content_lower or "generate" in content_lower or "生成代码" in content_lower:
        return TaskType.CODE_GENERATION
    
    # 文件操作
    if "file" in content_lower or "文件" in content_lower or "读取" in content_lower or "写入" in content_lower:
        return TaskType.FILE_OPERATION
    
    # 数据转换
    if "convert" in content_lower or "transform" in content_lower or "转换" in content_lower:
        return TaskType.DATA_TRANSFORM
    
    # 数据分析
    if "analyze" in content_lower or "analysis" in content_lower or "分析" in content_lower:
        return TaskType.ANALYSIS
    
    # 报告生成
    if "report" in content_lower or "报告" in content_lower:
        return TaskType.REPORT
    
    # 默认为代码生成
    return TaskType.CODE_GENERATION


def _format_parameter_table(parameter_table: Dict[str, Any]) -> Dict[str, Any]:
    """格式化参数表，截断过长的值"""
    formatted = {}
    for key, value in parameter_table.items():
        if isinstance(value, str) and len(value) > 200:
            formatted[key] = value[:200] + "..."
        elif isinstance(value, dict):
            formatted[key] = _format_parameter_table(value)
        elif isinstance(value, list):
            if len(value) > 10:
                formatted[key] = value[:10] + ["... (truncated)"]
            else:
                formatted[key] = value
        else:
            formatted[key] = value
    return formatted


def _find_relevant_params(
    task: "SubTask",
    parameter_table: Dict[str, Any],
) -> Dict[str, Any]:
    """查找与任务相关的参数"""
    relevant = {}
    
    # 获取任务信息（兼容两种字段名）
    service_id = getattr(task, 'service_id', None)
    tool_name = getattr(task, 'tool_name', None)
    content = getattr(task, 'content', None) or getattr(task, 'description', '') or ''
    
    # 任务描述中的关键词
    keywords = set()
    if service_id:
        keywords.add(service_id.lower())
    if tool_name:
        keywords.add(tool_name.lower())
    if content:
        # 提取描述中的关键词
        words = content.lower().split()
        keywords.update(w for w in words if len(w) > 3)
    
    # 匹配参数
    for key, value in parameter_table.items():
        key_lower = key.lower()
        for keyword in keywords:
            if keyword in key_lower or key_lower in keyword:
                relevant[key] = value
                break
    
    return relevant


def generate_and_save_tasks_md(
    state: "GlobalState",
    output_path: Optional[str] = None,
    config: Optional[TasksMDConfig] = None,
) -> str:
    """
    从 GlobalState 生成并保存 tasks.md
    
    Args:
        state: GlobalState 实例
        output_path: 输出路径（可选，默认为沙盒目录）
        config: 生成配置
    
    Returns:
        tasks.md 内容
    """
    # 确定输出路径
    if not output_path:
        session_id = state.session_id or "default"
        output_path = f"/data/sessions/{session_id}/todo-list.md"
    
    # 生成内容
    content = generate_tasks_md_content(
        subtasks=state.subtasks or [],
        parameter_table=state.extracted_parameters or {},
        session_id=state.session_id or "default",
        config=config,
        user_input=state.user_input or "",
        execution_plan=getattr(state, 'execution_plan', '') or "",
        file_paths=list(state.file_paths.values()) if state.file_paths else [],
    )
    
    # 保存文件
    if output_path.startswith("/data/") or output_path.startswith("/tmp/"):
        # 沙盒路径 - 需要通过 CodeAct 执行
        from utils.codeact_executor import execute_code_via_codeact
        
        code = f'''
import os

output_path = "{output_path}"
content = """{content.replace('"""', '\\"""')}"""

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"__TASKS_MD_SAVED__:{{output_path}}")
'''
        
        result = execute_code_via_codeact(
            task_description=f"保存 tasks.md 到 {output_path}",
            code_template=code,
            keep_alive=True,
        )
        
        if not result.is_success():
            raise RuntimeError(f"保存 tasks.md 失败: {result.error}")
    else:
        # 本地路径
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    return content


# ============================================================================
# 便捷函数
# ============================================================================

def create_simple_tasks_md(
    tasks: List[Dict[str, Any]],
    session_id: str = "default",
    parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """
    创建简单的 tasks.md（用于测试或简单场景）
    
    Args:
        tasks: 任务列表，每个任务是一个字典，包含：
            - id: 任务 ID
            - description: 任务描述
            - type: 任务类型（可选）
            - tool_name: 工具名称（可选）
            - parameters: 参数（可选）
        session_id: 会话 ID
        parameters: 参数表（可选）
    
    Returns:
        tasks.md 内容
    
    Example:
        tasks_md = create_simple_tasks_md([
            {"id": "task_1", "description": "分析数据", "type": "ANALYSIS"},
            {"id": "task_2", "description": "生成报告", "type": "REPORT"},
        ])
    """
    from state import SubTask, UserTaskType
    
    subtasks = []
    for task_dict in tasks:
        # 将字典转换为 SubTask（使用正确的字段名）
        task = SubTask(
            task_id=task_dict.get("id", f"task_{len(subtasks) + 1}"),
            task_type=UserTaskType.EXECUTE_PLAN,  # 默认类型
            content=task_dict.get("description", ""),
            dependencies=task_dict.get("dependencies", []),
        )
        subtasks.append(task)
    
    return generate_tasks_md_content(
        subtasks=subtasks,
        parameter_table=parameters or {},
        session_id=session_id,
    )


__all__ = [
    "generate_tasks_md_content",
    "generate_and_save_tasks_md",
    "create_simple_tasks_md",
    "TasksMDConfig",
    "TaskType",
    "TASK_TYPE_ICONS",
    "TASK_TYPE_LABELS",
]

