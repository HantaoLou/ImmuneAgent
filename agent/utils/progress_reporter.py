"""
进度报告工具函数（带过滤）
"""

from typing import Optional, Dict, Any, TYPE_CHECKING, Union
import re

if TYPE_CHECKING:
    from state import GlobalState


def should_report_message(message: str, event_type: Optional[str] = None) -> bool:
    """
    判断是否应该报告这个消息

    过滤掉调试信息、API密钥等用户不关心的内容

    Args:
        message: 消息内容
        event_type: 事件类型，某些类型总是报告
    """
    skip_keywords = [
        "API key",
        "has_api_key",
        "connection_config",
        "health_check",
        "kwargs:",
        "Execd endpoint",
        "Nginx",
        "dynamic host port",
        "docker network_mode",
        "polling_interval",
        "[DEBUG]",
        "model_config",
        "ImportError",
        "AttributeError",
        "Traceback",
        "pydantic",
    ]

    for keyword in skip_keywords:
        if keyword.lower() in message.lower():
            return False

    always_report_types = [
        "llm_thinking",
        "llm_reasoning",
        "llm_streaming",
        "subgraph_step",
        "knowledge_retrieval",
        "analysis_progress",
        "tool_result",
    ]

    if event_type and event_type in always_report_types:
        return True

    keep_patterns = [
        r"节点.*启动",
        r"节点.*完成",
        r"节点.*执行",
        r"LLM",
        r"思考",
        r"推理",
        r"分析",
        r"检索",
        r"计算",
        r"验证",
        r"生成",
        r"处理",
        r"正在",
        r"开始",
        r"完成",
        r"成功",
        r"失败",
        r"Task",
        r"Step \d+",
        r"Node \w+",
        r"调用工具",
        r"最终答案",
        r"Final answer",
        r"任务执行",
        r"Generating",
        r"Analyzing",
        r"Processing",
        r"Successfully",
        r"Retrieving",
        r"Searching",
        r"Fetching",
    ]

    for pattern in keep_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return True

    return False


def report_progress(
    state: "GlobalState",
    event_type: str,
    message: str,
    node_name: Optional[str] = None,
    task_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    progress_percent: Optional[int] = None,
):
    """报告执行进度（带过滤）"""
    if not should_report_message(message, event_type):
        return

    if state and hasattr(state, "progress_callback") and state.progress_callback:
        try:
            state.progress_callback(
                event_type=event_type,
                message=message,
                node_name=node_name,
                task_id=task_id,
                details=details or {},
                progress_percent=progress_percent,
            )
        except Exception as e:
            print(f"[ProgressReporter] Error reporting progress: {e}")


def report_node_start(
    state: "GlobalState", node_name: str, message: Optional[str] = None
):
    """报告节点开始执行"""
    msg = message or f"启动 {node_name} 节点"
    report_progress(
        state=state, event_type="node_start", message=msg, node_name=node_name
    )


def report_node_progress(
    state: "GlobalState",
    node_name: str,
    message: str,
    progress_percent: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """报告节点执行进度"""
    report_progress(
        state=state,
        event_type="node_progress",
        message=message,
        node_name=node_name,
        progress_percent=progress_percent,
        details=details,
    )


def report_node_complete(
    state: "GlobalState",
    node_name: str,
    message: Optional[str] = None,
    progress_percent: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """报告节点执行完成"""
    msg = message or f"完成 {node_name} 节点"
    report_progress(
        state=state,
        event_type="node_complete",
        message=msg,
        node_name=node_name,
        progress_percent=progress_percent,
        details=details,
    )


def report_task_start(
    state: "GlobalState",
    task_id: str,
    task_description: str,
    node_name: Optional[str] = None,
):
    """报告子任务开始"""
    report_progress(
        state=state,
        event_type="task_start",
        message=f"开始任务: {task_description}",
        task_id=task_id,
        node_name=node_name,
        details={"task_description": task_description},
    )


def report_task_progress(
    state: "GlobalState",
    task_id: str,
    message: str,
    progress_percent: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """报告子任务进度"""
    report_progress(
        state=state,
        event_type="task_progress",
        message=message,
        task_id=task_id,
        progress_percent=progress_percent,
        details=details,
    )


def report_task_complete(
    state: "GlobalState", task_id: str, task_description: str, success: bool = True
):
    """报告子任务完成"""
    report_progress(
        state=state,
        event_type="task_complete",
        message=f"任务{'完成' if success else '失败'}: {task_description}",
        task_id=task_id,
        details={"success": success, "task_description": task_description},
    )


def report_code_generation(
    state: "GlobalState", message: str, details: Optional[Dict[str, Any]] = None
):
    """报告代码生成"""
    report_progress(
        state=state, event_type="code_generation", message=message, details=details
    )


def report_code_execution(
    state: "GlobalState", message: str, details: Optional[Dict[str, Any]] = None
):
    """报告代码执行"""
    report_progress(
        state=state, event_type="code_execution", message=message, details=details
    )


def report_tool_call(
    state: "GlobalState",
    tool_name: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
):
    """报告工具调用"""
    report_progress(
        state=state,
        event_type="tool_call",
        message=message,
        details={"tool_name": tool_name, **(details or {})},
    )


def report_error(
    state: "GlobalState",
    error_message: str,
    node_name: Optional[str] = None,
    task_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """报告错误"""
    report_progress(
        state=state,
        event_type="error",
        message=error_message,
        node_name=node_name,
        task_id=task_id,
        details=details,
    )


def report_info(
    state: "GlobalState",
    message: str,
    node_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """报告一般信息"""
    report_progress(
        state=state,
        event_type="info",
        message=message,
        node_name=node_name,
        details=details,
    )


def report_llm_thinking(
    state: "GlobalState",
    thinking_content: str,
    step_name: str = "推理",
    node_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """
    报告 LLM 思考过程

    Args:
        state: 全局状态
        thinking_content: 思考内容（会自动截断)
        step_name: 步骤名称
        node_name: 节点名称
        details: 额外详情
    """
    truncated = (
        thinking_content[:300] if len(thinking_content) > 300 else thinking_content
    )

    report_progress(
        state=state,
        event_type="llm_thinking",
        message=f"💭 {step_name}: {truncated}",
        node_name=node_name,
        details=details or {"step": step_name, "full_content": thinking_content[:1000]},
    )


def report_llm_streaming(
    state: "GlobalState",
    chunk: str,
    accumulated: str = "",
    node_name: Optional[str] = None,
):
    """
    报告 LLM 流式输出

    Args:
        state: 全局状态
        chunk: 当前输出的文本块
        accumulated: 已累积的文本
        node_name: 节点名称
    """
    report_progress(
        state=state,
        event_type="llm_streaming",
        message=chunk,
        node_name=node_name,
        details={"accumulated_length": len(accumulated)},
    )


def report_llm_reasoning(
    state: "GlobalState",
    reasoning_step: str,
    step_number: int = 0,
    total_steps: int = 0,
    node_name: Optional[str] = None,
):
    """
    报告 LLM 推理步骤

    Args:
        state: 全局状态
        reasoning_step: 推理步骤描述
        step_number: 当前步骤号
        total_steps: 总步骤数
        node_name: 节点名称
    """
    step_info = f"[步骤 {step_number}/{total_steps}] " if total_steps > 0 else ""

    report_progress(
        state=state,
        event_type="llm_reasoning",
        message=f"🧠 {step_info}{reasoning_step}",
        node_name=node_name,
        details={"step_number": step_number, "total_steps": total_steps},
        progress_percent=int(
            (step_number / total_steps * 100) if total_steps > 0 else 0
        ),
    )


def report_tool_result(
    state: "GlobalState",
    tool_name: str,
    result_summary: str,
    success: bool = True,
    node_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """
    报告工具调用结果

    Args:
        state: 全局状态
        tool_name: 工具名称
        result_summary: 结果摘要
        success: 是否成功
        node_name: 节点名称
        details: 额外详情
    """
    icon = "✅" if success else "❌"
    truncated_summary = (
        result_summary[:200] if len(result_summary) > 200 else result_summary
    )

    report_progress(
        state=state,
        event_type="tool_result",
        message=f"{icon} 工具 [{tool_name}] 返回: {truncated_summary}",
        node_name=node_name,
        details={"tool_name": tool_name, "success": success, **(details or {})},
    )


def report_subgraph_step(
    state: "GlobalState",
    subgraph_name: str,
    step_name: str,
    message: str = "",
    progress_percent: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """
    报告子图内部步骤

    Args:
        state: 全局状态
        subgraph_name: 子图名称（如 general_qa, supervisor）
        step_name: 步骤名称
        message: 步骤消息
        progress_percent: 进度百分比
        details: 额外详情
    """
    full_message = f"[{subgraph_name}] {step_name}"
    if message:
        full_message += f" - {message}"

    report_progress(
        state=state,
        event_type="subgraph_step",
        message=full_message,
        node_name=subgraph_name,
        progress_percent=progress_percent,
        details={"subgraph": subgraph_name, "step": step_name, **(details or {})},
    )


def report_knowledge_retrieval(
    state: "GlobalState",
    source: str,
    query: str,
    results_count: int = 0,
    node_name: Optional[str] = None,
):
    """
    报告知识检索进度

    Args:
        state: 全局状态
        source: 数据源（如 Qdrant, Tavily, KnowledgeGraph）
        query: 查询内容
        results_count: 检索结果数量
        node_name: 节点名称
    """
    truncated_query = query[:100] if len(query) > 100 else query

    report_progress(
        state=state,
        event_type="knowledge_retrieval",
        message=f'🔍 从 {source} 检索: "{truncated_query}" ({results_count} 条结果)',
        node_name=node_name,
        details={"source": source, "results_count": results_count},
    )


def report_analysis_progress(
    state: "GlobalState",
    analysis_type: str,
    current_item: str = "",
    completed: int = 0,
    total: int = 0,
    node_name: Optional[str] = None,
):
    """
    报告分析进度

    Args:
        state: 全局状态
        analysis_type: 分析类型（如 "数据分析", "文献分析"）
        current_item: 当前分析项
        completed: 已完成数量
        total: 总数量
        node_name: 节点名称
    """
    progress_msg = f"📊 {analysis_type}"
    if current_item:
        truncated_item = current_item[:80] if len(current_item) > 80 else current_item
        progress_msg += f": {truncated_item}"
    if total > 0:
        progress_msg += f" ({completed}/{total})"

    progress_pct = int((completed / total * 100) if total > 0 else 0)

    report_progress(
        state=state,
        event_type="analysis_progress",
        message=progress_msg,
        node_name=node_name,
        progress_percent=progress_pct,
        details={
            "analysis_type": analysis_type,
            "completed": completed,
            "total": total,
        },
    )
