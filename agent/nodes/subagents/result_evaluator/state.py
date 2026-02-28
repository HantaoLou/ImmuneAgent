"""
Result Evaluator Subgraph State Definition

用于总结执行结果并生成最终报告的状态定义
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class TaskResultSummary(BaseModel):
    """单个任务的执行结果摘要"""
    task_id: str = Field(description="任务ID")
    task_type: str = Field(default="", description="任务类型")
    status: str = Field(default="", description="任务状态")
    content: str = Field(default="", description="任务内容")
    error: Optional[str] = Field(default=None, description="错误信息")
    output: Optional[Any] = Field(default=None, description="输出结果")
    output_files: List[str] = Field(default_factory=list, description="输出文件路径列表")
    execution_time: Optional[float] = Field(default=None, description="执行时间(秒)")


class ResultEvaluatorState(BaseModel):
    """
    Result Evaluator 子图状态

    功能：
    1. 收集所有任务的执行结果
    2. 收集 executor 产出的文件
    3. 结合 immunity 计划进行分析
    4. 生成最终总结报告
    """

    # ========== 输入 ==========
    # 用户原始输入
    user_input: str = Field(default="", description="用户原始输入")

    # Immunity 子图产出的执行计划
    execution_plan: str = Field(default="", description="执行计划")

    # 任务列表
    task_results: Dict[str, TaskResultSummary] = Field(
        default_factory=dict,
        description="任务执行结果，key=task_id"
    )

    # 所有任务
    all_tasks: List[TaskResultSummary] = Field(
        default_factory=list,
        description="所有任务列表"
    )

    # Executor 产出的文件路径
    output_files: List[str] = Field(
        default_factory=list,
        description="执行器产出的文件路径列表"
    )

    # ========== 执行统计 ==========
    total_tasks: int = Field(default=0, description="总任务数")
    completed_tasks: int = Field(default=0, description="完成任务数")
    failed_tasks: int = Field(default=0, description="失败任务数")

    # ========== 分析结果 ==========
    success_rate: float = Field(default=0.0, description="成功率")
    error_summary: str = Field(default="", description="错误摘要")
    key_findings: List[str] = Field(default_factory=list, description="关键发现")
    recommendations: List[str] = Field(default_factory=list, description="建议")

    # ========== 输出 ==========
    summary_report: str = Field(default="", description="总结报告")
    detailed_report: str = Field(default="", description="详细报告")
    report_path: Optional[str] = Field(default=None, description="报告文件路径")

    # ========== 系统配置 ==========
    sandbox_dir: str = Field(default="", description="沙盒目录")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    parent_state: Optional[Any] = Field(default=None, description="父状态引用")

