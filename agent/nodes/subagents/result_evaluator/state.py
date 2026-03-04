"""
Result Evaluator Subgraph State Definition

用于总结执行结果并生成最终报告的状态定义

增强功能：
1. 收集完整的分析流程信息（deep research, hypothesis, execution plan, task list）
2. 分析 task list 的推导依据
3. 收集并分析工具输出
4. 生成类似学术论文格式的分析报告
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


class DeepResearchInfo(BaseModel):
    """深度研究信息"""
    research_summary: str = Field(default="", description="研究摘要")
    key_insights: List[str] = Field(default_factory=list, description="关键洞察")
    evidence: List[str] = Field(default_factory=list, description="证据")
    knowledge_gaps: List[str] = Field(default_factory=list, description="知识空白")
    confidence: float = Field(default=0.0, description="研究置信度")


class HypothesisInfo(BaseModel):
    """假设信息"""
    hypothesis_summary: str = Field(default="", description="假设摘要")
    testable_predictions: List[str] = Field(default_factory=list, description="可验证的预测")
    confidence: float = Field(default=0.0, description="假设置信度")


class TaskListDerivation(BaseModel):
    """任务列表推导依据"""
    decomposition_summary: str = Field(default="", description="任务分解摘要")
    required_services: List[str] = Field(default_factory=list, description="所需服务列表")
    dependency_rationale: str = Field(default="", description="依赖关系依据")
    parallel_groups: Dict[str, Any] = Field(default_factory=dict, description="并行组信息")


class ToolOutputSummary(BaseModel):
    """工具输出摘要"""
    file_path: str = Field(default="", description="文件路径")
    file_type: str = Field(default="", description="文件类型")
    content_preview: str = Field(default="", description="内容预览")
    key_results: List[str] = Field(default_factory=list, description="关键结果")


class ResultEvaluatorState(BaseModel):
    """
    Result Evaluator 子图状态

    功能：
    1. 收集所有任务的执行结果
    2. 收集 executor 产出的文件
    3. 结合 immunity 计划进行分析
    4. 生成最终总结报告（txt 格式）

    增强功能：
    - 收集并展示 deep research 结果
    - 收集并展示 hypothesis
    - 分析 task list 的推导依据
    - 整合工具输出分析
    """

    # ========== 输入 ==========
    # 用户原始输入
    user_input: str = Field(default="", description="用户原始输入")

    # Immunity 子图产出的执行计划
    execution_plan: str = Field(default="", description="执行计划")

    # ========== 增强输入：分析流程信息 ==========
    # 深度研究信息
    deep_research: DeepResearchInfo = Field(
        default_factory=DeepResearchInfo,
        description="深度研究结果"
    )

    # 假设信息
    hypothesis: HypothesisInfo = Field(
        default_factory=HypothesisInfo,
        description="生成的假设"
    )

    # 任务列表推导依据
    task_list_derivation: TaskListDerivation = Field(
        default_factory=TaskListDerivation,
        description="任务列表的推导依据"
    )

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

    # ========== 增强输入：工具输出 ==========
    # Executor 产出的文件路径
    output_files: List[str] = Field(
        default_factory=list,
        description="执行器产出的文件路径列表"
    )

    # 工具输出摘要
    tool_output_summaries: List[ToolOutputSummary] = Field(
        default_factory=list,
        description="工具输出摘要列表"
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

    # ========== 增强分析结果 ==========
    methodology: str = Field(default="", description="分析方法描述")
    scientific_rationale: str = Field(default="", description="科学依据")
    limitations: List[str] = Field(default_factory=list, description="局限性")
    validation_recommendations: List[str] = Field(default_factory=list, description="验证建议")

    # ========== 输出 ==========
    summary_report: str = Field(default="", description="总结报告")
    detailed_report: str = Field(default="", description="详细报告")
    txt_report: str = Field(default="", description="TXT格式分析报告")
    report_path: Optional[str] = Field(default=None, description="报告文件路径")
    txt_report_path: Optional[str] = Field(default=None, description="TXT报告文件路径")

    # ========== 系统配置 ==========
    sandbox_dir: str = Field(default="", description="沙盒目录")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    parent_state: Optional[Any] = Field(default=None, description="父状态引用")

