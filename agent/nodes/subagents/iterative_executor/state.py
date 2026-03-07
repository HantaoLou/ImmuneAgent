# -*- coding: utf-8 -*-
"""
Iterative Executor 子图状态定义

定义 IterativeExecutorState，用于封装 iterative_executor 子图的内部状态。
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class IterationStatus(str, Enum):
    """迭代状态"""
    SUCCESS = "success"                     # 成功，无需继续优化
    NEEDS_IMPROVEMENT = "needs_improvement" # 需要改进
    FAILED = "failed"                       # 失败


class EvaluationLevel(str, Enum):
    """评估等级"""
    EXCELLENT = "excellent"    # 优秀 (90-100%)
    GOOD = "good"              # 良好 (70-89%)
    ACCEPTABLE = "acceptable"  # 可接受 (60-69%)
    POOR = "poor"              # 较差 (40-59%)
    FAILED = "failed"          # 失败 (0-39%)


class IterativeExecutorState(BaseModel):
    """
    Iterative Executor 子图状态
    
    这个状态类封装了 iterative_executor 子图执行过程中的所有状态信息。
    """
    
    # ==================== 基本信息 ====================
    
    # 会话 ID
    session_id: str = Field(description="唯一会话标识")
    
    # 用户原始输入
    user_input: str = Field(description="用户原始输入")
    
    # ==================== 可选输入 ====================
    
    # immunity 生成的实验计划（如有）
    execution_plan: Optional[str] = Field(
        default=None, 
        description="Immunity 生成的实验计划"
    )
    
    # supervisor 提取的参数表
    extracted_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supervisor 提取的参数表"
    )
    
    # 文件路径映射
    file_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="文件路径映射 {name: path}"
    )
    
    # 需要的 MCP 服务列表
    mcp_services: List[str] = Field(
        default_factory=list,
        description="需要的 MCP 服务列表"
    )
    
    # ==================== 配置 ====================
    
    # 最大迭代次数
    max_iterations: int = Field(
        default=3,
        description="最大迭代次数"
    )
    
    # 成功时是否提前退出
    early_stop_on_success: bool = Field(
        default=True,
        description="成功时是否提前退出"
    )
    
    # OpenSandbox ID
    opensandbox_id: Optional[str] = Field(
        default=None,
        description="OpenSandbox 实例 ID"
    )
    
    # 沙盒数据目录
    sandbox_data_dir: Optional[str] = Field(
        default=None,
        description="沙盒数据目录"
    )
    
    # ==================== 执行状态 ====================
    
    # 当前迭代次数
    current_iteration: int = Field(
        default=0,
        description="当前迭代次数"
    )
    
    # 迭代状态
    iteration_status: IterationStatus = Field(
        default=IterationStatus.NEEDS_IMPROVEMENT,
        description="迭代状态"
    )
    
    # 当前 tasks.md 内容
    current_tasks_md: Optional[str] = Field(
        default=None,
        description="当前 tasks.md 内容"
    )
    
    # ==================== 输出 ====================
    
    # 最终输出文件列表
    output_files: List[str] = Field(
        default_factory=list,
        description="输出文件列表"
    )
    
    # MCP 调用记录
    mcp_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="MCP 工具调用记录"
    )
    
    # 迭代历史
    iteration_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="迭代历史记录"
    )
    
    # 错误信息
    errors: List[str] = Field(
        default_factory=list,
        description="错误列表"
    )
    
    # 最终报告路径
    report_path: Optional[str] = Field(
        default=None,
        description="详细报告路径"
    )
    
    # ==================== 评估 ====================
    
    # 评估分数
    quality_score: float = Field(
        default=0.0,
        description="质量评估分数"
    )
    
    # 评估等级
    evaluation_level: EvaluationLevel = Field(
        default=EvaluationLevel.FAILED,
        description="评估等级"
    )
    
    # 评估详情
    evaluation_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="评估详情"
    )
    
    # ==================== 元数据 ====================
    
    # 开始时间
    start_time: Optional[str] = Field(
        default=None,
        description="执行开始时间"
    )
    
    # 结束时间
    end_time: Optional[str] = Field(
        default=None,
        description="执行结束时间"
    )
    
    class Config:
        use_enum_values = True


# ============================================================================
# 类型别名
# ============================================================================

# 迭代结果
IterationResult = Dict[str, Any]

# MCP 调用记录
MCPCallRecord = Dict[str, Any]

