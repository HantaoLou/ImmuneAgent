"""
Iterative OpenCode Executor - 数据类型定义

定义迭代执行所需的数据类、枚举和配置。

核心概念：
- IterationStatus: 单次迭代的执行状态
- IterationResult: 单次迭代的完整结果
- IterativeExecutionResult: 完整迭代流程的最终结果
- EvaluationCriteria: 评估标准配置
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class IterationStatus(str, Enum):
    """单次迭代状态"""
    SUCCESS = "success"                    # 完全成功，无需改进
    NEEDS_IMPROVEMENT = "needs_improvement"  # 部分成功，需要优化
    FAILED = "failed"                      # 失败，需要重试
    TIMEOUT = "timeout"                    # 超时


class OptimizationStrategy(str, Enum):
    """优化策略"""
    AUTO = "auto"          # 自动决定（根据评估结果）
    FIX_ERRORS = "fix_errors"  # 修复错误
    ENHANCE_QUALITY = "enhance_quality"  # 提升质量
    ADD_VALIDATION = "add_validation"  # 添加验证步骤


@dataclass
class EvaluationReport:
    """
    单次迭代的评估报告
    
    由 OpenCode 在评估阶段生成，用于判断是否需要优化。
    """
    # 文件检查
    files_generated: List[str] = field(default_factory=list)
    files_expected: List[str] = field(default_factory=list)
    files_missing: List[str] = field(default_factory=list)
    
    # 格式验证
    format_check: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 格式: {"file.csv": {"valid": True, "rows": 100, "columns": ["a", "b"]}}
    
    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 质量评分
    quality_score: float = 0.0  # 0.0 - 1.0
    
    # 状态判定
    status: IterationStatus = IterationStatus.NEEDS_IMPROVEMENT
    
    # 改进建议
    suggestions: List[str] = field(default_factory=list)
    
    # 执行统计
    execution_time_ms: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    
    def is_success(self) -> bool:
        """判断是否成功"""
        return self.status == IterationStatus.SUCCESS
    
    def needs_improvement(self) -> bool:
        """判断是否需要改进"""
        return self.status in (IterationStatus.NEEDS_IMPROVEMENT, IterationStatus.FAILED)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "files_generated": self.files_generated,
            "files_expected": self.files_expected,
            "files_missing": self.files_missing,
            "format_check": self.format_check,
            "errors": self.errors,
            "warnings": self.warnings,
            "quality_score": self.quality_score,
            "status": self.status.value,
            "suggestions": self.suggestions,
            "execution_time_ms": self.execution_time_ms,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationReport":
        """从字典创建"""
        if isinstance(data.get("status"), str):
            data["status"] = IterationStatus(data["status"])
        return cls(**data)


@dataclass
class IterationResult:
    """
    单次迭代的完整结果
    
    包含执行和评估的所有信息。
    """
    # 迭代编号
    iteration: int
    
    # 任务文件
    tasks_md_path: str
    
    # 输出目录
    output_dir: str
    
    # 评估报告
    evaluation: EvaluationReport
    
    # 执行状态
    status: IterationStatus
    
    # 标准输出/错误
    stdout: str = ""
    stderr: str = ""
    
    # 错误信息
    errors: List[str] = field(default_factory=list)
    
    # 改进建议（用于下一次优化）
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # 时间戳
    started_at: str = ""
    completed_at: str = ""
    
    # 输出文件
    output_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "iteration": self.iteration,
            "tasks_md_path": self.tasks_md_path,
            "output_dir": self.output_dir,
            "evaluation": self.evaluation.to_dict(),
            "status": self.status.value,
            "stdout": self.stdout[:500] if self.stdout else "",  # 截断
            "stderr": self.stderr[:500] if self.stderr else "",
            "errors": self.errors,
            "improvement_suggestions": self.improvement_suggestions,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output_files": self.output_files,
        }


@dataclass
class IterationSummary:
    """单次迭代的简要摘要（用于最终报告）"""
    iteration: int
    status: IterationStatus
    quality_score: float
    tasks_completed: int
    tasks_failed: int
    key_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "status": self.status.value,
            "quality_score": self.quality_score,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "key_errors": self.key_errors[:3] if self.key_errors else [],
        }


@dataclass
class IterativeExecutionResult:
    """
    完整迭代流程的最终结果
    
    这是 IterativeOpenCodeExecutor.execute() 的返回类型。
    """
    # 会话信息
    session_id: str
    
    # 迭代统计
    total_iterations: int
    max_iterations: int
    
    # 最终状态
    final_status: IterationStatus
    
    # 最终输出
    final_output_dir: str
    final_output_files: List[str]
    
    # 迭代历史
    iteration_history: List[IterationResult]
    iteration_summaries: List[IterationSummary]
    
    # 最终摘要
    final_summary: Dict[str, Any]
    
    # 最终报告路径
    final_report_path: str
    
    # 沙盒信息
    sandbox_id: Optional[str] = None
    
    # 总执行时间
    total_execution_time_ms: int = 0
    
    def is_success(self) -> bool:
        """判断是否成功"""
        return self.final_status == IterationStatus.SUCCESS
    
    def get_best_iteration(self) -> Optional[IterationResult]:
        """获取质量最高的迭代"""
        if not self.iteration_history:
            return None
        return max(self.iteration_history, key=lambda x: x.evaluation.quality_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "total_iterations": self.total_iterations,
            "max_iterations": self.max_iterations,
            "final_status": self.final_status.value,
            "final_output_dir": self.final_output_dir,
            "final_output_files": self.final_output_files,
            "iteration_summaries": [s.to_dict() for s in self.iteration_summaries],
            "final_summary": self.final_summary,
            "final_report_path": self.final_report_path,
            "sandbox_id": self.sandbox_id,
            "total_execution_time_ms": self.total_execution_time_ms,
        }


@dataclass
class EvaluationCriteria:
    """
    评估标准配置
    
    定义如何评估单次迭代的输出质量。
    """
    # 必须生成的输出文件
    required_output_files: List[str] = field(default_factory=list)
    
    # 期望的输出文件（可选）
    expected_output_files: List[str] = field(default_factory=list)
    
    # 质量阈值
    quality_threshold_success: float = 0.85      # 达到此分数视为成功
    quality_threshold_improvement: float = 0.60  # 低于此分数视为失败
    
    # 格式验证规则
    # 格式: {"file_pattern": {"type": "json|csv|fasta", "required_columns": [...]}}
    format_validators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 错误容忍度
    max_errors_allowed: int = 0          # 允许的最大错误数
    max_warnings_allowed: int = 5        # 允许的最大警告数
    
    # 特殊检查
    check_mcp_calls: bool = True         # 是否检查 MCP 调用是否成功
    check_code_execution: bool = True    # 是否检查代码执行是否成功
    
    @classmethod
    def default(cls) -> "EvaluationCriteria":
        """获取默认评估标准"""
        return cls(
            quality_threshold_success=0.85,
            quality_threshold_improvement=0.60,
            max_errors_allowed=0,
            max_warnings_allowed=5,
            check_mcp_calls=True,
            check_code_execution=True,
        )
    
    @classmethod
    def lenient(cls) -> "EvaluationCriteria":
        """宽松的评估标准"""
        return cls(
            quality_threshold_success=0.70,
            quality_threshold_improvement=0.40,
            max_errors_allowed=2,
            max_warnings_allowed=10,
            check_mcp_calls=False,
            check_code_execution=True,
        )
    
    @classmethod
    def strict(cls) -> "EvaluationCriteria":
        """严格的评估标准"""
        return cls(
            quality_threshold_success=0.95,
            quality_threshold_improvement=0.75,
            max_errors_allowed=0,
            max_warnings_allowed=2,
            check_mcp_calls=True,
            check_code_execution=True,
        )


@dataclass
class IterativeConfig:
    """
    迭代执行配置
    
    包含迭代行为和优化策略的配置。
    """
    # 最大迭代次数
    max_iterations: int = 3
    
    # 是否在成功时提前退出
    early_stop_on_success: bool = True
    
    # 评估标准
    evaluation_criteria: EvaluationCriteria = field(default_factory=EvaluationCriteria.default)
    
    # 优化策略
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.AUTO
    
    # 是否保留中间迭代结果
    keep_iteration_history: bool = True
    
    # 生成详细报告
    generate_detailed_report: bool = True
    
    # 超时设置
    iteration_timeout_seconds: int = 600  # 单次迭代超时
    
    # 输出目录模板
    workspace_template: str = "/tmp/sessions/{session_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "max_iterations": self.max_iterations,
            "early_stop_on_success": self.early_stop_on_success,
            "evaluation_criteria": asdict(self.evaluation_criteria),
            "optimization_strategy": self.optimization_strategy.value,
            "keep_iteration_history": self.keep_iteration_history,
            "generate_detailed_report": self.generate_detailed_report,
            "iteration_timeout_seconds": self.iteration_timeout_seconds,
            "workspace_template": self.workspace_template,
        }


@dataclass
class InputData:
    """
    输入数据结构
    
    这是 execute() 方法接收的标准化输入格式。
    """
    # 会话 ID（可选，自动生成）
    session_id: Optional[str] = None
    
    # 用户意图/任务描述
    user_intent: str = ""
    
    # 输入文件路径（沙盒服务器上的路径）
    input_files: List[str] = field(default_factory=list)
    
    # 参数表
    params: Dict[str, Any] = field(default_factory=dict)
    
    # 上下文信息
    context: Dict[str, Any] = field(default_factory=dict)
    
    # 可用的 MCP 工具列表
    mcp_tools: List[str] = field(default_factory=list)
    
    # 任务类型（用于指导任务生成）
    task_type: str = "general"  # general, analysis, prediction, pipeline
    
    # 优先级
    priority: str = "normal"  # low, normal, high
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "user_intent": self.user_intent,
            "input_files": self.input_files,
            "params": self.params,
            "context": self.context,
            "mcp_tools": self.mcp_tools,
            "task_type": self.task_type,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputData":
        """从字典创建"""
        return cls(
            session_id=data.get("session_id"),
            user_intent=data.get("user_intent", ""),
            input_files=data.get("input_files", []),
            params=data.get("params", {}),
            context=data.get("context", {}),
            mcp_tools=data.get("mcp_tools", []),
            task_type=data.get("task_type", "general"),
            priority=data.get("priority", "normal"),
        )


__all__ = [
    # 枚举
    "IterationStatus",
    "OptimizationStrategy",
    
    # 数据类
    "EvaluationReport",
    "IterationResult",
    "IterationSummary",
    "IterativeExecutionResult",
    "EvaluationCriteria",
    "IterativeConfig",
    "InputData",
]

