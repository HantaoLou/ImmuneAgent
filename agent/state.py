from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


# 用户任务类型枚举
class UserTaskType(str, Enum):
    """用户任务类型"""
    GENERAL_QA = "GENERAL_QA"
    EXECUTE_PLAN = "EXECUTE_PLAN"
    IMMUNOLOGY_TASK = "IMMUNOLOGY_TASK"

# 任务状态枚举
class TaskStatus(str, Enum):
    PENDING = "待执行"
    RUNNING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"
    DEPENDENCY_WAIT = "等待依赖"


# 单个任务模型（新增parallel_group_id，关联其依赖的并行任务组）
class SubTask(BaseModel):
    task_id: str = Field(description="唯一任务ID")
    task_type: UserTaskType = Field(description="任务类型")
    content: str = Field(description="任务内容")
    dependencies: List[str] = Field(default_factory=list, description="依赖的普通任务ID")
    parallel_group_id: Optional[str] = Field(default=None, description="依赖的并行任务组ID")
    result: Optional[Any] = Field(default=None, description="任务结果")


# 并行任务组模型（包含多个并行子任务）
class ParallelTaskGroup(BaseModel):
    group_id: str = Field(description="并行任务组ID（供依赖任务关联）")
    subtasks: List[SubTask] = Field(description="组内并行子任务")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="组整体状态")


# 全局状态类
class GlobalState(BaseModel):
    user_input: str = Field(description="用户原始输入")
    user_task_type: Optional[UserTaskType] = Field(default=None, description="用户任务类型")
    subtasks: List[SubTask] = Field(default_factory=list, description="普通子任务列表")
    parallel_task_groups: Dict[str, ParallelTaskGroup] = Field(default_factory=dict, description="并行任务组（key=group_id）")
    completed_tasks: Dict[str, SubTask] = Field(default_factory=dict, description="已完成的普通任务")
    completed_parallel_groups: Dict[str, ParallelTaskGroup] = Field(default_factory=dict, description="已完成的并行任务组")
    merged_result: Dict[str, Any] = Field(default_factory=dict, description="汇总结果")
    hitl_status: Optional[str] = Field(default=None, description="HITL状态")
    file_paths: Dict[str, str] = Field(default_factory=dict, description="文件路径")
    execution_plan: Optional[str] = Field(default=None, description="执行计划")
    sandbox_dir: str = Field(description="沙盒目录")
