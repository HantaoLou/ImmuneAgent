from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

# 任务状态枚举
class TaskStatus(str, Enum):
    PENDING = "待执行"
    READY = "就绪（依赖已完成）"
    RUNNING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"
    DEPENDENCY_WAIT = "等待依赖"
    CACHE_REUSE = "缓存复用执行"

# 标准化任务模型（专家Agent必须输出此格式）
class StandardTask(BaseModel):
    task_id: str = Field(description="唯一任务ID")
    task_type: str = Field(description="任务类型（如immunity_paper_retrieve）")
    task_content: str = Field(description="任务具体描述")
    core_params: Dict[str, Any] = Field(default_factory=dict, description="任务核心参数（用于缓存索引）")
    dependencies: List[str] = Field(default_factory=list, description="依赖的前置任务ID")
    required_mcp_tools: List[str] = Field(default_factory=list, description="期望使用的MCP工具列表（对应config中的tool_name）")
    cache_path: str = Field(default="./code_cache.json", description="代码缓存文件路径")

# Executor Agent子图状态模型
class ExecutorState(BaseModel):
    input_tasks: List[StandardTask] = Field(default_factory=list, description="来自专家Agent的标准化任务列表")
    task_status_map: Dict[str, TaskStatus] = Field(default_factory=dict, description="任务ID→状态映射")
    completed_tasks: Dict[str, Any] = Field(default_factory=dict, description="已完成任务结果")
    available_mcp_tools: Dict[str, Any] = Field(default_factory=dict, description="加载的MCP工具配置")
    final_execution_result: Dict[str, Any] = Field(default_factory=dict, description="汇总执行结果")