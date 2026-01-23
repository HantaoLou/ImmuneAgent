from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

# Task status enumeration
class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready (dependencies completed)"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPENDENCY_WAIT = "waiting_for_dependency"
    CACHE_REUSE = "cache_reuse"

# Standardized task model (expert Agent must output this format)
class StandardTask(BaseModel):
    task_id: str = Field(description="Unique task ID")
    task_type: str = Field(description="Task type (e.g., immunity_paper_retrieve)")
    task_content: str = Field(description="Specific task description")
    core_params: Dict[str, Any] = Field(default_factory=dict, description="Task core parameters (for cache indexing)")
    dependencies: List[str] = Field(default_factory=list, description="Dependent prerequisite task IDs")
    required_mcp_tools: List[str] = Field(default_factory=list, description="Expected MCP tools list (corresponding to tool_name in config)")
    cache_path: str = Field(default="./code_cache.json", description="Code cache file path")

# Executor Agent subgraph state model
class ExecutorState(BaseModel):
    input_tasks: List[StandardTask] = Field(default_factory=list, description="Standardized task list from expert Agent")
    task_status_map: Dict[str, TaskStatus] = Field(default_factory=dict, description="Task ID -> status mapping")
    completed_tasks: Dict[str, Any] = Field(default_factory=dict, description="Completed task results")
    available_mcp_tools: Dict[str, Any] = Field(default_factory=dict, description="Loaded MCP tool configuration")
    final_execution_result: Dict[str, Any] = Field(default_factory=dict, description="Summarized execution results")