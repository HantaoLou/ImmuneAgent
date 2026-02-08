from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# User task type enum
class UserTaskType(str, Enum):
    """User task type"""
    GENERAL_QA = "GENERAL_QA"
    EXECUTE_PLAN = "EXECUTE_PLAN"
    IMMUNOLOGY_TASK = "IMMUNOLOGY_TASK"
    USE_HISTORY = "USE_HISTORY"

# Task status enum
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPENDENCY_WAIT = "dependency_wait"


# Single task model (added parallel_group_id to associate dependent parallel task groups)
class SubTask(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    task_id: str = Field(description="Unique task ID")
    task_type: UserTaskType = Field(description="Task type")
    content: str = Field(description="Task content")
    dependencies: List[str] = Field(default_factory=list, description="Dependent regular task IDs")
    parallel_group_id: Optional[str] = Field(default=None, description="Dependent parallel task group ID")
    result: Optional[Any] = Field(default=None, description="Task result")


# Parallel task group model (contains multiple parallel subtasks)
class ParallelTaskGroup(BaseModel):
    group_id: str = Field(description="Parallel task group ID (for dependent task association)")
    subtasks: List[SubTask] = Field(description="Parallel subtasks within group")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Overall group status")


# Global state class
class GlobalState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    user_input: str = Field(description="User's original input")
    user_task_type: Optional[UserTaskType] = Field(default=None, description="User task type")
    subtasks: List[SubTask] = Field(default_factory=list, description="Regular subtask list")
    parallel_task_groups: Dict[str, ParallelTaskGroup] = Field(default_factory=dict, description="Parallel task groups (key=group_id)")
    completed_tasks: Dict[str, SubTask] = Field(default_factory=dict, description="Completed regular tasks")
    completed_parallel_groups: Dict[str, ParallelTaskGroup] = Field(default_factory=dict, description="Completed parallel task groups")
    merged_result: Dict[str, Any] = Field(default_factory=dict, description="Merged results")
    hitl_status: Optional[str] = Field(default=None, description="HITL status")
    file_paths: Dict[str, str] = Field(default_factory=dict, description="File paths")
    execution_plan: Optional[str] = Field(default=None, description="Execution plan")
    sandbox_dir: str = Field(description="Sandbox directory")
    use_react_executor: bool = Field(default=False, description="Use React executor for single task execution")
    react_max_steps: int = Field(default=3, description="Max React executor steps per task")
    use_react_supervisor: bool = Field(default=False, description="Use React supervisor for task classification")
    supervisor_decision: Optional[str] = Field(default=None, description="Supervisor decision label")
    supervisor_reasoning: Optional[str] = Field(default=None, description="Supervisor reasoning")
    
    # Preprocessing results - parameter table for the entire workflow
    session_id: Optional[str] = Field(default=None, description="Unique session ID for file management")
    opensandbox_id: Optional[str] = Field(default=None, description="OpenSandbox instance ID")
    sandbox_data_dir: Optional[str] = Field(default=None, description="Data directory in sandbox")
    extracted_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameter table extracted from user input")
    file_analyses: List[Dict[str, Any]] = Field(default_factory=list, description="File analysis results")