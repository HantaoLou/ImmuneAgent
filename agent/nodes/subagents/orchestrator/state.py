from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class OrchestratorTaskStatus(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class SubAgentAssignment(BaseModel):
    task_id: str
    agent_name: str
    task_content: str
    parallel_group_id: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    status: OrchestratorTaskStatus = OrchestratorTaskStatus.PENDING
    attempt: int = 0
    max_attempts: int = 2
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed: float = 0.0
    task_tools: List[str] = Field(default_factory=list)
    task_parameters: Dict[str, Any] = Field(default_factory=dict)
    task_inputs: List[str] = Field(default_factory=list)
    task_outputs: List[str] = Field(default_factory=list)


class SubAgentBundle(BaseModel):
    bundle_id: str
    agent_name: str
    task_ids: List[str] = Field(default_factory=list)
    combined_content: str = ""
    dependencies: List[str] = Field(default_factory=list)
    status: OrchestratorTaskStatus = OrchestratorTaskStatus.PENDING
    attempt: int = 0
    max_attempts: int = 2
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed: float = 0.0


class OrchestratorState(BaseModel):
    session_id: str
    user_input: str
    execution_plan: Optional[str] = None
    file_paths: Dict[str, str] = Field(default_factory=dict)
    assignments: List[SubAgentAssignment] = Field(default_factory=list)
    bundles: List[SubAgentBundle] = Field(default_factory=list)
    parallel_groups: Dict[str, List[str]] = Field(default_factory=dict)
    dependency_map: Dict[str, List[str]] = Field(default_factory=dict)
    bundle_dependency_map: Dict[str, List[str]] = Field(default_factory=dict)
    current_step: int = 0
    max_steps: int = 20
    react_log: List[Dict[str, Any]] = Field(default_factory=list)
