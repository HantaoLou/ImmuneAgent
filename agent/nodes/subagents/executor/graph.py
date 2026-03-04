"""
Executor Agent Subgraph (Optimized Version)

Responsible for executing tasks in the task list, main responsibilities:
1. Task initialization: Mark tasks without dependencies as ready, mark tasks with dependencies as waiting for dependencies
2. Parameter inference: Use LLM to infer task parameter values, trigger HITL if inference fails
3. Parallel execution: Execute ready tasks in parallel (with parallel limit)
4. Result reasoning: Use LLM to evaluate if results meet requirements, trigger HITL if not satisfied
5. Dependency management: Activate dependent tasks after task completion
6. Task scheduling: Take tasks from ready queue when parallel slots are available
7. Result summarization: Summarize results after all tasks are completed

Fully utilize LangGraph 1.0+ features:
- Use interrupt mechanism to implement true HITL
- Use checkpoint for state persistence
- Optimize asynchronous execution and state management
"""

from typing import Dict, List, Any, Optional, Literal, Union, Set
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pydantic import BaseModel, Field, ConfigDict, field_validator
from langgraph.graph import StateGraph, START, END
try:
    from langgraph.types import interrupt, Command
    INTERRUPT_AVAILABLE = True
except ImportError:
    # If interrupt is not available, define a placeholder function
    INTERRUPT_AVAILABLE = False
    def interrupt(value: Any = None):
        """Placeholder interrupt function (if LangGraph version doesn't support it)"""
        raise NotImplementedError("interrupt functionality requires LangGraph support, please ensure correct version is installed")
    Command = None


# ===================== Interrupt Helper Functions =====================

def safe_interrupt(interrupt_value: Any = None) -> Optional[Any]:
    """
    Safely call interrupt function
    
    LangGraph's interrupt mechanism:
    - On first call: Raises GraphInterrupt exception, LangGraph will catch and save state
    - On resume: interrupt() returns the resume value from Command(resume=...)
    
    Args:
        interrupt_value: Value passed during interrupt (used to identify interrupt reason)
    
    Returns:
        If resuming execution, returns resume value; if first call, returns None (but will raise exception)
    """
    if not INTERRUPT_AVAILABLE:
        return None
    
    try:
        # Try to call interrupt
        # If first call, this will raise GraphInterrupt exception
        # If resuming execution, this will return resume value
        resume_value = interrupt(interrupt_value)
        return resume_value
    except Exception as e:
        # On first call, interrupt will raise exception (this is normal behavior)
        # LangGraph will catch this exception and save state
        # We don't need to handle it here, let the exception propagate upward
        raise
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # If MemorySaver doesn't exist, use simple memory storage
    MemorySaver = None
import sys
import os
import json
import re
from pathlib import Path
from enum import Enum
from uuid import uuid4

import time

# Import main graph state and task models
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import SubTask, TaskStatus, UserTaskType, GlobalState, ParallelTaskGroup

# Module-level storage for parent_state (keyed by thread_id)
# This avoids cross-thread contamination when running multiple executors.
_parent_state_by_thread: Dict[str, GlobalState] = {}

# Import CodeAct subgraph
from nodes.subagents.code_act.graph import (
    build_codeact_subgraph,
    codeact_input_mapper,
    codeact_output_mapper,
    CodeActExecutionMode,
    CodeActState
)

# Import LLM factory
from utils.llm_factory import create_reasoning_advanced_llm, create_reasoning_llm

# ===================== Executor Subgraph State Model =====================

class ExecutorTaskStatus(str, Enum):
    """Executor internal task status"""
    READY = "ready"  # No dependencies or dependencies completed, can execute
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Execution successful
    FAILED = "failed"  # Execution failed
    WAITING_DEPENDENCY = "waiting_dependency"  # Waiting for dependent tasks to complete
    WAITING_HITL_PARAMS = "waiting_hitl_params"  # Waiting for user to provide parameters
    WAITING_HITL_CONFIRM = "waiting_hitl_confirm"  # Waiting for user confirmation to continue


class ErrorCategory(str, Enum):
    """Error category"""
    RETRYABLE = "retryable"  # Network errors, timeouts, etc., can retry
    NETWORK_ERROR = "network_error"  # Network-related errors, need special handling (re-add to task pool)
    CODE_ERROR = "code_error"  # Code logic errors, need to modify code
    PARAMETER_ERROR = "parameter_error"  # Incorrect parameters, need to modify parameters
    SYSTEM_ERROR = "system_error"  # System-level errors, may require manual intervention


class TaskExecutionResult(BaseModel):
    """Task execution result"""
    task_id: str
    status: ExecutorTaskStatus
    execution_mode: str  # "mcp_tool" or "codeact"
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parsed parameters")
    missing_parameters: List[str] = Field(default_factory=list, description="List of missing parameters")
    code: Optional[str] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    retry_count: int = 0
    execution_time: float = 0.0
    confidence_score: Optional[float] = Field(default=None, description="Result confidence score (0-1)")
    failure_analysis: Optional[str] = Field(default=None, description="Failure reason analysis")
    suggestions: Optional[List[str]] = Field(default_factory=list, description="Improvement suggestions")
    result_satisfied: Optional[bool] = Field(default=None, description="Whether result meets requirements")
    user_continue: Optional[bool] = Field(default=None, description="Whether user chose to continue execution")
    result_summary: Optional[Dict[str, Any]] = Field(default=None, description="Per-task execution summary")
    error_type: Optional[str] = Field(default=None, description="Error type if failed")
    revision_iteration: int = Field(default=0, description="CodeAct revision iteration count (to prevent infinite loops)")


class ExecutorState(BaseModel):
    """Executor subgraph state"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        from_attributes=True
    )
    
    # Input: Task list from task_decomposition
    subtasks: List[SubTask] = Field(default_factory=list, description="List of subtasks to execute")
    parallel_task_groups: Dict[str, Any] = Field(default_factory=dict, description="Parallel task groups")
    
    # Task status management
    task_status_map: Dict[str, ExecutorTaskStatus] = Field(default_factory=dict, description="Task ID -> status mapping")
    task_results: Dict[str, TaskExecutionResult] = Field(default_factory=dict, description="Task execution results")
    running_tasks: List[str] = Field(default_factory=list, description="List of currently running task IDs")
    
    # Execution configuration
    max_parallel_tasks: int = Field(default=3, description="Maximum number of parallel tasks")
    max_retries: int = Field(default=5, description="Maximum retry count")
    sandbox_dir: str = Field(default="DEFAULT_SANDBOX_DIR", description="Sandbox directory")
    use_react_executor: bool = Field(default=False, description="Use React executor for single task execution")
    react_max_steps: int = Field(default=3, description="Max React executor steps per task")
    
    # Execution statistics
    total_tasks: int = 0
    completed_count: int = 0
    failed_count: int = 0
    
    # Loop detection (prevent infinite loops)
    activate_iteration_count: int = Field(default=0, description="Consecutive activation iteration count (for deadlock detection)")
    max_activate_iterations: int = Field(default=10, description="Maximum consecutive activation iterations")
    
    # HITL related
    hitl_requests: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="HITL requests (task_id -> request info)")
    hitl_responses: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="HITL responses (task_id -> response info)")
    hitl_request_history: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="HITL request history (task_id -> last request)")
    hitl_response_history: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="HITL response history (task_id -> last response)")
    
    # Parent state reference (for accessing global state and updating HITL state)
    # Note: exclude=True to avoid LangGraph serialization validation failure, but keep in model for node use
    # Use Union type to allow accepting GlobalState instance or None during validation
    parent_state: Optional[GlobalState] = Field(default=None, exclude=True, description="Main graph state reference")
    thread_id: Optional[str] = Field(default=None, description="Thread ID for parent_state lookup")
    
    @field_validator('parent_state', mode='before')
    @classmethod
    def validate_parent_state(cls, v: Any) -> Optional[GlobalState]:
        """Validate parent_state, allow GlobalState instance or None"""
        # If it's a GlobalState instance or None, return directly
        if v is None or isinstance(v, GlobalState):
            return v
        # If it's a dict (during deserialization), try to convert to GlobalState
        if isinstance(v, dict):
            try:
                return GlobalState.model_validate(v)
            except:
                return None
        # Other cases return None
        return None


# ===================== Utility Functions =====================

def _load_tools_params_table() -> Dict[str, Dict[str, Any]]:
    """Load tools parameters table"""
    tools_params_path = agent_dir / "config" / "tools_params_table.json"
    
    if not tools_params_path.exists():
        return {}
    
    try:
        with open(tools_params_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tools_params_map = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for tool_name, params_info in item.items():
                        tools_params_map[tool_name] = params_info
        elif isinstance(data, dict):
            tools_params_map = data
        
        return tools_params_map
    except Exception as e:
        print(f"⚠ Failed to load tools parameters table: {e}")
        return {}


def _get_task_tool_names(task: SubTask) -> List[str]:
    """Extract tool names from task.result."""
    if not task.result or not isinstance(task.result, dict):
        return []
    tools = task.result.get("tools", [])
    tool_names = []
    if isinstance(tools, list):
        for tool_item in tools:
            if isinstance(tool_item, dict):
                tool_name = tool_item.get("tool_name") or tool_item.get("name", "")
            elif isinstance(tool_item, str):
                tool_name = tool_item
            else:
                tool_name = ""
            if tool_name:
                tool_names.append(tool_name)
    return tool_names


def _load_tool_limits() -> Dict[str, int]:
    """Load per-tool concurrency limits from env EXECUTOR_TOOL_LIMITS (json)."""
    raw = os.getenv("EXECUTOR_TOOL_LIMITS", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items() if int(v) > 0}
    except Exception:
        return {}
    return {}


def _load_tool_priorities() -> Dict[str, int]:
    """Load per-tool priority from env EXECUTOR_TOOL_PRIORITIES (json)."""
    raw = os.getenv("EXECUTOR_TOOL_PRIORITIES", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _restore_parent_state(state: ExecutorState) -> Optional[GlobalState]:
    """Restore parent_state from thread-scoped storage if missing."""
    if state.parent_state is not None:
        return state.parent_state
    if state.thread_id and state.thread_id in _parent_state_by_thread:
        object.__setattr__(state, "parent_state", _parent_state_by_thread[state.thread_id])
    return state.parent_state


def _is_file_type(param_name: str, param_type: Optional[str]) -> bool:
    """Check if a parameter is likely a file/path type"""
    param_name_lower = (param_name or "").lower()
    param_type_lower = (param_type or "").lower()
    keywords = [
        "file", "path", "pdb", "csv", "tsv", "json", "fasta", "fastq",
        "airr", "excel", "xlsx", "txt", "dir", "directory", "output", "input"
    ]
    return any(k in param_name_lower for k in keywords) or any(k in param_type_lower for k in keywords)


def _print_param_match_diagnostic(
    param_name: str,
    param_type: Optional[str],
    tool_name: str,
    available_files: Dict[str, Dict[str, Any]],
    provided_params: Dict[str, Any],
    reason: str
) -> None:
    """
    Print diagnostic information when a parameter is marked as missing,
    explaining why available files from preprocessing were not matched.
    
    Args:
        param_name: The parameter name that was not matched
        param_type: The expected parameter type
        tool_name: The tool requiring this parameter
        available_files: Files available from preprocessing
        provided_params: Parameters already provided/mapped
        reason: The reason category for the mismatch
    """
    print(f"\n  [ParamInfer] ⚠ DIAGNOSTIC: Parameter '{tool_name}.{param_name}' marked as MISSING")
    print(f"    Expected type: {param_type}")
    print(f"    Reason: {reason}")
    
    # Check if param_name exists in provided_params but wasn't used
    if param_name in provided_params:
        print(f"    ⚠ Parameter exists in provided_params: {provided_params[param_name]}")
        print(f"      → But was not matched. Possible type mismatch or validation failure.")
    
    # Check for potential matches with can_be_used_as
    potential_matches = []
    for file_key, file_info in available_files.items():
        if not isinstance(file_info, dict):
            continue
        can_be_used_as = file_info.get("can_be_used_as", [])
        if param_name in can_be_used_as or param_name.lower() in [c.lower() for c in can_be_used_as]:
            sandbox_path = file_info.get("sandbox_path", "")
            source_tool = file_info.get("source_tool", "")
            description = file_info.get("description", "")
            potential_matches.append({
                "file_key": file_key,
                "sandbox_path": sandbox_path,
                "source_tool": source_tool,
                "description": description,
                "can_be_used_as": can_be_used_as
            })
    
    if potential_matches:
        print(f"    ⚠ FOUND {len(potential_matches)} POTENTIAL MATCHES that should have been used:")
        for pm in potential_matches:
            print(f"      - File: {pm['sandbox_path']}")
            print(f"        From: {pm['source_tool']}")
            print(f"        Description: {pm['description']}")
            print(f"        can_be_used_as: {pm['can_be_used_as']}")
        print(f"    → BUG: These files should have been automatically matched!")
    
    # Check available files and explain why each wasn't matched
    if available_files:
        print(f"    Available files from preprocessing:")
        expected_exts = _expected_file_extensions(param_name, param_type)
        for file_key, file_info in available_files.items():
            if not isinstance(file_info, dict):
                continue
            sandbox_path = file_info.get("sandbox_path", "")
            file_type = file_info.get("file_type", "")
            mapped_to = file_info.get("mapped_to", [])
            can_be_used_as = file_info.get("can_be_used_as", [])
            source_tool = file_info.get("source_tool", "")
            
            # Determine why this file wasn't matched
            mismatch_reasons = []
            
            # Check if this file could be used for this param
            if param_name in can_be_used_as or param_name.lower() in [c.lower() for c in can_be_used_as]:
                mismatch_reasons.append(f"⚠ can_be_used_as includes '{param_name}' but NOT used - BUG!")
            elif can_be_used_as:
                mismatch_reasons.append(f"can_be_used_as: {can_be_used_as} (does not include {param_name})")
            
            # Check file extension
            if expected_exts:
                _, actual_ext = os.path.splitext(sandbox_path)
                actual_ext = actual_ext.lower().lstrip(".")
                if actual_ext not in expected_exts:
                    mismatch_reasons.append(f"extension mismatch (expected: {expected_exts}, actual: {actual_ext})")
            
            # Check if file was already mapped to this param
            if param_name in mapped_to:
                mismatch_reasons.append(f"already mapped to {param_name} but value not used")
            elif mapped_to:
                mismatch_reasons.append(f"mapped to different params: {mapped_to}")
            else:
                if not can_be_used_as:
                    mismatch_reasons.append("not mapped to any parameter and no can_be_used_as defined")
            
            print(f"      - {file_key}: {sandbox_path}")
            print(f"        Type: {file_type}, Source: {source_tool}, Mapped to: {mapped_to}")
            if can_be_used_as:
                print(f"        can_be_used_as: {can_be_used_as}")
            print(f"        Mismatch reasons: {', '.join(mismatch_reasons)}")
    else:
        print(f"    No files available from preprocessing.")
    
    # Suggest solution
    print(f"    💡 Suggestion: Check if the file type matches or if param name mapping needs to be added.")


def _looks_like_status_message(value: str) -> bool:
    """Detect progress/status strings that should never be treated as paths."""
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return (
        "status:" in lowered
        or "progress" in lowered
        or "completed" in lowered
        or "initializing" in lowered
        or "processing" in lowered
        or "loading" in lowered
    )


def _is_directory_param(param_name: str, param_type: Optional[str]) -> bool:
    """Check if a parameter expects a directory.
    
    CRITICAL: This is used to normalize file paths to directory paths.
    If a parameter name contains 'dir' or 'directory', we should extract
    the directory from any file path provided.
    """
    name_lower = (param_name or "").lower()
    type_lower = (param_type or "").lower()
    
    # Explicit directory parameter names
    explicit_dir_params = {
        "base_dir", "output_dir", "input_dir", "work_dir", "working_dir",
        "data_dir", "result_dir", "results_dir", "log_dir", "logdir",
        "temp_dir", "tmp_dir", "cache_dir", "config_dir"
    }
    
    return (
        name_lower in explicit_dir_params
        or "dir" in name_lower
        or "directory" in name_lower
        or "folder" in name_lower
        or "directory" in type_lower
    )


def _expected_file_extensions(param_name: str, param_type: Optional[str], param_desc: Optional[str] = None) -> Optional[set[str]]:
    """Infer expected file extensions from parameter metadata.

    Args:
        param_name: Parameter name
        param_type: Parameter type
        param_desc: Parameter description (optional, used to infer file types)

    Returns:
        Set of expected file extensions, or None if no specific type expected
    """
    lower_name = (param_name or "").lower()
    lower_type = (param_type or "").lower()
    lower_desc = (param_desc or "").lower()

    exts = set()
    # Check name, type, and description for file type hints
    combined_text = f"{lower_name} {lower_type} {lower_desc}"

    if "csv" in combined_text:
        exts.add("csv")
    if "tsv" in combined_text:
        exts.add("tsv")
    if "json" in combined_text:
        exts.add("json")
    if "rds" in combined_text:
        exts.add("rds")
    if "xlsx" in combined_text:
        exts.add("xlsx")
    if "xls" in combined_text:
        exts.add("xls")
    if "fasta" in combined_text or "fa" in combined_text:
        exts.update({"fasta", "fa"})
    if "fastq" in combined_text:
        exts.add("fastq")
    return exts or None


def _extract_file_candidates_from_text(text: Optional[str]) -> List[str]:
    """Extract file path candidates from free-form text."""
    # Ensure re module is available (avoid scope issues)
    import re as re_module
    if not text:
        return []
    
    # 修复问题1: 跳过包含 "Generated X files in" 或类似模式中的目录路径
    # 这些路径通常是输出目录，而不是实际的文件
    directory_patterns = [
        r"[Gg]enerated\s+\d+\s+files?\s+in\s+(/[^\s\"']+)",
        r"[Ss]aved\s+to\s+directory\s*[:=]?\s*(/[^\s\"']+)",
        r"[Oo]utput\s+directory\s*[:=]?\s*(/[^\s\"']+)",
    ]
    skip_paths = set()
    for pattern in directory_patterns:
        for match in re_module.findall(pattern, text):
            skip_paths.add(match.strip().strip(".,;:()[]{}<>\"'"))
    
    patterns = [
        r"[A-Za-z]:\\[^\s\"']+",  # Windows absolute paths
        r"/[^\s\"']+",  # Unix absolute paths
        r"(?:\.\.?[\\/])?[^\s\"']+\.(?:csv|tsv|json|xlsx|xls|fasta|fa|fastq|pdb|txt|rds)"  # Relative paths
    ]
    candidates: List[str] = []
    for pattern in patterns:
        for match in re_module.findall(pattern, text):
            cleaned = match.strip().strip(".,;:()[]{}<>\"'")
            if cleaned and not _looks_like_status_message(cleaned):
                # 跳过已识别的目录路径
                if cleaned not in skip_paths:
                    candidates.append(cleaned)
    seen = set()
    deduped = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_file_candidates_from_context(
    user_input: Optional[str],
    execution_plan: Optional[str],
    task_description: Optional[str]
) -> List[str]:
    """Collect file path candidates from task context."""
    candidates = []
    candidates.extend(_extract_file_candidates_from_text(user_input))
    candidates.extend(_extract_file_candidates_from_text(execution_plan))
    candidates.extend(_extract_file_candidates_from_text(task_description))
    seen = set()
    deduped = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _build_param_definition(param: Dict[str, Any]) -> Dict[str, Any]:
    """Build a parameter definition summary from tool metadata."""
    param_name = param.get("name", "")
    param_type = param.get("type", "")
    param_desc = param.get("description", "")
    options = param.get("options", [])
    param_required = param.get("required", True)  # Default to True if not specified
    # A parameter is optional if:
    # 1. The 'required' field is explicitly False, OR
    # 2. The type contains 'optional' (legacy support)
    is_optional = not param_required or "optional" in param_type.lower() or param_type.startswith("Optional")
    return {
        "name": param_name,
        "type": param_type,
        "description": param_desc,
        "options": options,
        "is_optional": is_optional,
        "is_file": _is_file_type(param_name, param_type),
        "expected_exts": sorted(_expected_file_extensions(param_name, param_type, param_desc) or []),
        "expects_directory": _is_directory_param(param_name, param_type),
        "is_output": _is_output_param(param_name, param_type),
    }


def _select_file_candidate(
    param_name: str,
    param_type: Optional[str],
    candidates: List[str],
    param_desc: Optional[str] = None
) -> Optional[str]:
    """Select a file candidate that best matches parameter expectations."""
    if not candidates:
        return None
    expected_exts = _expected_file_extensions(param_name, param_type, param_desc)

    # First, try exact matches
    for candidate in candidates:
        if _path_matches_expected_types(candidate, expected_exts):
            return candidate

    # If expecting CSV/TSV, also accept RDS files (will be converted in preprocessing)
    if expected_exts and expected_exts.issubset({"csv", "tsv"}):
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.lower().endswith('.rds'):
                return candidate

    # If expecting CSV/TSV, also accept Excel files (will be converted)
    if expected_exts and expected_exts.issubset({"csv", "tsv"}):
        for candidate in candidates:
            if isinstance(candidate, str):
                _, ext = os.path.splitext(candidate.lower())
                if ext.lstrip('.') in {"xls", "xlsx"}:
                    return candidate

    # Fallback: return first candidate
    return candidates[0] if candidates else None


def _auto_convert_rds_to_csv(
    rds_path: str,
    state: "ExecutorState",
    sandbox_id: Optional[str],
    output_csv_path: Optional[str] = None
) -> Optional[str]:
    """
    Auto-convert RDS file to CSV format using pyreadr library.
    
    Args:
        rds_path: Path to input RDS file
        state: Executor state
        sandbox_id: OpenSandbox instance ID
        output_csv_path: Optional output CSV file path
        
    Returns:
        Generated CSV file path, or None if conversion failed
    """
    from pathlib import Path as _PathLib
    
    if not rds_path or not rds_path.endswith(('.rds', '.RDS')):
        return None
    
    print(f"  [RDS->CSV] Auto-converting RDS to CSV: {rds_path}")
    
    # Determine output path
    if not output_csv_path:
        rds_path_normalized = rds_path.replace("\\", "/")
        rds_name = _PathLib(rds_path_normalized).stem
        
        # Get sandbox_data_dir from parent_state
        sandbox_data_dir = None
        if state.parent_state and hasattr(state.parent_state, 'sandbox_data_dir'):
            sandbox_data_dir = getattr(state.parent_state, 'sandbox_data_dir', None)
        
        if sandbox_data_dir:
            sandbox_data_dir = sandbox_data_dir.replace("\\", "/")
            # Convert server path to container path for sandbox code execution
            if sandbox_data_dir.startswith("/data/sessions/"):
                container_data_dir = sandbox_data_dir.replace("/data/sessions/", "/tmp/sessions/", 1)
            else:
                container_data_dir = sandbox_data_dir
            output_csv_path = f"{container_data_dir}/output/{rds_name}.csv"
        else:
            # Fallback: extract session_id
            import re
            session_id = None
            if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
                extracted = state.parent_state.extracted_parameters or {}
                files = extracted.get("files", {})
                for file_key, file_info in files.items():
                    if isinstance(file_info, dict):
                        sandbox_path = file_info.get("sandbox_path", "")
                        session_match = re.search(r'/(?:data|tmp)/sessions/([^/]+)/', sandbox_path)
                        if session_match:
                            session_id = session_match.group(1)
                            break
            
            if session_id:
                output_csv_path = f"/tmp/sessions/{session_id}/output/{rds_name}.csv"
            else:
                output_csv_path = f"/tmp/{rds_name}.csv"
    
    # Generate conversion code using pyreadr
    conversion_code = f'''
import pyreadr
import pandas as pd
import os

rds_path = "{rds_path}"
csv_path = "{output_csv_path}"

# Ensure output directory exists
output_dir = os.path.dirname(csv_path)
os.makedirs(output_dir, exist_ok=True)
try:
    os.chmod(output_dir, 0o777)
except Exception:
    pass

try:
    # Read RDS file
    result = pyreadr.read_r(rds_path)
    
    if not result:
        print("[ERROR] Failed to read RDS file or file is empty", flush=True)
        print("__RDS_CSV_FAILED__", flush=True)
    else:
        # Get the first data.frame or compatible object
        df = None
        for key, data in result.items():
            if isinstance(data, pd.DataFrame):
                df = data
                print(f"Found DataFrame: {{key}} with {{len(df)}} rows", flush=True)
                break
            elif isinstance(data, dict):
                # Try to convert dict to DataFrame
                try:
                    df = pd.DataFrame(data)
                    print(f"Converted dict to DataFrame: {{key}} with {{len(df)}} rows", flush=True)
                    break
                except Exception as e:
                    print(f"Could not convert {{key}} to DataFrame: {{e}}", flush=True)
                    continue
        
        if df is None:
            print("[ERROR] No compatible data object found in RDS file", flush=True)
            print("__RDS_CSV_FAILED__", flush=True)
        else:
            # Save to CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"[OK] Generated CSV with {{len(df)}} rows: {{csv_path}}", flush=True)
            print(f"__RDS_CSV_SUCCESS__:{{csv_path}}:{{len(df)}}", flush=True)
            
except ImportError:
    print("[ERROR] pyreadr library not available. Install with: pip install pyreadr", flush=True)
    print("__RDS_CSV_FAILED__", flush=True)
except Exception as e:
    print(f"[ERROR] Error converting RDS to CSV: {{e}}", flush=True)
    print("__RDS_CSV_FAILED__", flush=True)
'''
    
    # Execute in sandbox via codeact_executor (遵循架构原则：统一接口)
    if sandbox_id:
        try:
            from utils.codeact_executor import convert_rds_to_csv
            
            result = convert_rds_to_csv(
                rds_path=rds_path,
                output_csv_path=output_csv_path,
                sandbox_id=sandbox_id
            )
            
            if result.is_success() and "__RDS_CSV_SUCCESS__" in result.output:
                for line in result.output.split("\n"):
                    if "__RDS_CSV_SUCCESS__:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            csv_path = parts[1]
                            # Convert container path to server path if needed
                            if csv_path.startswith("/tmp/sessions/"):
                                csv_path = csv_path.replace("/tmp/sessions/", "/data/sessions/", 1)
                            print(f"  [RDS->CSV] Conversion successful: {csv_path}")
                            return csv_path
            else:
                print(f"  [RDS->CSV] Conversion failed: {result.error or 'Unknown error'}")
        except Exception as e:
            print(f"  [RDS->CSV] Conversion error: {e}")
    else:
        print(f"  [RDS->CSV] No sandbox_id, cannot execute conversion")
    
    return None


def _auto_convert_csv_to_fasta(
    pending_conversions: Dict[str, Any],
    state: "ExecutorState",
    sandbox_id: Optional[str]
) -> Optional[str]:
    """
    Auto-convert CSV to FASTA using pending conversion info.
    
    Args:
        pending_conversions: Dict of pending FASTA conversions from preprocessing
        state: Executor state
        sandbox_id: OpenSandbox instance ID
        
    Returns:
        Generated FASTA file path, or None if conversion failed
    """
    if not pending_conversions:
        return None
    
    # Get the first pending conversion
    for conv_key, conv_info in pending_conversions.items():
        if not isinstance(conv_info, dict):
            continue
        
        source_csv = conv_info.get("source_csv")
        seq_columns = conv_info.get("sequence_columns", [])
        suggested_path = conv_info.get("suggested_fasta_path")
        
        if not source_csv or not seq_columns:
            continue
        
        print(f"  [FASTA] Auto-converting CSV to FASTA: {source_csv}")
        print(f"  [FASTA] Sequence columns: {seq_columns}")
        
        # Generate conversion code
        fasta_code = f'''
import pandas as pd
import os

csv_path = "{source_csv}"
fasta_path = "{suggested_path}"
sequence_columns = {seq_columns}

# Ensure output directory exists with full permissions
output_dir = os.path.dirname(fasta_path)
os.makedirs(output_dir, exist_ok=True)
try:
    os.chmod(output_dir, 0o777)  # Allow all users to write
except Exception:
    pass

# Read CSV
df = pd.read_csv(csv_path)
print(f"Read CSV with {{len(df)}} rows", flush=True)

# Find available sequence columns
available_cols = [col for col in sequence_columns if col in df.columns]
if not available_cols:
    print(f"[WARN] No sequence columns found in CSV", flush=True)
    print("__FASTA_FAILED__")
else:
    print(f"Found sequence columns: {{available_cols}}", flush=True)
    
    # Try to find ID column
    id_col = None
    for candidate in ['main_name', 'name', 'id', 'ID', 'sample_id', 'cell_id', 'barcode']:
        if candidate in df.columns:
            id_col = candidate
            break
    
    if id_col:
        print(f"Using ID column: {{id_col}}", flush=True)
    
    # Generate FASTA
    seq_count = 0
    with open(fasta_path, 'w') as f:
        for idx, row in df.iterrows():
            for col in available_cols:
                seq = str(row[col]).strip()
                if seq and seq.lower() not in ['nan', 'none', '']:
                    if id_col:
                        seq_id = f"{{row[id_col]}}_{{col}}"
                    else:
                        seq_id = f"seq_{{idx}}_{{col}}"
                    f.write(f">{{seq_id}}\\n{{seq}}\\n")
                    seq_count += 1
    
    print(f"[OK] Generated FASTA with {{seq_count}} sequences: {{fasta_path}}", flush=True)
    print(f"__FASTA_SUCCESS__:{{fasta_path}}:{{seq_count}}")
'''
        
        # Try to execute in sandbox via codeact_executor (遵循架构原则：统一接口)
        if sandbox_id:
            try:
                from utils.codeact_executor import convert_csv_to_fasta
                
                result = convert_csv_to_fasta(
                    csv_path=source_csv,
                    output_path=suggested_path,
                    sequence_columns=seq_columns,
                    sandbox_id=sandbox_id
                )
                
                if result.is_success() and "__FASTA_SUCCESS__" in result.output:
                    for line in result.output.split("\n"):
                        if "__FASTA_SUCCESS__:" in line:
                            parts = line.split(":")
                            if len(parts) >= 2:
                                fasta_path = parts[1]
                                # Convert container path to server path if needed
                                if fasta_path.startswith("/tmp/sessions/"):
                                    fasta_path = fasta_path.replace("/tmp/sessions/", "/data/sessions/", 1)
                                print(f"  [FASTA] Conversion successful: {fasta_path}")
                                
                                # Update parent state's extracted_parameters
                                if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
                                    if "generated_fasta_files" not in state.parent_state.extracted_parameters:
                                        state.parent_state.extracted_parameters["generated_fasta_files"] = {}
                                    state.parent_state.extracted_parameters["generated_fasta_files"][conv_key] = fasta_path
                                    
                                    # Remove from pending conversions
                                    if "pending_fasta_conversions" in state.parent_state.extracted_parameters:
                                        state.parent_state.extracted_parameters["pending_fasta_conversions"].pop(conv_key, None)
                                
                                return fasta_path
                else:
                    print(f"  [FASTA] Conversion failed: {result.error or 'Unknown error'}")
            except Exception as e:
                print(f"  [FASTA] Conversion error: {e}")
        else:
            print(f"  [FASTA] No sandbox_id, cannot execute conversion")
        
        # Only try first conversion
        break
    
    return None


def _copy_file_to_session_directory(
    source_path: str,
    target_path: str,
    state: "ExecutorState",
    task_id: str
) -> Optional[str]:
    """
    Copy a file from MCP service output directory to session directory.
    
    This ensures all output files are consolidated in /data/sessions/{session_id}/output/
    for unified file management and easier access by subsequent tasks.
    
    Args:
        source_path: Original file path (e.g., /data/server/mcp_servers/...)
        target_path: Target path in session directory (e.g., /data/sessions/.../output/...)
        state: Executor state
        task_id: Task ID for logging
        
    Returns:
        Target path if copy succeeded, None if failed
    """
    # 使用 codeact_executor 统一接口 (遵循架构原则)
    from utils.codeact_executor import copy_file_in_sandbox, is_codeact_available
    
    if not is_codeact_available():
        print(f"  [FileCopy] CodeAct not available, skipping file copy")
        return None
    
    # Normalize paths
    source_path = source_path.replace("\\", "/")
    target_path = target_path.replace("\\", "/")
    
    # Get sandbox_id from parent state
    existing_sandbox_id = None
    if state.parent_state:
        merged_result = getattr(state.parent_state, 'merged_result', None) or {}
        existing_sandbox_id = merged_result.get('opensandbox_id')
    
    try:
        result = copy_file_in_sandbox(
            source_path=source_path,
            target_path=target_path,
            sandbox_id=existing_sandbox_id
        )
        
        if result.is_success() and "__FILE_COPIED__:" in result.output:
            return target_path  # Return server path
        elif "__FILE_NOT_FOUND__:" in result.output:
            print(f"  [FileCopy] Source file not found: {source_path}")
        elif result.error:
            print(f"  [FileCopy] Copy error: {result.error}")
    except Exception as e:
        print(f"  [FileCopy] Exception during copy: {e}")
    
    return None


def _expand_parameter_table_with_output(
    state: "ExecutorState",
    task: SubTask,
    result: "TaskExecutionResult"
) -> None:
    """
    Expand the parameter table with task output.
    
    Extracts file paths and key values from task output and adds them to
    parent_state.extracted_parameters for subsequent tasks to use.
    
    IMPORTANT: This function adds descriptions to extracted files based on the
    tool type, enabling subsequent tasks to correctly match parameters.
    
    Args:
        state: Executor state
        task: Completed task
        result: Task execution result
    """
    if not state.parent_state or not hasattr(state.parent_state, 'extracted_parameters'):
        return
    
    if not state.parent_state.extracted_parameters:
        state.parent_state.extracted_parameters = {
            "user_parameters": {},
            "files": {},
            "sandbox_file_paths": {},
            "task_outputs": {}
        }
    
    # Initialize task_outputs if not present
    if "task_outputs" not in state.parent_state.extracted_parameters:
        state.parent_state.extracted_parameters["task_outputs"] = {}
    
    task_id = task.task_id
    # Get tool name from task.result (SubTask doesn't have tool_name attribute)
    tool_names = _get_task_tool_names(task)
    tool_name = tool_names[0] if tool_names else "unknown"
    
    print(f"  [ParamTable] Processing output from {tool_name} (task {task_id})")
    
    # Extract file paths from output - support /tmp/, /data/, and absolute paths
    output_files = []
    output_metadata = {}  # Store additional metadata about each file
    output_data = result.output
    
    print(f"  [ParamTable] Output data type: {type(output_data)}")
    if output_data:
        if isinstance(output_data, str):
            print(f"  [ParamTable] Output preview: {output_data[:200]}...")
        elif isinstance(output_data, dict):
            print(f"  [ParamTable] Output keys: {list(output_data.keys())}")
    
    def _extract_files_from_dict(data: dict, prefix: str = "") -> None:
        """Recursively extract file paths from dict."""
        import re as regex_module
        for key, value in data.items():
            if isinstance(value, str):
                # Check for file paths - more inclusive pattern
                # Accept any absolute path with supported extensions
                is_output_key = key in ['output_file', 'result_file', 'output_path', 'fasta_file', 'csv_file']
                is_absolute_path = value.startswith('/')
                has_valid_ext = value.endswith(('.csv', '.fasta', '.fa', '.json', '.txt', '.rds', 
                                       '.h5', '.pdf', '.png', '.svg', '.tsv', '.xlsx'))
                
                if (is_absolute_path and has_valid_ext) or (is_output_key and has_valid_ext):
                    output_files.append(value)
                    output_metadata[value] = {
                            "key": key,
                            "prefix": prefix
                        }
                # Also extract file paths from message fields using regex
                elif key in ['message', 'content', 'output', 'result', 'info']:
                    file_pattern = r'((?:/tmp/sessions/|/data/sessions/|/data/)[^\s"\'\\]+\.(csv|fasta|fa|json|txt|rds|h5|pdf|png|svg|tsv|xlsx))'
                    matches = regex_module.findall(file_pattern, value, regex_module.IGNORECASE)
                    for match in matches:
                        file_path = match[0]
                        if file_path not in output_files:
                            output_files.append(file_path)
                            output_metadata[file_path] = {
                                "key": key,
                                "prefix": prefix,
                                "extracted_from": "message_regex"
                            }
            elif isinstance(value, dict):
                _extract_files_from_dict(value, prefix=f"{prefix}{key}.")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        _extract_files_from_dict(item, prefix=f"{prefix}{key}[{i}].")
    
    if isinstance(output_data, str):
        # Extract file paths using regex - support both /tmp/ and /data/
        import re
        file_pattern = r'((?:/tmp/sessions/|/data/)[^\s"\']+\.(csv|fasta|fa|json|txt|rds|h5|pdf|png|svg|tsv|xlsx))'
        matches = re.findall(file_pattern, output_data, re.IGNORECASE)
        output_files = [m[0] for m in matches]
        
        # Also try to parse as JSON
        try:
            parsed = json.loads(output_data)
            if isinstance(parsed, dict):
                _extract_files_from_dict(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(output_data, dict):
        _extract_files_from_dict(output_data)
    
    # Get session_output_dir early for use in implicit output file search
    session_output_dir = None
    if state.parent_state and hasattr(state.parent_state, 'sandbox_data_dir'):
        sandbox_data_dir = getattr(state.parent_state, 'sandbox_data_dir', None)
        if sandbox_data_dir:
            session_output_dir = f"{sandbox_data_dir}/output"
    
    # ENHANCED: For tools that have output_file=null but actually produce output files
    # We need to search for output files in the session output directory
    tool_lower = tool_name.lower()
    if not output_files and session_output_dir:
        # Check if this is a tool that should produce output files
        # ENHANCED: Support multiple file extensions per tool
        tools_with_implicit_output = {
            "analyze_vdj_batch": {
                "exts": [".tsv", ".csv"],  # AIRR format can be TSV or CSV
                "prefix": "airr_results",
                "description": "AIRR format V(D)J analysis results",
                "data_type": "airr_results"  # For semantic matching
            },
            "metabcr": {
                "exts": [".csv"],
                "prefix": "binding",
                "description": "Binding prediction results",
                "data_type": "binding_predictions"
            },
            "predict_tcr_binding": {
                "exts": [".csv"],
                "prefix": "predictions",
                "description": "TCR binding predictions",
                "data_type": "binding_predictions"
            },
        }
        
        for tool_key, tool_info in tools_with_implicit_output.items():
            if tool_key in tool_lower:
                print(f"  [ParamTable] Tool {tool_name} has no explicit output files, searching session output directory...")
                
                # Search for recently created files in session output directory
                import glob as glob_module
                import os as os_module
                
                # Search for all expected extensions
                expected_exts = tool_info["exts"]
                for expected_ext in expected_exts:
                    search_pattern = f"{session_output_dir}/*{expected_ext}"
                    potential_files = glob_module.glob(search_pattern)
                    
                    # Filter files by creation time (within last 5 minutes)
                    import time as time_module
                    current_time = time_module.time()
                    recent_threshold = 300  # 5 minutes
                    
                    for potential_file in potential_files:
                        try:
                            file_mtime = os_module.path.getmtime(potential_file)
                            if current_time - file_mtime < recent_threshold:
                                output_files.append(potential_file)
                                output_metadata[potential_file] = {
                                    "key": "output_file",
                                    "prefix": "",
                                    "implicit_output": True,
                                    "description": tool_info["description"],
                                    "data_type": tool_info.get("data_type", "unknown")
                                }
                                print(f"  [ParamTable] Found recent output file ({expected_ext}): {potential_file}")
                        except Exception as e:
                            print(f"  [ParamTable] Error checking file {potential_file}: {e}")
                
                break  # Only check the first matching tool
    
    # Generate meaningful descriptions based on tool type
    tool_output_descriptions = _get_tool_output_descriptions(tool_name)
    
    # Copy output files to session directory for unified file management
    # This ensures all outputs are in /data/sessions/{session_id}/output/
    # (session_output_dir already defined earlier for implicit output file search)
    
    # Copy files that are not already in session directory
    copied_files = {}  # Map original path -> session path
    for file_path in output_files:
        if session_output_dir and not file_path.startswith(session_output_dir):
            # File is outside session directory, copy it
            session_path = f"{session_output_dir}/{Path(file_path).name}"
            copy_result = _copy_file_to_session_directory(file_path, session_path, state, task_id)
            if copy_result:
                copied_files[file_path] = session_path
                print(f"  [ParamTable] Copied output file to session: {file_path} -> {session_path}")
    
    # Add output files to sandbox_file_paths for subsequent tasks
    # Ensure sandbox_file_paths exists in extracted_parameters
    if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
        if "sandbox_file_paths" not in state.parent_state.extracted_parameters:
            state.parent_state.extracted_parameters["sandbox_file_paths"] = {}
        if "files" not in state.parent_state.extracted_parameters:
            state.parent_state.extracted_parameters["files"] = {}
        if "task_outputs" not in state.parent_state.extracted_parameters:
            state.parent_state.extracted_parameters["task_outputs"] = {}
    
    for file_path in output_files:
        # Use session path if file was copied
        final_path = copied_files.get(file_path, file_path)
        file_key = f"{task_id}:{Path(file_path).name}"
        
        # Safely add to sandbox_file_paths
        if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
            if "sandbox_file_paths" in state.parent_state.extracted_parameters:
                state.parent_state.extracted_parameters["sandbox_file_paths"][file_key] = final_path
        
        # Determine file description based on tool and file metadata
        file_ext = Path(final_path).suffix.lstrip('.').lower()
        file_meta = output_metadata.get(file_path, {})
        file_key_name = file_meta.get("key", "output_file")
        
        # Get specific description for this file type from tool
        file_description = _infer_output_file_description(
            tool_name=tool_name,
            file_path=final_path,
            file_key=file_key_name,
            file_ext=file_ext,
            tool_descriptions=tool_output_descriptions
        )
        
        # Also update files dict with enriched metadata
        # Use final_path (session path if copied) for sandbox_path
        file_metadata = {
            "sandbox_path": final_path,
            "original_path": file_path if file_path != final_path else None,
            "type": file_ext,
            "data_type": f"{tool_name}_output",
            "source": "task_output",  # Tag: this file was generated by a task
            "source_task": task_id,
            "source_tool": tool_name,
            "description": file_description,
            "can_be_used_as": _get_compatible_param_types(tool_name, file_ext, file_key_name)
        }
        
        # Analyze CSV output files to extract column information
        # This is crucial for data integration - we need to know the actual column names
        if file_ext == "csv":
            columns = _analyze_csv_columns(final_path, state, task_id)
            if columns:
                file_metadata["columns"] = columns
                # For AIRR format from analyze_vdj_batch, identify the primary key
                if tool_name == "analyze_vdj_batch":
                    # AIRR format uses sequence_id as primary key
                    # Need to strip _Heavy_DNA suffix if present
                    file_metadata["primary_key"] = "sequence_id"
                    file_metadata["primary_key_transform"] = "strip_Heavy_DNA_suffix"
                    print(f"    Primary key: sequence_id (strip _Heavy_DNA suffix)")
                elif tool_name == "metabcr":
                    # MetaBCR typically uses main_name or similar
                    if "main_name" in columns:
                        file_metadata["primary_key"] = "main_name"
                    print(f"    Primary key: {file_metadata.get('primary_key', 'unknown')}")
                print(f"    Columns: {columns[:10]}{'...' if len(columns) > 10 else ''}")
        
        # Safely add to files dict
        if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
            if "files" in state.parent_state.extracted_parameters:
                state.parent_state.extracted_parameters["files"][file_key] = file_metadata
        
        print(f"  [ParamTable] Added output file: {final_path}")
        if file_path != final_path:
            print(f"    (copied from: {file_path})")
        print(f"    Description: {file_description}")
        print(f"    Compatible with: {_get_compatible_param_types(tool_name, file_ext, file_key_name)}")
    
    # Store task output summary with updated paths (using session paths if copied)
    final_output_files = [copied_files.get(fp, fp) for fp in output_files]
    if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
        if "task_outputs" in state.parent_state.extracted_parameters:
            state.parent_state.extracted_parameters["task_outputs"][task_id] = {
                "tool_name": tool_name,
                "status": "completed",
                "output_files": final_output_files,
                "original_output_files": output_files if copied_files else None,
                "parameters_used": result.parameters or {},
                "description": tool_output_descriptions.get("summary", f"Output from {tool_name}"),
            }
    
    if output_files:
        print(f"  [ParamTable] Task {task_id} added {len(output_files)} output files to parameter table")


def _get_tool_output_descriptions(tool_name: str) -> Dict[str, str]:
    """
    Get descriptions for tool outputs based on tool type.
    
    Returns:
        Dict with 'summary' and specific output descriptions
    """
    # Map tool names to their output descriptions
    tool_descriptions = {
        "analyze_vdj_batch": {
            "summary": "V(D)J recombination analysis results in AIRR format",
            "csv": "AIRR format CSV containing V/D/J gene segments, CDR3 sequences, and germline alignment data",
            "output_file": "AIRR format results with V(D)J annotations"
        },
        "metabcr": {
            "summary": "Antibody-antigen binding affinity prediction results",
            "csv": "CSV file with binding prediction scores, antibody-antigen pair results, and affinity metrics",
            "output_file": "Binding prediction results with affinity scores for each antibody-antigen combination"
        },
        "integrate_bcr_data_complete": {
            "summary": "Integrated BCR data with annotations",
            "rds": "Annotated RDS file with integrated BCR data from multiple sources",
            "csv": "Summary CSV with integrated BCR analysis results"
        },
        "antigen_binding_prediction_visualization": {
            "summary": "Antigen binding prediction visualization outputs",
            "pdf": "Binding distribution plots and statistical visualizations",
            "csv": "Visualization data with binding statistics"
        },
        "bcell_celltype_distribution_analysis": {
            "summary": "B cell type distribution analysis results",
            "csv": "Cell type distribution statistics and proportions",
            "pdf": "Distribution plots showing B cell subtype proportions"
        }
    }
    
    # Return tool-specific descriptions or generic ones
    tool_name_lower = tool_name.lower()
    for key, desc in tool_descriptions.items():
        if key.lower() in tool_name_lower or tool_name_lower in key.lower():
            return desc
    
    # Generic descriptions for unknown tools
    return {
        "summary": f"Output from {tool_name} tool",
        "csv": f"CSV data output from {tool_name}",
        "rds": f"RDS data output from {tool_name}",
        "output_file": f"Result file from {tool_name}"
    }


# ============================================================================
# Semantic type matching for parameter inference
# ============================================================================

# Map tool names to their output semantic types
# This defines what KIND of data each tool produces
TOOL_OUTPUT_SEMANTIC_TYPES: Dict[str, List[str]] = {
    "analyze_vdj_batch": ["airr_results", "antibody_analysis", "vdj_annotation"],
    "extract_cdr3_from_airr": ["cdr3_sequences", "antibody_analysis"],
    "igblast_query": ["airr_results", "antibody_analysis", "vdj_annotation"],
    "run_igblast": ["airr_results", "antibody_analysis", "vdj_annotation"],
    "metabcr": ["binding_predictions", "affinity_scores"],
    "integrate_bcr_data_complete": ["integrated_bcr_data", "seurat_object"],
    "integrate_tcr_data_complete": ["integrated_tcr_data", "seurat_object", "tcr_predictions"],
    "antigen_binding_prediction_visualization": ["visualization", "binding_statistics"],
    "bcell_celltype_distribution_analysis": ["cell_distribution", "statistics"],
    "tcr_clonotype_analysis": ["clonotype_statistics", "diversity_metrics"],
    "tcell_celltype_visualization": ["visualization", "celltype_statistics"],
    "tcr_binding_visualization": ["visualization", "binding_statistics"],
}

# Map parameter names to their expected semantic types
# This defines what KIND of data each parameter expects
PARAM_EXPECTED_SEMANTIC_TYPES: Dict[str, List[str]] = {
    # Antigen parameters - should ONLY accept antigen data, NOT antibody analysis results
    "antigen_file": ["antigen_data", "antigen_sequence", "antigen"],
    "antigen_data": ["antigen_data", "antigen_sequence", "antigen"],
    "antigen_sequences": ["antigen_data", "antigen_sequence", "antigen"],
    
    # Antibody parameters - CAN accept antibody analysis results (AIRR, etc.)
    "antibody_file": ["antibody_data", "antibody_sequence", "airr_results", "antibody_analysis"],
    "antibody_data": ["antibody_data", "antibody_sequence", "airr_results", "antibody_analysis"],
    
    # Generic CSV file - accepts various CSV outputs
    "csv_file": ["csv_data", "airr_results", "binding_predictions", "statistics"],
    
    # RDS file - accepts Seurat objects or integrated data
    "rds_file": ["rds_data", "seurat_object", "integrated_bcr_data", "integrated_tcr_data"],
    "input_file": ["rds_data", "seurat_object", "integrated_bcr_data", "integrated_tcr_data"],
    "input_rds": ["rds_data", "seurat_object", "integrated_bcr_data", "integrated_tcr_data"],
}


def _get_tool_output_semantic_types(tool_name: str) -> List[str]:
    """Get semantic types for a tool's output."""
    tool_lower = tool_name.lower()
    for key, types in TOOL_OUTPUT_SEMANTIC_TYPES.items():
        if key.lower() == tool_lower or key.lower() in tool_lower or tool_lower in key.lower():
            return types
    # Unknown tool - return generic type
    return ["unknown_output"]


def _get_param_expected_semantic_types(param_name: str) -> List[str]:
    """Get expected semantic types for a parameter."""
    param_lower = param_name.lower()
    for key, types in PARAM_EXPECTED_SEMANTIC_TYPES.items():
        if key.lower() == param_lower:
            return types
    # Check partial matches for common patterns
    if "antigen" in param_lower:
        return ["antigen_data", "antigen_sequence", "antigen"]
    if "antibody" in param_lower:
        return ["antibody_data", "antibody_sequence", "airr_results", "antibody_analysis"]
    # Unknown parameter - accept any type (no restriction)
    return []


def _semantic_types_match(tool_output_types: List[str], param_expected_types: List[str]) -> bool:
    """
    Check if tool output semantic types match parameter expected types.
    
    Returns True if:
    - Parameter has no type restrictions (empty expected types)
    - There is at least one overlapping type between output and expected
    """
    if not param_expected_types:
        # No restriction on this parameter
        return True
    if not tool_output_types:
        # Unknown tool output - be conservative and allow
        return True
    # Check for overlap
    return bool(set(tool_output_types) & set(param_expected_types))


def _check_dependency_output_semantic_match(
    dep_task_id: str,
    param_name: str,
    state: "ExecutorState"
) -> bool:
    """
    Check if a dependency task's output semantically matches a parameter's expected type.
    
    This prevents errors like assigning AIRR results (antibody analysis) to antigen_file.
    """
    # Get the tool that produced the dependency output
    dep_task = next((t for t in state.subtasks if t.task_id == dep_task_id), None)
    if not dep_task:
        return True  # Can't determine, allow by default
    
    # Get tools used in dependency task
    dep_result = dep_task.result if hasattr(dep_task, 'result') and dep_task.result else {}
    dep_tools = dep_result.get("tools", [])
    
    if not dep_tools:
        return True  # Can't determine tools, allow by default
    
    # Get semantic types of the dependency's output
    tool_output_types: List[str] = []
    for tool_item in dep_tools:
        tool_name = tool_item if isinstance(tool_item, str) else tool_item.get("tool_name") or tool_item.get("name", "")
        if tool_name:
            tool_output_types.extend(_get_tool_output_semantic_types(tool_name))
    
    # Get expected semantic types of the parameter
    param_expected_types = _get_param_expected_semantic_types(param_name)
    
    # Check if types match
    match = _semantic_types_match(tool_output_types, param_expected_types)
    
    if not match:
        print(f"    [SemanticMatch] REJECTED: {param_name} expects {param_expected_types}, "
              f"but dep task {dep_task_id} outputs {tool_output_types}")
    
    return match


def _analyze_csv_columns(
    file_path: str,
    state: "ExecutorState",
    task_id: str
) -> Optional[List[str]]:
    """
    Analyze a CSV file to extract column names.
    
    This is crucial for data integration - we need to know the actual column names
    to properly map data between different tools.
    
    Args:
        file_path: Path to the CSV file
        state: Executor state for running code
        task_id: Task ID for context
    
    Returns:
        List of column names, or None if analysis failed
    """
    try:
        # 使用 codeact_executor 统一接口 (遵循架构原则)
        from utils.codeact_executor import analyze_file_structure
        
        # Get existing sandbox ID
        existing_sandbox_id = None
        if state.parent_state and hasattr(state.parent_state, 'merged_result'):
            merged_result = getattr(state.parent_state, 'merged_result', None)
            if isinstance(merged_result, dict):
                existing_sandbox_id = merged_result.get("opensandbox_id")
        
        result = analyze_file_structure(
            file_path=file_path,
            sandbox_id=existing_sandbox_id
        )
        
        if result.is_success() and result.parsed_result:
            columns = result.parsed_result.get("column_names", [])
            if columns:
                return columns
        
        if not result.is_success():
            print(f"    [CSV Analysis] Failed to analyze {file_path}: {result.error}")
            
    except Exception as e:
        print(f"    [CSV Analysis] Error analyzing {file_path}: {e}")
    
    return None


def _infer_output_file_description(
    tool_name: str,
    file_path: str,
    file_key: str,
    file_ext: str,
    tool_descriptions: Dict[str, str]
) -> str:
    """
    Infer a meaningful description for an output file.
    
    Args:
        tool_name: Name of the tool that produced this output
        file_path: Path to the file
        file_key: Key name from the output (e.g., 'output_file', 'result_file')
        file_ext: File extension
        tool_descriptions: Tool-specific descriptions dict
        
    Returns:
        Human-readable description of the file
    """
    # First, check tool-specific descriptions
    if file_ext in tool_descriptions:
        return tool_descriptions[file_ext]
    if file_key in tool_descriptions:
        return tool_descriptions[file_key]
    
    # Infer from file name patterns
    file_name = Path(file_path).stem.lower()
    
    if 'bind' in file_name or 'affinity' in file_name:
        return f"Binding affinity prediction results from {tool_name}"
    elif 'vdj' in file_name or 'airr' in file_name:
        return f"V(D)J annotation results in AIRR format from {tool_name}"
    elif 'integrated' in file_name or 'merged' in file_name:
        return f"Integrated/merged data from {tool_name}"
    elif 'distribution' in file_name:
        return f"Distribution analysis results from {tool_name}"
    elif 'visualization' in file_name or 'plot' in file_name:
        return f"Visualization output from {tool_name}"
    
    # Generic description based on file type
    ext_descriptions = {
        'csv': f"Tabular data output from {tool_name}",
        'rds': f"R data object from {tool_name}",
        'fasta': f"Sequence data in FASTA format from {tool_name}",
        'json': f"JSON data output from {tool_name}",
        'pdf': f"PDF visualization from {tool_name}",
        'png': f"Image output from {tool_name}",
        'svg': f"Vector graphics from {tool_name}"
    }
    
    return ext_descriptions.get(file_ext, f"Output file from {tool_name}")


def _get_compatible_param_types(tool_name: str, file_ext: str, file_key: str) -> List[str]:
    """
    Determine which parameter types this output file can be used for.
    
    This enables automatic parameter matching for subsequent tools.
    
    CRITICAL: This function must return parameter names that match exactly with
    the parameter names defined in tools_params_table.json.
    
    Args:
        tool_name: Source tool name
        file_ext: File extension
        file_key: Key name from the output
        
    Returns:
        List of compatible parameter type names
    """
    compatible = []
    tool_lower = tool_name.lower()
    file_key_lower = (file_key or "").lower()
    
    # CSV files can be used for various parameters
    if file_ext == 'csv':
        compatible.extend(['csv_file', 'input_csv', 'data_file', 'test_file', 'input_file'])
        
        # Tool-specific compatibility
        if 'metabcr' in tool_lower:
            compatible.extend(['binding_results', 'prediction_file', 'metabcr_output'])
        elif 'vdj' in tool_lower or 'igblast' in tool_lower or 'analyze_vdj_batch' in tool_lower:
            # CRITICAL: analyze_vdj_batch outputs AIRR format which can be used for airr_results parameter
            compatible.extend([
                'airr_file', 'vdj_results', 'annotation_file',
                'airr_results',  # This is the exact parameter name used by extract_cdr3_from_airr
                'airr_data', 'vdj_output'
            ])
        elif 'integrate' in tool_lower:
            compatible.extend(['integrated_data', 'bcr_data'])
        elif 'nettcr' in tool_lower or 'tcr' in tool_lower:
            compatible.extend(['prediction_file', 'binding_results', 'tcr_predictions'])
    
    # TSV files (AIRR format is often TSV)
    elif file_ext == 'tsv':
        compatible.extend(['tsv_file', 'data_file', 'input_file'])
        if 'vdj' in tool_lower or 'igblast' in tool_lower or 'analyze_vdj_batch' in tool_lower:
            compatible.extend(['airr_results', 'airr_file', 'airr_data'])
    
    # RDS files
    elif file_ext == 'rds':
        compatible.extend(['rds_file', 'seurat_object', 'r_data', 'rds_path', 'input_rds', 'input_file'])
    
    # FASTA files
    elif file_ext in ['fasta', 'fa']:
        compatible.extend(['fasta_file', 'sequences', 'input_fasta', 'sequence_file'])
    
    # JSON files
    elif file_ext == 'json':
        compatible.extend(['json_file', 'config_file', 'metadata'])
    
    # Check file_key for additional hints
    if 'airr' in file_key_lower:
        compatible.extend(['airr_results', 'airr_file', 'airr_data'])
    if 'output' in file_key_lower:
        compatible.extend(['output_file', 'result_file'])
    
    return compatible


def run_parameter_inference_pipeline(
    user_input: Optional[str],
    execution_plan: Optional[str],
    task_description: Optional[str],
    tool_name: str,
    input_params: List[Dict[str, Any]],
    llm: Optional[Any] = None,
    provided_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run a standalone parameter inference pipeline for testing and analysis.
    Returns parameters, missing parameters, definitions, and sources.
    """
    provided_params = provided_params or {}
    extracted_params = _extract_parameters_from_context(
        user_input=user_input or "",
        execution_plan=execution_plan,
        task_description=task_description or "",
        tool_name=tool_name,
        tool_params=input_params,
        llm=llm
    )
    for key, value in provided_params.items():
        extracted_params.setdefault(key, value)
    file_candidates = _extract_file_candidates_from_context(
        user_input=user_input,
        execution_plan=execution_plan,
        task_description=task_description
    )

    parameters: Dict[str, Any] = {}
    missing_parameters: List[str] = []
    definitions: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    for param in input_params:
        definition = _build_param_definition(param)
        param_name = definition["name"]
        if not param_name:
            continue
        definitions[param_name] = definition
        param_type = definition["type"]
        param_desc = definition.get("description", "")
        is_optional = definition["is_optional"]
        is_file = definition["is_file"]
        is_output = definition["is_output"]

        if param_name in extracted_params:
            raw_value = extracted_params.get(param_name)
            if isinstance(raw_value, dict) and "__value" in raw_value:
                raw_value = raw_value.get("__value")
            normalized = _normalize_inferred_param_value(param_name, param_type, raw_value)
            if normalized is not None and _value_matches_expected_types(param_name, param_type, normalized):
                signature_issue = _validate_file_signature(param_name, param_type, normalized, param_desc)
                if signature_issue:
                    normalized = None
            if normalized is not None:
                parameters[param_name] = _normalize_base_dir_value(param_name, normalized)
                sources[param_name] = "context"
                continue
            if not is_optional:
                missing_parameters.append(f"{tool_name}.{param_name}")
            continue

        if is_file and not is_output:
            candidate = _select_file_candidate(param_name, param_type, file_candidates, param_desc)
            if candidate:
                signature_issue = _validate_file_signature(param_name, param_type, candidate, param_desc)
                if not signature_issue:
                    parameters[param_name] = _normalize_base_dir_value(param_name, candidate)
                    sources[param_name] = "file_context"
                    continue
            if not is_optional:
                missing_parameters.append(f"{tool_name}.{param_name}")
            continue

        param_demo = param.get("deme") or param.get("demo", "")
        if param_demo and not is_file and not param_name.endswith("_fields"):
            # Convert demo value to correct type based on param_type
            converted_demo = _normalize_inferred_param_value(param_name, param_type, param_demo, verbose=False)
            if converted_demo is not None:
                parameters[param_name] = converted_demo
                sources[param_name] = "demo"
            else:
                parameters[param_name] = param_demo
                sources[param_name] = "demo"
            continue

        if not is_optional:
            missing_parameters.append(f"{tool_name}.{param_name}")

    return {
        "parameters": parameters,
        "missing_parameters": missing_parameters,
        "definitions": definitions,
        "sources": sources,
    }

def _is_output_param(param_name: str, param_type: Optional[str]) -> bool:
    """Check if a parameter is an output path parameter."""
    name_lower = (param_name or "").lower()
    type_lower = (param_type or "").lower()
    
    # CRITICAL: Exclude parameters that are INPUT files with "result" in the name
    # These are typically output files from PREVIOUS tools used as input
    input_exceptions = {
        "airr_results",      # AIRR format results (input from previous analysis)
        "results",           # Generic results file (often input)
        "result_file",       # Result file as input
        "analysis_results",  # Analysis results as input
        "prediction_results", # Prediction results as input
        "binding_results",   # Binding results as input
        "previous_results",  # Previous results as input
    }
    
    # If this is an input exception, it's NOT an output param
    if name_lower in input_exceptions:
        return False
    
    # Output directory parameters - these should be auto-filled with session output dir
    output_dir_params = {
        "base_dir", "output_dir", "out_dir", "save_dir", "result_dir",
        "output_directory", "output_path", "save_path", "result_path"
    }
    if name_lower in output_dir_params:
        return True
    
    return (
        "output" in name_lower
        or "result" in name_lower
        or "save" in name_lower
        or "output" in type_lower
        or "result" in type_lower
    )


def _label_indicates_output(label: str) -> bool:
    """Check if an explicit label implies output intent."""
    label_lower = (label or "").lower()
    return any(
        key in label_lower
        for key in ["output", "result", "save", "save to", "保存", "输出", "结果"]
    )


def _path_matches_expected_types(value: str, expected_exts: Optional[set[str]]) -> bool:
    """Check if path extension matches expected types (if specified)."""
    if not expected_exts:
        return True
    if not isinstance(value, str):
        return False
    _, ext = os.path.splitext(value.strip())
    if not ext:
        return False
    return ext.lower().lstrip(".") in expected_exts


def _is_zip_file(path: str) -> bool:
    try:
        import zipfile
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def _looks_like_excel_zip(path: str) -> bool:
    try:
        import zipfile
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        return any(
            name == "[Content_Types].xml" or name.startswith("xl/")
            for name in names
        )
    except Exception:
        return False


def _looks_like_rds_header(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        if header.startswith(b"RDX"):
            return True
        # gzip-compressed RDS
        if header.startswith(b"\x1f\x8b"):
            return True
    except Exception:
        return False
    return False


def _validate_file_signature(param_name: str, param_type: Optional[str], value: Any, param_desc: Optional[str] = None) -> Optional[str]:
    """Validate file signature against expected type; return error string if invalid."""
    if not isinstance(value, str):
        return None
    if not os.path.exists(value):
        return None
    if _looks_like_excel_zip(value):
        if param_name in {"csv_file", "feature_data_path"} or "csv" in param_name.lower():
            return "excel_zip_for_csv"
        expected_exts = _expected_file_extensions(param_name, param_type, param_desc)
        if expected_exts and expected_exts.intersection({"csv", "tsv"}):
            # Allow Excel for CSV/TSV (convert later)
            return None
        if expected_exts and expected_exts.intersection({"xlsx", "xls"}):
            return None
        return "excel_zip_for_non_excel"
    expected_exts = _expected_file_extensions(param_name, param_type, param_desc)
    if expected_exts and "rds" in expected_exts:
        if not _looks_like_rds_header(value):
            return "rds_header_mismatch"
    return None


def _extract_error_type_from_exec_output(output: Any) -> Optional[str]:
    """Extract error_type from execution output if present."""
    if not output:
        return None
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            return None
    if not isinstance(output, dict):
        return None
    final_result = output.get("final_result")
    if isinstance(final_result, dict):
        error_type = final_result.get("error_type")
        if error_type:
            return error_type
    return output.get("error_type")


def _record_hitl_request(state: "ExecutorState", task_id: str, request: Dict[str, Any]) -> None:
    """Record HITL request and preserve request history."""
    state.hitl_requests[task_id] = request
    state.hitl_request_history[task_id] = request


def _record_hitl_response(state: "ExecutorState", task_id: str, response: Dict[str, Any]) -> None:
    """Record HITL response and preserve response history."""
    state.hitl_responses[task_id] = response
    state.hitl_response_history[task_id] = response
    if task_id not in state.hitl_request_history:
        state.hitl_request_history[task_id] = {
            "type": "unknown",
            "task_id": task_id,
            "message": "response_without_request"
        }


def _extract_file_path_from_description(value: str) -> Optional[str]:
    """Extract actual file path from descriptive strings like 'TCR rds file (/path/to/file.rds)'."""
    if not isinstance(value, str):
        return None
    
    # 修复问题1: 跳过包含 "Generated X files in" 或类似模式中的目录路径
    # 这些路径通常是输出目录，而不是实际的文件
    import re
    skip_patterns = [
        r"[Gg]enerated\s+\d+\s+files?\s+in\s+(/[^\s\"']+)",
        r"[Ss]aved\s+to\s+directory\s*[:=]?\s*(/[^\s\"']+)",
        r"[Oo]utput\s+directory\s*[:=]?\s*(/[^\s\"']+)",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, value):
            # 如果值包含这些模式，提取实际文件路径而不是目录
            # 从 message 中提取可能是不安全的，返回 None 让调用者使用其他来源
            pass
    
    # Try to extract path from parentheses: "description (/path/to/file.rds)"
    patterns = [
        r'\(([^)]+\.(?:csv|tsv|json|xlsx|xls|rds|fasta|fa|fastq|txt|pdb))\)',  # Path in parentheses
        r'\[([^\]]+\.(?:csv|tsv|json|xlsx|xls|rds|fasta|fa|fastq|txt|pdb))\]',  # Path in brackets
        r'([A-Za-z]:\\[^\s\"\'()]+\.(?:csv|tsv|json|xlsx|xls|rds|fasta|fa|fastq|txt|pdb))',  # Windows absolute path
        r'(/[^\s\"\'()]+\.(?:csv|tsv|json|xlsx|xls|rds|fasta|fa|fastq|txt|pdb))',  # Unix absolute path
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, value)
        if matches:
            # Return the first match that looks like a valid path
            for match in matches:
                if _is_path_like(match):
                    return match.strip()
    
    # If no pattern matched, check if the entire string is a valid path
    if _is_path_like(value) and not any(char in value for char in ['(', ')', '[', ']']) or value.startswith('/') or (len(value) > 2 and value[1:3] == ":\\"):
        return value.strip()
    
    return None


def _validate_file_in_parameter_table(
    file_path: str,
    state: "ExecutorState",
    param_name: str
) -> bool:
    """
    Validate that a file path exists in the parameter table.
    
    CRITICAL: All file parameters must come from the parameter table.
    This ensures that files are properly tracked and have valid sources.
    
    Args:
        file_path: File path to validate
        state: Executor state containing parameter table
        param_name: Parameter name (for logging)
    
    Returns:
        True if file is in parameter table, False otherwise
    """
    if not state.parent_state:
        return False
    
    extracted_params = state.parent_state.extracted_parameters or {}
    
    # Check in files dictionary
    files = extracted_params.get("files", {})
    for file_key, file_info in files.items():
        if isinstance(file_info, dict):
            original_path = file_info.get("original_path", "")
            sandbox_path = file_info.get("sandbox_path", "")
            if file_path == original_path or file_path == sandbox_path:
                print(f"    [ParamValidate] File found in parameter table (files): {file_path}")
                return True
    
    # Check in sandbox_file_paths
    sandbox_file_paths = extracted_params.get("sandbox_file_paths", {})
    for key, path in sandbox_file_paths.items():
        if file_path == path:
            print(f"    [ParamValidate] File found in parameter table (sandbox_file_paths): {file_path}")
            return True
    
    # Check in task_outputs
    task_outputs = extracted_params.get("task_outputs", {})
    for task_output in task_outputs.values():
        if isinstance(task_output, dict):
            output_files = task_output.get("output_files", [])
            if file_path in output_files:
                print(f"    [ParamValidate] File found in parameter table (task_outputs): {file_path}")
                return True
    
    # Check if it's a user-provided file from preprocessing
    # User-provided files should be in files dictionary, but check file_analyses as fallback
    if hasattr(state.parent_state, 'file_analyses'):
        for analysis in state.parent_state.file_analyses:
            if isinstance(analysis, dict):
                sandbox_path = analysis.get("sandbox_path", "")
                original_path = analysis.get("original_path", "")
            else:
                sandbox_path = getattr(analysis, "sandbox_path", "")
                original_path = getattr(analysis, "original_path", "")
            
            if file_path == original_path or file_path == sandbox_path:
                print(f"    [ParamValidate] File found in parameter table (file_analyses): {file_path}")
                return True
    
    print(f"    [ParamValidate] File NOT found in parameter table: {file_path}")
    return False


def _is_path_like(value: str) -> bool:
    """Heuristic to detect file-like paths."""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    if _looks_like_status_message(stripped):
        return False
    if stripped.startswith(("http://", "https://")):
        return True
    if stripped.startswith(("\\\\", "~/", "./", "../")):
        return True
    if len(stripped) > 2 and stripped[1:3] == ":\\":
        return True
    if "/" in stripped or "\\" in stripped:
        return True
    known_exts = {
        ".csv", ".tsv", ".json", ".xlsx", ".xls", ".rds",
        ".fasta", ".fa", ".fastq", ".txt"
    }
    _, ext = os.path.splitext(stripped)
    return ext.lower() in known_exts


def _looks_like_descriptive_text(value: str) -> bool:
    """
    Check if a value looks like descriptive text rather than a potential parameter value.
    Used to suppress verbose warnings for obviously invalid values from LLM task decomposition.
    """
    if not isinstance(value, str):
        return False
    
    stripped = value.strip()
    if not stripped:
        return True
    
    # Skip warnings for obvious descriptive text patterns
    # Long sentences or phrases with spaces
    if len(stripped) > 30 and ' ' in stripped:
        return True
    
    # Text starting with common description markers
    if stripped.startswith(('**', '*', '-', '•', '•', 'Step', 'Phase', 'Month', 'Use ', 'Validate', 'Build', 'Create', 'Generate', 'Analyze', 'Integrate')):
        return True
    
    # Contains common description patterns
    desc_patterns = [
        'dataset', 'analysis', 'pipeline', 'method', 'model', 'data',
        'file from', 'output from', 'results from', 'integrated',
        'established', 'validate', 'quality', 'predictions'
    ]
    lower_value = stripped.lower()
    if any(p in lower_value for p in desc_patterns) and len(stripped) > 15:
        return True
    
    # DOI or paper reference patterns
    if stripped.startswith('10.') and '/' in stripped:
        return True
    
    # arXiv or bioRxiv patterns
    if 'arxiv' in lower_value or 'biorxiv' in lower_value or 'doi:' in lower_value:
        return True
    
    return False


def _normalize_inferred_param_value(param_name: str, param_type: Optional[str], value: Any, verbose: bool = True) -> Optional[Any]:
    """Normalize inferred parameter values; reject invalid file/dir values and type mismatches.
    
    Args:
        param_name: Name of the parameter
        param_type: Type of the parameter
        value: The value to normalize
        verbose: If False, suppress verbose warnings for obviously invalid values
    """
    if value is None:
        return None
    if isinstance(value, str) and _looks_like_status_message(value):
        return None
    
    param_name_lower = (param_name or "").lower()
    param_type_lower = (param_type or "").lower()
    
    # CRITICAL: Validate boolean parameters
    # If param should be boolean but got a string that's not a boolean, reject it
    is_boolean_param = "bool" in param_type_lower
    
    if is_boolean_param:
        # If value is already a boolean, return it
        if isinstance(value, bool):
            return value
        # If value is a string, try to convert to boolean
        if isinstance(value, str):
            # Skip warnings for obviously descriptive text
            if _looks_like_descriptive_text(value):
                if verbose:
                    print(f"  [WARN] Rejected descriptive text '{value[:50]}...' for boolean param '{param_name}'")
                return None
            # Check for true/false values
            lower_value = value.lower().strip()
            if lower_value in ("true", "yes", "1", "t", "y"):
                return True
            if lower_value in ("false", "no", "0", "f", "n"):
                return False
            # If it doesn't match boolean patterns, reject it
            if verbose and len(value) < 50:
                print(f"  [WARN] Rejected non-boolean value '{value}' for boolean param '{param_name}'")
            return None
        return None
    
    # CRITICAL: Validate numeric parameters
    # If param should be numeric but got a string that's not a number, reject it
    is_numeric_param = (
        "int" in param_type_lower or 
        "integer" in param_type_lower or
        "float" in param_type_lower or
        "number" in param_type_lower or
        param_name_lower in ["timeout", "duration", "count", "num", "max", "min", "limit", "size", "batch_size", "max_iter", "n_jobs"]
    )
    
    if is_numeric_param:
        # If value is already a number, return it
        if isinstance(value, (int, float)):
            return value
        # If value is a string, try to convert to number
        if isinstance(value, str):
            # Skip warnings for obviously descriptive text
            if _looks_like_descriptive_text(value):
                return None
            # Reject if it looks like a tool name or service name (contains underscore and letters)
            if "_" in value and any(c.isalpha() for c in value):
                if verbose and len(value) < 50:  # Only warn for short tool names
                    print(f"  [WARN] Rejected non-numeric value '{value}' for numeric param '{param_name}'")
                return None
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                # Only print warning for values that might actually be intended as numbers
                if verbose and not _looks_like_descriptive_text(value) and len(value) < 30:
                    print(f"  [WARN] Failed to convert '{value}' to number for param '{param_name}'")
                return None
        return None

    if _is_directory_param(param_name, param_type):
        if isinstance(value, str) and _is_path_like(value):
            base, ext = os.path.splitext(value)
            if ext:
                return os.path.dirname(value)
            return value
        return None

    if _is_file_type(param_name, param_type):
        if isinstance(value, str):
            # First try to extract actual file path from descriptive strings
            extracted_path = _extract_file_path_from_description(value)
            if extracted_path:
                return extracted_path
            # If extraction failed, check if the value itself is a valid path
            if not _is_path_like(value):
                return None
            
            # 修复问题1: 检查路径是否真的是一个文件（而非目录）
            # 当参数期望 CSV/RDS 等文件类型时，如果路径以这些扩展名结尾但实际是一个目录，
            # 则需要拒绝这个值（例如 output.csv 实际上是一个目录，而不是 CSV 文件）
            # 这种情况发生在 predict_tcr_binding_complete 的 output_dir 被设置为 xxx.csv 时
            expected_exts = _expected_file_extensions(param_name, param_type, None)
            if expected_exts:
                # 检查路径扩展名是否符合预期
                _, ext = os.path.splitext(value)
                if ext.lower().lstrip(".") in expected_exts:
                    # 路径有正确的扩展名，但我们无法在这里验证它是否真的是文件
                    # 因为此时代理可能无法访问沙箱文件系统
                    # 我们返回值，让后续的文件存在性检查来处理
                    pass
            return value
        return value

    return value


def _get_task_by_id(state: "ExecutorState", task_id: str) -> Optional[SubTask]:
    for task in state.subtasks:
        if task.task_id == task_id:
            return task
    return None


def _summarize_dependency_output_for_llm(output: Any, max_len: int = 400) -> str:
    """Summarize dependency output for LLM match checks."""
    if output is None:
        return "No output."
    if isinstance(output, dict):
        final_result = output.get("final_result")
        if isinstance(final_result, dict):
            message = final_result.get("message")
            output_file = final_result.get("output_file") or final_result.get("output_path") or final_result.get("result_path")
            parts = []
            if message:
                parts.append(f"final_result.message: {message}")
            if output_file:
                parts.append(f"final_result.output_path: {output_file}")
            if parts:
                return " | ".join(parts)[:max_len]
        message = output.get("message")
        if isinstance(message, str) and message:
            return message[:max_len]
        return _summarize_output_brief(output, max_len=max_len)
    if isinstance(output, list):
        return f"list(len={len(output)})"
    if isinstance(output, str):
        return output[:max_len]
    return str(output)[:max_len]


def _dependency_value_type_matches(param_name: str, param_type: Optional[str], value: Any) -> bool:
    """Hard type check: mismatch => not compatible."""
    normalized = _normalize_inferred_param_value(param_name, param_type, value)
    if normalized is None:
        return False
    if isinstance(normalized, str):
        expected_ext = _infer_expected_extension(param_type)
        if expected_ext:
            _, actual_ext = os.path.splitext(normalized)
            actual_ext = actual_ext.lower().lstrip(".")
            if actual_ext and actual_ext != expected_ext:
                return False
        if _is_directory_param(param_name, param_type):
            base, ext = os.path.splitext(normalized)
            if ext:
                return False
    return True


def _value_matches_expected_types(param_name: str, param_type: Optional[str], value: Any) -> bool:
    """Validate inferred value against parameter type expectations."""
    normalized = _normalize_inferred_param_value(param_name, param_type, value)
    if normalized is None:
        return False
    if isinstance(normalized, str):
        if _is_directory_param(param_name, param_type):
            base, ext = os.path.splitext(normalized)
            if ext:
                return False
            return True
        if _is_file_type(param_name, param_type):
            expected_exts = _expected_file_extensions(param_name, param_type)
            if expected_exts and os.path.exists(normalized) and _is_zip_file(normalized):
                if not expected_exts.intersection({"xlsx", "xls"}):
                    return False
            if expected_exts and not _path_matches_expected_types(normalized, expected_exts):
                return False
    return True


def _llm_match_dependency_output(
    llm: Optional[Any],
    param_name: str,
    param_type: Optional[str],
    param_desc: str,
    task_desc: str,
    dep_task_desc: str,
    dep_output_summary: str,
) -> Optional[bool]:
    """Ask LLM if dependency output semantically matches param requirement."""
    if not llm or not param_desc:
        return None
    prompt = f"""You are judging whether a dependency output is a suitable input for a tool parameter.

Parameter:
- name: {param_name}
- type: {param_type}
- description: {param_desc}

Current task description:
{task_desc}

Dependency task description:
{dep_task_desc}

Dependency output summary:
{dep_output_summary}

Decision rules:
1) If type or file semantics do not match, answer NO.
2) If type matches but semantic purpose differs (e.g., experimental CSV vs required RDS), answer NO.
3) Only answer YES when the dependency output clearly satisfies the parameter description.

Return only JSON:
{{"match": true|false, "reason": "<short reason>"}}
"""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content="You are a careful data integration assistant."),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        import json as json_module
        data = json_module.loads(match.group())
        if isinstance(data, dict) and "match" in data:
            return bool(data.get("match"))
    except Exception:
        return None
    return None


def _extract_dependency_candidates(
    dep_output: Any, 
    is_file_param: bool,
    expected_file_type: Optional[str] = None
) -> List[str]:
    """Collect file-like candidates from dependency output.
    
    IMPORTANT: This function now prioritizes OUTPUT-related keys to avoid
    extracting input files that were used by the dependency task.
    Only files from output_file, output_path, result_path, etc. are extracted.
    
    ENHANCED: Also extracts files based on semantic type (e.g., AIRR format)
    and from result_path arrays.
    
    NEW: When expected_file_type is provided (e.g., "csv", "rds", "predictions"),
    filters candidates to match the expected type, especially when result_path
    contains an array of multiple files.
    """
    candidates: List[str] = []
    if dep_output is None:
        return candidates

    # Keys that indicate OUTPUT files (the actual results of the task)
    OUTPUT_KEYS = {
        "output_file", "output_path", "result_path", "result_file",
        "output", "outputs", "result", "results", "filepath", "file_path",
        "saved_file", "saved_path", "created_file", "generated_file"
    }
    
    # Keys that indicate INPUT files (should be excluded)
    INPUT_KEYS = {
        "input_file", "input_path", "csv_file", "rds_file", "input",
        "source_file", "source_path", "antigen_file", "antibody_file",
        "fasta_file", "sequences"
    }
    
    # Suffixes that indicate the main output file (priority order)
    PREDICTION_SUFFIXES = ["_predictions.csv", "_prediction.csv", "predictions.csv", "prediction.csv"]
    STATISTICS_SUFFIXES = ["_statistics.csv", "_stats.csv", "statistics.csv", "stats.csv"]
    
    if isinstance(dep_output, dict):
        # Prefer final_result
        if "final_result" in dep_output and isinstance(dep_output["final_result"], dict):
            candidates.extend(_extract_dependency_candidates(dep_output["final_result"], is_file_param, expected_file_type))
        
        # CHANGED: Only check OUTPUT-related keys, skip INPUT-related keys
        for key, value in dep_output.items():
            key_lower = key.lower()
            
            # Skip input-related keys
            if key_lower in INPUT_KEYS:
                continue
            
            # Only include output-related keys or keys in final_result
            is_output_key = key_lower in OUTPUT_KEYS or "output" in key_lower or "result" in key_lower
            
            if is_output_key:
                if isinstance(value, str) and _is_path_like(value):
                    # 修复问题1: 检查是否可能是目录路径（看起来像文件但实际是目录）
                    # 当 result_path 数组存在时，message 中的路径可能是目录
                    # 跳过这种情况，优先使用 result_path 中的实际文件路径
                    if key_lower == "message":
                        # 检查是否存在 result_path
                        result_path = dep_output.get("result_path") if isinstance(dep_output, dict) else None
                        if result_path and isinstance(result_path, list) and len(result_path) > 0:
                            # 存在 result_path，跳过从 message 中提取路径
                            print(f"  [ExtractCandidates] Skipping message path '{value[:50]}...' because result_path exists")
                            continue
                    candidates.append(value)
                elif isinstance(value, list):
                    # Special handling for result_path arrays with multiple files
                    # When expected_file_type is specified, filter by file type
                    if expected_file_type and key_lower == "result_path":
                        for item in value:
                            if isinstance(item, str) and _is_path_like(item):
                                # Check if this item matches the expected type
                                item_lower = item.lower()
                                if expected_file_type == "csv" and item_lower.endswith(".csv"):
                                    # Prioritize prediction files over statistics files
                                    if any(s in item_lower for s in PREDICTION_SUFFIXES):
                                        candidates.insert(0, item)  # Add to front
                                    else:
                                        candidates.append(item)
                                elif expected_file_type == "rds" and item_lower.endswith(".rds"):
                                    candidates.append(item)
                                elif expected_file_type == "txt" and item_lower.endswith(".txt"):
                                    candidates.append(item)
                                elif expected_file_type == "predictions":
                                    if any(s in item_lower for s in PREDICTION_SUFFIXES):
                                        candidates.append(item)
                                elif expected_file_type is None:
                                    candidates.append(item)
                    else:
                        for item in value:
                            if isinstance(item, str) and _is_path_like(item):
                                candidates.append(item)
                elif isinstance(value, dict):
                    # Recursively check nested dicts, but only for output-related content
                    candidates.extend(_extract_dependency_candidates(value, is_file_param, expected_file_type))
        
        # Check result messages only
        messages = dep_output.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("type") != "result":
                    continue
                raw = message.get("raw")
                if isinstance(raw, dict):
                    # From raw result, only extract output-related keys
                    for raw_key, raw_value in raw.items():
                        raw_key_lower = raw_key.lower()
                        if raw_key_lower in INPUT_KEYS:
                            continue
                        is_output_key = raw_key_lower in OUTPUT_KEYS or "output" in raw_key_lower or "result" in raw_key_lower
                        if is_output_key:
                            if isinstance(raw_value, str) and _is_path_like(raw_value):
                                candidates.append(raw_value)
                            elif isinstance(raw_value, list):
                                # Special handling for result_path arrays with multiple files
                                if expected_file_type and raw_key_lower == "result_path":
                                    for item in raw_value:
                                        if isinstance(item, str) and _is_path_like(item):
                                            item_lower = item.lower()
                                            if expected_file_type == "csv" and item_lower.endswith(".csv"):
                                                if any(s in item_lower for s in PREDICTION_SUFFIXES):
                                                    candidates.insert(0, item)
                                                else:
                                                    candidates.append(item)
                                            elif expected_file_type == "rds" and item_lower.endswith(".rds"):
                                                candidates.append(item)
                                            elif expected_file_type == "txt" and item_lower.endswith(".txt"):
                                                candidates.append(item)
                                            elif expected_file_type == "predictions":
                                                if any(s in item_lower for s in PREDICTION_SUFFIXES):
                                                    candidates.append(item)
                                            elif expected_file_type is None:
                                                candidates.append(item)
                                else:
                                    for item in raw_value:
                                        if isinstance(item, str) and _is_path_like(item):
                                            candidates.append(item)
                            elif isinstance(raw_value, dict):
                                candidates.extend(_extract_dependency_candidates(raw_value, is_file_param, expected_file_type))
        
        # ENHANCED: Also check for specific output format fields (like AIRR format)
        # These are typically in execution_result
        execution_result = dep_output.get("execution_result")
        if isinstance(execution_result, dict):
            # Check final_result in execution_result
            exec_final = execution_result.get("final_result")
            if isinstance(exec_final, dict):
                candidates.extend(_extract_dependency_candidates(exec_final, is_file_param, expected_file_type))
            
            # Check messages in execution_result
            exec_messages = execution_result.get("messages")
            if isinstance(exec_messages, list):
                for msg in exec_messages:
                    if isinstance(msg, dict) and msg.get("type") == "result":
                        raw = msg.get("raw")
                        if isinstance(raw, dict):
                            candidates.extend(_extract_dependency_candidates(raw, is_file_param, expected_file_type))
                            
    elif isinstance(dep_output, list):
        for item in dep_output:
            if isinstance(item, str) and _is_path_like(item):
                candidates.append(item)
            elif isinstance(item, dict):
                candidates.extend(_extract_dependency_candidates(item, is_file_param, expected_file_type))
    elif isinstance(dep_output, str) and _is_path_like(dep_output):
        candidates.append(dep_output)
    
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    
    # If expected_file_type is specified and we have multiple candidates,
    # prioritize the most relevant one
    if expected_file_type and len(unique) > 1:
        # For CSV type, prefer prediction files over statistics files
        if expected_file_type == "csv":
            for item in unique:
                if any(s in item.lower() for s in PREDICTION_SUFFIXES):
                    # Return only the prediction file as the primary output
                    return [item]
    
    return unique


_dependency_match_cache: Dict[str, bool] = {}


def _dependency_match_with_llm(
    *,
    llm: Any,
    tool_name: str,
    param_name: str,
    param_type: Optional[str],
    param_desc: Optional[str],
    task_description: str,
    dependency_task_description: str,
    dependency_output_summary: str,
    candidate_value: str,
) -> bool:
    """Use LLM to judge if dependency output matches parameter intent."""
    if not llm:
        return False
    cache_key = "|".join([
        tool_name or "",
        param_name or "",
        param_type or "",
        param_desc or "",
        task_description or "",
        dependency_task_description or "",
        dependency_output_summary or "",
        candidate_value or "",
    ])
    if cache_key in _dependency_match_cache:
        return _dependency_match_cache[cache_key]
    prompt = f"""You are validating whether a dependency output matches a tool parameter.

Tool: {tool_name}
Parameter: {param_name}
Parameter type: {param_type or 'unknown'}
Parameter description: {param_desc or 'none'}

Current task description:
{task_description}

Dependency task description:
{dependency_task_description}

Dependency output summary:
{dependency_output_summary}

Candidate value:
{candidate_value}

Rules:
- If types do not match, answer NO.
- If types match, still answer NO unless the dependency output intent clearly matches the parameter description.
- Answer only YES or NO.
"""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content="You are a precise validator. Reply only YES or NO."),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        text = response.content.strip().upper() if hasattr(response, "content") else str(response).strip().upper()
        result = text.startswith("YES")
        _dependency_match_cache[cache_key] = result
        return result
    except Exception:
        return False


def _normalize_base_dir_value(param_name: str, value: Any) -> Any:
    """Normalize directory-like values to directory paths when a file path is provided."""
    if not isinstance(value, str):
        return value
    param_lower = (param_name or "").lower()
    if "base_dir" not in param_lower and "_dir" not in param_lower and "directory" not in param_lower:
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped.endswith(("/", "\\", os.sep)):
        return value
    basename = os.path.basename(stripped)
    _, ext = os.path.splitext(basename)
    if ext:
        return os.path.dirname(stripped)
    return value


def _infer_expected_extension(param_type: Optional[str]) -> Optional[str]:
    if not param_type:
        return None
    lower = param_type.lower()
    if "csv" in lower:
        return "csv"
    if "tsv" in lower:
        return "tsv"
    if "json" in lower:
        return "json"
    if "rds" in lower:
        return "rds"
    if "xlsx" in lower or "excel" in lower:
        return "xlsx"
    if "xls" in lower:
        return "xls"
    if "fasta" in lower:
        return "fasta"
    if "fastq" in lower:
        return "fastq"
    return None


def _build_converted_path(input_path: str, target_ext: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}.{target_ext}"


def _convert_excel_to_delimited_via_codeact(
    state: "ExecutorState",
    source_path: str,
    output_path: str,
    target_ext: str,
    parent_task_id: str,
    param_name: str
) -> None:
    """Use CodeAct to convert Excel file to CSV/TSV."""
    delimiter = "," if target_ext == "csv" else "\t"
    task_id = f"preprocess_{parent_task_id}_{param_name}_{uuid4().hex[:8]}"
    task_description = (
        f"Convert an Excel file to {target_ext.upper()}.\n"
        f"Input Excel file: {source_path}\n"
        f"Output file: {output_path}\n"
        f"Requirements:\n"
        f"- Use pandas to read the Excel file (use the first sheet if multiple).\n"
        f"- Save as {target_ext.upper()} with delimiter '{delimiter}'.\n"
        f"- Use UTF-8 encoding.\n"
        f"- Do not write index.\n"
        f"- Ensure output directory exists.\n"
        f"- Print JSON with key 'output_path' pointing to the saved file."
    )
    preprocess_task = SubTask(
        task_id=task_id,
        task_type=UserTaskType.EXECUTE_PLAN,
        content=task_description,
        result={"tools": [], "inputs": []}
    )
    codeact_input = codeact_input_mapper(
        executor_state=state,
        task=preprocess_task,
        execution_mode=CodeActExecutionMode.CODEACT,
        parameters={},
        parent_state=state.parent_state  # Pass parent_state for parameter inference
    )
    codeact_graph = build_codeact_subgraph()
    codeact_output = codeact_graph.invoke(codeact_input)
    codeact_state = CodeActState.model_validate(codeact_output) if isinstance(codeact_output, dict) else codeact_output
    exec_result = codeact_output_mapper(codeact_state)
    if exec_result.get("status") != "success":
        error_message = exec_result.get("error") or exec_result.get("output") or "unknown error"
        raise RuntimeError(f"CodeAct conversion failed for {source_path}: {error_message}")


def _preprocess_parameters_with_codeact(
    task: SubTask,
    parameters: Dict[str, Any],
    state: "ExecutorState"
) -> Dict[str, Any]:
    """Detect file type mismatches and auto-convert inputs before execution.
    
    IMPORTANT: This function also injects actual parameter values from
    parent_state.extracted_parameters when the input parameters contain
    schema definitions instead of actual values.
    """
    # Import at function level to avoid "cannot access local variable" error
    from pathlib import Path as _Path
    
    print(f"  [ParamPreprocess] Called for task {task.task_id}")
    print(f"  [ParamPreprocess] Input parameters: {parameters}")
    
    # ========== CRITICAL FIX: Inject actual values from extracted_parameters ==========
    # The input `parameters` may contain schema definitions (description, type, required)
    # instead of actual values. We need to get actual values from parent_state.
    actual_params = {}
    
    # Step 1: Check if parameters contain schema definitions (need actual values)
    has_schema_defs = any(
        isinstance(v, dict) and ('type' in v or 'description' in v or 'required' in v)
        for v in parameters.values()
    )
    
    if has_schema_defs or not parameters:
        # Parameters contain schema definitions, need to get actual values
        if state.parent_state:
            # Get extracted_parameters from parent_state
            extracted_params = getattr(state.parent_state, 'extracted_parameters', None)
            if extracted_params:
                # extracted_parameters has structure: {"params": {...}, "files": {...}}
                params_section = extracted_params.get("params", {})
                files_section = extracted_params.get("files", {})
                
                # Get tool parameter schema to map extracted values
                task_result = task.result if isinstance(task.result, dict) else {}
                tools = task_result.get("tools", [])
                tool_name = None
                if tools:
                    first_tool = tools[0]
                    if isinstance(first_tool, dict):
                        tool_name = first_tool.get("tool_name") or first_tool.get("name")
                    elif isinstance(first_tool, str):
                        tool_name = first_tool
                
                if tool_name:
                    print(f"  [ParamPreprocess] Tool: {tool_name}, injecting actual values from extracted_parameters")
                
                # Map extracted params to tool parameters
                # Common parameter name mappings
                param_mappings = {
                    # TCR-related
                    "peptides": ["target_peptide", "peptide"],
                    "peptide": ["target_peptide", "peptides"],
                    "test_file": ["meta_csv_file", "input_file"],
                    "input_data": ["meta_csv_file", "test_file"],
                    # File paths
                    "input_file": ["meta_csv_file", "test_file"],
                    "input_csv": ["meta_csv_file"],
                    "input_rds": ["meta_rds_file"],
                    # Session info
                    "output_dir": None,  # Will be set from sandbox_data_dir
                }
                
                # Inject values from params section
                for param_name, value in params_section.items():
                    actual_params[param_name] = value
                    print(f"  [ParamPreprocess] Injected param: {param_name} = {value}")
                
                # Inject file paths from files section
                for file_key, file_info in files_section.items():
                    if isinstance(file_info, dict):
                        sandbox_path = file_info.get("sandbox_path", "")
                        if sandbox_path:
                            # Map file key to parameter name
                            mapped = False
                            for param_name, source_keys in param_mappings.items():
                                if source_keys and file_key in source_keys:
                                    if param_name not in actual_params:
                                        actual_params[param_name] = sandbox_path
                                        print(f"  [ParamPreprocess] Mapped file: {param_name} = {sandbox_path}")
                                        mapped = True
                                        break
                            if not mapped:
                                # Use original file key as parameter name
                                actual_params[file_key] = sandbox_path
                                print(f"  [ParamPreprocess] Added file: {file_key} = {sandbox_path}")
                
                # Add session-specific paths
                sandbox_data_dir = getattr(state.parent_state, 'sandbox_data_dir', None)
                if sandbox_data_dir:
                    actual_params["output_dir"] = f"{sandbox_data_dir}/output"
                    print(f"  [ParamPreprocess] Set output_dir: {actual_params['output_dir']}")
                
                print(f"  [ParamPreprocess] Injected {len(actual_params)} actual parameter values")
    
    # Merge: actual_params takes precedence, then original parameters
    updated = {**parameters, **actual_params}
    
    # Filter out schema definitions (keep only actual values)
    filtered_params = {}
    for key, value in updated.items():
        if isinstance(value, dict) and ('type' in value or 'description' in value):
            # This is a schema definition, not an actual value - skip it
            print(f"  [ParamPreprocess] Filtering out schema definition: {key}")
            continue
        filtered_params[key] = value
    updated = filtered_params
    
    if not updated:
        print(f"  [ParamPreprocess] Skipping: parameters is empty after filtering")
        return updated
    
    task_result = task.result if isinstance(task.result, dict) else {}
    tools = task_result.get("tools", [])
    print(f"  [ParamPreprocess] Tools: {tools}")
    if not tools:
        print(f"  [ParamPreprocess] Skipping: no tools")
        return updated

    tools_params_map = _load_tools_params_table() or {}

    # Tools that require FASTA input
    fasta_required_tools = [
        'analyze_vdj_batch', 'vquest_analysis', 'igblast_query',
        'run_igblast', 'vdj_analysis', 'analyze_sequences'
    ]
    
    # Sequence parameter names that typically hold FASTA file paths
    sequence_params = ['sequences', 'fasta_file', 'input_fasta', 'sequence_file']
    
    # Sequence column names to look for in CSV
    # BCR (antibody) columns
    seq_columns_to_check = [
        'Heavy_DNA', 'heavy_dna', 'Light_DNA', 'light_dna',
        'Heavy', 'Light', 'sequence', 'Sequence', 'seq',
        'nt_sequence', 'aa_sequence', 'cdr3', 'CDR3',
        # Additional common sequence column names
        'variant_seq', 'variant_seq_1', 'variant_seq_2', 'variant_seq_3',
        'VH', 'VL', 'VHH', 'vh', 'vl', 'vhh',
        'heavy_chain', 'light_chain', 'HeavyChain', 'LightChain',
        'hc_seq', 'lc_seq', 'HC_seq', 'LC_seq',
        'full_sequence', 'dna_sequence', 'nucleotide_sequence',
        # TCR (T cell receptor) columns - for receptor_type: TCR
        # Alpha chain
        'alpha_dna', 'alpha_seq', 'alpha_chain', 'TRA', 'tra', 'tra_seq',
        'CDR3a', 'cdr3a', 'cdr3_alpha', 'TRAV',
        # Beta chain  
        'beta_dna', 'beta_seq', 'beta_chain', 'TRB', 'trb', 'trb_seq',
        'CDR3b', 'cdr3b', 'cdr3_beta', 'TRBV',
        # Gamma chain
        'gamma_dna', 'gamma_seq', 'gamma_chain', 'TRG', 'trg', 'trg_seq',
        'CDR3g', 'cdr3g', 'cdr3_gamma', 'TRGV',
        # Delta chain
        'delta_dna', 'delta_seq', 'delta_chain', 'TRD', 'trd', 'trd_seq',
        'CDR3d', 'cdr3d', 'cdr3_delta', 'TRDV',
        # Generic TCR patterns
        'tcr_alpha', 'tcr_beta', 'tcr_gamma', 'tcr_delta',
        'TCR_alpha', 'TCR_beta', 'TCR_gamma', 'TCR_delta',
    ]

    for tool_item in tools:
        tool_name = tool_item if isinstance(tool_item, str) else tool_item.get("tool_name") or tool_item.get("name", "")
        if not tool_name:
            continue
        
        # ========== CSV to FASTA conversion for sequence tools ==========
        is_fasta_tool = tool_name.lower() in [t.lower() for t in fasta_required_tools]
        if is_fasta_tool:
            print(f"  [ParamPreprocess] Checking FASTA requirement for tool: {tool_name}")
            print(f"  [ParamPreprocess] Current parameters: {list(updated.keys())}")
            
            # Get tool parameter definitions to find all sequence-related parameters
            tool_params = tools_params_map.get(tool_name)
            if not tool_params:
                tool_name_lower = tool_name.lower()
                for key in tools_params_map.keys():
                    key_lower = key.lower()
                    if tool_name_lower in key_lower or key_lower in tool_name_lower:
                        tool_params = tools_params_map.get(key)
                        break
            
            # Collect all parameters that accept FASTA/CSV sequences
            sequence_param_names = set(sequence_params)  # Start with default list
            if tool_params:
                input_params = tool_params.get("input_params", [])
                for param in input_params:
                    param_name = param.get("name", "")
                    param_type = param.get("type", "").lower()
                    param_desc = param.get("description", "").lower()
                    # Check if parameter accepts sequences/FASTA/CSV
                    if any(keyword in param_type or keyword in param_desc 
                           for keyword in ['fasta', 'sequence', 'csv']):
                        sequence_param_names.add(param_name)
                        print(f"  [ParamPreprocess] Added sequence parameter from tool definition: {param_name}")
            
            print(f"  [ParamPreprocess] Sequence parameter names to check: {sequence_param_names}")
            
            for param_name in sequence_param_names:
                if param_name not in updated:
                    continue
                param_value = updated.get(param_name)
                print(f"  [ParamPreprocess] Found param {param_name} = {param_value}")
                if not isinstance(param_value, str):
                    print(f"  [ParamPreprocess] Skipping: param_value is not str, type={type(param_value)}")
                    continue
                
                # Check if it's a CSV file (more flexible detection)
                param_lower = param_value.lower()
                is_csv = (
                    param_lower.endswith('.csv') or
                    '/csv' in param_lower or
                    'csv' in param_lower.split('/')[-1].split('.')  # Check filename
                )
                
                # Also check if it's already a FASTA file
                is_fasta = (
                    param_lower.endswith(('.fasta', '.fa', '.fas')) or
                    '/fasta' in param_lower or
                    'fasta' in param_lower.split('/')[-1].split('.')
                )
                
                if is_fasta:
                    print(f"  [ParamPreprocess] Already a FASTA file, no conversion needed: {param_value}")
                    continue
                
                if not is_csv:
                    # Check if it might be a CSV based on file analysis
                    if state.parent_state and hasattr(state.parent_state, 'file_analyses'):
                        for analysis in state.parent_state.file_analyses:
                            # Handle both dict and FileAnalysis object
                            if isinstance(analysis, dict):
                                sandbox_path = analysis.get("sandbox_path", "")
                                original_path = analysis.get("original_path", "")
                                file_type = analysis.get("file_type", "")
                            else:
                                # FileAnalysis object
                                sandbox_path = getattr(analysis, "sandbox_path", "")
                                original_path = getattr(analysis, "original_path", "")
                                file_type = getattr(analysis, "file_type", "")
                            
                            if sandbox_path == param_value or original_path == param_value:
                                if file_type == 'csv':
                                    is_csv = True
                                    print(f"  [ParamPreprocess] Detected CSV from file analysis: {param_value}")
                                    break
                
                if not is_csv:
                    print(f"  [ParamPreprocess] Skipping: not detected as CSV file")
                    continue
                
                print(f"  [ParamPreprocess] Tool {tool_name} requires FASTA, but got CSV: {param_value}")
                
                # CRITICAL: Use the SANDBOX path (param_value), not the original path!
                # The original path (/data/benchmark_data/...) may not be accessible inside the sandbox container.
                # The sandbox path (/tmp/sessions/{session_id}/input/...) has the file already copied there.
                csv_path_for_conversion = param_value
                
                # If the param_value is the original path, try to find the corresponding sandbox path
                if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
                    extracted = state.parent_state.extracted_parameters or {}
                    files = extracted.get("files", {})
                    for file_key, file_info in files.items():
                        if isinstance(file_info, dict):
                            original_path = file_info.get("original_path", "")
                            sandbox_path = file_info.get("sandbox_path", "")
                            # If current param_value is original path, use sandbox path instead
                            if original_path and param_value == original_path and sandbox_path:
                                csv_path_for_conversion = sandbox_path
                                print(f"  [ParamPreprocess] Using sandbox path instead of original: {csv_path_for_conversion}")
                                break
                            # If current param_value is already sandbox path, keep it
                            elif sandbox_path and param_value == sandbox_path:
                                print(f"  [ParamPreprocess] Already using sandbox path: {csv_path_for_conversion}")
                                break
                
                print(f"  [ParamPreprocess] CSV path for conversion: {csv_path_for_conversion}")
                
                # Try to convert CSV to FASTA in the same sandbox session
                try:
                    fasta_path = _convert_csv_to_fasta_via_mcp(
                        csv_path=csv_path_for_conversion,  # Use sandbox path where file is already copied
                        state=state,
                        task_id=task.task_id
                    )
                    if fasta_path:
                        print(f"  [ParamPreprocess] Converted CSV to FASTA: {fasta_path}")
                        updated[param_name] = fasta_path
                    else:
                        print(f"  [ParamPreprocess] CSV to FASTA conversion failed or no sequence columns found")
                except Exception as conv_error:
                    print(f"  [ParamPreprocess] CSV to FASTA conversion error: {conv_error}")
        
        # ========== RDS to CSV conversion ==========
        # Check if tool requires CSV but got RDS file
        tool_params = tools_params_map.get(tool_name)
        if not tool_params:
            tool_name_lower = tool_name.lower()
            for key in tools_params_map.keys():
                key_lower = key.lower()
                if tool_name_lower in key_lower or key_lower in tool_name_lower:
                    tool_params = tools_params_map.get(key)
                    break
        
        if tool_params:
            input_params = tool_params.get("input_params", [])
            # Get service name to check if it's mixtcrpred or nettcr
            service_name = tool_params.get("service", "").lower()
            is_mixtcrpred_or_nettcr = service_name in ["mixtcrpred", "nettcr"]
            
            for param in input_params:
                param_name = param.get("name", "")
                param_type = param.get("type", "").lower()
                param_desc = param.get("description", "").lower()
                
                # Check if parameter expects CSV
                # Note: param_type can be "csv file", "csv_file", "Optional[csv file]", etc.
                expects_csv = (
                    "csv" in param_type or
                    "csv" in param_desc or
                    param_name.lower().endswith("_csv") or
                    param_name.lower() in ["csv_file", "input_csv", "data_csv", "input_file"]  # input_file may need CSV for some tools
                )
                
                # Special case: if param_name is "input_file" and param_type contains "csv", definitely expects CSV
                if param_name.lower() == "input_file" and "csv" in param_type:
                    expects_csv = True
                
                if param_name not in updated:
                    continue
                
                param_value = updated.get(param_name)
                if not isinstance(param_value, str):
                    continue
                
                # Special handling for mixtcrpred and nettcr services:
                # These services accept file paths (RDS or CSV) even if parameter type is "str"
                # If input_data or input_file contains an RDS file path, convert it to CSV
                if is_mixtcrpred_or_nettcr and param_name.lower() in ["input_data", "input_file"]:
                    param_lower = param_value.lower()
                    is_rds = (
                        param_lower.endswith(('.rds', '.RDS')) or
                        '/rds' in param_lower or
                        'rds' in param_lower.split('/')[-1].split('.')
                    )
                    if is_rds:
                        expects_csv = True
                        print(f"  [ParamPreprocess] Detected RDS file for {service_name} service parameter {param_name}, will convert to CSV")
                
                if not expects_csv:
                    continue
                
                # Check if it's an RDS file
                param_lower = param_value.lower()
                is_rds = (
                    param_lower.endswith(('.rds', '.RDS')) or
                    '/rds' in param_lower or
                    'rds' in param_lower.split('/')[-1].split('.')
                )
                
                # Also check if it's already a CSV file
                is_csv = (
                    param_lower.endswith('.csv') or
                    '/csv' in param_lower or
                    'csv' in param_lower.split('/')[-1].split('.')
                )
                
                if is_csv:
                    print(f"  [ParamPreprocess] Already a CSV file, no conversion needed: {param_value}")
                    continue
                
                if not is_rds:
                    # Check if it might be RDS based on file analysis
                    if state.parent_state and hasattr(state.parent_state, 'file_analyses'):
                        for analysis in state.parent_state.file_analyses:
                            if isinstance(analysis, dict):
                                sandbox_path = analysis.get("sandbox_path", "")
                                original_path = analysis.get("original_path", "")
                                file_type = analysis.get("file_type", "")
                            else:
                                sandbox_path = getattr(analysis, "sandbox_path", "")
                                original_path = getattr(analysis, "original_path", "")
                                file_type = getattr(analysis, "file_type", "")
                            
                            if sandbox_path == param_value or original_path == param_value:
                                if file_type.lower() == 'rds':
                                    is_rds = True
                                    print(f"  [ParamPreprocess] Detected RDS from file analysis: {param_value}")
                                    break
                
                if not is_rds:
                    continue
                
                print(f"  [ParamPreprocess] Tool {tool_name} requires CSV, but got RDS: {param_value}")
                
                # Use the SANDBOX path (param_value), not the original path
                rds_path_for_conversion = param_value
                
                # If the param_value is the original path, try to find the corresponding sandbox path
                if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
                    extracted = state.parent_state.extracted_parameters or {}
                    files = extracted.get("files", {})
                    for file_key, file_info in files.items():
                        if isinstance(file_info, dict):
                            original_path = file_info.get("original_path", "")
                            sandbox_path = file_info.get("sandbox_path", "")
                            if original_path and param_value == original_path and sandbox_path:
                                rds_path_for_conversion = sandbox_path
                                print(f"  [ParamPreprocess] Using sandbox path instead of original: {rds_path_for_conversion}")
                                break
                            elif sandbox_path and param_value == sandbox_path:
                                print(f"  [ParamPreprocess] Already using sandbox path: {rds_path_for_conversion}")
                                break
                
                print(f"  [ParamPreprocess] RDS path for conversion: {rds_path_for_conversion}")
                
                # Try to convert RDS to CSV
                try:
                    # Get sandbox_id from state
                    sandbox_id = None
                    if state.parent_state:
                        merged_result = getattr(state.parent_state, 'merged_result', None) or {}
                        sandbox_id = merged_result.get('opensandbox_id')
                    
                    csv_path = _auto_convert_rds_to_csv(
                        rds_path=rds_path_for_conversion,
                        state=state,
                        sandbox_id=sandbox_id
                    )
                    if csv_path:
                        print(f"  [ParamPreprocess] Converted RDS to CSV: {csv_path}")
                        updated[param_name] = csv_path
                        
                        # CRITICAL: Add converted CSV file to parameter table
                        # This ensures the converted file can be used by subsequent tasks
                        if state.parent_state:
                            csv_name = _Path(csv_path).name
                            file_key = f"rds_to_csv_{csv_name}"
                            
                            # Add to files dictionary
                            if not hasattr(state.parent_state, 'extracted_parameters') or not state.parent_state.extracted_parameters:
                                state.parent_state.extracted_parameters = {
                                    "files": {},
                                    "sandbox_file_paths": {},
                                    "task_outputs": {}
                                }
                            
                            extracted_params = state.parent_state.extracted_parameters
                            files = extracted_params.get("files", {})
                            sandbox_file_paths = extracted_params.get("sandbox_file_paths", {})
                            
                            # Add file metadata
                            files[file_key] = {
                                "original_path": csv_path,
                                "sandbox_path": csv_path,
                                "file_type": "csv",
                                "description": f"CSV file converted from RDS: {rds_path_for_conversion}",
                                "source": "rds_conversion",
                                "source_rds": rds_path_for_conversion
                            }
                            
                            # Add to sandbox_file_paths
                            sandbox_file_paths[file_key] = csv_path
                            
                            extracted_params["files"] = files
                            extracted_params["sandbox_file_paths"] = sandbox_file_paths
                            
                            print(f"  [ParamPreprocess] Added converted CSV to parameter table: {csv_path}")
                    else:
                        print(f"  [ParamPreprocess] RDS to CSV conversion failed")
                except Exception as conv_error:
                    print(f"  [ParamPreprocess] RDS to CSV conversion error: {conv_error}")
        
        # ========== Excel to CSV/TSV conversion (existing logic) ==========
        tool_params = tools_params_map.get(tool_name)
        if not tool_params:
            tool_name_lower = tool_name.lower()
            for key in tools_params_map.keys():
                key_lower = key.lower()
                if tool_name_lower in key_lower or key_lower in tool_name_lower:
                    tool_params = tools_params_map.get(key)
                    break
        if not tool_params:
            continue

        input_params = tool_params.get("input_params", [])
        for param in input_params:
            param_name = param.get("name", "")
            param_type = param.get("type", "")
            if not param_name or param_name not in updated:
                continue
            if "output" in param_name.lower():
                continue
            expected_ext = _infer_expected_extension(param_type)
            if expected_ext not in {"csv", "tsv"}:
                continue

            param_value = updated.get(param_name)
            if not isinstance(param_value, str):
                continue
            if not os.path.exists(param_value):
                # Skip local preprocessing for remote-only paths
                continue
            _, actual_ext = os.path.splitext(param_value)
            actual_ext = actual_ext.lower().lstrip(".")

            if actual_ext in {"xls", "xlsx"} or _looks_like_excel_zip(param_value):
                converted_path = _build_converted_path(param_value, expected_ext)
                try:
                    _convert_excel_to_delimited_via_codeact(
                        state=state,
                        source_path=param_value,
                        output_path=converted_path,
                        target_ext=expected_ext,
                        parent_task_id=task.task_id,
                        param_name=param_name
                    )
                    updated[param_name] = converted_path
                except Exception as conversion_error:
                    print(f"  ⚠ Excel conversion failed for {param_name}: {conversion_error}")

        # ========== NetTCR CSV preprocessing: add peptide column if missing ==========
        # NetTCR-2.2 requires CSV with columns: CDR3a, CDR3b, peptide (legacy format)
        # If peptide column is missing, we need to add it using the target peptide
        nettcr_tools = [
            'predict_tcr_binding_fast', 'predict_tcr_binding_ensemble',
            'predict_tcr_binding_complete', 'validate_tcr_input'
        ]
        is_nettcr_tool = tool_name.lower() in [t.lower() for t in nettcr_tools]
        
        if is_nettcr_tool:
            print(f"  [ParamPreprocess] Checking NetTCR CSV preprocessing for tool: {tool_name}")
            
            # Find CSV input parameters for NetTCR
            nettcr_csv_params = ['test_file', 'input_csv', 'input_file', 'csv_file']
            for param_name in nettcr_csv_params:
                if param_name not in updated:
                    continue
                    
                param_value = updated.get(param_name)
                print(f"  [ParamPreprocess] Checking param {param_name} = {param_value}")
                
                if not isinstance(param_value, str):
                    continue
                
                # Check if it's a CSV file
                param_lower = param_value.lower()
                is_csv = param_lower.endswith('.csv') or 'csv' in param_lower.split('/')[-1].split('.')
                
                if not is_csv:
                    continue
                
                print(f"  [ParamPreprocess] NetTCR input is CSV: {param_value}")
                
                # Try to add peptide column if needed
                try:
                    prepared_csv_path = _prepare_nettcr_csv_with_peptide(
                        csv_path=param_value,
                        state=state,
                        task_id=task.task_id
                    )
                    if prepared_csv_path and prepared_csv_path != param_value:
                        print(f"  [ParamPreprocess] Prepared NetTCR CSV with peptide: {prepared_csv_path}")
                        updated[param_name] = prepared_csv_path
                    else:
                        print(f"  [ParamPreprocess] CSV already has peptide column or preparation not needed")
                except Exception as nettcr_error:
                    print(f"  [ParamPreprocess] NetTCR CSV preparation error: {nettcr_error}")

    return updated


def _convert_csv_to_fasta_via_mcp(
    csv_path: str,
    state: "ExecutorState",
    task_id: str
) -> Optional[str]:
    """
    Convert CSV file to FASTA format directly in sandbox using Python built-in modules.
    
    IMPORTANT: This does NOT use MCP tools (convert_csv_to_fasta doesn't exist).
    It directly reads CSV and writes FASTA using Python's csv module.
    
    CRITICAL: Output MUST be written to a SHARED directory that is accessible by both:
    1. The conversion sandbox instance
    2. The MCP service (igblast) that will read the FASTA file
    
    The shared directory is: /tmp/sessions/{session_id}/output/
    This is mounted as a shared volume across all sandbox instances in the same session.
    
    Args:
        csv_path: Path to CSV file (should be original path on shared storage like /data/...)
        state: Executor state
        task_id: Parent task ID
        
    Returns:
        Path to generated FASTA file, or None if conversion failed
    """
    from pathlib import Path
    
    csv_path_normalized = csv_path.replace("\\", "/")
    csv_name = Path(csv_path_normalized).stem
    
    # SOLUTION: Write FASTA to the SESSION directory which is SHARED between
    # the sandbox and MCP services!
    #
    # PATH MAPPING:
    # - Container path: /tmp/sessions/{session_id}/... (used by sandbox code)
    # - Server path: /data/sessions/{session_id}/... (used by MCP services)
    # These are the SAME directory via volume mount: /data/sessions -> /tmp/sessions
    #
    # The conversion code runs INSIDE the container, so it needs /tmp/sessions path.
    # The returned path needs to be /data/sessions for MCP services.
    
    import re
    
    # Get sandbox_data_dir from parent_state (this is now SERVER path: /data/sessions/...)
    sandbox_data_dir = None
    if state.parent_state and hasattr(state.parent_state, 'sandbox_data_dir'):
        sandbox_data_dir = getattr(state.parent_state, 'sandbox_data_dir', None)
    
    if sandbox_data_dir:
        sandbox_data_dir = sandbox_data_dir.replace("\\", "/")
        # Convert server path to container path for sandbox code execution
        if sandbox_data_dir.startswith("/data/sessions/"):
            container_data_dir = sandbox_data_dir.replace("/data/sessions/", "/tmp/sessions/", 1)
        else:
            container_data_dir = sandbox_data_dir
        fasta_output_path = f"{container_data_dir}/output/{csv_name}_sequences.fasta"
        print(f"  [ParamPreprocess] Using sandbox_data_dir: {sandbox_data_dir}")
        print(f"  [ParamPreprocess] Container output path: {fasta_output_path}")
    else:
        # Fallback: try to extract session_id from various sources
        session_id = None
        
        # Try to extract from extracted_parameters
        # Note: sandbox_path is now SERVER path (/data/sessions/...) not container path
        if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
            extracted = state.parent_state.extracted_parameters or {}
            files = extracted.get("files", {})
            for file_key, file_info in files.items():
                if isinstance(file_info, dict):
                    sandbox_path = file_info.get("sandbox_path", "")
                    # Try both /data/sessions/ and /tmp/sessions/ patterns
                    session_match = re.search(r'/(?:data|tmp)/sessions/([^/]+)/', sandbox_path)
                    if session_match:
                        session_id = session_match.group(1)
                        break
        
        # If still not found, try to get from merged_result
        if not session_id and state.parent_state and hasattr(state.parent_state, 'merged_result'):
            merged_result = getattr(state.parent_state, 'merged_result', {}) or {}
            sandbox_data_dir_from_result = merged_result.get("sandbox_data_dir", "")
            if sandbox_data_dir_from_result:
                session_match = re.search(r'/(?:data|tmp)/sessions/([^/]+)', sandbox_data_dir_from_result)
                if session_match:
                    session_id = session_match.group(1)
        
        if session_id:
            # Use CONTAINER path for sandbox code execution
            fasta_output_path = f"/tmp/sessions/{session_id}/output/{csv_name}_sequences.fasta"
            print(f"  [ParamPreprocess] Extracted session_id: {session_id}")
        else:
            # Last fallback: use /tmp/ (may not be accessible by MCP services)
            fasta_output_path = f"/tmp/{csv_name}_sequences.fasta"
            print(f"  [ParamPreprocess] WARNING: No session directory found!")
            print(f"  [ParamPreprocess] FASTA may not be accessible by MCP services!")
    
    fallback_fasta_path = f"/tmp/{csv_name}_sequences.fasta"
    
    print(f"  [ParamPreprocess] Converting CSV to FASTA directly in sandbox (no MCP)")
    print(f"  [ParamPreprocess] Input CSV: {csv_path_normalized}")
    print(f"  [ParamPreprocess] Primary output path: {fasta_output_path}")
    print(f"  [ParamPreprocess] Fallback output path: {fallback_fasta_path}")
    
    # Direct conversion code using ONLY Python built-in modules (no pandas!)
    # This code tries multiple output locations to find one that is both:
    # 1. Writable from the sandbox
    # 2. Accessible by MCP services
    conversion_code = f'''
import csv
import os
import re

csv_path = "{csv_path_normalized}"
primary_fasta_path = "{fasta_output_path}"
fallback_fasta_path = "{fallback_fasta_path}"

# Sequence column patterns (case-insensitive matching)
# BCR (antibody) patterns
seq_column_patterns = [
    r"^heavy_dna$", r"^light_dna$", r"^heavy$", r"^light$",
    r"^sequence$", r"^seq$", r"^nt_sequence$", r"^aa_sequence$",
    r"^cdr3$", r"^vh$", r"^vl$", r"^vhh$",
    r"^heavy_chain$", r"^light_chain$", r"^hc_seq$", r"^lc_seq$",
    r"^full_sequence$", r"^dna_sequence$", r"^nucleotide_sequence$",
    r"^variant_seq.*$",  # Match variant_seq, variant_seq_1, variant_seq_2, etc.
    # TCR (T cell receptor) patterns - for receptor_type: TCR
    # Alpha chain
    r"^alpha_dna$", r"^alpha_seq$", r"^alpha_chain$", r"^tra$", r"^tra_seq$",
    r"^cdr3a$", r"^cdr3_alpha$", r"^trav.*$",  # TRAV, TRAV1, TRAV2, etc.
    # Beta chain
    r"^beta_dna$", r"^beta_seq$", r"^beta_chain$", r"^trb$", r"^trb_seq$",
    r"^cdr3b$", r"^cdr3_beta$", r"^trbv.*$",  # TRBV, TRBV1, TRBV2, etc.
    # Gamma chain
    r"^gamma_dna$", r"^gamma_seq$", r"^gamma_chain$", r"^trg$", r"^trg_seq$",
    r"^cdr3g$", r"^cdr3_gamma$", r"^trgv.*$",
    # Delta chain
    r"^delta_dna$", r"^delta_seq$", r"^delta_chain$", r"^trd$", r"^trd_seq$",
    r"^cdr3d$", r"^cdr3_delta$", r"^trdv.*$",
    # Generic TCR patterns
    r"^tcr_alpha.*$", r"^tcr_beta.*$", r"^tcr_gamma.*$", r"^tcr_delta.*$",
]

def is_valid_sequence(s):
    """Check if string looks like a valid biological sequence."""
    if not s or not isinstance(s, str):
        return False
    s = s.strip().upper()
    if len(s) < 10:
        return False
    # Check if it looks like nucleotide or amino acid sequence
    valid_chars = set("ACDEFGHIKLMNPQRSTVWYX*-UYBRN")
    seq_chars = set(s)
    # Allow up to 10% invalid characters
    return len(seq_chars - valid_chars) / max(len(seq_chars), 1) < 0.1

def matches_seq_pattern(col_name):
    """Check if column name matches a sequence pattern."""
    col_lower = col_name.lower().strip()
    for pattern in seq_column_patterns:
        if re.match(pattern, col_lower):
            return True
    return False

def write_fasta(fasta_path, rows, seq_cols, id_col):
    """Write FASTA file to the specified path."""
    output_dir = os.path.dirname(fasta_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        try:
            os.chmod(output_dir, 0o777)  # Allow all users to write
        except Exception:
            pass
    
    sequence_count = 0
    with open(fasta_path, "w", encoding="utf-8") as fasta_f:
        for i, row in enumerate(rows):
            for col in seq_cols:
                sequence = row.get(col)
                if is_valid_sequence(sequence):
                    if id_col and row.get(id_col):
                        seq_id = f"{{row[id_col]}}_{{col}}"
                    else:
                        seq_id = f"seq_{{i+1}}_{{col}}"
                    fasta_f.write(f">{{seq_id}}\\n{{sequence}}\\n")
                    sequence_count += 1
    return sequence_count

try:
    if not os.path.exists(csv_path):
        print(f"__CSV_NOT_FOUND__:{{csv_path}}")
    else:
        # Read CSV and find sequence columns
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames or []
        
        print(f"Read CSV: {{len(rows)}} rows, columns: {{columns}}")
        
        # Find sequence columns
        seq_cols = [c for c in columns if matches_seq_pattern(c)]
        print(f"Found sequence columns: {{seq_cols}}")
        
        if not seq_cols:
            print("__CSV_NO_SEQ_COLUMNS__")
        else:
            # Find ID column for naming sequences
            id_col = None
            for candidate in ['main_name', 'name', 'id', 'ID', 'sample_id', 'cell_id', 'barcode']:
                if candidate in columns:
                    id_col = candidate
                    break
            
            # Try primary path first (same directory as CSV, accessible by MCP)
            fasta_path = None
            sequence_count = 0
            
            try:
                sequence_count = write_fasta(primary_fasta_path, rows, seq_cols, id_col)
                if sequence_count > 0:
                    fasta_path = primary_fasta_path
                    print(f"Successfully wrote to primary path: {{primary_fasta_path}}")
            except OSError as e:
                print(f"Primary path failed ({{e}}), trying fallback...")
                try:
                    sequence_count = write_fasta(fallback_fasta_path, rows, seq_cols, id_col)
                    if sequence_count > 0:
                        fasta_path = fallback_fasta_path
                        print(f"Wrote to fallback path: {{fallback_fasta_path}}")
                        print("WARNING: Fallback path /tmp/ may not be accessible by MCP services!")
                except OSError as e2:
                    print(f"__CSV_TO_FASTA_ERROR__:Both paths failed: {{e}}, {{e2}}")
            
            if fasta_path and sequence_count > 0:
                print(f"__CSV_TO_FASTA_SUCCESS__:{{fasta_path}}:{{sequence_count}}")
            elif sequence_count == 0:
                print("__CSV_NO_VALID_SEQS__")
except Exception as e:
    import traceback
    print(f"__CSV_TO_FASTA_ERROR__:{{str(e)}}")
    traceback.print_exc()
'''
    
    # 使用 codeact_executor 统一接口 (遵循架构原则)
    try:
        from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
        
        if is_codeact_available():
            # Try to reuse existing sandbox from session
            existing_sandbox_id = None
            if state.parent_state:
                merged_result = getattr(state.parent_state, 'merged_result', None) or {}
                existing_sandbox_id = merged_result.get('opensandbox_id')
                if existing_sandbox_id:
                    print(f"  [ParamPreprocess] Reusing session sandbox for conversion: {existing_sandbox_id}")
            
            result = execute_code_via_codeact(
                task_description=f"将 CSV {csv_path_normalized} 转换为 FASTA 格式",
                code_template=conversion_code,
                sandbox_id=existing_sandbox_id,
                timeout_seconds=60,
                keep_alive=True
            )
            
            if result.is_success():
                stdout = result.output
                print(f"  [ParamPreprocess] Conversion output:")
                print(f"    stdout: {stdout[:1000]}..." if len(stdout) > 1000 else f"    stdout: {stdout}")
                
                if "__CSV_TO_FASTA_SUCCESS__:" in stdout:
                    for line in stdout.split("\n"):
                        if "__CSV_TO_FASTA_SUCCESS__:" in line:
                            parts = line.split(":")
                            if len(parts) >= 3:
                                container_fasta_path = parts[1].strip()
                                seq_count = parts[2].strip()
                                
                                # CRITICAL: Convert container path to server path for MCP services
                                if container_fasta_path.startswith("/tmp/sessions/"):
                                    server_fasta_path = container_fasta_path.replace("/tmp/sessions/", "/data/sessions/", 1)
                                else:
                                    server_fasta_path = container_fasta_path
                                
                                print(f"  [ParamPreprocess] Conversion successful: {server_fasta_path} ({seq_count} sequences)")
                                return server_fasta_path
                elif "__CSV_NOT_FOUND__:" in stdout:
                    print(f"  [ParamPreprocess] CSV file not found")
                elif "__CSV_NO_SEQ_COLUMNS__" in stdout:
                    print(f"  [ParamPreprocess] No sequence columns found in CSV")
                elif "__CSV_NO_VALID_SEQS__" in stdout:
                    print(f"  [ParamPreprocess] No valid sequences found in CSV columns")
                elif "__CSV_TO_FASTA_ERROR__:" in stdout:
                    print(f"  [ParamPreprocess] Conversion error in output")
                else:
                    print(f"  [ParamPreprocess] No expected markers in output")
            else:
                print(f"  [ParamPreprocess] Conversion failed: {result.error}")
        else:
            print(f"  [ParamPreprocess] OpenSandbox not enabled")
    except Exception as e:
        print(f"  [ParamPreprocess] Conversion execution error: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def _prepare_nettcr_csv_with_peptide(
    csv_path: str,
    state: "ExecutorState",
    task_id: str
) -> Optional[str]:
    """
    Prepare CSV file for NetTCR-2.2 by adding peptide column if missing.
    
    NetTCR-2.2 requires CSV with columns: CDR3a, CDR3b, peptide (legacy format)
    or native format: peptide, A1-A3, B1-B3.
    
    If the CSV doesn't have a peptide column, this function:
    1. Reads the CSV
    2. Checks if peptide column exists
    3. If missing, adds peptide column with the target peptide value
    4. Writes the prepared CSV to sandbox output directory
    
    Args:
        csv_path: Path to CSV file
        state: Executor state (used to get target peptide and sandbox directory)
        task_id: Parent task ID
        
    Returns:
        Path to prepared CSV (or original path if no preparation needed), None on error
    """
    from pathlib import Path
    import re
    
    csv_path_normalized = csv_path.replace("\\", "/")
    csv_name = Path(csv_path_normalized).stem
    
    # Get target peptide from extracted_parameters
    target_peptide = None
    if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
        extracted = state.parent_state.extracted_parameters or {}
        # Try various parameter names
        target_peptide = (
            extracted.get("target_peptide") or
            extracted.get("peptide") or
            extracted.get("epitope") or
            extracted.get("antigen_peptide")
        )
        
        # Also check user_parameters
        user_params = extracted.get("user_parameters", {})
        if not target_peptide and user_params:
            target_peptide = (
                user_params.get("peptide") or
                user_params.get("target_peptide") or
                user_params.get("epitope")
            )
    
    if not target_peptide:
        print(f"  [NetTCR Prep] No target peptide found in extracted_parameters, skipping preparation")
        return csv_path_normalized  # Return original path, let NetTCR handle it
    
    print(f"  [NetTCR Prep] Target peptide: {target_peptide}")
    
    # Determine output path
    sandbox_data_dir = None
    if state.parent_state and hasattr(state.parent_state, 'sandbox_data_dir'):
        sandbox_data_dir = getattr(state.parent_state, 'sandbox_data_dir', None)
    
    if sandbox_data_dir:
        sandbox_data_dir = sandbox_data_dir.replace("\\", "/")
        if sandbox_data_dir.startswith("/data/sessions/"):
            container_data_dir = sandbox_data_dir.replace("/data/sessions/", "/tmp/sessions/", 1)
        else:
            container_data_dir = sandbox_data_dir
        output_csv_path = f"{container_data_dir}/output/{csv_name}_nettcr_input.csv"
        server_csv_path = f"{sandbox_data_dir}/output/{csv_name}_nettcr_input.csv"
    else:
        output_csv_path = f"/tmp/{csv_name}_nettcr_input.csv"
        server_csv_path = output_csv_path
    
    print(f"  [NetTCR Prep] Output path: {output_csv_path}")
    
    # Code to prepare CSV in sandbox
    preparation_code = f'''
import csv
import os

csv_path = "{csv_path_normalized}"
output_path = "{output_csv_path}"
target_peptide = "{target_peptide}"

print(f"[NetTCR Prep] Reading CSV: {{csv_path}}")

# Check if file exists
if not os.path.exists(csv_path):
    print(f"__NETTCR_CSV_NOT_FOUND__:{{csv_path}}")
    exit(1)

# Read CSV
with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    rows = list(reader)

print(f"[NetTCR Prep] Columns: {{columns}}")
print(f"[NetTCR Prep] Rows: {{len(rows)}}")

# Check if peptide column already exists
peptide_col = None
for col in columns:
    if col.lower() == "peptide":
        peptide_col = col
        break

if peptide_col:
    # Check if all values are the target peptide
    all_match = all(row.get(peptide_col, "").strip().upper() == target_peptide.upper() for row in rows if row.get(peptide_col))
    if all_match:
        print(f"__NETTCR_NO_PREP_NEEDED__:{{csv_path}}")
        exit(0)
    else:
        print(f"[NetTCR Prep] Peptide column exists but values differ, will overwrite")

# Add peptide column
col_map = {{c.lower(): c for c in columns if c}}

# Auto-detect CDR3a and CDR3b columns
cdr3a_patterns = ["cdr3a", "cdr3_a", "cdr3alpha", "alpha_cdr3", "tra_cdr3"]
cdr3b_patterns = ["cdr3b", "cdr3_b", "cdr3beta", "beta_cdr3", "trb_cdr3"]

cdr3a_col = None
cdr3b_col = None

for pattern in cdr3a_patterns:
    if pattern in col_map:
        cdr3a_col = col_map[pattern]
        break

for pattern in cdr3b_patterns:
    if pattern in col_map:
        cdr3b_col = col_map[pattern]
        break

print(f"[NetTCR Prep] Detected CDR3a: {{cdr3a_col}}")
print(f"[NetTCR Prep] Detected CDR3b: {{cdr3b_col}}")

if not cdr3a_col or not cdr3b_col:
    print(f"[NetTCR Prep] Warning: CDR3a or CDR3b column not detected")

# Create output rows with peptide column
output_rows = []
for row in rows:
    new_row = dict(row)
    new_row["peptide"] = target_peptide
    output_rows.append(new_row)

# Build fieldnames
if "peptide" in col_map:
    fieldnames = list(columns)
else:
    fieldnames = list(columns) + ["peptide"]

# Create output directory if needed
os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

# Write output CSV
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(output_rows)

print(f"[NetTCR Prep] Prepared CSV with peptide column: {{output_path}}")
print(f"[NetTCR Prep] Total rows: {{len(output_rows)}}")
print(f"__NETTCR_CSV_PREPARED__:{{output_path}}:{{len(output_rows)}}")
'''
    
    # 使用 codeact_executor 统一接口 (遵循架构原则)
    try:
        from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
        
        if is_codeact_available():
            # Try to reuse existing sandbox from session
            existing_sandbox_id = None
            if state.parent_state:
                merged_result = getattr(state.parent_state, 'merged_result', None) or {}
                existing_sandbox_id = merged_result.get("opensandbox_id")
                if existing_sandbox_id:
                    print(f"  [NetTCR Prep] Reusing session sandbox: {existing_sandbox_id}")
            
            result = execute_code_via_codeact(
                task_description=f"准备 NetTCR 输入 CSV: {csv_path_normalized}",
                code_template=preparation_code,
                sandbox_id=existing_sandbox_id,
                timeout_seconds=60,
                keep_alive=True
            )
            
            if result.is_success():
                stdout = result.output
                print(f"  [NetTCR Prep] Output: {stdout[:500]}..." if len(stdout) > 500 else f"  [NetTCR Prep] Output: {stdout}")
                
                if "__NETTCR_CSV_PREPARED__:" in stdout:
                    print(f"  [NetTCR Prep] CSV prepared successfully: {server_csv_path}")
                    return server_csv_path
                elif "__NETTCR_NO_PREP_NEEDED__:" in stdout:
                    print(f"  [NetTCR Prep] No preparation needed, using original CSV")
                    return csv_path_normalized
                elif "__NETTCR_CSV_NOT_FOUND__:" in stdout:
                    print(f"  [NetTCR Prep] CSV file not found")
            else:
                print(f"  [NetTCR Prep] Execution failed: {result.error}")
        else:
            print(f"  [NetTCR Prep] CodeAct not available")
    except Exception as e:
        print(f"  [NetTCR Prep] Preparation error: {e}")
        import traceback
        traceback.print_exc()
    
    return None


# IMGT V gene CDR reference data (embedded for sandbox execution)
# This is a subset of the full reference, containing the most common V genes
_IMGT_CDR_REFERENCE_EMBEDDED = '''
{
  "trav": {
    "TRAV1-1": {"cdr1": "SSNNYN", "cdr2": "FYFSTLT"},
    "TRAV1-2": {"cdr1": "SSNYYN", "cdr2": "FYFSTLT"},
    "TRAV2": {"cdr1": "QSVSSN", "cdr2": "VQDSQQY"},
    "TRAV3": {"cdr1": "NPDSSN", "cdr2": "SYDQQY"},
    "TRAV4": {"cdr1": "SNYSNY", "cdr2": "FTLTGNT"},
    "TRAV5": {"cdr1": "SGFNYN", "cdr2": "VNSEQQY"},
    "TRAV6": {"cdr1": "TSNYNY", "cdr2": "VQDSQQY"},
    "TRAV7": {"cdr1": "SSGFGY", "cdr2": "YINQQY"},
    "TRAV8-1": {"cdr1": "SSNYYN", "cdr2": "VNSEQQY"},
    "TRAV8-2": {"cdr1": "SSNYYN", "cdr2": "VNSEQQY"},
    "TRAV8-3": {"cdr1": "SSNYYN", "cdr2": "VNSEQQY"},
    "TRAV8-4": {"cdr1": "SSNYYN", "cdr2": "VNSEQQY"},
    "TRAV8-5": {"cdr1": "SSNYYN", "cdr2": "VNSEQQY"},
    "TRAV8-6": {"cdr1": "SSNYYN", "cdr2": "VNSEQQY"},
    "TRAV9-1": {"cdr1": "SSNYYN", "cdr2": "IQNQQY"},
    "TRAV9-2": {"cdr1": "SSNYYN", "cdr2": "IQNQQY"},
    "TRAV10": {"cdr1": "SSFGGY", "cdr2": "ITGQQY"},
    "TRAV12-1": {"cdr1": "SNYNYY", "cdr2": "VTNSQQY"},
    "TRAV12-2": {"cdr1": "SNYNYY", "cdr2": "VTNSQQY"},
    "TRAV12-3": {"cdr1": "SNYNYY", "cdr2": "VTNSQQY"},
    "TRAV13-1": {"cdr1": "TSNYNY", "cdr2": "VTNQQY"},
    "TRAV13-2": {"cdr1": "TSNYNY", "cdr2": "VTNQQY"},
    "TRAV14": {"cdr1": "DNYNYN", "cdr2": "ITDQQY"},
    "TRAV14DV4": {"cdr1": "DNYNYN", "cdr2": "ITDQQY"},
    "TRAV16": {"cdr1": "TSGFGY", "cdr2": "IQNSQQY"},
    "TRAV17": {"cdr1": "DNYNYN", "cdr2": "ITDQQY"},
    "TRAV18": {"cdr1": "NDFSSN", "cdr2": "SYDQQY"},
    "TRAV19": {"cdr1": "NDSNYY", "cdr2": "SYDQQY"},
    "TRAV20": {"cdr1": "SDFGGY", "cdr2": "ITGQQY"},
    "TRAV21": {"cdr1": "DNNYNY", "cdr2": "ITNQQY"},
    "TRAV22": {"cdr1": "TNNYYY", "cdr2": "ITNQQY"},
    "TRAV23": {"cdr1": "NDFSSN", "cdr2": "SYDQQY"},
    "TRAV23DV6": {"cdr1": "NDFSSN", "cdr2": "SYDQQY"},
    "TRAV24": {"cdr1": "DTSNYN", "cdr2": "ITNQQY"},
    "TRAV25": {"cdr1": "SGSSYY", "cdr2": "ITQQQY"},
    "TRAV26-1": {"cdr1": "NSGFNY", "cdr2": "ITQQQY"},
    "TRAV26-2": {"cdr1": "NSGFNY", "cdr2": "ITQQQY"},
    "TRAV27": {"cdr1": "SSNSYY", "cdr2": "ITQQQY"},
    "TRAV29": {"cdr1": "NTDNYN", "cdr2": "ITNQQY"},
    "TRAV29DV5": {"cdr1": "NTDNYN", "cdr2": "ITNQQY"},
    "TRAV35": {"cdr1": "TSGFGY", "cdr2": "IQNSQQY"},
    "TRAV36": {"cdr1": "TSGFGY", "cdr2": "IQNSQQY"},
    "TRAV36DV7": {"cdr1": "TSGFGY", "cdr2": "IQNSQQY"},
    "TRAV38-1": {"cdr1": "NSGFNY", "cdr2": "ITQQQY"},
    "TRAV38-2": {"cdr1": "NSGFNY", "cdr2": "ITQQQY"},
    "TRAV39": {"cdr1": "SSNSYY", "cdr2": "ITQQQY"},
    "TRAV41": {"cdr1": "NNYNYN", "cdr2": "VTNQQY"}
  },
  "trbv": {
    "TRBV1": {"cdr1": "NDMSSN", "cdr2": "SYDQQY"},
    "TRBV2": {"cdr1": "SNHYYN", "cdr2": "FQNSQQY"},
    "TRBV3-1": {"cdr1": "SGHYYN", "cdr2": "FQNEQQY"},
    "TRBV3-2": {"cdr1": "SGHYYN", "cdr2": "FQNEQQY"},
    "TRBV4-1": {"cdr1": "NQNYYN", "cdr2": "SYDQQY"},
    "TRBV4-2": {"cdr1": "NQNYYN", "cdr2": "SYDQQY"},
    "TRBV4-3": {"cdr1": "NQNYYN", "cdr2": "SYDQQY"},
    "TRBV5-1": {"cdr1": "SSEYYN", "cdr2": "IQNQQY"},
    "TRBV5-2": {"cdr1": "SSEYYN", "cdr2": "IQNQQY"},
    "TRBV5-3": {"cdr1": "SSEYYN", "cdr2": "IQNQQY"},
    "TRBV5-4": {"cdr1": "SSEYYN", "cdr2": "IQNQQY"},
    "TRBV5-5": {"cdr1": "SSEYYN", "cdr2": "IQNQQY"},
    "TRBV5-6": {"cdr1": "SSEYYN", "cdr2": "IQNQQY"},
    "TRBV5-7": {"cdr1": "SSEYYN", "cdr2": "IQNQQY"},
    "TRBV5-8": {"cdr1": "SSEYYN", "cdr2": "IQNQQY"},
    "TRBV6-1": {"cdr1": "NSNNYN", "cdr2": "STGQQY"},
    "TRBV6-2": {"cdr1": "NSNNYN", "cdr2": "STGQQY"},
    "TRBV6-3": {"cdr1": "NSNNYN", "cdr2": "STGQQY"},
    "TRBV6-4": {"cdr1": "NSNNYN", "cdr2": "STGQQY"},
    "TRBV6-5": {"cdr1": "NSNNYN", "cdr2": "STGQQY"},
    "TRBV6-6": {"cdr1": "NSNNYN", "cdr2": "STGQQY"},
    "TRBV7-1": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV7-2": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV7-3": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV7-4": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV7-5": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV7-6": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV7-7": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV7-8": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV7-9": {"cdr1": "SQNYYN", "cdr2": "IQNSQQY"},
    "TRBV9": {"cdr1": "SSEYYN", "cdr2": "IQNSQQY"},
    "TRBV10-1": {"cdr1": "SGHYYN", "cdr2": "FQNEQQY"},
    "TRBV10-2": {"cdr1": "SGHYYN", "cdr2": "FQNEQQY"},
    "TRBV10-3": {"cdr1": "SGHYYN", "cdr2": "FQNEQQY"},
    "TRBV11-1": {"cdr1": "SSNYYN", "cdr2": "SYDQQY"},
    "TRBV11-2": {"cdr1": "SSNYYN", "cdr2": "SYDQQY"},
    "TRBV11-3": {"cdr1": "SSNYYN", "cdr2": "SYDQQY"},
    "TRBV12-1": {"cdr1": "NTGFYN", "cdr2": "STGQQY"},
    "TRBV12-2": {"cdr1": "NTGFYN", "cdr2": "STGQQY"},
    "TRBV12-3": {"cdr1": "NTGFYN", "cdr2": "STGQQY"},
    "TRBV12-4": {"cdr1": "NTGFYN", "cdr2": "STGQQY"},
    "TRBV12-5": {"cdr1": "NTGFYN", "cdr2": "STGQQY"},
    "TRBV13": {"cdr1": "SGFNYN", "cdr2": "VTNSQQY"},
    "TRBV14": {"cdr1": "NDSNYY", "cdr2": "SYDQQY"},
    "TRBV15": {"cdr1": "TSNYYN", "cdr2": "VTNQQY"},
    "TRBV16": {"cdr1": "NTGNYN", "cdr2": "STGQQY"},
    "TRBV17": {"cdr1": "NDSNYY", "cdr2": "SYDQQY"},
    "TRBV18": {"cdr1": "NSGFNY", "cdr2": "ITQQQY"},
    "TRBV19": {"cdr1": "SSNYYN", "cdr2": "SYDQQY"},
    "TRBV20-1": {"cdr1": "SNYNYN", "cdr2": "ITDQQY"},
    "TRBV20-2": {"cdr1": "SNYNYN", "cdr2": "ITDQQY"},
    "TRBV21-1": {"cdr1": "TSGFGY", "cdr2": "IQNSQQY"},
    "TRBV21-2": {"cdr1": "TSGFGY", "cdr2": "IQNSQQY"},
    "TRBV22-1": {"cdr1": "NTDNYN", "cdr2": "ITNQQY"},
    "TRBV22-2": {"cdr1": "NTDNYN", "cdr2": "ITNQQY"},
    "TRBV22-3": {"cdr1": "NTDNYN", "cdr2": "ITNQQY"},
    "TRBV23-1": {"cdr1": "NSNNYN", "cdr2": "STGQQY"},
    "TRBV24-1": {"cdr1": "STNSQN", "cdr2": "ITGQQY"},
    "TRBV25-1": {"cdr1": "NTDNYN", "cdr2": "ITNQQY"},
    "TRBV26": {"cdr1": "SSNYYN", "cdr2": "SYDQQY"},
    "TRBV27": {"cdr1": "SNYNYN", "cdr2": "ITDQQY"},
    "TRBV28": {"cdr1": "SSNYYN", "cdr2": "SYDQQY"},
    "TRBV29-1": {"cdr1": "NTGFYN", "cdr2": "STGQQY"},
    "TRBV29-2": {"cdr1": "NTGFYN", "cdr2": "STGQQY"},
    "TRBV30": {"cdr1": "NTSYYN", "cdr2": "IQNSQQY"}
  }
}
'''


def _convert_tcr_to_nettcr_format(
    csv_path: str,
    state: "ExecutorState",
    task_id: str,
    peptide: Optional[str] = None
) -> Optional[str]:
    """
    Convert TCR CSV file to NetTCR format directly in sandbox using Python built-in modules.
    
    Similar to CSV=>FASTA conversion, this runs pure Python code in the sandbox to:
    1. Read input CSV with TCR data (TRA_v_gene, TRB_v_gene, CDR3a, CDR3b columns)
    2. Infer CDR1/CDR2 from V gene names using embedded IMGT reference
    3. Output NetTCR format CSV (A1, A2, A3, B1, B2, B3, peptide columns)
    
    Args:
        csv_path: Path to input CSV file
        state: Executor state
        task_id: Parent task ID
        peptide: Target peptide sequence (optional, will try to detect from data)
        
    Returns:
        Path to generated NetTCR format CSV, or None if conversion failed
    """
    from pathlib import Path
    import re
    
    csv_path_normalized = csv_path.replace("\\", "/")
    csv_name = Path(csv_path_normalized).stem
    
    # Get output directory
    sandbox_data_dir = None
    if state.parent_state and hasattr(state.parent_state, 'sandbox_data_dir'):
        sandbox_data_dir = getattr(state.parent_state, 'sandbox_data_dir', None)
    
    if sandbox_data_dir:
        sandbox_data_dir = sandbox_data_dir.replace("\\", "/")
        if sandbox_data_dir.startswith("/data/sessions/"):
            container_data_dir = sandbox_data_dir.replace("/data/sessions/", "/tmp/sessions/", 1)
        else:
            container_data_dir = sandbox_data_dir
        output_path = f"{container_data_dir}/output/{csv_name}_nettcr_format.csv"
    else:
        # Fallback
        output_path = f"/tmp/{csv_name}_nettcr_format.csv"
    
    # Default peptide for common benchmarks (can be overridden)
    default_peptide = peptide or "ELAGIGILTV"  # MART-1 epitope, common in benchmarks
    
    print(f"  [ParamPreprocess] Converting TCR to NetTCR format directly in sandbox")
    print(f"  [ParamPreprocess] Input CSV: {csv_path_normalized}")
    print(f"  [ParamPreprocess] Output path: {output_path}")
    
    # Embedded IMGT reference for CDR inference
    imgt_reference = _IMGT_CDR_REFERENCE_EMBEDDED.strip()
    
    conversion_code = f'''
import csv
import json
import os
import re

csv_path = "{csv_path_normalized}"
output_path = "{output_path}"
default_peptide = "{default_peptide}"

# Embedded IMGT CDR reference
imgt_reference_json = """{imgt_reference}"""

try:
    imgt_ref = json.loads(imgt_reference_json)
    trav_ref = imgt_ref.get("trav", {{}})
    trbv_ref = imgt_ref.get("trbv", {{}})
except Exception as e:
    print(f"__TCR_TO_NETTCR_ERROR__:Failed to parse IMGT reference: {{e}}")
    exit(1)

def normalize_vgene(vgene):
    """Normalize V gene name to standard format."""
    if not vgene:
        return ""
    vgene = str(vgene).upper().strip()
    # Remove allele suffix
    if "*" in vgene:
        vgene = vgene.split("*")[0]
    # Replace underscores
    vgene = vgene.replace("_", "-")
    # Handle DV dual genes
    dv_map = {{
        "TRAV14DV4": "TRAV14", "TRAV23DV6": "TRAV23",
        "TRAV29DV5": "TRAV29", "TRAV36DV7": "TRAV36",
        "TRAV38-2DV8": "TRAV38-2"
    }}
    if vgene in dv_map:
        vgene = dv_map[vgene]
    # Remove D suffix
    if vgene.endswith("D") and len(vgene) > 4 and vgene[-2].isdigit():
        vgene = vgene[:-1]
    return vgene

def get_cdr12(vgene, chain_type):
    """Get CDR1 and CDR2 from V gene name."""
    if not vgene:
        return "", ""
    normalized = normalize_vgene(vgene)
    ref = trav_ref if chain_type == "trav" else trbv_ref
    if normalized in ref:
        return ref[normalized].get("cdr1", ""), ref[normalized].get("cdr2", "")
    # Fuzzy match: try base gene
    if "-" in normalized:
        base = normalized.rsplit("-", 1)[0]
        for gene, data in ref.items():
            if gene.startswith(base):
                return data.get("cdr1", ""), data.get("cdr2", "")
    return "", ""

# Column detection patterns
trav_patterns = ["tra_v_gene", "trav", "v_a_gene", "alpha_v", "tra_vgene", "v_alpha"]
trbv_patterns = ["trb_v_gene", "trbv", "v_b_gene", "beta_v", "trb_vgene", "v_beta"]
cdr3a_patterns = ["cdr3a", "cdr3_a", "cdr3alpha", "alpha_cdr3", "a_cdr3", "tra_cdr3"]
cdr3b_patterns = ["cdr3b", "cdr3_b", "cdr3beta", "beta_cdr3", "b_cdr3", "trb_cdr3"]
peptide_patterns = ["peptide", "epitope", "antigen", "target_peptide"]
id_patterns = ["main_name", "id", "name", "sample_id", "cell_id"]

def find_column(columns, patterns):
    """Find column matching any pattern."""
    col_map = {{c.lower(): c for c in columns if c and c.strip()}}
    for pattern in patterns:
        for col_lower, col_orig in col_map.items():
            if pattern == col_lower or pattern in col_lower:
                return col_orig
    return None

try:
    if not os.path.exists(csv_path):
        print(f"__TCR_TO_NETTCR_ERROR__:CSV not found: {{csv_path}}")
    else:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames or []
        
        print(f"Read CSV: {{len(rows)}} rows, columns: {{columns}}")
        
        # Find columns
        trav_col = find_column(columns, trav_patterns)
        trbv_col = find_column(columns, trbv_patterns)
        cdr3a_col = find_column(columns, cdr3a_patterns)
        cdr3b_col = find_column(columns, cdr3b_patterns)
        peptide_col = find_column(columns, peptide_patterns)
        id_col = find_column(columns, id_patterns)
        
        print(f"Detected columns: TRAV={{trav_col}}, TRBV={{trbv_col}}, CDR3a={{cdr3a_col}}, CDR3b={{cdr3b_col}}, peptide={{peptide_col}}")
        
        if not trav_col and not trbv_col:
            print("__TCR_TO_NETTCR_ERROR__:No V gene columns found")
        elif not cdr3a_col and not cdr3b_col:
            print("__TCR_TO_NETTCR_ERROR__:No CDR3 columns found")
        else:
            # Process rows
            stats = {{"success": 0, "partial": 0, "failed": 0}}
            nettcr_rows = []
            
            for row in rows:
                trav = row.get(trav_col, "") if trav_col else ""
                trbv = row.get(trbv_col, "") if trbv_col else ""
                c3a = row.get(cdr3a_col, "") if cdr3a_col else ""
                c3b = row.get(cdr3b_col, "") if cdr3b_col else ""
                pep = row.get(peptide_col, "") if peptide_col else default_peptide
                
                # Infer CDR1/CDR2
                a1, a2 = get_cdr12(trav, "trav") if trav else ("", "")
                b1, b2 = get_cdr12(trbv, "trbv") if trbv else ("", "")
                
                if a1 and a2 and b1 and b2:
                    stats["success"] += 1
                elif a1 or a2 or b1 or b2:
                    stats["partial"] += 1
                else:
                    stats["failed"] += 1
                
                # Build output row (keep original columns + add NetTCR columns)
                out_row = dict(row)
                out_row["A1"] = a1
                out_row["A2"] = a2
                out_row["A3"] = c3a
                out_row["B1"] = b1
                out_row["B2"] = b2
                out_row["B3"] = c3b
                out_row["peptide"] = pep or default_peptide
                nettcr_rows.append(out_row)
            
            # Write output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nettcr_columns = ["A1", "A2", "A3", "B1", "B2", "B3", "peptide"]
            fieldnames = [c for c in columns if c not in nettcr_columns] + nettcr_columns
            
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(nettcr_rows)
            
            print(f"Conversion stats: success={{stats['success']}}, partial={{stats['partial']}}, failed={{stats['failed']}}")
            print(f"__TCR_TO_NETTCR_SUCCESS__:{{output_path}}:{{len(nettcr_rows)}}")

except Exception as e:
    import traceback
    print(f"__TCR_TO_NETTCR_ERROR__:{{str(e)}}")
    traceback.print_exc()
'''
    
    # 使用 codeact_executor 统一接口 (遵循架构原则)
    try:
        from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
        
        if is_codeact_available():
            # Try to reuse existing sandbox from session
            existing_sandbox_id = None
            if state.parent_state:
                merged_result = getattr(state.parent_state, 'merged_result', None) or {}
                existing_sandbox_id = merged_result.get('opensandbox_id')
                if existing_sandbox_id:
                    print(f"  [ParamPreprocess] Reusing session sandbox for TCR conversion: {existing_sandbox_id}")
            
            result = execute_code_via_codeact(
                task_description=f"将 TCR CSV 转换为 NetTCR 格式: {csv_path}",
                code_template=conversion_code,
                sandbox_id=existing_sandbox_id,
                timeout_seconds=60,
                keep_alive=True
            )
            
            if result.is_success():
                stdout = result.output
                print(f"  [ParamPreprocess] TCR conversion output:")
                print(f"    stdout: {stdout[:1000]}..." if len(stdout) > 1000 else f"    stdout: {stdout}")
                
                if "__TCR_TO_NETTCR_SUCCESS__:" in stdout:
                    for line in stdout.split("\n"):
                        if "__TCR_TO_NETTCR_SUCCESS__:" in line:
                            parts = line.split(":")
                            if len(parts) >= 3:
                                container_output_path = parts[1].strip()
                                row_count = parts[2].strip()
                                
                                # Convert container path to server path
                                if container_output_path.startswith("/tmp/sessions/"):
                                    server_output_path = container_output_path.replace("/tmp/sessions/", "/data/sessions/", 1)
                                else:
                                    server_output_path = container_output_path
                                
                                print(f"  [ParamPreprocess] TCR conversion successful: {server_output_path} ({row_count} rows)")
                                return server_output_path
                elif "__TCR_TO_NETTCR_ERROR__:" in stdout:
                    print(f"  [ParamPreprocess] TCR conversion error in output")
            else:
                print(f"  [ParamPreprocess] Execution failed: {result.error}")
        else:
            print(f"  [ParamPreprocess] CodeAct not available")
    except Exception as e:
        print(f"  [ParamPreprocess] TCR conversion execution error: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def _extract_parameters_from_context(
    user_input: str,
    execution_plan: Optional[str],
    task_description: str,
    tool_name: str,
    tool_params: List[Dict[str, Any]],
    llm: Optional[Any]
) -> Dict[str, Any]:
    """Use LLM to extract parameter values from context (semantic matching)."""
    if not tool_params:
        return {}

    explicit_params = _extract_params_from_explicit_lines(
        user_input=user_input,
        execution_plan=execution_plan,
        tool_name=tool_name,
        tool_params=tool_params
    )
    if explicit_params and not llm:
        return explicit_params
    if not llm:
        return {}

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        param_info_list = []
        for param in tool_params:
            param_name = param.get("name", "")
            param_type = param.get("type", "")
            param_desc = param.get("description", "")
            if param_name:
                param_info = f"- {param_name}"
                if param_type:
                    param_info += f" (type: {param_type})"
                if param_desc:
                    param_info += f": {param_desc}"
                param_info_list.append(param_info)

        context_parts = []
        if user_input:
            context_parts.append(f"User Input: {user_input}")
        if execution_plan:
            context_parts.append(f"Execution Plan: {execution_plan}")
        if task_description:
            context_parts.append(f"Task Description: {task_description}")

        context_text = "\n\n".join(context_parts)

        extraction_prompt = f"""Please analyze the following context information and extract parameter values for tool "{tool_name}".

Context Information:
{context_text}

Tool Parameters:
{chr(10).join(param_info_list) if param_info_list else 'No parameter information available'}

Please extract parameter values based on semantic understanding (not keyword matching). Focus on intent and meaning.
Return JSON format with parameter names as keys and extracted values as values.
Only include parameters that can be clearly extracted from context."""

        messages = [
            SystemMessage(content="You are a professional parameter extraction expert. Use semantic understanding to map user intent to tool parameters."),
            HumanMessage(content=extraction_prompt)
        ]

        response = llm.invoke(messages)
        response_text = response.content.strip()

        extracted_params = {}
        try:
            extracted_params = json.loads(response_text.strip())
        except json.JSONDecodeError:
            import re
            json_block_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
            ]
            for pattern in json_block_patterns:
                matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        extracted_params = json.loads(match)
                        break
                    except json.JSONDecodeError:
                        continue
                if extracted_params:
                    break

            if not extracted_params:
                brace_count = 0
                start_idx = -1
                for i, char in enumerate(response_text):
                    if char == '{':
                        if brace_count == 0:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_idx != -1:
                            try:
                                json_str = response_text[start_idx:i+1]
                                extracted_params = json.loads(json_str)
                                break
                            except json.JSONDecodeError:
                                pass
                            start_idx = -1

        if extracted_params and isinstance(extracted_params, dict):
            normalized = {}
            for key, value in extracted_params.items():
                normalized[key] = _normalize_base_dir_value(key, value)
            if explicit_params:
                for key, value in explicit_params.items():
                    normalized[key] = value
            return normalized
    except Exception as e:
        print(f"  ⚠ Failed to extract parameters from context: {e}")
        if os.getenv("DEBUG_LLM_ERRORS", "false").lower() == "true":
            import traceback
            traceback.print_exc()

    return explicit_params or {}


def _extract_params_from_explicit_lines(
    user_input: str,
    execution_plan: Optional[str],
    tool_name: str,
    tool_params: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract parameters from explicit 'param: value' lines in context."""
    if not tool_params:
        return {}
    tool_lower = (tool_name or "").lower()
    context = "\n".join([part for part in [user_input, execution_plan] if part])
    if not context:
        return {}

    candidates: Dict[str, Any] = {}
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        label, raw_value = line.split(":", 1)
        line_lower = line.lower()
        label_lower = label.strip().lower()
        for param in tool_params:
            param_name = param.get("name", "")
            param_type = param.get("type", "")
            param_desc = param.get("description", "")
            if not param_name:
                continue
            variants = {
                param_name.lower(),
                param_name.lower().replace("_", " "),
                param_name.lower().replace("_", "-")
            }
            if param_type:
                variants.add(param_type.lower())
            if param_desc:
                desc_lower = param_desc.lower()
                if "fasta file" in desc_lower:
                    variants.add("fasta file")
                if "csv file" in desc_lower:
                    variants.add("csv file")
                if "rds file" in desc_lower:
                    variants.add("rds file")

            matched = any(v in line_lower for v in variants) or any(
                v in label_lower or label_lower in v for v in variants
            )
            if not matched:
                continue
            if _is_output_param(param_name, param_type) and not _label_indicates_output(label_lower):
                # Avoid treating user-provided input files as output paths
                continue
            if tool_lower and ("for " + tool_lower) in line_lower:
                pass
            elif tool_lower and tool_lower in line_lower:
                pass
            else:
                # If tool name not mentioned, still accept explicit param lines
                pass
            value = raw_value.strip().strip('"').strip("'")
            if not value:
                continue
            normalized = _normalize_inferred_param_value(param_name, param_type, value)
            if normalized is not None:
                candidates[param_name] = {
                    "__value": _normalize_base_dir_value(param_name, normalized),
                    "__explicit": True,
                    "__label": label_lower
                }

    return candidates


def _summarize_output_brief(output: Any, max_len: int = 200) -> str:
    """Create a brief summary of task output for logging/summary"""
    try:
        if isinstance(output, dict):
            keys = list(output.keys())
            return f"dict(keys={keys[:10]})"
        if isinstance(output, list):
            return f"list(len={len(output)})"
        if isinstance(output, str):
            return output[:max_len]
        return str(output)[:max_len]
    except Exception:
        return "<unavailable>"


def _extract_dependency_value(dep_output: Any, param_name: str, is_file_param: bool) -> Optional[Any]:
    """Try to extract a parameter value from dependency output"""
    if dep_output is None:
        return None
    param_lower = (param_name or "").lower()
    file_key_candidates = {
        "output_file", "output_path", "result_path", "result_file",
        "file", "file_path", "filepath", "path",
        "csv_file", "tsv_file", "json_file", "airr_results",
        "input_file", "input_path", "output", "outputs", "result", "results"
    }
    if isinstance(dep_output, dict):
        # Prefer final_result first (most reliable output payload)
        if "final_result" in dep_output and isinstance(dep_output["final_result"], dict):
            value = _extract_dependency_value(dep_output["final_result"], param_name, is_file_param)
            if value is not None:
                return value
        # Direct key match
        if param_name in dep_output:
            value = dep_output[param_name]
            if is_file_param and isinstance(value, str) and not _is_path_like(value):
                return None
            return value
        # Case-insensitive match
        for key, value in dep_output.items():
            if isinstance(key, str) and (param_lower == key.lower() or param_lower in key.lower()):
                if is_file_param and isinstance(value, str) and not _is_path_like(value):
                    continue
                return value
        # File-like fallback
        if is_file_param:
            # Prefer known output keys first
            for key, value in dep_output.items():
                if isinstance(key, str) and key.lower() in file_key_candidates:
                    if isinstance(value, str) and _is_path_like(value):
                        return value
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and _is_path_like(item):
                                return item
                    if isinstance(value, dict):
                        nested = _extract_dependency_value(value, param_name, is_file_param)
                        if nested is not None:
                            return nested
            # Inspect message lists for result payloads only
            messages = dep_output.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    if message.get("type") != "result":
                        continue
                    raw = message.get("raw")
                    if isinstance(raw, dict):
                        for key in file_key_candidates:
                            candidate = raw.get(key)
                            if isinstance(candidate, str) and _is_path_like(candidate):
                                return candidate
                            if isinstance(candidate, list):
                                for item in candidate:
                                    if isinstance(item, str) and _is_path_like(item):
                                        return item
                        nested = _extract_dependency_value(raw, param_name, is_file_param)
                        if nested is not None:
                            return nested
            # Final fallback: avoid status-like strings
            for key, value in dep_output.items():
                if key in {"messages", "content"}:
                    continue
                if isinstance(value, str) and _is_path_like(value):
                    return value
                if isinstance(value, dict) or isinstance(value, list):
                    nested = _extract_dependency_value(value, param_name, is_file_param)
                    if nested is not None:
                        return nested
        return None
    if isinstance(dep_output, list):
        # Prefer list of dicts
        for item in dep_output:
            if isinstance(item, dict):
                value = _extract_dependency_value(item, param_name, is_file_param)
                if value is not None:
                    return value
        # Fallback to first string for file params
        if is_file_param:
            for item in dep_output:
                if isinstance(item, str):
                    return item
        return None
    # String output
    if isinstance(dep_output, str):
        if is_file_param:
            return dep_output
    return dep_output


def _resolve_param_from_dependencies(
    task: SubTask,
    tool_name: str,
    param_name: str,
    param_type: Optional[str],
    param_desc: Optional[str],
    state: "ExecutorState",
    llm: Optional[Any],
) -> Optional[Any]:
    """Resolve parameter value from completed dependency task outputs.
    
    This function searches for parameter values from:
    1. Parameter table (extracted_parameters["files"]) - contains structured file metadata
    2. Task outputs (extracted_parameters["task_outputs"]) - contains output file lists
    3. Raw dependency output (dep_result.output) - contains raw execution results
    """
    if not task.dependencies:
        return None
    is_file_param = _is_file_type(param_name, param_type)
    is_directory_param = _is_directory_param(param_name, param_type)
    expected_exts = _expected_file_extensions(param_name, param_type, param_desc)
    
    for dep_task_id in task.dependencies:
        dep_result = state.task_results.get(dep_task_id)
        if not dep_result or dep_result.status != ExecutorTaskStatus.COMPLETED:
            continue
        
        # SEMANTIC CHECK: Verify that the dependency output type matches parameter expectation
        # For example: antigen_file should NOT accept AIRR results (which are antibody analysis)
        if is_file_param and not _check_dependency_output_semantic_match(dep_task_id, param_name, state):
            print(f"  [ResolveParam] Skipping dep {dep_task_id} for {param_name}: semantic type mismatch")
            continue
        
        # STEP 1: Try to find from parameter table (files) - most reliable source
        if is_file_param and state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
            extracted = state.parent_state.extracted_parameters or {}
            files_dict = extracted.get("files", {})
            
            # Look for files that match expected extensions and came from this dependency
            for file_key, file_info in files_dict.items():
                if not isinstance(file_info, dict):
                    continue
                
                # Check if this file came from the dependency task
                source_task = file_info.get("source_task", "")
                if source_task != dep_task_id:
                    continue
                
                file_path = file_info.get("sandbox_path", "")
                if not file_path:
                    continue
                
                # Check if file extension matches expected types
                if expected_exts and not _path_matches_expected_types(file_path, expected_exts):
                    continue
                
                # Check semantic compatibility
                can_be_used_as = file_info.get("can_be_used_as", [])
                param_name_lower = param_name.lower()
                
                # Direct parameter name match or semantic type match
                if param_name_lower in [p.lower() for p in can_be_used_as]:
                    print(f"  [ResolveParam] Found {param_name} from parameter table: {file_path}")
                    normalized = _normalize_inferred_param_value(param_name, param_type, file_path)
                    if normalized is not None:
                        return normalized
                
                # Check if file description matches parameter expectation
                file_desc = file_info.get("description", "").lower()
                data_type = file_info.get("data_type", "").lower()
                
                # Special handling for common parameter patterns
                param_lower = param_name.lower()
                if "airr" in param_lower and "airr" in data_type:
                    print(f"  [ResolveParam] Found AIRR results from parameter table: {file_path}")
                    normalized = _normalize_inferred_param_value(param_name, param_type, file_path)
                    if normalized is not None:
                        return normalized
        
        # STEP 2: Try to find from task_outputs - contains structured output info
        if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
            extracted = state.parent_state.extracted_parameters or {}
            task_outputs = extracted.get("task_outputs", {})
            dep_task_output = task_outputs.get(dep_task_id, {})
            
            if isinstance(dep_task_output, dict):
                output_files = dep_task_output.get("output_files", [])
                dep_tool_name = dep_task_output.get("tool_name", "")
                
                for output_file in output_files:
                    if not isinstance(output_file, str):
                        continue
                    
                    # Check if file extension matches
                    if expected_exts and not _path_matches_expected_types(output_file, expected_exts):
                        continue
                    
                    # Special handling: analyze_vdj_batch outputs AIRR format
                    if "airr" in param_name.lower() and dep_tool_name == "analyze_vdj_batch":
                        print(f"  [ResolveParam] Found AIRR output from analyze_vdj_batch: {output_file}")
                        normalized = _normalize_inferred_param_value(param_name, param_type, output_file)
                        if normalized is not None:
                            return normalized
                    
                    # Special handling: tcell service tools need integrated_tcr_data.rds from integrate_tcr_data_complete
                    # This ensures downstream tcell tools can find the fixed output filename
                    if dep_tool_name == "integrate_tcr_data_complete" and param_name == "input_file":
                        if output_file.endswith(".rds"):
                            print(f"  [ResolveParam] Found integrated TCR RDS from integrate_tcr_data_complete: {output_file}")
                            normalized = _normalize_inferred_param_value(param_name, param_type, output_file)
                            if normalized is not None:
                                return normalized
        
        # STEP 3: Fall back to raw dependency output extraction
        # Determine expected file type from expected_exts for filtering
        expected_file_type = None
        if expected_exts:
            if "csv" in expected_exts:
                expected_file_type = "csv"
            elif "rds" in expected_exts:
                expected_file_type = "rds"
            elif "txt" in expected_exts:
                expected_file_type = "txt"
        candidates = _extract_dependency_candidates(dep_result.output, is_file_param, expected_file_type)
        if not candidates:
            continue
        dep_task = next((t for t in state.subtasks if t.task_id == dep_task_id), None)
        dep_task_desc = dep_task.content if dep_task else ""
        dep_output_summary = _summarize_output_brief(dep_result.output, max_len=500)
        for candidate in candidates:
            normalized = _normalize_inferred_param_value(param_name, param_type, candidate)
            if normalized is None:
                continue
            if is_file_param:
                if not _path_matches_expected_types(str(normalized), expected_exts):
                    continue
            # Only accept if LLM confirms match (when available)
            if llm:
                if not _dependency_match_with_llm(
                    llm=llm,
                    tool_name=tool_name,
                    param_name=param_name,
                    param_type=param_type,
                    param_desc=param_desc,
                    task_description=task.content,
                    dependency_task_description=dep_task_desc,
                    dependency_output_summary=dep_output_summary,
                    candidate_value=str(normalized),
                ):
                    continue
                return normalized
            # No LLM: be conservative and require explicit extension match
            if expected_exts and _path_matches_expected_types(str(normalized), expected_exts):
                return normalized
    return None


def _tool_requires_matched_fields(input_params: List[Dict[str, Any]]) -> bool:
    """Check if tool requires matched join fields (csv_fields + rds_fields)."""
    names = {param.get("name") for param in input_params if isinstance(param, dict)}
    return "csv_fields" in names and "rds_fields" in names


def _find_primary_key_column_from_meta_csv(
    preprocess_files: Dict[str, Dict[str, Any]],
    csv_file_path: Optional[str] = None
) -> Optional[str]:
    """
    Find primary key column from user-provided meta CSV file.
    
    Looks for columns that represent primary keys, such as:
    - main_name
    - name
    - id
    - identifier
    - barcode
    - cell_id
    - etc.
    
    Args:
        preprocess_files: Files from preprocessing stage
        csv_file_path: Optional CSV file path to check (if provided, checks that specific file)
    
    Returns:
        Primary key column name, or None if not found
    """
    # Priority list of primary key column names (case-insensitive)
    primary_key_candidates = [
        "main_name", "mainname", "main_name",
        "name", "id", "identifier", "barcode", "cell_id", "cellid",
        "sample_id", "sampleid", "patient_id", "patientid",
        "sequence_id", "sequenceid", "seq_id", "seqid"
    ]
    
    print(f"  [FindPrimaryKey] Searching for meta CSV file in {len(preprocess_files)} preprocess files")
    
    # First, try to find meta CSV file from user input
    meta_csv_file = None
    for file_key, file_info in preprocess_files.items():
        if not isinstance(file_info, dict):
            continue
        
        # Check if this is a meta CSV file (user-provided metadata)
        file_key_lower = file_key.lower()
        data_type = (file_info.get("data_type") or file_info.get("type", "")).lower()
        original_path = file_info.get("original_path", "").lower()
        sandbox_path = file_info.get("sandbox_path", "")
        
        is_meta_csv = (
            "meta" in file_key_lower or
            "metadata" in file_key_lower or
            "meta" in data_type or
            "metadata" in data_type or
            ("meta" in original_path and original_path.endswith(".csv"))
        )
        
        if is_meta_csv and sandbox_path.endswith(".csv"):
            meta_csv_file = file_info
            print(f"  [FindPrimaryKey] Found meta CSV: {file_key} -> {sandbox_path}")
            break
    
    # If csv_file_path is provided, also check that specific file
    if csv_file_path and not meta_csv_file:
        print(f"  [FindPrimaryKey] Checking specific CSV file: {csv_file_path}")
        for file_key, file_info in preprocess_files.items():
            if isinstance(file_info, dict):
                sandbox_path = file_info.get("sandbox_path", "")
                if sandbox_path == csv_file_path or (csv_file_path and sandbox_path.endswith(csv_file_path.split("/")[-1])):
                    meta_csv_file = file_info
                    print(f"  [FindPrimaryKey] Found matching CSV file: {file_key} -> {sandbox_path}")
                    break
    
    if not meta_csv_file:
        print(f"  [FindPrimaryKey] No meta CSV file found")
        return None
    
    # Get columns from meta CSV file
    columns = meta_csv_file.get("columns", [])
    if not columns:
        # Try to get from column_names if columns is not available
        columns = meta_csv_file.get("column_names", [])
    
    if not columns:
        print(f"  [FindPrimaryKey] Meta CSV file found but no columns available")
        return None
    
    print(f"  [FindPrimaryKey] Meta CSV has {len(columns)} columns: {columns[:10]}{'...' if len(columns) > 10 else ''}")
    
    # Convert to lowercase for case-insensitive matching
    columns_lower = [col.lower() for col in columns]
    
    # Find the first matching primary key candidate
    for candidate in primary_key_candidates:
        candidate_lower = candidate.lower()
        # Try exact match first
        if candidate_lower in columns_lower:
            idx = columns_lower.index(candidate_lower)
            found_col = columns[idx]  # Return original case
            print(f"  [FindPrimaryKey] Found primary key column (exact match): {found_col}")
            return found_col
        # Try partial match (e.g., "main_name" matches "main_name_1")
        for col in columns:
            if candidate_lower in col.lower() or col.lower() in candidate_lower:
                print(f"  [FindPrimaryKey] Found primary key column (partial match): {col}")
                return col
    
    # If no match found, return the first column as fallback
    if columns:
        print(f"  [FindPrimaryKey] No primary key column found, using first column: {columns[0]}")
        return columns[0]
    
    return None


def _validate_matched_fields(tool_name: str, all_params: Dict[str, Any], missing_params: List[str]) -> None:
    """
    Ensure csv_fields and rds_fields are both present and aligned.
    
    IMPORTANT: csv_fields and rds_fields should have the same value.
    If csv_fields is set, automatically set rds_fields to the same value.
    """
    csv_fields = all_params.get("csv_fields")
    rds_fields = all_params.get("rds_fields")
    
    # Rule 1: If csv_fields is set, rds_fields should be the same
    if csv_fields and not rds_fields:
        all_params["rds_fields"] = csv_fields
        print(f"  [ParamInfer] Auto-set rds_fields={csv_fields} to match csv_fields for {tool_name}")
        return
    
    # Rule 2: If rds_fields is set but csv_fields is not, copy rds_fields to csv_fields
    if rds_fields and not csv_fields:
        all_params["csv_fields"] = rds_fields
        print(f"  [ParamInfer] Auto-set csv_fields={rds_fields} to match rds_fields for {tool_name}")
        return
    
    # Rule 3: If both are set but different, make them the same (use csv_fields as source of truth)
    if isinstance(csv_fields, str) and isinstance(rds_fields, str):
        csv_parts = [p.strip() for p in csv_fields.split(",") if p.strip()]
        rds_parts = [p.strip() for p in rds_fields.split(",") if p.strip()]
        if csv_parts != rds_parts:
            # Make rds_fields match csv_fields
            all_params["rds_fields"] = csv_fields
            print(f"  [ParamInfer] Aligned rds_fields={csv_fields} to match csv_fields for {tool_name}")


def _validate_integrate_bcr_inputs(tool_name: str, all_params: Dict[str, Any], missing_params: List[str]) -> None:
    """Validate inputs for integrate_bcr_data_complete to avoid Excel/zip inputs."""
    if tool_name != "integrate_bcr_data_complete":
        return
    csv_file = all_params.get("csv_file")
    if not isinstance(csv_file, str):
        return
    _, ext = os.path.splitext(csv_file)
    ext = ext.lower().lstrip(".")
    if ext in {"xls", "xlsx", "zip"} or _looks_like_excel_zip(csv_file):
        all_params.pop("csv_file", None)
        missing_params.append(f"{tool_name}.csv_file")
        print(f"  ⚠ {tool_name}.csv_file expects CSV, got Excel/zip: {csv_file}")


def _has_pandas() -> bool:
    """Check if pandas is available for local preprocessing."""
    try:
        import pandas  # noqa: F401
        return True
    except Exception:
        return False


def _build_task_result_summary(task: SubTask, result: TaskExecutionResult) -> Dict[str, Any]:
    """Build a per-task execution summary"""
    return {
        "task_id": task.task_id,
        "status": result.status.value if hasattr(result.status, "value") else str(result.status),
        "execution_mode": result.execution_mode,
        "parameters": result.parameters,
        "missing_parameters": result.missing_parameters,
        "output_brief": _summarize_output_brief(result.output),
        "error": result.error,
        "error_type": result.error_type or _extract_error_type_from_exec_output(result.output),
        "error_category": result.error_category.value if result.error_category else None,
        "confidence_score": result.confidence_score,
        "result_satisfied": result.result_satisfied,
        "failure_analysis": result.failure_analysis
    }


def classify_error(error: str, error_type: str) -> ErrorCategory:
    """Classify error type"""
    # Handle None or empty error
    if error is None:
        error = ""
    if error_type is None:
        error_type = "UnknownError"
    
    error_lower = error.lower() if error else ""
    error_type_lower = error_type.lower() if error_type else ""

    # Hard timeouts: avoid infinite retries, treat as system errors
    timeout_hard_keywords = [
        "task timeout",
        "execution timed out",
        "timed out after",
        "timeout after"
    ]
    if any(keyword in error_lower for keyword in timeout_hard_keywords):
        return ErrorCategory.SYSTEM_ERROR
    
    # Network errors (prioritize identification, need special handling)
    network_keywords = [
        "connection", "network", "dns", "socket", "timeout", "timed out",
        "connection refused", "connection reset", "connection aborted",
        "network unreachable", "host unreachable", "no route to host",
        "502", "503", "504", "connection error", "network error"
    ]
    if any(keyword in error_lower or keyword in error_type_lower for keyword in network_keywords):
        return ErrorCategory.NETWORK_ERROR
    
    # Retryable errors (other temporary errors)
    retryable_keywords = [
        "rate limit", "429", "retry", "busy", "temporary",
        "service unavailable", "too many requests", "throttle"
    ]
    if any(keyword in error_lower or keyword in error_type_lower for keyword in retryable_keywords):
        return ErrorCategory.RETRYABLE
    
    # Parameter errors
    param_keywords = [
        "parameter", "argument", "invalid argument", "missing required",
        "type error", "value error", "keyerror", "attributeerror"
    ]
    if any(keyword in error_lower or keyword in error_type_lower for keyword in param_keywords):
        return ErrorCategory.PARAMETER_ERROR
    
    # Code errors
    code_keywords = [
        "syntax", "indentation", "nameerror", "not defined",
        "logic error", "indexerror", "zerodivisionerror"
    ]
    if any(keyword in error_lower or keyword in error_type_lower for keyword in code_keywords):
        return ErrorCategory.CODE_ERROR
    
    # Default: system error
    return ErrorCategory.SYSTEM_ERROR


def _analyze_failure(error: str, error_type: str, error_category: ErrorCategory) -> str:
    """
    Analyze failure reason
    
    Args:
        error: Error message
        error_type: Error type
        error_category: Error category
    
    Returns:
        Failure reason analysis
    """
    # Handle None values
    if error is None:
        error = "Unknown error"
    if error_type is None:
        error_type = "UnknownError"
    
    error_category_str = error_category.value if (error_category and hasattr(error_category, 'value')) else str(error_category) if error_category else "Unknown"
    analysis = f"Error type: {error_category_str}\n"
    analysis += f"Error category: {error_type}\n"
    
    if error_category == ErrorCategory.NETWORK_ERROR:
        analysis += "Reason: Network connection issue, may be temporary network failure.\n"
        analysis += "Suggestion: System will automatically retry, network issues usually recover quickly."
    elif error_category == ErrorCategory.RETRYABLE:
        analysis += "Reason: Temporary error, may be service busy or rate limited.\n"
        analysis += "Suggestion: System will automatically retry, may succeed after waiting."
    elif error_category == ErrorCategory.PARAMETER_ERROR:
        analysis += "Reason: Incorrect parameters or missing required parameters.\n"
        analysis += "Suggestion: Check task parameter configuration, ensure all required parameters are provided and format is correct."
    elif error_category == ErrorCategory.CODE_ERROR:
        analysis += "Reason: Code logic error or syntax error.\n"
        analysis += "Suggestion: Check generated code, fix syntax or logic issues."
    else:
        analysis += "Reason: System-level error, may require manual intervention.\n"
        analysis += "Suggestion: Check system status, may need to restart service or contact administrator."
    
    return analysis


def _detect_deadlock(waiting_tasks: List[SubTask], state: ExecutorState) -> bool:
    """
    Detect deadlock: Check if waiting tasks form circular dependencies
    
    Args:
        waiting_tasks: List of tasks waiting for dependencies
        state: Executor state
    
    Returns:
        Whether deadlock is detected
    """
    # Build dependency graph
    task_ids = {task.task_id for task in waiting_tasks}
    dependency_graph = {}
    
    for task in waiting_tasks:
        # Only consider dependencies in the waiting task list
        deps = [dep_id for dep_id in task.dependencies if dep_id in task_ids]
        dependency_graph[task.task_id] = deps
    
    # Use DFS to detect cycles
    visited = set()
    rec_stack = set()
    
    def has_cycle(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in dependency_graph.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    # Check all nodes
    for task_id in task_ids:
        if task_id not in visited:
            if has_cycle(task_id):
                return True
    
    return False


def _generate_suggestions(
    error_category: ErrorCategory,
    error: str,
    retry_count: int,
    max_retries: int
) -> List[str]:
    """
    Generate improvement suggestions
    
    Args:
        error_category: Error category
        error: Error message
        retry_count: Current retry count
        max_retries: Maximum retry count
    
    Returns:
        List of suggestions
    """
    suggestions = []
    
    if error_category == ErrorCategory.NETWORK_ERROR:
        suggestions.append("Network error: System will automatically retry")
        if retry_count < max_retries:
            suggestions.append(f"Current retry count: {retry_count}/{max_retries}")
            suggestions.append("Suggestion: Check network connection, wait for automatic retry")
        else:
            suggestions.append("Maximum retry count reached, please check network connection or retry manually later")
    
    elif error_category == ErrorCategory.RETRYABLE:
        suggestions.append("Retryable error: System will automatically retry")
        if retry_count < max_retries:
            suggestions.append(f"Current retry count: {retry_count}/{max_retries}")
            suggestions.append("Suggestion: Wait for service recovery and automatic retry")
        else:
            suggestions.append("Maximum retry count reached, please retry manually later")
    
    elif error_category == ErrorCategory.PARAMETER_ERROR:
        suggestions.append("Parameter error: Need to correct task parameters")
        suggestions.append("Suggestion: Check task configuration, ensure parameter format is correct")
        suggestions.append("Suggestion: Check tools parameters table, confirm required parameters")
    
    elif error_category == ErrorCategory.CODE_ERROR:
        suggestions.append("Code error: Need to fix code logic")
        suggestions.append("Suggestion: Check generated code, fix syntax or logic issues")
        suggestions.append("Suggestion: Check error stack trace, locate specific problem")
    
    else:
        suggestions.append("System error: May require manual intervention")
        suggestions.append("Suggestion: Check system status and logs")
        suggestions.append("Suggestion: Contact system administrator")
    
    return suggestions


# ===================== Executor Nodes =====================

def initialize_tasks_node(state: ExecutorState) -> ExecutorState:
    """
    Initialize tasks node
    
    1. Expand parallel task groups, merge all tasks into subtasks
    2. Initialize task status mapping
    3. Mark tasks without dependencies as ready
    4. Mark tasks with dependencies as waiting for dependencies
    """
    # Expand tasks in parallel task groups, merge into subtasks
    all_tasks = list(state.subtasks)  # Copy list to avoid modifying original
    
    # Extract all tasks from parallel task groups
    # Note: Values in parallel_task_groups may be ParallelTaskGroup objects or dicts
    parallel_tasks_count = 0
    for group_id, group in state.parallel_task_groups.items():
        group_subtasks = None
        
        # Handle case where group may be object or dict
        if isinstance(group, dict):
            # If it's a dict, try to get subtasks
            group_subtasks = group.get('subtasks', [])
            # If subtasks is a list of dicts, need to convert to SubTask objects
            if group_subtasks and isinstance(group_subtasks[0], dict):
                try:
                    from state import SubTask
                    group_subtasks = [SubTask.model_validate(task_dict) for task_dict in group_subtasks]
                except Exception as e:
                    print(f"  ⚠ Failed to convert tasks in parallel task group {group_id} to SubTask objects: {e}")
                    continue
        elif hasattr(group, 'subtasks'):
            # If it's an object, directly get subtasks
            group_subtasks = group.subtasks
        
        if group_subtasks:
            for task in group_subtasks:
                # Ensure task is a SubTask object
                if isinstance(task, dict):
                    try:
                        from state import SubTask
                        task = SubTask.model_validate(task)
                    except Exception as e:
                        print(f"  ⚠ Failed to convert task {task.get('task_id', 'unknown')} to SubTask object: {e}")
                        continue
                
                # Check if task is already in subtasks (avoid duplicates)
                if not any(t.task_id == task.task_id for t in all_tasks):
                    all_tasks.append(task)
                    parallel_tasks_count += 1
                    print(f"  [DEBUG] Added task {task.task_id} from parallel task group {group_id}")
    
    # Update subtasks to complete list containing all tasks
    state.subtasks = all_tasks
    state.total_tasks = len(state.subtasks)
    
    serial_tasks_count = len(state.subtasks) - parallel_tasks_count
    print(f"✓ Initialized tasks: {state.total_tasks} tasks total (serial: {serial_tasks_count}, parallel: {parallel_tasks_count}, parallel groups: {len(state.parallel_task_groups)})")
    
    # Initialize status for all tasks
    for task in state.subtasks:
        if not task.dependencies:
            # Tasks without dependencies, directly mark as ready
            state.task_status_map[task.task_id] = ExecutorTaskStatus.READY
        else:
            # Tasks with dependencies, mark as waiting for dependencies
            state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_DEPENDENCY
    
    ready_count = sum(1 for s in state.task_status_map.values() if s == ExecutorTaskStatus.READY)
    print(f"  Ready tasks: {ready_count}, waiting for dependencies: {state.total_tasks - ready_count}")
    
    # =================================================================
    # Generate todo-list.md in sandbox directory
    # This allows codeact to track task progress via TodoListManager
    # =================================================================
    try:
        from .todolist_generator import generate_and_save_todolist_from_state
        
        # Get sandbox directory from parent_state or state
        sandbox_dir = None
        if state.parent_state:
            # Prefer sandbox_data_dir (session directory) over sandbox_dir
            sandbox_dir = getattr(state.parent_state, 'sandbox_data_dir', None) or state.sandbox_dir
        
        if sandbox_dir and state.parent_state:
            # Create a minimal GlobalState-like object for todolist generator
            # We use parent_state directly since it's a GlobalState
            todo_list = generate_and_save_todolist_from_state(state.parent_state, sandbox_dir)
            if todo_list:
                print(f"  📋 Generated todo-list.md with {len(todo_list.tasks)} tasks")
            else:
                print(f"  ⚠️ Failed to generate todo-list.md")
        else:
            print(f"  ⚠️ No sandbox directory available, skipping todo-list.md generation")
    except Exception as e:
        print(f"  ⚠️ Error generating todo-list.md: {e}")
        import traceback
        traceback.print_exc()
    
    return state


def infer_parameters_node(state: ExecutorState) -> ExecutorState:
    """
    Parameter inference node
    
    Infer parameters per ready subtask using:
    1. Dependency task outputs (completed results)
    2. User input and execution plan context
    3. Recommended values from tools parameters table
    4. LLM inference (fallback)
    """
    # Import at function level to avoid "cannot access local variable" error
    from pathlib import Path as _Path
    
    _restore_parent_state(state)
    
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if not ready_tasks:
        return state
    
    user_input = state.parent_state.user_input if state.parent_state else ""
    execution_plan = state.parent_state.execution_plan if state.parent_state else None
    
    # Get pre-extracted parameter table from preprocessing stage
    preprocess_params = {}
    preprocess_files = {}
    preprocess_file_paths = {}
    pending_fasta_conversions = {}
    if state.parent_state and hasattr(state.parent_state, 'extracted_parameters') and state.parent_state.extracted_parameters:
        extracted = state.parent_state.extracted_parameters
        preprocess_params = extracted.get("user_parameters", {})
        preprocess_files = extracted.get("files", {})
        preprocess_file_paths = extracted.get("sandbox_file_paths", {})
        pending_fasta_conversions = extracted.get("pending_fasta_conversions", {})
        # Also include generated FASTA files
        generated_fasta = extracted.get("generated_fasta_files", {})
        if generated_fasta:
            preprocess_file_paths.update(generated_fasta)
        print(f"  [ParamInfer] Using pre-extracted parameters: {len(preprocess_params)} params, {len(preprocess_files)} files")
        if pending_fasta_conversions:
            print(f"  [ParamInfer] Found {len(pending_fasta_conversions)} pending FASTA conversions")

    llm = create_reasoning_llm()
    tools_params_map = _load_tools_params_table() or {}
    
    # Initialize sandbox_data_dir at function scope to avoid scope issues
    sandbox_data_dir = None
    if state.parent_state and hasattr(state.parent_state, 'sandbox_data_dir'):
        sandbox_data_dir = getattr(state.parent_state, 'sandbox_data_dir', None)
    
    for task in ready_tasks:
        # Initialize task result
        if task.task_id not in state.task_results:
            state.task_results[task.task_id] = TaskExecutionResult(
                task_id=task.task_id,
                status=ExecutorTaskStatus.READY,
                execution_mode=""
            )
        
        result = state.task_results[task.task_id]
        task_result = task.result if isinstance(task.result, dict) else {}
        tools = task_result.get("tools", [])
        inputs = task_result.get("inputs", [])
        file_candidates = _extract_file_candidates_from_context(
            user_input=user_input,
            execution_plan=execution_plan,
            task_description=task.content
        )
        raw_task_params = task_result.get("parameters", {})
        provided_params: Dict[str, Any] = {}
        provided_params_by_tool: Dict[str, Dict[str, Any]] = {}
        
        # First, populate from pre-extracted parameters (lowest priority, will be overwritten)
        for key, value in preprocess_params.items():
            if key not in ['task_description', 'analysis_type', 'notes']:  # Skip meta fields
                provided_params[key] = value
        
        # Add file paths from preprocessing
        # Key insight: CSV files with sequence columns can be used as `sequences` parameter
        # The tool call hook will automatically convert CSV to FASTA when needed
        # BCR (antibody) columns
        sequence_column_names = [
            'heavy_dna', 'Heavy_DNA', 'HEAVY_DNA', 'light_dna', 'Light_DNA', 'LIGHT_DNA',
            'Heavy', 'HEAVY', 'Light', 'LIGHT', 'sequence', 'Sequence', 'seq', 'Seq',
            'nt_sequence', 'aa_sequence', 'cdr3', 'CDR3', 'vh', 'VH', 'vl', 'VL', 'vhh', 'VHH',
            # Additional common sequence column names
            'variant_seq', 'variant_seq_1', 'variant_seq_2', 'variant_seq_3',
            'heavy_chain', 'light_chain', 'HeavyChain', 'LightChain',
            'hc_seq', 'lc_seq', 'HC_seq', 'LC_seq',
            'full_sequence', 'dna_sequence', 'nucleotide_sequence',
            # TCR (T cell receptor) columns - for receptor_type: TCR
            # Alpha chain
            'alpha_dna', 'alpha_seq', 'alpha_chain', 'TRA', 'tra', 'tra_seq',
            'CDR3a', 'cdr3a', 'cdr3_alpha', 'TRAV',
            # Beta chain
            'beta_dna', 'beta_seq', 'beta_chain', 'TRB', 'trb', 'trb_seq',
            'CDR3b', 'cdr3b', 'cdr3_beta', 'TRBV',
            # Gamma chain
            'gamma_dna', 'gamma_seq', 'gamma_chain', 'TRG', 'trg', 'trg_seq',
            'CDR3g', 'cdr3g', 'cdr3_gamma', 'TRGV',
            # Delta chain
            'delta_dna', 'delta_seq', 'delta_chain', 'TRD', 'trd', 'trd_seq',
            'CDR3d', 'cdr3d', 'cdr3_delta', 'TRDV',
            # Generic TCR patterns
            'tcr_alpha', 'tcr_beta', 'tcr_gamma', 'tcr_delta',
            'TCR_alpha', 'TCR_beta', 'TCR_gamma', 'TCR_delta',
        ]
        
        # Track all available files from preprocessing for diagnostic purposes
        available_files_from_preprocess: Dict[str, Dict[str, Any]] = {}
        
        for file_key, file_info in preprocess_files.items():
            if isinstance(file_info, dict):
                sandbox_path = file_info.get("sandbox_path")
                if sandbox_path:
                    file_type = file_info.get("data_type") or file_info.get("type", "")
                    file_columns = file_info.get("columns") or []
                    original_path = file_info.get("original_path", "")
                    
                    # Store for diagnostics
                    available_files_from_preprocess[file_key] = {
                        "sandbox_path": sandbox_path,
                        "file_type": file_type,
                        "columns": file_columns,
                        "original_path": original_path,
                        "mapped_to": []  # Will track which params this file was mapped to
                    }
                    
                    # CRITICAL: Use data_type (user-specified purpose) for mapping, not just file_key
                    data_type_lower = file_type.lower()
                    
                    # Check for antigen-related files FIRST (before antibody detection)
                    # This ensures user-specified "antigen_file" is not overridden
                    is_antigen_file = (
                        "antigen" in file_key.lower() or 
                        "antigen" in data_type_lower or
                        data_type_lower in ["antigen_file", "antigen_data", "antigen_sequence", "antigen"]
                    )
                    
                    if is_antigen_file:
                        provided_params.setdefault("antigen_file", sandbox_path)
                        available_files_from_preprocess[file_key]["mapped_to"].append("antigen_file")
                        print(f"  [ParamInfer] Antigen file detected: {sandbox_path} (data_type={file_type})")
                    
                    # FASTA files directly map to sequence parameters
                    elif "fasta" in data_type_lower or "sequence" in file_key.lower():
                        provided_params.setdefault("fasta_file", sandbox_path)
                        provided_params.setdefault("input_fasta", sandbox_path)
                        provided_params.setdefault("sequences", sandbox_path)
                        available_files_from_preprocess[file_key]["mapped_to"].extend(["fasta_file", "input_fasta", "sequences"])
                    
                    # CSV files with sequence columns can also be used as `sequences` parameter
                    # The hook will convert CSV to FASTA at tool call time
                    if (data_type_lower == "csv" or sandbox_path.endswith('.csv')) and not is_antigen_file:
                        has_seq_cols = any(col in sequence_column_names for col in file_columns)
                        if has_seq_cols:
                            print(f"  [ParamInfer] CSV with sequence columns found: {sandbox_path}")
                            print(f"  [ParamInfer] Sequence columns: {[c for c in file_columns if c in sequence_column_names]}")
                            # Map CSV to sequences parameter - hook will auto-convert
                            provided_params.setdefault("sequences", sandbox_path)
                            provided_params.setdefault("fasta_file", sandbox_path)
                            provided_params.setdefault("input_fasta", sandbox_path)
                            available_files_from_preprocess[file_key]["mapped_to"].extend(["sequences", "fasta_file", "input_fasta"])
                        else:
                            # Regular CSV without sequence columns - could be antibody metadata
                            # Check if user specified antibody purpose
                            # "meta csv" / "metadata" files in BCR analysis are typically antibody data files
                            is_antibody_file = (
                                "antibody" in file_key.lower() or 
                                "antibody" in data_type_lower or
                                "meta" in file_key.lower() or  # "meta csv file" usually contains antibody info
                                "metadata" in file_key.lower()
                            )
                            if is_antibody_file:
                                # Map to both antibody_file and metadata_file for flexibility
                                provided_params.setdefault("antibody_file", sandbox_path)
                                provided_params.setdefault("metadata_file", sandbox_path)
                                available_files_from_preprocess[file_key]["mapped_to"].extend(["antibody_file", "metadata_file"])
                            else:
                                provided_params.setdefault("metadata_file", sandbox_path)
                                available_files_from_preprocess[file_key]["mapped_to"].append("metadata_file")
                    
                    # RDS files - map to multiple possible param names
                    if "rds" in file_type.lower() or sandbox_path.endswith('.rds') or sandbox_path.endswith('.RDS'):
                        provided_params.setdefault("rds_file", sandbox_path)
                        provided_params.setdefault("seurat_object", sandbox_path)
                        provided_params.setdefault("rds_path", sandbox_path)
                        provided_params.setdefault("input_rds", sandbox_path)
                        available_files_from_preprocess[file_key]["mapped_to"].extend(["rds_file", "seurat_object", "rds_path", "input_rds"])
                    
                    # NEW: Use 'can_be_used_as' field from tool outputs for smart parameter matching
                    # This enables automatic matching of tool output files to subsequent tool inputs
                    can_be_used_as = file_info.get("can_be_used_as", [])
                    source_tool = file_info.get("source_tool", "")
                    if can_be_used_as:
                        print(f"  [ParamInfer] Output from {source_tool}: {Path(sandbox_path).name}")
                        print(f"    Can be used as: {can_be_used_as}")
                        for param_type in can_be_used_as:
                            provided_params.setdefault(param_type, sandbox_path)
                            available_files_from_preprocess[file_key]["mapped_to"].append(param_type)
                        # Also store the file description for parameter matching
                        file_description = file_info.get("description", "")
                        if file_description:
                            print(f"    Description: {file_description}")
        
        # Print summary of available files from preprocessing
        if available_files_from_preprocess:
            print(f"  [ParamInfer] === Available files from preprocessing ===")
            for file_key, file_info in available_files_from_preprocess.items():
                print(f"    - {file_key}: {file_info['sandbox_path']}")
                print(f"      Type: {file_info['file_type']}, Mapped to: {file_info['mapped_to']}")
        
        # Add sandbox file path mappings
        for original_path, sandbox_path in preprocess_file_paths.items():
            if isinstance(sandbox_path, str) and sandbox_path.endswith('.fasta'):
                provided_params.setdefault("fasta_file", sandbox_path)
                provided_params.setdefault("input_fasta", sandbox_path)
                provided_params.setdefault("sequences", sandbox_path)
        
        # Debug: show what parameters were mapped
        if "sequences" in provided_params:
            print(f"  [ParamInfer] sequences parameter set to: {provided_params['sequences']}")
        
        # Then, apply task-specific parameters (higher priority)
        # BUT: Skip placeholder values like "task_XXX/output_name" - these should be resolved from actual outputs
        def _is_placeholder_value(val: Any) -> bool:
            """Check if value is a dependency placeholder.
            
            Patterns to detect:
            - 'task_001/output_name' - explicit task reference
            - 'predictions_from_task_003.csv' - LLM-generated placeholder
            - 'integrated_data_from_task_004.rds' - LLM-generated placeholder
            - 'output_from_task_XXX' - output reference
            """
            if not isinstance(val, str):
                return False
            import re
            val_lower = val.lower()
            # Pattern 1: task_XXX/something
            if re.match(r'^task_\d+/', val):
                return True
            # Pattern 2: something_from_task_XXX.ext or something_from_task_XXX
            if re.search(r'_from_task_\d+', val_lower):
                return True
            # Pattern 3: output_from_task_XXX or result_from_task_XXX
            if re.search(r'(output|result)_from_task_\d+', val_lower):
                return True
            # Pattern 4: task_XXX_output.ext pattern
            if re.match(r'^task_\d+_output', val_lower):
                return True
            return False
        
        # Build a set of all valid file paths from parameter table
        # Parameter table = user input files + tool output files
        # File type parameters MUST come from this table
        def _build_valid_file_paths_set() -> Set[str]:
            """Build set of all valid file paths from parameter table."""
            valid_paths: Set[str] = set()
            
            # 1. User input files (from preprocessing)
            for file_key, file_info in preprocess_files.items():
                if isinstance(file_info, dict):
                    sandbox_path = file_info.get("sandbox_path", "")
                    original_path = file_info.get("original_path", "")
                    if sandbox_path:
                        valid_paths.add(sandbox_path)
                    if original_path:
                        valid_paths.add(original_path)
            
            # 2. Tool output files (from task_outputs)
            if state.parent_state and hasattr(state.parent_state, 'extracted_parameters'):
                task_outputs = state.parent_state.extracted_parameters.get("task_outputs", {})
                for task_id, output_info in task_outputs.items():
                    if isinstance(output_info, dict):
                        output_files = output_info.get("output_files", [])
                        for out_file in output_files:
                            if isinstance(out_file, str) and out_file:
                                valid_paths.add(out_file)
            
            return valid_paths
        
        _valid_file_paths = _build_valid_file_paths_set()
        
        def _is_valid_file_param_value(val: Any, is_file_param: bool) -> bool:
            """Check if a file parameter value exists in the parameter table.
            
            Core principle: File type parameters MUST come from:
            1. User input files (preprocess_files)
            2. Tool output files (task_outputs)
            
            Any file path not in this table is considered invalid/fabricated.
            """
            if not is_file_param:
                return True  # Non-file params don't need validation
            
            if not isinstance(val, str):
                return False
            
            # Check if value exists in valid file paths
            return val in _valid_file_paths
        
        # Build a mapping from original paths to sandbox paths for path conversion
        original_to_sandbox_path: Dict[str, str] = {}
        for file_key, file_info in preprocess_files.items():
            if isinstance(file_info, dict):
                orig = file_info.get("original_path", "")
                sand = file_info.get("sandbox_path", "")
                if orig and sand and orig != sand:
                    original_to_sandbox_path[orig] = sand
        
        def _convert_to_sandbox_path(value: Any) -> Any:
            """Convert original path to sandbox path if applicable."""
            if isinstance(value, str) and value in original_to_sandbox_path:
                converted = original_to_sandbox_path[value]
                print(f"  [ParamInfer] Converting path: {value} -> {converted}")
                return converted
            return value
        
        if isinstance(raw_task_params, dict):
            for key, value in raw_task_params.items():
                # Skip placeholder values - they should be resolved from actual dependency outputs
                if _is_placeholder_value(value):
                    print(f"  [ParamInfer] Skipping placeholder value for {key}: {value}")
                    continue
                
                # CRITICAL: For file type parameters, validate that the value exists in parameter table
                # Parameter table = user input files + tool output files
                # This prevents using LLM-fabricated file paths that don't exist
                is_file_param = _is_file_type(key, None)  # Use param name to detect file type
                if is_file_param and isinstance(value, str):
                    if not _is_valid_file_param_value(value, is_file_param):
                        print(f"  [ParamInfer] Skipping invalid file path for {key}: {value} (not in parameter table)")
                        print(f"  [ParamInfer] Valid paths in table: {len(_valid_file_paths)} files")
                        continue
                
                # CRITICAL: Convert original paths to sandbox paths
                # Task decomposition may use original paths, but sandbox needs sandbox paths
                if isinstance(value, str) and value in original_to_sandbox_path:
                    converted_value = original_to_sandbox_path[value]
                    print(f"  [ParamInfer] Converting path for {key}: {value} -> {converted_value}")
                    value = converted_value
                
                if isinstance(key, str) and "." in key:
                    tool_prefix, param_key = key.split(".", 1)
                    provided_params_by_tool.setdefault(tool_prefix, {})[param_key] = value
                else:
                    provided_params[key] = value
        
        if not tools:
            # Tasks without tools, parameters are empty
            result.parameters = {}
            result.missing_parameters = []
            continue
        
        # Collect parameter requirements for all tools
        all_params = {}
        missing_params = []
        param_sources: Dict[str, str] = {}
        
        for tool_item in tools:
            tool_name = None
            if isinstance(tool_item, str):
                tool_name = tool_item
            elif isinstance(tool_item, dict):
                tool_name = tool_item.get("tool_name") or tool_item.get("name", "")
            
            if not tool_name:
                continue
            
            # Find tool parameter definition (with fuzzy matching)
            tool_params = tools_params_map.get(tool_name)
            if not tool_params:
                for key in tools_params_map.keys():
                    if "_" in key:
                        parts = key.split("_", 1)
                        if len(parts) > 1 and (parts[1] == tool_name or parts[0] == tool_name):
                            tool_params = tools_params_map.get(key)
                            break
                    if key == tool_name:
                        tool_params = tools_params_map.get(key)
                        break
            if not tool_params:
                tool_name_lower = tool_name.lower()
                for key in tools_params_map.keys():
                    key_lower = key.lower()
                    if tool_name_lower in key_lower or key_lower in tool_name_lower:
                        tool_params = tools_params_map.get(key)
                        break
            
            # If tool is not in parameter table, use inputs field to infer parameters
            if not tool_params:
                # Infer parameters from inputs field
                for input_param in inputs:
                    if input_param not in all_params:
                        tool_scoped_params = provided_params_by_tool.get(tool_name, {})
                        if input_param in tool_scoped_params or input_param in provided_params:
                            raw_value = tool_scoped_params.get(input_param, provided_params.get(input_param))
                            normalized = _normalize_inferred_param_value(input_param, None, raw_value)
                            if normalized is not None:
                                all_params[input_param] = _normalize_base_dir_value(input_param, normalized)
                                param_sources[input_param] = "task_parameters"
                                continue
                        dep_value = _resolve_param_from_dependencies(
                            task=task,
                            tool_name=tool_name,
                            param_name=input_param,
                            param_type=None,
                            param_desc=None,
                            state=state,
                            llm=llm,
                        )
                        if dep_value is not None:
                            all_params[input_param] = dep_value
                            param_sources[input_param] = "dependency"
                            continue
                        # Try to infer from context + LLM
                        if llm:
                            try:
                                from langchain_core.messages import SystemMessage, HumanMessage
                                inference_prompt = f"""
Please infer parameter value based on the following information:

Task description: {task.content}
User input: {user_input}
Execution plan: {execution_plan or 'None'}
Tool name: {tool_name}
Parameter name: {input_param}
Task input list: {inputs}
Dependency outputs: {[state.task_results.get(dep_id).output for dep_id in task.dependencies if dep_id in state.task_results]}

**CRITICAL TYPE VALIDATION RULES:**
1. If parameter name contains "timeout", "duration", "seconds", "count", "num", "max", "min": value MUST be a number (e.g., 60, 120), NOT a string or tool name
2. If parameter name contains "file", "path": value MUST be a valid file path starting with "/" or drive letter
3. If parameter name contains "enable", "disable", "skip", "use": value MUST be true or false
4. NEVER use tool names, service names, or unrelated text as parameter values
5. For timeout/duration parameters: use reasonable default values (e.g., 60, 120, 300 seconds)

Please infer the value of this parameter. If it cannot be inferred from the task description, or requires user input (such as file paths, user selections, etc.), return null.

Return JSON format:
{{
    "value": <parameter value matching the expected type, or null>,
    "can_infer": <true/false>,
    "reason": "<inference reason or why user input is required>"
}}
"""
                                messages = [
                                    SystemMessage(content="You are a professional parameter inference expert. You MUST strictly validate parameter types - never assign string values to numeric parameters, never use tool names as parameter values."),
                                    HumanMessage(content=inference_prompt)
                                ]
                                response = llm.invoke(messages)
                                response_text = response.content.strip()
                                
                                import re
                                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                                if json_match:
                                    # Ensure using global json module
                                    import json as json_module
                                    inference_data = json_module.loads(json_match.group())
                                    param_value = inference_data.get("value")
                                    can_infer = inference_data.get("can_infer", False)
                                    
                                    if param_value is not None and can_infer:
                                        normalized = _normalize_inferred_param_value(input_param, None, param_value)
                                        if normalized is not None:
                                            all_params[input_param] = _normalize_base_dir_value(input_param, normalized)
                                            param_sources[input_param] = "llm"
                                        else:
                                            missing_params.append(f"{tool_name}.{input_param}")
                                    else:
                                        missing_params.append(f"{tool_name}.{input_param}")
                                else:
                                    missing_params.append(f"{tool_name}.{input_param}")
                            except Exception as e:
                                print(f"  ⚠ Failed to infer parameter {input_param}: {e}")
                                missing_params.append(f"{tool_name}.{input_param}")
                        else:
                            # No LLM, directly mark as missing
                            missing_params.append(f"{tool_name}.{input_param}")
                continue
            
            input_params = tool_params.get("input_params", [])
            extracted_params = _extract_parameters_from_context(
                user_input=user_input,
                execution_plan=execution_plan,
                task_description=task.content,
                tool_name=tool_name,
                tool_params=input_params,
                llm=llm
            )
            
            # Special handling for integrate_bcr_data_complete: auto-fill csv_fields and rds_fields
            # This must be done BEFORE the parameter loop to prevent them from being marked as missing
            if tool_name == "integrate_bcr_data_complete":
                print(f"  [ParamInfer] Processing integrate_bcr_data_complete - auto-filling csv_fields and rds_fields")
                csv_file = all_params.get("csv_file", "")
                print(f"  [ParamInfer] CSV file: {csv_file}")
                print(f"  [ParamInfer] Preprocess files available: {len(preprocess_files)} files")
                
                # Try to infer csv_fields from meta CSV file
                csv_key_field = None
                if preprocess_files:
                    csv_key_field = _find_primary_key_column_from_meta_csv(
                        preprocess_files,
                        csv_file_path=csv_file
                    )
                    if csv_key_field:
                        print(f"  [ParamInfer] Found primary key column from meta CSV: {csv_key_field}")
                    else:
                        print(f"  [ParamInfer] No primary key column found in meta CSV")
                
                # Fallback: Check if CSV comes from a specific tool
                if not csv_key_field and csv_file and state.parent_state:
                    task_outputs = state.parent_state.extracted_parameters.get("task_outputs", {})
                    for task_output in task_outputs.values():
                        if isinstance(task_output, dict):
                            for out_file in task_output.get("output_files", []):
                                if out_file == csv_file:
                                    source_tool = task_output.get("tool_name", "")
                                    if source_tool == "analyze_vdj_batch":
                                        # AIRR format uses sequence_id
                                        csv_key_field = "sequence_id"
                                        print(f"  [ParamInfer] CSV from analyze_vdj_batch, using sequence_id as key")
                                    break
                
                # Final fallback: use default
                if not csv_key_field:
                    csv_key_field = "main_name"
                    print(f"  [ParamInfer] Using default csv_fields={csv_key_field} (no meta CSV found)")
                
                # Auto-fill csv_fields if not provided (BEFORE parameter loop)
                if "csv_fields" not in all_params or not all_params.get("csv_fields"):
                    all_params["csv_fields"] = csv_key_field
                    param_sources["csv_fields"] = "auto_inferred"
                    print(f"  [ParamInfer] ✓ Auto-filled csv_fields={csv_key_field} for {tool_name} (from meta CSV, BEFORE loop)")
                else:
                    print(f"  [ParamInfer] csv_fields already set: {all_params.get('csv_fields')}")
                
                # Auto-fill rds_fields to match csv_fields (Rule 2: same value)
                csv_fields_value = all_params.get("csv_fields", csv_key_field)
                if "rds_fields" not in all_params or not all_params.get("rds_fields"):
                    all_params["rds_fields"] = csv_fields_value  # Use same value as csv_fields
                    param_sources["rds_fields"] = "auto_inferred"
                    print(f"  [ParamInfer] ✓ Auto-filled rds_fields={csv_fields_value} for {tool_name} (same as csv_fields, BEFORE loop)")
                else:
                    # Ensure rds_fields matches csv_fields
                    if all_params.get("rds_fields") != csv_fields_value:
                        all_params["rds_fields"] = csv_fields_value
                        param_sources["rds_fields"] = "auto_inferred"
                        print(f"  [ParamInfer] ✓ Aligned rds_fields={csv_fields_value} to match csv_fields")
                    else:
                        print(f"  [ParamInfer] rds_fields already set and matches: {all_params.get('rds_fields')}")
            
            # Special handling for integrate_tcr_data_complete: always use "main_name" for csv_fields and rds_fields
            # This is simpler than BCR integration - no need to infer from meta CSV or trigger HITL
            if tool_name == "integrate_tcr_data_complete":
                print(f"  [ParamInfer] Processing integrate_tcr_data_complete - auto-filling csv_fields and rds_fields with 'main_name'")
                
                # Always use "main_name" for TCR integration
                all_params["csv_fields"] = "main_name"
                all_params["rds_fields"] = "main_name"
                param_sources["csv_fields"] = "auto_inferred"
                param_sources["rds_fields"] = "auto_inferred"
                print(f"  [ParamInfer] ✓ Auto-filled csv_fields=main_name and rds_fields=main_name for {tool_name}")
            
            # Special handling for bioinformatics, tcell, bcell service tools:
            # These tools typically require integrated RDS files as input
            # Auto-match the latest (highest n) integrated_n.rds file
            # Also auto-fill base_dir to sandbox output directory
            # IMPORTANT: Put values directly in all_params to prevent override by PRIORITY 0/1
            tool_service = tool_params.get("service", "")
            bioinformatics_services = {"bioinformatics", "tcell", "bcell", "immune"}
            if tool_service in bioinformatics_services:
                print(f"  [ParamInfer] Bioinformatics/tcell/bcell tool detected: {tool_name} (service={tool_service})")
                
                # Get sandbox output directory for base_dir
                # Note: sandbox_data_dir is already initialized at function scope above
                sandbox_output_dir = None
                if sandbox_data_dir:
                    sandbox_output_dir = f"{sandbox_data_dir}/output"
                    print(f"  [ParamInfer] Sandbox output dir: {sandbox_output_dir}")
                
                # Auto-fill base_dir parameter with sandbox output directory
                # Put directly in all_params to prevent override
                for p in input_params:
                    p_name = p.get("name", "")
                    p_type = p.get("type", "")
                    p_name_lower = p_name.lower()
                    
                    # Handle base_dir parameter - always use sandbox output directory
                    if "base_dir" in p_name_lower or p_name_lower == "base_directory":
                        if sandbox_output_dir and p_name not in all_params:
                            all_params[p_name] = sandbox_output_dir
                            param_sources[p_name] = "bioinformatics_auto"
                            print(f"  [ParamInfer] Auto-filled {p_name}={sandbox_output_dir} for {tool_service} tool (highest priority)")
                
                # Look for input_file parameter that expects RDS
                for p in input_params:
                    p_name = p.get("name", "")
                    p_type = p.get("type", "")
                    if p_name in ("input_file", "rds_file", "input_rds") and "rds" in p_type.lower():
                        rds_candidate = None
                        
                        # Strategy: Find all integrated_n.rds files and pick the one with highest n
                        # These files represent cumulative data integration results
                        integrated_files = []
                        
                        # Check all task outputs for integrated RDS files
                        task_outputs = state.parent_state.extracted_parameters.get("task_outputs", {}) if state.parent_state else {}
                        for task_output in task_outputs.values():
                            if isinstance(task_output, dict):
                                for out_file in task_output.get("output_files", []):
                                    if isinstance(out_file, str) and "integrated_" in out_file and out_file.endswith(".rds"):
                                        integrated_files.append(out_file)
                        
                        # Also check completed task results for integrated RDS files
                        # This is more reliable as task_outputs may not be updated yet
                        for task_result in state.task_results.values():
                            if task_result.status == ExecutorTaskStatus.COMPLETED and task_result.output:
                                output = task_result.output
                                if isinstance(output, dict):
                                    # Check final_result for output_file
                                    final_result = output.get("final_result", {})
                                    if isinstance(final_result, dict):
                                        for key in ["output_file", "output_path", "result_path"]:
                                            out_file = final_result.get(key)
                                            if isinstance(out_file, str) and "integrated_" in out_file and out_file.endswith(".rds"):
                                                if out_file not in integrated_files:
                                                    integrated_files.append(out_file)
                                    # Check messages for result type messages
                                    messages = output.get("messages", [])
                                    for msg in messages:
                                        if isinstance(msg, dict) and msg.get("type") == "result":
                                            raw = msg.get("raw", {})
                                            if isinstance(raw, dict):
                                                for key in ["output_file", "output_path", "result_path"]:
                                                    out_file = raw.get(key)
                                                    if isinstance(out_file, str) and "integrated_" in out_file and out_file.endswith(".rds"):
                                                        if out_file not in integrated_files:
                                                            integrated_files.append(out_file)
                        
                        # Also check preprocess_files for any integrated files
                        for file_key, file_info in preprocess_files.items():
                            if isinstance(file_info, dict):
                                sandbox_path = file_info.get("sandbox_path", "")
                                if "integrated_" in sandbox_path and sandbox_path.endswith(".rds"):
                                    if sandbox_path not in integrated_files:
                                        integrated_files.append(sandbox_path)
                        
                        # Find the one with highest n (most complete data)
                        if integrated_files:
                            import re
                            def extract_n(path: str) -> int:
                                match = re.search(r'integrated_(\d+)\.rds', path)
                                return int(match.group(1)) if match else 0
                            
                            integrated_files.sort(key=extract_n, reverse=True)
                            rds_candidate = integrated_files[0]
                            print(f"  [ParamInfer] Found {len(integrated_files)} integrated RDS files: {integrated_files}")
                            print(f"  [ParamInfer] Using highest n: {rds_candidate}")
                        
                        # Fallback: Use original RDS from preprocessing if no integrated files exist
                        if not rds_candidate:
                            for file_key, file_info in preprocess_files.items():
                                if isinstance(file_info, dict):
                                    sandbox_path = file_info.get("sandbox_path", "")
                                    if sandbox_path.endswith(".rds"):
                                        rds_candidate = sandbox_path
                                        print(f"  [ParamInfer] No integrated RDS found, using original: {rds_candidate}")
                                        break
                        
                        # Put directly in all_params to prevent override by PRIORITY 0/1
                        if rds_candidate and p_name not in all_params:
                            all_params[p_name] = rds_candidate
                            param_sources[p_name] = "bioinformatics_auto"
                            print(f"  [ParamInfer] Auto-filled {p_name}={rds_candidate} for bioinformatics tool (highest priority)")
            
            for param in input_params:
                param_name = param.get("name", "")
                param_type = param.get("type", "")
                param_desc = param.get("description", "")
                param_deme = param.get("deme") or param.get("demo", "")
                # CRITICAL FIX: Use 'required' field from tools_params_table, not type string
                # Before: is_optional = "optional" in param_type.lower() or param_type.startswith("Optional")
                # This was WRONG because tools_params_table uses 'required' field, not 'optional' in type
                is_required = param.get("required", True)  # Default to True (required) if not specified - safer
                is_optional = not is_required  # A parameter is optional if it's NOT required
                
                if not param_name:
                    continue
                
                # Debug: log parameter requirement status
                print(f"  [ParamInfer] Parameter '{param_name}' for tool '{tool_name}': required={is_required}, optional={is_optional}")
                
                # Check if parameter is already in all_params (from parameter inference results)
                # Also check if it's in missing_params with tool prefix
                param_already_handled = (
                    param_name in all_params or 
                    any(p.endswith(f".{param_name}") or p == param_name for p in missing_params) or
                    f"{tool_name}.{param_name}" in missing_params
                )
                
                if param_already_handled:
                    if param_name in all_params:
                        print(f"  [DEBUG] Parameter {param_name} already in all_params with value: {all_params[param_name]}")
                    else:
                        print(f"  [DEBUG] Parameter {param_name} already handled (in missing_params), skipping")
                    continue
                
                print(f"  [DEBUG] Processing parameter {param_name} for tool {tool_name}")
                
                tool_scoped_params = provided_params_by_tool.get(tool_name, {})

                # PRIORITY 0: Check task_outputs for dependency task output files
                # This is the most reliable source as we record output files after each task completes
                # IMPORTANT: Skip output parameters - they should be auto-generated, not resolved from dependencies
                # CRITICAL: Also check SEMANTIC TYPE MATCH - e.g., antigen_file should NOT accept antibody analysis results
                if task.dependencies and _is_file_type(param_name, param_type) and not _is_output_param(param_name, param_type):
                    expected_exts = _expected_file_extensions(param_name, param_type, param_desc)
                    task_outputs = state.parent_state.extracted_parameters.get("task_outputs", {}) if state.parent_state else {}
                    
                    for dep_task_id in task.dependencies:
                        dep_output_info = task_outputs.get(dep_task_id)
                        if not dep_output_info or dep_output_info.get("status") != "completed":
                            continue
                        
                        # SEMANTIC CHECK: Verify that the dependency output type matches parameter expectation
                        # For example: antigen_file should NOT accept AIRR results (which are antibody analysis)
                        if not _check_dependency_output_semantic_match(dep_task_id, param_name, state):
                            print(f"    [DEBUG] Skipping dep {dep_task_id} for {param_name}: semantic type mismatch")
                            continue
                        
                        output_files = dep_output_info.get("output_files", [])
                        for out_file in output_files:
                            if not isinstance(out_file, str):
                                continue
                            # Match file extension
                            if expected_exts and not _path_matches_expected_types(out_file, expected_exts):
                                continue
                            
                            # Special handling for mixtcrpred and nettcr services:
                            # Convert RDS to CSV if needed
                            service_name = tool_params.get("service", "").lower() if tool_params else ""
                            is_mixtcrpred_or_nettcr = service_name in ["mixtcrpred", "nettcr"]
                            is_rds_file = out_file.lower().endswith('.rds')
                            
                            if is_mixtcrpred_or_nettcr and is_rds_file and param_name.lower() in ["input_data", "input_file"]:
                                print(f"  [ParamInfer] Detected RDS file from dependency for {service_name} service parameter {param_name}, converting to CSV")
                                try:
                                    sandbox_id = None
                                    if state.parent_state:
                                        merged_result = getattr(state.parent_state, 'merged_result', None) or {}
                                        sandbox_id = merged_result.get('opensandbox_id')
                                    
                                    csv_path = _auto_convert_rds_to_csv(
                                        rds_path=out_file,
                                        state=state,
                                        sandbox_id=sandbox_id
                                    )
                                    if csv_path:
                                        print(f"  [ParamInfer] Converted RDS to CSV: {csv_path}")
                                        all_params[param_name] = csv_path
                                        param_sources[param_name] = "dependency_task_output"
                                        # Add to parameter table (same logic as PRIORITY 3)
                                        if state.parent_state:
                                            csv_name = _Path(csv_path).name
                                            file_key = f"rds_to_csv_{csv_name}"
                                            if not hasattr(state.parent_state, 'extracted_parameters') or not state.parent_state.extracted_parameters:
                                                state.parent_state.extracted_parameters = {"files": {}, "sandbox_file_paths": {}, "task_outputs": {}}
                                            extracted_params = state.parent_state.extracted_parameters
                                            files = extracted_params.get("files", {})
                                            sandbox_file_paths = extracted_params.get("sandbox_file_paths", {})
                                            files[file_key] = {
                                                "original_path": csv_path,
                                                "sandbox_path": csv_path,
                                                "file_type": "csv",
                                                "description": f"CSV file converted from RDS: {out_file}",
                                                "source": "rds_conversion",
                                                "source_rds": out_file
                                            }
                                            sandbox_file_paths[file_key] = csv_path
                                            extracted_params["files"] = files
                                            extracted_params["sandbox_file_paths"] = sandbox_file_paths
                                        break
                                    else:
                                        print(f"  [ParamInfer] RDS to CSV conversion failed, using original RDS")
                                        all_params[param_name] = out_file
                                        param_sources[param_name] = "dependency_task_output"
                                        break
                                except Exception as conv_error:
                                    print(f"  [ParamInfer] RDS to CSV conversion error: {conv_error}, using original RDS")
                                    all_params[param_name] = out_file
                                    param_sources[param_name] = "dependency_task_output"
                                    break
                            else:
                                # Found matching output file from dependency
                                all_params[param_name] = out_file
                                param_sources[param_name] = "dependency_task_output"
                                print(f"    [DEBUG] {param_name} resolved from task_outputs[{dep_task_id}]: {out_file}")
                                break
                        if param_name in all_params:
                            break
                    
                    if param_name in all_params:
                        continue

                # PRIORITY 1: Try to resolve from dependency outputs first
                # This is critical because task decomposition may use placeholder names
                # but actual file paths come from completed dependency tasks
                # IMPORTANT: Skip output parameters - they should be auto-generated, not resolved from dependencies
                if not _is_output_param(param_name, param_type):
                    dep_value = _resolve_param_from_dependencies(
                        task=task,
                        tool_name=tool_name,
                        param_name=param_name,
                        param_type=param_type,
                        param_desc=param_desc,
                        state=state,
                        llm=llm,
                    )
                    if dep_value is not None and _value_matches_expected_types(param_name, param_type, dep_value):
                        # Special handling for mixtcrpred and nettcr services:
                        # Convert RDS to CSV if needed
                        service_name = tool_params.get("service", "").lower() if tool_params else ""
                        is_mixtcrpred_or_nettcr = service_name in ["mixtcrpred", "nettcr"]
                        is_rds_file = isinstance(dep_value, str) and dep_value.lower().endswith('.rds')
                        
                        if is_mixtcrpred_or_nettcr and is_rds_file and param_name.lower() in ["input_data", "input_file"]:
                            print(f"  [ParamInfer] Detected RDS file from dependency resolution for {service_name} service parameter {param_name}, converting to CSV")
                            try:
                                sandbox_id = None
                                if state.parent_state:
                                    merged_result = getattr(state.parent_state, 'merged_result', None) or {}
                                    sandbox_id = merged_result.get('opensandbox_id')
                                
                                csv_path = _auto_convert_rds_to_csv(
                                    rds_path=dep_value,
                                    state=state,
                                    sandbox_id=sandbox_id
                                )
                                if csv_path:
                                    print(f"  [ParamInfer] Converted RDS to CSV: {csv_path}")
                                    all_params[param_name] = _convert_to_sandbox_path(csv_path)
                                    param_sources[param_name] = "dependency"
                                    # Add to parameter table
                                    if state.parent_state:
                                        csv_name = _Path(csv_path).name
                                        file_key = f"rds_to_csv_{csv_name}"
                                        if not hasattr(state.parent_state, 'extracted_parameters') or not state.parent_state.extracted_parameters:
                                            state.parent_state.extracted_parameters = {"files": {}, "sandbox_file_paths": {}, "task_outputs": {}}
                                        extracted_params = state.parent_state.extracted_parameters
                                        files = extracted_params.get("files", {})
                                        sandbox_file_paths = extracted_params.get("sandbox_file_paths", {})
                                        files[file_key] = {
                                            "original_path": csv_path,
                                            "sandbox_path": csv_path,
                                            "file_type": "csv",
                                            "description": f"CSV file converted from RDS: {dep_value}",
                                            "source": "rds_conversion",
                                            "source_rds": dep_value
                                        }
                                        sandbox_file_paths[file_key] = csv_path
                                        extracted_params["files"] = files
                                        extracted_params["sandbox_file_paths"] = sandbox_file_paths
                                    continue
                                else:
                                    print(f"  [ParamInfer] RDS to CSV conversion failed, using original RDS")
                            except Exception as conv_error:
                                print(f"  [ParamInfer] RDS to CSV conversion error: {conv_error}, using original RDS")
                        
                        # Convert to sandbox path if needed
                        all_params[param_name] = _convert_to_sandbox_path(dep_value)
                        param_sources[param_name] = "dependency"
                        print(f"    [DEBUG] {param_name} resolved from dependency: {dep_value}")
                        continue

                # PRIORITY 2: Check task-specified parameters (from decomposition or plan)
                # Only use if dependency resolution failed and value is a valid path
                if param_name in tool_scoped_params or param_name in provided_params:
                    raw_value = tool_scoped_params.get(param_name, provided_params.get(param_name))
                    print(f"    [DEBUG] {param_name} found in provided_params: {raw_value}")
                    if isinstance(raw_value, dict) and "__value" in raw_value:
                        raw_value = raw_value.get("__value")
                    normalized = _normalize_inferred_param_value(param_name, param_type, raw_value)
                    print(f"    [DEBUG] {param_name} normalized: {normalized}")
                    if normalized is not None:
                        expected_exts = _expected_file_extensions(param_name, param_type, param_desc)
                        is_excel = False
                        is_rds = False
                        if isinstance(normalized, str):
                            _, actual_ext = os.path.splitext(normalized)
                            actual_ext = actual_ext.lower().lstrip(".")
                            is_excel = actual_ext in {"xls", "xlsx"}
                            is_rds = actual_ext == "rds"
                        allow_excel_for_csv = (
                            expected_exts is not None
                            and expected_exts.issubset({"csv", "tsv"})
                            and is_excel
                        )
                        # Allow RDS files for CSV parameters - will be converted in preprocessing
                        allow_rds_for_csv = (
                            expected_exts is not None
                            and expected_exts.issubset({"csv", "tsv"})
                            and is_rds
                        )
                        
                        # Special handling for mixtcrpred and nettcr services:
                        # Convert RDS to CSV immediately
                        service_name = tool_params.get("service", "").lower() if tool_params else ""
                        is_mixtcrpred_or_nettcr = service_name in ["mixtcrpred", "nettcr"]
                        
                        if is_mixtcrpred_or_nettcr and is_rds and param_name.lower() in ["input_data", "input_file"]:
                            print(f"  [ParamInfer] Detected RDS file from task parameters for {service_name} service parameter {param_name}, converting to CSV")
                            try:
                                sandbox_id = None
                                if state.parent_state:
                                    merged_result = getattr(state.parent_state, 'merged_result', None) or {}
                                    sandbox_id = merged_result.get('opensandbox_id')
                                
                                csv_path = _auto_convert_rds_to_csv(
                                    rds_path=normalized,
                                    state=state,
                                    sandbox_id=sandbox_id
                                )
                                if csv_path:
                                    print(f"  [ParamInfer] Converted RDS to CSV: {csv_path}")
                                    normalized = csv_path
                                    # Add to parameter table
                                    if state.parent_state:
                                        csv_name = _Path(csv_path).name
                                        file_key = f"rds_to_csv_{csv_name}"
                                        if not hasattr(state.parent_state, 'extracted_parameters') or not state.parent_state.extracted_parameters:
                                            state.parent_state.extracted_parameters = {"files": {}, "sandbox_file_paths": {}, "task_outputs": {}}
                                        extracted_params = state.parent_state.extracted_parameters
                                        files = extracted_params.get("files", {})
                                        sandbox_file_paths = extracted_params.get("sandbox_file_paths", {})
                                        files[file_key] = {
                                            "original_path": csv_path,
                                            "sandbox_path": csv_path,
                                            "file_type": "csv",
                                            "description": f"CSV file converted from RDS: {normalized}",
                                            "source": "rds_conversion",
                                            "source_rds": normalized
                                        }
                                        sandbox_file_paths[file_key] = csv_path
                                        extracted_params["files"] = files
                                        extracted_params["sandbox_file_paths"] = sandbox_file_paths
                                else:
                                    print(f"  [ParamInfer] RDS to CSV conversion failed, keeping original RDS path")
                            except Exception as conv_error:
                                print(f"  [ParamInfer] RDS to CSV conversion error: {conv_error}")
                        
                        type_matches = _value_matches_expected_types(param_name, param_type, normalized)
                        print(f"    [DEBUG] {param_name} type_matches={type_matches}, expected_exts={expected_exts}, allow_excel={allow_excel_for_csv}, allow_rds={allow_rds_for_csv}")
                        if type_matches or allow_excel_for_csv or allow_rds_for_csv:
                            # Convert to sandbox path if needed
                            final_value = _convert_to_sandbox_path(_normalize_base_dir_value(param_name, normalized))
                            all_params[param_name] = final_value
                            param_sources[param_name] = "task_parameters"
                            print(f"    [DEBUG] {param_name} successfully added to all_params: {final_value}")
                            continue
                        else:
                            # Type validation failed - don't mark as missing yet, try other sources
                            print(f"    [DEBUG] {param_name} from provided_params failed type validation, trying other sources")
                    else:
                        print(f"    [DEBUG] {param_name} normalization returned None, trying other sources")

                # PRIORITY 3: Check extracted params from context
                # CRITICAL: All parameter values must come from parameter table
                # Values extracted from context must be validated against parameter table
                if param_name in extracted_params:
                    raw_value = extracted_params[param_name]
                    explicit_match = False
                    explicit_label = ""
                    if isinstance(raw_value, dict) and "__value" in raw_value:
                        explicit_match = bool(raw_value.get("__explicit"))
                        explicit_label = raw_value.get("__label") or ""
                        raw_value = raw_value.get("__value")
                    
                    # Validate that the extracted value is legitimate
                    # Reject values that look like DOI, paper IDs, or other non-parameter values
                    if isinstance(raw_value, str):
                        # Ensure re module is available (avoid scope issues)
                        import re as re_module
                        # Check if it looks like a DOI or paper ID (e.g., "10.1074/jbc.RA123.045678")
                        if re_module.match(r'^\d+\.\d+/[a-zA-Z0-9._-]+$', raw_value) or re_module.match(r'^[a-zA-Z0-9._-]+/\d+$', raw_value):
                            print(f"    [ParamValidate] Rejecting invalid parameter value (looks like DOI/paper ID): {raw_value}")
                            # CRITICAL FIX: Must add to missing_params if not optional
                            if not is_optional:
                                missing_params.append(f"{tool_name}.{param_name}")
                                print(f"    [ParamValidate] Added {param_name} to missing_params (invalid value rejected)")
                            continue
                        # For file parameters, must validate against parameter table
                        if _is_file_type(param_name, param_type):
                            # Check if file path exists in parameter table
                            is_in_param_table = _validate_file_in_parameter_table(
                                file_path=raw_value,
                                state=state,
                                param_name=param_name
                            )
                            if not is_in_param_table:
                                print(f"    [ParamValidate] Rejecting file parameter value not in parameter table: {raw_value}")
                                # CRITICAL FIX: Must add to missing_params if not optional
                                # Previously this was just 'continue' which skipped adding to missing_params
                                if not is_optional:
                                    missing_params.append(f"{tool_name}.{param_name}")
                                    print(f"    [ParamValidate] Added {param_name} to missing_params (required file param rejected)")
                                continue
                    
                    normalized = _normalize_inferred_param_value(param_name, param_type, raw_value)
                    if normalized is not None:
                        if _is_output_param(param_name, param_type):
                            if explicit_match:
                                if not _label_indicates_output(explicit_label):
                                    if not is_optional:
                                        missing_params.append(f"{tool_name}.{param_name}")
                                    continue
                            else:
                                context_text = " ".join([user_input or "", execution_plan or ""]).lower()
                                if isinstance(normalized, str) and normalized.lower() in context_text:
                                    if not is_optional:
                                        missing_params.append(f"{tool_name}.{param_name}")
                                    continue
                        expected_exts = _expected_file_extensions(param_name, param_type, param_desc)
                        is_excel = False
                        is_rds = False
                        if isinstance(normalized, str):
                            _, actual_ext = os.path.splitext(normalized)
                            actual_ext = actual_ext.lower().lstrip(".")
                            is_excel = actual_ext in {"xls", "xlsx"}
                            is_rds = actual_ext == "rds"
                        allow_excel_for_csv = (
                            explicit_match
                            and expected_exts is not None
                            and expected_exts.issubset({"csv", "tsv"})
                            and is_excel
                        )
                        # Allow RDS files for CSV parameters - will be converted in preprocessing
                        allow_rds_for_csv = (
                            expected_exts is not None
                            and expected_exts.issubset({"csv", "tsv"})
                            and is_rds
                        )
                        
                        # Special handling for mixtcrpred and nettcr services:
                        # These services require CSV files, so convert RDS to CSV immediately
                        service_name = tool_params.get("service", "").lower() if tool_params else ""
                        is_mixtcrpred_or_nettcr = service_name in ["mixtcrpred", "nettcr"]
                        
                        if is_mixtcrpred_or_nettcr and is_rds and param_name.lower() in ["input_data", "input_file"]:
                            # Convert RDS to CSV for mixtcrpred/nettcr services
                            print(f"  [ParamInfer] Detected RDS file for {service_name} service parameter {param_name}, converting to CSV")
                            try:
                                # Get sandbox_id from state
                                sandbox_id = None
                                if state.parent_state:
                                    merged_result = getattr(state.parent_state, 'merged_result', None) or {}
                                    sandbox_id = merged_result.get('opensandbox_id')
                                
                                csv_path = _auto_convert_rds_to_csv(
                                    rds_path=normalized,
                                    state=state,
                                    sandbox_id=sandbox_id
                                )
                                if csv_path:
                                    print(f"  [ParamInfer] Converted RDS to CSV: {csv_path}")
                                    normalized = csv_path
                                    # Update to use CSV path
                                    final_value = _convert_to_sandbox_path(_normalize_base_dir_value(param_name, normalized))
                                    all_params[param_name] = final_value
                                    param_sources[param_name] = "explicit" if explicit_match else "llm_context"
                                    
                                    # Add converted CSV to parameter table
                                    if state.parent_state:
                                        csv_name = _Path(csv_path).name
                                        file_key = f"rds_to_csv_{csv_name}"
                                        
                                        if not hasattr(state.parent_state, 'extracted_parameters') or not state.parent_state.extracted_parameters:
                                            state.parent_state.extracted_parameters = {
                                                "files": {},
                                                "sandbox_file_paths": {},
                                                "task_outputs": {}
                                            }
                                        
                                        extracted_params = state.parent_state.extracted_parameters
                                        files = extracted_params.get("files", {})
                                        sandbox_file_paths = extracted_params.get("sandbox_file_paths", {})
                                        
                                        files[file_key] = {
                                            "original_path": csv_path,
                                            "sandbox_path": csv_path,
                                            "file_type": "csv",
                                            "description": f"CSV file converted from RDS: {normalized}",
                                            "source": "rds_conversion",
                                            "source_rds": normalized
                                        }
                                        
                                        sandbox_file_paths[file_key] = csv_path
                                        extracted_params["files"] = files
                                        extracted_params["sandbox_file_paths"] = sandbox_file_paths
                                        print(f"  [ParamInfer] Added converted CSV to parameter table: {csv_path}")
                                    continue
                                else:
                                    print(f"  [ParamInfer] RDS to CSV conversion failed, keeping original RDS path")
                            except Exception as conv_error:
                                print(f"  [ParamInfer] RDS to CSV conversion error: {conv_error}")
                                # Fall through to allow RDS (will be converted in preprocessing)
                        
                        if _value_matches_expected_types(param_name, param_type, normalized) or allow_excel_for_csv or allow_rds_for_csv:
                            # Convert to sandbox path if needed
                            final_value = _convert_to_sandbox_path(_normalize_base_dir_value(param_name, normalized))
                            all_params[param_name] = final_value
                            param_sources[param_name] = "explicit" if explicit_match else "llm_context"
                            continue
                    if not is_optional:
                        missing_params.append(f"{tool_name}.{param_name}")
                    continue

                if _is_file_type(param_name, param_type) and not _is_output_param(param_name, param_type):
                    candidate = _select_file_candidate(param_name, param_type, file_candidates, param_desc)
                    if candidate:
                        signature_issue = _validate_file_signature(param_name, param_type, candidate, param_desc)
                        if not signature_issue:
                            # Convert to sandbox path if needed
                            final_value = _convert_to_sandbox_path(_normalize_base_dir_value(param_name, candidate))
                            all_params[param_name] = final_value
                            param_sources[param_name] = "file_context"
                            continue
                    if not is_optional:
                        missing_params.append(f"{tool_name}.{param_name}")
                        # Diagnostic: explain why available files were not matched
                        _print_param_match_diagnostic(
                            param_name=param_name,
                            param_type=param_type,
                            tool_name=tool_name,
                            available_files=available_files_from_preprocess,
                            provided_params=provided_params,
                            reason="file_type_mismatch_or_no_candidate"
                        )
                    continue
                
                # If non-file type and has recommended value, use it (with type conversion)
                if param_deme and not _is_file_type(param_name, param_type) and not param_name.endswith("_fields"):
                    # Convert demo value to correct type
                    converted_demo = _normalize_inferred_param_value(param_name, param_type, param_deme, verbose=False)
                    if converted_demo is not None:
                        all_params[param_name] = converted_demo
                        param_sources[param_name] = "demo"
                    else:
                        all_params[param_name] = param_deme
                        param_sources[param_name] = "demo"
                    continue
                
                # CRITICAL CONSTRAINT: Input file parameters can ONLY come from:
                # 1. User input (via preprocessing, already handled earlier)
                # 2. Dependency task outputs (already checked earlier)
                # DO NOT allow LLM to guess/infer file paths - this caused errors like "/J"
                is_input_file_param = _is_file_type(param_name, param_type) and not _is_output_param(param_name, param_type)
                if is_input_file_param and param_name not in all_params:
                    # This is an INPUT file parameter that wasn't found in:
                    # - provided_params (user input/preprocessing)
                    # - dependency task outputs
                    # We MUST NOT let LLM guess a file path - mark as missing
                    print(f"  [ParamInfer] ⚠ Input file parameter '{param_name}' not found in user input or dependency outputs. Skipping LLM inference.")
                    if not is_optional:
                        missing_params.append(f"{tool_name}.{param_name}")
                        _print_param_match_diagnostic(
                            param_name=param_name,
                            param_type=param_type,
                            tool_name=tool_name,
                            available_files=preprocess_files,
                            provided_params=provided_params,
                            reason="input_file_requires_user_or_dependency"
                        )
                    continue
                
                # Use LLM to infer parameter value
                if llm and param_name not in all_params:
                    try:
                        from langchain_core.messages import SystemMessage, HumanMessage
                        
                        inference_prompt = f"""
Please infer parameter value based on the following information:

Task description: {task.content}
User input: {user_input}
Execution plan: {execution_plan or 'None'}
Tool name: {tool_name}
Parameter name: {param_name}
Parameter type: {param_type}
Parameter description: {param_desc}
Task input list: {inputs}
Dependency outputs: {[state.task_results.get(dep_id).output for dep_id in task.dependencies if dep_id in state.task_results]}

**CRITICAL TYPE VALIDATION RULES:**
1. If parameter type contains "int" or "integer": value MUST be a number (e.g., 60, 120), NOT a string or tool name
2. If parameter type contains "float": value MUST be a decimal number (e.g., 0.5, 1.0)
3. If parameter type contains "bool": value MUST be true or false
4. If parameter type contains "enum": value MUST be one of the allowed options
5. If parameter type contains "file" or "path": value MUST be a valid file path starting with "/" or a drive letter
6. NEVER use tool names, service names, or unrelated text as parameter values
7. For timeout/duration parameters: use reasonable default values (e.g., 60, 120, 300 seconds)

Please infer the value of this parameter. If it cannot be inferred from the task description, or requires user input (such as file paths, user selections, etc.), return null.

Return JSON format:
{{
    "value": <parameter value matching the expected type, or null>,
    "can_infer": <true/false>,
    "reason": "<inference reason or why user input is required>"
}}
"""
                        messages = [
                            SystemMessage(content="You are a professional parameter inference expert. You MUST strictly validate parameter types - never assign string values to integer parameters, never use tool names as parameter values."),
                            HumanMessage(content=inference_prompt)
                        ]
                        
                        response = llm.invoke(messages)
                        response_text = response.content.strip()
                        
                        # Parse response
                        import re
                        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                        if json_match:
                            # Ensure using global json module
                            import json as json_module
                            inference_data = json_module.loads(json_match.group())
                            param_value = inference_data.get("value")
                            can_infer = inference_data.get("can_infer", False)
                            
                            if param_value is not None and can_infer:
                                normalized = _normalize_inferred_param_value(param_name, param_type, param_value)
                                if normalized is not None and _value_matches_expected_types(param_name, param_type, normalized):
                                    # Convert to sandbox path if needed
                                    final_value = _convert_to_sandbox_path(_normalize_base_dir_value(param_name, normalized))
                                    all_params[param_name] = final_value
                                    param_sources[param_name] = "llm"
                                elif not is_optional:
                                    missing_params.append(f"{tool_name}.{param_name}")
                            elif not is_optional:
                                missing_params.append(f"{tool_name}.{param_name}")
                        else:
                            if not is_optional:
                                missing_params.append(f"{tool_name}.{param_name}")
                    except Exception as e:
                        print(f"  ⚠ Failed to infer parameter {param_name}: {e}")
                        if not is_optional:
                            missing_params.append(f"{tool_name}.{param_name}")
                elif not is_optional and param_name not in all_params:
                    missing_params.append(f"{tool_name}.{param_name}")

                # Guard: if CSV/TSV is required but we only have Excel and no pandas, require HITL.
                if param_name in all_params:
                    expected_ext = _infer_expected_extension(param_type)
                    if expected_ext in {"csv", "tsv"}:
                        value = all_params.get(param_name)
                        if isinstance(value, str):
                            _, actual_ext = os.path.splitext(value)
                            actual_ext = actual_ext.lower().lstrip(".")
                            if actual_ext in {"xls", "xlsx"} and not _has_pandas() and not is_optional:
                                all_params.pop(param_name, None)
                                missing_params.append(f"{tool_name}.{param_name}")
                                print(f"  ⚠ {tool_name}.{param_name} expects {expected_ext}, but only Excel provided and pandas is unavailable.")
                if param_name in all_params:
                    signature_issue = _validate_file_signature(param_name, param_type, all_params.get(param_name), param_desc)
                    if signature_issue:
                        all_params.pop(param_name, None)
                        missing_params.append(f"{tool_name}.{param_name}")
                        print(f"  ⚠ {tool_name}.{param_name} failed file signature check: {signature_issue}")

            if _tool_requires_matched_fields(input_params):
                # Validate and align csv_fields and rds_fields (ensure they match)
                # Note: For integrate_bcr_data_complete and integrate_tcr_data_complete, 
                # csv_fields and rds_fields are already auto-filled BEFORE the parameter loop,
                # so this just ensures they match
                _validate_matched_fields(tool_name, all_params, missing_params)
                
                # Ensure these parameters are not in missing_params (they were auto-filled)
                if tool_name in ("integrate_bcr_data_complete", "integrate_tcr_data_complete"):
                    if "csv_fields" in all_params:
                        missing_params[:] = [p for p in missing_params if not p.endswith(".csv_fields")]
                    if "rds_fields" in all_params:
                        missing_params[:] = [p for p in missing_params if not p.endswith(".rds_fields")]
                csv_source = param_sources.get("csv_fields")
                rds_source = param_sources.get("rds_fields")
                # Accept auto_inferred as valid source along with explicit and dependency
                valid_sources = {"explicit", "dependency", "auto_inferred"}
                if csv_source not in valid_sources or rds_source not in valid_sources:
                    if "csv_fields" in all_params:
                        all_params.pop("csv_fields", None)
                    if "rds_fields" in all_params:
                        all_params.pop("rds_fields", None)
                    missing_params.append(f"{tool_name}.csv_fields")
                    missing_params.append(f"{tool_name}.rds_fields")

            _validate_integrate_bcr_inputs(tool_name, all_params, missing_params)
            
            # Auto-generate output file paths for output parameters
            # This prevents HITL for output_file type parameters that can be auto-generated
            # Note: sandbox_data_dir is already initialized at function scope above
            
            # Special handling for integrate_bcr_data_complete: rds_file and output_file
            # Rules:
            # - rds_file: Use the latest integrated_n.rds if exists, otherwise use original RDS
            # - output_file: Use integrated_{max_n+1}.rds, or integrated_1.rds if none exists
            if tool_name == "integrate_bcr_data_complete" and sandbox_data_dir:
                import re
                task_outputs = state.parent_state.extracted_parameters.get("task_outputs", {}) if state.parent_state else {}
                
                # Find all integrated_n.rds files from task_outputs
                integrated_files = []  # List of (n, path) tuples
                for task_out in task_outputs.values():
                    if isinstance(task_out, dict):
                        for out_file in task_out.get("output_files", []):
                            if isinstance(out_file, str) and "integrated_" in out_file and out_file.endswith(".rds"):
                                match = re.search(r'integrated_(\d+)\.rds', out_file)
                                if match:
                                    n = int(match.group(1))
                                    integrated_files.append((n, out_file))
                
                # Sort by n to find the latest
                integrated_files.sort(key=lambda x: x[0], reverse=True)
                max_n = integrated_files[0][0] if integrated_files else 0
                latest_integrated = integrated_files[0][1] if integrated_files else None
                
                print(f"  [ParamInfer] integrate_bcr_data_complete: found {len(integrated_files)} integrated files, max_n={max_n}")
                
                # Set rds_file: use latest integrated_n.rds if exists, otherwise use original RDS
                # IMPORTANT: Always use full absolute path for rds_file
                current_rds = all_params.get("rds_file", "")
                
                if latest_integrated:
                    # We have integrated files - use the latest one with full path
                    all_params["rds_file"] = latest_integrated
                    param_sources["rds_file"] = "auto_integrated"
                    missing_params[:] = [p for p in missing_params if not p.endswith(".rds_file")]
                    print(f"  [ParamInfer] rds_file={latest_integrated} (latest integrated, full path)")
                elif current_rds and not current_rds.startswith("/"):
                    # Current rds_file is a relative path - try to find full path
                    # Check if it matches integrated_n.rds pattern
                    match = re.search(r'integrated_(\d+)\.rds', current_rds)
                    if match:
                        n = int(match.group(1))
                        full_path = f"{sandbox_data_dir}/output/integrated_{n}.rds"
                        all_params["rds_file"] = full_path
                        param_sources["rds_file"] = "auto_integrated"
                        print(f"  [ParamInfer] rds_file converted to full path: {full_path}")
                # If no integrated file and no relative path issue, keep the original rds_file
                
                # Set output_file: always use integrated_{max_n+1}.rds
                next_n = max_n + 1
                auto_output = f"{sandbox_data_dir}/output/integrated_{next_n}.rds"
                all_params["output_file"] = auto_output
                param_sources["output_file"] = "auto_generated"
                missing_params[:] = [p for p in missing_params if not p.endswith(".output_file")]
                print(f"  [ParamInfer] output_file={auto_output} (next_n={next_n})")
            
            if sandbox_data_dir:
                for param in input_params:
                    p_name = param.get("name", "")
                    p_type = param.get("type", "")
                    p_desc = param.get("description", "").lower()
                    
                    # Check if this is an output parameter
                    if _is_output_param(p_name, p_type):
                        # Skip integrate_bcr_data_complete output_file - already handled above
                        if tool_name == "integrate_bcr_data_complete" and p_name == "output_file":
                            continue
                        # Special handling for metabcr output_file_path
                        elif tool_name == "metabcr" and p_name == "output_file_path":
                            # MetaBcr expects a directory path where it will create MetaBcr/bind/ subdirectory
                            auto_output = f"{sandbox_data_dir}/output"
                            all_params[p_name] = auto_output
                            param_sources[p_name] = "auto_generated"
                            missing_params[:] = [p for p in missing_params if not p.endswith(f".{p_name}")]
                            print(f"  [ParamInfer] Auto-generated {p_name}={auto_output} for {tool_name}")
                            # Note: MetaBcr output file name is dynamic (based on input file names)
                            # The actual output path will be extracted from MCP response after execution
                        # Special handling for integrate_tcr_data_complete output_path - should output RDS file
                        # Use FIXED filename "integrated_tcr_data.rds" so downstream tools can find it
                        elif tool_name == "integrate_tcr_data_complete" and p_name == "output_path":
                            auto_output = f"{sandbox_data_dir}/output/integrated_tcr_data.rds"
                            all_params[p_name] = auto_output
                            param_sources[p_name] = "auto_generated"
                            missing_params[:] = [p for p in missing_params if not p.endswith(f".{p_name}")]
                            print(f"  [ParamInfer] Auto-generated {p_name}={auto_output} for {tool_name} (fixed RDS output for downstream tools)")
                        else:
                            # For other output parameters, check if already valid before replacing
                            existing_value = all_params.get(p_name)
                            if existing_value and isinstance(existing_value, str):
                                if existing_value.startswith(sandbox_data_dir):
                                    continue  # Already a valid sandbox path
                                elif existing_value.startswith("/data/") or existing_value.startswith("/tmp/"):
                                    continue  # Already an absolute path
                            
                            # Generic output file - generate based on tool name, param type, and description
                            ext = "csv"  # default extension
                            if "rds" in p_type.lower() or "rds" in p_desc:
                                ext = "rds"
                            elif "fasta" in p_type.lower() or "fasta" in p_desc:
                                ext = "fasta"
                            elif "json" in p_type.lower() or "json" in p_desc:
                                ext = "json"
                            
                            auto_output = f"{sandbox_data_dir}/output/{tool_name}_{task.task_id}_output.{ext}"
                            all_params[p_name] = auto_output
                            param_sources[p_name] = "auto_generated"
                            missing_params[:] = [p for p in missing_params if not p.endswith(f".{p_name}")]
                            print(f"  [ParamInfer] Auto-generated {p_name}={auto_output} for {tool_name}")
        
        # Final check: Ensure all integrated_n.rds references use full absolute paths
        # This applies to ALL tools, not just integrate_bcr_data_complete
        if sandbox_data_dir:
            import re as re_module
            for param_name, param_value in list(all_params.items()):
                if isinstance(param_value, str):
                    # Check if it's a relative integrated_n.rds path
                    if not param_value.startswith("/") and re_module.search(r'integrated_\d+\.rds', param_value):
                        match = re_module.search(r'integrated_(\d+)\.rds', param_value)
                        if match:
                            n = int(match.group(1))
                            full_path = f"{sandbox_data_dir}/output/integrated_{n}.rds"
                            all_params[param_name] = full_path
                            print(f"  [ParamInfer] Converted relative path {param_name}={param_value} to {full_path}")
        
        # Update task result
        result.parameters = all_params
        # Log duplicate missing parameters for diagnosis
        if missing_params:
            dup_counts: Dict[str, int] = {}
            for item in missing_params:
                dup_counts[item] = dup_counts.get(item, 0) + 1
            duplicates = {k: v for k, v in dup_counts.items() if v > 1}
            if duplicates:
                tool_names = []
                for tool_item in tools:
                    if isinstance(tool_item, str):
                        tool_names.append(tool_item)
                    elif isinstance(tool_item, dict):
                        tool_names.append(tool_item.get("tool_name") or tool_item.get("name", ""))
                print(f"  ⚠ Duplicate missing parameters detected for task {task.task_id}: {duplicates}")
                print(f"     Tools list: {tool_names}")
        # Deduplicate missing parameters while preserving order
        seen_missing = set()
        deduped_missing = []
        for item in missing_params:
            if item in seen_missing:
                continue
            seen_missing.add(item)
            deduped_missing.append(item)
        result.missing_parameters = deduped_missing
        
        # If there are missing parameters, trigger HITL
        if result.missing_parameters:
            # Include inferred parameters for user reference
            inferred_params_info = {}
            for param_name, param_value in all_params.items():
                source = param_sources.get(param_name, "unknown")
                inferred_params_info[param_name] = {
                    "value": param_value,
                    "source": source
                }
            
            _record_hitl_request(state, task.task_id, {
                "type": "missing_parameters",
                "task_id": task.task_id,
                "task_description": task.content,
                "missing_parameters": result.missing_parameters,
                "inferred_parameters": inferred_params_info,  # Show what was already inferred
                "message": f"Task {task.task_id} requires the following parameters, please provide: {', '.join(result.missing_parameters)}"
            })
            # Important: Set task status to WAITING_HITL_PARAMS, not READY
            # This prevents execute_tasks_node from trying to execute these tasks
            state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_HITL_PARAMS
            print(f"  ⚠ Task {task.task_id} requires user-provided parameters: {', '.join(result.missing_parameters)}")
            print(f"  → Task status set to WAITING_HITL_PARAMS")
        else:
            # All parameters are ready, mark task as READY for execution
            if task.task_id in state.hitl_requests:
                # Remove HITL request if all parameters are now available
                del state.hitl_requests[task.task_id]
            state.task_status_map[task.task_id] = ExecutorTaskStatus.READY
            print(f"  ✓ Task {task.task_id} has all required parameters ({len(all_params)} parameters), marked as READY for execution")
    
    return state


def check_hitl_params_node(state: ExecutorState) -> Literal["hitl_params", "execute"]:
    """Check if HITL parameters are needed"""
    pending_hitl = [
        task_id for task_id, request in state.hitl_requests.items()
        if request.get("type") == "missing_parameters" and task_id not in state.hitl_responses
    ]
    
    print(f"  🔍 [check_hitl_params] HITL requests: {len(state.hitl_requests)}, pending: {len(pending_hitl)}")
    if pending_hitl:
        print(f"  → Routing to hitl_params node")
        return "hitl_params"
    else:
        print(f"  → Routing to execute node")
        return "execute"


def hitl_params_node(state: ExecutorState) -> ExecutorState:
    """
    HITL parameter request node
    
    Request parameters from user and wait for response (using interrupt)
    Supports receiving resume value when resuming from interrupt
    
    Workflow:
    1. First execution: Check if there are unresponded HITL requests, if so trigger interrupt
    2. Resume execution: Get user response from interrupt's resume value, update parameters, continue execution
    """
    print(f"  🔍 [hitl_params_node] Node called")
    
    # If parent_state doesn't exist, try to get from module variable
    _restore_parent_state(state)
    
    # Also update parent_state.hitl_status from thread-scoped storage if available
    if state.parent_state and state.thread_id and state.thread_id in _parent_state_by_thread:
        parent_state = _parent_state_by_thread[state.thread_id]
        if parent_state.hitl_status and (not state.parent_state.hitl_status or state.parent_state.hitl_status != parent_state.hitl_status):
            print(f"  [DEBUG] Updating parent_state.hitl_status from thread-scoped storage")
            state.parent_state.hitl_status = parent_state.hitl_status
    
    # Try to get resume value (if resuming execution)
    # Note: When resuming execution, interrupt() returns the value from Command(resume=...)
    # On first call, interrupt() raises an exception (this is normal)
    resume_value = None
    if INTERRUPT_AVAILABLE:
        try:
            # First try to get resume value (no parameters)
            # If resuming execution, this returns resume value
            # If first call, this raises exception, we'll handle it later
            print(f"  🔍 [hitl_params_node] Attempting to get resume value...")
            resume_value = interrupt()
            print(f"  ✓ [hitl_params_node] Got resume value: {resume_value}")
        except Exception as e:
            # On first call, interrupt() raises exception (normal behavior)
            # We'll call interrupt(value) again later when we need to interrupt
            print(f"  ✓ [hitl_params_node] interrupt() raised exception (normal on first call): {type(e).__name__}")
            resume_value = None
    
    # If there's a resume value, this is resuming from interrupt, process user response
    if resume_value is not None:
        print(f"  [DEBUG] Resume value received: {resume_value}, type: {type(resume_value)}")
        # resume_value may be Command object or dict
        # If Command object, need to extract resume field
        if hasattr(resume_value, 'resume'):
            resume_data = resume_value.resume
            print(f"  [DEBUG] Extracted resume_data from Command.resume: {resume_data}")
        elif isinstance(resume_value, dict) and 'resume' in resume_value:
            resume_data = resume_value['resume']
            print(f"  [DEBUG] Extracted resume_data from dict['resume']: {resume_data}")
        else:
            resume_data = resume_value
            print(f"  [DEBUG] Using resume_value directly as resume_data: {resume_data}")
        
        print(f"  [DEBUG] Final resume_data: {resume_data}, type: {type(resume_data)}")
        if isinstance(resume_data, dict):
            print(f"  [DEBUG] resume_data keys: {resume_data.keys()}, type field: {resume_data.get('type')}")
        
        if isinstance(resume_data, dict) and resume_data.get("type") == "response_parameters":
            responses = resume_data.get("responses", {})
            for task_id, response_data in responses.items():
                if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                    _record_hitl_response(state, task_id, response_data)
                    # Update parameters
                    if task_id in state.task_results:
                        result = state.task_results[task_id]
                        if "parameters" in response_data:
                            # Intelligent parameter mapping: map user-provided parameter names to tool-required parameter names
                            # Example: input_file -> input_file_path
                            param_mapping = {
                                "input_file": "input_file_path",
                                "output_file": "output_file_path",
                                "input_path": "input_file_path",
                                "output_path": "output_file_path",
                            }
                            
                            # Process parameter mapping
                            mapped_parameters = {}
                            skipped_params = set()  # Track skipped parameters (value is None, empty dict, empty string, or "skip" string)
                            for user_param_name, param_value in response_data["parameters"].items():
                                # If parameter value is None, empty dict, empty string, or "skip" string, it means user skipped it
                                is_skipped = (
                                    param_value is None or 
                                    param_value == {} or 
                                    (isinstance(param_value, str) and (
                                        param_value.strip() == "" or 
                                        param_value.strip().lower() == "skip"
                                    )) or
                                    (isinstance(param_value, dict) and len(param_value) == 0)
                                )
                                
                                if is_skipped:
                                    print(f"  [DEBUG] Parameter {user_param_name} was skipped by user (value: {param_value})")
                                    skipped_params.add(user_param_name)
                                    # Also add parameter name without tool prefix if applicable
                                    if '.' in user_param_name:
                                        skipped_params.add(user_param_name.split('.', 1)[1])
                                    continue
                                
                                # Check if mapping is needed
                                # If parameter name contains tool prefix (e.g., "tool_name.param_name"), need separate handling
                                if '.' in user_param_name:
                                    parts = user_param_name.split('.', 1)
                                    tool_prefix = parts[0]
                                    param_name = parts[1]
                                    # Try to map parameter name
                                    mapped_param_name = param_mapping.get(param_name, param_name)
                                    mapped_parameters[f"{tool_prefix}.{mapped_param_name}"] = param_value
                                    # Also keep original parameter name (in case tool accepts it)
                                    mapped_parameters[user_param_name] = param_value
                                else:
                                    # Try to map parameter name
                                    mapped_param_name = param_mapping.get(user_param_name, user_param_name)
                                    mapped_parameters[mapped_param_name] = param_value
                                    # Also keep original parameter name (in case tool accepts it)
                                    mapped_parameters[user_param_name] = param_value
                            
                            # Update parameters (only non-None values)
                            result.parameters.update(mapped_parameters)
                            
                            # Remove provided parameters from missing_parameters
                            # missing_parameters format may be "tool_name.param_name" or "param_name"
                            provided_params = set(mapped_parameters.keys())
                            # Also check original parameter names
                            provided_params.update(response_data["parameters"].keys())
                            
                            # Also remove skipped parameters from missing_parameters
                            # (User explicitly skipped them, so remove from missing list)
                            print(f"  [DEBUG] Skipped parameters: {skipped_params}")
                            print(f"  [DEBUG] Missing parameters before removal: {result.missing_parameters}")
                            
                            for skipped_param in skipped_params:
                                provided_params.add(skipped_param)
                                # Also add with tool prefix if applicable
                                # Extract parameter name from skipped_param (may have tool prefix)
                                skipped_param_name = skipped_param.split('.', 1)[1] if '.' in skipped_param else skipped_param
                                
                                for missing_param in list(result.missing_parameters):
                                    # Extract parameter name from missing_param (may have tool prefix)
                                    missing_param_name = missing_param.split('.', 1)[1] if '.' in missing_param else missing_param
                                    
                                    # Check multiple matching patterns
                                    param_matched = (
                                        missing_param == skipped_param or 
                                        missing_param.endswith(f".{skipped_param}") or
                                        ('.' in missing_param and missing_param.split('.', 1)[1] == skipped_param) or
                                        ('.' in skipped_param and missing_param == skipped_param.split('.', 1)[1]) or
                                        # Match by parameter name only (ignoring tool prefix)
                                        missing_param_name == skipped_param_name or
                                        missing_param.endswith(f".{skipped_param_name}") or
                                        ('.' in skipped_param and missing_param == skipped_param_name)
                                    )
                                    
                                    if param_matched:
                                        provided_params.add(missing_param)
                                        print(f"  [DEBUG] Matched skipped parameter '{skipped_param}' with missing parameter '{missing_param}'")
                            
                            result.missing_parameters = [
                                p for p in result.missing_parameters
                                if not any(
                                    # Exact match
                                    p == param_name or 
                                    # Suffix match (e.g., "tool.input_file_path" matches "input_file_path")
                                    p.endswith(f".{param_name}") or
                                    # Prefix match (e.g., "input_file_path" matches "tool.input_file_path")
                                    ('.' in p and p.split('.', 1)[1] == param_name) or
                                    # Reverse match (e.g., "input_file_path" matches "tool.input_file_path")
                                    ('.' in param_name and p == param_name.split('.', 1)[1]) or
                                    # Parameter name mapping match (e.g., "input_file" matches "input_file_path")
                                    (param_name in param_mapping and (
                                        p == param_mapping[param_name] or
                                        p.endswith(f".{param_mapping[param_name]}")
                                    ))
                                    for param_name in provided_params
                                )
                            ]
                            
                            print(f"  [DEBUG] Missing parameters after removal: {result.missing_parameters}")
                            
                        # If all required parameters are provided (or skipped), mark as ready
                        if not result.missing_parameters:
                            state.task_status_map[task_id] = ExecutorTaskStatus.READY
                            # Mark as responded (even if all parameters were skipped)
                            if task_id not in state.hitl_responses:
                                _record_hitl_response(state, task_id, response_data)
                            # Remove from hitl_requests since all parameters are now handled (provided or skipped)
                            if task_id in state.hitl_requests:
                                del state.hitl_requests[task_id]
                                print(f"  ✓ Task {task_id} removed from hitl_requests (all parameters handled)")
                            print(f"  ✓ Task {task_id} has all required parameters (or skipped optional ones), marked as READY for execution")
                        else:
                            print(f"  [DEBUG] Task {task_id} still has missing parameters: {result.missing_parameters}")
    
    # Also check parent_state.hitl_status as fallback (in case resume_value wasn't passed correctly)
    # This is important because resume_executor_after_interrupt may not correctly pass resume_value through interrupt()
    print(f"  [DEBUG] Checking parent_state.hitl_status: {state.parent_state.hitl_status if (state.parent_state and hasattr(state.parent_state, 'hitl_status')) else 'No parent_state or hitl_status'}")
    if state.parent_state and hasattr(state.parent_state, 'hitl_status') and state.parent_state.hitl_status:
        try:
            # Ensure using global json module
            import json as json_module
            hitl_data = json_module.loads(state.parent_state.hitl_status)
            print(f"  [DEBUG] Parsed hitl_data from parent_state.hitl_status: type={hitl_data.get('type')}, responses={list(hitl_data.get('responses', {}).keys())}")
            if hitl_data.get("type") == "response_parameters":
                # Process user response (fallback: passed through parent_state)
                responses = hitl_data.get("responses", {})
                print(f"  [DEBUG] Processing {len(responses)} responses from parent_state.hitl_status")
                for task_id, response_data in responses.items():
                    print(f"  [DEBUG] Processing response for task {task_id}: {task_id in state.hitl_requests}, already responded: {task_id in state.hitl_responses}")
                    if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                        _record_hitl_response(state, task_id, response_data)
                        # Update parameters
                        if task_id in state.task_results:
                            result = state.task_results[task_id]
                            if "parameters" in response_data:
                                # Use same intelligent parameter mapping logic
                                param_mapping = {
                                    "input_file": "input_file_path",
                                    "output_file": "output_file_path",
                                    "input_path": "input_file_path",
                                    "output_path": "output_file_path",
                                }
                                
                                mapped_parameters = {}
                                skipped_params = set()  # Track skipped parameters (value is None, empty dict, or empty string)
                                for user_param_name, param_value in response_data["parameters"].items():
                                    # If parameter value is None, empty dict, empty string, or "skip" string, it means user skipped it
                                    is_skipped = (
                                        param_value is None or 
                                        param_value == {} or 
                                        (isinstance(param_value, str) and (
                                            param_value.strip() == "" or 
                                            param_value.strip().lower() == "skip"
                                        )) or
                                        (isinstance(param_value, dict) and len(param_value) == 0)
                                    )
                                    
                                    if is_skipped:
                                        print(f"  [DEBUG] Parameter {user_param_name} was skipped by user (value: {param_value})")
                                        skipped_params.add(user_param_name)
                                        # Also add parameter name without tool prefix if applicable
                                        if '.' in user_param_name:
                                            skipped_params.add(user_param_name.split('.', 1)[1])
                                        continue
                                    
                                    if '.' in user_param_name:
                                        parts = user_param_name.split('.', 1)
                                        tool_prefix = parts[0]
                                        param_name = parts[1]
                                        mapped_param_name = param_mapping.get(param_name, param_name)
                                        mapped_parameters[f"{tool_prefix}.{mapped_param_name}"] = param_value
                                        mapped_parameters[user_param_name] = param_value
                                    else:
                                        mapped_param_name = param_mapping.get(user_param_name, user_param_name)
                                        mapped_parameters[mapped_param_name] = param_value
                                        mapped_parameters[user_param_name] = param_value
                                
                                result.parameters.update(mapped_parameters)
                                provided_params = set(mapped_parameters.keys())
                                provided_params.update(response_data["parameters"].keys())
                                
                                # Also remove skipped parameters from missing_parameters
                                print(f"  [DEBUG] Skipped parameters: {skipped_params}")
                                print(f"  [DEBUG] Missing parameters before removal: {result.missing_parameters}")
                                
                                for skipped_param in skipped_params:
                                    provided_params.add(skipped_param)
                                    # Also add with tool prefix if applicable
                                    # Try multiple matching patterns to ensure we catch all variations
                                    for missing_param in list(result.missing_parameters):
                                        # Check multiple matching patterns
                                        param_matched = (
                                            missing_param == skipped_param or 
                                            missing_param.endswith(f".{skipped_param}") or
                                            ('.' in missing_param and missing_param.split('.', 1)[1] == skipped_param) or
                                            ('.' in skipped_param and missing_param == skipped_param.split('.', 1)[1]) or
                                            ('.' in skipped_param and missing_param.endswith(f".{skipped_param.split('.', 1)[1]}"))
                                        )
                                        if param_matched:
                                            provided_params.add(missing_param)
                                            print(f"  [DEBUG] Matched skipped parameter '{skipped_param}' with missing parameter '{missing_param}'")
                                
                                result.missing_parameters = [
                                    p for p in result.missing_parameters
                                    if not any(
                                        p == param_name or 
                                        p.endswith(f".{param_name}") or
                                        ('.' in p and p.split('.', 1)[1] == param_name) or
                                        (param_name in param_mapping and (
                                            p == param_mapping[param_name] or
                                            p.endswith(f".{param_mapping[param_name]}")
                                        ))
                                        for param_name in provided_params
                                    )
                                ]
                                
                                print(f"  [DEBUG] Missing parameters after removal: {result.missing_parameters}")
                                
                            if not result.missing_parameters:
                                state.task_status_map[task_id] = ExecutorTaskStatus.READY
                                # Mark as responded (even if all parameters were skipped)
                                if task_id not in state.hitl_responses:
                                    _record_hitl_response(state, task_id, response_data)
                                # Remove from hitl_requests since all parameters are now handled (provided or skipped)
                                if task_id in state.hitl_requests:
                                    del state.hitl_requests[task_id]
                                    print(f"  ✓ Task {task_id} removed from hitl_requests (all parameters handled)")
                                print(f"  ✓ Task {task_id} has all required parameters (or skipped optional ones), marked as READY for execution")
                            else:
                                print(f"  [WARN] Task {task_id} still has missing parameters after skip: {result.missing_parameters}")
        except Exception as e:
            print(f"  ⚠ Failed to parse HITL response: {e}")
    
    # If there are still unresponded requests, set HITL request info and trigger interrupt
    print(f"  🔍 [hitl_params_node] Checking unresponded requests...")
    remaining_requests = [
        task_id for task_id in state.hitl_requests.keys()
        if task_id not in state.hitl_responses
    ]
    print(f"  🔍 [hitl_params_node] Unresponded requests: {len(remaining_requests)}, request list: {remaining_requests}")
    print(f"  🔍 [hitl_params_node] parent_state exists: {state.parent_state is not None}")
    
    if remaining_requests and state.parent_state:
        print(f"  ✓ [hitl_params_node] Conditions met, starting to build hitl_messages")
        hitl_messages = []
        for task_id in remaining_requests:
            request = state.hitl_requests[task_id]
            print(f"  🔍 [hitl_params_node] Processing request {task_id}: type={request.get('type')}")
            if request.get("type") == "missing_parameters":
                hitl_messages.append({
                    "task_id": task_id,
                    "message": request["message"],
                    "type": request["type"],
                    "missing_parameters": request.get("missing_parameters", []),
                    "inferred_parameters": request.get("inferred_parameters", {})  # Include inferred params for user reference
                })
            else:
                print(f"  ⚠ [hitl_params_node] Request {task_id} type is not missing_parameters: {request.get('type')}")
        
        print(f"  🔍 [hitl_params_node] hitl_messages count: {len(hitl_messages)}")
        
        if hitl_messages:
            import json as json_module
            state.parent_state.hitl_status = json_module.dumps({
                "type": "missing_parameters",
                "requests": hitl_messages
            }, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"HITL Request: User needs to provide parameters")
            print(f"{'='*60}")
            for msg in hitl_messages:
                print(f"Task {msg['task_id']}: {msg['message']}")
            print(f"{'='*60}\n")
            
            # Trigger interrupt, pause execution and wait for user response
            # On first call, interrupt() raises GraphInterrupt exception
            # LangGraph will catch this exception, save state, and return result with __interrupt__ field
            # On resume, caller needs to use Command(resume=...) to pass user response
            if INTERRUPT_AVAILABLE:
                interrupt_value = {
                    "type": "missing_parameters",
                    "requests": hitl_messages,
                    "message": "Waiting for user to provide parameters"
                }
                print(f"  🔔 [hitl_params_node] Preparing to trigger interrupt...")
                print(f"     Interrupt info: {len(hitl_messages)} tasks need parameters")
                print(f"     Interrupt value: {interrupt_value}")
                # interrupt will raise exception (this is normal behavior)
                # LangGraph will catch and save state
                # Exception will propagate upward for LangGraph to handle
                # Note: This exception should not be caught, let it propagate to LangGraph
                try:
                    interrupt(interrupt_value)
                    print(f"  ⚠ [hitl_params_node] interrupt() did not raise exception, this may be abnormal")
                except Exception as interrupt_e:
                    # interrupt should raise exception, this is normal
                    print(f"  ✓ [hitl_params_node] interrupt() raised exception (normal): {type(interrupt_e).__name__}")
                    print(f"     Exception value: {getattr(interrupt_e, 'value', None)}")
                    # Re-raise exception for LangGraph to handle
                    raise
            else:
                # Fallback: use state marker
                print("  ⚠ Note: interrupt functionality unavailable, will use state marker method")
    
    return state


# Note: Resource checking is now done inside execute_tasks_node, no longer needs separate node
# Keep this function as helper (if needed)
def _check_resources_available(state: ExecutorState) -> bool:
    """
    Check if there are enough resources to execute tasks (helper function)
    
    Returns:
        True: Has resources to execute tasks
        False: No resources, need to wait
    """
    # Get all ready tasks (excluding running ones)
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if not ready_tasks:
        return False
    
    # Calculate number of executable tasks (considering parallel limit)
    available_slots = state.max_parallel_tasks - len(state.running_tasks)
    if available_slots <= 0:
        return False
    
    return True


def execute_tasks_node(state: ExecutorState) -> ExecutorState:
    """
    Execute tasks node
    
    1. Get all ready tasks (excluding running ones)
    2. Limit parallel count
    3. Execute tasks in parallel
    4. Update task status
    """
    # Restore parent_state from thread-scoped storage if missing
    _restore_parent_state(state)
    
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if not ready_tasks:
        return state
    
    # Filter out tasks missing REQUIRED parameters (these tasks should wait for HITL response)
    # Note: Optional parameters that are skipped should not prevent execution
    tasks_with_params = []
    llm = create_reasoning_llm()
    tools_params_map = _load_tools_params_table()
    
    for task in ready_tasks:
        task_result = state.task_results.get(task.task_id)
        task_id = task.task_id
        
        # IMPORTANT: If user has responded to HITL and skipped parameters, check if missing parameters were skipped
        if task_result and task_result.missing_parameters and task_id in state.hitl_responses:
            hitl_response = state.hitl_responses[task_id]
            if "parameters" in hitl_response:
                # Check if any missing parameters were skipped by user
                skipped_missing = []
                for missing_param in task_result.missing_parameters:
                    # Extract parameter name (may have tool prefix)
                    param_name = missing_param.split('.', 1)[1] if '.' in missing_param else missing_param
                    # Check if this parameter was in the response (even if value is empty/skipped)
                    for user_param_name, param_value in hitl_response["parameters"].items():
                        user_param_name_only = user_param_name.split('.', 1)[1] if '.' in user_param_name else user_param_name
                        # Check if parameter was skipped (None, empty dict, empty string, or "skip" string)
                        is_skipped = (
                            param_value is None or 
                            param_value == {} or 
                            (isinstance(param_value, str) and (
                                param_value.strip() == "" or 
                                param_value.strip().lower() == "skip"
                            )) or
                            (isinstance(param_value, dict) and len(param_value) == 0)
                        )
                        # Match parameter names (with or without tool prefix)
                        if (param_name == user_param_name_only or 
                            missing_param == user_param_name or
                            missing_param.endswith(f".{user_param_name_only}") or
                            user_param_name.endswith(f".{param_name}")):
                            if is_skipped:
                                skipped_missing.append(missing_param)
                                print(f"  [DEBUG] Parameter {missing_param} was skipped by user in HITL response, removing from missing list")
                
                # Remove skipped parameters from missing_parameters
                if skipped_missing:
                    task_result.missing_parameters = [
                        p for p in task_result.missing_parameters 
                        if p not in skipped_missing
                    ]
                    print(f"  [DEBUG] After removing skipped parameters, remaining missing: {task_result.missing_parameters}")
                    
                    # If all missing parameters were skipped, mark as ready
                    if not task_result.missing_parameters:
                        state.task_status_map[task_id] = ExecutorTaskStatus.READY
                        # Mark as responded (even if all parameters were skipped)
                        if task_id not in state.hitl_responses:
                            # Create a minimal response to mark as responded
                            _record_hitl_response(state, task_id, {"parameters": {}})
                        # Remove from hitl_requests since all parameters were skipped
                        if task_id in state.hitl_requests:
                            del state.hitl_requests[task_id]
                            print(f"  ✓ Task {task_id} removed from hitl_requests (all parameters skipped)")
                        print(f"  ✓ Task {task_id} all missing parameters were skipped by user, marked as READY for execution")
        
        if task_result and task_result.missing_parameters:
            # Check if missing parameters are optional
            # Get task tools to check parameter definitions
            task_tools = []
            if task.result and isinstance(task.result, dict):
                tools = task.result.get("tools", [])
                if isinstance(tools, list):
                    for tool_item in tools:
                        if isinstance(tool_item, dict):
                            tool_name = tool_item.get("tool_name") or tool_item.get("name", "")
                        elif isinstance(tool_item, str):
                            tool_name = tool_item
                        else:
                            tool_name = None
                        if tool_name:
                            task_tools.append(tool_name)
            
            # Check each missing parameter to see if it's optional
            required_missing_params = []
            for missing_param in task_result.missing_parameters:
                # Extract tool name and parameter name
                if '.' in missing_param:
                    tool_name, param_name = missing_param.split('.', 1)
                else:
                    # Try to find tool name from task tools
                    tool_name = task_tools[0] if task_tools else None
                    param_name = missing_param
                
                is_optional = False
                if tool_name and tool_name in tools_params_map:
                    tool_params = tools_params_map[tool_name]
                    input_params = tool_params.get("input_params", [])
                    for param in input_params:
                        if param.get("name") == param_name:
                            # CRITICAL FIX: Use 'required' field from tools_params_table
                            is_required = param.get("required", True)  # Default to True (required)
                            is_optional = not is_required
                            break
                
                if not is_optional:
                    required_missing_params.append(missing_param)
            
            if required_missing_params:
                # Task missing required parameters, should not execute, should wait for HITL response
                print(f"  ⏸ Task {task.task_id} missing required parameters, skipping execution: {required_missing_params}")
                # Ensure status is WAITING_HITL_PARAMS
                if state.task_status_map.get(task.task_id) != ExecutorTaskStatus.WAITING_HITL_PARAMS:
                    state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_HITL_PARAMS
                continue
            else:
                # All missing parameters are optional, allow execution
                print(f"  ✓ Task {task.task_id} missing only optional parameters, allowing execution: {task_result.missing_parameters}")
                # Remove optional missing parameters from the list
                task_result.missing_parameters = []
                # Remove from hitl_requests since all parameters are handled
                if task.task_id in state.hitl_requests:
                    del state.hitl_requests[task.task_id]
                    print(f"  ✓ Task {task.task_id} removed from hitl_requests (all parameters handled)")
        
        # Also check: if task is READY and has no missing parameters, remove from hitl_requests
        if task.task_id in state.hitl_requests:
            task_result = state.task_results.get(task.task_id)
            if task_result and not task_result.missing_parameters:
                del state.hitl_requests[task.task_id]
                print(f"  ✓ Task {task.task_id} removed from hitl_requests (no missing parameters)")
        
        tasks_with_params.append(task)
    
    if not tasks_with_params:
        if ready_tasks:
            print(f"  ⚠ {len(ready_tasks)} ready tasks, but all missing parameters, waiting for HITL response")
        return state
    
    # Calculate number of executable tasks (considering parallel limit)
    available_slots = state.max_parallel_tasks - len(state.running_tasks)
    if available_slots <= 0:
        return state

    # Apply per-tool concurrency limits and priority
    tool_limits = _load_tool_limits()
    tool_priorities = _load_tool_priorities()
    running_tool_counts: Dict[str, int] = {}
    for running_id in state.running_tasks:
        running_task = next((t for t in state.subtasks if t.task_id == running_id), None)
        if not running_task:
            continue
        running_tools = _get_task_tool_names(running_task)
        tool_key = running_tools[0] if running_tools else ""
        running_tool_counts[tool_key] = running_tool_counts.get(tool_key, 0) + 1

    def task_priority(task: SubTask) -> int:
        tool_names = _get_task_tool_names(task)
        tool_key = tool_names[0] if tool_names else ""
        return tool_priorities.get(tool_key, 0)

    tasks_to_execute = []
    for task in sorted(tasks_with_params, key=task_priority, reverse=True):
        if len(tasks_to_execute) >= available_slots:
            break
        tool_names = _get_task_tool_names(task)
        tool_key = tool_names[0] if tool_names else ""
        if tool_key in tool_limits:
            if running_tool_counts.get(tool_key, 0) >= tool_limits[tool_key]:
                continue
        tasks_to_execute.append(task)
        if tool_key in tool_limits:
            running_tool_counts[tool_key] = running_tool_counts.get(tool_key, 0) + 1

    print(f"🔄 Starting execution of {len(tasks_to_execute)} tasks (currently running: {len(state.running_tasks)}, max parallel: {state.max_parallel_tasks})")
    
    if not tasks_to_execute:
        return state

    # Execute tasks in parallel with per-task timeout
    for task in tasks_to_execute:
        state.task_status_map[task.task_id] = ExecutorTaskStatus.RUNNING
        state.running_tasks.append(task.task_id)

    task_timeout_seconds = int(os.getenv("EXECUTOR_TASK_TIMEOUT_SECONDS", "3600"))
    future_map = {}
    start_times = {}
    finalized_task_ids = set()

    def handle_result(task: SubTask, result: TaskExecutionResult):
        existing = state.task_results.get(result.task_id)
        iteration_entry = {
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "error": result.error,
            "error_type": result.error_type,
            "execution_time": result.execution_time,
            "code": result.code
        }
        history = []
        if existing and existing.result_summary and isinstance(existing.result_summary, dict):
            history = existing.result_summary.get("code_iterations", []) or []
        history.append(iteration_entry)
        if result.result_summary is None:
            result.result_summary = _build_task_result_summary(task, result)
        if isinstance(result.result_summary, dict):
            result.result_summary["code_iterations"] = history
        state.task_results[result.task_id] = result
        if task.task_id in state.running_tasks:
            state.running_tasks.remove(task.task_id)

        if result.status == ExecutorTaskStatus.COMPLETED:
            state.completed_count += 1
            state.task_status_map[result.task_id] = ExecutorTaskStatus.COMPLETED
            print(f"  ✓ Task {result.task_id} executed successfully (took {result.execution_time:.2f}s)")
            
            # Expand parameter table with task outputs (for subsequent tasks to use)
            _expand_parameter_table_with_output(state, task, result)
        else:
            should_retry = False
            if result.error_category and result.error_category == ErrorCategory.NETWORK_ERROR:
                if result.retry_count < state.max_retries:
                    should_retry = True
                    result.retry_count += 1
                    state.task_status_map[result.task_id] = ExecutorTaskStatus.READY
                    print(f"  ⚠ Task {result.task_id} network error, re-adding to task pool (retry {result.retry_count}/{state.max_retries})")
                else:
                    state.failed_count += 1
                    state.task_status_map[result.task_id] = ExecutorTaskStatus.FAILED
                    print(f"  ✗ Task {result.task_id} network error, max retries reached, marked as failed")
            elif result.error_category and result.error_category == ErrorCategory.RETRYABLE:
                if result.retry_count < state.max_retries:
                    should_retry = True
                    result.retry_count += 1
                    state.task_status_map[result.task_id] = ExecutorTaskStatus.READY
                    print(f"  ⚠ Task {result.task_id} retryable error, re-adding to task pool (retry {result.retry_count}/{state.max_retries})")
                else:
                    state.failed_count += 1
                    state.task_status_map[result.task_id] = ExecutorTaskStatus.FAILED
                    print(f"  ✗ Task {result.task_id} retryable error, max retries reached, marked as failed")
            else:
                state.failed_count += 1
                state.task_status_map[result.task_id] = ExecutorTaskStatus.FAILED
                error_category_str = result.error_category.value if (result.error_category and hasattr(result.error_category, 'value')) else str(result.error_category) if result.error_category else "Unknown"
                print(f"  ✗ Task {result.task_id} execution failed ({error_category_str}): {result.error}")

            if should_retry:
                error_category_str = result.error_category.value if (result.error_category and hasattr(result.error_category, 'value')) else str(result.error_category) if result.error_category else "Unknown"
                result.failure_analysis = f"Error type: {error_category_str}, Error message: {result.error[:200] if result.error else 'No error message'}"
                result.suggestions = [
                    "Task will automatically retry",
                    f"Current retry count: {result.retry_count}/{state.max_retries}"
                ]

            if result.result_summary is None:
                result.result_summary = _build_task_result_summary(task, result)

    with ThreadPoolExecutor(max_workers=len(tasks_to_execute)) as executor:
        for task in tasks_to_execute:
            future = executor.submit(_execute_single_task, task, state)
            future_map[future] = task
            start_times[future] = time.time()

        futures = set(future_map.keys())
        while futures:
            done, not_done = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
            now = time.time()

            for future in list(done):
                task = future_map[future]
                if task.task_id in finalized_task_ids:
                    futures.remove(future)
                    continue
                try:
                    result = future.result()
                except Exception as e:
                    result = TaskExecutionResult(
                        task_id=task.task_id,
                        status=ExecutorTaskStatus.FAILED,
                        execution_mode="",
                        error=str(e),
                        error_category=ErrorCategory.SYSTEM_ERROR
                    )
                handle_result(task, result)
                finalized_task_ids.add(task.task_id)
                futures.remove(future)

            for future in list(not_done):
                task = future_map[future]
                if task.task_id in finalized_task_ids:
                    futures.remove(future)
                    continue
                if now - start_times.get(future, now) > task_timeout_seconds:
                    future.cancel()
                    timeout_result = TaskExecutionResult(
                        task_id=task.task_id,
                        status=ExecutorTaskStatus.FAILED,
                        execution_mode="",
                        error=f"Task timeout after {task_timeout_seconds}s",
                        error_category=ErrorCategory.SYSTEM_ERROR
                    )
                    handle_result(task, timeout_result)
                    finalized_task_ids.add(task.task_id)
                    futures.remove(future)
    
    return state


def _check_and_handle_streaming_task(output: Any, task: SubTask) -> Optional[Dict[str, Any]]:
    """
    Check and handle streaming task (streaming_task)
    
    Args:
        output: Tool execution output
        task: Task object
        
    Returns:
        If streaming task detected and successfully handled, returns processing result dict; otherwise returns None
    """
    try:
        # Parse output, find streaming_task type
        import json as json_module
        import re
        import ast
        
        # First try to convert output to Python object (if it's a stringified dict)
        parsed_output = None
        if isinstance(output, str):
            # Try to parse with ast.literal_eval (safe, supports Python literals)
            try:
                parsed_output = ast.literal_eval(output)
                print(f"  🔍 [streaming_task] Successfully parsed string output as Python object")
            except (ValueError, SyntaxError) as e:
                # If failed, keep as string
                parsed_output = output
                print(f"  🔍 [streaming_task] Cannot parse as Python object, keeping string format: {e}")
        else:
            parsed_output = output
        
        output_str = str(parsed_output)
        
        # Search for streaming_task keyword
        if "streaming_task" not in output_str:
            return None
        
        print(f"  🔍 [streaming_task] Detected output that may contain streaming_task")
        
        # Method 1: If parsed_output is dict, check directly
        if isinstance(parsed_output, dict):
            # Check top level
            if parsed_output.get("type") == "streaming_task":
                task_id = parsed_output.get("task_id")
                service_id = parsed_output.get("service_id")
                if task_id and service_id:
                    print(f"  🔍 [streaming_task] Detected streaming task: service_id={service_id}, task_id={task_id}")
                    return _handle_streaming_task(service_id, task_id, task)
            
            # Check output field
            if "output" in parsed_output:
                output_value = parsed_output["output"]
                if isinstance(output_value, list):
                    for item in output_value:
                        if isinstance(item, dict):
                            item_text = item.get("text", "")
                            if item_text and "streaming_task" in item_text:
                                try:
                                    # Try to parse JSON
                                    streaming_info = json_module.loads(item_text)
                                    if streaming_info.get("type") == "streaming_task":
                                        task_id = streaming_info.get("task_id")
                                        service_id = streaming_info.get("service_id")
                                        if task_id and service_id:
                                            print(f"  🔍 [streaming_task] Detected streaming task: service_id={service_id}, task_id={task_id}")
                                            return _handle_streaming_task(service_id, task_id, task)
                                except (json_module.JSONDecodeError, TypeError):
                                    # If JSON parsing fails, try regex extraction
                                    task_id_match = re.search(r'"task_id"\s*:\s*"([^"]+)"', item_text)
                                    service_id_match = re.search(r'"service_id"\s*:\s*"([^"]+)"', item_text)
                                    if task_id_match and service_id_match:
                                        task_id = task_id_match.group(1)
                                        service_id = service_id_match.group(1)
                                        print(f"  🔍 [streaming_task] Detected streaming task via regex: service_id={service_id}, task_id={task_id}")
                                        return _handle_streaming_task(service_id, task_id, task)
        
        # Method 2: If parsed_output is list, check elements
        if isinstance(parsed_output, list):
            for item in parsed_output:
                if isinstance(item, dict):
                    # Check text field
                    item_text = item.get("text", "")
                    if item_text and "streaming_task" in item_text:
                        try:
                            streaming_info = json_module.loads(item_text)
                            if streaming_info.get("type") == "streaming_task":
                                task_id = streaming_info.get("task_id")
                                service_id = streaming_info.get("service_id")
                                if task_id and service_id:
                                    print(f"  🔍 [streaming_task] Detected streaming task: service_id={service_id}, task_id={task_id}")
                                    return _handle_streaming_task(service_id, task_id, task)
                        except (json_module.JSONDecodeError, TypeError):
                            # If JSON parsing fails, try regex extraction
                            task_id_match = re.search(r'"task_id"\s*:\s*"([^"]+)"', item_text)
                            service_id_match = re.search(r'"service_id"\s*:\s*"([^"]+)"', item_text)
                            if task_id_match and service_id_match:
                                task_id = task_id_match.group(1)
                                service_id = service_id_match.group(1)
                                print(f"  🔍 [streaming_task] Detected streaming task via regex: service_id={service_id}, task_id={task_id}")
                                return _handle_streaming_task(service_id, task_id, task)
        
        # Method 3: If still string, try regex extraction directly
        if isinstance(output, str) or (isinstance(parsed_output, str) and parsed_output == output):
            # Find task_id and service_id (supports single and double quotes)
            task_id_match = re.search(r'["\']task_id["\']\s*:\s*["\']([^"\']+)["\']', output_str)
            service_id_match = re.search(r'["\']service_id["\']\s*:\s*["\']([^"\']+)["\']', output_str)
            type_match = re.search(r'["\']type["\']\s*:\s*["\']streaming_task["\']', output_str)
            
            if task_id_match and service_id_match and type_match:
                task_id = task_id_match.group(1)
                service_id = service_id_match.group(1)
                print(f"  🔍 [streaming_task] Detected streaming task from string via regex: service_id={service_id}, task_id={task_id}")
                return _handle_streaming_task(service_id, task_id, task)
        
        return None
    except Exception as e:
        print(f"  ⚠ [streaming_task] Error checking streaming task: {e}")
        return None


def _handle_streaming_task(service_id: str, task_id: str, task: SubTask) -> Dict[str, Any]:
    """
    Handle streaming task: establish SSE connection and receive messages
    
    Args:
        service_id: Service ID
        task_id: Task ID
        task: Task object
        
    Returns:
        Processing result dictionary
    """
    from pathlib import Path as _PathLib
    import json as json_module
    
    try:
        # Load MCP server configuration
        # Calculate agent directory correctly
        # __file__ is at: agent/nodes/subagents/executor/graph.py
        # Need to go up 4 levels: executor -> subagents -> nodes -> agent
        agent_dir = _PathLib(__file__).parent.parent.parent.parent
        mcp_servers_path = agent_dir / "config" / "mcp_servers.json"
        
        if not mcp_servers_path.exists():
            return {
                "status": ExecutorTaskStatus.FAILED,
                "error": f"MCP server configuration file does not exist: {mcp_servers_path}",
                "output": None
            }
        
        with open(mcp_servers_path, "r", encoding="utf-8") as f:
            mcp_servers = json_module.load(f)
        
        if service_id not in mcp_servers:
            return {
                "status": ExecutorTaskStatus.FAILED,
                "error": f"Service {service_id} not in configuration",
                "output": None
            }
        
        server_config = mcp_servers[service_id]
        base_url = server_config.get("url", "")
        
        # Build SSE endpoint URL by replacing /sse with /stream/{task_id}
        # Examples:
        #   http://117.10.59.114:40001/mcp/8088/sse -> http://117.10.59.114:40001/mcp/8088/stream/{task_id}
        #   http://127.0.0.1:8088/sse -> http://127.0.0.1:8088/stream/{task_id}
        if base_url.endswith("/sse"):
            sse_url = base_url[:-4] + f"/stream/{task_id}"
        elif "/sse" in base_url:
            # Handle cases like /sse?param=value
            sse_url = base_url.replace("/sse", f"/stream/{task_id}")
        else:
            # Fallback: append /stream/{task_id}
            from urllib.parse import urlparse
            parsed_url = urlparse(base_url)
            host = parsed_url.hostname
            port = parsed_url.port
            if host and port:
                sse_url = f"http://{host}:{port}/stream/{task_id}"
            else:
                return {
                    "status": ExecutorTaskStatus.FAILED,
                    "error": f"Cannot build stream URL from service configuration: {base_url}",
                    "output": None
                }
        
        print(f"  🔍 [streaming_task] Built SSE URL: {sse_url}")
        print(f"  🔍 [streaming_task] Service configuration: base_url={base_url}")
        
        # Establish SSE connection and receive messages
        try:
            result = _receive_sse_messages(sse_url, task_id, service_id, timeout=3600)  # Default 1 hour timeout
            print(f"  🔍 [streaming_task] SSE processing returned result: status={result.get('status')}")
            return result
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"  ✗ [streaming_task] Exception occurred when calling _receive_sse_messages: {e}")
            print(f"  {error_traceback[:500]}")
            return {
                "status": ExecutorTaskStatus.FAILED,
                "error": f"Exception occurred when calling SSE receive function: {str(e)}",
                "output": None
            }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"  ✗ [streaming_task] Failed to process streaming task: {e}")
        print(f"  {error_traceback[:500]}")
        return {
            "status": ExecutorTaskStatus.FAILED,
            "error": f"Failed to process streaming task: {str(e)}",
            "output": None
        }


def _receive_sse_messages(sse_url: str, task_id: str, service_id: str, timeout: int = 3600) -> Dict[str, Any]:
    """
    Receive SSE messages until task completion
    
    Args:
        sse_url: SSE endpoint URL
        task_id: Task ID
        service_id: Service ID
        timeout: Timeout in seconds
        
    Returns:
        Processing result dictionary
    """
    try:
        import requests
        import time
        from datetime import datetime
        
        print(f"  🔍 [streaming_task] Starting to receive SSE messages: {sse_url}")
        print(f"  🔍 [streaming_task] Timeout setting: {timeout} seconds")
        
        start_time = time.time()
        all_messages = []
        final_result = None
        task_completed = False
        task_failed = False
        
        # Establish SSE connection
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache"
        }
        
        # Add API key header if available (required for Nginx proxy authentication)
        import os
        api_key = os.environ.get("OPEN_SANDBOX_API_KEY") or os.environ.get("SANDBOX_API_KEY")
        if api_key:
            headers["OPEN-SANDBOX-API-KEY"] = api_key
            print(f"  🔍 [streaming_task] Added API key header for authentication")
        
        try:
            print(f"  🔍 [streaming_task] Establishing SSE connection: {sse_url}")
            print(f"  🔍 [streaming_task] Request headers (without API key): Accept={headers.get('Accept')}, Cache-Control={headers.get('Cache-Control')}")
            response = requests.get(sse_url, headers=headers, stream=True, timeout=timeout)
            print(f"  🔍 [streaming_task] Received response, status code: {response.status_code}")
            
            # Check response status
            if response.status_code != 200:
                error_msg = f"SSE connection failed with status code {response.status_code}"
                print(f"  ✗ [streaming_task] {error_msg}")
                return {
                    "status": ExecutorTaskStatus.FAILED,
                    "error": error_msg,
                    "output": {
                        "task_id": task_id,
                        "service_id": service_id,
                        "response_status": response.status_code,
                        "response_text": response.text[:500] if hasattr(response, 'text') else None
                    }
                }
            
            print(f"  ✓ [streaming_task] SSE connection established, status code: {response.status_code}")
            
            # Read SSE messages line by line
            for line in response.iter_lines(decode_unicode=True):
                if time.time() - start_time > timeout:
                    print(f"  ⚠ [streaming_task] Message reception timeout")
                    break
                
                if not line:
                    continue
                
                # Parse SSE message format
                # SSE format: data: {...}
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    try:
                        import json as json_module
                        message_data = json_module.loads(data_str)
                        
                        message_type = message_data.get("type", "")
                        # Extract data field if exists (MCP server may wrap data in "data" field)
                        data_field = message_data.get("data", {})
                        if isinstance(data_field, dict):
                            # If data field exists, merge it with message_data for easier access
                            # But keep original structure in raw
                            message_content = message_data.get("content", message_data.get("message", ""))
                            if not message_content and data_field:
                                # Try to construct message from data field
                                if "status" in data_field:
                                    message_content = f"Status: {data_field.get('status')}"
                                if "progress_percent" in data_field:
                                    message_content += f", Progress: {data_field.get('progress_percent')}%"
                        else:
                            message_content = message_data.get("content", message_data.get("message", ""))
                        
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        all_messages.append({
                            "timestamp": timestamp,
                            "type": message_type,
                            "content": message_content,
                            "raw": message_data
                        })
                        
                        print(f"  📨 [streaming_task] [{timestamp}] {message_type}: {message_content[:100]}")
                        
                        # Check task status (check both top level and data field)
                        status = message_data.get("status")
                        if isinstance(data_field, dict):
                            status = status or data_field.get("status")
                        
                        # Handle different message types
                        if message_type == "result":
                            # type: result 消息是工具的执行结果，包含output_file等关键信息
                            # 这是最重要的消息，保存为final_result
                            final_result = message_data
                            print(f"  ✓ [streaming_task] Received result message (tool execution result)")
                            
                            # 修复问题2: 检查 result 消息内容是否包含错误信息
                            # 即使 status 字段为 success，如果 message 内容以 "Error:" 开头，也应标记为失败
                            if message_content and isinstance(message_content, str):
                                content_stripped = message_content.strip()
                                # 检测常见错误模式
                                error_patterns = [
                                    "Error:",
                                    "error:",
                                    "ERROR:",
                                    "Integration failed",
                                    "Processing failed",
                                    "Execution failed",
                                    "Failed to",
                                    "Exception:",
                                ]
                                for pattern in error_patterns:
                                    if content_stripped.startswith(pattern) or f"\n{pattern}" in content_stripped:
                                        task_failed = True
                                        error_msg = content_stripped
                                        print(f"  ✗ [streaming_task] Detected error in result message: {error_msg[:200]}")
                                        break
                            # 不要break，继续等待end消息
                        elif message_type == "task_completed" or (message_type == "progress" and status == "completed"):
                            # task_completed 或 progress消息中status为completed，只是表示任务完成
                            # 但可能还有result消息在后面，不要立即break
                            task_completed = True
                            # 如果没有result消息，使用这个作为final_result
                            if final_result is None:
                                final_result = message_data
                            print(f"  ✓ [streaming_task] Task completed (waiting for end message)")
                            # 不要break，继续等待end消息
                        elif message_type == "task_failed" or status == "failed":
                            task_failed = True
                            final_result = message_data
                            error_msg = message_data.get("error", message_data.get("message", "Task failed"))
                            if isinstance(data_field, dict) and not error_msg:
                                error_msg = data_field.get("error", data_field.get("message", "Task failed"))
                            print(f"  ✗ [streaming_task] Task failed: {error_msg}")
                            # 不要break，继续等待end消息
                        elif message_type == "error":
                            # Handle error message type
                            task_failed = True
                            # 如果没有result消息，使用error作为final_result
                            if final_result is None:
                                final_result = message_data
                            error_msg = message_data.get("error", message_data.get("message", "Task failed"))
                            if isinstance(data_field, dict) and not error_msg:
                                error_msg = data_field.get("error", data_field.get("message", "Task failed"))
                            if not error_msg:
                                error_msg = message_content or "Task failed with error"
                            print(f"  ✗ [streaming_task] Task failed with error: {error_msg}")
                            # 不要break，继续等待end消息
                        elif message_type == "end":
                            # type: end 消息标志着工具推送SSE消息完毕
                            # 只有收到end消息才应该结束
                            print(f"  ✓ [streaming_task] Received end message, SSE stream finished")
                            # 根据之前的状态确定最终状态
                            if task_failed:
                                break
                            elif task_completed or final_result is not None:
                                # 如果收到了result消息或completed消息，标记为完成
                                task_completed = True
                                break
                            else:
                                # End without error or completion, check if we have enough info
                                # If we have messages, assume completed (some servers may not send explicit completion)
                                if len(all_messages) > 0:
                                    task_completed = True
                                    if final_result is None:
                                        final_result = message_data
                                    print(f"  ✓ [streaming_task] Task ended, assuming completed")
                                    break
                        elif message_type == "progress" or message_type == "status":
                            # Update progress information (check both top level and data field)
                            progress = None
                            if isinstance(data_field, dict):
                                # MCP server format: {"type": "progress", "data": {"progress_percent": 6.0, ...}}
                                progress = data_field.get("progress_percent")
                                if progress is None:
                                    progress = data_field.get("progress")
                                if progress is None:
                                    progress = data_field.get("percentage")
                            
                            # Fallback to top level
                            if progress is None:
                                progress = message_data.get("progress")
                            if progress is None:
                                progress = message_data.get("percentage")
                            if progress is None:
                                progress = 0
                            
                            # Extract additional progress info from data field
                            batch_info = ""
                            if isinstance(data_field, dict):
                                batch_current = data_field.get("batch_current")
                                batch_total = data_field.get("batch_total")
                                elapsed_minutes = data_field.get("elapsed_minutes")
                                if batch_current is not None and batch_total is not None:
                                    batch_info = f" ({batch_current}/{batch_total} batches)"
                                if elapsed_minutes is not None:
                                    batch_info += f", elapsed: {elapsed_minutes:.1f} min"
                            
                            print(f"  📊 [streaming_task] Progress: {progress}%{batch_info}")
                        
                    except json_module.JSONDecodeError:
                        # If not JSON, treat as text message
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        all_messages.append({
                            "timestamp": timestamp,
                            "type": "text",
                            "content": data_str,
                            "raw": data_str
                        })
                        print(f"  📨 [streaming_task] [{timestamp}] text: {data_str[:100]}")
                
                elif line.startswith("event: "):
                    # SSE event type
                    event_type = line[7:]
                    print(f"  🔔 [streaming_task] Event type: {event_type}")
                
                elif line.startswith("id: "):
                    # SSE message ID
                    msg_id = line[4:]
                    # Can be used for resuming
        
        except requests.exceptions.Timeout:
            return {
                "status": ExecutorTaskStatus.FAILED,
                "error": f"SSE message reception timeout ({timeout} seconds)",
                "output": {
                    "messages": all_messages,
                    "final_result": final_result
                }
            }
        except requests.exceptions.RequestException as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"  ✗ [streaming_task] SSE connection error: {str(e)}")
            print(f"  {error_traceback[:500]}")
            return {
                "status": ExecutorTaskStatus.FAILED,
                "error": f"SSE connection error: {str(e)}",
                "output": {
                    "messages": all_messages,
                    "final_result": final_result
                }
            }
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"  ✗ [streaming_task] Exception occurred while receiving SSE messages: {str(e)}")
            print(f"  {error_traceback[:500]}")
            return {
                "status": ExecutorTaskStatus.FAILED,
                "error": f"Exception occurred while receiving SSE messages: {str(e)}",
                "output": {
                    "messages": all_messages,
                    "final_result": final_result
                }
            }
        
        # Return result based on task status
        if task_completed:
            print(f"  ✓ [streaming_task] Task completed successfully, received {len(all_messages)} messages")
            return {
                "status": ExecutorTaskStatus.COMPLETED,  # Use ExecutorTaskStatus enum
                "output": {
                    "task_id": task_id,
                    "service_id": service_id,
                    "messages": all_messages,
                    "final_result": final_result,
                    "total_messages": len(all_messages)
                },
                "error": None
            }
        elif task_failed:
            error_msg = final_result.get("error", final_result.get("message", "Task failed")) if final_result else "Task failed"
            print(f"  ✗ [streaming_task] Task failed: {error_msg}")
            return {
                "status": ExecutorTaskStatus.FAILED,  # Use ExecutorTaskStatus enum
                "error": error_msg,
                "output": {
                    "task_id": task_id,
                    "service_id": service_id,
                    "messages": all_messages,
                    "final_result": final_result
                }
            }
        else:
            # Timeout or connection interrupted, but may have partial messages
            print(f"  ⚠ [streaming_task] Task not completed (may have timed out or connection interrupted), received {len(all_messages)} messages")
            return {
                "status": ExecutorTaskStatus.FAILED,  # Use ExecutorTaskStatus enum
                "error": "Task not completed (may have timed out or connection interrupted)",
                "output": {
                    "task_id": task_id,
                    "service_id": service_id,
                    "messages": all_messages,
                    "final_result": final_result,
                    "partial": True
                }
            }
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"  ✗ [streaming_task] Failed to receive SSE messages: {e}")
        print(f"  {error_traceback[:500]}")
        return {
            "status": ExecutorTaskStatus.FAILED,
            "error": f"Failed to receive SSE messages: {str(e)}",
            "output": {
                "task_id": task_id,
                "service_id": service_id,
                "error": str(e),
                "traceback": error_traceback[:500]
            }
        }


def _execute_single_task(task: SubTask, state: ExecutorState) -> TaskExecutionResult:
    """Execute a single task"""
    # Restore parent_state from thread-scoped storage if missing
    # This is critical because this function runs in ThreadPoolExecutor
    _restore_parent_state(state)
    
    start_time = time.time()
    result = state.task_results.get(task.task_id, TaskExecutionResult(
        task_id=task.task_id,
        status=ExecutorTaskStatus.RUNNING,
        execution_mode="",
        retry_count=0
    ))
    result.status = ExecutorTaskStatus.RUNNING
    
    try:
        # Determine execution mode
        task_result = task.result if isinstance(task.result, dict) else {}
        tools = task_result.get("tools", [])
        
        # Check if the only tool is "codeact" - it's not an MCP tool, it's our internal code executor
        tool_names = []
        for t in tools:
            if isinstance(t, str):
                tool_names.append(t)
            elif isinstance(t, dict):
                tool_names.append(t.get("tool_name", t.get("name", "")))
        
        # If only codeact is in tools, use CODEACT mode instead of MCP_TOOL
        is_only_codeact = len(tool_names) == 1 and tool_names[0] == "codeact"
        
        if tools and len(tools) > 0 and not is_only_codeact:
            execution_mode = CodeActExecutionMode.MCP_TOOL
        else:
            execution_mode = CodeActExecutionMode.CODEACT
        
        result.execution_mode = execution_mode.value
        
        # Get parsed parameters
        try:
            parameters = _preprocess_parameters_with_codeact(task, result.parameters, state)
        except Exception as preprocess_error:
            result.status = ExecutorTaskStatus.FAILED
            result.error = f"Parameter preprocessing failed: {preprocess_error}"
            return result
        result.parameters = parameters
        
        def _run_codeact(
            execution_mode: CodeActExecutionMode,
            params: Dict[str, Any],
            previous_code: Optional[str] = None,
            previous_error: Optional[str] = None,
            error_category: Optional[str] = None,
            revision_plan: Optional[Any] = None,
            revision_iteration: int = 0
        ):
            # Get revision_iteration from task result to maintain state across invocations
            if revision_iteration == 0:
                revision_iteration = result.revision_iteration if hasattr(result, 'revision_iteration') else 0
            
            # Build CodeAct subgraph input
            codeact_input = codeact_input_mapper(
                executor_state=state,
                task=task,
                execution_mode=execution_mode,
                parameters=params,
                previous_code=previous_code,
                previous_error=previous_error,
                error_category=error_category,
                revision_plan=revision_plan,
                revision_iteration=revision_iteration,
                parent_state=state.parent_state  # Pass parent_state for parameter inference
            )

            # Call CodeAct subgraph
            codeact_graph = build_codeact_subgraph()
            try:
                codeact_output = codeact_graph.invoke(codeact_input)
            except Exception as codeact_error:
                # Wrap CodeAct invocation error with more context
                import traceback
                error_traceback = traceback.format_exc()
                print(f"\n{'='*80}")
                print(f"【CodeAct 子图调用错误 - Task {task.task_id}】")
                print(f"{'='*80}")
                print(f"错误类型: {type(codeact_error).__name__}")
                print(f"错误信息: {codeact_error}")
                print(f"\n完整错误堆栈:")
                print(error_traceback)
                print(f"{'='*80}\n")
                # Re-raise to be caught by outer exception handler
                raise RuntimeError(f"CodeAct subgraph invocation failed for task {task.task_id}: {codeact_error}") from codeact_error

            # Convert dict output to CodeActState object (LangGraph returns dict)
            if isinstance(codeact_output, dict):
                codeact_state = CodeActState.model_validate(codeact_output)
            else:
                codeact_state = codeact_output

            # Process execution result
            exec_result = codeact_output_mapper(codeact_state)
            return exec_result, codeact_state

        if state.use_react_executor:
            from nodes.subagents.executor.react_executor import execute_with_react
            return execute_with_react(
                task=task,
                state=state,
                result=result,
                base_execution_mode=execution_mode,
                parameters=parameters,
                run_codeact=_run_codeact,
                handle_streaming=_check_and_handle_streaming_task,
                classify_error=classify_error,
                analyze_failure=_analyze_failure,
                generate_suggestions=_generate_suggestions,
                status_enum=ExecutorTaskStatus,
                error_category_enum=ErrorCategory,
                fix_code_mode=CodeActExecutionMode.FIX_CODE,
                fix_parameter_mode=CodeActExecutionMode.FIX_PARAMETER,
                max_steps=state.react_max_steps
            )

        exec_result, codeact_state = _run_codeact(execution_mode, parameters)
        def _get_next_action(error_category: Optional[ErrorCategory]) -> str:
            if error_category == ErrorCategory.PARAMETER_ERROR:
                return "parameter_fix"
            if error_category == ErrorCategory.CODE_ERROR:
                return "code_fix"
            if error_category in [ErrorCategory.RETRYABLE, ErrorCategory.NETWORK_ERROR]:
                return "retry"
            if error_category == ErrorCategory.SYSTEM_ERROR:
                return "manual_check"
            return "retry"

        if exec_result.get("status") == "success":
            output = exec_result.get("output")
            
            print(f"  🔍 [execute_task] Checking if output is streaming task, output type: {type(output)}")
            if isinstance(output, str):
                print(f"  🔍 [execute_task] Output string length: {len(output)}, first 200 chars: {output[:200]}")
            
            # Check if it's a streaming task (streaming_task)
            streaming_result = _check_and_handle_streaming_task(output, task)
            if streaming_result:
                print(f"  ✓ [execute_task] Detected streaming task, starting SSE connection processing...")
                # If streaming task, use streaming processing result
                # Note: Streaming tasks must wait for SSE connection to complete, only mark as COMPLETED when truly completed
                streaming_status = streaming_result.get("status")
                
                # Handle status (may be ExecutorTaskStatus enum or string)
                if streaming_status is None:
                    # If status is None, mark as failed
                    print(f"  ⚠ [streaming_task] Streaming result has no status, marking as failed")
                    result.status = ExecutorTaskStatus.FAILED
                    result.error = streaming_result.get("error", "Streaming task returned no status")
                elif isinstance(streaming_status, ExecutorTaskStatus):
                    result.status = streaming_status
                elif isinstance(streaming_status, str):
                    # If string, convert to enum
                    if streaming_status.lower() == "completed" or streaming_status == "COMPLETED":
                        result.status = ExecutorTaskStatus.COMPLETED
                    elif streaming_status.lower() == "failed" or streaming_status == "FAILED":
                        result.status = ExecutorTaskStatus.FAILED
                    else:
                        # If status unclear, mark as failed (should not happen)
                        result.status = ExecutorTaskStatus.FAILED
                        result.error = f"Streaming task returned unknown status: {streaming_status}"
                else:
                    # If status unclear, mark as failed (should not happen)
                    result.status = ExecutorTaskStatus.FAILED
                    result.error = f"Streaming task returned unknown status type: {type(streaming_status)}"
                
                result.output = streaming_result.get("output", output)
                if streaming_result.get("error"):
                    result.error = streaming_result.get("error")
                    # If there's an error, ensure status is FAILED
                    if result.status != ExecutorTaskStatus.FAILED:
                        result.status = ExecutorTaskStatus.FAILED
                if result.status == ExecutorTaskStatus.FAILED and not result.error_type:
                    result.error_type = _extract_error_type_from_exec_output(result.output)
                
                # Ensure status is set (defensive check)
                if result.status is None:
                    print(f"  ⚠ [streaming_task] Status is still None after processing, defaulting to FAILED")
                    result.status = ExecutorTaskStatus.FAILED
                    if not result.error:
                        result.error = "Streaming task processing completed but status was not set"
                
                # Safe status logging
                try:
                    status_value = result.status.value if result.status else "None"
                    print(f"  ✓ [streaming_task] Streaming task processing completed, status: {status_value}")
                except AttributeError:
                    print(f"  ⚠ [streaming_task] Streaming task processing completed, status: {result.status} (cannot get .value)")
            else:
                # Normal task, directly use original result
                result.status = ExecutorTaskStatus.COMPLETED
                result.output = output
            
            extracted_error_type = _extract_error_type_from_exec_output(result.output)
            if extracted_error_type and not result.error_type:
                result.error_type = extracted_error_type

            result.code = exec_result.get("code")
            # Save revision_iteration from codeact_state to maintain state across invocations
            if 'codeact_state' in locals() and hasattr(codeact_state, 'revision_iteration'):
                result.revision_iteration = codeact_state.revision_iteration
            elif 'codeact_state' in locals() and isinstance(codeact_state, dict):
                result.revision_iteration = codeact_state.get('revision_iteration', 0)
            result.result_summary = {
                "status": "success",
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "error_type": result.error_type,
                "next_action": "none"
            }
        else:
            result.status = ExecutorTaskStatus.FAILED
            # Always record code, even if execution failed (code may have been generated)
            result.code = exec_result.get("code")
            # Save revision_iteration from codeact_state to maintain state across invocations
            if 'codeact_state' in locals() and hasattr(codeact_state, 'revision_iteration'):
                result.revision_iteration = codeact_state.revision_iteration
            elif 'codeact_state' in locals() and isinstance(codeact_state, dict):
                result.revision_iteration = codeact_state.get('revision_iteration', 0)
            result.error = exec_result.get("error") or "Execution failed"
            error_type = exec_result.get("error_type")
            if not error_type:
                error_type = _extract_error_type_from_exec_output(exec_result.get("output"))
            error_type = error_type or "UnknownError"
            result.error_type = error_type
            # Ensure error is not None before classification
            error_for_classification = result.error if result.error else "Execution failed"
            result.error_category = classify_error(error_for_classification, error_type)
            
            # Generate error analysis and suggestions (ensure error is not None)
            error_for_analysis = result.error if result.error else "Execution failed"
            result.failure_analysis = _analyze_failure(error_for_analysis, error_type, result.error_category)
            result.suggestions = _generate_suggestions(result.error_category, error_for_analysis, result.retry_count, state.max_retries)
            result.result_summary = {
                "status": "failed",
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "error_type": error_type,
                "error_category": result.error_category.value if result.error_category else None,
                "next_action": _get_next_action(result.error_category)
            }
    
    except Exception as e:
        result.status = ExecutorTaskStatus.FAILED
        error_msg = str(e)
        error_type = type(e).__name__
        result.error_type = error_type
        result.error_category = classify_error(error_msg, error_type)
        # Record detailed error info for debugging
        import traceback
        error_traceback = traceback.format_exc()
        
        # Print full error traceback to console (no truncation for console output)
        print(f"\n{'='*80}")
        print(f"【Task {task.task_id} 执行错误】")
        print(f"{'='*80}")
        print(f"错误类型: {error_type}")
        print(f"错误信息: {error_msg}")
        print(f"\n完整错误堆栈:")
        print(error_traceback)
        print(f"{'='*80}\n")
        
        # Store full error traceback in result (increase limit to 10000 chars for detailed debugging)
        error_traceback_stored = error_traceback
        if len(error_traceback_stored) > 10000:
            error_traceback_stored = error_traceback_stored[:10000] + f"\n... (truncated, total length: {len(error_traceback)} chars)"
        result.error = f"{error_msg}\n\n完整错误堆栈:\n{error_traceback_stored}"
        
        # Try to extract code and revision_iteration from CodeAct state if available (even if execution failed)
        # This ensures we log code and maintain revision state even when there's an exception
        try:
            if 'codeact_state' in locals() and hasattr(codeact_state, 'generated_code'):
                result.code = codeact_state.generated_code
                if hasattr(codeact_state, 'revision_iteration'):
                    result.revision_iteration = codeact_state.revision_iteration
            elif 'codeact_state' in locals() and isinstance(codeact_state, dict):
                result.code = codeact_state.get('generated_code')
                result.revision_iteration = codeact_state.get('revision_iteration', 0)
        except:
            pass  # If code extraction fails, continue without code
        
        # Generate error analysis and suggestions
        result.failure_analysis = _analyze_failure(error_msg, error_type, result.error_category)
        result.suggestions = _generate_suggestions(result.error_category, error_msg, result.retry_count, state.max_retries)
        result.result_summary = {
            "status": "failed",
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "error_type": error_type,
            "error_category": result.error_category.value if result.error_category else None,
            "next_action": "manual_check" if result.error_category == ErrorCategory.SYSTEM_ERROR else "retry"
        }
        
        print(f"  ✗ Task {task.task_id} execution exception: {error_msg}")
    
    result.execution_time = time.time() - start_time
    return result


def analyze_results_node(state: ExecutorState) -> ExecutorState:
    """
    Result reasoning node
    
    Analyze task execution results. Only trigger HITL confirmation if there's an explicit error.
    If no error, assume result is satisfactory and continue execution.
    """
    # Analyze completed tasks
    for task in state.subtasks:
        result = state.task_results.get(task.task_id)
        if not result or result.status != ExecutorTaskStatus.COMPLETED:
            continue
        
        # If already analyzed, skip
        if result.result_satisfied is not None:
            continue
        
        # Check if there's an explicit error in the result
        has_explicit_error = False
        error_message = ""
        
        if result.output:
            output_str = str(result.output).lower() if result.output else ""
            output_data = result.output if isinstance(result.output, dict) else {}
            
            # Check for error indicators in output
            if isinstance(result.output, dict):
                # Check for error status
                status = output_data.get("status", "")
                error_type = output_data.get("error_type", "")
                error = output_data.get("error", "")
                
                if status in ["error", "failed", "failure"]:
                    has_explicit_error = True
                    error_message = error or error_type or "Task failed"
                elif error_type or error:
                    has_explicit_error = True
                    error_message = error or error_type
                    
                # Check nested final_result
                final_result = output_data.get("final_result", {})
                if isinstance(final_result, dict):
                    if final_result.get("status") in ["error", "failed"]:
                        has_explicit_error = True
                        error_message = final_result.get("message", "") or final_result.get("error", "")
            elif isinstance(result.output, str):
                # Check string output for error patterns
                if '"status": "error"' in output_str or '"status": "failed"' in output_str:
                    has_explicit_error = True
                    error_message = "Task execution returned error status"
        
        # ========== NEW: Validate data effectiveness ==========
        # Even if no explicit error, check if the returned data is actually valid
        if not has_explicit_error:
            tool_name = _extract_tool_name_from_task(task)
            data_valid, data_error = _validate_result_data_validity(result, tool_name)
            if not data_valid:
                has_explicit_error = True
                error_message = data_error
                print(f"  ⚠ Task {task.task_id} returned invalid data: {data_error}")
        
        # If no explicit error, mark as satisfied and continue without HITL
        if not has_explicit_error:
            result.result_satisfied = True
            # Calculate confidence score based on multiple factors
            tool_name = _extract_tool_name_from_task(task)
            result.confidence_score = _calculate_confidence_score(result, tool_name)
            print(f"  ✓ Task {task.task_id} completed without errors (confidence: {result.confidence_score:.2f}), continuing...")
        else:
            # Only trigger HITL for explicit errors
            result.result_satisfied = False
            result.confidence_score = 0.3
            _record_hitl_request(state, task.task_id, {
                "type": "result_confirmation",
                "task_id": task.task_id,
                "task_description": task.content,
                "result": str(result.output)[:500],
                "reason": error_message,
                "message": f"Task {task.task_id} failed with error: {error_message}. Continue executing subsequent tasks?"
            })
            state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_HITL_CONFIRM
            print(f"  ⚠ Task {task.task_id} has error, needs user confirmation")
    
        # Build per-task summary after execution analysis
        if result.result_summary is None:
            result.result_summary = _build_task_result_summary(task, result)
    
    return state


def _extract_tool_name_from_task(task: SubTask) -> str:
    """Extract tool name from task content or result."""
    # Try to get tool name from task result
    if task.result and isinstance(task.result, dict):
        tools = task.result.get("tools", [])
        if tools and isinstance(tools, list) and len(tools) > 0:
            first_tool = tools[0]
            if isinstance(first_tool, dict):
                return first_tool.get("tool_name", first_tool.get("name", ""))
            elif isinstance(first_tool, str):
                return first_tool
    
    # Try to extract from task content
    content = task.content.lower() if task.content else ""
    
    # Common tool name patterns in task content
    tool_patterns = [
        "extract_cdr3_from_airr", "analyze_vdj_batch", "integrate_tcr_data_complete",
        "integrate_bcr_data_complete", "predict_tcr_binding_complete", "metabcr",
        "tcr_clonotype_analysis", "tcell_celltype_visualization", "tcr_binding_visualization"
    ]
    
    for pattern in tool_patterns:
        if pattern in content:
            return pattern
    
    return ""


def _validate_result_data_validity(result: TaskExecutionResult, tool_name: str) -> tuple:
    """
    Validate if the returned data is actually valid, not just "status: success" with null data.
    
    Args:
        result: Task execution result
        tool_name: Name of the tool that was executed
        
    Returns:
        (is_valid, error_message) - is_valid is True if data is valid, False otherwise
    """
    if not result.output:
        # Empty output might be normal for some tools
        return True, ""
    
    output_data = result.output if isinstance(result.output, dict) else {}
    
    # ========== Tool-specific validation ==========
    
    # For extract_cdr3_from_airr - check if CDR3 results are all null
    if tool_name == "extract_cdr3_from_airr":
        cdr3_results = output_data.get("cdr3_results", [])
        if cdr3_results and len(cdr3_results) > 0:
            # Check if all records have all null values
            all_null_count = 0
            for record in cdr3_results:
                if isinstance(record, dict):
                    if all(v is None for v in record.values()):
                        all_null_count += 1
            
            if all_null_count == len(cdr3_results):
                return False, f"All {len(cdr3_results)} CDR3 records are null - no valid data extracted from input"
            elif all_null_count > len(cdr3_results) * 0.9:
                # More than 90% are null - likely a problem
                return False, f"{all_null_count}/{len(cdr3_results)} CDR3 records are null - data extraction mostly failed"
    
    # For analyze_vdj_batch - check if V/D/J annotations are present
    if tool_name == "analyze_vdj_batch":
        # Check for AIRR format results
        airr_results = output_data.get("airr_results", [])
        if airr_results and len(airr_results) > 0:
            # Check if at least some records have V gene annotations
            v_gene_count = sum(
                1 for r in airr_results 
                if isinstance(r, dict) and r.get("v_call")
            )
            if v_gene_count == 0:
                return False, f"No V gene annotations found in {len(airr_results)} AIRR records"
    
    # For tcell visualization tools - check if input file exists error
    if tool_name in ["tcr_clonotype_analysis", "tcell_celltype_visualization", "tcr_binding_visualization"]:
        # Check for "file does not exist" error in message
        message = output_data.get("message", "") or output_data.get("content", "")
        if "does not exist" in message.lower() or "file not found" in message.lower():
            return False, f"Input file not found: {message}"
    
    # For integration tools - check if output file was actually created
    if tool_name in ["integrate_tcr_data_complete", "integrate_bcr_data_complete"]:
        result_path = output_data.get("result_path", [])
        if isinstance(result_path, str):
            result_path = [result_path]
        if not result_path or not any(".rds" in str(p) for p in result_path):
            # Check in final_result as well
            final_result = output_data.get("final_result", {})
            if isinstance(final_result, dict):
                result_path = final_result.get("result_path", [])
                if isinstance(result_path, str):
                    result_path = [result_path]
            if not result_path or not any(".rds" in str(p) for p in result_path):
                return False, "Integration completed but no RDS output file was generated"
    
    return True, ""


def _calculate_confidence_score(result: TaskExecutionResult, tool_name: str) -> float:
    """
    Calculate confidence score based on multiple factors.
    
    Factors:
    1. Execution status (success/failed)
    2. Output file existence
    3. Data validity (non-empty, correct format)
    4. Execution time reasonability
    5. Statistics information presence
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    base_score = 0.5
    
    output_data = result.output if isinstance(result.output, dict) else {}
    
    # Factor 1: Check if output files were generated
    result_paths = output_data.get("result_path", [])
    if isinstance(result_paths, str):
        result_paths = [result_paths]
    if result_paths and len(result_paths) > 0:
        base_score += 0.2
    
    # Factor 2: Check for statistics information
    if "statistics" in output_data:
        base_score += 0.1
    
    # Factor 3: Check for meaningful message
    message = output_data.get("message", "")
    if message and len(message) > 20:
        base_score += 0.1
    
    # Factor 4: Tool-specific checks
    if tool_name == "extract_cdr3_from_airr":
        cdr3_results = output_data.get("cdr3_results", [])
        if cdr3_results:
            valid_count = sum(
                1 for r in cdr3_results 
                if isinstance(r, dict) and any(v is not None for v in r.values())
            )
            validity_ratio = valid_count / len(cdr3_results) if cdr3_results else 0
            base_score += 0.1 * validity_ratio
    
    # Factor 5: Check execution time (too fast might indicate no actual processing)
    if result.execution_time < 1.0 and result_paths:
        # Executed in less than 1 second but has output files - suspicious
        base_score -= 0.05
    elif result.execution_time > 60:
        # Long execution time - likely did real work
        base_score += 0.05
    
    # Clamp to [0.0, 1.0]
    return min(1.0, max(0.0, base_score))


def check_hitl_confirm_node(state: ExecutorState) -> Literal["hitl_confirm", "activate"]:
    """Check if HITL confirmation is needed"""
    pending_hitl = [
        task_id for task_id, request in state.hitl_requests.items()
        if request.get("type") == "result_confirmation" and task_id not in state.hitl_responses
    ]
    
    if pending_hitl:
        return "hitl_confirm"
    else:
        return "activate"


def hitl_confirm_node(state: ExecutorState) -> ExecutorState:
    """
    HITL confirmation node
    
    Request user confirmation whether to continue execution and wait for response
    Supports receiving resume value when resuming from interrupt
    
    Workflow:
    1. First execution: Check if there are unresponded HITL confirmation requests, if so trigger interrupt
    2. Resume execution: Get user confirmation from interrupt's resume value, update status, continue execution
    """
    # Try to get resume value (if resuming execution)
    resume_value = None
    if INTERRUPT_AVAILABLE:
        try:
            resume_value = interrupt()
        except Exception:
            # On first call, interrupt() raises exception (normal behavior)
            resume_value = None
    
    # If there's a resume value, process user confirmation response
    if resume_value is not None:
        # resume_value may be Command object or dict
        if hasattr(resume_value, 'resume'):
            resume_data = resume_value.resume
        elif isinstance(resume_value, dict) and 'resume' in resume_value:
            resume_data = resume_value['resume']
        else:
            resume_data = resume_value
        
        if isinstance(resume_data, dict) and resume_data.get("type") == "response_confirmation":
            responses = resume_data.get("responses", {})
            for task_id, response_data in responses.items():
                if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                    _record_hitl_response(state, task_id, response_data)
                    # Update user choice
                    if task_id in state.task_results:
                        result = state.task_results[task_id]
                        result.user_continue = response_data.get("continue", True)
                        # Mark as completed (regardless of user choice to continue or stop)
                        state.task_status_map[task_id] = ExecutorTaskStatus.COMPLETED
                        print(f"  ✓ Task {task_id} user confirmation: {'continue' if result.user_continue else 'stop'}")
    
    pending_requests = [
        (task_id, request) for task_id, request in state.hitl_requests.items()
        if request.get("type") == "result_confirmation" and task_id not in state.hitl_responses
    ]
    
    if not pending_requests:
        return state
    
    # Also check parent_state.hitl_status as fallback (in case resume_value wasn't passed correctly)
    # This is important because resume_executor_after_interrupt may not correctly pass resume_value through interrupt()
    # First, try to update parent_state.hitl_status from thread-scoped storage if available
    if state.parent_state and state.thread_id and state.thread_id in _parent_state_by_thread:
        parent_state = _parent_state_by_thread[state.thread_id]
        if parent_state.hitl_status and (not hasattr(state.parent_state, 'hitl_status') or not state.parent_state.hitl_status or state.parent_state.hitl_status != parent_state.hitl_status):
            print(f"  [DEBUG] Updating parent_state.hitl_status from thread-scoped storage for hitl_confirm")
            state.parent_state.hitl_status = parent_state.hitl_status
    
    print(f"  [DEBUG] Checking parent_state.hitl_status: {state.parent_state.hitl_status if (state.parent_state and hasattr(state.parent_state, 'hitl_status')) else 'No parent_state or hitl_status'}")
    if state.parent_state and hasattr(state.parent_state, 'hitl_status') and state.parent_state.hitl_status:
        try:
            # Ensure using global json module
            import json as json_module
            hitl_data = json_module.loads(state.parent_state.hitl_status)
            print(f"  [DEBUG] Parsed hitl_data from parent_state.hitl_status: type={hitl_data.get('type')}, responses={list(hitl_data.get('responses', {}).keys())}")
            if hitl_data.get("type") == "response_confirmation":
                responses = hitl_data.get("responses", {})
                for task_id, response_data in responses.items():
                    if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                        _record_hitl_response(state, task_id, response_data)
                        # Update user choice
                        if task_id in state.task_results:
                            result = state.task_results[task_id]
                            result.user_continue = response_data.get("continue", True)
                            state.task_status_map[task_id] = ExecutorTaskStatus.COMPLETED
                            print(f"  ✓ Task {task_id} user confirmation: {'continue' if result.user_continue else 'stop'}")
        except Exception as e:
            print(f"  ⚠ Failed to parse HITL response: {e}")
    
    # If there are still unresponded requests, set HITL request info and trigger interrupt
    remaining_requests = [
        task_id for task_id in state.hitl_requests.keys()
        if task_id not in state.hitl_responses
    ]
    if remaining_requests and state.parent_state:
        hitl_messages = []
        for task_id in remaining_requests:
            request = state.hitl_requests[task_id]
            if request.get("type") == "result_confirmation":
                hitl_messages.append({
                    "task_id": task_id,
                    "message": request["message"],
                    "type": request["type"],
                    "result": request.get("result", ""),
                    "reason": request.get("reason", "")
                })
        
        if hitl_messages:
            state.parent_state.hitl_status = json.dumps({
                "type": "result_confirmation",
                "requests": hitl_messages
            }, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"HITL Request: User needs to confirm whether to continue")
            print(f"{'='*60}")
            for msg in hitl_messages:
                print(f"Task {msg['task_id']}: {msg['message']}")
            print(f"{'='*60}\n")
            
            # Trigger interrupt, pause execution and wait for user confirmation
            if INTERRUPT_AVAILABLE:
                try:
                    interrupt({
                        "type": "result_confirmation",
                        "requests": hitl_messages,
                        "message": "Waiting for user confirmation whether to continue"
                    })
                except Exception as e:
                    # interrupt will raise exception (this is normal behavior)
                    # LangGraph will catch and save state
                    raise
            else:
                print("  ⚠ Note: interrupt functionality unavailable, will use state marker method")
    
    return state


def activate_dependent_tasks_node(state: ExecutorState) -> ExecutorState:
    """
    Activate dependent tasks node
    
    Check all tasks waiting for dependencies, if all dependencies are completed, mark as ready
    If dependent tasks failed, also mark as ready (allow continued execution, but may fail)
    """
    waiting_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.WAITING_DEPENDENCY
    ]
    
    activated_count = 0
    for task in waiting_tasks:
        # Check if all dependencies are completed or failed
        # Note: Even if dependencies failed, allow subsequent tasks to continue (may be for error handling or cleanup)
        all_deps_finished = all(
            dep_id in state.task_results and
            state.task_results[dep_id].status in [
                ExecutorTaskStatus.COMPLETED,
                ExecutorTaskStatus.FAILED
            ]
            for dep_id in task.dependencies
        )
        
        if all_deps_finished:
            # Check if any dependencies failed
            has_failed_deps = any(
                dep_id in state.task_results and
                state.task_results[dep_id].status == ExecutorTaskStatus.FAILED
                for dep_id in task.dependencies
            )
            
            if has_failed_deps:
                # If dependencies failed, still activate task, but log warning
                print(f"  ⚠ Task {task.task_id} has failed dependencies, but will still activate (allow error handling)")
            
            state.task_status_map[task.task_id] = ExecutorTaskStatus.READY
            activated_count += 1
            print(f"  ✓ Task {task.task_id} dependencies completed, activated")
    
    # Update loop counter
    if activated_count > 0:
        # If tasks were activated, reset counter
        state.activate_iteration_count = 0
        print(f"✓ Activated {activated_count} new tasks")
    else:
        # If no tasks were activated, increment counter
        state.activate_iteration_count += 1
        if waiting_tasks:
            print(f"  ⚠ {len(waiting_tasks)} tasks still waiting for dependencies (consecutive activation iterations: {state.activate_iteration_count}/{state.max_activate_iterations})")
            # Print dependency status for each waiting task for debugging
            for task in waiting_tasks[:3]:  # Only print first 3 to avoid log being too long
                dep_statuses = []
                for dep_id in task.dependencies:
                    if dep_id in state.task_results:
                        dep_statuses.append(f"{dep_id}:{state.task_results[dep_id].status.value}")
                    else:
                        dep_statuses.append(f"{dep_id}:not_executed")
                print(f"    Task {task.task_id} dependency status: {', '.join(dep_statuses)}")
    
    return state


def check_completion_node(state: ExecutorState) -> Literal["infer_params", "activate", "summary"]:
    """
    Check completion status node
    
    Determine if all tasks are completed and decide next action
    
    Return logic:
    - "summary": All tasks completed or failed, or no ready tasks and no running tasks (cannot continue execution)
    - "infer_params": There are ready tasks to execute
    - "activate": No ready tasks, but there may be tasks waiting for dependencies that need activation, and there are running tasks (may activate new dependent tasks later)
    """
    # Check if all tasks are completed or failed
    all_completed = all(
        state.task_status_map.get(task.task_id) in [
            ExecutorTaskStatus.COMPLETED,
            ExecutorTaskStatus.FAILED
        ]
        for task in state.subtasks
    )
    
    if all_completed:
        print(f"  ✓ All tasks completed, preparing to summarize results")
        return "summary"
    
    # Check if there are still ready tasks (not running)
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if ready_tasks:
        # Have new ready tasks, reset loop counter
        state.activate_iteration_count = 0
        print(f"  ✓ {len(ready_tasks)} ready tasks, starting parameter inference")
        return "infer_params"
    
    # Check if there are running tasks
    running_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.RUNNING
        or task.task_id in state.running_tasks
    ]
    
    # Check if there are tasks waiting for dependencies
    waiting_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.WAITING_DEPENDENCY
    ]
    
    # Key logic: If no ready tasks and no running tasks, there won't be new task completions, and no new dependent tasks will be activated
    # Should end execution at this point
    if not running_tasks:
        # No running tasks, means no new tasks will complete
        # If there are still tasks waiting for dependencies, their dependencies will never complete (may be deadlock or dependency failure)
        if waiting_tasks:
            # Check if it's a deadlock: all waiting tasks depend on each other, or dependency chain forms a cycle
            deadlock_detected = _detect_deadlock(waiting_tasks, state)
            
            if deadlock_detected:
                print(f"  ⚠ Deadlock detected: {len(waiting_tasks)} tasks form circular dependencies, marking as failed and ending")
                # Mark all deadlocked tasks as failed
                for task in waiting_tasks:
                    state.task_status_map[task.task_id] = ExecutorTaskStatus.FAILED
                    if task.task_id not in state.task_results:
                        state.task_results[task.task_id] = TaskExecutionResult(
                            task_id=task.task_id,
                            status=ExecutorTaskStatus.FAILED,
                            execution_mode="",
                            error="Deadlock detected: task dependency chain forms a cycle"
                        )
            else:
                # Check if all dependent tasks have failed
                all_deps_failed = True
                for task in waiting_tasks:
                    deps_status = [
                        state.task_results.get(dep_id, TaskExecutionResult(
                            task_id=dep_id,
                            status=ExecutorTaskStatus.FAILED,
                            execution_mode=""
                        )).status
                        for dep_id in task.dependencies
                    ]
                    # If all dependencies failed, mark as failed
                    if all(s == ExecutorTaskStatus.FAILED for s in deps_status):
                        state.task_status_map[task.task_id] = ExecutorTaskStatus.FAILED
                        if task.task_id not in state.task_results:
                            state.task_results[task.task_id] = TaskExecutionResult(
                                task_id=task.task_id,
                                status=ExecutorTaskStatus.FAILED,
                                execution_mode="",
                            error="All dependent tasks have failed, cannot continue execution"
                        )
                    else:
                        all_deps_failed = False
                
                if all_deps_failed:
                    print(f"  ⚠ All dependencies of tasks waiting for dependencies have failed, marking as failed and ending")
                else:
                    print(f"  ⚠ No ready tasks and no running tasks, but {len(waiting_tasks)} tasks waiting for dependencies, their dependencies may never complete, marking as failed and ending")
                    # Mark remaining tasks waiting for dependencies as failed
                    for task in waiting_tasks:
                        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.WAITING_DEPENDENCY:
                            state.task_status_map[task.task_id] = ExecutorTaskStatus.FAILED
                            if task.task_id not in state.task_results:
                                state.task_results[task.task_id] = TaskExecutionResult(
                                    task_id=task.task_id,
                                    status=ExecutorTaskStatus.FAILED,
                                    execution_mode="",
                                    error="Dependent tasks cannot complete, causing execution to stop"
                                )
            return "summary"
        else:
            # No ready tasks, no running tasks, no tasks waiting for dependencies, but tasks not all completed
            # Check if there are uninitialized tasks or tasks in other states
            uninitialized_tasks = [
                task for task in state.subtasks
                if task.task_id not in state.task_status_map
            ]
            
            other_state_tasks = [
                task for task in state.subtasks
                if state.task_status_map.get(task.task_id) in [
                    ExecutorTaskStatus.WAITING_HITL_PARAMS,
                    ExecutorTaskStatus.WAITING_HITL_CONFIRM
                ]
            ]
            
            if uninitialized_tasks:
                print(f"  ⚠ Detected {len(uninitialized_tasks)} uninitialized tasks, marking as failed and ending")
                for task in uninitialized_tasks:
                    state.task_status_map[task.task_id] = ExecutorTaskStatus.FAILED
                    if task.task_id not in state.task_results:
                        state.task_results[task.task_id] = TaskExecutionResult(
                            task_id=task.task_id,
                            status=ExecutorTaskStatus.FAILED,
                            execution_mode="",
                            error="Task not properly initialized"
                        )
                return "summary"
            elif other_state_tasks:
                # Have HITL waiting tasks, but no running tasks, means these tasks may never be responded to
                # For safety, mark as failed and end
                print(f"  ⚠ Detected {len(other_state_tasks)} HITL waiting tasks, but no running tasks, marking as failed and ending")
                for task in other_state_tasks:
                    state.task_status_map[task.task_id] = ExecutorTaskStatus.FAILED
                    if task.task_id not in state.task_results:
                        state.task_results[task.task_id] = TaskExecutionResult(
                            task_id=task.task_id,
                            status=ExecutorTaskStatus.FAILED,
                            execution_mode="",
                            error="HITL wait timeout, cannot continue execution"
                        )
                return "summary"
            else:
                # May be state inconsistency, force end
                print(f"  ⚠ No ready tasks, no running tasks, no tasks waiting for dependencies, but tasks not all completed, may be state inconsistency, forcing end")
                return "summary"
    
    # Have running tasks, check if there are tasks waiting for dependencies that can be activated
    if waiting_tasks:
        # Check if these waiting tasks can actually be activated
        # Note: Even if dependencies failed, allow activation (activate_dependent_tasks_node will handle)
        can_activate = False
        for task in waiting_tasks:
            all_deps_finished = all(
                dep_id in state.task_results and
                state.task_results[dep_id].status in [
                    ExecutorTaskStatus.COMPLETED,
                    ExecutorTaskStatus.FAILED
                ]
                for dep_id in task.dependencies
            )
            if all_deps_finished:
                can_activate = True
                break
        
        if can_activate:
            print(f"  ✓ Tasks waiting for dependencies can be activated (currently {len(running_tasks)} running tasks)")
            return "activate"
        else:
            # All waiting tasks cannot be activated, but have running tasks, continue waiting
            print(f"  ⚠ {len(waiting_tasks)} tasks waiting for dependencies, but dependencies not yet completed, continue waiting (currently {len(running_tasks)} running tasks)")
            return "activate"
    else:
        # No tasks waiting for dependencies, but have running tasks, continue waiting
        print(f"  ⚠ No tasks waiting for dependencies, but {len(running_tasks)} running tasks, continue waiting")
        return "activate"


def summary_results_node(state: ExecutorState) -> ExecutorState:
    """
    Summary results node
    
    Summarize execution results of all tasks, including:
    1. Task-level summary
    2. Group-level summary (if there are parallel task groups)
    """
    print(f"\n{'='*60}")
    print(f"Execution Completion Summary")
    print(f"{'='*60}")
    print(f"Total tasks: {state.total_tasks}")
    print(f"Completed: {state.completed_count}")
    print(f"Failed: {state.failed_count}")
    
    # Group-level summary
    if state.parallel_task_groups:
        print(f"\nParallel Task Groups Summary:")
        print(f"{'='*60}")
        
        for group_id, group in state.parallel_task_groups.items():
            # Get all tasks in group
            group_tasks = []
            if isinstance(group, dict):
                group_tasks = group.get('subtasks', [])
            elif hasattr(group, 'subtasks'):
                group_tasks = group.subtasks
            
            if not group_tasks:
                continue
            
            # Count task statuses in group
            group_completed = 0
            group_failed = 0
            group_results = []
            
            for task in group_tasks:
                task_id = task.task_id if hasattr(task, 'task_id') else task.get('task_id', '')
                result = state.task_results.get(task_id)
                if result:
                    if result.status == ExecutorTaskStatus.COMPLETED:
                        group_completed += 1
                    elif result.status == ExecutorTaskStatus.FAILED:
                        group_failed += 1
                    group_results.append({
                        'task_id': task_id,
                        'status': result.status.value,
                        'output': result.output,
                        'error': result.error
                    })
            
            print(f"\nGroup {group_id}:")
            print(f"  Task count: {len(group_tasks)}")
            print(f"  Completed: {group_completed}")
            print(f"  Failed: {group_failed}")
            
            # If there are failed tasks, show detailed information
            if group_failed > 0:
                print(f"  Failed task details:")
                for task_result in group_results:
                    if task_result['status'] == ExecutorTaskStatus.FAILED.value:
                        print(f"    - {task_result['task_id']}: {task_result['error'][:100] if task_result['error'] else 'Unknown error'}")
    
    print(f"{'='*60}\n")
    
    # =================================================================
    # Update todo-list.md with final task statuses
    # This ensures the todo-list reflects the final execution state
    # =================================================================
    try:
        from .todolist_generator import update_task_status_in_todolist, TodoTaskStatus
        
        # Get sandbox directory from parent_state or state
        sandbox_dir = None
        if state.parent_state:
            sandbox_dir = getattr(state.parent_state, 'sandbox_data_dir', None) or state.sandbox_dir
        
        if sandbox_dir:
            # Map ExecutorTaskStatus to TodoTaskStatus
            status_map = {
                ExecutorTaskStatus.COMPLETED: TodoTaskStatus.COMPLETED,
                ExecutorTaskStatus.FAILED: TodoTaskStatus.FAILED,
                ExecutorTaskStatus.READY: TodoTaskStatus.PENDING,
                ExecutorTaskStatus.RUNNING: TodoTaskStatus.IN_PROGRESS,
                ExecutorTaskStatus.WAITING_DEPENDENCY: TodoTaskStatus.PENDING,
                ExecutorTaskStatus.WAITING_HITL_PARAMS: TodoTaskStatus.PENDING,
                ExecutorTaskStatus.WAITING_HITL_CONFIRM: TodoTaskStatus.PENDING,
            }
            
            updated_count = 0
            for task in state.subtasks:
                executor_status = state.task_status_map.get(task.task_id)
                if executor_status:
                    todo_status = status_map.get(executor_status, TodoTaskStatus.PENDING)
                    
                    # Get result and error from task_results
                    task_result = state.task_results.get(task.task_id)
                    result_dict = None
                    error_msg = None
                    
                    if task_result:
                        if task_result.output:
                            if isinstance(task_result.output, dict):
                                result_dict = task_result.output
                            else:
                                result_dict = {"output": str(task_result.output)}
                        if task_result.error:
                            error_msg = task_result.error
                    
                    # Update task status in todo-list.md
                    if update_task_status_in_todolist(
                        sandbox_dir=sandbox_dir,
                        task_id=task.task_id,
                        status=todo_status,
                        result=result_dict,
                        error=error_msg
                    ):
                        updated_count += 1
            
            if updated_count > 0:
                print(f"  📋 Updated {updated_count} task statuses in todo-list.md")
    except Exception as e:
        print(f"  ⚠️ Error updating todo-list.md: {e}")
        import traceback
        traceback.print_exc()
    
    return state


# ===================== Build Executor Subgraph =====================

def build_executor_subgraph():
    """Build Executor subgraph (using LangGraph 1.0+ features)"""
    graph = StateGraph(ExecutorState)
    
    # Add nodes
    graph.add_node("initialize", initialize_tasks_node)
    graph.add_node("infer_params", infer_parameters_node)
    graph.add_node("check_hitl_params", check_hitl_params_node)  # Conditional node
    graph.add_node("hitl_params", hitl_params_node)
    graph.add_node("execute", execute_tasks_node)
    graph.add_node("analyze_results", analyze_results_node)
    graph.add_node("check_hitl_confirm", check_hitl_confirm_node)  # Conditional node
    graph.add_node("hitl_confirm", hitl_confirm_node)
    graph.add_node("activate", activate_dependent_tasks_node)
    graph.add_node("summary", summary_results_node)
    
    # Define flow
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "infer_params")
    
    # After parameter inference, check if HITL is needed
    graph.add_conditional_edges(
        "infer_params",
        check_hitl_params_node,
        {
            "hitl_params": "hitl_params",
            "execute": "execute"  # Direct execution, resource checking done inside execute node
        }
    )
    
    # After HITL parameter request, directly execute (resource checking done inside execute node)
    graph.add_edge("hitl_params", "execute")
    
    # After execution, analyze results
    graph.add_edge("execute", "analyze_results")
    
    # After result analysis, check if HITL confirmation is needed
    graph.add_conditional_edges(
        "analyze_results",
        check_hitl_confirm_node,
        {
            "hitl_confirm": "hitl_confirm",
            "activate": "activate"
        }
    )
    
    # After HITL confirmation, activate dependent tasks
    graph.add_edge("hitl_confirm", "activate")
    
    # After activating dependencies, check completion status
    # Note: Direct routing here, no need for additional check_completion node
    graph.add_conditional_edges(
        "activate",
        check_completion_node,
        {
            "infer_params": "infer_params",
            "activate": "activate",  # If there are still dependencies that can be activated, continue activating
            "summary": "summary"  # All tasks completed or cannot continue, end
        }
    )
    
    graph.add_edge("summary", END)
    
    # Use MemorySaver as checkpoint (can be replaced with persistent storage)
    if MemorySaver:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    else:
        # If MemorySaver unavailable, use default compilation
        return graph.compile()


# ===================== State Mapping Functions =====================

def executor_input_mapper(global_state: GlobalState) -> ExecutorState:
    """
    Map main graph state to Executor subgraph state
    
    Args:
        global_state: Main graph global state
    
    Returns:
        Executor subgraph state
    """
    # Use model_construct to bypass strict validation, directly use SubTask objects
    # This avoids Pydantic v2 strict validation issues with nested models
    executor_state = ExecutorState.model_construct(
        subtasks=global_state.subtasks,
        parallel_task_groups=global_state.parallel_task_groups,
        sandbox_dir=global_state.sandbox_dir,
        use_react_executor=global_state.use_react_executor,
        react_max_steps=global_state.react_max_steps,
        parent_state=global_state
    )
    
    return executor_state


def executor_output_mapper(executor_state: Union[ExecutorState, Dict], global_state: GlobalState) -> GlobalState:
    """
    Map Executor subgraph state back to main graph state
    
    Args:
        executor_state: Executor subgraph state (subtasks already merged with parallel groups)
                       Can be ExecutorState object or dict (from LangGraph invoke)
        global_state: Main graph global state
    
    Returns:
        Updated main graph state
    """
    # IMPORTANT: Executor merges parallel task groups into subtasks during initialization
    # So executor_state.subtasks contains ALL tasks (serial + parallel)
    # We should update global_state.subtasks to match executor_state.subtasks to avoid counting issues
    
    # Handle case where executor_state might be a dict (from LangGraph invoke)
    if isinstance(executor_state, dict):
        executor_subtasks = executor_state.get("subtasks", [])
        task_results_raw = executor_state.get("task_results", {})
        total_tasks = executor_state.get("total_tasks", 0)
        completed_count = executor_state.get("completed_count", 0)
        failed_count = executor_state.get("failed_count", 0)
        hitl_requests = executor_state.get("hitl_requests", {})
        hitl_responses = executor_state.get("hitl_responses", {})
    else:
        executor_subtasks = executor_state.subtasks if hasattr(executor_state, 'subtasks') else []
        task_results_raw = executor_state.task_results if hasattr(executor_state, 'task_results') else {}
        total_tasks = executor_state.total_tasks if hasattr(executor_state, 'total_tasks') else 0
        completed_count = executor_state.completed_count if hasattr(executor_state, 'completed_count') else 0
        failed_count = executor_state.failed_count if hasattr(executor_state, 'failed_count') else 0
        hitl_requests = executor_state.hitl_requests if hasattr(executor_state, 'hitl_requests') else {}
        hitl_responses = executor_state.hitl_responses if hasattr(executor_state, 'hitl_responses') else {}
    
    # Update global_state.subtasks to match executor's merged subtasks
    # This ensures consistency between executor_state.total_tasks and test's all_tasks count
    if executor_subtasks:
        # Create a mapping of task_id -> task for quick lookup
        executor_task_map = {}
        for task in executor_subtasks:
            if hasattr(task, 'task_id'):
                executor_task_map[task.task_id] = task
            elif isinstance(task, dict):
                executor_task_map[task.get('task_id', '')] = task
        
        # Update existing tasks in global_state.subtasks with executor results
        updated_subtasks = []
        for task in global_state.subtasks:
            if task.task_id in executor_task_map:
                # Update with executor's version (which may have been merged from parallel groups)
                updated_task = executor_task_map[task.task_id]
                if hasattr(updated_task, 'task_id'):
                    updated_subtasks.append(updated_task)
                else:
                    updated_subtasks.append(task)  # Keep original if can't update
            else:
                updated_subtasks.append(task)
        
        # Add any tasks from executor that are not in global_state.subtasks
        # (these are tasks that were in parallel groups and merged)
        for executor_task in executor_subtasks:
            task_id = executor_task.task_id if hasattr(executor_task, 'task_id') else executor_task.get('task_id', '')
            if not any(t.task_id == task_id for t in updated_subtasks):
                if hasattr(executor_task, 'task_id'):
                    updated_subtasks.append(executor_task)
        
        global_state.subtasks = updated_subtasks
    
    # Update task results (including tasks in subtasks and parallel task groups)
    all_tasks_to_update = list(global_state.subtasks)
    
    # Extract all tasks from parallel task groups (for backward compatibility)
    for group_id, group in global_state.parallel_task_groups.items():
        if hasattr(group, 'subtasks') and group.subtasks:
            for task in group.subtasks:
                if not any(t.task_id == task.task_id for t in all_tasks_to_update):
                    all_tasks_to_update.append(task)
    
    # Update results for all tasks
    for task in all_tasks_to_update:
        task_result = task_results_raw.get(task.task_id)
        if task_result:
            # Handle both object and dict result
            if hasattr(task_result, 'status'):
                is_completed = task_result.status == ExecutorTaskStatus.COMPLETED
                status_value = task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status)
                output = task_result.output if hasattr(task_result, 'output') else None
                execution_mode = task_result.execution_mode if hasattr(task_result, 'execution_mode') else None
                code = task_result.code if hasattr(task_result, 'code') else None
                confidence = task_result.confidence_score if hasattr(task_result, 'confidence_score') else None
                summary = task_result.result_summary if hasattr(task_result, 'result_summary') else None
                error = task_result.error if hasattr(task_result, 'error') else None
                error_type = task_result.error_type if hasattr(task_result, 'error_type') else None
                error_cat = task_result.error_category.value if hasattr(task_result, 'error_category') and task_result.error_category else None
                exec_time = int(task_result.execution_time * 1000) if hasattr(task_result, 'execution_time') and task_result.execution_time else None
                failure_analysis = task_result.failure_analysis if hasattr(task_result, 'failure_analysis') else None
                suggestions = task_result.suggestions if hasattr(task_result, 'suggestions') else None
            else:
                status_raw = task_result.get('status', 'unknown')
                is_completed = status_raw == ExecutorTaskStatus.COMPLETED.value if isinstance(status_raw, str) else False
                status_value = status_raw.value if hasattr(status_raw, 'value') else str(status_raw)
                output = task_result.get('output')
                execution_mode = task_result.get('execution_mode')
                code = task_result.get('code')
                confidence = task_result.get('confidence_score')
                summary = task_result.get('result_summary')
                error = task_result.get('error')
                error_type = task_result.get('error_type')
                error_cat = task_result.get('error_category')
                exec_time = task_result.get('execution_time_ms')
                failure_analysis = task_result.get('failure_analysis')
                suggestions = task_result.get('suggestions')
            
            if is_completed:
                # Update task result
                if not task.result:
                    task.result = {}
                if isinstance(task.result, dict):
                    task.result["execution_result"] = output
                    task.result["execution_mode"] = execution_mode
                    task.result["code"] = code
                    task.result["confidence_score"] = confidence
                    task.result["execution_summary"] = summary
                
                # Mark task as completed
                global_state.completed_tasks[task.task_id] = task
    
    # Build task_results dict for merged_result
    formatted_task_results = {}
    for task_id, result in task_results_raw.items():
        if hasattr(result, 'status'):
            formatted_task_results[task_id] = {
                "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                "execution_mode": result.execution_mode if hasattr(result, 'execution_mode') else None,
                "error": result.error if hasattr(result, 'error') else None,
                "error_type": result.error_type if hasattr(result, 'error_type') else None,
                "error_category": result.error_category.value if hasattr(result, 'error_category') and result.error_category else None,
                "execution_time_ms": int(result.execution_time * 1000) if hasattr(result, 'execution_time') and result.execution_time else None,
                "confidence_score": result.confidence_score if hasattr(result, 'confidence_score') else None,
                "failure_analysis": result.failure_analysis if hasattr(result, 'failure_analysis') else None,
                "suggestions": result.suggestions if hasattr(result, 'suggestions') else None,
                "summary": result.result_summary if hasattr(result, 'result_summary') else None
            }
        else:
            formatted_task_results[task_id] = {
                "status": result.get('status', 'unknown'),
                "execution_mode": result.get('execution_mode'),
                "error": result.get('error'),
                "error_type": result.get('error_type'),
                "error_category": result.get('error_category'),
                "execution_time_ms": result.get('execution_time_ms'),
                "confidence_score": result.get('confidence_score'),
                "failure_analysis": result.get('failure_analysis'),
                "suggestions": result.get('suggestions'),
                "summary": result.get('result_summary') or result.get('summary')
            }
    
    # Update summary results
    global_state.merged_result["executor_results"] = {
        "total_tasks": total_tasks,
        "completed": completed_count,
        "failed": failed_count,
        "task_results": formatted_task_results
    }
    
    # Update HITL status (if any)
    if hitl_requests:
        pending_hitl = [
            task_id for task_id in hitl_requests.keys()
            if task_id not in hitl_responses
        ]
        if pending_hitl:
            global_state.hitl_status = json.dumps({
                "type": "request",
                "requests": [
                    hitl_requests[task_id]
                    for task_id in pending_hitl
                ]
            }, ensure_ascii=False)
    
    return global_state


# ===================== Executor Subgraph Execution Wrapper Function (Supports Interrupt) =====================

def execute_executor_with_interrupt_support(
    executor_graph,
    initial_state: ExecutorState,
    thread_id: str = "default",
    resume_value: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Execute Executor subgraph with interrupt detection and resume support
    
    Args:
        executor_graph: Compiled Executor subgraph
        initial_state: Initial state
        thread_id: Thread ID (for checkpoint)
        resume_value: Resume value (if resuming execution)
    
    Returns:
        Dictionary containing execution result and interrupt information:
        {
            "result": ExecutorState,  # Execution result
            "interrupted": bool,  # Whether interrupted
            "interrupt_data": Any,  # Interrupt data (if any)
            "needs_resume": bool  # Whether resume is needed
        }
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # If resuming execution, use Command(resume=...)
    if resume_value is not None and INTERRUPT_AVAILABLE and Command is not None:
        input_data = Command(resume=resume_value)
    else:
        # First execution, use normal state
        # Save parent_state reference so it can be restored in nodes
        saved_parent_state = None
        if isinstance(initial_state, dict):
            input_data = dict(initial_state)
            input_data["thread_id"] = thread_id
        else:
            # Save parent_state reference (because it will be excluded)
            saved_parent_state = getattr(initial_state, 'parent_state', None)
            object.__setattr__(initial_state, "thread_id", thread_id)
            input_data = initial_state.model_dump(exclude={'parent_state'}, mode='json')
    
    # Use stream() to detect interrupts
    interrupted = False
    interrupt_data = None
    final_result = None
    
    # Store parent_state in thread-scoped storage so nodes can access it
    if saved_parent_state is not None:
        _parent_state_by_thread[thread_id] = saved_parent_state
    print(f"  🔍 [execute_executor] Saved parent_state to thread storage: {saved_parent_state is not None}")
    
    try:
        # Use stream to execute step by step, can detect interrupts
        # LangGraph interrupt mechanism:
        # - When interrupt() is called, it raises GraphInterrupt exception
        # - LangGraph catches this exception, saves state, and returns special format in stream
        # - Interrupt info may be in chunk's "__interrupt__" field, or propagated as exception
        
        print(f"  🔍 [execute_executor] Starting stream execution, config: {config}")
        print(f"  🔍 [execute_executor] saved_parent_state exists: {saved_parent_state is not None}")
        chunk_count = 0
        try:
            stream_iter = executor_graph.stream(input_data, config=config)
        except Exception as stream_init_error:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"  ✗ [execute_executor] Failed to initialize stream: {type(stream_init_error).__name__}: {stream_init_error}")
            print(f"  完整错误堆栈:\n{error_traceback}")
            # Re-raise with full context
            raise RuntimeError(f"Failed to initialize executor stream: {stream_init_error}") from stream_init_error
        
        for chunk in stream_iter:
            chunk_count += 1
            print(f"  🔍 [execute_executor] Received chunk #{chunk_count}: keys={list(chunk.keys()) if isinstance(chunk, dict) else 'not dict'}")
            
            # Restore parent_state in each chunk (if exists)
            # Because parent_state is excluded, it won't be passed in state, need manual restoration
            for key, value in chunk.items():
                if isinstance(value, dict):
                    try:
                        state_obj = ExecutorState.model_validate(value)
                        # Restore parent_state (get from module variable)
                        if saved_parent_state is not None:
                            object.__setattr__(state_obj, 'parent_state', saved_parent_state)
                        final_result = state_obj
                    except:
                        pass
            # Check if there's an interrupt (LangGraph may use different field names)
            # Possible formats:
            # 1. chunk["__interrupt__"] - direct interrupt field
            # 2. chunk itself contains interrupt info
            # 3. Exception caught and included in chunk
            
            # Check all possible keys
            chunk_keys = list(chunk.keys()) if isinstance(chunk, dict) else []
            
            # Check if there's interrupt field
            if "__interrupt__" in chunk:
                interrupted = True
                interrupt_obj = chunk["__interrupt__"]
                print(f"  ✓ Interrupt detected (via __interrupt__ field)")
                
                # Extract actual value from Interrupt object (safely handle None)
                try:
                    if interrupt_obj is None:
                        interrupt_data = None
                    elif hasattr(interrupt_obj, 'value'):
                        interrupt_data = getattr(interrupt_obj, 'value', None)
                        if interrupt_data is not None:
                            print(f"  🔍 [execute_executor] Extracted value from Interrupt object: type={type(interrupt_data)}")
                    elif isinstance(interrupt_obj, dict) and 'value' in interrupt_obj:
                        interrupt_data = interrupt_obj.get('value')
                        # If value itself is Interrupt object, continue extracting (safely)
                        if interrupt_data is not None and hasattr(interrupt_data, 'value'):
                            interrupt_data = getattr(interrupt_data, 'value', None)
                    else:
                        interrupt_data = interrupt_obj
                except Exception as extract_e:
                    print(f"  ⚠ [execute_executor] Error extracting interrupt value: {extract_e}")
                    interrupt_data = interrupt_obj if interrupt_obj is not None else None
                
                # Get current state (state before interrupt)
                for key, value in chunk.items():
                    if key != "__interrupt__":
                        if isinstance(value, dict):
                            try:
                                final_result = ExecutorState.model_validate(value)
                                # Restore parent_state
                                if saved_parent_state is not None:
                                    object.__setattr__(final_result, 'parent_state', saved_parent_state)
                            except:
                                pass
                break
            elif isinstance(chunk, dict) and any("interrupt" in str(k).lower() for k in chunk_keys):
                # Check if there are keys containing "interrupt"
                for key, value in chunk.items():
                    if "interrupt" in str(key).lower():
                        interrupted = True
                        interrupt_data = value
                        print(f"  ✓ Interrupt detected (via {key} field)")
                        break
                if interrupted:
                    break
            
            # Normal state updates
            for key, value in chunk.items():
                if isinstance(value, dict):
                    try:
                        final_result = ExecutorState.model_validate(value)
                        # Restore parent_state (because excluded, won't be passed in state)
                        if saved_parent_state is not None:
                            object.__setattr__(final_result, 'parent_state', saved_parent_state)
                    except:
                        pass
        
        # If no interrupt, get final result
        if not interrupted and final_result is None:
            # Use invoke to get final result
            try:
                output = executor_graph.invoke(input_data, config=config)
                if isinstance(output, dict):
                    final_result = ExecutorState.model_validate(output)
                else:
                    final_result = output
            except Exception as invoke_e:
                # If invoke also raises exception, may be interrupt
                if "interrupt" in str(invoke_e).lower() or "GraphInterrupt" in str(type(invoke_e).__name__):
                    interrupted = True
                    interrupt_data = getattr(invoke_e, 'value', None) or str(invoke_e)
                    print(f"  ✓ Interrupt detected (via invoke exception)")
                else:
                    raise
        
    except Exception as e:
        # If interrupt raises exception, this is normal behavior
        # LangGraph will handle this exception in stream
        import traceback
        error_traceback = traceback.format_exc()
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        # Always log full error traceback for debugging
        print(f"  🔍 [execute_executor] Exception caught: {error_type}: {e}")
        print(f"  完整错误堆栈:\n{error_traceback}")
        
        if "interrupt" in error_str or "GraphInterrupt" in error_type:
            interrupted = True
            # Try to extract interrupt info from exception
            interrupt_data = getattr(e, 'value', None)
            if interrupt_data is None:
                # Try to extract from exception message
                if hasattr(e, 'args') and e.args:
                    interrupt_data = e.args[0]
                else:
                    interrupt_data = str(e)
            print(f"  ✓ Interrupt detected (via exception: {error_type})")
            print(f"     Interrupt data: {interrupt_data}")
        else:
            # Check if it's other type of exception but may contain interrupt info
            # Sometimes LangGraph wraps interrupt info in other exceptions
            if hasattr(e, 'value'):
                interrupt_data = e.value
                if interrupt_data and isinstance(interrupt_data, dict):
                    interrupted = True
                    print(f"  ✓ Interrupt detected (via exception.value: {error_type})")
                else:
                    # Other exception, re-raise with full context
                    print(f"  ✗ Unexpected exception: {error_type}: {e}")
                    print(f"  完整错误堆栈:\n{error_traceback}")
                    raise RuntimeError(f"Unexpected exception in executor stream: {e}") from e
            else:
                # Other exception, re-raise with full context
                print(f"  ✗ Unexpected exception: {error_type}: {e}")
                print(f"  完整错误堆栈:\n{error_traceback}")
                raise RuntimeError(f"Unexpected exception in executor stream: {e}") from e
    
    # Ensure interrupt_data is dict format (if tuple or other type, convert to dict)
    if interrupt_data is not None:
        try:
            # First, if interrupt_data is Interrupt object, extract its value (safely)
            if hasattr(interrupt_data, 'value'):
                extracted = getattr(interrupt_data, 'value', None)
                if extracted is not None:
                    interrupt_data = extracted
                    print(f"  🔍 [execute_executor] Extracted value from Interrupt object: type={type(interrupt_data)}")
            
            # If extracted value is still Interrupt object (nested case), continue extracting
            if interrupt_data is not None and hasattr(interrupt_data, 'value'):
                extracted = getattr(interrupt_data, 'value', None)
                if extracted is not None:
                    interrupt_data = extracted
                    print(f"  🔍 [execute_executor] Extracted value from nested Interrupt object: type={type(interrupt_data)}")
            
            if isinstance(interrupt_data, tuple):
                # LangGraph interrupt may return tuple, need to convert to dict
                if len(interrupt_data) == 2:
                    # May be (key, value) format
                    value = interrupt_data[1]
                    # If value is Interrupt object, extract its value (safely)
                    if value is not None and hasattr(value, 'value'):
                        extracted_value = getattr(value, 'value', None)
                        if extracted_value is not None:
                            value = extracted_value
                    interrupt_data = {"key": interrupt_data[0], "value": value}
                elif len(interrupt_data) == 1:
                    # May be single value
                    value = interrupt_data[0]
                    if value is not None and hasattr(value, 'value'):
                        extracted_value = getattr(value, 'value', None)
                        if extracted_value is not None:
                            value = extracted_value
                    interrupt_data = {"value": value}
                else:
                    # Multiple values, convert to dict (safely)
                    interrupt_data = {
                        f"arg_{i}": (getattr(v, 'value', None) if (v is not None and hasattr(v, 'value')) else v) 
                        for i, v in enumerate(interrupt_data)
                    }
                print(f"  🔍 [execute_executor] Converted interrupt_data from tuple to dict: {interrupt_data}")
        except Exception as extract_e:
            print(f"  ⚠ [execute_executor] Error extracting interrupt_data: {extract_e}")
            import traceback
            print(f"  {traceback.format_exc()[:300]}")
            # Keep original interrupt_data if extraction fails
        
        # After try-except, check if interrupt_data is still not a dict
        if interrupt_data is not None and not isinstance(interrupt_data, dict):
            # If other type (like string), convert to dict
            if isinstance(interrupt_data, str):
                # Try to parse JSON string
                try:
                    import json as json_module
                    interrupt_data = json_module.loads(interrupt_data)
                except:
                    interrupt_data = {"value": interrupt_data}
            else:
                interrupt_data = {"value": interrupt_data}
            print(f"  🔍 [execute_executor] Converted interrupt_data to dict: {interrupt_data}")
    
    return {
        "result": final_result,
        "interrupted": interrupted,
        "interrupt_data": interrupt_data,
        "needs_resume": interrupted
    }


def resume_executor_after_interrupt(
    executor_graph,
    thread_id: str,
    resume_value: Any
) -> Dict[str, Any]:
    """
    Resume Executor subgraph execution (after interrupt)
    
    Args:
        executor_graph: Compiled Executor subgraph
        thread_id: Thread ID (must be same as when interrupted)
        resume_value: Resume value (user response)
    
    Returns:
        Execution result dictionary (same format as execute_executor_with_interrupt_support)
    """
    if not INTERRUPT_AVAILABLE or Command is None:
        raise ValueError("interrupt functionality unavailable, cannot resume execution")
    
    # Update parent_state.hitl_status in thread-scoped storage if resume_value contains user response
    if thread_id in _parent_state_by_thread and resume_value and isinstance(resume_value, dict):
        import json as json_module
        try:
            if resume_value.get("type") in ["response_parameters", "response_confirmation"]:
                _parent_state_by_thread[thread_id].hitl_status = json_module.dumps(resume_value, ensure_ascii=False)
                print(f"  [DEBUG] Updated parent_state.hitl_status from resume_value (type: {resume_value.get('type')})")
        except Exception as e:
            print(f"  [WARN] Failed to update parent_state.hitl_status: {e}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Use Command(resume=...) to resume execution
    input_data = Command(resume=resume_value)
    
    # Continue execution
    interrupted = False
    interrupt_data = None
    final_result = None
    
    try:
        for chunk in executor_graph.stream(input_data, config=config):
            if "__interrupt__" in chunk:
                interrupted = True
                interrupt_obj = chunk["__interrupt__"]
                # Extract actual value from Interrupt object (safely handle None)
                try:
                    if interrupt_obj is None:
                        interrupt_data = None
                    elif hasattr(interrupt_obj, 'value'):
                        interrupt_data = getattr(interrupt_obj, 'value', None)
                        # If extracted value is still an Interrupt object, continue extracting
                        if interrupt_data is not None and hasattr(interrupt_data, 'value'):
                            interrupt_data = getattr(interrupt_data, 'value', None)
                    elif isinstance(interrupt_obj, dict) and 'value' in interrupt_obj:
                        interrupt_data = interrupt_obj.get('value')
                        if interrupt_data is not None and hasattr(interrupt_data, 'value'):
                            interrupt_data = getattr(interrupt_data, 'value', None)
                    else:
                        interrupt_data = interrupt_obj
                except Exception as extract_e:
                    print(f"  ⚠ [resume_executor] Error extracting interrupt value: {extract_e}")
                    interrupt_data = interrupt_obj
                break
            else:
                for key, value in chunk.items():
                    if isinstance(value, dict):
                        try:
                            final_result = ExecutorState.model_validate(value)
                        except:
                            pass
        
        if not interrupted and final_result is None:
            output = executor_graph.invoke(input_data, config=config)
            if isinstance(output, dict):
                final_result = ExecutorState.model_validate(output)
            else:
                final_result = output
                
    except Exception as e:
        if "interrupt" in str(e).lower() or "GraphInterrupt" in str(type(e).__name__):
            interrupted = True
            try:
                interrupt_data = getattr(e, 'value', None)
                # If interrupt_data is Interrupt object, extract its value (safely)
                if interrupt_data is not None and hasattr(interrupt_data, 'value'):
                    interrupt_data = getattr(interrupt_data, 'value', None)
                if interrupt_data is None:
                    interrupt_data = str(e)
            except Exception as extract_e:
                print(f"  ⚠ [resume_executor] Error extracting exception value: {extract_e}")
                interrupt_data = str(e)
        else:
            raise
    
    return {
        "result": final_result,
        "interrupted": interrupted,
        "interrupt_data": interrupt_data,
        "needs_resume": interrupted
    }
