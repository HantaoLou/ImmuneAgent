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

from typing import Dict, List, Any, Optional, Literal, Union
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
import json
from pathlib import Path
from enum import Enum

# Module-level variable to store current parent_state
# This allows nodes to access it (because parent_state is excluded during serialization)
_current_parent_state = None
import time

# Import main graph state and task models
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from agent.state import SubTask, TaskStatus, UserTaskType, GlobalState, ParallelTaskGroup

# Module-level variable to store current parent_state
# This allows nodes to access it (because parent_state is excluded during serialization)
_current_parent_state = None

# Import CodeAct subgraph
from agent.nodes.subagents.code_act.graph import (
    build_codeact_subgraph,
    codeact_input_mapper,
    codeact_output_mapper,
    CodeActExecutionMode,
    CodeActState
)

# Import LLM factory
from agent.utils.llm_factory import create_reasoning_advanced_llm, create_reasoning_llm

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
    
    # Parent state reference (for accessing global state and updating HITL state)
    # Note: exclude=True to avoid LangGraph serialization validation failure, but keep in model for node use
    # Use Union type to allow accepting GlobalState instance or None during validation
    parent_state: Optional[GlobalState] = Field(default=None, exclude=True, description="Main graph state reference")
    
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


def classify_error(error: str, error_type: str) -> ErrorCategory:
    """Classify error type"""
    error_lower = error.lower()
    error_type_lower = error_type.lower()
    
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
    analysis = f"Error type: {error_category.value}\n"
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
                    from agent.state import SubTask
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
                        from agent.state import SubTask
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
    
    return state


def infer_parameters_node(state: ExecutorState) -> ExecutorState:
    """
    Parameter inference node (optimized version)
    
    Prioritize using parameter inference results from task_decomposition.
    For already inferred parameters:
    - source_type is DETERMINED: directly use inferred value
    - source_type is FROM_TASK: wait for dependent task completion to get value
    - source_type is USER_REQUIRED: mark as missing parameter, trigger HITL
    For parameters without inference results, use LLM to re-infer (with caching mechanism).
    
    Optimizations:
    1. Batch process parameter inference to reduce LLM calls
    2. Cache inference results to avoid duplicate inference
    3. Prioritize using results from completed dependent tasks
    """
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if not ready_tasks:
        return state
    
    # Get parameter inference results from global_state
    parameter_inference_results = {}
    if state.parent_state and state.parent_state.merged_result:
        parameter_inference_results = state.parent_state.merged_result.get("parameter_inference_results", {})
    
    # Parameter inference cache (avoid duplicate inference for same parameters)
    inference_cache = {}
    
    llm = create_reasoning_llm()
    tools_params_map = _load_tools_params_table()
    
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
        
        if not tools:
            # Tasks without tools, parameters are empty
            result.parameters = {}
            result.missing_parameters = []
            continue
        
        # Collect parameter requirements for all tools
        all_params = {}
        missing_params = []
        
        # First use parameter inference results from task_decomposition
        task_inference = parameter_inference_results.get(task.task_id)
        if task_inference:
            inference_params = task_inference.get("parameters", {})
            for param_name, param_info in inference_params.items():
                source_type = param_info.get("source_type", "")
                
                if source_type == "determined":
                    # Directly use inferred value
                    param_value = param_info.get("value")
                    if param_value is not None:
                        all_params[param_name] = param_value
                        print(f"  ✓ Task {task.task_id} parameter {param_name} using inferred value: {param_value}")
                    else:
                        # Inferred value is empty, mark as missing
                        tool_name = task_inference.get("tool_name", "").split(",")[0].strip() if task_inference.get("tool_name") else ""
                        missing_params.append(f"{tool_name}.{param_name}" if tool_name else param_name)
                
                elif source_type == "from_task":
                    # Parameter value comes from dependent task, need to wait for dependent task completion
                    source_task_id = param_info.get("source_task_id")
                    source_output_key = param_info.get("source_output_key", param_name)
                    
                    # Check if dependent task is completed
                    if source_task_id in state.task_results:
                        dep_result = state.task_results[source_task_id]
                        if dep_result.status == ExecutorTaskStatus.COMPLETED:
                            # Get parameter value from dependent task result
                            if isinstance(dep_result.output, dict):
                                param_value = dep_result.output.get(source_output_key)
                            elif isinstance(dep_result.output, str):
                                # Try to parse string output
                                try:
                                    import json as json_module
                                    output_dict = json_module.loads(dep_result.output)
                                    param_value = output_dict.get(source_output_key)
                                except:
                                    param_value = dep_result.output
                            else:
                                param_value = dep_result.output
                            
                            if param_value is not None:
                                all_params[param_name] = param_value
                                print(f"  ✓ Task {task.task_id} parameter {param_name} obtained from task {source_task_id}: {param_value}")
                            else:
                                tool_name = task_inference.get("tool_name", "").split(",")[0].strip() if task_inference.get("tool_name") else ""
                                missing_params.append(f"{tool_name}.{param_name}" if tool_name else param_name)
                        else:
                            # Dependent task not completed, mark as missing (but this is normal, waiting for dependency)
                            print(f"  ⏳ Task {task.task_id} parameter {param_name} waiting for dependent task {source_task_id} to complete")
                            # Don't add to missing_params, as this is a dependency relationship, not truly missing
                    else:
                        # Dependent task doesn't exist or hasn't been executed, mark as missing
                        tool_name = task_inference.get("tool_name", "").split(",")[0].strip() if task_inference.get("tool_name") else ""
                        missing_params.append(f"{tool_name}.{param_name}" if tool_name else param_name)
                
                elif source_type == "user_required":
                    # Requires user input, mark as missing parameter
                    tool_name = task_inference.get("tool_name", "").split(",")[0].strip() if task_inference.get("tool_name") else ""
                    missing_params.append(f"{tool_name}.{param_name}" if tool_name else param_name)
        
        for tool_item in tools:
            tool_name = None
            if isinstance(tool_item, str):
                tool_name = tool_item
            elif isinstance(tool_item, dict):
                tool_name = tool_item.get("tool_name") or tool_item.get("name", "")
            
            if not tool_name:
                continue
            
            # Find tool parameter definition
            tool_params = tools_params_map.get(tool_name)
            
            # If tool is not in parameter table, use inputs field to infer parameters
            if not tool_params:
                # Infer parameters from inputs field
                for input_param in inputs:
                    if input_param not in all_params:
                        # Try to infer from task description
                        if llm:
                            try:
                                from langchain_core.messages import SystemMessage, HumanMessage
                                inference_prompt = f"""
Please infer parameter value based on the following information:

Task description: {task.content}
Tool name: {tool_name}
Parameter name: {input_param}
Task input list: {inputs}

Please infer the value of this parameter. If it cannot be inferred from the task description, or requires user input (such as file paths, user selections, etc.), return null.

Return JSON format:
{{
    "value": <parameter value or null>,
    "can_infer": <true/false>,
    "reason": "<inference reason or why user input is required>"
}}
"""
                                messages = [
                                    SystemMessage(content="You are a professional parameter inference expert, able to extract parameter values from task descriptions."),
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
                                        all_params[input_param] = param_value
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
            for param in input_params:
                param_name = param.get("name", "")
                param_type = param.get("type", "")
                param_desc = param.get("description", "")
                is_optional = "optional" in param_type.lower() or param_type.startswith("Optional")
                
                if not param_name:
                    continue
                
                # If parameter is already in inference results, skip (already processed)
                if param_name in all_params or any(p.endswith(f".{param_name}") or p == param_name for p in missing_params):
                    continue
                
                # Use LLM to infer parameter value (only for parameters without inference results)
                if llm and param_name not in all_params:
                    try:
                        from langchain_core.messages import SystemMessage, HumanMessage
                        
                        inference_prompt = f"""
Please infer parameter value based on the following information:

Task description: {task.content}
Tool name: {tool_name}
Parameter name: {param_name}
Parameter type: {param_type}
Parameter description: {param_desc}
Task input list: {inputs}

Please infer the value of this parameter. If it cannot be inferred from the task description, or requires user input (such as file paths, user selections, etc.), return null.

Return JSON format:
{{
    "value": <parameter value or null>,
    "can_infer": <true/false>,
    "reason": "<inference reason or why user input is required>"
}}
"""
                        messages = [
                            SystemMessage(content="You are a professional parameter inference expert, able to extract parameter values from task descriptions."),
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
                                all_params[param_name] = param_value
                            elif not is_optional:
                                missing_params.append(f"{tool_name}.{param_name}")
                        else:
                            if not is_optional:
                                missing_params.append(f"{tool_name}.{param_name}")
                    except Exception as e:
                        print(f"  ⚠ Failed to infer parameter {param_name}: {e}")
                        if not is_optional:
                            missing_params.append(f"{tool_name}.{param_name}")
        
        # Update task result
        result.parameters = all_params
        result.missing_parameters = missing_params
        
        # If there are missing parameters, trigger HITL
        if missing_params:
            state.hitl_requests[task.task_id] = {
                "type": "missing_parameters",
                "task_id": task.task_id,
                "task_description": task.content,
                "missing_parameters": missing_params,
                "message": f"Task {task.task_id} requires the following parameters, please provide: {', '.join(missing_params)}"
            }
            # Important: Set task status to WAITING_HITL_PARAMS, not READY
            # This prevents execute_tasks_node from trying to execute these tasks
            state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_HITL_PARAMS
            print(f"  ⚠ Task {task.task_id} requires user-provided parameters: {', '.join(missing_params)}")
            print(f"  → Task status set to WAITING_HITL_PARAMS")
    
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
    global _current_parent_state
    
    print(f"  🔍 [hitl_params_node] Node called")
    
    # If parent_state doesn't exist, try to get from module variable
    if state.parent_state is None and _current_parent_state is not None:
        print(f"  ✓ [hitl_params_node] Restored parent_state from module variable")
        object.__setattr__(state, 'parent_state', _current_parent_state)
    
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
        # resume_value may be Command object or dict
        # If Command object, need to extract resume field
        if hasattr(resume_value, 'resume'):
            resume_data = resume_value.resume
        elif isinstance(resume_value, dict) and 'resume' in resume_value:
            resume_data = resume_value['resume']
        else:
            resume_data = resume_value
        
        if isinstance(resume_data, dict) and resume_data.get("type") == "response_parameters":
            responses = resume_data.get("responses", {})
            for task_id, response_data in responses.items():
                if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                    state.hitl_responses[task_id] = response_data
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
                            for user_param_name, param_value in response_data["parameters"].items():
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
                            
                            # Update parameters
                            result.parameters.update(mapped_parameters)
                            
                            # Remove provided parameters from missing_parameters
                            # missing_parameters format may be "tool_name.param_name" or "param_name"
                            provided_params = set(mapped_parameters.keys())
                            # Also check original parameter names
                            provided_params.update(response_data["parameters"].keys())
                            
                            result.missing_parameters = [
                                p for p in result.missing_parameters
                                if not any(
                                    # Exact match
                                    p == param_name or 
                                    # Suffix match (e.g., "tool.input_file_path" matches "input_file_path")
                                    p.endswith(f".{param_name}") or
                                    # Prefix match (e.g., "input_file_path" matches "tool.input_file_path")
                                    ('.' in p and p.split('.', 1)[1] == param_name) or
                                    # Parameter name mapping match (e.g., "input_file" matches "input_file_path")
                                    (param_name in param_mapping and (
                                        p == param_mapping[param_name] or
                                        p.endswith(f".{param_mapping[param_name]}")
                                    ))
                                    for param_name in provided_params
                                )
                            ]
                        # If all required parameters are provided, mark as ready
                        if not result.missing_parameters:
                            state.task_status_map[task_id] = ExecutorTaskStatus.READY
                            print(f"  ✓ Task {task_id} has all required parameters, marked as ready")
    
    # First check if there are HITL responses (get from parent_state as fallback)
    if state.parent_state and state.parent_state.hitl_status:
        try:
            # Ensure using global json module
            import json as json_module
            hitl_data = json_module.loads(state.parent_state.hitl_status)
            if hitl_data.get("type") == "response_parameters":
                # Process user response (fallback: passed through parent_state)
                responses = hitl_data.get("responses", {})
                for task_id, response_data in responses.items():
                    if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                        state.hitl_responses[task_id] = response_data
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
                                for user_param_name, param_value in response_data["parameters"].items():
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
                            if not result.missing_parameters:
                                state.task_status_map[task_id] = ExecutorTaskStatus.READY
                                print(f"  ✓ Task {task_id} has all required parameters, marked as ready")
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
                    "missing_parameters": request.get("missing_parameters", [])
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
    ready_tasks = [
        task for task in state.subtasks
        if state.task_status_map.get(task.task_id) == ExecutorTaskStatus.READY
        and task.task_id not in state.running_tasks
    ]
    
    if not ready_tasks:
        return state
    
    # Filter out tasks missing parameters (these tasks should wait for HITL response)
    # Note: Task status may be READY, but if parameters are missing, should not execute
    tasks_with_params = []
    for task in ready_tasks:
        task_result = state.task_results.get(task.task_id)
        if task_result and task_result.missing_parameters:
            # Task missing parameters, should not execute, should wait for HITL response
            print(f"  ⏸ Task {task.task_id} missing parameters, skipping execution: {task_result.missing_parameters}")
            # Ensure status is WAITING_HITL_PARAMS
            if state.task_status_map.get(task.task_id) != ExecutorTaskStatus.WAITING_HITL_PARAMS:
                state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_HITL_PARAMS
            continue
        tasks_with_params.append(task)
    
    if not tasks_with_params:
        if ready_tasks:
            print(f"  ⚠ {len(ready_tasks)} ready tasks, but all missing parameters, waiting for HITL response")
        return state
    
    # Calculate number of executable tasks (considering parallel limit)
    available_slots = state.max_parallel_tasks - len(state.running_tasks)
    if available_slots <= 0:
        return state
    
    tasks_to_execute = tasks_with_params[:available_slots]
    print(f"🔄 Starting execution of {len(tasks_to_execute)} tasks (currently running: {len(state.running_tasks)}, max parallel: {state.max_parallel_tasks})")
    
    # Execute tasks
    for task in tasks_to_execute:
        # Mark as running
        state.task_status_map[task.task_id] = ExecutorTaskStatus.RUNNING
        state.running_tasks.append(task.task_id)
        
        # Execute task
        result = _execute_single_task(task, state)
        
        # Update task result and status
        state.task_results[result.task_id] = result
        state.running_tasks.remove(task.task_id)
        
        if result.status == ExecutorTaskStatus.COMPLETED:
            state.completed_count += 1
            state.task_status_map[result.task_id] = ExecutorTaskStatus.COMPLETED
            print(f"  ✓ Task {result.task_id} executed successfully (took {result.execution_time:.2f}s)")
        else:
            # Handle failed tasks: decide whether to re-add to task pool based on error type and retry count
            should_retry = False
            
            if result.error_category == ErrorCategory.NETWORK_ERROR:
                # Network error: always re-add to task pool (until max retries reached)
                if result.retry_count < state.max_retries:
                    should_retry = True
                    result.retry_count += 1
                    state.task_status_map[result.task_id] = ExecutorTaskStatus.READY
                    print(f"  ⚠ Task {result.task_id} network error, re-adding to task pool (retry {result.retry_count}/{state.max_retries})")
                else:
                    state.failed_count += 1
                    state.task_status_map[result.task_id] = ExecutorTaskStatus.FAILED
                    print(f"  ✗ Task {result.task_id} network error, max retries reached, marked as failed")
            
            elif result.error_category == ErrorCategory.RETRYABLE:
                # Other retryable errors: if max retries not reached, re-add to task pool
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
                # Non-retryable errors: directly mark as failed
                state.failed_count += 1
                state.task_status_map[result.task_id] = ExecutorTaskStatus.FAILED
                print(f"  ✗ Task {result.task_id} execution failed ({result.error_category.value}): {result.error}")
            
            # If task needs retry, record error info but keep ready status
            if should_retry:
                # Save error info for later analysis
                result.failure_analysis = f"Error type: {result.error_category.value}, Error message: {result.error[:200]}"
                result.suggestions = [
                    "Task will automatically retry",
                    f"Current retry count: {result.retry_count}/{state.max_retries}"
                ]
    
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
    try:
        # Load MCP server configuration
        from pathlib import Path
        import json as json_module
        
        agent_dir = Path(__file__).parent.parent.parent
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
        
        # Extract host and port from base_url
        # Format: http://host:port/sse
        from urllib.parse import urlparse
        parsed_url = urlparse(base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        
        if not host or not port:
            return {
                "status": ExecutorTaskStatus.FAILED,
                "error": f"Cannot extract host and port from service configuration: {base_url}",
                "output": None
            }
        
        # Build SSE endpoint URL
        sse_url = f"http://{host}:{port}/stream/{task_id}"
        print(f"  🔍 [streaming_task] Built SSE URL: {sse_url}")
        print(f"  🔍 [streaming_task] Service configuration: host={host}, port={port}, base_url={base_url}")
        
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
        
        try:
            print(f"  🔍 [streaming_task] Establishing SSE connection: {sse_url}")
            print(f"  🔍 [streaming_task] Request headers: {headers}")
            response = requests.get(sse_url, headers=headers, stream=True, timeout=timeout)
            print(f"  🔍 [streaming_task] Received response, status code: {response.status_code}")
            response.raise_for_status()
            
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
                        message_content = message_data.get("content", message_data.get("message", ""))
                        
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        all_messages.append({
                            "timestamp": timestamp,
                            "type": message_type,
                            "content": message_content,
                            "raw": message_data
                        })
                        
                        print(f"  📨 [streaming_task] [{timestamp}] {message_type}: {message_content[:100]}")
                        
                        # Check task status
                        if message_type == "task_completed" or message_data.get("status") == "completed":
                            task_completed = True
                            final_result = message_data
                            print(f"  ✓ [streaming_task] Task completed")
                            break
                        elif message_type == "task_failed" or message_data.get("status") == "failed":
                            task_failed = True
                            final_result = message_data
                            error_msg = message_data.get("error", message_data.get("message", "Task failed"))
                            print(f"  ✗ [streaming_task] Task failed: {error_msg}")
                            break
                        elif message_type == "progress" or message_type == "status":
                            # Update progress information
                            progress = message_data.get("progress", message_data.get("percentage", 0))
                            print(f"  📊 [streaming_task] Progress: {progress}%")
                        
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
        
        if tools and len(tools) > 0:
            execution_mode = CodeActExecutionMode.MCP_TOOL
        else:
            execution_mode = CodeActExecutionMode.CODEACT
        
        result.execution_mode = execution_mode.value
        
        # Get parsed parameters
        parameters = result.parameters
        
        # Build CodeAct subgraph input
        codeact_input = codeact_input_mapper(
            executor_state=state,
            task=task,
            execution_mode=execution_mode,
            parameters=parameters
        )
        
        # Call CodeAct subgraph
        codeact_graph = build_codeact_subgraph()
        codeact_output = codeact_graph.invoke(codeact_input)
        
        # Convert dict output to CodeActState object (LangGraph returns dict)
        if isinstance(codeact_output, dict):
            codeact_state = CodeActState.model_validate(codeact_output)
        else:
            codeact_state = codeact_output
        
        # Process execution result
        exec_result = codeact_output_mapper(codeact_state)
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
                if isinstance(streaming_status, ExecutorTaskStatus):
                    result.status = streaming_status
                elif streaming_status == ExecutorTaskStatus.COMPLETED:
                    result.status = ExecutorTaskStatus.COMPLETED
                elif streaming_status == ExecutorTaskStatus.FAILED:
                    result.status = ExecutorTaskStatus.FAILED
                elif isinstance(streaming_status, str):
                    # If string, convert to enum
                    if streaming_status == "completed" or streaming_status == "COMPLETED":
                        result.status = ExecutorTaskStatus.COMPLETED
                    elif streaming_status == "failed" or streaming_status == "FAILED":
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
                    result.status = ExecutorTaskStatus.FAILED
                
                if result.status:
                    print(f"  ✓ [streaming_task] Streaming task processing completed, status: {result.status.value}")
                else:
                    print(f"  ⚠ [streaming_task] Streaming task processing completed, but status is None")
            else:
                # Normal task, directly use original result
                result.status = ExecutorTaskStatus.COMPLETED
                result.output = output
            
            result.code = exec_result.get("code")
        else:
            result.status = ExecutorTaskStatus.FAILED
            result.error = exec_result.get("error", "Execution failed")
            error_type = exec_result.get("error_type", "UnknownError")
            result.error_category = classify_error(result.error, error_type)
            
            # Generate error analysis and suggestions
            result.failure_analysis = _analyze_failure(result.error, error_type, result.error_category)
            result.suggestions = _generate_suggestions(result.error_category, result.error, result.retry_count, state.max_retries)
    
    except Exception as e:
        result.status = ExecutorTaskStatus.FAILED
        error_msg = str(e)
        error_type = type(e).__name__
        result.error_category = classify_error(error_msg, error_type)
        # Record detailed error info for debugging
        import traceback
        error_traceback = traceback.format_exc()
        if len(error_traceback) > 1000:
            error_traceback = error_traceback[:1000] + "..."
        result.error = f"{error_msg}\n{error_traceback}"
        
        # Generate error analysis and suggestions
        result.failure_analysis = _analyze_failure(error_msg, error_type, result.error_category)
        result.suggestions = _generate_suggestions(result.error_category, error_msg, result.retry_count, state.max_retries)
        
        print(f"  ✗ Task {task.task_id} execution exception: {error_msg}")
    
    result.execution_time = time.time() - start_time
    return result


def analyze_results_node(state: ExecutorState) -> ExecutorState:
    """
    Result reasoning node
    
    Use LLM to reason about task execution results and determine if requirements are met.
    If not satisfied, trigger HITL to ask user if they want to continue.
    """
    llm = create_reasoning_advanced_llm()
    
    # Analyze completed tasks
    for task in state.subtasks:
        result = state.task_results.get(task.task_id)
        if not result or result.status != ExecutorTaskStatus.COMPLETED:
            continue
        
        # If already analyzed, skip
        if result.result_satisfied is not None:
            continue
        
        if llm and result.output:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                
                analysis_prompt = f"""
Please evaluate whether the following task execution result meets requirements:

Task ID: {result.task_id}
Task description: {task.content}
Execution mode: {result.execution_mode}
Execution result: {str(result.output)[:1000]}

Please return JSON format:
{{
    "satisfied": <true/false>,
    "confidence": <float between 0-1>,
    "reason": "<evaluation reason>",
    "needs_user_confirmation": <true/false>
}}
"""
                messages = [
                    SystemMessage(content="You are a professional task execution result evaluation expert."),
                    HumanMessage(content=analysis_prompt)
                ]
                
                response = llm.invoke(messages)
                response_text = response.content.strip()
                
                # Parse response
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    # Ensure using global json module
                    import json as json_module
                    analysis_data = json_module.loads(json_match.group())
                    result.result_satisfied = analysis_data.get("satisfied", True)
                    result.confidence_score = analysis_data.get("confidence", 0.5)
                    needs_confirmation = analysis_data.get("needs_user_confirmation", False)
                    
                    # If not satisfied or needs user confirmation, trigger HITL
                    if not result.result_satisfied or needs_confirmation:
                        state.hitl_requests[task.task_id] = {
                            "type": "result_confirmation",
                            "task_id": task.task_id,
                            "task_description": task.content,
                            "result": str(result.output)[:500],
                            "reason": analysis_data.get("reason", ""),
                            "message": f"Task {task.task_id} execution result may not meet requirements. Continue executing subsequent tasks?"
                        }
                        state.task_status_map[task.task_id] = ExecutorTaskStatus.WAITING_HITL_CONFIRM
                        print(f"  ⚠ Task {task.task_id} result needs user confirmation")
            except Exception as e:
                print(f"  ⚠ Failed to analyze result: {e}")
                result.result_satisfied = True  # Default to satisfied
    
    return state


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
                    state.hitl_responses[task_id] = response_data
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
    
    # First check if there are HITL responses (get from parent_state as fallback)
    if state.parent_state and state.parent_state.hitl_status:
        try:
            # Ensure using global json module
            import json as json_module
            hitl_data = json_module.loads(state.parent_state.hitl_status)
            if hitl_data.get("type") == "response_confirmation":
                responses = hitl_data.get("responses", {})
                for task_id, response_data in responses.items():
                    if task_id in state.hitl_requests and task_id not in state.hitl_responses:
                        state.hitl_responses[task_id] = response_data
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
        parent_state=global_state
    )
    
    return executor_state


def executor_output_mapper(executor_state: ExecutorState, global_state: GlobalState) -> GlobalState:
    """
    Map Executor subgraph state back to main graph state
    
    Args:
        executor_state: Executor subgraph state
        global_state: Main graph global state
    
    Returns:
        Updated main graph state
    """
    # Update task results (including tasks in subtasks and parallel task groups)
    all_tasks_to_update = list(global_state.subtasks)
    
    # Extract all tasks from parallel task groups
    for group_id, group in global_state.parallel_task_groups.items():
        if hasattr(group, 'subtasks') and group.subtasks:
            for task in group.subtasks:
                if not any(t.task_id == task.task_id for t in all_tasks_to_update):
                    all_tasks_to_update.append(task)
    
    # Update results for all tasks
    for task in all_tasks_to_update:
        task_result = executor_state.task_results.get(task.task_id)
        if task_result and task_result.status == ExecutorTaskStatus.COMPLETED:
            # Update task result
            if not task.result:
                task.result = {}
            if isinstance(task.result, dict):
                task.result["execution_result"] = task_result.output
                task.result["execution_mode"] = task_result.execution_mode
                task.result["code"] = task_result.code
                task.result["confidence_score"] = task_result.confidence_score
            
            # Mark task as completed
            global_state.completed_tasks[task.task_id] = task
    
    # Update summary results
    global_state.merged_result["executor_results"] = {
        "total_tasks": executor_state.total_tasks,
        "completed": executor_state.completed_count,
        "failed": executor_state.failed_count,
        "task_results": {
            task_id: {
                "status": result.status.value,
                "execution_mode": result.execution_mode,
                "error": result.error,
                "error_category": result.error_category.value if result.error_category else None,
                "confidence_score": result.confidence_score,
                "failure_analysis": result.failure_analysis,
                "suggestions": result.suggestions
            }
            for task_id, result in executor_state.task_results.items()
        }
    }
    
    # Update HITL status (if any)
    if executor_state.hitl_requests:
        pending_hitl = [
            task_id for task_id in executor_state.hitl_requests.keys()
            if task_id not in executor_state.hitl_responses
        ]
        if pending_hitl:
            global_state.hitl_status = json.dumps({
                "type": "request",
                "requests": [
                    executor_state.hitl_requests[task_id]
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
            input_data = initial_state
        else:
            # Save parent_state reference (because it will be excluded)
            saved_parent_state = getattr(initial_state, 'parent_state', None)
            input_data = initial_state.model_dump(exclude={'parent_state'}, mode='json')
    
    # Use stream() to detect interrupts
    interrupted = False
    interrupt_data = None
    final_result = None
    
    # Store parent_state in module variable so nodes can access it
    # Note: This is not thread-safe, but can be used in single-threaded tests
    global _current_parent_state
    _current_parent_state = saved_parent_state
    print(f"  🔍 [execute_executor] Saved parent_state to module variable: {_current_parent_state is not None}")
    
    try:
        # Use stream to execute step by step, can detect interrupts
        # LangGraph interrupt mechanism:
        # - When interrupt() is called, it raises GraphInterrupt exception
        # - LangGraph catches this exception, saves state, and returns special format in stream
        # - Interrupt info may be in chunk's "__interrupt__" field, or propagated as exception
        
        print(f"  🔍 [execute_executor] Starting stream execution, config: {config}")
        print(f"  🔍 [execute_executor] saved_parent_state exists: {saved_parent_state is not None}")
        chunk_count = 0
        for chunk in executor_graph.stream(input_data, config=config):
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
                
                # Extract actual value from Interrupt object
                if hasattr(interrupt_obj, 'value'):
                    interrupt_data = interrupt_obj.value
                    print(f"  🔍 [execute_executor] Extracted value from Interrupt object: type={type(interrupt_data)}")
                elif isinstance(interrupt_obj, dict) and 'value' in interrupt_obj:
                    interrupt_data = interrupt_obj['value']
                    # If value itself is Interrupt object, continue extracting
                    if hasattr(interrupt_data, 'value'):
                        interrupt_data = interrupt_data.value
                else:
                    interrupt_data = interrupt_obj
                
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
        error_str = str(e).lower()
        error_type = type(e).__name__
        
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
                    # Other exception, re-raise
                    print(f"  ✗ Unexpected exception: {error_type}: {e}")
                    raise
            else:
                # Other exception, re-raise
                print(f"  ✗ Unexpected exception: {error_type}: {e}")
                raise
    
    # Ensure interrupt_data is dict format (if tuple or other type, convert to dict)
    if interrupt_data is not None:
        # First, if interrupt_data is Interrupt object, extract its value
        if hasattr(interrupt_data, 'value'):
            interrupt_data = interrupt_data.value
            print(f"  🔍 [execute_executor] Extracted value from Interrupt object: type={type(interrupt_data)}")
        
        # If extracted value is still Interrupt object (nested case), continue extracting
        if hasattr(interrupt_data, 'value'):
            interrupt_data = interrupt_data.value
            print(f"  🔍 [execute_executor] Extracted value from nested Interrupt object: type={type(interrupt_data)}")
        
        if isinstance(interrupt_data, tuple):
            # LangGraph interrupt may return tuple, need to convert to dict
            if len(interrupt_data) == 2:
                # May be (key, value) format
                value = interrupt_data[1]
                # If value is Interrupt object, extract its value
                if hasattr(value, 'value'):
                    value = value.value
                interrupt_data = {"key": interrupt_data[0], "value": value}
            elif len(interrupt_data) == 1:
                # May be single value
                value = interrupt_data[0]
                if hasattr(value, 'value'):
                    value = value.value
                interrupt_data = {"value": value}
            else:
                # Multiple values, convert to dict
                interrupt_data = {f"arg_{i}": (v.value if hasattr(v, 'value') else v) for i, v in enumerate(interrupt_data)}
            print(f"  🔍 [execute_executor] Converted interrupt_data from tuple to dict: {interrupt_data}")
        elif not isinstance(interrupt_data, dict):
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
                # Extract actual value from Interrupt object
                if hasattr(interrupt_obj, 'value'):
                    interrupt_data = interrupt_obj.value
                elif isinstance(interrupt_obj, dict) and 'value' in interrupt_obj:
                    interrupt_data = interrupt_obj['value']
                    if hasattr(interrupt_data, 'value'):
                        interrupt_data = interrupt_data.value
                else:
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
            interrupt_data = getattr(e, 'value', None)
            # If interrupt_data is Interrupt object, extract its value
            if interrupt_data and hasattr(interrupt_data, 'value'):
                interrupt_data = interrupt_data.value
            if interrupt_data is None:
                interrupt_data = str(e)
        else:
            raise
    
    return {
        "result": final_result,
        "interrupted": interrupted,
        "interrupt_data": interrupt_data,
        "needs_resume": interrupted
    }
