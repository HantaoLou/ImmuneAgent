"""
TodoList Generator for Executor Subgraph

Responsible for:
1. Converting SubTask[] from task_decomposition to TodoTask[] for codeact
2. Generating todo-list.md in sandbox directory
3. Updating task status in todo-list.md after execution

This module bridges task_decomposition output with codeact execution.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import re
from collections import deque

# Add agent directory to path
import sys
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import SubTask, UserTaskType, ParallelTaskGroup, GlobalState
from nodes.subagents.code_act.todo_list import (
    TodoTask,
    TodoTaskType,
    TodoTaskStatus,
    TodoList,
    TodoListSession,
    TodoListManager
)


# =============================================================================
# Type Mapping Functions
# =============================================================================

# List of known MCP tool name prefixes/patterns
MCP_TOOL_PATTERNS = [
    # NetTCR tools
    "check_peptide_support", "predict_tcr_binding", "integrate_tcr_data",
    # IgBLAST tools
    "analyze_vdj_batch", "extract_cdr3", "igblast",
    # MetaBCR tools
    "metabcr",
    # Integration tools
    "integrate_bcr_data", "integrate_", 
    # Analysis tools
    "predict_", "analyze_", "evaluate_", "visualize_",
    # File conversion tools
    "convert_csv_to", "convert_xlsx_to",
    # Other common patterns
    "tcr_clonotype", "tcr_binding", "tcell_", "bcell_", "immune_",
]

# Tools that are NOT MCP tools (use codeact)
NON_MCP_TOOLS = ["codeact", "general", "python", "script"]


def _map_task_type_to_todotype(
    task_type: UserTaskType, 
    task_content: str, 
    tool_name: Optional[str] = None
) -> TodoTaskType:
    """
    Map UserTaskType to TodoTaskType based on task type, content, and tool name
    
    Priority order:
    1. If tool_name is provided and matches known MCP tool patterns -> MCP_TOOL
    2. If tool_name is 'codeact' or non-MCP -> GENERAL
    3. Check content for file operations
    4. Check content for MCP tool indicators
    5. Default to GENERAL
    
    Args:
        task_type: Original UserTaskType
        task_content: Task content/description
        tool_name: Optional tool name extracted from subtask
    
    Returns:
        Corresponding TodoTaskType
    """
    content_lower = task_content.lower()
    
    # PRIORITY 1: Use tool_name if provided
    if tool_name:
        tool_name_lower = tool_name.lower()
        
        # Check if it's explicitly a non-MCP tool
        if tool_name_lower in NON_MCP_TOOLS:
            return TodoTaskType.GENERAL
        
        # Check against known MCP tool patterns
        for pattern in MCP_TOOL_PATTERNS:
            if pattern in tool_name_lower:
                return TodoTaskType.MCP_TOOL
        
        # If tool_name contains underscore (typical for MCP tools) and not codeact
        if "_" in tool_name_lower and "codeact" not in tool_name_lower:
            return TodoTaskType.MCP_TOOL
        
        # If tool_name exists but doesn't match patterns, still consider it MCP
        # (it's likely a new tool we haven't added to the list)
        if tool_name_lower and tool_name_lower not in NON_MCP_TOOLS:
            return TodoTaskType.MCP_TOOL
    
    # PRIORITY 2: Check for file operations in content
    if "upload" in content_lower or "file_upload" in content_lower:
        return TodoTaskType.FILE_UPLOAD
    if "convert" in content_lower and ("csv" in content_lower or "fasta" in content_lower):
        return TodoTaskType.FILE_CONVERT
    if "analyze" in content_lower and "file" in content_lower:
        return TodoTaskType.FILE_ANALYSIS
    if "mcp_tool" in content_lower or "tool:" in content_lower:
        return TodoTaskType.MCP_TOOL
    
    # PRIORITY 3: Check for tool names in content
    for indicator in MCP_TOOL_PATTERNS:
        if indicator in content_lower:
            return TodoTaskType.MCP_TOOL
    
    # PRIORITY 4: Check for codeact indicator
    if "codeact" in content_lower or "[codeact]" in content_lower:
        return TodoTaskType.GENERAL
    
    return TodoTaskType.GENERAL


def _extract_tool_name_from_subtask(subtask: SubTask) -> Optional[str]:
    """
    Extract tool name from SubTask's result metadata
    
    Args:
        subtask: SubTask object
    
    Returns:
        Tool name if found, None otherwise
    """
    if not subtask.result:
        return None
    
    if isinstance(subtask.result, dict):
        tools = subtask.result.get("tools", [])
        if tools and isinstance(tools, list) and len(tools) > 0:
            first_tool = tools[0]
            if isinstance(first_tool, dict):
                return first_tool.get("tool_name") or first_tool.get("name")
            elif isinstance(first_tool, str):
                return first_tool
    
    return None


def _extract_parameters_from_subtask(
    subtask, 
    extracted_parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract actual parameter values from subtask, using extracted_parameters
    as the source of truth for values.
    
    The subtask.result.get("parameters") contains parameter DEFINITIONS (schema),
    NOT actual values. Actual values come from extracted_parameters which is
    populated during the parameter extraction phase.
    
    Args:
        subtask: SubTask containing parameter definitions
        extracted_parameters: Dict of actual parameter values extracted from context
        
    Returns:
        Dict of parameter name -> actual value
    """
    actual_params = {}
    
    if extracted_parameters:
        # extracted_parameters contains the actual values we need
        actual_params.update(extracted_parameters)
    
    if isinstance(subtask.result, dict):
        # Get parameter DEFINITIONS from the task
        param_definitions = subtask.result.get("parameters", {})
        
        # Filter out schema-only entries (descriptions, types, etc.)
        # and only keep actual values
        for param_name, param_value in param_definitions.items():
            # If it's a dict with type/description keys, it's a schema definition
            if isinstance(param_value, dict):
                if 'type' in param_value or 'description' in param_value:
                    # This is a schema definition, not an actual value
                    # Check if we have an actual value from extracted_parameters
                    if param_name not in actual_params:
                        # No actual value available, skip this schema definition
                        continue
                else:
                    # This might be an actual dict value
                    if param_name not in actual_params:
                        actual_params[param_name] = param_value
            else:
                # This is an actual value
                if param_name not in actual_params:
                    actual_params[param_name] = param_value
        
        # Handle inputs/outputs as fallback
        if not actual_params:
            inputs = subtask.result.get("inputs", [])
            if inputs and len(inputs) == 1:
                actual_params["input"] = inputs[0]
            
            outputs = subtask.result.get("outputs", [])
            if outputs and len(outputs) == 1:
                actual_params["output"] = outputs[0]
    
    return actual_params


# =============================================================================
# Conversion Functions
# =============================================================================

# =============================================================================
# Topological Sorting Functions (P1: Priority based on dependencies)
# =============================================================================

def _calculate_topological_priorities(
    task_ids: List[str],
    dependencies_map: Dict[str, List[str]]
) -> Dict[str, int]:
    """
    Calculate task priorities using topological sorting.
    
    Tasks with no dependencies get the lowest priority numbers (highest priority).
    Tasks that depend on others get higher numbers based on dependency depth.
    
    Args:
        task_ids: List of all task IDs
        dependencies_map: Dict mapping task_id -> list of dependency task_ids
    
    Returns:
        Dict mapping task_id -> priority (1=highest priority, higher=lower priority)
    """
    # Build reverse dependency graph (who depends on me)
    dependents: Dict[str, Set[str]] = {tid: set() for tid in task_ids}
    for tid, deps in dependencies_map.items():
        for dep in deps:
            if dep in dependents:
                dependents[dep].add(tid)
    
    # Calculate depth using BFS from root tasks (no dependencies)
    depths: Dict[str, int] = {}
    queue = deque()
    
    # Find root tasks (no dependencies)
    for tid in task_ids:
        if not dependencies_map.get(tid):
            depths[tid] = 0
            queue.append(tid)
    
    # BFS to calculate depths
    while queue:
        current = queue.popleft()
        current_depth = depths[current]
        
        for dependent in dependents.get(current, []):
            if dependent not in depths:
                depths[dependent] = current_depth + 1
                queue.append(dependent)
    
    # Handle any tasks not reached (circular dependencies or missing deps)
    for tid in task_ids:
        if tid not in depths:
            depths[tid] = max(depths.values()) + 1 if depths else 0
    
    # Convert depths to priorities (depth 0 -> priority 1, etc.)
    priorities = {tid: depth + 1 for tid, depth in depths.items()}
    
    return priorities


def convert_subtask_to_todotask(
    subtask, 
    priority: int = 5,
    extracted_parameters: Optional[Dict[str, Any]] = None
) -> TodoTask:
    """
    Convert a single SubTask to TodoTask
    
    Args:
        subtask: SubTask from task_decomposition
        priority: Task priority (1=highest, 5=lowest)
        extracted_parameters: Actual parameter values extracted from context
    
    Returns:
        TodoTask object for codeact execution
    """
    # Extract tool name first (needed for type determination)
    tool_name = _extract_tool_name_from_subtask(subtask)
    
    # Determine task type (P0: now uses tool_name for accurate type detection)
    todo_type = _map_task_type_to_todotype(
        task_type=subtask.task_type,
        task_content=subtask.content,
        tool_name=tool_name
    )
    
    # Extract actual parameter values (not schema definitions)
    parameters = _extract_parameters_from_subtask(subtask, extracted_parameters)
    
    # Add tool name to parameters if available
    if tool_name:
        parameters["tool_name"] = tool_name
    
    # Build description
    description = subtask.content
    if tool_name:
        description = f"[{tool_name}] {description}"
    
    return TodoTask(
        id=subtask.task_id,
        type=todo_type,
        status=TodoTaskStatus.PENDING,
        priority=priority,
        description=description,
        parameters=parameters,
        dependencies=subtask.dependencies.copy() if subtask.dependencies else [],
        result=None,
        error=None,
        started_at=None,
        completed_at=None
    )


def convert_subtasks_to_todolist(
    subtasks: List[SubTask],
    parallel_task_groups: Dict[str, ParallelTaskGroup],
    session_id: str,
    sandbox_dir: str,
    sandbox_id: Optional[str] = None,
    extracted_parameters: Optional[Dict[str, Any]] = None
) -> TodoList:
    """
    Convert all SubTasks (serial + parallel) to TodoList
    
    P1 Enhancement: Uses topological sorting to calculate priorities.
    Tasks with no dependencies get priority=1 (highest).
    Tasks that depend on others get higher priority numbers based on depth.
    
    Args:
        subtasks: List of serial SubTasks
        parallel_task_groups: Dict of parallel task groups
        session_id: Session ID
        sandbox_dir: Sandbox directory path
        sandbox_id: Optional sandbox instance ID
        extracted_parameters: Dict of actual parameter values from context
    
    Returns:
        TodoList object containing all tasks
    """
    # Collect all subtasks first
    all_subtasks: List[SubTask] = []
    
    # Add serial tasks
    all_subtasks.extend(subtasks)
    
    # Add parallel tasks
    for group_id, group in parallel_task_groups.items():
        if hasattr(group, 'subtasks') and group.subtasks:
            all_subtasks.extend(group.subtasks)
    
    # Build task IDs and dependencies map for topological sorting
    task_ids = [st.task_id for st in all_subtasks]
    dependencies_map = {
        st.task_id: st.dependencies.copy() if st.dependencies else []
        for st in all_subtasks
    }
    
    # P1: Calculate priorities using topological sorting
    priorities = _calculate_topological_priorities(task_ids, dependencies_map)
    
    # Convert to TodoTasks with calculated priorities
    all_todo_tasks = []
    for subtask in all_subtasks:
        priority = priorities.get(subtask.task_id, 5)
        # Pass extracted_parameters to get actual values
        todo_task = convert_subtask_to_todotask(
            subtask, 
            priority=priority,
            extracted_parameters=extracted_parameters
        )
        all_todo_tasks.append(todo_task)
    
    # Sort tasks by priority for cleaner output
    all_todo_tasks.sort(key=lambda t: (t.priority, t.id))
    
    # Create session info
    session = TodoListSession(
        session_id=session_id,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sandbox_id=sandbox_id,
        sandbox_dir=sandbox_dir
    )
    
    return TodoList(
        session=session,
        tasks=all_todo_tasks
    )


# =============================================================================
# File Operations
# =============================================================================

def generate_todo_list_markdown(todo_list: TodoList) -> str:
    """
    Generate markdown content for todo-list.md
    
    Args:
        todo_list: TodoList object
    
    Returns:
        Markdown formatted string
    """
    lines = [
        "# Task List",
        "",
        "## Session Info",
        f"- session_id: {todo_list.session.session_id}",
        f"- created_at: {todo_list.session.created_at}",
    ]
    
    if todo_list.session.sandbox_id:
        lines.append(f"- sandbox_id: {todo_list.session.sandbox_id}")
    if todo_list.session.sandbox_dir:
        lines.append(f"- sandbox_dir: {todo_list.session.sandbox_dir}")
    
    lines.append("")
    lines.append("## Tasks")
    lines.append("")
    
    for task in todo_list.tasks:
        # Task header (no emoji for cleaner parsing)
        lines.append(f"### Task: {task.id}")
        lines.append(f"- id: {task.id}")
        lines.append(f"- type: {task.type.value if hasattr(task.type, 'value') else str(task.type)}")
        lines.append(f"- status: {task.status.value if hasattr(task.status, 'value') else str(task.status)}")
        lines.append(f"- priority: {task.priority}")
        lines.append(f"- description: {task.description}")
        
        # Parameters
        if task.parameters:
            lines.append("- parameters:")
            for key, value in task.parameters.items():
                # Format value for readability
                if isinstance(value, str) and len(value) > 100:
                    value_str = f"{value[:97]}..."
                else:
                    value_str = str(value)
                lines.append(f"    {key}: {value_str}")
        else:
            lines.append("- parameters: {}")
        
        # Dependencies
        if task.dependencies:
            lines.append(f"- dependencies: {', '.join(task.dependencies)}")
        else:
            lines.append("- dependencies: []")
        
        # Result
        if task.result:
            lines.append("- result:")
            if isinstance(task.result, dict):
                for key, value in task.result.items():
                    if isinstance(value, str) and len(value) > 100:
                        value_str = f"{value[:97]}..."
                    else:
                        value_str = str(value)
                    lines.append(f"    {key}: {value_str}")
            else:
                lines.append(f"    {task.result}")
        else:
            lines.append("- result: null")
        
        # Error
        if task.error:
            lines.append(f"- error: {task.error}")
        else:
            lines.append("- error: null")
        
        # Timestamps
        if task.started_at:
            lines.append(f"- started_at: {task.started_at}")
        if task.completed_at:
            lines.append(f"- completed_at: {task.completed_at}")
        
        lines.append("")
    
    return "\n".join(lines)


def save_todo_list_to_sandbox(
    todo_list: TodoList, 
    sandbox_dir: str, 
    filename: str = "todo-list.md",
    opensandbox_id: Optional[str] = None,
    use_remote: bool = True
) -> Tuple[str, Optional[str]]:
    """
    Save TodoList as markdown file to sandbox directory
    
    Args:
        todo_list: TodoList object
        sandbox_dir: Sandbox directory path (remote path like /data/sessions/{session_id})
        filename: Output filename (default: todo-list.md)
        opensandbox_id: OpenSandbox instance ID for remote saving (optional, creates new if None)
        use_remote: If True, save to remote sandbox via CodeAct
    
    Returns:
        Tuple of (file_path, sandbox_id):
            - file_path: Path to saved file
            - sandbox_id: Sandbox ID that can be reused (or None if local save)
    
    Raises:
        IOError: If file cannot be saved
    """
    # Generate markdown content
    markdown_content = generate_todo_list_markdown(todo_list)
    
    # Determine if this is a remote sandbox path (Linux-style paths indicate remote)
    is_remote_path = sandbox_dir.startswith('/data/') or sandbox_dir.startswith('/tmp/')
    
    # For remote paths, ALWAYS use CodeAct to save to remote sandbox
    # (even if opensandbox_id is not provided, we create a new sandbox)
    if use_remote and is_remote_path:
        # Save to remote sandbox via CodeAct
        return _save_to_remote_sandbox(markdown_content, sandbox_dir, filename, opensandbox_id)
    else:
        # Fallback to local file system
        sandbox_path = Path(sandbox_dir)
        sandbox_path.mkdir(parents=True, exist_ok=True)
        
        todo_file = sandbox_path / filename
        
        # Write to file
        with open(todo_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📝 Todo list saved to: {todo_file}")
        return str(todo_file), None  # Local save, no sandbox_id


def _save_to_remote_sandbox(
    content: str, 
    sandbox_dir: str, 
    filename: str,
    opensandbox_id: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Save content to remote sandbox via CodeAct
    
    Args:
        content: File content to save
        sandbox_dir: Remote sandbox directory path (server path, e.g., /data/sessions/...)
        filename: Output filename
        opensandbox_id: Optional OpenSandbox instance ID (if None, creates new sandbox)
    
    Returns:
        Tuple of (remote_path, sandbox_id) - sandbox_id can be reused for subsequent operations
    """
    try:
        from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
        from utils.sandbox_paths import get_container_path
        
        if not is_codeact_available():
            raise RuntimeError("CodeAct not available for remote saving")
        
        # Convert server path to container path for writing
        # Server path: /data/sessions/{session_id}/...  (read-only in container)
        # Container path: /tmp/sessions/{session_id}/...  (writable in container)
        container_dir = get_container_path(sandbox_dir)
        
        # Escape content for Python string
        escaped_content = content.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        
        # Build Python code to save the file
        save_code = f'''
import os

# Ensure directory exists
os.makedirs("{container_dir}", exist_ok=True)

# Write todo-list.md
file_path = "{container_dir}/{filename}"
content = """{escaped_content}"""

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"__TODO_SAVED__:{{file_path}}")
'''
        
        # Execute via CodeAct - sandbox_id can be None (creates new sandbox)
        result = execute_code_via_codeact(
            task_description=f"保存 Todo list 到远程沙盒: {container_dir}/{filename}",
            code_template=save_code,
            sandbox_id=opensandbox_id,  # Can be None - will create new sandbox
            timeout_seconds=60,
            keep_alive=True  # Keep sandbox alive for reuse
        )
        
        # CodeActResult is a dataclass, use is_success() method
        if result and result.is_success():
            remote_path = f"{sandbox_dir}/{filename}"  # Return server path for reference
            returned_sandbox_id = result.sandbox_id  # Get sandbox_id for reuse
            print(f"📝 Todo list saved to remote sandbox: {remote_path}")
            if returned_sandbox_id:
                print(f"📦 Sandbox ID for reuse: {returned_sandbox_id}")
            return remote_path, returned_sandbox_id
        else:
            # 提供更详细的错误信息用于诊断
            if result:
                error_parts = []
                if result.error:
                    error_parts.append(f"error={result.error}")
                if result.output:
                    error_parts.append(f"output={result.output[:200]}")  # 截断输出
                if result.returncode is not None:
                    error_parts.append(f"returncode={result.returncode}")
                error_detail = ", ".join(error_parts) if error_parts else "no details available"
            else:
                error_detail = "result is None"
            raise IOError(f"Failed to save to remote sandbox: {error_detail}")
            
    except Exception as e:
        print(f"⚠️ Failed to save todo-list to remote sandbox: {e}")
        raise


def update_task_status_in_todolist(
    sandbox_dir: str,
    task_id: str,
    status: TodoTaskStatus,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    filename: str = "todo-list.md"
) -> bool:
    """
    Update task status in todo-list.md
    
    Args:
        sandbox_dir: Sandbox directory path
        task_id: Task ID to update
        status: New status
        result: Optional result dictionary
        error: Optional error message
        filename: Todo list filename
    
    Returns:
        True if update successful, False otherwise
    """
    try:
        manager = TodoListManager(sandbox_dir, filename)
        
        if not manager.todo_list_exists():
            print(f"⚠️ Todo list not found: {sandbox_dir}/{filename}")
            return False
        
        # Read current todo list
        todo_list = manager.read_todo_list()
        
        # Find and update task
        for task in todo_list.tasks:
            if task.id == task_id:
                task.status = status
                
                if result:
                    task.result = result
                
                if error:
                    task.error = error
                
                # Update timestamps
                if status == TodoTaskStatus.IN_PROGRESS:
                    task.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elif status in [TodoTaskStatus.COMPLETED, TodoTaskStatus.FAILED, TodoTaskStatus.SKIPPED]:
                    task.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                break
        else:
            print(f"⚠️ Task not found in todo list: {task_id}")
            return False
        
        # Save updated todo list
        save_todo_list_to_sandbox(todo_list, sandbox_dir, filename)
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to update task status: {e}")
        return False


# =============================================================================
# Integration Functions for Executor
# =============================================================================

def generate_and_save_todolist_from_state(
    global_state: GlobalState,
    sandbox_dir: Optional[str] = None,
    opensandbox_id: Optional[str] = None
) -> Optional[TodoList]:
    """
    Generate and save TodoList from GlobalState
    
    This is the main entry point for executor to generate todo-list.md
    
    IMPORTANT: Parameter values are extracted from global_state.extracted_parameters
    which contains actual values (not schema definitions). The subtask.result.parameters
    contains parameter DEFINITIONS, not values.
    
    Args:
        global_state: GlobalState containing subtasks, parallel_task_groups, and extracted_parameters
        sandbox_dir: Optional sandbox directory (defaults to global_state.sandbox_dir)
        opensandbox_id: Optional OpenSandbox instance ID for remote saving
    
    Returns:
        TodoList if successful, None otherwise
    """
    # Determine sandbox directory
    sandbox = sandbox_dir or global_state.sandbox_dir or global_state.sandbox_data_dir
    if not sandbox:
        print("⚠️ No sandbox directory specified")
        return None
    
    # Get opensandbox_id from state if not provided
    sandbox_id = opensandbox_id or global_state.opensandbox_id or global_state.merged_result.get('opensandbox_id')
    
    # Check if there are tasks to convert
    total_tasks = len(global_state.subtasks)
    for group in global_state.parallel_task_groups.values():
        if hasattr(group, 'subtasks'):
            total_tasks += len(group.subtasks)
    
    if total_tasks == 0:
        print("⚠️ No tasks to convert to TodoList")
        return None
    
    print(f"📋 Converting {total_tasks} tasks to TodoList...")
    
    # Extract actual parameter values from global_state
    # This is the key fix: use extracted_parameters for actual values
    extracted_params = global_state.extracted_parameters or {}
    
    # Also check merged_result for additional parameters (from supervisor)
    merged_result = global_state.merged_result or {}
    if isinstance(merged_result, dict):
        # Merge params from merged_result
        merged_params = merged_result.get("extracted_parameters", {}).get("params", {})
        if merged_params:
            extracted_params = {**extracted_params, **merged_params}
    
    print(f"📋 Using extracted parameters: {list(extracted_params.keys())}")
    
    # Convert to TodoList with actual parameter values
    todo_list = convert_subtasks_to_todolist(
        subtasks=global_state.subtasks,
        parallel_task_groups=global_state.parallel_task_groups,
        session_id=global_state.session_id or "unknown",
        sandbox_dir=sandbox,
        sandbox_id=global_state.session_id,
        extracted_parameters=extracted_params
    )
    
    # Save to sandbox (remote if opensandbox_id available)
    # Returns (file_path, sandbox_id) - sandbox_id can be reused
    saved_path, returned_sandbox_id = save_todo_list_to_sandbox(
        todo_list, sandbox, opensandbox_id=sandbox_id
    )
    
    # Update global_state with sandbox_id if we got a new one
    if returned_sandbox_id and not global_state.opensandbox_id:
        global_state.opensandbox_id = returned_sandbox_id
        print(f"📦 Updated GlobalState.opensandbox_id: {returned_sandbox_id}")
    
    return todo_list


# =============================================================================
# Utility Functions
# =============================================================================

def get_task_summary(todo_list: TodoList) -> Dict[str, Any]:
    """
    Get summary statistics of TodoList
    
    Args:
        todo_list: TodoList object
    
    Returns:
        Summary dictionary
    """
    status_counts = {}
    type_counts = {}
    
    for task in todo_list.tasks:
        # Count by status
        status_str = task.status.value if hasattr(task.status, 'value') else str(task.status)
        status_counts[status_str] = status_counts.get(status_str, 0) + 1
        
        # Count by type
        type_str = task.type.value if hasattr(task.type, 'value') else str(task.type)
        type_counts[type_str] = type_counts.get(type_str, 0) + 1
    
    return {
        "total_tasks": len(todo_list.tasks),
        "status_counts": status_counts,
        "type_counts": type_counts,
        "session_id": todo_list.session.session_id
    }


if __name__ == "__main__":
    # Test conversion
    print("=" * 60)
    print("Testing TodoList Generator with P0 + P1 enhancements")
    print("=" * 60)
    
    # P0 Test: Create subtasks with various tool names to test type detection
    test_subtasks = [
        SubTask(
            task_id="task_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="Check MART-1 Peptide Support",
            dependencies=[],
            result={"tools": [{"tool_name": "check_peptide_support"}], "parameters": {"peptides": "ELAGIGILTV"}}
        ),
        SubTask(
            task_id="task_002",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="TCR Binding Prediction",
            dependencies=["task_001"],
            result={"tools": [{"tool_name": "predict_tcr_binding_complete"}], "parameters": {"test_file": "/data/test.csv"}}
        ),
        SubTask(
            task_id="task_003",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="Integrate TCR Prediction Data",
            dependencies=["task_002"],
            result={"tools": [{"tool_name": "integrate_tcr_data_complete"}], "parameters": {}}
        ),
        SubTask(
            task_id="task_004",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="Evaluate TCR Binding Predictions",
            dependencies=["task_002"],
            result={"tools": [{"tool_name": "codeact"}], "parameters": {}}
        ),
    ]
    
    # Convert to TodoList
    todo_list = convert_subtasks_to_todolist(
        subtasks=test_subtasks,
        parallel_task_groups={},
        session_id="test_session_001",
        sandbox_dir="/tmp/test_sandbox"
    )
    
    # Print summary
    summary = get_task_summary(todo_list)
    print(f"\n📊 TodoList Summary:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Status: {summary['status_counts']}")
    print(f"  Types: {summary['type_counts']}")
    
    # P0 Test: Verify type detection
    print(f"\n🔍 P0 Test: Type Detection Results:")
    for task in todo_list.tasks:
        print(f"  {task.id}: type={task.type.value}, tool={task.parameters.get('tool_name', 'N/A')}")
    
    # P1 Test: Verify priority order
    print(f"\n📈 P1 Test: Priority Order (lower = higher priority):")
    sorted_tasks = sorted(todo_list.tasks, key=lambda t: t.priority)
    for task in sorted_tasks:
        deps = task.dependencies or []
        print(f"  {task.id}: priority={task.priority}, dependencies={deps}")
    
    # Verify P1: No task has lower priority than its dependencies
    print(f"\n✅ P1 Verification: Checking dependency-priority consistency...")
    priority_map = {t.id: t.priority for t in todo_list.tasks}
    all_valid = True
    for task in todo_list.tasks:
        for dep_id in task.dependencies:
            if dep_id in priority_map:
                if task.priority <= priority_map[dep_id]:
                    print(f"  ❌ ERROR: {task.id} (p={task.priority}) should have higher priority than {dep_id} (p={priority_map[dep_id]})")
                    all_valid = False
    if all_valid:
        print(f"  ✅ All tasks have correct priority order!")
    
    # Generate markdown
    print("\n📄 Generated Markdown:")
    print("-" * 40)
    print(generate_todo_list_markdown(todo_list))

