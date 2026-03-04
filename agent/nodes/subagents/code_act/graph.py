"""
CodeAct Agent Subgraph

Responsible for code generation and execution, including:
1. MCP tool call code generation and execution
2. General code generation and execution
3. Code fixing (fixing code errors and parameter errors)
4. Code execution and error handling
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, START, END
import sys
import os
import subprocess
import json
from pathlib import Path
from enum import Enum

# Import main graph state and task models
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import SubTask, GlobalState
from utils.llm_factory import create_code_llm
from nodes.subagents.code_act.prompt import (
    MCP_TOOL_CODE_SYSTEM_PROMPT,
    get_mcp_tool_code_user_prompt,
    CODEACT_SYSTEM_PROMPT,
    get_codeact_user_prompt,
    FIX_CODE_SYSTEM_PROMPT,
    get_fix_code_user_prompt,
    FIX_PARAMETER_SYSTEM_PROMPT,
    get_fix_parameter_user_prompt
)
from nodes.subagents.code_act.trajectory import (
    CodeTrajectory,
    TrajectoryPool,
    TrajectoryStatus,
    build_react_steps_from_trajectory,
    summarize_react_steps
)
from nodes.subagents.code_act.revision import (
    RevisionPlan,
    RevisionStrategy,
    create_revision_plan,
    execute_revision_plan
)
from nodes.subagents.code_act.todo_list import (
    TodoTask,
    TodoTaskStatus,
    TodoTaskType,
    TodoList,
    TodoListManager
)
# P2: Import File Parameter Table
from nodes.subagents.code_act.file_param_table import (
    FileParameter,
    FileParameterTable,
    FileSource,
    get_parameter_inference_prompt,
    create_file_param_from_user_input,
    create_file_param_from_task_output,
    extract_file_info_from_task_result
)

# ===================== CodeAct Subgraph State Model =====================

class CodeActExecutionMode(str, Enum):
    """CodeAct execution mode"""
    MCP_TOOL = "mcp_tool"  # Generate code to call MCP tools
    CODEACT = "codeact"  # Generate code based on task description
    FIX_CODE = "fix_code"  # Fix code errors
    FIX_PARAMETER = "fix_parameter"  # Fix parameter errors


class CodeActState(BaseModel):
    """CodeAct subgraph state"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        from_attributes=True  # Allow creation from attributes (Pydantic v2)
    )
    
    task: SubTask = Field(description="Task to execute")
    task_description: str = Field(description="Task description")
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="List of tools matched for the task")
    inputs: List[str] = Field(default_factory=list, description="Task input parameters")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters (parsed)")
    execution_mode: CodeActExecutionMode = Field(description="Execution mode")
    
    # All available tools for data transformation suggestions (from reference service)
    all_available_tools: List[Dict[str, Any]] = Field(default_factory=list, description="All available tools for potential data transformation")
    
    # Fix related
    previous_code: Optional[str] = Field(default=None, description="Previous code (for fixing)")
    previous_error: Optional[str] = Field(default=None, description="Previous error message (for fixing)")
    error_category: Optional[str] = Field(default=None, description="Error category")
    revision_plan: Optional[Any] = Field(default=None, description="Revision plan (for intelligent fixing)")
    revision_iteration: int = Field(default=0, description="Revision iteration count")
    
    # Suggested data transformation tool (from Revision analysis)
    suggested_transform_tool: Optional[Dict[str, Any]] = Field(default=None, description="Suggested tool for data transformation")
    
    # Code generation and execution results
    generated_code: Optional[str] = Field(default=None, description="Generated code")
    execution_result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")
    
    # Pre-execution code (e.g., CSV to FASTA conversion)
    # This code will be prepended to generated code and executed in the SAME sandbox session
    pre_execution_code: Optional[str] = Field(default=None, description="Code to execute before main code (e.g., file format conversion)")
    
    # Trajectory recording (SE-Agent style)
    trajectory_history: List[CodeTrajectory] = Field(default_factory=list, description="Trajectory history for current task")
    trajectory_pool_id: Optional[str] = Field(default=None, description="Associated trajectory pool ID")
    current_trajectory: Optional[CodeTrajectory] = Field(default=None, description="Currently recording trajectory")
    
    # Parent state reference
    parent_state: Optional[GlobalState] = Field(default=None, description="Main graph state reference")
    
    # ===================== Todo List Fields (New Architecture) =====================
    # Todo list management
    todo_list_path: Optional[str] = Field(default=None, description="Path to todo-list.md in sandbox")
    todo_list: Optional[TodoList] = Field(default=None, description="Cached todo list")
    current_todo_task: Optional[TodoTask] = Field(default=None, description="Current task being executed")
    todo_manager: Optional[TodoListManager] = Field(default=None, description="Todo list manager instance")
    
    # ===================== Data Exploration Fields =====================
    # Data exploration results (populated by explore_data_node)
    data_exploration_result: Optional[Dict[str, Any]] = Field(default=None, description="Data exploration result (columns, dtypes, sample, etc.)")
    needs_data_exploration: bool = Field(default=False, description="Whether this task needs data exploration")
    explored_columns: Optional[List[str]] = Field(default=None, description="Discovered column names from exploration")
    
    # ===================== Output Validation Fields =====================
    # Output validation results (populated by validate_output_node)
    output_validation_result: Optional[Dict[str, Any]] = Field(default=None, description="Output validation result")
    output_constraints: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Output constraints from task parameters")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings collected")
    
    # ===================== File Parameter Table Fields (P2) =====================
    # File parameter table for dynamic parameter inference
    file_parameter_table: Optional[FileParameterTable] = Field(default=None, description="File parameter table for the session")
    inferred_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Parameters inferred by LLM")
    user_input_context: Optional[str] = Field(default=None, description="Original user input for context")
    completed_tasks_summary: List[Dict[str, Any]] = Field(default_factory=list, description="Summary of completed tasks")


# ===================== Trajectory Recording Helper Functions =====================

def _start_trajectory(state: CodeActState) -> CodeTrajectory:
    """
    Start recording new trajectory
    
    Args:
        state: CodeAct state
    
    Returns:
        Newly created trajectory
    """
    from datetime import datetime
    import hashlib
    
    # Handle case where execution_mode may be enum or string
    execution_mode_value = state.execution_mode
    if hasattr(execution_mode_value, 'value'):
        execution_mode_value = execution_mode_value.value
    else:
        execution_mode_value = str(execution_mode_value)
    
    trajectory = CodeTrajectory(
        trajectory_id="",  # Generated later
        task_id=state.task.task_id,
        execution_mode=execution_mode_value,
        generated_code="",  # Filled later
        status=TrajectoryStatus.PARTIAL,  # Initial status
        parameters=state.parameters.copy(),
        tools=state.tools.copy(),
        inputs=state.inputs.copy()
    )
    
    # Generate trajectory ID
    timestamp_str = trajectory.timestamp.strftime("%Y%m%d_%H%M%S_%f")
    task_hash = hashlib.md5(state.task.task_id.encode()).hexdigest()[:8]
    trajectory.trajectory_id = f"{state.task.task_id}_{timestamp_str}_{task_hash}"
    
    return trajectory


def _update_trajectory_code(trajectory: CodeTrajectory, code: str, generation_time: float = 0.0):
    """
    Update trajectory code generation information
    
    Args:
        trajectory: Trajectory
        code: Generated code
        generation_time: Generation time
    """
    trajectory.generated_code = code
    trajectory.code_length = len(code)
    trajectory.code_generation_time = generation_time


def _finalize_trajectory(trajectory: CodeTrajectory, execution_result: Dict[str, Any], execution_time: float = 0.0):
    """
    Finalize trajectory recording
    
    Args:
        trajectory: Trajectory
        execution_result: Execution result
        execution_time: Execution time
    """
    trajectory.execution_result = execution_result
    trajectory.execution_time = execution_time
    
    # Set status based on execution result
    if execution_result.get("status") == "success":
        trajectory.status = TrajectoryStatus.SUCCESS
    else:
        trajectory.status = TrajectoryStatus.FAILED
        trajectory.error_type = execution_result.get("error_type")
        trajectory.error_message = execution_result.get("error")
        trajectory.error_traceback = execution_result.get("error_traceback")
        trajectory.error_category = execution_result.get("error_category")


def _save_trajectory_to_pool(state: CodeActState, trajectory: CodeTrajectory):
    """
    Save trajectory to trajectory pool
    
    Args:
        state: CodeAct state
        trajectory: Trajectory to save
    """
    # Add to trajectory history
    state.trajectory_history.append(trajectory)

    # Add React trace summary for logging and storage
    react_steps = build_react_steps_from_trajectory(trajectory)
    if react_steps:
        trajectory.metadata["react_steps"] = [step.model_dump() for step in react_steps]
        trajectory.metadata["react_summary"] = summarize_react_steps(react_steps, max_steps=5)
        print(f"  ℹ React summary: {trajectory.metadata['react_summary']}")
    
    # TODO: Integrate TrajectoryPool for persistent storage
    # Currently saved in memory, can add persistence later


# ===================== CodeAct Nodes =====================

def _generate_mcp_tool_code_directly(
    tool_name: str,
    parameters: Dict[str, Any],
    task_description: str = ""
) -> str:
    """
    Generate MCP tool call code directly using template, without LLM.
    
    This ensures:
    1. Correct import: from core.tool_interface import call_tool
    2. Proper parameter passing
    3. No simulation or placeholder code
    4. Real tool execution
    
    Args:
        tool_name: Name of the MCP tool to call
        parameters: Parameters to pass to the tool
        task_description: Optional task description for context
    
    Returns:
        Generated Python code that calls the MCP tool
    """
    # Filter out non-parameter fields (like tool_name, output_file descriptions)
    # These are often included in task parameters but shouldn't be passed to the tool
    filtered_params = {}
    # Meta fields that should NOT be passed to tools
    meta_fields = {
        'tool_name',      # Tool identification
        'output_file',    # Output description (not a real parameter)
        'description',    # Parameter description
        'type',           # Parameter type
        'required',       # Required flag
        'sandbox_dir',    # Environment parameter (should not be passed to tools)
        'todo_list_path', # Environment parameter (should not be passed to tools)
        'session_id',     # Session identifier (should not be passed to tools)
    }
    
    for key, value in parameters.items():
        # Skip meta fields and description dicts
        if key in meta_fields:
            continue
        if isinstance(value, dict) and 'description' in value and 'type' in value:
            # This is a parameter schema, not actual parameter value
            continue
        filtered_params[key] = value
    
    # Generate the code
    # Note: Use string formatting carefully - we want the tool_name value to be embedded,
    # not a variable reference in the generated code
    code = f'''# MCP Tool Call: {tool_name}
# Task: {task_description[:100] if task_description else "N/A"}
# Generated by direct template (no LLM simulation)

from core.tool_interface import call_tool
import json

# Tool: {tool_name}
# Parameters: {json.dumps(filtered_params, ensure_ascii=False, default=str)[:200]}

tool_result = call_tool(
    tool_name="{tool_name}",
    parameters={repr(filtered_params)}
)

# Process result
if tool_result["status"] == "success":
    # Tool call successful - return actual output
    result = {{
        "status": "success",
        "output": tool_result["output"],
        "tool_name": tool_result.get("tool_name", "{tool_name}"),
        "service_id": tool_result.get("service_id"),
        "execution_time_ms": tool_result.get("execution_time_ms")
    }}
    print(f"[MCP Tool] {tool_name} executed successfully")
    print(f"[MCP Tool] Output type: {{type(tool_result['output']).__name__}}")
else:
    # Tool call failed - return error details
    result = {{
        "status": "failed",
        "error": tool_result.get("error", "Unknown error"),
        "error_type": tool_result.get("error_type", "ToolError"),
        "tool_name": tool_result.get("tool_name", "{tool_name}"),
        "service_id": tool_result.get("service_id")
    }}
    print(f"[MCP Tool] {tool_name} failed: {{tool_result.get('error', 'Unknown error')}}")
'''
    return code


def _generate_code_with_llm(
    system_prompt: str,
    user_prompt: str,
    fallback_code: str = None
) -> str:
    """
    Use LLM to generate code
    
    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        fallback_code: Fallback code (used when LLM is unavailable)
    
    Returns:
        Generated code
    """
    llm = create_code_llm()
    
    if not llm:
        print("  ⚠ LLM unavailable, using fallback code")
        return fallback_code or "# LLM unavailable, cannot generate code"
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        code = response.content.strip()
        
        # Remove possible markdown code block markers
        if code.startswith("```python"):
            code = code[9:]  # Remove ```python
        elif code.startswith("```"):
            code = code[3:]  # Remove ```
        
        if code.endswith("```"):
            code = code[:-3]  # Remove trailing ```
        
        code = code.strip()
        
        if not code:
            print("  ⚠ LLM returned empty code, using fallback code")
            return fallback_code or "# LLM returned empty code"
        
        print(f"  ✓ LLM code generation successful (length: {len(code)} characters)")
        return code
    
    except Exception as e:
        print(f"  ⚠ LLM code generation failed: {e}, using fallback code")
        return fallback_code or f"# LLM code generation failed: {e}"


def _check_sandbox_available(state: CodeActState, provider: Optional[str] = None) -> tuple[bool, str]:
    """
    Check if sandbox environment is available
    
    Args:
        state: CodeAct state
    
    Returns:
        (is_available, sandbox_directory_path)
    """
    provider = (provider or _get_sandbox_provider()).lower()
    if provider == "opensandbox":
        return True, "opensandbox"
    return _check_local_sandbox_available(state)


def _check_local_sandbox_available(state: CodeActState) -> tuple[bool, str]:
    """Check local filesystem sandbox availability."""
    sandbox_dir = None
    
    # Try to get sandbox directory from parent_state
    if state.parent_state and hasattr(state.parent_state, 'sandbox_dir'):
        sandbox_dir = state.parent_state.sandbox_dir
    
    # If no sandbox directory, check environment variable or use default
    if not sandbox_dir or sandbox_dir == "DEFAULT_SANDBOX_DIR":
        import os
        sandbox_dir = os.getenv("SANDBOX_DIR")
        if not sandbox_dir:
            # Use temporary directory as fallback
            import tempfile
            sandbox_dir = tempfile.gettempdir()
    
    # Check if directory exists or can be created
    try:
        from pathlib import Path
        sandbox_path = Path(sandbox_dir)
        if not sandbox_path.exists():
            sandbox_path.mkdir(parents=True, exist_ok=True)
        # Check if writable
        test_file = sandbox_path / ".test_write"
        try:
            test_file.write_text("test")
            test_file.unlink()
            return True, str(sandbox_path)
        except Exception:
            return False, str(sandbox_path)
    except Exception:
        return False, sandbox_dir or ""


def _get_sandbox_provider() -> str:
    """Get sandbox provider selection for CodeAct."""
    return os.getenv("CODEACT_SANDBOX_PROVIDER", "local").lower()


def _parse_execution_output(stdout: str, stderr: str, returncode: Optional[int]) -> Dict[str, Any]:
    """Parse CodeAct execution output from stdout/stderr.
    
    Handles the following cases:
    1. Normal execution with __CODEACT_RESULT__ marker in stdout
    2. Execution failure with non-zero returncode
    3. Import/Runtime errors in stderr even if returncode is 0/None
    4. Missing dependencies (ModuleNotFoundError, ImportError)
    """
    result_marker = "__CODEACT_RESULT__"
    parsed_result = None
    for line in (stdout or "").splitlines():
        if result_marker in line:
            payload = line.split(result_marker, 1)[1]
            try:
                parsed_result = json.loads(payload)
            except Exception:
                parsed_result = payload

    # Detect errors from stderr even if returncode is 0/None
    # This handles cases where the sandbox doesn't properly report return codes
    stderr_error_type = None
    stderr_error_msg = None
    
    if stderr:
        stderr_lower = stderr.lower()
        # Detect common Python errors from stderr
        if "modulenotfounderror" in stderr_lower or "no module named" in stderr_lower:
            stderr_error_type = "ModuleNotFoundError"
            # Extract the missing module name
            import re
            match = re.search(r"No module named '([^']+)'", stderr, re.IGNORECASE)
            if match:
                stderr_error_msg = f"Missing dependency: {match.group(1)}"
            else:
                stderr_error_msg = "Missing Python dependency"
        elif "importerror" in stderr_lower:
            stderr_error_type = "ImportError"
            stderr_error_msg = stderr.split('\n')[-1] if '\n' in stderr else stderr[:500]
        elif "traceback" in stderr_lower and "error" in stderr_lower:
            # Generic error with traceback
            lines = stderr.strip().split('\n')
            # Get the last line which usually contains the error
            for line in reversed(lines):
                if line.strip() and ('error' in line.lower() or 'exception' in line.lower()):
                    stderr_error_msg = line.strip()
                    # Try to extract error type
                    if ':' in line:
                        error_parts = line.split(':', 1)
                        stderr_error_type = error_parts[0].strip()
                    break
            if not stderr_error_msg:
                stderr_error_msg = stderr[-500:] if len(stderr) > 500 else stderr

    error = None
    error_type = None
    
    # IMPORTANT: Always check stderr for errors, regardless of parsed_result
    # This handles cases where the code crashed before __CODEACT_RESULT__ was printed
    # or where the sandbox doesn't properly report return codes
    
    if isinstance(parsed_result, dict):
        status = parsed_result.get("status", "success")
        output = parsed_result.get("output")
        error = parsed_result.get("error")
        error_type = parsed_result.get("error_type")
        
        # CRITICAL FIX: Check stderr for errors even if parsed_result indicates success
        # This handles the case where import fails before the try-except block
        if stderr_error_type or stderr_error_msg:
            # stderr has error indicators - this is a real failure
            if status == "success":
                # Result says success but stderr has errors - trust stderr
                status = "failed"
                error = stderr_error_msg or error
                error_type = stderr_error_type or error_type
                print(f"  ⚠ Detected error in stderr despite 'success' status: {error_type}: {error}")
            elif not error:
                # Result says failed but no error details - use stderr info
                error = stderr_error_msg or "Execution failed"
                error_type = error_type or stderr_error_type
    else:
        # No __CODEACT_RESULT__ found - execution likely failed before completing
        # Check returncode first
        returncode_indicates_failure = returncode is not None and returncode != 0
        
        # Also check if stderr has errors (even if returncode is 0/None)
        stderr_indicates_failure = stderr_error_type is not None or stderr_error_msg is not None
        
        if returncode_indicates_failure or stderr_indicates_failure:
            status = "failed"
            output = (stdout or "")[-1000:] if stdout else None
            error = stderr_error_msg or ((stderr or "")[-2000:] if stderr else (stdout or "")[-2000:]) or "Execution failed"
            error_type = stderr_error_type
        else:
            # No explicit error, but also no result marker - this might be unexpected
            # Return the stdout as output but mark as potentially incomplete
            status = "success"
            output = parsed_result or ((stdout or "")[-1000:] if stdout else None)
            if stderr:
                # Include warning about stderr even in "success" case
                error = f"Warning: stderr output detected: {stderr[-500:]}" if len(stderr) > 500 else f"Warning: stderr output detected: {stderr}"

    return {
        "status": status,
        "output": output,
        "error": error,
        "error_type": error_type,
    }


def _build_subprocess_kwargs() -> Dict[str, Any]:
    """Build subprocess kwargs with resource limits when supported."""
    kwargs: Dict[str, Any] = {}
    if os.name != "nt":
        try:
            import resource
            max_memory_mb = int(os.getenv("CODEACT_MAX_MEMORY_MB", "0"))
            max_cpu_seconds = int(os.getenv("CODEACT_MAX_CPU_SECONDS", "0"))

            def _set_limits():
                if max_memory_mb > 0:
                    bytes_limit = max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
                if max_cpu_seconds > 0:
                    resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds))

            if max_memory_mb > 0 or max_cpu_seconds > 0:
                kwargs["preexec_fn"] = _set_limits
        except Exception:
            pass
    return kwargs


def _strip_code_fences(code: str) -> str:
    """
    Remove all markdown code fences from generated code.
    
    This handles cases where LLM returns code wrapped in ```python ... ```
    or where code contains embedded markdown examples.
    
    Args:
        code: Code that may contain markdown code fences
        
    Returns:
        Clean code with all code fences removed
    """
    if not code:
        return code
    
    lines = code.split('\n')
    cleaned_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detect code fence start/end
        if stripped.startswith('```'):
            # Check if this is a code fence with language tag (like ```python)
            # or a closing fence (just ```)
            in_code_block = not in_code_block
            continue  # Skip the fence line itself
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def _ensure_code_executable(code: str, has_sandbox: bool, sandbox_dir: str = None) -> str:
    """
    Ensure code is executable, add necessary wrapping and error handling
    
    Args:
        code: Original code
        has_sandbox: Whether sandbox environment exists
        sandbox_dir: Sandbox directory path
    
    Returns:
        Executable code
    """
    result_marker = "__CODEACT_RESULT__"
    
    # First, strip any markdown code fences that LLM might have included
    code = _strip_code_fences(code)
    
    # Auto-install wrapper for missing dependencies (especially for OpenSandbox)
    # Uses uv for faster installation (10-100x faster than pip)
    auto_install_wrapper = '''
# Auto-install missing dependencies wrapper (uses uv for speed)
import sys
import subprocess
import shutil

def _auto_install_if_missing(module_name, package_name=None):
    """Try to import a module, install if missing using uv (fallback to pip)."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        package = package_name or module_name
        print(f"[AutoInstall] Installing missing dependency: {package}")
        
        # Try uv first (much faster than pip)
        if shutil.which("uv"):
            try:
                subprocess.check_call(
                    ["uv", "pip", "install", "--system", "-q", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"[AutoInstall] Successfully installed via uv: {package}")
                return True
            except Exception as e:
                print(f"[AutoInstall] uv install failed, trying pip: {e}")
        
        # Fallback to pip with --break-system-packages for Ubuntu 24.04+
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[AutoInstall] Successfully installed via pip: {package}")
            return True
        except Exception as e:
            print(f"[AutoInstall] Failed to install {package}: {e}")
            return False

def _ensure_common_dependencies():
    """Ensure common data science dependencies are available."""
    common_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
    ]
    for module, package in common_packages:
        _auto_install_if_missing(module, package)

# Auto-detect and install imports from code
# IMPORTANT: Call this BEFORE any imports that might need these packages
_ensure_common_dependencies()
'''

    # Check if code has common imports that might need auto-installation
    needs_auto_install = any([
        "import pandas" in code,
        "from pandas" in code,
        "import numpy" in code,
        "from numpy" in code,
        "import sklearn" in code,
        "from sklearn" in code,
        "import scipy" in code,
        "from scipy" in code,
    ])

    # If code already contains result setting, return directly
    if "result" in code and ("=" in code.split("result")[0] or "result = {" in code):
        # Ensure code has complete error handling
        if "try:" not in code or "except" not in code:
            # Add error handling wrapper
            wrapped_code = f"""
try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
    # Ensure result variable exists
    if 'result' not in locals():
        result = {{"status": "success", "output": "Code execution completed"}}
except Exception as e:
    result = {{
        "status": "failed",
        "error": str(e),
        "error_type": type(e).__name__
    }}
"""
            code = wrapped_code.strip()
        # Ensure result marker is printed for subprocess parsing
        if result_marker not in code:
            code += f"""
try:
    import json as _json
    print("{result_marker}" + _json.dumps(result, ensure_ascii=False))
except Exception:
    try:
        print("{result_marker}" + str(result))
    except Exception:
        pass
"""
        
        # Add auto-install wrapper if needed (for sandbox execution)
        if needs_auto_install and has_sandbox:
            code = auto_install_wrapper + code
        
        return code
    
    # If code doesn't have result setting, add it
    if "result" not in code:
        code += "\nresult = {\"status\": \"success\", \"output\": \"Code execution completed\"}"
    
    # Add error handling
    if "try:" not in code:
        wrapped_code = f"""
try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    result = {{
        "status": "failed",
        "error": str(e),
        "error_type": type(e).__name__
    }}
"""
        code = wrapped_code.strip()

    if result_marker not in code:
        code += f"""
try:
    import json as _json
    print("{result_marker}" + _json.dumps(result, ensure_ascii=False))
except Exception:
    try:
        print("{result_marker}" + str(result))
    except Exception:
        pass
"""
    
    # Add auto-install wrapper if needed (for sandbox execution)
    if needs_auto_install and has_sandbox:
        code = auto_install_wrapper + code
    
    return code


def validate_generated_code(
    code: str,
    mode: CodeActExecutionMode,
    enforce_tool_call: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Lightweight validation for generated code.

    Ensures MCP tool calls use the unified call_tool interface and that code is
    free from obvious formatting artifacts.
    
    CRITICAL: Also detects simulation/placeholder code patterns that must be rejected:
    - "return 'Tool called successfully'"
    - "# Simulate the call"
    - Mock implementations without actual tool calls
    """
    issues: List[str] = []
    if not code or not code.strip():
        issues.append("Generated code is empty.")
    if "```" in code:
        issues.append("Generated code contains markdown code fences.")
    if enforce_tool_call is None:
        enforce_tool_call = mode == CodeActExecutionMode.MCP_TOOL

    if enforce_tool_call:
        if "call_tool" not in code:
            issues.append("MCP tool code must call call_tool().")
        if "invoke_mcp_tool_sync" in code:
            issues.append("MCP tool code should not call invoke_mcp_tool_sync().")
        
        # ============================================================
        # CRITICAL: Detect simulation/placeholder code patterns
        # These are ABSOLUTELY FORBIDDEN in MCP tool calls
        # ============================================================
        simulation_patterns = [
            "return \"Tool called successfully\"",
            "return 'Tool called successfully'",
            "# Simulate the call",
            "# Simulate",
            "Simulate the call to a tool",
            "mock",
            "# Mock",
            "# Placeholder",
            "# placeholder",
            "pass  # TODO",
            "# This is a simulation",
            "def call_tool():",  # Defines its own call_tool instead of importing
        ]
        
        code_lower = code.lower()
        for pattern in simulation_patterns:
            if pattern.lower() in code_lower:
                issues.append(
                    f"CRITICAL: Simulation/placeholder code detected: '{pattern}'. "
                    "MCP tool code MUST call the real tool using 'from core.tool_interface import call_tool'. "
                    "Simulation and mock implementations are STRICTLY FORBIDDEN."
                )
        
        # Check for correct import
        if "from core.tool_interface import call_tool" not in code:
            # Check if it imports from wrong module
            if "from mcp import call_tool" in code:
                issues.append(
                    "CRITICAL: Wrong import path. Use 'from core.tool_interface import call_tool', NOT 'from mcp import call_tool'."
                )
            elif "import call_tool" in code and "core.tool_interface" not in code:
                issues.append(
                    "CRITICAL: call_tool must be imported from core.tool_interface. "
                    "Use: from core.tool_interface import call_tool"
                )

    if issues:
        return {
            "valid": False,
            "error": " ".join(issues),
            "error_type": "CodeValidationError",
            "error_category": "code_error"
        }
    return {"valid": True}


def _build_validation_error_code(error_message: str) -> str:
    """Build a minimal executable snippet that reports validation failure."""
    return f"""result = {{
    "status": "failed",
    "error": {repr(error_message)},
    "error_type": "CodeValidationError",
    "error_category": "code_error"
}}"""


def _generate_fix_code_for_state(state: "CodeActState", has_sandbox: bool, sandbox_dir: str) -> "CodeActState":
    """Generate fix code using existing fix_code flow."""
    previous_code = state.previous_code or ""
    previous_error = state.previous_error or ""
    error_category = state.error_category

    if not previous_code or not previous_error:
        state.generated_code = "# Missing necessary fix information (original code or error information)"
        return state

    if state.revision_plan:
        print(f"  🔄 Using Revision plan for intelligent fix (strategy: {state.revision_plan.strategy.value})")
        print(f"     Root cause: {state.revision_plan.root_cause[:100]}...")
        print(f"     Orthogonal strategy: {'Yes' if state.revision_plan.orthogonal else 'No'}")

        generated_code = execute_revision_plan(
            revision_plan=state.revision_plan,
            original_code=previous_code,
            original_error=previous_error,
            task_description=state.task_description,
            parameters=state.parameters
        )
    else:
        user_prompt = get_fix_code_user_prompt(
            previous_code=previous_code,
            previous_error=previous_error,
            error_category=error_category
        )

        fallback_code = f"""
# Fix code errors
# Previous error: {previous_error}
# Original code:
{previous_code}

# Fallback fix: try basic fix
try:
    {previous_code}
    result = {{"status": "success", "output": "Code execution successful"}}
except Exception as e:
    result = {{"status": "failed", "error": str(e)}}
"""

        generated_code = _generate_code_with_llm(
            system_prompt=FIX_CODE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_code=fallback_code
        )

    state.generated_code = _ensure_code_executable(
        generated_code,
        has_sandbox=has_sandbox,
        sandbox_dir=sandbox_dir
    )
    return state


def _find_and_activate_venv(agent_dir: Path) -> Optional[Path]:
    """
    Find and activate project virtual environment
    
    Args:
        agent_dir: Agent directory path
    
    Returns:
        Python interpreter path of virtual environment, returns None if not found
    """
    # Find virtual environment directory (.venv)
    venv_paths = [
        agent_dir / ".venv",
        agent_dir.parent / ".venv",
        Path.cwd() / ".venv",
    ]
    
    for venv_path in venv_paths:
        if venv_path.exists() and venv_path.is_dir():
            # Determine Python interpreter path
            if os.name == 'nt':  # Windows
                python_exe = venv_path / "Scripts" / "python.exe"
            else:  # Unix/Linux
                python_exe = venv_path / "bin" / "python"
            
            if python_exe.exists():
                print(f"  ✓ Found virtual environment: {venv_path}")
                print(f"     Python interpreter: {python_exe}")
                return python_exe
    
    return None


def _activate_venv_in_sys_path(venv_python: Path) -> None:
    """
    Activate virtual environment in sys.path
    
    Args:
        venv_python: Python interpreter path of virtual environment
    """
    import sys
    import site
    
    venv_dir = venv_python.parent.parent
    
    # Determine site-packages path
    if os.name == 'nt':  # Windows
        site_packages = venv_dir / "Lib" / "site-packages"
    else:  # Unix/Linux
        import sysconfig
        site_packages = Path(sysconfig.get_path('purelib', vars={'base': str(venv_dir)}))
    
    # Add virtual environment's site-packages to the front of sys.path
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
        print(f"  ✓ Added virtual environment site-packages to sys.path: {site_packages}")
    
    # Also add virtual environment root directory (for importing project modules)
    if str(venv_dir) not in sys.path:
        sys.path.insert(0, str(venv_dir))
    
    # Execute site.addsitedir to ensure virtual environment is properly activated
    try:
        site.addsitedir(str(site_packages))
    except Exception as e:
        print(f"  ⚠ Cannot add virtual environment site-packages: {e}")
    
    # Verify if key packages can be imported
    try:
        import langchain_mcp_adapters
        print(f"  ✓ Virtual environment activated successfully, can import langchain-mcp-adapters")
    except ImportError:
        print(f"  ⚠ Warning: Virtual environment activated, but cannot import langchain-mcp-adapters")


# ===================== Data Exploration Node =====================

DATA_EXPLORATION_PROMPT = """You are a data exploration expert. Analyze the provided data file and extract its structure.

# Task
Explore the data file and return a JSON object with the following information:
- columns: list of column names
- dtypes: data types of each column
- shape: (rows, columns)
- sample_data: first 3 rows as a list of dictionaries
- null_counts: number of null values per column
- suggestions: brief suggestions for which columns might be useful for the task

# Code Template
```python
import pandas as pd
import json
import os

file_path = "{file_path}"
task_description = "{task_description}"

result = {{"status": "failed", "error": None, "output": None}}

try:
    if not os.path.exists(file_path):
        result["error"] = f"File not found: {{file_path}}"
    else:
        # Determine file type and read
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\\t')
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            # Try CSV as default
            df = pd.read_csv(file_path)
        
        # Extract structure
        exploration = {{
            "columns": list(df.columns),
            "dtypes": {{col: str(dtype) for col, dtype in df.dtypes.items()}},
            "shape": list(df.shape),
            "sample_data": df.head(3).to_dict(orient='records'),
            "null_counts": {{col: int(df[col].isna().sum()) for col in df.columns}},
            "unique_counts": {{col: int(df[col].nunique()) for col in df.columns}}
        }}
        
        # Print exploration results for debugging
        print(f"[Data Exploration] Columns: {{exploration['columns']}}")
        print(f"[Data Exploration] Shape: {{exploration['shape']}}")
        print(f"[Data Exploration] Dtypes: {{exploration['dtypes']}}")
        
        result["status"] = "success"
        result["output"] = exploration
        
except Exception as e:
    result["error"] = str(e)
    print(f"[Data Exploration Error] {{e}}")

# Output result as JSON
print("__EXPLORATION_RESULT__" + json.dumps(result, ensure_ascii=False, default=str))
```

Return only Python code."""


def codeact_explore_data_node(state: CodeActState) -> CodeActState:
    """
    CodeAct node: Explore data structure
    
    This node explores input data files to understand their structure before code generation.
    This helps the code generation node to:
    1. Know exact column names (no guessing!)
    2. Understand data types
    3. Identify potential issues
    
    The exploration is done by generating and executing a simple exploration script.
    """
    import time
    
    # Check if task needs data exploration
    parameters = state.parameters or {}
    
    # Determine if data exploration is needed
    needs_exploration = False
    file_to_explore = None
    
    # Check for common data file parameters
    data_file_params = ['prediction_file', 'input_file', 'data_file', 'file_path', 'csv_file', 'input_path']
    for param in data_file_params:
        if param in parameters and parameters[param]:
            file_to_explore = parameters[param]
            needs_exploration = True
            break
    
    # Also check inputs list for file paths
    if not needs_exploration and state.inputs:
        for inp in state.inputs:
            if isinstance(inp, str) and any(ext in inp.lower() for ext in ['.csv', '.tsv', '.xlsx', '.json']):
                file_to_explore = inp
                needs_exploration = True
                break
    
    state.needs_data_exploration = needs_exploration
    
    if not needs_exploration:
        print(f"  ℹ Data exploration: Not needed for this task")
        return state
    
    print(f"  🔍 Starting data exploration for: {file_to_explore}")
    
    # Check sandbox availability
    sandbox_provider = _get_sandbox_provider()
    has_sandbox, sandbox_dir = _check_sandbox_available(state, provider=sandbox_provider)
    
    if not has_sandbox:
        print(f"  ⚠ Data exploration: No sandbox available, skipping")
        return state
    
    # Generate exploration code
    exploration_code = DATA_EXPLORATION_PROMPT.format(
        file_path=file_to_explore,
        task_description=state.task_description.replace('"', '\\"')
    )
    
    # Ensure code is executable
    exploration_code = _ensure_code_executable(exploration_code, has_sandbox=has_sandbox, sandbox_dir=sandbox_dir)
    
    # Execute exploration code
    print(f"  ▶ Executing data exploration...")
    
    try:
        # Use the same execution mechanism
        if sandbox_provider == "opensandbox":
            from utils.opensandbox_executor import run_code_in_opensandbox_sync
            
            timeout_seconds = int(os.getenv("CODEACT_TIMEOUT_SECONDS", "180"))
            # 每次创建新沙盒，不再复用（避免连接失败问题）
            # 文件通过 session_id 组织在同一目录下
            
            exploration_result = run_code_in_opensandbox_sync(
                code=exploration_code,
                task_id=f"{state.task.task_id}_explore",
                timeout_seconds=30,  # Short timeout for exploration
                existing_sandbox_id=None,  # 不复用，每次创建新沙盒
                keep_alive=False
            )
            
            stdout = exploration_result.get("stdout", "")
        else:
            # Local sandbox execution
            import subprocess
            sandbox_path = Path(sandbox_dir)
            sandbox_path.mkdir(parents=True, exist_ok=True)
            code_file = sandbox_path / f"explore_{state.task.task_id}.py"
            code_file.write_text(exploration_code, encoding="utf-8")
            
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(sandbox_path)
            )
            stdout = result.stdout
        
        # Parse exploration result
        import json
        if "__EXPLORATION_RESULT__" in stdout:
            result_json = stdout.split("__EXPLORATION_RESULT__")[-1].strip()
            try:
                parsed_result = json.loads(result_json)
                if parsed_result.get("status") == "success":
                    state.data_exploration_result = parsed_result.get("output", {})
                    state.explored_columns = state.data_exploration_result.get("columns", [])
                    print(f"  ✅ Data exploration successful!")
                    print(f"     Columns found: {state.explored_columns}")
                    print(f"     Shape: {state.data_exploration_result.get('shape')}")
                else:
                    print(f"  ⚠ Data exploration failed: {parsed_result.get('error')}")
            except json.JSONDecodeError as e:
                print(f"  ⚠ Failed to parse exploration result: {e}")
        else:
            print(f"  ⚠ No exploration result found in output")
            
    except Exception as e:
        print(f"  ⚠ Data exploration error: {e}")
    
    return state


# ===================== Output Validation Node =====================

def codeact_validate_output_node(state: CodeActState) -> CodeActState:
    """
    CodeAct node: Validate output
    
    This node validates the execution output against expected constraints.
    If output violates constraints, it can trigger a retry with warnings.
    
    Constraints can be:
    1. Provided in task parameters (output_constraints)
    2. Inferred from task type (e.g., F1 score should be > 0)
    """
    import time
    
    execution_result = state.execution_result
    
    if not execution_result:
        print(f"  ℹ Output validation: No execution result to validate")
        return state
    
    if execution_result.get("status") != "success":
        print(f"  ℹ Output validation: Skipping (execution failed)")
        return state
    
    output = execution_result.get("output", {})
    
    if not output or not isinstance(output, dict):
        print(f"  ℹ Output validation: No structured output to validate")
        return state
    
    print(f"  🔍 Starting output validation...")
    
    # Get constraints from task parameters or state
    parameters = state.parameters or {}
    constraints = parameters.get("output_constraints") or state.output_constraints or {}
    
    # If no explicit constraints, try to infer from task description
    if not constraints:
        constraints = _infer_output_constraints(state.task_description, output)
    
    if not constraints:
        print(f"  ℹ Output validation: No constraints defined")
        return state
    
    # Validate each constraint
    warnings = []
    all_passed = True
    
    for field_name, constraint in constraints.items():
        if field_name not in output:
            continue
            
        value = output.get(field_name)
        min_val = constraint.get("min")
        max_val = constraint.get("max")
        description = constraint.get("description", "")
        
        # Check min constraint
        if min_val is not None and value is not None:
            if value < min_val:
                warning = f"{field_name}={value} is below minimum {min_val}"
                if description:
                    warning += f" ({description})"
                warnings.append(warning)
                all_passed = False
        
        # Check max constraint
        if max_val is not None and value is not None:
            if value > max_val:
                warning = f"{field_name}={value} exceeds maximum {max_val}"
                if description:
                    warning += f" ({description})"
                warnings.append(warning)
                all_passed = False
        
        # Check for suspicious zero values (common error indicator)
        if constraint.get("non_zero", False) and value == 0:
            warning = f"{field_name} is 0, which may indicate incorrect column selection"
            warnings.append(warning)
            all_passed = False
    
    # Store validation result
    state.output_validation_result = {
        "passed": all_passed,
        "warnings": warnings,
        "constraints_checked": list(constraints.keys())
    }
    state.validation_warnings = warnings
    
    if all_passed:
        print(f"  ✅ Output validation passed")
    else:
        print(f"  ⚠ Output validation warnings:")
        for warning in warnings:
            print(f"     - {warning}")
        
        # If all scores are 0, add to error for potential retry
        output_values = [v for k, v in output.items() if isinstance(v, (int, float))]
        if output_values and all(v == 0 for v in output_values):
            state.execution_result["validation_warning"] = (
                "All output values are 0. This may indicate incorrect column selection. "
                "Please verify that the correct columns are being used for computation."
            )
    
    return state


def _infer_output_constraints(task_description: str, output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Infer output constraints from task description and output structure.
    
    This provides automatic constraint inference for common task types.
    """
    constraints = {}
    task_lower = task_description.lower()
    
    # F1/scoring tasks
    if "f1" in task_lower or "score" in task_lower or "metric" in task_lower:
        if "f1_score" in output or "f1" in output:
            constraints["f1_score"] = {"min": 0.0, "max": 1.0, "non_zero": True, 
                                       "description": "F1 score should be between 0 and 1"}
        if "precision" in output:
            constraints["precision"] = {"min": 0.0, "max": 1.0}
        if "recall" in output:
            constraints["recall"] = {"min": 0.0, "max": 1.0}
        if "accuracy" in output:
            constraints["accuracy"] = {"min": 0.0, "max": 1.0, "non_zero": True}
    
    # Count/quantity tasks
    if "count" in task_lower or "number" in task_lower or "total" in task_lower:
        for key in output:
            if "count" in key.lower() or "number" in key.lower() or "total" in key.lower():
                constraints[key] = {"min": 0}
    
    # Percentage tasks
    if "percentage" in task_lower or "ratio" in task_lower or "rate" in task_lower:
        for key in output:
            if any(term in key.lower() for term in ["percentage", "ratio", "rate", "pct", "%"]):
                constraints[key] = {"min": 0.0, "max": 100.0}
    
    return constraints


def codeact_generate_code_node(state: CodeActState) -> CodeActState:
    """
    CodeAct node: Generate code
    
    Generate different code based on execution mode:
    - mcp_tool: Generate code to call MCP tools
    - codeact: Generate code to complete tasks based on task description
    - fix_code: Fix code errors
    - fix_parameter: Fix parameter errors
    
    Note: This node only generates code, does not execute code. Code execution is handled by codeact_execute_code_node.
    If there's no sandbox environment, generates directly executable code (with necessary error handling and result setting),
    ensuring code can be properly executed in subsequent execution nodes.
    
    Also records code generation trajectory (SE-Agent style).
    """
    import time
    
    # Start recording trajectory
    if not state.current_trajectory:
        state.current_trajectory = _start_trajectory(state)
    
    task = state.task
    mode = state.execution_mode
    parameters = state.parameters
    
    # Record code generation start time
    generation_start_time = time.time()
    
    # Check sandbox environment (for generating appropriate code, but not executing in this node)
    sandbox_provider = _get_sandbox_provider()
    has_sandbox, sandbox_dir = _check_sandbox_available(state, provider=sandbox_provider)
    if not has_sandbox:
        print(f"  ⚠ Sandbox environment unavailable, will generate code executable in fallback mode (directory: {sandbox_dir})")
    
    enforce_tool_call = mode == CodeActExecutionMode.MCP_TOOL

    if mode == CodeActExecutionMode.MCP_TOOL:
        # Generate code to call MCP tools
        tools = state.tools
        if tools:
            # Use first tool
            tool = tools[0] if isinstance(tools, list) else tools
            tool_name = tool.get("tool_name") or tool.get("name", "unknown_tool")
            tool_description = tool.get("description", "")
            
            # Check if there's a revision plan with data transformation strategy
            # This happens after a failure when Revision mechanism suggests a data transformation
            if state.revision_plan and state.revision_plan.strategy == RevisionStrategy.DATA_TRANSFORM:
                print(f"  🔄 Using Revision plan for MCP tool with data transformation")
                print(f"     Strategy: {state.revision_plan.strategy.value}")
                print(f"     Suggested tool: {state.revision_plan.suggested_tool.get('tool_name', 'N/A') if state.revision_plan.suggested_tool else 'N/A'}")
                
                # Use RevisionExecutor to generate code with data transformation
                generated_code = execute_revision_plan(
                    revision_plan=state.revision_plan,
                    original_code=state.previous_code or "",
                    original_error=state.previous_error or "",
                    task_description=state.task_description,
                    parameters=parameters
                )
            else:
                # ============================================================
                # CRITICAL: MCP_TOOL mode uses DIRECT TEMPLATE, not LLM!
                # This ensures correct imports and prevents simulation code.
                # ============================================================
                print(f"  🔧 MCP_TOOL mode: Using direct template (no LLM) for tool: {tool_name}")
                
                generated_code = _generate_mcp_tool_code_directly(
                    tool_name=tool_name,
                    parameters=parameters,
                    task_description=state.task_description
                )
                
                print(f"  ✓ Generated code using direct template ({len(generated_code)} chars)")
            
            # Ensure code is executable (especially when no sandbox environment)
            state.generated_code = _ensure_code_executable(
                generated_code,
                has_sandbox=has_sandbox,
                sandbox_dir=sandbox_dir
            )
        else:
            state.generated_code = "# No matching tool found"
    
    elif mode == CodeActExecutionMode.FIX_CODE:
        # Fix code errors: generate fixed code based on previous error information
        state = _generate_fix_code_for_state(state, has_sandbox, sandbox_dir)
    
    elif mode == CodeActExecutionMode.FIX_PARAMETER:
        # Fix parameter errors: adjust parameters based on previous error information
        previous_code = state.previous_code or ""
        previous_error = state.previous_error or ""
        error_category = state.error_category
        
        if not previous_code or not previous_error:
            state.generated_code = "# Missing necessary fix information (original code or error information)"
            return state
        
        # Use LLM to generate fix code
        user_prompt = get_fix_parameter_user_prompt(
            previous_code=previous_code,
            previous_error=previous_error,
            error_category=error_category,
            parameters=parameters
        )
        
        # Fallback code
        fallback_code = f"""
# Fix parameter errors
# Previous error: {previous_error}
# Original code:
{previous_code}

# Fallback fix: use provided parameters
parameters = {repr(parameters)}
try:
    # Try to execute with corrected parameters
    result = {{"status": "success", "output": "Parameter fix successful (placeholder)"}}
except Exception as e:
    result = {{"status": "failed", "error": str(e)}}
"""
        
        generated_code = _generate_code_with_llm(
            system_prompt=FIX_PARAMETER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_code=fallback_code
        )
        
        # Ensure code is executable (especially when no sandbox environment)
        state.generated_code = _ensure_code_executable(
            generated_code,
            has_sandbox=has_sandbox,
            sandbox_dir=sandbox_dir
        )
    
    else:
        # codeact mode: generate code based on task description
        task_desc = state.task_description
        inputs = state.inputs
        parameters = state.parameters
        
        # Extract output_constraints from task parameters (if provided)
        output_constraints = parameters.get("output_constraints") if parameters else None
        
        # Extract column_hints from task parameters (if provided)
        # This enables semantic column matching instead of fuzzy keyword matching
        column_hints = parameters.get("column_hints") if parameters else None
        
        # Use LLM to generate code
        user_prompt = get_codeact_user_prompt(
            task_description=task_desc,
            inputs=inputs,
            outputs=None,
            output_constraints=output_constraints,
            column_hints=column_hints
        )
        
        # Fallback code
        fallback_code = f"""
# Generate code based on task description
# Task: {task_desc}
print("Executing task: {task_desc}")
result = {{"status": "success", "output": "Task execution result (placeholder)"}}
"""
        
        generated_code = _generate_code_with_llm(
            system_prompt=CODEACT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_code=fallback_code
        )
        
        # Ensure code is executable (especially when no sandbox environment)
        state.generated_code = _ensure_code_executable(
            generated_code,
            has_sandbox=has_sandbox,
            sandbox_dir=sandbox_dir
        )
    
    # Validate generated code
    validation = validate_generated_code(
        state.generated_code or "",
        state.execution_mode,
        enforce_tool_call=enforce_tool_call
    )
    if not validation.get("valid", False):
        print(f"  ⚠ Code validation failed: {validation.get('error')}")
        state.previous_code = state.generated_code
        state.previous_error = validation.get("error")
        state.error_category = validation.get("error_category")

        if state.execution_mode != CodeActExecutionMode.FIX_CODE:
            state.execution_mode = CodeActExecutionMode.FIX_CODE
            state.revision_plan = None
            state = _generate_fix_code_for_state(state, has_sandbox, sandbox_dir)

            second_validation = validate_generated_code(
                state.generated_code or "",
                state.execution_mode,
                enforce_tool_call=enforce_tool_call
            )
            if not second_validation.get("valid", False):
                state.generated_code = _ensure_code_executable(
                    _build_validation_error_code(second_validation.get("error", "Code validation failed")),
                    has_sandbox=has_sandbox,
                    sandbox_dir=sandbox_dir
                )
        else:
            state.generated_code = _ensure_code_executable(
                _build_validation_error_code(validation.get("error", "Code validation failed")),
                has_sandbox=has_sandbox,
                sandbox_dir=sandbox_dir
            )

    # Record code generation trajectory
    generation_time = time.time() - generation_start_time
    if state.current_trajectory:
        _update_trajectory_code(
            state.current_trajectory,
            state.generated_code,
            generation_time
        )
        state.current_trajectory.sandbox_used = has_sandbox
    
    return state


def codeact_execute_code_node(state: CodeActState) -> CodeActState:
    """
    CodeAct node: Execute code
    
    Execute generated code in sandbox environment. If no sandbox environment, use fallback to execute directly.
    
    Also records execution trajectory (SE-Agent style).
    """
    import time
    import os
    
    code = state.generated_code
    if not code:
        state.execution_result = {
            "status": "failed",
            "error": "No code generated",
            "error_type": "NoCodeError"
        }
        print(f"  ✗ Execution failed: No code generated")
        
        # Record failed trajectory
        if state.current_trajectory:
            _finalize_trajectory(state.current_trajectory, state.execution_result, 0.0)
            _save_trajectory_to_pool(state, state.current_trajectory)
            state.current_trajectory = None
        
        return state
    
    # Record execution start time
    execution_start_time = time.time()
    
    # Check sandbox environment
    sandbox_provider = _get_sandbox_provider()
    print(f"  ℹ Sandbox provider from config: {sandbox_provider}")
    print(f"  ℹ Execution mode: {state.execution_mode}")
    
    # Note: MCP_TOOL mode can now use OpenSandbox
    # OpenSandbox executor will handle dependency installation and config file setup
    if state.execution_mode == CodeActExecutionMode.MCP_TOOL or str(state.execution_mode) == "mcp_tool":
        if sandbox_provider == "opensandbox":
            print("  ℹ MCP tool execution mode with OpenSandbox: will install dependencies and setup config in sandbox")

    has_sandbox, sandbox_dir = _check_sandbox_available(state, provider=sandbox_provider)
    allow_unsafe = os.getenv("ALLOW_UNSAFE_CODEACT", "false").lower() == "true"
    if not has_sandbox and not allow_unsafe:
        state.execution_result = {
            "status": "failed",
            "error": "Sandbox unavailable and unsafe execution is disabled (set ALLOW_UNSAFE_CODEACT=true to override)"
        }
        if state.current_trajectory:
            _finalize_trajectory(state.current_trajectory, state.execution_result, 0.0)
            _save_trajectory_to_pool(state, state.current_trajectory)
            state.current_trajectory = None
        return state
    print(f"  ℹ Sandbox environment check: available={has_sandbox}, directory={sandbox_dir}")
    print(f"  ℹ Final sandbox provider: {sandbox_provider}")
    
    try:
        if sandbox_provider == "opensandbox":
            print("  🚀 Using OpenSandbox for code execution")
            from utils.opensandbox_executor import run_code_in_opensandbox_sync, is_opensandbox_enabled

            if not is_opensandbox_enabled():
                state.execution_result = {
                    "status": "failed",
                    "error": "OpenSandbox is not enabled (set CODEACT_SANDBOX_PROVIDER=opensandbox or OPENSANDBOX_ENABLED=true)",
                    "error_type": "SandboxNotEnabled",
                }
                return state

            timeout_seconds = int(os.getenv("CODEACT_TIMEOUT_SECONDS", "180"))  # Default: 3 minutes
            
            # 每次创建新沙盒，不再复用（避免连接失败问题）
            # 文件通过 session_id 组织在同一目录下
            
            sandbox_result = run_code_in_opensandbox_sync(
                code=code,
                task_id=state.task.task_id,
                timeout_seconds=timeout_seconds,
                existing_sandbox_id=None,  # 不复用，每次创建新沙盒
                keep_alive=False,  # 不保持沙盒存活
            )
            
            if sandbox_result.get("error"):
                state.execution_result = {
                    "status": "failed",
                    "error": sandbox_result.get("error"),
                    "error_type": sandbox_result.get("error_type"),
                    "sandbox_provider": "opensandbox",
                    "sandbox_id": sandbox_result.get("sandbox_id"),
                }
                return state

            stdout = sandbox_result.get("stdout", "")
            stderr = sandbox_result.get("stderr", "")
            returncode = sandbox_result.get("returncode")
            parsed = _parse_execution_output(stdout, stderr, returncode)
            state.execution_result = {
                **parsed,
                "stderr": stderr[-1000:] if stderr else "",
                "returncode": returncode,
                "sandbox_dir": f"opensandbox://{sandbox_result.get('sandbox_id')}",
                "sandbox_provider": "opensandbox",
                "sandbox_image": sandbox_result.get("image"),
            }
            print(f"  ✓ OpenSandbox execution completed, status={parsed.get('status')}, returncode={returncode}")
        elif has_sandbox:
            # Execute code in sandbox via subprocess for isolation
            import os
            from pathlib import Path
            
            timeout_seconds = int(os.getenv("CODEACT_TIMEOUT_SECONDS", "180"))  # Default: 3 minutes
            print(f"  🔄 Starting code execution (local sandbox subprocess mode, timeout={timeout_seconds}s)...")
            print(f"  ℹ Code length: {len(code)} characters")
            
            sandbox_path = Path(sandbox_dir)
            sandbox_path.mkdir(parents=True, exist_ok=True)
            code_file = sandbox_path / f"codeact_{state.task.task_id}.py"
            code_file.write_text(code, encoding="utf-8")
            
            venv_python = _find_and_activate_venv(agent_dir)
            python_exe = str(venv_python) if venv_python else sys.executable
            
            env = os.environ.copy()
            agent_dir_str = str(agent_dir)
            existing_pythonpath = env.get("PYTHONPATH", "")
            if agent_dir_str not in existing_pythonpath:
                sep = ";" if os.name == "nt" else ":"
                env["PYTHONPATH"] = agent_dir_str + (sep + existing_pythonpath if existing_pythonpath else "")
            # Force UTF-8 to avoid Windows GBK encoding errors in subprocess output
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            
            try:
                completed = subprocess.run(
                    [python_exe, str(code_file)],
                    cwd=str(sandbox_path),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    timeout=timeout_seconds,
                    **_build_subprocess_kwargs()
                )
            except subprocess.TimeoutExpired as e:
                error_msg = f"Execution timed out after {timeout_seconds}s"
                print(f"  ✗ {error_msg}")
                print(f"  ⚠ This may indicate:")
                print(f"     - Code is waiting for network/IO operations")
                print(f"     - Infinite loop or deadlock in code")
                print(f"     - MCP tool call is hanging")
                print(f"  💡 Consider:")
                print(f"     - Increasing CODEACT_TIMEOUT_SECONDS (current: {timeout_seconds}s)")
                print(f"     - Using OpenSandbox for better isolation (if not in MCP_TOOL mode)")
                print(f"     - Reviewing the generated code for potential issues")
                
                state.execution_result = {
                    "status": "failed",
                    "error": error_msg,
                    "error_type": "TimeoutError",
                    "stderr": (e.stderr or "")[-1000:] if hasattr(e, "stderr") else "",
                    "returncode": None,
                    "sandbox_dir": sandbox_dir,
                    "timeout_seconds": timeout_seconds
                }
                return state
            
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            parsed = _parse_execution_output(stdout, stderr, completed.returncode)
            state.execution_result = {
                **parsed,
                "stderr": stderr[-1000:] if stderr else "",
                "returncode": completed.returncode,
                "sandbox_dir": sandbox_dir
            }
            print(f"  ✓ Execution completed, status={parsed.get('status')}, returncode={completed.returncode}")
        else:
            # No sandbox environment: fallback, execute directly
            print(f"  ⚠ Using fallback execution (no sandbox environment)")
            print(f"  🔄 Starting code execution (fallback mode)...")
            
            # Important: Activate specified virtual environment before executing code
            import sys
            import site
            
            # 1. Find and activate virtual environment
            venv_python = _find_and_activate_venv(agent_dir)
            if venv_python:
                _activate_venv_in_sys_path(venv_python)
            else:
                print(f"  ⚠ Virtual environment not found, will use current Python environment")
            
            # 2. Add agent directory to path (if not already)
            agent_dir_str = str(agent_dir)
            if agent_dir_str not in sys.path:
                sys.path.insert(0, agent_dir_str)
            
            # Ensure UTF-8 output in fallback mode (avoid Windows GBK issues)
            os.environ.setdefault("PYTHONIOENCODING", "utf-8")
            os.environ.setdefault("PYTHONUTF8", "1")
            try:
                if hasattr(sys.stdout, "reconfigure"):
                    sys.stdout.reconfigure(encoding="utf-8")
                if hasattr(sys.stderr, "reconfigure"):
                    sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass
            
            # 3. Ensure all necessary paths are in sys.path
            try:
                # If virtual environment found, ensure its site-packages is in path
                if venv_python:
                    venv_dir = venv_python.parent.parent
                    if os.name == 'nt':
                        venv_site_packages = venv_dir / "Lib" / "site-packages"
                    else:
                        import sysconfig
                        venv_site_packages = Path(sysconfig.get_path('purelib', vars={'base': str(venv_dir)}))
                    
                    if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
                        sys.path.insert(0, str(venv_site_packages))
                
                # Also add system site-packages (as backup)
                system_site_packages = site.getsitepackages()
                for sp in system_site_packages:
                    if sp not in sys.path:
                        sys.path.append(sp)  # Add to end, prioritize virtual environment packages
            except Exception as e:
                print(f"  ⚠ Error configuring site-packages: {e}")
            
            # 4. Verify virtual environment is properly activated
            try:
                import langchain_mcp_adapters
                print(f"  ✓ Virtual environment activated, can import langchain-mcp-adapters")
            except ImportError:
                print(f"  ⚠ Warning: Cannot import langchain-mcp-adapters, virtual environment may not be properly activated")
            
            # Pre-import tool interfaces so generated code can use them
            try:
                from utils.mcp_helper import invoke_mcp_tool_sync
                from core.tool_interface import call_tool
                local_namespace = {
                    "invoke_mcp_tool_sync": invoke_mcp_tool_sync,
                    "call_tool": call_tool,
                    "__builtins__": __builtins__
                }
            except ImportError as e:
                import traceback
                print(f"  ⚠ Cannot import mcp_helper: {e}")
                print(f"     Error details: {traceback.format_exc()}")
                print(f"     Python path: {sys.path[:5]}...")
                local_namespace = {
                    "__builtins__": __builtins__
                }
            
            exec(code, {"__builtins__": __builtins__}, local_namespace)
            print(f"  ✓ Code execution completed")
            
            # Extract execution result
            result = local_namespace.get("result", "Execution successful, no return result")
            output = str(local_namespace.get("output", result))
            
            state.execution_result = {
                "status": "success",
                "output": output,
                "result": result,
                "sandbox_used": False
            }
            print(f"  ✓ Execution successful, result: {output[:100]}...")
    
    except SyntaxError as e:
        # Syntax error
        error_msg = f"Syntax error: {str(e)}"
        error_details = {
            "status": "failed",
            "error": error_msg,
            "error_type": "SyntaxError",
            "error_line": getattr(e, 'lineno', None),
            "error_text": getattr(e, 'text', None),
            "sandbox_used": has_sandbox,
            "code_preview": code[:500] if code else None
        }
        state.execution_result = error_details
        print(f"  ✗ Execution failed (syntax error): {error_msg}")
        print(f"     Error line number: {error_details.get('error_line')}")
        print(f"     Error code: {error_details.get('error_text')}")
        print(f"     Code preview: {code[:200]}...")
    
    except NameError as e:
        # Name error (undefined variable or function)
        error_msg = f"Name error: {str(e)}"
        error_details = {
            "status": "failed",
            "error": error_msg,
            "error_type": "NameError",
            "sandbox_used": has_sandbox,
            "code_preview": code[:500] if code else None
        }
        state.execution_result = error_details
        print(f"  ✗ Execution failed (name error): {error_msg}")
        print(f"     Code preview: {code[:200]}...")
    
    except ImportError as e:
        # Import error
        error_msg = f"Import error: {str(e)}"
        error_details = {
            "status": "failed",
            "error": error_msg,
            "error_type": "ImportError",
            "sandbox_used": has_sandbox,
            "code_preview": code[:500] if code else None
        }
        state.execution_result = error_details
        print(f"  ✗ Execution failed (import error): {error_msg}")
        print(f"     Code preview: {code[:200]}...")
    
    except Exception as e:
        # Other errors
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        error_details = {
            "status": "failed",
            "error": error_msg,
            "error_type": type(e).__name__,
            "error_traceback": error_traceback,
            "sandbox_used": has_sandbox,
            "code_preview": code[:500] if code else None
        }
        state.execution_result = error_details
        print(f"  ✗ Execution failed ({type(e).__name__}): {error_msg}")
        print(f"     Error stack:")
        print(f"     {error_traceback}")
        print(f"     Code preview: {code[:200]}...")
    
    # Record execution trajectory
    execution_time = time.time() - execution_start_time
    if state.current_trajectory:
        _finalize_trajectory(state.current_trajectory, state.execution_result, execution_time)
        _save_trajectory_to_pool(state, state.current_trajectory)
        state.current_trajectory = None
    
    return state


# ===================== Build CodeAct Subgraph =====================

def codeact_revision_node(state: CodeActState) -> CodeActState:
    """
    CodeAct Revision node: Analyze failures and generate Revision plan
    
    When code execution fails, use Revision mechanism for deep analysis and intelligent fix.
    """
    # Always increment revision_iteration to prevent infinite loops
    state.revision_iteration += 1
    print(f"  ℹ Revision iteration incremented to {state.revision_iteration}")
    
    # Check if there are failed trajectories
    if not state.trajectory_history:
        print("  ⚠ No trajectory history, cannot perform Revision analysis")
        print("  ⚠ Will use basic fix mode instead")
        # Set basic fix information from execution_result if available
        if state.execution_result:
            state.previous_code = state.generated_code
            state.previous_error = state.execution_result.get("error", "Unknown error")
            state.error_category = state.execution_result.get("error_category", "unknown")
        return state
    
    # Get latest failed trajectory
    failed_trajectories = [t for t in state.trajectory_history if t.status == TrajectoryStatus.FAILED]
    if not failed_trajectories:
        print("  ℹ No failed trajectories, no Revision needed")
        # Still update fix information from execution_result if available
        if state.execution_result:
            state.previous_code = state.generated_code
            state.previous_error = state.execution_result.get("error", "Unknown error")
            state.error_category = state.execution_result.get("error_category", "unknown")
        return state
    
    latest_failed = failed_trajectories[-1]
    previous_failed = failed_trajectories[:-1] if len(failed_trajectories) > 1 else []
    
    print(f"  🔍 Starting Revision analysis (failed trajectory: {latest_failed.trajectory_id})")
    
    # Create Revision plan
    revision_plan = create_revision_plan(latest_failed, previous_failed)
    
    # Update state (revision_iteration already incremented at the start)
    state.revision_plan = revision_plan
    state.previous_code = latest_failed.generated_code
    state.previous_error = latest_failed.error_message or str(latest_failed.error_type)
    state.error_category = latest_failed.error_category
    
    # Store suggested tool if available
    if revision_plan.suggested_tool:
        state.suggested_transform_tool = revision_plan.suggested_tool
    
    print(f"  ✓ Revision plan generated successfully")
    print(f"     Strategy: {revision_plan.strategy.value}")
    print(f"     Root cause: {revision_plan.root_cause[:150]}...")
    print(f"     Confidence: {revision_plan.confidence:.2f}")
    print(f"     Iteration count: {state.revision_iteration}")
    
    # Show suggested transformation tool if available
    if revision_plan.suggested_tool:
        tool_info = revision_plan.suggested_tool
        print(f"  💡 SUGGESTED DATA TRANSFORMATION TOOL:")
        print(f"     Tool: {tool_info.get('tool_name', 'unknown')}")
        print(f"     Service: {tool_info.get('service', 'unknown')}")
        print(f"     Description: {tool_info.get('description', 'No description')[:100]}...")
    
    return state


# ===================== Todo List Management Nodes (New Architecture) =====================

def read_todo_node(state: CodeActState) -> CodeActState:
    """
    Read todo-list.md from sandbox directory
    
    This node reads the task list and caches it in the state.
    If no todo-list.md exists, it creates an empty one.
    """
    import json
    
    print("=" * 60)
    print("📖 [Todo] Reading todo-list.md from sandbox...")
    print("=" * 60)
    
    # Determine sandbox directory
    # IMPORTANT: Use string operations to preserve Unix-style paths for remote sandbox
    sandbox_dir = None
    if state.todo_list_path:
        # Extract directory from todo_list_path using string operations
        # to avoid Windows path conversion
        todo_path = state.todo_list_path
        if '/' in todo_path:
            sandbox_dir = todo_path.rsplit('/', 1)[0]  # Unix-style
        else:
            sandbox_dir = str(Path(todo_path).parent)  # Fallback for local paths
    elif state.parent_state and hasattr(state.parent_state, 'session_id'):
        session_id = state.parent_state.session_id
        sandbox_dir = f"/data/sessions/{session_id}"
    elif state.parameters.get("sandbox_dir"):
        sandbox_dir = state.parameters.get("sandbox_dir")
    else:
        # Use default sandbox directory
        import tempfile
        sandbox_dir = tempfile.gettempdir()
    
    print(f"  📁 Sandbox directory: {sandbox_dir}")
    
    # Get opensandbox_id for remote operations
    opensandbox_id = None
    if state.parent_state:
        opensandbox_id = getattr(state.parent_state, 'opensandbox_id', None)
        if not opensandbox_id:
            opensandbox_id = state.parent_state.merged_result.get('opensandbox_id')
    
    if opensandbox_id:
        print(f"  🔗 OpenSandbox ID: {opensandbox_id}")
    
    # Initialize todo manager with opensandbox_id for remote operations
    state.todo_manager = TodoListManager(sandbox_dir, opensandbox_id=opensandbox_id)
    
    # Check if todo-list.md exists
    if state.todo_manager.todo_list_exists():
        try:
            state.todo_list = state.todo_manager.read_todo_list()
            print(f"  ✓ Todo list loaded: {len(state.todo_list.tasks)} tasks")
            
            # Show progress summary
            summary = state.todo_manager.get_progress_summary(state.todo_list)
            print(f"  📊 Progress: {summary}")
        except Exception as e:
            print(f"  ⚠ Failed to parse todo list: {e}")
            state.todo_list = None
    else:
        print(f"  ℹ No todo-list.md found, will check for task from state")
        state.todo_list = None
    
    # Store todo list path - use string version to preserve Unix-style paths
    state.todo_list_path = state.todo_manager.todo_list_path_str
    
    return state


def select_next_task_node(state: CodeActState) -> CodeActState:
    """
    Select the next pending task from the todo list
    
    Selection logic:
    1. If todo list exists, select next pending task
    2. If no todo list, use the task from state (backward compatibility)
    3. Update state.current_todo_task
    """
    print("=" * 60)
    print("🎯 [Todo] Selecting next task...")
    print("=" * 60)
    
    # Case 1: Todo list exists
    if state.todo_list and state.todo_manager:
        next_task = state.todo_manager.get_next_pending_task(state.todo_list)
        
        if next_task:
            state.current_todo_task = next_task
            print(f"  ✓ Selected task: {next_task.id}")
            print(f"     Type: {next_task.type}")
            print(f"     Priority: {next_task.priority}")
            print(f"     Description: {next_task.description[:100]}...")
            
            # Update task status to IN_PROGRESS
            state.todo_manager.update_task_status(
                next_task.id,
                TodoTaskStatus.IN_PROGRESS
            )
            
            # Map TodoTask to CodeActState fields
            state.task_description = next_task.description
            
            # Merge parameters carefully:
            # - Environment parameters (sandbox_dir, todo_list_path, session_id) should be preserved in state
            # - But should NOT override actual tool parameters from todo-list
            # - Tool parameters from todo-list should take precedence
            merged_params = {}
            
            # Environment parameters to preserve (for sandbox operations)
            env_params = {'sandbox_dir', 'todo_list_path', 'session_id', 'opensandbox_id'}
            
            # First, preserve environment parameters from current state
            for key in env_params:
                if key in state.parameters:
                    merged_params[key] = state.parameters[key]
            
            # Then, add actual task parameters (these are the real tool parameters)
            # Note: todo-list may contain parameter descriptions, not actual values
            # The filtering will happen in _generate_mcp_tool_code_directly
            for key, value in next_task.parameters.items():
                merged_params[key] = value
            
            state.parameters = merged_params
            
            # Debug: Log parameter merge
            print(f"     Parameters: {list(merged_params.keys())}")
            if env_params.intersection(merged_params.keys()):
                print(f"     (包含环境参数: {env_params.intersection(merged_params.keys())})")
            
            # Set execution mode based on task type
            if next_task.type == TodoTaskType.MCP_TOOL:
                state.execution_mode = CodeActExecutionMode.MCP_TOOL
                
                # CRITICAL: Extract tool_name from parameters and set state.tools
                # This enables _generate_mcp_tool_code_directly to generate correct code
                tool_name = next_task.parameters.get("tool_name")
                if tool_name:
                    state.tools = [{"tool_name": tool_name, "name": tool_name}]
                    print(f"     Tool: {tool_name}")
                else:
                    print(f"  ⚠ MCP_TOOL task has no tool_name in parameters!")
                    state.tools = []
            elif next_task.type == TodoTaskType.FILE_CONVERT:
                state.execution_mode = CodeActExecutionMode.CODEACT
            else:
                state.execution_mode = CodeActExecutionMode.CODEACT
            
            return state
        else:
            print("  ℹ No pending tasks found in todo list")
    
    # Case 2: No todo list or no pending tasks - use existing task (backward compatibility)
    if state.task and state.task.content:
        print(f"  ℹ Using task from state (backward compatibility)")
        print(f"     Task ID: {state.task.task_id}")
        print(f"     Description: {state.task.content[:100]}...")
        
        # Create a TodoTask from the existing task
        state.current_todo_task = TodoTask(
            id=state.task.task_id,
            type=TodoTaskType.GENERAL,
            status=TodoTaskStatus.IN_PROGRESS,
            priority=5,
            description=state.task.content,
            parameters=state.parameters
        )
        
        return state
    
    # Case 3: No task at all
    print("  ⚠ No task found - nothing to execute")
    state.current_todo_task = None
    
    return state


# ===================== P2: Helper Functions =====================

def _build_file_parameter_table_from_parent_state(
    parent_state: Any,
    session_id: str = "unknown"
) -> FileParameterTable:
    """
    Build FileParameterTable from parent_state.extracted_parameters
    
    Args:
        parent_state: GlobalState containing extracted_parameters
        session_id: Session ID for the table
        
    Returns:
        FileParameterTable with files from extracted_parameters
    """
    table = FileParameterTable(session_id=session_id)
    
    if not parent_state:
        return table
    
    # Get extracted_parameters from parent_state
    extracted_params = None
    if hasattr(parent_state, 'extracted_parameters'):
        extracted_params = parent_state.extracted_parameters
    elif isinstance(parent_state, dict):
        extracted_params = parent_state.get('extracted_parameters', {})
    
    if not extracted_params:
        return table
    
    # Also try merged_result as fallback
    if not extracted_params and hasattr(parent_state, 'merged_result'):
        merged = parent_state.merged_result or {}
        extracted_params = merged.get('extracted_parameters', {})
    
    # Extract files from extracted_parameters
    files_dict = extracted_params.get('files', {})
    
    for key, file_info in files_dict.items():
        # Handle different file_info formats
        if isinstance(file_info, dict):
            path = file_info.get('sandbox_path') or file_info.get('path', '')
            if not path:
                continue
            
            file_type = file_info.get('format') or file_info.get('file_type', '')
            if not file_type and path:
                file_type = path.split('.')[-1] if '.' in path else 'unknown'
            
            # Build description from available info
            desc_parts = []
            if file_info.get('data_type'):
                desc_parts.append(f"Data type: {file_info['data_type']}")
            if file_info.get('row_count'):
                desc_parts.append(f"Rows: {file_info['row_count']}")
            if file_info.get('columns'):
                desc_parts.append(f"Columns: {len(file_info['columns'])}")
            
            description = file_info.get('description') or '; '.join(desc_parts) or f"User file: {key}"
            
            # Create FileParameter
            fp = create_file_param_from_user_input(
                key=key,
                path=path,
                description=description,
                file_type=file_type,
                metadata={
                    'columns': file_info.get('columns', []),
                    'row_count': file_info.get('row_count'),
                    'data_type': file_info.get('data_type'),
                    'can_be_used_as': file_info.get('can_be_used_as', [])
                }
            )
            table.add_file(fp)
    
    print(f"  📁 Built FileParameterTable from parent_state: {len(table.files)} files")
    return table


# ===================== P2: Parameter Inference Node =====================

def infer_parameters_node(state: CodeActState) -> CodeActState:
    """
    Infer parameters for the current task using LLM
    
    This node:
    1. Gets the file parameter table (priority: parent_state > todo_manager)
    2. Uses LLM to infer parameter values based on:
       - User's original input
       - File parameter table (available files with descriptions)
       - Completed tasks summary
       - Current tool's parameter requirements
    3. Updates state.parameters with inferred values
    """
    print("=" * 60)
    print("🔮 [P2] Inferring parameters...")
    print("=" * 60)
    
    # Skip if no current task
    if not state.current_todo_task:
        print("  ⚠ No current task, skipping parameter inference")
        return state
    
    # PRIORITY 1: Build file parameter table from parent_state.extracted_parameters
    # This is the primary source - parameters are stored in GlobalState, not in sandbox files
    if state.parent_state:
        session_id = "unknown"
        if hasattr(state.parent_state, 'session_id'):
            session_id = state.parent_state.session_id or "unknown"
        state.file_parameter_table = _build_file_parameter_table_from_parent_state(
            state.parent_state, 
            session_id=session_id
        )
        print(f"  📁 File parameter table from parent_state: {len(state.file_parameter_table.files)} files")
    
    # PRIORITY 2: Fall back to todo_manager (for task outputs that were saved to file)
    if (not state.file_parameter_table or len(state.file_parameter_table.files) == 0) and state.todo_manager:
        state.file_parameter_table = state.todo_manager.get_file_parameter_table()
        print(f"  📁 File parameter table from todo_manager: {len(state.file_parameter_table.files)} files")
    
    # Determine tool name and requirements
    tool_name = state.current_todo_task.parameters.get("tool_name", "")
    
    # If this is not an MCP tool task, skip inference
    if state.current_todo_task.type != TodoTaskType.MCP_TOOL:
        print(f"  ℹ Task type is {state.current_todo_task.type}, using task parameters directly")
        state.parameters = {**state.parameters, **state.current_todo_task.parameters}
        return state
    
    # If no tool specified, skip inference
    if not tool_name:
        print("  ⚠ No tool_name specified, using task parameters directly")
        state.parameters = {**state.parameters, **state.current_todo_task.parameters}
        return state
    
    # Get tool parameter definitions
    tool_params = _get_tool_parameter_definitions(tool_name, state.all_available_tools)
    
    if not tool_params:
        print(f"  ⚠ No parameter definitions found for tool: {tool_name}")
        print(f"  ℹ Will try to use extracted parameters from parameter table")
        # Don't fallback to schema descriptions! 
        # Instead, try to get actual values from extracted_params
        # This will be handled below (extracted_params extraction happens next)
        # Just skip the LLM inference step and use what we have
        tool_params = []  # Empty list, but continue to extract params from parameter table
    
    if tool_params:
        print(f"  🔧 Tool: {tool_name}")
        print(f"  📋 Parameters to infer: {[p.get('name') for p in tool_params]}")
    
    # PRIORITY 3: Try to get actual parameter values from parent_state.extracted_parameters
    # These are the actual values extracted by supervisor, not schema definitions
    extracted_params = {}
    if state.parent_state:
        if hasattr(state.parent_state, 'extracted_parameters'):
            ext = state.parent_state.extracted_parameters or {}
            # Get params section which contains actual values (not schema)
            extracted_params = ext.get('params', {})
            # Also get files section for file paths
            files = ext.get('files', {})
            for key, file_info in files.items():
                if isinstance(file_info, dict) and 'sandbox_path' in file_info:
                    extracted_params[key] = file_info['sandbox_path']
        elif isinstance(state.parent_state, dict):
            ext = state.parent_state.get('extracted_parameters', {})
            extracted_params = ext.get('params', {})
            files = ext.get('files', {})
            for key, file_info in files.items():
                if isinstance(file_info, dict) and 'sandbox_path' in file_info:
                    extracted_params[key] = file_info['sandbox_path']
    
    # Also check merged_result for extracted_parameters
    if not extracted_params and state.parent_state:
        merged = None
        if hasattr(state.parent_state, 'merged_result'):
            merged = state.parent_state.merged_result
        elif isinstance(state.parent_state, dict):
            merged = state.parent_state.get('merged_result', {})
        
        if merged:
            ext = merged.get('extracted_parameters', {})
            extracted_params = ext.get('params', {})
            files = ext.get('files', {})
            for key, file_info in files.items():
                if isinstance(file_info, dict) and 'sandbox_path' in file_info:
                    extracted_params[key] = file_info['sandbox_path']
    
    if extracted_params:
        print(f"  📦 Extracted parameters from parent_state: {list(extracted_params.keys())}")
    
    # PRIORITY 4: Extract semantic hints from TodoTask.parameters
    # TodoTask.parameters contains valuable schema hints that help inference:
    # - Dict values with 'description'/'type'/'required': schema definitions (guidance)
    # - Non-dict values (strings, numbers): semantic hints (direct values)
    semantic_hints = {}
    schema_hints = {}  # For logging/debugging
    
    task_params = state.current_todo_task.parameters or {}
    for key, value in task_params.items():
        if key == 'tool_name':
            continue  # Skip meta field
        if isinstance(value, dict):
            # This is a schema definition - contains valuable guidance
            schema_hints[key] = value
            # Check if there's a default or example value
            if 'default' in value:
                semantic_hints[key] = value['default']
            elif 'example' in value:
                semantic_hints[key] = value['example']
        elif isinstance(value, str) and value and not value.startswith('{'):
            # This is a semantic hint - direct value suggestion
            # Examples: "Peptide support status", "Validation results report"
            semantic_hints[key] = value
        elif isinstance(value, (int, float, bool)):
            # Direct numeric/boolean values
            semantic_hints[key] = value
    
    if schema_hints:
        print(f"  📝 Schema hints available: {list(schema_hints.keys())}")
    if semantic_hints:
        print(f"  💡 Semantic hints: {semantic_hints}")
    
    # If no tool_params, skip LLM inference and use extracted_params directly
    if not tool_params:
        print("  ℹ No tool parameter definitions, using extracted parameters directly")
        if extracted_params:
            print(f"  ✓ Using extracted parameters from parameter table:")
            for key, value in extracted_params.items():
                print(f"    - {key}: {value}")
            # Merge with existing parameters, extracted_params take precedence
            state.parameters = {**state.parameters, **extracted_params}
            # Also apply semantic hints if they contain actual values (not schema descriptions)
            for key, value in semantic_hints.items():
                if not isinstance(value, dict):
                    state.parameters[key] = value
        else:
            print("  ⚠ No extracted parameters available, keeping existing parameters")
        return state
    
    # Use LLM to infer parameters
    try:
        llm = create_code_llm()
        
        if not llm:
            print("  ⚠ LLM not available")
            # Fallback: use extracted_params first, then semantic hints
            if extracted_params:
                print(f"  ✓ Using extracted parameters from parameter table:")
                for key, value in extracted_params.items():
                    print(f"    - {key}: {value}")
                state.parameters = {**state.parameters, **extracted_params}
            else:
                print("  ⚠ No extracted parameters available")
            return state
        
        # Build prompt with semantic hints
        prompt = get_parameter_inference_prompt(
            tool_name=tool_name,
            tool_description=state.current_todo_task.description,
            tool_parameters=tool_params,
            user_input=state.user_input_context or "",
            file_param_table=state.file_parameter_table or FileParameterTable(session_id="empty"),
            completed_tasks=state.completed_tasks_summary,
            semantic_hints=semantic_hints,  # Pass semantic hints for better inference
            extracted_params=extracted_params  # Pass known actual values as context
        )
        
        # Call LLM
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Parse response
        inferred = _parse_parameter_inference_response(response_text)
        
        if inferred:
            print(f"  ✓ Inferred parameters:")
            for key, value in inferred.items():
                print(f"    - {key}: {value}")
            
            # Merge parameters: extracted_params > inferred > existing params
            # Priority: extracted_params (actual values from parameter table) > LLM inferred > existing
            # DO NOT use state.current_todo_task.parameters as base - they are schema descriptions!
            state.inferred_parameters = {**inferred, **extracted_params}  # extracted_params override inferred
            state.parameters = {**state.parameters, **state.inferred_parameters}
        else:
            print("  ⚠ Failed to parse inference response")
            # Still use extracted_params if available
            if extracted_params:
                print(f"  ✓ Using extracted parameters from parameter table:")
                for key, value in extracted_params.items():
                    print(f"    - {key}: {value}")
                state.parameters = {**state.parameters, **extracted_params}
            # else: keep existing parameters, don't pollute with schema descriptions
    
    except Exception as e:
        print(f"  ⚠ Parameter inference failed: {e}")
        # Still use extracted_params if available
        if extracted_params:
            print(f"  ✓ Using extracted parameters from parameter table:")
            for key, value in extracted_params.items():
                print(f"    - {key}: {value}")
            state.parameters = {**state.parameters, **extracted_params}
        # else: keep existing parameters, don't pollute with schema descriptions
    
    return state


def _get_tool_parameter_definitions(tool_name: str, all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get parameter definitions for a tool from the available tools list"""
    for tool in all_tools:
        if tool.get("tool_name") == tool_name or tool.get("name") == tool_name:
            return tool.get("parameters", [])
    return []


def _parse_parameter_inference_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response to extract inferred parameters"""
    import re
    
    # Try to extract JSON from the response
    try:
        # Try direct JSON parse
        result = json.loads(response_text.strip())
        if isinstance(result, dict) and "parameters" in result:
            return result["parameters"]
        return result
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in code blocks
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and "parameters" in result:
                    return result["parameters"]
                elif isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
    
    return None


# ===================== P2: Extract File Parameters Node =====================

def extract_file_params_node(state: CodeActState) -> CodeActState:
    """
    Extract file parameters from task execution result
    
    This node:
    1. Analyzes the execution result for output files
    2. Creates FileParameter entries for each output file
    3. Updates the file parameter table
    4. Saves the updated table
    """
    print("=" * 60)
    print("📤 [P2] Extracting file parameters from result...")
    print("=" * 60)
    
    # Skip if no execution result
    if not state.execution_result:
        print("  ℹ No execution result, skipping file extraction")
        return state
    
    # Skip if execution failed
    if state.execution_result.get("status") != "success":
        print("  ℹ Execution failed, skipping file extraction")
        return state
    
    # Get current task info
    task_id = state.current_todo_task.id if state.current_todo_task else "unknown"
    task_description = state.current_todo_task.description if state.current_todo_task else "Unknown task"
    
    # Extract file parameters from result
    file_params = extract_file_info_from_task_result(
        task_id=task_id,
        task_description=task_description,
        task_result=state.execution_result
    )
    
    if not file_params:
        print("  ℹ No output files detected in result")
        return state
    
    print(f"  📁 Found {len(file_params)} output file(s):")
    for fp in file_params:
        print(f"    - [{fp.key}] {fp.path}")
        print(f"      Description: {fp.description}")
    
    # Update file parameter table
    if state.file_parameter_table:
        for fp in file_params:
            state.file_parameter_table.add_file(fp)
        print(f"  ✓ Updated file parameter table")
    else:
        # Create new table if needed
        session_id = "unknown"
        if state.todo_list and state.todo_list.session:
            session_id = state.todo_list.session.session_id
        state.file_parameter_table = FileParameterTable(session_id=session_id)
        for fp in file_params:
            state.file_parameter_table.add_file(fp)
    
    # Save to TodoListManager
    if state.todo_manager:
        state.todo_manager.add_file_parameters(file_params)
    
    # Add to completed tasks summary for future inference
    completed_summary = {
        "id": task_id,
        "description": task_description,
        "result_summary": str(state.execution_result.get("output", ""))[:200],
        "output_files": [fp.path for fp in file_params]
    }
    state.completed_tasks_summary.append(completed_summary)
    
    return state


def update_todo_node(state: CodeActState) -> CodeActState:
    """
    Update task status in todo-list.md after execution
    
    Updates the current task's status based on execution result:
    - Success -> COMPLETED
    - Failure -> FAILED
    """
    print("=" * 60)
    print("📝 [Todo] Updating task status...")
    print("=" * 60)
    
    if not state.current_todo_task:
        print("  ℹ No current task to update")
        return state
    
    task_id = state.current_todo_task.id
    
    # Determine new status based on execution result
    if state.execution_result:
        if state.execution_result.get("status") == "success":
            new_status = TodoTaskStatus.COMPLETED
            # Ensure result is a dict (TodoTask.result requires Dict type)
            raw_output = state.execution_result.get("output", {})
            if isinstance(raw_output, dict):
                result = raw_output
            else:
                # Wrap string output in a dict
                result = {"output": raw_output}
            error = None
            print(f"  ✓ Task {task_id} completed successfully")
        else:
            new_status = TodoTaskStatus.FAILED
            result = None
            error = state.execution_result.get("error", "Unknown error")
            print(f"  ✗ Task {task_id} failed: {error[:100]}...")
    else:
        new_status = TodoTaskStatus.FAILED
        result = None
        error = "No execution result"
        print(f"  ✗ Task {task_id} failed: No execution result")
    
    # Update todo list if manager exists
    if state.todo_manager:
        success = state.todo_manager.update_task_status(
            task_id,
            new_status,
            result=result,
            error=error
        )
        if success:
            print(f"  ✓ Todo list updated")
        else:
            print(f"  ⚠ Failed to update todo list")
        
        # Show progress summary
        if state.todo_manager._cached_todo_list:
            summary = state.todo_manager.get_progress_summary()
            print(f"  📊 Progress: {summary}")
    
    # Update current task status
    state.current_todo_task.status = new_status
    state.current_todo_task.result = result
    state.current_todo_task.error = error
    
    return state


def has_pending_tasks(state: CodeActState) -> str:
    """
    Router: Check if there are more pending tasks
    
    Returns:
        "continue": More pending tasks exist
        "end": No more pending tasks
    """
    print("  🔍 Checking for pending tasks...")
    
    # Check todo list for pending tasks
    if state.todo_list and state.todo_manager:
        pending_count = state.todo_manager.get_pending_count(state.todo_list)
        
        if pending_count > 0:
            print(f"  ℹ Found {pending_count} pending task(s), continuing...")
            return "continue"
        else:
            print(f"  ℹ No more pending tasks, finishing...")
            return "end"
    
    # No todo list - single task mode
    print(f"  ℹ Single task mode, finishing...")
    return "end"


def build_codeact_subgraph(use_todo_mode: bool = True):
    """
    Build CodeAct subgraph
    
    Args:
        use_todo_mode: If True, use new todo-list driven mode;
                       If False, use legacy single-task mode
    
    New Architecture (todo mode) with P2 enhancements:
    START → read_todo → select_next_task → infer_parameters → explore_data → generate_code → execute_code → extract_file_params → validate_output → update_todo → check_pending
                                                                                                                                                                                      ↓
                                                                                                                                                                         "continue" → select_next_task
                                                                                                                                                                         "end" → END
    
    Legacy Architecture (single-task mode):
    START → explore_data → generate_code → execute_code → validate_output → [success? END : revision → generate_code]
    """
    graph = StateGraph(CodeActState)
    
    # Add core nodes
    graph.add_node("explore_data", codeact_explore_data_node)
    graph.add_node("generate_code", codeact_generate_code_node)
    graph.add_node("execute_code", codeact_execute_code_node)
    graph.add_node("validate_output", codeact_validate_output_node)
    graph.add_node("revision", codeact_revision_node)
    
    # Add P2 nodes for parameter inference and file extraction
    graph.add_node("infer_parameters", infer_parameters_node)
    graph.add_node("extract_file_params", extract_file_params_node)
    
    # Add todo management nodes (new architecture)
    if use_todo_mode:
        graph.add_node("read_todo", read_todo_node)
        graph.add_node("select_next_task", select_next_task_node)
        graph.add_node("update_todo", update_todo_node)
        
        # P2 Enhanced flow:
        # START → read_todo → select_next_task → infer_parameters → explore_data → generate_code → execute_code → extract_file_params → validate_output → update_todo
        graph.add_edge(START, "read_todo")
        graph.add_edge("read_todo", "select_next_task")
        graph.add_edge("select_next_task", "infer_parameters")  # P2: Add parameter inference
        graph.add_edge("infer_parameters", "explore_data")      # P2: Then explore data
        graph.add_edge("explore_data", "generate_code")
    else:
        # Legacy flow with parameter inference:
        # START → infer_parameters → explore_data → generate_code
        graph.add_edge(START, "infer_parameters")  # P2: Add parameter inference even in legacy mode
        graph.add_edge("infer_parameters", "explore_data")
        graph.add_edge("explore_data", "generate_code")  # FIX: 添加缺失的边
    
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", "extract_file_params")  # P2: Extract files before validation
    graph.add_edge("extract_file_params", "validate_output")  # P2: Then validate
    
    # Router: After execution, decide next step
    def should_revise_or_update(state: CodeActState) -> str:
        """
        Determine next step after execution:
        1. If failed and can retry -> "revision"
        2. If failed and max retries reached -> "update_todo" (todo mode) or "end" (legacy)
        3. If success -> "update_todo" (todo mode) or "end" (legacy)
        """
        # Check if revision is needed
        if state.execution_result and state.execution_result.get("status") == "failed":
            error_type = state.execution_result.get("error_type", "")
            error = state.execution_result.get("error", "")
            
            # Don't retry on timeout errors
            if error_type == "TimeoutError" or "timed out" in error.lower():
                print(f"  ⚠ Execution timed out, skipping Revision")
                if use_todo_mode:
                    return "update_todo"
                return "end"
            
            # Check iteration limit
            current_iteration = state.revision_iteration
            max_iterations = 3
            
            if current_iteration < max_iterations:
                next_iteration = current_iteration + 1
                print(f"  🔄 Entering Revision (iteration {next_iteration}/{max_iterations})")
                return "revision"
            else:
                print(f"  ⚠ Maximum Revision iterations ({max_iterations}) reached")
                if use_todo_mode:
                    return "update_todo"
                return "end"
        
        # Success - update todo or end
        if use_todo_mode:
            return "update_todo"
        return "end"
    
    if use_todo_mode:
        # Todo mode: validate_output -> revision or update_todo
        graph.add_conditional_edges(
            "validate_output",
            should_revise_or_update,
            {
                "revision": "revision",
                "update_todo": "update_todo"
            }
        )
        
        # After revision, go back to generate_code
        graph.add_edge("revision", "generate_code")
        
        # After update_todo, check for pending tasks
        graph.add_conditional_edges(
            "update_todo",
            has_pending_tasks,
            {
                "continue": "select_next_task",
                "end": END
            }
        )
    else:
        # Legacy mode: validate_output -> revision or end
        graph.add_conditional_edges(
            "validate_output",
            should_revise_or_update,
            {
                "revision": "revision",
                "end": END
            }
        )
        
        # After revision, regenerate code
        graph.add_edge("revision", "generate_code")
    
    return graph.compile()


# ===================== State Mapping Functions =====================

def codeact_input_mapper(executor_state: Any, task: SubTask, execution_mode: CodeActExecutionMode, 
                         parameters: Dict[str, Any] = None, previous_code: str = None, 
                         previous_error: str = None, error_category: str = None,
                         revision_plan: Any = None, revision_iteration: int = 0,
                         parent_state: GlobalState = None) -> CodeActState:
    """
    Map Executor state to CodeAct subgraph state
    
    Args:
        executor_state: Executor state (optional)
        task: Task to execute
        execution_mode: Execution mode
        parameters: Parsed parameters
        previous_code: Previous code (for fixing)
        previous_error: Previous error (for fixing)
        error_category: Error category (for fixing)
        parent_state: Parent GlobalState reference (for opensandbox_id passing)
    
    Returns:
        CodeAct subgraph state
    """
    task_result = task.result if isinstance(task.result, dict) else {}
    tools = task_result.get("tools", [])
    inputs = task_result.get("inputs", [])
    
    # Validate and convert tools: ensure it's a list of dictionaries
    if isinstance(tools, list):
        # Filter and convert: only keep dict-type elements, convert strings to dict format
        validated_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                validated_tools.append(tool)
            elif isinstance(tool, str):
                # String tool name, convert to dict format (for error handling tests)
                validated_tools.append({"name": tool, "type": "unknown"})
            # Other types ignored
        tools = validated_tools
    else:
        tools = []
    
    # Extract todo_list_path from parameters if present
    todo_list_path = None
    if parameters:
        todo_list_path = parameters.get("todo_list_path")
    
    # If todo_list_path not in parameters, try to build from session_id
    if not todo_list_path:
        # Try parameters first
        session_id = parameters.get("session_id") if parameters else None
        # Then try parent_state
        if not session_id and parent_state and hasattr(parent_state, 'session_id'):
            session_id = parent_state.session_id
        # Build todo_list_path from session_id
        if session_id:
            todo_list_path = f"/data/sessions/{session_id}/todo-list.md"
    
    # Extract user_input_context from parent_state for parameter inference
    user_input_context = None
    if parent_state:
        if hasattr(parent_state, 'user_input'):
            user_input_context = parent_state.user_input
        elif isinstance(parent_state, dict):
            user_input_context = parent_state.get('user_input')
    
    # Ensure task is correctly passed (Pydantic v2 compatibility)
    try:
        return CodeActState(
            task=task,
            task_description=task.content,
            tools=tools,
            inputs=inputs,
            parameters=parameters or {},
            execution_mode=execution_mode,
            previous_code=previous_code,
            previous_error=previous_error,
            error_category=error_category,
            revision_plan=revision_plan,
            revision_iteration=revision_iteration,
            parent_state=parent_state,
            todo_list_path=todo_list_path,  # Pass todo_list_path from parameters
            user_input_context=user_input_context  # Pass user_input for parameter inference
        )
    except Exception:
        # If direct construction fails, use model_validate
        task_dict = task.model_dump() if hasattr(task, 'model_dump') else task.dict() if hasattr(task, 'dict') else task
        return CodeActState.model_validate({
            "task": task_dict,
            "task_description": task.content,
            "tools": tools,
            "inputs": inputs,
            "parameters": parameters or {},
            "execution_mode": execution_mode,
            "previous_code": previous_code,
            "previous_error": previous_error,
            "error_category": error_category,
            "revision_plan": revision_plan,
            "revision_iteration": revision_iteration,
            "parent_state": parent_state,
            "todo_list_path": todo_list_path,  # Pass todo_list_path from parameters
            "user_input_context": user_input_context  # Pass user_input for parameter inference
        })


def codeact_output_mapper(codeact_state: Union[CodeActState, Dict[str, Any], list]) -> Dict[str, Any]:
    """
    Map CodeAct subgraph state back to execution result
    
    Args:
        codeact_state: CodeAct subgraph state (can be CodeActState object, dict, or list)
    
    Returns:
        Execution result dictionary
    """
    # Handle list type (from LangGraph stream/invoke in some cases)
    if isinstance(codeact_state, list):
        # Try to find a valid element in the list
        for item in codeact_state:
            if isinstance(item, dict):
                codeact_state = item
                break
            elif hasattr(item, 'execution_result'):
                codeact_state = item
                break
        else:
            # No valid element found
            return {
                "status": "failed",
                "code": None,
                "output": None,
                "error": f"codeact_output_mapper received list with {len(codeact_state)} items, none convertible to CodeActState",
                "error_type": "InvalidStateType",
                "error_category": "state_mapping"
            }
    
    # If dict, convert to CodeActState object
    if isinstance(codeact_state, dict):
        try:
            codeact_state = CodeActState.model_validate(codeact_state)
        except Exception as e:
            return {
                "status": "failed",
                "code": None,
                "output": None,
                "error": f"Failed to validate CodeActState: {e}",
                "error_type": "StateValidationError",
                "error_category": "state_mapping"
            }
    
    # Handle None input
    if codeact_state is None:
        return {
            "status": "failed",
            "code": None,
            "output": None,
            "error": "codeact_state is None",
            "error_type": "NullState",
            "error_category": "state_mapping"
        }
    
    # Handle None execution_result with diagnostic info
    if not codeact_state.execution_result:
        # Build diagnostic error message
        error_parts = []
        if not codeact_state.generated_code:
            error_parts.append("No code was generated")
        else:
            error_parts.append(f"Code was generated ({len(codeact_state.generated_code)} chars) but execution_result is None")
        
        if codeact_state.previous_error:
            error_parts.append(f"Previous error: {codeact_state.previous_error[:200]}")
        
        return {
            "status": "failed",
            "code": codeact_state.generated_code,
            "output": None,
            "error": "; ".join(error_parts) if error_parts else "execution_result is None",
            "error_type": "NoExecutionResult",
            "error_category": codeact_state.error_category or "execution"
        }
    
    # Handle execution_result being a list (unexpected but can happen)
    exec_result = codeact_state.execution_result
    if isinstance(exec_result, list):
        # Try to extract the first dict element or convert to a usable format
        if len(exec_result) > 0 and isinstance(exec_result[0], dict):
            exec_result = exec_result[0]
        else:
            # The list contains non-dict items, treat as output
            return {
                "status": "success",
                "code": codeact_state.generated_code,
                "output": str(exec_result),
                "error": None,
                "error_type": None,
                "error_category": codeact_state.error_category or "execution"
            }
    
    # Now exec_result should be a dict
    if not isinstance(exec_result, dict):
        return {
            "status": "failed",
            "code": codeact_state.generated_code,
            "output": str(exec_result),
            "error": f"execution_result is not a dict: {type(exec_result).__name__}",
            "error_type": "InvalidExecutionResult",
            "error_category": codeact_state.error_category or "execution"
        }
    
    return {
        "status": exec_result.get("status", "unknown"),
        "code": codeact_state.generated_code,
        "output": exec_result.get("output"),
        "error": exec_result.get("error"),
        "error_type": exec_result.get("error_type"),
        "error_category": exec_result.get("error_category") or codeact_state.error_category or "execution"
    }

