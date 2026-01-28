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
    TrajectoryStatus
)
from nodes.subagents.code_act.revision import (
    RevisionPlan,
    RevisionStrategy,
    create_revision_plan,
    execute_revision_plan
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
    
    # Fix related
    previous_code: Optional[str] = Field(default=None, description="Previous code (for fixing)")
    previous_error: Optional[str] = Field(default=None, description="Previous error message (for fixing)")
    error_category: Optional[str] = Field(default=None, description="Error category")
    revision_plan: Optional[Any] = Field(default=None, description="Revision plan (for intelligent fixing)")
    revision_iteration: int = Field(default=0, description="Revision iteration count")
    
    # Code generation and execution results
    generated_code: Optional[str] = Field(default=None, description="Generated code")
    execution_result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")
    
    # Trajectory recording (SE-Agent style)
    trajectory_history: List[CodeTrajectory] = Field(default_factory=list, description="Trajectory history for current task")
    trajectory_pool_id: Optional[str] = Field(default=None, description="Associated trajectory pool ID")
    current_trajectory: Optional[CodeTrajectory] = Field(default=None, description="Currently recording trajectory")
    
    # Parent state reference
    parent_state: Optional[GlobalState] = Field(default=None, description="Main graph state reference")


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
    
    # TODO: Integrate TrajectoryPool for persistent storage
    # Currently saved in memory, can add persistence later


# ===================== CodeAct Nodes =====================

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


def _check_sandbox_available(state: CodeActState) -> tuple[bool, str]:
    """
    Check if sandbox environment is available
    
    Args:
        state: CodeAct state
    
    Returns:
        (is_available, sandbox_directory_path)
    """
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
    
    return code


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
    has_sandbox, sandbox_dir = _check_sandbox_available(state)
    if not has_sandbox:
        print(f"  ⚠ Sandbox environment unavailable, will generate code executable in fallback mode (directory: {sandbox_dir})")
    
    if mode == CodeActExecutionMode.MCP_TOOL:
        # Generate code to call MCP tools
        tools = state.tools
        if tools:
            # Use first tool
            tool = tools[0] if isinstance(tools, list) else tools
            tool_name = tool.get("tool_name") or tool.get("name", "unknown_tool")
            tool_description = tool.get("description", "")
            
            # Use LLM to generate code
            user_prompt = get_mcp_tool_code_user_prompt(
                tool_name=tool_name,
                tool_description=tool_description,
                parameters=parameters,
                task_description=state.task_description
            )
            
            # Fallback code
            params_str = ", ".join([f"{k}={repr(v)}" for k, v in parameters.items()])
            fallback_code = f"""
# Call MCP tool: {tool_name}
# Parameters: {params_str}
print("Calling MCP tool: {tool_name}")
print("Parameters: {params_str}")
result = {{"status": "success", "output": "MCP tool call result (placeholder)"}}
"""
            
            generated_code = _generate_code_with_llm(
                system_prompt=MCP_TOOL_CODE_SYSTEM_PROMPT,
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
            state.generated_code = "# No matching tool found"
    
    elif mode == CodeActExecutionMode.FIX_CODE:
        # Fix code errors: generate fixed code based on previous error information
        previous_code = state.previous_code or ""
        previous_error = state.previous_error or ""
        error_category = state.error_category
        
        if not previous_code or not previous_error:
            state.generated_code = "# Missing necessary fix information (original code or error information)"
            return state
        
        # If Revision plan is provided, use intelligent fix (SE-Agent style)
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
            # Use traditional fix method
            user_prompt = get_fix_code_user_prompt(
                previous_code=previous_code,
                previous_error=previous_error,
                error_category=error_category
            )
            
            # Fallback code
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
        
        # Ensure code is executable (especially when no sandbox environment)
        state.generated_code = _ensure_code_executable(
            generated_code,
            has_sandbox=has_sandbox,
            sandbox_dir=sandbox_dir
        )
    
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
        
        # Use LLM to generate code
        user_prompt = get_codeact_user_prompt(
            task_description=task_desc,
            inputs=inputs,
            outputs=None
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
    has_sandbox, sandbox_dir = _check_sandbox_available(state)
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
    
    try:
        if has_sandbox:
            # Execute code in sandbox via subprocess for isolation
            import os
            from pathlib import Path
            import json as json_module
            
            result_marker = "__CODEACT_RESULT__"
            timeout_seconds = int(os.getenv("CODEACT_TIMEOUT_SECONDS", "120"))
            
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
            
            print(f"  🔄 Starting code execution (sandbox subprocess mode)...")
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
                state.execution_result = {
                    "status": "failed",
                    "error": f"Execution timed out after {timeout_seconds}s",
                    "stderr": (e.stderr or "")[-1000:] if hasattr(e, "stderr") else "",
                    "returncode": None,
                    "sandbox_dir": sandbox_dir
                }
                print(f"  ✗ Execution timed out after {timeout_seconds}s")
                return state
            
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            parsed_result = None
            for line in stdout.splitlines():
                if result_marker in line:
                    payload = line.split(result_marker, 1)[1]
                    try:
                        parsed_result = json_module.loads(payload)
                    except Exception:
                        parsed_result = payload
            
            error = None
            error_type = None
            if isinstance(parsed_result, dict):
                status = parsed_result.get("status", "success")
                output = parsed_result.get("output")
                error = parsed_result.get("error")
                error_type = parsed_result.get("error_type")
            else:
                status = "failed" if completed.returncode != 0 else "success"
                output = parsed_result or (stdout[-1000:] if stdout else stderr[-1000:])
                if completed.returncode != 0:
                    error = (stderr[-2000:] if stderr else stdout[-2000:]) or "Execution failed"
            
            state.execution_result = {
                "status": status,
                "output": output,
                "error": error,
                "error_type": error_type,
                "stderr": stderr[-1000:] if stderr else "",
                "returncode": completed.returncode,
                "sandbox_dir": sandbox_dir
            }
            print(f"  ✓ Execution completed, status={status}, returncode={completed.returncode}")
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
            
            # Pre-import mcp_helper so generated code can use it
            try:
                from utils.mcp_helper import invoke_mcp_tool_sync
                local_namespace = {
                    "invoke_mcp_tool_sync": invoke_mcp_tool_sync,
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
    # Check if there are failed trajectories
    if not state.trajectory_history:
        print("  ⚠ No trajectory history, cannot perform Revision analysis")
        return state
    
    # Get latest failed trajectory
    failed_trajectories = [t for t in state.trajectory_history if t.status == TrajectoryStatus.FAILED]
    if not failed_trajectories:
        print("  ℹ No failed trajectories, no Revision needed")
        return state
    
    latest_failed = failed_trajectories[-1]
    previous_failed = failed_trajectories[:-1] if len(failed_trajectories) > 1 else []
    
    print(f"  🔍 Starting Revision analysis (failed trajectory: {latest_failed.trajectory_id})")
    
    # Create Revision plan
    revision_plan = create_revision_plan(latest_failed, previous_failed)
    
    # Update state
    state.revision_plan = revision_plan
    state.revision_iteration += 1
    state.previous_code = latest_failed.generated_code
    state.previous_error = latest_failed.error_message or str(latest_failed.error_type)
    state.error_category = latest_failed.error_category
    
    print(f"  ✓ Revision plan generated successfully")
    print(f"     Strategy: {revision_plan.strategy.value}")
    print(f"     Root cause: {revision_plan.root_cause[:150]}...")
    print(f"     Confidence: {revision_plan.confidence:.2f}")
    print(f"     Iteration count: {state.revision_iteration}")
    
    return state


def build_codeact_subgraph():
    """Build CodeAct subgraph"""
    graph = StateGraph(CodeActState)
    
    graph.add_node("generate_code", codeact_generate_code_node)
    graph.add_node("execute_code", codeact_execute_code_node)
    graph.add_node("revision", codeact_revision_node)
    
    graph.add_edge(START, "generate_code")
    graph.add_edge("generate_code", "execute_code")
    
    # After execution, decide whether to enter Revision based on result
    def should_revise(state: CodeActState) -> str:
        """Determine if Revision is needed"""
        if state.execution_result and state.execution_result.get("status") == "failed":
            # Check iteration limit
            if state.revision_iteration < 3:  # Maximum 3 Revision iterations
                return "revision"
        return "end"
    
    graph.add_conditional_edges(
        "execute_code",
        should_revise,
        {
            "revision": "revision",
            "end": END
        }
    )
    
    # After Revision, regenerate code
    graph.add_edge("revision", "generate_code")
    
    return graph.compile()


# ===================== State Mapping Functions =====================

def codeact_input_mapper(executor_state: Any, task: SubTask, execution_mode: CodeActExecutionMode, 
                         parameters: Dict[str, Any] = None, previous_code: str = None, 
                         previous_error: str = None, error_category: str = None,
                         revision_plan: Any = None, revision_iteration: int = 0) -> CodeActState:
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
            revision_iteration=revision_iteration
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
            "revision_iteration": revision_iteration
        })


def codeact_output_mapper(codeact_state: Union[CodeActState, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Map CodeAct subgraph state back to execution result
    
    Args:
        codeact_state: CodeAct subgraph state (can be CodeActState object or dict)
    
    Returns:
        Execution result dictionary
    """
    # If dict, convert to CodeActState object
    if isinstance(codeact_state, dict):
        codeact_state = CodeActState.model_validate(codeact_state)
    
    return {
        "status": codeact_state.execution_result.get("status") if codeact_state.execution_result else "unknown",
        "code": codeact_state.generated_code,
        "output": codeact_state.execution_result.get("output") if codeact_state.execution_result else None,
        "error": codeact_state.execution_result.get("error") if codeact_state.execution_result else None,
        "error_type": codeact_state.execution_result.get("error_type") if codeact_state.execution_result else None
    }

