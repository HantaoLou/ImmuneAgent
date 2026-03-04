"""
Todo List Manager for CodeAct Subgraph

Responsible for:
1. Reading todo-list.md from sandbox directory
2. Parsing task definitions
3. Selecting next pending task
4. Updating task status and results

Todo-list.md format:
```markdown
# Task List

## Session Info
- session_id: 20260302_abc123
- created_at: 2026-03-02 10:00:00
- sandbox_id: sb_xxx

## Tasks

### Task 1: 文件上传
- id: task_001
- type: file_upload
- status: pending
- priority: 1
- description: 将本地 CSV 文件上传到沙箱 input 目录
- parameters:
    source_path: /local/data.csv
    target_path: /data/sessions/{session_id}/input/data.csv
- result: null
- error: null
```
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from pathlib import Path
import re
import json
import yaml
from datetime import datetime

# Import FileParameterTable
from nodes.subagents.code_act.file_param_table import (
    FileParameter,
    FileParameterTable,
    FileSource,
    create_file_param_from_user_input,
    create_file_param_from_task_output,
    extract_file_info_from_task_result
)


class TodoTaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TodoTaskType(str, Enum):
    """Task type enumeration"""
    FILE_UPLOAD = "file_upload"
    FILE_ANALYSIS = "file_analysis"
    FILE_CONVERT = "file_convert"  # e.g., CSV to FASTA
    MCP_TOOL = "mcp_tool"
    GENERAL = "general"


class TodoTask(BaseModel):
    """Single task model"""
    model_config = ConfigDict(
        use_enum_values=False,  # Keep enum types for proper comparison
        validate_assignment=True
    )
    
    id: str = Field(description="Unique task ID")
    type: TodoTaskType = Field(description="Task type")
    status: TodoTaskStatus = Field(default=TodoTaskStatus.PENDING, description="Task status")
    priority: int = Field(default=5, description="Priority (1=highest, 5=lowest)")
    description: str = Field(description="Task description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    dependencies: List[str] = Field(default_factory=list, description="Dependent task IDs")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    started_at: Optional[str] = Field(default=None, description="Start timestamp")
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp")


class TodoListSession(BaseModel):
    """Session information"""
    session_id: str = Field(description="Session ID")
    created_at: str = Field(description="Creation timestamp")
    sandbox_id: Optional[str] = Field(default=None, description="Sandbox instance ID")
    sandbox_dir: Optional[str] = Field(default=None, description="Sandbox directory path")


class TodoList(BaseModel):
    """Complete todo list model"""
    model_config = ConfigDict(
        use_enum_values=False,  # Keep enum types in nested models
        validate_assignment=True
    )
    
    session: TodoListSession = Field(description="Session information")
    tasks: List[TodoTask] = Field(default_factory=list, description="List of tasks")
    file_parameter_table: Optional[FileParameterTable] = Field(
        default=None, 
        description="File parameter table for dynamic parameter inference"
    )


class TodoListManager:
    """
    Manager for reading, parsing, and updating todo-list.md
    
    Supports both local filesystem and remote sandbox (via opensandbox_id).
    When opensandbox_id is provided, file operations are done via CodeAct.
    
    New in P2: Also manages FileParameterTable for dynamic parameter inference.
    """
    
    DEFAULT_FILENAME = "todo-list.md"
    FILE_PARAMS_FILENAME = "file-params.json"
    
    def __init__(self, sandbox_dir: str, filename: str = None, opensandbox_id: str = None):
        """
        Initialize TodoListManager
        
        Args:
            sandbox_dir: Sandbox directory path (e.g., /data/sessions/{session_id}/)
            filename: Todo list filename (default: todo-list.md)
            opensandbox_id: OpenSandbox instance ID for remote operations
        """
        self.sandbox_dir_str = sandbox_dir  # Keep original string for remote paths
        self.sandbox_dir = Path(sandbox_dir)
        self.filename = filename or self.DEFAULT_FILENAME
        self.opensandbox_id = opensandbox_id
        self._cached_todo_list: Optional[TodoList] = None
        self._cached_file_params: Optional[FileParameterTable] = None
        
        # Determine if this is a remote sandbox path
        self._is_remote = str(sandbox_dir).startswith('/data/') or str(sandbox_dir).startswith('/tmp/')
        
        # For remote paths, use string concatenation to preserve Unix-style paths
        # For local paths, use Path for proper cross-platform handling
        if self._is_remote:
            # Ensure sandbox_dir ends with / for proper concatenation
            base_dir = sandbox_dir.rstrip('/')
            self.todo_list_path_str = f"{base_dir}/{self.filename}"
            self.file_params_path_str = f"{base_dir}/{self.FILE_PARAMS_FILENAME}"
            # Also set Path version for local fallback operations
            self.todo_list_path = Path(self.todo_list_path_str)
            self.file_params_path = Path(self.file_params_path_str)
        else:
            self.todo_list_path = self.sandbox_dir / self.filename
            self.file_params_path = self.sandbox_dir / self.FILE_PARAMS_FILENAME
            self.todo_list_path_str = str(self.todo_list_path)
            self.file_params_path_str = str(self.file_params_path)
    
    def todo_list_exists(self) -> bool:
        """Check if todo-list.md exists (supports remote sandbox)"""
        if self._is_remote and self.opensandbox_id:
            # Check via CodeAct for remote sandbox
            return self._check_remote_exists()
        else:
            # Local filesystem check
            return self.todo_list_path.exists()
    
    def _check_remote_exists(self) -> bool:
        """Check if file exists in remote sandbox via CodeAct"""
        try:
            from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
            
            if not is_codeact_available():
                return False
            
            # Use string path for remote sandbox (Unix-style)
            check_code = f'''
import os
path = "{self.todo_list_path_str}"
exists = os.path.exists(path)
print(f"__EXISTS__:{{exists}}")
'''
            result = execute_code_via_codeact(
                task_description=f"检查远程文件是否存在: {self.todo_list_path}",
                code_template=check_code,
                sandbox_id=self.opensandbox_id,
                timeout_seconds=10,
                keep_alive=True  # 保持沙盒存活
            )
            
            # CodeActResult 是 dataclass，使用属性访问而不是 .get()
            if result and result.is_success():
                output = result.output or ""
                return "__EXISTS__:True" in str(output)
            return False
        except Exception as e:
            print(f"⚠️ Failed to check remote file existence: {e}")
            return False
    
    def read_todo_list(self) -> TodoList:
        """
        Read and parse todo-list.md from sandbox
        
        Returns:
            TodoList: Parsed todo list
            
        Raises:
            FileNotFoundError: If todo-list.md doesn't exist
            ValueError: If parsing fails
        """
        if not self.todo_list_exists():
            raise FileNotFoundError(f"Todo list not found: {self.todo_list_path}")
        
        content = self._read_file_content()
        todo_list = self._parse_markdown(content)
        self._cached_todo_list = todo_list
        return todo_list
    
    def _read_file_content(self) -> str:
        """Read file content (supports both local and remote sandbox)"""
        # For remote sandbox with opensandbox_id
        if self._is_remote and self.opensandbox_id:
            return self._read_remote_file()
        
        # For local filesystem
        if self.todo_list_path.exists():
            return self.todo_list_path.read_text(encoding="utf-8")
        
        raise FileNotFoundError(f"Cannot read todo list: {self.todo_list_path}")
    
    def _read_remote_file(self) -> str:
        """Read file content from remote sandbox via CodeAct"""
        try:
            from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
            
            if not is_codeact_available():
                raise RuntimeError("CodeAct not available for remote file reading")
            
            # Use string path for remote sandbox (Unix-style)
            read_code = f'''
import os
path = "{self.todo_list_path_str}"
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Use a marker to help extract content
    print("__CONTENT_START__")
    print(content)
    print("__CONTENT_END__")
else:
    print("__NOT_FOUND__")
'''
            result = execute_code_via_codeact(
                task_description=f"读取远程文件: {self.todo_list_path}",
                code_template=read_code,
                sandbox_id=self.opensandbox_id,
                timeout_seconds=30,
                keep_alive=True  # 保持沙盒存活
            )
            
            # CodeActResult 是 dataclass，使用属性访问而不是 .get()
            if result and result.is_success():
                output = str(result.output or "")
                if "__NOT_FOUND__" in output:
                    raise FileNotFoundError(f"Todo list not found in remote sandbox: {self.todo_list_path}")
                
                # Extract content between markers
                if "__CONTENT_START__" in output and "__CONTENT_END__" in output:
                    start = output.find("__CONTENT_START__") + len("__CONTENT_START__")
                    end = output.find("__CONTENT_END__")
                    return output[start:end].strip()
                
                return output
            else:
                error = result.error if result else "No result"
                raise RuntimeError(f"Failed to read remote file: {error}")
                
        except Exception as e:
            raise FileNotFoundError(f"Failed to read todo list from remote sandbox: {e}")
    
    def _parse_markdown(self, content: str) -> TodoList:
        """
        Parse markdown content into TodoList
        
        Args:
            content: Markdown content
            
        Returns:
            TodoList: Parsed todo list
        """
        lines = content.strip().split("\n")
        
        # Parse session info
        session_info = {}
        in_session_section = False
        
        # Parse tasks
        tasks = []
        current_task = None
        in_parameters = False
        parameters_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Detect session section
            if stripped == "## Session Info":
                in_session_section = True
                continue
            
            if stripped.startswith("## ") and in_session_section:
                in_session_section = False
            
            # Parse session info
            if in_session_section and stripped.startswith("- "):
                match = re.match(r"- (\w+):\s*(.+)", stripped)
                if match:
                    key, value = match.groups()
                    session_info[key] = value
            
            # Detect task header
            # Support both formats:
            # - New format with emoji: "### ⏳ Task: task_001"
            # - Old format: "### Task 1: 文件上传"
            is_task_header = (
                (stripped.startswith("### ") and "Task:" in stripped) or  # New format with emoji
                (stripped.startswith("### Task ") and ":" in stripped)    # Old format
            )
            
            if is_task_header:
                # Save previous task
                if current_task:
                    if in_parameters and parameters_lines:
                        current_task["parameters"] = self._parse_yaml_block(parameters_lines)
                    tasks.append(self._create_task(current_task))
                
                # Start new task - extract task ID from header
                # Format: "### ⏳ Task: test_task_001" or "### Task 1: 文件上传"
                if "Task:" in stripped:
                    # New format: extract ID after "Task:"
                    task_title = stripped.split("Task:", 1)[1].strip()
                else:
                    # Old format: extract after first colon
                    task_title = stripped.split(":", 1)[1].strip()
                current_task = {"description": task_title}
                in_parameters = False
                parameters_lines = []
                continue
            
            # Parse task properties
            if current_task is not None:
                # Check if this is a property line (starts with "- key:")
                prop_match = re.match(r"- (\w+):\s*(.*)", stripped)
                
                if prop_match:
                    key, value = prop_match.groups()
                    
                    # Handle parameters section - collect lines until next property
                    if key == "parameters":
                        in_parameters = True
                        parameters_lines = []  # Reset for new parameters
                        continue
                    
                    # Any other property ends parameter collection
                    if in_parameters:
                        # First, save collected parameters
                        if parameters_lines:
                            current_task["parameters"] = self._parse_yaml_block(parameters_lines)
                            parameters_lines = []
                        in_parameters = False
                    
                    # Parse property value
                    if key == "dependencies":
                        # Parse list like [task_001, task_002]
                        if value.startswith("[") and value.endswith("]"):
                            deps = [d.strip() for d in value[1:-1].split(",") if d.strip()]
                            current_task[key] = deps
                        else:
                            current_task[key] = []
                    elif key in ("priority",):
                        current_task[key] = int(value) if value else 5
                    elif value == "null" or value == "":
                        current_task[key] = None
                    else:
                        current_task[key] = value
                
                elif in_parameters:
                    # Collect indented parameter lines
                    # Parameters are lines that are indented (original line starts with spaces)
                    # or look like "key: value" without leading dash
                    if line.startswith("    ") or line.startswith("\t"):
                        # This is an indented parameter line
                        clean_line = stripped
                        if ":" in clean_line:
                            parameters_lines.append(clean_line)
                    elif stripped and not stripped.startswith("-"):
                        # Parameter without leading dash
                        if ":" in stripped:
                            parameters_lines.append(stripped)
        
        # Don't forget the last task
        if current_task:
            if in_parameters and parameters_lines:
                current_task["parameters"] = self._parse_yaml_block(parameters_lines)
            tasks.append(self._create_task(current_task))
        
        # Create TodoList
        session = TodoListSession(
            session_id=session_info.get("session_id", "unknown"),
            created_at=session_info.get("created_at", datetime.now().isoformat()),
            sandbox_id=session_info.get("sandbox_id"),
            sandbox_dir=session_info.get("sandbox_dir")
        )
        
        return TodoList(session=session, tasks=tasks)
    
    def _parse_yaml_block(self, lines: List[str]) -> Dict[str, Any]:
        """Parse YAML-style parameter block"""
        try:
            yaml_content = "\n".join(lines)
            return yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            # Fallback: parse simple key: value pairs
            result = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    result[key.strip()] = value.strip()
            return result
    
    def _create_task(self, task_dict: Dict[str, Any]) -> TodoTask:
        """Create TodoTask from dictionary"""
        # Parse result field: convert string to dict if needed
        result = task_dict.get("result")
        if isinstance(result, str):
            if result.lower() in ("null", "none", ""):
                result = None
            else:
                # Try to parse as JSON
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, ValueError):
                    # Keep as dict with raw value
                    result = {"raw": result}
        
        # Parse type field: convert string to enum if needed
        task_type = task_dict.get("type", TodoTaskType.GENERAL)
        if not isinstance(task_type, TodoTaskType):
            try:
                task_type = TodoTaskType(task_type)
            except (ValueError, TypeError):
                task_type = TodoTaskType.GENERAL
        
        # Parse status field: convert string to enum if needed
        status = task_dict.get("status", TodoTaskStatus.PENDING)
        if not isinstance(status, TodoTaskStatus):
            try:
                status = TodoTaskStatus(status)
            except (ValueError, TypeError):
                status = TodoTaskStatus.PENDING
        
        return TodoTask(
            id=task_dict.get("id", f"task_{datetime.now().timestamp()}"),
            type=task_type,
            status=status,
            priority=task_dict.get("priority", 5),
            description=task_dict.get("description", ""),
            parameters=task_dict.get("parameters", {}),
            dependencies=task_dict.get("dependencies", []),
            result=result,
            error=task_dict.get("error")
        )
    
    def get_next_pending_task(self, todo_list: TodoList = None) -> Optional[TodoTask]:
        """
        Get the next pending task to execute
        
        Selection logic:
        1. Filter tasks with status=pending
        2. Filter tasks whose dependencies are all completed
        3. Sort by priority (ascending, 1=highest)
        4. Return the first task
        
        Args:
            todo_list: TodoList to search (uses cached if not provided)
            
        Returns:
            TodoTask: Next task to execute, or None if no pending tasks
        """
        if todo_list is None:
            todo_list = self._cached_todo_list
        
        if todo_list is None:
            return None
        
        # Helper function to get status value (handle both enum and string)
        def get_status_value(status):
            if isinstance(status, TodoTaskStatus):
                return status.value
            return str(status)
        
        # Build completed task IDs set (compare by value to handle both enum and string)
        completed_ids = {
            task.id for task in todo_list.tasks 
            if get_status_value(task.status) == TodoTaskStatus.COMPLETED.value
        }
        
        # Find pending tasks with satisfied dependencies
        pending_tasks = []
        for task in todo_list.tasks:
            # Check if task is pending (compare by value)
            if get_status_value(task.status) != TodoTaskStatus.PENDING.value:
                continue
            
            # Check dependencies
            deps_satisfied = all(
                dep_id in completed_ids 
                for dep_id in task.dependencies
            )
            
            if deps_satisfied:
                pending_tasks.append(task)
        
        # Sort by priority and return first
        if pending_tasks:
            pending_tasks.sort(key=lambda t: t.priority)
            return pending_tasks[0]
        
        return None
    
    def update_task_status(
        self,
        task_id: str,
        status: TodoTaskStatus,
        result: Dict[str, Any] = None,
        error: str = None
    ) -> bool:
        """
        Update task status in todo-list.md
        
        Args:
            task_id: Task ID to update
            status: New status
            result: Execution result (for completed tasks)
            error: Error message (for failed tasks)
            
        Returns:
            bool: True if update succeeded
        """
        if self._cached_todo_list is None:
            self.read_todo_list()
        
        # Update task in cached list
        task_updated = False
        for task in self._cached_todo_list.tasks:
            if task.id == task_id:
                task.status = status
                task.result = result
                task.error = error
                
                if status == TodoTaskStatus.IN_PROGRESS:
                    task.started_at = datetime.now().isoformat()
                elif status in (TodoTaskStatus.COMPLETED, TodoTaskStatus.FAILED):
                    task.completed_at = datetime.now().isoformat()
                
                task_updated = True
                break
        
        if not task_updated:
            return False
        
        # Write back to file
        return self._write_todo_list()
    
    def _write_todo_list(self) -> bool:
        """Write cached todo list back to markdown file (supports remote sandbox)"""
        if self._cached_todo_list is None:
            return False
        
        content = self._render_markdown(self._cached_todo_list)
        
        # For remote sandbox with opensandbox_id
        if self._is_remote and self.opensandbox_id:
            return self._write_remote_file(content)
        
        # For local filesystem
        try:
            # Ensure directory exists
            self.sandbox_dir.mkdir(parents=True, exist_ok=True)
            
            # Write file
            self.todo_list_path.write_text(content, encoding="utf-8")
            return True
        except Exception as e:
            print(f"Error writing todo list: {e}")
            return False
    
    def _write_remote_file(self, content: str) -> bool:
        """Write file content to remote sandbox via CodeAct"""
        try:
            from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
            
            if not is_codeact_available():
                print("⚠️ CodeAct not available for remote file writing")
                return False
            
            # Escape content for Python string
            escaped_content = content.replace('\\', '\\\\').replace('"""', '\\"\\"\\')
            
            # Use string paths for remote sandbox (Unix-style)
            write_code = f'''
import os

# Ensure directory exists
os.makedirs("{self.sandbox_dir_str}", exist_ok=True)

# Write file
file_path = "{self.todo_list_path_str}"
content = """{escaped_content}"""

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"__WRITTEN__:{{file_path}}")
'''
            result = execute_code_via_codeact(
                task_description=f"写入远程文件: {self.todo_list_path_str}",
                code_template=write_code,
                sandbox_id=self.opensandbox_id,
                timeout_seconds=30,
                keep_alive=True  # 保持沙盒存活
            )
            
            # CodeActResult 是 dataclass，使用属性访问而不是 .get()
            if result and result.is_success():
                output = str(result.output or "")
                if "__WRITTEN__:" in output:
                    print(f"  📝 Todo list written to remote: {self.todo_list_path_str}")
                    return True
            
            error = result.error if result else "No result"
            print(f"⚠️ Failed to write remote file: {error}")
            return False
            
        except Exception as e:
            print(f"⚠️ Error writing to remote sandbox: {e}")
            return False
    
    def _render_markdown(self, todo_list: TodoList) -> str:
        """Render TodoList to markdown format"""
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
        
        for i, task in enumerate(todo_list.tasks, 1):
            lines.append(f"### Task {i}: {task.description}")
            lines.append(f"- id: {task.id}")
            lines.append(f"- type: {task.type.value if hasattr(task.type, 'value') else task.type}")
            lines.append(f"- status: {task.status.value if hasattr(task.status, 'value') else task.status}")
            lines.append(f"- priority: {task.priority}")
            lines.append(f"- description: {task.description}")
            
            if task.dependencies:
                lines.append(f"- dependencies: {task.dependencies}")
            
            if task.parameters:
                lines.append("- parameters:")
                for key, value in task.parameters.items():
                    lines.append(f"    {key}: {value}")
            
            if task.result:
                lines.append(f"- result: {json.dumps(task.result)}")
            elif task.status == TodoTaskStatus.COMPLETED:
                lines.append("- result: {}")
            else:
                lines.append("- result: null")
            
            if task.error:
                lines.append(f"- error: {task.error}")
            else:
                lines.append("- error: null")
            
            if task.started_at:
                lines.append(f"- started_at: {task.started_at}")
            if task.completed_at:
                lines.append(f"- completed_at: {task.completed_at}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def create_todo_list(
        self,
        session_id: str,
        sandbox_id: str = None,
        sandbox_dir: str = None,
        tasks: List[TodoTask] = None
    ) -> TodoList:
        """
        Create a new todo list
        
        Args:
            session_id: Session ID
            sandbox_id: Sandbox instance ID
            sandbox_dir: Sandbox directory path
            tasks: Initial tasks
            
        Returns:
            TodoList: Created todo list
        """
        session = TodoListSession(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            sandbox_id=sandbox_id,
            sandbox_dir=sandbox_dir or str(self.sandbox_dir)
        )
        
        todo_list = TodoList(
            session=session,
            tasks=tasks or []
        )
        
        self._cached_todo_list = todo_list
        self._write_todo_list()
        
        return todo_list
    
    def add_task(self, task: TodoTask) -> bool:
        """
        Add a new task to the todo list
        
        Args:
            task: Task to add
            
        Returns:
            bool: True if task was added successfully
        """
        if self._cached_todo_list is None:
            self.read_todo_list()
        
        self._cached_todo_list.tasks.append(task)
        return self._write_todo_list()
    
    def get_pending_count(self, todo_list: TodoList = None) -> int:
        """Get count of pending tasks"""
        if todo_list is None:
            todo_list = self._cached_todo_list
        
        if todo_list is None:
            return 0
        
        pending_value = TodoTaskStatus.PENDING.value
        return sum(
            1 for task in todo_list.tasks 
            if (task.status.value if isinstance(task.status, TodoTaskStatus) else str(task.status)) == pending_value
        )
    
    def get_progress_summary(self, todo_list: TodoList = None) -> Dict[str, int]:
        """Get progress summary by status"""
        if todo_list is None:
            todo_list = self._cached_todo_list
        
        if todo_list is None:
            return {}
        
        summary = {status.value: 0 for status in TodoTaskStatus}
        for task in todo_list.tasks:
            status_val = task.status.value if isinstance(task.status, TodoTaskStatus) else str(task.status)
            if status_val in summary:
                summary[status_val] += 1
        
        return summary
        

    # ===================== File Parameter Table Management (P2) =====================
    
    def get_file_parameter_table(self) -> FileParameterTable:
        """
        Get the file parameter table, loading from file if necessary
        
        Returns:
            FileParameterTable: The file parameter table
        """
        if self._cached_file_params is None:
            self._load_file_params()
        
        # If still None, create a new empty table
        if self._cached_file_params is None:
            session_id = "unknown"
            if self._cached_todo_list and self._cached_todo_list.session:
                session_id = self._cached_todo_list.session.session_id
            self._cached_file_params = FileParameterTable(session_id=session_id)
        
        return self._cached_file_params
    
    def _load_file_params(self) -> None:
        """Load file parameter table from JSON file"""
        try:
            if self._is_remote and self.opensandbox_id:
                content = self._read_remote_file_params()
            else:
                if self.file_params_path.exists():
                    content = self.file_params_path.read_text(encoding="utf-8")
                else:
                    return
            
            if content:
                data = json.loads(content)
                self._cached_file_params = FileParameterTable.from_dict(data)
                print(f"  📁 Loaded file parameter table with {len(self._cached_file_params.files)} files")
        
        except Exception as e:
            print(f"  ⚠️ Failed to load file params: {e}")
    
    def _read_remote_file_params(self) -> Optional[str]:
        """Read file params from remote sandbox via CodeAct"""
        try:
            from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
            
            if not is_codeact_available():
                return None
            
            # Use string path for remote sandbox (Unix-style)
            read_code = f'''
import os
import json
path = "{self.file_params_path_str}"
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    print("__CONTENT__:" + content)
else:
    print("__NOT_FOUND__")
'''
            result = execute_code_via_codeact(
                task_description=f"读取文件参数表: {self.file_params_path_str}",
                code_template=read_code,
                sandbox_id=self.opensandbox_id,
                timeout_seconds=30,
                keep_alive=True  # 保持沙盒存活
            )
            
            # CodeActResult 是 dataclass，使用属性访问而不是 .get()
            if result and result.is_success():
                output = str(result.output or "")
                if "__NOT_FOUND__" in output:
                    return None
                if "__CONTENT__:" in output:
                    return output.split("__CONTENT__:")[1].strip()
            
            return None
        
        except Exception as e:
            print(f"  ⚠️ Failed to read remote file params: {e}")
            return None
    
    def add_file_parameter(self, file_param: FileParameter) -> bool:
        """
        Add a file parameter to the table
        
        Args:
            file_param: File parameter to add
        
        Returns:
            bool: True if successful
        """
        table = self.get_file_parameter_table()
        table.add_file(file_param)
        return self._save_file_params()
    
    def add_file_parameters(self, file_params: List[FileParameter]) -> bool:
        """
        Add multiple file parameters to the table
        
        Args:
            file_params: List of file parameters to add
        
        Returns:
            bool: True if successful
        """
        table = self.get_file_parameter_table()
        for fp in file_params:
            table.add_file(fp)
        return self._save_file_params()
    
    def update_file_params_from_task_result(
        self, 
        task_id: str, 
        task_description: str, 
        task_result: Dict[str, Any]
    ) -> bool:
        """
        Extract and add file parameters from a completed task's result
        
        Args:
            task_id: ID of the completed task
            task_description: Description of the task
            task_result: Result dictionary from task execution
        
        Returns:
            bool: True if successful
        """
        file_params = extract_file_info_from_task_result(
            task_id=task_id,
            task_description=task_description,
            task_result=task_result
        )
        
        if file_params:
            print(f"  📁 Extracted {len(file_params)} file(s) from task {task_id}")
            return self.add_file_parameters(file_params)
        
        return True  # No files extracted is still "success"
    
    def _save_file_params(self) -> bool:
        """Save file parameter table to JSON file"""
        if self._cached_file_params is None:
            return True
        
        try:
            content = json.dumps(self._cached_file_params.to_dict(), indent=2, ensure_ascii=False)
            
            if self._is_remote and self.opensandbox_id:
                return self._write_remote_file_params(content)
            else:
                self.sandbox_dir.mkdir(parents=True, exist_ok=True)
                self.file_params_path.write_text(content, encoding="utf-8")
                print(f"  📁 Saved file parameter table to: {self.file_params_path}")
                return True
        
        except Exception as e:
            print(f"  ⚠️ Failed to save file params: {e}")
            return False
    
    def _write_remote_file_params(self, content: str) -> bool:
        """Write file params to remote sandbox via CodeAct"""
        try:
            from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
            
            if not is_codeact_available():
                return False
            
            # Escape content for JSON
            escaped_content = content.replace('\\', '\\\\').replace('"""', '\\"\\"\\')
            
            # Use string paths for remote sandbox (Unix-style)
            write_code = f'''
import os
import json

os.makedirs("{self.sandbox_dir_str}", exist_ok=True)
path = "{self.file_params_path_str}"
content = """{escaped_content}"""

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"__WRITTEN__:{{path}}")
'''
            result = execute_code_via_codeact(
                task_description=f"保存文件参数表: {self.file_params_path_str}",
                code_template=write_code,
                sandbox_id=self.opensandbox_id,
                timeout_seconds=30,
                keep_alive=True  # 保持沙盒存活
            )
            
            # CodeActResult 是 dataclass，使用属性访问而不是 .get()
            if result and result.is_success():
                output = str(result.output or "")
                if "__WRITTEN__:" in output:
                    return True
            
            return False
        
        except Exception as e:
            print(f"  ⚠️ Failed to write remote file params: {e}")
            return False
    
    # ===================== Markdown Rendering with File Params =====================
    
    def _render_markdown(self, todo_list: TodoList) -> str:
        """Render TodoList to markdown format (includes file parameter table)"""
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
        
        # Add File Parameter Table section (P2)
        if todo_list.file_parameter_table and todo_list.file_parameter_table.files:
            lines.append(todo_list.file_parameter_table.to_markdown())
            lines.append("")
        
        lines.append("## Tasks")
        lines.append("")
        
        for i, task in enumerate(todo_list.tasks, 1):
            lines.append(f"### Task {i}: {task.description}")
            lines.append(f"- id: {task.id}")
            lines.append(f"- type: {task.type.value if hasattr(task.type, 'value') else task.type}")
            lines.append(f"- status: {task.status.value if hasattr(task.status, 'value') else task.status}")
            lines.append(f"- priority: {task.priority}")
            lines.append(f"- description: {task.description}")
            
            if task.dependencies:
                lines.append(f"- dependencies: {task.dependencies}")
            
            # P2: Parameters are now semantic hints, not actual values
            if task.parameters:
                lines.append("- parameters:")
                for key, value in task.parameters.items():
                    # Mark as semantic hint
                    lines.append(f"    {key}: {value}  # (semantic hint)")
            
            if task.result:
                lines.append(f"- result: {json.dumps(task.result)}")
            elif task.status == TodoTaskStatus.COMPLETED:
                lines.append("- result: {}")
            else:
                lines.append("- result: null")
            
            if task.error:
                lines.append(f"- error: {task.error}")
            else:
                lines.append("- error: null")
            
            if task.started_at:
                lines.append(f"- started_at: {task.started_at}")
            if task.completed_at:
                lines.append(f"- completed_at: {task.completed_at}")
            
            lines.append("")
        
        return "\n".join(lines)


# ===================== Code Generation Helpers =====================

def generate_code_to_read_todo_list(sandbox_dir: str) -> str:
    """
    Generate Python code to read todo-list.md from sandbox
    
    This code will be executed in the sandbox environment.
    """
    return f'''
import os
import json

TODO_PATH = "{sandbox_dir}/todo-list.md"

if not os.path.exists(TODO_PATH):
    print(json.dumps({{"error": "todo-list.md not found", "path": TODO_PATH}}))
else:
    with open(TODO_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    print(json.dumps({{"content": content, "path": TODO_PATH}}))
'''


def generate_code_to_update_todo_list(sandbox_dir: str, content: str) -> str:
    """
    Generate Python code to update todo-list.md in sandbox
    
    Args:
        sandbox_dir: Sandbox directory path
        content: New markdown content
    """
    # Use base64 encoding to avoid escaping issues
    import base64
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
    
    return f'''
import os
import json
import base64

TODO_PATH = "{sandbox_dir}/todo-list.md"
CONTENT_B64 = "{encoded_content}"
CONTENT = base64.b64decode(CONTENT_B64).decode('utf-8')

try:
    os.makedirs(os.path.dirname(TODO_PATH), exist_ok=True)
    with open(TODO_PATH, "w", encoding="utf-8") as f:
        f.write(CONTENT)
    print(json.dumps({{"success": True, "path": TODO_PATH}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
'''

