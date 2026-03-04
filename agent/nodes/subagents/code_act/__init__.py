"""
CodeAct Agent Subgraph Module

Responsible for code generation and execution, including:
1. MCP tool call code generation
2. General code generation
3. Code fixing (fix code errors and parameter errors)
4. Code execution and error handling
5. Todo list driven task execution (new architecture)
6. P2: Dynamic parameter inference using LLM
7. P2: File parameter extraction and management

New Architecture:
- Read todo-list.md from sandbox
- Select next pending task
- P2: Infer parameters using file parameter table
- Generate and execute code
- P2: Extract output files to file parameter table
- Update task status
- Loop until all tasks complete
"""

from .graph import (
    # Core functions
    build_codeact_subgraph,
    codeact_input_mapper,
    codeact_output_mapper,
    # State models
    CodeActState,
    CodeActExecutionMode,
    # New todo management nodes
    read_todo_node,
    select_next_task_node,
    update_todo_node,
    has_pending_tasks,
    # P2: Parameter inference and file extraction nodes
    infer_parameters_node,
    extract_file_params_node
)

from .todo_list import (
    # Todo models
    TodoTask,
    TodoTaskStatus,
    TodoTaskType,
    TodoList,
    TodoListSession,
    # Todo manager
    TodoListManager,
    # Code generation helpers
    generate_code_to_read_todo_list,
    generate_code_to_update_todo_list
)

# P2: File parameter table
from .file_param_table import (
    FileParameter,
    FileParameterTable,
    FileSource,
    get_parameter_inference_prompt,
    create_file_param_from_user_input,
    create_file_param_from_task_output,
    extract_file_info_from_task_result
)

__all__ = [
    # Core functions
    "build_codeact_subgraph",
    "codeact_input_mapper",
    "codeact_output_mapper",
    # State models
    "CodeActState",
    "CodeActExecutionMode",
    # New todo management nodes
    "read_todo_node",
    "select_next_task_node",
    "update_todo_node",
    "has_pending_tasks",
    # P2: Parameter inference and file extraction nodes
    "infer_parameters_node",
    "extract_file_params_node",
    # Todo models
    "TodoTask",
    "TodoTaskStatus",
    "TodoTaskType",
    "TodoList",
    "TodoListSession",
    # Todo manager
    "TodoListManager",
    # Code generation helpers
    "generate_code_to_read_todo_list",
    "generate_code_to_update_todo_list",
    # P2: File parameter table
    "FileParameter",
    "FileParameterTable",
    "FileSource",
    "get_parameter_inference_prompt",
    "create_file_param_from_user_input",
    "create_file_param_from_task_output",
    "extract_file_info_from_task_result"
]

