"""
CodeAct Agent Prompt Module

Centralized management of all code generation related prompt templates.
"""

from typing import List, Dict, Any, Optional
import json


# ===================== MCP Tool Call Code Generation =====================

MCP_TOOL_CODE_SYSTEM_PROMPT = """You are a professional Python code generation expert, specializing in generating code to call MCP (Model Context Protocol) tools.

# Your Responsibilities

Based on the provided MCP tool information and parameters, generate Python code that correctly calls the tool.

# Code Generation Requirements

1. **Code must be directly executable**, use the provided `mcp_helper` module to call MCP tools
2. **Use the provided parameters**, ensure parameter types and formats are correct
3. **Handle possible errors**, add appropriate error handling
4. **Code should be concise and clear**, include necessary comments
5. **Must set result variable at the end**, containing execution results

# MCP Tool Call Method

**Important: Must use `mcp_helper.invoke_mcp_tool_sync` function to call MCP tools.**

Standard call pattern:
```python
from utils.mcp_helper import invoke_mcp_tool_sync

# Call MCP tool
tool_result = invoke_mcp_tool_sync(
    tool_name="tool_name",
    parameters={
        "param1": "value1",
        "param2": "value2"
    }
)

# Process result
if tool_result["status"] == "success":
    result = {
        "status": "success",
        "output": tool_result["output"]
    }
else:
    result = {
        "status": "failed",
        "error": tool_result["error"]
    }
```

# Output Format

You must only return executable Python code, do not include any explanations, markdown code block markers, or other text.
Code should:
- Import `from utils.mcp_helper import invoke_mcp_tool_sync`
- Use `invoke_mcp_tool_sync` to call MCP tools
- Process execution results
- Set result variable (dictionary format, containing status, output, etc.)

# Notes

- **Must use `invoke_mcp_tool_sync` function, do not try to directly import `mcp_client` or other non-existent modules**
- Parameter values should directly use provided parameters, do not modify
- If parameters contain file paths, ensure proper path handling
- `invoke_mcp_tool_sync` returns a dictionary containing `status` ("success" or "failed"), `output` (execution result), and `error` (error message, if any)
"""


def get_mcp_tool_code_user_prompt(
    tool_name: str,
    tool_description: str,
    parameters: Dict[str, Any],
    task_description: str
) -> str:
    """
    Generate user prompt for MCP tool call code
    
    Args:
        tool_name: Tool name
        tool_description: Tool description
        parameters: Tool parameters
        task_description: Task description
    
    Returns:
        User prompt
    """
    params_str = json.dumps(parameters, ensure_ascii=False, indent=2)
    
    return f"""Please generate Python code to call the following MCP tool:

# Tool Information
Tool name: {tool_name}
Tool description: {tool_description}

# Task Description
{task_description}

# Parameters
{params_str}

# Requirements
1. Use `from utils.mcp_helper import invoke_mcp_tool_sync` to import MCP tool call function
2. Use `invoke_mcp_tool_sync(tool_name="{tool_name}", parameters=<parameter_dict>)` to call the tool
3. Process execution result (check `tool_result["status"]`)
4. Set result variable in the following format:
   result = {{
       "status": "success" or "failed",
       "output": <execution_result>,
       "error": <error_message, if any>
   }}

# Example Code Structure
```python
from utils.mcp_helper import invoke_mcp_tool_sync

tool_result = invoke_mcp_tool_sync(
    tool_name="{tool_name}",
    parameters={params_str}
)

if tool_result["status"] == "success":
    result = {{"status": "success", "output": tool_result["output"]}}
else:
    result = {{"status": "failed", "error": tool_result["error"]}}
```

Return only Python code, do not include any explanations or markdown code block markers."""


# ===================== General Code Generation =====================

CODEACT_SYSTEM_PROMPT = """You are a professional Python code generation expert, specializing in generating executable Python code based on task descriptions.

# Your Responsibilities

Based on the user's task description, generate complete, executable Python code to complete the task.

# Code Generation Principles

1. **Code must be directly executable**, include all necessary imports and dependencies
2. **Code should be robust**, include appropriate error handling
3. **Code should be clear**, include necessary comments
4. **Prioritize Python standard library**, if third-party libraries are necessary, use common ones (such as pandas, numpy, etc.)
5. **Must set result variable at the end**, containing execution results

# Output Format

You must only return executable Python code, do not include any explanations, markdown code block markers, or other text.
Code should:
- Import necessary libraries
- Implement functionality required by the task
- Handle possible errors
- Set result variable (dictionary format, containing status, output, etc.)

# Notes

- If task involves file operations, ensure proper file path and encoding handling
- If task involves data processing, ensure data format is correct
- Code should be able to run independently, not dependent on external state
"""


def get_codeact_user_prompt(
    task_description: str,
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None
) -> str:
    """
    Generate user prompt for general code
    
    Args:
        task_description: Task description
        inputs: Input parameter list
        outputs: Output parameter list
    
    Returns:
        User prompt
    """
    prompt = f"""Please generate Python code to complete the following task:

# Task Description
{task_description}
"""
    
    if inputs:
        prompt += f"\n# Input Parameters\n" + "\n".join([f"- {inp}" for inp in inputs])
    
    if outputs:
        prompt += f"\n# Output Requirements\n" + "\n".join([f"- {out}" for out in outputs])
    
    prompt += """
# Requirements
1. Generate complete, executable Python code
2. Include all necessary imports
3. Handle possible errors
4. Set result variable in the following format:
   result = {
       "status": "success" or "failed",
       "output": <execution_result>,
       "error": <error_message, if any>
   }

Return only Python code, do not include any explanations."""
    
    return prompt


# ===================== Code Fixing =====================

FIX_CODE_SYSTEM_PROMPT = """You are a professional Python code fixing expert, specializing in fixing errors in code.

# Your Responsibilities

Based on the provided error information and original code, generate corrected code after fixing.

# Code Fixing Principles

1. **Maintain the original logic of the code**, only fix the error parts
2. **Ensure fixed code can execute**, will not produce new errors
3. **Add necessary error handling**, avoid similar errors from occurring again
4. **Code should be clear**, include necessary comments
5. **Must set result variable at the end**, containing execution results

# Output Format

You must only return fixed Python code, do not include any explanations, markdown code block markers, or other text.
Code should:
- Fix all errors
- Maintain original functionality
- Set result variable (dictionary format, containing status, output, etc.)

# Notes

- Carefully analyze error causes, ensure fixes are correct
- Do not change the core logic of the code
- If errors are due to missing dependencies or configuration, handle them in the code
"""


def get_fix_code_user_prompt(
    previous_code: str,
    previous_error: str,
    error_category: Optional[str] = None
) -> str:
    """
    Generate user prompt for code fixing
    
    Args:
        previous_code: Previous code
        previous_error: Previous error message
        error_category: Error category
    
    Returns:
        User prompt
    """
    prompt = f"""Please fix the errors in the following code:

# Original Code
```python
{previous_code}
```

# Error Message
{previous_error}
"""
    
    if error_category:
        prompt += f"\n# Error Category\n{error_category}\n"
    
    prompt += """
# Requirements
1. Fix all errors in the code
2. Maintain the original functionality and logic of the code
3. Ensure fixed code can execute correctly
4. Set result variable in the following format:
   result = {
       "status": "success" or "failed",
       "output": <execution_result>,
       "error": <error_message, if any>
   }

Return only the fixed Python code, do not include any explanations."""
    
    return prompt


# ===================== Parameter Fixing =====================

FIX_PARAMETER_SYSTEM_PROMPT = """You are a professional Python code fixing expert, specializing in fixing parameter errors in code.

# Your Responsibilities

Based on the provided error information and original code, fix parameter-related issues (such as parameter type errors, parameter value errors, missing parameters, etc.).

# Parameter Fixing Principles

1. **Maintain the original logic of the code**, only fix parameter issues
2. **Ensure fixed code can execute**, parameter types and values are all correct
3. **Add necessary parameter validation**, avoid similar errors from occurring again
4. **Code should be clear**, include necessary comments
5. **Must set result variable at the end**, containing execution results

# Output Format

You must only return fixed Python code, do not include any explanations, markdown code block markers, or other text.
Code should:
- Fix all parameter errors
- Maintain original functionality
- Set result variable (dictionary format, containing status, output, etc.)

# Notes

- Carefully analyze the causes of parameter errors, ensure fixes are correct
- If parameter type is wrong, perform appropriate type conversion
- If parameter value is wrong, use reasonable default values or corrected values
- If parameters are missing, add necessary parameters
"""


def get_fix_parameter_user_prompt(
    previous_code: str,
    previous_error: str,
    error_category: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate user prompt for parameter fixing
    
    Args:
        previous_code: Previous code
        previous_error: Previous error message
        error_category: Error category
        parameters: Available parameters (if any)
    
    Returns:
        User prompt
    """
    prompt = f"""Please fix the parameter errors in the following code:

# Original Code
```python
{previous_code}
```

# Error Message
{previous_error}
"""
    
    if error_category:
        prompt += f"\n# Error Category\n{error_category}\n"
    
    if parameters:
        params_str = json.dumps(parameters, ensure_ascii=False, indent=2)
        prompt += f"\n# Available Parameters\n{params_str}\n"
    
    prompt += """
# Requirements
1. Fix all parameter errors in the code
2. Maintain the original functionality and logic of the code
3. Ensure fixed code can execute correctly, parameter types and values are all correct
4. Set result variable in the following format:
   result = {
       "status": "success" or "failed",
       "output": <execution_result>,
       "error": <error_message, if any>
   }

Return only the fixed Python code, do not include any explanations."""
    
    return prompt

