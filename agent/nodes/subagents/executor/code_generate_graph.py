from typing import Dict, Any
import os
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import json
from utils.code_cache_manager import CodeCacheManager
from nodes.subagents.executor.state import StandardTask

# CodeAct subgraph state model
class CodeActState(BaseModel):
    task: StandardTask = Field(description="Standardized task to execute")
    mcp_tool_config: Dict[str, Any] = Field(default_factory=dict, description="MCP tool configuration")
    execution_result: Dict[str, Any] = Field(default_factory=dict, description="Code execution result")
    cached: bool = Field(default=False, description="Whether valid code has been cached")

# Node 1: Generate code (prioritize MCP tool call code)
def generate_code_node(state: CodeActState) -> CodeActState:
    """Generate code: prioritize calling MCP tools, if no tools then generate general analysis code"""
    task = state.task
    mcp_tools = state.mcp_tool_config.get("mcp_tools", [])
    code = ""
    code_description = ""
    related_mcp_tool = None

    # 1. Prioritize matching required MCP tools, generate call code
    for required_tool in task.required_mcp_tools:
        mcp_tool = next((t for t in mcp_tools if t["tool_name"] == required_tool), None)
        if mcp_tool:
            related_mcp_tool = required_tool
            code_description = f"Call MCP tool {required_tool} to complete {task.task_type} task"
            # Reuse example code from configuration, replace core parameters
            code = mcp_tool["example_code"]
            for param_name, param_value in task.core_params.items():
                if isinstance(param_value, str):
                    code = code.replace(f"{param_name}=\"\"", f"{param_name}=\"{param_value}\"")
                else:
                    code = code.replace(f"{param_name}=None", f"{param_name}={param_value}")
            break

    # 2. When no MCP tools, generate general analysis code (simulate LLM generation, can actually call Claude/GPT)
    if not code:
        code_description = f"Generate general code to complete {task.task_type} task"
        code = f"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Core parameters: {task.core_params}
def process_task():
    print(\"Executing general analysis task: {task.task_content}\")
    return \"General analysis task execution completed\"

result = process_task()
print(result)
""".strip()

    # 3. Save code to state (for subsequent execution + caching)
    state.execution_result["generated_code"] = code
    state.execution_result["code_description"] = code
    state.execution_result["related_mcp_tool"] = related_mcp_tool
    return state

# Node 2: Execute code (safe sandbox execution, simplified here)
def execute_code_node(state: CodeActState) -> CodeActState:
    """Execute generated code, get execution result"""
    code = state.execution_result.get("generated_code", "")
    try:
        if os.getenv("ALLOW_UNSAFE_CODEACT", "false").lower() != "true":
            state.execution_result["status"] = "failed"
            state.execution_result["error"] = "Unsafe exec is disabled (set ALLOW_UNSAFE_CODEACT=true to override)"
            return state
        # Safely execute code (actual scenario needs sandbox environment, such as execjs, pyodide)
        local_namespace = {}
        exec(code, globals(), local_namespace)
        # Extract execution result (default extract result variable)
        execution_output = local_namespace.get("result", "Execution successful, no return result")
        state.execution_result["status"] = "success"
        state.execution_result["output"] = execution_output
    except Exception as e:
        state.execution_result["status"] = "failed"
        state.execution_result["error"] = str(e)
    return state

# Node 3: Cache valid code (only cache successfully executed code)
def cache_valid_code_node(state: CodeActState) -> CodeActState:
    """Cache successfully executed code for subsequent reuse"""
    if state.execution_result.get("status") != "success":
        return state
    
    task = state.task
    code = state.execution_result.get("generated_code", "")
    code_description = state.execution_result.get("code_description", "")
    related_mcp_tool = state.execution_result.get("related_mcp_tool")

    # Call cache tool to add cache
    CodeCacheManager.add_cached_code(
        task_type=task.task_type,
        core_params=task.core_params,
        executable_code=code,
        code_description=code_description,
        related_mcp_tool=related_mcp_tool,
        cache_path=task.cache_path
    )
    state.cached = True
    return state

# Build CodeAct subgraph
def build_codeact_graph():
    graph = StateGraph(CodeActState)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute_code", execute_code_node)
    graph.add_node("cache_valid_code", cache_valid_code_node)

    # Define flow rules
    graph.add_edge(START, "generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", "cache_valid_code")
    graph.add_edge("cache_valid_code", END)

    return graph.compile()