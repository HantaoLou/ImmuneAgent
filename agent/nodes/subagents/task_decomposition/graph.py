"""
Task Decomposition Agent Subgraph

Responsible for decomposing complex tasks or execution plans into structured subtasks, identifying dependencies and parallel task groups.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import sys
import json
import re
import os
from pathlib import Path

from .prompt import (
    TASK_DECOMPOSITION_SYSTEM_PROMPT, 
    get_task_decomposition_user_prompt,
    PARALLEL_INFERENCE_SYSTEM_PROMPT,
    get_parallel_inference_user_prompt,
    COARSE_DECOMPOSITION_SYSTEM_PROMPT,
    get_coarse_decomposition_user_prompt
)
from .tool_categorizer import load_service_list, get_tools_by_service_ids, get_service_summary

# Import main graph state (for state mapping)
# Add agent directory to path (support importing from subgraph directory)
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, SubTask, ParallelTaskGroup, UserTaskType, TaskStatus

# Parameter inference result types
from enum import Enum

class ParameterSourceType(str, Enum):
    """Parameter source type"""
    DETERMINED = "determined"  # Determined parameter value
    FROM_TASK = "from_task"  # Determined by a task's result
    USER_REQUIRED = "user_required"  # Requires user input


class ParameterInferenceResult(BaseModel):
    """Parameter inference result"""
    param_name: str = Field(description="Parameter name")
    source_type: ParameterSourceType = Field(description="Parameter source type")
    value: Optional[Any] = Field(default=None, description="Parameter value (if source_type is determined)")
    source_task_id: Optional[str] = Field(default=None, description="Source task ID (if source_type is from_task)")
    source_output_key: Optional[str] = Field(default=None, description="Source task's output key (if source_type is from_task)")
    user_prompt: Optional[str] = Field(default=None, description="User prompt information (if source_type is user_required)")
    reason: Optional[str] = Field(default=None, description="Inference reason or explanation")


class TaskParameterInference(BaseModel):
    """Task parameter inference result"""
    task_id: str = Field(description="Task ID")
    tool_name: Optional[str] = Field(default=None, description="Tool name")
    parameters: Dict[str, ParameterInferenceResult] = Field(default_factory=dict, description="Parameter inference results")
    inference_summary: Optional[str] = Field(default=None, description="Inference summary")

# LLM-related imports (using public LLM factory)
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_advanced_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_advanced_llm = None
    HumanMessage = None
    SystemMessage = None
    print("Warning: langchain-related libraries not installed, task decomposition functionality will be unavailable")


# ---------------------- Task Decomposition State Model ----------------------
class TaskDecompositionState(BaseModel):
    """Task Decomposition Agent subgraph state"""
    user_input: str = Field(description="User's original input")
    execution_plan: Optional[str] = Field(default=None, description="User-provided execution plan (if any)")
    available_tools: List[Dict[str, Any]] = Field(default_factory=list, description="List of all available tools (MCP tools, Skills, persistent tools)")
    # Stage 0: Coarse decomposition results
    required_service_ids: List[str] = Field(default_factory=list, description="List of required service_ids determined by coarse decomposition")
    filtered_tools: List[Dict[str, Any]] = Field(default_factory=list, description="List of tools filtered by service_id")
    # Stage 1: Fine decomposition results (serialized task list, includes dependencies but not parallel relationships)
    raw_tasks: List[Dict[str, Any]] = Field(default_factory=list, description="Raw task list from fine decomposition (JSON format)")
    # Stage 2: Parallel inference results
    subtasks: List[SubTask] = Field(default_factory=list, description="List of decomposed regular subtasks")
    parallel_task_groups: Dict[str, ParallelTaskGroup] = Field(default_factory=dict, description="Parallel task groups")
    decomposition_summary: Optional[str] = Field(default=None, description="Overall description of task decomposition")
    # Stage 3: Parameter inference results
    # NOTE: Parameter inference has been moved to executor subgraph.
    # These fields are kept for compatibility but are no longer populated here.
    parameter_inference_results: Dict[str, TaskParameterInference] = Field(default_factory=dict, description="Task parameter inference results (deprecated, handled in executor)")
    parameter_inference_summary: Optional[str] = Field(default=None, description="Overall description of parameter inference (deprecated, handled in executor)")
    context_extracted_params: Dict[str, Dict[str, Dict[str, Any]]] = Field(default_factory=dict, description="Context-extracted parameters (deprecated, handled in executor)")


# ---------------------- LLM Instantiation (using public LLM factory) ----------------------
def _get_llm():
    """
    Get reasoning model instance (for task decomposition)
    
    Use public LLM factory to create reasoning model, prefer models with good reasoning performance.
    
    Returns:
        LLM instance, returns None if all are unavailable
    """
    if not LLM_AVAILABLE or create_reasoning_advanced_llm is None:
        return None
    
    # Use reasoning model (for task decomposition, slightly lower temperature for more structured output)
    return create_reasoning_advanced_llm(temperature=0.2)


# ---------------------- CodeAct Tool Creation Function =====================
def _create_codeact_tool() -> Dict[str, Any]:
    """
    Create CodeAct tool (virtual MCP tool)
    
    When existing MCP tools cannot support the task, use CodeAct to write and execute code.
    The actual code writing is done by subsequent subgraphs, here it's just used as a tool identifier.
    
    Returns:
        CodeAct tool dictionary
    """
    return {
        "name": "codeact",
        "service": "codeact",
        "description": "Code execution tool. When existing MCP tools cannot complete complex tasks, use Python code writing and execution to accomplish them. Supports file operations, data analysis, visualization, API calls, and other types of tasks. Code writing and execution are completed by subsequent subgraphs.",
        "tool": [{
            "tool_name": "codeact",
            "description": "Code execution tool for writing and executing Python code to complete complex tasks",
            "parameters": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task to be completed",
                    "required": True
                },
                "required_libraries": {
                    "type": "array",
                    "description": "List of potentially required Python libraries (e.g., numpy, pandas, matplotlib, etc.)",
                    "required": False
                },
                "input_files": {
                    "type": "array",
                    "description": "List of input file paths",
                    "required": False
                },
                "output_files": {
                    "type": "array",
                    "description": "List of output file paths",
                    "required": False
                }
            }
        }]
    }


def _check_and_add_codeact_to_tasks(raw_tasks: List[Dict[str, Any]], available_tools: List[Dict[str, Any]]):
    """
    Check tasks in fine decomposition results, if a task has no matching tool, add codeact tool
    
    Args:
        raw_tasks: Raw task list from fine decomposition
        available_tools: List of available tools
    """
    if not raw_tasks:
        return
    
    # Get names of all available tools
    available_tool_names = set()
    for tool in available_tools:
        tool_name = tool.get("name", "")
        if tool_name:
            available_tool_names.add(tool_name)
        # Also check tool names in the tool field
        if "tool" in tool and isinstance(tool["tool"], list):
            for t in tool["tool"]:
                tool_name = t.get("tool_name", "")
                if tool_name:
                    available_tool_names.add(tool_name)
    
    codeact_added = False
    for task in raw_tasks:
        task_tools = task.get("tools", [])
        if not task_tools:
            # If task has no matching tool, add codeact
            if "codeact" not in available_tool_names:
                task["tools"] = ["codeact"]
                codeact_added = True
                print(f"  ⚠ Task {task.get('task_id', 'N/A')} has no matching tool, added codeact")
        else:
            # Check if all tools matched by the task are in the available tools list
            task_tool_names = [t if isinstance(t, str) else t.get("tool_name", "") if isinstance(t, dict) else "" for t in task_tools]
            matched = any(tool_name in available_tool_names for tool_name in task_tool_names if tool_name)
            if not matched:
                # If matched tools are not in the available list, add codeact
                if "codeact" not in task_tool_names:
                    task["tools"].append("codeact")
                    codeact_added = True
                    print(f"  ⚠ Task {task.get('task_id', 'N/A')} matched tools are unavailable, added codeact")
    
    if codeact_added:
        print(f"  ✓ Added codeact tool as fallback for some tasks")


def _load_tools_params_table() -> Dict[str, Dict[str, Any]]:
    """
    Load tools parameters table
    
    Returns:
        Mapping dictionary from tool name to parameter information
    """
    tools_params_path = agent_dir / "config" / "tools_params_table.json"
    
    if not tools_params_path.exists():
        print(f"⚠ Tools parameters table file does not exist: {tools_params_path}")
        return {}
    
    try:
        with open(tools_params_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # tools_params_table.json is an array, each element is an object
        # Each object's key is the tool name, value is parameter information
        tools_params_map = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for tool_name, params_info in item.items():
                        tools_params_map[tool_name] = params_info
        elif isinstance(data, dict):
            # If it's directly a dictionary, also support
            tools_params_map = data
        
        print(f"✓ Successfully loaded tools parameters table, {len(tools_params_map)} tools in total")
        return tools_params_map
    except Exception as e:
        print(f"⚠ Failed to load tools parameters table: {e}")
        return {}


def _get_llm_for_inference():
    """
    Get LLM instance for parameter inference
    
    Returns:
        LLM instance, returns None if unavailable
    """
    if not LLM_AVAILABLE or create_reasoning_advanced_llm is None:
        return None
    
    # Use reasoning model (for parameter inference, slightly lower temperature for more accurate inference)
    return create_reasoning_advanced_llm(temperature=0.1)


def _match_task_parameters(raw_tasks: List[Dict[str, Any]]):
    """
    Match parameter descriptions for each task (look up by tool name from tools_params_table.json)
    
    Args:
        raw_tasks: Raw task list from fine decomposition
    """
    if not raw_tasks:
        return
    
    # Load tools parameters table
    tools_params_map = _load_tools_params_table()
    if not tools_params_map:
        print("⚠ Tools parameters table is empty, skipping parameter matching")
        return
    
    matched_count = 0
    for task in raw_tasks:
        task_tools = task.get("tools", [])
        if not task_tools:
            continue
        
        # Collect input parameters for all tools
        all_inputs = []
        all_outputs = []
        
        for tool_item in task_tools:
            # Tool may be a string or dictionary
            tool_name = None
            if isinstance(tool_item, str):
                tool_name = tool_item
            elif isinstance(tool_item, dict):
                tool_name = tool_item.get("tool_name") or tool_item.get("name", "")
            
            if not tool_name:
                continue
            
            # Look up in tools parameters table (supports multiple tool name formats)
            tool_params = None
            
            # Method 1: Direct match tool name
            tool_params = tools_params_map.get(tool_name)
            
            # Method 2: If direct match fails, try matching service_tool_name format
            if not tool_params:
                # Tool names in tools_params_table.json may be in "service_tool_name" format (e.g., "airr_search_airr_repertoires")
                # Try matching the complete service_tool_name
                for key in tools_params_map.keys():
                    # Check if it's service_tool_name format and tool_name part matches
                    if "_" in key:
                        parts = key.split("_", 1)
                        if len(parts) > 1 and parts[1] == tool_name:
                            tool_params = tools_params_map.get(key)
                            break
                    # Or exact match
                    if key == tool_name:
                        tool_params = tools_params_map.get(key)
                        break
            
            # Method 3: If tool name itself is in service_tool_name format, try direct match
            if not tool_params and "_" in tool_name:
                # Tool name may be in "service_tool_name" format, try direct match
                tool_params = tools_params_map.get(tool_name)
            
            if tool_params:
                # Extract input parameters
                input_params = tool_params.get("input_params", [])
                for param in input_params:
                    param_name = param.get("name", "")
                    param_type = param.get("type", "")
                    if param_name:
                        param_desc = f"{param_name}: {param_type}"
                        if param_desc not in all_inputs:
                            all_inputs.append(param_desc)
                
                # Output parameters: Currently tools_params_table.json has no explicit output parameters
                # Can infer from tool type, or leave empty for subsequent nodes to handle
                # Temporarily not adding output parameters here as there's no explicit information in the table
                matched_count += 1
        
        # Update task's inputs and outputs
        if all_inputs:
            # If task already has inputs, merge and deduplicate
            existing_inputs = task.get("inputs", [])
            if isinstance(existing_inputs, list):
                # Merge and deduplicate
                combined_inputs = list(set(existing_inputs + all_inputs))
            else:
                combined_inputs = all_inputs
            task["inputs"] = combined_inputs
        
        # outputs temporarily not updated, as tools_params_table.json has no output parameter information
        # If task already has outputs, keep them
        if "outputs" not in task:
            task["outputs"] = []
    
    if matched_count > 0:
        print(f"✓ Matched parameter descriptions for {matched_count} tools")


# ---------------------- Node 0: Coarse Decomposition Node - Determine Required Tool Types ----------------------
def coarse_decomposition_node(state: TaskDecompositionState) -> TaskDecompositionState:
    """
    Stage 0: Coarse decomposition node
    
    Do not pass tool list, only determine the main tool types required by the task based on user input and execution plan.
    This avoids prompt length issues caused by overly long tool lists.
    
    Args:
        state: Task Decomposition subgraph state
    
    Returns:
        Updated state (required_service_ids contains list of required service_ids)
    """
    import time
    stage_start = time.time()
    
    user_input = state.user_input
    execution_plan = state.execution_plan
    
    print(f"  [粗分解] 开始阶段 0: 粗分解...")
    
    # Load service list (for coarse decomposition prompt)
    load_start = time.time()
    service_list = load_service_list()
    print(f"  [粗分解] 加载服务列表完成 (耗时: {time.time() - load_start:.2f}秒, 服务数: {len(service_list)})")
    
    # Use LLM for coarse decomposition (pass service_list, not tool list)
    llm = _get_llm()
    if llm is not None:
        print(f"  [粗分解] 调用 LLM 进行粗分解...")
        llm_start = time.time()
        result = _coarse_decompose_with_llm(user_input, execution_plan, service_list, llm)
        llm_elapsed = time.time() - llm_start
        print(f"  [粗分解] LLM 调用完成 (耗时: {llm_elapsed:.2f}秒)")
        
        if result:
            # Extract required service_ids
            service_ids = result.get("required_service_ids", [])
            state.required_service_ids = service_ids
            print(f"  ✓ 粗分解完成 (使用 LLM)")
            print(f"    所需服务: {', '.join(service_ids)}")
        else:
            # Fallback when LLM fails: use all services
            all_service_ids = [s.get("service_id", "") for s in service_list if s.get("service_id")]
            state.required_service_ids = all_service_ids
            print(f"  ⚠ 粗分解失败，使用所有服务")
    else:
        # Fallback when LLM is unavailable
        all_service_ids = [s.get("service_id", "") for s in service_list if s.get("service_id")]
        state.required_service_ids = all_service_ids
        print(f"  ⚠ LLM 不可用，使用所有服务")
    
    # Filter tools by service_id
    filter_start = time.time()
    if state.required_service_ids:
        state.filtered_tools = get_tools_by_service_ids(state.available_tools, state.required_service_ids)
        filter_elapsed = time.time() - filter_start
        print(f"  [粗分解] 工具过滤完成 (耗时: {filter_elapsed:.2f}秒)")
        print(f"    过滤后工具数: {len(state.filtered_tools)} / {len(state.available_tools)}")
        
        # Fallback strategy: if no tools matched, add codeact service
        if len(state.filtered_tools) == 0:
            print(f"  ⚠ 没有匹配的 MCP 工具，添加 codeact 服务作为后备")
            if "codeact" not in state.required_service_ids:
                state.required_service_ids.append("codeact")
            # Add codeact tool
            codeact_tool = _create_codeact_tool()
            state.filtered_tools = [codeact_tool]
    else:
        # If no service specified, use all tools
        state.filtered_tools = state.available_tools
    
    stage_elapsed = time.time() - stage_start
    print(f"  ✓ 阶段 0 完成 (总耗时: {stage_elapsed:.2f}秒)\n")
    
    return state


# ---------------------- Node 1: Fine Decomposition Node - Detailed Decomposition Based on Filtered Tools ----------------------
def fine_decomposition_node(state: TaskDecompositionState) -> TaskDecompositionState:
    """
    Stage 1: Fine decomposition node
    
    Based on tools filtered by coarse decomposition, perform detailed task decomposition and tool matching.
    Focus only on task decomposition, tool matching, and dependency identification, not parallel execution.
    
    Args:
        state: Task Decomposition subgraph state
    
    Returns:
        Updated state (raw_tasks contains fine decomposition results)
    """
    user_input = state.user_input
    execution_plan = state.execution_plan
    filtered_tools = state.filtered_tools  # Use filtered tools
    
    # Fallback strategy: if filtered tools are empty, add codeact tool
    if len(filtered_tools) == 0:
        print(f"⚠ Fine decomposition stage matched no tools, adding codeact tool as fallback")
        codeact_tool = _create_codeact_tool()
        filtered_tools = [codeact_tool]
        state.filtered_tools = filtered_tools
        # Ensure codeact is in required_service_ids
        if "codeact" not in state.required_service_ids:
            state.required_service_ids.append("codeact")
    
    # Use LLM for fine decomposition (using filtered tools)
    import time
    stage_start = time.time()
    print(f"  [细分解] 开始阶段 1: 细分解...")
    print(f"  [细分解] 可用工具数: {len(filtered_tools)}")
    
    llm = _get_llm()
    if llm is not None:
        print(f"  [细分解] 调用 LLM 进行细分解...")
        llm_start = time.time()
        result = _decompose_task_with_llm(user_input, execution_plan, filtered_tools, llm)
        llm_elapsed = time.time() - llm_start
        print(f"  [细分解] LLM 调用完成 (耗时: {llm_elapsed:.2f}秒)")
        
        if result:
            # Fine decomposition: store raw task list (JSON format)
            state.raw_tasks = result.get("tasks", [])
            state.decomposition_summary = result.get("decomposition_summary", "")
            print(f"  ✓ 细分解完成 (使用 LLM)")
            print(f"    序列化任务数: {len(state.raw_tasks)}")
            
            # Fallback strategy: check if any tasks in fine decomposition results have no matching tools
            check_start = time.time()
            _check_and_add_codeact_to_tasks(state.raw_tasks, filtered_tools)
            print(f"  [细分解] 工具检查完成 (耗时: {time.time() - check_start:.2f}秒)")
            
            # Match parameter descriptions for each task
            param_start = time.time()
            _match_task_parameters(state.raw_tasks)
            print(f"  [细分解] 参数匹配完成 (耗时: {time.time() - param_start:.2f}秒)")
        else:
            # Fallback when LLM fails
            print(f"  ⚠ LLM 调用失败，使用后备方案")
            fallback_start = time.time()
            fallback_result = _decompose_task_fallback(user_input, execution_plan)
            state.raw_tasks = [task.model_dump() for task in fallback_result.get("subtasks", [])]
            state.decomposition_summary = fallback_result.get("decomposition_summary", "Using fallback for task decomposition")
            print(f"  [细分解] 后备方案完成 (耗时: {time.time() - fallback_start:.2f}秒)")
            # Also match parameter descriptions for fallback tasks
            _match_task_parameters(state.raw_tasks)
    else:
        # Fallback when LLM is unavailable
        print(f"  ⚠ LLM 不可用，使用后备方案")
        fallback_start = time.time()
        fallback_result = _decompose_task_fallback(user_input, execution_plan)
        state.raw_tasks = [task.model_dump() for task in fallback_result.get("subtasks", [])]
        state.decomposition_summary = fallback_result.get("decomposition_summary", "Using fallback for task decomposition")
        print(f"  [细分解] 后备方案完成 (耗时: {time.time() - fallback_start:.2f}秒)")
        # Also match parameter descriptions for fallback tasks
        _match_task_parameters(state.raw_tasks)
    
    stage_elapsed = time.time() - stage_start
    print(f"  ✓ 阶段 1 完成 (总耗时: {stage_elapsed:.2f}秒)\n")
    
    return state


# ---------------------- Node 2: Stage 2 - Parallel Task Inference Node ----------------------
def parallel_inference_node(state: TaskDecompositionState) -> TaskDecompositionState:
    """
    Stage 2: Parallel task inference node
    
    Based on the serialized task list from Stage 1, infer which tasks can be executed in parallel.
    
    Args:
        state: Task Decomposition subgraph state (contains raw_tasks)
    
    Returns:
        Updated state (contains final subtasks and parallel_task_groups)
    """
    import time
    stage_start = time.time()
    
    raw_tasks = state.raw_tasks
    
    if not raw_tasks:
        print("  ⚠ 阶段 1 未生成任务，跳过并行推断")
        return state
    
    print(f"  [并行推断] 开始阶段 2: 并行推断...")
    print(f"  [并行推断] 输入任务数: {len(raw_tasks)}")
    
    # Use LLM for parallel inference
    llm = _get_llm()
    if llm is not None:
        print(f"  [并行推断] 调用 LLM 进行并行推断...")
        llm_start = time.time()
        result = _infer_parallel_tasks_with_llm(raw_tasks, llm)
        llm_elapsed = time.time() - llm_start
        print(f"  [并行推断] LLM 调用完成 (耗时: {llm_elapsed:.2f}秒)")
        
        if result:
            # Convert and store final results
            convert_start = time.time()
            final_result = _validate_and_convert_decomposition(result)
            print(f"  [并行推断] 结果转换完成 (耗时: {time.time() - convert_start:.2f}秒)")
            
            state.subtasks = final_result.get("subtasks", [])
            state.parallel_task_groups = final_result.get("parallel_task_groups", {})
            if result.get("parallel_inference_summary"):
                if state.decomposition_summary:
                    state.decomposition_summary += f"\n\nParallel inference explanation: {result.get('parallel_inference_summary')}"
                else:
                    state.decomposition_summary = f"Parallel inference explanation: {result.get('parallel_inference_summary')}"
            print(f"  ✓ 阶段 2 并行推断完成 (使用 LLM)")
            print(f"    串行任务数: {len(state.subtasks)}")
            print(f"    并行任务组数: {len(state.parallel_task_groups)}")
        else:
            # Fallback when LLM fails: all tasks execute serially
            print("  ⚠ 并行推断失败，所有任务将串行执行")
            final_result = _validate_and_convert_decomposition({"tasks": raw_tasks, "parallel_task_groups": []})
            state.subtasks = final_result.get("subtasks", [])
            state.parallel_task_groups = {}
    else:
        # Fallback when LLM is unavailable: all tasks execute serially
        print("  ⚠ LLM 不可用，所有任务将串行执行")
        final_result = _validate_and_convert_decomposition({"tasks": raw_tasks, "parallel_task_groups": []})
        state.subtasks = final_result.get("subtasks", [])
        state.parallel_task_groups = {}
    
    stage_elapsed = time.time() - stage_start
    print(f"  ✓ 阶段 2 完成 (总耗时: {stage_elapsed:.2f}秒)\n")
    
    return state


# ---------------------- Node 3: Stage 3 - Parameter Inference Node ----------------------
def infer_parameters_node(state: TaskDecompositionState) -> TaskDecompositionState:
    """
    Stage 3: Parameter inference node
    
    For each task, infer its tool's parameter values. Supports three result types:
    1. Determined parameter values: Parameters that can be directly inferred from task description
    2. Parameter source: Parameter values determined by a task's execution result
    3. User required: Requires user to provide additional information or files
    
    Args:
        state: Task Decomposition subgraph state (contains decomposed tasks)
    
    Returns:
        Updated state (contains parameter inference results)
    """
    all_tasks = state.subtasks + [
        task for group in state.parallel_task_groups.values()
        for task in group.subtasks
    ]
    
    if not all_tasks:
        print("⚠ No tasks need parameter inference")
        return state
    
    llm = _get_llm_for_inference()
    tools_params_map = _load_tools_params_table()
    
    # Step 1: Extract parameters from context (user input, execution plan, task descriptions)
    # This creates a parameter table for each task/tool combination
    print("  Step 1: Extracting parameters from context (user input, execution plan, task descriptions)...")
    context_extracted_params = {}  # task_id -> tool_name -> param_name -> value
    
    for task in all_tasks:
        task_id = task.task_id
        task_result = task.result if isinstance(task.result, dict) else {}
        tools = task_result.get("tools", [])
        
        if not tools:
            continue
        
        context_extracted_params[task_id] = {}
        
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
            if not tool_params:
                # Try fuzzy matching (same logic as below)
                for key in tools_params_map.keys():
                    if "_" in key:
                        parts = key.split("_", 1)
                        if len(parts) > 1 and (parts[1] == tool_name or parts[0] == tool_name):
                            tool_params = tools_params_map.get(key)
                            if tool_params:
                                break
                    if key == tool_name:
                        tool_params = tools_params_map.get(key)
                        if tool_params:
                            break
                
                if not tool_params:
                    tool_name_lower = tool_name.lower()
                    for key in tools_params_map.keys():
                        key_lower = key.lower()
                        if tool_name_lower in key_lower or key_lower in tool_name_lower:
                            tool_params = tools_params_map.get(key)
                            if tool_params:
                                break
            
            if tool_params:
                # Extract parameters from context using semantic analysis
                extracted = _extract_parameters_from_context(
                    state.user_input,
                    state.execution_plan,
                    task.content,
                    tool_name,
                    tool_params.get("input_params", []),
                    llm
                )
                if extracted:
                    context_extracted_params[task_id][tool_name] = extracted
    
    extracted_count = sum(len(tools) for tools in context_extracted_params.values())
    print(f"  ✓ Extracted parameters from context for {extracted_count} tool(s)")
    
    # Print extracted parameters table
    if context_extracted_params:
        print(f"\n  【从上下文抽取的参数表】")
        print(f"  {'='*70}")
        for task_id, tools_params in context_extracted_params.items():
            print(f"  Task {task_id}:")
            for tool_name, extracted_params in tools_params.items():
                print(f"    工具: {tool_name}")
                if extracted_params:
                    print(f"    抽取的参数:")
                    for param_name, param_value in extracted_params.items():
                        print(f"      - {param_name}: {param_value}")
                else:
                    print(f"    未抽取到参数")
            print()
        print(f"  {'='*70}")
    
    # Store extracted parameters in state for logging/debugging
    state.context_extracted_params = context_extracted_params
    
    # Step 2: Infer parameters for each task using the extracted parameter table
    inference_results = {}
    
    for task in all_tasks:
        task_id = task.task_id
        task_result = task.result if isinstance(task.result, dict) else {}
        tools = task_result.get("tools", [])
        inputs = task_result.get("inputs", [])
        
        if not tools:
            # No tool task, skip parameter inference
            continue
        
        # Collect parameter inference results for all tools
        task_params = {}
        task_tool_names = []
        
        for tool_item in tools:
            tool_name = None
            if isinstance(tool_item, str):
                tool_name = tool_item
            elif isinstance(tool_item, dict):
                tool_name = tool_item.get("tool_name") or tool_item.get("name", "")
            
            if not tool_name:
                continue
            
            if tool_name not in task_tool_names:
                task_tool_names.append(tool_name)
            
            # Get extracted parameters for this task/tool combination
            extracted_params = context_extracted_params.get(task_id, {}).get(tool_name, {})
            
            # Find tool parameter definition (supports fuzzy matching)
            tool_params = tools_params_map.get(tool_name)
            
            # If direct match fails, try fuzzy matching
            if not tool_params:
                # Method 1: Try matching service_tool_name format (e.g., "af3_alphafold3" or "af3")
                for key in tools_params_map.keys():
                    # Check if it's service_tool_name format and tool_name part matches
                    if "_" in key:
                        parts = key.split("_", 1)
                        if len(parts) > 1 and (parts[1] == tool_name or parts[0] == tool_name):
                            tool_params = tools_params_map.get(key)
                            if tool_params:
                                print(f"  [DEBUG] Tool {tool_name} matched to {key} in parameter table")
                                break
                    # Or exact match
                    if key == tool_name:
                        tool_params = tools_params_map.get(key)
                        if tool_params:
                            break
                
                # Method 2: If still not found, try partial matching (e.g., "alphafold3" matches "af3")
                if not tool_params:
                    tool_name_lower = tool_name.lower()
                    for key in tools_params_map.keys():
                        key_lower = key.lower()
                        # Check if tool_name contains key, or key contains tool_name
                        if tool_name_lower in key_lower or key_lower in tool_name_lower:
                            tool_params = tools_params_map.get(key)
                            if tool_params:
                                print(f"  [DEBUG] Tool {tool_name} found in parameter table via partial match: {key}")
                                break
                
                # Method 3: If still not found, try inferring from inputs field (fallback)
                if not tool_params:
                    print(f"  [WARN] Tool {tool_name} not found in parameter table, using inputs field for inference")
                    for input_param in inputs:
                        if input_param not in task_params:
                            inference_result = _infer_single_parameter(
                                task, tool_name, input_param, None, None, None, None, None,
                                inputs, llm, all_tasks, extracted_params
                            )
                            task_params[input_param] = inference_result
                    continue
            
            # Get parameter definitions from tools parameters table
            input_params = tool_params.get("input_params", [])
            for param in input_params:
                param_name = param.get("name", "")
                param_type = param.get("type", "")
                param_desc = param.get("description", "")
                param_deme = param.get("deme") or param.get("demo", "")  # Get example value
                param_options = param.get("options", [])  # Get enum options
                is_optional = "optional" in param_type.lower() or param_type.startswith("Optional")
                
                if not param_name or param_name in task_params:
                    continue
                
                # Infer single parameter (pass extracted_params for priority checking)
                inference_result = _infer_single_parameter(
                    task, tool_name, param_name, param_type, param_desc, param_deme, param_options, is_optional,
                    inputs, llm, all_tasks, extracted_params
                )
                task_params[param_name] = inference_result
        
        # Create task's parameter inference result
        if task_params:
            task_inference = TaskParameterInference(
                task_id=task_id,
                tool_name=", ".join(task_tool_names) if task_tool_names else None,
                parameters=task_params,
                inference_summary=None
            )
            inference_results[task_id] = task_inference
    
    state.parameter_inference_results = inference_results
    
    # Generate overall summary
    total_tasks = len(inference_results)
    total_params = sum(len(r.parameters) for r in inference_results.values())
    determined_count = sum(
        sum(1 for p in r.parameters.values() if p.source_type == ParameterSourceType.DETERMINED)
        for r in inference_results.values()
    )
    from_task_count = sum(
        sum(1 for p in r.parameters.values() if p.source_type == ParameterSourceType.FROM_TASK)
        for r in inference_results.values()
    )
    user_required_count = sum(
        sum(1 for p in r.parameters.values() if p.source_type == ParameterSourceType.USER_REQUIRED)
        for r in inference_results.values()
    )
    
    state.parameter_inference_summary = (
        f"Parameter inference completed: {total_tasks} tasks, {total_params} parameters. "
        f"Determined: {determined_count}, from tasks: {from_task_count}, user required: {user_required_count}"
    )
    
    print(f"✓ Stage 3 parameter inference completed")
    print(f"  Inferred tasks: {total_tasks}")
    print(f"  Inferred parameters: {total_params} (determined: {determined_count}, from tasks: {from_task_count}, user required: {user_required_count})")
    
    return state


def _extract_parameters_from_context(
    user_input: str,
    execution_plan: Optional[str],
    task_description: str,
    tool_name: str,
    tool_params: List[Dict[str, Any]],
    llm: Optional[Any]
) -> Dict[str, Any]:
    """
    Extract parameter values from user input, execution plan, and task description using semantic analysis
    
    Args:
        user_input: User's original input
        execution_plan: User-provided execution plan (if any)
        task_description: Task description
        tool_name: Tool name
        tool_params: List of tool parameter definitions
        llm: LLM instance for semantic analysis
    
    Returns:
        Dictionary mapping parameter names to extracted values (if found)
    """
    if not llm or not tool_params:
        return {}
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Build parameter information for LLM
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
        
        # Build context information
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

Please carefully analyze the semantic meaning of the context and extract parameter values that match the tool parameters. 
Important principles:
1. **Semantic matching**: Match parameters based on meaning, not just keyword matching. For example:
   - "organism" parameter can match "species", "organism name", "target organism", etc.
   - "sequence" parameter can match "protein sequence", "DNA sequence", "input sequence", etc.
   - "file" or "path" parameters can match file paths, file names, input files mentioned in context
   
1.5. **File path extraction**: **CRITICAL** - When extracting file paths, extract ONLY the actual file path, NOT descriptive text:
   - CORRECT: If context says "raw TCR rds file: /data/benchmark_data/tcr_benckmark/PBMC_vaccine_ECCITE_TCR.rds", extract: "/data/benchmark_data/tcr_benckmark/PBMC_vaccine_ECCITE_TCR.rds"
   - WRONG: Do NOT extract "TCR rds file (/data/benchmark_data/tcr_benckmark/PBMC_vaccine_ECCITE_TCR.rds)" or similar descriptive strings
   - Always extract the bare file path without any surrounding descriptive text, parentheses, or explanations
   
2. **Context understanding**: Understand the context to determine which values correspond to which parameters:
   - If user mentions "analyze human proteins", extract organism="human" or species="human"
   - If user provides a file path, extract it as file/path parameter
   - If user specifies numeric values (like "top 100 results"), extract as max_results=100
   
3. **Input vs Output files**: **CRITICAL** - Distinguish between input files and output files:
   - **Input files**: Files explicitly provided by the user (e.g., "fasta文件: /path/to/file.fasta", "抗体文件: /path/to/file.csv", "抗原文件: /path/to/file.xlsx")
     - These should be extracted as input parameters (e.g., "input_file", "sequences", "antibody_file", "antigen_file")
   - **Output files**: Files that will be generated by the tool (e.g., "output_file_path", "output_file")
     - **DO NOT** extract user-provided input file paths as output file parameters
     - Only extract output file paths if the user explicitly specifies an output file path (e.g., "输出文件: /path/to/output.txt")
     - If user does not specify output file path, **DO NOT** include output file parameters in the result
   
4. **File type distinction**: **CRITICAL** - When extracting file parameters, carefully distinguish between different file types and purposes:
   - **Antibody files** (抗体文件): Files containing antibody sequences/data (e.g., CSV files with antibody information)
     - Match to parameters like "antibody_file", "antibody_csv", "antibody_data"
     - Example: If user says "抗体文件: /path/to/flu-simple.csv", extract antibody_file="/path/to/flu-simple.csv"
   - **Antigen files** (抗原文件): Files containing antigen sequences/data (e.g., XLSX files with antigen information, or FASTA files with antigen sequences)
     - Match to parameters like "antigen_file", "antigen_csv", "antigen_data", "antigen_fasta"
     - Example: If user says "抗原文件: /path/to/flu_bind_variant_seq.xlsx", extract antigen_file="/path/to/flu_bind_variant_seq.xlsx"
   - **Sequence files** (序列文件): Files containing sequence data (e.g., FASTA files)
     - Match to parameters like "sequences", "input_file", "sequence_file", "fasta_file"
     - Example: If user says "fasta文件: /path/to/flu.fasta", extract sequences="/path/to/flu.fasta" or input_file="/path/to/flu.fasta"
   - **IMPORTANT**: Do NOT confuse antibody files with antigen files. They are different and should be extracted to different parameters.
     - If user provides both "抗体文件" and "抗原文件", extract them to "antibody_file" and "antigen_file" respectively
     - Do NOT use the same file path for both antibody_file and antigen_file unless the user explicitly states they are the same file
   
5. **Return format**: Return JSON format with parameter names as keys and extracted values as values.
   Only include parameters that can be clearly extracted from context. If a parameter cannot be determined, do not include it.

Example output:
{{
  "organism": "human",
  "max_results": 100,
  "input_file": "/path/to/file.fasta",
  "antibody_file": "/path/to/antibody.csv",
  "antigen_file": "/path/to/antigen.xlsx"
}}

Return only the JSON object, no additional text."""
        
        system_message_content = """You are a professional parameter extraction expert. Your task is to analyze context information (user input, execution plan, task description) and extract parameter values for tools using semantic understanding.

Key capabilities:
1. **Semantic matching**: Understand parameter meanings and match them with context information semantically, not just by keywords
2. **Context comprehension**: Understand the overall context to determine which values correspond to which parameters
3. **Accurate extraction**: Only extract values that can be clearly determined from context, avoid guessing
3.5. **File path extraction**: **CRITICAL** - Extract ONLY the actual file path from context, NOT descriptive text. For example, if context says "raw TCR rds file: /path/to/file.rds", extract "/path/to/file.rds" (not "TCR rds file (/path/to/file.rds)" or similar descriptive strings)
4. **Input/Output distinction**: **CRITICAL** - Never extract user-provided input file paths as output file parameters. Output files are generated by tools, not provided by users.
5. **File type distinction**: **CRITICAL** - Carefully distinguish between different file types:
   - **Antibody files** (抗体文件): Files containing antibody data - extract to "antibody_file" or similar parameters
   - **Antigen files** (抗原文件): Files containing antigen data - extract to "antigen_file" or similar parameters
   - **Sequence files** (序列文件, fasta文件): Files containing sequence data - extract to "sequences", "input_file", or similar parameters
   - **IMPORTANT**: Do NOT confuse antibody files with antigen files. They are different entities and should be extracted to different parameters.

Important: 
- Focus on semantic meaning rather than exact keyword matching to ensure parameters are correctly extracted and used by tools.
- **DO NOT** extract input file paths (provided by user) as output file parameters (generated by tool).
- **DO NOT** use the same file path for both antibody_file and antigen_file unless the user explicitly states they are the same file.
- When user provides both "抗体文件" and "抗原文件", extract them to "antibody_file" and "antigen_file" respectively."""
        
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=extraction_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON response
        extracted_params = {}
        
        # Try to parse entire response as JSON
        try:
            extracted_params = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from code blocks
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
            
            # Try to extract nested JSON object
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
            print(f"  [DEBUG] Extracted {len(extracted_params)} parameters from context for tool {tool_name}")
            return extracted_params
        
    except Exception as e:
        print(f"  ⚠ Failed to extract parameters from context: {e}")
        if os.getenv("DEBUG_LLM_ERRORS", "false").lower() == "true":
            import traceback
            traceback.print_exc()
    
    return {}


def _infer_single_parameter(
    task: SubTask,
    tool_name: str,
    param_name: str,
    param_type: Optional[str],
    param_desc: Optional[str],
    param_deme: Optional[str],
    param_options: Optional[List[str]],
    is_optional: bool,
    inputs: List[str],
    llm: Optional[Any],
    all_tasks: List[SubTask],
    extracted_params: Optional[Dict[str, Any]] = None,
    user_input: Optional[str] = None
) -> ParameterInferenceResult:
    """
    Infer value for a single parameter
    
    Priority order:
    1. Extracted parameters from context (user input, execution plan, task description)
    2. Dependency task outputs
    3. Demo/recommended values from tools parameters table
    4. LLM inference based on context
    5. User required (fallback)
    
    Args:
        task: Current task
        tool_name: Tool name
        param_name: Parameter name
        param_type: Parameter type
        param_desc: Parameter description
        param_deme: Parameter example value (recommended value)
        param_options: Parameter enum options (if enum type)
        is_optional: Whether optional
        inputs: Task input list
        llm: LLM instance
        all_tasks: All tasks list (for checking parameter sources)
        extracted_params: Pre-extracted parameters from context (user input, execution plan, task description)
    
    Returns:
        Parameter inference result
    """
    def _normalize_base_dir_value(param_name: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if "base_dir" not in param_name.lower():
            return value
        stripped = value.strip()
        if not stripped:
            return value
        # If it already looks like a directory (ends with separator), keep it
        if stripped.endswith(("/", "\\", os.sep)):
            return value
        basename = os.path.basename(stripped)
        _, ext = os.path.splitext(basename)
        if ext:
            return os.path.dirname(stripped)
        return value
    # Priority 1: Check dependency task outputs FIRST (before context extraction)
    # This ensures that if a parameter can be obtained from a dependency task's output,
    # it takes precedence over context-extracted values
    if task.dependencies:
        # Determine if parameter is file type (needed for matching dependency outputs)
        is_file_type = False
        if param_type:
            param_type_lower = param_type.lower()
            is_file_type = any(keyword in param_type_lower for keyword in [
                'file', 'path', 'pdb', 'csv', 'tsv', 'json', 'fasta', 'fastq', 
                'airr', 'excel', 'xlsx', 'txt', 'dir', 'directory'
            ])
        
        # Check each dependency task's outputs
        for dep_task_id in task.dependencies:
            dep_task = next((t for t in all_tasks if t.task_id == dep_task_id), None)
            if dep_task:
                dep_result = dep_task.result if isinstance(dep_task.result, dict) else {}
                dep_outputs = dep_result.get("outputs", [])
                
                # Try to match parameter with dependency task outputs
                # Match by parameter name, output name, or semantic similarity
                for output in dep_outputs:
                    if isinstance(output, str):
                        # Direct match: parameter name matches output name
                        if param_name.lower() in output.lower() or output.lower() in param_name.lower():
                            return ParameterInferenceResult(
                                param_name=param_name,
                                source_type=ParameterSourceType.FROM_TASK,
                                source_task_id=dep_task_id,
                                source_output_key=output,
                                reason=f"Parameter value will be obtained from dependency task {dep_task_id}'s output: {output}"
                            )
                        
                        # Semantic match for file types: check if output file type matches parameter requirement
                        if is_file_type:
                            # Check if output file extension/type matches parameter requirement
                            output_lower = output.lower()
                            param_lower = param_name.lower()
                            
                            # Match file types: if parameter needs CSV and output is CSV, or parameter needs FASTA and output is FASTA
                            file_type_matches = [
                                ('csv', 'csv'), ('fasta', 'fasta'), ('xlsx', 'xlsx'), 
                                ('json', 'json'), ('tsv', 'tsv'), ('txt', 'txt')
                            ]
                            for param_type_keyword, output_type_keyword in file_type_matches:
                                if param_type_keyword in param_lower and output_type_keyword in output_lower:
                                    return ParameterInferenceResult(
                                        param_name=param_name,
                                        source_type=ParameterSourceType.FROM_TASK,
                                        source_task_id=dep_task_id,
                                        source_output_key=output,
                                        reason=f"Parameter value will be obtained from dependency task {dep_task_id}'s output: {output} (file type match)"
                                    )
                            
                            # Special case: if parameter name contains "antigen" and output contains "antigen"
                            # This handles cases where task outputs antigen data (even if format differs slightly)
                            if 'antigen' in param_lower and 'antigen' in output_lower:
                                return ParameterInferenceResult(
                                    param_name=param_name,
                                    source_type=ParameterSourceType.FROM_TASK,
                                    source_task_id=dep_task_id,
                                    source_output_key=output,
                                    reason=f"Parameter value will be obtained from dependency task {dep_task_id}'s output: {output} (semantic match: antigen)"
                                )
                            
                            # Special case: if parameter name contains "antibody" and output contains "antibody"
                            if 'antibody' in param_lower and 'antibody' in output_lower:
                                return ParameterInferenceResult(
                                    param_name=param_name,
                                    source_type=ParameterSourceType.FROM_TASK,
                                    source_task_id=dep_task_id,
                                    source_output_key=output,
                                    reason=f"Parameter value will be obtained from dependency task {dep_task_id}'s output: {output} (semantic match: antibody)"
                                )
                            
                            # Special case: if parameter is input_file, input_data, sequences, etc.
                            # This handles generic input parameters that should use dependency outputs
                            input_param_patterns = [
                                ('input', 'file'),  # input_file, input_path
                                ('input', 'data'),  # input_data
                                ('sequences',),     # sequences
                                ('data', 'path'),   # data_path
                            ]
                            for pattern in input_param_patterns:
                                if len(pattern) == 2:
                                    if pattern[0] in param_lower and pattern[1] in param_lower:
                                        return ParameterInferenceResult(
                                            param_name=param_name,
                                            source_type=ParameterSourceType.FROM_TASK,
                                            source_task_id=dep_task_id,
                                            source_output_key=output,
                                            reason=f"Parameter value will be obtained from dependency task {dep_task_id}'s output: {output} (pattern match: {pattern[0]}+{pattern[1]})"
                                        )
                                else:
                                    if pattern[0] in param_lower:
                                        return ParameterInferenceResult(
                                            param_name=param_name,
                                            source_type=ParameterSourceType.FROM_TASK,
                                            source_task_id=dep_task_id,
                                            source_output_key=output,
                                            reason=f"Parameter value will be obtained from dependency task {dep_task_id}'s output: {output} (pattern match: {pattern[0]})"
                                        )
    
    # Priority 2: Check extracted parameters from context (user input, execution plan, task description)
    # Only use context-extracted values if not available from dependency tasks
    if extracted_params and param_name in extracted_params:
        extracted_value = extracted_params[param_name]
        if extracted_value is not None:
            extracted_value = _normalize_base_dir_value(param_name, extracted_value)
            # Validation: Check if this is an output file parameter being set to an input file path
            # Output file parameters should not be set to user-provided input file paths
            is_output_param = any(keyword in param_name.lower() for keyword in ['output', 'result', 'save', 'export'])
            input_file_indicators = ['input', '输入', 'file', '文件', 'fasta', 'csv', 'xlsx', 'txt']
            
            if is_output_param and isinstance(extracted_value, str):
                # Check if the value looks like an input file (user-provided file path)
                # If it's mentioned in user input as an input file, don't use it as output
                if any(indicator in str(extracted_value).lower() for indicator in input_file_indicators):
                    # This might be an input file being incorrectly extracted as output
                    # Skip this extraction and let it fall through to other inference methods
                    print(f"  [WARN] Parameter {param_name} extracted value {extracted_value} appears to be an input file, skipping for output parameter")
                    # Don't return here, let it fall through to other inference methods
                else:
                    # Valid output file path, use it
                    return ParameterInferenceResult(
                        param_name=param_name,
                        source_type=ParameterSourceType.DETERMINED,
                        value=extracted_value,
                        reason=f"Parameter value extracted from user input/execution plan/task description: {extracted_value}"
                    )
            else:
                # Not an output parameter, or not a string, use the extracted value
                return ParameterInferenceResult(
                    param_name=param_name,
                    source_type=ParameterSourceType.DETERMINED,
                    value=extracted_value,
                    reason=f"Parameter value extracted from user input/execution plan/task description: {extracted_value}"
                )
    
    # Determine if parameter is file type
    is_file_type = False
    if param_type:
        param_type_lower = param_type.lower()
        is_file_type = any(keyword in param_type_lower for keyword in [
            'file', 'path', 'pdb', 'csv', 'tsv', 'json', 'fasta', 'fastq', 
            'airr', 'excel', 'xlsx', 'txt', 'dir', 'directory'
        ])
    
    # Priority 3: If not file type and has example value, use example value as recommended value
    if not is_file_type and param_deme:
        # If example value is not empty, directly use as determined parameter value
        return ParameterInferenceResult(
            param_name=param_name,
            source_type=ParameterSourceType.DETERMINED,
            value=param_deme,
            reason=f"Using recommended value from tools parameters table: {param_deme}"
        )
    
    # Priority 4: CONSTRAINT - Input file parameters can ONLY come from:
    # 1. User input (via preprocessing, already handled in extracted_params)
    # 2. Dependency task outputs (already checked in Priority 1)
    # If we reach here and it's an input file type, it MUST be user_required
    # DO NOT allow LLM to guess/infer file paths - this caused errors like "/J"
    is_output_param = any(keyword in param_name.lower() for keyword in ['output', 'result', 'save', 'export', 'out'])
    if is_file_type and not is_output_param:
        # This is an INPUT file parameter that wasn't found in:
        # - extracted_params (user input/preprocessing)
        # - dependency task outputs
        # We MUST NOT let LLM guess a file path - mark as user required
        return ParameterInferenceResult(
            param_name=param_name,
            source_type=ParameterSourceType.USER_REQUIRED,
            user_prompt=f"Please provide the input file path for parameter '{param_name}'",
            reason=f"Input file parameter '{param_name}' must be provided by user input or from a previous task output. LLM inference is not allowed for file paths."
        )
    
    # Priority 5: Use LLM to infer parameter values (if no LLM, mark as user required)
    if not llm:
        return ParameterInferenceResult(
            param_name=param_name,
            source_type=ParameterSourceType.USER_REQUIRED,
            user_prompt=f"Please provide value for parameter {param_name}",
            reason="LLM unavailable, requires user to provide parameter value"
        )
    
    # Use LLM to infer parameter values
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Build dependency task information
        dep_tasks_info = []
        if task.dependencies:
            for dep_task_id in task.dependencies:
                dep_task = next((t for t in all_tasks if t.task_id == dep_task_id), None)
                if dep_task:
                    dep_result = dep_task.result if isinstance(dep_task.result, dict) else {}
                    dep_outputs = dep_result.get("outputs", [])
                    dep_tasks_info.append(f"Task {dep_task_id}: {dep_task.content[:100]}... (Outputs: {', '.join(dep_outputs) if dep_outputs else 'None'})")
        
        # Build recommendation hint
        recommendation_hint = ""
        if param_deme:
            if is_file_type:
                recommendation_hint = f"\nRecommended value from tools parameters table (for reference only, user needs to provide actual file): {param_deme}"
            else:
                recommendation_hint = f"\nRecommended value from tools parameters table: {param_deme}"
        
        # Build enum options hint
        options_hint = ""
        if param_options and isinstance(param_options, list) and len(param_options) > 0:
            options_hint = f"\nOptional values list: {', '.join(param_options)}"
            if not param_deme:
                # If no recommended value, hint LLM can recommend a common value from options
                options_hint += " (Please recommend the most commonly used option based on task context and common usage scenarios)"
        
        # Build parameter type guidance
        type_guidance = ""
        if param_type:
            type_lower = param_type.lower()
            if "enum" in type_lower and param_options:
                type_guidance = "\nHint: This is an enum type, please select the most commonly used value from the optional values list as recommendation."
            elif "int" in type_lower or "float" in type_lower:
                type_guidance = "\nHint: This is a numeric type, please provide a reasonable default value based on task context and common usage scenarios."
            elif "bool" in type_lower:
                type_guidance = "\nHint: This is a boolean type, usually the default value is false, unless the task explicitly requires true."
            elif "str" in type_lower and not is_file_type:
                type_guidance = "\nHint: This is a string type, please provide a reasonable example value or default value based on task context and parameter name."
            elif "optional" in type_lower:
                type_guidance = "\nHint: This is an optional parameter, if not explicitly specified in the task description, you can provide None or a reasonable default value."
        
        # Note: extracted_params already checked, so we don't need to check again here
        inference_prompt = f"""
Please infer the parameter value based on the following information:

Task description: {task.content}
Tool name: {tool_name}
Parameter name: {param_name}
Parameter type: {param_type or 'Unknown'}
Parameter description: {param_desc or 'None'}{recommendation_hint}{options_hint}{type_guidance}
Task input list: {inputs}
Dependency task information: {chr(10).join(dep_tasks_info) if dep_tasks_info else 'No dependency tasks'}

Note: Parameters have already been checked from user input, execution plan, and task description. If not found there, please infer based on the following principles.

Please infer the value of this parameter. Return JSON format, supporting three cases:
1. If parameter value can be determined (including inferred from task description, using recommended value, or providing common default value based on parameter type): {{"source_type": "determined", "value": <parameter_value>, "reason": "<inference_reason>"}}
2. If parameter value comes from a dependency task's result: {{"source_type": "from_task", "source_task_id": "<task_id>", "source_output_key": "<output_key>", "reason": "<explanation>"}}
3. If user input is required (only for file paths, user-specific choices, etc.): {{"source_type": "user_required", "user_prompt": "<prompt_info>", "reason": "<why_user_input_needed>"}}

Important principles:
1. **Prioritize providing recommended values**: Even if not explicitly specified in the task description, you should provide recommended values according to the following rules:
   - If there's a recommended value in the tools parameters table, use it first
   - If it's an enum type, recommend the most commonly used value from options (e.g., organism usually recommends "human", species usually recommends "human")
   - If it's a numeric type, provide reasonable default values (e.g., max_results usually recommends 100 or 1000, timeout usually recommends 60 or 300)
   - If it's a boolean type, usually recommend false, unless the task explicitly requires true
   - If it's an optional string type, you can provide None or a reasonable example value
   
2. **Special handling for file types**: Only file path type parameters should be marked as user_required, other types should try to provide recommended values

3. **Task context priority**: If the task description explicitly specifies a parameter value, use the value from the task description; otherwise use the recommended value

4. **Dependency task check**: If the parameter value can be obtained from a dependency task's result, prioritize marking it as from_task

Please follow the above principles to provide recommended values as much as possible, reducing the number of parameters that require manual user input.
"""
        system_message_content = """You are a professional parameter inference expert, capable of extracting parameter values from task descriptions and determining parameter sources.

Your core responsibilities are:
1. **Actively provide recommended values**: For non-file type parameters, even if not explicitly specified in the task description, provide reasonable recommended values based on parameter type, common usage scenarios, and recommended values in the tools parameters table
2. **Intelligent inference**: Infer the most commonly used parameter values based on parameter name, type, description, and task context
3. **Reduce user input**: Mark parameters as "determined" type as much as possible, reducing the number of parameters that require manual user input
4. **Special handling for file types**: Only file path type parameters should be marked as "user_required"

Common recommended value rules:
- organism/species parameters: Usually recommend "human" or "Homo sapiens"
- Numeric parameters like max_results/max_sequences: Usually recommend 100, 1000, or 10000
- timeout parameters: Usually recommend 60, 300, or 600
- bool type parameters: Usually recommend false, unless the task explicitly requires true
- enum type parameters: Recommend the most commonly used value from options
- Optional type parameters: Can provide None or reasonable default values"""
        
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=inference_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse response - use more robust JSON extraction method
        import re
        inference_data = None
        
        # Method 1: Try to parse entire response as JSON
        try:
            inference_data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Method 2: Try to extract JSON code block
        if not inference_data:
            json_block_patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
            ]
            for pattern in json_block_patterns:
                matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        inference_data = json.loads(match)
                        break
                    except json.JSONDecodeError:
                        continue
                if inference_data:
                    break
        
        # Method 3: Try to extract nested JSON object (match complete {} pairs)
        if not inference_data:
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
                            inference_data = json.loads(json_str)
                            break
                        except json.JSONDecodeError:
                            pass
                        start_idx = -1
        
        if inference_data:
            source_type_str = inference_data.get("source_type", "user_required")
            
            if source_type_str == "determined":
                inferred_value = _normalize_base_dir_value(param_name, inference_data.get("value"))
                return ParameterInferenceResult(
                    param_name=param_name,
                    source_type=ParameterSourceType.DETERMINED,
                    value=inferred_value,
                    reason=inference_data.get("reason", "Parameter value inferred from task description")
                )
            elif source_type_str == "from_task":
                return ParameterInferenceResult(
                    param_name=param_name,
                    source_type=ParameterSourceType.FROM_TASK,
                    source_task_id=inference_data.get("source_task_id"),
                    source_output_key=inference_data.get("source_output_key", param_name),
                    reason=inference_data.get("reason", f"Parameter value will be obtained from task {inference_data.get('source_task_id')}'s result")
                )
            else:
                return ParameterInferenceResult(
                    param_name=param_name,
                    source_type=ParameterSourceType.USER_REQUIRED,
                    user_prompt=inference_data.get("user_prompt", f"Please provide value for parameter {param_name}"),
                    reason=inference_data.get("reason", "User input required for parameter value")
                )
        else:
            # Parsing failed, mark as user required
            return ParameterInferenceResult(
                param_name=param_name,
                source_type=ParameterSourceType.USER_REQUIRED,
                user_prompt=f"Please provide value for parameter {param_name}",
                reason="Failed to parse LLM response"
            )
    except Exception as e:
        print(f"  ⚠ Failed to infer parameter {param_name}: {e}")
        return ParameterInferenceResult(
            param_name=param_name,
            source_type=ParameterSourceType.USER_REQUIRED,
            user_prompt=f"Please provide value for parameter {param_name}",
            reason=f"Inference process error: {str(e)[:100]}"
        )


def _infer_parallel_tasks_with_llm(
    tasks: List[Dict[str, Any]],
    llm
) -> Optional[Dict[str, Any]]:
    """
    Use LLM for parallel task inference
    
    Args:
        tasks: Task list from Stage 1 decomposition (JSON format)
        llm: LLM instance
    
    Returns:
        Dictionary containing tasks, parallel_task_groups, parallel_inference_summary, returns None if failed
    """
    # Use parallel inference prompt template
    system_prompt = PARALLEL_INFERENCE_SYSTEM_PROMPT
    user_prompt = get_parallel_inference_user_prompt(tasks)
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Try to parse JSON format response
        result = _parse_parallel_inference_response(response_text)

        print(f"Parallel inference result: {result}")
        
        return result
        
    except Exception as e:
        # Check error type
        error_str = str(e).lower()
        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            print(f"⚠ LLM API Key authentication failed, will use fallback: {type(e).__name__}")
        elif "timeout" in error_str or "timed out" in error_str or "APITimeoutError" in type(e).__name__:
            print(f"⚠ LLM API call timeout, will use fallback: {type(e).__name__}")
        elif "rate limit" in error_str or "429" in error_str:
            print(f"⚠ LLM API rate limit, will use fallback: {type(e).__name__}")
        else:
            print(f"⚠ LLM parallel inference failed, will use fallback: {type(e).__name__}: {str(e)[:100]}")
        
        import os
        if os.getenv("DEBUG_LLM_ERRORS", "false").lower() == "true":
            import traceback
            traceback.print_exc()
        
        return None


def _parse_parallel_inference_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM's parallel inference response
    
    Args:
        response_text: Text returned by LLM
    
    Returns:
        Dictionary containing tasks, parallel_task_groups, parallel_inference_summary
    """
    # Method 1: Try to parse entire response as JSON
    try:
        result = json.loads(response_text.strip())
        if isinstance(result, dict) and ("tasks" in result or "parallel_task_groups" in result):
            return result
    except json.JSONDecodeError:
        pass
    
    # Method 2: Try to extract JSON code block
    json_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and ("tasks" in result or "parallel_task_groups" in result):
                    return result
            except json.JSONDecodeError:
                continue
    
    # Method 3: Try to extract nested JSON object
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
                    result = json.loads(json_str)
                    if isinstance(result, dict) and ("tasks" in result or "parallel_task_groups" in result):
                        return result
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    # If JSON parsing fails, return empty result
    print("Warning: Unable to parse parallel inference JSON format")
    return {
        "tasks": [],
        "parallel_task_groups": [],
        "parallel_inference_summary": "Failed to parse LLM response"
    }


def _coarse_decompose_with_llm(
    user_input: str,
    execution_plan: Optional[str],
    service_list: List[Dict[str, Any]],
    llm
) -> Optional[Dict[str, Any]]:
    """
    Use LLM for coarse decomposition (determine required service_ids)
    
    Args:
        user_input: User's task description
        execution_plan: User-provided execution plan (if any)
        service_list: Service list (contains service_id and description)
        llm: LLM instance
    
    Returns:
        Dictionary containing required_service_ids, returns None if failed
    """
    system_prompt = COARSE_DECOMPOSITION_SYSTEM_PROMPT
    user_prompt = get_coarse_decomposition_user_prompt(user_input, execution_plan, service_list)
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse response
        result = _parse_coarse_decomposition_response(response_text)
        return result
        
    except Exception as e:
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            print(f"⚠ LLM API Key authentication failed: {error_type}")
        elif "timeout" in error_str or "timed out" in error_str or "APITimeoutError" in error_type:
            print(f"⚠ LLM API call timeout: {error_type}")
        elif "rate limit" in error_str or "429" in error_str:
            print(f"⚠ LLM API rate limit: {error_type}")
        else:
            print(f"⚠ LLM coarse decomposition failed: {error_type}: {str(e)[:100]}")
        
        import os
        if os.getenv("DEBUG_LLM_ERRORS", "false").lower() == "true":
            import traceback
            traceback.print_exc()
        
        return None


def _parse_coarse_decomposition_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM's coarse decomposition response
    
    Args:
        response_text: Text returned by LLM
    
    Returns:
        Dictionary containing required_tool_categories
    """
    # Method 1: Try to parse entire response as JSON
    try:
        result = json.loads(response_text.strip())
        if isinstance(result, dict) and "required_service_ids" in result:
            return result
    except json.JSONDecodeError:
        pass
    
    # Method 2: Try to extract JSON code block
    json_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and "required_service_ids" in result:
                    return result
            except json.JSONDecodeError:
                continue
    
    # Method 3: Try to extract nested JSON object
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
                    result = json.loads(json_str)
                    if isinstance(result, dict) and "required_service_ids" in result:
                        return result
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    # If JSON parsing fails, return empty result
    print("Warning: Unable to parse coarse decomposition JSON format")
    return {
        "required_service_ids": []
    }


def _decompose_task_with_llm(
    user_input: str, 
    execution_plan: Optional[str],
    available_tools: List[Dict[str, Any]],
    llm
) -> Optional[Dict[str, Any]]:
    """
    Use LLM for Stage 1 task decomposition (only decompose tasks and match tools, not considering parallel execution)
    
    Args:
        user_input: User's task description
        execution_plan: User-provided execution plan (if any)
        available_tools: List of available tools
        llm: LLM instance
    
    Returns:
        Dictionary containing tasks, decomposition_summary, returns None if failed
    """
    # Use Stage 1 prompt template
    system_prompt = TASK_DECOMPOSITION_SYSTEM_PROMPT
    user_prompt = get_task_decomposition_user_prompt(user_input, execution_plan, available_tools)
    
    # Check input length (rough estimate, 1 Chinese character ≈ 2 tokens)
    total_length = len(system_prompt) + len(user_prompt)
    print(f"📊 Fine decomposition input length: {total_length} characters, tool count: {len(available_tools)}")
    if total_length > 30000:  # If total length exceeds 30000 characters, may need further optimization
        print(f"⚠ Prompt is long ({total_length} characters), if API call fails, will use fallback")
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        print(f"🔄 Starting LLM call for fine decomposition (tool count: {len(available_tools)})...")
        import time
        start_time = time.time()
        
        # LLM call (timeout set by LLM factory at creation, default 120 seconds)
        try:
            response = llm.invoke(messages)
        except Exception as invoke_error:
            elapsed = time.time() - start_time
            error_type = type(invoke_error).__name__
            print(f"⚠ LLM call exception (elapsed {elapsed:.1f}s): {error_type}: {str(invoke_error)[:200]}")
            raise invoke_error
        
        elapsed = time.time() - start_time
        print(f"✓ LLM call completed (elapsed {elapsed:.1f}s), response length: {len(response.content) if response.content else 0} characters")
        response_text = response.content.strip()
        
        # Try to parse JSON format response
        result = _parse_decomposition_response(response_text)
        
        return result
        
    except Exception as e:
        # Check error type
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            print(f"⚠ LLM API Key authentication failed, will use fallback: {error_type}")
        elif "timeout" in error_str or "timed out" in error_str or "APITimeoutError" in error_type:
            print(f"⚠ LLM API call timeout, will use fallback: {error_type}")
        elif "rate limit" in error_str or "429" in error_str:
            print(f"⚠ LLM API rate limit, will use fallback: {error_type}")
        elif "400" in error_str or "badrequest" in error_str or "invalidparameter" in error_str or "range of input" in error_str:
            print(f"⚠ LLM API input parameter error (possibly input too long), will use fallback: {error_type}")
            print(f"  Hint: Tool list may be too long, automatically optimized")
        else:
            print(f"⚠ LLM task decomposition failed, will use fallback: {error_type}: {str(e)[:100]}")
        
        import os
        if os.getenv("DEBUG_LLM_ERRORS", "false").lower() == "true":
            import traceback
            traceback.print_exc()
        
        return None


def _parse_decomposition_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM's Stage 1 task decomposition response
    
    Args:
        response_text: Text returned by LLM
    
    Returns:
        Dictionary containing tasks, decomposition_summary
    """
    # Method 1: Try to parse entire response as JSON
    try:
        result = json.loads(response_text.strip())
        if isinstance(result, dict) and "tasks" in result:
            return result
    except json.JSONDecodeError:
        pass
    
    # Method 2: Try to extract JSON code block
    json_block_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in json_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and "tasks" in result:
                    return result
            except json.JSONDecodeError:
                continue
    
    # Method 3: Try to extract nested JSON object
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
                    result = json.loads(json_str)
                    if isinstance(result, dict) and "tasks" in result:
                        return result
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    # Method 4: If JSON parsing fails, use fallback
    print("Warning: Unable to parse Stage 1 JSON format, using fallback")
    return {
        "tasks": [],
        "decomposition_summary": "Failed to parse LLM response, using fallback"
    }


def _map_task_type_to_enum(task_type_str: str, task_content: str) -> UserTaskType:
    """
    Map task type string to UserTaskType enum
    
    Args:
        task_type_str: Task type string (may come from LLM or user input)
        task_content: Task content (for auxiliary judgment)
    
    Returns:
        UserTaskType enum value
    """
    # First try direct conversion
    task_type_upper = task_type_str.upper()
    try:
        return UserTaskType(task_type_upper)
    except ValueError:
        pass
    
    # If direct conversion fails, intelligently judge based on content
    content_lower = task_content.lower()
    
    # Check if contains execution plan related keywords
    if any(keyword in content_lower for keyword in ["execute", "plan", "step", "follow", "analyze data", "generate report"]):
        return UserTaskType.EXECUTE_PLAN
    
    # Check if contains immunology related keywords
    if any(keyword in content_lower for keyword in ["immune", "antigen", "antibody", "vaccine", "immune system", "immune cell"]):
        return UserTaskType.IMMUNOLOGY_TASK
    
    # Default to general Q&A
    return UserTaskType.GENERAL_QA


def _validate_and_convert_decomposition(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and convert task decomposition results
    
    Supports two formats:
    1. New format: {"tasks": [...], "parallel_task_groups": [...]} (includes tool information)
    2. Old format: {"subtasks": [...], "parallel_task_groups": [...]} (compatible with old version)
    
    Args:
        result: Raw result returned by LLM
    
    Returns:
        Validated and converted result
    """
    subtasks = []
    parallel_task_groups = {}
    
    # Process regular subtasks (supports new format "tasks" and old format "subtasks")
    task_list = result.get("tasks") or result.get("subtasks", [])
    if isinstance(task_list, list):
        for i, task_data in enumerate(task_list):
            try:
                # Check if task_data is dict, skip if not
                if not isinstance(task_data, dict):
                    print(f"Warning: Subtask {i+1} is not dict format, type is {type(task_data).__name__}, skipping")
                    continue
                
                # Ensure task ID exists
                task_id = task_data.get("task_id") or f"task_{i+1}"
                
                # Convert task type (if string is provided, try to convert to UserTaskType)
                task_type_str = task_data.get("task_type", "GENERAL_QA")
                # Use name or content as task description
                task_content = task_data.get("content") or task_data.get("description") or task_data.get("name", "")
                task_type = _map_task_type_to_enum(task_type_str, task_content)
                
                # Build task content (includes tool information)
                content_parts = []
                if task_data.get("name"):
                    content_parts.append(f"Task name: {task_data['name']}")
                if task_data.get("description"):
                    content_parts.append(f"Task description: {task_data['description']}")
                elif task_content:
                    content_parts.append(task_content)
                
                # Add tool information
                if task_data.get("tools") and isinstance(task_data["tools"], list):
                    tool_names = [tool.get("tool_name", tool.get("name", "")) for tool in task_data["tools"] if tool.get("tool_name") or tool.get("name")]
                    if tool_names:
                        content_parts.append(f"Tools: {', '.join(tool_names)}")
                
                # Add input/output information
                if task_data.get("inputs"):
                    content_parts.append(f"Inputs: {', '.join(task_data['inputs'])}")
                if task_data.get("outputs"):
                    content_parts.append(f"Outputs: {', '.join(task_data['outputs'])}")
                
                final_content = "\n".join(content_parts) if content_parts else task_content
                
                subtask = SubTask(
                    task_id=task_id,
                    task_type=task_type,
                    content=final_content,
                    dependencies=task_data.get("dependencies", []),
                    parallel_group_id=task_data.get("parallel_group_id")
                )
                
                # Store tool information in result field (as metadata)
                if task_data.get("tools") or task_data.get("parameters"):
                    subtask.result = {
                        "tools": task_data.get("tools", []),
                        "parameters": task_data.get("parameters", {}),
                        "inputs": task_data.get("inputs", []),
                        "outputs": task_data.get("outputs", [])
                    }
                subtasks.append(subtask)
            except Exception as e:
                task_data_str = str(task_data)[:100] if task_data else "None"
                print(f"Warning: Skipping invalid subtask data at index {i}: {e}, data type: {type(task_data).__name__}, value: {task_data_str}")
                continue
    
    # Process parallel task groups (supports new format "tasks" and old format "subtasks")
    if "parallel_task_groups" in result and isinstance(result["parallel_task_groups"], list):
        for group_idx, group_data in enumerate(result["parallel_task_groups"]):
            try:
                # Check if group_data is dict, skip if not
                if not isinstance(group_data, dict):
                    print(f"Warning: Parallel task group {group_idx+1} is not dict format, type is {type(group_data).__name__}, skipping")
                    continue
                
                group_id = group_data.get("group_id") or f"group_{len(parallel_task_groups)+1}"
                group_subtasks = []
                
                # Supports new format "tasks" and old format "subtasks"
                group_task_list = group_data.get("tasks") or group_data.get("subtasks", [])
                if isinstance(group_task_list, list):
                    for i, task_data in enumerate(group_task_list):
                        try:
                            # Check if task_data is dict, skip if not
                            if not isinstance(task_data, dict):
                                print(f"Warning: Task {i+1} in parallel group {group_id} is not dict format, type is {type(task_data).__name__}, value: {str(task_data)[:100]}, skipping")
                                continue
                            
                            task_id = task_data.get("task_id") or f"{group_id}_task_{i+1}"
                            task_type_str = task_data.get("task_type", "GENERAL_QA")
                            task_content = task_data.get("content") or task_data.get("description") or task_data.get("name", "")
                            task_type = _map_task_type_to_enum(task_type_str, task_content)
                            
                            # Build task content (includes tool information)
                            content_parts = []
                            if task_data.get("name"):
                                content_parts.append(f"Task name: {task_data['name']}")
                            if task_data.get("description"):
                                content_parts.append(f"Task description: {task_data['description']}")
                            elif task_content:
                                content_parts.append(task_content)
                            
                            # Add tool information
                            if task_data.get("tools") and isinstance(task_data["tools"], list):
                                tool_names = [tool.get("tool_name", tool.get("name", "")) for tool in task_data["tools"] if tool.get("tool_name") or tool.get("name")]
                                if tool_names:
                                    content_parts.append(f"Tools: {', '.join(tool_names)}")
                            
                            final_content = "\n".join(content_parts) if content_parts else task_content
                            
                            subtask = SubTask(
                                task_id=task_id,
                                task_type=task_type,
                                content=final_content,
                                dependencies=task_data.get("dependencies", []),
                                parallel_group_id=group_id
                            )
                            
                            # Store tool information in result field (as metadata)
                            if task_data.get("tools") or task_data.get("parameters"):
                                subtask.result = {
                                    "tools": task_data.get("tools", []),
                                    "parameters": task_data.get("parameters", {}),
                                    "inputs": task_data.get("inputs", []),
                                    "outputs": task_data.get("outputs", [])
                                }
                            
                            group_subtasks.append(subtask)
                        except Exception as e:
                            task_data_str = str(task_data)[:100] if task_data else "None"
                            print(f"Warning: Skipping invalid task {i+1} in parallel group {group_id}: {e}, data type: {type(task_data).__name__}, value: {task_data_str}")
                            continue
                
                parallel_group = ParallelTaskGroup(
                    group_id=group_id,
                    subtasks=group_subtasks,
                    status=TaskStatus.PENDING
                )
                parallel_task_groups[group_id] = parallel_group
            except Exception as e:
                print(f"Warning: Skipping invalid parallel task group: {e}")
                continue
    
    return {
        "subtasks": subtasks,
        "parallel_task_groups": parallel_task_groups,
        "decomposition_summary": result.get("decomposition_summary", "")
    }


def _decompose_task_fallback(
    user_input: str, 
    execution_plan: Optional[str]
) -> Dict[str, Any]:
    """
    Fallback: Simple task decomposition when LLM is unavailable
    
    Args:
        user_input: User's task description
        execution_plan: User-provided execution plan (if any)
    
    Returns:
        Dictionary containing subtasks, parallel_task_groups, decomposition_summary
    """
    subtasks = []
    parallel_task_groups = {}
    
    # If there's an execution plan, try to extract steps from it
    if execution_plan:
        # Simple step extraction (by line or numbering)
        lines = execution_plan.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Extract step content
                content = re.sub(r'^[\d\-•\.\s]+', '', line).strip()
                if content:
                    subtask = SubTask(
                        task_id=f"task_{i+1}",
                        task_type=UserTaskType.EXECUTE_PLAN,
                        content=content,
                        dependencies=[f"task_{j}" for j in range(1, i)] if i > 0 else [],
                        parallel_group_id=None
                    )
                    subtasks.append(subtask)
    else:
        # If no execution plan, create a simple task
        subtask = SubTask(
            task_id="task_1",
            task_type=UserTaskType.GENERAL_QA,
            content=user_input,
            dependencies=[],
            parallel_group_id=None
        )
        subtasks.append(subtask)
    
    return {
        "subtasks": subtasks,
        "parallel_task_groups": parallel_task_groups,
        "decomposition_summary": "Using fallback for task decomposition, recommend configuring LLM API Key for better decomposition results"
    }


# ---------------------- Tool Description Cleaning Function =====================
def _clean_tool_description(description: str) -> str:
    """
    Clean and optimize tool description to make it more suitable for semantic matching in task_decomposition subgraph
    
    Optimization strategy:
    1. Remove redundant spaces and line breaks
    2. Remove "Args:", "Yields:" etc. parameter sections (parameter info already in params)
    3. Retain core functionality description, keep it concise and clear
    4. Normalize format for easier LLM semantic matching
    """
    if not description:
        return ""
    
    # Remove leading whitespace
    description = description.strip()
    
    # Remove leading indentation (common 4-space indentation in code docs)
    lines = description.split('\n')
    cleaned_lines = []
    in_args_section = False
    
    for line in lines:
        # Remove leading 4-space indentation
        line = re.sub(r'^ {4,}', '', line).rstrip()
        
        # Detect "Args:", "Yields:", "Returns:" etc. markers
        if re.match(r'^(Args|Yields|Returns|Parameters|Example):', line, re.IGNORECASE):
            in_args_section = True
            continue
        
        # If in args section, skip parameter description lines
        if in_args_section:
            # Parameter lines usually formatted as "param_name: description" or "    param_name: description"
            if re.match(r'^\s*[a-z_]+(\s|:).*', line, re.IGNORECASE):
                continue
            # If encounter empty line, might be end of args section
            if not line:
                continue
            # If encounter new paragraph (non-parameter format), stop skipping
            if line and not re.match(r'^\s*[a-z_]+(\s|:).*', line, re.IGNORECASE):
                in_args_section = False
        
        # Retain non-parameter sections
        if not in_args_section:
            cleaned_lines.append(line)
    
    description = '\n'.join(cleaned_lines)
    
    # Normalize line breaks: merge multiple consecutive line breaks to at most two
    description = re.sub(r'\n{3,}', '\n\n', description)
    
    # Remove trailing spaces
    description = '\n'.join(line.rstrip() for line in description.split('\n'))
    
    # Remove leading empty lines
    description = description.lstrip('\n')
    
    # Final cleanup
    description = description.strip()
    
    return description


# ---------------------- Tool Loading Function =====================
def _load_available_tools() -> List[Dict[str, Any]]:
    """
    Load list of available tools (MCP tools, Skills, persistent tools)
    
    Load tool information from mcp_tools.json (already contains tool list and parameter information)
    
    Returns:
        Tool list, each tool contains name, description, tool fields, etc.
    """
    tools = []
    
    # 1. Load MCP tool configuration
    try:
        mcp_tools_path = agent_dir / "config" / "mcp_tools.json"
        if mcp_tools_path.exists():
            with open(mcp_tools_path, 'r', encoding='utf-8') as f:
                mcp_config = json.load(f)
                
                # Handle two formats:
                # 1. If dict, try to get "mcp_tools" key
                # 2. If list, use directly
                if isinstance(mcp_config, dict):
                    mcp_tools = mcp_config.get("mcp_tools", [])
                elif isinstance(mcp_config, list):
                    mcp_tools = mcp_config
                else:
                    print(f"Warning: MCP tool configuration format incorrect, expected dict or list, actual {type(mcp_config)}")
                    mcp_tools = []
                
                for tool in mcp_tools:
                    tool_name = tool.get("name", "")
                    if not tool_name:
                        continue
                    
                    # Get tool description and optimize (remove redundant format, easier for LLM semantic matching)
                    tool_description = _clean_tool_description(tool.get("description", ""))
                    
                    # Get tool parameters
                    tool_params = tool.get("params", [])
                    
                    # Build parameter description (for LLM understanding)
                    params_description = ""
                    if tool_params:
                        params_parts = []
                        for param in tool_params:
                            param_name = param.get("name", "")
                            param_desc = param.get("description", "")
                            param_type = param.get("type", "string")
                            param_required = param.get("required", False)
                            param_default = param.get("default")
                            
                            param_str = f"- {param_name}"
                            if param_type:
                                param_str += f" ({param_type})"
                            if param_required:
                                param_str += " [Required]"
                            if param_default is not None:
                                param_str += f" [Default: {param_default}]"
                            if param_desc:
                                param_str += f": {param_desc}"
                            
                            params_parts.append(param_str)
                        
                        if params_parts:
                            params_description = "\nParameters:\n" + "\n".join(params_parts)
                    
                    # Build complete tool description (includes parameter information)
                    full_description = tool_description
                    if params_description:
                        full_description = f"{tool_description}\n{params_description}"
                    
                    # Standardize tool format (each tool name corresponds to one executable tool)
                    standardized_tool = {
                        "name": tool_name,
                        "description": full_description,
                        "service": tool.get("service", ""),
                        "tool": [{
                            "tool_name": tool_name,
                            "description": full_description,
                            "parameters": tool_params  # Retain parameter info for subsequent use
                        }]
                    }
                    
                    # Include tool dependency info if specified
                    if tool.get("depends_on"):
                        standardized_tool["depends_on"] = tool.get("depends_on")
                        standardized_tool["execution_order"] = tool.get("execution_order", "after_dependencies")
                    
                    tools.append(standardized_tool)
                
                print(f"Successfully loaded {len(tools)} MCP tools")
        else:
            print(f"Warning: MCP tool configuration file does not exist: {mcp_tools_path}")
    except Exception as e:
        print(f"Warning: Failed to load MCP tool configuration: {e}")
        if os.getenv("DEBUG_LLM_ERRORS", "false").lower() == "true":
            import traceback
            traceback.print_exc()
    
    # 2. TODO: Load Skills (if exists)
    # 3. TODO: Load persistent code cache tools (if exists)
    
    return tools


# ---------------------- State Mapping Functions =====================
def task_decomposition_input_mapper(global_state: GlobalState) -> TaskDecompositionState:
    """
    Main graph → subgraph state mapping
    
    Map main graph's GlobalState to TaskDecompositionState, extract information needed by subgraph.
    Automatically load available tool information.
    
    Args:
        global_state: Main graph's global state
    
    Returns:
        TaskDecompositionState: Subgraph state
    """
    # Load available tools
    available_tools = _load_available_tools()
    
    return TaskDecompositionState(
        user_input=global_state.user_input,
        execution_plan=global_state.execution_plan,
        available_tools=available_tools,
        required_service_ids=[],
        filtered_tools=[],
        subtasks=[],
        parallel_task_groups={},
        decomposition_summary=None
    )


def task_decomposition_output_mapper(
    subgraph_output: TaskDecompositionState | dict, 
    global_state: GlobalState
) -> GlobalState:
    """
    Subgraph → main graph state mapping
    
    Sync TaskDecompositionState results from subgraph back to main graph's GlobalState.
    
    Args:
        subgraph_output: State output from subgraph (may be TaskDecompositionState object or dict)
        global_state: Main graph's global state (will be updated)
    
    Returns:
        GlobalState: Updated main graph state
    """
    
    # Handle dict format state (LangGraph may return dict)
    if isinstance(subgraph_output, dict):
        subgraph_output = TaskDecompositionState(**subgraph_output)
    
    # Sync subtask list
    if subgraph_output.subtasks:
        global_state.subtasks = subgraph_output.subtasks
    
    # Sync parallel task groups
    if subgraph_output.parallel_task_groups:
        global_state.parallel_task_groups = subgraph_output.parallel_task_groups
    
    # Store decomposition summary in merged_result
    if not global_state.merged_result:
        global_state.merged_result = {}
    
    if subgraph_output.decomposition_summary:
        global_state.merged_result["decomposition_summary"] = subgraph_output.decomposition_summary
    
    # Return updated global state
    return global_state


# ---------------------- Build Task Decomposition Agent Subgraph ----------------------
def build_task_decomposition_subgraph():
    """
    Build task decomposition Agent subgraph (three-stage decomposition)
    
    Stage 0: Coarse decomposition (determine required tool types, no tool list passed)
    Stage 1: Fine decomposition (detailed task decomposition and tool matching based on filtered tools)
    Stage 2: Parallel task inference (infer parallel relationships based on fine decomposition results)
    Parameter inference is handled in executor subgraph.
    
    Returns:
        Compiled subgraph
    """
    graph = StateGraph(TaskDecompositionState)
    
    # Add nodes
    graph.add_node("coarse_decompose", coarse_decomposition_node)  # Stage 0: Coarse decomposition
    graph.add_node("fine_decompose", fine_decomposition_node)  # Stage 1: Fine decomposition
    graph.add_node("infer_parallel", parallel_inference_node)  # Stage 2: Parallel inference
    
    # Define flow rules
    graph.add_edge(START, "coarse_decompose")
    graph.add_edge("coarse_decompose", "fine_decompose")  # Enter fine decomposition after coarse decomposition
    graph.add_edge("fine_decompose", "infer_parallel")  # Enter parallel inference after fine decomposition
    graph.add_edge("infer_parallel", END)  # End after parallel inference
    
    return graph.compile()

