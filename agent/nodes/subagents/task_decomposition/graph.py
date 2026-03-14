"""
Task Decomposition Agent Subgraph (Simplified)

Two-stage decomposition:
- Stage 0: Coarse decomposition - determine required service_ids
- Stage 1: Fine decomposition - decompose tasks with tools and dependencies

Output: subtasks[] with task_id, content, tools[], dependencies[]
Parameter inference is handled by the orchestrator/opencode_executor.
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, START, END
import sys
import json
import re
import os
from pathlib import Path
import time

from .prompt import (
    COARSE_DECOMPOSITION_SYSTEM_PROMPT,
    get_coarse_decomposition_user_prompt,
    TASK_DECOMPOSITION_SYSTEM_PROMPT,
    get_task_decomposition_user_prompt,
    get_task_decomposition_user_prompt_with_skills,
)
from .tool_categorizer import (
    load_service_list,
    get_tools_by_service_ids,
)
from .skill_loader import (
    get_cached_skills,
    get_cached_task_guide,
    get_skills_for_services,
    format_skills_for_prompt,
    format_task_guide_for_prompt,
)

agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, SubTask, UserTaskType

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_advanced_llm

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_advanced_llm = None
    HumanMessage = None
    SystemMessage = None


class TaskDecompositionState(BaseModel):
    """Simplified Task Decomposition State"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    progress_callback: Optional[Callable] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    parent_state: Optional[Any] = Field(default=None)

    user_input: str = Field(description="User's original input")
    execution_plan: Optional[str] = Field(default=None)
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)

    required_service_ids: List[str] = Field(default_factory=list)
    filtered_tools: List[Dict[str, Any]] = Field(default_factory=list)
    raw_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    subtasks: List[SubTask] = Field(default_factory=list)
    decomposition_summary: Optional[str] = Field(default=None)

    def get_llm(
        self, purpose: str = "reasoning", node_name: Optional[str] = None, **kwargs
    ) -> Optional[Any]:
        if self.parent_state and hasattr(self.parent_state, "get_llm"):
            return self.parent_state.get_llm(
                purpose=purpose, node_name=node_name or "task_decomposition", **kwargs
            )
        from utils.llm_factory import create_llm_with_thinking

        return create_llm_with_thinking(
            purpose=purpose,
            progress_callback=self.progress_callback,
            session_id=self.session_id,
            node_name=node_name or "task_decomposition",
            **kwargs,
        )


def _get_llm_from_state(
    state: TaskDecompositionState, purpose: str = "reasoning_advanced"
):
    if hasattr(state, "get_llm") and callable(getattr(state, "get_llm", None)):
        return state.get_llm(purpose=purpose, node_name="task_decomposition")
    if not LLM_AVAILABLE or create_reasoning_advanced_llm is None:
        return None
    return create_reasoning_advanced_llm(temperature=0.2)


def _create_codeact_tool() -> Dict[str, Any]:
    return {
        "name": "codeact",
        "service": "codeact",
        "description": "Code execution tool for writing and executing Python code to complete complex tasks",
        "tool": [{"tool_name": "codeact", "description": "Code execution tool"}],
    }


def _extract_response_text(response) -> str:
    response_text = ""

    if hasattr(response, "content") and response.content:
        content = response.content
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
            response_text = "".join(texts)

    if not response_text and hasattr(response, "additional_kwargs"):
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            response_text = reasoning

    if not response_text and hasattr(response, "response_metadata"):
        metadata = response.response_metadata
        if isinstance(metadata, dict):
            for key in ["content", "text", "reasoning_content", "output"]:
                if key in metadata and metadata[key]:
                    response_text = str(metadata[key])
                    break

    if not response_text:
        if hasattr(response, "text"):
            response_text = response.text
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", str(response))

    return response_text.strip() if response_text else ""


def _parse_coarse_decomposition_response(response_text: str) -> Dict[str, Any]:
    try:
        result = json.loads(response_text.strip())
        if isinstance(result, dict) and "required_service_ids" in result:
            return result
    except json.JSONDecodeError:
        pass

    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and "required_service_ids" in result:
                    return result
            except json.JSONDecodeError:
                continue

    brace_count, start_idx = 0, -1
    for i, char in enumerate(response_text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    result = json.loads(response_text[start_idx : i + 1])
                    if isinstance(result, dict) and "required_service_ids" in result:
                        return result
                except json.JSONDecodeError:
                    pass
                start_idx = -1

    print("Warning: Unable to parse coarse decomposition JSON")
    return {"required_service_ids": []}


def _parse_decomposition_response(response_text: str) -> Dict[str, Any]:
    try:
        result = json.loads(response_text.strip())
        if isinstance(result, dict) and "tasks" in result:
            return result
    except json.JSONDecodeError:
        pass

    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict) and "tasks" in result:
                    return result
            except json.JSONDecodeError:
                continue

    brace_count, start_idx = 0, -1
    for i, char in enumerate(response_text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    result = json.loads(response_text[start_idx : i + 1])
                    if isinstance(result, dict) and "tasks" in result:
                        return result
                except json.JSONDecodeError:
                    pass
                start_idx = -1

    print("Warning: Unable to parse fine decomposition JSON")
    return {"tasks": [], "decomposition_summary": "Failed to parse LLM response"}


def _map_task_type_to_enum(task_type_str: str, task_content: str) -> UserTaskType:
    task_type_upper = task_type_str.upper()
    try:
        return UserTaskType(task_type_upper)
    except ValueError:
        pass

    content_lower = task_content.lower()
    if any(kw in content_lower for kw in ["execute", "plan", "step", "analyze data"]):
        return UserTaskType.EXECUTE_PLAN
    if any(kw in content_lower for kw in ["immune", "antigen", "antibody", "vaccine"]):
        return UserTaskType.IMMUNOLOGY_TASK
    return UserTaskType.GENERAL_QA


def _validate_and_convert_decomposition(result: Dict[str, Any]) -> Dict[str, Any]:
    subtasks = []
    task_list = result.get("tasks") or result.get("subtasks", [])

    for i, task_data in enumerate(task_list):
        if not isinstance(task_data, dict):
            continue

        task_id = task_data.get("task_id") or f"task_{i + 1}"
        task_type_str = task_data.get("task_type", "GENERAL_QA")
        task_content = (
            task_data.get("content")
            or task_data.get("description")
            or task_data.get("name", "")
        )
        task_type = _map_task_type_to_enum(task_type_str, task_content)

        content_parts = []
        if task_data.get("name"):
            content_parts.append(f"Task name: {task_data['name']}")
        if task_data.get("description"):
            content_parts.append(f"Task description: {task_data['description']}")
        elif task_content:
            content_parts.append(task_content)

        if task_data.get("tools") and isinstance(task_data["tools"], list):
            tool_names = [
                tool.get("tool_name", tool.get("name", ""))
                for tool in task_data["tools"]
                if tool.get("tool_name") or tool.get("name")
            ]
            if tool_names:
                content_parts.append(f"Tools: {', '.join(tool_names)}")

        final_content = "\n".join(content_parts) if content_parts else task_content

        subtask = SubTask(
            task_id=task_id,
            task_type=task_type,
            content=final_content,
            dependencies=task_data.get("dependencies", []),
            parallel_group_id=task_data.get("parallel_group_id"),
        )

        if task_data.get("tools"):
            subtask.result = {
                "tools": task_data.get("tools", []),
                "parameters": task_data.get("parameters", {}),
                "inputs": task_data.get("inputs", []),
                "outputs": task_data.get("outputs", []),
            }
        subtasks.append(subtask)

    return {"subtasks": subtasks}


def _decompose_task_fallback(
    user_input: str, execution_plan: Optional[str]
) -> Dict[str, Any]:
    plan_text = execution_plan or user_input
    subtask = SubTask(
        task_id="task_001",
        task_type=UserTaskType.EXECUTE_PLAN,
        content=plan_text,
        dependencies=[],
    )
    subtask.result = {"tools": [], "parameters": {}, "inputs": [], "outputs": []}

    return {
        "subtasks": [subtask],
        "decomposition_summary": "Fallback: Single task created from user input",
    }


def _load_available_tools() -> List[Dict[str, Any]]:
    tools = []
    mcp_tools_path = (
        Path(__file__).parent.parent.parent.parent / "mcp_tools_config.json"
    )

    if mcp_tools_path.exists():
        try:
            with open(mcp_tools_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                if "mcpServers" in config:
                    for server_name, server_config in config["mcpServers"].items():
                        if server_config.get("disabled"):
                            continue
                        tools.append(
                            {
                                "name": server_name,
                                "description": server_config.get("description", ""),
                                "service": server_config.get("service", server_name),
                                "tool": server_config.get("tools", []),
                            }
                        )
            print(f"Loaded {len(tools)} MCP tools")
        except Exception as e:
            print(f"Warning: Failed to load MCP tool config: {e}")

    return tools


def coarse_decomposition_node(state: TaskDecompositionState) -> TaskDecompositionState:
    stage_start = time.time()
    print("  [Coarse] Starting Stage 0: Coarse decomposition...")

    user_input = state.user_input
    execution_plan = state.execution_plan

    service_list = load_service_list()
    print(f"  [Coarse] Loaded {len(service_list)} services")

    llm = _get_llm_from_state(state, "reasoning_advanced")
    if llm is not None:
        print("  [Coarse] Calling LLM for coarse decomposition...")
        system_prompt = COARSE_DECOMPOSITION_SYSTEM_PROMPT
        user_prompt = get_coarse_decomposition_user_prompt(
            user_input, execution_plan, service_list
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            response_text = _extract_response_text(response)
            result = _parse_coarse_decomposition_response(response_text)

            if result:
                state.required_service_ids = result.get("required_service_ids", [])
                print(f"  [OK] Coarse decomposition complete")
                print(f"    Required services: {', '.join(state.required_service_ids)}")
            else:
                all_service_ids = [
                    s.get("service_id", "") for s in service_list if s.get("service_id")
                ]
                state.required_service_ids = all_service_ids
                print("  [WARN] Coarse decomposition failed, using all services")
        except Exception as e:
            print(f"[WARN] LLM coarse decomposition failed: {e}")
            all_service_ids = [
                s.get("service_id", "") for s in service_list if s.get("service_id")
            ]
            state.required_service_ids = all_service_ids
    else:
        all_service_ids = [
            s.get("service_id", "") for s in service_list if s.get("service_id")
        ]
        state.required_service_ids = all_service_ids
        print("  [WARN] LLM unavailable, using all services")

    if state.required_service_ids:
        state.filtered_tools = get_tools_by_service_ids(
            state.available_tools, state.required_service_ids
        )
        print(
            f"  [Coarse] Filtered tools: {len(state.filtered_tools)} / {len(state.available_tools)}"
        )

        if len(state.filtered_tools) == 0:
            print("  [WARN] No matched MCP tools, adding codeact as fallback")
            if "codeact" not in state.required_service_ids:
                state.required_service_ids.append("codeact")
            state.filtered_tools = [_create_codeact_tool()]
    else:
        state.filtered_tools = state.available_tools

    print(f"  [OK] Stage 0 complete ({time.time() - stage_start:.2f}s)\n")
    return state


def fine_decomposition_node(state: TaskDecompositionState) -> TaskDecompositionState:
    stage_start = time.time()
    print("  [Fine] Starting Stage 1: Fine decomposition...")

    user_input = state.user_input
    execution_plan = state.execution_plan
    filtered_tools = state.filtered_tools

    if len(filtered_tools) == 0:
        print("  [WARN] No filtered tools, adding codeact as fallback")
        filtered_tools = [_create_codeact_tool()]
        state.filtered_tools = filtered_tools
        if "codeact" not in state.required_service_ids:
            state.required_service_ids.append("codeact")

    print(f"  [Fine] Available tools: {len(filtered_tools)}")

    llm = _get_llm_from_state(state, "reasoning_advanced")
    if llm is not None:
        print("  [Fine] Calling LLM for fine decomposition...")

        skills_info = None
        task_guide = None

        if state.required_service_ids:
            try:
                skills = get_skills_for_services(state.required_service_ids)
                if skills:
                    skills_info = format_skills_for_prompt(skills, max_skills=10)
                    print(f"  Loaded {len(skills)} skills")

                task_guide_content = get_cached_task_guide()
                if task_guide_content:
                    task_guide = format_task_guide_for_prompt(
                        task_guide_content, max_length=2500
                    )
            except Exception as e:
                print(f"  [WARN] Failed to load skills/guide: {e}")

        system_prompt = TASK_DECOMPOSITION_SYSTEM_PROMPT
        if skills_info or task_guide:
            user_prompt = get_task_decomposition_user_prompt_with_skills(
                user_input,
                execution_plan,
                filtered_tools,
                skills_info=skills_info,
                task_guide=task_guide,
            )
        else:
            user_prompt = get_task_decomposition_user_prompt(
                user_input, execution_plan, filtered_tools
            )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed = time.time() - start_time

            response_text = _extract_response_text(response)
            print(
                f"  [OK] LLM call completed ({elapsed:.1f}s), response: {len(response_text) if response_text else 0} chars"
            )

            result = _parse_decomposition_response(response_text)

            if result and result.get("tasks"):
                state.raw_tasks = result.get("tasks", [])
                state.decomposition_summary = result.get("decomposition_summary", "")
                print(f"  [OK] Fine decomposition complete")
                print(f"    Tasks: {len(state.raw_tasks)}")
            else:
                print("  [WARN] LLM returned empty tasks, using fallback")
                fallback_result = _decompose_task_fallback(user_input, execution_plan)
                state.raw_tasks = [
                    task.model_dump() for task in fallback_result.get("subtasks", [])
                ]
                state.decomposition_summary = fallback_result.get(
                    "decomposition_summary", "Using fallback"
                )
        except Exception as e:
            print(f"[WARN] LLM task decomposition failed: {e}")
            fallback_result = _decompose_task_fallback(user_input, execution_plan)
            state.raw_tasks = [
                task.model_dump() for task in fallback_result.get("subtasks", [])
            ]
            state.decomposition_summary = fallback_result.get(
                "decomposition_summary", "Using fallback"
            )
    else:
        print("  [WARN] LLM unavailable, using fallback")
        fallback_result = _decompose_task_fallback(user_input, execution_plan)
        state.raw_tasks = [
            task.model_dump() for task in fallback_result.get("subtasks", [])
        ]
        state.decomposition_summary = fallback_result.get(
            "decomposition_summary", "Using fallback"
        )

    convert_start = time.time()
    final_result = _validate_and_convert_decomposition({"tasks": state.raw_tasks})
    state.subtasks = final_result.get("subtasks", [])
    print(f"  [Fine] Conversion complete ({time.time() - convert_start:.2f}s)")

    print(f"  [OK] Stage 1 complete ({time.time() - stage_start:.2f}s)\n")
    return state


def task_decomposition_input_mapper(
    global_state: GlobalState,
) -> TaskDecompositionState:
    available_tools = _load_available_tools()

    return TaskDecompositionState(
        user_input=global_state.user_input,
        execution_plan=getattr(global_state, "execution_plan", None),
        available_tools=available_tools,
        required_service_ids=[],
        filtered_tools=[],
        subtasks=[],
        progress_callback=getattr(global_state, "progress_callback", None),
        session_id=getattr(global_state, "session_id", None),
        parent_state=global_state,
    )


def task_decomposition_output_mapper(
    subgraph_output: TaskDecompositionState | dict, global_state: GlobalState
) -> GlobalState:
    if isinstance(subgraph_output, dict):
        subgraph_output = TaskDecompositionState(**subgraph_output)

    if subgraph_output.subtasks:
        global_state.subtasks = subgraph_output.subtasks

    if not global_state.merged_result:
        global_state.merged_result = {}

    if subgraph_output.decomposition_summary:
        global_state.merged_result["decomposition_summary"] = (
            subgraph_output.decomposition_summary
        )

    return global_state


def build_task_decomposition_subgraph():
    graph = StateGraph(TaskDecompositionState)

    graph.add_node("coarse_decompose", coarse_decomposition_node)
    graph.add_node("fine_decompose", fine_decomposition_node)

    graph.add_edge(START, "coarse_decompose")
    graph.add_edge("coarse_decompose", "fine_decompose")
    graph.add_edge("fine_decompose", END)

    return graph.compile()
