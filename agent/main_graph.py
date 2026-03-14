# agent/main_graph.py
"""
Bio-Agent Main Graph (Refactored v2)

Flow:
1. init => classify (immunity) => immunity subgraph => task_decomposition => orchestrator => result_evaluation => END
2. init => classify (has_plan) => task_decomposition => orchestrator => result_evaluation => END
3. init => classify (general_qa) => general_qa subgraph => extract_answer => END
4. init => classify (model_training) => model_training (generates subtask) => orchestrator => result_evaluation => END
"""

from typing import Optional, Dict, Any, List
from langgraph.graph import StateGraph, START, END
from pathlib import Path
import sys
import json
import uuid
import asyncio
import os
from datetime import datetime

agent_dir = Path(__file__).parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, TaskStatus, UserTaskType, SubTask
from prompts import CLASSIFY_SYSTEM_PROMPT, CLASSIFY_USER_PROMPT
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class TaskCategory(str, Enum):
    """任务分类枚举"""

    IMMUNITY = "immunity"
    GENERAL_QA = "general_qa"
    MODEL_TRAINING = "model_training"
    HAS_PLAN = "has_plan"
    UNKNOWN = "unknown"


def _extract_file_paths(user_input: str) -> list:
    """Extract file paths from user input."""
    import re

    patterns = [
        r"/[a-zA-Z0-9_\-./]+\.(csv|xlsx|xls|tsv|json|fasta|fa|txt|rds|RDS)",
        r"/data/[a-zA-Z0-9_\-./]+",
    ]

    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, user_input)
        paths.extend(matches)

    return list(set(paths))


def _classify_file_source(path: str):
    """Classify file source type: url, local, or remote."""
    if path.startswith(("http://", "https://")):
        return "url"
    try:
        if os.path.exists(path):
            return "local"
    except:
        pass
    return "remote"


async def _copy_files_to_session_dir(file_paths: list, session_id: str) -> dict:
    """Copy/upload files to session input directory using OpenSandbox."""
    from utils.codeact_executor import execute_code_via_codeact_async

    input_dir = f"/data/sessions/{session_id}/input"
    copied_files = {}

    for src_path in file_paths:
        source_type = _classify_file_source(src_path)
        filename = os.path.basename(
            src_path.split("?")[0]
        )  # Remove query params for URLs
        dst_path = f"{input_dir}/{filename}"

        if source_type == "url":
            code = f'''
import os
import urllib.request

url = "{src_path}"
target = "{dst_path}"

try:
    target_dir = os.path.dirname(target)
    os.makedirs(target_dir, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    print(f"Downloaded: {{url}} -> {{target}}")
except Exception as e:
    print(f"Download error: {{e}}")
'''
            task_desc = f"Download {src_path} to {dst_path}"

        elif source_type == "local":
            try:
                with open(src_path, "rb") as f:
                    content = f.read()
                try:
                    text_content = content.decode("utf-8")
                    content_str = repr(text_content)
                except UnicodeDecodeError:
                    import base64

                    content_str = (
                        f"base64.b64decode('{base64.b64encode(content).decode()}')"
                    )

                code = f'''
import os
import base64

content = {content_str}
target = "{dst_path}"

try:
    target_dir = os.path.dirname(target)
    os.makedirs(target_dir, exist_ok=True)
    if isinstance(content, str):
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        with open(target, 'wb') as f:
            f.write(content)
    print(f"Uploaded: {src_path} -> {{target}}")
except Exception as e:
    print(f"Upload error: {{e}}")
'''
            except Exception as e:
                print(f"[Init] Cannot read local file {src_path}: {e}")
                continue
            task_desc = f"Upload {src_path} to {dst_path}"

        else:  # remote - file on sandbox server
            code = f'''
import os
import shutil

source = "{src_path}"
target = "{dst_path}"

try:
    if os.path.exists(source):
        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(source, target)
        print(f"Copied: {{source}} -> {{target}}")
    else:
        print(f"File not found on server: {{source}}")
except Exception as e:
    print(f"Copy error: {{e}}")
'''
            task_desc = f"Copy {src_path} to {dst_path}"

        try:
            result = await execute_code_via_codeact_async(
                task_description=task_desc,
                code_template=code,
                timeout_seconds=60,
                keep_alive=False,
            )
            if result.status.value == "success":
                copied_files[src_path] = dst_path
                print(f"[Init] {source_type}: {src_path} -> {dst_path}")
            else:
                print(f"[Init] Failed ({source_type}): {src_path}: {result.error}")
        except Exception as e:
            print(f"[Init] Error ({source_type}): {src_path}: {e}")

    return copied_files


async def init_node(state: GlobalState) -> GlobalState:
    """初始化节点"""
    if not state.session_id:
        state.session_id = str(uuid.uuid4())[:8]

    session_id = state.session_id

    print(f"\n{'=' * 60}")
    print(f"[Init] Session ID: {session_id}")
    print(
        f"[Init] User Input: {state.user_input[:100]}..."
        if len(state.user_input) > 100
        else f"[Init] User Input: {state.user_input}"
    )
    print(f"{'=' * 60}")

    if not state.sandbox_dir:
        state.sandbox_dir = f"/data/sessions/{session_id}"

    if not state.sandbox_data_dir:
        state.sandbox_data_dir = f"/data/sessions/{session_id}"

    if not state.merged_result:
        state.merged_result = {}

    file_paths = _extract_file_paths(state.user_input)
    print(f"[Init] Extracted file paths: {file_paths}")

    copied_files = await _copy_files_to_session_dir(file_paths, session_id)
    state.merged_result["copied_files"] = copied_files
    state.merged_result["input_dir"] = f"/data/sessions/{session_id}/input"

    return state


def classify_node(state: GlobalState) -> GlobalState:
    """任务分类节点"""
    print(f"\n{'=' * 60}")
    print("[Classify] 开始任务分类...")
    print(f"{'=' * 60}")

    user_input = state.user_input

    llm = state.get_llm(purpose="reasoning", node_name="classify")

    if not llm:
        print("[Classify] LLM 不可用，默认分类为 general_qa")
        state.supervisor_decision = TaskCategory.GENERAL_QA.value
        return state

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        user_prompt = CLASSIFY_USER_PROMPT.format(user_input=user_input)
        messages = [
            SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        print("[Classify] 调用 LLM 进行分类...")
        response = llm.invoke(messages)

        # Extract content from response - handle different response formats
        # GLM-4.5 with native thinking returns empty content, actual content in additional_kwargs['reasoning_content']
        response_text = ""

        # First check if content is non-empty
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

        # If content is empty, check reasoning_content or additional_kwargs
        if not response_text:
            if hasattr(response, "additional_kwargs"):
                # Check for reasoning_content in additional_kwargs (GLM native thinking)
                reasoning = response.additional_kwargs.get("reasoning_content", "")
                if reasoning:
                    response_text = reasoning

        # Still empty? Check response_metadata
        if not response_text and hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if isinstance(metadata, dict):
                # Check various possible keys
                for key in ["content", "text", "reasoning_content", "output"]:
                    if key in metadata and metadata[key]:
                        response_text = str(metadata[key])
                        break

        # Last resort: use text attribute or str(response)
        if not response_text:
            if hasattr(response, "text"):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            elif isinstance(response, dict):
                response_text = response.get("content", str(response))
            else:
                response_text = str(response)

        result = _parse_json_response(response_text)

        category_str = result.get("category", "general_qa").lower()
        confidence = result.get("confidence", 0.5)
        reason = result.get("reason", "")

        try:
            category = TaskCategory(category_str)
        except ValueError:
            print(f"[Classify] 未知的分类: {category_str}，默认为 general_qa")
            category = TaskCategory.GENERAL_QA

        state.supervisor_decision = category.value
        state.supervisor_reasoning = f"置信度: {confidence}, 原因: {reason}"

        print(f"[Classify] 分类结果: {category.value}")
        print(f"[Classify] 置信度: {confidence}")
        print(f"[Classify] 原因: {reason}")

    except Exception as e:
        print(f"[Classify] 分类失败: {e}")
        state.supervisor_decision = TaskCategory.GENERAL_QA.value

    print(f"{'=' * 60}")
    return state


def classify_router(state: GlobalState) -> str:
    """根据分类结果路由到不同分支"""
    decision = state.supervisor_decision or TaskCategory.GENERAL_QA.value

    if decision == TaskCategory.IMMUNITY.value:
        return "immunity_branch"
    elif decision == TaskCategory.HAS_PLAN.value:
        return "has_plan_branch"
    elif decision == TaskCategory.MODEL_TRAINING.value:
        return "model_training_branch"
    else:
        return "general_qa_branch"


async def immunity_node(state: GlobalState) -> GlobalState:
    """Immunity 子图节点"""
    print(f"\n{'=' * 60}")
    print("[Immunity] 启动 Immunity 子图...")
    print(f"{'=' * 60}")

    try:
        from nodes.subagents.immunity import (
            build_immunity_subgraph,
            immunity_input_mapper,
            immunity_output_mapper,
        )

        immunity_graph = build_immunity_subgraph()
        immunity_state = immunity_input_mapper(state)

        print("[Immunity] 执行 Immunity 子图...")
        result = immunity_graph.invoke(immunity_state)

        if isinstance(result, dict):
            from nodes.subagents.immunity.graph import ImmunityState

            immunity_state = ImmunityState(**result)
        else:
            immunity_state = result

        state = immunity_output_mapper(immunity_state, state)

        execution_plan = state.merged_result.get("immunity_plan", {}).get(
            "final_enhanced_plan", ""
        )
        if execution_plan:
            state.execution_plan = execution_plan
            print(f"[Immunity] 获取到执行计划，长度: {len(execution_plan)}")

        print(f"[Immunity] Immunity 子图完成")

    except Exception as e:
        print(f"[Immunity] Immunity 子图执行失败: {e}")
        import traceback

        traceback.print_exc()
        state.execution_plan = f"Immunity 子图执行失败: {str(e)}"

    print(f"{'=' * 60}")
    return state


def task_decomposition_node(state: GlobalState) -> GlobalState:
    """Task Decomposition 子图节点"""
    print(f"\n{'=' * 60}")
    print("[TaskDecomposition] 启动 Task Decomposition 子图...")
    print(f"{'=' * 60}")

    try:
        from nodes.subagents.task_decomposition.graph import (
            build_task_decomposition_subgraph,
            task_decomposition_input_mapper,
            task_decomposition_output_mapper,
        )

        td_graph = build_task_decomposition_subgraph()
        td_state = task_decomposition_input_mapper(state)

        print("[TaskDecomposition] 执行 Task Decomposition 子图...")
        result = td_graph.invoke(td_state)

        if isinstance(result, dict):
            from nodes.subagents.task_decomposition.graph import TaskDecompositionState

            td_state = TaskDecompositionState(**result)
        else:
            td_state = result

        state = task_decomposition_output_mapper(td_state, state)

        print(f"[TaskDecomposition] Task Decomposition 子图完成")

    except Exception as e:
        print(f"[TaskDecomposition] Task Decomposition 子图执行失败: {e}")
        import traceback

        traceback.print_exc()

    print(f"{'=' * 60}")
    return state


async def orchestrator_node(state: GlobalState) -> GlobalState:
    """Orchestrator Node - bundles tasks by domain and dispatches to sub-agents"""
    print(f"\n{'=' * 60}")
    print("[Orchestrator] Starting orchestrator...")
    print(f"{'=' * 60}")

    try:
        from nodes.subagents.orchestrator import orchestrator_node as _orchestrator_node

        state = await _orchestrator_node(state)

        print(f"[Orchestrator] Orchestrator completed")

    except Exception as e:
        print(f"[Orchestrator] Orchestrator execution failed: {e}")
        import traceback

        traceback.print_exc()
        if not state.merged_result:
            state.merged_result = {}
        state.merged_result["orchestrator_error"] = str(e)

    print(f"{'=' * 60}")
    return state


def result_evaluator_node(state: GlobalState) -> GlobalState:
    """Result Evaluator 子图节点"""
    print(f"\n{'=' * 60}")
    print("[ResultEvaluator] 启动 Result Evaluator 子图...")
    print(f"{'=' * 60}")

    try:
        from nodes.subagents.result_evaluator import (
            build_result_evaluator_subgraph,
            result_evaluator_input_mapper,
            result_evaluator_output_mapper,
        )

        evaluator_graph = build_result_evaluator_subgraph()
        evaluator_state = result_evaluator_input_mapper(state)

        print("[ResultEvaluator] 执行 Result Evaluator 子图...")
        result = evaluator_graph.invoke(evaluator_state)

        if isinstance(result, dict):
            from nodes.subagents.result_evaluator.state import ResultEvaluatorState

            evaluator_state = ResultEvaluatorState(**result)
        else:
            evaluator_state = result

        state = result_evaluator_output_mapper(evaluator_state, state)

        evaluation = state.merged_result.get("result_evaluation", {})
        state.merged_result["evaluate_result"] = evaluation

        print(f"[ResultEvaluator] Result Evaluator 子图完成")

    except Exception as e:
        print(f"[ResultEvaluator] Result Evaluator 子图执行失败: {e}")
        import traceback

        traceback.print_exc()

    print(f"{'=' * 60}")
    return state


async def general_qa_node(state: GlobalState) -> GlobalState:
    """General QA 子图节点"""
    print(f"\n{'=' * 60}")
    print("[GeneralQA] 启动 General QA 子图...")
    print(f"{'=' * 60}")

    try:
        from nodes.subagents.general_qa import (
            build_general_qa_subgraph,
            general_qa_input_mapper,
            general_qa_output_mapper,
        )

        qa_graph = build_general_qa_subgraph()
        qa_state = general_qa_input_mapper(state)

        print("[GeneralQA] 执行 General QA 子图...")
        result = qa_graph.invoke(qa_state)

        if isinstance(result, dict):
            from nodes.subagents.general_qa.state import GeneralQAState

            # Ensure GlobalState is available for type resolution
            try:
                from agent.state import GlobalState
                import sys

                state_module = sys.modules.get("agent.nodes.subagents.general_qa.state")
                if state_module and not hasattr(state_module, "GlobalState"):
                    state_module.GlobalState = GlobalState
                GeneralQAState.model_rebuild()
            except ImportError:
                pass

            qa_state = GeneralQAState(**result)
        else:
            qa_state = result

        state = general_qa_output_mapper(qa_state, state)

        print(f"[GeneralQA] General QA 子图完成")

    except Exception as e:
        print(f"[GeneralQA] General QA 子图执行失败: {e}")
        import traceback

        traceback.print_exc()

    print(f"{'=' * 60}")
    return state


def extract_answer_node(state: GlobalState) -> GlobalState:
    """提取答案节点"""
    print(f"\n{'=' * 60}")
    print("[ExtractAnswer] 提取答案...")
    print(f"{'=' * 60}")

    answer = state.merged_result.get("general_qa_answer")

    if not answer:
        answer = state.merged_result.get("general_qa_conclusion")

    if answer:
        state.merged_result["final_answer"] = str(answer)
        print(f"[ExtractAnswer] 提取到答案: {str(answer)[:200]}...")
    else:
        state.merged_result["final_answer"] = "未能获取到答案"
        print(f"[ExtractAnswer] 未能提取到答案")

    print(f"{'=' * 60}")
    return state


def _load_available_services() -> list:
    """Load available MCP training services from config with descriptions."""
    config_path = Path(__file__).parent / "config" / "service_list.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            services = json.load(f)
            training_services = [
                {
                    "service_id": s.get("service_id"),
                    "description": s.get("description", ""),
                }
                for s in services
                if s.get("service_id") and s.get("service_id").lower().endswith("train")
            ]
            print(
                f"[ModelTraining] Available training services: {[s['service_id'] for s in training_services]}"
            )
            return training_services
    except Exception as e:
        print(f"[ModelTraining] Failed to load service list: {e}")
        return []


def _extract_service_name(
    state: GlobalState, available_services: list
) -> Optional[str]:
    """Extract MCP server name from user input using LLM."""
    from prompts import MODEL_TRAINING_EXTRACT_SERVICE_PROMPT
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = state.get_llm(purpose="reasoning", node_name="model_training")
    if not llm:
        print("[ModelTraining] LLM not available, cannot extract service name")
        return None

    services_lines = []
    for s in available_services:
        services_lines.append(f"- {s['service_id']}: {s['description']}")
    services_str = "\n".join(services_lines)

    service_ids = [s["service_id"] for s in available_services]

    prompt = MODEL_TRAINING_EXTRACT_SERVICE_PROMPT.format(
        available_services=services_str, user_input=state.user_input
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
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

        result = _parse_json_response(response_text)
        service_name = result.get("service_name")
        confidence = result.get("confidence", 0.0)

        if service_name and service_name in service_ids:
            print(
                f"[ModelTraining] Extracted service: {service_name} (confidence: {confidence})"
            )
            return service_name
        else:
            print(f"[ModelTraining] Service not found or invalid: {service_name}")
            return None

    except Exception as e:
        print(f"[ModelTraining] Failed to extract service name: {e}")
        return None


def model_training_node(state: GlobalState) -> GlobalState:
    """Model Training Node - generates subtask for model training."""
    print(f"\n{'=' * 60}")
    print("[ModelTraining] Starting model training subtask generation...")
    print(f"{'=' * 60}")

    available_services = _load_available_services()
    if not available_services:
        print("[ModelTraining] No services available")
        state.merged_result["training_error"] = "No MCP services configured"
        print(f"{'=' * 60}")
        return state

    service_name = _extract_service_name(state, available_services)

    if not service_name:
        print("[ModelTraining] Could not determine target service from user input")
        state.merged_result["training_error"] = "Could not determine target service"
        print(f"{'=' * 60}")
        return state

    session_id = state.session_id or "default"
    input_dir = f"/data/sessions/{session_id}/input"

    task_content = (
        f"Retrain the model for {service_name}. Please call the training tool in the {service_name} service "
        f"using the data files in {input_dir} to complete the model training task."
    )

    subtask = SubTask(
        task_id=f"train_{service_name}_{session_id[:8]}",
        task_type=UserTaskType.EXECUTE_PLAN,
        content=task_content,
        dependencies=[],
    )

    state.subtasks = [subtask]
    state.merged_result["model_training_service"] = service_name

    print(f"[ModelTraining] Generated subtask:")
    print(f"  - task_id: {subtask.task_id}")
    print(f"  - service: {service_name}")

    print(f"{'=' * 60}")
    return state
    print(f"  - copied_files: {copied_files}")
    print(f"  - content: {task_content[:100]}...")

    print(f"{'=' * 60}")
    return state


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """解析 JSON 响应"""
    import re

    text = response_text.strip()

    # Remove thinking tags if present (GLM native thinking mode)
    thinking_patterns = [
        r"<think[^>]*>.*?</think\s*>",
        r"<thinking[^>]*>.*?</thinking\s*>",
        r"<reasoning[^>]*>.*?</reasoning\s*>",
    ]
    for pattern in thinking_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = None

    try:
        if text.startswith("{"):
            json_match = text
        else:
            json_block_patterns = [
                r"```json\s*(\{.*?\})\s*```",
                r"```\s*(\{.*?\})\s*```",
            ]
            for pattern in json_block_patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    json_match = matches[0]
                    break

            if not json_match:
                brace_start = text.find("{")
                if brace_start >= 0:
                    depth = 0
                    for i, char in enumerate(text[brace_start:], brace_start):
                        if char == "{":
                            depth += 1
                        elif char == "}":
                            depth -= 1
                            if depth == 0:
                                json_match = text[brace_start : i + 1]
                                break

        if json_match:
            return json.loads(json_match)
    except (json.JSONDecodeError, Exception) as e:
        print(f"[WARN] JSON 解析失败: {e}")
        print(f"[WARN] 原始响应: {text[:200]}...")

    return {
        "category": "general_qa",
        "confidence": 0.5,
        "reason": "解析失败，使用默认分类",
    }


def build_main_graph():
    """构建主图"""
    graph = StateGraph(GlobalState)

    # 添加节点
    graph.add_node("init", init_node)
    graph.add_node("classify", classify_node)
    graph.add_node("immunity", immunity_node)
    graph.add_node("task_decomposition", task_decomposition_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("result_evaluator", result_evaluator_node)
    graph.add_node("general_qa", general_qa_node)
    graph.add_node("extract_answer", extract_answer_node)
    graph.add_node("model_training", model_training_node)

    # 添加边
    graph.add_edge(START, "init")
    graph.add_edge("init", "classify")

    # 分类路由
    graph.add_conditional_edges(
        "classify",
        classify_router,
        {
            "immunity_branch": "immunity",
            "has_plan_branch": "task_decomposition",
            "general_qa_branch": "general_qa",
            "model_training_branch": "model_training",
        },
    )

    # Immunity branch flow
    graph.add_edge("immunity", "task_decomposition")
    graph.add_edge("task_decomposition", "orchestrator")
    graph.add_edge("orchestrator", "result_evaluator")
    graph.add_edge("result_evaluator", END)

    # General QA 分支流程
    graph.add_edge("general_qa", "extract_answer")
    graph.add_edge("extract_answer", END)

    # Model Training branch flow
    graph.add_edge("model_training", "orchestrator")
    graph.add_edge("orchestrator", "result_evaluator")
    graph.add_edge("result_evaluator", END)

    return graph.compile()


async def run_agent_async(user_input: str, **kwargs) -> GlobalState:
    """异步运行 agent"""
    graph = build_main_graph()

    initial_state = GlobalState(
        user_input=user_input,
        session_id=kwargs.get("session_id"),
        progress_callback=kwargs.get("progress_callback"),
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["session_id", "progress_callback"]
        },
    )

    result = await graph.ainvoke(initial_state)
    if isinstance(result, dict):
        return GlobalState(**result)
    return result


def run_agent(user_input: str, **kwargs) -> GlobalState:
    """同步运行 agent"""
    graph = build_main_graph()

    initial_state = GlobalState(
        user_input=user_input,
        session_id=kwargs.get("session_id"),
        progress_callback=kwargs.get("progress_callback"),
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["session_id", "progress_callback"]
        },
    )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, graph.ainvoke(initial_state))
                result = future.result(timeout=3600)
        else:
            result = loop.run_until_complete(graph.ainvoke(initial_state))
    except RuntimeError:
        result = asyncio.run(graph.ainvoke(initial_state))

    if isinstance(result, dict):
        return GlobalState(**result)
    return result


if __name__ == "__main__":
    import asyncio

    async def main():
        test_input = "请帮我分析 TCR 序列的功能特征"
        result = await run_agent_async(test_input)
        print(f"\n最终结果: {result.merged_result.get('final_answer', 'No result')}")

    asyncio.run(main())
