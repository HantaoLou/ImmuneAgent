"""
参数推断 + Executor 执行完整流程测试用例

测试完整流程：参数推断 → Executor 执行，包括：
1. 任务分解（粗分解 → 细分解 → 并行推断 → 参数推断）
2. Executor 执行（参数推断 → 任务执行 → 结果分析）
3. 完整的结果记录

运行方式：pytest tests/test_executor_parameter_inference_with_execution.py -v
"""

import os
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import uuid

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from nodes.subagents.task_decomposition.graph import (
    build_task_decomposition_subgraph,
    task_decomposition_input_mapper,
    task_decomposition_output_mapper,
    TaskDecompositionState,
    ParameterSourceType
)
from nodes.subagents.executor.graph import (
    build_executor_subgraph,
    executor_input_mapper,
    executor_output_mapper,
    ExecutorState,
    ExecutorTaskStatus
)
from state import GlobalState

# 导入LLM工厂（用于自动生成参数）
try:
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None


# ===================== 测试用例数据 =====================

# 预设参数配置（用于自动响应HITL请求）
PRESET_PARAMETERS = {
}

# 简单级别测试用例（用于完整流程测试）
EXECUTION_TEST_CASES = [
    {
        "name": "完整流程",
        "user_input": "抗体设计：我有一个针对H5N1的候选抗体序列，想评估其潜在广谱性和亲和力（如与group1族流感病毒）和可开发性。我应使用哪些工具来进行分析？请说明每一步的目的。",
        "execution_plan": None,
        "description": "简单任务，测试完整流程",
        "preset_parameters": PRESET_PARAMETERS  # 预设参数
    },
]


# ===================== 核心辅助函数 =====================

def run_full_decomposition_flow(user_input: str, execution_plan: str = None) -> Tuple[GlobalState, TaskDecompositionState]:
    """
    运行完整的任务分解流程：粗分解 → 细分解 → 并行推断 → 参数推断
    
    Args:
        user_input: 用户输入
        execution_plan: 执行计划（可选）
    
    Returns:
        (包含分解后任务的 GlobalState, 包含中间状态和参数推断结果的 TaskDecompositionState)
    """
    task_decomposition_subgraph = build_task_decomposition_subgraph()
    
    initial_state = GlobalState(
        user_input=user_input,
        execution_plan=execution_plan,
        sandbox_dir="./sandbox"
    )
    
    decomposition_input = task_decomposition_input_mapper(initial_state)
    decomposition_output = task_decomposition_subgraph.invoke(decomposition_input)
    final_state = task_decomposition_output_mapper(decomposition_output, initial_state)
    
    if isinstance(decomposition_output, dict):
        decomposition_state = TaskDecompositionState(**decomposition_output)
    else:
        decomposition_state = decomposition_output
    
    return final_state, decomposition_state


def generate_parameters_with_llm(task_id: str, task_description: str, missing_params: List[str], 
                                 preset_params: Dict = None) -> Dict[str, Any]:
    """
    使用LLM生成缺失的参数值
    
    Args:
        task_id: 任务ID
        task_description: 任务描述
        missing_params: 缺失的参数列表（格式：tool_name.param_name 或 param_name）
        preset_params: 预设参数字典
    
    Returns:
        参数字典
    """
    generated_params = {}
    preset_params = preset_params or {}
    
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        # 如果没有LLM，使用预设参数
        for param in missing_params:
            # 参数格式可能是 "tool_name.param_name" 或 "param_name"
            param_name = param.split('.')[-1] if '.' in param else param
            if param_name in preset_params:
                generated_params[param_name] = preset_params[param_name]
        return generated_params
    
    llm = create_reasoning_llm()
    if not llm:
        # LLM不可用，使用预设参数
        for param in missing_params:
            param_name = param.split('.')[-1] if '.' in param else param
            if param_name in preset_params:
                generated_params[param_name] = preset_params[param_name]
        return generated_params
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # 构建提示
        params_list = '\n'.join([f"- {p}" for p in missing_params])
        preset_list = '\n'.join([f"- {k}: {v}" for k, v in list(preset_params.items())[:10]])
        
        prompt = f"""
请为以下任务生成缺失的参数值：

任务ID: {task_id}
任务描述: {task_description}

缺失的参数:
{params_list}

可用的预设参数（可以参考）:
{preset_list}

请为每个缺失的参数生成合理的测试值。对于文件路径，使用测试路径（如 ./test_data/xxx）。对于枚举类型，选择合理的值。对于数值类型，使用合理的默认值。

返回JSON格式：
{{
    "parameters": {{
        "参数名1": "参数值1",
        "参数名2": "参数值2",
        ...
    }}
}}

注意：
1. 参数名应该是参数的实际名称（去掉工具名前缀，如 "tool_name.param_name" -> "param_name"）
2. 对于文件路径参数，使用测试路径
3. 对于布尔值，使用 true/false
4. 对于数值，使用合理的默认值
"""
        
        messages = [
            SystemMessage(content="你是一个专业的参数生成助手，能够根据任务描述生成合理的测试参数值。"),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # 解析JSON响应
        import re
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            generated_params = result.get("parameters", {})
        
    except Exception as e:
        print(f"  ⚠ LLM生成参数失败: {e}，使用预设参数")
    
    # 合并预设参数（优先使用预设参数）
    for param in missing_params:
        param_name = param.split('.')[-1] if '.' in param else param
        if param_name in preset_params and param_name not in generated_params:
            generated_params[param_name] = preset_params[param_name]
    
    return generated_params


def auto_resolve_hitl_requests(executor_state: ExecutorState, preset_params: Dict = None) -> Dict[str, Dict]:
    """
    自动解析HITL请求，生成参数响应
    
    Args:
        executor_state: ExecutorState
        preset_params: 预设参数字典
    
    Returns:
        HITL响应字典 {task_id: {parameters: {...}}}
    """
    responses = {}
    preset_params = preset_params or {}
    
    for task_id, request in executor_state.hitl_requests.items():
        if request.get("type") == "missing_parameters":
            missing_params = request.get("missing_parameters", [])
            task_description = request.get("task_description", "")
            
            # 获取任务信息
            task = next((t for t in executor_state.subtasks if t.task_id == task_id), None)
            if task:
                task_description = task.content
            
            # 生成参数
            generated_params = generate_parameters_with_llm(
                task_id, task_description, missing_params, preset_params
            )
            
            if generated_params:
                responses[task_id] = {
                    "parameters": generated_params,
                    "continue": True
                }
                print(f"  ✓ 为任务 {task_id} 自动生成 {len(generated_params)} 个参数")
    
    return responses


def run_executor_flow(global_state: GlobalState, preset_params: Dict = None, 
                     auto_resolve_hitl: bool = True) -> Tuple[GlobalState, ExecutorState, Dict]:
    """
    运行 Executor 子图执行流程
    
    Args:
        global_state: 包含分解后任务的 GlobalState
        preset_params: 预设参数字典（用于自动响应HITL）
        auto_resolve_hitl: 是否自动解析HITL请求
    
    Returns:
        (更新后的 GlobalState, ExecutorState, execution_info)
        execution_info 包含: interrupted, interrupt_data, execution_errors, execution_logs
    """
    executor_subgraph = build_executor_subgraph()
    
    # 构建 Executor 输入
    executor_input = executor_input_mapper(global_state)
    
    # 调试：检查传递给 executor 的任务数量（包括并行任务组）
    total_tasks_in_global = len(global_state.subtasks)
    parallel_tasks_count = 0
    for group_id, group in global_state.parallel_task_groups.items():
        if hasattr(group, 'subtasks'):
            parallel_tasks_count += len(group.subtasks)
        elif isinstance(group, dict):
            parallel_tasks_count += len(group.get('subtasks', []))
    
    total_expected_tasks = total_tasks_in_global + parallel_tasks_count
    
    print(f"  [DEBUG] global_state 任务统计:")
    print(f"    - 串行任务: {total_tasks_in_global} 个")
    print(f"    - 并行任务组: {len(global_state.parallel_task_groups)} 个（包含 {parallel_tasks_count} 个任务）")
    print(f"    - 总任务数: {total_expected_tasks} 个")
    print(f"  [DEBUG] executor_input 初始任务数量: {len(executor_input.subtasks)}")
    print(f"  [DEBUG] executor_input 并行任务组数量: {len(executor_input.parallel_task_groups)}")
    
    if len(executor_input.subtasks) != total_expected_tasks:
        print(f"  ⚠ 警告: executor_input 初始只有 {len(executor_input.subtasks)} 个任务，但应该包含 {total_expected_tasks} 个任务（包括并行任务组）")
        print(f"  ⚠ 注意: executor 的 initialize_tasks_node 应该会展开并行任务组")
    
    # 执行 Executor 子图
    # 使用唯一的 thread_id 避免冲突
    thread_id = f"test_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    execution_info = {
        'interrupted': False,
        'interrupt_data': None,
        'execution_errors': [],
        'execution_logs': [],
        'hitl_requests': []
    }
    
    executor_state = None
    input_data = executor_input.model_dump(exclude={'parent_state'}, mode='json')
    
    try:
        # 使用 stream() 来检测中断和错误
        execution_info['execution_logs'].append("开始执行 Executor 子图...")
        
        for chunk in executor_subgraph.stream(input_data, config=config):
            # 检查是否有中断
            if "__interrupt__" in chunk:
                execution_info['interrupted'] = True
                execution_info['interrupt_data'] = chunk["__interrupt__"]
                execution_info['execution_logs'].append(f"检测到中断: {execution_info['interrupt_data']}")
                
                # 提取 HITL 请求信息
                if isinstance(execution_info['interrupt_data'], dict):
                    interrupt_type = execution_info['interrupt_data'].get('type', 'unknown')
                    if interrupt_type == 'missing_parameters':
                        requests = execution_info['interrupt_data'].get('requests', [])
                        execution_info['hitl_requests'].extend(requests)
                        execution_info['execution_logs'].append(f"HITL请求: 需要用户提供参数，涉及 {len(requests)} 个任务")
                
                # 获取当前状态（中断前的状态）
                for key, value in chunk.items():
                    if key != "__interrupt__":
                        if isinstance(value, dict):
                            try:
                                executor_state = ExecutorState.model_validate(value)
                            except:
                                pass
                
                # 如果启用了自动解析HITL，尝试自动响应
                if auto_resolve_hitl and executor_state:
                    execution_info['execution_logs'].append("尝试自动解析HITL请求...")
                    hitl_responses = auto_resolve_hitl_requests(executor_state, preset_params)
                    
                    if hitl_responses:
                        execution_info['execution_logs'].append(f"自动生成 {len(hitl_responses)} 个HITL响应")
                        
                        # 更新executor_state的HITL响应
                        for task_id, response in hitl_responses.items():
                            executor_state.hitl_responses[task_id] = response
                            
                            # 更新任务参数
                            if task_id in executor_state.task_results:
                                result = executor_state.task_results[task_id]
                                if "parameters" in response:
                                    result.parameters.update(response["parameters"])
                                    # 从missing_parameters中移除已提供的参数
                                    provided_params = set(response["parameters"].keys())
                                    result.missing_parameters = [
                                        p for p in result.missing_parameters
                                        if not any(
                                            p == param_name or p.endswith(f".{param_name}")
                                            for param_name in provided_params
                                        )
                                    ]
                                
                                # 如果所有必需参数都已提供，标记为就绪
                                if not result.missing_parameters:
                                    executor_state.task_status_map[task_id] = ExecutorTaskStatus.READY
                                    execution_info['execution_logs'].append(f"任务 {task_id} 已获得所有必需参数，标记为就绪")
                        
                        # 尝试恢复执行
                        try:
                            try:
                                from langgraph.types import Command
                                INTERRUPT_AVAILABLE = True
                            except ImportError:
                                INTERRUPT_AVAILABLE = False
                                Command = None
                            
                            if INTERRUPT_AVAILABLE and Command is not None:
                                # 构建恢复命令
                                resume_data = {
                                    "type": "response_parameters",
                                    "responses": hitl_responses
                                }
                                resume_command = Command(resume=resume_data)
                                
                                execution_info['execution_logs'].append("使用自动生成的参数恢复执行...")
                                
                                # 继续执行
                                for resume_chunk in executor_subgraph.stream(resume_command, config=config):
                                    if "__interrupt__" in resume_chunk:
                                        # 再次中断，记录新的中断信息
                                        execution_info['interrupted'] = True
                                        execution_info['interrupt_data'] = resume_chunk["__interrupt__"]
                                        execution_info['execution_logs'].append("恢复执行后再次中断")
                                        break
                                    else:
                                        for node_name, node_output in resume_chunk.items():
                                            if isinstance(node_output, dict):
                                                try:
                                                    executor_state = ExecutorState.model_validate(node_output)
                                                    execution_info['execution_logs'].append(f"节点 {node_name} 执行完成（恢复后）")
                                                except:
                                                    pass
                                
                                # 如果恢复执行后没有再次中断，继续获取最终结果
                                if not execution_info.get('interrupted'):
                                    execution_info['execution_logs'].append("恢复执行完成，获取最终结果...")
                                    final_output = executor_subgraph.invoke(resume_command, config=config)
                                    if isinstance(final_output, dict):
                                        executor_state = ExecutorState.model_validate(final_output)
                                    else:
                                        executor_state = final_output
                                    execution_info['execution_logs'].append("Executor 子图执行完成（自动恢复后）")
                        except Exception as resume_error:
                            execution_info['execution_logs'].append(f"自动恢复执行失败: {resume_error}")
                            execution_info['execution_errors'].append({
                                'error_type': type(resume_error).__name__,
                                'error_message': str(resume_error),
                                'traceback': None
                            })
                
                break
            else:
                # 正常的状态更新，记录节点执行情况
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, dict):
                        try:
                            executor_state = ExecutorState.model_validate(node_output)
                            execution_info['execution_logs'].append(f"节点 {node_name} 执行完成")
                        except:
                            pass
        
        # 如果没有中断，获取最终结果
        if not execution_info['interrupted'] and executor_state is None:
            execution_info['execution_logs'].append("使用 invoke 获取最终结果...")
            executor_output = executor_subgraph.invoke(input_data, config=config)
            if isinstance(executor_output, dict):
                executor_state = ExecutorState.model_validate(executor_output)
            else:
                executor_state = executor_output
            execution_info['execution_logs'].append("Executor 子图执行完成")
        
    except Exception as e:
        error_msg = str(e)
        error_traceback = None
        try:
            import traceback
            error_traceback = traceback.format_exc()
        except:
            pass
        
        execution_info['execution_errors'].append({
            'error_type': type(e).__name__,
            'error_message': error_msg,
            'traceback': error_traceback
        })
        
        execution_info['execution_logs'].append(f"执行异常: {error_msg}")
        
        # 如果是 interrupt 异常，这是正常行为
        if "interrupt" in error_msg.lower() or "GraphInterrupt" in type(e).__name__:
            execution_info['interrupted'] = True
            execution_info['interrupt_data'] = getattr(e, 'value', None) or error_msg
            execution_info['execution_logs'].append("检测到 interrupt 异常（正常行为）")
        else:
            # 其他异常，记录详细信息
            execution_info['execution_logs'].append(f"执行失败: {error_traceback or error_msg}")
    
    # 如果 executor_state 仍然为 None，尝试从输入创建
    if executor_state is None:
        try:
            executor_state = ExecutorState.model_validate(input_data)
        except:
            # 如果还是失败，创建一个空状态用于记录
            executor_state = ExecutorState.model_construct(
                subtasks=global_state.subtasks,
                parallel_task_groups=global_state.parallel_task_groups,
                sandbox_dir=global_state.sandbox_dir
            )
    
    # 映射回 GlobalState
    updated_global_state = executor_output_mapper(executor_state, global_state)
    
    return updated_global_state, executor_state, execution_info


def extract_parameter_inference_results(decomposition_state: TaskDecompositionState) -> Tuple[Dict, Dict, Dict]:
    """
    从 TaskDecompositionState 中提取参数推断结果
    
    Returns:
        (task_results_dict, task_info_dict, stats_dict)
    """
    task_results_dict = {}
    task_info_dict = {}
    determined_count = 0
    from_task_count = 0
    user_required_count = 0
    
    # 获取所有任务（包括并行任务组中的任务）
    all_tasks = decomposition_state.subtasks + [
        task for group in decomposition_state.parallel_task_groups.values()
        for task in group.subtasks
    ]
    
    for task in all_tasks:
        task_id = task.task_id
        task_info_dict[task_id] = {
            'content': task.content,
            'task_type': task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)
        }
        
        if task_id in decomposition_state.parameter_inference_results:
            inference_result = decomposition_state.parameter_inference_results[task_id]
            params_dict = {}
            
            for param_name, param_result in inference_result.parameters.items():
                if hasattr(param_result, 'model_dump'):
                    param_data = param_result.model_dump()
                elif isinstance(param_result, dict):
                    param_data = param_result
                else:
                    param_data = {
                        'source_type': param_result.source_type.value if hasattr(param_result.source_type, 'value') else str(param_result.source_type),
                        'value': getattr(param_result, 'value', None),
                        'source_task_id': getattr(param_result, 'source_task_id', None),
                        'source_output_key': getattr(param_result, 'source_output_key', None),
                        'user_prompt': getattr(param_result, 'user_prompt', None),
                        'reason': getattr(param_result, 'reason', None)
                    }
                
                params_dict[param_name] = param_data
                
                source_type = param_data.get('source_type', 'unknown')
                if source_type == ParameterSourceType.DETERMINED.value:
                    determined_count += 1
                elif source_type == ParameterSourceType.FROM_TASK.value:
                    from_task_count += 1
                elif source_type == ParameterSourceType.USER_REQUIRED.value:
                    user_required_count += 1
            
            task_results_dict[task_id] = {
                'parameters': params_dict,
                'tool_name': inference_result.tool_name
            }
    
    stats = {
        'determined_count': determined_count,
        'from_task_count': from_task_count,
        'user_required_count': user_required_count,
        'total_count': determined_count + from_task_count + user_required_count
    }
    
    return task_results_dict, task_info_dict, stats


def extract_execution_results(executor_state: ExecutorState, execution_info: Dict = None, 
                              global_state: GlobalState = None) -> Dict:
    """
    从 ExecutorState 中提取执行结果
    
    Args:
        executor_state: ExecutorState
        execution_info: 执行信息（可选）
        global_state: GlobalState（可选，用于获取完整任务列表）
    
    Returns:
        执行结果字典
    """
    execution_results = {
        'total_tasks': executor_state.total_tasks,
        'completed_count': executor_state.completed_count,
        'failed_count': executor_state.failed_count,
        'task_results': {},
        'task_status_map': {},
        'hitl_requests': {},
        'hitl_responses': {},
        'execution_info': execution_info or {}
    }
    
    # 提取任务状态映射
    for task_id, status in executor_state.task_status_map.items():
        execution_results['task_status_map'][task_id] = status.value if hasattr(status, 'value') else str(status)
    
    # 提取 HITL 请求
    for task_id, request in executor_state.hitl_requests.items():
        execution_results['hitl_requests'][task_id] = {
            'type': request.get('type', 'unknown'),
            'message': request.get('message', ''),
            'missing_parameters': request.get('missing_parameters', [])
        }
    
    # 提取 HITL 响应
    for task_id, response in executor_state.hitl_responses.items():
        execution_results['hitl_responses'][task_id] = response
    
    # 提取任务执行结果
    for task_id, result in executor_state.task_results.items():
        execution_results['task_results'][task_id] = {
            'status': result.status.value if hasattr(result.status, 'value') else str(result.status),
            'execution_mode': result.execution_mode,
            'parameters': result.parameters,
            'missing_parameters': result.missing_parameters,
            'code': result.code,
            'output': str(result.output)[:500] if result.output else None,  # 限制输出长度
            'error': result.error,
            'error_category': result.error_category.value if result.error_category else None,
            'retry_count': result.retry_count,
            'execution_time': result.execution_time,
            'confidence_score': result.confidence_score,
            'failure_analysis': result.failure_analysis,
            'suggestions': result.suggestions,
            'result_satisfied': result.result_satisfied,
            'user_continue': result.user_continue
        }
    
    # 获取所有任务列表（优先使用 global_state，如果没有则使用 executor_state）
    all_tasks = []
    if global_state:
        # 从 global_state 获取所有任务（包括并行任务组中的任务）
        all_tasks = list(global_state.subtasks)
        # 从并行任务组中提取所有任务
        for group_id, group in global_state.parallel_task_groups.items():
            if hasattr(group, 'subtasks') and group.subtasks:
                for task in group.subtasks:
                    # 检查任务是否已经在列表中（避免重复）
                    if not any(t.task_id == task.task_id for t in all_tasks):
                        all_tasks.append(task)
        print(f"  [DEBUG] 使用 global_state 获取任务列表，共 {len(all_tasks)} 个任务（串行: {len(global_state.subtasks)} 个，并行组: {len(global_state.parallel_task_groups)} 个）")
    else:
        # 从 executor_state 获取所有任务（executor 应该已经展开了并行任务组）
        all_tasks = executor_state.subtasks
        print(f"  [DEBUG] 使用 executor_state 获取任务列表，共 {len(all_tasks)} 个任务")
    
    # 对于没有执行结果的任务，也记录其状态和详细错误原因
    for task in all_tasks:
        if task.task_id not in execution_results['task_results']:
            status = execution_results['task_status_map'].get(task.task_id, '未初始化')
            
            # 分析任务未执行的具体原因
            error_reasons = []
            failure_analysis_parts = []
            suggestions_list = []
            
            # 1. 检查任务是否在executor中被初始化
            if task.task_id not in executor_state.task_status_map:
                error_reasons.append("任务未在executor中初始化")
                failure_analysis_parts.append(f"任务 {task.task_id} 没有被executor的initialize_tasks_node处理。")
                suggestions_list.append("检查executor_input_mapper是否正确传递了所有任务（包括并行任务组中的任务）")
            
            # 2. 检查任务状态
            if status == '未初始化':
                error_reasons.append("任务状态未初始化")
                failure_analysis_parts.append(f"任务 {task.task_id} 的状态为'未初始化'，说明它没有被executor正确处理。")
            elif status == ExecutorTaskStatus.WAITING_DEPENDENCY.value:
                # 检查依赖任务的状态
                if task.dependencies:
                    dep_statuses = []
                    for dep_id in task.dependencies:
                        dep_status = execution_results['task_status_map'].get(dep_id, '未知')
                        dep_result = execution_results['task_results'].get(dep_id, {})
                        dep_final_status = dep_result.get('status', dep_status)
                        dep_statuses.append(f"{dep_id}: {dep_final_status}")
                    error_reasons.append(f"等待依赖任务完成（依赖: {', '.join(task.dependencies)}）")
                    failure_analysis_parts.append(f"任务 {task.task_id} 的依赖任务状态: {', '.join(dep_statuses)}")
                    suggestions_list.append("检查依赖任务是否成功完成，如果依赖任务失败，当前任务可能无法执行")
            elif status == ExecutorTaskStatus.WAITING_HITL_PARAMS.value:
                hitl_request = execution_results['hitl_requests'].get(task.task_id, {})
                missing_params = hitl_request.get('missing_parameters', [])
                error_reasons.append(f"等待用户提供参数（缺失参数: {len(missing_params)} 个）")
                failure_analysis_parts.append(f"任务 {task.task_id} 需要用户提供以下参数: {', '.join(missing_params)}")
                suggestions_list.append("提供缺失的参数或检查自动参数生成逻辑")
            elif status == ExecutorTaskStatus.WAITING_HITL_CONFIRM.value:
                error_reasons.append("等待用户确认执行结果")
                failure_analysis_parts.append(f"任务 {task.task_id} 的执行结果需要用户确认")
                suggestions_list.append("检查任务执行结果是否满足要求")
            elif status == ExecutorTaskStatus.READY.value:
                error_reasons.append("任务处于就绪状态但未被执行")
                failure_analysis_parts.append(f"任务 {task.task_id} 已就绪但未被executor执行，可能是执行流程提前结束或遇到错误")
                suggestions_list.append("检查executor的执行流程是否正常完成，查看执行日志了解详情")
            
            # 3. 检查是否在并行任务组中
            in_parallel_group = False
            for group_id, group in (global_state.parallel_task_groups.items() if global_state else {}):
                if hasattr(group, 'subtasks') and any(t.task_id == task.task_id for t in group.subtasks):
                    in_parallel_group = True
                    error_reasons.append(f"任务在并行任务组 {group_id} 中")
                    failure_analysis_parts.append(f"任务 {task.task_id} 属于并行任务组 {group_id}，可能没有被正确展开到executor的subtasks中")
                    suggestions_list.append("检查executor的initialize_tasks_node是否正确展开了并行任务组")
                    break
            
            # 4. 检查executor处理的任务数量
            if len(executor_state.subtasks) < len(all_tasks):
                error_reasons.append(f"executor只处理了 {len(executor_state.subtasks)} 个任务，而总任务数为 {len(all_tasks)}")
                failure_analysis_parts.append(f"executor的subtasks中只有 {len(executor_state.subtasks)} 个任务，但实际应该有 {len(all_tasks)} 个任务")
                suggestions_list.append("检查executor_input_mapper是否正确传递了所有任务，包括并行任务组中的任务")
            
            # 构建详细的错误信息
            error_message = f"任务未被执行。原因分析：\n"
            error_message += f"1. 状态: {status}\n"
            if error_reasons:
                error_message += f"2. 具体原因:\n"
                for i, reason in enumerate(error_reasons, 1):
                    error_message += f"   {i}. {reason}\n"
            
            # 构建失败分析
            failure_analysis = "\n".join(failure_analysis_parts) if failure_analysis_parts else None
            
            execution_results['task_results'][task.task_id] = {
                'status': status,
                'execution_mode': '',
                'parameters': {},
                'missing_parameters': execution_results['hitl_requests'].get(task.task_id, {}).get('missing_parameters', []),
                'code': None,
                'output': None,
                'error': error_message,
                'error_category': 'SYSTEM_ERROR' if status == '未初始化' else 'PARAMETER_ERROR' if status == ExecutorTaskStatus.WAITING_HITL_PARAMS.value else None,
                'retry_count': 0,
                'execution_time': 0.0,
                'confidence_score': None,
                'failure_analysis': failure_analysis,
                'suggestions': suggestions_list,
                'result_satisfied': None,
                'user_continue': None
            }
            # 如果任务状态未初始化，也添加到状态映射中
            if task.task_id not in execution_results['task_status_map']:
                execution_results['task_status_map'][task.task_id] = '未初始化'
    
    # 更新总任务数（使用实际任务数）
    execution_results['total_tasks'] = len(all_tasks)
    
    return execution_results


def _load_tools_params_table() -> Dict[str, Dict[str, Any]]:
    """加载工具参数表"""
    tools_params_path = Path(__file__).parent.parent / "config" / "tools_params_table.json"
    if not tools_params_path.exists():
        return {}
    
    try:
        with open(tools_params_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tools_params_map = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for tool_name, params_info in item.items():
                        tools_params_map[tool_name] = params_info
        elif isinstance(data, dict):
            tools_params_map = data
        
        return tools_params_map
    except Exception as e:
        print(f"⚠ 加载工具参数表失败: {e}")
        return {}


def get_parameter_details(tool_name: str, param_name: str) -> Dict[str, Any]:
    """
    获取参数的详细信息
    
    Args:
        tool_name: 工具名称
        param_name: 参数名称（可能是 "args" 或具体参数名，也可能是 "tool_name.param_name" 格式）
    
    Returns:
        参数字典，包含 name, description, demo, type 等信息
    """
    tools_params_map = _load_tools_params_table()
    
    # 解析参数名（可能是 "tool_name.param_name" 格式）
    if '.' in param_name:
        parts = param_name.split('.', 1)
        actual_tool_name = parts[0]
        actual_param_name = parts[1]
    else:
        actual_tool_name = tool_name
        actual_param_name = param_name
    
    # 首先尝试精确匹配
    tool_params = tools_params_map.get(actual_tool_name, {})
    
    # 如果精确匹配失败，尝试模糊匹配
    if not tool_params:
        # 尝试查找包含工具名的键（例如：prepare_bindcraft_config -> bindcraft_prepare_bindcraft_config）
        for key in tools_params_map.keys():
            # 检查是否以工具名结尾，或者工具名是键的一部分
            if key.endswith(actual_tool_name) or actual_tool_name in key:
                tool_params = tools_params_map.get(key, {})
                if tool_params:
                    # 找到匹配的工具，使用找到的键
                    break
    
    if not tool_params:
        # 如果还是找不到，返回基本信息
        return {'name': actual_param_name, 'description': '', 'demo': '', 'type': '', 'tool_not_found': True}
    
    input_params = tool_params.get('input_params', [])
    
    # 如果参数名是 "args" 或 "args: "，返回所有参数的汇总信息
    if actual_param_name.strip() in ['args', 'args:', 'args: ']:
        all_params_info = []
        for param in input_params:
            param_info = {
                'name': param.get('name', ''),
                'description': param.get('description', ''),
                'type': param.get('type', ''),
                'demo': param.get('demo', param.get('deme', ''))  # 兼容拼写错误
            }
            if param_info['name']:
                all_params_info.append(param_info)
        
        if all_params_info:
            # 返回所有参数的汇总信息
            param_details_list = []
            for p in all_params_info:
                detail = p['name']
                if p.get('type'):
                    detail += f" ({p['type']})"
                if p.get('description'):
                    detail += f": {p['description']}"
                param_details_list.append(detail)
            
            return {
                'name': 'args',
                'description': f"额外参数，包含以下 {len(all_params_info)} 个参数: {', '.join([p['name'] for p in all_params_info if p['name']])}",
                'all_params': all_params_info,
                'param_count': len(all_params_info),
                'param_details': param_details_list
            }
        else:
            return {'name': 'args', 'description': '额外参数（具体参数未定义）', 'demo': '', 'type': ''}
    
    # 查找具体参数
    for param in input_params:
        if param.get('name', '').strip() == actual_param_name.strip():
            return {
                'name': param.get('name', actual_param_name),
                'description': param.get('description', ''),
                'type': param.get('type', ''),
                'demo': param.get('demo', param.get('deme', ''))  # 兼容拼写错误
            }
    
    # 如果找不到，返回基本信息
    return {'name': actual_param_name, 'description': '', 'demo': '', 'type': ''}


def format_parameter_display(tool_name: str, param_name: str) -> str:
    """
    格式化参数显示，如果是args则显示详细信息
    
    Args:
        tool_name: 工具名称
        param_name: 参数名称（可能是 "tool_name.param_name" 格式）
    
    Returns:
        格式化后的参数字符串
    """
    # 解析参数名（可能是 "tool_name.param_name" 格式）
    if '.' in param_name:
        parts = param_name.split('.', 1)
        actual_tool_name = parts[0]
        actual_param_name = parts[1]
    else:
        actual_tool_name = tool_name
        actual_param_name = param_name
    
    # 获取参数详细信息
    param_details = get_parameter_details(actual_tool_name, actual_param_name)
    
    # 如果是args且有详细信息，显示所有参数
    if actual_param_name.strip() in ['args', 'args:', 'args: ']:
        if param_details.get('param_details'):
            # 显示所有参数的详细信息
            return f"{param_name}: [{'; '.join(param_details['param_details'])}]"
        elif param_details.get('all_params'):
            # 只显示参数名列表
            param_names = [p['name'] for p in param_details['all_params'] if p.get('name')]
            if param_names:
                return f"{param_name}: [{', '.join(param_names)}]"
        if param_details.get('description'):
            return f"{param_name}: {param_details['description']}"
        else:
            return param_name
    
    # 如果是具体参数，显示参数名、类型和描述
    display_parts = [param_name]
    if param_details.get('type'):
        display_parts.append(f"类型: {param_details['type']}")
    if param_details.get('description'):
        display_parts.append(f"描述: {param_details['description']}")
    elif param_details.get('type'):
        # 如果没有描述但有类型，至少显示类型
        return f"{param_name} (类型: {param_details['type']})"
    
    if len(display_parts) > 1:
        return f"{display_parts[0]} ({', '.join(display_parts[1:])})"
    else:
        return param_name


def safe_get_raw_tasks(decomposition_state: TaskDecompositionState) -> List[Dict]:
    """安全地获取 raw_tasks，处理各种可能的类型"""
    if not hasattr(decomposition_state, 'raw_tasks') or not decomposition_state.raw_tasks:
        return []
    
    raw_tasks = decomposition_state.raw_tasks
    
    if isinstance(raw_tasks, list):
        return raw_tasks
    
    if isinstance(raw_tasks, str):
        try:
            parsed = json.loads(raw_tasks)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
    
    return []


def prepare_log_data(decomposition_state: TaskDecompositionState, 
                     task_results_dict: Dict, 
                     task_info_dict: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    准备日志数据
    
    Returns:
        (coarse_decomposition, fine_decomposition, parameter_inference)
    """
    from nodes.subagents.task_decomposition.tool_categorizer import get_service_summary_by_id
    
    # 粗分解数据
    service_summaries = {}
    for service_id in decomposition_state.required_service_ids:
        summary = get_service_summary_by_id(service_id)
        if summary:
            service_summaries[service_id] = summary
    
    coarse_decomposition = {
        'required_service_ids': decomposition_state.required_service_ids,
        'filtered_tools_count': len(decomposition_state.filtered_tools),
        'service_summaries': service_summaries
    }
    
    # 细分解数据
    fine_decomposition = {
        'raw_tasks': safe_get_raw_tasks(decomposition_state),
        'decomposition_summary': getattr(decomposition_state, 'decomposition_summary', None)
    }
    
    # 参数推断数据
    parameter_inference = {
        'task_results': task_results_dict,
        'task_info': task_info_dict,
        'summary': getattr(decomposition_state, 'parameter_inference_summary', None)
    }
    
    return coarse_decomposition, fine_decomposition, parameter_inference


# ===================== 日志记录功能 =====================

def save_complete_test_log(test_case_name: str, execution_plan: Optional[str], 
                          coarse_decomposition: Dict, fine_decomposition: Dict,
                          parameter_inference: Dict, execution_results: Dict):
    """保存完整的测试日志（包括参数推断和执行结果）"""
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"complete_flow_{test_case_name}_{timestamp}.md"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"# 完整流程测试日志（参数推断 + Executor 执行）\n\n")
        f.write(f"**测试用例**: {test_case_name}\n\n")
        f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 1. 计划
        f.write("## 1. 执行计划\n\n")
        if execution_plan:
            f.write(f"```\n{execution_plan}\n```\n\n")
        else:
            f.write("无执行计划\n\n")
        f.write("---\n\n")
        
        # 2. 粗分解结果
        f.write("## 2. 粗分解结果\n\n")
        f.write(f"**所需服务ID**: {', '.join(coarse_decomposition.get('required_service_ids', []))}\n\n")
        f.write(f"**筛选后工具数量**: {coarse_decomposition.get('filtered_tools_count', 0)}\n\n")
        if coarse_decomposition.get('service_summaries'):
            f.write("**服务摘要**:\n\n")
            for service_id, summary in coarse_decomposition['service_summaries'].items():
                f.write(f"- **{service_id}**: {summary}\n")
            f.write("\n")
        f.write("---\n\n")
        
        # 3. 细分解结果
        f.write("## 3. 细分解结果\n\n")
        raw_tasks = fine_decomposition.get('raw_tasks', [])
        f.write(f"**任务数量**: {len(raw_tasks)}\n\n")
        if raw_tasks:
            f.write("### 任务列表\n\n")
            for i, task in enumerate(raw_tasks, 1):
                f.write(f"#### 任务 {i}: {task.get('task_id', f'task_{i}')}\n\n")
                f.write(f"**描述**: {task.get('description', task.get('name', task.get('content', '')))}\n\n")
                
                tools = task.get('tools', [])
                if tools:
                    tool_names = []
                    for tool in tools:
                        if isinstance(tool, str):
                            tool_names.append(tool)
                        elif isinstance(tool, dict):
                            tool_names.append(tool.get('tool_name', tool.get('name', '')))
                    f.write(f"**工具**: {', '.join(tool_names) if tool_names else '无'}\n\n")
                
                deps = task.get('dependencies', [])
                if deps:
                    f.write(f"**依赖**: {', '.join(deps)}\n\n")
                
                f.write("\n")
        
        if fine_decomposition.get('decomposition_summary'):
            f.write("### 分解摘要\n\n")
            f.write(f"{fine_decomposition['decomposition_summary']}\n\n")
        f.write("---\n\n")
        
        # 4. 参数推断结果
        f.write("## 4. 参数推断结果\n\n")
        inference_results = parameter_inference.get('task_results', {})
        f.write(f"**推断任务数**: {len(inference_results)}\n\n")
        
        total_determined = 0
        total_from_task = 0
        total_user_required = 0
        
        for task_id, result in inference_results.items():
            f.write(f"### 任务 {task_id}\n\n")
            
            task_info = parameter_inference.get('task_info', {}).get(task_id, {})
            if task_info.get('content'):
                f.write(f"**任务描述**: {task_info['content'][:100]}...\n\n")
            
            tool_name = result.get('tool_name')
            if tool_name:
                f.write(f"**工具**: {tool_name}\n\n")
            
            params = result.get('parameters', {})
            if params:
                f.write(f"**参数推断结果** ({len(params)} 个):\n\n")
                for param_name, param_data in params.items():
                    # 获取参数的详细信息（如果是args，显示所有参数）
                    param_details = get_parameter_details(tool_name or '', param_name)
                    param_display_name = param_name
                    
                    # 检查是否是args参数
                    is_args_param = param_name.strip() in ['args', 'args:', 'args: ']
                    
                    # 如果是args且有详细信息，显示完整的参数列表
                    if is_args_param:
                        if param_details.get('param_details'):
                            # 显示args参数的详细信息
                            f.write(f"- `{param_display_name}`: 额外参数，包含以下 {param_details.get('param_count', 0)} 个参数：\n")
                            for param_detail in param_details['param_details']:
                                f.write(f"  - {param_detail}\n")
                            # 继续显示参数来源信息
                            if isinstance(param_data, dict):
                                source_type = param_data.get('source_type', 'unknown')
                            else:
                                source_type = param_data.source_type.value if hasattr(param_data, 'source_type') else 'unknown'
                            
                            if source_type == ParameterSourceType.DETERMINED.value:
                                value = param_data.get('value') if isinstance(param_data, dict) else param_data.value
                                value_str = str(value)
                                if len(value_str) > 100:
                                    value_str = value_str[:100] + "..."
                                f.write(f"  → 参数值: {value_str} **[确定值]** ✓\n")
                                total_determined += 1
                            elif source_type == ParameterSourceType.FROM_TASK.value:
                                source_task = param_data.get('source_task_id') if isinstance(param_data, dict) else param_data.source_task_id
                                source_key = param_data.get('source_output_key') if isinstance(param_data, dict) else param_data.source_output_key
                                key_info = f" (输出键: {source_key})" if source_key else ""
                                f.write(f"  → 来自任务 `{source_task}`{key_info} **[任务结果]** ✓\n")
                                total_from_task += 1
                            elif source_type == ParameterSourceType.USER_REQUIRED.value:
                                prompt = param_data.get('user_prompt') if isinstance(param_data, dict) else param_data.user_prompt
                                f.write(f"  → {prompt} **[需要用户提供]** ✓\n")
                                total_user_required += 1
                            else:
                                f.write(f"  → 未知类型\n")
                        elif param_details.get('tool_not_found'):
                            # 工具未在参数表中找到
                            if isinstance(param_data, dict):
                                source_type = param_data.get('source_type', 'unknown')
                            else:
                                source_type = param_data.source_type.value if hasattr(param_data, 'source_type') else 'unknown'
                            
                            if source_type == ParameterSourceType.USER_REQUIRED.value:
                                prompt = param_data.get('user_prompt') if isinstance(param_data, dict) else param_data.user_prompt
                                f.write(f"- `{param_display_name}`: {prompt}（工具 `{tool_name}` 未在参数表中找到，无法显示具体参数列表） **[需要用户提供]** ✓\n")
                                total_user_required += 1
                            else:
                                f.write(f"- `{param_display_name}`: 额外参数（工具 `{tool_name}` 未在参数表中找到，无法显示具体参数列表）\n")
                        elif param_details.get('all_params'):
                            # 有参数但格式不对，至少显示参数名
                            param_names = [p['name'] for p in param_details['all_params'] if p.get('name')]
                            if param_names:
                                f.write(f"- `{param_display_name}`: 额外参数，包含以下参数: {', '.join(param_names)}\n")
                                # 继续显示参数来源信息
                                if isinstance(param_data, dict):
                                    source_type = param_data.get('source_type', 'unknown')
                                else:
                                    source_type = param_data.source_type.value if hasattr(param_data, 'source_type') else 'unknown'
                                
                                if source_type == ParameterSourceType.USER_REQUIRED.value:
                                    prompt = param_data.get('user_prompt') if isinstance(param_data, dict) else param_data.user_prompt
                                    f.write(f"  → {prompt} **[需要用户提供]** ✓\n")
                                    total_user_required += 1
                                elif source_type == ParameterSourceType.FROM_TASK.value:
                                    source_task = param_data.get('source_task_id') if isinstance(param_data, dict) else param_data.source_task_id
                                    source_key = param_data.get('source_output_key') if isinstance(param_data, dict) else param_data.source_output_key
                                    key_info = f" (输出键: {source_key})" if source_key else ""
                                    f.write(f"  → 来自任务 `{source_task}`{key_info} **[任务结果]** ✓\n")
                                    total_from_task += 1
                            else:
                                f.write(f"- `{param_display_name}`: 额外参数（具体参数未定义）\n")
                        else:
                            # 没有参数信息，只显示来源
                            if isinstance(param_data, dict):
                                source_type = param_data.get('source_type', 'unknown')
                            else:
                                source_type = param_data.source_type.value if hasattr(param_data, 'source_type') else 'unknown'
                            
                            if source_type == ParameterSourceType.USER_REQUIRED.value:
                                prompt = param_data.get('user_prompt') if isinstance(param_data, dict) else param_data.user_prompt
                                f.write(f"- `{param_display_name}`: {prompt} **[需要用户提供]** ✓\n")
                                total_user_required += 1
                            elif source_type == ParameterSourceType.FROM_TASK.value:
                                source_task = param_data.get('source_task_id') if isinstance(param_data, dict) else param_data.source_task_id
                                source_key = param_data.get('source_output_key') if isinstance(param_data, dict) else param_data.source_output_key
                                key_info = f" (输出键: {source_key})" if source_key else ""
                                f.write(f"- `{param_display_name}`: 来自任务 `{source_task}`{key_info} **[任务结果]** ✓\n")
                                total_from_task += 1
                            else:
                                f.write(f"- `{param_display_name}`: 额外参数（具体参数未定义）\n")
                    else:
                        # 非args参数，正常显示
                        param_additional_info = ""
                        if param_details.get('type'):
                            param_additional_info = f" (类型: {param_details['type']})"
                        if param_details.get('description'):
                            if param_additional_info:
                                param_additional_info += f", 描述: {param_details['description']}"
                            else:
                                param_additional_info = f" (描述: {param_details['description']})"
                        
                        if isinstance(param_data, dict):
                            source_type = param_data.get('source_type', 'unknown')
                        else:
                            source_type = param_data.source_type.value if hasattr(param_data, 'source_type') else 'unknown'
                        
                        if source_type == ParameterSourceType.DETERMINED.value:
                            value = param_data.get('value') if isinstance(param_data, dict) else param_data.value
                            value_str = str(value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                            f.write(f"- `{param_display_name}`{param_additional_info}: {value_str} **[确定值]** ✓\n")
                            total_determined += 1
                        elif source_type == ParameterSourceType.FROM_TASK.value:
                            source_task = param_data.get('source_task_id') if isinstance(param_data, dict) else param_data.source_task_id
                            source_key = param_data.get('source_output_key') if isinstance(param_data, dict) else param_data.source_output_key
                            key_info = f" (输出键: {source_key})" if source_key else ""
                            f.write(f"- `{param_display_name}`{param_additional_info}: 来自任务 `{source_task}`{key_info} **[任务结果]** ✓\n")
                            total_from_task += 1
                        elif source_type == ParameterSourceType.USER_REQUIRED.value:
                            prompt = param_data.get('user_prompt') if isinstance(param_data, dict) else param_data.user_prompt
                            f.write(f"- `{param_display_name}`{param_additional_info}: {prompt} **[需要用户提供]** ✓\n")
                            total_user_required += 1
                        else:
                            f.write(f"- `{param_display_name}`{param_additional_info}: 未知类型\n")
                f.write("\n")
            else:
                f.write("**参数推断结果**: 无参数\n\n")
            
            f.write("---\n\n")
        
        # 参数推断统计
        total_params = total_determined + total_from_task + total_user_required
        f.write("### 参数推断统计\n\n")
        f.write(f"- **确定的参数值**: {total_determined}\n")
        f.write(f"- **来自任务的参数**: {total_from_task}\n")
        f.write(f"- **需要用户提供的参数**: {total_user_required}\n")
        f.write(f"- **总参数数**: {total_params}\n\n")
        f.write("---\n\n")
        
        # 5. Executor 执行结果
        f.write("## 5. Executor 执行结果\n\n")
        
        # 执行信息
        execution_info = execution_results.get('execution_info', {})
        if execution_info.get('interrupted'):
            f.write(f"**执行状态**: ⚠ **中断**（HITL请求）\n\n")
            if execution_info.get('interrupt_data'):
                interrupt_data = execution_info['interrupt_data']
                if isinstance(interrupt_data, dict):
                    f.write(f"**中断类型**: {interrupt_data.get('type', 'unknown')}\n")
                    f.write(f"**中断消息**: {interrupt_data.get('message', '')}\n\n")
                else:
                    f.write(f"**中断数据**: {str(interrupt_data)[:200]}\n\n")
        else:
            f.write(f"**执行状态**: ✓ **完成**\n\n")
        
        # 执行错误
        execution_errors = execution_info.get('execution_errors', [])
        if execution_errors:
            f.write("### 执行错误\n\n")
            for i, error in enumerate(execution_errors, 1):
                f.write(f"#### 错误 {i}\n\n")
                f.write(f"**错误类型**: {error.get('error_type', 'unknown')}\n")
                f.write(f"**错误消息**: {error.get('error_message', '')}\n")
                if error.get('traceback'):
                    traceback_str = error['traceback']
                    if len(traceback_str) > 1000:
                        traceback_str = traceback_str[:1000] + "..."
                    f.write(f"**堆栈跟踪**:\n\n```\n{traceback_str}\n```\n\n")
            f.write("---\n\n")
        
        # 执行日志
        execution_logs = execution_info.get('execution_logs', [])
        if execution_logs:
            f.write("### 执行日志\n\n")
            for log in execution_logs:
                f.write(f"- {log}\n")
            f.write("\n---\n\n")
        
        # HITL 请求
        hitl_requests = execution_results.get('hitl_requests', {})
        hitl_responses = execution_results.get('hitl_responses', {})
        if hitl_requests:
            f.write("### HITL 请求\n\n")
            for task_id, request in hitl_requests.items():
                f.write(f"#### 任务 {task_id}\n\n")
                f.write(f"**类型**: {request.get('type', 'unknown')}\n")
                f.write(f"**消息**: {request.get('message', '')}\n")
                missing_params = request.get('missing_parameters', [])
                if missing_params:
                    f.write(f"**缺失参数** ({len(missing_params)} 个):\n\n")
                    # 获取任务信息以确定工具名
                    task_info = None
                    for task_result in execution_results.get('task_results', {}).values():
                        # 从参数推断结果中获取工具名
                        pass
                    
                    # 尝试从任务结果中获取工具名
                    task_result = execution_results.get('task_results', {}).get(task_id, {})
                    # 从参数推断结果中获取工具名
                    tool_name = None
                    if task_id in parameter_inference.get('task_results', {}):
                        tool_name = parameter_inference['task_results'][task_id].get('tool_name')
                    
                    for param in missing_params:
                        # 格式化参数显示
                        formatted_param = format_parameter_display(tool_name or '', param)
                        f.write(f"- {formatted_param}\n")
                    f.write("\n")
                
                # 显示自动生成的响应
                if task_id in hitl_responses:
                    response = hitl_responses[task_id]
                    if 'parameters' in response:
                        auto_params = response['parameters']
                        f.write(f"\n**自动生成的参数** ({len(auto_params)} 个):\n\n")
                        for param_name, param_value in auto_params.items():
                            value_str = str(param_value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                            f.write(f"- `{param_name}`: {value_str}\n")
                        f.write("\n> **说明**: 这些参数由测试框架自动生成（使用预设参数或LLM生成），用于自动化测试。\n")
                
                f.write("\n")
            f.write("---\n\n")
        
        f.write(f"**总任务数**: {execution_results.get('total_tasks', 0)}\n")
        f.write(f"**已完成**: {execution_results.get('completed_count', 0)}\n")
        f.write(f"**失败**: {execution_results.get('failed_count', 0)}\n\n")
        
        # 任务状态统计
        task_status_map = execution_results.get('task_status_map', {})
        if task_status_map:
            status_counts = {}
            for status in task_status_map.values():
                status_counts[status] = status_counts.get(status, 0) + 1
            f.write("**任务状态统计**:\n\n")
            for status, count in status_counts.items():
                f.write(f"- {status}: {count} 个\n")
            f.write("\n")
        
        task_results = execution_results.get('task_results', {})
        if task_results:
            f.write("### 任务执行详情\n\n")
            for task_id, result in task_results.items():
                f.write(f"#### 任务 {task_id}\n\n")
                
                status = result.get('status', 'unknown')
                status_emoji = "✓" if status == ExecutorTaskStatus.COMPLETED.value else "✗" if status == ExecutorTaskStatus.FAILED.value else "⏳"
                f.write(f"**状态**: {status} {status_emoji}\n\n")
                
                execution_mode = result.get('execution_mode', 'unknown')
                f.write(f"**执行模式**: {execution_mode}\n\n")
                
                execution_time = result.get('execution_time', 0)
                f.write(f"**执行时间**: {execution_time:.2f} 秒\n\n")
                
                retry_count = result.get('retry_count', 0)
                if retry_count > 0:
                    f.write(f"**重试次数**: {retry_count}\n\n")
                
                # 参数信息
                parameters = result.get('parameters', {})
                missing_parameters = result.get('missing_parameters', [])
                if parameters:
                    f.write(f"**执行参数** ({len(parameters)} 个):\n\n")
                    for param_name, param_value in list(parameters.items())[:5]:  # 只显示前5个
                        value_str = str(param_value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        f.write(f"- `{param_name}`: {value_str}\n")
                    if len(parameters) > 5:
                        f.write(f"- ... 还有 {len(parameters) - 5} 个参数\n")
                    f.write("\n")
                
                # 缺失参数
                if missing_parameters:
                    f.write(f"**缺失参数** ({len(missing_parameters)} 个):\n\n")
                    # 获取工具名（从任务结果或参数推断结果中）
                    tool_name = None
                    # 尝试从当前任务的参数推断结果中获取工具名
                    if task_id in parameter_inference.get('task_results', {}):
                        tool_name = parameter_inference['task_results'][task_id].get('tool_name')
                    
                    for param in missing_parameters:
                        # 格式化参数显示
                        formatted_param = format_parameter_display(tool_name or '', param)
                        f.write(f"- {formatted_param}\n")
                    f.write("\n")
                    f.write("> **说明**: 这些参数需要用户提供，导致任务无法执行。\n\n")
                
                # 代码（如果有）
                code = result.get('code')
                if code:
                    f.write(f"**执行代码**:\n\n```python\n{code[:500]}{'...' if len(code) > 500 else ''}\n```\n\n")
                
                # 输出结果
                output = result.get('output')
                if output:
                    output_str = str(output)
                    if len(output_str) > 500:
                        output_str = output_str[:500] + "..."
                    f.write(f"**执行输出**:\n\n```\n{output_str}\n```\n\n")
                
                # 错误信息（如果有）
                error = result.get('error')
                if error:
                    error_str = str(error)
                    if len(error_str) > 500:
                        error_str = error_str[:500] + "..."
                    f.write(f"**错误信息**:\n\n```\n{error_str}\n```\n\n")
                    
                    error_category = result.get('error_category')
                    if error_category:
                        f.write(f"**错误类别**: {error_category}\n\n")
                    
                    failure_analysis = result.get('failure_analysis')
                    if failure_analysis:
                        f.write(f"**失败分析**: {failure_analysis}\n\n")
                    
                    suggestions = result.get('suggestions', [])
                    if suggestions:
                        f.write(f"**改进建议**:\n\n")
                        for suggestion in suggestions:
                            f.write(f"- {suggestion}\n")
                        f.write("\n")
                
                # 置信度
                confidence_score = result.get('confidence_score')
                if confidence_score is not None:
                    f.write(f"**置信度**: {confidence_score:.2f}\n\n")
                
                # 结果满意度
                result_satisfied = result.get('result_satisfied')
                if result_satisfied is not None:
                    f.write(f"**结果满意度**: {'满足' if result_satisfied else '不满足'}\n\n")
                
                f.write("---\n\n")
        else:
            f.write("**任务执行结果**: 无执行结果\n\n")
        
        # 执行统计
        f.write("### 执行统计\n\n")
        f.write(f"- **总任务数**: {execution_results.get('total_tasks', 0)}\n")
        f.write(f"- **成功完成**: {execution_results.get('completed_count', 0)}\n")
        f.write(f"- **执行失败**: {execution_results.get('failed_count', 0)}\n")
        
        total_tasks = execution_results.get('total_tasks', 0)
        completed = execution_results.get('completed_count', 0)
        if total_tasks > 0:
            success_rate = (completed / total_tasks) * 100
            f.write(f"- **成功率**: {success_rate:.1f}%\n")
        
        f.write("\n")
    
    print(f"\n✓ 完整流程测试日志已保存: {log_file}")


# ===================== 测试类 =====================

class TestCompleteFlow:
    """完整流程测试（参数推断 + Executor 执行）"""
    
    @pytest.mark.parametrize("test_case", EXECUTION_TEST_CASES)
    def test_complete_flow(self, test_case):
        """测试完整流程：参数推断 → Executor 执行"""
        print(f"\n{'='*80}")
        print(f"测试用例: {test_case['name']}")
        print(f"描述: {test_case['description']}")
        print(f"用户输入: {test_case['user_input']}")
        if test_case['execution_plan']:
            print(f"执行计划:\n{test_case['execution_plan']}")
        print(f"{'='*80}\n")
        
        # 步骤1: 任务分解和参数推断
        print("[步骤1] 执行任务分解（粗分解 → 细分解 → 并行推断 → 参数推断）...")
        global_state, decomposition_state = run_full_decomposition_flow(
            test_case['user_input'],
            test_case['execution_plan']
        )
        
        assert global_state is not None, "GlobalState 不应该为 None"
        assert len(global_state.subtasks) > 0, "应该生成至少一个子任务"
        print(f"✓ 任务分解完成，生成了 {len(global_state.subtasks)} 个子任务")
        
        # 验证参数推断结果
        assert hasattr(decomposition_state, 'parameter_inference_results'), \
            "参数推断结果应该在 task_decomposition 阶段生成"
        print(f"✓ 参数推断完成")
        
        # 提取参数推断结果
        task_results_dict, task_info_dict, stats = extract_parameter_inference_results(decomposition_state)
        print(f"✓ 参数推断统计: 确定值={stats['determined_count']}, 来自任务={stats['from_task_count']}, 需要用户={stats['user_required_count']}")
        
        # 步骤2: Executor 执行
        print("\n[步骤2] 执行 Executor 子图（参数推断 → 任务执行 → 结果分析）...")
        try:
            # 获取测试用例的预设参数
            preset_params = test_case.get('preset_parameters', {})
            updated_global_state, executor_state, execution_info = run_executor_flow(
                global_state, 
                preset_params=preset_params,
                auto_resolve_hitl=True
            )
            
            # 检查是否中断
            if execution_info.get('interrupted'):
                print(f"⚠ Executor 执行中断（HITL请求）")
                if execution_info.get('hitl_requests'):
                    print(f"  需要用户提供参数的任务数: {len(execution_info['hitl_requests'])}")
            else:
                print(f"✓ Executor 执行完成")
            
            # 检查执行错误
            if execution_info.get('execution_errors'):
                print(f"⚠ 执行过程中有 {len(execution_info['execution_errors'])} 个错误")
            
            # 提取执行结果（传入 global_state 以确保获取所有任务）
            execution_results = extract_execution_results(executor_state, execution_info, global_state)
            print(f"✓ 执行统计: 总任务={execution_results['total_tasks']}, 完成={execution_results['completed_count']}, 失败={execution_results['failed_count']}")
            
            # 检查任务数量是否匹配
            if len(global_state.subtasks) != executor_state.total_tasks:
                print(f"  ⚠ 警告: global_state 有 {len(global_state.subtasks)} 个任务，但 executor 只处理了 {executor_state.total_tasks} 个任务")
                print(f"  ⚠ 这可能是因为 executor 只接收了部分任务，或者任务在传递过程中丢失了")
            
            # 显示任务状态
            ready_count = sum(1 for s in execution_results['task_status_map'].values() if s == ExecutorTaskStatus.READY.value)
            waiting_hitl_count = sum(1 for s in execution_results['task_status_map'].values() 
                                     if s in [ExecutorTaskStatus.WAITING_HITL_PARAMS.value, ExecutorTaskStatus.WAITING_HITL_CONFIRM.value])
            if ready_count > 0:
                print(f"  ⚠ 仍有 {ready_count} 个任务处于就绪状态但未执行")
            if waiting_hitl_count > 0:
                print(f"  ⚠ 有 {waiting_hitl_count} 个任务等待 HITL 响应")
            
        except Exception as e:
            print(f"✗ Executor 执行失败: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            print(error_traceback)
            
            # 即使执行失败，也记录参数推断结果
            execution_results = {
                'total_tasks': len(global_state.subtasks),
                'completed_count': 0,
                'failed_count': 0,
                'task_results': {},
                'task_status_map': {},
                'hitl_requests': {},
                'hitl_responses': {},
                'execution_info': {
                    'interrupted': False,
                    'interrupt_data': None,
                    'execution_errors': [{
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'traceback': error_traceback
                    }],
                    'execution_logs': [f"执行异常: {str(e)}"]
                }
            }
            executor_state = None
        
        # 准备日志数据
        coarse_decomposition, fine_decomposition, parameter_inference = prepare_log_data(
            decomposition_state, task_results_dict, task_info_dict
        )
        
        # 保存完整日志
        save_complete_test_log(
            test_case['name'],
            test_case['execution_plan'],
            coarse_decomposition,
            fine_decomposition,
            parameter_inference,
            execution_results
        )
        
        # 验证：至少应该有一些参数推断结果
        assert stats['total_count'] > 0, \
            "应该至少有一些参数推断结果。正确识别参数来源（确定值、来自任务、需要用户提供）都算成功。"
        
        # 验证：如果执行成功，应该有执行结果
        if executor_state and execution_results.get('total_tasks', 0) > 0:
            assert execution_results.get('completed_count', 0) + execution_results.get('failed_count', 0) > 0, \
                "应该有至少一个任务完成或失败"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

