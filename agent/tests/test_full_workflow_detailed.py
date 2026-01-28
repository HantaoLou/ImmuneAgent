"""
完整工作流详细测试用例

测试从 supervisor -> task_decomposition -> executor 的全流程，并详细记录所有中间过程。

记录内容包括：
1. 用户原始输入
2. 执行计划（如果有）
3. Supervisor 分类结果
4. Task Decomposition 任务分解结果
5. Executor 参数推断结果（包括从用户输入、执行计划、依赖任务输出中提取的参数）
6. 每个任务的输入输出及任务信息
7. 任务执行过程及顺序（包括并发执行和优先级策略）
8. 每个任务的执行汇总（execution_summary）
9. 汇总的结果

运行方式：pytest tests/test_full_workflow_detailed.py::test_full_workflow_detailed -v -s
"""

import os
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType, SubTask
from main_graph import build_main_graph
from nodes.subagents.supervisor.graph import (
    build_supervisor_subgraph,
    supervisor_input_mapper,
    supervisor_output_mapper
)
from nodes.subagents.task_decomposition.graph import (
    build_task_decomposition_subgraph,
    task_decomposition_input_mapper,
    task_decomposition_output_mapper
)
from nodes.subagents.executor.graph import (
    build_executor_subgraph,
    executor_input_mapper,
    executor_output_mapper,
    execute_executor_with_interrupt_support,
    resume_executor_after_interrupt,
    ExecutorTaskStatus
)
from utils.hitl_interaction import handle_hitl_interrupt

# 导入测试日志记录器
try:
    from test_logger import init_global_logger, get_global_logger, save_global_logger, TestCaseLogger
except ImportError:
    import sys
    test_dir = Path(__file__).parent
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
    from test_logger import init_global_logger, get_global_logger, save_global_logger, TestCaseLogger


def _serialize_interrupt_data(data: Any) -> Dict[str, Any]:
    """序列化中断数据"""
    if data is None:
        return {}
    
    if hasattr(data, 'value') and hasattr(data, 'id'):
        return {
            "type": "Interrupt",
            "id": str(getattr(data, 'id', '')),
            "value": _serialize_interrupt_data(getattr(data, 'value', {}))
        }
    
    if isinstance(data, dict):
        return {k: _serialize_interrupt_data(v) for k, v in data.items()}
    
    if isinstance(data, list):
        return [_serialize_interrupt_data(item) for item in data]
    
    try:
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        return str(data)


def extract_interrupt_value(obj: Any, max_depth: int = 5) -> Any:
    """递归提取中断值"""
    if max_depth <= 0:
        return obj
    
    if hasattr(obj, 'value'):
        return extract_interrupt_value(obj.value, max_depth - 1)
    
    if isinstance(obj, dict):
        if 'value' in obj:
            return extract_interrupt_value(obj['value'], max_depth - 1)
        if 'type' in obj:
            return obj
        return obj
    
    if isinstance(obj, tuple):
        if len(obj) >= 2:
            return extract_interrupt_value(obj[1], max_depth - 1)
        elif len(obj) == 1:
            return extract_interrupt_value(obj[0], max_depth - 1)
        return obj
    
    return obj


@pytest.fixture(scope="module", autouse=True)
def setup_global_logger():
    """初始化全局日志记录器"""
    test_file_name = Path(__file__).stem
    init_global_logger(test_file_name)
    yield
    save_global_logger()


def test_full_workflow_detailed(request=None, test_case_logger=None):
    """
    测试完整工作流并详细记录所有中间过程
    
    流程：
    1. Supervisor: 分类用户任务
    2. Task Decomposition: 分解任务
    3. Executor: 推断参数、执行任务（支持并发执行和优先级策略）、汇总结果
    """
    # 获取日志记录器
    logger = test_case_logger
    if logger is None:
        global_logger = get_global_logger()
        if global_logger:
            test_case_name = request.node.name if request else "test_full_workflow_detailed"
            logger = global_logger.get_test_case_logger(test_case_name)
    
    # 创建测试目录
    test_dir = Path("./sandbox/full_workflow_detailed_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== 步骤 1: 准备用户输入 ====================
    
        # 3. invoke lineage_analysis service's all tools
        # 4. invoke r_data_integration service's all tools
        # 4. invoke bioinformatics service's all tools
        # 5. invoke alphafold3
    user_input = '''
        please design a computational method to identifiy broadly neutralizing antibodies against H5N1.

        mcp servers only consider the following:
        - igblast
        - metabcr
        parameters:
        - fasta file for analyze_vdj_batch: /data_new/workspace/flu-simple.fasta
        - antibody file for metabcr: /data_new/workspace/flu-simple.csv
        - antigen file for metabcr: /data_new/workspace/Copy of flu_bind_variant_seq.xlsx
    '''
    execution_plan = None  # 可以在这里提供执行计划
    
    print(f"\n{'='*80}")
    print(f"【完整工作流详细测试】")
    print(f"{'='*80}")
    print(f"用户输入: {user_input}")
    if execution_plan:
        print(f"执行计划: {execution_plan}")
    print(f"测试目录: {test_dir.absolute()}")
    print(f"{'='*80}\n")
    
    # 记录用户原始输入
    if logger:
        logger.log_initial_state({
            "user_input": user_input,
            "execution_plan": execution_plan,
            "sandbox_dir": str(test_dir)
        }, "用户原始输入和执行计划")
    
    # 创建初始全局状态
    global_state = GlobalState(
        user_input=user_input,
        execution_plan=execution_plan,
        sandbox_dir=str(test_dir)
    )
    
    if logger:
        logger.log_initial_state(global_state, "初始 GlobalState")
    
    # ==================== 步骤 2: Supervisor 分类 ====================
    print(f"\n{'='*60}")
    print(f"【步骤 1: Supervisor 任务分类】")
    print(f"{'='*60}")
    
    supervisor_subgraph = build_supervisor_subgraph()
    supervisor_input = supervisor_input_mapper(global_state)
    
    if logger:
        logger.log_node_execution("supervisor_input_mapper", global_state, supervisor_input, 
                                 "Supervisor 输入映射")
    
    supervisor_output = supervisor_subgraph.invoke(supervisor_input)
    global_state = supervisor_output_mapper(supervisor_output, global_state)
    
    if logger:
        logger.log_node_execution("supervisor_subgraph", supervisor_input, supervisor_output,
                                 "Supervisor 任务分类完成")
        logger.log_node_execution("supervisor_output_mapper", supervisor_output, global_state,
                                 "Supervisor 输出映射")
    
    print(f"✓ 任务分类结果: {global_state.user_task_type}")
    print(f"  分类原因: {supervisor_output.classification_reason if hasattr(supervisor_output, 'classification_reason') else 'N/A'}")
    
    # ==================== 步骤 3: Task Decomposition ====================
    print(f"\n{'='*60}")
    print(f"【步骤 2: Task Decomposition 任务分解】")
    print(f"{'='*60}")
    
    task_decomposition_subgraph = build_task_decomposition_subgraph()
    task_decomp_input = task_decomposition_input_mapper(global_state)
    
    if logger:
        logger.log_node_execution("task_decomposition_input_mapper", global_state, task_decomp_input,
                                 "Task Decomposition 输入映射")
    
    task_decomp_output = task_decomposition_subgraph.invoke(task_decomp_input)
    global_state = task_decomposition_output_mapper(task_decomp_output, global_state)
    
    if logger:
        logger.log_node_execution("task_decomposition_subgraph", task_decomp_input, task_decomp_output,
                                 "Task Decomposition 任务分解完成")
        logger.log_node_execution("task_decomposition_output_mapper", task_decomp_output, global_state,
                                 "Task Decomposition 输出映射")
    
    # 记录任务分解结果
    print(f"✓ 任务分解完成")
    print(f"  分解的任务数: {len(global_state.subtasks)}")
    print(f"  并行任务组数: {len(global_state.parallel_task_groups)}")
    print(f"  注意: 参数推断将在 Executor 子图中进行")
    
    # 记录每个任务的详细信息
    if logger:
        task_order = []
        for task in global_state.subtasks:
            task_info = {
                "task_id": task.task_id,
                "content": task.content,
                "dependencies": task.dependencies,
                "task_type": task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
                "tools": []
            }
            
            if isinstance(task.result, dict):
                tools = task.result.get("tools", [])
                for tool in tools:
                    if isinstance(tool, dict):
                        task_info["tools"].append(tool.get("tool_name", tool.get("name", "")))
                    else:
                        task_info["tools"].append(str(tool))
                task_info["inputs"] = task.result.get("inputs", [])
                task_info["outputs"] = task.result.get("outputs", [])
            
            task_order.append(task_info)
        
        # 记录并行任务组
        for group_id, group in global_state.parallel_task_groups.items():
            for task in group.subtasks:
                task_info = {
                    "task_id": task.task_id,
                    "content": task.content,
                    "dependencies": task.dependencies,
                    "parallel_group_id": group_id,
                    "tools": []
                }
                
                if isinstance(task.result, dict):
                    tools = task.result.get("tools", [])
                    for tool in tools:
                        if isinstance(tool, dict):
                            task_info["tools"].append(tool.get("tool_name", tool.get("name", "")))
                        else:
                            task_info["tools"].append(str(tool))
                    task_info["inputs"] = task.result.get("inputs", [])
                    task_info["outputs"] = task.result.get("outputs", [])
                
                task_order.append(task_info)
        
        logger.log_task_order(task_order)
    
    # 打印从上下文中抽取的参数表（这些参数将在 executor 中用于参数推断）
    if hasattr(task_decomp_output, 'context_extracted_params') and task_decomp_output.context_extracted_params:
        print(f"\n  【从上下文抽取的参数表（将在 Executor 中使用）】")
        print(f"  {'='*70}")
        for task_id, tools_params in task_decomp_output.context_extracted_params.items():
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
    
    # 记录分解摘要
    if hasattr(task_decomp_output, 'decomposition_summary') and task_decomp_output.decomposition_summary:
        print(f"\n  分解摘要: {task_decomp_output.decomposition_summary}")
    
    # ==================== 步骤 4: Executor 执行 ====================
    print(f"\n{'='*60}")
    print(f"【步骤 3: Executor 任务执行】")
    print(f"{'='*60}")
    print(f"Executor 功能:")
    print(f"  - 参数推断: 基于用户输入、执行计划、依赖任务输出进行动态参数推断")
    print(f"  - 并发执行: 支持多任务并行执行，可配置工具级并发限制")
    print(f"  - 优先级策略: 支持基于工具类型的任务优先级调度")
    print(f"  - 超时控制: 支持单任务超时限制")
    print(f"  - 结果汇总: 每个任务执行后生成执行汇总")
    print(f"{'='*60}")
    
    # 检查是否有任务需要执行
    all_tasks = global_state.subtasks + [
        task for group in global_state.parallel_task_groups.values()
        for task in group.subtasks
    ]
    
    if not all_tasks:
        print("⚠ 没有任务需要执行，跳过 Executor 步骤")
        if logger:
            logger.log_summary({
                "user_input": user_input,
                "execution_plan": execution_plan,
                "task_classification": global_state.user_task_type.value if hasattr(global_state.user_task_type, 'value') else str(global_state.user_task_type),
                "total_tasks": 0,
                "completed": 0,
                "failed": 0,
                "note": "没有任务需要执行"
            })
        return
    
    executor_subgraph = build_executor_subgraph()
    executor_input = executor_input_mapper(global_state)
    
    if logger:
        logger.log_node_execution("executor_input_mapper", global_state, executor_input,
                                 "Executor 输入映射")
    
    # 执行工作流（支持HITL中断）
    thread_id = "full_workflow_detailed_test"
    print(f"\n开始执行任务...")
    print(f"线程ID: {thread_id}\n")
    
    # 首次执行
    try:
        result = execute_executor_with_interrupt_support(
            executor_subgraph,
            executor_input,
            thread_id=thread_id
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"\n{'='*80}")
        print(f"【Executor 执行错误】")
        print(f"{'='*80}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print(f"\n完整错误堆栈:")
        print(f"{error_traceback}")
        print(f"{'='*80}\n")
        
        if logger:
            logger.log_node_execution("executor_subgraph", executor_input, None,
                                     f"Executor 执行失败: {type(e).__name__}: {e}")
            # 记录完整错误信息
            logger.log_summary({
                "error": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_traceback": error_traceback
            })
        
        # 重新抛出异常，让测试失败
        raise
    
    if logger:
        logger.log_node_execution("executor_subgraph", executor_input, result.get("result"),
                                 "Executor 首次执行")
    
    # 处理中断循环
    iteration_count = 0
    max_iterations = 50
    
    while result.get("interrupted", False) and iteration_count < max_iterations:
        iteration_count += 1
        interrupt_data = result.get("interrupt_data")
        
        if not interrupt_data:
            if executor_input.parent_state and executor_input.parent_state.hitl_status:
                try:
                    interrupt_data = json.loads(executor_input.parent_state.hitl_status)
                except:
                    pass
        
        if interrupt_data:
            actual_interrupt_data = extract_interrupt_value(interrupt_data)
            
            if not isinstance(actual_interrupt_data, dict):
                actual_interrupt_data = {"value": actual_interrupt_data}
            
            print(f"\n{'='*60}")
            print(f"【HITL 中断 #{iteration_count}】")
            print(f"{'='*60}")
            print(f"中断数据: {actual_interrupt_data}")
            
            if logger:
                serialized_data = _serialize_interrupt_data(actual_interrupt_data)
                logger.log_hitl_request(
                    task_id=actual_interrupt_data.get("task_id", "unknown"),
                    request_type=actual_interrupt_data.get("type", "unknown"),
                    request_data=serialized_data
                )
            
            try:
                user_response = handle_hitl_interrupt(
                    actual_interrupt_data,
                    callback=None,
                    use_file=False
                )
                
                if logger:
                    serialized_response = _serialize_interrupt_data(user_response)
                    logger.log_hitl_response(
                        task_id=actual_interrupt_data.get("task_id", "unknown"),
                        response_type=user_response.get("type", "unknown"),
                        response_data=serialized_response
                    )
                
                try:
                    global_state.hitl_status = json.dumps(user_response, ensure_ascii=False)
                except Exception as json_e:
                    global_state.hitl_status = json.dumps(_serialize_interrupt_data(user_response), ensure_ascii=False)
                
                print(f"\n恢复执行...\n")
                
                result = resume_executor_after_interrupt(
                    executor_subgraph,
                    thread_id=thread_id,
                    resume_value=user_response
                )
                
                if logger:
                    logger.log_node_execution("executor_subgraph", None, result.get("result"),
                                             f"恢复执行 #{iteration_count}")
                
            except KeyboardInterrupt:
                print("\n用户退出，终止执行")
                break
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"\n⚠ HITL交互处理失败: {e}")
                print(f"错误堆栈:\n{error_traceback}")
                break
        else:
            print(f"\n⚠ 检测到中断，但无法获取中断数据")
            break
    
    # 获取最终结果
    final_state = result.get("result")
    if final_state is None:
        if executor_input.parent_state:
            try:
                executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
                final_result = executor_subgraph.invoke(
                    executor_input_dict,
                    config={"configurable": {"thread_id": thread_id}}
                )
                final_state = final_result
            except:
                final_state = executor_input
    
    if final_state is None:
        final_state = executor_input
    
    # 映射回全局状态
    global_state = executor_output_mapper(final_state, global_state)
    
    if logger:
        logger.log_node_execution("executor_output_mapper", final_state, global_state,
                                 "Executor 输出映射")
        logger.log_node_execution("executor_subgraph", executor_input, final_state,
                                 "Executor 最终状态")
    
    # 记录参数推断结果（从 executor 状态中提取）
    if hasattr(final_state, 'task_results') and final_state.task_results:
        print(f"\n  【参数推断和执行汇总】")
        print(f"  {'='*70}")
        for task_id, task_result in final_state.task_results.items():
            print(f"  Task {task_id}:")
            if task_result.result_summary:
                print(f"    执行汇总: {json.dumps(task_result.result_summary, ensure_ascii=False, indent=2)[:200]}...")
            # 从任务结果中提取参数推断信息（如果存在）
            if task_result.output:
                try:
                    output_data = json.loads(task_result.output) if isinstance(task_result.output, str) else task_result.output
                    if isinstance(output_data, dict) and "inferred_params" in output_data:
                        print(f"    推断的参数: {json.dumps(output_data['inferred_params'], ensure_ascii=False, indent=2)[:200]}...")
                except:
                    pass
        print(f"  {'='*70}")
    
    # 记录所有任务执行结果（优化：提取final_result和关键消息，过滤进度消息）
    if logger:
        for task_id, task_result in final_state.task_results.items():
            # 准备result字典，包含完整的关键信息
            result_dict = {
                "confidence_score": task_result.confidence_score,
                "retry_count": task_result.retry_count
            }
            
            # 处理output：提取type为result的消息（工具执行结果），过滤掉progress消息
            if task_result.output:
                output = task_result.output
                
                # 如果是字符串，尝试解析
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except:
                        # 如果解析失败，保存完整原始字符串
                        result_dict["output"] = output
                        output = None
                
                # 如果是字典，提取result类型的消息
                if isinstance(output, dict):
                    # 提取task_id和service_id
                    if "task_id" in output:
                        result_dict["task_id"] = output["task_id"]
                    if "service_id" in output:
                        result_dict["service_id"] = output["service_id"]
                    
                    # 优先提取final_result（如果存在）
                    if "final_result" in output and output["final_result"]:
                        result_dict["final_result"] = output["final_result"]
                    
                    # 从messages中提取type为result的消息（工具执行结果）
                    if "messages" in output and isinstance(output["messages"], list):
                        result_messages = []
                        progress_count = 0
                        
                        for msg in output["messages"]:
                            if isinstance(msg, dict):
                                msg_type = msg.get("type", "")
                                
                                # 只保留type为result的消息（工具执行结果）
                                if msg_type == "result":
                                    result_messages.append(msg)
                                elif msg_type == "progress":
                                    progress_count += 1
                                # 也保留error和end类型的消息（用于错误诊断）
                                elif msg_type in ["error", "end"]:
                                    result_messages.append(msg)
                        
                        # 记录result消息
                        if result_messages:
                            result_dict["result_messages"] = result_messages
                            result_dict["result_messages_count"] = len(result_messages)
                        
                        # 记录统计信息
                        result_dict["total_messages"] = len(output["messages"])
                        result_dict["progress_messages_filtered"] = progress_count
                    
                    # 保存完整的output结构（但messages中只包含result消息）
                    result_dict["output"] = output
                elif output is not None:
                    # 如果不是字典也不是字符串，直接保存
                    result_dict["output"] = output
            
            # 添加错误信息（包含完整堆栈）
            if task_result.error:
                result_dict["error"] = task_result.error  # 包含完整错误堆栈
                result_dict["error_full"] = task_result.error  # 确保完整错误信息被记录
            if task_result.error_category:
                result_dict["error_category"] = task_result.error_category.value if (hasattr(task_result.error_category, 'value')) else str(task_result.error_category)
            
            # 添加执行汇总信息
            if task_result.result_summary:
                result_dict["execution_summary"] = task_result.result_summary
            
            # 添加代码信息到 result_dict（即使代码为空或失败，也记录）
            if task_result.code:
                result_dict["generated_code"] = task_result.code
            elif task_result.execution_mode == "codeact":
                # 如果是 CodeAct 模式但没有代码，记录为 None（表示代码生成失败）
                result_dict["generated_code"] = None
                result_dict["code_generation_failed"] = True
            
            logger.log_task_execution(
                task_id=task_id,
                status=task_result.status.value if (task_result.status and hasattr(task_result.status, 'value')) else str(task_result.status) if task_result.status else "None",
                execution_mode=task_result.execution_mode,
                result=result_dict,
                error=task_result.error if task_result.error else None  # 不截断错误信息，包含完整堆栈
            )
            
            # 记录 CodeAct 代码（无论成功或失败都要记录）
            if task_result.code:
                logger.log_codeact_output(
                    task_id=task_id,
                    generated_code=task_result.code,  # 不截断代码
                    execution_result={
                        "status": task_result.status.value if (task_result.status and hasattr(task_result.status, 'value')) else str(task_result.status) if task_result.status else "None",
                        "output": task_result.output if task_result.output else None,  # 不截断输出
                        "error": task_result.error if task_result.error else None,  # 包含完整错误堆栈
                        "error_category": task_result.error_category.value if (task_result.error_category and hasattr(task_result.error_category, 'value')) else str(task_result.error_category) if task_result.error_category else None
                    }
                )
            elif task_result.execution_mode == "codeact":
                # 如果是 CodeAct 模式但没有代码，也记录（表示代码生成失败）
                logger.log_codeact_output(
                    task_id=task_id,
                    generated_code=None,  # 代码生成失败
                    execution_result={
                        "status": task_result.status.value if (task_result.status and hasattr(task_result.status, 'value')) else str(task_result.status) if task_result.status else "None",
                        "output": task_result.output if task_result.output else None,
                        "error": task_result.error if task_result.error else "Code generation failed",
                        "error_category": task_result.error_category.value if (task_result.error_category and hasattr(task_result.error_category, 'value')) else str(task_result.error_category) if task_result.error_category else None
                    }
                )
    
    # ==================== 步骤 5: 汇总结果 ====================
    print(f"\n{'='*80}")
    print(f"【执行完成总结】")
    print(f"{'='*80}")
    print(f"总任务数: {final_state.total_tasks}")
    print(f"已完成: {final_state.completed_count}")
    print(f"失败: {final_state.failed_count}")
    print(f"HITL请求数: {len(final_state.hitl_requests)}")
    print(f"HITL响应数: {len(final_state.hitl_responses)}")
    print(f"中断迭代次数: {iteration_count}")
    
    # 打印每个任务的执行状态
    print(f"\n任务执行详情:")
    
    for task in all_tasks:
        task_result = final_state.task_results.get(task.task_id)
        if task_result:
            status_icon = "✓" if task_result.status == ExecutorTaskStatus.COMPLETED else "✗" if task_result.status == ExecutorTaskStatus.FAILED else "⏳"
            status_str = task_result.status.value if (task_result.status and hasattr(task_result.status, 'value')) else str(task_result.status) if task_result.status else "None"
            print(f"  {status_icon} {task.task_id}: {status_str}")
            if task_result.error:
                print(f"     错误: {task_result.error[:100]}")
            if task_result.retry_count > 0:
                print(f"     重试次数: {task_result.retry_count}")
            if task_result.result_summary:
                summary_preview = json.dumps(task_result.result_summary, ensure_ascii=False)[:100]
                print(f"     执行汇总: {summary_preview}...")
        else:
            print(f"  ? {task.task_id}: 未执行")
    
    # 打印 executor 汇总信息（从 merged_result 中提取）
    if global_state.merged_result and "executor_results" in global_state.merged_result:
        executor_results = global_state.merged_result["executor_results"]
        print(f"\n  Executor 执行汇总:")
        print(f"    总任务数: {executor_results.get('total_tasks', 0)}")
        print(f"    已完成: {executor_results.get('completed', 0)}")
        print(f"    失败: {executor_results.get('failed', 0)}")
    
    # 记录汇总结果
    if logger:
        logger.log_summary({
            "user_input": user_input,
            "execution_plan": execution_plan,
            "task_classification": global_state.user_task_type.value if hasattr(global_state.user_task_type, 'value') else str(global_state.user_task_type),
            "total_tasks": final_state.total_tasks,
            "completed": final_state.completed_count,
            "failed": final_state.failed_count,
            "hitl_requests_count": len(final_state.hitl_requests),
            "hitl_responses_count": len(final_state.hitl_responses),
            "interrupt_iterations": iteration_count,
            "tasks_count": len(all_tasks),
            "parallel_groups_count": len(global_state.parallel_task_groups)
        })
    
    print(f"{'='*80}\n")
    
    # 验证基本结果
    assert final_state.total_tasks == len(all_tasks), f"任务数不匹配: {final_state.total_tasks} != {len(all_tasks)}"
    assert final_state.completed_count + final_state.failed_count <= final_state.total_tasks, "完成+失败数不应超过总任务数"
    
    print(f"✓ 完整工作流详细测试完成")


@pytest.fixture(autouse=True)
def test_case_logger(request):
    """为每个测试用例创建日志记录器"""
    global_logger = get_global_logger()
    if global_logger:
        test_case_name = request.node.name
        logger = global_logger.get_test_case_logger(test_case_name)
        yield logger
        global_logger.finish_test_case(test_case_name)
    else:
        yield None


# 主函数，支持直接运行
if __name__ == "__main__":
    # 初始化日志
    test_file_name = Path(__file__).stem
    init_global_logger(test_file_name)
    
    try:
        # 运行测试
        test_full_workflow_detailed()
        
    finally:
        # 保存日志
        save_global_logger()

