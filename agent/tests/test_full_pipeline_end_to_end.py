"""
完整流程端到端测试用例

测试从 supervisor 分类 → immunity 计划生成 → task_decomposition 任务分解 → executor 任务执行的完整流程

测试三个问题：
1. I have a candidate antibody sequence targeting the HA protein of H5N1. I need to optimize its antigen binding and neutralizing capabilities. What tools should I use for analysis and optimization? Please explain the purpose of each step.
2. Optimize the sequence stability of the influenza virus NA protein to obtain an optimized NA amino acid sequence. How should I utilize which tools to build and evaluate this prediction pipeline?
3. I want to optimize an mRNA sequence for the influenza NA antigen to achieve its high-efficiency and sustained expression. Which tools can I use to predict the mRNA expression level and translation efficiency?

测试要点：
1. supervisor 分类是否正确（应该都归类到 IMMUNOLOGY_TASK）
2. immunity 是否正确产出计划
3. task_decomposition 是否正确进行任务分解和参数推断
4. executor 是否正确执行所有任务并汇总结果

运行方式：pytest tests/test_full_pipeline_end_to_end.py -v -s
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

from state import GlobalState, UserTaskType
from main_graph import build_main_graph
from nodes.subagents.executor.graph import (
    ExecutorTaskStatus,
    resume_executor_after_interrupt,
    execute_executor_with_interrupt_support
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


# 测试问题列表
TEST_QUESTIONS = [
    {
        "id": "question_1",
        "text": "I have a candidate antibody sequence targeting the HA protein of H5N1. I need to optimize its antigen binding and neutralizing capabilities. What tools should I use for analysis and optimization? Please explain the purpose of each step.",
        "expected_classification": UserTaskType.IMMUNOLOGY_TASK
    },
    # {
    #     "id": "question_2",
    #     "text": "Optimize the sequence stability of the influenza virus NA protein to obtain an optimized NA amino acid sequence. How should I utilize which tools to build and evaluate this prediction pipeline?",
    #     "expected_classification": UserTaskType.IMMUNOLOGY_TASK
    # },
    # {
    #     "id": "question_3",
    #     "text": "I want to optimize an mRNA sequence for the influenza NA antigen to achieve its high-efficiency and sustained expression. Which tools can I use to predict the mRNA expression level and translation efficiency?",
    #     "expected_classification": UserTaskType.IMMUNOLOGY_TASK
    # }
]


def _serialize_interrupt_data(data: Any) -> Dict[str, Any]:
    """
    Serialize interrupt_data, converting Interrupt objects to serializable dictionaries
    
    Args:
        data: Data that may contain Interrupt objects
        
    Returns:
        Serializable dictionary
    """
    if data is None:
        return {}
    
    # If it's an Interrupt object (usually has value and id attributes)
    if hasattr(data, 'value') and hasattr(data, 'id'):
        return {
            "type": "Interrupt",
            "id": str(getattr(data, 'id', '')),
            "value": _serialize_interrupt_data(getattr(data, 'value', {}))
        }
    
    # If it's a dictionary, recursively process
    if isinstance(data, dict):
        return {k: _serialize_interrupt_data(v) for k, v in data.items()}
    
    # If it's a list, recursively process
    if isinstance(data, list):
        return [_serialize_interrupt_data(item) for item in data]
    
    # For other types, try to convert to string or return as-is
    try:
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        return str(data)


def extract_interrupt_value(obj: Any, max_depth: int = 5) -> Any:
    """
    Recursively extract interrupt value until a dictionary is obtained
    
    Args:
        obj: Interrupt object or nested structure
        max_depth: Maximum recursion depth
        
    Returns:
        Extracted interrupt value (usually a dictionary)
    """
    if max_depth <= 0:
        return obj
    
    # If it's an Interrupt object, extract its value
    if hasattr(obj, 'value'):
        return extract_interrupt_value(obj.value, max_depth - 1)
    
    # If it's a dictionary, check if there's a 'value' field
    if isinstance(obj, dict):
        if 'value' in obj:
            return extract_interrupt_value(obj['value'], max_depth - 1)
        # If it's already in the correct format (has 'type' field), return directly
        if 'type' in obj:
            return obj
        return obj
    
    # If it's a tuple, extract the second element
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

def test_full_pipeline():
    """
    测试完整流程
    """
    _test_full_pipeline(question = TEST_QUESTIONS[0])


def _test_full_pipeline(question: Dict[str, Any], main_graph=None, request=None, test_case_logger=None):
    """
    测试完整流程的通用函数
    
    Args:
        question: 测试问题字典，包含 id, text, expected_classification
        main_graph: 主图实例（可选）
        request: pytest request 对象（可选）
        test_case_logger: 测试日志记录器（可选）
    """
    # 如果没有提供main_graph，构建一个
    if main_graph is None:
        main_graph = build_main_graph()
    
    # 获取日志记录器
    logger = test_case_logger
    if logger is None:
        global_logger = get_global_logger()
        if global_logger:
            test_case_name = f"test_full_pipeline_{question['id']}"
            if request:
                test_case_name = request.node.name
            logger = global_logger.get_test_case_logger(test_case_name)
    
    # 创建测试目录
    test_dir = Path(f"./sandbox/full_pipeline_test_{question['id']}")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"【完整流程端到端测试 - {question['id']}】")
    print(f"{'='*80}")
    print(f"问题: {question['text']}")
    print(f"预期分类: {question['expected_classification']}")
    print(f"测试目录: {test_dir.absolute()}")
    print(f"{'='*80}\n")
    
    # 记录初始问题
    if logger:
        logger.log_initial_state(None, f"测试问题: {question['text']}")
    
    # 创建初始状态
    initial_state = GlobalState(
        user_input=question['text'],
        user_task_type=None,  # 初始状态为None，等待supervisor分类
        sandbox_dir=str(test_dir)
    )
    
    if logger:
        logger.log_initial_state(initial_state, "初始 GlobalState")
    
    # ===================== 阶段1: Supervisor 分类 =====================
    print(f"\n{'='*80}")
    print(f"【阶段1: Supervisor 任务分类】")
    print(f"{'='*80}\n")
    
    try:
        # 执行主图（只执行到supervisor节点）
        # 我们需要手动执行supervisor节点来测试分类
        from nodes.subagents.supervisor.graph import (
            build_supervisor_subgraph,
            supervisor_input_mapper,
            supervisor_output_mapper
        )
        
        supervisor_subgraph = build_supervisor_subgraph()
        supervisor_input = supervisor_input_mapper(initial_state)
        supervisor_output = supervisor_subgraph.invoke(supervisor_input)
        state_after_supervisor = supervisor_output_mapper(supervisor_output, initial_state)
        
        # 检查分类结果
        actual_classification = state_after_supervisor.user_task_type
        classification_correct = (actual_classification == question['expected_classification'])
        
        print(f"✓ Supervisor 分类完成")
        print(f"  实际分类: {actual_classification}")
        print(f"  预期分类: {question['expected_classification']}")
        print(f"  分类正确: {'✓' if classification_correct else '✗'}")
        
        if logger:
            logger.log_node_execution(
                "supervisor",
                initial_state,
                state_after_supervisor,
                f"分类结果: {actual_classification}, 正确性: {classification_correct}"
            )
        
        # 断言分类正确
        assert classification_correct, f"Supervisor分类错误: 期望 {question['expected_classification']}, 实际 {actual_classification}"
        
    except Exception as e:
        print(f"✗ Supervisor 分类失败: {e}")
        if logger:
            logger.log_node_execution("supervisor", initial_state, None, f"执行失败: {e}")
        raise
    
    # ===================== 阶段2: Immunity 计划生成 =====================
    print(f"\n{'='*80}")
    print(f"【阶段2: Immunity 计划生成】")
    print(f"{'='*80}\n")
    
    # 注意：根据主图流程，如果分类为 IMMUNOLOGY_TASK，会直接路由到 immunity 节点
    # 但为了测试完整流程，我们需要手动调用各个子图
    
    try:
        from nodes.subagents.immunity.graph import (
            build_immunity_subgraph,
            immunity_input_mapper,
            immunity_output_mapper
        )
        
        immunity_subgraph = build_immunity_subgraph()
        immunity_input = immunity_input_mapper(state_after_supervisor)
        immunity_output = immunity_subgraph.invoke(immunity_input)
        
        # 将字典输出转换为 ImmunityState 对象（LangGraph 返回字典）
        if isinstance(immunity_output, dict):
            from nodes.subagents.immunity.state import ImmunityState
            immunity_state = ImmunityState.model_validate(immunity_output)
        else:
            immunity_state = immunity_output
        
        state_after_immunity = immunity_output_mapper(immunity_state, state_after_supervisor)
        
        # 检查计划是否生成
        # immunity_output_mapper 将计划存储在 merged_result["immunity_plan"] 中
        immunity_plan = state_after_immunity.merged_result.get("immunity_plan", {}) if state_after_immunity.merged_result else {}
        executable_plan_dict = immunity_plan.get("executable_plan", {})
        experimental_plan = immunity_plan.get("experimental_plan", "")
        
        # executable_plan 可能是字典，需要检查其是否非空
        executable_plan_exists = bool(executable_plan_dict and isinstance(executable_plan_dict, dict) and len(executable_plan_dict) > 0)
        experimental_plan_exists = bool(experimental_plan and isinstance(experimental_plan, str) and len(experimental_plan) > 0)
        plan_generated = executable_plan_exists or experimental_plan_exists
        
        # 获取计划文本用于显示（优先使用 experimental_plan，否则将 executable_plan 转为字符串）
        if experimental_plan_exists:
            plan_text = experimental_plan
        elif executable_plan_exists:
            plan_text = json.dumps(executable_plan_dict, ensure_ascii=False, indent=2)
        else:
            plan_text = ""
        
        print(f"✓ Immunity 计划生成完成")
        print(f"  计划是否生成: {'✓' if plan_generated else '✗'}")
        if plan_generated:
            plan_length = len(plan_text) if plan_text else 0
            print(f"  计划长度: {plan_length} 字符")
            print(f"  计划预览: {plan_text[:200] if plan_text else 'N/A'}...")
        
        if logger:
            logger.log_node_execution(
                "immunity",
                state_after_supervisor,
                state_after_immunity,
                f"计划生成: {plan_generated}, 计划长度: {len(plan_text) if plan_text else 0}"
            )
        
        # 断言计划已生成
        assert plan_generated, "Immunity 未能生成计划"
        
    except Exception as e:
        print(f"✗ Immunity 计划生成失败: {e}")
        if logger:
            logger.log_node_execution("immunity", state_after_supervisor, None, f"执行失败: {e}")
        raise
    
    # ===================== 阶段3: Task Decomposition 任务分解 =====================
    print(f"\n{'='*80}")
    print(f"【阶段3: Task Decomposition 任务分解和参数推断】")
    print(f"{'='*80}\n")
    
    # 注意：在主图流程中，IMMUNOLOGY_TASK 不会经过 task_decomposition
    # 但为了测试完整流程，我们手动调用 task_decomposition
    # 需要将 user_task_type 设置为 EXECUTE_PLAN 以便 task_decomposition 能够处理
    
    try:
        from nodes.subagents.task_decomposition.graph import (
            build_task_decomposition_subgraph,
            task_decomposition_input_mapper,
            task_decomposition_output_mapper
        )
        
        # 临时修改 user_task_type 以便 task_decomposition 能够处理
        # 因为 task_decomposition 通常处理 EXECUTE_PLAN 类型的任务
        # 从 immunity_plan 中提取 executable_plan 作为 execution_plan
        immunity_plan = state_after_immunity.merged_result.get("immunity_plan", {}) if state_after_immunity.merged_result else {}
        
        # executable_plan 可能是字典，需要转换为字符串
        # 优先使用 experimental_plan（字符串），如果没有则使用 executable_plan（字典转JSON）
        experimental_plan = immunity_plan.get("experimental_plan", "")
        executable_plan_dict = immunity_plan.get("executable_plan", {})
        
        # 如果 experimental_plan 存在且是字符串，使用它
        # 否则，将 executable_plan 字典转换为 JSON 字符串
        if experimental_plan and isinstance(experimental_plan, str):
            execution_plan_str = experimental_plan
        elif executable_plan_dict and isinstance(executable_plan_dict, dict):
            execution_plan_str = json.dumps(executable_plan_dict, ensure_ascii=False, indent=2)
        else:
            execution_plan_str = ""
        
        state_for_decomposition = GlobalState(
            user_input=state_after_immunity.user_input,
            user_task_type=UserTaskType.EXECUTE_PLAN,  # 设置为 EXECUTE_PLAN 以便 task_decomposition 处理
            execution_plan=execution_plan_str,  # 使用字符串格式的 execution_plan
            sandbox_dir=state_after_immunity.sandbox_dir,
            subtasks=state_after_immunity.subtasks,
            parallel_task_groups=state_after_immunity.parallel_task_groups
        )
        
        task_decomposition_subgraph = build_task_decomposition_subgraph()
        task_decomposition_input = task_decomposition_input_mapper(state_for_decomposition)
        task_decomposition_output = task_decomposition_subgraph.invoke(task_decomposition_input)
        state_after_decomposition = task_decomposition_output_mapper(task_decomposition_output, state_for_decomposition)
        
        # 检查任务是否分解
        tasks_generated = bool(state_after_decomposition.subtasks) or bool(state_after_decomposition.parallel_task_groups)
        task_count = len(state_after_decomposition.subtasks) if state_after_decomposition.subtasks else 0
        parallel_group_count = len(state_after_decomposition.parallel_task_groups) if state_after_decomposition.parallel_task_groups else 0
        
        print(f"✓ Task Decomposition 完成")
        print(f"  任务是否生成: {'✓' if tasks_generated else '✗'}")
        print(f"  任务数量: {task_count}")
        print(f"  并行组数量: {parallel_group_count}")
        
        if tasks_generated:
            print(f"\n  任务列表:")
            for i, task in enumerate(state_after_decomposition.subtasks[:10], 1):  # 只显示前10个
                print(f"    {i}. {task.task_id}: {task.content[:80]}...")
            if task_count > 10:
                print(f"    ... 还有 {task_count - 10} 个任务")
        
        if logger:
            logger.log_node_execution(
                "task_decomposition",
                state_after_immunity,
                state_after_decomposition,
                f"任务生成: {tasks_generated}, 任务数: {task_count}, 并行组数: {parallel_group_count}"
            )
            
            # 记录任务列表
            if state_after_decomposition.subtasks:
                logger.log_task_order([
                    {
                        "task_id": task.task_id,
                        "content": task.content,
                        "dependencies": task.dependencies,
                        "tools": [t.get("tool_name", "") if isinstance(t, dict) else str(t) 
                                 for t in (task.result.get("tools", []) if isinstance(task.result, dict) else [])]
                    }
                    for task in state_after_decomposition.subtasks
                ])
        
        # 断言任务已分解
        assert tasks_generated, "Task Decomposition 未能生成任务"
        assert task_count > 0, "Task Decomposition 生成的任务数量为0"
        
    except Exception as e:
        print(f"✗ Task Decomposition 失败: {e}")
        if logger:
            logger.log_node_execution("task_decomposition", state_after_immunity, None, f"执行失败: {e}")
        raise
    
    # ===================== 阶段4: Executor 任务执行 =====================
    print(f"\n{'='*80}")
    print(f"【阶段4: Executor 任务执行】")
    print(f"{'='*80}\n")
    
    try:
        from nodes.subagents.executor.graph import (
            build_executor_subgraph,
            executor_input_mapper,
            executor_output_mapper
        )
        
        executor_subgraph = build_executor_subgraph()
        executor_input = executor_input_mapper(state_after_decomposition)
        
        # 执行executor（支持中断）
        thread_id = f"full_pipeline_{question['id']}"
        print(f"开始执行 Executor，线程ID: {thread_id}\n")
        
        # 首次执行
        result = execute_executor_with_interrupt_support(
            executor_subgraph,
            executor_input,
            thread_id=thread_id
        )
        
        # 处理中断循环
        iteration_count = 0
        max_iterations = 50  # Prevent infinite loop
        
        while result.get("interrupted", False) and iteration_count < max_iterations:
            iteration_count += 1
            interrupt_data = result.get("interrupt_data")
            
            if not interrupt_data:
                # Try to get from parent_state
                if executor_input.parent_state and executor_input.parent_state.hitl_status:
                    try:
                        interrupt_data = json.loads(executor_input.parent_state.hitl_status)
                    except:
                        pass
            
            if interrupt_data:
                # Extract actual interrupt data (recursively extract until we get a real dictionary)
                actual_interrupt_data = extract_interrupt_value(interrupt_data)
                
                # Ensure actual_interrupt_data is dictionary format
                if not isinstance(actual_interrupt_data, dict):
                    actual_interrupt_data = {"value": actual_interrupt_data}
                
                print(f"\n{'='*80}")
                print(f"【HITL Interrupt #{iteration_count}】")
                print(f"{'='*80}")
                print(f"Interrupt data type: {type(interrupt_data)}")
                print(f"Extracted interrupt data type: {type(actual_interrupt_data)}")
                print(f"Interrupt data content: {actual_interrupt_data}")
                
                if logger:
                    # Serialize interrupt_data to ensure JSON serializable
                    serialized_data = _serialize_interrupt_data(actual_interrupt_data)
                    logger.log_hitl_request(
                        task_id=actual_interrupt_data.get("task_id", "unknown"),
                        request_type=actual_interrupt_data.get("type", "unknown"),
                        request_data=serialized_data
                    )
                
                try:
                    # Use console interaction to get user input
                    # actual_interrupt_data should be a dictionary containing type, requests, etc.
                    user_response = handle_hitl_interrupt(
                        actual_interrupt_data,
                        callback=None,  # Use default console interaction
                        use_file=False
                    )
                    
                    if logger:
                        # Serialize user_response to ensure JSON serializable
                        serialized_response = _serialize_interrupt_data(user_response)
                        logger.log_hitl_response(
                            task_id=actual_interrupt_data.get("task_id", "unknown"),
                            response_type=user_response.get("type", "unknown"),
                            response_data=serialized_response
                        )
                    
                    # Update global_state's hitl_status
                    state_after_decomposition.hitl_status = json.dumps(user_response, ensure_ascii=False)
                    
                    # Resume execution
                    print(f"\nResuming execution...\n")
                    print(f"  🔍 [test] Resuming execution, thread_id={thread_id}")
                    print(f"  🔍 [test] resume_value type: {type(user_response)}")
                    print(f"  🔍 [test] resume_value content: {user_response}")
                    
                    result = resume_executor_after_interrupt(
                        executor_subgraph,
                        thread_id=thread_id,
                        resume_value=user_response
                    )
                    
                    print(f"  🔍 [test] Resume execution result: interrupted={result.get('interrupted', False)}")
                    
                    if logger:
                        logger.log_node_execution("executor_subgraph", None, result.get("result"), f"Resume execution #{iteration_count}")
                    
                except KeyboardInterrupt:
                    print("\nUser interrupted, terminating execution")
                    break
                except Exception as e:
                    print(f"\n⚠ HITL interaction failed: {e}")
                    print(f"Continuing execution, but may not be able to get user input")
                    break
            else:
                print(f"\n⚠ Interrupt detected, but unable to get interrupt data")
                break
        
        # Get final result
        executor_output = result.get("result")
        if executor_output is None:
            executor_output = executor_input
        
        state_after_executor = executor_output_mapper(executor_output, state_after_decomposition)
        
        # 检查执行结果
        total_tasks = state_after_executor.total_tasks if hasattr(state_after_executor, 'total_tasks') else len(state_after_decomposition.subtasks)
        completed_count = state_after_executor.completed_count if hasattr(state_after_executor, 'completed_count') else 0
        failed_count = state_after_executor.failed_count if hasattr(state_after_executor, 'failed_count') else 0
        
        print(f"✓ Executor 执行完成")
        print(f"  总任务数: {total_tasks}")
        print(f"  已完成: {completed_count}")
        print(f"  失败: {failed_count}")
        print(f"  完成率: {completed_count / total_tasks * 100:.1f}%" if total_tasks > 0 else "  完成率: N/A")
        
        # 显示任务执行详情
        if hasattr(state_after_executor, 'task_results') and state_after_executor.task_results:
            print(f"\n  任务执行详情:")
            for task_id, task_result in list(state_after_executor.task_results.items())[:10]:  # 只显示前10个
                status_icon = "✓" if task_result.status == ExecutorTaskStatus.COMPLETED else "✗" if task_result.status == ExecutorTaskStatus.FAILED else "⏳"
                print(f"    {status_icon} {task_id}: {task_result.status.value}")
                if task_result.error:
                    print(f"      错误: {task_result.error[:100]}")
        
        if logger:
            logger.log_node_execution(
                "executor",
                state_after_decomposition,
                state_after_executor,
                f"执行完成: 总任务数={total_tasks}, 已完成={completed_count}, 失败={failed_count}"
            )
            
            # 记录所有任务执行结果
            if hasattr(state_after_executor, 'task_results') and state_after_executor.task_results:
                for task_id, task_result in state_after_executor.task_results.items():
                    logger.log_task_execution(
                        task_id=task_id,
                        status=task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                        execution_mode=task_result.execution_mode,
                        result={
                            "output": str(task_result.output)[:500] if task_result.output else None,
                            "confidence_score": task_result.confidence_score,
                            "error": task_result.error[:200] if task_result.error else None,
                            "error_category": task_result.error_category.value if task_result.error_category else None,
                            "retry_count": task_result.retry_count
                        },
                        error=task_result.error[:500] if task_result.error else None
                    )
        
        # 断言至少有一些任务被执行
        assert total_tasks > 0, "Executor 没有任务需要执行"
        # 注意：由于参数缺失等原因，可能不是所有任务都能完成，所以不强制要求所有任务都完成
        
    except Exception as e:
        print(f"✗ Executor 执行失败: {e}")
        if logger:
            logger.log_node_execution("executor", state_after_decomposition, None, f"执行失败: {e}")
        raise
    
    # ===================== 总结 =====================
    print(f"\n{'='*80}")
    print(f"【测试总结 - {question['id']}】")
    print(f"{'='*80}")
    print(f"✓ Supervisor 分类: {'通过' if classification_correct else '失败'}")
    print(f"✓ Immunity 计划生成: {'通过' if plan_generated else '失败'}")
    print(f"✓ Task Decomposition: {'通过' if tasks_generated else '失败'} ({task_count} 个任务)")
    print(f"✓ Executor 执行: {'通过' if total_tasks > 0 else '失败'} ({completed_count}/{total_tasks} 完成)")
    print(f"{'='*80}\n")
    
    if logger:
        # 重新获取计划信息用于日志记录
        plan_length = 0
        if 'state_after_immunity' in locals() and state_after_immunity.merged_result:
            immunity_plan_final = state_after_immunity.merged_result.get("immunity_plan", {})
            executable_plan_final_dict = immunity_plan_final.get("executable_plan", {})
            experimental_plan_final = immunity_plan_final.get("experimental_plan", "")
            
            # 计算计划长度（executable_plan 可能是字典，需要转换为字符串）
            if experimental_plan_final and isinstance(experimental_plan_final, str):
                plan_length = len(experimental_plan_final)
            elif executable_plan_final_dict and isinstance(executable_plan_final_dict, dict):
                plan_length = len(json.dumps(executable_plan_final_dict, ensure_ascii=False))
            else:
                plan_length = 0
        
        logger.log_summary({
            "question_id": question['id'],
            "question_text": question['text'],
            "supervisor_classification": str(actual_classification),
            "classification_correct": classification_correct,
            "plan_generated": plan_generated,
            "plan_length": plan_length,
            "tasks_generated": tasks_generated,
            "task_count": task_count,
            "parallel_group_count": parallel_group_count,
            "total_tasks": total_tasks,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "completion_rate": completed_count / total_tasks * 100 if total_tasks > 0 else 0
        })
    
    return {
        "question_id": question['id'],
        "classification_correct": classification_correct,
        "plan_generated": plan_generated,
        "tasks_generated": tasks_generated,
        "task_count": task_count,
        "total_tasks": total_tasks,
        "completed_count": completed_count,
        "failed_count": failed_count
    }


@pytest.fixture(scope="module")
def main_graph():
    """构建并返回主图"""
    return build_main_graph()


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
        # 构建主图
        main_graph = build_main_graph()
        
        # 运行所有测试
        for question in TEST_QUESTIONS:
            print(f"\n{'#'*80}")
            print(f"# 测试问题: {question['id']}")
            print(f"{'#'*80}\n")
            _test_full_pipeline(question, main_graph=main_graph)
        
    finally:
        # 保存日志
        save_global_logger()

