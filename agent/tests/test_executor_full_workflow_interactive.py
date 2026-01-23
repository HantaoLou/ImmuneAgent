"""
Executor 完整工作流交互式测试用例

测试使用 igblast 和 metabcr 工具的工作流

任务列表（2个任务）：
1. task_001 - 使用 igblast 工具批量分析 V(D)J 重组（analyze_vdj_batch）
2. task_002 - 使用 metabcr 工具预测抗体-抗原结合亲和力（metabcr）

这是一个交互式测试，会在触发HITL时暂停，等待用户在控制台输入参数。

运行方式：pytest tests/test_executor_full_workflow_interactive.py::test_full_protein_mrna_workflow -v -s
（-s 参数用于显示print输出，-v 用于详细输出）
"""

import os
import pytest
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from nodes.subagents.executor.graph import (
    build_executor_subgraph,
    ExecutorState,
    ExecutorTaskStatus,
    TaskExecutionResult,
    executor_input_mapper,
    executor_output_mapper,
    execute_executor_with_interrupt_support,
    resume_executor_after_interrupt
)
from state import SubTask, UserTaskType, GlobalState, ParallelTaskGroup
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


def _ensure_executor_state(result):
    """确保结果是 ExecutorState 对象"""
    if isinstance(result, dict):
        return ExecutorState.model_validate(result)
    return result


def _set_parent_state_safely(executor_state: ExecutorState, global_state: GlobalState):
    """安全地设置 parent_state"""
    object.__setattr__(executor_state, 'parent_state', global_state)


def _serialize_interrupt_data(data: Any) -> Dict[str, Any]:
    """
    将 interrupt_data 中的 Interrupt 对象转换为可序列化的字典
    
    Args:
        data: 可能包含 Interrupt 对象的数据
        
    Returns:
        可序列化的字典
    """
    if data is None:
        return {}
    
    # 如果是 Interrupt 对象（通常有 value 和 id 属性）
    if hasattr(data, 'value') and hasattr(data, 'id'):
        return {
            "type": "Interrupt",
            "id": str(getattr(data, 'id', '')),
            "value": _serialize_interrupt_data(getattr(data, 'value', {}))
        }
    
    # 如果是字典，递归处理
    if isinstance(data, dict):
        return {k: _serialize_interrupt_data(v) for k, v in data.items()}
    
    # 如果是列表，递归处理
    if isinstance(data, list):
        return [_serialize_interrupt_data(item) for item in data]
    
    # 如果是其他可序列化类型，直接返回
    try:
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        # 如果无法序列化，转换为字符串
        return str(data)


def _create_executor_state_for_test(global_state: GlobalState, **kwargs) -> ExecutorState:
    """为测试创建 ExecutorState"""
    try:
        return ExecutorState(
            subtasks=global_state.subtasks,
            parallel_task_groups=global_state.parallel_task_groups,
            sandbox_dir=kwargs.get("sandbox_dir", global_state.sandbox_dir),
            parent_state=global_state,
            max_parallel_tasks=kwargs.get("max_parallel_tasks", 3),
            max_retries=kwargs.get("max_retries", 5),
        )
    except Exception as e:
        executor_state = ExecutorState.model_construct(
            subtasks=global_state.subtasks,
            parallel_task_groups=global_state.parallel_task_groups,
            sandbox_dir=kwargs.get("sandbox_dir", global_state.sandbox_dir),
            max_parallel_tasks=kwargs.get("max_parallel_tasks", 3),
            max_retries=kwargs.get("max_retries", 5),
        )
        _set_parent_state_safely(executor_state, global_state)
        return executor_state


@pytest.fixture(scope="module", autouse=True)
def setup_global_logger():
    """初始化全局日志记录器"""
    test_file_name = Path(__file__).stem
    init_global_logger(test_file_name)
    yield
    save_global_logger()


def create_full_workflow_tasks() -> list[SubTask]:
    """
    创建完整工作流的任务列表（使用 igblast 和 metabcr 工具）
    
    流程：
    1. task_001 - 使用 igblast 工具批量分析 V(D)J 重组
    2. task_002 - 使用 metabcr 工具预测抗体-抗原结合亲和力
    """
    tasks = []
    
    # Task 1: 使用 igblast 工具分析 V(D)J 重组
    task1 = SubTask(
        task_id="task_001",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="Use igblast to batch analyze V(D)J recombination from antibody sequences",
        result={
            "tools": [
                {
                    "tool_name": "analyze_vdj_batch",
                    "name": "analyze_vdj_batch",
                    "service": "igblast",
                    "description": "Batch analyze V(D)J recombination using IgBLAST+ChangeO, output AIRR format results"
                }
            ],
            "inputs": [
                "analyze_vdj_batch.sequences",  # FASTA file with antibody sequences
                "analyze_vdj_batch.organism",   # human, mouse, rabbit, rat, rhesus, pig
                "analyze_vdj_batch.receptor_type",  # Ig or TCR
                "analyze_vdj_batch.locus"  # IGH, IGK, IGL, TRA, TRB, TRG, TRD
            ],
            "outputs": ["airr_results"]  # AIRR format results
        },
        dependencies=[]
    )
    tasks.append(task1)
    
    # Task 2: 使用 metabcr 工具预测抗体-抗原结合亲和力
    task2 = SubTask(
        task_id="task_002",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="Use metabcr to predict antibody-antigen binding affinity",
        result={
            "tools": [
                {
                    "tool_name": "metabcr",
                    "name": "metabcr",
                    "service": "metabcr",
                    "description": "MetaBCR deep learning model predicts antibody-antigen binding affinity, stream progress"
                }
            ],
            "inputs": [
                "metabcr.antibody_file",  # CSV file with antibody sequences
                "metabcr.antigen_file",   # CSV file with antigen sequences
                "metabcr.antigen_name",   # flu, sars, rsv, hiv
                "metabcr.output_file_path"  # Output file path
            ],
            "outputs": ["binding_affinity_results"]
        },
        dependencies=["task_001"]  # May use results from task_001
    )
    tasks.append(task2)
    
    return tasks


def test_full_protein_mrna_workflow(executor_subgraph=None, request=None, test_case_logger=None):
    """
    完整工作流测试（使用 igblast 和 metabcr 工具）
    
    这是一个交互式测试，会在触发HITL时暂停，等待用户输入。
    测试包含2个任务：
    1. 使用 igblast 工具批量分析 V(D)J 重组
    2. 使用 metabcr 工具预测抗体-抗原结合亲和力
    """
    # 如果没有提供executor_subgraph，构建一个
    if executor_subgraph is None:
        executor_subgraph = build_executor_subgraph()
    
    # 获取日志记录器
    logger = test_case_logger
    if logger is None:
        global_logger = get_global_logger()
        if global_logger:
            test_case_name = request.node.name if request else "test_full_protein_mrna_workflow"
            logger = global_logger.get_test_case_logger(test_case_name)
    
    # 创建测试目录
    test_dir = Path("./sandbox/full_workflow_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"【完整工作流交互式测试】")
    print(f"{'='*80}")
    print(f"测试目录: {test_dir.absolute()}")
    print(f"此测试将使用2个任务：")
    print(f"  1. task_001 - 使用 igblast 工具批量分析 V(D)J 重组")
    print(f"  2. task_002 - 使用 metabcr 工具预测抗体-抗原结合亲和力")
    print(f"当触发HITL时，请在控制台输入所需参数")
    print(f"{'='*80}\n")
    
    # 创建任务列表
    tasks = create_full_workflow_tasks()
    
    if logger:
        logger.log_initial_state(None, f"创建了 {len(tasks)} 个任务")
        logger.log_task_order([
            {
                "task_id": task.task_id,
                "content": task.content,
                "dependencies": task.dependencies,
                "tools": [t.get("tool_name", "") if isinstance(t, dict) else str(t) for t in (task.result.get("tools", []) if isinstance(task.result, dict) else [])]
            }
            for task in tasks
        ])
    
    # 创建全局状态
    global_state = GlobalState(
        user_input="执行完整的蛋白抗原优化流程：下载流感NA蛋白序列，预测结构，识别功能位点，计算突变稳定性，设计优化序列",
        user_task_type=UserTaskType.EXECUTE_PLAN,
        subtasks=tasks,
        sandbox_dir=str(test_dir)
    )
    
    if logger:
        logger.log_initial_state(global_state, "初始 GlobalState")
    
    # 创建执行器状态
    try:
        executor_input = executor_input_mapper(global_state)
    except Exception as e:
        executor_input = _create_executor_state_for_test(global_state)
    _set_parent_state_safely(executor_input, global_state)
    
    if logger:
        logger.log_initial_state(executor_input, "转换后的 ExecutorState")
    
    # 执行工作流（支持HITL中断）
    thread_id = "full_workflow_test"
    print(f"\n开始执行工作流...")
    print(f"线程ID: {thread_id}\n")
    
    # 首次执行
    result = execute_executor_with_interrupt_support(
        executor_subgraph,
        executor_input,
        thread_id=thread_id
    )
    
    if logger:
        logger.log_node_execution("executor_subgraph", executor_input, result.get("result"), "首次执行")
    
    # 处理中断循环
    iteration_count = 0
    max_iterations = 50  # 防止无限循环
    
    while result.get("interrupted", False) and iteration_count < max_iterations:
        iteration_count += 1
        interrupt_data = result.get("interrupt_data")
        
        if not interrupt_data:
            # 尝试从parent_state获取
            if executor_input.parent_state and executor_input.parent_state.hitl_status:
                try:
                    interrupt_data = json.loads(executor_input.parent_state.hitl_status)
                except:
                    pass
        
        if interrupt_data:
            # 提取实际的中断数据（递归提取，直到得到真正的字典）
            def extract_interrupt_value(obj, max_depth=5):
                """递归提取中断值，直到得到字典"""
                if max_depth <= 0:
                    return obj
                
                # 如果对象是None，直接返回
                if obj is None:
                    return obj
                
                # 如果是 Interrupt 对象，提取其 value
                if hasattr(obj, 'value'):
                    value = getattr(obj, 'value', None)
                    if value is not None:
                        return extract_interrupt_value(value, max_depth - 1)
                    else:
                        # value是None，返回对象本身
                        return obj
                
                # 如果是字典，检查是否有 'value' 字段
                if isinstance(obj, dict):
                    if 'value' in obj:
                        value = obj['value']
                        if value is not None:
                            return extract_interrupt_value(value, max_depth - 1)
                        else:
                            # value是None，如果已经有type字段，返回字典本身
                            if 'type' in obj:
                                return obj
                    # 如果已经是正确的格式（有 'type' 字段），直接返回
                    if 'type' in obj:
                        return obj
                    return obj
                
                # 如果是 tuple，提取第二个元素
                if isinstance(obj, tuple):
                    if len(obj) >= 2:
                        return extract_interrupt_value(obj[1], max_depth - 1)
                    elif len(obj) == 1:
                        return extract_interrupt_value(obj[0], max_depth - 1)
                    return obj
                
                return obj
            
            actual_interrupt_data = extract_interrupt_value(interrupt_data)
            
            # 确保 actual_interrupt_data 是字典格式
            if not isinstance(actual_interrupt_data, dict):
                actual_interrupt_data = {"value": actual_interrupt_data}
            
            print(f"\n{'='*80}")
            print(f"【HITL 中断 #{iteration_count}】")
            print(f"{'='*80}")
            print(f"原始中断数据类型: {type(interrupt_data)}")
            print(f"提取后的中断数据类型: {type(actual_interrupt_data)}")
            print(f"中断数据内容: {actual_interrupt_data}")

            if logger:
                # 序列化 interrupt_data，确保可以 JSON 序列化
                serialized_data = _serialize_interrupt_data(actual_interrupt_data)
                logger.log_hitl_request(
                    task_id=actual_interrupt_data.get("task_id", "unknown"),
                    request_type=actual_interrupt_data.get("type", "unknown"),
                    request_data=serialized_data
                )
            
            try:
                # 使用控制台交互获取用户输入
                # actual_interrupt_data 应该是包含 type、requests 等字段的字典
                print(f"  🔍 [test] 准备调用 handle_hitl_interrupt，interrupt_data type: {type(actual_interrupt_data)}")
                user_response = handle_hitl_interrupt(
                    actual_interrupt_data,
                    callback=None,  # 使用默认控制台交互
                    use_file=False
                )
                print(f"  ✓ [test] handle_hitl_interrupt 成功返回，response type: {type(user_response)}")
                
                if logger:
                    try:
                        # 序列化 user_response，确保可以 JSON 序列化
                        serialized_response = _serialize_interrupt_data(user_response)
                        logger.log_hitl_response(
                            task_id=actual_interrupt_data.get("task_id", "unknown"),
                            response_type=user_response.get("type", "unknown"),
                            response_data=serialized_response
                        )
                    except Exception as log_e:
                        print(f"  ⚠ [test] 记录HITL响应日志失败: {log_e}")
                
                # 更新global_state的hitl_status
                try:
                    global_state.hitl_status = json.dumps(user_response, ensure_ascii=False)
                except Exception as json_e:
                    print(f"  ⚠ [test] 序列化user_response失败: {json_e}")
                    # 尝试使用安全的序列化方法
                    try:
                        global_state.hitl_status = json.dumps(_serialize_interrupt_data(user_response), ensure_ascii=False)
                    except Exception as json_e2:
                        print(f"  ⚠ [test] 安全序列化也失败: {json_e2}")
                
                # 恢复执行
                print(f"\n恢复执行...\n")
                print(f"  🔍 [test] 恢复执行，thread_id={thread_id}")
                print(f"  🔍 [test] resume_value 类型: {type(user_response)}")
                print(f"  🔍 [test] resume_value 内容: {user_response}")
                
                try:
                    result = resume_executor_after_interrupt(
                        executor_subgraph,
                        thread_id=thread_id,
                        resume_value=user_response
                    )
                    print(f"  🔍 [test] 恢复执行结果: interrupted={result.get('interrupted', False)}")
                except Exception as resume_e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    print(f"  ✗ [test] resume_executor_after_interrupt 失败: {resume_e}")
                    print(f"  {error_traceback}")
                    raise
                
                if logger:
                    try:
                        logger.log_node_execution("executor_subgraph", None, result.get("result"), f"恢复执行 #{iteration_count}")
                    except Exception as log_e:
                        print(f"  ⚠ [test] 记录执行日志失败: {log_e}")
                
            except KeyboardInterrupt:
                print("\n用户退出，终止执行")
                break
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"\n⚠ HITL交互处理失败: {e}")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误堆栈:\n{error_traceback}")
                print(f"继续执行，但可能无法获取用户输入")
                # 不再直接break，而是尝试继续
                # 如果错误严重，让异常继续传播
                if "NoneType" in str(e) and "value" in str(e):
                    # 这是已知的错误，尝试继续
                    print(f"  ℹ [test] 检测到NoneType.value错误，尝试继续执行...")
                    break
                else:
                    # 其他错误，重新抛出
                    raise
        else:
            print(f"\n⚠ 检测到中断，但无法获取中断数据")
            break
    
    # 获取最终结果
    final_state = result.get("result")
    if final_state is None:
        # 尝试从最后一次执行获取
        if executor_input.parent_state:
            try:
                executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
                final_result = executor_subgraph.invoke(
                    executor_input_dict,
                    config={"configurable": {"thread_id": thread_id}}
                )
                final_state = _ensure_executor_state(final_result)
            except:
                final_state = executor_input
    
    if final_state is None:
        final_state = executor_input
    
    final_state = _ensure_executor_state(final_state)
    
    # 记录最终结果
    if logger:
        logger.log_node_execution("executor_subgraph", executor_input, final_state, "最终状态")
        
        # 记录所有任务执行结果
        for task_id, task_result in final_state.task_results.items():
            logger.log_task_execution(
                task_id=task_id,
                status=task_result.status.value if (task_result.status and hasattr(task_result.status, 'value')) else str(task_result.status) if task_result.status else "None",
                execution_mode=task_result.execution_mode,
                result={
                    "output": str(task_result.output)[:500] if task_result.output else None,
                    "confidence_score": task_result.confidence_score,
                    "error": task_result.error[:200] if task_result.error else None,
                    "error_category": task_result.error_category.value if (task_result.error_category and hasattr(task_result.error_category, 'value')) else str(task_result.error_category) if task_result.error_category else None,
                    "retry_count": task_result.retry_count
                },
                error=task_result.error[:500] if task_result.error else None
            )
            
            # 如果有生成的代码，记录
            if task_result.code:
                logger.log_codeact_output(
                    task_id=task_id,
                    generated_code=task_result.code[:1000],
                    execution_result={
                        "status": task_result.status.value if (task_result.status and hasattr(task_result.status, 'value')) else str(task_result.status) if task_result.status else "None",
                        "output": str(task_result.output)[:500] if task_result.output else None
                    }
                )
        
        # 记录HITL交互
        for task_id, hitl_request in final_state.hitl_requests.items():
            logger.log_hitl_request(
                task_id=task_id,
                request_type=hitl_request.get("type", "unknown"),
                request_data=hitl_request
            )
        
        for task_id, hitl_response in final_state.hitl_responses.items():
            logger.log_hitl_response(
                task_id=task_id,
                response_type="response",
                response_data=hitl_response
            )
        
        # 记录总结
        logger.log_summary({
            "total_tasks": final_state.total_tasks,
            "completed": final_state.completed_count,
            "failed": final_state.failed_count,
            "hitl_requests_count": len(final_state.hitl_requests),
            "hitl_responses_count": len(final_state.hitl_responses),
            "interrupt_iterations": iteration_count,
            "tools_used": 9,  # 9个任务，使用多个工具
            "tasks_count": len(tasks)
        })
    
    # 打印执行总结
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
    for task in tasks:
        task_result = final_state.task_results.get(task.task_id)
        if task_result:
            status_icon = "✓" if task_result.status == ExecutorTaskStatus.COMPLETED else "✗" if task_result.status == ExecutorTaskStatus.FAILED else "⏳"
            status_str = task_result.status.value if (task_result.status and hasattr(task_result.status, 'value')) else str(task_result.status) if task_result.status else "None"
            print(f"  {status_icon} {task.task_id}: {status_str}")
            if task_result.error:
                print(f"     错误: {task_result.error[:100]}")
            if task_result.retry_count > 0:
                print(f"     重试次数: {task_result.retry_count}")
        else:
            print(f"  ? {task.task_id}: 未执行")
    
    print(f"{'='*80}\n")
    
    # 验证基本结果
    assert final_state.total_tasks == len(tasks), f"任务数不匹配: {final_state.total_tasks} != {len(tasks)}"
    assert final_state.completed_count + final_state.failed_count <= final_state.total_tasks, "完成+失败数不应超过总任务数"
    
    print(f"✓ 完整工作流测试完成")


# 为了支持pytest，需要添加fixture
@pytest.fixture(scope="module")
def executor_subgraph():
    """构建并返回 Executor 子图"""
    return build_executor_subgraph()


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
        # 构建executor子图
        executor_subgraph = build_executor_subgraph()
        
        # 运行测试
        test_full_protein_mrna_workflow(executor_subgraph=executor_subgraph)
        
    finally:
        # 保存日志
        save_global_logger()

