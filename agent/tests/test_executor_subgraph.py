"""
Executor Subgraph 单独测试用例

测试 executor subgraph 的独立功能，重点测试：
1. 参数推断功能（使用 LLM 推断参数值）
2. HITL 中断功能（参数请求和结果确认）
3. 任务执行流程
4. 依赖管理
5. 并行执行控制

运行方式：pytest tests/test_executor_subgraph.py -v
"""

import os
import pytest
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

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
    ErrorCategory,
    executor_input_mapper,
    executor_output_mapper,
    execute_executor_with_interrupt_support,
    resume_executor_after_interrupt
)
from state import SubTask, UserTaskType, GlobalState

# 导入测试日志记录器
try:
    from test_logger import init_global_logger, get_global_logger, save_global_logger, TestCaseLogger
except ImportError:
    # 如果直接导入失败，尝试从 tests 模块导入
    import sys
    test_dir = Path(__file__).parent
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
    from test_logger import init_global_logger, get_global_logger, save_global_logger, TestCaseLogger


def _ensure_executor_state(result):
    """确保结果是 ExecutorState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        return ExecutorState.model_validate(result)
    return result

def _set_parent_state_safely(executor_state: ExecutorState, global_state: GlobalState):
    """安全地设置 parent_state（绕过 Pydantic 验证）"""
    object.__setattr__(executor_state, 'parent_state', global_state)

def _create_executor_state_for_test(global_state: GlobalState, **kwargs) -> ExecutorState:
    """为测试创建 ExecutorState（处理 Pydantic v2 的嵌套模型验证）"""
    # 直接创建 ExecutorState，让 Pydantic 处理嵌套模型
    # 由于 ExecutorState 已经配置了 from_attributes=True，可以直接传递对象
    try:
        return ExecutorState(
            subtasks=global_state.subtasks,
            parallel_task_groups=global_state.parallel_task_groups,
            sandbox_dir=kwargs.get("sandbox_dir", global_state.sandbox_dir),
            parent_state=global_state,
            max_parallel_tasks=kwargs.get("max_parallel_tasks", 3),
            max_retries=kwargs.get("max_retries", 2),
        )
    except Exception as e:
        # 如果直接创建失败，使用 model_construct 来绕过验证（仅用于测试）
        # 注意：model_construct 不会进行验证，所以我们需要确保数据正确
        executor_state = ExecutorState.model_construct(
            subtasks=global_state.subtasks,
            parallel_task_groups=global_state.parallel_task_groups,
            sandbox_dir=kwargs.get("sandbox_dir", global_state.sandbox_dir),
            max_parallel_tasks=kwargs.get("max_parallel_tasks", 3),
            max_retries=kwargs.get("max_retries", 2),
        )
        # 使用 object.__setattr__ 绕过 Pydantic 的验证机制来设置 parent_state
        _set_parent_state_safely(executor_state, global_state)
        return executor_state


@pytest.fixture(scope="module")
def executor_subgraph():
    """构建并返回 Executor 子图"""
    return build_executor_subgraph()


@pytest.fixture(scope="module", autouse=True)
def setup_global_logger():
    """初始化全局日志记录器（模块级别）"""
    # 初始化全局日志记录器
    test_file_name = Path(__file__).stem  # 获取文件名（不含扩展名）
    init_global_logger(test_file_name)
    yield
    # 测试模块结束时保存日志
    save_global_logger()


@pytest.fixture(autouse=True)
def test_case_logger(request):
    """为每个测试用例创建日志记录器"""
    global_logger = get_global_logger()
    if global_logger:
        test_case_name = request.node.name
        logger = global_logger.get_test_case_logger(test_case_name)
        yield logger
        # 测试用例结束时完成记录
        global_logger.finish_test_case(test_case_name)
    else:
        # 如果没有全局日志记录器，返回一个空的记录器（向后兼容）
        yield None


@pytest.fixture
def sample_global_state():
    """示例全局状态"""
    return GlobalState(
        user_input="执行抗体分析任务",
        user_task_type=UserTaskType.EXECUTE_PLAN,
        subtasks=[],
        sandbox_dir="./sandbox"
    )


@pytest.fixture
def sample_task_with_tool():
    """带工具的任务（需要参数推断）"""
    return SubTask(
        task_id="test_task_001",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="搜索COVID-19相关的抗体库数据，组织类型为血液",
        result={
            "tools": [
                {
                    "tool_name": "search_airr_repertoires",
                    "name": "search_airr_repertoires",
                    "service": "airr",
                    "description": "搜索AIRR数据库中的抗体库"
                }
            ],
            "inputs": ["disease", "tissue"],
            "outputs": ["repertoire_data"]
        }
    )


@pytest.fixture
def sample_task_without_tool():
    """无工具的任务（codeact模式）"""
    return SubTask(
        task_id="test_task_002",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="计算1+1的结果",
        result={
            "tools": [],
            "inputs": [],
            "outputs": ["result"]
        }
    )


@pytest.fixture
def sample_task_with_missing_params():
    """需要用户提供参数的任务（触发HITL）"""
    return SubTask(
        task_id="test_task_003",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="分析抗体序列数据，需要提供序列文件",
        result={
            "tools": [
                {
                    "tool_name": "analyze_sequences",
                    "name": "analyze_sequences",
                    "service": "analysis",
                    "description": "分析抗体序列"
                }
            ],
            "inputs": ["sequence_file"],
            "outputs": ["analysis_result"]
        }
    )


@pytest.fixture
def sample_tasks_with_dependencies():
    """带依赖关系的任务列表"""
    task1 = SubTask(
        task_id="task_001",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="下载数据",
        result={"tools": [], "inputs": [], "outputs": ["data"]},
        dependencies=[]
    )
    task2 = SubTask(
        task_id="task_002",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="处理数据",
        result={"tools": [], "inputs": [], "outputs": ["processed_data"]},
        dependencies=["task_001"]
    )
    task3 = SubTask(
        task_id="task_003",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="生成报告",
        result={"tools": [], "inputs": [], "outputs": ["report"]},
        dependencies=["task_002"]
    )
    return [task1, task2, task3]


class TestExecutorSubgraphBasic:
    """Executor Subgraph 基础功能测试"""
    
    def test_subgraph_build(self, executor_subgraph):
        """测试子图构建是否成功"""
        assert executor_subgraph is not None
        print("✓ Executor Subgraph 构建成功")
    
    def test_subgraph_invoke_basic(self, executor_subgraph, sample_global_state, request, test_case_logger):
        """测试子图基本调用"""
        logger = test_case_logger
        
        # 使用 input_mapper 转换状态
        try:
            executor_input = executor_input_mapper(sample_global_state)
        except Exception as e:
            # 如果直接创建失败，使用 model_validate
            executor_input = _create_executor_state_for_test(sample_global_state)
            _set_parent_state_safely(executor_input, sample_global_state)  # 保持引用
        
        if logger:
            # 记录初始状态
            logger.log_initial_state(sample_global_state, "初始 GlobalState")
            logger.log_initial_state(executor_input, "转换后的 ExecutorState")
            logger.log_task_order([
                {"task_id": task.task_id, "content": task.content, "dependencies": task.dependencies}
                for task in executor_input.subtasks
            ])
        
        # 调用子图（提供 configurable 参数以满足 checkpointer 要求）
        # 临时保存 parent_state，然后移除以避免 LangGraph 序列化验证错误
        saved_parent_state = executor_input.parent_state
        executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
        executor_output = executor_subgraph.invoke(
            executor_input_dict,
            config={"configurable": {"thread_id": "test_thread"}}
        )
        # 恢复 parent_state
        if saved_parent_state:
            _set_parent_state_safely(executor_input, saved_parent_state)
        
        # 确保输出是 ExecutorState 对象
        result = _ensure_executor_state(executor_output)
        
        if logger:
            # 记录最终状态
            logger.log_node_execution("executor_subgraph", executor_input, result, "完整子图执行")
            
            # 记录任务执行结果
            for task_id, task_result in result.task_results.items():
                logger.log_task_execution(
                    task_id=task_id,
                    status=task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                    execution_mode=task_result.execution_mode,
                    result={"output": task_result.output, "confidence_score": task_result.confidence_score},
                    error=task_result.error
                )
            
            # 记录总结
            logger.log_summary({
                "total_tasks": result.total_tasks,
                "completed": result.completed_count,
                "failed": result.failed_count,
                "hitl_requests_count": len(result.hitl_requests),
                "hitl_responses_count": len(result.hitl_responses)
            })
        
        assert result is not None
        assert hasattr(result, 'subtasks')
        assert result.total_tasks == len(sample_global_state.subtasks)
        print(f"✓ Executor Subgraph 基本调用成功")


class TestParameterInference:
    """参数推断功能测试"""
    
    def test_parameter_inference_with_llm(self, executor_subgraph, sample_task_with_tool, request, test_case_logger):
        """测试使用 LLM 推断参数值"""
        logger = test_case_logger
        
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_with_tool],
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        if logger:
            logger.log_initial_state(executor_input, "转换后的 ExecutorState")
            logger.log_task_order([
                {"task_id": task.task_id, "content": task.content, "dependencies": task.dependencies}
                for task in executor_input.subtasks
            ])
        
        # 执行到参数推断节点
        # 由于是完整流程，我们需要执行整个子图
        # 但可以通过检查状态来验证参数推断
        
        # 手动调用参数推断逻辑（用于测试）
        from nodes.subagents.executor.graph import infer_parameters_node
        
        # 先初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input_before = executor_input
        executor_input = initialize_tasks_node(executor_input)
        if logger:
            logger.log_node_execution("initialize_tasks", executor_input_before, executor_input, "初始化任务")
        
        # 执行参数推断
        executor_input_before = executor_input
        executor_input = infer_parameters_node(executor_input)
        if logger:
            logger.log_node_execution("infer_parameters", executor_input_before, executor_input, "参数推断")
        
        # 验证参数推断结果
        task_result = executor_input.task_results.get(sample_task_with_tool.task_id)
        assert task_result is not None
        assert hasattr(task_result, 'parameters')
        assert hasattr(task_result, 'missing_parameters')
        
        if logger:
            # 记录参数推断结果
            logger.log_parameter_inference(
                task_id=sample_task_with_tool.task_id,
                parameters=task_result.parameters,
                missing_parameters=task_result.missing_parameters,
                llm_used=True
            )
            
            logger.log_summary({
                "parameter_inference_completed": True,
                "inferred_parameters_count": len(task_result.parameters),
                "missing_parameters_count": len(task_result.missing_parameters)
            })
        
        print(f"✓ 参数推断完成")
        print(f"  推断的参数: {task_result.parameters}")
        print(f"  缺失的参数: {task_result.missing_parameters}")
    
    def test_parameter_inference_without_tool(self, executor_subgraph, sample_task_without_tool):
        """测试无工具任务的参数推断（应该为空）"""
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_without_tool],
            sandbox_dir="./sandbox"
        )
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        # 初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input = initialize_tasks_node(executor_input)
        
        # 执行参数推断
        from nodes.subagents.executor.graph import infer_parameters_node
        executor_input = infer_parameters_node(executor_input)
        
        # 验证无工具任务的参数为空
        task_result = executor_input.task_results.get(sample_task_without_tool.task_id)
        assert task_result is not None
        assert task_result.parameters == {}
        assert task_result.missing_parameters == []
        
        print(f"✓ 无工具任务参数推断正确（参数为空）")


class TestHITLParameterRequest:
    """HITL 参数请求功能测试"""
    
    def test_hitl_parameter_request_trigger(self, executor_subgraph, sample_task_with_missing_params, request, test_case_logger):
        """测试触发 HITL 参数请求"""
        logger = test_case_logger
        
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_with_missing_params],
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        if logger:
            logger.log_initial_state(executor_input, "转换后的 ExecutorState")
            logger.log_task_order([
                {"task_id": task.task_id, "content": task.content, "dependencies": task.dependencies}
                for task in executor_input.subtasks
            ])
        
        # 初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input_before = executor_input
        executor_input = initialize_tasks_node(executor_input)
        if logger:
            logger.log_node_execution("initialize_tasks", executor_input_before, executor_input, "初始化任务")
        
        # 执行参数推断
        from nodes.subagents.executor.graph import infer_parameters_node
        executor_input_before = executor_input
        executor_input = infer_parameters_node(executor_input)
        if logger:
            logger.log_node_execution("infer_parameters", executor_input_before, executor_input, "参数推断")
        
        # 记录参数推断结果
        task_result = executor_input.task_results.get(sample_task_with_missing_params.task_id)
        if logger and task_result:
            logger.log_parameter_inference(
                task_id=sample_task_with_missing_params.task_id,
                parameters=task_result.parameters,
                missing_parameters=task_result.missing_parameters,
                llm_used=True
            )
        
        # 检查是否触发了 HITL 请求
        assert sample_task_with_missing_params.task_id in executor_input.hitl_requests
        hitl_request = executor_input.hitl_requests[sample_task_with_missing_params.task_id]
        assert hitl_request["type"] == "missing_parameters"
        assert "missing_parameters" in hitl_request
        
        if logger:
            # 记录 HITL 请求
            logger.log_hitl_request(
                task_id=sample_task_with_missing_params.task_id,
                request_type=hitl_request["type"],
                request_data=hitl_request
            )
            
            # 检查任务状态
            assert executor_input.task_status_map[sample_task_with_missing_params.task_id] == ExecutorTaskStatus.WAITING_HITL_PARAMS
            
            logger.log_summary({
                "hitl_triggered": True,
                "hitl_type": "missing_parameters",
                "missing_parameters": hitl_request.get('missing_parameters', [])
            })
        
        # 检查任务状态
        assert executor_input.task_status_map[sample_task_with_missing_params.task_id] == ExecutorTaskStatus.WAITING_HITL_PARAMS
        
        print(f"✓ HITL 参数请求已触发")
        print(f"  请求类型: {hitl_request['type']}")
        print(f"  缺失参数: {hitl_request.get('missing_parameters', [])}")
    
    def test_hitl_parameter_response_handling(self, executor_subgraph, sample_task_with_missing_params, request, test_case_logger):
        """测试 HITL 参数响应处理"""
        logger = test_case_logger
        
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_with_missing_params],
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        if logger:
            logger.log_initial_state(executor_input, "转换后的 ExecutorState")
            logger.log_task_order([
                {"task_id": task.task_id, "content": task.content, "dependencies": task.dependencies}
                for task in executor_input.subtasks
            ])
        
        # 初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input_before = executor_input
        executor_input = initialize_tasks_node(executor_input)
        if logger:
            logger.log_node_execution("initialize_tasks", executor_input_before, executor_input, "初始化任务")
        
        # 执行参数推断（触发 HITL 请求）
        from nodes.subagents.executor.graph import infer_parameters_node
        executor_input_before = executor_input
        executor_input = infer_parameters_node(executor_input)
        if logger:
            logger.log_node_execution("infer_parameters", executor_input_before, executor_input, "参数推断")
        
        # 记录参数推断结果和 HITL 请求
        task_result = executor_input.task_results.get(sample_task_with_missing_params.task_id)
        if logger:
            if task_result:
                logger.log_parameter_inference(
                    task_id=sample_task_with_missing_params.task_id,
                    parameters=task_result.parameters,
                    missing_parameters=task_result.missing_parameters,
                    llm_used=True
                )
            
            if sample_task_with_missing_params.task_id in executor_input.hitl_requests:
                logger.log_hitl_request(
                    task_id=sample_task_with_missing_params.task_id,
                    request_type=executor_input.hitl_requests[sample_task_with_missing_params.task_id]["type"],
                    request_data=executor_input.hitl_requests[sample_task_with_missing_params.task_id]
                )
        
        # 模拟用户响应
        user_response = {
            "type": "response_parameters",
            "responses": {
                sample_task_with_missing_params.task_id: {
                    "parameters": {
                        "sequence_file": "/path/to/sequences.fasta"
                    }
                }
            }
        }
        global_state.hitl_status = json.dumps(user_response, ensure_ascii=False)
        
        if logger:
            # 记录用户响应
            logger.log_hitl_response(
                task_id=sample_task_with_missing_params.task_id,
                response_type="response_parameters",
                response_data=user_response["responses"][sample_task_with_missing_params.task_id]
            )
        
        # 处理 HITL 响应
        from nodes.subagents.executor.graph import hitl_params_node
        executor_input_before = executor_input
        executor_input = hitl_params_node(executor_input)
        if logger:
            logger.log_node_execution("hitl_params", executor_input_before, executor_input, "处理 HITL 参数响应")
        
        # 验证响应已处理
        assert sample_task_with_missing_params.task_id in executor_input.hitl_responses
        
        # 验证参数已更新
        task_result = executor_input.task_results.get(sample_task_with_missing_params.task_id)
        assert task_result is not None
        assert "sequence_file" in task_result.parameters
        assert task_result.parameters["sequence_file"] == "/path/to/sequences.fasta"
        
        # 验证任务状态已更新为就绪
        assert executor_input.task_status_map[sample_task_with_missing_params.task_id] == ExecutorTaskStatus.READY
        
        if logger:
            logger.log_summary({
                "hitl_response_handled": True,
                "parameters_updated": True,
                "task_status": "READY"
            })
        
        print(f"✓ HITL 参数响应处理成功")
        print(f"  更新后的参数: {task_result.parameters}")
    
    def test_hitl_parameter_check_node(self, executor_subgraph, sample_task_with_missing_params):
        """测试 HITL 参数检查节点"""
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_with_missing_params],
            sandbox_dir="./sandbox"
        )
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        # 初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input = initialize_tasks_node(executor_input)
        
        # 执行参数推断（触发 HITL 请求）
        from nodes.subagents.executor.graph import infer_parameters_node
        executor_input = infer_parameters_node(executor_input)
        
        # 检查 HITL 状态
        from nodes.subagents.executor.graph import check_hitl_params_node
        route = check_hitl_params_node(executor_input)
        
        # 应该有 HITL 请求，应该路由到 hitl_params
        assert route == "hitl_params"
        
        print(f"✓ HITL 参数检查节点正确路由到 hitl_params")


class TestHITLResultConfirmation:
    """HITL 结果确认功能测试"""
    
    def test_hitl_result_confirmation_trigger(self, executor_subgraph, sample_task_without_tool):
        """测试触发 HITL 结果确认"""
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_without_tool],
            sandbox_dir="./sandbox"
        )
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        # 初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input = initialize_tasks_node(executor_input)
        
        # 模拟任务已完成，但结果需要确认
        task_result = TaskExecutionResult(
            task_id=sample_task_without_tool.task_id,
            status=ExecutorTaskStatus.COMPLETED,
            execution_mode="codeact",
            output={"result": "2", "status": "success"},
            result_satisfied=None  # 未分析
        )
        executor_input.task_results[sample_task_without_tool.task_id] = task_result
        executor_input.task_status_map[sample_task_without_tool.task_id] = ExecutorTaskStatus.COMPLETED
        
        # 执行结果分析（可能触发 HITL 确认）
        from nodes.subagents.executor.graph import analyze_results_node
        executor_input = analyze_results_node(executor_input)
        
        # 检查是否触发了 HITL 确认请求
        # 注意：这取决于 LLM 的分析结果，可能不会每次都触发
        # 但我们可以检查任务结果是否已分析
        task_result = executor_input.task_results.get(sample_task_without_tool.task_id)
        assert task_result is not None
        
        # 如果触发了 HITL，检查请求
        if sample_task_without_tool.task_id in executor_input.hitl_requests:
            hitl_request = executor_input.hitl_requests[sample_task_without_tool.task_id]
            assert hitl_request["type"] == "result_confirmation"
            assert executor_input.task_status_map[sample_task_without_tool.task_id] == ExecutorTaskStatus.WAITING_HITL_CONFIRM
            print(f"✓ HITL 结果确认请求已触发")
        else:
            print(f"✓ 结果分析完成，未触发 HITL（结果满足要求）")
    
    def test_hitl_result_confirmation_response(self, executor_subgraph, sample_task_without_tool):
        """测试 HITL 结果确认响应处理"""
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_without_tool],
            sandbox_dir="./sandbox"
        )
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        # 初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input = initialize_tasks_node(executor_input)
        
        # 模拟任务已完成并触发了 HITL 确认
        task_result = TaskExecutionResult(
            task_id=sample_task_without_tool.task_id,
            status=ExecutorTaskStatus.COMPLETED,
            execution_mode="codeact",
            output={"result": "2", "status": "success"},
            result_satisfied=False  # 不满足要求
        )
        executor_input.task_results[sample_task_without_tool.task_id] = task_result
        executor_input.task_status_map[sample_task_without_tool.task_id] = ExecutorTaskStatus.WAITING_HITL_CONFIRM
        
        executor_input.hitl_requests[sample_task_without_tool.task_id] = {
            "type": "result_confirmation",
            "task_id": sample_task_without_tool.task_id,
            "message": "任务结果需要确认"
        }
        
        # 模拟用户响应（继续执行）
        user_response = {
            "type": "response_confirmation",
            "responses": {
                sample_task_without_tool.task_id: {
                    "continue": True
                }
            }
        }
        global_state.hitl_status = json.dumps(user_response, ensure_ascii=False)
        
        # 处理 HITL 响应
        from nodes.subagents.executor.graph import hitl_confirm_node
        executor_input = hitl_confirm_node(executor_input)
        
        # 验证响应已处理
        assert sample_task_without_tool.task_id in executor_input.hitl_responses
        
        # 验证用户选择已更新
        task_result = executor_input.task_results.get(sample_task_without_tool.task_id)
        assert task_result is not None
        assert task_result.user_continue is True
        
        # 验证任务状态已更新为已完成
        assert executor_input.task_status_map[sample_task_without_tool.task_id] == ExecutorTaskStatus.COMPLETED
        
        print(f"✓ HITL 结果确认响应处理成功")
        print(f"  用户选择: {'继续' if task_result.user_continue else '停止'}")


class TestTaskExecutionFlow:
    """任务执行流程测试"""
    
    def test_task_initialization(self, executor_subgraph, sample_tasks_with_dependencies):
        """测试任务初始化（无依赖→就绪，有依赖→等待依赖）"""
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=sample_tasks_with_dependencies,
            sandbox_dir="./sandbox"
        )
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        
        # 执行初始化
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input = initialize_tasks_node(executor_input)
        
        # 验证任务状态
        assert executor_input.task_status_map["task_001"] == ExecutorTaskStatus.READY  # 无依赖
        assert executor_input.task_status_map["task_002"] == ExecutorTaskStatus.WAITING_DEPENDENCY  # 有依赖
        assert executor_input.task_status_map["task_003"] == ExecutorTaskStatus.WAITING_DEPENDENCY  # 有依赖
        
        print(f"✓ 任务初始化正确")
        print(f"  就绪任务: task_001")
        print(f"  等待依赖: task_002, task_003")
    
    def test_dependency_activation(self, executor_subgraph, sample_tasks_with_dependencies):
        """测试依赖任务激活"""
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=sample_tasks_with_dependencies,
            sandbox_dir="./sandbox"
        )
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        
        # 初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input = initialize_tasks_node(executor_input)
        
        # 模拟 task_001 已完成
        task1_result = TaskExecutionResult(
            task_id="task_001",
            status=ExecutorTaskStatus.COMPLETED,
            execution_mode="codeact"
        )
        executor_input.task_results["task_001"] = task1_result
        executor_input.task_status_map["task_001"] = ExecutorTaskStatus.COMPLETED
        
        # 激活依赖任务
        from nodes.subagents.executor.graph import activate_dependent_tasks_node
        executor_input = activate_dependent_tasks_node(executor_input)
        
        # 验证 task_002 已激活（task_001 已完成）
        assert executor_input.task_status_map["task_002"] == ExecutorTaskStatus.READY
        
        # task_003 应该还在等待（task_002 未完成）
        assert executor_input.task_status_map["task_003"] == ExecutorTaskStatus.WAITING_DEPENDENCY
        
        print(f"✓ 依赖任务激活正确")
        print(f"  task_002 已激活（依赖 task_001 已完成）")
        print(f"  task_003 仍在等待（依赖 task_002 未完成）")


class TestParallelExecution:
    """并行执行控制测试"""
    
    def test_parallel_execution_limit(self, executor_subgraph):
        """测试并行执行上限控制"""
        # 创建多个无依赖任务
        tasks = [
            SubTask(
                task_id=f"task_{i:03d}",
                task_type=UserTaskType.EXECUTE_PLAN,
                content=f"执行任务 {i}",
                result={"tools": [], "inputs": [], "outputs": []},
                dependencies=[]
            )
            for i in range(5)
        ]
        
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state, max_parallel_tasks=2)
        executor_input.max_parallel_tasks = 2  # 设置并行上限为2
        
        # 初始化任务
        from nodes.subagents.executor.graph import initialize_tasks_node
        executor_input = initialize_tasks_node(executor_input)
        
        # 执行任务节点
        from nodes.subagents.executor.graph import execute_tasks_node
        executor_input = execute_tasks_node(executor_input)
        
        # 验证并行执行数量不超过上限
        # 注意：由于任务执行可能很快完成，这里主要验证逻辑正确性
        assert len(executor_input.running_tasks) <= executor_input.max_parallel_tasks
        
        print(f"✓ 并行执行上限控制正确")
        print(f"  最大并行数: {executor_input.max_parallel_tasks}")
        print(f"  当前运行数: {len(executor_input.running_tasks)}")


class TestFullWorkflow:
    """完整工作流测试"""
    
    def test_full_executor_workflow(self, executor_subgraph, sample_task_without_tool, request, test_case_logger):
        """测试完整的 executor 工作流"""
        logger = test_case_logger
        
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_without_tool],
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        if logger:
            logger.log_initial_state(executor_input, "转换后的 ExecutorState")
            logger.log_task_order([
                {"task_id": task.task_id, "content": task.content, "dependencies": task.dependencies}
                for task in executor_input.subtasks
            ])
        
        # 执行完整子图（提供 configurable 参数以满足 checkpointer 要求）
        # 临时保存 parent_state，然后移除以避免 LangGraph 序列化验证错误
        saved_parent_state = executor_input.parent_state
        executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
        executor_output = executor_subgraph.invoke(
            executor_input_dict,
            config={"configurable": {"thread_id": "test_thread"}}
        )
        # 恢复 parent_state
        if saved_parent_state:
            _set_parent_state_safely(executor_input, saved_parent_state)
        result = _ensure_executor_state(executor_output)
        
        if logger:
            # 记录完整执行流程
            logger.log_node_execution("executor_subgraph", executor_input, result, "完整子图执行")
            
            # 记录所有任务执行结果
            for task_id, task_result in result.task_results.items():
                logger.log_task_execution(
                    task_id=task_id,
                    status=task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                    execution_mode=task_result.execution_mode,
                    result={"output": task_result.output, "confidence_score": task_result.confidence_score},
                    error=task_result.error
                )
                
                # 如果有生成的代码，记录 codeact 输出
                if task_result.code:
                    logger.log_codeact_output(
                        task_id=task_id,
                        generated_code=task_result.code,
                        execution_result={
                            "status": task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                            "output": task_result.output,
                            "error": task_result.error
                        }
                    )
            
            # 记录 HITL 交互
            for task_id, hitl_request in result.hitl_requests.items():
                logger.log_hitl_request(task_id, hitl_request.get("type", "unknown"), hitl_request)
            
            for task_id, hitl_response in result.hitl_responses.items():
                logger.log_hitl_response(task_id, "response", hitl_response)
            
            # 记录总结
            logger.log_summary({
                "total_tasks": result.total_tasks,
                "completed": result.completed_count,
                "failed": result.failed_count,
                "hitl_requests_count": len(result.hitl_requests),
                "hitl_responses_count": len(result.hitl_responses)
            })
        
        # 验证执行完成
        assert result.total_tasks == 1
        assert result.completed_count + result.failed_count == 1
        
        # 检查任务执行结果，验证是否有错误
        if result.task_results:
            for task_id, task_result in result.task_results.items():
                if task_result.status == ExecutorTaskStatus.FAILED:
                    print(f"  ⚠ 任务 {task_id} 执行失败: {task_result.error}")
                    # 如果任务失败，应该记录错误信息
                    assert task_result.error is not None, f"任务 {task_id} 失败但未记录错误信息"
                    # 打印详细的错误信息以便调试
                    if task_result.error_category:
                        print(f"    错误分类: {task_result.error_category.value}")
                    if task_result.failure_analysis:
                        print(f"    失败分析: {task_result.failure_analysis}")
                elif task_result.status == ExecutorTaskStatus.COMPLETED:
                    print(f"  ✓ 任务 {task_id} 执行成功")
                    # 验证成功任务有输出
                    assert task_result.output is not None or task_result.code is not None, \
                        f"任务 {task_id} 成功但无输出或代码"
        
        print(f"✓ 完整工作流执行完成")
        print(f"  总任务数: {result.total_tasks}")
        print(f"  已完成: {result.completed_count}")
        print(f"  失败: {result.failed_count}")


class TestInterruptMechanism:
    """Interrupt 机制测试"""
    
    def test_interrupt_trigger_for_missing_parameters(self, executor_subgraph, sample_task_with_missing_params, request, test_case_logger):
        """测试参数缺失时触发 interrupt"""
        logger = test_case_logger
        
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_with_missing_params],
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        if logger:
            logger.log_initial_state(executor_input, "转换后的 ExecutorState")
        
        # 使用 interrupt 支持的执行函数
        thread_id = f"test_interrupt_{sample_task_with_missing_params.task_id}"
        result = execute_executor_with_interrupt_support(
            executor_subgraph,
            executor_input,
            thread_id=thread_id
        )
        
        if logger:
            logger.log_node_execution("executor_subgraph", executor_input, result.get("result"), "执行（可能中断）")
            if result["interrupted"]:
                logger.log_hitl_request(
                    task_id=sample_task_with_missing_params.task_id,
                    request_type="missing_parameters",
                    request_data=result.get("interrupt_data", {})
                )
        
        # 验证是否触发了 interrupt
        # 注意：由于参数推断可能成功，不一定总是触发 interrupt
        if result["interrupted"]:
            assert result["interrupt_data"] is not None
            interrupt_data = result["interrupt_data"]
            assert isinstance(interrupt_data, dict)
            assert interrupt_data.get("type") == "missing_parameters"
            print(f"✓ Interrupt 已触发（参数缺失）")
            print(f"  中断数据: {interrupt_data}")
        else:
            print(f"✓ 执行完成，未触发 interrupt（参数推断成功）")
    
    def test_interrupt_resume_for_parameters(self, executor_subgraph, sample_task_with_missing_params, request, test_case_logger):
        """测试 interrupt 恢复（提供参数后继续执行）"""
        logger = test_case_logger
        
        global_state = GlobalState(
            user_input="执行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[sample_task_with_missing_params],
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        # 首次执行（可能触发 interrupt）
        thread_id = f"test_resume_{sample_task_with_missing_params.task_id}"
        result = execute_executor_with_interrupt_support(
            executor_subgraph,
            executor_input,
            thread_id=thread_id
        )
        
        if result["interrupted"]:
            if logger:
                logger.log_hitl_request(
                    task_id=sample_task_with_missing_params.task_id,
                    request_type="missing_parameters",
                    request_data=result.get("interrupt_data", {})
                )
            
            # 模拟用户提供参数
            resume_value = {
                "type": "response_parameters",
                "responses": {
                    sample_task_with_missing_params.task_id: {
                        "parameters": {
                            "sequence_file": "/path/to/test_sequences.fasta"
                        }
                    }
                }
            }
            
            if logger:
                logger.log_hitl_response(
                    task_id=sample_task_with_missing_params.task_id,
                    response_type="response_parameters",
                    response_data=resume_value["responses"][sample_task_with_missing_params.task_id]
                )
            
            # 恢复执行
            resume_result = resume_executor_after_interrupt(
                executor_subgraph,
                thread_id=thread_id,
                resume_value=resume_value
            )
            
            if logger:
                logger.log_node_execution("executor_subgraph", None, resume_result.get("result"), "恢复执行")
            
            # 验证恢复执行成功
            assert resume_result["result"] is not None
            final_state = resume_result["result"]
            
            # 验证参数已更新
            task_result = final_state.task_results.get(sample_task_with_missing_params.task_id)
            if task_result:
                assert "sequence_file" in task_result.parameters
                assert task_result.parameters["sequence_file"] == "/path/to/test_sequences.fasta"
            
            print(f"✓ Interrupt 恢复成功")
            print(f"  参数已更新: {task_result.parameters if task_result else 'N/A'}")
        else:
            print(f"✓ 首次执行完成，未触发 interrupt（跳过恢复测试）")


class TestComplexScenarios:
    """复杂场景测试"""
    
    def test_mixed_mcp_and_codeact_tasks(self, executor_subgraph, request, test_case_logger):
        """测试混合 MCP 工具和 CodeAct 任务"""
        logger = test_case_logger
        
        # 创建混合任务列表
        tasks = [
            SubTask(
                task_id="mcp_task_001",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="搜索COVID-19相关的抗体库数据",
                result={
                    "tools": [
                        {
                            "tool_name": "search_airr_repertoires",
                            "name": "search_airr_repertoires",
                            "service": "airr",
                            "description": "搜索AIRR数据库"
                        }
                    ],
                    "inputs": ["disease", "tissue"],
                    "outputs": ["repertoire_data"]
                },
                dependencies=[]
            ),
            SubTask(
                task_id="codeact_task_001",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="处理搜索结果，计算统计信息",
                result={
                    "tools": [],
                    "inputs": [],
                    "outputs": ["statistics"]
                },
                dependencies=["mcp_task_001"]
            ),
            SubTask(
                task_id="mcp_task_002",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="保存处理结果到数据库",
                result={
                    "tools": [
                        {
                            "tool_name": "save_data",
                            "name": "save_data",
                            "service": "storage",
                            "description": "保存数据"
                        }
                    ],
                    "inputs": ["data"],
                    "outputs": ["saved_id"]
                },
                dependencies=["codeact_task_001"]
            )
        ]
        
        global_state = GlobalState(
            user_input="执行混合任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState（混合任务）")
            logger.log_task_order([
                {"task_id": task.task_id, "content": task.content, "dependencies": task.dependencies}
                for task in tasks
            ])
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        # 执行完整流程
        thread_id = "test_mixed_tasks"
        executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
        result = executor_subgraph.invoke(
            executor_input_dict,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        final_state = _ensure_executor_state(result)
        
        if logger:
            logger.log_node_execution("executor_subgraph", executor_input, final_state, "混合任务执行")
            
            # 记录所有任务执行结果
            for task_id, task_result in final_state.task_results.items():
                logger.log_task_execution(
                    task_id=task_id,
                    status=task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                    execution_mode=task_result.execution_mode,
                    result={"output": task_result.output},
                    error=task_result.error
                )
            
            logger.log_summary({
                "total_tasks": final_state.total_tasks,
                "completed": final_state.completed_count,
                "failed": final_state.failed_count,
                "mcp_tasks": 2,
                "codeact_tasks": 1
            })
        
        # 验证任务执行
        assert final_state.total_tasks == 3
        print(f"✓ 混合任务执行完成")
        print(f"  总任务数: {final_state.total_tasks}")
        print(f"  已完成: {final_state.completed_count}")
        print(f"  失败: {final_state.failed_count}")
    
    def test_complex_dependency_chain(self, executor_subgraph, request, test_case_logger):
        """测试复杂的依赖链（多个层级）"""
        logger = test_case_logger
        
        # 创建复杂的依赖链：A -> B -> C, A -> D -> E, B -> E
        tasks = [
            SubTask(
                task_id="task_A",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="任务A（根任务）",
                result={"tools": [], "inputs": [], "outputs": ["data_A"]},
                dependencies=[]
            ),
            SubTask(
                task_id="task_B",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="任务B（依赖A）",
                result={"tools": [], "inputs": [], "outputs": ["data_B"]},
                dependencies=["task_A"]
            ),
            SubTask(
                task_id="task_C",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="任务C（依赖B）",
                result={"tools": [], "inputs": [], "outputs": ["data_C"]},
                dependencies=["task_B"]
            ),
            SubTask(
                task_id="task_D",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="任务D（依赖A）",
                result={"tools": [], "inputs": [], "outputs": ["data_D"]},
                dependencies=["task_A"]
            ),
            SubTask(
                task_id="task_E",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="任务E（依赖D和B）",
                result={"tools": [], "inputs": [], "outputs": ["data_E"]},
                dependencies=["task_D", "task_B"]
            )
        ]
        
        global_state = GlobalState(
            user_input="执行复杂依赖链",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState（复杂依赖链）")
            logger.log_task_order([
                {"task_id": task.task_id, "content": task.content, "dependencies": task.dependencies}
                for task in tasks
            ])
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        # 执行完整流程
        thread_id = "test_complex_dependencies"
        executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
        result = executor_subgraph.invoke(
            executor_input_dict,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        final_state = _ensure_executor_state(result)
        
        if logger:
            logger.log_node_execution("executor_subgraph", executor_input, final_state, "复杂依赖链执行")
            
            # 记录任务执行顺序
            execution_order = []
            for task_id in ["task_A", "task_B", "task_C", "task_D", "task_E"]:
                if task_id in final_state.task_results:
                    task_result = final_state.task_results[task_id]
                    execution_order.append({
                        "task_id": task_id,
                        "status": task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                        "execution_time": task_result.execution_time
                    })
            logger.log_task_order(execution_order)
            
            logger.log_summary({
                "total_tasks": final_state.total_tasks,
                "completed": final_state.completed_count,
                "failed": final_state.failed_count,
                "dependency_levels": 3
            })
        
        # 验证依赖关系正确执行
        assert final_state.total_tasks == 5
        
        # 验证 task_A 应该先执行（无依赖）
        if "task_A" in final_state.task_results:
            task_a_result = final_state.task_results["task_A"]
            # task_A 应该已完成或失败（不应该还在等待）
            assert task_a_result.status in [ExecutorTaskStatus.COMPLETED, ExecutorTaskStatus.FAILED]
        
        print(f"✓ 复杂依赖链执行完成")
        print(f"  总任务数: {final_state.total_tasks}")
        print(f"  已完成: {final_state.completed_count}")
        print(f"  失败: {final_state.failed_count}")
    
    def test_parallel_execution_with_dependencies(self, executor_subgraph, request, test_case_logger):
        """测试并行执行控制（有依赖关系的任务）"""
        logger = test_case_logger
        
        # 创建多个独立任务（可以并行执行）
        # 以及一些有依赖的任务
        tasks = [
            SubTask(
                task_id=f"independent_task_{i}",
                task_type=UserTaskType.EXECUTE_PLAN,
                content=f"独立任务 {i}",
                result={"tools": [], "inputs": [], "outputs": [f"data_{i}"]},
                dependencies=[]
            )
            for i in range(5)
        ]
        
        # 添加一个有依赖的任务
        tasks.append(
            SubTask(
                task_id="dependent_task",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="依赖任务（依赖前3个独立任务）",
                result={"tools": [], "inputs": [], "outputs": ["final_data"]},
                dependencies=["independent_task_0", "independent_task_1", "independent_task_2"]
            )
        )
        
        global_state = GlobalState(
            user_input="执行并行任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState（并行执行）")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state, max_parallel_tasks=2)
        executor_input.max_parallel_tasks = 2  # 限制并行数为2
        _set_parent_state_safely(executor_input, global_state)
        
        # 执行完整流程
        thread_id = "test_parallel_execution"
        executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
        result = executor_subgraph.invoke(
            executor_input_dict,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        final_state = _ensure_executor_state(result)
        
        if logger:
            logger.log_node_execution("executor_subgraph", executor_input, final_state, "并行执行")
            logger.log_summary({
                "total_tasks": final_state.total_tasks,
                "completed": final_state.completed_count,
                "failed": final_state.failed_count,
                "max_parallel_tasks": executor_input.max_parallel_tasks
            })
        
        # 验证并行执行控制
        assert final_state.total_tasks == 6
        assert executor_input.max_parallel_tasks == 2
        
        print(f"✓ 并行执行控制测试完成")
        print(f"  总任务数: {final_state.total_tasks}")
        print(f"  最大并行数: {executor_input.max_parallel_tasks}")
        print(f"  已完成: {final_state.completed_count}")
    
    def test_multiple_hitl_requests(self, executor_subgraph, request, test_case_logger):
        """测试多个 HITL 请求（多个任务需要参数）"""
        logger = test_case_logger
        
        # 创建多个需要参数的任务
        tasks = [
            SubTask(
                task_id=f"task_needs_params_{i}",
                task_type=UserTaskType.EXECUTE_PLAN,
                content=f"任务 {i}，需要参数",
                result={
                    "tools": [
                        {
                            "tool_name": f"tool_{i}",
                            "name": f"tool_{i}",
                            "service": "test",
                            "description": f"测试工具 {i}"
                        }
                    ],
                    "inputs": [f"param_{i}"],
                    "outputs": [f"result_{i}"]
                },
                dependencies=[]
            )
            for i in range(3)
        ]
        
        global_state = GlobalState(
            user_input="执行多个需要参数的任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState（多个HITL请求）")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state)
        _set_parent_state_safely(executor_input, global_state)
        
        # 执行到参数推断节点
        from nodes.subagents.executor.graph import initialize_tasks_node, infer_parameters_node
        executor_input = initialize_tasks_node(executor_input)
        executor_input = infer_parameters_node(executor_input)
        
        if logger:
            logger.log_node_execution("infer_parameters", None, executor_input, "参数推断（多个任务）")
            
            # 记录所有 HITL 请求
            for task_id, hitl_request in executor_input.hitl_requests.items():
                logger.log_hitl_request(
                    task_id=task_id,
                    request_type=hitl_request.get("type", "unknown"),
                    request_data=hitl_request
                )
        
        # 验证多个 HITL 请求
        hitl_count = len(executor_input.hitl_requests)
        assert hitl_count > 0, "应该至少有一个 HITL 请求"
        
        print(f"✓ 多个 HITL 请求测试完成")
        print(f"  HITL 请求数量: {hitl_count}")
        for task_id in executor_input.hitl_requests.keys():
            print(f"    任务 {task_id} 需要参数")
    
    def test_task_failure_and_retry(self, executor_subgraph, request, test_case_logger):
        """测试任务失败和重试机制"""
        logger = test_case_logger
        
        # 创建一个会失败的任务（使用不存在的工具）
        task = SubTask(
            task_id="failing_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行会失败的任务",
            result={
                "tools": [
                    {
                        "tool_name": "non_existent_tool",
                        "name": "non_existent_tool",
                        "service": "test",
                        "description": "不存在的工具"
                    }
                ],
                "inputs": [],
                "outputs": ["result"]
            },
            dependencies=[]
        )
        
        global_state = GlobalState(
            user_input="执行会失败的任务",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[task],
            sandbox_dir="./sandbox"
        )
        
        if logger:
            logger.log_initial_state(global_state, "初始 GlobalState（失败任务）")
        
        try:
            executor_input = executor_input_mapper(global_state)
        except Exception as e:
            executor_input = _create_executor_state_for_test(global_state, max_retries=2)
        executor_input.max_retries = 2
        _set_parent_state_safely(executor_input, global_state)
        
        # 执行完整流程
        thread_id = "test_task_failure"
        executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
        result = executor_subgraph.invoke(
            executor_input_dict,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        final_state = _ensure_executor_state(result)
        
        if logger:
            logger.log_node_execution("executor_subgraph", executor_input, final_state, "失败任务执行")
            
            # 记录失败任务的结果
            if "failing_task" in final_state.task_results:
                task_result = final_state.task_results["failing_task"]
                logger.log_task_execution(
                    task_id="failing_task",
                    status=task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                    execution_mode=task_result.execution_mode,
                    result={"output": task_result.output},
                    error=task_result.error
                )
            
            logger.log_summary({
                "total_tasks": final_state.total_tasks,
                "completed": final_state.completed_count,
                "failed": final_state.failed_count,
                "max_retries": executor_input.max_retries
            })
        
        # 验证任务失败处理
        assert final_state.total_tasks == 1
        
        if "failing_task" in final_state.task_results:
            task_result = final_state.task_results["failing_task"]
            # 任务应该失败
            if task_result.status == ExecutorTaskStatus.FAILED:
                assert task_result.error is not None
                print(f"✓ 任务失败处理正确")
                print(f"  错误信息: {task_result.error[:100] if task_result.error else 'N/A'}")
                if task_result.error_category:
                    print(f"  错误分类: {task_result.error_category.value}")
            else:
                print(f"✓ 任务执行完成（状态: {task_result.status})")

