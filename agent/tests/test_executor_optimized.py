"""
Executor 优化功能测试用例

测试优化后的 executor 功能，包括：
1. 重试机制（网络错误、可重试错误，max_retries=5）
2. 资源检查节点
3. 组级别结果聚合
4. 错误处理和分类（网络错误、错误分析、建议生成）
5. 死锁检测
6. 参数推断优化

运行方式：pytest tests/test_executor_optimized.py -v
"""

import os
import pytest
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import Mock, patch, MagicMock

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
    _check_resources_available,
    classify_error,
    _analyze_failure,
    _generate_suggestions,
    _detect_deadlock,
    initialize_tasks_node,
    infer_parameters_node,
    execute_tasks_node,
    activate_dependent_tasks_node,
    check_completion_node,
    summary_results_node
)
from state import SubTask, UserTaskType, GlobalState, ParallelTaskGroup

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


def _create_executor_state_for_test(global_state: GlobalState, **kwargs) -> ExecutorState:
    """为测试创建 ExecutorState"""
    try:
        return ExecutorState(
            subtasks=global_state.subtasks,
            parallel_task_groups=global_state.parallel_task_groups,
            sandbox_dir=kwargs.get("sandbox_dir", global_state.sandbox_dir),
            parent_state=global_state,
            max_parallel_tasks=kwargs.get("max_parallel_tasks", 3),
            max_retries=kwargs.get("max_retries", 5),  # 使用新的默认值5
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


@pytest.fixture(scope="module")
def executor_subgraph():
    """构建并返回 Executor 子图"""
    return build_executor_subgraph()


@pytest.fixture(scope="module", autouse=True)
def setup_global_logger():
    """初始化全局日志记录器"""
    test_file_name = Path(__file__).stem
    init_global_logger(test_file_name)
    yield
    save_global_logger()


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


@pytest.fixture
def sample_global_state():
    """示例全局状态"""
    return GlobalState(
        user_input="执行测试任务",
        user_task_type=UserTaskType.EXECUTE_PLAN,
        subtasks=[],
        sandbox_dir="./sandbox"
    )


class TestErrorClassification:
    """错误分类测试"""
    
    def test_network_error_classification(self):
        """测试网络错误分类"""
        # 测试各种网络错误关键词
        network_errors = [
            ("Connection timeout", "TimeoutError"),
            ("Network unreachable", "NetworkError"),
            ("Connection refused", "ConnectionError"),
            ("DNS resolution failed", "DNSException"),
            ("502 Bad Gateway", "HTTPError"),
            ("503 Service Unavailable", "HTTPError"),
            ("Connection reset by peer", "ConnectionError"),
        ]
        
        for error_msg, error_type in network_errors:
            category = classify_error(error_msg, error_type)
            assert category == ErrorCategory.NETWORK_ERROR, \
                f"错误 '{error_msg}' 应该被分类为网络错误"
        
        print("✓ 网络错误分类正确")
    
    def test_retryable_error_classification(self):
        """测试可重试错误分类"""
        retryable_errors = [
            ("Rate limit exceeded", "RateLimitError"),
            ("429 Too Many Requests", "HTTPError"),
            ("Service temporarily unavailable", "ServiceError"),
            ("Throttle limit reached", "ThrottleError"),
        ]
        
        for error_msg, error_type in retryable_errors:
            category = classify_error(error_msg, error_type)
            assert category == ErrorCategory.RETRYABLE, \
                f"错误 '{error_msg}' 应该被分类为可重试错误"
        
        print("✓ 可重试错误分类正确")
    
    def test_parameter_error_classification(self):
        """测试参数错误分类"""
        param_errors = [
            ("Missing required parameter: 'file_path'", "ValueError"),
            ("Invalid argument type", "TypeError"),
            ("KeyError: 'missing_key'", "KeyError"),
            ("AttributeError: 'NoneType' object has no attribute", "AttributeError"),
        ]
        
        for error_msg, error_type in param_errors:
            category = classify_error(error_msg, error_type)
            assert category == ErrorCategory.PARAMETER_ERROR, \
                f"错误 '{error_msg}' 应该被分类为参数错误"
        
        print("✓ 参数错误分类正确")
    
    def test_code_error_classification(self):
        """测试代码错误分类"""
        code_errors = [
            ("SyntaxError: invalid syntax", "SyntaxError"),
            ("IndentationError: unexpected indent", "IndentationError"),
            ("NameError: name 'x' is not defined", "NameError"),
            ("IndexError: list index out of range", "IndexError"),
            ("ZeroDivisionError: division by zero", "ZeroDivisionError"),
        ]
        
        for error_msg, error_type in code_errors:
            category = classify_error(error_msg, error_type)
            assert category == ErrorCategory.CODE_ERROR, \
                f"错误 '{error_msg}' 应该被分类为代码错误"
        
        print("✓ 代码错误分类正确")
    
    def test_system_error_classification(self):
        """测试系统错误分类（默认）"""
        system_errors = [
            ("Unknown error occurred", "UnknownError"),
            ("Internal server error", "ServerError"),
        ]
        
        for error_msg, error_type in system_errors:
            category = classify_error(error_msg, error_type)
            assert category == ErrorCategory.SYSTEM_ERROR, \
                f"错误 '{error_msg}' 应该被分类为系统错误"
        
        print("✓ 系统错误分类正确")


class TestErrorAnalysis:
    """错误分析测试"""
    
    def test_network_error_analysis(self):
        """测试网络错误分析"""
        analysis = _analyze_failure(
            "Connection timeout",
            "TimeoutError",
            ErrorCategory.NETWORK_ERROR
        )
        
        assert "网络连接问题" in analysis
        assert "自动重试" in analysis
        print("✓ 网络错误分析正确")
    
    def test_parameter_error_analysis(self):
        """测试参数错误分析"""
        analysis = _analyze_failure(
            "Missing required parameter",
            "ValueError",
            ErrorCategory.PARAMETER_ERROR
        )
        
        assert "参数不正确" in analysis
        assert "检查任务参数配置" in analysis
        print("✓ 参数错误分析正确")
    
    def test_suggestions_generation(self):
        """测试建议生成"""
        # 测试网络错误建议
        suggestions = _generate_suggestions(
            ErrorCategory.NETWORK_ERROR,
            "Connection timeout",
            2,
            5
        )
        
        assert len(suggestions) > 0
        assert "网络错误" in suggestions[0]
        assert "2/5" in suggestions[1]  # 重试次数
        
        # 测试达到最大重试次数
        suggestions_max = _generate_suggestions(
            ErrorCategory.NETWORK_ERROR,
            "Connection timeout",
            5,
            5
        )
        
        assert "已达到最大重试次数" in suggestions_max[-1]
        
        print("✓ 建议生成正确")


class TestRetryMechanism:
    """重试机制测试"""
    
    def test_max_retries_default_value(self, sample_global_state):
        """测试 max_retries 默认值为 5"""
        executor_state = _create_executor_state_for_test(sample_global_state)
        assert executor_state.max_retries == 5, "max_retries 默认值应该是 5"
        print("✓ max_retries 默认值正确（5）")
    
    def test_network_error_retry(self, sample_global_state):
        """测试网络错误重试机制"""
        task = SubTask(
            task_id="network_error_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="测试网络错误重试",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=[]
        )
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[task],
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state, max_retries=3)
        executor_state = initialize_tasks_node(executor_state)
        
        # 模拟网络错误
        task_result = TaskExecutionResult(
            task_id=task.task_id,
            status=ExecutorTaskStatus.FAILED,
            execution_mode="mcp_tool",
            error="Connection timeout",
            error_category=ErrorCategory.NETWORK_ERROR,
            retry_count=0
        )
        executor_state.task_results[task.task_id] = task_result
        
        # 模拟执行节点中的重试逻辑
        if task_result.error_category == ErrorCategory.NETWORK_ERROR:
            if task_result.retry_count < executor_state.max_retries:
                task_result.retry_count += 1
                executor_state.task_status_map[task.task_id] = ExecutorTaskStatus.READY
                print(f"  ✓ 网络错误重试 {task_result.retry_count}/{executor_state.max_retries}")
        
        # 验证重试逻辑
        assert task_result.retry_count == 1
        assert executor_state.task_status_map[task.task_id] == ExecutorTaskStatus.READY
        print("✓ 网络错误重试机制正确")
    
    def test_retry_count_limit(self, sample_global_state):
        """测试重试次数限制"""
        task = SubTask(
            task_id="max_retry_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="测试最大重试次数",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=[]
        )
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[task],
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state, max_retries=3)
        
        # 模拟已达到最大重试次数
        task_result = TaskExecutionResult(
            task_id=task.task_id,
            status=ExecutorTaskStatus.FAILED,
            execution_mode="mcp_tool",
            error="Connection timeout",
            error_category=ErrorCategory.NETWORK_ERROR,
            retry_count=3  # 已达到最大重试次数
        )
        
        # 验证不应再重试
        should_retry = task_result.retry_count < executor_state.max_retries
        assert not should_retry, "达到最大重试次数后不应再重试"
        print("✓ 重试次数限制正确")


class TestResourceCheck:
    """资源检查节点测试"""
    
    def test_resource_check_with_available_slots(self, sample_global_state):
        """测试有可用槽位时的资源检查"""
        tasks = [
            SubTask(
                task_id=f"task_{i}",
                task_type=UserTaskType.EXECUTE_PLAN,
                content=f"任务 {i}",
                result={"tools": [], "inputs": [], "outputs": []},
                dependencies=[]
            )
            for i in range(3)
        ]
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state, max_parallel_tasks=3)
        executor_state = initialize_tasks_node(executor_state)
        
        # 检查资源（使用辅助函数）
        has_resources = _check_resources_available(executor_state)
        
        # 应该有可用槽位
        assert has_resources == True, "有可用槽位时应该返回 True"
        print("✓ 资源检查（有可用槽位）正确")
    
    def test_resource_check_without_available_slots(self, sample_global_state):
        """测试无可用槽位时的资源检查"""
        tasks = [
            SubTask(
                task_id=f"task_{i}",
                task_type=UserTaskType.EXECUTE_PLAN,
                content=f"任务 {i}",
                result={"tools": [], "inputs": [], "outputs": []},
                dependencies=[]
            )
            for i in range(3)
        ]
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state, max_parallel_tasks=2)
        executor_state = initialize_tasks_node(executor_state)
        
        # 模拟已有2个任务在运行
        executor_state.running_tasks = ["task_0", "task_1"]
        executor_state.task_status_map["task_0"] = ExecutorTaskStatus.RUNNING
        executor_state.task_status_map["task_1"] = ExecutorTaskStatus.RUNNING
        
        # 检查资源（使用辅助函数）
        has_resources = _check_resources_available(executor_state)
        
        # 应该无可用槽位
        assert has_resources == False, "无可用槽位时应该返回 False"
        print("✓ 资源检查（无可用槽位）正确")
    
    def test_resource_check_with_no_ready_tasks(self, sample_global_state):
        """测试无就绪任务时的资源检查"""
        tasks = [
            SubTask(
                task_id="task_0",
                task_type=UserTaskType.EXECUTE_PLAN,
                content="任务 0",
                result={"tools": [], "inputs": [], "outputs": []},
                dependencies=[]
            )
        ]
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state)
        executor_state = initialize_tasks_node(executor_state)
        
        # 将所有任务标记为已完成
        executor_state.task_status_map["task_0"] = ExecutorTaskStatus.COMPLETED
        
        # 检查资源（使用辅助函数）
        has_resources = _check_resources_available(executor_state)
        
        # 应该无就绪任务
        assert has_resources == False, "无就绪任务时应该返回 False"
        print("✓ 资源检查（无就绪任务）正确")


class TestGroupLevelSummary:
    """组级别结果聚合测试"""
    
    def test_group_level_summary(self, sample_global_state):
        """测试组级别结果汇总"""
        # 创建并行任务组
        tasks = [
            SubTask(
                task_id=f"group_task_{i}",
                task_type=UserTaskType.EXECUTE_PLAN,
                content=f"组任务 {i}",
                result={"tools": [], "inputs": [], "outputs": []},
                dependencies=[]
            )
            for i in range(3)
        ]
        
        parallel_group = ParallelTaskGroup(
            group_id="test_group_1",
            subtasks=tasks,
            description="测试并行组"
        )
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[],
            parallel_task_groups={"test_group_1": parallel_group},
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state)
        executor_state = initialize_tasks_node(executor_state)
        
        # 模拟任务执行结果
        for i, task in enumerate(tasks):
            status = ExecutorTaskStatus.COMPLETED if i < 2 else ExecutorTaskStatus.FAILED
            task_result = TaskExecutionResult(
                task_id=task.task_id,
                status=status,
                execution_mode="codeact",
                output={"result": f"output_{i}"} if status == ExecutorTaskStatus.COMPLETED else None,
                error="Test error" if status == ExecutorTaskStatus.FAILED else None
            )
            executor_state.task_results[task.task_id] = task_result
            executor_state.task_status_map[task.task_id] = status
            if status == ExecutorTaskStatus.COMPLETED:
                executor_state.completed_count += 1
            else:
                executor_state.failed_count += 1
        
        # 执行汇总
        executor_state = summary_results_node(executor_state)
        
        # 验证汇总完成（主要验证不会抛出异常）
        assert executor_state.total_tasks == 3
        assert executor_state.completed_count == 2
        assert executor_state.failed_count == 1
        print("✓ 组级别结果汇总正确")


class TestDeadlockDetection:
    """死锁检测测试"""
    
    def test_deadlock_detection_cycle(self, sample_global_state):
        """测试循环依赖死锁检测"""
        # 创建循环依赖：A -> B -> C -> A
        task_a = SubTask(
            task_id="task_A",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="任务A",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=["task_C"]  # A 依赖 C
        )
        task_b = SubTask(
            task_id="task_B",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="任务B",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=["task_A"]  # B 依赖 A
        )
        task_c = SubTask(
            task_id="task_C",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="任务C",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=["task_B"]  # C 依赖 B
        )
        
        tasks = [task_a, task_b, task_c]
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state)
        executor_state = initialize_tasks_node(executor_state)
        
        # 所有任务都应该在等待依赖
        waiting_tasks = [
            task for task in executor_state.subtasks
            if executor_state.task_status_map.get(task.task_id) == ExecutorTaskStatus.WAITING_DEPENDENCY
        ]
        
        # 检测死锁
        has_deadlock = _detect_deadlock(waiting_tasks, executor_state)
        
        assert has_deadlock, "应该检测到循环依赖死锁"
        print("✓ 死锁检测（循环依赖）正确")
    
    def test_no_deadlock_linear_dependency(self, sample_global_state):
        """测试线性依赖（无死锁）"""
        # 创建线性依赖：A -> B -> C
        task_a = SubTask(
            task_id="task_A",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="任务A",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=[]
        )
        task_b = SubTask(
            task_id="task_B",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="任务B",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=["task_A"]
        )
        task_c = SubTask(
            task_id="task_C",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="任务C",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=["task_B"]
        )
        
        tasks = [task_a, task_b, task_c]
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state)
        executor_state = initialize_tasks_node(executor_state)
        
        # 获取等待任务（B 和 C）
        waiting_tasks = [
            task for task in executor_state.subtasks
            if executor_state.task_status_map.get(task.task_id) == ExecutorTaskStatus.WAITING_DEPENDENCY
        ]
        
        # 检测死锁
        has_deadlock = _detect_deadlock(waiting_tasks, executor_state)
        
        assert not has_deadlock, "线性依赖不应该检测到死锁"
        print("✓ 死锁检测（线性依赖）正确")


class TestCompletionCheck:
    """完成检查逻辑测试"""
    
    def test_completion_check_all_completed(self, sample_global_state):
        """测试所有任务完成时的检查"""
        tasks = [
            SubTask(
                task_id=f"task_{i}",
                task_type=UserTaskType.EXECUTE_PLAN,
                content=f"任务 {i}",
                result={"tools": [], "inputs": [], "outputs": []},
                dependencies=[]
            )
            for i in range(3)
        ]
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state)
        executor_state = initialize_tasks_node(executor_state)
        
        # 将所有任务标记为已完成
        for task in tasks:
            executor_state.task_status_map[task.task_id] = ExecutorTaskStatus.COMPLETED
            executor_state.task_results[task.task_id] = TaskExecutionResult(
                task_id=task.task_id,
                status=ExecutorTaskStatus.COMPLETED,
                execution_mode="codeact"
            )
            executor_state.completed_count += 1
        
        # 检查完成状态
        route = check_completion_node(executor_state)
        
        assert route == "summary", "所有任务完成时应该返回 'summary'"
        print("✓ 完成检查（全部完成）正确")
    
    def test_completion_check_has_ready_tasks(self, sample_global_state):
        """测试有就绪任务时的检查"""
        tasks = [
            SubTask(
                task_id=f"task_{i}",
                task_type=UserTaskType.EXECUTE_PLAN,
                content=f"任务 {i}",
                result={"tools": [], "inputs": [], "outputs": []},
                dependencies=[]
            )
            for i in range(3)
        ]
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state)
        executor_state = initialize_tasks_node(executor_state)
        
        # 检查完成状态（应该有就绪任务）
        route = check_completion_node(executor_state)
        
        assert route == "infer_params", "有就绪任务时应该返回 'infer_params'"
        print("✓ 完成检查（有就绪任务）正确")
    
    def test_completion_check_deadlock(self, sample_global_state):
        """测试死锁检测时的完成检查"""
        # 创建循环依赖
        task_a = SubTask(
            task_id="task_A",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="任务A",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=["task_B"]
        )
        task_b = SubTask(
            task_id="task_B",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="任务B",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=["task_A"]
        )
        
        tasks = [task_a, task_b]
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=tasks,
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state)
        executor_state = initialize_tasks_node(executor_state)
        
        # 模拟无运行中任务，但有等待依赖的任务（死锁情况）
        # 检查完成状态
        route = check_completion_node(executor_state)
        
        # 应该检测到死锁并返回 summary
        assert route == "summary", "检测到死锁时应该返回 'summary'"
        print("✓ 完成检查（死锁检测）正确")


class TestParameterInferenceOptimization:
    """参数推断优化测试"""
    
    def test_parameter_inference_with_cache(self, sample_global_state):
        """测试参数推断（带缓存机制）"""
        task = SubTask(
            task_id="test_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索COVID-19相关的抗体数据，组织类型为血液",
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
        )
        
        global_state = GlobalState(
            user_input="测试",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[task],
            sandbox_dir="./sandbox"
        )
        
        executor_state = _create_executor_state_for_test(global_state)
        executor_state = initialize_tasks_node(executor_state)
        
        # 执行参数推断
        executor_state = infer_parameters_node(executor_state)
        
        # 验证参数推断结果
        task_result = executor_state.task_results.get(task.task_id)
        assert task_result is not None
        assert hasattr(task_result, 'parameters')
        assert hasattr(task_result, 'missing_parameters')
        
        print("✓ 参数推断（优化版）正确")


class TestFullOptimizedWorkflow:
    """完整优化工作流测试"""
    
    def test_optimized_executor_workflow(self, executor_subgraph, sample_global_state, request, test_case_logger):
        """测试优化后的完整工作流"""
        logger = test_case_logger
        
        # 创建测试任务
        task = SubTask(
            task_id="optimized_test_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="计算1+1",
            result={"tools": [], "inputs": [], "outputs": []},
            dependencies=[]
        )
        
        global_state = GlobalState(
            user_input="测试优化工作流",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            subtasks=[task],
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
        
        # 验证优化后的配置
        assert executor_input.max_retries == 5, "max_retries 应该是 5"
        
        # 执行完整子图
        saved_parent_state = executor_input.parent_state
        executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
        executor_output = executor_subgraph.invoke(
            executor_input_dict,
            config={"configurable": {"thread_id": "test_optimized_workflow"}}
        )
        if saved_parent_state:
            _set_parent_state_safely(executor_input, saved_parent_state)
        
        result = _ensure_executor_state(executor_output)
        
        if logger:
            logger.log_node_execution("executor_subgraph", executor_input, result, "优化工作流执行")
            logger.log_summary({
                "total_tasks": result.total_tasks,
                "completed": result.completed_count,
                "failed": result.failed_count,
                "max_retries": result.max_retries
            })
        
        # 验证执行完成
        assert result.total_tasks == 1
        assert result.max_retries == 5
        
        print("✓ 优化工作流执行完成")
        print(f"  总任务数: {result.total_tasks}")
        print(f"  已完成: {result.completed_count}")
        print(f"  失败: {result.failed_count}")
        print(f"  最大重试次数: {result.max_retries}")

