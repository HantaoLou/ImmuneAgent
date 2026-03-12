# -*- coding: utf-8 -*-
"""
Iterative Executor 子图构建

构建 iterative_executor 子图，提供与 main_graph 的集成接口。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state import GlobalState

from langgraph.graph import StateGraph, START, END

from nodes.subagents.iterative_executor.state import IterativeExecutorState
from nodes.subagents.iterative_executor.node import (
    iterative_executor_node,
    iterative_executor_node_async,
)
from nodes.subagents.iterative_executor.input_mapper import (
    iterative_executor_input_mapper,
)
from nodes.subagents.iterative_executor.output_mapper import (
    iterative_executor_output_mapper,
)


# ============================================================================
# 子图节点函数
# ============================================================================


def _input_adapter_node(state: "GlobalState") -> IterativeExecutorState:
    """
    输入适配节点

    将 GlobalState 转换为 IterativeExecutorState。

    Args:
        state: 全局状态

    Returns:
        IterativeExecutorState: 子图状态
    """
    return iterative_executor_input_mapper(state)


def _execution_node(state: IterativeExecutorState) -> IterativeExecutorState:
    """
    执行节点

    执行迭代任务的核心逻辑。

    Args:
        state: 子图状态

    Returns:
        IterativeExecutorState: 更新后的子图状态
    """
    # 这里可以拆分为多个子节点：
    # 1. prepare_environment - 准备环境
    # 2. generate_tasks - 生成任务
    # 3. execute_iteration - 执行迭代
    # 4. evaluate - 评估结果
    # 5. optimize - 优化任务

    # 目前使用简化的单节点实现
    return state


def _output_adapter_node(state: IterativeExecutorState) -> "GlobalState":
    """
    输出适配节点

    将 IterativeExecutorState 转换回 GlobalState。
    注意：这个节点需要访问原始的 GlobalState，所以实际使用时会通过其他方式处理。

    Args:
        state: 子图状态

    Returns:
        GlobalState: 全局状态（需要通过其他方式获取）
    """
    # 这个节点实际上不单独使用，因为需要原始的 GlobalState
    # 实际的输出映射在 main_graph 层面完成
    return state


# ============================================================================
# 子图构建函数
# ============================================================================


def build_iterative_executor_subgraph():
    """
    构建 iterative_executor 子图

    子图结构:
    START → execute → END

    注意：iterative_executor 是一个相对简单的子图，
    大部分复杂逻辑在 IterativeOpenCodeExecutor 内部处理。

    Returns:
        CompiledGraph: 编译后的子图
    """
    graph = StateGraph(IterativeExecutorState)

    # 添加节点
    # 目前使用单节点模式，后续可以拆分为多个节点
    graph.add_node("execute", _execute_internal)

    # 添加边
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)

    return graph.compile()


def _execute_internal(state: IterativeExecutorState) -> IterativeExecutorState:
    """
    内部执行函数

    这个函数封装了 IterativeOpenCodeExecutor 的调用逻辑。

    Args:
        state: 子图状态

    Returns:
        IterativeExecutorState: 更新后的状态
    """
    import asyncio
    from datetime import datetime
    from nodes.subagents.iterative_executor.node import (
        ITERATIVE_EXECUTOR_AVAILABLE,
        _create_opencode_config,
        _create_evaluation_criteria,
        _update_state_from_result,
    )
    from nodes.subagents.iterative_executor.task_generator import (
        generate_tasks_md,
        get_required_output_files,
    )
    from nodes.subagents.iterative_executor.input_mapper import prepare_executor_input

    print("[IterativeExecutor Subgraph] 开始执行...")

    # 设置开始时间
    state.start_time = datetime.now().isoformat()

    if not ITERATIVE_EXECUTOR_AVAILABLE:
        print("[IterativeExecutor Subgraph] IterativeOpenCodeExecutor 不可用")
        state.iteration_status = "failed"
        state.errors.append("IterativeOpenCodeExecutor not available")
        state.end_time = datetime.now().isoformat()
        return state

    try:
        from coding_agent.iterative_executor import (
            IterativeOpenCodeExecutor,
            EvaluationCriteria,
        )
        from coding_agent.iterative_executor import IterationStatus as ResultStatus

        # 创建配置
        # 需要从 GlobalState 获取配置信息，这里简化处理
        sandbox_domain = os.getenv("OPENSANDBOX_DOMAIN", "http://117.10.59.114:40001")
        config = OpenCodeConfig(
            model_provider=os.getenv("OPENCODE_MODEL_PROVIDER", "glm-4.7"),
            sandbox_domain=sandbox_domain,
            api_key="",
        )

        # 创建评估标准
        evaluation_criteria = EvaluationCriteria(
            required_output_files=get_required_output_files(state),
            min_quality_score=0.6,
            early_stop_on_success=state.early_stop_on_success,
        )

        # 创建执行器
        executor = IterativeOpenCodeExecutor(
            config=config,
            max_iterations=state.max_iterations,
            evaluation_criteria=evaluation_criteria,
        )

        # 准备输入数据
        input_data = prepare_executor_input(state)

        # 执行
        result = asyncio.run(executor.execute(input_data))

        # 更新状态
        state = _update_state_from_result(state, result)

    except Exception as e:
        import traceback

        print(f"[IterativeExecutor Subgraph] 执行失败: {e}")
        traceback.print_exc()
        state.iteration_status = "failed"
        state.errors.append(str(e))

    # 设置结束时间
    state.end_time = datetime.now().isoformat()

    return state


# 为简化导入的 OpenCodeConfig
class OpenCodeConfig:
    """简化的 OpenCodeConfig（避免循环导入）"""

    def __init__(self, model_provider, sandbox_domain, api_key):
        self.model_provider = model_provider
        self.sandbox_domain = sandbox_domain
        self.api_key = api_key


# ============================================================================
# Main Graph 集成接口
# ============================================================================


def iterative_executor_subgraph_node(state: "GlobalState") -> "GlobalState":
    """
    供 main_graph 使用的子图节点函数

    这个函数封装了完整的子图调用流程：
    1. 输入映射 (GlobalState → IterativeExecutorState)
    2. 子图执行
    3. 输出映射 (IterativeExecutorState → GlobalState)

    Args:
        state: 全局状态

    Returns:
        GlobalState: 更新后的全局状态
    """
    # 直接调用 node.py 中的节点函数
    return iterative_executor_node(state)


# ============================================================================
# 延迟加载的子图实例
# ============================================================================

_iterative_executor_subgraph = None


def get_iterative_executor_subgraph():
    """
    获取 iterative_executor 子图实例（单例模式）

    Returns:
        CompiledGraph: 编译后的子图
    """
    global _iterative_executor_subgraph
    if _iterative_executor_subgraph is None:
        print("[IterativeExecutor] 编译子图...")
        _iterative_executor_subgraph = build_iterative_executor_subgraph()
        print("[IterativeExecutor] 子图编译完成")
    return _iterative_executor_subgraph
