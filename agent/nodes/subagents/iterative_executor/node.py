# -*- coding: utf-8 -*-
"""
Iterative Executor 节点实现

实现 iterative_executor_node，使用 IterativeOpenCodeExecutor 执行任务。
"""

from typing import TYPE_CHECKING, Dict, Any, Optional
from datetime import datetime
import os
import asyncio

if TYPE_CHECKING:
    from state import GlobalState

from nodes.subagents.iterative_executor.state import (
    IterativeExecutorState,
    IterationStatus,
    EvaluationLevel,
)
from nodes.subagents.iterative_executor.input_mapper import (
    iterative_executor_input_mapper,
    prepare_executor_input,
)
from nodes.subagents.iterative_executor.output_mapper import (
    iterative_executor_output_mapper,
)
from nodes.subagents.iterative_executor.task_generator import (
    generate_tasks_md,
    get_required_output_files,
)


# ============================================================================
# IterativeOpenCodeExecutor 可用性检查
# ============================================================================

ITERATIVE_EXECUTOR_AVAILABLE = False
IterativeOpenCodeExecutor = None
OpenCodeConfig = None
EvaluationCriteria = None

try:
    from coding_agent.iterative_executor import (
        IterativeOpenCodeExecutor,
        EvaluationCriteria,
    )
    from coding_agent.config import OpenCodeConfig
    ITERATIVE_EXECUTOR_AVAILABLE = True
    print("[IterativeExecutor] IterativeOpenCodeExecutor 可用")
except ImportError as e:
    print(f"[IterativeExecutor] 警告: 无法导入 IterativeOpenCodeExecutor: {e}")


# ============================================================================
# LLM 可用性检查
# ============================================================================

LLM_AVAILABLE = False
create_reasoning_llm = None

try:
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    print("[IterativeExecutor] 警告: LLM 不可用")


# ============================================================================
# 核心节点实现
# ============================================================================

def iterative_executor_node(state: "GlobalState") -> "GlobalState":
    """
    迭代执行节点 - 使用 IterativeOpenCodeExecutor
    
    这个节点是 task_decomposition + executor 的合并版本。
    
    📦 沙盒职责：
    - 读取/写入文件到: /data/sessions/{session_id}/
    - 输入文件: input/ 目录
    - 输出文件: output/ 目录
    - 任务文件: .agent/tasks.md
    
    执行流程:
    1. 映射 GlobalState → IterativeExecutorState
    2. 准备 IterativeOpenCodeExecutor 配置
    3. 执行迭代任务
    4. 映射结果回 GlobalState
    
    Args:
        state: 全局状态
        
    Returns:
        GlobalState: 更新后的全局状态
    """
    print("=" * 60)
    print("Iterative Executor 节点启动")
    print("=" * 60)
    
    # 检查 IterativeOpenCodeExecutor 是否可用
    if not ITERATIVE_EXECUTOR_AVAILABLE:
        print("  [回退] IterativeOpenCodeExecutor 不可用")
        return _iterative_executor_fallback(state)
    
    try:
        # 1. 映射全局状态到子图状态
        print("  [1/4] 映射全局状态到子图状态...")
        executor_state = iterative_executor_input_mapper(state)
        
        # 设置开始时间
        executor_state.start_time = datetime.now().isoformat()
        
        print(f"        Session ID: {executor_state.session_id}")
        print(f"        MCP 服务: {executor_state.mcp_services}")
        
        # 2. 准备配置
        print("  [2/4] 准备 IterativeOpenCodeExecutor 配置...")
        config = _create_opencode_config(state)
        evaluation_criteria = _create_evaluation_criteria(executor_state)
        
        # 3. 执行迭代任务
        print("  [3/4] 执行 IterativeOpenCodeExecutor...")
        print(f"        最大迭代次数: {executor_state.max_iterations}")
        print(f"        早停: {executor_state.early_stop_on_success}")
        
        # 使用同步方式调用异步执行器
        result = asyncio.run(_execute_iterative(
            config=config,
            evaluation_criteria=evaluation_criteria,
            executor_state=executor_state,
            state=state,
        ))
        
        # 4. 映射结果回全局状态
        print("  [4/4] 映射结果到全局状态...")
        
        # 更新 executor_state 的输出信息
        if result:
            executor_state = _update_state_from_result(executor_state, result)
        
        # 映射到 GlobalState
        state = iterative_executor_output_mapper(executor_state, state)
        
        # 打印摘要
        print("=" * 60)
        print("Iterative Executor 节点完成")
        print(f"  - 最终状态: {executor_state.iteration_status}")
        print(f"  - 迭代次数: {executor_state.current_iteration}")
        print(f"  - 质量分数: {executor_state.quality_score:.2f}")
        print(f"  - 输出文件: {len(executor_state.output_files)} 个")
        print("=" * 60)
        
        return state
        
    except Exception as e:
        import traceback
        print(f"  [错误] 节点执行失败: {e}")
        traceback.print_exc()
        print("  [回退] 使用回退模式")
        return _iterative_executor_fallback(state)


async def _execute_iterative(
    config,
    evaluation_criteria,
    executor_state: IterativeExecutorState,
    state: "GlobalState",
):
    """
    执行 IterativeOpenCodeExecutor
    
    Args:
        config: OpenCode 配置
        evaluation_criteria: 评估标准
        executor_state: 执行器状态
        state: 全局状态
        
    Returns:
        IterativeExecutionResult: 执行结果
    """
    # 创建执行器
    executor = IterativeOpenCodeExecutor(
        config=config,
        max_iterations=executor_state.max_iterations,
        evaluation_criteria=evaluation_criteria,
    )
    
    # 准备输入数据
    input_data = prepare_executor_input(executor_state)
    
    # 执行
    result = await executor.execute(input_data)
    
    return result


def _update_state_from_result(
    executor_state: IterativeExecutorState,
    result,
) -> IterativeExecutorState:
    """
    从 IterativeExecutionResult 更新 IterativeExecutorState
    
    Args:
        executor_state: 执行器状态
        result: IterativeExecutionResult
        
    Returns:
        IterativeExecutorState: 更新后的状态
    """
    from coding_agent.iterative_executor import IterationStatus as ResultStatus
    
    # 更新迭代状态
    if hasattr(result, 'final_status'):
        status_map = {
            ResultStatus.SUCCESS: IterationStatus.SUCCESS,
            ResultStatus.NEEDS_IMPROVEMENT: IterationStatus.NEEDS_IMPROVEMENT,
            ResultStatus.FAILED: IterationStatus.FAILED,
        }
        executor_state.iteration_status = status_map.get(
            result.final_status, 
            IterationStatus.NEEDS_IMPROVEMENT
        )
    
    # 更新迭代次数
    if hasattr(result, 'total_iterations'):
        executor_state.current_iteration = result.total_iterations
    
    # 更新输出文件
    if hasattr(result, 'final_output_files'):
        executor_state.output_files = result.final_output_files
    
    # 更新 MCP 调用记录
    if hasattr(result, 'all_mcp_calls'):
        executor_state.mcp_calls = [
            _convert_mcp_call(call) 
            for call in result.all_mcp_calls
        ]
    
    # 更新迭代历史
    if hasattr(result, 'iteration_history'):
        executor_state.iteration_history = [
            _convert_iteration_result(iter_result)
            for iter_result in result.iteration_history
        ]
    
    # 更新报告路径
    if hasattr(result, 'report_path'):
        executor_state.report_path = result.report_path
    
    # 更新质量分数和评估等级
    if hasattr(result, 'evaluation_result'):
        eval_result = result.evaluation_result
        if isinstance(eval_result, dict):
            executor_state.quality_score = eval_result.get('overall_score', 0.0) / 100.0
            executor_state.evaluation_details = eval_result
    
    # 更新结束时间
    executor_state.end_time = datetime.now().isoformat()
    
    # 根据质量分数设置评估等级
    executor_state.evaluation_level = _score_to_level(executor_state.quality_score)
    
    return executor_state


def _convert_mcp_call(call) -> Dict[str, Any]:
    """转换 MCP 调用记录为字典格式"""
    if hasattr(call, 'to_dict'):
        return call.to_dict()
    elif isinstance(call, dict):
        return call
    else:
        return {
            "tool_name": getattr(call, 'tool_name', 'unknown'),
            "service_name": getattr(call, 'service_name', ''),
            "parameters": getattr(call, 'parameters', {}),
            "success": getattr(call, 'success', False),
            "error": getattr(call, 'error', ''),
        }


def _convert_iteration_result(iter_result) -> Dict[str, Any]:
    """转换迭代结果为字典格式"""
    if isinstance(iter_result, dict):
        return iter_result
    else:
        return {
            "iteration": getattr(iter_result, 'iteration', 0),
            "status": str(getattr(iter_result, 'status', 'unknown')),
            "evaluation_score": getattr(iter_result, 'evaluation_score', 0.0),
            "output_files": getattr(iter_result, 'output_files', []),
        }


def _score_to_level(score: float) -> EvaluationLevel:
    """将分数转换为评估等级"""
    if score >= 0.9:
        return EvaluationLevel.EXCELLENT
    elif score >= 0.7:
        return EvaluationLevel.GOOD
    elif score >= 0.6:
        return EvaluationLevel.ACCEPTABLE
    elif score >= 0.4:
        return EvaluationLevel.POOR
    else:
        return EvaluationLevel.FAILED


def _create_opencode_config(state: "GlobalState") -> "OpenCodeConfig":
    """
    创建 OpenCode 配置
    
    Args:
        state: 全局状态
        
    Returns:
        OpenCodeConfig: 配置对象
    """
    # 获取配置参数
    sandbox_domain = os.getenv("OPENSANDBOX_DOMAIN", "https://opensandbox.cn")
    api_key = os.getenv("ZHIPU_API_KEY", "")
    
    # MCP 配置路径
    mcp_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "config",
        "mcp_servers.json"
    )
    
    config = OpenCodeConfig(
        model_provider="glm-4-flash",
        sandbox_domain=sandbox_domain,
        api_key=api_key,
        mcp_config_path=mcp_config_path if os.path.exists(mcp_config_path) else None,
    )
    
    return config


def _create_evaluation_criteria(executor_state: IterativeExecutorState) -> "EvaluationCriteria":
    """
    创建评估标准
    
    Args:
        executor_state: 执行器状态
        
    Returns:
        EvaluationCriteria: 评估标准
    """
    # 获取必需的输出文件
    required_files = get_required_output_files(executor_state)
    
    criteria = EvaluationCriteria(
        required_output_files=required_files,
        min_quality_score=0.6,
        early_stop_on_success=executor_state.early_stop_on_success,
    )
    
    return criteria


# ============================================================================
# 回退实现
# ============================================================================

def _iterative_executor_fallback(state: "GlobalState") -> "GlobalState":
    """
    Iterative Executor 节点回退实现
    
    当 IterativeOpenCodeExecutor 不可用时，使用原有的 task_decomposition + executor 流程。
    
    Args:
        state: 全局状态
        
    Returns:
        GlobalState: 更新后的全局状态
    """
    print("=" * 60)
    print("Iterative Executor 节点 (回退模式)")
    print("=" * 60)
    
    # 确保 merged_result 存在
    if not state.merged_result:
        state.merged_result = {}
    
    state.merged_result["iterative_executor"] = {
        "status": "fallback",
        "message": "IterativeOpenCodeExecutor 不可用，使用回退模式",
        "total_iterations": 0,
        "output_files": [],
        "mcp_calls_count": 0,
        "quality_score": 0.0,
        "errors": ["IterativeOpenCodeExecutor not available"],
    }
    
    # 尝试调用 task_decomposition + executor
    try:
        # 导入回退处理器
        from main_graph import (
            task_decomposition_node,
            executor_node,
        )
        
        # 执行 task_decomposition
        print("  [回退] 调用 task_decomposition_node...")
        state = task_decomposition_node(state)
        
        # 执行 executor
        print("  [回退] 调用 executor_node...")
        state = executor_node(state)
        
        state.merged_result["iterative_executor"]["fallback_executed"] = True
        
    except Exception as e:
        print(f"  [回退错误] 回退执行失败: {e}")
        state.merged_result["iterative_executor"]["fallback_error"] = str(e)
    
    print("=" * 60)
    return state


# ============================================================================
# 异步节点包装器
# ============================================================================

async def iterative_executor_node_async(state: "GlobalState") -> "GlobalState":
    """
    异步版本的 iterative_executor_node
    
    Args:
        state: 全局状态
        
    Returns:
        GlobalState: 更新后的全局状态
    """
    # 由于 iterative_executor_node 内部使用 asyncio.run，
    # 这里直接调用同步版本
    return iterative_executor_node(state)

