# agent/main_graph.py
"""
Bio-Agent 主图定义

================================================================================
核心架构原则：所有任务在沙盒中执行
================================================================================

1. 沙盒执行原则
   - 所有代码执行都在远程沙盒 (OpenSandbox) 中进行
   - 所有生成的文件都保存在沙盒目录: /data/sessions/{session_id}/
   - 只有 CodeAct 子图直接与 OpenSandbox 沟通
   - 其他子图通过 utils/codeact_executor.py 统一接口执行代码

2. 沙盒目录结构
   /data/sessions/{session_id}/
   ├── input/              # 用户上传的输入文件
   ├── output/             # 工具执行产生的输出文件
   │   ├── reports/        # 各类报告 (Markdown, TXT)
   │   └── *.csv, *.json   # 工具输出数据
   ├── reports/            # 分析报告 (兼容旧结构)
   ├── todo-list.md        # 任务列表 (CodeAct Todo 模式读取)
   └── workspace/          # 工作空间

3. 路径转换规则
   - 服务器路径 (MCP工具用): /data/sessions/{session_id}/...
   - 容器路径 (代码执行用): /tmp/sessions/{session_id}/...
   - 转换函数: utils/sandbox_paths.get_server_path() / get_container_path()

4. 子图职责
   - supervisor: 预处理输入，生成 session_id，初始化沙盒目录
   - immunity: 生成实验计划，保存报告到沙盒
   - task_decomposition: 分解任务，生成 SubTask 列表
   - executor (CodeAct Todo 模式): 
     * 将 SubTask 转换为 todo-list.md
     * 循环读取和执行任务
     * 更新任务状态
   - result_evaluator: 收集结果，生成最终报告

================================================================================
流程图
================================================================================

START → supervisor → [路由]
       ├── immunity → task_decomposition → executor (CodeAct Todo) → result_evaluator → END
       ├── task_decomposition → executor (CodeAct Todo) → result_evaluator → END
       └── general_qa → END

================================================================================
Executor 节点 (CodeAct Todo 模式)
================================================================================

executor_node 执行流程:
1. 生成 todo-list.md（从 state.subtasks 转换）
2. 调用 CodeAct 子图 (todo 模式):
   - read_todo: 读取 todo-list.md
   - select_next_task: 选择下一个待执行任务
   - infer_parameters: 推断任务参数
   - explore_data: 数据探索
   - generate_code: 生成代码
   - execute_code: 执行代码
   - extract_file_params: 提取输出文件
   - validate_output: 验证输出
   - update_todo: 更新任务状态
   - [循环直到所有任务完成]
3. 映射结果回 GlobalState

================================================================================
"""

from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
import sys
import os
import json
import re
import uuid
from datetime import datetime

# 添加 agent 目录到路径
agent_dir = Path(__file__).parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType, SubTask, ensure_global_state_rebuilt


# =============================================================================
# LLM 相关导入
# =============================================================================
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None
    HumanMessage = None
    SystemMessage = None
    print("Warning: langchain-related libraries not installed, will use keyword matching as fallback")


# =============================================================================
# Supervisor 子图导入（重构版本）
# =============================================================================
try:
    from nodes.subagents.supervisor.graph_refactored import (
        build_supervisor_subgraph,
        supervisor_input_mapper,
        supervisor_output_mapper,
        SupervisorState,
    )
    SUPERVISOR_SUBGRAPH_AVAILABLE = True
    print("[Main Graph] 使用重构后的 supervisor 子图 (graph_refactored.py)")
except ImportError as e:
    SUPERVISOR_SUBGRAPH_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入重构后的 supervisor 子图: {e}")
    print("[Main Graph] 将使用简化版 supervisor 节点")


# =============================================================================
# Immunity 子图导入
# =============================================================================
try:
    from nodes.subagents.immunity import (
        build_immunity_subgraph,
        immunity_input_mapper,
        immunity_output_mapper,
        ImmunityState,
    )
    IMMUNITY_SUBGRAPH_AVAILABLE = True
    print("[Main Graph] 使用 immunity 子图")
except ImportError as e:
    IMMUNITY_SUBGRAPH_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 immunity 子图: {e}")
    print("[Main Graph] 将使用简化版 immunity 节点")


# =============================================================================
# Task Decomposition 子图导入
# =============================================================================
try:
    from nodes.subagents.task_decomposition.graph import (
        build_task_decomposition_subgraph,
        task_decomposition_input_mapper,
        task_decomposition_output_mapper,
        TaskDecompositionState,
    )
    TASK_DECOMPOSITION_SUBGRAPH_AVAILABLE = True
    print("[Main Graph] 使用 task_decomposition 子图")
except ImportError as e:
    TASK_DECOMPOSITION_SUBGRAPH_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 task_decomposition 子图: {e}")
    print("[Main Graph] 将使用简化版 task_decomposition 节点")


# =============================================================================
# CodeAct 子图导入（替代 Executor，使用 todo-list 驱动模式）
# =============================================================================
try:
    from nodes.subagents.code_act import (
        build_codeact_subgraph,
        CodeActState,
        CodeActExecutionMode,
        TodoTask,
        TodoTaskStatus,
        TodoTaskType,
        TodoList,
        TodoListManager,
    )
    CODEACT_SUBGRAPH_AVAILABLE = True
    print("[Main Graph] 使用 codeact 子图 (todo-list 驱动模式)")
except ImportError as e:
    CODEACT_SUBGRAPH_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 codeact 子图: {e}")
    print("[Main Graph] 将使用简化版 executor 节点")


# =============================================================================
# TodoList Generator 导入（用于将 SubTask 转换为 todo-list.md）
# =============================================================================
try:
    from nodes.subagents.executor.todolist_generator import (
        convert_subtasks_to_todolist,
        generate_and_save_todolist_from_state,
    )
    TODOLIST_GENERATOR_AVAILABLE = True
except ImportError as e:
    TODOLIST_GENERATOR_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 todolist_generator: {e}")


# =============================================================================
# Result Evaluator 子图导入
# =============================================================================
try:
    from nodes.subagents.result_evaluator import (
        build_result_evaluator_subgraph,
        result_evaluator_input_mapper,
        result_evaluator_output_mapper,
        ResultEvaluatorState,
    )
    RESULT_EVALUATOR_SUBGRAPH_AVAILABLE = True
    print("[Main Graph] 使用 result_evaluator 子图")
except ImportError as e:
    RESULT_EVALUATOR_SUBGRAPH_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 result_evaluator 子图: {e}")
    print("[Main Graph] 将使用简化版 result_evaluator 节点")


# =============================================================================
# Iterative Executor 子图导入（新增 - 替代 task_decomposition + executor）
# =============================================================================
try:
    from nodes.subagents.iterative_executor import (
        iterative_executor_node,
        iterative_executor_input_mapper,
        iterative_executor_output_mapper,
        IterativeExecutorState,
        ITERATIVE_EXECUTOR_AVAILABLE,
    )
    ITERATIVE_EXECUTOR_SUBGRAPH_AVAILABLE = True
    print(f"[Main Graph] 使用 iterative_executor 子图 (可用性: {ITERATIVE_EXECUTOR_AVAILABLE})")
except ImportError as e:
    ITERATIVE_EXECUTOR_SUBGRAPH_AVAILABLE = False
    ITERATIVE_EXECUTOR_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 iterative_executor 子图: {e}")
    print("[Main Graph] 将使用 task_decomposition + executor 流程")


# =============================================================================
# Mem0 记忆管理导入
# =============================================================================
try:
    from utils.mem0_manager import (
        save_immunity_trace_sync,
        check_all_tasks_completed_successfully,
        get_memory_client,
    )
    MEM0_MANAGER_AVAILABLE = True
    print("[Main Graph] Mem0 记忆管理可用")
except ImportError as e:
    MEM0_MANAGER_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 mem0_manager: {e}")


# =============================================================================
# Supervisor 节点 - 回退实现（当子图不可用时）
# =============================================================================

def _generate_session_id() -> str:
    """生成唯一会话 ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


def _get_llm():
    """获取 LLM 实例"""
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        return None
    return create_reasoning_llm(temperature=0.1)


def _classify_task_type_fallback(user_input: str) -> UserTaskType:
    """任务分类回退函数（关键词匹配）"""
    user_input_lower = user_input.lower()
    
    if any(keyword in user_input_lower for keyword in [
        "execute", "plan", "step", "follow", "according to",
        "执行", "计划", "步骤", "按照"
    ]):
        return UserTaskType.EXECUTE_PLAN
    
    if any(keyword in user_input_lower for keyword in [
        "immun", "antigen", "antibody", "vaccine", "immune",
        "免疫", "抗原", "抗体", "疫苗"
    ]):
        return UserTaskType.IMMUNOLOGY_TASK
    
    return UserTaskType.GENERAL_QA


def _supervisor_node_fallback(state: GlobalState) -> GlobalState:
    """
    Supervisor 节点回退实现（当子图不可用时使用）
    
    简化版本：仅进行基本的任务分类
    """
    print("=" * 60)
    print("Supervisor 节点 (回退模式)")
    print("=" * 60)
    
    user_input = state.user_input
    print(f"  用户输入: {user_input[:100]}...")
    
    # 生成 session_id
    session_id = state.session_id or _generate_session_id()
    state.session_id = session_id
    print(f"  Session ID: {session_id}")
    
    # 任务分类
    task_type = _classify_task_type_fallback(user_input)
    state.user_task_type = task_type
    print(f"  任务类型: {task_type.value}")
    
    print("=" * 60)
    return state


# =============================================================================
# Supervisor 节点 - 主实现（使用子图）
# =============================================================================

# 预编译 supervisor 子图（延迟加载）
_supervisor_subgraph = None


def _get_supervisor_subgraph():
    """获取 supervisor 子图实例（单例模式）"""
    global _supervisor_subgraph
    if _supervisor_subgraph is None and SUPERVISOR_SUBGRAPH_AVAILABLE:
        print("[Main Graph] 编译 supervisor 子图...")
        _supervisor_subgraph = build_supervisor_subgraph()
        print("[Main Graph] supervisor 子图编译完成")
    return _supervisor_subgraph


def supervisor_node(state: GlobalState) -> GlobalState:
    """
    Supervisor 节点 - 使用重构后的子图
    
    📦 沙盒职责：
    - 生成唯一的 session_id
    - 初始化沙盒目录结构: /data/sessions/{session_id}/
    - 设置 sandbox_data_dir 和 opensandbox_id
    
    执行步骤（由子图完成）:
    1. preprocess - 轻量预处理（生成 session ID，LLM 提取）
    2. detect_files - 检测文件（分类 LOCAL/REMOTE/URL）
    3. upload_files - 上传文件到沙盒 input/ 目录（调用 CodeAct）[条件执行]
    4. analyze_files - 分析文件内容（调用 CodeAct）[条件执行]
    5. build_params - 构建参数表
    6. classify - 任务分类
    
    所有文件操作通过 CodeAct 在沙盒中执行。
    """
    print("=" * 60)
    print("Supervisor 节点启动")
    print("=" * 60)
    
    # 检查子图是否可用
    if not SUPERVISOR_SUBGRAPH_AVAILABLE:
        print("  [回退] 使用简化版 supervisor")
        return _supervisor_node_fallback(state)
    
    # 获取子图
    subgraph = _get_supervisor_subgraph()
    if subgraph is None:
        print("  [错误] 子图不可用，使用回退模式")
        return _supervisor_node_fallback(state)
    
    try:
        # 1. 映射全局状态到子图状态
        print("  [1/3] 映射全局状态到子图状态...")
        supervisor_state = supervisor_input_mapper(state)
        
        # 2. 执行子图
        print("  [2/3] 执行 supervisor 子图...")
        print("        流程: preprocess → detect_files → upload_files → analyze_files → build_params → classify")
        
        result_state = subgraph.invoke(supervisor_state)
        
        # 调试：打印 result_state 类型
        print(f"  [DEBUG] result_state type: {type(result_state).__name__}")
        if isinstance(result_state, dict):
            print(f"  [DEBUG] result_state keys: {list(result_state.keys())[:10]}")
        elif isinstance(result_state, list):
            print(f"  [DEBUG] result_state is list, len={len(result_state)}")
            if len(result_state) > 0:
                print(f"  [DEBUG] first element type: {type(result_state[0]).__name__}")
        
        # 3. 映射子图结果回全局状态
        print("  [3/3] 映射子图结果到全局状态...")
        state = supervisor_output_mapper(result_state, state)
        
        print("=" * 60)
        print(f"Supervisor 节点完成 → 路由到: {state.user_task_type.value if state.user_task_type else 'unknown'}")
        print("=" * 60)
        
        return state
        
    except Exception as e:
        import traceback
        print(f"  [错误] 子图执行失败: {e}")
        print(f"  [错误] 异常类型: {type(e).__name__}")
        print(f"  [错误] 完整堆栈:")
        traceback.print_exc()
        print("  [回退] 使用简化版 supervisor")
        return _supervisor_node_fallback(state)


# =============================================================================
# 路由函数
# =============================================================================

def supervisor_router(state: GlobalState) -> str:
    """
    Supervisor 后的路由
    
    根据任务类型决定下一个节点:
    - IMMUNOLOGY_TASK → immunity
    - EXECUTE_PLAN → task_decomposition  
    - GENERAL_QA → general_qa
    """
    task_type = state.user_task_type
    
    if task_type == UserTaskType.IMMUNOLOGY_TASK:
        return "immunity"
    elif task_type == UserTaskType.EXECUTE_PLAN:
        return "task_decomposition"
    else:
        return "general_qa"


# =============================================================================
# Immunity 节点 - 回退实现（当子图不可用时）
# =============================================================================

def _immunity_node_fallback(state: GlobalState) -> GlobalState:
    """
    Immunity 节点回退实现（当子图不可用时使用）
    
    简化版本：仅记录信息，不执行实际分析
    """
    print("=" * 60)
    print("Immunity 节点 (回退模式)")
    print("=" * 60)
    
    print(f"  用户输入: {state.user_input[:100]}...")
    print(f"  Session ID: {state.session_id}")
    
    # 生成简单的实验计划
    simple_plan = f"""
# 实验计划（简化版）

## 研究问题
{state.user_input}

## 说明
Immunity 子图不可用，无法生成完整的实验计划。
请检查 immunity 子图的依赖是否正确安装。
"""
    
    state.execution_plan = simple_plan
    if not state.merged_result:
        state.merged_result = {}
    state.merged_result["immunity_plan"] = {
        "original_question": state.user_input,
        "experimental_plan": simple_plan,
        "fallback": True
    }
    
    print("=" * 60)
    return state


# =============================================================================
# Immunity 节点 - 主实现（使用子图）
# =============================================================================

# 预编译 immunity 子图（延迟加载）
_immunity_subgraph = None


def _get_immunity_subgraph():
    """获取 immunity 子图实例（单例模式）"""
    global _immunity_subgraph
    if _immunity_subgraph is None and IMMUNITY_SUBGRAPH_AVAILABLE:
        print("[Main Graph] 编译 immunity 子图...")
        _immunity_subgraph = build_immunity_subgraph()
        print("[Main Graph] immunity 子图编译完成")
    return _immunity_subgraph


def immunity_node(state: GlobalState) -> GlobalState:
    """
    Immunity 节点 - 使用子图
    
    📦 沙盒职责：
    - 所有报告保存到: /data/sessions/{session_id}/output/reports/
    - 生成文件: retrieval_*.md, deep_research_*.md, hypothesis_*.md, planning_*.md, evaluation_*.md
    
    完整工作流（由子图完成）:
    1. query_decomposition - 查询分解
    2. retrieval - 信息检索（Qdrant + Tavily + Web）
    3. deep_research - 深度研究（使用 deep_research 子图）
    4. hypothesis_generation - 假设生成
    5. planning - 实验计划生成 ⭐
    6. evaluation - 计划评估
    
    通过 CodeAct 统一接口保存报告到沙盒。
    """
    print("=" * 60)
    print("Immunity 节点启动")
    print("=" * 60)
    
    # 检查子图是否可用
    if not IMMUNITY_SUBGRAPH_AVAILABLE:
        print("  [回退] 使用简化版 immunity")
        return _immunity_node_fallback(state)
    
    # 获取子图
    subgraph = _get_immunity_subgraph()
    if subgraph is None:
        print("  [错误] 子图不可用，使用回退模式")
        return _immunity_node_fallback(state)
    
    try:
        # 1. 映射全局状态到子图状态
        print("  [1/3] 映射全局状态到子图状态...")
        immunity_state = immunity_input_mapper(state)
        
        # 2. 执行子图
        print("  [2/3] 执行 immunity 子图...")
        print("        流程: query_decomposition → retrieval → deep_research → hypothesis_generation → planning → evaluation")
        
        result_state = subgraph.invoke(immunity_state)
        
        # 3. 映射子图结果回全局状态
        print("  [3/3] 映射子图结果到全局状态...")
        state = immunity_output_mapper(result_state, state)
        
        print("=" * 60)
        print("Immunity 节点完成")
        print(f"  - 执行计划长度: {len(state.execution_plan or '')} 字符")
        print("=" * 60)
        
        return state
        
    except Exception as e:
        print(f"  [错误] 子图执行失败: {e}")
        import traceback
        traceback.print_exc()
        print("  [回退] 使用简化版 immunity")
        return _immunity_node_fallback(state)


# =============================================================================
# Task Decomposition 节点 - 回退实现（当子图不可用时）
# =============================================================================

def _task_decomposition_node_fallback(state: GlobalState) -> GlobalState:
    """
    Task Decomposition 节点回退实现（当子图不可用时使用）
    
    简化版本：仅创建一个简单的任务
    """
    print("=" * 60)
    print("Task Decomposition 节点 (回退模式)")
    print("=" * 60)
    
    print(f"  用户输入: {state.user_input[:100]}...")
    print(f"  Session ID: {state.session_id}")
    
    # 创建简单的子任务
    simple_subtask = SubTask(
        task_id="task_1",
        task_type=UserTaskType.EXECUTE_PLAN,
        content=state.user_input,
        dependencies=[],
        parallel_group_id=None
    )
    
    state.subtasks = [simple_subtask]
    if not state.merged_result:
        state.merged_result = {}
    state.merged_result["task_decomposition"] = {
        "fallback": True,
        "message": "Task decomposition subgraph unavailable, using simplified task structure"
    }
    
    print(f"  创建了 1 个简化任务")
    print("=" * 60)
    return state


# =============================================================================
# Task Decomposition 节点 - 主实现（使用子图）
# =============================================================================

# 预编译 task_decomposition 子图（延迟加载）
_task_decomposition_subgraph = None


def _get_task_decomposition_subgraph():
    """获取 task_decomposition 子图实例（单例模式）"""
    global _task_decomposition_subgraph
    if _task_decomposition_subgraph is None and TASK_DECOMPOSITION_SUBGRAPH_AVAILABLE:
        print("[Main Graph] 编译 task_decomposition 子图...")
        _task_decomposition_subgraph = build_task_decomposition_subgraph()
        print("[Main Graph] task_decomposition 子图编译完成")
    return _task_decomposition_subgraph


def task_decomposition_node(state: GlobalState) -> GlobalState:
    """
    Task Decomposition 节点 - 使用子图
    
    📦 沙盒职责：
    - 生成 todo-list.md 保存到: /data/sessions/{session_id}/todo-list.md
    - 任务分解结果保存在 state.subtasks 和 state.parallel_task_groups
    
    执行步骤（由子图完成）:
    1. coarse_decompose - 粗粒度分解（确定所需服务类型）
    2. fine_decompose - 细粒度分解（详细任务分解和工具匹配）
    3. infer_parallel - 并行任务推断（识别可并行执行的任务组）
    
    todo-list.md 通过 CodeAct 保存到沙盒，供 executor 读取和更新。
    """
    print("=" * 60)
    print("Task Decomposition 节点启动")
    print("=" * 60)
    
    # 检查子图是否可用
    if not TASK_DECOMPOSITION_SUBGRAPH_AVAILABLE:
        print("  [回退] 使用简化版 task_decomposition")
        return _task_decomposition_node_fallback(state)
    
    # 获取子图
    subgraph = _get_task_decomposition_subgraph()
    if subgraph is None:
        print("  [错误] 子图不可用，使用回退模式")
        return _task_decomposition_node_fallback(state)
    
    try:
        # 1. 映射全局状态到子图状态
        print("  [1/3] 映射全局状态到子图状态...")
        decomposition_state = task_decomposition_input_mapper(state)
        
        # 2. 执行子图
        print("  [2/3] 执行 task_decomposition 子图...")
        print("        流程: coarse_decompose → fine_decompose → infer_parallel")
        
        result_state = subgraph.invoke(decomposition_state)
        
        # 3. 映射子图结果回全局状态
        print("  [3/3] 映射子图结果到全局状态...")
        state = task_decomposition_output_mapper(result_state, state)
        
        # 打印分解结果摘要
        num_subtasks = len(state.subtasks) if state.subtasks else 0
        num_parallel_groups = len(state.parallel_task_groups) if state.parallel_task_groups else 0
        print(f"  - 子任务数量: {num_subtasks}")
        print(f"  - 并行任务组数量: {num_parallel_groups}")
        
        print("=" * 60)
        print("Task Decomposition 节点完成")
        print("=" * 60)
        
        return state
        
    except Exception as e:
        print(f"  [错误] 子图执行失败: {e}")
        import traceback
        traceback.print_exc()
        print("  [回退] 使用简化版 task_decomposition")
        return _task_decomposition_node_fallback(state)


def general_qa_node(state: GlobalState) -> GlobalState:
    """通用问答节点 (待实现)"""
    print("=" * 60)
    print("General QA 节点 (待实现)")
    print("=" * 60)
    return state


# =============================================================================
# Executor 节点 - 使用 CodeAct Todo 模式
# =============================================================================

def _executor_node_fallback(state: GlobalState) -> GlobalState:
    """
    Executor 节点回退实现（当 CodeAct 子图不可用时使用）
    
    简化版本：仅标记任务为已完成
    """
    print("  [回退] CodeAct 子图不可用，使用简化实现")
    
    if not state.subtasks:
        print("  [警告] 没有任务需要执行")
        return state
    
    # 标记所有任务为已完成
    for task in state.subtasks:
        task.result = {"fallback": True, "message": "CodeAct subgraph unavailable"}
        state.completed_tasks[task.task_id] = task
    
    if not state.merged_result:
        state.merged_result = {}
    state.merged_result["executor_results"] = {
        "fallback": True,
        "message": "CodeAct subgraph unavailable",
        "completed_count": len(state.subtasks)
    }
    
    print(f"  [回退] 标记了 {len(state.subtasks)} 个任务为已完成")
    return state


# 预编译 codeact 子图（延迟加载，todo 模式）
_codeact_subgraph = None


def _get_codeact_subgraph():
    """获取 codeact 子图实例（单例模式，todo 模式）"""
    global _codeact_subgraph
    if _codeact_subgraph is None and CODEACT_SUBGRAPH_AVAILABLE:
        print("[Main Graph] 编译 codeact 子图 (todo 模式)...")
        _codeact_subgraph = build_codeact_subgraph(use_todo_mode=True)
        print("[Main Graph] codeact 子图编译完成")
    return _codeact_subgraph


def _codeact_input_mapper(global_state: GlobalState) -> CodeActState:
    """
    将 GlobalState 映射到 CodeActState（用于 todo 模式）
    
    CodeAct todo 模式需要：
    - parent_state: 引用 GlobalState 获取 session_id, opensandbox_id
    - todo_list_path: todo-list.md 的路径
    - task: 一个占位 SubTask（用于满足 CodeActState 必填字段）
    - execution_mode: 默认 CODEACT
    """
    # 创建一个占位 SubTask（CodeAct todo 模式会从 todo-list.md 读取真实任务）
    placeholder_task = SubTask(
        task_id="placeholder",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="[CodeAct Todo Mode] Tasks will be loaded from todo-list.md",
        dependencies=[],
        parallel_group_id=None
    )
    
    # 确定 todo-list.md 路径
    session_id = global_state.session_id
    if session_id:
        todo_list_path = f"/data/sessions/{session_id}/todo-list.md"
    else:
        todo_list_path = None
    
    # 构建 CodeActState
    return CodeActState(
        task=placeholder_task,
        task_description="[CodeAct Todo Mode] Executing tasks from todo-list.md",
        tools=[],
        inputs=[],
        parameters={
            "sandbox_dir": f"/data/sessions/{session_id}" if session_id else None,
            "todo_list_path": todo_list_path,
        },
        execution_mode=CodeActExecutionMode.CODEACT,
        parent_state=global_state,
        todo_list_path=todo_list_path,
        user_input_context=global_state.user_input,
    )


def _codeact_output_mapper(codeact_state: CodeActState | dict | list, global_state: GlobalState) -> GlobalState:
    """
    将 CodeActState 结果映射回 GlobalState
    
    从 CodeAct todo 模式获取：
    - todo_list: 更新后的任务列表
    - completed_tasks: 已完成的任务
    
    Args:
        codeact_state: CodeAct 子图返回的状态，可能是 CodeActState 对象、dict 或 list
        global_state: 全局状态
    """
    if not codeact_state:
        return global_state
    
    # 处理 list 类型输入（从 LangGraph stream 返回的某些节点）
    if isinstance(codeact_state, list):
        # 尝试从 list 中找到包含 todo_list 的元素
        for item in codeact_state:
            if isinstance(item, dict):
                if "todo_list" in item or hasattr(item, 'todo_list'):
                    codeact_state = item
                    break
            elif hasattr(item, 'todo_list'):
                codeact_state = item
                break
        else:
            # 没有找到有效元素，返回原状态
            print("  [_codeact_output_mapper] 警告: list 中未找到有效的 todo_list 数据")
            return global_state
    
    # 处理 dict 类型输入（从 LangGraph stream 返回）
    if isinstance(codeact_state, dict):
        # 尝试从 dict 中提取 todo_list
        todo_list_data = codeact_state.get("todo_list")
        todo_list_path = codeact_state.get("todo_list_path")
        
        if todo_list_data:
            # 如果 todo_list 是 TodoList 对象
            if hasattr(todo_list_data, 'tasks'):
                todo_list = todo_list_data
            # 如果是 dict，尝试转换为 TodoList
            elif isinstance(todo_list_data, dict):
                from nodes.subagents.code_act.todo_list import TodoList
                try:
                    todo_list = TodoList(**todo_list_data)
                except Exception:
                    # 转换失败，跳过
                    return global_state
            else:
                return global_state
            
            # 处理 todo_list
            completed_tasks = {}
            for todo_task in todo_list.tasks:
                if str(getattr(todo_task, 'status', '')).endswith('COMPLETED') or \
                   (hasattr(todo_task, 'status') and todo_task.status == TodoTaskStatus.COMPLETED):
                    subtask = SubTask(
                        task_id=todo_task.id,
                        task_type=UserTaskType.EXECUTE_PLAN,
                        content=todo_task.description,
                        dependencies=getattr(todo_task, 'dependencies', []),
                        parallel_group_id=None
                    )
                    subtask.result = {
                        "status": "completed",
                        "output": getattr(todo_task, 'result', None),
                        "execution_result": getattr(todo_task, 'result', None)
                    }
                    completed_tasks[todo_task.id] = subtask
            
            global_state.completed_tasks = completed_tasks
            
            if not global_state.merged_result:
                global_state.merged_result = {}
            
            total_tasks = len(todo_list.tasks)
            completed_count = len(completed_tasks)
            failed_count = sum(1 for t in todo_list.tasks 
                             if str(getattr(t, 'status', '')).endswith('FAILED') or
                             (hasattr(t, 'status') and t.status == TodoTaskStatus.FAILED))
            
            global_state.merged_result["executor_results"] = {
                "total_tasks": total_tasks,
                "completed_count": completed_count,
                "failed_count": failed_count,
                "todo_list_path": todo_list_path
            }
        
        return global_state
    
    # 处理 CodeActState 对象类型输入
    # 从 todo_list 获取执行结果
    if codeact_state.todo_list:
        completed_tasks = {}
        for todo_task in codeact_state.todo_list.tasks:
            if todo_task.status == TodoTaskStatus.COMPLETED:
                # 创建 SubTask 结果
                subtask = SubTask(
                    task_id=todo_task.id,
                    task_type=UserTaskType.EXECUTE_PLAN,
                    content=todo_task.description,
                    dependencies=todo_task.dependencies,
                    parallel_group_id=None
                )
                subtask.result = {
                    "status": "completed",
                    "output": todo_task.result,
                    "execution_result": todo_task.result
                }
                completed_tasks[todo_task.id] = subtask
        
        global_state.completed_tasks = completed_tasks
        
        # 更新 merged_result
        if not global_state.merged_result:
            global_state.merged_result = {}
        
        total_tasks = len(codeact_state.todo_list.tasks)
        completed_count = len(completed_tasks)
        failed_count = sum(1 for t in codeact_state.todo_list.tasks if t.status == TodoTaskStatus.FAILED)
        
        global_state.merged_result["executor_results"] = {
            "total_tasks": total_tasks,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "todo_list_path": codeact_state.todo_list_path
        }
    
    return global_state


def executor_node(state: GlobalState) -> GlobalState:
    """
    Executor 节点 - 使用 CodeAct 子图 (todo-list 驱动模式)
    
    📦 沙盒职责：
    - 读取 todo-list.md: /data/sessions/{session_id}/todo-list.md
    - 执行 MCP 工具，输出保存到: /data/sessions/{session_id}/output/
    - 参数预处理文件（CSV转FASTA等）在沙盒中执行
    - 更新任务状态到 todo-list.md
    
    执行步骤：
    1. 生成 todo-list.md（从 state.subtasks 转换）
    2. 调用 CodeAct 子图 (todo 模式):
       - read_todo - 读取 todo-list.md
       - select_next_task - 选择下一个待执行任务
       - infer_parameters - 推断任务参数
       - explore_data - 数据探索
       - generate_code - 生成代码
       - execute_code - 执行代码
       - extract_file_params - 提取输出文件
       - validate_output - 验证输出
       - update_todo - 更新任务状态
       - 循环直到所有任务完成
    3. 映射结果回 GlobalState
    
    所有代码执行通过 CodeAct 在沙盒中进行。
    """
    print("=" * 60)
    print("Executor 节点启动 (CodeAct Todo 模式)")
    print("=" * 60)
    
    # 检查是否有任务需要执行
    if not state.subtasks:
        print("  [信息] 没有任务需要执行")
        return state
    
    # 检查子图是否可用
    if not CODEACT_SUBGRAPH_AVAILABLE:
        return _executor_node_fallback(state)
    
    # 获取子图
    subgraph = _get_codeact_subgraph()
    if subgraph is None:
        print("  [错误] CodeAct 子图不可用，使用回退模式")
        return _executor_node_fallback(state)
    
    try:
        # 1. 生成 todo-list.md（从 SubTask 转换）
        print("  [1/4] 生成 todo-list.md...")
        
        if TODOLIST_GENERATOR_AVAILABLE and state.session_id:
            # 获取 opensandbox_id 用于远程保存
            opensandbox_id = state.opensandbox_id
            if not opensandbox_id and state.merged_result:
                opensandbox_id = state.merged_result.get('opensandbox_id')
            
            todo_list = generate_and_save_todolist_from_state(
                global_state=state,
                opensandbox_id=opensandbox_id
            )
            if todo_list:
                print(f"        ✓ 已生成 todo-list.md: {len(todo_list.tasks)} 个任务")
            else:
                print("        ⚠ 生成 todo-list.md 失败，将继续尝试执行")
        else:
            print("        ⚠ TodolistGenerator 不可用或无 session_id，CodeAct 将尝试读取已有文件")
        
        # 2. 映射全局状态到 CodeAct 状态
        print("  [2/4] 映射全局状态到 CodeAct 状态...")
        codeact_state = _codeact_input_mapper(state)
        
        # 3. 执行 CodeAct 子图 (todo 模式)
        print("  [3/4] 执行 CodeAct 子图 (todo 模式)...")
        print("        流程: read_todo → select_next_task → infer_params → explore_data → generate_code → execute_code → validate_output → update_todo → [循环]")
        
        result_state = subgraph.invoke(codeact_state)
        
        # 4. 映射 CodeAct 结果回全局状态
        print("  [4/4] 映射 CodeAct 结果到全局状态...")
        state = _codeact_output_mapper(result_state, state)
        
        # 打印执行结果摘要
        if state.merged_result and "executor_results" in state.merged_result:
            results = state.merged_result["executor_results"]
            print(f"  - 总任务数: {results.get('total_tasks', 0)}")
            print(f"  - 已完成: {results.get('completed_count', 0)}")
            print(f"  - 失败: {results.get('failed_count', 0)}")
        
        print("=" * 60)
        print("Executor 节点完成 (CodeAct Todo 模式)")
        print("=" * 60)
        
        return state
        
    except Exception as e:
        print(f"  [错误] CodeAct 子图执行失败: {e}")
        import traceback
        traceback.print_exc()
        print("  [回退] 使用简化版 executor")
        return _executor_node_fallback(state)


# =============================================================================
# Result Evaluator 节点 - 回退实现（当子图不可用时）
# =============================================================================

def _result_evaluator_node_fallback(state: GlobalState) -> GlobalState:
    """
    Result Evaluator 节点回退实现（当子图不可用时使用）
    
    简化版本：仅生成基本摘要
    """
    print("  [回退] Result Evaluator 子图不可用，使用简化实现")
    
    if not state.merged_result:
        state.merged_result = {}
    
    # 生成基本摘要
    completed_count = len(state.completed_tasks)
    total_count = len(state.subtasks)
    
    state.merged_result["result_evaluation"] = {
        "fallback": True,
        "message": "Result Evaluator subgraph unavailable",
        "summary": f"已完成 {completed_count}/{total_count} 个任务",
        "completed_tasks": completed_count,
        "total_tasks": total_count
    }
    
    print(f"  [回退] 生成了基本摘要: {completed_count}/{total_count} 任务完成")
    return state


# =============================================================================
# Result Evaluator 节点 - 主实现（使用子图）
# =============================================================================

# 预编译 result_evaluator 子图（延迟加载）
_result_evaluator_subgraph = None


def _get_result_evaluator_subgraph():
    """获取 result_evaluator 子图实例（单例模式）"""
    global _result_evaluator_subgraph
    if _result_evaluator_subgraph is None and RESULT_EVALUATOR_SUBGRAPH_AVAILABLE:
        print("[Main Graph] 编译 result_evaluator 子图...")
        _result_evaluator_subgraph = build_result_evaluator_subgraph()
        print("[Main Graph] result_evaluator 子图编译完成")
    return _result_evaluator_subgraph


def result_evaluator_node(state: GlobalState) -> GlobalState:
    """
    Result Evaluator 节点 - 使用子图
    
    📦 沙盒职责：
    - 从沙盒收集所有输出文件: /data/sessions/{session_id}/output/
    - 生成最终报告保存到: /data/sessions/{session_id}/output/reports/
    - 输出文件: result_evaluation_*.md, analysis_report_*.txt
    
    执行步骤（由子图完成）:
    1. 结果收集 - 通过 CodeAct 从沙盒 output/ 目录收集所有工具输出文件
    2. 结果分析 - 分析成功/失败情况，提取关键发现
    3. 报告生成 - 使用 LLM 生成最终总结报告，保存到沙盒
    
    通过 CodeAct 统一接口读取和保存文件。
    """
    print("=" * 60)
    print("Result Evaluator 节点启动")
    print("=" * 60)
    
    # 检查子图是否可用
    if not RESULT_EVALUATOR_SUBGRAPH_AVAILABLE:
        return _result_evaluator_node_fallback(state)
    
    # 获取子图
    subgraph = _get_result_evaluator_subgraph()
    if subgraph is None:
        print("  [错误] 子图不可用，使用回退模式")
        return _result_evaluator_node_fallback(state)
    
    try:
        # 1. 映射全局状态到子图状态
        print("  [1/3] 映射全局状态到子图状态...")
        evaluator_state = result_evaluator_input_mapper(state)
        
        # 2. 执行子图
        print("  [2/3] 执行 result_evaluator 子图...")
        print("        流程: collect_results → analyze_results → generate_report")
        
        result_state = subgraph.invoke(evaluator_state)
        
        # 3. 映射子图结果回全局状态
        print("  [3/3] 映射子图结果到全局状态...")
        state = result_evaluator_output_mapper(result_state, state)
        
        # 打印评估结果摘要
        evaluation = state.merged_result.get("result_evaluation", {})
        if evaluation.get("txt_report_path"):
            print(f"  - TXT 报告路径: {evaluation['txt_report_path']}")
        if evaluation.get("key_findings"):
            print(f"  - 关键发现数: {len(evaluation['key_findings'])}")
        
        print("=" * 60)
        print("Result Evaluator 节点完成")
        print("=" * 60)
        
        return state
        
    except Exception as e:
        print(f"  [错误] 子图执行失败: {e}")
        import traceback
        traceback.print_exc()
        print("  [回退] 使用简化版 result_evaluator")
        return _result_evaluator_node_fallback(state)


# =============================================================================
# Memory Saver 节点 - 流程结束后存储 Mem0 记忆
# =============================================================================

def memory_saver_node(state: GlobalState) -> GlobalState:
    """
    Memory Saver 节点 - 在流程结束后存储 Mem0 记忆
    
    📋 执行逻辑：
    1. 检查所有任务是否完美完成（所有任务成功，无失败）
    2. 如果完美完成，将 Immunity 产出存储到 Mem0
    3. 存储内容包括：
       - 用户输入（向量化）
       - 优化查询
       - 研究摘要
       - 假设摘要
       - 实验计划
       - 评估结果
       - Todo-List 执行摘要
    """
    print("=" * 60)
    print("Memory Saver 节点启动")
    print("=" * 60)
    
    if not MEM0_MANAGER_AVAILABLE:
        print("  [跳过] Mem0 管理器不可用")
        return state
    
    try:
        # 1. 检查是否所有任务完美完成
        is_perfect, summary = check_all_tasks_completed_successfully(state.merged_result or {})
        
        print(f"  任务执行统计:")
        print(f"    - 总任务数: {summary['total_tasks']}")
        print(f"    - 完成任务: {summary['completed_tasks']}")
        print(f"    - 失败任务: {summary['failed_tasks']}")
        print(f"    - 成功率: {summary['success_rate']*100:.1f}%")
        print(f"    - 是否完美: {is_perfect}")
        
        if not is_perfect:
            print("  [跳过] 任务未完美完成，不存储记忆")
            return state
        
        # 2. 获取 Immunity 产出
        immunity_plan = state.merged_result.get("immunity_plan", {})
        if not immunity_plan:
            print("  [跳过] 没有 Immunity 产出，不存储记忆")
            return state
        
        # 3. 存储到 Mem0
        print("  [存储] 正在将 Immunity 产出存储到 Mem0...")
        
        trace_id = save_immunity_trace_sync(
            user_input=state.user_input,
            optimized_questions=immunity_plan.get("optimized_questions", []),
            research_summary=immunity_plan.get("research_summary", ""),
            hypothesis_summary=immunity_plan.get("hypothesis_summary", ""),
            final_enhanced_plan=immunity_plan.get("final_enhanced_plan", ""),
            final_evaluation=immunity_plan.get("evaluation", ""),
            execution_plan=immunity_plan.get("experimental_plan", state.execution_plan or ""),
            todo_list_summary=summary,
            tool_calls=[],  # 可以后续从 executor_results 中提取
            output_paths=list(state.file_paths.values()) if state.file_paths else [],
            execution_time_seconds=0.0,  # 可以后续添加
            session_id=state.session_id,
            status="success"
        )
        
        if trace_id:
            print(f"  ✅ 记忆存储成功: {trace_id}")
        else:
            print("  ⚠️ 记忆存储失败")
        
        print("=" * 60)
        print("Memory Saver 节点完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"  [错误] Memory Saver 执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    return state


# =============================================================================
# 构建主图
# =============================================================================

def build_main_graph(use_iterative_executor: bool = None):
    """
    构建主图
    
    ================================================================================
    沙盒执行架构
    ================================================================================
    
    所有任务在沙盒中执行，所有文件保存在沙盒目录:
    
    /data/sessions/{session_id}/
    ├── input/              # 用户上传的输入文件
    ├── output/             # 工具执行产生的输出文件
    │   ├── reports/        # 各类报告
    │   └── *.csv, *.json   # 工具输出数据
    ├── todo-list.md        # 任务列表 (CodeAct Todo 模式)
    └── workspace/          # 工作空间
    
    ================================================================================
    流程（根据 use_iterative_executor 选择）
    ================================================================================
    
    **原流程** (use_iterative_executor=False):
    START → supervisor → [路由]
           ├── immunity → task_decomposition → executor (CodeAct Todo) → result_evaluator → memory_saver → END
           ├── task_decomposition → executor (CodeAct Todo) → result_evaluator → memory_saver → END
           └── general_qa → END
    
    **新流程** (use_iterative_executor=True):
    START → supervisor → [路由]
           ├── immunity → iterative_executor → result_evaluator → memory_saver → END
           ├── iterative_executor → result_evaluator → memory_saver → END
           └── general_qa → END
    
    ================================================================================
    参数
    ================================================================================
    
    Args:
        use_iterative_executor: 是否使用 IterativeExecutor 替代 task_decomposition + executor
                               - None: 自动检测（根据 ITERATIVE_EXECUTOR_AVAILABLE）
                               - True: 强制使用 iterative_executor
                               - False: 使用原流程
    
    ================================================================================
    """
    # 确保 GlobalState 模型已重建，以解析 TodoList 等前向引用
    # 这是 Pydantic v2 + LangGraph 的必要步骤
    ensure_global_state_rebuilt()
    
    # 决定是否使用 iterative_executor
    if use_iterative_executor is None:
        use_iterative = ITERATIVE_EXECUTOR_SUBGRAPH_AVAILABLE and ITERATIVE_EXECUTOR_AVAILABLE
    else:
        use_iterative = use_iterative_executor
    
    print(f"[Main Graph] 构建 {'iterative_executor' if use_iterative else 'task_decomposition + executor'} 流程")
    
    graph = StateGraph(GlobalState)
    
    # 添加公共节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("immunity", immunity_node)
    graph.add_node("result_evaluator", result_evaluator_node)
    graph.add_node("memory_saver", memory_saver_node)
    graph.add_node("general_qa", general_qa_node)
    
    if use_iterative:
        # ================================================================
        # 新流程：使用 iterative_executor
        # ================================================================
        graph.add_node("iterative_executor", iterative_executor_node)
        
        # 添加边: START → supervisor
        graph.add_edge(START, "supervisor")
        
        # 条件边: supervisor → 根据任务类型路由
        graph.add_conditional_edges(
            "supervisor",
            supervisor_router,
            {
                "immunity": "immunity",
                "task_decomposition": "iterative_executor",  # 重定向到 iterative_executor
                "general_qa": "general_qa"
            }
        )
        
        # 执行流程边
        # immunity 完成后流转到 iterative_executor
        graph.add_edge("immunity", "iterative_executor")
        # iterative_executor 完成后流转到 result_evaluator
        graph.add_edge("iterative_executor", "result_evaluator")
        
    else:
        # ================================================================
        # 原流程：使用 task_decomposition + executor
        # ================================================================
        graph.add_node("task_decomposition", task_decomposition_node)
        graph.add_node("executor", executor_node)
        
        # 添加边: START → supervisor
        graph.add_edge(START, "supervisor")
        
        # 条件边: supervisor → 根据任务类型路由
        graph.add_conditional_edges(
            "supervisor",
            supervisor_router,
            {
                "immunity": "immunity",
                "task_decomposition": "task_decomposition",
                "general_qa": "general_qa"
            }
        )
        
        # 执行流程边
        # immunity 完成后流转到 task_decomposition (实验计划 → 任务分解)
        graph.add_edge("immunity", "task_decomposition")
        # task_decomposition 完成后流转到 executor (任务分解 → 任务执行)
        graph.add_edge("task_decomposition", "executor")
        # executor 完成后流转到 result_evaluator (任务执行 → 结果评估)
        graph.add_edge("executor", "result_evaluator")
    
    # 公共结束流程
    # result_evaluator 完成后流转到 memory_saver (结果评估 → 记忆存储)
    graph.add_edge("result_evaluator", "memory_saver")
    # memory_saver 完成后结束
    graph.add_edge("memory_saver", END)
    # general_qa 直接结束
    graph.add_edge("general_qa", END)
    
    return graph.compile()


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    # 测试主图
    print("构建主图...")
    graph = build_main_graph()
    print("主图构建完成!")
    print(f"节点: {list(graph.nodes.keys())}")
    
    # 测试 supervisor 子图集成
    print("\n测试 supervisor 子图集成...")
    test_state = GlobalState(
        user_input="分析 /data/benchmark/flu_benchmark/260129_flu_metadata.csv 文件中的抗体数据，预测哪些抗体可以中和 H1N1 病毒。使用 igblast 服务。",
        sandbox_dir="/tmp/test"
    )
    
    print(f"\n用户输入: {test_state.user_input[:80]}...")
    
    # 仅测试 supervisor 节点
    result = supervisor_node(test_state)
    
    print(f"\n结果:")
    print(f"  - Session ID: {result.session_id}")
    print(f"  - 任务类型: {result.user_task_type}")
    print(f"  - 参数表: {len(result.extracted_parameters) if result.extracted_parameters else 0} 字段")
    print(f"  - 文件路径: {result.file_paths}")
