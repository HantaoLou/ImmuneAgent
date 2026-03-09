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

# 导入进度报告工具
from utils.progress_reporter import (
    report_node_start,
    report_node_progress,
    report_node_complete,
    report_task_start,
    report_task_progress,
    report_task_complete,
    report_code_generation,
    report_code_execution,
    report_tool_call,
    report_error,
    report_info,
    report_llm_thinking,
    report_llm_reasoning,
    report_subgraph_step,
    report_knowledge_retrieval,
    report_analysis_progress,
)


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
    print(
        "Warning: langchain-related libraries not installed, will use keyword matching as fallback"
    )


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
    print(
        f"[Main Graph] 使用 iterative_executor 子图 (可用性: {ITERATIVE_EXECUTOR_AVAILABLE})"
    )
except ImportError as e:
    ITERATIVE_EXECUTOR_SUBGRAPH_AVAILABLE = False
    ITERATIVE_EXECUTOR_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 iterative_executor 子图: {e}")
    print("[Main Graph] 将使用 task_decomposition + executor 流程")


# =============================================================================
# General QA 子图导入
# =============================================================================
try:
    from nodes.subagents.general_qa import (
        build_general_qa_subgraph,
        general_qa_input_mapper,
        general_qa_output_mapper,
        GeneralQAState,
    )

    GENERAL_QA_SUBGRAPH_AVAILABLE = True
    print("[Main Graph] 使用 general_qa 子图")
except ImportError as e:
    GENERAL_QA_SUBGRAPH_AVAILABLE = False
    print(f"[Main Graph] 警告: 无法导入 general_qa 子图: {e}")
    print("[Main Graph] 将使用简化版 general_qa 节点")


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


def _get_llm(progress_callback=None):
    """获取 LLM 实例，支持思考过程捕获"""
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        return None

    # 如果有progress_callback，尝试创建带思考捕获的LLM
    if progress_callback:
        try:
            from utils.llm_factory import create_llm_with_callback

            llm = create_llm_with_callback(
                purpose="reasoning",
                temperature=0.1,
                progress_callback=progress_callback,
            )
            if llm:
                return llm
        except Exception as e:
            print(
                f"[Warning] Failed to create LLM with callback: {e}, falling back to standard LLM"
            )

    # 回退到标准LLM
    return create_reasoning_llm(temperature=0.1)


def _classify_task_type_fallback(user_input: str) -> UserTaskType:
    """任务分类回退函数（关键词匹配）"""
    user_input_lower = user_input.lower()

    if any(
        keyword in user_input_lower
        for keyword in [
            "execute",
            "plan",
            "step",
            "follow",
            "according to",
            "执行",
            "计划",
            "步骤",
            "按照",
        ]
    ):
        return UserTaskType.EXECUTE_PLAN

    if any(
        keyword in user_input_lower
        for keyword in [
            "immun",
            "antigen",
            "antibody",
            "vaccine",
            "immune",
            "免疫",
            "抗原",
            "抗体",
            "疫苗",
        ]
    ):
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
    Supervisor 节点 - 使用子图

    📦 沙盒职责：
    - 预处理输入（提取参数、分析文件）
    - 创建 OpenSandbox 实例（如需要）
    - 初始化沙盒目录结构
    - 保存用户上传的文件到沙盒

    执行步骤（由子图完成）:
    1. preprocess - 输入预处理
    2. detect_files - 检测文件引用
    3. upload_files - 上传文件到沙盒
    4. analyze_files - 分析文件内容
    5. build_params - 构建参数表
    6. classify - 任务分类

    所有文件操作通过 CodeAct 在沙盒中执行。
    """
    print("=" * 60)
    print("Supervisor 节点启动")
    print("=" * 60)

    report_node_start(state, "supervisor", "正在初始化任务处理...")
    report_llm_thinking(
        state, "分析用户输入，提取关键信息和意图", "输入分析", "supervisor"
    )

    if not SUPERVISOR_SUBGRAPH_AVAILABLE:
        print("  [回退] 使用简化版 supervisor")
        report_node_progress(
            state, "supervisor", "使用简化模式处理", progress_percent=50
        )
        return _supervisor_node_fallback(state)

    subgraph = _get_supervisor_subgraph()
    if subgraph is None:
        print("  [错误] 子图不可用，使用回退模式")
        report_error(
            state, "Subgraph not available, using fallback", node_name="supervisor"
        )
        return _supervisor_node_fallback(state)

    try:
        print("  [1/3] 映射全局状态到子图状态...")
        report_subgraph_step(
            state,
            "supervisor",
            "状态映射",
            "将全局状态映射到子图状态",
            progress_percent=33,
        )
        supervisor_state = supervisor_input_mapper(state)

        print("  [2/3] 执行 supervisor 子图...")
        print(
            "        流程: preprocess → detect_files → upload_files → analyze_files → build_params → classify"
        )
        report_subgraph_step(
            state,
            "supervisor",
            "执行子图",
            "预处理、文件分析、参数构建、任务分类",
            progress_percent=66,
        )
        report_llm_thinking(
            state, "使用大模型分析任务特征，确定最佳执行路径", "任务分类", "supervisor"
        )

        result_state = subgraph.invoke(supervisor_state)

        print(f"  [DEBUG] result_state type: {type(result_state).__name__}")
        if isinstance(result_state, dict):
            print(f"  [DEBUG] result_state keys: {list(result_state.keys())[:10]}")
        elif isinstance(result_state, list):
            print(f"  [DEBUG] result_state is list, len={len(result_state)}")
            if len(result_state) > 0:
                print(f"  [DEBUG] first element type: {type(result_state[0]).__name__}")

        print("  [3/3] 映射子图结果到全局状态...")
        report_subgraph_step(
            state,
            "supervisor",
            "结果映射",
            "将子图结果同步到全局状态",
            progress_percent=90,
        )
        state = supervisor_output_mapper(result_state, state)

        print("=" * 60)
        print(
            f"Supervisor 节点完成 → 路由到: {state.user_task_type.value if state.user_task_type else 'unknown'}"
        )
        print("=" * 60)

        task_type_str = state.user_task_type.value if state.user_task_type else "未知"

        # 🔥 报告节点完成（带100%进度）
        report_node_complete(
            state, "supervisor", f"任务分类完成: {task_type_str}", progress_percent=100
        )
        report_task_complete(
            state, "supervisor_task", f"Supervisor 任务完成", success=True
        )
        report_info(
            state,
            f"已确定任务类型为: {task_type_str}，准备执行相应流程",
            node_name="supervisor",
        )

        return state

    except Exception as e:
        import traceback

        print(f"  [错误] 子图执行失败: {e}")
        print(f"  [错误] 异常类型: {type(e).__name__}")
        print(f"  [错误] 完整堆栈:")
        traceback.print_exc()
        print("  [回退] 使用简化版 supervisor")
        report_error(
            state, f"Subgraph execution failed: {str(e)}", node_name="supervisor"
        )
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
        print(
            "        流程: preprocess → detect_files → upload_files → analyze_files → build_params → classify"
        )

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
        print(
            f"Supervisor 节点完成 → 路由到: {state.user_task_type.value if state.user_task_type else 'unknown'}"
        )
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
    - EXECUTE_PLAN → task_decomposition (或 iterative_executor)
    - USE_HISTORY → use_history (待实现，当前路由到 general_qa)
    - GENERAL_QA → general_qa

    注意: USE_HISTORY 是为历史记忆检索预留的类型，当前路由到 general_qa。
    """
    task_type = state.user_task_type

    routing_map = {
        UserTaskType.IMMUNOLOGY_TASK: "immunity",
        UserTaskType.EXECUTE_PLAN: "task_decomposition",
        UserTaskType.USE_HISTORY: "general_qa",  # 待实现: 应路由到 use_history 节点
        UserTaskType.GENERAL_QA: "general_qa",
    }

    return routing_map.get(task_type, "general_qa")


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
        "fallback": True,
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
        print(
            "        流程: query_decomposition → retrieval → deep_research → hypothesis_generation → planning → evaluation"
        )

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
        parallel_group_id=None,
    )

    state.subtasks = [simple_subtask]
    if not state.merged_result:
        state.merged_result = {}
    state.merged_result["task_decomposition"] = {
        "fallback": True,
        "message": "Task decomposition subgraph unavailable, using simplified task structure",
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

    # 报告节点开始
    report_node_start(
        state, "task_decomposition", "Analyzing task and creating execution plan..."
    )

    # 检查子图是否可用
    if not TASK_DECOMPOSITION_SUBGRAPH_AVAILABLE:
        print("  [回退] 使用简化版 task_decomposition")
        report_node_progress(
            state, "task_decomposition", "Using fallback mode", progress_percent=50
        )
        return _task_decomposition_node_fallback(state)

    # 获取子图
    subgraph = _get_task_decomposition_subgraph()
    if subgraph is None:
        print("  [错误] 子图不可用，使用回退模式")
        report_error(
            state,
            "Subgraph not available, using fallback",
            node_name="task_decomposition",
        )
        return _task_decomposition_node_fallback(state)

    try:
        # 1. 映射全局状态到子图状态
        print("  [1/3] 映射全局状态到子图状态...")
        report_node_progress(
            state, "task_decomposition", "Mapping global state", progress_percent=33
        )
        decomposition_state = task_decomposition_input_mapper(state)

        # 2. 执行子图
        print("  [2/3] 执行 task_decomposition 子图...")
        print("        流程: coarse_decompose → fine_decompose → infer_parallel")
        report_node_progress(
            state,
            "task_decomposition",
            "Decomposing task: coarse analysis, detailed planning, identifying parallel tasks",
            progress_percent=66,
        )

        result_state = subgraph.invoke(decomposition_state)

        # 3. 映射子图结果回全局状态
        print("  [3/3] 映射子图结果到全局状态...")
        state = task_decomposition_output_mapper(result_state, state)

        # 打印分解结果摘要
        num_subtasks = len(state.subtasks) if state.subtasks else 0
        num_parallel_groups = (
            len(state.parallel_task_groups) if state.parallel_task_groups else 0
        )
        print(f"  - 子任务数量: {num_subtasks}")
        print(f"  - 并行任务组数量: {num_parallel_groups}")

        # 报告节点完成
        report_node_complete(
            state,
            "task_decomposition",
            f"Task decomposed into {num_subtasks} subtasks and {num_parallel_groups} parallel groups",
        )

        print("=" * 60)
        print("Task Decomposition 节点完成")
        print("=" * 60)

        return state

    except Exception as e:
        print(f"  [错误] 子图执行失败: {e}")
        import traceback

        traceback.print_exc()
        print("  [回退] 使用简化版 task_decomposition")
        report_error(
            state,
            f"Subgraph execution failed: {str(e)}",
            node_name="task_decomposition",
        )
        return _task_decomposition_node_fallback(state)


# =============================================================================
# General QA 节点 - 回退实现（当子图不可用时）
# =============================================================================


def _general_qa_node_fallback(state: GlobalState) -> GlobalState:
    """
    General QA 节点回退实现（当子图不可用时使用）

    简化版本：使用基础 LLM 直接回答
    """
    print("=" * 60)
    print("General QA 节点 (回退模式)")
    print("=" * 60)

    user_input = state.user_input
    print(f"  用户输入: {user_input[:100]}...")

    if not LLM_AVAILABLE:
        state.merged_result = state.merged_result or {}
        state.merged_result["general_qa_answer"] = "LLM 不可用，无法处理通用问答"
        state.merged_result["general_qa_error"] = "LLM not available"
        return state

    try:
        # 🔥 使用带思考捕获的LLM
        progress_callback = (
            state.progress_callback if hasattr(state, "progress_callback") else None
        )
        llm = _get_llm(progress_callback=progress_callback)

        if llm is None:
            state.merged_result = state.merged_result or {}
            state.merged_result["general_qa_answer"] = "无法创建 LLM 实例"
            state.merged_result["general_qa_error"] = "Failed to create LLM"
            return state

        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = """你是一个专业的生物医学问答助手。请根据用户的问题，提供准确、专业的回答。

回答要求：
1. 简洁明了，直接回答问题
2. 如果涉及专业术语，请提供简要解释
3. 如果问题不清晰，请说明你的理解并给出最佳猜测"""

        # 🔥 报告LLM思考开始
        report_llm_thinking(
            state,
            f"分析用户问题: {user_input[:100]}",
            step_name="问题分析",
            node_name="general_qa",
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        # 🔥 报告LLM思考完成
        report_llm_thinking(
            state,
            f"生成回答: {answer[:100]}",
            step_name="答案生成",
            node_name="general_qa",
        )

        state.merged_result = state.merged_result or {}
        state.merged_result["general_qa_answer"] = answer
        state.merged_result["general_qa_fallback"] = True

        print(f"  回答: {answer[:200]}...")

    except Exception as e:
        print(f"  [错误] LLM 调用失败: {e}")
        state.merged_result = state.merged_result or {}
        state.merged_result["general_qa_answer"] = None
        state.merged_result["general_qa_error"] = str(e)

    print("=" * 60)
    return state


# =============================================================================
# General QA 节点 - 主实现（使用子图）
# =============================================================================

_general_qa_subgraph = None


def _get_general_qa_subgraph():
    """获取 general_qa 子图实例（单例模式）"""
    global _general_qa_subgraph
    if _general_qa_subgraph is None and GENERAL_QA_SUBGRAPH_AVAILABLE:
        print("[Main Graph] 编译 general_qa 子图...")
        _general_qa_subgraph = build_general_qa_subgraph()
        print("[Main Graph] general_qa 子图编译完成")
    return _general_qa_subgraph


def general_qa_node(state: GlobalState) -> GlobalState:
    """
    General QA 节点 - 使用子图

    📦 功能职责：
    - 处理通用问答任务（非免疫学、非计划执行类问题）
    - 支持多种问题类型：选择题、计算题、文本匹配等
    - 集成知识检索工具（Qdrant、Tavily、 知识图谱等）

    执行步骤（由子图完成）:
    1. N0: 输入预处理与问题分类
    2. N1: 问题分解与领域定位
    3. N2: 计算/算法需求识别（条件执行）
    4. N3: 跨领域知识检索
    5. N4: 计算步骤分解（条件执行）
    6. N6: 核心推理
    7. N8: 答案生成与验证
    8. N10: 异常处理（恢复模式）

    通过子图统一接口执行问答流程。
    """
    print("=" * 60)
    print("General QA 节点启动")
    print("=" * 60)

    # 报告节点开始
    report_node_start(state, "general_qa", "正在处理您的问题...")
    report_llm_thinking(
        state,
        "分析问题类型，确定最佳处理策略",
        step_name="问题分析",
        node_name="general_qa",
    )

    # 报告子图步骤
    report_subgraph_step(
        state,
        "general_qa",
        "N3_knowledge_retrieval",
        "正在从知识库检索相关信息",
        progress_percent=30,
    )
    report_llm_thinking(
        state,
        "使用多步骤推理分析问题，检索相关知识，生成准确答案",
        step_name="核心推理",
        node_name="general_qa",
    )

    if not GENERAL_QA_SUBGRAPH_AVAILABLE:
        print("  [回退] 使用简化版 general_qa")
        report_node_progress(state, "general_qa", "使用简化模式", progress_percent=50)
        return _general_qa_node_fallback(state)

    subgraph = _get_general_qa_subgraph()
    if subgraph is None:
        print("  [错误] 子图不可用，使用回退模式")
        report_error(state, "子图不可用", node_name="general_qa")
        return _general_qa_node_fallback(state)

    try:
        print("  [1/3] 映射全局状态到子图状态...")
        report_subgraph_step(
            state, "general_qa", "input_mapping", "准备分析问题", progress_percent=10
        )
        qa_state = general_qa_input_mapper(state)

        # 🔥 报告问题分类产物
        if hasattr(qa_state, "question_type_label") and qa_state.question_type_label:
            report_llm_thinking(
                state,
                f"问题类型: {qa_state.question_type_label} | 格式: {qa_state.answer_format_label or 'text'}",
                step_name="问题分类",
                node_name="general_qa",
                details={
                    "question_type": qa_state.question_type_label,
                    "answer_format": qa_state.answer_format_label,
                    "core_keywords": qa_state.core_keywords[:5]
                    if qa_state.core_keywords
                    else [],
                },
            )

        print("  [2/3] 执行 general_qa 子图...")
        report_subgraph_step(
            state,
            "general_qa",
            "执行流程",
            "问题分析 → 知识检索 → 推理 → 生成答案",
            progress_percent=30,
        )
        result_state = subgraph.invoke(qa_state)

        # 🔥 报告各步骤的产物（增强版）

        # 1. 问题分类结果
        if hasattr(qa_state, "question_type_label") and qa_state.question_type_label:
            report_subgraph_step(
                state,
                "general_qa",
                "问题分类",
                f"类型: {qa_state.question_type_label} | 格式: {qa_state.answer_format_label or 'text'}",
                progress_percent=15,
            )

        # 2. 知识检索结果
        if (
            hasattr(result_state, "domain_knowledge_map")
            and result_state.domain_knowledge_map
        ):
            knowledge_summary = []
            for domain, data in result_state.domain_knowledge_map.items():
                if isinstance(data, dict):
                    if data.get("facts"):
                        facts = data["facts"]
                        facts_preview = facts[:3] if len(facts) > 3 else facts
                        knowledge_summary.append(
                            f"{domain}: {', '.join(facts_preview)}"
                        )
                    elif data.get("context"):
                        context_preview = data["context"][:100]
                        knowledge_summary.append(f"{domain}: {context_preview}...")

            if knowledge_summary:
                knowledge_text = "; ".join(knowledge_summary)
                report_llm_thinking(
                    state,
                    f"检索到知识: {knowledge_text[:300]}",
                    step_name="知识检索",
                    node_name="general_qa",
                    details={
                        "domains": list(result_state.domain_knowledge_map.keys()),
                        "knowledge_preview": knowledge_text[:500],
                    },
                )

        # 3. 关键事实
        if hasattr(result_state, "key_facts") and result_state.key_facts:
            facts_list = list(result_state.key_facts.items())[:5]
            facts_text = "; ".join([f"{k}: {v[:80]}" for k, v in facts_list])
            report_llm_thinking(
                state,
                f"提取事实: {facts_text[:250]}",
                step_name="事实提取",
                node_name="general_qa",
                details={
                    "fact_count": len(result_state.key_facts),
                    "key_facts": dict(list(result_state.key_facts.items())[:5]),
                },
            )

        # 4. 核心推理结论
        if hasattr(result_state, "core_conclusion") and result_state.core_conclusion:
            report_llm_thinking(
                state,
                f"推理结论: {result_state.core_conclusion[:200]}",
                step_name="核心推理",
                node_name="general_qa",
                details={
                    "conclusion": result_state.core_conclusion[:500],
                    "confidence": result_state.match_confidence_label
                    if hasattr(result_state, "match_confidence_label")
                    else "Unknown",
                },
            )

        # 5. 最终答案
        if hasattr(result_state, "final_answer") and result_state.final_answer:
            answer_preview = result_state.final_answer[:150]
            report_llm_thinking(
                state,
                f"生成答案: {answer_preview}",
                step_name="答案生成",
                node_name="general_qa",
                details={
                    "answer_length": len(result_state.final_answer),
                    "format": result_state.answer_format_label
                    if hasattr(result_state, "answer_format_label")
                    else "text",
                },
            )

        if hasattr(result_state, "core_conclusion") and result_state.core_conclusion:
            report_llm_thinking(
                state,
                f"推理结论: {result_state.core_conclusion[:200]}",
                step_name="核心推理",
                node_name="general_qa",
            )

        # 报告推理进度
        if hasattr(result_state, "core_conclusion") and result_state.core_conclusion:
            report_llm_thinking(
                state,
                result_state.core_conclusion[:500],
                step_name="推理结论",
                node_name="general_qa",
            )

        print("  [3/3] 映射子图结果到全局状态...")
        report_subgraph_step(
            state, "general_qa", "output_mapping", "整理答案", progress_percent=90
        )
        state = general_qa_output_mapper(result_state, state)

        # 🔥 确保答案正确传递 - 如果merged_result中没有答案，直接从result_state提取
        if not state.merged_result.get("general_qa_answer"):
            if hasattr(result_state, "final_answer") and result_state.final_answer:
                state.merged_result["general_qa_answer"] = result_state.final_answer
                print(
                    f"  [修复] 从final_answer提取答案: {str(result_state.final_answer)[:100]}..."
                )
            elif (
                hasattr(result_state, "core_conclusion")
                and result_state.core_conclusion
            ):
                state.merged_result["general_qa_answer"] = result_state.core_conclusion
                print(
                    f"  [修复] 从core_conclusion提取答案: {str(result_state.core_conclusion)[:100]}..."
                )

        if state.merged_result:
            answer = state.merged_result.get("general_qa_answer")
            error = state.merged_result.get("general_qa_error")
            if answer:
                print(f"  - 最终答案: {str(answer)[:200]}...")
                # 🔥 报告完成进度
                report_node_complete(
                    state,
                    "general_qa",
                    f"问题处理完成: {str(answer)[:100]}...",
                    details={"answer_length": len(str(answer))},
                )
                report_task_complete(
                    state, "general_qa_task", f"General QA 任务完成", success=True
                )
            if error:
                print(f"  - 错误信息: {error}")
                report_error(state, f"General QA 失败: {error}", node_name="general_qa")

        print("=" * 60)
        print("General QA 节点完成")
        print("=" * 60)

        return state

    except Exception as e:
        print(f"  [错误] 子图执行失败: {e}")
        import traceback

        traceback.print_exc()
        print("  [回退] 使用简化版 general_qa")
        return _general_qa_node_fallback(state)


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
        "completed_count": len(state.subtasks),
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
        parallel_group_id=None,
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


def _codeact_output_mapper(
    codeact_state: CodeActState | dict | list, global_state: GlobalState
) -> GlobalState:
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
                if "todo_list" in item or hasattr(item, "todo_list"):
                    codeact_state = item
                    break
            elif hasattr(item, "todo_list"):
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
            if hasattr(todo_list_data, "tasks"):
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
                if str(getattr(todo_task, "status", "")).endswith("COMPLETED") or (
                    hasattr(todo_task, "status")
                    and todo_task.status == TodoTaskStatus.COMPLETED
                ):
                    subtask = SubTask(
                        task_id=todo_task.id,
                        task_type=UserTaskType.EXECUTE_PLAN,
                        content=todo_task.description,
                        dependencies=getattr(todo_task, "dependencies", []),
                        parallel_group_id=None,
                    )
                    subtask.result = {
                        "status": "completed",
                        "output": getattr(todo_task, "result", None),
                        "execution_result": getattr(todo_task, "result", None),
                    }
                    completed_tasks[todo_task.id] = subtask

            global_state.completed_tasks = completed_tasks

            if not global_state.merged_result:
                global_state.merged_result = {}

            total_tasks = len(todo_list.tasks)
            completed_count = len(completed_tasks)
            failed_count = sum(
                1
                for t in todo_list.tasks
                if str(getattr(t, "status", "")).endswith("FAILED")
                or (hasattr(t, "status") and t.status == TodoTaskStatus.FAILED)
            )

            global_state.merged_result["executor_results"] = {
                "total_tasks": total_tasks,
                "completed_count": completed_count,
                "failed_count": failed_count,
                "todo_list_path": todo_list_path,
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
                    parallel_group_id=None,
                )
                subtask.result = {
                    "status": "completed",
                    "output": todo_task.result,
                    "execution_result": todo_task.result,
                }
                completed_tasks[todo_task.id] = subtask

        global_state.completed_tasks = completed_tasks

        # 更新 merged_result
        if not global_state.merged_result:
            global_state.merged_result = {}

        total_tasks = len(codeact_state.todo_list.tasks)
        completed_count = len(completed_tasks)
        failed_count = sum(
            1
            for t in codeact_state.todo_list.tasks
            if t.status == TodoTaskStatus.FAILED
        )

        global_state.merged_result["executor_results"] = {
            "total_tasks": total_tasks,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "todo_list_path": codeact_state.todo_list_path,
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

    # 报告节点开始
    report_node_start(state, "executor", "Starting task execution...")

    # 检查是否有任务需要执行
    if not state.subtasks:
        print("  [信息] 没有任务需要执行")
        report_node_complete(state, "executor", "No tasks to execute")
        return state

    # 检查子图是否可用
    if not CODEACT_SUBGRAPH_AVAILABLE:
        report_node_progress(
            state, "executor", "Using fallback mode", progress_percent=50
        )
        return _executor_node_fallback(state)

    # 获取子图
    subgraph = _get_codeact_subgraph()
    if subgraph is None:
        print("  [错误] CodeAct 子图不可用，使用回退模式")
        report_error(
            state,
            "CodeAct subgraph not available, using fallback",
            node_name="executor",
        )
        return _executor_node_fallback(state)

    try:
        # 1. 生成 todo-list.md（从 SubTask 转换）
        print("  [1/4] 生成 todo-list.md...")
        report_node_progress(
            state,
            "executor",
            "Generating task list (todo-list.md)",
            progress_percent=25,
        )

        if TODOLIST_GENERATOR_AVAILABLE and state.session_id:
            # 获取 opensandbox_id 用于远程保存
            opensandbox_id = state.opensandbox_id
            if not opensandbox_id and state.merged_result:
                opensandbox_id = state.merged_result.get("opensandbox_id")

            todo_list = generate_and_save_todolist_from_state(
                global_state=state, opensandbox_id=opensandbox_id
            )
            if todo_list:
                print(f"        ✓ 已生成 todo-list.md: {len(todo_list.tasks)} 个任务")
                report_info(
                    state,
                    f"Generated {len(todo_list.tasks)} tasks",
                    node_name="executor",
                )
            else:
                print("        ⚠ 生成 todo-list.md 失败，将继续尝试执行")
        else:
            print(
                "        ⚠ TodolistGenerator 不可用或无 session_id，CodeAct 将尝试读取已有文件"
            )

        # 2. 映射全局状态到 CodeAct 状态
        print("  [2/4] 映射全局状态到 CodeAct 状态...")
        report_node_progress(
            state,
            "executor",
            "Mapping global state to CodeAct state",
            progress_percent=37,
        )
        codeact_state = _codeact_input_mapper(state)

        # 3. 执行 CodeAct 子图 (todo 模式)
        print("  [3/4] 执行 CodeAct 子图 (todo 模式)...")
        print(
            "        流程: read_todo → select_next_task → infer_params → explore_data → generate_code → execute_code → validate_output → update_todo → [循环]"
        )
        report_node_progress(
            state,
            "executor",
            "Executing tasks via CodeAct subgraph...",
            progress_percent=50,
        )

        result_state = subgraph.invoke(codeact_state)

        # 4. 映射 CodeAct 结果回全局状态
        print("  [4/4] 映射 CodeAct 结果到全局状态...")
        report_node_progress(
            state,
            "executor",
            "Mapping execution results to global state",
            progress_percent=87,
        )
        state = _codeact_output_mapper(result_state, state)

        # 打印执行结果摘要
        if state.merged_result and "executor_results" in state.merged_result:
            results = state.merged_result["executor_results"]
            print(f"  - 总任务数: {results.get('total_tasks', 0)}")
            print(f"  - 已完成: {results.get('completed_count', 0)}")
            print(f"  - 失败: {results.get('failed_count', 0)}")

            # 报告完成情况
            completed = results.get("completed_count", 0)
            total = results.get("total_tasks", 0)
            failed = results.get("failed_count", 0)
            report_node_complete(
                state,
                "executor",
                f"Task execution completed: {completed}/{total} tasks completed, {failed} failed",
            )

        print("=" * 60)
        print("Executor 节点完成 (CodeAct Todo 模式)")
        print("=" * 60)

        return state

    except Exception as e:
        print(f"  [错误] CodeAct 子图执行失败: {e}")
        import traceback

        traceback.print_exc()
        print("  [回退] 使用简化版 executor")
        report_error(state, f"CodeAct execution failed: {str(e)}", node_name="executor")
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
        "total_tasks": total_count,
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

    # 报告节点开始
    report_node_start(state, "result_evaluator", "Evaluating execution results...")

    # 检查子图是否可用
    if not RESULT_EVALUATOR_SUBGRAPH_AVAILABLE:
        report_node_progress(
            state, "result_evaluator", "Using fallback mode", progress_percent=50
        )
        return _result_evaluator_node_fallback(state)

    # 获取子图
    subgraph = _get_result_evaluator_subgraph()
    if subgraph is None:
        print("  [错误] 子图不可用，使用回退模式")
        report_error(
            state,
            "Subgraph not available, using fallback",
            node_name="result_evaluator",
        )
        return _result_evaluator_node_fallback(state)

    try:
        # 1. 映射全局状态到子图状态
        print("  [1/3] 映射全局状态到子图状态...")
        report_node_progress(
            state, "result_evaluator", "Mapping global state", progress_percent=33
        )
        evaluator_state = result_evaluator_input_mapper(state)

        # 2. 执行子图
        print("  [2/3] 执行 result_evaluator 子图...")
        print("        流程: collect_results → analyze_results → generate_report")
        report_node_progress(
            state,
            "result_evaluator",
            "Collecting results, analyzing execution, generating final report",
            progress_percent=66,
        )

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

        # 报告节点完成
        report_node_complete(state, "result_evaluator", "Result evaluation completed")

        print("=" * 60)
        print("Result Evaluator 节点完成")
        print("=" * 60)

        return state

    except Exception as e:
        print(f"  [错误] 子图执行失败: {e}")
        import traceback

        traceback.print_exc()
        print("  [回退] 使用简化版 result_evaluator")
        report_error(
            state, f"Subgraph execution failed: {str(e)}", node_name="result_evaluator"
        )
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
        is_perfect, summary = check_all_tasks_completed_successfully(
            state.merged_result or {}
        )

        print(f"  任务执行统计:")
        print(f"    - 总任务数: {summary['total_tasks']}")
        print(f"    - 完成任务: {summary['completed_tasks']}")
        print(f"    - 失败任务: {summary['failed_tasks']}")
        print(f"    - 成功率: {summary['success_rate'] * 100:.1f}%")
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
            execution_plan=immunity_plan.get(
                "experimental_plan", state.execution_plan or ""
            ),
            todo_list_summary=summary,
            tool_calls=[],  # 可以后续从 executor_results 中提取
            output_paths=list(state.file_paths.values()) if state.file_paths else [],
            execution_time_seconds=0.0,  # 可以后续添加
            session_id=state.session_id,
            status="success",
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
        use_iterative = (
            ITERATIVE_EXECUTOR_SUBGRAPH_AVAILABLE and ITERATIVE_EXECUTOR_AVAILABLE
        )
    else:
        use_iterative = use_iterative_executor

    print(
        f"[Main Graph] 构建 {'iterative_executor' if use_iterative else 'task_decomposition + executor'} 流程"
    )

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
                "general_qa": "general_qa",
            },
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
                "general_qa": "general_qa",
            },
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


def run_agent(
    user_input: str,
    sandbox_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    use_iterative_executor: Optional[bool] = None,
    verbose: bool = True,
) -> GlobalState:
    """
    运行agent的便捷函数

    Args:
        user_input: 用户输入
        sandbox_dir: 沙盒目录（可选，默认自动生成）
        session_id: 会话ID（可选，默认自动生成）
        use_iterative_executor: 是否使用迭代执行器（可选，默认自动检测）
        verbose: 是否打印详细日志

    Returns:
        GlobalState: 最终状态
    """
    import uuid

    # 生成默认值
    if not session_id:
        session_id = str(uuid.uuid4())
    if not sandbox_dir:
        sandbox_dir = f"./sandbox/sessions/{session_id}"

    if verbose:
        print("=" * 80)
        print("Bio-Agent 执行器")
        print("=" * 80)
        print(f"Session ID: {session_id}")
        print(f"Sandbox Dir: {sandbox_dir}")
        print(f"User Input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        print("=" * 80)

    # 构建状态
    initial_state = GlobalState(
        user_input=user_input,
        sandbox_dir=sandbox_dir,
        session_id=session_id,
    )

    # 构建图
    if verbose:
        print("\n构建执行图...")
    graph = build_main_graph(use_iterative_executor=use_iterative_executor)

    if verbose:
        print("执行图构建完成!")
        print(f"节点列表: {list(graph.nodes.keys())}")
        print("\n开始执行...\n")

    # 执行图
    result = graph.invoke(initial_state)

    if verbose:
        print("\n" + "=" * 80)
        print("执行完成!")
        print("=" * 80)

        # 打印结果摘要
        if hasattr(result, "user_task_type"):
            print(
                f"任务类型: {result.user_task_type.value if result.user_task_type else 'Unknown'}"
            )

        if hasattr(result, "completed_tasks"):
            print(f"完成任务数: {len(result.completed_tasks)}")

        if hasattr(result, "merged_result") and result.merged_result:
            if "executor_results" in result.merged_result:
                exec_results = result.merged_result["executor_results"]
                print(
                    f"执行统计: {exec_results.get('completed_count', 0)}/{exec_results.get('total_tasks', 0)} 任务完成"
                )

        if hasattr(result, "file_paths"):
            print(f"生成文件数: {len(result.file_paths)}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bio-Agent 执行器")
    parser.add_argument("--input", "-i", type=str, help="用户输入（问题或任务描述）")
    parser.add_argument("--sandbox-dir", type=str, default=None, help="沙盒目录路径")
    parser.add_argument("--session-id", type=str, default=None, help="会话ID")
    parser.add_argument("--iterative", action="store_true", help="使用迭代执行器")
    parser.add_argument(
        "--no-iterative", action="store_true", help="不使用迭代执行器（使用传统流程）"
    )
    parser.add_argument(
        "--test",
        choices=["general_qa", "supervisor", "full"],
        default="full",
        help="测试模式",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="减少输出")

    args = parser.parse_args()

    # 确定是否使用迭代执行器
    use_iterative = None
    if args.iterative:
        use_iterative = True
    elif args.no_iterative:
        use_iterative = False

    verbose = not args.quiet

    # 测试模式
    if args.test == "general_qa":
        # 测试通用问答
        test_input = args.input or "介绍一下你自己"
        print(f"\n测试通用问答: {test_input}\n")

        result = run_agent(
            user_input=test_input,
            sandbox_dir=args.sandbox_dir,
            session_id=args.session_id,
            use_iterative_executor=False,
            verbose=verbose,
        )

        if hasattr(result, "merged_result") and result.merged_result:
            answer = result.merged_result.get(
                "general_qa_answer"
            ) or result.merged_result.get("general_qa_conclusion")
            if answer:
                print(f"\n回答:\n{answer}\n")

    elif args.test == "supervisor":
        # 仅测试 supervisor 节点
        test_input = (
            args.input
            or "分析 /data/benchmark/flu_benchmark/260129_flu_metadata.csv 文件中的抗体数据"
        )

        print("\n测试 Supervisor 节点...")
        print(f"输入: {test_input}\n")

        test_state = GlobalState(
            user_input=test_input,
            sandbox_dir=args.sandbox_dir or "/tmp/test",
            session_id=args.session_id or "test-supervisor",
        )

        result = supervisor_node(test_state)

        print(f"\nSupervisor 结果:")
        print(f"  - Session ID: {result.session_id}")
        print(
            f"  - 任务类型: {result.user_task_type.value if result.user_task_type else 'Unknown'}"
        )
        print(f"  - OpenSandbox ID: {result.opensandbox_id}")
        print(
            f"  - 参数字段数: {len(result.extracted_parameters) if result.extracted_parameters else 0}"
        )
        print(f"  - 文件路径数: {len(result.file_paths)}")

        if result.supervisor_decision:
            print(f"  - 决策: {result.supervisor_decision}")
        if result.supervisor_reasoning:
            print(f"  - 推理: {result.supervisor_reasoning[:200]}...")

    else:  # full test
        # 完整流程测试
        if not args.input:
            print("警告: 未提供输入，使用默认测试输入")
            test_input = "介绍一下你自己"
        else:
            test_input = args.input

        print(f"\n执行完整流程测试")
        print(f"输入: {test_input}\n")

        result = run_agent(
            user_input=test_input,
            sandbox_dir=args.sandbox_dir,
            session_id=args.session_id,
            use_iterative_executor=use_iterative,
            verbose=verbose,
        )

        # 打印最终状态
        print("\n" + "=" * 80)
        print("最终状态摘要:")
        print("=" * 80)

        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
            for key, value in result_dict.items():
                if key in ["user_input", "session_id", "sandbox_dir", "user_task_type"]:
                    print(f"{key}: {value}")
                elif key in ["subtasks", "completed_tasks"]:
                    if isinstance(value, dict):
                        print(f"{key}: {len(value)} 项")
                    elif isinstance(value, list):
                        print(f"{key}: {len(value)} 项")
                elif key == "merged_result" and isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        if isinstance(v, dict):
                            print(f"  - {k}: {len(v)} 字段")
                        elif isinstance(v, list):
                            print(f"  - {k}: {len(v)} 项")
                        else:
                            print(f"  - {k}: {type(v).__name__}")

        print("=" * 80)
        print("测试完成!")
        print("=" * 80)
