# graph/main_graph.py
from typing import Dict
from state import GlobalState, UserTaskType
from langgraph.graph import StateGraph, START, END
from nodes.subagents.supervisor.graph import build_supervisor_subgraph, supervisor_input_mapper, supervisor_output_mapper

# 初始化所有子图
supervisor_subgraph = build_supervisor_subgraph()


# 主图核心节点：任务路由（检查依赖+选择子图）
def task_router_node(state: GlobalState) -> Dict[str, str]:
    """
    1. 优先执行待完成的并行任务组
    2. 检查普通任务的依赖（普通任务+并行组），执行可运行的任务
    """
    # 优先执行并行任务组
    if state.parallel_task_groups:
        group_id = next(iter(state.parallel_task_groups.keys()))
        return {"type": "parallel", "target_id": group_id}
    
    # 检查普通任务的依赖
    for task in state.subtasks:
        # 检查普通任务依赖是否完成
        dep_ok = all(dep in state.completed_tasks for dep in task.dependencies)
        # 检查并行组依赖是否完成
        parallel_dep_ok = (task.parallel_group_id is None) or (task.parallel_group_id in state.completed_parallel_groups)
        if dep_ok and parallel_dep_ok:
            return {"type": task.task_type.value, "target_id": task.task_id}
    
    # 所有任务完成
    return {"type": "complete"}


# 构建主图
def build_main_graph():
    main_graph = StateGraph(GlobalState)
    
    # 主图节点1：调用监督者子图
    def run_supervisor_subgraph(state: GlobalState) -> GlobalState:
        # 主图→子图：映射状态
        subgraph_input = supervisor_input_mapper(state)
        # 执行子图
        subgraph_output = supervisor_subgraph.invoke(subgraph_input)
        # 子图→主图：同步结果
        return supervisor_output_mapper(subgraph_output, state)
    main_graph.add_node("supervisor", run_supervisor_subgraph)

    
    
    # 定义主图流转规则
    main_graph.add_edge(START, "supervisor")
    
    # 监督者→路由：根据任务类型调用对应子图
    # def post_supervisor_router(state: GlobalState) -> str:
    #     userTaskType = state.user_task_type
    #     if userTaskType == UserTaskType.IMMUNOLOGY_TASK:
    #         return "immunity"
    #     elif userTaskType == UserTaskType.EXECUTE_PLAN:
    #         return "executor"
    #     else:
    #         return "general_qa"  # 其他任务类型（如CodeAct）

    # main_graph.add_conditional_edges("supervisor", post_supervisor_router, {
    #     "immunity": "immunity",
    #     "executor": "executor",
    #     "general_qa": "general_qa"
    # })

    # 监督者→结束
    main_graph.add_edge("supervisor", END)

    return main_graph.compile()
