# subgraphs/parallel_group_subgraph.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

from state import SubTask, ParallelTaskGroup, TaskStatus, GlobalState

# 并行子图状态
class ParallelGroupSubgraphState(BaseModel):
    parallel_group: ParallelTaskGroup = Field(description="待执行的并行任务组")
    completed_group: Optional[ParallelTaskGroup] = Field(default=None, description="执行完成的并行任务组")

# 子图节点：并行执行组内任务
def run_parallel_node(state: ParallelGroupSubgraphState) -> Dict[str, Any]:
    """线程池并行执行组内任务"""
    group = state.parallel_group
    def execute_single_task(task: SubTask) -> SubTask:
        # 模拟任务执行
        task.result = f"并行任务结果：{task.content}"
        return task
    with ThreadPoolExecutor(max_workers=5) as executor:
        completed_subtasks = list(executor.map(execute_single_task, group.subtasks))
    # 更新并行组状态为完成
    completed_group = group.model_copy(update={"status": TaskStatus.COMPLETED, "subtasks": completed_subtasks})
    return {"completed_group": completed_group}

# 构建并行子图
def build_parallel_group_subgraph():
    subgraph = StateGraph(ParallelGroupSubgraphState)
    subgraph.add_node("run_parallel", run_parallel_node)
    subgraph.add_edge(START, "run_parallel")
    subgraph.add_edge("run_parallel", END)
    return subgraph.compile()

# 主图→子图的状态映射（传入指定并行组）
def parallel_input_mapper(global_state: GlobalState, group_id: str) -> ParallelGroupSubgraphState:
    return ParallelGroupSubgraphState(parallel_group=global_state.parallel_task_groups[group_id])

# 子图→主图的状态映射（同步完成的并行组）
def parallel_output_mapper(subgraph_output: ParallelGroupSubgraphState, global_state: GlobalState) -> GlobalState:
    group = subgraph_output.completed_group
    global_state.completed_parallel_groups[group.group_id] = group
    del global_state.parallel_task_groups[group.group_id]  # 从待执行组中移除
    return global_state