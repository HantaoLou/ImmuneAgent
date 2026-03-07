# -*- coding: utf-8 -*-
"""
Iterative Executor 子模块

提供基于 IterativeOpenCodeExecutor 的迭代执行能力。

主要组件:
- IterativeExecutorState: 子图状态定义
- iterative_executor_node: 主节点函数
- build_iterative_executor_subgraph: 构建子图
- iterative_executor_input_mapper: 输入映射
- iterative_executor_output_mapper: 输出映射

使用方式:

```python
from nodes.subagents.iterative_executor import (
    iterative_executor_node,
    iterative_executor_input_mapper,
    iterative_executor_output_mapper,
    IterativeExecutorState,
)

# 在 main_graph 中使用
def main_graph():
    graph = StateGraph(GlobalState)
    graph.add_node("iterative_executor", iterative_executor_node)
    # ...
```
"""

# 状态定义
from nodes.subagents.iterative_executor.state import (
    IterativeExecutorState,
    IterationStatus,
    EvaluationLevel,
)

# 核心节点
from nodes.subagents.iterative_executor.node import (
    iterative_executor_node,
    iterative_executor_node_async,
    ITERATIVE_EXECUTOR_AVAILABLE,
)

# 子图构建
from nodes.subagents.iterative_executor.graph import (
    build_iterative_executor_subgraph,
    iterative_executor_subgraph_node,
    get_iterative_executor_subgraph,
)

# 输入输出映射
from nodes.subagents.iterative_executor.input_mapper import (
    iterative_executor_input_mapper,
    prepare_executor_input,
)

from nodes.subagents.iterative_executor.output_mapper import (
    iterative_executor_output_mapper,
    extract_mcp_call_summary,
    format_iteration_summary,
)

# 任务生成
from nodes.subagents.iterative_executor.task_generator import (
    generate_tasks_md,
    optimize_tasks_md,
    get_required_output_files,
)

# 提示词
from nodes.subagents.iterative_executor.prompts import (
    TASK_GENERATION_SYSTEM_PROMPT,
    TASK_GENERATION_USER_PROMPT,
    FILE_PROCESSING_TASKS_TEMPLATE,
    NETTCR_ANALYSIS_TASKS_TEMPLATE,
    IGBLAST_ANALYSIS_TASKS_TEMPLATE,
)


# ============================================================================
# 公共接口
# ============================================================================

__all__ = [
    # 状态
    "IterativeExecutorState",
    "IterationStatus",
    "EvaluationLevel",
    
    # 节点
    "iterative_executor_node",
    "iterative_executor_node_async",
    "iterative_executor_subgraph_node",
    "ITERATIVE_EXECUTOR_AVAILABLE",
    
    # 子图
    "build_iterative_executor_subgraph",
    "get_iterative_executor_subgraph",
    
    # 映射器
    "iterative_executor_input_mapper",
    "iterative_executor_output_mapper",
    "prepare_executor_input",
    "extract_mcp_call_summary",
    "format_iteration_summary",
    
    # 任务生成
    "generate_tasks_md",
    "optimize_tasks_md",
    "get_required_output_files",
    
    # 提示词
    "TASK_GENERATION_SYSTEM_PROMPT",
    "TASK_GENERATION_USER_PROMPT",
    "FILE_PROCESSING_TASKS_TEMPLATE",
    "NETTCR_ANALYSIS_TASKS_TEMPLATE",
    "IGBLAST_ANALYSIS_TASKS_TEMPLATE",
]


# ============================================================================
# 模块信息
# ============================================================================

__version__ = "1.0.0"
__author__ = "Bio-Agent Team"

