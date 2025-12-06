"""基于 langgraph-bigtool 的动态执行节点实现

该模块使用 langgraph-bigtool 框架替换原有的 dynamic_execute_node，
提供更高效的工具检索和执行机制，并支持可选的人机交互功能。"""

import asyncio
import json
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from langgraph_bigtool.graph import create_agent
from pydantic import BaseModel

from common.factory import get_mcp_client, get_reasoning_model
from usecases.cell.state.state import ExecuteState
from usecases.execute.graph.generic_executor import get_all_tools


async def build_tool_registry(config: RunnableConfig) -> Dict[str, BaseTool]:
    """
    构建工具注册表，包含所有可用的 MCP 工具和本地工具

    Args:
        config: 运行时配置

    Returns:
        Dict[str, BaseTool]: 工具ID到工具对象的映射
    """
    print("[build_tool_registry] 开始构建工具注册表")

    try:
        # 复用 generic_executor 中的 get_all_tools 逻辑
        tools = await get_all_tools(config)

        # 构建工具注册表
        tool_registry = {}
        for tool in tools:
            tool_registry[tool.name] = tool

        print(
            f"[build_tool_registry] 工具注册表构建完成，共 {len(tool_registry)} 个工具"
        )

        return tool_registry

    except Exception as e:
        print(f"[build_tool_registry] 构建工具注册表时发生错误: {str(e)}")
        return {}


async def populate_tool_store(
    tool_registry: Dict[str, BaseTool], store: InMemoryStore
) -> None:
    """
    将工具信息填充到存储中，用于语义检索

    Args:
        tool_registry: 工具注册表
        store: 内存存储实例
    """
    print("[populate_tool_store] 开始填充工具存储")

    namespace = ("tools",)

    for tool_id, tool in tool_registry.items():
        # 构建工具的描述信息用于语义检索
        tool_description = f"{tool.name}: {tool.description}"

        # 将工具信息存储到 store 中，格式需要与embedding搜索兼容
        # 根据InMemoryStore的要求，只存储在fields中指定的字段用于embedding搜索
        await store.aput(
            namespace,
            tool_id,
            {
                "description": tool_description,  # 只存储description字段用于embedding搜索
            },
        )

    print(f"[populate_tool_store] 工具存储填充完成，共存储 {len(tool_registry)} 个工具")


async def dynamic_execute_node_bigtool(
    state: ExecuteState, config: RunnableConfig, *, store=None
) -> ExecuteState:
    """
    基于 langgraph-bigtool 的动态执行节点

    该函数使用 langgraph-bigtool 框架来执行任务，提供智能的工具检索和执行能力。
    人机交互功能应该在图级别配置，而不是在单个节点中。

    Args:
        state: Cell模块的状态对象
        config: 运行时配置
        store: LangGraph 存储实例（自动注入）

    Returns:
        ExecuteState: 更新后的cell状态
    """
    # 如果没有提供 store，创建一个默认的 InMemoryStore
    if store is None:
        from langchain_openai import OpenAIEmbeddings

        from usecases.cell.cell_config import get_cell_runnable_config

        # 获取嵌入模型配置
        cell_config = get_cell_runnable_config()
        embedding_config = cell_config["configurable"]["model_config"][
            "embedding_model"
        ]
        embeddings = OpenAIEmbeddings(
            model=embedding_config["model"], **embedding_config["params"]
        )

        # 创建默认的 InMemoryStore
        store = InMemoryStore(
            index={
                "embed": embeddings,
                "dims": 1536,
                "fields": ["description"],
            }
        )

    print("[dynamic_execute_node_bigtool] 开始基于 langgraph-bigtool 的动态执行")

    try:
        # 1. 检查是否有任务需要执行
        tasks = state.decomposed_tasks
        if not tasks:
            print("[dynamic_execute_node_bigtool] 没有需要执行的任务")
            return state

        print(f"[dynamic_execute_node_bigtool] 提取到任务: {tasks}")

        # 2. 构建工具注册表
        tool_registry = await build_tool_registry(config)
        if not tool_registry:
            print("[dynamic_execute_node_bigtool] 工具注册表为空，无法执行任务")
            return state

        # 3. 使用传入的存储实例并填充工具信息
        if store is not None:
            await populate_tool_store(tool_registry, store)

        # 4. 获取语言模型
        llm = get_reasoning_model(config)

        # 5. 创建 bigtool agent
        print("[dynamic_execute_node_bigtool] 创建 bigtool agent")
        agent_builder = create_agent(
            llm=llm,
            tool_registry=tool_registry,
            limit=3,  # 每次最多检索3个工具
            filter=None,  # 不使用过滤器
            namespace_prefix=("tools",),
        )

        # 6. 编译 agent
        agent_graph = agent_builder.compile(checkpointer=MemorySaver(), store=store)

        # 7. 执行所有任务
        print("[dynamic_execute_node_bigtool] 开始执行任务")
        task_results = []

        for i, task in enumerate(tasks):
            print(
                f"[dynamic_execute_node_bigtool] 执行任务 {i + 1}/{len(tasks)}: {task}"
            )

            # 构建初始状态
            initial_state = {
                "messages": [{"role": "user", "content": task}],
                "selected_tool_ids": [],
            }

            # 执行任务
            execution_config = {**config, "recursion_limit": 50}
            result = await agent_graph.ainvoke(initial_state, config=execution_config)

            task_results.append({"task": task, "result": result, "status": "completed"})

            print(f"[dynamic_execute_node_bigtool] 任务 {i + 1} 执行完成")

        # 8. 更新状态
        print("[dynamic_execute_node_bigtool] 所有任务执行完成，更新状态")
        updated_state = state.model_copy()

        return updated_state

    except Exception as e:
        print(f"[dynamic_execute_node_bigtool] 执行过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
        return state


# 为了保持向后兼容性，提供一个别名
dynamic_execute_node = dynamic_execute_node_bigtool
