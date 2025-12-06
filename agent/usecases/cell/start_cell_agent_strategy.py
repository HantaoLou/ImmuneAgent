#!/usr/bin/env python3
"""
测试strategy_selection_node节点的简单LangGraph流程
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import uuid

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import SecretStr

from common.constants import QWEN_BASE_URL, QWEN_MODEL_VLLM, REASONING_MODEL
from usecases.cell.graph.planning_graph import (
    should_retry_strategy_input,
    strategy_selection_node,
    user_input_strategy_node,
)
from usecases.cell.state.state import State


def create_test_graph():
    """创建测试图"""
    memory = MemorySaver()
    graph = StateGraph(State)

    # 添加节点
    graph.add_node("user_input_strategy", user_input_strategy_node)
    graph.add_node("strategy_selection", strategy_selection_node)

    # 添加边
    graph.add_edge(START, "user_input_strategy")
    graph.add_conditional_edges(
        "user_input_strategy",
        should_retry_strategy_input,
        {"continue": "strategy_selection", "retry": "user_input_strategy"},
    )
    graph.add_edge("strategy_selection", END)

    return graph.compile(checkpointer=memory)


def test_strategy_selection():
    """
    测试user_input_strategy_node和strategy_selection_node节点
    测试输入验证和重新输入功能
    参考官方文档run_planning_graph函数的持续运行模式
    """
    print("=== 测试user_input_strategy和strategy_selection节点 ===")
    print("提示：输入为空时会要求重新输入，输入有效内容才能继续")

    # 定义测试用的用户问题
    user_question = "测试单细胞数据分析策略选择"

    # 创建测试图
    test_graph = create_test_graph()

    # 初始状态
    initial_state = State(
        refine_plan="",
        plan_confirmed=False,
        model_upload_files={"flu": "", "rsv": "", "sars": "", "a1a11": ""},
        strategy_upload_files="",
        strategy_input_valid=True,
        selected_model=[],
        selected_strategy=[],
        original_question=user_question,
        optimized_questions=[],
        generated_plan="",
        context="",
        individual_plans=[],
    )
    # 配置
    _uuid = uuid.uuid4()
    from common.constants import STANDARDIZED_WORKING_DIRECTORY

    work_directory = os.path.join(STANDARDIZED_WORKING_DIRECTORY, str(_uuid))

    os.makedirs(work_directory, exist_ok=True)
    from cell_config import get_cell_runnable_config

    config = get_cell_runnable_config(_uuid, work_directory)

    try:
        print("开始运行图...")

        # 先运行到第一次中断 (参考官方案例)
        _ = list(test_graph.stream(initial_state, config))

        # 持续运行直到工作流完成 (参考官方案例)
        while True:
            # 检查工作流状态
            state = test_graph.get_state(config)

            # 如果工作流已完成（没有下一个节点），退出循环
            if not state.next:
                break

            # 如果有下一个节点，说明需要继续执行（可能被中断）
            print(state.interrupts[0].value)
            user_input = input("> ").strip()

            # 使用用户输入恢复工作流 (参考官方案例)
            _ = list(test_graph.stream(Command(resume=user_input), config))

        # 获取最终状态
        final_state = test_graph.get_state(config).values
        print(f"\n=== 工作流执行完成 ===")
        print(f"最终状态: {final_state}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("脚本开始执行")
    try:
        test_strategy_selection()
    except Exception as e:
        print(f"主函数执行失败: {e}")
        import traceback

        traceback.print_exc()
    print("脚本执行结束")
