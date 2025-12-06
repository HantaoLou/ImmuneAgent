#!/usr/bin/env python3
"""
测试MetaBCR流程的简单LangGraph流程
START -> select_metabcr_model_node -> execute_metabcr_node -> END
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
    execute_metabcr_node,
    select_metabcr_model_node,
    should_retry_metabcr_input,
)
from usecases.cell.state.state import State


def create_test_metabcr_graph():
    """创建MetaBCR测试图"""
    memory = MemorySaver()
    graph = StateGraph(State)

    # 添加节点
    graph.add_node("select_metabcr_model", select_metabcr_model_node)
    graph.add_node("execute_metabcr", execute_metabcr_node)

    # 添加边
    graph.add_edge(START, "select_metabcr_model")
    graph.add_edge("select_metabcr_model", "execute_metabcr")

    # 使用条件边处理MetaBCR输入验证
    graph.add_conditional_edges(
        "execute_metabcr",
        should_retry_metabcr_input,
        {
            "skip_to_integrate": END,  # 用户选择跳过，结束流程
            "retry": "execute_metabcr",  # 成功或失败都返回自身节点
        },
    )

    return graph.compile(checkpointer=memory)


def test_metabcr_flow():
    """
    测试MetaBCR完整流程
    START -> select_metabcr_model_node -> execute_metabcr_node -> END
    """
    print("=== 测试MetaBCR完整流程 ===")
    print("流程：模型选择 -> MetaBCR执行")

    # 创建测试图
    test_graph = create_test_metabcr_graph()

    # 初始化状态
    initial_state = State(
        refine_plan="分析H5N1流感病毒的抗体-抗原相互作用，使用Meta-BCR进行预测",
        plan_confirmed=True,
        model_upload_files={},
        strategy_upload_files="",
        strategy_input_valid=True,
        metabcr_input_valid=True,
        selected_model=[],
        selected_strategy=[],
        metabcr_result="",
        original_question="测试MetaBCR分析",
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
        print("开始运行MetaBCR流程...")

        # 先运行到第一次中断
        _ = list(test_graph.stream(initial_state, config))

        # 持续运行直到工作流完成
        while True:
            # 检查工作流状态
            state = test_graph.get_state(config)

            # 如果工作流已完成（没有下一个节点），退出循环
            if not state.next:
                break

            # 如果有下一个节点，说明需要继续执行（可能被中断）
            print(state.interrupts[0].value)
            user_input = input("> ").strip()

            # 使用用户输入恢复工作流
            _ = list(test_graph.stream(Command(resume=user_input), config))

        # 获取最终状态
        final_state = test_graph.get_state(config).values
        print(f"\n=== MetaBCR流程执行完成 ===")
        print(f"选择的模型: {final_state.get('selected_model', [])}")
        print(f"MetaBCR结果: {final_state.get('metabcr_result', '')}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("MetaBCR测试脚本开始执行")
    try:
        test_metabcr_flow()
    except Exception as e:
        print(f"主函数执行失败: {e}")
        import traceback

        traceback.print_exc()
    print("MetaBCR测试脚本执行结束")
