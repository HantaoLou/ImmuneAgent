import uuid
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from common.factory import get_default_model
from common.prompts import AntibodyPrompt
from common.util.retrieval_utils import remove_think_tags
from usecases.antibody.antibody_config import get_antibody_runnable_config
from usecases.antibody.graph.execute_graph import create_execute_agent_workflow
from usecases.antibody.graph.retrieval_graph import create_rag_graph
from usecases.antibody.state.state import PlanState
from usecases.antibody.tool.planning_tools import tool_node


def validate_csv_file(file_path: str) -> tuple[bool, str]:
    """
    验证CSV文件是否存在且包含必需的字段

    参数:
        file_path: CSV文件路径
    返回:
        (是否有效, 错误信息)
    """
    try:
        # 检查文件是否存在
        path = Path(file_path)
        if not path.exists():
            return False, f"文件不存在: {file_path}"

        # 检查是否为CSV文件
        if path.suffix.lower() != ".csv":
            return False, f"文件不是CSV格式: {file_path}"

        # 读取CSV文件并检查字段
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return False, f"无法读取CSV文件: {str(e)}"

        # 检查必需的字段
        required_columns = {
            "Heavy",
            "Light",
            "experiment",
            "variant_seq",
            "label",
            "Label",
        }
        existing_columns = set(df.columns)
        missing_columns = required_columns - existing_columns

        if missing_columns:
            return False, f"CSV文件缺少必需字段: {', '.join(missing_columns)}"

        return True, "文件验证通过"

    except Exception as e:
        return False, f"验证过程中发生错误: {str(e)}"


def select_metabcr_path_node(state: PlanState):
    """
    选择MetaBCR预测的CSV文件路径

    参数:
        state: 当前状态，包含生成的计划
    返回:
        更新后的状态，包含处理后的反馈结果
    """
    message = "请选择：输入进行MetaBCR预测的CSV输入文件路径（如果不进行MetaBCR预测，请直接按Enter）： "
    user_input = interrupt(message)

    if user_input.strip() == "":
        state.csv_path = ""
        state.csv_validation_passed = True
    else:
        # 验证CSV文件
        # is_valid, _ = validate_csv_file(user_input.strip())
        is_valid = True
        if is_valid:
            state.csv_path = user_input.strip()
            state.csv_validation_passed = True
        else:
            state.csv_validation_passed = False
    return state


def execute_tools_node(state: PlanState, config: RunnableConfig):
    """使用优化后的方案优化计划，通过子图实现"""
    csv_path = state.csv_path
    tool_calls = []
    if csv_path:
        metabcr_tool_call = {
            "name": "metabcr_tool",
            "args": {"state": state},
            "id": "tool_call_id_2",
            "type": "tool_call",
        }
        tool_calls.append(metabcr_tool_call)
    refine_plan_tool_call = {
        "name": "refine_plan_tool",
        "args": {"state": state, "config": config},
        "id": "tool_call_id_1",
        "type": "tool_call",
    }
    tool_calls.append(refine_plan_tool_call)
    message_with_multiple_tool_calls = AIMessage(content="", tool_calls=tool_calls)
    result = tool_node.invoke({"messages": [message_with_multiple_tool_calls]})

    # 根据tool_call_id区分工具结果
    optimize_plan_result = ""
    metabcr_result = ""

    for msg in result.get("messages", []):
        if hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
            if msg.tool_call_id == "tool_call_id_1":
                optimize_plan_result = msg.content
            elif msg.tool_call_id == "tool_call_id_2":
                metabcr_result = msg.content

    # 直接修改原state
    state.refine_result = optimize_plan_result
    state.metabcr_result = metabcr_result
    return state


def human_feedback_node(state: PlanState):
    """人类反馈节点"""
    message = "请选择下一步操作：\n1. 结束流程\n2. 继续优化计划\n请输入选择（1或2）： "
    user_choice = interrupt(message)
    state.user_choice = user_choice
    return state


def refine_plan_node(state: PlanState, config: RunnableConfig):
    """使用模型优化计划的节点"""
    # 获取用户反馈意见
    message = "请输入您对当前计划的修改意见（直接按Enter表示无修改意见）： "
    user_feedback = interrupt(message)
    state.user_feedback = user_feedback

    # 如果用户有反馈意见，则进行计划优化
    if user_feedback and user_feedback.strip():
        refine_plan_prompt = ChatPromptTemplate.from_template(
            AntibodyPrompt.REFINE_PLAN_PROMPT
        )
        model = get_default_model(config)
        runnable = refine_plan_prompt | model | StrOutputParser() | remove_think_tags
        refined_plan = runnable.invoke(
            {
                "plan": state.generated_plan,
                "context": state.context,
                "user_feedback": user_feedback,
                "evaluation_feedback": state.refine_result,
                "specific_tasks": state.optimized_question,
            }
        )

        # 添加打印语句
        print("Plan:", state.generated_plan)
        print("Context:", state.context)
        print("User Feedback:", user_feedback)
        print("Evaluation Feedback:", state.refine_result)
        print("Specific Tasks:", state.optimized_question)

        print(f"\n===== 优化后的研究计划 =====\n{refined_plan}")
        state.refine_plan = refined_plan
        state.generated_plan = refined_plan
        state.csv_path = ""
        # 置空移到条件判断后，在流转到execute后置空，或使用标志
        # 这里暂不置空，改由条件函数处理后在别处置空，但为简单，引入标志
        state.just_refined = True
    else:
        state.just_refined = False
    return state


def should_continue_to_refine(state: PlanState):
    """
    决定是否继续到refine节点或重新选择CSV路径

    参数:
        state: 当前状态
    返回:
        字符串，表示下一步操作
    """
    # 检查CSV验证是否通过
    if state.csv_validation_passed:
        return "refine"
    else:
        return "retry_select"


def should_continue_after_feedback(state: PlanState):
    """
    根据用户反馈决定下一步操作

    参数:
        state: 当前状态
    返回:
        字符串，表示下一步操作
    """
    user_choice = state.user_choice or ""
    if user_choice == "1":
        return "execute"
    elif user_choice == "2":
        return "refine"
    else:
        # 默认到执行
        return "execute"


def should_continue_after_refine(state: PlanState):
    """
    根据用户反馈意见决定下一步操作

    参数:
        state: 当前状态
    返回:
        字符串，表示下一步操作
    """
    if state.just_refined:
        state.user_feedback = ""  # 现在置空
        state.just_refined = False
        return "execute"
    user_feedback = state.user_feedback or ""
    if user_feedback and user_feedback.strip():
        return "execute"
    else:
        return "refine"


def create_planning_graph():
    """
    创建抗体设计规划工作流图

    返回:
        编译后的LangGraph工作流图
    """
    workflow = StateGraph(PlanState)

    # 创建检索子图
    retrieval_subgraph = create_rag_graph()
    execute_subgraph = create_execute_agent_workflow()

    workflow.add_node("retrieval_assistant", retrieval_subgraph)
    workflow.add_node("select_metabcr_path_node", select_metabcr_path_node)
    workflow.add_node("execute_tools_node", execute_tools_node)
    workflow.add_node("human_feedback_node", human_feedback_node)
    workflow.add_node("refine_plan_node", refine_plan_node)
    workflow.add_node("execute_assistant", execute_subgraph)

    # 设置入口点
    workflow.set_entry_point("retrieval_assistant")

    # 连接节点
    workflow.add_edge("retrieval_assistant", "select_metabcr_path_node")

    # 添加条件边
    workflow.add_conditional_edges(
        "select_metabcr_path_node",
        should_continue_to_refine,
        {"refine": "execute_tools_node", "retry_select": "select_metabcr_path_node"},
    )
    workflow.add_edge("execute_tools_node", "human_feedback_node")
    workflow.add_conditional_edges(
        "human_feedback_node",
        should_continue_after_feedback,
        {"execute": "execute_assistant", "refine": "refine_plan_node"},
    )
    workflow.add_conditional_edges(
        "refine_plan_node",
        should_continue_after_refine,
        {"refine": "refine_plan_node", "execute": "execute_tools_node"},
    )
    workflow.add_edge("execute_assistant", END)

    # 编译工作流
    graph = workflow.compile(
        checkpointer=MemorySaver(),
    )

    try:
        print("\n===== 工作流程图代码 =====")
        print("可以将以下代码复制到任意Mermaid编辑器中查看图形:")
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"生成Mermaid代码时出错: {str(e)}")

    return graph


def run_planning_workflow(query, config: RunnableConfig):
    """
    运行交互式计划工作流

    参数:
        query: 用户查询
    返回:
        最终状态
    """
    print(f"\n===== 开始规划工作流 =====")
    print(f"查询: {query}")

    # 创建全局图实例
    graph = create_planning_graph()

    # 初始化工作流
    initial_state = PlanState(
        original_question=query,
        optimized_question="",
        optimized_questions=[],
        context="",
        generated_plan="",
        refine_plan="",
        refine_result="",
        user_feedback=None,
        csv_path=None,
        csv_validation_passed=False,
        user_choice=None,
        just_refined=False,
        metabcr_result=None,
    )

    # 配置
    config = get_antibody_runnable_config(uuid.uuid4())

    # 先运行到第一次中断
    _ = list(graph.stream(initial_state, config))

    # 持续运行直到工作流完成
    while True:
        # 检查工作流状态
        state = graph.get_state(config)

        # 如果工作流已完成（没有下一个节点），退出循环
        if not state.next:
            break

        # 如果有下一个节点，说明需要继续执行（可能被中断）
        print(state.interrupts[0].value)
        user_input = input("> ").strip()

        # 使用用户输入恢复工作流
        _ = list(graph.stream(Command(resume=user_input), config))

    # 获取最终状态
    final_state = graph.get_state(config).values

    # 修正访问方式：使用字典访问而不是属性访问
    if isinstance(final_state, dict):
        final_plan = final_state.get("generated_plan", "")
    else:
        final_plan = getattr(final_state, "generated_plan", "")

    print(f"\n===== 工作流执行完成 =====\n{final_plan}")
    return final_plan


if __name__ == "__main__":
    query = input("Please enter your research question: ")
    from usecases._debug import get_debug_runnable_config

    rc = get_debug_runnable_config()
    run_planning_workflow(query, rc)
