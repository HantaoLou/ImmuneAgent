import uuid
from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

from common.factory import get_reasoning_model
from common.prompts import AntibodyPrompt
from schema.common_schemas import ToolSelection
from usecases.antibody.antibody_config import get_antibody_runnable_config
from usecases.antibody.state.state import ExecuteState
from usecases.antibody.tool.planning_tools import analysis_tool_node


def extract_tool_node(state: ExecuteState, config: RunnableConfig):
    """分析当前步骤，选择合适的工具"""
    plan = state.generated_plan
    model = get_reasoning_model(config)
    prompt = ChatPromptTemplate.from_template(AntibodyPrompt.SELECT_TOOLS__PROMPT)

    structured_model = model.with_structured_output(ToolSelection)
    runnable = prompt | structured_model
    response = runnable.invoke({"plan": plan})
    tool_list = response.tools
    print("选择的工具:")
    for i, tool in enumerate(tool_list, 1):
        print(f"  {i}. {tool}")
    state.tool_list = tool_list
    return state


def input_instruction_node(state: ExecuteState, config: RunnableConfig):
    """获取用户指令的节点"""
    print("\n=== 获取用户指令 ===")

    # 显示分析阶段推荐的工具
    recommended_tools = state.tool_list
    if recommended_tools:
        print(f"\n分析阶段推荐的工具: {', '.join(recommended_tools)}")

    execution_count = state.execution_count

    # 构建用户指令描述，包含推荐工具信息
    description_parts = [
        "请输入您的指令，例如：",
        "- '请调用MetaBCR工具分析/path/to/file.csv'",
        "- '使用FDG工具处理数据文件/path/to/data.xlsx'",
        "- '运行AlphaFold3分析/path/to/input.csv'",
        "- 输入 'exit' 或 'quit' 结束工具调用",
    ]

    if recommended_tools:
        description_parts.insert(
            1, f"\n💡 分析阶段推荐工具: {', '.join(recommended_tools)}\n"
        )

    # 人机交互：获取用户指令
    user_instruction = interrupt(
        {
            "task": "智能工具调用",
            "description": "\n".join(description_parts),
            "available_tools": ["MetaBCR分析工具", "FDG分析工具", "AlphaFold3分析工具"],
            "recommended_tools": recommended_tools,
            "execution_count": execution_count,
        }
    )
    state.current_instruction = user_instruction.strip() if user_instruction else ""
    return state


def execute_tool_node(state: ExecuteState, config: RunnableConfig):
    """使用 create_react_agent 执行工具的智能节点"""
    print("\n=== 智能工具执行节点 ===")

    user_instruction = state.current_instruction

    if not user_instruction:
        print("\n输入为空，跳过执行")
        return state

    print(f"\n[执行指令] {user_instruction}")

    # 获取推理模型
    reasoning_model = get_reasoning_model(config)

    # 创建 ReAct Agent
    react_agent = create_react_agent(reasoning_model, analysis_tool_node)

    # 初始化工具执行结果
    tool_results = state.tool_results.copy()
    execution_count = state.execution_count

    try:
        # 使用 ReAct Agent 处理用户指令
        result = react_agent.invoke(
            {"messages": [HumanMessage(content=user_instruction)]}
        )

        # 提取工具执行结果和AI的完整回复
        tool_output = ""
        tool_calls_made = []
        ai_final_response = ""

        if isinstance(result, dict) and "messages" in result:
            for message in result["messages"]:
                if hasattr(message, "content") and hasattr(message, "name"):
                    # 这是工具调用的结果
                    tool_name = getattr(message, "name", "unknown")
                    tool_result = message.content
                    tool_calls_made.append(f"{tool_name}: {tool_result}")
                    print(f"\n[工具执行结果] {tool_name}: {tool_result}")
                elif hasattr(message, "content") and not hasattr(message, "name"):
                    # 这是 AI 的回复（可能是最终总结）
                    if message.content.strip():
                        ai_final_response = message.content
                        print(f"\n[AI回复] {ai_final_response}")

            # 组合完整的输出
            if tool_calls_made and ai_final_response:
                tool_output = f"工具调用: {'; '.join(tool_calls_made)}\n\nAI总结: {ai_final_response}"
            elif tool_calls_made:
                tool_output = f"工具调用: {'; '.join(tool_calls_made)}"
            elif ai_final_response:
                tool_output = ai_final_response
            else:
                tool_output = "ReAct Agent执行完成，但未获取到具体输出"

        # 记录执行结果
        execution_count += 1
        tool_results[execution_count] = {
            "instruction": user_instruction,
            "output": tool_output,
            "timestamp": str(uuid.uuid4())[:8],
        }

        print(f"\n[执行完成] 第 {execution_count} 次工具调用")

    except Exception as e:
        error_msg = f"工具执行错误: {str(e)}"
        print(f"\n[错误] {error_msg}")

        execution_count += 1
        tool_results[execution_count] = {
            "instruction": user_instruction,
            "error": error_msg,
            "timestamp": str(uuid.uuid4())[:8],
        }

    state.tool_results = tool_results
    state.execution_count = execution_count
    state.current_instruction = ""
    return state


def should_continue_execution(state: ExecuteState) -> str:
    """决定是否继续执行的条件边函数"""
    current_instruction = state.current_instruction

    # 检查是否是退出指令
    if current_instruction.lower() in ["exit", "quit", "退出", "结束"]:
        print("\n用户选择结束工具调用")
        return "end"

    # 如果有指令且不为空，继续执行
    if current_instruction and current_instruction.strip():
        return "execute"

    # 否则继续获取指令
    return "get_instruction"


def show_summary_node(state: ExecuteState, config: RunnableConfig):
    """显示执行汇总的节点"""
    print("\n=== 执行汇总 ===")

    tool_results = state.tool_results
    execution_count = state.execution_count

    if not tool_results:
        print("没有执行任何工具")
    else:
        print(f"总共执行了 {execution_count} 次工具调用:")
        for idx, result in tool_results.items():
            print(f"\n执行 {idx}:")
            print(f"  指令: {result.get('instruction', '未知')}")
            if "error" in result:
                print(f"  错误: {result['error']}")
            else:
                output = result.get("output", "无输出")
                if len(output) > 100:
                    print(f"  结果: {output[:100]}...")
                else:
                    print(f"  结果: {output}")
    state.execution_complete = True
    return state


def create_execute_agent_workflow():
    """创建基于 create_react_agent 的执行代理工作流"""
    workflow = StateGraph(ExecuteState)

    # 添加节点
    workflow.add_node("extract_tool_node", extract_tool_node)
    workflow.add_node("input_instruction_node", input_instruction_node)
    workflow.add_node("execute_tool_node", execute_tool_node)
    workflow.add_node("show_summary_node", show_summary_node)

    # 设置入口点 - 从分析节点开始
    workflow.set_entry_point("extract_tool_node")

    # 添加边：analyze -> get_instruction
    workflow.add_edge("extract_tool_node", "input_instruction_node")

    # 添加条件边：根据用户指令决定下一步
    workflow.add_conditional_edges(
        "input_instruction_node",
        should_continue_execution,
        {
            "execute": "execute_tool_node",
            "end": "show_summary_node",
            "input_instruction_node": "input_instruction_node",
        },
    )

    # 执行完工具后，回到获取指令节点
    workflow.add_edge("execute_tool_node", "input_instruction_node")

    # 显示汇总后结束
    workflow.add_edge("show_summary_node", END)

    # 编译工作流
    graph = workflow.compile(checkpointer=MemorySaver())
    try:
        print("\n===== 工作流程图代码 =====")
        print("可以将以下代码复制到任意Mermaid编辑器中查看图形:")
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"生成Mermaid代码时出错: {str(e)}")
    return graph


def run_execute_agent(plan: List[str] = None, config: RunnableConfig = None):
    """
    运行基于 create_react_agent 的执行代理工作流
    参数:
        plan: 执行计划列表（可选，主要用于上下文参考）
    返回:
        最终状态
    """

    # 配置
    config = get_antibody_runnable_config(uuid.uuid4())

    # 创建初始状态对象
    initial_state = ExecuteState(
        generated_plan=plan,
        tool_list=[],
        tool_results={},
        execution_count=0,
        execution_complete=False,
        current_instruction="",
    )

    execute_graph = create_execute_agent_workflow()

    try:
        # 运行工作流，处理所有中断
        _ = list(execute_graph.stream(initial_state, config))

        while True:
            state = execute_graph.get_state(config)
            if not state.next:
                break

            # 处理中断
            if state.interrupts:
                interrupt_info = state.interrupts[0].value
                if isinstance(interrupt_info, dict):
                    print(f"\n=== {interrupt_info.get('task', '任务')} ===")
                    print(interrupt_info.get("description", ""))
                    if "available_tools" in interrupt_info:
                        print(
                            f"\n可用工具: {', '.join(interrupt_info['available_tools'])}"
                        )
                    if "execution_count" in interrupt_info:
                        print(f"已执行次数: {interrupt_info['execution_count']}")
                else:
                    print(f"\n{interrupt_info}")

                user_input = input("> ").strip()
                _ = list(execute_graph.stream(Command(resume=user_input), config))
            else:
                break

        # 获取最终状态
        final_state_snapshot = execute_graph.get_state(config)
        final_state = ExecuteState(**final_state_snapshot.values)

        # 显示最终结果
        if final_state.tool_results:
            print("\n===== 最终执行结果汇总 =====")
            for idx, result in final_state.tool_results.items():
                print(f"\n执行 {idx}:")
                print(f"  指令: {result.get('instruction', '未知')}")
                if "error" in result:
                    print(f"  错误: {result['error']}")
                else:
                    print(f"  结果: {result.get('output', '无输出')[:200]}...")

        return final_state

    except Exception as e:
        error_msg = f"执行过程中出错: {str(e)}"
        print(f"[错误] {error_msg}")
        return ExecuteState(
            generated_plan="错误状态",
            tool_list=[],
            execution_complete=True,
            current_instruction=error_msg,
        )
