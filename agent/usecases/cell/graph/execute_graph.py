import asyncio
import json
import os
import uuid
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from common.factory import get_reasoning_model
from usecases.cell.cell_config import get_cell_runnable_config
from usecases.cell.state.state import ExecuteState
from usecases.deepagents.tools import hil
from usecases.execute.graph.generic_executor import LOCAL_TOOLS
from usecases.execute.graph.generic_executor import build_graph as build_executor_graph
from usecases.execute.graph.generic_executor import init_state as init_executor_state

# 定义cell专用的本地工具集合
CELL_LOCAL_TOOLS = []


class TaskInfo(BaseModel):
    """任务信息模型"""

    task_id: str = Field(description="任务ID")
    name: str = Field(description="任务名称")
    description: str = Field(description="任务描述")
    tools: list[str] = Field(description="使用的工具")
    inputs: list[str] = Field(description="输入数据")
    outputs: list[str] = Field(description="输出数据")
    parameters: Dict[str, Any] = Field(description="参数设置")


class TaskExtractionResult(BaseModel):
    """任务提取结果模型"""

    tasks: list[TaskInfo] = Field(description="提取的任务列表")


async def task_decomposition_node(
    state: ExecuteState, config: RunnableConfig
) -> ExecuteState:
    """
    任务分解节点 - 将refine_plan分解为具体的执行任务

    该节点接收planning_graph.py中的refine_plan，并将其分解为
    具体的、可执行的任务步骤列表。

    Args:
        state: Cell模块的状态对象，包含refine_plan
        config: 运行时配置

    Returns:
        ExecuteState: 更新后的cell状态，包含分解后的任务列表
    """
    print("[task_decomposition_node] 开始任务分解节点")

    try:
        # 获取refine_plan
        refine_plan = state.refine_plan

        if not refine_plan or refine_plan.strip() == "":
            print("[task_decomposition_node] 没有可分解的计划")
            return state
        from common.prompts import CellPrompt

        # 创建任务分解链
        decomposition_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CellPrompt.SYSTEM_EXTRACTION_TASK_PROMPT),
                ("user", CellPrompt.USER_EXTRACTION_TASK_PROMPT),
            ]
        )
        model = get_reasoning_model(config)
        output_parser = JsonOutputParser(pydantic_object=TaskExtractionResult)
        decomposition_chain = decomposition_prompt | model | output_parser
        # 执行任务分解
        decomposed_tasks = decomposition_chain.invoke({"plan": refine_plan})

        # JsonOutputParser返回的是字典，不是TaskExtractionResult实例
        # 需要从字典中获取tasks列表
        if isinstance(decomposed_tasks, dict) and "tasks" in decomposed_tasks:
            tasks_list = decomposed_tasks["tasks"]
            print(
                f"[task_decomposition_node] 任务分解完成，共提取到 {len(tasks_list)} 个任务"
            )

            # 从任务列表中提取每个任务的description字段
            # 存储到state.decomposed_tasks列表中
            task_descriptions = []
            for task_dict in tasks_list:
                if isinstance(task_dict, dict) and "description" in task_dict:
                    description = task_dict["description"]
                    if description and description.strip():
                        task_descriptions.append(description.strip())
        else:
            print(f"[task_decomposition_node] 错误：无法从分解结果中获取tasks列表")
            task_descriptions = []

        state.decomposed_tasks = task_descriptions
        print(f"[task_decomposition_node] 成功提取 {len(task_descriptions)} 个任务描述")

        return state

    except Exception as e:
        import traceback

        print(f"[task_decomposition_node] 任务分解失败: {e}")
        print(f"[task_decomposition_node] 错误类型: {type(e).__name__}")
        print(f"[task_decomposition_node] 详细堆栈信息:")
        print(traceback.format_exc())
        return state


async def dynamic_execute_node(
    state: ExecuteState, config: RunnableConfig
) -> ExecuteState:
    """
    动态执行节点 - 调用execute agent进行工具执行

    该节点将cell state中的任务转换为execute agent可以理解的格式，
    然后调用execute agent进行动态工具执行，最后将结果映射回cell state。

    Args:
        state: Cell模块的状态对象
        config: 运行时配置

    Returns:
        ExecuteState: 更新后的cell状态
    """
    print("[dynamic_execute_node] 开始动态执行节点")

    # 复用 generic_executor 中的 get_all_tools 逻辑
    from usecases.execute.graph.generic_executor import get_all_tools

    tools = await get_all_tools(config)

    # 保存原始工具列表
    original_tools = LOCAL_TOOLS.copy()

    try:
        # 1. 从state中提取任务列表
        tasks = state.decomposed_tasks

        if not tasks:
            print("[dynamic_execute_node] 没有需要执行的任务")
            return state

        print(f"[dynamic_execute_node] 提取到任务: {tasks}")

        # 2. 初始化execute agent状态
        # executor_state = init_executor_state(tasks)

        # 3. 注入cell专用工具（使用hil包装）
        hil_wrapped_tools = [hil(tool) for tool in tools] if tools else []
        LOCAL_TOOLS.extend(hil_wrapped_tools)

        print(
            f"[dynamic_execute_node] 注入了 {len(hil_wrapped_tools)} 个hil包装的cell专用工具"
        )
        import json

        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        from common.runner import GraphRunner

        model = get_reasoning_model(config)
        # 4. 构建并运行execute graph
        # 创建 ReAct Agent with checkpointer for HIL support
        react_agent = create_react_agent(model, LOCAL_TOOLS, checkpointer=MemorySaver())

        # 用于收集执行结果的容器
        task_results = []
        task_messages = []
        all_tool_results = []  # 收集所有任务的工具执行结果
        current_task_tool_results = []  # 当前任务的工具执行结果
        summary_printed = False  # 防止重复打印汇总信息的标志

        # 使用 GraphRunner 来支持 HIL 交互
        def collect_values(values):
            """收集每个节点的状态更新值"""
            print(f"[DEBUG] 收到values: {values}")
            task_results.append(values)

            # 检查是否包含工具执行结果
            if "messages" in values:
                for message in values["messages"]:
                    # 检查是否为ToolMessage（工具执行结果）
                    if hasattr(message, "__class__") and "ToolMessage" in str(
                        message.__class__
                    ):
                        content = getattr(message, "content", "")
                        tool_name = getattr(message, "name", "unknown")

                        # 过滤掉HIL拒绝消息，只收集真正的工具执行结果
                        is_hil_rejection = isinstance(content, str) and (
                            "被人类拒绝" in content
                            or "user reject" in content.lower()
                            or content.startswith("工具")
                            and "拒绝" in content
                        )

                        if not is_hil_rejection:
                            # 检查是否已经收集过这个工具结果（基于message_id去重）
                            message_id = getattr(message, "id", "unknown")
                            tool_call_id = getattr(message, "tool_call_id", "unknown")

                            # 检查是否已存在相同的工具结果
                            already_exists = any(
                                result["message_id"] == message_id
                                and result["tool_call_id"] == tool_call_id
                                for result in all_tool_results
                            )

                            if not already_exists:
                                tool_result = {
                                    "task_index": i + 1,  # 添加任务索引
                                    "tool_name": tool_name,
                                    "tool_call_id": tool_call_id,
                                    "content": content,
                                    "message_id": message_id,
                                    "timestamp": getattr(message, "timestamp", None),
                                    "is_successful": True,  # 标记为成功执行的工具结果
                                }
                                current_task_tool_results.append(tool_result)
                                all_tool_results.append(tool_result)  # 同时添加到总列表
                                # 显示完整的工具执行结果内容
                                print(
                                    f"[DEBUG] 提取到工具执行结果: 任务{i + 1} - {tool_result['tool_name']} -> 完整内容: {content}"
                                )
                            else:
                                print(
                                    f"[DEBUG] 跳过重复的工具结果: 任务{i + 1} - {tool_name} (message_id: {message_id})"
                                )
                        else:
                            # 记录HIL拒绝信息但不作为工具执行结果
                            print(
                                f"[DEBUG] 跳过HIL拒绝消息: 任务{i + 1} - {tool_name} -> {content[:100]}..."
                            )

        def collect_messages(message):
            """收集AI生成的消息"""
            print(message.content, end="")
            task_messages.append(message)

        runner = (
            GraphRunner(react_agent)
            .with_message_handler(collect_messages)
            .with_value_handler(collect_values)
        )

        for i, task in enumerate(tasks):
            print(f"[dynamic_execute_node] 执行任务 {i + 1}/{len(tasks)}: {task}")

            # 清空之前任务的结果（但保留all_tool_results）
            task_results.clear()
            task_messages.clear()
            current_task_tool_results.clear()  # 只清空当前任务的工具结果

            # 构建初始状态
            initial_state = {"messages": [HumanMessage(content=task)]}

            # 使用异步流处理来支持 HIL 交互
            async def run_task_with_hil():
                thread_id = f"task_{i + 1}_{uuid.uuid4().hex[:8]}"
                task_config = {"configurable": {"thread_id": thread_id}}

                ret = await runner.run(initial_state, task_config)
                while ret is not None:
                    print(f"\n[HIL] 工具调用请求: {ret}")
                    print("请选择操作:")
                    print("  y/yes/accept - 批准执行")
                    print("  n/no/reject - 拒绝执行")
                    print("  edit - 修改参数后执行")

                    user_input = input("您的选择: ").strip().lower()

                    if user_input in ["y", "yes", "accept"]:
                        # 批准执行 - 需要传递原始参数
                        # 从ret中提取原始工具参数
                        original_args = None
                        try:
                            # ret是Interrupt对象，参数信息在ret.value字符串中
                            if hasattr(ret, "value") and ret.value:
                                # 从interrupt消息中解析参数: "call tool: tool_name with {params}"
                                import re

                                match = re.search(r"with (.+)$", str(ret.value))
                                if match:
                                    params_str = match.group(1)
                                    # 使用ast.literal_eval安全解析参数字典
                                    import ast

                                    original_args = ast.literal_eval(params_str)
                                    print(f"[DEBUG] 成功解析参数: {original_args}")
                                    print(f"[DEBUG] 参数数量: {len(original_args)}")
                                else:
                                    print(f"[DEBUG] 无法从消息中提取参数: {ret.value}")
                            else:
                                print(f"[DEBUG] ret对象没有value属性或value为空: {ret}")
                        except Exception as e:
                            print(f"[DEBUG] 参数解析失败: {e}")
                            import traceback

                            traceback.print_exc()

                        if original_args is not None:
                            response = json.dumps(
                                {"accept": True, "args": original_args}
                            )
                        else:
                            response = json.dumps({"accept": True})
                    elif user_input in ["n", "no", "reject"]:
                        # 拒绝执行
                        response = json.dumps({"accept": False})
                    elif user_input == "edit":
                        # 修改参数
                        print("请输入修改后的参数 (JSON格式):")
                        try:
                            new_params = input("新参数: ")
                            params = json.loads(new_params)
                            response = json.dumps({"accept": True, "args": params})
                        except json.JSONDecodeError:
                            print("JSON格式错误，拒绝执行")
                            response = json.dumps({"accept": False})
                            import traceback

                            traceback.print_exc()
                    else:
                        print("无效输入，拒绝执行")
                        response = json.dumps({"accept": False})
                        import traceback

                        traceback.print_exc()

                    ret = await runner.resume(response, task_config)

                # 返回收集到的结果
                return {
                    "interrupt_result": ret,  # 最后的中断结果（通常为None表示完成）
                    "values": task_results.copy(),  # 所有状态更新
                    "messages": task_messages.copy(),  # 所有AI消息
                    "tool_results": current_task_tool_results.copy(),  # 当前任务的工具执行结果
                    "final_state": task_results[-1]
                    if task_results
                    else None,  # 最终状态
                }

            # 执行任务
            result = await run_task_with_hil()
            print(f"\n[dynamic_execute_node] 任务 {i + 1} 执行完成")
            print(f"[DEBUG] 任务结果: {result}")
            print(f"[DEBUG] 收集到 {len(result['values'])} 个状态更新")
            print(f"[DEBUG] 收集到 {len(result['messages'])} 条消息")
            print(f"[DEBUG] 收集到 {len(result['tool_results'])} 个工具执行结果")

            # 打印当前任务的工具执行结果摘要（完整内容）
            for idx, tool_result in enumerate(result["tool_results"]):
                print(
                    f"[DEBUG] 任务{i + 1}工具结果 {idx + 1}: {tool_result['tool_name']} -> 完整内容: {tool_result['content']}"
                )

            if result["final_state"]:
                print(f"[DEBUG] 最终状态包含的键: {list(result['final_state'].keys())}")

        # 6. 将ExecutorState的结果映射回原始ExecuteState对象
        # 这里我们需要根据执行结果更新原始state的相关字段
        updated_state = state.model_copy()  # 创建state的副本

        # 将所有工具执行结果存储到state中（如果state有相应字段的话）
        # 这里可以根据实际需要添加字段来存储工具执行结果
        if hasattr(updated_state, "tool_execution_results"):
            updated_state.tool_execution_results = all_tool_results
            print(f"[DEBUG] 存储了 {len(all_tool_results)} 个工具执行结果到state中")

        # 打印所有任务的工具执行结果汇总（避免重复，只显示汇总信息）
        if not summary_printed:
            print(f"\n[DEBUG] === 所有任务工具执行结果汇总 ===")
            print(f"[DEBUG] 总共收集到 {len(all_tool_results)} 个工具执行结果")
            # 按任务分组显示，避免重复
            task_groups = {}
            for tool_result in all_tool_results:
                task_idx = tool_result["task_index"]
                if task_idx not in task_groups:
                    task_groups[task_idx] = []
                task_groups[task_idx].append(tool_result)

            for task_idx, results in task_groups.items():
                print(f"[DEBUG] 任务{task_idx}: 共{len(results)}个工具调用结果")
                for idx, tool_result in enumerate(results):
                    print(
                        f"[DEBUG]   工具{idx + 1}: {tool_result['tool_name']} -> 完整内容: {tool_result['content']}"
                    )
            summary_printed = True  # 标记已打印汇总信息

        return updated_state

    except Exception as e:
        print(f"[dynamic_execute_node] 执行过程中发生错误: {str(e)}")
        print("[dynamic_execute_node] === 详细错误堆栈信息 ===")
        import traceback

        traceback.print_exc()
        return state
    finally:
        # 7. 恢复原始工具列表
        LOCAL_TOOLS.clear()
        LOCAL_TOOLS.extend(original_tools)
        print("[dynamic_execute_node] 已恢复原始工具列表")


def create_cell_execute_graph() -> StateGraph:
    """
    创建包含任务分解和动态执行节点的cell图

    该函数创建一个完整的执行流程：
    1. 任务分解节点：将refine_plan分解为具体任务
    2. 动态执行节点：执行分解后的任务

    Returns:
        StateGraph: 配置好的状态图
    """
    # 创建主图
    graph = StateGraph(ExecuteState)

    # 添加任务分解节点（第一个节点）
    graph.add_node("task_decomposition", task_decomposition_node)

    # 添加动态执行节点
    graph.add_node("dynamic_execute", dynamic_execute_node)

    # 设置图的入口点为任务分解节点
    graph.set_entry_point("task_decomposition")

    # 设置节点之间的连接
    graph.add_edge("task_decomposition", "dynamic_execute")

    # 设置结束点
    graph.add_edge("dynamic_execute", END)

    return graph


async def run_execute_graph(user_question, config: RunnableConfig = None):
    """
    运行执行图的流程函数
    参考planning_graph.py中run_planning_graph函数的实现

    Args:
        user_question: 用户问题
        config: 运行配置（可选）
    """
    # 创建执行图
    graph = create_cell_execute_graph()

    # 编译图以支持流式执行和状态管理
    app = graph.compile(checkpointer=MemorySaver())

    # 初始化执行状态
    initial_state = ExecuteState(
        refine_plan=user_question,  # 将用户问题作为初始的精化计划
        decomposed_tasks=[],  # 初始化空的任务列表
    )

    # 配置工作目录和运行配置
    if config is None:
        _uuid = uuid.uuid4()
        config = get_cell_runnable_config(_uuid)

    print(f"=== 开始执行图流程 ===")
    print(f"用户问题: {user_question}")

    try:
        # 先运行到第一次中断，支持subgraph
        async for _ in app.astream(initial_state, config, subgraphs=True):
            pass

        # 持续运行直到工作流完成
        while True:
            # 检查工作流状态
            state = app.get_state(config)

            # 如果工作流已完成（没有下一个节点），退出循环
            if not state.next:
                break

            # 如果有中断需要处理
            if state.interrupts:
                print(state.interrupts[0].value)
                user_input = input("> ").strip()

                # 使用用户输入恢复工作流，支持subgraph
                async for _ in app.astream(
                    Command(resume=user_input), config, subgraphs=True
                ):
                    pass
            else:
                # 如果没有中断但有下一个节点，继续执行
                async for _ in app.astream(None, config, subgraphs=True):
                    pass

        # 获取最终状态
        final_state = app.get_state(config).values
        # 移除重复的完成信息打印，由main函数统一处理

        return final_state

    except Exception as e:
        print(f"\n执行图流程出现错误: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":

    def main():
        """主函数：运行执行图流程测试"""
        print("=== execute_graph.py 流程测试程序 ===")
        print("正在加载工具和配置...")
        from test_constant import TestConstant

        # 测试用户问题
        test_question = TestConstant.TAST

        try:
            # 运行执行图流程
            result = asyncio.run(run_execute_graph(test_question))

            if result:
                print("\n=== 流程测试完成 ===")
                print("执行图流程成功完成")
            else:
                print("\n=== 流程测试失败 ===")

        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except Exception as e:
            print(f"\n测试失败: {str(e)}")
            print("\n=== 详细错误堆栈信息 ===")
            import traceback

            traceback.print_exc()
            print("=== 错误堆栈信息结束 ===")

    # 运行主函数
    main()
