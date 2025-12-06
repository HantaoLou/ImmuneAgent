import json
import operator

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from common.factory import get_mcp_client, get_reasoning_model
from usecases.execute.graph.single_tool_executor import single_tool_executor
from usecases.execute.interrupts import (
    ArgConfirmationInterrupt,
    ArgConfirmationResult,
    ConfirmToolCallInterrupt,
    ConfirmToolCallResult,
    ToolCallResult,
)
from usecases.execute.prompts import TASK_ANALYSE_PROMPT

LOCAL_TOOLS = []


class ExecutorState(TypedDict):
    """执行器状态"""

    tasks: list[str]
    idx: int
    task_outputs: Annotated[list[dict], ..., operator.add]
    finished: bool
    tools: list[ToolCall]
    local_tools: list[BaseTool]
    failure_count: int  # 当前任务的连续失败次数


def _current_task(state: ExecutorState) -> str:
    return state["tasks"][state["idx"]]


def init_state(tasks: list[str]) -> ExecutorState:
    return {
        "tasks": tasks,
        "idx": 0,
        "task_outputs": [],
        "finished": False,
        "tools": [],
        "local_tools": [],
        "failure_count": 0,
    }


async def fetch_task_step(state: ExecutorState):
    if state["idx"] >= len(state["tasks"]):
        return {"finished": True}
    return state


async def get_all_tools(config: RunnableConfig):
    """
    获取所有可用的工具，优雅处理 MCP 服务器连接失败。
    
    如果某些 MCP 服务器不可用，会跳过它们并继续使用其他可用的工具。
    """
    try:
        client = await get_mcp_client(config)
        tools = await client.get_tools()
        return tools + LOCAL_TOOLS
    except Exception as e:
        # 如果批量获取失败，尝试逐个服务器连接
        print(f"⚠️ 批量获取工具失败: {str(e)}")
        print("🔄 尝试逐个服务器连接...")
        
        from common.factory import get_all_mcp_servers, get_mcp_client
        from langchain_core.runnables.config import RunnableConfig
        
        all_tools = []
        all_servers = get_all_mcp_servers()
        
        # 获取配置中指定的服务器ID
        mcp_config = config.get("configurable", {}).get("mcp_config", {})
        service_ids = mcp_config.get("service_ids", [])
        
        # 如果 service_ids 为空或未指定，使用配置中的默认服务器列表
        # 注意：不自动连接所有服务器，避免连接不必要的服务器
        if not service_ids:
            # 如果没有指定，使用常见的服务器列表
            # 可以根据实际需求调整这个默认列表
            service_ids = ["metabcr", "airr", "anarci"]  # 默认只连接最常用的服务器
            print(f"⚠️ 未指定 service_ids，使用默认列表: {service_ids}")
        
        print(f"📋 尝试连接的服务器: {service_ids}")
        
        # 逐个尝试连接每个服务器
        for server_id in service_ids:
            if server_id not in all_servers:
                print(f"⚠️ 服务器 {server_id} 不在配置中，跳过")
                continue
                
            try:
                # 为单个服务器创建配置
                single_server_config = RunnableConfig(
                    configurable={
                        **config.get("configurable", {}),
                        "mcp_config": {"service_ids": [server_id]}
                    }
                )
                
                single_client = await get_mcp_client(single_server_config)
                server_tools = await single_client.get_tools()
                
                if server_tools:
                    all_tools.extend(server_tools)
                    print(f"✅ 成功连接 {server_id} 服务器，获取 {len(server_tools)} 个工具")
                else:
                    print(f"⚠️ {server_id} 服务器连接成功但没有可用工具")
                    
            except Exception as server_error:
                # 单个服务器连接失败，记录但继续
                error_msg = str(server_error)
                if "503" in error_msg or "Service Unavailable" in error_msg:
                    print(f"⚠️ {server_id} 服务器不可用 (503)，跳过")
                elif "Connection" in error_msg or "timeout" in error_msg.lower():
                    print(f"⚠️ {server_id} 服务器连接超时，跳过")
                else:
                    print(f"⚠️ {server_id} 服务器连接失败: {error_msg[:100]}...")
                continue
        
        if not all_tools:
            print("❌ 所有服务器连接失败，无法获取任何工具")
            # 返回空列表而不是抛出异常，让系统能够继续运行
            return LOCAL_TOOLS
        
        print(f"✅ 成功获取 {len(all_tools)} 个工具（来自 {len([s for s in service_ids if s in all_servers])} 个服务器）")
        return all_tools + LOCAL_TOOLS


def _format_task_outputs(task_outputs: list[dict]):
    output_docs = ""
    for output in task_outputs:
        for tool_name, tool_out in output.items():
            output_docs += f"tool_name: {tool_name}\n"
            output_docs += f"tool_output: {tool_out}\n"
    return output_docs


async def analyse_task_step(state: ExecutorState, config: RunnableConfig):
    """分析"""
    tools = await get_all_tools(config)

    llm = get_reasoning_model(config)
    chain = PromptTemplate.from_template(TASK_ANALYSE_PROMPT) | llm.bind_tools(tools)
    ret: AIMessage = chain.invoke(
        {
            "task": _current_task(state),
            # 把之前调用过工具的输出作为上下文，供后续工具使用
            # TODO: 如果需要其他来源的上下文，也在这里传入
            "context": f"Previous tool results:\n {_format_task_outputs(state['task_outputs'])}",
        }
    )
    return {**state, "tools": ret.tool_calls}


def _find_tool(tools: list[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


async def execute_tools_step(state: ExecutorState, config: RunnableConfig):
    """执行所有工具（通过调用基于生成器的单个工具执行器）"""

    tool_results = {}
    tools = await get_all_tools(config)
    has_failure = False

    # 检查是否达到最大失败次数
    MAX_FAILURES = 3
    if state["failure_count"] >= MAX_FAILURES:
        print(
            f"[execute_tools_step] Task failed {MAX_FAILURES} times, skipping to next task"
        )
        return {
            **state,
            "idx": state["idx"] + 1,
            "task_outputs": [{"error": "Task skipped due to repeated failures"}],
            "failure_count": 0,  # 重置失败计数
        }

    # 直接对 tools 进行循环，使用生成器处理中断
    for tool_call in state["tools"]:
        print(f"[execute_tools_step] Processing tool call: {tool_call['name']}")
        tool_name = tool_call["name"]
        _tool = _find_tool(tools, tool_name)
        if not _tool:
            print(f"[execute_tools_step] Tool not found: {tool_name}")
            tool_results[tool_name] = {"error": f"Tool '{tool_name}' not found"}
            has_failure = True
            continue

        try:
            # 使用生成器执行单个工具
            executor = single_tool_executor(tool_call, _tool, config)

            # 启动生成器
            interrupt_obj = await executor.__anext__()

            while True:
                # 最终结果
                if isinstance(interrupt_obj, ToolCallResult):
                    if interrupt_obj.success:
                        tool_results[tool_name] = interrupt_obj.result
                        print(f"[execute_tools_step] Tool call succeeded: {tool_name}")
                    else:
                        print(
                            f"[execute_tools_step] Tool call failed: {tool_name} - {interrupt_obj.msg}"
                        )
                        tool_results[tool_name] = {"error": interrupt_obj.msg}
                        has_failure = True
                    break
                else:
                    response = stdio_interrupt_handler(interrupt_obj)
                    interrupt_obj = await executor.asend(response)
        except Exception as e:
            print(
                f"[execute_tools_step] Exception during tool execution: {tool_name} - {str(e)}"
            )
            tool_results[tool_name] = {"error": f"Exception: {str(e)}"}
            has_failure = True

    # 根据执行结果更新状态 - 关键修复：始终递增任务索引以防止无限循环
    next_idx = state["idx"] + 1

    if has_failure:
        # 有失败，增加失败计数，但仍然递增任务索引以继续下一个任务
        print(
            f"[execute_tools_step] Task {state['idx']} had failures, moving to next task"
        )
        return {
            **state,
            "idx": next_idx,
            "task_outputs": [tool_results],
            "failure_count": 0,  # 重置失败计数，因为我们已经移动到下一个任务
        }
    else:
        # 全部成功，递增任务索引，重置失败计数
        print(
            f"[execute_tools_step] Task {state['idx']} completed successfully, moving to next task"
        )
        return {
            **state,
            "idx": next_idx,
            "task_outputs": [tool_results],
            "failure_count": 0,
        }


def router(state: ExecutorState) -> str:
    """
    结束还是获取下一个 task
    """
    current_idx = state.get("idx", 0)
    total_tasks = len(state.get("tasks", []))

    # 日志输出当前状态
    print(f"[router] Current task index: {current_idx}, Total tasks: {total_tasks}")

    if state.get("finished") or current_idx >= total_tasks:
        print("[router] Workflow finished")
        return END

    print(f"[router] Continuing to next task: {current_idx}")
    return analyse_task_step.__name__


def _link(graph: StateGraph, step_from, step_to):
    if not isinstance(step_from, str):
        step_from = step_from.__name__
    if not isinstance(step_to, str):
        step_to = step_to.__name__
    graph.add_edge(step_from, step_to)


def build_graph() -> StateGraph:
    ret = StateGraph(ExecutorState)
    (
        ret.add_node(fetch_task_step)
        .add_node(analyse_task_step)
        .add_node(execute_tools_step)
    )
    ret.add_conditional_edges(
        fetch_task_step.__name__,
        router,
        {END: END, analyse_task_step.__name__: analyse_task_step.__name__},
    )
    _link(ret, START, fetch_task_step)
    _link(ret, analyse_task_step, execute_tools_step)
    _link(ret, execute_tools_step, fetch_task_step)

    return ret


def stdio_interrupt_handler(interrupt_obj):
    """处理标准输入输出中断"""
    if isinstance(interrupt_obj, ConfirmToolCallInterrupt):
        response_msg = input(f"确认调用工具 {interrupt_obj.tool_name}? (yes/no): ")
        return ConfirmToolCallResult(msg=response_msg)
    elif isinstance(interrupt_obj, ArgConfirmationInterrupt):
        # 显示当前值和默认值信息
        value_display = (
            interrupt_obj.provided_value
            if interrupt_obj.provided_value is not None
            else "<未提供>"
        )
        default_display = (
            f" [默认: {interrupt_obj.default_value}]"
            if interrupt_obj.default_value is not None
            else ""
        )
        type_display = f" ({interrupt_obj.arg_type})" if interrupt_obj.arg_type else ""

        confirmed_args_input = input(
            f"请输入确认的参数 ({interrupt_obj.arg_name}: {value_display}{type_display}{default_display}): "
        ).strip()

        # 如果用户输入为空且有默认值，使用默认值
        if not confirmed_args_input and interrupt_obj.default_value is not None:
            converted_value = interrupt_obj.default_value
            print(f"使用默认值: {interrupt_obj.default_value}")
        else:
            # 根据参数类型进行转换
            converted_value = confirmed_args_input
            if interrupt_obj.arg_type and confirmed_args_input:
                try:
                    if interrupt_obj.arg_type == "integer":
                        converted_value = int(confirmed_args_input)
                    elif interrupt_obj.arg_type == "number":
                        converted_value = float(confirmed_args_input)
                    elif interrupt_obj.arg_type == "boolean":
                        converted_value = confirmed_args_input.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                            "是",
                        )
                    elif interrupt_obj.arg_type == "array":
                        converted_value = json.loads(confirmed_args_input)
                    elif interrupt_obj.arg_type == "object":
                        converted_value = json.loads(confirmed_args_input)
                    # string 类型保持原样
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"参数类型转换失败: {e}")
                    # 如果转换失败且有默认值，使用默认值
                    if interrupt_obj.default_value is not None:
                        converted_value = interrupt_obj.default_value
                        print(f"使用默认值: {interrupt_obj.default_value}")
                    else:
                        print("使用原始字符串值")
                        converted_value = confirmed_args_input

        return ArgConfirmationResult(
            confirmed_args={interrupt_obj.arg_name: converted_value}
        )
    else:
        raise ValueError(f"未知的中断类型: {type(interrupt_obj)}")


if __name__ == "__main__":

    @tool(parse_docstring=True)
    def metabcr_data_preprocess():
        """
        perform data preprocess for metabcr, and get final input file
        """
        return {"metabcr_input_file_path": "/a/b/c/d"}

    @tool(parse_docstring=True)
    def metabcr_predict(input_path):
        """
        Perform metabcr predict on BCR seq against antigen

        :param input_path: input path
        """
        return {}

    LOCAL_TOOLS.extend([metabcr_data_preprocess, metabcr_predict])
    s = init_state(
        [
            "pre process data for metabcr",
            "use metabcr to predict the functionality of the BCR against the antigen?",
        ]
    )

    # 会先调用 metabcr_data_preprocess
    # 得到的结果作为 metabcr 的输入

    g = build_graph()
    import asyncio

    from langgraph.checkpoint.memory import InMemorySaver

    from common.runner import GraphRunner
    from usecases._debug import get_debug_runnable_config

    config = get_debug_runnable_config()
    wf = g.compile(checkpointer=InMemorySaver())
    print(wf.get_graph().draw_mermaid())
    runner = GraphRunner(graph=wf)

    async def run():
        ret = await runner.run(s, config)
        # 在循环中 resume 直到 Graph 结束
        return ret

    ret = asyncio.run(run())
