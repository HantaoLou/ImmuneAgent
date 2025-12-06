from typing import Any, AsyncGenerator, Union

from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from usecases.execute.interrupts import (
    ArgConfirmationInterrupt,
    ArgConfirmationResult,
    ConfirmToolCallInterrupt,
    ConfirmToolCallResult,
    ToolCallResult,
)


def _is_positive(msg: str) -> bool:
    """判断用户输入是否为肯定"""
    return msg in {"yes", "y", "是", "对", "Y", "Yes", "YES", "OK", "ok", ""}


async def single_tool_executor(
    tool_call: ToolCall, tool: BaseTool, config: RunnableConfig = None
) -> AsyncGenerator[
    Union[ArgConfirmationInterrupt, ConfirmToolCallInterrupt, ToolCallResult],
    Union[ArgConfirmationResult, ConfirmToolCallResult],
]:
    """基于异步生成器的单个工具执行器"""

    # 步骤1: 确认工具参数
    print(f"[confirm_single_tool_args_step] Processing tool call: {tool_call['name']}")

    provided_args = tool_call["args"].copy()
    declared_args = tool.args if hasattr(tool, "args") else {}

    # 过滤只保留工具声明的参数
    provided_args = {k: v for k, v in provided_args.items() if k in declared_args}

    for arg_name, arg_schema in declared_args.items():
        provided_value = provided_args.get(arg_name, None)

        # 获取参数类型和默认值
        arg_type = arg_schema.get("type", "string")
        default_value = arg_schema.get("default", None)

        # 确保 provided_value 是字符串或 None
        if provided_value is not None and not isinstance(provided_value, str):
            provided_value = str(provided_value)

        # 创建参数确认中断
        arg_interrupt = ArgConfirmationInterrupt(
            tool_name=tool.name,
            arg_name=arg_name,
            provided_value=provided_value,
            arg_type=arg_type,
            default_value=default_value,
        )
        # 等待用户确认参数
        confirmed_arg: ArgConfirmationResult = yield arg_interrupt
        provided_args.update(confirmed_arg.confirmed_args)

    # 创建执行确认中断
    exec_interrupt = ConfirmToolCallInterrupt(
        tool_name=tool.name,
        args=provided_args,
    )

    # 等待用户确认执行
    exec_response: ConfirmToolCallResult = yield exec_interrupt

    execution_approved = _is_positive(exec_response.msg)

    # 步骤3: 执行工具
    if not execution_approved:
        yield ToolCallResult(
            success=False,
            msg="用户取消了工具执行",
            result=None,
        )
    else:
        print(
            f"[execute_single_tool_step] Executing tool: {tool_call['name']} with args: {provided_args}"
        )
        result = await tool.ainvoke(provided_args, config)
        print("result: ", result)
        # 异步生成器通过 StopAsyncIteration 返回最终值
        yield ToolCallResult(
            success=True,
            msg="工具执行成功",
            result=result,
        )
