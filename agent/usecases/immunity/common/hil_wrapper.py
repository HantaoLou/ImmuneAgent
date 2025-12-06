from typing import Any, Optional
from pydantic import BaseModel

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from typing import Callable
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt


class UserInput(BaseModel):
    accept: bool
    args: Optional[dict[str, Any]] = None

def hil(t: BaseTool) -> BaseTool:

    def wrapper_fn(**tool_input):
        print(f"call tool: {t.name} with {tool_input}")
        user_input = interrupt(f"call tool: {t.name} with {tool_input}")

        try:
            print(f"[DEBUG] 接收到的user_input: {user_input}")
            print(f"[DEBUG] user_input类型: {type(user_input)}")
            user_input_model = UserInput.model_validate_json(user_input)
            print(f"[DEBUG] 解析后的模型: accept={user_input_model.accept}, args={user_input_model.args}")
            if user_input_model.accept:
                if user_input_model.args is not None:
                    print(f"user accept to call tool: {t.name} with {user_input_model.args}")
                    # 检查工具是否支持异步调用
                    if hasattr(t, 'ainvoke'):
                        import asyncio
                        return asyncio.run(t.ainvoke(user_input_model.args))
                    else:
                        return t.invoke(user_input_model.args)
                else:
                    print(f"calling with default args: {tool_input}")
                    # 检查工具是否支持异步调用
                    if hasattr(t, 'ainvoke'):
                        import asyncio
                        return asyncio.run(t.ainvoke(tool_input))
                    else:
                        return t.invoke(tool_input)
            else:
                return f"user reject to call tool: {t.name} with {tool_input}"
        except Exception as e:
            import traceback


            stack_trace = traceback.format_exc()
            print(f"error: {e}\n{stack_trace}")


            return f"error: {e}"

    wrapper = tool(t.name, description=t.description)(wrapper_fn)
    wrapper.args_schema = t.args_schema
    wrapper.return_direct = t.return_direct
    wrapper.description = t.description
    return wrapper


def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""
    from langchain_core.tools import tool as create_tool
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(  
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        import asyncio
        
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call",
            "tool_info": {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema
            }
        }
        # response = interrupt([request])[0]
        response = interrupt(request)
        # approve the tool call
        if response["type"] == "accept":
            # 优先使用异步调用，避免StructuredTool同步调用错误
            if hasattr(tool, 'ainvoke'):
                tool_response = asyncio.run(tool.ainvoke(tool_input, config))
            else:
                tool_response = tool.invoke(tool_input, config)
        # update tool call args
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            # 优先使用异步调用，避免StructuredTool同步调用错误
            if hasattr(tool, 'ainvoke'):
                tool_response = asyncio.run(tool.ainvoke(tool_input, config))
            else:
                tool_response = tool.invoke(tool_input, config)
        # respond to the LLM with user feedback
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt