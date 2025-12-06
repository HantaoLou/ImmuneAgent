from functools import wraps
from typing import Any, Callable, Optional
import asyncio

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_core.runnables.config import RunnableConfig
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing_extensions import Annotated

from usecases.deepagents.prompts import (
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
    WRITE_TODOS_DESCRIPTION,
)
from usecases.deepagents.state import DeepAgentState, Todo
from common.factory import get_mcp_client, get_all_mcp_servers


async def _get_tool_schema_from_mcp(tool_name: str) -> Optional[dict]:
    """
    从MCP服务获取工具的schema
    
    Args:
        tool_name: 工具名称
        
    Returns:
        工具的JSON schema字典，如果找不到则返回None
    """
    try:
        all_servers = get_all_mcp_servers()
        
        # 遍历所有MCP服务器，查找包含该工具的服务
        for service_id in all_servers.keys():
            try:
                config = RunnableConfig(configurable={"mcp_config": {"service_ids": [service_id]}})
                client = await get_mcp_client(config)
                tools = await client.get_tools()
                
                # 查找匹配的工具
                for tool in tools:
                    if tool.name.lower() == tool_name.lower():
                        # 获取工具的args_schema
                        args_schema = getattr(tool, "args_schema", None)
                        
                        if args_schema:
                            # 尝试转换为JSON schema
                            if hasattr(args_schema, "model_json_schema"):
                                schema = args_schema.model_json_schema()
                                print(f"[FileValidation] 成功获取工具 {tool_name} 的schema (来自服务 {service_id})")
                                return schema
                            elif hasattr(args_schema, "schema"):
                                schema = args_schema.schema()
                                print(f"[FileValidation] 成功获取工具 {tool_name} 的schema (来自服务 {service_id})")
                                return schema
                            elif isinstance(args_schema, dict):
                                # 如果已经是字典，直接返回
                                print(f"[FileValidation] 成功获取工具 {tool_name} 的schema (来自服务 {service_id})")
                                return args_schema
            except Exception as e:
                # 某个服务器连接失败，继续尝试下一个
                continue
        
        return None
    except Exception as e:
        return None


@tool(description=WRITE_TODOS_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files"""
    return list(state.get("files", {}).keys())


@tool(description=TOOL_DESCRIPTION)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file."""
    mock_filesystem = state.get("files", {})
    if file_path not in mock_filesystem:
        return f"Error: File '{file_path}' not found"

    # Get file content
    content = mock_filesystem[file_path]

    # Handle empty file
    if not content or content.strip() == "":
        return "System reminder: File exists but has empty contents"

    # Split content into lines
    lines = content.splitlines()

    # Apply line offset and limit
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    # Handle case where offset is beyond file length
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    # Format output with line numbers (cat -n format)
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i]

        # Truncate lines longer than 2000 characters
        if len(line_content) > 2000:
            line_content = line_content[:2000]

        # Line numbers start at 1, so add 1 to the index
        line_number = i + 1
        result_lines.append(f"{line_number:6d}\t{line_content}")

    return "\n".join(result_lines)


def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write to a file."""
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(description=EDIT_DESCRIPTION)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    replace_all: bool = False,
) -> str:
    """Write to a file."""
    mock_filesystem = state.get("files", {})
    # Check if file exists in mock filesystem
    if file_path not in mock_filesystem:
        return f"Error: File '{file_path}' not found"

    # Get current file content
    content = mock_filesystem[file_path]

    # Check if old_string exists in the file
    if old_string not in content:
        return f"Error: String not found in file: '{old_string}'"

    # If not replace_all, check for uniqueness
    if not replace_all:
        occurrences = content.count(old_string)
        if occurrences > 1:
            return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        elif occurrences == 0:
            return f"Error: String not found in file: '{old_string}'"

    # Perform the replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacement_count = content.count(old_string)
        result_msg = f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
    else:
        new_content = content.replace(
            old_string, new_string, 1
        )  # Replace only first occurrence
        result_msg = f"Successfully replaced string in '{file_path}'"

    # Update the mock filesystem
    mock_filesystem[file_path] = new_content
    return Command(
        update={
            "files": mock_filesystem,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


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
            print(
                f"[DEBUG] 解析后的模型: accept={user_input_model.accept}, args={user_input_model.args}"
            )
            if user_input_model.accept:
                # 确定最终使用的参数
                final_args = user_input_model.args if user_input_model.args is not None else tool_input
                
                try:
                    from usecases.immunity.common.file_param_validator import validate_file_params_for_tool
                    
                    # 获取工具的schema
                    tool_schema = None
                    
                    # 首先尝试从工具对象本身获取schema
                    if hasattr(t, "args_schema") and t.args_schema:
                        # 将Pydantic schema转换为字典格式
                        if hasattr(t.args_schema, "model_json_schema"):
                            tool_schema = t.args_schema.model_json_schema()
                        elif hasattr(t.args_schema, "schema"):
                            tool_schema = t.args_schema.schema()
                    
                    # 如果工具对象没有schema，尝试从MCP服务获取
                    if not tool_schema:
                        print(f"[FileValidation] 工具 {t.name} 没有args_schema，尝试从MCP服务获取...")
                        tool_schema = asyncio.run(_get_tool_schema_from_mcp(t.name))
                    
                    # 异步校验文件参数（在调用工具前）
                    print(f"[FileValidation] 开始校验工具 {t.name} 的文件参数...")
                    validated_args = asyncio.run(
                        validate_file_params_for_tool(
                            tool_args=final_args,
                            tool_schema=tool_schema
                        )
                    )
                    
                    if validated_args != final_args:
                        print(f"[FileValidation] 文件参数已更新，使用校验后的参数")
                        final_args = validated_args
                    else:
                        print(f"[FileValidation] 文件参数校验完成，无需转换")
                        
                except Exception as validation_error:
                    # 文件校验失败不应该阻止工具执行，只记录警告
                    import traceback
                    print(f"[FileValidation] 警告：文件参数校验失败，继续使用原始参数: {validation_error}")
                    print(f"[FileValidation] 详细错误: {traceback.format_exc()}")
                
                if user_input_model.args is not None:
                    print(
                        f"user accept to call tool: {t.name} with {final_args}"
                    )
                    # 检查工具是否支持异步调用
                    if hasattr(t, "ainvoke"):
                        return asyncio.run(t.ainvoke(final_args))
                    else:
                        return t.invoke(final_args)
                else:
                    print(f"calling with default args: {final_args}")
                    # 检查工具是否支持异步调用
                    if hasattr(t, "ainvoke"):
                        return asyncio.run(t.ainvoke(final_args))
                    else:
                        return t.invoke(final_args)
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
