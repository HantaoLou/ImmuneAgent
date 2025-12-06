from typing import Any, Optional

from pydantic import BaseModel


class Interrupts:
    ARG_CONFIRM = "arg_confirm"
    TOOL_CONFIRM = "tool_confirm"


class ArgConfirmationInterrupt(BaseModel):
    tool_name: str
    arg_name: str
    provided_value: Optional[str] = None  # 允许 None 值，避免 ValidationError
    arg_type: Optional[str] = (
        None  # 参数类型 (string, number, integer, boolean, array, object)
    )
    default_value: Optional[Any] = None  # 参数默认值


class ArgConfirmationResult(BaseModel):
    confirmed_args: Optional[dict] = None


class ConfirmToolCallInterrupt(BaseModel):
    tool_name: str
    args: dict


class ConfirmToolCallResult(BaseModel):
    msg: str


class ToolCallResult(BaseModel):
    success: bool
    msg: str
    result: Any
