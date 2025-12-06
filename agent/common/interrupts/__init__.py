from enum import StrEnum
from typing import Literal
from pydantic import BaseModel


class ToolCallRequest(BaseModel):
    """
    Request approval or edit of tool call from human
    """
    tool_name: str
    args: dict
    allowed_actions: list[Literal["accept", "reject", "edit"]]
    

class ToolCallResult(BaseModel):
    tool_name: str
    action: Literal["accept", "reject", "edit"]
    # if action is edit, provide new args
    args: dict | None = None

class ConversationRequest(BaseModel):
    """
    Request natural language from human. This is useful when clarifying 
    user's intention or provide additional information.
    """
    # message from agent
    message: str

class ConversationResult(BaseModel):
    # message from human
    message: str

class InterruptRequest(BaseModel):
    """
    Agent -> Human
    """
    type: Literal["tool_call", "conversation"]
    data: list[ToolCallRequest] | ConversationRequest


class InterruptResponse(BaseModel):
    """
    Human -> Agent
    """
    type: Literal["tool_call", "conversation"]
    data: list[ToolCallResult] | ConversationResult
