"""React-style state models for standardized reasoning traces."""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ReactStepType(str, Enum):
    OBS = "OBS"
    THINK = "THINK"
    ACT = "ACT"
    RESULT = "RESULT"


class ReactStep(BaseModel):
    step_type: ReactStepType = Field(description="React step type")
    content: str = Field(description="Step content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Step timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")


class ThoughtAction(BaseModel):
    thought: str = Field(description="Reasoning or plan")
    action: str = Field(description="Action name")
    action_input: Optional[Any] = Field(default=None, description="Action input payload")
    observation: Optional[str] = Field(default=None, description="Observation after action")


class ReactLoopState(BaseModel):
    steps: List[ReactStep] = Field(default_factory=list, description="React steps")
    max_steps: int = Field(default=10, description="Maximum steps allowed")
    terminated: bool = Field(default=False, description="Whether the loop terminated")
    termination_reason: Optional[str] = Field(default=None, description="Termination reason")

