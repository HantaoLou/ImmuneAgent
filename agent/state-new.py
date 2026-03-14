from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# Global state class
class GlobalState(BaseModel):
    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    session_id: str = Field(description="Session ID")
    user_input: str = Field(description="User's original input")
    task_results: Dict[str, Any] = Field(
        default_factory=dict, description="task results"
    )
    final_result: str = Field(description="final result")
