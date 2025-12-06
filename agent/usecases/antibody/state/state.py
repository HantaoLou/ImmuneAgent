from typing import Dict, List, Optional

from pydantic import BaseModel


class RetrievalState(BaseModel):
    """RAG图状态类"""

    original_question: str
    optimized_question: str
    optimized_questions: List[str]
    context: str = ""
    generated_plan: str


class PlanState(BaseModel):
    """研究计划状态"""

    original_question: str
    optimized_question: str
    optimized_questions: List[str]
    context: str
    generated_plan: str
    refine_plan: str
    refine_result: str
    user_feedback: Optional[str]
    csv_path: Optional[str]
    csv_validation_passed: bool
    user_choice: Optional[str]
    just_refined: bool
    metabcr_result: Optional[str]


class ExecuteState(BaseModel):
    """执行器状态"""

    generated_plan: str
    tool_list: List[str] = []
    tool_results: Dict[int, Dict] = {}
    execution_complete: bool = False
    current_instruction: str = ""
    execution_count: int = 0
