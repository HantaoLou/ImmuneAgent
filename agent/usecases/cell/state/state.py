import operator
from typing import Annotated, Dict, List, Optional

from pydantic import BaseModel


class State(BaseModel):
    refine_plan: str
    plan_confirmed: bool
    model_upload_files: Dict[str, str]
    strategy_upload_files: str = ""
    strategy_input_valid: bool = True
    metabcr_input_valid: bool = True
    metabcr_skip: bool = False  # 是否跳过MetaBCR模型预测
    standardized_files: str = ""  # 存储标准化的文件路径
    combine_fields: str = ""  # 存储用户输入的组合字段
    standardize_status: str = "pending"  # 标准化状态: pending(待处理), processing(处理中), completed(已完成), skipped(已跳过), failed(失败)
    bcr_file_path: str = ""  # BCR文件路径
    bcr_input_valid: bool = False  # BCR文件输入是否有效
    bcr_skip: bool = False  # 是否跳过BCR文件处理
    rds_file_path: str = ""  # RDS文件路径
    rds_bcr_input_valid: bool = False  # RDS和BCR文件整合输入是否有效
    selected_model: List[str]
    selected_strategy: List[str]
    metabcr_result: Optional[str] = ""
    original_question: str
    optimized_questions: List[str]
    generated_plan: str
    context: str
    individual_plans: Annotated[List[str], operator.add] = []
    should_end: bool = False  # 控制策略选择节点是否结束


class RetrievalState(BaseModel):
    """RAG图状态类"""

    original_question: str
    optimized_questions: List[str] = []
    optimized_question: str = ""
    generated_plan: str = ""
    context: str = ""


class ParallelPlanState(RetrievalState):
    """并行计划生成状态"""

    query: str = ""  # 单个查询
    individual_plans: Annotated[List[str], operator.add] = []


class ExecuteState(BaseModel):
    """Execute Graph专用状态类"""

    refine_plan: str
    decomposed_tasks: List[str] = []
