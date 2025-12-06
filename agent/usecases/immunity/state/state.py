import operator
from typing import Annotated, Any, Dict, List

from pydantic import BaseModel
from usecases.immunity.schema.common_schemas import Citation, PlanStep, TaskInfo

class RetrievalState(BaseModel):
    """RAG图状态类 - Base state for retrieval workflows"""

    original_question: str
    optimized_questions: List[str] = []
    optimized_question: str = ""
    generated_plan: str = ""
    context: str = ""
    retrieval_docs: List[Any] = []
    citations: List[Citation] = []


class ImprovedCellState(RetrievalState):
    """
    Extended state for improved workflow with reordered stages.

    The key difference is that research and hypothesis inform planning.
    """

    # Override individual_plans to use Dict type for compatibility
    individual_plans: Annotated[List[Dict[str, Any]], operator.add] = []

    # Deep research fields (populated earlier in workflow)
    deep_research_findings: Dict[str, Any] = {}
    research_confidence: float = 0.0
    research_insights: List[str] = []
    research_evidence: List[str] = []
    research_gaps: List[str] = []
    research_recommendations: List[str] = []
    research_summary: str = ""
    research_report_path: str = ""  # Stage3 深度研究分析文件路径

    # Hypothesis fields (populated before planning)
    hypothesis: Dict[str, Any] = {}
    hypothesis_confidence: float = 0.0
    testable_predictions: List[str] = []
    hypothesis_summary: str = ""
    hypothesis_report_path: str = ""  # Stage4 假设生成文件路径

    # Research-informed planning fields
    research_informed_plan: str = ""
    final_enhanced_plan: str = ""
    planning_report_path: str = ""  # Stage5 研究计划文件路径

    # Evaluation fields
    final_evaluation: str = ""
    evaluation_report_path: str = ""  # Stage6 评估报告文件路径

    # Retrieval fields
    retrieval_report_path: str = ""  # Stage2 检索结果文件路径

    decomposed_tasks: List[TaskInfo] = []
    plan_step_details: List[PlanStep] = []
    approved_plan: Dict[str, Any] = {}
    plan_confirmation_status: str = "pending"
    
    # Task execution results - 任务执行结果
    task_results: List[Dict[str, Any]] = []
    
    # Plan detection flags - 计划检测标志
    skip_planning: bool = False  # 是否跳过计划生成阶段
    is_user_provided_plan: bool = False  # 用户是否直接提供了计划
    
    # CSV result collection - CSV结果收集
    merged_csv_result_path: str = ""  # 最终合并的CSV文件路径


class ParallelPlanState(RetrievalState):
    """Parallel plan generation state"""

    query: str = ""  # Single query
    individual_plans: Annotated[List[Dict], operator.add] = []
