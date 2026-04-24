"""
Result Evaluator Subgraph State Definition

State definition for summarizing execution results and generating final reports

Enhanced features:
1. Collect complete analysis pipeline information (deep research, hypothesis, execution plan, task list)
2. Analyze task list derivation basis
3. Collect and analyze tool outputs
4. Generate academic paper style analysis reports
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field, ConfigDict


class TaskResultSummary(BaseModel):
    """Summary of a single task's execution result"""

    task_id: str = Field(description="Task ID")
    task_type: str = Field(default="", description="Task type")
    status: str = Field(default="", description="Task status")
    content: str = Field(default="", description="Task content")
    error: Optional[str] = Field(default=None, description="Error message")
    output: Optional[Any] = Field(default=None, description="Output result")
    output_files: List[str] = Field(
        default_factory=list, description="Output file path list"
    )
    execution_time: Optional[float] = Field(
        default=None, description="Execution time (seconds)"
    )


class DeepResearchInfo(BaseModel):
    """Deep research information"""

    research_summary: str = Field(default="", description="Research summary")
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    evidence: List[str] = Field(default_factory=list, description="Evidence")
    knowledge_gaps: List[str] = Field(
        default_factory=list, description="Knowledge gaps"
    )
    confidence: float = Field(default=0.0, description="Research confidence")


class HypothesisInfo(BaseModel):
    """Hypothesis information"""

    hypothesis_summary: str = Field(default="", description="Hypothesis summary")
    testable_predictions: List[str] = Field(
        default_factory=list, description="Testable predictions"
    )
    confidence: float = Field(default=0.0, description="Hypothesis confidence")


class TaskListDerivation(BaseModel):
    """Task list derivation basis"""

    decomposition_summary: str = Field(
        default="", description="Task decomposition summary"
    )
    required_services: List[str] = Field(
        default_factory=list, description="Required services list"
    )
    dependency_rationale: str = Field(default="", description="Dependency rationale")
    parallel_groups: Dict[str, Any] = Field(
        default_factory=dict, description="Parallel group information"
    )


class ToolOutputSummary(BaseModel):
    """Tool output summary"""

    file_path: str = Field(default="", description="File path")
    file_type: str = Field(default="", description="File type")
    content_preview: str = Field(default="", description="Content preview")
    key_results: List[str] = Field(default_factory=list, description="Key results")
    content_summary: str = Field(
        default="", description="LLM-summarized content summary"
    )
    file_size: int = Field(default=0, description="File size (bytes)")
    row_count: Optional[int] = Field(
        default=None, description="Row count for CSV/table files"
    )
    columns: List[str] = Field(
        default_factory=list, description="Column names for CSV files"
    )
    statistics: Dict[str, Any] = Field(
        default_factory=dict, description="File statistics"
    )


class ResultEvaluatorState(BaseModel):
    """
    Result Evaluator Subgraph State

    Features:
    1. Collect execution results from all tasks
    2. Collect files produced by executor
    3. Analyze in combination with immunity plan
    4. Generate final summary report (txt format)

    Enhanced features:
    - Collect and display deep research results
    - Collect and display hypothesis
    - Analyze task list derivation basis
    - Integrate tool output analysis
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ========== SSE progress callback ==========
    # IMPORTANT: Do NOT store progress_callback in state - it cannot be serialized by LangGraph.
    # The callback is retrieved dynamically from the global registry via session_id in get_llm().
    session_id: Optional[str] = Field(default=None, description="Session ID")
    # Note: exclude=True to avoid LangGraph serialization failure (circular reference to GlobalState)
    parent_state: Optional[Any] = Field(
        default=None, exclude=True, description="Parent state reference"
    )

    # ========== Input ==========
    # User's original input
    user_input: str = Field(default="", description="User's original input")

    # Execution plan produced by Immunity subgraph
    execution_plan: str = Field(default="", description="Execution plan")

    # ========== Enhanced Input: Analysis Pipeline Information ==========
    # Deep research information
    deep_research: DeepResearchInfo = Field(
        default_factory=DeepResearchInfo, description="Deep research results"
    )

    # Hypothesis information
    hypothesis: HypothesisInfo = Field(
        default_factory=HypothesisInfo, description="Generated hypothesis"
    )

    # Task list derivation basis
    task_list_derivation: TaskListDerivation = Field(
        default_factory=TaskListDerivation, description="Task list derivation basis"
    )

    # Task list
    task_results: Dict[str, TaskResultSummary] = Field(
        default_factory=dict, description="Task execution results, key=task_id"
    )

    # All tasks
    all_tasks: List[TaskResultSummary] = Field(
        default_factory=list, description="All tasks list"
    )

    # ========== Enhanced Input: Tool Outputs ==========
    # File paths produced by Executor
    output_files: List[str] = Field(
        default_factory=list, description="File paths produced by executor"
    )

    # Tool output summaries
    tool_output_summaries: List[ToolOutputSummary] = Field(
        default_factory=list, description="Tool output summary list"
    )

    # ========== Execution Statistics ==========
    total_tasks: int = Field(default=0, description="Total task count")
    completed_tasks: int = Field(default=0, description="Completed task count")
    failed_tasks: int = Field(default=0, description="Failed task count")

    # ========== Analysis Results ==========
    success_rate: float = Field(default=0.0, description="Success rate")
    error_summary: str = Field(default="", description="Error summary")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )

    # ========== Enhanced Analysis Results ==========
    methodology: str = Field(default="", description="Analysis method description")
    scientific_rationale: str = Field(default="", description="Scientific rationale")
    limitations: List[str] = Field(default_factory=list, description="Limitations")
    validation_recommendations: List[str] = Field(
        default_factory=list, description="Validation recommendations"
    )

    # ========== Output ==========
    summary_report: str = Field(default="", description="Summary report")
    detailed_report: str = Field(default="", description="Detailed report")
    txt_report: str = Field(default="", description="TXT format analysis report")
    report_path: Optional[str] = Field(default=None, description="Report file path")
    txt_report_path: Optional[str] = Field(
        default=None, description="TXT report file path"
    )

    # ========== System Configuration ==========
    sandbox_dir: str = Field(default="", description="Sandbox directory")

    def get_llm(
        self, purpose: str = "reasoning", node_name: Optional[str] = None, **kwargs
    ) -> Optional[Any]:
        """
        Get LLM instance (recommended method)

        Reuses the parent graph's get_llm method; falls back to local callback if parent graph is unavailable.

        Args:
            purpose: Model purpose, options: "reasoning", "bioinformatics", "reasoning_advanced", "code"
            node_name: Node name
            **kwargs: Additional parameters passed to LLM creation function

        Returns:
            LLM instance, or None if creation fails
        """
        if self.parent_state and hasattr(self.parent_state, "get_llm"):
            return self.parent_state.get_llm(
                purpose=purpose, node_name=node_name or "result_evaluator", **kwargs
            )

        from agent.utils.llm_factory import create_llm_with_thinking

        # Do NOT pass progress_callback - it cannot be serialized by LangGraph.
        # The factory will retrieve it from the global registry using session_id.
        return create_llm_with_thinking(
            purpose=purpose,
            session_id=self.session_id,
            node_name=node_name or "result_evaluator",
            **kwargs,
        )
