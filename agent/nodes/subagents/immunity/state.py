"""
Immunity Agent Subgraph State Definition

Complete immunology agent workflow state, including:
Query Decomposition → Retrieval → Deep Research → Hypothesis Generation → Planning → Evaluation
"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field, ConfigDict


class ImmunityState(BaseModel):
    """
    Immunity Subgraph State

    Complete workflow:
    1. Query Decomposition: Decompose user question into optimized sub-questions
    2. Retrieval: Retrieve relevant information from knowledge base and web
    3. Deep Research: Conduct in-depth analysis of retrieved content
    4. Hypothesis Generation: Generate testable hypotheses based on research results
    5. Planning: Generate executable experimental plan ⭐
    6. Evaluation: Evaluate the scientific validity and feasibility of the plan
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # IMPORTANT: Do NOT store progress_callback in state - it cannot be serialized by LangGraph.
    # The callback is retrieved dynamically from the global registry via session_id in get_llm().
    session_id: Optional[str] = Field(
        default=None, description="Session ID from GlobalState for progress tracking"
    )

    parent_state: Optional[Any] = Field(
        default=None,
        exclude=True,
        description="Parent GlobalState reference for accessing shared data",
    )

    # ========== Input ==========
    # User original input
    original_question: str = Field(default="", description="User original question")

    # Task list from task_decomposition (optional, if task decomposition results already exist)
    subtasks: List[Any] = Field(default_factory=list, description="Subtask list")
    parallel_task_groups: Dict[str, Any] = Field(
        default_factory=dict, description="Parallel task groups"
    )

    # ========== Stage 1: Query Decomposition ==========
    optimized_questions: List[str] = Field(
        default_factory=list, description="Optimized query list"
    )
    optimized_question: str = Field(default="", description="Optimized single query")

    # ========== Stage 2: Retrieval ==========
    context: str = Field(default="", description="Retrieved context")
    retrieval_docs: List[Any] = Field(
        default_factory=list, description="Retrieved document list"
    )
    citations: List[Any] = Field(default_factory=list, description="Citation list")
    retrieval_report_path: Optional[str] = Field(
        default=None, description="Retrieval report file path"
    )

    # ========== Stage 3: Deep Research ==========
    deep_research_findings: Dict[str, Any] = Field(
        default_factory=dict, description="Deep research results"
    )
    research_confidence: float = Field(default=0.0, description="Research confidence")
    research_insights: List[str] = Field(
        default_factory=list, description="Research insights"
    )
    research_evidence: List[str] = Field(
        default_factory=list, description="Research evidence"
    )
    research_gaps: List[str] = Field(default_factory=list, description="Knowledge gaps")
    research_recommendations: List[str] = Field(
        default_factory=list, description="Research recommendations"
    )
    research_summary: str = Field(default="", description="Research summary")

    # ========== Stage 4: Hypothesis Generation ==========
    hypothesis: Dict[str, Any] = Field(
        default_factory=dict, description="Generated hypothesis"
    )
    hypothesis_confidence: float = Field(
        default=0.0, description="Hypothesis confidence"
    )
    testable_predictions: List[str] = Field(
        default_factory=list, description="Testable predictions"
    )
    hypothesis_summary: str = Field(default="", description="Hypothesis summary")

    # ========== Stage 5: Planning ⭐ ==========
    generated_plan: str = Field(default="", description="Generated plan (initial)")
    final_enhanced_plan: str = Field(default="", description="Final enhanced plan")
    research_informed_plan: str = Field(
        default="", description="Research-informed plan"
    )
    plan_steps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Plan step details"
    )
    plan_summary: str = Field(default="", description="Plan summary")

    # ========== Stage 6: Evaluation ==========
    final_evaluation: str = Field(default="", description="Final evaluation result")

    # ========== Task Decomposition Related ==========
    decomposed_tasks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Decomposed task list"
    )

    # ========== Output: Executable Experimental Plan ==========
    executable_plan: Dict[str, Any] = Field(
        default_factory=dict,
        description="Executable experimental plan (includes tasks, parameters, etc.)",
    )

    # ========== Plan Detection Flags ==========
    skip_planning: bool = Field(
        default=False, description="Whether to skip planning stage"
    )
    is_user_provided_plan: bool = Field(
        default=False, description="Whether user directly provided the plan"
    )

    # ========== System Configuration ==========
    sandbox_dir: str = Field(
        default="",
        description="Sandbox directory (sandbox_data_dir, e.g., /data/sessions/{session_id})",
    )
    local_sandbox_dir: str = Field(
        default="",
        description="Local sandbox directory for fallback (e.g., D:/path/to/sandbox)",
    )
    parent_state: Optional[Any] = Field(
        default=None, description="Parent state reference"
    )

    # ========== Mem0 Cache Related ==========
    cache_hit: bool = Field(default=False, description="Whether cache hit from Mem0")
    skip_immunity_stages: bool = Field(
        default=False, description="Whether to skip immunity stages due to cache hit"
    )
    cache_input_hash: str = Field(
        default="", description="Hash of the input for cache lookup"
    )

    def get_llm(
        self, purpose: str = "reasoning", node_name: Optional[str] = None, **kwargs
    ) -> Optional[Any]:
        """
        Get LLM instance (recommended method)

        Always retrieves progress_callback from global registry via session_id,
        ensuring correct retrieval even when self.progress_callback is None.

        Args:
            purpose: Model purpose, options: "reasoning", "bioinformatics", "reasoning_advanced", "code"
            node_name: Node name
            **kwargs: Additional parameters passed to LLM creation function

        Returns:
            LLM instance, or None if creation fails
        """
        from utils.llm_factory import create_llm_with_thinking

        # [FIX] Do NOT use self.progress_callback - it cannot be serialized by LangGraph.
        # Retrieve callback dynamically from global registry via session_id.
        progress_callback = None
        if self.session_id:
            try:
                import sys
                from pathlib import Path

                backend_dir = Path(__file__).parent.parent.parent.parent / "backend"
                project_root = backend_dir.parent

                if str(backend_dir) not in sys.path:
                    sys.path.insert(0, str(backend_dir))
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))

                from backend import progress_tracker as pt_module

                progress_callback = pt_module.get_progress_callback(self.session_id)
            except (ImportError, AttributeError) as e:
                print(f"[ImmunityState.get_llm] Failed to get callback: {e}")

        return create_llm_with_thinking(
            purpose=purpose,
            progress_callback=progress_callback,
            session_id=self.session_id,
            node_name=node_name,
            **kwargs,
        )
