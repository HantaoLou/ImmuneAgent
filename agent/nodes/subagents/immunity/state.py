"""
Immunity Agent Subgraph State Definition

Complete immunology agent workflow state, including:
Query Decomposition → Retrieval → Deep Research → Hypothesis Generation → Planning → Evaluation
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


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
    
    # ========== Input ==========
    # User original input
    original_question: str = Field(default="", description="User original question")
    
    # Task list from task_decomposition (optional, if task decomposition results already exist)
    subtasks: List[Any] = Field(default_factory=list, description="Subtask list")
    parallel_task_groups: Dict[str, Any] = Field(default_factory=dict, description="Parallel task groups")
    parameter_inference_results: Dict[str, Any] = Field(default_factory=dict, description="Parameter inference results")
    
    # ========== Stage 1: Query Decomposition ==========
    optimized_questions: List[str] = Field(default_factory=list, description="Optimized query list")
    optimized_question: str = Field(default="", description="Optimized single query")
    
    # ========== Stage 2: Retrieval ==========
    context: str = Field(default="", description="Retrieved context")
    retrieval_docs: List[Any] = Field(default_factory=list, description="Retrieved document list")
    citations: List[Any] = Field(default_factory=list, description="Citation list")
    retrieval_report_path: Optional[str] = Field(default=None, description="Retrieval report file path")
    
    # ========== Stage 3: Deep Research ==========
    deep_research_findings: Dict[str, Any] = Field(default_factory=dict, description="Deep research results")
    research_confidence: float = Field(default=0.0, description="Research confidence")
    research_insights: List[str] = Field(default_factory=list, description="Research insights")
    research_evidence: List[str] = Field(default_factory=list, description="Research evidence")
    research_gaps: List[str] = Field(default_factory=list, description="Knowledge gaps")
    research_recommendations: List[str] = Field(default_factory=list, description="Research recommendations")
    research_summary: str = Field(default="", description="Research summary")
    
    # ========== Stage 4: Hypothesis Generation ==========
    hypothesis: Dict[str, Any] = Field(default_factory=dict, description="Generated hypothesis")
    hypothesis_confidence: float = Field(default=0.0, description="Hypothesis confidence")
    testable_predictions: List[str] = Field(default_factory=list, description="Testable predictions")
    hypothesis_summary: str = Field(default="", description="Hypothesis summary")
    
    # ========== Stage 5: Planning ⭐ ==========
    generated_plan: str = Field(default="", description="Generated plan (initial)")
    final_enhanced_plan: str = Field(default="", description="Final enhanced plan")
    research_informed_plan: str = Field(default="", description="Research-informed plan")
    plan_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Plan step details")
    plan_summary: str = Field(default="", description="Plan summary")
    
    # ========== Stage 6: Evaluation ==========
    final_evaluation: str = Field(default="", description="Final evaluation result")
    
    # ========== Task Decomposition Related ==========
    decomposed_tasks: List[Dict[str, Any]] = Field(default_factory=list, description="Decomposed task list")
    
    # ========== Output: Executable Experimental Plan ==========
    executable_plan: Dict[str, Any] = Field(default_factory=dict, description="Executable experimental plan (includes tasks, parameters, etc.)")
    
    # ========== Plan Detection Flags ==========
    skip_planning: bool = Field(default=False, description="Whether to skip planning stage")
    is_user_provided_plan: bool = Field(default=False, description="Whether user directly provided the plan")
    
    # ========== System Configuration ==========
    sandbox_dir: str = Field(description="Sandbox directory")
    parent_state: Optional[Any] = Field(default=None, description="Parent state reference")

