"""
State models for the ImmuneAgent system.
Modularized for better organization and Qdrant integration.
"""

import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class PlanPhase(Enum):
    """Planning phase enum"""

    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETED = "completed"


class ImmuneAgentState(BaseModel):
    """Main state for the ImmuneAgent covering all immunology domains."""

    # Core question and context
    question: str = ""  # Primary question field used by nodes
    research_question: str = ""  # Alias for compatibility
    original_question: str = ""
    context: str = ""
    retrieved_context: Optional[str] = None  # Context from retrieval
    domain: str = "general_immunology"
    analysis_type: str = "general_immunology"
    execution_parameters: Dict[str, Any] = Field(default_factory=dict)

    # Workflow phase
    phase: PlanPhase = PlanPhase.PLANNING

    # Hypothesis generation
    hypothesis: Optional[str] = None  # Current hypothesis
    hypotheses: List[str] = Field(default_factory=list)
    hypothesis_result: Optional[Any] = None
    testable_predictions: List[str] = Field(default_factory=list)
    mechanistic_model: str = ""

    # Research planning
    plan: str = ""  # Primary plan field used by nodes
    generated_plan: str = ""
    research_plan: Optional[Any] = None
    selected_tools: List[str] = Field(default_factory=list)
    methodology: Dict[str, List[str]] = Field(default_factory=dict)

    # Tool execution results
    execution_results: Dict[str, Any] = Field(default_factory=dict)
    analysis_completed: bool = False

    # Results and findings
    results: List[Dict[str, Any]] = Field(default_factory=list)  # Primary results field
    key_findings: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0  # Single confidence score
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    validation: Optional[str] = None  # Validation text
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    therapeutic_targets: List[str] = Field(default_factory=list)
    final_report: str = ""

    # RAG and retrieval
    optimized_questions: List[str] = Field(default_factory=list)
    optimized_question: str = ""
    individual_plans: Annotated[List[str], operator.add] = Field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list)
    relevance_scores: List[float] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)

    # Metadata
    messages: List[BaseMessage] = Field(default_factory=list)  # LangChain messages
    iteration_count: int = 0
    timestamp: str = ""  # ISO format string
    error_messages: List[str] = Field(default_factory=list)


class RetrievalState(BaseModel):
    """State for retrieval operations."""

    query: str
    expanded_queries: List[str] = Field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_documents: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    context: str = ""
    relevance_threshold: float = 0.5


class PlanningState(BaseModel):
    """State for planning operations."""

    objective: str
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    research_plan: Dict[str, Any] = Field(default_factory=dict)
    selected_tools: List[str] = Field(default_factory=list)
    methodology: Dict[str, List[str]] = Field(default_factory=dict)
    timeline: Dict[str, str] = Field(default_factory=dict)
    confidence_score: float = 0.0
    feasibility_score: float = 0.0


class ExecutionState(BaseModel):
    """State for tool execution."""

    tools_to_execute: List[str] = Field(default_factory=list)
    execution_queue: List[Dict[str, Any]] = Field(default_factory=list)
    execution_results: Dict[str, Any] = Field(default_factory=dict)
    successful_tools: List[str] = Field(default_factory=list)
    failed_tools: List[str] = Field(default_factory=list)
    execution_logs: List[str] = Field(default_factory=list)


class ValidationState(BaseModel):
    """State for validation and synthesis."""

    results_to_validate: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    hypothesis_evaluation: Dict[str, bool] = Field(default_factory=dict)
    integrated_findings: List[str] = Field(default_factory=list)
    final_report: str = ""
    recommendations: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)


# Export all state models
__all__ = [
    "ImmuneAgentState",
    "RetrievalState",
    "PlanningState",
    "ExecutionState",
    "ValidationState",
]
