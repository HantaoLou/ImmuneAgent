"""
Tools for ImmuneAgent.
"""

from .execution_tools import (
    TOOL_REGISTRY,
    ToolExecutor,
    count_total_tools,
    get_tool_info,
    get_tools_by_category,
)
from .hypothesis_tools import Hypothesis, HypothesisGenerator, format_hypotheses_as_text
from .planning_tools import (
    ExecutionStep,
    PlanningEngine,
    PlanPhase,
    ResearchHypothesis,
    ResearchPlan,
    ToolSpecification,
    format_plan_as_text,
)
from .retrieval_tools import (
    ImmunologyRetriever,
    expand_query,
    retrieve_immunology_knowledge,
)

__all__ = [
    # Retrieval
    "ImmunologyRetriever",
    "retrieve_immunology_knowledge",
    "expand_query",
    # Execution
    "ToolExecutor",
    "TOOL_REGISTRY",
    "get_tools_by_category",
    "get_tool_info",
    "count_total_tools",
    # Hypothesis
    "Hypothesis",
    "HypothesisGenerator",
    "format_hypotheses_as_text",
    # Planning
    "PlanPhase",
    "ResearchHypothesis",
    "ToolSpecification",
    "ExecutionStep",
    "ResearchPlan",
    "PlanningEngine",
    "format_plan_as_text",
]
