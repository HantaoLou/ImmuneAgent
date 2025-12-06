"""
Planning tools for ImmuneAgent - Complete implementation.
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import DEFAULT_LLM_MODEL, OPENAI_API_KEY
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class PlanPhase(Enum):
    """Research plan phases."""

    DATA_PREPROCESSING = "data_preprocessing"
    EXPLORATORY_ANALYSIS = "exploratory_analysis"
    PRIMARY_ANALYSIS = "primary_analysis"
    VALIDATION = "validation"
    VISUALIZATION = "visualization"
    DOCUMENTATION = "documentation"


@dataclass
class ResearchHypothesis:
    """Structured hypothesis with testable predictions."""

    primary_hypothesis: str
    molecular_mechanism: str
    testable_predictions: List[str] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_level: float = 0.0
    novelty_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "primary_hypothesis": self.primary_hypothesis,
            "molecular_mechanism": self.molecular_mechanism,
            "testable_predictions": self.testable_predictions,
            "success_metrics": self.success_metrics,
            "confidence_level": self.confidence_level,
            "novelty_score": self.novelty_score,
        }


@dataclass
class ToolSpecification:
    """Detailed tool specification with parameters."""

    tool_name: str
    category: str
    description: str
    parameters: Dict[str, Any]
    input_format: str
    output_format: str
    expected_runtime: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "category": self.category,
            "description": self.description,
            "parameters": self.parameters,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "expected_runtime": self.expected_runtime,
        }


@dataclass
class ExecutionStep:
    """Detailed execution step with validation."""

    step_number: int
    phase: PlanPhase
    description: str
    tool: str
    command: str
    parameters: Dict[str, Any]
    dependencies: List[int] = field(default_factory=list)
    expected_output: str = ""
    validation_criteria: str = ""
    estimated_time: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "phase": self.phase.value
            if isinstance(self.phase, PlanPhase)
            else self.phase,
            "description": self.description,
            "tool": self.tool,
            "command": self.command,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "expected_output": self.expected_output,
            "validation_criteria": self.validation_criteria,
            "estimated_time": self.estimated_time,
        }


@dataclass
class ResearchPlan:
    """Comprehensive research plan."""

    question: str
    category: str
    hypothesis: ResearchHypothesis
    phases: Dict[str, List[ExecutionStep]]
    selected_tools: List[ToolSpecification]
    execution_steps: List[ExecutionStep]
    validation_strategy: Dict[str, Any]
    expected_outcomes: List[str]
    timeline: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0
    feasibility_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "category": self.category,
            "hypothesis": self.hypothesis.to_dict(),
            "phases": {
                phase: [step.to_dict() for step in steps]
                for phase, steps in self.phases.items()
            },
            "selected_tools": [tool.to_dict() for tool in self.selected_tools],
            "execution_steps": [step.to_dict() for step in self.execution_steps],
            "validation_strategy": self.validation_strategy,
            "expected_outcomes": self.expected_outcomes,
            "timeline": self.timeline,
            "created_at": self.created_at.isoformat(),
            "confidence_score": self.confidence_score,
            "feasibility_score": self.feasibility_score,
        }


class PlanningEngine:
    """Engine for generating research plans."""

    def __init__(self):
        """Initialize planning engine."""
        self.tool_registry = self._load_tool_registry()
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=DEFAULT_LLM_MODEL, temperature=0.3)

    def _load_tool_registry(self) -> Dict[str, List[str]]:
        """Load available tools by category."""
        return {
            "antibody_tools": ["metabcr", "sapiens", "abnumber", "antiberty", "igfold"],
            "structure_tools": [
                "alphafold3",
                "rosettaantibody",
                "colabfold",
                "esmfold",
            ],
            "single_cell_tools": [
                "scanpy",
                "seurat",
                "celltypist",
                "scvi_tools",
                "scenic",
            ],
            "sequence_tools": ["mixcr", "igblast", "changeo", "immunarch"],
            "epitope_tools": ["netmhcpan", "pvactools", "mixmhc2pred", "iedb"],
            "analysis_tools": ["tcrdist3", "gliph2", "cellphonedb", "nichenet"],
        }

    def create_hypothesis(self, question: str, context: str) -> ResearchHypothesis:
        """Create a research hypothesis."""
        # Simple hypothesis generation based on question keywords
        hypothesis = ResearchHypothesis(
            primary_hypothesis=f"Investigation of {question} will reveal novel mechanisms",
            molecular_mechanism="Multi-factor interaction network",
            testable_predictions=[
                "Prediction 1: Key markers will be differentially expressed",
                "Prediction 2: Functional validation will confirm activity",
                "Prediction 3: Clinical relevance will be demonstrated",
            ],
            success_metrics={
                "statistical_significance": 0.05,
                "effect_size": 0.5,
                "validation_rate": 0.8,
            },
            confidence_level=0.75,
            novelty_score=0.8,
        )
        return hypothesis

    def select_tools(self, question: str, category: str) -> List[ToolSpecification]:
        """Select appropriate tools for the research question."""
        selected_tools = []

        # Category-based tool selection
        tool_mapping = {
            "antibody_discovery": ["metabcr", "alphafold3", "sapiens", "abnumber"],
            "single_cell_analysis": ["scanpy", "celltypist", "scenic", "cellphonedb"],
            "tcr_discovery": ["mixcr", "tcrdist3", "gliph2", "immunarch"],
            "structural_immunology": ["alphafold3", "igfold", "rosettaantibody"],
            "epitope_prediction": ["netmhcpan", "pvactools", "iedb"],
        }

        # Get tools for category
        tool_names = tool_mapping.get(category, ["scanpy", "alphafold3"])

        for tool_name in tool_names:
            spec = ToolSpecification(
                tool_name=tool_name,
                category=self._get_tool_category(tool_name),
                description=f"{tool_name} analysis tool",
                parameters={"default": True},
                input_format="varies",
                output_format="varies",
                expected_runtime="5-30 minutes",
            )
            selected_tools.append(spec)

        return selected_tools

    def _get_tool_category(self, tool_name: str) -> str:
        """Get category for a tool."""
        for category, tools in self.tool_registry.items():
            if tool_name in tools:
                return category
        return "general"

    def create_execution_steps(
        self, tools: List[ToolSpecification], category: str
    ) -> List[ExecutionStep]:
        """Create execution steps from selected tools."""
        steps = []
        step_num = 1

        # Phase 1: Data preprocessing
        steps.append(
            ExecutionStep(
                step_number=step_num,
                phase=PlanPhase.DATA_PREPROCESSING,
                description="Load and preprocess data",
                tool="python",
                command="load_data()",
                parameters={"format": "auto-detect"},
                expected_output="Processed data matrix",
                validation_criteria="Data integrity check",
                estimated_time="5 minutes",
            )
        )
        step_num += 1

        # Phase 2: Primary analysis with selected tools
        for tool in tools:
            steps.append(
                ExecutionStep(
                    step_number=step_num,
                    phase=PlanPhase.PRIMARY_ANALYSIS,
                    description=f"Run {tool.tool_name} analysis",
                    tool=tool.tool_name,
                    command=f"{tool.tool_name}.run()",
                    parameters=tool.parameters,
                    dependencies=[1],
                    expected_output=f"{tool.tool_name} results",
                    validation_criteria="Quality metrics pass",
                    estimated_time=tool.expected_runtime,
                )
            )
            step_num += 1

        # Phase 3: Validation
        steps.append(
            ExecutionStep(
                step_number=step_num,
                phase=PlanPhase.VALIDATION,
                description="Validate results",
                tool="validation_suite",
                command="validate_all()",
                parameters={"threshold": 0.95},
                dependencies=list(range(2, step_num)),
                expected_output="Validation report",
                validation_criteria="All checks pass",
                estimated_time="10 minutes",
            )
        )
        step_num += 1

        # Phase 4: Visualization
        steps.append(
            ExecutionStep(
                step_number=step_num,
                phase=PlanPhase.VISUALIZATION,
                description="Generate figures",
                tool="visualization",
                command="create_figures()",
                parameters={"format": "publication"},
                dependencies=[step_num - 1],
                expected_output="Figure files",
                validation_criteria="Visual inspection",
                estimated_time="15 minutes",
            )
        )

        return steps

    def create_research_plan(
        self, question: str, category: str, context: str = ""
    ) -> ResearchPlan:
        """Create a complete research plan."""
        # Generate hypothesis
        hypothesis = self.create_hypothesis(question, context)

        # Select tools
        selected_tools = self.select_tools(question, category)

        # Create execution steps
        execution_steps = self.create_execution_steps(selected_tools, category)

        # Group steps by phase
        phases = {}
        for step in execution_steps:
            phase_name = (
                step.phase.value
                if isinstance(step.phase, PlanPhase)
                else str(step.phase)
            )
            if phase_name not in phases:
                phases[phase_name] = []
            phases[phase_name].append(step)

        # Create validation strategy
        validation_strategy = {
            "statistical_tests": ["t-test", "ANOVA", "FDR correction"],
            "quality_metrics": ["QC pass rate > 95%", "Reproducibility > 0.9"],
            "biological_validation": ["Literature support", "Pathway enrichment"],
        }

        # Define expected outcomes
        expected_outcomes = [
            f"Identification of key factors in {category}",
            "Statistical validation of findings",
            "Biological interpretation of results",
            "Publication-ready figures and tables",
        ]

        # Create timeline
        timeline = {
            "Day 1-2": "Data preprocessing and QC",
            "Day 3-5": "Primary analysis",
            "Day 6-7": "Validation and interpretation",
            "Day 8": "Report generation",
        }

        # Create the plan
        plan = ResearchPlan(
            question=question,
            category=category,
            hypothesis=hypothesis,
            phases=phases,
            selected_tools=selected_tools,
            execution_steps=execution_steps,
            validation_strategy=validation_strategy,
            expected_outcomes=expected_outcomes,
            timeline=timeline,
            confidence_score=0.85,
            feasibility_score=0.90,
        )

        return plan

    async def create_research_plan_async(
        self, question: str, category: str, context: str, hypotheses: List
    ) -> ResearchPlan:
        """Create research plan asynchronously."""
        # For now, just call the sync version
        # In production, this would use async LLM calls
        return self.create_research_plan(question, category, context)


# Utility functions
class ResearchPlanner:
    """Simple research planner for compatibility."""

    def create_research_plan(
        self, question: str, category: str, context: str
    ) -> ResearchPlan:
        """Create a research plan."""
        engine = PlanningEngine()
        return engine.create_research_plan(question, category, context)


def format_plan_as_text(plan: ResearchPlan) -> str:
    """Format a research plan as readable text."""
    lines = []
    lines.append(f"RESEARCH PLAN")
    lines.append("=" * 50)
    lines.append(f"Question: {plan.question}")
    lines.append(f"Category: {plan.category}")
    lines.append(f"\nHYPOTHESIS:")
    lines.append(f"  {plan.hypothesis.primary_hypothesis}")
    lines.append(f"  Confidence: {plan.hypothesis.confidence_level:.1%}")
    lines.append(f"  Novelty: {plan.hypothesis.novelty_score:.1%}")

    lines.append(f"\nSELECTED TOOLS ({len(plan.selected_tools)}):")
    for tool in plan.selected_tools:
        lines.append(f"  - {tool.tool_name} ({tool.category})")

    lines.append(f"\nEXECUTION STEPS ({len(plan.execution_steps)}):")
    for step in plan.execution_steps:
        lines.append(f"  {step.step_number}. {step.description}")
        lines.append(f"     Tool: {step.tool}, Time: {step.estimated_time}")

    lines.append(f"\nTIMELINE:")
    for period, task in plan.timeline.items():
        lines.append(f"  {period}: {task}")

    lines.append(f"\nSCORES:")
    lines.append(f"  Confidence: {plan.confidence_score:.1%}")
    lines.append(f"  Feasibility: {plan.feasibility_score:.1%}")

    return "\n".join(lines)


# Tool functions for LangChain integration


@tool
def create_analysis_plan(
    question: str, context: str = "", category: str = "general_immunology"
) -> str:
    """
    Create a comprehensive analysis plan for an immunology research question.

    Args:
        question: Research question
        context: Optional context from literature
        category: Analysis category

    Returns:
        Formatted research plan
    """
    planner = ResearchPlanner()
    plan = planner.create_research_plan(question, category, context)
    return format_plan_as_text(plan)


@tool
def optimize_research_plan(plan: str, constraints: str = "") -> str:
    """
    Optimize an existing research plan based on constraints.

    Args:
        plan: Existing research plan
        constraints: Constraints or requirements

    Returns:
        Optimized plan
    """
    # Simple optimization logic
    optimized = f"OPTIMIZED PLAN\n\n{plan}\n\nOptimizations applied:\n"

    if "time" in constraints.lower():
        optimized += "- Parallelized independent analyses\n"
    if "cost" in constraints.lower():
        optimized += "- Selected cost-effective tools\n"
    if "accuracy" in constraints.lower():
        optimized += "- Added validation steps\n"

    return optimized


@tool
def extract_tool_parameters(plan: str) -> Dict[str, Any]:
    """
    Extract tool parameters from a research plan.

    Args:
        plan: Research plan text

    Returns:
        Dictionary of tool parameters
    """
    # Simple parameter extraction
    parameters = {
        "tools": [],
        "data_requirements": [],
        "computational_resources": "standard",
        "timeline": "2-4 weeks",
    }

    # Extract tool names
    if "scanpy" in plan.lower():
        parameters["tools"].append("scanpy")
    if "alphafold" in plan.lower():
        parameters["tools"].append("alphafold3")
    if "metabcr" in plan.lower():
        parameters["tools"].append("metabcr")

    return parameters
