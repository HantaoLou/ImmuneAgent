"""
Hypothesis generation tools for ImmuneAgent.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from common.factory import get_default_model, get_reasoning_model
from usecases.immunology.immunology_config import get_immunology_model_config


@dataclass
class Hypothesis:
    """Structured hypothesis with predictions and validation."""

    statement: str
    rationale: str
    predictions: List[str] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    novelty: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "statement": self.statement,
            "rationale": self.rationale,
            "predictions": self.predictions,
            "experiments": self.experiments,
            "confidence": self.confidence,
            "novelty": self.novelty,
        }


class HypothesisGenerator:
    """Generate and refine scientific hypotheses."""

    async def generate_hypotheses_async(
        self, question: str, context: str, config: Optional[RunnableConfig] = None
    ) -> List[Hypothesis]:
        """Generate hypotheses asynchronously."""

        prompt = ChatPromptTemplate.from_template("""Generate 3 scientific hypotheses for this immunology research question.

Question: {question}

Context:
{context}

For each hypothesis provide:
1. A clear, testable hypothesis statement
2. Scientific rationale
3. 3 specific predictions
4. 2 key experiments to test it
5. Confidence level (0-1)
6. Novelty score (0-1)

Format as JSON:
{{
  "hypotheses": [
    {{
      "statement": "...",
      "rationale": "...",
      "predictions": ["...", "...", "..."],
      "experiments": [
        {{"name": "...", "method": "...", "expected_outcome": "..."}},
        {{"name": "...", "method": "...", "expected_outcome": "..."}}
      ],
      "confidence": 0.X,
      "novelty": 0.X
    }}
  ]
}}""")
        # 如果没有提供配置，使用默认的免疫学配置
        if config is None:
            config = get_immunology_model_config()

        model = get_reasoning_model(config)
        chain = prompt | model | StrOutputParser()
        try:
            response = await chain.ainvoke({"question": question, "context": context})

            # Parse JSON response
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                hypotheses = []
                for h_data in data.get("hypotheses", []):
                    hypothesis = Hypothesis(
                        statement=h_data.get("statement", ""),
                        rationale=h_data.get("rationale", ""),
                        predictions=h_data.get("predictions", []),
                        experiments=h_data.get("experiments", []),
                        confidence=h_data.get("confidence", 0.5),
                        novelty=h_data.get("novelty", 0.5),
                    )
                    hypotheses.append(hypothesis)

                return hypotheses
            else:
                return self._generate_fallback_hypotheses(question)

        except Exception as e:
            print(f"Error generating hypotheses: {e}")
            return self._generate_fallback_hypotheses(question)

    def generate_hypotheses(
        self, question: str, context: str, config: Optional[RunnableConfig] = None
    ) -> List[Hypothesis]:
        """Synchronous wrapper for hypothesis generation."""
        try:
            # 检查是否已经在事件循环中运行
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，使用 asyncio.create_task 或返回 fallback
            print("⚠️ Running in existing event loop, using fallback hypotheses")
            return self._generate_fallback_hypotheses(question)
        except RuntimeError:
            # 没有运行的事件循环，可以安全地创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.generate_hypotheses_async(question, context, config)
                )
            finally:
                loop.close()

    def _generate_fallback_hypotheses(self, question: str) -> List[Hypothesis]:
        """Generate fallback hypotheses without API."""

        hypotheses = []

        # Hypothesis 1: Mechanism-based
        h1 = Hypothesis(
            statement=f"The phenomenon in '{question}' is mediated by specific molecular pathways",
            rationale="Based on known immunological mechanisms",
            predictions=[
                "Key signaling pathways will be activated",
                "Specific markers will be upregulated",
                "Functional changes will be observed",
            ],
            experiments=[
                {
                    "name": "RNA-seq",
                    "method": "Transcriptomics",
                    "expected_outcome": "Differential expression",
                },
                {
                    "name": "Flow cytometry",
                    "method": "Cell phenotyping",
                    "expected_outcome": "Marker validation",
                },
            ],
            confidence=0.7,
            novelty=0.5,
        )
        hypotheses.append(h1)

        # Hypothesis 2: Cell-based
        h2 = Hypothesis(
            statement=f"Specific cell populations drive the response in '{question}'",
            rationale="Cell-cell interactions are critical in immune responses",
            predictions=[
                "Distinct cell clusters will be identified",
                "Cell-cell communication will be detected",
                "Temporal dynamics will be revealed",
            ],
            experiments=[
                {
                    "name": "scRNA-seq",
                    "method": "Single-cell analysis",
                    "expected_outcome": "Cell heterogeneity",
                },
                {
                    "name": "CellPhoneDB",
                    "method": "Interaction analysis",
                    "expected_outcome": "Communication networks",
                },
            ],
            confidence=0.75,
            novelty=0.6,
        )
        hypotheses.append(h2)

        # Hypothesis 3: Therapeutic
        h3 = Hypothesis(
            statement=f"Targeting identified mechanisms will have therapeutic benefit",
            rationale="Immunomodulation can alter disease outcomes",
            predictions=[
                "Target inhibition will reduce pathology",
                "Biomarkers will predict response",
                "Safety profile will be acceptable",
            ],
            experiments=[
                {
                    "name": "In vitro validation",
                    "method": "Cell culture",
                    "expected_outcome": "Target engagement",
                },
                {
                    "name": "In vivo models",
                    "method": "Animal studies",
                    "expected_outcome": "Efficacy demonstration",
                },
            ],
            confidence=0.6,
            novelty=0.7,
        )
        hypotheses.append(h3)

        return hypotheses


def format_hypotheses_as_text(hypotheses: List[Hypothesis]) -> str:
    """Format hypotheses as readable text."""
    lines = []
    lines.append("GENERATED HYPOTHESES")
    lines.append("=" * 50)

    for i, hyp in enumerate(hypotheses, 1):
        lines.append(f"\nHypothesis {i}:")
        lines.append(f"  Statement: {hyp.statement}")
        lines.append(f"  Rationale: {hyp.rationale}")
        lines.append(f"  Confidence: {hyp.confidence:.1%}")
        lines.append(f"  Novelty: {hyp.novelty:.1%}")
        lines.append(f"  Predictions:")
        for pred in hyp.predictions:
            lines.append(f"    - {pred}")
        lines.append(f"  Key Experiments:")
        for exp in hyp.experiments:
            lines.append(
                f"    - {exp.get('name', 'Unknown')}: {exp.get('method', 'N/A')}"
            )

    return "\n".join(lines)


# Tool functions for LangChain integration


@tool
def generate_hypothesis(
    question: str, context: str = "", config: Optional[RunnableConfig] = None
) -> str:
    """
    Generate a scientific hypothesis for an immunology question.

    Args:
        question: Research question
        context: Optional context from literature

    Returns:
        Formatted hypothesis with predictions
    """
    generator = HypothesisGenerator()
    hypotheses = generator.generate_hypotheses(question, context, config)

    if hypotheses:
        # Return the top hypothesis
        return format_hypotheses_as_text(hypotheses[:1])
    else:
        return f"Hypothesis: The phenomenon described in '{question}' involves complex immunological interactions requiring further investigation."


@tool
def validate_hypothesis(
    hypothesis: str, evidence: str, criteria: str = "scientific validity"
) -> str:
    """
    Validate a hypothesis against evidence and criteria.

    Args:
        hypothesis: The hypothesis to validate
        evidence: Available evidence
        criteria: Validation criteria

    Returns:
        Validation assessment
    """
    llm = get_default_model(get_immunology_model_config())

    prompt = f"""
    Validate the following hypothesis:
    
    Hypothesis: {hypothesis}
    
    Evidence: {evidence}
    
    Criteria: {criteria}
    
    Provide:
    1. Strength of evidence (weak/moderate/strong)
    2. Consistency with known biology
    3. Testability assessment
    4. Potential confounders
    5. Overall validity score (0-100%)
    """

    response = llm.invoke(prompt)
    return response.content


@tool
def refine_hypothesis(hypothesis: str, feedback: str) -> str:
    """
    Refine a hypothesis based on feedback.

    Args:
        hypothesis: Original hypothesis
        feedback: Feedback or new information

    Returns:
        Refined hypothesis
    """
    llm = get_default_model(get_immunology_model_config())

    prompt = f"""
    Refine the following hypothesis based on feedback:
    
    Original: {hypothesis}
    
    Feedback: {feedback}
    
    Generate an improved hypothesis that:
    1. Addresses the feedback
    2. Maintains scientific rigor
    3. Includes testable predictions
    4. Considers alternative explanations
    """

    response = llm.invoke(prompt)
    return response.content


# Export hypothesis components
__all__ = [
    "Hypothesis",
    "HypothesisGenerator",
    "format_hypotheses_as_text",
    "generate_hypothesis",
    "validate_hypothesis",
    "refine_hypothesis",
]
