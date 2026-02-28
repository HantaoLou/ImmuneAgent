"""
HLE Optimizations Integration Module

This module integrates all HLE optimization components into the general_qa
subgraph, providing a unified interface for HLE-level question answering.

Usage:
    from agent.nodes.subagents.general_qa.hle_optimizations import HLEOptimizedQA
    
    qa = HLEOptimizedQA(llm=llm)
    result = await qa.answer(question, domain="genetics")
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import asyncio
import logging

# Import all HLE optimization components
from .exact_match_optimizer import ExactMatchOptimizer, AnswerVariantGenerator
from .confidence_calibrator import (
    ConfidenceCalibrator, 
    ConfidenceCalibrator,
    QuestionDifficulty
)
from .original_problem_handler import (
    OriginalProblemHandler,
    FirstPrinciplesReasoner
)
from .deep_reasoning_tree import (
    DeepReasoningTree,
    ReasoningTreeBuilder,
    ReasoningNode,
    ReasoningPath
)
from .multi_agent_framework import (
    MultiAgentFramework,
    AgentRole,
    DebateMechanism
)
from .domain_reasoning_templates import (
    GeneticsReasoningTemplate,
    MolecularBiologyTemplate,
    ClinicalDiagnosisTemplate,
    CommonPitfallsRegistry,
    get_template_for_domain
)
from .concept_knowledge_graph import (
    ConceptKnowledgeGraph,
    ConceptContrast
)
from .reasoning_chain_validator import (
    ReasoningChainValidator,
    ValidationResult,
    ReasoningStep as ValidatedReasoningStep,
    LogicalConnector
)
from .llm_retry_wrapper import LLMRetryWrapper, RecoveryMode, RetryConfig
from .timeout_strategy import (
    AdaptiveTimeoutStrategy,
    ComplexityEstimator,
    ComplexityLevel
)

logger = logging.getLogger(__name__)


@dataclass
class HLEAnswerResult:
    """Result from HLE-optimized question answering"""
    answer: str
    normalized_answer: str
    answer_variants: List[str]
    confidence: float
    calibrated_confidence: float
    
    # Reasoning details
    reasoning_trace: List[Dict[str, Any]]
    reasoning_tree_stats: Optional[Dict[str, Any]] = None
    multi_agent_result: Optional[Dict[str, Any]] = None
    
    # Validation
    validation_result: Optional[Dict[str, Any]] = None
    pitfalls_checked: List[str] = field(default_factory=list)
    concepts_identified: List[str] = field(default_factory=list)
    
    # Metadata
    difficulty_level: str = "unknown"
    strategies_used: List[str] = field(default_factory=list)
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HLEOptimizedQA:
    """
    HLE-Optimized Question Answering System
    
    Integrates all optimization components for answering HLE-level questions:
    1. Question analysis and complexity estimation
    2. Domain-specific reasoning templates
    3. Multi-agent reasoning (for complex questions)
    4. Deep reasoning tree exploration
    5. Reasoning chain validation
    6. Confidence calibration
    7. Answer normalization for exact matching
    """
    
    def __init__(
        self,
        llm: Any = None,
        enable_multi_agent: bool = True,
        enable_deep_reasoning: bool = True,
        enable_validation: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize the HLE-optimized QA system.
        
        Args:
            llm: The language model to use
            enable_multi_agent: Enable multi-agent reasoning
            enable_deep_reasoning: Enable reasoning tree exploration
            enable_validation: Enable reasoning chain validation
            strict_mode: Use strict validation rules
        """
        self.llm = llm
        self.enable_multi_agent = enable_multi_agent
        self.enable_deep_reasoning = enable_deep_reasoning
        self.enable_validation = enable_validation
        self.strict_mode = strict_mode
        
        # Initialize components
        self.exact_matcher = ExactMatchOptimizer()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.original_problem_handler = OriginalProblemHandler()
        self.concept_graph = ConceptKnowledgeGraph()
        self.chain_validator = ReasoningChainValidator(strict_mode=strict_mode)
        self.timeout_strategy = AdaptiveTimeoutStrategy()
        self.complexity_estimator = ComplexityEstimator()
        
        # Optional multi-agent framework
        self.multi_agent_framework = None
        if enable_multi_agent:
            self.multi_agent_framework = MultiAgentFramework(llm=llm)
        
        # Retry wrapper for LLM
        if llm:
            self.llm_wrapper = LLMRetryWrapper(
                llm=llm,
                config=RetryConfig(max_retries=3)
            )
        else:
            self.llm_wrapper = None
        
        # Domain templates
        self.domain_templates = {
            "genetics": GeneticsReasoningTemplate(),
            "molecular_biology": MolecularBiologyTemplate(),
            "clinical": ClinicalDiagnosisTemplate()
        }
    
    async def answer(
        self,
        question: str,
        domain: Optional[str] = None,
        question_type: Optional[str] = None,
        options: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> HLEAnswerResult:
        """
        Answer a question using HLE-optimized strategies.
        
        Args:
            question: The question to answer
            domain: Optional domain hint (e.g., "genetics", "molecular_biology")
            question_type: Optional question type (e.g., "multiple_choice", "calculation")
            options: For multiple choice, the available options
            context: Additional context for reasoning
            
        Returns:
            HLEAnswerResult with answer and full reasoning trace
        """
        import time
        start_time = time.time()
        
        context = context or {}
        reasoning_trace = []
        strategies_used = []
        
        # Step 1: Analyze question complexity
        complexity_factors = self.complexity_estimator.estimate(question, question_type, domain)
        complexity_level = self.complexity_estimator.get_complexity_level(complexity_factors)
        
        reasoning_trace.append({
            "step": "complexity_analysis",
            "level": complexity_level.value,
            "score": complexity_factors.calculate_score()
        })
        
        # Step 2: Get domain-specific context
        domain_template = self.domain_templates.get(domain) if domain else None
        concept_context = self.concept_graph.get_concept_context_for_question(question)
        
        concepts_identified = concept_context.get("identified_concepts", [])
        pitfalls_checked = []
        
        # Check for common pitfalls
        if domain:
            pitfall_warnings = CommonPitfallsRegistry.check_for_pitfall(question, domain)
            pitfalls_checked = [w.pitfall_name for w in pitfall_warnings]
        
        reasoning_trace.append({
            "step": "domain_analysis",
            "domain": domain,
            "concepts": concepts_identified,
            "pitfalls": pitfalls_checked[:3]  # Limit for readability
        })
        
        # Step 3: Choose reasoning strategy based on complexity
        answer = None
        confidence = 0.0
        multi_agent_result = None
        reasoning_tree_stats = None
        
        if complexity_level in [ComplexityLevel.VERY_COMPLEX, ComplexityLevel.HLE_LEVEL]:
            # Use multi-agent for very complex questions
            if self.enable_multi_agent and self.multi_agent_framework:
                strategies_used.append("multi_agent")
                try:
                    ma_result = await self.multi_agent_framework.solve(
                        question, domain or "", context
                    )
                    answer = ma_result.answer
                    confidence = ma_result.confidence
                    multi_agent_result = {
                        "consensus_method": ma_result.consensus_method,
                        "voting_result": ma_result.voting_result,
                        "agent_count": len(ma_result.agent_responses)
                    }
                    reasoning_trace.extend(ma_result.reasoning_trace)
                except Exception as e:
                    logger.warning(f"Multi-agent reasoning failed: {e}")
                    strategies_used.append("multi_agent_failed")
        
        # Step 4: If no answer yet, use deep reasoning tree
        if answer is None and self.enable_deep_reasoning:
            strategies_used.append("deep_reasoning_tree")
            tree_result = await self._deep_reasoning_solve(question, domain, context)
            if tree_result:
                answer = tree_result.get("answer")
                confidence = tree_result.get("confidence", 0.5)
                reasoning_tree_stats = tree_result.get("stats")
                reasoning_trace.append({
                    "step": "deep_reasoning",
                    "stats": reasoning_tree_stats
                })
        
        # Step 5: Fallback to original problem handler for novel questions
        if answer is None or confidence < 0.5:
            if complexity_factors.novel_concepts:
                strategies_used.append("original_problem_handler")
                original_result = self.original_problem_handler.handle(question, context)
                if original_result.answer:
                    answer = original_result.answer
                    confidence = original_result.confidence
                    reasoning_trace.append({
                        "step": "original_problem",
                        "strategy": original_result.primary_strategy.value
                    })
        
        # Step 6: If still no answer, use simple LLM call
        if answer is None and self.llm_wrapper:
            strategies_used.append("llm_fallback")
            llm_result = await self.llm_wrapper.call(
                self._build_prompt(question, domain, domain_template, options),
                context
            )
            if llm_result.success:
                answer = str(llm_result.result)
                confidence = 0.5
        
        # Step 7: Validate reasoning chain
        validation_result = None
        if self.enable_validation and reasoning_trace:
            # Convert trace to validated steps
            steps = []
            for i, trace_item in enumerate(reasoning_trace):
                if "step" in trace_item:
                    steps.append(ValidatedReasoningStep(
                        step_id=i,
                        premise=str(trace_item.get("step", "")),
                        conclusion=str(trace_item.get("result", ""))
                    ))
            
            if steps:
                validation = self.chain_validator.validate(steps, question_type)
                validation_result = {
                    "is_valid": validation.is_valid,
                    "confidence": validation.confidence,
                    "errors": [{"type": e.error_type.value, "description": e.description} 
                              for e in validation.errors[:3]]
                }
        
        # Step 8: Calibrate confidence
        difficulty_enum = self._map_complexity_to_difficulty(complexity_level)
        
        # Check for unfamiliar concepts
        unfamiliar_concepts = []
        if concepts_identified:
            # Check if the first identified concept is in our knowledge graph
            first_concept = concepts_identified[0] if concepts_identified else None
            if first_concept and not self.concept_graph.get_concept(first_concept):
                unfamiliar_concepts = concepts_identified
        
        calibration = self.confidence_calibrator.calibrate(
            confidence,
            reasoning_quality={
                "missing_steps": validation_result.get("errors", []) if validation_result else [],
                "unfamiliar_concepts": unfamiliar_concepts
            },
            question_difficulty=difficulty_enum,
            multi_agent_result=multi_agent_result
        )
        
        calibrated_confidence = calibration.calibrated_confidence
        
        # Step 9: Normalize answer for exact matching
        normalized_answer = self.exact_matcher.normalize_answer(answer) if answer else ""
        answer_variants = self.exact_matcher.variant_generator.generate_variants(
            answer
        ) if answer else []
        
        # Step 10: Add uncertainty expression if needed
        if calibration.uncertainty_expression and calibration.uncertainty_expression.should_express:
            uncertainty = calibration.uncertainty_expression
            answer = f"{uncertainty.suggested_prefix} {answer}. {uncertainty.suggested_suffix}."
        
        total_time = time.time() - start_time
        
        return HLEAnswerResult(
            answer=answer or "",
            normalized_answer=normalized_answer,
            answer_variants=answer_variants[:5],  # Limit variants
            confidence=confidence,
            calibrated_confidence=calibrated_confidence,
            reasoning_trace=reasoning_trace,
            reasoning_tree_stats=reasoning_tree_stats,
            multi_agent_result=multi_agent_result,
            validation_result=validation_result,
            pitfalls_checked=pitfalls_checked,
            concepts_identified=concepts_identified,
            difficulty_level=complexity_level.value,
            strategies_used=strategies_used,
            total_time=total_time,
            metadata={
                "domain": domain,
                "question_type": question_type,
                "has_options": options is not None
            }
        )
    
    async def _deep_reasoning_solve(
        self,
        question: str,
        domain: Optional[str],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Use deep reasoning tree to solve"""
        try:
            tree = DeepReasoningTree(max_depth=6, max_branches=3)
            builder = ReasoningTreeBuilder(tree)
            
            # Initialize with question
            builder.from_question(question, domain or "")
            
            # This is a simplified version - in practice would involve
            # actual LLM calls to explore reasoning branches
            
            # Get best path
            best_path, stats = builder.finalize()
            
            if best_path and best_path.final_answer:
                return {
                    "answer": best_path.final_answer,
                    "confidence": best_path.total_confidence,
                    "stats": stats
                }
            
            return None
        except Exception as e:
            logger.warning(f"Deep reasoning failed: {e}")
            return None
    
    def _build_prompt(
        self,
        question: str,
        domain: Optional[str],
        template: Optional[Any],
        options: Optional[List[str]]
    ) -> str:
        """Build a prompt for the LLM"""
        prompt_parts = [question]
        
        if template:
            prompt_parts.insert(0, template.generate_prompt(question) if hasattr(template, 'generate_prompt') else "")
        
        if options:
            prompt_parts.append("\nOptions:")
            for i, opt in enumerate(options):
                prompt_parts.append(f"  {chr(65+i)}. {opt}")
        
        return "\n\n".join(prompt_parts)
    
    def _map_complexity_to_difficulty(self, level: ComplexityLevel) -> QuestionDifficulty:
        """Map complexity level to question difficulty"""
        mapping = {
            ComplexityLevel.SIMPLE: QuestionDifficulty.SIMPLE,
            ComplexityLevel.MODERATE: QuestionDifficulty.MODERATE,
            ComplexityLevel.COMPLEX: QuestionDifficulty.HARD,
            ComplexityLevel.VERY_COMPLEX: QuestionDifficulty.HLE_LEVEL,
            ComplexityLevel.HLE_LEVEL: QuestionDifficulty.HLE_LEVEL
        }
        return mapping.get(level, QuestionDifficulty.HLE_LEVEL)
    
    def get_concept_explanation(self, concept: str) -> str:
        """Get explanation for a concept"""
        return self.concept_graph.explain_concept(concept)
    
    def get_concept_contrast(self, concept_a: str, concept_b: str) -> Optional[str]:
        """Get contrast between two concepts"""
        contrast = self.concept_graph.get_contrast(concept_a, concept_b)
        if contrast:
            return contrast.get_comparison_text()
        return None
    
    def check_answer_match(
        self,
        predicted: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Check if predicted answer matches ground truth"""
        result = self.exact_matcher.exact_match_score(predicted, ground_truth)
        return {
            "is_match": result.is_match,
            "score": result.score,
            "match_type": result.match_type,
            "predicted_normalized": result.predicted_normalized,
            "truth_normalized": result.truth_normalized
        }


# Factory function for easy instantiation
def create_hle_qa(
    llm: Any = None,
    strict_mode: bool = False
) -> HLEOptimizedQA:
    """
    Create an HLE-optimized QA system.
    
    Args:
        llm: Language model to use
        strict_mode: Use strict validation
        
    Returns:
        Configured HLEOptimizedQA instance
    """
    return HLEOptimizedQA(
        llm=llm,
        enable_multi_agent=True,
        enable_deep_reasoning=True,
        enable_validation=True,
        strict_mode=strict_mode
    )

