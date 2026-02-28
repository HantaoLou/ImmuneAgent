"""
Confidence Calibrator for HLE

HLE research shows that models exhibit high RMS calibration errors (>70%),
meaning they are overconfident in wrong answers. This module calibrates
confidence scores to be more realistic and helps decide when to express
uncertainty.

Key Features:
- Multi-factor confidence calibration
- Overconfidence penalty for HLE-level questions
- Uncertainty expression decision making
- Reasoning quality assessment integration
"""

import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class QuestionDifficulty(Enum):
    """Difficulty levels for questions"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    HARD = "hard"
    HLE_LEVEL = "hle_level"  # Maximum difficulty
    NOVEL = "novel"  # Original/never-seen-before


class ConfidenceFactor(Enum):
    """Factors that affect confidence calibration"""
    REASONING_COMPLETENESS = "reasoning_completeness"
    KNOWLEDGE_SOURCE_RELIABILITY = "knowledge_source_reliability"
    MULTI_AGENT_AGREEMENT = "multi_agent_agreement"
    CALCULATION_VERIFICATION = "calculation_verification"
    CONCEPT_FAMILIARITY = "concept_familiarity"
    ANSWER_FORMAT_COMPLIANCE = "answer_format_compliance"
    CROSS_VALIDATION = "cross_validation"
    HISTORICAL_ACCURACY = "historical_accuracy"


@dataclass
class UncertaintyExpression:
    """Represents an uncertainty expression for the answer"""
    should_express: bool
    level: str  # "low", "medium", "high"
    reason: str
    suggested_prefix: str
    suggested_suffix: str
    
    # Example prefixes/suffixes
    PREFIX_TEMPLATES = {
        "low": ["I believe", "It appears that", "Based on available information,"],
        "medium": ["I think", "My understanding is", "It seems likely that"],
        "high": ["I'm uncertain about", "I'm not confident about", "There's significant uncertainty regarding"]
    }
    
    SUFFIX_TEMPLATES = {
        "low": ["though verification is recommended", "based on current knowledge"],
        "medium": ["but this requires verification", "however I'm not entirely certain"],
        "high": ["I recommend consulting additional sources", "further investigation is needed"]
    }


@dataclass
class CalibrationResult:
    """Result of confidence calibration"""
    original_confidence: float
    calibrated_confidence: float
    calibration_factors: Dict[str, float]
    adjustments: List[Tuple[str, float, str]]  # (factor, adjustment, reason)
    uncertainty_expression: Optional[UncertaintyExpression]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfidenceCalibrator:
    """
    Calibrates model confidence scores for HLE-level questions.
    
    Research shows that LLMs are systematically overconfident on difficult
    questions. This calibrator adjusts confidence based on multiple factors
    to provide more realistic confidence estimates.
    """
    
    # Default calibration parameters based on HLE research
    DEFAULT_PARAMS = {
        # Maximum confidence cap for HLE-level questions
        "max_confidence_hle": 0.90,
        "max_confidence_hard": 0.95,
        "max_confidence_moderate": 0.98,
        
        # Adjustment factors (multiplied)
        "missing_step_penalty": 0.70,
        "speculation_penalty": 0.50,
        "agent_disagreement_penalty": 0.60,
        "calculation_not_verified_penalty": 0.85,
        "unfamiliar_concept_penalty": 0.75,
        "format_noncompliance_penalty": 0.80,
        
        # Bonus factors
        "cross_validation_bonus": 1.10,
        "multiple_source_agreement_bonus": 1.15,
        "calculation_verified_bonus": 1.05,
        
        # Uncertainty thresholds
        "express_uncertainty_threshold": 0.50,
        "high_uncertainty_threshold": 0.30,
    }
    
    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize the confidence calibrator.
        
        Args:
            params: Custom calibration parameters (overrides defaults)
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._calibration_history: List[CalibrationResult] = []
    
    def calibrate(
        self,
        raw_confidence: float,
        reasoning_quality: Optional[Dict[str, Any]] = None,
        question_difficulty: QuestionDifficulty = QuestionDifficulty.HLE_LEVEL,
        multi_agent_result: Optional[Dict[str, Any]] = None,
        knowledge_sources: Optional[List[str]] = None,
        calculation_verified: Optional[bool] = None
    ) -> CalibrationResult:
        """
        Calibrate confidence score based on multiple factors.
        
        Args:
            raw_confidence: Original confidence score (0.0 to 1.0)
            reasoning_quality: Dict with reasoning quality metrics
            question_difficulty: Estimated difficulty of the question
            multi_agent_result: Results from multi-agent reasoning (if any)
            knowledge_sources: List of knowledge sources used
            calculation_verified: Whether calculations were verified
            
        Returns:
            CalibrationResult with calibrated confidence and details
        """
        reasoning_quality = reasoning_quality or {}
        adjustments = []
        calibration_factors = {}
        
        # Start with raw confidence
        calibrated = raw_confidence
        
        # Factor 1: Apply difficulty-based cap
        max_cap = self._get_max_confidence(question_difficulty)
        if calibrated > max_cap:
            adjustments.append((
                "difficulty_cap",
                max_cap - calibrated,
                f"Cap applied for {question_difficulty.value} question"
            ))
            calibrated = min(calibrated, max_cap)
        
        calibration_factors["difficulty_cap"] = max_cap
        
        # Factor 2: Reasoning completeness
        if reasoning_quality.get("missing_steps"):
            penalty = self.params["missing_step_penalty"]
            adjustment = calibrated * (penalty - 1)
            adjustments.append((
                "missing_steps",
                adjustment,
                f"{len(reasoning_quality['missing_steps'])} missing reasoning steps"
            ))
            calibrated *= penalty
        
        calibration_factors["reasoning_completeness"] = (
            1.0 if not reasoning_quality.get("missing_steps") 
            else self.params["missing_step_penalty"]
        )
        
        # Factor 3: Knowledge source reliability
        source_penalty = self._assess_knowledge_reliability(knowledge_sources)
        if source_penalty < 1.0:
            adjustment = calibrated * (source_penalty - 1)
            adjustments.append((
                "knowledge_source",
                adjustment,
                "Knowledge from speculative or unreliable source"
            ))
            calibrated *= source_penalty
        
        calibration_factors["knowledge_reliability"] = source_penalty
        
        # Factor 4: Multi-agent agreement
        if multi_agent_result:
            agreement_factor = self._assess_agent_agreement(multi_agent_result)
            if agreement_factor < 1.0:
                adjustment = calibrated * (agreement_factor - 1)
                adjustments.append((
                    "agent_disagreement",
                    adjustment,
                    "Agents disagree on the answer"
                ))
                calibrated *= agreement_factor
            elif agreement_factor > 1.0:
                adjustment = calibrated * (agreement_factor - 1)
                adjustments.append((
                    "agent_agreement",
                    adjustment,
                    "All agents agree on the answer"
                ))
                calibrated = min(calibrated * agreement_factor, max_cap)
            
            calibration_factors["multi_agent_agreement"] = agreement_factor
        
        # Factor 5: Calculation verification
        if calculation_verified is not None:
            if calculation_verified:
                bonus = self.params["calculation_verified_bonus"]
                adjustment = calibrated * (bonus - 1)
                adjustments.append((
                    "calculation_verified",
                    adjustment,
                    "Calculation was independently verified"
                ))
                calibrated = min(calibrated * bonus, max_cap)
            else:
                penalty = self.params["calculation_not_verified_penalty"]
                adjustment = calibrated * (penalty - 1)
                adjustments.append((
                    "calculation_unverified",
                    adjustment,
                    "Calculation was not verified"
                ))
                calibrated *= penalty
        
        calibration_factors["calculation_verification"] = (
            self.params["calculation_verified_bonus"] if calculation_verified 
            else (self.params["calculation_not_verified_penalty"] if calculation_verified is False else 1.0)
        )
        
        # Factor 6: Concept familiarity
        if reasoning_quality.get("unfamiliar_concepts"):
            penalty = self.params["unfamiliar_concept_penalty"] ** len(
                reasoning_quality["unfamiliar_concepts"]
            )
            penalty = max(penalty, 0.5)  # Don't penalize too heavily
            adjustment = calibrated * (penalty - 1)
            adjustments.append((
                "unfamiliar_concepts",
                adjustment,
                f"{len(reasoning_quality['unfamiliar_concepts'])} unfamiliar concepts"
            ))
            calibrated *= penalty
        
        # Factor 7: Answer format compliance
        if reasoning_quality.get("format_issues"):
            penalty = self.params["format_noncompliance_penalty"]
            adjustment = calibrated * (penalty - 1)
            adjustments.append((
                "format_noncompliance",
                adjustment,
                "Answer format doesn't fully comply with requirements"
            ))
            calibrated *= penalty
        
        # Factor 8: Cross-validation (if available)
        if reasoning_quality.get("cross_validated"):
            bonus = self.params["cross_validation_bonus"]
            adjustment = calibrated * (bonus - 1)
            adjustments.append((
                "cross_validation",
                adjustment,
                "Answer was cross-validated through multiple approaches"
            ))
            calibrated = min(calibrated * bonus, max_cap)
        
        # Ensure confidence is in valid range
        calibrated = max(0.0, min(1.0, calibrated))
        
        # Determine if uncertainty should be expressed
        uncertainty = self._determine_uncertainty_expression(
            calibrated, question_difficulty, reasoning_quality
        )
        
        result = CalibrationResult(
            original_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_factors=calibration_factors,
            adjustments=adjustments,
            uncertainty_expression=uncertainty,
            metadata={
                "question_difficulty": question_difficulty.value,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        self._calibration_history.append(result)
        return result
    
    def _get_max_confidence(self, difficulty: QuestionDifficulty) -> float:
        """Get maximum allowed confidence for a difficulty level"""
        caps = {
            QuestionDifficulty.SIMPLE: 0.99,
            QuestionDifficulty.MODERATE: self.params["max_confidence_moderate"],
            QuestionDifficulty.HARD: self.params["max_confidence_hard"],
            QuestionDifficulty.HLE_LEVEL: self.params["max_confidence_hle"],
            QuestionDifficulty.NOVEL: 0.85,  # Very conservative for novel problems
        }
        return caps.get(difficulty, self.params["max_confidence_hle"])
    
    def _assess_knowledge_reliability(self, sources: Optional[List[str]]) -> float:
        """
        Assess reliability of knowledge sources.
        
        Returns a factor (0.5 to 1.2) to multiply confidence by.
        """
        if not sources:
            return 0.8  # No sources cited - moderate penalty
        
        # Define source reliability scores
        reliability_scores = {
            "textbook": 1.0,
            "peer_reviewed_paper": 1.0,
            "established_database": 0.95,
            "review_article": 0.95,
            "clinical_guideline": 0.95,
            "preprint": 0.85,
            "website": 0.75,
            "wikipedia": 0.70,
            "speculation": 0.50,
            "internal_reasoning": 0.80,
            "tool_output": 0.90,
        }
        
        scores = []
        for source in sources:
            source_lower = source.lower()
            for key, score in reliability_scores.items():
                if key in source_lower:
                    scores.append(score)
                    break
            else:
                scores.append(0.75)  # Default for unknown sources
        
        # Return average reliability, but penalize if only low-quality sources
        avg_score = sum(scores) / len(scores)
        
        # Bonus for multiple high-quality sources
        high_quality_count = sum(1 for s in scores if s >= 0.95)
        if high_quality_count >= 2:
            avg_score *= self.params["multiple_source_agreement_bonus"]
        
        return min(avg_score, 1.2)
    
    def _assess_agent_agreement(self, multi_agent_result: Dict[str, Any]) -> float:
        """
        Assess multi-agent agreement level.
        
        Returns a factor (0.5 to 1.15) based on agent agreement.
        """
        agents = multi_agent_result.get("agents", [])
        answers = multi_agent_result.get("answers", [])
        
        if not answers:
            return 1.0  # No multi-agent used, no adjustment
        
        # Check answer consistency
        unique_answers = set(str(a).strip().lower() for a in answers if a)
        
        if len(unique_answers) == 1:
            # All agents agree - bonus
            return self.params["multiple_source_agreement_bonus"]
        elif len(unique_answers) == len(answers):
            # All agents disagree - heavy penalty
            return self.params["agent_disagreement_penalty"] * 0.8
        else:
            # Partial agreement - moderate penalty based on agreement ratio
            agreement_ratio = 1 - (len(unique_answers) - 1) / len(answers)
            return 0.7 + 0.3 * agreement_ratio
    
    def _determine_uncertainty_expression(
        self,
        confidence: float,
        difficulty: QuestionDifficulty,
        reasoning_quality: Dict[str, Any]
    ) -> Optional[UncertaintyExpression]:
        """
        Determine if and how to express uncertainty.
        
        HLE research shows that overconfident wrong answers are a major issue.
        This helps decide when to express uncertainty.
        """
        # Threshold for expressing uncertainty
        threshold = self.params["express_uncertainty_threshold"]
        high_uncertainty_threshold = self.params["high_uncertainty_threshold"]
        
        # Lower thresholds for difficult questions
        if difficulty in [QuestionDifficulty.HLE_LEVEL, QuestionDifficulty.NOVEL]:
            threshold += 0.15  # More conservative
            high_uncertainty_threshold += 0.10
        
        # Adjust based on reasoning quality issues
        if reasoning_quality.get("missing_steps") or reasoning_quality.get("unfamiliar_concepts"):
            threshold += 0.10
        
        should_express = confidence < threshold
        
        if not should_express:
            return None
        
        # Determine uncertainty level
        if confidence < high_uncertainty_threshold:
            level = "high"
        elif confidence < threshold:
            level = "medium"
        else:
            level = "low"
        
        # Generate reason
        reasons = []
        if confidence < 0.3:
            reasons.append("very low confidence in the answer")
        if reasoning_quality.get("missing_steps"):
            reasons.append("incomplete reasoning chain")
        if reasoning_quality.get("unfamiliar_concepts"):
            reasons.append("unfamiliar concepts involved")
        if difficulty == QuestionDifficulty.NOVEL:
            reasons.append("this is a novel/unique problem")
        
        reason = "; ".join(reasons) if reasons else "general uncertainty"
        
        # Select appropriate templates
        import random
        prefix = random.choice(UncertaintyExpression.PREFIX_TEMPLATES[level])
        suffix = random.choice(UncertaintyExpression.SUFFIX_TEMPLATES[level])
        
        return UncertaintyExpression(
            should_express=True,
            level=level,
            reason=reason,
            suggested_prefix=prefix,
            suggested_suffix=suffix
        )
    
    def get_calibration_statistics(self) -> Dict[str, Any]:
        """Get statistics about calibrations performed"""
        if not self._calibration_history:
            return {"total_calibrations": 0}
        
        confidences = [c.calibrated_confidence for c in self._calibration_history]
        originals = [c.original_confidence for c in self._calibration_history]
        
        # Count adjustments
        adjustment_counts = {}
        for calibration in self._calibration_history:
            for factor, _, _ in calibration.adjustments:
                adjustment_counts[factor] = adjustment_counts.get(factor, 0) + 1
        
        return {
            "total_calibrations": len(self._calibration_history),
            "average_original_confidence": sum(originals) / len(originals),
            "average_calibrated_confidence": sum(confidences) / len(confidences),
            "average_adjustment": (sum(confidences) - sum(originals)) / len(confidences),
            "adjustment_frequency": adjustment_counts,
            "uncertainty_expressed_count": sum(
                1 for c in self._calibration_history 
                if c.uncertainty_expression and c.uncertainty_expression.should_express
            )
        }
    
    def estimate_question_difficulty(
        self,
        question_text: str,
        question_type: str,
        domain: Optional[str] = None,
        requires_calculation: bool = False,
        multi_step: bool = False
    ) -> QuestionDifficulty:
        """
        Estimate the difficulty level of a question.
        
        This is used to apply appropriate confidence caps.
        """
        difficulty_score = 0
        
        # Length factor
        if len(question_text) > 500:
            difficulty_score += 1
        if len(question_text) > 1000:
            difficulty_score += 1
        
        # Multi-step reasoning
        if multi_step or "step" in question_text.lower():
            difficulty_score += 1
        
        # Calculation required
        if requires_calculation:
            difficulty_score += 1
        
        # Complex domains
        complex_domains = ["quantum", "molecular", "genetics", "biochemistry", "immunology"]
        if domain and any(d in domain.lower() for d in complex_domains):
            difficulty_score += 1
        
        # Novel concepts (heuristics)
        novel_indicators = ["novel", "new", "recent", "unpublished", "hypothetical"]
        if any(indicator in question_text.lower() for indicator in novel_indicators):
            difficulty_score += 2
        
        # Advanced question types
        if question_type in ["proof", "derivation", "synthesis", "design"]:
            difficulty_score += 2
        
        # Map to difficulty level
        if difficulty_score >= 5:
            return QuestionDifficulty.NOVEL
        elif difficulty_score >= 4:
            return QuestionDifficulty.HLE_LEVEL
        elif difficulty_score >= 2:
            return QuestionDifficulty.HARD
        elif difficulty_score >= 1:
            return QuestionDifficulty.MODERATE
        else:
            return QuestionDifficulty.SIMPLE

