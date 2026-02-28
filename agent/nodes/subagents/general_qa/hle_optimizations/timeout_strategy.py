"""
Adaptive Timeout Strategy for HLE

Provides dynamic timeout allocation based on question complexity:
- Simple questions: Short timeout
- Complex multi-step: Longer timeout
- Very complex/HLE-level: Maximum timeout

Key Features:
- ComplexityEstimator: Estimates question complexity
- AdaptiveTimeoutStrategy: Allocates timeouts dynamically
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for questions"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"
    HLE_LEVEL = "hle_level"


@dataclass
class ComplexityFactors:
    """Factors that contribute to question complexity"""
    length: int = 0
    multi_step: bool = False
    requires_calculation: bool = False
    domain_count: int = 0
    has_subquestions: bool = False
    requires_reasoning: bool = False
    has_constraints: bool = False
    novel_concepts: bool = False
    
    def calculate_score(self) -> int:
        """Calculate complexity score from factors"""
        score = 0
        
        if self.length > 500:
            score += 1
        if self.length > 1000:
            score += 1
        if self.length > 2000:
            score += 1
        
        if self.multi_step:
            score += 1
        if self.requires_calculation:
            score += 1
        if self.domain_count > 1:
            score += 1
        if self.domain_count > 3:
            score += 1
        if self.has_subquestions:
            score += 1
        if self.requires_reasoning:
            score += 1
        if self.has_constraints:
            score += 1
        if self.novel_concepts:
            score += 2
        
        return score


@dataclass
class TimeoutAllocation:
    """Result of timeout allocation"""
    timeout_seconds: float
    complexity_level: ComplexityLevel
    complexity_score: int
    factors: ComplexityFactors
    reasoning: str
    breakdown: Dict[str, float] = field(default_factory=dict)


class ComplexityEstimator:
    """
    Estimates the complexity of a question.
    
    Uses multiple heuristics to determine how complex
    a question is and how much time it might need.
    """
    
    # Multi-step indicators
    MULTI_STEP_PATTERNS = [
        r"step[s]?\s+\d+",
        r"first[,\.].*then",
        r"after\s+.*[,\.].*before",
        r"calculate.*explain",
        r"determine.*verify",
        r"compare.*contrast"
    ]
    
    # Calculation indicators
    CALCULATION_PATTERNS = [
        r"\d+\s*[\+\-\*\/\=]",
        r"calculate|compute|determine the value",
        r"what (is|are) the (ratio|percentage|probability)",
        r"how many|how much",
        r"km|rate|velocity|concentration|molarity",
        r"genotype|phenotype|frequency"
    ]
    
    # Constraint indicators
    CONSTRAINT_PATTERNS = [
        r"assuming that",
        r"given that",
        r"if.*then",
        r"under.*conditions?",
        r"subject to",
        r"limited to",
        r"at most|at least|exactly"
    ]
    
    # Novel/unusual concept indicators
    NOVEL_PATTERNS = [
        r"novel|new|recent|unpublished",
        r"hypothetical|theoretical",
        r"unusual|rare|unique",
        r"first.*to",
        r"previously.*unknown"
    ]
    
    # Domain keywords
    DOMAIN_KEYWORDS = {
        "genetics": ["gene", "allele", "genotype", "phenotype", "inheritance", "chromosome", "mutation"],
        "molecular_biology": ["protein", "rna", "dna", "transcription", "translation", "enzyme", "pathway"],
        "biochemistry": ["reaction", "metabolism", "kinetics", "catalyst", "substrate", "cofactor"],
        "cell_biology": ["cell", "membrane", "organelle", "division", "mitosis", "meiosis"],
        "immunology": ["antibody", "antigen", "immune", "lymphocyte", "mhc", "t cell", "b cell"],
        "clinical": ["diagnosis", "treatment", "patient", "symptom", "disease", "therapy"],
        "evolution": ["selection", "adaptation", "fitness", "drift", "speciation"],
        "ecology": ["population", "community", "ecosystem", "trophic", "niche"]
    }
    
    # Reasoning type patterns
    REASONING_PATTERNS = [
        r"why|explain|reason",
        r"mechanism|how does",
        r"cause|effect",
        r"relationship between",
        r"compare|contrast|distinguish"
    ]
    
    def estimate(
        self,
        question: str,
        question_type: Optional[str] = None,
        domain: Optional[str] = None
    ) -> ComplexityFactors:
        """
        Estimate the complexity factors of a question.
        
        Args:
            question: The question text
            question_type: Optional question type hint
            domain: Optional domain hint
            
        Returns:
            ComplexityFactors with detected complexity indicators
        """
        factors = ComplexityFactors()
        
        # Factor 1: Length
        factors.length = len(question)
        
        # Factor 2: Multi-step detection
        factors.multi_step = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in self.MULTI_STEP_PATTERNS
        )
        
        # Factor 3: Calculation requirement
        factors.requires_calculation = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in self.CALCULATION_PATTERNS
        )
        
        # Factor 4: Domain count
        detected_domains = set()
        question_lower = question.lower()
        for domain_name, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in question_lower for kw in keywords):
                detected_domains.add(domain_name)
        factors.domain_count = len(detected_domains)
        
        # Factor 5: Sub-questions
        factors.has_subquestions = (
            question.count("?") > 1 or
            bool(re.search(r'\d+\.\s+\w', question)) or  # Numbered list
            bool(re.search(r'[a-z]\)\s+\w', question))   # Lettered list
        )
        
        # Factor 6: Reasoning required
        factors.requires_reasoning = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in self.REASONING_PATTERNS
        )
        
        # Factor 7: Constraints
        factors.has_constraints = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in self.CONSTRAINT_PATTERNS
        )
        
        # Factor 8: Novel concepts
        factors.novel_concepts = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in self.NOVEL_PATTERNS
        )
        
        return factors
    
    def get_complexity_level(self, factors: ComplexityFactors) -> ComplexityLevel:
        """Convert complexity factors to a level"""
        score = factors.calculate_score()
        
        if score >= 8:
            return ComplexityLevel.HLE_LEVEL
        elif score >= 6:
            return ComplexityLevel.VERY_COMPLEX
        elif score >= 4:
            return ComplexityLevel.COMPLEX
        elif score >= 2:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE


class AdaptiveTimeoutStrategy:
    """
    Allocates timeouts dynamically based on question complexity.
    
    Timeout tiers:
    - Simple: 30-60 seconds
    - Moderate: 60-120 seconds  
    - Complex: 120-300 seconds
    - Very Complex: 300-600 seconds
    - HLE Level: 600-900 seconds
    """
    
    # Default timeouts by complexity level
    DEFAULT_TIMEOUTS = {
        ComplexityLevel.SIMPLE: 45.0,
        ComplexityLevel.MODERATE: 90.0,
        ComplexityLevel.COMPLEX: 180.0,
        ComplexityLevel.VERY_COMPLEX: 420.0,
        ComplexityLevel.HLE_LEVEL: 720.0
    }
    
    # Maximum timeouts (hard caps)
    MAX_TIMEOUTS = {
        ComplexityLevel.SIMPLE: 60.0,
        ComplexityLevel.MODERATE: 120.0,
        ComplexityLevel.COMPLEX: 300.0,
        ComplexityLevel.VERY_COMPLEX: 600.0,
        ComplexityLevel.HLE_LEVEL: 900.0
    }
    
    # Factor bonuses (additional seconds)
    FACTOR_BONUSES = {
        "multi_step": 30.0,
        "requires_calculation": 45.0,
        "high_domain_count": 20.0,  # >2 domains
        "has_subquestions": 25.0,
        "requires_reasoning": 30.0,
        "has_constraints": 15.0,
        "novel_concepts": 60.0
    }
    
    def __init__(
        self,
        base_timeouts: Optional[Dict[ComplexityLevel, float]] = None,
        max_timeouts: Optional[Dict[ComplexityLevel, float]] = None
    ):
        """
        Initialize the adaptive timeout strategy.
        
        Args:
            base_timeouts: Custom base timeouts
            max_timeouts: Custom max timeouts
        """
        self.base_timeouts = base_timeouts or self.DEFAULT_TIMEOUTS.copy()
        self.max_timeouts = max_timeouts or self.MAX_TIMEOUTS.copy()
        self.complexity_estimator = ComplexityEstimator()
        
        self._allocation_history: List[TimeoutAllocation] = []
    
    def allocate_timeout(
        self,
        question: str,
        question_type: Optional[str] = None,
        domain: Optional[str] = None,
        node_name: Optional[str] = None,
        additional_time: float = 0.0
    ) -> TimeoutAllocation:
        """
        Allocate an appropriate timeout for a question.
        
        Args:
            question: The question text
            question_type: Optional question type hint
            domain: Optional domain hint
            node_name: Optional node name (for node-specific adjustments)
            additional_time: Extra time to add
            
        Returns:
            TimeoutAllocation with timeout and reasoning
        """
        # Estimate complexity
        factors = self.complexity_estimator.estimate(question, question_type, domain)
        complexity_level = self.complexity_estimator.get_complexity_level(factors)
        
        # Get base timeout for this level
        base_timeout = self.base_timeouts[complexity_level]
        max_timeout = self.max_timeouts[complexity_level]
        
        # Calculate bonuses
        bonuses = {}
        total_bonus = 0.0
        
        if factors.multi_step:
            bonus = self.FACTOR_BONUSES["multi_step"]
            bonuses["multi_step"] = bonus
            total_bonus += bonus
        
        if factors.requires_calculation:
            bonus = self.FACTOR_BONUSES["requires_calculation"]
            bonuses["calculation"] = bonus
            total_bonus += bonus
        
        if factors.domain_count > 2:
            bonus = self.FACTOR_BONUSES["high_domain_count"]
            bonuses["domain_count"] = bonus
            total_bonus += bonus
        
        if factors.has_subquestions:
            bonus = self.FACTOR_BONUSES["has_subquestions"]
            bonuses["subquestions"] = bonus
            total_bonus += bonus
        
        if factors.requires_reasoning:
            bonus = self.FACTOR_BONUSES["requires_reasoning"]
            bonuses["reasoning"] = bonus
            total_bonus += bonus
        
        if factors.has_constraints:
            bonus = self.FACTOR_BONUSES["has_constraints"]
            bonuses["constraints"] = bonus
            total_bonus += bonus
        
        if factors.novel_concepts:
            bonus = self.FACTOR_BONUSES["novel_concepts"]
            bonuses["novel"] = bonus
            total_bonus += bonus
        
        # Node-specific adjustments
        node_adjustment = self._get_node_adjustment(node_name)
        if node_adjustment:
            bonuses["node_adjustment"] = node_adjustment
            total_bonus += node_adjustment
        
        # Add any additional time requested
        if additional_time > 0:
            bonuses["additional"] = additional_time
            total_bonus += additional_time
        
        # Calculate final timeout (capped at max)
        final_timeout = min(base_timeout + total_bonus, max_timeout)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            complexity_level, 
            factors, 
            base_timeout,
            total_bonus,
            final_timeout
        )
        
        allocation = TimeoutAllocation(
            timeout_seconds=final_timeout,
            complexity_level=complexity_level,
            complexity_score=factors.calculate_score(),
            factors=factors,
            reasoning=reasoning,
            breakdown={
                "base": base_timeout,
                **bonuses,
                "total": final_timeout
            }
        )
        
        self._allocation_history.append(allocation)
        
        logger.info(
            f"Timeout allocated: {final_timeout:.1f}s "
            f"(complexity: {complexity_level.value}, score: {factors.calculate_score()})"
        )
        
        return allocation
    
    def _get_node_adjustment(self, node_name: Optional[str]) -> float:
        """Get timeout adjustment for specific nodes"""
        if not node_name:
            return 0.0
        
        # Nodes that typically need more time
        heavy_nodes = {
            "n7_complete_inference": 60.0,
            "n3_knowledge_retrieval": 30.0,
            "n4_calculation_decomposition": 45.0,
            "n8_answer_generation": 15.0
        }
        
        return heavy_nodes.get(node_name, 0.0)
    
    def _generate_reasoning(
        self,
        level: ComplexityLevel,
        factors: ComplexityFactors,
        base: float,
        bonus: float,
        final: float
    ) -> str:
        """Generate human-readable reasoning for the allocation"""
        parts = [f"Base timeout for {level.value}: {base:.1f}s"]
        
        if factors.multi_step:
            parts.append("Multi-step reasoning detected")
        if factors.requires_calculation:
            parts.append("Calculation required")
        if factors.domain_count > 1:
            parts.append(f"Cross-domain ({factors.domain_count} domains)")
        if factors.novel_concepts:
            parts.append("Novel concepts involved")
        
        parts.append(f"Total bonus: +{bonus:.1f}s")
        parts.append(f"Final timeout: {final:.1f}s")
        
        return "; ".join(parts)
    
    def get_timeout_for_node(
        self,
        node_name: str,
        question: Optional[str] = None
    ) -> float:
        """
        Get timeout for a specific node.
        
        Args:
            node_name: Name of the node
            question: Optional question for complexity estimation
            
        Returns:
            Timeout in seconds
        """
        # Default node timeouts
        node_timeouts = {
            "n0_input_preprocessing": 30.0,
            "n1_question_decomposition": 60.0,
            "n2_calculation_recognition": 30.0,
            "n3_knowledge_retrieval": 90.0,
            "n4_calculation_decomposition": 120.0,
            "n6_initial_inference": 120.0,
            "n7_complete_inference": 180.0,
            "n8_answer_generation": 60.0,
            "n10_exception_handling": 30.0
        }
        
        base = node_timeouts.get(node_name, 60.0)
        
        if question:
            # Adjust based on question complexity
            allocation = self.allocate_timeout(question)
            
            # Scale node timeout based on overall complexity
            complexity_multiplier = {
                ComplexityLevel.SIMPLE: 0.8,
                ComplexityLevel.MODERATE: 1.0,
                ComplexityLevel.COMPLEX: 1.3,
                ComplexityLevel.VERY_COMPLEX: 1.6,
                ComplexityLevel.HLE_LEVEL: 2.0
            }.get(allocation.complexity_level, 1.0)
            
            return base * complexity_multiplier
        
        return base
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about timeout allocations"""
        if not self._allocation_history:
            return {"total_allocations": 0}
        
        by_level = {}
        for alloc in self._allocation_history:
            level = alloc.complexity_level.value
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(alloc.timeout_seconds)
        
        stats = {
            "total_allocations": len(self._allocation_history),
            "average_timeout": sum(a.timeout_seconds for a in self._allocation_history) / len(self._allocation_history),
            "by_complexity_level": {
                level: {
                    "count": len(timeouts),
                    "average": sum(timeouts) / len(timeouts)
                }
                for level, timeouts in by_level.items()
            }
        }
        
        return stats
    
    def update_from_feedback(
        self,
        question: str,
        actual_time: float,
        was_successful: bool
    ):
        """
        Update strategy based on execution feedback.
        
        This allows the strategy to learn from actual execution times.
        """
        allocation = self.allocate_timeout(question)
        
        # Log the feedback for potential future learning
        logger.info(
            f"Timeout feedback: allocated={allocation.timeout_seconds:.1f}s, "
            f"actual={actual_time:.1f}s, successful={was_successful}"
        )
        
        # Future: Implement actual learning here
        # For now, just log for analysis


def get_adaptive_timeout(
    question: str,
    question_type: Optional[str] = None,
    domain: Optional[str] = None
) -> float:
    """
    Convenience function to get an adaptive timeout.
    
    Args:
        question: The question text
        question_type: Optional question type
        domain: Optional domain
        
    Returns:
        Timeout in seconds
    """
    strategy = AdaptiveTimeoutStrategy()
    allocation = strategy.allocate_timeout(question, question_type, domain)
    return allocation.timeout_seconds

