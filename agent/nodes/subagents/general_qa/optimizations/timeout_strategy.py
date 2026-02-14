"""
Timeout Strategy - P1 Priority Optimization

Implements dynamic timeout strategy based on question complexity:
1. Simple calculation: 30 seconds
2. Knowledge retrieval: 120 seconds
3. Complex reasoning: 180 seconds
4. Multi-tool queries: 300 seconds
5. Deep research: up to 33 minutes
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class ComplexityLevel(Enum):
    """Question complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class QueryType(Enum):
    """Types of queries"""
    CALCULATION = "calculation"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    MULTI_CHOICE = "multi_choice"
    OPEN_ENDED = "open_ended"
    SEQUENCE_ANALYSIS = "sequence_analysis"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"


@dataclass
class TimeoutConfig:
    """Configuration for timeout values"""
    # Base timeouts (seconds)
    simple_timeout: int = 30
    moderate_timeout: int = 120
    complex_timeout: int = 180
    very_complex_timeout: int = 300
    
    # Special timeouts (seconds)
    calculation_timeout: int = 60
    paperqa_timeout: int = 300  # 5 minutes
    deep_research_timeout: int = 1980  # 33 minutes
    multi_tool_timeout: int = 300
    
    # Timeout multipliers
    paperqa_multiplier: float = 2.5
    deep_research_multiplier: float = 15.0
    multi_tool_multiplier: float = 1.5


# Complexity indicators
COMPLEXITY_INDICATORS = {
    ComplexityLevel.SIMPLE: [
        # Simple calculation keywords
        "calculate", "compute", "determine", "what is", "how many",
        # Simple fact queries
        "what", "which", "name", "list"
    ],
    ComplexityLevel.MODERATE: [
        # Analysis keywords
        "analyze", "compare", "explain", "describe", "summarize",
        # Relationship keywords
        "relationship", "interaction", "association", "between"
    ],
    ComplexityLevel.COMPLEX: [
        # Complex reasoning
        "why", "how does", "mechanism", "pathway", "process",
        # Multi-step
        "first", "then", "step", "sequence", "order",
        # Deep analysis
        "evaluate", "assess", "interpret", "derive"
    ],
    ComplexityLevel.VERY_COMPLEX: [
        # Research-level
        "optimize", "best", "most effective", "novel",
        # Multi-domain
        "integrate", "synthesize", "comprehensive",
        # Open-ended
        "discuss", "propose", "hypothesize", "investigate"
    ],
}

# Query type indicators
QUERY_TYPE_INDICATORS = {
    QueryType.CALCULATION: [
        "calculate", "compute", "bud", "determination", "equation",
        "formula", "ratio", "percentage", "rate", "concentration",
        "mol", "mmol", "μg", "mg", "ml", "dose", "weight"
    ],
    QueryType.KNOWLEDGE_RETRIEVAL: [
        "what is", "define", "explain", "describe", "tell me about",
        "information about", "details on", "characteristics of"
    ],
    QueryType.MULTI_CHOICE: [
        "which of the following", "which option", "select the",
        "choose the", "the correct answer", "option a", "option b"
    ],
    QueryType.SEQUENCE_ANALYSIS: [
        "sequence", "amino acid", "nucleotide", "dna", "rna",
        "protein sequence", "gene sequence", "motif", "domain"
    ],
    QueryType.COMPARISON: [
        "compare", "difference between", "versus", "vs", "better",
        "worse", "superior", "inferior", "advantage", "disadvantage"
    ],
    QueryType.EXPLANATION: [
        "explain", "why", "how", "reason", "mechanism", "cause",
        "principle", "theory", "concept"
    ],
}


@dataclass
class TimeoutStrategyResult:
    """Result of timeout strategy analysis"""
    complexity: ComplexityLevel
    query_type: QueryType
    base_timeout: int
    recommended_timeout: int
    multipliers_applied: List[str] = field(default_factory=list)
    reasoning: str = ""


class TimeoutStrategy:
    """
    Determines appropriate timeout values based on question characteristics
    """
    
    def __init__(self, config: Optional[TimeoutConfig] = None):
        self.config = config or TimeoutConfig()
        self.complexity_indicators = COMPLEXITY_INDICATORS
        self.query_type_indicators = QUERY_TYPE_INDICATORS
    
    def analyze_question(self, question_text: str, 
                        context: Optional[Dict[str, Any]] = None) -> TimeoutStrategyResult:
        """
        Analyze a question and determine appropriate timeout
        
        Args:
            question_text: The question to analyze
            context: Optional context with additional info (e.g., number of tools)
            
        Returns:
            TimeoutStrategyResult with recommended timeout
        """
        # Step 1: Determine complexity level
        complexity = self._determine_complexity(question_text)
        
        # Step 2: Determine query type
        query_type = self._determine_query_type(question_text)
        
        # Step 3: Get base timeout
        base_timeout = self._get_base_timeout(complexity)
        
        # Step 4: Apply multipliers
        timeout, multipliers = self._apply_multipliers(
            base_timeout, query_type, question_text, context
        )
        
        # Step 5: Generate reasoning
        reasoning = self._generate_reasoning(complexity, query_type, timeout, multipliers)
        
        return TimeoutStrategyResult(
            complexity=complexity,
            query_type=query_type,
            base_timeout=base_timeout,
            recommended_timeout=timeout,
            multipliers_applied=multipliers,
            reasoning=reasoning
        )
    
    def _determine_complexity(self, text: str) -> ComplexityLevel:
        """Determine the complexity level of a question"""
        text_lower = text.lower()
        word_count = len(text.split())
        
        scores = {}
        for level, indicators in self.complexity_indicators.items():
            score = sum(1 for ind in indicators if ind in text_lower)
            scores[level] = score
        
        # Factor in length
        if word_count > 100:
            for level in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]:
                scores[level] = scores.get(level, 0) + 1
        
        # Get the level with highest score
        if not any(scores.values()):
            return ComplexityLevel.MODERATE  # Default
        
        max_level = max(scores.keys(), key=lambda k: scores[k])
        
        # If there's a tie, prefer higher complexity
        if scores.get(ComplexityLevel.VERY_COMPLEX, 0) == scores.get(max_level, 0):
            return ComplexityLevel.VERY_COMPLEX
        if scores.get(ComplexityLevel.COMPLEX, 0) == scores.get(max_level, 0):
            return ComplexityLevel.COMPLEX
        
        return max_level
    
    def _determine_query_type(self, text: str) -> QueryType:
        """Determine the type of query"""
        text_lower = text.lower()
        
        scores = {}
        for qtype, indicators in self.query_type_indicators.items():
            score = sum(1 for ind in indicators if ind in text_lower)
            scores[qtype] = score
        
        if not any(scores.values()):
            return QueryType.OPEN_ENDED
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _get_base_timeout(self, complexity: ComplexityLevel) -> int:
        """Get base timeout for a complexity level"""
        timeout_map = {
            ComplexityLevel.SIMPLE: self.config.simple_timeout,
            ComplexityLevel.MODERATE: self.config.moderate_timeout,
            ComplexityLevel.COMPLEX: self.config.complex_timeout,
            ComplexityLevel.VERY_COMPLEX: self.config.very_complex_timeout,
        }
        return timeout_map[complexity]
    
    def _apply_multipliers(self, base_timeout: int, query_type: QueryType,
                           text: str, context: Optional[Dict]) -> Tuple[int, List[str]]:
        """Apply relevant multipliers to base timeout"""
        timeout = base_timeout
        multipliers = []
        
        # Calculation queries get extra time
        if query_type == QueryType.CALCULATION:
            timeout = max(timeout, self.config.calculation_timeout)
            multipliers.append(f"calculation_query (set to {self.config.calculation_timeout}s)")
        
        # Multi-tool queries
        if context and "tool_count" in context:
            tool_count = context["tool_count"]
            if tool_count > 3:
                timeout = int(timeout * self.config.multi_tool_multiplier)
                multipliers.append(f"multi_tool ({tool_count} tools, x{self.config.multi_tool_multiplier})")
        
        # Multi-choice with multiple options
        if query_type == QueryType.MULTI_CHOICE:
            # Count options
            options = len(re.findall(r'\b[A-H][\.\)]', text, re.IGNORECASE))
            if options > 4:
                timeout = int(timeout * 1.3)
                multipliers.append(f"many_options ({options} options, x1.3)")
        
        # Check for keywords that suggest need for PaperQA
        paperqa_keywords = ["paper", "research", "study", "published", "article", "literature"]
        if any(kw in text.lower() for kw in paperqa_keywords):
            timeout = int(timeout * self.config.paperqa_multiplier)
            multipliers.append(f"paperqa_keyword (x{self.config.paperqa_multiplier})")
        
        # Check for deep research triggers
        research_keywords = ["comprehensive", "thorough", "all available", "systematic review"]
        if any(kw in text.lower() for kw in research_keywords):
            timeout = int(timeout * self.config.deep_research_multiplier)
            multipliers.append(f"deep_research (x{self.config.deep_research_multiplier})")
        
        return timeout, multipliers
    
    def _generate_reasoning(self, complexity: ComplexityLevel, 
                           query_type: QueryType, timeout: int,
                           multipliers: List[str]) -> str:
        """Generate human-readable reasoning for the timeout"""
        parts = [
            f"Question complexity: {complexity.value}",
            f"Query type: {query_type.value}",
        ]
        
        if multipliers:
            parts.append("Multipliers applied:")
            for m in multipliers:
                parts.append(f"  - {m}")
        
        parts.append(f"Final timeout: {timeout}s ({timeout/60:.1f} min)")
        
        return "\n".join(parts)
    
    def get_timeout_for_node(self, node_name: str, 
                             question_text: str,
                             context: Optional[Dict] = None) -> int:
        """
        Get timeout specifically for a node
        
        Args:
            node_name: Name of the node (e.g., "n3_knowledge_retrieval")
            question_text: The question being processed
            context: Additional context
            
        Returns:
            Timeout in seconds
        """
        analysis = self.analyze_question(question_text, context)
        timeout = analysis.recommended_timeout
        
        # Node-specific adjustments
        if "knowledge_retrieval" in node_name:
            # Knowledge retrieval can take longer
            timeout = max(timeout, self.config.paperqa_timeout)
        
        elif "inference" in node_name:
            # Inference needs time for reasoning
            timeout = min(timeout, self.config.complex_timeout)
        
        elif "deep_research" in node_name:
            timeout = self.config.deep_research_timeout
        
        return timeout
    
    def get_report(self, question_text: str, 
                   context: Optional[Dict] = None) -> str:
        """Generate a detailed timeout analysis report"""
        analysis = self.analyze_question(question_text, context)
        
        lines = ["# Timeout Strategy Analysis\n"]
        lines.append(f"**Question**: {question_text[:100]}...\n")
        lines.append(f"## Analysis")
        lines.append(f"- Complexity: {analysis.complexity.value}")
        lines.append(f"- Query Type: {analysis.query_type.value}")
        lines.append(f"- Base Timeout: {analysis.base_timeout}s")
        lines.append(f"- Recommended Timeout: {analysis.recommended_timeout}s ({analysis.recommended_timeout/60:.1f} min)")
        
        if analysis.multipliers_applied:
            lines.append("\n## Multipliers Applied")
            for m in analysis.multipliers_applied:
                lines.append(f"- {m}")
        
        lines.append("\n## Reasoning")
        lines.append(analysis.reasoning)
        
        return "\n".join(lines)


# Convenience function
def determine_timeout_strategy(question_text: str,
                               context: Optional[Dict] = None) -> int:
    """
    Quick function to get recommended timeout
    
    Args:
        question_text: The question to analyze
        context: Optional context (e.g., {"tool_count": 5})
        
    Returns:
        Recommended timeout in seconds
    """
    strategy = TimeoutStrategy()
    analysis = strategy.analyze_question(question_text, context)
    return analysis.recommended_timeout


def get_node_timeout(node_name: str, question_text: str,
                     context: Optional[Dict] = None) -> int:
    """
    Quick function to get timeout for a specific node
    
    Returns timeout in seconds
    """
    strategy = TimeoutStrategy()
    return strategy.get_timeout_for_node(node_name, question_text, context)
