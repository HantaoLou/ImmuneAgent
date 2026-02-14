"""
Constraint Extractor - P0 Priority Optimization

Enhanced extraction of constraints from question text:
1. Negative constraints (cannot, except, not)
2. Exclusive constraints (only, category 1)
3. Conditional constraints (if...then)
4. Comparative constraints (better, best, optimal)
5. Quantitative constraints (numbers, ranges)
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class ConstraintType(Enum):
    """Types of constraints"""
    NEGATIVE = "negative"           # Cannot, except, not allowed
    EXCLUSIVE = "exclusive"         # Only one, category 1
    CONDITIONAL = "conditional"     # If...then, when
    COMPARATIVE = "comparative"     # Best, optimal, most
    QUANTITATIVE = "quantitative"   # Numbers, ranges
    TEMPORAL = "temporal"           # Time-based
    SPATIAL = "spatial"             # Location-based
    LOGICAL = "logical"             # AND, OR, NOT logic


@dataclass
class Constraint:
    """Represents a single constraint"""
    constraint_type: ConstraintType
    original_text: str
    extracted_value: str
    negated_entities: List[str] = field(default_factory=list)
    required_entities: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    consequence: Optional[str] = None
    confidence: float = 1.0
    source_sentence: str = ""


@dataclass
class ConstraintAnalysisResult:
    """Result of constraint analysis"""
    constraints: List[Constraint] = field(default_factory=list)
    negative_constraints: List[Constraint] = field(default_factory=list)
    exclusive_constraints: List[Constraint] = field(default_factory=list)
    conditional_constraints: List[Constraint] = field(default_factory=list)
    comparative_constraints: List[Constraint] = field(default_factory=list)
    quantitative_constraints: List[Constraint] = field(default_factory=list)
    
    # Summary for quick access
    all_negated_entities: Set[str] = field(default_factory=set)
    all_required_entities: Set[str] = field(default_factory=set)
    key_constraints: List[str] = field(default_factory=list)


# Comprehensive constraint patterns
CONSTRAINT_PATTERNS = {
    ConstraintType.NEGATIVE: [
        # Cannot patterns
        (r"cannot\s+(?:be\s+)?(?:classified|categorized|considered)\s+as\s+(\w+)", "cannot_be_classified"),
        (r"cannot\s+(\w+(?:\s+\w+){0,5})", "cannot_action"),
        (r"can\s+not\s+(\w+(?:\s+\w+){0,5})", "cannot_action"),
        (r"is\s+unable\s+to\s+(\w+(?:\s+\w+){0,5})", "unable_to"),
        
        # Except patterns
        (r"except\s+(?:for\s+)?(\w+(?:\s+\w+){0,5})", "except"),
        (r"with\s+the\s+exception\s+of\s+(\w+(?:\s+\w+){0,5})", "except"),
        (r"excluding\s+(\w+(?:\s+\w+){0,5})", "exclude"),
        
        # Not patterns
        (r"does\s+not\s+(\w+(?:\s+\w+){0,5})", "does_not"),
        (r"is\s+not\s+(\w+(?:\s+\w+){0,5})", "is_not"),
        (r"are\s+not\s+(\w+(?:\s+\w+){0,5})", "are_not"),
        (r"will\s+not\s+(\w+(?:\s+\w+){0,5})", "will_not"),
        (r"should\s+not\s+(\w+(?:\s+\w+){0,5})", "should_not"),
        (r"must\s+not\s+(\w+(?:\s+\w+){0,5})", "must_not"),
        (r"not\s+(?:allowed|permitted|possible)\s+to\s+(\w+(?:\s+\w+){0,5})", "not_allowed"),
        
        # Never patterns
        (r"never\s+(\w+(?:\s+\w+){0,5})", "never"),
        
        # Exclude patterns
        (r"exclude[sd]?\s+(\w+(?:\s+\w+){0,5})", "exclude"),
        (r"excluding\s+(\w+(?:\s+\w+){0,5})", "exclude"),
        
        # Impossible patterns
        (r"impossible\s+(?:to\s+)?(\w+(?:\s+\w+){0,5})", "impossible"),
        (r"it\s+is\s+impossible\s+(?:to\s+)?(\w+(?:\s+\w+){0,5})", "impossible"),
        
        # No patterns
        (r"no\s+(\w+(?:\s+\w+){0,3})\s+(?:is|are|can|should|will)", "no_entity"),
        (r"there\s+is\s+no\s+(\w+(?:\s+\w+){0,5})", "no_entity"),
    ],
    
    ConstraintType.EXCLUSIVE: [
        # Only patterns
        (r"only\s+(?:one\s+)?(\w+(?:\s+\w+){0,5})(?:\s+is|\s+can|\s+should)?", "only"),
        (r"the\s+only\s+(\w+(?:\s+\w+){0,5})", "the_only"),
        (r"is\s+the\s+only\s+(\w+(?:\s+\w+){0,5})", "is_only"),
        
        # Category patterns
        (r"category\s+(\d+)[:\s]+(\w+(?:\s+\w+){0,5})", "category"),
        (r"categories?\s+(\d+(?:\s+and\s+\d+)*)[:\s]+(\w+(?:\s+\w+){0,5})", "category"),
        
        # Single patterns
        (r"single\s+(\w+(?:\s+\w+){0,5})", "single"),
        (r"a\s+single\s+(\w+(?:\s+\w+){0,5})", "single"),
        
        # Unique patterns
        (r"unique\s+(\w+(?:\s+\w+){0,5})", "unique"),
        (r"is\s+unique\s+to\s+(\w+(?:\s+\w+){0,5})", "unique_to"),
        
        # Exclusively patterns
        (r"exclusively\s+(\w+(?:\s+\w+){0,5})", "exclusively"),
        (r"is\s+exclusively\s+(\w+(?:\s+\w+){0,5})", "is_exclusively"),
        
        # Solely patterns
        (r"solely\s+(\w+(?:\s+\w+){0,5})", "solely"),
        (r"is\s+solely\s+(\w+(?:\s+\w+){0,5})", "is_solely"),
        
        # Merely patterns
        (r"merely\s+(\w+(?:\s+\w+){0,5})", "merely"),
        
        # One and only patterns
        (r"the\s+one\s+and\s+only\s+(\w+(?:\s+\w+){0,5})", "one_and_only"),
    ],
    
    ConstraintType.CONDITIONAL: [
        # If...then patterns
        (r"if\s+([^,.]+),?\s+then\s+([^,.]+)", "if_then"),
        (r"if\s+([^,.]+),?\s+([^,.]+)\s+will", "if_will"),
        
        # When patterns
        (r"when\s+([^,.]+),?\s+then?\s*([^,.]+)", "when_then"),
        (r"when\s+([^,.]+),?\s+([^,.]+)", "when"),
        
        # Given patterns
        (r"given\s+(?:that\s+)?([^,.]+),?\s+([^,.]+)", "given"),
        (r"provided\s+(?:that\s+)?([^,.]+),?\s+([^,.]+)", "provided"),
        
        # Unless patterns
        (r"unless\s+([^,.]+)", "unless"),
        
        # In case patterns
        (r"in\s+case\s+(?:of\s+)?([^,.]+),?\s+([^,.]+)", "in_case"),
        
        # Only if patterns
        (r"only\s+if\s+([^,.]+)", "only_if"),
    ],
    
    ConstraintType.COMPARATIVE: [
        # Best patterns
        (r"(?:the\s+)?best\s+(?:way|method|approach)\s+(?:to\s+)?(\w+(?:\s+\w+){0,5})", "best_way"),
        (r"(?:the\s+)?best\s+(\w+(?:\s+\w+){0,5})", "best"),
        (r"optimally?\s+(\w+(?:\s+\w+){0,5})", "optimal"),
        (r"most\s+effective\s+(\w+(?:\s+\w+){0,5})", "most_effective"),
        
        # Better patterns
        (r"better\s+(?:than\s+)?(\w+(?:\s+\w+){0,5})", "better"),
        (r"superior\s+(?:to\s+)?(\w+(?:\s+\w+){0,5})", "superior"),
        (r"prefer(?:red|ably)?\s+(?:over\s+)?(\w+(?:\s+\w+){0,5})", "preferred"),
        
        # Dominant patterns
        (r"(?:which\s+)?(\w+)\s+(?:is\s+)?(?:the\s+)?dominant\s+(\w+)", "dominant"),
        (r"dominates?\s+(\w+(?:\s+\w+){0,5})", "dominates"),
        
        # Primary patterns
        (r"(?:the\s+)?primary\s+(\w+(?:\s+\w+){0,5})", "primary"),
        (r"(?:the\s+)?main\s+(\w+(?:\s+\w+){0,5})", "main"),
        (r"(?:the\s+)?key\s+(\w+(?:\s+\w+){0,5})", "key"),
        (r"(?:the\s+)?major\s+(\w+(?:\s+\w+){0,5})", "major"),
    ],
    
    ConstraintType.QUANTITATIVE: [
        # Number patterns
        (r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(\w+)", "range"),
        (r"(?:at\s+least|minimum)\s+(\d+(?:\.\d+)?)\s*(\w+)", "minimum"),
        (r"(?:at\s+most|maximum)\s+(\d+(?:\.\d+)?)\s*(\w+)", "maximum"),
        (r"exactly\s+(\d+(?:\.\d+)?)\s*(\w+)", "exactly"),
        (r"approximately\s+(\d+(?:\.\d+)?)\s*(\w+)", "approximately"),
        
        # Percentage patterns
        (r"(\d+(?:\.\d+)?)\s*%", "percentage"),
        (r"(\d+(?:\.\d+)?)\s*percent", "percentage"),
        
        # Frequency patterns
        (r"(\d+(?:\.\d+)?)\s*(?:times?|x)\s+(?:per|every)\s+(\w+)", "frequency"),
        
        # Order patterns
        (r"(\d+(?:st|nd|rd|th))\s+(\w+)", "ordinal"),
        (r"first\s+(\w+)", "first"),
        (r"last\s+(\w+)", "last"),
    ],
    
    ConstraintType.LOGICAL: [
        # AND patterns
        (r"both\s+(\w+)\s+and\s+(\w+)", "both_and"),
        (r"(\w+)\s+and\s+(\w+)\s+(?:are|is|should|must|can)", "and"),
        
        # OR patterns
        (r"either\s+(\w+)\s+or\s+(\w+)", "either_or"),
        (r"(\w+)\s+or\s+(\w+)", "or"),
        
        # NOT patterns
        (r"not\s+both\s+(\w+)\s+and\s+(\w+)", "not_both"),
        (r"neither\s+(\w+)\s+nor\s+(\w+)", "neither_nor"),
    ],
}


class ConstraintExtractor:
    """
    Enhanced constraint extraction from question text
    """
    
    def __init__(self):
        self.patterns = CONSTRAINT_PATTERNS
    
    def extract_all_constraints(self, text: str) -> ConstraintAnalysisResult:
        """
        Extract all types of constraints from text
        
        Args:
            text: Question or context text
            
        Returns:
            ConstraintAnalysisResult with all extracted constraints
        """
        result = ConstraintAnalysisResult()
        
        # Split into sentences for better context
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            # Extract constraints by type
            self._extract_constraints_from_sentence(sentence, result)
        
        # Post-process to identify key constraints
        self._identify_key_constraints(result)
        
        # Categorize constraints
        result.negative_constraints = [c for c in result.constraints if c.constraint_type == ConstraintType.NEGATIVE]
        result.exclusive_constraints = [c for c in result.constraints if c.constraint_type == ConstraintType.EXCLUSIVE]
        result.conditional_constraints = [c for c in result.constraints if c.constraint_type == ConstraintType.CONDITIONAL]
        result.comparative_constraints = [c for c in result.constraints if c.constraint_type == ConstraintType.COMPARATIVE]
        result.quantitative_constraints = [c for c in result.constraints if c.constraint_type == ConstraintType.QUANTITATIVE]
        
        # Aggregate entities
        for c in result.constraints:
            result.all_negated_entities.update(c.negated_entities)
            result.all_required_entities.update(c.required_entities)
        
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = re.sub(r'\b(?:e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|Prof)\.', lambda m: m.group().replace('.', '@'), text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('@', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_constraints_from_sentence(self, sentence: str, result: ConstraintAnalysisResult):
        """Extract constraints from a single sentence"""
        for constraint_type, patterns in self.patterns.items():
            for pattern, pattern_name in patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                
                for match in matches:
                    constraint = self._create_constraint_from_match(
                        constraint_type, match, pattern_name, sentence
                    )
                    if constraint:
                        result.constraints.append(constraint)
    
    def _create_constraint_from_match(self, constraint_type: ConstraintType,
                                       match: re.Match, pattern_name: str,
                                       sentence: str) -> Optional[Constraint]:
        """Create a Constraint object from a regex match"""
        groups = match.groups()
        
        if not groups:
            return None
        
        constraint = Constraint(
            constraint_type=constraint_type,
            original_text=match.group(0),
            extracted_value=groups[0] if groups else "",
            source_sentence=sentence,
            confidence=0.9  # High confidence for pattern matches
        )
        
        # Process based on constraint type
        if constraint_type == ConstraintType.NEGATIVE:
            constraint.negated_entities = self._extract_entities(groups[0])
        
        elif constraint_type == ConstraintType.EXCLUSIVE:
            constraint.required_entities = self._extract_entities(groups[0])
            if len(groups) > 1 and groups[1]:
                constraint.required_entities.extend(self._extract_entities(groups[1]))
        
        elif constraint_type == ConstraintType.CONDITIONAL:
            constraint.condition = groups[0] if groups else None
            if len(groups) > 1:
                constraint.consequence = groups[1]
        
        elif constraint_type == ConstraintType.COMPARATIVE:
            constraint.required_entities = self._extract_entities(groups[0])
        
        elif constraint_type == ConstraintType.QUANTITATIVE:
            constraint.extracted_value = " ".join(str(g) for g in groups if g)
        
        return constraint
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity-like terms from text"""
        # Remove common stop words
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "to", "of",
                      "in", "for", "on", "with", "at", "by", "from", "as", "into"}
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Filter stop words and return
        return [w for w in words if w not in stop_words]
    
    def _identify_key_constraints(self, result: ConstraintAnalysisResult):
        """Identify which constraints are most important"""
        key_constraints = []
        
        # Negative constraints are always key
        for c in result.constraints:
            if c.constraint_type == ConstraintType.NEGATIVE:
                key_constraints.append(f"NEGATIVE: {c.original_text}")
        
        # Exclusive constraints are always key
        for c in result.constraints:
            if c.constraint_type == ConstraintType.EXCLUSIVE:
                key_constraints.append(f"EXCLUSIVE: {c.original_text}")
        
        # Comparative constraints with "best" or "dominant" are key
        for c in result.constraints:
            if c.constraint_type == ConstraintType.COMPARATIVE:
                if any(word in c.original_text.lower() for word in ["best", "dominant", "primary", "key"]):
                    key_constraints.append(f"COMPARATIVE: {c.original_text}")
        
        result.key_constraints = key_constraints
    
    def get_constraint_summary(self, text: str) -> str:
        """Generate a human-readable constraint summary"""
        result = self.extract_all_constraints(text)
        
        lines = ["# Constraint Analysis Summary\n"]
        
        if result.negative_constraints:
            lines.append("## Negative Constraints (MUST NOT)")
            for c in result.negative_constraints:
                lines.append(f"- {c.original_text}")
                if c.negated_entities:
                    lines.append(f"  - Negated: {', '.join(c.negated_entities)}")
        
        if result.exclusive_constraints:
            lines.append("\n## Exclusive Constraints (ONLY)")
            for c in result.exclusive_constraints:
                lines.append(f"- {c.original_text}")
                if c.required_entities:
                    lines.append(f"  - Required: {', '.join(c.required_entities)}")
        
        if result.conditional_constraints:
            lines.append("\n## Conditional Constraints")
            for c in result.conditional_constraints:
                lines.append(f"- IF {c.condition} THEN {c.consequence}")
        
        if result.comparative_constraints:
            lines.append("\n## Comparative Constraints")
            for c in result.comparative_constraints:
                lines.append(f"- {c.original_text}")
        
        if result.key_constraints:
            lines.append("\n## Key Constraints (Summary)")
            for kc in result.key_constraints:
                lines.append(f"- {kc}")
        
        return "\n".join(lines)
    
    def check_option_against_constraints(self, option_text: str,
                                          result: ConstraintAnalysisResult) -> Tuple[bool, List[str]]:
        """
        Check if an option violates any constraints
        
        Returns (is_valid, list_of_violations)
        """
        violations = []
        option_lower = option_text.lower()
        
        # Check negative constraints
        for c in result.negative_constraints:
            for entity in c.negated_entities:
                if entity in option_lower:
                    violations.append(f"Violates negative constraint: contains '{entity}' which is negated in '{c.original_text}'")
        
        # Check exclusive constraints
        # If there's an exclusive constraint, the option must contain the required entity
        for c in result.exclusive_constraints:
            if c.required_entities:
                has_required = any(entity in option_lower for entity in c.required_entities)
                if not has_required:
                    violations.append(f"Missing exclusive requirement: option should contain one of {c.required_entities}")
        
        is_valid = len(violations) == 0
        return is_valid, violations


# Convenience function
def extract_all_constraints(text: str) -> Dict[str, Any]:
    """
    Quick function to extract constraints
    
    Returns dict with constraint information
    """
    extractor = ConstraintExtractor()
    result = extractor.extract_all_constraints(text)
    
    return {
        "constraints": [
            {
                "type": c.constraint_type.value,
                "original": c.original_text,
                "value": c.extracted_value,
                "negated_entities": c.negated_entities,
                "required_entities": c.required_entities,
                "condition": c.condition,
                "consequence": c.consequence
            }
            for c in result.constraints
        ],
        "key_constraints": result.key_constraints,
        "all_negated_entities": list(result.all_negated_entities),
        "all_required_entities": list(result.all_required_entities),
        "negative_count": len(result.negative_constraints),
        "exclusive_count": len(result.exclusive_constraints),
        "conditional_count": len(result.conditional_constraints)
    }

