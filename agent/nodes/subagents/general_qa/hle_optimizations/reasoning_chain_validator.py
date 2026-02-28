"""
Reasoning Chain Validator for HLE

Validates reasoning chains for:
- Logical consistency
- Premise-conclusion connection
- Completeness of required steps
- Detection of logical fallacies

Key Features:
- LogicalConnector: Types of logical connections
- ValidationResult: Detailed validation result
- ReasoningChainValidator: Main validation logic
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re


class LogicalConnector(Enum):
    """Types of logical connections between reasoning steps"""
    DEDUCTION = "deduction"      # General to specific
    INDUCTION = "induction"      # Specific to general
    ABDUCTION = "abduction"      # Inference to best explanation
    CAUSAL = "causal"           # Cause and effect
    TEMPORAL = "temporal"        # Time-based sequence
    DEFINITIONAL = "definitional"  # By definition
    ANALOGICAL = "analogical"    # By analogy
    CALCULATION = "calculation"  # Mathematical derivation
    UNKNOWN = "unknown"


class ValidationErrorType(Enum):
    """Types of validation errors"""
    MISSING_PREMISE = "missing_premise"
    LOGICAL_GAP = "logical_gap"
    CIRCULAR_REASONING = "circular_reasoning"
    CONTRADICTION = "contradiction"
    INCOMPLETE_CHAIN = "incomplete_chain"
    IRRELEVANT_STEP = "irrelevant_step"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FALLACY = "fallacy"


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain"""
    step_id: int
    premise: str
    conclusion: str
    connector: LogicalConnector = LogicalConnector.UNKNOWN
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def get_keywords(self) -> Set[str]:
        """Extract keywords from premise and conclusion"""
        text = f"{self.premise} {self.conclusion}"
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return set(words)


@dataclass
class ValidationError:
    """A validation error found in reasoning"""
    error_type: ValidationErrorType
    step_id: Optional[int]  # None for chain-level errors
    description: str
    severity: str  # "critical", "major", "minor"
    suggestion: str


@dataclass
class ValidationResult:
    """Result of validating a reasoning chain"""
    is_valid: bool
    confidence: float
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    missing_steps: List[str] = field(default_factory=list)
    connected_pairs: List[Tuple[int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_critical_errors(self) -> List[ValidationError]:
        return [e for e in self.errors if e.severity == "critical"]
    
    def get_major_errors(self) -> List[ValidationError]:
        return [e for e in self.errors if e.severity == "major"]


class ReasoningChainValidator:
    """
    Validates reasoning chains for logical consistency.
    
    Checks:
    1. Each step is logically connected to the next
    2. Premises support conclusions
    3. Required steps are not missing
    4. No contradictions exist
    5. Evidence is sufficient
    """
    
    # Required steps by question type
    REQUIRED_STEPS = {
        "genetics_calculation": [
            "identify_inheritance_pattern",
            "determine_genotypes",
            "calculate_probabilities",
            "verify_sum_to_one"
        ],
        "mechanism_explanation": [
            "identify_components",
            "describe_process",
            "explain_regulation"
        ],
        "clinical_diagnosis": [
            "extract_findings",
            "generate_differentials",
            "evaluate_differentials",
            "select_diagnosis"
        ],
        "calculation": [
            "identify_parameters",
            "select_formula",
            "perform_calculation",
            "verify_result"
        ]
    }
    
    # Transition keywords that indicate logical connection
    TRANSITION_KEYWORDS = {
        "therefore", "thus", "hence", "so", "consequently",
        "because", "since", "as", "given that",
        "implies", "suggests", "indicates",
        "leads to", "results in", "causes",
        "follows that", "means that"
    }
    
    # Fallacy patterns to detect
    FALLACY_PATTERNS = {
        "circular_reasoning": {
            "pattern": r"(.+?)\s+(?:therefore|thus|so)\s+\1",
            "description": "Conclusion restates the premise"
        },
        "post_hoc": {
            "pattern": r"(.+?)\s+happened (?:after|before)\s+(.+?)\s*,?\s*(?:so|therefore)\s+\2\s+caused\s+\1",
            "description": "Assuming causation from correlation"
        },
        "hasty_generalization": {
            "pattern": r"(?:all|every)\s+\w+\s+(?:are|is)\s+\w+\s+(?:because|since)\s+(?:one|a single|this)",
            "description": "Generalizing from insufficient examples"
        }
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, be more stringent in validation
        """
        self.strict_mode = strict_mode
    
    def validate(
        self,
        reasoning_chain: List[ReasoningStep],
        question_type: Optional[str] = None,
        question: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a reasoning chain.
        
        Args:
            reasoning_chain: List of reasoning steps
            question_type: Type of question (for required step checking)
            question: Original question (for context)
            
        Returns:
            ValidationResult with detailed findings
        """
        errors = []
        warnings = []
        strengths = []
        missing_steps = []
        connected_pairs = []
        
        if not reasoning_chain:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                errors=[ValidationError(
                    error_type=ValidationErrorType.INCOMPLETE_CHAIN,
                    step_id=None,
                    description="Empty reasoning chain",
                    severity="critical",
                    suggestion="Provide reasoning steps"
                )]
            )
        
        # Check 1: Logical connection between consecutive steps
        for i in range(1, len(reasoning_chain)):
            prev_step = reasoning_chain[i-1]
            curr_step = reasoning_chain[i]
            
            is_connected, connection_type = self._check_connection(prev_step, curr_step)
            
            if is_connected:
                connected_pairs.append((prev_step.step_id, curr_step.step_id))
            else:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.LOGICAL_GAP,
                    step_id=curr_step.step_id,
                    description=f"Step {curr_step.step_id} not logically connected to step {prev_step.step_id}",
                    severity="major" if not self.strict_mode else "critical",
                    suggestion=f"Add intermediate reasoning or clarify the connection"
                ))
        
        # Check 2: Premise-conclusion consistency within steps
        for step in reasoning_chain:
            if not self._is_premise_conclusion_consistent(step):
                errors.append(ValidationError(
                    error_type=ValidationErrorType.MISSING_PREMISE,
                    step_id=step.step_id,
                    description=f"Step {step.step_id}: conclusion doesn't follow from premise",
                    severity="major",
                    suggestion="Ensure the conclusion logically follows from the premise"
                ))
        
        # Check 3: Circular reasoning
        conclusions = [s.conclusion.lower() for s in reasoning_chain]
        premises = [s.premise.lower() for s in reasoning_chain]
        
        for i, (conc, prem) in enumerate(zip(conclusions, premises)):
            if conc == prem:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.CIRCULAR_REASONING,
                    step_id=reasoning_chain[i].step_id,
                    description=f"Step {reasoning_chain[i].step_id} appears to be circular",
                    severity="critical",
                    suggestion="Provide independent justification for the conclusion"
                ))
        
        # Check 4: Contradictions
        contradictions = self._find_contradictions(reasoning_chain)
        for i, j in contradictions:
            errors.append(ValidationError(
                error_type=ValidationErrorType.CONTRADICTION,
                step_id=None,
                description=f"Step {reasoning_chain[i].step_id} contradicts step {reasoning_chain[j].step_id}",
                severity="critical",
                suggestion="Resolve the contradiction between these steps"
            ))
        
        # Check 5: Fallacy detection
        chain_text = " ".join(
            f"{s.premise} therefore {s.conclusion}" 
            for s in reasoning_chain
        )
        
        for fallacy_name, fallacy_info in self.FALLACY_PATTERNS.items():
            if re.search(fallacy_info["pattern"], chain_text, re.IGNORECASE):
                errors.append(ValidationError(
                    error_type=ValidationErrorType.FALLACY,
                    step_id=None,
                    description=f"Potential {fallacy_name}: {fallacy_info['description']}",
                    severity="major",
                    suggestion="Review reasoning for logical fallacies"
                ))
        
        # Check 6: Required steps (if question type provided)
        if question_type and question_type in self.REQUIRED_STEPS:
            required = self.REQUIRED_STEPS[question_type]
            chain_content = " ".join(
                f"{s.premise} {s.conclusion}" 
                for s in reasoning_chain
            ).lower()
            
            for req_step in required:
                # Simple check - can be enhanced
                if req_step.replace("_", " ") not in chain_content:
                    missing_steps.append(req_step)
            
            if missing_steps and self.strict_mode:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.INCOMPLETE_CHAIN,
                    step_id=None,
                    description=f"Missing required reasoning steps: {missing_steps}",
                    severity="major",
                    suggestion="Include the missing reasoning steps"
                ))
        
        # Identify strengths
        if len(connected_pairs) == len(reasoning_chain) - 1:
            strengths.append("All reasoning steps are logically connected")
        
        if all(s.evidence for s in reasoning_chain):
            strengths.append("All steps have supporting evidence")
        
        if len(reasoning_chain) >= 3:
            strengths.append("Detailed multi-step reasoning")
        
        # Calculate confidence
        if errors:
            critical_count = len([e for e in errors if e.severity == "critical"])
            major_count = len([e for e in errors if e.severity == "major"])
            confidence = max(0, 1.0 - critical_count * 0.3 - major_count * 0.1)
        else:
            confidence = 1.0
        
        # Apply missing steps penalty
        if missing_steps:
            confidence *= (1 - 0.1 * len(missing_steps))
        
        is_valid = len([e for e in errors if e.severity in ["critical", "major"]]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=min(1.0, max(0.0, confidence)),
            errors=errors,
            warnings=warnings,
            strengths=strengths,
            missing_steps=missing_steps,
            connected_pairs=connected_pairs,
            metadata={
                "total_steps": len(reasoning_chain),
                "connection_rate": len(connected_pairs) / max(1, len(reasoning_chain) - 1)
            }
        )
    
    def _check_connection(
        self, 
        step1: ReasoningStep, 
        step2: ReasoningStep
    ) -> Tuple[bool, str]:
        """
        Check if two steps are logically connected.
        
        Returns:
            Tuple of (is_connected, connection_type)
        """
        # Check 1: Conclusion of step1 appears in premise of step2
        conc_keywords = step1.get_keywords()
        prem_keywords = step2.get_keywords()
        
        overlap = conc_keywords & prem_keywords
        
        if overlap:
            return True, "keyword_overlap"
        
        # Check 2: Direct text overlap
        if step1.conclusion.lower() in step2.premise.lower():
            return True, "conclusion_in_premise"
        
        if step2.premise.lower() in step1.conclusion.lower():
            return True, "premise_in_conclusion"
        
        # Check 3: Transition keywords
        combined = f"{step1.conclusion} {step2.premise}".lower()
        if any(kw in combined for kw in self.TRANSITION_KEYWORDS):
            return True, "transition_keyword"
        
        # Check 4: Numerical connection (calculation → result)
        if step1.connector == LogicalConnector.CALCULATION:
            return True, "calculation_result"
        
        return False, "no_connection"
    
    def _is_premise_conclusion_consistent(self, step: ReasoningStep) -> bool:
        """Check if conclusion follows from premise"""
        if not step.premise or not step.conclusion:
            return False
        
        # Very basic check - if premise and conclusion are completely unrelated
        prem_words = set(step.premise.lower().split())
        conc_words = set(step.conclusion.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further", "then", "once"}
        
        prem_content = prem_words - common_words
        conc_content = conc_words - common_words
        
        if not prem_content or not conc_content:
            return True  # Can't determine
        
        # Check for some overlap
        overlap = prem_content & conc_content
        
        # In strict mode, require more overlap
        min_overlap = 1 if self.strict_mode else 0
        
        return len(overlap) > min_overlap
    
    def _find_contradictions(
        self, 
        chain: List[ReasoningStep]
    ) -> List[Tuple[int, int]]:
        """Find pairs of contradicting steps"""
        contradictions = []
        
        # Simplified contradiction detection
        negation_words = {"not", "no", "never", "neither", "none", "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely"}
        
        for i, step1 in enumerate(chain):
            for j, step2 in enumerate(chain):
                if i >= j:
                    continue
                
                # Check if same topic but with negation
                text1 = f"{step1.premise} {step1.conclusion}".lower()
                text2 = f"{step2.premise} {step2.conclusion}".lower()
                
                # Very simplified - check if one has negation and other doesn't
                # on similar content
                words1 = set(text1.split())
                words2 = set(text2.split())
                
                has_negation1 = bool(words1 & negation_words)
                has_negation2 = bool(words2 & negation_words)
                
                content1 = words1 - negation_words
                content2 = words2 - negation_words
                
                # If similar content but one negated, might be contradiction
                if has_negation1 != has_negation2:
                    overlap = content1 & content2
                    if len(overlap) >= 3:  # Significant overlap
                        contradictions.append((i, j))
        
        return contradictions
    
    def get_required_steps(self, question_type: str) -> List[str]:
        """Get required steps for a question type"""
        return self.REQUIRED_STEPS.get(question_type, [])
    
    def suggest_connections(
        self, 
        chain: List[ReasoningStep]
    ) -> List[Dict[str, Any]]:
        """Suggest improvements for disconnected steps"""
        suggestions = []
        
        for i in range(1, len(chain)):
            prev_step = chain[i-1]
            curr_step = chain[i]
            
            is_connected, _ = self._check_connection(prev_step, curr_step)
            
            if not is_connected:
                # Generate suggestion
                prev_keywords = prev_step.get_keywords()
                curr_keywords = curr_step.get_keywords()
                
                suggestion = {
                    "between_steps": (prev_step.step_id, curr_step.step_id),
                    "suggestion": f"Add reasoning to connect '{prev_step.conclusion[:50]}...' to '{curr_step.premise[:50]}...'",
                    "shared_concepts": list(prev_keywords & curr_keywords) if prev_keywords & curr_keywords else "None found",
                    "recommended_connector": self._suggest_connector(prev_step, curr_step)
                }
                suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_connector(
        self, 
        step1: ReasoningStep, 
        step2: ReasoningStep
    ) -> str:
        """Suggest a logical connector between steps"""
        # Based on step types
        if step2.connector == LogicalConnector.CALCULATION:
            return "Using this value, we can calculate..."
        elif step2.connector == LogicalConnector.DEDUCTION:
            return "Therefore, it follows that..."
        elif step2.connector == LogicalConnector.ABDUCTION:
            return "This suggests that..."
        else:
            return "This leads us to consider..."

