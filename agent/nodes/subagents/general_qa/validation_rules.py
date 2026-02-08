"""
Logical Contradiction Detection Rules

Rule-based validation system to detect logical contradictions between nodes,
ensuring consistency and accuracy of the reasoning chain.
"""

from typing import Dict, Any, List, Optional, Tuple
import re


class ContradictionDetector:
    """Logical contradiction detector for biomedical QA reasoning chain"""
    
    def __init__(self):
        self.detection_rules = {
            "Judgment": self._check_judgment_contradictions,
            "Calculation": self._check_calculation_contradictions,
            "Analysis": self._check_analysis_contradictions,
            "Enumeration": self._check_enumeration_contradictions
        }
    
    def detect_contradictions(
        self,
        question_type: str,
        experiment_analysis: Dict[str, Any],
        logical_derivation: Dict[str, Any],
        domain_knowledge: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Detect logical contradictions in the reasoning chain
        
        Args:
            question_type: Question type (Judgment/Calculation/Analysis/Enumeration)
            experiment_analysis: Node 3 output (experiment analysis)
            logical_derivation: Node 4 output (logical derivation)
            domain_knowledge: Node 2 output (domain knowledge, optional)
        
        Returns:
            Tuple of (has_contradiction: bool, contradiction_reports: List[str])
        """
        question_type_key = self._get_question_type_key(question_type)
        
        if question_type_key not in self.detection_rules:
            return False, []
        
        check_func = self.detection_rules[question_type_key]
        contradictions = check_func(experiment_analysis, logical_derivation, domain_knowledge)
        
        return len(contradictions) > 0, contradictions
    
    def _get_question_type_key(self, question_type: str) -> str:
        """Map question type string to detection rule key"""
        question_type_lower = str(question_type).lower()
        if "judgment" in question_type_lower:
            return "Judgment"
        elif "calculation" in question_type_lower:
            return "Calculation"
        elif "analysis" in question_type_lower:
            return "Analysis"
        elif "enumeration" in question_type_lower:
            return "Enumeration"
        return "Judgment"  # Default
    
    def _check_judgment_contradictions(
        self,
        experiment_analysis: Dict[str, Any],
        logical_derivation: Dict[str, Any],
        domain_knowledge: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Check contradictions for judgment type questions"""
        contradictions = []
        
        # Rule 1: Impact direction must be consistent with preliminary conclusion
        impact_judgments = experiment_analysis.get("Impact Direction Judgment", [])
        if not isinstance(impact_judgments, list):
            impact_judgments = [impact_judgments] if impact_judgments else []
        
        preliminary_conclusion = logical_derivation.get("Preliminary Conclusion", "").lower()
        
        # Extract impact direction keywords
        impact_keywords = {
            "promote": ["promote", "increase", "enhance", "stimulate", "activate"],
            "inhibit": ["inhibit", "decrease", "reduce", "suppress", "block"],
            "no_effect": ["no effect", "no impact", "unaffected", "unchanged"]
        }
        
        # Check if impact direction matches conclusion
        for impact in impact_judgments:
            impact_lower = str(impact).lower()
            
            # Determine impact direction
            impact_direction = None
            for direction, keywords in impact_keywords.items():
                if any(kw in impact_lower for kw in keywords):
                    impact_direction = direction
                    break
            
            if impact_direction:
                # Check consistency with conclusion
                if impact_direction == "promote" and any(kw in preliminary_conclusion for kw in ["decrease", "reduce", "lower", "inhibit"]):
                    contradictions.append(f"Node 3 impact direction '{impact}' (promotes) contradicts Node 4 conclusion '{preliminary_conclusion}' (suggests decrease)")
                elif impact_direction == "inhibit" and any(kw in preliminary_conclusion for kw in ["increase", "promote", "higher", "enhance"]):
                    contradictions.append(f"Node 3 impact direction '{impact}' (inhibits) contradicts Node 4 conclusion '{preliminary_conclusion}' (suggests increase)")
        
        # Rule 2: Check for bias-related conclusion reversal (Priority 2.2)
        # Extract bias claims from preliminary conclusion
        prelim_bias_claims = self._extract_bias_claims_from_text(preliminary_conclusion)
        
        # Check Option Matching Priority for contradictions
        option_priority = logical_derivation.get("Option Matching Priority", "").lower()
        if option_priority and prelim_bias_claims:
            # Extract which option is ranked highest
            highest_option_match = re.search(r'^([a-e])', option_priority)
            if highest_option_match:
                highest_option = highest_option_match.group(1).upper()
                # Check if the highest-ranked option contradicts preliminary conclusion
                # This is a simplified check - full check would require option text
                # But we can check if option priority mentions contradictions
                if "contradict" in option_priority and highest_option in option_priority.split("contradict")[0]:
                    contradictions.append(f"Node 4 Option Matching Priority ranks '{highest_option}' as highest, but also mentions it contradicts Preliminary Conclusion - this is a logical error")
        
        return contradictions
    
    def _extract_bias_claims_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract bias claims from text (e.g., "θ unbiased, π biased")
        
        Returns:
            Dict mapping item names to bias status ("biased" or "unbiased")
        """
        bias_claims = {}
        text_lower = text.lower()
        
        # Pattern 1: "X unbiased, Y biased"
        pattern1 = r'(\w+)\s+(?:is\s+)?(?:unbiased|not\s+biased|no\s+bias)'
        matches1 = re.findall(pattern1, text_lower)
        for match in matches1:
            bias_claims[match] = "unbiased"
        
        pattern2 = r'(\w+)\s+(?:is\s+)?(?:biased|has\s+bias|underestimated|overestimated)'
        matches2 = re.findall(pattern2, text_lower)
        for match in matches2:
            bias_claims[match] = "biased"
        
        return bias_claims
    
    def _check_calculation_contradictions(
        self,
        experiment_analysis: Dict[str, Any],
        logical_derivation: Dict[str, Any],
        domain_knowledge: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Check contradictions for calculation type questions"""
        contradictions = []
        
        # Rule 1: Formula substitution parameters must match experiment data
        formula_process = logical_derivation.get("Formula Substitution Process", [])
        if not isinstance(formula_process, list):
            formula_process = [formula_process] if formula_process else []
        
        # Extract parameter values from formula process
        param_values = {}
        for step in formula_process:
            step_str = str(step).lower()
            # Look for parameter assignments like "f=0.2" or "f = 0.2"
            matches = re.findall(r'([a-z_]+)\s*[=:]\s*([0-9.]+)', step_str)
            for param, value in matches:
                try:
                    param_values[param] = float(value)
                except ValueError:
                    pass
        
        # Extract parameter values from experiment analysis
        impact_judgments = experiment_analysis.get("Impact Direction Judgment", [])
        if not isinstance(impact_judgments, list):
            impact_judgments = [impact_judgments] if impact_judgments else []
        
        # Look for parameter values in impact judgments (e.g., "f decreases from 0.3 to 0.2")
        for impact in impact_judgments:
            impact_str = str(impact).lower()
            # Look for patterns like "from 0.3 to 0.2" or "f=0.2"
            value_matches = re.findall(r'([a-z_]+)\s*[=:]\s*([0-9.]+)|from\s+([0-9.]+)\s+to\s+([0-9.]+)', impact_str)
            for match in value_matches:
                if match[0] and match[1]:  # param=value format
                    param = match[0]
                    try:
                        exp_value = float(match[1])
                        if param in param_values and abs(param_values[param] - exp_value) > 0.01:
                            contradictions.append(f"Node 4 formula uses {param}={param_values[param]}, but Node 3 experiment data shows {param}={exp_value}, parameter mismatch")
                    except ValueError:
                        pass
                elif match[2] and match[3]:  # from X to Y format
                    try:
                        final_value = float(match[3])
                        # Check if any formula parameter matches this value
                        for param, formula_value in param_values.items():
                            if abs(formula_value - final_value) > 0.01:
                                # This might be a contradiction, but we need more context
                                pass
                    except ValueError:
                        pass
        
        return contradictions
    
    def _check_analysis_contradictions(
        self,
        experiment_analysis: Dict[str, Any],
        logical_derivation: Dict[str, Any],
        domain_knowledge: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Check contradictions for analysis type questions"""
        contradictions = []
        
        # Rule 1: Causal chain must form a closed loop (operation → mechanism → impact)
        causal_chain = logical_derivation.get("Causal Chain Analysis", [])
        if not isinstance(causal_chain, list):
            causal_chain = [causal_chain] if causal_chain else []
        
        operation_breakdown = experiment_analysis.get("Operation Breakdown", [])
        if not isinstance(operation_breakdown, list):
            operation_breakdown = [operation_breakdown] if operation_breakdown else []
        
        # Check if causal chain starts with operations from experiment analysis
        if causal_chain and operation_breakdown:
            first_step = str(causal_chain[0]).lower()
            # Check if first step mentions any operation
            operation_mentioned = any(op.lower() in first_step for op in operation_breakdown if op)
            if not operation_mentioned:
                contradictions.append(f"Causal chain first step '{causal_chain[0]}' does not mention any operation from experiment analysis '{operation_breakdown}'")
        
        # Check if causal chain has logical jumps (missing intermediate steps)
        if len(causal_chain) < 3:
            contradictions.append(f"Causal chain has only {len(causal_chain)} steps, should have at least 3 steps (operation → mechanism → impact)")
        
        return contradictions
    
    def _check_enumeration_contradictions(
        self,
        experiment_analysis: Dict[str, Any],
        logical_derivation: Dict[str, Any],
        domain_knowledge: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Check contradictions for enumeration type questions"""
        contradictions = []
        
        # Rule 1: Enumeration results must have domain knowledge support
        enumeration_results = logical_derivation.get("Enumeration Results", [])
        if not isinstance(enumeration_results, list):
            enumeration_results = [enumeration_results] if enumeration_results else []
        
        if domain_knowledge:
            # Check if enumeration items are mentioned in domain knowledge
            knowledge_text = str(domain_knowledge).lower()
            for item in enumeration_results:
                item_str = str(item).lower()
                # Extract key terms from enumeration item
                key_terms = re.findall(r'\b[a-z]+\b', item_str)
                # Check if any key term appears in domain knowledge
                if key_terms:
                    found = any(term in knowledge_text for term in key_terms if len(term) > 3)
                    if not found:
                        contradictions.append(f"Enumeration item '{item}' has no clear support in domain knowledge")
        
        return contradictions


def detect_logical_contradictions(
    question_type: str,
    experiment_analysis: Dict[str, Any],
    logical_derivation: Dict[str, Any],
    domain_knowledge: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Convenience function to detect logical contradictions
    
    Args:
        question_type: Question type string
        experiment_analysis: Node 3 output
        logical_derivation: Node 4 output
        domain_knowledge: Node 2 output (optional)
    
    Returns:
        Tuple of (has_contradiction: bool, contradiction_reports: List[str])
    """
    detector = ContradictionDetector()
    return detector.detect_contradictions(question_type, experiment_analysis, logical_derivation, domain_knowledge)

