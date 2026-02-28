"""
Error Analyzer - Analyzes errors to extract knowledge gaps and reasoning traps

This module analyzes WRONG answers to:
1. Classify the type of error (concept, logic, calculation, knowledge gap)
2. Identify what knowledge was missing or misunderstood
3. Identify reasoning traps that led to the wrong answer
4. Generate guidance for future attempts

Key Principle:
We DON'T just remember "A was wrong, so answer B".
Instead, we analyze WHY A was wrong and WHAT knowledge/reasoning is needed.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re

# 支持两种导入路径（测试环境使用 nodes.xxx，运行环境使用 agent.nodes.xxx）
try:
    from agent.nodes.subagents.general_qa.answer_cache import (
        ErrorCategory,
        ErrorAnalysisCache,
        cache_error_analysis,
    )
except ImportError:
    from nodes.subagents.general_qa.answer_cache import (
        ErrorCategory,
        ErrorAnalysisCache,
        cache_error_analysis,
    )


# ========== Error Classification Patterns ==========

# Patterns that indicate concept confusion
CONCEPT_CONFUSION_PATTERNS = {
    # Genetics
    ("fst", "dxy"): "Fst vs Dxy - Fst measures population differentiation due to structure, Dxy measures absolute sequence divergence",
    ("gene_flow", "genetic_drift"): "Gene flow vs Genetic drift - gene flow homogenizes populations, drift increases divergence randomly",
    ("hardy_weinberg", "linkage_disequilibrium"): "HWE vs LD - HWE is about single locus allele frequencies, LD is about multi-locus associations",
    
    # Molecular Biology
    ("attenuation", "repression"): "Attenuation vs Repression - attenuation terminates transcription early, repression prevents initiation",
    ("activator", "enhancer"): "Activator vs Enhancer - activators are proteins, enhancers are DNA sequences",
    ("promoter", "operator"): "Promoter vs Operator - promoter is for transcription initiation, operator is for repressor binding",
    
    # Enzyme Kinetics
    ("competitive", "non_competitive"): "Competitive vs Non-competitive inhibition - competitive affects Km, non-competitive affects Vmax",
    ("km", "vmax"): "Km vs Vmax - Km is substrate concentration at half-max velocity, Vmax is maximum velocity",
    
    # Protein Structure
    ("alpha_helix", "beta_sheet"): "Alpha helix vs Beta sheet - helix is right-handed coiled, sheet is extended strands",
    ("primary", "secondary"): "Primary vs Secondary structure - primary is amino acid sequence, secondary is local folding patterns",
    
    # Immunology
    ("mhc_i", "mhc_ii"): "MHC I vs MHC II - MHC I presents to CD8+ T cells (endogenous), MHC II to CD4+ T cells (exogenous)",
    ("antibody", "antigen"): "Antibody vs Antigen - antibody is produced by immune system, antigen is what it binds to",
}

# Error type indicator keywords
ERROR_INDICATORS = {
    ErrorCategory.CONCEPT_ERROR: [
        "confused", "misunderstood", "mixed up", "thought it was",
        "incorrect definition", "wrong concept", "concept error",
    ],
    ErrorCategory.LOGIC_ERROR: [
        "therefore", "thus", "so", "because", "implies",
        "if then", "should be", "must be", "reasoning",
    ],
    ErrorCategory.CALCULATION_ERROR: [
        "calculate", "compute", "formula", "equation",
        "result", "value", "number", "count",
    ],
    ErrorCategory.KNOWLEDGE_GAP: [
        "did not know", "missing", "lacking", "no information",
        "not found", "unknown", "unaware",
    ],
    ErrorCategory.MISINTERPRETATION: [
        "misread", "misunderstood the question", "thought it asked",
        "interpreted as", "assumed", "overlooked",
    ],
    ErrorCategory.OVERSIMPLIFICATION: [
        "simplified", "ignored", "assumed", "overlooked",
        "didn't consider", "neglected", "generalized",
    ],
}


# ========== Error Classifier ==========

class ErrorClassifier:
    """
    Classifies errors based on the answer and reasoning
    
    Uses multiple signals:
    1. Content analysis of wrong answer vs correct answer
    2. Reasoning path analysis
    3. Domain-specific patterns
    """
    
    def __init__(self):
        # Common confused concept pairs
        self.confusion_pairs = CONCEPT_CONFUSION_PATTERNS
        
        # Error indicators
        self.error_indicators = ERROR_INDICATORS
    
    def classify_error(self,
                       question: str,
                       wrong_answer: str,
                       correct_answer: str,
                       reasoning_path: Optional[List[str]] = None,
                       key_knowledge: Optional[List[str]] = None,
                       domain: str = "") -> Tuple[ErrorCategory, str, List[str]]:
        """
        Classify the type of error made
        
        Args:
            question: The original question
            wrong_answer: The wrong answer given
            correct_answer: The correct answer
            reasoning_path: The reasoning steps taken
            key_knowledge: Knowledge used in reasoning
            domain: Problem domain
            
        Returns:
            (error_category, error_description, confused_concepts)
        """
        confused_concepts = []
        error_description = ""
        
        # Check for concept confusion
        confused_concepts = self._detect_concept_confusion(
            question, wrong_answer, correct_answer, key_knowledge
        )
        
        # Check error type based on question content
        error_type = self._detect_error_type(
            question, wrong_answer, correct_answer, reasoning_path
        )
        
        # Generate error description
        error_description = self._generate_error_description(
            error_type, wrong_answer, correct_answer, confused_concepts, reasoning_path
        )
        
        return error_type, error_description, confused_concepts
    
    def _detect_concept_confusion(self,
                                  question: str,
                                  wrong_answer: str,
                                  correct_answer: str,
                                  key_knowledge: Optional[List[str]]) -> List[str]:
        """Detect if there was concept confusion"""
        confused = []
        
        # Combine all text for analysis
        all_text = f"{question} {wrong_answer} {correct_answer}"
        if key_knowledge:
            all_text += " " + " ".join(key_knowledge)
        all_text_lower = all_text.lower()
        
        # Check for known confusion pairs
        for (concept1, concept2), explanation in self.confusion_pairs.items():
            # Normalize concept names for matching
            c1_patterns = [concept1, concept1.replace("_", " ")]
            c2_patterns = [concept2, concept2.replace("_", " ")]
            
            c1_found = any(p in all_text_lower for p in c1_patterns)
            c2_found = any(p in all_text_lower for p in c2_patterns)
            
            if c1_found and c2_found:
                confused.append(f"{concept1} vs {concept2}")
            elif c1_found or c2_found:
                # Check if the answer choice indicates confusion
                if wrong_answer and correct_answer:
                    # If answer changed between options mentioning these concepts
                    pass  # Complex analysis would go here
        
        return confused
    
    def _detect_error_type(self,
                           question: str,
                           wrong_answer: str,
                           correct_answer: str,
                           reasoning_path: Optional[List[str]]) -> ErrorCategory:
        """Detect the type of error based on content"""
        
        # Check for calculation indicators
        calc_patterns = [
            r'\d+\s*[\+\-\*\/\=]\s*\d+',  # Math expressions
            r'how many', r'how much', r'calculate', r'compute',
            r'what is the value', r'what is the number',
            r'percentage', r'fraction', r'ratio', r'rate',
        ]
        question_lower = question.lower()
        for pattern in calc_patterns:
            if re.search(pattern, question_lower):
                # Check if answers are numeric
                try:
                    float(wrong_answer.replace(',', '').replace('%', ''))
                    float(correct_answer.replace(',', '').replace('%', ''))
                    return ErrorCategory.CALCULATION_ERROR
                except ValueError:
                    pass
        
        # Check for logic indicators
        logic_patterns = [
            r'if.*then', r'therefore', r'because', r'should',
            r'would', r'could', r'must be', r'necessarily',
            r'which of the following', r'all of the following except',
        ]
        for pattern in logic_patterns:
            if re.search(pattern, question_lower):
                return ErrorCategory.LOGIC_ERROR
        
        # Check reasoning path for error indicators
        if reasoning_path:
            reasoning_text = " ".join(reasoning_path).lower()
            
            for error_type, indicators in self.error_indicators.items():
                for indicator in indicators:
                    if indicator in reasoning_text:
                        return error_type
        
        # Check for knowledge gap indicators
        gap_patterns = [
            r'not mentioned', r'not discussed', r'no data',
            r'insufficient', r'lack of',
        ]
        for pattern in gap_patterns:
            if re.search(pattern, question_lower):
                return ErrorCategory.KNOWLEDGE_GAP
        
        # Default to concept error for MCQ
        if re.search(r'[A-Ea-e]\.', question) or len(wrong_answer) <= 2:
            return ErrorCategory.CONCEPT_ERROR
        
        return ErrorCategory.UNKNOWN
    
    def _generate_error_description(self,
                                    error_type: ErrorCategory,
                                    wrong_answer: str,
                                    correct_answer: str,
                                    confused_concepts: List[str],
                                    reasoning_path: Optional[List[str]]) -> str:
        """Generate a human-readable error description"""
        
        descriptions = []
        
        # Basic error type description
        type_descriptions = {
            ErrorCategory.CONCEPT_ERROR: "Concept understanding error",
            ErrorCategory.LOGIC_ERROR: "Reasoning logic error",
            ErrorCategory.CALCULATION_ERROR: "Calculation or numerical error",
            ErrorCategory.KNOWLEDGE_GAP: "Missing required knowledge",
            ErrorCategory.MISINTERPRETATION: "Question misinterpretation",
            ErrorCategory.OVERSIMPLIFICATION: "Over-simplified the problem",
            ErrorCategory.CONFUSION: "Confused related concepts",
            ErrorCategory.UNKNOWN: "Unknown error type",
        }
        
        descriptions.append(type_descriptions.get(error_type, "Error"))
        
        # Add answer comparison
        if wrong_answer and correct_answer:
            descriptions.append(f"Answered '{wrong_answer}' but correct is '{correct_answer}'")
        
        # Add confusion info
        if confused_concepts:
            descriptions.append(f"Potential concept confusion: {', '.join(confused_concepts)}")
        
        return ". ".join(descriptions)


# ========== Knowledge Gap Identifier ==========

class KnowledgeGapIdentifier:
    """
    Identifies knowledge gaps from wrong answers
    
    Analyzes:
    1. What knowledge was used vs what should have been used
    2. What concepts were misunderstood
    3. What additional knowledge would help
    """
    
    # Domain-specific knowledge requirements
    DOMAIN_KNOWLEDGE_REQUIREMENTS = {
        "genetics": [
            "Mendelian inheritance patterns",
            "Hardy-Weinberg equilibrium",
            "Linkage and recombination",
            "Population genetics concepts (Fst, Dxy, gene flow)",
            "Chromosome structure and behavior",
        ],
        "molecular_biology": [
            "Transcription regulation mechanisms",
            "Translation process",
            "DNA replication",
            "Operon systems (lac, trp)",
            "Epigenetic modifications",
        ],
        "biochemistry": [
            "Enzyme kinetics (Michaelis-Menten)",
            "Metabolic pathways",
            "Protein structure levels",
            "Ligand binding principles",
            "Thermodynamics of reactions",
        ],
        "immunology": [
            "MHC class I vs II pathways",
            "B cell vs T cell functions",
            "Antibody structure and types",
            "Immune response cascades",
            "Tolerance mechanisms",
        ],
        "clinical_medicine": [
            "Diagnostic criteria",
            "Treatment guidelines",
            "Drug mechanisms and interactions",
            "Risk factor assessment",
            "Clinical trial interpretation",
        ],
        "protein_structure": [
            "Primary to quaternary structure",
            "Folding principles",
            "Structure-function relationships",
            "Stability determinants",
            "Common motifs",
        ],
        "enzyme_kinetics": [
            "Michaelis-Menten equation",
            "Inhibition types and effects",
            "Allosteric regulation",
            "Cofactor requirements",
            "pH and temperature effects",
        ],
    }
    
    # Common reasoning traps by domain
    DOMAIN_REASONING_TRAPS = {
        "genetics": [
            "Confusing allele frequency with genotype frequency",
            "Assuming Hardy-Weinberg when selection/migration exists",
            "Mixing up recombination frequency and map distance",
            "Confusing Fst (differentiation) with diversity measures",
        ],
        "molecular_biology": [
            "Confusing transcriptional and translational regulation",
            "Mixing up attenuation and repression mechanisms",
            "Overlooking multiple regulation layers",
            "Ignoring context-dependence of regulation",
        ],
        "biochemistry": [
            "Confusing Km and Vmax effects",
            "Mixing up competitive and non-competitive inhibition",
            "Ignoring enzyme concentration effects",
            "Oversimplifying pathway regulation",
        ],
        "enzyme_kinetics": [
            "Assuming linearity outside valid range",
            "Ignoring enzyme concentration in rate calculations",
            "Confusing initial rate with steady state",
            "Mixing up substrate and product inhibition",
        ],
    }
    
    def __init__(self):
        self.domain_requirements = self.DOMAIN_KNOWLEDGE_REQUIREMENTS
        self.domain_traps = self.DOMAIN_REASONING_TRAPS
    
    def identify_gaps(self,
                      question: str,
                      wrong_answer: str,
                      correct_answer: str,
                      used_knowledge: Optional[List[str]],
                      domain: str,
                      error_category: ErrorCategory) -> Tuple[List[str], List[str], str, str]:
        """
        Identify knowledge gaps and reasoning traps
        
        Args:
            question: The question
            wrong_answer: The wrong answer
            correct_answer: The correct answer
            used_knowledge: Knowledge that was used
            domain: Problem domain
            error_category: Type of error identified
            
        Returns:
            (missing_knowledge, wrong_knowledge, reasoning_trap, correct_direction)
        """
        missing_knowledge = []
        wrong_knowledge = []
        reasoning_trap = ""
        correct_direction = ""
        
        # Normalize domain
        domain_lower = domain.lower() if domain else ""
        matched_domain = None
        for key in self.domain_requirements.keys():
            if key in domain_lower or domain_lower in key:
                matched_domain = key
                break
        
        # Identify missing knowledge based on domain requirements
        if matched_domain:
            requirements = self.domain_requirements.get(matched_domain, [])
            used_knowledge_lower = [k.lower() for k in (used_knowledge or [])]
            
            for req in requirements:
                # Check if this requirement was used
                req_lower = req.lower()
                used = any(req_lower in k for k in used_knowledge_lower)
                
                # If not used and seems relevant to question
                if not used and self._is_relevant_to_question(req, question, wrong_answer, correct_answer):
                    missing_knowledge.append(req)
        
        # Identify wrong knowledge based on error type
        if error_category == ErrorCategory.CONCEPT_ERROR:
            # Likely misunderstanding of concepts
            wrong_knowledge = self._identify_misunderstood_concepts(
                question, wrong_answer, correct_answer, used_knowledge
            )
        elif error_category == ErrorCategory.LOGIC_ERROR:
            # Likely wrong application of correct concepts
            wrong_knowledge = self._identify_wrong_logic(
                question, wrong_answer, correct_answer, used_knowledge
            )
        elif error_category == ErrorCategory.CALCULATION_ERROR:
            # Likely wrong formula or calculation approach
            wrong_knowledge = self._identify_calculation_error(
                question, wrong_answer, correct_answer
            )
        
        # Identify reasoning trap
        if matched_domain:
            traps = self.domain_traps.get(matched_domain, [])
            for trap in traps:
                if self._is_trap_triggered(trap, question, wrong_answer, correct_answer):
                    reasoning_trap = trap
                    break
        
        # Generate correct direction
        correct_direction = self._generate_correct_direction(
            error_category, wrong_answer, correct_answer, missing_knowledge, reasoning_trap
        )
        
        return missing_knowledge, wrong_knowledge, reasoning_trap, correct_direction
    
    def _is_relevant_to_question(self, 
                                  requirement: str, 
                                  question: str,
                                  wrong_answer: str,
                                  correct_answer: str) -> bool:
        """Check if a knowledge requirement is relevant to the question"""
        all_text = f"{question} {wrong_answer} {correct_answer}".lower()
        
        # Extract key terms from requirement
        terms = requirement.lower().split()
        
        # Check if multiple terms appear
        matches = sum(1 for term in terms if term in all_text)
        
        return matches >= 2
    
    def _identify_misunderstood_concepts(self,
                                         question: str,
                                         wrong_answer: str,
                                         correct_answer: str,
                                         used_knowledge: Optional[List[str]]) -> List[str]:
        """Identify concepts that were misunderstood"""
        wrong_knowledge = []
        
        # Check for common misunderstandings based on answer difference
        if used_knowledge:
            for knowledge in used_knowledge:
                # If knowledge was used but answer was wrong, it might be misunderstood
                # This is a simplified heuristic
                if wrong_answer in knowledge or knowledge in wrong_answer:
                    wrong_knowledge.append(f"Possible misunderstanding: {knowledge}")
        
        return wrong_knowledge
    
    def _identify_wrong_logic(self,
                              question: str,
                              wrong_answer: str,
                              correct_answer: str,
                              used_knowledge: Optional[List[str]]) -> List[str]:
        """Identify wrong logical reasoning"""
        wrong_knowledge = []
        
        # Common logic errors
        logic_errors = [
            ("correlation", "causation", "Confusing correlation with causation"),
            ("necessary", "sufficient", "Confusing necessary with sufficient conditions"),
            ("all", "some", "Overgeneralizing from some to all"),
            ("if", "only if", "Confusing 'if' with 'only if' direction"),
        ]
        
        question_lower = question.lower()
        for term1, term2, error in logic_errors:
            if term1 in question_lower or term2 in question_lower:
                wrong_knowledge.append(error)
        
        return wrong_knowledge
    
    def _identify_calculation_error(self,
                                    question: str,
                                    wrong_answer: str,
                                    correct_answer: str) -> List[str]:
        """Identify calculation-related errors"""
        wrong_knowledge = []
        
        # Try to extract numeric values
        try:
            wrong_num = float(wrong_answer.replace(',', '').replace('%', '').strip())
            correct_num = float(correct_answer.replace(',', '').replace('%', '').strip())
            
            # Check common calculation errors
            if abs(wrong_num - correct_num) < 0.01:
                # Very close - likely rounding error
                wrong_knowledge.append("Precision/rounding error")
            elif abs(wrong_num - correct_num * 2) < 0.01:
                # Double the correct answer
                wrong_knowledge.append("Possible double-counting error")
            elif abs(wrong_num - correct_num / 2) < 0.01:
                # Half the correct answer
                wrong_knowledge.append("Possible missing factor of 2")
            elif abs(wrong_num - 1/correct_num) < 0.01:
                # Reciprocal
                wrong_knowledge.append("Possible reciprocal error")
            else:
                wrong_knowledge.append("Calculation formula or parameter error")
        except ValueError:
            wrong_knowledge.append("Numerical calculation error")
        
        return wrong_knowledge
    
    def _is_trap_triggered(self,
                           trap: str,
                           question: str,
                           wrong_answer: str,
                           correct_answer: str) -> bool:
        """Check if a reasoning trap was likely triggered"""
        trap_lower = trap.lower()
        all_text = f"{question} {wrong_answer} {correct_answer}".lower()
        
        # Extract key terms from trap description
        terms = [t for t in trap_lower.split() if len(t) > 4]
        
        # Check if trap is relevant
        matches = sum(1 for term in terms if term in all_text)
        
        return matches >= 2
    
    def _generate_correct_direction(self,
                                    error_category: ErrorCategory,
                                    wrong_answer: str,
                                    correct_answer: str,
                                    missing_knowledge: List[str],
                                    reasoning_trap: str) -> str:
        """Generate guidance for correct direction"""
        directions = []
        
        if error_category == ErrorCategory.CONCEPT_ERROR:
            directions.append("Re-examine the fundamental concepts involved")
            if missing_knowledge:
                directions.append(f"Focus on understanding: {missing_knowledge[0]}")
        
        elif error_category == ErrorCategory.LOGIC_ERROR:
            directions.append("Check the logical chain step by step")
            directions.append("Verify each inference is valid")
        
        elif error_category == ErrorCategory.CALCULATION_ERROR:
            directions.append("Re-check the formula and input parameters")
            directions.append("Verify units and conversions")
        
        elif error_category == ErrorCategory.KNOWLEDGE_GAP:
            directions.append("Gather more information on the topic")
            if missing_knowledge:
                directions.append(f"Research: {', '.join(missing_knowledge[:3])}")
        
        if reasoning_trap:
            directions.append(f"Avoid trap: {reasoning_trap}")
        
        return "; ".join(directions)


# ========== Integrated Error Analyzer ==========

class ErrorAnalyzer:
    """
    Integrated error analyzer that combines classification and gap identification
    """
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.gap_identifier = KnowledgeGapIdentifier()
    
    def analyze_error(self,
                      question: str,
                      wrong_answer: str,
                      correct_answer: str,
                      domain: str,
                      reasoning_path: Optional[List[str]] = None,
                      used_knowledge: Optional[List[str]] = None) -> ErrorAnalysisCache:
        """
        Perform complete error analysis
        
        Args:
            question: The question text
            wrong_answer: The wrong answer given
            correct_answer: The correct answer
            domain: Problem domain
            reasoning_path: Reasoning steps taken
            used_knowledge: Knowledge used in reasoning
            
        Returns:
            ErrorAnalysisCache with complete analysis
        """
        # Classify the error
        error_category, error_description, confused_concepts = self.classifier.classify_error(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            reasoning_path=reasoning_path,
            key_knowledge=used_knowledge,
            domain=domain,
        )
        
        # Identify knowledge gaps
        missing_knowledge, wrong_knowledge, reasoning_trap, correct_direction = self.gap_identifier.identify_gaps(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            used_knowledge=used_knowledge,
            domain=domain,
            error_category=error_category,
        )
        
        # Create cache entry
        error_cache = ErrorAnalysisCache(
            question_text=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            error_category=error_category.value,
            error_description=error_description,
            confused_concepts=confused_concepts,
            missing_knowledge=missing_knowledge,
            wrong_knowledge=wrong_knowledge,
            reasoning_trap=reasoning_trap,
            correct_direction=correct_direction,
            domain=domain,
        )
        
        return error_cache
    
    def analyze_and_cache(self,
                          question: str,
                          wrong_answer: str,
                          correct_answer: str,
                          domain: str,
                          reasoning_path: Optional[List[str]] = None,
                          used_knowledge: Optional[List[str]] = None,
                          difficulty_level: str = "medium") -> bool:
        """
        Analyze error and save to cache
        
        Returns:
            True if saved successfully
        """
        print(f"[ErrorAnalyzer] 开始分析错误: {question[:50]}...", flush=True)
        
        analysis = self.analyze_error(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            domain=domain,
            reasoning_path=reasoning_path,
            used_knowledge=used_knowledge,
        )
        
        print(f"[ErrorAnalyzer] 错误分析完成: category={analysis.error_category}", flush=True)
        
        result = cache_error_analysis(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            error_category=analysis.error_category,
            error_description=analysis.error_description,
            missing_knowledge=analysis.missing_knowledge,
            reasoning_trap=analysis.reasoning_trap,
            correct_direction=analysis.correct_direction,
            domain=domain,
            confused_concepts=analysis.confused_concepts,
            wrong_knowledge=analysis.wrong_knowledge,
            difficulty_level=difficulty_level,
        )
        
        print(f"[ErrorAnalyzer] cache_error_analysis 返回: {result}", flush=True)
        return result


# ========== Convenience Functions ==========

def analyze_error(question: str,
                  wrong_answer: str,
                  correct_answer: str,
                  domain: str,
                  **kwargs) -> ErrorAnalysisCache:
    """Convenience function to analyze an error"""
    analyzer = ErrorAnalyzer()
    return analyzer.analyze_error(
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        domain=domain,
        **kwargs
    )


def analyze_and_cache_error(question: str,
                            wrong_answer: str,
                            correct_answer: str,
                            domain: str,
                            **kwargs) -> bool:
    """Convenience function to analyze and cache an error"""
    analyzer = ErrorAnalyzer()
    return analyzer.analyze_and_cache(
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        domain=domain,
        **kwargs
    )


