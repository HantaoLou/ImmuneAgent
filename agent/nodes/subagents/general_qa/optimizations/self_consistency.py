"""
Self-Consistency Checker - P1 Priority Optimization

Validates answer consistency through:
1. Answer extraction from multiple LLM calls
2. Consistency scoring across answers
3. Confidence calibration
4. Disagreement detection and resolution
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json


class AnswerType(Enum):
    """Types of answers"""
    SINGLE_CHOICE = "single_choice"
    NUMERICAL = "numerical"
    TEXT = "text"
    LIST = "list"
    BOOLEAN = "boolean"


@dataclass
class ExtractedAnswer:
    """An extracted answer from a response"""
    raw_response: str
    answer_text: str
    answer_type: AnswerType
    confidence: float
    option_letter: Optional[str] = None
    numerical_value: Optional[float] = None
    reasoning: str = ""
    entities: List[str] = field(default_factory=list)


@dataclass
class ConsistencyResult:
    """Result of consistency check"""
    is_consistent: bool
    consistency_score: float
    agreed_answer: str
    answer_distribution: Dict[str, int]
    disagreement_points: List[str]
    final_confidence: float
    recommended_action: str


class SelfConsistencyChecker:
    """
    Checks consistency across multiple answers
    """
    
    def __init__(self, consistency_threshold: float = 0.7):
        self.consistency_threshold = consistency_threshold
    
    def extract_answers_from_responses(self, responses: List[str],
                                       question_type: Optional[str] = None) -> List[ExtractedAnswer]:
        """
        Extract answers from multiple LLM responses
        
        Args:
            responses: List of LLM response texts
            question_type: Optional hint about expected answer type
            
        Returns:
            List of ExtractedAnswer objects
        """
        extracted = []
        
        for response in responses:
            answer = self._extract_single_answer(response, question_type)
            if answer:
                extracted.append(answer)
        
        return extracted
    
    def _extract_single_answer(self, response: str, 
                               question_type: Optional[str] = None) -> Optional[ExtractedAnswer]:
        """Extract answer from a single response"""
        if not response:
            return None
        
        # Determine answer type from response
        answer_type = self._determine_answer_type(response, question_type)
        
        # Extract based on type
        if answer_type == AnswerType.SINGLE_CHOICE:
            return self._extract_choice_answer(response)
        elif answer_type == AnswerType.NUMERICAL:
            return self._extract_numerical_answer(response)
        elif answer_type == AnswerType.BOOLEAN:
            return self._extract_boolean_answer(response)
        else:
            return self._extract_text_answer(response)
    
    def _determine_answer_type(self, response: str, 
                               question_type: Optional[str]) -> AnswerType:
        """Determine the type of answer in the response"""
        if question_type:
            if "choice" in question_type.lower() or "option" in question_type.lower():
                return AnswerType.SINGLE_CHOICE
            if "number" in question_type.lower() or "calculate" in question_type.lower():
                return AnswerType.NUMERICAL
        
        # Check for option letters
        if re.search(r'\b(?:answer|option|选择)\s*(?:is|:|：)\s*[A-Ha-h]', response):
            return AnswerType.SINGLE_CHOICE
        
        # Check for numerical answer
        if re.search(r'(?:answer|结果|value)\s*(?:is|:|：|≈)\s*[\d.]+', response):
            return AnswerType.NUMERICAL
        
        # Check for boolean
        if re.search(r'\b(yes|no|true|false|是|否|对|错)\b', response, re.IGNORECASE):
            return AnswerType.BOOLEAN
        
        return AnswerType.TEXT
    
    def _extract_choice_answer(self, response: str) -> ExtractedAnswer:
        """Extract single choice answer (A, B, C, etc.)"""
        # Look for explicit answer markers
        patterns = [
            r'(?:answer|选择|option)\s*(?:is|:|：)\s*([A-Ha-h])',
            r'([A-Ha-h])[\.\)]\s*(?:is\s+correct|正确|right)',
            r'(?:choose|选择)\s*([A-Ha-h])',
            r'最终答案[：:]\s*([A-Ha-h])',
            r'final\s+answer[：:]\s*([A-Ha-h])',
            r'correct\s+option[：:]\s*([A-Ha-h])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                option = match.group(1).upper()
                return ExtractedAnswer(
                    raw_response=response,
                    answer_text=option,
                    answer_type=AnswerType.SINGLE_CHOICE,
                    option_letter=option,
                    confidence=0.9,
                    reasoning=self._extract_reasoning(response)
                )
        
        # Fallback: look for any option letter mentioned prominently
        all_options = re.findall(r'\b([A-Ha-h])\b', response)
        if all_options:
            # Count occurrences
            from collections import Counter
            counts = Counter(all_options)
            most_common = counts.most_common(1)[0]
            
            return ExtractedAnswer(
                raw_response=response,
                answer_text=most_common[0].upper(),
                answer_type=AnswerType.SINGLE_CHOICE,
                option_letter=most_common[0].upper(),
                confidence=0.5,  # Lower confidence for implicit extraction
                reasoning=self._extract_reasoning(response)
            )
        
        return ExtractedAnswer(
            raw_response=response,
            answer_text="",
            answer_type=AnswerType.SINGLE_CHOICE,
            confidence=0.0,
            reasoning="Could not extract answer"
        )
    
    def _extract_numerical_answer(self, response: str) -> ExtractedAnswer:
        """Extract numerical answer"""
        patterns = [
            r'(?:answer|结果|value)\s*(?:is|:|：|≈)\s*([\d.]+(?:\s*%)?)',
            r'([\d.]+)\s*(?:mg|μg|ml|mM|μM|nM|pmol|nmol|μmol|mmol|mol|g|kg)',
            r'(?:equal\s+to|equals?)\s*([\d.]+)',
            r'(?:计算结果|result)[：:]\s*([\d.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                try:
                    # Remove percentage sign and convert
                    value_str = value_str.replace('%', '').strip()
                    value = float(value_str)
                    
                    return ExtractedAnswer(
                        raw_response=response,
                        answer_text=str(value),
                        answer_type=AnswerType.NUMERICAL,
                        numerical_value=value,
                        confidence=0.9,
                        reasoning=self._extract_reasoning(response)
                    )
                except ValueError:
                    pass
        
        return ExtractedAnswer(
            raw_response=response,
            answer_text="",
            answer_type=AnswerType.NUMERICAL,
            confidence=0.0
        )
    
    def _extract_boolean_answer(self, response: str) -> ExtractedAnswer:
        """Extract boolean (yes/no) answer"""
        lower = response.lower()
        
        # Look for explicit yes/no
        yes_patterns = [r'\byes\b', r'\btrue\b', r'\b是\b', r'\b对\b', r'\bcorrect\b']
        no_patterns = [r'\bno\b', r'\bfalse\b', r'\b否\b', r'\b错\b', r'\bincorrect\b']
        
        for pattern in yes_patterns:
            if re.search(pattern, lower):
                return ExtractedAnswer(
                    raw_response=response,
                    answer_text="Yes",
                    answer_type=AnswerType.BOOLEAN,
                    confidence=0.9,
                    reasoning=self._extract_reasoning(response)
                )
        
        for pattern in no_patterns:
            if re.search(pattern, lower):
                return ExtractedAnswer(
                    raw_response=response,
                    answer_text="No",
                    answer_type=AnswerType.BOOLEAN,
                    confidence=0.9,
                    reasoning=self._extract_reasoning(response)
                )
        
        return ExtractedAnswer(
            raw_response=response,
            answer_text="",
            answer_type=AnswerType.BOOLEAN,
            confidence=0.0
        )
    
    def _extract_text_answer(self, response: str) -> ExtractedAnswer:
        """Extract text answer"""
        # Try to find the main answer statement
        patterns = [
            r'(?:answer|答案|conclusion)[：:]\s*([^.\n]+)',
            r'(?:therefore|thus|所以|因此),?\s*([^.\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                # Extract entities
                entities = self._extract_entities(answer_text)
                
                return ExtractedAnswer(
                    raw_response=response,
                    answer_text=answer_text,
                    answer_type=AnswerType.TEXT,
                    confidence=0.8,
                    entities=entities,
                    reasoning=self._extract_reasoning(response)
                )
        
        # Fallback: use first sentence
        sentences = re.split(r'[.!?。！？]', response)
        if sentences:
            return ExtractedAnswer(
                raw_response=response,
                answer_text=sentences[0].strip()[:200],
                answer_type=AnswerType.TEXT,
                confidence=0.5,
                reasoning=""
            )
        
        return ExtractedAnswer(
            raw_response=response,
            answer_text=response[:200],
            answer_type=AnswerType.TEXT,
            confidence=0.3
        )
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning portion from response"""
        # Look for reasoning sections
        patterns = [
            r'(?:reasoning|分析|reason|because)[：:]\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            r'(?:分析过程|推理过程)[：:]\s*(.+?)(?=\n\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
        
        return ""
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract important entities from text"""
        # Capitalized words
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Technical terms (simple heuristic)
        tech_terms = re.findall(r'\b[A-Z]{2,}\b|\b\w+[A-Z]\w*\b', text)
        
        return list(set(entities + tech_terms))
    
    def check_consistency(self, answers: List[ExtractedAnswer]) -> ConsistencyResult:
        """
        Check consistency across multiple extracted answers
        
        Args:
            answers: List of extracted answers
            
        Returns:
            ConsistencyResult with analysis
        """
        if not answers:
            return ConsistencyResult(
                is_consistent=False,
                consistency_score=0.0,
                agreed_answer="",
                answer_distribution={},
                disagreement_points=["No answers to check"],
                final_confidence=0.0,
                recommended_action="retry"
            )
        
        # Get answer distribution
        answer_distribution = self._get_answer_distribution(answers)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(answer_distribution, len(answers))
        
        # Find agreed answer
        agreed_answer = self._find_agreed_answer(answer_distribution, answers)
        
        # Identify disagreement points
        disagreement_points = self._identify_disagreements(answers)
        
        # Calculate final confidence
        final_confidence = self._calculate_final_confidence(
            consistency_score, answers, answer_distribution
        )
        
        # Determine recommended action
        recommended_action = self._determine_recommended_action(
            consistency_score, disagreement_points
        )
        
        return ConsistencyResult(
            is_consistent=consistency_score >= self.consistency_threshold,
            consistency_score=consistency_score,
            agreed_answer=agreed_answer,
            answer_distribution=answer_distribution,
            disagreement_points=disagreement_points,
            final_confidence=final_confidence,
            recommended_action=recommended_action
        )
    
    def _get_answer_distribution(self, answers: List[ExtractedAnswer]) -> Dict[str, int]:
        """Get distribution of answers"""
        distribution = {}
        
        for answer in answers:
            key = self._normalize_answer(answer.answer_text, answer.answer_type)
            if key:
                distribution[key] = distribution.get(key, 0) + 1
        
        return distribution
    
    def _normalize_answer(self, answer: str, answer_type: AnswerType) -> str:
        """Normalize answer for comparison"""
        if answer_type == AnswerType.SINGLE_CHOICE:
            return answer.upper().strip()
        elif answer_type == AnswerType.NUMERICAL:
            try:
                return f"{float(answer):.4f}"
            except ValueError:
                return answer.strip()
        elif answer_type == AnswerType.BOOLEAN:
            return answer.lower().strip()
        else:
            # For text, normalize whitespace and case
            return " ".join(answer.lower().split())
    
    def _calculate_consistency_score(self, distribution: Dict[str, int], 
                                     total: int) -> float:
        """Calculate consistency score based on distribution"""
        if not distribution or total == 0:
            return 0.0
        
        # Get the most common answer
        max_count = max(distribution.values())
        
        # Score is the proportion of answers that agree
        return max_count / total
    
    def _find_agreed_answer(self, distribution: Dict[str, int],
                           answers: List[ExtractedAnswer]) -> str:
        """Find the most agreed-upon answer"""
        if not distribution:
            return ""
        
        # Find the most common answer
        max_key = max(distribution.keys(), key=lambda k: distribution[k])
        
        # If it's a numerical answer, try to get more precise value
        for answer in answers:
            if self._normalize_answer(answer.answer_text, answer.answer_type) == max_key:
                return answer.answer_text
        
        return max_key
    
    def _identify_disagreements(self, answers: List[ExtractedAnswer]) -> List[str]:
        """Identify points of disagreement"""
        disagreements = []
        
        # Group by answer
        by_answer = {}
        for answer in answers:
            key = self._normalize_answer(answer.answer_text, answer.answer_type)
            if key not in by_answer:
                by_answer[key] = []
            by_answer[key].append(answer)
        
        # If multiple distinct answers
        if len(by_answer) > 1:
            disagreements.append(f"Multiple answers found: {list(by_answer.keys())}")
            
            # Compare reasoning for clues
            reasoning_sets = []
            for key, answer_list in by_answer.items():
                reasoning_keywords = set()
                for a in answer_list:
                    if a.reasoning:
                        words = set(a.reasoning.lower().split())
                        reasoning_keywords.update(words)
                reasoning_sets.append((key, reasoning_keywords))
            
            # Find differing keywords
            if len(reasoning_sets) >= 2:
                common = reasoning_sets[0][1]
                for _, keywords in reasoning_sets[1:]:
                    common = common & keywords
                
                if not common:
                    disagreements.append("No common reasoning elements across different answers")
        
        return disagreements
    
    def _calculate_final_confidence(self, consistency_score: float,
                                    answers: List[ExtractedAnswer],
                                    distribution: Dict[str, int]) -> float:
        """Calculate final confidence score"""
        # Base confidence from consistency
        confidence = consistency_score
        
        # Factor in individual answer confidences
        if answers:
            avg_confidence = sum(a.confidence for a in answers) / len(answers)
            confidence = 0.6 * confidence + 0.4 * avg_confidence
        
        # Penalize if there are many different answers
        unique_answers = len(distribution)
        if unique_answers > 2:
            confidence *= 0.8 / (unique_answers - 1)
        
        return min(1.0, max(0.0, confidence))
    
    def _determine_recommended_action(self, consistency_score: float,
                                      disagreements: List[str]) -> str:
        """Determine what action to take"""
        if consistency_score >= 0.9:
            return "accept"
        elif consistency_score >= self.consistency_threshold:
            return "accept_with_caution"
        elif consistency_score >= 0.4:
            return "request_verification"
        else:
            return "retry"
    
    def run_consistency_check(self, responses: List[str],
                              question_type: Optional[str] = None) -> ConsistencyResult:
        """
        Full consistency check pipeline
        
        Args:
            responses: List of LLM responses
            question_type: Optional hint about answer type
            
        Returns:
            ConsistencyResult with all analysis
        """
        answers = self.extract_answers_from_responses(responses, question_type)
        return self.check_consistency(answers)
    
    def get_consistency_report(self, responses: List[str],
                               question_type: Optional[str] = None) -> str:
        """Generate human-readable consistency report"""
        result = self.run_consistency_check(responses, question_type)
        
        lines = ["# Self-Consistency Analysis Report\n"]
        
        lines.append(f"## Summary")
        lines.append(f"- Is Consistent: {result.is_consistent}")
        lines.append(f"- Consistency Score: {result.consistency_score:.2%}")
        lines.append(f"- Final Confidence: {result.final_confidence:.2%}")
        lines.append(f"- Recommended Action: {result.recommended_action}")
        
        lines.append(f"\n## Answer Distribution")
        for answer, count in result.answer_distribution.items():
            lines.append(f"- '{answer}': {count} vote(s)")
        
        if result.disagreement_points:
            lines.append(f"\n## Disagreement Points")
            for point in result.disagreement_points:
                lines.append(f"- {point}")
        
        lines.append(f"\n## Agreed Answer")
        lines.append(f"**{result.agreed_answer}**")
        
        return "\n".join(lines)
