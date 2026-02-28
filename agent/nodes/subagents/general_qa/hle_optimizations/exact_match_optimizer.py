"""
Exact Match Optimizer for HLE

HLE requires exact string matching for 76-80% of questions.
This module provides sophisticated answer normalization and variant generation
to maximize exact match success rate.

Key Features:
- Numeric answer normalization (42, 42.0, 42.00 → 42)
- Scientific notation handling (1.5e-3, 0.0015, 1.5×10⁻³)
- Chemical formula normalization (H2O, H₂O)
- Multiple choice answer normalization (A, (A), Option A, a → A)
- Unit-aware matching
- Answer variant generation for fuzzy matching
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import math


class AnswerType(Enum):
    """Classification of answer types for appropriate normalization"""
    NUMERIC = "numeric"
    SCIENTIFIC_NOTATION = "scientific_notation"
    PERCENTAGE = "percentage"
    CHEMICAL_FORMULA = "chemical_formula"
    GENE_NAME = "gene_name"
    PROTEIN_NAME = "protein_name"
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_TEXT = "short_text"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class NormalizationRule:
    """Rule for normalizing a specific answer type"""
    answer_type: AnswerType
    patterns: List[str]  # Regex patterns to match
    normalizer: Callable[[str], str]
    variants_generator: Optional[Callable[[str], List[str]]] = None
    case_sensitive: bool = False
    strip_whitespace: bool = True


@dataclass
class MatchResult:
    """Result of an exact match attempt"""
    is_match: bool
    score: float  # 0.0 to 1.0
    predicted_normalized: str
    truth_normalized: str
    match_type: str  # "exact", "variant", "numeric_approx", "none"
    details: Dict[str, Any] = field(default_factory=dict)


class AnswerVariantGenerator:
    """Generates all valid variants of an answer for fuzzy matching"""
    
    # Unicode subscript/superscript mappings
    SUBSCRIPT_MAP = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    SUPERSCRIPT_MAP = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        '⁻': '-', '⁺': '+'
    }
    
    def __init__(self):
        self._numeric_pattern = re.compile(r'^-?\d+\.?\d*$')
        self._scientific_pattern = re.compile(
            r'^-?\d+\.?\d*\s*[eE×xX]\s*[-+]?\d+$|'
            r'^-?\d+\.?\d*\s*×\s*10\^?[-+]?\d+$|'
            r'^-?\d+\.\d+\s*[eE][-+]?\d+$'
        )
        self._percentage_pattern = re.compile(r'^-?\d+\.?\d*\s*%$')
        self._chemical_pattern = re.compile(
            r'^[A-Z][a-z]?(?:\d+|[₀-₉])*(?:[A-Z][a-z]?(?:\d+|[₀-₉])*)*$'
        )
        self._option_pattern = re.compile(
            r'^(?:option\s*)?[\(\[]?\s*([A-Ha-h])\s*[\)\]]?$',
            re.IGNORECASE
        )
    
    def classify_answer_type(self, answer: str) -> AnswerType:
        """Classify the type of answer for appropriate handling"""
        answer = answer.strip()
        
        if not answer:
            return AnswerType.UNKNOWN
        
        # Check for multiple choice option
        if self._option_pattern.match(answer):
            return AnswerType.MULTIPLE_CHOICE
        
        # Check for percentage
        if self._percentage_pattern.match(answer):
            return AnswerType.PERCENTAGE
        
        # Check for scientific notation
        if self._scientific_pattern.match(answer):
            return AnswerType.SCIENTIFIC_NOTATION
        
        # Check for pure numeric
        if self._numeric_pattern.match(answer):
            return AnswerType.NUMERIC
        
        # Check for chemical formula
        if self._chemical_pattern.match(answer):
            return AnswerType.CHEMICAL_FORMULA
        
        # Check for gene name (uppercase with numbers)
        if re.match(r'^[A-Z][A-Z0-9-]+$', answer):
            return AnswerType.GENE_NAME
        
        # Check for list (comma or semicolon separated)
        if ',' in answer or ';' in answer:
            return AnswerType.LIST
        
        # Default to short text
        if len(answer) <= 100:
            return AnswerType.SHORT_TEXT
        
        return AnswerType.UNKNOWN
    
    def generate_variants(self, answer: str, answer_type: Optional[AnswerType] = None) -> List[str]:
        """Generate all valid variants of an answer"""
        if answer_type is None:
            answer_type = self.classify_answer_type(answer)
        
        variants = [answer]
        
        generator_map = {
            AnswerType.NUMERIC: self._numeric_variants,
            AnswerType.SCIENTIFIC_NOTATION: self._scientific_variants,
            AnswerType.PERCENTAGE: self._percentage_variants,
            AnswerType.CHEMICAL_FORMULA: self._chemical_variants,
            AnswerType.GENE_NAME: self._gene_variants,
            AnswerType.MULTIPLE_CHOICE: self._option_variants,
            AnswerType.LIST: self._list_variants,
            AnswerType.SHORT_TEXT: self._text_variants,
        }
        
        generator = generator_map.get(answer_type)
        if generator:
            variants.extend(generator(answer))
        
        return list(set(variants))
    
    def _numeric_variants(self, answer: str) -> List[str]:
        """Generate numeric variants: 42, 42.0, 42.00, +42, etc."""
        variants = []
        answer = answer.strip()
        
        try:
            num = float(answer)
            
            # Integer form
            if num == int(num):
                variants.append(str(int(num)))
            
            # Float forms with different precision
            variants.append(str(num))
            variants.append(f"{num:.1f}")
            variants.append(f"{num:.2f}")
            variants.append(f"{num:.3f}")
            
            # With sign
            if num > 0:
                variants.append(f"+{num}")
                variants.append(f"+{int(num)}" if num == int(num) else f"+{num}")
            
            # Scientific notation
            if abs(num) >= 1000 or (abs(num) < 0.01 and num != 0):
                variants.append(f"{num:.2e}")
                variants.append(f"{num:.2E}")
            
        except ValueError:
            pass
        
        return variants
    
    def _scientific_variants(self, answer: str) -> List[str]:
        """Generate scientific notation variants"""
        variants = []
        
        # Parse the number from various scientific notation formats
        normalized = self._normalize_scientific(answer)
        if normalized:
            try:
                num = float(normalized)
                
                # Different notation styles
                variants.append(f"{num:.2e}")
                variants.append(f"{num:.2E}")
                variants.append(f"{num:.3e}")
                variants.append(f"{num:.3E}")
                
                # Decimal form
                variants.append(str(num))
                variants.append(f"{num:.6f}")
                
                # With ×10^ format
                exp = int(f"{num:.0e}".split('e')[1])
                mantissa = num / (10 ** exp)
                variants.append(f"{mantissa:.1f}×10^{exp}")
                variants.append(f"{mantissa:.1f} x 10^{exp}")
                
            except ValueError:
                pass
        
        return variants
    
    def _normalize_scientific(self, answer: str) -> Optional[str]:
        """Normalize scientific notation to standard format"""
        answer = answer.strip()
        
        # Handle ×10^ format
        match = re.match(r'^(-?\d+\.?\d*)\s*[×xX]\s*10\^?(-?\d+)$', answer)
        if match:
            mantissa, exp = match.groups()
            return f"{mantissa}e{exp}"
        
        # Handle x format
        answer = re.sub(r'\s*[xX]\s*', 'e', answer)
        
        return answer
    
    def _percentage_variants(self, answer: str) -> List[str]:
        """Generate percentage variants"""
        variants = []
        
        # Extract numeric value
        match = re.match(r'^(-?\d+\.?\d*)\s*%$', answer.strip())
        if match:
            num = float(match.group(1))
            
            # Different formats
            variants.append(f"{num}%")
            variants.append(f"{num:.1f}%")
            variants.append(f"{num:.2f}%")
            
            # Decimal form
            variants.append(str(num / 100))
            variants.append(f"{num/100:.4f}")
            
            # With space
            variants.append(f"{num} %")
        
        return variants
    
    def _chemical_variants(self, answer: str) -> List[str]:
        """Generate chemical formula variants (handle subscripts)"""
        variants = []
        answer = answer.strip()
        
        # Convert subscripts to regular numbers
        regular = answer
        for sub, num in self.SUBSCRIPT_MAP.items():
            regular = regular.replace(sub, num)
        variants.append(regular)
        
        # Convert regular numbers to subscripts
        subscript = answer
        for num, sub in self.SUBSCRIPT_MAP.items():
            subscript = subscript.replace(num, sub)
        variants.append(subscript)
        
        # Also try with Unicode subscript conversion
        variants.append(self._to_subscript_notation(regular))
        
        return variants
    
    def _to_subscript_notation(self, formula: str) -> str:
        """Convert H2O to H₂O"""
        result = ""
        for char in formula:
            if char.isdigit():
                result += self._num_to_subscript(char)
            else:
                result += char
        return result
    
    def _num_to_subscript(self, num: str) -> str:
        """Convert single digit to subscript"""
        reverse_map = {v: k for k, v in self.SUBSCRIPT_MAP.items()}
        return reverse_map.get(num, num)
    
    def _gene_variants(self, answer: str) -> List[str]:
        """Generate gene name variants"""
        variants = []
        answer = answer.strip()
        
        # Exact case (gene names are case-sensitive)
        variants.append(answer)
        
        # With dash variations
        if '-' in answer:
            variants.append(answer.replace('-', ''))
        
        # Common prefixes
        if not answer.startswith(('BRCA', 'TP53', 'EGFR', 'KRAS')):
            # Try adding common prefixes
            pass  # Don't add speculative variants for genes
        
        return list(set(variants))
    
    def _option_variants(self, answer: str) -> List[str]:
        """Generate multiple choice option variants"""
        variants = []
        answer = answer.strip()
        
        # Extract the option letter
        match = self._option_pattern.match(answer)
        if match:
            letter = match.group(1).upper()
            
            # Various formats
            variants.append(letter)
            variants.append(f"({letter})")
            variants.append(f"[{letter}]")
            variants.append(f"Option {letter}")
            variants.append(f"option {letter}")
            variants.append(f"Option{letter}")
            variants.append(letter.lower())
            variants.append(f"({letter.lower()})")
        
        return variants
    
    def _list_variants(self, answer: str) -> List[str]:
        """Generate list variants (comma/semicolon separated)"""
        variants = []
        answer = answer.strip()
        
        # Split by comma or semicolon
        items = re.split(r'[;,]\s*', answer)
        items = [item.strip() for item in items if item.strip()]
        
        # Different separators
        variants.append(', '.join(items))
        variants.append('; '.join(items))
        variants.append(','.join(items))
        variants.append(';'.join(items))
        
        # With 'and' before last item
        if len(items) > 1:
            variants.append(', '.join(items[:-1]) + ' and ' + items[-1])
            variants.append(', '.join(items[:-1]) + ' & ' + items[-1])
        
        # Different orderings for unordered lists
        if len(items) <= 4:  # Only for small lists to avoid explosion
            from itertools import permutations
            for perm in permutations(items):
                variants.append(', '.join(perm))
        
        return list(set(variants))
    
    def _text_variants(self, answer: str) -> List[str]:
        """Generate text variants"""
        variants = []
        answer = answer.strip()
        
        # Basic variations
        variants.append(answer)
        variants.append(answer.lower())
        variants.append(answer.upper())
        variants.append(answer.capitalize())
        variants.append(answer.title())
        
        # Remove extra whitespace
        variants.append(' '.join(answer.split()))
        
        # Remove punctuation at ends
        variants.append(answer.strip('.,;:'))
        
        return list(set(v for v in variants if v))


class ExactMatchOptimizer:
    """
    HLE Exact Match Optimizer
    
    Optimizes answer matching for HLE's strict exact-match requirements.
    Uses sophisticated normalization and variant generation to maximize
    match success rate.
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize the exact match optimizer.
        
        Args:
            tolerance: Numeric tolerance for approximate matching (default 1%)
        """
        self.tolerance = tolerance
        self.variant_generator = AnswerVariantGenerator()
        self._match_history: List[MatchResult] = []
    
    def exact_match_score(
        self, 
        predicted: str, 
        ground_truth: str,
        question_type: Optional[str] = None
    ) -> MatchResult:
        """
        Calculate exact match score between predicted and ground truth.
        
        Args:
            predicted: The model's predicted answer
            ground_truth: The correct answer
            question_type: Optional question type hint for better matching
            
        Returns:
            MatchResult with score and details
        """
        predicted = str(predicted).strip() if predicted else ""
        ground_truth = str(ground_truth).strip() if ground_truth else ""
        
        # Step 1: Direct exact match
        if predicted == ground_truth:
            return MatchResult(
                is_match=True,
                score=1.0,
                predicted_normalized=predicted,
                truth_normalized=ground_truth,
                match_type="exact"
            )
        
        # Step 2: Classify answer types
        pred_type = self.variant_generator.classify_answer_type(predicted)
        truth_type = self.variant_generator.classify_answer_type(ground_truth)
        
        # Step 3: Generate variants
        pred_variants = set(self.variant_generator.generate_variants(predicted, pred_type))
        truth_variants = set(self.variant_generator.generate_variants(ground_truth, truth_type))
        
        # Step 4: Check variant intersection
        common_variants = pred_variants & truth_variants
        if common_variants:
            return MatchResult(
                is_match=True,
                score=1.0,
                predicted_normalized=list(common_variants)[0],
                truth_normalized=ground_truth,
                match_type="variant",
                details={"matching_variants": list(common_variants)}
            )
        
        # Step 5: Numeric approximation (if both are numeric)
        if pred_type in [AnswerType.NUMERIC, AnswerType.SCIENTIFIC_NOTATION, AnswerType.PERCENTAGE]:
            if truth_type in [AnswerType.NUMERIC, AnswerType.SCIENTIFIC_NOTATION, AnswerType.PERCENTAGE]:
                numeric_result = self._numeric_approximate_match(predicted, ground_truth)
                if numeric_result.is_match:
                    return numeric_result
        
        # Step 6: Fuzzy text matching (last resort)
        fuzzy_score = self._fuzzy_text_match(predicted, ground_truth)
        if fuzzy_score > 0.9:  # Very high threshold for HLE
            return MatchResult(
                is_match=True,
                score=fuzzy_score,
                predicted_normalized=predicted,
                truth_normalized=ground_truth,
                match_type="fuzzy_high",
                details={"fuzzy_score": fuzzy_score}
            )
        
        # No match
        return MatchResult(
            is_match=False,
            score=0.0,
            predicted_normalized=predicted,
            truth_normalized=ground_truth,
            match_type="none",
            details={
                "pred_type": pred_type.value,
                "truth_type": truth_type.value,
                "pred_variants_count": len(pred_variants),
                "truth_variants_count": len(truth_variants)
            }
        )
    
    def _numeric_approximate_match(self, predicted: str, ground_truth: str) -> MatchResult:
        """Check if two numeric values are approximately equal"""
        try:
            # Extract numeric values
            pred_num = self._extract_numeric(predicted)
            truth_num = self._extract_numeric(ground_truth)
            
            if pred_num is None or truth_num is None:
                return MatchResult(
                    is_match=False, score=0.0,
                    predicted_normalized=predicted,
                    truth_normalized=ground_truth,
                    match_type="none"
                )
            
            # Calculate relative error
            if truth_num == 0:
                relative_error = abs(pred_num)
            else:
                relative_error = abs(pred_num - truth_num) / abs(truth_num)
            
            if relative_error <= self.tolerance:
                return MatchResult(
                    is_match=True,
                    score=1.0 - relative_error,
                    predicted_normalized=str(pred_num),
                    truth_normalized=str(truth_num),
                    match_type="numeric_approx",
                    details={
                        "predicted_numeric": pred_num,
                        "truth_numeric": truth_num,
                        "relative_error": relative_error
                    }
                )
            
        except (ValueError, TypeError):
            pass
        
        return MatchResult(
            is_match=False, score=0.0,
            predicted_normalized=predicted,
            truth_normalized=ground_truth,
            match_type="none"
        )
    
    def _extract_numeric(self, value: str) -> Optional[float]:
        """Extract numeric value from string"""
        value = value.strip()
        
        # Handle percentage
        if value.endswith('%'):
            try:
                return float(value.rstrip('%'))
            except ValueError:
                return None
        
        # Handle scientific notation
        value = self.variant_generator._normalize_scientific(value) or value
        
        try:
            return float(value)
        except ValueError:
            # Try to extract first number
            match = re.search(r'-?\d+\.?\d*', value)
            if match:
                return float(match.group())
        
        return None
    
    def _fuzzy_text_match(self, text1: str, text2: str) -> float:
        """
        Calculate fuzzy text similarity score.
        Uses multiple methods and takes the maximum.
        """
        # Normalize both texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        if t1 == t2:
            return 1.0
        
        # Levenshtein distance based similarity
        lev_score = self._levenshtein_similarity(t1, t2)
        
        # Token-based similarity
        token_score = self._token_similarity(t1, t2)
        
        # Character n-gram similarity
        ngram_score = self._ngram_similarity(t1, t2, n=3)
        
        return max(lev_score, token_score, ngram_score)
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein distance-based similarity"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return 0.0
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        distance = previous_row[-1]
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)
    
    def _token_similarity(self, s1: str, s2: str) -> float:
        """Calculate token-based (Jaccard) similarity"""
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def _ngram_similarity(self, s1: str, s2: str, n: int = 3) -> float:
        """Calculate n-gram based similarity"""
        def get_ngrams(s, n):
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ngrams1 = get_ngrams(s1, n)
        ngrams2 = get_ngrams(s2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union)
    
    def normalize_answer(self, answer: str, answer_type: Optional[AnswerType] = None) -> str:
        """
        Normalize an answer to its canonical form.
        
        Args:
            answer: The answer to normalize
            answer_type: Optional type hint for better normalization
            
        Returns:
            Normalized answer string
        """
        if not answer:
            return ""
        
        answer = answer.strip()
        
        if answer_type is None:
            answer_type = self.variant_generator.classify_answer_type(answer)
        
        # Get the first variant as canonical form
        variants = self.variant_generator.generate_variants(answer, answer_type)
        
        if variants:
            # For numeric, prefer integer form
            if answer_type == AnswerType.NUMERIC:
                try:
                    num = float(answer)
                    if num == int(num):
                        return str(int(num))
                except ValueError:
                    pass
            
            # For options, prefer uppercase single letter
            if answer_type == AnswerType.MULTIPLE_CHOICE:
                match = self.variant_generator._option_pattern.match(answer)
                if match:
                    return match.group(1).upper()
            
            return variants[0]
        
        return answer
    
    def get_match_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent matches"""
        if not self._match_history:
            return {"total_matches": 0}
        
        exact = sum(1 for m in self._match_history if m.match_type == "exact")
        variant = sum(1 for m in self._match_history if m.match_type == "variant")
        approx = sum(1 for m in self._match_history if m.match_type == "numeric_approx")
        fuzzy = sum(1 for m in self._match_history if m.match_type == "fuzzy_high")
        
        return {
            "total_matches": len(self._match_history),
            "exact_matches": exact,
            "variant_matches": variant,
            "approximate_matches": approx,
            "fuzzy_matches": fuzzy,
            "success_rate": (exact + variant + approx + fuzzy) / len(self._match_history)
        }

