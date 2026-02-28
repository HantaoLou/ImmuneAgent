"""
Answer Cache System - Question-Answer Caching with Learning from Errors

This module implements a comprehensive caching system that:
1. Stores correct answers with reasoning paths for direct retrieval
2. Stores error analyses to guide future reasoning (not simple "wrong->opposite")
3. Manages cache validity based on knowledge stability and time decay
4. Provides interfaces for cache lookup, storage, and invalidation

Key Design Principles:
- Correct answers: Cache complete answer + reasoning + knowledge points
- Error answers: Cache error analysis + knowledge gaps + reasoning traps (NOT the wrong answer itself)
- Error utilization: Inject warnings and supplement retrieval, NOT direct answer substitution
"""

import json
import hashlib
import time
import os
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


# ========== Enums ==========

class ErrorCategory(str, Enum):
    """Error category classification"""
    CONCEPT_ERROR = "concept_error"           # 概念理解错误
    LOGIC_ERROR = "logic_error"               # 推理逻辑错误
    CALCULATION_ERROR = "calculation_error"   # 计算错误
    KNOWLEDGE_GAP = "knowledge_gap"           # 知识缺失
    MISINTERPRETATION = "misinterpretation"   # 题目误解
    OVERSIMPLIFICATION = "oversimplification" # 过度简化
    CONFUSION = "confusion"                   # 概念混淆
    UNKNOWN = "unknown"                       # 未知错误类型


class KnowledgeStability(str, Enum):
    """Knowledge stability level - affects cache TTL"""
    PERMANENT = "permanent"           # 物理常数、分子量等 - 永久有效
    HIGH = "high"                     # 基础遗传学定律、经典生化反应 - 1年
    MEDIUM = "medium"                 # 蛋白质功能、信号通路 - 6个月
    LOW = "low"                       # 临床指南、药物适应症 - 3个月
    VOLATILE = "volatile"             # 前沿研究、最新发现 - 1个月


class CacheSource(str, Enum):
    """Source of the cached answer"""
    TEST = "test"                     # 测试环境（有标准答案验证）
    PRODUCTION = "production"         # 生产环境（用户反馈或高置信度）
    USER_FEEDBACK = "user_feedback"   # 用户反馈标记


# ========== Data Classes ==========

@dataclass
class ReasoningStep:
    """A single reasoning step"""
    step_number: int
    description: str
    premise: str
    operation: str
    conclusion: str
    confidence: float = 1.0


@dataclass
class AnswerCache:
    """
    Cache entry for a CORRECT answer
    
    This stores the complete answer with reasoning to enable:
    1. Direct retrieval for identical questions
    2. Reasoning path reference for similar questions
    3. Knowledge point reuse
    4. Reasoning analysis for understanding HOW the answer was derived
    
    Enhanced cache structure:
    - 问题 (question_text)
    - 答案 (final_answer)
    - 领域知识 (domain_knowledge) - N3节点获取的完整知识
    - 推理路线 (inference_chain) - N7节点的完整推理步骤
    - 推理分析 (reasoning_analysis) - LLM总结的推理过程
    - 关键线索 (critical_hints) - 得出答案的关键线索
    """
    # Identification
    question_hash: str                           # SHA256 hash of normalized question
    question_text: str                           # Original question text
    
    # Answer
    final_answer: str                            # The correct answer
    is_correct: bool = True                      # Always True for AnswerCache
    
    # Reasoning (for reference and learning)
    reasoning_path: List[str] = field(default_factory=list)      # Key reasoning steps (simplified)
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)  # Detailed steps
    
    # ========== NEW: Enhanced Reasoning & Knowledge ==========
    # Complete domain knowledge from N3 (领域知识映射)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    # Complete inference chain from N7 (推理链路)
    inference_chain: List[Dict[str, Any]] = field(default_factory=list)
    # LLM-generated reasoning analysis (推理分析总结)
    reasoning_analysis: str = ""
    # Critical hints that led to the answer (关键线索)
    critical_hints: List[str] = field(default_factory=list)
    # Knowledge-application mapping (知识-推理步骤映射)
    knowledge_application_map: Dict[str, str] = field(default_factory=dict)
    
    # Knowledge (legacy support)
    key_knowledge: List[str] = field(default_factory=list)       # Key knowledge points used
    domain: str = ""                             # Problem domain (genetics, biochemistry, etc.)
    sub_domain: str = ""                         # Sub-domain for finer classification
    
    # Metadata
    confidence: float = 1.0                      # Confidence score (0-1)
    source: str = CacheSource.TEST.value         # Where this cache came from
    model_version: str = ""                      # LLM version used
    
    # Timing
    timestamp: float = field(default_factory=time.time)  # When cached
    last_validated: float = field(default_factory=time.time)  # Last validation time
    validation_count: int = 1                    # How many times validated correct
    
    # Validity
    knowledge_stability: str = KnowledgeStability.MEDIUM.value
    is_validated: bool = True                    # Has been validated against ground truth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnswerCache':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ErrorAnalysisCache:
    """
    Cache entry for analyzing an ERROR - NOT the wrong answer itself!
    
    Key principle: We don't cache "A was wrong so choose B".
    Instead, we cache WHY A was wrong and WHAT knowledge is needed.
    
    This enables:
    1. Supplementing knowledge retrieval with missing knowledge
    2. Injecting warnings about known traps
    3. Guiding reasoning to avoid repeated mistakes
    """
    # Identification
    question_hash: str                           # SHA256 hash of normalized question
    question_text: str                           # Original question text
    
    # Error info (for identification, NOT for "reversing")
    wrong_answer: str = ""                       # What we answered wrong (for detection only)
    correct_answer: Optional[str] = None         # Ground truth if available
    
    # Error analysis (THE CORE - what we actually use)
    error_category: str = ErrorCategory.UNKNOWN.value  # Type of error
    error_description: str = ""                  # Detailed description of WHY it was wrong
    confused_concepts: List[str] = field(default_factory=list)  # Concepts that were confused
    
    # Knowledge gaps (for N3 retrieval enhancement)
    missing_knowledge: List[str] = field(default_factory=list)   # Knowledge we lacked
    wrong_knowledge: List[str] = field(default_factory=list)     # Knowledge we misunderstood
    
    # Reasoning traps (for N7 warning injection)
    reasoning_trap: str = ""                     # The trap we fell into
    trap_description: str = ""                   # Why this is a trap
    correct_direction: str = ""                  # What direction we should have taken
    
    # Key insights
    key_insight: str = ""                        # The key insight needed to solve correctly
    alternative_approach: str = ""               # Alternative approach to try
    
    # Metadata
    domain: str = ""
    sub_domain: str = ""
    difficulty_level: str = "medium"             # easy/medium/hard
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorAnalysisCache':
        """Create from dictionary"""
        return cls(**data)


# ========== Cache Validity Checker ==========

class CacheValidityChecker:
    """
    Check if cached answers are still valid
    
    Factors:
    1. Time decay based on knowledge stability
    2. Model version consistency
    3. Domain-specific TTL rules
    """
    
    # TTL in days for each stability level
    TTL_DAYS = {
        KnowledgeStability.PERMANENT.value: float('inf'),
        KnowledgeStability.HIGH.value: 365,
        KnowledgeStability.MEDIUM.value: 180,
        KnowledgeStability.LOW.value: 90,
        KnowledgeStability.VOLATILE.value: 30,
    }
    
    # Domain-specific stability mapping
    DOMAIN_STABILITY = {
        # High stability domains
        "physics": KnowledgeStability.PERMANENT,
        "molecular_weight": KnowledgeStability.PERMANENT,
        "genetics_basic": KnowledgeStability.HIGH,
        "biochemistry_basic": KnowledgeStability.HIGH,
        "molecular_biology_basic": KnowledgeStability.HIGH,
        
        # Medium stability domains
        "genetics": KnowledgeStability.MEDIUM,
        "biochemistry": KnowledgeStability.MEDIUM,
        "molecular_biology": KnowledgeStability.MEDIUM,
        "immunology": KnowledgeStability.MEDIUM,
        "microbiology": KnowledgeStability.MEDIUM,
        "protein_structure": KnowledgeStability.MEDIUM,
        "enzyme_kinetics": KnowledgeStability.MEDIUM,
        
        # Low stability domains
        "clinical_medicine": KnowledgeStability.LOW,
        "clinical_guidelines": KnowledgeStability.LOW,
        "pharmacology": KnowledgeStability.LOW,
        "drug_indications": KnowledgeStability.LOW,
        
        # Volatile domains
        "frontier_research": KnowledgeStability.VOLATILE,
        "clinical_trials": KnowledgeStability.VOLATILE,
        "new_therapies": KnowledgeStability.VOLATILE,
    }
    
    def __init__(self, current_model_version: str = ""):
        self.current_model_version = current_model_version
    
    def get_stability_for_domain(self, domain: str) -> str:
        """Get knowledge stability level for a domain"""
        # Try exact match first
        if domain in self.DOMAIN_STABILITY:
            return self.DOMAIN_STABILITY[domain].value
        
        # Try partial match
        domain_lower = domain.lower()
        for key, stability in self.DOMAIN_STABILITY.items():
            if key in domain_lower or domain_lower in key:
                return stability.value
        
        # Default to medium
        return KnowledgeStability.MEDIUM.value
    
    def is_cache_valid(self, 
                       cache_entry: AnswerCache, 
                       current_time: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if a cache entry is still valid
        
        Returns:
            (is_valid, confidence_modifier)
            - is_valid: Whether the cache should be used
            - confidence_modifier: Factor to adjust confidence (0-1)
        """
        if current_time is None:
            current_time = time.time()
        
        # Get TTL based on stability
        stability = cache_entry.knowledge_stability
        ttl_days = self.TTL_DAYS.get(stability, 180)  # Default 6 months
        
        # Check if expired
        age_days = (current_time - cache_entry.timestamp) / 86400
        
        if ttl_days != float('inf') and age_days > ttl_days:
            return False, 0.0
        
        # Calculate time decay factor (max 30% decay)
        if ttl_days == float('inf'):
            time_factor = 1.0
        else:
            time_factor = 1.0 - (age_days / ttl_days) * 0.3
            time_factor = max(0.7, time_factor)
        
        # Check model version consistency
        model_factor = 1.0
        if self.current_model_version and cache_entry.model_version:
            if cache_entry.model_version != self.current_model_version:
                # Version mismatch - reduce confidence
                model_factor = 0.8
        
        # Validation count bonus (more validations = higher trust)
        validation_factor = min(1.0, 0.9 + cache_entry.validation_count * 0.02)
        
        # Combined confidence modifier
        confidence_modifier = time_factor * model_factor * validation_factor
        
        # Valid if modifier > 0.5
        is_valid = confidence_modifier > 0.5
        
        return is_valid, confidence_modifier
    
    def is_error_cache_valid(self,
                             error_entry: ErrorAnalysisCache,
                             current_time: Optional[float] = None) -> bool:
        """
        Check if an error analysis cache is still valid
        
        Error caches are generally valid for longer because:
        - The error analysis (why something was wrong) remains relevant
        - The knowledge gaps identified remain valid
        - The reasoning traps remain traps
        """
        if current_time is None:
            current_time = time.time()
        
        # Error analysis is valid for 2 years by default
        max_age_days = 730
        
        age_days = (current_time - error_entry.timestamp) / 86400
        
        return age_days <= max_age_days


# ========== Answer Cache Manager ==========

class AnswerCacheManager:
    """
    Manages storage and retrieval of answer caches
    
    Features:
    - Separate storage for correct answers and error analyses
    - JSON-based persistence
    - Memory index for fast lookup
    - Thread-safe operations
    - LRU eviction when size limit reached
    """
    
    def __init__(self,
                 cache_dir: Optional[str] = None,
                 max_correct_cache_size: int = 10000,
                 max_error_cache_size: int = 5000):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cache files. If None, uses default location.
            max_correct_cache_size: Maximum number of correct answer caches
            max_error_cache_size: Maximum number of error analysis caches
        """
        if cache_dir is None:
            # Default to agent cache directory
            cache_dir = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "..", "..", "cache", "answer_cache"
            )
        
        self.cache_dir = Path(cache_dir)
        self.correct_cache_dir = self.cache_dir / "correct"
        self.error_cache_dir = self.cache_dir / "errors"
        
        # Create directories
        self.correct_cache_dir.mkdir(parents=True, exist_ok=True)
        self.error_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_correct_cache_size = max_correct_cache_size
        self.max_error_cache_size = max_error_cache_size
        
        # Memory index: hash -> filepath
        self._correct_index: Dict[str, str] = {}
        self._error_index: Dict[str, str] = {}
        
        # Access tracking for LRU
        self._access_times: Dict[str, float] = {}
        
        # Thread lock
        self._lock = threading.RLock()
        
        # Validity checker
        self.validity_checker = CacheValidityChecker()
        
        # Load existing caches
        self._load_indexes()
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for consistent hashing"""
        # Remove extra whitespace
        normalized = " ".join(question.split())
        # Convert to lowercase for consistency
        normalized = normalized.lower().strip()
        return normalized
    
    def _compute_hash(self, question: str) -> str:
        """Compute SHA256 hash of normalized question"""
        normalized = self._normalize_question(question)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:32]
    
    def _load_indexes(self):
        """Load existing cache files into memory index"""
        # Load correct answer index
        for filepath in self.correct_cache_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                question_hash = data.get('question_hash', '')
                if question_hash:
                    self._correct_index[question_hash] = str(filepath)
            except Exception:
                continue
        
        # Load error analysis index
        for filepath in self.error_cache_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                question_hash = data.get('question_hash', '')
                if question_hash:
                    self._error_index[question_hash] = str(filepath)
            except Exception:
                continue
    
    def _evict_lru(self, cache_type: str):
        """Evict least recently used entries"""
        if cache_type == "correct":
            index = self._correct_index
            max_size = self.max_correct_cache_size
        else:
            index = self._error_index
            max_size = self.max_error_cache_size
        
        if len(index) < max_size:
            return
        
        # Find LRU entries
        entries_with_access = [
            (h, self._access_times.get(h, 0))
            for h in index.keys()
        ]
        entries_with_access.sort(key=lambda x: x[1])
        
        # Remove 10% of entries
        num_to_remove = max(1, len(index) // 10)
        for hash_val, _ in entries_with_access[:num_to_remove]:
            filepath = index.get(hash_val)
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            del index[hash_val]
            if hash_val in self._access_times:
                del self._access_times[hash_val]
    
    # ========== Save Methods ==========
    
    def save_correct_answer(self,
                            question: str,
                            answer: str,
                            reasoning_path: List[str],
                            key_knowledge: List[str],
                            domain: str,
                            confidence: float = 1.0,
                            model_version: str = "",
                            source: str = CacheSource.TEST.value,
                            reasoning_steps: Optional[List[Dict[str, Any]]] = None,
                            sub_domain: str = "",
                            # ========== NEW: Enhanced parameters ==========
                            domain_knowledge: Optional[Dict[str, Any]] = None,
                            inference_chain: Optional[List[Dict[str, Any]]] = None,
                            reasoning_analysis: str = "",
                            critical_hints: Optional[List[str]] = None,
                            knowledge_application_map: Optional[Dict[str, str]] = None) -> bool:
        """
        Save a correct answer to cache
        
        Enhanced to store complete reasoning trace and knowledge-application mapping.
        
        Args:
            question: The question text
            answer: The correct answer
            reasoning_path: Key reasoning steps (as strings)
            key_knowledge: Knowledge points used
            domain: Problem domain
            confidence: Confidence score (0-1)
            model_version: LLM version used
            source: Where this answer came from
            reasoning_steps: Detailed reasoning steps
            sub_domain: Sub-domain for finer classification
            domain_knowledge: Complete domain knowledge from N3 (领域知识映射)
            inference_chain: Complete inference chain from N7 (推理链路)
            reasoning_analysis: LLM-generated reasoning analysis (推理分析总结)
            critical_hints: Critical hints that led to the answer (关键线索)
            knowledge_application_map: Knowledge-to-reasoning-step mapping (知识-推理步骤映射)
            
        Returns:
            True if saved successfully
        """
        question_hash = self._compute_hash(question)
        
        # Determine knowledge stability
        stability = self.validity_checker.get_stability_for_domain(domain)
        
        cache_entry = AnswerCache(
            question_hash=question_hash,
            question_text=question,
            final_answer=answer,
            reasoning_path=reasoning_path,
            reasoning_steps=reasoning_steps or [],
            # NEW fields
            domain_knowledge=domain_knowledge or {},
            inference_chain=inference_chain or [],
            reasoning_analysis=reasoning_analysis,
            critical_hints=critical_hints or [],
            knowledge_application_map=knowledge_application_map or {},
            # Legacy fields
            key_knowledge=key_knowledge,
            domain=domain,
            sub_domain=sub_domain,
            confidence=confidence,
            model_version=model_version,
            source=source,
            knowledge_stability=stability,
        )
        
        with self._lock:
            # Check if need to evict
            self._evict_lru("correct")
            
            # Save to file
            filepath = self.correct_cache_dir / f"{question_hash}.json"
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(cache_entry.to_dict(), f, ensure_ascii=False, indent=2)
                
                # Update index
                self._correct_index[question_hash] = str(filepath)
                self._access_times[question_hash] = time.time()
                
                return True
            except Exception as e:
                print(f"Error saving correct answer cache: {e}")
                return False
    
    def save_error_analysis(self,
                            question: str,
                            wrong_answer: str,
                            correct_answer: Optional[str],
                            error_category: str,
                            error_description: str,
                            missing_knowledge: List[str],
                            reasoning_trap: str,
                            correct_direction: str,
                            domain: str,
                            confused_concepts: Optional[List[str]] = None,
                            wrong_knowledge: Optional[List[str]] = None,
                            key_insight: str = "",
                            difficulty_level: str = "medium") -> bool:
        """
        Save an error analysis to cache
        
        Args:
            question: The question text
            wrong_answer: What we answered wrong (for detection only)
            correct_answer: The correct answer (if known)
            error_category: Type of error (concept/logic/calculation/gap)
            error_description: Detailed description of WHY it was wrong
            missing_knowledge: Knowledge we lacked
            reasoning_trap: The trap we fell into
            correct_direction: What direction we should have taken
            domain: Problem domain
            confused_concepts: Concepts that were confused
            wrong_knowledge: Knowledge we misunderstood
            key_insight: The key insight needed
            difficulty_level: Problem difficulty
            
        Returns:
            True if saved successfully
        """
        question_hash = self._compute_hash(question)
        
        error_entry = ErrorAnalysisCache(
            question_hash=question_hash,
            question_text=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            error_category=error_category,
            error_description=error_description,
            confused_concepts=confused_concepts or [],
            missing_knowledge=missing_knowledge,
            wrong_knowledge=wrong_knowledge or [],
            reasoning_trap=reasoning_trap,
            correct_direction=correct_direction,
            key_insight=key_insight,
            domain=domain,
            difficulty_level=difficulty_level,
        )
        
        with self._lock:
            # Check if need to evict
            self._evict_lru("errors")
            
            # Save to file
            filepath = self.error_cache_dir / f"{question_hash}.json"
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(error_entry.to_dict(), f, ensure_ascii=False, indent=2)
                
                # Update index
                self._error_index[question_hash] = str(filepath)
                self._access_times[question_hash] = time.time()
                
                print(f"[AnswerCache] 错误分析已保存: {filepath}", flush=True)
                return True
            except Exception as e:
                print(f"[AnswerCache] 保存错误分析缓存失败: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return False
    
    # ========== Lookup Methods ==========
    
    def lookup_correct_answer(self, 
                              question: str,
                              check_validity: bool = True) -> Optional[Tuple[AnswerCache, float]]:
        """
        Look up a correct answer cache
        
        Args:
            question: The question text
            check_validity: Whether to check cache validity
            
        Returns:
            (cache_entry, confidence_modifier) or None if not found/invalid
        """
        question_hash = self._compute_hash(question)
        
        with self._lock:
            if question_hash not in self._correct_index:
                return None
            
            filepath = self._correct_index[question_hash]
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cache_entry = AnswerCache.from_dict(data)
                
                # Update access time
                self._access_times[question_hash] = time.time()
                
                # Check validity if requested
                if check_validity:
                    is_valid, confidence_modifier = self.validity_checker.is_cache_valid(cache_entry)
                    if not is_valid:
                        return None
                    return cache_entry, confidence_modifier
                
                return cache_entry, 1.0
                
            except Exception as e:
                print(f"Error loading correct answer cache: {e}")
                return None
    
    def lookup_error_analysis(self, 
                              question: str,
                              check_validity: bool = True) -> Optional[ErrorAnalysisCache]:
        """
        Look up an error analysis cache
        
        Args:
            question: The question text
            check_validity: Whether to check cache validity
            
        Returns:
            ErrorAnalysisCache or None if not found/invalid
        """
        question_hash = self._compute_hash(question)
        
        with self._lock:
            if question_hash not in self._error_index:
                return None
            
            filepath = self._error_index[question_hash]
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                error_entry = ErrorAnalysisCache.from_dict(data)
                
                # Update access time
                self._access_times[question_hash] = time.time()
                
                # Check validity if requested
                if check_validity:
                    if not self.validity_checker.is_error_cache_valid(error_entry):
                        return None
                
                return error_entry
                
            except Exception as e:
                print(f"Error loading error analysis cache: {e}")
                return None
    
    def lookup_combined(self, 
                        question: str) -> Dict[str, Any]:
        """
        Look up both correct answer and error analysis
        
        Returns:
            {
                'has_correct': bool,
                'correct_cache': Optional[AnswerCache],
                'confidence_modifier': float,
                'has_error': bool,
                'error_cache': Optional[ErrorAnalysisCache],
            }
        """
        result = {
            'has_correct': False,
            'correct_cache': None,
            'confidence_modifier': 0.0,
            'has_error': False,
            'error_cache': None,
        }
        
        # Look up correct answer
        correct_result = self.lookup_correct_answer(question)
        if correct_result:
            result['has_correct'] = True
            result['correct_cache'] = correct_result[0]
            result['confidence_modifier'] = correct_result[1]
        
        # Look up error analysis
        error_cache = self.lookup_error_analysis(question)
        if error_cache:
            result['has_error'] = True
            result['error_cache'] = error_cache
        
        return result
    
    # ========== Utility Methods ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'correct_cache_count': len(self._correct_index),
                'error_cache_count': len(self._error_index),
                'max_correct_cache_size': self.max_correct_cache_size,
                'max_error_cache_size': self.max_error_cache_size,
            }
    
    def clear_all(self):
        """Clear all caches"""
        with self._lock:
            # Clear correct caches
            for filepath in self._correct_index.values():
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            self._correct_index.clear()
            
            # Clear error caches
            for filepath in self._error_index.values():
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            self._error_index.clear()
            
            # Clear access times
            self._access_times.clear()
    
    def invalidate(self, question: str) -> bool:
        """Invalidate cache for a specific question"""
        question_hash = self._compute_hash(question)
        invalidated = False
        
        with self._lock:
            # Invalidate correct cache
            if question_hash in self._correct_index:
                filepath = self._correct_index[question_hash]
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                del self._correct_index[question_hash]
                invalidated = True
            
            # Invalidate error cache
            if question_hash in self._error_index:
                filepath = self._error_index[question_hash]
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                del self._error_index[question_hash]
                invalidated = True
            
            # Remove from access times
            if question_hash in self._access_times:
                del self._access_times[question_hash]
        
        return invalidated


# ========== Global Cache Instance ==========

_global_cache_manager: Optional[AnswerCacheManager] = None


def get_cache_manager() -> AnswerCacheManager:
    """Get or create the global cache manager instance"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = AnswerCacheManager()
    return _global_cache_manager


# ========== Convenience Functions ==========

def cache_correct_answer(question: str,
                         answer: str,
                         reasoning_path: List[str],
                         key_knowledge: List[str],
                         domain: str,
                         **kwargs) -> bool:
    """Convenience function to save a correct answer"""
    return get_cache_manager().save_correct_answer(
        question=question,
        answer=answer,
        reasoning_path=reasoning_path,
        key_knowledge=key_knowledge,
        domain=domain,
        **kwargs
    )


def cache_error_analysis(question: str,
                         wrong_answer: str,
                         error_category: str,
                         error_description: str,
                         missing_knowledge: List[str],
                         reasoning_trap: str,
                         correct_direction: str,
                         domain: str,
                         correct_answer: Optional[str] = None,  # 添加为显式参数
                         **kwargs) -> bool:
    """Convenience function to save an error analysis"""
    return get_cache_manager().save_error_analysis(
        question=question,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        error_category=error_category,
        error_description=error_description,
        missing_knowledge=missing_knowledge,
        reasoning_trap=reasoning_trap,
        correct_direction=correct_direction,
        domain=domain,
        **kwargs
    )


def lookup_answer_cache(question: str) -> Dict[str, Any]:
    """Convenience function to look up caches"""
    return get_cache_manager().lookup_combined(question)


