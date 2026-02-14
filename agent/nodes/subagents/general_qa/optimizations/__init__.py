"""
GeneralQA Agent Optimizations Module

This module contains all optimization components for the GeneralQA agent,
organized by priority levels:

Priority 0 (Critical - Direct Answer Quality Impact):
- MCQ Option Analyzer: Deep analysis of multiple choice options
- Tool Question Matcher: Pre-evaluate tool relevance
- Constraint Extractor: Enhanced constraint extraction

Priority 1 (High - Reliability Impact):
- Timeout Strategy: Dynamic timeout based on complexity
- Self-Consistency Checker: Validate answer consistency
- Tool Fallback Chain: Graceful degradation for tool failures

Priority 2 (Medium - Efficiency Impact):
- Parallel Knowledge Retriever: Concurrent retrieval
- Decision Logger: Explainability logging
- State Completeness Checker: State validation

Priority 3 (Low - Performance Impact):
- Query Cache: Caching for repeated queries
- Checkpoint Manager: Session recovery

Usage:
    from agent.nodes.subagents.general_qa.optimizations import (
        MCQOptionAnalyzer,
        ToolQuestionMatcher,
        TimeoutStrategy,
        # ... etc
    )
"""

# P0 - Critical optimizations
from .mcq_analyzer import (
    MCQOptionAnalyzer,
    OptionAnalysis,
    EntityInfo,
    VectorType,
    OptionMatchStatus,
    analyze_option_semantics,
    VECTOR_DATABASE,
)

from .tool_matcher import (
    ToolQuestionMatcher,
    ToolInfo,
    ToolCategory,
    QuestionDomain,
    pre_evaluate_tool_relevance,
    get_question_domains,
    filter_tools_by_domain,
    TOOL_DEFINITIONS,
)

from .constraint_extractor import (
    ConstraintExtractor,
    Constraint,
    ConstraintType,
    ConstraintAnalysisResult,
    extract_all_constraints,
)

# P1 - High priority optimizations
from .timeout_strategy import (
    TimeoutStrategy,
    TimeoutConfig,
    TimeoutStrategyResult,
    ComplexityLevel,
    QueryType,
    determine_timeout_strategy,
    get_node_timeout,
)

from .self_consistency import (
    SelfConsistencyChecker,
    ExtractedAnswer,
    ConsistencyResult,
    AnswerType,
)

from .fallback_chain import (
    ToolFallbackChain,
    FallbackResult,
    FallbackLevel,
    ToolHealth,
    ToolStatus,
    get_fallback_chain_for_domain,
    DEFAULT_FALLBACK_CHAINS,
)

# P2 - Medium priority optimizations
from .parallel_retrieval import (
    ParallelKnowledgeRetriever,
    RetrievalResult,
    AggregatedResult,
    SourceType,
    parallel_retrieve,
    SOURCE_QUALITY_WEIGHTS,
)

from .decision_logger import (
    DecisionLogger,
    DecisionEntry,
    DecisionLog,
    DecisionType,
    get_decision_logger,
)

from .state_completeness import (
    StateCompletenessChecker,
    ValidationResult,
    ValidationIssue,
    ValidationLevel,
    NODE_REQUIREMENTS,
    DEFAULT_VALUES,
)

# P3 - Low priority optimizations
from .caching import (
    QueryCache,
    CacheEntry,
    CacheStats,
    cached_knowledge_retrieval,
    get_global_cache,
)

from .checkpoint import (
    CheckpointManager,
    NodeCheckpoint,
    SessionCheckpoint,
    CheckpointStatus,
    create_checkpoint_manager,
)

# Domain knowledge enhancement for N0
from .domain_knowledge_enhancement import (
    DomainKnowledgeEnhancer,
    DomainKnowledge,
    BiomedicalDomain,
    ConstraintType,
    KeyConstraint,
    enhance_n0_context,
    get_critical_hints_for_question,
    DOMAIN_KNOWLEDGE_DB,
)

# Specialized analyzers (for specific domain deep-dives)
from .sequence_analyzer import (
    SequenceAnalyzer,
    TranslationResult,
    SequenceType,
    ReadingFrame,
    translate_dna,
    analyze_sequence_question,
    CODON_TABLE,
    START_CODONS,
    STOP_CODONS,
)

from .pathway_analyzer import (
    PathwayAnalyzer,
    PathwayEdge,
    PathwayNode,
    PathResult,
    EdgeType,
    analyze_pathway_relationship,
)

# Integration helpers
from .integration import (
    optimize_n3_tool_selection,
    extract_n3_constraints,
    get_n3_timeout,
    analyze_mcq_options_for_n7,
    check_answer_consistency,
    validate_option_constraints,
    get_n7_timeout,
    validate_state_completeness,
    ensure_state_completeness,
    start_decision_session,
    log_inference_decision,
    finalize_decision_session,
    get_cached_knowledge,
    cache_knowledge_result,
    create_checkpoint_manager,
    save_node_checkpoint,
    get_tool_fallback_chain,
    is_duet_vector,
    is_dual_plasmid,
    get_optimization_report,
    OPTIMIZATIONS_AVAILABLE,
    # N0 enhancement (Critical)
    enhance_n0_question_processing,
    get_n0_critical_rules,
    get_n0_common_pitfalls,
    # DNA/Pathway analysis
    analyze_dna_translation,
    is_translation_question,
    is_pathway_relationship_question,
    analyze_pathway_question,
)

# Version info
__version__ = "1.0.0"
__author__ = "Bio-Agent Team"

# Public API
__all__ = [
    # P0 - Critical
    "MCQOptionAnalyzer",
    "OptionAnalysis",
    "EntityInfo",
    "VectorType",
    "OptionMatchStatus",
    "analyze_option_semantics",
    "VECTOR_DATABASE",
    
    "ToolQuestionMatcher",
    "ToolInfo",
    "ToolCategory",
    "QuestionDomain",
    "pre_evaluate_tool_relevance",
    "get_question_domains",
    "filter_tools_by_domain",
    "TOOL_DEFINITIONS",
    
    "ConstraintExtractor",
    "Constraint",
    "ConstraintType",
    "ConstraintAnalysisResult",
    "extract_all_constraints",
    
    # P1 - High
    "TimeoutStrategy",
    "TimeoutConfig",
    "TimeoutStrategyResult",
    "ComplexityLevel",
    "QueryType",
    "determine_timeout_strategy",
    "get_node_timeout",
    
    "SelfConsistencyChecker",
    "ExtractedAnswer",
    "ConsistencyResult",
    "AnswerType",
    
    "ToolFallbackChain",
    "FallbackResult",
    "FallbackLevel",
    "ToolHealth",
    "ToolStatus",
    "get_fallback_chain_for_domain",
    "DEFAULT_FALLBACK_CHAINS",
    
    # P2 - Medium
    "ParallelKnowledgeRetriever",
    "RetrievalResult",
    "AggregatedResult",
    "SourceType",
    "parallel_retrieve",
    "SOURCE_QUALITY_WEIGHTS",
    
    "DecisionLogger",
    "DecisionEntry",
    "DecisionLog",
    "DecisionType",
    "get_decision_logger",
    
    "StateCompletenessChecker",
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    "NODE_REQUIREMENTS",
    "DEFAULT_VALUES",
    
    # P3 - Low
    "QueryCache",
    "CacheEntry",
    "CacheStats",
    "cached_knowledge_retrieval",
    "get_global_cache",
    
    "CheckpointManager",
    "NodeCheckpoint",
    "SessionCheckpoint",
    "CheckpointStatus",
    "create_checkpoint_manager",
    
    # Domain knowledge enhancement for N0
    "DomainKnowledgeEnhancer",
    "DomainKnowledge",
    "BiomedicalDomain",
    "ConstraintType",
    "KeyConstraint",
    "enhance_n0_context",
    "get_critical_hints_for_question",
    "DOMAIN_KNOWLEDGE_DB",
    
    # Specialized analyzers
    "SequenceAnalyzer",
    "TranslationResult",
    "SequenceType",
    "ReadingFrame",
    "translate_dna",
    "analyze_sequence_question",
    "CODON_TABLE",
    "START_CODONS",
    "STOP_CODONS",
    
    "PathwayAnalyzer",
    "PathwayEdge",
    "PathwayNode",
    "PathResult",
    "EdgeType",
    "analyze_pathway_relationship",
    
    # Integration helpers
    "optimize_n3_tool_selection",
    "extract_n3_constraints",
    "get_n3_timeout",
    "analyze_mcq_options_for_n7",
    "check_answer_consistency",
    "validate_option_constraints",
    "get_n7_timeout",
    "validate_state_completeness",
    "ensure_state_completeness",
    "start_decision_session",
    "log_inference_decision",
    "finalize_decision_session",
    "get_cached_knowledge",
    "cache_knowledge_result",
    "save_node_checkpoint",
    "get_tool_fallback_chain",
    "is_duet_vector",
    "is_dual_plasmid",
    "get_optimization_report",
    "OPTIMIZATIONS_AVAILABLE",
    # NEW specialized analysis
    "analyze_dna_translation",
    "is_translation_question",
    "is_pathway_relationship_question",
    "analyze_pathway_question",
    # N0 enhancement (Critical)
    "enhance_n0_question_processing",
    "get_n0_critical_rules",
    "get_n0_common_pitfalls",
]
