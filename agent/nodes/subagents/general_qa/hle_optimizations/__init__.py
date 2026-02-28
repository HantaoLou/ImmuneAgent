"""
HLE (Humanity's Last Exam) Optimizations Module

This module provides specialized optimizations for tackling HLE-level questions:
- ExactMatchOptimizer: Precise answer matching for HLE's strict requirements
- ConfidenceCalibrator: Calibrate confidence to avoid overconfidence
- OriginalProblemHandler: Handle original/novel problems that can't be searched
- DeepReasoningTree: Tree-based reasoning instead of linear chains
- MultiAgentFramework: Multi-agent collaboration for complex reasoning
- DomainReasoningTemplates: Enhanced templates for genetics, molecular biology, clinical diagnosis
- ConceptKnowledgeGraph: Knowledge graph for concept relationships
- ReasoningChainValidator: Validate reasoning chain consistency
- HLEOptimizedQA: Unified interface for HLE-level question answering

HLE Key Characteristics:
- 76-80% questions require exact string matching
- 5+ options for multiple choice (not standard 4)
- Questions are original and cannot be found via web search
- Master/PhD level difficulty
- Multi-step deep reasoning required
- 14% questions require multimodal understanding
"""

from .exact_match_optimizer import ExactMatchOptimizer, AnswerVariantGenerator
from .confidence_calibrator import (
    ConfidenceCalibrator, 
    UncertaintyExpression,
    QuestionDifficulty,
    ConfidenceFactor
)
from .original_problem_handler import (
    OriginalProblemHandler, 
    FirstPrinciplesReasoner,
    ProblemStrategy,
    ProblemAnalysis
)
from .deep_reasoning_tree import (
    DeepReasoningTree, 
    ReasoningNode, 
    ReasoningPath,
    ReasoningTreeBuilder,
    ReasoningNodeType
)
from .multi_agent_framework import (
    MultiAgentFramework, 
    AgentRole,
    DebateMechanism,
    BaseAgent,
    AgentResponse
)
from .domain_reasoning_templates import (
    GeneticsReasoningTemplate,
    MolecularBiologyTemplate,
    ClinicalDiagnosisTemplate,
    CommonPitfallsRegistry,
    ReasoningTemplate,
    ReasoningStep,
    get_template_for_domain
)
from .concept_knowledge_graph import (
    ConceptKnowledgeGraph,
    ConceptContrast,
    ConceptRelation,
    ConceptNode
)
from .reasoning_chain_validator import (
    ReasoningChainValidator,
    ValidationResult,
    LogicalConnector,
    ReasoningStep as ValidatedReasoningStep,
    ValidationError,
    ValidationErrorType
)
from .llm_retry_wrapper import (
    LLMRetryWrapper, 
    RecoveryMode,
    RetryConfig,
    RetryResult,
    FailureType,
    RecoveryStrategy,
    create_retry_wrapper
)
from .timeout_strategy import (
    AdaptiveTimeoutStrategy, 
    ComplexityEstimator,
    ComplexityLevel,
    ComplexityFactors,
    TimeoutAllocation,
    get_adaptive_timeout
)
from .hle_integrator import (
    HLEOptimizedQA,
    HLEAnswerResult,
    create_hle_qa
)
from .integration_optimizer import (
    IntegrationOptimizer,
    is_result_successful,
    is_timeout_recovery,
    classify_error,
    estimate_complexity,
    get_adaptive_timeout as get_integrated_timeout,
    get_domain_reasoning_template,
    get_global_optimizer,
    quick_evaluate,
    EvaluationResult,
    DOMAIN_TEMPLATES,
    COMPLEXITY_TIMEOUTS
)

__all__ = [
    # Main Integration
    'HLEOptimizedQA',
    'HLEAnswerResult',
    'create_hle_qa',
    
    # Integration Optimizer (NEW)
    'IntegrationOptimizer',
    'is_result_successful',
    'is_timeout_recovery',
    'classify_error',
    'estimate_complexity',
    'get_integrated_timeout',
    'get_domain_reasoning_template',
    'get_global_optimizer',
    'quick_evaluate',
    'EvaluationResult',
    'DOMAIN_TEMPLATES',
    'COMPLEXITY_TIMEOUTS',
    
    # Exact Matching
    'ExactMatchOptimizer',
    'AnswerVariantGenerator',
    
    # Confidence
    'ConfidenceCalibrator',
    'UncertaintyExpression',
    'QuestionDifficulty',
    'ConfidenceFactor',
    
    # Original Problems
    'OriginalProblemHandler',
    'FirstPrinciplesReasoner',
    'ProblemStrategy',
    'ProblemAnalysis',
    
    # Deep Reasoning
    'DeepReasoningTree',
    'ReasoningNode',
    'ReasoningPath',
    'ReasoningTreeBuilder',
    'ReasoningNodeType',
    
    # Multi-Agent
    'MultiAgentFramework',
    'AgentRole',
    'DebateMechanism',
    'BaseAgent',
    'AgentResponse',
    
    # Domain Templates
    'GeneticsReasoningTemplate',
    'MolecularBiologyTemplate',
    'ClinicalDiagnosisTemplate',
    'CommonPitfallsRegistry',
    'ReasoningTemplate',
    'ReasoningStep',
    'get_template_for_domain',
    
    # Knowledge Graph
    'ConceptKnowledgeGraph',
    'ConceptContrast',
    'ConceptRelation',
    'ConceptNode',
    
    # Validation
    'ReasoningChainValidator',
    'ValidationResult',
    'LogicalConnector',
    'ValidatedReasoningStep',
    'ValidationError',
    'ValidationErrorType',
    
    # System Stability
    'LLMRetryWrapper',
    'RecoveryMode',
    'RetryConfig',
    'RetryResult',
    'FailureType',
    'RecoveryStrategy',
    'create_retry_wrapper',
    
    # Timeout
    'AdaptiveTimeoutStrategy',
    'ComplexityEstimator',
    'ComplexityLevel',
    'ComplexityFactors',
    'TimeoutAllocation',
    'get_adaptive_timeout',
]

__version__ = '1.0.0'

