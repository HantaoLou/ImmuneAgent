"""
Optimization Integration Helper

This module provides integration functions to apply optimizations
in the GeneralQA graph nodes.
"""

from typing import Dict, List, Any, Optional, Tuple
import time

# Import optimization modules
try:
    from .mcq_analyzer import MCQOptionAnalyzer, analyze_option_semantics, VectorType
    from .tool_matcher import ToolQuestionMatcher, pre_evaluate_tool_relevance, QuestionDomain
    from .constraint_extractor import ConstraintExtractor, extract_all_constraints
    from .timeout_strategy import TimeoutStrategy, determine_timeout_strategy, get_node_timeout
    from .self_consistency import SelfConsistencyChecker
    from .fallback_chain import ToolFallbackChain, get_fallback_chain_for_domain
    from .parallel_retrieval import ParallelKnowledgeRetriever, SourceType
    from .decision_logger import DecisionLogger, DecisionType, get_decision_logger
    from .state_completeness import StateCompletenessChecker, ValidationLevel
    from .caching import QueryCache, get_global_cache
    from .checkpoint import CheckpointManager, CheckpointStatus
    
    # N0 domain knowledge enhancement (CRITICAL)
    from .domain_knowledge_enhancement import (
        DomainKnowledgeEnhancer, 
        enhance_n0_context,
        get_critical_hints_for_question,
        BiomedicalDomain
    )
    
    # Specialized analyzers (for domain deep-dives when needed)
    from .sequence_analyzer import SequenceAnalyzer, translate_dna, analyze_sequence_question
    from .pathway_analyzer import PathwayAnalyzer, analyze_pathway_relationship
    
    OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    OPTIMIZATIONS_AVAILABLE = False
    print(f"Warning: Optimization modules not available: {e}")


# ===================== N3 Node Optimization Integrations =====================

def optimize_n3_tool_selection(state, available_tools: List) -> Tuple[List, Dict[str, float]]:
    """
    Optimize tool selection for N3 knowledge retrieval node
    
    Args:
        state: GeneralQAState object
        available_tools: List of available tool objects
        
    Returns:
        Tuple of (selected_tools, relevance_scores)
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return available_tools, {}
    
    question_text = getattr(state, 'cleaned_text', '') or getattr(state, 'user_input', '')
    tool_names = [t.name for t in available_tools] if available_tools else []
    
    # Use Tool Question Matcher to pre-evaluate relevance
    matcher = ToolQuestionMatcher()
    
    # Detect domains
    domains = matcher.detect_question_domains(question_text)
    domain_names = [d.value for d, _ in domains[:3]]  # Top 3 domains
    
    # Rank tools by relevance
    ranked_tools = matcher.rank_tools_by_relevance(tool_names, question_text)
    
    # Select relevant tools (threshold: 0.1)
    selected_names = matcher.select_relevant_tools(tool_names, question_text, threshold=0.1)
    
    # Convert back to tool objects
    selected_tools = [t for t in available_tools if t.name in selected_names]
    
    # If no tools selected, use all available
    if not selected_tools:
        selected_tools = available_tools
    
    # Build relevance scores dict
    relevance_scores = {name: score for name, score in ranked_tools[:20]}
    
    # Log decision
    try:
        logger = get_decision_logger()
        logger.log_tool_selection(
            node_name="n3_knowledge_retrieval",
            available_tools=tool_names,
            selected_tools=[t.name for t in selected_tools],
            relevance_scores=relevance_scores,
            domain_info={'domains': domain_names, 'question_length': len(question_text)}
        )
    except Exception:
        pass
    
    return selected_tools, relevance_scores


def extract_n3_constraints(state) -> Dict[str, Any]:
    """
    Extract constraints for N3 node
    
    Args:
        state: GeneralQAState object
        
    Returns:
        Dict with constraint information
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {'constraints': [], 'negated_entities': [], 'required_entities': []}
    
    question_text = getattr(state, 'cleaned_text', '') or getattr(state, 'user_input', '')
    restrictions = getattr(state, 'inference_core_restrictions', [])
    
    extractor = ConstraintExtractor()
    
    # Extract from question text
    result = extractor.extract_all_constraints(question_text)
    
    # Add manually specified restrictions
    for restriction in restrictions:
        result.constraints.append(type('Constraint', (), {
            'constraint_type': type('ConstraintType', (), {'value': 'manual'}),
            'original_text': restriction,
            'extracted_value': restriction,
            'negated_entities': [],
            'required_entities': []
        })())
    
    return {
        'constraints': [{'type': c.constraint_type.value, 'text': c.original_text} 
                       for c in result.constraints],
        'negated_entities': list(result.all_negated_entities),
        'required_entities': list(result.all_required_entities),
        'key_constraints': result.key_constraints,
        'full_result': result
    }


def get_n3_timeout(state) -> int:
    """
    Get recommended timeout for N3 node
    
    Args:
        state: GeneralQAState object
        
    Returns:
        Timeout in seconds
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return 300  # Default 5 minutes
    
    question_text = getattr(state, 'cleaned_text', '') or getattr(state, 'user_input', '')
    
    strategy = TimeoutStrategy()
    return strategy.get_timeout_for_node("n3_knowledge_retrieval", question_text)


# ===================== N7 Node Optimization Integrations =====================

def analyze_mcq_options_for_n7(state, options: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyze MCQ options for N7 inference node
    
    Args:
        state: GeneralQAState object
        options: Dict mapping option_id to option_text
        
    Returns:
        Dict with analysis results
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {'analyses': {}, 'scores': {}, 'recommendation': None}
    
    question_text = getattr(state, 'cleaned_text', '') or getattr(state, 'user_input', '')
    
    analyzer = MCQOptionAnalyzer()
    analyses = analyzer.analyze_all_options(options, question_text)
    
    # Evaluate for co-expression if relevant
    if 'co-expression' in question_text.lower() or 'chaperone' in question_text.lower():
        scores = analyzer.evaluate_for_coexpression(analyses)
    else:
        scores = {opt_id: analysis.confidence for opt_id, analysis in analyses.items()}
    
    # Find best option
    best_option = max(scores.keys(), key=lambda k: scores[k]) if scores else None
    
    # Log decision
    try:
        logger = get_decision_logger()
        logger.log_option_comparison(
            node_name="n7_complete_inference",
            options=options,
            scores=scores,
            winner=best_option or "",
            reasoning=f"Option analysis based on vector type, structure, and entity matching"
        )
    except Exception:
        pass
    
    return {
        'analyses': {opt_id: {
            'vector_type': a.vector_type.value,
            'structure_type': a.structure_type,
            'entities': [e.name for e in a.entities],
            'confidence': a.confidence,
            'match_status': a.match_status.value,
            'match_reason': a.match_reason
        } for opt_id, a in analyses.items()},
        'scores': scores,
        'recommendation': best_option,
        'recommendation_score': scores.get(best_option, 0) if best_option else 0
    }


def check_answer_consistency(responses: List[str], question_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Check consistency across multiple answer responses
    
    Args:
        responses: List of LLM response texts
        question_type: Optional hint about expected answer type
        
    Returns:
        Dict with consistency check results
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {'is_consistent': True, 'consistency_score': 0.8, 'agreed_answer': ''}
    
    checker = SelfConsistencyChecker()
    result = checker.run_consistency_check(responses, question_type)
    
    return {
        'is_consistent': result.is_consistent,
        'consistency_score': result.consistency_score,
        'agreed_answer': result.agreed_answer,
        'final_confidence': result.final_confidence,
        'recommended_action': result.recommended_action,
        'answer_distribution': result.answer_distribution,
        'disagreement_points': result.disagreement_points
    }


def validate_option_constraints(option_text: str, constraint_info: Dict) -> Tuple[bool, List[str]]:
    """
    Validate if an option violates any constraints
    
    Args:
        option_text: The option text to validate
        constraint_info: Constraint info from extract_n3_constraints
        
    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return True, []
    
    extractor = ConstraintExtractor()
    
    # Get the full result if available
    full_result = constraint_info.get('full_result')
    if full_result:
        return extractor.check_option_against_constraints(option_text, full_result)
    
    # Otherwise, do basic check
    violations = []
    option_lower = option_text.lower()
    
    for entity in constraint_info.get('negated_entities', []):
        if entity.lower() in option_lower:
            violations.append(f"Contains negated entity: {entity}")
    
    return len(violations) == 0, violations


def get_n7_timeout(state) -> int:
    """
    Get recommended timeout for N7 node
    
    Args:
        state: GeneralQAState object
        
    Returns:
        Timeout in seconds
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return 180  # Default 3 minutes
    
    question_text = getattr(state, 'cleaned_text', '') or getattr(state, 'user_input', '')
    
    strategy = TimeoutStrategy()
    return strategy.get_timeout_for_node("n7_complete_inference", question_text)


# ===================== State Management Integrations =====================

def validate_state_completeness(state, node_name: str) -> Dict[str, Any]:
    """
    Validate state completeness for a node
    
    Args:
        state: GeneralQAState object
        node_name: Name of the node to validate for
        
    Returns:
        Dict with validation results
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {'is_valid': True, 'is_complete': True, 'missing_fields': [], 'issues': []}
    
    # Convert state to dict
    state_dict = {}
    for attr in dir(state):
        if not attr.startswith('_'):
            try:
                state_dict[attr] = getattr(state, attr)
            except Exception:
                pass
    
    checker = StateCompletenessChecker()
    result = checker.validate_state(state_dict, node_name)
    
    return {
        'is_valid': result.is_valid,
        'is_complete': result.is_complete,
        'missing_fields': result.missing_fields,
        'errors': [{'field': e.field_name, 'message': e.message} for e in result.errors],
        'warnings': [{'field': w.field_name, 'message': w.message} for w in result.warnings]
    }


def ensure_state_completeness(state, node_name: str) -> Tuple[Any, Dict]:
    """
    Ensure state completeness with auto-fill
    
    Args:
        state: GeneralQAState object
        node_name: Name of the node
        
    Returns:
        Tuple of (updated_state_dict, validation_result)
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {}, {'is_valid': True, 'is_complete': True}
    
    # Convert state to dict
    state_dict = {}
    for attr in dir(state):
        if not attr.startswith('_'):
            try:
                state_dict[attr] = getattr(state, attr)
            except Exception:
                pass
    
    checker = StateCompletenessChecker()
    updated_dict, result = checker.ensure_completeness(state_dict, node_name)
    
    # Apply updates to state
    for key, value in result.auto_filled_fields.items():
        if hasattr(state, key):
            try:
                setattr(state, key, value)
            except Exception:
                pass
    
    return updated_dict, {
        'is_valid': result.is_valid,
        'is_complete': result.is_complete,
        'auto_filled': result.auto_filled_fields,
        'issues': [{'level': i.level.value, 'field': i.field_name, 'message': i.message} 
                   for i in result.issues]
    }


# ===================== Decision Logging Integration =====================

def start_decision_session(session_id: str, question_id: str, question_text: str):
    """Start a new decision logging session"""
    if not OPTIMIZATIONS_AVAILABLE:
        return
    
    logger = get_decision_logger()
    logger.start_session(session_id, question_id, question_text)


def log_inference_decision(node_name: str, 
                           input_context: Dict,
                           options: List[Dict],
                           selected: str,
                           reasoning: str,
                           confidence: float):
    """Log an inference decision"""
    if not OPTIMIZATIONS_AVAILABLE:
        return
    
    logger = get_decision_logger()
    logger.log_decision(
        decision_type=DecisionType.ANSWER_GENERATION,
        node_name=node_name,
        input_context=input_context,
        options_considered=options,
        selected_option=selected,
        reasoning=reasoning,
        confidence=confidence
    )


def finalize_decision_session(final_answer: str, final_confidence: float) -> str:
    """Finalize decision session and get report"""
    if not OPTIMIZATIONS_AVAILABLE:
        return ""
    
    logger = get_decision_logger()
    logger.finalize_session(final_answer, final_confidence)
    return logger.get_decision_summary()


# ===================== Caching Integration =====================

def get_cached_knowledge(query: str) -> Optional[Any]:
    """Get cached knowledge result"""
    if not OPTIMIZATIONS_AVAILABLE:
        return None
    
    cache = get_global_cache()
    return cache.get(query)


def cache_knowledge_result(query: str, result: Any, ttl: float = 3600):
    """Cache knowledge result"""
    if not OPTIMIZATIONS_AVAILABLE:
        return
    
    cache = get_global_cache()
    cache.set(query, result, ttl=ttl)


# ===================== Checkpoint Integration =====================

def create_checkpoint_manager(checkpoint_dir: Optional[str] = None):
    """Create or get checkpoint manager"""
    if not OPTIMIZATIONS_AVAILABLE:
        return None
    
    return CheckpointManager(checkpoint_dir=checkpoint_dir)


def save_node_checkpoint(checkpoint_manager, node_name: str, state, 
                         status: str = "completed", error: Optional[str] = None):
    """Save checkpoint for a node"""
    if not OPTIMIZATIONS_AVAILABLE or checkpoint_manager is None:
        return
    
    status_enum = CheckpointStatus.COMPLETED if status == "completed" else CheckpointStatus.FAILED
    
    # Convert state to dict
    state_dict = {}
    for attr in dir(state):
        if not attr.startswith('_'):
            try:
                state_dict[attr] = getattr(state, attr)
            except Exception:
                pass
    
    checkpoint_manager.save_node_checkpoint(node_name, state_dict, status_enum, error)


# ===================== Fallback Chain Integration =====================

def get_tool_fallback_chain(domain: str) -> List[str]:
    """Get fallback chain for a domain"""
    return get_fallback_chain_for_domain(domain)


# ===================== N0 Node Enhancement (Critical) =====================

def enhance_n0_question_processing(question_text: str) -> Dict[str, Any]:
    """
    Main N0 enhancement function - should be called at start of N0 processing
    
    This provides:
    1. Multi-domain detection
    2. Key constraint extraction  
    3. Critical hints that affect answer logic
    4. Domain knowledge injection
    
    Args:
        question_text: The full question text
        
    Returns:
        Dict with all enhancement information
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {'error': 'Optimizations not available', 'enhanced': False}
    
    try:
        enhancer = DomainKnowledgeEnhancer()
        context = enhancer.generate_enhanced_context(question_text)
        
        return {
            'enhanced': True,
            'detected_domains': context.get('detected_domains', []),
            'key_constraints': context.get('key_constraints', []),
            'domain_knowledge': context.get('domain_knowledge', []),
            'critical_hints': context.get('critical_hints', []),
            'cross_domain': context.get('cross_domain', False),
            'enhancement_summary': context.get('enhancement_summary', '')
        }
    except Exception as e:
        return {'error': str(e), 'enhanced': False}


def get_n0_critical_rules(question_text: str) -> List[str]:
    """
    Get critical rules that should be applied for this question
    
    These are domain-specific rules that affect HOW to reason about the question
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return []
    
    try:
        enhancer = DomainKnowledgeEnhancer()
        context = enhancer.generate_enhanced_context(question_text)
        
        rules = []
        for dk in context.get('domain_knowledge', []):
            rules.extend(dk.get('rules', []))
        
        return rules
    except Exception:
        return []


def get_n0_common_pitfalls(question_text: str) -> List[str]:
    """
    Get common pitfalls to avoid for this question type
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return []
    
    try:
        enhancer = DomainKnowledgeEnhancer()
        context = enhancer.generate_enhanced_context(question_text)
        
        pitfalls = []
        for dk in context.get('domain_knowledge', []):
            pitfalls.extend(dk.get('common_pitfalls', []))
        
        return pitfalls
    except Exception:
        return []


# ===================== Utility Functions =====================

def is_duet_vector(option_text: str) -> bool:
    """Quick check if option text contains a Duet vector"""
    if not OPTIMIZATIONS_AVAILABLE:
        return "duet" in option_text.lower()
    
    result = analyze_option_semantics(option_text)
    return result.get('is_duet_vector', False)


def is_dual_plasmid(option_text: str) -> bool:
    """Quick check if option text describes dual plasmid system"""
    if not OPTIMIZATIONS_AVAILABLE:
        return " and " in option_text.lower() and "plasmid" in option_text.lower()
    
    result = analyze_option_semantics(option_text)
    return result.get('is_dual_plasmid', False)


def get_optimization_report() -> str:
    """Get overall optimization status report"""
    lines = ["# Optimization Module Status\n"]
    
    if OPTIMIZATIONS_AVAILABLE:
        lines.append("Status: ✅ All optimizations available\n")
        
        # Cache stats
        try:
            cache = get_global_cache()
            stats = cache.get_stats()
            lines.append(f"## Cache Statistics")
            lines.append(f"- Hits: {stats.hits}")
            lines.append(f"- Misses: {stats.misses}")
            lines.append(f"- Hit Rate: {stats.hit_rate:.1%}\n")
        except Exception:
            pass
    else:
        lines.append("Status: ❌ Optimizations not available")
    
    return "\n".join(lines)


# ===================== Sequence Analysis Integration =====================

def analyze_dna_translation(question_text: str, sequence: str) -> Dict[str, Any]:
    """
    Analyze DNA translation question with proper start codon detection
    
    Args:
        question_text: The full question text
        sequence: DNA/RNA sequence
        
    Returns:
        Dict with analysis and answer
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {'error': 'Optimizations not available'}
    
    analyzer = SequenceAnalyzer()
    result = analyzer.analyze_translation_question(question_text, sequence)
    
    return {
        'protein_sequence': result['protein_sequence'],
        'start_codon_found': result['start_codon_found'],
        'start_codon_position': result['start_codon_position'],
        'mrna_sequence': result['mrna_sequence'],
        'codons': result['codons'],
        'warnings': result['warnings'],
        'explanation': result['explanation'],
        'key_insight': f"Translation starts from {'first ATG' if result['start_codon_found'] else 'position 0'}"
    }


def is_translation_question(question_text: str) -> bool:
    """Check if question is about DNA/RNA translation"""
    keywords = [
        'dna sequence', 'rna sequence', 'translate', 'transcription',
        'amino acid sequence', 'protein sequence', 'codon', 
        'first protein', 'single letter code'
    ]
    question_lower = question_text.lower()
    return any(kw in question_lower for kw in keywords)


def is_pathway_relationship_question(question_text: str) -> bool:
    """Check if question is about pathway relationship"""
    keywords = [
        'pathway', 'relationship between', 'expression', 
        'coefficient', 'proportional', '¬∝'
    ]
    question_lower = question_text.lower()
    return any(kw in question_lower for kw in keywords)


def analyze_pathway_question(pathway_text: str, 
                            entity_a: str, 
                            entity_b: str,
                            options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Analyze pathway relationship question
    
    Args:
        pathway_text: The pathway notation text
        entity_a: First entity (source)
        entity_b: Second entity (target)  
        options: Optional dict of answer choices
        
    Returns:
        Dict with analysis and recommendation
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {'error': 'Optimizations not available'}
    
    analyzer = PathwayAnalyzer()
    result = analyzer.analyze_relationship_question(
        pathway_text, entity_a, entity_b, options
    )
    
    return {
        'paths_found': result['paths_found'],
        'has_negative_feedback': result['has_negative_feedback'],
        'direct_relationship_exists': result['direct_relationship_exists'],
        'key_insight': result['key_insight'],
        'recommendation': result.get('recommendation'),
        'path_details': result.get('path_details', [])
    }

