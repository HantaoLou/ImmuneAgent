"""
Test cases for optimization modules

Run with: pytest agent/nodes/subagents/general_qa/optimizations/tests/ -v
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from agent.nodes.subagents.general_qa.optimizations import (
    MCQOptionAnalyzer,
    analyze_option_semantics,
    ToolQuestionMatcher,
    pre_evaluate_tool_relevance,
    ConstraintExtractor,
    extract_all_constraints,
    TimeoutStrategy,
    determine_timeout_strategy,
    SelfConsistencyChecker,
    ToolFallbackChain,
    get_fallback_chain_for_domain,
    ParallelKnowledgeRetriever,
    DecisionLogger,
    get_decision_logger,
    StateCompletenessChecker,
    QueryCache,
    CheckpointManager,
    is_duet_vector,
    is_dual_plasmid,
    OPTIMIZATIONS_AVAILABLE,
)


class TestMCQOptionAnalyzer:
    """Test MCQ option analyzer"""
    
    def test_basic_option_analysis(self):
        """Test basic option analysis"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        analyzer = MCQOptionAnalyzer()
        option_text = "pCDFDuet-1 vector with spectinomycin resistance"
        
        result = analyze_option_semantics(option_text)
        
        assert 'entities' in result
        assert 'vector_type' in result
        assert result['is_duet_vector'] == True
    
    def test_duet_vector_detection(self):
        """Test Duet vector detection"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        assert is_duet_vector("pCDFDuet-1") == True
        assert is_duet_vector("pETDuet-1") == True
        assert is_duet_vector("pET-28a(+)") == False
    
    def test_dual_plasmid_detection(self):
        """Test dual plasmid detection"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        assert is_dual_plasmid("pET-28a and pCDF-1b plasmids") == True
        assert is_dual_plasmid("pCDFDuet-1") == False
    
    def test_multiple_options_analysis(self):
        """Test analysis of multiple options"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        analyzer = MCQOptionAnalyzer()
        options = {
            "A": "pCDFDuet-1 vector with spectinomycin",
            "B": "pET-28a and pCDF-1b dual plasmid system",
            "C": "pACYCDuet-1 with chloramphenicol"
        }
        
        analyses = analyzer.analyze_all_options(options)
        
        assert len(analyses) == 3
        assert analyses["A"].vector_type.value == "duet_vector"
        assert analyses["B"].vector_type.value == "dual_plasmid"


class TestToolQuestionMatcher:
    """Test tool-question matcher"""
    
    def test_domain_detection(self):
        """Test question domain detection"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        matcher = ToolQuestionMatcher()
        
        # Molecular cloning question
        domains = matcher.detect_question_domains(
            "What is the best way to co-express chaperones using Duet vectors?"
        )
        assert len(domains) > 0
        
        # Genetics question
        domains = matcher.detect_question_domains(
            "What is the genomic mutation rate in small populations?"
        )
        assert len(domains) > 0
    
    def test_tool_relevance(self):
        """Test tool relevance evaluation"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        question = "What genes are associated with diabetes?"
        tools = ["query_disgenet", "query_omim", "query_proteinatlas", "query_string"]
        
        relevant_tools = pre_evaluate_tool_relevance(question, tools)
        
        assert len(relevant_tools) > 0
        assert "query_disgenet" in relevant_tools or "query_omim" in relevant_tools


class TestConstraintExtractor:
    """Test constraint extractor"""
    
    def test_negative_constraint(self):
        """Test negative constraint extraction"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        text = "The answer cannot be classified as category 2"
        result = extract_all_constraints(text)
        
        assert result['negative_count'] > 0
    
    def test_exclusive_constraint(self):
        """Test exclusive constraint extraction"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        text = "Only one of these is the correct answer"
        result = extract_all_constraints(text)
        
        assert result['exclusive_count'] > 0
    
    def test_option_validation(self):
        """Test option validation against constraints"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        extractor = ConstraintExtractor()
        constraint_result = extractor.extract_all_constraints("cannot be category 2")
        
        is_valid, violations = extractor.check_option_against_constraints(
            "This is category 2 answer", constraint_result
        )
        
        # Should detect violation
        assert is_valid == False or len(violations) > 0


class TestTimeoutStrategy:
    """Test timeout strategy"""
    
    def test_simple_question_timeout(self):
        """Test timeout for simple questions"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        question = "What is 2 + 2?"
        timeout = determine_timeout_strategy(question)
        
        assert timeout > 0
        assert timeout < 100  # Should be quick
    
    def test_complex_question_timeout(self):
        """Test timeout for complex questions"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        question = """
        Analyze the comprehensive molecular mechanisms of protein folding
        in the endoplasmic reticulum and explain how chaperones facilitate
        proper folding while preventing aggregation.
        """
        timeout = determine_timeout_strategy(question)
        
        assert timeout > 60  # Should be longer


class TestSelfConsistencyChecker:
    """Test self-consistency checker"""
    
    def test_consistent_answers(self):
        """Test with consistent answers"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        checker = SelfConsistencyChecker()
        responses = [
            "The answer is A. Option A is correct because...",
            "I choose A. Based on my analysis...",
            "Answer: A. The reason is..."
        ]
        
        result = checker.run_consistency_check(responses, "multi_choice")
        
        assert result.is_consistent == True
        assert result.consistency_score > 0.5
        assert result.agreed_answer == "A"
    
    def test_inconsistent_answers(self):
        """Test with inconsistent answers"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        checker = SelfConsistencyChecker()
        responses = [
            "The answer is A.",
            "I believe the answer is B.",
            "Option C is correct."
        ]
        
        result = checker.run_consistency_check(responses, "multi_choice")
        
        assert result.is_consistent == False
        assert result.consistency_score < 0.7


class TestToolFallbackChain:
    """Test tool fallback chain"""
    
    def test_fallback_chain(self):
        """Test fallback chain retrieval"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        chain = get_fallback_chain_for_domain("gene_disease")
        
        assert len(chain) > 0
        assert "query_disgenet" in chain
    
    def test_tool_health_tracking(self):
        """Test tool health tracking"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        fallback = ToolFallbackChain()
        
        # Record failures
        fallback.record_failure("test_tool", "Error")
        fallback.record_failure("test_tool", "Error")
        fallback.record_failure("test_tool", "Error")
        
        # Should be marked as unavailable after threshold
        assert fallback.is_tool_available("test_tool") == False


class TestDecisionLogger:
    """Test decision logger"""
    
    def test_session_logging(self):
        """Test session logging"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        logger = DecisionLogger()
        logger.start_session("test_session", "q1", "Test question?")
        
        logger.log_decision(
            decision_type="tool_selection",
            node_name="n3",
            input_context={},
            options_considered=[{"tool": "a"}],
            selected_option="tool_a",
            reasoning="Best match",
            confidence=0.9
        )
        
        summary = logger.get_decision_summary()
        
        assert "Test question" in summary
        assert len(logger.current_log.entries) == 1


class TestStateCompletenessChecker:
    """Test state completeness checker"""
    
    def test_state_validation(self):
        """Test state validation"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        checker = StateCompletenessChecker()
        
        # Missing required field
        state = {"question": "Test?"}
        result = checker.validate_state(state, "input_preprocessing")
        
        assert result.is_valid == False
        assert "question_id" in result.missing_fields
    
    def test_state_auto_fill(self):
        """Test state auto-fill"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        checker = StateCompletenessChecker(auto_fill=True)
        
        state = {"question": "Test?", "question_id": "q1"}
        updated, result = checker.ensure_completeness(state, "n3_knowledge_retrieval")
        
        # Should auto-fill missing fields
        assert len(result.get('auto_filled', {})) > 0 or result['is_complete']


class TestQueryCache:
    """Test query cache"""
    
    def test_cache_set_get(self):
        """Test cache set and get"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        cache = QueryCache(persist=False)
        
        cache.set("test query", {"result": "data"})
        result = cache.get("test query")
        
        assert result == {"result": "data"}
    
    def test_cache_miss(self):
        """Test cache miss"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        cache = QueryCache(persist=False)
        
        result = cache.get("nonexistent query")
        
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        cache = QueryCache(persist=False)
        
        cache.set("q1", "result1")
        cache.get("q1")  # hit
        cache.get("q2")  # miss
        
        stats = cache.get_stats()
        
        assert stats.hits == 1
        assert stats.misses == 1


class TestCheckpointManager:
    """Test checkpoint manager"""
    
    def test_session_checkpoint(self):
        """Test session checkpoint"""
        if not OPTIMIZATIONS_AVAILABLE:
            pytest.skip("Optimizations not available")
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            
            manager.start_session("s1", "q1", "Test question?")
            manager.save_node_checkpoint("n1", {"data": "test"})
            
            assert manager.current_session is not None
            assert "n1" in manager.current_session.node_checkpoints


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

