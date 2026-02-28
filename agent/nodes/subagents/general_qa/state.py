from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class GeneralQAState(BaseModel):
    """General QA Agent State - Complete state model for all 12 nodes
    
    Enhanced with:
    - Self-Consistency multi-path reasoning
    - Chain-of-Thought explicit reasoning chain
    - Calculation cross-verification
    - Iterative knowledge retrieval
    - Meta-cognitive monitoring
    - Smart exception diagnosis
    """
    
    # ========== N0: Input Preprocessing & Question Classification ==========
    user_input: str = Field(description="User's original input (question text, may include options/experimental data)")
    cleaned_text: Optional[str] = Field(default=None, description="Cleaned question text after preprocessing")
    question_type_label: Optional[str] = Field(default=None, description="Question type label: Multiple Choice/Text Matching/Mechanism Explanation/Numerical Calculation/Logical Calculation/Professional Algorithm")
    question_category_standard: Optional[str] = Field(default=None, description="Standardized question category: Calculation-[subcategory], ClinicalDecision-[subcategory], ProfessionalKnowledge-[subcategory]")
    category_specific_constraints: Optional[List[str]] = Field(default=None, description="Category-specific constraints for solving this type of question (e.g., '必须完成数值推导' for Calculation, '适配最新临床指南' for ClinicalDecision)")
    data_completeness_label: Optional[str] = Field(default=None, description="Data completeness label: Complete/Partial Missing/Severe Missing")
    question_options: Optional[List[str]] = Field(default=None, description="Extracted options for multiple choice questions (option text only)")
    answer_format_label: Optional[str] = Field(default=None, description="Expected answer format: Single Choice/Multi-Select/Numeric/Short Text/Long Text/Sequence/Formula/List/Procedure/Code-Command")
    core_keywords: Optional[List[str]] = Field(default=None, description="Core keywords essential for answering the question (e.g., Fst, genetic differentiation, T cell, antigen presentation, X0, meiosis)")
    option_features: Optional[Dict[str, str]] = Field(default=None, description="Core feature of each option for multiple choice questions (e.g., {'A': 'Genetic load', 'B': 'XY vs ZW systems'})")
    synonyms: Optional[List[str]] = Field(default=None, description="Normalized retrieval keywords for knowledge retrieval (e.g., ['barrier element', 'insulator', 'chromatin boundary'])")
    tool_intent: Optional[Dict[str, str]] = Field(default=None, description="Tool usage requirements: {'query_go_term': 'YES|NO', 'query_knowledge_graph': 'YES|NO'}")
    
    # ========== Structured Three-Dimensional Information (通用结构化三维度) ==========
    # 核心优化：结构化三维度+通用子字段，全领域适配，不绑定任何领域词
    structured_subject: Optional[Dict[str, Any]] = Field(default=None, description="Subject (主体) - 结构化信息: type (主体类型), attribute (主体属性)")
    structured_condition: Optional[Dict[str, Any]] = Field(default=None, description="Condition (条件) - 结构化信息: type (条件类型), key_features (关键特征)")
    structured_goal: Optional[Dict[str, Any]] = Field(default=None, description="Goal (目标) - 结构化信息: type (目标类型), constraint (目标约束)")
    
    # ========== Domain Enhancement (N0 optimization) ==========
    domain_enhancement: Optional[Dict[str, Any]] = Field(default=None, description="Domain enhancement data from N0 optimization including detected domains, key constraints, critical hints, etc.")
    
    # ========== Phase 2 Optimizations (Multi-Step Reasoning) ==========
    multi_step_recommended: Optional[bool] = Field(default=None, description="Whether multi-step reasoning is recommended for this question")
    problem_type: Optional[str] = Field(default=None, description="Detected problem type for multi-step reasoning (e.g., genetics_calculation, clinical_diagnosis)")
    
    # ========== Metadata (for HLE optimizations and other modules) ==========
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata storage for HLE optimizations and other modules (hle_domain_template, hle_pitfall_count, etc.)")
    
    # ========== N1: Question Decomposition & Domain Localization ==========
    structured_conditions: Optional[Dict[str, Any]] = Field(default=None, description="Structured conditions dictionary extracted from question")

    # ========== Semantic Condition Extraction (NEW: 条件语义结构化) ==========
    semantic_conditions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Semantic/formal representation of problem conditions for Critic verification. Contains: randomness (type: independent_per_sample/uniform), imputation (method: reference_genome, assumption: ancestral_allele), data_constraints (list of specific constraints), statistics_affected (which statistics are affected by conditions)"
    )
    core_domains: Optional[List[str]] = Field(default=None, description="Core domain list identified from question")
    research_objective: Optional[str] = Field(default=None, description="Research objective/purpose extracted from question")
    key_entities: Optional[List[str]] = Field(default=None, description="Key biomedical entities or terms extracted from question")
    answer_constraints: Optional[List[str]] = Field(default=None, description="Explicit answer constraints (units, precision, orientation, count)")
    category_specific_solution_steps: Optional[List[str]] = Field(default=None, description="Category-specific solution steps template (e.g., for Calculation: ['提取计算参数', '确定公式', '计算数值', '对比临界值', '得出结论'])")
    retrieval_sub_questions: Optional[List[str]] = Field(default=None, description="Retrieval sub-questions for knowledge retrieval (at least 3 questions)")
    critical_constraints: Optional[List[str]] = Field(default=None, description="Critical constraints that significantly affect the answer (e.g., 'k4影响极大', '理想条件下')")
    key_constraints: Optional[List[str]] = Field(default=None, description="Key logical constraints from question text (e.g., 'only 1 homozygous', 'single crossover', 'only for antihypertensive') - core logical conditions that must be used in inference")
    negative_constraints: Optional[List[str]] = Field(default=None, description="Negative constraints (cannot/except/not occur) - constraints that explicitly prohibit certain answers")
    exclusive_constraints: Optional[List[str]] = Field(default=None, description="Exclusive constraints (category 1/only 1/single) - constraints that require exactly one specific answer")
    strong_restrictions: Optional[List[str]] = Field(default=None, description="Strong restrictions (necessarily true/must be) - constraints that impose strict requirements")
    inference_core_restrictions: Optional[List[str]] = Field(default=None, description="Core restrictions needed for inference - merged from negative/exclusive/strong/key constraints")
    auto_retry_count: Optional[int] = Field(default=0, description="Auto-retry count for node interruption recovery")
    n7_to_n6_retry_count: Optional[int] = Field(default=0, description="Retry count specifically for N7->N6 retry path to prevent infinite loops")
    node_visit_count: Optional[Dict[str, int]] = Field(default=None, description="Track number of visits to each node to prevent infinite loops")
    
    # ========== N2: Calculation/Algorithm Requirement Recognition ==========
    calculation_type_label: Optional[str] = Field(default=None, description="Calculation type label: Numerical/Algorithm")
    key_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Key parameters/formula clues extracted for calculation")
    
    # ========== N3: Cross-Domain Knowledge Retrieval ==========
    domain_knowledge_map: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Domain-knowledge mapping table (domain -> knowledge points)")
    key_facts: Optional[Dict[str, str]] = Field(default=None, description="Key facts for option judgment (e.g., {'molecular_function_1': 'recruits histone acetyltransferase', 'action_mechanism': 'enhances acetylation'})")
    knowledge_validity_label: Optional[str] = Field(default=None, description="Knowledge validity label: Valid/Invalid/Missing (Invalid if fails factuality check)")
    knowledge_unreliable: Optional[bool] = Field(default=None, description="Whether knowledge is unreliable due to tool call failures")
    knowledge_authority_source: Optional[Dict[str, str]] = Field(default=None, description="Authority source for each knowledge point (e.g., 'Lehninger Biochemistry', 'JNC8 Hypertension Guidelines')")
    paperqa_result: Optional[Dict[str, Any]] = Field(default=None, description="PaperQA literature retrieval results including evidence, confidence, papers discovered, etc.")
    deep_research_result: Optional[Dict[str, Any]] = Field(default=None, description="Deep Research analysis results including final report, research brief, etc.")
    parameter_constraints: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Parameter constraints extracted from knowledge base: parameter_name -> {range: {min, max}, sign: positive/negative, physical_constraints: [...]}")
    
    # ========== N4: Calculation Step Decomposition & Formula Matching ==========
    calculation_steps: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of calculation steps")
    matched_formula: Optional[Dict[str, Any]] = Field(default=None, description="Matched formula with parameter definitions")
    unit_conversion_rules: Optional[List[str]] = Field(default=None, description="Unit conversion rules")
    formula_match_result: Optional[str] = Field(default=None, description="Formula match result: Match Success/Match Failed")
    
    # ========== N5: Algorithm Parameter Extraction & Applicability Validation ==========
    algorithm_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Algorithm input parameters dictionary")
    applicability_result: Optional[str] = Field(default=None, description="Applicability result: Applicable/Not Applicable")
    alternative_algorithms: Optional[List[str]] = Field(default=None, description="Alternative algorithm suggestions if current algorithm is not applicable")
    
    # ========== N6: Initial Association Inference ==========
    phenomenon_knowledge_match_table: Optional[Dict[str, Any]] = Field(default=None, description="Phenomenon-knowledge matching table")
    core_molecular_function: Optional[str] = Field(default=None, description="Primary molecular function conclusion (strictly molecular function, NOT biological process)")
    match_confidence_label: Optional[str] = Field(default=None, description="Match confidence label: High/Medium/Low (High only if knowledge is Valid)")
    need_recheck: Optional[bool] = Field(default=None, description="Whether inference needs recheck due to unreliable knowledge")
    category_logic_check: Optional[str] = Field(default=None, description="Category logic check result: Pass/Fail (checks if reasoning follows category-specific solution paradigm)")
    
    # ========== N7: Complete Logical Inference (with dynamic/calculation derivation) ==========
    closed_inference_path: Optional[List[Dict[str, Any]]] = Field(default=None, description="Closed inference pathway (including calculation process, must cover all category_specific_solution_steps)")
    core_conclusion: Optional[str] = Field(default=None, description="Core conclusion from inference (must be based on facts/calculations, not subjective judgment)")
    fact_verification_result: Optional[str] = Field(default=None, description="Fact verification result: '结论与事实一致'/'结论与事实不一致' (from fact_verification step)")
    
    # ========== Enhancement: Self-Consistency 多路径推理 ==========
    inference_paths: Optional[List[Dict[str, Any]]] = Field(default=None, description="Multiple inference paths generated with different temperatures")
    self_consistency_result: Optional[Dict[str, Any]] = Field(default=None, description="Self-consistency voting result: {consensus_answer, consensus_ratio, confidence_level, answer_votes}")
    
    # ========== Enhancement: Chain-of-Thought 显式推理链 ==========
    structured_inference_steps: Optional[List[Dict[str, Any]]] = Field(default=None, description="Structured inference steps with premise, operation, conclusion, and dependencies")
    reasoning_depth: Optional[int] = Field(default=None, description="Reasoning depth metric (number of logical hops)")
    inference_chain_coherent: Optional[bool] = Field(default=None, description="Whether the inference chain is logically coherent")
    
    # ========== Enhancement: Calculation Cross-Verification ==========
    calculation_verification: Optional[Dict[str, Any]] = Field(default=None, description="Calculation verification result: {symbolic_result, numerical_result, all_match, discrepancy}")
    needs_calculation_retry: Optional[bool] = Field(default=None, description="Whether calculation needs to be retried due to verification failure")
    
    # ========== Enhancement: Iterative Knowledge Retrieval ==========
    retrieval_iterations: Optional[int] = Field(default=0, description="Number of knowledge retrieval iterations performed")
    knowledge_gaps_identified: Optional[List[str]] = Field(default=None, description="Knowledge gaps identified during retrieval")
    follow_up_questions: Optional[List[str]] = Field(default=None, description="Follow-up questions generated for iterative retrieval")
    
    # ========== Smart N3 Retry Mechanism ==========
    n3_queried_terms: Optional[Dict[str, List[str]]] = Field(default=None, description="Terms already queried in N3, grouped by source (e.g., {'knowledge_graph': ['Watterson'], 'go_term': ['genetic diversity']})")
    n3_empty_query_count: Optional[int] = Field(default=0, description="Count of consecutive empty query results in N3")
    n3_skip_llm_tools: Optional[bool] = Field(default=False, description="Whether to skip LLM tool queries (when previous attempts returned empty)")
    n3_use_deep_research_only: Optional[bool] = Field(default=False, description="Whether to rely only on Deep Research results (when knowledge base has no relevant content)")
    
    # ========== N3 Loop Optimization: Intelligent Query Strategy ==========
    n3_failed_entities: Optional[Dict[str, List[str]]] = Field(default=None, description="Entities that returned empty results, grouped by entity_type (e.g., {'biological_process': ['Watterson estimator'], 'gene/protein': ['theta']}) - skip these in subsequent queries")
    n3_confidence_history: Optional[List[float]] = Field(default=None, description="History of knowledge confidence values from each N3 visit - used to detect if confidence is improving")
    n3_no_improvement_count: Optional[int] = Field(default=0, description="Count of consecutive N3 visits where confidence did not improve")
    n3_domain_type: Optional[str] = Field(default=None, description="Detected domain type for tool selection: 'genetics', 'statistics', 'clinical', 'biochemistry', 'general'")
    n3_skip_specific_tools: Optional[List[str]] = Field(default=None, description="Tools to skip based on domain type (e.g., ['query_knowledge_graph', 'query_go_term'] for statistics questions)")
    
    # ========== Enhancement: Meta-Cognitive Monitoring ==========
    meta_cognitive_assessment: Optional[Dict[str, Any]] = Field(default=None, description="Meta-cognitive assessment: {goal_alignment, constraint_coverage, knowledge_gaps, reasoning_coherence, needs_backtracking}")
    needs_backtracking: Optional[bool] = Field(default=None, description="Whether the reasoning needs to backtrack to a previous node")
    
    # ========== Enhancement: Smart Exception Diagnosis ==========
    root_cause_diagnosis: Optional[str] = Field(default=None, description="Root cause diagnosis for exception")
    retry_strategy: Optional[Dict[str, Any]] = Field(default=None, description="Smart retry strategy: {target_node, action, params, reason}")
    
    # ========== Enhancement: Tool Intent Recognition ==========
    required_tools: Optional[List[str]] = Field(default=None, description="Tools required for answering this question")
    recommended_tools: Optional[List[str]] = Field(default=None, description="Tools recommended but not required")
    
    # ========== N8: Multi-Type Answer Generation ==========
    structured_answer: Optional[Dict[str, Any]] = Field(default=None, description="Structured answer (option matching table/numerical result/text answer)")
    # X-Masters multi-solution fields
    candidate_answers: Optional[List[Dict[str, Any]]] = Field(default=None, description="Multiple candidate answers generated by N8 (3-5 solutions)")
    num_candidates: Optional[int] = Field(default=3, description="Number of candidate answers to generate (default: 3)")
    
    # ========== N8.5: Critic Review ==========
    critiqued_answers: Optional[List[Dict[str, Any]]] = Field(default=None, description="Critiqued and corrected answers from N8.5")
    
    # ========== N8.6: Rewriter Synthesis ==========
    rewritten_answers: Optional[List[Dict[str, Any]]] = Field(default=None, description="Synthesized answers from N8.6 Rewriter")
    
    # ========== N9: Result Validation & Consistency Judgment ==========
    consistency_label: Optional[str] = Field(default=None, description="Consistency label: Consistent/Inconsistent (Consistent only if format_valid + step_complete + fact_correct all Valid)")
    reliability_score: Optional[float] = Field(default=None, description="Reliability score (1-5): 5 only if all checks pass, deduct 2 for format/step issues, deduct 5 for fact errors")
    format_valid_label: Optional[str] = Field(default=None, description="Answer format validity: Valid/Invalid")
    format_issues: Optional[List[str]] = Field(default=None, description="List of detected answer format issues")
    step_complete_label: Optional[str] = Field(default=None, description="Step completeness label: Valid/Invalid (checks if n7 steps cover all category_specific_solution_steps)")
    fact_correct_label: Optional[str] = Field(default=None, description="Fact correctness label: Valid/Invalid (checks if answer matches authoritative facts/calculations/guidelines)")
    fact_check_result: Optional[Dict[str, Any]] = Field(default=None, description="Fact check result: {'correct_function': '...', 'answer_matches_fact': true|false, 'error_reason': '...'}")
    
    # ========== N10: Knowledge/Calculation Exception Handling ==========
    exception_type_label: Optional[str] = Field(default=None, description="Exception type label: Knowledge Missing/Formula Match Failed/Algorithm Not Applicable/Inference Path Inconsistent/Calculation Result Abnormal")
    solution_suggestion: Optional[str] = Field(default=None, description="Solution suggestion: Retry/Manual Intervention")
    retry_count: Optional[int] = Field(default=0, description="Number of retries attempted (max 2)")
    retry_target_node: Optional[str] = Field(default=None, description="Target node to retry (e.g., 'n7_complete_inference', 'n8_answer_generation')")
    
    # ========== N11: Manual Intervention Trigger ==========
    manual_intervention_guide: Optional[str] = Field(default=None, description="Manual intervention guide")
    intermediate_result_snapshot: Optional[Dict[str, Any]] = Field(default=None, description="Intermediate result snapshot for continuation")
    
    # ========== Tool Usage Tracking ==========
    tool_calls_history: Optional[List[Dict[str, Any]]] = Field(default=None, description="History of all tool calls made by LLM across all nodes, including tool name, arguments, results, and node context")
    
    # ========== Answer Cache System ==========
    # Cache hit status
    cache_hit: Optional[bool] = Field(default=None, description="Whether a valid cache was found for this question")
    cached_answer: Optional[str] = Field(default=None, description="Cached answer if cache hit")
    cached_reasoning: Optional[List[str]] = Field(default=None, description="Cached reasoning path if cache hit")
    cache_confidence_modifier: Optional[float] = Field(default=None, description="Confidence modifier from cache validity check")
    
    # Error cache utilization
    error_cache_found: Optional[bool] = Field(default=None, description="Whether an error analysis cache was found")
    error_warnings_from_cache: Optional[List[str]] = Field(default=None, description="Error warnings to inject from error cache")
    missing_knowledge_from_cache: Optional[List[str]] = Field(default=None, description="Missing knowledge identified from error cache")
    reasoning_trap_from_cache: Optional[str] = Field(default=None, description="Reasoning trap identified from error cache")
    correct_direction_from_cache: Optional[str] = Field(default=None, description="Correct direction hint from error cache")
    
    # Cache trigger metadata
    should_cache_result: Optional[bool] = Field(default=None, description="Whether this result should be cached")
    cache_correctness_known: Optional[bool] = Field(default=None, description="Whether we know if the answer is correct (from test)")
    ground_truth_answer: Optional[str] = Field(default=None, description="Ground truth answer for cache validation")
    
    # ========== Final Output ==========
    final_answer: Optional[str] = Field(default=None, description="Final answer to return to user")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")

