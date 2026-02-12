from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class GeneralQAState(BaseModel):
    """General QA Agent State - Complete state model for all 12 nodes"""
    
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
    
    # ========== N1: Question Decomposition & Domain Localization ==========
    structured_conditions: Optional[Dict[str, Any]] = Field(default=None, description="Structured conditions dictionary extracted from question")
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
    
    # ========== Final Output ==========
    final_answer: Optional[str] = Field(default=None, description="Final answer to return to user")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")

