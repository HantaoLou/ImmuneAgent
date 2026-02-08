"""
General QA Agent Subgraph

Handles user's general question-answering requests, using LLM to provide scientific and rigorous answers.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import sys
import json
import re
from pathlib import Path

from .enums import QuestionType, Domain, ReasoningStrategy
from .prompt import (
    get_question_parsing_prompt,
    get_knowledge_activation_prompt,
    get_knowledge_activation_prompt_with_research,
    get_experiment_analysis_prompt,
    get_logical_derivation_prompt,
    get_final_validation_prompt,
    format_final_answer,
    get_question_format_type_prompt,
    get_option_extraction_prompt,
    get_key_info_extraction_prompt,
    get_domain_classification_prompt,
    get_comprehensive_parsing_prompt,
    get_constraint_tagging_prompt
)
from .reasoning_rules import get_reasoning_rule
from .validation_rules import detect_logical_contradictions
from .json_fixer import fix_json_format, generate_format_fix_prompt
from .universal_matching_engine import universal_matching_engine, match_multiple_knowledge_points
from .constraint_normalizer import normalize_constraint_list

# Import main graph state (for state mapping)
# Add agent directory to path (support import from subgraph directory)
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState

# LLM-related imports (using common LLM factory)
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None
    HumanMessage = None
    SystemMessage = None
    print("Warning: langchain libraries not installed, general QA functionality will be unavailable")


# ---------------------- General QA State Model ----------------------
class GeneralQAState(BaseModel):
    """General QA Agent subgraph state"""
    # Original input
    user_input: str = Field(description="User's original input (question)")
    question_options: List[str] = Field(default_factory=list, description="Question options list (if any)")
    
    # Question parsing node output
    question_format_type: Optional[str] = Field(default=None, description="Question format type: Short Answer/Judgment/Multiple Choice")
    question_type: Optional[str] = Field(default=None, description="Question content type: Judgment/Calculation/Analysis/Enumeration")
    domain: Optional[str] = Field(default=None, description="Question domain (major category): Immunology/Cell Biology/Biochemistry/etc.")
    subdomain: Optional[str] = Field(default=None, description="Question subdomain (specific category): BCR/TCR/etc.")
    key_info: Dict[str, str] = Field(
        default_factory=dict, 
        description="""
        Extracted structured key information, containing 4 dimensions:
        - Analysis Object: Core indicators/substances studied in the question (e.g., Watterson's θ, T cells, mRNA)
        - Experimental Conditions: Interventions/treatments (e.g., random SNV filtering, drug treatment, gene knockout)
        - Constraints: Prerequisites specified in the question (e.g., arbitrarily large sample size, no completely missing sites, 37°C)
        - Target Output: Specific output required by the question (e.g., judge which statistic has bias, calculate π value)
        """
    )
    entities: List[str] = Field(default_factory=list, description="Identified entities (e.g., proteins, genes)")
    constraint_tags: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="""
        Constraint tags extracted from question constraints, categorized into 3 types:
        - site_constraints: Global site constraints (e.g., ["no_global_missing_snp", "only_snp_no_other_variants"])
        - sample_operation_constraints: Sample operation constraints (e.g., ["random_filtering", "reference_imputation", "minority_filtering"])
        - sample_size_constraints: Sample size constraints (e.g., ["large_sample", "small_sample", "unknown_sample_size"])
        """
    )
    # Generic architecture: Constraint Priority (C1/C2)
    core_constraints: List[str] = Field(
        default_factory=list,
        description="""
        Core constraints (C1): Necessary and sufficient conditions that determine the knowledge conclusion.
        If C1 is not satisfied, the conclusion cannot hold. If C1 is satisfied, the conclusion can hold.
        C1 is used for precise matching with knowledge condition sets (Kc).
        """
    )
    secondary_constraints: List[str] = Field(
        default_factory=list,
        description="""
        Secondary constraints (C2): Only affect conclusion details, do not change the core conclusion.
        C2 affects specific values, expressions, scope of application, etc., but not core judgments.
        """
    )
    # Universal constraint hierarchy (通用约束分层结构)
    constraint_hierarchy: Dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Universal constraint hierarchy structure (通用约束分层字段，所有知识点通用):
        {
            "C1_core_constraint": [],  # 核心约束：由process_data节点根据知识Kc自动填充
            "C2_secondary_constraint": [],  # 次要约束：从题目中提取的所有自然语言约束
            "constraint_extract_rule": "从题目中提取所有与数据操作/数据完整性/样本特征相关的约束，放入C2"
        }
        """
    )
    parse_error: Optional[str] = Field(default=None, description="Parsing error message (if any)")
    
    # Knowledge activation node output
    domain_knowledge: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="""
        Activated domain knowledge, key is "Analysis Object" (e.g., CD8+ T cells, Watterson's θ), value is structured knowledge with 4 required dimensions:
        - Core Definition: Academic definition or essential description of the object
        - Key Rules/Properties: Core principles affecting subsequent reasoning (e.g., formulas, functions, mechanisms)
        - Association with Experimental Conditions: Object's response to common experimental treatments (e.g., effects of antibody treatment)
        - Scenario Constraint Mapping Table: Knowledge-scenario binding table showing how object properties/behaviors vary under different constraint combinations (based on constraint_tags)
          Format: List of dictionaries, each dict contains:
            - constraint_combination: List of constraint tags (e.g., ["large_sample", "random_filtering", "no_global_missing_snp"])
            - Additional dynamic fields: Field names and values should be determined based on the specific question context and analysis objects
              * For population genetics questions with multiple estimators: fields like "Watterson's θ 偏倚性", "π（核苷酸多样性）偏倚性", "偏倚方向"
              * For other questions: fields should reflect the key properties/behaviors relevant to the question
              * Field names are NOT fixed (parameter_1, parameter_2, parameter_3 are just examples, not requirements)
        """
    )
    # Generic architecture: Conditionalized Knowledge (Kc→Kr)
    conditionalized_knowledge: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="""
        Conditionalized knowledge in standardized format: Kc → Kr
        Each entry contains:
        - Kc (condition_set): Set of core constraints (C1) that must be satisfied for the conclusion to hold
          Format: List[str] of constraint identifiers (e.g., ["large_sample", "random_filtering", "no_global_missing_snp"])
        - Kr (conclusion): The conclusion that holds when Kc is satisfied
          Format: str describing the conclusion (can be judgment, value, direction, mechanism, etc.)
        - knowledge_id: Unique identifier for this knowledge entry
        - analysis_object: The analysis object this knowledge applies to
        - metadata: Additional metadata (e.g., source, confidence, domain)
        """
    )
    knowledge_error: Optional[str] = Field(default=None, description="Knowledge activation error message (if any)")
    # Legacy fields for compatibility (deprecated)
    activated_modules: List[str] = Field(default_factory=list, description="Activated knowledge modules list (deprecated)")
    knowledge_context: Dict[str, Any] = Field(default_factory=dict, description="Knowledge module context (deprecated)")
    
    # Data processing node output
    experiment_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""
        Experiment condition impact analysis, containing 4 required dimensions:
        - Operation Breakdown: Break down "Experimental Conditions" into specific analyzable operations (e.g., "anti-CTLA-4 treatment", "37°C incubation")
        - Knowledge Association Basis: Domain knowledge corresponding to each operation (from second node)
        - Impact Direction Judgment: Operation's impact on analysis object (e.g., "promotes proliferation", "no effect", "causes bias")
        - Core Hypothesis Validation: Whether implicit assumptions in question are reasonable (e.g., "filtering does not lose segregating sites in large samples")
        """
    )
    # Generic architecture: Match Degree Judgment Results
    match_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="""
        Match degree judgment results: C1 ⊇ Kc matching results
        Each entry contains:
        - knowledge_id: Identifier of the matched knowledge entry
        - match_type: "full_match" (C1 = Kc), "partial_match" (C1 ⊇ Kc with extra elements), "no_match" (C1 does not contain Kc)
        - matched_Kr: The conclusion (Kr) from the matched knowledge entry (if matched)
        - match_score: Match quality score (0.0 to 1.0)
        - matched_constraints: List of constraints from Kc that were matched in C1
        - unmatched_constraints: List of constraints from Kc that were not in C1 (if partial match)
        - extra_constraints: List of constraints in C1 that are not in Kc (if partial match)
        """
    )
    # Universal matching result (通用匹配结果，由process_data节点生成)
    universal_matching_result: Dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Universal matching result from universal matching engine (通用匹配结果，适配所有知识点):
        {
            "knowledge_point_1": {
                "constraint_hierarchy_updated": {...},
                "preliminary_conclusion_universal": "...",
                "matching_info": {...}
            },
            "knowledge_point_2": {...}
        }
        """
    )
    analysis_error: Optional[str] = Field(default=None, description="Data processing node analysis error (if any)")
    # Legacy fields for compatibility (deprecated)
    experimental_conditions: Dict[str, Any] = Field(default_factory=dict, description="Identified experimental conditions (deprecated)")
    potential_errors: List[str] = Field(default_factory=list, description="Identified potential error sources (deprecated)")
    data_quality: Optional[str] = Field(default=None, description="Data quality assessment (deprecated)")
    
    # Reasoning engine node output
    logical_derivation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""
        Logical derivation result for the question, adapted to different dimensions based on question type:
        1. General dimensions (required for all types):
           - Derivation Strategy: Reasoning method adapted to question type (e.g., "impact direction → option matching", "formula substitution calculation")
           - Core Evidence Chain: Data from previous nodes that derivation depends on (e.g., "Node 3 impact judgment → Node 2 formula")
           - Preliminary Conclusion: Derivation conclusion before option matching (e.g., "experimental group proliferation rate higher than control", "π is underestimated")
        2. Type-specific dimensions:
           - Judgment Type: Need to add "Option Matching Priority" (e.g., "A>B>C, because A matches impact direction")
           - Calculation Type: Need to add "Formula Substitution Process", "Numerical Calculation Result"
           - Analysis Type: Need to add "Causal Chain Analysis" (e.g., "antibody → blocks negative signal → proliferation increases")
           - Enumeration Type: Need to add "Enumeration Results"
        """
    )
    # Generic architecture: Precise Conclusion vs Generalized Reasoning
    precise_conclusion: Optional[str] = Field(
        default=None,
        description="""
        Precise conclusion (Kr) from matched knowledge entry.
        This is used when C1 ⊇ Kc match is successful (full_match or partial_match).
        If this field is set, it should be used as the primary conclusion instead of generalized reasoning.
        """
    )
    generalized_reasoning: Optional[str] = Field(
        default=None,
        description="""
        Generalized reasoning result from LLM natural language inference.
        This is used when no C1 ⊇ Kc match is found (no_match).
        Contains the reasoning process and conclusion generated through LLM inference.
        """
    )
    derivation_error: Optional[str] = Field(default=None, description="Reasoning engine node derivation error (if any)")
    derivation_warning: Optional[str] = Field(
        default=None,
        description="""
        Derivation warning from conclusion validation node. If set, indicates that preliminary conclusion failed substantive validation:
        - "constraint_mapping_mismatch": Preliminary conclusion is inconsistent with constraint-knowledge mapping table results
        - "rebuttal_evidence_contradiction": Preliminary conclusion contradicts rebuttal evidence from reasoning engine
        - "both_validations_failed": Both validations failed
        If derivation_warning is set, the system should return to reasoning_engine_node for re-derivation.
        """
    )
    # Legacy fields for compatibility (deprecated)
    reasoning_strategy: Optional[ReasoningStrategy] = Field(default=None, description="Selected reasoning strategy (deprecated)")
    reasoning_steps: List[str] = Field(default_factory=list, description="Reasoning steps (deprecated)")
    intermediate_conclusions: List[str] = Field(default_factory=list, description="Intermediate conclusions (deprecated)")
    
    # Conclusion validation node output
    final_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""
        Five-step method final result, containing 5 required dimensions (closed-loop key):
        - Final Answer: Matched question option (e.g., "A", "B" or specific numerical value)
        - Validation Results: Multi-dimensional validation conclusions (previous logic consistency/biological reasonableness/option matching accuracy)
        - Complete Reasoning Chain: Core contributions of five nodes in sequence (Node 1 → Node 2 → Node 3 → Node 4 → Node 5)
        - Key Evidence Summary: Core supporting information from all nodes (e.g., domain knowledge formulas, experimental impact judgments)
        - Common Pitfall Reminders: Easy-to-confuse points in question or exclusion reasons for wrong options (e.g., "Option D wrong because ignores large sample assumption")
        """
    )
    # Universal verification result (通用验证结果)
    universal_verification_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""
        Universal verification result (三层通用实质验证规则结果):
        {
            "universal_verification_result": "PASSED" | "FAILED",
            "verification_rules": [...],
            "verification_details": {
                "Rule1_check": "PASSED" | "FAILED",
                "Rule2_check": "PASSED" | "FAILED",
                "Rule3_check": "PASSED" | "FAILED",
                "matching_accuracy": 100.0
            },
            "verification_failed_reason": "..." (if FAILED)
        }
        """
    )
    final_error: Optional[str] = Field(default=None, description="Conclusion validation node final processing error (if any)")
    # Legacy fields for compatibility (deprecated)
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results (logical, biological, statistical) (deprecated)")
    answer_options: List[str] = Field(default_factory=list, description="Answer options (if any) (deprecated)")
    matched_option: Optional[str] = Field(default=None, description="Matched answer option (deprecated)")
    
    # Final output
    answer: Optional[str] = Field(default=None, description="LLM-generated answer")
    confidence: Optional[str] = Field(default=None, description="Answer confidence description")
    related_topics: List[str] = Field(default_factory=list, description="Related questions or topics")
    sources_suggested: List[str] = Field(default_factory=list, description="Suggested references or research directions")
    output_template: Optional[str] = Field(default="researcher", description="Output template type: student/researcher/clinician/custom")
    custom_output_fields: Optional[List[str]] = Field(default=None, description="Custom output fields (required if template is 'custom')")


# ---------------------- LLM Instantiation (using common LLM factory) ----------------------
def _get_llm():
    """
    Get reasoning model instance (for general QA)
    
    Uses common LLM factory to create reasoning model, prioritizing models with good reasoning performance.
    
    Returns:
        LLM instance, returns None if unavailable
    """
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        return None
    
    # Use reasoning model (for general QA, slightly higher temperature for more natural answers)
    return create_reasoning_llm(temperature=0.3)


# ---------------------- Node 1: Question Parsing Node (Optimized) ----------------------
def question_parsing_node(state: GeneralQAState) -> GeneralQAState:
    """
    Question parsing node (optimized with step-by-step LLM analysis)
    
    New workflow:
    1. Determine question format type (Short Answer/Judgment/Multiple Choice) - using LLM
    2. If Multiple Choice, extract options - using LLM
    3. Extract key information (4 dimensions) - using LLM
    4. Classify domain (two-level: major domain and subdomain) - using LLM
    5. Determine question content type (Judgment/Calculation/Analysis/Enumeration) - using LLM
    6. Update state with all parsed information
    
    Args:
        state: General QA subgraph state
    
    Returns:
        Updated state (containing question format type, question type, options, key information, domain, subdomain, etc.)
    """
    # Input validation
    if not state.user_input or not state.user_input.strip():
        raise ValueError("user_input cannot be empty")
    
    print("=" * 60)
    print("Node 1: Question Parsing (Step-by-step LLM Analysis)")
    print("=" * 60)
    
    # Get LLM instance
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable, cannot parse question. Please check LLM configuration.")
    
    # Step 1: Determine question format type (Short Answer/Judgment/Multiple Choice)
    print("\n[Step 1] Determining question format type...")
    format_type_prompt = get_question_format_type_prompt(state.user_input)
    format_type_response = _call_llm_for_parsing(llm, format_type_prompt)
    if not format_type_response:
        raise RuntimeError("LLM call failed for format type determination.")
    
    try:
        format_type_data, repair_level = fix_json_format(format_type_response, required_keys=["question_format_type"])
        if repair_level != "none":
            print(f"⚠ JSON format repaired at {repair_level} level")
        question_format_type = format_type_data.get("question_format_type", "Unknown")
        print(f"✓ Question format type: {question_format_type}")
    except ValueError as e:
        raise ValueError(f"Cannot parse format type response: {str(e)}")
    
    # Step 2: Extract options if Multiple Choice
    question_options = state.question_options if state.question_options else []
    if question_format_type == "Multiple Choice" and not question_options:
        print("\n[Step 2] Extracting options from multiple-choice question...")
        option_prompt = get_option_extraction_prompt(state.user_input)
        option_response = _call_llm_for_parsing(llm, option_prompt)
        if option_response:
            try:
                option_data, repair_level = fix_json_format(option_response, required_keys=["options"])
                if repair_level != "none":
                    print(f"⚠ JSON format repaired at {repair_level} level")
                extracted_options = option_data.get("options", [])
                if extracted_options:
                    question_options = extracted_options
                    print(f"✓ Extracted {len(question_options)} options")
            except ValueError as e:
                print(f"⚠ Failed to parse options: {str(e)}, continuing without options")
        else:
            print("⚠ Failed to extract options, continuing without options")
    elif question_format_type != "Multiple Choice":
        print(f"\n[Step 2] Skipping option extraction (format type: {question_format_type})")
    
    # Step 3: Extract key information
    print("\n[Step 3] Extracting key information...")
    key_info_prompt = get_key_info_extraction_prompt(state.user_input, question_options)
    key_info_response = _call_llm_for_parsing(llm, key_info_prompt)
    if not key_info_response:
        raise RuntimeError("LLM call failed for key information extraction.")
    
    try:
        key_info_data, repair_level = fix_json_format(key_info_response, required_keys=["key_info"])
        if repair_level != "none":
            print(f"⚠ JSON format repaired at {repair_level} level")
        key_info = key_info_data.get("key_info", {})
        
        # Validate key_info has required dimensions
        required_dims = ["Analysis Object", "Experimental Conditions", "Constraints", "Target Output"]
        missing_dims = [dim for dim in required_dims if dim not in key_info]
        if missing_dims:
            raise ValueError(f"Key info missing required dimensions: {missing_dims}")
        
        print(f"✓ Key information extracted ({len(key_info)} dimensions)")
        for dim in required_dims:
            if key_info.get(dim) and key_info[dim] not in ["None", "", "N/A"]:
                print(f"  - {dim}: {key_info[dim][:50]}...")
    except ValueError as e:
        raise ValueError(f"Cannot parse key information response: {str(e)}")
    
    # Step 3.5: Parse and tag constraints (new constraint tagging module)
    print("\n[Step 3.5] Parsing and tagging constraints...")
    constraints_text = key_info.get("Constraints", "").strip()
    constraint_tags = {
        "site_constraints": [],
        "sample_operation_constraints": [],
        "sample_size_constraints": []
    }
    
    if constraints_text and constraints_text not in ["None", "", "N/A", "Not mentioned"]:
        constraint_tagging_prompt = get_constraint_tagging_prompt(state.user_input, constraints_text)
        constraint_tagging_response = _call_llm_for_parsing(llm, constraint_tagging_prompt)
        if constraint_tagging_response:
            try:
                constraint_tags_data, repair_level = fix_json_format(
                    constraint_tagging_response,
                    required_keys=["site_constraints", "sample_operation_constraints", "sample_size_constraints"]
                )
                if repair_level != "none":
                    print(f"⚠ JSON format repaired at {repair_level} level")
                
                constraint_tags = {
                    "site_constraints": constraint_tags_data.get("site_constraints", []),
                    "sample_operation_constraints": constraint_tags_data.get("sample_operation_constraints", []),
                    "sample_size_constraints": constraint_tags_data.get("sample_size_constraints", [])
                }
                
                # Validate constraint_tags structure
                for key in constraint_tags:
                    if not isinstance(constraint_tags[key], list):
                        constraint_tags[key] = []
                
                total_tags = sum(len(tags) for tags in constraint_tags.values())
                print(f"✓ Constraint tags extracted ({total_tags} tags total)")
                for tag_type, tags in constraint_tags.items():
                    if tags:
                        print(f"  - {tag_type}: {', '.join(tags)}")
            except ValueError as e:
                print(f"⚠ Failed to parse constraint tags: {str(e)}, using empty tags")
                constraint_tags = {
                    "site_constraints": [],
                    "sample_operation_constraints": [],
                    "sample_size_constraints": []
                }
        else:
            print("⚠ Failed to extract constraint tags, using empty tags")
    else:
        print("⚠ No constraints found in question, using empty tags")
    
    # Step 4: Classify domain (two-level: major domain and subdomain)
    print("\n[Step 4] Classifying domain (two-level)...")
    domain_prompt = get_domain_classification_prompt(state.user_input, key_info)
    domain_response = _call_llm_for_parsing(llm, domain_prompt)
    if not domain_response:
        raise RuntimeError("LLM call failed for domain classification.")
    
    try:
        domain_data, repair_level = fix_json_format(domain_response, required_keys=["domain"])
        if repair_level != "none":
            print(f"⚠ JSON format repaired at {repair_level} level")
        domain = domain_data.get("domain", "General")
        subdomain = domain_data.get("subdomain", "Not specified")
        print(f"✓ Domain: {domain}")
        print(f"✓ Subdomain: {subdomain}")
    except ValueError as e:
        print(f"⚠ Failed to parse domain classification: {str(e)}, using defaults")
        domain = "General"
        subdomain = "Not specified"
    
    # Step 5: Determine question content type (Judgment/Calculation/Analysis/Enumeration)
    print("\n[Step 5] Determining question content type...")
    # Use comprehensive prompt that includes content type determination
    content_type_prompt = get_question_parsing_prompt(state.user_input, question_options)
    content_type_response = _call_llm_for_parsing(llm, content_type_prompt)
    if not content_type_response:
        raise RuntimeError("LLM call failed for content type determination.")
    
    try:
        content_type_data, repair_level = fix_json_format(content_type_response, required_keys=["question_type"])
        if repair_level != "none":
            print(f"⚠ JSON format repaired at {repair_level} level")
        question_type_raw = content_type_data.get("question_type", "Unknown")
        # Normalize question type
        question_type_mapping = {
            "Judgment Type": "Judgment",
            "Calculation Type": "Calculation",
            "Analysis Type": "Analysis",
            "Enumeration Type": "Enumeration"
        }
        question_type = question_type_mapping.get(question_type_raw, question_type_raw)
        print(f"✓ Question content type: {question_type}")
    except ValueError as e:
        print(f"⚠ Failed to parse content type: {str(e)}, using default")
        question_type = "Analysis"  # Default fallback
    
    # Step 6: Initialize universal constraint hierarchy (通用约束分层初始化)
    print("\n[Step 6] Initializing universal constraint hierarchy...")
    # 提取所有约束到C2，C1留空（由process_data节点根据知识Kc填充）
    all_constraints = []
    if constraints_text and constraints_text not in ["None", "", "N/A", "Not mentioned"]:
        # 从constraint_tags中提取所有约束
        for tag_type, tags in constraint_tags.items():
            all_constraints.extend(tags)
        # 如果constraint_tags为空，尝试从key_info的Constraints字段提取
        if not all_constraints:
            # 简单提取：将Constraints文本作为单个约束（后续可优化）
            all_constraints = [constraints_text]
    
    constraint_hierarchy = {
        "C1_core_constraint": [],  # 核心约束：空列表，由process_data节点根据知识Kc填充
        "C2_secondary_constraint": all_constraints,  # 次要约束：提取所有自然语言约束
        "constraint_extract_rule": "从题目中提取所有与数据操作/数据完整性/样本特征相关的约束，放入C2"
    }
    print(f"✓ Constraint hierarchy initialized: {len(all_constraints)} constraints in C2, C1 empty (will be filled by process_data)")
    
    # Step 7: Update state with all parsed information
    print("\n[Step 7] Updating state...")
    state.question_format_type = question_format_type
    state.question_type = question_type
    state.question_options = question_options
    state.key_info = key_info
    state.constraint_tags = constraint_tags
    state.constraint_hierarchy = constraint_hierarchy
    state.domain = domain
    state.subdomain = subdomain
    state.parse_error = None
    
    print("\n" + "=" * 60)
    print("✓ Question parsing completed successfully!")
    print("=" * 60)
    print(f"  Format type: {state.question_format_type}")
    print(f"  Content type: {state.question_type}")
    print(f"  Domain: {state.domain}")
    print(f"  Subdomain: {state.subdomain}")
    print(f"  Options: {len(state.question_options)} options" if state.question_options else "  Options: None")
    print(f"  Key info dimensions: {len(state.key_info)}")
    total_constraint_tags = sum(len(tags) for tags in state.constraint_tags.values())
    print(f"  Constraint tags: {total_constraint_tags} tags ({len(state.constraint_tags.get('site_constraints', []))} site, {len(state.constraint_tags.get('sample_operation_constraints', []))} operation, {len(state.constraint_tags.get('sample_size_constraints', []))} size)")
    
    return state


def _call_llm_for_parsing(llm, prompt: str) -> Optional[str]:
    """
    Call LLM for question parsing (with retry mechanism)
    
    Args:
        llm: LLM instance
        prompt: Prompt text
        
    Returns:
        LLM returned text, returns None if failed
    """
    try:
        # Use lower temperature for more stable structured output
        messages = [
            HumanMessage(content=prompt)
        ]
        
        # Try to call LLM (use lower temperature for more stable JSON output)
        # Note: Need to adjust based on actual LLM interface
        # If LLM supports temperature parameter, should set it during creation, here use default
        response = llm.invoke(messages)
        response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        return response_text
        
    except Exception as e:
        print(f"⚠ LLM call exception: {type(e).__name__}: {str(e)[:100]}")
        return None


def _extract_question_core(question_text: str) -> str:
    """
    Helper function: Extract core question from question text (e.g., "Is experimental group proliferation rate higher than control?"), remove redundant descriptions
    
    Args:
        question_text: Full question text
    
    Returns:
        Core question string
    """
    # Simplified logic: Extract sentence starting with "Question:", "Ask:", "Q:", etc.
    question_markers = ["Question:", "Ask:", "Q:", "What", "How", "Which", "Why", "Is", "Are", "Does", "Do"]
    for marker in question_markers:
        if marker in question_text:
            core = question_text.split(marker)[-1].strip()
            # Remove option references (e.g., "Which is correct? A.xxx B.xxx" → keep "Which is correct?")
            if any(opt in core for opt in ["A.", "B.", "C.", "D.", "E."]):
                parts = [part for part in core.split() if not part.startswith(("A.", "B.", "C.", "D.", "E."))]
                if parts:
                    core = " ".join(parts[:10])  # Take first 10 words
            return core
    
    # Fallback: Take last 100 characters
    return question_text[-100:].strip()


def _normalize_item_name(item: str) -> str:
    """
    Normalize item names for bias claim matching (generic approach)
    
    This function normalizes item names to enable matching of variations.
    Uses a generic approach that works for any item name, not specific to particular problems.
    
    Args:
        item: Item name (e.g., "theta", "watterson", "pi", "T cell", "BCR", "θ", "π")
    
    Returns:
        Normalized item name (lowercase, stripped, with special character normalization)
    """
    # Generic normalization: lowercase and strip
    normalized = item.lower().strip()
    
    # Normalize common special characters and their text equivalents
    # This helps match "θ" with "theta", "π" with "pi", etc.
    special_char_mappings = {
        "θ": "theta",
        "θ_w": "theta",
        "watterson's θ": "theta",
        "watterson θ": "theta",
        "π": "pi",
        "nucleotide diversity π": "pi",
        "nucleotide diversity": "pi",
    }
    
    # Check if normalized string contains any special character mappings
    for special_char, normalized_name in special_char_mappings.items():
        if special_char in normalized:
            # Replace the special character with its normalized name
            normalized = normalized.replace(special_char, normalized_name)
            # Also remove possessive forms and extra words
            normalized = normalized.replace("'s", "").replace("'", "")
            # Extract just the core name (e.g., "watterson theta" -> "theta")
            if normalized_name in normalized:
                # If the normalized name appears, try to extract it
                parts = normalized.split()
                if normalized_name in parts:
                    normalized = normalized_name
                elif any(normalized_name in part for part in parts):
                    # Find the part containing the normalized name
                    for part in parts:
                        if normalized_name in part:
                            normalized = normalized_name
                            break
    
    return normalized


def _check_preliminary_final_consistency(
    preliminary_conclusion: str,
    final_answer: str,
    question_type: str,
    question_options: List[str]
) -> Dict[str, Any]:
    """
    Check if Final Answer is consistent with Preliminary Conclusion
    
    Priority 1.2: Post-Validation Consistency Check
    
    Args:
        preliminary_conclusion: Node 4's Preliminary Conclusion
        final_answer: Node 5's Final Answer
        question_type: Question type string
        question_options: List of question options
    
    Returns:
        Dict with keys:
        - is_consistent: bool
        - contradictions: List[str] of contradiction descriptions
        - corrected_answer: Optional[str] - corrected answer if inconsistency found
    """
    contradictions = []
    corrected_answer = None
    
    # Extract key claims from preliminary conclusion
    # Generic approach: Works for bias-related questions, judgment questions, and other types
    # For bias-related: extract which items are biased/unbiased
    # For judgment: extract yes/no, higher/lower, etc.
    # For calculation: extract numerical values and comparisons
    
    prelim_lower = preliminary_conclusion.lower()
    final_lower = final_answer.lower()
    
    # Check if this is a bias-related question (generic detection, not specific to particular problems)
    is_bias_question = any(keyword in prelim_lower or keyword in final_lower 
                          for keyword in ["biased", "bias", "unbiased", "underestimated", "overestimated"])
    
    if is_bias_question:
        # Priority 3.1: Explicit Bias Direction Validation (generic, works for any items)
        # Pattern matching uses generic word patterns that capture any item name
        # These patterns work for any biomedical/statistical concepts, not just specific examples
        # Pattern design: Captures item names (words, phrases, special characters) followed by bias indicators
        bias_patterns = [
            # Pattern 1: "X unbiased" or "X is unbiased" or "X has no bias" (generic item name)
            # Captures item name (1-5 words max to avoid over-matching) followed by unbiased indicators
            (r"([a-zα-ωθπ0-9'\-]+(?:\s+[a-zα-ωθπ0-9'\-]+){0,4})\s+(?:is\s+)?(?:unbiased|not\s+biased|no\s+bias|remains\s+unbiased)", "unbiased"),
            # Pattern 2: "X biased" or "X is biased" or "X underestimated/overestimated" (generic)
            (r"([a-zα-ωθπ0-9'\-]+(?:\s+[a-zα-ωθπ0-9'\-]+){0,4})\s+(?:is\s+)?(?:biased|has\s+bias|underestimated|overestimated|likely\s+underestimated)", "biased"),
            # Pattern 3: "Only X is biased" (generic)
            (r"only\s+([a-zα-ωθπ0-9'\-]+(?:\s+[a-zα-ωθπ0-9'\-]+){0,4})\s+(?:is\s+)?(?:biased|has\s+bias)", "only_biased"),
            # Pattern 4: "Only X is unbiased" (generic)
            (r"only\s+([a-zα-ωθπ0-9'\-]+(?:\s+[a-zα-ωθπ0-9'\-]+){0,4})\s+(?:is\s+)?(?:unbiased|not\s+biased)", "only_unbiased"),
            # Pattern 5: "X remains unbiased, Y is biased" (explicit pairing - generic)
            # This pattern captures two items in a single sentence, useful for comparative statements
            (r"([a-zα-ωθπ0-9'\-]+(?:\s+[a-zα-ωθπ0-9'\-]+){0,4})\s+(?:remains\s+|is\s+)?(?:unbiased|not\s+biased).*?([a-zα-ωθπ0-9'\-]+(?:\s+[a-zα-ωθπ0-9'\-]+){0,4})\s+(?:is\s+|likely\s+)?(?:biased|underestimated)", "pair"),
        ]
        
        prelim_bias_claims = {}
        for pattern, bias_type in bias_patterns:
            matches = re.findall(pattern, prelim_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if bias_type == "pair" and len(match) >= 2:
                        # Handle paired claims: first is unbiased, second is biased
                        item1 = match[0].lower() if match[0] else ""
                        item2 = match[1].lower() if len(match) > 1 and match[1] else ""
                        if item1:
                            prelim_bias_claims[item1] = "unbiased"
                        if item2:
                            prelim_bias_claims[item2] = "biased"
                    else:
                        item = match[0] if match[0] else match[1] if len(match) > 1 else ""
                        if item:
                            # Normalize item names
                            item_normalized = _normalize_item_name(item.lower())
                            prelim_bias_claims[item_normalized] = bias_type
                else:
                    item = match.lower()
                    if item:
                        item_normalized = _normalize_item_name(item)
                        prelim_bias_claims[item_normalized] = bias_type
        
        # Extract bias claims from final answer (using same patterns)
        final_bias_claims = {}
        for pattern, bias_type in bias_patterns:
            matches = re.findall(pattern, final_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if bias_type == "pair" and len(match) >= 2:
                        item1 = match[0].lower() if match[0] else ""
                        item2 = match[1].lower() if len(match) > 1 and match[1] else ""
                        if item1:
                            final_bias_claims[_normalize_item_name(item1)] = "unbiased"
                        if item2:
                            final_bias_claims[_normalize_item_name(item2)] = "biased"
                    else:
                        item = match[0] if match[0] else match[1] if len(match) > 1 else ""
                        if item:
                            final_bias_claims[_normalize_item_name(item.lower())] = bias_type
                else:
                    item = match.lower()
                    if item:
                        final_bias_claims[_normalize_item_name(item)] = bias_type
        
        # Check for contradictions in bias claims
        for item, prelim_bias in prelim_bias_claims.items():
            if item in final_bias_claims:
                final_bias = final_bias_claims[item]
                if prelim_bias == "unbiased" and final_bias in ["biased", "only_biased"]:
                    contradictions.append(f"Preliminary Conclusion states '{item}' is unbiased, but Final Answer claims it is biased")
                elif prelim_bias == "biased" and final_bias in ["unbiased", "only_unbiased"]:
                    contradictions.append(f"Preliminary Conclusion states '{item}' is biased, but Final Answer claims it is unbiased")
        
        # Priority 3.1: Enhanced bias direction validation and correction
        if contradictions and question_options:
            # Extract bias claims from each option
            option_bias_maps = {}
            for option in question_options:
                option_lower = option.lower()
                option_claims = {}
                for pattern, bias_type in bias_patterns:
                    matches = re.findall(pattern, option_lower, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            if bias_type == "pair" and len(match) >= 2:
                                item1 = _normalize_item_name(match[0].lower() if match[0] else "")
                                item2 = _normalize_item_name(match[1].lower() if len(match) > 1 and match[1] else "")
                                if item1:
                                    option_claims[item1] = "unbiased"
                                if item2:
                                    option_claims[item2] = "biased"
                            else:
                                item = _normalize_item_name((match[0] if match[0] else match[1] if len(match) > 1 else "").lower())
                                if item:
                                    option_claims[item] = bias_type
                        else:
                            item = _normalize_item_name(match.lower())
                            if item:
                                option_claims[item] = bias_type
                option_bias_maps[option] = option_claims
            
            # Find option that matches preliminary conclusion
            # First, try to match all items from preliminary conclusion
            best_match_option = None
            best_match_score = 0
            
            for option, option_claims in option_bias_maps.items():
                match_score = 0
                option_matches = True
                
                # Check if option's bias claims match preliminary conclusion
                for item, prelim_bias in prelim_bias_claims.items():
                    # Try to find matching item in option claims (with normalization)
                    matched_item = None
                    for o_item, o_bias in option_claims.items():
                        # Normalize both items for comparison
                        normalized_item = _normalize_item_name(item)
                        normalized_o_item = _normalize_item_name(o_item)
                        
                        # Check if items match (exact match or one contains the other)
                        if normalized_item == normalized_o_item:
                            matched_item = o_item
                            break
                        elif normalized_item in normalized_o_item or normalized_o_item in normalized_item:
                            # Partial match (e.g., "theta" matches "watterson's theta")
                            matched_item = o_item
                            break
                    
                    if matched_item and matched_item in option_claims:
                        option_bias = option_claims[matched_item]
                        # Check for contradiction
                        if prelim_bias == "unbiased" and option_bias in ["biased", "only_biased"]:
                            option_matches = False
                            break
                        elif prelim_bias == "biased" and option_bias in ["unbiased", "only_unbiased"]:
                            option_matches = False
                            break
                        else:
                            # Match found and consistent
                            match_score += 1
                    else:
                        # Generic check: If preliminary conclusion mentions an item but option doesn't explicitly mention it,
                        # check if option's "only X" pattern contradicts preliminary conclusion
                        # e.g., prelim says "item1 unbiased, item2 biased", option says "only item1 biased" -> contradiction
                        if "only" in option.lower():
                            # "Only X biased" implies other items are unbiased
                            only_item = None
                            for o_item in option_claims.keys():
                                # Normalize for comparison
                                normalized_item = _normalize_item_name(item)
                                normalized_o_item = _normalize_item_name(o_item)
                                
                                # Check if this item appears in the option text or matches
                                if normalized_item == normalized_o_item or normalized_item in normalized_o_item or normalized_o_item in normalized_item:
                                    only_item = o_item
                                    break
                                elif o_item in option.lower() or any(word in option.lower() for word in o_item.split()):
                                    # Also check if item name appears in option text
                                    if normalized_item in option.lower() or any(word in option.lower() for word in item.split()):
                                        only_item = o_item
                                        break
                            
                            if only_item:
                                normalized_only = _normalize_item_name(only_item)
                                normalized_item = _normalize_item_name(item)
                                if normalized_only != normalized_item:
                                    # Option says "only X biased" but prelim says "Y biased" -> contradiction if X != Y
                                    if prelim_bias == "biased":
                                        option_matches = False
                                        break
                                elif normalized_only == normalized_item and prelim_bias == "unbiased":
                                    # Option says "only X biased" but prelim says "X unbiased" -> contradiction
                                    option_matches = False
                                    break
                
                if option_matches and match_score > best_match_score:
                    best_match_option = option
                    best_match_score = match_score
            
            if best_match_option:
                corrected_answer = best_match_option
    else:
        # For non-bias questions, use generic semantic consistency check
        # Extract key semantic elements from preliminary conclusion and final answer
        # This is a simplified check - for complex cases, rely on LLM validation in Node 5
        
        # Basic checks for common contradiction patterns (generic, not specific to particular problems)
        # Check for direct negation (e.g., "yes" vs "no", "higher" vs "lower")
        negation_pairs = [
            (["yes", "true", "correct"], ["no", "false", "incorrect"]),
            (["higher", "greater", "more", "increases"], ["lower", "less", "decreases", "reduces"]),
            (["promotes", "enhances", "stimulates"], ["inhibits", "suppresses", "blocks"]),
        ]
        
        for positive_terms, negative_terms in negation_pairs:
            prelim_has_positive = any(term in prelim_lower for term in positive_terms)
            prelim_has_negative = any(term in prelim_lower for term in negative_terms)
            final_has_positive = any(term in final_lower for term in positive_terms)
            final_has_negative = any(term in final_lower for term in negative_terms)
            
            if prelim_has_positive and final_has_negative:
                contradictions.append(f"Preliminary Conclusion suggests positive outcome ({', '.join([t for t in positive_terms if t in prelim_lower][:2])}), but Final Answer suggests negative outcome ({', '.join([t for t in negative_terms if t in final_lower][:2])})")
            elif prelim_has_negative and final_has_positive:
                contradictions.append(f"Preliminary Conclusion suggests negative outcome ({', '.join([t for t in negative_terms if t in prelim_lower][:2])}), but Final Answer suggests positive outcome ({', '.join([t for t in positive_terms if t in final_lower][:2])})")
    
    return {
        "is_consistent": len(contradictions) == 0,
        "contradictions": contradictions,
        "corrected_answer": corrected_answer
    }


def _extract_options_from_text(question_text: str) -> List[str]:
    """
    Helper function: Extract question options from question text
    
    Args:
        question_text: Full question text
    
    Returns:
        List of extracted options, empty list if none found
    """
    options = []
    
    # Pattern 1: "A. xxx", "B. xxx", etc.
    pattern1 = re.compile(r'([A-E])\.\s*([^\n]+?)(?=\n\s*[A-E]\.|$)', re.MULTILINE)
    matches1 = pattern1.findall(question_text)
    if matches1:
        options = [f"{letter}. {text.strip()}" for letter, text in matches1]
        return options
    
    # Pattern 2: "Answer Choices:" followed by options
    if "Answer Choices:" in question_text or "Answer choices:" in question_text:
        choices_section = question_text.split("Answer Choices:")[-1] if "Answer Choices:" in question_text else question_text.split("Answer choices:")[-1]
        # Try to extract A. B. C. etc.
        pattern2 = re.compile(r'([A-E])\.\s*([^\n]+)', re.MULTILINE)
        matches2 = pattern2.findall(choices_section)
        if matches2:
            options = [f"{letter}. {text.strip()}" for letter, text in matches2]
            return options
    
    # Pattern 3: True/False questions
    if question_text.strip().endswith("True") or question_text.strip().endswith("False"):
        if "True" in question_text and "False" in question_text:
            return ["True", "False"]
    
    # Pattern 4: Look for numbered options (1), (2), etc.
    pattern4 = re.compile(r'\((\d+)\)\s*([^\n(]+?)(?=\n\s*\(\d+\)|$)', re.MULTILINE)
    matches4 = pattern4.findall(question_text)
    if matches4:
        options = [f"({num}) {text.strip()}" for num, text in matches4]
        return options
    
    return []


# ---------------------- Node 2: Knowledge Activation Node (Optimized) ----------------------
def knowledge_activation_node(state: GeneralQAState) -> GeneralQAState:
    """
    Knowledge activation node (optimized)
    
    Core tasks:
    1. Locate domain
    2. Call corresponding knowledge module (based on analysis objects)
    3. Validate knowledge scope
    
    Generate structured domain knowledge based on analysis objects using LLM, containing 3 dimensions:
    - Core Definition
    - Key Rules/Properties
    - Association with Experimental Conditions
    
    Args:
        state: General QA subgraph state
    
    Returns:
        Updated state (containing activated domain knowledge)
    """
    # Step 1: Check if first node's output is valid
    if state.parse_error:
        raise ValueError(f"First node parsing failed, cannot activate knowledge: {state.parse_error}")
    
    if not state.key_info or "Analysis Object" not in state.key_info:
        raise ValueError("First node did not extract analysis object, cannot activate knowledge")
    
    # Extract analysis objects
    analysis_object_raw = state.key_info["Analysis Object"].strip()
    if not analysis_object_raw or analysis_object_raw in ["None", "", "N/A", "Not mentioned"]:
        raise ValueError("First node parsed analysis object as empty, no need to activate knowledge")
    
    # Handle multiple analysis objects (e.g., split "Watterson's θ, π" into list)
    if "," in analysis_object_raw:
        analysis_objects = [
            obj.strip() 
            for obj in analysis_object_raw.split(",")
            if obj.strip() and obj.strip() not in ["None", "N/A"]
        ]
    else:
        analysis_objects = [analysis_object_raw]
    
    if not analysis_objects:
        raise ValueError("Cannot extract valid objects from analysis objects")
    
    # Extract additional context from key_info to enhance knowledge activation
    experimental_conditions = state.key_info.get("Experimental Conditions", "").strip()
    constraints = state.key_info.get("Constraints", "").strip()
    target_output = state.key_info.get("Target Output", "").strip()
    
    # Build context string for prompt enhancement
    additional_context_parts = []
    if experimental_conditions and experimental_conditions not in ["None", "", "N/A", "Not mentioned"]:
        additional_context_parts.append(f"Experimental Conditions: {experimental_conditions}")
    if constraints and constraints not in ["None", "", "N/A", "Not mentioned"]:
        additional_context_parts.append(f"Constraints: {constraints}")
    if target_output and target_output not in ["None", "", "N/A", "Not mentioned"]:
        additional_context_parts.append(f"Target Output: {target_output}")
    
    additional_context = "\n".join(additional_context_parts) if additional_context_parts else None
    
    print(f"Activating domain knowledge for analysis objects {analysis_objects}...")
    if additional_context:
        print(f"Additional context from question:")
        for part in additional_context_parts:
            print(f"  - {part}")
    
    # Step 1.5: Try to retrieve deep research context (optional enhancement)
    research_context = ""
    research_brief = ""
    use_research = False  # Default to False, only set to True if research succeeds
    
    # Try to run deep research for enhanced knowledge
    try:
        print("🔍 Attempting deep research to enhance knowledge activation...")
        from nodes.subagents.deep_research import run_deep_research
        import asyncio
        
        # Generate research question from analysis objects and additional context
        research_question_parts = [
            f"Provide comprehensive domain knowledge about: {', '.join(analysis_objects)}"
        ]
        research_question_parts.append("Focus on: 1) Core definitions, 2) Key formulas/rules/properties, 3) Responses to experimental conditions and treatments.")
        
        if experimental_conditions and experimental_conditions not in ["None", "", "N/A", "Not mentioned"]:
            research_question_parts.append(f"Specifically consider these experimental conditions: {experimental_conditions}")
        if constraints and constraints not in ["None", "", "N/A", "Not mentioned"]:
            research_question_parts.append(f"Note these constraints: {constraints}")
        if target_output and target_output not in ["None", "", "N/A", "Not mentioned"]:
            research_question_parts.append(f"The target output is: {target_output}")
        
        research_question = " ".join(research_question_parts)
        
        # Run deep research (with timeout and limited iterations for efficiency)
        try:
            # Safely handle async call in sync context
            # Check if there's already an event loop running
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, we can't use asyncio.run()
                # In this case, skip deep research or use a different approach
                print("⚠️ Already in async context, skipping deep research (would require async node)")
                use_research = False
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                research_result = asyncio.run(
                    run_deep_research(
                        research_question,
                        max_researcher_iterations=3,  # Limit iterations for efficiency
                        return_full_state=True
                    )
                )
                
                if research_result:
                    research_context = research_result.get("final_report", "")
                    research_brief = research_result.get("research_brief", "")
                    
                    if research_context or research_brief:
                        use_research = True
                        print(f"✓ Deep research completed successfully")
                        print(f"  - Research brief length: {len(research_brief)} characters")
                        print(f"  - Research report length: {len(research_context)} characters")
                    else:
                        print("⚠️ Deep research returned empty results, falling back to LLM-only mode")
        except asyncio.TimeoutError:
            print("⚠️ Deep research timed out, falling back to LLM-only mode")
        except Exception as e:
            print(f"⚠️ Deep research execution failed: {type(e).__name__}: {e}")
            print("   Attempting to extract partial results from deep research state...")
            
            # Try to extract partial results even if final report generation failed
            # This can happen when documents are retrieved but LLM API calls fail
            try:
                # If we have a partial state, try to extract useful information
                if 'research_result' in locals() and research_result:
                    # Check if we have any useful information from partial execution
                    partial_context = research_result.get("final_report", "")
                    partial_brief = research_result.get("research_brief", "")
                    
                    # Also check for notes or findings that might have been collected
                    notes = research_result.get("notes", [])
                    findings = research_result.get("findings", "")
                    
                    # If we have any useful content, use it
                    if notes:
                        partial_context = "\n".join(notes) if isinstance(notes, list) else str(notes)
                    elif findings:
                        partial_context = findings
                    
                    if partial_context or partial_brief:
                        research_context = partial_context
                        research_brief = partial_brief
                        use_research = True
                        print(f"   ✓ Extracted partial research results:")
                        print(f"     - Research brief length: {len(research_brief)} characters")
                        print(f"     - Research context length: {len(research_context)} characters")
                        print("   Using partial results to enhance knowledge activation")
                    else:
                        print("   No partial results available, falling back to LLM-only mode")
                else:
                    print("   No research state available, falling back to LLM-only mode")
            except Exception as extract_error:
                print(f"   Failed to extract partial results: {extract_error}")
                print("   Falling back to LLM-only mode")
            
            import traceback
            traceback.print_exc()
    except ImportError as e:
        print(f"⚠️ Deep research module not available: {e}")
        print("   This may be due to missing dependencies (e.g., langchain.chat_models)")
        print("   Using LLM-only mode (this is normal and expected)")
    except Exception as e:
        print(f"⚠️ Deep research initialization failed: {type(e).__name__}: {e}")
        print("   Using LLM-only mode (this is normal and expected)")
        import traceback
        traceback.print_exc()
    
    # Extract constraint tags from state (for scenario constraint mapping table generation)
    constraint_tags = state.constraint_tags if state.constraint_tags else {}
    if constraint_tags:
        print(f"📋 Constraint tags available for scenario mapping: {sum(len(tags) for tags in constraint_tags.values())} tags")
        for tag_type, tags in constraint_tags.items():
            if tags:
                print(f"  - {tag_type}: {', '.join(tags)}")
    
    # Step 2: Generate prompt (with or without research context) and call LLM
    if use_research and (research_context or research_brief):
        print("📚 Using deep research context to enhance knowledge activation...")
        # Get question type from state (for enhanced prompt)
        question_type_str = str(state.question_type) if state.question_type else "Unknown"
        prompt = get_knowledge_activation_prompt_with_research(
            analysis_objects=analysis_objects,
            question_type=question_type_str,  # Required parameter
            research_context=research_context,
            research_brief=research_brief,
            experimental_conditions=experimental_conditions if experimental_conditions and experimental_conditions not in ["None", "", "N/A", "Not mentioned"] else None,
            constraints=constraints if constraints and constraints not in ["None", "", "N/A", "Not mentioned"] else None,
            target_output=target_output if target_output and target_output not in ["None", "", "N/A", "Not mentioned"] else None,
            constraint_tags=constraint_tags if constraint_tags else None
        )
    else:
        print("📝 Using standard LLM-only knowledge activation...")
        prompt = get_knowledge_activation_prompt(
            analysis_objects=analysis_objects,
            experimental_conditions=experimental_conditions if experimental_conditions and experimental_conditions not in ["None", "", "N/A", "Not mentioned"] else None,
            constraints=constraints if constraints and constraints not in ["None", "", "N/A", "Not mentioned"] else None,
            target_output=target_output if target_output and target_output not in ["None", "", "N/A", "Not mentioned"] else None,
            constraint_tags=constraint_tags if constraint_tags else None
        )
    
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable, cannot activate knowledge. Please check LLM configuration.")
    
    llm_response = _call_llm_for_parsing(llm, prompt)
    if not llm_response:
        raise RuntimeError("LLM call failed, cannot get knowledge response.")
    
    print(f"LLM knowledge response:\n{llm_response}\n")
    
    # Step 3: Parse LLM returned JSON using unified fix_json_format
    try:
        knowledge_data, repair_level = fix_json_format(llm_response)
        if repair_level != "none":
            print(f"⚠ JSON format repaired at {repair_level} level")
    except ValueError as e:
        raise ValueError(f"LLM output is not standard JSON: {str(e)}, raw response: {llm_response[:200]}")
    
    if not isinstance(knowledge_data, dict):
        raise ValueError(f"LLM returned knowledge data format error, should be dictionary type")
    
    # Step 4: Validate knowledge structure (Universal Conditional Knowledge Format)
    # Priority: Check for new universal format (Conditional Knowledge), fallback to legacy format
    for obj, knowledge in knowledge_data.items():
        if not isinstance(knowledge, dict):
            raise ValueError(f"Analysis object {obj} knowledge format error, should be dictionary type")
        
        # Verify no non-ASCII key names (should only be English)
        for key in knowledge.keys():
            if not all(ord(c) < 128 for c in key):
                raise ValueError(f"Analysis object {obj} knowledge contains non-ASCII key name (should be English only): {key}. Please ensure prompt requires English key names.")
        
        # Check if using new universal format (Conditional Knowledge) or legacy format
        has_conditional_knowledge = "Conditional Knowledge" in knowledge
        has_basic_info = "Basic Info" in knowledge
        
        if has_conditional_knowledge:
            # Validate new universal format
            print(f"  ✓ {obj}: Using universal Conditional Knowledge format")
            
            # Validate Basic Info
            if has_basic_info:
                basic_info = knowledge.get("Basic Info", {})
                if not isinstance(basic_info, dict):
                    raise ValueError(f"Analysis object {obj} Basic Info must be a dictionary")
                if "definition" not in basic_info:
                    print(f"    ⚠ {obj}: Basic Info missing 'definition' field")
            
            # Validate Conditional Knowledge (MANDATORY for universal format)
            conditional_knowledge = knowledge.get("Conditional Knowledge", {})
            if not isinstance(conditional_knowledge, dict):
                raise ValueError(f"Analysis object {obj} Conditional Knowledge must be a dictionary")
            
            # Check for at least one Kc-Kr pair
            kc_keys = [k for k in conditional_knowledge.keys() if k.startswith("Kc")]
            kr_keys = [k for k in conditional_knowledge.keys() if k.startswith("Kr")]
            
            if not kc_keys:
                raise ValueError(f"Analysis object {obj} Conditional Knowledge must contain at least one Kc (condition set)")
            if not kr_keys:
                raise ValueError(f"Analysis object {obj} Conditional Knowledge must contain at least one Kr (conclusion)")
            
            # Validate each Kc is a list
            for kc_key in kc_keys:
                kc_value = conditional_knowledge.get(kc_key)
                if not isinstance(kc_value, list):
                    raise ValueError(f"Analysis object {obj} Conditional Knowledge {kc_key} must be a list of constraint keywords")
            
            # Validate default_Kr exists
            if "default_Kr" not in conditional_knowledge:
                print(f"    ⚠ {obj}: Conditional Knowledge missing 'default_Kr' field, adding default")
                conditional_knowledge["default_Kr"] = "UNKNOWN: need more constraints"
            
            print(f"    ✓ {obj}: Conditional Knowledge validated ({len(kc_keys)} Kc-Kr pairs)")
        else:
            # Legacy format validation (for backward compatibility)
            print(f"  ⚠ {obj}: Using legacy format (missing Conditional Knowledge), will attempt conversion")
            required_dimensions = ["Core Definition", "Key Rules/Properties", "Association with Experimental Conditions", "Scenario Constraint Mapping Table"]
            missing_dims = [dim for dim in required_dimensions if dim not in knowledge]
            if missing_dims:
                raise ValueError(f"Analysis object {obj} knowledge missing required dimensions (legacy format): {missing_dims}")
            
            # Validate Scenario Constraint Mapping Table structure (legacy)
            mapping_table = knowledge.get("Scenario Constraint Mapping Table")
            if mapping_table is not None:
                if not isinstance(mapping_table, list):
                    raise ValueError(f"Analysis object {obj} Scenario Constraint Mapping Table must be a list, got {type(mapping_table)}")
                
                for i, entry in enumerate(mapping_table):
                    if not isinstance(entry, dict):
                        raise ValueError(f"Analysis object {obj} Scenario Constraint Mapping Table entry {i} must be a dictionary")
                    if "constraint_combination" not in entry:
                        raise ValueError(f"Analysis object {obj} Scenario Constraint Mapping Table entry {i} missing required field: constraint_combination")
                    if not isinstance(entry.get("constraint_combination"), list):
                        raise ValueError(f"Analysis object {obj} Scenario Constraint Mapping Table entry {i} constraint_combination must be a list")
    
    # Step 5: Update state (knowledge activation successful)
    state.domain_knowledge = knowledge_data
    state.knowledge_error = None
    
    print(f"✓ Domain knowledge activation successful, covering analysis objects: {list(knowledge_data.keys())}")
    for obj, knowledge in knowledge_data.items():
        # Check format type
        has_conditional = "Conditional Knowledge" in knowledge
        if has_conditional:
            conditional_knowledge = knowledge.get("Conditional Knowledge", {})
            kc_count = len([k for k in conditional_knowledge.keys() if k.startswith("Kc")])
            print(f"  - {obj}: Universal format - {kc_count} Kc-Kr pairs in Conditional Knowledge")
        else:
            mapping_table = knowledge.get("Scenario Constraint Mapping Table", [])
            mapping_table_size = len(mapping_table) if isinstance(mapping_table, list) else 0
            print(f"  - {obj}: Legacy format - {len(knowledge)} knowledge dimensions, {mapping_table_size} scenario constraint mappings")
    
    return state


# ---------------------- Generic Architecture: Match Degree Judgment Helper ----------------------
def _judge_match_degree(
    core_constraints: List[str],
    knowledge_Kc: List[str]
) -> Dict[str, Any]:
    """
    Generic architecture: Match degree judgment (C1 ⊇ Kc)
    
    Determines if core constraints (C1) contain/satisfy knowledge condition set (Kc).
    
    Args:
        core_constraints: List of core constraint identifiers (C1)
        knowledge_Kc: List of constraint identifiers from knowledge condition set (Kc)
    
    Returns:
        Dict containing:
        - match_type: "full_match", "partial_match", or "no_match"
        - match_score: Match quality score (0.0 to 1.0)
        - matched_constraints: List of constraints from Kc that are in C1
        - unmatched_constraints: List of constraints from Kc that are not in C1
        - extra_constraints: List of constraints in C1 that are not in Kc
    """
    if not knowledge_Kc:
        # If Kc is empty, consider it a match (no conditions required)
        return {
            "match_type": "full_match",
            "match_score": 1.0,
            "matched_constraints": [],
            "unmatched_constraints": [],
            "extra_constraints": core_constraints.copy()
        }
    
    if not core_constraints:
        # If C1 is empty but Kc is not, no match
        return {
            "match_type": "no_match",
            "match_score": 0.0,
            "matched_constraints": [],
            "unmatched_constraints": knowledge_Kc.copy(),
            "extra_constraints": []
        }
    
    # Convert to sets for easier comparison
    C1_set = set(core_constraints)
    Kc_set = set(knowledge_Kc)
    
    # Calculate match
    matched = Kc_set.intersection(C1_set)
    unmatched = Kc_set - C1_set
    extra = C1_set - Kc_set
    
    # Determine match type
    if len(unmatched) == 0:
        if len(extra) == 0:
            match_type = "full_match"  # C1 = Kc
            match_score = 1.0
        else:
            match_type = "partial_match"  # C1 ⊇ Kc with extra elements
            # Score based on how many Kc elements are matched
            match_score = len(matched) / len(Kc_set) if Kc_set else 0.0
    else:
        match_type = "no_match"  # C1 does not contain all of Kc
        # Score based on how many Kc elements are matched
        match_score = len(matched) / len(Kc_set) if Kc_set else 0.0
    
    return {
        "match_type": match_type,
        "match_score": match_score,
        "matched_constraints": list(matched),
        "unmatched_constraints": list(unmatched),
        "extra_constraints": list(extra)
    }


# ---------------------- Constraint-Knowledge Mapping Validation Helper ----------------------
def _match_constraint_tags_to_mapping_table(
    constraint_tags: Dict[str, List[str]],
    mapping_table: List[Dict[str, Any]],
    analysis_object: str
) -> Dict[str, Any]:
    """
    Match constraint tags from question parsing with Scenario Constraint Mapping Table from knowledge activation.
    
    This function performs one-to-one matching to ensure constraint-specific reasoning and avoid generalization errors.
    
    Args:
        constraint_tags: Constraint tags from question parsing (dict with keys: site_constraints, sample_operation_constraints, sample_size_constraints)
        mapping_table: Scenario Constraint Mapping Table from domain knowledge (list of dicts)
        analysis_object: Name of the analysis object (for error messages)
    
    Returns:
        Dict containing:
        - matched_entries: List of matching mapping table entries
        - match_score: Match quality score (0-1)
        - unmatched_constraints: List of constraint tags that don't match any entry
        - match_details: Detailed matching information
    """
    if not constraint_tags or not mapping_table:
        return {
            "matched_entries": [],
            "match_score": 0.0,
            "unmatched_constraints": [],
            "match_details": "No constraint tags or mapping table available"
        }
    
    # Flatten all constraint tags into a single list
    all_constraint_tags = []
    for tag_type, tags in constraint_tags.items():
        all_constraint_tags.extend(tags)
    
    if not all_constraint_tags:
        return {
            "matched_entries": [],
            "match_score": 0.0,
            "unmatched_constraints": [],
            "match_details": "No constraint tags available"
        }
    
    # Match each mapping table entry against constraint tags
    matched_entries = []
    match_scores = []
    match_details_list = []
    
    for entry in mapping_table:
        entry_constraints = entry.get("constraint_combination", [])
        if not isinstance(entry_constraints, list):
            continue
        
        # Calculate match score: how many constraint tags from entry are present in question's constraint tags
        matched_count = sum(1 for ec in entry_constraints if ec in all_constraint_tags)
        total_entry_constraints = len(entry_constraints)
        
        if total_entry_constraints > 0:
            match_score = matched_count / total_entry_constraints
        else:
            match_score = 0.0
        
        # Consider it a match if at least 50% of entry's constraints are present
        if match_score >= 0.5:
            matched_entries.append(entry)
            match_scores.append(match_score)
            match_details_list.append({
                "entry_constraints": entry_constraints,
                "matched_constraints": [ec for ec in entry_constraints if ec in all_constraint_tags],
                "match_score": match_score
            })
    
    # Calculate overall match score
    if match_scores:
        overall_match_score = sum(match_scores) / len(match_scores)
    else:
        overall_match_score = 0.0
    
    # Find unmatched constraints
    matched_constraint_set = set()
    for details in match_details_list:
        matched_constraint_set.update(details["matched_constraints"])
    unmatched_constraints = [tag for tag in all_constraint_tags if tag not in matched_constraint_set]
    
    return {
        "matched_entries": matched_entries,
        "match_score": overall_match_score,
        "unmatched_constraints": unmatched_constraints,
        "match_details": match_details_list
    }


# ---------------------- Node 3: Data Processing Node (Optimized) ----------------------
def data_processing_node(state: GeneralQAState) -> GeneralQAState:
    """
    Data processing node (optimized)
    
    Core tasks:
    1. Analyze experimental treatment impacts
    2. Verify implicit assumptions
    3. Identify error sources
    
    Analysis dimensions: Treatment → Variable change → Result impact causal chain, distinguish technical errors vs biological signals.
    
    Args:
        state: General QA subgraph state
    
    Returns:
        Updated state (containing experiment analysis with 4 dimensions)
    """
    # Step 1: Check if previous nodes' outputs are valid (dependency chain validation)
    # Validate first node: check if experimental conditions exist
    if state.parse_error:
        raise ValueError(f"First node parsing failed, cannot analyze: {state.parse_error}")
    
    if not state.key_info or "Experimental Conditions" not in state.key_info:
        raise ValueError("First node did not extract experimental conditions, cannot analyze")
    
    experiment_condition = state.key_info["Experimental Conditions"].strip()
    # Handle empty experimental conditions: generate default analysis instead of raising error
    if experiment_condition in ["None", "", "N/A", "Not mentioned"]:
        print("⚠ Experimental conditions are empty, generating default analysis result")
        # Generate default analysis structure for questions without experimental conditions
        state.experiment_analysis = {
            "Operation Breakdown": ["No experimental conditions specified"],
            "Knowledge Association Basis": ["No experimental conditions to associate with domain knowledge"],
            "Impact Direction Judgment": ["No experimental impact to analyze"],
            "Core Hypothesis Validation": "No experimental conditions specified in the question. Analysis focuses on theoretical/conceptual aspects."
        }
        state.analysis_error = None
        print("✓ Default experiment analysis generated (no experimental conditions)")
        return state
    
    # Validate second node: check if domain knowledge exists
    if state.knowledge_error:
        raise ValueError(f"Second node knowledge activation failed, cannot analyze: {state.knowledge_error}")
    
    if not state.domain_knowledge:
        raise ValueError("Second node did not activate domain knowledge, cannot analyze")
    
    # Extract analysis object (for prompt positioning)
    analysis_object = state.key_info.get("Analysis Object", "").strip()
    if not analysis_object or analysis_object in ["None", "", "N/A"]:
        analysis_object = "Unspecified analysis object"
    
    # Step 1.5: Universal Matching Engine (NEW - 通用匹配引擎)
    print("\n[Step 1.5] Running universal matching engine...")
    constraint_hierarchy = state.constraint_hierarchy if state.constraint_hierarchy else {
        "C1_core_constraint": [],
        "C2_secondary_constraint": []
    }
    
    # 提取domain_knowledge中的Conditional Knowledge，执行通用匹配
    universal_matching_results = {}
    if state.domain_knowledge:
        # 为每个知识点提取Conditional Knowledge并执行匹配
        for knowledge_name, knowledge_data in state.domain_knowledge.items():
            conditional_knowledge = knowledge_data.get("Conditional Knowledge", {})
            if conditional_knowledge:
                print(f"  Matching constraints for '{knowledge_name}'...")
                match_result = universal_matching_engine(
                    constraint_hierarchy=constraint_hierarchy,
                    conditional_knowledge=conditional_knowledge,
                    analysis_object_name=knowledge_name
                )
                universal_matching_results[knowledge_name] = match_result
                
                # 更新constraint_hierarchy（使用第一个知识点的C1锁定结果，或合并所有）
                if match_result["constraint_hierarchy_updated"]["C1_core_constraint"]:
                    # 合并所有知识点的C1约束
                    existing_C1 = set(constraint_hierarchy.get("C1_core_constraint", []))
                    new_C1 = set(match_result["constraint_hierarchy_updated"]["C1_core_constraint"])
                    constraint_hierarchy["C1_core_constraint"] = list(existing_C1.union(new_C1))
                    print(f"    ✓ C1 locked: {constraint_hierarchy['C1_core_constraint']}")
                    print(f"    ✓ Match type: {match_result['matching_info']['match_type']}, Score: {match_result['matching_info']['match_score']:.2f}")
                else:
                    print(f"    ⚠ No C1 locked (no matching Kc)")
            else:
                print(f"  ⚠ No Conditional Knowledge found for '{knowledge_name}'")
    
    # 更新state中的constraint_hierarchy和universal_matching_result
    state.constraint_hierarchy = constraint_hierarchy
    state.universal_matching_result = universal_matching_results
    
    # Step 1.6: Constraint-Knowledge Mapping Validation (Legacy - 保留兼容性)
    print("\n[Step 1.6] Performing constraint-knowledge mapping validation (legacy)...")
    constraint_tags = state.constraint_tags if state.constraint_tags else {}
    matched_mappings = {}
    constraint_validation_results = {}
    
    # Handle multiple analysis objects (split by comma if needed)
    if "," in analysis_object:
        analysis_objects_list = [obj.strip() for obj in analysis_object.split(",")]
    else:
        analysis_objects_list = [analysis_object]
    
    for obj in analysis_objects_list:
        # Find the corresponding knowledge entry
        obj_knowledge = None
        for key, knowledge in state.domain_knowledge.items():
            # Try to match analysis object name (fuzzy matching)
            if obj.lower() in key.lower() or key.lower() in obj.lower():
                obj_knowledge = knowledge
                break
        
        # If exact match not found, try first entry
        if obj_knowledge is None and state.domain_knowledge:
            obj_knowledge = list(state.domain_knowledge.values())[0]
        
        if obj_knowledge:
            mapping_table = obj_knowledge.get("Scenario Constraint Mapping Table", [])
            if isinstance(mapping_table, list) and mapping_table:
                match_result = _match_constraint_tags_to_mapping_table(
                    constraint_tags=constraint_tags,
                    mapping_table=mapping_table,
                    analysis_object=obj
                )
                matched_mappings[obj] = match_result
                constraint_validation_results[obj] = match_result
                
                if match_result["matched_entries"]:
                    print(f"✓ Found {len(match_result['matched_entries'])} matching constraint scenarios for '{obj}'")
                    print(f"  Match score: {match_result['match_score']:.2f}")
                    for i, entry in enumerate(match_result["matched_entries"][:3], 1):  # Show top 3
                        constraints = entry.get("constraint_combination", [])
                        print(f"  Match {i}: {', '.join(constraints)}")
                else:
                    print(f"⚠ No matching constraint scenarios found for '{obj}'")
                    if match_result["unmatched_constraints"]:
                        print(f"  Unmatched constraints: {', '.join(match_result['unmatched_constraints'][:5])}")
            else:
                print(f"⚠ No Scenario Constraint Mapping Table available for '{obj}'")
                constraint_validation_results[obj] = {
                    "matched_entries": [],
                    "match_score": 0.0,
                    "unmatched_constraints": list(constraint_tags.get("site_constraints", [])) + 
                                            list(constraint_tags.get("sample_operation_constraints", [])) + 
                                            list(constraint_tags.get("sample_size_constraints", [])),
                    "match_details": "No mapping table available"
                }
        else:
            print(f"⚠ No domain knowledge found for '{obj}'")
            constraint_validation_results[obj] = {
                "matched_entries": [],
                "match_score": 0.0,
                "unmatched_constraints": [],
                "match_details": "No domain knowledge available"
            }
    
    # Step 2: Generate prompt and call LLM (must use LLM, no fallback)
    # Pass matched mappings to prompt for constraint-specific analysis
    print(f"\n[Step 2] Associating experimental conditions with domain knowledge, analyzing impact on '{analysis_object}'...")
    prompt = get_experiment_analysis_prompt(
        experiment_condition=experiment_condition,
        domain_knowledge=state.domain_knowledge,
        analysis_object=analysis_object,
        constraint_tags=constraint_tags,
        matched_mappings=matched_mappings
    )
    
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable, cannot perform experiment analysis. Please check LLM configuration.")
    
    llm_response = _call_llm_for_parsing(llm, prompt)
    if not llm_response:
        raise RuntimeError("LLM call failed, cannot get experiment analysis response.")
    
    print(f"LLM experiment analysis response:\n{llm_response}\n")
    
    # Step 3: Parse LLM result using unified fix_json_format
    required_dims = ["Constraint-Knowledge Match Summary", "Operation Breakdown", "Knowledge Association Basis", "Impact Direction Judgment", "Core Hypothesis Validation"]
    try:
        analysis_data, repair_level = fix_json_format(llm_response, required_keys=required_dims)
        if repair_level != "none":
            print(f"⚠ JSON format repaired at {repair_level} level")
    except ValueError as e:
        raise ValueError(f"LLM output is not standard JSON: {str(e)}, raw response: {llm_response[:200]}")
    
    if not isinstance(analysis_data, dict):
        raise ValueError(f"LLM returned experiment analysis data format error, should be dictionary type")
    
    # Validate required dimensions (must use English key names)
    required_dimensions = ["Constraint-Knowledge Match Summary", "Operation Breakdown", "Knowledge Association Basis", "Impact Direction Judgment", "Core Hypothesis Validation"]
    
    # Verify no non-ASCII key names (should only be English)
    for key in analysis_data.keys():
        if not all(ord(c) < 128 for c in key):
            raise ValueError(f"LLM returned non-ASCII key name (should be English only): {key}. Please ensure prompt requires English key names.")
    
    # Check required dimensions
    for dim in required_dimensions:
        if dim not in analysis_data:
            raise ValueError(f"Experiment analysis missing required dimension: {dim}, raw response: {llm_response[:200]}")
    
    # Additional validation: operation breakdown and knowledge association count match (avoid misalignment)
    operations = analysis_data["Operation Breakdown"]
    knowledge_links = analysis_data["Knowledge Association Basis"]
    if not isinstance(operations, list) or not isinstance(knowledge_links, list):
        raise ValueError(f"Operation Breakdown and Knowledge Association Basis must be lists")
    if len(operations) != len(knowledge_links):
        raise ValueError(f"Operation Breakdown ({len(operations)} items) and Knowledge Association Basis ({len(knowledge_links)} items) count mismatch")
    
    # Step 4: Add constraint validation results to analysis data
    analysis_data["Constraint Validation Results"] = constraint_validation_results
    
    # Step 5: Update state (analysis successful)
    state.experiment_analysis = analysis_data
    state.analysis_error = None
    
    print(f"✓ Experiment condition-knowledge association analysis successful, obtained impact direction judgment")
    constraint_match_summary = analysis_data.get("Constraint-Knowledge Match Summary", "N/A")
    print(f"  Constraint-Knowledge Match: {constraint_match_summary[:100]}..." if len(str(constraint_match_summary)) > 100 else f"  Constraint-Knowledge Match: {constraint_match_summary}")
    print(f"  Operation breakdown: {len(operations)} operations")
    print(f"  Impact judgments: {len(analysis_data.get('Impact Direction Judgment', []))} items")
    
    return state


# ---------------------- Node 4: Reasoning Engine Node (Optimized) ----------------------
def reasoning_engine_node(state: GeneralQAState) -> GeneralQAState:
    """
    Reasoning engine node (optimized)
    
    Core tasks:
    1. Select appropriate derivation strategy (counterexample verification / stepwise calculation / classification enumeration / variable control)
    2. Build structured reasoning chain
    
    Reasoning types: Causal reasoning, conditional reasoning, enumeration reasoning, calculation reasoning.
    
    Args:
        state: General QA subgraph state
    
    Returns:
        Updated state (containing logical derivation with type-specific dimensions)
    """
    # Step 0: Check if this is a re-derivation due to derivation_warning
    if state.derivation_warning:
        print(f"\n⚠ RE-DERIVATION TRIGGERED due to derivation_warning: {state.derivation_warning}")
        print(f"  Previous preliminary conclusion failed substantive validation, re-deriving with attention to:")
        if state.derivation_warning == "constraint_mapping_mismatch":
            print(f"    - Constraint-knowledge mapping consistency")
        elif state.derivation_warning == "rebuttal_evidence_contradiction":
            print(f"    - Rebuttal evidence consistency")
        elif state.derivation_warning == "both_validations_failed":
            print(f"    - Both constraint mapping and rebuttal evidence consistency")
        print(f"  Please ensure the new derivation addresses the validation failure.\n")
    
    # Step 0: Check if this is a re-derivation due to derivation_warning
    if state.derivation_warning:
        print(f"\n⚠ RE-DERIVATION TRIGGERED due to derivation_warning: {state.derivation_warning}")
        print(f"  Previous preliminary conclusion failed substantive validation, re-deriving with attention to:")
        if state.derivation_warning == "constraint_mapping_mismatch":
            print(f"    - Constraint-knowledge mapping consistency")
        elif state.derivation_warning == "rebuttal_evidence_contradiction":
            print(f"    - Rebuttal evidence consistency")
        elif state.derivation_warning == "both_validations_failed":
            print(f"    - Both constraint mapping and rebuttal evidence consistency")
        print(f"  Please ensure the new derivation addresses the validation failure.\n")
    
    # Step 1: Validate previous nodes' output validity (dependency chain check)
    # Check Node 1: Is question type clear?
    if state.parse_error or not state.question_type:
        raise ValueError(f"Node 1 output invalid (no question type), cannot derive: {state.parse_error}")
    
    # Check Node 2: Is there domain knowledge?
    if state.knowledge_error or not state.domain_knowledge:
        raise ValueError(f"Node 2 output invalid (no domain knowledge), cannot derive: {state.knowledge_error}")
    
    # Check Node 3: Is there experiment analysis result? (Enumeration type questions may not have experiment analysis)
    question_type_str = str(state.question_type)
    is_enumeration = "Enumeration" in question_type_str
    if not is_enumeration and (state.analysis_error or not state.experiment_analysis):
        raise ValueError(f"Node 3 output invalid (no experiment analysis), cannot derive: {state.analysis_error}")
    
    # Extract key information
    analysis_object = state.key_info.get("Analysis Object", "").strip()
    if not analysis_object or analysis_object in ["None", "", "N/A"]:
        analysis_object = "Unspecified analysis object"
    
    question_text = state.user_input  # Full question text
    question_core = _extract_question_core(question_text)  # Core question (e.g., "Is it higher than control?")
    if not question_core:
        raise ValueError("Cannot extract core question from question text, derivation terminated")
    
    # Extract constraint tags for rebuttal evidence validation
    constraint_tags = state.constraint_tags if state.constraint_tags else {}
    if constraint_tags:
        print(f"📋 Using constraint tags for rebuttal evidence validation: {sum(len(tags) for tags in constraint_tags.values())} tags")
    
    # Extract derivation_warning for re-derivation guidance
    derivation_warning = state.derivation_warning
    
    # Step 2: Generate type-specific prompt
    print(f"Deriving for '{question_type_str}' question, core: {question_core[:50]}...")
    prompt = get_logical_derivation_prompt(
        question_type=question_type_str,
        question_core=question_core,
        domain_knowledge=state.domain_knowledge,
        experiment_analysis=state.experiment_analysis if not is_enumeration else {},
        analysis_object=analysis_object,
        constraint_tags=constraint_tags if constraint_tags else None,
        derivation_warning=derivation_warning
    )
    
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable, cannot perform logical derivation. Please check LLM configuration.")
    
    llm_response = _call_llm_for_parsing(llm, prompt)
    if not llm_response:
        raise RuntimeError("LLM call failed, cannot get logical derivation response.")
    
    print(f"LLM logical derivation response:\n{llm_response[:200]}...")  # Print first 200 chars to avoid too long
    
    # Step 3: Parse LLM result using unified fix_json_format
    required_general_dims = ["Derivation Strategy", "Core Evidence Chain", "Preliminary Conclusion", "Rebuttal Evidence"]
    try:
        derivation_data, repair_level = fix_json_format(llm_response, required_keys=required_general_dims)
        if repair_level != "none":
            print(f"⚠ JSON format repaired at {repair_level} level")
    except ValueError as e:
        raise ValueError(f"LLM output is not standard JSON: {str(e)}, raw response: {llm_response[:200]}")
    
    if not isinstance(derivation_data, dict):
        raise ValueError(f"LLM returned logical derivation data format error, should be dictionary type")
    
    # Verify no non-ASCII key names (should only be English)
    for key in derivation_data.keys():
        if not all(ord(c) < 128 for c in key):
            raise ValueError(f"LLM returned non-ASCII key name (should be English only): {key}. Please ensure prompt requires English key names.")
    
    # Validate general dimensions (required for all question types)
    required_general_dims = ["Derivation Strategy", "Core Evidence Chain", "Preliminary Conclusion", "Rebuttal Evidence"]
    for dim in required_general_dims:
        if dim not in derivation_data:
            raise ValueError(f"Logical derivation missing general dimension: {dim}, raw response: {llm_response[:200]}")
    
    # Validate type-specific dimensions
    type_specific_dims = {
        "Judgment": ["Option Matching Priority"],
        "Calculation": ["Formula Substitution Process", "Numerical Calculation Result"],
        "Analysis": ["Causal Chain Analysis"],
        "Enumeration": ["Enumeration Results"]
    }
    
    # Map question type to validation key
    validation_key = None
    if "Judgment" in question_type_str:
        validation_key = "Judgment"
    elif "Calculation" in question_type_str:
        validation_key = "Calculation"
    elif "Analysis" in question_type_str:
        validation_key = "Analysis"
    elif "Enumeration" in question_type_str:
        validation_key = "Enumeration"
    
    if validation_key and validation_key in type_specific_dims:
        for dim in type_specific_dims[validation_key]:
            if dim not in derivation_data:
                raise ValueError(f"{question_type_str} question missing type-specific dimension: {dim}")
    
    # Step 3.5: Validate Rebuttal Evidence structure
    rebuttal_evidence = derivation_data.get("Rebuttal Evidence")
    if rebuttal_evidence:
        if isinstance(rebuttal_evidence, dict):
            # Check if rebuttal evidence has expected structure
            expected_fields = ["counter_hypothesis", "counter_evidence_considered", "rebuttal_analysis", "rebuttal_conclusion"]
            missing_fields = [field for field in expected_fields if field not in rebuttal_evidence]
            if missing_fields:
                print(f"⚠ Rebuttal Evidence missing some fields: {missing_fields}, but continuing...")
            else:
                print(f"✓ Rebuttal Evidence validated: counter-hypothesis considered and analyzed")
        else:
            print(f"⚠ Rebuttal Evidence is not a dictionary, but continuing...")
    else:
        print(f"⚠ Rebuttal Evidence not found in derivation data, but continuing...")
    
    # Step 3.6: Extract universal matching results and generate universal reasoning (NEW)
    print("\n[Step 3.6] Extracting universal matching results for reasoning...")
    universal_matching_result = state.universal_matching_result if state.universal_matching_result else {}
    C1_locked = {}
    final_preliminary_conclusion_universal = {}
    
    if universal_matching_result:
        for knowledge_name, match_result in universal_matching_result.items():
            # 提取C1
            C1 = match_result.get("constraint_hierarchy_updated", {}).get("C1_core_constraint", [])
            C1_locked[knowledge_name] = C1
            
            # 提取Kr结论
            Kr = match_result.get("preliminary_conclusion_universal", "UNKNOWN")
            final_preliminary_conclusion_universal[knowledge_name] = Kr
            
            print(f"  {knowledge_name}: C1={C1}, Kr={Kr[:50]}...")
    
    # 构建通用推理步骤
    universal_reasoning_steps = [
        "Step1: 获取process_data的universal_matching_result",
        "Step2: 提取匹配成功的C1_core_constraint和对应的Kr结论",
        "Step3: 组合所有分析对象的Kr结论，形成最终初步结论",
        "Step4: 执行通用反驳证据校验"
    ]
    
    # 将通用推理结果添加到derivation_data
    derivation_data["universal_reasoning_steps"] = universal_reasoning_steps
    derivation_data["C1_locked"] = C1_locked
    derivation_data["final_preliminary_conclusion_universal"] = final_preliminary_conclusion_universal
    
    # 通用反驳证据校验（领域无关）
    rebuttal_evidence_universal = {
        "rebuttal_rule": "若C1_locked非空且匹配成功，则反驳证据为「无矛盾」；若C1为空，反驳证据为「核心约束缺失」",
        "rebuttal_result": "NO_CONTRADICTION" if C1_locked and any(C1_locked.values()) else "CORE_CONSTRAINT_MISSING",
        "rebuttal_reason": "C1_locked非空，Kc是C2的子集，匹配成功" if C1_locked and any(C1_locked.values()) else "C1为空，无匹配Kc"
    }
    derivation_data["rebuttal_evidence_universal"] = rebuttal_evidence_universal
    
    # Step 4: Update state (derivation successful)
    state.logical_derivation = derivation_data
    state.derivation_error = None
    # Note: Keep derivation_warning until validation passes (don't clear it here)
    # It will be cleared in conclusion_validation_node if validations pass
    
    print(f"✓ Logical derivation successful, preliminary conclusion: {derivation_data.get('Preliminary Conclusion', 'N/A')[:50]}...")
    if rebuttal_evidence and isinstance(rebuttal_evidence, dict):
        rebuttal_conclusion = rebuttal_evidence.get("rebuttal_conclusion", "N/A")
        print(f"  Rebuttal Evidence: {rebuttal_conclusion[:100]}..." if len(str(rebuttal_conclusion)) > 100 else f"  Rebuttal Evidence: {rebuttal_conclusion}")
    print(f"  Universal Reasoning: {len(universal_reasoning_steps)} steps, {len(C1_locked)} knowledge points matched")
    
    if state.derivation_warning:
        print(f"  ⚠ Re-derivation completed, will re-validate against: {state.derivation_warning}")
    
    return state


# ---------------------- Substantive Validation Helper Functions ----------------------
def _validate_constraint_mapping_consistency(
    preliminary_conclusion: str,
    constraint_validation_results: Dict[str, Dict[str, Any]],
    analysis_object: str
) -> Dict[str, Any]:
    """
    Validate if preliminary conclusion is consistent with constraint-knowledge mapping table results.
    
    This is a substantive validation (not just formal check) that ensures the conclusion
    aligns with the matched constraint scenarios from the mapping table.
    
    Args:
        preliminary_conclusion: Preliminary conclusion from reasoning engine
        constraint_validation_results: Constraint validation results from data processing node
        analysis_object: Name of the analysis object
    
    Returns:
        Dict containing:
        - is_consistent: bool
        - validation_details: str describing the validation result
        - mismatch_reasons: List of reasons if inconsistent
    """
    if not constraint_validation_results:
        return {
            "is_consistent": True,
            "validation_details": "No constraint validation results available, skipping constraint mapping validation",
            "mismatch_reasons": []
        }
    
    # Find validation result for the analysis object
    obj_validation = None
    for obj, validation in constraint_validation_results.items():
        if analysis_object.lower() in obj.lower() or obj.lower() in analysis_object.lower():
            obj_validation = validation
            break
    
    # If no exact match, use first available validation
    if obj_validation is None and constraint_validation_results:
        obj_validation = list(constraint_validation_results.values())[0]
    
    if not obj_validation:
        return {
            "is_consistent": True,
            "validation_details": "No constraint validation result found for analysis object, skipping validation",
            "mismatch_reasons": []
        }
    
    matched_entries = obj_validation.get("matched_entries", [])
    if not matched_entries:
        return {
            "is_consistent": True,
            "validation_details": "No matched constraint scenarios found, cannot validate consistency",
            "mismatch_reasons": []
        }
    
    # Extract key claims from preliminary conclusion and matched scenarios
    prelim_lower = preliminary_conclusion.lower()
    mismatch_reasons = []
    
    # Check each matched entry's parameter values against preliminary conclusion
    for entry in matched_entries:
        constraint_combo = entry.get("constraint_combination", [])
        
        # Get all dynamic parameter fields (excluding constraint_combination)
        param_fields = {k: str(v).lower() for k, v in entry.items() if k != "constraint_combination"}
        all_param_values = " ".join(param_fields.values())
        
        # Check for contradictions between parameters and preliminary conclusion
        # Check for bias-related contradictions
        if "unbiased" in all_param_values and ("biased" in prelim_lower or "bias" in prelim_lower):
            # Check if conclusion explicitly contradicts
            if any(word in prelim_lower for word in ["biased", "has bias", "is biased", "underestimated", "overestimated"]):
                # Find which field contains the contradiction
                contradicting_field = next((k for k, v in param_fields.items() if "unbiased" in v), "unknown field")
                mismatch_reasons.append(
                    f"Matched scenario {constraint_combo} field '{contradicting_field}' indicates unbiased but preliminary conclusion suggests bias"
                )
        
        if any("biased" in v or "underestimated" in v or "overestimated" in v for v in param_fields.values()):
            if "unbiased" in prelim_lower or "no bias" in prelim_lower:
                # Find which field contains the contradiction
                contradicting_field = next((k for k, v in param_fields.items() if "biased" in v or "underestimated" in v or "overestimated" in v), "unknown field")
                mismatch_reasons.append(
                    f"Matched scenario {constraint_combo} field '{contradicting_field}' indicates bias but preliminary conclusion suggests unbiased"
                )
        
        # Check for direction mismatches (e.g., "promotes" vs "inhibits", "increases" vs "decreases")
        direction_keywords = {
            "positive": ["promotes", "increases", "enhances", "higher", "greater"],
            "negative": ["inhibits", "decreases", "reduces", "lower", "less"]
        }
        
        for direction, keywords in direction_keywords.items():
            param_has_direction = any(kw in all_param_values for kw in keywords)
            prelim_has_opposite = any(kw in prelim_lower for kw in direction_keywords["negative" if direction == "positive" else "positive"])
            
            if param_has_direction and prelim_has_opposite:
                mismatch_reasons.append(
                    f"Matched scenario {constraint_combo} indicates {direction} direction but preliminary conclusion suggests opposite"
                )
    
    is_consistent = len(mismatch_reasons) == 0
    
    if is_consistent:
        validation_details = f"Preliminary conclusion is consistent with matched constraint scenarios ({len(matched_entries)} scenarios checked)"
    else:
        validation_details = f"Preliminary conclusion has {len(mismatch_reasons)} inconsistencies with matched constraint scenarios"
    
    return {
        "is_consistent": is_consistent,
        "validation_details": validation_details,
        "mismatch_reasons": mismatch_reasons
    }


def _validate_rebuttal_evidence_consistency(
    preliminary_conclusion: str,
    rebuttal_evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate if preliminary conclusion is consistent with rebuttal evidence from reasoning engine.
    
    This is a substantive validation that ensures the conclusion doesn't contradict
    the rebuttal analysis that was performed during reasoning.
    
    Args:
        preliminary_conclusion: Preliminary conclusion from reasoning engine
        rebuttal_evidence: Rebuttal evidence dict from logical derivation
    
    Returns:
        Dict containing:
        - is_consistent: bool
        - validation_details: str describing the validation result
        - contradiction_reasons: List of reasons if inconsistent
    """
    if not rebuttal_evidence or not isinstance(rebuttal_evidence, dict):
        return {
            "is_consistent": True,
            "validation_details": "No rebuttal evidence available, skipping rebuttal validation",
            "contradiction_reasons": []
        }
    
    rebuttal_conclusion = rebuttal_evidence.get("rebuttal_conclusion", "").lower()
    rebuttal_analysis = rebuttal_evidence.get("rebuttal_analysis", [])
    counter_hypothesis = rebuttal_evidence.get("counter_hypothesis", "").lower()
    
    if not rebuttal_conclusion and not rebuttal_analysis:
        return {
            "is_consistent": True,
            "validation_details": "Rebuttal evidence incomplete, skipping validation",
            "contradiction_reasons": []
        }
    
    prelim_lower = preliminary_conclusion.lower()
    contradiction_reasons = []
    
    # Check if rebuttal conclusion contradicts preliminary conclusion
    # Rebuttal conclusion should strengthen the preliminary conclusion, not contradict it
    if rebuttal_conclusion:
        # If rebuttal says "strengthens" or "validates", that's good
        if "strengthens" in rebuttal_conclusion or "validates" in rebuttal_conclusion or "consistent" in rebuttal_conclusion:
            # This is good, no contradiction
            pass
        elif "contradicts" in rebuttal_conclusion or "inconsistent" in rebuttal_conclusion:
            contradiction_reasons.append(
                "Rebuttal conclusion explicitly states contradiction with preliminary conclusion"
            )
    
    # Check rebuttal analysis for rejected counter-evidence
    if isinstance(rebuttal_analysis, list):
        for analysis_item in rebuttal_analysis:
            if isinstance(analysis_item, str):
                analysis_lower = analysis_item.lower()
                # If analysis says "REJECTED" but the reason suggests the counter-evidence is actually valid
                if "rejected" in analysis_lower:
                    # Check if rejection reason is weak or if it suggests the counter-evidence might be valid
                    if "but" in analysis_lower or "however" in analysis_lower or "might" in analysis_lower:
                        # Weak rejection, might indicate potential contradiction
                        if any(word in prelim_lower for word in ["unbiased", "no bias"]) and "biased" in counter_hypothesis:
                            contradiction_reasons.append(
                                f"Weak rejection of counter-evidence: {analysis_item[:100]}"
                            )
    
    # Check if counter hypothesis aligns with preliminary conclusion (they should be opposites)
    if counter_hypothesis:
        # Extract key claims from both
        prelim_has_positive = any(word in prelim_lower for word in ["yes", "higher", "promotes", "increases", "biased"])
        counter_has_negative = any(word in counter_hypothesis for word in ["no", "lower", "inhibits", "decreases", "unbiased"])
        
        # They should be opposites - if both are positive or both negative, there's a problem
        if prelim_has_positive and not counter_has_negative:
            contradiction_reasons.append(
                "Counter hypothesis does not appear to be opposite of preliminary conclusion"
            )
    
    is_consistent = len(contradiction_reasons) == 0
    
    if is_consistent:
        validation_details = "Preliminary conclusion is consistent with rebuttal evidence analysis"
    else:
        validation_details = f"Preliminary conclusion has {len(contradiction_reasons)} contradictions with rebuttal evidence"
    
    return {
        "is_consistent": is_consistent,
        "validation_details": validation_details,
        "contradiction_reasons": contradiction_reasons
    }


def _perform_universal_verification(state: GeneralQAState) -> Dict[str, Any]:
    """
    执行三层通用实质验证规则（领域无关）
    
    三层规则：
    1. Rule1: C1_core_constraint非空（核心约束已锁定）
    2. Rule2: Kc与C1的匹配度为100%（无缺失/冲突）
    3. Rule3: 初步结论与匹配的Kr完全一致（无结论篡改）
    
    Args:
        state: General QA subgraph state
    
    Returns:
        通用验证结果字典
    """
    verification_rules = [
        "Rule1: C1_core_constraint非空（核心约束已锁定）",
        "Rule2: Kc与C1的匹配度为100%（无缺失/冲突）",
        "Rule3: 初步结论与匹配的Kr完全一致（无结论篡改）"
    ]
    
    # 提取必要信息
    constraint_hierarchy = state.constraint_hierarchy if state.constraint_hierarchy else {}
    C1 = constraint_hierarchy.get("C1_core_constraint", [])
    universal_matching_result = state.universal_matching_result if state.universal_matching_result else {}
    logical_derivation = state.logical_derivation if state.logical_derivation else {}
    preliminary_conclusion = logical_derivation.get("Preliminary Conclusion", "")
    
    verification_details = {}
    failed_reasons = []
    
    # Rule1: C1非空检查
    if C1:
        verification_details["Rule1_check"] = "PASSED"
    else:
        verification_details["Rule1_check"] = "FAILED"
        failed_reasons.append("C1_core_constraint为空，核心约束未锁定")
    
    # Rule2: Kc与C1匹配度检查
    matching_accuracy = 0.0
    if universal_matching_result:
        all_match_scores = []
        for knowledge_name, match_result in universal_matching_result.items():
            match_info = match_result.get("matching_info", {})
            match_score = match_info.get("match_score", 0.0)
            match_type = match_info.get("match_type", "no_match")
            all_match_scores.append(match_score)
            
            if match_type == "no_match":
                failed_reasons.append(f"{knowledge_name}: 无匹配的Kc")
            elif match_score < 1.0:
                failed_reasons.append(f"{knowledge_name}: 匹配度不足100% ({match_score:.2%})")
        
        if all_match_scores:
            matching_accuracy = sum(all_match_scores) / len(all_match_scores) * 100.0
    else:
        failed_reasons.append("无universal_matching_result，无法验证匹配度")
    
    if matching_accuracy >= 100.0:
        verification_details["Rule2_check"] = "PASSED"
    else:
        verification_details["Rule2_check"] = "FAILED"
        if not failed_reasons or not any("匹配度" in r for r in failed_reasons):
            failed_reasons.append(f"Kc与C1匹配度不足100% ({matching_accuracy:.1f}%)")
    
    verification_details["matching_accuracy"] = matching_accuracy
    
    # Rule3: 初步结论与Kr一致性检查
    if universal_matching_result and preliminary_conclusion:
        # 检查初步结论是否与匹配的Kr一致
        all_Kr = []
        for knowledge_name, match_result in universal_matching_result.items():
            Kr = match_result.get("preliminary_conclusion_universal", "")
            if Kr and Kr != "UNKNOWN":
                all_Kr.append(Kr)
        
        # 简单检查：初步结论中应包含Kr的关键信息
        # 这是一个简化的检查，实际可以更复杂
        prelim_lower = preliminary_conclusion.lower()
        kr_consistent = False
        for Kr in all_Kr:
            # 提取Kr中的关键判断词（如UNBIASED, BIASED_DOWN等）
            if "unbiased" in Kr.lower() and "unbiased" in prelim_lower:
                kr_consistent = True
                break
            elif "biased" in Kr.lower() and "biased" in prelim_lower:
                kr_consistent = True
                break
        
        if kr_consistent or not all_Kr:
            verification_details["Rule3_check"] = "PASSED"
        else:
            verification_details["Rule3_check"] = "FAILED"
            failed_reasons.append("初步结论与匹配的Kr不一致")
    else:
        verification_details["Rule3_check"] = "PASSED"  # 如果没有匹配结果，跳过此检查
        if not universal_matching_result:
            failed_reasons.append("无universal_matching_result，无法验证结论一致性")
    
    # 综合判断
    all_passed = (
        verification_details.get("Rule1_check") == "PASSED" and
        verification_details.get("Rule2_check") == "PASSED" and
        verification_details.get("Rule3_check") == "PASSED"
    )
    
    result = {
        "universal_verification_result": "PASSED" if all_passed else "FAILED",
        "verification_rules": verification_rules,
        "verification_details": verification_details
    }
    
    if not all_passed:
        result["verification_failed_reason"] = "; ".join(failed_reasons)
    
    return result


# ---------------------- Node 5: Conclusion Validation Node (Optimized) ----------------------
def conclusion_validation_node(state: GeneralQAState) -> GeneralQAState:
    """
    Conclusion validation node (optimized)
    
    Core tasks:
    1. Internal logic consistency check
    2. External biological/technical/statistical reasonableness validation
    3. Precise option matching
    
    Validation methods: Reverse derivation verification, error exclusion method, domain knowledge cross-validation.
    
    Args:
        state: General QA subgraph state
    
    Returns:
        Updated state (containing final result with 5 required dimensions)
    """
    # Step 1: Full validation of previous four nodes' validity (any node error terminates, avoid error accumulation)
    node_errors = []
    if state.parse_error:
        node_errors.append(f"Node 1: {state.parse_error}")
    if state.knowledge_error:
        node_errors.append(f"Node 2: {state.knowledge_error}")
    if state.analysis_error:
        node_errors.append(f"Node 3: {state.analysis_error}")
    if state.derivation_error:
        node_errors.append(f"Node 4: {state.derivation_error}")
    
    if node_errors:
        raise ValueError(f"Previous nodes have errors, cannot complete final validation: {'; '.join(node_errors)}")
    
    # Extract previous core information (ensure non-empty)
    question_options = state.question_options
    logical_derivation = state.logical_derivation
    domain_knowledge = state.domain_knowledge
    experiment_analysis = state.experiment_analysis
    question_type = state.question_type
    
    # Check which fields are missing and provide detailed error message
    missing_fields = []
    if not logical_derivation:
        missing_fields.append("logical_derivation (Node 4 output)")
    if not domain_knowledge:
        missing_fields.append("domain_knowledge (Node 2 output)")
    if not experiment_analysis:
        missing_fields.append("experiment_analysis (Node 3 output)")
    if not question_type:
        missing_fields.append("question_type (Node 1 output)")
    
    # Note: question_options can be empty for some question types (e.g., True/False, open-ended)
    # But we should try to extract options from question text if not provided
    if not question_options:
        # Try to extract options from user input
        question_text = state.user_input
        extracted_options = _extract_options_from_text(question_text)
        if extracted_options:
            question_options = extracted_options
            state.question_options = extracted_options
            print(f"⚠ Options extracted from question text: {len(extracted_options)} options found")
    
    if missing_fields:
        error_msg = f"Previous nodes' core information missing: {', '.join(missing_fields)}"
        if not question_options:
            error_msg += " (also: no question options found)"
        raise ValueError(error_msg)
    
    # Step 2: Detect logical contradictions before LLM validation (rule-based check)
    has_contradiction, contradiction_reports = detect_logical_contradictions(
        question_type=str(question_type),
        experiment_analysis=experiment_analysis,
        logical_derivation=logical_derivation,
        domain_knowledge=domain_knowledge
    )
    
    if has_contradiction:
        print(f"⚠ Logical contradictions detected:")
        for report in contradiction_reports:
            print(f"  - {report}")
        # Continue with validation but mark contradictions in final result
    
    # Step 2.5: Substantive Validation (NEW - replaces formal validation)
    print("\n[Step 2.5] Performing substantive validation (constraint-knowledge mapping & rebuttal evidence consistency)...")
    
    # Extract preliminary conclusion
    preliminary_conclusion = logical_derivation.get("Preliminary Conclusion", "") if isinstance(logical_derivation, dict) else ""
    if not preliminary_conclusion:
        raise ValueError("Cannot perform substantive validation: Preliminary Conclusion is missing")
    
    # Extract analysis object
    analysis_object = state.key_info.get("Analysis Object", "").strip()
    if not analysis_object or analysis_object in ["None", "", "N/A"]:
        analysis_object = "Unspecified analysis object"
    
    # Extract constraint validation results from experiment analysis
    constraint_validation_results = experiment_analysis.get("Constraint Validation Results", {}) if isinstance(experiment_analysis, dict) else {}
    
    # Extract rebuttal evidence from logical derivation
    rebuttal_evidence = logical_derivation.get("Rebuttal Evidence", {}) if isinstance(logical_derivation, dict) else {}
    
    # Validation 1: Constraint-Knowledge Mapping Consistency
    print("  [Validation 1] Checking consistency with constraint-knowledge mapping table...")
    constraint_validation = _validate_constraint_mapping_consistency(
        preliminary_conclusion=preliminary_conclusion,
        constraint_validation_results=constraint_validation_results,
        analysis_object=analysis_object
    )
    
    if constraint_validation["is_consistent"]:
        print(f"    ✓ {constraint_validation['validation_details']}")
    else:
        print(f"    ✗ {constraint_validation['validation_details']}")
        for reason in constraint_validation["mismatch_reasons"][:3]:  # Show top 3
            print(f"      - {reason}")
    
    # Validation 2: Rebuttal Evidence Consistency
    print("  [Validation 2] Checking consistency with rebuttal evidence...")
    rebuttal_validation = _validate_rebuttal_evidence_consistency(
        preliminary_conclusion=preliminary_conclusion,
        rebuttal_evidence=rebuttal_evidence
    )
    
    if rebuttal_validation["is_consistent"]:
        print(f"    ✓ {rebuttal_validation['validation_details']}")
    else:
        print(f"    ✗ {rebuttal_validation['validation_details']}")
        for reason in rebuttal_validation["contradiction_reasons"][:3]:  # Show top 3
            print(f"      - {reason}")
    
    # Determine if derivation warning should be set
    validation_1_passed = constraint_validation["is_consistent"]
    validation_2_passed = rebuttal_validation["is_consistent"]
    
    if not validation_1_passed and not validation_2_passed:
        derivation_warning = "both_validations_failed"
        warning_message = f"Both validations failed: {constraint_validation['validation_details']}; {rebuttal_validation['validation_details']}"
    elif not validation_1_passed:
        derivation_warning = "constraint_mapping_mismatch"
        warning_message = f"Constraint mapping validation failed: {constraint_validation['validation_details']}"
    elif not validation_2_passed:
        derivation_warning = "rebuttal_evidence_contradiction"
        warning_message = f"Rebuttal evidence validation failed: {rebuttal_validation['validation_details']}"
    else:
        derivation_warning = None
        warning_message = None
    
    # Set derivation_warning in state
    state.derivation_warning = derivation_warning
    
    if derivation_warning:
        print(f"\n⚠ DERIVATION WARNING: {derivation_warning}")
        print(f"  {warning_message}")
        print(f"  → Returning to reasoning_engine_node for re-derivation...")
        # Don't proceed with final validation, return state with warning
        # The graph routing will handle returning to reasoning_engine_node
        # Clear final_result if it exists from previous attempt
        state.final_result = None
        return state
    
    print(f"\n✓ Both substantive validations passed, proceeding with final validation...")
    # Clear derivation_warning since validations passed
    state.derivation_warning = None
    
    # Step 2.6: Universal Verification (三层通用实质验证规则) (NEW)
    print("\n[Step 2.6] Performing universal verification (three-layer rules)...")
    universal_verification_result = _perform_universal_verification(state)
    
    if universal_verification_result["universal_verification_result"] == "FAILED":
        verification_failed_reason = universal_verification_result.get('verification_failed_reason', 'Unknown reason')
        print(f"✗ Universal verification FAILED: {verification_failed_reason}")
        # 验证失败，设置final_error并打回process_data节点重新匹配
        state.final_result = None
        state.final_error = f"Universal verification failed: {verification_failed_reason}"
        state.universal_verification_result = universal_verification_result
        # 设置一个标志，让路由逻辑知道需要打回process_data
        state.derivation_warning = "universal_verification_failed"  # 使用derivation_warning作为路由标志
        return state
    else:
        print(f"✓ Universal verification PASSED")
        state.universal_verification_result = universal_verification_result
    
    # Step 3: Generate closed-loop prompt and call LLM
    print("Completing final validation and option matching, generating five-step method closed-loop result...")
    prompt = get_final_validation_prompt(
        question_options=question_options,
        logical_derivation=logical_derivation,
        domain_knowledge=domain_knowledge,
        experiment_analysis=experiment_analysis,
        question_type=str(question_type)
    )
    
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable, cannot perform final validation. Please check LLM configuration.")
    
    llm_response = _call_llm_for_parsing(llm, prompt)
    if not llm_response:
        raise RuntimeError("LLM call failed, cannot get final validation response.")
    
    print(f"LLM final validation response:\n{llm_response[:300]}...")  # Print first 300 chars to avoid too long
    
    # Step 4: Parse LLM result with auto-fix (three-level repair strategy)
    required_final_dims = ["Final Answer", "Validation Results", "Complete Reasoning Chain", "Key Evidence Summary", "Common Pitfall Reminders"]
    
    try:
        final_data, repair_level = fix_json_format(llm_response, required_keys=required_final_dims)
        if repair_level != "none":
            print(f"⚠ JSON format repaired at {repair_level} level")
    except ValueError as e:
        # If all repair attempts fail, try LLM-based format correction
        print(f"⚠ All JSON format repairs failed, attempting LLM-based correction...")
        fix_prompt = generate_format_fix_prompt(
            original_response=llm_response,
            required_keys=required_final_dims,
            error_message=str(e)
        )
        llm_fix_response = _call_llm_for_parsing(llm, fix_prompt)
        if llm_fix_response:
            try:
                final_data, repair_level = fix_json_format(llm_fix_response, required_keys=required_final_dims)
                print(f"✓ JSON format corrected via LLM retry ({repair_level} level)")
            except ValueError:
                raise ValueError(f"Failed to parse JSON even after LLM-based correction. Raw response: {llm_response[:300]}")
        else:
            raise ValueError(f"LLM format correction call failed. Original error: {str(e)}")
    
    if not isinstance(final_data, dict):
        raise ValueError(f"LLM returned final result data format error, should be dictionary type")
    
    # Verify no non-ASCII key names (should only be English)
    for key in final_data.keys():
        if not all(ord(c) < 128 for c in key):
            raise ValueError(f"LLM returned non-ASCII key name (should be English only): {key}. Please ensure prompt requires English key names.")
    
    # Validate final result's 5 required dimensions
    for dim in required_final_dims:
        if dim not in final_data:
            raise ValueError(f"Final result missing required dimension: {dim}, raw response: {llm_response[:300]}")
    
    # Add contradiction reports to final result if any
    if has_contradiction:
        if "Validation Results" in final_data:
            final_data["Validation Results"] += f" | Logical Contradictions Detected: {len(contradiction_reports)} issues found"
        if "Common Pitfall Reminders" in final_data:
            if isinstance(final_data["Common Pitfall Reminders"], list):
                final_data["Common Pitfall Reminders"].extend(contradiction_reports)
            else:
                final_data["Common Pitfall Reminders"] = [final_data["Common Pitfall Reminders"]] + contradiction_reports
    
    # Step 5: Post-Validation Consistency Check (Priority 1.2)
    # Check if Final Answer is consistent with Preliminary Conclusion
    preliminary_conclusion = logical_derivation.get("Preliminary Conclusion", "") if isinstance(logical_derivation, dict) else ""
    final_answer = final_data.get("Final Answer", "")
    
    if preliminary_conclusion and final_answer:
        consistency_check_result = _check_preliminary_final_consistency(
            preliminary_conclusion=preliminary_conclusion,
            final_answer=final_answer,
            question_type=str(question_type),
            question_options=question_options
        )
        
        if not consistency_check_result["is_consistent"]:
            print(f"⚠ CRITICAL: Final Answer contradicts Preliminary Conclusion!")
            print(f"  Preliminary Conclusion: {preliminary_conclusion[:200]}")
            print(f"  Final Answer: {final_answer[:200]}")
            print(f"  Contradictions: {consistency_check_result['contradictions']}")
            
            # Try to correct based on Preliminary Conclusion
            corrected_answer = consistency_check_result.get("corrected_answer")
            if corrected_answer:
                print(f"  → Correcting Final Answer to: {corrected_answer[:200]}")
                final_data["Final Answer"] = corrected_answer
                if "Validation Results" in final_data:
                    # Priority 3.2: Improved validation results reporting
                    original_validation = final_data["Validation Results"]
                    # Remove any existing Preliminary-Final consistency statement
                    if "Preliminary-Final consistency" in original_validation:
                        original_validation = original_validation.split("Preliminary-Final consistency")[0].rstrip("; ")
                    final_data["Validation Results"] = f"{original_validation}; Preliminary-Final consistency (CORRECTED, original answer contradicted Preliminary Conclusion '{preliminary_conclusion[:100]}', corrected to match)"
            else:
                # If correction failed, add detailed error report
                if "Validation Results" in final_data:
                    original_validation = final_data["Validation Results"]
                    # Remove any existing Preliminary-Final consistency statement
                    if "Preliminary-Final consistency" in original_validation:
                        original_validation = original_validation.split("Preliminary-Final consistency")[0].rstrip("; ")
                    contradiction_summary = ', '.join(consistency_check_result['contradictions'][:2])
                    final_data["Validation Results"] = f"{original_validation}; Preliminary-Final consistency (FAIL, Final Answer contradicts Preliminary Conclusion: {contradiction_summary})"
        else:
            # No contradiction found - add explicit consistency confirmation
            if "Validation Results" in final_data:
                original_validation = final_data["Validation Results"]
                # Check if consistency statement already exists
                if "Preliminary-Final consistency" not in original_validation:
                    final_data["Validation Results"] = f"{original_validation}; Preliminary-Final consistency (Pass, Final Answer is consistent with Preliminary Conclusion)"
    
    # Step 6: Update state (closed-loop successful)
    state.final_result = final_data
    state.final_error = None
    
    print(f"✓ Five-step method closed-loop completed! Final answer: {final_data.get('Final Answer', 'N/A')[:50]}...")
    
    return state


# ---------------------- Dynamic Confidence Calculation Helper ----------------------
def _calculate_dynamic_confidence(
    constraint_validation_results: Dict[str, Dict[str, Any]],
    rebuttal_evidence: Dict[str, Any],
    validation_results_text: str,
    analysis_object: str,
    universal_matching_result: Optional[Dict[str, Any]] = None,
    universal_verification_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    通用置信度计算规则（领域无关，固定权重）
    
    固定权重（所有知识点通用）：
    1. C1_locked_weight: 40% - 核心约束是否锁定
    2. matching_accuracy_weight: 30% - Kc-C1匹配度
    3. verification_result_weight: 30% - 三层验证是否通过
    
    Args:
        constraint_validation_results: Constraint validation results from data processing node (legacy)
        rebuttal_evidence: Rebuttal evidence from logical derivation (legacy)
        validation_results_text: Validation results text from final validation
        analysis_object: Name of the analysis object
        universal_matching_result: Universal matching result from process_data node (NEW)
        universal_verification_result: Universal verification result from validate_conclusion node (NEW)
    
    Returns:
        Dict containing:
        - total_confidence: float (0-100)
        - confidence_calculation_rule: Dict with fixed weights
        - confidence_breakdown: Dict with scores for each indicator
        - confidence_level: str (High/Medium/Low)
    """
    # 通用置信度计算规则（固定权重，所有知识点通用）
    confidence_calculation_rule = {
        "C1_locked_weight": 40,
        "matching_accuracy_weight": 30,
        "verification_result_weight": 30
    }
    
    # Indicator 1: C1_locked_weight (40%)
    C1_locked_score = 0.0
    C1_locked_details = ""
    
    if universal_matching_result:
        # 检查是否有C1被锁定
        has_C1 = False
        for knowledge_name, match_result in universal_matching_result.items():
            C1 = match_result.get("constraint_hierarchy_updated", {}).get("C1_core_constraint", [])
            if C1:
                has_C1 = True
                break
        
        if has_C1:
            C1_locked_score = 40.0
            C1_locked_details = "C1核心约束已锁定"
        else:
            C1_locked_score = 0.0
            C1_locked_details = "C1核心约束未锁定"
    else:
        # 回退到legacy方法
        if constraint_validation_results:
            # 检查是否有匹配的约束场景
            has_match = any(
                result.get("matched_entries", [])
                for result in constraint_validation_results.values()
            )
            if has_match:
                C1_locked_score = 40.0
                C1_locked_details = "Legacy: 约束场景匹配成功"
            else:
                C1_locked_score = 0.0
                C1_locked_details = "Legacy: 无匹配的约束场景"
        else:
            C1_locked_score = 0.0
            C1_locked_details = "无匹配结果"
    
    # Indicator 2: matching_accuracy_weight (30%)
    matching_accuracy_score = 0.0
    matching_accuracy_details = ""
    
    if universal_matching_result:
        all_match_scores = []
        for knowledge_name, match_result in universal_matching_result.items():
            match_info = match_result.get("matching_info", {})
            match_score = match_info.get("match_score", 0.0)
            all_match_scores.append(match_score)
        
        if all_match_scores:
            avg_match_score = sum(all_match_scores) / len(all_match_scores)
            matching_accuracy_score = avg_match_score * 30.0
            matching_accuracy_details = f"Kc-C1匹配度: {avg_match_score:.1%}"
        else:
            matching_accuracy_score = 0.0
            matching_accuracy_details = "无匹配分数"
    else:
        # 回退到legacy方法
        if constraint_validation_results:
            all_scores = [
                result.get("match_score", 0.0)
                for result in constraint_validation_results.values()
            ]
            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                matching_accuracy_score = avg_score * 30.0
                matching_accuracy_details = f"Legacy: 约束匹配度: {avg_score:.1%}"
            else:
                matching_accuracy_score = 0.0
                matching_accuracy_details = "Legacy: 无匹配分数"
        else:
            matching_accuracy_score = 0.0
            matching_accuracy_details = "无匹配结果"
    
    # Indicator 3: verification_result_weight (30%)
    verification_result_score = 0.0
    verification_result_details = ""
    
    if universal_verification_result:
        verification_result = universal_verification_result.get("universal_verification_result", "FAILED")
        if verification_result == "PASSED":
            verification_result_score = 30.0
            verification_result_details = "三层验证全部通过"
        else:
            verification_result_score = 0.0
            verification_result_details = f"验证失败: {universal_verification_result.get('verification_failed_reason', 'Unknown')}"
    else:
        # 回退到legacy方法：检查validation_results_text
        if validation_results_text:
            # 简单检查：如果包含"Pass"或"consistent"，认为通过
            validation_lower = validation_results_text.lower()
            if "pass" in validation_lower or "consistent" in validation_lower:
                verification_result_score = 30.0
                verification_result_details = "Legacy: 验证通过"
            else:
                verification_result_score = 15.0  # 部分分数
                verification_result_details = "Legacy: 验证部分通过"
        else:
            verification_result_score = 0.0
            verification_result_details = "无验证结果"
    
    # 计算总置信度
    total_confidence = C1_locked_score + matching_accuracy_score + verification_result_score
    
    # 确定置信度等级
    if total_confidence >= 80:
        confidence_level = "High"
    elif total_confidence >= 50:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    return {
        "total_confidence": total_confidence,
        "confidence_calculation_rule": confidence_calculation_rule,
        "confidence_breakdown": {
            "C1_locked_score": C1_locked_score,
            "C1_locked_details": C1_locked_details,
            "matching_accuracy_score": matching_accuracy_score,
            "matching_accuracy_details": matching_accuracy_details,
            "verification_result_score": verification_result_score,
            "verification_result_details": verification_result_details
        },
        "confidence_level": confidence_level
    }
    
    if constraint_validation_results:
        # Find validation result for the analysis object
        obj_validation = None
        for obj, validation in constraint_validation_results.items():
            if analysis_object.lower() in obj.lower() or obj.lower() in analysis_object.lower():
                obj_validation = validation
                break
        
        if obj_validation is None and constraint_validation_results:
            obj_validation = list(constraint_validation_results.values())[0]
        
        if obj_validation:
            match_score = obj_validation.get("match_score", 0.0)
            matched_entries = obj_validation.get("matched_entries", [])
            
            if matched_entries:
                # Determine match level based on match_score
                if match_score >= 0.8:  # 80%+ of constraints matched
                    constraint_match_score = 40.0  # Full match
                    constraint_match_details = f"Full match (match_score={match_score:.2f}, {len(matched_entries)} scenarios)"
                elif match_score >= 0.5:  # 50-80% matched
                    constraint_match_score = 20.0  # Partial match
                    constraint_match_details = f"Partial match (match_score={match_score:.2f}, {len(matched_entries)} scenarios)"
                else:  # <50% matched
                    constraint_match_score = 0.0  # No match
                    constraint_match_details = f"No match (match_score={match_score:.2f}, {len(matched_entries)} scenarios)"
            else:
                constraint_match_score = 0.0
                constraint_match_details = "No matched constraint scenarios"
        else:
            constraint_match_score = 0.0
            constraint_match_details = "No constraint validation result available"
    else:
        constraint_match_score = 0.0
        constraint_match_details = "No constraint validation results"
    
    # Indicator 2: Rebuttal Evidence Sufficiency (30%)
    rebuttal_score = 0.0
    rebuttal_details = ""
    
    if rebuttal_evidence and isinstance(rebuttal_evidence, dict):
        rebuttal_conclusion = rebuttal_evidence.get("rebuttal_conclusion", "").lower()
        rebuttal_analysis = rebuttal_evidence.get("rebuttal_analysis", [])
        counter_evidence_considered = rebuttal_evidence.get("counter_evidence_considered", [])
        
        # Check if rebuttal is sufficient and consistent
        if rebuttal_conclusion:
            # Check for positive indicators (strengthens, validates, consistent)
            if any(word in rebuttal_conclusion for word in ["strengthens", "validates", "consistent", "confirms"]):
                rebuttal_score = 30.0  # No contradiction
                rebuttal_details = "Rebuttal evidence shows no contradiction, strengthens conclusion"
            elif any(word in rebuttal_conclusion for word in ["contradicts", "inconsistent", "weak"]):
                rebuttal_score = 10.0  # Insufficient evidence
                rebuttal_details = "Rebuttal evidence has contradictions or weak analysis"
            else:
                # Neutral or unclear
                if isinstance(rebuttal_analysis, list) and len(rebuttal_analysis) >= 2:
                    rebuttal_score = 30.0  # Sufficient analysis
                    rebuttal_details = f"Rebuttal evidence sufficient ({len(rebuttal_analysis)} analysis items)"
                else:
                    rebuttal_score = 10.0  # Insufficient evidence
                    rebuttal_details = "Rebuttal evidence insufficient (limited analysis)"
        elif isinstance(counter_evidence_considered, list) and len(counter_evidence_considered) > 0:
            # Has counter-evidence but no clear conclusion
            rebuttal_score = 10.0
            rebuttal_details = "Rebuttal evidence present but conclusion unclear"
        else:
            rebuttal_score = 0.0
            rebuttal_details = "No rebuttal evidence available"
    else:
        rebuttal_score = 0.0
        rebuttal_details = "No rebuttal evidence"
    
    # Indicator 3: Internal Logic Consistency (30%)
    logic_consistency_score = 0.0
    logic_consistency_details = ""
    
    if validation_results_text:
        validation_lower = validation_results_text.lower()
        
        # Check for consistency indicators
        consistency_keywords = {
            "pass": ["pass", "consistent", "aligned", "matches", "correct"],
            "partial": ["partial", "mostly", "generally", "some"],
            "fail": ["fail", "contradict", "inconsistent", "mismatch", "error"]
        }
        
        pass_count = sum(1 for keyword in consistency_keywords["pass"] if keyword in validation_lower)
        fail_count = sum(1 for keyword in consistency_keywords["fail"] if keyword in validation_lower)
        
        if fail_count == 0 and pass_count >= 2:
            logic_consistency_score = 30.0  # Full consistency
            logic_consistency_details = "Internal logic fully consistent (multiple validations passed)"
        elif fail_count == 0 and pass_count >= 1:
            logic_consistency_score = 20.0  # Partial consistency
            logic_consistency_details = "Internal logic mostly consistent (some validations passed)"
        elif fail_count > 0:
            logic_consistency_score = 0.0  # Inconsistent
            logic_consistency_details = f"Internal logic inconsistent ({fail_count} validation failures detected)"
        else:
            logic_consistency_score = 15.0  # Neutral
            logic_consistency_details = "Internal logic consistency unclear from validation results"
    else:
        logic_consistency_score = 0.0
        logic_consistency_details = "No validation results available"
    
    # Calculate total confidence (sum of all indicators)
    total_confidence = constraint_match_score + rebuttal_score + logic_consistency_score
    
    # Generate confidence description
    confidence_level = "High" if total_confidence >= 70 else "Medium" if total_confidence >= 40 else "Low"
    
    confidence_description = f"Confidence: {total_confidence:.1f}% ({confidence_level}) - " \
                           f"Constraint-Knowledge Match: {constraint_match_score:.1f}% ({constraint_match_details}), " \
                           f"Rebuttal Evidence: {rebuttal_score:.1f}% ({rebuttal_details}), " \
                           f"Logic Consistency: {logic_consistency_score:.1f}% ({logic_consistency_details})"
    
    return {
        "total_confidence": total_confidence,
        "confidence_breakdown": {
            "constraint_knowledge_match": {
                "score": constraint_match_score,
                "max_score": 40.0,
                "details": constraint_match_details
            },
            "rebuttal_evidence_sufficiency": {
                "score": rebuttal_score,
                "max_score": 30.0,
                "details": rebuttal_details
            },
            "internal_logic_consistency": {
                "score": logic_consistency_score,
                "max_score": 30.0,
                "details": logic_consistency_details
            }
        },
        "confidence_description": confidence_description,
        "confidence_level": confidence_level
    }


# ---------------------- Node 6: Final Answer Generation Node (Optimized) ----------------------
def final_answer_node(state: GeneralQAState) -> GeneralQAState:
    """
    Final answer generation node (optimized)
    
    Extract final answer from Node 5's final_result and format it based on output template.
    Supports multiple output formats: student (simplified), researcher (rigorous), clinician (quick reference), custom.
    
    Args:
        state: General QA subgraph state
    
    Returns:
        Updated state (containing final answer formatted according to template)
    """
    # Check if Node 5 completed successfully
    if state.final_error or not state.final_result:
        # 提供更详细的错误信息
        error_msg = state.final_error if state.final_error else "Final result is None"
        if state.universal_verification_result:
            verification_result = state.universal_verification_result.get("universal_verification_result", "UNKNOWN")
            if verification_result == "FAILED":
                verification_failed_reason = state.universal_verification_result.get("verification_failed_reason", "Unknown reason")
                error_msg = f"Universal verification failed: {verification_failed_reason}"
        raise RuntimeError(f"Node 5 final validation failed, cannot generate final answer: {error_msg}")
    
    # Extract final result
    final_result = state.final_result
    
    # Determine output template (default to researcher)
    output_template = state.output_template or "researcher"
    
    # Format final answer based on template
    try:
        formatted_answer = format_final_answer(
            final_result=final_result,
            output_template=output_template,
            custom_fields=state.custom_output_fields
        )
    except Exception as e:
        print(f"⚠ Output formatting failed: {e}, using default format")
        formatted_answer = final_result.get("Final Answer", "")
    
    # Set formatted answer
    state.answer = formatted_answer
    
    # Step 2: Calculate dynamic confidence based on 3 weighted indicators
    print("\n[Step 2] Calculating dynamic confidence...")
    
    # Extract required information for confidence calculation
    constraint_validation_results = {}
    if state.experiment_analysis and isinstance(state.experiment_analysis, dict):
        constraint_validation_results = state.experiment_analysis.get("Constraint Validation Results", {})
    
    rebuttal_evidence = {}
    if state.logical_derivation and isinstance(state.logical_derivation, dict):
        rebuttal_evidence = state.logical_derivation.get("Rebuttal Evidence", {})
    
    validation_results_text = final_result.get("Validation Results", "")
    
    analysis_object = state.key_info.get("Analysis Object", "").strip()
    if not analysis_object or analysis_object in ["None", "", "N/A"]:
        analysis_object = "Unspecified analysis object"
    
    # Calculate dynamic confidence (使用通用置信度计算规则)
    confidence_result = _calculate_dynamic_confidence(
        constraint_validation_results=constraint_validation_results,
        rebuttal_evidence=rebuttal_evidence,
        validation_results_text=validation_results_text,
        analysis_object=analysis_object,
        universal_matching_result=state.universal_matching_result if state.universal_matching_result else None,
        universal_verification_result=state.universal_verification_result if state.universal_verification_result else None
    )
    
    # Format confidence with breakdown (使用通用置信度计算规则)
    total_confidence = confidence_result["total_confidence"]
    confidence_breakdown = confidence_result["confidence_breakdown"]
    confidence_level = confidence_result["confidence_level"]
    confidence_calculation_rule = confidence_result.get("confidence_calculation_rule", {})
    
    # 使用通用字段名格式化
    confidence_text = f"Confidence: {total_confidence:.1f}% ({confidence_level})\n\n" \
                     f"Confidence Calculation Rule (Universal):\n" \
                     f"  - C1_locked_weight: {confidence_calculation_rule.get('C1_locked_weight', 40)}%\n" \
                     f"  - matching_accuracy_weight: {confidence_calculation_rule.get('matching_accuracy_weight', 30)}%\n" \
                     f"  - verification_result_weight: {confidence_calculation_rule.get('verification_result_weight', 30)}%\n\n" \
                     f"Confidence Breakdown:\n" \
                     f"1. C1 Locked: {confidence_breakdown['C1_locked_score']:.1f}% / {confidence_calculation_rule.get('C1_locked_weight', 40)}% - {confidence_breakdown['C1_locked_details']}\n" \
                     f"2. Matching Accuracy: {confidence_breakdown['matching_accuracy_score']:.1f}% / {confidence_calculation_rule.get('matching_accuracy_weight', 30)}% - {confidence_breakdown['matching_accuracy_details']}\n" \
                     f"3. Verification Result: {confidence_breakdown['verification_result_score']:.1f}% / {confidence_calculation_rule.get('verification_result_weight', 30)}% - {confidence_breakdown['verification_result_details']}\n\n" \
                     f"Validation Results: {validation_results_text}"
    
    state.confidence = confidence_text
    
    print(f"✓ Universal confidence calculated: {total_confidence:.1f}% ({confidence_level})")
    print(f"  - C1 Locked: {confidence_breakdown['C1_locked_score']:.1f}%")
    print(f"  - Matching Accuracy: {confidence_breakdown['matching_accuracy_score']:.1f}%")
    print(f"  - Verification Result: {confidence_breakdown['verification_result_score']:.1f}%")
    
    # Extract key evidence summary as related topics
    key_evidence = final_result.get("Key Evidence Summary", [])
    if isinstance(key_evidence, list):
        state.related_topics = key_evidence
    else:
        state.related_topics = [key_evidence] if key_evidence else []
    
    # Extract common pitfall reminders as sources suggested
    pitfalls = final_result.get("Common Pitfall Reminders", [])
    if isinstance(pitfalls, list):
        state.sources_suggested = pitfalls
    else:
        state.sources_suggested = [pitfalls] if pitfalls else []
    
    # Extract complete reasoning chain for reference
    reasoning_chain = final_result.get("Complete Reasoning Chain", "")
    
    print(f"✓ Final answer formatted using '{output_template}' template")
    print(f"  Final Answer: {final_result.get('Final Answer', 'N/A')[:100]}...")
    if reasoning_chain:
        print(f"  Reasoning Chain: {reasoning_chain[:100]}...")
    
    return state


# ---------------------- State Mapping Functions ----------------------
def general_qa_input_mapper(global_state: GlobalState) -> GeneralQAState:
    """
    Main graph → Subgraph state mapping
    
    Map main graph's GlobalState to GeneralQAState, extract information needed by subgraph.
    
    Args:
        global_state: Main graph's global state
    
    Returns:
        GeneralQAState: Subgraph state
    """
    # Try to extract options from merged_result (if any)
    question_options = []
    if global_state.merged_result:
        # Check if there is option information
        if "question_options" in global_state.merged_result:
            question_options = global_state.merged_result["question_options"]
        elif "options" in global_state.merged_result:
            question_options = global_state.merged_result["options"]
    
    return GeneralQAState(
        user_input=global_state.user_input,
        question_options=question_options,
        # All other fields use default values, will be filled step by step in subgraph nodes
    )


def general_qa_output_mapper(subgraph_output: GeneralQAState | dict, global_state: GlobalState) -> GlobalState:
    """
    Subgraph → Main graph state mapping
    
    Synchronize subgraph's GeneralQAState results back to main graph's GlobalState.
    
    Args:
        subgraph_output: Subgraph output state (may be GeneralQAState object or dictionary)
        global_state: Main graph's global state (will be updated)
    
    Returns:
        GlobalState: Updated main graph state
    """
    
    # Handle dictionary format state (LangGraph may return dictionary)
    if isinstance(subgraph_output, dict):
        subgraph_output = GeneralQAState(**subgraph_output)
    
    # Store answer to merged_result
    if not global_state.merged_result:
        global_state.merged_result = {}
    
    if subgraph_output.answer:
        global_state.merged_result["general_qa_answer"] = subgraph_output.answer
    
    if subgraph_output.confidence:
        global_state.merged_result["general_qa_confidence"] = subgraph_output.confidence
    
    if subgraph_output.related_topics:
        global_state.merged_result["general_qa_related_topics"] = subgraph_output.related_topics
    
    if subgraph_output.sources_suggested:
        global_state.merged_result["general_qa_sources"] = subgraph_output.sources_suggested
    
    # Save detailed node states for debugging
    node_states_detail = {}
    
    # Node 1: Question Parsing (always record, even if failed)
    node_states_detail["node1_parse_question"] = {
        "question_format_type": str(subgraph_output.question_format_type) if subgraph_output.question_format_type else None,
        "question_type": str(subgraph_output.question_type) if subgraph_output.question_type else None,
        "key_info_keys": list(subgraph_output.key_info.keys()) if subgraph_output.key_info else None,
        "key_info_preview": {k: str(v)[:100] for k, v in list(subgraph_output.key_info.items())[:3]} if subgraph_output.key_info else None,
        "constraint_tags": subgraph_output.constraint_tags if subgraph_output.constraint_tags else {},
        "parse_error": subgraph_output.parse_error,
        "domain": str(subgraph_output.domain) if subgraph_output.domain else None,
        "subdomain": str(subgraph_output.subdomain) if subgraph_output.subdomain else None,
        "question_options_count": len(subgraph_output.question_options) if subgraph_output.question_options else 0,
        "status": "success" if not subgraph_output.parse_error and subgraph_output.question_type else "failed"
    }
    
    # Node 2: Knowledge Activation (always record)
    node_states_detail["node2_activate_knowledge"] = {
        "domain_knowledge_keys": list(subgraph_output.domain_knowledge.keys()) if isinstance(subgraph_output.domain_knowledge, dict) else None,
        "domain_knowledge_count": len(subgraph_output.domain_knowledge) if isinstance(subgraph_output.domain_knowledge, dict) else 0,
        "knowledge_error": subgraph_output.knowledge_error,
        "status": "success" if not subgraph_output.knowledge_error and subgraph_output.domain_knowledge else "failed"
    }
    
    # Node 3: Data Processing (always record)
    node_states_detail["node3_process_data"] = {
        "experiment_analysis_keys": list(subgraph_output.experiment_analysis.keys()) if isinstance(subgraph_output.experiment_analysis, dict) else None,
        "experiment_analysis_preview": {k: str(v)[:100] if isinstance(v, (str, list)) else str(v)[:100] for k, v in list(subgraph_output.experiment_analysis.items())[:2]} if isinstance(subgraph_output.experiment_analysis, dict) else None,
        "analysis_error": subgraph_output.analysis_error,
        "status": "success" if not subgraph_output.analysis_error and subgraph_output.experiment_analysis else "failed"
    }
    
    # Node 4: Reasoning Engine (always record)
    node_states_detail["node4_reasoning_engine"] = {
        "logical_derivation_keys": list(subgraph_output.logical_derivation.keys()) if isinstance(subgraph_output.logical_derivation, dict) else None,
        "preliminary_conclusion": subgraph_output.logical_derivation.get("Preliminary Conclusion", "")[:300] if isinstance(subgraph_output.logical_derivation, dict) else None,
        "derivation_strategy": subgraph_output.logical_derivation.get("Derivation Strategy", "")[:200] if isinstance(subgraph_output.logical_derivation, dict) else None,
        "derivation_error": subgraph_output.derivation_error,
        "derivation_warning": subgraph_output.derivation_warning,
        "status": "success" if not subgraph_output.derivation_error and subgraph_output.logical_derivation else "failed"
    }
    
    # Node 5: Conclusion Validation (always record)
    node_states_detail["node5_validate_conclusion"] = {
        "final_result_keys": list(subgraph_output.final_result.keys()) if isinstance(subgraph_output.final_result, dict) else None,
        "final_answer": subgraph_output.final_result.get("Final Answer", "")[:300] if isinstance(subgraph_output.final_result, dict) else None,
        "validation_results": subgraph_output.final_result.get("Validation Results", "")[:200] if isinstance(subgraph_output.final_result, dict) else None,
        "derivation_warning": subgraph_output.derivation_warning,
        "final_error": subgraph_output.final_error,
        "status": "success" if not subgraph_output.final_error and subgraph_output.final_result else "failed"
    }
    
    # Node 6: Final Answer Generation (always record)
    node_states_detail["node6_generate_answer"] = {
        "answer_length": len(subgraph_output.answer) if subgraph_output.answer else 0,
        "answer_preview": subgraph_output.answer[:300] if subgraph_output.answer else None,
        "confidence": subgraph_output.confidence[:300] if subgraph_output.confidence else None,
        "related_topics_count": len(subgraph_output.related_topics) if subgraph_output.related_topics else 0,
        "related_topics_preview": subgraph_output.related_topics[:3] if subgraph_output.related_topics else None,
        "status": "success" if subgraph_output.answer else "failed"
    }
    
    if node_states_detail:
        global_state.merged_result["general_qa_node_states"] = node_states_detail
    
    # Return updated global state
    return global_state


# ---------------------- Build General QA Agent Subgraph ----------------------
def build_general_qa_subgraph():
    """
    Build optimized general QA Agent subgraph
    
    New architecture contains 6 core nodes:
    1. Question parsing node - Automatically classify question type, extract key information
    2. Knowledge activation node - Automatically activate corresponding knowledge modules based on domain
    3. Data processing node - Analyze experimental condition impacts, identify error sources
    4. Reasoning engine node - Select reasoning strategy based on question type
    5. Conclusion validation node - Validate conclusion reasonableness, match answer options
    6. Final answer generation node - Generate final answer based on all processing results
    
    Returns:
        Compiled subgraph
    """
    graph = StateGraph(GeneralQAState)
    
    # Add all nodes
    graph.add_node("parse_question", question_parsing_node)  # Node 1: Question parsing
    graph.add_node("activate_knowledge", knowledge_activation_node)  # Node 2: Knowledge activation
    graph.add_node("process_data", data_processing_node)  # Node 3: Data processing
    graph.add_node("reasoning_engine", reasoning_engine_node)  # Node 4: Reasoning engine
    graph.add_node("validate_conclusion", conclusion_validation_node)  # Node 5: Conclusion validation
    graph.add_node("generate_answer", final_answer_node)  # Node 6: Final answer generation
    
    # Define flow rules
    graph.add_edge(START, "parse_question")
    graph.add_edge("parse_question", "activate_knowledge")
    graph.add_edge("activate_knowledge", "process_data")
    graph.add_edge("process_data", "reasoning_engine")
    graph.add_edge("reasoning_engine", "validate_conclusion")
    
    # Conditional edge: check derivation_warning and universal_verification_result
    def should_rederive(state: GeneralQAState) -> str:
        """
        Determine next node based on derivation_warning and universal_verification_result
        
        Routing logic:
        1. If universal_verification_result is FAILED -> return to process_data (重新匹配)
        2. If derivation_warning exists -> return to reasoning_engine (重新推理)
        3. Otherwise -> proceed to generate_answer
        """
        # 检查通用验证结果
        if state.universal_verification_result:
            verification_result = state.universal_verification_result.get("universal_verification_result", "PASSED")
            if verification_result == "FAILED":
                return "process_data"  # 验证失败，打回process_data重新匹配
        
        # 检查推理警告
        if state.derivation_warning:
            # 如果是universal_verification_failed，已经在上面的检查中处理了
            if state.derivation_warning != "universal_verification_failed":
                return "reasoning_engine"  # Return to reasoning engine for re-derivation
        
        # 检查final_result是否存在
        if not state.final_result:
            # 如果没有final_result，检查是否有错误
            if state.final_error:
                # 如果有错误，根据错误类型决定路由
                if "universal verification" in state.final_error.lower():
                    return "process_data"  # 验证失败，打回process_data
                else:
                    return "reasoning_engine"  # 其他错误，打回reasoning_engine
            else:
                # 没有final_result也没有错误，可能是验证失败但未设置错误，打回process_data
                return "process_data"
        
        return "generate_answer"  # Proceed to final answer generation
    
    graph.add_conditional_edges(
        "validate_conclusion",
        should_rederive,
        {
            "process_data": "process_data",  # 新增：验证失败时打回process_data
            "reasoning_engine": "reasoning_engine",
            "generate_answer": "generate_answer"
        }
    )
    
    graph.add_edge("generate_answer", END)
    
    return graph.compile()

