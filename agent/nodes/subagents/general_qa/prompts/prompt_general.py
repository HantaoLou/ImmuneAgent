"""
General Domain Prompt Module
Default prompt implementation for questions that don't match specific domains
All prompts are in English as required.
"""

from typing import Dict, List, Any, Optional
from .base import (
    get_base_input_preprocessing_prompt,
    get_base_question_decomposition_prompt,
    get_base_calculation_algorithm_recognition_prompt,
    get_base_knowledge_retrieval_prompt,
    get_base_calculation_decomposition_prompt,
    get_base_algorithm_validation_prompt,
    get_base_initial_inference_prompt,
    get_base_complete_inference_prompt,
    get_base_answer_generation_prompt,
    get_base_result_validation_prompt,
    get_base_exception_handling_prompt,
    get_base_manual_intervention_prompt,
)

# 通用领域配置
DOMAIN_CONFIG = {
    "name": "General",
    "priority_tools": [
        "query_knowledge_graph",
        "query_gene_info",
        "query_proteinatlas",
        "query_go_term"
    ],
    "common_entities": [],
    "calculation_focus": [],
    "validation_criteria": [
        "Must verify against general biomedical principles",
        "Check logical consistency",
        "Verify fact accuracy"
    ]
}


# ========== N0: Input Preprocessing & Question Classification ==========

def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0: Input preprocessing - General domain (no domain-specific enhancements)"""
    return get_base_input_preprocessing_prompt(user_input)


# ========== N1: Question Decomposition & Domain Localization ==========

def get_question_decomposition_prompt(
    cleaned_text: str,
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None
) -> str:
    """N1: Question decomposition - General domain"""
    return get_base_question_decomposition_prompt(
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )


# ========== N2: Calculation/Algorithm Requirement Recognition ==========

def get_calculation_algorithm_recognition_prompt(cleaned_text: str, question_type_label: str) -> str:
    """N2: Calculation/algorithm recognition - General domain"""
    return get_base_calculation_algorithm_recognition_prompt(cleaned_text, question_type_label)


# ========== N3: Cross-Domain Knowledge Retrieval ==========

def get_knowledge_retrieval_prompt(
    core_domains: List[str],
    calculation_type: str = None,
    algorithm_domain: str = None,
    research_objective: str = None,
    structured_conditions: Dict[str, Any] = None,
    key_entities: List[str] = None,
    answer_format_label: str = None,
    question_type_label: str = None,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    synonyms: List[str] = None,
    cleaned_text: str = None  # ENHANCEMENT: Add for data analysis
) -> str:
    """N3: Knowledge retrieval - General domain"""
    return get_base_knowledge_retrieval_prompt(
        core_domains, calculation_type, algorithm_domain, research_objective,
        structured_conditions, key_entities, answer_format_label, question_type_label,
        structured_subject, structured_condition, structured_goal, synonyms,
        cleaned_text=cleaned_text  # ENHANCEMENT: Pass cleaned_text
    )


# ========== N4: Calculation Step Decomposition & Formula Matching ==========

def get_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any]
) -> str:
    """N4: Calculation decomposition - General domain"""
    return get_base_calculation_decomposition_prompt(cleaned_text, key_parameters, domain_knowledge)


# ========== N5: Algorithm Parameter Extraction & Applicability Validation ==========

def get_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any]
) -> str:
    """N5: Algorithm validation - General domain"""
    return get_base_algorithm_validation_prompt(cleaned_text, algorithm_name, domain_knowledge)


# ========== N6: Initial Association Inference ==========

def get_initial_inference_prompt(
    cleaned_text: str,
    research_objective: str,
    key_entities: List[str],
    retrieved_knowledge: Dict[str, Any],
    question_options: List[str] = None,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None
) -> str:
    """N6: Initial inference - General domain"""
    return get_base_initial_inference_prompt(
        cleaned_text, research_objective, key_entities, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal
    )


# ========== N7: Complete Logical Inference ==========

def get_complete_inference_prompt(
    cleaned_text: str,
    research_objective: str,
    initial_associations: List[Dict[str, Any]],
    retrieved_knowledge: Dict[str, Any],
    question_options: List[str] = None,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    calculation_result: Any = None
) -> str:
    """N7: Complete inference - General domain"""
    return get_base_complete_inference_prompt(
        cleaned_text, research_objective, initial_associations, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal,
        calculation_result
    )


# ========== N8: Multi-Type Answer Generation ==========

def get_answer_generation_prompt(
    core_conclusion: str,
    question_type_label: str,
    question_options: List[str] = None,
    calculation_result: Any = None,
    answer_format_label: str = None,
    answer_constraints: List[str] = None,
    structured_goal: Dict[str, Any] = None
) -> str:
    """N8: Answer generation - General domain"""
    return get_base_answer_generation_prompt(
        core_conclusion, question_type_label, question_options,
        calculation_result, answer_format_label, answer_constraints, structured_goal
    )


# ========== N9: Result Validation & Consistency Judgment ==========

def get_result_validation_prompt(
    structured_answer: Dict[str, Any],
    closed_inference_path: List[Dict[str, Any]],
    answer_format_label: str = None,
    question_options: List[str] = None,
    answer_constraints: List[str] = None,
    question_type_label: str = None,
    hard_constraints: List[str] = None,
    structured_goal: Dict[str, Any] = None,
    core_keywords: str = "N/A",
    option_features: str = "N/A"
) -> str:
    """N9: Result validation - General domain"""
    return get_base_result_validation_prompt(
        structured_answer, closed_inference_path, answer_format_label,
        question_options, answer_constraints, question_type_label,
        hard_constraints, structured_goal, core_keywords, option_features
    )


# ========== N10: Knowledge/Calculation Exception Handling ==========

def get_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any]
) -> str:
    """N10: Exception handling - General domain"""
    return get_base_exception_handling_prompt(exception_type, exception_context)


# ========== N11: Manual Intervention Trigger ==========

def get_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any]
) -> str:
    """N11: Manual intervention - General domain"""
    return get_base_manual_intervention_prompt(exception_type, intermediate_results)


# ========== Domain-specific functions ==========

def get_domain_tools() -> List[str]:
    """Return priority tools for general domain"""
    return DOMAIN_CONFIG["priority_tools"]


def get_domain_extraction_rules() -> str:
    """Return domain-specific extraction rules for cross-domain merging"""
    return """
**General Domain Extraction Rules:**
- Extract general biomedical entities and concepts
- Identify common biological processes and mechanisms
- Apply general biomedical principles
"""

