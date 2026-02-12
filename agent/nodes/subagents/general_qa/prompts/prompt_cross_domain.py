"""
Cross-Domain Prompt Module
Handles questions that span multiple domains
All prompts are in English as required.
"""

from typing import Dict, List, Any, Optional, Tuple
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
from .domain_mapper import get_prompt_module

# 存储涉及的领域（由domain_mapper设置）
_domains: List[str] = []

# 高频跨领域组合（预编译）
PRECOMPILED_CROSS_DOMAINS: Dict[Tuple[str, ...], Dict[str, Any]] = {
    ("Genetics", "Bioinformatics"): {
        "extraction_rules": """
**Genetics+Bioinformatics Combined Rules:**
- Extract genetic variants and computational parameters
- Identify population genetics formulas and statistical tests
- Combine genetic inheritance patterns with computational analysis
""",
        "tools": ["query_gwas_catalog", "query_genebass", "query_variant", "query_knowledge_graph"],
    },
    ("Immunology", "Clinical Medicine"): {
        "extraction_rules": """
**Immunology+Clinical Medicine Combined Rules:**
- Extract immune cell types and clinical treatment information
- Identify immunotherapy and clinical decision criteria
- Combine immune mechanisms with treatment guidelines
""",
        "tools": ["query_tcr_mcpas", "query_drug_for_disease", "query_celltype_marker"],
    },
}


def get_precompiled_rules(domains: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """获取预编译的跨领域规则"""
    domain_key = tuple(sorted(domains))  # 排序后作为key
    return PRECOMPILED_CROSS_DOMAINS.get(domain_key)


def _get_domain_enhancements(domains: List[str], node_name: str, **kwargs) -> str:
    """收集所有涉及领域的增强规则"""
    enhancements = []
    
    # 检查是否有预编译规则
    domain_tuple = tuple(sorted(domains))
    precompiled = get_precompiled_rules(domain_tuple)
    if precompiled and "extraction_rules" in precompiled:
        enhancements.append(precompiled["extraction_rules"])
    
    # 收集各领域的增强规则
    for domain in domains:
        try:
            domain_module = get_prompt_module(domain=domain)
            if hasattr(domain_module, 'get_domain_extraction_rules'):
                rules = domain_module.get_domain_extraction_rules()
                if rules:
                    enhancements.append(f"**{domain} Rules:**\n{rules}")
        except:
            continue
    
    return "\n\n".join(enhancements) if enhancements else ""


# ========== N0: Input Preprocessing & Question Classification ==========

def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0: Input preprocessing - Cross-domain (merged rules from all domains)"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    domains = _domains if _domains else []
    
    if domains:
        enhancements = _get_domain_enhancements(domains, "n0_input_preprocessing", user_input=user_input)
        if enhancements:
            return base_prompt + f"\n\n**Multi-Domain Extraction Rules (merged from {len(domains)} domains):**\n{enhancements}"
    
    return base_prompt


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
    """N1: Question decomposition - Cross-domain"""
    base_prompt = get_base_question_decomposition_prompt(
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )
    
    domains = _domains if _domains else []
    if domains:
        enhancements = _get_domain_enhancements(domains, "n1_question_decomposition", cleaned_text=cleaned_text)
        if enhancements:
            return base_prompt + f"\n\n**Multi-Domain Decomposition Rules:**\n{enhancements}"
    
    return base_prompt


# ========== N2: Calculation/Algorithm Requirement Recognition ==========

def get_calculation_algorithm_recognition_prompt(cleaned_text: str, question_type_label: str) -> str:
    """N2: Calculation/algorithm recognition - Cross-domain"""
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
    synonyms: List[str] = None
) -> str:
    """N3: Knowledge retrieval - Cross-domain (merged tools from all domains)"""
    base_prompt = get_base_knowledge_retrieval_prompt(
        core_domains, calculation_type, algorithm_domain, research_objective,
        structured_conditions, key_entities, answer_format_label, question_type_label,
        structured_subject, structured_condition, structured_goal, synonyms
    )
    
    domains = _domains if _domains else core_domains
    if domains:
        # 收集所有领域的工具
        all_tools = set()
        for domain in domains:
            try:
                domain_module = get_prompt_module(domain=domain)
                if hasattr(domain_module, 'get_domain_tools'):
                    tools = domain_module.get_domain_tools()
                    all_tools.update(tools)
            except:
                continue
        
        # 检查预编译规则
        domain_tuple = tuple(sorted(domains))
        precompiled = get_precompiled_rules(domain_tuple)
        if precompiled and "tools" in precompiled:
            all_tools.update(precompiled["tools"])
        
        if all_tools:
            tools_str = ", ".join(sorted(all_tools))
            return base_prompt + f"\n\n**Multi-Domain Tools (merged from {len(domains)} domains):**\nPriority tools: {tools_str}"
    
    return base_prompt


# ========== N4-N11: Use base templates ==========

def get_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any]
) -> str:
    """N4: Calculation decomposition - Cross-domain"""
    return get_base_calculation_decomposition_prompt(cleaned_text, key_parameters, domain_knowledge)


def get_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any]
) -> str:
    """N5: Algorithm validation - Cross-domain"""
    return get_base_algorithm_validation_prompt(cleaned_text, algorithm_name, domain_knowledge)


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
    """N6: Initial inference - Cross-domain"""
    return get_base_initial_inference_prompt(
        cleaned_text, research_objective, key_entities, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal
    )


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
    """N7: Complete inference - Cross-domain"""
    return get_base_complete_inference_prompt(
        cleaned_text, research_objective, initial_associations, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal,
        calculation_result
    )


def get_answer_generation_prompt(
    core_conclusion: str,
    question_type_label: str,
    question_options: List[str] = None,
    calculation_result: Any = None,
    answer_format_label: str = None,
    answer_constraints: List[str] = None,
    structured_goal: Dict[str, Any] = None
) -> str:
    """N8: Answer generation - Cross-domain"""
    return get_base_answer_generation_prompt(
        core_conclusion, question_type_label, question_options,
        calculation_result, answer_format_label, answer_constraints, structured_goal
    )


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
    """N9: Result validation - Cross-domain"""
    return get_base_result_validation_prompt(
        structured_answer, closed_inference_path, answer_format_label,
        question_options, answer_constraints, question_type_label,
        hard_constraints, structured_goal, core_keywords, option_features
    )


def get_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any]
) -> str:
    """N10: Exception handling - Cross-domain"""
    return get_base_exception_handling_prompt(exception_type, exception_context)


def get_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any]
) -> str:
    """N11: Manual intervention - Cross-domain"""
    return get_base_manual_intervention_prompt(exception_type, intermediate_results)


# ========== Domain-specific functions ==========

def get_domain_tools(domains: Optional[List[str]] = None) -> List[str]:
    """Return merged priority tools from all involved domains"""
    target_domains = domains or _domains
    if not target_domains:
        return []
    
    all_tools = set()
    
    # 检查预编译规则
    domain_tuple = tuple(sorted(target_domains))
    precompiled = get_precompiled_rules(domain_tuple)
    if precompiled and "tools" in precompiled:
        all_tools.update(precompiled["tools"])
    
    # 收集各领域的工具
    for domain in target_domains:
        try:
            domain_module = get_prompt_module(domain=domain)
            if hasattr(domain_module, 'get_domain_tools'):
                tools = domain_module.get_domain_tools()
                all_tools.update(tools)
        except:
            continue
    
    # 按领域优先级排序（Genetics > Immunology > Clinical Medicine > ...）
    priority_order = {
        "Genetics": 1,
        "Immunology": 2,
        "Clinical Medicine": 3,
        "Bioinformatics": 4,
        "Biochemistry": 5,
        "Molecular Biology": 6,
    }
    
    # 按优先级排序工具
    sorted_tools = sorted(all_tools, key=lambda t: priority_order.get(t, 99))
    return sorted_tools


def get_domain_extraction_rules(domains: Optional[List[str]] = None) -> str:
    """Return merged domain-specific extraction rules"""
    target_domains = domains or _domains
    if not target_domains:
        return ""
    
    rules = []
    
    # 检查预编译规则
    domain_tuple = tuple(sorted(target_domains))
    precompiled = get_precompiled_rules(domain_tuple)
    if precompiled and "extraction_rules" in precompiled:
        rules.append(precompiled["extraction_rules"])
    
    # 收集各领域的规则
    for domain in target_domains:
        try:
            domain_module = get_prompt_module(domain=domain)
            if hasattr(domain_module, 'get_domain_extraction_rules'):
                domain_rules = domain_module.get_domain_extraction_rules()
                if domain_rules:
                    rules.append(f"**{domain} Rules:**\n{domain_rules}")
        except:
            continue
    
    return "\n\n".join(rules) if rules else ""

