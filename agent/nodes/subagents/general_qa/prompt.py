"""
Prompt definitions for General QA subgraph nodes
All prompts are in English as required.

This module serves as a unified entry point that routes to domain-specific prompts.
Maintains backward compatibility with existing code.
"""

from typing import Dict, List, Any, Optional
from .prompts.domain_mapper import get_prompt_module, detect_domain_from_state


def _get_prompt_func(
    func_name: str,
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    user_input: Optional[str] = None,
    core_domains: Optional[List[str]] = None,
    *args,
    **kwargs
):
    """
    Internal helper to route to domain-specific prompt function
    
    Args:
        func_name: Name of the prompt function to call
        domain: Domain from raw_subject or core_domains
        question_type: Question type label
        user_input: User input text (for domain identification)
        core_domains: Already identified core domains list
        *args, **kwargs: Arguments to pass to the prompt function
    
    Returns:
        Prompt string from domain-specific module
    """
    # Get the appropriate prompt module
    module = get_prompt_module(
        domain=domain,
        question_type=question_type,
        user_input=user_input,
        core_domains=core_domains
    )
    
    # Get the function from the module
    func = getattr(module, func_name)
    
    # Merge user_input into kwargs if provided (for prompt functions that need it)
    if user_input and 'user_input' not in kwargs:
        kwargs['user_input'] = user_input
    
    # Check if the function accepts core_domains parameter before passing it
    import inspect
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        accepts_core_domains = 'core_domains' in params
    except (ValueError, TypeError):
        # If signature inspection fails, assume it doesn't accept core_domains
        accepts_core_domains = False
    
    # Merge core_domains into kwargs if provided and function accepts it
    # This allows core_domains to be passed to the actual prompt function
    if core_domains is not None and accepts_core_domains and 'core_domains' not in kwargs:
        kwargs['core_domains'] = core_domains
    
    # For functions that require core_domains as first positional argument,
    # extract it from kwargs and pass as positional argument
    if 'core_domains' in kwargs and accepts_core_domains:
        core_domains_arg = kwargs.pop('core_domains')
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # If the first parameter is 'core_domains', pass it as positional argument
            if params and params[0] == 'core_domains':
                return func(core_domains_arg, *args, **kwargs)
        except (ValueError, TypeError):
            # If signature inspection fails, try passing core_domains as keyword argument
            kwargs['core_domains'] = core_domains_arg
    
    # Otherwise, call normally
    return func(*args, **kwargs)


# ========== N0: Input Preprocessing & Question Classification ==========

def get_input_preprocessing_prompt(
    user_input: str,
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N0: Input preprocessing and question classification
    
    Args:
        user_input: User input text
        domain: Domain from raw_subject (optional, will be auto-detected if not provided)
        question_type: Question type label (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_input_preprocessing_prompt",
        domain=domain,
        question_type=question_type,
        user_input=user_input,
        core_domains=core_domains
    )


# ========== N1: Question Decomposition & Domain Localization ==========

def get_question_decomposition_prompt(
    cleaned_text: str, 
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None,
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N1: Question decomposition and domain localization
    
    Args:
        cleaned_text: Cleaned question text
        question_type_label: Question type label
        structured_subject: Structured subject information
        structured_condition: Structured condition information
        structured_goal: Structured goal information
        question_category_standard: Question category standard
        category_specific_constraints: Category-specific constraints
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional, defaults to question_type_label)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_question_decomposition_prompt",
        domain=domain,
        question_type=question_type or question_type_label,
        core_domains=core_domains,
        cleaned_text=cleaned_text,
        question_type_label=question_type_label,
        structured_subject=structured_subject,
        structured_condition=structured_condition,
        structured_goal=structured_goal,
        question_category_standard=question_category_standard,
        category_specific_constraints=category_specific_constraints
    )


# ========== N2: Calculation/Algorithm Requirement Recognition ==========

def get_calculation_algorithm_recognition_prompt(
    cleaned_text: str,
    question_type_label: str,
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N2: Calculation/algorithm requirement recognition
    
    Args:
        cleaned_text: Cleaned question text
        question_type_label: Question type label
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_calculation_algorithm_recognition_prompt",
        domain=domain,
        question_type=question_type or question_type_label,
        core_domains=core_domains,
        cleaned_text=cleaned_text,
        question_type_label=question_type_label
    )


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
    domain: Optional[str] = None,
    question_type: Optional[str] = None
) -> str:
    """
    Prompt for N3: Cross-domain knowledge retrieval
    
    Args:
        core_domains: Core domains list
        calculation_type: Calculation type
        algorithm_domain: Algorithm domain
        research_objective: Research objective
        structured_conditions: Structured conditions
        key_entities: Key entities list
        answer_format_label: Answer format label
        question_type_label: Question type label
        structured_subject: Structured subject information
        structured_condition: Structured condition information
        structured_goal: Structured goal information
        synonyms: Synonyms list
        domain: Domain from raw_subject (optional, will use core_domains[0] if not provided)
        question_type: Question type (optional)
    
    Returns:
        Prompt string
    """
    # Use first core_domain if domain not provided
    if not domain and core_domains and len(core_domains) > 0:
        domain = core_domains[0]
    
    return _get_prompt_func(
        "get_knowledge_retrieval_prompt",
        domain=domain,
        question_type=question_type or question_type_label,
        core_domains=core_domains,  # For domain routing in _get_prompt_func (will also be added to kwargs)
        # All other arguments go to kwargs to be passed to the actual prompt function
        calculation_type=calculation_type,
        algorithm_domain=algorithm_domain,
        research_objective=research_objective,
        structured_conditions=structured_conditions,
        key_entities=key_entities,
        answer_format_label=answer_format_label,
        question_type_label=question_type_label,
        structured_subject=structured_subject,
        structured_condition=structured_condition,
        structured_goal=structured_goal,
        synonyms=synonyms
    )


# ========== N4: Calculation Step Decomposition & Formula Matching ==========

def get_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any],
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N4: Calculation step decomposition and formula matching
    
    Args:
        cleaned_text: Cleaned question text
        key_parameters: Key parameters dictionary
        domain_knowledge: Domain knowledge dictionary
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_calculation_decomposition_prompt",
        domain=domain,
        question_type=question_type,
        core_domains=core_domains,
        cleaned_text=cleaned_text,
        key_parameters=key_parameters,
        domain_knowledge=domain_knowledge
    )


# ========== N5: Algorithm Parameter Extraction & Applicability Validation ==========

def get_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any],
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N5: Algorithm parameter extraction and applicability validation
    
    Args:
        cleaned_text: Cleaned question text
        algorithm_name: Algorithm name
        domain_knowledge: Domain knowledge dictionary
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_algorithm_validation_prompt",
        domain=domain,
        question_type=question_type,
        core_domains=core_domains,
        cleaned_text=cleaned_text,
        algorithm_name=algorithm_name,
        domain_knowledge=domain_knowledge
    )


# ========== N6: Initial Association Inference ==========

def get_initial_inference_prompt(
    cleaned_text: str,
    research_objective: str,
    key_entities: List[str],
    retrieved_knowledge: Dict[str, Any],
    question_options: List[str] = None,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N6: Initial association inference
    
    Args:
        cleaned_text: Cleaned question text
        research_objective: Research objective
        key_entities: Key entities list
        retrieved_knowledge: Retrieved knowledge dictionary
        question_options: Question options list
        structured_subject: Structured subject information
        structured_condition: Structured condition information
        structured_goal: Structured goal information
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_initial_inference_prompt",
        domain=domain,
        question_type=question_type,
        core_domains=core_domains,
        cleaned_text=cleaned_text,
        research_objective=research_objective,
        key_entities=key_entities,
        retrieved_knowledge=retrieved_knowledge,
        question_options=question_options,
        structured_subject=structured_subject,
        structured_condition=structured_condition,
        structured_goal=structured_goal
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
    calculation_result: Any = None,
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N7: Complete logical inference
    
    Args:
        cleaned_text: Cleaned question text
        research_objective: Research objective
        initial_associations: Initial associations list
        retrieved_knowledge: Retrieved knowledge dictionary
        question_options: Question options list
        structured_subject: Structured subject information
        structured_condition: Structured condition information
        structured_goal: Structured goal information
        calculation_result: Calculation result
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_complete_inference_prompt",
        domain=domain,
        question_type=question_type,
        core_domains=core_domains,
        cleaned_text=cleaned_text,
        research_objective=research_objective,
        initial_associations=initial_associations,
        retrieved_knowledge=retrieved_knowledge,
        question_options=question_options,
        structured_subject=structured_subject,
        structured_condition=structured_condition,
        structured_goal=structured_goal,
        calculation_result=calculation_result
    )


# ========== N8: Multi-Type Answer Generation ==========

def get_answer_generation_prompt(
    core_conclusion: str,
    question_type_label: str,
    question_options: List[str] = None,
    calculation_result: Any = None,
    answer_format_label: str = None,
    answer_constraints: List[str] = None,
    structured_goal: Dict[str, Any] = None,
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N8: Multi-type answer generation
    
    Args:
        core_conclusion: Core conclusion
        question_type_label: Question type label
        question_options: Question options list
        calculation_result: Calculation result
        answer_format_label: Answer format label
        answer_constraints: Answer constraints list
        structured_goal: Structured goal information
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_answer_generation_prompt",
        domain=domain,
        question_type=question_type or question_type_label,
        core_domains=core_domains,
        core_conclusion=core_conclusion,
        question_type_label=question_type_label,
        question_options=question_options,
        calculation_result=calculation_result,
        answer_format_label=answer_format_label,
        answer_constraints=answer_constraints,
        structured_goal=structured_goal
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
    option_features: str = "N/A",
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N9: Result validation and consistency judgment
    
    Args:
        structured_answer: Structured answer dictionary
        closed_inference_path: Closed inference path list
        answer_format_label: Answer format label
        question_options: Question options list
        answer_constraints: Answer constraints list
        question_type_label: Question type label
        hard_constraints: Hard constraints list
        structured_goal: Structured goal information
        core_keywords: Core keywords
        option_features: Option features
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_result_validation_prompt",
        domain=domain,
        question_type=question_type or question_type_label,
        core_domains=core_domains,
        structured_answer=structured_answer,
        closed_inference_path=closed_inference_path,
        answer_format_label=answer_format_label,
        question_options=question_options,
        answer_constraints=answer_constraints,
        question_type_label=question_type_label,
        hard_constraints=hard_constraints,
        structured_goal=structured_goal,
        core_keywords=core_keywords,
        option_features=option_features
    )


# ========== N10: Knowledge/Calculation Exception Handling ==========

def get_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any],
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N10: Knowledge/calculation exception handling
    
    Args:
        exception_type: Exception type
        exception_context: Exception context dictionary
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_exception_handling_prompt",
        domain=domain,
        question_type=question_type,
        core_domains=core_domains,
        exception_type=exception_type,
        exception_context=exception_context
    )


# ========== N11: Manual Intervention Trigger ==========

def get_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any],
    domain: Optional[str] = None,
    question_type: Optional[str] = None,
    core_domains: Optional[List[str]] = None
) -> str:
    """
    Prompt for N11: Manual intervention trigger
    
    Args:
        exception_type: Exception type
        intermediate_results: Intermediate results dictionary
        domain: Domain from raw_subject (optional)
        question_type: Question type (optional)
        core_domains: Already identified core domains list (optional)
    
    Returns:
        Prompt string
    """
    return _get_prompt_func(
        "get_manual_intervention_prompt",
        domain=domain,
        question_type=question_type,
        core_domains=core_domains,
        exception_type=exception_type,
        intermediate_results=intermediate_results
    )
