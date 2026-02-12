#!/usr/bin/env python3
"""
Domain Prompt Template Generator
Automatically generates a domain-specific prompt module template
"""

import os
import sys
from pathlib import Path

# Template for domain prompt module
DOMAIN_PROMPT_TEMPLATE = '''"""
{domain_name} Domain-Specific Prompts
Optimized for {domain_name_lower} questions
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

# 领域特定配置
DOMAIN_CONFIG = {{
    "name": "{domain_name}",
    "priority_tools": [
        # TODO: Add priority tools for this domain
        # Example: "query_gwas_catalog", "query_genebass"
    ],
    "tool_priority": {{
        # TODO: Add tool priority mapping
        # Example: "query_gwas_catalog": 1
    }},
    "fallback_tools": {{
        # TODO: Add fallback tool mappings
        # Example: "query_gwas_catalog": "query_variant"
    }},
    "common_entities": [
        # TODO: Add common entities for this domain
        # Example: "allele", "genotype", "phenotype"
    ],
    "calculation_focus": [
        # TODO: Add calculation focus areas
        # Example: "Hardy-Weinberg equilibrium"
    ],
    "validation_criteria": [
        # TODO: Add validation criteria
        # Example: "Must verify against domain principles"
    ],
    "extraction_rules": """
**{domain_name}-Specific Extraction Rules:**
1. TODO: Add extraction rule 1
2. TODO: Add extraction rule 2
3. TODO: Add extraction rule 3
"""
}}


# ========== N0: Input Preprocessing & Question Classification ==========

def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0 prompt with {domain_name_lower}-specific enhancements"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    {domain_name_lower}_enhancements = f"""

**{domain_name}-Specific Extraction Rules:**

1. **TODO: Add extraction rule 1**:
   - TODO: Add details
   - Add to structured_subject.attribute: "TODO"

2. **TODO: Add extraction rule 2**:
   - TODO: Add details
   - Add to structured_condition.key_features: "TODO"

**{domain_name}-Specific Category Constraints:**
- For "ProfessionalKnowledge-{domain_name}": ["TODO: Add constraints"]
- For "Calculation-{domain_name}": ["TODO: Add constraints"]
"""
    
    return base_prompt + {domain_name_lower}_enhancements


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
    """N1 prompt with {domain_name_lower}-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )
    
    {domain_name_lower}_enhancements = """

**{domain_name}-Specific Decomposition Patterns:**

1. **TODO: Add decomposition pattern 1**:
   - Sub-objective 1: TODO
   - Sub-objective 2: TODO
   - Sub-objective 3: TODO

**{domain_name}-Specific Domain Identification:**
- Core domains should include: "TODO" as appropriate
- Use precise domain names (e.g., "TODO" not just "{domain_name}")
"""
    
    return base_prompt + {domain_name_lower}_enhancements


# ========== N2-N11: Use base templates or add domain-specific enhancements ==========

def get_calculation_algorithm_recognition_prompt(cleaned_text: str, question_type_label: str) -> str:
    """N2: Calculation/algorithm recognition - {domain_name} domain"""
    return get_base_calculation_algorithm_recognition_prompt(cleaned_text, question_type_label)


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
    """N3 prompt with {domain_name_lower}-specific knowledge retrieval"""
    base_prompt = get_base_knowledge_retrieval_prompt(
        core_domains, calculation_type, algorithm_domain, research_objective,
        structured_conditions, key_entities, answer_format_label, question_type_label,
        structured_subject, structured_condition, structured_goal, synonyms
    )
    
    {domain_name_lower}_enhancements = """

**{domain_name}-Specific Tool Usage:**

1. **Priority Tools for {domain_name} Questions**:
   - TODO: Add tool usage guidance

2. **Tool Call Strategy**:
   - TODO: Add strategy

3. **Knowledge Retrieval Focus**:
   - TODO: Add focus areas
"""
    
    return base_prompt + {domain_name_lower}_enhancements


def get_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any]
) -> str:
    """N4: Calculation decomposition - {domain_name} domain"""
    return get_base_calculation_decomposition_prompt(cleaned_text, key_parameters, domain_knowledge)


def get_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any]
) -> str:
    """N5: Algorithm validation - {domain_name} domain"""
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
    """N6: Initial inference - {domain_name} domain"""
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
    """N7: Complete inference - {domain_name} domain"""
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
    """N8: Answer generation - {domain_name} domain"""
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
    """N9: Result validation - {domain_name} domain"""
    return get_base_result_validation_prompt(
        structured_answer, closed_inference_path, answer_format_label,
        question_options, answer_constraints, question_type_label,
        hard_constraints, structured_goal, core_keywords, option_features
    )


def get_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any]
) -> str:
    """N10: Exception handling - {domain_name} domain"""
    return get_base_exception_handling_prompt(exception_type, exception_context)


def get_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any]
) -> str:
    """N11: Manual intervention - {domain_name} domain"""
    return get_base_manual_intervention_prompt(exception_type, intermediate_results)


# ========== Domain-specific functions ==========

def get_domain_tools() -> List[str]:
    """Return priority tools for {domain_name_lower} domain"""
    return DOMAIN_CONFIG["priority_tools"]


def get_domain_extraction_rules() -> str:
    """Return domain-specific extraction rules for cross-domain merging"""
    return DOMAIN_CONFIG["extraction_rules"]
'''


def generate_domain_prompt(domain_name: str, output_dir: str = None) -> str:
    """
    Generate a domain-specific prompt module template
    
    Args:
        domain_name: Name of the domain (e.g., "Biochemistry", "Microbiology")
        output_dir: Output directory (default: prompts/ folder)
    
    Returns:
        Path to generated file
    """
    # Normalize domain name
    domain_name_lower = domain_name.lower().replace(" ", "_")
    module_name = f"prompt-{domain_name_lower}.py"
    
    # Determine output directory
    if output_dir is None:
        # Get the prompts directory relative to this script
        script_dir = Path(__file__).parent
        prompts_dir = script_dir.parent / "prompts"
    else:
        prompts_dir = Path(output_dir)
    
    output_path = prompts_dir / module_name
    
    # Check if file already exists
    if output_path.exists():
        response = input(f"File {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print(f"Cancelled. File not overwritten.")
            return str(output_path)
    
    # Generate content
    content = DOMAIN_PROMPT_TEMPLATE.format(
        domain_name=domain_name,
        domain_name_lower=domain_name_lower
    )
    
    # Write file
    output_path.write_text(content, encoding='utf-8')
    print(f"✓ Generated domain prompt module: {output_path}")
    print(f"  Domain: {domain_name}")
    print(f"  Module: {module_name}")
    print(f"\nNext steps:")
    print(f"  1. Edit {module_name} to fill in TODO sections")
    print(f"  2. Add domain mapping in domain_mapper.py")
    print(f"  3. Add domain tools in tool_trigger.py")
    print(f"  4. Test with domain-specific questions")
    
    return str(output_path)


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python generate_domain_prompt.py <domain_name> [output_dir]")
        print("\nExample:")
        print("  python generate_domain_prompt.py Biochemistry")
        print("  python generate_domain_prompt.py Microbiology")
        sys.exit(1)
    
    domain_name = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_domain_prompt(domain_name, output_dir)


if __name__ == "__main__":
    main()

