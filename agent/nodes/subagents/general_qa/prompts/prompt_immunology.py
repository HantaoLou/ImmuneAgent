"""
Immunology Domain-Specific Prompts
Optimized for immunology questions
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
DOMAIN_CONFIG = {
    "name": "Immunology",
    "priority_tools": [
        "query_tcr_mcpas",
        "query_celltype_marker",
        "query_ppi",
        "query_proteinatlas",
        "query_knowledge_graph"
    ],
    "tool_priority": {
        "query_tcr_mcpas": 1,
        "query_celltype_marker": 2,
        "query_ppi": 3,
        "query_proteinatlas": 4,
        "query_knowledge_graph": 5
    },
    "fallback_tools": {
        "query_tcr_mcpas": "query_celltype_marker",
    },
    "common_entities": [
        "T cell", "B cell", "NK cell", "macrophage",
        "TCR", "BCR", "MHC", "antigen", "antibody",
        "allelic exclusion", "positive selection", "negative selection"
    ],
    "calculation_focus": [],
    "validation_criteria": [
        "Must verify against immunological principles",
        "Check V(D)J recombination rules",
        "Verify cell development stages",
        "Check receptor-ligand interactions"
    ],
    "extraction_rules": """
**Immunology-Specific Extraction Rules:**
1. Immune Cell Identification: Extract immune cell types and subtypes
2. Receptor/Ligand Extraction: Identify receptors, ligands, and interactions
3. Immune Mechanism Keywords: Extract core immunological mechanisms
4. Immune System Notation: Preserve exact immune system notation
"""
}


def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0 prompt with immunology-specific enhancements"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    immunology_enhancements = f"""

**Immunology-Specific Extraction Rules:**

1. **Immune Cell Identification**: Extract immune cell types and subtypes explicitly:
   - Cell types: T cell, B cell, NK cell, macrophage, dendritic cell, neutrophil, etc.
   - T cell subtypes: CD4+ T cell, CD8+ T cell, naive T cell, memory T cell, regulatory T cell (Treg), etc.
   - B cell subtypes: naive B cell, memory B cell, plasma cell, etc.
   - Add to structured_subject.attribute: "cell_type: [type], subtype: [subtype]"

2. **Receptor/Ligand Extraction**: Identify receptors, ligands, and their interactions:
   - Receptors: TCR (T cell receptor), BCR (B cell receptor), MHC class I/II, Fc receptors, etc.
   - Ligands: antigens, cytokines, chemokines, antibodies
   - Interactions: antigen presentation, T cell activation, antibody binding
   - Add to structured_condition.key_features: "receptor: [receptor], ligand: [ligand], interaction: [type]"

3. **Immune Mechanism Keywords**: Extract core immunological mechanisms:
   - V(D)J recombination, allelic exclusion, allelic inclusion
   - Positive selection, negative selection
   - Antigen presentation, MHC restriction
   - Phagocytosis, opsonization, complement activation
   - Add to core_keywords: ["allelic exclusion", "positive selection", "MHC class I", ...] if present

4. **Immune System Notation**: Preserve exact immune system notation:
   - MHC alleles (e.g., "HLA-A*02:01")
   - TCR/BCR sequences (e.g., "V(D)J transcripts")
   - Cell markers (e.g., "CD4+", "CD8+", "CD19+")
   - DO NOT modify or normalize these notations

**Immunology-Specific Category Constraints:**
- For "vdj_bcr_tcr": ["Must follow V(D)J recombination rules", "Verify allelic inclusion/exclusion logic", "Check TCR/BCR chain pairing"]
- For "immune_cells": ["Must verify cell type-specific functions", "Check cell activation state", "Verify receptor expression"]
- For "mhc_binding": ["Must verify MHC-peptide binding rules", "Check MHC restriction", "Verify T cell recognition"]
- For "ProfessionalKnowledge-Immunology": ["Must verify against immunological principles", "Check immune cell development stages", "Verify receptor-ligand interactions"]
"""
    
    return base_prompt + immunology_enhancements


def get_question_decomposition_prompt(
    cleaned_text: str,
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None
) -> str:
    """N1 prompt with immunology-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )
    
    immunology_enhancements = """

**Immunology-Specific Decomposition Patterns:**

1. **V(D)J Recombination Questions**:
   - Sub-objective 1: Identify V(D)J recombination mechanism (heavy chain, light chain, alpha chain, beta chain)
   - Sub-objective 2: Determine allelic exclusion vs allelic inclusion patterns
   - Sub-objective 3: Analyze cell development checkpoints (positive/negative selection)

2. **Immune Cell Function Questions**:
   - Sub-objective 1: Identify cell type and activation state
   - Sub-objective 2: Determine cell-specific functions (e.g., antigen presentation, antibody production)
   - Sub-objective 3: Analyze cell-cell interactions and signaling pathways

3. **Antigen Recognition Questions**:
   - Sub-objective 1: Identify antigen type and structure
   - Sub-objective 2: Determine MHC presentation pathway (class I vs class II)
   - Sub-objective 3: Analyze T cell recognition and activation requirements

**Immunology-Specific Domain Identification:**
- Core domains should include: "T Cell Biology", "B Cell Biology", "Antigen Presentation", "V(D)J Recombination", "Immune Cell Development" as appropriate
- Use precise domain names (e.g., "T Cell Engineering, Allelic Exclusion" not just "Immunology")
"""
    
    return base_prompt + immunology_enhancements


def get_calculation_algorithm_recognition_prompt(cleaned_text: str, question_type_label: str) -> str:
    """N2: Calculation/algorithm recognition - Immunology domain"""
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
    synonyms: List[str] = None,
    cleaned_text: str = None  # ENHANCEMENT: Add for data analysis
) -> str:
    """N3 prompt with immunology-specific knowledge retrieval"""
    base_prompt = get_base_knowledge_retrieval_prompt(
        core_domains, calculation_type, algorithm_domain, research_objective,
        structured_conditions, key_entities, answer_format_label, question_type_label,
        structured_subject, structured_condition, structured_goal, synonyms,
        cleaned_text=cleaned_text  # ENHANCEMENT: Pass cleaned_text
    )
    
    immunology_enhancements = """

**Immunology-Specific Tool Usage:**

1. **Priority Tools for Immunology Questions**:
   - query_tcr_mcpas: Use for TCR sequences, antigen specificity, T cell receptor data
   - query_celltype_marker: Use for immune cell markers, cell type identification
   - query_ppi: Use for receptor-ligand interactions, immune signaling pathways
   - query_proteinatlas: Use for immune-related protein functions and locations
   - query_knowledge_graph: Use for general immune system relationships

2. **Tool Call Strategy**:
   - For V(D)J/TCR questions: Start with query_tcr_mcpas
   - For cell type questions: Start with query_celltype_marker
   - For receptor-ligand questions: Start with query_ppi, then query_proteinatlas
   - For general immune mechanisms: Use query_knowledge_graph

3. **Knowledge Retrieval Focus**:
   - Extract immune cell development stages
   - Identify receptor-ligand binding rules
   - Retrieve MHC restriction patterns
   - Find immune signaling pathway components
"""
    
    return base_prompt + immunology_enhancements


def get_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any]
) -> str:
    """N4: Calculation decomposition - Immunology domain"""
    return get_base_calculation_decomposition_prompt(cleaned_text, key_parameters, domain_knowledge)


def get_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any]
) -> str:
    """N5: Algorithm validation - Immunology domain"""
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
    """N6: Initial inference - Immunology domain"""
    base_prompt = get_base_initial_inference_prompt(
        cleaned_text, research_objective, key_entities, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal
    )
    
    immunology_enhancements = """

**Immunology-Specific Inference Rules:**

1. **Immune Mechanism Logic**:
   - Apply V(D)J recombination rules: heavy chain first, then light chain (B cells) or beta chain first, then alpha chain (T cells)
   - Apply allelic exclusion: most cells express single receptor, but some exceptions exist (allelic inclusion)
   - Apply selection checkpoints: positive selection (MHC binding), negative selection (self-reactivity)

2. **Cell Development Logic**:
   - T cell development: thymus → positive selection → negative selection → mature T cell
   - B cell development: bone marrow → allelic exclusion → mature B cell → activation → plasma cell
   - Memory formation: activated cells → memory cells (long-lived)

3. **Receptor-Ligand Logic**:
   - MHC class I: presents endogenous antigens to CD8+ T cells
   - MHC class II: presents exogenous antigens to CD4+ T cells
   - TCR recognition: requires both antigen and MHC (MHC restriction)
"""
    
    return base_prompt + immunology_enhancements


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
    """N7: Complete inference - Immunology domain"""
    base_prompt = get_base_complete_inference_prompt(
        cleaned_text, research_objective, initial_associations, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal,
        calculation_result
    )
    
    immunology_enhancements = """

**Immunology-Specific Complete Inference Rules:**

1. **Immune Mechanism Inference**:
   - Apply V(D)J recombination rules consistently
   - Verify allelic exclusion/inclusion patterns match cell development stages
   - Check selection checkpoints are correctly applied

2. **Cell Development Inference**:
   - Verify cell development pathway matches question context
   - Check receptor expression matches cell type
   - Ensure memory formation logic is correct

3. **Receptor-Ligand Inference**:
   - Verify MHC class matches antigen type
   - Check TCR recognition requirements are met
   - Ensure MHC restriction is correctly applied

**Immunology-Specific Option Analysis Strategy:**

For immunology multiple choice questions, apply these verification rules:

1. **Cell Development Options**:
   - Check if each option correctly describes cell development stages
   - Verify receptor expression matches cell type
   - Eliminate options with incorrect developmental sequence

2. **Mechanism Options**:
   - Verify V(D)J recombination steps are correctly described
   - Check selection mechanisms (positive/negative) are correctly applied
   - Eliminate options that violate immunological principles

3. **Receptor-Ligand Options**:
   - Verify MHC-antigen-TCR relationships are correct
   - Check co-stimulatory molecule requirements
   - Eliminate options with incorrect receptor-ligand pairing

4. **Cross-Verification**:
   - Compare options against immunology databases (McPAS-TCR, etc.)
   - Verify cell markers match known cell types
   - Check signaling pathways are correctly described
"""
    
    return base_prompt + immunology_enhancements


def get_answer_generation_prompt(
    core_conclusion: str,
    question_type_label: str,
    question_options: List[str] = None,
    calculation_result: Any = None,
    answer_format_label: str = None,
    answer_constraints: List[str] = None,
    structured_goal: Dict[str, Any] = None
) -> str:
    """N8: Answer generation - Immunology domain"""
    base_prompt = get_base_answer_generation_prompt(
        core_conclusion, question_type_label, question_options,
        calculation_result, answer_format_label, answer_constraints, structured_goal
    )
    
    immunology_enhancements = """

**Immunology-Specific Answer Format:**

1. **Cell Type Answers**: Include cell type and subtype (e.g., "CD4+ T cell", "naive B cell")
2. **Receptor Answers**: Include receptor name and chain (e.g., "TCR alpha chain", "BCR heavy chain")
3. **Mechanism Answers**: Include specific mechanism (e.g., "allelic exclusion", "positive selection")
4. **MHC Answers**: Include MHC class and allele if specified (e.g., "MHC class I, HLA-A*02:01")
"""
    
    return base_prompt + immunology_enhancements


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
    """N9: Result validation - Immunology domain"""
    return get_base_result_validation_prompt(
        structured_answer, closed_inference_path, answer_format_label,
        question_options, answer_constraints, question_type_label,
        hard_constraints, structured_goal, core_keywords, option_features
    )


def get_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any]
) -> str:
    """N10: Exception handling - Immunology domain"""
    return get_base_exception_handling_prompt(exception_type, exception_context)


def get_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any]
) -> str:
    """N11: Manual intervention - Immunology domain"""
    return get_base_manual_intervention_prompt(exception_type, intermediate_results)


def get_domain_tools() -> List[str]:
    """Return priority tools for immunology domain"""
    return DOMAIN_CONFIG["priority_tools"]


def get_domain_extraction_rules() -> str:
    """Return domain-specific extraction rules for cross-domain merging"""
    return DOMAIN_CONFIG["extraction_rules"]

