"""
Genetics Domain-Specific Prompts
Optimized for genetics and genomics questions
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
    get_calculation_guide,
    get_calculation_validation_rules,
)

# 领域特定配置
DOMAIN_CONFIG = {
    "name": "Genetics",
    "priority_tools": [
        "query_gwas_catalog",
        "query_genebass",
        "query_variant",
        "query_omim",
        "query_disgenet",
        "query_gene_info"
    ],
    "tool_priority": {
        "query_gwas_catalog": 1,
        "query_genebass": 2,
        "query_variant": 3,
        "query_omim": 4,
        "query_disgenet": 5,
        "query_gene_info": 6
    },
    "fallback_tools": {
        "query_gwas_catalog": "query_variant",
        "query_genebass": "query_gwas_catalog",
    },
    "common_entities": [
        "allele", "genotype", "phenotype", "locus", "haplotype",
        "SNP", "variant", "mutation", "inheritance pattern"
    ],
    "calculation_focus": [
        "Hardy-Weinberg equilibrium",
        "genetic linkage",
        "recombination frequency",
        "Fst (fixation index)",
        "nucleotide diversity (pi)",
        "Watterson's estimator (theta)"
    ],
    "validation_criteria": [
        "Must verify against population genetics principles",
        "Check HWE assumptions",
        "Verify inheritance pattern consistency"
    ],
    "extraction_rules": """
**Genetics-Specific Extraction Rules:**
1. Inheritance Pattern Recognition: Identify and extract inheritance patterns explicitly
2. Genotype/Phenotype Extraction: Extract genotype and phenotype relationships
3. Population Genetics Parameters: Identify population genetics concepts
4. Genetic Variant Notation: Preserve exact variant notation
"""
}


# ========== N0: Input Preprocessing & Question Classification ==========

def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0 prompt with genetics-specific enhancements"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    genetics_enhancements = f"""

**Genetics-Specific Extraction Rules:**

1. **Inheritance Pattern Recognition**: Identify and extract inheritance patterns explicitly:
   - Autosomal dominant/recessive
   - X-linked dominant/recessive
   - Mitochondrial inheritance
   - Add to structured_condition.key_features: "inheritance pattern: [pattern]"

2. **Genotype/Phenotype Extraction**: Extract genotype and phenotype relationships:
   - Identify genotype notation (e.g., "0/0", "0/1", "1/1", "AA", "Aa", "aa")
   - Extract phenotype descriptions
   - Add to structured_subject.attribute: "genotype: [genotype], phenotype: [phenotype]"

3. **Population Genetics Parameters**: Identify population genetics concepts:
   - HWE (Hardy-Weinberg equilibrium) assumptions
   - Fst, pi, theta, genetic differentiation
   - Allele frequencies, genotype frequencies
   - Add to core_keywords: ["Fst", "HWE", "pi", "theta", ...] if present

4. **Genetic Variant Notation**: Preserve exact variant notation:
   - SNP IDs (e.g., "rs123456")
   - Genomic coordinates (e.g., "chr1:123456")
   - Variant notation (e.g., "c.123A>G")
   - DO NOT modify or normalize these notations

**Genetics-Specific Category Constraints:**
- For "ProfessionalKnowledge-Genetics": ["Must verify against Mendelian genetics principles", "Check population genetics assumptions", "Verify inheritance pattern logic"]
- For "Calculation-PopulationGenetics": ["Must apply HWE equations", "Verify sample size assumptions", "Check allele frequency calculations"]
"""
    
    return base_prompt + genetics_enhancements


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
    """N1 prompt with genetics-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )
    
    genetics_enhancements = """

**Genetics-Specific Decomposition Patterns:**

1. **Inheritance Pattern Questions**:
   - Sub-objective 1: Identify inheritance pattern from pedigree or description
   - Sub-objective 2: Apply Mendelian genetics principles
   - Sub-objective 3: Calculate genotype/phenotype probabilities

2. **Population Genetics Questions**:
   - Sub-objective 1: Extract population parameters (allele frequencies, sample size)
   - Sub-objective 2: Apply HWE or population genetics formulas
   - Sub-objective 3: Verify assumptions and interpret results

3. **Genetic Linkage Questions**:
   - Sub-objective 1: Identify linked loci and recombination events
   - Sub-objective 2: Calculate recombination frequency
   - Sub-objective 3: Determine genetic map distances

**Genetics-Specific Domain Identification:**
- Core domains should include: "Population Genetics", "Mendelian Genetics", "Genetic Linkage", "Molecular Genetics" as appropriate
- Use precise domain names (e.g., "Population Genetics, Fst Analysis" not just "Genetics")
"""
    
    return base_prompt + genetics_enhancements


# ========== N2: Calculation/Algorithm Requirement Recognition ==========

def get_calculation_algorithm_recognition_prompt(cleaned_text: str, question_type_label: str) -> str:
    """N2: Calculation/algorithm recognition - Genetics domain"""
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
    """N3 prompt with genetics-specific knowledge retrieval"""
    base_prompt = get_base_knowledge_retrieval_prompt(
        core_domains, calculation_type, algorithm_domain, research_objective,
        structured_conditions, key_entities, answer_format_label, question_type_label,
        structured_subject, structured_condition, structured_goal, synonyms
    )
    
    genetics_enhancements = """

**Genetics-Specific Tool Usage:**

1. **Priority Tools for Genetics Questions**:
   - query_gwas_catalog: Use for GWAS associations, genetic variants, SNP-disease associations
   - query_genebass: Use for gene-phenotype associations from rare variant burden tests
   - query_variant: Use for variant data, SNP positions, genomic coordinates
   - query_omim: Use for Mendelian inheritance disease data, single-gene disorders
   - query_disgenet: Use for disease-gene associations
   - query_gene_info: Use for gene information, genomic coordinates

2. **Tool Call Strategy**:
   - For inheritance pattern questions: Start with query_omim, query_disgenet
   - For population genetics questions: Start with query_gwas_catalog, query_genebass
   - For variant questions: Start with query_variant, query_gwas_catalog
   - For gene-phenotype questions: Start with query_genebass, query_omim

3. **Knowledge Retrieval Focus**:
   - Extract inheritance patterns and Mendelian genetics principles
   - Identify population genetics parameters and formulas
   - Retrieve variant annotation data
   - Find gene-disease associations
"""
    
    return base_prompt + genetics_enhancements


# ========== N4: Calculation Step Decomposition & Formula Matching ==========

def get_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any]
) -> str:
    """N4 prompt with genetics-specific calculation decomposition"""
    base_prompt = get_base_calculation_decomposition_prompt(cleaned_text, key_parameters, domain_knowledge)
    calculation_guide = get_calculation_guide()
    validation_rules = get_calculation_validation_rules()["Genetics"]
    
    genetics_calculation_rules = f"""

{calculation_guide}

**Genetics-Specific Calculation Rules:**

1. **Hardy-Weinberg Calculations**:
   - Verify HWE assumptions: random mating, no selection, no mutation, no migration, large population
   - Apply HWE equation: p² + 2pq + q² = 1 where p + q = 1
   - Check: p² + 2pq + q² must equal 1 (within rounding error)

2. **Population Genetics Parameters**:
   - Theta (Watterson's estimator): θ = S / Σ(1/i) where S is number of segregating sites, i from 1 to n-1
   - Pi (nucleotide diversity): π = Σ(2pq) summed over all polymorphic sites
   - Fst: Fst = (HT - HS) / HT where HT is total heterozygosity, HS is subpopulation heterozygosity
   - Verify: theta ≥ 0, pi ≥ 0, Fst ∈ [0,1]

3. **Genetic Probability Calculations**:
   - Inheritance probabilities must sum to 1 for all possible outcomes
   - Conditional probabilities: P(A|B) = P(A and B) / P(B)
   - Verify: all probabilities ∈ [0,1], sum of probabilities = 1

**Validation Rules:**
{chr(10).join(f"- {rule}" for rule in validation_rules)}
"""
    
    return base_prompt + genetics_calculation_rules


# ========== N5: Algorithm Parameter Extraction & Applicability Validation ==========

def get_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any]
) -> str:
    """N5: Algorithm validation - Genetics domain"""
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
    """N6: Initial inference - Genetics domain"""
    base_prompt = get_base_initial_inference_prompt(
        cleaned_text, research_objective, key_entities, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal
    )
    
    genetics_enhancements = """

**Genetics-Specific Inference Rules:**

1. **Mendelian Genetics Logic**:
   - Apply inheritance pattern rules: autosomal dominant/recessive, X-linked, mitochondrial
   - Calculate genotype probabilities based on inheritance pattern
   - Verify genotype-phenotype relationships

2. **Population Genetics Logic**:
   - Apply HWE principles: p² + 2pq + q² = 1
   - Calculate allele and genotype frequencies
   - Verify HWE assumptions are met

3. **Genetic Linkage Logic**:
   - Calculate recombination frequency from observed phenotypes
   - Determine genetic map distances
   - Verify linkage relationships
"""
    
    return base_prompt + genetics_enhancements


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
    """N7: Complete inference - Genetics domain"""
    base_prompt = get_base_complete_inference_prompt(
        cleaned_text, research_objective, initial_associations, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal,
        calculation_result
    )
    
    genetics_enhancements = """

**Genetics-Specific Complete Inference Rules:**

1. **Inheritance Pattern Inference**:
   - Apply Mendelian genetics principles consistently
   - Verify inheritance pattern matches observed phenotypes
   - Calculate probabilities for all possible genotypes

2. **Population Genetics Inference**:
   - Apply HWE equations correctly
   - Verify population parameters are consistent
   - Check calculation results against genetic principles

3. **Genetic Logic Chain**:
   - Build complete logic chain from genotype to phenotype
   - Verify each step follows genetic principles
   - Ensure final conclusion matches genetic laws
"""
    
    return base_prompt + genetics_enhancements


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
    """N8: Answer generation - Genetics domain"""
    base_prompt = get_base_answer_generation_prompt(
        core_conclusion, question_type_label, question_options,
        calculation_result, answer_format_label, answer_constraints, structured_goal
    )
    
    genetics_enhancements = """

**Genetics-Specific Answer Format:**

1. **Genotype Answers**: Include genotype notation (e.g., "0/0", "AA", "Aa")
2. **Probability Answers**: Include probability values with proper format (e.g., "0.25", "1/4")
3. **Population Genetics Answers**: Include parameter values with units if applicable (e.g., "Fst = 0.15", "pi = 0.001")
4. **Inheritance Pattern Answers**: Include inheritance pattern (e.g., "autosomal dominant", "X-linked recessive")
"""
    
    return base_prompt + genetics_enhancements


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
    """N9: Result validation - Genetics domain"""
    base_prompt = get_base_result_validation_prompt(
        structured_answer, closed_inference_path, answer_format_label,
        question_options, answer_constraints, question_type_label,
        hard_constraints, structured_goal, core_keywords, option_features
    )
    
    genetics_enhancements = """

**Genetics-Specific Validation Rules:**

1. **Genotype Validation**: Verify genotype notation is correct and consistent
2. **Probability Validation**: Verify probabilities sum to 1 and are in [0,1] range
3. **HWE Validation**: Verify HWE calculations satisfy p² + 2pq + q² = 1
4. **Inheritance Pattern Validation**: Verify inheritance pattern matches observed phenotypes
"""
    
    return base_prompt + genetics_enhancements


# ========== N10: Knowledge/Calculation Exception Handling ==========

def get_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any]
) -> str:
    """N10: Exception handling - Genetics domain"""
    return get_base_exception_handling_prompt(exception_type, exception_context)


# ========== N11: Manual Intervention Trigger ==========

def get_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any]
) -> str:
    """N11: Manual intervention - Genetics domain"""
    return get_base_manual_intervention_prompt(exception_type, intermediate_results)


# ========== Domain-specific functions ==========

def get_domain_tools() -> List[str]:
    """Return priority tools for genetics domain"""
    return DOMAIN_CONFIG["priority_tools"]


def get_domain_extraction_rules() -> str:
    """Return domain-specific extraction rules for cross-domain merging"""
    return DOMAIN_CONFIG["extraction_rules"]

