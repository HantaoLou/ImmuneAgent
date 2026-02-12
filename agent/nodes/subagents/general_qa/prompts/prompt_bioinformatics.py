"""
Bioinformatics Domain-Specific Prompts
Optimized for bioinformatics questions
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
    "name": "Bioinformatics",
    "priority_tools": [
        "query_variant",
        "query_gwas_catalog",
        "query_genebass",
        "query_knowledge_graph",
        "query_gene_info",
        "query_go_term"
    ],
    "tool_priority": {
        "query_variant": 1,
        "query_gwas_catalog": 2,
        "query_genebass": 3,
        "query_knowledge_graph": 4,
        "query_gene_info": 5,
        "query_go_term": 6
    },
    "fallback_tools": {
        "query_variant": "query_gwas_catalog",
        "query_genebass": "query_gwas_catalog",
    },
    "common_entities": [
        "theta", "pi", "Fst", "Watterson",
        "variant calling", "phasing", "VCF", "FASTA",
        "chi-square", "statistical test"
    ],
    "calculation_focus": [
        "Watterson's estimator (theta)",
        "nucleotide diversity (pi)",
        "Fst (fixation index)",
        "Chi-square test",
        "statistical tests"
    ],
    "validation_criteria": [
        "Must verify computational assumptions",
        "Check data quality requirements",
        "Verify statistical test procedures",
        "Check population genetics formulas"
    ],
    "extraction_rules": """
**Bioinformatics-Specific Extraction Rules:**
1. Algorithm/Method Identification: Extract computational methods and algorithms
2. Data Format Extraction: Identify data formats and structures
3. Computational Parameters: Extract computational parameters and constraints
4. Bioinformatics Keywords: Extract core bioinformatics concepts
"""
}


def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0 prompt with bioinformatics-specific enhancements"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    bioinformatics_enhancements = f"""

**Bioinformatics-Specific Extraction Rules:**

1. **Algorithm/Method Identification**: Extract computational methods and algorithms:
   - Statistical methods: Chi-square test, t-test, F-test, permutation test
   - Population genetics: Watterson's estimator (theta), nucleotide diversity (pi), Fst
   - Sequence analysis: alignment, variant calling, phasing
   - Add to structured_condition.key_features: "method: [method], parameters: [params]"

2. **Data Format Extraction**: Identify data formats and structures:
   - File formats: VCF, FASTA, BAM, SAM
   - Data types: phased samples, variant calls, sequence data
   - Data quality: quality scores, missing data patterns
   - Add to structured_condition.key_features: "data_format: [format], data_quality: [quality]"

3. **Computational Parameters**: Extract computational parameters and constraints:
   - Sample size, variant counts, sequence lengths
   - Quality thresholds, filtering criteria
   - Computational assumptions (e.g., HWE, no missing variants)
   - Add to structured_condition.key_features: "sample_size: [N], quality_threshold: [threshold]"

4. **Bioinformatics Keywords**: Extract core bioinformatics concepts:
   - Population genetics parameters (theta, pi, Fst)
   - Statistical tests (chi-square, permutation)
   - Data processing steps (filtering, imputation, phasing)
   - Add to core_keywords: ["theta", "pi", "Fst", "chi-square", ...] if present

**Bioinformatics-Specific Category Constraints:**
- For "Calculation-PopulationGenetics": ["Must apply population genetics formulas correctly", "Verify computational assumptions", "Check data quality requirements"]
- For "ProfessionalAlgorithm-Bioinformatics": ["Must follow algorithm specifications", "Verify parameter validity", "Check computational constraints"]
"""
    
    return base_prompt + bioinformatics_enhancements


def get_question_decomposition_prompt(
    cleaned_text: str,
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None
) -> str:
    """N1 prompt with bioinformatics-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )
    
    bioinformatics_enhancements = """

**Bioinformatics-Specific Decomposition Patterns:**

1. **Population Genetics Calculation Questions**:
   - Sub-objective 1: Extract population parameters (sample size, variant counts, allele frequencies)
   - Sub-objective 2: Identify computational method (theta, pi, Fst calculation)
   - Sub-objective 3: Apply formula with correct parameters and verify assumptions

2. **Statistical Test Questions**:
   - Sub-objective 1: Identify test type (chi-square, t-test, permutation)
   - Sub-objective 2: Extract test parameters (observed/expected values, degrees of freedom)
   - Sub-objective 3: Perform test and interpret results

3. **Data Processing Questions**:
   - Sub-objective 1: Identify data processing steps (filtering, imputation, phasing)
   - Sub-objective 2: Determine impact of processing on downstream analysis
   - Sub-objective 3: Evaluate bias or error introduced by processing

**Bioinformatics-Specific Domain Identification:**
- Core domains should include: "Population Genetics", "Statistical Analysis", "Sequence Analysis", "Variant Analysis" as appropriate
- Use precise domain names (e.g., "Population Genetics, Theta Calculation" not just "Bioinformatics")
"""
    
    return base_prompt + bioinformatics_enhancements


def get_calculation_algorithm_recognition_prompt(cleaned_text: str, question_type_label: str) -> str:
    """N2: Calculation/algorithm recognition - Bioinformatics domain"""
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
    """N3 prompt with bioinformatics-specific knowledge retrieval"""
    base_prompt = get_base_knowledge_retrieval_prompt(
        core_domains, calculation_type, algorithm_domain, research_objective,
        structured_conditions, key_entities, answer_format_label, question_type_label,
        structured_subject, structured_condition, structured_goal, synonyms
    )
    
    bioinformatics_enhancements = """

**Bioinformatics-Specific Tool Usage:**

1. **Priority Tools for Bioinformatics Questions**:
   - query_variant: Use for variant data, SNP positions, genomic coordinates
   - query_gwas_catalog: Use for GWAS associations, genetic variants
   - query_genebass: Use for gene-phenotype associations, rare variants
   - query_knowledge_graph: Use for general bioinformatics relationships
   - query_gene_info: Use for gene information, genomic coordinates
   - query_go_term: Use for functional annotation, biological processes

2. **Tool Call Strategy**:
   - For variant questions: Start with query_variant, query_gwas_catalog
   - For gene-phenotype questions: Start with query_genebass
   - For functional annotation: Start with query_go_term
   - For general relationships: Use query_knowledge_graph

3. **Knowledge Retrieval Focus**:
   - Extract population genetics formulas and methods
   - Identify statistical test procedures
   - Retrieve variant annotation data
   - Find computational algorithm specifications
"""
    
    return base_prompt + bioinformatics_enhancements


def get_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any]
) -> str:
    """N4 prompt with bioinformatics-specific calculation decomposition"""
    base_prompt = get_base_calculation_decomposition_prompt(cleaned_text, key_parameters, domain_knowledge)
    calculation_guide = get_calculation_guide()
    validation_rules = get_calculation_validation_rules()["Bioinformatics"]
    
    bioinformatics_calculation_rules = f"""

{calculation_guide}

**Bioinformatics-Specific Calculation Rules:**

1. **Population Genetics Calculations**:
   - Theta (Watterson's estimator): θ = S / Σ(1/i) where S is number of segregating sites, i from 1 to n-1
   - Pi (nucleotide diversity): π = Σ(2pq) for all polymorphic sites
   - Fst: Fst = (HT - HS) / HT where HT is total heterozygosity, HS is subpopulation heterozygosity
   - Verify: theta and pi should be positive, Fst ∈ [0,1]

2. **Statistical Test Calculations**:
   - Chi-square: χ² = Σ((O-E)²/E) where O is observed, E is expected
   - Degrees of freedom: df = (rows-1) × (columns-1)
   - Verify: chi-square ≥ 0, p-value ∈ [0,1]

3. **Data Quality Verification**:
   - Check sample size is sufficient for statistical power
   - Verify data quality thresholds are met
   - Confirm computational assumptions (HWE, no missing data) are satisfied

**Validation Rules:**
{chr(10).join(f"- {rule}" for rule in validation_rules)}
"""
    
    return base_prompt + bioinformatics_calculation_rules


def get_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any]
) -> str:
    """N5: Algorithm validation - Bioinformatics domain"""
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
    """N6: Initial inference - Bioinformatics domain"""
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
    """N7: Complete inference - Bioinformatics domain"""
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
    """N8: Answer generation - Bioinformatics domain"""
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
    """N9: Result validation - Bioinformatics domain"""
    return get_base_result_validation_prompt(
        structured_answer, closed_inference_path, answer_format_label,
        question_options, answer_constraints, question_type_label,
        hard_constraints, structured_goal, core_keywords, option_features
    )


def get_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any]
) -> str:
    """N10: Exception handling - Bioinformatics domain"""
    return get_base_exception_handling_prompt(exception_type, exception_context)


def get_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any]
) -> str:
    """N11: Manual intervention - Bioinformatics domain"""
    return get_base_manual_intervention_prompt(exception_type, intermediate_results)


def get_domain_tools() -> List[str]:
    """Return priority tools for bioinformatics domain"""
    return DOMAIN_CONFIG["priority_tools"]


def get_domain_extraction_rules() -> str:
    """Return domain-specific extraction rules for cross-domain merging"""
    return DOMAIN_CONFIG["extraction_rules"]

