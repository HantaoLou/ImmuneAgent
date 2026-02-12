"""
Clinical Medicine Domain-Specific Prompts
Optimized for clinical medicine questions
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
    "name": "Clinical Medicine",
    "priority_tools": [
        "query_drug_interaction",
        "query_drug_for_disease",
        "query_disease_for_drug",
        "query_omim",
        "query_disgenet",
        "query_hpo_term"
    ],
    "tool_priority": {
        "query_drug_interaction": 1,
        "query_drug_for_disease": 2,
        "query_disease_for_drug": 3,
        "query_omim": 4,
        "query_disgenet": 5,
        "query_hpo_term": 6
    },
    "fallback_tools": {
        "query_drug_for_disease": "query_disease_for_drug",
    },
    "common_entities": [
        "hypertension", "diabetes", "medication", "drug",
        "treatment", "diagnosis", "patient", "guideline"
    ],
    "calculation_focus": [],
    "validation_criteria": [
        "Must verify against clinical guidelines",
        "Check drug contraindications",
        "Verify drug interactions",
        "Check evidence-based recommendations"
    ],
    "extraction_rules": """
**Clinical Medicine-Specific Extraction Rules:**
1. Patient Characteristics Extraction: Extract patient demographics and clinical features
2. Diagnostic Information Extraction: Identify diagnostic criteria and test results
3. Treatment Information Extraction: Extract treatment-related information
4. Clinical Decision Keywords: Extract core clinical decision-making concepts
"""
}


def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0 prompt with clinical medicine-specific enhancements"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    clinical_enhancements = f"""

**Clinical Medicine-Specific Extraction Rules:**

1. **Patient Characteristics Extraction**: Extract patient demographics and clinical features:
   - Age, gender, comorbidities
   - Symptoms, signs, clinical presentation
   - Medical history, family history
   - Add to structured_subject.attribute: "patient_profile: [age, gender, comorbidities]"

2. **Diagnostic Information Extraction**: Identify diagnostic criteria and test results:
   - Diagnostic criteria (e.g., JNC8 for hypertension, ADA for diabetes)
   - Laboratory values (e.g., blood pressure, glucose levels, lipid profiles)
   - Imaging findings, pathology results
   - Add to structured_condition.key_features: "diagnostic_criteria: [criteria], lab_values: [values]"

3. **Treatment Information Extraction**: Extract treatment-related information:
   - Current medications, drug classes
   - Treatment guidelines (e.g., JNC8, ACC/AHA)
   - Contraindications, drug interactions
   - Add to structured_condition.hard_constraints: ["contraindicated: [drug]", "drug_interaction: [drugs]"] if present

4. **Clinical Decision Keywords**: Extract core clinical decision-making concepts:
   - Treatment guidelines, evidence-based medicine
   - Drug selection, dosage, administration route
   - Monitoring parameters, follow-up care
   - Add to core_keywords: ["JNC8", "hypertension", "antihypertensive", ...] if present

**Clinical Medicine-Specific Category Constraints:**
- For "ClinicalDecision-Hypertension": ["Must follow JNC8 or ACC/AHA guidelines", "Exclude contraindications", "Verify drug compatibility", "Consider patient comorbidities"]
- For "ClinicalDecision-Diabetes": ["Must follow ADA guidelines", "Check glucose control targets", "Verify drug interactions"]
- For "ProfessionalKnowledge-ClinicalMedicine": ["Must verify against clinical guidelines", "Check evidence-based recommendations", "Verify drug safety profiles"]
"""
    
    return base_prompt + clinical_enhancements


def get_question_decomposition_prompt(
    cleaned_text: str,
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None
) -> str:
    """N1 prompt with clinical medicine-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(
        cleaned_text, question_type_label, structured_subject,
        structured_condition, structured_goal, question_category_standard,
        category_specific_constraints
    )
    
    clinical_enhancements = """

**Clinical Medicine-Specific Decomposition Patterns:**

1. **Diagnosis Questions**:
   - Sub-objective 1: Identify diagnostic criteria and required tests
   - Sub-objective 2: Apply diagnostic guidelines (e.g., JNC8, ADA)
   - Sub-objective 3: Determine diagnosis based on criteria and test results

2. **Treatment Selection Questions**:
   - Sub-objective 1: Identify applicable treatment guidelines (e.g., JNC8 for hypertension)
   - Sub-objective 2: Filter treatment options based on contraindications and patient characteristics
   - Sub-objective 3: Select optimal treatment(s) from approved options

3. **Drug Interaction Questions**:
   - Sub-objective 1: Identify all medications and drug classes
   - Sub-objective 2: Check for drug-drug interactions
   - Sub-objective 3: Determine safe medication combinations

4. **Clinical Decision Questions**:
   - Sub-objective 1: Assess patient characteristics and comorbidities
   - Sub-objective 2: Apply clinical guidelines and evidence-based recommendations
   - Sub-objective 3: Make clinical decision considering safety and efficacy

**Clinical Medicine-Specific Domain Identification:**
- Core domains should include: "Hypertension Management", "Diabetes Care", "Cardiology", "Pharmacology" as appropriate
- Use precise domain names (e.g., "Hypertension Management, JNC8 Guidelines" not just "Clinical Medicine")
"""
    
    return base_prompt + clinical_enhancements


def get_calculation_algorithm_recognition_prompt(cleaned_text: str, question_type_label: str) -> str:
    """N2: Calculation/algorithm recognition - Clinical Medicine domain"""
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
    """N3 prompt with clinical medicine-specific knowledge retrieval"""
    base_prompt = get_base_knowledge_retrieval_prompt(
        core_domains, calculation_type, algorithm_domain, research_objective,
        structured_conditions, key_entities, answer_format_label, question_type_label,
        structured_subject, structured_condition, structured_goal, synonyms
    )
    
    clinical_enhancements = """

**Clinical Medicine-Specific Tool Usage:**

1. **Priority Tools for Clinical Medicine Questions**:
   - query_drug_interaction: Use for drug-drug interactions, medication safety
   - query_drug_for_disease: Use for finding drugs for specific diseases
   - query_disease_for_drug: Use for finding diseases treatable by specific drugs
   - query_omim: Use for genetic diseases, inheritance patterns
   - query_disgenet: Use for disease-gene associations
   - query_hpo_term: Use for phenotype queries, clinical observations

2. **Tool Call Strategy**:
   - For drug selection questions: Start with query_drug_for_disease, then query_drug_interaction
   - For drug interaction questions: Start with query_drug_interaction
   - For genetic disease questions: Start with query_omim, query_disgenet
   - For phenotype questions: Start with query_hpo_term

3. **Knowledge Retrieval Focus**:
   - Extract treatment guidelines and recommendations
   - Identify drug contraindications and interactions
   - Retrieve disease-gene associations
   - Find evidence-based treatment options
"""
    
    return base_prompt + clinical_enhancements


def get_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any]
) -> str:
    """N4: Calculation decomposition - Clinical Medicine domain"""
    return get_base_calculation_decomposition_prompt(cleaned_text, key_parameters, domain_knowledge)


def get_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any]
) -> str:
    """N5: Algorithm validation - Clinical Medicine domain"""
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
    """N6: Initial inference - Clinical Medicine domain"""
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
    """N7 prompt with clinical medicine-specific inference"""
    base_prompt = get_base_complete_inference_prompt(
        cleaned_text, research_objective, initial_associations, retrieved_knowledge,
        question_options, structured_subject, structured_condition, structured_goal,
        calculation_result
    )
    
    clinical_enhancements = """

**Clinical Medicine-Specific Inference Rules:**

1. **Clinical Decision Logic**:
   - Apply treatment guidelines step-by-step (e.g., JNC8: lifestyle → first-line drugs → combination therapy)
   - Exclude all contraindications before selecting treatments
   - Consider patient comorbidities and drug interactions
   - Verify drug compatibility and safety profiles

2. **Drug Selection Logic**:
   - First-line drugs: ACE inhibitors, ARBs, thiazide diuretics (hypertension)
   - Avoid contraindicated drugs (e.g., ACE inhibitors in pregnancy)
   - Consider drug interactions (e.g., avoid combining certain antihypertensives)
   - Verify dosage and administration route

3. **Diagnostic Logic**:
   - Apply diagnostic criteria strictly (e.g., BP ≥140/90 for hypertension)
   - Consider differential diagnoses
   - Verify test results against reference ranges
"""
    
    return base_prompt + clinical_enhancements


def get_answer_generation_prompt(
    core_conclusion: str,
    question_type_label: str,
    question_options: List[str] = None,
    calculation_result: Any = None,
    answer_format_label: str = None,
    answer_constraints: List[str] = None,
    structured_goal: Dict[str, Any] = None
) -> str:
    """N8 prompt with clinical medicine-specific answer generation"""
    base_prompt = get_base_answer_generation_prompt(
        core_conclusion, question_type_label, question_options,
        calculation_result, answer_format_label, answer_constraints, structured_goal
    )
    
    clinical_enhancements = """

**Clinical Medicine-Specific Answer Format:**

1. **Drug Answers**: Include drug name, class, and dosage if specified (e.g., "Lisinopril (ACE inhibitor), 10mg daily")
2. **Treatment Answers**: Include treatment plan with rationale (e.g., "First-line: ACE inhibitor or ARB, per JNC8 guidelines")
3. **Diagnosis Answers**: Include diagnostic criteria met (e.g., "Hypertension: BP ≥140/90, per JNC8 criteria")
4. **Guideline Answers**: Reference specific guidelines (e.g., "Per JNC8 guidelines, recommend...")
"""
    
    return base_prompt + clinical_enhancements


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
    """N9: Result validation - Clinical Medicine domain"""
    return get_base_result_validation_prompt(
        structured_answer, closed_inference_path, answer_format_label,
        question_options, answer_constraints, question_type_label,
        hard_constraints, structured_goal, core_keywords, option_features
    )


def get_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any]
) -> str:
    """N10: Exception handling - Clinical Medicine domain"""
    return get_base_exception_handling_prompt(exception_type, exception_context)


def get_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any]
) -> str:
    """N11: Manual intervention - Clinical Medicine domain"""
    return get_base_manual_intervention_prompt(exception_type, intermediate_results)


def get_domain_tools() -> List[str]:
    """Return priority tools for clinical medicine domain"""
    return DOMAIN_CONFIG["priority_tools"]


def get_domain_extraction_rules() -> str:
    """Return domain-specific extraction rules for cross-domain merging"""
    return DOMAIN_CONFIG["extraction_rules"]

