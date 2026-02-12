"""
Base Prompt Templates
Contains common prompt templates used across all domains
All prompts are in English as required.
"""

from typing import Dict, List, Any


def get_calculation_guide() -> str:
    """
    Universal calculation step guide for all domains
    All domains should use this template in calculation-related prompts
    """
    return """
**Calculation Step Guide (Universal - All Domains):**

1. **Parameter Extraction**: Extract all key parameters and confirm their validity
   - Identify all numerical values, units, and ranges
   - Verify parameter units are consistent (convert if necessary)
   - Check parameter ranges are valid (e.g., frequency ∈ [0,1], concentration > 0, probability ≤ 1)
   - Identify missing parameters and mark them explicitly in missing_core_entities

2. **Formula Selection**: Select appropriate formula based on domain and question type
   - For population genetics: Hardy-Weinberg equation (p²+2pq+q²=1), theta, pi, Fst formulas
   - For concentration: C1V1 = C2V2, dilution formulas
   - For statistical tests: Chi-square (χ² = Σ((O-E)²/E)), t-test, F-test formulas
   - Verify formula applicability conditions (e.g., HWE assumptions, sample size requirements)

3. **Step-by-Step Calculation**: Perform calculation showing each intermediate step
   - Show substitution of values into formula explicitly
   - Calculate intermediate results step by step
   - Track units throughout calculation (ensure unit consistency)
   - Show all mathematical operations clearly

4. **Result Verification**: Verify result conforms to domain constraints
   - Check result is within expected range:
     * Probabilities: ∈ [0,1]
     * Frequencies: ∈ [0,1]
     * Counts: ≥ 0 (integers)
     * Concentrations: > 0
   - Verify units are correct and match question requirements
   - Check result makes biological/clinical sense
   - Compare with critical/reference values if applicable (e.g., chi-square critical values)
   - Verify result satisfies all constraints from the question
"""


def get_calculation_validation_rules() -> Dict[str, List[str]]:
    """Domain-specific calculation validation rules"""
    return {
        "Genetics": [
            "Allele frequencies must sum to 1: p + q = 1",
            "Genotype frequencies must sum to 1: p² + 2pq + q² = 1",
            "Probabilities must be ∈ [0,1]",
            "Fst must be ∈ [0,1]"
        ],
        "Biochemistry": [
            "Concentrations must be > 0",
            "Molecular weights must be > 0",
            "Reaction rates must be ≥ 0"
        ],
        "Bioinformatics": [
            "Theta and pi must be ≥ 0",
            "Fst must be ∈ [0,1]",
            "Chi-square must be ≥ 0",
            "P-values must be ∈ [0,1]"
        ],
        "Clinical Medicine": [
            "Dosages must be > 0",
            "Dosing frequencies must be positive integers",
            "Drug concentrations must be within therapeutic range"
        ]
    }


# ========== N0: Input Preprocessing & Question Classification ==========

def get_base_input_preprocessing_prompt(user_input: str) -> str:
    """
    Base template for N0: Input Preprocessing
    Domain-specific modules will enhance this template
    """
    return f"""You are a biomedical question analysis expert. Your task is to preprocess the input question, classify its type, and extract structured three-dimensional information.

Input Question:
{user_input}

Please perform the following tasks:
1. Clean the input text: Remove redundant descriptions, normalize formatting, extract core question content
2. Classify question type: Determine if this is a Multiple Choice question, Text Matching question, Mechanism Explanation question, Numerical Calculation question, Logical Calculation question, or Professional Algorithm question
   - **CRITICAL**: If question asks for "minimum number" based on grouping/logical rules (e.g., "minimum [reagents] to distinguish [entities]"), classify as "Logical Calculation", NOT "Numerical Calculation"

**CRITICAL: Standardized Question Category Classification - MANDATORY**
You MUST classify the question into one of three standardized categories and assign category-specific constraints:

1. **Calculation-[subcategory]**: Questions requiring numerical computation, formula application, or statistical tests
   - Subcategories: ChiSquare, SurfaceArea, Concentration, MolecularWeight, etc.
   - Category-specific constraints: ["Must complete numerical derivation", "Conclusion based on calculated values, not subjective judgment", "Must verify calculation results against critical/reference values"]
   - Examples: Chi-square tests, lipid surface area calculations, concentration calculations

2. **ClinicalDecision-[subcategory]**: Questions requiring clinical guideline application, medication selection, or treatment recommendations
   - Subcategories: Hypertension, Diabetes, Cardiology, etc.
   - Category-specific constraints: ["Must follow latest clinical guidelines", "Exclude all contraindications", "Plan must comply with diagnostic and treatment principles", "Must verify drug compatibility"]
   - Examples: Hypertension medication recommendations, treatment protocol selection

3. **ProfessionalKnowledge-[subcategory]**: Questions requiring domain-specific knowledge matching, causal logic derivation, or conceptual understanding
   - Subcategories: LipidBiophysics, Genetics, Immunology, Biochemistry, etc.
   - Category-specific constraints: ["Conclusion must match authoritative textbooks/industry standards", "Logic chain based on causal facts", "Must verify knowledge fact consistency"]
   - Examples: Lipid membrane biophysics, genetic inheritance patterns, molecular mechanisms

**Output Format**: Add to JSON response:
- "question_category_standard": "Calculation-[subcategory]|ClinicalDecision-[subcategory]|ProfessionalKnowledge-[subcategory]" - MANDATORY: Standardized category with subcategory
- "category_specific_constraints": [list of constraints matching the category]
3. Extract answer options: If options are provided (e.g., A/B/C choices), extract them in order as plain text (no labels). Otherwise return an empty list
4. Determine expected answer format: Choose one of Single Choice, Multi-Select, Numeric, Short Text, Long Text, Sequence, Formula, List, Procedure, Code-Command
5. Assess data completeness: Evaluate if the question contains all necessary information (Complete/Partial Missing/Severe Missing)
   - **CRITICAL**: If question mentions "DNA sequence provided" but no actual sequence text is in the input, mark as "Incomplete-MissingCoreEntity" with missing_core_entities: ["DNA sequence text"]

**CRITICAL: Extract Structured Three-Dimensional Information - MANDATORY**
You MUST extract structured information in three dimensions. This is REQUIRED, not optional. Missing any dimension or sub-field will result in "data_completeness_label" being marked as "Severe Missing".

**YOU MUST OUTPUT ALL THREE DIMENSIONS IN THE JSON RESPONSE:**

**Dimension 1: Subject - REQUIRED**
- type: Subject type (e.g., individual/biological entity, cell, molecule, experimental system, substance) - describe the type, not domain-specific
- attribute: Subject attributes (e.g., type/fragment/state) - describe properties relevant to the question, not domain-specific characteristics
- Example: {{"type": "molecule", "attribute": "protein-protein interaction complex"}}

**Dimension 2: Condition - REQUIRED**
- type: Condition type (e.g., abnormal phenomenon, experimental treatment, observation result, calculation premise) - describe the type, not domain-specific
- key_features: Key features (quantitative/qualitative details) - ONLY include numbers that are relevant to the result (e.g., "4/95 low efficiency" is kept, but "95 samples" alone is meaningless and should be filtered out)
  - **CRITICAL: Implicit Contradiction Extraction**: You MUST identify and extract implicit contradictions or conflicts in the question text:
    * Pattern 1 (Capture mechanism mismatch): If question mentions a capture method targeting one region (e.g., 3' end) but the target is in a different region (e.g., 5' end) → Add to key_features: "implicit contradiction: capture method targets [region A], but target located at [region B]"
    * Pattern 2 (Experimental logic): If question involves distinguishing entities using reagents (e.g., antibodies, primers) → Add to key_features: "experimental logic: reagents can target shared/common regions or entity-specific regions"
    * Pattern 3 (Minimal set calculation): If question asks for "minimum number" to distinguish multiple entities → Add to key_features: "logical constraint: grouping-based minimal set calculation (not simple count)"
- hard_constraints: Hard constraints (MANDATORY if present) - List of conditions that MUST NOT be violated (e.g., "contraindicated XX", "exclude XX", "must not use XX", "prohibited XX"). These are absolute prohibitions that the answer must strictly avoid.
- Example: {{"type": "experimental treatment", "key_features": "SEC-MALS detected 300kDa and 210kDa peaks after mixing, implicit contradiction: capture method targets region A, but target located at region B", "hard_constraints": ["contraindicated drug A", "exclude plan B"]}}

**Dimension 3: Goal - REQUIRED**
- type: Goal type (e.g., cause analysis, mechanism derivation, conclusion judgment, calculation result, structure identification) - describe what needs to be determined
- constraint: Goal constraint (e.g., "most likely", "incorrect is", "must match XX") - capture implicit requirements in the question stem
- intent: Goal intent direction (MANDATORY) - the directional intent of the question, choose ONE:
  * "ask_defect" - asking for limitations, drawbacks, disadvantages, assumptions, simplifications, errors
  * "ask_cause" - asking for causes, reasons, why something happens
  * "ask_advantage" - asking for advantages, benefits, strengths
  * "ask_mechanism" - asking for mechanisms, how something works
  * "ask_example" - asking for examples, instances, cases
  * "ask_definition" - asking for definitions, what something is
  * "ask_comparison" - asking for comparisons, differences, similarities
  * "ask_procedure" - asking for procedures, steps, methods
  * "neutral" - neutral question without clear directional intent
- Example: {{"type": "conclusion judgment", "constraint": "most correct answer", "intent": "ask_defect"}}

**IMPORTANT RULES:**
1. Numbers should ONLY be placed in "condition.key_features", and only keep "core numbers relevant to the result"
2. DO NOT mix information from "subject, condition, goal" - each dimension must be structured separately
3. Each sub-field describes "information attributes" only, not domain-specific terms (e.g., "subject.type=cell", "subject.type=molecule", "subject.type=experimental system" are all valid)
4. If any dimension or sub-field is missing, mark "data_completeness_label" as "Severe Missing"
5. **YOU MUST INCLUDE structured_subject, structured_condition, and structured_goal in your JSON response - they are MANDATORY fields**

**CRITICAL: Information Validation & Precision Rules - MANDATORY**

**Hard Rule 1: Precise Value/Entity Extraction**
- structured_condition.key_features MUST strictly match numerical values and entity names from the question text.
- FORBIDDEN: rewriting, paraphrasing, or copying errors (e.g., if question says "55uM", you MUST write "55uM", NOT "50uM" or "55 μM").
- FORBIDDEN: changing entity names (e.g., if question says "G1-6", you MUST write "G1-6", NOT "G1 to G6" or "G1-G6 mutants").
- You MUST verify extracted values/entities against the original question text before outputting.

**Hard Rule 2: Missing Information Marking - ENHANCED**
- If the question is missing core entities (e.g., mutation types for G1-6, specific parameters, key experimental conditions), you MUST:
  - Set "data_completeness_label" to "Incomplete-MissingCoreEntity"
  - Add a field "missing_core_entities": ["entity1", "entity2", ...] listing what is missing
- **CRITICAL**: If question mentions "DNA sequence provided" or "sequence is shown below" but no actual sequence text appears in the input, you MUST:
  - Set "data_completeness_label" to "Incomplete-MissingCoreEntity"
  - Add "missing_core_entities": ["DNA sequence text"]
- DO NOT proceed with extraction if core entities are missing - mark them explicitly.

**Hard Rule 3: Error Interception**
- If you detect extraction errors (wrong numerical values, incorrect entity names, mismatched information), you MUST:
  - Set "data_completeness_label" to "Invalid-ExtractError"
  - Add a field "extraction_errors": ["error1", "error2", ...] describing the errors
  - Add a field "correction_hint": "hint for correction"
- This will terminate subsequent reasoning and return correction prompts.

Answer format guidance:
- Single Choice: one option is correct
- Multi-Select: multiple options are correct (e.g., "select all that apply")
- Numeric: a numerical value is required (with units if specified)
- Sequence: DNA/RNA/protein/oligo sequence output is required
- Formula: a mathematical expression is required
- List: multiple items (e.g., drugs, genes, steps) are required
- Procedure: stepwise method or protocol is required
- Code-Command: command line or code snippet is required

**CRITICAL: Core Keywords Extraction - MANDATORY**
- You MUST extract and mark core keywords that are essential for answering the question:
  * Pattern: Identify key concepts, methods, entities mentioned in the question
  * Add a field "core_keywords": ["keyword1", "keyword2", ...] listing these essential terms
  * These keywords will guide knowledge retrieval and option matching in downstream nodes

**CRITICAL: Option Feature Extraction - MANDATORY for Multiple Choice**
- For multiple-choice questions, you MUST extract core features from EACH option:
  * Pattern: For each option, identify its key entity/concept
  * Add a field "option_features": {{"A": "feature1", "B": "feature2", ...}} mapping each option to its core feature
  * This enables precise option matching in downstream inference

**CRITICAL: Retrieval Keyword Normalization - MANDATORY**
- You MUST extract and normalize retrieval keywords for knowledge retrieval:
  * Extract key biological terms from the question
  * Create a "synonyms" field: synonyms = [standard_term1, standard_term2, ...]
  * This ensures comprehensive knowledge retrieval across terminology variations

**CRITICAL: Tool Intent Marking - MANDATORY**
- You MUST determine if tool usage is required for this question:
  * Add a field "tool_intent": {{"query_go_term": "YES|NO", "query_knowledge_graph": "YES|NO"}}
  * Rules:
    - If question asks about "molecular function", "mechanism", "modification", "protein function" → query_go_term: "YES", query_knowledge_graph: "YES"
    - If question involves "epigenetics", "chromatin", "protein function", "molecular mechanism" → query_go_term: "YES", query_knowledge_graph: "YES"
  * This ensures mandatory tool calls for molecular function/mechanism questions

Output your response in JSON format:
{{
    "cleaned_text": "cleaned question text",
    "question_type_label": "Multiple Choice|Text Matching|Mechanism Explanation|Numerical Calculation|Logical Calculation|Professional Algorithm",
    "question_category_standard": "Calculation-[subcategory]|ClinicalDecision-[subcategory]|ProfessionalKnowledge-[subcategory]",
    "data_completeness_label": "Complete|Partial Missing|Severe Missing|Incomplete-MissingCoreEntity|Invalid-ExtractError",
    "core_keywords": ["keyword1", "keyword2", ...],
    "option_features": {{"A": "feature1", "B": "feature2", ...}},
    "synonyms": ["standard_term1", "standard_term2", ...],
    "tool_intent": {{"query_go_term": "YES|NO", "query_knowledge_graph": "YES|NO"}},
    "missing_core_entities": ["entity1", "entity2", ...] if data_completeness_label == "Incomplete-MissingCoreEntity",
    "extraction_errors": ["error1", "error2", ...] if data_completeness_label == "Invalid-ExtractError",
    "correction_hint": "hint for correction" if data_completeness_label == "Invalid-ExtractError",
    "key_constraints": ["constraint1", "constraint2", ...] if present,
    "negative_constraints": ["constraint1", "constraint2", ...] if present,
    "exclusive_constraints": ["constraint1", "constraint2", ...] if present,
    "strong_restrictions": ["restriction1", "restriction2", ...] if present,
    "question_options": ["option text 1", "option text 2", ...],
    "answer_format_label": "Single Choice|Multi-Select|Numeric|Short Text|Long Text|Sequence|Formula|List|Procedure|Code-Command",
    "structured_subject": {{
        "type": "subject type",
        "attribute": "subject attributes relevant to the question"
    }},
    "structured_condition": {{
        "type": "condition type",
        "key_features": "key features with only core numbers relevant to the result",
        "hard_constraints": ["constraint1", "constraint2", ...] if any absolute prohibitions exist
    }},
    "structured_goal": {{
        "type": "goal type",
        "constraint": "goal constraint",
        "intent": "ask_defect|ask_cause|ask_advantage|ask_mechanism|ask_example|ask_definition|ask_comparison|ask_procedure|neutral"
    }},
    "reasoning": "brief explanation of your classification and structured extraction"
}}
"""


# ========== N1: Question Decomposition & Domain Localization ==========

def get_base_question_decomposition_prompt(
    cleaned_text: str,
    question_type_label: str,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None,
    question_category_standard: str = None,
    category_specific_constraints: List[str] = None
) -> str:
    """Base template for N1: Question Decomposition"""
    # Build structured information context
    subject_info = ""
    condition_info = ""
    goal_info = ""
    
    if structured_subject:
        subject_type = structured_subject.get("type", "N/A")
        subject_attr = structured_subject.get("attribute", "N/A")
        subject_info = f"Subject: type={subject_type}, attribute={subject_attr}"
    
    if structured_condition:
        condition_type = structured_condition.get("type", "N/A")
        condition_features = structured_condition.get("key_features", "N/A")
        condition_info = f"Condition: type={condition_type}, key_features={condition_features}"
    
    if structured_goal:
        goal_type = structured_goal.get("type", "N/A")
        goal_constraint = structured_goal.get("constraint", "N/A")
        goal_info = f"Goal: type={goal_type}, constraint={goal_constraint}"
    
    structured_context = ""
    if subject_info or condition_info or goal_info:
        structured_context = f"""
**Structured Three-Dimensional Information from N0:**
{subject_info}
{condition_info}
{goal_info}
"""
    
    return f"""You are a biomedical question analysis expert. Your task is to decompose the question and identify its core domain, based on the structured three-dimensional information from N0.

Question (cleaned):
{cleaned_text}

Question Type: {question_type_label}
Question Category: {question_category_standard if question_category_standard else "Not classified"}
Category-Specific Constraints: {", ".join(category_specific_constraints) if category_specific_constraints else "None"}
{structured_context}

**CRITICAL REQUIREMENTS (based on structured information from N0):**

**Hard Rule 4: Constraint Anchoring**
- research_objective MUST place the question's strong constraints (e.g., "necessarily true", "most correct", "significant difference", "must be", "only") as the CORE judgment criterion in the FIRST sentence.
- Format: "Determine which option is [strong constraint] based on [subject.attribute] and [condition.key_features]"
- Example: If question asks "which is necessarily true", research_objective MUST start with "Determine which option is necessarily true..."

**Hard Rule 5: Missing Information Propagation**
- If N0's data_completeness_label is "Incomplete-MissingCoreEntity" or "Invalid-ExtractError", you MUST:
  - Set "research_objective" to "Cannot proceed: missing core entities or extraction errors"
  - DO NOT proceed to knowledge retrieval
  - Return immediately with error status

1. **Core Research Objective - ENHANCED**: 
   MUST be explicitly bound to N0's structured information using this fixed format (all-domain universal):
   → English format: "Combining the characteristics of [subject.attribute], based on the key features of [condition.key_features], complete [goal.type] (satisfying [goal.constraint])"
   - **CRITICAL: Bind Category-Specific Solution Steps**: You MUST include category-specific solution steps in research_objective, and these steps MUST match the question's actual solving paradigm, NOT reuse steps from other question types:
     * For Calculation questions: "Follow calculation steps to derive [calculated value], compare with critical value to determine [conclusion]"
     * For ClinicalDecision questions: "Follow clinical guideline steps to filter options, verify compatibility, output recommendations"
     * For ProfessionalKnowledge questions: "Associate core knowledge, derive causal logic, verify fact consistency, draw conclusion"
     * For LogicalCalculation questions: "Follow logical derivation steps, group entities, define distinguishing rules, derive minimal set"
   - **CRITICAL: Decompose into Executable Sub-Objectives**: For complex questions, you MUST break down research_objective into specific sub-objectives that match the question's actual solving paradigm
   → **If question has strong constraints (necessarily true/most correct), place them in the FIRST sentence as the core judgment criterion**
   → **FORBIDDEN**: Generic descriptions without category-specific steps

2. **Key Entities (key_entities)**: 
   MUST include N0's "subject core information + condition key features", do not omit or add irrelevant entities.

3. **Structured Conditions (structured_conditions)**: 
   For experimental questions, supplement "experimental operations" (put into condition.key_features).
   For clinical questions, supplement "diagnostic/treatment related" (put into condition.key_features).
   No need to add separate fields (keep universal).

Please perform the following tasks:
1. Extract structured conditions: Identify objective conditions, experimental settings, constraints mentioned in the question
2. Identify core domains: Determine which biomedical domain(s) this question belongs to (e.g., Genetics, Immunology, Cell Biology, Biochemistry, Clinical Medicine, etc.)
   - **CRITICAL: Domain Precision - ENHANCED**: You MUST identify domains at the TEST POINT level, NOT just broad categories:
     * Instead of "Genetics", use "Population Genetics, Fst Analysis, Hybrid Zone Dynamics"
     * Instead of "Immunology", use "T Cell Engineering, Antigen Presentation, Receptor Function"
     * Use precise domain names that reflect the specific test point

Output your response in JSON format:
{{
    "research_objective": "core research objective with category-specific steps",
    "key_entities": ["entity1", "entity2", ...],
    "core_domains": ["domain1", "domain2", ...],
    "structured_conditions": {{
        "experimental_conditions": "experimental settings if applicable",
        "clinical_conditions": "diagnostic/treatment conditions if applicable"
    }},
    "reasoning": "brief explanation of decomposition and domain identification"
}}
"""


# ========== N2: Calculation/Algorithm Requirement Recognition ==========

def get_base_calculation_algorithm_recognition_prompt(cleaned_text: str, question_type_label: str) -> str:
    """Base template for N2: Calculation/Algorithm Recognition"""
    return f"""You are a biomedical calculation and algorithm analysis expert. Your task is to identify calculation or algorithm requirements in the question.

Question (cleaned):
{cleaned_text}

Question Type: {question_type_label}

**CRITICAL: Calculation Type Classification - ENHANCED**
- **Numerical Calculation**: Pure numerical computation (e.g., "calculate concentration", "compute molecular weight")
- **Logical Calculation**: Questions asking for "minimum number" or "optimal set" based on grouping/logical rules, NOT simple counting
  * Key indicators: "minimum number of X to distinguish Y", "optimal combination", "minimal set"
  * Pattern: If question asks "minimum [reagents] to distinguish [multiple entities]" and entities can be grouped by shared characteristics → This is Logical Calculation (grouping-based), NOT Numerical Calculation
- **Algorithm**: Step-by-step procedure or computational method

Please perform the following tasks:
1. Determine calculation type: 
   - **Logical Calculation**: If question asks for "minimum number" or "optimal set" based on grouping/logical rules
   - **Numerical**: If this is a pure numerical computation
   - **Algorithm**: If it requires a professional algorithm
2. Extract key parameters: Identify key parameters, formulas, or algorithm-related clues mentioned in the question
   - **For Logical Calculation**: Extract grouping information (e.g., "entities in same family/category", "reagents can target shared vs entity-specific regions")

Output your response in JSON format:
{{
    "calculation_type_label": "Numerical|Logical Calculation|Algorithm|None",
    "key_parameters": {{
        "parameters": ["param1", "param2", ...],
        "formula_clues": ["clue1", "clue2", ...],
        "algorithm_name": "algorithm name if applicable"
    }},
    "reasoning": "brief explanation of your identification"
}}
"""


# ========== N3: Cross-Domain Knowledge Retrieval ==========

def get_base_knowledge_retrieval_prompt(
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
    """Base template for N3: Knowledge Retrieval"""
    domains_str = ", ".join(core_domains) if core_domains else "General"
    entities_str = ", ".join(key_entities) if key_entities else "N/A"
    synonyms_str = ", ".join(synonyms) if synonyms else "N/A"
    
    # Extract hard_constraints for display in prompt
    hard_constraints = []
    if structured_condition and isinstance(structured_condition, dict):
        hard_constraints = structured_condition.get("hard_constraints", [])
    hard_constraints_str = ", ".join(hard_constraints) if hard_constraints else "N/A"
    
    return f"""You are a biomedical knowledge retrieval expert. Your task is to retrieve relevant knowledge from multiple domains with PRECISE retrieval based on structured information.

Core Domains Identified: {domains_str}
Calculation Type: {calculation_type if calculation_type else "N/A"}
Algorithm Domain: {algorithm_domain if algorithm_domain else "N/A"}
Research Objective: {research_objective if research_objective else "N/A"}
Key Entities: {entities_str}
Synonyms/Normalized Terms: {synonyms_str}
Hard Constraints (MUST NOT violate): {hard_constraints_str}

**CRITICAL: Tool Usage Strategy**
- Use available tools to retrieve domain-specific knowledge
- Prioritize tools that match the identified core domains
- Retrieve information about key entities, relationships, and domain-specific concepts
- Verify retrieved knowledge against hard constraints

Please perform the following tasks:
1. Identify knowledge retrieval targets: Based on research_objective and key_entities, determine what knowledge needs to be retrieved
2. Use appropriate tools: Call relevant tools to retrieve domain-specific knowledge
3. Extract key facts: From retrieved knowledge, extract key facts relevant to answering the question
4. Verify constraints: Ensure retrieved knowledge does not violate hard constraints

**CRITICAL: Output Format - domain_knowledge_map Structure**
You MUST output knowledge in the following standardized format:

Output your response in JSON format:
{{
    "domain_knowledge_map": {{
        "domain1": {{
            "foundational_knowledge": ["foundational knowledge point 1", "foundational knowledge point 2", ...],
            "specialized_knowledge": ["specialized knowledge point 1", "specialized knowledge point 2", ...]
        }},
        "domain2": {{
            "foundational_knowledge": [...],
            "specialized_knowledge": [...]
        }}
    }},
    "key_facts": {{"fact_key1": "fact_value1", "fact_key2": "fact_value2", ...}},
    "knowledge_validity_label": "Valid|Invalid|Missing",
    "knowledge_unreliable": false,
    "knowledge_confidence": 0.0-1.0,
    "reasoning": "brief explanation of knowledge retrieval strategy and results"
}}

**Format Requirements:**
- foundational_knowledge: List of strings (basic domain knowledge points)
- specialized_knowledge: List of strings (domain-specific detailed knowledge points)
- Each knowledge point should be a clear, concise string describing the knowledge
- Group knowledge by domain (use core_domains identified earlier)
"""


# ========== N4: Calculation Step Decomposition & Formula Matching ==========

def get_base_calculation_decomposition_prompt(
    cleaned_text: str,
    key_parameters: Dict[str, Any],
    domain_knowledge: Dict[str, Any]
) -> str:
    """Base template for N4: Calculation Decomposition"""
    calculation_guide = get_calculation_guide()
    
    return f"""You are a biomedical calculation expert. Your task is to decompose calculation steps and match appropriate formulas.

Question:
{cleaned_text}

Key Parameters:
{key_parameters}

Domain Knowledge:
{domain_knowledge}

{calculation_guide}

Please perform the following tasks:
1. Decompose calculation steps: Break down the calculation logic following: Problem Objective → Core Variables → Formula Selection → Parameter Substitution → Unit Conversion → Result Verification
2. Match formulas: Identify appropriate biomedical formulas from your knowledge
3. Define parameters: Clearly define each parameter in the matched formula
4. Identify unit conversion rules: Determine necessary unit conversions
5. Verify result: Check result conforms to domain constraints

Output your response in JSON format:
{{
    "calculation_steps": [
        {{"step_number": 1, "step_description": "step 1", "step_type": "objective|variable|formula|substitution|conversion|verification"}},
        ...
    ],
    "matched_formula": {{
        "formula_name": "formula name",
        "formula_expression": "formula expression",
        "parameter_definitions": {{"param1": "definition", ...}},
        "applicability_scenario": "when this formula applies"
    }},
    "unit_conversion_rules": ["rule1", "rule2", ...],
    "formula_match_result": "Match Success|Match Failed",
    "reasoning": "brief explanation of decomposition and matching"
}}
"""


# ========== N5: Algorithm Parameter Extraction & Applicability Validation ==========

def get_base_algorithm_validation_prompt(
    cleaned_text: str,
    algorithm_name: str,
    domain_knowledge: Dict[str, Any]
) -> str:
    """Base template for N5: Algorithm Validation"""
    return f"""You are a biomedical algorithm expert. Your task is to extract algorithm parameters and validate applicability.

Question:
{cleaned_text}

Algorithm Name: {algorithm_name}

Algorithm Knowledge:
{domain_knowledge}

Please perform the following tasks:
1. Extract required parameters: Identify mandatory parameters
2. Extract optional parameters: Identify optional parameters
3. Validate applicability: Check if the problem scenario matches the algorithm's applicable scope
4. Suggest alternatives: If the algorithm is not applicable, suggest alternative algorithms

Output your response in JSON format:
{{
    "algorithm_parameters": {{
        "required_parameters": {{"param1": "value or description", ...}},
        "optional_parameters": {{"param1": "value or description", ...}}
    }},
    "applicability_result": "Applicable|Not Applicable",
    "applicability_reasoning": "explanation of applicability assessment",
    "alternative_algorithms": ["algorithm1", "algorithm2", ...] if not applicable,
    "reasoning": "brief explanation of parameter extraction and validation"
}}
"""


# ========== N6: Initial Association Inference ==========

def get_base_initial_inference_prompt(
    cleaned_text: str,
    research_objective: str,
    key_entities: List[str],
    retrieved_knowledge: Dict[str, Any],
    question_options: List[str] = None,
    structured_subject: Dict[str, Any] = None,
    structured_condition: Dict[str, Any] = None,
    structured_goal: Dict[str, Any] = None
) -> str:
    """Base template for N6: Initial Inference"""
    entities_str = ", ".join(key_entities) if key_entities else "N/A"
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(question_options)]) if question_options else "N/A"
    
    return f"""You are a biomedical inference expert. Your task is to perform initial association inference based on retrieved knowledge.

Question:
{cleaned_text}

Research Objective: {research_objective}
Key Entities: {entities_str}
Question Options:
{options_str}

Retrieved Knowledge:
{retrieved_knowledge}

Please perform the following tasks:
1. Associate entities: Link key entities with retrieved knowledge
2. Identify relationships: Find relationships between entities and concepts
3. Form initial associations: Create initial logical associations based on knowledge
4. Match with options: If applicable, match initial associations with question options

Output your response in JSON format:
{{
    "initial_associations": [
        {{"entity1": "entity1", "entity2": "entity2", "relationship": "relationship type", "evidence": "evidence from knowledge"}},
        ...
    ],
    "option_associations": {{"A": "association", "B": "association", ...}} if applicable,
    "reasoning": "brief explanation of initial inference"
}}
"""


# ========== N7: Complete Logical Inference ==========

def get_base_complete_inference_prompt(
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
    """Base template for N7: Complete Inference"""
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(question_options)]) if question_options else "N/A"
    
    return f"""You are a biomedical inference expert. Your task is to perform complete logical inference to derive the final conclusion.

Question:
{cleaned_text}

Research Objective: {research_objective}
Initial Associations:
{initial_associations}

Retrieved Knowledge:
{retrieved_knowledge}

Question Options:
{options_str}

Calculation Result (if applicable):
{calculation_result if calculation_result else "N/A"}

Please perform the following tasks:
1. Build inference chain: Construct a complete logical inference chain from initial associations to final conclusion
2. Apply domain logic: Apply domain-specific logical rules and principles
3. Verify consistency: Check consistency with retrieved knowledge and constraints
4. Derive conclusion: Derive the final conclusion based on complete inference

Output your response in JSON format:
{{
    "closed_inference_path": [
        {{"step": 1, "premise": "premise1", "inference": "inference1", "conclusion": "conclusion1"}},
        ...
    ],
    "core_conclusion": "final conclusion",
    "reasoning": "brief explanation of complete inference"
}}
"""


# ========== N8: Multi-Type Answer Generation ==========

def get_base_answer_generation_prompt(
    core_conclusion: str,
    question_type_label: str,
    question_options: List[str] = None,
    calculation_result: Any = None,
    answer_format_label: str = None,
    answer_constraints: List[str] = None,
    structured_goal: Dict[str, Any] = None
) -> str:
    """Base template for N8: Answer Generation"""
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(question_options)]) if question_options else "N/A"
    goal_intent = structured_goal.get("intent", "N/A") if structured_goal else "N/A"
    
    return f"""You are a biomedical answer generation expert. Your task is to generate answers in the appropriate format based on question type.

Core Conclusion:
{core_conclusion}

Question Type: {question_type_label}
Question Options (if applicable):
{options_str}

Calculation Result (if applicable):
{calculation_result if calculation_result else "N/A"}

Answer Format: {answer_format_label if answer_format_label else "N/A"}
Answer Constraints: {answer_constraints if answer_constraints else "N/A"}
Goal Intent: {goal_intent}

**CRITICAL: Answer Format Enforcement - MANDATORY**

Your final_answer MUST match answer_format_label EXACTLY:
- **Single Choice** → MUST output exactly one option label (e.g., "A", "B")
- **Multi-Select** → MUST output comma-separated option labels (e.g., "A, C, D")
- **Numeric** → MUST output numeric value with units
- **Sequence** → MUST output exact sequence
- **Formula** → MUST output mathematical expression
- **List** → MUST output list format (e.g., ["item1", "item2", ...])
- **Procedure** → MUST output step-by-step procedure
- **Code-Command** → MUST output code/command snippet

Please generate the answer according to question type:

For Multiple Choice Questions:
- Generate an option matching table, indicating which option matches the conclusion
- If answer format is Single Choice, choose exactly one option label
- If answer format is Multi-Select, list all correct option labels separated by commas

For Numerical Questions:
- Output calculation steps + final result + units
- Indicate precision control

For Text Matching Questions:
- Output True/False judgment + logical verification process

For Sequence or Formula Questions:
- Output the exact sequence or expression required

For List or Procedure Questions:
- Output the items or steps as a concise list

Output your response in JSON format:
{{
    "structured_answer": {{
        "answer_type": "Multiple Choice|Numerical|Text Matching|Sequence|Formula|List|Procedure|Code-Command",
        "answer_content": {{
            "option_matching_table": {{"A": "match|exclude", "B": "match|exclude", ...}} if multiple choice,
            "numerical_result": "result with units" if numerical,
            "text_answer": "answer text" if text matching,
            "sequence_result": "sequence output" if sequence,
            "formula_result": "expression output" if formula,
            "list_result": ["item1", "item2", ...] if list,
            "procedure_steps": ["step1", "step2", ...] if procedure,
            "code_command": "command or code snippet" if code-command
        }},
        "final_answer": "final answer to return"
    }},
    "reasoning": "brief explanation of answer generation"
}}
"""


# ========== N9: Result Validation & Consistency Judgment ==========

def get_base_result_validation_prompt(
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
    """Base template for N9: Result Validation"""
    hard_constraints_str = ", ".join(hard_constraints) if hard_constraints else "N/A"
    goal_type = structured_goal.get("type") if structured_goal else "N/A"
    goal_constraint = structured_goal.get("constraint") if structured_goal else "N/A"
    
    return f"""You are a biomedical answer validation expert. Your task is to validate the answer and assess its consistency with the inference pathway.

Structured Answer:
{structured_answer}

Closed Inference Path:
{closed_inference_path}

Question Type: {question_type_label if question_type_label else "N/A"}
Answer Format: {answer_format_label if answer_format_label else "N/A"}
Answer Constraints: {answer_constraints if answer_constraints else "N/A"}
Question Options: {question_options if question_options else "N/A"}
Hard Constraints: {hard_constraints_str}
Goal Type: {goal_type}
Goal Constraint: {goal_constraint}
Core Keywords: {core_keywords}
Option Features: {option_features}

**CRITICAL VALIDATION RULES - MANDATORY CHECKS:**

Please perform the following validation checks:
1. Format validation: Verify answer format matches answer_format_label
2. Constraint validation: Verify answer satisfies all constraints (hard_constraints, answer_constraints)
3. Consistency validation: Verify answer is consistent with closed_inference_path
4. Logical validation: Verify answer makes logical sense
5. Domain validation: Verify answer conforms to domain-specific rules

Output your response in JSON format:
{{
    "validation_result": "Valid|Invalid",
    "validation_errors": ["error1", "error2", ...] if invalid,
    "consistency_score": 0.0-1.0,
    "reasoning": "brief explanation of validation"
}}
"""


# ========== N10: Knowledge/Calculation Exception Handling ==========

def get_base_exception_handling_prompt(
    exception_type: str,
    exception_context: Dict[str, Any]
) -> str:
    """Base template for N10: Exception Handling"""
    return f"""You are a biomedical exception handling expert. Your task is to handle exceptions and find alternative solutions.

Exception Type: {exception_type}

Exception Context:
{exception_context}

Please perform the following tasks:
1. Analyze error: Understand the root cause of the error
2. Identify alternatives: Find alternative approaches or tools
3. Suggest solution: Propose a solution to handle the exception
4. Continue processing: If possible, continue with alternative approach

Output your response in JSON format:
{{
    "exception_type_label": "refined exception type",
    "error_analysis": "analysis of the error",
    "alternative_approaches": ["approach1", "approach2", ...],
    "solution_suggestion": "proposed solution",
    "can_continue": true|false,
    "reasoning": "brief explanation of exception handling"
}}
"""


# ========== N11: Manual Intervention Trigger ==========

def get_base_manual_intervention_prompt(
    exception_type: str,
    intermediate_results: Dict[str, Any]
) -> str:
    """Base template for N11: Manual Intervention"""
    return f"""You are a biomedical system coordinator. The system has encountered an exception that requires manual intervention.

Exception Type: {exception_type}

Intermediate Results:
{intermediate_results}

Please generate a manual intervention guide that includes:
1. Current status: What has been completed so far
2. Exception details: What went wrong and why
3. Next steps: What needs to be done manually
4. Intermediate result snapshot: Key intermediate results that can be used for continuation

Output your response in JSON format:
{{
    "manual_intervention_guide": "detailed guide for manual intervention",
    "intermediate_result_snapshot": {{
        "completed_steps": ["step1", "step2", ...],
        "key_findings": ["finding1", "finding2", ...],
        "exception_details": "detailed exception information",
        "suggested_next_steps": ["step1", "step2", ...]
    }},
    "reasoning": "brief explanation"
}}
"""

