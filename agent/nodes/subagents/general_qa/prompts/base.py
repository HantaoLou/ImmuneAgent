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
   - **CRITICAL: True/False Question Detection**: If the question options are ONLY "True" and "False" (or "true" and "false"), you MUST classify it as "Text Matching" (NOT "Multiple Choice"). This is a True/False judgment question, not a multiple choice question. The answer format should be "Short Text" and the answer must be "True" or "False", not option letters like "A" or "B".

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

**CRITICAL: Experimental Data Structured Extraction - ENHANCEMENT**
If the question contains experimental data (numbers, conditions, comparisons, tables, results), you MUST extract structured experimental data:

**Experimental Data Detection Patterns:**
- Numerical data with units (e.g., "3:8 ratio", "55uM", "300kDa")
- Experimental groups/conditions (e.g., "CA biotypes", "MA biotypes", "treatment A vs B")
- Performance/outcome measures (e.g., "did well", "decreased activity", "higher levels")
- Comparisons (e.g., "A performed better than B", "X increased while Y decreased")

**Structured Experimental Data Format - Add to JSON response:**
- "experimental_data": {{
    "has_experimental_data": true|false,
    "groups": [
        {{"name": "group1", "description": "description of group1"}},
        {{"name": "group2", "description": "description of group2"}}
    ],
    "conditions": [
        {{"name": "condition1", "values": "values or description"}},
        ...
    ],
    "outcomes": [
        {{"group": "group_name", "condition": "condition_name", "outcome": "result description"}},
        ...
    ],
    "comparisons": [
        {{"type": "better/worse/higher/lower", "entity1": "name", "entity2": "name", "context": "context"}},
        ...
    ],
    "key_numbers": [
        {{"value": "3:8", "context": "sucrose:raffinose ratio for CA biotypes"}},
        ...
    ]
  }}

**IMPORTANT**: This experimental data extraction is MANDATORY for data-rich questions. The extracted data will be used for downstream reasoning instead of relying solely on general knowledge.

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
- Single Choice: one option is correct (for multiple choice questions with 3+ options)
- Multi-Select: multiple options are correct (e.g., "select all that apply")
- Short Text: for True/False questions (answer must be "True" or "False") or short text matching
- Numeric: a numerical value is required (with units if specified)
- Sequence: DNA/RNA/protein/oligo sequence output is required
- Formula: a mathematical expression is required
- List: multiple items (e.g., drugs, genes, steps) are required
- Procedure: stepwise method or protocol is required
- Code-Command: command line or code snippet is required

**CRITICAL: Entity Identification Question Detection - MANDATORY**
When determining answer_format_label, you MUST distinguish between:
1. **Entity Identification Questions**: Questions that ask to "identify/determine what [entity]" from given data (e.g., "What protein does this sequence represent?", "Identify the molecule from this spectrum", "Which gene is encoded by this DNA sequence?")
   - For these questions, the answer should be the **ENTITY NAME** (e.g., "rhodopsin", "BRCA1", "glucose"), NOT the data format
   - **MUST use "Short Text" as answer_format_label**, even if the question mentions sequences, structures, or other data formats
   - Examples:
     * "What protein does this amino acid sequence represent?" → answer_format_label: "Short Text" (answer: protein name)
     * "Identify the gene from this DNA sequence" → answer_format_label: "Short Text" (answer: gene name)
     * "Which molecule is this structure?" → answer_format_label: "Short Text" (answer: molecule name)

2. **Data Format Questions**: Questions that ask to "provide/give/show the [data format]" (e.g., "What is the sequence?", "Provide the DNA sequence", "Give me the structure")
   - For these questions, use the appropriate format label (e.g., "Sequence", "Procedure")
   - Examples:
     * "What is the amino acid sequence?" → answer_format_label: "Sequence" (answer: the sequence itself)
     * "Provide the DNA sequence" → answer_format_label: "Sequence" (answer: the sequence itself)

**Key Distinction**: 
- If question asks "What [entity] is this [data]?" or "Identify [entity] from [data]" → Use "Short Text" (answer is entity name)
- If question asks "What is the [data]?" or "Provide the [data]" → Use appropriate format label (answer is the data itself)

**CRITICAL: True/False Question Format**
- If question options are ONLY "True" and "False", use "Short Text" as answer_format_label
- The answer must be the word "True" or "False", NOT option letters (A, B, etc.)

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
    # CRITICAL: Handle both dict and list formats for structured fields
    subject_info = ""
    condition_info = ""
    goal_info = ""
    
    if structured_subject:
        if isinstance(structured_subject, dict):
            subject_type = structured_subject.get("type", "N/A")
            subject_attr = structured_subject.get("attribute", "N/A")
            subject_info = f"Subject: type={subject_type}, attribute={subject_attr}"
        else:
            subject_info = f"Subject: {structured_subject}"
    
    if structured_condition:
        if isinstance(structured_condition, dict):
            condition_type = structured_condition.get("type", "N/A")
            condition_features = structured_condition.get("key_features", "N/A")
            condition_info = f"Condition: type={condition_type}, key_features={condition_features}"
        else:
            condition_info = f"Condition: {structured_condition}"
    
    if structured_goal:
        if isinstance(structured_goal, dict):
            goal_type = structured_goal.get("type", "N/A")
            goal_constraint = structured_goal.get("constraint", "N/A")
            goal_info = f"Goal: type={goal_type}, constraint={goal_constraint}"
        else:
            goal_info = f"Goal: {structured_goal}"
    
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

2. **Key Entities (key_entities) - CRITICAL EXTRACTION RULES**: 
   MUST include N0's "subject core information + condition key features", do not omit or add irrelevant entities.
   
   **CRITICAL: Entity Extraction Rules (实体提取规则) - MANDATORY:**
   - **ONLY extract entities EXPLICITLY mentioned in the question text**
   - **DO NOT extract entities from:**
     * Examples or sample data in the question
     * Context or background information
     * Unrelated parts of the question
     * Sequence data itself (for sequence identification questions)
   - **For sequence identification questions** (answer_format_label == "Sequence"):
     * DO NOT extract protein/gene names from the sequence or question context
     * DO NOT extract entities like "CD47", "macrophage engulfment" unless they are EXPLICITLY mentioned in the question text
     * Focus on sequence-related entities: "amino acid sequence", "protein sequence", "DNA sequence", etc.
   - **For protein identification questions** (NOT sequence-based):
     * Only extract protein/gene names that are DIRECTLY mentioned in the question text
     * Do NOT extract entities from examples or unrelated context
   
   **Examples:**
   - ✅ CORRECT: Question "What is the function of CD47?" → Extract: ["CD47"]
   - ❌ WRONG: Question "What protein does this sequence represent? MAEQVALSRT..." → Do NOT extract: ["CD47", "macrophage engulfment"] (these are not mentioned)
   - ✅ CORRECT: Sequence question → Extract: ["amino acid sequence", "protein sequence"]

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

   **CRITICAL: Domain-to-Prompt-Module Mapping - MANDATORY**
   You MUST identify domains that can be mapped to available prompt modules. Use these mappings:
   
   | Fine-Grained Domain | Maps to Module |
   |---------------------|----------------|
   | Population Genetics, HWE, Fst, linkage, GWAS, variant | genetics |
   | T cell, B cell, TCR, BCR, MHC, antigen, antibody | immunology |
   | Drug, medication, hypertension, diabetes, treatment | clinical_medicine |
   | Gene expression, transcription, translation, pathway | molecular_biology |
   | Enzyme, metabolism, concentration, binding, kinetics | biochemistry |
   | Sequence analysis, alignment, variant calling | bioinformatics |
   | Cell signaling, receptor, membrane, lipid | biophysics |
   | Hematopoiesis, stem cell, differentiation | cell_biology |
   | Virus, bacteria, pathogen, infection | microbiology |
   | Aphid, insect, plant-herbivore, host adaptation | entomology |
   | Sugar metabolism, carbohydrate, raffinose | biochemistry |
   
   **IMPORTANT**: 
   - If domain doesn't fit existing modules, map to closest related module
   - For cross-domain questions, list ALL relevant modules
   - Example: "Insect Physiology, Sugar Metabolism" → ["biochemistry", "entomology_cross"]

Output your response in JSON format:
{{
    "research_objective": "core research objective with category-specific steps",
    "key_entities": ["entity1", "entity2", ...],
    "core_domains": ["domain1", "domain2", ...],
    "prompt_modules": ["module1", "module2", ...],  // MANDATORY: Maps to available prompt modules
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
    synonyms: List[str] = None,
    cleaned_text: str = None  # ENHANCEMENT: Add original question text for data analysis
) -> str:
    """Base template for N3: Knowledge Retrieval"""
    domains_str = ", ".join(core_domains) if core_domains else "General"
    entities_str = ", ".join(key_entities) if key_entities else "N/A"
    synonyms_str = ", ".join(synonyms) if synonyms else "N/A"
    
    # Extract hard_constraints for display in prompt
    # CRITICAL: Handle both dict and list formats for structured_condition
    hard_constraints = []
    if structured_condition:
        if isinstance(structured_condition, dict):
            hard_constraints = structured_condition.get("hard_constraints", [])
        elif isinstance(structured_condition, list):
            # If it's a list, use it directly as hard_constraints
            hard_constraints = structured_condition
    hard_constraints_str = ", ".join(str(c) for c in hard_constraints) if hard_constraints else "N/A"
    
    # ENHANCEMENT: Detect if question contains experimental data
    has_experimental_data = False
    data_analysis_instruction = ""
    if cleaned_text:
        # Check for patterns indicating experimental data
        data_patterns = [
            r'\d+\s*(?:%|percent|mL|mg|μM|mM|M|g|kg|nm|μm|mm|cm|°C|K|Gy|units|x10\^)',
            r'(?:experiment|study|trial|assay|measurement|observation|results)',
            r'(?:table|figure|data|values?|presented below)',
            r'(?:group|condition|treatment|control)',
            r'(?:vs\.?|versus|compared|comparison)',
            r'(?:significant|difference|effect|change|increase|decrease)'
        ]
        import re
        has_experimental_data = any(re.search(p, cleaned_text, re.IGNORECASE) for p in data_patterns)
        
        if has_experimental_data:
            data_analysis_instruction = """
**CRITICAL: DATA-FIRST ANALYSIS - EXPERIMENTAL DATA DETECTED**

This question contains experimental data with numerical values, conditions, and results. You MUST:

**STEP 1: EXTRACT ALL DATA (MANDATORY)**
Create a structured data table from the question text:
- Groups/Conditions: List all experimental groups (e.g., "CA biotype on diet X", "MA biotype on diet Y")
- Variables: Identify independent variables (treatments, conditions) and dependent variables (outcomes, measurements)
- Values: Extract ALL numerical values with units
- Comparisons: Note which groups are being compared

**STEP 2: ANALYZE DATA PATTERNS (MANDATORY)**
- Trend Analysis: What patterns exist in the data? (e.g., "X increased in condition A vs B")
- Significant Differences: Which comparisons show meaningful differences?
- Anomalies: Are there any unexpected or contradictory findings?

**STEP 3: DERIVE DATA-BASED CONCLUSIONS (MANDATORY)**
What conclusions can be drawn DIRECTLY from the data, WITHOUT needing external knowledge?

**STEP 4: KNOWLEDGE RETRIEVAL (SECONDARY)**
Only retrieve knowledge to:
- Explain WHY the observed patterns occurred (mechanism)
- Verify if the data patterns align with known biology
- Interpret technical terms or concepts

**CRITICAL WARNING**:
- DO NOT use general knowledge to OVERRIDE or IGNORE experimental data
- If data shows "X performed well on Y", the observation is FACT
- Knowledge should EXPLAIN, not CONTRADICT the data
- If option A claims something that contradicts the data → Option A is likely FALSE
- If option B claims something that matches the data → Option B is likely TRUE

**Example**:
- Data shows: "CA did well on raffinose-rich diet (3:8 ratio)"
- Option A claims: "CA has enhanced RFO metabolism" 
- Knowledge says: "Cotton (CA's host) has low RFO, CA has low α-galactosidase"
- Analysis: Data shows good performance, but knowledge says CA shouldn't digest raffinose well
- Resolution: Either (1) CA uses alternative mechanism, or (2) the data interpretation needs care
- For "which is NOT true": The option claiming "enhanced metabolism" WITHOUT evidence from data → likely FALSE
"""

    return f"""You are a biomedical knowledge retrieval expert. Your task is to retrieve relevant knowledge from multiple domains with PRECISE retrieval based on structured information.

Core Domains Identified: {domains_str}
Calculation Type: {calculation_type if calculation_type else "N/A"}
Algorithm Domain: {algorithm_domain if algorithm_domain else "N/A"}
Research Objective: {research_objective if research_objective else "N/A"}
Key Entities: {entities_str}
Synonyms/Normalized Terms: {synonyms_str}
Hard Constraints (MUST NOT violate): {hard_constraints_str}

{data_analysis_instruction}

**CRITICAL: Tool Usage Strategy**
- Use available tools to retrieve domain-specific knowledge
- Prioritize tools that match the identified core domains
- Retrieve information about key entities, relationships, and domain-specific concepts
- Verify retrieved knowledge against hard constraints
- **For data-rich questions**: Only retrieve knowledge needed to INTERPRET data, not to REPLACE data analysis

**CRITICAL: FOR MASS SPECTROMETRY / PROTEIN MODIFICATION QUESTIONS**

If the question involves LC-MS/MS, protein modifications, or mass calculations:
1. **Identify the modification chemistry**:
   - Alkylation with iodoacetamide: adds +57.02 Da (carbamidomethylation) to cysteine
   - Biotinylation reagents: need to look up the exact molecular weight of the reagent used
   - Click chemistry reagents (like DADPS): need to calculate the total modification mass

2. **Calculate modification masses systematically**:
   - Start with the modification reagent's molecular weight
   - Add/subtract any mass changes from the reaction (e.g., -H, -OH, +linker)
   - Account for any cleavage that removes parts of the modification

3. **For MS data analysis questions**:
   - Identify what is STATIC modification (fixed) vs VARIABLE modification (to be determined)
   - Calculate the unknown modification mass (x) based on MS/MS data interpretation
   - Common modification masses to consider: 57 (carbamidomethyl), 144 (biotin-PEG), etc.

Please perform the following tasks:
1. Identify knowledge retrieval targets: Based on research_objective and key_entities, determine what knowledge needs to be retrieved
2. Use appropriate tools: Call relevant tools to retrieve domain-specific knowledge
3. Extract key facts: From retrieved knowledge, extract key facts relevant to answering the question
4. Verify constraints: Ensure retrieved knowledge does not violate hard constraints
5. **For data-rich questions**: Extract and analyze data from the question first
6. **For MS/calculation questions**: Retrieve molecular weights and modification chemistry information

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
    "extracted_data": {{
        "variables": ["var1", "var2", ...],
        "data_points": {{"condition1": "value1", "condition2": "value2", ...}},
        "patterns": ["pattern1", "pattern2", ...]
    }} if experimental data detected,
    "key_facts": {{"fact_key1": "fact_value1", "fact_key2": "fact_value2", ...}},
    "knowledge_validity_label": "Valid|Invalid|Missing",
    "knowledge_unreliable": false,
    "knowledge_confidence": 0.0-1.0,
    "data_sufficient": true|false,  # ENHANCEMENT: Flag if data in question is sufficient to answer
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
1. **Phenomenon-Knowledge Matching**: Match phenomena/entities mentioned in the question with relevant knowledge points from retrieved knowledge
   - For each domain in retrieved knowledge, identify which phenomena/entities match with which knowledge points
   - Create a match table mapping phenomena to knowledge with confidence scores
2. **Core Molecular Function Identification**: If applicable, identify the core molecular function or mechanism involved
3. **Match Confidence Assessment**: Assess the confidence level of the matches (High/Medium/Low)
4. **Recheck Flag**: Determine if the inference needs recheck due to knowledge reliability issues

**CRITICAL: Output Format Requirements**
- You MUST return a "phenomenon_knowledge_match_table" that maps domains to their matched phenomena and knowledge points
- The match table should have this structure: {{"domain1": {{"phenomena": ["phenomenon1", ...], "matched_knowledge": ["knowledge1", ...], "confidence": "High|Medium|Low"}}, ...}}
- If no clear matches are found, return an empty dict {{}} but still set match_confidence_label to "Low"

Output your response in JSON format:
{{
    "phenomenon_knowledge_match_table": {{
        "domain1": {{
            "phenomena": ["phenomenon1", "phenomenon2", ...],
            "matched_knowledge": ["knowledge_point1", "knowledge_point2", ...],
            "confidence": "High|Medium|Low"
        }},
        "domain2": {{
            ...
        }}
    }},
    "core_molecular_function": "core molecular function or mechanism if applicable, else null",
    "match_confidence_label": "High|Medium|Low",
    "need_recheck": true|false,
    "reasoning": "brief explanation of initial inference and matching process"
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
    
    # Extract goal intent for specialized reasoning
    # CRITICAL: Handle both dict and non-dict formats for structured_goal
    if structured_goal and isinstance(structured_goal, dict):
        goal_intent = structured_goal.get("intent", "neutral")
        goal_constraint = structured_goal.get("constraint", "")
    else:
        goal_intent = "neutral"
        goal_constraint = ""
    
    # ENHANCEMENT: Systematic option analysis instruction for multiple choice questions
    option_analysis_instruction = ""
    if question_options and len(question_options) > 0:
        option_labels = [chr(65+i) for i in range(len(question_options))]
        
        # CRITICAL: Detect multi-statement questions (I, II, III, IV pattern)
        multi_statement_instruction = ""
        if cleaned_text and any(marker in cleaned_text for marker in ["I.", "II.", "III.", "IV.", "V.", "VI.", "I,", "II,", "III,"]):
            multi_statement_instruction = """
**CRITICAL: MULTI-STATEMENT QUESTION DETECTED - SPECIAL HANDLING REQUIRED**

This question contains multiple statements (I, II, III, IV, V, etc.) and asks which statements are TRUE/FALSE.

**MANDATORY VERIFICATION PROCEDURE:**

**Step 1: List ALL individual statements from the question**
- Extract each statement: I, II, III, IV, V, etc.
- Quote each statement EXACTLY as it appears

**Step 2: Verify EACH statement against domain knowledge (INDEPENDENTLY)**
For each statement:
- Statement I: "[exact text]" → Knowledge Check: [What does domain knowledge say?] → Verdict: TRUE/FALSE/UNCERTAIN
- Statement II: "[exact text]" → Knowledge Check: [What does domain knowledge say?] → Verdict: TRUE/FALSE/UNCERTAIN
- Statement III: "[exact text]" → Knowledge Check: [What does domain knowledge say?] → Verdict: TRUE/FALSE/UNCERTAIN
- Continue for ALL statements...

**Step 3: Match verified statements to answer options**
- Check each option to see which statements it combines
- Mark option as MATCH if ALL its statements are TRUE (for "which are true" questions)
- Mark option as MATCH if ALL its statements are FALSE (for "which are NOT true" questions)

**Step 4: Cross-verify with option analysis**
- Re-check that the selected option matches the statement-level verdicts
- If there's a conflict, the statement-level analysis takes precedence

**CRITICAL WARNING - AVOID STATEMENT MISCLASSIFICATION:**
- DO NOT assume a statement is TRUE without domain knowledge verification
- DO NOT skip any statement - analyze ALL of them
- DO NOT let option patterns influence your statement-level analysis
- Common errors:
  * Assuming "glycerol supports swarming" when it actually doesn't
  * Assuming "metal chelators inhibit motility" without verifying mechanism
  * Confusing "can" vs "typically does" vs "always does"

**Example:**
- Question: "Which statements about P. aeruginosa are TRUE?"
- Statement I: "Twitching motility is typically initiated by stab inoculation" → TRUE (standard lab protocol)
- Statement II: "10-cm twitching plates typically contain about 25 ml agar" → TRUE (standard protocol)
- Statement III: "It can swarm with glycerol as carbon source" → VERIFY: Does P. aeruginosa swarm on glycerol? → FALSE (requires specific carbon sources)
- Statement IV: "Metal chelators can inhibit swarming motility" → TRUE (chelators bind metals needed for flagellar function)
- Statement V: "After washing and concentrating, culture appears blue-green" → TRUE (pyocyanin pigment)
- If goal asks for TRUE statements: Option combining I, II, IV is correct
- If goal asks for FALSE statements: Option containing only III is correct
"""

        option_analysis_instruction = f"""
{multi_statement_instruction}
**CRITICAL: SYSTEMATIC OPTION ANALYSIS - MANDATORY FOR MULTIPLE CHOICE QUESTIONS**

You MUST analyze EACH option systematically. DO NOT skip any option. Follow this EXACT format:

For each option {', '.join(option_labels)}:
- **Option Analysis Step N**: Analyze option [LABEL]
  - Option Content: [exact option text]
  - Knowledge Verification: [What domain knowledge says about this statement]
  - Fact Check: Is this statement TRUE, FALSE, or UNCERTAIN based on available knowledge?
  - Constraint Check: Does this option violate any constraints from the question?
  - Verdict: [MATCH / EXCLUDE / UNCERTAIN] with reasoning

**MANDATORY OUTPUT REQUIREMENT**: 
- You MUST include analysis for ALL {len(question_options)} options
- Missing any option analysis will result in INVALID response
- The verdict for each option must be explicit (MATCH/EXCLUDE/UNCERTAIN)

**Special Handling for Goal Intent:**
- If goal_intent is "ask_defect" (寻找错误/缺陷): The answer should be the option that is FALSE/INCORRECT
- If goal_intent is "ask_advantage" (寻找优势): The answer should be the option describing advantages
- If goal_intent is "ask_cause" (寻找原因): The answer should be the option explaining causation
- If goal_constraint contains "not true", "incorrect", "false": Answer should be the FALSE option

**CRITICAL: AVOID LOGICAL LEAP ERRORS (避免逻辑跳跃错误) - MANDATORY**

1. **Observation ≠ Causation (观察≠因果)**:
   - If data shows "X performed well on Y diet" → This is an OBSERVATION, not a causal explanation
   - DO NOT conclude "X is adapted to Y" without supporting knowledge
   - Example: "CA did well on raffinose diet" does NOT mean "CA has enhanced raffinose metabolism"
   - The observation might be due to: experimental error, alternative mechanisms, or misleading data

2. **Experimental Data vs Domain Knowledge (实验数据vs领域知识)**:
   - If experimental data contradicts domain knowledge, explicitly note the CONFLICT
   - DO NOT override domain knowledge with single experimental observation
   - Consider: Is this a known phenomenon or an unusual result?

3. **Correlation ≠ Causation (相关≠因果)**:
   - "A happened, then B happened" does NOT mean "A caused B"
   - Always verify causal claims against known mechanisms
   - If option claims causation (e.g., "due to", "owing to"), verify the mechanism is correct

4. **Data Interpretation Rules (数据解读规则)**:
   - "did well" means good PERFORMANCE, not necessarily PREFERENCE or ADAPTATION
   - "performed better" requires comparing against a baseline
   - Numerical data (e.g., "3:8 ratio") should be interpreted with biological context

5. **Mechanism Verification (机制验证)**:
   - If option mentions a mechanism (e.g., "galactosidase activity"), verify:
     * Is this enzyme relevant to the organism/condition?
     * Is the direction of change (increase/decrease) consistent with the cause?
     * Is the stated cause actually correct?

**Example of Correct Reasoning**:
- Question: "CA did well on diet with sucrose:raffinose (3:8). Which is NOT true?"
- Option A: "CA has enhanced ability to metabolize RFOs than MA"
- Analysis: 
  * Data shows: CA did well on high-raffinose diet (observation)
  * Knowledge says: CA is cotton-adapted, cotton has low RFOs, CA has low α-galactosidase
  * Conflict: Observation contradicts knowledge
  * Resolution: Either (1) data is anomalous, or (2) CA uses alternative mechanism
  * Option claim: "CA has enhanced RFO metabolism" - This VIOLATES domain knowledge
  * Verdict: EXCLUDE (this is the FALSE statement if goal is "not true")
"""
    
    return f"""You are a biomedical inference expert. Your task is to perform complete logical inference to derive the final conclusion.

Question:
{cleaned_text}

Research Objective: {research_objective}
Goal Intent: {goal_intent}
Goal Constraint: {goal_constraint}

Initial Associations:
{initial_associations}

Retrieved Knowledge:
{retrieved_knowledge}

Question Options:
{options_str}

Calculation Result (if applicable):
{calculation_result if calculation_result else "N/A"}

{option_analysis_instruction}

**INFERENCE CHAIN REQUIREMENTS:**
1. Build inference chain: Construct a complete logical inference chain from initial associations to final conclusion
2. Apply domain logic: Apply domain-specific logical rules and principles
3. **For multiple choice**: Each inference step should verify one option against knowledge
4. Verify consistency: Check consistency with retrieved knowledge and constraints
5. Derive conclusion: Derive the final conclusion based on complete inference

**CRITICAL: Avoid Logical Fallacies**
- DO NOT make logical leaps without evidence from retrieved knowledge
- DO NOT confuse correlation with causation
- DO NOT generalize from incomplete data
- VERIFY each inference step has supporting evidence

Output your response in JSON format:
{{
    "closed_inference_path": [
        {{"step": 1, "premise": "premise1", "inference": "inference1", "conclusion": "conclusion1"}},
        ...
    ],
    "option_analysis": {{
        "A": {{"verdict": "MATCH|EXCLUDE|UNCERTAIN", "reasoning": "...", "fact_check": "TRUE|FALSE|UNCERTAIN"}},
        "B": {{"verdict": "MATCH|EXCLUDE|UNCERTAIN", "reasoning": "...", "fact_check": "TRUE|FALSE|UNCERTAIN"}},
        ...
    }} if multiple choice,
    "statement_analysis": {{
        "I": {{"statement": "exact text", "verdict": "TRUE|FALSE|UNCERTAIN", "reasoning": "...", "knowledge_source": "..."}},
        "II": {{"statement": "exact text", "verdict": "TRUE|FALSE|UNCERTAIN", "reasoning": "...", "knowledge_source": "..."}},
        ...
    }} if multi-statement question detected,
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
    # CRITICAL: Handle both dict and non-dict formats for structured_goal
    goal_intent = structured_goal.get("intent", "N/A") if structured_goal and isinstance(structured_goal, dict) else "N/A"
    
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

**First, determine the question intent:**
- **Entity Identification Questions**: If the question asks to "identify/determine what [entity]" from given data (e.g., "What protein does this sequence represent?", "Identify the molecule from this spectrum"), the answer MUST be the **ENTITY NAME** (e.g., "rhodopsin", "BRCA1", "glucose"), NOT the data format itself, even if answer_format_label is "Sequence" or "Procedure"
- **Data Format Questions**: If the question asks to "provide/give/show the [data format]" (e.g., "What is the sequence?", "Provide the DNA sequence"), output the data format itself

**Then, match answer_format_label:**
- **Single Choice** → MUST output exactly one option label (e.g., "A", "B")
- **Multi-Select** → MUST output comma-separated option labels (e.g., "A, C, D")
- **Numeric** → MUST output numeric value with units
- **Sequence** → MUST output exact sequence (UNLESS this is an entity identification question - then output entity name)
- **Formula** → MUST output mathematical expression
- **List** → MUST output list format (e.g., ["item1", "item2", ...])
- **Procedure** → MUST output step-by-step procedure (UNLESS this is an entity identification question - then output entity name)
- **Code-Command** → MUST output code/command snippet
- **Short Text** → MUST output concise text answer (e.g., entity name, True/False, short phrase)

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
    # CRITICAL: Handle both dict and non-dict formats for structured_goal
    if structured_goal and isinstance(structured_goal, dict):
        goal_type = structured_goal.get("type", "N/A")
        goal_constraint = structured_goal.get("constraint", "N/A")
    else:
        goal_type = "N/A"
        goal_constraint = "N/A"
    
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
    # Safely format exception_context for display
    import json
    try:
        context_str = json.dumps(exception_context, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        # Fallback to string representation if JSON serialization fails
        context_str = str(exception_context)
    
    return f"""You are a biomedical exception handling expert. Your task is to handle exceptions and find alternative solutions.

Exception Type: {exception_type}

Exception Context:
{context_str}

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

