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
    Base template for N0: Input Preprocessing (OPTIMIZED - reduced from ~240 to ~80 lines)
    """
    return f"""[Role] Biomedical question analysis expert.

[Input]
{user_input}

[Tasks]
1. Clean text → extract core question
2. Classify type: Multiple Choice|Text Matching|Mechanism Explanation|Numerical Calculation|Logical Calculation|Professional Algorithm
3. Extract options (if MCQ)
4. Determine answer format: Single Choice|Multi-Select|Numeric|Short Text|Sequence|Formula|List|Procedure|Code-Command
5. Assess completeness: Complete|Partial Missing|Severe Missing|Incomplete-MissingCoreEntity

[Category Classification] Choose ONE:
- Calculation-[sub]: numerical computation → constraints: ["must complete numerical derivation"]
- ClinicalDecision-[sub]: clinical guidelines → constraints: ["follow latest guidelines", "exclude contraindications"]
- ProfessionalKnowledge-[sub]: domain knowledge → constraints: ["match authoritative sources"]

[Structured 3D Info - MANDATORY]
- Subject: {{type, attribute}}
- Condition: {{type, key_features, hard_constraints}}
- Goal: {{type, constraint, intent: ask_defect|ask_cause|ask_advantage|ask_mechanism|ask_example|ask_definition|ask_comparison|ask_procedure|neutral}}

[Extraction Rules]
- Values: MUST match exactly (e.g., "55uM" not "50uM")
- Entity names: MUST match exactly (e.g., "G1-6" not "G1 to G6")
- Missing core data → data_completeness_label="Incomplete-MissingCoreEntity"

[Special Cases]
- True/False only options → answer_format_label="Short Text" (answer: "True"/"False")
- "Identify X from data" → answer_format_label="Short Text" (entity name, not data)
- "minimum number to distinguish" → question_type_label="Logical Calculation"

[Keywords] Extract:
- core_keywords: ["kw1", "kw2", ...]
- option_features: {{"A": "feature1", "B": "feature2", ...}}
- synonyms: normalized terms for retrieval
- tool_intent: {{"query_go_term": "YES|NO", "query_knowledge_graph": "YES|NO"}}

[JSON Output]
{{
    "cleaned_text": "cleaned question",
    "question_type_label": "Multiple Choice|Text Matching|...",
    "question_category_standard": "Calculation-[sub]|ClinicalDecision-[sub]|ProfessionalKnowledge-[sub]",
    "category_specific_constraints": [...],
    "data_completeness_label": "Complete|Partial Missing|...",
    "core_keywords": [...],
    "option_features": {{...}},
    "synonyms": [...],
    "tool_intent": {{...}},
    "question_options": [...],
    "answer_format_label": "Single Choice|...",
    "structured_subject": {{"type": "...", "attribute": "..."}},
    "structured_condition": {{"type": "...", "key_features": "...", "hard_constraints": [...]}},
    "structured_goal": {{"type": "...", "constraint": "...", "intent": "..."}},
    "key_constraints": [...],
    "missing_core_entities": [...] // if incomplete
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
    """Base template for N1: Question Decomposition (OPTIMIZED)"""
    # Build compact structured context
    subject_info = ""
    condition_info = ""
    goal_info = ""
    
    if structured_subject and isinstance(structured_subject, dict):
        subject_info = f"S:{structured_subject.get('type', '?')}|{structured_subject.get('attribute', '?')}"
    if structured_condition and isinstance(structured_condition, dict):
        condition_info = f"C:{structured_condition.get('type', '?')}|{str(structured_condition.get('key_features', '?'))[:50]}"
    if structured_goal and isinstance(structured_goal, dict):
        goal_info = f"G:{structured_goal.get('type', '?')}|{structured_goal.get('intent', 'neutral')}"
    
    structured_context = f"[N0 Context] {subject_info} || {condition_info} || {goal_info}" if (subject_info or condition_info or goal_info) else ""
    
    return f"""[Role] Biomedical question decomposition expert.

[Input]
Question: {cleaned_text}
Type: {question_type_label} | Category: {question_category_standard or "N/A"}
Constraints: {", ".join(category_specific_constraints) if category_specific_constraints else "None"}
{structured_context}

[Objective Format]
"Determine [goal.type] based on [subject.attribute] with [condition.key_features]"
- Include category-specific steps: Calculation→derive/compare, Clinical→filter/verify, Knowledge→associate/derive

[Entity Rules]
- ONLY extract EXPLICITLY mentioned entities
- DO NOT extract from examples, context, or sequences
- Example: "What is CD47?" → ["CD47"]; "What protein is this sequence?" → ["amino acid sequence"]

[Domain Mapping] Identify test-point level domains:
- Population Genetics → genetics
- T cell/MHC/antibody → immunology
- Drug/treatment → clinical_medicine
- Transcription/pathway → molecular_biology
- Enzyme/metabolism → biochemistry

[JSON Output]
{{
    "research_objective": "Determine [goal] based on [subject] with [condition]",
    "key_entities": ["entity1", ...],
    "core_domains": ["domain1", ...],
    "prompt_modules": ["module1", ...],
    "structured_conditions": {{}},
    "reasoning": "..."
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

[Data Mode] Experimental data detected:
1. EXTRACT: groups, conditions, values, comparisons
2. ANALYZE: patterns, significant differences, anomalies
3. CONCLUDE: from data FIRST, then verify with knowledge
4. WARN: Don't override experimental data with general knowledge
5. If data contradicts knowledge → note CONFLICT explicitly
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
    
    # OPTIMIZED: Compact option analysis instruction (reduced from ~120 lines to ~40 lines)
    # Uses prompt_utils templates for common patterns
    option_analysis_instruction = ""
    if question_options and len(question_options) > 0:
        option_labels = [chr(65+i) for i in range(len(question_options))]
        
        # Multi-statement handling (compact version)
        multi_statement_instruction = ""
        if cleaned_text and any(marker in cleaned_text for marker in ["I.", "II.", "III.", "IV.", "V.", "VI.", "I,", "II,", "III,"]):
            multi_statement_instruction = """
[Multi-Statement Mode] Statements I,II,III... detected:
1. Extract each statement → quote EXACTLY
2. Verify independently: Statement I → knowledge check → TRUE/FALSE/UNCERTAIN
3. Match to options: all TRUE statements → correct option (or all FALSE for "not true" questions)
4. WARN: Don't assume TRUE without verification; analyze ALL statements
"""

        option_analysis_instruction = f"""
{multi_statement_instruction}[Option Analysis] For each option {', '.join(option_labels)}:
- Content: [exact text]
- Knowledge Check: [what domain knowledge says]
- Fact: TRUE/FALSE/UNCERTAIN
- Constraint Check: [violates any?]
- Verdict: MATCH/EXCLUDE/UNCERTAIN

[Goal Intent Handling]
- ask_defect → select FALSE option
- ask_advantage → select advantage option
- ask_cause → select causal explanation
- "not true"/"incorrect" → select FALSE option

[Logic Rules - MANDATORY]
1. Observation ≠ Causation: "X did well on Y" ≠ "X adapted to Y"
2. Data First: Don't override experimental data with general knowledge
3. Correlation ≠ Causation: A before B ≠ A caused B
4. Mechanism Verify: Check enzyme relevance, direction consistency
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

**CRITICAL: Enumeration/Format Questions with Numbered Options**
If the question asks you to "Express your answer as" a specific format like "(1,2,3), (4,5,6)" or mentions numbered options (1), (2), (3), etc.:
- You MUST output the answer in EXACTLY the requested numbered format
- Do NOT output prose descriptions like "Bi-allelic recombination..." 
- Do NOT output scientific explanations in the final_answer field
- Example: If asked "Which mechanisms contribute? Answer as (X,Y,Z)" where X,Y,Z are numbers from (1)-(6):
  - Your final_answer MUST be "(1,4,5), (1,3,4,5,6)" or similar number format
  - NOT "receptor editing, cell doublets, etc."
- The final_answer field should contain ONLY the formatted answer, no explanations

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

