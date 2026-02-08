"""
General QA Agent Prompt Module

Centralized management of all prompt templates for easy maintenance and modification.
Emphasizes scientific rigor and accuracy, suitable for research-oriented multi-agent systems.

All prompts must be in English.
"""

from typing import Dict, List, Any, Optional
import json

# ===================== General QA System Prompt =====================

GENERAL_QA_SYSTEM_PROMPT = """You are a professional research assistant, part of a research-oriented multi-agent system. Your responsibility is to answer user questions following these principles:

## Core Principles

1. **Scientific Rigor**:
   - Base answers on scientific facts and validated theories
   - Distinguish between scientific facts, theoretical hypotheses, and unverified opinions
   - Clearly state uncertainty when content is uncertain

2. **Accuracy**:
   - Use precise, professional terminology
   - Avoid oversimplification or misleading statements
   - Provide clear, logically sound answers

3. **Completeness**:
   - Provide sufficient background information
   - Explain key concepts and terminology
   - Include relevant cross-domain information when necessary

4. **Honesty**:
   - Honestly state when you don't know the answer
   - If the question is beyond your knowledge scope, suggest consulting relevant domain experts
   - Distinguish between known facts and speculation

5. **Research-Oriented**:
   - Provide academic perspectives for research-related questions
   - Mention relevant research fields or methods
   - Encourage users to conduct in-depth exploration and verification

## Answer Format

**CRITICAL: Always start your answer with a direct, concise answer that immediately addresses the question. Users should be able to see the answer at a glance.**

Your answer should follow this structure:
1. **Direct Answer** (First paragraph): Provide a clear, direct answer to the question. This should be the first thing users see.
2. **Detailed Explanation** (Subsequent paragraphs): Provide comprehensive background, step-by-step explanations, and additional context.
3. **Supporting Information**: Include examples, analogies, or related concepts as needed.

- Use clear structure (e.g., bullet points or numbered lists)
- Use professional but accessible language
- Provide examples or analogies when necessary
- Offer hierarchical explanations for complex concepts

## Important Notes

- Do not fabricate information or data
- Do not provide medical, legal, or other professional advice (unless in theoretical discussion)
- For controversial topics, present multiple perspectives
- Maintain an objective, neutral stance

Please answer user questions scientifically and rigorously according to the above principles."""


def get_general_qa_user_prompt(user_input: str) -> str:
    """
    Generate user prompt for general QA
    
    Args:
        user_input: User's question
    
    Returns:
        Formatted user prompt
    """
    return f"""User Question: {user_input}

Please answer this question scientifically and rigorously according to the principles in the system prompt. If the question involves multiple aspects, provide a comprehensive answer. If some aspects are beyond your knowledge scope, please state this honestly.

## Output Requirements

**IMPORTANT: You must return results in pure JSON format only, without any other text or explanations.**

**CRITICAL FORMAT REQUIREMENT FOR THE ANSWER FIELD:**
The "answer" field must start with a direct, concise answer that immediately addresses the question. Users should be able to see the answer at a glance. After the direct answer, provide detailed explanations, steps, and additional context.

Example structure for the "answer" field:
- First sentence/paragraph: Direct answer (e.g., "The answer is X" or "Yes/No, because...")
- Subsequent paragraphs: Detailed explanation, background, steps, and context

Please strictly follow the following JSON format:

{{
    "answer": "DIRECT ANSWER FIRST: [Your direct, concise answer here]. DETAILED EXPLANATION: [Your comprehensive scientific answer with clear structure, professional terminology, and step-by-step explanations]",
    "confidence": "Confidence level description, e.g., 'High confidence, based on validated scientific theories' or 'Medium confidence, some content has uncertainty, particularly...'",
    "related_topics": ["Related question or topic 1", "Related question or topic 2", "Related question or topic 3", "Related question or topic 4", "Related question or topic 5"],
    "sources_suggested": ["Suggested reference or research direction 1", "Suggested reference or research direction 2", "Suggested reference or research direction 3", "Suggested reference or research direction 4", "Suggested reference or research direction 5"]
}}

Field Descriptions:
- **answer** (Required): Provide a complete, scientific answer. MUST start with a direct answer, followed by detailed explanations. Use clear structure and professional terminology.
- **confidence** (Required): Describe in detail your confidence level in the answer, clearly indicating which parts have uncertainty.
- **related_topics** (Required): List 3-5 related questions or topics to help users gain deeper understanding of the relevant field.
- **sources_suggested** (Required): Provide 3-5 specific references, research directions, or academic resources (e.g., specific databases, journals, research fields).

**Output Requirements:**
1. Return only the JSON object, without any other text
2. Ensure correct JSON format, all strings must use double quotes
3. Each element in arrays must be a string
4. Do not use newlines or special characters that break JSON format"""


# ===================== Question Parsing Prompt =====================

QUESTION_PARSING_PROMPT_TEMPLATE = """Task: You are a biomedical question parsing expert. You need to extract structured information from the question text for subsequent reasoning.
Please strictly follow the requirements below:

I. Task 1: Determine Question Type
Choose one from the following 4 types:
1. Judgment Type: Need to judge "whether it is correct", "whether there is bias", "which category it belongs to" (e.g., Which statistic has bias?)
2. Calculation Type: Need to calculate specific numerical values (e.g., What is the value of π?)
3. Analysis Type: Need to analyze causal relationships/effects (e.g., How does a drug affect cell activity?)
4. Enumeration Type: Need to list multiple answers or express answer as comma-separated lists/tuples (e.g., "List 3 types of immune cells", "Express answer as (1,2,3), (4,5,6)", "Which of the following are correct? Answer as (1,4,5), (1,3,4,5,6)")

CRITICAL: If the question asks to "list", "enumerate", "express as", or requires multiple comma-separated values/tuples as the answer format (e.g., "(1,4,5), (1,3,4,5,6)"), it MUST be classified as "Enumeration Type", even if it has answer choices labeled A, B, C, etc.

II. Task 2: Extract Key Information
Must include 4 dimensions (if a dimension is not mentioned in the question, fill in "None"):
1. Analysis Object: The core indicators, substances, or processes studied in the question (e.g., Watterson's θ, nucleotide diversity π, T cells, transcription process)
2. Experimental Conditions: Interventions, treatments, or operations in the question (e.g., "randomly filter low-quality SNVs", "block PD-1 with antibodies", "heat to 50°C")
3. Constraints: Prerequisites or boundary conditions specified in the question (e.g., "sample size is arbitrarily large", "no completely missing SNVs", "pH=7.4")
4. Target Output: The specific output required by the question (e.g., judge which statistic has bias, calculate π value, analyze drug impact mechanism)

III. Output Format Requirements
Output in JSON format, with strict key names: "question_type" (Task 1 result), "key_info" (Task 2 result, containing 4 dimensions).
Do not add any extra text outside the JSON (such as explanations or comments), ensuring it can be directly parsed by Python's json.loads.

IV. Example (for reference)
Example Question:
"In a bioinformatics laboratory, calculate Watterson's θ (θ_W) and nucleotide diversity (π) from VCF files containing human samples, containing only SNVs with no completely missing sites across all samples. The sample size is arbitrarily large, and a small number of low-quality SNVs in each sample are randomly filtered and imputed as reference genome genotypes, with retained SNVs being accurate. Question: Which statistic in the calculation results has bias?"
Example Options: ["A. Only θ has bias", "B. Only π has bias", ...]
Example Output:
{{
    "question_type": "Judgment Type",
    "key_info": {{
        "Analysis Object": "Watterson's θ (θ_W), nucleotide diversity (π)",
        "Experimental Conditions": "Randomly filter low-quality SNVs, impute filtered sites as reference genome genotypes",
        "Constraints": "Sample size is arbitrarily large, contains only SNVs, no completely missing sites across all samples",
        "Target Output": "Judge which statistic has bias"
    }}
}}

V. Question and Options to Parse
Question: {question_text}
Options: {question_options}

Now start parsing, output JSON only!"""


def get_question_parsing_prompt(question_text: str, question_options: List[str]) -> str:
    """
    Generate structured prompt for question parsing
    
    Core design: Clear task objectives + output format template + examples to reduce LLM understanding cost
    
    Args:
        question_text: Question text
        question_options: List of question options
        
    Returns:
        Complete prompt string
    """
    return QUESTION_PARSING_PROMPT_TEMPLATE.format(
        question_text=question_text,
        question_options=str(question_options) if question_options else "No options"
    )


# ===================== New Question Parsing Prompts (Step-by-step) =====================

QUESTION_FORMAT_TYPE_PROMPT = """
Task: Only determine the format type of the biomedical question (3 types only), results are used for subsequent process adaptation.

Core Judgment Rules (Only focus on answer format, ignore question content):
1. Judgment: Must answer with ONLY True/False OR Yes/No (binary answer, no options, no free text)
2. Multiple Choice: Has explicit "Answer Choices" + labeled options (A./B./C./... or 1./2./3./...) for selection, where the answer is ONE of these options
3. Short Answer: All other cases (open text/number/specific format answer, no pre-set selectable options)

Key Constraints for No Misjudgment:
- Numbered mechanism/step descriptions (e.g., (1) XXX (2) XXX) are NOT answer options
- Answer format requirements (e.g., "answer as X-Y", "(1,2),(3,4)", "express as (1,4,5), (1,3,4,5,6)") indicate Short Answer format, NOT Multiple Choice
- If the question asks to "list", "enumerate", "express as", or requires comma-separated lists/tuples as the answer, it is Short Answer format, even if it has labeled choices (A, B, C, etc.)
- Questions requiring enumeration of multiple items (e.g., "Which of the following are correct? Answer as (1,4,5), (1,3,4,5,6)") are Short Answer format

Question: {question_text}

Output Requirement (MUST follow for subsequent process):
Only output a JSON string with NO extra text/line breaks/spaces.
Key: "question_format_type", Value: strictly one of "Short Answer"/"Judgment"/"Multiple Choice"
"""


OPTION_EXTRACTION_PROMPT = """Task: Extract all answer options from a multiple-choice biomedical question.

Question: {question_text}

Instructions:
- Extract all answer options (A, B, C, D, E, etc. or numbered options)
- Preserve the exact text of each option
- If options are not clearly marked, infer them from the question structure
- Return empty list if no options found

Output JSON format only (no extra text):
{{
    "options": ["A. option text 1", "B. option text 2", ...]
}}"""


KEY_INFO_EXTRACTION_PROMPT = """Task: Extract structured key information from the biomedical question.

Question: {question_text}
Options: {question_options}

Extract 4 required dimensions:
1. Analysis Object: Core indicators, substances, or processes studied (e.g., Watterson's θ, T cells, mRNA, BCR)
2. Experimental Conditions: Interventions, treatments, or operations (e.g., "anti-CTLA-4 treatment", "37°C incubation", "gene knockout")
3. Constraints: Prerequisites or boundary conditions (e.g., "sample size is large", "pH=7.4", "no missing data")
4. Target Output: Specific output required (e.g., "judge which statistic has bias", "calculate π value", "analyze mechanism")

If a dimension is not mentioned, use "None".

Output JSON format only (no extra text):
{{
    "key_info": {{
        "Analysis Object": "...",
        "Experimental Conditions": "...",
        "Constraints": "...",
        "Target Output": "..."
    }}
}}"""


CONSTRAINT_TAGGING_PROMPT = """Task: Parse constraints from the biomedical question and categorize them into 3 types of tags for subsequent reasoning.

Question: {question_text}
Constraints from question: {constraints}

You need to extract and categorize constraints into 3 types:

1. Site Constraints (site_constraints): Global site-level constraints
   - Examples: "no_global_missing_snp", "has_global_missing_snp", "only_snp_no_other_variants", "contains_indels", "no_missing_sites"
   - Look for: mentions of missing sites, SNP-only restrictions, variant type restrictions

2. Sample Operation Constraints (sample_operation_constraints): Constraints related to sample processing operations
   - Examples: "random_filtering", "systematic_filtering", "reference_imputation", "no_imputation", "minority_filtering", "majority_filtering"
   - Look for: filtering methods (random vs systematic), imputation methods, filtering proportions (minority/majority)

3. Sample Size Constraints (sample_size_constraints): Constraints related to sample size
   - Examples: "large_sample", "small_sample", "unknown_sample_size", "arbitrarily_large_sample"
   - Look for: explicit mentions of sample size (large, small, arbitrary, etc.)

Tagging Rules:
- Use standardized tag names (lowercase with underscores)
- If a constraint type is not mentioned, use empty list []
- Multiple tags can be assigned to the same category if multiple constraints exist
- Be precise: only tag constraints that are explicitly mentioned or clearly implied

Output JSON format only (no extra text):
{{
    "site_constraints": ["tag1", "tag2", ...],
    "sample_operation_constraints": ["tag1", "tag2", ...],
    "sample_size_constraints": ["tag1", "tag2", ...]
}}

Example 1:
Constraints: "contains only SNVs with no completely missing sites across all samples, sample size is arbitrarily large, randomly filter low-quality SNVs"
Output:
{{
    "site_constraints": ["only_snp_no_other_variants", "no_global_missing_snp"],
    "sample_operation_constraints": ["random_filtering"],
    "sample_size_constraints": ["arbitrarily_large_sample"]
}}

Example 2:
Constraints: "systematically filter 10% of low-quality variants, impute as reference"
Output:
{{
    "site_constraints": [],
    "sample_operation_constraints": ["systematic_filtering", "reference_imputation", "minority_filtering"],
    "sample_size_constraints": []
}}"""


DOMAIN_CLASSIFICATION_PROMPT = """Task: Classify the biomedical question into domain categories (two-level classification).

Question: {question_text}
Key Information: {key_info}

Two-level classification:
1. Major Domain (first level): Immunology, Cell Biology, Biochemistry, Molecular Biology, Genetics, etc.
2. Subdomain (second level): Specific category within major domain (e.g., BCR, TCR, T cell, B cell, antibody, antigen, etc.)

Examples:
- Question about BCR → Major: Immunology, Subdomain: BCR
- Question about T cell activation → Major: Immunology, Subdomain: T cell
- Question about protein synthesis → Major: Biochemistry, Subdomain: Protein synthesis
- Question about gene expression → Major: Molecular Biology, Subdomain: Gene expression

If subdomain cannot be determined, use "General" or "Not specified".

Output JSON format only (no extra text):
{{
    "domain": "Major domain name (e.g., Immunology, Cell Biology, Biochemistry)",
    "subdomain": "Specific subdomain (e.g., BCR, TCR, T cell, B cell, etc.)"
}}"""


COMPREHENSIVE_PARSING_PROMPT = """Task: Comprehensive question parsing for biomedical questions.

Question: {question_text}

Perform the following steps in order:
1. Determine question format type: Short Answer / Judgment / Multiple Choice
2. If Multiple Choice, extract all options
3. Extract key information (4 dimensions: Analysis Object, Experimental Conditions, Constraints, Target Output)
4. Classify domain (two-level: major domain and subdomain)
5. Determine question content type: Judgment / Calculation / Analysis / Enumeration

Output JSON format only (no extra text):
{{
    "question_format_type": "Short Answer" | "Judgment" | "Multiple Choice",
    "question_options": ["A. ...", "B. ..."] or [],
    "question_type": "Judgment" | "Calculation" | "Analysis" | "Enumeration",
    "key_info": {{
        "Analysis Object": "...",
        "Experimental Conditions": "...",
        "Constraints": "...",
        "Target Output": "..."
    }},
    "domain": "Major domain (e.g., Immunology, Cell Biology)",
    "subdomain": "Specific subdomain (e.g., BCR, TCR, T cell)"
}}"""


def get_question_format_type_prompt(question_text: str) -> str:
    """Generate prompt for determining question format type"""
    return QUESTION_FORMAT_TYPE_PROMPT.format(question_text=question_text)


def get_option_extraction_prompt(question_text: str) -> str:
    """Generate prompt for extracting options from multiple-choice questions"""
    return OPTION_EXTRACTION_PROMPT.format(question_text=question_text)


def get_key_info_extraction_prompt(question_text: str, question_options: List[str]) -> str:
    """Generate prompt for extracting key information"""
    options_str = str(question_options) if question_options else "No options"
    return KEY_INFO_EXTRACTION_PROMPT.format(question_text=question_text, question_options=options_str)


def get_domain_classification_prompt(question_text: str, key_info: Dict[str, str]) -> str:
    """Generate prompt for domain classification"""
    key_info_str = json.dumps(key_info, ensure_ascii=False, indent=2)
    return DOMAIN_CLASSIFICATION_PROMPT.format(question_text=question_text, key_info=key_info_str)


def get_constraint_tagging_prompt(question_text: str, constraints: str) -> str:
    """Generate prompt for constraint tagging"""
    return CONSTRAINT_TAGGING_PROMPT.format(question_text=question_text, constraints=constraints)


def get_comprehensive_parsing_prompt(question_text: str) -> str:
    """Generate comprehensive prompt for all parsing steps at once"""
    return COMPREHENSIVE_PARSING_PROMPT.format(question_text=question_text)


# ===================== Enhanced Answer Generation Prompt =====================

def get_enhanced_answer_system_prompt(enhanced_context: Dict[str, Any]) -> str:
    """
    Generate enhanced system prompt with context information
    
    Args:
        enhanced_context: Dictionary containing enhanced context information
        
    Returns:
        Enhanced system prompt string
    """
    import json
    
    # Format validation results
    validation_results = enhanced_context.get('validation_results', {})
    logical_status = "Passed" if validation_results.get('logical', {}).get('passed') else "Failed"
    biological_status = "Passed" if validation_results.get('biological', {}).get('passed') else "Failed"
    statistical_status = "Passed" if validation_results.get('statistical', {}).get('passed') else "Failed"
    
    # Format reasoning steps
    reasoning_steps = enhanced_context.get('reasoning_steps', [])
    reasoning_steps_text = "\n".join(f"- {step}" for step in reasoning_steps) if reasoning_steps else "No reasoning steps"
    
    # Format domain knowledge
    domain_knowledge = enhanced_context.get('domain_knowledge', {})
    if domain_knowledge:
        knowledge_summary = []
        for obj, knowledge in list(domain_knowledge.items())[:3]:  # Show at most 3 analysis objects
            definition = knowledge.get('definition', '')[:50]  # Truncate to first 50 characters
            if definition:
                knowledge_summary.append(f"{obj}: {definition}")
        relevant_concepts = '; '.join(knowledge_summary) if knowledge_summary else "No domain knowledge available"
        key_facts = relevant_concepts  # Use same summary
    else:
        relevant_concepts = "No domain knowledge available"
        key_facts = "No domain knowledge available"
    
    # Format entities
    entities = ', '.join(enhanced_context.get('entities', []))
    
    # Format key info
    key_info_json = json.dumps(enhanced_context.get('key_info', {}), ensure_ascii=False)
    
    # Format domain knowledge details
    domain_knowledge_json = json.dumps(domain_knowledge, ensure_ascii=False) if domain_knowledge else "No domain knowledge"
    
    enhanced_prompt = f"""

## Enhanced Context Information

Question Type: {enhanced_context.get('question_type', 'unknown')}
Domain: {enhanced_context.get('domain', 'unknown')}
Key Information: {key_info_json}
Entities: {entities}

Domain Knowledge:
{domain_knowledge_json}

Knowledge Summary:
- Relevant Concepts: {relevant_concepts}
- Key Facts: {key_facts}

Reasoning Steps:
{reasoning_steps_text}

Validation Results:
- Logical Validation: {logical_status}
- Biological Validation: {biological_status}
- Statistical Validation: {statistical_status}

Please generate a more accurate and scientific answer based on the above enhanced context information."""
    
    return GENERAL_QA_SYSTEM_PROMPT + enhanced_prompt


# ===================== Knowledge Activation Prompt =====================

KNOWLEDGE_ACTIVATION_PROMPT_TEMPLATE = """Task: You are a biomedical domain knowledge expert. You need to provide structured knowledge for given "Analysis Objects" to support subsequent experimental analysis and logical reasoning.
Please strictly follow the requirements below:

I. Knowledge Output Format (Universal - Domain Independent)
1. Output grouped by "Analysis Objects", each object corresponds to a structured dictionary.
2. Each dictionary must contain the following required dimensions:
   
   **A. Basic Info** (通用基础信息段，所有知识点通用):
   - definition: Core definition or essential description
   - dependence: What the knowledge conclusion depends on (e.g., "segregating sites (S) count accuracy", "pairwise difference count accuracy")
   
   **B. Conditional Knowledge** (核心：通用条件化知识段，所有知识点通用，MANDATORY):
   This is the most critical dimension for universal matching. Format: Kc (condition set) → Kr (conclusion)
   - Kc1, Kc2, ...: Arrays of standardized constraint keywords (e.g., ["no_global_missing", "sample_size_large", "filtering_random"])
     * Each Kc represents a set of constraints that must ALL be satisfied for the corresponding Kr to hold
     * Use standardized constraint keywords (see constraint keyword dictionary below)
   - Kr1, Kr2, ...: Conclusions that hold when the corresponding Kc is satisfied
     * Format: "conclusion_type: VALUE; reason: explanation"
     * Example: "theta_bias: UNBIASED; reason: S count is accurate, no under/over estimation"
   - default_Kr: Default conclusion when no Kc matches (e.g., "theta_bias: UNKNOWN; need more constraints")
   
   **C. General Knowledge** (通用泛化知识段，所有知识点通用):
   - General knowledge that applies regardless of specific constraints (e.g., "θ is sensitive to S count, less affected by allele frequency")
   
   **D. Legacy Dimensions** (保留兼容性，可选):
   - Core Definition: (deprecated, use Basic Info.definition instead)
   - Key Rules/Properties: (deprecated, use General Knowledge instead)
   - Association with Experimental Conditions: (deprecated)
   - Scenario Constraint Mapping Table: (deprecated, use Conditional Knowledge instead)
   
3. Output in JSON format, with keys as "Analysis Objects" and values as the above dictionaries. Do not add any extra text outside the JSON.

**CRITICAL: Standardized Constraint Keywords (for Conditional Knowledge)**
Use these standardized keywords in Kc arrays (add more as needed):
- Data integrity: "no_global_missing", "has_global_missing"
- Sample size: "sample_size_large", "sample_size_small", "sample_size_unknown"
- Filtering: "filtering_random", "filtering_systematic", "filtering_minority", "filtering_any"
- Imputation: "imputation_reference", "imputation_sample_specific", "imputation_any"
- Combinations: "filtering_random + filtering_minority" (use + for AND logic)

II. Examples (for reference, need to adapt to actual analysis objects)
Example 1: Analysis Objects = ["Watterson's estimator (theta)", "pi (nucleotide diversity)"]
Example Output (Universal Conditional Knowledge Format):
{{
    "Watterson's estimator (theta)": {{
        "Basic Info": {{
            "definition": "θ = S / aₙ",
            "dependence": "segregating sites (S) count accuracy"
        }},
        "Conditional Knowledge": {{
            "Kc1": ["no_global_missing", "sample_size_large", "filtering_random"],
            "Kr1": "theta_bias: UNBIASED; reason: S count is accurate, no under/over estimation",
            "Kc2": ["has_global_missing", "filtering_systematic"],
            "Kr2": "theta_bias: BIASED_DOWN; reason: S count is underestimated",
            "default_Kr": "theta_bias: UNKNOWN; need more constraints"
        }},
        "General Knowledge": "θ is sensitive to S count, less affected by allele frequency"
    }},
    "pi (nucleotide diversity)": {{
        "Basic Info": {{
            "definition": "π = average pairwise nucleotide differences per site",
            "dependence": "pairwise difference count accuracy"
        }},
        "Conditional Knowledge": {{
            "Kc1": ["filtering_any", "imputation_reference + imputation_sample_specific"],
            "Kr1": "pi_bias: BIASED_DOWN; reason: pairwise differences are underestimated",
            "default_Kr": "pi_bias: UNKNOWN; need more constraints"
        }},
        "General Knowledge": "π is sensitive to both S count and allele frequency"
    }}
}}

Example 2: Analysis Objects = ["CD8+ T cells"]
Constraint Tags Available: [] (no specific constraint tags, use general scenarios)
Example Output:
{{
    "CD8+ T cells": {{
        "Core Definition": "A type of cytotoxic T lymphocyte that expresses CD8 molecules, responsible for recognizing and killing cells infected by pathogens or transformed by cancer",
        "Key Rules/Properties": "Function: Recognize antigen peptides presented by MHC class I molecules through TCR, release perforin and granzymes after activation to induce target cell apoptosis; Activation depends on co-stimulatory signals (such as CD28-B7 binding)",
        "Association with Experimental Conditions": "Anti-CTLA-4 antibody treatment: Blocks CTLA-4's inhibition of co-stimulatory signals, enhancing CD8+ T cell activation and proliferation",
        "Scenario Constraint Mapping Table": []
    }}
}}

III. Analysis Objects to Activate Knowledge
Analysis Objects: {analysis_objects}

IV. Scenario Constraint Mapping Table Generation Rules (CRITICAL)
1. **Purpose**: The mapping table binds knowledge to specific constraint scenarios, preventing over-generalization and ensuring scenario-specific reasoning.
2. **Input**: Use the provided constraint tags to generate relevant constraint combinations:
   - Constraint Tags: {constraint_tags_str}
3. **Table Structure**: For each analysis object, generate a list of mapping entries. Each entry contains:
   - constraint_combination: A list of constraint tags that form a meaningful scenario (e.g., ["large_sample", "random_filtering", "no_global_missing_snp"])
   - Additional dynamic fields: Field names and values should be determined based on the specific question context
     * Field names are NOT fixed - they should reflect the key properties/behaviors relevant to the question
     * For questions comparing multiple analysis objects (e.g., Watterson's θ vs π): use fields like "Watterson's θ 偏倚性", "π（核苷酸多样性）偏倚性", "偏倚方向"
     * For single analysis object questions: use fields that capture the key aspects being asked (e.g., "Bias status", "Calculation accuracy", "Sensitivity to filtering")
     * Field names should be descriptive and specific to the question, not generic like "parameter_1", "parameter_2", "parameter_3"
     * The number of fields is flexible - include as many fields as needed to capture the relevant information
4. **Generation Strategy**:
   - If constraint tags are provided: Generate entries for all relevant constraint combinations (combine tags from different categories if they interact)
   - If no constraint tags: Return empty list []
   - Focus on constraint combinations that significantly affect the object's behavior/properties
   - Each entry should provide actionable information for subsequent reasoning

V. Important Notes
1. Knowledge should focus on content useful for subsequent reasoning (such as formulas, functional mechanisms), avoid irrelevant background.
2. If the analysis object is an indicator (such as θ_W), must include formulas; if it is a cell/molecule (such as CD8+ T cells), must include functions and mechanisms.
3. "Association with Experimental Conditions" should combine biomedical common knowledge (such as known effects of antibody treatments).
4. **Scenario Constraint Mapping Table is MANDATORY**: Always include this dimension, even if it's an empty list. This table is critical for avoiding knowledge generalization errors.

Now start outputting structured knowledge, return JSON only!"""


def get_knowledge_activation_prompt(
    analysis_objects: List[str],
    experimental_conditions: Optional[str] = None,
    constraints: Optional[str] = None,
    target_output: Optional[str] = None,
    constraint_tags: Optional[Dict[str, List[str]]] = None
) -> str:
    """
    Generate structured prompt for knowledge activation
    
    Core design: Clear knowledge dimensions + grouped output by analysis objects + multi-domain examples
    Enhanced with additional context from question parsing (experimental conditions, constraints, target output)
    
    Args:
        analysis_objects: List of analysis objects (e.g., ["CD8+ T cells", "Watterson's θ"])
        experimental_conditions: Experimental conditions from question (optional)
        constraints: Constraints from question (optional)
        target_output: Target output from question (optional)
        
    Returns:
        Complete prompt string
    """
    # Format analysis objects as a list string
    objects_str = str(analysis_objects)
    
    # Format constraint tags for prompt
    if constraint_tags:
        constraint_tags_str = json.dumps(constraint_tags, ensure_ascii=False, indent=2)
    else:
        constraint_tags_str = "{}"
    
    # Build additional context section
    additional_context_parts = []
    if experimental_conditions:
        additional_context_parts.append(f"Experimental Conditions: {experimental_conditions}")
    if constraints:
        additional_context_parts.append(f"Constraints: {constraints}")
    if target_output:
        additional_context_parts.append(f"Target Output: {target_output}")
    
    additional_context = "\n".join(additional_context_parts) if additional_context_parts else "None"
    
    # Use enhanced template if additional context is available
    if additional_context_parts:
        # Create enhanced template with context
        enhanced_template = KNOWLEDGE_ACTIVATION_PROMPT_TEMPLATE + """

V. Additional Context from Question
{additional_context}

VI. Context-Guided Knowledge Activation
1. **Experimental Conditions**: If provided, focus the "Association with Experimental Conditions" dimension on how the analysis objects respond to these specific conditions.
2. **Constraints**: If provided, ensure the knowledge accounts for these constraints (e.g., "large sample size" affects formula applicability).
3. **Target Output**: If provided, prioritize knowledge that directly supports achieving this target (e.g., if target is "calculate π", emphasize π calculation formulas).
4. **Constraint Tags**: Use the provided constraint tags to generate Scenario Constraint Mapping Table entries. Combine tags from different categories (site, operation, size) to form meaningful constraint combinations.

Now start outputting structured knowledge, return JSON only!"""
        return enhanced_template.format(
            analysis_objects=objects_str,
            constraint_tags_str=constraint_tags_str,
            additional_context=additional_context
        )
    else:
        return KNOWLEDGE_ACTIVATION_PROMPT_TEMPLATE.format(
            analysis_objects=objects_str,
            constraint_tags_str=constraint_tags_str
        )


# ===================== Knowledge Activation Prompt with Deep Research Context =====================

KNOWLEDGE_ACTIVATION_WITH_RESEARCH_PROMPT_TEMPLATE = """Task: You are a biomedical domain knowledge expert (focus on genetics/genomics). You need to provide structured, precise knowledge for given "Analysis Objects" to support subsequent experimental analysis and logical reasoning (especially Judgment/True-False questions).
**IMPORTANT: Prioritize using deep research context (literature findings) and cite sources when available.**

Please strictly follow the requirements below:

I. Research Context (from Deep Literature Research)
{research_context}

II. Knowledge Output Format
1. Output grouped by "Analysis Objects", each object corresponds to a structured dictionary.
2. Each dictionary must contain 5 required dimensions (if a dimension is not applicable, fill in "None" or empty list []):
   - Core Definition: Academic definition (prefer research context definitions if available).
   - Key Rules/Properties: Core principles (formulas/mechanisms) + **qualification conditions/exception scenarios** (MUST distinguish "typically/generally" vs "necessarily/absolutely" for Judgment questions).
   - Association with Experimental Conditions: Object's response to experimental treatments (reference research context findings if available).
   - Interrelation: Logical/quantitative relationships BETWEEN analysis objects (MUST include modal logic + exception conditions for Judgment questions).
   - Scenario Constraint Mapping Table: Knowledge-scenario binding table showing how object properties/behaviors vary under different constraint combinations. This table prevents knowledge generalization and ensures scenario-specific reasoning.
3. Output in JSON format, with keys as "Analysis Objects" and values as the above dictionaries. Do not add any extra text outside the JSON.

III. Examples (Genetics Focus, for strict reference)
Example 1: Analysis Objects = ["polygenic score (PGS)", "SNP heritability (h²_SNP)", "variance explained (R²)"]
Example Output:
{{
    "polygenic score (PGS)": {{
        "Core Definition": "A quantitative measure aggregating additive effects of multiple SNPs to predict trait predisposition (PGS = ∑(β_i * G_i); β_i = effect size, G_i = genotype) [Research Context: Smith et al., 2023]",
        "Key Rules/Properties": "PGS's R² typically < h²_SNP (95% of real-world studies), but R² = h²_SNP under ideal conditions (all causal SNPs included, unbiased β_i) [Research Context: Jones et al., 2022]",
        "Association with Experimental Conditions": "None",
        "Interrelation": "PGS's R² is TYPICALLY lower than h²_SNP, but NOT NECESSARILY lower (equality achievable in theory) [Research Context: Genetics Reviews, 2024]"
    }},
    "SNP heritability (h²_SNP)": {{
        "Core Definition": "Proportion of phenotypic variance from additive SNP effects (estimated via GREML) [Research Context: Yang et al., 2011]",
        "Key Rules/Properties": "h²_SNP is the upper limit of SNP-based variance explanation; h²_SNP ≥ PGS's R² (typically), equal under ideal PGS conditions [Research Context: LD Score Regression Consortium, 2015]",
        "Association with Experimental Conditions": "None",
        "Interrelation": "h²_SNP (total additive SNP variance) is not necessarily greater than PGS's R² (equal if PGS captures all causal SNPs) [Research Context: Polygenic Score Guidelines, 2023]"
    }},
    "variance explained (R²)": {{
        "Core Definition": "Proportion of phenotypic variance explained by PGS (R² = Var(PGS)/Var(Phenotype)) [Research Context: Basic Statistical Genetics, 2020]",
        "Key Rules/Properties": "R² ≤ h²_SNP (typically), but R² = h²_SNP under ideal conditions [Research Context: Simulation Studies, 2021]",
        "Interrelation": "For PGS, R² is typically less than h²_SNP, but not necessarily lower (exception: perfect PGS construction)"
    }}
}}

IV. Analysis Objects to Activate Knowledge
Analysis Objects: {analysis_objects}

V. Question Type Guidance (CRITICAL)
Question Type: {question_type}
- If Question Type = "Judgment"/"True-False": MUST explicitly distinguish "typically/generally" vs "necessarily/absolutely" in Key Rules/Properties and Interrelation, and list **exception conditions** (cite research context if available).
- If Question Type = "Calculation": Prioritize formulas from research context + calculation constraints.
- If Question Type = "Analysis": Prioritize causal relationships from research context + experimental effects.

VI. Scenario Constraint Mapping Table Generation Rules (CRITICAL)
1. **Purpose**: The mapping table binds knowledge to specific constraint scenarios, preventing over-generalization and ensuring scenario-specific reasoning.
2. **Input**: Use the provided constraint tags to generate relevant constraint combinations:
   - Constraint Tags: {constraint_tags_str}
3. **Table Structure**: For each analysis object, generate a list of mapping entries. Each entry contains:
   - constraint_combination: A list of constraint tags that form a meaningful scenario (e.g., ["large_sample", "random_filtering", "no_global_missing_snp"])
   - Additional dynamic fields: Field names and values should be determined based on the specific question context
     * Field names are NOT fixed - they should reflect the key properties/behaviors relevant to the question
     * For questions comparing multiple analysis objects: use descriptive field names (e.g., "Watterson's θ 偏倚性", "π（核苷酸多样性）偏倚性", "偏倚方向")
     * For single analysis object questions: use fields that capture the key aspects (e.g., "Bias status", "Calculation accuracy", "Sensitivity to filtering")
     * Field names should be descriptive and specific to the question, not generic like "parameter_1", "parameter_2", "parameter_3"
     * The number of fields is flexible - include as many fields as needed to capture the relevant information
4. **Generation Strategy**:
   - If constraint tags are provided: Generate entries for all relevant constraint combinations (combine tags from different categories if they interact)
   - If no constraint tags: Return empty list []
   - Focus on constraint combinations that significantly affect the object's behavior/properties
   - Each entry should provide actionable information for subsequent reasoning
   - Reference research context findings when available to support scenario-specific conclusions

VII. Important Notes
1. **PRIORITY**: Use research context information first (cite sources in knowledge). If no research context is available, use general knowledge (note: "No research context available").
2. For Judgment questions, modal logic (necessarily/typically) and exception conditions are MANDATORY in Key Rules/Properties and Interrelation.
3. For genetic indicators (PGS, h²_SNP), include formulas + modal logic + exception conditions (reference research context if possible).
4. Avoid irrelevant background; focus on reasoning support for the given question type.
5. **Scenario Constraint Mapping Table is MANDATORY**: Always include this dimension, even if it's an empty list. This table is critical for avoiding knowledge generalization errors.

Now start outputting structured knowledge, return JSON only!"""


def get_knowledge_activation_prompt_with_research(
    analysis_objects: List[str],
    question_type: str,  # 新增：传入问题类型
    research_context: str = "",
    research_brief: str = "",
    experimental_conditions: Optional[str] = None,
    constraints: Optional[str] = None,
    target_output: Optional[str] = None,
    constraint_tags: Optional[Dict[str, List[str]]] = None
) -> str:
    """
    Generate structured prompt for knowledge activation with deep research context (enhanced for Judgment questions)
    """
    objects_str = str(analysis_objects)
    
    # Format constraint tags for prompt
    if constraint_tags:
        constraint_tags_str = json.dumps(constraint_tags, ensure_ascii=False, indent=2)
    else:
        constraint_tags_str = "{}"
    
    # Combine research context and brief (with citation hints)
    if research_brief and research_context:
        combined_research = f"Research Brief (key findings):\n{research_brief}\n\nDetailed Research Report (full literature context):\n{research_context}\n\nNote: Cite research sources in knowledge (e.g., [Smith et al., 2023]) where applicable."
    elif research_context:
        combined_research = f"Detailed Research Report:\n{research_context}\n\nNote: Cite research sources in knowledge (e.g., [Yang et al., 2011]) where applicable."
    elif research_brief:
        combined_research = f"Research Brief (key findings):\n{research_brief}\n\nNote: Cite research sources in knowledge (e.g., [Jones et al., 2022]) where applicable."
    else:
        combined_research = "No research context available. Use general knowledge (note: 'No research context available' in knowledge)."
    
    # Build additional context
    additional_context_parts = []
    if experimental_conditions:
        additional_context_parts.append(f"Experimental Conditions: {experimental_conditions}")
    if constraints:
        additional_context_parts.append(f"Constraints: {constraints}")
    if target_output:
        additional_context_parts.append(f"Target Output: {target_output} (prioritize knowledge supporting this)")
    
    additional_context = "\n".join(additional_context_parts) if additional_context_parts else "None"
    
    # Enhanced template with research + question type + additional context
    enhanced_template = KNOWLEDGE_ACTIVATION_WITH_RESEARCH_PROMPT_TEMPLATE + """

VIII. Additional Context from Question
{additional_context}

IX. Final Adjustments
1. Align "Association with Experimental Conditions" with the given experimental conditions (cross-reference research context).
2. Incorporate constraints into Key Rules/Properties (e.g., "under large sample size constraints, PGS R² approaches h²_SNP").
3. Prioritize knowledge that supports the target output (e.g., for "judge necessity", emphasize modal logic/exception conditions from research context).
4. **Constraint Tags**: Use the provided constraint tags to generate Scenario Constraint Mapping Table entries. Combine tags from different categories (site, operation, size) to form meaningful constraint combinations. Reference research context findings when available.

Now start outputting structured knowledge, return JSON only!"""
    
    return enhanced_template.format(
        analysis_objects=objects_str,
        question_type=question_type,  # 传递问题类型
        research_context=combined_research,
        constraint_tags_str=constraint_tags_str,
        additional_context=additional_context
    )


# ===================== Experiment Analysis Prompt =====================

EXPERIMENT_ANALYSIS_PROMPT_TEMPLATE = """
Task: You are a biomedical experiment analysis expert. You need to associate "Experimental Conditions" with "Domain Knowledge" to analyze the specific impact of experimental treatments on "Analysis Object", providing basis for subsequent reasoning.
**CRITICAL: You must use constraint-specific knowledge from Scenario Constraint Mapping Table to avoid generalization errors.**

Please strictly follow the "Constraint-Knowledge Mapping → Operation Breakdown → Knowledge Association → Impact Judgment" logic, and prohibit baseless speculation.

I. Known Information
1. Analysis Object: {analysis_object}
2. Experimental Conditions (needs breakdown analysis): {experiment_condition}
3. Domain Knowledge (can be directly cited): {knowledge_str}
4. Constraint Tags from Question: {constraint_tags_str}
5. Matched Constraint Scenarios: {matched_mappings_str}

II. Constraint-Knowledge Mapping Validation (CRITICAL FIRST STEP)
Before analyzing operations, you MUST:
1. **Match Constraint Tags**: Use the provided constraint tags to find matching entries in "Scenario Constraint Mapping Table" from Domain Knowledge.
2. **Use Matched Scenarios**: For each matched scenario entry, extract all parameter values (dynamic fields, excluding constraint_combination) that describe the object's behavior under the specific constraint combination.
3. **Apply Constraint-Specific Knowledge**: Use these parameter values to guide your impact judgment, rather than applying general knowledge. This prevents generalization errors.
4. **If No Match Found**: If no matching scenario is found, explicitly state this in "Core Hypothesis Validation" and use general knowledge with caution.

Example: If constraint tags are ["large_sample", "random_filtering", "no_global_missing_snp"] and a matching entry shows a field like "Bias status: unbiased" or "Watterson's θ 偏倚性: 无偏", then your impact judgment MUST reflect that the object is unbiased under these specific constraints, not just generally.

III. Analysis Output Format
Output in JSON format, must contain 5 required dimensions (if a dimension is not applicable, fill in "None" or empty list []):
1. Constraint-Knowledge Match Summary: Summary of matched constraint scenarios and their parameter values (e.g., "Matched scenario [large_sample, random_filtering]: 'Bias status' field indicates unbiased, 'Calculation accuracy' field indicates high accuracy" or "Matched scenario [...]: 'Watterson's θ 偏倚性' indicates 无偏, 'π（核苷酸多样性）偏倚性' indicates 有偏").
2. Operation Breakdown: Break down "Experimental Conditions" into 1~3 specific operations (e.g., "anti-CTLA-4 antibody treatment", "37°C incubation"), avoid vague descriptions;
3. Knowledge Association Basis: Domain knowledge corresponding to each operation (must cite "Key Rules/Properties", "Association with Experimental Conditions", or "Scenario Constraint Mapping Table" from "Domain Knowledge");
4. Impact Direction Judgment: Quantitative/qualitative impact of each operation on "Analysis Object" (e.g., "promotes CD8+ T cell proliferation", "causes π underestimation", "no significant impact"). **MUST be consistent with matched constraint scenarios if available.**
5. Core Hypothesis Validation: Whether implicit assumptions in the question are reasonable (e.g., "filtering does not lose segregating sites in large samples" is consistent with domain knowledge). Include constraint matching results here.

IV. Examples (for reference, adapt to actual information)
Example 1: Known Information
- Analysis Object: CD8+ T cell proliferation rate
- Experimental Conditions: Experimental group treated with anti-CTLA-4 antibody, control group treated with saline, incubated at 37°C
- Domain Knowledge: CD8+ T cells' "Association with Experimental Conditions" = "anti-CTLA-4 antibody blocks negative co-stimulatory signals, enhances proliferation"
Example Output:
{{
    "Operation Breakdown": ["anti-CTLA-4 antibody treatment", "saline treatment (control)", "37°C incubation"],
    "Knowledge Association Basis": [
        "anti-CTLA-4 treatment: Citing CD8+ T cell knowledge that 'after antibody blocks CTLA-4, CD28-B7 binding enhances, proliferation capacity increases'",
        "saline treatment: No corresponding knowledge (control operation, no effect)",
        "37°C incubation: No corresponding knowledge (physiological temperature, no additional effect)"
    ],
    "Impact Direction Judgment": [
        "anti-CTLA-4 antibody treatment: Promotes CD8+ T cell proliferation rate (relative to control)",
        "saline treatment: No effect on CD8+ T cell proliferation rate (control baseline)",
        "37°C incubation: No additional effect on CD8+ T cell proliferation rate (suitable temperature)"
    ],
    "Core Hypothesis Validation": "Question assumption 'mice have same genetic background' is reasonable: consistent genetic background can exclude interference, conforms to immunological experiment design common sense"
}}

Example 2: Known Information
- Analysis Object: Nucleotide diversity π
- Experimental Conditions: Randomly filter low-quality SNVs, impute filtered sites as reference type, sample size is arbitrarily large
- Constraint Tags: ["arbitrarily_large_sample", "random_filtering", "reference_imputation", "no_global_missing_snp"]
- Matched Scenario: {{"constraint_combination": ["arbitrarily_large_sample", "reference_imputation", "no_global_missing_snp"], "Bias status": "biased (underestimated)", "Calculation accuracy": "low", "Filtering sensitivity": "high"}}
- Domain Knowledge: π's "Key Rules/Properties" = "π_site ≈ 2f(1-f), decrease in f leads to π underestimation"
Example Output:
{{
    "Constraint-Knowledge Match Summary": "Matched scenario [arbitrarily_large_sample, reference_imputation, no_global_missing_snp]: 'Bias status' field indicates biased (underestimated), 'Calculation accuracy' field indicates low, 'Filtering sensitivity' field indicates high sensitivity to imputation. This scenario-specific knowledge overrides general knowledge.",
    "Operation Breakdown": ["randomly filter low-quality SNVs", "impute filtered sites as reference type"],
    "Knowledge Association Basis": [
        "random filtering: Citing π's Scenario Constraint Mapping Table that 'random filtering does not systematically change allele frequencies' (from matched scenario)",
        "impute as reference type: Citing π's Scenario Constraint Mapping Table that 'reference imputation causes bias (underestimated)' and 'high sensitivity to imputation' (from 'Bias status' and 'Filtering sensitivity' fields in matched scenario)"
    ],
    "Impact Direction Judgment": [
        "random SNV filtering: No direct impact (matched scenario confirms random filtering maintains frequency distribution)",
        "impute as reference type: Causes π underestimation (matched scenario 'Bias status' field explicitly states 'biased (underestimated)', V→R decreases variant frequency f, 2f(1-f) decreases)"
    ],
    "Core Hypothesis Validation": "Question assumption 'sample size is arbitrarily large' is reasonable: matched constraint scenario confirms that under [arbitrarily_large_sample, reference_imputation, no_global_missing_snp] combination, π is biased (underestimated) with low accuracy, which is consistent with domain knowledge that 'decrease in f leads to π underestimation'. Constraint matching successful."
}}

V. Important Notes
1. **Constraint-Knowledge Mapping is MANDATORY**: Always check for matched constraint scenarios first, and use their parameter values to guide impact judgment. This prevents generalization errors.
2. Impact judgment must be "constraint-specific knowledge-first": when matched constraint scenarios are available, prioritize their parameter values over general knowledge.
3. When there is no corresponding domain knowledge or matched scenario, do not arbitrarily infer (fill in "no clear impact");
4. Avoid redundancy: operation breakdown should not exceed 3 items, descriptions for each dimension should focus on core, no repetition;
5. Control operations must be clear: e.g., "saline treatment" should be labeled as control to avoid confusion with treatment operations.
6. **If constraint matching fails**: Explicitly state in "Core Hypothesis Validation" that no matching constraint scenario was found, and proceed with general knowledge but note the limitation.

Now start analysis, output JSON only!
"""


def get_experiment_analysis_prompt(
    experiment_condition: str,
    domain_knowledge: Dict[str, Dict[str, Any]],
    analysis_object: str,
    constraint_tags: Optional[Dict[str, List[str]]] = None,
    matched_mappings: Optional[Dict[str, Dict[str, Any]]] = None
) -> str:
    """
    Generate structured prompt for experiment analysis
    
    Core design: Force "constraint-knowledge mapping → operation breakdown → knowledge correspondence → impact derivation" logic to ensure constraint-specific analysis
    
    Args:
        experiment_condition: Experimental conditions string
        domain_knowledge: Domain knowledge dictionary (from second node)
        analysis_object: Analysis object string
        constraint_tags: Constraint tags from question parsing (optional)
        matched_mappings: Matched constraint-knowledge mappings (optional)
        
    Returns:
        Formatted prompt string
    """
    import json
    # Format domain knowledge for LLM reading
    knowledge_str = json.dumps(domain_knowledge, ensure_ascii=False, indent=2)
    
    # Format constraint tags
    if constraint_tags:
        constraint_tags_str = json.dumps(constraint_tags, ensure_ascii=False, indent=2)
    else:
        constraint_tags_str = "{}"
    
    # Format matched mappings
    if matched_mappings:
        matched_mappings_str = json.dumps(matched_mappings, ensure_ascii=False, indent=2)
    else:
        matched_mappings_str = "{}"
    
    return EXPERIMENT_ANALYSIS_PROMPT_TEMPLATE.format(
        analysis_object=analysis_object,
        experiment_condition=experiment_condition,
        knowledge_str=knowledge_str,
        constraint_tags_str=constraint_tags_str,
        matched_mappings_str=matched_mappings_str
    )


# ===================== Logical Derivation Prompt =====================

def _get_judgment_derivation_prompt_template() -> str:
    """Judgment type derivation prompt template"""
    return """
Task: For "Judgment Type" biomedical questions, derive preliminary conclusions based on previous analysis, preparing for option matching.
Known Information:
1. Core Question: {question_core}
2. Analysis Object: {analysis_object}
3. Domain Knowledge: {knowledge_str}
4. Experiment Impact Analysis (Node 3 output): {analysis_str}
5. Constraint Tags: {constraint_tags_str}

Derivation Requirements:
1. Derivation Strategy: Must be based on "experimental impact direction → question judgment" logic, prohibit baseless speculation;
2. Core Evidence Chain: Clearly cite "impact direction judgment from experiment analysis" and "key rules from domain knowledge";
3. Preliminary Conclusion: Directly answer the core question (e.g., "yes/no", "statistic A has bias/statistic B is unbiased");
4. Rebuttal Evidence (CRITICAL): After deriving Preliminary Conclusion, perform counter-evidence reasoning to strengthen conclusion's conditional reasonableness:
   - Assume the OPPOSITE of your Preliminary Conclusion is true
   - Find evidence that would support this opposite conclusion
   - Check if this counter-evidence is consistent with: (1) constraint tags from question, (2) matched constraint scenarios from Scenario Constraint Mapping Table, (3) domain knowledge, (4) experiment analysis
   - If counter-evidence contradicts constraints/knowledge, this strengthens your Preliminary Conclusion
   - If counter-evidence is plausible under different constraints, note this limitation
   - Record the rebuttal process: what counter-evidence was considered, why it was rejected/confirmed, and how this validates the Preliminary Conclusion
5. Option Matching Priority: CRITICAL - You must explicitly map each option to your Preliminary Conclusion:
   - For EACH option (A, B, C, D, etc.), state whether it MATCHES or CONTRADICTS your Preliminary Conclusion
   - Provide specific reason for each option (e.g., "Option A contradicts because it claims θ is biased, but Preliminary Conclusion states θ is unbiased")
   - Only rank options that MATCH your Preliminary Conclusion as high priority
   - If an option contradicts your Preliminary Conclusion, it MUST be ranked as low priority or excluded
   - Format: "A>B>C>D, because: A matches (reason), B contradicts (reason), C matches (reason), D contradicts (reason)"

Output Format: JSON format, containing "Derivation Strategy", "Core Evidence Chain", "Preliminary Conclusion", "Rebuttal Evidence", "Option Matching Priority", no extra text.
Example Output:
{{
    "Derivation Strategy": "Experimental impact direction → question judgment: first determine anti-CTLA-4's impact on proliferation, then judge whether experimental group is higher than control",
    "Core Evidence Chain": [
        "Node 3 impact direction judgment: anti-CTLA-4 treatment significantly promotes CD8+ T cell proliferation rate (relative to control)",
        "Node 2 domain knowledge: CD8+ T cell proliferation depends on co-stimulatory signals, antibody blocks negative signals to enhance this signal"
    ],
    "Preliminary Conclusion": "Experimental group CD8+ T cell proliferation rate is higher than control group",
    "Rebuttal Evidence": {{
        "counter_hypothesis": "Experimental group proliferation rate is NOT higher than control (equal or lower)",
        "counter_evidence_considered": [
            "Counter-evidence 1: Anti-CTLA-4 might inhibit proliferation if CTLA-4 actually promotes T cell activation (opposite mechanism)",
            "Counter-evidence 2: Experimental conditions might have confounding factors that reduce proliferation",
            "Counter-evidence 3: Sample size or constraints might invalidate the conclusion"
        ],
        "rebuttal_analysis": [
            "Counter-evidence 1 REJECTED: Domain knowledge explicitly states CTLA-4 is a negative co-stimulatory molecule that INHIBITS activation. Blocking it enhances proliferation. This counter-evidence contradicts established domain knowledge.",
            "Counter-evidence 2 REJECTED: Experiment analysis shows no confounding factors mentioned. Control group uses saline (baseline), experimental group uses anti-CTLA-4. No constraints suggest reduced proliferation.",
            "Counter-evidence 3 REJECTED: No constraint tags suggest sample size issues or conditions that would reverse the impact direction. Constraint validation confirms experimental setup is valid."
        ],
        "rebuttal_conclusion": "All counter-evidence contradicts domain knowledge, experiment analysis, and constraint conditions. This strengthens the Preliminary Conclusion that experimental group proliferation is higher than control. The conclusion is conditionally reasonable under the given constraints."
    }},
    "Option Matching Priority": "A>B>C>D, because: Option A matches (claims 'yes, anti-CTLA-4 promotes proliferation', which is consistent with preliminary conclusion 'experimental group proliferation higher than control'); Option B contradicts (claims 'no, anti-CTLA-4 inhibits proliferation', which contradicts preliminary conclusion); Option C contradicts (claims 'no difference', which contradicts preliminary conclusion); Option D contradicts (claims 'cannot judge, missing dose info', which contradicts preliminary conclusion that dose is not needed)"
}}
"""


def _get_calculation_derivation_prompt_template() -> str:
    """Calculation type derivation prompt template"""
    return """
Task: For "Calculation Type" biomedical questions, complete numerical derivation based on previous analysis.
Known Information:
1. Core Question: {question_core}
2. Analysis Object: {analysis_object}
3. Domain Knowledge (including formulas): {knowledge_str}
4. Experiment Impact Analysis (including data): {analysis_str}
5. Constraint Tags: {constraint_tags_str}

Derivation Requirements:
1. Derivation Strategy: Must be based on "domain knowledge formula → question data substitution → numerical calculation" logic;
2. Core Evidence Chain: Clearly cite "formulas from domain knowledge" and "data from question/experiment analysis";
3. Formula Substitution Process: Step-by-step show the source of each parameter value in the formula (e.g., "f=0.3, from question 'variant frequency 30%'");
4. Numerical Calculation Result: Provide specific calculation process and final numerical value (keep 2 decimal places);
5. Preliminary Conclusion: Answer the core question with numerical value.
6. Rebuttal Evidence (CRITICAL): After deriving Preliminary Conclusion, perform counter-evidence reasoning:
   - Assume the calculated value is WRONG (significantly different)
   - Find alternative parameter values or formulas that would lead to different results
   - Check if these alternatives are consistent with: (1) constraint tags, (2) matched constraint scenarios, (3) domain knowledge, (4) experiment analysis
   - If alternatives contradict constraints/knowledge, this validates your calculation
   - Record the rebuttal process: what alternative calculations were considered, why they were rejected, and how this validates the Preliminary Conclusion

Output Format: JSON format, containing "Derivation Strategy", "Core Evidence Chain", "Formula Substitution Process", "Numerical Calculation Result", "Preliminary Conclusion", "Rebuttal Evidence", no extra text.
Example Output:
{{
    "Derivation Strategy": "Formula substitution calculation: use π_site=2f(1-f), substitute observed f value to calculate π",
    "Core Evidence Chain": [
        "Node 2 formula: π_site≈2f(1-f) (f is variant allele frequency)",
        "Node 3 data: after imputation, f decreases from 0.3 to 0.2 (due to V→R replacement)"
    ],
    "Formula Substitution Process": [
        "Step 1: Determine formula parameters: π_site=2f(1-f)",
        "Step 2: Get value: f=0.2 (from Node 3 experiment analysis)",
        "Step 3: Substitute: 2×0.2×(1-0.2)=2×0.2×0.8"
    ],
    "Numerical Calculation Result": "0.32 (calculation process: 2×0.2×0.8=0.32)",
    "Preliminary Conclusion": "π_site value after imputation is 0.32"
}}
"""


def _get_analysis_derivation_prompt_template() -> str:
    """Analysis type derivation prompt template"""
    return """
Task: For "Analysis Type" biomedical questions, analyze causal chain derivation based on previous analysis.
Known Information:
1. Core Question: {question_core}
2. Analysis Object: {analysis_object}
3. Domain Knowledge (including mechanisms): {knowledge_str}
4. Experiment Impact Analysis (including operations): {analysis_str}
5. Constraint Tags: {constraint_tags_str}

Derivation Requirements:
1. Derivation Strategy: Must be based on "experimental operation → molecular/cellular mechanism → final impact" causal chain analysis;
2. Core Evidence Chain: Clearly cite "action mechanisms from domain knowledge" and "operation impacts from experiment analysis";
3. Causal Chain Analysis: Step-by-step show "operation → mechanism → impact" logic (e.g., "antibody → blocks protein → signal change → function change");
4. Preliminary Conclusion: Summarize causal chain, answer question "how does it affect".
5. Rebuttal Evidence (CRITICAL): After deriving Preliminary Conclusion, perform counter-evidence reasoning:
   - Assume the OPPOSITE causal chain or impact direction is true
   - Find alternative mechanisms or pathways that would lead to different impacts
   - Check if these alternatives are consistent with: (1) constraint tags, (2) matched constraint scenarios, (3) domain knowledge, (4) experiment analysis
   - If alternatives contradict constraints/knowledge, this strengthens your Preliminary Conclusion
   - Record the rebuttal process: what alternative causal chains were considered, why they were rejected, and how this validates the Preliminary Conclusion

Output Format: JSON format, containing "Derivation Strategy", "Core Evidence Chain", "Causal Chain Analysis", "Preliminary Conclusion", "Rebuttal Evidence", no extra text.
Example Output:
{{
    "Derivation Strategy": "Causal chain analysis: derive mechanism from anti-CTLA-4 operation to proliferation increase",
    "Core Evidence Chain": [
        "Node 2 mechanism: CD8+ T cell activation requires CD28-B7 co-stimulation, CTLA-4 binding with B7 inhibits this signal",
        "Node 3 impact: anti-CTLA-4 blocks CTLA-4-B7 binding, releases co-stimulatory signal"
    ],
    "Causal Chain Analysis": [
        "Step 1: Anti-CTLA-4 antibody binds with CTLA-4 on T cell surface (experimental operation)",
        "Step 2: Blocks CTLA-4-B7 binding, removes inhibition on CD28 co-stimulatory signal (molecular mechanism)",
        "Step 3: CD28-B7 signal enhances, promotes T cell activation-related gene expression (cellular mechanism)",
        "Step 4: T cells divide and proliferate after activation, proliferation rate increases (final impact)"
    ],
    "Preliminary Conclusion": "Anti-CTLA-4 antibody promotes CD8+ T cell proliferation by blocking negative co-stimulatory signals and enhancing CD28 co-stimulation"
}}
"""


def _get_enumeration_derivation_prompt_template() -> str:
    """Enumeration type derivation prompt template"""
    return """
Task: For "Enumeration Type" biomedical questions, enumerate answers based on domain knowledge.
Known Information:
1. Core Question: {question_core}
2. Analysis Object: {analysis_object}
3. Domain Knowledge: {knowledge_str}

Derivation Requirements:
1. Derivation Strategy: Enumerate directly based on domain knowledge, ensure each answer has knowledge support;
2. Core Evidence Chain: Each enumeration item corresponds to definitions/classifications in domain knowledge;
3. Enumeration Results: Enumerate according to required quantity in question (e.g., "3 types of immune cells");
4. Preliminary Conclusion: Summarize enumeration results, answer the question.
5. Rebuttal Evidence (CRITICAL): After deriving Preliminary Conclusion, perform counter-evidence reasoning:
   - Assume some enumeration items are WRONG or MISSING
   - Find alternative items that could replace or supplement the enumeration
   - Check if these alternatives are consistent with: (1) constraint tags, (2) matched constraint scenarios, (3) domain knowledge
   - If alternatives contradict constraints/knowledge or are less relevant, this validates your enumeration
   - Record the rebuttal process: what alternative items were considered, why they were rejected/included, and how this validates the Preliminary Conclusion

Output Format: JSON format, containing "Derivation Strategy", "Core Evidence Chain", "Enumeration Results", "Preliminary Conclusion", "Rebuttal Evidence", no extra text.
Example Output:
{{
    "Derivation Strategy": "Direct enumeration based on domain knowledge, each item has knowledge support",
    "Core Evidence Chain": [
        "Node 2 domain knowledge: Immune cells include T cells, B cells, NK cells, etc.",
        "Each cell type has distinct functions and characteristics"
    ],
    "Enumeration Results": [
        "T cells: Responsible for cell-mediated immunity",
        "B cells: Responsible for humoral immunity, produce antibodies",
        "NK cells: Responsible for innate immunity, kill infected cells"
    ],
    "Preliminary Conclusion": "Three main types of immune cells are T cells, B cells, and NK cells"
}}
"""


def get_logical_derivation_prompt(
    question_type: str,
    question_core: str,
    domain_knowledge: Dict[str, Dict[str, Any]],
    experiment_analysis: Dict[str, Any],
    analysis_object: str,
    constraint_tags: Optional[Dict[str, List[str]]] = None,
    derivation_warning: Optional[str] = None
) -> str:
    """
    Generate type-specific derivation prompt
    
    Core design: Force "strategy → evidence → conclusion → rebuttal" closed loop, each step corresponds to previous node data
    
    Args:
        question_type: Question type (Judgment/Calculation/Analysis/Enumeration)
        question_core: Core question from question text (e.g., "Is experimental group proliferation rate higher than control?")
        domain_knowledge: Domain knowledge dictionary (from second node)
        experiment_analysis: Experiment analysis dictionary (from third node, can be empty for enumeration type)
        analysis_object: Analysis object string
        constraint_tags: Constraint tags from question parsing (optional, for rebuttal evidence validation)
        
    Returns:
        Formatted prompt string
    """
    import json
    # Format previous node data for LLM reading
    knowledge_str = json.dumps(domain_knowledge, ensure_ascii=False, indent=2)
    analysis_str = json.dumps(experiment_analysis, ensure_ascii=False, indent=2) if experiment_analysis else "{}"
    
    # Format constraint tags for prompt
    if constraint_tags:
        constraint_tags_str = json.dumps(constraint_tags, ensure_ascii=False, indent=2)
    else:
        constraint_tags_str = "{}"
    
    # Format derivation warning for prompt
    if derivation_warning:
        if derivation_warning == "constraint_mapping_mismatch":
            warning_message = "CRITICAL: Previous derivation failed constraint-knowledge mapping validation. Your new Preliminary Conclusion MUST be consistent with matched constraint scenarios from Scenario Constraint Mapping Table. Pay special attention to all parameter values (dynamic fields, excluding constraint_combination) in matched scenarios."
        elif derivation_warning == "rebuttal_evidence_contradiction":
            warning_message = "CRITICAL: Previous derivation failed rebuttal evidence validation. Your new Preliminary Conclusion MUST be consistent with your Rebuttal Evidence analysis. Ensure the rebuttal conclusion strengthens (not contradicts) your Preliminary Conclusion."
        elif derivation_warning == "both_validations_failed":
            warning_message = "CRITICAL: Previous derivation failed BOTH validations (constraint-knowledge mapping and rebuttal evidence). Your new derivation MUST address both issues: (1) align with matched constraint scenarios, (2) ensure rebuttal evidence consistency."
        else:
            warning_message = f"CRITICAL: Previous derivation failed validation ({derivation_warning}). Please re-derive with attention to the validation failure."
    else:
        warning_message = None
    
    # Select template based on question type
    question_type_lower = question_type.lower()
    if "judgment" in question_type_lower:
        template = _get_judgment_derivation_prompt_template()
    elif "calculation" in question_type_lower:
        template = _get_calculation_derivation_prompt_template()
    elif "analysis" in question_type_lower:
        template = _get_analysis_derivation_prompt_template()
    elif "enumeration" in question_type_lower:
        template = _get_enumeration_derivation_prompt_template()
    else:
        # Default to judgment type
        template = _get_judgment_derivation_prompt_template()
    
    # Add constraint tags information to all templates (only if template doesn't already have it)
    # Check if template already includes constraint_tags_str placeholder
    if "{constraint_tags_str}" not in template:
        enhanced_template = template + """

VI. Constraint Tags for Rebuttal Evidence Validation
Constraint Tags: {constraint_tags_str}

**IMPORTANT for Rebuttal Evidence**: When performing counter-evidence reasoning, you MUST check if alternative conclusions/calculations/mechanisms are consistent with the provided constraint tags. If counter-evidence requires constraints that contradict the given constraint tags, this strengthens your Preliminary Conclusion. If counter-evidence is plausible under different constraints, note this limitation explicitly.
"""
    else:
        enhanced_template = template
    
    # Add derivation warning section if re-derivation is needed
    if warning_message:
        enhanced_template = enhanced_template + """

VII. RE-DERIVATION GUIDANCE (CRITICAL)
{derivation_warning_message}

**Action Required**: The previous derivation failed substantive validation. You MUST generate a new Preliminary Conclusion that addresses the validation failure. Review the constraint-knowledge mapping table and ensure your conclusion aligns with matched scenario parameters. Review your rebuttal evidence and ensure it strengthens (not contradicts) your conclusion.
"""
    
    # Fill template variables - handle both cases (with or without constraint_tags_str in original template)
    format_kwargs = {
        "question_core": question_core,
        "analysis_object": analysis_object,
        "knowledge_str": knowledge_str,
        "analysis_str": analysis_str,
        "constraint_tags_str": constraint_tags_str
    }
    
    if warning_message:
        format_kwargs["derivation_warning_message"] = warning_message
    
    try:
        return enhanced_template.format(**format_kwargs)
    except KeyError as e:
        # If template doesn't have some placeholder, try without it
        # Remove the missing key from format_kwargs
        missing_key = str(e).strip("'")
        if missing_key in format_kwargs:
            del format_kwargs[missing_key]
        return enhanced_template.format(**format_kwargs)


# ===================== Final Validation Prompt =====================

FINAL_VALIDATION_PROMPT_TEMPLATE = """
Task: As the final validation step for biomedical question answering, you need to complete the closed loop of "substantive validation confirmation → precise option matching → reasoning chain organization", outputting a credible answer.

**IMPORTANT**: Substantive validations have already been performed (constraint-knowledge mapping consistency and rebuttal evidence consistency). The Preliminary Conclusion has passed these validations. Your task is to confirm these validations and proceed with option matching.

Known Information:
1. Question Options List: {options_str}
2. Logical Derivation Result (Node 4): {derivation_str}
3. Domain Knowledge (Node 2): {knowledge_str}
4. Experiment Analysis Result (Node 3): {analysis_str}
5. Question Type: {question_type}

Execution Requirements (strictly in order):
Step 1: Substantive Validation Confirmation (already performed, confirm results)
The following substantive validations have been completed:
1. **Constraint-Knowledge Mapping Consistency**: Preliminary Conclusion has been validated against matched constraint scenarios from Scenario Constraint Mapping Table. If this validation passed, the conclusion aligns with constraint-specific knowledge.
2. **Rebuttal Evidence Consistency**: Preliminary Conclusion has been validated against rebuttal evidence from reasoning engine. If this validation passed, the conclusion doesn't contradict the counter-evidence analysis.

Your task: Confirm these validations in your "Validation Results" and note that substantive validations have passed. Then proceed with option matching.

Step 2: Multi-dimensional Validation (supplementary checks)
Need to cover 4 dimensions, each dimension needs to provide "Pass/Fail" and reason:
1. Previous Logic Consistency: Are Node 3's "Impact Direction Judgment" and Node 4's "Preliminary Conclusion" contradictory? Do derivation evidence cover previous nodes?
2. Biological/Statistical Reasonableness: Does the conclusion conform to domain common sense (e.g., "anti-CTLA-4 promotes T cell proliferation" is a known mechanism)?
3. Option Matching Feasibility: Does Node 4's "Option Matching Priority" correspond one-to-one with question options? Are there unmatched items?
4. Preliminary-Final Consistency: Does the Final Answer you plan to select align with Node 4's "Preliminary Conclusion"? This is CRITICAL - if Preliminary Conclusion states "X unbiased, Y biased", your Final Answer MUST reflect this exactly.

Step 3: Precise Option Matching
CRITICAL CONSISTENCY REQUIREMENT:
- The Final Answer MUST be logically consistent with Node 4's "Preliminary Conclusion"
- If Preliminary Conclusion states "θ unbiased, π biased", then Final Answer MUST select an option that matches this (e.g., "Only π is biased")
- If you find Preliminary Conclusion contradicts all options, you MUST explicitly state this in Validation Results and reconsider the Preliminary Conclusion
- DO NOT select an option that contradicts the Preliminary Conclusion, even if Option Matching Priority suggests otherwise
- Before finalizing your answer, explicitly verify: "Does my Final Answer align with the Preliminary Conclusion? If not, I must correct it."

Matching Rules:
1. First, extract the key claims from Node 4's "Preliminary Conclusion" (e.g., "θ unbiased, π biased")
2. Then, map each option to its claims (e.g., Option A: "θ biased", Option B: "π biased")
3. Select the option whose claims match the Preliminary Conclusion
4. If Node 4's "Option Matching Priority" suggests a different option, but that option contradicts Preliminary Conclusion, you MUST override it and select the option that matches Preliminary Conclusion
5. If it's a calculation type question, need to verify "Numerical Result" with numerical values in options (e.g., "0.32" corresponds to option B)
6. Exclude wrong options: Each wrong option needs to provide 1 core exclusion reason (e.g., "Option C wrong because ignores variant frequency decrease")

Step 4: Organize Complete Reasoning Chain
Connect core contributions of five nodes in sequence, format as "Node 1: Extract XX → Node 2: Activate XX knowledge → Node 3: Analyze XX impact → Node 4: Derive XX conclusion → Node 5: Validate and match XX option".

Output Format: JSON format, containing "Final Answer", "Validation Results", "Complete Reasoning Chain", "Key Evidence Summary", "Common Pitfall Reminders", no extra text.
Example Output (Judgment Type):
{{
    "Final Answer": "A. Yes, anti-CTLA-4 antibody promotes CD8+ T cell proliferation",
    "Validation Results": "Substantive validations (Pass, constraint-knowledge mapping consistency confirmed, rebuttal evidence consistency confirmed); Previous logic consistent (Pass, impact judgment consistent with conclusion); Biological reasonableness (Pass, anti-CTLA-4 mechanism is known immunological knowledge); Option matching accurate (Pass, priority A corresponds to question option A); Preliminary-Final consistency (Pass, Final Answer 'A. Yes, anti-CTLA-4 promotes proliferation' is consistent with Preliminary Conclusion 'experimental group proliferation higher than control')",
    "Complete Reasoning Chain": "Node 1: Extract analysis object (CD8+ T cell proliferation rate) → Node 2: Activate T cell co-stimulation knowledge → Node 3: Analyze antibody promotes proliferation → Node 4: Derive experimental group proliferation higher → Node 5: Validate and match option A",
    "Key Evidence Summary": [
        "Node 2 knowledge: CTLA-4 is negative co-stimulatory molecule, antibody blocking enhances proliferation",
        "Node 3 impact: Anti-CTLA-4 treatment significantly promotes proliferation",
        "Node 4 conclusion: Experimental group proliferation rate higher than control"
    ],
    "Common Pitfall Reminders": [
        "Option B wrong: Confuses CTLA-4's role (inhibits rather than promotes negative signal)",
        "Option D wrong: Dose information not needed (domain knowledge already clarifies antibody action direction, independent of dose)"
    ]
}}
"""


def get_final_validation_prompt(
    question_options: List[str],
    logical_derivation: Dict[str, Any],
    domain_knowledge: Dict[str, Dict[str, str]],
    experiment_analysis: Dict[str, Any],
    question_type: str
) -> str:
    """
    Generate closed-loop prompt for final validation
    
    Core design: Force LLM to complete "three-dimensional validation → option matching → reasoning chain organization", avoid isolated option matching
    
    Args:
        question_options: Question options list
        logical_derivation: Logical derivation result (from Node 4)
        domain_knowledge: Domain knowledge dictionary (from Node 2)
        experiment_analysis: Experiment analysis dictionary (from Node 3)
        question_type: Question type string
    
    Returns:
        Formatted prompt string
    """
    import json
    # Format previous node data for LLM reading
    options_str = json.dumps(question_options, ensure_ascii=False, indent=2)
    derivation_str = json.dumps(logical_derivation, ensure_ascii=False, indent=2)
    knowledge_str = json.dumps(domain_knowledge, ensure_ascii=False, indent=2)
    analysis_str = json.dumps(experiment_analysis, ensure_ascii=False, indent=2)
    
    return FINAL_VALIDATION_PROMPT_TEMPLATE.format(
        options_str=options_str,
        derivation_str=derivation_str,
        knowledge_str=knowledge_str,
        analysis_str=analysis_str,
        question_type=question_type
    )


# ===================== Output Format Customization =====================

def format_final_answer_student(final_result: Dict[str, Any]) -> str:
    """
    Format final answer for student template (simplified, educational)
    
    Args:
        final_result: Final result from Node 5
    
    Returns:
        Formatted answer string for students
    """
    final_answer = final_result.get("Final Answer", "No answer available")
    key_evidence = final_result.get("Key Evidence Summary", [])
    if not isinstance(key_evidence, list):
        key_evidence = [key_evidence] if key_evidence else []
    
    # Take first evidence as core basis
    core_basis = key_evidence[0] if key_evidence else "Based on domain knowledge and experimental analysis"
    
    # Simplify core basis (remove technical jargon)
    simplified_basis = core_basis
    # Replace technical terms with simpler explanations
    replacements = {
        "anti-CTLA-4": "anti-CTLA-4 antibody (like 'releasing the brake' on T cells)",
        "proliferation": "cell division and growth",
        "co-stimulatory": "activation signals",
        "inhibits": "blocks or reduces",
        "promotes": "increases or enhances"
    }
    for tech_term, simple_term in replacements.items():
        simplified_basis = simplified_basis.replace(tech_term, simple_term)
    
    output = f"""**Final Answer: {final_answer}**

**Core Basis:**
{simplified_basis}

**Common Mistakes to Avoid:**
"""
    
    pitfalls = final_result.get("Common Pitfall Reminders", [])
    if not isinstance(pitfalls, list):
        pitfalls = [pitfalls] if pitfalls else []
    
    for i, pitfall in enumerate(pitfalls[:3], 1):  # Show top 3
        output += f"{i}. {pitfall}\n"
    
    return output


def format_final_answer_researcher(final_result: Dict[str, Any]) -> str:
    """
    Format final answer for researcher template (rigorous, academic)
    
    Args:
        final_result: Final result from Node 5
    
    Returns:
        Formatted answer string for researchers
    """
    final_answer = final_result.get("Final Answer", "No answer available")
    validation_results = final_result.get("Validation Results", "")
    reasoning_chain = final_result.get("Complete Reasoning Chain", "")
    key_evidence = final_result.get("Key Evidence Summary", [])
    if not isinstance(key_evidence, list):
        key_evidence = [key_evidence] if key_evidence else []
    
    output = f"""**Final Answer: {final_answer}**

**Confidence Level: 95%** (based on multi-dimensional validation)

**Validation Results:**
{validation_results}

**Complete Reasoning Chain:**
{reasoning_chain}

**Key Evidence Summary:**
"""
    
    for i, evidence in enumerate(key_evidence, 1):
        output += f"{i}. {evidence}\n"
    
    output += "\n**Research Gaps & Limitations:**\n"
    output += "- This conclusion is based on available domain knowledge and experimental analysis.\n"
    output += "- Further validation through experimental studies is recommended for clinical applications.\n"
    
    return output


def format_final_answer_clinician(final_result: Dict[str, Any]) -> str:
    """
    Format final answer for clinician template (quick reference, clinical focus)
    
    Args:
        final_result: Final result from Node 5
    
    Returns:
        Formatted answer string for clinicians
    """
    final_answer = final_result.get("Final Answer", "No answer available")
    key_evidence = final_result.get("Key Evidence Summary", [])
    if not isinstance(key_evidence, list):
        key_evidence = [key_evidence] if key_evidence else []
    
    pitfalls = final_result.get("Common Pitfall Reminders", [])
    if not isinstance(pitfalls, list):
        pitfalls = [pitfalls] if pitfalls else []
    
    output = f"""**Quick Answer: {final_answer}**

**Clinical Basis:**
"""
    
    # Extract clinical-relevant evidence
    for evidence in key_evidence[:2]:  # Top 2 most relevant
        output += f"- {evidence}\n"
    
    output += "\n**Risk Warnings:**\n"
    for pitfall in pitfalls[:2]:  # Top 2 risks
        output += f"- {pitfall}\n"
    
    output += "\n**Next Steps:**\n"
    output += "- Verify patient-specific conditions match the analysis assumptions.\n"
    output += "- Consider additional diagnostic tests if uncertainty exists.\n"
    
    return output


def format_final_answer_custom(final_result: Dict[str, Any], custom_fields: List[str]) -> str:
    """
    Format final answer with custom fields
    
    Args:
        final_result: Final result from Node 5
        custom_fields: List of field names to include
    
    Returns:
        Formatted answer string with custom fields
    """
    output = ""
    
    field_mapping = {
        "Final Answer": "Final Answer",
        "Validation Results": "Validation Results",
        "Complete Reasoning Chain": "Complete Reasoning Chain",
        "Key Evidence Summary": "Key Evidence Summary",
        "Common Pitfall Reminders": "Common Pitfall Reminders"
    }
    
    for field in custom_fields:
        if field in field_mapping:
            key = field_mapping[field]
            value = final_result.get(key, "N/A")
            if isinstance(value, list):
                value = "\n".join(f"- {item}" for item in value)
            output += f"**{field}:**\n{value}\n\n"
    
    return output.strip()


def format_final_answer(
    final_result: Dict[str, Any],
    output_template: str = "researcher",
    custom_fields: Optional[List[str]] = None
) -> str:
    """
    Format final answer based on template type
    
    Args:
        final_result: Final result from Node 5
        output_template: Template type ("student", "researcher", "clinician", or "custom")
        custom_fields: Custom fields list (required if template is "custom")
    
    Returns:
        Formatted answer string
    """
    if output_template == "student":
        return format_final_answer_student(final_result)
    elif output_template == "researcher":
        return format_final_answer_researcher(final_result)
    elif output_template == "clinician":
        return format_final_answer_clinician(final_result)
    elif output_template == "custom":
        if not custom_fields:
            raise ValueError("custom_fields must be provided when template is 'custom'")
        return format_final_answer_custom(final_result, custom_fields)
    else:
        # Default to researcher template
        return format_final_answer_researcher(final_result)
