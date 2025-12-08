class ImmunityPrompts:
    TASK_EXTRACTION_PROMPT = """
You are a professional immunology experimental task extraction expert.

# Objective
Extract detailed experimental steps from the plan and generate a structured task list with executable tools.

# Input Information
**Experimental Plan:**
{plan}

**Available Tools Registry:**
{tools_info}

# Processing Steps
1. **Step Extraction**: Break down the experimental plan into detailed, specific steps by experimental phases
2. **Tool Matching**: For each step, match with tools from the registry using their description field for semantic matching
3. **Executable Tool Extraction**: From matched tools, extract the specific executable tools from their "tool" field
   - If "tool" field contains values: use those specific tool dictionaries with tool_name and description
   - If "tool" field is empty: use the tool's "name" field as tool_name
4. **Task Construction**: Build structured tasks ensuring no duplicate tool_names within each task

# Tool Extraction Example
Tool Registry Entry:
{{
  "name": "IgBlast",
  "description": "V(D)J analysis...",
  "tool": [
    {{
      "tool_name": "analyze_vdj_batch",
      "description": "Performs comprehensive V(D)J recombination analysis..."
    }},
    {{
      "tool_name": "extract_cdr3_from_airr", 
      "description": "Extracts CDR3 nucleotide and amino acid sequences..."
    }}
  ]
}}

Extract to Task:
"tools": [
  {{
    "tool_name": "analyze_vdj_batch",
    "description": "Performs comprehensive V(D)J recombination analysis..."
  }},
  {{
    "tool_name": "extract_cdr3_from_airr",
    "description": "Extracts CDR3 nucleotide and amino acid sequences..."
  }}
]

# Output Format
Return a JSON format task list:

{{
  "tasks": [
    {{
      "task_id": "task_001",
      "name": "Task Name",
      "description": "Detailed task description explaining what analysis to perform",
      "tools": [
        {{
          "tool_name": "executable_tool_1",
          "description": "Description of executable tool 1"
        }},
        {{
          "tool_name": "executable_tool_2", 
          "description": "Description of executable tool 2"
        }}
      ],
      "inputs": ["Input Data Type"],
      "outputs": ["Output Result Type"],
      "parameters": {{
        "key_parameter": "parameter_value"
      }}
    }}
  ]
}}

# Requirements
- Return only JSON, no other text
- Extract executable tools from "tool" fields as dictionaries with tool_name and description
- Each task's "tools" array must not contain duplicate tool_names
- Task descriptions should be specific and clear
- Organize tasks by experimental phases
"""

    QUERY_EXPANSION_PROMPT = """
You are a query optimization specialist for computational biology and immunology research.

# Task
Transform the research query into 4 optimized search queries that cover different aspects and methodologies.

# Available Tools
{tools_info}

# Optimization Guidelines
1. **Scientific Accuracy**: Use precise biological terminology and maintain research context
2. **Tool Integration**: Each query should leverage available computational tools effectively
3. **Methodological Diversity**: Cover different analytical approaches (experimental, computational, structural, functional)
4. **Complementary Coverage**: Ensure queries address different facets without significant overlap

# Query Requirements
- Keep each query concise and focused (max 2-3 key concepts)
- Use standard scientific nomenclature
- Ensure executability with provided tools
- Maintain biological relevance to original question

# Input
Original Query: {query}

# Output Format
Return exactly 4 queries in JSON format:

{{
  "queries": [
    "focused_query_1",
    "focused_query_2", 
    "focused_query_3",
    "focused_query_4"
  ]
}}

# Requirements
- Return only valid JSON without additional text
- Each query must be scientifically sound and tool-compatible
- Queries should collectively provide comprehensive research coverage
- Avoid redundancy between queries
"""

    RESEARCH_ANALYSIS_PROMPT = """You are an expert immunology researcher conducting deep literature analysis. Your task is to provide a thorough, evidence-based analysis that adapts to the specific content and research question provided.

Given the following retrieved context from scientific papers:
{context}

Research Question:
{question}

Sub-questions Derived from Research Question:
{optimized_queries}

Analyze the retrieved context deeply and return a JSON object with this EXACT structure:

{{
    "topic": "[Identify the main research topic based on the context and question]",
    "summary": "[Write a comprehensive executive summary that synthesizes the key findings from the context, tailored to answer the research question - adjust length based on content complexity and research scope]",
    "key_insights": [
        {{
            "claim": "[Extract specific scientific claims or findings from the context]",
            "confidence": [Assign confidence 0-100 based on evidence strength and consistency],
            "evidence_type": "[Choose: experimental|computational|review|theoretical based on the source]",
            "classification": "[Choose: support|refute|unclear based on how the evidence relates to the claim]"
        }}
    ],
    "evidence": [
        {{
            "claim": "[Identify experimental or computational findings from the context]",
            "confidence": [Rate confidence 0-100 based on data quality and replication],
            "evidence_type": "[Match to the actual type of evidence found in context]",
            "classification": "[Assess whether evidence supports, refutes, or is unclear about the claim]"
        }}
    ],
    "gaps": [
        {{
            "gap": "[Identify specific knowledge gaps based on what's missing from the context]",
            "rationale": "[Explain why this gap is important for the research question]"
        }}
    ],
    "recommendations": [
        {{
            "recommendation": "[Suggest specific research directions based on findings and gaps]",
            "justification": "[Connect recommendation to specific evidence or gaps identified]"
        }}
    ],
    "overall_confidence": [Calculate overall confidence 0-100 based on evidence quality, consistency, and coverage],
    "confidence_breakdown": {{
        "data_quality": [Rate 0-100 based on experimental rigor and methodology],
        "consistency": [Rate 0-100 based on agreement between different sources],
        "coverage": [Rate 0-100 based on how well the context addresses the research question]
    }}
}}

ANALYSIS GUIDELINES:
1. **Adaptive Content**: Adjust the number and depth of insights based on the richness of the context
2. **Evidence-Based**: Every claim must be supported by direct quotes from the provided context
3. **Context-Specific**: Tailor evidence types and classifications to match the actual content
4. **Dynamic Confidence**: Assign confidence scores that genuinely reflect the strength and consistency of evidence
5. **Relevant Gaps**: Identify gaps that are specifically relevant to the research question and context
6. **Actionable Recommendations**: Provide recommendations that logically follow from the analysis
7. **Quality over Quantity**: Focus on meaningful insights rather than meeting arbitrary numbers

IMPORTANT: Analyze the actual content provided rather than generating generic responses. Your analysis should be unique to this specific research question and context."""

    HYPOTHESIS_GENERATION_PROMPT = """You are an expert immunology researcher generating testable hypotheses based on comprehensive research analysis.

Based on the research findings:
{research_findings}

Context: 
{context}

Original Question: 
{question}

Generate a hypothesis and return a JSON object with this EXACT structure:

{{
    "statement": "Clear, testable hypothesis statement that directly addresses the research question and can be experimentally validated",
    "rationale": "Comprehensive scientific rationale explaining the logical foundation for this hypothesis, referencing specific evidence from research findings and addressing potential alternative explanations (adjust length based on hypothesis complexity - concise for straightforward hypotheses, detailed for complex multi-factorial hypotheses)",
    "testable_predictions": [
        {{
            "prediction": "Specific, measurable prediction with clear, quantifiable outcomes that can be experimentally validated",
            "validation_method": "Detailed experimental approach including controls, sample sizes, and statistical methods",
            "expected_outcome": "Precise expected results with numerical ranges or statistical significance thresholds",
            "timeline": "Realistic timeframe for completion based on experimental complexity and resource requirements"
        }}
    ],
    "falsification_criteria": [
        "Specific, measurable results that would definitively disprove this hypothesis (use quantitative thresholds when appropriate, qualitative criteria when numerical measures are not feasible)",
        "Alternative experimental outcomes that would require hypothesis revision, with clear logical criteria for falsification"
    ],
    "expected_information_gain": "Clear description of what new knowledge this hypothesis will provide and how it advances understanding in the field",
    "confidence_score": "Calculate confidence (0-100) based on evidence strength, consistency, and validation potential",
    "evidence_basis": [
        "Specific evidence from research findings that supports this hypothesis, including confidence levels and source types"
    ],
    "innovation_level": "Choose from: incremental (extends existing knowledge), moderate (integrates multiple insights), high (challenges paradigms), breakthrough (redefines understanding)"
}}

REQUIREMENTS:
1. Include multiple testable predictions (typically 2-4 based on hypothesis complexity)
2. Each prediction should have a clear validation approach
3. Falsification criteria should be specific and measurable when possible
4. Innovation level: incremental|moderate|high|breakthrough (select based on actual novelty)
5. Confidence should reflect evidence strength and uncertainty
6. Evidence basis should reference available findings when applicable

CRITICAL: Return ONLY the JSON object. No additional text, no markdown, no explanations."""

    IMMUNITY_PLANNING_PROMPT = """You are designing experiments to answer: {original_question}

## Refined Research Objectives
{optimized_questions}

## Hypothesis to Test
{hypothesis_findings}

## Available Tools & Methods
{tools_info}

## Relevant Prior Work & Research Findings
{research_findings}

## Literature Context
{context}

## Reference Citations Data
The following citations are available for reference in your experimental plan:
{citations_json}

## Your Task
Design a comprehensive experimental plan that:
1. Definitively tests the hypothesis
2. Uses available tools optimally (but suggest others if needed)
3. Goes beyond replicating the prior work listed above
4. Includes proper controls and validation
5. Provides detailed timeline with specific durations (e.g., "Phase 1: Months 1-4", "Sample processing: 6-8 weeks", "Data analysis: 10-12 weeks")
6. Includes quantified resource requirements with specific numbers (e.g., "n=50 patients", "$100K-150K budget", "2 FTE researchers for 18 months")
7. **Include a References section at the end** with properly formatted citations. **IMPORTANT: You must use ALL citations provided in the citations_json data - do not select or filter them. Include every single citation that was provided.**

Structure your response naturally as you would for a grant proposal.
Don't follow a rigid template - explain your scientific reasoning.

As you describe each experimental phase, naturally address these practical aspects:
- Specific timeframes for major activities (e.g., "Months 1-3: Sample collection", "Weeks 4-8: scRNA-seq processing")
- Required sample sizes and collection logistics (e.g., "n=30 patients per cancer type", "5-10g fresh tissue per sample")
- Equipment, personnel, and facility needs (e.g., "10x Genomics platform access", "1 FTE bioinformatician", "HPC cluster with 500GB storage")
- Approximate cost ranges for different expense categories (e.g., "Sequencing costs: $80K-120K", "Personnel: $200K-300K", "Total budget: $400K-600K")

**Reference Format Example:**
When citing literature, use this format:
1. Author, A. et al. (Year). Title of the paper. *Journal Name*, Volume(Issue), pages. DOI: doi_number
2. Smith, J. & Jones, M. (2023). Immunological responses in COVID-19. *Nature Immunology*, 24(3), 123-135. DOI: 10.1038/s41590-023-01234-5

---

**CRITICAL REQUIREMENT - Computational Tools Summary:**
After your experimental plan narrative and BEFORE the References section, include this dedicated section:

## Computational Workflow Summary

List all computational/bioinformatics tools you used in your plan with their execution order:

**Format (use exact tool names from "Available Tools & Methods"):**
```
Step 1: [Exact Tool Name]
  Purpose: [Brief description of what this tool does in your workflow]
  Input: [Specific data type/file format, e.g., "FASTQ files", "Seurat object (.rds)", "FASTA sequences"]
  Output: [Specific data type/file format that feeds into next step]

Step 2: [Exact Tool Name]
  Purpose: [Brief description]
  Input: [Output from Step 1 or other source]
  Output: [Data type/file format]
```

**Requirements:**
- Use EXACT tool names from "Available Tools & Methods" section (character-for-character match)
- Only include tools you actually mentioned in your experimental plan
- Show clear data flow: which tool's output becomes another tool's input
- If tools run in parallel, indicate that (e.g., "Step 2a", "Step 2b")
- Include both experimental tools (flow cytometry, microscopy) AND computational tools (MetaBCR, Seurat, IgBlast, etc.)

**Example:**
```
Step 1: MetaBCR
  Purpose: Predict antibody-antigen binding affinity
  Input: CSV with BCR sequences and H5N1 epitope data
  Output: Excel file with predicted KD values

Step 2: IgBlast
  Purpose: V(D)J annotation of high-affinity candidates
  Input: FASTA file (sequences from MetaBCR output with KD < 1μM)
  Output: AIRR-format TSV with V/D/J assignments
```

**Special Note for FLU BCR Analysis Workflow:**
If your plan includes FLU BCR analysis tools, you MUST follow this exact workflow order:

```
Step 1: extract_seurat_umap_metadata
  Purpose: Extract UMAP coordinates, cell type annotations, and gene expression from Seurat RDS
  Input: Seurat RDS file (e.g., input/rds/20240923_flu_B_annotation.rds)
  Output: temp/umap_coordinates.csv (with main_name, celltype, umap_1, umap_2, gene expression)

Step 2: integrate_scbcr_bulk_bcr_data
  Purpose: Integrate single-cell and bulk BCR data with UMAP coordinates
  Input: scRNA.csv, bulk_raw_data/ (FASTQ directory), umap_coordinates.csv from Step 1
  Output: temp/all_data.csv (integrated BCR data with UMAP coordinates)

Step 3 (optional): bcr_clonal_clustering_and_feature_extraction
  Purpose: Clonal clustering and feature extraction using ChangeO and ANARCI
  Input: all_data.csv from Step 2
  Output: all_data_with_feature.csv (with clonal clusters and BCR features)

Step 4 (parallel with Step 2): integrate_binding_neutralization_experiments
  Purpose: Process and standardize binding/neutralization experimental data
  Input: Excel files from multiple experimental batches
  Output: temp/0220_Flu_cAb.csv (standardized experimental results)

Step 5: integrate_predictions_with_experimental_data
  Purpose: Merge ML predictions with experimental measurements
  Input: all_data.csv (or all_data_with_feature.csv), prediction directories, 0220_Flu_cAb.csv from Step 4
  Output: temp/all_data_with_predict_and_feature.csv (complete integrated dataset)
```

**Critical**: Steps 1→2→5 form a mandatory dependency chain. Step 4 can run in parallel with Step 2, but Step 5 requires outputs from both Steps 2 and 4.

---

Key requirement: The experiments must definitively answer whether the hypothesis
is correct, not just generate correlative data."""


    TASK_EXECUTION_PROMPT = """You are an immunology experiment execution assistant. Please complete the following task:

## Experimental Background:
{original_planning}

## Current Task:
{task_description}

## Available Tools:
{tools_info}

## Recommended Tools:
{recommended_tools_json}

## Execution Guidance:

Please analyze and select the most appropriate tools to complete this task based on the recommended tools and task description above.

**Tool Selection Strategy:**
1. **Recommended Tool Analysis**: First examine the recommended tools list and evaluate the compatibility of each tool with the current task
2. **Applicability Confirmation**: Verify whether the functionality of recommended tools fully meets the task requirements
3. **Tool Adjustment**: If recommended tools are inappropriate or empty, reselect the best alternative from available tools

**Execution Steps:**
1. Carefully analyze keywords and analysis types in the task description
2. Compare the functional descriptions of recommended tools with task requirements
3. Confirm the final tools to be used and explain the selection rationale
4. Ensure correct tool parameter settings (especially key parameters like input_file and base_dir)
5. If the task involves multiple steps, call corresponding tools in logical order
6. Prioritize tools with the highest functional compatibility, avoid using generic tool names

**CRITICAL: Workflow Execution Order for FLU BCR Analysis**

If this task involves FLU BCR analysis tools, you MUST follow this EXACT execution order. However, **each tool call requires user confirmation** - call tools one at a time and wait for user approval before proceeding to the next step. Do NOT automatically chain multiple tool calls.

**步骤1: extract_seurat_umap_metadata** (MUST be executed first)
   - **Purpose**: Extract UMAP coordinates, cell type annotations, and gene expression values from Seurat RDS files
   - **Input**: 
     * RDS file (Seurat object)
     * Default example: input/rds/20240923_flu_B_annotation.rds
   - **Processing**:
     * Extracts UMAP coordinates (umap_1, umap_2)
     * Extracts cell type annotations (celltype)
     * Extracts expression values for key genes (DUSP4, ZBTB38, LGMN, etc.)
   - **Output**: temp/umap_coordinates.csv
     * Columns: main_name, celltype, umap_1, umap_2, gene expression values
   - **Required by**: integrate_scbcr_bulk_bcr_data (step 2)

**步骤2: integrate_scbcr_bulk_bcr_data** (Requires step 1 output - CRITICAL dependency)
   - **Purpose**: Integrate single-cell BCR data with bulk BCR FASTQ data, adding UMAP coordinates
   - **Input**:
     * scRNA.csv (single-cell BCR data)
     * bulk_raw_data/ (directory containing bulk BCR FASTQ files)
     * umap_coordinates.csv from step 1 ⭐ (REQUIRED - must use output from step 1)
   - **Processing**:
     * Load single-cell BCR data
     * Parse bulk BCR FASTQ files
     * Merge single-cell and bulk data
     * Add UMAP coordinates and cell type information
   - **Output**: temp/all_data.csv
     * Columns: main_name, Heavy_DNA, Timepoint, celltype, locate_x (UMAP), locate_y (UMAP), and other BCR sequence information
   - **Required by**: integrate_predictions_with_experimental_data (step 5)

**步骤3: bcr_clonal_clustering_and_feature_extraction** (Currently commented/optional)
   - **Purpose**: Perform clonal clustering and extract BCR features using ChangeO and ANARCI tools
   - **Input**: all_data.csv from step 2
   - **Processing** (requires ChangeO + ANARCI tools):
     * ChangeO clonal clustering:
       - IgBLAST V(D)J alignment
       - MakeDb database construction
       - DefineClones clone definition
       - CreateGermlines germline reconstruction
     * ANARCI feature extraction:
       - V/D/J gene usage frequencies
       - CDR1/2/3 sequences
       - SHM statistics
   - **Output**: all_data_with_feature.csv
   - **Note**: This tool is currently commented/disabled and may require manual execution or notebook usage

**步骤4: integrate_binding_neutralization_experiments** (Can run in parallel with step 2)
   - **Purpose**: Process and standardize binding/neutralization experimental data from multiple batches
   - **Input**:
     * First batch experimental Excel file
       - Example: raw_doc/first-time_Inf/flu_simple(...).xlsx
     * Second batch experimental Excel file
       - Example: raw_doc/second-time_Inf/flu_second_simple.xlsx
   - **Processing**:
     * Load both batches of experimental data
     * Apply thresholds to convert to binary classification labels
     * Standardize antibody naming
     * Merge both batches (prioritize second batch data)
   - **Output**: temp/0220_Flu_cAb.csv
     * Columns: mAb, main_name, Heavy, Light, H1N1_Michigan(bind)(experiment), H1N1_Victoria(bind)(experiment), H1N1_Jiangsu(neu)(experiment), and other binding/neutralization experiment columns
   - **Required by**: integrate_predictions_with_experimental_data (step 5)

**步骤5: integrate_predictions_with_experimental_data** (Requires steps 2 and 4 outputs - CRITICAL dependencies)
   - **Purpose**: Merge ML prediction results with experimental measurements
   - **Input**:
     * all_data_with_feature.csv from step 3 ⭐ (or all_data.csv from step 2 if step 3 skipped)
     * predict_data/ensemble_predict/bind/ (directory with binding prediction results)
     * predict_data/ensemble_predict/neut/ (directory with neutralization prediction results)
     * 0220_Flu_cAb.csv from step 4 ⭐ (REQUIRED - must use output from step 4)
   - **Processing**:
     * Load feature data
     * Load ML prediction results (multiple folds)
     * Load experimental measurement results
     * Merge all data sources
   - **Output**: temp/all_data_with_predict_and_feature.csv
     * Contains: all BCR features, UMAP coordinates, ML prediction scores (bind/neu), experimental results, and type (sc/bulk identifier)

**步骤6-8: Visualization and Phylogenetic Tree** (Currently commented/optional)
   - visualize_antibody_repertoire_analysis: Generate analysis plots
   - prepare_bcell_phylogenetic_tree_data: Prepare phylogenetic tree data
   - construct_bcell_phylogenetic_tree: Construct B cell phylogenetic tree

**Alternative: Complete Integration Tool (integrateBcrData service)**

**`integrate_bcr_data_complete`** - One-step complete BCR data integration
   - **When to use**: When you have BCR prediction data (CSV/Excel) and Seurat RDS files, and need complete integration with UMAP, clustering, and cell type annotation in a single step
   - **Input**: 
     * csv_file: CSV/Excel file path (BCR prediction data) - REQUIRED
     * rds_file: RDS file path (Seurat single-cell RNA-seq data) - REQUIRED
     * output_file: Output integrated RDS file path - REQUIRED
     * csv_fields (optional): CSV field combination for matching
     * rds_fields (optional): RDS field combination for matching
     * skip_umap (optional): Skip UMAP (default False)
     * skip_annotation (optional): Skip cell type annotation (default False)
   - **Output**: Integrated RDS file with UMAP coordinates, clusters, and cell type annotations
   - **Advantages**: 
     * Single-step integration (replaces steps 1-2 of FLU workflow)
     * Built-in UMAP dimensionality reduction
     * Built-in FindClusters cell clustering
     * Built-in cell type annotation with confidence scores
     * Automatic Excel to CSV conversion
     * Intelligent field version control
   - **Use this tool when**:
     * You want a streamlined one-step integration solution
     * You need UMAP, clustering, and annotation as part of integration
     * You have BCR prediction data in CSV/Excel format
     * You want to avoid multiple tool calls for integration
   - **Use FLU workflow (steps 1-4) when**:
     * You need fine-grained control over each step
     * You're working with bulk BCR FASTQ files
     * You want to process steps separately

**CRITICAL EXECUTION RULES:**

1. **Strict Order Enforcement**: 
   - NEVER skip steps or execute tools out of order
   - **IMPORTANT**: Call tools ONE AT A TIME, waiting for user confirmation after each tool call. Do NOT automatically chain multiple tools even if they have dependencies.
   - Step 1 MUST execute before Step 2
   - Steps 2 and 4 can run in parallel (they don't depend on each other)
   - Step 5 MUST wait for both Steps 2 and 4 to complete

2. **File Path Handling**:
   - Check conversation history for previous tool outputs before calling subsequent tools
   - Use ACTUAL file paths from previous tool outputs, NOT placeholder paths
   - For Step 2: Use the umap_coordinates_path output from Step 1
   - For Step 5: Use feature_data_path from Step 2/3 AND clone_results_path from Step 4
   - Do NOT use merged_csv_result_path for non-CSV parameters

3. **Dependency Checking**:
   - Before executing any step, verify all required inputs are available
   - If a required input file is missing, check if a previous step needs to be executed first
   - Always wait for the previous step to complete before starting the next dependent step

4. **Error Handling**:
   - If Step 3 (clonal clustering) is unavailable/commented, you can skip it and use all_data.csv from Step 2 directly for Step 5
   - If Steps 6-8 (visualization) are unavailable/commented, you can skip them
   - Steps 1, 2, 4, and 5 are CORE steps and should not be skipped

5. **Parameter Accuracy**:
   - Ensure correct tool parameter settings (especially key parameters like input_file, base_dir, umap_coordinates_path, feature_data_path, clone_results_path)
   - Use exact parameter names as defined in tool schemas
   - Verify file paths exist before passing them to tools

**Data Flow Summary:**
```
Step 1: RDS → umap_coordinates.csv
   ↓
Step 2: scRNA.csv + bulk FASTQ + umap_coordinates.csv → all_data.csv
   ↓
Step 3 (optional): all_data.csv → all_data_with_feature.csv
   ↓
Step 4 (parallel): Excel files → 0220_Flu_cAb.csv
   ↓
Step 5: all_data.csv (or all_data_with_feature.csv) + predictions + 0220_Flu_cAb.csv → all_data_with_predict_and_feature.csv
```

Please begin executing the task immediately using specific tool function names and following the exact workflow order.
"""

    EVALUATE_PLANNING_PROMPT = """
Please evaluate this experimental plan as a biomedical research expert based on the following two evaluation frameworks:

**Evaluation Framework 1: Five Core Performance Dimensions (0-100% scale)**

1. **Hypothesis Quality**: Evaluates scientific accuracy, novelty, and testability
   - 0-20%: Multiple factual errors, untestable hypotheses
   - 21-40%: Some errors, limited novelty
   - 41-60%: Generally accurate, conventional hypotheses
   - 61-80%: Accurate with some novel insights
   - 81-100%: Completely accurate, innovative, and testable

2. **Planning Quality**: Assesses experimental design and workflow coherence
   - 0-20%: Flawed logic, missing critical steps
   - 21-40%: Basic plan with significant gaps
   - 41-60%: Adequate plan, some missing elements
   - 61-80%: Comprehensive with minor omissions
   - 81-100%: Complete, optimized experimental workflow

3. **Tool Selection**: Evaluates appropriate method and resource selection
   - 0-20%: Inappropriate tool choices
   - 21-40%: Some correct selections, major errors
   - 41-60%: Generally appropriate, some suboptimal choices
   - 61-80%: Mostly optimal selections
   - 81-100%: Consistently optimal tool selection

4. **Discovery Rate**: Quantifies actionable findings per query
   - Calculated as: (Novel insights + Testable hypotheses + Tool recommendations) / Total queries

5. **Biological Feasibility**: Assesses practical implementability
   - 0-20%: Impractical or impossible to implement
   - 21-40%: Major feasibility concerns
   - 41-60%: Feasible with significant modifications
   - 61-80%: Feasible with minor adjustments
   - 81-100%: Immediately implementable

**Evaluation Framework 2: Eight Human Expert Evaluation Criteria (1-5 Likert Scale)**

1. **Scientific Rigor**: 1=Multiple errors; 5=Methodologically sound
2. **Innovation Score**: 1=Derivative; 5=Novel approaches
3. **Practical Utility**: 1=Not actionable; 5=Immediately applicable
4. **Code Generation Success**: 1=Non-functional; 5=Production-ready
5. **Hypothesis Quality**: 1=Untestable; 5=Clear, falsifiable
6. **Planning Quality**: 1=Incomplete; 5=Comprehensive workflow
7. **Tool Selection Accuracy**: 1=Inappropriate; 5=Optimal choices
8. **Biological Feasibility**: 1=Impossible; 5=Readily executable

Please evaluate the following experimental plan:
{plan}

**Evaluation Requirements:**
1. Evaluate using BOTH frameworks - provide scores for all Framework 1 dimensions (0-100%) and all Framework 2 dimensions (1-5 scale)
2. Provide specific scores and detailed rationale for each dimension in both frameworks
3. Identify strengths and weaknesses of the plan
4. Offer specific improvement recommendations
5. Assess the innovation and feasibility of the experimental design
6. Analyze expected discovery potential and biological significance
7. For Discovery Rate in Framework 1, calculate the ratio based on identified novel insights, testable hypotheses, and tool recommendations
8. Calculate Likert Average as the mean of all 8 Framework 2 scores
9. Calculate Framework 1 Average as the mean of all 5 Framework 1 percentage scores
10. Provide an overall assessment integrating insights from both evaluation frameworks

**Required Output Format - Final Summary:**

✅ Final Summary

**Framework 1 Results (0-100% scale):**
| Metric | Score |
|--------|-------|
| Hypothesis Quality | [XX]% |
| Planning Quality | [XX]% |
| Tool Selection | [XX]% |
| Discovery Rate | [XX]% |
| Biological Feasibility | [XX]% |
| **Framework 1 Average** | **[XX.X]%** |

**Framework 2 Results (1-5 Likert scale):**
| Metric | Score |
|--------|-------|
| Scientific Rigor | [X.X] |
| Innovation Score | [X.X] |
| Practical Utility | [X.X] |
| Code Generation Success | [X.X] |
| Hypothesis Quality | [X.X] |
| Planning Quality | [X.X] |
| Tool Selection Accuracy | [X.X] |
| Biological Feasibility | [X.X] |
| **Likert Average** | **[X.X] / 5.0** |

**Overall Assessment:**
| Summary Metric | Value |
|----------------|-------|
| Framework 1 Average | [XX.X]% |
| Framework 2 Average | [X.X] / 5.0 |
| Overall Grade | **[Grade] ([Description])** |

**Grading Scale:**
- 90-100%: A+ (Exceptional Research Plan)
- 85-89%: A (Outstanding Research Plan)
- 80-84%: A- (Outstanding Research Plan)
- 75-79%: B+ (Good Research Plan)
- 70-74%: B (Good Research Plan)
- 65-69%: B- (Satisfactory Research Plan)
- 60-64%: C+ (Adequate Research Plan)
- 55-59%: C (Adequate Research Plan)
- 50-54%: C- (Below Average Research Plan)
- Below 50%: D/F (Poor Research Plan)
"""
