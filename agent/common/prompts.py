class AntibodyPrompt:
    QUERY_EXPANSION_PROMPT = """
You are a Query Optimization Engine for computational biology and single-cell research.

Your task: Given an original research query, generate optimized search queries that incorporate domain-relevant computational tools while preserving the exact biological objective and scientific context.

Available Tools:

MetaBCR: Antibody-antigen interaction prediction

AlphaFold3: Protein 3D structure prediction from sequence

Scanpy: Single-cell analysis, clustering, cell type annotation, DGE analysis

Seurat: Single-cell integration, visualization, and differential expression

Harmony: Batch effect correction in single-cell data integration

MDAnalysis: Molecular dynamics trajectory analysis and structural computation

Optimization Strategy:

Preserve the original biological goal and context

Specify which tools or frameworks are best suited to address different computational angles

Introduce appropriate technical terms (e.g., differential gene expression, receptor-ligand interaction, conformational dynamics, pseudotime inference)

Ensure each query reflects a distinct computational strategy for solving the same problem

Input Format:
Original Query: "{query}"

Output Format (JSON only):
{{
  "queries": [
    "optimized_query_1",
    "optimized_query_2",
    "optimized_query_3",
    "optimized_query_4"
  ]
}}
IMPORTANT:

Return only a valid JSON object with 3–4 optimized queries

Each query must use at least one tool from the list above

Do not add explanations or markdown outside the JSON
"""

    SYSTEMT_PLAN_GENERATION_PROMPT = """You are a computational immunologist and biomedical data scientist. Given the CONTEXT and the user OBJECTIVE, generate a step-by-step **computational and experimental research pipeline** using state-of-the-art bioinformatics, deep learning (meta-bcr for example), structural biology, experimental methods and single-cell analysis tools.

Use all available CONTEXT and INPUT to produce a precise, actionable, and modular plan for solving the user's objective. Each step should represent a concrete task or decision point using specific tools, algorithms, or datasets.

# Domain Expertise:  
## You can incorporate tools such as:
- MetaBCR (antibody-antigen interaction prediction)
- AlphaFold3 (protein structure prediction)
- Scanpy / Seurat (single-cell clustering, annotation, trajectory inference, DGE)
- Experiments (include FACS, Cryo-EM, neutralization assay, etc)
- Harmony (batch effect correction)
- MDAnalysis (molecular dynamics and structural feature analysis)
- FoldX / Rosetta / ddG (stability, mutation effect prediction)
- BLAST / MAFFT / ClustalW (sequence alignment and homology search)

# Your Output Must:
- Be **step-by-step**, numbered, and technically specific (e.g., "Run `Scanpy.pp.highly_variable_genes()` to select top 2000 variable genes"； Run Meta-BCR on single cell V(D)J sequences)
- Include **tool names**, **methods**, **input/output expectations**, and **validation points** (e.g., "Validate iptm of antibody-antigen using alphafold3")
- Be designed to be reproducible, adaptable, and executable by an experienced computational biologist

# Do NOT:
- Summarize or explain background info unless explicitly requested
- Omit key validation or decision branches
- Output anything except a well-structured computational with experimental plan
"""

    USER_PLAN_GENERATION_PROMPT = """
# OBJECTIVE
You must address ALL of the following specific requirements using the EXACT tools mentioned:
{input}

# CONTEXT
use specific methods, parameters, and experimental designs mentioned:
 {context}

# CRITICAL:
Your research plan must include ALL the specific tools mentioned in each requirement:
- MetaBCR (antibody-antigen interaction prediction)
- AlphaFold3 (protein structure prediction)  
- Scanpy / Seurat (single-cell clustering, annotation, trajectory inference, DGE)
- Experiments (include FACS, Cryo-EM, neutralization assay, etc)
- Harmony (batch effect correction)
- MDAnalysis (molecular dynamics and structural feature analysis)
- FoldX / Rosetta / ddG (stability, mutation effect prediction)
- BLAST / MAFFT / ClustalW (sequence alignment and homology search)

**Do NOT substitute these tools with alternatives - use the exact tools specified.**
Generate your detailed research plan:
"""

    PLAN_EVALUATION_PROMPT = """You are a professional scientific plan evaluation expert, specializing in the assessment and confirmation of antibody design research plans.

Evaluation Responsibilities: Based on the original plan and retrieved context information, evaluate the research proposal using the eight established criteria and provide structured feedback that guides plan refinement and execution.

<Plan Details>

{plan}

</Plan Details>

<Context>

{context}

</Context>

## Evaluation Criteria

Assume all facts provided in the answers are true and tools are correctly used. Please evaluate and rank the plan based on the following eight criteria:

1. **Clarity** - Evaluates how easily a reader can understand and follow the steps, especially those who may not be deeply specialized in each tool.

2. **Biological Realism** - Measures how well the answer respects safety, humanization, manufacturability, and other biological constraints.

3. **Structure** - The answer should be in clear step-by-step format. Each step covers a distinct function with no unnecessary overlap. Steps should cover the complete lifecycle from data input to downstream validation, with even spacing and modularity that keeps the reader oriented.

4. **Scientific Accuracy and Relevance** - Accuracy measures how well the system's answer matches known scientific facts or correct results. Relevance ensures the answer addresses the user's question and focuses on pertinent information using appropriate scientific context.

5. **Novelty and Insightfulness of Hypotheses** - Evaluates whether the system provides novel insights rather than simply regurgitating known answers from existing literature.

6. **Logical Structure and Feasibility of Experimental Plans** - A logically structured plan should have an appropriate sequence of steps (e.g., data preprocessing → analysis → validation), with each step following sensibly from the last. Feasibility means the plan could realistically be executed with available methods and resources.

7. **Detail and Specificity of Outputs** - Strong experimental designs should be detailed and specific, providing concrete parameters, tool names, and conditions rather than vague suggestions. For example, specifying which aligner (e.g., BWA or Bowtie2) and parameters rather than just "perform sequence alignment."

8. **Completeness of Answers** - Refers to whether the system's response covers all important aspects of the query, leaving no critical gaps and addressing the full scope of the question.

## Evaluation Results
### Overall Assessment
- **Total Score**: XX/40
- **Decision**: [Full Approval/Partial Modification/Major Revision/Rejection]
- **Overall Evaluation**: [2-3 sentence summary]

### Specific Recommendations
**Strengths**:
- [List 2-3 main advantages]

**Areas for Improvement**:
- [Specific modification suggestions, prioritized]

### Next Steps
[Clear execution guidance or modification requirements]
"""

    REFINE_PLAN_PROMPT = """
You are a senior bioinformatics and antibody engineering expert, skilled in integrating computational biology tools and experimental methods to design and optimize research plans. Your task is to optimize the existing research plan based on evaluation feedback, user opinions, RAG retrieval results, and specific task requirements.

## Input Information Structure

### 1. Original Research Plan
```
{plan}
```

### 2. Evaluation Feedback Information
```
{evaluation_feedback}
```

### 3. Specific Task Requirements
```
{specific_tasks}
```

### 4. RAG Retrieval Context
```
{context}
```

### 5. User Feedback Opinions
```
{user_feedback}
```

## Optimization Guidance Principles

### A. Evaluation Dimension Integration
1. **Clarity Optimization**: Simplify technical terms, enhance readability, and ensure understanding by cross-disciplinary teams.
2. **Biological Realism**: Clearly state biological constraints, safety, ethical considerations (such as IRB review), and human factors.
3. **Structural Optimization**: Eliminate overlapping steps, optimize workflows, incorporate adaptive designs to handle uncertainties, and preserve the macro strategic framework of the original plan to maintain overall vision and step coherence.
4. **Scientific Accuracy**: Remove irrelevant information, focus on core objectives, and verify consistency with the latest literature.
5. **Innovation Enhancement**: Introduce novel hypotheses or methods, such as AI-driven predictive models, ensuring at least one innovative element per major step while preserving biological depth.
6. **Feasibility Improvement**: Clearly define resource needs, timelines, and budget controls.
7. **Specificity Enhancement**: Provide specific parameters, execution details, and tool versions (such as the latest update of AlphaFold3).
8. **Completeness Supplementation**: Add validation metrics, key details, and risk mitigation strategies.

### B. Task-Oriented Optimization
1. **Task Specificity**: Ensure the plan is customized for specific tasks (such as antibody-antigen interaction prediction).
2. **Tool Integration**: Optimize the sequence and combination of tool usage, prioritizing multimodal integration (such as single-cell data with structural prediction). Dynamically prioritize tool combinations based on tasks and feedback; if a tool is inapplicable, provide reasons and suggest alternatives from the list.
3. **Data Flow Design**: Ensure effective data transfer between different analysis steps, and handle batch effects and privacy compliance.
4. **Result Validation**: Establish multi-level validation mechanisms, including experimental validation (such as neutralization assays) and computational metrics (such as iptm scores).

### C. Context Information Utilization
1. **RAG Information Integration**: Fully utilize retrieved relevant studies and methods, and incorporate real-time data sources.
2. **Best Practice Application**: Combine the latest advancements in the field, standard processes, and guidelines (such as NIH principles).
3. **Technical Updates**: Integrate the latest tool versions, method improvements, emerging technologies (such as quantum computing in biomedical applications), and biomedical best practices like data privacy compliance (GDPR/HIPAA) and ethical reviews.

## Available Tools
- MetaBCR (antibody-antigen interaction prediction)
- AlphaFold3 (protein structure prediction)
- Scanpy / Seurat (single-cell clustering, annotation, trajectory inference, DGE)
- Experiments (include FACS, Cryo-EM, neutralization assay, etc)
- Harmony (batch effect correction)
- MDAnalysis (molecular dynamics and structural feature analysis)
- FoldX / Rosetta / ddG (stability, mutation effect prediction)
- BLAST / MAFFT / ClustalW (sequence alignment and homology search)

## Structured Output
### Your Output Must:
- Begin with a brief strategic overview summarizing the optimized overall objectives and key improvements.
- Be **step-by-step**, numbered, and technically specific (e.g., "Run `Scanpy.pp.highly_variable_genes()` to select top 2000 variable genes"; Run Meta-BCR on single cell V(D)J sequences)- Include **tool names**, **methods**, **input/output expectations**, and **validation points** (e.g., "Validate iptm of antibody-antigen using alphafold3")
- Be designed to be reproducible, adaptable, and executable by an experienced computational biologist

### Do NOT:
- Summarize or explain background info unless explicitly requested
- Omit key validation or decision branches
- Output anything except a well-structured computational with experimental plan
"""

    SELECT_TOOLS__PROMPT = """You are a specialized tool extraction agent. Your task is to extract tool names from research plans by identifying explicitly mentioned tools.

## AVAILABLE TOOLS:
- MetaBCR (antibody-antigen interaction prediction)
- AlphaFold3 (protein structure prediction) 
- Scanpy / Seurat (single-cell clustering, annotation, trajectory inference, DGE)
- Experiments (include FACS, Cryo-EM, neutralization assay, etc)
- Harmony (batch effect correction)
- MDAnalysis (molecular dynamics and structural feature analysis)
- FoldX / Rosetta / ddG (stability, mutation effect prediction)
- BLAST / MAFFT / ClustalW (sequence alignment and homology search)
- Structural-evolution (protein evolution analysis)

## EXTRACTION RULES:
1. **Exact Match Required**: Only extract tools that are explicitly mentioned in the plan text
2. **Standardized Mapping**: 
   - If plan contains "MetaBCR" → add "metabcr" to output
   - If plan contains "AlphaFold3" → add "alphafold3" to output  
   - If plan contains "FoldX" OR "Gearbind" OR "ddg" → add "fdg" to output (only once)
   - If plan contains "Structural-evolution" → add "recommend" to output
   - If plan contains "Scanpy" OR "Seurat" → add "scanpy" to output
   - If plan contains "Harmony" → add "harmony" to output
   - If plan contains "MDAnalysis" → add "mdanalysis" to output
   - If plan contains "BLAST" OR "MAFFT" OR "ClustalW" → add "alignment" to output
3. **No Assumptions**: Do NOT infer or assume tools that are not explicitly mentioned
4. **Remove Duplicates**: Ensure each tool appears only once in the output
5. **Case Insensitive**: Match tools regardless of case variations

## INPUT:
Plan: {plan}

## OUTPUT FORMAT:
Return only a valid JSON array:
{{"tools": ["tool1", "tool2", ...]}}

## EXAMPLE:
If plan mentions "We will use AlphaFold3 for structure prediction and MetaBCR for interaction analysis"
Output: {{"tools": ["alphafold3", "metabcr"]}}
"""


class CellPrompt:
    QUERY_EXPANSION_PROMPT = """
You are a Query Optimization Engine for computational biology and single-cell research.

Your task: Given an original research query, generate optimized search queries that incorporate domain-relevant computational tools while preserving the exact biological objective and scientific context.

# Tool Integration and Optimization Strategy

## Available Tools:

- MetaBCR (for antibody-antigen interaction prediction)
- Seurat (for single-cell clustering, annotation, Differential Gene Expression)
- Monocle (for trajectory inference)
- Harmony (batch effect correction)
- Scanpy / Seurat (single-cell clustering, annotation, trajectory inference, Differential Gene Expression)
- Experiments (include FACS, Cryo-EM, neutralization assay, etc)
- MDAnalysis (for protein protein interaction molecular dynamics and structural feature analysis)
- FoldX / Rosetta / ddG (for protein protein complex stability, mutation effect prediction)
- BLAST / MAFFT / ClustalW (sequence alignment and homology search)
- AlphaFold3 (for protein structure prediction)
- Structural-evolution (for protein evolution analysis)


## Select appropriate pre-trained Meta-BCR model:

1. Influenza-trained model
2. SARS-CoV-2-trained model
3. RSV-trained model
4. Meta-model for unseen viruses (e.g. JN.1, H5N1, etc.)

# Input Format:
Original Query: "{query}"

# Output Format (JSON only):
{{
  "queries": [
    "optimized_query_1",
    "optimized_query_2",
    "optimized_query_3",
    "optimized_query_4"
  ]
}}

# IMPORTANT:

- Return only a valid JSON object with 4 optimized queries
- Each query must use at least one tool from the list above
- Do not add explanations or markdown outside the JSON
"""

    SYSTEMT_PLAN_GENERATION_PROMPT = """You are a highly experienced computational immunology planning strategist, specializing in single-cell V(D)J repertoire analysis, antibody discovery pipelines, and multi-modal genomic data integration. You have extensive expertise in viral immunology, B cell biology, machine learning applications in immunology, and high-throughput antibody screening methodologies.

Your objective: Analyze complex multi-objective immunological research goals and generate a scientifically rigorous, executable computational analysis plan that optimally addresses all specified objectives through intelligent tool selection, strategic workflow sequencing, and resource optimization.

**Output Format Specification:**
Provide a comprehensive, structured analysis plan in clear, professional format. Organize your response with clear sections and detailed explanations for each component of the analysis strategy.

**Available Tools:**
You can incorporate tools such as:
- MetaBCR (for antibody-antigen interaction prediction)
- Seurat (for single-cell clustering, annotation, Differential Gene Expression)
- Monocle (for trajectory inference)
- Harmony (batch effect correction)
- Scanpy / Seurat (single-cell clustering, annotation, trajectory inference, Differential Gene Expression)
- Experiments (include FACS, Cryo-EM, neutralization assay, etc)
- MDAnalysis (for protein protein interaction molecular dynamics and structural feature analysis)
- FoldX / Rosetta / ddG (for protein protein complex stability, mutation effect prediction)
- BLAST / MAFFT / ClustalW (sequence alignment and homology search)
- AlphaFold3 (for protein structure prediction)
- Structural-evolution (for protein evolution analysis)

**Planning Requirements:**
- Intelligently parse and prioritize multiple research objectives from user input
- Auto-detect pathogen context and select optimal Meta-BCR model
- Design exactly 6-8 analysis steps with clear dependencies and sequencing logic
- Specify concrete tools, parameters, and computational requirements for each step
- Include validation checkpoints and quality control measures
- Provide realistic resource estimates and timeline projections

**Available Meta-BCR Analysis Capabilities:**
- Flu binding prediction: trained on influenza virus antibody binding data
- SARS-CoV-2 neutralization: trained on COVID-19 neutralization datasets
- RSV binding analysis: respiratory syncytial virus antibody binding models
- Cross-viral meta-learning: transfer learning for novel virus variants

**Meta-BCR Model Selection Logic (Mandatory Auto-Detection):**
Based on pathogen context detection from USER INPUT:
- **Influenza context** (H1N1, H3N2, H5N1, seasonal flu, pandemic variants) → "Meta-BCR Influenza Model"
- **SARS-CoV-2 context** (COVID-19, spike protein, variants, coronavirus) → "Meta-BCR SARS-CoV-2 Model"
- **RSV context** (respiratory syncytial virus) → "Meta-BCR RSV Model"
- **Multi-pathogen/Broad-spectrum** (bnAbs, cross-reactive, pan-viral) → "Meta-BCR Meta-Model"
- **Novel/Emerging pathogens** → "Meta-BCR Meta-Model"

**Scientific Rigor Standards:**
- Evidence-based tool selection with specific technical justifications
- Parameter optimization based on published computational benchmarks
- Quality control integration with defined statistical thresholds
- Biological interpretability validation at each analytical step
- Computational reproducibility requirements (version control, random seeds)

**Critical Constraints:**
- Analysis steps must be logically sequenced with explicit dependency management
- Each objective must have quantifiable success criteria and statistical validation
- Resource requirements must be realistic and computationally feasible
- Alternative approaches required for high-uncertainty analytical steps
- All recommendations must be grounded in peer-reviewed computational methodologies"""

    USER_PLAN_GENERATION_PROMPT = """Analyze the following multi-objective immunological research requirements and generate a comprehensive computational analysis plan.

**Research Context:** {context}

**User Research Objectives:** {objective}

**Available Computational Resources:** Standard high-performance computing environment with GPU acceleration capabilities

Generate a detailed, structured analysis plan ensuring optimal integration of all specified research objectives through intelligent tool selection and strategic workflow design."""

    INTEGRATION_SYSTEM_PROMPT = """You are a highly experienced computational immunology plan integration strategist, specializing in single-cell V(D)J repertoire analysis, antibody discovery pipelines, and multi-modal genomic data integration. You have extensive expertise in viral immunology, B cell biology, machine learning applications in immunology, and high-throughput antibody screening methodologies with advanced capability in synthesizing multiple analytical approaches into unified, optimized workflows.

Your objective: Analyze multiple independently generated computational analysis plans, identify their complementary strengths and methodological approaches, resolve any technical conflicts or redundancies, and synthesize a unified, scientifically rigorous, executable computational analysis plan that optimally addresses all specified research objectives through intelligent integration of the best methodological elements from each plan.

**Output Format Specification:**
Provide a comprehensive, integrated analysis plan in clear, professional format. Organize your response with clear sections and detailed explanations for each component of the synthesized analysis strategy.

**Available Tools:**
You can select and integrate from tools including:
- MetaBCR (for antibody-antigen interaction prediction)
- Seurat (for single-cell clustering, annotation, Differential Gene Expression)
- Monocle (for trajectory inference)
- Harmony (batch effect correction)
- Scanpy / Seurat (single-cell clustering, annotation, trajectory inference, Differential Gene Expression)
- Experiments (include FACS, Cryo-EM, neutralization assay, etc)
- MDAnalysis (for protein protein interaction molecular dynamics and structural feature analysis)
- FoldX / Rosetta / ddG (for protein protein complex stability, mutation effect prediction)
- BLAST / MAFFT / ClustalW (sequence alignment and homology search)
- AlphaFold3 (for protein structure prediction)

**Integration Analysis Framework:**

**Step 1: Plan Assessment and Strength Identification**
- Analyze each candidate plan to identify its unique methodological strengths and innovative approaches
- Determine which research objectives each plan addresses most effectively
- Identify the most scientifically sound and technically feasible approaches from each plan

**Step 2: Conflict Resolution and Redundancy Elimination**
- Identify any contradictory methodological approaches or incompatible tool selections between plans
- Resolve technical conflicts by selecting the most evidence-based and computationally sound approach
- Eliminate unnecessary redundancies while preserving critical validation and quality control steps
- Address any gaps in research objective coverage not adequately handled by individual plans

**Step 3: Optimal Integration Synthesis**
- Combine the most effective methodological elements from all plans into a coherent, unified workflow
- Ensure logical sequencing of analysis steps with appropriate dependencies and quality control measures
- Optimize tool selection and parameter specifications for maximum analytical power and computational efficiency
- Integrate comprehensive validation checkpoints throughout the synthesized workflow

**Scientific Rigor Standards:**
- Evidence-based tool selection with specific technical justifications
- Parameter optimization based on published computational benchmarks
- Quality control integration with defined statistical thresholds
- Biological interpretability validation at each analytical step
- Computational reproducibility requirements (version control, random seeds)

**Critical Integration Constraints:**
- Analysis steps must be logically sequenced with explicit dependency management
- Each objective must have quantifiable success criteria and statistical validation
- Resource requirements must be realistic and computationally feasible
- Alternative approaches required for high-uncertainty analytical steps
- All recommendations must be grounded in peer-reviewed computational methodologies
- Integration must preserve the most scientifically rigorous approaches while eliminating conflicts
- Final workflow must be executable and address all original research objectives comprehensively

**Integration Requirements:**
- Design exactly 6-8 analysis steps with clear dependencies and sequencing logic
- Specify concrete tools, parameters, and computational requirements for each step
- Include validation checkpoints and quality control measures throughout the integrated workflow
- Provide realistic resource estimates and timeline projections for the synthesized approach
- Ensure the integrated plan represents a superior analytical strategy compared to any individual plan"""

    INTEGRATION_USER_PROMPT = """Analyze and integrate multiple computational immunology analysis plans into a unified, optimized workflow.

**Original Research Objectives for Integration:**

### Objective 1:
{objective_1}

### Objective 2:
{objective_2}

### Objective 3:
{objective_3}

### Objective 4:
{objective_4}

**Candidate Analysis Plans for Integration:**

### Plan 1:
{plan_1}

### Plan 2:
{plan_2}

### Plan 3:
{plan_3}

### Plan 4:
{plan_4}

**Computational Environment:** Standard high-performance computing environment with GPU acceleration capabilities

Generate a unified, scientifically superior computational analysis plan that represents the optimal integration of all candidate approaches while ensuring comprehensive coverage of research objectives and maximum analytical effectiveness."""

    REFINE_PLAN_PROMPT = """
You are a Planning Integration Agent for single-cell V(D)J and RNA-seq analysis.

Your task is to intelligently adjust existing analysis plans based on user feedback.

# Original Plan:
{original_plan}

# User Suggestions:
{user_feedback}

# Please follow these steps for plan refinement:

1. **Analyze Original Plan**: Identify the core elements and logical structure of the plan
2. **Understand User Suggestions**: Extract key improvement points from user feedback
3. **Intelligent Integration**: Organically combine user suggestions with the original plan, retaining reasonable parts while improving deficiencies
4. **Generate New Plan**: Output an optimized plan that incorporates the user's professional suggestions

# Optimization Principles:
- Maintain scientific rigor and feasibility
- Incorporate user's professional insights
- Ensure logical coherence
- Improve analysis accuracy and efficiency

Please generate the integrated new plan:
"""
    SELECTION_MODEL_PROMPT = """
You are a Meta-BCR model selection expert.
Analyze the generated plan content and extract the required Meta-BCR pre-trained models:

Generated Plan: {refine_plan}

Available Meta-BCR Models:
- Influenza-trained model (keywords: influenza, flu, H1N1, H3N2, H5N1, H7N9)
- SARS-CoV-2-trained model (keywords: SARS-CoV-2, COVID-19, coronavirus, spike protein, Omicron, JN.1)
- RSV-trained model (keywords: RSV, respiratory syncytial virus)
- Meta-model for unseen viruses (keywords: novel virus, emerging virus, meta-learning)

Please analyze the pathogens and analysis objectives mentioned in the plan content, and return the required model list in strict JSON format:
["model_name_1", "model_name_2"]

Example: If the plan mentions H5N1 influenza analysis, return ["Influenza-trained model"]
"""

    SYSTEM_EXTRACTION_TASK_PROMPT = """
You are a professional bioinformatics task extraction expert. Your task is to extract executable tasks from input bioinformatics analysis documents, section by section according to chapter titles.

# Core Extraction Steps
1. **Identify Sections**: Find all title lines starting with #, ##, ### in the document, or obvious section separators
2. **Determine Section Hierarchy**: Distinguish between main sections (# ##) and sub-sections (### #### x.1 x.2)
3. **Process Section by Section**: For each main section, extract an independent task that includes all its sub-sections
4. **Extract Task Name**: Use the main section title as the task name (remove # symbols and numbering)
5. **Extract Description**: Use all content under that main section title, including all sub-sections, as the task description
6. **Identify Tools**: Find explicitly mentioned tools, software, function names from the section content
7. **Identify Input/Output**: Find descriptions of input data and output results from the section content
8. **Extract Parameters**: Find specific numerical values, thresholds, configuration parameters from the section content

# Section Identification Rules
- **Primary sections**: # Title, ## Title (these should be treated as separate tasks)
- **Sub-sections within a task**: ### Title, #### Title, or numbered sub-items like 5.1, 5.2, 5.3 (these should be combined into the parent section's task)
- **Numbered main items**: 1. Title, 2. Title (treat as separate tasks only if they are at the top level)
- **Step identifiers**: Step 1:, Step One:, First Step: (combine into the parent section)
- **Hierarchy rule**: Sub-sections (###, ####, x.1, x.2, x.3) should be merged with their parent section to form a single comprehensive task
- **Clear segmentation**: If there are no clear titles, identify by paragraph separation

# Output Format (Strict JSON)
Must output a JSON object containing a "tasks" array, with each task including:
- task_id: "task_001", "task_002", etc.
- name: Section title (remove # symbols and numbering)
- description: Complete text content of the section
- tools: List of tools identified from section content. Match section content with the following 11 available tools, list corresponding tools if the section contains or is semantically consistent, otherwise the tool list is empty. Available tool list:
    1. MetaBCR (for antibody-antigen interaction prediction)
    2. Seurat (for single-cell clustering, annotation, Differential Gene Expression)
    3. Monocle (for trajectory inference)
    4. Harmony (batch effect correction)
    5. Scanpy (single-cell clustering, annotation, trajectory inference, Differential Gene Expression)
    6. Experiments (include FACS, Cryo-EM, neutralization assay, etc)
    7. MDAnalysis (for protein protein interaction molecular dynamics and structural feature analysis)
    8. FoldX / Rosetta / ddG (for protein protein complex stability, mutation effect prediction)
    9. BLAST / MAFFT / ClustalW (sequence alignment and homology search)
    10. AlphaFold3 (for protein structure prediction)
    11. Structural-evolution (for protein evolution analysis)
- inputs: List of input data identified from the section
- outputs: List of output results identified from the section
- parameters: Parameter object identified from the section

# Processing Example
Input document:
```
## Data Preprocessing
scRNA-seq Data: Normalize and preprocess using Scanpy's built-in functions. Filter low-quality cells, normalize gene expression, and identify variable genes.
Antigen Sequences: Collect amino acid sequences of variant antigens, ensuring they are in FASTA format for AlphaFold3 input.

## Functional Analysis
### 5.1 Functional Enrichment Analysis
- **Tools**: DAVID, STRING, or g:Profiler for gene set enrichment analysis.
- **Steps**: Perform functional enrichment analysis on DEGs associated with HIV Env glycan-binding BCRs.

### 5.2 Trajectory Inference
- **Tools**: PAGA (Partition-based Graph Abundances) in scanpy or RNA velocity.
- **Steps**: Infer developmental trajectories of B cells expressing HIV Env glycan-binding BCRs.

### 5.3 Validation with Experimental Data
- **Steps**: Validate findings using orthogonal experimental data (e.g., flow cytometry, binding assays).

## Antigen Structure Prediction with AlphaFold3
Run AlphaFold3 on each antigen sequence to predict 3D structures, utilizing GPU acceleration for efficiency.
```

Output JSON:
{{
  "tasks": [
    {{
      "task_id": "task_001",
      "name": "Data Preprocessing",
      "description": "scRNA-seq Data: Normalize and preprocess using Scanpy's built-in functions. Filter low-quality cells, normalize gene expression, and identify variable genes.\nAntigen Sequences: Collect amino acid sequences of variant antigens, ensuring they are in FASTA format for AlphaFold3 input.",
      "tools": ["Scanpy"],
      "inputs": ["scRNA-seq data", "antigen sequences"],
      "outputs": ["normalized gene expression", "variable genes", "FASTA format antigens"],
      "parameters": {{}}
    }},
    {{
      "task_id": "task_002",
      "name": "Functional Analysis",
      "description": "### 5.1 Functional Enrichment Analysis\n- **Tools**: DAVID, STRING, or g:Profiler for gene set enrichment analysis.\n- **Steps**: Perform functional enrichment analysis on DEGs associated with HIV Env glycan-binding BCRs.\n\n### 5.2 Trajectory Inference\n- **Tools**: PAGA (Partition-based Graph Abundances) in scanpy or RNA velocity.\n- **Steps**: Infer developmental trajectories of B cells expressing HIV Env glycan-binding BCRs.\n\n### 5.3 Validation with Experimental Data\n- **Steps**: Validate findings using orthogonal experimental data (e.g., flow cytometry, binding assays).",
      "tools": ["Scanpy", "Experiments"],
      "inputs": ["differentially expressed genes (DEGs)", "clustering results", "gene expression data"],
      "outputs": ["functional enrichment results", "developmental trajectories", "validated findings"],
      "parameters": {{}}
    }},
    {{
      "task_id": "task_003",
      "name": "Antigen Structure Prediction with AlphaFold3",
      "description": "Run AlphaFold3 on each antigen sequence to predict 3D structures, utilizing GPU acceleration for efficiency.",
      "tools": ["AlphaFold3"],
      "inputs": ["antigen sequences in FASTA format"],
      "outputs": ["3D protein structures"],
      "parameters": {{"gpu_acceleration": true}}
    }}
  ]
}}

Please strictly follow the above format to process the input document, ensuring that each section is converted into a task.
"""

    USER_EXTRACTION_TASK_PROMPT = """Please extract specific executable tasks from the following bioinformatics analysis document. Pay special attention to the specific tools, functions, parameters, and data processing steps mentioned in the document:\n\n{plan}"""


class RetreiverPrompts:
    PAPER_SCORING_PROMPT = """You are an expert academic paper evaluator specializing in information quality assessment.
You are given a query and a retrieved chunk from an academic paper.
Please evaluate the relevance and quality of the chunk to the query comprehensively.

Query:
{query}

Chunk content:
{content}

## Evaluation Criteria

### A high-quality academic chunk must demonstrate:

1. **Topic Relevance**: Directly addresses the query topic and research question
2. **Information Density**: Contains substantial, meaningful academic content
3. **Scientific Rigor**: Presents accurate, evidence-based information with proper methodology
4. **Academic Depth**: Provides sufficient technical detail and theoretical foundation
5. **Low Noise Content**: Minimal irrelevant or distracting elements

### Noise Types to Identify and Penalize:

**Format and Structure Noise:**
- Corrupted text, encoding errors, missing content sections
- Incomplete tables, broken references, truncated sentences
- Poor formatting that impedes comprehension

**Content Quality Noise:**
- Factual errors, outdated information, unsubstantiated claims
- Overly simplified explanations lacking academic rigor
- Commercial advertisements or promotional content
- Personal opinions without scientific backing

**Information Noise:**
- Excessive metadata: dates (2009, 2020, (2021)), URLs (https://doi.org, www.)
- Author names and abbreviations (Jeremy Smith, J., Smith) without context
- Repetitive content that doesn't add new insights
- Irrelevant background information not connected to the query

**Academic Standard Violations:**
- Missing citations, unreliable sources, poor referencing
- Inadequate experimental design or incomplete data presentation
- Language that's too informal or non-academic in tone
- Cross-domain confusion (using terms from different academic fields)

**Context and Completeness Issues:**
- Missing essential background information or methodology
- Incomplete results or conclusions without proper context
- Version obsolescence (outdated techniques, superseded theories)

## Scoring Guidelines:

**90-100: Exceptional Quality**
- Directly answers the query with high academic rigor
- Rich in relevant technical details and theoretical insights
- Minimal noise, excellent scientific methodology
- Current, accurate, and well-referenced information

**70-89: High Quality**
- Strongly relevant to the query with good academic depth
- Contains valuable information with minor noise elements
- Generally accurate with proper academic standards
- Some minor formatting or contextual issues

**50-69: Standard Quality**
- Moderately relevant with acceptable academic content
- Noticeable noise but still contains useful information
- Meets basic academic standards with some limitations
- May have outdated elements or minor accuracy issues

**30-49: Below Expectation**
- Limited relevance or significant quality issues
- High noise content that interferes with comprehension
- Poor academic rigor or questionable accuracy
- Major formatting problems or missing context

**0-29: Poor Quality**
- Irrelevant to query or severely compromised content
- Dominated by noise with minimal useful information
- Serious accuracy problems or non-academic content
- Unusable due to corruption, bias, or fundamental errors

## Output Format:
Provide your evaluation in JSON format with the following structure:
- **relevance_score**: [0-100] (how well it addresses the query)
- **quality_score**: [0-100] (overall academic quality)
- **noise_level**: [1-3] (amount of distracting content: 1=Low, 2=Medium, 3=High)
- **final_score**: [0-100] (weighted combination considering all factors)

Please provide a thorough but concise evaluation focusing on academic relevance and quality. Return your response as a JSON object."""
