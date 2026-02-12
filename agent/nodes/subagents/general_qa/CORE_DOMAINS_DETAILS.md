# 核心领域Prompt详细实现

本文档提供Genetics、Immunology、Clinical Medicine、Bioinformatics四个核心领域的完整Prompt实现细节。

## 一、Immunology 领域

### 1.1 N0: Input Preprocessing - Immunology增强

```python
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
```

### 1.2 N1: Question Decomposition - Immunology增强

```python
def get_question_decomposition_prompt(...) -> str:
    """N1 prompt with immunology-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(...)
    
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

4. **Immune Response Questions**:
   - Sub-objective 1: Identify immune response type (innate vs adaptive, humoral vs cellular)
   - Sub-objective 2: Determine key immune mediators (cytokines, antibodies, complement)
   - Sub-objective 3: Analyze immune response regulation and memory formation

**Immunology-Specific Domain Identification:**
- Core domains should include: "T Cell Biology", "B Cell Biology", "Antigen Presentation", "V(D)J Recombination", "Immune Cell Development" as appropriate
- Use precise domain names (e.g., "T Cell Engineering, Allelic Exclusion" not just "Immunology")
"""
    
    return base_prompt + immunology_enhancements
```

### 1.3 N3: Knowledge Retrieval - Immunology增强

```python
def get_knowledge_retrieval_prompt(...) -> str:
    """N3 prompt with immunology-specific knowledge retrieval"""
    base_prompt = get_base_knowledge_retrieval_prompt(...)
    
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
```

### 1.4 N6/N7: Inference - Immunology增强

```python
def get_initial_inference_prompt(...) -> str:
    """N6 prompt with immunology-specific inference"""
    base_prompt = get_base_initial_inference_prompt(...)
    
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
```

### 1.5 N8: Answer Generation - Immunology增强

```python
def get_answer_generation_prompt(...) -> str:
    """N8 prompt with immunology-specific answer generation"""
    base_prompt = get_base_answer_generation_prompt(...)
    
    immunology_enhancements = """

**Immunology-Specific Answer Format:**

1. **Cell Type Answers**: Include cell type and subtype (e.g., "CD4+ T cell", "naive B cell")
2. **Receptor Answers**: Include receptor name and chain (e.g., "TCR alpha chain", "BCR heavy chain")
3. **Mechanism Answers**: Include specific mechanism (e.g., "allelic exclusion", "positive selection")
4. **MHC Answers**: Include MHC class and allele if specified (e.g., "MHC class I, HLA-A*02:01")
"""
    
    return base_prompt + immunology_enhancements
```

## 二、Clinical Medicine 领域

### 2.1 N0: Input Preprocessing - Clinical Medicine增强

```python
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
```

### 2.2 N1: Question Decomposition - Clinical Medicine增强

```python
def get_question_decomposition_prompt(...) -> str:
    """N1 prompt with clinical medicine-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(...)
    
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
```

### 2.3 N3: Knowledge Retrieval - Clinical Medicine增强

```python
def get_knowledge_retrieval_prompt(...) -> str:
    """N3 prompt with clinical medicine-specific knowledge retrieval"""
    base_prompt = get_base_knowledge_retrieval_prompt(...)
    
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
```

### 2.4 N6/N7: Inference - Clinical Medicine增强

```python
def get_complete_inference_prompt(...) -> str:
    """N7 prompt with clinical medicine-specific inference"""
    base_prompt = get_base_complete_inference_prompt(...)
    
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
```

### 2.5 N8: Answer Generation - Clinical Medicine增强

```python
def get_answer_generation_prompt(...) -> str:
    """N8 prompt with clinical medicine-specific answer generation"""
    base_prompt = get_base_answer_generation_prompt(...)
    
    clinical_enhancements = """

**Clinical Medicine-Specific Answer Format:**

1. **Drug Answers**: Include drug name, class, and dosage if specified (e.g., "Lisinopril (ACE inhibitor), 10mg daily")
2. **Treatment Answers**: Include treatment plan with rationale (e.g., "First-line: ACE inhibitor or ARB, per JNC8 guidelines")
3. **Diagnosis Answers**: Include diagnostic criteria met (e.g., "Hypertension: BP ≥140/90, per JNC8 criteria")
4. **Guideline Answers**: Reference specific guidelines (e.g., "Per JNC8 guidelines, recommend...")
"""
    
    return base_prompt + clinical_enhancements
```

## 三、Bioinformatics 领域

### 3.1 N0: Input Preprocessing - Bioinformatics增强

```python
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
```

### 3.2 N1: Question Decomposition - Bioinformatics增强

```python
def get_question_decomposition_prompt(...) -> str:
    """N1 prompt with bioinformatics-specific decomposition"""
    base_prompt = get_base_question_decomposition_prompt(...)
    
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
```

### 3.3 N3: Knowledge Retrieval - Bioinformatics增强

```python
def get_knowledge_retrieval_prompt(...) -> str:
    """N3 prompt with bioinformatics-specific knowledge retrieval"""
    base_prompt = get_base_knowledge_retrieval_prompt(...)
    
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
```

### 3.4 N4: Calculation Decomposition - Bioinformatics增强

```python
def get_calculation_decomposition_prompt(...) -> str:
    """N4 prompt with bioinformatics-specific calculation decomposition"""
    base_prompt = get_base_calculation_decomposition_prompt(...)
    
    # 引入计算类通用模板
    calculation_guide = get_calculation_guide()
    
    bioinformatics_enhancements = f"""

{calculation_guide}

**Bioinformatics-Specific Calculation Rules:**

1. **Population Genetics Calculations**:
   - Theta (Watterson's estimator): θ = S / Σ(1/i) where S is number of segregating sites
   - Pi (nucleotide diversity): π = Σ(2pq) for all sites
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
"""
    
    return base_prompt + bioinformatics_enhancements
```

## 四、领域配置总结

### 4.1 Immunology配置

```python
DOMAIN_CONFIG = {
    "name": "Immunology",
    "priority_tools": [
        "query_tcr_mcpas",
        "query_celltype_marker",
        "query_ppi",
        "query_proteinatlas",
        "query_knowledge_graph"
    ],
    "common_entities": [
        "T cell", "B cell", "NK cell", "macrophage",
        "TCR", "BCR", "MHC", "antigen", "antibody",
        "allelic exclusion", "positive selection", "negative selection"
    ],
    "calculation_focus": [],  # Immunology较少涉及计算
    "validation_criteria": [
        "Must verify against immunological principles",
        "Check V(D)J recombination rules",
        "Verify cell development stages",
        "Check receptor-ligand interactions"
    ]
}
```

### 4.2 Clinical Medicine配置

```python
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
    "common_entities": [
        "hypertension", "diabetes", "medication", "drug",
        "treatment guideline", "contraindication", "drug interaction"
    ],
    "calculation_focus": [],  # Clinical Medicine较少涉及计算
    "validation_criteria": [
        "Must follow clinical guidelines (JNC8, ADA, etc.)",
        "Exclude all contraindications",
        "Verify drug compatibility",
        "Check evidence-based recommendations"
    ]
}
```

### 4.3 Bioinformatics配置

```python
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
    "common_entities": [
        "variant", "SNP", "theta", "pi", "Fst",
        "chi-square", "statistical test", "population genetics"
    ],
    "calculation_focus": [
        "Watterson's estimator (theta)",
        "Nucleotide diversity (pi)",
        "Fst (fixation index)",
        "Chi-square test",
        "Hardy-Weinberg equilibrium"
    ],
    "validation_criteria": [
        "Must apply formulas correctly",
        "Verify computational assumptions",
        "Check data quality requirements",
        "Verify statistical test results"
    ]
}
```

