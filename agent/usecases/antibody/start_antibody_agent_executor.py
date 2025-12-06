# 测试完整流程
import inspect
import os
import sys

# 添加agent目录到Python路径
agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, agent_dir)

from usecases.antibody.graph.execute_graph import run_execute_agent

if __name__ == "__main__":
    query = """
# Research Plan: Integrative Study of Antibody-Antigen Interactions Using Computational and Experimental Approaches

## **Objectives**

1. Predict antibody-antigen interactions using MetaBCR.
2. Predict protein structures using AlphaFold3.
3. Perform single-cell clustering, annotation, trajectory inference, and differential gene expression (DGE) 
   analysis using Scanpy/Seurat.
4. Correct batch effects in single-cell data using Harmony.
5. Analyze molecular dynamics and structural features using MDAnalysis.
6. Predict the stability and mutation effects on protein interactions using FoldX/Rosetta/ddG.
7. Perform sequence alignment and homology search using BLAST, MAFFT, and ClustalW.

---

## **Tools Overview**

- **MetaBCR**: For predicting antibody-antigen interactions.
- **AlphaFold3**: For high-resolution protein structure prediction.
- **Scanpy/Seurat**: For single-cell RNA sequencing (scRNA-seq) data analysis, including clustering, annotation, trajectory inference, and DGE.
- **Harmony**: For batch effect correction in scRNA-seq data.
- **MDAnalysis**: For molecular dynamics simulations and structural feature analysis.
- **FoldX/Rosetta/ddG**: For predicting protein stability and the effects of mutations on binding affinity. 
- **BLAST/MAFFT/ClustalW**: For sequence alignment and homology search.

---

## **Research Plan**

### **Phase 1: Data Collection and Preprocessing**

**Objective:** Collect and preprocess datasets for antibody-antigen interactions, single-cell RNA-seq data, 
and protein sequences.

#### **Tasks:**

1. **Antibody-Antigen Interaction Data:**
   - Gather experimental data on known antibody-antigen interactions.
   - Use BLAST to identify homologous sequences of antibodies and antigens for comparative analysis.        
   - Perform sequence alignment using MAFFT/ClustalW to prepare input for MetaBCR.

2. **Single-Cell RNA-seq Data:**
   - Collect scRNA-seq datasets from experiments involving antibody-producing cells (e.g., B cells, plasma cells).
   - Correct batch effects in the data using Harmony.
   - Normalize and preprocess the data for downstream analysis.

3. **Protein Sequences:**
   - Retrieve protein sequences of antibodies and antigens from public databases (e.g., UniProt, GenBank).  
   - Perform sequence alignment using MAFFT/ClustalW to prepare input for AlphaFold3.

---

### **Phase 2: Computational Analysis**

**Objective:** Predict antibody-antigen interactions, predict protein structures, and analyze molecular dynamics.

#### **Tasks:**

1. **Antibody-Antigen Interaction Prediction:**
   - Use MetaBCR to predict potential interactions between antibodies and antigens based on sequence data.  
   - Validate predictions using known experimental data.

2. **Protein Structure Prediction:**
   - Use AlphaFold3 to predict high-resolution structures of antibodies and antigens.
   - Analyze the predicted structures for key binding motifs and interaction sites.

3. **Molecular Dynamics Analysis:**
   - Use MDAnalysis to simulate molecular dynamics of antibody-antigen complexes.
   - Identify critical residues and structural features involved in binding.

4. **Stability and Mutation Analysis:**
   - Predict the effects of mutations on protein stability and binding affinity using FoldX/Rosetta/ddG.    
   - Prioritize mutations that could enhance or disrupt interactions for experimental validation.

---

### **Phase 3: Single-Cell Analysis**

**Objective:** Analyze single-cell RNA-seq data to understand cellular responses and validate computational 
predictions.

#### **Tasks:**

1. **Clustering and Annotation:**
   - Use Scanpy/Seurat to cluster cells based on gene expression profiles.
   - Annotate clusters using known marker genes for antibody-producing cells.

2. **Trajectory Inference:**
   - Infer developmental trajectories of B cells and plasma cells using Scanpy/Seurat.
   - Identify key genes involved in differentiation and activation.

3. **Differential Gene Expression (DGE):**
   - Perform DGE analysis to identify genes upregulated or downregulated in response to antigen stimulation.   - Validate predicted antibody-antigen interactions by checking the expression of relevant genes.

4. **Batch Effect Correction:**
   - Use Harmony to correct batch effects in integrated datasets.
   - Ensure robust integration of data from different experiments or batches.

---

### **Phase 4: Experimental Validation**

**Objective:** Validate computational predictions using experimental techniques.

#### **Tasks:**

1. **Flow Cytometry (FACS):**
   - Isolate specific cell populations (e.g., B cells, plasma cells) for further analysis.
   - Validate the expression of predicted marker genes and interaction partners.

2. **Cryo-EM and X-ray Crystallography:**
   - Experimentally determine the structures of antibody-antigen complexes to validate AlphaFold3 predictions.
   - Identify critical residues involved in binding at near-atomic resolution.

3. **Neutralization Assays:**
   - Test the functional activity of predicted antibody-antigen interactions using neutralization assays.   
   - Validate the effects of mutations on binding affinity and functionality.

---

## **Timeline**

| **Phase**               | **Duration (Months)** |
| ----------------------- | --------------------- |
| Data Collection         | 2                     |
| Computational Analysis  | 3                     |
| Single-Cell Analysis    | 3                     |
| Experimental Validation | 4                     |

---

## **Expected Outcomes**

1. A comprehensive understanding of antibody-antigen interactions at the molecular and cellular levels.     
2. High-resolution structural models of antibody-antigen complexes validated by experimental techniques.    
3. Identification of key residues and mutations that influence binding affinity and stability.
4. Insights into the cellular responses of antibody-producing cells during antigen stimulation.

---

## **Conclusion**

This research plan integrates computational tools (MetaBCR, AlphaFold3, Scanpy/Seurat, Harmony, MDAnalysis, 
FoldX/Rosetta/ddG) with experimental techniques (FACS, Cryo-EM, neutralization assays) to provide a holistic understanding of antibody-antigen interactions. By combining these approaches, we aim to advance our knowledge of immune responses and inform the design of therapeutic antibodies.
    """
    from usecases._debug import get_debug_runnable_config

    rc = get_debug_runnable_config()
    run_execute_agent(query, rc)
