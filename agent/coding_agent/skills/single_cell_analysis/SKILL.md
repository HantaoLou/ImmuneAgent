---
name: single_cell_analysis
description: Single-cell RNA-seq analysis covering all immune and non-immune cell types (~50 subtypes). Two-phase workflow: Python preprocessing → R downstream analysis. Handles h5ad, RDS, CSV input with automatic format detection.
---

## Single-Cell Analysis Skill

Full scRNA-seq analysis pipeline covering immune cells (T, B, NK, monocytes, macrophages, DCs, granulocytes, ILCs) and non-immune cells (epithelial, endothelial, fibroblast, pericyte, smooth muscle, erythrocyte).

## Workflow: Two Phases

### Phase 1: Preprocessing (Python/scanpy)
Sequential pipeline — each block feeds the next:

| Block | Script | Input | Output |
|-------|--------|-------|--------|
| Load | 01_load.py | Raw data file (.h5ad/.rds/.csv/.mtx) | loaded_adata.h5ad |
| QC | 02_qc.py | loaded_adata.h5ad | qc_adata.h5ad |
| Normalize | 03_normalize.py | qc_adata.h5ad | normalized_adata.h5ad |
| Dim Reduction | 04_dim_reduction.py | normalized_adata.h5ad | dimred_adata.h5ad |
| Clustering | 05_clustering.py | dimred_adata.h5ad | clustered_adata.h5ad |
| Annotation | 06_celltype_annotation.py | clustered_adata.h5ad | annotated_adata.h5ad |

### Phase 2: Downstream Analysis (R/Seurat)
Independent blocks — run any combination based on the research question:

| Block | Script | Analysis | Key Parameters |
|-------|--------|----------|---------------|
| BCR Analysis | 08_bcr_analysis.py | Isotype, clonotype, V gene usage | species |
| TCR Analysis | 08_tcr_analysis.py | CDR3 length, clonotype, exhaustion | species |
| DEG | 09_deg.R | Differential expression + volcano | group1, group2, logfc_threshold |
| Trajectory | 10_trajectory.R | Monocle3 pseudotime | lineage, root_celltype |
| Dotplot | 11_dotplot.R | Marker gene expression | marker_set (all/tcell/bcell/myeloid/nk/granulocyte/stromal) |
| CellChat | 12_cellchat.R | Cell-cell communication | db_type |
| Isolate | 13_isolate.R | Lineage subsetting | lineage (T_cell/B_cell/Monocyte/NK/DC/...) |
| TCR Binding | 14_tcr_binding.R | Binding prediction visualization | binding_threshold |
| Integration | 15_integration.R | Multi-source data merge | csv/rds fields |
| TCR CellType | 16_tcr_celltype.R | T cell subtype UMAP | annotation_level |
| TCR Trajectory | 17_tcr_trajectory.R | T cell pseudotime | root_celltype |
| TCR Clonotype | 18_tcr_clonotype.R | Clonotype expansion | - |

## When to Use MCP Tools vs Composable Scripts

**Use MCP tools directly** when:
- The task matches a standard analysis the MCP server handles (e.g., "run DEG analysis on this RDS file")
- No adaptation is needed — the data is in expected format

**Use composable scripts** when:
- The input data format needs adaptation (non-standard column names, different species)
- The agent needs to chain multiple analysis steps
- Custom parameters or workflow variations are needed

## Species Detection
- Human: gene names uppercase (IGHV, MT-), use `species: human`, `mito_prefix: MT-`
- Mouse: gene names lowercase (Ighv, mt-), use `species: mouse`, `mito_prefix: mt-`

## Cell Types Supported (~50)
- **T cells** (14): CD8 naive/effector/memory/exhausted/cytotoxic/tissue-resident, CD4 naive/Th1/Th2/Th17/Treg/Tfh, MAIT, γδ
- **B cells** (8): Naive, Memory, Germinal Center, Plasma, Activated, Proliferating, Atypical, Transitional
- **Myeloid** (10): Classical/Intermediate/NC monocytes, M1/M2 macrophages, cDC1/cDC2/pDC/mature DC, general macrophage
- **NK/ILC** (6): NK, CD56bright, CD56dim, ILC1, ILC2, ILC3
- **Granulocytes** (4): Neutrophil, Eosinophil, Basophil, Mast Cell
- **Non-immune** (8+): Epithelial, Endothelial, Lymphatic Endothelial, Fibroblast, Pericyte, Smooth Muscle, Megakaryocyte, Erythrocyte, HSC/Progenitor
