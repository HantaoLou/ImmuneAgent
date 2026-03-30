## Common Gotchas

## UMAP Missing
Data may not have pre-computed UMAP coordinates. All R modules auto-compute UMAP when missing, but Python scripts (04_dim_reduction.py) must be run first for scanpy workflows.

## RDS Version Mismatch
Seurat v4 vs v5 objects have different internal structures. If reading fails with "Error in validObject", the user needs to run `UpdateSeuratObject()` first.

## Species Prefix
Mitochondrial gene prefix differs: `MT-` (human) vs `mt-` (mouse). Wrong prefix → QC filters out all/no cells. Check gene names in the data to detect species.

## Column Name Aliases
Labs use different column names for the same concept:
- Isotype: isotype, c_call, c_gene, constant_region, ig_class
- Clone ID: clone_id, clonotype_id, clonotype, clone
- Cell type: cell_type, celltype, cell_label, annotation, seurat_clusters
- Sample: sample, sample_id, donor, patient, subject

## Large Datasets (>50k cells)
Consider subsampling or lower clustering resolution. Trajectory analysis (monocle3) is particularly memory-intensive. Set cluster_resolution < 0.5 for large datasets.

## Missing R Packages
The R modules require: Seurat, monocle3, ggplot2, cowplot, Nebulosa, dplyr, ggrepel, ggrastr, CellChat. If a package is missing, the R script will fail with a clear error message.

## TCR Column Detection
TCR columns are auto-detected by the R modules. Supported formats:
- Individual: TRA_v_gene, TRB_v_gene, CDR3a, CDR3b
- Combined: CTgene, CTaa, CTstrict, Frequency
- If no TCR columns are found, TCR-specific analyses are skipped gracefully.

## Binding Prediction Columns
Binding columns are detected by prefix: bind_predict.*, output.*, bind_output.*
If no binding columns exist, binding visualization is skipped.
