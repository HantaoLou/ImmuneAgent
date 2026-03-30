---
name: cell-trajectory-analysis
description: Perform single-cell RNA-seq trajectory inference using Monocle3. Synchronizes Seurat embeddings, identifies differentiation root nodes, calculates pseudotime, and generates an analysis report.
allowed-tools: Bash, Read
---

# Cell Trajectory Analysis Skill (Monocle3)

A professional bioinformatics tool for single-cell RNA-seq trajectory inference analysis. This skill processes Seurat objects, builds trajectory graphs, calculates pseudotime, and generates comprehensive analysis reports.

## Critical constraint

**Execute the main Python script exactly once** — regardless of success or failure, run the analysis once and return the outcome.

## Prerequisites

Before running this skill, ensure the following R packages are installed:
- Seurat
- monocle3
- ggplot2
- dplyr
- magrittr
- patchwork
- optparse

Run the installation script if needed:

```bash
Rscript install_packages.R
```

## Parameters

Required parameters:
- `rds_path` (string): Path to the input Seurat (.rds) file
- `target_group` (string): Specific group name within group_col to analyze (e.g., 'EP', 'Control', 'Tumor')
- `root_type` (string): Cell type to define as the trajectory starting point (e.g., 'Basal', 'Stem')

Optional parameters:
- `group_col` (string): Metadata column for grouping (default: 'condition')
- `label_col` (string): Metadata column for cell type annotations (default: 'celltype')
- `output_path` (string): Output directory path (default: './results')

## Execution

### Step 1: Verify input

Ensure the RDS file exists at the specified path and is a valid Seurat object.

### Step 2: Run the trajectory analysis

Execute the main Python script with appropriate parameters:

```bash
python main.py --rds_path "path/to/data.rds" --target_group "EP" --root_type "Basal" --group_col "condition" --label_col "celltype" --output_path "./results"
```

Example:

```bash
python main.py \
  --rds_path "/data/pbmc_data.rds" \
  --target_group "Stimulated" \
  --root_type "Naive" \
  --group_col "orig.ident" \
  --label_col "seurat_clusters" \
  --output_path "./trajectory_results"
```

### Step 3: Return the result

After execution, return the outcome to the user:

**Success**: Display the report path and output artifacts:
- Trajectory Analysis Report (Markdown)
- Pseudotime UMAP plot (PNG and PDF)
- Trajectory result RDS file

**Failure**: Show the error details and suggest troubleshooting steps.

## Output Artifacts

The skill generates the following files in the output directory:
- `trajectory_result.rds`: Monocle3 CDS object with pseudotime
- `pseudotime_umap.pdf`: High-quality publication-ready plot
- `pseudotime_umap.png`: Quick preview image for report
- `Trajectory_Analysis_Report.md`: Comprehensive analysis report

## Analysis Process

The skill performs these steps automatically:

1. **Data Loading**: Reads Seurat object and subsets to target group
2. **Coordinate Synchronization**: Inherits UMAP/t-SNE/PCA coordinates from Seurat
3. **Trajectory Construction**: Builds principal graph using Monocle3
4. **Root Identification**: Locates trajectory root based on specified cell type
5. **Pseudotime Calculation**: Computes pseudotime for all cells
6. **Visualization**: Generates trajectory plots colored by pseudotime
7. **Report Generation**: Creates a detailed Markdown analysis report

## Troubleshooting

- **Missing packages**: Run `Rscript install_packages.R` to install required R packages
- **Low memory**: For large datasets, consider subsetting the data before analysis
- **Root type not found**: Verify the root_type value matches exactly with cell type labels in the Seurat object
- **Coordinate sync failed**: Ensure the Seurat object contains UMAP, t-SNE, or PCA embeddings

## Downstream Analysis

After trajectory analysis, consider:
- Differential expression analysis along pseudotime using `graph_test`
- Branch point analysis for bifurcating trajectories
- Gene expression heatmaps ordered by pseudotime
- Functional enrichment of branch-specific genes
