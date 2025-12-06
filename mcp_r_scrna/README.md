# mcp_r_scrna: Single-Cell RNA-seq Analysis MCP Server

Comprehensive single-cell RNA-seq preprocessing and analysis tools using Seurat/R via MCP (Model Context Protocol).

## Overview

This MCP server provides 10 specialized tools for end-to-end scRNA-seq analysis, following the external R script execution pattern for robust, production-ready workflows.

## Features

- **Quality Control**: Cell filtering based on gene counts, UMI counts, and mitochondrial percentage
- **Normalization**: SCTransform variance stabilization with optional covariate regression
- **Integration**: Harmony batch correction for multi-sample datasets
- **Clustering**: Graph-based clustering (Leiden/Louvain) with UMAP visualization
- **Doublet Detection**: DoubletFinder-based doublet identification and removal
- **Differential Expression**: Statistical testing (Wilcoxon, MAST, DESeq2) with volcano plots
- **Marker Detection**: Automated marker gene discovery for all clusters
- **Pathway Enrichment**: GO and KEGG enrichment analysis using clusterProfiler
- **Dimensionality Reduction**: PCA, UMAP, tSNE embeddings
- **Cell Subsetting**: Flexible cell filtering based on metadata

## Architecture

```
mcp_r_scrna/
├── scrna_mcp_server.py    # Python MCP server (stdio transport)
├── config.json            # Configuration
├── scripts/               # External R scripts (10 tools)
│   ├── qc_filtering.R
│   ├── normalization_sct.R
│   ├── integration_harmony.R
│   ├── clustering_analysis.R
│   ├── doublet_detection.R
│   ├── deg_analysis.R
│   ├── marker_detection.R
│   ├── pathway_enrichment.R
│   ├── dim_reduction.R
│   └── subset_cells.R
├── tests/                 # Test suite
├── config/                # Additional configs
└── output/                # Analysis outputs (organized by tool)
```

## Installation

### Prerequisites

- Python 3.12+
- R 4.3+
- UV package manager

### Python Dependencies

```bash
pip install fastmcp>=2.12.4 mcp
```

### R Dependencies

Install required R packages:

```r
# Core packages
install.packages(c("Seurat", "ggplot2", "dplyr", "patchwork", "viridis", "jsonlite"))

# Integration
install.packages("devtools")
devtools::install_github("immunogenomics/harmony")

# Doublet detection
remotes::install_github('chris-mcginnis-ucsf/DoubletFinder')

# Pathway enrichment
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "org.Mm.eg.db", "enrichplot"))
```

## Usage

### Starting the Server

```bash
cd /Users/ahleyliu/LocalDoc/ImmuneAgent/mcp_r_scrna
python scrna_mcp_server.py
```

Server runs on **stdio transport** (NOT SSE) on port 8090.

### Tool Workflows

#### Complete scRNA-seq Pipeline

```python
from mcp.client import Client

# 1. Quality Control
result = client.call_tool("run_qc_filtering", {
    "input_rds": "/path/to/raw_seurat.rds",
    "min_genes": 200,
    "max_genes": 6000,
    "min_counts": 1000,
    "mt_percent": 20.0
})
filtered_rds = result["generated_files"][0]  # seurat_filtered.rds

# 2. Normalization (SCTransform)
result = client.call_tool("run_normalization_sct", {
    "input_rds": filtered_rds,
    "vars_to_regress": ["percent.mt"],
    "n_variable_features": 3000
})
normalized_rds = result["generated_files"][0]

# 3. Integration (if multi-sample)
result = client.call_tool("run_integration_harmony", {
    "input_rds": normalized_rds,
    "batch_variable": "orig.ident",
    "dims": 30
})
integrated_rds = result["generated_files"][0]

# 4. Clustering
result = client.call_tool("run_clustering_analysis", {
    "input_rds": integrated_rds,
    "resolution": 0.8,
    "dims": 30,
    "algorithm": "leiden"
})
clustered_rds = result["generated_files"][0]

# 5. Doublet Detection (optional)
result = client.call_tool("run_doublet_detection", {
    "input_rds": clustered_rds,
    "expected_doublet_rate": 0.08
})
singlets_rds = result["generated_files"][0]

# 6. Marker Detection
result = client.call_tool("run_marker_detection", {
    "input_rds": singlets_rds,
    "group_by": "seurat_clusters",
    "only_pos": True,
    "top_n": 10
})
# Output: all_markers.csv, top10_markers.csv, dotplot, heatmap

# 7. Differential Expression
result = client.call_tool("run_deg_analysis", {
    "input_rds": singlets_rds,
    "group_by": "seurat_clusters",
    "ident_1": "0",
    "ident_2": "1",
    "test_use": "wilcox"
})
deg_csv = result["generated_files"][0]  # deg_0_vs_1.csv

# 8. Pathway Enrichment
result = client.call_tool("run_pathway_enrichment", {
    "input_rds": singlets_rds,
    "deg_csv": deg_csv,
    "organism": "human",
    "ontology": "BP"
})
# Output: go_enrichment_all.csv, kegg_enrichment.csv, plots
```

#### Quick Visualization

```python
# Generate embeddings
result = client.call_tool("run_dim_reduction", {
    "input_rds": "/path/to/seurat.rds",
    "methods": ["PCA", "UMAP", "tSNE"],
    "dims": 30
})
# Output: PCA/UMAP/tSNE embeddings, plots
```

#### Cell Subsetting

```python
# Extract B cells only
result = client.call_tool("run_subset_cells", {
    "input_rds": "/path/to/seurat.rds",
    "subset_column": "celltype",
    "subset_values": ["B cell", "Plasma cell"],
    "invert": False
})
bcell_rds = result["generated_files"][0]  # seurat_subset.rds
```

## Tool Reference

### 1. run_qc_filtering

Filters cells based on QC metrics.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `min_genes` (int, default=200): Minimum genes per cell
- `max_genes` (int, default=6000): Maximum genes per cell
- `min_counts` (int, default=1000): Minimum UMI counts
- `mt_percent` (float, default=20.0): Maximum mitochondrial %

**Outputs**:
- `seurat_filtered.rds`: Filtered Seurat object
- `qc_statistics.csv`: QC metrics
- QC plots (violin, scatter)

### 2. run_normalization_sct

SCTransform normalization with variance stabilization.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `vars_to_regress` (list, optional): Variables to regress out (e.g., ["percent.mt"])
- `n_variable_features` (int, default=3000): Number of variable features

**Outputs**:
- `seurat_normalized.rds`: Normalized Seurat object
- `variable_features.csv`: Top variable genes
- `pca_embeddings.csv`: PCA coordinates
- Diagnostic plots (elbow, variable features, PCA loadings)

### 3. run_integration_harmony

Harmony batch correction for multi-sample integration.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `batch_variable` (str, default="orig.ident"): Metadata column for batches
- `dims` (int, default=30): Number of PCA dimensions
- `theta` (list, optional): Diversity clustering penalty

**Outputs**:
- `seurat_integrated.rds`: Integrated Seurat object
- `harmony_embeddings.csv`: Harmony coordinates
- Before/after UMAP comparisons

### 4. run_clustering_analysis

Graph-based clustering with UMAP visualization.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `resolution` (float, default=0.8): Clustering resolution
- `dims` (int, default=30): Number of dimensions
- `algorithm` (str, default="leiden"): "leiden" or "louvain"

**Outputs**:
- `seurat_clustered.rds`: Clustered Seurat object
- `cluster_assignments.csv`: Cell-cluster mapping
- `cluster_statistics.csv`: Cluster sizes
- UMAP plots

### 5. run_doublet_detection

DoubletFinder-based doublet detection.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `expected_doublet_rate` (float, default=0.08): Expected doublet rate (8%)
- `pN` (float, default=0.25): Proportion of artificial doublets
- `pK` (float, default=0.09): PC neighborhood size
- `dims` (int, default=20): Number of PCs

**Outputs**:
- `seurat_singlets.rds`: Singlet-only Seurat object
- `doublet_assignments.csv`: Doublet classifications
- `doublet_statistics.csv`: Detection metrics
- UMAP plots (before/after doublet removal)

### 6. run_deg_analysis

Differential expression analysis between groups.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `group_by` (str, default="seurat_clusters"): Grouping column
- `ident_1` (str): First identity (required)
- `ident_2` (str, optional): Second identity (None = all others)
- `test_use` (str, default="wilcox"): Statistical test
- `logfc_threshold` (float, default=0.25): Min log fold-change
- `min_pct` (float, default=0.1): Min expression percentage

**Outputs**:
- `deg_<comparison>.csv`: DEG table
- `volcano_<comparison>.pdf`: Volcano plot
- `heatmap_<comparison>.pdf`: Top DEGs heatmap
- `summary_<comparison>.csv`: Statistics

### 7. run_marker_detection

Marker gene detection for all clusters.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `group_by` (str, default="seurat_clusters"): Grouping column
- `only_pos` (bool, default=True): Only positive markers
- `min_pct` (float, default=0.25): Min expression percentage
- `logfc_threshold` (float, default=0.5): Min log fold-change
- `top_n` (int, default=10): Top N markers per cluster

**Outputs**:
- `all_markers.csv`: All markers
- `top10_markers.csv`: Top N markers per cluster
- `marker_summary.csv`: Statistics per cluster
- Dot plot, heatmap, feature plots

### 8. run_pathway_enrichment

GO and KEGG pathway enrichment analysis.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `deg_csv` (str): Path to DEG results CSV (required)
- `organism` (str, default="human"): "human" or "mouse"
- `ontology` (str, default="BP"): GO ontology ("BP", "MF", "CC")
- `pvalue_cutoff` (float, default=0.05): P-value threshold
- `qvalue_cutoff` (float, default=0.2): Q-value threshold

**Outputs**:
- `go_enrichment_all.csv`: GO results
- `kegg_enrichment.csv`: KEGG results
- `enrichment_summary.csv`: Statistics
- Dot plots, bar plots, network plots

### 9. run_dim_reduction

Dimensionality reduction for visualization.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `methods` (list, default=["PCA", "UMAP", "tSNE"]): Methods to run
- `dims` (int, default=30): Number of dimensions
- `n_neighbors` (int, default=30): UMAP neighbors
- `min_dist` (float, default=0.3): UMAP min distance

**Outputs**:
- `seurat_with_reductions.rds`: Seurat with embeddings
- `pca_embeddings.csv`, `umap_embeddings.csv`, `tsne_embeddings.csv`
- Visualization plots for all methods

### 10. run_subset_cells

Subset cells based on metadata criteria.

**Parameters**:
- `input_rds` (str): Path to input Seurat RDS file
- `subset_column` (str): Metadata column to filter on
- `subset_values` (list): Values to keep
- `invert` (bool, default=False): If True, exclude specified values

**Outputs**:
- `seurat_subset.rds`: Subsetted Seurat object
- `retained_cells.csv`: Cell barcodes
- `subset_statistics.csv`: Before/after statistics
- UMAP comparison plots

## Design Principles

This server follows critical design patterns from ImmuneAgent sessions 3-5:

1. **External R Script Execution**: Never embed R logic in Python
2. **Subprocess Pattern**: Use `subprocess.run(["Rscript", script_path])`
3. **Timeout Management**: 3600s timeout for all tools (configurable)
4. **Structured JSON Responses**: Always return status, files, and messages
5. **stdio Transport**: Use stdio (NOT SSE) for MCP communication
6. **Batch Processing**: Support multiple samples via Harmony integration
7. **File Tracking**: Collect all generated files (RDS, CSV, PDF, PNG)

## Output Organization

All outputs are organized by tool name:

```
output/
├── qc_filtering/
│   ├── seurat_filtered.rds
│   ├── qc_statistics.csv
│   └── qc_*.pdf
├── normalization_sct/
│   ├── seurat_normalized.rds
│   ├── variable_features.csv
│   └── *.pdf
├── clustering_analysis/
│   ├── seurat_clustered.rds
│   ├── cluster_assignments.csv
│   └── umap_*.pdf
└── ...
```

## Integration with ImmuneAgent

### LangGraph Workflow Integration (Stage 5)

```python
# In agent/usecases/ImmuneAgent/graph/planning_graph.py

from mcp.client import Client

# Stage 5: Research Planning - Tool Selection
async def research_planning_node(state: ImprovedCellState):
    # Connect to mcp_r_scrna
    mcp_client = Client("mcp_r_scrna", transport="stdio")

    # Execute scRNA-seq workflow based on user query
    if "clustering" in state.optimized_questions[0].lower():
        result = await mcp_client.call_tool("run_clustering_analysis", {
            "input_rds": state.seurat_file_path,
            "resolution": 0.8
        })
        state.mcp_tool_results["clustering"] = result

    if "differential expression" in state.optimized_questions[1].lower():
        result = await mcp_client.call_tool("run_deg_analysis", {
            "input_rds": state.seurat_file_path,
            "ident_1": "0",
            "ident_2": "1"
        })
        state.mcp_tool_results["deg"] = result

    return state
```

### Registration in agent/config/config.py

```python
MCP_SERVERS = {
    "mcp_r_scrna": {
        "port": 8090,
        "transport": "stdio",
        "command": ["python", "/Users/ahleyliu/LocalDoc/ImmuneAgent/mcp_r_scrna/scrna_mcp_server.py"],
        "description": "Single-cell RNA-seq preprocessing and analysis",
        "tools": 10
    }
}
```

## Testing

```bash
cd /Users/ahleyliu/LocalDoc/ImmuneAgent/mcp_r_scrna
python -m pytest tests/test_scrna_tools.py -v
```

## Troubleshooting

### R Package Installation Issues

If Seurat or Harmony fails to install:

```r
# Use binary packages (faster)
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("Seurat", type = "binary")

# Or compile from source (slower but latest)
install.packages("Seurat", type = "source")
```

### Memory Issues

For large datasets (>50K cells):

```python
# Increase timeout and use subset_cells first
result = client.call_tool("run_subset_cells", {
    "input_rds": large_rds,
    "subset_column": "seurat_clusters",
    "subset_values": ["0", "1", "2"]  # Focus on specific clusters
})
```

### UMAP Not Found

Ensure PCA or Harmony is run before clustering:

```python
# Always run normalization (includes PCA) first
client.call_tool("run_normalization_sct", {...})
# Then clustering (includes UMAP)
client.call_tool("run_clustering_analysis", {...})
```

## References

- **Seurat**: [https://satijalab.org/seurat/](https://satijalab.org/seurat/)
- **Harmony**: [https://github.com/immunogenomics/harmony](https://github.com/immunogenomics/harmony)
- **DoubletFinder**: [https://github.com/chris-mcginnis-ucsf/DoubletFinder](https://github.com/chris-mcginnis-ucsf/DoubletFinder)
- **clusterProfiler**: [https://bioconductor.org/packages/clusterProfiler/](https://bioconductor.org/packages/clusterProfiler/)

## Version History

- **v1.0.0** (2025-10-08): Initial release
  - 10 scRNA-seq analysis tools
  - stdio transport
  - External R script execution pattern
  - Comprehensive test suite

## License

Part of ImmuneAgent project. For internal use.

## Contact

For issues or questions, refer to:
- `/Users/ahleyliu/LocalDoc/ImmuneAgent/.claude/tasks/context_session_10_r_bioinformatics.md`
- `/Users/ahleyliu/LocalDoc/ImmuneAgent/PRP_BCELL_MCP.md`
