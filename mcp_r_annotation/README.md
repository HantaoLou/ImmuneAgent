# mcp_r_annotation: Cell Type Annotation MCP Server

**Version**: 1.0.0
**Port**: 8095
**Transport**: stdio
**Timeout**: 3600s
**Priority**: HIGH

## Overview

The Cell Type Annotation MCP Server provides automated and manual cell type annotation tools for single-cell RNA-seq data. It supports:

- **SingleR reference-based annotation** with multiple curated reference datasets
- **Automated marker gene detection** using statistical tests
- **Manual annotation workflows** based on marker gene lists
- **Cross-reference validation** to verify annotation quality
- **Confidence scoring** for annotation reliability
- **Export utilities** for downstream analysis

## Quick Start

### 1. Installation

```bash
cd mcp_r_annotation

# Install R dependencies
Rscript -e "install.packages(c('Seurat', 'dplyr', 'ggplot2', 'jsonlite'))"
Rscript -e "BiocManager::install(c('SingleR', 'celldex', 'SingleCellExperiment', 'SeuratDisk'))"

# Download reference datasets (first time only)
Rscript scripts/download_references.R
```

### 2. Start Server

```bash
# Stdio transport (recommended)
python annotation_mcp_server.py
```

### 3. Test Tools

```bash
# Example: Run SingleR annotation
python tests/test_annotation_tools.py
```

## Reference Datasets

The server uses SingleR with curated reference datasets from the `celldex` package:

### Human References

| Dataset | Size | Cell Types | Recommended For |
|---------|------|------------|-----------------|
| **HumanPrimaryCellAtlasData** | 113 MB | Broad range (epithelial, immune, stromal) | General purpose |
| **BlueprintEncodeData** | 95 MB | Immune + stromal cells | Immune/stromal annotation |
| **MonacoImmuneData** | 87 MB | 29 immune cell types | **Immune cells (RECOMMENDED)** |
| **DatabaseImmuneCellExpressionData** | 72 MB | Sorted immune populations | Immune validation |
| **NovershternHematopoieticData** | 68 MB | Hematopoietic progenitors | Hematopoietic studies |

### Mouse References

| Dataset | Size | Cell Types | Recommended For |
|---------|------|------------|-----------------|
| **MouseRNAseqData** | 105 MB | Broad mouse cell types | Mouse scRNA-seq |

### Reference Management

```bash
# Download all references
Rscript scripts/download_references.R

# Download specific reference
Rscript scripts/download_references.R --dataset HumanPrimaryCellAtlasData

# Check reference status
Rscript scripts/download_references.R --check

# Re-download (if corrupted)
Rscript scripts/download_references.R --force
```

References are cached in `reference_data/` directory for fast loading.

## MCP Tools (6 tools)

### 1. run_singler_annotation

**Automated cell type annotation using SingleR**

```python
from mcp_client import MCPClient

client = MCPClient("mcp_r_annotation", port=8095)

result = client.call_tool("run_singler_annotation", {
    "input_rds": "/path/to/seurat.rds",
    "reference_dataset": "MonacoImmuneData",  # Recommended for immune cells
    "label_type": "label.main",  # or "label.fine" for detailed subtypes
    "cluster_column": "seurat_clusters"
})
```

**Parameters**:
- `input_rds`: Path to Seurat RDS file (required)
- `reference_dataset`: Reference dataset name (default: `HumanPrimaryCellAtlasData`)
  - Options: `HumanPrimaryCellAtlasData`, `BlueprintEncodeData`, `MonacoImmuneData`, `DatabaseImmuneCellExpressionData`, `NovershternHematopoieticData`, `MouseRNAseqData`
- `label_type`: Annotation granularity (default: `label.main`)
  - `label.main`: Broad cell types (e.g., "T cells", "B cells")
  - `label.fine`: Detailed subtypes (e.g., "CD4 T cells", "CD8 T cells")
- `cluster_column`: Metadata column with cluster IDs (default: `seurat_clusters`)

**Outputs**:
- `singler_annotation.csv`: Cluster-to-celltype mapping with confidence
- `annotation_summary.csv`: Cell type distribution statistics
- `seurat_with_singler.rds`: Updated Seurat object with annotations
- `plots/singler_umap.pdf`: UMAP visualization

**Metadata Added**:
- `singler_<dataset>`: Cell type annotation column
- `singler_<dataset>_confidence`: Confidence scores

---

### 2. detect_cluster_markers

**Identify cluster-specific marker genes for manual annotation**

```python
result = client.call_tool("detect_cluster_markers", {
    "input_rds": "/path/to/seurat.rds",
    "test_use": "wilcox",
    "only_pos": True,
    "min_pct": 0.25,
    "logfc_threshold": 0.5,
    "top_n": 10
})
```

**Parameters**:
- `input_rds`: Path to Seurat RDS file (required)
- `test_use`: Statistical test (default: `wilcox`)
  - Options: `wilcox` (Wilcoxon), `bimod`, `roc`, `t`, `MAST`
- `only_pos`: Only return upregulated markers (default: `True`)
- `min_pct`: Minimum fraction of cells expressing gene (default: 0.25)
- `logfc_threshold`: Minimum log2 fold change (default: 0.5)
- `top_n`: Number of top markers per cluster to highlight (default: 10)

**Outputs**:
- `all_markers.csv`: All markers with statistics (p-value, log2FC, pct.1, pct.2)
- `top_markers.csv`: Top N markers per cluster
- `marker_summary.csv`: Cluster-level marker statistics
- `plots/marker_heatmap.pdf`: Heatmap of top markers
- `plots/marker_dotplot.pdf`: Dotplot visualization
- `plots/top_marker_violins.pdf`: Violin plots for top marker per cluster
- `plots/marker_featureplots.pdf`: Feature plots on UMAP

---

### 3. annotate_by_markers

**Manual annotation based on marker gene knowledge**

```python
result = client.call_tool("annotate_by_markers", {
    "input_rds": "/path/to/seurat.rds",
    "marker_list": {
        "0": "T cells",
        "1": "B cells",
        "2": "Monocytes",
        "3": "NK cells"
    },
    "cluster_column": "seurat_clusters",
    "new_column": "manual_celltype"
})
```

**Parameters**:
- `input_rds`: Path to Seurat RDS file (required)
- `marker_list`: Dictionary mapping cluster IDs to cell types (required)
- `cluster_column`: Metadata column with cluster IDs (default: `seurat_clusters`)
- `new_column`: Name for new annotation column (default: `manual_celltype`)

**Outputs**:
- `annotation_mapping.csv`: Applied cluster-to-celltype mapping
- `annotation_summary.csv`: Cell type distribution
- `seurat_with_manual_annotation.rds`: Updated Seurat object
- `plots/manual_annotation_umap.pdf`: Before/after UMAP comparison
- `plots/cluster_composition.pdf`: Stacked bar plot

**Metadata Added**:
- `<new_column>`: Manual cell type annotations

---

### 4. validate_annotation

**Cross-reference validation for annotation quality control**

```python
# Validate against SingleR
result = client.call_tool("validate_annotation", {
    "input_rds": "/path/to/seurat.rds",
    "annotation_column1": "manual_celltype",
    "reference_dataset": "MonacoImmuneData"
})

# Compare two annotation columns
result = client.call_tool("validate_annotation", {
    "input_rds": "/path/to/seurat.rds",
    "annotation_column1": "manual_celltype",
    "annotation_column2": "singler_MonacoImmuneData"
})
```

**Parameters**:
- `input_rds`: Path to Seurat RDS file (required)
- `annotation_column1`: First annotation column to validate (required)
- `annotation_column2`: Second annotation column for comparison (optional)
- `reference_dataset`: Reference for SingleR validation if column2 not provided

**Outputs**:
- `confusion_matrix.csv`: Cell count matrix comparing annotations
- `discrepancies.csv`: Cells with mismatched annotations
- `celltype_agreement.csv`: Per-celltype agreement rates
- `plots/confusion_heatmap.pdf`: Normalized confusion matrix
- `plots/celltype_agreement.pdf`: Bar plot of agreement rates
- `plots/umap_comparison.pdf`: Side-by-side UMAP comparison

**Metrics**:
- Overall agreement rate (%)
- Per-celltype agreement rates
- Discrepancy counts

---

### 5. score_annotation_confidence

**Calculate confidence scores for annotation reliability**

```python
result = client.call_tool("score_annotation_confidence", {
    "input_rds": "/path/to/seurat.rds",
    "annotation_column": "manual_celltype",
    "marker_genes": {
        "T cells": ["CD3D", "CD3E"],
        "B cells": ["CD79A", "MS4A1"],
        "Monocytes": ["CD14", "LYZ"]
    }
})
```

**Parameters**:
- `input_rds`: Path to Seurat RDS file (required)
- `annotation_column`: Column containing cell type annotations (required)
- `marker_genes`: Optional dict mapping cell types to marker genes

**Outputs**:
- `confidence_scores.csv`: Confidence scores per cell type
- `seurat_with_confidence.rds`: Updated Seurat object
- `plots/confidence_scores.pdf`: Bar plot of overall confidence
- `plots/confidence_heatmap.pdf`: Heatmap of score components
- `plots/umap_confidence.pdf`: UMAP colored by confidence

**Confidence Components**:
1. **Cluster Homogeneity**: Entropy-based measure (0-1)
2. **Marker Expression Score**: Mean expression of marker genes (0-1)
3. **Silhouette Score**: Separation from other cell types (0-1)
4. **Overall Confidence**: Average of available components

**Metadata Added**:
- `<annotation_column>_confidence`: Confidence score per cell

---

### 6. export_annotations

**Export annotations to various formats**

```python
# Export to CSV (default)
result = client.call_tool("export_annotations", {
    "input_rds": "/path/to/seurat.rds",
    "annotation_columns": ["manual_celltype", "singler_MonacoImmuneData"],
    "export_format": "csv",
    "include_umap": True
})

# Export to h5ad for scanpy
result = client.call_tool("export_annotations", {
    "input_rds": "/path/to/seurat.rds",
    "export_format": "h5ad"
})
```

**Parameters**:
- `input_rds`: Path to Seurat RDS file (required)
- `annotation_columns`: List of columns to export (default: all annotation columns)
- `export_format`: Export format (default: `csv`)
  - Options: `csv`, `tsv`, `h5ad` (AnnData), `loom`
- `include_umap`: Include UMAP coordinates (default: `True`)

**Outputs**:
- `annotations.csv/tsv`: Metadata table with annotations
- `seurat_annotated.h5ad`: AnnData format (if format="h5ad")
- `seurat_annotated.loom`: Loom format (if format="loom")
- `export_summary.csv`: Export statistics
- `<column>_distribution.csv`: Per-column distribution stats

---

## Workflows

### Workflow 1: Automated SingleR Annotation

```python
# 1. Run SingleR with broad reference
result1 = client.call_tool("run_singler_annotation", {
    "input_rds": "data/pbmc.rds",
    "reference_dataset": "HumanPrimaryCellAtlasData",
    "label_type": "label.main"
})

# 2. Validate with immune-specific reference
result2 = client.call_tool("validate_annotation", {
    "input_rds": result1["generated_files"]["seurat_with_singler.rds"],
    "annotation_column1": "singler_HumanPrimaryCellAtlasData",
    "reference_dataset": "MonacoImmuneData"
})

# 3. Score confidence
result3 = client.call_tool("score_annotation_confidence", {
    "input_rds": result1["generated_files"]["seurat_with_singler.rds"],
    "annotation_column": "singler_HumanPrimaryCellAtlasData"
})
```

### Workflow 2: Manual Annotation with Markers

```python
# 1. Detect markers
result1 = client.call_tool("detect_cluster_markers", {
    "input_rds": "data/pbmc.rds",
    "test_use": "wilcox",
    "top_n": 10
})

# 2. Review markers and create mapping
# (Manually review result1["generated_files"]["top_markers.csv"])

# 3. Apply manual annotation
result2 = client.call_tool("annotate_by_markers", {
    "input_rds": "data/pbmc.rds",
    "marker_list": {
        "0": "CD14+ Monocytes",
        "1": "CD4 T cells",
        "2": "CD8 T cells",
        "3": "B cells",
        "4": "NK cells"
    }
})

# 4. Validate against SingleR
result3 = client.call_tool("validate_annotation", {
    "input_rds": result2["generated_files"]["seurat_with_manual_annotation.rds"],
    "annotation_column1": "manual_celltype",
    "reference_dataset": "MonacoImmuneData"
})

# 5. Score confidence with known markers
result4 = client.call_tool("score_annotation_confidence", {
    "input_rds": result2["generated_files"]["seurat_with_manual_annotation.rds"],
    "annotation_column": "manual_celltype",
    "marker_genes": {
        "CD14+ Monocytes": ["CD14", "LYZ"],
        "CD4 T cells": ["CD4", "IL7R"],
        "CD8 T cells": ["CD8A", "CD8B"],
        "B cells": ["CD79A", "MS4A1"],
        "NK cells": ["GNLY", "NKG7"]
    }
})
```

### Workflow 3: Export for Downstream Analysis

```python
# Export annotations with UMAP coordinates
result = client.call_tool("export_annotations", {
    "input_rds": "data/pbmc_annotated.rds",
    "annotation_columns": ["manual_celltype", "singler_MonacoImmuneData",
                          "manual_celltype_confidence"],
    "export_format": "csv",
    "include_umap": True
})

# Use in Python/scanpy
import pandas as pd
annotations = pd.read_csv(result["generated_files"]["annotations.csv"])
```

---

## Known Marker Genes

The server includes a curated list of known marker genes (from `config.json`):

| Cell Type | Marker Genes |
|-----------|--------------|
| T cells | CD3D, CD3E, CD3G, IL7R |
| CD4 T cells | CD4, IL7R |
| CD8 T cells | CD8A, CD8B |
| B cells | CD79A, CD79B, MS4A1, CD19 |
| Naive B cells | IGHD, TCL1A |
| Memory B cells | CD27, TNFRSF13B |
| Plasma cells | MZB1, SDC1, JCHAIN, IGKC |
| NK cells | GNLY, NKG7, NCAM1 |
| Monocytes | CD14, FCGR3A, LYZ |
| Classical Monocytes | CD14, S100A8, S100A9 |
| Non-Classical Monocytes | FCGR3A, MS4A7 |
| Macrophages | CD68, CD163, MSR1 |
| Dendritic cells | FCER1A, CD1C, CLEC10A |
| pDC | LILRA4, IRF7, CLEC4C |
| Endothelial cells | PECAM1, VWF, CLDN5 |
| Fibroblasts | COL1A1, COL3A1, DCN |
| Epithelial cells | EPCAM, KRT8, KRT18 |

---

## Configuration

Edit `config/config.json` to customize:

- Default parameters for each tool
- Reference dataset preferences
- Statistical test preferences
- Known marker gene lists
- Output directory structure

---

## Output Structure

All tool outputs are organized in `output/<tool_name>/`:

```
output/
├── run_singler_annotation/
│   ├── singler_annotation.csv
│   ├── annotation_summary.csv
│   ├── seurat_with_singler.rds
│   ├── result.json
│   └── plots/
│       └── singler_umap.pdf
├── detect_cluster_markers/
│   ├── all_markers.csv
│   ├── top_markers.csv
│   ├── marker_summary.csv
│   ├── result.json
│   └── plots/
│       ├── marker_heatmap.pdf
│       ├── marker_dotplot.pdf
│       ├── top_marker_violins.pdf
│       └── marker_featureplots.pdf
└── ...
```

---

## Troubleshooting

### Issue: Reference dataset not found

**Solution**: Download reference datasets
```bash
Rscript scripts/download_references.R
```

### Issue: SingleR annotation fails

**Possible causes**:
1. Input RDS is not a Seurat object
2. Data is not normalized (SingleR requires log-normalized counts)
3. Reference dataset is corrupted

**Solution**:
```R
# Ensure data is normalized in Seurat
seurat_obj <- NormalizeData(seurat_obj)
saveRDS(seurat_obj, "normalized.rds")
```

### Issue: Marker detection returns no markers

**Possible causes**:
1. `logfc_threshold` or `min_pct` too stringent
2. Clusters are not well-separated

**Solution**: Adjust parameters
```python
result = client.call_tool("detect_cluster_markers", {
    "input_rds": "data/pbmc.rds",
    "logfc_threshold": 0.25,  # Lower threshold
    "min_pct": 0.1            # Lower minimum percentage
})
```

### Issue: Export to h5ad fails

**Solution**: Install SeuratDisk
```R
remotes::install_github("mojaveazure/seurat-disk")
```

---

## Dependencies

### R Packages

**Required**:
- `Seurat` (>= 4.0)
- `SingleR`
- `celldex`
- `SingleCellExperiment`
- `dplyr`
- `jsonlite`
- `ggplot2`

**Optional**:
- `SeuratDisk` (for h5ad/loom export)

### Python Packages

- `fastmcp` (>= 2.12.4)
- `jsonschema`

---

## Integration with ImmuneAgent

This server integrates into the ImmuneAgent workflow at **Stage 5: Research Planning**.

Example LangGraph integration:

```python
from agent.common.factory import create_mcp_client

# Create MCP client
annotation_client = create_mcp_client("mcp_r_annotation", port=8095)

# Use in LangGraph workflow
state.mcp_tool_results["annotation"] = annotation_client.call_tool(
    "run_singler_annotation",
    {
        "input_rds": state.seurat_file,
        "reference_dataset": "MonacoImmuneData"
    }
)
```

---

## References

1. **SingleR**: Aran et al. (2019). "Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage." *Nature Immunology* 20, 163–172.

2. **celldex**: Heng et al. (2008). "The Immunological Genome Project: networks of gene expression in immune cells." *Nature Immunology* 9, 1091–1094.

3. **MonacoImmuneData**: Monaco et al. (2019). "RNA-Seq Signatures Normalized by mRNA Abundance Allow Absolute Deconvolution of Human Immune Cell Types." *Cell Reports* 26, 1627-1640.

---

## Support

For issues or questions:
- Check `output/<tool_name>/result.json` for detailed error messages
- Review R script logs in stdout
- Verify reference datasets with `Rscript scripts/download_references.R --check`

---

**Version**: 1.0.0
**Last Updated**: 2025-10-08
**Maintainer**: ImmuneAgent Team
