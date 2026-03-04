# T Cell Analysis Best Practices Guide

This document provides best practice recommendations for T cell single-cell analysis.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Tool Execution Order](#tool-execution-order)
3. [Parameter Optimization](#parameter-optimization)
4. [Result Interpretation](#result-interpretation)
5. [Troubleshooting](#troubleshooting)
6. [Performance Optimization](#performance-optimization)

---

## Data Preparation

### Input Data Requirements

| Data Type | Format | Requirements | Description |
|-----------|--------|--------------|-------------|
| Single-cell Data | RDS | Seurat object | Must contain UMAP coordinates |
| Prediction Results | CSV | NetTCR output | Contains score, rank, binding columns |

### Data Quality Checklist

```markdown
□ RDS file loads correctly
□ Seurat object contains UMAP coordinates
□ Metadata column names follow conventions (no special characters)
□ CSV file encoding is UTF-8
□ Barcode column format is consistent
□ Cell count is reasonable (recommend > 500)
```

### Barcode Matching Best Practices

```python
# Recommended barcode formats
# Format 1: Simple barcode
barcode: "AAACCTGAGAACTGTA-1"

# Format 2: With sample prefix
barcode: "Sample1_AAACCTGAGAACTGTA-1"

# Matching field configuration
{
    "csv_fields": "barcode",
    "rds_fields": "barcode"
}

# If CSV and RDS barcode formats differ
{
    "csv_fields": "sequence_id",  # Column name in CSV
    "rds_fields": "cell_id"       # Column name in RDS metadata
}
```

---

## Tool Execution Order

### Required Sequence

```
┌─────────────────────────────────────────────────────────────┐
│                 Tool Execution Order (Strict)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1 (Must be first)                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  integrate_tcr_data_complete                         │   │
│  │  - Integrate TCR prediction results                  │   │
│  │  - Update Seurat metadata                            │   │
│  │  - Execute UMAP and annotation                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  Step 2 (Can execute in parallel)                           │
│  ┌───────────┬───────────┬───────────┬─────────────────┐   │
│  │ Cell Type │ Marker    │ Trajectory│ Binding Viz/    │   │
│  │ Viz       │ Dotplot   │ Analysis  │ Clonotype       │   │
│  └───────────┴───────────┴───────────┴─────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Graph

```yaml
# Tool dependencies
integrate_tcr_data_complete: []  # No dependencies, must execute first

tcell_celltype_visualization:
  - integrate_tcr_data_complete

tcell_marker_dotplot_analysis:
  - integrate_tcr_data_complete

tcell_trajectory_analysis:
  - integrate_tcr_data_complete

tcr_binding_visualization:
  - integrate_tcr_data_complete
  - nettcr.predict_tcr_binding_*  # Requires prediction results first

tcr_clonotype_analysis:
  - integrate_tcr_data_complete
```

### Parallel Execution Recommendations

```python
# Analysis tools that can be executed in parallel
parallel_analysis = [
    "tcell_celltype_visualization",
    "tcell_marker_dotplot_analysis",
    "tcell_trajectory_analysis",
    "tcr_binding_visualization",
    "tcr_clonotype_analysis"
]

# All these tools depend on integrate_tcr_data_complete completing first
```

---

## Parameter Optimization

### integrate_tcr_data_complete

#### Barcode Matching Parameters

```python
# Scenario 1: Standard 10X format
{
    "csv_fields": "barcode",
    "rds_fields": "barcode"
}

# Scenario 2: With sample prefix
{
    "csv_fields": "sample_id,cell_barcode",
    "rds_fields": "orig.ident,barcode"
}

# Scenario 3: Custom matching
{
    "csv_fields": "sequence_id",  # Unique identifier in CSV
    "rds_fields": "cell_name"     # Corresponding column in RDS
}
```

#### Performance-related Parameters

```python
# Quick integration (skip optional steps)
{
    "skip_umap": true,       # If UMAP coordinates already exist
    "skip_annotation": true  # If only merging metadata is needed
}

# Complete integration
{
    "skip_umap": false,
    "skip_annotation": false
}
```

### tcell_trajectory_analysis

#### Parameter Tuning Guide

| Parameter | Default | Adjustment Suggestion |
|-----------|---------|----------------------|
| num_dim | 50 | Reduce with noisy data (30-40) |
| cluster_resolution | 0.001 | Increase if trajectory is too complex (0.005-0.01) |
| min_gene_cells | 3 | Reduce for rare gene analysis (1-2) |

```python
# High-resolution trajectory (more branches)
{
    "num_dim": 50,
    "cluster_resolution": 0.005,
    "min_gene_cells": 3
}

# Simplified trajectory (main branches)
{
    "num_dim": 30,
    "cluster_resolution": 0.0005,
    "min_gene_cells": 5
}
```

### tcr_clonotype_analysis

#### Grouping Strategy

```python
# Group by sample
{
    "group_by": "orig.ident",
    "top_n": 20
}

# Group by condition
{
    "group_by": "condition",
    "top_n": 30
}

# Group by binding status
{
    "group_by": "is_binder",
    "top_n": 50
}
```

---

## Result Interpretation

### Cell Type Annotation

#### Expected Distribution

| Cell Type | Typical Proportion | Abnormal Situation |
|-----------|-------------------|-------------------|
| Naive CD8/CD4 | 20-40% | Too high may indicate sample quality issues |
| Effector CD8 | 10-30% | Elevated after acute infection/vaccination |
| Memory CD8 | 5-20% | Chronic infection/previous exposure |
| Exhausted CD8 | 1-10% | Tumor/chronic infection |
| Treg | 2-10% | Too high may indicate immunosuppression |

#### Marker Gene Validation

```python
# Validate annotation quality
# 1. Check marker gene dotplot
# 2. Confirm marker genes are highly expressed in corresponding subtypes
# 3. Check for "double positive" cells

# Common issues
# - Unclear marker gene expression: may need to adjust annotation parameters
# - Many "Unknown" cells: consider adding marker genes or adjusting thresholds
```

### Trajectory Analysis

#### Pseudotime Interpretation

```python
# Pseudotime value range: 0 to maximum
# 0 = starting point (root cell)
# Maximum = terminal state

# Typical differentiation paths
CD8 Path:
  Naive (pseudotime ≈ 0)
    ↓
  Effector (pseudotime ≈ medium)
    ↓
  Memory/Exhausted (pseudotime ≈ high)

CD4 Path:
  Naive (pseudotime ≈ 0)
    ↓
  Differentiation branches (Th1/Th2/Th17/Treg/Tfh)
```

#### Trajectory Gene Analysis

```python
# Moran's I interpretation
> 0.5  : Strong spatial autocorrelation (key differentiation genes)
0.2-0.5: Moderate correlation
< 0.2  : Weak correlation or irrelevant

# Focus on genes with high Moran's I values
# These genes may drive the differentiation process
```

### Clonotype Analysis

#### Diversity Index Interpretation

| Index | Range | High Value Meaning | Low Value Meaning |
|-------|-------|-------------------|------------------|
| Shannon | 0-10+ | High diversity | Clonal expansion |
| Simpson | 0-1 | High diversity | Dominant clone |
| Clonality | 0-1 | Clonal | Diverse |

```python
# Typical scenarios
# Acute infection: Specific clones expand massively, clonality increases
# Healthy control: High diversity, low clonality
# Tumor infiltrating: May show dominant clones
```

#### Expanded Clone Definition

```python
# Common definitions
# Expanded clone: frequency > 1%
# Medium clone: frequency 0.1-1%
# Rare clone: frequency < 0.1%

# In results, focus on
# - Total proportion of Top 10 clonotypes
# - Clonotype overlap across different conditions
```

---

## Troubleshooting

### Issue 1: Cell Count Mismatch After Integration

```
Problem: integrated_cells << total_cells
```

**Troubleshooting Steps:**

1. Check if barcode formats are consistent
2. Verify `csv_fields` and `rds_fields` configuration
3. Check for special characters or spaces

**Solution:**

```python
# Print barcode samples for comparison
# CSV: ["AAACCTGAGAACTGTA-1", "Sample1_AAACCTGAGAACTGTA-1"]
# RDS: ["AAACCTGAGAACTGTA_1", "AAACCTGAGAACTGTA-1"]

# Adjust matching strategy
{
    "csv_fields": "barcode",
    "rds_fields": "barcode",
    "force_standardize": true  # Standardize barcode format
}
```

### Issue 2: Abnormal Cell Type Annotation

```
Problem: Many cells labeled as "Unknown"
```

**Troubleshooting Steps:**

1. Check if marker genes exist in the data
2. Verify data quality (genes detected, UMI counts)
3. Check for batch effects

**Solution:**

```python
# Try different annotation levels
{
    "annotation_level": "immune"  # Start with coarse-grained annotation
}

# Or manually check marker gene expression
# Confirm if data supports subtype distinction
```

### Issue 3: Trajectory Analysis Failure

```
Problem: Trajectory analysis timeout or error
```

**Troubleshooting Steps:**

1. Check if cell count is sufficient (recommend > 500)
2. Verify there are enough differentiation states
3. Check memory usage

**Solution:**

```python
# Reduce computational complexity
{
    "num_dim": 30,           # Reduce number of principal components
    "cluster_resolution": 0.0005,  # Lower resolution
    "min_gene_cells": 5      # Increase gene filtering threshold
}

# Or pre-filter cells
# Keep only T cells for trajectory analysis
```

### Issue 4: Binding Visualization No Results

```
Problem: tcr_binding_visualization output is empty
```

**Troubleshooting Steps:**

1. Confirm RDS contains prediction columns
2. Check if column names are correct (bind_predict, score, etc.)
3. Verify prediction value range

**Solution:**

```python
# Ensure integration step includes prediction data
# Step 1: Prediction
{
    "tool": "predict_tcr_binding_complete",
    "arguments": {
        "test_file": "tcr_data.csv",
        "output_dir": "nettcr_output"
    }
}

# Step 2: Integration (must use prediction output)
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "input_csv": "nettcr_output/predictions.csv",  # Use prediction results
        "input_rds": "meta.rds",
        "output_path": "integrated.rds"
    }
}

# Step 3: Visualization
{
    "tool": "tcr_binding_visualization",
    "arguments": {
        "input_file": "integrated.rds",
        "base_dir": "binding_output"
    }
}
```

---

## Performance Optimization

### Large Dataset Processing

```python
# When cell count > 50,000

# Strategy 1: Pre-filtering
# Keep only T cells for downstream analysis
# Can use immune_isolate_cells tool

# Strategy 2: Lower resolution
{
    "num_dim": 30,
    "cluster_resolution": 0.0005
}

# Strategy 3: Batch processing
# Analyze separately by sample or condition
```

### Memory Optimization

```python
# Memory-constrained environments

# 1. Skip unnecessary steps
{
    "skip_umap": true,
    "skip_annotation": true
}

# 2. Use fewer principal components
{
    "num_dim": 20
}

# 3. Regularly clean intermediate files
```

### Runtime Estimates

| Tool | 5,000 cells | 20,000 cells | 50,000 cells |
|------|-------------|--------------|--------------|
| integrate_tcr_data_complete | 2-3 min | 5-10 min | 15-30 min |
| tcell_celltype_visualization | 30 sec | 1-2 min | 3-5 min |
| tcell_marker_dotplot_analysis | 20 sec | 1 min | 2-3 min |
| tcell_trajectory_analysis | 3-5 min | 10-15 min | 30-60 min |
| tcr_binding_visualization | 30 sec | 1-2 min | 3-5 min |
| tcr_clonotype_analysis | 1 min | 2-3 min | 5-10 min |

---

## Integration with Other Tools

### Integration with NetTCR

```python
# Standard workflow
NetTCR Prediction → integrate_tcr_data_complete → tcr_binding_visualization

# Notes
# 1. NetTCR output must be in CSV format
# 2. Must contain barcode column for matching
# 3. Prediction columns are automatically added to RDS metadata
```

### Integration with IgBLAST

```python
# Integration after V(D)J analysis
IgBLAST analyze_vdj_batch → integrate_tcr_data_complete

# Notes
# 1. AIRR format output can be directly integrated
# 2. CDR3 sequences can be used for clonotype analysis
# 3. V/D/J gene information is preserved in metadata
```

### Integration with immune Module

```python
# Broader immune cell analysis
T Cell Analysis → immune_comprehensive_analysis

# Workflow
1. First use tcell tools for T cell-specific analysis
2. Then use immune tools for overall immune cell analysis
3. Compare results for a more comprehensive perspective
```

---

## Summary

### Key Points

1. **Execution Order is Critical** - `integrate_tcr_data_complete` must execute first
2. **Data Quality Matters** - Ensure barcode matching and data quality
3. **Parameters Need Tuning** - Adjust parameters based on data characteristics
4. **Results Need Validation** - Interpret results with biological knowledge

### Quick Reference Card

```
┌────────────────────────────────────────────────────────────┐
│              T Cell Analysis Quick Reference               │
├────────────────────────────────────────────────────────────┤
│ Must run first: integrate_tcr_data_complete               │
│ Visualization:  tcell_celltype_visualization              │
│                 tcell_marker_dotplot_analysis             │
│                 tcr_binding_visualization                 │
│ Analysis:       tcell_trajectory_analysis                 │
│                 tcr_clonotype_analysis                    │
│────────────────────────────────────────────────────────────│
│ Minimum cells: 500+                                       │
│ Recommended cells: 5,000-50,000                           │
│────────────────────────────────────────────────────────────│
│ Input format: RDS (Seurat) + CSV (NetTCR/IgBLAST)         │
│ Output format: RDS + PNG + CSV                            │
└────────────────────────────────────────────────────────────┘
```

For more examples, please refer to [examples.md](./examples.md).
