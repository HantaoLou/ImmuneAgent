# NetTCR Best Practices Guide

This document provides best practice recommendations for using NetTCR for TCR-peptide binding prediction.

## Table of Contents

1. [Input Data Preparation](#input-data-preparation)
2. [Tool Selection Strategy](#tool-selection-strategy)
3. [Interpreting Prediction Results](#interpreting-prediction-results)
4. [Threshold Setting Guide](#threshold-setting-guide)
5. [Performance Optimization](#performance-optimization)
6. [Error Handling](#error-handling)
7. [Integration with Downstream Analysis](#integration-with-downstream-analysis)

---

## Input Data Preparation

### Format Selection Recommendations

| Format | Recommendation | Use Case | Pros | Cons |
|--------|----------------|----------|------|------|
| Native 7-CDR | ⭐⭐⭐⭐⭐ | Complete CDR1-3 data available | Highest accuracy | Requires complete data |
| Traditional CDR3+V | ⭐⭐⭐⭐ | Only CDR3 and V gene available | Minimal data requirement | Relies on CDR1/2 inference |

### Data Quality Checklist

```markdown
□ Peptide sequence length 8-15 amino acids (MHC-I)
□ CDR3 sequence length 5-25 amino acids
□ Amino acid sequences contain only standard 20 amino acids
□ V gene names follow IMGT nomenclature (e.g., TRAV1-2)
□ No duplicate records (optional: keep unique sequences)
□ Consistent encoding (UTF-8 recommended)
```

### Input Validation Workflow

```
Raw Data → convert_tcr_to_nettcr_format → validate_tcr_input → Prediction
```

**Examples:**

```json
// Step 1: Format conversion (if needed)
{
    "tool": "convert_tcr_to_nettcr_format",
    "arguments": {
        "input_data": "/path/to/tcr_data.csv",
        "peptide": "GILGFVFTL",
        "output_dir": "/path/to/output"
    }
}

// Step 2: Data validation
{
    "tool": "validate_tcr_input",
    "arguments": {
        "test_file": "/path/to/output/tcr_data_nettcr_format.csv"
    }
}
```

---

## Tool Selection Strategy

### Decision Tree

```
Start
  │
  ├─ Need statistical reports?
  │   ├─ Yes → predict_tcr_binding_complete
  │   └─ No ↓
  │
  ├─ Large-scale high-throughput analysis?
  │   ├─ Yes → predict_tcr_binding_fast
  │   └─ No ↓
  │
  └─ Need legacy API compatibility?
      ├─ Yes → predict_tcr_binding_ensemble
      └─ No → predict_tcr_binding_fast
```

### Tool Comparison

| Tool | Output Content | Use Case | Performance |
|------|----------------|----------|-------------|
| `predict_tcr_binding_fast` | Prediction CSV | High-throughput screening | Fastest |
| `predict_tcr_binding_complete` | CSV + Statistics + Report | Complete analysis | Moderate |
| `predict_tcr_binding_ensemble` | Prediction CSV | API compatibility | Fast |

### Recommended Workflows

#### Initial Analysis

```json
{
    "tool": "predict_tcr_binding_complete",
    "arguments": {
        "test_file": "/path/to/input.csv",
        "output_dir": "/path/to/output",
        "rank_threshold": 2.0
    }
}
```

#### Batch Processing

```json
{
    "tool": "predict_tcr_binding_fast",
    "arguments": {
        "test_file": "/path/to/large_input.csv",
        "output_file": "/path/to/output.csv",
        "percentile_rank": true
    }
}
```

---

## Interpreting Prediction Results

### Score Interpretation

| Score Range | Binding Probability | Recommended Action |
|-------------|---------------------|-------------------|
| 0.8 - 1.0 | Very High | High-priority candidate |
| 0.6 - 0.8 | High | Priority for validation |
| 0.4 - 0.6 | Moderate | Optional validation |
| 0.2 - 0.4 | Low | Typically ignore |
| 0.0 - 0.2 | Very Low | Unlikely to bind |

### Percentile Rank Interpretation

| Rank Range | Interpretation | Classification |
|------------|----------------|----------------|
| 0 - 0.5% | Better than 99.5% of negative controls | Strong binder |
| 0.5 - 2.0% | Better than 98% of negative controls | Moderate binder |
| 2.0 - 5.0% | Better than 95% of negative controls | Weak binder |
| > 5.0% | Below 95% of negative controls | Non-binder |

**Note:** Percentile rank is only available for 26 pre-trained peptides.

### Output File Descriptions

#### predictions.csv

| Column | Description | Usage |
|--------|-------------|-------|
| peptide | Target peptide | Grouping analysis |
| A1-A3, B1-B3 | TCR CDR sequences | Source tracing |
| score | Binding score | Sorting and filtering |
| percentile_rank | Percentile ranking | Normalized comparison |
| is_binder | Is binder | Binary classification |

#### statistics.csv

| Column | Description | Usage |
|--------|-------------|-------|
| peptide | Peptide sequence | Identifier |
| total_count | Total TCR count | Statistics |
| binder_count | Number of binders | Statistics |
| binder_rate | Binding rate | Evaluation |
| avg_score | Average score | Comparison |
| avg_rank | Average ranking | Comparison |

---

## Threshold Setting Guide

### Recommended rank_threshold Values

| Application Scenario | Recommended Threshold | Description |
|---------------------|----------------------|-------------|
| Stringent screening | 0.5% - 1.0% | For confirmatory experiments, reduce false positives |
| Standard screening | 2.0% | Balance sensitivity and specificity |
| Relaxed screening | 5.0% | Exploratory analysis, don't miss potential candidates |

### Threshold Impact Analysis

```
rank_threshold = 0.5%:
  - High precision, low recall
  - Suitable for validation experiments with limited resources
  
rank_threshold = 2.0%:
  - Balanced precision and recall
  - Suitable for routine screening
  
rank_threshold = 5.0%:
  - High recall, low precision
  - Suitable for exploratory research
```

### Selecting Threshold Based on Peptide Type

```python
# Pre-trained peptides (support percentile ranking)
{
    "rank_threshold": 2.0,  # Use percentile ranking
    "percentile_rank": true
}

# Novel peptides (pan-specific model)
{
    # percentile_rank will return NaN, rely on score
    "score_threshold": 0.5  # Recommend using score threshold
}
```

---

## Performance Optimization

### Batch Processing Optimization

#### Batching Strategy

```python
# For very large datasets (> 100,000 TCRs), recommend batch processing
batch_size = 10000

for i in range(0, total_records, batch_size):
    batch_file = f"batch_{i//batch_size}.csv"
    # Process each batch
    {
        "tool": "predict_tcr_binding_fast",
        "arguments": {
            "test_file": batch_file,
            "output_file": f"output_batch_{i//batch_size}.csv"
        }
    }
```

#### Memory Optimization

```python
# For memory-constrained environments
{
    "tool": "predict_tcr_binding_fast",
    "arguments": {
        "test_file": "/path/to/input.csv",
        "output_file": "/path/to/output.csv",  # Specify output to avoid memory accumulation
        "percentile_rank": false  # Disabling percentile ranking can slightly improve speed
    }
}
```

### Performance Benchmarks

| Data Scale | Estimated Time | Memory Requirement |
|------------|---------------|-------------------|
| 100 TCRs | ~2 seconds | < 100 MB |
| 1,000 TCRs | ~15 seconds | ~200 MB |
| 10,000 TCRs | ~2 minutes | ~500 MB |
| 100,000 TCRs | ~15 minutes | ~1 GB |

---

## Error Handling

### Common Errors and Solutions

#### Error 1: Missing Required Column

```
Error: Missing required column 'peptide'
```

**Solution:**

```python
# Check CSV column names
# Ensure one of the following columns exists:
# - peptide (recommended)
# - Peptide
# - PEPTIDE
# - epitope
```

#### Error 2: Invalid V Gene Format

```
Error: Invalid V gene format: 'TRAV1' (expected 'TRAV1-2')
```

**Solution:**

```python
# Use correct IMGT nomenclature
# Correct: TRAV1-2, TRBV7-9
# Incorrect: TRAV1, TRBV7

# Or use convert_tcr_to_nettcr_format for automatic handling
{
    "tool": "convert_tcr_to_nettcr_format",
    "arguments": {
        "input_data": "/path/to/data.csv",
        "peptide": "GILGFVFTL"
    }
}
```

#### Error 3: Invalid Peptide Sequence

```
Error: Invalid peptide sequence: 'GILGFVFTLX' (contains non-standard amino acid)
```

**Solution:**

```python
# Ensure only standard 20 amino acids are used
# Check and remove non-standard characters
valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
peptide = ''.join(c for c in peptide if c in valid_aa)
```

#### Error 4: CDR3 Sequence Too Long

```
Warning: CDR3 sequence exceeds recommended length (30 > 25)
```

**Solution:**

```python
# Check if CDR3 extraction is correct
# CDR3 is typically 5-25 amino acids
# Overly long sequences may indicate extraction errors
```

### Error Handling Best Practices

```python
# Recommended error handling workflow
def predict_with_validation(input_file):
    # 1. Validate input
    validation = call_tool("validate_tcr_input", {"test_file": input_file})
    
    if not validation["valid"]:
        print(f"Input validation failed: {validation['errors']}")
        return None
    
    # 2. Check warnings
    if validation["warnings"]:
        print(f"Warnings: {validation['warnings']}")
    
    # 3. Execute prediction
    try:
        result = call_tool("predict_tcr_binding_complete", {
            "test_file": input_file,
            "output_dir": "/path/to/output"
        })
        return result
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None
```

---

## Integration with Downstream Analysis

### Integration with T Cell Analysis Pipeline

```
TCR Data → NetTCR Prediction → TCR Data Integration → Visualization Analysis
     ↓              ↓                    ↓                    ↓
  CSV Input    predictions.csv       RDS Integration      UMAP Charts
```

#### Integration Example

```json
// Step 1: NetTCR prediction
{
    "tool": "predict_tcr_binding_complete",
    "arguments": {
        "test_file": "/path/to/tcr_data.csv",
        "output_dir": "/path/to/nettcr_output"
    }
}

// Step 2: Integration with single-cell data
{
    "tool": "integrate_tcr_data_complete",
    "arguments": {
        "nettcr_predictions": "/path/to/nettcr_output/predictions.csv",
        "input_rds": "/path/to/meta.rds",
        "output_path": "/path/to/integrated.rds"
    }
}

// Step 3: Visualization
{
    "tool": "tcr_binding_visualization",
    "arguments": {
        "input_rds": "/path/to/integrated.rds",
        "output_path": "/path/to/figures"
    }
}
```

### Integration with Clonotype Analysis

```json
{
    "tool": "tcr_clonotype_analysis",
    "arguments": {
        "input_rds": "/path/to/integrated.rds",
        "nettcr_predictions": "/path/to/nettcr_output/predictions.csv",
        "output_path": "/path/to/clonotype_output"
    }
}
```

### Data Flow Diagram

```
┌─────────────────┐
│   Raw TCR Data   │
│   (CSV/TSV)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ convert_tcr_to_ │
│ nettcr_format   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ validate_tcr_   │
│ input           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ predict_tcr_    │
│ binding_complete│
└────────┬────────┘
         │
         ├──────────────────┐
         ▼                  ▼
┌─────────────────┐ ┌─────────────────┐
│ predictions.csv │ │ statistics.csv  │
└────────┬────────┘ └─────────────────┘
         │
         ▼
┌─────────────────┐
│ integrate_tcr_  │
│ data_complete   │
└────────┬────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ tcr_binding_    │ │ tcr_clonotype_  │ │ tcell_other     │
│ visualization   │ │ analysis        │ │ analysis        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Summary

### Key Recommendations

1. **Prioritize Data Preparation** - Always validate input using `validate_tcr_input`
2. **Choose the Right Tool** - Select fast/complete tools based on requirements
3. **Understand Threshold Meanings** - Relationship between rank_threshold and percentile_rank
4. **Integrated Analysis** - Combine with T cell analysis toolchain for comprehensive insights
5. **Iterative Optimization** - Adjust parameters and analysis strategies based on preliminary results

### Quick Reference Card

```
┌────────────────────────────────────────────────────────┐
│              NetTCR Quick Reference                │
├────────────────────────────────────────────────────────┤
│ Best Input Format: Native 7-CDR (peptide, A1-A3, B1-B3)│
│ Recommended Threshold: rank_threshold = 2.0%           │
│ High-throughput: predict_tcr_binding_fast              │
│ Complete Analysis: predict_tcr_binding_complete        │
│ Pre-trained Peptides: 26 (support percentile ranking)  │
│ Pan-specific: Any peptide (no percentile ranking)      │
└────────────────────────────────────────────────────────┘
```

For more questions, please refer to [examples.md](./examples.md) or contact the support team.
