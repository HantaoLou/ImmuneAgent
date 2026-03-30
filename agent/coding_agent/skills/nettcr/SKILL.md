---
name: nettcr
description: NetTCR-2.2 TCR-peptide binding prediction. Synchronous response. Use for predicting T cell receptor binding affinity to peptides.
---

## NetTCR-2.2 TCR-Peptide Binding Prediction

**Response mode**: Synchronous dict (NOT streaming_task)

## Tools

| Tool | Use case | Recommended |
|------|----------|-------------|
| `list_available_peptides` | List 26 pretrained peptides | Discovery |
| `check_peptide_support` | Check if peptide has a model | Pre-validation |
| `validate_tcr_input` | Validate input CSV format | Recommended |
| `predict_tcr_binding_fast` | Single-model fast prediction | Screening |
| `predict_tcr_binding_complete` | Full pipeline + statistics | **Best quality** |

**Recommended**: `predict_tcr_binding_complete` (reproduces official NetTCR-2.2 results)

## Input CSV Format

Two supported formats:

**Native (recommended)**:
```
peptide,A1,A2,A3,B1,B2,B3
GILGFVFTL,TSESTM,,CAVSANSGTYKYIF,SGDLS,,CASSIRSSYEQYF
```

**Legacy**:
```
peptide,CDR3a,CDR3b,TRA_v_gene,TRB_v_gene
GILGFVFTL,CAVSANSGTYKYIF,CASSIRSSYEQYF,TRAV12-2,TRBV6-5
```

All 7 columns must exist in native format even if A1/A2/B1/B2 are empty strings.

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_file` | str | **required** | CSV file path |
| `output_dir` | str | **required** (complete) | Output directory |
| `rank_threshold` | float | 2.0 | Percentile rank cutoff |
| `percentile_rank` | bool | true | Include percentile rank |

## Workflow

1. Run `check_peptide_support` with target peptides
2. Run `validate_tcr_input` on input file
3. Run `predict_tcr_binding_complete` for full analysis

## Gotchas

1. All 7 columns (peptide, A1, A2, A3, B1, B2, B3) MUST exist even if empty — missing columns cause KeyError
2. Peptide length must be 8-15 AA (MHC-I only). Longer peptides silently produce no predictions.
3. Output CSV path is NOT the one you specify — read `result_path` from the response
4. Duplicate sequences are NOT deduplicated. Dedup upstream if needed.
5. Empty CDR1/CDR2 values become NaN floats during CSV read — the server handles this but watch for it in downstream processing
6. When MCP server is unreachable, OpenCode may fabricate results silently — always verify output files exist
