---
name: nettcr
description: NetTCR-2.2 TCR-peptide binding prediction. Returns synchronous dict (no streaming). Use when predicting T cell receptor (TCR) binding affinity to target peptides or filtering T cell epitopes.
---

## NetTCR-2.2 TCR-Peptide Binding Prediction

**Response mode**: Synchronous return (not streaming_task)

---

## Tools (6)

| Tool | Description | Recommended |
|------|-------------|-------------|
| `list_available_peptides` | List 26 peptides with pretrained models | Discovery |
| `check_peptide_support` | Check if a peptide has a pretrained model | Pre-validation |
| `validate_tcr_input` | Validate input CSV format | Optional |
| `predict_tcr_binding_fast` | Single-model fast prediction | Screening |
| `predict_tcr_binding_ensemble` | 20-model ensemble prediction | Good |
| `predict_tcr_binding_complete` | Full prediction (TCRbase + percentile ranking) | **Best quality** |

**Recommended**: `predict_tcr_binding_complete` (reproduces official NetTCR-2.2 web server results)

---

## Input CSV Format

```csv
peptide,A1,A2,A3,B1,B2,B3
GILGFVFTL,TSESTM,,CAVSANSGTYKYIF,SGDLS,,CASSIRSSYEQYF
```

| Column | Description | Required |
|--------|-------------|----------|
| `peptide` | Target peptide sequence | **required** |
| `A3` | TCR alpha chain CDR3 | Recommended |
| `B3` | TCR beta chain CDR3 | **key field** |
| A1/A2/B1/B2 | CDR1/CDR2 | Can be empty/NaN — treated as zero-length and zero-padded |

---

## `predict_tcr_binding_complete` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_file` | str | **required** | CSV file path (with 7 columns above) |
| `output_dir` | str | `null` | Output directory for results |
| `rank_threshold` | float | `2.0` | Percentile rank threshold for binder classification |
| `percentile_rank` | bool | `true` | Include percentile rank column |

## `predict_tcr_binding_fast` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_file` | str | **required** | CSV file path (with 7 columns above) |
| `output_file` | str | `null` | Output CSV file path |
| `percentile_rank` | bool | `true` | Include percentile rank column |

## `check_peptide_support` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `peptides` | str | **required** | Comma-separated peptide sequences |

## `validate_tcr_input` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_file` | str | **required** | CSV file path to validate |

---

## Return Value (streaming async generator)

Tools yield progress updates then a final result dict:
```json
{"type": "result", "status": "success", "message": "...", "result_path": [...], "statistics": {...}}
```

---

## Constraints

1. Input must be a CSV file (7-column format), does not accept raw sequence strings
2. NetTCR supports 26 pretrained peptides; use `check_peptide_support` to verify first
3. Results are returned as streaming async generator (progress + final result)
