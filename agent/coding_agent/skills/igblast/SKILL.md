---
name: igblast
description: IgBLAST V(D)J gene alignment for antibody/BCR/TCR nucleotide sequences. Returns streaming_task (async). Use when identifying V/D/J genes, extracting CDR3 sequences, or producing AIRR-format output.
---

## Tools

| Tool | Description |
|------|-------------|
| `analyze_vdj_batch` | V(D)J alignment (main tool), outputs AIRR-format CSV |
| `extract_cdr3_from_airr` | Convert AIRR CSV to CDR3 sequences |

---

## `analyze_vdj_batch` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | list\|str | **required** | `[{"id":"seq1","sequence":"ATGCAG..."}]` or server-side FASTA file path |
| `organism` | str | `"human"` | `human` / `mouse` / `rabbit` / `rat` / `rhesus` / `pig` |
| `receptor_type` | str | `"Ig"` | `Ig` (antibody) or `TCR` |
| `locus` | str | `"IGH"` | `IGH` / `IGK` / `IGL` / `TRA` / `TRB` / `TRG` / `TRD` |
| `timeout` | int | `7200` | Timeout in seconds (300-14400) |

## `extract_cdr3_from_airr` Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `airr_results` | list\|str | AIRR record list, or server-side CSV/TSV/JSON file path |

## standard AIRR fields ##

`v_call`, `d_call`, `j_call`, `junction`, `junction_aa`, `productive`

## Constraints

1. **Input must be nucleotide sequences** (ATGC), not amino acid sequences
2. No `output_dir` parameter; the output path is in the `output_file` field
3. `output_file` is a server-side path; pass it directly to downstream tools like `extract_cdr3_from_airr`
4. The input data for `extract_cdr3_from_airr` must be the complete CSV file content, not a partial subset extracted from a CSV file, and must not use simulated/mock data
5. `extract_cdr3_from_airr` should only be used when the CSV file has column names in standard AIRR format
