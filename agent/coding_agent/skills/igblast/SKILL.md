---
name: igblast
description: V(D)J recombination analysis via IgBLAST + ChangeO. Returns streaming_task (async). Use for antibody/TCR gene assignment, CDR extraction, SHM detection.
---

## IgBLAST V(D)J Analysis

**Response mode**: Streaming async (`streaming_task` — must poll streaming URL)

## Tools

| Tool | Use case |
|------|----------|
| `analyze_vdj_batch` | Core V(D)J analysis → AIRR format output |
| `extract_cdr3_from_airr` | Extract CDR3 from AIRR results |

## Workflow

1. Call `analyze_vdj_batch` with sequences file
2. Poll the streaming URL until `type=result`
3. Optionally call `extract_cdr3_from_airr` on the AIRR output

## Key Parameters (analyze_vdj_batch)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | str | **required** | FASTA or CSV file path |
| `organism` | str | `human` | human / mouse / rhesus_monkey |
| `receptor_type` | str | `Ig` | Ig (BCR/antibody) or TCR |
| `locus` | str | `IGH` | IGH/IGK/IGL (BCR) or TRA/TRB (TCR) |
| `timeout` | int | 7200 | Seconds |

## Input Formats

**FASTA** (preferred): Standard FASTA with sequence headers starting with `>`

**CSV** (auto-converted): Must have sequence columns:
- BCR: Heavy_DNA, Light_DNA, Heavy, Light, VH, VL, or sequence
- TCR: alpha_dna, beta_dna, TRA, TRB, CDR3a, CDR3b

## Output

AIRR format TSV with columns: sequence_id, v_call, d_call, j_call, cdr1, cdr2, cdr3, junction, v_identity

## Gotchas

1. Response is `streaming_task` — you MUST poll the streaming URL until `type=result`
2. CSV input is auto-converted to FASTA — column names must match expected aliases
3. Large files are auto-batched but may need increased timeout (default 7200s)
4. May need flat-params fix like nettcr (Session 11 note: "igblast may need same fix")
5. When server unreachable, OpenCode may fabricate results — verify AIRR output exists
