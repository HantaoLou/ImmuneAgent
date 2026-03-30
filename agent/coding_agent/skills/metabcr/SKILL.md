---
name: metabcr
description: MetaBCR antibody-antigen binding prediction. Synchronous response. Use for predicting antibody binding affinity to antigens (flu, SARS, RSV, HIV).
---

## MetaBCR Antibody-Antigen Binding Prediction

**Response mode**: Synchronous dict (NOT streaming_task)

## Tools

| Tool | Description |
|------|-------------|
| `metabcr` | Antibody-antigen binding prediction via deep learning ensemble |

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `antibody_file` | str | **required** | CSV file path with antibody sequences (heavy_chain + light_chain columns) |
| `antigen_file` | str | **required** | CSV file path with antigen sequences (variant_name + variant_seq columns) |
| `antigen_name` | str | `"flu"` | Antigen type: `flu` / `sars` / `rsv` / `hiv` |
| `task_name` | str | `"bind"` | Prediction task: `bind` (binding) or `neu` (neutralization) |
| `config_date` | str | `"250312"` | Model config date stamp (must match antigen-task combination) |
| `output_file_path` | str | null | Output directory — pass an ABSOLUTE path (the working directory) so results are saved there |

## Input Format

**CSV** with columns:
- `heavy_chain` (aliases: Heavy, VH)
- `light_chain` (aliases: Light, VL)
- `sequence_id` (optional)

## Output

Returns `{"type": "result", "status": "success", "output_file": "<path>", "total_antibodies": N}`.
Output CSV is written to `<output_file_path>/MetaBcr/<task_name>/`.

## RSV Binding/Neutralization Prediction

IMPORTANT: Always use `os.getcwd()` or `$(pwd)` to get the absolute working directory and pass it as `output_file_path`.

### RSV Binding (bash: WD=$(pwd))
metabcr(antibody_file="/abs/path/rsv_abs.csv", antigen_file="/abs/path/rsv_antigens.csv", antigen_name="rsv", task_name="bind", config_date="250224", output_file_path="/abs/path/to/output/dir/")

### RSV Neutralization
metabcr(antibody_file="/abs/path/rsv_abs.csv", antigen_file="/abs/path/rsv_antigens.csv", antigen_name="rsv", task_name="neu", config_date="250225", output_file_path="/abs/path/to/output/dir/")

## Available Antigen Sequences
Pre-built antigen files in `config/antigens/`:
- rsv_antigens.csv — RSV-A and RSV-B F protein (574 aa)
- flu_antigens.csv — 41 influenza HA variants (565-566 aa)
- sars_antigens.csv — 24 SARS-CoV-2 RBD variants (222-223 aa)

Full sequence catalog: config/antigen_sequences.json

## Model Config Dates
| Antigen | Bind | Neu |
|---------|------|-----|
| flu | 250312 | 240905 |
| sars | 0611 | 1024 |
| rsv | 250224 | 250225 |

## Gotchas

1. Response is synchronous — do NOT poll streaming URLs
2. Requires PAIRED heavy/light chain sequences. Single-chain predictions are not supported.
3. Prediction accuracy varies by antigen type — flu models are most mature
4. Ensemble model recommended for best accuracy (default)
5. Input from IgBLAST AIRR output needs column renaming (v_call→VH sequence, not gene name)
6. Always pass the correct config_date for the antigen-task combination — using the wrong date will fail with a config_not_found error
