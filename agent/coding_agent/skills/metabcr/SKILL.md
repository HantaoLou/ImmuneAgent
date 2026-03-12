---
name: metabcr
description: MetaBCR antibody-antigen binding and neutralization prediction. Returns synchronous dict (no streaming). Use when predicting BCR/antibody functional activity against flu, SARS, RSV, or HIV antigens.
---

## MetaBCR Antibody-Antigen Functional Prediction

**Response mode**: Synchronous return (not streaming_task)

MetaBCR predicts antibody-antigen binding affinity (bind) and neutralization activity (neu). Supports 4 viruses: flu, sars, rsv, hiv.

---

## Tool Parameters

Tool name: `metabcr` (flat parameter passing)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `antibody_file` | str | **required** | Antibody CSV file path (must contain sequence columns) |
| `antigen_file` | str | **required** | Antigen CSV file path |
| `antigen_name` | str | `"flu"` | Antigen type: `flu` / `sars` / `rsv` / `hiv` |
| `task_name` | str | `"bind"` | Prediction task: `bind` (binding) or `neu` (neutralization) |
| `config_date` | str | `"250312"` | Model config date stamp |
| `output_file_path` | str | `null` | Output directory path (optional) |

---

## Return Value

Tools yield progress updates then a final result dict:
```json
{"type": "result", "status": "success", "output_file": "...", "session_id": "...", "total_antibodies": 50}
```

---

## Constraints

1. `antigen_name` only supports `flu` / `sars` / `rsv` / `hiv`
2. `antibody_file` must contain antibody sequence columns (Heavy and Light chains)
3. Each antibody row is combined with each antigen sequence for prediction
4. Not all antigens support `neu` task — verify compatibility
