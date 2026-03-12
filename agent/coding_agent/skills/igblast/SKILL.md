---
name: igblast
description: IgBLAST V(D)J gene alignment for antibody/BCR/TCR nucleotide sequences. Returns streaming_task (async). Use when identifying V/D/J genes, extracting CDR3 sequences, or producing AIRR-format output.
---

## Tools

| Tool | Description |
|------|-------------|
| `analyze_vdj_batch` | V(D)J alignment (main tool), outputs AIRR-format CSV |
| `extract_cdr3_from_airr` | Extract CDR3 sequences from AIRR CSV |

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

---

## Response Pattern: Async streaming_task

Calling `analyze_vdj_batch` returns immediately without waiting for analysis:
```json
{"type": "streaming_task", "task_id": "abc-123", "service_id": "igblast"}
```

**You MUST poll via bash for the real result. NEVER fabricate any fields.**

The streaming URL is provided in the current prompt under "streaming_task polling URLs" (derived from config by the executor). Format: `<igblast streaming base>/<task_id>`. Use that URL directly — do not hardcode or construct it yourself.

```bash
# Step 1: Read the real task_id from the tool response JSON
TASK_ID="<real uuid from tool response>"
STREAM_FILE="/tmp/stream_${TASK_ID}.txt"

# Step 2: Use the igblast streaming URL from the prompt (replace <task_id> with actual value)
# Example: curl -sN "http://<host>/mcp/8088/stream/${TASK_ID}" > "${STREAM_FILE}" 2>&1 &
echo "PID=$! file=${STREAM_FILE}"

# Step 3: Check every 60s (max 30 minutes)
grep -m1 '"type":"result"' "${STREAM_FILE}" 2>/dev/null || echo "waiting..."
```

When `type=result` appears, parse the JSON and read `output_file` and `session_id` fields.

Final result event format:
```json
{"type": "result", "status": "success",
 "output_file": "/data/server/.../airr_results_<id>.csv",
 "session_id": "abc12345", "total_sequences": 10, "format": "AIRR"}
```

Output CSV contains standard AIRR fields: `v_call`, `d_call`, `j_call`, `junction`, `junction_aa`, `productive`

---

## Constraints

1. **Input must be nucleotide sequences** (ATGC), not amino acid sequences
2. No `output_dir` parameter; the output path is in the `output_file` field
3. `output_file` is a server-side path; pass it directly to downstream tools like `extract_cdr3_from_airr`
