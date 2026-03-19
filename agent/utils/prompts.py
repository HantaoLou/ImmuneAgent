"""
OpenCode Executor Prompt Templates

Separates prompts from business logic for easier maintenance and adjustment.
"""

from datetime import datetime
from typing import Optional


OPENCODE_PLAN_PROMPT = """Please read the task list in {task_md_path}, analyze the tasks, and generate an execution plan.
Note: This is plan mode, only analysis is performed, no actual operations are executed."""


OPENCODE_EXECUTE_PROMPT = """
# OpenCode Task Execution System

## 📁 Workspace Structure

### Input Files
- Location: `/data/sessions/{session_id}/input/`
- All required input files are located in this directory

### Task Definition
- Location: `/data/sessions/{session_id}/{bundle_id}/task.md`
- Contains the task list and requirements

### Output Directory
- Location: `/data/sessions/{session_id}/output/`
- ALL task outputs MUST be saved to this directory

---

## 🎯 Core Execution Principles

1. **File Discovery**: All files you need can be found within `/data/sessions/{session_id}/`
2. **Output Consistency**: All task outputs must be saved to `/data/sessions/{session_id}/output/`
3. **Tool Priority**: Before executing any task, ALWAYS check if there's a suitable skill or MCP tool available
4. **Error Recovery**: When a task fails, try alternative approaches:
   - Switch to a different skill
   - Use an alternative MCP tool
   - Revise the code implementation
   - Break down into smaller subtasks

---

## 🔌 MCP Tool Integration

### SSE Streaming Tasks

When an MCP tool returns a `streaming_task` response, retrieve real-time results via SSE endpoint.

#### Response Format
```json
{{
  "type": "streaming_task",
  "task_id": "a1b2c3d4",
  "service_id": "igblast",
  "message": "Task started, please get progress via SSE endpoint"
}}
```

#### SSE Connection
```bash
# Endpoint: http://mcp.{{service_id}}.immuneagent.cn:50001/stream/{{task_id}}
curl -N "http://mcp.{{service_id}}.immuneagent.cn:50001/stream/{{task_id}}"
```

#### Message Types

| Type | Meaning | Action |
|------|---------|--------|
| `progress` | Task in progress | Log progress, continue monitoring |
| `result` | Final result available | **Extract and save result data** |
| `error` | Task failed | Log error, handle failure |
| `end` | Stream ended | Stop receiving |

#### Example Stream
```
data: {{"type": "progress", "data": {{"status": "initializing", "message": "Starting..."}}}}
data: {{"type": "progress", "data": {{"status": "processing", "progress_percent": 50}}}}
data: {{"type": "result", "status": "success", "output_file": "/path/to/result.csv", "total": 100}}
data: {{"type": "end"}}
```

---

## 📊 File Operations - CRITICAL RULES

### Rule 1: Column Preservation
When converting column names or transforming data:
- ✅ **MUST** preserve ALL original columns
- ✅ Can ADD new columns
- ❌ CANNOT remove or drop existing columns

**Correct Example:**
```python
# Input:  ['name', 'age', 'city']
# Output: ['name', 'age', 'city', 'name_standardized']  # Added column, kept all originals
```

**Incorrect Example:**
```python
# Input:  ['name', 'age', 'city']
# Output: ['name', 'age']  # ❌ Lost 'city' column
```

### Rule 2: Column Alignment for Merging
When merging multiple files of the same type:
- Columns with identical names = SAME column (align data)
- Preserve columns that exist in only one file (fill with null/empty for missing rows)

**Correct Example:**
```python
# File1: ['id', 'name', 'age']
# File2: ['id', 'name', 'city']
# Merged: ['id', 'name', 'age', 'city']  # ✅ All columns preserved, 'id' and 'name' aligned
```

### Rule 3: Primary Key Requirement
When extracting columns from a file:
- **ALWAYS** include `main_name` column as the primary key
- If source file lacks `main_name`, identify the primary key column and create/use it as `main_name`
- The `main_name` column is essential for merging results from multiple operations

**Correct Example:**
```python
# Task: Extract ['peptide', 'score'] from input.csv
# Input: ['main_name', 'peptide', 'score', 'other_col']
# Output: ['main_name', 'peptide', 'score']  # ✅ main_name included
```

### Pre-Completion Validation Checklist
Before finalizing any file operation:
- [ ] All original columns present in output
- [ ] No data rows lost during transformation
- [ ] Column names correctly aligned when merging
- [ ] Missing values handled appropriately (null/empty)
- [ ] `main_name` column present (or created if source lacked it)

---

## 📝 Logging Requirements

### Log File Location
- **Path**: `/data/sessions/{session_id}/output/{bundle_id}/task_execution_log.json`
- **Initialize**: `echo '[]' > /data/sessions/{session_id}/output/{bundle_id}/task_execution_log.json`

### Log Entry Template
Execute this immediately after each task completion:
```bash
python3 << 'EOF'
import json
log_path = '/data/sessions/{session_id}/output/{bundle_id}/task_execution_log.json'
with open(log_path) as f:
    records = json.load(f)
records.append({{
    "task_id": "task_N",
    "task_name": "Brief task description",
    "task_type": "MCP_TOOL | CODE_GENERATION | FILE_OPERATION | ANALYSIS | REPORT",
    "status": "success | failed",
    "mcp_tool_name": "server.tool_name",  # Required if task_type is MCP_TOOL
    "output_files": ["path/to/output/file"],
    "output_data": {{}},  # Key return data summary
    "error_message": null  # Required if status is failed
}})
with open(log_path, 'w') as f:
    json.dump(records, f, indent=2)
EOF
```

**Note**: Do NOT use `open(path, 'r+')` mode to avoid file corruption.

### Log Field Specifications

| Field | Required | Type | Description |
|-------|:--------:|------|-------------|
| `task_id` | ✅ | string | Sequential ID: `task_1`, `task_2`, ... |
| `task_name` | ✅ | string | Brief description of the task |
| `task_type` | ✅ | enum | `MCP_TOOL`, `CODE_GENERATION`, `FILE_OPERATION`, `ANALYSIS`, `REPORT` |
| `status` | ✅ | enum | `success` or `failed` |
| `mcp_tool_name` | ⚠️ | string | Required if `task_type` is `MCP_TOOL`. Format: `server.tool` |
| `output_files` | ✅ | array | List of output file paths |
| `output_data` | ⭕ | object | Summary of key return data |
| `error_message` | ⚠️ | string | Required if `status` is `failed` |

---

## ✅ Task Completion Checklist

After completing each task, verify:
- [ ] MCP streaming tasks have retrieved results via SSE
- [ ] Log entry appended to `task_execution_log.json`
- [ ] Output files saved to the correct output directory
- [ ] File operations comply with critical rules (no column loss, main_name preserved)
"""


def get_opencode_runner_prompt(session_id: str, bundle_id: Optional[str] = None) -> str:
    """
    Get OpenCode execution prompt

    Args:
        session_id: Session ID for path construction
        bundle_id: Optional bundle ID for workspace isolation

    Returns:
        Formatted prompt string
    """
    if bundle_id:
        workspace_dir = f"/data/sessions/{session_id}/{bundle_id}"
    else:
        workspace_dir = f"/data/sessions/{session_id}"

    output_dir = f"/data/sessions/{session_id}/output"

    prompt = OPENCODE_EXECUTE_PROMPT.format(
        bundle_id=bundle_id,
        session_id=session_id,
    )

    return f"""#!/bin/bash
# OpenCode Task Execution Script
# Generated at: {datetime.now().isoformat()}

# Force unbuffered output for real-time SSE streaming
export PYTHONUNBUFFERED=1
export NODE_OPTIONS="--no-warnings --no-deprecation"
export TERM=dumb
export NO_COLOR=1

if [ -f "/opt/opensandbox/code-interpreter-env.sh" ]; then
    echo "Activating OpenSandbox virtual environment..."
    source /opt/opensandbox/code-interpreter-env.sh python 3.13 2>/dev/null || true
fi

cd {workspace_dir}

# Create OpenCode XDG directories
mkdir -p {workspace_dir}/opencode/data
mkdir -p {workspace_dir}/opencode/config/opencode
mkdir -p {workspace_dir}/opencode/state
mkdir -p {workspace_dir}/opencode/bin

# Create output directory (shared across bundles)
mkdir -p {output_dir}

echo "=== Starting OpenCode Task Execution ==="
echo "Working directory: {workspace_dir}"
echo "Session ID: {session_id}"
echo "Task file: {workspace_dir}/task.md"
echo "Timestamp: $(date -Iseconds)"

# Debug: check if opencode is available
echo "Checking opencode installation..."
which opencode || echo "opencode not found in PATH"

# Run OpenCode with JSON format for structured streaming output
# --print-logs: output logs to stderr for visibility
# --format json: structured JSON events for easier parsing
# --thinking: show thinking blocks for transparency
# Use stdbuf for unbuffered output if available
if command -v stdbuf &> /dev/null; then
    echo "Running OpenCode with JSON format (stdbuf)..."
    stdbuf -oL -eL opencode run --print-logs --format json --thinking 2>&1 << 'OPENCODE_PROMPT_EOF'
{prompt}
OPENCODE_PROMPT_EOF
else
    echo "Running OpenCode with JSON format..."
    opencode run --print-logs --format json --thinking 2>&1 << 'OPENCODE_PROMPT_EOF'
{prompt}
OPENCODE_PROMPT_EOF
fi

OPENCODE_EXIT_CODE=$?

echo ""
echo "===OPENCODE_DONE==="
echo "OpenCode exit code: $OPENCODE_EXIT_CODE"
echo "Timestamp: $(date -Iseconds)"
"""
