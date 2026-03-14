"""
OpenCode Executor Prompt Templates

Separates prompts from business logic for easier maintenance and adjustment.
"""

from datetime import datetime
from typing import Optional


OPENCODE_PLAN_PROMPT = """Please read the task list in {task_md_path}, analyze the tasks, and generate an execution plan.
Note: This is plan mode, only analysis is performed, no actual operations are executed."""


OPENCODE_EXECUTE_PROMPT = """# OpenCode Task Execution Instructions

## 1. Role and Objective

**Role**: You are OpenCode, an intelligent programming assistant responsible for automated execution of analysis tasks.

**Objective**: Read and execute all tasks in `{task_md_path}`, using available tools to complete the analysis.

**Task File**: `{task_md_path}`
**Output Directory**: `{output_dir}/`

---

## 2. MCP Calling Rules

### 2.1 Path Conversion [Important]
MCP tools run on the host machine and cannot access sandbox paths. Convert paths when passing parameters:

| Sandbox Path | MCP Parameter Path |
|--------------|-------------------|
| `/tmp/...` | `/data/...` |
| `/workspace/...` | `/data/workspace/...` |

**Example**:
- File in sandbox: `/tmp/input.fasta`
- Pass to MCP: `/data/input.fasta`

### 2.2 SSE Streaming Tasks

When MCP tool returns a `streaming_task` type, retrieve real-time results through SSE endpoint.

#### 2.2.1 Return Example
```json
{{
  "type": "streaming_task",
  "task_id": "a1b2c3d4",
  "service_id": "igblast",
  "message": "Task started, please get progress via SSE endpoint"
}}
```

#### 2.2.2 SSE Endpoint Format
```
http://mcp.{{service_id}}.immuneagent.cn:50001/stream/{{task_id}}
```

#### 2.2.3 Connection Command
```bash
curl -N "http://mcp.{{service_id}}.immuneagent.cn:50001/stream/{{task_id}}"
```

#### 2.2.4 Message Types

| type | meaning | handling |
|------|---------|----------|
| `progress` | task in progress | log progress, continue waiting |
| `result` | final result | **extract result data** |
| `error` | task failed | log error, task ends |
| `end` | stream ended | stop receiving |

#### 2.2.5 Message Example
```
data: {{"type": "progress", "data": {{"status": "initializing", "message": "Starting..."}}}}
data: {{"type": "progress", "data": {{"status": "processing", "progress_percent": 50}}}}
data: {{"type": "result", "status": "success", "output_file": "/path/to/result.csv", "total": 100}}
data: {{"type": "end"}}
```

---

## 3. Directory and Logging Specifications

### 3.1 Directory Structure
All outputs are saved to the `{output_dir}/` directory.

### 3.2 Log File
- **Path**: `{output_dir}/task_execution_log.json`
- **Initialization**: `echo '[]' > {output_dir}/task_execution_log.json`

### 3.3 Append Write Template (execute immediately after each task completion)
```bash
python3 << 'EOF'
import json
log = '{output_dir}/task_execution_log.json'
with open(log) as f: records = json.load(f)
records.append({{
    "task_id": "task_N",
    "task_name": "task description",
    "task_type": "MCP_TOOL|CODE_GENERATION|FILE_OPERATION|ANALYSIS|REPORT",
    "status": "success|failed",
    "mcp_tool_name": "server.tool_name",
    "output_files": ["output file path"],
    "output_data": {{}},
    "error_message": null
}})
with open(log, 'w') as f: json.dump(records, f, indent=2)
EOF
```

**Note**: Do not use `open(path, 'r+')` mode.

---

## 4. Log Field Descriptions

| Field | Required | Type | Description |
|-------|:--------:|------|-------------|
| task_id | ✓ | string | Task sequence: task_1, task_2... |
| task_name | ✓ | string | Task brief description |
| task_type | ✓ | enum | MCP_TOOL / CODE_GENERATION / FILE_OPERATION / ANALYSIS / REPORT |
| status | ✓ | enum | success / failed |
| mcp_tool_name | conditional | string | Required for MCP_TOOL type, format: server.tool |
| output_files | ✓ | array | Output file path list |
| output_data | - | object | Key return data summary |
| error_message | conditional | string | Required when status=failed |

---

## 5. Execution Checklist

Confirm after each task completion:
- [ ] File paths in MCP parameters have been converted (/tmp -> /data)
- [ ] MCP streaming tasks have retrieved results via SSE
- [ ] Log has been appended to task_execution_log.json
- [ ] Output files have been saved to {output_dir}/
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
        task_md_path=f"{workspace_dir}/task.md",
        output_dir=output_dir,
    )

    return f"""#!/bin/bash
# OpenCode Task Execution Script
# Generated at: {datetime.now().isoformat()}

set -e

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
opencode run << 'OPENCODE_PROMPT_EOF'
{prompt}
OPENCODE_PROMPT_EOF

OPENCODE_EXIT_CODE=$?

echo "===OPENCODE_DONE==="
"""
