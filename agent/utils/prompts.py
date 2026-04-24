"""
OpenCode Executor Prompt Templates

Separates prompts from business logic for easier maintenance and adjustment.
"""

from datetime import datetime
from typing import Optional


OPENCODE_PLAN_PROMPT = """Please read the task list in {task_md_path}, analyze the tasks, and generate an execution plan.
Note: This is plan mode, only analysis is performed, no actual operations are executed."""


OPENCODE_EXECUTE_PROMPT = """# Task Executor

## Role
Execute all tasks defined in `task.md`.

## Instructions
1. Read and execute ALL tasks from `task.md`
2. Follow the rules in `AGENTS.md` for logging and file operations
3. Use MCP tools when available; fall back to code generation only when no suitable tool exists
4. After EACH task completion, append a log entry to `task_execution_log.json`

## Output Directory
Save all outputs to: `{output_dir}/`
"""


OPENCODE_AGENTS_CONTENT = """# Task Execution Rules

## MANDATORY: Task Execution Log (HIGHEST PRIORITY)

**You MUST write a log entry for EVERY task you execute.**
**Immediately after completing each task (success or failure), run the script below.**

- Log path: `{log_path}`
- The file is pre-initialized as `[]`. Append one JSON object per task.
- **Note**: Full OpenCode JSON output is automatically saved to `opencode.log` in the same directory.

### Log Entry Format

| Field | Required | Description |
|-------|:--------:|-------------|
| `task_id` | Yes | Sequential: `task_1`, `task_2`, ... |
| `task_name` | Yes | Brief description |
| `task_type` | Yes | `MCP_TOOL`, `CODE_GENERATION`, `FILE_OPERATION`, `ANALYSIS`, `REPORT` |
| `status` | Yes | `success` or `failed` |
| `mcp_tool_name` | No | Required if type is `MCP_TOOL` |
| `input_params` | YES | Input parameters |
| `output_files` | Yes | Output file paths |
| `output_data` | No | Key result data summary |
| `error_message` | No | Required if `status` is `failed` |

### Append Log Entry Script

```bash
python3 << 'EOF'
import json
from datetime import datetime
log_path = '{log_path}'
with open(log_path) as f:
    records = json.load(f)

records.append({{
    "task_id": "task_N",
    "task_name": "Brief description",
    "task_type": "MCP_TOOL | CODE_GENERATION | FILE_OPERATION | ANALYSIS | REPORT",
    "status": "success | failed",
    "mcp_tool_name": "server.tool_name",
    "output_files": ["path/to/output/file"],
    "output_data": {{}},
    "error_message": null,
    "timestamp": datetime.now().isoformat()
}})
with open(log_path, 'w') as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"Log written: {{len(records)}} records total")
EOF
```

---

## Input Data Directory (CRITICAL)

**User-provided data files are located in:**
- `/data/sessions/{session_id}/input/`

**When processing data:**
1. ALWAYS check for input files in `/data/sessions/{session_id}/input/` first
2. Use the provided data files from this directory (e.g., CSV, FASTA, JSON, etc.)
3. DO NOT generate mock/synthetic data unless explicitly requested
4. If the required data file is not found in `/data/sessions/{session_id}/input/`, report the issue instead of creating fake data

**Common input file paths:**
- `/data/sessions/{session_id}/input/data.csv`
- `/data/sessions/{session_id}/input/sequences.fasta`
- `/data/sessions/{session_id}/input/config.json`
- `/data/sessions/{session_id}/input/` (for all uploaded files)

---

## Security Constraint

- Only access files under `/data/sessions/{session_id}/`
- Never access system directories (`/etc/`, `/usr/`, `/root/`, `/home/`)
- Save all outputs to `{output_dir}/`

---

## Memory Optimization (CRITICAL)

**The sandbox memory limit is 16GB with 2-hour timeout. Scripts may be killed if they exceed this limit.**

**Actual available memory**: After system and runtime overhead, you have approximately **15GB** for data processing.

### Memory-Safe Practices

1. **Process Data in Chunks**
   ```python
   # ❌ BAD: Load entire file into memory
   df = pd.read_csv("large_file.csv")  # May OOM
   
   # ✅ GOOD: Read in chunks
   for chunk in pd.read_csv("large_file.csv", chunksize=10000):
       process(chunk)
   ```

2. **Avoid Loading Large Files Completely**
   - Use `chunksize` parameter in pandas
   - Use generators instead of lists
   - Process line-by-line when possible

3. **Memory-Efficient Visualization**
   ```python
   # ❌ BAD: Create huge plots with millions of points
   plt.plot(x_millions, y_millions)
   
   # ✅ GOOD: Sample or aggregate data first
   df_sample = df.sample(n=10000)  # Sample 10k points
   plt.plot(df_sample['x'], df_sample['y'])
   
   # Or aggregate
   df_agg = df.groupby('category').agg({{'value': 'mean'}})
   ```

4. **Clear Unused Variables**
   ```python
   import gc
   del large_dataframe
   gc.collect()
   ```

5. **Check File Size Before Loading**
   ```bash
   # Check file size first
   ls -lh large_file.csv
   # If > 100MB, use chunked reading
   ```

### Memory Warning Signs

If you see "killed due to memory constraints":
- The script used too much memory
- Switch to chunked processing
- Reduce data size (sampling, filtering)
- Avoid loading multiple large files simultaneously

**⚠️ ALWAYS prefer memory-efficient approaches over convenience.**

---

## File Operations Rules

### Rule 1: Column Preservation
- MUST preserve ALL original columns
- CAN add new columns
- CANNOT remove or drop existing columns

### Rule 2: Column Alignment for Merging
- Identical column names = SAME column (align data)
- Preserve columns unique to one file (fill null/empty for missing rows)

### Rule 3: Primary Key Requirement
- ALWAYS include `main_name` column as the primary key
- If source file lacks `main_name`, identify the PK column and alias it

---

## MCP Tool Integration

### MCP Tool Lookup (CRITICAL)

**NEVER use `skill_mcp` tool to look up or invoke MCP tools.**

### Streaming Task Flow (CRITICAL)

**IMPORTANT: When an MCP tool returns a `streaming_task` response, the task is NOT complete yet.** 

You MUST connect to the SSE endpoint and wait for the final `result`, `error`, or `end` message before proceeding to the next task.

**Step 1: Recognize Streaming Task Response**
```json
{{
  "type": "streaming_task",
  "task_id": "a1b2c3d4",
  "service_id": "igblast",
  "message": "Task started, please get progress via SSE endpoint"
}}
```

**Step 2: Connect to SSE Endpoint and Wait for Completion**
```bash
# Use curl to connect and monitor the SSE stream
curl -N "http://mcp.{{service_id}}.immuneagent.cn:50001/stream/{{task_id}}"
```

**Step 3: Monitor Until Task Completes**

The SSE stream will emit messages. Continue reading until you see a completion signal:

| Type | Meaning | Required Action |
|------|---------|-----------------|
| `progress` | Task in progress | **Continue monitoring** - do NOT proceed yet |
| `result` | Final result available | **Task complete** - extract data, proceed to next task |
| `error` | Task failed | **Task failed** - handle error, then proceed |
| `end` | Stream ended | **Task complete** - stop receiving |

**Example SSE Session:**
```
# Connection established, monitoring...
data: {{"type":"progress","percent":25,"message":"Processing..."}}
data: {{"type":"progress","percent":50,"message":"Still working..."}}
data: {{"type":"progress","percent":75,"message":"Almost done..."}}
data: {{"type":"result","data":{{...}}}}  # <- NOW task is complete
data: {{"type":"end"}}                     # <- Stream closing
```

**⚠️ CRITICAL RULE: Do NOT proceed to the next task until you receive `result`, `error`, or `end`.**

**Step 4: After Completion**
1. Extract result data from the `result` message
2. Save output files as needed
3. Append log entry to `task_execution_log.json`
4. NOW you can proceed to the next task

### Non-Streaming MCP Tools

For MCP tools that return immediate results (not `streaming_task`), the task is complete upon receiving the response. Proceed normally.

---

## Background Task Handling (CRITICAL)

**⚠️ `run_in_background` parameter is REQUIRED for ALL `task` tool calls.**

**Two usage modes:**
- **Task Delegation** → `run_in_background=false` (wait for completion)
- **Parallel Exploration** → `run_in_background=true` (async, must call `background_output`)

**IMPORTANT NOTE**: The term "wait for results" below means you MUST call the `background_output` TOOL. It does NOT mean you should pause execution or stop working. You MUST actively call the tool to retrieve results.

### Mode 1: Task Delegation (run_in_background=false)

Use this when you need result immediately.

```typescript
// Direct delegation - waits for completion
const result = task({{
  subagent_type: "explore",
  description: "Find patterns",
  prompt: "...",
  load_skills: [],
  run_in_background: false
}})
// → Returns: full task result directly
```

### Mode 2: Parallel Exploration (run_in_background=true)

Use this for parallel research. **MUST call `background_output` TOOL to retrieve results.**

```typescript
// Step 1: Launch background task
task({{
  subagent_type: "explore",
  description: "Find patterns",
  prompt: "...",
  load_skills: [],
  run_in_background: true
}})
// → Returns: "Background Task ID: bg_xxx\nStatus: pending..."

// Step 2: IMMEDIATELY call background_output tool to wait for results
background_output({{
  task_id: "bg_xxx",
  block: true,
  timeout: 60000
}})
```

### ❌ Wrong Patterns

1. **Missing run_in_background parameter (CRITICAL ERROR):**
    ```typescript
    task({{
      subagent_type: "explore",
      prompt: "..."
    }})
    // ❌ WRONG - Missing REQUIRED run_in_background parameter
    // Error: "Invalid arguments: 'run_in_background' parameter is REQUIRED"
    ```

2. **Only mentioning "waiting" without calling background_output:**
    ```typescript
    // Launch background task
    task({{ subagent_type: "explore", ..., run_in_background: true }})
    // Then saying "I'll wait for results" WITHOUT calling background_output
    // ❌ WRONG - no tool call, you MUST call background_output tool
    ```

3. **Starting multiple tasks without polling:**
    ```typescript
    task({{ subagent_type: "explore", ..., run_in_background: true }})  // bg_1
    task({{ subagent_type: "librarian", ..., run_in_background: true }})  // bg_2
    // ❌ WRONG - must call background_output for each
    ```

### Correct Multi-Task Pattern (Parallel)

```typescript
// Launch multiple background tasks in parallel
const r1 = task({{ subagent_type: "explore", ..., run_in_background: true }})
const r2 = task({{ subagent_type: "librarian", ..., run_in_background: true }})

// Poll each one to get results using background_output tool
background_output({{ task_id: "bg_1", block: true }})
background_output({{ task_id: "bg_2", block: true }})
```

**⚠️ CRITICAL: After launching tasks with run_in_background=true, you MUST call the background_output TOOL. Do not just mention waiting - actively call the tool.**

---

## Task Completion Checklist

After completing each task, verify:
- [ ] **Log entry appended to `task_execution_log.json` (MANDATORY)**
- [ ] **MCP streaming tasks: waited for `result`/`error`/`end` message before proceeding**
- [ ] Output files saved to the correct output directory
- [ ] File operations comply with rules (no column loss, main_name preserved)

**Note**: All OpenCode JSON output is automatically captured in `opencode.log` - no manual event collection needed.

**⚠️ NEVER skip to the next task while an MCP streaming task is still in progress.**
"""


_RUNNER_ENV_BLOCK = """\
export PYTHONUNBUFFERED=1
export NODE_OPTIONS="--no-warnings --no-deprecation"
export TERM=dumb
export NO_COLOR=1

if [ -f "/opt/opensandbox/code-interpreter-env.sh" ]; then
    echo "Activating OpenSandbox virtual environment..."
    source /opt/opensandbox/code-interpreter-env.sh python 3.13 2>/dev/null || true
fi"""


_RUNNER_SETUP_BLOCK = """\
# Create OpenCode XDG directories
mkdir -p "$WORK_DIR/opencode/data"
mkdir -p "$WORK_DIR/opencode/config/opencode"
mkdir -p "$WORK_DIR/opencode/state"
mkdir -p "$WORK_DIR/opencode/bin"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
echo '[]' > "$LOG_DIR/task_execution_log.json"

# Initialize opencode.log (clear if exists)
: > "$LOG_DIR/opencode.log"

# Create AGENTS.md in WORK_DIR root (opencode reads from working directory)
cat > "$WORK_DIR/AGENTS.md" << 'AGENTS_EOF'
{agents_content}
AGENTS_EOF

echo "=== Starting OpenCode Task Execution ==="
echo "Working directory: $WORK_DIR"
echo "Session ID: $SESSION_ID"
echo "Task file: $WORK_DIR/task.md"
echo "Timestamp: $(date -Iseconds)"
"""


_RUNNER_EXEC_BLOCK = """\
# Execute OpenCode and capture all JSON output (append mode to prevent overwrites)
echo "=== Starting OpenCode (output logged to $LOG_DIR/opencode.log) ==="

# Execute with explicit instructions to read AGENTS.md first
# Use --no-interactive to prevent OpenCode from waiting for user input
# Use --continue-on-error to keep running even if one task fails
opencode run "Execute all tasks in task.md." --file "$WORK_DIR/task.md" --format json 2>&1 | tee -a "$LOG_DIR/opencode.log"

OPENCODE_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "===OPENCODE_DONE==="
echo "OpenCode exit code: $OPENCODE_EXIT_CODE"
echo "OpenCode log: $LOG_DIR/opencode.log"
echo "Timestamp: $(date -Iseconds)"
"""


def get_opencode_runner_prompt(session_id: str, bundle_id: Optional[str] = None) -> str:
    """
    Generate the OpenCode runner shell script.

    Prerequisites:
    - task.md must be written by sandbox before calling this function

    Args:
        session_id: Session ID for path construction
        bundle_id: Optional bundle ID for workspace isolation

    Returns:
        Complete runner shell script as a string
    """
    if bundle_id:
        workspace_dir = f"/data/sessions/{session_id}/{bundle_id}"
        output_dir = f"/data/sessions/{session_id}/output/{bundle_id}"
        log_dir = f"/data/sessions/{session_id}/output/{bundle_id}"
    else:
        workspace_dir = f"/data/sessions/{session_id}"
        output_dir = f"/data/sessions/{session_id}/output"
        log_dir = f"/data/sessions/{session_id}/output"

    # AGENTS.md content
    agents_content = OPENCODE_AGENTS_CONTENT.format(
        log_path=f"{log_dir}/task_execution_log.json",
        output_dir=output_dir,
        session_id=session_id,
    )

    # Main prompt (simplified, AGENTS.md will be auto-loaded by OpenCode)
    prompt = OPENCODE_EXECUTE_PROMPT.format(output_dir=output_dir)

    return f"""#!/bin/bash
# OpenCode Task Execution Script
# Generated at: {datetime.now().isoformat()}

{_RUNNER_ENV_BLOCK}

cd {workspace_dir}

WORK_DIR="{workspace_dir}"
OUTPUT_DIR="{output_dir}"
LOG_DIR="{log_dir}"
SESSION_ID="{session_id}"

{_RUNNER_SETUP_BLOCK.format(agents_content=agents_content)}

which opencode || echo "opencode not found in PATH"

{_RUNNER_EXEC_BLOCK}
"""
