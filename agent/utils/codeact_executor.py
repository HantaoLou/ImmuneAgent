"""
CodeAct Executor - Unified sandbox code execution interface

This is the sole entry point for other subgraphs to communicate with OpenSandbox.
All sandbox operations should be executed through this interface, not by directly
calling opensandbox_executor.

Architecture principles:
- Other subgraphs (executor, result_evaluator, immunity, supervisor) do not call OpenSandbox directly
- All code execution requests are sent to the CodeAct subgraph
- CodeAct is responsible for: generating code -> executing code -> returning results
- OpenSandbox is a remote sandbox; only the CodeAct subgraph interacts with it directly

Usage examples:
    from utils.codeact_executor import execute_code_via_codeact

    # Scenario 1: CSV to FASTA conversion
    result = execute_code_via_codeact(
        task_description="Convert /data/sessions/xxx/input/data.csv to FASTA format",
        sandbox_id=existing_sandbox_id,
        keep_alive=True
    )

    # Scenario 2: Read remote file
    result = execute_code_via_codeact(
        task_description="Read all .csv files under /data/sessions/xxx/output directory",
        sandbox_id=opensandbox_id
    )
"""

from __future__ import annotations

import os
import json
import asyncio
import time
from typing import Any, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import sys

# Get agent directory
AGENT_DIR = Path(__file__).parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


class CodeActExecutionStatus(str, Enum):
    """Execution status"""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PENDING = "pending"


@dataclass
class CodeActResult:
    """CodeAct execution result"""

    status: CodeActExecutionStatus
    output: str = ""
    error: str = ""
    sandbox_id: Optional[str] = None
    returncode: int = 0
    parsed_result: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0

    def is_success(self) -> bool:
        return self.status == CodeActExecutionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "sandbox_id": self.sandbox_id,
            "returncode": self.returncode,
            "parsed_result": self.parsed_result,
            "execution_time_ms": self.execution_time_ms,
        }


def is_codeact_available() -> bool:
    """Check whether CodeAct is available"""
    try:
        from utils.opensandbox_executor import is_opensandbox_enabled

        return is_opensandbox_enabled()
    except ImportError:
        return False


def execute_code_via_codeact(
    task_description: str,
    code_template: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    timeout_seconds: int = 120,
    keep_alive: bool = True,
    env: Optional[Dict[str, str]] = None,
    auto_parse_json: bool = True,
) -> CodeActResult:
    """
        Execute code in sandbox via CodeAct (synchronous interface)

        This is the sole entry point for other subgraphs to communicate with OpenSandbox.

        Args:
            task_description: Task description (natural language describing the task to execute)
            code_template: Optional code template (if provided, execute this code directly without LLM generation)
            sandbox_id: Existing sandbox ID (reuse sandbox)
            timeout_seconds: Timeout in seconds
            keep_alive: Whether to keep the sandbox alive
            env: Environment variables
            auto_parse_json: Whether to automatically parse JSON in output

        Returns:
            CodeActResult: Object containing execution results

        Example:
            # Simple task - let CodeAct generate code
            result = execute_code_via_codeact(
                task_description="Read first 5 lines of /data/sessions/xxx/output/result.csv"
            )

            # Specified code - execute directly
            result = execute_code_via_codeact(
                task_description="Execute file copy",
                code_template=\'\'\'
    import shutil
    shutil.copy("/data/source.csv", "/data/target.csv")
    print("Done")
    \'\'\'
            )
    """
    start_time = time.time()

    try:
        # If code template is provided, execute directly
        if code_template:
            return _execute_code_direct(
                code=code_template,
                task_description=task_description,
                sandbox_id=sandbox_id,
                timeout_seconds=timeout_seconds,
                keep_alive=keep_alive,
                env=env,
                auto_parse_json=auto_parse_json,
            )

        # Otherwise, generate code via CodeAct subgraph and execute
        return _execute_via_codeact_subgraph(
            task_description=task_description,
            sandbox_id=sandbox_id,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
            auto_parse_json=auto_parse_json,
        )

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        return CodeActResult(
            status=CodeActExecutionStatus.ERROR,
            error=f"CodeAct execution failed: {str(e)}",
            execution_time_ms=execution_time,
        )


async def execute_code_via_codeact_async(
    task_description: str,
    code_template: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    timeout_seconds: int = 120,
    keep_alive: bool = True,
    env: Optional[Dict[str, str]] = None,
    auto_parse_json: bool = True,
) -> CodeActResult:
    """
    Async version of execute_code_via_codeact
    """
    # Run synchronous version in event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: execute_code_via_codeact(
            task_description=task_description,
            code_template=code_template,
            sandbox_id=sandbox_id,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
            env=env,
            auto_parse_json=auto_parse_json,
        ),
    )


def _execute_code_direct(
    code: str,
    task_description: str,
    sandbox_id: Optional[str],
    timeout_seconds: int,
    keep_alive: bool,
    env: Optional[Dict[str, str]],
    auto_parse_json: bool,
) -> CodeActResult:
    """
    Execute code directly (without LLM generation)

    Use this method when the caller already knows what code to execute.
    """
    start_time = time.time()

    try:
        from utils.opensandbox_executor import (
            run_code_in_opensandbox_sync,
            is_opensandbox_enabled,
        )

        if not is_opensandbox_enabled():
            return CodeActResult(
                status=CodeActExecutionStatus.ERROR,
                error="OpenSandbox not enabled. Set CODEACT_SANDBOX_PROVIDER=opensandbox or OPENSANDBOX_ENABLED=true",
            )

        # Prepare environment variables
        exec_env = env or {}
        if "OPENSANDBOX_SKIP_MCP_INSTALL" not in exec_env:
            exec_env["OPENSANDBOX_SKIP_MCP_INSTALL"] = "true"

        # Execute code
        # Note: sandbox ID is no longer reused; a new sandbox is created each time
        # Files are organized by session_id in the same directory, not by sandbox ID
        result = run_code_in_opensandbox_sync(
            code=code,
            task_id=f"codeact_direct_{int(time.time())}",
            timeout_seconds=timeout_seconds,
            existing_sandbox_id=None,  # Do not reuse; create new sandbox each time
            keep_alive=False,  # Do not keep sandbox alive
            env=exec_env,
        )

        execution_time = int((time.time() - start_time) * 1000)

        if result is None:
            return CodeActResult(
                status=CodeActExecutionStatus.ERROR,
                error="Sandbox returned empty result",
                execution_time_ms=execution_time,
            )

        # Parse result
        stdout = result.get("stdout", "") + result.get("formatted_output", "")
        stderr = result.get("stderr", "")
        error_msg = result.get("error", "")
        returncode = result.get("returncode", 0)
        result_sandbox_id = result.get("sandbox_id", sandbox_id)

        # Determine status
        # Note: returncode of None should not be treated as an error (OpenSandbox SDK may not return returncode)
        # Only treat it as an execution error when returncode is explicitly non-None and non-0
        if error_msg or (returncode is not None and returncode != 0):
            status = CodeActExecutionStatus.ERROR
        else:
            status = CodeActExecutionStatus.SUCCESS

        # Auto-parse JSON
        parsed_result = None
        if auto_parse_json and stdout:
            parsed_result = _extract_json_from_output(stdout)

        return CodeActResult(
            status=status,
            output=stdout,
            error=stderr or error_msg,
            sandbox_id=result_sandbox_id,
            returncode=returncode,
            parsed_result=parsed_result,
            execution_time_ms=execution_time,
        )

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        return CodeActResult(
            status=CodeActExecutionStatus.ERROR,
            error=f"Direct execution failed: {str(e)}",
            execution_time_ms=execution_time,
        )


def _execute_via_codeact_subgraph(
    task_description: str,
    sandbox_id: Optional[str],
    timeout_seconds: int,
    keep_alive: bool,
    auto_parse_json: bool,
) -> CodeActResult:
    """
    Execute task via CodeAct subgraph (LLM generates code)

    Use this method when the task requires LLM understanding and code generation.
    """
    start_time = time.time()

    try:
        # Import CodeAct subgraph components
        from nodes.subagents.code_act.graph import (
            build_codeact_subgraph,
            CodeActState,
            CodeActExecutionMode,
        )
        from state import SubTask

        # Create temporary task
        temp_task = SubTask(
            id=f"codeact_exec_{int(time.time())}",
            description=task_description,
            service_id="codeact",
            tool_name="execute_code",
            status="pending",
            result={},
        )

        # Build CodeAct state
        codeact_state = CodeActState(
            task=temp_task,
            task_description=task_description,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.CODEACT,
        )

        # If there is an existing sandbox_id, inject it into state
        if sandbox_id:
            codeact_state.existing_sandbox_id = sandbox_id

        # Build and execute subgraph
        subgraph = build_codeact_subgraph()
        result_state = subgraph.invoke(codeact_state.model_dump())

        execution_time = int((time.time() - start_time) * 1000)

        # Parse result
        if isinstance(result_state, dict):
            exec_result = result_state.get("execution_result", {})
            generated_code = result_state.get("generated_code", "")
            result_sandbox_id = (
                result_state.get("sandbox_id")
                or exec_result.get("sandbox_id")
                or sandbox_id
            )
        else:
            exec_result = getattr(result_state, "execution_result", {}) or {}
            generated_code = getattr(result_state, "generated_code", "")
            result_sandbox_id = getattr(result_state, "sandbox_id", None) or sandbox_id

        # Extract output
        output = exec_result.get("output", "") or exec_result.get("stdout", "")
        error = exec_result.get("error", "") or exec_result.get("stderr", "")
        status_str = exec_result.get("status", "unknown")

        # Determine status
        if status_str == "success":
            status = CodeActExecutionStatus.SUCCESS
        elif status_str == "timeout":
            status = CodeActExecutionStatus.TIMEOUT
        else:
            status = (
                CodeActExecutionStatus.ERROR
                if error
                else CodeActExecutionStatus.SUCCESS
            )

        # Auto-parse JSON
        parsed_result = None
        if auto_parse_json and output:
            parsed_result = _extract_json_from_output(output)

        return CodeActResult(
            status=status,
            output=output,
            error=error,
            sandbox_id=result_sandbox_id,
            parsed_result=parsed_result,
            execution_time_ms=execution_time,
        )

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        # If subgraph call fails, fall back to direct execution
        print(
            f"[CodeActExecutor] Subgraph execution failed, falling back to direct: {e}"
        )
        return CodeActResult(
            status=CodeActExecutionStatus.ERROR,
            error=f"CodeAct subgraph failed: {str(e)}",
            execution_time_ms=execution_time,
        )


def _extract_json_from_output(output: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON result from output

    Supports multiple formats:
    1. Marker format: __JSON_START__ ... __JSON_END__
    2. Direct JSON: {...}
    3. JSON array: [...]
    """
    import re

    # Try to extract marker-formatted JSON
    marker_patterns = [
        r"__JSON_START__\s*(.*?)\s*__JSON_END__",
        r"__OUTPUT_FILES_JSON_START__\s*(.*?)\s*__OUTPUT_FILES_JSON_END__",
        r"__RESULT_JSON__\s*(.*?)\s*__END_RESULT_JSON__",
    ]

    for pattern in marker_patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # Try to extract JSON object directly
    json_object_pattern = r"\{[^{}]*\}"
    matches = re.findall(json_object_pattern, output, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, dict) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            continue

    # Try to extract JSON array
    json_array_pattern = r"\[[^\]]*\]"
    matches = re.findall(json_array_pattern, output, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, list) and len(result) > 0:
                return {"items": result}
        except json.JSONDecodeError:
            continue

    return None


# ==================== Convenience Functions ====================


def read_remote_file(
    file_path: str, sandbox_id: Optional[str] = None, max_lines: int = 1000
) -> CodeActResult:
    """
    Read a file in the remote sandbox

    Args:
        file_path: Remote file path
        sandbox_id: Sandbox ID
        max_lines: Maximum number of lines to read

    Returns:
        CodeActResult, output contains file content
    """
    code = f'''
import os

file_path = "{file_path}"
max_lines = {max_lines}

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = []
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line.rstrip('\\n'))
        print(f"__FILE_LINES__:{{len(lines)}}")
        for line in lines:
            print(line)
else:
    print(f"__FILE_NOT_FOUND__:{{file_path}}")
'''

    return execute_code_via_codeact(
        task_description=f"Read remote file {file_path}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True,
    )


def list_remote_directory(
    dir_path: str, sandbox_id: Optional[str] = None, pattern: str = "*"
) -> CodeActResult:
    """
    List files in a remote sandbox directory

    Args:
        dir_path: Remote directory path
        sandbox_id: Sandbox ID
        pattern: File matching pattern (glob format)

    Returns:
        CodeActResult, parsed_result contains file list
    """
    code = f'''
import os
import json
from pathlib import Path

dir_path = "{dir_path}"
pattern = "{pattern}"

results = []
path = Path(dir_path)

if path.exists() and path.is_dir():
    for file_path in path.glob(pattern):
        if file_path.is_file():
            results.append({{
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "ext": file_path.suffix
            }})

print("__DIR_LIST_START__")
print(json.dumps(results, ensure_ascii=False))
print("__DIR_LIST_END__")
'''

    result = execute_code_via_codeact(
        task_description=f"List remote directory {dir_path}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True,
    )

    return result


def copy_file_in_sandbox(
    source_path: str, target_path: str, sandbox_id: Optional[str] = None
) -> CodeActResult:
    """
    Copy a file within the sandbox

    Args:
        source_path: Source file path
        target_path: Target file path
        sandbox_id: Sandbox ID

    Returns:
        CodeActResult
    """
    code = f'''
import os
import shutil

source = "{source_path}"
target = "{target_path}"

try:
    if os.path.exists(source):
        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(source, target)
        print(f"__FILE_COPIED__:{{target}}")
    else:
        print(f"__FILE_NOT_FOUND__:{{source}}")
except Exception as e:
    print(f"__COPY_ERROR__:{{str(e)}}")
'''

    return execute_code_via_codeact(
        task_description=f"Copy file {source_path} to {target_path}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True,
    )


def convert_csv_to_fasta(
    csv_path: str,
    output_path: str,
    sequence_columns: Optional[List[str]] = None,
    sandbox_id: Optional[str] = None,
) -> CodeActResult:
    """
    Convert CSV file to FASTA format

    Args:
        csv_path: CSV file path
        output_path: Output FASTA file path
        sequence_columns: List of sequence column names (auto-detected if None)
        sandbox_id: Sandbox ID

    Returns:
        CodeActResult
    """
    seq_cols_str = str(sequence_columns) if sequence_columns else "None"

    code = f'''
import os
import csv

csv_path = "{csv_path}"
output_path = "{output_path}"
sequence_columns = {seq_cols_str}

    # Supported sequence column names
SEQ_COLUMN_PATTERNS = [
    'sequence', 'seq', 'cdr3', 'CDR3',
    'heavy_dna', 'light_dna', 'Heavy_DNA', 'Light_DNA',
    'vdj_sequence', 'nucleotide_sequence'
]

def find_sequence_columns(headers):
    """Auto-detect sequence columns"""
    found = []
    for header in headers:
        header_lower = header.lower()
        for pattern in SEQ_COLUMN_PATTERNS:
            if pattern.lower() in header_lower:
                found.append(header)
                break
    return found

try:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        if sequence_columns is None:
            sequence_columns = find_sequence_columns(headers)

        if not sequence_columns:
            print("__CSV_NO_SEQ_COLUMNS__")
        else:
            # Conversion path (inside container) - use unified /data/sessions path
            container_output = output_path
            os.makedirs(os.path.dirname(container_output), exist_ok=True)

            with open(container_output, 'w') as out_f:
                count = 0
                for row_idx, row in enumerate(reader):
                    for seq_col in sequence_columns:
                        seq = row.get(seq_col, '').strip()
                        if seq and len(seq) > 0:
                            header_id = row.get('id', row.get('ID', f'seq_{{row_idx}}'))
                            out_f.write(f"> {{header_id}}_{{seq_col}}\\n")
                            out_f.write(f"{{seq}}\\n")
                            count += 1

                print(f"__CSV_TO_FASTA_SUCCESS__:{{container_output}}:{{count}}")
except FileNotFoundError:
    print(f"__CSV_NOT_FOUND__:{{csv_path}}")
except Exception as e:
    print(f"__CSV_TO_FASTA_ERROR__:{{str(e)}}")
'''

    return execute_code_via_codeact(
        task_description=f"Convert CSV {csv_path} to FASTA format",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True,
    )


# ==================== More Convenience Functions ====================


def convert_rds_to_csv(
    rds_path: str,
    output_csv_path: Optional[str] = None,
    sandbox_id: Optional[str] = None,
) -> CodeActResult:
    """
    Convert RDS file to CSV format

    Args:
        rds_path: RDS file path
        output_csv_path: Output CSV file path (optional, defaults to same name)
        sandbox_id: Sandbox ID

    Returns:
        CodeActResult
    """
    code = f'''
import pyreadr
import pandas as pd
import os

rds_path = "{rds_path}"
csv_path = "{output_csv_path or rds_path.replace(".rds", ".csv").replace(".RDS", ".csv")}"

# Ensure output directory exists
output_dir = os.path.dirname(csv_path)
os.makedirs(output_dir, exist_ok=True)
try:
    os.chmod(output_dir, 0o777)
except Exception:
    pass

try:
    # Read RDS file
    result = pyreadr.read_r(rds_path)
    
    if not result:
        print("[ERROR] Failed to read RDS file or file is empty", flush=True)
        print("__RDS_CSV_FAILED__", flush=True)
    else:
        # Get the first data.frame or compatible object
        df = None
        for key, data in result.items():
            if isinstance(data, pd.DataFrame):
                df = data
                print(f"Found DataFrame: {{key}} with {{len(df)}} rows", flush=True)
                break
            elif isinstance(data, dict):
                # Try to convert dict to DataFrame
                try:
                    df = pd.DataFrame(data)
                    print(f"Converted dict to DataFrame: {{key}} with {{len(df)}} rows", flush=True)
                    break
                except Exception as e:
                    print(f"Could not convert {{key}} to DataFrame: {{e}}", flush=True)
                    continue
        
        if df is None:
            print("[ERROR] No compatible data object found in RDS file", flush=True)
            print("__RDS_CSV_FAILED__", flush=True)
        else:
            # Save to CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"[OK] Generated CSV with {{len(df)}} rows: {{csv_path}}", flush=True)
            print(f"__RDS_CSV_SUCCESS__:{{csv_path}}:{{len(df)}}", flush=True)
            
except ImportError:
    print("[ERROR] pyreadr library not available. Install with: pip install pyreadr", flush=True)
    print("__RDS_CSV_FAILED__", flush=True)
except Exception as e:
    print(f"[ERROR] Error converting RDS to CSV: {{e}}", flush=True)
    print("__RDS_CSV_FAILED__", flush=True)
'''

    return execute_code_via_codeact(
        task_description=f"Convert RDS {rds_path} to CSV format",
        code_template=code,
        sandbox_id=sandbox_id,
        timeout_seconds=120,
        keep_alive=True,
    )


def analyze_file_structure(
    file_path: str, sandbox_id: Optional[str] = None
) -> CodeActResult:
    """
    Analyze file structure (column names, row count, data types, etc.)

    Args:
        file_path: File path
        sandbox_id: Sandbox ID

    Returns:
        CodeActResult, parsed_result contains file structure information
    """
    code = f'''
import os
import json

file_path = "{file_path}"

result = {{
    "path": file_path,
    "exists": False,
    "file_type": None,
    "row_count": 0,
    "column_names": [],
    "columns_info": {{}}
}}

if not os.path.exists(file_path):
    print(f"__FILE_NOT_FOUND__:{{file_path}}")
else:
    result["exists"] = True
    ext = os.path.splitext(file_path)[1].lower()
    result["file_type"] = ext
    
    try:
        if ext == '.csv':
            import csv
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                result["column_names"] = reader.fieldnames or []
                rows = list(reader)
                result["row_count"] = len(rows)
                
                # Sample first few rows to infer types
                if rows and result["column_names"]:
                    for col in result["column_names"]:
                        sample_values = [row.get(col, '') for row in rows[:5]]
                        result["columns_info"][col] = {{
                            "sample_values": sample_values,
                            "type_guess": "string"
                        }}
                        # Check if numeric
                        try:
                            [float(v) for v in sample_values if v]
                            result["columns_info"][col]["type_guess"] = "numeric"
                        except ValueError:
                            pass
                        
        elif ext in ['.fasta', '.fa']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                seq_count = 0
                for line in f:
                    if line.startswith('>'):
                        seq_count += 1
                result["row_count"] = seq_count
                result["column_names"] = ["sequence_id", "sequence"]
                
        elif ext == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    result["row_count"] = len(data)
                    if data and isinstance(data[0], dict):
                        result["column_names"] = list(data[0].keys())
                elif isinstance(data, dict):
                    result["column_names"] = list(data.keys())
                    
        print("__FILE_ANALYSIS_START__")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("__FILE_ANALYSIS_END__")
        
    except Exception as e:
        print(f"__ANALYSIS_ERROR__:{{str(e)}}")
'''

    return execute_code_via_codeact(
        task_description=f"Analyze file structure {file_path}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True,
    )


def prepare_nettcr_input(
    input_csv: str,
    output_path: str,
    tcr_columns: Optional[List[str]] = None,
    sandbox_id: Optional[str] = None,
) -> CodeActResult:
    """
    Prepare NetTCR input file

    Args:
        input_csv: Input CSV file path
        output_path: Output file path
        tcr_columns: TCR sequence column name list
        sandbox_id: Sandbox ID

    Returns:
        CodeActResult
    """
    tcr_cols_str = str(tcr_columns) if tcr_columns else "None"

    code = f'''
import os
import csv
import json

input_csv = "{input_csv}"
output_path = "{output_path}"
tcr_columns = {tcr_cols_str}

    # NetTCR standard column name mapping
NETTCR_COLUMN_PATTERNS = [
    'tcr_sequence', 'CDR3', 'cdr3', 'CDR3_beta', 'cdr3_beta',
    'tcr_seq', 'tcr', 'TCR_sequence'
]

def find_tcr_columns(headers):
    """Auto-detect TCR sequence columns"""
    found = []
    for header in headers:
        header_lower = header.lower()
        for pattern in NETTCR_COLUMN_PATTERNS:
            if pattern.lower() in header_lower:
                found.append(header)
                break
    return found

try:
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        
        if tcr_columns is None:
            tcr_columns = find_tcr_columns(headers)
        
        if not tcr_columns:
            print("__NETTCR_NO_TCR_COLUMNS__")
        else:
            # Prepare output directory - use unified /data/sessions path
            container_output = output_path
            os.makedirs(os.path.dirname(container_output), exist_ok=True)
            
            results = []
            for row in reader:
                for tcr_col in tcr_columns:
                    tcr_seq = row.get(tcr_col, '').strip()
                    if tcr_seq and len(tcr_seq) > 0:
                        results.append({{
                            "tcr_sequence": tcr_seq,
                            "original_column": tcr_col
                        }})
            
            # Write output
            with open(container_output, 'w', encoding='utf-8') as out_f:
                writer = csv.DictWriter(out_f, fieldnames=["tcr_sequence", "original_column"])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"__NETTCR_PREP_SUCCESS__:{{container_output}}:{{len(results)}}")
            
except FileNotFoundError:
    print(f"__NETTCR_INPUT_NOT_FOUND__:{{input_csv}}")
except Exception as e:
    print(f"__NETTCR_PREP_ERROR__:{{str(e)}}")
'''

    return execute_code_via_codeact(
        task_description=f"Prepare NetTCR input file {input_csv}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True,
    )


# ==================== Exports ====================

__all__ = [
    # Main interface
    "execute_code_via_codeact",
    "execute_code_via_codeact_async",
    "is_codeact_available",
    # Result classes
    "CodeActResult",
    "CodeActExecutionStatus",
    # Convenience functions
    "read_remote_file",
    "list_remote_directory",
    "copy_file_in_sandbox",
    "convert_csv_to_fasta",
    "convert_rds_to_csv",
    "analyze_file_structure",
    "prepare_nettcr_input",
]
