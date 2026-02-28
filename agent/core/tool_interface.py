"""Unified tool-call interface for CodeAct and future agents."""

from typing import Any, Dict, Optional, List, Callable
import time
import os

from utils.mcp_helper import invoke_mcp_tool_sync


# ==================== Tool Call Hook System ====================
# Pre-processing hooks: executed before tool call, can modify parameters
# Post-processing hooks: executed after tool call, can modify result

_pre_call_hooks: List[Callable[[str, Dict[str, Any], Optional[Dict]], Dict[str, Any]]] = []
_post_call_hooks: List[Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = []


def register_pre_call_hook(hook: Callable[[str, Dict[str, Any], Optional[Dict]], Dict[str, Any]]):
    """
    Register a pre-call hook.
    
    Hook signature: (tool_name, parameters, config) -> modified_parameters
    The hook can modify parameters before the tool is called.
    """
    _pre_call_hooks.append(hook)


def register_post_call_hook(hook: Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]):
    """
    Register a post-call hook.
    
    Hook signature: (tool_name, parameters, result) -> modified_result
    The hook can modify the result after the tool is called.
    """
    _post_call_hooks.append(hook)


def clear_hooks():
    """Clear all registered hooks."""
    _pre_call_hooks.clear()
    _post_call_hooks.clear()


# ==================== Built-in Hooks ====================

def _csv_to_fasta_hook(tool_name: str, parameters: Dict[str, Any], config: Optional[Dict]) -> Dict[str, Any]:
    """
    Pre-call hook: Auto-convert CSV to FASTA for tools that require FASTA input.
    
    Only converts if CSV contains sequence columns (heavy_dna, Heavy_DNA, etc.)
    
    Supports tools like: analyze_vdj_batch, vquest_analysis, igblast_query
    """
    # Tools that require FASTA input
    fasta_required_tools = [
        'analyze_vdj_batch', 'vquest_analysis', 'igblast_query',
        'run_igblast', 'vdj_analysis', 'analyze_sequences'
    ]
    
    # Parameter names that typically hold sequence file paths
    sequence_params = ['sequences', 'fasta_file', 'input_fasta', 'sequence_file']
    
    if tool_name.lower() not in [t.lower() for t in fasta_required_tools]:
        return parameters
    
    # Check if any sequence parameter points to a CSV file
    for param_name in sequence_params:
        if param_name not in parameters:
            continue
        
        file_path = parameters[param_name]
        if not isinstance(file_path, str):
            continue
        
        # If it's a CSV file, check for sequence columns first
        if file_path.lower().endswith('.csv'):
            print(f"  [ToolHook] Detected CSV file for {param_name}: {file_path}")
            
            # Get sandbox_id from config, environment variable, or global state
            sandbox_id = None
            if config:
                sandbox_id = config.get("sandbox_id") or config.get("opensandbox_id")
            if not sandbox_id:
                sandbox_id = os.environ.get("OPENSANDBOX_ID")
            
            # Check if CSV has sequence columns before converting
            has_seq_cols = _check_csv_has_sequence_columns(file_path, sandbox_id)
            if not has_seq_cols:
                print(f"  [ToolHook] CSV does not contain sequence columns (heavy_dna, Heavy_DNA, etc.), skipping conversion")
                continue
            
            print(f"  [ToolHook] CSV has sequence columns, auto-converting to FASTA for {tool_name}...")
            
            # Try to convert
            fasta_path = _convert_csv_to_fasta_for_tool(file_path, sandbox_id)
            if fasta_path:
                print(f"  [ToolHook] Conversion successful: {fasta_path}")
                parameters[param_name] = fasta_path
            else:
                print(f"  [ToolHook] Conversion failed, using original path")
    
    return parameters


def _check_csv_has_sequence_columns(csv_path: str, sandbox_id: Optional[str] = None) -> bool:
    """
    Check if CSV file contains sequence columns.
    
    Sequence columns include: heavy_dna, Heavy_DNA, light_dna, Light_DNA, 
    Heavy, Light, sequence, Sequence, variant_seq, etc.
    """
    # Sequence column names to look for (case variations)
    seq_columns = [
        'heavy_dna', 'Heavy_DNA', 'HEAVY_DNA',
        'light_dna', 'Light_DNA', 'LIGHT_DNA',
        'Heavy', 'HEAVY', 'Light', 'LIGHT',
        'sequence', 'Sequence', 'SEQUENCE',
        'seq', 'Seq', 'SEQ',
        'nt_sequence', 'NT_Sequence',
        'aa_sequence', 'AA_Sequence',
        'cdr3', 'CDR3', 'cdr3_aa', 'CDR3_AA', 'cdr3_nt', 'CDR3_NT',
        'vh', 'VH', 'vl', 'VL', 'vhh', 'VHH',
        # Additional common sequence column names (e.g., variant_seq_1, variant_seq_2, ...)
        'variant_seq', 'variant_seq_1', 'variant_seq_2', 'variant_seq_3',
        'heavy_chain', 'light_chain', 'HeavyChain', 'LightChain',
        'hc_seq', 'lc_seq', 'HC_seq', 'LC_seq',
        'full_sequence', 'dna_sequence', 'nucleotide_sequence',
    ]
    
    # Regex patterns for dynamic column name matching
    import re
    seq_patterns = [
        r'^variant_seq',  # variant_seq, variant_seq_1, variant_seq_2, etc.
        r'^heavy_',       # heavy_dna, heavy_chain, etc.
        r'^light_',       # light_dna, light_chain, etc.
        r'_seq$',         # hc_seq, lc_seq, etc.
        r'_sequence$',    # full_sequence, dna_sequence, etc.
        r'^sequence$',    # sequence
        r'^seq$',         # seq
        r'^cdr3',         # cdr3, CDR3
        r'^v[hl]$',       # vh, vl
        r'^vhh$',         # vhh
    ]
    
    def _matches_seq_pattern(col_name: str) -> bool:
        """Check if column name matches any sequence pattern."""
        col_lower = col_name.lower()
        if col_name in seq_columns or col_lower in [c.lower() for c in seq_columns]:
            return True
        for pattern in seq_patterns:
            if re.match(pattern, col_lower, re.IGNORECASE):
                return True
        return False
    
    # PRIORITY 1: If we're already inside a sandbox (file exists locally), check directly
    # This is the most efficient path when running inside OpenSandbox
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=0)
            columns = list(df.columns)
            found = [c for c in columns if _matches_seq_pattern(c)]
            print(f"  [ToolHook] Direct CSV check: {len(found)} sequence columns found in {csv_path}")
            print(f"  [ToolHook] Matched columns: {found}")
            return len(found) > 0
        except Exception as e:
            print(f"  [ToolHook] Direct CSV check error: {e}")
    
    # PRIORITY 2: If sandbox_id provided and file doesn't exist locally, try remote check
    if sandbox_id and not os.path.exists(csv_path):
        check_code = f'''
import pandas as pd
import re
import sys

csv_path = "{csv_path}"
seq_columns = {seq_columns}

# Regex patterns for dynamic column name matching
seq_patterns = [
    r'^variant_seq',  # variant_seq, variant_seq_1, variant_seq_2, etc.
    r'^heavy_',       # heavy_dna, heavy_chain, etc.
    r'^light_',       # light_dna, light_chain, etc.
    r'_seq$',         # hc_seq, lc_seq, etc.
    r'_sequence$',    # full_sequence, dna_sequence, etc.
    r'^sequence$',    # sequence
    r'^seq$',         # seq
    r'^cdr3',         # cdr3, CDR3
    r'^v[hl]$',       # vh, vl
    r'^vhh$',         # vhh
]

def _matches_seq_pattern(col_name):
    col_lower = col_name.lower()
    if col_name in seq_columns or col_lower in [c.lower() for c in seq_columns]:
        return True
    for pattern in seq_patterns:
        if re.match(pattern, col_lower, re.IGNORECASE):
            return True
    return False

try:
    # Only read header (first row)
    df = pd.read_csv(csv_path, nrows=0)
    columns = list(df.columns)
    
    # Check if any sequence column exists
    found = [c for c in columns if _matches_seq_pattern(c)]
    if found:
        print(f"__HAS_SEQ_COLS__:{{','.join(found)}}")
    else:
        print("__NO_SEQ_COLS__")
except Exception as e:
    print(f"__CHECK_ERROR__:{{e}}")
'''
        try:
            from utils.opensandbox_executor import run_code_in_opensandbox_sync
            result = run_code_in_opensandbox_sync(
                code=check_code,
                task_id="check_csv_seq_cols",
                timeout_seconds=60,
                env={"OPENSANDBOX_SKIP_MCP_INSTALL": "true"}
            )
            
            if result:
                stdout = result.get("stdout", "") + result.get("formatted_output", "")
                if "__HAS_SEQ_COLS__:" in stdout:
                    return True
                elif "__NO_SEQ_COLS__" in stdout:
                    return False
        except Exception as e:
            print(f"  [ToolHook] Remote check error: {e}")
    
    # Default to False if check fails
    return False


def _convert_csv_to_fasta_for_tool(csv_path: str, sandbox_id: Optional[str] = None) -> Optional[str]:
    """
    Convert CSV file to FASTA format for tool consumption.
    
    Args:
        csv_path: Path to CSV file (in sandbox)
        sandbox_id: OpenSandbox instance ID
        
    Returns:
        Path to generated FASTA file, or None if conversion failed
    """
    import os
    from pathlib import Path
    
    # Generate output FASTA path
    # IMPORTANT: Always write to /tmp/ because shared storage (/data/) is READ-ONLY in sandbox
    csv_name = Path(csv_path).stem
    # Always use /tmp/ for output - shared storage paths are read-only
    fasta_path = f"/tmp/{csv_name}_sequences.fasta"
    
    # Sequence column names to look for
    seq_columns = [
        'Heavy_DNA', 'heavy_dna', 'Light_DNA', 'light_dna',
        'Heavy', 'Light', 'sequence', 'Sequence', 'seq',
        'nt_sequence', 'aa_sequence', 'cdr3', 'CDR3',
        # Additional common sequence column names
        'variant_seq', 'variant_seq_1', 'variant_seq_2', 'variant_seq_3',
        'VH', 'VL', 'VHH', 'vh', 'vl', 'vhh',
        'heavy_chain', 'light_chain', 'HeavyChain', 'LightChain',
        'hc_seq', 'lc_seq', 'HC_seq', 'LC_seq',
        'full_sequence', 'dna_sequence', 'nucleotide_sequence',
    ]
    
    # Regex patterns for dynamic column name matching
    import re
    seq_patterns = [
        r'^variant_seq',  # variant_seq, variant_seq_1, variant_seq_2, etc.
        r'^heavy_',       # heavy_dna, heavy_chain, etc.
        r'^light_',       # light_dna, light_chain, etc.
        r'_seq$',         # hc_seq, lc_seq, etc.
        r'_sequence$',    # full_sequence, dna_sequence, etc.
        r'^sequence$',    # sequence
        r'^seq$',         # seq
        r'^cdr3',         # cdr3, CDR3
        r'^v[hl]$',       # vh, vl
        r'^vhh$',         # vhh
    ]
    
    def _matches_seq_pattern(col_name: str) -> bool:
        col_lower = col_name.lower()
        if col_name in seq_columns or col_lower in [c.lower() for c in seq_columns]:
            return True
        for pattern in seq_patterns:
            if re.match(pattern, col_lower, re.IGNORECASE):
                return True
        return False
    
    # PRIORITY 1: If file exists locally (e.g., inside sandbox), convert directly
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            available = [c for c in df.columns if _matches_seq_pattern(c)]
            print(f"  [ToolHook] Found sequence columns: {available}")
            if available:
                # Ensure output directory exists
                os.makedirs(os.path.dirname(fasta_path) if os.path.dirname(fasta_path) else '.', exist_ok=True)
                
                # Find ID column
                id_col = None
                for c in ['main_name', 'name', 'id', 'ID', 'barcode', 'cell_id']:
                    if c in df.columns:
                        id_col = c
                        break
                
                count = 0
                with open(fasta_path, 'w') as f:
                    for idx, row in df.iterrows():
                        for col in available:
                            seq = str(row[col]).strip()
                            if seq and seq.lower() not in ['nan', 'none', '']:
                                seq_id = f"{row[id_col]}_{col}" if id_col else f"seq_{idx}_{col}"
                                f.write(f">{seq_id}\n{seq}\n")
                                count += 1
                if count > 0:
                    print(f"  [ToolHook] Direct CSV to FASTA: {count} sequences -> {fasta_path}")
                    return fasta_path
                else:
                    print(f"  [ToolHook] No valid sequences found in CSV columns: {available}")
            else:
                print(f"  [ToolHook] No sequence columns found in CSV: {list(df.columns)[:10]}")
        except Exception as e:
            print(f"  [ToolHook] Direct conversion error: {e}")
    
    # PRIORITY 2: If file doesn't exist locally but sandbox_id provided, convert remotely
    # Normalize paths to forward slashes for Linux sandbox
    csv_path_normalized = csv_path.replace("\\", "/")
    fasta_path_normalized = fasta_path.replace("\\", "/")
    
    if sandbox_id and not os.path.exists(csv_path):
        # Use built-in csv module to avoid pandas dependency
        conversion_code = f'''
import csv
import os
import re

csv_path = "{csv_path_normalized}"
fasta_path = "{fasta_path_normalized}"
seq_columns = {seq_columns}

# Regex patterns for dynamic column name matching
seq_patterns = [
    r'^variant_seq',  # variant_seq, variant_seq_1, variant_seq_2, etc.
    r'^heavy_',       # heavy_dna, heavy_chain, etc.
    r'^light_',       # light_dna, light_chain, etc.
    r'_seq$',         # hc_seq, lc_seq, etc.
    r'_sequence$',    # full_sequence, dna_sequence, etc.
    r'^sequence$',    # sequence
    r'^seq$',         # seq
    r'^cdr3',         # cdr3, CDR3
    r'^v[hl]$',       # vh, vl
    r'^vhh$',         # vhh
]

def _matches_seq_pattern(col_name):
    col_lower = col_name.lower()
    if col_name in seq_columns or col_lower in [c.lower() for c in seq_columns]:
        return True
    for pattern in seq_patterns:
        if re.match(pattern, col_lower, re.IGNORECASE):
            return True
    return False

try:
    # Use built-in csv module instead of pandas
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    
    print(f"Read CSV: {{len(rows)}} rows, columns: {{headers}}", flush=True)
    
    # Find available sequence columns with pattern matching
    available = [col for col in headers if _matches_seq_pattern(col)]
    if not available:
        print(f"[WARN] No sequence columns in CSV: {{headers[:15]}}", flush=True)
        print("__CSV_TO_FASTA_FAILED__")
    else:
        print(f"Found columns: {{available}}", flush=True)
        
        # Find ID column
        id_col = None
        for c in ['main_name', 'name', 'id', 'ID', 'barcode', 'cell_id', 'sample_id', 'antibody_id']:
            if c in headers:
                id_col = c
                break
        
        # Generate FASTA
        os.makedirs(os.path.dirname(fasta_path) if os.path.dirname(fasta_path) else '.', exist_ok=True)
        count = 0
        with open(fasta_path, 'w') as f:
            for idx, row in enumerate(rows):
                for col in available:
                    seq = str(row.get(col, '')).strip()
                    if seq and seq.lower() not in ['nan', 'none', '', 'na', 'null']:
                        seq_id = f"{{row[id_col]}}_{{col}}" if id_col and row.get(id_col) else f"seq_{{idx}}_{{col}}"
                        f.write(f">{{seq_id}}\\n{{seq}}\\n")
                        count += 1
        
        print(f"Generated FASTA: {{count}} sequences -> {{fasta_path}}", flush=True)
        print(f"__CSV_TO_FASTA_SUCCESS__:{{fasta_path}}:{{count}}")
except Exception as e:
    print(f"[ERROR] {{e}}", flush=True)
    import traceback
    traceback.print_exc()
    print("__CSV_TO_FASTA_FAILED__")
'''
        try:
            from utils.opensandbox_executor import run_code_in_opensandbox_sync
            result = run_code_in_opensandbox_sync(
                code=conversion_code,
                task_id="csv_to_fasta_conversion",
                timeout_seconds=120,
                env={"OPENSANDBOX_SKIP_MCP_INSTALL": "true"}
            )
            
            stdout = result.get("stdout", "") + result.get("formatted_output", "") if result else ""
            if result and "__CSV_TO_FASTA_SUCCESS__" in stdout:
                for line in stdout.split("\n"):
                    if "__CSV_TO_FASTA_SUCCESS__:" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            return parts[1]
        except Exception as e:
            print(f"  [ToolHook] Remote conversion error: {e}")
    
    return None


# Register built-in hooks
register_pre_call_hook(_csv_to_fasta_hook)


# ==================== MCP Output Parsing ====================

def _parse_mcp_tool_output(output: Any) -> tuple[Any, Optional[str], Optional[str]]:
    """
    Parse MCP tool output to extract inner status/error information.
    
    MCP tools return responses in format:
    [{"type": "text", "text": "{\"status\": \"error\", \"error\": \"...\"}"}]
    
    This function extracts the inner status/error from the nested payload.
    
    Args:
        output: Raw output from MCP tool (usually a list of content blocks)
        
    Returns:
        Tuple of (parsed_output, error_message, error_type)
        - parsed_output: Extracted data or original output
        - error_message: Error message if tool execution failed, None otherwise
        - error_type: Error type if available, None otherwise
    """
    import json
    
    if not isinstance(output, list):
        return output, None, None
    
    # Extract text content from MCP response
    text_content = None
    for item in output:
        if isinstance(item, dict) and item.get("type") == "text":
            text_content = item.get("text", "")
            break
    
    if not text_content:
        return output, None, None
    
    # Try to parse as JSON
    try:
        parsed = json.loads(text_content)
        
        # Check for tool-level error
        if isinstance(parsed, dict):
            inner_status = parsed.get("status", "")
            inner_error = parsed.get("error")
            inner_error_type = parsed.get("error_type")
            
            if inner_status == "error" or inner_error:
                error_msg = inner_error or parsed.get("message", "Tool execution failed")
                return parsed, error_msg, inner_error_type
            
            # Check for streaming_task response (task submitted successfully)
            if parsed.get("type") == "streaming_task":
                return parsed, None, None
            
            # Normal success response
            return parsed, None, None
        
        return parsed, None, None
        
    except json.JSONDecodeError:
        # Not JSON, return as-is
        return text_content, None, None


# ==================== Main Tool Call Function ====================

def call_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    service_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Call an MCP tool with a unified response schema.
    
    Supports pre-call and post-call hooks for parameter/result transformation.

    Returns a dict with:
    status/output/error/error_type/execution_time_ms/tool_name/service_id
    """
    start_time = time.monotonic()
    result: Dict[str, Any] = {}
    
    # Execute pre-call hooks
    modified_params = parameters.copy()
    for hook in _pre_call_hooks:
        try:
            modified_params = hook(tool_name, modified_params, config)
        except Exception as e:
            print(f"  [ToolHook] Pre-call hook error: {e}")

    try:
        result = invoke_mcp_tool_sync(tool_name=tool_name, parameters=modified_params, config=config)
        status = result.get("status", "failed")
        output = result.get("output")
        error = result.get("error")
        error_type = result.get("error_type")
        
        # CRITICAL: Parse nested MCP tool response to detect tool-level errors
        # MCP returns: [{"type": "text", "text": "{\"status\": \"error\", \"error\": \"...\"}"}]
        # We need to extract the inner status/error from the payload
        if status == "success" and output is not None:
            parsed_output, inner_error, inner_error_type = _parse_mcp_tool_output(output)
            if inner_error:
                # Tool communication succeeded but tool execution failed
                status = "failed"
                error = inner_error
                error_type = inner_error_type or "ToolExecutionError"
                output = parsed_output
                print(f"  [ToolInterface] MCP tool {tool_name} execution failed: {error[:200]}...")
        
        if status != "success" and not error_type:
            error_type = "ToolError"
    except Exception as exc:
        status = "failed"
        output = None
        error = str(exc)
        error_type = type(exc).__name__

    execution_time_ms = int((time.monotonic() - start_time) * 1000)
    final_result = {
        "status": status,
        "output": output,
        "error": error,
        "error_type": error_type,
        "execution_time_ms": execution_time_ms,
        "tool_name": tool_name,
        "service_id": service_id or (result.get("service_id") if isinstance(result, dict) else None)
    }
    
    # Execute post-call hooks
    for hook in _post_call_hooks:
        try:
            final_result = hook(tool_name, modified_params, final_result)
        except Exception as e:
            print(f"  [ToolHook] Post-call hook error: {e}")
    
    return final_result

