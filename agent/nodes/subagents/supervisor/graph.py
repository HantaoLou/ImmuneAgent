from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import shutil
import sys
import os
import json
import re
import asyncio
import uuid
from datetime import datetime
from pathlib import Path


# ---------------------- File Source Types ----------------------
class FileSourceType(str, Enum):
    """File source type enumeration"""
    LOCAL = "local"       # Local file (on the machine running agent), needs upload to sandbox
    REMOTE = "remote"     # File on sandbox server (e.g., /data/...), already exists, no upload needed
    URL = "url"           # Download link, needs to be downloaded in sandbox


class DetectedFile(BaseModel):
    """Detected file information"""
    path: str = Field(description="File path or URL")
    source_type: FileSourceType = Field(description="File source type")
    suggested_name: Optional[str] = Field(default=None, description="Suggested filename (for URLs)")

from .prompt import TASK_CLASSIFICATION_SYSTEM_PROMPT, get_task_classification_user_prompt

# Import main graph state (for state mapping)
# Add agent directory to path (support import from subgraph directory)
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType

# LLM-related imports (using common LLM factory)
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None
    HumanMessage = None
    SystemMessage = None
    print("Warning: langchain-related libraries not installed, will use keyword matching as fallback")


# ---------------------- LLM Structured Extraction Schema ----------------------
class ExtractedFile(BaseModel):
    """File information extracted by LLM"""
    path: str = Field(description="File path or URL")
    purpose: str = Field(description="File purpose, e.g., metadata, antigen_sequence, antibody_data")
    format: str = Field(default="unknown", description="File format: csv, fasta, pdb, rds, json, txt")
    source: str = Field(default="remote", description="File source: local / remote / url")


class ExtractedParam(BaseModel):
    """Parameter extracted by LLM"""
    name: str = Field(description="Parameter name (snake_case format)")
    value: Any = Field(description="Parameter value")
    description: Optional[str] = Field(default=None, description="Parameter description")


class LLMExtractionResult(BaseModel):
    """LLM structured extraction result"""
    task_description: str = Field(default="", description="One-sentence description of the task user wants to accomplish")
    target_organism: Optional[str] = Field(default=None, description="Target organism/pathogen (e.g., H5N1, SARS-CoV-2)")
    mcp_services: List[str] = Field(default_factory=list, description="List of MCP services to use")
    files: List[ExtractedFile] = Field(default_factory=list, description="List of detected files")
    parameters: List[ExtractedParam] = Field(default_factory=list, description="List of extracted parameters")
    notes: List[str] = Field(default_factory=list, description="User's special requirements or notes")
    analysis_type: str = Field(default="other", description="Analysis type: antibody_discovery / structure_prediction / sequence_analysis / data_integration / other")


# ---------------------- LLM Extraction Prompts ----------------------
PARAMETER_EXTRACTION_SYSTEM_PROMPT = """You are a bioinformatics task analysis expert. Your task is to extract structured information from user input.

## Output Format Requirements
Output strictly in the following JSON format, do not include any other text:

```json
{
  "task_description": "A one-sentence description of the task the user wants to accomplish",
  "target_organism": "Target organism/pathogen (e.g., H5N1, SARS-CoV-2, flu), null if not specified",
  "mcp_services": ["service_name_1", "service_name_2"],
  "files": [
    {
      "path": "Full file path or URL",
      "purpose": "File purpose (e.g., metadata, antigen_sequence, antibody_data, benchmark_data)",
      "format": "File format (csv, fasta, pdb, rds, json, txt)",
      "source": "Source type: local / remote / url"
    }
  ],
  "parameters": [
    {
      "name": "Parameter name (use snake_case format)",
      "value": "Parameter value",
      "description": "Parameter description"
    }
  ],
  "notes": ["User's special requirements or notes"],
  "analysis_type": "Analysis type"
}
```

## File Source Classification Rules
- Paths starting with /data/, /home/sandbox/, /opt/, /mnt/, /shared/ → "remote" (files on the server)
- Starting with http:// or https:// → "url" (download links)
- Windows paths (e.g., C:\\, D:/) or relative paths (./, ../) → "local" (local files)

## Parameter Name Normalization Rules
- Use snake_case format (lowercase letters, underscore separated)
- Common mappings:
  - "antigen file" → "antigen_file"
  - "metadata" → "metadata_file"  
  - "RDS file" → "rds_file"
  - "output directory" → "output_dir"
  - "antibody file" → "antibody_file"

## Analysis Types
- antibody_discovery: Antibody discovery, broadly neutralizing antibody identification
- structure_prediction: Protein structure prediction
- sequence_analysis: Sequence analysis, alignment, annotation
- data_integration: Data integration, statistical analysis
- other: Other types

## Important Notes
1. File paths must be extracted completely, do not truncate
2. If user explicitly specifies MCP services, extract the complete service name list
3. **CRITICAL**: File paths should ONLY be placed in the "files" field, NOT in "parameters"
   - If user writes "- metadata: /path/to/file.csv", extract it as a file with purpose="metadata", NOT as a parameter
   - The "parameters" field is ONLY for non-file values like thresholds, model names, counts, etc.
4. Use notes for user's special requirements (e.g., "All tools must be utilized")
5. Avoid duplicate information - each piece of information should appear in only one place
"""

PARAMETER_EXTRACTION_USER_PROMPT = """Please extract structured information from the following user input:

---
{user_input}
---

Output strictly in JSON format, do not include any explanation or other text."""


# ---------------------- Data Models ----------------------
class ExtractedParameter(BaseModel):
    """Parameter extracted from user input"""
    name: str = Field(description="Parameter name")
    value: Any = Field(description="Parameter value")
    source: str = Field(default="user_input", description="Parameter source: user_input, file, inferred")
    description: Optional[str] = Field(default=None, description="Parameter description")


class FileAnalysis(BaseModel):
    """File analysis result"""
    original_path: str = Field(description="Original file path")
    sandbox_path: str = Field(description="File path in sandbox")
    file_type: str = Field(description="File type: csv, fasta, pdb, json, txt, etc.")
    file_size: int = Field(default=0, description="File size in bytes")
    row_count: Optional[int] = Field(default=None, description="Row count (for tabular files)")
    column_names: Optional[List[str]] = Field(default=None, description="Column names (for tabular files)")
    sample_data: Optional[str] = Field(default=None, description="Sample data preview")
    content_summary: Optional[str] = Field(default=None, description="Content summary generated by LLM")
    detected_data_type: Optional[str] = Field(default=None, description="Detected data type: sequence, structure, metadata, etc.")
    suggested_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Suggested parameters inferred from file")


class InputPreprocessResult(BaseModel):
    """Input preprocessing result"""
    session_id: str = Field(description="Unique session ID for traceability and file directory organization")
    extracted_parameters: List[ExtractedParameter] = Field(default_factory=list, description="List of parameters extracted from user input")
    file_analyses: List[FileAnalysis] = Field(default_factory=list, description="List of file analysis results")
    processed_input: str = Field(description="Processed user input (with file paths removed, etc.)")
    has_files: bool = Field(default=False, description="Whether files are included")
    parameter_table: Dict[str, Any] = Field(default_factory=dict, description="Parameter table for subsequent processes")
    sandbox_id: Optional[str] = Field(default=None, description="OpenSandbox instance ID (if created)")
    sandbox_data_dir: Optional[str] = Field(default=None, description="Data directory path in sandbox")


# ---------------------- Supervisor State Model ----------------------
class SupervisorState(BaseModel):
    """Supervisor Agent subgraph state"""
    user_input: str = Field(description="User's original input")
    user_task_type: Optional[UserTaskType] = Field(default=None, description="User task type")
    uploaded_files: List[str] = Field(default_factory=list, description="List of uploaded file paths (original paths)")
    sandbox_file_paths: Dict[str, str] = Field(default_factory=dict, description="Sandbox file path mapping (original path -> sandbox path)")
    sandbox_dir: str = Field(description="Sandbox directory path")
    execution_plan: Optional[str] = Field(default=None, description="Execution plan (if user provided a plan)")
    
    # Input preprocessing result fields
    session_id: Optional[str] = Field(default=None, description="Unique session ID")
    preprocess_result: Optional[InputPreprocessResult] = Field(default=None, description="Input preprocessing result")
    extracted_parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameter table")
    file_analyses: List[FileAnalysis] = Field(default_factory=list, description="File analysis results")
    opensandbox_id: Optional[str] = Field(default=None, description="OpenSandbox instance ID")
    sandbox_data_dir: Optional[str] = Field(default=None, description="Data directory path in sandbox")


# ---------------------- Input Preprocessing Node ----------------------
def _generate_session_id() -> str:
    """Generate unique session ID, format: date_time_shortUUID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


def preprocess_user_input_node(state: SupervisorState) -> SupervisorState:
    """
    Input preprocessing node:
    1. Generate unique session ID
    2. Use LLM for structured extraction (files, parameters, services, etc.)
    3. Validate and classify file sources
    4. Upload files to OpenSandbox (using unique directory)
    5. Run code in OpenSandbox to analyze files
    6. Use LLM to analyze file content
    7. Generate parameter table for subsequent processes
    """
    # Check if preprocessing was already done (avoid duplicate processing)
    if state.session_id and state.extracted_parameters and state.file_analyses:
        print("=" * 60)
        print("Input Preprocessing Node - SKIPPED (already done)")
        print(f"  Session ID: {state.session_id}")
        print(f"  Files: {len(state.file_analyses)}, Parameters: {len(state.extracted_parameters)}")
        print("=" * 60)
        return state
    
    print("=" * 60)
    print("Input Preprocessing Node Started")
    print("=" * 60)
    
    user_input = state.user_input
    
    # 0. Generate or reuse session ID
    session_id = state.session_id or _generate_session_id()
    print(f"  Session ID: {session_id}")
    
    # Two path conventions for the same directory:
    # 1. Container path (/tmp/sessions/...) - used inside sandbox containers
    # 2. Server path (/data/sessions/...) - used by MCP services and in parameter tables
    # 
    # These are the SAME directory, connected via volume mount:
    #   Host: /data/sessions -> Container: /tmp/sessions
    #
    # MCP services run on the server and access /data/sessions directly
    # Sandbox code runs in containers and accesses /tmp/sessions
    
    # Server path - for parameter table and MCP service calls
    sandbox_data_dir = f"/data/sessions/{session_id}"
    sandbox_input_dir = f"{sandbox_data_dir}/input"
    sandbox_output_dir = f"{sandbox_data_dir}/output"
    
    # Container path - for code executed inside sandbox containers
    sandbox_container_dir = f"/tmp/sessions/{session_id}"
    sandbox_container_input = f"{sandbox_container_dir}/input"
    sandbox_container_output = f"{sandbox_container_dir}/output"
    
    # 1. Use LLM for structured extraction
    llm_result = _llm_extract_structured_input(user_input)
    
    # Print LLM extraction results
    print(f"  [Task] Task Description: {llm_result.task_description}")
    if llm_result.target_organism:
        print(f"  [Target] Target Organism: {llm_result.target_organism}")
    if llm_result.mcp_services:
        print(f"  [Services] MCP Services: {llm_result.mcp_services}")
    if llm_result.analysis_type != 'other':
        print(f"  [Type] Analysis Type: {llm_result.analysis_type}")
    
    # 2. Convert LLM extracted file info to DetectedFile list
    detected_files = _convert_llm_files_to_detected(llm_result)
    
    # Add already uploaded files (state.uploaded_files)
    for uploaded_file in state.uploaded_files:
        if uploaded_file not in [f.path for f in detected_files]:
            source_type = _classify_file_source(uploaded_file)
            detected_files.append(DetectedFile(
                path=uploaded_file, 
                source_type=source_type
            ))
    
    # Group by source type
    local_files = []   # Local files, need upload
    remote_files = []  # Sandbox server files, verify existence
    url_files = []     # URL, download in sandbox
    
    for f in detected_files:
        if f.source_type == FileSourceType.URL:
            url_files.append(f)
            print(f"  [URL] URL File: {f.path}")
        elif f.source_type == FileSourceType.REMOTE:
            remote_files.append(f)
            print(f"  [Remote] Remote File: {f.path}")
        elif f.source_type == FileSourceType.LOCAL:
            if Path(f.path).exists():
                local_files.append(f)
                print(f"  [Local] Local File: {f.path}")
            else:
                print(f"  [WARN] Local file not found, skipping: {f.path}")
    
    # Aggregate valid files
    valid_files = local_files + remote_files + url_files
    print(f"  Valid Files: local={len(local_files)}, remote={len(remote_files)}, url={len(url_files)}")
    
    # 3. Convert LLM extraction result to parameter dict
    extracted_params = _convert_llm_result_to_params(llm_result)
    print(f"  Extracted Parameters: {json.dumps(extracted_params, ensure_ascii=False, indent=2)[:300]}...")
    
    # 4. Process files: different strategies based on source type
    file_analyses = []
    sandbox_file_paths = dict(state.sandbox_file_paths)
    sandbox_id = None
    
    if valid_files:
        # Use OpenSandbox to upload and analyze files
        try:
            sandbox_id, sandbox_file_paths, file_analyses = asyncio.run(
                _upload_and_analyze_files_in_opensandbox(
                    local_files=local_files,
                    remote_files=remote_files,
                    url_files=url_files,
                    user_input=user_input,
                    sandbox_container_input=sandbox_container_input,  # Container path for code execution
                    sandbox_server_input=sandbox_input_dir,           # Server path for MCP services
                    llm_result=llm_result
                )
            )
            
            # Merge parameters inferred from file analysis
            for analysis in file_analyses:
                if analysis.suggested_parameters:
                    for key, value in analysis.suggested_parameters.items():
                        if key not in extracted_params:
                            extracted_params[key] = value
                            
        except Exception as e:
            print(f"  [WARN] OpenSandbox file processing failed, falling back to local analysis: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to local analysis (local files only)
            for f in local_files:
                local_analysis = _analyze_file_locally(f.path, sandbox_input_dir)
                if local_analysis:
                    file_analyses.append(local_analysis)
                    sandbox_file_paths[f.path] = f"{sandbox_input_dir}/{Path(f.path).name}"
            
            # Remote files: create placeholder FileAnalysis with LLM-extracted info
            for f in remote_files:
                sandbox_file_paths[f.path] = f.path
                # Find matching file info from LLM extraction
                llm_file_info = _find_llm_file_info(f.path, llm_result)
                file_type = Path(f.path).suffix.lstrip('.').lower() or 'unknown'
                purpose = llm_file_info.get('purpose') if llm_file_info else 'unknown'
                
                # Generate meaningful summary based on file type and purpose
                summary_parts = []
                summary_parts.append(f"Remote {file_type.upper()} file on sandbox server.")
                if purpose and purpose != 'unknown':
                    summary_parts.append(f"Purpose: {purpose}.")
                summary_parts.append("Content will be analyzed in sandbox during execution.")
                
                file_analysis = FileAnalysis(
                    original_path=f.path,
                    sandbox_path=f.path,  # Remote files use original path
                    file_type=file_type,
                    file_size=0,  # Unknown until analyzed in sandbox
                    detected_data_type=purpose if purpose != 'unknown' else None,
                    content_summary=" ".join(summary_parts),
                    suggested_parameters=None,
                )
                file_analyses.append(file_analysis)
                print(f"  [Fallback] Remote file registered: {Path(f.path).name} ({purpose})")
            
            # URL files: create placeholder FileAnalysis
            for f in url_files:
                sandbox_path = f"{sandbox_input_dir}/{f.suggested_name or 'downloaded_file'}"
                sandbox_file_paths[f.path] = sandbox_path
                file_analysis = FileAnalysis(
                    original_path=f.path,
                    sandbox_path=sandbox_path,
                    file_type=Path(f.suggested_name or f.path).suffix.lstrip('.').lower() or 'unknown',
                    file_size=0,
                    detected_data_type=None,
                    content_summary=f"URL file: will be downloaded and analyzed when executed in sandbox.",
                    suggested_parameters=None,
                )
                file_analyses.append(file_analysis)
    
    # 5. Use LLM for file content summarization (if files exist and LLM available)
    if file_analyses and LLM_AVAILABLE:
        file_analyses = _llm_summarize_files(file_analyses, user_input)
    
    # 5.5. CSV → FASTA conversion for igblast (if needed)
    fasta_mappings = {}
    mcp_services = llm_result.mcp_services or []
    # Also check task description for igblast keywords
    task_lower = (llm_result.task_description or "").lower() + " " + user_input.lower()
    if any(kw in task_lower for kw in ['igblast', 'vdj', 'v(d)j', 'antibody', 'bcr', 'tcr', 'immunoglobulin']):
        mcp_services = list(set(mcp_services + ['igblast']))
    
    if mcp_services:
        # 安全地运行异步代码（处理事件循环可能不存在的情况）
        try:
            # 尝试获取正在运行的事件循环
            loop = asyncio.get_running_loop()
            # 如果有正在运行的事件循环，需要在新线程中运行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    _convert_csv_to_fasta_if_needed(
                        file_analyses=file_analyses,
                        mcp_services=mcp_services,
                        sandbox_id=sandbox_id,
                        sandbox_input_dir=sandbox_input_dir,
                        sandbox_output_dir=sandbox_output_dir
                    )
                )
                file_analyses, fasta_mappings = future.result()
        except RuntimeError:
            # 没有正在运行的事件循环，可以直接使用 asyncio.run
            file_analyses, fasta_mappings = asyncio.run(
                _convert_csv_to_fasta_if_needed(
                    file_analyses=file_analyses,
                    mcp_services=mcp_services,
                    sandbox_id=sandbox_id,
                    sandbox_input_dir=sandbox_input_dir,
                    sandbox_output_dir=sandbox_output_dir
                )
            )
    
    # 6. Generate parameter table
    parameter_table = _build_parameter_table(extracted_params, file_analyses, sandbox_file_paths, fasta_mappings)
    
    # 7. Generate processed user input (remove file paths, etc.)
    valid_file_paths = [f.path for f in valid_files]
    processed_input = _clean_user_input(user_input, valid_file_paths)
    
    # Build preprocessing result
    preprocess_result = InputPreprocessResult(
        session_id=session_id,
        extracted_parameters=[
            ExtractedParameter(name=k, value=v, source="user_input")
            for k, v in extracted_params.items()
        ],
        file_analyses=file_analyses,
        processed_input=processed_input,
        has_files=len(file_analyses) > 0,
        parameter_table=parameter_table,
        sandbox_id=sandbox_id,
        sandbox_data_dir=sandbox_data_dir
    )
    
    # Update state
    state.session_id = session_id
    state.preprocess_result = preprocess_result
    state.extracted_parameters = parameter_table
    state.file_analyses = file_analyses
    state.sandbox_file_paths = sandbox_file_paths
    # Convert DetectedFile list to string list
    state.uploaded_files = [f.path for f in valid_files]
    state.opensandbox_id = sandbox_id
    state.sandbox_data_dir = sandbox_data_dir
    
    print(f"  Parameter Table: {json.dumps(parameter_table, ensure_ascii=False, indent=2)[:500]}")
    print(f"  File Analysis Count: {len(file_analyses)}")
    print(f"  Sandbox Data Dir: {sandbox_data_dir}")
    if sandbox_id:
        print(f"  OpenSandbox ID: {sandbox_id}")
    print("Input Preprocessing Node Completed")
    print("=" * 60)
    
    return state


async def _upload_and_analyze_files_in_opensandbox(
    local_files: List[DetectedFile],
    remote_files: List[DetectedFile],
    url_files: List[DetectedFile],
    user_input: str,
    sandbox_container_input: str = "/tmp/sessions/default/input",
    sandbox_server_input: str = "/data/sessions/default/input",
    llm_result: Optional[LLMExtractionResult] = None
) -> tuple[Optional[str], Dict[str, str], List[FileAnalysis]]:
    """
    Process three types of files in OpenSandbox
    
    Args:
        local_files: List of local files (need upload)
        remote_files: List of sandbox server files (use directly)
        url_files: List of URL files (need download in sandbox)
        user_input: User's original input
        sandbox_container_input: Input directory path INSIDE the container (/tmp/sessions/...)
        sandbox_server_input: Input directory path on SERVER (/data/sessions/...) - used for MCP services
        llm_result: LLM extraction result for file purpose info
    
    Returns:
        (sandbox_id, sandbox_file_paths, file_analyses)
        Note: sandbox_file_paths uses SERVER paths (/data/sessions/...) for MCP service access
    """
    from utils.opensandbox_executor import (
        is_opensandbox_enabled,
        run_code_in_opensandbox,
    )
    
    if not is_opensandbox_enabled():
        raise RuntimeError("OpenSandbox not enabled")
    
    # Prepare file data
    files_data = {}
    
    # 1. Local files: read content, need upload to sandbox
    for f in local_files:
        try:
            # Get user-specified purpose from LLM extraction
            llm_file_info = _find_llm_file_info(f.path, llm_result) if llm_result else None
            user_purpose = llm_file_info.get('purpose') if llm_file_info else None
            
            with open(f.path, 'rb') as file:
                content = file.read()
            # Try to decode as text, otherwise use base64
            try:
                text_content = content.decode('utf-8')
                files_data[f.path] = {
                    "type": "local_text",
                    "name": Path(f.path).name,
                    "content": text_content,
                    "user_purpose": user_purpose  # Pass user-specified purpose
                }
            except UnicodeDecodeError:
                import base64
                files_data[f.path] = {
                    "type": "local_binary",
                    "name": Path(f.path).name,
                    "content": base64.b64encode(content).decode('ascii'),
                    "user_purpose": user_purpose  # Pass user-specified purpose
                }
        except Exception as e:
            print(f"  [WARN] Cannot read local file {f.path}: {e}")
            continue
    
    # 2. Sandbox server files: no upload needed, just analyze
    for f in remote_files:
        # Get user-specified purpose from LLM extraction
        llm_file_info = _find_llm_file_info(f.path, llm_result) if llm_result else None
        user_purpose = llm_file_info.get('purpose') if llm_file_info else None
        files_data[f.path] = {
            "type": "remote",
            "name": Path(f.path).name,
            "sandbox_path": f.path,  # Keep original path
            "user_purpose": user_purpose  # Pass user-specified purpose
        }
    
    # 3. URL files: download in sandbox
    for f in url_files:
        url_filename = f.suggested_name or "downloaded_file"
        # Get user-specified purpose from LLM extraction
        llm_file_info = _find_llm_file_info(f.path, llm_result) if llm_result else None
        user_purpose = llm_file_info.get('purpose') if llm_file_info else None
        files_data[f.path] = {
            "type": "url",
            "name": url_filename,
            "user_purpose": user_purpose  # Pass user-specified purpose
        }
    
    if not files_data:
        return None, {}, []
    
    # Generate code to execute in sandbox (uses CONTAINER path)
    analysis_code = _generate_file_analysis_code(files_data, sandbox_container_input)
    
    # Execute in OpenSandbox
    result = await run_code_in_opensandbox(
        code=analysis_code,
        task_id="file_preprocess",
        timeout_seconds=120,
        env={"OPENSANDBOX_SKIP_MCP_INSTALL": "true"},
    )
    
    sandbox_id = result.get("sandbox_id")
    stdout = result.get("stdout", "") + result.get("formatted_output", "")
    
    # Debug: print sandbox execution result
    print(f"  [Debug] Sandbox result keys: {list(result.keys())}")
    if result.get("stderr"):
        print(f"  [Debug] Sandbox stderr: {result.get('stderr')[:500]}")
    if not stdout:
        print(f"  [Debug] No stdout captured, checking result: {str(result)[:500]}")
    else:
        print(f"  [Debug] Stdout length: {len(stdout)}, contains __FILE_ANALYSIS_RESULT__: {'__FILE_ANALYSIS_RESULT__' in stdout}")
        if "__FILE_ANALYSIS_RESULT__" not in stdout:
            print(f"  [Debug] Stdout preview: {stdout[:1000]}")
    
    # Parse analysis result
    sandbox_file_paths = {}
    file_analyses = []
    
    # Extract JSON result from output
    if "__FILE_ANALYSIS_RESULT__" in stdout:
        try:
            json_str = stdout.split("__FILE_ANALYSIS_RESULT__")[1].strip()
            print(f"  [Debug] After split, json_str length: {len(json_str)}")
            print(f"  [Debug] json_str preview: {json_str[:300]}")
            
            # Find JSON end position
            json_end = json_str.find("\n__END__")
            if json_end > 0:
                json_str = json_str[:json_end]
            else:
                print(f"  [Debug] __END__ not found, using full string")
            
            print(f"  [Debug] Final json_str: {json_str[:500]}")
            analysis_result = json.loads(json_str)
            print(f"  [Debug] Parsed result: files count = {len(analysis_result.get('files', []))}")
            
            for file_info in analysis_result.get("files", []):
                original_path = file_info.get("original_path", "")
                container_path = file_info.get("sandbox_path", "")
                file_status = file_info.get("status", "ok")
                
                # CRITICAL: Convert container path (/tmp/sessions/...) to server path (/data/sessions/...)
                # MCP services access files via /data/sessions, not /tmp/sessions
                if container_path.startswith("/tmp/sessions/"):
                    sandbox_path = container_path.replace("/tmp/sessions/", "/data/sessions/", 1)
                else:
                    sandbox_path = container_path
                
                print(f"  [Debug] Processing file: {original_path} -> {sandbox_path} (container: {container_path}, status: {file_status})")
                
                if original_path and sandbox_path:
                    sandbox_file_paths[original_path] = sandbox_path
                    
                    # Get LLM file info for purpose/data type
                    llm_file_info = _find_llm_file_info(original_path, llm_result) if llm_result else None
                    purpose = llm_file_info.get('purpose') if llm_file_info else None
                    
                    # Handle different file statuses
                    if file_status == "not_found":
                        # File not found in container, but might be accessible to MCP services
                        error_msg = file_info.get('error', '')
                        if "Check if /data is mounted" in error_msg:
                            # This is a remote file that's not accessible in container
                            # But MCP services on server can access it directly
                            content_summary = f"Remote file on server (not accessible in container). MCP services can access it directly at {container_path}. Purpose: {purpose or 'unknown'}."
                            # Update sandbox_path to use original server path for MCP services
                            if container_path.startswith("/data/"):
                                sandbox_path = container_path  # Keep server path for MCP
                        else:
                            content_summary = f"File not found in sandbox. {error_msg} Purpose: {purpose or 'unknown'}."
                    elif file_status == "remote_only":
                        # Remote file that's only accessible to MCP services
                        content_summary = f"Remote file on server. MCP services can access it directly. Purpose: {purpose or 'unknown'}."
                        # Ensure we use the original server path for MCP services
                        if container_path.startswith("/data/"):
                            sandbox_path = container_path
                    elif file_status == "analysis_failed":
                        # Analysis failed but file might still be accessible
                        error_msg = file_info.get('error', '')
                        content_summary = f"File analysis failed: {error_msg}. File may still be accessible to MCP services. Purpose: {purpose or 'unknown'}."
                    else:
                        content_summary = None  # Will be filled by LLM later
                    
                    analysis = FileAnalysis(
                        original_path=original_path,
                        sandbox_path=sandbox_path,
                        file_type=file_info.get("file_type", "unknown"),
                        file_size=file_info.get("file_size", 0),
                        row_count=file_info.get("row_count"),
                        column_names=file_info.get("column_names"),
                        sample_data=file_info.get("sample_data"),
                        detected_data_type=file_info.get("detected_data_type") or purpose,
                        suggested_parameters=file_info.get("suggested_parameters"),
                        content_summary=content_summary,
                    )
                    file_analyses.append(analysis)
                    
        except (json.JSONDecodeError, IndexError) as e:
            print(f"  [WARN] Failed to parse file analysis result: {e}")
            print(f"  [Debug] json_str was: {json_str[:500] if 'json_str' in dir() else 'undefined'}")
    
    return sandbox_id, sandbox_file_paths, file_analyses


def _generate_file_analysis_code(files_data: Dict[str, Dict], sandbox_input_dir: str) -> str:
    """
    Generate file analysis code to execute in OpenSandbox
    
    Supports three file sources:
    - local_text / local_binary: Local files, content included in files_data
    - remote: Sandbox server files, analyze using original path directly
    - url: Download link, download in sandbox
    
    Args:
        files_data: File data dictionary
        sandbox_input_dir: Input directory path in sandbox
    """
    # Serialize file data to JSON
    files_json = json.dumps(files_data, ensure_ascii=False)
    
    code = f'''
import os
import json
import csv
from pathlib import Path

# File data
files_data = {files_json}

# Create session-specific file directory with full permissions
# This ensures MCP services (running as different users) can also write to these directories
import os
user_files_dir = Path("{sandbox_input_dir}")
# CRITICAL: Create all parent directories with full permissions
user_files_dir.mkdir(parents=True, exist_ok=True)
# Set permissions to 777 (rwxrwxrwx) so all users can read/write
try:
    # Set permissions on input directory
    os.chmod(str(user_files_dir), 0o777)
    # Set permissions on parent session directory
    if user_files_dir.parent.exists():
        os.chmod(str(user_files_dir.parent), 0o777)
    # Set permissions on grandparent sessions directory
    if user_files_dir.parent.parent.exists():
        os.chmod(str(user_files_dir.parent.parent), 0o777)
except Exception as perm_err:
    print(f"Permission setting warning (non-critical): {{perm_err}}")
    pass  # Ignore permission errors on Windows or if already exists

# Also create output directory with full permissions
output_dir = user_files_dir.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(str(output_dir), 0o777)
except Exception:
    pass

print(f"Created directories: input={{user_files_dir}}, output={{output_dir}}")
output_dir.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(str(output_dir), 0o777)
except Exception:
    pass

results = {{"files": []}}

def analyze_file(sandbox_path: str, original_path: str, user_purpose: str = None) -> dict:
    """Analyze a single file
    
    Args:
        sandbox_path: Path to file in sandbox
        original_path: Original file path
        user_purpose: User-specified file purpose (takes priority over auto-detection)
    """
    file_size = os.path.getsize(sandbox_path)
    suffix = Path(sandbox_path).suffix.lower()
    
    file_result = {{
        "original_path": original_path,
        "sandbox_path": sandbox_path,
        "file_type": suffix.lstrip(".") or "unknown",
        "file_size": file_size,
        "row_count": None,
        "column_names": None,
        "sample_data": None,
        "detected_data_type": None,
        "suggested_parameters": {{}},
    }}
    
    # Auto-detected type (may be overridden by user_purpose)
    auto_detected_type = None
    
    try:
        # Analyze based on file type
        if suffix in (".csv", ".tsv"):
            delimiter = "\\t" if suffix == ".tsv" else ","
            with open(sandbox_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
                if rows:
                    file_result["column_names"] = rows[0]
                    file_result["row_count"] = len(rows) - 1
                    file_result["sample_data"] = "\\n".join([delimiter.join(row) for row in rows[:5]])
                    
                    # Auto-detect data type from column names
                    cols_lower = [c.lower() for c in rows[0]]
                    if any(kw in " ".join(cols_lower) for kw in ["sequence", "seq", "cdr", "heavy", "light"]):
                        auto_detected_type = "antibody_sequence"
                    elif any(kw in " ".join(cols_lower) for kw in ["antigen", "epitope"]):
                        auto_detected_type = "antigen_data"
                    else:
                        auto_detected_type = "tabular_data"
                        
        elif suffix in (".fasta", ".fa", ".fna"):
            with open(sandbox_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(2000)
            file_result["row_count"] = content.count(">")
            file_result["sample_data"] = content[:500]
            auto_detected_type = "sequence"
            
        elif suffix == ".pdb":
            with open(sandbox_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(2000)
            file_result["sample_data"] = content[:500]
            auto_detected_type = "protein_structure"
            
        elif suffix == ".json":
            with open(sandbox_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            if isinstance(data, dict):
                file_result["column_names"] = list(data.keys())[:20]
            elif isinstance(data, list):
                file_result["row_count"] = len(data)
                if data and isinstance(data[0], dict):
                    file_result["column_names"] = list(data[0].keys())[:20]
            file_result["sample_data"] = json.dumps(data, ensure_ascii=False, indent=2)[:500]
            auto_detected_type = "json_data"
            
        elif suffix == ".rds":
            # RDS is R language serialization format, cannot read directly
            file_result["sample_data"] = "[R data file, requires R environment to read]"
            auto_detected_type = "r_data"
            
        else:
            with open(sandbox_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(1000)
            file_result["sample_data"] = content[:500]
            auto_detected_type = "text"
    except Exception as e:
        file_result["sample_data"] = f"[Analysis failed: {{str(e)}}]"
    
    # CRITICAL: User-specified purpose takes priority over auto-detection
    if user_purpose:
        file_result["detected_data_type"] = user_purpose
        print(f"  [INFO] Using user-specified purpose: {{user_purpose}} (auto-detected was: {{auto_detected_type}})")
    else:
        file_result["detected_data_type"] = auto_detected_type
    
    return file_result


for original_path, file_info in files_data.items():
    file_type = file_info["type"]
    file_name = file_info["name"]
    # Get user-specified purpose (priority over auto-detection)
    user_purpose = file_info.get("user_purpose")
    
    try:
        # Process based on file source type
        if file_type == "local_text":
            # Local text file: write to sandbox
            sandbox_path = str(user_files_dir / file_name)
            with open(sandbox_path, "w", encoding="utf-8") as f:
                f.write(file_info["content"])
            print(f"[OK] Local file uploaded: {{original_path}} -> {{sandbox_path}}", flush=True)
            
        elif file_type == "local_binary":
            # Local binary file: decode and write to sandbox
            sandbox_path = str(user_files_dir / file_name)
            import base64
            with open(sandbox_path, "wb") as f:
                f.write(base64.b64decode(file_info["content"]))
            print(f"[OK] Local binary file uploaded: {{original_path}} -> {{sandbox_path}}", flush=True)
            
        elif file_type == "remote":
            # Sandbox server file: try to copy to session directory for isolation
            source_path = file_info.get("sandbox_path", original_path)
            
            # Strategy 1: Try to access file directly (if /data is mounted)
            if os.path.exists(source_path):
                # Copy file to session directory to avoid polluting original data
                import shutil
                sandbox_path = str(user_files_dir / file_name)
                try:
                    shutil.copy2(source_path, sandbox_path)
                    print(f"[OK] Remote file copied: {{source_path}} -> {{sandbox_path}}", flush=True)
                except Exception as copy_err:
                    print(f"[WARN] Failed to copy remote file: {{copy_err}}", flush=True)
                    # Fall through to use original path
                    sandbox_path = source_path
            else:
                # Strategy 2: File not accessible in container, use original path
                # MCP services run on the server and can access /data directly
                print(f"[INFO] Remote file not accessible in container ({{source_path}}), using original path for MCP services", flush=True)
                print(f"[INFO] This is expected if /data is not mounted. MCP services can access the file directly.", flush=True)
                # Use original path - MCP services will access it directly
                sandbox_path = source_path
                
                # Try to create a placeholder or read via alternative method
                # For now, we'll use the original path and let MCP services handle it
                # But we still need to analyze the file if possible
                # Check if we can read it via alternative paths or methods
                alternative_paths = [
                    source_path,  # Original path
                    source_path.replace("/data/", "/mnt/data/"),  # Alternative mount point
                    source_path.replace("/data/", "/shared/data/"),  # Another alternative
                ]
                
                accessible_path = None
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        accessible_path = alt_path
                        print(f"[INFO] Found file at alternative path: {{alt_path}}", flush=True)
                        break
                
                if accessible_path:
                    # Copy from alternative path
                    import shutil
                    sandbox_path = str(user_files_dir / file_name)
                    try:
                        shutil.copy2(accessible_path, sandbox_path)
                        print(f"[OK] Remote file copied from alternative path: {{accessible_path}} -> {{sandbox_path}}", flush=True)
                    except Exception as copy_err:
                        print(f"[WARN] Failed to copy from alternative path: {{copy_err}}", flush=True)
                        sandbox_path = source_path  # Fall back to original
                else:
                    # File truly not accessible in container
                    # Use original path - MCP services on server can access it
                    sandbox_path = source_path
                    print(f"[INFO] Using original server path for MCP services: {{sandbox_path}}", flush=True)
                    # Note: We'll still try to analyze, but if it fails, that's OK
                    # The file will be accessible to MCP services which run on the server
            
        elif file_type == "url":
            # URL file: download in sandbox
            sandbox_path = str(user_files_dir / file_name)
            import urllib.request
            print(f"⏳ Downloading URL: {{original_path}}", flush=True)
            urllib.request.urlretrieve(original_path, sandbox_path)
            print(f"[OK] URL download complete: {{original_path}} -> {{sandbox_path}}", flush=True)
        else:
            print(f"✗ Unknown file type: {{file_type}} for {{original_path}}", flush=True)
            continue
        
        # Analyze file with user-specified purpose
        file_result = analyze_file(sandbox_path, original_path, user_purpose)
        results["files"].append(file_result)
        
    except Exception as e:
        print(f"✗ File processing failed: {{original_path}} - {{e}}", flush=True)

# Output results
print("__FILE_ANALYSIS_RESULT__", flush=True)
print(json.dumps(results, ensure_ascii=False), flush=True)
print("__END__", flush=True)
'''
    return code


def _analyze_file_locally(file_path: str, sandbox_input_dir: str = "/data/sessions/default/input") -> Optional[FileAnalysis]:
    """Analyze file locally (fallback when OpenSandbox unavailable)"""
    try:
        path = Path(file_path)
        if not path.exists():
            return None
            
        suffix = path.suffix.lower()
        file_size = path.stat().st_size
        
        analysis = FileAnalysis(
            original_path=file_path,
            sandbox_path=f"{sandbox_input_dir}/{path.name}",
            file_type=suffix.lstrip('.'),
            file_size=file_size
        )
        
        # Simple analysis
        if suffix in ('.csv', '.tsv'):
            import csv
            delimiter = '\t' if suffix == '.tsv' else ','
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
                if rows:
                    analysis.column_names = rows[0]
                    analysis.row_count = len(rows) - 1
                    analysis.sample_data = '\n'.join([delimiter.join(row) for row in rows[:5]])
                    
        elif suffix in ('.fasta', '.fa', '.fna'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2000)
            analysis.row_count = content.count('>')
            analysis.sample_data = content[:500]
            analysis.detected_data_type = "sequence"
            
        return analysis
        
    except Exception as e:
        print(f"  [WARN] Local file analysis failed: {file_path} - {e}")
        return None


def _detect_file_paths(user_input: str) -> List[DetectedFile]:
    """
    Detect file paths from user input and identify file source types
    
    Supports three sources:
    1. Local file (LOCAL): Windows/Mac/Linux local paths, files exist on the machine running agent
    2. Sandbox server file (REMOTE): Paths starting with /data/, files already exist on sandbox server
    3. URL: Download links starting with http:// or https://
    """
    detected_paths = []
    
    # Regex patterns for matching file paths
    patterns = [
        # Unix absolute path
        r'(?:^|\s)(/[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)(?:\s|$|,|，)',
        # Windows path
        r'(?:^|\s)([A-Za-z]:[\\\/][a-zA-Z0-9_\-.\\/]+\.[a-zA-Z0-9]+)(?:\s|$|,|，)',
        # Relative path
        r'(?:^|\s)(\.{0,2}/[a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)(?:\s|$|,|，)',
        # URL
        r'(https?://[^\s]+\.[a-zA-Z0-9]+)',
        # Common file extension patterns
        r'(?:^|\s|["\'])([a-zA-Z0-9_\-./\\:]+\.(?:csv|tsv|fasta|fa|fna|pdb|json|txt|xlsx|xls|rds))(?:\s|$|["\',，])',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, user_input, re.IGNORECASE)
        detected_paths.extend(matches)
    
    # Deduplicate and classify file sources
    seen_paths = set()
    detected_files = []
    
    for path in detected_paths:
        path = path.strip()
        if not path or len(path) <= 3 or path in seen_paths:
            continue
        seen_paths.add(path)
        
        # Identify file source type
        source_type = _classify_file_source(path)
        
        # Generate suggested filename (for URLs)
        suggested_name = None
        if source_type == FileSourceType.URL:
            # Extract filename from URL
            url_path = path.split('?')[0]  # Remove query parameters
            suggested_name = url_path.split('/')[-1] if '/' in url_path else "downloaded_file"
        
        detected_files.append(DetectedFile(
            path=path,
            source_type=source_type,
            suggested_name=suggested_name
        ))
    
    return detected_files


def _classify_file_source(path: str) -> FileSourceType:
    """
    Classify file source type based on path format
    
    Classification rules:
    1. Starts with http:// or https:// -> URL
    2. Starts with /data/ (common sandbox server path) -> REMOTE
    3. Other Unix absolute paths (/xxx):
       - If exists locally -> LOCAL
       - If not exists locally -> REMOTE (assume sandbox server path)
    4. Windows path or relative path -> LOCAL
    """
    # URL
    if path.startswith(('http://', 'https://')):
        return FileSourceType.URL
    
    # Common sandbox server path prefixes
    remote_prefixes = [
        '/data/',
        '/home/sandbox/',
        '/opt/data/',
        '/mnt/',
        '/shared/',
    ]
    
    # Check if it's clearly a sandbox server path
    for prefix in remote_prefixes:
        if path.startswith(prefix):
            return FileSourceType.REMOTE
    
    # Unix absolute path
    if path.startswith('/'):
        # Check if exists locally
        if Path(path).exists():
            return FileSourceType.LOCAL
        else:
            # If not exists, assume sandbox server path
            return FileSourceType.REMOTE
    
    # Windows path or relative path -> LOCAL
    return FileSourceType.LOCAL


def _extract_parameters_from_input(user_input: str) -> Dict[str, Any]:
    """
    Extract parameters from user input (regex fallback)
    
    Supported formats:
    1. Markdown list format:
       use the following mcp services:
       - igblast
       - metabcr
       
    2. Key-value format:
       - metadata: /path/to/file.csv
       - antigen file: /path/to/file.csv
       
    3. Standard format:
       key=value
       key: value
    """
    params = {}
    
    # Analyze line by line
    lines = user_input.strip().split('\n')
    current_section = None
    current_list = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            # Empty line: save current list
            if current_section and current_list:
                params[current_section] = current_list if len(current_list) > 1 else current_list[0]
                current_section = None
                current_list = []
            continue
        
        # Check if it's a section header (ends with colon, next line is list)
        if line.endswith(':') and not line.startswith('-'):
            # Save previous list
            if current_section and current_list:
                params[current_section] = current_list if len(current_list) > 1 else current_list[0]
            
            # Extract header as section name
            section_name = _extract_section_name(line[:-1])
            if section_name:
                current_section = section_name
                current_list = []
            continue
        
        # Check if it's a list item
        if line.startswith('-'):
            item_content = line[1:].strip()
            
            # Check if list item contains key-value pair (- key: value)
            kv_match = re.match(r'^([^:]+):\s*(.+)$', item_content)
            if kv_match:
                key = kv_match.group(1).strip()
                value = kv_match.group(2).strip()
                # Normalize key name
                key_normalized = _normalize_param_key(key)
                params[key_normalized] = _parse_value(value)
            else:
                # Pure list item, add to current list
                if item_content:
                    current_list.append(item_content)
            continue
        
        # Check if it's a standalone key-value pair (key: value or key=value)
        kv_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_\s]*)\s*[:=]\s*(.+)$', line)
        if kv_match:
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            key_normalized = _normalize_param_key(key)
            if key_normalized not in params:
                params[key_normalized] = _parse_value(value)
    
    # Process last list
    if current_section and current_list:
        params[current_section] = current_list if len(current_list) > 1 else current_list[0]
    
    return params


def _extract_section_name(text: str) -> Optional[str]:
    """Extract section name from header text"""
    text = text.lower().strip()
    
    # Common section header mappings
    section_patterns = {
        r'mcp\s*services?': 'mcp_services',
        r'services?': 'services',
        r'parameters?': 'parameters',
        r'files?': 'files',
        r'notes?': 'notes',
        r'input\s*files?': 'input_files',
        r'output\s*files?': 'output_files',
        r'options?': 'options',
        r'settings?': 'settings',
    }
    
    for pattern, name in section_patterns.items():
        if re.search(pattern, text):
            return name
    
    # If no match, try to extract the last word
    words = text.split()
    if words:
        return _normalize_param_key(words[-1])
    
    return None


def _normalize_param_key(key: str) -> str:
    """Normalize parameter name"""
    # Remove common prefix words
    key = key.lower().strip()
    key = re.sub(r'^(the|a|an)\s+', '', key)
    
    # Convert spaces and hyphens to underscores
    key = re.sub(r'[\s\-]+', '_', key)
    
    # Remove invalid characters
    key = re.sub(r'[^a-z0-9_]', '', key)
    
    return key


def _parse_value(value: str) -> Any:
    """Parse parameter value, convert to appropriate type"""
    value = value.strip()
    
    # Try to parse as number
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # Boolean values
    if value.lower() in ('true', 'yes'):
        return True
    if value.lower() in ('false', 'no'):
        return False
    
    return value




def _llm_summarize_files(file_analyses: List[FileAnalysis], user_input: str) -> List[FileAnalysis]:
    """Use LLM to summarize file analysis results"""
    try:
        llm = _get_llm()
        if not llm:
            return file_analyses
        
        for analysis in file_analyses:
            # Skip if no sample data and already has a content summary (placeholder)
            if not analysis.sample_data and analysis.content_summary:
                print(f"  [INFO] Skipping LLM summary for {Path(analysis.original_path).name}: no sample data available")
                continue
            
            # Skip if no useful data to analyze
            if not analysis.sample_data and not analysis.column_names and not analysis.row_count:
                # Generate a basic summary based on file type and detected data type
                if not analysis.content_summary:
                    analysis.content_summary = f"{analysis.file_type.upper()} file, data type: {analysis.detected_data_type or 'unknown'}. Actual content analysis pending."
                continue
            
            try:
                prompt = f"""Analyze the following file content and provide a concise summary:

File path: {analysis.sandbox_path}
File type: {analysis.file_type}
Detected data type: {analysis.detected_data_type}
Row count: {analysis.row_count}
Column names: {analysis.column_names}
Sample data:
{analysis.sample_data or '[No sample data available]'}

User's original request: {user_input}

Please provide:
1. A brief summary of the file content (within 50 words)
2. What bioinformatics analysis task this file might be used for
3. Key parameters to extract (if any)

Return in JSON format: {{"summary": "...", "suggested_task": "...", "key_parameters": {{...}}}}"""
                
                messages = [
                    SystemMessage(content="You are a bioinformatics expert skilled at analyzing various biological data files."),
                    HumanMessage(content=prompt)
                ]
                
                response = llm.invoke(messages)
                result_text = response.content.strip()
                
                # Try to parse JSON
                try:
                    # Extract JSON part
                    json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group())
                        new_summary = result.get('summary', '')
                        # Only update if we got a non-empty summary
                        if new_summary:
                            analysis.content_summary = new_summary
                        if result.get('key_parameters'):
                            if analysis.suggested_parameters is None:
                                analysis.suggested_parameters = {}
                            analysis.suggested_parameters.update(result['key_parameters'])
                except json.JSONDecodeError:
                    if result_text:
                        analysis.content_summary = result_text[:200]
                    
            except Exception as e:
                print(f"  [WARN] LLM file analysis failed: {analysis.sandbox_path} - {e}")
        
        return file_analyses
        
    except Exception as e:
        print(f"  [WARN] LLM analysis failed: {e}")
        return file_analyses


# ---------------------- CSV → FASTA Conversion ----------------------
# Sequence column names that indicate antibody/nucleotide sequences
SEQUENCE_COLUMN_NAMES = [
    'heavy_dna', 'Heavy_DNA', 'light_dna', 'Light_DNA',
    'heavy', 'Heavy', 'light', 'Light',
    'sequence', 'Sequence', 'seq', 'Seq',
    'nt_sequence', 'NT_Sequence', 'aa_sequence', 'AA_Sequence',
    'cdr3', 'CDR3', 'cdr3_aa', 'CDR3_AA', 'cdr3_nt', 'CDR3_NT',
    'vh', 'VH', 'vl', 'VL', 'vhh', 'VHH',
    'antibody_sequence', 'Antibody_Sequence',
    # Additional common sequence column names (e.g., variant_seq_1, variant_seq_2, ...)
    'variant_seq', 'variant_seq_1', 'variant_seq_2', 'variant_seq_3',
    'heavy_chain', 'light_chain', 'HeavyChain', 'LightChain',
    'hc_seq', 'lc_seq', 'HC_seq', 'LC_seq',
    'full_sequence', 'dna_sequence', 'nucleotide_sequence',
]

# Regex patterns for dynamic column name matching (e.g., variant_seq_N)
SEQUENCE_COLUMN_PATTERNS = [
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


def _matches_sequence_column(col_name: str) -> bool:
    """Check if a column name matches known sequence column patterns."""
    import re
    col_lower = col_name.lower()
    # Check exact match first
    if col_name in SEQUENCE_COLUMN_NAMES or col_lower in [c.lower() for c in SEQUENCE_COLUMN_NAMES]:
        return True
    # Check regex patterns
    for pattern in SEQUENCE_COLUMN_PATTERNS:
        if re.match(pattern, col_lower, re.IGNORECASE):
            return True
    return False


def _generate_csv_to_fasta_code(
    csv_sandbox_path: str,
    fasta_output_path: str,
    sequence_columns: List[str],
    id_column: Optional[str] = None
) -> str:
    """
    Generate Python code to convert CSV to FASTA format in sandbox
    
    Args:
        csv_sandbox_path: CSV file path in sandbox
        fasta_output_path: Output FASTA file path
        sequence_columns: List of column names containing sequences
        id_column: Column to use as sequence ID (optional)
    """
    code = f'''
import pandas as pd
import os

csv_path = "{csv_sandbox_path}"
fasta_path = "{fasta_output_path}"
sequence_columns = {sequence_columns}
id_column = {repr(id_column)}

# Read CSV
df = pd.read_csv(csv_path)
print(f"Read CSV with {{len(df)}} rows", flush=True)

# Find available sequence columns
available_cols = [col for col in sequence_columns if col in df.columns]
if not available_cols:
    print(f"[WARN] No sequence columns found in CSV: {{list(df.columns)}}", flush=True)
else:
    print(f"Found sequence columns: {{available_cols}}", flush=True)
    
    # Determine ID column
    id_col = None
    if id_column and id_column in df.columns:
        id_col = id_column
    else:
        # Try common ID column names
        for candidate in ['main_name', 'name', 'id', 'ID', 'sample_id', 'cell_id', 'barcode']:
            if candidate in df.columns:
                id_col = candidate
                break
    
    if id_col:
        print(f"Using ID column: {{id_col}}", flush=True)
    else:
        print("No ID column found, using row index", flush=True)
    
    # Generate FASTA
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    
    seq_count = 0
    with open(fasta_path, 'w') as f:
        for idx, row in df.iterrows():
            for col in available_cols:
                seq = str(row[col]).strip()
                if seq and seq.lower() not in ['nan', 'none', '']:
                    # Create ID
                    if id_col:
                        seq_id = f"{{row[id_col]}}_{{col}}"
                    else:
                        seq_id = f"seq_{{idx}}_{{col}}"
                    
                    f.write(f">{{seq_id}}\\n{{seq}}\\n")
                    seq_count += 1
    
    print(f"[OK] Generated FASTA with {{seq_count}} sequences: {{fasta_path}}", flush=True)
    print(f"__FASTA_GENERATED__:{{fasta_path}}:{{seq_count}}")
'''
    return code


async def _convert_csv_to_fasta_if_needed(
    file_analyses: List[FileAnalysis],
    mcp_services: List[str],
    sandbox_id: Optional[str],
    sandbox_input_dir: str,
    sandbox_output_dir: str
) -> Tuple[List[FileAnalysis], Dict[str, str]]:
    """
    Check if CSV files contain sequence data and igblast is needed.
    If so, generate FASTA files from CSV.
    
    Returns:
        Updated file_analyses list and new sandbox_file_paths mappings
    """
    new_file_analyses = list(file_analyses)
    new_mappings = {}
    
    # Check if igblast is in required services
    needs_igblast = any(
        svc.lower() in ['igblast', 'vdj_analysis', 'analyze_vdj', 'analyze_vdj_batch']
        for svc in mcp_services
    )
    
    if not needs_igblast:
        return new_file_analyses, new_mappings
    
    print(f"  [FASTA] igblast detected, checking for sequence CSV files...")
    
    # Find CSV files with sequence columns
    for analysis in file_analyses:
        if analysis.file_type != 'csv':
            continue
        
        # Check if this CSV has sequence columns
        if not analysis.column_names:
            continue
        
        seq_cols = [col for col in analysis.column_names if _matches_sequence_column(col)]
        if not seq_cols:
            continue
        
        print(f"  [FASTA] Found sequence columns in {Path(analysis.original_path).name}: {seq_cols}")
        
        # Generate FASTA file path
        csv_name = Path(analysis.original_path).stem
        fasta_path = f"{sandbox_output_dir}/{csv_name}_sequences.fasta"
        
        # If we have OpenSandbox, generate FASTA in sandbox
        # Import here to avoid circular dependency
        try:
            from utils.opensandbox_executor import is_opensandbox_enabled, run_code_in_opensandbox_sync
            opensandbox_available = is_opensandbox_enabled()
        except ImportError:
            opensandbox_available = False
        
        if sandbox_id and opensandbox_available:
            try:
                code = _generate_csv_to_fasta_code(
                    csv_sandbox_path=analysis.sandbox_path,
                    fasta_output_path=fasta_path,
                    sequence_columns=seq_cols,
                    id_column=None  # Will auto-detect
                )
                
                print(f"  [FASTA] Executing CSV→FASTA conversion in sandbox {sandbox_id}...")
                result = await asyncio.to_thread(
                    run_code_in_opensandbox_sync,
                    code=code,
                    task_id=f"csv_to_fasta_{Path(analysis.original_path).stem}",
                    timeout_seconds=120,
                    env={"OPENSANDBOX_SKIP_MCP_INSTALL": "true"}
                )
                
                if result and "__FASTA_GENERATED__" in result.get("stdout", ""):
                    # Parse result to get sequence count
                    stdout = result.get("stdout", "")
                    for line in stdout.split("\n"):
                        if "__FASTA_GENERATED__:" in line:
                            parts = line.split(":")
                            if len(parts) >= 3:
                                fasta_path = parts[1]
                                seq_count = int(parts[2])
                                
                                # Add new FASTA file to analyses
                                fasta_analysis = FileAnalysis(
                                    original_path=f"generated_from:{analysis.original_path}",
                                    sandbox_path=fasta_path,
                                    file_type="fasta",
                                    file_size=0,
                                    row_count=seq_count,
                                    detected_data_type="antibody_fasta",
                                    content_summary=f"FASTA file generated from {Path(analysis.original_path).name} with {seq_count} sequences. Columns used: {seq_cols}",
                                    suggested_parameters={"fasta_file": fasta_path}
                                )
                                new_file_analyses.append(fasta_analysis)
                                new_mappings[f"generated_fasta:{analysis.original_path}"] = fasta_path
                                print(f"  [OK] Generated FASTA: {fasta_path} ({seq_count} sequences)")
                                break
            except Exception as e:
                print(f"  [WARN] Failed to generate FASTA from {analysis.original_path}: {e}")
        else:
            # Record that FASTA should be generated at execution time
            print(f"  [INFO] FASTA generation will be done at execution time for {analysis.original_path}")
            # Add placeholder entry to parameter table
            new_mappings[f"pending_fasta:{analysis.original_path}"] = {
                "source_csv": analysis.sandbox_path,
                "sequence_columns": seq_cols,
                "suggested_fasta_path": fasta_path
            }
    
    return new_file_analyses, new_mappings


def _get_compatible_param_types_for_file(file_ext: str, data_type: Optional[str] = None) -> List[str]:
    """
    Determine which parameter types this file can be used for.
    
    This is critical for automatic parameter matching during execution.
    The returned list must match parameter names defined in tools_params_table.json.
    
    Args:
        file_ext: File extension (without dot, e.g., 'csv', 'rds')
        data_type: Detected data type from file analysis (e.g., 'antibody_fasta')
        
    Returns:
        List of compatible parameter type names
    """
    compatible = []
    file_ext_lower = (file_ext or "").lower()
    data_type_lower = (data_type or "").lower()
    
    # CSV files can be used for various parameters
    if file_ext_lower == 'csv':
        compatible.extend(['csv_file', 'input_csv', 'data_file', 'test_file', 'input_file'])
        
        # Data type specific compatibility
        if 'airr' in data_type_lower or 'vdj' in data_type_lower:
            compatible.extend([
                'airr_file', 'vdj_results', 'annotation_file',
                'airr_results', 'airr_data', 'vdj_output'
            ])
        elif 'binding' in data_type_lower or 'prediction' in data_type_lower:
            compatible.extend(['binding_results', 'prediction_file'])
        elif 'tcr' in data_type_lower:
            compatible.extend(['tcr_predictions', 'prediction_file', 'binding_results'])
    
    # TSV files (AIRR format is often TSV)
    elif file_ext_lower == 'tsv':
        compatible.extend(['tsv_file', 'data_file', 'input_file'])
        if 'airr' in data_type_lower or 'vdj' in data_type_lower:
            compatible.extend(['airr_results', 'airr_file', 'airr_data'])
    
    # RDS files (R data objects, often Seurat objects)
    elif file_ext_lower == 'rds':
        compatible.extend(['rds_file', 'seurat_object', 'r_data', 'rds_path', 'input_rds', 'input_file'])
    
    # H5AD files (AnnData objects for single-cell)
    elif file_ext_lower == 'h5ad' or file_ext_lower == 'h5':
        compatible.extend(['h5ad_file', 'anndata_object', 'input_h5ad', 'input_file'])
    
    # FASTA files
    elif file_ext_lower in ['fasta', 'fa']:
        compatible.extend(['fasta_file', 'sequences', 'input_fasta', 'sequence_file'])
        if 'antibody' in data_type_lower:
            compatible.extend(['antibody_fasta', 'antibody_sequences'])
        elif 'tcr' in data_type_lower:
            compatible.extend(['tcr_fasta', 'tcr_sequences'])
    
    # JSON files
    elif file_ext_lower == 'json':
        compatible.extend(['json_file', 'config_file', 'metadata'])
    
    # Excel files
    elif file_ext_lower in ['xlsx', 'xls']:
        compatible.extend(['excel_file', 'xlsx_file', 'data_file', 'input_file'])
    
    return compatible


def _build_parameter_table(
    extracted_params: Dict[str, Any],
    file_analyses: List[FileAnalysis],
    sandbox_file_paths: Dict[str, str],
    fasta_mappings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build parameter table, integrating all extracted parameters and file information
    
    File entries use unique keys based on purpose/data_type + filename to avoid collision.
    
    IMPORTANT: Each file entry includes 'can_be_used_as' field that lists compatible
    parameter names. This is used by the executor for automatic parameter matching.
    
    IMPORTANT: Each entry is tagged with its source (user_input, task_output, etc.)
    to enable source-based constraint checking.
    """
    param_table = {
        "user_parameters": extracted_params,
        "files": {},
        "inferred_parameters": {},
    }
    
    # Track used keys to avoid collision
    used_keys = set()
    
    for analysis in file_analyses:
        # Get file extension
        file_ext = analysis.file_type or Path(analysis.sandbox_path).suffix.lstrip('.').lower()
        
        # CRITICAL: Determine which parameters this file can be used for
        can_be_used_as = _get_compatible_param_types_for_file(file_ext, analysis.detected_data_type)
        
        file_info = {
            "original_path": analysis.original_path,
            "sandbox_path": analysis.sandbox_path,
            "type": analysis.file_type,
            "data_type": analysis.detected_data_type,
            "row_count": analysis.row_count,
            "columns": analysis.column_names,
            "summary": analysis.content_summary,
            "can_be_used_as": can_be_used_as,  # CRITICAL: enables automatic parameter matching
            "source": "user_input",  # Tag: this file was provided by user
            "source_tool": "preprocess",  # Mark as from preprocessing, not from a tool
        }
        
        # Generate unique key: prefer detected_data_type, fallback to filename
        base_key = analysis.detected_data_type or Path(analysis.original_path).stem
        base_key = base_key.replace(' ', '_').lower()
        
        # Ensure uniqueness
        key = base_key
        counter = 1
        while key in used_keys:
            key = f"{base_key}_{counter}"
            counter += 1
        used_keys.add(key)
        
        param_table["files"][key] = file_info
        
        # Merge inferred parameters
        if analysis.suggested_parameters:
            param_table["inferred_parameters"].update(analysis.suggested_parameters)
    
    # Add file path mappings
    param_table["sandbox_file_paths"] = sandbox_file_paths
    
    # Add FASTA mappings if available (for igblast integration)
    if fasta_mappings:
        param_table["generated_fasta_files"] = {}
        param_table["pending_fasta_conversions"] = {}
        for key, value in fasta_mappings.items():
            if key.startswith("generated_fasta:"):
                param_table["generated_fasta_files"][key] = value
            elif key.startswith("pending_fasta:"):
                param_table["pending_fasta_conversions"][key] = value
    
    return param_table


def _clean_user_input(user_input: str, file_paths: List[str]) -> str:
    """Clean user input, remove file paths, etc."""
    cleaned = user_input
    for path in file_paths:
        cleaned = cleaned.replace(path, "[FILE]")
    return cleaned.strip()


# ---------------------- LLM Instantiation (using common LLM factory) ----------------------
def _get_llm():
    """
    Get reasoning model instance (for task classification)
    
    Use the common LLM factory to create a reasoning model, prioritizing models with good reasoning performance.
    
    Returns:
        LLM instance, returns None if all are unavailable
    """
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        return None
    
    # Use reasoning model (for task classification)
    return create_reasoning_llm(temperature=0.1)


# ---------------------- LLM Structured Extraction Functions ----------------------
def _llm_extract_structured_input(user_input: str) -> LLMExtractionResult:
    """
    Extract structured information from user input using LLM
    
    Args:
        user_input: User's original input text
        
    Returns:
        LLMExtractionResult: Structured extraction result
    """
    llm = _get_llm()
    
    if not llm:
        print("  [WARN] LLM not available, using regex fallback")
        return _fallback_regex_extraction(user_input)
    
    try:
        messages = [
            SystemMessage(content=PARAMETER_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=PARAMETER_EXTRACTION_USER_PROMPT.format(user_input=user_input))
        ]
        
        print("  [LLM] Using LLM to extract structured information...")
        response = llm.invoke(messages)
        result_text = response.content.strip()
        
        # Extract JSON part
        json_str = _extract_json_from_response(result_text)
        if json_str:
            try:
                result_dict = json.loads(json_str)
                result = LLMExtractionResult.model_validate(result_dict)
                print(f"  [OK] LLM extraction successful: {len(result.files)} files, {len(result.parameters)} params, {len(result.mcp_services)} services")
                return result
            except (json.JSONDecodeError, Exception) as e:
                print(f"  [WARN] JSON parsing failed: {e}")
        
        print("  [WARN] Cannot parse LLM response, using regex fallback")
        return _fallback_regex_extraction(user_input)
        
    except Exception as e:
        print(f"  [WARN] LLM extraction failed: {e}")
        return _fallback_regex_extraction(user_input)


def _extract_json_from_response(text: str) -> Optional[str]:
    """Extract JSON string from LLM response"""
    # Try direct parsing
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    
    # Try to extract from markdown code block
    json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()
    
    # Try to find JSON object
    brace_start = text.find('{')
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[brace_start:i+1]
    
    return None


def _fallback_regex_extraction(user_input: str) -> LLMExtractionResult:
    """Regex fallback extraction"""
    result = LLMExtractionResult(task_description="")
    
    # Extract file paths
    detected_files = _detect_file_paths(user_input)
    for f in detected_files:
        ext = Path(f.path).suffix.lower().lstrip('.') or 'unknown'
        result.files.append(ExtractedFile(
            path=f.path,
            purpose="unknown",
            format=ext,
            source=f.source_type.value
        ))
    
    # Extract MCP services
    service_pattern = r'(?:use|using|mcp\s*services?)[:\s]*[-\s]*(\w+)'
    for match in re.finditer(service_pattern, user_input, re.IGNORECASE):
        service = match.group(1).lower()
        if service not in ['the', 'following', 'these']:
            result.mcp_services.append(service)
    
    # Extract services from list
    list_pattern = r'^\s*-\s*(\w+)\s*$'
    for match in re.finditer(list_pattern, user_input, re.MULTILINE):
        item = match.group(1).lower()
        if item in ['igblast', 'metabcr', 'r_data_integration', 'bioinformatics', 'antibody_annotator']:
            if item not in result.mcp_services:
                result.mcp_services.append(item)
    
    # Extract target organism
    organism_pattern = r'\b(H5N1|H1N1|SARS-CoV-2|COVID|flu|influenza)\b'
    organism_match = re.search(organism_pattern, user_input, re.IGNORECASE)
    if organism_match:
        result.target_organism = organism_match.group(1)
    
    return result


def _convert_llm_result_to_params(llm_result: LLMExtractionResult) -> Dict[str, Any]:
    """Convert LLM extraction result to parameter dictionary
    
    Note: File information is stored separately in the 'files' field of parameter_table,
    not duplicated in the main params to avoid duplicate keys with same values.
    """
    params = {}
    
    # Add MCP services
    if llm_result.mcp_services:
        params['mcp_services'] = llm_result.mcp_services
    
    # Add target organism
    if llm_result.target_organism:
        params['target_organism'] = llm_result.target_organism
    
    # Add task description
    if llm_result.task_description:
        params['task_description'] = llm_result.task_description
    
    # Add analysis type
    if llm_result.analysis_type and llm_result.analysis_type != 'other':
        params['analysis_type'] = llm_result.analysis_type
    
    # Add extracted parameters (exclude file paths to avoid duplication)
    # Collect all file paths from llm_result.files for deduplication check
    file_paths = {f.path for f in llm_result.files}
    
    for param in llm_result.parameters:
        # Skip if the parameter value is a file path (already in files list)
        if isinstance(param.value, str) and param.value in file_paths:
            continue
        params[param.name] = param.value
    
    # Add notes
    if llm_result.notes:
        params['notes'] = llm_result.notes
    
    return params


def _convert_llm_files_to_detected(llm_result: LLMExtractionResult) -> List[DetectedFile]:
    """Convert LLM extracted file info to DetectedFile list"""
    detected_files = []
    
    for f in llm_result.files:
        # Map source string to FileSourceType
        if f.source == 'url':
            source_type = FileSourceType.URL
        elif f.source == 'local':
            source_type = FileSourceType.LOCAL
        else:
            source_type = FileSourceType.REMOTE
        
        detected_files.append(DetectedFile(
            path=f.path,
            source_type=source_type,
            suggested_name=Path(f.path).name if f.path else None
        ))
    
    return detected_files


def _find_llm_file_info(file_path: str, llm_result: LLMExtractionResult) -> Optional[Dict[str, str]]:
    """Find matching file info from LLM extraction result"""
    if not llm_result or not llm_result.files:
        return None
    
    for f in llm_result.files:
        if f.path == file_path:
            return {
                'purpose': f.purpose,
                'format': f.format,
                'source': f.source,
            }
    
    # Fuzzy match by filename
    file_name = Path(file_path).name.lower()
    for f in llm_result.files:
        if Path(f.path).name.lower() == file_name:
            return {
                'purpose': f.purpose,
                'format': f.format,
                'source': f.source,
            }
    
    return None


# ---------------------- Node 1: User Description Classification Node ----------------------
def user_description_classify_node(state: SupervisorState) -> SupervisorState:
    """
    User description classification node:
    1. Based on user input, determine which type the task belongs to: 【General Q&A】, 【Execute Given Plan】, or 【Immunology-Related Task】
    2. If user uploaded files, download them to the agreed sandbox directory
    """
    user_input = state.user_input
    
    # 1. Determine task type
    task_type = _classify_user_task_type(user_input)
    state.user_task_type = task_type
    
    # 2. Check and process uploaded files
    # Skip if files were already processed by preprocess_user_input_node
    if state.uploaded_files and not state.sandbox_file_paths and not state.file_analyses:
        # Ensure sandbox directory exists
        sandbox_path = Path(state.sandbox_dir)
        sandbox_path.mkdir(parents=True, exist_ok=True)
        
        # Download/copy files to sandbox directory (only for local files that exist)
        for uploaded_file_path in state.uploaded_files:
            # Skip remote paths (they are handled by OpenSandbox)
            if uploaded_file_path.startswith(('/data/', '/home/', '/opt/', '/mnt/')):
                continue
            sandbox_file_path = _download_file_to_sandbox(
                uploaded_file_path, 
                sandbox_path
            )
            if sandbox_file_path:
                state.sandbox_file_paths[uploaded_file_path] = sandbox_file_path
    
    return state


def _classify_user_task_type_with_llm(user_input: str, llm) -> Optional[UserTaskType]:
    """
    Use LLM to classify task type based on user input
    
    Args:
        user_input: User input
        llm: LLM instance
    
    Returns:
        Task type, returns None if classification fails
    """
    # Use centralized prompt templates
    system_prompt = TASK_CLASSIFICATION_SYSTEM_PROMPT
    user_prompt = get_task_classification_user_prompt(user_input)

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        result_text = response.content.strip()
        
        # Extract task type (allow partial matching for robustness)
        result_lower = result_text.lower()
        if "plan" in result_lower or "execute" in result_lower:
            return UserTaskType.EXECUTE_PLAN
        elif "immun" in result_lower or "antigen" in result_lower or "antibody" in result_lower:
            return UserTaskType.IMMUNOLOGY_TASK
        elif "general" in result_lower or "qa" in result_lower or "q&a" in result_lower:
            return UserTaskType.GENERAL_QA
        else:
            # If cannot parse, try fuzzy matching
            if any(word in result_lower for word in ["plan", "step", "execute", "instruction"]):
                return UserTaskType.EXECUTE_PLAN
            elif any(word in result_lower for word in ["immun", "antigen", "antibody", "vaccine"]):
                return UserTaskType.IMMUNOLOGY_TASK
            else:
                return UserTaskType.GENERAL_QA
                
    except Exception as e:
        # Check if it's an authentication error (API Key error)
        error_str = str(e).lower()
        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            print(f"[WARN] LLM API Key authentication failed, will use keyword matching as fallback: {type(e).__name__}")
            print(f"  Tip: Please check if the API Key in environment variables is correctly configured")
        elif "rate limit" in error_str or "429" in error_str:
            print(f"[WARN] LLM API rate limit exceeded, will use keyword matching as fallback: {type(e).__name__}")
        else:
            print(f"[WARN] LLM task type classification failed, will use keyword matching as fallback: {type(e).__name__}: {str(e)[:100]}")
        
        return None

def _classify_user_task_type(user_input: str) -> UserTaskType:
    """
    Classify task type based on user input (prioritize LLM, fallback to keyword matching on failure)
    
    Args:
        user_input: User input
    
    Returns:
        Task type
    """
    # Try using LLM classification (using common LLM factory)
    llm = _get_llm()
    if llm is not None:
        result = _classify_user_task_type_with_llm(user_input, llm)
        if result is not None:
            print(f"LLM classified task type: {result.value}")
            return result
    
    # When LLM is unavailable or fails, use keyword matching as fallback
    print("Using keyword matching as fallback")
    user_input_lower = user_input.lower()
    
    # Check execution plan related keywords
    if any(keyword in user_input_lower for keyword in [
        "execute", "plan", "step", "follow", "according to", "instruction",
        "执行", "计划", "步骤", "按照", "依据", "流程"
    ]):
        return UserTaskType.EXECUTE_PLAN
    
    # Check immunology related keywords
    if any(keyword in user_input_lower for keyword in [
        "immun", "antigen", "antibody", "vaccine", "immune system", "immune cell", "t cell", "b cell", "immune response",
        "免疫", "抗原", "抗体", "疫苗", "免疫系统", "免疫细胞", "t细胞", "b细胞", "免疫反应"
    ]):
        return UserTaskType.IMMUNOLOGY_TASK
    
    # Default to general Q&A
    return UserTaskType.GENERAL_QA


def _download_file_to_sandbox(source_file_path: str, sandbox_dir: Path) -> Optional[str]:
    """
    Download/copy file to sandbox directory
    
    Args:
        source_file_path: Source file path (may be URL or local path)
        sandbox_dir: Sandbox directory path
    
    Returns:
        File path in sandbox, returns None if failed
    """
    try:
        source_path = Path(source_file_path)
        
        # If it's a URL, need to download (simplified handling here, may need requests library in practice)
        if source_file_path.startswith(("http://", "https://")):
            # TODO: Implement HTTP file download logic
            # Return None for now, can be extended later
            print(f"Warning: URL file download functionality not yet implemented: {source_file_path}")
            return None
        
        # If it's a local file, copy to sandbox directory
        if source_path.exists() and source_path.is_file():
            # Generate target file path (preserve filename)
            target_file_path = sandbox_dir / source_path.name
            
            # If target file exists, add numeric suffix to avoid overwriting
            counter = 1
            while target_file_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                target_file_path = sandbox_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy file
            shutil.copy2(source_path, target_file_path)
            print(f"File copied to sandbox: {source_path} -> {target_file_path}")
            return str(target_file_path)
        else:
            print(f"Warning: Source file does not exist: {source_file_path}")
            return None
            
    except Exception as e:
        print(f"Error: Failed to copy file to sandbox {source_file_path}: {str(e)}")
        return None


# ---------------------- State Mapping Functions =====================
def supervisor_input_mapper(global_state: GlobalState) -> SupervisorState:
    """
    Main graph → subgraph state mapping
    
    Map the main graph's GlobalState to SupervisorState, extracting information needed by the subgraph.
    
    Args:
        global_state: Main graph's global state
    
    Returns:
        SupervisorState: Subgraph state
    """
    uploaded_files = list(global_state.file_paths.keys()) if global_state.file_paths else []
    
    # Get existing file analyses from global state (if preprocessing was already done)
    existing_file_analyses = []
    if global_state.file_analyses:
        for fa in global_state.file_analyses:
            if isinstance(fa, dict):
                existing_file_analyses.append(FileAnalysis(**fa))
            elif isinstance(fa, FileAnalysis):
                existing_file_analyses.append(fa)
    
    return SupervisorState(
        user_input=global_state.user_input,
        user_task_type=None,  # Will be determined in subgraph
        uploaded_files=uploaded_files,
        sandbox_file_paths=dict(global_state.file_paths) if global_state.file_paths else {},
        sandbox_dir=global_state.sandbox_dir,
        execution_plan=global_state.execution_plan,
        # Pass through session-related fields to avoid re-creating session
        session_id=global_state.session_id,
        sandbox_data_dir=global_state.sandbox_data_dir,
        opensandbox_id=global_state.opensandbox_id,
        extracted_parameters=global_state.extracted_parameters,
        file_analyses=existing_file_analyses,
    )


def supervisor_output_mapper(subgraph_output: SupervisorState | dict, global_state: GlobalState) -> GlobalState:
    """
    Subgraph → main graph state mapping
    
    Synchronize the subgraph's SupervisorState results back to the main graph's GlobalState.
    
    Args:
        subgraph_output: Subgraph output state (may be SupervisorState object or dict)
        global_state: Main graph's global state (will be updated)
    
    Returns:
        GlobalState: Updated main graph state
    """
    
    # Handle dict format state (LangGraph may return dict)
    if isinstance(subgraph_output, dict):
        subgraph_output = SupervisorState(**subgraph_output)
    
    # Store task type classification result to user_task_type
    # 确保 user_task_type 是枚举类型而不是字符串
    if subgraph_output.user_task_type:
        task_type = subgraph_output.user_task_type
        # 如果是字符串，转换为枚举类型
        if isinstance(task_type, str):
            try:
                task_type = UserTaskType(task_type)
            except (ValueError, KeyError):
                # 如果转换失败，保持原值（可能是 None 或其他值）
                pass
        global_state.user_task_type = task_type
    
    # Synchronize execution plan (if execution plan was determined)
    if subgraph_output.execution_plan:
        global_state.execution_plan = subgraph_output.execution_plan
    
    # Synchronize sandbox file paths (if needed, can store to merged_result)
    if subgraph_output.sandbox_file_paths:
        global_state.file_paths = subgraph_output.sandbox_file_paths
    
    # Sync preprocessing result (parameter table, file analysis)
    if subgraph_output.extracted_parameters:
        # Store parameter table in merged_result for subsequent processes
        if global_state.merged_result is None:
            global_state.merged_result = {}
        global_state.merged_result["extracted_parameters"] = subgraph_output.extracted_parameters
        global_state.merged_result["file_analyses"] = [
            {
                "sandbox_path": fa.sandbox_path,
                "file_type": fa.file_type,
                "data_type": fa.detected_data_type,
                "columns": fa.column_names,
                "summary": fa.content_summary,
            }
            for fa in subgraph_output.file_analyses
        ] if subgraph_output.file_analyses else []
    
    # Sync session ID, OpenSandbox ID and data directory (for reuse in subsequent processes)
    if global_state.merged_result is None:
        global_state.merged_result = {}
    
    if subgraph_output.session_id:
        global_state.merged_result["session_id"] = subgraph_output.session_id
    
    if subgraph_output.opensandbox_id:
        global_state.merged_result["opensandbox_id"] = subgraph_output.opensandbox_id
    
    if subgraph_output.sandbox_data_dir:
        global_state.merged_result["sandbox_data_dir"] = subgraph_output.sandbox_data_dir
        global_state.merged_result["sandbox_input_dir"] = f"{subgraph_output.sandbox_data_dir}/input"
        global_state.merged_result["sandbox_output_dir"] = f"{subgraph_output.sandbox_data_dir}/output"
    
    # Return updated global state
    return global_state


# ---------------------- Build Supervisor Agent Subgraph ----------------------
def build_supervisor_subgraph():
    """
    Build Supervisor Agent subgraph
    
    Flow:
    1. preprocess_user_input - Input preprocessing (parameter extraction, file transfer, LLM analysis)
    2. classify_user_description - Task type classification
    
    Use common LLM factory to create LLM instance, prioritizing models with good reasoning performance.
    
    Returns:
        Compiled subgraph
    """
    graph = StateGraph(SupervisorState)
    
    # Node 1: Input preprocessing
    graph.add_node("preprocess_user_input", preprocess_user_input_node)
    
    # Node 2: Task classification
    graph.add_node("classify_user_description", user_description_classify_node)
    
    # Flow: START -> preprocess_user_input -> classify_user_description -> END
    graph.add_edge(START, "preprocess_user_input")
    graph.add_edge("preprocess_user_input", "classify_user_description")
    graph.add_edge("classify_user_description", END)
    
    return graph.compile()
