"""
Immunity Agent Subgraph

Complete workflow:
Query Decomposition → Retrieval → Deep Research → Hypothesis Generation → Planning ⭐ → Evaluation

Reference implementation: antibody_gen/agent/usecases/immunity
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import re
import os
import time
import asyncio
import sys
import concurrent.futures
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from state import GlobalState
from utils.llm_factory import (
    create_reasoning_advanced_llm,
    create_reasoning_llm,
    create_bioinformatics_llm,
)
from .state import ImmunityState
from .prompts import ImmunityPrompts


def get_progress_callback_by_session(session_id: Optional[str]) -> Optional[Any]:
    """
    Get progress callback from global registry by session_id

    Args:
        session_id: Session ID to look up

    Returns:
        Progress callback function if found, None otherwise
    """
    if not session_id:
        return None

    try:
        backend_dir = Path(__file__).parent.parent.parent.parent / "backend"
        project_root = backend_dir.parent

        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from backend import progress_tracker as pt_module

        callback = pt_module.get_progress_callback(session_id)
        print(
            f"[Immunity] Got callback for session {session_id}: {callback is not None}"
        )
        return callback
    except (ImportError, AttributeError) as e:
        print(f"[Immunity] Failed to get callback: {e}")
        return None


# Mem0 Memory Management
try:
    from utils.mem0_manager import (
        get_memory_client,
        check_immunity_cache_sync,
        generate_input_hash,
    )

    MEM0_AVAILABLE = True
except ImportError as e:
    MEM0_AVAILABLE = False
    print(f"[Immunity] Warning: mem0_manager unavailable: {e}")

# Import search tools for LLM binding
from tools.search import web_search, knowledge_search, read_webpage

# Import deep_research subgraph
from nodes.subagents.deep_research.deep_researcher import (
    deep_researcher,
    get_default_config as get_deep_research_config,
)

# Add agent directory to path
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))


# ===================== Progress Logging Utilities =====================


class ImmunityProgressLogger:
    """
    Progress logger - provides detailed progress output and time statistics
    """

    # Stage definitions
    STAGES = [
        ("query_decomposition", "📝 Query Decomposition", 1),
        ("retrieval", "📚 Immunology Retrieval", 2),
        ("deep_research", "[Deep Research] Deep Research Analysis", 3),
        ("hypothesis_generation", "🧬 Hypothesis Generation", 4),
        ("planning", "[TOOL] Plan Generation", 5),
        ("evaluation", "[STAT] Plan Evaluation", 6),
    ]
    TOTAL_STAGES = 6

    def __init__(self):
        self.stage_start_times: Dict[str, float] = {}
        self.workflow_start_time: float = 0
        self.current_stage: str = ""

    def _timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%H:%M:%S")

    def _flush_print(self, msg: str):
        """Force flush output"""
        print(msg)
        sys.stdout.flush()

    def start_workflow(self):
        """Start workflow"""
        self.workflow_start_time = time.perf_counter()
        self._flush_print("\n" + "=" * 80)
        self._flush_print(f"[{self._timestamp()}] [START] IMMUNITY SUBGRAPH workflow started")
        self._flush_print("=" * 80)

    def end_workflow(self, success: bool = True):
        """End workflow"""
        total_time = time.perf_counter() - self.workflow_start_time
        status = "[SUCCESS] completed successfully" if success else "[ERROR] Failed"
        self._flush_print("\n" + "=" * 80)
        self._flush_print(f"[{self._timestamp()}] {status}")
        self._flush_print(f"  Total time: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
        self._flush_print("=" * 80)

    def start_stage(self, stage_name: str, description: str = ""):
        """Start a stage"""
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = time.perf_counter()

        # Find stage number
        stage_num = 0
        for name, _, num in self.STAGES:
            if name == stage_name:
                stage_num = num
                break

        progress_pct = (stage_num / self.TOTAL_STAGES) * 100

        self._flush_print("\n" + "-" * 80)
        self._flush_print(
            f"[{self._timestamp()}] [RUN] STAGE {stage_num}/{self.TOTAL_STAGES} ({progress_pct:.0f}%): {stage_name}"
        )
        if description:
            self._flush_print(f"  [INFO] {description}")
        self._flush_print("-" * 80)

    def end_stage(self, stage_name: str, success: bool = True, details: str = ""):
        """End a stage"""
        elapsed = time.perf_counter() - self.stage_start_times.get(
            stage_name, time.perf_counter()
        )

        # Find stage number
        stage_num = 0
        for name, _, num in self.STAGES:
            if name == stage_name:
                stage_num = num
                break

        progress_pct = (stage_num / self.TOTAL_STAGES) * 100
        status = "[SUCCESS]" if success else "[ERROR]"

        self._flush_print(
            f"[{self._timestamp()}] {status} STAGE {stage_num}/{self.TOTAL_STAGES} completed ({progress_pct:.0f}%)"
        )
        self._flush_print(f"  ⏱️ Stage time: {elapsed:.2f} seconds")
        if details:
            self._flush_print(f"  [STAT] {details}")

    def log_llm_start(self, model_info: dict, prompt_len: int):
        """Log LLM call start"""
        self._flush_print(f"[{self._timestamp()}] 🤖 Starting LLM call...")
        self._flush_print(f"  📦 Model: {model_info.get('model', 'unknown')}")
        self._flush_print(f"  🌡️ Temperature: {model_info.get('temperature', 'N/A')}")
        self._flush_print(f"  ⏰ Timeout: {model_info.get('timeout', 'N/A')}s")
        self._flush_print(f"  📝 Prompt length: {prompt_len} characters")
        self._flush_print(f"  ⏳ Waiting for response...")

    def log_llm_end(self, elapsed: float, response_len: int = 0, success: bool = True):
        """Log LLM call end"""
        status = "[SUCCESS]" if success else "[ERROR]"
        self._flush_print(f"[{self._timestamp()}] {status} LLM call completed")
        self._flush_print(f"  ⏱️ Response time: {elapsed:.2f} seconds")
        if response_len > 0:
            self._flush_print(f"  📤 Response length: {response_len} characters")

    def log_info(self, message: str):
        """Log info message"""
        self._flush_print(f"[{self._timestamp()}] ℹ️ {message}")

    def log_warning(self, message: str):
        """Log warning"""
        self._flush_print(f"[{self._timestamp()}] [WARN]️ {message}")

    def log_error(self, message: str, error: Exception = None):
        """Log error"""
        self._flush_print(f"[{self._timestamp()}] [ERROR] {message}")
        if error:
            self._flush_print(f"  Error details: {type(error).__name__}: {str(error)}")


# Global progress logger instance
_progress_logger = ImmunityProgressLogger()


# ===================== Helper Functions =====================


def _get_llm_with_callback(state, purpose="bioinformatics"):
    """
    Create LLM instance with progress_callback

    Prefer state.get_llm() method to ensure SSE push works correctly.

    Args:
        state: ImmunityState instance
        purpose: LLM purpose ("bioinformatics", "reasoning", "code", etc.)

    Returns:
         LLM instance (with or without progress_callback)
    """
    # [DEBUG] Check if session_id is available
    has_session_id = hasattr(state, "session_id") and state.session_id
    has_parent_state = hasattr(state, "parent_state") and state.parent_state
    has_get_llm = hasattr(state, "get_llm") and callable(
        getattr(state, "get_llm", None)
    )

    print(f"[Immunity] _get_llm_with_callback state check:")
    print(f"  - has_session_id: {has_session_id}")
    print(f"  - has_parent_state: {has_parent_state}")
    print(f"  - has_get_llm: {has_get_llm}")

    # [HOT] Prefer state.get_llm() method
    if has_get_llm:
        print(f"[Immunity] Using state.get_llm() method")
        return state.get_llm(purpose=purpose, node_name="immunity")

    # [FALLBACK] If no get_llm method, get callback via session_id and create LLM
    progress_callback = None
    session_id = None
    if has_session_id:
        session_id = state.session_id
    elif has_parent_state:
        session_id = getattr(state.parent_state, "session_id", None)
        print(f"[Immunity] Retrieved from parent_state: session_id={session_id}")

    # Get callback from global registry via session_id
    if session_id:
        try:
            import sys
            from pathlib import Path

            backend_dir = Path(__file__).parent.parent.parent.parent / "backend"
            project_root = backend_dir.parent

            if str(backend_dir) not in sys.path:
                sys.path.insert(0, str(backend_dir))
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from backend import progress_tracker as pt_module

            progress_callback = pt_module.get_progress_callback(session_id)
            print(
                f"[Immunity] Got callback from global registry: {progress_callback is not None}"
            )
        except (ImportError, AttributeError) as e:
            print(f"[Immunity] Failed to get callback: {e}")

    # Create LLM instance with SSE push
    if progress_callback or session_id:
        from utils.llm_factory import create_llm_with_thinking

        print(
            f"[Immunity] Creating LLM with thinking: progress_callback={progress_callback is not None}, session_id={session_id}"
        )
        return create_llm_with_thinking(
            purpose=purpose,
            progress_callback=progress_callback,
            session_id=session_id,
            node_name="immunity",
        )
    else:
        # Fallback to normal creation
        print(f"[Immunity] [WARN] No progress_callback or session_id, using normal LLM")
        if purpose == "bioinformatics":
            return create_bioinformatics_llm()
        elif purpose == "reasoning":
            return create_reasoning_llm()
        else:
            return create_bioinformatics_llm()


def _load_tools_json() -> str:
    """
    Load tool information (JSON format)

    Returns:
        JSON string of tool information
    """
    mcp_tools_path = agent_dir / "config" / "mcp_tools.json"

    try:
        if mcp_tools_path.exists():
            with open(mcp_tools_path, "r", encoding="utf-8") as f:
                tools_data = json.load(f)
                return json.dumps(tools_data, ensure_ascii=False, indent=2)
        else:
            print(f"[WARN]️ mcp_tools.json does not exist: {mcp_tools_path}")
            return "[]"
    except Exception as e:
        print(f"[WARN]️ Failed to load tool information: {e}")
        return "[]"


def _get_opensandbox_id(state: "ImmunityState") -> Optional[str]:
    """
    Get opensandbox_id from state or parent_state

    Architecture principle: all subgraphs get opensandbox_id via parent_state.merged_result
    to reuse the sandbox instance created by supervisor
    """
    # 1. Get from parent_state.merged_result (preferred)
    if state.parent_state:
        merged_result = getattr(state.parent_state, "merged_result", None) or {}
        if isinstance(merged_result, dict):
            opensandbox_id = merged_result.get("opensandbox_id")
            if opensandbox_id:
                print(f"[Immunity] Got opensandbox_id: {opensandbox_id}")
                return opensandbox_id

    # 2. Get directly from parent_state
    if state.parent_state:
        opensandbox_id = getattr(state.parent_state, "opensandbox_id", None)
        if opensandbox_id:
            print(f"[Immunity] Got opensandbox_id from parent_state: {opensandbox_id}")
            return opensandbox_id

    # 3. Get from state itself (if available)
    opensandbox_id = getattr(state, "opensandbox_id", None)
    if opensandbox_id:
        print(f"[Immunity] Got opensandbox_id from state: {opensandbox_id}")
        return opensandbox_id

    print("[Immunity] [WARN]️ opensandbox_id not found, will create new sandbox")
    return None


def _save_report(
    content: str,
    report_type: str,
    sandbox_dir: Optional[str] = None,
    local_sandbox_dir: Optional[str] = None,
    opensandbox_id: Optional[str] = None,
    progress_callback: Optional[callable] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Save report to file in sandbox output directory

    Report save strategy (architecture principle: unified execution via CodeAct):
    1. Remote sandbox: save via CodeAct to {sandbox_dir}/output/reports/
    2. Local fallback: save to {local_sandbox_dir}/output/reports/
    3. [HOT] Also push file content to frontend via SSE

    Args:
        content: Report content
        report_type: Report type (retrieval, deep_research, hypothesis, planning, evaluation)
        sandbox_dir: Sandbox directory (sandbox_data_dir like /data/sessions/{session_id})
        local_sandbox_dir: Local sandbox directory (fallback for local testing)
        opensandbox_id: OpenSandbox instance ID to reuse (IMPORTANT for session continuity)
        progress_callback: SSE progress callback for pushing file content to frontend
        session_id: Session ID for logging

    Returns:
        Saved file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report_type}_{timestamp}.md"

    print(f"[Immunity] _save_report called:")
    print(f"  - report_type: {report_type}")
    print(f"  - sandbox_dir: {sandbox_dir}")
    print(f"  - local_sandbox_dir: {local_sandbox_dir}")
    print(f"  - opensandbox_id: {opensandbox_id}")

    # List of save paths to try (sorted by priority)
    save_paths = []

    # 1. Sandbox path: /data/sessions/{session_id}/output/reports/
    #    Save to remote sandbox via CodeAct (architecture principle: sole entry point to OpenSandbox)
    if sandbox_dir:
        sandbox_dir_normalized = str(sandbox_dir).replace("\\", "/")
        is_unix_path = sandbox_dir_normalized.startswith("/")
        remote_path = f"{sandbox_dir_normalized.rstrip('/')}/output/reports/{filename}"

        # Remote sandbox path, must be saved via CodeAct
        if is_unix_path:
            try:
                # Save file via CodeAct unified interface (architecture principle: sole entry point to OpenSandbox)
                from utils.codeact_executor import (
                    execute_code_via_codeact,
                    is_codeact_available,
                )

                if is_codeact_available():
                    print(f"[Immunity] CodeAct available, preparing to save report to remote sandbox...")

                    # Escape special characters in content (including newlines)
                    escaped_content = (
                        content.replace("\\", "\\\\")
                        .replace('"""', '\\"\\"\\"')
                        .replace("'''", "\\'\\'\\'")
                    )

                    # Container path
                    container_path = remote_path.replace(
                        "/data/sessions/", "/data/sessions/", 1
                    )

                    # Use safer code template (avoid triple-quote issues)
                    import base64

                    content_b64 = base64.b64encode(content.encode("utf-8")).decode(
                        "ascii"
                    )

                    save_code = f'''
import os
import base64

# Base64 encoded content
content_b64 = "{content_b64}"
file_path = "{container_path}"

try:
    # Decode content
    content = base64.b64decode(content_b64).decode('utf-8')
    
    # Create directory
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"__REPORT_SAVED__:{{file_path}}")
    print(f"__CONTENT_LENGTH__:{{len(content)}}")
except Exception as e:
    print(f"__REPORT_ERROR__:{{str(e)}}")
'''

                    print(f"[Immunity] Calling CodeAct to save report...")
                    print(f"  - container_path: {container_path}")
                    print(f"  - opensandbox_id: {opensandbox_id}")

                    result = execute_code_via_codeact(
                        task_description=f"Save {report_type} report to remote sandbox",
                        code_template=save_code,
                        sandbox_id=opensandbox_id,  # Pass sandbox_id to reuse sandbox
                        timeout_seconds=60,
                        keep_alive=True,
                    )

                    print(f"[Immunity] CodeAct execution result:")
                    print(f"  - status: {result.status}")
                    print(
                        f"  - output: {result.output[:200] if result.output else 'N/A'}..."
                    )
                    print(f"  - error: {result.error}")
                    print(f"  - sandbox_id: {result.sandbox_id}")

                    # Modified logic: consider success if output contains __REPORT_SAVED__
                    # No longer require result.is_success(), as sandbox connection may have non-fatal errors
                    if result.output and "__REPORT_SAVED__:" in result.output:
                        print(
                            f"📄 {report_type} report saved to remote sandbox via CodeAct: {remote_path}"
                        )
                        if progress_callback:
                            try:
                                report_type_display = {
                                    "retrieval": "Retrieval Report",
                                    "deep_research": "Deep Research Report",
                                    "hypothesis": "Hypothesis Report",
                                    "planning": "Experimental Plan",
                                    "evaluation": "Evaluation Report",
                                }.get(report_type, report_type)
                                progress_callback(
                                    event_type="file_content",
                                    message=f"📄 {report_type_display}saved to sandbox",
                                    details={
                                        "file_type": report_type,
                                        "file_name": filename,
                                        "file_path": remote_path,
                                        "content": content,
                                        "content_length": len(content),
                                        "node": "immunity",
                                        "session_id": session_id,
                                    },
                                )
                                print(
                                    f"  [SUCCESS] Pushed {report_type} file content to frontend ({len(content)} characters)"
                                )
                            except Exception as e:
                                print(f"  [WARN]️ Failed to push file content to frontend: {e}")
                        return remote_path
                    else:
                        print(
                            f"[WARN]️ Failed to save to remote sandbox via CodeAct: {result.error}"
                        )
                        print(f"ℹ️ Falling back to local save")
                else:
                    print(f"ℹ️ CodeAct not available, falling back to local save")

            except Exception as e:
                print(f"[WARN]️ Failed to save to remote sandbox via CodeAct: {e}")
                import traceback

                traceback.print_exc()
                print(f"ℹ️ Falling back to local save")

        # If local Unix path, save directly
        if is_unix_path and os.name != "nt":
            save_paths.append(
                (
                    "sandbox",
                    Path(sandbox_dir_normalized) / "output" / "reports" / filename,
                )
            )

    # 2. Local fallback path: {local_sandbox_dir}/output/reports/
    if local_sandbox_dir:
        local_dir_normalized = str(local_sandbox_dir).replace("\\", "/")
        save_paths.append(
            ("local", Path(local_dir_normalized) / "output" / "reports" / filename)
        )

    # 3. If no path provided, use output/reports in current directory
    if not save_paths:
        save_paths.append(("fallback", Path("output") / "reports" / filename))

    # Try each path until success
    for path_type, report_file in save_paths:
        try:
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"📄 {report_type} report saved to: {report_file} ({path_type})")
            if progress_callback:
                try:
                    report_type_display = {
                        "retrieval": "Retrieval Report",
                        "deep_research": "Deep Research Report",
                        "hypothesis": "Hypothesis Report",
                        "planning": "Experimental Plan",
                        "evaluation": "Evaluation Report",
                    }.get(report_type, report_type)
                    progress_callback(
                        event_type="file_content",
                        message=f"📄 {report_type_display}saved locally",
                        details={
                            "file_type": report_type,
                            "file_name": filename,
                            "file_path": str(report_file),
                            "content": content,
                            "content_length": len(content),
                            "node": "immunity",
                            "session_id": session_id,
                        },
                    )
                    print(
                        f"  [SUCCESS] Pushed {report_type} file content to frontend ({len(content)} characters)"
                    )
                except Exception as e:
                    print(f"  [WARN]️ Failed to push file content to frontend: {e}")
            return str(report_file)
        except Exception as e:
            print(
                f"[WARN]️ Failed to save {report_type} report to {path_type} path {report_file}: {e}"
            )
            continue

    print(f"[ERROR] All save attempts failed for {report_type} report")
    return ""


def _clean_json_response(response_text: str) -> Dict[str, Any]:
    """
    Clean and parse JSON response

    Args:
        response_text: Text returned by LLM

    Returns:
        Parsed JSON dictionary
    """
    # Try direct parsing
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON code blocks
    json_block_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]

    for pattern in json_block_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Try extracting the first JSON object
    json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # If all fail, return empty dictionary
    return {}


# ===================== Stage 0: Cache Check Node (Mem0) =====================


def cache_check_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 0: Cache Check Node (Mem0)

    Check if a similar question has been cached in Mem0

    If cache hit:
    - Load all results directly from cache
    - Set skip_immunity_stages=True to skip subsequent stages

    If cache miss:
    - Continue executing subsequent stages
    """
    _progress_logger.start_stage("cache_check", "Checking Mem0 cache")

    if not MEM0_AVAILABLE:
        _progress_logger.log_info("Mem0 unavailable, skipping cache check")
        _progress_logger.end_stage(
            "cache_check", success=True, details="Skipped (Mem0 unavailable)"
        )
        return state

    if not state.original_question:
        _progress_logger.log_warning("No original question, skipping cache check")
        _progress_logger.end_stage(
            "cache_check", success=True, details="Skipped (no input)"
        )
        return state

    try:
        _progress_logger.log_info(f"Checking cache: {state.original_question[:100]}...")

        # Generate input hash
        input_hash = generate_input_hash(state.original_question)
        state.cache_input_hash = input_hash
        _progress_logger.log_info(f"Input hash: {input_hash}")

        # Check cache
        is_cached, cached_trace = check_immunity_cache_sync(
            user_input=state.original_question,
            score_threshold=0.90,  # Require 90%+ similarity
        )

        if is_cached and cached_trace:
            _progress_logger.log_info("[SUCCESS] Cache hit! Loading results from Mem0...")

            # Load all results from cache
            state.cache_hit = True
            state.skip_immunity_stages = True

            # Load optimized queries
            state.optimized_questions = cached_trace.optimized_questions or [
                state.original_question
            ]
            state.optimized_question = "; ".join(state.optimized_questions)

            # Load research results
            state.research_summary = cached_trace.research_summary or ""
            state.research_confidence = 80.0  # Cached results default to high confidence

            # Load hypothesis results
            state.hypothesis_summary = cached_trace.hypothesis_summary or ""
            state.hypothesis_confidence = 80.0

            # Load plan
            state.final_enhanced_plan = cached_trace.final_enhanced_plan or ""
            state.research_informed_plan = cached_trace.final_enhanced_plan or ""
            state.generated_plan = cached_trace.final_enhanced_plan or ""
            state.execution_plan = cached_trace.execution_plan or ""

            # Load evaluation
            state.final_evaluation = cached_trace.final_evaluation or ""

            # Load Todo-List summary
            if cached_trace.todo_list_summary:
                state.decomposed_tasks = cached_trace.todo_list_summary.get("tasks", [])

            _progress_logger.log_info(
                f"  - Optimized queries count: {len(state.optimized_questions)}"
            )
            _progress_logger.log_info(
                f"  - Research summary length: {len(state.research_summary)}"
            )
            _progress_logger.log_info(f"  - Plan length: {len(state.final_enhanced_plan)}")

            _progress_logger.end_stage(
                "cache_check", success=True, details="[SUCCESS] Cache hit"
            )
        else:
            _progress_logger.log_info("[ERROR] Cache miss, will continue executing immunity stages")
            state.cache_hit = False
            state.skip_immunity_stages = False
            _progress_logger.end_stage(
                "cache_check", success=True, details="Cache miss"
            )

    except Exception as e:
        _progress_logger.log_error("Cache check failed", e)
        import traceback

        traceback.print_exc()
        state.cache_hit = False
        state.skip_immunity_stages = False
        _progress_logger.end_stage("cache_check", success=False, details=str(e)[:50])

    return state


# ===================== Stage 1: Query Decomposition Node =====================


def query_decomposition_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 1: Query Decomposition Node

    Decompose user question into optimized sub-questions.
    This node has search tools bound to help understand research context.
    """
    # Start workflow (when first node executes)
    if not _progress_logger.workflow_start_time:
        _progress_logger.start_workflow()

    _progress_logger.start_stage("query_decomposition", "Decompose user question into optimized sub-queries")

    if not state.original_question:
        _progress_logger.log_warning("No original question, skipping query decomposition")
        _progress_logger.end_stage(
            "query_decomposition", success=True, details="Skipped (no input)"
        )
        return state

    _progress_logger.log_info(f"Original question: {state.original_question[:150]}...")

    llm = _get_llm_with_callback(state, "bioinformatics")
    if not llm:
        _progress_logger.log_warning("LLM unavailable, using original question")
        state.optimized_questions = [state.original_question]
        state.optimized_question = state.original_question
        _progress_logger.end_stage(
            "query_decomposition", success=True, details="Fallback (no LLM)"
        )
        return state

    try:
        from langchain_core.messages import (
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )
        from langchain_core.output_parsers import JsonOutputParser

        tools_info = _load_tools_json()

        # Use reference project's QUERY_EXPANSION_PROMPT
        query_expansion_prompt = ImmunityPrompts.QUERY_EXPANSION_PROMPT.format(
            tools_info=tools_info, query=state.original_question
        )

        # Define output schema
        class QueryExpansion(BaseModel):
            queries: List[str]

        output_parser = JsonOutputParser(pydantic_object=QueryExpansion)

        messages = [
            SystemMessage(
                content="You are a professional query optimization expert capable of decomposing complex research queries into multiple optimized sub-queries."
            ),
            HumanMessage(content=query_expansion_prompt),
        ]

        # Bind search tools to LLM for understanding research context
        llm_with_tools = llm.bind_tools([web_search, knowledge_search])

        # Log LLM call
        llm_info = {
            "model": getattr(llm, "model", getattr(llm, "model_name", "unknown")),
            "temperature": getattr(llm, "temperature", "N/A"),
            "timeout": getattr(llm, "timeout", "N/A"),
            "tools_bound": ["web_search", "knowledge_search"],
        }
        _progress_logger.log_llm_start(llm_info, len(query_expansion_prompt))

        start_time = time.perf_counter()

        # Invoke LLM with tools, handle tool calls if any
        max_tool_iterations = 2  # Limit tool call iterations for query decomposition
        tool_iterations = 0

        while tool_iterations < max_tool_iterations:
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                _progress_logger.log_llm_end(elapsed, 0, success=False)
                raise

            # Check if LLM made tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_iterations += 1
                _progress_logger.log_info(
                    f"  [TOOL] LLM requested tool call (iteration {tool_iterations}/{max_tool_iterations})"
                )

                # Add AI message with tool calls to conversation
                messages.append(response)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")

                    _progress_logger.log_info(
                        f"    - Execute tool: {tool_name}({tool_args})"
                    )

                    # Execute tool
                    tool_result = _execute_tool_call(tool_name, tool_args)

                    # Add tool result to messages
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_id)
                    )

                # Continue loop to get final response
                continue

            # No tool calls, we have the final response
            break

        elapsed = time.perf_counter() - start_time
        _progress_logger.log_llm_end(elapsed, len(str(response)) if response else 0)

        if tool_iterations > 0:
            _progress_logger.log_info(f"  [STAT] Total tool calls: {tool_iterations}")

        # Parse the response to extract queries
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Try to parse as JSON for structured output
        try:
            parsed = output_parser.parse(response_content)
            if isinstance(parsed, dict) and "queries" in parsed:
                state.optimized_questions = parsed["queries"]
            elif hasattr(parsed, "queries"):
                state.optimized_questions = parsed.queries
            else:
                # Fallback: use structured output on original LLM
                structured_llm = llm.with_structured_output(QueryExpansion)
                structured_response = structured_llm.invoke(messages)
                if hasattr(structured_response, "queries"):
                    state.optimized_questions = structured_response.queries
                elif isinstance(structured_response, dict):
                    state.optimized_questions = structured_response.get(
                        "queries", [state.original_question]
                    )
                else:
                    state.optimized_questions = [state.original_question]
        except Exception as parse_error:
            _progress_logger.log_warning(
                f"JSON parsing failed, trying structured_output: {parse_error}"
            )
            # Fallback: use structured output
            structured_llm = llm.with_structured_output(QueryExpansion)
            structured_response = structured_llm.invoke(messages)
            if hasattr(structured_response, "queries"):
                state.optimized_questions = structured_response.queries
            elif isinstance(structured_response, dict):
                state.optimized_questions = structured_response.get(
                    "queries", [state.original_question]
                )
            else:
                state.optimized_questions = [state.original_question]

        if not state.optimized_questions:
            state.optimized_questions = [state.original_question]

        state.optimized_question = "; ".join(state.optimized_questions)

        # Output decomposition results
        _progress_logger.log_info(f"Generated sub-queries count: {len(state.optimized_questions)}")
        for i, q in enumerate(state.optimized_questions, 1):
            _progress_logger.log_info(f"  Sub-query {i}: {q[:80]}...")

        _progress_logger.end_stage(
            "query_decomposition",
            success=True,
            details=f"Generated {len(state.optimized_questions)} optimized queries",
        )

    except Exception as e:
        _progress_logger.log_error("Query decomposition failed", e)
        state.optimized_questions = [state.original_question]
        state.optimized_question = state.original_question
        _progress_logger.end_stage(
            "query_decomposition", success=False, details="Fallback using original question"
        )

    return state


# ===================== Stage 2: Retrieval Node =====================


def retrieval_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 2: Retrieval Node

    Parallel execution of three retrieval methods:
    1. retrieve: Retrieve from Qdrant vector database
    2. web_search_node: Tavily API web search
    3. web_retrieval_search: Multiple Web source retrieval

    Retrieval results are used for:
    - Stage 3: Deep research analysis
    - Stage 5: Plan generation (citation references)
    """
    _progress_logger.start_stage("retrieval", "Parallel execution of three retrieval methods")

    if not state.optimized_questions:
        _progress_logger.log_warning("No optimized queries, skipping retrieval")
        _progress_logger.end_stage("retrieval", success=True, details="Skipped (no queries)")
        return state

    try:
        from .retrieval_tools import parallel_retrieval_sync

        _progress_logger.log_info("Executing three retrieval methods:")
        _progress_logger.log_info("  1. Qdrant vector database retrieval")
        _progress_logger.log_info("  2. Tavily API web search")
        _progress_logger.log_info("  3. Web multi-source retrieval")
        _progress_logger.log_info(f"Retrieval queries count: {len(state.optimized_questions)}")

        start_time = time.perf_counter()

        retrieval_results = parallel_retrieval_sync(
            queries=state.optimized_questions,
            original_question=state.original_question,
            k_per_query=10,
        )

        elapsed = time.perf_counter() - start_time

        # Extract retrieval results
        state.context = retrieval_results.get("context", "")
        state.retrieval_docs = retrieval_results.get("retrieval_docs", [])
        state.citations = retrieval_results.get("citations", [])

        # Generate retrieval summary
        retrieval_summary = f"""
Retrieval Completed (Parallel Retrieval):
- Optimized queries count: {len(state.optimized_questions)}
- Retrieved documents count: {len(state.retrieval_docs)}
- Citations count: {len(state.citations)}
- Context length: {len(state.context)} characters

Retrieval Methods:
1. Qdrant vector database retrieval
2. Tavily API web search
3. Web retrieval (multiple sources)

Retrieved Documents (Top 10):
{chr(10).join([f"{i + 1}. **{doc.get('title', 'N/A')}** (Relevance: {doc.get('relevance_score', 0):.2f})" + chr(10) + f"   - Source: {doc.get('source', 'N/A')}" + chr(10) + f"   - Summary: {doc.get('summary', '')[:200]}..." for i, doc in enumerate(state.retrieval_docs[:10])])}

Main Citations (Top 10):
{chr(10).join([f"{i + 1}. {cite.get('author', 'N/A')} et al. ({cite.get('year', 'N/A')}). {cite.get('title', 'N/A')}. *{cite.get('journal', 'N/A')}*" + (f" DOI: {cite.get('doi', '')}" if cite.get("doi") else "") for i, cite in enumerate(state.citations[:10])])}
"""

        _progress_logger.log_info(f"Retrieval completed, time: {elapsed:.2f} seconds")
        _progress_logger.log_info(f"  - Retrieved docs count: {len(state.retrieval_docs)}")
        _progress_logger.log_info(f"  - Citations count: {len(state.citations)}")
        _progress_logger.log_info(f"  - Context length: {len(state.context)} characters")

        # Save retrieval report
        report_path = _save_report(
            retrieval_summary,
            "retrieval",
            state.sandbox_dir,
            state.local_sandbox_dir,
            opensandbox_id=_get_opensandbox_id(state),
            progress_callback=get_progress_callback_by_session(state.session_id),
            session_id=state.session_id,
        )
        state.retrieval_report_path = report_path

        _progress_logger.end_stage(
            "retrieval",
            success=True,
            details=f"Docs: {len(state.retrieval_docs)}, Citations: {len(state.citations)}, time: {elapsed:.1f}s",
        )

    except Exception as e:
        _progress_logger.log_error("Retrieval failed", e)
        import traceback

        traceback.print_exc()
        # Use empty context on failure
        state.context = ""
        state.retrieval_docs = []
        state.citations = []
        _progress_logger.end_stage(
            "retrieval", success=False, details="Retrieval error, using empty context"
        )

    return state


# ===================== Stage 3: Deep Research Node (using deep_research subgraph) =====================


def deep_research_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 3: Deep Research Node

    Uses the deep_research subgraph to conduct in-depth analysis of research questions.
    The deep_research subgraph provides multi-step research with web search and synthesis.
    """
    _progress_logger.start_stage("deep_research", "Multi-step analysis using deep_research subgraph")

    if not state.original_question:
        _progress_logger.log_warning("No original question, skipping deep research")
        _progress_logger.end_stage(
            "deep_research", success=True, details="Skipped (no input)"
        )
        return state

    try:
        # Prepare the research question combining original question and retrieval context
        research_question = state.original_question

        # Add retrieval context if available
        if state.context:
            research_question = f"""
Research Topic: {state.original_question}

Retrieved background materials:
{state.context[:4000]}

Please conduct in-depth research and answer the above question based on the background materials.
"""

        # Add optimized queries if available
        if state.optimized_questions:
            sub_queries = "\n".join([f"- {q}" for q in state.optimized_questions[:5]])
            research_question += f"\n\nFocus on the following sub-questions:\n{sub_queries}"

        _progress_logger.log_info(f"Research question: {state.original_question[:100]}...")
        _progress_logger.log_info("Multi-step analysis using deep_research subgraph...")
        _progress_logger.log_info(
            "Config: max_iterations=3, max_concurrent=2, max_tool_calls=6"
        )

        # Get deep_research subgraph configuration
        dr_config = get_deep_research_config(
            thread_id=f"immunity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            max_researcher_iterations=3,  # Limit iterations for efficiency
            max_concurrent_research_units=2,
            max_react_tool_calls=6,
        )

        # Prepare input for deep_research subgraph
        research_input = {"messages": [{"role": "user", "content": research_question}]}

        # Run deep_research subgraph asynchronously
        async def run_deep_research_async():
            from langgraph.checkpoint.memory import MemorySaver
            from nodes.subagents.deep_research.deep_researcher import (
                deep_researcher_builder,
            )

            _progress_logger.log_info("Compiling deep_research subgraph...")

            # Compile with memory checkpointing
            graph = deep_researcher_builder.compile(checkpointer=MemorySaver())

            _progress_logger.log_info(
                "Starting deep_research subgraph execution (may take a while)..."
            )

            return await graph.ainvoke(research_input, dr_config)

        # Execute async function with proper event loop handling
        start_time = time.perf_counter()
        result = None

        def run_with_proper_cleanup(coro_factory):
            """Run async coroutine with proper cleanup to avoid 'Event loop is closed' errors."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro_factory())
            finally:
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                try:
                    loop.run_until_complete(loop.shutdown_default_executor())
                except Exception:
                    pass
                finally:
                    loop.close()

        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an async context, we need to run in a separate thread
                _progress_logger.log_info("Detected running event loop, using thread pool...")

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="deep_research_"
                ) as executor:
                    future = executor.submit(
                        run_with_proper_cleanup, run_deep_research_async
                    )
                    result = future.result(timeout=1800)  # 30 minute timeout
            except RuntimeError:
                # No running loop, safe to use our cleanup function
                _progress_logger.log_info("Using new event loop...")
                result = run_with_proper_cleanup(run_deep_research_async)

        except concurrent.futures.TimeoutError:
            elapsed = time.perf_counter() - start_time
            _progress_logger.log_warning(
                f"deep_research subgraph execution timeout ({elapsed:.1f}seconds)，Using fallback"
            )
            result = None
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            _progress_logger.log_error(f"deep_research subgraph execution failed: {e}", e)
            result = None

        elapsed = time.perf_counter() - start_time

        # Handle case where deep_research failed or timed out
        if result is None:
            _progress_logger.log_warning(
                "deep_research returned no results, using retrieval context as research summary"
            )
            # Fallback: use retrieval context as research summary
            if state.context:
                state.research_summary = f"""
<research_findings>
    <research_finding>
        {state.context[:3000]}
    </research_finding>
</research_findings>
"""
                state.research_confidence = 40.0
            else:
                state.research_summary = "Deep research could not be completed. Please analyze based on the original question."
                state.research_confidence = 20.0
            _progress_logger.end_stage(
                "deep_research", success=True, details="Using fallback"
            )
            return state

        _progress_logger.log_info(f"deep_research subgraph execution completed，time: {elapsed:.2f} seconds")

        # Extract results from deep_research subgraph
        final_report = result.get("final_report", "")
        research_brief = result.get("research_brief", "")
        notes = result.get("notes", [])

        # Map results back to ImmunityState
        if final_report:
            state.research_summary = f"""
<research_findings>
    <research_finding>
        {final_report}
    </research_finding>
</research_findings>
"""
            state.deep_research_findings = {
                "final_report": final_report,
                "research_brief": research_brief,
                "notes": notes,
                "topic": state.original_question,
            }

            # Extract insights from notes
            state.research_insights = []
            state.research_evidence = []
            for note in notes[:10]:
                if isinstance(note, str):
                    state.research_insights.append(note[:500])  # Truncate long notes

            # Set confidence based on results
            state.research_confidence = 80.0 if final_report else 50.0

            _progress_logger.log_info(f"Deep research completed:")
            _progress_logger.log_info(f"  - Final report length: {len(final_report)} characters")
            _progress_logger.log_info(f"  - Research brief length: {len(research_brief)} characters")
            _progress_logger.log_info(f"  - Notes count: {len(notes)}")
            _progress_logger.log_info(f"  - Confidence: {state.research_confidence:.1f}%")

            # Save research report
            report_path = _save_report(
                state.research_summary,
                "deep_research",
                state.sandbox_dir,
                state.local_sandbox_dir,
                opensandbox_id=_get_opensandbox_id(state),
                progress_callback=get_progress_callback_by_session(state.session_id),
                session_id=state.session_id,
            )

            _progress_logger.end_stage(
                "deep_research",
                success=True,
                details=f"Report: {len(final_report)}characters, Notes: {len(notes)}, time: {elapsed:.1f}s",
            )
        else:
            _progress_logger.log_warning("Deep research returned empty results, using fallback")
            # Fallback to simple context-based research
            state.research_summary = f"""
<research_findings>
    <research_finding>
        Research Topic: {state.original_question}
        
        Based on retrieval context:
        {state.context[:2000] if state.context else "No context available"}
        
        Optimized queries:
        {chr(10).join([f"- {q}" for q in state.optimized_questions[:5]])}
    </research_finding>
</research_findings>
"""
            state.research_confidence = 50.0
            _progress_logger.end_stage(
                "deep_research", success=True, details="Fallback using retrieval context"
            )

    except Exception as e:
        _progress_logger.log_error("Deep research failed", e)
        import traceback

        traceback.print_exc()

        # Fallback: create research summary from available context
        state.research_summary = f"""
<research_findings>
    <research_finding>
        Research Topic: {state.original_question}
        
        Note: Deep research subgraph encountered an error. Using available context.
        
        Context from retrieval:
        {state.context[:2000] if state.context else "No context available"}
        
        Optimized queries:
        {chr(10).join([f"- {q}" for q in state.optimized_questions[:5]])}
    </research_finding>
</research_findings>
"""
        state.research_confidence = 40.0
        _progress_logger.end_stage(
            "deep_research", success=False, details="Using context fallback"
        )

    return state


# ===================== Stage 4: Hypothesis Generation Node =====================


def hypothesis_generation_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 4: Hypothesis Generation Node

    Generate testable hypotheses based on research results
    """
    _progress_logger.start_stage(
        "hypothesis_generation", "Generate testable hypotheses based on research results"
    )

    if not state.research_summary:
        _progress_logger.log_warning("No research results, skipping hypothesis generation")
        _progress_logger.end_stage(
            "hypothesis_generation", success=True, details="Skipped (no research results)"
        )
        return state

    llm = _get_llm_with_callback(state, "bioinformatics")
    if not llm:
        _progress_logger.log_warning("LLM unavailable, skipping hypothesis generation")
        _progress_logger.end_stage(
            "hypothesis_generation", success=True, details="Skipped (no LLM)"
        )
        return state

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        # Use reference project's HYPOTHESIS_GENERATION_PROMPT
        context = state.context if state.context else ""
        question = f"""
<questions>
    <question>
        {state.original_question}
    </question>
</questions>
"""

        # Use reference project's HYPOTHESIS_GENERATION_PROMPT
        hypothesis_prompt = ImmunityPrompts.HYPOTHESIS_GENERATION_PROMPT.format(
            research_findings=state.research_summary, context=context, question=question
        )

        # Use JSON parser (not using with_structured_output, as dict type is not supported)
        from langchain_core.output_parsers import JsonOutputParser

        output_parser = JsonOutputParser()

        # ZhipuAI requires at least one user message, so send prompt as user message
        messages = [HumanMessage(content=hypothesis_prompt)]

        llm_info = {
            "type": type(llm).__name__,
            "model": getattr(llm, "model", getattr(llm, "model_name", None)),
            "temperature": getattr(llm, "temperature", None),
            "timeout": getattr(llm, "timeout", None),
            "max_retries": getattr(llm, "max_retries", None),
        }
        prompt_stats = {
            "original_question_len": len(state.original_question or ""),
            "research_summary_len": len(state.research_summary or ""),
            "context_len": len(context),
            "question_block_len": len(question),
        }
        _progress_logger.log_info(f"Prompt stats: {prompt_stats}")

        # Log LLM call
        _progress_logger.log_llm_start(llm_info, len(hypothesis_prompt))

        # Directly invoke LLM, then parse JSON
        start_time = time.perf_counter()
        try:
            response = llm.invoke(messages)
            elapsed = time.perf_counter() - start_time
            _progress_logger.log_llm_end(
                elapsed, len(response.content) if hasattr(response, "content") else 0
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            _progress_logger.log_llm_end(elapsed, 0, success=False)
            raise
        # [DEBUG] Detailed response debugging
        _progress_logger.log_info(f"[DEBUG] Response type: {type(response).__name__}")
        _progress_logger.log_info(
            f"[DEBUG] Response attributes: {dir(response)[:10]}..."
        )
        _progress_logger.log_info(
            f"[DEBUG] Has 'content' attribute: {hasattr(response, 'content')}"
        )
        _progress_logger.log_info(
            f"[DEBUG] Has 'additional_kwargs': {hasattr(response, 'additional_kwargs')}"
        )

        # [CRITICAL] Check for reasoning_content (native thinking mode)
        if hasattr(response, "additional_kwargs"):
            additional_kwargs = response.additional_kwargs
            _progress_logger.log_info(
                f"[DEBUG] additional_kwargs keys: {list(additional_kwargs.keys()) if additional_kwargs else 'None'}"
            )
            if additional_kwargs and "reasoning_content" in additional_kwargs:
                reasoning = additional_kwargs["reasoning_content"]
                _progress_logger.log_info(
                    f"[DEBUG] Found reasoning_content! Length: {len(reasoning) if reasoning else 0}"
                )
                _progress_logger.log_info(
                    f"[DEBUG] reasoning_content preview: {reasoning[:200] if reasoning else 'None'}"
                )

        if hasattr(response, "content"):
            raw_content = response.content
            _progress_logger.log_info(
                f"[DEBUG] Content type: {type(raw_content).__name__}"
            )
            _progress_logger.log_info(f"[DEBUG] Content is None: {raw_content is None}")
            _progress_logger.log_info(
                f"[DEBUG] Content is str: {isinstance(raw_content, str)}"
            )
            _progress_logger.log_info(
                f"[DEBUG] Content is list: {isinstance(raw_content, list)}"
            )
            if raw_content:
                if isinstance(raw_content, str):
                    _progress_logger.log_info(
                        f"[DEBUG] Content length: {len(raw_content)}"
                    )
                    _progress_logger.log_info(
                        f"[DEBUG] Content preview: {raw_content[:300]}"
                    )
                elif isinstance(raw_content, list):
                    _progress_logger.log_info(
                        f"[DEBUG] Content list length: {len(raw_content)}"
                    )
                    if len(raw_content) > 0:
                        _progress_logger.log_info(
                            f"[DEBUG] First item type: {type(raw_content[0]).__name__}"
                        )
                        _progress_logger.log_info(
                            f"[DEBUG] First item: {str(raw_content[0])[:300]}"
                        )

        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # [CRITICAL] If content is a list (multimodal response), special handling needed
        if isinstance(response_content, list):
            _progress_logger.log_info(
                f"[DEBUG] Detected list response, extracting text..."
            )
            text_parts = []
            for item in response_content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif hasattr(item, "content"):
                    text_parts.append(item.content)
            response_content = "\n".join(text_parts)
            _progress_logger.log_info(
                f"[DEBUG] Extracted text length: {len(response_content)}"
            )
            _progress_logger.log_info(
                f"[DEBUG] Extracted text preview: {response_content[:300]}"
            )

        # [VALIDATION] Check if response is empty
        if (
            not response_content
            or not isinstance(response_content, str)
            or not response_content.strip()
        ):
            _progress_logger.log_error("LLM returned empty or invalid response")
            _progress_logger.log_error(
                f"Response content type: {type(response_content).__name__}"
            )
            _progress_logger.log_error(
                f"Response content value: {repr(response_content)}"
            )
            _progress_logger.end_stage(
                "hypothesis_generation", success=False, details="LLM returned empty response"
            )
            return state

        # Use robust JSON extraction tool
        from utils.json_extractor import extract_json_from_llm_response

        # First try JsonOutputParser
        hypothesis_data = None
        try:
            parsed = output_parser.parse(response_content)
            if isinstance(parsed, dict):
                hypothesis_data = parsed
        except Exception as e:
            _progress_logger.log_warning(
                f"JsonOutputParser parsing failed, trying robust extraction: {e}"
            )

        # If JsonOutputParser fails, use robust extraction function
        if hypothesis_data is None or not isinstance(hypothesis_data, dict):
            hypothesis_data = extract_json_from_llm_response(
                response_content, default={}, log_errors=True
            )

            if not hypothesis_data:
                _progress_logger.log_warning(
                    f"JSON extraction failed, response content preview: {response_content[:200] if response_content else 'empty response'}"
                )

        if hypothesis_data and hypothesis_data.get("statement"):
            state.hypothesis = hypothesis_data
            state.hypothesis_confidence = float(
                hypothesis_data.get("confidence_score", 70.0)
            )

            # Extract testable predictions
            predictions = hypothesis_data.get("testable_predictions", [])
            state.testable_predictions = []
            for pred in predictions:
                if isinstance(pred, dict):
                    state.testable_predictions.append(pred.get("prediction", ""))
                else:
                    state.testable_predictions.append(str(pred))

            # Generate hypothesis summary
            hypothesis_summary = f"""
Hypothesis Statement: {hypothesis_data.get("statement", "Not specified")}

Confidence Score: {hypothesis_data.get("confidence_score", 0)}%
Innovation Level: {hypothesis_data.get("innovation_level", "moderate")}

Testable Predictions:
{chr(10).join([f"- {pred.get('prediction', '')} (Timeline: {pred.get('timeline', 'TBD')})" for pred in predictions[:5]])}

Validation Methods:
{chr(10).join([f"- {pred.get('validation_method', '')}" for pred in predictions[:5]])}

Expected Outcomes:
{chr(10).join([f"- {pred.get('expected_outcome', '')}" for pred in predictions[:5]])}

Falsification Criteria:
{chr(10).join([f"- {criteria}" for criteria in hypothesis_data.get("falsification_criteria", [])[:5]])}

Evidence Basis:
{chr(10).join([f"- {evidence}" for evidence in hypothesis_data.get("evidence_basis", [])[:5]])}

Expected Information Gain:
{hypothesis_data.get("expected_information_gain", "To be determined")}

Scientific Rationale:
{hypothesis_data.get("rationale", "Not provided")}
"""

            state.hypothesis_summary = f"""
<hypothesis_findings>
    <hypothesis_finding>
        {hypothesis_summary}
    </hypothesis_finding>
</hypothesis_findings>
"""

            _progress_logger.log_info(f"Hypothesis generation completed:")
            _progress_logger.log_info(
                f"  - Hypothesis: {hypothesis_data.get('statement', 'Not specified')[:100]}..."
            )
            _progress_logger.log_info(f"  - Confidence: {state.hypothesis_confidence:.1f}%")
            _progress_logger.log_info(
                f"  - Innovation level: {hypothesis_data.get('innovation_level', 'moderate')}"
            )

            # Save hypothesis report
            report_path = _save_report(
                state.hypothesis_summary,
                "hypothesis",
                state.sandbox_dir,
                state.local_sandbox_dir,
                opensandbox_id=_get_opensandbox_id(state),
                progress_callback=get_progress_callback_by_session(state.session_id),
                session_id=state.session_id,
            )

            _progress_logger.end_stage(
                "hypothesis_generation",
                success=True,
                details=f"Confidence: {state.hypothesis_confidence:.0f}%, Predictions: {len(predictions)}",
            )
        else:
            _progress_logger.log_warning("Unable to parse hypothesis results")
            _progress_logger.end_stage(
                "hypothesis_generation", success=False, details="JSON parsing failed"
            )

    except Exception as e:
        _progress_logger.log_error("Hypothesis generation failed", e)
        import traceback

        traceback.print_exc()
        _progress_logger.end_stage(
            "hypothesis_generation", success=False, details=str(e)[:50]
        )

    return state


# ===================== Stage 5: Plan Generation Node ⭐ =====================


def _execute_tool_call(tool_name: str, tool_args: dict) -> str:
    """
    Execute a tool call and return the result.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool

    Returns:
        Tool execution result as string
    """
    try:
        if tool_name == "web_search":
            return web_search.invoke(
                tool_args.get("query", ""), tool_args.get("max_results", 5)
            )
        elif tool_name == "knowledge_search":
            return knowledge_search.invoke(
                tool_args.get("query", ""),
                tool_args.get("k", 5),
                tool_args.get("collections", None),
            )
        elif tool_name == "read_webpage":
            return read_webpage.invoke(
                tool_args.get("url", ""), tool_args.get("max_chars", 10000)
            )
        else:
            return f"[Error] Unknown tool: {tool_name}"
    except Exception as e:
        return f"[Error] Tool {tool_name} execution failed: {str(e)}"


def planning_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 5: Planning Node ⭐

    Generate executable experimental plan based on research results and hypotheses.
    This node has search tools bound to help LLM find latest experimental methods.
    """
    _progress_logger.start_stage("planning", "Generate executable experimental plan based on research results and hypotheses")

    llm = _get_llm_with_callback(state, "bioinformatics")
    if not llm:
        _progress_logger.log_warning("LLM unavailable, using simple plan generation")
        _progress_logger.end_stage("planning", success=True, details="Fallback using simple plan")
        return _generate_simple_plan(state)

    try:
        from langchain_core.messages import (
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )

        tools_info = _load_tools_json()

        # Format optimized queries
        format_queries = []
        for i, query in enumerate(state.optimized_questions, 1):
            format_queries.append(f"""
<sub_questions>
    <q{i}>
        {query}
    </q{i}>
</sub_questions>
""")
        optimized_questions_text = (
            "\n\n".join(format_queries) if format_queries else "None"
        )

        # Get citations (from retrieval node)
        if state.citations:
            citations_json = json.dumps(state.citations, ensure_ascii=False, indent=2)
        else:
            citations_json = "[]"
        context = state.context if state.context else ""

        # Use reference project's IMMUNITY_PLANNING_PROMPT
        planning_prompt = ImmunityPrompts.IMMUNITY_PLANNING_PROMPT.format(
            original_question=state.original_question,
            optimized_questions=optimized_questions_text,
            hypothesis_findings=state.hypothesis_summary,
            tools_info=tools_info,
            research_findings=state.research_summary,
            context=context,
            citations_json=citations_json,
        )

        # ZhipuAI requires at least one user message, so send prompt as user message
        messages = [HumanMessage(content=planning_prompt)]

        # Bind search tools to LLM for planning (helps find latest methods)
        llm_with_tools = llm.bind_tools([web_search, knowledge_search, read_webpage])

        llm_info = {
            "type": type(llm).__name__,
            "model": getattr(llm, "model", getattr(llm, "model_name", None)),
            "temperature": getattr(llm, "temperature", None),
            "timeout": getattr(llm, "timeout", None),
            "max_retries": getattr(llm, "max_retries", None),
            "tools_bound": ["web_search", "knowledge_search", "read_webpage"],
        }
        prompt_stats = {
            "original_question_len": len(state.original_question or ""),
            "optimized_questions_count": len(state.optimized_questions or []),
            "optimized_questions_len": len(optimized_questions_text),
            "hypothesis_summary_len": len(state.hypothesis_summary or ""),
            "research_summary_len": len(state.research_summary or ""),
            "context_len": len(context),
            "citations_count": len(state.citations or []),
            "citations_json_len": len(citations_json),
            "tools_info_len": len(tools_info),
        }
        _progress_logger.log_info(f"Prompt stats: {prompt_stats}")

        # Log LLM call
        _progress_logger.log_llm_start(llm_info, len(planning_prompt))

        start_time = time.perf_counter()

        # Invoke LLM with tools, handle tool calls if any
        max_tool_iterations = 3  # Limit tool call iterations
        tool_iterations = 0

        while tool_iterations < max_tool_iterations:
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                _progress_logger.log_llm_end(elapsed, 0, success=False)
                raise

            # Check if LLM made tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_iterations += 1
                _progress_logger.log_info(
                    f"  [TOOL] LLM requested tool call (iteration {tool_iterations}/{max_tool_iterations})"
                )

                # Add AI message with tool calls to conversation
                messages.append(response)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")

                    _progress_logger.log_info(
                        f"    - Execute tool: {tool_name}({tool_args})"
                    )

                    # Execute tool
                    tool_result = _execute_tool_call(tool_name, tool_args)

                    # Add tool result to messages
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_id)
                    )

                # Continue loop to get final response
                continue

            # No tool calls, we have the final response
            break

        elapsed = time.perf_counter() - start_time
        _progress_logger.log_llm_end(
            elapsed, len(response.content) if hasattr(response, "content") else 0
        )

        if tool_iterations > 0:
            _progress_logger.log_info(f"  [STAT] Total tool calls: {tool_iterations}")

        plan_content = (
            response.content.strip() if hasattr(response, "content") else str(response)
        )

        state.final_enhanced_plan = plan_content
        state.research_informed_plan = plan_content
        state.generated_plan = plan_content

        _progress_logger.log_info(f"Plan generation completed:")
        _progress_logger.log_info(f"  - Plan length: {len(plan_content)} characters")

        # Save plan report
        report_path = _save_report(
            plan_content,
            "planning",
            state.sandbox_dir,
            state.local_sandbox_dir,
            opensandbox_id=_get_opensandbox_id(state),
            progress_callback=get_progress_callback_by_session(state.session_id),
            session_id=state.session_id,
        )

        _progress_logger.end_stage(
            "planning",
            success=True,
            details=f"Plan length: {len(plan_content)} characters, time: {elapsed:.1f}s",
        )

    except Exception as e:
        _progress_logger.log_error("Plan generation failed", e)
        import traceback

        traceback.print_exc()
        # Fallback solution
        _progress_logger.end_stage(
            "planning", success=False, details="Fallback using simple plan"
        )
        return _generate_simple_plan(state)

    return state


def _generate_simple_plan(state: ImmunityState) -> ImmunityState:
    """Generate simple plan (fallback solution)"""
    plan_lines = ["# Experimental Plan\n"]
    plan_lines.append(f"## Overview\n")
    plan_lines.append(f"Based on research question: {state.original_question}\n\n")

    if state.hypothesis_summary:
        plan_lines.append(f"## Hypothesis\n")
        plan_lines.append(f"{state.hypothesis_summary}\n\n")

    if state.research_summary:
        plan_lines.append(f"## Research Background\n")
        plan_lines.append(f"{state.research_summary[:500]}...\n\n")

    plan_lines.append(f"## Experimental Steps\n")
    plan_lines.append(f"(Detailed plan needs to be generated by LLM)\n")

    state.final_enhanced_plan = "".join(plan_lines)
    return state


# ===================== Stage 6: Evaluation Node =====================


def evaluation_node(state: ImmunityState) -> ImmunityState:
    """
    Stage 6: Evaluation Node

    Evaluate the scientific validity and feasibility of the experimental plan
    """
    _progress_logger.start_stage("evaluation", "Evaluate scientific validity and feasibility of experimental plan")

    if not state.final_enhanced_plan:
        _progress_logger.log_warning("No plan to evaluate")
        state.final_evaluation = "No plan generated, cannot evaluate"
        _progress_logger.end_stage("evaluation", success=True, details="Skipped (no plan)")
        return state

    # If user-provided plan, skip evaluation
    if state.is_user_provided_plan:
        _progress_logger.log_info("User-provided plan, skipping automatic evaluation")
        state.final_evaluation = f"""User-provided execution plan:

{state.final_enhanced_plan}

---
Evaluation Note: This plan was directly provided by the user, automatic evaluation step has been skipped.
"""
        report_path = _save_report(
            state.final_evaluation,
            "evaluation",
            state.sandbox_dir,
            state.local_sandbox_dir,
            opensandbox_id=_get_opensandbox_id(state),
            progress_callback=get_progress_callback_by_session(state.session_id),
            session_id=state.session_id,
        )
        _progress_logger.end_stage(
            "evaluation", success=True, details="Skipped (user plan)"
        )
        return state

    llm = _get_llm_with_callback(state, "bioinformatics")
    if not llm:
        _progress_logger.log_warning("LLM unavailable, skipping evaluation")
        state.final_evaluation = "LLM unavailable, cannot perform evaluation"
        _progress_logger.end_stage("evaluation", success=True, details="Skipped (no LLM)")
        return state

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        # Use reference project's EVALUATE_PLANNING_PROMPT
        evaluation_prompt = ImmunityPrompts.EVALUATE_PLANNING_PROMPT.format(
            plan=state.final_enhanced_plan
        )

        # ZhipuAI requires at least one user message, so send prompt as user message
        messages = [HumanMessage(content=evaluation_prompt)]

        llm_info = {
            "model": getattr(llm, "model", getattr(llm, "model_name", None)),
            "temperature": getattr(llm, "temperature", None),
            "timeout": getattr(llm, "timeout", None),
        }
        _progress_logger.log_llm_start(llm_info, len(evaluation_prompt))

        start_time = time.perf_counter()
        response = llm.invoke(messages)
        elapsed = time.perf_counter() - start_time

        evaluation_content = (
            response.content.strip() if hasattr(response, "content") else str(response)
        )

        _progress_logger.log_llm_end(elapsed, len(evaluation_content))

        state.final_evaluation = evaluation_content

        _progress_logger.log_info(f"Evaluation completed:")
        _progress_logger.log_info(f"  - Evaluation report length: {len(evaluation_content)} characters")

        # Save evaluation report
        full_evaluation = evaluation_content + "\n\n" + state.original_question
        report_path = _save_report(
            full_evaluation,
            "evaluation",
            state.sandbox_dir,
            state.local_sandbox_dir,
            opensandbox_id=_get_opensandbox_id(state),
            progress_callback=get_progress_callback_by_session(state.session_id),
            session_id=state.session_id,
        )

        _progress_logger.end_stage(
            "evaluation",
            success=True,
            details=f"Report length: {len(evaluation_content)} characters, time: {elapsed:.1f}s",
        )

    except Exception as e:
        _progress_logger.log_error("Evaluation failed", e)
        import traceback

        traceback.print_exc()
        state.final_evaluation = f"Evaluation process error: {str(e)}"
        _progress_logger.end_stage("evaluation", success=False, details=str(e)[:50])

    return state


# ===================== Input/Output Mapping =====================


def immunity_input_mapper(global_state: GlobalState) -> ImmunityState:
    """
    Map main graph state to Immunity subgraph state

    Args:
        global_state: Main graph global state

    Returns:
        Immunity subgraph state
    """
    # sandbox_data_dir is sandbox server path (e.g., /data/sessions/{session_id})
    # sandbox_dir is local sandbox directory (e.g., D:/path/to/sandbox)
    sandbox_data_dir = global_state.sandbox_data_dir or ""
    local_sandbox_dir = global_state.sandbox_dir or ""

    # [DEBUG] Check progress_callback in global_state
    has_progress_callback = global_state.session_id
    has_session_id = hasattr(global_state, "session_id") and global_state.session_id
    print(f"[Immunity] immunity_input_mapper state check:")
    print(f"  - global_state.progress_callback: {has_progress_callback}")
    print(f"  - global_state.session_id: {has_session_id}")
    print(f"  - global_state.get_llm method exists: {hasattr(global_state, 'get_llm')}")

    immunity_state = ImmunityState(
        original_question=global_state.user_input,
        subtasks=global_state.subtasks,
        parallel_task_groups=global_state.parallel_task_groups,
        sandbox_dir=sandbox_data_dir,  # Primary path: sandbox server path
        local_sandbox_dir=local_sandbox_dir,  # Fallback path: local path
        parent_state=global_state,
        # [FIX] Do NOT pass progress_callback - it cannot be serialized by LangGraph.
        # The callback is retrieved dynamically from global registry via session_id in get_llm().
        session_id=global_state.session_id,
        # Mem0 cache-related field initialization
        cache_hit=False,
        skip_immunity_stages=False,
        cache_input_hash="",
    )

    # [DEBUG] Check immunity_state after creation
    print(f"[Immunity] ImmunityState after creation:")
    print(f"  - immunity_state.session_id: {immunity_state.session_id}")
    print(f"  - immunity_state.parent_state: {immunity_state.parent_state is not None}")

    return immunity_state


def immunity_output_mapper(
    immunity_state: ImmunityState, global_state: GlobalState
) -> GlobalState:
    """
    Map Immunity subgraph state back to main graph state

    Args:
        immunity_state: Immunity subgraph state (can be ImmunityState object or dict)
        global_state: Main graph global state

    Returns:
        Updated main graph state
    """
    # Handle case where immunity_state might be a dict (from LangGraph invoke)
    if isinstance(immunity_state, dict):
        original_question = immunity_state.get("original_question", "")
        optimized_questions = immunity_state.get("optimized_questions", [])
        research_summary = immunity_state.get("research_summary", "")
        hypothesis_summary = immunity_state.get("hypothesis_summary", "")
        final_enhanced_plan = immunity_state.get("final_enhanced_plan", "")
        plan_steps = immunity_state.get("plan_steps", [])
        plan_summary = immunity_state.get("plan_summary", "")
        final_evaluation = immunity_state.get("final_evaluation", "")
        executable_plan = immunity_state.get("executable_plan", {})
        generated_plan = immunity_state.get("generated_plan", "")
        skip_planning = immunity_state.get("skip_planning", False)
        research_confidence = immunity_state.get("research_confidence", 0.0)
        hypothesis_confidence = immunity_state.get("hypothesis_confidence", 0.0)
    else:
        original_question = getattr(immunity_state, "original_question", "")
        optimized_questions = getattr(immunity_state, "optimized_questions", []) or []
        research_summary = getattr(immunity_state, "research_summary", "")
        hypothesis_summary = getattr(immunity_state, "hypothesis_summary", "")
        final_enhanced_plan = getattr(immunity_state, "final_enhanced_plan", "")
        plan_steps = getattr(immunity_state, "plan_steps", []) or []
        plan_summary = getattr(immunity_state, "plan_summary", "")
        final_evaluation = getattr(immunity_state, "final_evaluation", "")
        executable_plan = getattr(immunity_state, "executable_plan", {}) or {}
        generated_plan = getattr(immunity_state, "generated_plan", "")
        skip_planning = getattr(immunity_state, "skip_planning", False)
        research_confidence = getattr(immunity_state, "research_confidence", 0.0)
        hypothesis_confidence = getattr(immunity_state, "hypothesis_confidence", 0.0)

    # Store complete experimental plan to merged_result
    if not global_state.merged_result:
        global_state.merged_result = {}

    global_state.merged_result["immunity_plan"] = {
        "original_question": original_question,
        "optimized_questions": optimized_questions,
        "research_summary": research_summary,
        "hypothesis_summary": hypothesis_summary,
        "experimental_plan": final_enhanced_plan,
        "final_enhanced_plan": final_enhanced_plan,
        "plan_steps": plan_steps,
        "plan_summary": plan_summary,
        "evaluation": final_evaluation,
        "executable_plan": executable_plan,
    }

    # Persist execution plan to global state for downstream logging/usage
    execution_plan = None
    if final_enhanced_plan:
        execution_plan = final_enhanced_plan
    elif generated_plan:
        execution_plan = generated_plan
    elif plan_summary:
        execution_plan = plan_summary

    if not execution_plan:
        if skip_planning:
            execution_plan = "PLAN_NOT_GENERATED: skip_planning is true"
        elif not original_question:
            execution_plan = "PLAN_NOT_GENERATED: original_question is empty"
        else:
            execution_plan = "PLAN_NOT_GENERATED: planning output is empty"

    global_state.execution_plan = execution_plan

    # Use progress logger to output completion info
    _progress_logger.log_info(f"Immunity subgraph completed:")
    _progress_logger.log_info(f"  - Optimized queries count: {len(optimized_questions)}")
    _progress_logger.log_info(f"  - Research confidence: {research_confidence:.1f}%")
    _progress_logger.log_info(f"  - Hypothesis confidence: {hypothesis_confidence:.1f}%")
    _progress_logger.log_info(
        f"  - Plan document length: {len(final_enhanced_plan or '')} characters"
    )

    # [HOT] Push generated file content to frontend (via SSE)
    if global_state.session_id:
        try:
            # Get session_id and sandbox directory
            session_id = getattr(global_state, "session_id", None)
            sandbox_dir = getattr(global_state, "sandbox_dir", None)

            if session_id and sandbox_dir:
                # Push execution plan
                if final_enhanced_plan or execution_plan:
                    plan_content = final_enhanced_plan or execution_plan
                    global_get_progress_callback_by_session(state.session_id)(
                        event_type="file_content",
                        message=f"📄 Experimental plan generation completed",
                        details={
                            "file_type": "execution_plan",
                            "file_name": "planning_report.md",
                            "content": plan_content,
                            "content_length": len(plan_content),
                            "node": "immunity",
                        },
                    )
                    _progress_logger.log_info(
                        f"  [SUCCESS] Pushed experimental plan to frontend ({len(plan_content)} characters)"
                    )

                # Push research report (if available)
                if research_summary:
                    global_get_progress_callback_by_session(state.session_id)(
                        event_type="file_content",
                        message=f"📚 Research report generated",
                        details={
                            "file_type": "research_report",
                            "file_name": "research_report.md",
                            "content": research_summary,
                            "content_length": len(research_summary),
                            "node": "immunity",
                        },
                    )
                    _progress_logger.log_info(
                        f"  [SUCCESS] Pushed research report to frontend ({len(research_summary)} characters)"
                    )

                # Push hypothesis report (if available)
                if hypothesis_summary:
                    global_get_progress_callback_by_session(state.session_id)(
                        event_type="file_content",
                        message=f"🧬 Hypothesis report generated",
                        details={
                            "file_type": "hypothesis_report",
                            "file_name": "hypothesis_report.md",
                            "content": hypothesis_summary,
                            "content_length": len(hypothesis_summary),
                            "node": "immunity",
                        },
                    )
                    _progress_logger.log_info(
                        f"  [SUCCESS] Pushed hypothesis report to frontend ({len(hypothesis_summary)} characters)"
                    )

                # Push evaluation report (if available)
                if final_evaluation:
                    global_get_progress_callback_by_session(state.session_id)(
                        event_type="file_content",
                        message=f"[STAT] Evaluation report generated",
                        details={
                            "file_type": "evaluation_report",
                            "file_name": "evaluation_report.md",
                            "content": final_evaluation,
                            "content_length": len(final_evaluation),
                            "node": "immunity",
                        },
                    )
                    _progress_logger.log_info(
                        f"  [SUCCESS] Pushed evaluation report to frontend ({len(final_evaluation)} characters)"
                    )

        except Exception as e:
            _progress_logger.log_warning(f"Failed to push file content to frontend: {e}")
            import traceback

            traceback.print_exc()

    # End workflow
    _progress_logger.end_workflow(success=True)

    return global_state


# ===================== Build Immunity Subgraph =====================


def _should_skip_stages(state: ImmunityState) -> str:
    """
    Conditional routing: determine whether to skip subsequent stages

    Returns:
        "skip_to_planning": Skip intermediate stages, go directly to planning (for cache hit)
        "continue_retrieval": Continue executing retrieval stage
    """
    if state.skip_immunity_stages or state.cache_hit:
        return "skip_to_planning"
    return "continue_retrieval"


def _should_skip_to_evaluation(state: ImmunityState) -> str:
    """
    Conditional routing: determine whether to skip to evaluation (skip planning on cache hit)

    Returns:
        "skip_to_evaluation": Skip planning, go directly to evaluation
        "continue_planning": Continue executing planning stage
    """
    if state.skip_immunity_stages or state.cache_hit:
        return "skip_to_evaluation"
    return "continue_planning"


def build_immunity_subgraph():
    """
    Build Immunity Agent subgraph

    Complete workflow:
    Cache Check → Query Decomposition → Retrieval → Deep Research → Hypothesis Generation → Planning ⭐ → Evaluation

    If cache hit, skip intermediate stages and use cached results directly

    Returns:
        Compiled subgraph
    """
    graph = StateGraph(ImmunityState)

    # Add all nodes
    graph.add_node("cache_check", cache_check_node)  # Stage 0: Cache Check (NEW!)
    graph.add_node("query_decomposition", query_decomposition_node)  # Stage 1
    graph.add_node("retrieval", retrieval_node)  # Stage 2: Retrieval node
    graph.add_node("deep_research", deep_research_node)  # Stage 3
    graph.add_node("hypothesis_generation", hypothesis_generation_node)  # Stage 4
    graph.add_node("planning", planning_node)  # Stage 5 ⭐
    graph.add_node("evaluation", evaluation_node)  # Stage 6

    # Define flow rules
    graph.add_edge(START, "cache_check")  # Check cache first
    graph.add_edge(
        "cache_check", "query_decomposition"
    )  # Always execute query_decomposition regardless of cache hit (for logging)

    # Conditional routing: after query_decomposition decide whether to skip intermediate stages
    graph.add_conditional_edges(
        "query_decomposition",
        _should_skip_stages,
        {
            "skip_to_planning": "planning",  # Cache hit, skip intermediate stages
            "continue_retrieval": "retrieval",  # Normal flow
        },
    )

    # Normal flow edges
    graph.add_edge("retrieval", "deep_research")
    graph.add_edge("deep_research", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "planning")

    # Conditional routing: after planning decide whether to skip to evaluation
    # Note: Even on cache hit, we execute the planning node (to generate final plan)
    graph.add_edge("planning", "evaluation")
    graph.add_edge("evaluation", END)

    return graph.compile()
