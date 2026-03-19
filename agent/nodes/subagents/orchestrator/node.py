"""
Orchestrator Node

Bundles tasks by domain and dispatches each bundle as a single sub-agent session.

Architecture:
  1. Route individual subtasks to domains (immune, rna, structural, bioinformatics)
  2. Bundle all same-domain tasks into one SubAgentBundle
  3. Compute inter-bundle dependencies
  4. ReAct loop dispatches bundles (parallel when independent, sequential when dependent)
  5. Each bundle = one opencode_executor call with a multi-step workflow prompt
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Tuple
from pathlib import Path

agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType
from utils.progress_reporter import (
    report_node_start,
    report_node_progress,
    report_node_complete,
    report_error,
    report_info,
)
from .state import (
    OrchestratorState,
    SubAgentAssignment,
    SubAgentBundle,
    OrchestratorTaskStatus,
)
from .router import route_task, route_task_by_tools, get_domain_description
from utils.opencode_executor import OpenCodeExecutor

logger = logging.getLogger(__name__)

SIMPLE_TASK_THRESHOLD = 20


def _collect_input_files_from_sandbox(
    session_id: str, sandbox_dir: str
) -> Dict[str, str]:
    """
    Collect input files from /data/sessions/{session_id}/input directory.

    Args:
        session_id: Session ID
        sandbox_dir: Sandbox directory path

    Returns:
        Dict mapping file_name -> file_path
    """
    from utils.codeact_executor import is_codeact_available, execute_code_via_codeact

    file_paths = {}

    if not session_id:
        return file_paths

    if not is_codeact_available():
        logger.warning("CodeAct/OpenSandbox not available, cannot collect input files")
        return file_paths

    input_dir = f"/data/sessions/{session_id}/input"

    collector_code = f'''
import os
import json
from pathlib import Path

input_dir = Path("{input_dir}")
results = {{}}

if input_dir.exists():
    for file_path in input_dir.rglob("*"):
        if file_path.is_file():
            results[file_path.name] = str(file_path)

print("__INPUT_FILES_JSON_START__")
print(json.dumps(results, ensure_ascii=False))
print("__INPUT_FILES_JSON_END__")
'''

    try:
        result = execute_code_via_codeact(
            task_description=f"Collect input files from {input_dir}",
            code_template=collector_code,
            sandbox_id=None,
            timeout_seconds=30,
            keep_alive=True,
        )

        if not result.is_success():
            logger.warning(f"Failed to collect input files: {result.error}")
            return file_paths

        import re

        stdout = result.output
        json_match = re.search(
            r"__INPUT_FILES_JSON_START__\s*(.*?)\s*__INPUT_FILES_JSON_END__",
            stdout,
            re.DOTALL,
        )

        if json_match:
            import json

            file_paths = json.loads(json_match.group(1))
            logger.info(f"Collected {len(file_paths)} input files from {input_dir}")
        else:
            logger.warning(f"No input files JSON found in output")

    except Exception as e:
        logger.warning(f"Exception while collecting input files: {e}")

    return file_paths


def _build_bundle_prompt(bundle: SubAgentBundle) -> str:
    """Build a single prompt for entire bundle workflow."""
    return bundle.combined_content


def _get_progress_callback_by_session(session_id: str):
    """Get progress callback for session from global registry."""
    if not session_id:
        return None

    try:
        import sys
        from pathlib import Path

        backend_dir = Path(__file__).parent.parent.parent.parent.parent / "backend"
        project_root = backend_dir.parent

        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from backend import progress_tracker as pt_module

        callback = pt_module.get_progress_callback(session_id)
        print(
            f"[Orchestrator] Got callback for session {session_id}: {callback is not None}"
        )
        return callback
    except Exception as e:
        print(f"[Orchestrator] Failed to get callback: {e}")
        import traceback

        traceback.print_exc()
        return None


async def _run_bundle_async(
    bundle: SubAgentBundle,
    output_base: str,
    global_state: GlobalState,
) -> Dict:
    """Execute a bundle as a single sub-agent session."""
    bundle_output_dir = os.path.join(output_base, bundle.bundle_id)
    os.makedirs(bundle_output_dir, exist_ok=True)

    prompt = _build_bundle_prompt(bundle)

    session_id = global_state.session_id or "default"
    timeout = 600

    progress_callback = _get_progress_callback_by_session(
        global_state.session_id if global_state.session_id else None
    )

    start_ts = time.time()
    result = await OpenCodeExecutor.execute(
        session_id=session_id,
        bundle_id=bundle.bundle_id,
        task=prompt,
        timeout=timeout,
        progress_callback=progress_callback,
        node_name=f"orchestrator_{bundle.bundle_id}",
    )
    elapsed = time.time() - start_ts

    return {
        "success": result.get("status") == "success",
        "session_id": result.get("sandbox_id"),
        "output_dir": bundle_output_dir,
        "summary": (result.get("result", {}).get("stdout", "") or "")[:2000],
        "elapsed": elapsed,
    }


async def orchestrator_node(state: GlobalState) -> GlobalState:
    """Orchestrator node - executes tasks from task_md_content."""
    report_node_start(state, "orchestrator", "Starting orchestrated execution...")

    if not state.task_md_content:
        report_node_complete(state, "orchestrator", "No tasks to execute")
        return state

    if not state.file_paths:
        if state.session_id:
            state.file_paths = _collect_input_files_from_sandbox(
                session_id=state.session_id,
                sandbox_dir=state.sandbox_dir,
            )
        else:
            state.file_paths = {}

    task_md = state.task_md_content

    agent_name = route_task(task_md, ["immune", "rna", "structural", "bioinformatics"])
    if not agent_name:
        agent_name = "bioinformatics"

    output_dir = os.path.join(state.sandbox_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    report_info(
        state,
        f"Executing task from task.md ({len(task_md)} chars) using {agent_name} agent",
        node_name="orchestrator",
    )

    bundle = SubAgentBundle(
        bundle_id="bundle_main",
        agent_name=agent_name,
        task_ids=[],
        combined_content=task_md,
    )

    try:
        result = await _run_bundle_async(
            bundle,
            output_dir,
            state,
        )
        report_node_complete(
            state, "orchestrator", "Task execution completed successfully"
        )

        if not state.merged_result:
            state.merged_result = {}
        state.merged_result["orchestrator_result"] = {
            "status": "completed" if result.get("success") else "failed",
            "bundle_id": bundle.bundle_id,
            "agent_name": agent_name,
            "output_dir": result.get("output_dir"),
            "summary": result.get("summary"),
            "elapsed": result.get("elapsed", 0),
        }
    except Exception as e:
        import traceback

        error_msg = str(e)
        tb_str = traceback.format_exc()
        print(f"[Orchestrator] Execution error: {error_msg}")
        report_error(state, f"Execution error: {error_msg}", node_name="orchestrator")

        if not state.merged_result:
            state.merged_result = {}
        state.merged_result["orchestrator_result"] = {
            "status": "failed",
            "error": error_msg,
            "traceback": tb_str,
        }

    return state
