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

from state import GlobalState, SubTask, UserTaskType
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


def _build_bundle_prompt(
    bundle: SubAgentBundle,
    output_dir: str,
    orch: OrchestratorState = None,
    completed_bundles: Dict[str, SubAgentBundle] = None,
) -> str:
    """Build a single prompt for the entire bundle workflow."""
    input_context = ""
    if orch:
        ctx_parts = []
        if orch.file_paths:
            files_str = ", ".join(orch.file_paths.values())
            ctx_parts.append(f"Files: {files_str}")
        if orch.user_input:
            ctx_parts.append(f"Original User Request: {orch.user_input}")
        if ctx_parts:
            input_context = "\n## Input Context\n" + "\n".join(ctx_parts) + "\n"

    upstream_section = ""
    if completed_bundles and bundle.dependencies:
        upstream_parts = []
        for dep_id in bundle.dependencies:
            dep_bundle = completed_bundles.get(dep_id)
            if dep_bundle and dep_bundle.result:
                dep_output_dir = dep_bundle.result.get("output_dir", "")
                dep_summary = dep_bundle.result.get("summary", "")[:500]
                upstream_parts.append(
                    f"### From {dep_id} ({dep_bundle.agent_name})\n"
                    f"- Output directory: {dep_output_dir}\n"
                    f"- Summary: {dep_summary}"
                )
        if upstream_parts:
            upstream_section = (
                "\n## Upstream Results (from dependent bundles)\n"
                + "\n\n".join(upstream_parts)
                + "\n"
            )

    domain_desc = get_domain_description(bundle.agent_name)
    n = len(bundle.task_ids)

    return f"""Execute the following {n} task{"s" if n > 1 else ""} as a complete {domain_desc} workflow.
Output directory: {output_dir}/
{input_context}{upstream_section}
## Workflow Steps (execute in order, pass results between steps)

{bundle.combined_content}

## Rules
1. Execute all steps in the order listed. Results from earlier steps feed into later steps.
2. Use native MCP tools - never fabricate results.
3. Write all output files to {output_dir}/
4. If a tool call fails, log the error and continue to the next step.
5. After each step, briefly state what was produced before moving on."""


def orchestrator_node(state: GlobalState) -> GlobalState:
    """Orchestrator node - bundles tasks by domain and dispatches each bundle."""
    report_node_start(state, "orchestrator", "Starting orchestrated execution...")

    if not state.subtasks:
        report_node_complete(state, "orchestrator", "No tasks to execute")
        return state

    if not state.file_paths:
        state.file_paths = _collect_input_files_from_sandbox(
            session_id=state.session_id,
            sandbox_dir=state.sandbox_dir,
        )

    orch = _build_orchestrator_state(state)

    if len(orch.assignments) < SIMPLE_TASK_THRESHOLD:
        _bundle_single_agent(orch)
        logger.info(
            "Simple-task mode: %d tasks < %d threshold, single bundle",
            len(orch.assignments),
            SIMPLE_TASK_THRESHOLD,
        )
    else:
        _bundle_assignments(orch)

    output_base = os.path.join(state.sandbox_dir, "output")
    os.makedirs(output_base, exist_ok=True)
    progress_callback = state.progress_callback

    total_bundles = len(orch.bundles)
    total_tasks = len(orch.assignments)
    report_info(
        state,
        f"Orchestrating {total_tasks} tasks in {total_bundles} domain bundle(s)",
        node_name="orchestrator",
    )
    for b in orch.bundles:
        logger.info(
            "Bundle %s (%s): %d tasks %s deps=%s",
            b.bundle_id,
            b.agent_name,
            len(b.task_ids),
            b.task_ids,
            b.dependencies,
        )

    loop = asyncio.new_event_loop()
    try:
        while orch.current_step < orch.max_steps:
            orch.current_step += 1

            pending, ready, failed_retriable, completed, failed = _observe_bundles(orch)

            orch.react_log.append(
                {
                    "step": orch.current_step,
                    "phase": "observe",
                    "pending": len(pending),
                    "ready": len(ready),
                    "failed_retriable": len(failed_retriable),
                    "completed": len(completed),
                    "failed": len(failed),
                }
            )

            if not ready and not failed_retriable:
                is_stuck = len(pending) > 0
                orch.react_log.append(
                    {
                        "step": orch.current_step,
                        "phase": "reason",
                        "decision": "finish" if not is_stuck else "stuck",
                    }
                )
                if is_stuck:
                    stuck_ids = [b.bundle_id for b in pending]
                    logger.warning(
                        "Orchestrator stuck: %d bundle(s) blocked by failed deps: %s",
                        len(pending),
                        stuck_ids,
                    )
                    report_error(
                        state,
                        f"Orchestrator stuck: {len(pending)} bundle(s) blocked "
                        f"by failed dependencies: {stuck_ids}",
                        node_name="orchestrator",
                    )
                    for b in pending:
                        b.status = OrchestratorTaskStatus.FAILED
                        b.error = "Blocked by failed upstream dependency"
                        _sync_bundle_to_assignments(orch, b)
                break

            for b in failed_retriable:
                b.status = OrchestratorTaskStatus.RETRYING
                b.attempt += 1

            dispatch_list = ready + failed_retriable

            orch.react_log.append(
                {
                    "step": orch.current_step,
                    "phase": "reason",
                    "decision": "dispatch",
                    "dispatch_count": len(dispatch_list),
                    "bundles": [b.bundle_id for b in dispatch_list],
                }
            )

            done_count = len(completed)
            pct = int((done_count / total_bundles) * 80) + 10
            report_node_progress(
                state,
                "orchestrator",
                f"Dispatching {len(dispatch_list)} bundle(s) "
                f"(step {orch.current_step}, {done_count}/{total_bundles} done)",
                progress_percent=min(pct, 90),
            )

            _dispatch_bundles(orch, dispatch_list, output_base, state, loop)

        for b in orch.bundles:
            if b.status not in (
                OrchestratorTaskStatus.COMPLETED,
                OrchestratorTaskStatus.FAILED,
            ):
                b.status = OrchestratorTaskStatus.FAILED
                b.error = f"Orchestrator reached max steps ({orch.max_steps}) before completion"
                _sync_bundle_to_assignments(orch, b)
    finally:
        loop.close()

    return _map_results_to_global(orch, state)


def _build_orchestrator_state(state: GlobalState) -> OrchestratorState:
    available = ["immune", "rna", "structural", "bioinformatics"]
    assignments = []
    dep_map: Dict[str, List[str]] = {}
    pg_map: Dict[str, List[str]] = {}
    seen_ids: set = set()

    def _add_subtask(st: SubTask) -> None:
        if st.task_id in seen_ids:
            return
        seen_ids.add(st.task_id)

        task_tools = []
        if isinstance(st.result, dict):
            raw_tools = st.result.get("tools", [])
            task_tools = [
                t["tool_name"] if isinstance(t, dict) else t for t in raw_tools
            ]

        agent_name = (
            route_task_by_tools(task_tools)
            if task_tools
            else route_task(st.content, available)
        )
        if not agent_name or agent_name not in available:
            agent_name = route_task(st.content, available)

        a = SubAgentAssignment(
            task_id=st.task_id,
            agent_name=agent_name,
            task_content=st.content,
            parallel_group_id=st.parallel_group_id,
            dependencies=list(st.dependencies),
            task_tools=task_tools,
        )
        assignments.append(a)

        if st.dependencies:
            dep_map[st.task_id] = list(st.dependencies)
        if st.parallel_group_id:
            pg_map.setdefault(st.parallel_group_id, []).append(st.task_id)

    for st in state.subtasks:
        _add_subtask(st)

    return OrchestratorState(
        session_id=state.session_id or "",
        user_input=state.user_input,
        execution_plan=state.execution_plan,
        file_paths=getattr(state, "file_paths", {}) or {},
        assignments=assignments,
        parallel_groups=pg_map,
        dependency_map=dep_map,
    )


def _bundle_assignments(orch: OrchestratorState) -> None:
    """Group assignments by agent_name into bundles. One bundle per domain."""
    agent_tasks: Dict[str, List[SubAgentAssignment]] = {}
    for a in orch.assignments:
        agent_tasks.setdefault(a.agent_name, []).append(a)

    task_to_bundle: Dict[str, str] = {}

    for agent_name, assignments in agent_tasks.items():
        bundle_id = f"bundle_{agent_name}"

        sorted_assignments = _topo_sort_assignments(assignments, orch.dependency_map)

        task_ids = [a.task_id for a in sorted_assignments]

        content_parts = []
        for i, a in enumerate(sorted_assignments, 1):
            step_lines = [f"### Step {i}: {a.task_id}", a.task_content]
            if a.task_tools:
                step_lines.append(f"- Tools: {', '.join(a.task_tools)}")
            content_parts.append("\n".join(step_lines))
        combined = "\n\n".join(content_parts)

        bundle = SubAgentBundle(
            bundle_id=bundle_id,
            agent_name=agent_name,
            task_ids=task_ids,
            combined_content=combined,
        )
        orch.bundles.append(bundle)

        for tid in task_ids:
            task_to_bundle[tid] = bundle_id

    for bundle in orch.bundles:
        deps: set = set()
        for tid in bundle.task_ids:
            for dep_tid in orch.dependency_map.get(tid, []):
                dep_bundle = task_to_bundle.get(dep_tid)
                if dep_bundle and dep_bundle != bundle.bundle_id:
                    deps.add(dep_bundle)
        if deps:
            bundle.dependencies = sorted(deps)
            orch.bundle_dependency_map[bundle.bundle_id] = bundle.dependencies


def _bundle_single_agent(orch: OrchestratorState) -> None:
    """Bundle ALL assignments into a single 'general' bundle for simple tasks."""
    sorted_assignments = _topo_sort_assignments(orch.assignments, orch.dependency_map)

    for a in sorted_assignments:
        a.agent_name = "general"

    task_ids = [a.task_id for a in sorted_assignments]

    content_parts = []
    for i, a in enumerate(sorted_assignments, 1):
        step_lines = [f"### Step {i}: {a.task_id}", a.task_content]
        if a.task_tools:
            step_lines.append(f"- Tools: {', '.join(a.task_tools)}")
        content_parts.append("\n".join(step_lines))
    combined = "\n\n".join(content_parts)

    bundle = SubAgentBundle(
        bundle_id="bundle_general",
        agent_name="general",
        task_ids=task_ids,
        combined_content=combined,
    )
    orch.bundles.append(bundle)


def _topo_sort_assignments(
    assignments: List[SubAgentAssignment],
    dep_map: Dict[str, List[str]],
) -> List[SubAgentAssignment]:
    """Topological sort assignments by intra-group dependencies."""
    ids_in_group = {a.task_id for a in assignments}
    by_id = {a.task_id: a for a in assignments}

    in_degree: Dict[str, int] = {a.task_id: 0 for a in assignments}
    dependents: Dict[str, List[str]] = {a.task_id: [] for a in assignments}

    for a in assignments:
        for dep in dep_map.get(a.task_id, []):
            if dep in ids_in_group:
                in_degree[a.task_id] += 1
                dependents[dep].append(a.task_id)

    queue = [a.task_id for a in assignments if in_degree[a.task_id] == 0]
    result = []
    while queue:
        tid = queue.pop(0)
        result.append(by_id[tid])
        for child in dependents[tid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(assignments):
        logger.warning("Cycle detected in intra-bundle deps, using original order")
        return assignments
    return result


def _observe_bundles(
    orch: OrchestratorState,
) -> Tuple[
    List[SubAgentBundle],
    List[SubAgentBundle],
    List[SubAgentBundle],
    List[SubAgentBundle],
    List[SubAgentBundle],
]:
    """Returns (pending, ready, failed_retriable, completed, failed_terminal)"""
    completed_ids = {
        b.bundle_id
        for b in orch.bundles
        if b.status == OrchestratorTaskStatus.COMPLETED
    }

    pending = []
    ready = []
    failed_retriable = []
    completed = []
    failed_terminal = []

    for b in orch.bundles:
        if b.status == OrchestratorTaskStatus.COMPLETED:
            completed.append(b)
        elif b.status == OrchestratorTaskStatus.FAILED:
            if b.attempt < b.max_attempts:
                failed_retriable.append(b)
            else:
                failed_terminal.append(b)
        elif b.status == OrchestratorTaskStatus.DISPATCHED:
            b.status = OrchestratorTaskStatus.FAILED
            b.error = b.error or "Dispatch interrupted (orphaned DISPATCHED status)"
            if b.attempt < b.max_attempts:
                failed_retriable.append(b)
            else:
                failed_terminal.append(b)
        elif b.status in (
            OrchestratorTaskStatus.PENDING,
            OrchestratorTaskStatus.RETRYING,
        ):
            deps = orch.bundle_dependency_map.get(b.bundle_id, [])
            if all(d in completed_ids for d in deps):
                ready.append(b)
            else:
                pending.append(b)

    return pending, ready, failed_retriable, completed, failed_terminal


def _dispatch_bundles(
    orch: OrchestratorState,
    bundles: List[SubAgentBundle],
    output_base: str,
    global_state: GlobalState,
    loop: asyncio.AbstractEventLoop,
):
    """Dispatch bundles - parallel via asyncio.gather when multiple are ready."""
    completed_map: Dict[str, SubAgentBundle] = {
        b.bundle_id: b
        for b in orch.bundles
        if b.status == OrchestratorTaskStatus.COMPLETED
    }

    for b in bundles:
        b.status = OrchestratorTaskStatus.DISPATCHED
        b.attempt = max(b.attempt, 1)

    if len(bundles) == 1:
        b = bundles[0]
        try:
            result = loop.run_until_complete(
                _run_bundle_async(b, output_base, global_state, orch, completed_map)
            )
            b.status = OrchestratorTaskStatus.COMPLETED
            b.result = result
            b.elapsed = result.get("elapsed", 0.0)
            _sync_bundle_to_assignments(orch, b)
        except Exception as e:
            b.status = OrchestratorTaskStatus.FAILED
            b.error = str(e)
            _sync_bundle_to_assignments(orch, b)
        return

    async def _gather():
        coros = [
            _run_bundle_async(b, output_base, global_state, orch, completed_map)
            for b in bundles
        ]
        return await asyncio.gather(*coros, return_exceptions=True)

    try:
        results = loop.run_until_complete(_gather())
    except Exception as e:
        for b in bundles:
            if b.status == OrchestratorTaskStatus.DISPATCHED:
                b.status = OrchestratorTaskStatus.FAILED
                b.error = f"Parallel dispatch failed: {e}"
                _sync_bundle_to_assignments(orch, b)
        return

    for b, result in zip(bundles, results):
        if isinstance(result, Exception):
            b.status = OrchestratorTaskStatus.FAILED
            b.error = str(result)
        else:
            b.status = OrchestratorTaskStatus.COMPLETED
            b.result = result
            b.elapsed = result.get("elapsed", 0.0)
        _sync_bundle_to_assignments(orch, b)


async def _run_bundle_async(
    bundle: SubAgentBundle,
    output_base: str,
    global_state: GlobalState,
    orch: OrchestratorState = None,
    completed_bundles: Dict[str, SubAgentBundle] = None,
) -> Dict:
    """Execute a bundle as a single sub-agent session."""
    bundle_output_dir = os.path.join(output_base, bundle.bundle_id)
    os.makedirs(bundle_output_dir, exist_ok=True)

    prompt = _build_bundle_prompt(bundle, bundle_output_dir, orch, completed_bundles)

    session_id = global_state.session_id or "default"
    timeout = max(900, 900 * len(bundle.task_ids))

    start_ts = time.time()
    result = await OpenCodeExecutor.execute(
        session_id=session_id,
        bundle_id=bundle.bundle_id,
        task=prompt,
        timeout=timeout,
        progress_callback=global_state.progress_callback,
        node_name=f"orchestrator_{bundle.bundle_id}",
    )
    elapsed = time.time() - start_ts

    return {
        "success": result.get("status") == "success",
        "session_id": result.get("sandbox_id"),
        "output_dir": bundle_output_dir,
        "summary": (result.get("result", {}).get("stdout", "") or "")[:2000],
        "elapsed": elapsed,
        "task_count": len(bundle.task_ids),
    }


def _sync_bundle_to_assignments(
    orch: OrchestratorState,
    bundle: SubAgentBundle,
) -> None:
    """Propagate bundle status to all its individual assignments."""
    tid_set = set(bundle.task_ids)
    for a in orch.assignments:
        if a.task_id not in tid_set:
            continue
        if bundle.status == OrchestratorTaskStatus.COMPLETED:
            a.status = OrchestratorTaskStatus.COMPLETED
            a.result = bundle.result
            a.elapsed = bundle.elapsed
        elif bundle.status == OrchestratorTaskStatus.FAILED:
            a.status = OrchestratorTaskStatus.FAILED
            a.error = bundle.error


def _map_results_to_global(
    orch: OrchestratorState,
    state: GlobalState,
) -> GlobalState:
    completed_tasks: Dict[str, SubTask] = {}
    total = len(orch.assignments)
    completed_count = 0
    failed_count = 0

    for a in orch.assignments:
        if a.status == OrchestratorTaskStatus.COMPLETED:
            completed_count += 1
            st = SubTask(
                task_id=a.task_id,
                task_type=UserTaskType.EXECUTE_PLAN,
                content=a.task_content,
                dependencies=a.dependencies,
                parallel_group_id=a.parallel_group_id,
            )
            st.result = {
                "status": "completed",
                "agent": a.agent_name,
                "output": a.result,
                "elapsed": a.elapsed,
            }
            completed_tasks[a.task_id] = st
        elif a.status == OrchestratorTaskStatus.FAILED:
            failed_count += 1
            st = SubTask(
                task_id=a.task_id,
                task_type=UserTaskType.EXECUTE_PLAN,
                content=a.task_content,
                dependencies=a.dependencies,
                parallel_group_id=a.parallel_group_id,
            )
            st.result = {
                "status": "failed",
                "agent": a.agent_name,
                "error": a.error,
                "attempts": a.attempt,
            }
            completed_tasks[a.task_id] = st

    state.completed_tasks = completed_tasks

    if not state.merged_result:
        state.merged_result = {}

    state.merged_result["executor_results"] = {
        "total_tasks": total,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "mode": "orchestrator_simple"
        if len(orch.bundles) == 1 and orch.bundles[0].agent_name == "general"
        else "orchestrator",
        "react_steps": orch.current_step,
        "bundles": [
            {
                "bundle_id": b.bundle_id,
                "agent": b.agent_name,
                "tasks": b.task_ids,
                "status": b.status.value,
                "elapsed": b.elapsed,
            }
            for b in orch.bundles
        ],
    }

    report_node_complete(
        state,
        "orchestrator",
        f"Orchestration done: {completed_count}/{total} completed, "
        f"{failed_count} failed, {len(orch.bundles)} bundle(s)",
    )

    return state
