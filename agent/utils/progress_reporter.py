"""
Progress reporting utility functions (with filtering)
"""

from typing import Optional, Dict, Any, TYPE_CHECKING, Union
import re

if TYPE_CHECKING:
    from state import GlobalState


def should_report_message(message: str, event_type: Optional[str] = None) -> bool:
    """
    Determine whether this message should be reported

    Filters out debug info, API keys, and other content users don't care about

    Args:
        message: Message content
        event_type: Event type, certain types are always reported
    """
    skip_keywords = [
        "API key",
        "has_api_key",
        "connection_config",
        "health_check",
        "kwargs:",
        "Execd endpoint",
        "Nginx",
        "dynamic host port",
        "docker network_mode",
        "polling_interval",
        "[DEBUG]",
        "model_config",
        "ImportError",
        "AttributeError",
        "Traceback",
        "pydantic",
    ]

    for keyword in skip_keywords:
        if keyword.lower() in message.lower():
            return False

    always_report_types = [
        "llm_thinking",
        "llm_reasoning",
        "llm_streaming",
        "subgraph_step",
        "knowledge_retrieval",
        "analysis_progress",
        "tool_result",
    ]

    if event_type and event_type in always_report_types:
        return True

    keep_patterns = [
        r"Node.*started",
        r"Node.*completed",
        r"Node.*executing",
        r"LLM",
        r"Thinking",
        r"Reasoning",
        r"Analyzing",
        r"Retrieving",
        r"Computing",
        r"Validating",
        r"Generating",
        r"Processing",
        r"In progress",
        r"Starting",
        r"Completed",
        r"Success",
        r"Failed",
        r"Task",
        r"Step \d+",
        r"Node \w+",
        r"Calling tool",
        r"Final answer",
        r"Final answer",
        r"Task executing",
        r"Generating",
        r"Analyzing",
        r"Processing",
        r"Successfully",
        r"Retrieving",
        r"Searching",
        r"Fetching",
    ]

    for pattern in keep_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return True

    return False


def report_progress(
    state: "GlobalState",
    event_type: str,
    message: str,
    node_name: Optional[str] = None,
    task_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    progress_percent: Optional[int] = None,
):
    """Report execution progress (with filtering)"""
    if not should_report_message(message, event_type):
        return

    if state and hasattr(state, "progress_callback") and state.progress_callback:
        try:
            state.progress_callback(
                event_type=event_type,
                message=message,
                node_name=node_name,
                task_id=task_id,
                details=details or {},
                progress_percent=progress_percent,
            )
        except Exception as e:
            print(f"[ProgressReporter] Error reporting progress: {e}")


def report_node_start(
    state: "GlobalState", node_name: str, message: Optional[str] = None
):
    """Report node execution started"""
    msg = message or f"Starting {node_name} node"
    report_progress(
        state=state, event_type="node_start", message=msg, node_name=node_name
    )


def report_node_progress(
    state: "GlobalState",
    node_name: str,
    message: str,
    progress_percent: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """Report node execution progress"""
    report_progress(
        state=state,
        event_type="node_progress",
        message=message,
        node_name=node_name,
        progress_percent=progress_percent,
        details=details,
    )


def report_node_complete(
    state: "GlobalState",
    node_name: str,
    message: Optional[str] = None,
    progress_percent: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """Report node execution completed"""
    msg = message or f"Completed {node_name} node"
    report_progress(
        state=state,
        event_type="node_complete",
        message=msg,
        node_name=node_name,
        progress_percent=progress_percent,
        details=details,
    )


def report_task_start(
    state: "GlobalState",
    task_id: str,
    task_description: str,
    node_name: Optional[str] = None,
):
    """Report subtask started"""
    report_progress(
        state=state,
        event_type="task_start",
        message=f"Starting task: {task_description}",
        task_id=task_id,
        node_name=node_name,
        details={"task_description": task_description},
    )


def report_task_progress(
    state: "GlobalState",
    task_id: str,
    message: str,
    progress_percent: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """Report subtask progress"""
    report_progress(
        state=state,
        event_type="task_progress",
        message=message,
        task_id=task_id,
        progress_percent=progress_percent,
        details=details,
    )


def report_task_complete(
    state: "GlobalState", task_id: str, task_description: str, success: bool = True
):
    """Report subtask completed"""
    report_progress(
        state=state,
        event_type="task_complete",
        message=f"Task {'completed' if success else 'failed'}: {task_description}",
        task_id=task_id,
        details={"success": success, "task_description": task_description},
    )


def report_code_generation(
    state: "GlobalState", message: str, details: Optional[Dict[str, Any]] = None
):
    """Report code generation"""
    report_progress(
        state=state, event_type="code_generation", message=message, details=details
    )


def report_code_execution(
    state: "GlobalState", message: str, details: Optional[Dict[str, Any]] = None
):
    """Report code execution"""
    report_progress(
        state=state, event_type="code_execution", message=message, details=details
    )


def report_tool_call(
    state: "GlobalState",
    tool_name: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
):
    """Report tool call"""
    report_progress(
        state=state,
        event_type="tool_call",
        message=message,
        details={"tool_name": tool_name, **(details or {})},
    )


def report_error(
    state: "GlobalState",
    error_message: str,
    node_name: Optional[str] = None,
    task_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """Report error"""
    report_progress(
        state=state,
        event_type="error",
        message=error_message,
        node_name=node_name,
        task_id=task_id,
        details=details,
    )


def report_info(
    state: "GlobalState",
    message: str,
    node_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """Report general information"""
    report_progress(
        state=state,
        event_type="info",
        message=message,
        node_name=node_name,
        details=details,
    )


def report_llm_thinking(
    state: "GlobalState",
    thinking_content: str,
    step_name: str = "Reasoning",
    node_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """
    Report LLM thinking process

    Args:
        state: Global state
        thinking_content: Thinking content (auto-truncated)
        step_name: Step name
        node_name: Node name
        details: Additional details
    """
    truncated = (
        thinking_content[:300] if len(thinking_content) > 300 else thinking_content
    )

    report_progress(
        state=state,
        event_type="llm_thinking",
        message=f"\U0001f4ad {step_name}: {truncated}",
        node_name=node_name,
        details=details or {"step": step_name, "full_content": thinking_content[:1000]},
    )


def report_llm_streaming(
    state: "GlobalState",
    chunk: str,
    accumulated: str = "",
    node_name: Optional[str] = None,
):
    """
    Report LLM streaming output

    Args:
        state: Global state
        chunk: Current text chunk
        accumulated: Accumulated text so far
        node_name: Node name
    """
    report_progress(
        state=state,
        event_type="llm_streaming",
        message=chunk,
        node_name=node_name,
        details={"accumulated_length": len(accumulated)},
    )


def report_llm_reasoning(
    state: "GlobalState",
    reasoning_step: str,
    step_number: int = 0,
    total_steps: int = 0,
    node_name: Optional[str] = None,
):
    """
    Report LLM reasoning step

    Args:
        state: Global state
        reasoning_step: Reasoning step description
        step_number: Current step number
        total_steps: Total number of steps
        node_name: Node name
    """
    step_info = f"[Step {step_number}/{total_steps}] " if total_steps > 0 else ""

    report_progress(
        state=state,
        event_type="llm_reasoning",
        message=f"\U0001f9e0 {step_info}{reasoning_step}",
        node_name=node_name,
        details={"step_number": step_number, "total_steps": total_steps},
        progress_percent=int(
            (step_number / total_steps * 100) if total_steps > 0 else 0
        ),
    )


def report_tool_result(
    state: "GlobalState",
    tool_name: str,
    result_summary: str,
    success: bool = True,
    node_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """
    Report tool call result

    Args:
        state: Global state
        tool_name: Tool name
        result_summary: Result summary
        success: Whether successful
        node_name: Node name
        details: Additional details
    """
    icon = "\u2705" if success else "\u274c"
    truncated_summary = (
        result_summary[:200] if len(result_summary) > 200 else result_summary
    )

    report_progress(
        state=state,
        event_type="tool_result",
        message=f"{icon} Tool [{tool_name}] returned: {truncated_summary}",
        node_name=node_name,
        details={"tool_name": tool_name, "success": success, **(details or {})},
    )


def report_subgraph_step(
    state: "GlobalState",
    subgraph_name: str,
    step_name: str,
    message: str = "",
    progress_percent: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """
    Report subgraph internal step

    Args:
        state: Global state
        subgraph_name: Subgraph name (e.g., general_qa, supervisor)
        step_name: Step name
        message: Step message
        progress_percent: Progress percentage
        details: Additional details
    """
    full_message = f"[{subgraph_name}] {step_name}"
    if message:
        full_message += f" - {message}"

    report_progress(
        state=state,
        event_type="subgraph_step",
        message=full_message,
        node_name=subgraph_name,
        progress_percent=progress_percent,
        details={"subgraph": subgraph_name, "step": step_name, **(details or {})},
    )


def report_knowledge_retrieval(
    state: "GlobalState",
    source: str,
    query: str,
    results_count: int = 0,
    node_name: Optional[str] = None,
):
    """
    Report knowledge retrieval progress

    Args:
        state: Global state
        source: Data source (e.g., Qdrant, Tavily, KnowledgeGraph)
        query: Query content
        results_count: Number of retrieval results
        node_name: Node name
    """
    truncated_query = query[:100] if len(query) > 100 else query

    report_progress(
        state=state,
        event_type="knowledge_retrieval",
        message=f'\U0001f50d Retrieving from {source}: "{truncated_query}" ({results_count} results)',
        node_name=node_name,
        details={"source": source, "results_count": results_count},
    )


def report_analysis_progress(
    state: "GlobalState",
    analysis_type: str,
    current_item: str = "",
    completed: int = 0,
    total: int = 0,
    node_name: Optional[str] = None,
):
    """
    Report analysis progress

    Args:
        state: Global state
        analysis_type: Analysis type (e.g., "Data analysis", "Literature analysis")
        current_item: Current analysis item
        completed: Number of completed items
        total: Total number of items
        node_name: Node name
    """
    progress_msg = f"\U0001f4ca {analysis_type}"
    if current_item:
        truncated_item = current_item[:80] if len(current_item) > 80 else current_item
        progress_msg += f": {truncated_item}"
    if total > 0:
        progress_msg += f" ({completed}/{total})"

    progress_pct = int((completed / total * 100) if total > 0 else 0)

    report_progress(
        state=state,
        event_type="analysis_progress",
        message=progress_msg,
        node_name=node_name,
        progress_percent=progress_pct,
        details={
            "analysis_type": analysis_type,
            "completed": completed,
            "total": total,
        },
    )
