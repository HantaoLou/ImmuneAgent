"""Unified tool-call interface for CodeAct and future agents."""

from typing import Any, Dict, Optional
import time

from utils.mcp_helper import invoke_mcp_tool_sync


def call_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    service_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Call an MCP tool with a unified response schema.

    Returns a dict with:
    status/output/error/error_type/execution_time_ms/tool_name/service_id
    """
    start_time = time.monotonic()
    result: Dict[str, Any] = {}

    try:
        result = invoke_mcp_tool_sync(tool_name=tool_name, parameters=parameters, config=config)
        status = result.get("status", "failed")
        output = result.get("output")
        error = result.get("error")
        error_type = result.get("error_type")
        if status != "success" and not error_type:
            error_type = "ToolError"
    except Exception as exc:
        status = "failed"
        output = None
        error = str(exc)
        error_type = type(exc).__name__

    execution_time_ms = int((time.monotonic() - start_time) * 1000)
    return {
        "status": status,
        "output": output,
        "error": error,
        "error_type": error_type,
        "execution_time_ms": execution_time_ms,
        "tool_name": tool_name,
        "service_id": service_id or (result.get("service_id") if isinstance(result, dict) else None)
    }

