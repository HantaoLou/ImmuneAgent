"""
SSE (Server-Sent Events) Handler for streaming tasks.

This module provides SSE connection handling for MCP tools that return streaming_task responses.
When a tool returns {"type": "streaming_task", "task_id": "...", "service_id": "..."}, 
this module establishes an SSE connection to receive the actual results.

## MCP Tool SSE Message Format (based on nettcr_mcp_server)

Progress messages:
    {
        "type": "progress",
        "data": {
            "session_id": "xxx",
            "status": "loading|predicting|saving|completed",
            "message": "Processing...",
            "progress_percent": 50.0,
            "elapsed_seconds": 30.5,
            "timestamp": 1234567890.0
        }
    }

Result messages:
    {
        "type": "result",
        "status": "success|error",
        "session_id": "xxx",
        "message": "Task completed!",
        "result_path": ["/path/to/output.csv"],
        "statistics": {...},
        "processing_time_ms": 12345
    }

End messages:
    {"type": "end"}

Usage:
    from utils.sse_handler import handle_streaming_task_response

    result = handle_streaming_task_response(
        tool_output=output,
        service_id=service_id,
        timeout=3600,
        progress_callback=lambda msg: print(f"Progress: {msg}")
    )
"""

import json
import os
import re
import time
import ast
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable, List


# MCP servers configuration cache
_mcp_servers_cache: Optional[Dict[str, Any]] = None


def _get_agent_dir() -> Path:
    """Get the agent directory path."""
    # This file is at: agent/utils/sse_handler.py
    # Need to go up 1 level to get agent dir
    return Path(__file__).parent.parent


def _load_mcp_servers() -> Dict[str, Any]:
    """Load MCP servers configuration."""
    global _mcp_servers_cache
    
    if _mcp_servers_cache is not None:
        return _mcp_servers_cache
    
    try:
        agent_dir = _get_agent_dir()
        mcp_servers_path = agent_dir / "config" / "mcp_servers.json"
        
        if not mcp_servers_path.exists():
            print(f"  [WARN] [SSE] MCP servers config not found: {mcp_servers_path}")
            return {}
        
        with open(mcp_servers_path, "r", encoding="utf-8") as f:
            _mcp_servers_cache = json.load(f)
        
        return _mcp_servers_cache
    except Exception as e:
        print(f"  [WARN] [SSE] Failed to load MCP servers config: {e}")
        return {}


def detect_streaming_task(output: Any) -> Optional[Tuple[str, str]]:
    """
    Detect if the output is a streaming_task response.
    
    Args:
        output: Raw output from MCP tool (usually a list of content blocks)
        
    Returns:
        Tuple of (task_id, service_id) if streaming task detected, None otherwise
    """
    try:
        # Parse output, find streaming_task type
        parsed_output = None
        
        # First try to convert output to Python object (if it's a stringified dict)
        if isinstance(output, str):
            try:
                parsed_output = ast.literal_eval(output)
                print(f"  [SSE] Successfully parsed string output as Python object")
            except (ValueError, SyntaxError):
                parsed_output = output
        else:
            parsed_output = output
        
        output_str = str(parsed_output)
        
        # Search for streaming_task keyword
        if "streaming_task" not in output_str:
            return None
        
        print(f"  [SSE] [SSE] Detected output that may contain streaming_task")
        
        # Method 1: If parsed_output is dict, check directly
        if isinstance(parsed_output, dict):
            # Check top level
            if parsed_output.get("type") == "streaming_task":
                task_id = parsed_output.get("task_id")
                service_id = parsed_output.get("service_id")
                if task_id and service_id:
                    print(f"  [SSE] [SSE] Detected streaming task: service_id={service_id}, task_id={task_id}")
                    return (task_id, service_id)
            
            # Check output field
            if "output" in parsed_output:
                output_value = parsed_output["output"]
                if isinstance(output_value, list):
                    for item in output_value:
                        if isinstance(item, dict):
                            item_text = item.get("text", "")
                            if item_text and "streaming_task" in item_text:
                                try:
                                    streaming_info = json.loads(item_text)
                                    if streaming_info.get("type") == "streaming_task":
                                        task_id = streaming_info.get("task_id")
                                        service_id = streaming_info.get("service_id")
                                        if task_id and service_id:
                                            print(f"  [SSE] [SSE] Detected streaming task: service_id={service_id}, task_id={task_id}")
                                            return (task_id, service_id)
                                except json.JSONDecodeError:
                                    # Try regex extraction
                                    task_id_match = re.search(r'"task_id"\s*:\s*"([^"]+)"', item_text)
                                    service_id_match = re.search(r'"service_id"\s*:\s*"([^"]+)"', item_text)
                                    if task_id_match and service_id_match:
                                        return (task_id_match.group(1), service_id_match.group(1))
        
        # Method 2: If parsed_output is list, check elements
        if isinstance(parsed_output, list):
            for item in parsed_output:
                if isinstance(item, dict):
                    item_text = item.get("text", "")
                    if item_text and "streaming_task" in item_text:
                        try:
                            streaming_info = json.loads(item_text)
                            if streaming_info.get("type") == "streaming_task":
                                task_id = streaming_info.get("task_id")
                                service_id = streaming_info.get("service_id")
                                if task_id and service_id:
                                    print(f"  [SSE] [SSE] Detected streaming task: service_id={service_id}, task_id={task_id}")
                                    return (task_id, service_id)
                        except json.JSONDecodeError:
                            task_id_match = re.search(r'"task_id"\s*:\s*"([^"]+)"', item_text)
                            service_id_match = re.search(r'"service_id"\s*:\s*"([^"]+)"', item_text)
                            if task_id_match and service_id_match:
                                return (task_id_match.group(1), service_id_match.group(1))
        
        # Method 3: If still string, try regex extraction directly
        if isinstance(output, str) or (isinstance(parsed_output, str) and parsed_output == output):
            task_id_match = re.search(r'["\']task_id["\']\s*:\s*["\']([^"\']+)["\']', output_str)
            service_id_match = re.search(r'["\']service_id["\']\s*:\s*["\']([^"\']+)["\']', output_str)
            type_match = re.search(r'["\']type["\']\s*:\s*["\']streaming_task["\']', output_str)
            
            if task_id_match and service_id_match and type_match:
                task_id = task_id_match.group(1)
                service_id = service_id_match.group(1)
                print(f"  [SSE] [SSE] Detected streaming task via regex: service_id={service_id}, task_id={task_id}")
                return (task_id, service_id)
        
        return None
    except Exception as e:
        print(f"  [WARN] [SSE] Error detecting streaming task: {e}")
        return None


def _build_sse_url(service_id: str, task_id: str) -> Optional[str]:
    """
    Build SSE URL from service configuration and task ID.
    
    Args:
        service_id: Service ID (e.g., "nettcr")
        task_id: Task ID from streaming_task response
        
    Returns:
        SSE URL or None if cannot build
    """
    mcp_servers = _load_mcp_servers()
    
    if service_id not in mcp_servers:
        print(f"  [WARN] [SSE] Service {service_id} not in configuration")
        return None
    
    server_config = mcp_servers[service_id]
    base_url = server_config.get("url", "")
    
    if not base_url:
        print(f"  [WARN] [SSE] Service {service_id} has no URL configured")
        return None
    
    # Build SSE endpoint URL by replacing /sse with /stream/{task_id}
    # Examples:
    #   http://117.10.59.114:40001/mcp/8088/sse -> http://117.10.59.114:40001/mcp/8088/stream/{task_id}
    #   http://127.0.0.1:8088/sse -> http://127.0.0.1:8088/stream/{task_id}
    if base_url.endswith("/sse"):
        sse_url = base_url[:-4] + f"/stream/{task_id}"
    elif "/sse" in base_url:
        sse_url = base_url.replace("/sse", f"/stream/{task_id}")
    else:
        # Fallback: try to construct from base URL
        from urllib.parse import urlparse
        parsed_url = urlparse(base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if host and port:
            sse_url = f"http://{host}:{port}/stream/{task_id}"
        else:
            print(f"  [WARN] [SSE] Cannot build stream URL from: {base_url}")
            return None
    
    print(f"  [SSE] [SSE] Built SSE URL: {sse_url}")
    return sse_url


# Progress callback type
ProgressCallback = Callable[[Dict[str, Any]], None]

# Global progress callbacks (can be set by external code)
_progress_callbacks: List[ProgressCallback] = []
_progress_callbacks_lock = threading.Lock()


def register_progress_callback(callback: ProgressCallback) -> None:
    """
    Register a global progress callback for all SSE tasks.
    
    Args:
        callback: Function that receives progress messages
    """
    with _progress_callbacks_lock:
        _progress_callbacks.append(callback)


def unregister_progress_callback(callback: ProgressCallback) -> None:
    """Remove a registered progress callback."""
    with _progress_callbacks_lock:
        if callback in _progress_callbacks:
            _progress_callbacks.remove(callback)


def _notify_progress_callbacks(message: Dict[str, Any]) -> None:
    """Notify all registered progress callbacks."""
    with _progress_callbacks_lock:
        for callback in _progress_callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"  [WARN] [SSE] Progress callback error: {e}")


def receive_sse_messages(
    sse_url: str, 
    task_id: str, 
    service_id: str, 
    timeout: int = 3600,
    progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Any]:
    """
    Receive SSE messages until task completion.
    
    This function establishes an SSE connection and blocks until:
    1. A "result" message is received (success)
    2. An "error" or "task_failed" message is received (failure)
    3. An "end" message is received after result (completion)
    4. Timeout is reached (failure)
    
    Args:
        sse_url: SSE endpoint URL (e.g., http://host:port/stream/{task_id})
        task_id: Task ID from streaming_task response
        service_id: Service ID (e.g., "nettcr", "igblast")
        timeout: Timeout in seconds (default: 1 hour)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Processing result dictionary with keys:
        - status: "success" or "failed"
        - output: Task output data (contains result message content)
        - error: Error message if failed
        - messages: List of all received messages
        - progress: Final progress percentage
        - elapsed_seconds: Total elapsed time
    """
    try:
        import requests
    except ImportError:
        return {
            "status": "failed",
            "error": "requests module not installed. Install with: pip install requests",
            "output": None,
            "messages": [],
            "progress": 0,
            "elapsed_seconds": 0
        }
    
    print(f"  [SSE] [SSE] Starting to receive SSE messages: {sse_url}")
    print(f"  [SSE] [SSE] Timeout setting: {timeout} seconds")
    print(f"  [SSE] [SSE] Task ID: {task_id}, Service: {service_id}")
    
    start_time = time.time()
    all_messages: List[Dict[str, Any]] = []
    final_result: Optional[Dict[str, Any]] = None
    task_completed = False
    task_failed = False
    last_progress = 0.0
    last_status = "initializing"
    
    # Establish SSE connection
    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }
    
    # Add API key header if available
    api_key = os.environ.get("OPEN_SANDBOX_API_KEY") or os.environ.get("SANDBOX_API_KEY")
    if api_key:
        headers["OPEN-SANDBOX-API-KEY"] = api_key
        print(f"  [SSE] [SSE] Added API key header for authentication")
    
    try:
        print(f"  [SSE] [SSE] Establishing SSE connection...")
        response = requests.get(sse_url, headers=headers, stream=True, timeout=timeout)
        print(f"  [SSE] [SSE] Received response, status code: {response.status_code}")
        
        # Check response status
        if response.status_code != 200:
            error_msg = f"SSE connection failed with status code {response.status_code}"
            print(f"  [ERR] [SSE] {error_msg}")
            return {
                "status": "failed",
                "error": error_msg,
                "output": {
                    "task_id": task_id,
                    "service_id": service_id,
                    "response_status": response.status_code
                },
                "messages": [],
                "progress": 0,
                "elapsed_seconds": time.time() - start_time
            }
        
        print(f"  [OK] [SSE] SSE connection established, waiting for messages...")
        
        # Read SSE messages line by line
        for line in response.iter_lines(decode_unicode=True):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"  [WARN] [SSE] Message reception timeout after {elapsed:.1f}s")
                break
            
            if not line:
                continue
            
            # Skip heartbeat/comments
            if line.startswith(":"):
                continue
            
            # Parse SSE message format
            # SSE format: data: {...}
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                
                # Handle empty data
                if not data_str.strip():
                    continue
                
                try:
                    message_data = json.loads(data_str)
                    
                    message_type = message_data.get("type", "")
                    data_field = message_data.get("data", {})
                    
                    # Build message content for display
                    if isinstance(data_field, dict):
                        message_content = data_field.get("message", message_data.get("message", ""))
                        last_status = data_field.get("status", last_status)
                        last_progress = data_field.get("progress_percent", last_progress)
                    else:
                        message_content = message_data.get("content", message_data.get("message", ""))
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    message_record = {
                        "timestamp": timestamp,
                        "type": message_type,
                        "content": message_content,
                        "status": last_status,
                        "progress": last_progress,
                        "elapsed_seconds": round(elapsed, 1),
                        "raw": message_data
                    }
                    all_messages.append(message_record)
                    
                    # Print progress (truncated)
                    content_preview = str(message_content)[:100] if message_content else ""
                    if content_preview:
                        print(f"  📨 [SSE] [{timestamp}] {message_type}: {content_preview}")
                    else:
                        print(f"  📨 [SSE] [{timestamp}] {message_type}")
                    
                    # Notify progress callbacks
                    _notify_progress_callbacks(message_record)
                    if progress_callback:
                        try:
                            progress_callback(message_record)
                        except Exception as cb_err:
                            print(f"  [WARN] [SSE] Progress callback error: {cb_err}")
                    
                    # Check task status
                    status = message_data.get("status")
                    if isinstance(data_field, dict):
                        status = status or data_field.get("status")
                    
                    # Handle different message types based on nettcr_mcp_server format
                    if message_type == "result":
                        final_result = message_data
                        result_status = message_data.get("status", "unknown")
                        print(f"  [OK] [SSE] Received result message (status: {result_status})")
                        
                        # Check for error in result status
                        if result_status == "error":
                            task_failed = True
                            error_msg = message_data.get("message", message_data.get("error", "Task failed"))
                            print(f"  [ERR] [SSE] Error in result: {error_msg[:200]}")
                        # Check for error patterns in message content
                        elif message_content and isinstance(message_content, str):
                            content_stripped = message_content.strip()
                            error_patterns = [
                                "Error:", "error:", "ERROR:",
                                "Integration failed", "Processing failed",
                                "Execution failed", "Failed to", "Exception:"
                            ]
                            for pattern in error_patterns:
                                if content_stripped.startswith(pattern) or f"\n{pattern}" in content_stripped:
                                    task_failed = True
                                    print(f"  [ERR] [SSE] Detected error in result: {content_stripped[:200]}")
                                    break
                        
                        if not task_failed:
                            task_completed = True
                    
                    elif message_type == "task_completed" or (message_type == "progress" and status == "completed"):
                        task_completed = True
                        if final_result is None:
                            final_result = message_data
                        last_progress = 100.0
                        print(f"  [OK] [SSE] Task completed successfully")
                    
                    elif message_type == "task_failed" or status == "failed":
                        task_failed = True
                        final_result = message_data
                        error_msg = message_data.get("error", message_data.get("message", "Task failed"))
                        if isinstance(data_field, dict) and not error_msg:
                            error_msg = data_field.get("error", data_field.get("message", "Task failed"))
                        print(f"  [ERR] [SSE] Task failed: {error_msg}")
                    
                    elif message_type == "error":
                        task_failed = True
                        if final_result is None:
                            final_result = message_data
                        error_msg = message_data.get("error", message_data.get("message", "Task failed"))
                        if not error_msg:
                            error_msg = message_content or "Task failed with error"
                        print(f"  [ERR] [SSE] Error: {error_msg}")
                    
                    elif message_type == "end":
                        print(f"  [OK] [SSE] Received end message, stream finished")
                        # End message means stream is complete
                        # If we have a result or saw completion, we're done
                        if task_failed:
                            break
                        elif task_completed or final_result is not None:
                            break
                        else:
                            # End without explicit completion - check if we have useful messages
                            if len(all_messages) > 1:  # More than just the end message
                                # Find the last non-end, non-progress message as result
                                for msg in reversed(all_messages):
                                    if msg.get("type") not in ("end", "progress", "status", "heartbeat"):
                                        final_result = msg.get("raw")
                                        task_completed = True
                                        print(f"  [OK] [SSE] Task ended, using last message as result")
                                        break
                            break
                    
                    elif message_type == "progress" or message_type == "status":
                        # Extract progress info
                        if isinstance(data_field, dict):
                            progress_val = data_field.get("progress_percent") or data_field.get("progress") or data_field.get("percentage")
                            if progress_val is not None:
                                last_progress = float(progress_val)
                            
                            # Extract batch info
                            batch_current = data_field.get("batch_current")
                            batch_total = data_field.get("batch_total")
                            if batch_current is not None and batch_total is not None:
                                print(f"  [SSE] [SSE] Progress: {last_progress}% (batch {batch_current}/{batch_total}) - {data_field.get('message', '')}")
                            else:
                                print(f"  [SSE] [SSE] Progress: {last_progress}% - {data_field.get('message', last_status)}")
                
                except json.JSONDecodeError as json_err:
                    # Not valid JSON, log and continue
                    print(f"  [WARN] [SSE] Could not parse as JSON: {data_str[:100]}...")
                    pass
        
        # Calculate final elapsed time
        final_elapsed = time.time() - start_time
        
        # Determine final result
        if task_failed:
            error_msg = "Task failed"
            if final_result:
                if isinstance(final_result, dict):
                    error_msg = final_result.get("error") or final_result.get("message") or str(final_result.get("data", {}))
                else:
                    error_msg = str(final_result)
            return {
                "status": "failed",
                "error": error_msg,
                "output": final_result,
                "messages": all_messages,
                "progress": last_progress,
                "elapsed_seconds": round(final_elapsed, 1)
            }
        elif task_completed or final_result is not None:
            # Extract useful output from result
            output = final_result
            if isinstance(final_result, dict):
                # Extract key fields for easier consumption
                result_path = final_result.get("result_path", [])
                statistics = final_result.get("statistics", {})
                message = final_result.get("message", "")
                
                # If there's a structured output, prefer that
                if result_path or statistics:
                    output = {
                        "result_path": result_path,
                        "statistics": statistics,
                        "message": message,
                        "raw": final_result
                    }
            
            return {
                "status": "success",
                "output": output,
                "error": None,
                "messages": all_messages,
                "progress": last_progress,
                "elapsed_seconds": round(final_elapsed, 1)
            }
        else:
            return {
                "status": "failed",
                "error": "SSE stream ended without completion or result",
                "output": None,
                "messages": all_messages,
                "progress": last_progress,
                "elapsed_seconds": round(final_elapsed, 1)
            }
    
    except requests.exceptions.Timeout:
        final_elapsed = time.time() - start_time
        return {
            "status": "failed",
            "error": f"SSE connection timeout after {timeout} seconds",
            "output": None,
            "messages": all_messages,
            "progress": last_progress,
            "elapsed_seconds": round(final_elapsed, 1)
        }
    except requests.exceptions.RequestException as e:
        final_elapsed = time.time() - start_time
        return {
            "status": "failed",
            "error": f"SSE connection error: {str(e)}",
            "output": None,
            "messages": all_messages,
            "progress": last_progress,
            "elapsed_seconds": round(final_elapsed, 1)
        }
    except Exception as e:
        import traceback
        final_elapsed = time.time() - start_time
        print(f"  [ERR] [SSE] Exception: {e}")
        print(f"  {traceback.format_exc()[:500]}")
        return {
            "status": "failed",
            "error": f"SSE processing error: {str(e)}",
            "output": None,
            "messages": all_messages,
            "progress": last_progress,
            "elapsed_seconds": round(final_elapsed, 1)
        }


def handle_streaming_task_response(
    tool_output: Any,
    service_id: Optional[str] = None,
    timeout: int = 3600,
    progress_callback: Optional[ProgressCallback] = None
) -> Optional[Dict[str, Any]]:
    """
    Handle streaming_task response by establishing SSE connection and waiting for results.
    
    This is the main entry point for handling streaming tasks.
    
    IMPORTANT: This function BLOCKS until the SSE task completes or fails.
    All MCP tools return streaming_task initially, and this function ensures
    we wait for actual results before returning.
    
    Args:
        tool_output: Raw output from MCP tool call
        service_id: Optional service ID (will be extracted from output if not provided)
        timeout: Timeout in seconds (default: 1 hour)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict with keys:
        - status: "success" or "failed" or None (if not a streaming task)
        - output: Task output data
        - error: Error message if failed
        - messages: List of SSE messages received
        - progress: Final progress percentage
        - elapsed_seconds: Total elapsed time
        Or None if output is not a streaming_task response
    """
    # Detect streaming task
    detected = detect_streaming_task(tool_output)
    
    if detected is None:
        return None
    
    task_id, detected_service_id = detected
    
    # Use provided service_id or detected one
    actual_service_id = service_id or detected_service_id
    
    if not actual_service_id:
        return {
            "status": "failed",
            "error": "Cannot determine service_id for streaming task",
            "output": None,
            "messages": [],
            "progress": 0,
            "elapsed_seconds": 0
        }
    
    print(f"  [SSE] [SSE] Detected streaming_task: {task_id} (service: {actual_service_id})")
    
    # Build SSE URL
    sse_url = _build_sse_url(actual_service_id, task_id)
    
    if not sse_url:
        return {
            "status": "failed",
            "error": f"Cannot build SSE URL for service {actual_service_id}",
            "output": None,
            "messages": [],
            "progress": 0,
            "elapsed_seconds": 0
        }
    
    # Receive SSE messages (this will block until completion)
    return receive_sse_messages(sse_url, task_id, actual_service_id, timeout, progress_callback)

