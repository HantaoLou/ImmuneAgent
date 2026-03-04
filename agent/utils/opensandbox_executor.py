"""
OpenSandbox execution helper for CodeAct.

This module provides a thin wrapper around the OpenSandbox SDK to execute
generated code in a remote sandbox when configured.

Supports ConnectionConfig for advanced connection settings including:
- Custom domain/endpoint
- API key authentication
- Request timeout configuration
"""

from __future__ import annotations

import asyncio
import base64
import time
import urllib.request
import urllib.error
import json
import os
from datetime import timedelta
from typing import Any, Dict, Optional
from pathlib import Path
import sys


def is_opensandbox_enabled() -> bool:
    """Check whether OpenSandbox should be used for CodeAct execution."""
    provider = os.getenv("CODEACT_SANDBOX_PROVIDER", "").lower()
    if provider == "opensandbox":
        return True
    return os.getenv("OPENSANDBOX_ENABLED", "false").lower() == "true"


def _load_env_json(env_key: str) -> Dict[str, Any]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _collect_log_text(log_entries: Optional[Any]) -> str:
    if not log_entries:
        return ""
    if isinstance(log_entries, str):
        return log_entries
    texts = []
    for entry in log_entries:
        text = getattr(entry, "text", None)
        if text is None:
            text = str(entry)
        texts.append(text)
    return "\n".join(texts)


def _extract_logs(execution: Any) -> tuple[str, str]:
    logs = getattr(execution, "logs", None)
    if not logs:
        return "", ""
    stdout = _collect_log_text(getattr(logs, "stdout", None))
    stderr = _collect_log_text(getattr(logs, "stderr", None))
    return stdout, stderr


def _format_execution(execution: Any) -> str:
    """
    Format execution output similar to the example code.
    Combines stdout, stderr, and error information.
    """
    stdout = _collect_log_text(getattr(getattr(execution, "logs", None), "stdout", None))
    stderr = _collect_log_text(getattr(getattr(execution, "logs", None), "stderr", None))
    
    if hasattr(execution, "error") and execution.error:
        error_name = getattr(execution.error, "name", "Error")
        error_value = getattr(execution.error, "value", str(execution.error))
        stderr = "\n".join(
            [
                stderr,
                f"[error] {error_name}: {error_value}",
            ]
        ).strip()
    
    output = stdout.strip()
    if stderr:
        output = "\n".join([output, f"[stderr]\n{stderr}"]).strip()
    return output or "(no output)"


def _get_agent_dir() -> Path:
    """Get agent directory path."""
    # This file is at agent/utils/opensandbox_executor.py
    # Agent dir is parent of utils
    return Path(__file__).parent.parent


def _normalize_proxy_base(base: str) -> str:
    base = base.strip()
    if base.startswith("http://"):
        base = base[len("http://"):]
    elif base.startswith("https://"):
        base = base[len("https://"):]
    return base.rstrip("/")


def _build_proxy_endpoint(current_endpoint: str, proxy_base: str, execd_port: int) -> str:
    proxy_base = _normalize_proxy_base(proxy_base)
    if "/proxy/" in current_endpoint:
        suffix = current_endpoint.split("/proxy/", 1)[1].lstrip("/")
        prefix = current_endpoint.split("/proxy/", 1)[0].rstrip("/")
        host_port = prefix.split("/")[-1]
        if ":" in host_port:
            host_port = host_port.split(":", 1)[1]
        return f"{proxy_base}/{host_port}/proxy/{suffix}"
    return f"{proxy_base}/proxy/{execd_port}"


def _apply_execd_proxy_override(sandbox: Any, proxy_base: str) -> None:
    """Rewrite execd endpoint to go through a fixed API proxy base."""
    try:
        from opensandbox.adapters.factory import AdapterFactory
        from opensandbox.constants import DEFAULT_EXECD_PORT
        from opensandbox.models.sandboxes import SandboxEndpoint
    except Exception:
        return

    if not proxy_base:
        return

    # Try to reuse any existing endpoint string for suffix extraction.
    current_endpoint = None
    for attr in ("_command_service", "_filesystem_service", "_health_service", "_metrics_service"):
        service = getattr(sandbox, attr, None)
        if service is not None and hasattr(service, "execd_endpoint"):
            endpoint_obj = getattr(service, "execd_endpoint", None)
            if endpoint_obj and hasattr(endpoint_obj, "endpoint"):
                current_endpoint = endpoint_obj.endpoint
                break

    if not current_endpoint:
        current_endpoint = f"127.0.0.1:{DEFAULT_EXECD_PORT}"

    new_endpoint_value = _build_proxy_endpoint(
        current_endpoint=current_endpoint,
        proxy_base=proxy_base,
        execd_port=DEFAULT_EXECD_PORT,
    )
    print(f"[opensandbox] Override execd endpoint: {current_endpoint} -> {new_endpoint_value}")

    new_endpoint = SandboxEndpoint(endpoint=new_endpoint_value)
    factory = AdapterFactory(sandbox.connection_config)
    sandbox._filesystem_service = factory.create_filesystem_service(new_endpoint)
    sandbox._command_service = factory.create_command_service(new_endpoint)
    sandbox._health_service = factory.create_health_service(new_endpoint)
    sandbox._metrics_service = factory.create_metrics_service(new_endpoint)


async def _setup_mcp_support_in_sandbox(sandbox: Any) -> None:
    """
    Setup MCP support in OpenSandbox by installing dependencies and copying config files.
    
    This allows MCP tool calls to work in OpenSandbox environment.
    """
    try:
        agent_dir = _get_agent_dir()
        config_dir = agent_dir / "config"
        
        async def _run_and_capture(cmd: str, label: str) -> tuple[str, str]:
            execution = await sandbox.commands.run(cmd)
            stdout, stderr = _extract_logs(execution)
            if stdout:
                print(f"  [{label}] stdout:\n{stdout}")
            if stderr:
                print(f"  [{label}] stderr:\n{stderr}")
            return stdout, stderr

        # 1. MCP dependencies must be pre-installed in the sandbox image
        # Removed auto-install logic - image should have langchain-mcp-adapters pre-installed
        print("  ℹ Assuming MCP dependencies are pre-installed in the sandbox image")
        
        # 2. Create config directory in sandbox
        await sandbox.commands.run("mkdir -p /tmp/agent_config")
        
        # 3. Copy MCP configuration files
        from opensandbox.models import WriteEntry
        write_entries = []
        
        config_files = ["mcp_servers.json", "mcp_tools.json"]
        for config_file in config_files:
            config_path = config_dir / config_file
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_content = f.read()
                sandbox_config_path = f"/tmp/agent_config/{config_file}"
                write_entries.append(WriteEntry(path=sandbox_config_path, data=config_content, mode=0o644))
        
        # 4. Copy necessary Python modules (core, utils, tools)
        # Create module directories first
        await sandbox.commands.run("mkdir -p /tmp/agent_modules/core /tmp/agent_modules/utils /tmp/agent_modules/tools")
        
        # Copy core module
        core_dir = agent_dir / "core"
        if core_dir.exists():
            for py_file in core_dir.glob("*.py"):
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                sandbox_file = f"/tmp/agent_modules/core/{py_file.name}"
                write_entries.append(WriteEntry(path=sandbox_file, data=content, mode=0o644))
        
        # Copy utils module (selective - only mcp_helper)
        utils_dir = agent_dir / "utils"
        utils_files_to_copy = ["mcp_helper.py"]
        if utils_dir.exists():
            for py_file in utils_files_to_copy:
                utils_file = utils_dir / py_file
                if utils_file.exists():
                    with open(utils_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Fix import paths for sandbox environment
                    # Replace agent_dir and config path resolution for sandbox
                    import re
                    # Replace agent_dir definition
                    content = re.sub(
                        r'agent_dir\s*=\s*Path\(__file__\)\.parent\.parent',
                        'agent_dir = Path("/tmp")  # Sandbox: config is at /tmp/agent_config',
                        content
                    )
                    # Replace config path resolution in _load_tool_to_service_map
                    content = re.sub(
                        r'mcp_tools_path\s*=\s*agent_dir\s*/\s*"config"\s*/\s*"mcp_tools\.json"',
                        'mcp_tools_path = Path("/tmp/agent_config") / "mcp_tools.json"',
                        content
                    )
                    # Replace config path in _get_single_server_client and _get_mcp_client
                    content = re.sub(
                        r'mcp_servers_path\s*=\s*agent_dir\s*/\s*"config"\s*/\s*"mcp_servers\.json"',
                        'mcp_servers_path = Path("/tmp/agent_config") / "mcp_servers.json"',
                        content
                    )
                    sandbox_file = f"/tmp/agent_modules/utils/{py_file}"
                    write_entries.append(WriteEntry(path=sandbox_file, data=content, mode=0o644))
        
        # 5. Copy tools module (for data transformation functions like convert_tcr_to_nettcr_format)
        # This is needed by Revision mechanism for data_transform strategy
        tools_dir = agent_dir / "tools"
        if tools_dir.exists():
            # Copy reference.py (contains convert_tcr_to_nettcr_format) and _output.py (dependency)
            tools_files_to_copy = ["reference.py", "_output.py", "__init__.py"]
            for py_file in tools_files_to_copy:
                tools_file = tools_dir / py_file
                if tools_file.exists():
                    with open(tools_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Fix import paths for sandbox environment
                    import re
                    # Replace config path for IMGT reference file
                    content = re.sub(
                        r'config_path\s*=\s*Path\(__file__\)\.parent\.parent\s*/\s*"config"\s*/\s*"imgt_vgene_cdr_reference\.json"',
                        'config_path = Path("/tmp/agent_config") / "imgt_vgene_cdr_reference.json"',
                        content
                    )
                    sandbox_file = f"/tmp/agent_modules/tools/{py_file}"
                    write_entries.append(WriteEntry(path=sandbox_file, data=content, mode=0o644))
            
            # Copy IMGT reference data file (needed for CDR inference)
            imgt_ref_file = config_dir / "imgt_vgene_cdr_reference.json"
            if imgt_ref_file.exists():
                with open(imgt_ref_file, "r", encoding="utf-8") as f:
                    imgt_content = f.read()
                write_entries.append(WriteEntry(
                    path="/tmp/agent_config/imgt_vgene_cdr_reference.json",
                    data=imgt_content,
                    mode=0o644
                ))
        
        # 6. Create __init__.py files
        write_entries.append(WriteEntry(path="/tmp/agent_modules/__init__.py", data="", mode=0o644))
        write_entries.append(WriteEntry(path="/tmp/agent_modules/core/__init__.py", data="", mode=0o644))
        write_entries.append(WriteEntry(path="/tmp/agent_modules/utils/__init__.py", data="", mode=0o644))
        write_entries.append(WriteEntry(path="/tmp/agent_modules/tools/__init__.py", data="", mode=0o644))
        
        # 7. Create a setup script
        setup_script = """#!/usr/bin/env python
import sys
import os
from pathlib import Path

# Add agent modules to path
sys.path.insert(0, '/tmp/agent_modules')
sys.path.insert(0, '/tmp/agent_modules/third_party')

# Set config directory environment variable
os.environ['AGENT_CONFIG_DIR'] = '/tmp/agent_config'

# Fix mcp_helper to use sandbox config
try:
    import utils.mcp_helper as mcp_helper_module
    if hasattr(mcp_helper_module, 'agent_dir'):
        mcp_helper_module.agent_dir = Path('/tmp/agent_config').parent
except ImportError:
    pass

print("MCP support setup completed")
"""
        write_entries.append(WriteEntry(path="/tmp/setup_mcp.py", data=setup_script, mode=0o755))
        
        def _entry_bytes(entry: Any) -> tuple[bytes, str]:
            encoding = getattr(entry, "encoding", None) or "utf-8"
            data = entry.data
            if isinstance(data, bytes):
                return data, encoding
            if isinstance(data, str):
                return data.encode(encoding), encoding
            try:
                # IOBase or file-like
                raw = data.read()
                if isinstance(raw, str):
                    return raw.encode(encoding), encoding
                return raw, encoding
            except Exception:
                return b"", encoding

        async def _write_file_via_command(entry: Any) -> None:
            content_bytes, _encoding = _entry_bytes(entry)
            b64 = base64.b64encode(content_bytes).decode("ascii")
            path_json = json.dumps(entry.path)
            mode = entry.mode if getattr(entry, "mode", None) else 0o644
            cmd = f"""python3 - <<'PY'
import base64, os
path = {path_json}
data = base64.b64decode({json.dumps(b64)})
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "wb") as f:
    f.write(data)
os.chmod(path, {mode})
PY"""
            await sandbox.commands.run(cmd)

        # Write via command first (avoids /files/upload 500 behind proxy), then fallback to API uploads
        if write_entries:
            try:
                for entry in write_entries:
                    await _write_file_via_command(entry)
                print(f"  ✓ Copied {len(write_entries)} files to sandbox (command)")
            except Exception as cmd_exc:
                print(f"  ⚠ Command-based write failed, retrying API upload: {cmd_exc}")
                try:
                    await sandbox.files.write_files(write_entries)
                    print(f"  ✓ Copied {len(write_entries)} files to sandbox (bulk upload)")
                except Exception as bulk_exc:
                    print(f"  ⚠ Bulk file upload failed, retrying per-file: {bulk_exc}")
                    for entry in write_entries:
                        await sandbox.files.write_file(
                            entry.path,
                            entry.data,
                            mode=entry.mode,
                            owner=entry.owner,
                            group=entry.group,
                            encoding=getattr(entry, "encoding", "utf-8"),
                        )
                    print(f"  ✓ Copied {len(write_entries)} files to sandbox (per-file upload)")
        await sandbox.commands.run("python /tmp/setup_mcp.py")
        print("  ✓ MCP support setup completed in sandbox")
        
    except Exception as e:
        print(f"  ⚠ Failed to setup MCP support in sandbox: {e}")
        # Don't fail completely, just warn


def _get_connection_config() -> Optional[Any]:
    """
    Get ConnectionConfig for OpenSandbox if configured.
    Returns None if using default configuration.
    
    Note: The request_timeout in ConnectionConfig may control health check timeout.
    For health check timeout issues, try increasing SANDBOX_REQUEST_TIMEOUT_SECONDS.
    """
    try:
        from opensandbox.config import ConnectionConfig
    except ImportError:
        return None
    
    domain = os.getenv("SANDBOX_DOMAIN") or os.getenv("OPEN_SANDBOX_DOMAIN")
    api_key = os.getenv("SANDBOX_API_KEY") or os.getenv("OPEN_SANDBOX_API_KEY")
    # Increase default timeout for health check (30s is often too short for image pull)
    timeout_seconds = int(os.getenv("SANDBOX_REQUEST_TIMEOUT_SECONDS", "300"))  # Default 5 minutes
    debug_enabled = os.getenv("OPENSANDBOX_DEBUG", "false").lower() == "true"
    
    if domain or api_key:
        return ConnectionConfig(
            domain=domain or "localhost:8080",
            api_key=api_key,
            request_timeout=timedelta(seconds=timeout_seconds),
            debug=debug_enabled,
        )
    
    return None


async def run_code_in_opensandbox(
    code: str,
    task_id: str,
    timeout_seconds: int,
    image: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    ready_timeout_seconds: Optional[int] = None,
    existing_sandbox_id: Optional[str] = None,
    keep_alive: bool = False,
) -> Dict[str, Any]:
    """
    Execute code in OpenSandbox and return stdout/stderr/returncode metadata.
    
    Supports ConnectionConfig via environment variables:
    - SANDBOX_DOMAIN: OpenSandbox server domain (e.g., "api.opensandbox.io" or "localhost:8080")
    - SANDBOX_API_KEY: API key for authentication
    - SANDBOX_REQUEST_TIMEOUT_SECONDS: Request timeout in seconds (default: 300)
    - OPENSANDBOX_READY_TIMEOUT_SECONDS: Ready timeout for health check in seconds (default: 300)
    - OPENSANDBOX_HEALTH_CHECK_MODE: execd_command|lifecycle_status|skip (default: execd_command)
    - OPENSANDBOX_HEALTH_CHECK_POLL_INTERVAL_MS: Health check poll interval in ms (default: 200)
    - OPENSANDBOX_SKIP_HEALTH_CHECK: If true, skip readiness checks (default: false)
    - OPENSANDBOX_BOOTSTRAP_SKIP_HEALTH_CHECK: If true, create sandbox without SDK health check
      and run manual readiness checks in this client (default: false)
    - OPENSANDBOX_SKIP_MCP_INSTALL: If true, skip pip install of MCP dependencies (default: false)
      Use this when your image already has langchain-mcp-adapters pre-installed (e.g., :with-mcp images)
    - OPENSANDBOX_PYTHON_CMD: Python executable inside sandbox (default: python3)
    - OPENSANDBOX_PYTHON_VERSION: Python version to activate via code-interpreter-env.sh (default: 3.13)
    - OPENSANDBOX_USE_VENV: If true, source /opt/opensandbox/code-interpreter-env.sh before running Python
      This is needed for custom images where pandas etc. are installed in a venv (default: true)
    - OPENSANDBOX_SHOW_PROGRESS: If true, show progress messages during execution (default: true)
    - OPENSANDBOX_PROGRESS_INTERVAL_SECONDS: Interval for progress messages (default: 10)
    
    Args:
        code: Python code to execute
        task_id: Unique task identifier
        timeout_seconds: Sandbox auto-termination timeout (default: 10 minutes)
        image: Docker image name (default: from OPENSANDBOX_IMAGE env var)
        env: Environment variables for sandbox
        ready_timeout_seconds: Maximum time to wait for sandbox to be ready (default: 300s)
        existing_sandbox_id: If provided, connect to existing sandbox instead of creating new one
        keep_alive: If True, don't terminate sandbox after execution (for reuse)
    """
    try:
        from opensandbox.sandbox import Sandbox
        from opensandbox.models import WriteEntry
    except Exception as exc:
        return {
            "error": f"OpenSandbox SDK not available: {exc}",
            "error_type": type(exc).__name__,
        }

    image = image or os.getenv(
        "OPENSANDBOX_IMAGE",
        "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.1",
    )
    
    # Get connection config if available
    connection_config = _get_connection_config()
    
    # Prepare environment variables if needed
    sandbox_env = {}
    if env:
        sandbox_env.update(env)
    env_json = _load_env_json("OPENSANDBOX_ENV_JSON")
    if env_json:
        sandbox_env.update(env_json)
    
    # CRITICAL: Pass API key to sandbox for MCP calls
    # MCP services behind Nginx proxy may require this header
    api_key = os.getenv("OPEN_SANDBOX_API_KEY") or os.getenv("SANDBOX_API_KEY")
    if api_key:
        sandbox_env["OPEN_SANDBOX_API_KEY"] = api_key
        sandbox_env["SANDBOX_API_KEY"] = api_key
        print(f"[opensandbox] API key passed to sandbox environment")
    
    sandbox = None
    try:
        # Log connection configuration for diagnostics
        print(f"[opensandbox] Creating sandbox with image: {image}")
        if connection_config:
            domain = getattr(connection_config, 'domain', None) or os.getenv("SANDBOX_DOMAIN", "default")
            has_api_key = bool(getattr(connection_config, 'api_key', None) or os.getenv("SANDBOX_API_KEY"))
            timeout = getattr(connection_config, 'request_timeout', None)
            print(f"[opensandbox] Connection config: domain={domain}, has_api_key={has_api_key}, timeout={timeout}")
        else:
            print(f"[opensandbox] Using default connection (no custom config)")
        
        # Get ready_timeout (health check timeout)
        # This is critical for first-time image pulls which can take longer
        # Default is 30s in SDK, but we increase it to 300s (5 minutes) for reliability
        ready_timeout = ready_timeout_seconds or int(os.getenv("OPENSANDBOX_READY_TIMEOUT_SECONDS", "300"))

        # Sandbox lifetime must be >= 60s per server schema.
        sandbox_timeout_seconds = max(
            int(os.getenv("OPENSANDBOX_SANDBOX_TIMEOUT_SECONDS", str(timeout_seconds))),
            60,
        )

        # Command timeout is independent from sandbox lifetime.
        command_timeout_seconds = int(
            os.getenv("OPENSANDBOX_COMMAND_TIMEOUT_SECONDS", str(timeout_seconds))
        )
        
        print(f"[opensandbox] Ready timeout (health check): {ready_timeout}s")
        print(f"[opensandbox] Sandbox timeout (auto-termination): {sandbox_timeout_seconds}s")
        print(f"[opensandbox] Command timeout: {command_timeout_seconds}s")

        health_check_mode = os.getenv("OPENSANDBOX_HEALTH_CHECK_MODE", "execd_command").strip().lower()
        skip_health_check = os.getenv("OPENSANDBOX_SKIP_HEALTH_CHECK", "false").lower() == "true"
        bootstrap_skip_health_check = (
            os.getenv("OPENSANDBOX_BOOTSTRAP_SKIP_HEALTH_CHECK", "false").lower() == "true"
        )
        health_check_poll_interval_ms = int(os.getenv("OPENSANDBOX_HEALTH_CHECK_POLL_INTERVAL_MS", "200"))
        proxy_base = os.getenv("OPENSANDBOX_EXECD_PROXY_BASE", "").strip()

        warned_endpoint = False

        def _is_private_endpoint(endpoint: str) -> bool:
            host = endpoint.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
            if host.endswith((".cluster.local", ".svc", ".svc.cluster.local")):
                return True
            octets = host.split(".")
            if len(octets) == 4 and all(part.isdigit() for part in octets):
                nums = list(map(int, octets))
                return (
                    nums[0] == 10
                    or (nums[0] == 172 and 16 <= nums[1] <= 31)
                    or (nums[0] == 192 and nums[1] == 168)
                    or nums[0] == 127
                )
            return False

        async def _warn_endpoint_if_private(sbx: Any) -> None:
            nonlocal warned_endpoint
            if warned_endpoint:
                return
            try:
                from opensandbox.constants import DEFAULT_EXECD_PORT
                endpoint = await sbx.get_endpoint(DEFAULT_EXECD_PORT)
                if endpoint:
                    endpoint_value = endpoint.endpoint
                    print(f"[opensandbox] Execd endpoint resolved: {endpoint_value}")
                    if "/proxy/" in endpoint_value:
                        print(
                            "[opensandbox] ⚠ Execd endpoint uses dynamic host port proxy. "
                            "If you are using Nginx on a fixed port, it will not proxy this dynamic port. "
                            "Expose the execd host port directly or switch to docker network_mode=host."
                        )
                    if _is_private_endpoint(endpoint_value):
                        print(
                            "[opensandbox] ⚠ Execd endpoint appears private. "
                            f"endpoint={endpoint_value}. "
                            "If your client is outside the host/network, health checks and commands may fail."
                        )
                    warned_endpoint = True
            except Exception:
                # Ignore endpoint warnings if resolution fails early
                pass

        async def _custom_health_check(sbx: Any) -> bool:
            await _warn_endpoint_if_private(sbx)
            if health_check_mode == "lifecycle_status":
                try:
                    info = await sbx.get_info()
                    state = (info.status.state or "").upper()
                    reason = (info.status.reason or "").upper()
                    if state == "RUNNING":
                        return True
                    if "READY" in reason or "RUNNING" in reason:
                        return True
                    return False
                except Exception:
                    return False
            # Default: execd command probe
            try:
                execution = await sbx.commands.run("echo health_check")
                stdout, _stderr = _extract_logs(execution)
                if getattr(execution, "error", None):
                    return False
                return "health_check" in stdout or stdout.strip() != ""
            except Exception:
                return False
        
        if skip_health_check or bootstrap_skip_health_check:
            print("[opensandbox] Skip health check enabled; sandbox may not be ready yet.")
        else:
            print(f"[opensandbox] Health check mode: {health_check_mode}")

        # Create sandbox following SDK documentation
        # image is the first positional argument
        # connection_config, timeout, env, ready_timeout are keyword arguments
        create_kwargs = {
            "timeout": timedelta(seconds=sandbox_timeout_seconds),
            "ready_timeout": timedelta(seconds=ready_timeout),
        }
        
        # Connection config if available
        if connection_config:
            create_kwargs["connection_config"] = connection_config
        
        # Only add optional parameters if they are provided
        if sandbox_env:
            create_kwargs["env"] = sandbox_env
        if not skip_health_check and not bootstrap_skip_health_check and health_check_mode != "skip":
            create_kwargs["health_check"] = _custom_health_check
            create_kwargs["health_check_polling_interval"] = timedelta(
                milliseconds=health_check_poll_interval_ms
            )
        if skip_health_check or bootstrap_skip_health_check or health_check_mode == "skip" or proxy_base:
            create_kwargs["skip_health_check"] = True
        
        # Check if we should connect to an existing sandbox instead of creating new one
        if existing_sandbox_id:
            print(f"[opensandbox] Connecting to existing sandbox: {existing_sandbox_id}")
            try:
                connect_kwargs = {
                    "connection_config": connection_config,
                    "health_check": _custom_health_check if not (skip_health_check or bootstrap_skip_health_check or health_check_mode == "skip") else None,
                    "connect_timeout": timedelta(seconds=ready_timeout),
                    "health_check_polling_interval": timedelta(milliseconds=health_check_poll_interval_ms),
                    "skip_health_check": skip_health_check or bootstrap_skip_health_check or health_check_mode == "skip" or bool(proxy_base),
                }
                sandbox = await Sandbox.connect(existing_sandbox_id, **connect_kwargs)
                print(f"[opensandbox] Connected to existing sandbox: {getattr(sandbox, 'id', existing_sandbox_id)}")
            except Exception as connect_exc:
                print(f"[opensandbox] Failed to connect to existing sandbox: {connect_exc}")
                print(f"[opensandbox] Falling back to creating new sandbox...")
                existing_sandbox_id = None  # Reset to trigger new sandbox creation
        
        if not existing_sandbox_id:
            print(f"[opensandbox] Creating sandbox with kwargs: {list(create_kwargs.keys())}")
        
        # image is positional, other params are keyword
        if not sandbox:
            try:
                sandbox = await Sandbox.create(image, **create_kwargs)
            except Exception as create_exc:
                if "HTTP 422" in str(create_exc):
                    print("[opensandbox] SDK create failed with 422, retrying via REST body fallback")

                    def _build_base_url() -> str:
                        domain = os.getenv("SANDBOX_DOMAIN") or os.getenv("OPEN_SANDBOX_DOMAIN") or "localhost:8080"
                        if domain.startswith(("http://", "https://")):
                            return f"{domain}/v1"
                        return f"http://{domain}/v1"

                    api_key = os.getenv("OPEN_SANDBOX_API_KEY") or os.getenv("SANDBOX_API_KEY") or ""
                    base_url = _build_base_url()
                    create_body = {
                        "image": {"uri": image},
                        "timeout": sandbox_timeout_seconds,
                        "resourceLimits": {"cpu": "1", "memory": "2Gi"},
                        "entrypoint": ["tail", "-f", "/dev/null"],
                    }
                    if sandbox_env:
                        create_body["env"] = sandbox_env
                    
                    # Add volume bindings from environment variable
                    # Format: "/host/path:/container/path:ro,/host/path2:/container/path2"
                    volume_bindings_env = os.getenv("OPENSANDBOX_VOLUME_BINDINGS", "")
                    if volume_bindings_env:
                        volume_bindings = []
                        for binding in volume_bindings_env.split(","):
                            binding = binding.strip()
                            if binding:
                                parts = binding.split(":")
                                if len(parts) >= 2:
                                    vb = {
                                        "hostPath": parts[0],
                                        "containerPath": parts[1],
                                    }
                                    if len(parts) >= 3:
                                        vb["readOnly"] = parts[2].lower() == "ro"
                                    volume_bindings.append(vb)
                        if volume_bindings:
                            create_body["volumeBindings"] = volume_bindings
                            print(f"[opensandbox] Adding volume bindings: {volume_bindings}")

                    req = urllib.request.Request(
                        f"{base_url}/sandboxes",
                        data=json.dumps(create_body).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "OPEN-SANDBOX-API-KEY": api_key,
                        },
                        method="POST",
                    )
                    try:
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            payload = json.loads(resp.read().decode("utf-8"))
                        sandbox_id = payload.get("id")
                        if not sandbox_id:
                            raise RuntimeError(f"REST create returned no id: {payload}")
                    except urllib.error.HTTPError as http_err:
                        try:
                            err_payload = http_err.read().decode("utf-8")
                        except Exception:
                            err_payload = ""
                        message = f"REST create failed: HTTP {http_err.code}"
                        if err_payload:
                            message = f"{message} - {err_payload}"
                        raise RuntimeError(message) from http_err

                    connect_kwargs = {
                        "connection_config": connection_config,
                        "health_check": _custom_health_check if health_check_mode != "skip" else None,
                        "connect_timeout": timedelta(seconds=ready_timeout),
                        "health_check_polling_interval": timedelta(
                            milliseconds=health_check_poll_interval_ms
                        ),
                        "skip_health_check": skip_health_check
                        or bootstrap_skip_health_check
                        or health_check_mode == "skip"
                        or bool(proxy_base),
                    }
                    sandbox = await Sandbox.connect(sandbox_id, **connect_kwargs)
                else:
                    raise
        print(f"[opensandbox] Sandbox created successfully: {getattr(sandbox, 'id', 'N/A')}")

        if proxy_base:
            _apply_execd_proxy_override(sandbox, proxy_base)

        manual_health_check = (
            (bootstrap_skip_health_check or proxy_base)
            and not skip_health_check
            and health_check_mode != "skip"
        )
        if manual_health_check:
            await _warn_endpoint_if_private(sandbox)
            deadline = time.time() + ready_timeout
            attempt = 0
            last_error: Optional[Exception] = None
            while time.time() < deadline:
                attempt += 1
                try:
                    is_ready = await _custom_health_check(sandbox)
                except Exception as exc:
                    last_error = exc
                    is_ready = False
                if is_ready:
                    print(
                        f"[opensandbox] Sandbox passed manual health check after {attempt} attempts"
                    )
                    break
                await asyncio.sleep(health_check_poll_interval_ms / 1000.0)
            else:
                error_detail = f"Last error: {last_error}" if last_error else "Health check returned false continuously"
                return {
                    "error": (
                        f"Sandbox health check timed out after {ready_timeout}s "
                        f"({attempt} attempts). {error_detail}"
                    ),
                    "error_type": "SandboxReadyTimeoutException",
                    "sandbox_id": getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None),
                    "image": image,
                }

        async with sandbox:
            # Setup MCP support if code might use MCP tools
            # Check if code imports call_tool or mcp_helper, or force via env
            force_mcp_setup = (
                os.getenv("OPENSANDBOX_FORCE_MCP_SETUP", "false").lower() == "true"
                or (env or {}).get("OPENSANDBOX_FORCE_MCP_SETUP", "").lower() == "true"
            )
            if (
                force_mcp_setup
                or "call_tool" in code
                or "mcp_helper" in code
                or "from core.tool_interface" in code
            ):
                try:
                    await _setup_mcp_support_in_sandbox(sandbox)
                except Exception as setup_e:
                    print(f"  ⚠ MCP setup warning: {setup_e}, code may still work if dependencies are pre-installed")
            
            code_path = f"/tmp/codeact_{task_id}.py"
            
            # Get sandbox ID for environment variable
            current_sandbox_id = getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None) or ""
            
            # Get API key to pass to sandbox code
            api_key_for_sandbox = os.getenv("OPEN_SANDBOX_API_KEY") or os.getenv("SANDBOX_API_KEY") or ""
            
            # Prepend minimal setup code - no blocking imports
            setup_code = f"""import sys
import os

# Add agent modules to path (before any imports)
sys.path.insert(0, '/tmp/agent_modules')
sys.path.insert(0, '/tmp/agent_modules/third_party')
os.environ['AGENT_CONFIG_DIR'] = '/tmp/agent_config'
os.environ['OPENSANDBOX_ID'] = '{current_sandbox_id}'

# Pass API key for MCP service authentication
if '{api_key_for_sandbox}':
    os.environ['OPEN_SANDBOX_API_KEY'] = '{api_key_for_sandbox}'
    os.environ['SANDBOX_API_KEY'] = '{api_key_for_sandbox}'
    print("[sandbox] API key loaded for MCP authentication")

"""
            full_code = setup_code + code
            
            # Use write_file for single file (consistent with example code)
            await sandbox.files.write_file(code_path, full_code)

            python_cmd = os.getenv("OPENSANDBOX_PYTHON_CMD", "python3")
            python_version = os.getenv("OPENSANDBOX_PYTHON_VERSION", "3.13")
            
            # Check if we should source the virtual environment before running Python
            # This is needed for custom images where packages are installed in a venv
            venv_setup_script = "/opt/opensandbox/code-interpreter-env.sh"
            use_venv = os.getenv("OPENSANDBOX_USE_VENV", "true").lower() == "true"
            
            # Build the execution command
            if use_venv:
                # Source venv first with python version, then run Python (ensures pandas etc. are available)
                exec_cmd = f"source {venv_setup_script} python {python_version} && {python_cmd} {code_path}"
            else:
                exec_cmd = f"{python_cmd} {code_path}"
            
            # Progress feedback for long-running commands
            progress_interval = int(os.getenv("OPENSANDBOX_PROGRESS_INTERVAL_SECONDS", "10"))
            show_progress = os.getenv("OPENSANDBOX_SHOW_PROGRESS", "true").lower() == "true"
            
            async def _run_with_progress():
                """Run command with periodic progress feedback."""
                start_time = time.time()
                cmd_task = asyncio.create_task(
                    sandbox.commands.run(exec_cmd)
                )
                
                while not cmd_task.done():
                    try:
                        # Wait for either completion or progress interval
                        await asyncio.wait_for(
                            asyncio.shield(cmd_task),
                            timeout=progress_interval
                        )
                    except asyncio.TimeoutError:
                        # Command still running, print progress
                        elapsed = int(time.time() - start_time)
                        if show_progress:
                            print(f"  ⏳ 命令执行中... 已运行 {elapsed}s / 超时 {command_timeout_seconds}s")
                        
                        # Check if we've exceeded the total timeout
                        if command_timeout_seconds > 0 and elapsed >= command_timeout_seconds:
                            cmd_task.cancel()
                            try:
                                await cmd_task
                            except asyncio.CancelledError:
                                pass
                            raise asyncio.TimeoutError()
                
                return await cmd_task
            
            try:
                if show_progress:
                    print(f"  ▶ 开始执行代码 (超时: {command_timeout_seconds}s)...")
                execution = await _run_with_progress()
                if show_progress:
                    print(f"  ✓ 代码执行完成")
            except asyncio.TimeoutError:
                print(f"  ⏱ 命令超时 ({command_timeout_seconds}s)")
                return {
                    "error": f"Command timeout after {command_timeout_seconds}s",
                    "error_type": "TimeoutError",
                    "sandbox_id": getattr(sandbox, "id", None)
                    or getattr(sandbox, "sandbox_id", None),
                    "image": image,
                }
            stdout, stderr = _extract_logs(execution)
            formatted_output = _format_execution(execution)
            returncode = (
                getattr(execution, "returncode", None)
                if hasattr(execution, "returncode")
                else getattr(execution, "exit_code", None)
            )

            if (
                "command not found" in (stderr or "").lower()
                and python_cmd == "python3"
            ):
                fallback_cmd = "python"
                if show_progress:
                    print(f"  ⚠ python3 未找到，尝试 python...")
                
                # Build fallback command with same venv setup
                if use_venv:
                    fallback_exec_cmd = f"source {venv_setup_script} python {python_version} && {fallback_cmd} {code_path}"
                else:
                    fallback_exec_cmd = f"{fallback_cmd} {code_path}"
                
                async def _run_fallback_with_progress():
                    """Run fallback command with progress."""
                    start_time = time.time()
                    cmd_task = asyncio.create_task(
                        sandbox.commands.run(fallback_exec_cmd)
                    )
                    while not cmd_task.done():
                        try:
                            await asyncio.wait_for(asyncio.shield(cmd_task), timeout=progress_interval)
                        except asyncio.TimeoutError:
                            elapsed = int(time.time() - start_time)
                            if show_progress:
                                print(f"  ⏳ 命令执行中... 已运行 {elapsed}s / 超时 {command_timeout_seconds}s")
                            if command_timeout_seconds > 0 and elapsed >= command_timeout_seconds:
                                cmd_task.cancel()
                                try:
                                    await cmd_task
                                except asyncio.CancelledError:
                                    pass
                                raise asyncio.TimeoutError()
                    return await cmd_task
                
                try:
                    execution = await _run_fallback_with_progress()
                    if show_progress:
                        print(f"  ✓ 代码执行完成")
                except asyncio.TimeoutError:
                    print(f"  ⏱ 命令超时 ({command_timeout_seconds}s)")
                    return {
                        "error": f"Command timeout after {command_timeout_seconds}s",
                        "error_type": "TimeoutError",
                        "sandbox_id": getattr(sandbox, "id", None)
                        or getattr(sandbox, "sandbox_id", None),
                        "image": image,
                    }
                stdout, stderr = _extract_logs(execution)
                formatted_output = _format_execution(execution)
                returncode = (
                    getattr(execution, "returncode", None)
                    if hasattr(execution, "returncode")
                    else getattr(execution, "exit_code", None)
                )

            return {
                "stdout": stdout,
                "stderr": stderr,
                "formatted_output": formatted_output,  # Added formatted output
                "returncode": returncode,
                "sandbox_id": getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None),
                "image": image,
                "has_error": hasattr(execution, "error") and execution.error is not None,
            }
    except Exception as exc:
        error_msg = str(exc)
        error_type = type(exc).__name__
        
        # Provide detailed diagnostics for different error types
        diagnostics = []
        
        # Handle HTTP 503 (Service Unavailable)
        if "503" in error_msg or "Service Unavailable" in error_msg:
            domain = getattr(connection_config, 'domain', None) if connection_config else os.getenv('SANDBOX_DOMAIN', 'localhost:8080')
            diagnostics.append("HTTP 503 错误诊断:")
            diagnostics.append(f"  1. 服务可能暂时过载，建议稍后重试")
            diagnostics.append(f"  2. 健康检查端点正常，但创建 sandbox 失败，可能是资源不足")
            diagnostics.append(f"  3. 检查服务健康状态: curl http://{domain}/health")
            diagnostics.append(f"  4. 检查是否需要 API key: {'已设置' if os.getenv('SANDBOX_API_KEY') else '未设置（可能需要）'}")
            diagnostics.append(f"  5. 当前配置的域名: {domain}")
            diagnostics.append(f"  6. 尝试手动测试创建 sandbox:")
            diagnostics.append(f"     curl -X POST http://{domain}/api/v1/sandboxes \\")
            diagnostics.append(f"          -H 'Content-Type: application/json' \\")
            api_key = os.getenv('SANDBOX_API_KEY')
            if api_key:
                diagnostics.append(f"          -H 'Authorization: Bearer {api_key[:10]}...' \\")
            diagnostics.append(f"          -d '{{\"image\": \"{image}\"}}'")
            diagnostics.append(f"  7. 如果健康检查正常但创建失败，可能是:")
            diagnostics.append(f"     - 服务资源不足（CPU/内存/容器）")
            diagnostics.append(f"     - 需要 API key 认证")
            diagnostics.append(f"     - 请求格式不正确")
        
        # Handle connection errors
        elif "Network connectivity" in error_msg or "connection" in error_msg.lower():
            diagnostics.append("连接诊断:")
            diagnostics.append(f"  1. 检查 SANDBOX_DOMAIN 环境变量: {os.getenv('SANDBOX_DOMAIN', '未设置 (使用默认值)')}")
            diagnostics.append(f"  2. 检查 SANDBOX_API_KEY 环境变量: {'已设置' if os.getenv('SANDBOX_API_KEY') else '未设置'}")
            diagnostics.append(f"  3. 检查 OpenSandbox 服务是否运行")
            diagnostics.append(f"  4. 检查网络连接和防火墙设置")
            if connection_config:
                domain = getattr(connection_config, 'domain', None)
                diagnostics.append(f"  5. 当前配置的域名: {domain}")
                diagnostics.append(f"  6. 测试健康检查: curl http://{domain}/health")
        
        # Handle authentication errors
        elif "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg or "Forbidden" in error_msg:
            diagnostics.append("认证错误诊断:")
            diagnostics.append(f"  1. 检查 SANDBOX_API_KEY 环境变量: {'已设置' if os.getenv('SANDBOX_API_KEY') else '未设置（必需）'}")
            diagnostics.append(f"  2. 验证 API key 是否有效")
            diagnostics.append(f"  3. 检查 API key 格式是否正确")
        
        # Handle health check timeout
        elif "health check" in error_msg.lower() and ("timeout" in error_msg.lower() or "timed out" in error_msg.lower()):
            domain = getattr(connection_config, 'domain', None) if connection_config else os.getenv('SANDBOX_DOMAIN', 'localhost:8080')
            current_ready_timeout = ready_timeout_seconds or int(os.getenv("OPENSANDBOX_READY_TIMEOUT_SECONDS", "300"))
            diagnostics.append("健康检查超时诊断:")
            diagnostics.append(f"  1. Sandbox 创建请求可能已接受，但健康检查失败")
            diagnostics.append(f"  2. 可能原因:")
            diagnostics.append(f"     - 镜像拉取需要更长时间（首次使用或网络慢）")
            diagnostics.append(f"     - 容器启动需要更长时间（资源不足）")
            diagnostics.append(f"     - Sandbox 内部服务启动失败")
            diagnostics.append(f"  3. 解决方案:")
            diagnostics.append(f"     - 增加 ready_timeout（健康检查等待时间）")
            diagnostics.append(f"     - 检查镜像是否存在: docker pull {image}")
            diagnostics.append(f"     - 检查服务资源是否充足")
            diagnostics.append(f"     - 稍后重试（可能是临时资源不足）")
            diagnostics.append(f"  4. 当前配置:")
            diagnostics.append(f"     - 域名: {domain}")
            diagnostics.append(f"     - 镜像: {image}")
            diagnostics.append(f"     - Ready timeout (健康检查): {current_ready_timeout}秒")
            request_timeout = getattr(connection_config, 'request_timeout', None) if connection_config else None
            if request_timeout:
                diagnostics.append(f"     - Request timeout (API请求): {request_timeout}")
            else:
                diagnostics.append(f"     - Request timeout (API请求): 默认值（300秒）")
            diagnostics.append(f"  5. 建议设置环境变量:")
            diagnostics.append(f"     OPENSANDBOX_READY_TIMEOUT_SECONDS=600  # 增加到 10 分钟（健康检查）")
            diagnostics.append(f"     SANDBOX_REQUEST_TIMEOUT_SECONDS=300  # API 请求超时（已设置）")
        
        # Generic error
        else:
            diagnostics.append("错误诊断:")
            diagnostics.append(f"  1. 错误类型: {error_type}")
            diagnostics.append(f"  2. 错误消息: {error_msg}")
            if connection_config:
                domain = getattr(connection_config, 'domain', None)
                diagnostics.append(f"  3. 当前配置的域名: {domain}")
        
        print(f"[opensandbox] ❌ 创建 sandbox 失败: {error_type}: {error_msg}")
        if diagnostics:
            print(f"[opensandbox] " + "\n[opensandbox] ".join(diagnostics))
        
        return {
            "error": error_msg,
            "error_type": error_type,
            "sandbox_id": getattr(sandbox, "id", None) or (getattr(sandbox, "sandbox_id", None) if sandbox else None),
            "image": image,
            "diagnostics": "\n".join(diagnostics) if diagnostics else None,
            "is_retryable": "503" in error_msg or "Service Unavailable" in error_msg,  # Mark 503 as retryable
        }
    finally:
        # Improved cleanup: kill and close (similar to example code)
        # Only kill if keep_alive is False (i.e., sandbox should not be reused)
        if sandbox is not None and not keep_alive:
            try:
                await sandbox.kill()
                await sandbox.close()
                print(f"[opensandbox] Sandbox terminated (keep_alive=False)")
            except Exception as cleanup_exc:
                # Log but don't fail on cleanup errors
                print(f"[opensandbox] Cleanup warning: {cleanup_exc}")
        elif sandbox is not None and keep_alive:
            # Sandbox kept alive for reuse - just close the connection
            print(f"[opensandbox] Sandbox kept alive for reuse: {getattr(sandbox, 'id', 'unknown')}")


def run_code_in_opensandbox_sync(
    code: str,
    task_id: str,
    timeout_seconds: int,
    image: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    ready_timeout_seconds: Optional[int] = None,
    existing_sandbox_id: Optional[str] = None,
    keep_alive: bool = False,
) -> Dict[str, Any]:
    """Sync wrapper for OpenSandbox execution.
    
    Args:
        code: Python code to execute
        task_id: Unique task identifier
        timeout_seconds: Sandbox auto-termination timeout
        image: Docker image name
        env: Environment variables for sandbox
        ready_timeout_seconds: Maximum time to wait for sandbox to be ready
        existing_sandbox_id: If provided, connect to existing sandbox instead of creating new one
        keep_alive: If True, don't terminate sandbox after execution (for reuse)
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    run_code_in_opensandbox(
                        code=code,
                        task_id=task_id,
                        timeout_seconds=timeout_seconds,
                        image=image,
                        env=env,
                        ready_timeout_seconds=ready_timeout_seconds,
                        existing_sandbox_id=existing_sandbox_id,
                        keep_alive=keep_alive,
                    ),
                )
                return future.result()
    except RuntimeError:
        pass

    return asyncio.run(
        run_code_in_opensandbox(
            code=code,
            task_id=task_id,
            timeout_seconds=timeout_seconds,
            image=image,
            env=env,
            ready_timeout_seconds=ready_timeout_seconds,
            existing_sandbox_id=existing_sandbox_id,
            keep_alive=keep_alive,
        )
    )

