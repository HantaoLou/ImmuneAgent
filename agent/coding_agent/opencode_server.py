"""
OpenCode Server — manages an opencode server process and communicates via REST API.

Replaces the CLI-based OpenCodeExecutor with the server-mode pattern
proven in immune_executor. The key advantage: server mode supports
native MCP tool calls, two-tier agent architecture (build → @codeact),
and runtime config injection.

Architecture:
    IterativeOpenCodeExecutor
      → OpenCodeServer.start()          → opencode serve --port <dynamic>
      → OpenCodeServer.inject_config()  → PATCH /global/config (skills, MCP, codeact)
      → OpenCodeServer.execute_prompt() → POST /session + POST /session/:id/prompt_async
      → OpenCodeServer.poll()           → GET /session/status + GET /session/:id/message
      → OpenCodeServer.stop()           → kill server process
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from coding_agent.config import (
    ExecutionResult,
    ExecutionStatus,
    OpenCodeConfig,
    DEFAULT_MCP_CONFIG,
)
from utils.opencode_client import OpenCodeClient, OpenCodeError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Codeact agent definition (English, mirrors immune_executor)
# ---------------------------------------------------------------------------

_CODEACT_AGENT_DEF = {
    "mode": "subagent",
    "description": (
        "Immunology MCP tool expert. Calls igblast, metabcr, nettcr and other "
        "analysis services through OpenCode native MCP interface. Uses bash "
        "background processes to poll real results for streaming_task async tools."
    ),
    "permission": {
        "bash": "allow",
    },
}


# ---------------------------------------------------------------------------
# OpenCodeServer
# ---------------------------------------------------------------------------

class OpenCodeServer:
    """
    Manages an opencode server process and communicates via REST API.

    Lifecycle:
        server = OpenCodeServer(config)
        await server.start()
        await server.inject_config(workspace_dir)
        result = await server.execute_prompt("do something")
        await server.stop()
    """

    def __init__(
        self,
        config: OpenCodeConfig,
        progress_callback: Optional[callable] = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self._process: Optional[subprocess.Popen] = None
        self._client: Optional[OpenCodeClient] = None
        self._port: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, startup_timeout: int = 30) -> None:
        """Start opencode serve on a dynamic port, wait for health check."""
        self._port = _find_free_port()
        env = self._build_env()

        opencode_bin = self.config.opencode_bin_path or _find_opencode_binary()
        cmd = [opencode_bin, "serve", "--port", str(self._port)]

        logger.info("Starting opencode server: %s (port %d)", " ".join(cmd), self._port)

        self._process = subprocess.Popen(
            cmd,
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        base_url = f"http://127.0.0.1:{self._port}"
        self._client = OpenCodeClient(base_url=base_url)

        # Poll health check with exponential backoff
        deadline = time.time() + startup_timeout
        delay = 0.5
        while time.time() < deadline:
            await asyncio.sleep(delay)
            if self._client.check_health():
                logger.info("OpenCode server healthy on port %d", self._port)
                return
            delay = min(delay * 1.5, 3.0)

        # Startup failed — collect output for diagnostics
        self._kill_process()
        raise RuntimeError(
            f"OpenCode server failed to start within {startup_timeout}s on port {self._port}"
        )

    async def inject_config(self, workspace_dir: str) -> None:
        """PATCH skills + MCP + codeact agent into running server."""
        if not self._client:
            raise RuntimeError("Server not started. Call start() first.")

        skills_dir = Path(__file__).parent / "skills"
        mcp_config = self._build_mcp_config()

        patch = {
            "mcp": mcp_config,
            "agent": {"codeact": _CODEACT_AGENT_DEF},
            "permission": {"external_directory": "allow"},
        }
        if skills_dir.is_dir():
            patch["skills"] = {"paths": [str(skills_dir.resolve())]}

        self._client.patch_config(patch)
        logger.info("Config injected: skills=%s, mcp=%s",
                     skills_dir.exists(), list(mcp_config.keys()))

        # Verify skills loaded
        loaded = self._client.list_skills()
        logger.info("Loaded skills: %s", loaded)

    async def execute_prompt(
        self,
        prompt: str,
        timeout: int = 1800,
    ) -> ExecutionResult:
        """Create session, send prompt, poll until done, return result."""
        if not self._client:
            raise RuntimeError("Server not started. Call start() first.")

        start_ts = time.time()

        session_id = self._client.create_session()
        logger.info("Created session %s, sending prompt (%d chars)",
                     session_id, len(prompt))

        self._client.send_prompt(session_id, prompt)

        success = await self._monitor(session_id, timeout=timeout)
        summary = self._extract_summary(session_id)

        elapsed_ms = int((time.time() - start_ts) * 1000)

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if success else ExecutionStatus.TIMEOUT,
            stdout=summary,
            execution_time_ms=elapsed_ms,
            sandbox_id=f"server-{self._port}",
        )

    async def stop(self) -> None:
        """Kill the opencode server process."""
        if self._client:
            self._client.close()
            self._client = None
        self._kill_process()
        logger.info("OpenCode server stopped (port %d)", self._port)

    # ------------------------------------------------------------------
    # Session monitoring (mirrors immune_executor._monitor)
    # ------------------------------------------------------------------

    async def _monitor(
        self,
        session_id: str,
        timeout: int = 1800,
        poll_interval: int = 3,
    ) -> bool:
        """Poll session until idle. Returns True on completion, False on timeout."""
        start_ts = time.time()
        seen_parts: dict[str, str] = {}
        child_sids: set[str] = set()
        consecutive_idle = 0
        min_response_messages = 1
        idle_threshold = 2

        while time.time() - start_ts < timeout:
            await asyncio.sleep(poll_interval)
            elapsed = time.time() - start_ts

            msgs = self._client.get_messages(session_id)
            task_parts: list[dict] = []

            # Scan main session messages
            for msg in msgs:
                for part in msg.get("parts", []):
                    if part.get("type") != "tool":
                        continue
                    state = part.get("state", {})
                    status = state.get("status", "")
                    tool = part.get("tool", "")
                    pid = part.get("id") or part.get("callID") or str(id(part))

                    if tool == "task":
                        task_parts.append(part)
                        sub = state.get("metadata", {}).get("sessionId", "")
                        if sub:
                            child_sids.add(sub)

                    prev = seen_parts.get(pid)
                    if prev == status:
                        continue
                    seen_parts[pid] = status

                    if status in ("running", "completed"):
                        self._log_tool_event(tool, state.get("title", ""), status)

            # Scan child sessions (codeact subagent)
            for csid in list(child_sids):
                try:
                    for msg in self._client.get_messages(csid):
                        for part in msg.get("parts", []):
                            if part.get("type") != "tool":
                                continue
                            state = part.get("state", {})
                            status = state.get("status", "")
                            tool = part.get("tool", "")
                            pid = (part.get("id") or part.get("callID") or str(id(part))) + f"@{csid[:8]}"
                            prev = seen_parts.get(pid)
                            if prev == status:
                                continue
                            seen_parts[pid] = status
                            if status in ("running", "completed"):
                                self._log_tool_event(tool, state.get("title", ""), status, prefix="[codeact] ")
                except Exception:
                    pass

            # Check completion
            status_map = self._client.get_session_status()
            entry = status_map.get(session_id, {})
            is_busy = isinstance(entry, dict) and entry.get("type") == "busy"
            running_tasks = sum(
                1 for p in task_parts
                if p.get("state", {}).get("status") == "running"
            )

            consecutive_idle = 0 if is_busy else consecutive_idle + 1
            has_response = len(msgs) >= min_response_messages

            logger.debug(
                "Poll %.0fs | busy=%s idle=%d msgs=%d tasks=%d/%d running",
                elapsed, is_busy, consecutive_idle, len(msgs),
                len(task_parts), running_tasks,
            )

            if (
                consecutive_idle >= idle_threshold
                and has_response
                and running_tasks == 0
            ):
                logger.info("Session complete after %.1fs", time.time() - start_ts)
                return True

        logger.warning("Session timed out after %ds", timeout)
        return False

    # ------------------------------------------------------------------
    # Config builders (ported from opencode_executor.py)
    # ------------------------------------------------------------------

    def _build_env(self) -> Dict[str, str]:
        """Build environment variables for the opencode server process."""
        env: Dict[str, str] = {}

        # API keys
        model_lower = self.config.model_provider.lower()

        if "glm" in model_lower and self.config.api_key:
            env["ZHIPUAI_API_KEY"] = self.config.api_key
            env["ZHIPU_API_KEY"] = self.config.api_key
            env["OPENAI_API_KEY"] = "not-used"
            env["ANTHROPIC_API_KEY"] = "not-used"
        elif "claude" in model_lower and self.config.api_key:
            env["OPENCODE_MODEL"] = self.config.model_provider
            env["ANTHROPIC_API_KEY"] = self.config.api_key
            env["OPENAI_API_KEY"] = "not-used"
        elif "gpt" in model_lower or "openai" in model_lower:
            env["OPENCODE_MODEL"] = self.config.model_provider
            if self.config.api_key:
                env["OPENAI_API_KEY"] = self.config.api_key
            env["ANTHROPIC_API_KEY"] = "not-used"
        elif "deepseek" in model_lower and self.config.api_key:
            env["OPENCODE_MODEL"] = self.config.model_provider
            env["DEEPSEEK_API_KEY"] = self.config.api_key
            env["OPENAI_API_KEY"] = "not-used"
            env["ANTHROPIC_API_KEY"] = "not-used"
        else:
            env["OPENCODE_MODEL"] = self.config.model_provider
            if "OPENAI_API_KEY" not in env:
                env["OPENAI_API_KEY"] = self.config.api_key or "not-used"
            if "ANTHROPIC_API_KEY" not in env:
                env["ANTHROPIC_API_KEY"] = "not-used"

        env["NODE_OPTIONS"] = "--no-warnings"

        return env

    def _build_mcp_config(self) -> Dict[str, Any]:
        """Build MCP server config in OpenCode format."""
        if self.config.mcp_config_path and Path(self.config.mcp_config_path).exists():
            with open(self.config.mcp_config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
                return self._convert_mcp_config(raw_config)

        return DEFAULT_MCP_CONFIG.get("servers", {})

    def _convert_mcp_config(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Bio-Agent MCP config to OpenCode format."""
        opencode_config: Dict[str, Any] = {}

        for server_name, server_config in raw_config.items():
            oc_server: Dict[str, Any] = {}

            transport = server_config.get("transport", server_config.get("type", "sse"))
            if transport in ("sse", "http", "websocket"):
                oc_server["type"] = "remote"
            else:
                oc_server["type"] = "local"

            oc_server["enabled"] = True
            oc_server["oauth"] = False

            if "url" in server_config:
                oc_server["url"] = server_config["url"]

            if "timeout" in server_config:
                oc_server["timeout"] = server_config["timeout"] * 1000

            if "headers" in server_config:
                oc_server["headers"] = server_config["headers"]

            if "command" in server_config:
                oc_server["command"] = server_config["command"]
            if "args" in server_config:
                oc_server["args"] = server_config["args"]
            if "env" in server_config:
                oc_server["env"] = server_config["env"]

            opencode_config[server_name] = oc_server

        return opencode_config

    def _derive_streaming_urls(self) -> Dict[str, str]:
        """Derive streaming poll URLs from MCP server registration URLs."""
        streaming: Dict[str, str] = {}
        if not self.config.mcp_config_path or not Path(self.config.mcp_config_path).exists():
            return streaming
        try:
            with open(self.config.mcp_config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, cfg in raw.items():
                url = cfg.get("url", "")
                if url.endswith("/sse"):
                    streaming[name] = url[:-4] + "/stream"
        except Exception:
            pass
        return streaming

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_summary(self, session_id: str) -> str:
        """Extract assistant text from session messages."""
        msgs = self._client.get_messages(session_id)
        texts = []
        for msg in msgs:
            role = (msg.get("info") or {}).get("role") or msg.get("role", "")
            if role != "assistant":
                continue
            for part in msg.get("parts", []):
                if part.get("type") == "text":
                    t = part.get("text", "").strip()
                    if t:
                        texts.append(t)
        return "\n\n".join(texts)

    def _log_tool_event(self, tool: str, title: str, status: str, prefix: str = "") -> None:
        tag = ">" if status == "running" else "v"
        msg = f"  {prefix}[{tool.upper()} {tag}] {title}"
        logger.info(msg)
        if self.progress_callback:
            try:
                self.progress_callback(
                    event_type="sandbox_exec",
                    message=f"{prefix}{tool}: {title}",
                    details={"tool": tool, "status": status},
                )
            except Exception:
                pass

    def _kill_process(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
        self._process = None

    def _report_progress(self, event_type: str, message: str, details: Optional[Dict] = None) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(event_type=event_type, message=message, details=details or {})
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_opencode_binary() -> str:
    """Locate the opencode binary in PATH or common locations."""
    import shutil
    found = shutil.which("opencode")
    if found:
        return found
    # Check common nvm locations
    nvm_dir = os.environ.get("NVM_DIR", os.path.expanduser("~/.nvm"))
    for node_dir in Path(nvm_dir).glob("versions/node/*/bin/opencode"):
        if node_dir.exists():
            return str(node_dir)
    raise FileNotFoundError(
        "opencode binary not found. Install with: npm install -g opencode-ai@latest"
    )
