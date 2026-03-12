"""
LocalSandbox — drop-in replacement for OpenSandbox SDK for local execution.

Implements the same async API surface used throughout the codebase:
  sandbox.files.write_file(path, content)
  sandbox.files.read_file(path)
  sandbox.commands.run(cmd)
  sandbox.kill() / sandbox.close()
  sandbox.id
"""

from __future__ import annotations

import asyncio
import os
import uuid
from asyncio.subprocess import PIPE
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class _Logs:
    stdout: str = ""
    stderr: str = ""


@dataclass
class CommandResult:
    logs: _Logs = field(default_factory=_Logs)
    error: Optional[str] = None
    exit_code: int = 0


class LocalFiles:
    async def write_file(self, path: str, content) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")


class LocalCommands:
    def __init__(self, env: Dict[str, str]):
        self._env = env

    async def run(self, cmd: str) -> CommandResult:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=PIPE, stderr=PIPE, env=self._env,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        return CommandResult(
            logs=_Logs(stdout=stdout, stderr=stderr),
            error=stderr if proc.returncode != 0 else None,
            exit_code=proc.returncode or 0,
        )


class LocalSandbox:
    def __init__(self, env: Dict[str, str]):
        self.id = f"local-{uuid.uuid4().hex[:8]}"
        self.files = LocalFiles()
        self.commands = LocalCommands(env)

    async def kill(self):
        pass

    async def close(self):
        pass
