from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from opensandbox.sandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.execd import ExecutionHandlers, RunCommandOpts

from agent.utils.prompts import get_opencode_runner_prompt

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = None
    find_dotenv = None

if load_dotenv is not None and find_dotenv is not None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

logger = logging.getLogger(__name__)


@dataclass
class OpenCodeConfig:
    model_provider: str = "glm-4.7"
    api_key: str = field(default_factory=lambda: os.getenv("ZHIPUAI_API_KEY") or "")
    sandbox_domain: str = field(
        default_factory=lambda: os.getenv("SANDBOX_DOMAIN")
        or os.getenv("OPENSANDBOX_DOMAIN")
        or "localhost:8080"
    )
    sandbox_image: str = field(
        default_factory=lambda: os.getenv("OPENSANDBOX_IMAGE")
        or "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.1"
    )
    sandbox_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "7200"))
    )
    sandbox_ready_timeout_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("OPENSANDBOX_READY_TIMEOUT_SECONDS", "60")
        )
    )
    sandbox_memory: str = field(
        default_factory=lambda: os.getenv("OPENSANDBOX_MEMORY", "16Gi")
    )
    sandbox_cpu: str = field(default_factory=lambda: os.getenv("OPENSANDBOX_CPU", "4"))
    opencode_install_command: str = "npm install -g opencode-ai@latest"
    debug: bool = field(
        default_factory=lambda: os.getenv("OPENSANDBOX_DEBUG", "false").lower()
        == "true"
    )


class OpenCodeExecutor:
    def __init__(
        self,
        session_id: str,
        task: str,
        bundle_id: Optional[str] = None,
        iteration: int = 0,
        timeout: int = 600,
        config: Optional[OpenCodeConfig] = None,
        progress_callback: Optional[Callable] = None,
        node_name: str = "opencode_executor",
    ):
        self.session_id = session_id
        self.bundle_id = bundle_id
        self.task = task
        self.iteration = iteration
        self.timeout = timeout
        self.config = config or OpenCodeConfig()
        self.sandbox: Sandbox | None = None
        self._logs: list[str] = []
        self.progress_callback = progress_callback
        self.node_name = node_name

        if bundle_id:
            self.workspace_dir = f"/data/sessions/{session_id}/{bundle_id}"
        else:
            self.workspace_dir = f"/data/sessions/{session_id}"

        self._execution_id: Optional[str] = None
        self._execution_result: Any = None
        self._execution_error: Optional[str] = None
        self._log_file_path: Optional[str] = None
        self._tool_events: list[Dict[str, Any]] = []

    @classmethod
    async def execute(
        cls,
        session_id: str,
        task: str,
        bundle_id: Optional[str] = None,
        iteration: int = 0,
        timeout: int = 600,
        config: Optional[OpenCodeConfig] = None,
        progress_callback: Optional[Callable] = None,
        node_name: str = "opencode_executor",
    ) -> Dict[str, Any]:
        executor = cls(
            session_id,
            task,
            bundle_id,
            iteration,
            timeout,
            config,
            progress_callback,
            node_name,
        )
        return await executor.run()

    def _init_log_file(self):
        log_dir = f"/data/sessions/{self.session_id}"
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            pass
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{self.bundle_id}" if self.bundle_id else ""
        self._log_file_path = f"{log_dir}/opencode_log{suffix}_{ts}.txt"
        self._log(f"日志文件: {self._log_file_path}")

    def _write_log_file(self, text: str):
        if not self._log_file_path:
            return
        try:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass

    def _log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] [{self.session_id}] {message}"
        self._logs.append(log_entry)
        self._write_log_file(log_entry)
        print(f"[opencode_executor] {message}")
        getattr(logger, level.lower(), logger.info)(log_entry)

    async def run(self) -> Dict[str, Any]:
        self._init_log_file()
        self._log("开始执行 OpenCode 任务")

        try:
            sandbox = await self._create_sandbox()
            await self._setup_opencode()
            result = await self._execute_task()

            self._log(f"任务执行完成，状态: {result.get('status')}")
            return {
                "status": "success",
                "session_id": self.session_id,
                "logs": self._logs,
                "result": result,
            }
        except Exception as e:
            import traceback

            tb_str = traceback.format_exc()
            self._log(f"任务执行失败: {str(e)}", "ERROR")
            self._log(f"Traceback:\n{tb_str}", "ERROR")
            try:
                await self._write_errors_to_task_log(
                    [
                        {
                            "error_message": f"任务执行失败: {str(e)}",
                            "error_source": "run_exception",
                            "traceback": tb_str[:2000],
                            "task_name": self.task[:100],
                        }
                    ]
                )
            except Exception:
                pass
            return {
                "status": "error",
                "session_id": self.session_id,
                "error": str(e),
                "traceback": tb_str,
                "logs": self._logs,
            }
        finally:
            await self._cleanup()

    async def _create_sandbox(self) -> Sandbox:
        # Use sandbox_timeout_seconds from config for sandbox lifecycle
        # and self.timeout for command execution
        sandbox_timeout = self.config.sandbox_timeout_seconds
        self._log(
            f"步骤1: 创建沙盒 (sandbox_timeout={sandbox_timeout}s, cmd_timeout={self.timeout}s, "
            f"memory={self.config.sandbox_memory}, cpu={self.config.sandbox_cpu})"
        )

        connection_config = ConnectionConfig(
            domain=self.config.sandbox_domain,
            api_key=self.config.api_key,
            request_timeout=timedelta(seconds=max(sandbox_timeout, self.timeout)),
            debug=self.config.debug,
        )

        env = self._build_env()
        image = self.config.sandbox_image
        resource = {
            "cpu": self.config.sandbox_cpu,
            "memory": self.config.sandbox_memory,
        }

        self._log(
            f"创建沙盒: image={image}, domain={self.config.sandbox_domain}, resource={resource}"
        )

        sandbox = await Sandbox.create(
            image,
            connection_config=connection_config,
            timeout=timedelta(seconds=sandbox_timeout),
            ready_timeout=timedelta(seconds=self.config.sandbox_ready_timeout_seconds),
            env=env,
            resource=resource,
        )

        self.sandbox = sandbox
        self._log(f"沙盒创建成功: {sandbox.id}")
        return sandbox

    def _get_stdout(self, result: Any) -> str:
        logs = getattr(result, "logs", None)
        if not logs:
            return ""
        stdout = getattr(logs, "stdout", None)
        if not stdout:
            return ""
        if isinstance(stdout, str):
            return stdout
        return "\n".join(getattr(entry, "text", str(entry)) for entry in stdout)

    async def _setup_opencode(self):
        self._log("步骤2: 初始化 OpenCode 配置")

        check = await self.sandbox.commands.run("which opencode || echo 'not found'")  # type: ignore
        if "not found" not in self._get_stdout(check):
            self._log("OpenCode 已安装")
        else:
            self._log("安装 OpenCode...")
            result = await self.sandbox.commands.run(
                self.config.opencode_install_command
            )  # type: ignore

            if result.error:
                fallback = "curl -fsSL https://opencode.ai/install | bash"
                result = await self.sandbox.commands.run(fallback)  # type: ignore
                if result.error:
                    raise RuntimeError(f"OpenCode 安装失败: {result.error}")

        # 安装 coding-helper 工具
        self._log("安装 coding-helper 工具...")
        coding_helper_result = await self.sandbox.commands.run(
            "npm install -g coding-helper"
        )  # type: ignore
        if coding_helper_result.error:
            self._log(
                f"coding-helper 安装警告: {coding_helper_result.error}", "WARNING"
            )
        else:
            self._log("coding-helper 工具安装完成")

        # 配置 GLM Coding Plan
        self._log("配置 GLM Coding Plan...")
        coding_plan_result = await self.sandbox.commands.run(
            "coding-helper auth glm_coding_plan_china " + self.config.api_key
        )  # type: ignore
        if coding_plan_result.error:
            self._log(
                f"GLM Coding Plan 配置警告: {coding_plan_result.error}", "WARNING"
            )
        else:
            self._log("GLM Coding Plan 配置完成")

        await self._configure_opencode()
        self._log("OpenCode 配置完成")

    async def _configure_opencode(self, workspace_dir: str = "/workspace"):
        self._log("配置 OpenCode 参数")

        model_lower = self.config.model_provider.lower()
        opencode_config: dict = {"$schema": "https://opencode.ai/config.json"}

        # 添加 oh-my-opencode 插件配置
        opencode_config["plugin"] = ["oh-my-opencode@latest"]
        self._log("已配置 oh-my-opencode 插件")

        if "glm" in model_lower:
            version = model_lower.replace("glm-", "").replace("glm", "")
            model_id = f"glm-{version}" if version else "glm-5"
            opencode_config["provider"] = {  # type: ignore
                "zhipuai": {"api": "https://open.bigmodel.cn/api/coding/paas/v4"}
            }
            opencode_config["model"] = f"zhipuai/{model_id}"
        else:
            opencode_config["model"] = self.config.model_provider

        mcp_servers = self._build_mcp_servers_config()
        if mcp_servers:
            opencode_config["mcp"] = mcp_servers
            self._log(f"MCP 配置已加载: {list(mcp_servers.keys())}")

        opencode_config["permission"] = {
            "external_directory": {
                "/data/**": "allow",
                "/tmp/**": "allow",
            },
        }
        self._log(f"权限配置: 仅允许 /data/**,与/tmp/**，拒绝其他所有外部目录")

        opencode_config["instructions"] = ["./AGENTS.md"]

        config_path = f"{self.workspace_dir}/opencode/config/opencode/opencode.json"
        await self.sandbox.files.write_file(
            config_path, json.dumps(opencode_config, indent=2)
        )  # type: ignore

        self._log(f"配置文件已写入: {config_path}")

        # 复制 OpenCode skills 到沙盒
        await self._copy_skills_to_sandbox()

    async def _copy_skills_to_sandbox(self) -> None:
        """
        复制 OpenCode skills 到沙盒

        将 agent/coding_agent/skills/ 下所有子目录整体复制到
        沙盒的 {workspace_dir}/.agents/skills/ 下。
        在沙盒中，HOME 被设置为 workspace_dir，
        因此 OpenCode 可从 ~/.agents/skills/<name>/ 加载 skills。
        """
        import shutil

        skills_src_dir = Path(__file__).parent.parent / "coding_agent" / "skills"

        if not skills_src_dir.exists():
            self._log(f"Skills 目录不存在: {skills_src_dir}")
            return

        skills_dest_dir = f"{self.workspace_dir}/.agents/skills"
        await self.sandbox.commands.run(f"mkdir -p {skills_dest_dir}")

        skill_count = 0
        for skill_dir in sorted(skills_src_dir.iterdir()):
            if not skill_dir.is_dir():
                continue

            dest_skill_dir = f"{skills_dest_dir}/{skill_dir.name}"
            await self.sandbox.commands.run(f"mkdir -p {dest_skill_dir}")

            for file_path in sorted(skill_dir.rglob("*")):
                if not file_path.is_file():
                    continue

                rel_path = file_path.relative_to(skill_dir).as_posix()
                dest_path = f"{dest_skill_dir}/{rel_path}"

                dest_parent = str(Path(rel_path).parent)
                if dest_parent != ".":
                    await self.sandbox.commands.run(
                        f"mkdir -p {dest_skill_dir}/{dest_parent}"
                    )

                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()

                await self.sandbox.files.write_file(dest_path, file_content)

            skill_count += 1
            file_count = sum(1 for _ in skill_dir.rglob("*") if _.is_file())
            self._log(f"已复制 skill: {skill_dir.name} ({file_count} 个文件)")

        self._log(f"共复制 {skill_count} 个 skills 到 {skills_dest_dir}/")

    def _build_mcp_servers_config(self) -> Dict[str, Any]:
        mcp_config_path = (
            Path(__file__).parent.parent / "config" / "mcp_servers_opencode.json"
        )
        if mcp_config_path.exists():
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    async def _execute_task(self) -> Dict[str, Any]:
        self._log(f"步骤3: 执行任务 (timeout={self.timeout}s)")

        await self._create_runner_script()
        await self.sandbox.files.write_file(f"{self.workspace_dir}/task.md", self.task)

        cmd = f"cd {self.workspace_dir} && bash {self.workspace_dir}/runner.sh"
        self._log(f"开始执行: {cmd}")
        self._log(
            f"命令详情 - 长度: {len(cmd)}, timeout: {self.timeout}s, workspace: {self.workspace_dir}"
        )
        start_time = time.time()

        opts = RunCommandOpts(
            timeout=timedelta(seconds=self.timeout),
            working_directory=self.workspace_dir,
        )

        handlers = self._create_nonblocking_handlers()
        self._log(f"Handlers created: {handlers is not None}")

        execution = None
        try:
            self._log(f"调用 sandbox.commands.run() - 开始流式传输")
            execution = await self.sandbox.commands.run(
                cmd, opts=opts, handlers=handlers
            )
            self._log(f"sandbox.commands.run() returned - 流式传输完成")
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self._log(f"执行超时 ({self.timeout}s)", "ERROR")
            self.sandbox = None
            return {
                "status": "error",
                "error": f"Timeout after {self.timeout}s",
                "elapsed_seconds": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            import traceback as _tb
            import httpx

            error_type = type(e).__name__
            error_module = type(e).__module__
            self._log(f"命令执行异常: {e}", "ERROR")
            self._log(f"异常类型: {error_module}.{error_type}", "ERROR")
            self._log(f"已执行时间: {elapsed:.1f}s / {self.timeout}s", "ERROR")

            if isinstance(e, httpx.RemoteProtocolError):
                self._log(f"RemoteProtocolError 详情: {str(e)}", "ERROR")
                tb_str = _tb.format_exc()
                self._log(f"完整堆栈:\n{tb_str[:1000]}", "ERROR")

                recent_events = self._tool_events[-5:] if self._tool_events else []
                if recent_events:
                    self._log(f"最近 {len(recent_events)} 个工具事件:", "ERROR")
                    for i, evt in enumerate(recent_events):
                        self._log(
                            f"  事件 {i + 1}: {evt.get('type', 'unknown')} - {evt.get('tool', evt.get('tool_use_id', 'N/A'))}",
                            "ERROR",
                        )

                self._log(f"可能原因分析:", "ERROR")
                self._log(f"  1. 执行超时 - 当前已执行 {elapsed:.1f}s", "ERROR")
                self._log(
                    f"  2. Sandbox内存不足(OOM) - 检查是否有大数据加载或可视化", "ERROR"
                )
                self._log(f"  3. 网络连接中断", "ERROR")

            self.sandbox = None
            return {"status": "error", "error": str(e), "elapsed_seconds": elapsed}

        elapsed = time.time() - start_time
        self._log(f"命令执行完成 ({elapsed:.1f}s)")

        if not execution:
            self.sandbox = None
            return {
                "status": "error",
                "error": "No execution result",
                "elapsed_seconds": elapsed,
            }

        result = self._convert_execution_to_dict(execution)
        stdout = self._get_stdout(execution)
        stderr_lines = self._get_stderr_lines(execution)
        self._log(f"stdout 长度: {len(stdout)} chars")

        if "===OPENCODE_DONE===" in stdout:
            self._log("检测到完成标记")
        else:
            self._log("未检测到完成标记", "WARNING")

        errors = self._collect_errors_from_output(stderr_lines)
        if errors:
            self._log(f"从输出中提取到 {len(errors)} 条错误信息")
            await self._write_errors_to_task_log(errors)

        await self._write_tool_events_to_task_log()

        await self._ensure_task_log_not_empty(stdout, stderr_lines)

        return result

    def _get_stderr_lines(self, execution) -> List[str]:
        logs = getattr(execution, "logs", None)
        if not logs:
            return []
        stderr = getattr(logs, "stderr", None)
        if not stderr:
            return []
        return [
            getattr(msg, "text", str(msg)) for msg in stderr if getattr(msg, "text", "")
        ]

    def _collect_errors_from_output(
        self, stderr_lines: List[str]
    ) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []

        error_patterns = [
            (
                r"Traceback \(most recent call last\):\n(.+?)(?:\n[A-Za-z_]+Error)",
                "traceback",
            ),
            (r"^.*(?:Error|ERROR|FATAL|FAILED|Exception)\s*:.+$", "error_line"),
            (r"exit code:\s*([1-9]\d*)", "exit_code"),
            (r"command not found:\s*(\S+)", "command_not_found"),
            (r"No such file or directory:\s*['\"]?(\S+?)['\"]?\s*$", "file_not_found"),
            (r"Permission denied:\s*(.+)", "permission_denied"),
            (r"ModuleNotFoundError:\s*(.+)", "module_not_found"),
            (r"ImportError:\s*(.+)", "import_error"),
            (r"ConnectionError:\s*(.+)", "connection_error"),
            (r"TimeoutError:\s*(.+)", "timeout_error"),
            (r"(?:HTTP|request) error:\s*(\d+)\s*(.+)", "http_error"),
        ]

        seen_errors: set = set()

        combined = "\n".join(stderr_lines)

        for pattern, error_type in error_patterns:
            for match in re.finditer(pattern, combined, re.IGNORECASE | re.MULTILINE):
                error_text = match.group(0).strip()
                if len(error_text) > 500:
                    error_text = error_text[:500] + "...(truncated)"
                if "(truncated)" not in error_text and "\n" in error_text:
                    error_text = error_text.split("\n")[0]
                error_key = error_text[:200]
                if error_key not in seen_errors:
                    seen_errors.add(error_key)
                    errors.append(
                        {
                            "error_message": error_text,
                            "error_source": f"output_{error_type}",
                            "task_name": self.task[:100],
                        }
                    )

        if stderr_lines and not errors:
            error_text = "\n".join(stderr_lines[:10])
            if len(error_text) > 1000:
                error_text = error_text[:1000]
            errors.append(
                {
                    "error_message": error_text,
                    "error_source": "stderr_output",
                    "task_name": self.task[:100],
                }
            )

        return errors

    async def _write_errors_to_task_log(self, errors: List[Dict[str, Any]]) -> None:
        if not self.sandbox or not errors:
            return

        import re as _re

        if self.bundle_id:
            log_path = f"/data/sessions/{self.session_id}/output/{self.bundle_id}/task_execution_log.json"
        else:
            log_path = (
                f"/data/sessions/{self.session_id}/output/task_execution_log.json"
            )

        try:
            existing_content = await self.sandbox.files.read_file(log_path)
            if isinstance(existing_content, bytes):
                existing_content = existing_content.decode("utf-8")
            existing_content = existing_content.strip()
            if existing_content:
                records = json.loads(existing_content)
            else:
                records = []
        except Exception:
            records = []

        if not isinstance(records, list):
            records = []

        existing_ids = {r.get("task_id", "") for r in records}
        task_counter = len(records)

        for i, err in enumerate(errors):
            task_counter += 1
            task_id = f"error_{task_counter}"
            while task_id in existing_ids:
                task_counter += 1
                task_id = f"error_{task_counter}"
            existing_ids.add(task_id)

            record = {
                "task_id": task_id,
                "task_name": err.get("task_name", "unknown error"),
                "task_type": "ANALYSIS",
                "status": "failed",
                "input_parameters": {},
                "output_files": [],
                "output_data": {},
                "error_message": err.get("error_message", ""),
            }
            if err.get("error_source"):
                record["error_source"] = err.get("error_source")
            if err.get("traceback"):
                record["traceback"] = err.get("traceback")[:2000]

            records.append(record)

        new_content = json.dumps(records, indent=2, ensure_ascii=False)
        log_dir = _re.sub(r"/[^/]+$", "", log_path)
        try:
            await self.sandbox.commands.run(f"mkdir -p {log_dir}")
        except Exception:
            pass

        try:
            await self.sandbox.files.write_file(log_path, new_content)
            self._log(f"已写入 {len(errors)} 条错误记录到 task_execution_log.json")
        except Exception as e:
            self._log(f"写入 task_execution_log.json 失败: {e}", "ERROR")

    async def _write_tool_events_to_task_log(self) -> None:
        if not self._tool_events:
            return

        tool_uses = [e for e in self._tool_events if e["type"] == "tool_use"]
        tool_results = [e for e in self._tool_events if e["type"] == "tool_result"]

        if not tool_uses:
            return

        result_by_id = {}
        for r in tool_results:
            result_by_id[r.get("tool_use_id", "")] = r

        new_records = []
        for i, tu in enumerate(tool_uses):
            evt_id = tu.get("id", "")

            tool_name = tu.get("tool", "unknown")
            tool_input = tu.get("input", {})
            tr = result_by_id.get(evt_id, {})
            is_error = tr.get("is_error", False)
            content = tr.get("content", "")

            if isinstance(content, list):
                content_str = " ".join(
                    c.get("text", str(c)) if isinstance(c, dict) else str(c)
                    for c in content
                )[:1000]
            elif isinstance(content, str):
                content_str = content[:1000]
            else:
                content_str = str(content)[:1000] if content else ""

            tool_type_map = {
                "bash": "CODE_GENERATION",
                "write": "FILE_OPERATION",
                "edit": "FILE_OPERATION",
                "read": "FILE_OPERATION",
                "glob": "FILE_OPERATION",
                "grep": "FILE_OPERATION",
            }
            task_type = tool_type_map.get(tool_name, "MCP_TOOL")

            cmd = tool_input.get("command", "")
            file_path = tool_input.get("file_path", tool_input.get("path", ""))

            description = cmd[:200] if cmd else file_path
            if not description:
                description = tool_name

            output_files = []
            if file_path and not is_error:
                output_files.append(file_path)

            record = {
                "task_id": f"tool_{i + 1}",
                "task_name": description,
                "task_type": task_type,
                "status": "failed" if is_error else "success",
                "input_parameters": {
                    k: v for k, v in tool_input.items() if k != "content"
                },
                "output_files": output_files,
                "output_data": {
                    "tool_event_id": evt_id,
                    "tool": tool_name,
                    "source": "event_stream",
                    "result_preview": content_str[:500] if content_str else None,
                },
            }
            if is_error and content_str:
                record["error_message"] = content_str[:500]

            new_records.append(record)

        if not new_records:
            return

        sandbox_written = False
        if self.sandbox:
            try:
                await self._write_records_to_sandbox(new_records)
                sandbox_written = True
            except Exception as e:
                self._log(f"写入工具事件到沙盒失败: {e}", "WARNING")
                self.sandbox = None

        if not sandbox_written:
            self._log(f"沙盒不可达，跳过 {len(new_records)} 条工具事件写入")

    async def _write_records_to_sandbox(self, new_records: list) -> None:
        if self.bundle_id:
            log_path = f"/data/sessions/{self.session_id}/output/{self.bundle_id}/task_execution_log.json"
        else:
            log_path = (
                f"/data/sessions/{self.session_id}/output/task_execution_log.json"
            )

        try:
            existing_content = await self.sandbox.files.read_file(log_path)
            if isinstance(existing_content, bytes):
                existing_content = existing_content.decode("utf-8")
            existing_content = existing_content.strip()
            records = json.loads(existing_content) if existing_content else []
        except Exception:
            records = []

        if not isinstance(records, list):
            records = []

        existing_tool_ids = {
            r.get("output_data", {}).get("tool_event_id", "") for r in records
        }
        filtered = [
            r
            for r in new_records
            if r["output_data"]["tool_event_id"] not in existing_tool_ids
        ]

        if not filtered:
            return

        records.extend(filtered)
        new_content = json.dumps(records, indent=2, ensure_ascii=False)

        import re as _re

        log_dir = _re.sub(r"/[^/]+$", "", log_path)
        try:
            await self.sandbox.commands.run(f"mkdir -p {log_dir}")
        except Exception:
            pass

        try:
            await self.sandbox.files.write_file(log_path, new_content)
            self._log(f"已从事件流写入 {len(filtered)} 条工具记录到沙盒")
        except Exception as e:
            self._log(f"写入工具事件到沙盒失败: {e}", "ERROR")
            self.sandbox = None

    async def _ensure_task_log_not_empty(
        self, stdout: str, stderr_lines: List[str]
    ) -> None:
        if not self.sandbox:
            return

        if self.bundle_id:
            log_path = f"/data/sessions/{self.session_id}/output/{self.bundle_id}/task_execution_log.json"
        else:
            log_path = (
                f"/data/sessions/{self.session_id}/output/task_execution_log.json"
            )

        try:
            existing_content = await self.sandbox.files.read_file(log_path)
            if isinstance(existing_content, bytes):
                existing_content = existing_content.decode("utf-8")
            records = json.loads(existing_content.strip())
            if isinstance(records, list) and len(records) > 0:
                return
        except Exception:
            pass

        import re as _re

        has_done_marker = "===OPENCODE_DONE===" in stdout
        has_errors = bool(stderr_lines)

        summary_status = "success" if has_done_marker and not has_errors else "failed"

        summary_reasons = []
        if not has_done_marker:
            summary_reasons.append("未检测到 OpenCode 完成标记")
        if has_errors:
            summary_reasons.append(f"stderr 包含 {len(stderr_lines)} 行输出")
        if not stdout.strip():
            summary_reasons.append("stdout 为空")

        summary_record = {
            "task_id": "execution_summary",
            "task_name": "OpenCode 执行总览",
            "task_type": "ANALYSIS",
            "status": summary_status,
            "input_parameters": {"task_preview": self.task[:200]},
            "output_files": [],
            "output_data": {
                "has_done_marker": has_done_marker,
                "stderr_line_count": len(stderr_lines),
                "stdout_length": len(stdout),
                "summary_reasons": summary_reasons,
            },
        }

        if stderr_lines:
            summary_record["error_message"] = "\n".join(stderr_lines[:5])
        elif not has_done_marker:
            summary_record["error_message"] = "; ".join(summary_reasons)

        new_content = json.dumps([summary_record], indent=2, ensure_ascii=False)
        log_dir = _re.sub(r"/[^/]+$", "", log_path)
        try:
            await self.sandbox.commands.run(f"mkdir -p {log_dir}")
        except Exception:
            pass

        try:
            await self.sandbox.files.write_file(log_path, new_content)
            self._log(
                f"task_execution_log.json 为空，已写入执行总览记录 (status={summary_status})"
            )
        except Exception as e:
            self._log(f"写入执行总览到 task_execution_log.json 失败: {e}", "ERROR")

    def _create_nonblocking_handlers(self) -> ExecutionHandlers | None:
        progress_queue = self._get_progress_queue()
        if not progress_queue:
            self._log("No progress queue, handlers disabled")
            return None

        node_name = self.node_name
        session_id = self.session_id
        log_func = self._log
        tool_events = self._tool_events
        event_counter = [0]
        import time as _time

        def emit(evt_type: str, msg: str, details: dict = None):
            try:
                from backend.progress_tracker import ProgressEvent, ProgressEventType

                event = ProgressEvent(
                    session_id=session_id,
                    event_type=ProgressEventType(evt_type),
                    node_name=node_name,
                    message=msg,
                    details=details or {},
                )
                progress_queue.put_nowait(event)
            except Exception as e:
                log_func(f"emit error: {e}", "ERROR")

        async def on_init(evt):
            event_counter[0] += 1
            ts = _time.strftime("%H:%M:%S")
            log_func(f"[{ts}] [Handler #{event_counter[0]}] on_init")
            emit("opencode_init", "OpenCode started", {"id": getattr(evt, "id", None)})

        async def on_stdout(msg):
            event_counter[0] += 1
            ts = _time.strftime("%H:%M:%S")
            text = msg.text.strip() if msg.text else ""
            if not text:
                return

            log_func(f"[{ts}] [stdout #{event_counter[0]}] {text[:200]}")
            log_func(f"[RAW stdout] {text}", "RAW")

            try:
                data = json.loads(text)
                evt_type = data.get("type", "unknown")
                log_func(f"[JSON] type={evt_type}")

                if evt_type == "thinking":
                    content = data.get("content", "")[:500]
                    emit("opencode_thinking", content, {"type": evt_type})
                elif evt_type == "tool_use":
                    tool = data.get("name", "unknown")
                    tool_events.append(
                        {
                            "type": "tool_use",
                            "tool": tool,
                            "input": data.get("input", {}),
                            "id": data.get("id", ""),
                            "timestamp": str(msg.timestamp) if msg.timestamp else "",
                        }
                    )
                    emit(
                        "opencode_tool",
                        f"Using: {tool}",
                        {"type": evt_type, "tool": tool},
                    )
                elif evt_type == "tool_result":
                    tool_events.append(
                        {
                            "type": "tool_result",
                            "content": data.get("content", ""),
                            "is_error": data.get("is_error", False),
                            "tool_use_id": data.get("tool_use_id", ""),
                            "timestamp": str(msg.timestamp) if msg.timestamp else "",
                        }
                    )
                    emit("opencode_tool_result", "Tool completed", {"type": evt_type})
                elif evt_type == "message":
                    content = data.get("content", "")[:500]
                    emit("opencode_message", content, {"type": evt_type})
                else:
                    emit("opencode_event", f"[{evt_type}]", {"type": evt_type})
            except json.JSONDecodeError:
                emit("opencode_stdout", text, {"ts": str(msg.timestamp)})

        async def on_stderr(msg):
            event_counter[0] += 1
            ts = _time.strftime("%H:%M:%S")
            text = msg.text.strip() if msg.text else ""
            if text:
                log_func(
                    f"[{ts}] [stderr #{event_counter[0]}] {text[:50]}...", "WARNING"
                )
                emit("opencode_stderr", text, {"ts": str(msg.timestamp)})

        async def on_result(res):
            event_counter[0] += 1
            ts = _time.strftime("%H:%M:%S")
            log_func(f"[{ts}] [Handler #{event_counter[0]}] on_result")
            emit("opencode_result", "Result", {"ts": str(res.timestamp)})

        async def on_error(err):
            event_counter[0] += 1
            ts = _time.strftime("%H:%M:%S")
            log_func(
                f"[{ts}] [Handler #{event_counter[0]}] on_error: {err.name}", "ERROR"
            )
            emit("opencode_error", f"{err.name}: {err.value}", {"value": err.value})

        async def on_execution_complete(evt):
            event_counter[0] += 1
            ts = _time.strftime("%H:%M:%S")
            log_func(f"[{ts}] [Handler #{event_counter[0]}] on_execution_complete")
            emit(
                "opencode_complete",
                "Done",
                {"ms": getattr(evt, "execution_time_in_millis", 0)},
            )

        return ExecutionHandlers(
            on_init=on_init,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            on_result=on_result,
            on_error=on_error,
            on_execution_complete=on_execution_complete,
        )

    def _get_progress_queue(self):
        try:
            from backend.progress_tracker import get_progress_tracker

            tracker = get_progress_tracker(self.session_id)
            if tracker:
                return tracker.queue
        except Exception as e:
            self._log(f"get queue error: {e}", "ERROR")
        return None

    def _create_streaming_handlers_with_lifecycle(self) -> ExecutionHandlers:
        progress_callback = self.progress_callback
        node_name = self.node_name
        log_func = self._log
        state = self

        def _safe_callback(*args, **kwargs):
            if not progress_callback:
                return
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon(lambda: progress_callback(*args, **kwargs))
            except Exception as e:
                log_func(f"[Handler] callback error: {e}", "ERROR")

        async def on_init(evt):
            state._execution_id = evt.id
            log_func(f"[Handler] on_init: execution_id={evt.id}")
            _safe_callback(
                event_type="opencode_init",
                message="OpenCode execution started",
                node_name=node_name,
                details={"execution_id": evt.id},
            )

        async def on_stdout(msg):
            text = msg.text.strip() if msg.text else ""
            if not text:
                return
            _safe_callback(
                event_type="opencode_stdout",
                message=text,
                node_name=node_name,
                details={"timestamp": msg.timestamp},
            )

        async def on_stderr(msg):
            text = msg.text.strip() if msg.text else ""
            if text:
                log_func(f"[Handler] stderr: {text}", "WARNING")
                _safe_callback(
                    event_type="opencode_stderr",
                    message=text,
                    node_name=node_name,
                    details={"timestamp": msg.timestamp, "is_error": True},
                )

        async def on_result(res):
            state._execution_result = res
            log_func(f"[Handler] on_result received")
            _safe_callback(
                event_type="opencode_result",
                message="Result received",
                node_name=node_name,
                details={"timestamp": res.timestamp},
            )

        async def on_error(err):
            state._execution_error = f"{err.name}: {err.value}"
            log_func(f"[Handler] on_error: {err.name}: {err.value}", "ERROR")
            _safe_callback(
                event_type="opencode_error",
                message=f"Error: {err.name}",
                node_name=node_name,
                details={"error_name": err.name, "error_value": err.value},
            )

        async def on_execution_complete(evt):
            log_func(
                f"[Handler] on_execution_complete: time={evt.execution_time_in_millis}ms"
            )
            _safe_callback(
                event_type="opencode_complete",
                message="Execution completed",
                node_name=node_name,
                details={"execution_time_ms": evt.execution_time_in_millis},
            )

        return ExecutionHandlers(
            on_init=on_init,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            on_result=on_result,
            on_error=on_error,
            on_execution_complete=on_execution_complete,
        )

    def _convert_execution_to_dict(self, execution) -> Dict[str, Any]:
        result = {
            "status": "success",
            "execution_id": execution.id,
            "logs": {
                "stdout": [],
                "stderr": [],
            },
        }

        if execution.logs:
            if execution.logs.stdout:
                result["logs"]["stdout"] = [
                    {"text": msg.text, "timestamp": msg.timestamp}
                    for msg in execution.logs.stdout
                ]
            if execution.logs.stderr:
                result["logs"]["stderr"] = [
                    {"text": msg.text, "timestamp": msg.timestamp}
                    for msg in execution.logs.stderr
                ]

        if execution.result:
            result["results"] = [
                {"text": r.text, "timestamp": r.timestamp} for r in execution.result
            ]

        if execution.error:
            result["status"] = "error"
            result["error"] = {
                "name": execution.error.name,
                "value": execution.error.value,
                "traceback": execution.error.traceback,
            }

        return result

    async def _create_runner_script(self) -> str:
        script = get_opencode_runner_prompt(
            session_id=self.session_id, bundle_id=self.bundle_id
        )
        script_path = f"{self.workspace_dir}/runner.sh"
        await self.sandbox.files.write_file(script_path, script)  # type: ignore
        await self.sandbox.commands.run(f"chmod +x {script_path}")  # type: ignore
        return script_path

    def _build_env(self) -> Dict[str, str]:
        env = {
            "XDG_DATA_HOME": f"{self.workspace_dir}/opencode/data",
            "XDG_CONFIG_HOME": f"{self.workspace_dir}/opencode/config",
            "XDG_STATE_HOME": f"{self.workspace_dir}/opencode/state",
            "HOME": self.workspace_dir,
            "NODE_OPTIONS": "--no-warnings",
            "R_LIBS_USER": f"{self.workspace_dir}/R/library",
        }

        model_lower = self.config.model_provider.lower()

        if "glm" in model_lower and self.config.api_key:
            env["ZHIPUAI_API_KEY"] = self.config.api_key
            env["ZHIPU_API_KEY"] = self.config.api_key
            env["OPENAI_API_KEY"] = "not-used"
            env["ANTHROPIC_API_KEY"] = "not-used"
        elif "claude" in model_lower and self.config.api_key:
            env["ANTHROPIC_API_KEY"] = self.config.api_key
            env["OPENAI_API_KEY"] = "not-used"
        elif "gpt" in model_lower or "openai" in model_lower:
            if self.config.api_key:
                env["OPENAI_API_KEY"] = self.config.api_key
            env["ANTHROPIC_API_KEY"] = "not-used"

        return env

    async def _cleanup(self):
        if self.sandbox:
            self._log("清理沙盒资源")
            try:
                await self.sandbox.close()
            except Exception as e:
                self._log(f"清理时出错: {e}", "WARNING")
