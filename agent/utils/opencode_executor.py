from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Callable

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
    model_provider: str = "glm-4.5"
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
        default_factory=lambda: int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "1800"))
    )
    sandbox_ready_timeout_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("OPENSANDBOX_READY_TIMEOUT_SECONDS", "30")
        )
    )
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

    def _log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] [{self.session_id}] {message}"
        self._logs.append(log_entry)
        getattr(logger, level.lower(), logger.info)(log_entry)

    async def run(self) -> Dict[str, Any]:
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
        self._log(f"步骤1: 创建沙盒 (timeout={self.timeout}s)")

        connection_config = ConnectionConfig(
            domain=self.config.sandbox_domain,
            api_key=self.config.api_key,
            request_timeout=timedelta(seconds=self.timeout),
            debug=self.config.debug,
        )

        env = self._build_env()
        image = self.config.sandbox_image

        self._log(f"创建沙盒: image={image}, domain={self.config.sandbox_domain}")

        sandbox = await Sandbox.create(
            image,
            connection_config=connection_config,
            timeout=timedelta(seconds=self.timeout),
            ready_timeout=timedelta(seconds=self.config.sandbox_ready_timeout_seconds),
            env=env,
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
            return

        self._log("安装 OpenCode...")
        result = await self.sandbox.commands.run(self.config.opencode_install_command)  # type: ignore

        if result.error:
            fallback = "curl -fsSL https://opencode.ai/install | bash"
            result = await self.sandbox.commands.run(fallback)  # type: ignore
            if result.error:
                raise RuntimeError(f"OpenCode 安装失败: {result.error}")

        await self._configure_opencode()
        self._log("OpenCode 配置完成")

    async def _configure_opencode(self, workspace_dir: str = "/workspace"):
        self._log("配置 OpenCode 参数")

        model_lower = self.config.model_provider.lower()
        opencode_config: dict = {"$schema": "https://opencode.ai/config.json"}

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
            },
        }
        self._log(
            f"权限配置: 仅允许 /data/**，拒绝其他所有外部目录"
        )

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

        OpenCode 从 ~/.agents/skills/<name>/SKILL.md 加载 skills
        在沙盒中，HOME 被设置为 workspace_dir
        """
        skills_src_dir = Path(__file__).parent.parent / "coding_agent" / "skills"

        if not skills_src_dir.exists():
            self._log(f"Skills 目录不存在: {skills_src_dir}")
            return

        skills_dest_dir = f"{self.workspace_dir}/.agents/skills"
        await self.sandbox.commands.run(f"mkdir -p {skills_dest_dir}")

        skill_count = 0
        for skill_dir in skills_src_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    with open(skill_file, "r", encoding="utf-8") as f:
                        skill_content = f.read()

                    dest_path = f"{skills_dest_dir}/{skill_dir.name}/SKILL.md"
                    await self.sandbox.commands.run(
                        f"mkdir -p {skills_dest_dir}/{skill_dir.name}"
                    )
                    await self.sandbox.files.write_file(dest_path, skill_content)
                    skill_count += 1

        if skill_count > 0:
            self._log(f"已复制 {skill_count} 个 skills 到 {skills_dest_dir}/")

    def _build_mcp_servers_config(self) -> Dict[str, Any]:
        mcp_config_path = (
            Path(__file__).parent.parent / "config" / "mcp_servers_opencode.json"
        )
        if mcp_config_path.exists():
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    async def _execute_task(self) -> Dict[str, Any]:
        self._log(
            f"步骤3: 执行任务 (iteration={self.iteration}, timeout={self.timeout}s)"
        )

        await self._create_runner_script()
        await self.sandbox.files.write_file(f"{self.workspace_dir}/task.md", self.task)

        cmd = f"cd {self.workspace_dir} && bash {self.workspace_dir}/runner.sh"

        handlers = self._create_streaming_handlers()

        opts = RunCommandOpts(
            timeout=timedelta(seconds=self.timeout),
            working_directory=self.workspace_dir,
        )

        self._log(f"开始执行命令: {cmd[:100]}...")
        start_time = time.time()

        try:
            execution = await self.sandbox.commands.run(
                cmd, opts=opts, handlers=handlers
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self._log(f"命令执行异常 ({elapsed:.1f}s): {e}", "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "elapsed_seconds": elapsed,
            }

        elapsed = time.time() - start_time
        self._log(f"命令执行完成 ({elapsed:.1f}s)")

        result = self._convert_execution_to_dict(execution)

        if execution.error:
            self._log(
                f"命令返回错误: {execution.error.name}: {execution.error.value}",
                "ERROR",
            )
            result["error_details"] = {
                "name": execution.error.name,
                "value": execution.error.value,
                "traceback": execution.error.traceback,
            }

        stdout = self._get_stdout(execution)
        if "===OPENCODE_DONE===" not in stdout:
            self._log("警告: 未检测到完成标记 '===OPENCODE_DONE==='", "WARNING")
            self._log(f"stdout 前500字符: {stdout[:500]}", "WARNING")
            if elapsed < 5:
                self._log("命令执行时间过短，可能未正常执行", "WARNING")
                result["status"] = "error"
                result["error"] = "命令执行时间过短，OpenCode 可能未正常启动"
                result["elapsed_seconds"] = elapsed

        return result

    def _create_streaming_handlers(self) -> ExecutionHandlers | None:
        if not self.progress_callback:
            return None

        progress_callback = self.progress_callback
        node_name = self.node_name
        session_id = self.session_id

        async def on_init(event):
            progress_callback(
                event_type="opencode_init",
                message=f"OpenCode execution started",
                node_name=node_name,
                details={
                    "execution_id": event.id,
                    "session_id": session_id,
                },
            )

        async def on_stdout(message):
            text = message.text.strip() if message.text else ""
            if text:
                progress_callback(
                    event_type="opencode_stdout",
                    message=text,
                    node_name=node_name,
                    details={
                        "timestamp": message.timestamp,
                        "is_error": False,
                    },
                )

        async def on_stderr(message):
            text = message.text.strip() if message.text else ""
            if text:
                progress_callback(
                    event_type="opencode_stderr",
                    message=text,
                    node_name=node_name,
                    details={
                        "timestamp": message.timestamp,
                        "is_error": True,
                    },
                )

        async def on_result(result):
            text = result.text if result.text else ""
            progress_callback(
                event_type="opencode_result",
                message="OpenCode execution result received",
                node_name=node_name,
                details={
                    "result_text": text[:500] if text else None,
                    "timestamp": result.timestamp,
                },
            )

        async def on_error(error):
            progress_callback(
                event_type="opencode_error",
                message=f"OpenCode error: {error.name}: {error.value}",
                node_name=node_name,
                details={
                    "error_name": error.name,
                    "error_value": error.value,
                    "traceback": error.traceback,
                },
            )

        async def on_execution_complete(event):
            progress_callback(
                event_type="opencode_complete",
                message="OpenCode execution completed",
                node_name=node_name,
                details={
                    "execution_time_ms": event.execution_time_in_millis,
                    "timestamp": event.timestamp,
                },
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
