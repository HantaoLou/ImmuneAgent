"""
OpenCode Executor - 在 OpenSandbox 中运行 OpenCode

这是核心执行模块，提供以下功能：
1. 创建和管理 OpenSandbox 沙盒实例
2. 在沙盒中安装和配置 OpenCode
3. 执行 tasks.md 中的任务
4. 收集和返回执行结果

架构：
┌─────────────────────────────────────────────────────────────────────┐
│                         Bio-Agent 主流程                              │
│                    (main_graph.py - 流程编排)                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ tasks.md + context.json
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OpenSandbox 沙盒环境                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    OpenCode Agent (TUI)                        │  │
│  │  • 读取 tasks.md                                               │  │
│  │  • 执行任务（MCP 工具调用 / 代码生成）                           │  │
│  │  • 生成输出文件                                                 │  │
│  │  • 更新任务状态                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from coding_agent.config import (
    ExecutionResult,
    ExecutionStatus,
    OpenCodeConfig,
    OpenCodeMode,
    TaskExecutionRecord,
    TaskType,
    DEFAULT_MCP_CONFIG,
    DEFAULT_OPENCODE_CONFIG,
)
from coding_agent.prompts import get_opencode_prompt


class OpenCodeExecutor:
    """
    在 OpenSandbox 中运行 OpenCode 的执行器

    使用方式:
        executor = OpenCodeExecutor(config)
        sandbox = await executor.create_sandbox()
        result = await executor.execute_tasks(tasks_md_path)
        await executor.cleanup()
    """

    def __init__(
        self,
        config: Optional[OpenCodeConfig] = None,
        sandbox_domain: Optional[str] = None,
        model_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ):
        """
        初始化 OpenCode 执行器

        Args:
            config: OpenCode 配置对象（如果提供，其他参数将被忽略）
            sandbox_domain: OpenSandbox 服务域名
            model_provider: 模型提供商（glm-4.7, claude-sonnet-4, gpt-4o 等）
            api_key: API 密钥
            progress_callback: SSE 进度回调函数，用于实时推送执行状态
        """
        if config:
            self.config = config
        else:
            self.config = OpenCodeConfig(
                sandbox_domain=sandbox_domain or "localhost:8080",
                model_provider=model_provider or "glm-4.7",
                api_key=api_key,
            )

        self.sandbox = None
        self._installed = False
        self.progress_callback = progress_callback

    def _report_progress(
        self, event_type: str, message: str, details: Optional[Dict] = None
    ):
        """通过 progress_callback 报告执行进度"""
        if self.progress_callback:
            try:
                self.progress_callback(
                    event_type=event_type,
                    message=message,
                    details=details or {},
                )
            except Exception as e:
                print(f"[OpenCodeExecutor] Error reporting progress: {e}")

    async def create_sandbox(self, image: Optional[str] = None) -> Any:
        """
        创建沙盒实例

        Args:
            image: Docker 镜像名称（可选，默认使用配置中的镜像）

        Returns:
            Sandbox 实例
        """
        try:
            from opensandbox.sandbox import Sandbox
            from opensandbox.config import ConnectionConfig
        except ImportError as e:
            raise RuntimeError(
                f"OpenSandbox SDK 未安装。请运行: pip install opensandbox\n"
                f"详细错误: {e}"
            )

        # 准备连接配置
        connection_config = ConnectionConfig(
            domain=self.config.sandbox_domain,
            api_key=self.config.api_key,
            request_timeout=timedelta(seconds=self.config.sandbox_timeout_seconds),
            debug=self.config.debug,
        )

        # 准备环境变量
        env = self._build_sandbox_env()

        # CRITICAL: 镜像优先级
        # 1. 显式传入的 image 参数（最高优先级）
        # 2. 环境变量 OPENSANDBOX_IMAGE（推荐方式）
        # 3. self.config.sandbox_image（配置中的值）
        #
        # 注意：无论什么情况下，都必须优先使用 .env 中的 OPENSANDBOX_IMAGE
        env_image = os.getenv("OPENSANDBOX_IMAGE")
        if env_image:
            # 环境变量存在，忽略传入参数和配置中的值
            image = env_image
            self._log(f"[OK] 使用环境变量 OPENSANDBOX_IMAGE 中的镜像: {image}")
        elif image:
            self._log(f"使用传入的镜像参数: {image}")
        else:
            image = self.config.sandbox_image
            # 检查配置中的镜像是否与环境变量一致
            if env_image and image != env_image:
                self._log(
                    f"[WARN] 警告: 配置中的镜像 ({image}) 与环境变量 OPENSANDBOX_IMAGE ({env_image}) 不一致"
                )
                self._log(f"[WARN] 将使用环境变量中的镜像: {env_image}")
                image = env_image
            self._log(f"使用配置中的镜像: {image}")

        self._log(f"创建沙盒: image={image}, domain={self.config.sandbox_domain}")

        # 创建沙盒
        self.sandbox = await Sandbox.create(
            image,
            connection_config=connection_config,
            timeout=timedelta(seconds=self.config.sandbox_timeout_seconds),
            ready_timeout=timedelta(seconds=self.config.sandbox_ready_timeout_seconds),
            env=env,
        )

        self._log(f"沙盒创建成功: {getattr(self.sandbox, 'id', 'N/A')}")

        # 安装 OpenCode
        await self._install_opencode()

        return self.sandbox

    async def _install_opencode(self) -> None:
        """在沙盒中安装 OpenCode"""
        if self._installed:
            return

        self._log("安装 OpenCode...")

        # 检查是否已安装
        check_result = await self.sandbox.commands.run(
            "which opencode || echo 'not found'"
        )
        if "not found" not in self._get_stdout(check_result):
            self._log("OpenCode 已安装")
            self._installed = True
            return

        # 安装 OpenCode
        install_cmd = self.config.opencode_install_command
        result = await self.sandbox.commands.run(install_cmd)

        if result.error:
            # 尝试备用安装方式
            self._log("npm 安装失败，尝试备用方式...")
            fallback_cmd = "curl -fsSL https://opencode.ai/install | bash"
            result = await self.sandbox.commands.run(fallback_cmd)

            if result.error:
                raise RuntimeError(f"OpenCode 安装失败: {result.error}")

        self._log("OpenCode 安装成功")
        self._installed = True

        # 对于 GLM 模型，启动本地代理
        model_lower = self.config.model_provider.lower()
        if "glm" in model_lower:
            await self._setup_glm_proxy()

    async def _setup_glm_proxy(self) -> None:
        """设置 GLM 代理服务器"""
        self._log("设置 GLM 代理...")

        # 1. 上传代理脚本
        proxy_script_path = Path(__file__).parent / "glm_proxy.py"
        if proxy_script_path.exists():
            proxy_content = proxy_script_path.read_text(encoding="utf-8")
            # 替换 API Key 占位符
            proxy_content = proxy_content.replace(
                'os.getenv("ZHIPU_API_KEY")', f'"{self.config.api_key}"'
            )
            await self.sandbox.files.write_file("/root/glm_proxy.py", proxy_content)
            self._log("已上传 GLM 代理脚本")
        else:
            self._log("警告: 找不到 glm_proxy.py，使用直接连接模式")
            return

        # 2. 安装依赖
        await self.sandbox.commands.run("pip install aiohttp httpx -q")

        # 3. 启动代理服务器（后台运行）
        result = await self.sandbox.commands.run(
            "nohup python /root/glm_proxy.py --port 8765 > /tmp/glm_proxy.log 2>&1 &"
        )

        # 4. 等待代理启动
        import asyncio

        await asyncio.sleep(2)

        # 5. 验证代理是否运行
        check_result = await self.sandbox.commands.run(
            "curl -s http://localhost:8765/v1/models | head -c 100 || echo 'proxy_not_ready'"
        )
        if "proxy_not_ready" in self._get_stdout(check_result):
            self._log("警告: GLM 代理启动失败，使用直接连接模式")
        else:
            self._log("GLM 代理启动成功: http://localhost:8765")
            self._glm_proxy_url = "http://localhost:8765"

    async def _configure_opencode(self, workspace_dir: str = "/workspace") -> None:
        """
        配置 OpenCode（模型、MCP 等）

        OpenCode 配置文件格式（官方源码 internal/config/config.go）:
        - 文件名: .opencode.json
        - 位置优先级:
          1. 工作目录/.opencode.json (最高)
          2. ~/.opencode.json
          3. ~/.config/opencode/.opencode.json

        智谱 GLM Coding Plan 配置参考:
        https://docs.bigmodel.cn/cn/coding-plan/tool/opencode
        """
        import copy

        self._log("配置 OpenCode...")

        # 构建配置 - 只使用 OpenCode 支持的配置键
        # 参考：https://opencode.ai/docs/config/
        opencode_config = {
            "$schema": "https://opencode.ai/config.json",
        }

        # 模型配置 - 使用智谱 GLM
        model_lower = self.config.model_provider.lower()

        if "glm" in model_lower:
            # 智谱 GLM 配置
            # 提取模型版本（glm-5, glm-4.7, glm-4-flash 等）
            model_version = model_lower.replace("glm-", "").replace("glm", "")
            model_id = f"glm-{model_version}" if model_version else "glm-5"

            # 配置智谱 provider（使用 Coding API 端点）
            opencode_config["provider"] = {
                "zhipuai": {"api": "https://open.bigmodel.cn/api/coding/paas/v4"}
            }

            # 设置模型（OpenCode 使用 "model" 字段，不是 "agents"）
            opencode_config["model"] = f"zhipuai/{model_id}"
        else:
            # 其他模型
            opencode_config["model"] = self.config.model_provider

        # MCP 服务器配置
        # OpenCode 使用 "mcp" 作为顶级键（参考官方文档 https://opencode.ai/docs/config/）
        mcp_servers = self._build_mcp_servers_config()
        if mcp_servers:
            opencode_config["mcp"] = mcp_servers

        config_json = json.dumps(opencode_config, indent=2)
        self._log(f"OpenCode 配置:\n{config_json}")

        # OpenCode 配置文件位置优先级（根据官方文档）：
        # 1. ~/.config/opencode/opencode.json（全局配置）
        # 2. {workspace_dir}/opencode.json（项目配置，不带点）
        #
        # 由于我们设置了 XDG_CONFIG_HOME=/tmp/opencode/config，全局配置位置为：
        # /tmp/opencode/config/opencode/opencode.json

        # 全局配置路径（OpenCode 会从这里读取）
        global_config_path = "/tmp/opencode/config/opencode/opencode.json"

        # 清理所有可能存在的旧配置文件（避免残留 "agents" 等不支持的键）
        # 注意：不清理要写入的目标路径
        old_config_paths = [
            "/tmp/opencode/config/opencode/config.json",
            "/tmp/opencode/.opencode.json",
            f"{workspace_dir}/opencode.json",
            f"{workspace_dir}/.opencode.json",
            "/root/.config/opencode/opencode.json",
        ]
        for old_path in old_config_paths:
            await self.sandbox.commands.run(f"rm -f {old_path} 2>/dev/null || true")
        self._log(f"已清理旧配置文件: {old_config_paths}")

        # 确保目录存在并写入全局配置
        dir_path = "/".join(global_config_path.split("/")[:-1])
        await self.sandbox.commands.run(f"mkdir -p {dir_path}")
        await self.sandbox.files.write_file(global_config_path, config_json)

        # 验证配置文件是否写入成功
        check_result = await self.sandbox.commands.run(
            f"cat {global_config_path} | head -5"
        )
        self._log(f"配置文件验证: {self._get_stdout(check_result)[:200]}")

        self._log(f"OpenCode 配置已写入: {global_config_path}")

    def _build_mcp_servers_config(self) -> Dict[str, Any]:
        """
        构建 MCP 服务器配置（使用 OpenCode 原生格式）

        直接读取 mcp_servers_opencode.json，无需格式转换。

        OpenCode 官方格式:
        {
            "server_name": {
                "type": "remote",
                "url": "http://...",     # 不带 /sse 后缀，OpenCode 会自动添加
                "enabled": true,
                "timeout": 5000
            }
        }
        """
        opencode_config_path = (
            Path(__file__).parent.parent / "config" / "mcp_servers_opencode.json"
        )

        if opencode_config_path.exists():
            with open(opencode_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self._log(f"已加载 OpenCode MCP 配置: {opencode_config_path}")
                return config

        self._log(f"警告: 未找到 OpenCode MCP 配置文件: {opencode_config_path}")
        return {}

    def _build_sandbox_env(self) -> Dict[str, str]:
        """构建沙盒环境变量"""
        env = {}

        # OpenCode XDG 目录配置
        # 将 OpenCode 的数据目录指向 /tmp/opencode/ 而不是 /root/
        # 这避免触发 "external_directory (/root/*)" 权限请求
        env["XDG_DATA_HOME"] = "/tmp/opencode/data"
        env["XDG_CONFIG_HOME"] = "/tmp/opencode/config"
        env["XDG_STATE_HOME"] = "/tmp/opencode/state"
        # 设置 HOME 环境变量（某些情况下 OpenCode 仍会使用）
        env["HOME"] = "/tmp/opencode"

        # 根据模型类型设置 API Key 和相关配置
        model_lower = self.config.model_provider.lower()

        if "glm" in model_lower and self.config.api_key:
            # 智谱 GLM Coding Plan 配置
            # 参考：https://docs.bigmodel.cn/cn/coding-plan/tool/opencode
            env["ZHIPUAI_API_KEY"] = self.config.api_key
            # 兼容旧版变量名
            env["ZHIPU_API_KEY"] = self.config.api_key
            # 添加必要的占位符
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
            # 默认使用配置的模型
            env["OPENCODE_MODEL"] = self.config.model_provider
            if "OPENAI_API_KEY" not in env:
                env["OPENAI_API_KEY"] = self.config.api_key or "not-used"
            if "ANTHROPIC_API_KEY" not in env:
                env["ANTHROPIC_API_KEY"] = "not-used"

        # 设置 Node.js 环境变量
        env["NODE_OPTIONS"] = "--no-warnings"

        return env

    async def execute_tasks(
        self,
        tasks_md_path: str,
        workspace_dir: str = "/workspace",
        mode: Optional[OpenCodeMode] = None,
        iteration: int = 0,
    ) -> ExecutionResult:
        """
        执行 tasks.md 中的任务

        Args:
            tasks_md_path: tasks.md 文件路径（沙盒内）
            workspace_dir: 工作目录
            mode: 执行模式（build 或 plan）
            iteration: 迭代编号（用于生成独立的日志文件）

        Returns:
            ExecutionResult: 执行结果（包含 task_records 任务执行记录）
        """
        start_time = time.time()

        if not self.sandbox:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error="沙盒未创建，请先调用 create_sandbox()",
            )

        mode = mode or self.config.opencode_mode

        try:
            # 1. 配置 OpenCode
            self._report_progress(
                "sandbox_exec", "📦 配置 OpenCode 环境...", {"phase": "configure"}
            )
            await self._configure_opencode(workspace_dir)

            # 2. 确保输出目录存在
            await self.sandbox.commands.run(f"mkdir -p {workspace_dir}/output")

            # 3. 生成执行脚本（避免命令行传参限制）
            self._report_progress(
                "sandbox_exec", "📝 生成执行脚本...", {"phase": "script_generation"}
            )
            runner_script = await self._generate_runner_script(
                workspace_dir, tasks_md_path, mode, iteration
            )

            # 4. 使用 stdbuf 后台执行脚本（无缓冲，实时输出）
            output_file = f"{workspace_dir}/.opencode_output_iter{iteration}.log"
            status_file = f"{workspace_dir}/.opencode_status.json"

            # 后台执行命令（使用 script 模拟终端，实现实时输出）
            # script 命令创建伪终端，强制 OpenCode 以交互模式运行
            # -q: 静默模式（不打印开始/结束消息）
            # -f: 每次写入后立即刷新（确保实时输出）
            background_cmd = f"""cd {workspace_dir} && script -q -f -c "bash {runner_script}" {output_file} &"""
            self._log(f"执行后台命令: {background_cmd[:100]}...")
            await self.sandbox.commands.run(background_cmd)

            self._report_progress(
                "sandbox_exec",
                f"[START] OpenCode 任务已启动 (模式: {mode.value})",
                {"phase": "execution_started", "mode": mode.value},
            )
            self._log(f"OpenCode 任务已启动 (模式: {mode.value})")

            # 5. 文件轮询检测完成（避免 SSE 超时）
            max_wait_seconds = self.config.sandbox_timeout_seconds
            poll_interval = 5  # 每5秒检查一次
            elapsed = 0
            consecutive_connection_errors = 0  # 连续连接错误计数
            max_connection_errors = 5  # 最大允许连续错误次数
            last_reported_len = 0  # 记录上次已解析的日志长度（避免重复报告）

            while elapsed < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                # 检查是否完成
                try:
                    output = await self.sandbox.files.read_file(output_file)
                    consecutive_connection_errors = 0  # 成功读取，重置计数器

                    # 【新增】实时解析并报告进度
                    last_reported_len = self._parse_and_report_progress(
                        output, iteration=iteration, last_reported_len=last_reported_len
                    )

                    if "===OPENCODE_DONE===" in output or "Script done" in output:
                        self._log("OpenCode 任务完成")
                        break
                except Exception as e:
                    error_str = str(e).lower()
                    # 检测沙盒连接断开
                    if "connect" in error_str or "connection" in error_str:
                        consecutive_connection_errors += 1
                        if consecutive_connection_errors >= max_connection_errors:
                            self._log(
                                f"❌ 沙盒连接已断开，连续 {consecutive_connection_errors} 次连接失败"
                            )
                            self._report_progress(
                                "sandbox_exec",
                                "❌ 沙盒连接断开",
                                {
                                    "phase": "connection_lost",
                                    "consecutive_errors": consecutive_connection_errors,
                                    "elapsed": elapsed,
                                },
                            )
                            break

                if elapsed % 30 == 0:
                    self._log(f"等待 OpenCode 执行... ({elapsed}s/{max_wait_seconds}s)")
                    self._report_progress(
                        "sandbox_exec",
                        f"⏳ 等待中... ({elapsed}s/{max_wait_seconds}s)",
                        {
                            "phase": "waiting",
                            "elapsed": elapsed,
                            "max_wait": max_wait_seconds,
                        },
                    )

            # 6. 读取最终输出
            try:
                stdout = await self.sandbox.files.read_file(output_file)
            except Exception as e:
                stdout = f"无法读取输出文件: {e}"

            # 7. 收集执行结果
            execution_time = int((time.time() - start_time) * 1000)

            # 读取执行摘要（由 runner.sh 生成）
            summary = await self._read_execution_summary(workspace_dir)
            output_files = await self._list_output_files(workspace_dir)

            # 【新增】解析任务执行记录
            task_records = await self._collect_task_records(
                workspace_dir, stdout, output_files
            )

            # 确定状态
            if "===OPENCODE_DONE===" in stdout or "Script done" in stdout:
                status = ExecutionStatus.SUCCESS
            elif "error" in stdout.lower() or "failed" in stdout.lower():
                status = ExecutionStatus.FAILED
            else:
                status = ExecutionStatus.SUCCESS

            # 【新增】从任务记录中提取完成/失败的任务列表
            completed_tasks = [r.task_id for r in task_records if r.is_success()]
            failed_tasks = [r.task_id for r in task_records if not r.is_success()]

            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr="",
                error=None,
                sandbox_id=getattr(self.sandbox, "id", None),
                execution_time_ms=execution_time,
                output_files=output_files,
                summary=summary,
                task_records=task_records,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"执行失败: {str(e)}",
                sandbox_id=getattr(self.sandbox, "id", None) if self.sandbox else None,
                execution_time_ms=execution_time,
            )

    def _parse_and_report_progress(
        self, log_content: str, iteration: int = 0, last_reported_len: int = 0
    ) -> int:
        """
        解析 OpenCode 日志，识别关键事件并通过 progress_callback 报告

        Args:
            log_content: 当前日志内容
            iteration: 迭代编号
            last_reported_len: 上次已解析的日志长度（避免重复报告）

        Returns:
            新的日志长度（下次调用时传入）
        """
        if len(log_content) <= last_reported_len:
            return len(log_content)

        new_content = log_content[last_reported_len:]
        if not new_content:
            return len(log_content)

        lines = new_content.split("\n")

        # ========== 1. 增强的 MCP 工具调用解析 ==========
        # OpenCode 格式: → mcp server.tool_name 或 mcp server.tool_name
        mcp_call_pattern = r"(?:→\s*)?mcp\s+([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)"

        i = 0
        while i < len(lines):
            line = lines[i]

            # 检测 MCP 调用开始
            mcp_match = re.search(mcp_call_pattern, line, re.IGNORECASE)
            if mcp_match:
                server = mcp_match.group(1)
                tool_name = mcp_match.group(2)
                full_tool_name = f"{server}.{tool_name}"

                # 报告 MCP 调用开始
                self._report_progress(
                    "sandbox_exec",
                    f"🔧 开始调用 MCP: {full_tool_name}",
                    {
                        "phase": "mcp_call_start",
                        "tool_name": full_tool_name,
                        "server": server,
                        "iteration": iteration,
                    },
                )

                # 查找参数（可能在同一行或下一行）
                params_summary = ""
                json_pattern = r"\{[^{}]*\}"

                # 在同一行查找 JSON
                json_match = re.search(json_pattern, line[mcp_match.end() :])
                if not json_match:
                    # 在后面的行查找 JSON（最多检查 3 行）
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_line = lines[j].strip()
                        if next_line.startswith("{"):
                            try:
                                params = json.loads(next_line)
                                # 提取关键参数
                                params_summary = self._extract_key_params(params)
                                break
                            except:
                                pass

                # 报告参数摘要
                if params_summary:
                    self._report_progress(
                        "sandbox_exec",
                        f"   参数: {params_summary}",
                        {
                            "phase": "mcp_params",
                            "tool_name": full_tool_name,
                            "params_summary": params_summary,
                            "iteration": iteration,
                        },
                    )

            # 检测 MCP 响应（JSON 格式）
            elif (
                line.strip().startswith("{")
                and '"success"' in line
                or '"status"' in line
                or '"result"' in line
            ):
                try:
                    response = json.loads(line.strip())
                    is_success = (
                        response.get("success", True)
                        or response.get("status") == "success"
                    )

                    # 提取响应摘要
                    result_summary = self._extract_result_summary(response)

                    status_icon = "✅" if is_success else "❌"
                    self._report_progress(
                        "sandbox_exec",
                        f"   {status_icon} 响应: {result_summary}",
                        {
                            "phase": "mcp_response",
                            "success": is_success,
                            "result_summary": result_summary,
                            "iteration": iteration,
                        },
                    )
                except json.JSONDecodeError:
                    pass

            i += 1

        # ========== 2. 解析任务完成 ==========
        task_patterns = [
            r"Task\s+(\w+)\s+completed",
            r"Task\s+(\w+)\s+finished",
            r"Completed\s+task[:\s]+(\w+)",
            r"任务\s+(\S+)\s+完成",
        ]

        for pattern in task_patterns:
            matches = re.findall(pattern, new_content, re.IGNORECASE)
            for task_id in matches:
                self._report_progress(
                    "sandbox_exec",
                    f"✅ 任务完成: {task_id}",
                    {
                        "phase": "task_complete",
                        "task_id": task_id,
                        "iteration": iteration,
                    },
                )

        # ========== 3. 解析文件生成 ==========
        file_patterns = [
            r"Generated\s+file[:\s]+(\S+)",
            r"Created\s+file[:\s]+(\S+)",
            r"Saved\s+file[:\s]+(\S+)",
            r"Output\s+file[:\s]+(\S+)",
            r"生成文件[:\s]+(\S+)",
            r"Wrote\s+file[:\s]+(\S+)",
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, new_content, re.IGNORECASE)
            for filepath in matches:
                filename = filepath.split("/")[-1]
                self._report_progress(
                    "sandbox_exec",
                    f"📄 生成文件: {filename}",
                    {
                        "phase": "file_generated",
                        "file_path": filepath,
                        "iteration": iteration,
                    },
                )

        # ========== 4. 解析代码执行 ==========
        code_patterns = [
            r"Executing\s+code[:\s]+(.+)",
            r"Running\s+code[:\s]+(.+)",
            r"执行代码[:\s]+(.+)",
        ]

        for pattern in code_patterns:
            matches = re.findall(pattern, new_content, re.IGNORECASE)
            for code_snippet in matches:
                preview = (
                    code_snippet[:100] + "..."
                    if len(code_snippet) > 100
                    else code_snippet
                )
                self._report_progress(
                    "sandbox_exec",
                    f"💻 执行代码: {preview}",
                    {
                        "phase": "code_execution",
                        "code_preview": preview,
                        "iteration": iteration,
                    },
                )

        # ========== 5. 解析错误信息 ==========
        error_patterns = [
            r"ERROR[:\s]+(.+)",
            r"Error[:\s]+(.+)",
            r"Exception[:\s]+(.+)",
            r"Failed[:\s]+(.+)",
            r"timeout[:\s]+(.+)",
        ]

        for pattern in error_patterns:
            matches = re.findall(pattern, new_content, re.IGNORECASE)
            for error_msg in matches:
                preview = error_msg[:150] + "..." if len(error_msg) > 150 else error_msg
                self._report_progress(
                    "error",
                    f"❌ 错误: {preview}",
                    {
                        "phase": "error",
                        "error_message": error_msg,
                        "iteration": iteration,
                    },
                )

        # ========== 6. 解析进度信息 ==========
        progress_patterns = [
            (r"Read\s+(\S+)", "📖 读取"),
            (r"Analyzing\s+(.+)", "📊 分析"),
            (r"Processing\s+(.+)", "⚙️ 处理"),
            (r"Calculating\s+(.+)", "🔢 计算"),
            (r"Loading\s+(.+)", "⏳ 加载"),
        ]

        for pattern, icon in progress_patterns:
            matches = re.findall(pattern, new_content, re.IGNORECASE)
            for item in matches:
                preview = item[:80] + "..." if len(item) > 80 else item
                self._report_progress(
                    "sandbox_exec",
                    f"{icon} {preview}",
                    {
                        "phase": "progress",
                        "item": item,
                        "iteration": iteration,
                    },
                )

        return len(log_content)

    def _extract_key_params(self, params: dict, max_len: int = 80) -> str:
        """提取参数的关键信息"""
        if not params or not isinstance(params, dict):
            return ""

        # 优先显示的关键参数
        key_params = []
        priority_keys = [
            "file",
            "path",
            "input",
            "output",
            "sequence",
            "query",
            "data",
            "name",
            "type",
        ]

        for key in priority_keys:
            if key in params:
                value = params[key]
                if isinstance(value, str):
                    # 截断长字符串
                    if len(value) > 30:
                        value = value[:30] + "..."
                    key_params.append(f"{key}={value}")
                elif isinstance(value, (int, float, bool)):
                    key_params.append(f"{key}={value}")

        # 如果没有优先参数，显示前 3 个
        if not key_params:
            for key, value in list(params.items())[:3]:
                if isinstance(value, str) and len(value) > 30:
                    value = value[:30] + "..."
                elif isinstance(value, (int, float, bool)):
                    pass
                else:
                    value = str(type(value).__name__)
                key_params.append(f"{key}={value}")

        result = ", ".join(key_params[:4])  # 最多显示 4 个参数
        return result[:max_len] + "..." if len(result) > max_len else result

    def _extract_result_summary(self, response: dict, max_len: int = 100) -> str:
        """提取响应的关键信息"""
        if not response or not isinstance(response, dict):
            return "无响应内容"

        # 检查是否有错误
        if "error" in response and response["error"]:
            error = response["error"]
            if isinstance(error, str):
                return f"错误: {error[:max_len]}"
            elif isinstance(error, dict):
                return f"错误: {error.get('message', str(error)[:max_len])}"

        # 检查是否有结果
        if "result" in response:
            result = response["result"]
            if isinstance(result, str):
                return result[:max_len] + "..." if len(result) > max_len else result
            elif isinstance(result, dict):
                # 提取关键信息
                keys = list(result.keys())[:5]
                return f"包含 {len(result)} 个字段: {', '.join(keys)}"
            elif isinstance(result, list):
                return f"包含 {len(result)} 条数据"
            else:
                return str(type(result).__name__)

        # 检查状态
        if "status" in response:
            return f"状态: {response['status']}"

        # 检查 success
        if "success" in response:
            return "成功" if response["success"] else "失败"

        # 默认返回键列表
        keys = list(response.keys())[:5]
        return f"包含字段: {', '.join(keys)}"

    async def _generate_runner_script(
        self,
        workspace_dir: str,
        tasks_md_path: str,
        mode: OpenCodeMode,
        iteration: int = 0,
    ) -> str:
        """
        生成 OpenCode 执行脚本

        使用脚本文件而非命令行传参，避免参数长度限制和转义问题。

        Args:
            workspace_dir: 工作目录
            tasks_md_path: tasks.md 文件路径
            mode: 执行模式
            iteration: 迭代编号（用于生成独立的完成标记）
        """
        output_dir = f"{workspace_dir}/output"

        # 构建 prompt（使用独立的提示词模块）
        mode_str = "plan" if mode == OpenCodeMode.PLAN else "execute"
        prompt = get_opencode_prompt(
            mode=mode_str,
            tasks_md_path=tasks_md_path,
            output_dir=output_dir,
            iteration=iteration,
        )

        script_content = f'''#!/bin/bash
# OpenCode 任务执行脚本
# 生成时间: {datetime.now().isoformat()}

set -e

# ========== 激活 Python 虚拟环境 ==========
echo "=== 激活 Python 虚拟环境 ==="

# 方式 1: 如果镜像中有 code-interpreter-env.sh（OpenSandbox 标准方式）
if [ -f "/opt/opensandbox/code-interpreter-env.sh" ]; then
    echo "激活 OpenSandbox 虚拟环境..."
    source /opt/opensandbox/code-interpreter-env.sh python 3.13 2>/dev/null || true
fi

# 方式 2: 直接激活 venv（备用方案）
if [ -d "/opt/opensandbox/venv" ]; then
    echo "激活 /opt/opensandbox/venv..."
    export PATH="/opt/opensandbox/venv/bin:$PATH"
    export PYTHONPATH="/opt/opensandbox/venv/lib/python3.13/site-packages:$PYTHONPATH"
    export VIRTUAL_ENV="/opt/opensandbox/venv"
elif [ -d "/root/.venv" ]; then
    echo "激活 /root/.venv..."
    export PATH="/root/.venv/bin:$PATH"
    export PYTHONPATH="/root/.venv/lib/python3.11/site-packages:$PYTHONPATH"
    export VIRTUAL_ENV="/root/.venv"
fi

# 验证 Python 环境
echo "验证 Python 环境:"
python3 --version 2>/dev/null || echo "Warning: Python not found"
python3 -c "import sys; print(f'Python executable: {{sys.executable}}')" 2>/dev/null || true
python3 -c "import pandas; print(f'pandas: {{pandas.__version__}}')" 2>/dev/null || echo "Warning: pandas not available"
python3 -c "import pyreadr; print('pyreadr: OK')" 2>/dev/null || echo "Warning: pyreadr not available"
echo "=== Python 环境激活完成 ==="
echo ""

cd {workspace_dir}

# 创建 OpenCode XDG 目录（避免访问 /root/）
mkdir -p /tmp/opencode/data
mkdir -p /tmp/opencode/config
mkdir -p /tmp/opencode/state
mkdir -p /tmp/opencode/bin

# 创建输出目录和初始化任务日志
mkdir -p {output_dir}
echo '[]' > {output_dir}/task_execution_log.json

echo "=== 开始执行 OpenCode 任务 ==="
echo "任务文件: {tasks_md_path}"
echo "模式: {mode.value}"
echo "工作目录: $(pwd)"
echo "XDG_DATA_HOME: $XDG_DATA_HOME"
echo "任务日志文件: {output_dir}/task_execution_log.json"
echo ""

# 【新增】 打印 OpenCode 配置
echo ""
echo "=== OpenCode 配置信息 ==="
if [ -f "/tmp/opencode/config/opencode/opencode.json" ]; then
    echo "全局配置文件: /tmp/opencode/config/opencode/opencode.json"
    cat /tmp/opencode/config/opencode/opencode.json
elif [ -f "{workspace_dir}/opencode.json" ]; then
    echo "工作目录配置: {workspace_dir}/opencode.json"
    cat {workspace_dir}/opencode.json
elif [ -f "{workspace_dir}/.opencode.json" ]; then
    echo "工作目录配置(隐藏): {workspace_dir}/.opencode.json"
    cat {workspace_dir}/.opencode.json
else
    echo "警告: 未找到 OpenCode 配置文件"
fi
echo "=== 配置信息结束 ==="
echo ""

# 执行 OpenCode
# 使用 heredoc 避免 shell 转义问题
opencode run << 'OPENCODE_PROMPT_EOF'
{prompt}
OPENCODE_PROMPT_EOF

OPENCODE_EXIT_CODE=$?

echo ""
echo "=== OpenCode 执行完成 (exit code: $OPENCODE_EXIT_CODE) ==="

# 生成执行摘要
SUMMARY_FILE="{output_dir}/execution_summary.json"
mkdir -p {output_dir}

# 读取任务日志统计
TASK_LOG="{output_dir}/task_execution_log.json"
if [ -f "$TASK_LOG" ]; then
    TOTAL_TASKS=$(python3 -c "import json; print(len(json.load(open('$TASK_LOG'))))" 2>/dev/null || echo "0")
    SUCCESS_TASKS=$(python3 -c "import json; print(sum(1 for r in json.load(open('$TASK_LOG')) if r.get('status')=='success'))" 2>/dev/null || echo "0")
    FAILED_TASKS=$(python3 -c "import json; print(sum(1 for r in json.load(open('$TASK_LOG')) if r.get('status')=='failed'))" 2>/dev/null || echo "0")
else
    TOTAL_TASKS=0
    SUCCESS_TASKS=0
    FAILED_TASKS=0
fi

cat > "$SUMMARY_FILE" << SUMMARY_EOF
{{
  "status": "success",
  "exit_code": $OPENCODE_EXIT_CODE,
  "completed_at": "$(date -Iseconds)",
  "workspace": "{workspace_dir}",
  "output_dir": "{output_dir}",
  "task_statistics": {{
    "total": $TOTAL_TASKS,
    "success": $SUCCESS_TASKS,
    "failed": $FAILED_TASKS
  }}
}}
SUMMARY_EOF

echo "执行摘要已保存到: $SUMMARY_FILE"

# 显示任务执行日志摘要
echo ""
echo "=== 任务执行日志摘要 ==="
if [ -f "$TASK_LOG" ]; then
    TASK_LOG_PATH="$TASK_LOG" python3 << 'PYTHON_SUMMARY_EOF'
import json
import os
import traceback

task_log = os.environ.get('TASK_LOG_PATH', '')
records = []  # 先初始化

try:
    if task_log and os.path.exists(task_log):
        with open(task_log) as f:
            content = f.read().strip()
            if content:
                records = json.loads(content)
            else:
                print('任务日志文件为空')
    else:
        print(f'任务日志路径无效: {{task_log}}')
except Exception as e:
    print(f'解析任务日志失败: {{e}}')
    traceback.print_exc()

if records:
    print(f'总任务数: {{len(records)}}')
    for r in records:
        status_icon = '[SUCCESS]' if r.get('status') == 'success' else '[ERROR]'
        print(f"  {{status_icon}} {{r.get('task_id', 'N/A')}}: {{r.get('task_name', 'N/A')}}")
        if r.get('mcp_tool_name'):
            print(f"      MCP: {{r.get('mcp_tool_name')}}")
        if r.get('output_files'):
            print(f"      输出: {{', '.join(r.get('output_files', []))}}")
PYTHON_SUMMARY_EOF
    if [ $? -ne 0 ]; then
        echo "无法解析任务日志"
    fi
else
    echo "未找到任务执行日志"
fi

# 列出输出文件
echo ""
echo "=== 输出文件列表 ==="
ls -la {output_dir}/ 2>/dev/null || echo "无输出文件"

# 标记完成
echo "===OPENCODE_DONE===" >> {workspace_dir}/.opencode_output_iter{iteration}.log

exit $OPENCODE_EXIT_CODE
'''

        # 写入脚本文件（强制覆盖旧版本）
        script_path = f"{workspace_dir}/runner.sh"
        # 先删除可能存在的旧脚本
        await self.sandbox.commands.run(f"rm -f {script_path} 2>/dev/null || true")
        await self.sandbox.files.write_file(script_path, script_content)

        # 添加执行权限
        await self.sandbox.commands.run(f"chmod +x {script_path}")

        self._log(f"执行脚本已生成: {script_path}")
        return script_path

    def _build_opencode_prompt(self, tasks_md_path: str, mode: OpenCodeMode) -> str:
        """构建 OpenCode 执行提示"""
        if mode == OpenCodeMode.PLAN:
            return f"""请阅读 {tasks_md_path} 中的任务列表，分析任务并生成执行计划。
注意：这是 plan 模式，只进行分析，不执行实际操作。"""
        else:
            return f"""请阅读 {tasks_md_path} 中的任务列表，按顺序执行所有任务。

执行过程中请注意：
1. 对于 MCP_TOOL 类型任务，使用 call_tool() 调用对应工具
2. 对于 CODE_GENERATION 类型任务，生成并执行代码
3. 完成每个任务后，验证输出文件是否生成
4. 最后生成 execution_summary.json 到 output/ 目录

执行完成后，请汇报任务完成情况。"""

    async def upload_context(
        self,
        context: Dict[str, Any],
        target_path: str = "/workspace/.agent/context.json",
    ) -> None:
        """
        上传上下文文件到沙盒

        Args:
            context: 上下文数据
            target_path: 目标路径
        """
        if not self.sandbox:
            raise RuntimeError("沙盒未创建")

        content = json.dumps(context, indent=2, ensure_ascii=False)
        await self.sandbox.files.write_file(target_path, content)
        self._log(f"上传上下文文件: {target_path}")

    async def upload_file(
        self,
        local_path: str,
        sandbox_path: str,
    ) -> None:
        """
        上传本地文件到沙盒

        Args:
            local_path: 本地文件路径
            sandbox_path: 沙盒内目标路径
        """
        if not self.sandbox:
            raise RuntimeError("沙盒未创建")

        with open(local_path, "rb") as f:
            content = f.read()

        await self.sandbox.files.write_file(sandbox_path, content)
        self._log(f"上传文件: {local_path} -> {sandbox_path}")

    async def download_file(
        self,
        sandbox_path: str,
        local_path: str,
    ) -> None:
        """
        从沙盒下载文件

        Args:
            sandbox_path: 沙盒内文件路径
            local_path: 本地目标路径
        """
        if not self.sandbox:
            raise RuntimeError("沙盒未创建")

        content = await self.sandbox.files.read_file(sandbox_path)

        # 确保目录存在
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as f:
            if isinstance(content, str):
                f.write(content.encode("utf-8"))
            else:
                f.write(content)

        self._log(f"下载文件: {sandbox_path} -> {local_path}")

    async def download_results(self, output_dir: str) -> Dict[str, Any]:
        """
        下载执行结果

        Args:
            output_dir: 沙盒内输出目录

        Returns:
            执行结果摘要
        """
        if not self.sandbox:
            raise RuntimeError("沙盒未创建")

        # 尝试读取执行摘要
        summary_path = f"{output_dir}/execution_summary.json"
        try:
            summary_content = await self.sandbox.files.read_file(summary_path)
            return json.loads(summary_content)
        except Exception:
            return {"status": "unknown", "message": "未找到执行摘要"}

    async def _read_execution_summary(
        self, workspace_dir: str
    ) -> Optional[Dict[str, Any]]:
        """读取执行摘要"""
        if not self.sandbox:
            return None

        summary_path = f"{workspace_dir}/output/execution_summary.json"
        try:
            content = await self.sandbox.files.read_file(summary_path)
            return json.loads(content)
        except Exception:
            return None

    async def _list_output_files(self, workspace_dir: str) -> List[str]:
        """列出输出文件"""
        if not self.sandbox:
            return []

        output_dir = f"{workspace_dir}/output"
        try:
            result = await self.sandbox.commands.run(
                f"find {output_dir} -type f 2>/dev/null || echo ''"
            )
            stdout = self._get_stdout(result)
            if stdout.strip():
                return [f.strip() for f in stdout.strip().split("\n") if f.strip()]
        except Exception:
            pass
        return []

    # =========================================================================
    # 任务执行记录收集方法
    # =========================================================================

    async def _collect_task_records(
        self, workspace_dir: str, stdout: str, output_files: List[str]
    ) -> List[TaskExecutionRecord]:
        """
        收集任务执行记录

        优先从 OpenCode 生成的 task_execution_log.json 读取，如果不存在则从输出解析。

        Args:
            workspace_dir: 工作目录
            stdout: OpenCode 执行输出
            output_files: 输出文件列表

        Returns:
            任务执行记录列表
        """
        records: List[TaskExecutionRecord] = []

        # 1. 【优先】从 OpenCode 生成的任务执行日志读取
        task_log_path = f"{workspace_dir}/output/task_execution_log.json"
        try:
            if self.sandbox:
                content = await self.sandbox.files.read_file(task_log_path)
                if isinstance(content, bytes):
                    content = content.decode("utf-8")

                log_data = json.loads(content)
                if isinstance(log_data, list):
                    for item in log_data:
                        # 转换 task_type 字符串为枚举
                        task_type_str = item.get("task_type", "MCP_TOOL")
                        try:
                            task_type = TaskType(task_type_str)
                        except ValueError:
                            task_type = TaskType.MCP_TOOL

                        # 转换 status 字符串为枚举
                        status_str = item.get("status", "pending")
                        try:
                            status = ExecutionStatus(status_str)
                        except ValueError:
                            status = ExecutionStatus.PENDING

                        record = TaskExecutionRecord(
                            task_id=item.get("task_id", "unknown"),
                            task_name=item.get("task_name", "未命名任务"),
                            task_type=task_type,
                            status=status,
                            start_time=item.get("start_time"),
                            end_time=item.get("end_time"),
                            execution_time_ms=item.get("execution_time_ms", 0),
                            parameters=item.get("parameters", {}),
                            input_files=item.get("input_files", []),
                            mcp_tool_name=item.get("mcp_tool_name"),
                            mcp_server=item.get("mcp_server"),
                            output_files=item.get("output_files", []),
                            output_data=item.get("output_data"),
                            mcp_response=item.get("mcp_response"),
                            action_description=item.get("action_description", ""),
                            raw_output=item.get("raw_output", ""),
                            error_message=item.get("error_message"),
                            retry_count=item.get("retry_count", 0),
                        )
                        records.append(record)

                    self._log(
                        f"从 task_execution_log.json 读取到 {len(records)} 条任务记录"
                    )
                    return records
        except Exception as e:
            self._log(f"读取 task_execution_log.json 失败: {e}，尝试从输出解析")

        # 2. 【备选】从 OpenCode 输出解析（兼容旧模式）
        self._log("使用备选方案：从输出解析任务记录")

        # 2.1 读取原始 tasks.md（解析任务列表）
        tasks_md_path = f"{workspace_dir}/tasks.md"
        original_tasks = await self._parse_tasks_md(tasks_md_path)

        # 2.2 解析 OpenCode 输出中的 MCP 调用记录
        mcp_calls = self._parse_mcp_calls_from_output(stdout)

        # 2.3 尝试读取 OpenCode 生成的执行记录（如果存在）
        agent_records = await self._read_agent_records(workspace_dir)

        # 2.4 合并信息生成任务记录
        for i, task_info in enumerate(original_tasks):
            task_id = f"task_{i + 1}"
            task_name = task_info.get("name", f"任务 {i + 1}")
            task_type = task_info.get("type", TaskType.MCP_TOOL)
            task_params = task_info.get("parameters", {})

            # 查找对应的 MCP 调用
            mcp_call = None
            for call in mcp_calls:
                if call.get("task_index") == i or call.get("task_name") == task_name:
                    mcp_call = call
                    break

            # 查找对应的输出文件
            task_output_files = [
                f
                for f in output_files
                if task_name.lower() in f.lower() or f"task_{i + 1}" in f.lower()
            ]

            # 查找对应的 agent 记录
            agent_record = None
            for record in agent_records:
                if record.get("task_id") == task_id or record.get("task_index") == i:
                    agent_record = record
                    break

            # 构建 TaskExecutionRecord
            record = TaskExecutionRecord(
                task_id=task_id,
                task_name=task_name,
                task_type=task_type,
                status=ExecutionStatus.SUCCESS
                if (mcp_call or agent_record or task_output_files)
                else ExecutionStatus.PENDING,
                parameters=task_params,
                input_files=task_info.get("input_files", []),
                mcp_tool_name=mcp_call.get("tool_name") if mcp_call else None,
                mcp_server=mcp_call.get("server") if mcp_call else None,
                output_files=task_output_files,
                output_data=mcp_call.get("response")
                if mcp_call
                else agent_record.get("output_data")
                if agent_record
                else None,
                mcp_response=mcp_call.get("raw_response") if mcp_call else None,
                action_description=self._generate_action_description(
                    task_name, task_type, mcp_call, task_output_files
                ),
                raw_output=mcp_call.get("raw_output", "") if mcp_call else "",
            )

            records.append(record)

        # 3. 如果没有解析到任务，但输出中有 MCP 调用，则创建基于 MCP 调用的记录
        if not records and mcp_calls:
            for i, call in enumerate(mcp_calls):
                record = TaskExecutionRecord(
                    task_id=f"mcp_call_{i + 1}",
                    task_name=f"MCP 调用: {call.get('tool_name', 'Unknown')}",
                    task_type=TaskType.MCP_TOOL,
                    status=ExecutionStatus.SUCCESS
                    if call.get("success")
                    else ExecutionStatus.FAILED,
                    parameters=call.get("parameters", {}),
                    mcp_tool_name=call.get("tool_name"),
                    mcp_server=call.get("server"),
                    output_data=call.get("response"),
                    mcp_response=call.get("raw_response"),
                    action_description=call.get("description", ""),
                    raw_output=call.get("raw_output", ""),
                )
                records.append(record)

        # 4. 如果仍然没有记录，创建一个汇总记录
        if not records:
            record = TaskExecutionRecord(
                task_id="summary",
                task_name="整体执行",
                task_type=TaskType.ANALYSIS,
                status=ExecutionStatus.SUCCESS
                if "===OPENCODE_DONE===" in stdout
                else ExecutionStatus.FAILED,
                output_files=output_files,
                action_description="执行所有任务",
                raw_output=stdout[:2000] if len(stdout) > 2000 else stdout,
            )
            records.append(record)

        return records

    async def _parse_tasks_md(self, tasks_md_path: str) -> List[Dict[str, Any]]:
        """
        解析 tasks.md 文件，提取任务列表

        Args:
            tasks_md_path: tasks.md 文件路径

        Returns:
            任务信息列表
        """
        tasks = []

        if not self.sandbox:
            return tasks

        try:
            content = await self.sandbox.files.read_file(tasks_md_path)
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            # 简单解析：查找 "## Task" 或 "- [ ]" 或 "1." 等任务标记
            lines = content.split("\n")
            current_task = None

            for line in lines:
                line = line.strip()

                # 匹配 "## Task N:" 或 "### Task N:" 格式
                if line.startswith("## Task") or line.startswith("### Task"):
                    if current_task:
                        tasks.append(current_task)
                    current_task = {
                        "name": line.replace("#", "").strip(),
                        "type": TaskType.MCP_TOOL,
                        "parameters": {},
                        "input_files": [],
                    }

                # 匹配 "- [ ] Task" 格式
                elif line.startswith("- [ ]") or line.startswith("- [x]"):
                    if current_task:
                        tasks.append(current_task)
                    current_task = {
                        "name": line.replace("- [ ]", "").replace("- [x]", "").strip(),
                        "type": TaskType.MCP_TOOL,
                        "parameters": {},
                        "input_files": [],
                    }

                # 匹配 "1. Task" 或 "N. Task" 格式
                elif line and line[0].isdigit() and ". " in line[:5]:
                    if current_task:
                        tasks.append(current_task)
                    current_task = {
                        "name": line.split(". ", 1)[1] if ". " in line else line,
                        "type": TaskType.MCP_TOOL,
                        "parameters": {},
                        "input_files": [],
                    }

                # 收集参数信息（如果当前在任务块中）
                elif current_task:
                    # 检测参数描述
                    if "参数" in line or "parameter" in line.lower():
                        current_task["parameters"]["_desc"] = line
                    # 检测文件路径
                    if "/" in line and (
                        ".csv" in line or ".fasta" in line or ".txt" in line
                    ):
                        current_task["input_files"].append(line.strip())
                    # 检测任务类型
                    if "MCP" in line or "工具" in line:
                        current_task["type"] = TaskType.MCP_TOOL
                    elif "代码" in line or "code" in line.lower():
                        current_task["type"] = TaskType.CODE_GENERATION
                    elif "分析" in line or "analysis" in line.lower():
                        current_task["type"] = TaskType.ANALYSIS

            # 添加最后一个任务
            if current_task:
                tasks.append(current_task)

        except Exception as e:
            self._log(f"解析 tasks.md 失败: {e}")

        return tasks

    def _parse_mcp_calls_from_output(self, stdout: str) -> List[Dict[str, Any]]:
        """
        从 OpenCode 输出中解析 MCP 调用记录

        OpenCode 输出格式示例:
        → mcp nettcr.predict_tcr_binding
        {"success": true, "result": {...}}

        Args:
            stdout: OpenCode 执行输出

        Returns:
            MCP 调用记录列表
        """
        calls = []

        # 正则匹配 MCP 调用
        # 格式 1: → mcp server.tool_name
        # 格式 2: mcp server.tool_name
        import re

        # 匹配 MCP 工具调用
        mcp_pattern = r"(?:→\s*)?mcp\s+([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)"

        # 匹配 JSON 响应（在 MCP 调用后面）
        json_pattern = r"\{[^{}]*\}"

        lines = stdout.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # 查找 MCP 调用
            match = re.search(mcp_pattern, line, re.IGNORECASE)
            if match:
                server = match.group(1)
                tool_name = match.group(2)

                # 查找参数和响应（可能在同一行或后面的行）
                parameters = {}
                response = None
                raw_response = None

                # 在同一行查找 JSON
                json_match = re.search(json_pattern, line[match.end() :])
                if json_match:
                    try:
                        raw_response = json_match.group(0)
                        response = json.loads(raw_response)
                    except json.JSONDecodeError:
                        pass

                # 在后面的行查找 JSON 响应
                if not response:
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if next_line.startswith("{"):
                            try:
                                raw_response = next_line
                                response = json.loads(next_line)
                                break
                            except json.JSONDecodeError:
                                # 尝试多行 JSON
                                json_text = "\n".join(
                                    lines[j : min(j + 10, len(lines))]
                                )
                                try:
                                    response = json.loads(json_text.split("\n\n")[0])
                                    raw_response = json_text[:500]
                                    break
                                except json.JSONDecodeError:
                                    pass

                # 提取参数（如果有）
                if response and isinstance(response, dict):
                    parameters = {
                        k: v
                        for k, v in response.items()
                        if k not in ["success", "status", "result", "data", "error"]
                    }

                call_record = {
                    "server": server,
                    "tool_name": f"{server}.{tool_name}",
                    "parameters": parameters,
                    "response": response.get("result")
                    if response and "result" in response
                    else response,
                    "raw_response": raw_response,
                    "success": response.get("success", True) if response else True,
                    "raw_output": line,
                    "description": f"调用 {server} 服务的 {tool_name} 工具",
                }
                calls.append(call_record)

            i += 1

        return calls

    async def _read_agent_records(self, workspace_dir: str) -> List[Dict[str, Any]]:
        """
        读取 OpenCode 生成的执行记录（如果存在）

        OpenCode 可能会在 .agent/ 目录下生成执行记录

        Args:
            workspace_dir: 工作目录

        Returns:
            执行记录列表
        """
        records = []

        if not self.sandbox:
            return records

        # 尝试读取 .agent/execution_log.json
        log_path = f"{workspace_dir}/.agent/execution_log.json"
        try:
            content = await self.sandbox.files.read_file(log_path)
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
        except Exception:
            pass

        # 尝试读取 output/task_results.json
        results_path = f"{workspace_dir}/output/task_results.json"
        try:
            content = await self.sandbox.files.read_file(results_path)
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                records.extend(data)
        except Exception:
            pass

        return records

    def _generate_action_description(
        self,
        task_name: str,
        task_type: TaskType,
        mcp_call: Optional[Dict[str, Any]],
        output_files: List[str],
    ) -> str:
        """
        生成任务执行描述

        Args:
            task_name: 任务名称
            task_type: 任务类型
            mcp_call: MCP 调用信息（如果有）
            output_files: 输出文件列表

        Returns:
            任务执行描述
        """
        desc_parts = [f"任务: {task_name}"]

        if task_type == TaskType.MCP_TOOL and mcp_call:
            tool = mcp_call.get("tool_name", "未知工具")
            params = mcp_call.get("parameters", {})
            desc_parts.append(f"调用了 MCP 工具: {tool}")
            if params:
                param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
                desc_parts.append(f"参数: {param_str}")

        elif task_type == TaskType.CODE_GENERATION:
            desc_parts.append("生成了代码并执行")

        elif task_type == TaskType.ANALYSIS:
            desc_parts.append("执行了数据分析")

        elif task_type == TaskType.REPORT:
            desc_parts.append("生成了报告文件")

        if output_files:
            files_str = ", ".join(Path(f).name for f in output_files[:3])
            if len(output_files) > 3:
                files_str += f" 等 {len(output_files)} 个文件"
            desc_parts.append(f"输出: {files_str}")

        return "\n".join(desc_parts)

    async def cleanup(self) -> None:
        """清理沙盒"""
        if self.sandbox:
            try:
                await self.sandbox.kill()
                await self.sandbox.close()
                self._log("沙盒已清理")
            except Exception as e:
                self._log(f"清理沙盒时出错: {e}")
            finally:
                self.sandbox = None

    def _log(self, message: str) -> None:
        """打印日志并发送进度更新"""
        if self.config.show_progress:
            print(f"[OpenCode] {message}")

        # 发送进度更新（过滤敏感信息）
        self._send_user_friendly_progress(message)

    def _send_user_friendly_progress(self, message: str) -> None:
        """
        将日志消息转换为用户友好的进度更新并发送

        过滤规则：
        1. 跳过包含 JSON 配置的消息
        2. 跳过过长的消息（>200字符）
        3. 将技术性消息转换为用户友好消息
        """
        if not self.progress_callback:
            return

        # 跳过包含敏感配置的消息
        skip_keywords = [
            '"$schema"',
            '"provider"',
            '"mcp"',
            '"type":',
            '"url":',
            "config.json",
            "opencode.json",
            "OpenCode 配置:",
            "配置文件验证",
            "已清理旧配置文件",
        ]

        if any(keyword in message for keyword in skip_keywords):
            # 发送简化的状态消息
            if "config" in message.lower() or "配置" in message:
                self._report_progress(
                    event_type="sandbox_exec",
                    message="⚙️ 配置 OpenCode 环境...",
                    details={"phase": "opencode_config"},
                )
            return

        # 跳过过长的消息
        if len(message) > 200:
            return

        # 消息映射表：将技术性消息转换为用户友好消息
        message_mappings = {
            "安装": "📥 安装 OpenCode...",
            "OpenCode 安装成功": "[SUCCESS] OpenCode 安装完成",
            "设置 GLM 代理": "[TOOL] 设置 GLM 代理...",
            "GLM 代理启动成功": "[SUCCESS] GLM 代理已启动",
            "创建沙盒": "🐳 创建沙盒环境...",
            "沙盒创建成功": "[SUCCESS] 沙盒环境已就绪",
            "配置 OpenCode": "⚙️ 配置 OpenCode...",
        }

        # 查找匹配的消息
        friendly_message = None
        for key, value in message_mappings.items():
            if key in message:
                friendly_message = value
                break

        # 如果没有匹配，发送简化版消息
        if not friendly_message:
            # 清理消息前缀
            clean_message = message.strip()
            friendly_message = clean_message

        # 发送进度更新
        self._report_progress(
            event_type="sandbox_exec",
            message=friendly_message,
            details={"raw": message, "source": "opencode"},
        )

    def _get_stdout(self, result: Any) -> str:
        """获取标准输出"""
        logs = getattr(result, "logs", None)
        if not logs:
            return ""
        stdout = getattr(logs, "stdout", None)
        if not stdout:
            return ""
        if isinstance(stdout, str):
            return stdout
        return "\n".join(getattr(entry, "text", str(entry)) for entry in stdout)

    def _get_stderr(self, result: Any) -> str:
        """获取标准错误"""
        logs = getattr(result, "logs", None)
        if not logs:
            return ""
        stderr = getattr(logs, "stderr", None)
        if not stderr:
            return ""
        if isinstance(stderr, str):
            return stderr
        return "\n".join(getattr(entry, "text", str(entry)) for entry in stderr)


# ============================================================================
# 同步包装器
# ============================================================================


class OpenCodeExecutorSync:
    """OpenCode 执行器的同步包装器"""

    def __init__(self, config: Optional[OpenCodeConfig] = None):
        self._executor = OpenCodeExecutor(config)
        self._loop = None

    def create_sandbox(self, image: Optional[str] = None) -> Any:
        """创建沙盒（同步）"""
        return self._run_async(self._executor.create_sandbox(image))

    def execute_tasks(
        self,
        tasks_md_path: str,
        workspace_dir: str = "/workspace",
        mode: Optional[OpenCodeMode] = None,
    ) -> ExecutionResult:
        """执行任务（同步）"""
        return self._run_async(
            self._executor.execute_tasks(tasks_md_path, workspace_dir, mode)
        )

    def upload_context(self, context: Dict[str, Any], target_path: str = None) -> None:
        """上传上下文（同步）"""
        return self._run_async(self._executor.upload_context(context, target_path))

    def cleanup(self) -> None:
        """清理沙盒（同步）"""
        return self._run_async(self._executor.cleanup())

    def _run_async(self, coro):
        """运行异步协程"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
        except RuntimeError:
            pass
        return asyncio.run(coro)


__all__ = [
    "OpenCodeExecutor",
    "OpenCodeExecutorSync",
]
