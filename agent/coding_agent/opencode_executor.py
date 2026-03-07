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
    ):
        """
        初始化 OpenCode 执行器
        
        Args:
            config: OpenCode 配置对象（如果提供，其他参数将被忽略）
            sandbox_domain: OpenSandbox 服务域名
            model_provider: 模型提供商（glm-4.7, claude-sonnet-4, gpt-4o 等）
            api_key: API 密钥
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
            self._log(f"✓ 使用环境变量 OPENSANDBOX_IMAGE 中的镜像: {image}")
        elif image:
            self._log(f"使用传入的镜像参数: {image}")
        else:
            image = self.config.sandbox_image
            # 检查配置中的镜像是否与环境变量一致
            if env_image and image != env_image:
                self._log(f"⚠ 警告: 配置中的镜像 ({image}) 与环境变量 OPENSANDBOX_IMAGE ({env_image}) 不一致")
                self._log(f"⚠ 将使用环境变量中的镜像: {env_image}")
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
        check_result = await self.sandbox.commands.run("which opencode || echo 'not found'")
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
                'os.getenv("ZHIPU_API_KEY")',
                f'"{self.config.api_key}"'
            )
            await self.sandbox.files.write_file(
                "/root/glm_proxy.py",
                proxy_content
            )
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
                "zhipuai": {
                    "api": "https://open.bigmodel.cn/api/coding/paas/v4"
                }
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
        check_result = await self.sandbox.commands.run(f"cat {global_config_path} | head -5")
        self._log(f"配置文件验证: {self._get_stdout(check_result)[:200]}")
        
        self._log(f"OpenCode 配置已写入: {global_config_path}")
    
    def _build_mcp_servers_config(self) -> Dict[str, Any]:
        """
        构建 MCP 服务器配置（使用正确的 OpenCode 格式）
        
        OpenCode 官方格式（参考 anomalyco/opencode packages/opencode/src/config/config.ts）:
        {
            "server_name": {
                "type": "remote",        # 必需：远程服务器使用 "remote"，本地使用 "local"
                "url": "http://...",     # remote 模式必需
                "enabled": true,         # 可选：是否启用
                "timeout": 5000,         # 可选：超时时间（毫秒）
                "headers": {},           # 可选：HTTP 头
                "command": ["..."],      # local 模式必需
                "environment": {}        # local 模式可选
            }
        }
        
        注意：
        - OpenCode 使用 "mcp" 作为顶级配置键（不是 "mcpServers"）
        - OpenCode 使用 "remote" 而不是 "sse" 作为远程服务器的类型
        - OpenCode 使用 "local" 而不是 "stdio" 作为本地服务器的类型
        - 需要 "enabled: true" 才能启用 MCP 服务器
        
        支持读取 Bio-Agent 的 mcp_servers.json 格式并自动转换。
        """
        # 从配置文件读取（如果存在）
        if self.config.mcp_config_path and Path(self.config.mcp_config_path).exists():
            with open(self.config.mcp_config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
                return self._convert_mcp_config_to_opencode_format(raw_config)
        
        # 使用默认 MCP 配置
        return DEFAULT_MCP_CONFIG.get("servers", {})
    
    def _convert_mcp_config_to_opencode_format(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Bio-Agent 的 MCP 配置转换为 OpenCode 格式
        
        Bio-Agent 格式（mcp_servers.json）:
        {
            "nettcr": {
                "transport": "sse",
                "url": "http://...",
                "timeout": 36000,
                "sse_read_timeout": 36000
            }
        }
        
        OpenCode 正确格式（参考：anomalyco/opencode config.ts）:
        {
            "nettcr": {
                "type": "remote",        // 必需：远程服务器使用 "remote"，本地使用 "local"
                "url": "http://...",     // 必需：服务器 URL
                "enabled": true,         // 可选：是否启用
                "timeout": 30000         // 可选：超时时间（毫秒）
            }
        }
        
        注意：
        - OpenCode 使用 "mcp" 而不是 "mcpServers" 作为顶级键
        - OpenCode 使用 "remote" 而不是 "sse" 作为远程服务器的类型
        - OpenCode 需要 "enabled: true" 才能启用 MCP 服务器
        """
        opencode_config = {}
        
        for server_name, server_config in raw_config.items():
            # 构建 OpenCode 格式的配置
            opencode_server = {}
            
            # 转换 transport -> type
            # OpenCode 官方格式：远程服务器使用 "remote"，本地使用 "local"
            # 参考：https://opencode.ai/docs/mcp-servers/
            transport = server_config.get("transport", server_config.get("type", "sse"))
            
            # OpenCode 只支持 "local" 和 "remote" 两种类型
            if transport in ("sse", "http", "websocket"):
                opencode_server["type"] = "remote"
            else:
                opencode_server["type"] = "local"
            
            # 启用 MCP 服务器（必需！）
            opencode_server["enabled"] = True
            
            # 禁用 OAuth（我们的 MCP 服务器通过 nginx 代理，不需要认证）
            opencode_server["oauth"] = False
            
            # URL（远程服务器必需）
            if "url" in server_config:
                # OpenCode 远程 MCP 会自动处理 SSE 连接
                # URL 指向 MCP 服务的基础路径，不需要手动指定 /sse
                url = server_config["url"]
                # 如果 URL 以 /sse 结尾，保留它（nginx 代理需要完整路径）
                opencode_server["url"] = url
            
            # 超时配置（可选）
            if "timeout" in server_config:
                # Bio-Agent 使用秒，OpenCode 使用毫秒
                timeout_seconds = server_config["timeout"]
                opencode_server["timeout"] = timeout_seconds * 1000
            
            # Headers（可选）
            if "headers" in server_config:
                opencode_server["headers"] = server_config["headers"]
            
            # stdio 本地模式的字段
            if "command" in server_config:
                opencode_server["command"] = server_config["command"]
            if "args" in server_config:
                opencode_server["args"] = server_config["args"]
            if "env" in server_config:
                opencode_server["env"] = server_config["env"]
            
            opencode_config[server_name] = opencode_server
        
        return opencode_config
    
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
    ) -> ExecutionResult:
        """
        执行 tasks.md 中的任务
        
        Args:
            tasks_md_path: tasks.md 文件路径（沙盒内）
            workspace_dir: 工作目录
            mode: 执行模式（build 或 plan）
        
        Returns:
            ExecutionResult: 执行结果（包含 task_records 任务执行记录）
        """
        start_time = time.time()
        
        if not self.sandbox:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error="沙盒未创建，请先调用 create_sandbox()"
            )
        
        mode = mode or self.config.opencode_mode
        
        try:
            # 1. 配置 OpenCode
            await self._configure_opencode(workspace_dir)
            
            # 2. 确保输出目录存在
            await self.sandbox.commands.run(f"mkdir -p {workspace_dir}/output")
            
            # 3. 生成执行脚本（避免命令行传参限制）
            runner_script = await self._generate_runner_script(workspace_dir, tasks_md_path, mode)
            
            # 4. 使用 nohup 后台执行脚本
            output_file = f"{workspace_dir}/.opencode_output.log"
            status_file = f"{workspace_dir}/.opencode_status.json"
            
            # 后台执行命令
            background_cmd = f'''cd {workspace_dir} && nohup bash {runner_script} > {output_file} 2>&1 &'''
            self._log(f"执行后台命令: {background_cmd[:100]}...")
            await self.sandbox.commands.run(background_cmd)
            
            self._log(f"OpenCode 任务已启动 (模式: {mode.value})")
            
            # 5. 文件轮询检测完成（避免 SSE 超时）
            max_wait_seconds = self.config.sandbox_timeout_seconds
            poll_interval = 5  # 每5秒检查一次
            elapsed = 0
            
            while elapsed < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                
                # 检查是否完成
                try:
                    output = await self.sandbox.files.read_file(output_file)
                    if "===OPENCODE_DONE===" in output:
                        self._log("OpenCode 任务完成")
                        break
                except Exception:
                    pass  # 文件还未创建或正在写入
                
                if elapsed % 30 == 0:  # 每30秒打印进度
                    self._log(f"等待 OpenCode 执行... ({elapsed}s/{max_wait_seconds}s)")
            
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
            task_records = await self._collect_task_records(workspace_dir, stdout, output_files)
            
            # 确定状态
            if "===OPENCODE_DONE===" in stdout:
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
                sandbox_id=getattr(self.sandbox, 'id', None),
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
                sandbox_id=getattr(self.sandbox, 'id', None) if self.sandbox else None,
                execution_time_ms=execution_time,
            )
    
    async def _generate_runner_script(
        self,
        workspace_dir: str,
        tasks_md_path: str,
        mode: OpenCodeMode,
    ) -> str:
        """
        生成 OpenCode 执行脚本
        
        使用脚本文件而非命令行传参，避免参数长度限制和转义问题。
        """
        output_dir = f"{workspace_dir}/output"
        
        # 构建 prompt（读取自 tasks.md）
        if mode == OpenCodeMode.PLAN:
            prompt = f"""请阅读 {tasks_md_path} 中的任务列表，分析任务并生成执行计划。
注意：这是 plan 模式，只进行分析，不执行实际操作。"""
        else:
            prompt = f"""请阅读 {tasks_md_path} 中的任务列表，按顺序执行所有任务。

【核心原则】MCP 是 OpenCode 的工具接口，只能通过对话调用，Python 脚本无法调用！

【MCP 调用方式 - 必须这样调用】
在对话中直接说：
  "调用 nettcr 的 check_peptide_support，参数 peptide=ELAGIGILTV"

OpenCode 会显示：
  → mcp nettcr.check_peptide_support
  {{"supported": true, "score": 0.85, ...}}

【绝对禁止】
❌ 不要写 Python 脚本检测 MCP
❌ 不要写 json.dumps 模拟返回
❌ 不要写 "call_method": "simulated"

【任务执行日志要求 - 重要！】
每完成一个任务，你必须将任务执行记录追加写入到 {workspace_dir}/output/task_execution_log.json 文件中。

格式如下（JSON 数组，每个任务一个对象）：
```json
[
  {{
    "task_id": "task_1",
    "task_name": "检查肽段支持",
    "task_type": "MCP_TOOL",
    "status": "success",
    "start_time": "2026-03-07T10:30:00Z",
    "end_time": "2026-03-07T10:30:05Z",
    "execution_time_ms": 5000,
    "parameters": {{"peptide": "ELAGIGILTV"}},
    "input_files": [],
    "mcp_tool_name": "nettcr.check_peptide_support",
    "mcp_server": "nettcr",
    "output_files": [],
    "output_data": {{"supported": true, "score": 0.85}},
    "action_description": "调用 nettcr 的 check_peptide_support 工具检查肽段 ELAGIGILTV 是否支持",
    "error_message": null
  }},
  {{
    "task_id": "task_2",
    "task_name": "预测 TCR 结合",
    "task_type": "MCP_TOOL",
    "status": "success",
    "start_time": "2026-03-07T10:30:10Z",
    "end_time": "2026-03-07T10:31:00Z",
    "execution_time_ms": 50000,
    "parameters": {{"peptide": "ELAGIGILTV", "tcr_sequence": "CASS..."}},
    "input_files": ["/data/input/tcr.fasta"],
    "mcp_tool_name": "nettcr.predict_tcr_binding",
    "mcp_server": "nettcr",
    "output_files": ["/workspace/output/prediction_result.csv"],
    "output_data": {{"predictions": [...], "score": 0.92}},
    "action_description": "调用 nettcr 的 predict_tcr_binding 工具预测 TCR 与肽段结合",
    "error_message": null
  }}
]
```

【日志写入方式 - 必须按此方式操作】
1. 首次创建空日志文件：echo '[]' > {workspace_dir}/output/task_execution_log.json
2. 每完成一个任务后，使用以下完整脚本追加记录：
   ```bash
   python3 << 'WRITE_LOG_EOF'
import json
log_path = '{workspace_dir}/output/task_execution_log.json'
with open(log_path, 'r') as f:
    records = json.load(f)
records.append({{
    "task_id": "task_1",
    "task_name": "任务描述",
    "task_type": "MCP_TOOL",
    "status": "success",
    "mcp_tool_name": "工具名称",
    "output_files": ["输出文件路径"],
    "error_message": None
}})
with open(log_path, 'w') as f:
    json.dump(records, f, indent=2)
WRITE_LOG_EOF
   ```

【禁止事项】
❌ 不要引用未定义的变量（如 task_log、record 等）
❌ 不要使用 with open(path, 'r+') 模式（可能导致问题）
❌ 不要在 heredoc 外部定义变量然后在内部引用

【记录字段说明】
- task_id: 任务编号（task_1, task_2, ...）
- task_name: 任务名称/描述
- task_type: MCP_TOOL / CODE_GENERATION / FILE_OPERATION / ANALYSIS / REPORT
- status: success / failed / pending
- parameters: 使用的参数（JSON 对象）
- mcp_tool_name: MCP 工具全名（如 nettcr.predict_tcr_binding）
- mcp_server: MCP 服务名称（如 nettcr）
- output_files: 输出文件路径列表
- output_data: 工具返回的关键数据
- action_description: 一句话描述做了什么
- error_message: 如果失败，记录错误信息

输出要求：所有输出保存到 {output_dir}/ 目录"""

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
        status_icon = '✅' if r.get('status') == 'success' else '❌'
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
echo "===OPENCODE_DONE===" >> {workspace_dir}/.opencode_output.log

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
        target_path: str = "/workspace/.agent/context.json"
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
    
    async def _read_execution_summary(self, workspace_dir: str) -> Optional[Dict[str, Any]]:
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
            result = await self.sandbox.commands.run(f"find {output_dir} -type f 2>/dev/null || echo ''")
            stdout = self._get_stdout(result)
            if stdout.strip():
                return [f.strip() for f in stdout.strip().split('\n') if f.strip()]
        except Exception:
            pass
        return []
    
    # =========================================================================
    # 任务执行记录收集方法
    # =========================================================================
    
    async def _collect_task_records(
        self,
        workspace_dir: str,
        stdout: str,
        output_files: List[str]
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
                    
                    self._log(f"从 task_execution_log.json 读取到 {len(records)} 条任务记录")
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
            task_id = f"task_{i+1}"
            task_name = task_info.get("name", f"任务 {i+1}")
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
                f for f in output_files 
                if task_name.lower() in f.lower() or f"task_{i+1}" in f.lower()
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
                status=ExecutionStatus.SUCCESS if (mcp_call or agent_record or task_output_files) else ExecutionStatus.PENDING,
                parameters=task_params,
                input_files=task_info.get("input_files", []),
                mcp_tool_name=mcp_call.get("tool_name") if mcp_call else None,
                mcp_server=mcp_call.get("server") if mcp_call else None,
                output_files=task_output_files,
                output_data=mcp_call.get("response") if mcp_call else agent_record.get("output_data") if agent_record else None,
                mcp_response=mcp_call.get("raw_response") if mcp_call else None,
                action_description=self._generate_action_description(task_name, task_type, mcp_call, task_output_files),
                raw_output=mcp_call.get("raw_output", "") if mcp_call else "",
            )
            
            records.append(record)
        
        # 3. 如果没有解析到任务，但输出中有 MCP 调用，则创建基于 MCP 调用的记录
        if not records and mcp_calls:
            for i, call in enumerate(mcp_calls):
                record = TaskExecutionRecord(
                    task_id=f"mcp_call_{i+1}",
                    task_name=f"MCP 调用: {call.get('tool_name', 'Unknown')}",
                    task_type=TaskType.MCP_TOOL,
                    status=ExecutionStatus.SUCCESS if call.get("success") else ExecutionStatus.FAILED,
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
                status=ExecutionStatus.SUCCESS if "===OPENCODE_DONE===" in stdout else ExecutionStatus.FAILED,
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
                    if "/" in line and (".csv" in line or ".fasta" in line or ".txt" in line):
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
        mcp_pattern = r'(?:→\s*)?mcp\s+([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)'
        
        # 匹配 JSON 响应（在 MCP 调用后面）
        json_pattern = r'\{[^{}]*\}'
        
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
                json_match = re.search(json_pattern, line[match.end():])
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
                                json_text = "\n".join(lines[j:min(j+10, len(lines))])
                                try:
                                    response = json.loads(json_text.split("\n\n")[0])
                                    raw_response = json_text[:500]
                                    break
                                except json.JSONDecodeError:
                                    pass
                
                # 提取参数（如果有）
                if response and isinstance(response, dict):
                    parameters = {k: v for k, v in response.items() 
                                  if k not in ["success", "status", "result", "data", "error"]}
                
                call_record = {
                    "server": server,
                    "tool_name": f"{server}.{tool_name}",
                    "parameters": parameters,
                    "response": response.get("result") if response and "result" in response else response,
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
        output_files: List[str]
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
        """打印日志"""
        if self.config.show_progress:
            print(f"[OpenCode] {message}")
    
    def _get_stdout(self, result: Any) -> str:
        """获取标准输出"""
        logs = getattr(result, 'logs', None)
        if not logs:
            return ""
        stdout = getattr(logs, 'stdout', None)
        if not stdout:
            return ""
        if isinstance(stdout, str):
            return stdout
        return "\n".join(getattr(entry, 'text', str(entry)) for entry in stdout)
    
    def _get_stderr(self, result: Any) -> str:
        """获取标准错误"""
        logs = getattr(result, 'logs', None)
        if not logs:
            return ""
        stderr = getattr(logs, 'stderr', None)
        if not stderr:
            return ""
        if isinstance(stderr, str):
            return stderr
        return "\n".join(getattr(entry, 'text', str(entry)) for entry in stderr)


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
        return self._run_async(
            self._executor.upload_context(context, target_path)
        )
    
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

