"""
OpenSandbox 辅助工具类

提供便捷的沙盒管理接口，包括：
1. 创建和管理沙盒
2. 文件操作（上传、下载、复制、创建）
3. 代码执行
4. 其他实用功能
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import textwrap
import urllib.request
import urllib.error
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

# 加载 .env 文件中的环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装 python-dotenv，跳过加载
    pass


class OpenSandboxHelper:
    """OpenSandbox 辅助类，提供便捷的沙盒操作接口"""
    
    def __init__(
        self,
        timeout: Optional[int] = 600,
        connection_config: Optional[Any] = None,
        ready_timeout: Optional[int] = 10,
    ):
        """
        初始化 OpenSandboxHelper
        
        Args:
            image: Docker 镜像名称（默认从环境变量获取）
            timeout_seconds: 沙盒自动终止超时时间（秒）
            env: 环境变量字典
            connection_config: 连接配置（可选）
            ready_timeout_seconds: 健康检查超时时间（秒）
        """
    def __init__(
        self,
        timeout: Optional[int] = 600,
        connection_config: Optional[Any] = None,
        ready_timeout: Optional[int] = 10,
    ):
        """
        初始化 OpenSandboxHelper
        
        Args:
            image: Docker 镜像名称（默认从环境变量获取)
            timeout_seconds: 沙盒自动终止超时时间（秒)
            env: 环境变量字典
            connection_config: 连接配置(可选)
            ready_timeout_seconds: 健康检查超时时间(秒)
        """
        self.image = self._get_image_from_env()
        print(f"  [opensandbox] 使用镜像: {self.image}")
        self.timeout = timeout
        self.connection_config = connection_config or self._get_connection_config()
        self.ready_timeout = ready_timeout
        
        self.sandbox: Optional[Any] = None
        self.sandbox_id: Optional[str] = None
        self._is_context_manager = False
        self.timeout = timeout
        self.connection_config = connection_config or self._get_connection_config()
        self.ready_timeout = ready_timeout
        
        self.sandbox: Optional[Any] = None
        self.sandbox_id: Optional[str] = None
        self._is_context_manager = False
    
    def _get_connection_config(self) -> Optional[Any]:
        """获取连接配置"""
        try:
            from opensandbox.config import ConnectionConfig
        except ImportError:
            return None
        
        domain = os.getenv("SANDBOX_DOMAIN")
        api_key = os.getenv("SANDBOX_API_KEY")
        debug_enabled = os.getenv("OPENSANDBOX_DEBUG", "false").lower() == "true"
        
        if domain or api_key:
            return ConnectionConfig(
                domain=domain or "localhost:8080",
                api_key=api_key,
                request_timeout=timedelta(seconds=self.timeout),
                debug=debug_enabled,
            )
        
        return None
    
    async def create_sandbox(self) -> str:
        """
        创建沙盒
        
        Args:
            image: Docker 镜像名称（可选，覆盖初始化时的设置）
            timeout_seconds: 超时时间（可选，覆盖初始化时的设置）
            env: 环境变量（可选，会与初始化时的环境变量合并）
            ready_timeout_seconds: 健康检查超时时间（可选）
        
        Returns:
            沙盒ID
        """
        if self.sandbox is not None:
            print(f"  ℹ 沙盒已存在，ID: {self.sandbox_id}")
            return self.sandbox_id
        
        try:
            from opensandbox.sandbox import Sandbox
        except ImportError as e:
            raise ImportError(f"OpenSandbox SDK 未安装: {e}")
        
        # 准备环境变量（与 opensandbox_executor.py 保持一致）
        sandbox_env = {}
        
        # 从环境变量 OPENSANDBOX_ENV_JSON 加载额外的环境变量
        env_json_str = os.getenv("OPENSANDBOX_ENV_JSON", "").strip()
        if env_json_str:
            try:
                env_json = json.loads(env_json_str)
                if isinstance(env_json, dict):
                    sandbox_env.update(env_json)
            except Exception:
                pass  # 忽略 JSON 解析错误
        
        # 关键：传递 API keys 和其他重要环境变量到沙盒
        # 这些环境变量在沙盒中执行代码时会被使用（如 LLM API keys）
        important_env_vars = [
            "DASHSCOPE_API_KEY",
            "QIANFAN_API_KEY", 
            "ZHIPU_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPEN_SANDBOX_API_KEY",
            "SANDBOX_API_KEY",
            # 可以添加其他需要的环境变量
        ]
        
        for env_var in important_env_vars:
            value = os.getenv(env_var)
            if value:
                sandbox_env[env_var] = value
        
        # 也可以传递所有以特定前缀开头的环境变量
        # 例如：DASHCOPE_, QIANFAN_, ZHIPU_ 等
        env_prefixes = ["DASHCOPE_", "QIANFAN_", "ZHIPU_", "OPENAI_", "ANTHROPIC_"]
        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in env_prefixes):
                if key not in sandbox_env:  # 避免覆盖已设置的值
                    sandbox_env[key] = value
        
        # 准备创建参数
        create_kwargs = {
            "timeout": timedelta(seconds=self.timeout),
            "ready_timeout": timedelta(seconds=self.ready_timeout),
        }
        
        if self.connection_config:
            create_kwargs["connection_config"] = self.connection_config
        
        if sandbox_env:
            create_kwargs["env"] = sandbox_env
            print(f"[opensandbox] 已传递 {len(sandbox_env)} 个环境变量到沙盒")
            # 打印关键环境变量（隐藏敏感值）
            key_vars = [k for k in sandbox_env.keys() if "API_KEY" in k or "SECRET" in k]
            if key_vars:
                print(f"[opensandbox] 关键环境变量: {', '.join(key_vars)}")

        # 创建沙盒
        try:
            self.sandbox = await Sandbox.create(self.image, **create_kwargs)
            self.sandbox_id = getattr(self.sandbox, "id", None) or getattr(
                self.sandbox, "sandbox_id", None
            )
            print(f"[opensandbox] ✓ 沙盒创建成功，ID: {self.sandbox_id}")
            return self.sandbox_id
        except Exception as e:
            error_msg = str(e)
            print(f"[opensandbox] ✗ 沙盒创建失败: {error_msg}")
            raise
    
    async def upload_file(
        self, local_path: Union[str, Path], sandbox_path: str, mode: int = 0o644
    ) -> bool:
        """
        上传本地文件到沙盒
        
        Args:
            local_path: 本地文件路径
            sandbox_path: 沙盒中的目标路径
            mode: 文件权限（八进制，默认 0o644）
        
        Returns:
            是否成功
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"本地文件不存在: {local_path}")
        
        try:
            # 读取文件内容
            if local_path.is_dir():
                raise ValueError("不支持上传目录，请使用 upload_directory()")
            
            with open(local_path, "rb") as f:
                content = f.read()
            
            # 写入沙盒
            await self.sandbox.files.write_file(
                sandbox_path, content, mode=mode, encoding="binary"
            )
            print(f"  ✓ 已上传文件: {local_path} -> {sandbox_path}")
            return True
        except Exception as e:
            print(f"  ✗ 上传文件失败: {e}")
            raise
    
    async def upload_directory(
        self, local_dir: Union[str, Path], sandbox_dir: str
    ) -> int:
        """
        上传本地目录到沙盒（递归）
        
        Args:
            local_dir: 本地目录路径
            sandbox_dir: 沙盒中的目标目录路径
        
        Returns:
            上传的文件数量
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            raise NotADirectoryError(f"本地目录不存在或不是目录: {local_dir}")
        
        try:
            from opensandbox.models import WriteEntry
            
            write_entries = []
            
            # 递归收集所有文件
            for file_path in local_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_dir)
                    sandbox_path = f"{sandbox_dir.rstrip('/')}/{relative_path.as_posix()}"
                    
                    with open(file_path, "rb") as f:
                        content = f.read()
                    
                    write_entries.append(
                        WriteEntry(path=sandbox_path, data=content, mode=0o644)
                    )
            
            if not write_entries:
                print(f"  ℹ 目录为空: {local_dir}")
                return 0
            
            # 批量写入
            await self.sandbox.files.write_files(write_entries)
            print(f"  ✓ 已上传目录: {local_dir} -> {sandbox_dir} ({len(write_entries)} 个文件)")
            return len(write_entries)
        except Exception as e:
            print(f"  ✗ 上传目录失败: {e}")
            raise
    
    async def copy_file(
        self, source_path: str, dest_path: str, mode: int = 0o644
    ) -> bool:
        """
        在沙盒内复制文件
        
        Args:
            source_path: 源文件路径（沙盒内）
            dest_path: 目标文件路径（沙盒内）
            mode: 文件权限（八进制，默认 0o644）
        
        Returns:
            是否成功
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            # 读取源文件
            content = await self.sandbox.files.read_file(source_path)
            
            # 写入目标文件
            await self.sandbox.files.write_file(dest_path, content, mode=mode)
            print(f"  ✓ 已复制文件: {source_path} -> {dest_path}")
            return True
        except Exception as e:
            print(f"  ✗ 复制文件失败: {e}")
            raise
    
    async def download_file(
        self, sandbox_path: str, local_path: Union[str, Path]
    ) -> bool:
        """
        从沙盒下载文件到本地
        
        Args:
            sandbox_path: 沙盒中的文件路径
            local_path: 本地目标路径
        
        Returns:
            是否成功
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            # 读取沙盒文件
            content = await self.sandbox.files.read_file(sandbox_path)
            
            # 写入本地文件
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, "wb") as f:
                if isinstance(content, str):
                    f.write(content.encode("utf-8"))
                else:
                    f.write(content)
            
            print(f"  ✓ 已下载文件: {sandbox_path} -> {local_path}")
            return True
        except Exception as e:
            print(f"  ✗ 下载文件失败: {e}")
            raise
    
    async def download_from_url(
        self, url: str, sandbox_path: str, mode: int = 0o644
    ) -> bool:
        """
        从 URL 下载文件到沙盒
        
        Args:
            url: 文件 URL
            sandbox_path: 沙盒中的目标路径
            mode: 文件权限（八进制，默认 0o644）
        
        Returns:
            是否成功
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            # 在沙盒中使用 wget 或 curl 下载
            # 优先使用 wget，如果没有则使用 curl
            download_cmd = f"wget -q -O {sandbox_path} {url} || curl -s -o {sandbox_path} {url}"
            execution = await self.sandbox.commands.run(download_cmd)
            returncode = getattr(execution, "returncode", None) or getattr(
                execution, "exit_code", None
            )
            
            if returncode == 0:
                # 设置文件权限
                await self.sandbox.commands.run(f"chmod {oct(mode)[2:]} {sandbox_path}")
                print(f"  ✓ 已从 URL 下载文件: {url} -> {sandbox_path}")
                return True
            else:
                stdout, stderr = self._extract_logs(execution)
                raise RuntimeError(f"下载失败: {stdout}\n{stderr}")
        except Exception as e:
            print(f"  ✗ 从 URL 下载文件失败: {e}")
            raise
    
    async def create_file(
        self,
        sandbox_path: str,
        content: Union[str, bytes],
        mode: int = 0o644,
        encoding: str = "utf-8",
    ) -> bool:
        """
        在沙盒中创建新文件
        
        Args:
            sandbox_path: 沙盒中的文件路径
            content: 文件内容（字符串或字节）
            mode: 文件权限（八进制，默认 0o644）
            encoding: 编码方式（默认 utf-8）
        
        Returns:
            是否成功
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            # 确保内容是字节
            if isinstance(content, str):
                content_bytes = content.encode(encoding)
            else:
                content_bytes = content
            
            await self.sandbox.files.write_file(
                sandbox_path, content_bytes, mode=mode, encoding="binary"
            )
            print(f"  ✓ 已创建文件: {sandbox_path}")
            return True
        except Exception as e:
            print(f"  ✗ 创建文件失败: {e}")
            raise
    
    async def read_file(self, sandbox_path: str, encoding: str = "utf-8") -> Union[str, bytes]:
        """
        读取沙盒中的文件
        
        Args:
            sandbox_path: 沙盒中的文件路径
            encoding: 编码方式（默认 utf-8，如果为 None 则返回字节）
        
        Returns:
            文件内容（字符串或字节）
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            content = await self.sandbox.files.read_file(sandbox_path)
            
            if encoding and isinstance(content, bytes):
                return content.decode(encoding)
            elif encoding and isinstance(content, str):
                return content
            else:
                return content
        except Exception as e:
            print(f"  ✗ 读取文件失败: {e}")
            raise
    
    async def file_exists(self, sandbox_path: str) -> bool:
        """
        检查沙盒中的文件或目录是否存在
        
        Args:
            sandbox_path: 文件或目录路径
        
        Returns:
            是否存在
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            cmd = f"test -e {sandbox_path} && echo 'exists' || echo 'not_exists'"
            execution = await self.sandbox.commands.run(cmd)
            stdout, _ = self._extract_logs(execution)
            return "exists" in stdout
        except Exception:
            return False
    
    async def get_file_info(self, sandbox_path: str) -> Dict[str, Any]:
        """
        获取沙盒中文件的信息
        
        Args:
            sandbox_path: 文件路径
        
        Returns:
            文件信息字典，包含 size, mode, mtime 等
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            # 使用 stat 命令获取文件信息
            cmd = f"stat -c '%s %a %Y %n' {sandbox_path} 2>/dev/null || stat -f '%z %A %m %N' {sandbox_path} 2>/dev/null || echo ''"
            execution = await self.sandbox.commands.run(cmd)
            stdout, stderr = self._extract_logs(execution)
            
            if not stdout.strip():
                raise FileNotFoundError(f"文件不存在: {sandbox_path}")
            
            parts = stdout.strip().split()
            if len(parts) >= 3:
                return {
                    "size": int(parts[0]),
                    "mode": parts[1],
                    "mtime": int(parts[2]),
                    "path": sandbox_path,
                }
            else:
                raise ValueError(f"无法解析文件信息: {stdout}")
        except Exception as e:
            print(f"  ✗ 获取文件信息失败: {e}")
            raise
    
    async def list_files(self, sandbox_dir: str = "/", recursive: bool = False) -> List[str]:
        """
        列出沙盒中的文件
        
        Args:
            sandbox_dir: 要列出的目录路径（默认 /）
            recursive: 是否递归列出子目录
        
        Returns:
            文件路径列表
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            # 使用 find 命令列出文件
            if recursive:
                cmd = f"find {sandbox_dir} -type f 2>/dev/null || true"
            else:
                cmd = f"find {sandbox_dir} -maxdepth 1 -type f 2>/dev/null || true"
            
            execution = await self.sandbox.commands.run(cmd)
            stdout, _ = self._extract_logs(execution)
            
            files = [line.strip() for line in stdout.split("\n") if line.strip()]
            return files
        except Exception as e:
            print(f"  ✗ 列出文件失败: {e}")
            raise
    
    async def delete_file(self, sandbox_path: str) -> bool:
        """
        删除沙盒中的文件或目录
        
        Args:
            sandbox_path: 要删除的文件或目录路径
        
        Returns:
            是否成功
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            # 使用 rm 命令删除
            cmd = f"rm -rf {sandbox_path}"
            execution = await self.sandbox.commands.run(cmd)
            returncode = getattr(execution, "returncode", None) or getattr(
                execution, "exit_code", None
            )
            
            # 判断是否成功：returncode 为 0 或 None（且没有错误）都视为成功
            has_error = hasattr(execution, "error") and execution.error is not None
            
            if returncode == 0 or (returncode is None and not has_error):
                print(f"  ✓ 已删除: {sandbox_path}")
                return True
            else:
                stdout, stderr = self._extract_logs(execution)
                error_msg = f"删除失败: returncode={returncode}"
                if stdout:
                    error_msg += f", stdout={stdout}"
                if stderr:
                    error_msg += f", stderr={stderr}"
                if has_error:
                    error_msg += f", error={execution.error}"
                raise RuntimeError(error_msg)
        except Exception as e:
            print(f"  ✗ 删除文件失败: {e}")
            raise
    
    async def execute_code(
        self,
        code: str,
        timeout_seconds: Optional[int] = None,
        python_cmd: str = "python3",
    ) -> Dict[str, Any]:
        """
        在沙盒中执行代码
        
        Args:
            code: Python 代码字符串
            timeout_seconds: 执行超时时间（秒，None 表示使用默认值）
            python_cmd: Python 命令（默认 python3）
        
        Returns:
            执行结果字典，包含 stdout, stderr, returncode 等
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            # 确保 /tmp 目录存在
            await self.sandbox.commands.run("mkdir -p /tmp")
            
            # 创建临时代码文件
            import uuid
            code_path = f"/tmp/code_{uuid.uuid4().hex[:8]}.py"
            
            # 处理代码：去除公共前导空白（解决多行字符串缩进问题）
            # textwrap.dedent 会自动去除所有行的公共前导空白
            processed_code = textwrap.dedent(code).strip()
            
            # 写入代码
            await self.sandbox.files.write_file(code_path, processed_code.encode("utf-8"))
            
            # 执行代码
            # 直接使用指定的 Python 解释器执行代码
            # 如果 python_cmd 是虚拟环境的 Python 解释器路径，它会自动加载虚拟环境的包
            cmd = f"{python_cmd} {code_path}"
            print(f"  ▶ 执行代码: {code_path}")
            print(f"  ▶ 执行命令: {cmd}")
            
            execution = await self.sandbox.commands.run(cmd)
            print(f"  ▶ 执行结果: {execution}")
            
            # 提取结果
            stdout, stderr = self._extract_logs(execution)
            returncode = getattr(execution, "returncode", None) or getattr(
                execution, "exit_code", None
            )
            
            # 清理临时文件
            try:
                await self.delete_file(code_path)
            except Exception:
                pass  # 忽略清理错误
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "has_error": hasattr(execution, "error") and execution.error is not None,
            }
        except Exception as e:
            print(f"  ✗ 执行代码失败: {e}")
            raise
    
    async def run_command(
        self, command: str, timeout_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        在沙盒中执行命令
        
        Args:
            command: 要执行的命令
            timeout_seconds: 超时时间（秒，None 表示使用默认值）
        
        Returns:
            执行结果字典，包含 stdout, stderr, returncode 等
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            print(f"  ▶ 执行命令: {command}")
            execution = await self.sandbox.commands.run(command)
            
            stdout, stderr = self._extract_logs(execution)
            returncode = getattr(execution, "returncode", None) or getattr(
                execution, "exit_code", None
            )
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "has_error": hasattr(execution, "error") and execution.error is not None,
            }
        except Exception as e:
            print(f"  ✗ 执行命令失败: {e}")
            raise
    
    async def get_info(self) -> Dict[str, Any]:
        """
        获取沙盒信息
        
        Returns:
            沙盒信息字典
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        try:
            info = await self.sandbox.get_info()
            return {
                "id": getattr(info, "id", None),
                "status": getattr(info, "status", None),
                "image": getattr(info, "image", None),
                "created_at": getattr(info, "created_at", None),
            }
        except Exception as e:
            print(f"  ✗ 获取沙盒信息失败: {e}")
            raise
    
    def _extract_logs(self, execution: Any) -> tuple[str, str]:
        """提取执行日志"""
        logs = getattr(execution, "logs", None)
        if not logs:
            return "", ""
        
        stdout = self._collect_log_text(getattr(logs, "stdout", None))
        stderr = self._collect_log_text(getattr(logs, "stderr", None))
        return stdout, stderr
    
    def _collect_log_text(self, log_entries: Optional[Any]) -> str:
        """收集日志文本"""
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
    
    async def close(self) -> None:
        """关闭并清理沙盒"""
        if self.sandbox is not None:
            try:
                await self.sandbox.kill()
                await self.sandbox.close()
                print(f"[opensandbox] ✓ 沙盒已关闭: {self.sandbox_id}")
            except Exception as e:
                print(f"[opensandbox] ⚠ 关闭沙盒时出错: {e}")
            finally:
                self.sandbox = None
                self.sandbox_id = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self._is_context_manager = True
        await self.create_sandbox()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if not self._is_context_manager:
            return
        await self.close()
    
    def __del__(self):
        """析构函数，确保资源清理"""
        if self.sandbox is not None:
            # 注意：在析构函数中不能使用 await，所以这里只是警告
            print(f"[opensandbox] ⚠ 警告: 沙盒 {self.sandbox_id} 未正确关闭，请使用 close() 或 async with")
    
    async def execute_subgraph_in_sandbox(
        self,
        subgraph_module_path: str,
        subgraph_builder_name: str,
        input_mapper_name: str,
        output_mapper_name: str,
        input_state: Any,
        agent_dir: str = "/data/server/ImmuneAgent_2.0/agent",
    ) -> Dict[str, Any]:
        """
        在沙盒中执行子图的通用函数
        
        Args:
            subgraph_module_path: 子图模块的导入路径，如 "nodes.subagents.supervisor.react_supervisor"
            subgraph_builder_name: 构建子图的函数名，如 "build_react_supervisor_subgraph"
            input_mapper_name: 输入映射函数名，如 "supervisor_input_mapper"
            output_mapper_name: 输出映射函数名，如 "supervisor_output_mapper"
            input_state: 输入状态对象（需要可序列化）
            agent_dir: 服务器上 agent 目录的路径，默认 "/data/server/ImmuneAgent_2.0/agent"
        
        Returns:
            执行结果字典，包含 output, error 等信息
        """
        if self.sandbox is None:
            raise RuntimeError("沙盒未创建，请先调用 create_sandbox()")
        
        # 序列化输入状态为 JSON
        try:
            if hasattr(input_state, "model_dump"):
                input_state_dict = input_state.model_dump(mode="json")
            elif hasattr(input_state, "__dict__"):
                input_state_dict = {k: v for k, v in input_state.__dict__.items() if not k.startswith("_")}
            else:
                input_state_dict = {"value": str(input_state)}
            input_state_json = json.dumps(input_state_dict, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"无法序列化输入状态: {e}")
        
        # 查找虚拟环境
        venv_paths = [
            f"{agent_dir}/.venv",
            f"{agent_dir}/../.venv",
        ]
        
        venv_dir = None
        for venv_path in venv_paths:
            if await self.file_exists(f"{venv_path}/bin/activate"):
                venv_dir = venv_path
                break
        
        # 构建虚拟环境设置代码
        venv_setup_code = ""
        if venv_dir:
            venv_setup_code = f"""
# 手动添加虚拟环境的 site-packages 到 sys.path
import sys
import sysconfig
import os
from pathlib import Path

venv_dir = Path("{venv_dir}")
if venv_dir.exists():
    lib_dir = venv_dir / "lib"
    venv_python_versions = []
    if lib_dir.exists():
        for item in lib_dir.iterdir():
            if item.is_dir() and item.name.startswith("python"):
                version_str = item.name.replace("python", "")
                try:
                    major, minor = map(int, version_str.split("."))
                    venv_python_versions.append((major, minor, item))
                except:
                    pass
    
    current_version = (sys.version_info.major, sys.version_info.minor)
    matching_version = None
    for major, minor, path in venv_python_versions:
        if (major, minor) == current_version:
            matching_version = (major, minor, path)
            break
    
    if not matching_version and venv_python_versions:
        matching_version = venv_python_versions[0]
    
    possible_paths = []
    if matching_version:
        major, minor, path = matching_version
        possible_paths = [
            path / "site-packages",
            path / "dist-packages",
        ]
    else:
        possible_paths = [
            venv_dir / "lib" / f"python{{sys.version_info.major}}.{{sys.version_info.minor}}" / "site-packages",
            Path(sysconfig.get_path('purelib', vars={{'base': str(venv_dir)}})),
        ]
    
    site_packages_found = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            site_packages_found = path
            break
    
    if not site_packages_found and lib_dir.exists():
        for item in lib_dir.iterdir():
            if item.is_dir() and item.name.startswith("python"):
                site_packages = item / "site-packages"
                if site_packages.exists():
                    site_packages_found = site_packages
                    break
    
    if site_packages_found and str(site_packages_found) not in sys.path:
        sys.path.insert(0, str(site_packages_found))

"""
        
        # 构建执行代码
        # 转义 JSON 字符串以便在 Python 代码中使用
        input_state_json_escaped = json.dumps(input_state_json)
        
        execute_code = f"""
import sys
import os
import json
from pathlib import Path

# 添加服务器上的 agent 目录到 Python 路径
agent_dir = Path("{agent_dir}")
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

try:
    # 导入子图模块
    from {subgraph_module_path} import (
        {subgraph_builder_name},
        {input_mapper_name},
        {output_mapper_name}
    )
    
    # 构建子图
    subgraph = {subgraph_builder_name}()
    
    # 反序列化输入状态
    input_state_dict = json.loads({input_state_json_escaped})
    
    # 尝试创建输入状态对象（支持多种状态类型）
    input_state = None
    try:
        # 尝试从子图模块导入状态类型
        module = __import__("{subgraph_module_path}", fromlist=["ReactSupervisorState", "SupervisorState"])
        for state_class_name in ["ReactSupervisorState", "SupervisorState"]:
            if hasattr(module, state_class_name):
                state_class = getattr(module, state_class_name)
                try:
                    input_state = state_class(**input_state_dict)
                    break
                except:
                    continue
    except:
        pass
    
    # 如果无法创建状态对象，尝试使用 GlobalState
    if input_state is None:
        try:
            from state import GlobalState
            input_state = GlobalState(**input_state_dict)
        except:
            # 如果都失败，使用字典
            input_state = input_state_dict
    
    # 调用子图
    output = subgraph.invoke(input_state)
    
    # 序列化输出（递归处理嵌套的 Pydantic 模型）
    def serialize_for_json(obj):
        # Recursively serialize object, handling Pydantic models and non-serializable types
        if hasattr(obj, 'model_dump'):
            # Pydantic model
            return obj.model_dump(mode="json")
        elif isinstance(obj, dict):
            return {{k: serialize_for_json(v) for k, v in obj.items()}}
        elif isinstance(obj, (list, tuple)):
            return [serialize_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            # Regular object, convert to dict
            return {{k: serialize_for_json(v) for k, v in obj.__dict__.items() if not k.startswith("_")}}
        else:
            # Other types, convert to string
            return str(obj)
    
    try:
        output_dict = serialize_for_json(output)
    except Exception as serialize_error:
        # If serialization fails, try using model_dump_json
        try:
            if hasattr(output, 'model_dump_json'):
                import json as json_module
                output_dict = json_module.loads(output.model_dump_json())
            else:
                output_dict = {{"value": str(output), "serialize_error": str(serialize_error)}}
        except:
            output_dict = {{"value": str(output)}}
    
    result = {{
        "status": "success",
        "output": output_dict
    }}
    
except Exception as e:
    import traceback
    result = {{
        "status": "failed",
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()
    }}

# 输出结果（使用特殊标记以便解析）
# 使用 sys.stdout.flush() 确保输出立即刷新
import sys
try:
    result_json = json.dumps(result, ensure_ascii=False)
    # 输出到单独一行，避免与其他输出混合
    print("\\n__SUBGRAPH_RESULT__" + result_json + "\\n", flush=True)
except Exception as print_err:
    # 如果 JSON 序列化失败，尝试使用 repr
    try:
        error_result = {{
            "status": "failed",
            "error": f"无法序列化结果: {{print_err}}",
            "result_type": str(type(result)),
            "result_repr": repr(result)[:500]
        }}
        print("\\n__SUBGRAPH_RESULT__" + json.dumps(error_result, ensure_ascii=False) + "\\n", flush=True)
    except:
        print("\\n__SUBGRAPH_RESULT__" + str(result) + "\\n", flush=True)
sys.stdout.flush()
"""
        
        # 合并代码
        full_code = venv_setup_code + execute_code
        
        # 执行代码
        exec_result = await self.execute_code(full_code, python_cmd="python3")
        
        # 解析结果
        stdout = exec_result.get("stdout", "")
        stderr = exec_result.get("stderr", "")
        
        # 从 stdout 中提取结果
        # 查找所有包含 __SUBGRAPH_RESULT__ 的行，使用最后一行（因为可能有多个输出）
        result = None
        result_lines = [line for line in stdout.splitlines() if "__SUBGRAPH_RESULT__" in line]
        
        if result_lines:
            # 使用最后一行（最新的结果）
            result_line = result_lines[-1]
            try:
                # 提取标记后的内容
                result_str = result_line.split("__SUBGRAPH_RESULT__", 1)[1].strip()
                
                # 尝试解析 JSON
                result = json.loads(result_str)
            except json.JSONDecodeError as e:
                # JSON 解析失败，尝试提取更多上下文
                print(f"[opensandbox] JSON 解析失败: {e}")
                print(f"[opensandbox] 尝试解析的内容: {result_str[:200]}...")
                
                # 尝试提取 JSON 对象（可能被其他文本包围）
                import re
                json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        result = {
                            "status": "failed",
                            "error": f"无法解析结果 JSON: {e}",
                            "error_detail": f"原始内容: {result_str[:500]}",
                            "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,  # 只保留最后1000字符
                            "stderr": stderr[-1000:] if len(stderr) > 1000 else stderr
                        }
                else:
                    result = {
                        "status": "failed",
                        "error": f"无法解析结果 JSON: {e}",
                        "error_detail": f"未找到 JSON 对象，原始内容: {result_str[:500]}",
                        "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
                        "stderr": stderr[-1000:] if len(stderr) > 1000 else stderr
                    }
            except Exception as e:
                result = {
                    "status": "failed",
                    "error": f"解析结果时发生异常: {e}",
                    "error_type": type(e).__name__,
                    "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
                    "stderr": stderr[-1000:] if len(stderr) > 1000 else stderr
                }
        else:
            # 未找到结果标记，检查是否有错误输出
            error_info = ""
            if stderr:
                error_info = f"Stderr: {stderr[-500:] if len(stderr) > 500 else stderr}"
            elif stdout:
                # 检查 stdout 中是否有错误信息
                error_lines = [line for line in stdout.splitlines() if any(keyword in line.lower() for keyword in ["error", "exception", "traceback", "failed"])]
                if error_lines:
                    error_info = f"可能的错误: {'; '.join(error_lines[-3:])}"
            
            result = {
                "status": "failed",
                "error": "未找到执行结果标记",
                "error_detail": error_info,
                "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
                "stderr": stderr[-1000:] if len(stderr) > 1000 else stderr
            }
        
        return result


# 便捷函数（同步包装）
def create_sandbox_helper(
    image: Optional[str] = None,
    timeout_seconds: int = 600,
    env: Optional[Dict[str, str]] = None,
) -> OpenSandboxHelper:
    """
    创建 OpenSandboxHelper 实例（同步函数）
    
    Args:
        image: Docker 镜像名称
        timeout_seconds: 超时时间（秒）
        env: 环境变量字典
    
    Returns:
        OpenSandboxHelper 实例
    """
    return OpenSandboxHelper()


# 示例用法
if __name__ == "__main__":
    async def example():
        # 方式2: 手动管理
        print("\n=== 示例: 手动管理沙盒 ===")
        helper = OpenSandboxHelper()
        try:
            await helper.create_sandbox()
            
            # 测试在沙盒中调用 supervisor 子图
            print("\n=== 测试: 在沙盒中调用 supervisor 子图 ===")
            
            # 子图文件在服务器上的路径
            supervisor_file_path = "/data/server/ImmuneAgent_2.0/agent/nodes/subagents/supervisor/react_supervisor.py"
            
            # 检查文件是否存在（通过挂载卷）
            file_exists = await helper.file_exists(supervisor_file_path)
            print(f"子图文件是否存在: {file_exists}")
            
            if file_exists:
                # 在沙盒中执行代码，尝试导入并调用 supervisor 子图
                # 重要：先激活虚拟环境，然后使用虚拟环境的 Python 解释器执行代码
                agent_dir = "/data/server/ImmuneAgent_2.0/agent"
                venv_paths = [
                    f"{agent_dir}/.venv",
                    f"{agent_dir}/../.venv",
                ]
                
                # 查找虚拟环境目录（用于添加 site-packages 到 sys.path）
                venv_dir = None
                for venv_path in venv_paths:
                    # 检查虚拟环境目录是否存在
                    if await helper.file_exists(f"{venv_path}/bin/activate"):
                        venv_dir = venv_path
                        print(f"✓ 找到虚拟环境: {venv_path}")
                        break
                
                if not venv_dir:
                    print("⚠ 未找到虚拟环境，将使用系统 Python")
                
                # 构建代码，手动添加虚拟环境的 site-packages 到 sys.path
                venv_setup_code = ""
                if venv_dir:
                    venv_setup_code = f"""
# 手动添加虚拟环境的 site-packages 到 sys.path
import sys
import sysconfig
import os
from pathlib import Path

print(f"当前 Python 版本: {{sys.version_info.major}}.{{sys.version_info.minor}}.{{sys.version_info.micro}}")
print(f"Python 解释器: {{sys.executable}}")

venv_dir = Path("{venv_dir}")
if venv_dir.exists():
    # 首先检查虚拟环境的 Python 版本
    lib_dir = venv_dir / "lib"
    venv_python_versions = []
    if lib_dir.exists():
        for item in lib_dir.iterdir():
            if item.is_dir() and item.name.startswith("python"):
                version_str = item.name.replace("python", "")
                try:
                    major, minor = map(int, version_str.split("."))
                    venv_python_versions.append((major, minor, item))
                except:
                    pass
    
    # 检查版本匹配
    current_version = (sys.version_info.major, sys.version_info.minor)
    matching_version = None
    for major, minor, path in venv_python_versions:
        if (major, minor) == current_version:
            matching_version = (major, minor, path)
            break
    
    if not matching_version and venv_python_versions:
        print(f"⚠ 警告: 虚拟环境的 Python 版本与当前 Python 版本不匹配！")
        print(f"  当前 Python 版本: {{current_version[0]}}.{{current_version[1]}}")
        print(f"  虚拟环境中的版本: {{', '.join([f'{{v[0]}}.{{v[1]}}' for v in venv_python_versions])}}")
        print(f"  这可能导致二进制扩展模块（如 pydantic_core）无法加载")
        print(f"  建议: 使用与虚拟环境匹配的 Python 版本，或在沙盒中安装依赖包")
        # 仍然尝试使用第一个可用的版本
        matching_version = venv_python_versions[0]
    
    # 尝试多种可能的 site-packages 路径
    possible_paths = []
    if matching_version:
        major, minor, path = matching_version
        possible_paths = [
            path / "site-packages",
            path / "dist-packages",
        ]
    else:
        # 如果没有找到匹配的版本，尝试当前版本
        possible_paths = [
            venv_dir / "lib" / f"python{{sys.version_info.major}}.{{sys.version_info.minor}}" / "site-packages",
            Path(sysconfig.get_path('purelib', vars={{'base': str(venv_dir)}})),
            venv_dir / "local" / "lib" / f"python{{sys.version_info.major}}.{{sys.version_info.minor}}" / "site-packages",
            venv_dir / "local" / "lib" / f"python{{sys.version_info.major}}.{{sys.version_info.minor}}" / "dist-packages",
        ]
    
    # 如果标准路径不存在，尝试查找实际的 site-packages
    site_packages_found = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            site_packages_found = path
            break
    
    # 如果还是没找到，尝试遍历 lib 目录查找
    if not site_packages_found and lib_dir.exists():
        for item in lib_dir.iterdir():
            if item.is_dir() and item.name.startswith("python"):
                site_packages = item / "site-packages"
                if site_packages.exists():
                    site_packages_found = site_packages
                    break
                dist_packages = item / "dist-packages"
                if dist_packages.exists():
                    site_packages_found = dist_packages
                    break
    
    if site_packages_found:
        if str(site_packages_found) not in sys.path:
            sys.path.insert(0, str(site_packages_found))
            print(f"✓ 已添加虚拟环境 site-packages: {{site_packages_found}}")
        else:
            print(f"✓ 虚拟环境 site-packages 已在路径中: {{site_packages_found}}")
    else:
        print(f"⚠ 无法找到虚拟环境 site-packages，尝试的路径:")
        for path in possible_paths:
            print(f"  - {{path}} (存在: {{path.exists()}})")
else:
    print(f"⚠ 虚拟环境目录不存在: {{venv_dir}}")

"""
                
                # 使用系统 Python 执行代码，但手动加载虚拟环境的包
                test_code = """
import sys
import os
from pathlib import Path

# 添加服务器上的 agent 目录到 Python 路径
agent_dir = Path("/data/server/ImmuneAgent_2.0/agent")
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

print(f"Python 路径: {sys.path[:3]}...")
print(f"Python 解释器: {sys.executable}")

try:
    # 尝试导入 supervisor 子图
    from nodes.subagents.supervisor.react_supervisor import (
        build_react_supervisor_subgraph,
        supervisor_input_mapper,
        supervisor_output_mapper
    )
    print("✓ 成功导入 supervisor 子图模块")
    
    # 尝试构建子图
    subgraph = build_react_supervisor_subgraph()
    print(f"✓ 成功构建 supervisor 子图: {type(subgraph)}")
    
    # 创建测试状态
    from state import GlobalState
    test_state = GlobalState(
        user_input="测试输入：分析抗体序列",
        sandbox_dir="/tmp/test_sandbox"
    )
    print("✓ 成功创建测试状态")
    print(f"  初始状态:")
    print(f"    - user_input: {test_state.user_input}")
    print(f"    - user_task_type: {test_state.user_task_type}")
    print(f"    - supervisor_decision: {test_state.supervisor_decision}")
    print(f"    - supervisor_reasoning: {test_state.supervisor_reasoning}")
    print(f"    - file_paths: {test_state.file_paths}")
    
    # 映射输入
    input_state = supervisor_input_mapper(test_state)
    print(f"✓ 成功映射输入状态: {type(input_state)}")
    print(f"  输入状态:")
    print(f"    - user_input: {input_state.user_input}")
    print(f"    - user_task_type: {input_state.user_task_type}")
    print(f"    - uploaded_files: {input_state.uploaded_files}")
    
    # 调用子图（使用 invoke）
    try:
        output = subgraph.invoke(input_state)
        print("✓ 成功调用 supervisor 子图")
        print(f"输出类型: {type(output)}")
        
        # 展示子图输出详情
        print("")
        print("  子图输出详情:")
        # 处理字典类型的输出
        if isinstance(output, dict):
            for key, value in output.items():
                if value is not None:
                    if isinstance(value, str) and len(value) > 200:
                        print(f"    - {key}: {value[:200]}...")
                    else:
                        print(f"    - {key}: {value}")
        else:
            # 处理对象类型的输出
            if hasattr(output, 'user_task_type'):
                print(f"    - user_task_type: {output.user_task_type}")
            if hasattr(output, 'decision'):
                print(f"    - decision: {output.decision}")
            if hasattr(output, 'reasoning'):
                reasoning_str = str(output.reasoning)
                if len(reasoning_str) > 200:
                    print(f"    - reasoning: {reasoning_str[:200]}...")
                else:
                    print(f"    - reasoning: {reasoning_str}")
            if hasattr(output, 'sandbox_file_paths'):
                print(f"    - sandbox_file_paths: {output.sandbox_file_paths}")
            if hasattr(output, 'uploaded_files'):
                print(f"    - uploaded_files: {output.uploaded_files}")
        
        # 映射回全局状态
        updated_state = supervisor_output_mapper(output, test_state)
        print("✓ 成功映射输出状态")
        
        # 展示更新后的状态变化
        print("")
        print("  更新后的状态:")
        print(f"    - user_input: {updated_state.user_input}")
        print(f"    - user_task_type: {updated_state.user_task_type} (变化: {test_state.user_task_type} -> {updated_state.user_task_type})")
        print(f"    - supervisor_decision: {updated_state.supervisor_decision} (变化: {test_state.supervisor_decision} -> {updated_state.supervisor_decision})")
        if updated_state.supervisor_reasoning:
            reasoning_preview = updated_state.supervisor_reasoning[:100] + "..." if len(updated_state.supervisor_reasoning) > 100 else updated_state.supervisor_reasoning
            print(f"    - supervisor_reasoning: {reasoning_preview}")
        else:
            print(f"    - supervisor_reasoning: {updated_state.supervisor_reasoning}")
        print(f"    - file_paths: {updated_state.file_paths} (变化: {len(test_state.file_paths)} -> {len(updated_state.file_paths)} 个文件)")
        
        # 如果有推理内容，完整展示
        if updated_state.supervisor_reasoning:
            print("")
            print("  完整推理内容:")
            print(f"    {updated_state.supervisor_reasoning}")
        
    except Exception as invoke_e:
        print(f"✗ 调用子图失败: {type(invoke_e).__name__}: {invoke_e}")
        import traceback
        traceback.print_exc()
        
except ImportError as import_e:
    print(f"✗ 导入失败: {import_e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ 执行失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
"""
                
                # 合并虚拟环境设置代码和测试代码
                full_test_code = venv_setup_code + test_code
                
                # 使用系统 Python 执行代码（虚拟环境的包已手动添加到 sys.path）
                result = await helper.execute_code(full_test_code, python_cmd="python3")
                print(f"\n执行结果:")
                print(f"stdout:\n{result['stdout']}")
                if result['stderr']:
                    print(f"stderr:\n{result['stderr']}")
                if result['returncode'] != 0:
                    print(f"返回码: {result['returncode']}")
            else:
                print(f"⚠ 子图文件不存在: {supervisor_file_path}")
                print("提示: 确保沙盒已正确挂载 /data 卷")
            
        finally:
            await helper.close()
    
    # 运行示例
    asyncio.run(example())

