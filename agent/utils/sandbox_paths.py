"""
沙盒路径管理工具

统一管理沙盒目录路径，确保所有子图使用一致的路径格式。

沙盒目录结构：
/data/sessions/{session_id}/
├── input/          # 用户上传的输入文件
├── output/         # 工具执行产生的输出文件
│   ├── reports/    # 各类报告
│   └── *.csv, *.json, etc.  # 工具输出
├── reports/        # 分析报告（兼容旧结构）
├── todo-list.md    # 任务列表
└── workspace/      # 工作空间

路径类型：
1. 服务器路径 (server_path): /data/sessions/{session_id}/...
2. 容器路径 (container_path): /tmp/sessions/{session_id}/...
3. 本地路径 (local_path): D:/path/to/sandbox/...

架构原则：
- 所有文件操作都在沙盒中执行
- 通过 CodeAct 统一接口执行代码
- 输出文件统一保存到 output/ 目录
"""

from typing import Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import os


@dataclass
class SandboxPaths:
    """
    沙盒路径集合
    
    包含一个会话所需的所有沙盒路径信息。
    """
    session_id: str
    server_base: str           # /data/sessions/{session_id}
    container_base: str        # /tmp/sessions/{session_id}
    local_base: Optional[str]  # 本地路径（如果有）
    opensandbox_id: Optional[str]  # OpenSandbox 实例 ID
    
    @property
    def input_dir(self) -> str:
        """服务器输入目录"""
        return f"{self.server_base}/input"
    
    @property
    def output_dir(self) -> str:
        """服务器输出目录"""
        return f"{self.server_base}/output"
    
    @property
    def reports_dir(self) -> str:
        """服务器报告目录"""
        return f"{self.server_base}/output/reports"
    
    @property
    def workspace_dir(self) -> str:
        """服务器工作空间目录"""
        return f"{self.server_base}/workspace"
    
    @property
    def todo_list_path(self) -> str:
        """todo-list.md 服务器路径"""
        return f"{self.server_base}/todo-list.md"
    
    @property
    def container_input_dir(self) -> str:
        """容器输入目录"""
        return f"{self.container_base}/input"
    
    @property
    def container_output_dir(self) -> str:
        """容器输出目录"""
        return f"{self.container_base}/output"
    
    @property
    def container_reports_dir(self) -> str:
        """容器报告目录"""
        return f"{self.container_base}/output/reports"
    
    def server_to_container(self, server_path: str) -> str:
        """将服务器路径转换为容器路径"""
        if server_path.startswith("/data/sessions/"):
            return server_path.replace("/data/sessions/", "/tmp/sessions/", 1)
        return server_path
    
    def container_to_server(self, container_path: str) -> str:
        """将容器路径转换为服务器路径"""
        if container_path.startswith("/tmp/sessions/"):
            return container_path.replace("/tmp/sessions/", "/data/sessions/", 1)
        return container_path
    
    def get_output_path(self, filename: str, subdir: str = "") -> Tuple[str, str]:
        """
        获取输出文件的路径
        
        Args:
            filename: 文件名
            subdir: 子目录（如 "reports"）
        
        Returns:
            (服务器路径, 容器路径)
        """
        if subdir:
            server = f"{self.output_dir}/{subdir}/{filename}"
            container = f"{self.container_output_dir}/{subdir}/{filename}"
        else:
            server = f"{self.output_dir}/{filename}"
            container = f"{self.container_output_dir}/{filename}"
        return server, container
    
    def get_input_path(self, filename: str) -> Tuple[str, str]:
        """
        获取输入文件的路径
        
        Returns:
            (服务器路径, 容器路径)
        """
        return f"{self.input_dir}/{filename}", f"{self.container_input_dir}/{filename}"


def create_sandbox_paths(
    session_id: str,
    local_base: Optional[str] = None,
    opensandbox_id: Optional[str] = None
) -> SandboxPaths:
    """
    创建沙盒路径集合
    
    Args:
        session_id: 会话 ID
        local_base: 本地沙盒目录（可选）
        opensandbox_id: OpenSandbox 实例 ID（可选）
    
    Returns:
        SandboxPaths 实例
    """
    return SandboxPaths(
        session_id=session_id,
        server_base=f"/data/sessions/{session_id}",
        container_base=f"/tmp/sessions/{session_id}",
        local_base=local_base,
        opensandbox_id=opensandbox_id
    )


def get_sandbox_paths_from_state(state: any) -> SandboxPaths:
    """
    从状态对象获取沙盒路径
    
    Args:
        state: GlobalState 或子图状态
    
    Returns:
        SandboxPaths 实例
    """
    session_id = getattr(state, 'session_id', None) or 'unknown'
    sandbox_data_dir = getattr(state, 'sandbox_data_dir', None)
    local_base = getattr(state, 'sandbox_dir', None)
    opensandbox_id = getattr(state, 'opensandbox_id', None)
    
    # 如果已有 sandbox_data_dir，从中提取 session_id
    if sandbox_data_dir:
        if '/sessions/' in sandbox_data_dir:
            session_id = sandbox_data_dir.split('/sessions/')[-1].split('/')[0]
    
    return create_sandbox_paths(
        session_id=session_id,
        local_base=local_base,
        opensandbox_id=opensandbox_id
    )


def ensure_sandbox_dirs(sandbox_paths: SandboxPaths) -> None:
    """
    确保沙盒目录存在（通过 CodeAct 执行）
    
    在沙盒中创建以下目录：
    - input/
    - output/
    - output/reports/
    - workspace/
    """
    from utils.codeact_executor import execute_code_via_codeact
    
    code = f'''
import os

dirs = [
    "{sandbox_paths.container_input_dir}",
    "{sandbox_paths.container_output_dir}",
    "{sandbox_paths.container_reports_dir}",
    "{sandbox_paths.container_workspace_dir if hasattr(sandbox_paths, 'container_workspace_dir') else sandbox_paths.container_base + '/workspace'}"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {{d}}")

print("__SANDBOX_DIRS_CREATED__")
'''
    
    result = execute_code_via_codeact(
        task_description=f"创建沙盒目录结构",
        code_template=code,
        sandbox_id=sandbox_paths.opensandbox_id,
        keep_alive=True
    )
    
    if result.is_success():
        print(f"  [SandboxPaths] 沙盒目录结构已创建")
    else:
        print(f"  [SandboxPaths] 创建目录失败: {result.error}")


# ===================== 便捷函数 =====================

def get_output_file_path(state: any, filename: str, subdir: str = "") -> str:
    """
    获取输出文件的服务器路径
    
    Args:
        state: 状态对象
        filename: 文件名
        subdir: 子目录（如 "reports"）
    
    Returns:
        服务器路径（用于 MCP 工具）
    """
    paths = get_sandbox_paths_from_state(state)
    server_path, _ = paths.get_output_path(filename, subdir)
    return server_path


def get_container_path(server_path: str) -> str:
    """
    将服务器路径转换为容器路径
    
    容器内代码执行时需要使用容器路径。
    """
    if server_path.startswith("/data/sessions/"):
        return server_path.replace("/data/sessions/", "/tmp/sessions/", 1)
    return server_path


def get_server_path(container_path: str) -> str:
    """
    将容器路径转换为服务器路径
    
    MCP 工具需要使用服务器路径。
    """
    if container_path.startswith("/tmp/sessions/"):
        return container_path.replace("/tmp/sessions/", "/data/sessions/", 1)
    return container_path


# ===================== 导出 =====================

__all__ = [
    "SandboxPaths",
    "create_sandbox_paths",
    "get_sandbox_paths_from_state",
    "ensure_sandbox_dirs",
    "get_output_file_path",
    "get_container_path",
    "get_server_path",
]

