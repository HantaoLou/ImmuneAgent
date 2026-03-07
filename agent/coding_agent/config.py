"""
Coding Agent 配置模块

定义 OpenCode + OpenSandbox 集成的配置类和结果模型。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path


class OpenCodeMode(str, Enum):
    """OpenCode 执行模式"""
    BUILD = "build"  # 全权限开发模式
    PLAN = "plan"    # 只读分析模式


class ExecutionStatus(str, Enum):
    """执行状态"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    PENDING = "pending"


class TaskType(str, Enum):
    """任务类型"""
    MCP_TOOL = "MCP_TOOL"              # MCP 工具调用
    CODE_GENERATION = "CODE_GENERATION" # 代码生成
    FILE_OPERATION = "FILE_OPERATION"   # 文件操作
    DATA_TRANSFORM = "DATA_TRANSFORM"   # 数据转换
    ANALYSIS = "ANALYSIS"               # 数据分析
    REPORT = "REPORT"                   # 报告生成


@dataclass
class OpenCodeConfig:
    """OpenCode 配置"""
    
    # 模型配置
    model_provider: str = "glm-4.7"  # 支持: glm-4.7, claude-sonnet-4, gpt-4o, deepseek-chat
    api_key: Optional[str] = None
    
    # 沙盒配置
    sandbox_domain: str = "localhost:8080"
    # 注意：sandbox_image 必须从环境变量 OPENSANDBOX_IMAGE 读取
    # 这里的默认值仅在环境变量未设置时使用
    sandbox_image: str = field(default_factory=lambda: os.getenv(
        "OPENSANDBOX_IMAGE",
        "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.1"
    ))
    sandbox_timeout_seconds: int = 1800  # 30 分钟
    sandbox_ready_timeout_seconds: int = 300  # 5 分钟
    
    # OpenCode 配置
    opencode_mode: OpenCodeMode = OpenCodeMode.BUILD
    opencode_install_command: str = "npm install -g opencode-ai@latest"
    
    # 工作目录配置
    workspace_base: str = "/tmp/sessions"
    
    # MCP 配置
    mcp_config_path: Optional[str] = None  # 默认为 None，运行时自动查找
    
    # Common Tools MCP 配置
    enable_common_tools: bool = False  # 是否启用 Common Tools MCP 服务
    common_tools_mcp_url: str = "http://host.docker.internal:40002/mcp/sse"  # Common Tools MCP URL
    
    # 调试配置
    debug: bool = False
    show_progress: bool = True
    
    @classmethod
    def from_env(cls) -> "OpenCodeConfig":
        """从环境变量加载配置"""
        # 自动查找 MCP 配置文件
        mcp_config_path = os.getenv("MCP_SERVERS_CONFIG")
        if not mcp_config_path:
            # 尝试默认路径
            default_paths = [
                "config/mcp_servers.json",
                "agent/config/mcp_servers.json",
                os.path.join(os.path.dirname(__file__), "..", "config", "mcp_servers.json"),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    mcp_config_path = path
                    break
        
        return cls(
            model_provider=os.getenv("OPENCODE_MODEL_PROVIDER", "glm-4.7"),
            api_key=os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
            sandbox_domain=os.getenv("SANDBOX_DOMAIN", "localhost:8080"),
            sandbox_image=os.getenv("OPENSANDBOX_IMAGE", "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.1"),
            sandbox_timeout_seconds=int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "1800")),
            sandbox_ready_timeout_seconds=int(os.getenv("OPENSANDBOX_READY_TIMEOUT_SECONDS", "300")),
            opencode_mode=OpenCodeMode(os.getenv("OPENCODE_MODE", "build")),
            mcp_config_path=mcp_config_path,
            debug=os.getenv("OPENCODE_DEBUG", "false").lower() == "true",
            show_progress=os.getenv("OPENCODE_SHOW_PROGRESS", "true").lower() == "true",
        )


@dataclass
class TaskExecutionRecord:
    """
    单个任务的执行记录
    
    记录每个任务的详细执行信息，包括：
    - 任务描述和参数
    - 执行状态和时间
    - 输出结果和产物
    - MCP 调用详情
    """
    
    # 任务标识
    task_id: str                                    # 任务唯一标识 (如 "task_1", "task_2")
    task_name: str                                  # 任务名称/描述
    task_type: TaskType = TaskType.MCP_TOOL         # 任务类型
    
    # 执行状态
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[str] = None                # 开始时间 (ISO 格式)
    end_time: Optional[str] = None                  # 结束时间 (ISO 格式)
    execution_time_ms: int = 0                      # 执行耗时（毫秒）
    
    # 输入参数
    parameters: Dict[str, Any] = field(default_factory=dict)     # 使用的参数
    input_files: List[str] = field(default_factory=list)         # 输入文件
    mcp_tool_name: Optional[str] = None              # MCP 工具名称 (如 "nettcr.predict_tcr_binding")
    mcp_server: Optional[str] = None                 # MCP 服务器名称 (如 "nettcr")
    
    # 输出结果
    output_files: List[str] = field(default_factory=list)        # 输出文件
    output_data: Optional[Dict[str, Any]] = None                # 输出数据 (解析后的结果)
    mcp_response: Optional[Dict[str, Any]] = None               # MCP 原始响应
    
    # 执行详情
    action_description: str = ""                     # 做了什么事情的描述
    raw_output: str = ""                             # 原始输出 (stdout 片段)
    error_message: Optional[str] = None             # 错误信息
    retry_count: int = 0                             # 重试次数
    
    def is_success(self) -> bool:
        """检查任务是否成功"""
        return self.status == ExecutionStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time_ms": self.execution_time_ms,
            "parameters": self.parameters,
            "input_files": self.input_files,
            "mcp_tool_name": self.mcp_tool_name,
            "mcp_server": self.mcp_server,
            "output_files": self.output_files,
            "output_data": self.output_data,
            "mcp_response": self.mcp_response,
            "action_description": self.action_description,
            "raw_output": self.raw_output,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式的报告"""
        lines = [
            f"### {self.task_id}: {self.task_name}",
            "",
            f"**状态**: {'✅ 成功' if self.is_success() else '❌ 失败'}",
            f"**类型**: {self.task_type.value}",
            f"**执行时间**: {self.execution_time_ms}ms",
            "",
        ]
        
        if self.parameters:
            lines.append("**使用参数**:")
            lines.append("```json")
            lines.append(json.dumps(self.parameters, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
        
        if self.action_description:
            lines.append("**执行操作**:")
            lines.append(self.action_description)
            lines.append("")
        
        if self.output_files:
            lines.append("**输出文件**:")
            for f in self.output_files:
                lines.append(f"- `{f}`")
            lines.append("")
        
        if self.output_data:
            lines.append("**输出结果**:")
            lines.append("```json")
            lines.append(json.dumps(self.output_data, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
        
        if self.error_message:
            lines.append("**错误信息**:")
            lines.append(f"```\n{self.error_message}\n```")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class ExecutionResult:
    """执行结果"""
    
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    sandbox_id: Optional[str] = None
    execution_time_ms: int = 0
    
    # 输出文件
    output_files: List[str] = field(default_factory=list)
    
    # 执行摘要
    summary: Optional[Dict[str, Any]] = None
    
    # 任务完成情况
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    
    # 【新增】任务执行记录 - 详细记录每个任务的执行情况
    task_records: List[TaskExecutionRecord] = field(default_factory=list)
    
    def is_success(self) -> bool:
        """检查是否执行成功"""
        return self.status == ExecutionStatus.SUCCESS
    
    def get_task_record(self, task_id: str) -> Optional[TaskExecutionRecord]:
        """根据 task_id 获取任务记录"""
        for record in self.task_records:
            if record.task_id == task_id:
                return record
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "sandbox_id": self.sandbox_id,
            "execution_time_ms": self.execution_time_ms,
            "output_files": self.output_files,
            "summary": self.summary,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "task_records": [r.to_dict() for r in self.task_records],
        }
    
    def to_execution_report(self) -> str:
        """生成 Markdown 格式的执行报告"""
        lines = [
            "# 任务执行报告",
            "",
            f"**总体状态**: {'✅ 成功' if self.is_success() else '❌ 失败'}",
            f"**沙盒 ID**: {self.sandbox_id or 'N/A'}",
            f"**总执行时间**: {self.execution_time_ms}ms ({self.execution_time_ms / 1000:.2f}s)",
            f"**完成任务数**: {len(self.completed_tasks)}",
            f"**失败任务数**: {len(self.failed_tasks)}",
            "",
        ]
        
        if self.task_records:
            lines.append("---")
            lines.append("")
            lines.append("## 任务详情")
            lines.append("")
            for record in self.task_records:
                lines.append(record.to_markdown())
                lines.append("---")
                lines.append("")
        
        if self.output_files:
            lines.append("## 输出文件列表")
            lines.append("")
            for f in self.output_files:
                lines.append(f"- `{f}`")
            lines.append("")
        
        if self.error:
            lines.append("## 错误信息")
            lines.append("")
            lines.append(f"```\n{self.error}\n```")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class TaskContext:
    """任务上下文"""
    
    session_id: str
    user_input: str
    execution_plan: Optional[str] = None
    parameter_table: Dict[str, Any] = field(default_factory=dict)
    file_paths: List[str] = field(default_factory=list)
    
    # 沙盒信息
    sandbox_id: Optional[str] = None
    sandbox_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "user_input": self.user_input,
            "execution_plan": self.execution_plan,
            "parameter_table": self.parameter_table,
            "file_paths": self.file_paths,
            "sandbox_id": self.sandbox_id,
            "sandbox_dir": self.sandbox_dir,
        }


@dataclass
class TasksMDConfig:
    """tasks.md 生成配置"""
    
    include_context: bool = True
    include_parameters: bool = True
    include_file_info: bool = True
    max_task_description_length: int = 500
    
    # 模板配置
    template_name: str = "default"
    
    @classmethod
    def default(cls) -> "TasksMDConfig":
        """获取默认配置"""
        return cls()


# MCP 配置模板
# OpenCode 官方格式参考: internal/config/config.go 中的 MCPServer 结构
# type: "sse" 或 "stdio"
DEFAULT_MCP_CONFIG = {
    "servers": {
        "nettcr": {
            "type": "sse",
            "url": "http://localhost:8080/mcp/nettcr"
        },
        "igblast": {
            "type": "sse",
            "url": "http://localhost:8080/mcp/igblast"
        },
        "metabcr": {
            "type": "sse",
            "url": "http://localhost:8080/mcp/metabcr"
        },
        "bcell": {
            "type": "sse",
            "url": "http://localhost:8080/mcp/bcell"
        },
        "tcell": {
            "type": "sse",
            "url": "http://localhost:8080/mcp/tcell"
        }
    }
}


# OpenCode 配置模板
# 参考：https://docs.bigmodel.cn/cn/coding-plan/tool/opencode
# GLM Coding Plan 使用专门的 Coding API 端点
DEFAULT_OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "zhipuai": {
            "api": "https://open.bigmodel.cn/api/coding/paas/v4"
        }
    },
    "tools": {
        "bash": True,
        "read": True,
        "write": True,
        "grep": True
    }
}


__all__ = [
    "OpenCodeMode",
    "ExecutionStatus",
    "TaskType",
    "OpenCodeConfig",
    "ExecutionResult",
    "TaskContext",
    "TasksMDConfig",
    "DEFAULT_MCP_CONFIG",
    "DEFAULT_OPENCODE_CONFIG",
]

