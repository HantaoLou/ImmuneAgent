# -*- coding: utf-8 -*-
"""
Execution Tracker - 执行过程追踪器

功能：
1. 实时追踪 OpenCode 执行过程
2. 解析 MCP 工具调用记录
3. 记录任务执行时间线
4. 生成详细的执行报告

使用示例：
    tracker = ExecutionTracker(session_id="test_001", workspace_dir="/tmp/sessions/test_001")
    
    # 开始迭代
    tracker.start_iteration(0)
    
    # 记录 MCP 调用
    tracker.record_mcp_call(
        tool_name="check_peptide_support",
        service_name="nettcr",
        parameters={"peptides": "GILGFVFTL"},
        result="支持",
        success=True
    )
    
    # 结束迭代
    iter_result = tracker.end_iteration(
        tasks_path="/tmp/sessions/test_001/tasks/tasks_v0.md",
        output_dir="/tmp/sessions/test_001/iterations/iter_0/output",
        evaluation={"quality_score": 0.85}
    )
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MCPLogEntry:
    """MCP 日志条目"""
    timestamp: str
    log_level: str
    message: str
    raw_line: str = ""


@dataclass
class TaskExecution:
    """任务执行记录"""
    task_id: str
    task_description: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "pending"
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ExecutionTracker:
    """
    执行过程追踪器
    
    追踪 OpenCode 执行过程中的各种事件，包括：
    - MCP 工具调用
    - 任务执行状态
    - 输出文件生成
    - 错误和警告
    """
    
    # MCP 调用日志匹配模式
    MCP_PATTERNS = [
        # OpenCode 标准 MCP 日志格式
        r'\[MCP\].*?(\w+_\w+|\w+).*?calling|Calling MCP tool: (\w+)',
        r'MCP.*?tool.*?:\s*["\']?(\w+)["\']?',
        r'Executing.*?(\w+_\w+).*?with params',
        # NetTCR 特定格式
        r'(check_peptide_support|validate_tcr_input|predict_tcr_binding|list_available_peptides)',
        # 通用格式
        r'Calling tool:\s*(\w+)',
        r'Tool\s+(\w+)\s+called',
        # OpenCode → mcp 格式
        r'→\s*mcp\s+(\w+)\.(\w+)',
    ]
    
    # 参数提取模式
    PARAM_PATTERNS = [
        # call_tool(parameters={...}) 格式
        r'parameters\s*=\s*(\{[^}]*\})',
        r'parameters\s*=\s*(\[[^\]]*\])',
        # call_tool(..., {"key": "value"}) 格式
        r'call_tool\s*\([^)]*parameters\s*=\s*(\{[^}]*\})',
        # 参数行格式: key=value 或 key: value
        r'(?:peptide|peptides|tcr|tcrs|hla|threshold|file|path)\s*[=:]\s*["\']?([^\s,"\']+)["\']?',
        # JSON 参数格式
        r'arguments\s*=\s*(\{[^}]*\})',
        r'args\s*=\s*(\{[^}]*\})',
        # OpenCode 参数行
        r'param(?:eter)?s?\s*[=:]\s*(\{[^}]*\})',
    ]
    
    def __init__(
        self,
        session_id: str,
        workspace_dir: str,
        enable_file_logging: bool = True,
    ):
        """
        初始化追踪器
        
        Args:
            session_id: 会话 ID
            workspace_dir: 工作空间目录
            enable_file_logging: 是否启用文件日志
        """
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.enable_file_logging = enable_file_logging
        
        # 当前迭代状态
        self._current_iteration: int = -1
        self._iteration_start_time: Optional[datetime] = None
        self._iteration_logs: List[str] = []
        self._iteration_mcp_calls: List[Dict[str, Any]] = []
        self._iteration_tasks: Dict[str, TaskExecution] = {}
        
        # 全局记录
        self._all_mcp_calls: List[Dict[str, Any]] = []
        self._all_logs: List[str] = []
        
        # 文件日志路径
        self._log_file: Optional[Path] = None
        if enable_file_logging:
            log_dir = Path(workspace_dir) / "reports"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / "execution_trace.log"
    
    # ========================================================================
    # 迭代追踪
    # ========================================================================
    
    def start_iteration(self, iteration: int) -> None:
        """
        开始追踪一个新迭代
        
        Args:
            iteration: 迭代编号
        """
        self._current_iteration = iteration
        self._iteration_start_time = datetime.now()
        self._iteration_logs = []
        self._iteration_mcp_calls = []
        self._iteration_tasks = {}
        
        self._log(f"=== 迭代 {iteration} 开始 ===")
        self._log(f"开始时间: {self._iteration_start_time.isoformat()}")
    
    def end_iteration(
        self,
        tasks_path: str,
        output_dir: str,
        evaluation: Dict[str, Any],
        opencode_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        结束当前迭代并生成记录
        
        Args:
            tasks_path: tasks.md 文件路径
            output_dir: 输出目录
            evaluation: 评估结果
            opencode_result: OpenCode 执行结果 (ExecutionResult)
            
        Returns:
            迭代记录字典
        """
        end_time = datetime.now()
        duration_ms = int((end_time - self._iteration_start_time).total_seconds() * 1000) if self._iteration_start_time else 0
        
        # 收集 OpenCode 日志
        opencode_log = ""
        environment_info = {}
        
        if opencode_result:
            opencode_log = getattr(opencode_result, 'stdout', '') or ""
            if hasattr(opencode_result, 'sandbox_id'):
                environment_info['sandbox_id'] = opencode_result.sandbox_id
            if hasattr(opencode_result, 'execution_time_ms'):
                environment_info['opencode_execution_time_ms'] = opencode_result.execution_time_ms
        
        # 读取 tasks.md 内容
        tasks_md_content = ""
        try:
            if tasks_path and Path(tasks_path).exists():
                tasks_md_content = Path(tasks_path).read_text(encoding='utf-8')
        except Exception:
            pass
        
        # 构建迭代记录
        iteration_record = {
            "iteration": self._current_iteration,
            "start_time": self._iteration_start_time.isoformat() if self._iteration_start_time else None,
            "end_time": end_time.isoformat(),
            "duration_ms": duration_ms,
            "tasks_md_path": tasks_path,
            "tasks_md_content": tasks_md_content,
            "output_dir": output_dir,
            "evaluation": evaluation,
            "opencode_log": opencode_log,
            "mcp_calls": self._iteration_mcp_calls.copy(),
            "task_timeline": [
                {
                    "task_id": task_id,
                    "task_description": task.task_description,
                    "start_time": task.start_time,
                    "end_time": task.end_time,
                    "status": task.status,
                    "output_files": task.output_files,
                    "errors": task.errors,
                }
                for task_id, task in self._iteration_tasks.items()
            ],
            "environment_info": environment_info,
            "logs": self._iteration_logs.copy(),
        }
        
        # 保存到全局记录
        self._all_mcp_calls.extend(self._iteration_mcp_calls)
        self._all_logs.extend(self._iteration_logs)
        
        self._log(f"=== 迭代 {self._current_iteration} 结束 ===")
        self._log(f"结束时间: {end_time.isoformat()}")
        self._log(f"持续时长: {duration_ms}ms")
        self._log(f"MCP 调用次数: {len(self._iteration_mcp_calls)}")
        
        return iteration_record
    
    # ========================================================================
    # MCP 调用追踪
    # ========================================================================
    
    def record_mcp_call(
        self,
        tool_name: str,
        service_name: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        result: str = "",
        success: bool = True,
        error: str = "",
        duration_ms: int = 0,
    ) -> None:
        """
        记录一次 MCP 工具调用
        
        Args:
            tool_name: 工具名称
            service_name: 服务名称 (如 nettcr, igblast)
            parameters: 调用参数
            result: 返回结果 (预览)
            success: 是否成功
            error: 错误信息
            duration_ms: 执行时长
        """
        call_record = {
            "tool_name": tool_name,
            "service_name": service_name,
            "parameters": parameters or {},
            "result_preview": result[:500] if result else "",  # 截断
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "iteration": self._current_iteration,
        }
        
        self._iteration_mcp_calls.append(call_record)
        self._log(f"MCP 调用: {service_name}.{tool_name} -> {'成功' if success else '失败'}")
    
    def parse_mcp_calls_from_log(self, log_content: str) -> List[Dict[str, Any]]:
        """
        从 OpenCode 日志中解析 MCP 调用记录
        
        Args:
            log_content: 日志内容
            
        Returns:
            解析出的 MCP 调用列表
        """
        calls = []
        lines = log_content.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in self.MCP_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    tool_name = match.group(1) or match.group(2) or ""
                    if not tool_name:
                        # OpenCode → mcp service.tool 格式
                        tool_name = match.group(2) if match.lastindex >= 2 else ""
                    
                    if tool_name:
                        # 尝试提取服务名
                        service_name = ""
                        if '_' in tool_name:
                            parts = tool_name.split('_')
                            service_name = parts[0]
                        elif match.lastindex and match.lastindex >= 2:
                            service_name = match.group(1)  # service name
                        
                        # 尝试提取参数（关键增强）
                        parameters = self._extract_parameters_from_context(lines, i, line)
                        
                        # 尝试提取结果预览
                        result_preview = self._extract_result_preview(lines, i)
                        
                        # 检查是否已记录（基于工具名 + 参数）
                        param_key = json.dumps(parameters, sort_keys=True)
                        existing = [c for c in calls if c['tool_name'] == tool_name and json.dumps(c['parameters'], sort_keys=True) == param_key]
                        if not existing:
                            calls.append({
                                "tool_name": tool_name,
                                "service_name": service_name,
                                "parameters": parameters,
                                "result_preview": result_preview,
                                "success": "error" not in line.lower() and "fail" not in line.lower(),
                                "error": "",
                                "timestamp": datetime.now().isoformat(),
                                "duration_ms": 0,
                                "iteration": self._current_iteration,
                                "log_line": line.strip(),
                            })
                        break
        
        return calls
    
    def _extract_parameters_from_context(self, lines: List[str], line_idx: int, current_line: str) -> Dict[str, Any]:
        """
        从日志上下文中提取参数
        
        检查当前行及前后几行，尝试解析参数
        """
        parameters = {}
        
        # 1. 先检查当前行
        params = self._try_extract_parameters(current_line)
        if params:
            return params
        
        # 2. 检查前 2 行到后 5 行的范围
        context_start = max(0, line_idx - 2)
        context_end = min(len(lines), line_idx + 6)
        
        context_text = '\n'.join(lines[context_start:context_end])
        params = self._try_extract_parameters(context_text)
        if params:
            return params
        
        # 3. 尝试从多行 JSON 块中提取
        json_params = self._try_extract_multiline_json(lines, line_idx)
        if json_params:
            return json_params
        
        return parameters
    
    def _try_extract_parameters(self, text: str) -> Dict[str, Any]:
        """尝试从文本中提取参数"""
        for pattern in self.PARAM_PATTERNS:
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    param_str = match.group(1)
                    # 尝试解析 JSON
                    try:
                        parsed = json.loads(param_str)
                        if isinstance(parsed, dict):
                            return parsed
                        elif isinstance(parsed, list):
                            return {"items": parsed}
                    except json.JSONDecodeError:
                        # 不是 JSON，尝试解析 key=value 格式
                        return self._parse_key_value_params(param_str)
            except Exception:
                continue
        
        # 尝试提取常见的键值对
        kv_pattern = r'(\w+)\s*[=:]\s*["\']?([^,"\s\']+)["\']?'
        matches = re.findall(kv_pattern, text)
        if matches:
            return {k: v for k, v in matches if k in ['peptide', 'peptides', 'tcr', 'tcrs', 'hla', 'threshold', 'file', 'path']}
        
        return {}
    
    def _try_extract_multiline_json(self, lines: List[str], start_idx: int) -> Dict[str, Any]:
        """尝试从多行文本中提取完整 JSON"""
        # 向后查找 JSON 块
        json_buffer = ""
        brace_count = 0
        in_json = False
        
        for i in range(start_idx, min(len(lines), start_idx + 20)):
            line = lines[i]
            for char in line:
                if char == '{':
                    in_json = True
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                
                if in_json:
                    json_buffer += char
                
                if in_json and brace_count == 0:
                    try:
                        parsed = json.loads(json_buffer)
                        if isinstance(parsed, dict):
                            # 过滤掉非参数字段
                            param_keys = ['peptide', 'peptides', 'tcr', 'tcrs', 'hla', 
                                         'threshold', 'file', 'path', 'sequence', 
                                         'sequences', 'cdr3a', 'cdr3b', 'input_file']
                            return {k: v for k, v in parsed.items() 
                                   if k.lower() in param_keys or any(pk in k.lower() for pk in param_keys)}
                    except json.JSONDecodeError:
                        pass
                    break
            
            if in_json and brace_count == 0:
                break
        
        return {}
    
    def _parse_key_value_params(self, text: str) -> Dict[str, Any]:
        """解析 key=value 或 key: value 格式的参数"""
        params = {}
        # 匹配 key=value 或 key: value，值可以带引号或不带
        patterns = [
            r'(\w+)\s*=\s*"([^"]*)"',
            r"(\w+)\s*=\s*'([^']*)'",
            r'(\w+)\s*=\s*([^\s,\)]+)',
            r'(\w+)\s*:\s*"([^"]*)"',
            r"(\w+)\s*:\s*'([^']*)'",
            r'(\w+)\s*:\s*([^\s,\)]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                if key.lower() not in ['tool', 'name', 'service', 'call']:  # 过滤非参数字段
                    params[key] = value
        
        return params
    
    def _extract_result_preview(self, lines: List[str], line_idx: int) -> str:
        """从日志中提取结果预览"""
        # 检查当前行及后 3 行
        context = '\n'.join(lines[line_idx:line_idx + 4])
        
        # 匹配结果模式
        result_patterns = [
            r'result\s*[:=]\s*(\{[^}]*\})',
            r'response\s*[:=]\s*(\{[^}]*\})',
            r'output\s*[:=]\s*(\{[^}]*\})',
            r'→\s*(\{[^}]*\})',  # OpenCode 响应格式
            r'(?:supported|success|result|score)\s*[:=]\s*(\w+)',
        ]
        
        for pattern in result_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                result = match.group(1)
                return result[:500] if len(result) > 500 else result
        
        return ""
    
    # ========================================================================
    # 任务追踪
    # ========================================================================
    
    def start_task(self, task_id: str, description: str) -> None:
        """
        开始一个任务
        
        Args:
            task_id: 任务 ID
            description: 任务描述
        """
        self._iteration_tasks[task_id] = TaskExecution(
            task_id=task_id,
            task_description=description,
            start_time=datetime.now().isoformat(),
            status="running",
        )
        self._log(f"任务开始: [{task_id}] {description[:50]}...")
    
    def end_task(
        self,
        task_id: str,
        status: str = "completed",
        output_files: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
    ) -> None:
        """
        结束一个任务
        
        Args:
            task_id: 任务 ID
            status: 任务状态 (completed, failed, skipped)
            output_files: 输出文件列表
            errors: 错误列表
        """
        if task_id in self._iteration_tasks:
            task = self._iteration_tasks[task_id]
            task.end_time = datetime.now().isoformat()
            task.status = status
            task.output_files = output_files or []
            task.errors = errors or []
            
            self._log(f"任务结束: [{task_id}] 状态={status}, 输出={len(task.output_files)}个文件")
    
    # ========================================================================
    # 日志记录
    # ========================================================================
    
    def _log(self, message: str) -> None:
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {message}"
        
        self._iteration_logs.append(log_line)
        
        if self._log_file:
            try:
                with open(self._log_file, 'a', encoding='utf-8') as f:
                    f.write(log_line + '\n')
            except Exception:
                pass
    
    def add_log(self, message: str) -> None:
        """添加自定义日志"""
        self._log(message)
    
    def add_opencode_log(self, log_content: str) -> None:
        """
        添加 OpenCode 执行日志并自动解析
        
        Args:
            log_content: OpenCode 执行日志
        """
        # 添加日志
        for line in log_content.split('\n'):
            if line.strip():
                self._iteration_logs.append(f"[OpenCode] {line}")
        
        # 解析 MCP 调用
        parsed_calls = self.parse_mcp_calls_from_log(log_content)
        
        # 合并到当前迭代（去重 - 基于 tool_name + parameters）
        existing_keys = {
            (c['tool_name'], json.dumps(c.get('parameters', {}), sort_keys=True))
            for c in self._iteration_mcp_calls
        }
        for call in parsed_calls:
            call_key = (
                call['tool_name'],
                json.dumps(call.get('parameters', {}), sort_keys=True)
            )
            if call_key not in existing_keys:
                self._iteration_mcp_calls.append(call)
                existing_keys.add(call_key)
    
    # ========================================================================
    # 报告生成
    # ========================================================================
    
    def generate_iteration_report(self, iteration: int) -> Dict[str, Any]:
        """
        生成单个迭代的详细报告
        
        Args:
            iteration: 迭代编号
            
        Returns:
            迭代报告字典
        """
        return {
            "session_id": self.session_id,
            "iteration": iteration,
            "generated_at": datetime.now().isoformat(),
            "mcp_calls_summary": {
                "total": len(self._iteration_mcp_calls),
                "successful": len([c for c in self._iteration_mcp_calls if c['success']]),
                "failed": len([c for c in self._iteration_mcp_calls if not c['success']]),
                "tools_used": list({c['tool_name'] for c in self._iteration_mcp_calls}),
            },
            "task_summary": {
                "total": len(self._iteration_tasks),
                "completed": len([t for t in self._iteration_tasks.values() if t.status == "completed"]),
                "failed": len([t for t in self._iteration_tasks.values() if t.status == "failed"]),
            },
            "mcp_calls": self._iteration_mcp_calls,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "description": t.task_description,
                    "status": t.status,
                    "output_files": t.output_files,
                    "errors": t.errors,
                }
                for t in self._iteration_tasks.values()
            ],
        }
    
    def generate_full_report(self) -> Dict[str, Any]:
        """
        生成完整的执行报告
        
        Returns:
            完整报告字典
        """
        return {
            "session_id": self.session_id,
            "workspace_dir": self.workspace_dir,
            "generated_at": datetime.now().isoformat(),
            "total_mcp_calls": len(self._all_mcp_calls),
            "all_mcp_calls": self._all_mcp_calls,
            "full_log": '\n'.join(self._all_logs),
        }
    
    def save_report(self, filepath: Optional[str] = None) -> str:
        """
        保存报告到文件
        
        Args:
            filepath: 目标文件路径，默认为 {workspace}/reports/execution_trace.json
            
        Returns:
            保存的文件路径
        """
        if not filepath:
            filepath = str(Path(self.workspace_dir) / "reports" / "execution_trace.json")
        
        report = self.generate_full_report()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath


__all__ = [
    "ExecutionTracker",
    "MCPLogEntry",
    "TaskExecution",
]

