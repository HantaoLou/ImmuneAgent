# -*- coding: utf-8 -*-
"""
Iterative OpenCode Executor - 迭代式任务执行器

核心特性：
1. 接收 JSON 数据作为输入
2. 自动根据输入生成 tasks.md
3. 迭代执行 + 评估 + 优化
4. 支持配置最大迭代次数

流程：
1. 接收 JSON 数据
2. 创建沙盒、准备环境
3. 利用 OpenCode 分析 JSON 数据，生成初始 tasks.md
4. 执行 tasks.md
5. 评估输出
6. 根据评估优化 tasks.md
7. 重复 4-6 直到达到最大迭代次数或评估通过
8. 生成最终总结

目录结构：
/data/sessions/{session_id}/
├── input/                      # 输入数据
├── output/                     # 最终输出（最后一次迭代）
├── tasks/                      # 任务迭代历史
│   ├── tasks_v0.md            # 初始生成的任务
│   ├── tasks_v1.md            # 第1次优化后
│   └── ...
├── iterations/                 # 迭代记录
│   ├── iter_0/
│   │   ├── output/            # 第0次执行输出
│   │   ├── evaluation.json    # 第0次评估报告
│   │   └── opencode.log       # 执行日志
│   └── ...
├── .agent/
│   ├── context.json           # 输入上下文
│   └── iteration_state.json   # 迭代状态跟踪
└── reports/
    ├── final_summary.json     # 最终总结
    └── final_report.md        # 人类可读报告
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from coding_agent.config import (
    ExecutionResult,
    ExecutionStatus,
    OpenCodeConfig,
    OpenCodeMode,
)
from coding_agent.opencode_executor import OpenCodeExecutor
from coding_agent.execution_tracker import ExecutionTracker


# ============================================================================
# 数据结构定义
# ============================================================================


class IterationStatus(str, Enum):
    """迭代状态"""

    SUCCESS = "success"  # 成功，无需继续优化
    NEEDS_IMPROVEMENT = "needs_improvement"  # 需要改进
    FAILED = "failed"  # 失败


class EvaluationLevel(str, Enum):
    """评估等级"""

    EXCELLENT = "excellent"  # 优秀 (90-100%)
    GOOD = "good"  # 良好 (70-89%)
    ACCEPTABLE = "acceptable"  # 可接受 (60-69%)
    POOR = "poor"  # 较差 (40-59%)
    FAILED = "failed"  # 失败 (0-39%)


@dataclass
class EvaluationCriteria:
    """评估标准配置"""

    # 必需的输出文件列表
    required_output_files: List[str] = field(default_factory=list)

    # 质量阈值
    min_quality_score: float = 0.6  # 最低质量分数

    # 格式验证规则
    format_validators: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 是否在成功时提前退出
    early_stop_on_success: bool = True

    @classmethod
    def default(cls) -> "EvaluationCriteria":
        """获取默认评估标准"""
        return cls(
            required_output_files=[],
            min_quality_score=0.6,
            format_validators={},
            early_stop_on_success=True,
        )


@dataclass
class MCPCallRecord:
    """MCP 工具调用记录"""

    # 工具名称
    tool_name: str

    # 服务名称 (如 nettcr, igblast)
    service_name: str = ""

    # 调用参数
    parameters: Dict[str, Any] = field(default_factory=dict)

    # 返回结果 (截断)
    result_preview: str = ""

    # 是否成功
    success: bool = True

    # 错误信息
    error: str = ""

    # 调用时间戳
    timestamp: str = ""

    # 执行时长 (毫秒)
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "service_name": self.service_name,
            "parameters": self.parameters,
            "result_preview": self.result_preview,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TaskTimelineEntry:
    """任务执行时间线条目"""

    # 任务 ID
    task_id: str

    # 任务描述 (简短)
    task_summary: str

    # 开始时间
    start_time: str = ""

    # 结束时间
    end_time: str = ""

    # 状态
    status: str = "pending"  # pending, running, completed, failed

    # 输出文件
    output_files: List[str] = field(default_factory=list)

    # 错误信息
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_summary": self.task_summary,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "output_files": self.output_files,
            "error": self.error,
        }


@dataclass
class IterationResult:
    """单次迭代结果"""

    # 迭代编号 (从 0 开始)
    iteration: int

    # 本次迭代使用的 tasks.md 路径
    tasks_md_path: str

    # 输出目录
    output_dir: str

    # 评估报告
    evaluation_report: Dict[str, Any] = field(default_factory=dict)

    # 迭代状态
    status: IterationStatus = IterationStatus.NEEDS_IMPROVEMENT

    # 错误信息
    errors: List[str] = field(default_factory=list)

    # 改进建议
    improvement_suggestions: List[str] = field(default_factory=list)

    # 质量分数 (0.0 - 1.0)
    quality_score: float = 0.0

    # 执行时间 (毫秒)
    execution_time_ms: int = 0

    # 生成的输出文件
    output_files: List[str] = field(default_factory=list)

    # ========== 新增字段：详细执行记录 ==========

    # OpenCode 完整日志
    opencode_log: str = ""

    # MCP 工具调用记录
    mcp_calls: List[MCPCallRecord] = field(default_factory=list)

    # 任务执行时间线
    task_timeline: List[TaskTimelineEntry] = field(default_factory=list)

    # 生成的 tasks.md 内容
    tasks_md_content: str = ""

    # 评估详情 (原始 JSON)
    evaluation_details: Dict[str, Any] = field(default_factory=dict)

    # 执行环境信息
    environment_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "iteration": self.iteration,
            "tasks_md_path": self.tasks_md_path,
            "output_dir": self.output_dir,
            "evaluation_report": self.evaluation_report,
            "status": self.status.value,
            "errors": self.errors,
            "improvement_suggestions": self.improvement_suggestions,
            "quality_score": self.quality_score,
            "execution_time_ms": self.execution_time_ms,
            "output_files": self.output_files,
            # 新增字段
            "opencode_log": self.opencode_log,
            "mcp_calls": [m.to_dict() for m in self.mcp_calls],
            "task_timeline": [t.to_dict() for t in self.task_timeline],
            "tasks_md_content": self.tasks_md_content,
            "evaluation_details": self.evaluation_details,
            "environment_info": self.environment_info,
        }


@dataclass
class IterativeExecutionResult:
    """完整迭代执行结果"""

    # 会话 ID
    session_id: str

    # 工作空间目录
    workspace_dir: str

    # 总迭代次数
    total_iterations: int

    # 最终状态
    final_status: IterationStatus

    # 最终输出目录
    final_output_dir: str

    # 最终输出文件列表
    final_output_files: List[str]

    # 迭代历史
    iteration_history: List[IterationResult] = field(default_factory=list)

    # 最终总结
    final_summary: Dict[str, Any] = field(default_factory=dict)

    # 最终报告路径
    final_report_path: str = ""

    # 沙盒 ID
    sandbox_id: Optional[str] = None

    # 总执行时间 (毫秒)
    total_execution_time_ms: int = 0

    # ========== 新增字段：全局执行记录 ==========

    # 完整执行日志 (所有迭代合并)
    full_execution_log: str = ""

    # 所有 MCP 调用汇总
    all_mcp_calls: List[MCPCallRecord] = field(default_factory=list)

    # 配置信息
    config_info: Dict[str, Any] = field(default_factory=dict)

    # 执行统计
    execution_stats: Dict[str, Any] = field(default_factory=dict)

    # 详细报告路径 (JSON 格式)
    detailed_report_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "workspace_dir": self.workspace_dir,
            "total_iterations": self.total_iterations,
            "final_status": self.final_status.value,
            "final_output_dir": self.final_output_dir,
            "final_output_files": self.final_output_files,
            "iteration_history": [r.to_dict() for r in self.iteration_history],
            "final_summary": self.final_summary,
            "final_report_path": self.final_report_path,
            "sandbox_id": self.sandbox_id,
            "total_execution_time_ms": self.total_execution_time_ms,
            # 新增字段
            "full_execution_log": self.full_execution_log,
            "all_mcp_calls": [m.to_dict() for m in self.all_mcp_calls],
            "config_info": self.config_info,
            "execution_stats": self.execution_stats,
            "detailed_report_path": self.detailed_report_path,
        }

    def is_success(self) -> bool:
        """检查是否执行成功"""
        return self.final_status == IterationStatus.SUCCESS


# ============================================================================
# 迭代执行器
# ============================================================================


class IterativeOpenCodeExecutor:
    """
    迭代式 OpenCode 执行器

    核心特性:
    1. 自动根据输入生成 tasks.md
    2. 迭代执行 + 评估 + 优化
    3. 可配置迭代次数和评估标准

    使用示例:
        executor = IterativeOpenCodeExecutor(
            config=OpenCodeConfig(model_provider="glm-5"),
            max_iterations=3,
        )

        input_data = {
            "session_id": "test_001",
            "input_files": ["/data/input.csv"],
            "params": {"threshold": 0.6},
        }

        result = await executor.execute(input_data)
        print(f"总迭代次数: {result.total_iterations}")
        print(f"最终状态: {result.final_status}")
    """

    def __init__(
        self,
        config: Optional[OpenCodeConfig] = None,
        max_iterations: int = 3,
        evaluation_criteria: Optional[EvaluationCriteria] = None,
        early_stop_on_success: bool = True,
        enable_tracking: bool = True,
        progress_callback: Optional[callable] = None,
    ):
        """
        初始化迭代执行器

        Args:
            config: OpenCode 配置
            max_iterations: 最大迭代次数（默认 3）
            evaluation_criteria: 评估标准
            early_stop_on_success: 是否在成功时提前退出
            enable_tracking: 是否启用详细执行追踪
            progress_callback: SSE 进度回调函数，用于实时推送执行状态
        """
        self.config = config or OpenCodeConfig.from_env()
        self.max_iterations = max_iterations
        self.evaluation_criteria = evaluation_criteria or EvaluationCriteria.default()
        self.early_stop_on_success = early_stop_on_success
        self.enable_tracking = enable_tracking
        self.progress_callback = progress_callback

        # 内部状态
        self._executor: Optional[OpenCodeExecutor] = None
        self._sandbox: Optional[Any] = None
        self._session_id: str = ""
        self._workspace: str = ""

        # 执行追踪器
        self._tracker: Optional[ExecutionTracker] = None

    def _report_progress(
        self, event_type: str, message: str, details: Optional[Dict] = None
    ):
        """
        通过 progress_callback 报告执行进度

        Args:
            event_type: 事件类型 (sandbox_exec, iteration_start, task_complete 等)
            message: 进度消息
            details: 详细信息
        """
        if self.progress_callback:
            try:
                self.progress_callback(
                    event_type=event_type,
                    message=message,
                    details=details or {},
                )
            except Exception as e:
                print(f"[IterativeExecutor] Error reporting progress: {e}")

    async def execute(
        self,
        input_data: Dict[str, Any],
    ) -> IterativeExecutionResult:
        """
        执行完整迭代流程

        Args:
            input_data: 包含 session_id, input_files, params 等的 JSON 数据

        Returns:
            IterativeExecutionResult: 完整执行结果
        """
        start_time = time.time()

        # 提取或生成 session_id
        self._session_id = input_data.get("session_id") or self._generate_session_id()
        self._workspace = f"/data/sessions/{self._session_id}"

        # 初始化执行追踪器
        if self.enable_tracking:
            self._tracker = ExecutionTracker(
                session_id=self._session_id,
                workspace_dir=self._workspace,
                enable_file_logging=True,
            )

        self._log(f"开始迭代执行: session_id={self._session_id}")
        self._log(f"最大迭代次数: {self.max_iterations}")

        # 报告开始执行
        self._report_progress(
            event_type="sandbox_exec",
            message="[START] 开始沙盒代码执行",
            details={
                "phase": "start",
                "session_id": self._session_id,
                "max_iterations": self.max_iterations,
                "workspace": self._workspace,
            },
        )

        try:
            # Step 2: 环境准备
            self._report_progress(
                event_type="sandbox_exec",
                message="📦 准备沙盒环境...",
                details={"phase": "environment_prepare"},
            )
            await self._prepare_environment(input_data)
            self._report_progress(
                event_type="sandbox_exec",
                message="[SUCCESS] 沙盒环境准备完成",
                details={"phase": "environment_ready"},
            )

            # Step 3: 生成初始 tasks.md
            self._report_progress(
                event_type="sandbox_exec",
                message="📝 生成任务列表...",
                details={"phase": "tasks_generation"},
            )
            tasks_v0 = await self._generate_initial_tasks(input_data)
            self._report_progress(
                event_type="sandbox_exec",
                message="[SUCCESS] 任务列表生成完成",
                details={"phase": "tasks_ready"},
            )

            iteration_history: List[IterationResult] = []
            current_tasks = tasks_v0

            # 迭代循环
            for i in range(self.max_iterations):
                self._log(f"\n{'=' * 60}")
                self._log(f"迭代 {i + 1}/{self.max_iterations}")
                self._log(f"{'=' * 60}")

                # 报告迭代开始
                self._report_progress(
                    event_type="iteration_start",
                    message=f"[RUN] 迭代 {i + 1}/{self.max_iterations}",
                    details={
                        "phase": "iteration_start",
                        "iteration": i + 1,
                        "max_iterations": self.max_iterations,
                    },
                )

                # 开始追踪当前迭代
                if self._tracker:
                    self._tracker.start_iteration(i)

                # Step 4: 执行 tasks.md
                execution_result = await self._execute_tasks_iteration(current_tasks, i)

                # Step 5: 评估输出
                self._report_progress(
                    event_type="sandbox_exec",
                    message=f"[STAT] 评估迭代 {i + 1} 的输出...",
                    details={"phase": "evaluation", "iteration": i + 1},
                )
                evaluation = await self._evaluate_outputs(i)

                # 报告评估结果
                self._report_progress(
                    event_type="sandbox_exec",
                    message=f"[SUCCESS] 评估完成: 得分 {evaluation.get('quality_score', 0):.1f}/1.0",
                    details={
                        "phase": "evaluation_complete",
                        "iteration": i + 1,
                        "quality_score": evaluation.get("quality_score", 0),
                    },
                )

                # 读取 tasks.md 内容
                tasks_md_content = ""
                try:
                    tasks_md_content = await self._sandbox.files.read_file(
                        current_tasks
                    )
                except Exception:
                    pass

                # 读取 OpenCode 日志
                opencode_log = ""
                try:
                    opencode_log_path = (
                        f"{self._workspace}/iterations/iter_{i}/opencode.log"
                    )
                    opencode_log = await self._sandbox.files.read_file(
                        opencode_log_path
                    )
                except Exception:
                    pass

                # 解析 MCP 调用
                mcp_calls: List[MCPCallRecord] = []
                if self._tracker and opencode_log:
                    parsed_calls = self._tracker.parse_mcp_calls_from_log(opencode_log)
                    for call in parsed_calls:
                        mcp_calls.append(
                            MCPCallRecord(
                                tool_name=call.get("tool_name", ""),
                                service_name=call.get("service_name", ""),
                                parameters=call.get("parameters", {}),
                                result_preview=call.get("result_preview", ""),
                                success=call.get("success", True),
                                error=call.get("error", ""),
                                timestamp=call.get("timestamp", ""),
                                duration_ms=call.get("duration_ms", 0),
                            )
                        )
                    self._tracker.add_opencode_log(opencode_log)

                # 构建迭代结果
                iter_result = IterationResult(
                    iteration=i,
                    tasks_md_path=current_tasks,
                    output_dir=f"{self._workspace}/iterations/iter_{i}/output",
                    evaluation_report=evaluation,
                    status=self._determine_status(evaluation),
                    errors=evaluation.get("errors", []),
                    improvement_suggestions=evaluation.get("suggestions", []),
                    quality_score=evaluation.get("quality_score", 0.0),
                    execution_time_ms=execution_result.execution_time_ms
                    if execution_result
                    else 0,
                    output_files=evaluation.get("files_generated", []),
                    # 新增字段
                    opencode_log=opencode_log,
                    mcp_calls=mcp_calls,
                    tasks_md_content=tasks_md_content,
                    evaluation_details=evaluation,
                    environment_info={
                        "sandbox_id": getattr(self._sandbox, "id", None)
                        if self._sandbox
                        else None,
                        "iteration": i,
                    },
                )
                iteration_history.append(iter_result)

                self._log(
                    f"迭代 {i} 评估结果: status={iter_result.status.value}, quality={iter_result.quality_score:.2f}"
                )

                # 检查是否成功
                if (
                    iter_result.status == IterationStatus.SUCCESS
                    and self.early_stop_on_success
                ):
                    self._log("评估通过，提前退出迭代")
                    break

                # Step 6: 优化 tasks.md (如果不是最后一次迭代)
                if i < self.max_iterations - 1:
                    self._log(f"生成优化后的 tasks_v{i + 1}.md...")
                    current_tasks = await self._optimize_tasks(
                        current_tasks, evaluation, i
                    )

            # Step 7: 生成最终总结
            final_result = await self._generate_final_summary(iteration_history)

            # 设置总执行时间
            final_result.total_execution_time_ms = int(
                (time.time() - start_time) * 1000
            )

            self._log(
                f"\n迭代执行完成: 总迭代次数={final_result.total_iterations}, 最终状态={final_result.final_status.value}"
            )

            return final_result

        except Exception as e:
            self._log(f"迭代执行失败: {e}")
            return IterativeExecutionResult(
                session_id=self._session_id,
                workspace_dir=self._workspace,
                total_iterations=0,
                final_status=IterationStatus.FAILED,
                final_output_dir="",
                final_output_files=[],
                final_summary={"error": str(e)},
            )

        finally:
            # 清理沙盒
            await self._cleanup()

    # ========================================================================
    # 内部方法 - 将在后续 Phase 中实现
    # ========================================================================

    async def _prepare_environment(self, input_data: Dict[str, Any]) -> None:
        """
        Step 2: 环境准备 - 创建沙盒和目录结构

        创建的目录结构:
        /data/sessions/{session_id}/
        ├── input/          # 输入数据
        ├── output/         # 最终输出
        ├── tasks/          # 任务迭代历史
        ├── iterations/     # 迭代记录
        ├── .agent/         # 上下文和状态
        └── reports/        # 报告文件

        Args:
            input_data: 输入数据，包含 input_files, params 等
        """
        self._log("Step 2: 准备环境...")

        # 2.1 创建执行器和沙盒（传递 progress_callback）
        self._executor = OpenCodeExecutor(
            self.config, progress_callback=self.progress_callback
        )
        self._sandbox = await self._executor.create_sandbox()
        self._log(f"  沙盒已创建: {getattr(self._sandbox, 'id', 'N/A')}")

        # 2.2 创建目录结构
        dirs_to_create = [
            f"{self._workspace}/input",
            f"{self._workspace}/output",
            f"{self._workspace}/tasks",
            f"{self._workspace}/iterations",
            f"{self._workspace}/.agent",
            f"{self._workspace}/reports",
        ]

        mkdir_cmd = f"mkdir -p {' '.join(dirs_to_create)}"
        await self._sandbox.commands.run(mkdir_cmd)
        self._log(f"  目录结构已创建: {self._workspace}")

        # 验证目录是否创建成功
        verify_result = await self._sandbox.commands.run(f"ls -la {self._workspace}/")
        self._log(f"  目录验证: {self._get_stdout(verify_result)}")

        # 2.3 处理输入文件
        input_files = input_data.get("input_files", [])
        if input_files:
            self._log(f"  处理 {len(input_files)} 个输入文件...")

            for file_path in input_files:
                # 判断是远程文件还是本地文件
                if file_path.startswith("/data/") or file_path.startswith("/tmp/"):
                    # 远程文件（沙盒服务器上的共享目录）- 复制
                    filename = Path(file_path).name
                    dest_path = f"{self._workspace}/input/{filename}"
                    await self._sandbox.commands.run(f"cp {file_path} {dest_path}")
                    self._log(f"    复制远程文件: {file_path} -> {dest_path}")
                else:
                    # 本地文件 - 需要上传
                    # TODO: 支持本地文件上传
                    self._log(f"    警告: 本地文件上传暂未实现: {file_path}")

        # 2.4 写入 context.json
        context_data = {
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "input_files": input_files,
            "params": input_data.get("params", {}),
            "user_input": input_data.get("user_input", ""),
            "task_type": input_data.get("task_type", "general"),
            "mcp_tools": input_data.get("mcp_tools", []),
        }

        context_json = json.dumps(context_data, indent=2, ensure_ascii=False)
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/context.json", context_json
        )
        self._log(f"  上下文已写入: {self._workspace}/.agent/context.json")

        # 2.5 初始化迭代状态文件
        iteration_state = {
            "session_id": self._session_id,
            "current_iteration": 0,
            "max_iterations": self.max_iterations,
            "started_at": datetime.now().isoformat(),
            "status": "preparing",
        }
        state_json = json.dumps(iteration_state, indent=2, ensure_ascii=False)
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/iteration_state.json", state_json
        )
        self._log("  环境准备完成")

    async def _generate_initial_tasks(self, input_data: Dict[str, Any]) -> str:
        """
        Step 3: 生成初始 tasks.md

        利用 OpenCode 分析输入数据，自动生成 tasks.md

        Args:
            input_data: 输入数据

        Returns:
            生成的 tasks.md 文件路径
        """
        self._log("Step 3: 生成初始 tasks.md...")

        tasks_v0_path = f"{self._workspace}/tasks/tasks_v0.md"

        # 确保 tasks 目录存在
        await self._sandbox.commands.run(f"mkdir -p {self._workspace}/tasks")

        # 构建生成 tasks.md 的 Prompt
        generation_prompt = f"""你是一个科学计算任务规划专家。

## 任务
分析输入数据和上下文，生成一个详细的任务执行计划 (tasks.md)。

## 输入信息

### 上下文文件
位置: {self._workspace}/.agent/context.json

### 输入文件目录
位置: {self._workspace}/input/

请先执行以下命令查看输入文件:
```bash
ls -la {self._workspace}/input/
```

然后读取上下文文件:
```bash
cat {self._workspace}/.agent/context.json
```

## tasks.md 格式要求

生成的 tasks.md 应包含以下结构:

```markdown
# 任务标题

> 会话 ID: `{self._session_id}`
> 生成时间: [当前时间]

## 背景
[描述任务背景和目标]

## 输入数据
[列出输入文件及其用途]

## Phase 1: [阶段名称]
### 任务 1: [任务名称]
- **任务 ID**: task_001
- **类型**: DATA_PROCESSING | ANALYSIS | MCP_TOOL_TEST | ...
- **优先级**: 高 | 中 | 低
- **描述**: [详细描述]

## Phase 2: ...
...

## 输出要求
[说明最终需要生成哪些文件]
```

## 注意事项

1. **MCP 工具**: 如果需要调用 MCP 工具（如 nettcr、igblast），请在任务描述中明确说明
2. **输出路径**: 所有输出文件应保存到 `{self._workspace}/output/` 目录
3. **任务依赖**: 在任务描述中说明任务之间的依赖关系
4. **可验证性**: 每个任务应有明确的输出文件或验证标准

## 输出

将生成的 tasks.md 保存到: {tasks_v0_path}

保存后，请确认文件已成功写入:
```bash
cat {tasks_v0_path} | head -20
```
"""

        # Step 1: 先写入 prompt 文件（必须在 execute_tasks 之前！）
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/generation_prompt.md", generation_prompt
        )

        # Step 2: 创建临时任务文件，引导 OpenCode 读取 prompt 并生成 tasks.md
        temp_tasks = f"""# 临时任务：生成执行计划

请阅读 {self._workspace}/.agent/generation_prompt.md 中的指令，生成 tasks.md 文件。

将生成的文件保存到: {tasks_v0_path}
"""
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/temp_generation_task.md", temp_tasks
        )

        # Step 3: 执行生成任务
        gen_result = await self._executor.execute_tasks(
            tasks_md_path=f"{self._workspace}/.agent/temp_generation_task.md",
            workspace_dir=self._workspace,
            mode=OpenCodeMode.BUILD,
            iteration=-1,  # Use -1 表示这是生成阶段
        )

        self._log(
            f"  生成任务执行状态: {gen_result.status.value if gen_result else 'None'}"
        )

        # 验证文件是否生成 - 先检查文件是否存在
        check_result = await self._sandbox.commands.run(
            f"test -f {tasks_v0_path} && echo 'exists' || echo 'not_found'"
        )
        file_status = self._get_stdout(check_result).strip()

        if file_status == "exists":
            try:
                tasks_content = await self._sandbox.files.read_file(tasks_v0_path)
                if tasks_content and len(tasks_content) > 50:
                    self._log(
                        f"  tasks.md 已生成: {tasks_v0_path} ({len(tasks_content)} 字符)"
                    )
                    self._log(f"  前 200 字符: {tasks_content[:200]}...")
                else:
                    self._log(
                        f"  tasks.md 内容过短 ({len(tasks_content) if tasks_content else 0} 字符)，使用默认模板"
                    )
                    default_tasks = self._create_default_tasks_md(input_data)
                    await self._sandbox.files.write_file(tasks_v0_path, default_tasks)
            except Exception as e:
                self._log(f"  读取 tasks.md 失败: {e}，使用默认模板")
                default_tasks = self._create_default_tasks_md(input_data)
                await self._sandbox.files.write_file(tasks_v0_path, default_tasks)
        else:
            self._log(f"  tasks.md 未生成，使用默认模板")
            # 列出 tasks 目录内容以便调试
            list_result = await self._sandbox.commands.run(
                f"ls -la {self._workspace}/tasks/ 2>&1 || echo 'dir not found'"
            )
            self._log(f"  tasks 目录内容: {self._get_stdout(list_result)[:200]}")
            default_tasks = self._create_default_tasks_md(input_data)
            await self._sandbox.files.write_file(tasks_v0_path, default_tasks)

        return tasks_v0_path

    def _create_default_tasks_md(self, input_data: Dict[str, Any]) -> str:
        """创建默认的 tasks.md 模板"""
        params = input_data.get("params", {})
        user_input = input_data.get("user_input", "执行计算任务")
        task_type = input_data.get("task_type", "general")

        return f"""# 计算任务执行计划

> 会话 ID: `{self._session_id}`
> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 模式: 自动生成（默认模板）

## 背景

用户请求: {user_input}

任务类型: {task_type}

## 输入数据

输入文件位于: {self._workspace}/input/

请先检查输入文件:
```bash
ls -la {self._workspace}/input/
```

参数:
```json
{json.dumps(params, indent=2, ensure_ascii=False)}
```

## Phase 1: 数据加载与验证

### 任务 1: 检查输入数据

- **任务 ID**: task_001
- **类型**: DATA_PROCESSING
- **优先级**: 高

**描述:**
1. 列出 input/ 目录中的所有文件
2. 检查文件格式和内容
3. 生成数据摘要报告

**输出:**
- {self._workspace}/output/data_summary.json

## Phase 2: 执行主要任务

### 任务 2: 执行计算

- **任务 ID**: task_002
- **类型**: ANALYSIS
- **优先级**: 高

**描述:**
根据用户请求和参数执行计算任务。

**输出:**
- {self._workspace}/output/results.json

## Phase 3: 生成报告

### 任务 3: 生成执行摘要

- **任务 ID**: task_003
- **类型**: REPORT
- **优先级**: 中

**描述:**
1. 汇总所有结果
2. 生成人类可读的报告

**输出:**
- {self._workspace}/output/execution_summary.json
- {self._workspace}/output/report.md

## 执行说明

1. 按顺序执行所有任务
2. 每完成一个任务，验证输出文件
3. 如遇错误，记录并尝试修复
4. 最终生成完整的执行报告
"""

    async def _execute_tasks_iteration(
        self, tasks_path: str, iteration: int
    ) -> Optional[ExecutionResult]:
        """
        Step 4: 执行单个 tasks.md

        Args:
            tasks_path: tasks.md 文件路径
            iteration: 迭代编号

        Returns:
            ExecutionResult: 执行结果
        """
        self._log(f"Step 4: 执行 tasks.md (迭代 {iteration})...")

        # 报告开始执行
        self._report_progress(
            event_type="sandbox_exec",
            message=f"[START] 开始执行迭代 {iteration + 1} 的任务...",
            details={
                "phase": "execution_start",
                "iteration": iteration + 1,
                "tasks_path": tasks_path,
            },
        )

        # 准备迭代输出目录
        iter_output_dir = f"{self._workspace}/iterations/iter_{iteration}"
        await self._sandbox.commands.run(f"mkdir -p {iter_output_dir}/output")

        # 复制当前 tasks.md 到迭代目录作为记录
        await self._sandbox.commands.run(f"cp {tasks_path} {iter_output_dir}/tasks.md")

        # 更新迭代状态
        await self._update_iteration_state(iteration, "executing")

        # 报告 OpenCode 执行开始
        self._report_progress(
            event_type="sandbox_exec",
            message=f"🤖 OpenCode 正在沙盒中执行任务...",
            details={
                "phase": "opencode_executing",
                "iteration": iteration + 1,
            },
        )

        # 执行任务
        result = await self._executor.execute_tasks(
            tasks_md_path=tasks_path,
            workspace_dir=self._workspace,
            mode=OpenCodeMode.BUILD,
            iteration=iteration,  # 传递迭代编号，生成独立日志文件
        )

        # 保存执行日志（即使为空也要尝试保存）
        log_content = result.stdout or result.error or "(no output)"
        if result.stdout or result.error:
            await self._sandbox.files.write_file(
                f"{iter_output_dir}/opencode.log", log_content
            )
            self._log(f"  执行日志已保存: {len(log_content)} 字符")
        else:
            # 尝试读取 runner.sh 的输出
            try:
                output_log = await self._sandbox.files.read_file(
                    f"{self._workspace}/.opencode_output.log"
                )
                if output_log:
                    await self._sandbox.files.write_file(
                        f"{iter_output_dir}/opencode.log", output_log
                    )
                    self._log(
                        f"  从 .opencode_output.log 恢复日志: {len(output_log)} 字符"
                    )
            except Exception:
                self._log("  无法保存执行日志（无可用内容）")

        # 复制输出文件到迭代目录
        await self._sandbox.commands.run(
            f"cp -r {self._workspace}/output/* {iter_output_dir}/output/ 2>/dev/null || true"
        )

        self._log(
            f"  执行完成: status={result.status.value}, time={result.execution_time_ms}ms"
        )

        # 报告执行完成
        self._report_progress(
            event_type="sandbox_exec",
            message=f"[SUCCESS] 迭代 {iteration + 1} 执行完成 (状态: {result.status.value})",
            details={
                "phase": "execution_complete",
                "iteration": iteration + 1,
                "status": result.status.value,
                "execution_time_ms": result.execution_time_ms,
                "output_files": len(result.output_files) if result.output_files else 0,
            },
        )

        return result

    async def _update_iteration_state(self, iteration: int, status: str) -> None:
        """更新迭代状态文件"""
        state = {
            "session_id": self._session_id,
            "current_iteration": iteration,
            "max_iterations": self.max_iterations,
            "status": status,
            "updated_at": datetime.now().isoformat(),
        }
        state_json = json.dumps(state, indent=2, ensure_ascii=False)
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/iteration_state.json", state_json
        )

    async def _evaluate_outputs(self, iteration: int) -> Dict[str, Any]:
        """
        Step 5: 评估输出

        评估本次迭代的输出质量，包括:
        1. 文件完整性 - 预期文件是否生成
        2. 格式正确性 - JSON/CSV 格式是否正确
        3. 内容有效性 - 数据是否合理
        4. 错误分析 - 日志中是否有错误

        Args:
            iteration: 迭代编号

        Returns:
            Dict: 评估报告
        """
        self._log(f"Step 5: 评估输出 (迭代 {iteration})...")

        iter_output_dir = f"{self._workspace}/iterations/iter_{iteration}/output"
        evaluation_path = (
            f"{self._workspace}/iterations/iter_{iteration}/evaluation.json"
        )

        # 更新状态
        await self._update_iteration_state(iteration, "evaluating")

        # 构建评估 Prompt
        evaluation_prompt = f"""你是一个质量评估专家。

## 任务
评估以下目录中的输出文件质量。

## 待评估目录
{iter_output_dir}

## 评估步骤

1. 列出所有生成的文件:
```bash
ls -la {iter_output_dir}/
```

2. 检查每个文件的内容:
```bash
# 对于 JSON 文件
for f in {iter_output_dir}/*.json; do
    echo "=== $f ==="
    python3 -c "import json; json.load(open('$f'))" && echo "Valid JSON" || echo "Invalid JSON"
    head -20 "$f"
done

# 对于 CSV 文件
for f in {iter_output_dir}/*.csv; do
    echo "=== $f ==="
    head -5 "$f"
    wc -l "$f"
done
```

3. 检查执行日志中的错误:
```bash
cat {self._workspace}/iterations/iter_{iteration}/opencode.log | grep -i "error\|fail\|exception" | head -20
```

## 评估维度

1. **文件完整性**: 是否生成了预期的输出文件
2. **格式正确性**: 文件格式是否正确（JSON/CSV 可解析）
3. **内容有效性**: 数据内容是否合理
4. **错误分析**: 日志中是否有错误

## 输出格式

将评估报告保存到: {evaluation_path}

```json
{{
  "files_generated": ["file1.csv", "file2.json"],
  "total_files": 2,
  "format_check": {{
    "file1.csv": {{"valid": true, "rows": 100, "error": null}},
    "file2.json": {{"valid": true, "keys": ["a", "b"], "error": null}}
  }},
  "errors": [],
  "warnings": ["警告信息（如果有）"],
  "quality_score": 0.85,
  "status": "success",
  "suggestions": [
    "改进建议1",
    "改进建议2"
  ],
  "details": {{
    "execution_errors": 0,
    "data_quality": "good",
    "completeness": 0.9
  }}
}}
```

## status 可能的值
- "success": 质量分数 >= 0.8，无严重错误
- "needs_improvement": 质量分数 0.4-0.79，或有小错误
- "failed": 质量分数 < 0.4，或有严重错误

## quality_score 计算规则
- 文件完整性: 40% (生成文件数/预期文件数)
- 格式正确性: 30% (有效文件数/总文件数)
- 内容有效性: 30% (基于数据质量判断)

请严格按照上述格式输出评估报告。
"""

        # 确保迭代目录存在（OpenCode 写入 evaluation.json 需要）
        iter_dir = f"{self._workspace}/iterations/iter_{iteration}"
        await self._sandbox.commands.run(f"mkdir -p {iter_dir}")

        # 写入评估 prompt
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/eval_prompt.md", evaluation_prompt
        )

        # 创建临时任务让 OpenCode 执行评估
        eval_task = f"""# 评估任务

请执行以下评估:

1. 阅读 {self._workspace}/.agent/eval_prompt.md 中的评估指令
2. 按照指令评估输出文件
3. 将评估报告保存到 {evaluation_path}

【重要】写入 JSON 文件时，请确保:
- 使用完整绝对路径: {evaluation_path}
- 先写入内容，再验证文件是否成功写入

完成后确认:
```bash
cat {evaluation_path}
```
"""
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/eval_task.md", eval_task
        )

        # 执行评估任务 - 使用迭代目录作为 workspace，这样输出路径匹配
        iter_workspace = f"{self._workspace}/iterations/iter_{iteration}"
        await self._sandbox.commands.run(f"mkdir -p {iter_workspace}/output")

        eval_result = await self._executor.execute_tasks(
            tasks_md_path=f"{self._workspace}/.agent/eval_task.md",
            workspace_dir=iter_workspace,
            mode=OpenCodeMode.BUILD,
            iteration=iteration,  # 传递迭代编号
        )

        self._log(
            f"  评估任务执行状态: {eval_result.status.value if eval_result else 'None'}"
        )

        # 读取评估结果 - 先检查文件是否存在
        check_result = await self._sandbox.commands.run(
            f"test -f {evaluation_path} && echo 'exists' || echo 'not_found'"
        )
        file_status = self._get_stdout(check_result).strip()

        if file_status == "exists":
            try:
                eval_content = await self._sandbox.files.read_file(evaluation_path)
                evaluation = json.loads(eval_content)
                self._log(
                    f"  评估完成: quality_score={evaluation.get('quality_score', 0):.2f}, status={evaluation.get('status', 'unknown')}"
                )
            except json.JSONDecodeError as e:
                self._log(f"  解析评估报告失败: {e}，使用默认评估")
                evaluation = await self._create_default_evaluation(iteration)
            except Exception as e:
                self._log(f"  读取评估报告失败: {e}，使用默认评估")
                evaluation = await self._create_default_evaluation(iteration)
        else:
            self._log(f"  评估报告未生成 (路径: {evaluation_path})，使用默认评估")
            # 列出迭代目录内容以便调试
            list_result = await self._sandbox.commands.run(
                f"ls -la {iter_workspace}/ 2>&1 || echo 'dir not found'"
            )
            self._log(f"  迭代目录内容: {self._get_stdout(list_result)[:200]}")
            evaluation = await self._create_default_evaluation(iteration)

        return evaluation

    async def _create_default_evaluation(self, iteration: int) -> Dict[str, Any]:
        """创建默认的评估报告（当 OpenCode 评估失败时使用）"""
        iter_output_dir = f"{self._workspace}/iterations/iter_{iteration}/output"

        # 列出生成的文件
        files = []
        try:
            result = await self._sandbox.commands.run(
                f"ls {iter_output_dir}/ 2>/dev/null || echo ''"
            )
            output = self._get_stdout(result)
            files = [
                f.strip()
                for f in output.strip().split("\n")
                if f.strip() and f.strip() != ""
            ]
        except Exception:
            pass

        # 简单评估：基于文件数量
        quality_score = min(1.0, len(files) / 3) if files else 0.0
        status = "success" if quality_score >= 0.6 else "needs_improvement"

        return {
            "files_generated": files,
            "total_files": len(files),
            "format_check": {},
            "errors": [],
            "warnings": ["使用自动评估（OpenCode 评估失败）"],
            "quality_score": quality_score,
            "status": status,
            "suggestions": ["建议检查 OpenCode 输出日志"],
            "details": {
                "evaluation_mode": "auto",
            },
        }

    def _get_stdout(self, result: Any) -> str:
        """获取命令执行的标准输出"""
        logs = getattr(result, "logs", None)
        if not logs:
            return ""
        stdout = getattr(logs, "stdout", None)
        if not stdout:
            return ""
        if isinstance(stdout, str):
            return stdout
        return "\n".join(getattr(entry, "text", str(entry)) for entry in stdout)

    async def _optimize_tasks(
        self, current_tasks: str, evaluation: Dict[str, Any], iteration: int
    ) -> str:
        """
        Step 6: 优化 tasks.md

        根据评估结果优化任务计划

        Args:
            current_tasks: 当前 tasks.md 路径
            evaluation: 评估报告
            iteration: 当前迭代编号

        Returns:
            优化后的 tasks.md 路径
        """
        self._log(f"Step 6: 优化 tasks.md (迭代 {iteration} -> {iteration + 1})...")

        next_iteration = iteration + 1
        tasks_v_next_path = f"{self._workspace}/tasks/tasks_v{next_iteration}.md"

        # 确保 tasks 目录存在
        await self._sandbox.commands.run(f"mkdir -p {self._workspace}/tasks")

        # 更新状态
        await self._update_iteration_state(iteration, "optimizing")

        # 读取当前 tasks.md
        try:
            current_tasks_content = await self._sandbox.files.read_file(current_tasks)
        except Exception as e:
            self._log(f"  读取当前 tasks.md 失败: {e}")
            current_tasks_content = "# 任务执行计划\n\n无法读取原始任务"

        # 构建优化 Prompt
        optimization_prompt = f"""你是一个任务优化专家。

## 当前任务文件
路径: {current_tasks}

内容:
```
{current_tasks_content[:5000]}
```

（如果内容过长，已截断，请使用以下命令查看完整内容）
```bash
cat {current_tasks}
```

## 评估报告
```json
{json.dumps(evaluation, indent=2, ensure_ascii=False)}
```

## 优化任务

根据评估报告中的问题和建议，优化任务执行计划。

### 优化方向

1. **解决错误**: 如果评估报告中有 errors，请修改任务以解决这些错误
2. **改进数据质量**: 如果 quality_score 较低，考虑添加数据验证步骤
3. **完善输出**: 如果缺少预期输出文件，添加生成这些文件的任务
4. **优化流程**: 根据评估建议改进任务流程

### 优化规则

1. 保留成功的任务不变
2. 修改有问题的任务
3. 添加缺失的步骤
4. 增强错误处理
5. 明确输出文件路径

## 输出

将优化后的 tasks.md 保存到: {tasks_v_next_path}

保存后确认:
```bash
head -50 {tasks_v_next_path}
```

## 格式要求

保持原有的 Markdown 格式:
- 任务 ID 递增
- 每个任务有明确的类型、优先级和描述
- 输出文件路径清晰
"""

        # 写入优化 prompt
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/optimize_prompt.md", optimization_prompt
        )

        # 创建优化任务
        opt_task = f"""# 任务优化

请执行以下优化:

1. 阅读 {self._workspace}/.agent/optimize_prompt.md 中的优化指令
2. 根据评估报告优化任务计划
3. 将优化后的任务保存到 {tasks_v_next_path}

重点:
- 解决评估报告中指出的问题
- 添加缺失的步骤
- 增强错误处理
"""
        await self._sandbox.files.write_file(
            f"{self._workspace}/.agent/optimize_task.md", opt_task
        )

        # 执行优化任务
        opt_result = await self._executor.execute_tasks(
            tasks_md_path=f"{self._workspace}/.agent/optimize_task.md",
            workspace_dir=self._workspace,
            mode=OpenCodeMode.BUILD,
            iteration=iteration,  # 传递迭代编号
        )

        self._log(
            f"  优化任务执行状态: {opt_result.status.value if opt_result else 'None'}"
        )

        # 验证优化后的文件 - 先检查文件是否存在
        check_result = await self._sandbox.commands.run(
            f"test -f {tasks_v_next_path} && echo 'exists' || echo 'not_found'"
        )
        file_status = self._get_stdout(check_result).strip()

        if file_status == "exists":
            try:
                new_tasks_content = await self._sandbox.files.read_file(
                    tasks_v_next_path
                )
                if new_tasks_content and len(new_tasks_content) > 50:
                    self._log(
                        f"  tasks_v{next_iteration}.md 已生成 ({len(new_tasks_content)} 字符)"
                    )
                else:
                    self._log(
                        f"  优化文件内容过短 ({len(new_tasks_content) if new_tasks_content else 0} 字符)，复制当前版本"
                    )
                    await self._sandbox.commands.run(
                        f"cp {current_tasks} {tasks_v_next_path}"
                    )
            except Exception as e:
                self._log(f"  读取优化后的 tasks.md 失败: {e}，复制当前版本")
                await self._sandbox.commands.run(
                    f"cp {current_tasks} {tasks_v_next_path}"
                )
        else:
            self._log(f"  优化未生成新文件，复制当前版本")
            await self._sandbox.commands.run(f"cp {current_tasks} {tasks_v_next_path}")

        return tasks_v_next_path

    async def _generate_final_summary(
        self, iteration_history: List[IterationResult]
    ) -> IterativeExecutionResult:
        """
        Step 7: 生成最终总结

        Args:
            iteration_history: 迭代历史记录

        Returns:
            IterativeExecutionResult: 完整的执行结果
        """
        self._log("Step 7: 生成最终总结...")

        # 确定最终迭代
        final_iteration = len(iteration_history) - 1 if iteration_history else 0
        final_iter_result = iteration_history[-1] if iteration_history else None

        # 确定最终状态
        if final_iter_result:
            final_status = final_iter_result.status
        else:
            final_status = IterationStatus.FAILED

        # 复制最后一次迭代的输出到主 output 目录
        if final_iteration >= 0:
            await self._sandbox.commands.run(
                f"cp -r {self._workspace}/iterations/iter_{final_iteration}/output/* {self._workspace}/output/ 2>/dev/null || true"
            )

        # 收集最终输出文件
        final_output_files = []
        try:
            result = await self._sandbox.commands.run(
                f"find {self._workspace}/output -type f 2>/dev/null || echo ''"
            )
            output = self._get_stdout(result)
            final_output_files = [
                f.strip() for f in output.strip().split("\n") if f.strip()
            ]
        except Exception:
            pass

        # 生成 final_summary.json
        final_summary = {
            "session_id": self._session_id,
            "total_iterations": len(iteration_history),
            "final_status": final_status.value,
            "final_quality_score": final_iter_result.quality_score
            if final_iter_result
            else 0,
            "iteration_summary": [
                {
                    "iteration": r.iteration,
                    "status": r.status.value,
                    "quality_score": r.quality_score,
                    "execution_time_ms": r.execution_time_ms,
                    "errors_count": len(r.errors),
                    "mcp_calls_count": len(r.mcp_calls),
                    "output_files_count": len(r.output_files),
                }
                for r in iteration_history
            ],
            "key_findings": {
                "best_iteration": max(
                    iteration_history, key=lambda x: x.quality_score
                ).iteration
                if iteration_history
                else 0,
                "best_quality_score": max([r.quality_score for r in iteration_history])
                if iteration_history
                else 0,
                "avg_quality_score": sum(r.quality_score for r in iteration_history)
                / len(iteration_history)
                if iteration_history
                else 0,
                "total_errors": sum(len(r.errors) for r in iteration_history),
            },
            "output_files": final_output_files,
            "completed_at": datetime.now().isoformat(),
        }

        # 保存 final_summary.json
        summary_path = f"{self._workspace}/reports/final_summary.json"
        await self._sandbox.files.write_file(
            summary_path, json.dumps(final_summary, indent=2, ensure_ascii=False)
        )
        self._log(f"  最终总结已保存: {summary_path}")

        # 收集所有 MCP 调用
        all_mcp_calls: List[MCPCallRecord] = []
        full_execution_log = ""
        for r in iteration_history:
            all_mcp_calls.extend(r.mcp_calls)
            if r.opencode_log:
                full_execution_log += f"\n\n{'=' * 60}\n迭代 {r.iteration} 日志\n{'=' * 60}\n{r.opencode_log}"

        # 生成详细报告
        detailed_report = {
            "session_id": self._session_id,
            "workspace_dir": self._workspace,
            "generated_at": datetime.now().isoformat(),
            "final_summary": final_summary,
            "iteration_details": [r.to_dict() for r in iteration_history],
            "all_mcp_calls": [m.to_dict() for m in all_mcp_calls],
            "mcp_summary": {
                "total_calls": len(all_mcp_calls),
                "successful_calls": len([m for m in all_mcp_calls if m.success]),
                "failed_calls": len([m for m in all_mcp_calls if not m.success]),
                "tools_used": list({m.tool_name for m in all_mcp_calls}),
                "services_used": list(
                    {m.service_name for m in all_mcp_calls if m.service_name}
                ),
            },
            "config_info": {
                "model_provider": self.config.model_provider,
                "sandbox_domain": self.config.sandbox_domain,
                "max_iterations": self.max_iterations,
                "early_stop_on_success": self.early_stop_on_success,
                "evaluation_criteria": {
                    "min_quality_score": self.evaluation_criteria.min_quality_score,
                    "required_output_files": self.evaluation_criteria.required_output_files,
                },
            },
        }

        # 保存详细报告
        detailed_report_path = (
            f"{self._workspace}/reports/detailed_execution_report.json"
        )
        await self._sandbox.files.write_file(
            detailed_report_path,
            json.dumps(detailed_report, indent=2, ensure_ascii=False),
        )
        self._log(f"  详细报告已保存: {detailed_report_path}")

        # 生成 final_report.md (人类可读)
        final_report = self._generate_final_report_md(
            iteration_history, final_summary, all_mcp_calls
        )
        report_path = f"{self._workspace}/reports/final_report.md"
        await self._sandbox.files.write_file(report_path, final_report)
        self._log(f"  最终报告已保存: {report_path}")

        # 同时复制到 output 目录
        await self._sandbox.commands.run(f"cp {summary_path} {self._workspace}/output/")
        await self._sandbox.commands.run(f"cp {report_path} {self._workspace}/output/")
        await self._sandbox.commands.run(
            f"cp {detailed_report_path} {self._workspace}/output/"
        )

        return IterativeExecutionResult(
            session_id=self._session_id,
            workspace_dir=self._workspace,
            total_iterations=len(iteration_history),
            final_status=final_status,
            final_output_dir=f"{self._workspace}/output",
            final_output_files=final_output_files,
            iteration_history=iteration_history,
            final_summary=final_summary,
            final_report_path=report_path,
            sandbox_id=getattr(self._sandbox, "id", None) if self._sandbox else None,
            # 新增字段
            full_execution_log=full_execution_log,
            all_mcp_calls=all_mcp_calls,
            config_info=detailed_report["config_info"],
            execution_stats={
                "total_iterations": len(iteration_history),
                "total_mcp_calls": len(all_mcp_calls),
                "total_output_files": len(final_output_files),
            },
            detailed_report_path=detailed_report_path,
        )

    def _generate_final_report_md(
        self,
        iteration_history: List[IterationResult],
        final_summary: Dict[str, Any],
        all_mcp_calls: List[MCPCallRecord] = None,
    ) -> str:
        """生成人类可读的最终报告 (Markdown 格式)"""

        all_mcp_calls = all_mcp_calls or []

        # 迭代历史表格
        iter_table = "| 迭代 | 状态 | 质量分数 | 错误数 | 建议数 | MCP调用 |\n"
        iter_table += "|------|------|----------|--------|--------|--------|\n"
        for r in iteration_history:
            iter_table += f"| {r.iteration} | {r.status.value} | {r.quality_score:.2f} | {len(r.errors)} | {len(r.improvement_suggestions)} | {len(r.mcp_calls)} |\n"

        # 输出文件列表
        output_files_md = "\n".join(
            [f"- `{f}`" for f in final_summary.get("output_files", [])]
        )

        # 改进建议汇总
        all_suggestions = []
        for r in iteration_history:
            all_suggestions.extend(r.improvement_suggestions)
        unique_suggestions = list(set(all_suggestions))[:5]  # 去重，最多显示5条
        suggestions_md = (
            "\n".join([f"- {s}" for s in unique_suggestions])
            if unique_suggestions
            else "无"
        )

        # MCP 调用统计
        mcp_summary_md = ""
        if all_mcp_calls:
            mcp_table = "| 工具名 | 服务 | 状态 | 调用时间 |\n"
            mcp_table += "|--------|------|------|----------|\n"
            for call in all_mcp_calls[:20]:  # 最多显示 20 条
                status_emoji = "成功" if call.success else "失败"
                mcp_table += f"| {call.tool_name} | {call.service_name or '-'} | {status_emoji} | {call.timestamp[:19] if call.timestamp else '-'} |\n"
            if len(all_mcp_calls) > 20:
                mcp_table += f"\n*... 还有 {len(all_mcp_calls) - 20} 条调用记录*\n"

            mcp_summary_md = f"""

## MCP 工具调用记录

**总计**: {len(all_mcp_calls)} 次调用
**成功**: {len([m for m in all_mcp_calls if m.success])} 次
**失败**: {len([m for m in all_mcp_calls if not m.success])} 次

{mcp_table}
"""

        report = f"""# 迭代执行报告

> 会话 ID: `{self._session_id}`
> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 执行概览

| 指标 | 值 |
|------|-----|
| **总迭代次数** | {final_summary["total_iterations"]} |
| **最终状态** | {final_summary["final_status"]} |
| **最终质量分数** | {final_summary["final_quality_score"]:.2f} |
| **最佳迭代** | #{final_summary["key_findings"]["best_iteration"]} |
| **最佳质量分数** | {final_summary["key_findings"]["best_quality_score"]:.2f} |
| **平均质量分数** | {final_summary["key_findings"]["avg_quality_score"]:.2f} |
| **总错误数** | {final_summary["key_findings"]["total_errors"]} |

## 迭代历史

{iter_table}
{mcp_summary_md}
## 输出文件

{output_files_md}

## 改进建议汇总

{suggestions_md}

## 详细报告

完整的 JSON 格式报告已保存到:
- reports/detailed_execution_report.json
- output/detailed_execution_report.json

## 下一步

1. 检查输出文件是否符合预期
2. 根据改进建议优化输入数据或参数
3. 如需重新执行，使用相同的 session_id 可复用沙盒

---

*本报告由 IterativeOpenCodeExecutor 自动生成*
"""
        return report

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _generate_session_id(self) -> str:
        """生成唯一的会话 ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"iterative_{timestamp}_{short_uuid}"

    def _determine_status(self, evaluation: Dict[str, Any]) -> IterationStatus:
        """根据评估结果确定迭代状态"""
        status_str = evaluation.get("status", "needs_improvement")
        quality_score = evaluation.get("quality_score", 0.0)

        # 根据状态字符串
        if status_str == "success":
            return IterationStatus.SUCCESS
        elif status_str == "failed":
            return IterationStatus.FAILED

        # 根据质量分数
        if quality_score >= self.evaluation_criteria.min_quality_score:
            return IterationStatus.SUCCESS
        elif quality_score < 0.3:
            return IterationStatus.FAILED
        else:
            return IterationStatus.NEEDS_IMPROVEMENT

    async def _cleanup(self) -> None:
        """清理资源"""
        if self._executor:
            await self._executor.cleanup()
            self._executor = None
            self._sandbox = None

    def _log(self, message: str) -> None:
        """打印日志并发送进度更新"""
        if self.config.show_progress:
            print(f"[IterativeExecutor] {message}")

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
                    message="⚙️ 配置沙盒环境...",
                    details={"phase": "config"},
                )
            return

        # 跳过过长的消息
        if len(message) > 200:
            return

        # 消息映射表：将技术性消息转换为用户友好消息
        message_mappings = {
            "Step 2:": "📦 准备沙盒环境...",
            "Step 3:": "📝 生成任务列表...",
            "Step 4:": "[START] 执行任务...",
            "Step 5:": "[STAT] 评估输出...",
            "Step 6:": "[TOOL] 优化任务...",
            "Step 7:": "[INFO] 生成最终报告...",
            "沙盒已创建": "[SUCCESS] 沙盒环境已就绪",
            "目录结构已创建": "[SUCCESS] 目录结构已创建",
            "环境准备完成": "[SUCCESS] 环境准备完成",
            "处理": "📂 处理输入文件...",
            "上下文已写入": "[SUCCESS] 上下文信息已保存",
            "迭代": lambda msg: f"[RUN] {msg}",
        }

        # 查找匹配的消息
        friendly_message = None
        for key, value in message_mappings.items():
            if key in message:
                if callable(value):
                    friendly_message = value(message)
                else:
                    friendly_message = value
                break

        # 如果没有匹配，发送简化版消息
        if not friendly_message:
            # 清理消息前缀
            clean_message = message.strip()
            if clean_message.startswith("  "):
                clean_message = clean_message[2:]
            friendly_message = clean_message

        # 发送进度更新
        self._report_progress(
            event_type="sandbox_exec",
            message=friendly_message,
            details={"raw": message},
        )


# ============================================================================
# 同步包装器
# ============================================================================


class IterativeOpenCodeExecutorSync:
    """迭代式 OpenCode 执行器的同步包装器"""

    def __init__(
        self,
        config: Optional[OpenCodeConfig] = None,
        max_iterations: int = 3,
        evaluation_criteria: Optional[EvaluationCriteria] = None,
        early_stop_on_success: bool = True,
        progress_callback: Optional[callable] = None,
    ):
        self._executor = IterativeOpenCodeExecutor(
            config=config,
            max_iterations=max_iterations,
            evaluation_criteria=evaluation_criteria,
            early_stop_on_success=early_stop_on_success,
            progress_callback=progress_callback,
        )

    def execute(self, input_data: Dict[str, Any]) -> IterativeExecutionResult:
        """执行迭代流程（同步）"""
        return self._run_async(self._executor.execute(input_data))

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
    # 枚举和数据类
    "IterationStatus",
    "EvaluationLevel",
    "EvaluationCriteria",
    "MCPCallRecord",
    "TaskTimelineEntry",
    "IterationResult",
    "IterativeExecutionResult",
    # 执行器
    "IterativeOpenCodeExecutor",
    "IterativeOpenCodeExecutorSync",
]
