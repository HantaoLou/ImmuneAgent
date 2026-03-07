"""
Coding Agent 模块 - 在 OpenSandbox 中运行 OpenCode

提供在远程沙盒中执行 Coding Agent 的能力，支持：
1. 通过 OpenCode CLI 执行任务
2. MCP 工具调用
3. 代码生成与执行
4. 结果收集与评估
5. 迭代式执行与优化

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
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    预置工具 & 环境                              │  │
│  │  • MCP 工具（NetTCR、IgBLAST、MetaBCR...）                      │  │
│  │  • Python/Node.js 运行时                                       │  │
│  │  • 文件系统（input/、output/、workspace/）                      │  │
│  │  • 浏览器自动化（Playwright）                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

使用示例：
    # 方式一：直接运行 Coding Agent
    from coding_agent import run_coding_agent_in_sandbox, TaskContext
    
    result = await run_coding_agent_in_sandbox(
        tasks_md_content=tasks_md,
        context=TaskContext(session_id="test", user_input="分析数据"),
        session_id="test",
    )
    
    # 方式二：从 GlobalState 运行（用于 main_graph.py）
    from coding_agent import coding_agent_node
    
    state = await coding_agent_node(state)
    
    # 方式三：使用迭代执行器（推荐）
    from coding_agent import IterativeOpenCodeExecutor, OpenCodeConfig
    
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

模块组成：
- config.py: 配置类和结果模型
- opencode_executor.py: OpenCode 执行器核心类
- iterative_executor.py: 迭代式执行器（自动生成tasks.md + 迭代优化）
- tasks_md_generator.py: tasks.md 生成器
- integration.py: 与 main_graph 集成接口
"""

# 配置类
from coding_agent.config import (
    OpenCodeConfig,
    OpenCodeMode,
    ExecutionStatus,
    ExecutionResult,
    TaskExecutionRecord,
    TaskType,
    TaskContext,
    TasksMDConfig,
    DEFAULT_MCP_CONFIG,
    DEFAULT_OPENCODE_CONFIG,
)

# 执行器
from coding_agent.opencode_executor import (
    OpenCodeExecutor,
    OpenCodeExecutorSync,
)

# 迭代执行器
from coding_agent.iterative_executor import (
    IterationStatus,
    EvaluationLevel,
    EvaluationCriteria,
    MCPCallRecord,
    TaskTimelineEntry,
    IterationResult,
    IterativeExecutionResult,
    IterativeOpenCodeExecutor,
    IterativeOpenCodeExecutorSync,
)

# 执行追踪器
from coding_agent.execution_tracker import (
    ExecutionTracker,
    MCPLogEntry,
    TaskExecution,
)

# tasks.md 生成器
from coding_agent.tasks_md_generator import (
    generate_tasks_md_content,
    generate_and_save_tasks_md,
    create_simple_tasks_md,
)

# 集成接口
from coding_agent.integration import (
    run_coding_agent_in_sandbox,
    run_coding_agent_sync,
    run_coding_agent_from_state,
    coding_agent_node,
    coding_agent_node_sync,
    execute_simple_tasks,
    execute_mcp_tool_in_sandbox,
)


__all__ = [
    # 配置类
    "OpenCodeConfig",
    "OpenCodeMode",
    "ExecutionStatus",
    "ExecutionResult",
    "TaskExecutionRecord",
    "TaskType",
    "TaskContext",
    "TasksMDConfig",
    "DEFAULT_MCP_CONFIG",
    "DEFAULT_OPENCODE_CONFIG",
    
    # 执行器
    "OpenCodeExecutor",
    "OpenCodeExecutorSync",
    
    # 迭代执行器
    "IterationStatus",
    "EvaluationLevel",
    "EvaluationCriteria",
    "MCPCallRecord",
    "TaskTimelineEntry",
    "IterationResult",
    "IterativeExecutionResult",
    "IterativeOpenCodeExecutor",
    "IterativeOpenCodeExecutorSync",
    
    # 执行追踪器
    "ExecutionTracker",
    "MCPLogEntry",
    "TaskExecution",
    
    # tasks.md 生成器
    "generate_tasks_md_content",
    "generate_and_save_tasks_md",
    "create_simple_tasks_md",
    
    # 集成接口
    "run_coding_agent_in_sandbox",
    "run_coding_agent_sync",
    "run_coding_agent_from_state",
    "coding_agent_node",
    "coding_agent_node_sync",
    "execute_simple_tasks",
    "execute_mcp_tool_in_sandbox",
]

__version__ = "1.2.0"
