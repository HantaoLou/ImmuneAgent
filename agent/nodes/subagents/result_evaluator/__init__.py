"""
Result Evaluator - 任务执行结果评估与总结模块

功能：
1. 收集所有任务的执行结果
2. 收集 executor 产出的文件
3. 结合 immunity 计划进行分析
4. 生成最终总结报告
"""

# 导入子图构建函数和映射器
from .graph import (
    build_result_evaluator_subgraph,
    result_evaluator_input_mapper,
    result_evaluator_output_mapper,
)

# 导入状态类
from .state import ResultEvaluatorState, TaskResultSummary

# 保留原有的 CodeAct 导出（向后兼容）
from .agent import CodeActAgent
from .executor import run_python_repl, run_r_code, run_bash_script, run_with_timeout
from .llm import get_llm

__all__ = [
    # 子图接口
    "build_result_evaluator_subgraph",
    "result_evaluator_input_mapper",
    "result_evaluator_output_mapper",
    # 状态类
    "ResultEvaluatorState",
    "TaskResultSummary",
    # CodeAct 原有导出（向后兼容）
    "CodeActAgent",
    "run_python_repl",
    "run_r_code",
    "run_bash_script",
    "run_with_timeout",
    "get_llm",
]
