"""
Result Evaluator - Task execution result evaluation and summarization module

Features:
1. Collect execution results from all tasks
2. Collect files produced by executor
3. Analyze in combination with immunity plan
4. Generate final summary report
"""

# Import subgraph build functions and mappers
from .graph import (
    build_result_evaluator_subgraph,
    result_evaluator_input_mapper,
    result_evaluator_output_mapper,
)

# Import state classes
from .state import ResultEvaluatorState, TaskResultSummary

# Preserve original CodeAct exports (backward compatible)
from .agent import CodeActAgent
from .executor import run_python_repl, run_r_code, run_bash_script, run_with_timeout
from .llm import get_llm

__all__ = [
    # Subgraph interface
    "build_result_evaluator_subgraph",
    "result_evaluator_input_mapper",
    "result_evaluator_output_mapper",
    # State classes
    "ResultEvaluatorState",
    "TaskResultSummary",
    # CodeAct original exports (backward compatible)
    "CodeActAgent",
    "run_python_repl",
    "run_r_code",
    "run_bash_script",
    "run_with_timeout",
    "get_llm",
]
