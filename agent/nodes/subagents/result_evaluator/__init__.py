"""
CodeAct - Code Execution Agent Module
Extracted from Biomni framework

This module provides a CodeAct agent that can:
1. Generate code based on task descriptions
2. Execute code (Python, R, Bash)
3. Observe results and iterate
"""

# Import from local modules (relative imports)
from .agent import CodeActAgent
from .executor import run_python_repl, run_r_code, run_bash_script, run_with_timeout
from .llm import get_llm

__all__ = [
    "CodeActAgent",
    "run_python_repl",
    "run_r_code",
    "run_bash_script",
    "run_with_timeout",
    "get_llm",
]
