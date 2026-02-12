"""
CodeAct - Code Execution Agent Module
Extracted from Biomni framework

This module provides a CodeAct agent that can:
1. Generate code based on task descriptions
2. Execute code (Python, R, Bash)
3. Observe results and iterate
"""

import sys
import os

# Ensure the package directory is in the path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from agent import CodeActAgent
from executor import run_python_repl, run_r_code, run_bash_script, run_with_timeout
from llm import get_llm

__all__ = [
    "CodeActAgent",
    "run_python_repl",
    "run_r_code",
    "run_bash_script",
    "run_with_timeout",
    "get_llm",
]
