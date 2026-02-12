"""X-Masters: Scattered-and-Stacked Agentic Workflow.

Implements the X-Masters paper's 4-stage workflow:
  Stage 1 (Solver): N independent CodeActAgents solve the same problem
  Stage 2 (Critic): Each solution is independently reviewed and corrected
  Stage 3 (Rewriter): All solutions are synthesized into new ones
  Stage 4 (Selector): The best solution is selected

All four stages are fully implemented with Send API fan-out.
"""
from .solver import solve, run_single_solver
from .critic import run_single_critic
from .rewriter import run_single_rewriter
from .selector import run_selector
from .graph import build_graph, compile_graph, XMastersState

__all__ = [
    "solve",
    "run_single_solver",
    "run_single_critic",
    "run_single_rewriter",
    "run_selector",
    "build_graph",
    "compile_graph",
    "XMastersState",
]
