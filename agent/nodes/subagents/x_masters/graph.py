"""X-Masters: Scattered-and-Stacked Agentic Workflow Graph.

Implements the 4-stage workflow from the SciMaster/X-Masters paper using
LangGraph's Send API for fan-out and operator.add reducers for fan-in.

Stages:
    1. Solver ×N:  Independent CodeActAgents solve the problem (Send fan-out)
    2. Critic ×N:  Each solution is reviewed and corrected   (Send fan-out)
    3. Rewriter ×N: All solutions synthesized into new ones  (Send fan-out)
    4. Selector ×1: Best solution is selected                (single node)

All four stages are fully implemented.

Execution model:
    Solvers, Critics, and Rewriters run sequentially within each Send instance
    due to executor.py's global _persistent_namespace and sys.stdout. LLM API
    calls are I/O-bound and benefit from Send's scheduling. True process-level
    parallelism is a future optimization.
"""

import operator
import logging
import os
import time
from datetime import datetime
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .solver import run_single_solver
from .critic import run_single_critic
from .rewriter import run_single_rewriter
from .selector import run_selector
from .tools import merge_search_results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SOLVERS = 5

# Output report directory (next to this file)
_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _init_report(problem: str) -> str:
    """Create a new report file and write the header. Returns the file path."""
    os.makedirs(_REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_REPORT_DIR, f"xmasters_report_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# X-Masters Execution Report\n\n")
        f.write(f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Problem\n\n{problem}\n\n")
        f.write(f"---\n\n")
    logger.info(f"Report initialized: {path}")
    return path


def _append_report(path: str, content: str):
    """Append a section to the report file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)


def _get_report_path() -> str | None:
    """Get the most recent report file path."""
    if not os.path.isdir(_REPORT_DIR):
        return None
    files = sorted(
        [f for f in os.listdir(_REPORT_DIR) if f.startswith("xmasters_report_")],
        reverse=True,
    )
    return os.path.join(_REPORT_DIR, files[0]) if files else None


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------
class SolverInput(TypedDict):
    """Input state for a single Solver Send instance."""
    problem: str
    solver_id: int
    # LLM configuration (passed through from XMastersState)
    llm: str
    source: str
    base_url: str
    api_key: str
    temperature: float
    timeout_seconds: int


class CriticInput(TypedDict):
    """Input state for a single Critic Send instance."""
    problem: str
    solution: str
    solver_id: int
    retrieved_context: str
    # NEW: Semantic conditions for verification
    semantic_conditions: dict
    # LLM configuration (passed through from XMastersState)
    llm: str
    source: str
    base_url: str
    api_key: str
    temperature: float
    timeout_seconds: int


class RewriterInput(TypedDict):
    """Input state for a single Rewriter Send instance."""
    problem: str
    all_solutions: list  # All critiqued solutions (strings)
    rewriter_id: int
    retrieved_context: str
    # LLM configuration (passed through from XMastersState)
    llm: str
    source: str
    base_url: str
    api_key: str
    temperature: float
    timeout_seconds: int


class XMastersState(TypedDict):
    """Global state for the X-Masters workflow."""
    problem: str
    # Stage 1 output: list of {solver_id, solution, success}
    # operator.add reducer auto-merges results from parallel Send instances
    solutions: Annotated[list[dict], operator.add]
    # Stage 2 output: list of {solver_id, solution, original_solution, success}
    critiqued_solutions: Annotated[list[dict], operator.add]
    # Stage 3 output: list of {rewriter_id, solution, success}
    rewritten_solutions: Annotated[list[dict], operator.add]
    # Stage 4 output
    final_answer: str
    # Knowledge context collected from Solver knowledge_search calls,
    # merged in solver_done and shared with downstream stages.
    retrieved_context: str
    # NEW: Semantic conditions for Critic verification (from GeneralQA N1)
    semantic_conditions: dict
    # LLM configuration
    llm: str
    source: str
    base_url: str
    api_key: str
    temperature: float
    timeout_seconds: int
    num_solvers: int
    _report_path: str  # internal: path to the Markdown report file


# ---------------------------------------------------------------------------
# Stage 0: Init report
# ---------------------------------------------------------------------------
def init_report_node(state: XMastersState) -> dict:
    """Initialize the Markdown report file at the start of the pipeline."""
    path = _init_report(state["problem"])
    return {"_report_path": path}


# ---------------------------------------------------------------------------
# Stage 1: Solver nodes
# ---------------------------------------------------------------------------
# Temperature schedule: spread temperatures across solvers for diversity.
# Lower temps → more deterministic; higher temps → more creative/exploratory.
_SOLVER_TEMPERATURES = [0.3, 0.5, 0.7, 0.9, 1.0]


def _get_solver_temperature(solver_id: int, num_solvers: int, base_temp: float) -> float:
    """Return a per-solver temperature for diversity.

    If *base_temp* is explicitly set (not the default 0.7), all solvers use it.
    Otherwise, temperatures are spread across ``_SOLVER_TEMPERATURES``.
    """
    if base_temp != 0.7:  # user explicitly set a temperature
        return base_temp
    if num_solvers == 1:
        return 0.7
    schedule = _SOLVER_TEMPERATURES
    return schedule[solver_id % len(schedule)]


def fan_out_to_solvers(state: XMastersState) -> list[Send]:
    """Fan-out: dispatch N Solver instances via Send API.

    Each Send creates an independent invocation of the solver_node
    with its own state copy containing the problem and solver_id.
    Each solver receives a different temperature for answer diversity.
    """
    n = state.get("num_solvers", NUM_SOLVERS)
    base_temp = state.get("temperature", 0.7)
    temps = [_get_solver_temperature(i, n, base_temp) for i in range(n)]
    logger.info(f"=== Stage 1: Dispatching {n} Solvers (temps={temps}) ===")
    return [
        Send("solver", {
            "problem": state["problem"],
            "solver_id": i,
            "llm": state.get("llm", ""),
            "source": state.get("source", ""),
            "base_url": state.get("base_url", ""),
            "api_key": state.get("api_key", ""),
            "temperature": temps[i],
            "timeout_seconds": state.get("timeout_seconds", 600),
        })
        for i in range(n)
    ]


def solver_node(state: SolverInput) -> dict:
    """Single Solver instance — invoked by Send.

    Runs one CodeActAgent on the problem and returns the result
    wrapped in a list for the operator.add reducer.
    """
    result = run_single_solver(
        problem=state["problem"],
        solver_id=state["solver_id"],
        llm=state.get("llm") or None,
        source=state.get("source") or None,
        base_url=state.get("base_url") or None,
        api_key=state.get("api_key") or None,
        temperature=state.get("temperature", 0.7),
        timeout_seconds=state.get("timeout_seconds", 600),
    )
    return {
        "solutions": [{
            "solver_id": result["solver_id"],
            "solution": result["solution"],
            "success": result["success"],
            "search_results": result.get("search_results", []),
        }]
    }


# ---------------------------------------------------------------------------
# Stage 2: Critic nodes
# ---------------------------------------------------------------------------
def fan_out_to_critics(state: XMastersState) -> list[Send]:
    """Fan-out: dispatch N Critic instances via Send API.

    Each Send creates an independent invocation of the critic_node
    with the problem and one Solver's solution.  Failed Solver solutions
    (success=False) are skipped — they pass through unchanged.
    """
    solutions = state.get("solutions", [])
    successful = [s for s in solutions if s.get("success", False)]
    failed = [s for s in solutions if not s.get("success", False)]

    logger.info(
        f"=== Stage 2: Dispatching {len(successful)} Critics "
        f"({len(failed)} failed solutions skipped) ==="
    )

    sends = [
        Send("critic", {
            "problem": state["problem"],
            "solution": s["solution"],
            "solver_id": s["solver_id"],
            "retrieved_context": state.get("retrieved_context", ""),
            "semantic_conditions": state.get("semantic_conditions"),  # NEW: Pass semantic conditions
            "llm": state.get("llm", ""),
            "source": state.get("source", ""),
            "base_url": state.get("base_url", ""),
            "api_key": state.get("api_key", ""),
            "temperature": state.get("temperature", 0.7),
            "timeout_seconds": state.get("timeout_seconds", 600),
        })
        for s in successful
    ]

    # If ALL solvers failed, we still need to produce critiqued_solutions
    # so downstream stages have something to work with.  Send a single
    # pass-through critic with the first failed solution.
    if not sends and failed:
        sends = [
            Send("critic_passthrough_failed", {
                "solver_id": failed[0]["solver_id"],
                "solution": failed[0]["solution"],
            })
        ]

    return sends


def critic_node(state: CriticInput) -> dict:
    """Single Critic instance — invoked by Send.

    Reviews one Solver's solution and returns the (possibly amended)
    result wrapped in a list for the operator.add reducer.
    """
    result = run_single_critic(
        problem=state["problem"],
        solution=state["solution"],
        solver_id=state["solver_id"],
        retrieved_context=state.get("retrieved_context", ""),
        semantic_conditions=state.get("semantic_conditions"),  # NEW: Pass semantic conditions
        llm=state.get("llm") or None,
        source=state.get("source") or None,
        base_url=state.get("base_url") or None,
        api_key=state.get("api_key") or None,
        temperature=state.get("temperature", 0.7),
        timeout_seconds=state.get("timeout_seconds", 600),
    )
    return {
        "critiqued_solutions": [{
            "solver_id": result["solver_id"],
            "solution": result["solution"],
            "original_solution": result["original_solution"],
            "success": result["success"],
        }]
    }


def critic_passthrough_failed_node(state: dict) -> dict:
    """Pass-through for failed Solver solutions that skip Critic review."""
    logger.info(f"=== Stage 2 (Critic): pass-through for failed solver {state.get('solver_id')} ===")
    return {
        "critiqued_solutions": [{
            "solver_id": state["solver_id"],
            "solution": state["solution"],
            "original_solution": state["solution"],
            "success": False,
        }]
    }


# ---------------------------------------------------------------------------
# Gather nodes (synchronization barriers between Send stages)
# ---------------------------------------------------------------------------
# LangGraph's conditional_edges from a Send node fire once PER INSTANCE,
# not once after all instances complete.  Without gather nodes, chaining
# Send → conditional_edges → Send causes the downstream fan-out to be
# called N times (once per upstream instance) instead of once.
#
# Gather nodes are plain edges (add_edge) that LangGraph waits for ALL
# instances to complete before proceeding, ensuring the reducer has merged
# all results before the next fan-out reads the state.
# ---------------------------------------------------------------------------
def solver_done(state: XMastersState) -> dict:
    """Gather node: waits for all Solver Send instances to complete.

    Also merges knowledge_search results collected by all Solvers into
    a single retrieved_context string for downstream stages.
    """
    solutions = state.get("solutions", [])
    n = len(solutions)

    # Collect all search results from all Solvers
    all_search_results = []
    for s in solutions:
        all_search_results.extend(s.get("search_results", []))

    context = merge_search_results(all_search_results)
    logger.info(
        f"=== Stage 1 complete: {n} solutions gathered, "
        f"{len(all_search_results)} search results → {len(context)} chars context ==="
    )

    # --- Write Stage 1 report ---
    rp = state.get("_report_path")
    if rp:
        lines = [f"## Stage 1: Solver Results ({n} solvers)\n\n"]
        for s in sorted(solutions, key=lambda x: x.get("solver_id", 0)):
            sid = s.get("solver_id", "?")
            ok = "[SUCCESS]" if s.get("success") else "[ERROR]"
            sr_count = len(s.get("search_results", []))
            lines.append(f"### Solver {sid} {ok}\n\n")
            lines.append(f"**Knowledge searches collected**: {sr_count}\n\n")
            lines.append(f"**Solution**:\n\n{s.get('solution', '(empty)')}\n\n")
        lines.append(f"**Retrieved context for downstream** ({len(context)} chars):\n\n")
        if context:
            lines.append(f"```\n{context[:2000]}{'...(truncated)' if len(context) > 2000 else ''}\n```\n\n")
        else:
            lines.append("_(no knowledge search results collected)_\n\n")
        lines.append("---\n\n")
        _append_report(rp, "".join(lines))

    return {"retrieved_context": context}


def critic_done(state: XMastersState) -> dict:
    """Gather node: waits for all Critic Send instances to complete."""
    critiqued = state.get("critiqued_solutions", [])
    n = len(critiqued)
    logger.info(f"=== Stage 2 complete: {n} critiqued solutions gathered ===")

    # --- Write Stage 2 report ---
    rp = state.get("_report_path")
    if rp:
        lines = [f"## Stage 2: Critic Results ({n} critics)\n\n"]
        for s in sorted(critiqued, key=lambda x: x.get("solver_id", 0)):
            sid = s.get("solver_id", "?")
            ok = "[SUCCESS]" if s.get("success") else "[ERROR]"
            lines.append(f"### Critic {sid} {ok}\n\n")
            lines.append(f"**Revised solution**:\n\n{s.get('solution', '(empty)')}\n\n")
        lines.append("---\n\n")
        _append_report(rp, "".join(lines))

    return {}


# ---------------------------------------------------------------------------
# Stage 3: Rewriter nodes
# ---------------------------------------------------------------------------
def fan_out_to_rewriters(state: XMastersState) -> list[Send]:
    """Fan-out: dispatch N Rewriter instances via Send API.

    Each Rewriter receives the problem AND all critiqued solutions.
    This matches the official X-Master implementation where every Rewriter
    sees all N solutions and independently synthesizes a new answer.
    """
    critiqued = state.get("critiqued_solutions", [])
    # Extract solution strings for the rewriter prompt
    all_solution_strs = [s["solution"] for s in critiqued]
    n = state.get("num_solvers", NUM_SOLVERS)

    logger.info(f"=== Stage 3: Dispatching {n} Rewriters ({len(all_solution_strs)} solutions) ===")

    return [
        Send("rewriter", {
            "problem": state["problem"],
            "all_solutions": all_solution_strs,
            "rewriter_id": i,
            "retrieved_context": state.get("retrieved_context", ""),
            "llm": state.get("llm", ""),
            "source": state.get("source", ""),
            "base_url": state.get("base_url", ""),
            "api_key": state.get("api_key", ""),
            "temperature": state.get("temperature", 0.7),
            "timeout_seconds": state.get("timeout_seconds", 600),
        })
        for i in range(n)
    ]


def rewriter_node(state: RewriterInput) -> dict:
    """Single Rewriter instance — invoked by Send.

    Synthesizes all critiqued solutions into a new answer and returns
    the result wrapped in a list for the operator.add reducer.
    """
    result = run_single_rewriter(
        problem=state["problem"],
        all_solutions=state["all_solutions"],
        rewriter_id=state["rewriter_id"],
        retrieved_context=state.get("retrieved_context", ""),
        llm=state.get("llm") or None,
        source=state.get("source") or None,
        base_url=state.get("base_url") or None,
        api_key=state.get("api_key") or None,
        temperature=state.get("temperature", 0.7),
        timeout_seconds=state.get("timeout_seconds", 600),
    )
    return {
        "rewritten_solutions": [{
            "rewriter_id": result["rewriter_id"],
            "solution": result["solution"],
            "success": result["success"],
        }]
    }


# ---------------------------------------------------------------------------
# Stage 4: Selector node
# ---------------------------------------------------------------------------
def selector_node(state: XMastersState) -> dict:
    """Single Selector instance — picks the best rewritten solution.

    Unlike Stages 1-3, the Selector is a single node (no fan-out).
    It receives all rewritten solutions and uses code + tools to verify
    each one before selecting the best.
    """
    rewritten = state.get("rewritten_solutions", [])
    all_solution_strs = [s["solution"] for s in rewritten]

    logger.info(f"=== Stage 4: Selector evaluating {len(all_solution_strs)} solutions ===")

    if not all_solution_strs:
        logger.warning("No rewritten solutions available for selection")
        return {"final_answer": ""}

    result = run_selector(
        problem=state["problem"],
        all_solutions=all_solution_strs,
        retrieved_context=state.get("retrieved_context", ""),
        llm=state.get("llm") or None,
        source=state.get("source") or None,
        base_url=state.get("base_url") or None,
        api_key=state.get("api_key") or None,
        temperature=state.get("temperature", 0.7),
        timeout_seconds=state.get("timeout_seconds", 600),
    )

    # --- Write Stage 3 + Stage 4 report ---
    rp = state.get("_report_path")
    if rp:
        # Stage 3: Rewriter results
        rewritten = state.get("rewritten_solutions", [])
        lines = [f"## Stage 3: Rewriter Results ({len(rewritten)} rewriters)\n\n"]
        for s in sorted(rewritten, key=lambda x: x.get("rewriter_id", 0)):
            rid = s.get("rewriter_id", "?")
            ok = "[SUCCESS]" if s.get("success") else "[ERROR]"
            lines.append(f"### Rewriter {rid} {ok}\n\n")
            lines.append(f"**Rewritten solution**:\n\n{s.get('solution', '(empty)')}\n\n")
        lines.append("---\n\n")

        # Stage 4: Selector result
        lines.append(f"## Stage 4: Final Answer (Selector)\n\n")
        lines.append(f"**Selected answer**:\n\n{result['solution']}\n\n")
        lines.append("---\n\n")
        lines.append(f"_Report complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
        _append_report(rp, "".join(lines))
        logger.info(f"Report written to: {rp}")

    return {"final_answer": result["solution"]}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """Build the X-Masters workflow graph.

    Graph structure:
        START
          → init_report (creates Markdown report file)
          → fan_out_to_solvers (conditional edge, returns Send ×N)
            → solver ×N (each returns {"solutions": [dict]} with search_results)
              → solver_done (gather + write Stage 1 report + merge search_results)
          → fan_out_to_critics (conditional edge, returns Send ×N)
            → critic ×N (each returns {"critiqued_solutions": [dict]})
              → critic_done (gather + write Stage 2 report)
          → fan_out_to_rewriters (conditional edge, returns Send ×N)
            → rewriter ×N (each returns {"rewritten_solutions": [dict]})
              → selector (write Stage 3 + 4 report)
          → END

    Each stage's results are written to a human-readable Markdown report
    file under x_masters/reports/.

    Gather nodes (solver_done, critic_done) are critical synchronization
    barriers.  Without them, conditional_edges from Send nodes fire once
    per instance, causing downstream fan-outs to be called N times instead
    of once.

    solver_done also merges knowledge_search results collected by all
    Solvers into retrieved_context, which is then passed to Critic,
    Rewriter, and Selector prompts.
    """
    builder = StateGraph(XMastersState)

    # Add nodes
    builder.add_node("init_report", init_report_node)
    builder.add_node("solver", solver_node)
    builder.add_node("solver_done", solver_done)
    builder.add_node("critic", critic_node)
    builder.add_node("critic_passthrough_failed", critic_passthrough_failed_node)
    builder.add_node("critic_done", critic_done)
    builder.add_node("rewriter", rewriter_node)
    builder.add_node("selector", selector_node)

    # Edges
    # START → init report file
    builder.add_edge(START, "init_report")
    # init_report → fan-out to N solvers via Send API
    builder.add_conditional_edges("init_report", fan_out_to_solvers, ["solver"])
    # All solvers → gather node (waits for ALL instances + reducer merge)
    builder.add_edge("solver", "solver_done")
    # Gather → fan-out to N critics via Send API
    builder.add_conditional_edges(
        "solver_done", fan_out_to_critics, ["critic", "critic_passthrough_failed"]
    )
    # All critics → gather node (waits for ALL instances + reducer merge)
    builder.add_edge("critic", "critic_done")
    builder.add_edge("critic_passthrough_failed", "critic_done")
    # Gather → fan-out to N rewriters via Send API
    builder.add_conditional_edges(
        "critic_done", fan_out_to_rewriters, ["rewriter"]
    )
    # All rewriters → selector → END
    builder.add_edge("rewriter", "selector")
    builder.add_edge("selector", END)

    return builder


def compile_graph():
    """Build and compile the X-Masters graph."""
    return build_graph().compile()
