#!/data/pkg_env_cache/conda_envs/bio-agent/bin/python
"""Test script for X-Masters workflow.

Test modes:
  1. test_single_solver: Quick validation of a single Solver instance
  2. test_full_graph:    Full 4-stage pipeline via graph.invoke()
                         Solver ×N → Critic ×N → Rewriter ×N → Selector ×1

Usage:
  python test_solver.py --mode single
  python test_solver.py --mode graph --num-solvers 2
  python test_solver.py --mode graph --num-solvers 2 --question-id 1
"""

import sys
import os
import csv
import time
import logging
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "agent"))

# Load environment variables from deep_research/.env (has complete config)
from dotenv import load_dotenv
deep_research_env = project_root / "agent" / "nodes" / "subagents" / "deep_research" / ".env"
load_dotenv(deep_research_env, override=True)

# Configure logging — show all stages clearly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

# CSV file with test questions
CSV_PATH = Path(__file__).parent / "hle_biomedical_high_relevance_with_strategies.csv"

DEFAULT_QUESTION = """
The predictive ability of a polygenic score, measured by variance explained, is necessarily lower than the SNP heritability for the phenotype. Answer with one of the following:
True
False
"""


def load_question_from_csv(question_id: int = 1) -> str:
    """Load a question from the CSV file by row index (0-based)."""
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == question_id:
                q = row["question"].strip()
                print(f"Loaded Q{question_id} [{row['id']}]: {q[:120]}...")
                return q
    print(f"Question ID {question_id} not found, using default question")
    return DEFAULT_QUESTION.strip()


def test_single_solver():
    """Quick test: run one Solver instance directly."""
    from nodes.subagents.x_masters.solver import run_single_solver

    print("=" * 70)
    print("Test: Single Solver")
    print("=" * 70)

    result = run_single_solver(
        problem=DEFAULT_QUESTION.strip(),
        solver_id=0,
        temperature=0.7,
        llm="deepseek-chat",
        source="Custom",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        timeout_seconds=300,
        verbose=True,
    )

    print(f"\nSuccess: {result['success']}")
    print(f"Solution: {result['solution'][:500]}")
    return result


def test_full_graph(num_solvers: int = 2, question_id: int = 1):
    """Full 4-stage X-Masters pipeline via graph.invoke().

    Runs: Solver ×N → Critic ×N → Rewriter ×N → Selector ×1
    """
    from nodes.subagents.x_masters.graph import compile_graph

    # Load question
    question = load_question_from_csv(question_id)

    print("\n" + "=" * 70)
    print(f"X-Masters Full Pipeline (N={num_solvers})")
    print(f"  Solver ×{num_solvers} → Critic ×{num_solvers} → "
          f"Rewriter ×{num_solvers} → Selector ×1")
    print("=" * 70)
    print(f"\nQuestion:\n  {question[:200]}...\n")

    # Compile graph
    graph = compile_graph()

    # Invoke
    t0 = time.time()
    result = graph.invoke({
        "problem": question,
        "solutions": [],
        "critiqued_solutions": [],
        "rewritten_solutions": [],
        "final_answer": "",
        "retrieved_context": "",
        "llm": "deepseek-chat",
        "source": "Custom",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.7,
        "timeout_seconds": 300,
        "num_solvers": num_solvers,
        "_report_path": "",
    })
    elapsed = time.time() - t0

    # --- Print results ---
    print("\n" + "=" * 70)
    print("STAGE 1: SOLVER RESULTS")
    print("=" * 70)
    solutions = result.get("solutions", [])
    print(f"  Total: {len(solutions)} solutions")
    for s in solutions:
        status = "OK" if s.get("success") else "FAIL"
        print(f"\n  [Solver {s['solver_id']}] [{status}]")
        print(f"    {s['solution'][:300]}")

    print("\n" + "=" * 70)
    print("STAGE 2: CRITIC RESULTS")
    print("=" * 70)
    critiqued = result.get("critiqued_solutions", [])
    print(f"  Total: {len(critiqued)} critiqued solutions")
    for s in critiqued:
        status = "OK" if s.get("success") else "FAIL"
        print(f"\n  [Critic for Solver {s['solver_id']}] [{status}]")
        print(f"    {s['solution'][:300]}")

    print("\n" + "=" * 70)
    print("STAGE 3: REWRITER RESULTS")
    print("=" * 70)
    rewritten = result.get("rewritten_solutions", [])
    print(f"  Total: {len(rewritten)} rewritten solutions")
    for s in rewritten:
        status = "OK" if s.get("success") else "FAIL"
        print(f"\n  [Rewriter {s['rewriter_id']}] [{status}]")
        print(f"    {s['solution'][:300]}")

    print("\n" + "=" * 70)
    print("STAGE 4: SELECTOR — FINAL ANSWER")
    print("=" * 70)
    final = result.get("final_answer", "")
    print(f"\n  {final[:500]}")

    print("\n" + "=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    print(f"  Solvers:   {len(solutions)} "
          f"({sum(1 for s in solutions if s.get('success'))} succeeded)")
    print(f"  Critics:   {len(critiqued)} "
          f"({sum(1 for s in critiqued if s.get('success'))} succeeded)")
    print(f"  Rewriters: {len(rewritten)} "
          f"({sum(1 for s in rewritten if s.get('success'))} succeeded)")
    print(f"  Final answer length: {len(final)} chars")
    print(f"  Total time: {elapsed:.1f}s")
    rp = result.get("_report_path", "")
    if rp:
        print(f"  Report file: {rp}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="X-Masters test")
    parser.add_argument("--mode", choices=["single", "graph"], default="graph",
                        help="Test mode: 'single' for one solver, 'graph' for full pipeline")
    parser.add_argument("--num-solvers", type=int, default=2,
                        help="Number of solvers (default: 2)")
    parser.add_argument("--question-id", type=int, default=1,
                        help="Question index from CSV (0-based, default: 1)")
    args = parser.parse_args()

    if args.mode == "single":
        test_single_solver()
    else:
        test_full_graph(num_solvers=args.num_solvers, question_id=args.question_id)
