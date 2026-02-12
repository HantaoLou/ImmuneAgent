"""X-Masters Stage 1: Solver

Runs N independent CodeActAgent instances on the same problem.
Each Solver independently reasons through the problem using code execution,
producing a diverse set of solutions through temperature-based sampling.

Architecture (instance-level isolation):
    For each of N solvers:
        1. Create fresh CodeActAgent instance (owns its own _namespace)
        2. Inject tools into the agent's instance-level namespace
        3. Run solver.go(problem)
        4. Extract <solution> from output
        5. Collect solution

    Each CodeActAgent holds its own _namespace dict and _captured_plots list,
    so parallel Send instances (LangGraph ThreadPoolExecutor) are thread-safe.
"""

import logging
import re

from .tools import inject_tools_to_namespace, make_tracked_knowledge_search

# ---------------------------------------------------------------------------
# Import CodeActAgent from result_evaluator
# ---------------------------------------------------------------------------
# Use absolute import to avoid conflict with top-level agent package
from agent.nodes.subagents.result_evaluator.agent import CodeActAgent
from agent.nodes.subagents.result_evaluator import executor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SOLVERS = 5

SOLVER_SYSTEM_PROMPT = """\
You are a helpful assistant assigned with the task of problem-solving.
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions to assist you throughout the process.

At each turn, you should first provide your step-by-step thinking and analysis.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output. Your code should be enclosed using "<execute>" tag, for example:
<execute>
print("Hello World!")
</execute>

2) When you have the final answer, provide it using "<solution>" tag, for example:
The answer is <solution> 42 </solution>.

## Available Tools

You have access to the following pre-loaded tool functions that you can call directly in your code:

### Search Tools
- **web_search(query, max_results=5)** — Search the web for current information and research findings
- **knowledge_search(query, k=5)** — Search internal knowledge base for domain-specific papers and data
- **read_webpage(url, max_chars=10000)** — Fetch and read the full text content of a URL (paper, article, documentation). Use this to read papers found via web_search instead of relying only on search snippets

### Biomedical Database Tools (88 tables, pre-loaded — call directly)
- **query_kg(entity_name, entity_type, relation, target_type, limit=50)** — Knowledge graph: gene-disease-drug-pathway associations (8.1M records)
- **query_expression(gene, tissue, min_expression=0, limit=50)** — GTEx tissue gene expression (TPM, 54 tissues)
- **query_disease_gene(gene, disease, min_score, limit=50)** — DisGeNET disease-gene associations
- **query_gene(gene_id, chromosome, limit=50)** — Ensembl gene annotations
- **query_protein_atlas(gene, tissue, subcellular_location, limit=50)** — Human Protein Atlas expression & localization
- **query_omim(gene, disease, limit=50)** — OMIM Mendelian disease-gene associations
- **query_ppi(gene_id, gene_id_b, experiment_type="all", organism_id, limit=50)** — BioGRID protein-protein interactions
- **query_drug_interaction(drug_name, drug_name_b, severity="all", limit=50)** — DDInter drug-drug interactions
- **query_binding(ligand_name, target_name, limit=50)** — BindingDB drug-target binding affinity (Ki/Kd/IC50)
- **query_variant(rs_id, chromosome, limit=50)** — Genetic variant/SNP data
- **query_gwas(disease_trait, gene, snp, p_value_threshold=5e-8, limit=50)** — GWAS Catalog SNP-trait associations
- **query_genebass(gene, phenotype, variant_type="plof", p_value_threshold=1e-6, limit=50)** — UK Biobank rare variant burden tests
- **query_tcr(epitope, pathology, cdr3_beta, mhc, limit=50)** — McPAS-TCR T cell receptor-antigen data
- **query_mirna_target(mirna, target_gene, min_score=80, limit=50)** — miRDB miRNA target predictions
- **query_mirna_validated(mirna, target_gene, species, limit=50)** — miRTarBase validated miRNA-target interactions
- **query_sgrna(target_gene, species="human", min_efficacy=0.5, limit=20)** — CRISPR sgRNA design sequences
- **query_go(term_id, name, keyword, namespace, limit=50)** — Gene Ontology terms
- **query_hpo(term_id, name, keyword, limit=50)** — Human Phenotype Ontology terms
- **query_geneset(gene_symbol, geneset_name, collection="hallmark", limit=50)** — MSigDB gene sets & pathways
- **query_drug_for_disease(disease_name, min_score=0.5, top_k=20)** — TxGNN AI-predicted drugs for disease
- **query_disease_for_drug(drug_name, min_score=0.5, top_k=20)** — TxGNN AI-predicted diseases for drug
- **query_depmap(cell_line, data_type="crispr_dependency", limit=50)** — DepMap cancer cell line dependencies
- **query_cell_markers(cell_type, marker_gene, limit=50)** — Cell type marker genes
- **query_virus_host(viral_protein, host_gene, limit=50)** — Virus-host protein interactions
- **query_drug_repurposing(drug_name, target, moa, limit=50)** — Broad drug repurposing hub

All tools are pre-loaded. Call them directly: `result = query_kg(entity_name="BRCA1"); print(result)`

## Code Execution

- Python (default): <execute> print("Hello") </execute>
- R code: <execute> #!R\nprint("Hello") </execute>
- Bash: <execute> #!BASH\necho "Hello" </execute>

You have access to all standard Python libraries (numpy, scipy, sympy, pandas, etc.) and can write arbitrary code for computation, data analysis, mathematical derivation, simulation, and verification.

## CRITICAL RULES

1. **EVERY response MUST contain either <execute> or <solution> tag.** No exceptions.
2. **When you have the final answer, you MUST use <solution> tag immediately.**
3. **Pure text responses without tags are NOT allowed** and will be rejected.
4. **When choosing between answer choices, prefer the most specific and mechanistically precise answer.** If one answer describes a specific mechanism (e.g., a named structural motif or pathway) and another describes a more general phenomenon that could encompass it, choose the specific one. A precise mechanistic explanation is always preferred over a vague or composite description.
5. **When you find a relevant primary research paper, read it carefully using read_webpage() and base your answer on the paper's actual conclusions**, not on textbook-level generalizations.
"""


# ---------------------------------------------------------------------------
# Initial Reasoning Guidance (IRG) — X-Master paper Section 2.3
# ---------------------------------------------------------------------------
# Injected after the user's problem as a first-person self-statement.
# This guides the model to "believe" it should use code for reasoning,
# rather than relying solely on internal knowledge.
SOLVER_IRG = """\
I need to solve this problem carefully. Let me think about what approach would be most effective.

I have access to an interactive coding environment where I can write and execute Python code. I should leverage this capability to:
- Query biomedical databases directly using the pre-loaded tools (query_kg, query_expression, query_disease_gene, query_gwas, query_ppi, etc.) when the problem involves genes, diseases, drugs, or molecular biology
- Search for relevant information using web_search() and knowledge_search() when I need broader facts or domain knowledge
- **Read full paper content** using read_webpage(url) when web_search returns a relevant paper — search snippets are often insufficient for nuanced questions
- Write code to perform calculations, mathematical derivations, or data analysis when the problem involves quantitative reasoning
- Verify my reasoning by implementing checks and running them programmatically
- Break down complex problems into smaller computational steps

CRITICAL: I should not rely solely on my internal knowledge or textbook-level understanding — real-world biology and chemistry often involve specialized mechanisms that deviate from simple textbook models. When I find a relevant paper via web_search, I MUST use read_webpage(url) to read its full content and extract the actual findings, rather than guessing from the title or snippet alone.

I should actively look for non-obvious or specialized explanations that go beyond standard textbook answers. The correct answer may involve a nuanced mechanism that requires reading primary literature to understand.

Let me start by analyzing what this problem requires and writing appropriate code.
"""


# ---------------------------------------------------------------------------
# SolverAgent: CodeActAgent with Solver-specific system prompt
# ---------------------------------------------------------------------------
class SolverAgent(CodeActAgent):
    """CodeActAgent specialized for X-Masters problem solving.
    
    Inherits all functionality from CodeActAgent but overrides:
    - System prompt: concise capability description + format rules
    - go() method: injects Initial Reasoning Guidance (IRG) after user message
    
    The IRG mechanism (X-Master paper Section 2.3) primes the model to
    actively write code for reasoning, computation, and verification,
    rather than relying solely on internal knowledge.
    """
    
    def _generate_system_prompt(self) -> str:
        """Generate Solver-specific system prompt."""
        return SOLVER_SYSTEM_PROMPT

    def go(self, prompt: str):
        """Execute the agent with IRG injection.

        Overrides CodeActAgent.go() to inject Initial Reasoning Guidance
        as an AIMessage after the user's problem. This makes the model
        "believe" it has already started thinking about using code,
        priming it to write code for reasoning rather than answering directly.
        """
        from langchain_core.messages import HumanMessage, AIMessage

        self.user_task = prompt

        # IRG is injected as an AIMessage — the model sees it as its own
        # first-person thought, priming code-based reasoning behavior.
        inputs = {
            "messages": [
                HumanMessage(content=prompt),
                AIMessage(content=SOLVER_IRG.strip()),
            ],
            "next_step": "generate",
        }
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []
    
        final_state = None
        step = 0
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            step += 1
            out = self._pretty_print(message)
            self.log.append(out)
            self._verbose_log(step, message)
            final_state = s

        self._conversation_state = final_state
        return self.log, message.content


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def extract_solution(text: str) -> str:
    """Extract content from <solution> tags.

    Args:
        text: LLM output text potentially containing <solution> tags

    Returns:
        Extracted solution text, or full text if no tags found
    """
    match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _reset_execution_env(namespace: dict = None, captured_plots: list = None):
    """Reset execution environment.

    Args:
        namespace: Instance-level namespace to clear. If None, clears global.
        captured_plots: Instance-level plots list to clear. If None, clears global.
    """
    if namespace is not None:
        namespace.clear()
    else:
        executor._persistent_namespace.clear()
    executor.clear_captured_plots(captured_plots)


def _inject_solver_tools(namespace: dict = None):
    """Inject Solver tools (web_search, knowledge_search) into the execution namespace.
    
    Args:
        namespace: Target namespace dict. If None, uses global.
    """
    if namespace is None:
        namespace = executor._persistent_namespace
    inject_tools_to_namespace(namespace)


def _go_verbose(solver, problem: str, solver_id: int):
    """Run solver with real-time verbose output.
    
    Streams execution steps to stdout, showing:
    - LLM thinking/planning
    - Code generation (<execute> blocks)
    - Code execution results (<observation>)
    
    Args:
        solver: CodeActAgent instance
        problem: The problem to solve
        solver_id: Solver identifier for log prefix
        
    Returns:
        Tuple of (log, last_content) same as solver.go()
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    solver.user_task = problem
    # Inject IRG as AIMessage — same as SolverAgent.go()
    inputs = {
        "messages": [
            HumanMessage(content=problem),
            AIMessage(content=SOLVER_IRG.strip()),
        ],
        "next_step": "generate",
    }
    config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
    solver.log = []
    
    final_state = None
    step = 0
    
    print(f"\n{'='*60}")
    print(f"[Solver {solver_id}] VERBOSE EXECUTION")
    print(f"{'='*60}")
    
    for s in solver.app.stream(inputs, stream_mode="values", config=config):
        message = s["messages"][-1]
        step += 1
        
        # Determine message type and format
        if isinstance(message, HumanMessage):
            msg_type = "HUMAN"
            prefix = "👤"
        elif isinstance(message, AIMessage):
            msg_type = "AI"
            prefix = "🤖"
        else:
            msg_type = "SYSTEM"
            prefix = "⚙️"
        
        content = message.content
        
        # Print step header
        print(f"\n{'─'*40} Step {step} ({msg_type}) {'─'*40}")
        
        # Highlight code blocks and observations
        if "<execute>" in content:
            print(f"{prefix} [CODE GENERATION]")
            # Extract and highlight code
            import re
            code_match = re.search(r'<execute>(.*?)</execute>', content, re.DOTALL)
            if code_match:
                print(f"```python\n{code_match.group(1).strip()}\n```")
            # Print surrounding reasoning (truncated)
            reasoning = re.sub(r'<execute>.*?</execute>', '[CODE BLOCK]', content, flags=re.DOTALL)
            if len(reasoning) > 500:
                print(f"\nReasoning: {reasoning[:500]}...")
            else:
                print(f"\nReasoning: {reasoning}")
        elif "<observation>" in content:
            print(f"{prefix} [EXECUTION RESULT]")
            print(content[:1000] if len(content) > 1000 else content)
        elif "<solution>" in content:
            print(f"{prefix} [FINAL SOLUTION]")
            # Extract and highlight solution
            import re
            sol_match = re.search(r'<solution>(.*?)</solution>', content, re.DOTALL)
            if sol_match:
                print(f">>> SOLUTION: {sol_match.group(1).strip()} <<<")
            print(f"\nFull response: {content}")
        else:
            # Regular message - truncate if too long
            if len(content) > 800:
                print(f"{prefix} {content[:800]}...")
            else:
                print(f"{prefix} {content}")
        
        out = solver._pretty_print(message)
        solver.log.append(out)
        final_state = s
    
    solver._conversation_state = final_state
    print(f"\n{'='*60}")
    print(f"[Solver {solver_id}] EXECUTION COMPLETE ({step} steps)")
    print(f"{'='*60}\n")
    
    return solver.log, message.content


# ---------------------------------------------------------------------------
# Core solver logic
# ---------------------------------------------------------------------------
def run_single_solver(
    problem: str,
    solver_id: int = 0,
    temperature: float = 0.7,
    llm: str = None,
    source: str = None,
    base_url: str = None,
    api_key: str = None,
    timeout_seconds: int = 120,
    verbose: bool = False,
) -> dict:
    """Run a single Solver instance.

    Creates a fresh CodeActAgent, overrides its system prompt,
    and runs it on the problem. The execution namespace is cleared
    before each run to ensure isolation.

    knowledge_search is wrapped with a tracked version that collects all
    search results. These are returned in the result dict so the graph
    can aggregate them and pass to downstream agents (Critic/Rewriter/Selector).

    Args:
        problem: The problem to solve
        solver_id: Identifier for this solver (0-indexed)
        temperature: LLM sampling temperature (0.7 for diversity)
        llm: LLM model name (None → env default)
        source: LLM provider (None → auto-detect)
        base_url: Custom API base URL
        api_key: API key
        timeout_seconds: Code execution timeout per block
        verbose: If True, print execution steps in real-time

    Returns:
        dict with keys:
            - solution (str): Extracted solution text
            - log (list[str]): Full conversation log
            - solver_id (int): Which solver produced this
            - success (bool): Whether the solver completed without error
            - search_results (list[str]): Collected knowledge_search results
    """
    logger.info(f"[Solver {solver_id}] Starting{'...' if not verbose else ' (verbose mode)...'}") 

    # 1. Create fresh SolverAgent instance (inherits from CodeActAgent)
    #    Each instance gets: new LLM session, new MemorySaver, new message history,
    #    and its own _namespace / _captured_plots for thread-safe execution.
    solver = SolverAgent(
        llm=llm,
        source=source,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        verbose=True,
        agent_label=f"Solver {solver_id}",
    )

    # 2. Inject solver tools into the agent's instance-level namespace
    _inject_solver_tools(solver._namespace)

    # 3. Replace knowledge_search with a tracked version that collects results.
    #    The tracked version has the same signature and returns the same result,
    #    but also appends each successful result to search_collector.
    search_collector = []
    solver._namespace["knowledge_search"] = make_tracked_knowledge_search(search_collector)

    # 4. Run the solver (with optional real-time verbose output)
    try:
        if verbose:
            # Verbose mode: stream and print each step in real-time
            log, last_content = _go_verbose(solver, problem, solver_id)
        else:
            log, last_content = solver.go(problem)
        solution = extract_solution(last_content)
        success = True
        logger.info(
            f"[Solver {solver_id}] Completed successfully, "
            f"collected {len(search_collector)} knowledge search result(s)"
        )
    except Exception as e:
        logger.error(f"[Solver {solver_id}] Failed: {e}")
        solution = f"[Solver {solver_id} failed: {str(e)}]"
        log = []
        success = False

    # 5. Clean up namespace after this solver (defensive)
    _reset_execution_env(solver._namespace, solver._captured_plots)

    return {
        "solution": solution,
        "log": log,
        "solver_id": solver_id,
        "success": success,
        "search_results": search_collector,
    }


def solve(
    problem: str,
    num_solvers: int = NUM_SOLVERS,
    temperature: float = 0.7,
    llm: str | None = None,
    source: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: int = 600,
) -> list[dict]:
    """Stage 1: Run N Solvers sequentially, return list of solutions.

    Each Solver independently attempts to solve the problem using the
    CodeActAgent's generate↔execute loop. Diversity comes from LLM
    sampling randomness (temperature=0.7).

    Args:
        problem: The problem to solve
        num_solvers: Number of independent solvers (default: 5)
        temperature: LLM temperature for sampling diversity
        llm: LLM model name (None → env default)
        source: LLM provider (None → auto-detect)
        base_url: Custom API base URL
        api_key: API key
        timeout_seconds: Code execution timeout per block

    Returns:
        List of dicts, each with: solution, log, solver_id, success
    """
    # Per-solver temperature schedule for diversity
    _TEMP_SCHEDULE = [0.3, 0.5, 0.7, 0.9, 1.0]

    def _solver_temp(sid: int) -> float:
        if temperature != 0.7:  # user explicitly set
            return temperature
        if num_solvers == 1:
            return 0.7
        return _TEMP_SCHEDULE[sid % len(_TEMP_SCHEDULE)]

    temps = [_solver_temp(i) for i in range(num_solvers)]
    logger.info(f"=== X-Masters Stage 1: Running {num_solvers} Solvers (temps={temps}) ===")
    logger.info(f"Problem: {problem[:200]}...")

    results = []
    for i in range(num_solvers):
        result = run_single_solver(
            problem=problem,
            solver_id=i,
            temperature=temps[i],
            llm=llm,
            source=source,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        results.append(result)

    success_count = sum(1 for r in results if r["success"])
    logger.info(
        f"=== Stage 1 Complete: {success_count}/{num_solvers} solvers succeeded ==="
    )

    return results


# ---------------------------------------------------------------------------
# LangGraph node interface
# ---------------------------------------------------------------------------
def solve_node(state: dict) -> dict:
    """LangGraph node wrapper for Stage 1 Solver.

    Reads "problem" from state, runs N solvers, returns solutions
    in the format expected by the X-Masters StateGraph reducer:
        solutions: Annotated[list[str], operator.add]

    Args:
        state: XMastersState dict with "problem" key

    Returns:
        {"solutions": [solution_1, ..., solution_N]}
    """
    results = solve(problem=state["problem"])
    solutions = [r["solution"] for r in results]
    return {"solutions": solutions}
