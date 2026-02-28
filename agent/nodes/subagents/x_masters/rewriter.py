"""X-Masters Stage 3: Rewriter

Synthesizes all critiqued solutions into a new, improved answer.  Each Rewriter
instance is a CodeActAgent that sees ALL N solutions from Stage 2 (Critic) and
produces a single refined answer by critically comparing, verifying, and
rewriting.

Key insight from the official X-Master source code: the Rewriter's prompt and
IRG are nearly identical to the Critic's — the only difference is that the
Rewriter sees all N solutions instead of just one.  The model naturally
synthesizes when presented with multiple candidate answers.

Architecture (mirrors Critic):
    For each of N Rewriter instances:
        1. Clear execution namespace (isolation)
        2. Inject tools (web_search, biomedical tools)
        3. Create fresh RewriterAgent instance
        4. Run rewriter.go(problem + all_solutions)
        5. Extract <solution> from output
        6. Collect rewritten solution (or fallback on failure)

Data flow:
    Input  — from XMastersState after Stage 2 fan-in:
        problem          (str):        the original question
        all_solutions    (list[str]):  all N critiqued solutions
        rewriter_id      (int):        which Rewriter instance this is

    Output — written to XMastersState.rewritten_solutions via operator.add:
        rewriter_id (int):  which Rewriter produced this
        solution    (str):  the Rewriter's synthesized answer
        success     (bool): True if Rewriter ran without error
"""

import logging
import re

from .tools import inject_lightweight_tools_to_namespace

# ---------------------------------------------------------------------------
# Import CodeActAgent from result_evaluator
# ---------------------------------------------------------------------------
# Use absolute import to avoid conflict with top-level agent package
from agent.nodes.subagents.result_evaluator.agent import CodeActAgent
from agent.nodes.subagents.result_evaluator import executor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rewriter system prompt
# ---------------------------------------------------------------------------
REWRITER_SYSTEM_PROMPT = """\
You are a critical reviewer assigned with the task of synthesizing multiple proposed solutions into a single, improved answer.
You will be given a PROBLEM and MULTIPLE PROPOSED SOLUTIONS from different solvers. Your job is to critically check each solution, identify the best reasoning, resolve contradictions, and write your own definitive answer.

To achieve this, you will be using an interactive coding environment equipped with tool functions.

At each turn, you should first provide your critical analysis.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output. Your code should be enclosed using "<execute>" tag, for example:
<execute>
print("Hello World!")
</execute>

2) When you have reached your final verdict, provide the answer using "<solution>" tag, for example:
The answer is <solution> 42 </solution>.

## Available Tools

You have access to the following pre-loaded tool functions that you can call directly in your code:

### Search Tools
- **web_search(query, max_results=5)** — Search the web for current information and research findings
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

You have access to all standard Python libraries (numpy, scipy, sympy, pandas, etc.) and can write arbitrary code for computation, verification, and analysis.

## CRITICAL RULES

1. **EVERY response MUST contain either <execute> or <solution> tag.** No exceptions.
2. **When you have the final answer, you MUST use <solution> tag immediately.**
3. **Pure text responses without tags are NOT allowed** and will be rejected.
4. **When choosing between answer choices, prefer the most specific and mechanistically precise answer.** If one answer describes a specific mechanism (e.g., a named structural motif or pathway) and another describes a more general phenomenon that could encompass it, choose the specific one. A precise mechanistic explanation is always preferred over a vague or composite description.
5. **When you find a relevant primary research paper, read it carefully using read_webpage() and base your answer on the paper's actual conclusions**, not on textbook-level generalizations.
"""


# ---------------------------------------------------------------------------
# Initial Reasoning Guidance (IRG) for Rewriter
# Mirrors Critic IRG per official X-Master source (rewrite_prefix == critic_prefix)
# ---------------------------------------------------------------------------
REWRITER_IRG = """\
I need to carefully review all proposed solutions and synthesize the best answer. Let me think about what each solver got right and wrong.

I have access to an interactive coding environment where I can write and execute Python code. I should leverage this capability to independently verify the proposed answers:
- Query biomedical databases (query_kg, query_expression, query_disease_gene, query_gwas, query_ppi, etc.) to fact-check claims about genes, diseases, drugs, or molecular biology
- Search for relevant information using web_search() to verify broader facts
- **Read full paper content** using read_webpage(url) when web_search returns a relevant paper — search snippets are often insufficient for nuanced questions
- Write code to independently re-derive or re-compute any calculations in the proposed solutions
- Check if the proposed answers are consistent with known facts and constraints
- Look for edge cases or assumptions that the original solvers may have missed

IMPORTANT - Handling Unavailable External Resources:
- If web_search returns "[Search Unavailable]" or empty results, STOP trying to search
- If read_webpage fails repeatedly, STOP trying to read webpages
- When external resources are unavailable, use your internal knowledge and the provided context
- DO NOT keep trying the same failing tool - proceed to your best answer using available information

CRITICAL: I should not simply agree with the majority — my value is in finding the correct answer by verifying each claim. When all solvers agree, I must be especially skeptical: they may all share the same blind spot (e.g., relying on textbook-level generalizations when a specialized mechanism is at play). I MUST use read_webpage(url) to read the actual content of relevant papers rather than guessing from titles or snippets.

Let me start by comparing the solutions and identifying where they agree and disagree, then verify the disputed points.
"""


# ---------------------------------------------------------------------------
# RewriterAgent: CodeActAgent with Rewriter-specific system prompt + IRG
# ---------------------------------------------------------------------------
class RewriterAgent(CodeActAgent):
    """CodeActAgent specialized for X-Masters solution synthesis.

    Inherits all functionality from CodeActAgent but overrides:
    - System prompt: instructs the model to synthesize multiple solutions
    - go() method: injects Initial Reasoning Guidance (IRG) after user message

    The IRG mechanism primes the model to critically compare all solutions
    and verify disputed claims using code, rather than blindly picking one.
    """

    def _generate_system_prompt(self) -> str:
        """Generate Rewriter-specific system prompt."""
        return REWRITER_SYSTEM_PROMPT

    def go(self, prompt: str):
        """Execute the rewriter agent with IRG injection.

        Overrides CodeActAgent.go() to inject Initial Reasoning Guidance
        as an AIMessage after the user's problem + all proposed solutions.
        """
        from langchain_core.messages import HumanMessage, AIMessage

        self.user_task = prompt

        inputs = {
            "messages": [
                HumanMessage(content=prompt),
                AIMessage(content=REWRITER_IRG.strip()),
            ],
            "next_step": "generate",
        }
        # OPTIMIZATION: Reduced recursion limit to fail fast when external resources unavailable
        config = {"recursion_limit": 100, "configurable": {"thread_id": 42}}
        self.log = []
        
        # Track consecutive empty observations to detect stuck states
        empty_observation_count = 0
        MAX_EMPTY_OBSERVATIONS = 5

        final_state = None
        step = 0
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            step += 1
            
            # Detect empty observations (stuck state detection)
            if isinstance(message, AIMessage) and "<observation></observation>" in message.content:
                empty_observation_count += 1
                if empty_observation_count >= MAX_EMPTY_OBSERVATIONS:
                    print(f"  ⚠ Detected {empty_observation_count} consecutive empty observations")
                    print(f"  → External resources unavailable, forcing early termination")
            elif isinstance(message, AIMessage) and "<observation>" in message.content:
                if "</observation>" in message.content:
                    obs_content = message.content.split("<observation>")[1].split("</observation>")[0]
                    if obs_content.strip():
                        empty_observation_count = 0
            
            out = self._pretty_print(message)
            self.log.append(out)
            self._verbose_log(step, message)
            final_state = s

        self._conversation_state = final_state
        return self.log, message.content


# ---------------------------------------------------------------------------
# Helper functions (shared pattern with critic.py)
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


def _inject_rewriter_tools(namespace: dict = None):
    """Inject lightweight tools (no knowledge_search) into the execution namespace.

    Rewriters receive pre-retrieved context via their prompt, so they only need
    web_search and biomedical DB tools for real-time verification.

    Args:
        namespace: Target namespace dict. If None, uses global.
    """
    if namespace is None:
        namespace = executor._persistent_namespace
    inject_lightweight_tools_to_namespace(namespace)


def _build_rewriter_prompt(problem: str, all_solutions: list, retrieved_context: str = "") -> str:
    """Build the user message for the Rewriter agent.

    Presents the problem and all proposed solutions in a structured format
    matching the official X-Master rewrite_user.txt template.

    Args:
        problem: The original question/problem
        all_solutions: List of solution strings from Stage 2 (Critic)
        retrieved_context: Pre-retrieved knowledge base context

    Returns:
        Formatted prompt string
    """
    parts = [f"## Problem\n\n{problem}\n"]

    for i, sol in enumerate(all_solutions, 1):
        parts.append(f"## Student {i}'s Solution\n\n{sol}\n")

    if retrieved_context:
        parts.append(
            "## Reference Knowledge\n\n"
            "The following relevant information was pre-retrieved from the knowledge base. "
            "Use it as reference but verify important claims independently.\n\n"
            f"{retrieved_context}\n"
        )

    parts.append(
        "## Your Job\n\n"
        "You should critically check the students' solution to the problem, "
        "then correct it if needed and write your own answer.\n\n"
        "You should not be overconfident in your knowledge and reasoning. "
        "Use the available tools (web_search, biomedical databases) "
        "to verify claims before reaching your conclusion."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core rewriter logic
# ---------------------------------------------------------------------------
def run_single_rewriter(
    problem: str,
    all_solutions: list,
    rewriter_id: int = 0,
    retrieved_context: str = "",
    temperature: float = 0.7,
    llm: str = None,
    source: str = None,
    base_url: str = None,
    api_key: str = None,
    timeout_seconds: int = 120,
) -> dict:
    """Run a single Rewriter instance on all critiqued solutions.

    Creates a fresh RewriterAgent, feeds it the problem + all solutions,
    and extracts the synthesized answer.  On failure, falls back to the
    first solution so that the Rewriter stage never makes things worse.

    Args:
        problem: The original problem
        all_solutions: List of all critiqued solution strings from Stage 2
        rewriter_id: Which Rewriter instance this is (for tracking)
        retrieved_context: Pre-retrieved knowledge base context (from Stage 0)
        temperature: LLM sampling temperature
        llm: LLM model name (None -> env default)
        source: LLM provider (None -> auto-detect)
        base_url: Custom API base URL
        api_key: API key
        timeout_seconds: Code execution timeout per block

    Returns:
        dict with keys:
            - solution (str): The Rewriter's synthesized answer
            - rewriter_id (int): Which Rewriter produced this
            - success (bool): Whether the Rewriter completed without error
            - log (list): Execution log
    """
    logger.info(f"[Rewriter {rewriter_id}] Starting synthesis...")

    # 1. Create fresh RewriterAgent instance (owns its own _namespace)
    rewriter = RewriterAgent(
        llm=llm,
        source=source,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        verbose=True,
        agent_label=f"Rewriter {rewriter_id}",
    )

    # 2. Inject lightweight tools (no knowledge_search) into namespace
    _inject_rewriter_tools(rewriter._namespace)

    # 3. Build the rewriter prompt with pre-retrieved context and run
    logger.info(f"[Rewriter {rewriter_id}] received retrieved_context: {len(retrieved_context)} chars")
    rewriter_prompt = _build_rewriter_prompt(problem, all_solutions, retrieved_context)

    # Fallback: use the first solution if rewriter fails
    fallback = all_solutions[0] if all_solutions else ""

    error_info = None
    try:
        log, last_content = rewriter.go(rewriter_prompt)
        rewritten_solution = extract_solution(last_content)
        success = True
        logger.info(f"[Rewriter {rewriter_id}] Completed successfully")
    except Exception as e:
        import traceback
        error_info = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"[Rewriter {rewriter_id}] Failed: {e}")
        logger.debug(f"[Rewriter {rewriter_id}] Traceback: {error_traceback}")
        rewritten_solution = fallback
        log = []
        success = False

    # 4. Clean up namespace after this rewriter (defensive)
    _reset_execution_env(rewriter._namespace, rewriter._captured_plots)

    return {
        "solution": rewritten_solution,
        "rewriter_id": rewriter_id,
        "log": log,
        "success": success,
        "error": error_info,  # Include error information for debugging
    }
