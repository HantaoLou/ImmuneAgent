"""X-Masters Stage 2: Critic

Reviews each Solver's solution independently, identifies flaws, and produces
a corrected (or confirmed) answer.  Each Critic instance is a CodeActAgent
with an adversarial system prompt that forces it to *first* look for errors
before deciding whether to amend or confirm the original solution.

Architecture (instance-level isolation, mirrors Solver):
    For each of N solutions from Stage 1:
        1. Create fresh CriticAgent instance (owns its own _namespace)
        2. Inject tools into the agent's instance-level namespace
        3. Run critic.go(problem + solution)
        4. Extract <solution> from output
        5. Collect critiqued solution (or fallback to original on failure)

    Each CodeActAgent holds its own _namespace dict and _captured_plots list,
    so parallel Send instances (LangGraph ThreadPoolExecutor) are thread-safe.

Data flow:
    Input  — from XMastersState after Stage 1 fan-in:
        problem  (str):   the original question
        solution (str):   one Solver's extracted answer
        solver_id (int):  which Solver produced it
        success  (bool):  whether the Solver succeeded

    Output — written to XMastersState.critiqued_solutions via operator.add:
        solver_id (int):  preserved from input
        solution  (str):  the Critic's (possibly amended) answer
        success   (bool): True if Critic ran without error
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
# Critic system prompt
# ---------------------------------------------------------------------------
CRITIC_SYSTEM_PROMPT = """\
You are a critical reviewer assigned with the task of verifying and improving a proposed solution.
You will be given a PROBLEM and a PROPOSED SOLUTION. Your job is to check the solution for errors and produce a corrected or confirmed answer.

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
# Initial Reasoning Guidance (IRG) for Critic — X-Master paper Section 2.3
# ---------------------------------------------------------------------------
CRITIC_IRG = """\
I need to carefully verify this proposed solution. Let me think about what could be wrong.

I have access to an interactive coding environment where I can write and execute Python code. I should leverage this capability to independently verify the proposed answer:
- Query biomedical databases (query_kg, query_expression, query_disease_gene, query_gwas, query_ppi, etc.) to fact-check claims about genes, diseases, drugs, or molecular biology
- Search for relevant information using web_search() to verify broader facts
- **Read full paper content** using read_webpage(url) when web_search returns a relevant paper — search snippets are often insufficient for nuanced questions
- Write code to independently re-derive or re-compute any calculations in the proposed solution
- Check if the proposed answer is consistent with known facts and constraints
- Look for edge cases or assumptions that the original solver may have missed

CRITICAL: I should not simply agree with the proposed solution — my value is in finding errors that others missed. In particular, I must challenge any reasoning that relies on textbook-level generalizations. Real-world biology and chemistry often involve specialized mechanisms that deviate from simple models. When I find a relevant paper, I MUST use read_webpage(url) to read its actual findings rather than assuming from the title alone.

I should specifically check whether the solver may have made an oversimplified peak/signal assignment, missed a specialized structural mechanism, or conflated two different phenomena.

Let me start by identifying the key claims in this solution and writing code to check them.
"""


# ---------------------------------------------------------------------------
# CriticAgent: CodeActAgent with Critic-specific system prompt + IRG
# ---------------------------------------------------------------------------
class CriticAgent(CodeActAgent):
    """CodeActAgent specialized for X-Masters critical review.

    Inherits all functionality from CodeActAgent but overrides:
    - System prompt: concise capability description + format rules
    - go() method: injects Initial Reasoning Guidance (IRG) after user message

    The IRG mechanism primes the model to independently verify solutions
    using code, rather than blindly agreeing or relying on internal knowledge.
    """

    def _generate_system_prompt(self) -> str:
        """Generate Critic-specific system prompt."""
        return CRITIC_SYSTEM_PROMPT

    def go(self, prompt: str):
        """Execute the critic agent with IRG injection.

        Overrides CodeActAgent.go() to inject Initial Reasoning Guidance
        as an AIMessage after the user's problem + proposed solution.
        """
        from langchain_core.messages import HumanMessage, AIMessage

        self.user_task = prompt

        inputs = {
            "messages": [
                HumanMessage(content=prompt),
                AIMessage(content=CRITIC_IRG.strip()),
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
# Helper functions (shared pattern with solver.py)
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


def _inject_critic_tools(namespace: dict = None):
    """Inject lightweight tools (no knowledge_search) into the execution namespace.

    Critics receive pre-retrieved context via their prompt, so they only need
    web_search and biomedical DB tools for real-time verification.

    Args:
        namespace: Target namespace dict. If None, uses global.
    """
    if namespace is None:
        namespace = executor._persistent_namespace
    inject_lightweight_tools_to_namespace(namespace)


def _build_critic_prompt(problem: str, solution: str, retrieved_context: str = "") -> str:
    """Build the user message for the Critic agent.

    Presents the problem and proposed solution in a structured format
    that the Critic's system prompt expects.

    Args:
        problem: The original question/problem
        solution: The Solver's proposed answer
        retrieved_context: Pre-retrieved knowledge base context

    Returns:
        Formatted prompt string
    """
    context_section = ""
    if retrieved_context:
        context_section = (
            f"\n## Reference Knowledge\n\n"
            f"The following relevant information was pre-retrieved from the knowledge base. "
            f"Use it as reference but verify important claims independently.\n\n"
            f"{retrieved_context}\n"
        )
    return f"""## Problem

{problem}

## Proposed Solution

{solution}
{context_section}
---

Please critically review the proposed solution above. Verify key claims using available tools. If you find errors, provide a corrected answer. If the solution is correct, confirm it."""


# ---------------------------------------------------------------------------
# Core critic logic
# ---------------------------------------------------------------------------
def run_single_critic(
    problem: str,
    solution: str,
    solver_id: int = 0,
    retrieved_context: str = "",
    temperature: float = 0.6,
    llm: str = None,
    source: str = None,
    base_url: str = None,
    api_key: str = None,
    timeout_seconds: int = 120,
) -> dict:
    """Run a single Critic instance on one Solver's solution.

    Creates a fresh CriticAgent, feeds it the problem + proposed solution,
    and extracts the (possibly amended) answer.  On failure, falls back to
    the original solution so that the Critic stage never makes things worse.

    Args:
        problem: The original problem
        solution: The Solver's proposed answer to review
        solver_id: Which Solver produced this solution (for tracking)
        retrieved_context: Pre-retrieved knowledge base context (from Stage 0)
        temperature: LLM sampling temperature
        llm: LLM model name (None → env default)
        source: LLM provider (None → auto-detect)
        base_url: Custom API base URL
        api_key: API key
        timeout_seconds: Code execution timeout per block

    Returns:
        dict with keys:
            - solution (str): The Critic's (possibly amended) answer
            - original_solution (str): The Solver's original answer
            - solver_id (int): Which Solver produced the original
            - success (bool): Whether the Critic completed without error
    """
    logger.info(f"[Critic {solver_id}] Starting review...")

    # 1. Create fresh CriticAgent instance (owns its own _namespace)
    critic = CriticAgent(
        llm=llm,
        source=source,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        verbose=True,
        agent_label=f"Critic {solver_id}",
    )

    # 2. Inject lightweight tools (no knowledge_search) into namespace
    _inject_critic_tools(critic._namespace)

    # 3. Build the critic prompt with pre-retrieved context and run
    logger.info(f"[Critic {solver_id}] received retrieved_context: {len(retrieved_context)} chars")
    critic_prompt = _build_critic_prompt(problem, solution, retrieved_context)

    try:
        log, last_content = critic.go(critic_prompt)
        revised_solution = extract_solution(last_content)
        success = True
        logger.info(f"[Critic {solver_id}] Completed successfully")
    except Exception as e:
        logger.error(f"[Critic {solver_id}] Failed: {e}")
        # Fallback: preserve the original solution
        revised_solution = solution
        log = []
        success = False

    # 4. Clean up namespace after this critic (defensive)
    _reset_execution_env(critic._namespace, critic._captured_plots)

    return {
        "solution": revised_solution,
        "original_solution": solution,
        "log": log,
        "solver_id": solver_id,
        "success": success,
    }
